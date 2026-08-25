"""Persistent post-download filtering for paper corpora.

Regex and stored-topic filters are evaluated independently for every paper and
then combined in application order. The generic SQLite tables deliberately
keep definitions and paper-level decisions separate from scrape statuses.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar

import regex

from paperminer.corpus.database import connect, get_asset, paper_rows, utc_now
from paperminer.corpus.documents import read_pdf_bytes, trim_reference_section


FILTER_FIELDS = {'title', 'abstract', 'full_text'}
FILTER_STATUSES = {'included', 'excluded', 'unavailable'}
JOIN_OPERATORS = {'and', 'or'}
DEFAULT_TIMEOUT_MS = 500
MAX_MATCHES_PER_FIELD = 1000
MAX_SNIPPETS_PER_FIELD = 3
SNIPPET_CONTEXT = 80
T = TypeVar('T')
_FilterStatus: TypeAlias = Literal['included', 'excluded', 'unavailable']
_RegexRule: TypeAlias = dict[str, str]
_RegexDefinition: TypeAlias = dict[str, Any]
_CompiledDefinition: TypeAlias = dict[str, list[tuple[_RegexRule, regex.Pattern]]]
_FilterOverview: TypeAlias = dict[str, Any]


def _require_type(value: object, expected_type: type[T], label: str) -> T:
    """Validate a value's type.

    Parameters
    ----------
    value : object
        Value to validate.
    expected_type : type[T]
        Required Python type.
    label : str
        Human-readable field label for errors.

    Returns
    -------
    T
        The original validated value.

    Raises
    ------
    ValueError
        If ``value`` is not an instance of ``expected_type``.
    """
    if not isinstance(value, expected_type):
        raise ValueError(f'{label} must be {expected_type.__name__}.')
    return value


def _normalize_rule(rule: object, label: str) -> _RegexRule:
    """Normalize one regular-expression rule.

    Parameters
    ----------
    rule : object
        Rule containing a name and pattern.
    label : str
        Rule location used in validation errors.

    Returns
    -------
    _RegexRule
        Normalized rule name and pattern.

    Raises
    ------
    ValueError
        If the rule is malformed or empty.
    """
    _require_type(rule, dict, label)
    name = str(rule.get('name') or '').strip()
    pattern = rule.get('pattern')
    if not name:
        raise ValueError(f'{label} requires a non-empty name.')
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f'{label} requires a non-empty string pattern.')
    return {'name': name, 'pattern': pattern}


def normalize_regex_definition(
    definition: object,
    fields: Iterable[str] | None = None,
    timeout_ms: int | None = None,
) -> _RegexDefinition:
    """Normalize a regular-expression filter definition.

    Parameters
    ----------
    definition : object
        Raw filter definition.
    fields : Iterable[str] or None, optional
        Content fields overriding those in the definition.
    timeout_ms : int, optional
        Per-field matching timeout overriding the definition.

    Returns
    -------
    _RegexDefinition
        Validated deterministic filter definition.

    Raises
    ------
    ValueError
        If names, rules, fields, or options are invalid.
    """
    _require_type(definition, dict, 'Filter definition')
    name = str(definition.get('name') or '').strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
        raise ValueError('Filter name must use only letters, numbers, dots, underscores, and hyphens.')

    include_mode = str(definition.get('include_mode', 'any')).lower()
    if include_mode not in {'any', 'all'}:
        raise ValueError('include_mode must be "any" or "all".')
    include_rules = definition.get('include', [])
    exclude_rules = definition.get('exclude', [])
    if not isinstance(include_rules, list) or not isinstance(exclude_rules, list):
        raise ValueError('include and exclude must be JSON lists.')
    include = [
        _normalize_rule(rule, f'include[{index}]')
        for index, rule in enumerate(include_rules)
    ]
    if not include:
        raise ValueError('A regex filter requires at least one include rule.')
    exclude = [
        _normalize_rule(rule, f'exclude[{index}]')
        for index, rule in enumerate(exclude_rules)
    ]
    rule_names = [rule['name'] for rule in include + exclude]
    if len(rule_names) != len(set(rule_names)):
        raise ValueError('Regex rule names must be unique within a filter.')

    resolved_fields = list(fields) if fields else definition.get(
        'fields', ['title', 'abstract', 'full_text']
    )
    if not isinstance(resolved_fields, (list, tuple)) or not resolved_fields:
        raise ValueError('fields must be a non-empty list.')
    resolved_fields = list(dict.fromkeys(str(field).lower() for field in resolved_fields))
    unknown_fields = set(resolved_fields) - FILTER_FIELDS
    if unknown_fields:
        raise ValueError(f'Unknown filter fields: {", ".join(sorted(unknown_fields))}.')

    resolved_timeout = timeout_ms if timeout_ms is not None else definition.get(
        'timeout_ms', DEFAULT_TIMEOUT_MS
    )
    if isinstance(resolved_timeout, bool) or not isinstance(resolved_timeout, int) or resolved_timeout < 1:
        raise ValueError('timeout_ms must be a positive integer.')
    case_sensitive = definition.get('case_sensitive', False)
    if not isinstance(case_sensitive, bool):
        raise ValueError('case_sensitive must be true or false.')
    description = definition.get('description', '')
    if not isinstance(description, str):
        raise ValueError('description must be a string.')

    return {
        'name': name,
        'description': description.strip(),
        'fields': resolved_fields,
        'case_sensitive': case_sensitive,
        'include_mode': include_mode,
        'include': include,
        'exclude': exclude,
        'timeout_ms': resolved_timeout,
    }


def load_regex_definition(
    path: str | PathLike[str],
    fields: Iterable[str] | None = None,
    timeout_ms: int | None = None,
) -> _RegexDefinition:
    """Load and normalize a JSON filter definition.

    Parameters
    ----------
    path : str or os.PathLike[str]
        JSON definition file.
    fields : Iterable[str] or None, optional
        Content fields overriding the file.
    timeout_ms : int, optional
        Matching timeout overriding the file.

    Returns
    -------
    _RegexDefinition
        Validated filter definition.

    Raises
    ------
    OSError
        If the definition file cannot be read.
    ValueError
        If the JSON or filter definition is invalid.
    """
    try:
        definition = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'Invalid filter JSON: {error}') from error
    return normalize_regex_definition(definition, fields=fields, timeout_ms=timeout_ms)


def _normalize_topic_rule(
    rule: object,
    label: str,
    valid_topic_ids: set[int],
) -> dict[str, Any]:
    """Normalize one probability or dominance topic condition.

    Parameters
    ----------
    rule : object
        Candidate mapping containing a rule name, topic ID, and threshold or
        dominance requirement.
    label : str
        Rule location used in validation errors.
    valid_topic_ids : set[int]
        Topic IDs defined by the referenced stored model.

    Returns
    -------
    dict[str, Any]
        Validated topic rule with explicit optional values.

    Raises
    ------
    ValueError
        If the rule is malformed, refers to an unknown topic, or contains no
        effective condition.
    """
    _require_type(rule, dict, label)
    name = str(rule.get('name') or '').strip()
    if not name:
        raise ValueError(f'{label} requires a non-empty name.')
    topic_id = rule.get('topic_id')
    if isinstance(topic_id, bool) or not isinstance(topic_id, int):
        raise ValueError(f'{label}.topic_id must be an integer.')
    if topic_id not in valid_topic_ids:
        raise ValueError(f'{label} refers to unknown topic {topic_id}.')
    minimum = rule.get('min_probability')
    if minimum is not None:
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 <= minimum <= 1:
            raise ValueError(f'{label}.min_probability must be between 0 and 1.')
        minimum = float(minimum)
    require_dominant = rule.get('require_dominant', False)
    if not isinstance(require_dominant, bool):
        raise ValueError(f'{label}.require_dominant must be true or false.')
    if minimum is None and not require_dominant:
        raise ValueError(
            f'{label} requires min_probability and/or require_dominant=true.'
        )
    return {
        'name': name,
        'topic_id': topic_id,
        'min_probability': minimum,
        'require_dominant': require_dominant,
    }


def normalize_topic_definition(
    conn: sqlite3.Connection,
    definition: object,
) -> dict[str, Any]:
    """Resolve and normalize a persistent topic-filter definition.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection containing stored topic models.
    definition : object
        Candidate topic-filter definition loaded from JSON.

    Returns
    -------
    dict[str, Any]
        Validated definition with the model name and immutable model ID.

    Raises
    ------
    ValueError
        If the definition is malformed, its model is unavailable, or any rule
        refers to an unknown topic.
    """
    _require_type(definition, dict, 'Filter definition')
    name = str(definition.get('name') or '').strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
        raise ValueError('Filter name must use only letters, numbers, dots, underscores, and hyphens.')
    model_reference = str(definition.get('model') or '').strip()
    if not model_reference:
        raise ValueError('A topic filter requires a stored model name or model ID.')
    model = conn.execute(
        'SELECT * FROM topic_models WHERE name = ? OR model_id = ?',
        (model_reference, model_reference),
    ).fetchone()
    if model is None:
        raise ValueError(f'No stored topic model matches {model_reference!r}.')
    valid_topic_ids = {
        row['topic_id']
        for row in conn.execute(
            'SELECT topic_id FROM topic_definitions WHERE model_id = ?',
            (model['model_id'],),
        ).fetchall()
    }
    include_mode = str(definition.get('include_mode', 'any')).lower()
    if include_mode not in {'any', 'all'}:
        raise ValueError('include_mode must be "any" or "all".')
    include_values = definition.get('include', [])
    exclude_values = definition.get('exclude', [])
    if not isinstance(include_values, list) or not isinstance(exclude_values, list):
        raise ValueError('include and exclude must be JSON lists.')
    include = [
        _normalize_topic_rule(rule, f'include[{index}]', valid_topic_ids)
        for index, rule in enumerate(include_values)
    ]
    if not include:
        raise ValueError('A topic filter requires at least one include rule.')
    exclude = [
        _normalize_topic_rule(rule, f'exclude[{index}]', valid_topic_ids)
        for index, rule in enumerate(exclude_values)
    ]
    rule_names = [rule['name'] for rule in include + exclude]
    if len(rule_names) != len(set(rule_names)):
        raise ValueError('Topic rule names must be unique within a filter.')
    description = definition.get('description', '')
    if not isinstance(description, str):
        raise ValueError('description must be a string.')
    return {
        'name': name,
        'description': description.strip(),
        'model': model['name'],
        'model_id': model['model_id'],
        'include_mode': include_mode,
        'include': include,
        'exclude': exclude,
    }


def load_topic_definition(
    conn: sqlite3.Connection,
    path: str | PathLike[str],
) -> dict[str, Any]:
    """Load and normalize a topic-filter JSON document.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection used to resolve the stored model.
    path : str or os.PathLike[str]
        Topic-filter JSON file.

    Returns
    -------
    dict[str, Any]
        Validated persistent topic-filter definition.

    Raises
    ------
    OSError
        If the definition file cannot be read.
    ValueError
        If the JSON or topic-filter definition is invalid.
    """
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'Invalid filter JSON: {error}') from error
    return normalize_topic_definition(conn, value)


def compile_regex_definition(definition: _RegexDefinition) -> _CompiledDefinition:
    """Compile every pattern in a filter definition.

    Parameters
    ----------
    definition : _RegexDefinition
        Normalized regular-expression filter definition.

    Returns
    -------
    _CompiledDefinition
        Include and exclude rules paired with compiled expressions.

    Raises
    ------
    ValueError
        If any regular expression is invalid.
    """
    flags = regex.VERSION1
    if not definition['case_sensitive']:
        flags |= regex.IGNORECASE
    compiled = {'include': [], 'exclude': []}
    for rule_type in compiled:
        for rule in definition[rule_type]:
            try:
                expression = regex.compile(rule['pattern'], flags)
            except regex.error as error:
                raise ValueError(f'Invalid regex in rule {rule["name"]!r}: {error}') from error
            compiled[rule_type].append((rule, expression))
    return compiled


def _decode_text_asset(asset: Mapping[str, Any] | None) -> str | None:
    """Decode and trim a stored text asset.

    Parameters
    ----------
    asset : Mapping[str, Any] or None
        Corpus asset row containing binary content.

    Returns
    -------
    str or None
        Usable text without references, or ``None``.
    """
    if asset is None:
        return None
    text = asset['content'].decode('utf-8', errors='replace')
    text = trim_reference_section(text)
    return text if text.strip() else None


def _paper_content(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    fields: Iterable[str],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Load the selected raw content fields for a paper.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : Mapping[str, Any]
        Corpus paper row.
    fields : Iterable[str]
        Content fields to load.

    Returns
    -------
    tuple[dict[str, str], dict[str, str], list[str]]
        Available content, the source of each field, and reasons that requested
        fields were unavailable.
    """
    content = {}
    sources = {}
    unavailable = []
    paper_id = paper['paper_id']

    if 'title' in fields:
        title = str(paper.get('title') or '').strip()
        if title:
            content['title'] = title
            sources['title'] = 'metadata:title'
        else:
            unavailable.append('title: missing')

    if 'abstract' in fields:
        try:
            abstract = _decode_text_asset(get_asset(conn, paper_id, 'abstract'))
        except Exception as error:
            abstract = None
            unavailable.append(f'abstract: unreadable ({error})')
        if abstract:
            content['abstract'] = abstract
            sources['abstract'] = 'corpus:abstract'
        elif not any(reason.startswith('abstract:') for reason in unavailable):
            unavailable.append('abstract: missing')

    if 'full_text' in fields:
        full_text = None
        text_error = None
        pdf_error = None
        try:
            full_text = _decode_text_asset(get_asset(conn, paper_id, 'text'))
        except Exception as error:
            text_error = error
        if full_text:
            sources['full_text'] = 'corpus:text'
        else:
            try:
                pdf_asset = get_asset(conn, paper_id, 'pdf')
            except Exception as error:
                pdf_asset = None
                pdf_error = error
            if pdf_asset is not None:
                try:
                    full_text = trim_reference_section(read_pdf_bytes(pdf_asset['content']))
                    if not full_text.strip():
                        full_text = None
                    else:
                        sources['full_text'] = 'corpus:pdf'
                except Exception as error:
                    pdf_error = error
        if full_text:
            content['full_text'] = full_text
        elif text_error or pdf_error:
            errors = '; '.join(str(error) for error in [text_error, pdf_error] if error)
            unavailable.append(f'full_text: unreadable ({errors})')
        else:
            unavailable.append('full_text: missing')
    return content, sources, unavailable


def _snippet(text: str, start: int, end: int) -> str:
    """Return normalized context surrounding a matched text span.

    Parameters
    ----------
    text : str
        Complete text containing the match.
    start : int
        Inclusive match start offset.
    end : int
        Exclusive match end offset.

    Returns
    -------
    str
        Whitespace-normalized context around the match.
    """
    snippet = text[max(0, start - SNIPPET_CONTEXT):min(len(text), end + SNIPPET_CONTEXT)]
    return re.sub(r'\s+', ' ', snippet).strip()


def _match_rule(
    expression: regex.Pattern,
    content: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect bounded matches for one expression across content fields.

    Parameters
    ----------
    expression : regex.Pattern
        Compiled regular expression to evaluate.
    content : Mapping[str, str]
        Content keyed by field name.
    timeout_seconds : float
        Maximum matching time for each field, in seconds.

    Returns
    -------
    tuple[list[dict[str, Any]], list[str]]
        Match evidence and fields whose evaluation timed out.
    """
    matches = []
    timed_out = []
    for field, text in content.items():
        count = 0
        snippets = []
        truncated = False
        try:
            for match in expression.finditer(text, timeout=timeout_seconds):
                count += 1
                if len(snippets) < MAX_SNIPPETS_PER_FIELD:
                    snippets.append(_snippet(text, match.start(), match.end()))
                if count >= MAX_MATCHES_PER_FIELD:
                    truncated = True
                    break
        except TimeoutError:
            timed_out.append(field)
        if count:
            matches.append({
                'field': field,
                'count': count,
                'count_truncated': truncated,
                'snippets': snippets,
            })
    return matches, timed_out


def evaluate_regex_paper(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    definition: _RegexDefinition,
    compiled: _CompiledDefinition,
) -> tuple[_FilterStatus, dict[str, Any], str]:
    """Evaluate one regex definition against one corpus paper.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : Mapping[str, Any]
        Corpus paper row to evaluate.
    definition : _RegexDefinition
        Normalized regex filter definition.
    compiled : _CompiledDefinition
        Compiled include and exclude rules.

    Returns
    -------
    tuple[_FilterStatus, dict[str, Any], str]
        Filter status, structured match evidence, and a joined unavailable
        reason.
    """
    content, sources, unavailable = _paper_content(conn, paper, definition['fields'])
    timeout_seconds = definition['timeout_ms'] / 1000
    evidence_matches = []
    timed_out = []
    matched_rules = {'include': set(), 'exclude': set()}

    for rule_type in ['include', 'exclude']:
        for rule, expression in compiled[rule_type]:
            matches, timeout_fields = _match_rule(expression, content, timeout_seconds)
            if matches:
                matched_rules[rule_type].add(rule['name'])
                evidence_matches.append({
                    'type': rule_type,
                    'name': rule['name'],
                    'pattern': rule['pattern'],
                    'matches': matches,
                })
            timed_out.extend(
                f'{rule_type}:{rule["name"]}:{field}' for field in timeout_fields
            )

    if matched_rules['exclude']:
        status = 'excluded'
    else:
        include_names = {rule['name'] for rule in definition['include']}
        if definition['include_mode'] == 'any':
            include_satisfied = bool(matched_rules['include'])
        else:
            include_satisfied = matched_rules['include'] == include_names
        if include_satisfied:
            status = 'included'
        elif unavailable or timed_out:
            status = 'unavailable'
        else:
            status = 'excluded'

    unavailable_reasons = unavailable + [f'regex timeout: {item}' for item in timed_out]
    evidence = {
        'available_fields': list(content),
        'sources': sources,
        'matched_include_rules': sorted(matched_rules['include']),
        'matched_exclude_rules': sorted(matched_rules['exclude']),
        'matches': evidence_matches,
        'unavailable': unavailable_reasons,
    }
    return status, evidence, '; '.join(unavailable_reasons)


def _topic_rule_matches(
    rule: Mapping[str, Any],
    probability: float,
    dominant_topic: int | None,
) -> bool:
    """Evaluate one normalized condition against a paper's topic result.

    Parameters
    ----------
    rule : Mapping[str, Any]
        Normalized probability or dominance condition.
    probability : float
        Paper probability for the rule's topic.
    dominant_topic : int or None
        Paper's dominant topic ID when prediction succeeded.

    Returns
    -------
    bool
        ``True`` when every configured condition is satisfied.
    """
    if rule['min_probability'] is not None and probability < rule['min_probability']:
        return False
    if rule['require_dominant'] and dominant_topic != rule['topic_id']:
        return False
    return True


def evaluate_topic_paper(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> tuple[_FilterStatus, dict[str, Any], str]:
    """Evaluate a stored topic-filter definition for one corpus paper.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection containing predictions and scores.
    paper : Mapping[str, Any]
        Corpus paper row to classify.
    definition : Mapping[str, Any]
        Normalized topic-filter definition.

    Returns
    -------
    tuple[_FilterStatus, dict[str, Any], str]
        Filter status, structured rule evidence, and an unavailable reason.
    """
    prediction = conn.execute(
        'SELECT status, dominant_topic_id FROM paper_topic_predictions '
        'WHERE model_id = ? AND paper_id = ?',
        (definition['model_id'], paper['paper_id']),
    ).fetchone()
    if prediction is None:
        reason = 'topic prediction missing'
        return 'unavailable', {'model_id': definition['model_id'], 'unavailable': [reason]}, reason
    if prediction['status'] != 'predicted':
        reason = prediction['status']
        return 'unavailable', {'model_id': definition['model_id'], 'unavailable': [reason]}, reason
    scores = {
        row['topic_id']: row['probability']
        for row in conn.execute(
            'SELECT topic_id, probability FROM paper_topic_scores '
            'WHERE model_id = ? AND paper_id = ?',
            (definition['model_id'], paper['paper_id']),
        ).fetchall()
    }
    required_ids = {rule['topic_id'] for rule in definition['include'] + definition['exclude']}
    if not required_ids <= scores.keys():
        reason = 'topic scores incomplete'
        return 'unavailable', {'model_id': definition['model_id'], 'unavailable': [reason]}, reason
    topic_names = {
        row['topic_id']: row['topic_name']
        for row in conn.execute(
            'SELECT topic_id, topic_name FROM topic_definitions WHERE model_id = ?',
            (definition['model_id'],),
        ).fetchall()
    }
    matched = {'include': [], 'exclude': []}
    evidence_rules = []
    for rule_type in ['include', 'exclude']:
        for rule in definition[rule_type]:
            probability = scores[rule['topic_id']]
            did_match = _topic_rule_matches(
                rule, probability, prediction['dominant_topic_id']
            )
            if did_match:
                matched[rule_type].append(rule['name'])
            evidence_rules.append({
                **rule,
                'type': rule_type,
                'topic_name': topic_names.get(rule['topic_id'], ''),
                'probability': probability,
                'is_dominant': prediction['dominant_topic_id'] == rule['topic_id'],
                'matched': did_match,
            })
    if matched['exclude']:
        status = 'excluded'
    elif definition['include_mode'] == 'any':
        status = 'included' if matched['include'] else 'excluded'
    else:
        status = 'included' if len(matched['include']) == len(definition['include']) else 'excluded'
    evidence = {
        'model': definition['model'],
        'model_id': definition['model_id'],
        'dominant_topic_id': prediction['dominant_topic_id'],
        'matched_include_rules': matched['include'],
        'matched_exclude_rules': matched['exclude'],
        'rules': evidence_rules,
        'unavailable': [],
    }
    return status, evidence, ''


def _database_path(conn: sqlite3.Connection) -> str:
    """Return the on-disk path for a corpus connection.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection.

    Returns
    -------
    str
        Main database path, or an empty string for an in-memory connection.
    """
    rows = conn.execute('PRAGMA database_list').fetchall()
    return next((row['file'] for row in rows if row['name'] == 'main'), '')


def topic_filter_staleness(conn: sqlite3.Connection) -> dict[str, bool]:
    """Check whether active topic filters use current corpus predictions.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.

    Returns
    -------
    dict[str, bool]
        Stale state keyed by each model ID used by an active topic filter.
    """
    filters = [item for item in active_filter_stack(conn) if item['method'] == 'topic']
    if not filters:
        return {}
    from paperminer.workflows.topics import topic_corpus_fingerprint

    db_path = _database_path(conn)
    results = {}
    for item in filters:
        model_id = item['definition']['model_id']
        if model_id in results:
            continue
        model = conn.execute(
            'SELECT prediction_corpus_fingerprint, text_fields_json '
            'FROM topic_models WHERE model_id = ?', (model_id,)
        ).fetchone()
        if model is None or not db_path:
            results[model_id] = True
            continue
        current = topic_corpus_fingerprint(
            db_path, json.loads(model['text_fields_json'])
        )['sha256']
        results[model_id] = current != model['prediction_corpus_fingerprint']
    return results


def active_filter_stack(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return active filters in expression order.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.

    Returns
    -------
    list[dict[str, Any]]
        Filter rows with decoded definitions.

    Raises
    ------
    json.JSONDecodeError
        If a stored filter definition is invalid JSON.
    """
    rows = conn.execute(
        'SELECT * FROM corpus_filters ORDER BY stack_position, filter_id'
    ).fetchall()
    filters = []
    for row in rows:
        item = dict(row)
        item['definition'] = json.loads(item.pop('definition_json'))
        filters.append(item)
    return filters


def filter_expression(filters: Sequence[Mapping[str, Any]]) -> str:
    """Build the explicit expression for an active filter stack.

    Parameters
    ----------
    filters : Sequence[Mapping[str, Any]]
        Active filters in evaluation order.

    Returns
    -------
    str
        Parenthesized left-to-right expression, or an empty string.
    """
    if not filters:
        return ''
    expression = filters[0]['name']
    for item in filters[1:]:
        expression = f'({expression} {item["join_operator"].upper()} {item["name"]})'
    return expression


def combine_status(left: str, right: str, operator: str) -> _FilterStatus:
    """Combine two filter decisions with three-valued Boolean logic.

    Parameters
    ----------
    left : {'excluded', 'included', 'unavailable'}
        Left-hand filter decision.
    right : {'excluded', 'included', 'unavailable'}
        Right-hand filter decision.
    operator : {'and', 'or'}
        Boolean operator joining the decisions.

    Returns
    -------
    str
        Combined filter status.

    Raises
    ------
    ValueError
        If either status or the operator is unsupported.
    """
    if left not in FILTER_STATUSES or right not in FILTER_STATUSES:
        raise ValueError('Unknown filter status.')
    if operator == 'and':
        if 'excluded' in {left, right}:
            return 'excluded'
        if left == right == 'included':
            return 'included'
        return 'unavailable'
    if operator == 'or':
        if 'included' in {left, right}:
            return 'included'
        if left == right == 'excluded':
            return 'excluded'
        return 'unavailable'
    raise ValueError(f'Unknown join operator: {operator}')


def _recompute_filter_state(conn: sqlite3.Connection) -> None:
    """Rebuild final per-paper decisions from the active filter stack.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection whose filter state is updated.
    """
    filters = active_filter_stack(conn)
    conn.execute('DELETE FROM paper_filter_state')
    if not filters:
        return
    expression = filter_expression(filters)
    result_rows = conn.execute(
        'SELECT filter_id, paper_id, status FROM paper_filter_results'
    ).fetchall()
    results = {(row['filter_id'], row['paper_id']): row['status'] for row in result_rows}
    now = utc_now()
    states = []
    for paper in paper_rows(conn):
        paper_id = paper['paper_id']
        status = results.get((filters[0]['filter_id'], paper_id), 'unavailable')
        for item in filters[1:]:
            right = results.get((item['filter_id'], paper_id), 'unavailable')
            status = combine_status(status, right, item['join_operator'])
        states.append((paper_id, status, expression, now))
    conn.executemany(
        'INSERT INTO paper_filter_state (paper_id, status, expression, updated_at) VALUES (?, ?, ?, ?)',
        states,
    )


def filter_overview(conn: sqlite3.Connection) -> _FilterOverview:
    """Summarize the active filter stack and its decisions.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.

    Returns
    -------
    _FilterOverview
        Active filters, their expression, final status counts, and unavailable
        reason counts.
    """
    filters = active_filter_stack(conn)
    stale_models = topic_filter_staleness(conn)
    paper_count = conn.execute('SELECT COUNT(*) FROM papers').fetchone()[0]
    for item in filters:
        item['stale'] = (
            stale_models.get(item['definition'].get('model_id'), False)
            if item['method'] == 'topic' else False
        )
        counts = dict(conn.execute(
            'SELECT status, COUNT(*) AS count FROM paper_filter_results '
            'WHERE filter_id = ? GROUP BY status',
            (item['filter_id'],),
        ).fetchall())
        recorded = sum(counts.values())
        counts['unavailable'] = counts.get('unavailable', 0) + max(0, paper_count - recorded)
        item['counts'] = {status: counts.get(status, 0) for status in sorted(FILTER_STATUSES)}
    if filters:
        counts = dict(conn.execute(
            'SELECT status, COUNT(*) AS count FROM paper_filter_state GROUP BY status'
        ).fetchall())
        recorded = sum(counts.values())
        counts['unavailable'] = counts.get('unavailable', 0) + max(0, paper_count - recorded)
        final_counts = {status: counts.get(status, 0) for status in sorted(FILTER_STATUSES)}
    else:
        final_counts = {'excluded': 0, 'included': paper_count, 'unavailable': 0}
    reasons = dict(conn.execute(
        "SELECT unavailable_reason, COUNT(*) FROM paper_filter_results "
        "WHERE status = 'unavailable' AND unavailable_reason != '' GROUP BY unavailable_reason"
    ).fetchall())
    return {
        'filters': filters,
        'expression': filter_expression(filters),
        'counts': final_counts,
        'unavailable_reasons': reasons,
        'stale_topic_filters': [item['name'] for item in filters if item['stale']],
    }


def apply_regex_filter(db_path: str | PathLike[str],
                       rules_path: str | PathLike[str],
                       fields: Iterable[str] | None = None,
                       join_operator: str | None = None,
                       replace: bool = False,
                       timeout_ms: int | None = None) -> _FilterOverview:
    """Apply or replace one named regex filter in a corpus.

    Parameters
    ----------
    db_path : str or pathlib.Path
        Path to the SQLite paper corpus.
    rules_path : str or pathlib.Path
        Path to the JSON regex definition.
    fields : Iterable[str] or None, optional
        Content fields overriding the definition.
    join_operator : {'and', 'or'} or None, optional
        Operator joining this filter to the preceding active filter.
    replace : bool, default=False
        Whether to replace an active filter with the same name.
    timeout_ms : int or None, optional
        Per-field regex timeout overriding the definition.

    Returns
    -------
    _FilterOverview
        Updated filter overview.

    Raises
    ------
    OSError
        If the rules file cannot be read.
    ValueError
        If the definition, join operator, or replacement request is invalid.
    """
    definition = load_regex_definition(rules_path, fields=fields, timeout_ms=timeout_ms)
    compiled = compile_regex_definition(definition)
    if join_operator is not None:
        join_operator = join_operator.lower()
        if join_operator not in JOIN_OPERATORS:
            raise ValueError('join_operator must be "and" or "or".')

    with connect(db_path) as conn:
        existing = conn.execute(
            'SELECT * FROM corpus_filters WHERE name = ?', (definition['name'],)
        ).fetchone()
        if existing is not None and not replace:
            raise ValueError(
                f'Filter {definition["name"]!r} is already active; use --replace to reevaluate it.'
            )
        stack_size = conn.execute('SELECT COUNT(*) FROM corpus_filters').fetchone()[0]
        now = utc_now()
        try:
            conn.execute('BEGIN')
            if existing is None:
                position = stack_size
                if position == 0 and join_operator is not None:
                    raise ValueError('The first filter cannot have a join operator.')
                resolved_join = None if position == 0 else (join_operator or 'and')
                cursor = conn.execute(
                    """
                    INSERT INTO corpus_filters (
                        name, method, description, definition_json, stack_position,
                        join_operator, created_at, updated_at
                    ) VALUES (?, 'regex', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        definition['name'], definition['description'],
                        json.dumps(definition, sort_keys=True), position,
                        resolved_join, now, now,
                    ),
                )
                filter_id = cursor.lastrowid
            else:
                if existing['method'] != 'regex':
                    raise ValueError(
                        f'Filter {definition["name"]!r} already exists with method '
                        f'{existing["method"]!r}.'
                    )
                position = existing['stack_position']
                if position == 0 and join_operator is not None:
                    raise ValueError('The first filter cannot have a join operator.')
                resolved_join = existing['join_operator'] if join_operator is None else join_operator
                conn.execute(
                    """
                    UPDATE corpus_filters
                    SET description = ?, definition_json = ?, join_operator = ?, updated_at = ?
                    WHERE filter_id = ?
                    """,
                    (
                        definition['description'], json.dumps(definition, sort_keys=True),
                        resolved_join, now, existing['filter_id'],
                    ),
                )
                filter_id = existing['filter_id']
                conn.execute('DELETE FROM paper_filter_results WHERE filter_id = ?', (filter_id,))

            for paper in paper_rows(conn):
                status, evidence, unavailable_reason = evaluate_regex_paper(
                    conn, paper, definition, compiled
                )
                conn.execute(
                    """
                    INSERT INTO paper_filter_results (
                        filter_id, paper_id, status, evidence_json,
                        unavailable_reason, evaluated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        filter_id, paper['paper_id'], status,
                        json.dumps(evidence, sort_keys=True), unavailable_reason, now,
                    ),
                )
            _recompute_filter_state(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return filter_overview(conn)


def apply_topic_filter(
    db_path: str | PathLike[str],
    rules_path: str | PathLike[str],
    join_operator: str | None = None,
    replace: bool = False,
) -> _FilterOverview:
    """Apply or replace one named stored-model topic filter.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the SQLite paper corpus.
    rules_path : str or os.PathLike[str]
        Topic-filter JSON definition.
    join_operator : {'and', 'or'} or None, optional
        Operator joining a new filter to the preceding active expression.
    replace : bool, default=False
        Whether to replace and reevaluate an active filter with the same name.

    Returns
    -------
    _FilterOverview
        Updated filter stack and paper-status summary.

    Raises
    ------
    OSError
        If the rules file or corpus cannot be accessed.
    ValueError
        If the definition, join operation, stored model, or stored scores are
        invalid or stale.
    """
    if join_operator is not None:
        join_operator = join_operator.lower()
        if join_operator not in JOIN_OPERATORS:
            raise ValueError('join_operator must be "and" or "or".')
    with connect(db_path) as conn:
        definition = load_topic_definition(conn, rules_path)
        model = conn.execute(
            'SELECT prediction_corpus_fingerprint, text_fields_json FROM topic_models '
            'WHERE model_id = ?', (definition['model_id'],)
        ).fetchone()
        from paperminer.workflows.topics import topic_corpus_fingerprint

        current_fingerprint = topic_corpus_fingerprint(
            db_path, json.loads(model['text_fields_json'])
        )['sha256']
        if current_fingerprint != model['prediction_corpus_fingerprint']:
            raise ValueError(
                f'Topic scores for {definition["model"]!r} are stale; '
                'run pm topics store again before applying this filter.'
            )
        existing = conn.execute(
            'SELECT * FROM corpus_filters WHERE name = ?', (definition['name'],)
        ).fetchone()
        if existing is not None and not replace:
            raise ValueError(
                f'Filter {definition["name"]!r} is already active; use --replace to reevaluate it.'
            )
        stack_size = conn.execute('SELECT COUNT(*) FROM corpus_filters').fetchone()[0]
        now = utc_now()
        try:
            conn.execute('BEGIN')
            if existing is None:
                position = stack_size
                if position == 0 and join_operator is not None:
                    raise ValueError('The first filter cannot have a join operator.')
                resolved_join = None if position == 0 else (join_operator or 'and')
                cursor = conn.execute(
                    """
                    INSERT INTO corpus_filters (
                        name, method, description, definition_json, stack_position,
                        join_operator, created_at, updated_at
                    ) VALUES (?, 'topic', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        definition['name'], definition['description'],
                        json.dumps(definition, sort_keys=True), position,
                        resolved_join, now, now,
                    ),
                )
                filter_id = cursor.lastrowid
            else:
                if existing['method'] != 'topic':
                    raise ValueError(
                        f'Filter {definition["name"]!r} already exists with method '
                        f'{existing["method"]!r}.'
                    )
                position = existing['stack_position']
                if position == 0 and join_operator is not None:
                    raise ValueError('The first filter cannot have a join operator.')
                resolved_join = existing['join_operator'] if join_operator is None else join_operator
                conn.execute(
                    """
                    UPDATE corpus_filters
                    SET description = ?, definition_json = ?, join_operator = ?, updated_at = ?
                    WHERE filter_id = ?
                    """,
                    (
                        definition['description'], json.dumps(definition, sort_keys=True),
                        resolved_join, now, existing['filter_id'],
                    ),
                )
                filter_id = existing['filter_id']
                conn.execute('DELETE FROM paper_filter_results WHERE filter_id = ?', (filter_id,))
            for paper in paper_rows(conn):
                status, evidence, unavailable_reason = evaluate_topic_paper(
                    conn, paper, definition
                )
                conn.execute(
                    """
                    INSERT INTO paper_filter_results (
                        filter_id, paper_id, status, evidence_json,
                        unavailable_reason, evaluated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        filter_id, paper['paper_id'], status,
                        json.dumps(evidence, sort_keys=True), unavailable_reason, now,
                    ),
                )
            _recompute_filter_state(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return filter_overview(conn)


def refresh_topic_filters(
    db_path: str | PathLike[str],
    model_id: str,
) -> None:
    """Reevaluate filters after a stored model's scores are refreshed.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the SQLite paper corpus.
    model_id : str
        Immutable identifier of the refreshed topic model.

    Returns
    -------
    None
        Matching active filters and the combined filter state are updated in
        the corpus transactionally.
    """
    with connect(db_path) as conn:
        filters = [
            item for item in active_filter_stack(conn)
            if item['method'] == 'topic' and item['definition'].get('model_id') == model_id
        ]
        if not filters:
            return
        now = utc_now()
        try:
            conn.execute('BEGIN')
            for item in filters:
                conn.execute(
                    'DELETE FROM paper_filter_results WHERE filter_id = ?',
                    (item['filter_id'],),
                )
                for paper in paper_rows(conn):
                    status, evidence, reason = evaluate_topic_paper(
                        conn, paper, item['definition']
                    )
                    conn.execute(
                        """
                        INSERT INTO paper_filter_results (
                            filter_id, paper_id, status, evidence_json,
                            unavailable_reason, evaluated_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item['filter_id'], paper['paper_id'], status,
                            json.dumps(evidence, sort_keys=True), reason, now,
                        ),
                    )
            _recompute_filter_state(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def reset_filters(
    db_path: str | PathLike[str],
    name: str | None = None,
    all_filters: bool = False,
) -> _FilterOverview:
    """Remove filters and recompute final paper decisions.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the SQLite paper corpus.
    name : str or None, optional
        Name of the single filter to remove.
    all_filters : bool, default=False
        Whether to remove the complete active stack.

    Returns
    -------
    _FilterOverview
        Updated filter overview.

    Raises
    ------
    ValueError
        If exactly one removal mode is not selected or ``name`` is not active.
    """
    if bool(name) == bool(all_filters):
        raise ValueError('Choose exactly one of a filter name or all filters.')
    with connect(db_path) as conn:
        try:
            conn.execute('BEGIN')
            if all_filters:
                conn.execute('DELETE FROM corpus_filters')
            else:
                cursor = conn.execute('DELETE FROM corpus_filters WHERE name = ?', (name,))
                if cursor.rowcount == 0:
                    raise ValueError(f'No active filter named {name!r}.')
                remaining = conn.execute(
                    'SELECT filter_id, join_operator FROM corpus_filters '
                    'ORDER BY stack_position, filter_id'
                ).fetchall()
                for position, row in enumerate(remaining):
                    conn.execute(
                        'UPDATE corpus_filters SET stack_position = ?, join_operator = ? WHERE filter_id = ?',
                        (
                            position,
                            None if position == 0 else row['join_operator'] or 'and',
                            row['filter_id'],
                        ),
                    )
            _recompute_filter_state(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return filter_overview(conn)


def current_filter_statuses(conn: sqlite3.Connection) -> dict[str, _FilterStatus]:
    """Return current final statuses keyed by paper ID.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.

    Returns
    -------
    dict[str, _FilterStatus]
        Final filter status for each evaluated paper ID.
    """
    stale_models = topic_filter_staleness(conn)
    stale = [model_id for model_id, is_stale in stale_models.items() if is_stale]
    if stale:
        raise ValueError(
            'Active topic filters use stale scores; run pm topics store again for: '
            + ', '.join(stale)
        )
    return {
        row['paper_id']: row['status']
        for row in conn.execute('SELECT paper_id, status FROM paper_filter_state')
    }
