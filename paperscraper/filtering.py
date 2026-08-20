"""Persistent post-download filtering for paper corpora.

Regex filters are evaluated independently for every paper and then combined in
application order.  The generic SQLite tables deliberately keep filter
definitions and paper-level decisions separate from scrape pipeline statuses.
"""

import json
import re
from pathlib import Path

import regex

from paperscraper.corpus import connect, get_asset, paper_rows, utc_now
from paperscraper.documents import read_pdf_bytes, trim_reference_section


FILTER_FIELDS = {'title', 'abstract', 'full_text'}
FILTER_STATUSES = {'included', 'excluded', 'unavailable'}
JOIN_OPERATORS = {'and', 'or'}
DEFAULT_TIMEOUT_MS = 500
MAX_MATCHES_PER_FIELD = 1000
MAX_SNIPPETS_PER_FIELD = 3
SNIPPET_CONTEXT = 80


def _require_type(value, expected_type, label):
    if not isinstance(value, expected_type):
        raise ValueError(f'{label} must be {expected_type.__name__}.')
    return value


def _normalize_rule(rule, label):
    _require_type(rule, dict, label)
    name = str(rule.get('name') or '').strip()
    pattern = rule.get('pattern')
    if not name:
        raise ValueError(f'{label} requires a non-empty name.')
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f'{label} requires a non-empty string pattern.')
    return {'name': name, 'pattern': pattern}


def normalize_regex_definition(definition, fields=None, timeout_ms=None):
    """Validate and return a deterministic regex filter definition."""
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


def load_regex_definition(path, fields=None, timeout_ms=None):
    """Load, validate, and normalize a JSON regex filter definition."""
    try:
        definition = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'Invalid filter JSON: {error}') from error
    return normalize_regex_definition(definition, fields=fields, timeout_ms=timeout_ms)


def compile_regex_definition(definition):
    """Compile all patterns before any corpus state is changed."""
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


def _decode_text_asset(asset):
    if asset is None:
        return None
    text = asset['content'].decode('utf-8', errors='replace')
    text = trim_reference_section(text)
    return text if text.strip() else None


def _paper_content(conn, paper, fields):
    """Load selected raw fields and return content, sources, and unavailable reasons."""
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


def _snippet(text, start, end):
    snippet = text[max(0, start - SNIPPET_CONTEXT):min(len(text), end + SNIPPET_CONTEXT)]
    return re.sub(r'\s+', ' ', snippet).strip()


def _match_rule(expression, content, timeout_seconds):
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


def evaluate_regex_paper(conn, paper, definition, compiled):
    """Evaluate one normalized definition against one corpus paper."""
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


def active_filter_stack(conn):
    """Return active filter rows in expression order with decoded definitions."""
    rows = conn.execute(
        'SELECT * FROM corpus_filters ORDER BY stack_position, filter_id'
    ).fetchall()
    filters = []
    for row in rows:
        item = dict(row)
        item['definition'] = json.loads(item.pop('definition_json'))
        filters.append(item)
    return filters


def filter_expression(filters):
    """Return the explicit left-to-right expression for an active filter stack."""
    if not filters:
        return ''
    expression = filters[0]['name']
    for item in filters[1:]:
        expression = f'({expression} {item["join_operator"].upper()} {item["name"]})'
    return expression


def combine_status(left, right, operator):
    """Combine two filter decisions with three-valued Boolean logic."""
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


def _recompute_filter_state(conn):
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


def filter_overview(conn):
    """Return active stack metadata and per-filter/final status totals."""
    filters = active_filter_stack(conn)
    paper_count = conn.execute('SELECT COUNT(*) FROM papers').fetchone()[0]
    for item in filters:
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
    }


def apply_regex_filter(db_path, rules_path, fields=None, join_operator=None,
                       replace=False, timeout_ms=None):
    """Apply or explicitly replace one named regex filter in a corpus."""
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


def reset_filters(db_path, name=None, all_filters=False):
    """Remove one named filter or the complete active stack and recompute state."""
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


def current_filter_statuses(conn):
    """Return current final statuses keyed by paper id."""
    return {
        row['paper_id']: row['status']
        for row in conn.execute('SELECT paper_id, status FROM paper_filter_state')
    }
