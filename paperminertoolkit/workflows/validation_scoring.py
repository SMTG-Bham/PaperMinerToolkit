"""Auditable precision, recall, and F1 from human validation decisions."""

from __future__ import annotations

import ast
import json
from typing import Any

RUN_FIELDS = ('extraction_mode', 'model_identifier', 'provider', 'context_length',
              'software_commit', 'run_date', 'numeric_matching_rule')
GAP_FIELDS = ('Band gap', 'All band gaps', 'Cited literature band gaps')
_ABSENT = object()


def _present(value: object) -> bool:
    """Whether a scalar field has an assessable value."""
    return str(value if value is not None else '').strip().casefold() not in {'', 'none', 'null', 'nan'}


def _counts() -> dict[str, int]:
    """Create one field's count accumulator."""
    return {'tp': 0, 'fp': 0, 'fn': 0, 'pending': 0}


def _decode(value: object) -> object:
    """Decode JSON or Python-style containers, retaining malformed cells."""
    raw = str(value if value is not None else '').strip()
    if not raw or raw.casefold() in {'none', 'null', 'nan'}:
        return _ABSENT
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        if not raw.startswith(('[', '{')):
            return raw
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
            return raw
        if not isinstance(parsed, (list, dict)):
            return raw
    return _ABSENT if parsed is None else parsed


def _child(path: str, part: object) -> str:
    """Build a stable JSON Pointer for a nested answer part."""
    escaped = str(part).replace('~', '~0').replace('/', '~1')
    return f'{path}/{escaped}'


def _component(path: str) -> str:
    """Combine list positions into one per-key statistic across records."""
    if not path:
        return '<value>'
    pieces = [part.replace('~1', '/').replace('~0', '~') for part in path.split('/')[1:]]
    result = ''
    for piece in pieces:
        result += '[]' if piece.isdecimal() else ('.' if result else '') + piece
    return result


def _bump(count: dict[str, int], components: dict[str, dict[str, int]],
          path: str, kind: str) -> None:
    """Increment both a recipe field and its individual part."""
    count[kind] += 1
    components.setdefault(_component(path), _counts())[kind] += 1


def _tree(value: object, path: str, kind: str, count: dict[str, int],
          components: dict[str, dict[str, int]]) -> None:
    """Count every populated leaf of a wholly missing, extra, or pending value."""
    before = count[kind]
    _tree_leaves(value, path, kind, count, components)
    if kind == 'pending' and count[kind] == before:
        _bump(count, components, path, 'pending')


def _tree_leaves(value: object, path: str, kind: str, count: dict[str, int],
                 components: dict[str, dict[str, int]]) -> None:
    """Walk a value without treating an empty child as a separate pending part."""
    if isinstance(value, dict):
        for key, child in value.items():
            _tree_leaves(child, _child(path, key), kind, count, components)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _tree_leaves(child, _child(path, index), kind, count, components)
    elif value is not _ABSENT and _present(value):
        _bump(count, components, path, kind)


def _extra_status(mark: object) -> str:
    """Read an extra's accuracy, preserving older boolean FP decisions."""
    if mark is True:
        return 'incorrect'
    if isinstance(mark, dict) and mark.get('status') in {'correct', 'incorrect'}:
        return mark['status']
    return 'unreviewed'


def _score_extra(value: object, path: str, status: str,
                 decision: dict[str, Any], count: dict[str, int],
                 components: dict[str, dict[str, int]]) -> None:
    """Score each populated leaf of an unpaired scraped value."""
    if isinstance(value, dict):
        for key, child in value.items():
            _score_extra(child, _child(path, key), status, decision, count, components)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _score_extra(child, _child(path, index), status, decision, count, components)
    elif value is not _ABSENT and _present(value):
        leaf_status = decision.get('values', {}).get(path, {}).get('status', status)
        _bump(count, components, path,
              'tp' if leaf_status == 'correct' else
              'fp' if leaf_status == 'incorrect' else 'pending')


def _score_extra_row(row: dict[str, Any], fields: list[str], mark: object,
                     field_decisions: dict[str, Any], part_decisions: dict[str, Any],
                     counts: dict[str, dict[str, int]],
                     components: dict[str, dict[str, dict[str, int]]]) -> None:
    """Apply row, field, then individual-part accuracy decisions."""
    status = _extra_status(mark)
    for field in fields:
        review = part_decisions.get(row['id'], {}).get(field, {})
        field_status = field_decisions.get(row['id'], {}).get(field, {}).get('status', status)
        _score_extra(_decode(row['values'][field]), '', field_status,
                     review, counts[field], components[field])


def _score_node(gold: object, scraped: object, path: str,
                decision: dict[str, Any], count: dict[str, int],
                components: dict[str, dict[str, int]]) -> None:
    """Score scalar leaves and human-paired list items recursively."""
    if isinstance(gold, dict) or isinstance(scraped, dict):
        if gold is not _ABSENT and not isinstance(gold, dict):
            _score_node(gold, _ABSENT, path, decision, count, components)
            _score_node(_ABSENT, scraped, path, decision, count, components)
            return
        if scraped is not _ABSENT and not isinstance(scraped, dict):
            _score_node(gold, _ABSENT, path, decision, count, components)
            _score_node(_ABSENT, scraped, path, decision, count, components)
            return
        left = gold if isinstance(gold, dict) else {}
        right = scraped if isinstance(scraped, dict) else {}
        for key in left.keys() | right.keys():
            _score_node(left.get(key, _ABSENT), right.get(key, _ABSENT),
                        _child(path, key), decision, count, components)
        return
    if isinstance(gold, list) or isinstance(scraped, list):
        if gold is not _ABSENT and not isinstance(gold, list):
            _score_node(gold, _ABSENT, path, decision, count, components)
            _score_node(_ABSENT, scraped, path, decision, count, components)
            return
        if scraped is not _ABSENT and not isinstance(scraped, list):
            _score_node(gold, _ABSENT, path, decision, count, components)
            _score_node(_ABSENT, scraped, path, decision, count, components)
            return
        left = gold if isinstance(gold, list) else []
        right = scraped if isinstance(scraped, list) else []
        review = decision.get('lists', {}).get(path, {})
        matches = review.get('matches', {})
        extras = review.get('extra', {})
        used: set[int] = set()
        for index, item in enumerate(left):
            match = matches.get(str(index), {})
            other = match.get('scraped')
            valid = (isinstance(other, int) and not isinstance(other, bool)
                     and 0 <= other < len(right) and other not in used)
            child_path = _child(path, index)
            if match.get('status') == 'missing' and other is None:
                _tree(item, child_path, 'fn', count, components)
            elif valid:
                used.add(other)
                _score_node(item, right[other], child_path, decision, count, components)
            else:
                _tree(item, child_path, 'pending', count, components)
        for index, item in enumerate(right):
            if index not in used:
                _score_extra(item, _child(path, index),
                             _extra_status(extras.get(str(index))), decision,
                             count, components)
        return
    a = gold is not _ABSENT and _present(gold)
    b = scraped is not _ABSENT and _present(scraped)
    if not a and not b:
        return
    status = decision.get('values', {}).get(path, {}).get('status')
    if a and b:
        if status == 'correct':
            _bump(count, components, path, 'tp')
        elif status == 'incorrect':
            _bump(count, components, path, 'fp')
            _bump(count, components, path, 'fn')
        else:
            _bump(count, components, path, 'pending')
    elif a:
        _bump(count, components, path, 'fn' if status == 'missing' else 'pending')
    else:
        status = decision.get('values', {}).get(path, {}).get(
            'status', _extra_status(decision.get('extraKeys', {}).get(path)))
        _bump(count, components, path,
              'tp' if status == 'correct' else 'fp' if status == 'incorrect' else 'pending')


def _metrics(count: dict[str, int]) -> dict[str, Any]:
    """Expose provisional ratios from decided items, even while work is pending."""
    result: dict[str, Any] = dict(count)
    result['complete'] = count['pending'] == 0
    result['precision'] = None
    result['recall'] = None
    result['f1'] = None
    tp, fp, fn = count['tp'], count['fp'], count['fn']
    if tp + fp:
        result['precision'] = tp / (tp + fp)
    if tp + fn:
        result['recall'] = tp / (tp + fn)
    if result['precision'] is not None and result['recall'] is not None:
        denominator = result['precision'] + result['recall']
        result['f1'] = 2 * result['precision'] * result['recall'] / denominator if denominator else 0.0
    return result


def score_review(session: dict[str, Any], decisions: dict[str, Any]) -> dict[str, Any]:
    """Score a review with explicit decisions for every nested answer part."""
    fields = session['fields']
    identity_fields = set(session['identity_fields'])
    structure_fields = session.get('structure_fields', {})
    counts = {field: _counts() for field in fields}
    components: dict[str, dict[str, dict[str, int]]] = {field: {} for field in fields}
    paper_pairs = decisions.get('paperPairs', {})
    material_pairs = decisions.get('materialPairs', {})
    field_decisions = decisions.get('fields', {})
    part_decisions = decisions.get('parts', {})
    marked_extra = decisions.get('extra', {})
    scraped_by_id = {paper['id']: paper for paper in session['scraped']}
    linked_papers: set[str] = set()
    pending_pairs = 0
    for paper in session['validation']:
        paper_id = paper_pairs.get(paper['id'], paper['suggested'])
        scraped_paper = scraped_by_id.get(paper_id)
        if scraped_paper:
            if paper_id in linked_papers:
                pending_pairs += 1
                scraped_paper = None
            else:
                linked_papers.add(paper_id)
        scraped_rows = {row['id']: row for row in scraped_paper['rows']} if scraped_paper else {}
        suggested = paper.get('material_suggestions', {}) if scraped_paper and paper_id == paper['suggested'] else {}
        used_rows: set[str] = set()
        for row in paper['rows']:
            explicit = row['id'] in material_pairs
            partner_id = material_pairs.get(row['id'], suggested.get(row['id']))
            partner = scraped_rows.get(partner_id)
            if partner_id and (not partner or partner_id in used_rows):
                pending_pairs += 1
                partner = None
            if partner:
                used_rows.add(partner_id)
                for field in fields:
                    review = (part_decisions.get(row['id'], {}).get(field, {})
                              if field in structure_fields else
                              {'values': {'': field_decisions.get(row['id'], {}).get(field, {})}})
                    _score_node(_decode(row['values'][field]), _decode(partner['values'][field]),
                                '', review, counts[field], components[field])
            elif explicit and partner_id is None:
                for field in fields:
                    _tree(_decode(row['values'][field]), '', 'fn',
                          counts[field], components[field])
            else:
                pending_pairs += 1
                for field in fields:
                    _bump(counts[field], components[field], '', 'pending')
        if scraped_paper:
            for row in scraped_paper['rows']:
                if row['id'] not in used_rows:
                    if marked_extra.get(row['id']):
                        _score_extra_row(row, fields, marked_extra[row['id']],
                                         field_decisions, part_decisions, counts, components)
                    else:
                        pending_pairs += 1
                        for field in fields:
                            _bump(counts[field], components[field], '', 'pending')
    for paper in session['scraped']:
        if paper['id'] not in linked_papers:
            for row in paper['rows']:
                if marked_extra.get(row['id']):
                    _score_extra_row(row, fields, marked_extra[row['id']],
                                     field_decisions, part_decisions, counts, components)
                else:
                    pending_pairs += 1
                    for field in fields:
                        _bump(counts[field], components[field], '', 'pending')
    run = decisions.get('run', {})
    missing_metadata = [key for key in RUN_FIELDS if not str(run.get(key, '')).strip()]
    scored = {}
    for field, count in counts.items():
        metric = _metrics(count)
        metric['type'] = ('identity' if field in identity_fields else
                          'numeric' if field == 'Band gap' else
                          'numeric list' if field in GAP_FIELDS else
                          structure_fields.get(field, 'scalar'))
        metric['parts'] = {name: _metrics(part)
                           for name, part in sorted(components[field].items())}
        scored[field] = metric
    return {'schema_version': 3, 'protocol': 'Human one-to-one paper/material/list-item pairing '
            'with decisions on every populated scalar leaf, including dictionary keys. '
            'Correct extras count TP; incorrect extras count FP. '
            'Incorrect matched leaves count one FP and one FN; empty values count neither. '
            'Undefined denominators are null. Suggested pairs never imply correctness.',
            'fingerprint': session['fingerprint'], 'sources': session['sources'],
            'recipe': session['recipe'], 'run': run,
            'papers_evaluated': len(session['validation']),
            'pending_pairs': pending_pairs, 'missing_metadata': missing_metadata,
            'fields': scored,
            'complete': not pending_pairs and not missing_metadata and all(v['complete'] for v in scored.values())}
