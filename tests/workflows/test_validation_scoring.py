"""Exercise human-review scoring of paired, missing, extra, and nested answer parts."""

from __future__ import annotations

from typing import Any

import pytest

from paperminertoolkit.workflows import validation_scoring as scoring
from paperminertoolkit.workflows.validation_scoring import RUN_FIELDS, score_review

COMPLETE_RUN = {key: 'recorded' for key in RUN_FIELDS}


def _row(row_id: str, values: dict[str, str]) -> dict[str, Any]:
    """Build one CSV-derived review row; fields left out are empty cells."""
    return {'id': row_id, 'values': values}


def _gold(paper_id: str, rows: list[dict[str, Any]], suggested: str | None = None,
          material_suggestions: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a validation paper with its provisional paper and material suggestions."""
    return {'id': paper_id, 'suggested': suggested, 'rows': rows,
            'material_suggestions': material_suggestions or {}}


def _scraped(paper_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a scraped paper."""
    return {'id': paper_id, 'rows': rows}


def _session(validation: list[dict[str, Any]], scraped: list[dict[str, Any]],
             fields: tuple[str, ...] = ('Material', 'Gap'),
             structure_fields: dict[str, str] | None = None) -> dict[str, Any]:
    """Build the session shape that ReviewApp.load hands to the scorer."""
    for paper in validation + scraped:
        for row in paper['rows']:
            row['values'] = {field: row['values'].get(field, '') for field in fields}
    return {'fingerprint': 'f' * 64, 'sources': {'validation': {'name': 'gold.csv'}},
            'recipe': 'test recipe', 'fields': list(fields), 'identity_fields': ['Material'],
            'structure_fields': structure_fields or {},
            'validation': validation, 'scraped': scraped}


def _paired_session(gold: dict[str, str], scraped: dict[str, str],
                    **options: Any) -> dict[str, Any]:
    """Build one validation row already suggested against one scraped row."""
    return _session([_gold('v:0', [_row('v:2', gold)], 's:0', {'v:2': 's:2'})],
                    [_scraped('s:0', [_row('s:2', scraped)])], **options)


def _counts(metric: dict[str, Any]) -> dict[str, int]:
    """Keep only the raw confusion counts from a scored field or part."""
    return {key: metric[key] for key in ('tp', 'fp', 'fn', 'pending')}


def _expect(**counts: int) -> dict[str, int]:
    """Fill unspecified confusion counts with zero."""
    return {'tp': 0, 'fp': 0, 'fn': 0, 'pending': 0, **counts}


@pytest.mark.parametrize(('cell', 'decoded'), [
    ('', scoring._ABSENT),
    (None, scoring._ABSENT),
    (' NaN ', scoring._ABSENT),
    ('None', scoring._ABSENT),
    ('null', scoring._ABSENT),
    ('[1, 2]', [1, 2]),
    ('{"k": "v"}', {'k': 'v'}),
    ("[{'value': '1 eV', 'conditions': None}]", [{'value': '1 eV', 'conditions': None}]),
    ('3', 3),
    ('"quoted"', 'quoted'),
    ('2 eV', '2 eV'),
    ('[unterminated', '[unterminated'),
    ('{1, 2}', '{1, 2}'),
])
def test_decode_distinguishes_missing_markers_containers_and_text(cell: object, decoded: object) -> None:
    """Treat missing markers as absent, parse JSON or Python containers, and keep other text."""
    assert scoring._decode(cell) == decoded


@pytest.mark.parametrize(('parts', 'component'), [
    ((), '<value>'),
    ((0, 'value'), '[].value'),
    (('key', 0, 'sub'), 'key[].sub'),
    ((0, 1), '[][]'),
    (('a/b', 'c~d'), 'a/b.c~d'),
    (('~1',), '~1'),
])
def test_components_merge_list_positions_and_unescape_pointer_keys(
    parts: tuple[object, ...], component: str,
) -> None:
    """Group list items by key while round-tripping JSON Pointer escapes."""
    path = ''
    for part in parts:
        path = scoring._child(path, part)
    assert scoring._component(path) == component


@pytest.mark.parametrize(('counts', 'precision', 'recall', 'f1', 'complete'), [
    (_expect(), None, None, None, True),
    (_expect(fp=2, fn=1), 0.0, 0.0, 0.0, True),
    (_expect(fp=1), 0.0, None, None, True),
    (_expect(tp=3, fp=1, pending=2), 0.75, 1.0, 6 / 7, False),
])
def test_metrics_leave_undefined_ratios_null_and_report_provisional_values(
    counts: dict[str, int], precision: float | None, recall: float | None,
    f1: float | None, complete: bool,
) -> None:
    """Compute ratios from decided items only and flag pending work as incomplete."""
    result = scoring._metrics(counts)
    assert result['precision'] == pytest.approx(precision)
    assert result['recall'] == pytest.approx(recall)
    assert result['f1'] == pytest.approx(f1)
    assert result['complete'] is complete
    assert _counts(result) == counts


@pytest.mark.parametrize(('gold', 'scraped', 'decision', 'expected'), [
    ('1 eV', '1 eV', {'status': 'correct'}, _expect(tp=1)),
    ('1 eV', '2 eV', {'status': 'incorrect'}, _expect(fp=1, fn=1)),
    ('1 eV', '1 eV', {}, _expect(pending=1)),
    ('1 eV', '', {'status': 'missing'}, _expect(fn=1)),
    ('1 eV', 'None', {'status': 'correct'}, _expect(pending=1)),
    ('', '4 eV', {'status': 'correct'}, _expect(tp=1)),
    ('nan', '4 eV', {'status': 'incorrect'}, _expect(fp=1)),
    ('', '4 eV', {}, _expect(pending=1)),
    ('', 'null', {'status': 'correct'}, _expect()),
])
def test_scalar_decisions_follow_the_scoring_protocol(
    gold: str, scraped: str, decision: dict[str, str], expected: dict[str, int],
) -> None:
    """Score matched, missing, extra, and empty scalar values from explicit decisions only."""
    session = _paired_session({'Material': 'A', 'Gap': gold}, {'Material': 'A', 'Gap': scraped})
    result = score_review(session, {'fields': {'v:2': {'Gap': decision}}})
    gap = result['fields']['Gap']
    assert _counts(gap) == expected
    assert {name: _counts(part) for name, part in gap['parts'].items()} == (
        {'<value>': expected} if any(expected.values()) else {}
    )
    assert _counts(result['fields']['Material']) == _expect(pending=1)


def test_review_reports_types_metadata_and_completion() -> None:
    """Label field types and require run metadata before a review is complete."""
    fields = ('Material', 'Band gap', 'All band gaps', 'Conditions', 'Notes')
    session = _session([], [], fields=fields,
                       structure_fields={'Band gap': 'list', 'All band gaps': 'list',
                                         'Conditions': 'dict'})
    unrecorded = score_review(session, {'run': {**COMPLETE_RUN, 'provider': '  '}})
    assert {field: metric['type'] for field, metric in unrecorded['fields'].items()} == {
        'Material': 'identity', 'Band gap': 'numeric', 'All band gaps': 'numeric list',
        'Conditions': 'dict', 'Notes': 'scalar',
    }
    assert unrecorded['missing_metadata'] == ['provider']
    assert unrecorded['complete'] is False
    assert score_review(session, {})['missing_metadata'] == list(RUN_FIELDS)

    recorded = score_review(session, {'run': COMPLETE_RUN})
    assert recorded['complete'] is True
    assert recorded['schema_version'] == 3
    assert (recorded['fingerprint'], recorded['recipe']) == ('f' * 64, 'test recipe')
    assert recorded['sources'] == session['sources']
    assert recorded['run'] == COMPLETE_RUN
    assert (recorded['papers_evaluated'], recorded['pending_pairs']) == (0, 0)


def test_explicit_missing_pairs_and_marked_extras_complete_a_review() -> None:
    """Count every populated leaf of a missing row as FN and of an incorrect extra as FP."""
    session = _paired_session(
        {'Material': 'A', 'Gaps': '[{"value": "1 eV", "method": ""}, {"value": ""}]'},
        {'Material': 'B', 'Gaps': '[{"value": "9 eV", "method": "GW"}]'},
        fields=('Material', 'Gaps'), structure_fields={'Gaps': 'list'},
    )
    unresolved = score_review(session, {'materialPairs': {'v:2': None}, 'run': COMPLETE_RUN})
    assert unresolved['pending_pairs'] == 1
    assert unresolved['complete'] is False

    result = score_review(session, {'materialPairs': {'v:2': None},
                                    'extra': {'s:2': {'status': 'incorrect'}},
                                    'run': COMPLETE_RUN})
    assert result['pending_pairs'] == 0
    assert result['complete'] is True
    assert _counts(result['fields']['Material']) == _expect(fp=1, fn=1)
    assert _counts(result['fields']['Gaps']) == _expect(fp=2, fn=1)
    assert _counts(result['fields']['Gaps']['parts']['[].value']) == _expect(fp=1, fn=1)
    assert _counts(result['fields']['Gaps']['parts']['[].method']) == _expect(fp=1)


def test_paper_overrides_drop_suggestions_and_unlinked_scraped_rows_stay_pending() -> None:
    """Material suggestions only apply to the suggested paper; every unclaimed row needs a decision."""
    session = _session(
        [_gold('v:0', [_row('v:2', {'Material': 'A', 'Gap': '1 eV'})], 's:0', {'v:2': 's:2'})],
        [_scraped('s:0', [_row('s:2', {'Material': 'A', 'Gap': '1 eV'})]),
         _scraped('s:1', [_row('s:3', {'Material': 'A', 'Gap': '1 eV'})])],
    )
    correct = {'v:2': {'Material': {'status': 'correct'}, 'Gap': {'status': 'correct'}}}

    suggested = score_review(session, {'fields': correct, 'extra': {'s:3': True}})
    assert _counts(suggested['fields']['Gap']) == _expect(tp=1, fp=1)
    assert suggested['pending_pairs'] == 0

    overridden = score_review(session, {'paperPairs': {'v:0': 's:1'}, 'fields': correct})
    assert overridden['pending_pairs'] == 3
    assert _counts(overridden['fields']['Gap']) == _expect(pending=3)

    repaired = score_review(session, {'paperPairs': {'v:0': 's:1'},
                                      'materialPairs': {'v:2': 's:3'},
                                      'extra': {'s:2': {'status': 'correct'}},
                                      'fields': correct})
    assert repaired['pending_pairs'] == 0
    assert _counts(repaired['fields']['Gap']) == _expect(tp=2)

    unpaired = score_review(session, {'paperPairs': {'v:0': None},
                                      'extra': {'s:2': True, 's:3': True}})
    assert unpaired['pending_pairs'] == 1
    assert _counts(unpaired['fields']['Gap']) == _expect(fp=2, pending=1)


def test_scraped_papers_and_rows_link_to_at_most_one_validation_partner() -> None:
    """A second claim on an already-linked scraped paper or row stays pending."""
    session = _session(
        [_gold('v:0', [_row('v:2', {'Material': 'A', 'Gap': '1 eV'}),
                       _row('v:3', {'Material': 'B', 'Gap': '2 eV'})], 's:0', {'v:2': 's:2'}),
         _gold('v:1', [_row('v:4', {'Material': 'C', 'Gap': '3 eV'})])],
        [_scraped('s:0', [_row('s:2', {'Material': 'A', 'Gap': '1 eV'})])],
    )
    decisions = {'paperPairs': {'v:1': 's:0'}, 'materialPairs': {'v:3': 's:2'},
                 'fields': {'v:2': {'Gap': {'status': 'correct'}},
                            'v:3': {'Gap': {'status': 'correct'}}}}
    result = score_review(session, decisions)
    assert result['papers_evaluated'] == 2
    assert result['pending_pairs'] > 0
    assert result['complete'] is False
    assert _counts(result['fields']['Gap']) == _expect(tp=1, pending=2)


def test_extra_rows_apply_row_then_field_then_part_decisions() -> None:
    """Narrower decisions override broader ones; legacy boolean marks mean incorrect."""
    session = _session([], [_scraped('s:0', [
        _row('s:2', {'Material': 'A', 'Gaps': '[{"value": "1 eV", "method": "PBE"}]'}),
        _row('s:3', {'Material': 'B', 'Gaps': '[{"value": "2 eV"}]'}),
        _row('s:4', {'Material': 'C'}),
        _row('s:5', {'Material': 'D'}),
    ])], fields=('Material', 'Gaps'), structure_fields={'Gaps': 'list'})
    result = score_review(session, {
        'extra': {'s:2': True, 's:3': {'status': 'correct'},
                  's:4': {'status': 'unreviewed'}, 's:5': False},
        'fields': {'s:2': {'Material': {'status': 'correct'}}},
        'parts': {'s:2': {'Gaps': {'values': {'/0/value': {'status': 'correct'}}}}},
    })
    assert _counts(result['fields']['Material']) == _expect(tp=2, pending=2)
    assert _counts(result['fields']['Gaps']) == _expect(tp=2, fp=1, pending=1)
    assert _counts(result['fields']['Gaps']['parts']['[].value']) == _expect(tp=2)
    assert _counts(result['fields']['Gaps']['parts']['[].method']) == _expect(fp=1)
    # Unmarked rows hold one pending item per field, even where the cell is empty.
    assert _counts(result['fields']['Gaps']['parts']['<value>']) == _expect(pending=1)
    assert result['pending_pairs'] == 1


def test_list_items_are_scored_through_human_pairings() -> None:
    """Score matched items per leaf, unmatched gold items as FN, and leftovers as extras."""
    session = _paired_session(
        {'Material': 'A', 'Gaps': '[{"value": "1 eV", "method": "PBE"}, '
                                  '{"value": "2 eV", "method": "HSE"}, '
                                  '{"value": "3 eV", "method": ""}]'},
        {'Material': 'A', 'Gaps': "[{'value': '2 eV', 'method': 'HSE'}, "
                                  "{'value': '1.1 eV', 'method': 'PBE'}, "
                                  "{'value': '9 eV', 'method': 'GW'}]"},
        fields=('Material', 'Gaps'), structure_fields={'Gaps': 'list'},
    )
    review = {
        'lists': {'': {'matches': {'0': {'scraped': 1}, '1': {'scraped': 0},
                                   '2': {'status': 'missing', 'scraped': None}},
                       'extra': {'2': {'status': 'incorrect'}}}},
        'values': {'/0/value': {'status': 'incorrect'}, '/0/method': {'status': 'correct'},
                   '/1/value': {'status': 'correct'}, '/1/method': {'status': 'correct'}},
    }
    result = score_review(session, {'fields': {'v:2': {'Material': {'status': 'correct'}}},
                                    'parts': {'v:2': {'Gaps': review}}, 'run': COMPLETE_RUN})
    gaps = result['fields']['Gaps']
    assert _counts(gaps) == _expect(tp=3, fp=3, fn=2)
    assert _counts(gaps['parts']['[].value']) == _expect(tp=1, fp=2, fn=2)
    assert gaps['parts']['[].value']['precision'] == pytest.approx(1 / 3)
    assert _counts(gaps['parts']['[].method']) == _expect(tp=2, fp=1)
    assert gaps['type'] == 'list'
    assert result['complete'] is True


def test_invalid_list_pairings_leave_items_pending() -> None:
    """Reused, boolean, textual, and out-of-range partners never count as matches."""
    session = _paired_session(
        {'Gaps': '[{"value": "1 eV"}, {"value": "2 eV"}, {"value": ""}, '
                 '{"value": "4 eV"}, {"value": "5 eV"}, {"value": "6 eV"}]'},
        {'Gaps': '[{"value": "1 eV"}]'},
        fields=('Material', 'Gaps'), structure_fields={'Gaps': 'list'},
    )
    matches = {'0': {'scraped': 0}, '1': {'scraped': 0}, '3': {'scraped': True},
               '4': {'scraped': 7}, '5': {'scraped': '0'}}
    result = score_review(session, {'parts': {'v:2': {'Gaps': {
        'lists': {'': {'matches': matches}},
        'values': {'/0/value': {'status': 'correct'}},
    }}}})
    gaps = result['fields']['Gaps']
    assert _counts(gaps) == _expect(tp=1, pending=5)
    assert _counts(gaps['parts']['[].value']) == _expect(tp=1, pending=4)
    # An empty unmatched item still needs one decision of its own.
    assert _counts(gaps['parts']['[]']) == _expect(pending=1)


@pytest.mark.parametrize(('extra_key', 'expected'), [
    ({}, _expect(tp=1, pending=2)),
    ({'extraKeys': {'/strain': True}}, _expect(tp=1, fp=1, pending=1)),
    ({'extraKeys': {'/strain': True}, 'values': {'/strain': {'status': 'correct'}}},
     _expect(tp=2, pending=1)),
])
def test_dictionary_keys_are_scored_as_individual_leaves(
    extra_key: dict[str, Any], expected: dict[str, int],
) -> None:
    """Pair dictionary keys by name and score scraped-only keys as extras."""
    session = _paired_session(
        {'Conditions': '{"temperature": "300 K", "pressure": "1 atm", "phase": ""}'},
        {'Conditions': '{"temperature": "300 K", "pressure": "1 atm", "strain": "2%"}'},
        fields=('Material', 'Conditions'), structure_fields={'Conditions': 'dict'},
    )
    review = {'values': {'/temperature': {'status': 'correct'},
                         **extra_key.get('values', {})},
              'extraKeys': extra_key.get('extraKeys', {})}
    result = score_review(session, {'parts': {'v:2': {'Conditions': review}}})
    conditions = result['fields']['Conditions']
    assert _counts(conditions) == expected
    assert _counts(conditions['parts']['temperature']) == _expect(tp=1)
    assert _counts(conditions['parts']['pressure']) == _expect(pending=1)
    assert 'phase' not in conditions['parts']


@pytest.mark.parametrize(('gold', 'scraped', 'review', 'expected'), [
    ('{"value": "1 eV"}', '1 eV',
     {'values': {'/value': {'status': 'missing'}, '': {'status': 'incorrect'}}},
     _expect(fp=1, fn=1)),
    ('plain text', '["x", "y"]',
     {'values': {'': {'status': 'missing'}},
      'lists': {'': {'extra': {'0': {'status': 'correct'}, '1': True}}}},
     _expect(tp=1, fp=1, fn=1)),
    ('[{"value": "1 eV"}]', '{"value": "1 eV"}', {}, _expect(pending=2)),
    ('["1 eV"]', '1 eV',
     {'values': {'': {'status': 'correct'}},
      'lists': {'': {'matches': {'0': {'status': 'missing', 'scraped': None}}}}},
     _expect(tp=1, fn=1)),
])
def test_mismatched_shapes_score_gold_as_missing_and_scraped_as_extra(
    gold: str, scraped: str, review: dict[str, Any], expected: dict[str, int],
) -> None:
    """Never pair a dictionary, list, and scalar with each other."""
    session = _paired_session({'Gaps': gold}, {'Gaps': scraped},
                              fields=('Material', 'Gaps'), structure_fields={'Gaps': 'list'})
    result = score_review(session, {'parts': {'v:2': {'Gaps': review}}})
    assert _counts(result['fields']['Gaps']) == expected
