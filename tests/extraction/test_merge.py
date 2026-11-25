"""Test value comparison and deterministic merging of chunk records."""

from __future__ import annotations

import math

import pytest

from paperminertoolkit.extraction import merge, values

RECIPE = {
    'record definition': {'identity fields': ['Material system']},
    'search fields': {
        'Material system': {'prompt': 'Material.', 'example': '<material>'},
        'Band gap': {'prompt': 'Gaps.', 'example': [{'value': '<number> eV'}]},
        'Notes': {'prompt': 'Notes.', 'example': '<notes>'},
    },
}


TASK_RECIPE = {
    'record definition': {'identity fields': ['Group', 'Time point']},
    'search fields': {
        'Group': {'prompt': 'Group label.', 'example': 'Group A'},
        'Time point': {'prompt': 'Time of assessment.', 'example': '1 h'},
        'Completion': {'prompt': 'Percentage that completed the task.', 'example': '0%'},
    },
}


def gap(value: str, method: str = 'None', gap_type: str = 'None') -> dict[str, str]:
    """Build one band-gap item."""
    return {'value': value, 'method_or_source': method, 'gap_type': gap_type, 'conditions': 'None'}


def assessment(group: str, time_point: str, completion: str) -> dict[str, str]:
    """Build one task-completion record."""
    return {'Group': group, 'Time point': time_point, 'Completion': completion}


@pytest.mark.parametrize('value', [None, math.nan, '', ' None ', 'nan', 'N/A', [], {}, '[]'])
def test_is_missing_recognises_empty_values(value: object) -> None:
    """Treat empty containers, NaN and missing-value tokens as missing."""
    assert values.is_missing(value)


def test_numbers_and_tolerances() -> None:
    """Read unsigned numbers and compare them within a relative tolerance."""
    assert values.numbers('1.2-1.4 eV') == [1.2, 1.4]
    assert values.numbers('2·04 eV') == [2.04]
    assert values.close(1.77, 1.8)
    assert not values.close(1.5, 1.6)
    assert values.printed_in('2.04 eV', [2.04, 3.0])
    assert not values.printed_in('0.92 eV', [2.04, 3.0])
    assert not values.printed_in('n/a', [1.0])


def test_same_value_compares_numbers_units_and_words() -> None:
    """Match values by numbers and remaining words, else by normalized text."""
    assert values.same_value('1.50 eV', '1.5 eV')
    assert not values.same_value('1.5 eV', '1.5 meV')
    assert not values.same_value('1.5 eV', '1.5-1.6 eV')
    assert not values.same_value('1.5 eV', '1.7 eV')
    assert values.same_value('HSE06, PBE', 'pbe; hse06')
    assert values.text_key('Cs2Sn(I0.88Br0.12)6') == values.text_key('cs2sn (i0.88br0.12) 6')


def test_identity_fields_fall_back_to_the_first_search_field() -> None:
    """Use the first search field when a recipe declares no identity fields."""
    assert merge.identity_fields(RECIPE) == ['Material system']
    assert merge.identity_fields({'search fields': {'Name': {}, 'Value': {}}}) == ['Name']


def test_record_key_requires_every_identity_field() -> None:
    """Return no key when an identity field is missing."""
    assert merge.record_key({'A': 'x', 'B': 'y'}, ['A', 'B']) == 'x | y'
    assert merge.record_key({'A': 'x', 'B': 'None'}, ['A', 'B']) == ''


def test_same_item_lets_missing_parts_match_anything() -> None:
    """Treat items as duplicates when every part present on both sides agrees."""
    assert merge.same_item(gap('1.5 eV'), gap('1.50 eV', 'PBE'))
    assert not merge.same_item(gap('1.5 eV', 'HSE06'), gap('1.5 eV', 'PBE'))
    assert not merge.same_item({'value': 'None'}, {'value': 'None'})
    assert not merge.same_item(gap('1.5 eV'), '1.5 eV')
    assert merge.same_item('1.5 eV', '1.50 eV')


def test_merge_items_keeps_the_fullest_duplicate_in_first_position() -> None:
    """Drop missing and duplicate items, preferring the more complete copy."""
    items = [gap('1.5 eV'), 'None', gap('2.0 eV', 'PBE'), gap('1.50 eV', 'HSE06', 'direct'), gap('2.0 eV')]

    assert merge.merge_items(items) == [gap('1.50 eV', 'HSE06', 'direct'), gap('2.0 eV', 'PBE')]
    assert merge.merge_items(['1.5 eV', '1.50 eV', 'direct', 'Direct']) == ['1.5 eV', 'direct']


def test_merge_chunk_records_groups_by_identity_and_keeps_first_scalars() -> None:
    """Merge records of the same material from different chunks."""
    records = [
        {'Material system': 'ZrTaN3 (A)', 'Band gap': [gap('0.45 eV', 'HSE06')], 'Notes': 'None'},
        {'Material system': 'LaWN3', 'Band gap': [gap('2.04 eV')], 'Notes': 'first'},
        {'Material system': '(A) zrtan3', 'Band gap': [gap('0.45 eV'), gap('1.06 eV')], 'Notes': 'table'},
        {'Material system': 'LaWN3', 'Band gap': 'None', 'Notes': 'second'},
        {'Material system': 'LaWN3', 'Band gap': gap('2.28 eV')},
        {'Material system': 'None', 'Band gap': [gap('3.0 eV'), gap('3.0 eV')]},
        {'Material system': 'None', 'Band gap': [gap('3.0 eV')]},
    ]

    merged = merge.merge_chunk_records(records, RECIPE)

    assert merged == [
        {'Material system': 'ZrTaN3 (A)', 'Band gap': [gap('0.45 eV', 'HSE06'), gap('1.06 eV')], 'Notes': 'table'},
        {'Material system': 'LaWN3', 'Band gap': [gap('2.04 eV'), gap('2.28 eV')], 'Notes': 'first'},
        {'Material system': 'None', 'Band gap': [gap('3.0 eV')]},
        {'Material system': 'None', 'Band gap': [gap('3.0 eV')]},
    ]
    assert records[0]['Band gap'] == [gap('0.45 eV', 'HSE06')]


def test_merge_chunk_records_drops_repeats_under_qualified_names() -> None:
    """Drop a record that repeats an earlier one's values under a qualified name."""
    records = [
        {'Material system': 'Cs3Bi2I9 (powder)', 'Band gap': [gap('1.9-2.2 eV')], 'Notes': 'None'},
        {'Material system': 'Cs3Bi2I9 (thin film)', 'Band gap': [gap('1.9-2.2 eV', 'UV-vis')], 'Notes': 'None'},
        {'Material system': 'Cs3Bi2I9, Layered #2', 'Band gap': [gap('2.1 eV')]},
        {'Material system': 'Cs3Sb2I9', 'Band gap': [gap('1.9-2.2 eV')]},
        {'Material system': 'Cs3Sb2I9 (film)', 'Band gap': [gap('1.9-2.2 eV'), gap('2.4 eV')]},
        {'Material system': 'Cs3Sb2I9 (crystal)', 'Band gap': 'None'},
        {'Material system': 'Cs3Sb2I9 [nanocrystal]', 'Band gap': 'None'},
    ]

    merged = merge.merge_chunk_records(records, RECIPE)

    assert [record['Material system'] for record in merged] == [
        'Cs3Bi2I9 (powder)', 'Cs3Bi2I9, Layered #2', 'Cs3Sb2I9', 'Cs3Sb2I9 (film)',
        'Cs3Sb2I9 (crystal)', 'Cs3Sb2I9 [nanocrystal]',
    ]


@pytest.mark.parametrize(('first', 'second', 'repeat'), [
    ('Cs3Bi2I9', 'Cs3Bi2I9, layered', True),
    ('Cs3Bi2I9 (powder)', 'Cs3Bi2I9; film', True),
    ('Cs3Bi2I9 thin film', 'Cs3Bi2I9 powder', False),
    ('Cs2Sn(I0.88Br0.12)6', 'Cs2Sn(I0.5Br0.5)6', False),
    ('(CH3NH3)PbI3', '(CH3CH2NH3)PbI3', False),
    ('Ba(Zr, Ti)O3', 'Ba(Zr, Hf)O3', False),
    ('(powder)', '(film)', False),
])
def test_merge_chunk_records_strips_only_separate_qualifiers(first: str, second: str, repeat: bool) -> None:
    """Treat bracketed words and trailing clauses as qualifiers, but not brackets inside a name."""
    records = [{'Material system': name, 'Band gap': [gap('2.1 eV')]} for name in (first, second)]

    merged = merge.merge_chunk_records(records, RECIPE)

    assert [record['Material system'] for record in merged] == ([first] if repeat else [first, second])


@pytest.mark.parametrize('records', [
    [assessment('Group A', '1 h', '0%'), assessment('Group A', '1 day', '0%')],
    [assessment('Group A', '1 h', '0%'), assessment('Group B', '1 h', '0%')],
    [assessment('control', '1 h', '0%'), assessment('control', '1 day', '0%')],
    [assessment('Group A', '1 h', '0%'), assessment('Group A', '1 day', '40%')],
], ids=['same-group-other-time', 'other-group-same-time', 'generic-group-name', 'different-values'])
def test_merge_chunk_records_keeps_separate_observations_with_equal_values(records: list[dict[str, str]]) -> None:
    """Keep every record whose identity fields differ by more than a qualifier."""
    assert merge.merge_chunk_records(records, TASK_RECIPE) == records


def test_merge_chunk_records_drops_repeats_qualified_on_one_identity_field() -> None:
    """Drop a repeat only when one identity field gains a qualifier and the others match."""
    records = [
        assessment('Group A', '1 h', '0%'),
        assessment('Group A (repeat)', '1 h', '0%'),
        assessment('Group A (repeat)', '1 h (late)', '0%'),
        assessment('Group A', 'None', '0%'),
        assessment('(control)', '1 h', '0%'),
        assessment('(treated)', '1 h', '0%'),
    ]

    merged = merge.merge_chunk_records(records, TASK_RECIPE)

    assert merged == [records[0], *records[2:]]


def test_same_content_compares_every_given_field() -> None:
    """Require the same items in each field and at least one reported field."""
    first = {'Band gap': [gap('2.0 eV'), gap('2.4 eV', 'PBE')], 'Notes': 'x'}
    assert merge.same_content(first, {'Band gap': [gap('2.4 eV'), gap('2.00 eV')], 'Notes': 'X'}, ['Band gap', 'Notes'])
    assert not merge.same_content(first, {'Band gap': [gap('2.0 eV'), gap('2.5 eV')]}, ['Band gap'])
    assert not merge.same_content(first, {'Band gap': [gap('2.0 eV'), gap('2.0 eV')]}, ['Band gap'])
    assert not merge.same_content({'Band gap': 'None'}, {}, ['Band gap'])
