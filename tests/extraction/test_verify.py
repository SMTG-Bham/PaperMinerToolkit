"""Test recipe checks that drop or edit extracted items."""

from __future__ import annotations

import json
from typing import Any

import pytest

from paperminertoolkit.extraction import verify
from paperminertoolkit.extraction.recipes import _validate_recipe

SOURCE = """The indirect band gap of LaWN3 is 2.04 eV (HSE06), and the optical gap measured by
UV-vis is 2.3 eV. The work function is 4.5 eV. Its Fermi level lies 0.3 eV below the CBM."""


def recipe(**checks: Any) -> dict[str, Any]:
    """Build a band-gap recipe with the given checks."""
    return {
        'record definition': {
            'subject': 'materials', 'singular': 'material', 'plural': 'materials', 'unit': 'a material',
            'identity fields': ['Material'],
        },
        'search fields': {
            'Material': {'prompt': 'Material.', 'example': '<material>'},
            'Gaps': {'prompt': 'Gaps.', 'example': [{'value': '<number> eV'}]},
            'Summary': {'prompt': 'Summary.', 'example': '<number> eV'},
        },
        'checks': checks,
    }


def gap(value: str, method: str = 'None', gap_type: str = 'None') -> dict[str, str]:
    """Build one band-gap item."""
    return {'value': value, 'method_or_source': method, 'gap_type': gap_type}


ALL_CHECKS = {
    'values in source': True,
    'unit': 'eV',
    'exclude terms': ['work function', 'Fermi level'],
    'unsupported terms': {'SOC': ['spin-orbit']},
    'drop empty': True,
}


def test_records_pass_unchanged_without_checks() -> None:
    """Return copies of the records when the recipe declares no checks."""
    records = [{'Material': 'LaWN3', 'Gaps': [gap('9.99 eV')]}]
    data = recipe()
    del data['checks']

    kept, dropped = verify.verify_records(records, data, SOURCE)

    assert kept == records and kept[0] is not records[0]
    assert dropped == []


def test_each_check_drops_or_edits_what_it_should() -> None:
    """Drop invented, excluded, wrong-unit and empty items and remove unsupported SOC."""
    records = [{
        'Material': 'LaWN3',
        'Gaps': [
            gap('2.04 eV', 'HSE06+SOC', 'indirect'),
            gap('0.92 eV', 'HSE06'),
            gap('4.5 eV', 'work function (UPS)'),
            gap('0.3 eV', 'DFT', 'Fermi level offset'),
            gap('3.64 V'),
            gap('None', 'PBE'),
            gap('2.3', 'UV-vis; with SOC'),
            gap('direct gap', 'PBE'),
        ],
        'Summary': '2.04 eV',
    }]

    kept, dropped = verify.verify_records(records, recipe(**ALL_CHECKS), SOURCE, 'paper-1')

    assert kept == [{
        'Material': 'LaWN3',
        'Gaps': [gap('2.04 eV', 'HSE06', 'indirect'), gap('2.3', 'UV-vis'), gap('direct gap', 'PBE')],
        'Summary': '2.04 eV',
    }]
    assert [(row['Check'], row['Action']) for row in dropped] == [
        ('unsupported terms', 'edited'),
        ('values in source', 'dropped'),
        ('exclude terms', 'dropped'),
        ('exclude terms', 'dropped'),
        ('unit', 'dropped'),
        ('drop empty', 'dropped'),
        ('unsupported terms', 'edited'),
    ]
    assert {row['Paper id'] for row in dropped} == {'paper-1'}
    assert json.loads(dropped[0]['Detail']) == gap('2.04 eV', 'HSE06', 'indirect')
    assert dropped[2]['Detail'] == 'mentions "work function"'


def test_records_left_empty_are_dropped_and_scalar_fields_reset() -> None:
    """Drop a record whose checked fields are all removed, using its own paper id."""
    records = [
        {'Material': 'Ghost', 'Gaps': [gap('0.92 eV')], 'Summary': '1.26 eV', 'Paper id': 'paper-2'},
        {'Material': 'Kept', 'Gaps': [gap('2.04 eV')], 'Summary': '1.26 eV'},
        {'Material': 'Plain', 'Gaps': ['2.3 eV', '7.7 eV']},
    ]

    kept, dropped = verify.verify_records(records, recipe(**ALL_CHECKS), SOURCE)

    assert kept == [
        {'Material': 'Kept', 'Gaps': [gap('2.04 eV')], 'Summary': 'None'},
        {'Material': 'Plain', 'Gaps': ['2.3 eV']},
    ]
    assert dropped[2] == {
        'Paper id': 'paper-2', 'Record': 'Ghost', 'Field': '', 'Item': '', 'Check': 'drop empty',
        'Action': 'dropped', 'Detail': 'record has no reported fields',
    }
    assert dropped[1]['Item'] == '1.26 eV'


def test_supported_terms_and_unchecked_fields_are_left_alone() -> None:
    """Keep SOC when the source mentions spin-orbit coupling, and skip other fields."""
    records = [{'Material': 'X', 'Gaps': [gap('9.9 eV', 'PBE+SOC'), 'None'], 'Summary': '7.7 eV'}]
    checks = {'fields': ['Summary'], 'unsupported terms': {'SOC': ['spin-orbit']}, 'values in source': True}

    kept, dropped = verify.verify_records(records, recipe(**checks), 'Spin–orbit coupling was included.')

    assert kept == [{'Material': 'X', 'Gaps': [gap('9.9 eV', 'PBE+SOC'), 'None'], 'Summary': 'None'}]
    assert [row['Field'] for row in dropped] == ['Summary']

    kept, dropped = verify.verify_records(
        [{'Material': 'X', 'Gaps': [gap('9.9 eV', 'PBE+SOC')]}],
        recipe(**{'unsupported terms': {'SOC': ['spin-orbit']}}),
        'Spin–orbit coupling was included.',
    )
    assert kept[0]['Gaps'] == [gap('9.9 eV', 'PBE+SOC')] and dropped == []


def test_remove_term_handles_separators_and_whole_strings() -> None:
    """Strip a term with its separator, leaving 'None' when nothing remains."""
    assert verify._remove_term('DFT; HSE06; SOC', 'SOC') == 'DFT; HSE06'
    assert verify._remove_term('PBE + soc', 'SOC') == 'PBE'
    assert verify._remove_term('SOC', 'SOC') == 'None'
    assert verify._remove_term('SOCS', 'SOC') == 'SOCS'


@pytest.mark.parametrize(('checks', 'message'), [
    ([], 'must be a JSON object'),
    ({'bogus': True}, 'unknown key'),
    ({'unit': 3}, 'wrong type'),
    ({'tolerance': True}, 'wrong type'),
    ({'fields': ['']}, 'list of strings'),
    ({'fields': ['Nope']}, 'unknown field'),
    ({'unsupported terms': {'SOC': 'spin-orbit'}}, 'lists of phrases'),
    ({'unsupported terms': {' ': []}}, 'lists of phrases'),
])
def test_recipe_validation_rejects_malformed_checks(checks: object, message: str) -> None:
    """Report malformed check blocks when a recipe is loaded."""
    data = recipe()
    data['checks'] = checks
    with pytest.raises(ValueError, match=message):
        _validate_recipe(data, 'test')


def test_recipe_validation_accepts_well_formed_checks() -> None:
    """Accept every supported check key."""
    data = recipe(**ALL_CHECKS, fields=['Gaps'], tolerance=0.01)
    data['checks']['value key'] = 'value'
    assert _validate_recipe(data, 'test')['checks']['tolerance'] == 0.01
