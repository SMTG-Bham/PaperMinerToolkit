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


def test_exclude_terms_keys_limit_which_parts_are_searched() -> None:
    """Keep a gap whose method mentions an excluded quantity only as context."""
    records = [{'Material': 'X', 'Gaps': [
        gap('2.04 eV', 'HSE06 with exciton binding energy correction', 'optical'),
        gap('4.5 eV', 'UPS', 'work function'),
    ]}]
    checks = {'exclude terms': ['exciton binding energy', 'work function'], 'exclude terms keys': ['value', 'gap_type']}

    kept, dropped = verify.verify_records(records, recipe(**checks), SOURCE)

    assert kept[0]['Gaps'] == [gap('2.04 eV', 'HSE06 with exciton binding energy correction', 'optical')]
    assert [row['Detail'] for row in dropped] == ['mentions "work function"']


def test_unit_accepts_a_list_of_units() -> None:
    """Keep values in any listed unit and name the units when dropping."""
    records = [{'Material': 'InSb', 'Gaps': [gap('231 meV'), gap('2.04 eV'), gap('185 K'), gap('2.04 to 2.3 eV')]}]
    source = 'InSb has a gap of 231 meV; 2.04 eV; Tc = 185 K; 2.3 eV.'

    kept, dropped = verify.verify_records(records, recipe(unit=['eV', 'meV']), source)

    assert kept[0]['Gaps'] == [gap('231 meV'), gap('2.04 eV'), gap('2.04 to 2.3 eV')]
    assert [row['Detail'] for row in dropped] == ['not in eV or meV']


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
    ({'unit': ['eV', '']}, 'list of strings'),
    ({'tolerance': True}, 'wrong type'),
    ({'fields': ['']}, 'list of strings'),
    ({'fields': ['Nope']}, 'unknown field'),
    ({'unsupported terms': {'SOC': 'spin-orbit'}}, 'lists of phrases'),
    ({'unsupported terms': {' ': []}}, 'lists of phrases'),
    ({'paper question': {'question': ' ', 'if no': {'move': {'Gaps': None}}}}, 'non-empty "question"'),
    ({'paper question': {'question': 'Q?', 'if no': 'move'}}, '"if no" must be'),
    ({'paper question': {'question': 'Q?', 'if no': {'move': {}}}}, '"if no" must be'),
    ({'paper question': {'question': 'Q?', 'if no': {'move': {'Gaps': 'Gaps'}}}}, 'unknown or identical'),
    ({'paper question': {'question': 'Q?', 'if no': {'move': {'Nope': None}}}}, 'unknown or identical'),
    ({'local words': {'key': 'gap_type', 'words': []}}, 'non-empty list of "words"'),
    ({'local words': {'key': ' ', 'words': ['direct']}}, 'non-empty list of "words"'),
    ({'local words': {'key': 'gap_type', 'words': ['direct'], 'bogus': 1}}, 'non-empty list of "words"'),
    ({'local words': {'key': 'gap_type', 'words': ['direct'], 'window': 0}}, 'positive number'),
    ({'local words': {'key': 'gap_type', 'words': ['direct'], 'window': True}}, 'positive number'),
    ({'local words': {'key': 'gap_type', 'words': ['direct'], 'fields': ['Nope']}}, 'known fields'),
    ({'local words': {'key': 'gap_type', 'words': ['direct'], 'fields': 'Gaps'}}, 'known fields'),
])
def test_recipe_validation_rejects_malformed_checks(checks: object, message: str) -> None:
    """Report malformed check blocks when a recipe is loaded."""
    data = recipe()
    data['checks'] = checks
    with pytest.raises(ValueError, match=message):
        _validate_recipe(data, 'test')


def test_recipe_validation_accepts_well_formed_checks() -> None:
    """Accept every supported check key."""
    data = recipe(**ALL_CHECKS, fields=['Gaps'], tolerance=0.01,
                  **{'paper question': {'question': 'Q?', 'if no': {'move': {'Gaps': 'Summary', 'Summary': None}}},
                     'local words': {'key': 'gap_type', 'words': ['direct'], 'fields': ['Gaps'], 'window': 50}})
    data['checks']['value key'] = 'value'
    assert _validate_recipe(data, 'test')['checks']['tolerance'] == 0.01


def paper_recipe() -> dict[str, Any]:
    """Build a recipe whose paper question moves gaps to a cited field and clears the summary."""
    data = recipe(**{'paper question': {'question': 'Own gaps?', 'if no': {'move': {'Gaps': 'Cited', 'Summary': None}}}})
    data['search fields']['Cited'] = {'prompt': 'Cited gaps.', 'example': [{'value': '<number> eV'}]}
    return data


def test_apply_paper_answer_moves_and_clears_fields_on_no() -> None:
    """Move listed items to their target field, merging duplicates, and audit each one."""
    records = [
        {'Material': 'X', 'Gaps': [gap('2.0 eV', 'PBE')], 'Summary': '2.0 eV', 'Cited': [gap('2.0 eV')]},
        {'Material': 'Y', 'Gaps': 'None', 'Summary': 'None'},
    ]

    kept, audit = verify.apply_paper_answer(records, paper_recipe(), False, 'paper-1')

    assert kept == [
        {'Material': 'X', 'Gaps': [], 'Summary': 'None', 'Cited': [gap('2.0 eV', 'PBE')]},
        {'Material': 'Y', 'Gaps': 'None', 'Summary': 'None'},
    ]
    assert [(row['Paper id'], row['Field'], row['Check'], row['Action'], row['Detail']) for row in audit] == [
        ('paper-1', 'Gaps', 'paper question', 'moved', 'to Cited'),
        ('paper-1', 'Summary', 'paper question', 'cleared', 'answered no'),
    ]
    assert records[0]['Gaps'] == [gap('2.0 eV', 'PBE')]


def test_apply_paper_answer_leaves_records_on_yes_unclear_or_no_question() -> None:
    """Change nothing unless the recipe asks a question and the answer is no."""
    records = [{'Material': 'X', 'Gaps': [gap('2.0 eV')], 'Paper id': 'paper-2'}]
    for data, answer in ((paper_recipe(), True), (paper_recipe(), None), (recipe(), False)):
        kept, audit = verify.apply_paper_answer(records, data, answer)
        assert kept == records and audit == []
    kept, audit = verify.apply_paper_answer(records, paper_recipe(), False)
    assert audit[0]['Paper id'] == 'paper-2' and kept[0]['Cited'] == [gap('2.0 eV')]


LOCAL_SOURCE = """The indirect gap of A is 1.10 eV.

Much later, B was found to have a gap of 2.50 eV, far from any other word about its nature.

Table 1 Gaps
| Material | Direct gap (eV) |
| --- | --- |
| C | 3.30 |"""


def test_local_words_are_kept_only_near_the_value() -> None:
    """Remove a gap type the paper does not print near the value, in the listed fields only."""
    checks = {'local words': {'key': 'gap_type', 'words': ['direct', 'indirect', 'optical'], 'fields': ['Gaps'], 'window': 40}}
    records = [{
        'Material': 'X',
        'Gaps': [gap('1.10 eV', 'PBE', 'indirect optical'), gap('2.50 eV', 'PBE', 'direct'), gap('3.30 eV', 'PBE', 'direct'),
                 gap('9.99 eV', 'PBE', 'direct'), gap('n/a', 'PBE', 'direct'), {'value': '2.50 eV', 'gap_type': 3}],
        'Summary': '2.50 eV',
    }]

    kept, audit = verify.verify_records(records, recipe(**checks), LOCAL_SOURCE, 'paper-1')

    assert [item['gap_type'] if isinstance(item, dict) else item for item in kept[0]['Gaps']] == [
        'indirect', 'None', 'direct', 'direct', 'direct', 3,
    ]
    assert [(row['Check'], row['Action']) for row in audit] == [('local words', 'edited'), ('local words', 'edited')]
    assert json.loads(audit[1]['Detail'])['gap_type'] == 'None'

    kept, audit = verify.verify_records(
        [{'Material': 'X', 'Gaps': [gap('2.50 eV', 'PBE', 'direct')]}],
        recipe(**{'local words': {'key': 'gap_type', 'words': ['direct'], 'fields': ['Summary']}}), LOCAL_SOURCE,
    )
    assert kept[0]['Gaps'][0]['gap_type'] == 'direct' and audit == []
