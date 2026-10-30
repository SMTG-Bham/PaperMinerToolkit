"""Exercise automatic recipe scoring, proposal merging, and the optimisation loop."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import paperminertoolkit.cli as cli
import paperminertoolkit.workflows.recipe_optimisation as optimisation
from paperminertoolkit.corpus import database as corpus
from paperminertoolkit.extraction.recipes import load_recipe


def _recipe() -> dict[str, Any]:
    """Return a small two-field recipe."""
    return {
        'record definition': {'subject': 'oxides', 'singular': 'material', 'plural': 'materials',
                              'unit': 'a distinct material', 'identity fields': ['Material']},
        'additional prompts': '',
        'search fields': {
            'Material': {'prompt': 'Material name', 'example': 'ZnO', 'aliases': ['Name']},
            'Band gap': {'prompt': 'Band gap', 'example': '3.3 eV', 'unit': 'eV'},
        },
    }


def _gold_rows() -> list[dict[str, Any]]:
    """Return validation rows in the reviewer's parsed layout."""
    return [
        {'id': 'v:2', 'values': {'Material': 'ZnO', 'Band gap': '3.30 eV'}},
        {'id': 'v:3', 'values': {'Material': 'TiO2', 'Band gap': '3.2 eV'}},
    ]


@pytest.mark.parametrize(('gold', 'scraped', 'expected'), [
    ('3.30 eV', '3.3 eV', True),
    ('3.30 eV', '3.31 eV', True),
    ('3.30 eV', '3.6 eV', False),
    ('3.3 eV', '3.3', True),
    ('Li7La3Zr2O12', 'li7la3zr2o12', True),
    ('anatase', 'rutile', False),
])
def test_values_match_normalizes_case_numbers_and_units(gold: str, scraped: str, expected: bool) -> None:
    """Treat formatting differences as agreement but different values as errors."""
    assert optimisation.values_match(gold, scraped) is expected


def test_score_paper_records_counts_paired_missing_and_extra_values() -> None:
    """Pair by identity, score each field, and report every kind of mismatch."""
    scraped = [
        {'Material': 'ZnO', 'Band gap': '3.3 eV'},
        {'Material': 'MgO', 'Band gap': '7.8 eV'},
    ]

    score = optimisation.score_paper_records(_gold_rows(), scraped, _recipe())

    assert score['counts']['Material'] == {'tp': 1, 'fp': 1, 'fn': 1, 'unreachable': 0}
    assert score['counts']['Band gap'] == {'tp': 1, 'fp': 1, 'fn': 1, 'unreachable': 0}
    assert (score['expected'], score['scraped'], score['paired']) == (2, 2, 1)
    assert sorted(item['kind'] for item in score['mismatches']) == ['extra record', 'missing record']


def test_score_paper_records_scores_structured_values_by_part() -> None:
    """Score list items and dictionary keys independently like the reviewer."""
    gold = [{'id': 'v:2', 'values': {
        'Material': 'ZnO',
        'Band gap': json.dumps([{'value': '3.3 eV', 'method': 'UV-vis'}, {'value': '3.4 eV', 'method': 'DFT'}]),
    }}]
    scraped = [{'Material': 'ZnO', 'Band gap': [{'value': '3.3 eV', 'method': 'UV-vis'}]}]

    score = optimisation.score_paper_records(gold, scraped, _recipe())

    assert score['counts']['Band gap'] == {'tp': 2, 'fp': 0, 'fn': 2, 'unreachable': 0}
    assert score['mismatches'][0]['field'] == 'Band gap'


def test_merge_recipe_keeps_schema_and_storage_settings() -> None:
    """Accept wording changes while restoring aliases, units, and identity fields."""
    original = _recipe()
    proposed = json.loads(json.dumps(original))
    proposed['additional prompts'] = 'Report every polymorph separately.'
    proposed['record definition']['identity fields'] = []
    proposed['search fields']['Material']['prompt'] = 'Reported material name or formula'
    proposed['search fields']['Material']['aliases'] = []
    del proposed['search fields']['Band gap']['unit']

    merged = optimisation._merge_recipe(original, proposed)

    assert merged['additional prompts'] == 'Report every polymorph separately.'
    assert merged['search fields']['Material']['prompt'] == 'Reported material name or formula'
    assert merged['search fields']['Material']['aliases'] == ['Name']
    assert merged['search fields']['Band gap']['unit'] == 'eV'
    assert merged['record definition']['identity fields'] == ['Material']


def test_merge_recipe_rejects_changed_fields() -> None:
    """Refuse proposals that would no longer line up with the validation CSV."""
    proposed = _recipe()
    proposed['search fields']['Phase'] = {'prompt': 'Phase', 'example': 'anatase'}

    with pytest.raises(ValueError, match="'Phase'"):
        optimisation._merge_recipe(_recipe(), proposed)


def test_parse_proposal_accepts_fenced_and_bare_recipes() -> None:
    """Read the wrapped proposal format and a bare recipe object."""
    wrapped = '```json\n' + json.dumps({'rationale': 'tighter', 'recipe': _recipe()}) + '\n```'

    assert optimisation._parse_proposal(wrapped) == (_recipe(), 'tighter')
    assert optimisation._parse_proposal('Here: ' + json.dumps(_recipe())) == (_recipe(), '')
    with pytest.raises(ValueError):
        optimisation._parse_proposal('no json here')


def _corpus(tmp_path: Path) -> Path:
    """Build a corpus with two papers that have downloaded text."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        for paper_id, doi, text in [('demo:1', '10.1/a', 'ZnO has a gap of 3.3 eV.'),
                                    ('demo:2', '10.1/b', 'TiO2 has a gap of 3.2 eV.')]:
            paper = {'paper_id': paper_id, 'doi': doi, 'title': f'Paper {paper_id}', 'sources': 'demo'}
            corpus.upsert_paper(conn, paper)
            corpus.add_asset(conn, paper, text, role='text', kind='text', mime_type='text/plain', source='demo')
    return db_path


def _validation(tmp_path: Path) -> Path:
    """Write a validation CSV for both corpus papers and one absent paper."""
    path = tmp_path / 'validation.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Identifier', 'Title', 'DOI', 'Material', 'Band gap'])
        writer.writerow(['', 'Paper demo:1', '10.1/a', 'ZnO', '3.3 eV'])
        writer.writerow(['', 'Paper demo:2', '10.1/b', 'TiO2', '3.2 eV'])
        writer.writerow(['', 'Missing paper', '10.1/missing', 'MgO', '7.8 eV'])
    return path


def test_optimise_recipe_keeps_only_improving_revisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrape, score, revise, and keep a revision only when the score rises."""
    recipe_path = tmp_path / 'recipe.json'
    recipe_path.write_text(json.dumps(_recipe()), encoding='utf-8')
    proposals = iter(['worse', 'better', 'same'])
    requests: list[str] = []

    def fake_scrape(text: str, recipe: dict[str, Any], model_config: Any = None) -> list[dict[str, Any]]:
        """Return records whose quality depends on the recipe's additional prompts."""
        material = text.split()[0]
        gap = text.split('of ')[1].rstrip('.')
        hint = recipe['additional prompts']
        if hint == 'better':
            return [{'Material': material, 'Band gap': gap}]
        if hint == 'worse':
            return []
        return [{'Material': material, 'Band gap': 'None'}]

    def fake_query(messages: list[dict[str, Any]], config: Any = None, max_output_tokens: int = 0) -> str:
        """Record the optimiser request and return a canned proposal."""
        requests.append(messages[1]['content'])
        proposal = load_recipe(str(recipe_path))
        proposal['additional prompts'] = next(proposals)
        return json.dumps({'rationale': proposal['additional prompts'], 'recipe': proposal})

    monkeypatch.setattr(optimisation, 'scrape_text', fake_scrape)
    monkeypatch.setattr(optimisation, 'query_text', fake_query)
    messages: list[str] = []

    summary = optimisation.optimise_recipe(_corpus(tmp_path), str(recipe_path), _validation(tmp_path),
                                           output_dir=tmp_path / 'out', iterations=3, patience=2,
                                           echo=messages.append)

    output = tmp_path / 'out'
    history = json.loads((output / 'history.json').read_text(encoding='utf-8'))
    assert summary['papers'] == 2
    assert summary['baseline_score'] == pytest.approx(2 / 3)
    assert summary['best_score'] == 1.0
    assert [attempt['accepted'] for attempt in history] == [True, False, True]
    assert json.loads((output / 'best_recipe.json').read_text(encoding='utf-8'))['additional prompts'] == 'better'
    assert '"mismatches"' in requests[0]
    assert any('Missing paper' in message or '10.1/missing' in message for message in messages)
    with (output / 'iteration_02' / 'scraped.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [row['Material'] for row in rows] == ['ZnO', 'TiO2']


def test_optimise_recipe_selects_on_holdout_papers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide held-out papers from the optimiser's mismatch examples."""
    recipe_path = tmp_path / 'recipe.json'
    recipe_path.write_text(json.dumps(_recipe()), encoding='utf-8')
    seen: list[dict[str, Any]] = []

    monkeypatch.setattr(optimisation, 'scrape_text', lambda text, recipe, model_config=None: [])

    def fake_query(messages: list[dict[str, Any]], config: Any = None, max_output_tokens: int = 0) -> str:
        """Record the optimiser request and return a canned proposal."""
        seen.append(json.loads(messages[1]['content']))
        return json.dumps({'recipe': _recipe()})

    monkeypatch.setattr(optimisation, 'query_text', fake_query)

    summary = optimisation.optimise_recipe(_corpus(tmp_path), str(recipe_path), _validation(tmp_path),
                                           output_dir=tmp_path / 'out', iterations=1, holdout=0.5,
                                           echo=lambda message: None)

    assert summary['holdout_papers'] == 1
    assert len({item['paper'] for item in seen[0]['mismatches']}) == 1
    score = json.loads((tmp_path / 'out' / 'iteration_00' / 'score.json').read_text(encoding='utf-8'))
    assert score['holdout']['papers'] == 1 and score['train']['papers'] == 1


def test_optimise_recipe_rejects_invalid_options(tmp_path: Path) -> None:
    """Validate options before reading inputs or calling a model."""
    with pytest.raises(ValueError, match='mode'):
        optimisation.optimise_recipe(tmp_path / 'db', 'sse', tmp_path / 'v.csv', mode='images')
    with pytest.raises(ValueError, match='holdout'):
        optimisation.optimise_recipe(tmp_path / 'db', 'sse', tmp_path / 'v.csv', holdout=1.0)


def test_recipe_optimise_cli_passes_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward CLI options and report workflow errors as usage-friendly failures."""
    db_path = tmp_path / 'papers.db'
    db_path.write_bytes(b'')
    validation = tmp_path / 'validation.csv'
    validation.write_text('DOI\n', encoding='utf-8')
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(cli, 'optimise_recipe', lambda *args, **kwargs: calls.append((args, kwargs)))

    result = CliRunner().invoke(cli.main, ['recipe', 'optimise', str(db_path), 'sse', str(validation),
                                           '--iterations', '3', '--holdout', '0.25',
                                           '--optimiser-model', 'big-model'])

    assert result.exit_code == 0, result.output
    args, kwargs = calls[0]
    assert args == (str(db_path), 'sse', str(validation))
    assert kwargs['iterations'] == 3
    assert kwargs['holdout'] == 0.25
    assert kwargs['optimiser_model'] == 'big-model'

    def fail(*args: Any, **kwargs: Any) -> None:
        """Simulate a workflow error."""
        raise ValueError('no papers')

    monkeypatch.setattr(cli, 'optimise_recipe', fail)
    failed = CliRunner().invoke(cli.main, ['recipe', 'optimise', str(db_path), 'sse', str(validation)])
    assert failed.exit_code != 0
    assert 'no papers' in failed.output


def test_optimise_recipe_stops_when_every_scrape_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Abort instead of optimising against a round with no successful scrapes."""
    def broken_scrape(text: str, recipe: dict[str, Any], model_config: Any = None) -> list[dict[str, Any]]:
        """Fail like a provider with missing credentials."""
        raise RuntimeError('Missing credentials')

    def unexpected_query(*args: Any, **kwargs: Any) -> str:
        """Fail the test if the optimiser is consulted."""
        raise AssertionError('optimiser should not be called')

    monkeypatch.setattr(optimisation, 'scrape_text', broken_scrape)
    monkeypatch.setattr(optimisation, 'query_text', unexpected_query)

    with pytest.raises(RuntimeError, match='Every paper failed.*Missing credentials'):
        optimisation.optimise_recipe(_corpus(tmp_path), _write_recipe(tmp_path),
                                     _validation(tmp_path), output_dir=tmp_path / 'out', echo=lambda message: None)
    assert (tmp_path / 'out' / 'iteration_00' / 'score.json').is_file()


def test_optimise_recipe_stops_when_the_optimiser_request_fails(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat provider failures as fatal rather than as rejected proposals."""
    def failing_query(*args: Any, **kwargs: Any) -> str:
        """Fail like an unreachable optimiser model."""
        raise ConnectionError('offline')

    monkeypatch.setattr(optimisation, 'scrape_text', lambda text, recipe, model_config=None: [])
    monkeypatch.setattr(optimisation, 'query_text', failing_query)

    with pytest.raises(RuntimeError, match='Optimiser model request failed: offline'):
        optimisation.optimise_recipe(_corpus(tmp_path), _write_recipe(tmp_path), _validation(tmp_path),
                                     output_dir=tmp_path / 'out', echo=lambda message: None)


def _write_recipe(tmp_path: Path) -> str:
    """Save the test recipe as a standalone recipe file."""
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(_recipe()), encoding='utf-8')
    return str(path)


def test_merge_recipe_keeps_fields_the_optimiser_omitted() -> None:
    """Fill fields left out of a proposal from the recipe being revised."""
    proposed = {'search fields': {'Band gap': {'prompt': 'Principal band gap', 'example': '3 eV'}}}

    merged = optimisation._merge_recipe(_recipe(), proposed)

    assert list(merged['search fields']) == ['Material', 'Band gap']
    assert merged['search fields']['Material'] == _recipe()['search fields']['Material']
    assert merged['search fields']['Band gap']['prompt'] == 'Principal band gap'


def test_score_paper_records_pairs_identities_regardless_of_word_order() -> None:
    """Pair 'Monoclinic C2/c Sb2O5' with 'Sb2O5 (monoclinic C2/c)'."""
    gold = [{'id': 'v:2', 'values': {'Material': 'Sb2O5 (monoclinic C2/c)', 'Band gap': '3.3 eV'}}]
    scraped = [{'Material': 'Monoclinic C2/c Sb2O5', 'Band gap': '3.3 eV'}]

    score = optimisation.score_paper_records(gold, scraped, _recipe())

    assert score['paired'] == 1
    assert score['counts']['Band gap'] == {'tp': 1, 'fp': 0, 'fn': 0, 'unreachable': 0}


def test_score_paper_records_separates_values_missing_from_the_text() -> None:
    """Count expected items whose numbers the text never prints as unreachable, not missed."""
    gold = [{'id': 'v:2', 'values': {
        'Material': 'ZnO',
        'Band gap': json.dumps([{'value': '3.30 eV', 'method': 'UV-vis'}, {'value': '3.41 eV', 'method': 'DFT'}]),
    }}]
    scraped = [{'Material': 'ZnO', 'Band gap': [{'value': '3.3 eV', 'method': 'UV-vis'}]}]
    text = 'Optical measurements give a 3.3 eV gap for ZnO; computed values are tabulated in the SI.'

    score = optimisation.score_paper_records(gold, scraped, _recipe(), source_text=text)

    assert score['counts']['Band gap'] == {'tp': 2, 'fp': 0, 'fn': 0, 'unreachable': 2}
    assert not [item for item in score['mismatches'] if item.get('field') == 'Band gap']
    metrics = optimisation._metrics(score['counts']['Band gap'])
    assert metrics['f1'] == 1.0
    assert metrics['f1_including_unreachable'] == pytest.approx(2 / 3)


def test_score_paper_records_flags_unreachable_parts_of_real_mismatches() -> None:
    """Tell the optimiser which expected parts it should not try to recover."""
    gold = [{'id': 'v:2', 'values': {'Material': 'ZnO', 'Band gap': json.dumps(['3.30 eV', '3.41 eV'])}}]
    scraped = [{'Material': 'ZnO', 'Band gap': ['3.9 eV']}]

    score = optimisation.score_paper_records(gold, scraped, _recipe(), source_text='ZnO: 3.30 eV.')

    assert score['counts']['Band gap'] == {'tp': 0, 'fp': 1, 'fn': 1, 'unreachable': 1}
    mismatch = next(item for item in score['mismatches'] if item.get('field') == 'Band gap')
    assert mismatch['not_in_text'] == ['3.41 eV']


def test_optimise_recipe_can_count_unreachable_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exclude values absent from the text by default and count them on request."""
    validation = tmp_path / 'validation.csv'
    with validation.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Identifier', 'Title', 'DOI', 'Material', 'Band gap'])
        writer.writerow(['', 'Paper demo:1', '10.1/a', 'ZnO', '["3.3 eV", "9.87 eV"]'])
    monkeypatch.setattr(optimisation, 'scrape_text',
                        lambda text, recipe, model_config=None: [{'Material': 'ZnO', 'Band gap': '3.3 eV'}])
    messages: list[str] = []

    excluded = optimisation.optimise_recipe(_corpus(tmp_path), _write_recipe(tmp_path), validation,
                                            output_dir=tmp_path / 'a', iterations=0, echo=messages.append)
    counted = optimisation.optimise_recipe(_corpus(tmp_path), _write_recipe(tmp_path), validation,
                                           output_dir=tmp_path / 'b', iterations=0, include_unreachable=True,
                                           echo=messages.append)

    assert excluded['best_score'] == 1.0
    assert counted['best_score'] == pytest.approx(0.8)
    assert any('1 of 3 expected values' in message and 'excluded' in message for message in messages)


@pytest.mark.parametrize(('text', 'reachable'), [
    ('a gap of 2·70 eV', True),
    ('a gap of 2,70 eV', True),
    ('a gap of 2.7 eV', True),
    ('a gap of 2.73 eV', True),
    ('a gap of 2.9 eV', False),
])
def test_gold_parts_accept_alternative_decimals_and_rounding(text: str, reachable: bool) -> None:
    """Treat text the scorer could match as reachable, including rounded and middle-dot decimals."""
    parts = optimisation._gold_parts({'value': '2.70 eV', 'method': 'UV-vis'},
                                     optimisation._text_numbers(text), tolerance=0.02)

    assert parts == [('value=2.70 eV', reachable), ('method=UV-vis', reachable)]
