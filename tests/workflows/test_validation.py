"""Exercise local validation input matching, persistence, and CLI wiring."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import csv
import http.client
import io
import json
import os
from pathlib import Path
import queue
import re
import threading
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from click.testing import CliRunner

import paperminertoolkit.cli as cli
import paperminertoolkit.workflows.validation as validation_workflow
from paperminertoolkit.workflows.validation import ReviewApp
from paperminertoolkit.workflows.validation_scoring import RUN_FIELDS, score_review


@pytest.fixture(autouse=True)
def isolated_review_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep review snapshots and saved-review discovery inside the test directory."""
    monkeypatch.setattr(validation_workflow, 'REVIEW_DIR', tmp_path / 'reviews')


def _csv(headers: list[str], rows: list[list[str]]) -> str:
    """Build a tiny CSV fixture without temporary files."""
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(headers)
    writer.writerows(rows)
    return text.getvalue()


def _gold() -> str:
    """Return validation rows with two materials and one empty paper."""
    return _csv(['Identifier', 'Title', 'DOI', 'Material system', 'Band gap'], [
        ['paper-a', 'First paper', '10.1234/ABC', 'Li2O', '1 eV'],
        ['paper-a', 'First paper', '10.1234/ABC', 'Na2O', '2 eV'],
        ['paper-b', 'Second paper', '', 'ZnO', '3 eV'],
        ['paper-empty', 'No gaps', '10.1234/empty', 'None', '[]'],
    ])


def _scraped() -> str:
    """Return scrape rows with a DOI match and one extra material."""
    return _csv(['Unnamed: 0', 'Paper id', 'doi', 'Material system', 'Band gap'], [
        ['0', 'different-id', 'https://doi.org/10.1234/abc', 'Na2O', '2 eV'],
        ['1', 'different-id', 'https://doi.org/10.1234/abc', 'Li2O', '1 eV'],
        ['2', 'paper-b', '', 'ZnO', '3 eV'],
        ['3', 'paper-extra', '10.9999/extra', 'MgO', '4 eV'],
    ])


def test_review_matches_papers_before_materials_and_keeps_empty_papers(tmp_path: Path) -> None:
    """DOI and metadata determine paper scope before recipe identities."""
    app = ReviewApp(output=tmp_path / 'review.json')
    session = app.load(_gold(), _scraped(), 'band_gap_validation')
    assert len(session['validation']) == 3
    first, second, empty = session['validation']
    assert first['match_reason'] == 'DOI'
    assert first['warning'] == 'Paper IDs differ; DOI matched.'
    assert second['match_reason'] == 'paper ID'
    assert empty['rows'] == []
    scraped = next(p for p in session['scraped'] if p['id'] == first['suggested'])
    by_name = {r['values']['Material system']: r['id'] for r in scraped['rows']}
    assert first['material_suggestions'][first['rows'][0]['id']] == by_name['Li2O']
    assert first['material_suggestions'][first['rows'][1]['id']] == by_name['Na2O']


def test_review_title_fallback_conflicts_and_ambiguous_matches(tmp_path: Path) -> None:
    """Fallback metadata never overrides a contradictory DOI."""
    app = ReviewApp(output=tmp_path / 'review.json')
    gold = _csv(['Title', 'DOI', 'Material system'], [
        ['Title only', '', 'A'], ['Conflict', '10.1/gold', 'B'], ['Ambiguous', '', 'C'],
    ])
    scraped = _csv(['Title', 'doi', 'Material system'], [
        ['Title only', '', 'A'], ['Conflict', '10.1/other', 'B'],
        ['Ambiguous', '', 'C'], ['Ambiguous', '', 'D'],
    ])
    session = app.load(gold, scraped, 'band_gap_validation')
    assert session['validation'][0]['match_reason'] == 'title'
    assert session['validation'][1]['suggested'] is None
    assert 'conflicts' in session['validation'][1]['warning']
    # Two scraped rows from the same title are one paper, not two candidates.
    assert session['validation'][2]['match_reason'] == 'title'

    ambiguous_gold = _csv(['Title', 'Material system'], [['Shared title', 'A']])
    ambiguous_scraped = _csv(['Paper id', 'Title', 'Material system'], [
        ['one', 'Shared title', 'A'], ['two', 'Shared title', 'B'],
    ])
    second = ReviewApp(output=tmp_path / 'ambiguous.json').load(
        ambiguous_gold, ambiguous_scraped, 'band_gap_validation'
    )
    assert second['validation'][0]['suggested'] is None
    assert 'Ambiguous title' in second['validation'][0]['warning']


def test_grouped_paper_flags_conflicting_metadata(tmp_path: Path) -> None:
    """Rows under one DOI retain a visible metadata conflict warning."""
    gold = _csv(['Identifier', 'DOI', 'Material system'], [
        ['first-id', '10.1/shared', 'A'], ['second-id', '10.1/shared', 'B'],
    ])
    scraped = _csv(['Paper id', 'doi', 'Material system'], [
        ['first-id', '10.1/shared', 'A'],
    ])
    session = ReviewApp(output=tmp_path / 'review.json').load(
        gold, scraped, 'band_gap_validation'
    )
    assert len(session['validation']) == 1
    assert 'conflicting paper id' in session['validation'][0]['warning']


def test_review_saves_and_resumes_without_touching_sources(tmp_path: Path) -> None:
    """The review file contains decisions and is bound to exact source bytes."""
    path = tmp_path / 'review.json'
    app = ReviewApp(output=path)
    session = app.load(_gold(), _scraped(), 'band_gap_validation')
    row = session['validation'][0]['rows'][0]
    choices = {'paperPairs': {}, 'materialPairs': {row['id']: None},
               'fields': {row['id']: {'Band gap': {'status': 'incorrect', 'note': 'Check unit'}}},
               'extra': {}, 'notes': {}}
    app.save(choices)
    assert ReviewApp(output=path).load(_gold(), _scraped(), 'band_gap_validation')['decisions'] == choices
    changed = ReviewApp(output=path).load(_gold() + '\n', _scraped(), 'band_gap_validation')
    assert changed['fingerprint'] != session['fingerprint']
    assert changed['review_path'] != str(path)
    assert changed['decisions'] == {}
    assert json.loads(path.read_text())['fingerprint'] == session['fingerprint']


def test_review_rejects_unusable_csv_and_reads_current_band_gap_data(tmp_path: Path) -> None:
    """Reject missing recipe columns and accept the maintained validation set."""
    app = ReviewApp(output=tmp_path / 'review.json')
    with pytest.raises(ValueError, match='no fields'):
        app.load('DOI,Other\n10.1/a,value\n', _scraped(), 'band_gap_validation')
    repo = Path(__file__).parents[2]
    validation = (repo / 'tests/data/verification_data/band_gaps.csv').read_text()
    session = app.load(validation, _scraped(), 'band_gap_validation')
    assert len(session['validation']) == 113
    assert sum(len(p['rows']) for p in session['validation']) >= 200
    assert sum(not p['rows'] for p in session['validation']) >= 40


def test_validate_cli_preloads_inputs_and_opens_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public command accepts repeatable file arguments."""
    monkeypatch.setattr(validation_workflow, 'REVIEW_DIR', tmp_path / 'reviews')
    gold = tmp_path / 'gold.csv'
    scraped = tmp_path / 'scraped.csv'
    gold.write_text(_gold())
    scraped.write_text(_scraped())
    called = {}

    def fake_serve(app: ReviewApp, *, open_browser: bool) -> None:
        """Capture the review app and browser option passed by the CLI."""
        called['app'] = app
        called['open_browser'] = open_browser

    monkeypatch.setattr(cli, 'serve_validation', fake_serve)
    result = CliRunner().invoke(cli.main, ['validate', 'gui', '--validation', str(gold),
                                            '--scraped', str(scraped), '--recipe',
                                            'band_gap_validation', '--no-browser'])
    assert result.exit_code == 0, result.output
    assert called['app'].session['validation']
    assert called['open_browser'] is False


@pytest.mark.parametrize('failure', ['serialization', 'replace'])
def test_failed_review_save_preserves_file_state_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Keep both saved decisions and application state when an atomic write fails."""
    path = tmp_path / 'review.json'
    app = ReviewApp(output=path)
    app.load(_gold(), _scraped(), 'band_gap_validation')
    initial = {'comment': 'café'}
    app.save(initial)
    before = path.read_bytes()
    candidate: dict[str, object] = {'comment': 'changed'}
    if failure == 'serialization':
        candidate['invalid'] = object()
        error = TypeError
    else:
        def fail_replace(source: str, destination: Path) -> None:
            """Simulate a filesystem failure at the final replacement boundary."""
            raise OSError('replace failed')

        monkeypatch.setattr(validation_workflow.os, 'replace', fail_replace)
        error = OSError
    with pytest.raises(error):
        app.save(candidate)
    assert path.read_bytes() == before
    assert app.decisions == initial
    assert app.session['decisions'] == initial
    assert not list(tmp_path.rglob('*.tmp'))


def test_snapshot_serialization_failure_cleans_temporary_file(tmp_path: Path) -> None:
    """Leave no half-written input snapshot after JSON serialization fails."""
    app = ReviewApp()
    app.input_snapshot = {'fingerprint': 'test', 'invalid': object()}
    with pytest.raises(TypeError):
        app._save_input_snapshot()
    assert not list(tmp_path.rglob('*.tmp'))
    assert not list(tmp_path.rglob('*.inputs.json'))


@pytest.mark.parametrize('options', [
    ['--validation'], ['--scraped', '--recipe'], ['--recipe'],
])
def test_validate_cli_requires_all_preload_options_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str],
) -> None:
    """Refuse a partial preload instead of opening a half-configured review."""
    values = {'--validation': tmp_path / 'gold.csv', '--scraped': tmp_path / 'scraped.csv',
              '--recipe': 'band_gap_validation'}
    values['--validation'].write_text(_gold())
    values['--scraped'].write_text(_scraped())
    monkeypatch.setattr(cli, 'serve_validation',
                        lambda app, *, open_browser: pytest.fail('server started'))
    arguments = [part for option in options for part in (option, str(values[option]))]
    result = CliRunner().invoke(cli.main, ['validate', 'gui', *arguments, '--no-browser'])
    assert result.exit_code == 2
    assert 'Supply --validation, --scraped, and --recipe together' in result.output


@pytest.mark.parametrize(('content', 'message'), [
    (b'DOI,Other\n10.1/a,value\n', 'Validation CSV has no fields from the selected recipe'),
    (b'DOI,Material system\n10.1/a,\xff\n', "'utf-8' codec can't decode"),
])
def test_validate_cli_reports_unusable_inputs_without_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes, message: str,
) -> None:
    """Turn input errors into a CLI error before any server starts."""
    gold = tmp_path / 'gold.csv'
    scraped = tmp_path / 'scraped.csv'
    gold.write_bytes(content)
    scraped.write_text(_scraped())
    monkeypatch.setattr(cli, 'serve_validation',
                        lambda app, *, open_browser: pytest.fail('server started'))
    result = CliRunner().invoke(cli.main, ['validate', 'gui', '--validation', str(gold),
                                            '--scraped', str(scraped), '--recipe',
                                            'band_gap_validation', '--no-browser'])
    assert result.exit_code == 1
    assert message in result.output


@pytest.mark.parametrize(('label', 'content', 'message'), [
    ('Validation', '', 'needs a header with unique column names'),
    ('Validation', 'DOI,DOI,Material system\n10.1/a,10.1/a,ZnO\n',
     'needs a header with unique column names'),
    ('Validation', 'DOI,Band gap\n10.1/a,1 eV\n', 'missing recipe identity field.*Material system'),
    ('Validation', 'Notes,Material system\nx,ZnO\n', 'needs a DOI, paper ID, or title column'),
    ('Validation', 'DOI,Material system\n10.1/a,ZnO,surplus\n', 'row 2 has more cells than its header'),
    ('Validation', 'DOI,Material system\n10.1/a,' + 'x' * 200_000 + '\n', 'cannot be read: field larger'),
    ('Scraped', 'doi,doi,Material system\n', 'needs a header with unique column names'),
])
def test_review_rejects_malformed_csv_inputs(tmp_path: Path, label: str, content: str, message: str) -> None:
    """Name the failing file and problem rather than building a misleading review."""
    gold, scraped = (content, _scraped()) if label == 'Validation' else (_gold(), content)
    app = ReviewApp(output=tmp_path / 'review.json')
    with pytest.raises(ValueError, match=f'^{label} CSV .*{message}'):
        app.load(gold, scraped, 'band_gap_validation')
    assert app.session is None
    assert not (tmp_path / 'review.json').exists()


def test_review_converts_python_literal_cells_and_keeps_raw_source_text(tmp_path: Path) -> None:
    """Normalize list cells for the browser, skip blank rows, and honour BOMs and unit headers."""
    gold = '﻿' + _csv(['DOI', 'Material system', 'All band gaps [eV]', 'Band gap'], [
        ['10.1/a', 'ZnO', "[{'value': '3.3 eV', 'conditions': None}]", '{1, 2}'],
        ['', '', '', ''],
        ['10.1/a', 'GaN', '[unterminated', '[{"value": "3.4 eV"}]'],
    ])
    session = ReviewApp(output=tmp_path / 'review.json').load(gold, _scraped(), 'band_gap_validation')
    (paper,) = session['validation']
    assert paper['meta']['doi'] == '10.1/a'
    zno, gan = paper['rows']
    assert (zno['line'], gan['line']) == (2, 4)
    assert zno['values']['All band gaps'] == '[{"value": "3.3 eV", "conditions": null}]'
    assert zno['raw_values']['All band gaps'] == "[{'value': '3.3 eV', 'conditions': None}]"
    assert zno['values']['Band gap'] == '{1, 2}'
    assert zno['values']['Cited literature band gaps'] == ''
    assert gan['values']['All band gaps'] == '[unterminated'
    assert gan['values']['Band gap'] == '[{"value": "3.4 eV"}]'


def _recipe(examples: dict[str, object], identity: list[str]) -> dict[str, Any]:
    """Build a minimal uploaded recipe with the given field examples."""
    return {
        'record definition': {'subject': 'samples', 'singular': 'sample', 'plural': 'samples',
                              'unit': 'a sample', 'identity fields': identity},
        'search fields': {name: {'description': name, 'example': example}
                          for name, example in examples.items()},
    }


@pytest.mark.parametrize('wrapped', [False, True])
def test_uploaded_recipe_detects_structured_fields_and_defaults_identity(
    tmp_path: Path, wrapped: bool,
) -> None:
    """Infer list and dictionary fields from cells and use the first field as identity."""
    recipe = _recipe({'Sample': 'S1', 'Phases': 'rutile', 'Conditions': '300 K',
                      'All band gaps': '3 eV', 'Notes': 'text'}, identity=[])
    gold = _csv(['Title', 'Sample', 'Phases', 'Conditions', 'All band gaps', 'Notes'], [
        ['Paper', 'TiO2 film', "['rutile', 'anatase']", '{"T": "300 K"}', '3 eV', 'plain'],
    ])
    scraped = _csv(['Title', 'Sample', 'Notes'], [['Paper', 'TiO2 films', '7']])
    session = ReviewApp(output=tmp_path / 'review.json').load(
        gold, scraped, '', recipe_text=json.dumps({'mine': recipe} if wrapped else recipe),
    )
    assert session['identity_fields'] == ['Sample']
    assert session['structure_fields'] == {'Phases': 'list', 'Conditions': 'dict',
                                           'All band gaps': 'list'}
    assert session['list_fields'] == ['Phases', 'All band gaps']
    assert session['recipe'] == 'custom'
    assert session['sources']['recipe']['name'] == 'uploaded recipe'
    (paper,) = session['validation']
    assert paper['match_reason'] == 'title'
    assert list(paper['material_suggestions'].values()) == [session['scraped'][0]['rows'][0]['id']]


@pytest.mark.parametrize(('options', 'message'), [
    ({'recipe_text': '{not json'}, 'Recipe JSON is invalid'),
    ({'recipe_text': '[]'}, 'must be a JSON object'),
    ({'validation_sha256': 'abc'}, 'SHA-256 values must be 64 lowercase'),
    ({'scraped_sha256': 'A' * 64}, 'SHA-256 values must be 64 lowercase'),
])
def test_review_rejects_invalid_recipe_text_and_source_hashes(
    tmp_path: Path, options: dict[str, str], message: str,
) -> None:
    """Validate uploaded recipes and declared source hashes before writing anything."""
    app = ReviewApp(output=tmp_path / 'review.json')
    with pytest.raises(ValueError, match=message):
        app.load(_gold(), _scraped(), 'band_gap_validation', **options)
    assert app.session is None
    assert not (tmp_path / 'review.json').exists()


def test_material_suggestions_are_one_to_one_and_skip_blank_identities(tmp_path: Path) -> None:
    """Give each scraped entry to its closest validation entry and leave unnamed rows unpaired."""
    gold = _csv(['DOI', 'Material system', 'Band gap'], [
        ['10.1/a', 'ZnO2', '2 eV'], ['10.1/a', 'ZnO', '1 eV'], ['10.1/a', '', '5 eV'],
    ])
    scraped = _csv(['doi', 'Material system', 'Band gap'], [
        ['10.1/a', 'ZnO', '1 eV'], ['10.1/a', '', '5 eV'],
    ])
    session = ReviewApp(output=tmp_path / 'review.json').load(gold, scraped, 'band_gap_validation')
    (paper,) = session['validation']
    assert len(paper['rows']) == 3
    zno = paper['rows'][1]
    assert paper['material_suggestions'] == {zno['id']: session['scraped'][0]['rows'][0]['id']}


def test_paper_already_suggested_by_doi_is_not_offered_again_by_id(tmp_path: Path) -> None:
    """A weaker metadata match cannot claim a paper another validation paper already holds."""
    gold = _csv(['Identifier', 'DOI', 'Material system'], [
        ['first', '10.1/shared', 'A'], ['shared-id', '', 'B'],
    ])
    scraped = _csv(['Paper id', 'doi', 'Material system'], [['shared-id', '10.1/shared', 'A']])
    by_doi, by_id = ReviewApp(output=tmp_path / 'review.json').load(
        gold, scraped, 'band_gap_validation',
    )['validation']
    assert by_doi['match_reason'] == 'DOI'
    assert by_doi['warning'] == 'Paper IDs differ; DOI matched.'
    assert by_id['suggested'] is None
    assert by_id['warning'] == 'paper ID matches a paper already suggested elsewhere; choose manually.'


def test_loaded_session_scores_to_completion_from_suggested_pairs(tmp_path: Path) -> None:
    """Accepting each suggestion and rejecting the unmatched scrape gives a complete score."""
    session = ReviewApp(output=tmp_path / 'review.json').load(_gold(), _scraped(), 'band_gap_validation')
    pairs = [row for paper in session['validation'] for row in paper['material_suggestions']]
    extra = next(paper for paper in session['scraped'] if paper['meta']['doi'] == '10.9999/extra')
    score = score_review(session, {
        'fields': {row: {'Material system': {'status': 'correct'}} for row in pairs},
        'parts': {row: {'Band gap': {'values': {'': {'status': 'correct'}}}} for row in pairs},
        'extra': {row['id']: {'status': 'incorrect'} for row in extra['rows']},
        'run': {key: 'recorded' for key in RUN_FIELDS},
    })
    assert len(pairs) == 3
    assert (score['papers_evaluated'], score['pending_pairs'], score['complete']) == (3, 0, True)
    for field in ('Material system', 'Band gap'):
        metric = score['fields'][field]
        assert (metric['tp'], metric['fp'], metric['fn']) == (3, 1, 0)
        assert (metric['precision'], metric['recall']) == (0.75, 1.0)


def test_review_refuses_a_saved_review_bound_to_other_inputs(tmp_path: Path) -> None:
    """Never overwrite decisions that were recorded against different source files."""
    path = Path(ReviewApp().load(_gold(), _scraped(), 'band_gap_validation')['review_path'])
    foreign = {'fingerprint': '0' * 64, 'decisions': {'notes': 'keep'}}
    path.write_text(json.dumps(foreign))
    with pytest.raises(ValueError, match='belongs to different inputs'):
        ReviewApp().load(_gold(), _scraped(), 'band_gap_validation')
    assert json.loads(path.read_text()) == foreign


def test_save_requires_loaded_inputs_and_object_decisions(tmp_path: Path) -> None:
    """Reject saves that have nowhere to go or cannot be represented as a review."""
    output = tmp_path / 'review.json'
    app = ReviewApp(output=output)
    with pytest.raises(ValueError, match='Select the review inputs first'):
        app.save({})
    app._save_input_snapshot()
    assert not validation_workflow.REVIEW_DIR.exists()
    app.load(_gold(), _scraped(), 'band_gap_validation')
    with pytest.raises(ValueError, match='must be an object'):
        app.save(['not', 'an', 'object'])
    assert json.loads(output.read_text())['decisions'] == {}


@pytest.mark.parametrize('cached', ['valid', 'corrupt', 'other inputs'])
def test_input_snapshot_reuses_valid_copies_and_repairs_damaged_ones(tmp_path: Path, cached: str) -> None:
    """Keep a matching CSV snapshot untouched and rewrite one that cannot restore the review."""
    fingerprint = ReviewApp(output=tmp_path / 'review.json').load(
        _gold(), _scraped(), 'band_gap_validation')['fingerprint']
    snapshot = validation_workflow.REVIEW_DIR / f'{fingerprint}.inputs.json'
    if cached == 'valid':
        snapshot.write_text(json.dumps({**json.loads(snapshot.read_text()), 'marker': 'kept'}))
    elif cached == 'corrupt':
        snapshot.write_text('{')
    else:
        snapshot.write_text(json.dumps({'fingerprint': '0' * 64, 'validation_csv': '',
                                        'scraped_csv': ''}))
    ReviewApp(output=tmp_path / 'review.json').load(_gold(), _scraped(), 'band_gap_validation')
    restored = json.loads(snapshot.read_text())
    assert restored['fingerprint'] == fingerprint
    assert (restored['validation_csv'], restored['scraped_csv']) == (_gold(), _scraped())
    assert ('marker' in restored) is (cached == 'valid')


def test_saved_reviews_list_newest_first_and_skip_untrustworthy_files(tmp_path: Path) -> None:
    """List genuine reviews with resumability, ignoring snapshots and malformed files."""
    review_dir = validation_workflow.REVIEW_DIR
    assert ReviewApp().list_reviews() == []
    default = ReviewApp().load(_gold(), _scraped(), 'band_gap_validation',
                               validation_name='gold.csv', scraped_name='scraped.csv')
    mine = ReviewApp(output=tmp_path / 'mine.json').load(_gold(), _scraped() + '\n',
                                                         'band_gap_validation')
    cache = f"{default['fingerprint']}.inputs.json"
    files = {
        review_dir / 'corrupt.json': '{',
        review_dir / 'array.json': '[]',
        review_dir / 'short.json': json.dumps({'fingerprint': 'xyz'}),
        review_dir / 'legacy.json': json.dumps({'fingerprint': 'a' * 64, 'recipe': 'sse'}),
        review_dir / 'escaped.json': json.dumps({'fingerprint': 'b' * 64,
                                                 'input_cache': f'../{cache}'}),
        review_dir / 'stale.json': json.dumps({'fingerprint': 'c' * 64, 'input_cache': cache}),
    }
    for path, content in files.items():
        path.write_text(content)
    for age, path in enumerate([review_dir / 'legacy.json', review_dir / 'escaped.json',
                                review_dir / 'stale.json', tmp_path / 'mine.json',
                                Path(default['review_path'])]):
        os.utime(path, (1000 + age, 1000 + age))

    reviews = ReviewApp(output=tmp_path / 'mine.json').list_reviews()
    assert [(review['fingerprint'], review['resumable']) for review in reviews] == [
        (default['fingerprint'], True), (mine['fingerprint'], True),
        ('c' * 64, False), ('b' * 64, False), ('a' * 64, False),
    ]
    assert (reviews[0]['validation_name'], reviews[0]['scraped_name']) == ('gold.csv', 'scraped.csv')
    assert reviews[0]['recipe'] == 'band_gap_validation'
    assert (reviews[-1]['recipe'], reviews[-1]['validation_name']) == ('sse', 'Validation CSV')
    assert all(re.fullmatch(r'[0-9a-f]{24}', review['id']) for review in reviews)
    assert len({review['id'] for review in reviews}) == len(reviews)
    assert ReviewApp().list_reviews() == reviews[:1] + reviews[2:]


def test_resume_review_restores_saved_inputs_and_decisions(tmp_path: Path) -> None:
    """Reopen a saved review from its local CSV copies and keep writing to the same file."""
    original = ReviewApp()
    original.load(_gold(), _scraped(), 'band_gap_validation', validation_name='gold.csv',
                  scraped_name='scraped.csv', validation_sha256='a' * 64)
    original.save({'notes': {'v:2': 'checked'}})
    app = ReviewApp(output=tmp_path / 'unused.json')
    (review,) = app.list_reviews()
    session = app.resume_review(review['id'])
    assert session['decisions'] == {'notes': {'v:2': 'checked'}}
    assert session['fingerprint'] == original.session['fingerprint']
    assert session['sources']['validation'] == {'name': 'gold.csv', 'sha256': 'a' * 64}
    assert app.review_path == original.review_path
    app.save({'notes': {}})
    assert json.loads(original.review_path.read_text())['decisions'] == {'notes': {}}
    assert not (tmp_path / 'unused.json').exists()


@pytest.mark.parametrize(('scenario', 'message'), [
    ('unknown id', 'no longer available'),
    ('no cache', 'no saved CSV copies'),
    ('nested cache', 'no saved CSV copies'),
    ('other fingerprint', 'do not match this review'),
    ('no recipe', 'missing its recipe'),
    ('edited copy', 'belongs to different inputs'),
])
def test_resume_review_rejects_missing_or_inconsistent_snapshots(scenario: str, message: str) -> None:
    """Refuse to resume from CSV copies that cannot reproduce the saved review."""
    session = ReviewApp().load(_gold(), _scraped(), 'band_gap_validation')
    review_path = Path(session['review_path'])
    snapshot_path = validation_workflow.REVIEW_DIR / f"{session['fingerprint']}.inputs.json"
    saved = json.loads(review_path.read_text())
    snapshot = json.loads(snapshot_path.read_text())
    if scenario == 'no cache':
        del saved['input_cache']
    elif scenario == 'nested cache':
        saved['input_cache'] = f"nested/{saved['input_cache']}"
    elif scenario == 'other fingerprint':
        snapshot['fingerprint'] = '0' * 64
    elif scenario == 'no recipe':
        snapshot['inputs']['recipe_name'] = ''
    elif scenario == 'edited copy':
        snapshot['scraped_csv'] += '9,late,10.1/late,CdS,2.4 eV\n'
    review_path.write_text(json.dumps(saved))
    snapshot_path.write_text(json.dumps(snapshot))

    app = ReviewApp()
    review_id = 'unknown' if scenario == 'unknown id' else app.list_reviews()[0]['id']
    with pytest.raises(ValueError, match=message):
        app.resume_review(review_id)
    assert app.session is None
    assert app._resume_path is None
    assert json.loads(review_path.read_text()) == saved


def _http(address: str, method: str, route: str, body: bytes | None = None, *,
          token: str | None = None,
          headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    """Send one request to a running review server and return status, headers, and body."""
    url = urlsplit(address)
    token = parse_qs(url.query)['token'][0] if token is None else token
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
    try:
        connection.request(method, f'{route}?token={token}', body=body, headers=headers or {})
        response = connection.getresponse()
        return (response.status, {key.lower(): value for key, value in response.getheaders()},
                response.read())
    finally:
        connection.close()


def _api(address: str, route: str, payload: object = None) -> tuple[int, Any]:
    """Call a JSON endpoint with GET, or with POST when a payload is supplied."""
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    status, _, content = _http(address, 'GET' if payload is None else 'POST', route, body)
    return status, json.loads(content)


@pytest.fixture
def review_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[ReviewApp], str]]:
    """Start review servers on loopback ports and stop them after the test."""
    servers: list[validation_workflow.ThreadingHTTPServer] = []
    threads: list[threading.Thread] = []
    addresses: queue.Queue[str] = queue.Queue()

    class RecordingServer(validation_workflow.ThreadingHTTPServer):
        """Keep a handle on each server so the fixture can shut it down."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Bind normally and record the server."""
            super().__init__(*args, **kwargs)
            servers.append(self)

    monkeypatch.setattr(validation_workflow, 'ThreadingHTTPServer', RecordingServer)
    monkeypatch.setattr(validation_workflow.webbrowser, 'open', addresses.put)

    def start(app: ReviewApp) -> str:
        """Serve one app in the background and return the URL it would open."""
        thread = threading.Thread(target=validation_workflow.serve, args=(app,), daemon=True)
        thread.start()
        threads.append(thread)
        return addresses.get(timeout=5)

    yield start
    for server in servers:
        server.shutdown()
    for thread in threads:
        thread.join(timeout=5)


def test_review_server_requires_its_token_and_serves_uncached_assets(
    review_server: Callable[[ReviewApp], str],
) -> None:
    """Gate every route on the session token and serve the page with that token embedded."""
    address = review_server(ReviewApp())
    token = parse_qs(urlsplit(address).query)['token'][0]
    assert re.fullmatch(r'http://127\.0\.0\.1:\d+/\?token=\S+', address)
    for method in ('GET', 'POST'):
        status, _, body = _http(address, method, '/api/config', token='wrong')
        assert (status, json.loads(body)) == (403, {'error': 'Invalid review session token.'})

    status, headers, page = _http(address, 'GET', '/')
    assert status == 200
    assert headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'no-store'
    assert headers['x-content-type-options'] == 'nosniff'
    assert f'token={token}'.encode() in page
    assert b'__TOKEN__' not in page
    status, headers, script = _http(address, 'GET', '/review.js')
    assert (status, headers['content-type']) == (200, 'text/javascript; charset=utf-8')
    assert script == validation_workflow.SCRIPT.read_bytes()
    status, headers, logo = _http(address, 'GET', '/brand.svg')
    assert (status, headers['content-type']) == (200, 'image/svg+xml; charset=utf-8')
    assert logo == validation_workflow.BRAND_LOGO.read_bytes()
    assert _api(address, '/missing') == (404, {'error': 'Not found.'})


def test_review_server_loads_saves_scores_exports_and_resumes(
    review_server: Callable[[ReviewApp], str],
) -> None:
    """Drive the browser API through one review, including its error responses."""
    app = ReviewApp()
    address = review_server(app)
    status, config = _api(address, '/api/config')
    assert status == 200
    assert config == {'recipes': ReviewApp.recipes(), 'session': None, 'reviews': []}
    assert 'band_gap_validation' in config['recipes']
    not_loaded = (400, {'error': 'Select the review inputs first.'})
    assert _api(address, '/api/export') == not_loaded
    assert _api(address, '/api/score', {'fingerprint': None, 'decisions': {}}) == not_loaded

    status, session = _api(address, '/api/load', {
        'validation': _gold(), 'scraped': _scraped(), 'recipe': 'band_gap_validation',
        'validation_name': 'gold.csv', 'scraped_name': 'scraped.csv',
    })
    assert status == 200
    fingerprint = session['fingerprint']
    assert session['sources']['validation']['name'] == 'gold.csv'
    decisions = {'notes': {'v:2': 'checked'}}
    assert _api(address, '/api/save', {'fingerprint': 'stale', 'decisions': decisions}) == (
        400, {'error': 'Input files changed; reload the review before saving.'})
    assert app.decisions == {}
    assert _api(address, '/api/save', {'fingerprint': fingerprint, 'decisions': decisions}) == (
        200, {'saved': True})
    assert json.loads(Path(session['review_path']).read_text())['decisions'] == decisions

    status, score = _api(address, '/api/score', {'fingerprint': fingerprint, 'decisions': {}})
    assert (status, score['fingerprint'], score['complete']) == (200, fingerprint, False)
    status, export = _api(address, '/api/export')
    assert (status, export['decisions']) == (200, decisions)
    assert export['score']['papers_evaluated'] == 3
    status, reviews = _api(address, '/api/reviews')
    assert status == 200
    assert [(review['fingerprint'], review['resumable']) for review in reviews] == [(fingerprint, True)]
    status, resumed = _api(address, '/api/resume', {'review_id': reviews[0]['id']})
    assert (status, resumed['decisions']) == (200, decisions)
    assert _api(address, '/api/resume', {'review_id': 'gone'}) == (
        400, {'error': 'That saved review is no longer available.'})

    assert _api(address, '/api/unknown', {}) == (404, {'error': 'Not found.'})
    assert _api(address, '/api/load', {'scraped': _scraped()}) == (400, {'error': "'validation'"})
    status, error = _api(address, '/api/load', {'validation': 'DOI,Other\n10.1/a,b\n',
                                                'scraped': _scraped(), 'recipe': 'band_gap_validation'})
    assert status == 400
    assert 'no fields from the selected recipe' in error['error']
    assert app.session['fingerprint'] == fingerprint
    status, _, body = _http(address, 'POST', '/api/save', b'{not json')
    assert status == 400
    # Rejected before the body is read, so no body is sent and the connection closes cleanly.
    for size in ('0', str(validation_workflow.MAX_REQUEST_BYTES + 1)):
        status, _, body = _http(address, 'POST', '/api/save', headers={'Content-Length': size})
        assert (status, json.loads(body)) == (
            400, {'error': 'Request is empty or exceeds the 32 MB limit.'})


def test_serve_prints_its_address_and_closes_on_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Stop quietly on Ctrl+C, release the port, and only open a browser on request."""
    closed = []

    class InterruptedServer(validation_workflow.ThreadingHTTPServer):
        """Simulate Ctrl+C as soon as serving starts."""

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            """Raise the interrupt a terminal user would send."""
            raise KeyboardInterrupt

        def server_close(self) -> None:
            """Record that the listening socket was released."""
            closed.append(True)
            super().server_close()

    monkeypatch.setattr(validation_workflow, 'ThreadingHTTPServer', InterruptedServer)
    monkeypatch.setattr(validation_workflow.webbrowser, 'open',
                        lambda address: pytest.fail('browser opened'))
    validation_workflow.serve(ReviewApp(), open_browser=False)
    output = capsys.readouterr().out
    assert re.search(r'^Validation review: http://127\.0\.0\.1:\d+/\?token=\S+$', output, re.MULTILINE)
    assert 'Press Ctrl+C to stop the review server.' in output
    assert closed == [True]
