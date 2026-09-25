"""Exercise local validation input matching, persistence, and CLI wiring."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import paperminertoolkit.cli as cli
import paperminertoolkit.workflows.validation as validation_workflow
from paperminertoolkit.workflows.validation import ReviewApp


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
        called['app'] = app
        called['open_browser'] = open_browser

    monkeypatch.setattr(cli, 'serve_validation', fake_serve)
    result = CliRunner().invoke(cli.main, ['validate', 'gui', '--validation', str(gold),
                                            '--scraped', str(scraped), '--recipe',
                                            'band_gap_validation', '--no-browser'])
    assert result.exit_code == 0, result.output
    assert called['app'].session['validation']
    assert called['open_browser'] is False
