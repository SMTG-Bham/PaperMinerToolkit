"""Check that --dry-run reports changes while leaving the corpus and files untouched."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd
import pytest
import requests
from click.testing import CliRunner

import paperminertoolkit.corpus.database as corpus
import paperminertoolkit.extraction.store as store
from paperminertoolkit import cli
from paperminertoolkit.corpus import filtering


def _forbid(name: str) -> Any:
    """Return a callable that fails the test if a side effect is attempted."""
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail on any call."""
        raise AssertionError(f'{name} was called during a dry run')
    return forbidden


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a corpus with scrape statuses, one stored PDF, and an active filter."""
    path = tmp_path / 'papers.db'
    pdf = tmp_path / 'a.pdf'
    pdf.write_bytes(b'%PDF-1.4 stand-in')
    first = {'paper_id': 'a', 'title': 'Keep LLZO', 'text_scrape_status': 'succeeded'}
    with closing(corpus.connect(path)) as conn, conn:
        corpus.upsert_paper(conn, first)
        corpus.upsert_paper(conn, {'paper_id': 'b', 'title': 'Drop LATP',
                                   'text_scrape_status': 'failed'})
        corpus.add_asset(conn, first, pdf, role='pdf', kind='pdf', mime_type='application/pdf')
    rules = tmp_path / 'keep.json'
    rules.write_text(json.dumps({'name': 'keep', 'fields': ['title'],
                                 'include': [{'name': 'keep', 'pattern': 'Keep'}]}),
                     encoding='utf-8')
    filtering.apply_regex_filter(path, rules)
    monkeypatch.setattr(requests.Session, 'request', _forbid('requests.Session.request'))
    return path


def _run(*arguments: str, input: str | None = None) -> dict[str, Any]:
    """Run a command with --json and return its parsed result."""
    result = CliRunner().invoke(cli.main, [*arguments, '--json'], input=input)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_reset_dry_run_reports_without_resetting(db_path: Path) -> None:
    """Count what a reset would clear and keep every status."""
    before = db_path.read_bytes()
    payload = _run('reset', str(db_path), '--dry-run')
    assert payload['dry_run'] is True and payload['papers_reset'] == 2
    assert db_path.read_bytes() == before
    human = CliRunner().invoke(cli.main, ['reset', str(db_path), '--dry-run'])
    assert 'Would reset 2 papers' in human.stdout


def test_filter_dry_runs_preview_decisions_and_keep_the_stack(db_path: Path, tmp_path: Path) -> None:
    """Show the decisions a new or removed filter would give, then change nothing."""
    before = db_path.read_bytes()
    rules = tmp_path / 'latp.json'
    rules.write_text(json.dumps({'name': 'latp', 'fields': ['title'],
                                 'include': [{'name': 'latp', 'pattern': 'LATP'}]}),
                     encoding='utf-8')
    added = _run('filter', 'regex', str(db_path), str(rules), '--join', 'or', '--dry-run')
    assert added['dry_run'] is True
    assert [item['name'] for item in added['filters']] == ['keep', 'latp']
    assert added['counts']['included'] == 2
    removed = _run('filter', 'reset', str(db_path), '--all', '--dry-run')
    assert removed['filters'] == [] and removed['dry_run'] is True
    assert db_path.read_bytes() == before
    status = _run('filter', 'status', str(db_path))
    assert [item['name'] for item in status['filters']] == ['keep']
    human = CliRunner().invoke(cli.main, ['filter', 'reset', str(db_path), '--all', '--dry-run'])
    assert 'Dry run: the corpus was not changed' in human.stdout


def test_store_dry_run_matches_columns_without_converting_or_writing(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report conversions and unmatched columns without a model call, prompt, or file."""
    scraped = tmp_path / 'scraped.csv'
    output = tmp_path / 'materials.csv'
    pd.DataFrame({'Paper id': ['a'], 'Name': ['LLZO'], 'Conductivity': ['1e-3'],
                  'Unknown': ['x']}).to_csv(scraped)
    monkeypatch.setattr(store, 'convert_units', _forbid('convert_units'))
    monkeypatch.setattr(store, 'load_recipe', lambda _: {
        'record definition': {'subject': 's', 'singular': 'm', 'plural': 'ms',
                              'unit': 'u', 'identity fields': ['Name']},
        'search fields': {'Name': {'prompt': 'n', 'example': 'LLZO'},
                          'Conductivity [S cm^-1]': {'prompt': 'c', 'example': '1'}}})
    before = db_path.read_bytes()
    payload = _run('store', str(db_path), str(scraped), str(output), 'sse', '--dry-run')
    assert payload['status'] == 'dry_run' and payload['dry_run'] is True
    assert payload['converted_columns'] == ['Conductivity [S cm^-1]']
    assert payload['skipped_columns'] == ['Unknown']
    assert payload['rows_written'] == payload['rows_total'] == 1
    assert not output.exists() and scraped.exists()
    assert not list(tmp_path.glob('.paperminertoolkit-converted-*'))
    assert db_path.read_bytes() == before


def test_download_dry_run_lists_attempts_without_requests(db_path: Path) -> None:
    """Separate stored assets from requests a run would make, ignoring filters for named papers."""
    before = db_path.read_bytes()
    payload = _run('download', str(db_path), '--format', 'pdf', '--source', 'arxiv',
                   '--no-supplements', '--no-abstract', '--dry-run')
    assert payload['dry_run'] is True and payload['papers_considered'] == 2
    assert payload['would_attempt']['pdfs'] == 1
    assert payload['already_stored']['pdfs'] == 1
    assert {paper['paper_id']: paper['attempt'] for paper in payload['papers']} == {
        'a': [], 'b': ['pdf']}
    named = _run('download', str(db_path), '--format', 'pdf', '--source', 'arxiv',
                 '--paper-id', 'b', '--dry-run')
    assert [paper['paper_id'] for paper in named['papers']] == ['b']
    assert 'supplements' in named['papers'][0]['attempt']
    assert db_path.read_bytes() == before
