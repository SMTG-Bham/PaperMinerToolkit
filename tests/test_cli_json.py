"""Exercise machine-readable workflow and inspection output for CLI agents."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, NoReturn

import pytest
from click.testing import CliRunner

from paperminertoolkit import cli
from paperminertoolkit.corpus import database as corpus


@pytest.mark.parametrize(('command', 'target', 'summary'), [
    ('search', 'search_for_papers', {
        'status': 'completed', 'paper_ids': ['paper:1'], 'source_results': {},
    }),
    ('download', 'download_papers', {'pdfs': 1, 'texts': 0, 'abstracts': 0}),
    ('enrich', 'enrich_corpus', {'succeeded': 1, 'partial': 0, 'failed': 1}),
])
def test_workflow_json_is_separate_from_progress(
    command: str, target: str, summary: dict[str, Any],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the workflow payload unmodified while routing progress to stderr."""
    def workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Simulate workflow chatter alongside its structured return value."""
        print('Workflow progress')
        return summary

    monkeypatch.setattr(cli, target, workflow)
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    arguments = ['query', str(db_path)] if command == 'search' else [str(db_path)]
    result = CliRunner().invoke(cli.main, [command, *arguments, '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == summary
    assert result.stderr == 'Workflow progress\n'


@pytest.mark.parametrize('search_status', ['partial', 'failed'])
def test_search_json_survives_partial_failure(
    search_status: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let callers recover saved paper IDs even when a search provider fails."""
    summary = {
        'status': search_status, 'paper_ids': ['saved:1'],
        'source_results': {'arxiv': {'status': 'failed'}},
    }
    monkeypatch.setattr(cli, 'search_for_papers', lambda *args, **kwargs: summary)
    result = CliRunner().invoke(cli.main, ['search', 'query', '--json'])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == summary
    assert 'Available results have been saved' in result.stderr


@pytest.mark.parametrize('command', ['download', 'enrich'])
@pytest.mark.parametrize('error', [
    ValueError('Invalid configuration'), RuntimeError('Unsupported schema'),
    OSError('Permission denied'), sqlite3.OperationalError('Database is locked'),
])
def test_workflow_operational_errors_leave_stdout_empty(
    command: str, error: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report expected failures without corrupting the JSON output channel."""
    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise the selected storage or configuration failure."""
        raise error

    monkeypatch.setattr(cli, 'download_papers' if command == 'download' else 'enrich_corpus', fail)
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    result = CliRunner().invoke(cli.main, [command, str(db_path), '--json'])
    assert result.exit_code == 1
    assert result.stdout == ''
    assert f'Error: {error}' in result.stderr


def test_json_inspection_reports_real_corpus_state(tmp_path: Path) -> None:
    """Expose storage, failures, search provenance, and filter decisions as data."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {
            'paper_id': 'paper:1', 'title': 'Example',
            'pdf_download_status': 'failed', 'num_text_materials': 3,
        })
        search_id = corpus.begin_search_run(conn, 'query', 'arxiv', ['arxiv'], 1)
        corpus.finish_search_run(conn, search_id, 'completed', {})
    runner = CliRunner()
    stats = runner.invoke(cli.main, ['corpus', 'stats', str(db_path), '--json'])
    assert stats.exit_code == 0, stats.output
    assert json.loads(stats.stdout)['stats']['papers'] == 1
    assert 'enrichment' in json.loads(stats.stdout)

    progress = runner.invoke(cli.main, ['status', str(db_path), '--json'])
    assert progress.exit_code == 0, progress.output
    assert json.loads(progress.stdout)['pipeline']['pdf_download_status'] == {'failed': 1}
    assert json.loads(progress.stdout)['text_materials'] == 3

    outfile = tmp_path / 'history'
    history = runner.invoke(cli.main, [
        'corpus', 'searches', str(db_path), '--json', '--outfile', str(outfile),
    ])
    assert history.exit_code == 0, history.output
    assert json.loads(history.stdout)[0]['search_id'] == search_id
    assert json.loads(outfile.with_suffix('.json').read_text()) == json.loads(history.stdout)
    assert 'Search history written' in history.stderr

    filters = runner.invoke(cli.main, ['filter', 'status', str(db_path), '--json'])
    assert filters.exit_code == 0, filters.output
    assert isinstance(json.loads(filters.stdout), dict)


@pytest.mark.parametrize('command', [
    ['corpus', 'stats'], ['corpus', 'searches'], ['status'], ['filter', 'status'],
])
def test_json_inspection_reports_invalid_databases_cleanly(
    command: list[str], tmp_path: Path,
) -> None:
    """Keep corrupt database diagnostics off stdout and avoid tracebacks."""
    db_path = tmp_path / 'corrupt.db'
    db_path.write_text('not sqlite')
    result = CliRunner().invoke(cli.main, [*command, str(db_path), '--json'])
    assert result.exit_code == 1
    assert result.stdout == ''
    assert 'Error:' in result.stderr


def test_config_json_omits_api_keys_without_mutating_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep effective model credentials out of machine-readable status."""
    profile = {'provider': 'local', 'model': 'example', 'api_key': 'secret-value'}
    monkeypatch.setattr(cli, 'get_model_profile', lambda _: profile)
    result = CliRunner().invoke(cli.main, ['config', 'status', '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        'text': {'provider': 'local', 'model': 'example'},
        'vision': {'provider': 'local', 'model': 'example'},
    }
    assert 'secret-value' not in result.output
    assert profile['api_key'] == 'secret-value'


def test_recipe_discovery_outputs_validated_bundled_and_custom_recipes(tmp_path: Path) -> None:
    """Allow recipe selection and customization without any model calls."""
    runner = CliRunner()
    listed = runner.invoke(cli.main, ['recipe', 'list', '--json'])
    assert listed.exit_code == 0, listed.output
    names = json.loads(listed.stdout)
    assert 'sse' in names
    shown = runner.invoke(cli.main, ['recipe', 'show', 'sse'])
    assert shown.exit_code == 0, shown.output
    recipe = json.loads(shown.stdout)
    assert recipe['record definition'] and recipe['search fields']
    custom = tmp_path / 'recipe.json'
    custom.write_text(shown.stdout)
    assert runner.invoke(cli.main, ['recipe', 'show', str(custom)]).stdout == shown.stdout
    custom.write_text('{}')
    invalid = runner.invoke(cli.main, ['recipe', 'show', str(custom)])
    assert invalid.exit_code == 1
    assert invalid.stdout == ''
    missing = runner.invoke(cli.main, ['recipe', 'show', 'missing-recipe'])
    assert missing.exit_code == 1
    assert missing.stdout == ''
