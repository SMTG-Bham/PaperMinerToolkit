"""Exercise paper gathering and search through the public CLI."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, NoReturn

import pytest
from click.testing import CliRunner

from paperminertoolkit import cli


def _summary(status: str = 'completed', paper_ids: list[str] | None = None) -> dict[str, Any]:
    """Build a small workflow result for CLI output and failure tests."""
    return {
        'search': {
            'search_id': 1, 'status': status, 'sources': ['arxiv'],
            'source_results': {'arxiv': {'status': 'completed', 'result_count': 1}},
            'paper_ids': ['arxiv:1234.5678'] if paper_ids is None else paper_ids,
            'result_count': 1, 'papers_added': 1, 'papers_updated': 0,
            'abstracts_stored': 1,
        },
        'downloads': None,
    }


def test_gather_help_exposes_a_complete_acquisition_workflow() -> None:
    """Advertise both acquisition commands and their operational controls."""
    runner = CliRunner()
    root = runner.invoke(cli.main, ['--help'])
    assert root.exit_code == 0
    assert 'probe' in root.output and 'gather' in root.output
    result = runner.invoke(cli.main, ['gather', '--help'])
    assert result.exit_code == 0
    for option in ['--source', '--download-source', '--count', '--format', '--enrich',
                   '--no-abstract', '--force', '--parallel', '--workers', '--json']:
        assert option in result.output


def test_gather_selects_search_and_download_sources_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve separate discovery providers, fallback providers, and run controls."""
    seen: dict[str, Any] = {}

    def collect(query: str, db_path: str, **kwargs: Any) -> dict[str, Any]:
        """Record the requested gathering workflow."""
        seen.update(query=query, db_path=db_path, **kwargs)
        return _summary()

    monkeypatch.setattr(cli, 'gather_papers', collect)
    result = CliRunner().invoke(cli.main, [
        'gather', 'solid electrolyte', 'corpus.db', '--source', 'arxiv',
        '--source', 'openalex', '--count', '4', '--format', 'pdf',
        '--download-source', 'unpaywall', '--download-source', 'arxiv',
        '--no-abstract', '--enrich', '--parallel', '--workers', '2', '--force',
    ])
    assert result.exit_code == 0, result.output
    assert seen == {
        'query': 'solid electrolyte', 'db_path': 'corpus.db',
        'source': ('arxiv', 'openalex'), 'count': 4, 'download_format': 'pdf',
        'download_sources': ['unpaywall', 'arxiv'], 'download_abstract': False,
        'enrich': True, 'parallel': True, 'workers': 2, 'force': True,
    }
    assert '1 matching papers in corpus.db' in result.output


def test_gather_json_keeps_progress_out_of_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow direct JSON parsing even when search and download helpers print."""
    summary = _summary()

    def collect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Simulate workflow progress accompanying the result."""
        print('Searching papers...')
        print('Download complete.')
        return summary

    monkeypatch.setattr(cli, 'gather_papers', collect)
    result = CliRunner().invoke(cli.main, ['gather', 'query', '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == summary
    assert 'Searching papers...' in result.stderr
    assert 'Download complete.' in result.stderr


@pytest.mark.parametrize('status', ['partial', 'failed'])
def test_gather_incomplete_search_preserves_json_and_exits_nonzero(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep partial results consumable while signaling a provider failure to scripts."""
    summary = _summary(status)
    summary['search']['source_results']['openalex'] = {
        'status': 'failed', 'result_count': 0, 'error': 'provider unavailable',
    }
    monkeypatch.setattr(cli, 'gather_papers', lambda *args, **kwargs: summary)
    result = CliRunner().invoke(cli.main, ['gather', 'query', '--json'])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == summary
    assert f'Search {status}: openalex failed' in result.stderr
    assert 'Available results have been saved' in result.stderr


def test_gather_no_matches_is_successful(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinguish a successful empty search from a failed provider."""
    monkeypatch.setattr(cli, 'gather_papers', lambda *args, **kwargs: _summary(paper_ids=[]))
    result = CliRunner().invoke(cli.main, ['gather', 'query'])
    assert result.exit_code == 0
    assert 'No matching papers to download.' in result.output


@pytest.mark.parametrize('arguments', [
    [], [''], ['  '], ['query', '--count', '0'], ['query', '--count', '-1'],
    ['query', '--workers', '0'], ['query', '--source', 'unpaywall'],
    ['query', '--download-source', 'invalid'], ['query', '--format', 'invalid'],
])
def test_gather_invalid_arguments_never_start_work(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing queries and invalid controls before provider or database work."""
    def unexpected(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail if invalid input reaches the workflow."""
        pytest.fail('Invalid input must not run gathering.')

    monkeypatch.setattr(cli, 'gather_papers', unexpected)
    result = CliRunner().invoke(cli.main, ['gather', *arguments])
    assert result.exit_code == 2
    assert 'Error:' in result.output


@pytest.mark.parametrize('command', ['gather', 'search'])
@pytest.mark.parametrize('error', [
    ValueError('Missing provider configuration'), RuntimeError('Provider unavailable'),
    OSError('Cannot write corpus'), sqlite3.OperationalError('Database is locked'),
])
def test_acquisition_errors_are_reported_cleanly(
    command: str,
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Present expected provider and storage errors without a traceback."""
    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise an expected workflow error."""
        raise error

    target = 'gather_papers' if command == 'gather' else 'search_for_papers'
    monkeypatch.setattr(cli, target, fail)
    result = CliRunner().invoke(cli.main, [command, 'query'])
    assert result.exit_code == 1
    assert f'Error: {error}' in result.output
    assert 'Traceback' not in result.output


def test_search_accepts_multiple_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose source selection through the established search-only command too."""
    seen = {}

    def search(query: str, db_path: str, **kwargs: Any) -> dict[str, Any]:
        """Capture source selection and return successful metadata counts."""
        seen.update(kwargs)
        return _summary()['search']

    monkeypatch.setattr(cli, 'search_for_papers', search)
    result = CliRunner().invoke(cli.main, [
        'search', 'query', '--source', 'arxiv', '--source', 'pubmed', '--count', '5',
    ])
    assert result.exit_code == 0, result.output
    assert seen['source'] == ('arxiv', 'pubmed')
    assert seen['count'] == 5


@pytest.mark.parametrize('arguments', [['query', '--count', '0'], ['  ']])
def test_search_rejects_invalid_input_without_creating_a_corpus(
    arguments: list[str], tmp_path: Path,
) -> None:
    """Prevent blank or unbounded searches from leaving a database behind."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.main, ['search', *arguments])
        assert result.exit_code == 2
        assert not Path('papers.db').exists()


def test_config_provider_alias_supports_the_same_json_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the older configuration command compatible with top-level probing."""
    from paperminertoolkit.workflows import diagnostics

    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {})
    monkeypatch.setattr(diagnostics, 'provider_status', lambda *args, **kwargs: [
        diagnostics.ProviderStatus('arxiv', 'arXiv', '', diagnostics.NOT_PROBED,
                                   'no credential needed'),
    ])
    runner = CliRunner()
    arguments = ['--no-probe', '--source', 'arxiv', '--json']
    probe = runner.invoke(cli.main, ['probe', *arguments])
    legacy = runner.invoke(cli.main, ['config', 'providers', *arguments])
    assert legacy.exit_code == probe.exit_code == 0
    assert json.loads(legacy.stdout) == json.loads(probe.stdout)
