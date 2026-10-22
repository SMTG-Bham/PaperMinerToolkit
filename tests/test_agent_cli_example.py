"""Exercise the agent subprocess example without contacting providers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import pytest

from examples import agent_cli


def test_example_preserves_partial_results_and_passes_safe_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep exit-one results while inspecting their exact search and first match."""
    search = {'status': 'partial', 'search_id': 42, 'paper_ids': ['paper:1']}
    listing = {'papers': [{'paper_id': 'paper:1'}], 'total': 1, 'limit': 20, 'offset': 0}
    paper = {'paper': {'paper_id': 'paper:1'}}
    responses = iter([(search, 1), (listing, 0), (paper, 0)])
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Return ordered search and inspection payloads and capture invocation options."""
        calls.append((argv, kwargs))
        payload, code = next(responses)
        return subprocess.CompletedProcess(argv, code, json.dumps(payload), 'progress\n')

    monkeypatch.setattr(agent_cli.subprocess, 'run', run)
    query = 'lithium; $(echo untrusted)'
    corpus = tmp_path / 'corpus files' / 'papers.db'
    code = agent_cli.main([
        query, '--corpus', str(corpus), '--source', 'arxiv', '--source', 'pubmed',
        '--count', '3', '--timeout', '12',
    ])

    captured = capsys.readouterr()
    assert code == 1
    assert json.loads(captured.out) == {
        'search': search, 'corpus': listing, 'first_match': paper,
    }
    assert 'Search incomplete' in captured.err
    assert captured.err.count('progress\n') == 3
    assert [argv[3:] for argv, _ in calls] == [
        ['search', query, str(corpus), '--source', 'arxiv', '--source', 'pubmed',
         '--count', '3', '--store-abstract', '--json'],
        ['corpus', 'list', str(corpus), '--limit', '20', '--offset', '0',
         '--search-id', '42', '--json'],
        ['corpus', 'show', str(corpus), 'paper:1', '--json'],
    ]
    for argv, options in calls:
        assert isinstance(argv, list)
        assert argv[:3] == [sys.executable, '-m', 'paperminertoolkit']
        assert options['stdin'] == subprocess.DEVNULL
        assert options.get('shell', False) is False
        assert options['timeout'] == 12
        assert options['check'] is False


@pytest.mark.parametrize('stdout', ['', 'not JSON'])
def test_example_handles_missing_or_invalid_json(
    stdout: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stop on early failures without pretending an empty payload is successful."""
    calls = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Simulate a provider failure before a valid result was emitted."""
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout, 'Provider unavailable\n')

    monkeypatch.setattr(agent_cli.subprocess, 'run', run)
    code = agent_cli.main(['query', '--corpus', str(tmp_path / 'papers.db')])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ''
    assert 'Provider unavailable' in captured.err
    assert 'JSON' in captured.err
    assert len(calls) == 1


@pytest.mark.parametrize('arguments', [[], ['--help']])
def test_example_help_runs_no_commands(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Make invocation without a workload safe for discovery by an agent."""
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        """Reject subprocess work during help rendering."""
        raise AssertionError('Help must not execute a command.')

    monkeypatch.setattr(agent_cli.subprocess, 'run', forbidden)
    if arguments:
        with pytest.raises(SystemExit) as error:
            agent_cli.main(arguments)
        assert error.value.code == 0
    else:
        assert agent_cli.main(arguments) == 0
    assert '--corpus' in capsys.readouterr().out


@pytest.mark.parametrize('arguments', [
    ['query', '--count', '0'], ['query', '--count', '1001'], ['   '],
    ['query', '--timeout', '0'],
])
def test_example_rejects_invalid_workloads_before_running_commands(
    arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate bounded counts, nonblank queries, and timeouts before any subprocess."""
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        """Reject subprocess work for invalid user input."""
        raise AssertionError('Invalid input must not execute a command.')

    monkeypatch.setattr(agent_cli.subprocess, 'run', forbidden)
    with pytest.raises(SystemExit) as error:
        agent_cli.main([*arguments, '--corpus', str(tmp_path / 'papers.db')])
    assert error.value.code == 2


def test_example_reports_unusable_output_directory_before_running_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Treat an invalid corpus parent as an operational error before acquisition."""
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        """Reject any provider command after path preparation has failed."""
        raise AssertionError('Invalid output path must not execute a command.')

    parent = tmp_path / 'file-not-directory'
    parent.write_text('keep this file')
    monkeypatch.setattr(agent_cli.subprocess, 'run', forbidden)
    assert agent_cli.main(['query', '--corpus', str(parent / 'papers.db')]) == 1
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err
    assert parent.read_text() == 'keep this file'
