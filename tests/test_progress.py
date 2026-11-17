"""Check --progress jsonl event streams and progress-bar suppression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import paperminertoolkit.extraction.scrape as scrape
from paperminertoolkit import _progress as progress
from paperminertoolkit import cli
from tests.extraction.test_scrape import FakeModelConfig, sample_recipe, write_corpus
from tests.test_cli_json import DOWNLOAD_SUMMARY


def _events(stderr: str) -> list[dict[str, Any]]:
    """Parse every JSON event line, checking that each one is a complete object."""
    events = [json.loads(line) for line in stderr.splitlines() if line.startswith('{')]
    assert all({'event', 'command', 'elapsed_s'} <= set(event) for event in events)
    return events


def test_jsonl_mode_writes_events_and_hides_bars(capsys: pytest.CaptureFixture[str]) -> None:
    """Emit events only in jsonl mode, and hide bars in every mode but auto."""
    progress.emit('ignored')
    assert not progress.bars_disabled()
    with progress.progress_mode('none', 'test'):
        assert progress.bars_disabled()
        progress.emit('ignored')
    with progress.progress_mode('jsonl', 'test'):
        assert progress.bars_disabled()
        progress.emit('paper_done', paper_id='p:1', done=1, total=2)
    assert not progress.bars_disabled()
    events = _events(capsys.readouterr().err)
    assert events == [{'event': 'paper_done', 'command': 'test',
                       'elapsed_s': events[0]['elapsed_s'], 'paper_id': 'p:1', 'done': 1,
                       'total': 2}]


def test_jsonl_mode_reports_the_error_that_ends_a_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Close the stream with an error event before the exception propagates."""
    with pytest.raises(RuntimeError, match='boom'):
        with progress.progress_mode('jsonl', 'test'):
            raise RuntimeError('boom')
    assert _events(capsys.readouterr().err)[-1]['event'] == 'error'
    with pytest.raises(ValueError, match='progress mode must be one of'):
        with progress.progress_mode('loud', 'test'):
            pass


def test_scrape_streams_one_event_per_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Report run start, each paper with its outcome, and run end, with bars hidden."""
    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'missing'},
                           {'paper_id': 'done', 'text_scrape_status': 'succeeded'}])
    bars: list[bool] = []

    class RecordingTqdm:
        """Progress bar double that records whether it was disabled."""

        def __init__(self, *args: Any, disable: bool = False, **kwargs: Any) -> None:
            """Remember the disable flag."""
            bars.append(disable)

        def __enter__(self) -> RecordingTqdm:
            """Return the bar."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Propagate exceptions."""
            return False

        def update(self, amount: int) -> None:
            """Accept progress updates."""

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', RecordingTqdm)
    with progress.progress_mode('jsonl', 'scrape'):
        scrape.scrape_papers(str(db_path), output_path=str(tmp_path / 'out.csv'))

    events = _events(capsys.readouterr().err)
    assert [event['event'] for event in events] == ['run_start', 'paper_done', 'paper_done',
                                                   'run_end']
    assert events[0]['total'] == 2
    assert {event['paper_id']: event['status'] for event in events[1:3]} == {
        'missing': 'failed', 'done': 'skipped'}
    assert [event['done'] for event in events[1:3]] == [1, 2]
    assert events[-1]['failed'] == 1
    assert bars == [True]


def test_cli_progress_option_streams_events_beside_the_json_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the JSON result on stdout while events and chatter share stderr."""
    def workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Emit events and print chatter like the real download."""
        progress.emit('run_start', total=1)
        print('Download complete: 1 PDF.')
        progress.emit('paper_done', paper_id='p:1', done=1, total=1)
        return DOWNLOAD_SUMMARY

    monkeypatch.setattr(cli, 'download_papers', workflow)
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    result = CliRunner().invoke(cli.main, ['download', str(db_path), '--progress', 'jsonl',
                                           '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == DOWNLOAD_SUMMARY
    events = _events(result.stderr)
    assert [event['event'] for event in events] == ['run_start', 'paper_done']
    assert {event['command'] for event in events} == {'download'}
    assert 'Download complete: 1 PDF.' in result.stderr

    quiet = CliRunner().invoke(cli.main, ['download', str(db_path), '--json'])
    assert _events(quiet.stderr) == []
