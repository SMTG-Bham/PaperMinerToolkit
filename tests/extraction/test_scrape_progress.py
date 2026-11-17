"""Test the scrape progress report behind ``pmt scrape-progress``."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import paperminertoolkit.corpus.database as corpus
from paperminertoolkit import cli
from paperminertoolkit.corpus import filtering
from paperminertoolkit.corpus.supplements import add_supplement, set_supplement_extraction_status
from paperminertoolkit.extraction import scrape_progress as progress


@pytest.fixture
def scraped_corpus(tmp_path: Path) -> Path:
    """Build five papers, one filtered out, and three SI files of one paper."""
    db_path = tmp_path / 'papers.db'
    rows = [
        {'paper_id': 'a', 'title': 'Keep a', 'text_scrape_status': 'succeeded'},
        {'paper_id': 'b', 'title': 'Keep b', 'text_scrape_status': 'failed',
         'last_error': 'No downloaded text or PDF asset found for text scrape.'},
        {'paper_id': 'c', 'title': 'Keep c', 'text_scrape_status': 'failed',
         'last_error': 'Local model request failed: 400'},
        {'paper_id': 'd', 'title': 'Keep d'},
        {'paper_id': 'e', 'title': 'Drop e'},
    ]
    with closing(corpus.connect(db_path)) as conn, conn:
        for row in rows:
            corpus.upsert_paper(conn, row)
        for index, status in enumerate(['failed', 'succeeded', 'unsupported']):
            blob_id = add_supplement(conn, rows[0], f'table {index}'.encode(),
                                     original_filename=f's{index}.txt')
            set_supplement_extraction_status(conn, 'a', blob_id, 'text', status, members={})
    rules = tmp_path / 'rules.json'
    rules.write_text(json.dumps({'name': 'keep', 'fields': ['title'],
                                 'include': [{'name': 'keep', 'pattern': 'Keep'}]}),
                     encoding='utf-8')
    filtering.apply_regex_filter(db_path, rules)
    return db_path


def test_progress_counts_only_selected_papers_and_work_a_rerun_would_do(scraped_corpus: Path) -> None:
    """Separate papers without a document from those a rerun retries, with their IDs."""
    result = progress.scrape_progress(scraped_corpus, 'text')

    assert result['scope'] == 4
    assert result['out_of_scope'] == 1
    assert result['stages']['text'] == {'succeeded': 1, 'no_asset': 1, 'failed': 1, 'not_attempted': 1}
    assert result['supplements']['text'] == {'failed': 1, 'succeeded': 1, 'unsupported': 1}
    assert result['remaining'] == {'papers': 2, 'supplement_files': 1}
    assert result['failed_papers'] == {'text': ['c']}
    assert result['no_asset_papers'] == {'text': ['b']}

    text = progress.report(result)
    assert '4 papers are included by the active corpus filters; 1 excluded or unavailable' in text
    assert 'A rerun would retry 2 papers and 1 supporting-information file.' in text
    assert 'running the same pmt scrape command again' in text
    assert '1 selected paper without downloaded text or PDF' in text
    assert 'Corpus fully scraped' not in text
    custom = progress.report(result, rerun='sbatch job.sbatch')
    assert 'submitting the same job again:\n  sbatch job.sbatch' in custom


def test_progress_reports_a_finished_corpus_and_rejects_unknown_modes(scraped_corpus: Path) -> None:
    """Say the corpus is done when nothing a rerun can do remains."""
    result: dict[str, Any] = progress.scrape_progress(scraped_corpus, 'text')
    result['remaining'] = {'papers': 0, 'supplement_files': 0}

    assert 'Corpus fully scraped' in progress.report(result)
    with pytest.raises(ValueError, match='mode must be one of'):
        progress.scrape_progress(scraped_corpus, 'everything')


def test_scrape_progress_command_prints_json_and_text(scraped_corpus: Path) -> None:
    """Expose the report and its paper IDs through pmt scrape-progress."""
    runner = CliRunner()
    result = runner.invoke(cli.main, ['scrape-progress', str(scraped_corpus), 'text', '--json'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload['mode'] == 'text'
    assert payload['failed_papers'] == {'text': ['c']}

    human = runner.invoke(cli.main, ['scrape-progress', str(scraped_corpus)])
    assert human.exit_code == 0, human.output
    assert 'text_scrape_status: 1 succeeded' in human.stdout

    bad = runner.invoke(cli.main, ['scrape-progress', str(scraped_corpus), 'everything'])
    assert bad.exit_code == 2
