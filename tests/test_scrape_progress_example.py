"""Check the Gaudi scrape progress report against a stand-in corpus."""

from __future__ import annotations

from typing import Any

import pytest

from examples.sol_gaudi import scrape_progress as progress


class Connection:
    """Stand in for a corpus connection that only needs closing."""

    def close(self) -> None:
        """Accept the close the report makes."""


@pytest.fixture
def corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve five papers, one filtered out, and three supplementary files of one paper."""
    papers = [
        {'paper_id': 'a', 'text_scrape_status': 'succeeded', 'last_error': ''},
        {'paper_id': 'b', 'text_scrape_status': 'failed',
         'last_error': 'No downloaded text or PDF asset found for text scrape.'},
        {'paper_id': 'c', 'text_scrape_status': 'failed', 'last_error': 'Local model request failed: 400'},
        {'paper_id': 'd', 'text_scrape_status': None, 'last_error': None},
        {'paper_id': 'e', 'text_scrape_status': None, 'last_error': None},
    ]
    supplements = {'a': [{'metadata': {'text_extraction_status': status}}
                         for status in ('failed', 'succeeded', 'unsupported')]}
    monkeypatch.setattr(progress, 'connect', lambda path: Connection())
    monkeypatch.setattr(progress, 'paper_rows', lambda conn: papers)
    monkeypatch.setattr(progress, 'active_filter_stack', lambda conn: [{'name': 'degradation text'}])
    monkeypatch.setattr(progress, 'current_filter_statuses',
                        lambda conn: {'a': 'included', 'b': 'included', 'c': 'included', 'd': 'included',
                                      'e': 'excluded'})
    monkeypatch.setattr(progress, 'get_supplements',
                        lambda conn, paper_id, include_content=False: supplements.get(paper_id, []))


def test_progress_counts_only_selected_papers_and_work_a_rerun_would_do(corpus: None) -> None:
    """Separate papers without a document from those a resubmission retries."""
    result = progress.scrape_progress('papers.db', 'text')

    assert result['scope'] == 4
    assert result['out_of_scope'] == 1
    assert result['stages']['text'] == {'succeeded': 1, 'no_asset': 1, 'failed': 1, 'not_attempted': 1}
    assert result['supplements']['text'] == {'failed': 1, 'succeeded': 1, 'unsupported': 1}
    assert result['remaining'] == {'papers': 2, 'supplement_files': 1}

    text = progress.report(result, '/data/papers.db')
    assert '4 papers are included by the active corpus filters; 1 excluded or unavailable' in text
    assert 'A resubmission would retry 2 papers and 1 supporting-information file.' in text
    assert 'sbatch --export=ALL,PMT_DB=/data/papers.db scrape_gaudi.sbatch' in text
    assert '1 selected paper without downloaded text or PDF' in text
    assert 'Corpus fully scraped' not in text


def test_progress_reports_a_finished_corpus_and_rejects_unknown_modes(corpus: None) -> None:
    """Say the corpus is done when nothing a rerun can do remains."""
    result: dict[str, Any] = progress.scrape_progress('papers.db', 'text')
    result['remaining'] = {'papers': 0, 'supplement_files': 0}

    assert 'Corpus fully scraped' in progress.report(result, 'papers.db')
    with pytest.raises(ValueError, match='mode must be one of'):
        progress.scrape_progress('papers.db', 'everything')
