"""Exercise SI acquisition through the general corpus download pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from paperminertoolkit.corpus import database as corpus
from paperminertoolkit.corpus.supplements import get_supplements
from paperminertoolkit.providers import supplements as provider
from paperminertoolkit.workflows import download


def test_supplement_only_discovers_stored_jats_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquire SI from retained article XML without fetching article or abstract."""
    path = tmp_path / 'papers.db'
    with corpus.connect(path) as conn:
        corpus.add_structured_document(
            conn, {'paper_id': 'one'},
            '<article xmlns:xlink="http://www.w3.org/1999/xlink"><back>'
            '<supplementary-material xlink:href="table.csv"><label>Table S1</label>'
            '</supplementary-material></back></article>', document_format='jats',
            source='pubmed', metadata={'source_url': 'https://example.org/article.xml'},
        )
    calls = []

    def fetch(candidate: provider.SupplementCandidate, session: Any = None) -> provider.DownloadedSupplement:
        """Return the supplementary table for a discovered reference."""
        calls.append(candidate.url)
        return provider.DownloadedSupplement(b'sample,value\nA,42\n', 'table.csv', 'text/csv', candidate.url)

    def reject(*args: Any, **kwargs: Any) -> None:
        """Refuse any unintended abstract or article lookup."""
        raise AssertionError('SI-only must not download main article assets')

    monkeypatch.setattr(provider, 'download_supplement', fetch)
    monkeypatch.setattr(download, '_download_abstract', reject)
    monkeypatch.setattr(download, '_configured_sources', reject)
    first = download.download_papers(path, download_format='supplement', supplement_sources=['structured'])
    second = download.download_papers(path, download_format='supplement', supplement_sources=['structured'])
    assert first['supplements'] == 1
    assert second['supplements'] == 0
    assert second['supplements_skipped'] == 1
    assert calls == ['https://example.org/table.csv']
    assert first['abstracts'] == first['texts'] == first['pdfs'] == 0
    with corpus.connect(path) as conn:
        assert get_supplements(conn, 'one')[0]['content'] == b'sample,value\nA,42\n'
        assert corpus.paper_rows(conn)[0]['supplement_download_status'] == 'succeeded'


@pytest.mark.parametrize(('format_name', 'enabled', 'expected'), [
    ('text', True, 1), ('pdf', True, 1), ('both', True, 1),
    ('abstract', True, 0), ('pdf', False, 0), ('supplement', False, 1),
])
def test_si_runs_by_default_for_article_downloads_with_explicit_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    format_name: str, enabled: bool, expected: int,
) -> None:
    """Run SI for eligible formats while honoring abstract-only and opt-out runs."""
    path = tmp_path / 'papers.db'
    with corpus.connect(path) as conn:
        paper = {'paper_id': 'one'}
        for role in ('text', 'pdf', 'abstract'):
            corpus.add_asset(conn, paper, role, role, role, 'text/plain')
    calls = []

    def acquire(conn: sqlite3.Connection, paper: dict[str, Any], **kwargs: Any) -> dict[str, int]:
        """Record independent SI source selection for one paper."""
        calls.append((paper['paper_id'], kwargs['sources']))
        return {'supplements': 2}

    monkeypatch.setattr(download, 'download_paper_supplements', acquire)
    monkeypatch.setattr(download, '_configured_sources', lambda values: ['pubmed'])
    summary = download.download_papers(
        path, download_format=format_name, download_supplements=enabled,
        supplement_sources=['structured'],
    )
    assert len(calls) == expected
    assert summary['supplements'] == 2 * expected
    if calls:
        assert calls == [('one', ['structured'])]


def test_si_failure_isolated_and_selected_papers_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue subsequent selected papers after an unexpected SI failure."""
    path = tmp_path / 'papers.db'
    with corpus.connect(path) as conn:
        for paper_id in ('one', 'two', 'unrelated'):
            corpus.upsert_paper(conn, {'paper_id': paper_id})
    calls = []

    def acquire(conn: sqlite3.Connection, paper: dict[str, Any], **kwargs: Any) -> dict[str, int]:
        """Fail one selected paper and allow the next one to continue."""
        calls.append(paper['paper_id'])
        if paper['paper_id'] == 'one':
            raise RuntimeError('temporary SI service failure')
        return {'supplements': 1}

    monkeypatch.setattr(download, 'download_paper_supplements', acquire)
    summary = download.download_papers(path, download_format='supplement', paper_ids=['one', 'two'])
    assert calls == ['one', 'two']
    assert summary['supplement_discovery_failed'] == summary['supplements'] == 1
    with corpus.connect(path) as conn:
        papers = {p['paper_id']: p for p in corpus.paper_rows(conn)}
        assert papers['one']['supplement_download_status'] == 'failed'
        assert papers['unrelated']['supplement_download_status'] == 'pending'
