"""Offline regressions for gathering only the canonical matches from a search."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from paperminertoolkit.corpus import database as corpus, filtering
from paperminertoolkit.workflows import download, gather, search


def _frame(source: str, papers: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build normalized search results while preserving supplied abstracts."""
    rows = []
    for paper in papers:
        row = corpus.normalize_paper({**paper, 'sources': source})
        row['abstract'] = paper.get('abstract', '')
        rows.append(row)
    return pd.DataFrame(rows, columns=search.SEARCH_FIELDS)


def _stub_searches(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: Mapping[str, pd.DataFrame | Exception],
) -> list[str]:
    """Replace provider requests while keeping actual corpus merge and history."""
    calls: list[str] = []

    def resolve(name: str) -> Callable[..., pd.DataFrame]:
        """Return the prepared provider response or provider failure."""
        def run(query: str, count: int) -> pd.DataFrame:
            """Record the requested source and produce its configured result."""
            calls.append(name)
            result = outcomes[name]
            if isinstance(result, Exception):
                raise result
            return result

        return run

    monkeypatch.setattr(search, '_source_search', resolve)
    return calls


def _stub_pdfs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Write local PDF fixtures instead of fetching provider content."""
    calls: list[str] = []

    def fetch(
        paper: Mapping[str, Any],
        filepath: str | Path,
        sources: Iterable[str],
    ) -> tuple[bool, str, str]:
        """Write a recognizable PDF asset and record the canonical paper ID."""
        calls.append(paper['paper_id'])
        Path(filepath).write_bytes(b'%PDF-1.7 fixture')
        return True, 'core', 'https://example.test/paper.pdf'

    monkeypatch.setattr(download, '_download_pdf_from_sources', fetch)
    return calls


def test_gather_deduplicates_search_matches_and_reuses_existing_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve provider aliases to canonical IDs and leave unrelated rows untouched."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'canonical:old', 'doi': '10.1234/old'})
        corpus.upsert_paper(conn, {'paper_id': 'unrelated', 'doi': '10.1234/other'})
        unrelated_before = corpus.find_paper(conn, {'paper_id': 'unrelated'})
    outcomes = {
        name: _frame(name, [
            {'paper_id': f'{name}:old', 'doi': '10.1234/old', 'abstract': 'Old abstract.'},
            {'paper_id': f'{name}:new', 'doi': '10.1234/new', 'abstract': 'New abstract.'},
        ])
        for name in ['core', 'openalex']
    }
    searches = _stub_searches(monkeypatch, outcomes)
    pdfs = _stub_pdfs(monkeypatch)

    first = gather.gather_papers(
        'materials', db_path, source=['core', 'openalex', 'core'],
        download_format='pdf', download_sources=['core'],
    )

    summary = first['search']
    assert searches == ['core', 'openalex']
    assert summary['paper_ids'] == ['canonical:old', 'core:new']
    assert summary['status'] == 'completed'
    assert summary['result_count'] == 4
    assert summary['papers_added'] == 1
    assert summary['papers_updated'] == 3
    assert summary['abstracts_stored'] == 4
    assert pdfs == ['canonical:old', 'core:new']
    assert first['downloads']['pdfs'] == 2
    assert first['downloads']['abstracts_skipped'] == 2
    with corpus.connect(db_path) as conn:
        links = conn.execute(
            'SELECT paper_id, source FROM paper_search_results WHERE search_id = ?',
            (summary['search_id'],),
        ).fetchall()
        assert {(row['paper_id'], row['source']) for row in links} == {
            (paper_id, source)
            for paper_id in summary['paper_ids'] for source in ['core', 'openalex']
        }
        assert corpus.find_paper(conn, {'paper_id': 'unrelated'}) == unrelated_before
        assert corpus.get_asset(conn, 'unrelated', 'pdf') is None
        assert corpus.get_asset(conn, 'canonical:old', 'pdf')['content'] == b'%PDF-1.7 fixture'

    second = gather.gather_papers(
        'materials', db_path, source=['core', 'openalex'],
        download_format='pdf', download_sources=['core'],
    )
    assert second['search']['paper_ids'] == summary['paper_ids']
    assert second['search']['papers_added'] == 0
    assert second['downloads']['pdfs'] == 0
    assert second['downloads']['pdfs_skipped'] == 2
    assert pdfs == ['canonical:old', 'core:new']


@pytest.mark.parametrize('format_name,store_abstract', [('pdf', False), ('abstract', True)])
def test_gather_abstract_format_overrides_disabled_abstract_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    store_abstract: bool,
) -> None:
    """Request abstract assets for abstract-only gathers and honor opt-out elsewhere."""
    db_path = tmp_path / 'papers.db'
    _stub_searches(monkeypatch, {
        'core': _frame('core', [{'paper_id': 'core:one', 'abstract': 'From search.'}]),
    })
    pdfs = _stub_pdfs(monkeypatch)
    result = gather.gather_papers(
        'materials', db_path, source='core', download_format=format_name,
        download_sources=['core'], download_abstract=False,
    )
    with corpus.connect(db_path) as conn:
        asset = corpus.get_asset(conn, 'core:one', 'abstract')
    assert (asset is not None) is store_abstract
    assert result['search']['abstracts_stored'] == int(store_abstract)
    assert pdfs == (['core:one'] if format_name == 'pdf' else [])


@pytest.mark.parametrize('status', ['completed', 'partial', 'failed'])
def test_gather_reports_search_outcomes_without_downloading_unrelated_papers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    """Distinguish empty, partial, and failed searches without a corpus-wide download."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'unrelated'})
    empty = _frame('core', [])
    match = _frame('core', [{'paper_id': 'core:one'}])
    _stub_searches(monkeypatch, {
        'core': match if status == 'partial' else RuntimeError('core offline') if status == 'failed' else empty,
        'openalex': empty if status == 'completed' else RuntimeError('openalex offline'),
    })
    pdfs = _stub_pdfs(monkeypatch)

    result = gather.gather_papers(
        'materials', db_path, source=['core', 'openalex'],
        download_format='pdf', download_sources=['core'], download_abstract=False,
    )

    assert result['search']['status'] == status
    assert pdfs == (['core:one'] if status == 'partial' else [])
    if status != 'partial':
        assert result['downloads'] is None
    with corpus.connect(db_path) as conn:
        assert corpus.search_history(conn)[0]['status'] == status
        assert corpus.get_asset(conn, 'unrelated', 'pdf') is None


@pytest.mark.parametrize('kwargs,message', [
    ({'query': '  '}, 'query must not be blank'),
    ({'count': 0}, 'count must be at least 1'),
    ({'workers': 0}, 'workers must be at least 1'),
    ({'source': ['core', 'unknown']}, 'search source'),
    ({'download_format': 'unknown'}, 'download_format'),
    ({'download_sources': ['unknown']}, 'download source'),
])
def test_gather_rejects_invalid_options_before_writing_or_searching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    """Validate user input before creating search history or making provider calls."""
    db_path = tmp_path / 'papers.db'
    calls = _stub_searches(monkeypatch, {})
    with pytest.raises(ValueError, match=message):
        gather.gather_papers(**{'query': 'materials', 'db_path': db_path, **kwargs})
    assert not db_path.exists()
    assert calls == []


def test_download_explicit_subset_respects_filters_and_empty_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip excluded, unevaluated, unknown, and unselected rows in scoped downloads."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        for paper_id, title in [('included', 'Keep'), ('unselected', 'Keep'), ('excluded', 'Drop')]:
            corpus.upsert_paper(conn, {'paper_id': paper_id, 'title': title})
    rules = tmp_path / 'rules.json'
    rules.write_text(json.dumps({
        'name': 'keep', 'fields': ['title'],
        'include': [{'name': 'keep', 'pattern': 'Keep'}],
    }), encoding='utf-8')
    filtering.apply_regex_filter(db_path, rules)
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'unevaluated', 'title': 'Keep'})
    calls = _stub_pdfs(monkeypatch)

    summary = download.download_papers(
        db_path, download_format='pdf', sources=['core'], download_abstract=False,
        paper_ids=(name for name in ['included', 'included', 'excluded', 'unevaluated', 'unknown']),
    )
    assert calls == ['included']
    assert summary['pdfs'] == 1
    empty = download.download_papers(
        db_path, download_format='pdf', sources=['core'], force=True, paper_ids=[],
    )
    assert not any(empty.values())
    filtered_out = download.download_papers(
        db_path, download_format='pdf', sources=['core'], force=True,
        paper_ids=['excluded', 'unevaluated', 'unknown'],
    )
    assert not any(filtered_out.values())
    assert calls == ['included']
    with corpus.connect(db_path) as conn:
        assert corpus.get_asset(conn, 'excluded', 'pdf') is None
        assert corpus.get_asset(conn, 'unselected', 'pdf') is None
        assert corpus.get_asset(conn, 'unevaluated', 'pdf') is None
