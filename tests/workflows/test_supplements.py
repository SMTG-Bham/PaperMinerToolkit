"""Regression tests for resumable supplementary-file acquisition in the corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paperminertoolkit.corpus import database as corpus
from paperminertoolkit.corpus.supplements import get_supplements
from paperminertoolkit.providers.supplements import DiscoveryResult, DownloadedSupplement, SupplementCandidate
from paperminertoolkit.workflows import supplements


def candidate(name: str = 's1.pdf', url: str = '') -> SupplementCandidate:
    """Build one explicit supporting PDF candidate."""
    return SupplementCandidate(url or f'https://files.example/{name}', 'structured', name)


def downloaded(item: SupplementCandidate) -> DownloadedSupplement:
    """Return distinct valid supporting bytes for an explicit candidate."""
    return DownloadedSupplement(b'%PDF-1.7 ' + item.url.encode(), item.filename, 'application/pdf', item.url)


def test_resume_skips_successful_files_and_retries_failed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial run stores good files and retries only missing candidates."""
    items = [candidate('s1.pdf'), candidate('s2.pdf')]
    monkeypatch.setattr(supplements.provider, 'discover_supplements', lambda *_args: DiscoveryResult(items))
    calls: list[str] = []

    def fetch(item: SupplementCandidate) -> DownloadedSupplement:
        """Fail one file once while retaining the first successful download."""
        calls.append(item.url)
        if item == items[1] and calls.count(item.url) == 1:
            raise RuntimeError('HTTP 503')
        return downloaded(item)

    monkeypatch.setattr(supplements.provider, 'download_supplement', fetch)
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        paper = {'paper_id': 'doi:10.1234/paper', 'doi': '10.1234/paper'}
        corpus.upsert_paper(conn, paper)
        first = supplements.download_paper_supplements(conn, paper, ['structured'])
        assert first['supplements'] == 1 and first['supplements_failed'] == 1
        assert paper['supplement_download_status'] == 'partial'
        assert len(get_supplements(conn, paper['paper_id'])) == 1
        second = supplements.download_paper_supplements(conn, paper, ['structured'])
        assert second['supplements'] == 1 and second['supplements_skipped'] == 1
        assert paper['supplement_download_status'] == 'succeeded'
        assert paper['metadata']['supplement_download_error'] == ''
        assert len(get_supplements(conn, paper['paper_id'])) == 2
        third = supplements.download_paper_supplements(conn, paper, ['structured'])
        assert third['supplements_skipped'] == 2
        assert len(calls) == 3


def test_duplicate_content_keeps_one_asset_and_all_source_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror URLs sharing bytes store one SI asset with resumable aliases."""
    items = [candidate(), candidate('mirror.pdf', 'https://mirror.example/si.pdf')]
    monkeypatch.setattr(supplements.provider, 'discover_supplements', lambda *_args: DiscoveryResult(items))
    monkeypatch.setattr(supplements.provider, 'download_supplement', lambda item: DownloadedSupplement(b'%PDF-1.7 same', item.filename, 'application/pdf', item.url))
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        paper = {'paper_id': 'doi:10.1234/paper'}
        corpus.upsert_paper(conn, paper)
        result = supplements.download_paper_supplements(conn, paper, ['structured'])
        assert result['supplements'] == 1
        assert result['supplements_skipped'] == 1
        assets = get_supplements(conn, paper['paper_id'])
        assert len(assets) == 1
        assert set(assets[0]['metadata']['source_urls']) == {item.url for item in items}


def test_discovery_error_is_failed_not_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No downloaded files after a discovery failure does not mean no SI exists."""
    monkeypatch.setattr(supplements.provider, 'discover_supplements', lambda *_args: DiscoveryResult(errors=['figshare: HTTP 503']))
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        paper = {'paper_id': 'doi:10.1234/paper'}
        corpus.upsert_paper(conn, paper)
        result = supplements.download_paper_supplements(conn, paper, ['figshare'])
        assert result['supplements_unavailable'] == 0
        assert result['supplements_failed'] == 0
        assert paper['supplement_download_status'] == 'failed'
        metadata = json.loads(conn.execute('SELECT metadata_json FROM papers').fetchone()[0])
        assert metadata['supplement_download_error'] == 'figshare: HTTP 503'


def test_unavailable_is_cached_until_inputs_change_or_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """New supplied SI URLs and enriched identifiers invalidate no-SI results."""
    calls: list[dict[str, Any]] = []

    def discover(paper: dict[str, Any], *_args: Any) -> DiscoveryResult:
        """Record the discovery inputs to verify cache invalidation."""
        calls.append(dict(paper))
        return DiscoveryResult()

    monkeypatch.setattr(supplements.provider, 'discover_supplements', discover)
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        paper = {'paper_id': 'paper:1'}
        corpus.upsert_paper(conn, paper)
        assert supplements.download_paper_supplements(conn, paper, ['structured'])['supplements_unavailable'] == 1
        supplements.download_paper_supplements(conn, paper, ['structured'])
        assert len(calls) == 1
        paper['metadata']['supplementary_urls'] = ['https://files.example/new.pdf']
        supplements.download_paper_supplements(conn, paper, ['structured'])
        assert len(calls) == 2
        assert calls[-1]['supplementary_urls'] == ['https://files.example/new.pdf']
        paper['doi'] = '10.1234/enriched'
        supplements.download_paper_supplements(conn, paper, ['structured'])
        assert len(calls) == 3
        supplements.download_paper_supplements(conn, paper, ['structured'], force=True)
        assert len(calls) == 4


def test_success_keeps_article_error_and_force_refreshes_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SI success preserves an unrelated main-article failure and supports refresh."""
    item = candidate()
    monkeypatch.setattr(supplements.provider, 'discover_supplements', lambda *_args: DiscoveryResult([item]))
    monkeypatch.setattr(supplements.provider, 'download_supplement', downloaded)
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        paper = {'paper_id': 'doi:10.1234/paper', 'last_error': 'article PDF refused', 'pdf_download_status': 'failed'}
        corpus.upsert_paper(conn, paper)
        supplements.download_paper_supplements(conn, paper, ['structured'])
        assert paper['last_error'] == 'article PDF refused'
        monkeypatch.setattr(supplements.provider, 'download_supplement', lambda item: DownloadedSupplement(b'%PDF-1.7 changed', item.filename, 'application/pdf', item.url))
        supplements.download_paper_supplements(conn, paper, ['structured'], force=True)
        assets = get_supplements(conn, paper['paper_id'])
        assert len(assets) == 1
        assert assets[0]['content'] == b'%PDF-1.7 changed'
        assert paper['last_error'] == 'article PDF refused'


def test_missing_old_candidate_is_retried_after_discovery_outage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An earlier failed file cannot silently vanish from a partial retry."""
    item = candidate()
    monkeypatch.setattr(supplements.provider, 'discover_supplements', lambda *_args: DiscoveryResult([item]))

    def fail(_item: SupplementCandidate) -> DownloadedSupplement:
        """Refuse the candidate for the first run."""
        raise RuntimeError('offline')

    monkeypatch.setattr(supplements.provider, 'download_supplement', fail)
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        paper = {'paper_id': 'doi:10.1234/paper'}
        corpus.upsert_paper(conn, paper)
        supplements.download_paper_supplements(conn, paper, ['structured'])
        monkeypatch.setattr(supplements.provider, 'discover_supplements', lambda *_args: DiscoveryResult(errors=['metadata down']))
        monkeypatch.setattr(supplements.provider, 'download_supplement', downloaded)
        summary = supplements.download_paper_supplements(conn, paper, ['structured'])
        assert summary['supplements'] == 1
        assert paper['supplement_download_status'] == 'partial'


def test_source_normalization_validates_and_preserves_registry_order() -> None:
    """Source validation happens before any database or network mutation."""
    assert supplements.normalize_supplement_sources(['all']) == list(supplements.SUPPLEMENT_SOURCES)
    assert supplements.normalize_supplement_sources(['publisher', 'structured', 'structured']) == ['structured', 'publisher']
    with pytest.raises(ValueError, match='Unknown supplementary sources'):
        supplements.normalize_supplement_sources(['made-up'])
    with pytest.raises(ValueError, match='At least one'):
        supplements.normalize_supplement_sources([])


def test_force_refresh_merges_into_existing_content_and_removes_stale_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When A changes to B's bytes, retain B's extraction state and remove old A."""
    items = [candidate('a.pdf'), candidate('b.pdf')]
    monkeypatch.setattr(supplements.provider, 'discover_supplements', lambda *_args: DiscoveryResult(items))
    monkeypatch.setattr(supplements.provider, 'download_supplement', downloaded)
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        paper = {'paper_id': 'doi:10.1234/paper'}
        corpus.upsert_paper(conn, paper)
        supplements.download_paper_supplements(conn, paper, ['structured'])
        monkeypatch.setattr(supplements.provider, 'download_supplement', lambda item: downloaded(items[1]))
        paper['supplement_text_scrape_status'] = 'succeeded'
        paper['store_status'] = 'stored'
        supplements.download_paper_supplements(conn, paper, ['structured'], force=True)
        assets = get_supplements(conn, paper['paper_id'])
        assert len(assets) == 1
        assert assets[0]['content'] == downloaded(items[1]).content
        assert set(assets[0]['metadata']['source_urls']) == {item.url for item in items}
        assert paper['supplement_text_scrape_status'] == 'pending'
        assert paper['store_status'] == 'pending'


def test_same_url_discovered_under_new_provider_updates_existing_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refreshing an existing URL through another source cannot retain stale bytes."""
    item = candidate()
    monkeypatch.setattr(supplements.provider, 'discover_supplements', lambda *_args: DiscoveryResult([item]))
    monkeypatch.setattr(supplements.provider, 'download_supplement', downloaded)
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        paper = {'paper_id': 'doi:10.1234/paper'}
        corpus.upsert_paper(conn, paper)
        supplements.download_paper_supplements(conn, paper, ['structured'])
        item.source = 'pubmed'
        monkeypatch.setattr(supplements.provider, 'download_supplement', lambda item: DownloadedSupplement(b'%PDF-1.7 revised', item.filename, 'application/pdf', item.url))
        supplements.download_paper_supplements(conn, paper, ['pubmed'], force=True)
        assets = get_supplements(conn, paper['paper_id'])
        assert len(assets) == 1
        assert assets[0]['content'] == b'%PDF-1.7 revised'
        assert assets[0]['metadata']['discovery_source'] == 'pubmed'
