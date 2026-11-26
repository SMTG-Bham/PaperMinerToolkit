"""Acquire supplementary files with resumable per-file provenance and paper status."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from paperminertoolkit.corpus.database import (
    get_structured_documents, set_pipeline_status, upsert_paper,
)
from paperminertoolkit.corpus.supplements import SUPPLEMENT_ROLE, add_supplement, get_supplements
from paperminertoolkit.providers import registry
from paperminertoolkit.providers import supplements as provider

SUPPLEMENT_SOURCES = tuple(registry.names(registry.SUPPLEMENT))


def normalize_supplement_sources(sources: Iterable[str] | None = None) -> list[str]:
    """Validate SI source names and expand the default or ``all`` selection.

    Parameters
    ----------
    sources : iterable of str or None, optional
        Requested names, defaulting to all supported SI sources.

    Returns
    -------
    list[str]
        Unique source names in registry order.

    Raises
    ------
    ValueError
        If a source is unknown or the explicit selection is empty.
    """
    values = [sources] if isinstance(sources, str) else list(sources) if sources is not None else ['all']
    selected = {str(value).strip().lower() for value in values}
    unknown = selected - {*SUPPLEMENT_SOURCES, 'all'}
    if unknown:
        raise ValueError(f'Unknown supplementary sources: {", ".join(sorted(unknown))}')
    if not selected:
        raise ValueError('At least one supplementary source is required')
    return [name for name in SUPPLEMENT_SOURCES if 'all' in selected or name in selected]


def _metadata(paper: Mapping[str, Any]) -> dict[str, Any]:
    """Decode imported or stored paper metadata without discarding existing fields."""
    value = paper.get('metadata', paper.get('metadata_json')) or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = {}
    return dict(value) if isinstance(value, Mapping) else {}


def _urls(asset: Mapping[str, Any]) -> set[str]:
    """Collect primary and duplicate-source URLs already represented by an asset."""
    metadata = asset.get('metadata') or {}
    values = [metadata.get('source_url'), metadata.get('resolved_url'), *(metadata.get('source_urls') or [])]
    return {provider.canonical_url(str(value)) for value in values if value}


def download_paper_supplements(
    conn: sqlite3.Connection,
    paper: dict[str, Any],
    sources: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Discover, download, and store a paper's SI without replacing article assets.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus database connection.
    paper : dict[str, Any]
        Owning paper row; pipeline status and metadata are also updated in place.
    sources : iterable of str or None, optional
        SI sources, independent of the article PDF/text source selection.
    force : bool, default=False
        Refresh discovery and stored files; otherwise resume missing files and
        reuse completed runs with the same source and structured-document scope.

    Returns
    -------
    dict[str, int]
        Downloaded, skipped, and failed file counts under ``supplements``,
        ``supplements_skipped``, and ``supplements_failed``; unavailable paper
        count under ``supplements_unavailable`` and incomplete-discovery paper count
        under ``supplement_discovery_failed``. Discovery errors are persisted
        separately and never treated as proof that no SI exists.
    """
    selected = normalize_supplement_sources(sources)
    paper_id = str(paper['paper_id'])
    stored = conn.execute('SELECT * FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()
    merged = dict(stored) if stored is not None else {}
    merged.update(paper)
    paper.update(merged)
    metadata = _metadata(paper)
    documents = get_structured_documents(conn, paper_id)
    discovery_paper = {**metadata, **paper}
    input_fields = ('paper_id', 'doi', 'pmid', 'pmcid', 'landing_page_url', 'article_url', 'url',
                    'supplementary_urls', 'supplement_urls', 'supplementary_url', 'supplement_url')
    scope = {'sources': selected,
             'documents': sorted((str(doc['blob_id']), str(doc['metadata'].get('source_url', '')))
                                 for doc in documents),
             'inputs': {key: discovery_paper.get(key) for key in input_fields}}
    # Normalize tuples and imported metadata to the JSON shape persisted below.
    scope = json.loads(json.dumps(scope, sort_keys=True))
    previous = metadata.get('supplement_discovery') or {}
    assets = get_supplements(conn, paper_id, include_content=False)
    summary = {'supplements': 0, 'supplements_skipped': 0, 'supplements_failed': 0,
               'supplements_unavailable': 0, 'supplement_discovery_failed': 0}
    if (not force and previous.get('scope') == scope
            and paper.get('supplement_download_status') in {'succeeded', 'unavailable'}):
        if assets or paper.get('supplement_download_status') == 'unavailable':
            summary['supplements_skipped'] = len(assets)
            summary['supplements_unavailable'] = int(not assets)
            return summary
    discovery_paper = {**metadata, **paper}
    discovery = provider.discover_supplements(discovery_paper, documents, selected)
    candidates = list(discovery.candidates)
    if not force and previous.get('scope') == scope:
        # Preserve an earlier failed file even if a transient discovery outage
        # now omits it; a partial run must not silently become complete.
        current_urls = {provider.canonical_url(candidate.url) for candidate in candidates}
        for candidate in previous.get('candidates') or []:
            if isinstance(candidate, Mapping) and candidate.get('url') not in current_urls:
                candidates.append(provider.SupplementCandidate(**candidate))
                current_urls.add(str(candidate['url']))
    errors = list(discovery.errors)
    summary['supplement_discovery_failed'] = int(bool(discovery.errors))
    known_urls = {url for asset in assets for url in _urls(asset)}
    by_blob = {str(asset['blob_id']).rsplit(':', 1)[-1]: asset for asset in assets}
    for candidate in candidates:
        url = provider.canonical_url(candidate.url)
        if not force and url in known_urls:
            summary['supplements_skipped'] += 1
            continue
        try:
            downloaded = provider.download_supplement(candidate)
            digest = hashlib.sha256(downloaded.content).hexdigest()
            duplicate = by_blob.get(digest)
            matching = next((asset for asset in assets if url in _urls(asset)), None)
            if duplicate is not None:
                details = dict(duplicate['metadata'])
                aliases = _urls(duplicate) | {url, downloaded.source_url}
                if matching is not None:
                    aliases.update(_urls(matching))
                details['source_urls'] = sorted(aliases)
                add_supplement(conn, paper, downloaded.content, source=duplicate['source'],
                               source_url=details.get('source_url', ''),
                               original_filename=duplicate['original_filename'],
                               mime_type=duplicate['mime_type'], metadata=details)
                duplicate['metadata'] = details
                if matching is not None and matching['blob_id'] != duplicate['blob_id']:
                    # A refreshed URL now has bytes already present under a
                    # mirror. Remove its obsolete link only after B is stored.
                    conn.execute(
                        'DELETE FROM paper_assets WHERE paper_id = ? AND role = ? '
                        'AND source = ? AND original_filename = ? AND blob_id = ?',
                        (paper_id, SUPPLEMENT_ROLE, matching['source'],
                         matching['original_filename'], matching['blob_id']),
                    )
                    for key in ('supplement_text_scrape_status', 'supplement_image_scrape_status', 'store_status'):
                        paper[key] = 'pending'
                summary['supplements_skipped'] += 1
            else:
                details = {**candidate.metadata, 'label': candidate.label,
                           'resolved_url': downloaded.source_url}
                source = matching['source'] if matching is not None else candidate.source
                source_url = matching['metadata'].get('source_url', url) if matching is not None else url
                filename = matching['original_filename'] if matching is not None else downloaded.filename
                if matching is not None:
                    details['source_urls'] = sorted(_urls(matching) | {url, downloaded.source_url})
                    details['discovery_source'] = candidate.source
                blob_id = add_supplement(
                    conn, paper, downloaded.content, source=source,
                    source_url=source_url, original_filename=filename,
                    mime_type=downloaded.mime_type, metadata=details,
                )
                by_blob = {str(asset['blob_id']).rsplit(':', 1)[-1]: asset for asset in get_supplements(conn, paper_id, include_content=False)}
                if str(blob_id).rsplit(':', 1)[-1] not in by_blob:
                    raise RuntimeError('supplement asset was not persisted')
                summary['supplements'] += 1
            assets = get_supplements(conn, paper_id, include_content=False)
            by_blob = {str(asset['blob_id']).rsplit(':', 1)[-1]: asset for asset in assets}
            known_urls = {known for asset in assets for known in _urls(asset)}
        except (RuntimeError, ValueError, OSError) as error:
            summary['supplements_failed'] += 1
            errors.append(f'{candidate.source} {url}: {error}')
    available = bool(get_supplements(conn, paper_id, include_content=False))
    if errors:
        status = 'partial' if available else 'failed'
    elif candidates or available:
        status = 'succeeded'
    else:
        status = 'unavailable'
        summary['supplements_unavailable'] = 1
    previous_error = str(metadata.get('supplement_download_error') or '')
    error = '; '.join(errors)
    last_error = paper.get('last_error', '')
    set_pipeline_status(paper, 'supplement_download_status', status, error or None)
    if not error:
        # The shared status helper clears last_error on success, but an SI
        # success must not erase a failure from the main article's pipeline.
        paper['last_error'] = '' if last_error == previous_error else last_error
    metadata['supplement_download_error'] = error
    metadata['supplement_discovery'] = {
        'scope': scope, 'candidates': [asdict(candidate) for candidate in candidates],
        'errors': discovery.errors, 'notes': discovery.notes,
        'file_errors': errors[len(discovery.errors):],
    }
    paper['metadata'] = metadata
    upsert_paper(conn, paper)
    return summary
