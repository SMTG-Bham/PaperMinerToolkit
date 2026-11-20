"""Search for papers and acquire assets only for that search's corpus matches."""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from typing import Any

from paperminertoolkit.providers import registry
from paperminertoolkit.workflows.download import DOWNLOAD_FORMATS, download_papers
from paperminertoolkit.workflows.search import search_for_papers


def gather_papers(
    query: str,
    db_path: str | PathLike[str] = 'papers.db',
    source: str | Iterable[str] = 'all',
    count: int = 200,
    download_format: str = 'both',
    download_sources: Iterable[str] | None = None,
    download_abstract: bool = True,
    enrich: bool = False,
    parallel: bool = False,
    workers: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Search providers, merge matches, and download their requested assets.

    Parameters
    ----------
    query : str
        Nonblank search expression.
    db_path : str or os.PathLike[str], default='papers.db'
        Path to the SQLite paper corpus.
    source : str or Iterable[str], default='all'
        Search provider names. ``all`` selects every search provider.
    count : int, default=200
        Maximum records requested from each search provider, at least one.
    download_format : {'abstract', 'both', 'pdf', 'text'}, default='both'
        Asset types to acquire for papers matched by this search.
    download_sources : Iterable[str] or None, default=None
        Ordered download providers. ``None`` or ``all`` uses configured sources.
    download_abstract : bool, default=True
        Store available search abstracts and retrieve missing ones. Abstracts
        are always requested when ``download_format`` is ``abstract``.
    enrich : bool, default=False
        Supplement search matches using configured enrichment providers.
    parallel : bool, default=False
        Search selected providers concurrently.
    workers : int or None, default=None
        Maximum search workers. A positive value enables parallel search.
    force : bool, default=False
        Redownload requested asset types even when already present.

    Returns
    -------
    dict[str, Any]
        ``search`` contains the search workflow summary, including canonical
        corpus identifiers and provider failures. ``downloads`` contains asset
        acquisition counts, or ``None`` when no search matches were returned.
        Downloads are restricted to matched identifiers and respect active
        corpus filters. Existing assets are skipped unless ``force`` is true.

    Raises
    ------
    ValueError
        If the query, count, workers, format, or provider selection is invalid,
        or required download configuration is unavailable.
    Exception
        If the sole selected search provider fails or corpus writes fail.
        Multi-provider search failures are reported in ``search.status`` as
        ``partial`` or ``failed``; successful matches remain available.
    """
    download_format = download_format.lower()
    if download_format not in DOWNLOAD_FORMATS:
        raise ValueError(
            f'download_format must be one of: {", ".join(sorted(DOWNLOAD_FORMATS))}'
        )
    selected_download_sources = (
        None if download_sources is None else list(download_sources)
    )
    registry.resolve_names(selected_download_sources, registry.PDF,
                           preserve_order=True, label='download')
    want_abstract = download_abstract or download_format == 'abstract'
    search_summary = search_for_papers(
        query,
        db_path=db_path,
        source=source,
        count=count,
        store_abstract=want_abstract,
        enrich=enrich,
        parallel=parallel,
        workers=workers,
    )
    downloads = None
    if search_summary['paper_ids'] and search_summary['status'] != 'failed':
        downloads = download_papers(
            db_path=db_path,
            download_format=download_format,
            sources=selected_download_sources,
            download_abstract=want_abstract,
            force=force,
            paper_ids=search_summary['paper_ids'],
        )
    return {'search': search_summary, 'downloads': downloads}
