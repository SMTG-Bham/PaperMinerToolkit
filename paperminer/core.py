"""Request helpers for the CORE API used by PaperMiner.

CORE aggregates open-access papers from institutional and subject repositories,
which makes it the one source here whose records are neither a publisher's nor
a preprint server's. A record therefore carries a CORE identifier of its own
rather than only a DOI, and that identifier is what reaches the API again
later; it is kept in the corpus ``core_id`` column.

A key is required. CORE answers a keyless request with a 401, so a run that has
none is better off not asking. Store one with ``ps_core_key`` or in
``CORE_API_KEY``; it travels as a bearer token rather than as a query
parameter, which is what separates this source from OpenAlex and PubMed.

The published rate limit depends on the plan a key belongs to, so requests are
paced at a rate chosen to be unobtrusive rather than to satisfy a documented
rule, in the same way as the preprint servers.

Search is offset-paged and reports a total, so a walk stops at whichever comes
first: the caller's count, the reported total, or a short page. CORE caps a
page at 100 records however many are asked for.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias
from urllib.parse import quote

from paperminer import provider
from paperminer.metadata import clean_doi

BASE_URL = 'https://api.core.ac.uk/v3'
SEARCH_URL = f'{BASE_URL}/search/works'
WORKS_URL = f'{BASE_URL}/works'
CORE_MIN_INTERVAL = 1.0
PAGE_SIZE = 100
_CoreRecord: TypeAlias = dict[str, Any]
LIMITER = provider.RateLimiter(CORE_MIN_INTERVAL)


def configured_api_key(settings: Mapping[str, str] | None = None) -> str | None:
    """Return the configured CORE API key.

    Parameters
    ----------
    settings : Mapping[str, str] or None, optional
        Loaded PaperMiner settings. Read from disk when omitted.

    Returns
    -------
    str or None
        API key, or ``None`` when none is configured.
    """
    from paperminer.settings import load_settings
    settings = settings if settings is not None else load_settings()
    return settings.get('core_api_key') or os.environ.get('CORE_API_KEY') or None


def request_headers(api_key: str | None = None) -> dict[str, str]:
    """Build CORE request headers, carrying the API key when one is configured.

    Parameters
    ----------
    api_key : str or None, optional
        API key to send. Read from settings when omitted.

    Returns
    -------
    dict[str, str]
        Headers containing the PaperMiner user agent and, when available, a
        bearer token.
    """
    headers = provider.default_headers()
    key = api_key if api_key is not None else configured_api_key()
    if key:
        headers['Authorization'] = f'Bearer {key}'
    return headers


def work_url(core_id: object) -> str:
    """Build the metadata URL for one CORE work.

    Parameters
    ----------
    core_id : object
        CORE work identifier.

    Returns
    -------
    str
        Work endpoint URL, or an empty string when no identifier is present.
    """
    identifier = str(core_id or '').strip()
    return f'{WORKS_URL}/{quote(identifier, safe="")}' if identifier else ''


def download_url(work: Mapping[str, Any]) -> str:
    """Return the best PDF location for a CORE work.

    Parameters
    ----------
    work : Mapping[str, Any]
        CORE work record.

    Returns
    -------
    str
        Download URL, or an empty string when the record names none.
    """
    direct = work.get('downloadUrl') or work.get('download_url')
    if direct:
        return str(direct)
    url = work_url(work.get('id'))
    return f'{url}/download' if url else ''


def request_json(url: str,
                 params: Mapping[str, object] | None = None,
                 api_key: str | None = None,
                 session: provider.HTTPClient | None = None,
                 timeout: float = provider.DEFAULT_TIMEOUT,
                 attempts: int = provider.DEFAULT_ATTEMPTS) -> _CoreRecord | None:
    """Request a CORE endpoint and decode its JSON payload.

    Parameters
    ----------
    url : str
        CORE endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    api_key : str or None, optional
        API key to send. Read from settings when omitted.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    timeout : float, default=60.0
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    dict[str, Any] or None
        Decoded payload, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If the request is rejected, or all request attempts fail.
    """
    return provider.request_mapping(url, label='CORE', limiter=LIMITER, params=params,
                                    headers=request_headers(api_key), session=session,
                                    timeout=timeout, attempts=attempts)


def get_work(core_id: object,
             api_key: str | None = None,
             session: provider.HTTPClient | None = None) -> _CoreRecord | None:
    """Fetch one CORE work by its identifier.

    Parameters
    ----------
    core_id : object
        CORE work identifier.
    api_key : str or None, optional
        API key to send.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Work record, or ``None`` when CORE holds no such work.

    Raises
    ------
    RuntimeError
        If the request cannot be completed.
    """
    url = work_url(core_id)
    if not url:
        return None
    return request_json(url, api_key=api_key, session=session)


def search_page(query: str,
                limit: int = PAGE_SIZE,
                offset: int = 0,
                api_key: str | None = None,
                session: provider.HTTPClient | None = None) -> _CoreRecord | None:
    """Fetch one page of CORE search results.

    Parameters
    ----------
    query : str
        Search expression.
    limit : int, default=PAGE_SIZE
        Records requested, capped at CORE's per-page maximum.
    offset : int, default=0
        Zero-based index of the first record requested.
    api_key : str or None, optional
        API key to send.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Parsed payload, or ``None`` when CORE holds nothing to return.

    Raises
    ------
    RuntimeError
        If the request cannot be completed.
    """
    params = {'q': query, 'limit': min(max(int(limit), 1), PAGE_SIZE), 'offset': max(int(offset), 0)}
    return request_json(SEARCH_URL, params=params, api_key=api_key, session=session)


def parse_records(payload: Mapping[str, Any] | None) -> list[_CoreRecord]:
    """Return the work records a CORE payload carries.

    CORE has answered under both ``results`` and ``data`` over the life of the
    v3 API, so both are read.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed CORE payload.

    Returns
    -------
    list[dict[str, Any]]
        Work records in payload order.
    """
    if not payload:
        return []
    records = payload.get('results') or payload.get('data') or []
    if not isinstance(records, Sequence) or isinstance(records, str):
        return []
    return [dict(record) for record in records if isinstance(record, Mapping)]


def total_results(payload: Mapping[str, Any] | None) -> int:
    """Read the total hit count a CORE search payload reports.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed CORE payload.

    Returns
    -------
    int
        Total matching works, or ``0`` when the payload reports none.
    """
    if not payload:
        return 0
    total = payload.get('totalHits') or payload.get('total') or payload.get('count')
    try:
        return max(int(total), 0)
    except (TypeError, ValueError):
        return 0


def _first(value: object) -> str:
    """Read a CORE field that may arrive as a value or as a list of them.

    CORE aggregates from repositories whose metadata is uneven, and a field
    that is a string for one record is a one-element list for another. Taking
    the first entry is what keeps a DOI from being stored as ``['10.1/x']``.

    Parameters
    ----------
    value : object
        Raw CORE field value.

    Returns
    -------
    str
        First usable value as text, or an empty string.
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for entry in value:
            text = provider.clean_text(entry)
            if text:
                return text
        return ''
    return provider.clean_text(value)


def _authors(work: Mapping[str, Any]) -> str:
    """Format a CORE author list as one semicolon-separated string.

    Parameters
    ----------
    work : Mapping[str, Any]
        CORE work record.

    Returns
    -------
    str
        Author names in record order.
    """
    names = []
    for author in work.get('authors') or []:
        if isinstance(author, Mapping):
            names.append(str(author.get('name') or author.get('fullName') or ''))
        else:
            names.append(str(author))
    return '; '.join(name for name in names if name)


def _journal(work: Mapping[str, Any]) -> str:
    """Extract a venue name from a CORE work record.

    A repository record often names no journal, in which case the publisher is
    the closest thing to one.

    Parameters
    ----------
    work : Mapping[str, Any]
        CORE work record.

    Returns
    -------
    str
        Journal or publisher name, or an empty string.
    """
    journal = work.get('journal') or work.get('publisher') or ''
    if isinstance(journal, Mapping):
        return _first(journal.get('title') or journal.get('name'))
    return _first(journal)


def _publication_date(work: Mapping[str, Any]) -> str:
    """Extract the best available publication date from a CORE work record.

    Parameters
    ----------
    work : Mapping[str, Any]
        CORE work record.

    Returns
    -------
    str
        Full date when the record carries one, a year when it carries only
        that, or an empty string.
    """
    for field in ('publishedDate', 'published_date', 'yearPublished', 'year'):
        value = work.get(field)
        if value:
            return str(value)
    return ''


def abstract(work: Mapping[str, Any]) -> str:
    """Return a CORE work's abstract.

    Parameters
    ----------
    work : Mapping[str, Any]
        CORE work record.

    Returns
    -------
    str
        Abstract text, or an empty string when the record carries none.
    """
    return _first(work.get('abstract'))


def work_to_paper(work: Mapping[str, Any]) -> _CoreRecord:
    """Map one CORE work onto PaperMiner's paper schema.

    Parameters
    ----------
    work : Mapping[str, Any]
        CORE work record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata plus the ``abstract`` extra that the corpus
        schema does not store directly.
    """
    core_id = str(work.get('id') or '')
    doi = clean_doi(_first(work.get('doi') or work.get('DOI')))
    if core_id:
        paper_id = f'core:{core_id}'
    elif doi:
        paper_id = f'doi:{doi}'
    else:
        paper_id = ''
    return {
        'paper_id': paper_id,
        'doi': doi,
        'core_id': core_id,
        'title': _first(work.get('title')),
        'journal': _journal(work),
        'publication_date': _publication_date(work),
        'authors': _authors(work),
        'sources': 'core',
        'pdf_url': download_url(work),
        'metadata_status': 'retrieved',
        'abstract': abstract(work),
    }


def resolve_core_id(paper: Mapping[str, Any]) -> str:
    """Resolve one paper row's CORE identifier from values it already holds.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    str
        CORE identifier, or an empty string when the row stores none.
    """
    return str(paper.get('core_id') or '').strip()
