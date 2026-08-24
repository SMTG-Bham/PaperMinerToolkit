"""Small request helpers for the Elsevier APIs used by PaperMiner.

This module centralizes Elsevier HTTP details so the rest of the package does
not depend on the unsupported ``elsapy`` wrapper, and maps Elsevier search
records onto PaperMiner's paper schema.

A key is required, and it travels in the ``X-ELS-APIKey`` header rather than as
a query parameter -- the only header-authenticated source here apart from CORE.
Entitlement is tied to the subscribing institution, so what a key can retrieve
depends on where the request comes from as much as on the key itself: a search
may return a record whose full text the same key cannot fetch.

Elsevier pages a search by handing back a link rather than by an offset the
caller computes, which is why :func:`next_page_url` exists and why there is no
cursor helper.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote_plus

import requests

from paperminer import provider
from paperminer.metadata import clean_doi

BASE_URL = 'https://api.elsevier.com/content'
ELSEVIER_MIN_INTERVAL = 0.2
LIMITER = provider.RateLimiter(ELSEVIER_MIN_INTERVAL)


def api_headers(api_key: str, accept: str = 'application/json') -> dict[str, str]:
    """Build standard Elsevier API headers.

    Parameters
    ----------
    api_key : str
        Configured Elsevier API key.
    accept : str, default='application/json'
        Media type requested from Elsevier.

    Returns
    -------
    dict[str, str]
        Headers containing the API key, media type, and user agent.
    """
    return {
        'X-ELS-APIKey': api_key,
        'Accept': accept,
        'User-Agent': provider.USER_AGENT,
    }


def get_json(
    api_key: str,
    url: str,
    params: Mapping[str, object] | None = None,
    timeout: float = provider.DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Request and decode an Elsevier JSON endpoint.

    Parameters
    ----------
    api_key : str
        Configured Elsevier API key.
    url : str
        Endpoint URL to request.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    timeout : float, default=60.0
        Request timeout in seconds.

    Returns
    -------
    dict[str, Any]
        Decoded JSON response body.

    Raises
    ------
    requests.RequestException
        If the request fails or the response has an error status.
    """
    response = requests.get(url, headers=api_headers(api_key), params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_content(api_key: str,
                url: str,
                accept: str,
                params: Mapping[str, object] | None = None,
                timeout: float = provider.DEFAULT_TIMEOUT) -> requests.Response:
    """Request raw content from an Elsevier endpoint.

    Parameters
    ----------
    api_key : str
        Configured Elsevier API key.
    url : str
        Endpoint URL to request.
    accept : str
        Media type requested from Elsevier.
    params : dict or None, optional
        Query parameters for the request.
    timeout : float, default=60.0
        Request timeout in seconds.

    Returns
    -------
    requests.Response
        Status-validated raw response.

    Raises
    ------
    requests.RequestException
        If the request fails or the response has an error status.
    """
    response = requests.get(url, headers=api_headers(api_key, accept=accept), params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response


def search_url(index: str, query: str, count: int, search_fields: str) -> str:
    """Build an Elsevier search URL.

    Parameters
    ----------
    index : str
        Elsevier index to search, such as ``scopus``.
    query : str
        Search expression.
    count : int
        Number of records requested per page.
    search_fields : str
        Elsevier field expression that wraps ``query``.

    Returns
    -------
    str
        Encoded Elsevier search URL.
    """
    index = index.lower()
    provider_query = f'{search_fields}({query})'
    url = f'{BASE_URL}/search/{index}?query={quote_plus(provider_query)}&count={count}'
    if index == 'scopus':
        url += '&cursor=*'
    return url


def article_url_from_doi(doi: str) -> str:
    """Build an Elsevier article retrieval URL.

    Parameters
    ----------
    doi : str
        DOI identifying the article.

    Returns
    -------
    str
        Encoded Elsevier article URL.
    """
    return f'{BASE_URL}/article/doi/{quote_plus(str(doi))}'

def configured_api_key(settings: Mapping[str, str] | None = None) -> str:
    """Return the configured Elsevier API key.

    Parameters
    ----------
    settings : Mapping[str, str] or None, optional
        Loaded PaperMiner settings. Read from disk when omitted.

    Returns
    -------
    str
        API key.

    Raises
    ------
    ValueError
        If no key is configured.
    """
    from paperminer.settings import load_settings
    settings = settings if settings is not None else load_settings()
    api_key = settings.get('elsevier_api_key') or os.environ.get('ELSEVIER_API_KEY')
    if not api_key:
        raise ValueError('Elsevier API key is not configured. Run ps_elsevier_key first.')
    return str(api_key)


def request(url: str,
            api_key: str,
            accept: str = 'application/json',
            params: Mapping[str, object] | None = None,
            session: provider.HTTPClient | None = None,
            timeout: float = provider.DEFAULT_TIMEOUT,
            attempts: int = provider.DEFAULT_ATTEMPTS) -> provider.ResponseLike | None:
    """Request an Elsevier endpoint with courtesy pacing and bounded retries.

    Parameters
    ----------
    url : str
        Endpoint URL to request.
    api_key : str
        Configured Elsevier API key.
    accept : str, default='application/json'
        Media type requested from Elsevier.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method. Defaults to :mod:`requests`.
    timeout : float, default=60.0
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    provider.ResponseLike or None
        Successful response, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If the request is rejected, or all request attempts fail.
    """
    return provider.request(url, label='Elsevier', limiter=LIMITER, params=params,
                            headers=api_headers(api_key, accept=accept), session=session,
                            timeout=timeout, attempts=attempts)


def request_json(url: str,
                 api_key: str,
                 params: Mapping[str, object] | None = None,
                 session: provider.HTTPClient | None = None,
                 timeout: float = provider.DEFAULT_TIMEOUT,
                 attempts: int = provider.DEFAULT_ATTEMPTS) -> dict[str, Any] | None:
    """Request an Elsevier endpoint and decode its JSON payload.

    Parameters
    ----------
    url : str
        Endpoint URL to request.
    api_key : str
        Configured Elsevier API key.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
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
        If the request is rejected, or the body is not a JSON object.
    """
    return provider.request_mapping(url, label='Elsevier', limiter=LIMITER, params=params,
                                    headers=api_headers(api_key), session=session,
                                    timeout=timeout, attempts=attempts)


def next_page_url(payload: Mapping[str, Any] | None) -> str:
    """Return the next-page link an Elsevier search payload carries.

    Elsevier pages by handing back a link rather than by an offset the caller
    computes, so a walk follows this rather than incrementing anything.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Decoded Elsevier search payload.

    Returns
    -------
    str
        Next-page URL, or an empty string when this is the last page.
    """
    results = (payload or {}).get('search-results') or {}
    for link in results.get('link') or []:
        if isinstance(link, Mapping) and link.get('@ref') == 'next':
            return str(link.get('@href') or '')
    return ''


def total_results(payload: Mapping[str, Any] | None) -> int:
    """Read the total hit count an Elsevier search payload reports.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Decoded Elsevier search payload.

    Returns
    -------
    int
        Total matching records, or ``0`` when the payload reports none.
    """
    results = (payload or {}).get('search-results') or {}
    try:
        return max(int(results.get('opensearch:totalResults')), 0)
    except (TypeError, ValueError):
        return 0


def parse_records(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return the entries an Elsevier search payload carries.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Decoded Elsevier search payload.

    Returns
    -------
    list[dict[str, Any]]
        Search entries in payload order.
    """
    results = (payload or {}).get('search-results') or {}
    entries = results.get('entry') or []
    return [dict(entry) for entry in entries if isinstance(entry, Mapping)]


def full_text_uri(record: Mapping[str, Any]) -> str:
    """Return the article endpoint a search record advertises full text at.

    Parameters
    ----------
    record : Mapping[str, Any]
        Elsevier search record or corpus paper row.

    Returns
    -------
    str
        Article retrieval URL, or an empty string when the record names none.
    """
    for value in _link_values(record.get('link') or record.get('elsevier_link')):
        text = str(value or '')
        if text.startswith(f'{BASE_URL}/article/'):
            return text
    return ''


def _link_values(link: object) -> list[Any]:
    """Flatten the several shapes Elsevier uses for a link field.

    Parameters
    ----------
    link : object
        Raw link value: a string, a mapping, or a list of either.

    Returns
    -------
    list[Any]
        Candidate link values.
    """
    if link is None:
        return []
    if isinstance(link, str):
        return [link]
    if isinstance(link, Mapping):
        return [link.get('@href') or link.get('href')]
    if isinstance(link, Sequence):
        values = []
        for entry in link:
            values.extend(_link_values(entry))
        return values
    return []


def record_to_paper(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map one Elsevier search record onto PaperMiner's paper schema.

    Parameters
    ----------
    record : Mapping[str, Any]
        Elsevier search record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata plus the ``abstract`` extra that the corpus
        schema does not store directly.
    """
    return {
        'paper_id': str(record.get('dc:identifier') or record.get('eid') or ''),
        'doi': clean_doi(record.get('prism:doi')),
        'title': provider.clean_text(record.get('dc:title')),
        'journal': provider.clean_text(record.get('prism:publicationName')),
        'publication_date': provider.clean_text(record.get('prism:coverDate')),
        'authors': provider.clean_text(record.get('dc:creator') or record.get('creator')),
        'sources': 'elsevier',
        'elsevier_link': full_text_uri(record),
        'metadata_status': 'retrieved',
        'abstract': provider.clean_text(record.get('dc:description')
                                        or record.get('prism:teaser')),
    }


def check_api_key(api_key: str, session: provider.HTTPClient | None = None) -> bool:
    """Validate an Elsevier API key with a minimal Scopus search.

    This lives here rather than in :mod:`paperminer.settings` so that settings
    need not import a source module: every other source module imports settings,
    and the one edge in the other direction was the reason this one could not.

    Parameters
    ----------
    api_key : str
        Elsevier API key to validate.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    bool
        Whether Elsevier accepted the key.
    """
    try:
        request_json(search_url('scopus', 'Test', 1, 'TITLE-ABS-KEY'), api_key, session=session)
    except (RuntimeError, requests.RequestException):
        return False
    return True


def full_text(payload: Mapping[str, Any] | None) -> str:
    """Read the original text out of an Elsevier full-text payload.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Decoded article-retrieval payload.

    Returns
    -------
    str
        Article text, or an empty string when the payload carries none or
        carries it as structured XML rather than as text.
    """
    if not payload:
        return ''
    text = payload.get('originalText')
    if text is None:
        text = (payload.get('full-text-retrieval-response') or {}).get('originalText')
    return text if isinstance(text, str) else ''
