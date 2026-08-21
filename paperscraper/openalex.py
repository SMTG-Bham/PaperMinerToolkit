"""Small request helpers for the OpenAlex API used by PaperScraper.

This module centralizes OpenAlex HTTP details and the mapping from OpenAlex
work records onto PaperScraper's paper schema so search and download code can
share one implementation. OpenAlex meters access against a daily credit budget;
requests without an API key still work but draw on a much smaller budget.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias
from urllib.parse import quote

import requests

from paperscraper.metadata import clean_doi
from paperscraper.settings import load_settings

BASE_URL = 'https://api.openalex.org'
WORKS_URL = f'{BASE_URL}/works'
RATE_LIMIT_URL = f'{BASE_URL}/rate-limit'
USER_AGENT = 'PaperScraper/0.0.1'
_OpenAlexRecord: TypeAlias = dict[str, Any]


class _ResponseLike(Protocol):
    """HTTP response surface used by OpenAlex helpers."""

    status_code: int
    headers: Mapping[str, str]

    def raise_for_status(self) -> None:
        """Raise when the response has an unsuccessful status."""
        ...

    def json(self) -> _OpenAlexRecord:
        """Decode the response JSON object."""
        ...


class _HTTPClient(Protocol):
    """HTTP client surface accepted for dependency injection."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> _ResponseLike:
        """Issue an HTTP GET request."""
        ...


def configured_api_key(settings: Mapping[str, str] | None = None) -> str | None:
    """Return the configured OpenAlex API key.

    Parameters
    ----------
    settings : Mapping[str, str] or None, optional
        Settings mapping to inspect before the environment.

    Returns
    -------
    str or None
        Configured API key, or ``None`` when no key is available.
    """
    settings = settings or load_settings()
    return settings.get('openalex_api_key') or os.environ.get('OPENALEX_API_KEY')


def request_headers() -> dict[str, str]:
    """Build OpenAlex request headers.

    Returns
    -------
    dict[str, str]
        Headers containing the PaperScraper user agent.
    """
    return {'User-Agent': USER_AGENT}


def request_params(
    params: Mapping[str, object] | None = None,
    api_key: str | None = None,
) -> dict[str, object]:
    """Copy query parameters and add an API key when configured.

    Requests without a key are served from a much smaller daily credit budget,
    so the key is attached whenever it is available but never invented.

    Parameters
    ----------
    params : Mapping[str, object] or None, optional
        Query parameters to copy.
    api_key : str or None, optional
        OpenAlex API key to add.

    Returns
    -------
    dict[str, object]
        Copied parameters, optionally including ``api_key``.
    """
    merged = dict(params or {})
    if api_key:
        merged['api_key'] = api_key
    return merged


def _budget_error(response: _ResponseLike) -> str:
    """Describe an exhausted OpenAlex credit budget using the rate-limit headers."""
    reset = response.headers.get('X-RateLimit-Reset')
    try:
        wait = f' Budget resets in {round(float(reset) / 3600, 1)} hours.' if reset else ''
    except (TypeError, ValueError):
        wait = ''
    return ('OpenAlex daily credit budget is exhausted.'
            f'{wait} Configure an API key with ps_openalex_key or OPENALEX_API_KEY '
            'to raise the budget.')


def request_json(
    url: str,
    params: Mapping[str, object] | None = None,
    api_key: str | None = None,
    session: _HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> _OpenAlexRecord | None:
    """Request an OpenAlex endpoint with bounded retry/backoff behavior.

    Parameters
    ----------
    url : str
        OpenAlex endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    api_key : str or None, optional
        OpenAlex API key to attach.
    session : _HTTPClient or None, optional
        HTTP client exposing a ``get`` method. Defaults to :mod:`requests`.
    timeout : int or float, default=60
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    _OpenAlexRecord or None
        Decoded JSON payload, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If the API key is rejected, the credit budget is exhausted, or all
        request attempts fail.
    """
    session = session or requests
    headers = request_headers()
    params = request_params(params, api_key)
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 404:
                return None
            if response.status_code == 401:
                raise RuntimeError('OpenAlex rejected the API key. Set a valid key with '
                                   'ps_openalex_key or OPENALEX_API_KEY, or unset it to use '
                                   'the smaller keyless budget.')
            if response.status_code == 429:
                raise RuntimeError(_budget_error(response))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            retry_after = getattr(getattr(error, 'response', None), 'headers', {}).get('Retry-After')
            try:
                delay = min(max(float(retry_after), 0), 60) if retry_after else 2 ** attempt
            except (TypeError, ValueError):
                delay = 2 ** attempt
            time.sleep(delay)
    raise RuntimeError(f'OpenAlex request failed after {attempts} attempts: {last_error}') from last_error


def work_url(identifier: str) -> str:
    """Build a single-work URL.

    Parameters
    ----------
    identifier : str
        OpenAlex W-identifier or ``doi:``-prefixed DOI.

    Returns
    -------
    str
        Encoded OpenAlex work URL.
    """
    return f'{WORKS_URL}/{quote(str(identifier), safe=":/")}'


def get_work(
    identifier: str,
    api_key: str | None = None,
    session: _HTTPClient | None = None,
) -> _OpenAlexRecord | None:
    """Fetch one OpenAlex work record.

    Parameters
    ----------
    identifier : str
        OpenAlex W-identifier or ``doi:``-prefixed DOI.
    api_key : str or None, optional
        OpenAlex API key to attach.
    session : _HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    _OpenAlexRecord or None
        Work record, or ``None`` when the work does not exist.

    Raises
    ------
    RuntimeError
        If the OpenAlex request cannot be completed.
    """
    return request_json(work_url(identifier), api_key=api_key, session=session)


def work_id(work: Mapping[str, Any]) -> str:
    """Extract the short W-identifier from an OpenAlex work record.

    Parameters
    ----------
    work : Mapping[str, Any]
        OpenAlex work record.

    Returns
    -------
    str
        Short W-identifier, or an empty string when unavailable.
    """
    identifier = str(work.get('id') or '')
    return identifier.rstrip('/').rsplit('/', 1)[-1] if identifier else ''


def reconstruct_abstract(inverted_index: Mapping[str, list[int]] | None) -> str:
    """Rebuild abstract text from an OpenAlex inverted index.

    Parameters
    ----------
    inverted_index : Mapping[str, list[int]] or None
        Mapping of tokens to their positions in the abstract.

    Returns
    -------
    str
        Reconstructed abstract, or an empty string for a missing index.
    """
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ''
    positions = [(position, token)
                 for token, indexes in inverted_index.items()
                 for position in indexes or []]
    return ' '.join(token for _, token in sorted(positions))


def work_to_paper(work: Mapping[str, Any]) -> dict[str, Any]:
    """Map an OpenAlex work onto PaperScraper's paper schema.

    Parameters
    ----------
    work : Mapping[str, Any]
        OpenAlex work record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata.
    """
    doi = clean_doi(work.get('doi')) if work.get('doi') else ''
    identifier = work_id(work)
    if doi:
        paper_id = f'doi:{doi}'
    elif identifier:
        paper_id = f'openalex:{identifier}'
    else:
        paper_id = ''
    authors = '; '.join(
        author for author in (
            ((authorship or {}).get('author') or {}).get('display_name')
            for authorship in work.get('authorships') or []
        ) if author
    )
    journal = ((work.get('primary_location') or {}).get('source') or {}).get('display_name')
    return {
        'paper_id': paper_id,
        'doi': doi,
        'title': work.get('title') or work.get('display_name') or '',
        'journal': journal or '',
        'publication_date': work.get('publication_date') or str(work.get('publication_year') or ''),
        'authors': authors,
        'sources': 'openalex',
        'pdf_url': (work.get('best_oa_location') or {}).get('pdf_url') or '',
        'metadata_status': 'retrieved',
    }


def pdf_candidates(work: Mapping[str, Any]) -> list[str]:
    """Return candidate PDF URLs for an OpenAlex work.

    Parameters
    ----------
    work : Mapping[str, Any]
        OpenAlex work record.

    Returns
    -------
    list[str]
        Deduplicated PDF candidates, most authoritative first.
    """
    candidates = [(work.get('best_oa_location') or {}).get('pdf_url')]
    for location in work.get('locations') or []:
        candidates.append((location or {}).get('pdf_url'))
    candidates.append((work.get('open_access') or {}).get('oa_url'))
    return list(dict.fromkeys(url for url in candidates if url))
