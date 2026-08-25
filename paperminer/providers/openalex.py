"""Small request helpers for the OpenAlex API used by PaperMiner.

This module centralizes OpenAlex HTTP details and the mapping from OpenAlex
work records onto PaperMiner's paper schema so search and download code can
share one implementation. OpenAlex meters access against a daily credit budget;
requests without an API key still work but draw on a much smaller budget.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias
from urllib.parse import quote

from paperminer.providers import base as provider
from paperminer.corpus.metadata import clean_doi
from paperminer.settings import load_settings

BASE_URL = 'https://api.openalex.org'
WORKS_URL = f'{BASE_URL}/works'
RATE_LIMIT_URL = f'{BASE_URL}/rate-limit'
# OpenAlex documents ten requests a second; pacing at that rate keeps a long
# cursor walk inside the published limit instead of relying on it not noticing.
OPENALEX_MIN_INTERVAL = 0.1
MAX_FILTER_VALUES = 100
WORK_SELECT_FIELDS = (
    'id',
    'doi',
    'ids',
    'title',
    'display_name',
    'publication_date',
    'publication_year',
    'language',
    'type',
    'biblio',
    'primary_location',
    'best_oa_location',
    'open_access',
    'authorships',
    'is_authors_truncated',
    'cited_by_count',
    'referenced_works_count',
    'referenced_works',
    'is_retracted',
    'primary_topic',
    'topics',
    'keywords',
    'concepts',
    'sustainable_development_goals',
)
_OpenAlexRecord: TypeAlias = dict[str, Any]
LIMITER = provider.RateLimiter(OPENALEX_MIN_INTERVAL)


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
        Headers containing the PaperMiner user agent.
    """
    return provider.default_headers()


def request_params(
    params: Mapping[str, object] | None = None,
    api_key: str | None = None,
    mailto: str = '',
) -> dict[str, object]:
    """Copy query parameters and add an API key and contact address.

    Requests without a key are served from a much smaller daily credit budget,
    so the key is attached whenever it is available but never invented. The
    ``mailto`` address identifies the client; OpenAlex now meters access with
    the daily credit budget rather than a polite pool, so only an API key
    raises that budget.

    Parameters
    ----------
    params : Mapping[str, object] or None, optional
        Query parameters to copy.
    api_key : str or None, optional
        OpenAlex API key to add.
    mailto : str, default=''
        Contact email address to add when non-empty.

    Returns
    -------
    dict[str, object]
        Copied parameters, optionally including ``api_key`` and ``mailto``.
    """
    merged = dict(params or {})
    if api_key:
        merged['api_key'] = api_key
    if mailto:
        merged['mailto'] = mailto
    return merged


def _budget_error(response: provider.ResponseLike) -> str:
    """Describe an exhausted OpenAlex credit budget using the rate-limit headers.

    Parameters
    ----------
    response : provider.ResponseLike
        The rate-limited response, read for its reset header.

    Returns
    -------
    str
        Message naming the reset window when the header supplies one.
    """
    reset = response.headers.get('X-RateLimit-Reset')
    try:
        wait = f' Budget resets in {round(float(reset) / 3600, 1)} hours.' if reset else ''
    except (TypeError, ValueError):
        wait = ''
    return ('OpenAlex daily credit budget is exhausted.'
            f'{wait} Configure an API key with pm_openalex_key or OPENALEX_API_KEY '
            'to raise the budget.')


def _terminal_error(response: provider.ResponseLike) -> str:
    """Report the OpenAlex statuses that are pointless to retry.

    A rejected key stays rejected, and a 429 here means the daily credit budget
    is gone rather than that the client is going too fast, so neither is worth
    another attempt. Every other client error takes the shared rule.

    Parameters
    ----------
    response : provider.ResponseLike
        Response to classify.

    Returns
    -------
    str
        Failure message, or an empty string to fall through to the shared rule.
    """
    if response.status_code == 401:
        return ('OpenAlex rejected the API key. Set a valid key with ps_openalex_key or '
                'OPENALEX_API_KEY, or unset it to use the smaller keyless budget.')
    if response.status_code == 429:
        return _budget_error(response)
    return ''


def request_json(
    url: str,
    params: Mapping[str, object] | None = None,
    api_key: str | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
    mailto: str = '',
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
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method. Defaults to :mod:`requests`.
    timeout : int or float, default=60
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.
    mailto : str, default=''
        Contact email address sent with the request.

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
    return provider.request_mapping(url, label='OpenAlex', limiter=LIMITER,
                                    params=request_params(params, api_key, mailto),
                                    session=session, timeout=timeout, attempts=attempts,
                                    client_error=_terminal_error)


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
    session: provider.HTTPClient | None = None,
) -> _OpenAlexRecord | None:
    """Fetch one OpenAlex work record.

    Parameters
    ----------
    identifier : str
        OpenAlex W-identifier or ``doi:``-prefixed DOI.
    api_key : str or None, optional
        OpenAlex API key to attach.
    session : provider.HTTPClient or None, optional
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


def works_page(filter_value: str,
               filter_name: str = 'doi',
               select: Sequence[str] | None = WORK_SELECT_FIELDS,
               per_page: int = MAX_FILTER_VALUES,
               api_key: str | None = None,
               session: provider.HTTPClient | None = None,
               mailto: str = '') -> list[_OpenAlexRecord]:
    """Request one OR-filtered page of OpenAlex works.

    ``per_page`` must cover the number of OR-joined filter values, because the
    endpoint otherwise returns only its default page size and silently drops
    the remaining matches.

    Parameters
    ----------
    filter_value : str
        OR-joined filter values, already normalized.
    filter_name : str, default='doi'
        Filter key applied to the values, normally ``doi`` or ``ids.openalex``.
    select : Sequence[str] or None, optional
        Root-level fields to request. OpenAlex rejects dotted field paths.
    per_page : int, default=100
        Page size requested from OpenAlex.
    api_key : str or None, optional
        OpenAlex API key to attach.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    mailto : str, default=''
        Contact email address sent with the request.

    Returns
    -------
    list[_OpenAlexRecord]
        Work records returned for the filter, in OpenAlex's own order.

    Raises
    ------
    RuntimeError
        If the OpenAlex request cannot be completed.
    """
    params: dict[str, object] = {
        'filter': f'{filter_name}:{filter_value}',
        'per-page': max(1, min(int(per_page), MAX_FILTER_VALUES)),
    }
    if select:
        params['select'] = ','.join(select)
    payload = request_json(WORKS_URL, params=params, api_key=api_key,
                           session=session, mailto=mailto) or {}
    return list(payload.get('results') or [])


def works_batch(identifiers: Sequence[str],
                filter_name: str = 'doi',
                select: Sequence[str] | None = WORK_SELECT_FIELDS,
                api_key: str | None = None,
                session: provider.HTTPClient | None = None,
                batch_size: int = MAX_FILTER_VALUES,
                mailto: str = '') -> dict[str, _OpenAlexRecord]:
    """Look up many OpenAlex works and key the results by their identifier.

    OpenAlex accepts at most 100 OR-joined values per filter, returns matches
    in an arbitrary order, and omits identifiers it does not know, so results
    are keyed rather than zipped back onto the request order.

    Parameters
    ----------
    identifiers : Sequence[str]
        Bare lowercase DOIs, or short ``W`` identifiers for ``ids.openalex``.
    filter_name : str, default='doi'
        Filter key applied to the identifiers.
    select : Sequence[str] or None, optional
        Root-level fields to request.
    api_key : str or None, optional
        OpenAlex API key to attach.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    batch_size : int, default=100
        Identifiers requested per page, capped at the OpenAlex maximum.
    mailto : str, default=''
        Contact email address sent with the request.

    Returns
    -------
    dict[str, _OpenAlexRecord]
        Work records keyed by cleaned DOI or short identifier.

    Raises
    ------
    ValueError
        If ``batch_size`` is not a positive integer.
    RuntimeError
        If an OpenAlex request cannot be completed.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')
    wanted = list(dict.fromkeys(identifier for identifier in identifiers if identifier))
    batch_size = min(batch_size, MAX_FILTER_VALUES)
    works = {}
    for chunk in provider.chunked(wanted, batch_size):
        results = works_page('|'.join(chunk),
                             filter_name=filter_name,
                             select=select,
                             per_page=len(chunk),
                             api_key=api_key,
                             session=session,
                             mailto=mailto)
        for work in results:
            key = clean_doi(work.get('doi')) if filter_name == 'doi' else work_id(work)
            if key:
                works[key] = work
    return works


def work_to_paper(work: Mapping[str, Any]) -> dict[str, Any]:
    """Map an OpenAlex work onto PaperMiner's paper schema.

    Parameters
    ----------
    work : Mapping[str, Any]
        OpenAlex work record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata plus the ``abstract`` extra that the corpus
        schema does not store directly. The abstract is empty unless the work
        was fetched with ``abstract_inverted_index``, which
        :data:`WORK_SELECT_FIELDS` deliberately omits.
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
        'abstract': reconstruct_abstract(work.get('abstract_inverted_index')),
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
