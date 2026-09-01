"""Small request helpers for the OpenAlex API used by PaperMinerToolkit.

This module centralizes OpenAlex HTTP details and the mapping from OpenAlex
work records onto PaperMinerToolkit's paper schema so search and download code can
share one implementation.

OpenAlex meters access two ways, and they fail for different reasons. A daily
credit budget, refilling at midnight UTC, is the one that binds: a key without
one draws on a tenth of the allowance a free key gets, and either can run out
partway through a corpus. Separately there is a ceiling of a hundred requests a
second, which pacing keeps to. Both are answered with 429, so the two are told
apart by what the response says is left rather than by the status alone --
retrying a rate trip works, and retrying an exhausted budget cannot.

The remaining budget is reported on every response, so a run refuses the next
request once nothing is left rather than discovering it as a refusal. See
:func:`check_budget`.
"""

from __future__ import annotations

import gzip
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias
from urllib.parse import quote

from paperminertoolkit.providers import base as provider
from paperminertoolkit.corpus.metadata import clean_doi
from paperminertoolkit.settings import load_settings

BASE_URL = 'https://api.openalex.org'
WORKS_URL = f'{BASE_URL}/works'
RATE_LIMIT_URL = f'{BASE_URL}/rate-limit'
CONTENT_BASE_URL = 'https://content.openalex.org/works'
# OpenAlex refuses above a hundred requests a second, so that is the ceiling
# pacing holds to. It is rarely the binding constraint: the daily credit budget
# runs out long before a run could sustain this rate, which is why exhausting
# the budget, not exceeding the rate, is what a long run has to plan for.
OPENALEX_MAX_PER_SECOND = 100
OPENALEX_MIN_INTERVAL = 1 / OPENALEX_MAX_PER_SECOND
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
    'has_content',
    'content_urls',
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
# OpenAlex's budget headers. X-RateLimit-Reset counts the seconds until the
# budget refills at midnight UTC, so unlike Elsevier's header of the same name
# it is a delay rather than a moment, and is converted on the way in.
BUDGET_LIMIT_HEADER = 'X-RateLimit-Limit'
BUDGET_REMAINING_HEADER = 'X-RateLimit-Remaining'
BUDGET_RESET_HEADER = 'X-RateLimit-Reset'
_budget: provider.Budget | None = None


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
        Headers containing the PaperMinerToolkit user agent.
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


def record_budget(response: provider.ResponseLike, api_key: str | None = None) -> provider.Budget | None:
    """Remember what one OpenAlex response said about the remaining budget.

    Parameters
    ----------
    response : provider.ResponseLike
        Response whose headers should be read.
    api_key : str or None, optional
        Key the request was made with, recorded as a digest.

    Returns
    -------
    provider.Budget or None
        Budget now remembered, or ``None`` when the response reported none.
    """
    global _budget
    headers = getattr(response, 'headers', None)
    remaining = provider.header_int(headers, BUDGET_REMAINING_HEADER)
    if remaining is None:
        return _budget
    limit = provider.header_int(headers, BUDGET_LIMIT_HEADER)
    reset = provider.header_int(headers, BUDGET_RESET_HEADER)
    _budget = provider.Budget(
        remaining=max(remaining, 0),
        limit=limit if limit is not None else -1,
        # Seconds until midnight UTC, so it becomes a moment here. A reported
        # zero is "refills now", which is different from an absent header
        # meaning the refill time is simply unknown.
        reset_at=time.time() + reset if reset is not None else 0.0,
        owner_fingerprint=provider.fingerprint(api_key or ''),
    )
    return _budget


def budget_status() -> provider.Budget | None:
    """Return the last budget OpenAlex reported, if any.

    Returns
    -------
    provider.Budget or None
        Remembered budget, or ``None`` when no response has reported one.
    """
    return _budget


def reset_budget() -> None:
    """Forget the remembered budget.

    Returns
    -------
    None
        The remembered budget is cleared in place.
    """
    global _budget
    _budget = None


def check_budget(api_key: str | None = None) -> None:
    """Refuse a request OpenAlex has already said it will not answer.

    The daily credit budget refills at midnight UTC and nothing before then
    changes it, so once it is gone every further request is a refusal. Raising
    here reports that once, in terms of when it lifts, rather than as a run of
    identical failures.

    Nothing is enforced until a response has actually reported a figure, and
    figures observed under one key never gate a request made with another.

    Parameters
    ----------
    api_key : str or None, optional
        Key the request is about to be made with.

    Returns
    -------
    None
        Nothing is returned when the request may proceed.

    Raises
    ------
    RuntimeError
        If the remembered budget for this key is exhausted.
    """
    budget = _budget
    if budget is None or not budget.exhausted:
        return
    if budget.owner_fingerprint != provider.fingerprint(api_key or ''):
        return
    raise RuntimeError(_budget_error_text(budget.limit, budget.reset_at - time.time(),
                                          keyed=bool(api_key)))


def _budget_error_text(limit: int, seconds_until_reset: float, keyed: bool) -> str:
    """Describe an exhausted OpenAlex credit budget.

    Parameters
    ----------
    limit : int
        Credits the day allows, or ``-1`` when OpenAlex did not say.
    seconds_until_reset : float
        Seconds until the budget refills, or a non-positive value when unknown.
    keyed : bool
        Whether the request carried an API key.

    Returns
    -------
    str
        Message naming the allowance, the wait, and the way to raise it.
    """
    allowance = f' of {limit} credits' if limit >= 0 else ''
    wait = (f' It refills in {round(seconds_until_reset / 3600, 1)} hours, at midnight UTC.'
            if seconds_until_reset > 0 else '')
    # A free key is worth ten times the keyless allowance and costs nothing, so
    # it is the first thing to suggest to a run that ran out without one.
    advice = ('' if keyed else
              ' A free API key raises the budget tenfold at no cost: run pmt config openalex-key '
              'or set OPENALEX_API_KEY.')
    return f'OpenAlex daily credit budget is exhausted, with 0{allowance} left.{wait}{advice}'


def _budget_error(response: provider.ResponseLike, api_key: str | None = None) -> str:
    """Describe an exhausted OpenAlex credit budget from a refused response.

    Parameters
    ----------
    response : provider.ResponseLike
        The rate-limited response, read for its budget headers.
    api_key : str or None, optional
        Key the refused request carried.

    Returns
    -------
    str
        Message naming the allowance and the reset window the headers supply.
    """
    headers = getattr(response, 'headers', None)
    limit = provider.header_int(headers, BUDGET_LIMIT_HEADER)
    reset = provider.header_int(headers, BUDGET_RESET_HEADER) or 0
    return _budget_error_text(limit if limit is not None else -1, float(reset),
                              keyed=bool(api_key))


def _terminal_error(response: provider.ResponseLike, api_key: str | None = None) -> str:
    """Report the OpenAlex statuses that are pointless to retry.

    A rejected key stays rejected. A 429 needs telling apart, because OpenAlex
    answers both of its limits with one: exceeding a hundred requests a second
    is worth another attempt after a pause, while an exhausted daily budget
    cannot succeed again until midnight UTC and retrying only spends the wait.
    The remaining-credit header separates them -- credits left means the refusal
    was about the rate, none left means the budget is gone. A 429 that reports
    no credits at all is read as exhaustion, which is the older behaviour and
    the safer guess, since retrying a spent budget achieves nothing.

    Parameters
    ----------
    response : provider.ResponseLike
        Response to classify.
    api_key : str or None, optional
        Key the request carried, used to word the advice.

    Returns
    -------
    str
        Failure message, or an empty string to fall through to the shared rule,
        which retries.
    """
    if response.status_code == 401:
        return ('OpenAlex rejected the API key. Set a valid key with pmt config openalex-key or '
                'OPENALEX_API_KEY, or unset it to use the smaller keyless budget.')
    if response.status_code == 429:
        remaining = provider.header_int(getattr(response, 'headers', None),
                                        BUDGET_REMAINING_HEADER)
        if remaining is not None and remaining > 0:
            return ''
        return _budget_error(response, api_key)
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
    check_budget(api_key)
    return provider.request_mapping(url, label='OpenAlex', limiter=LIMITER,
                                    params=request_params(params, api_key, mailto),
                                    session=session, timeout=timeout, attempts=attempts,
                                    client_error=lambda answer: _terminal_error(answer, api_key),
                                    on_response=lambda answer: record_budget(answer, api_key))


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
    """Map an OpenAlex work onto PaperMinerToolkit's paper schema.

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


def grobid_xml_url(work: Mapping[str, Any]) -> str:
    """Return the paid OpenAlex GROBID TEI endpoint for one work.

    Parameters
    ----------
    work : Mapping[str, Any]
        OpenAlex work record containing ``has_content`` or ``content_urls``.

    Returns
    -------
    str
        TEI content URL, or an empty string when no GROBID parse is available.
    """
    content_urls = work.get('content_urls') or {}
    if isinstance(content_urls, Mapping) and content_urls.get('grobid_xml'):
        return str(content_urls['grobid_xml'])
    has_content = work.get('has_content') or {}
    if not isinstance(has_content, Mapping) or not has_content.get('grobid_xml'):
        return ''
    identifier = work_id(work)
    return f'{CONTENT_BASE_URL}/{identifier}.grobid-xml' if identifier else ''


def request_content(
    url: str,
    api_key: str,
    session: provider.HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> provider.ResponseLike | None:
    """Download one metered OpenAlex full-text content object.

    Parameters
    ----------
    url : str
        URL under ``content.openalex.org``.
    api_key : str
        OpenAlex API key. Content downloads do not support keyless access.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    timeout : float, default=60
        Request timeout in seconds.
    attempts : int, default=4
        Maximum request attempts.

    Returns
    -------
    provider.ResponseLike or None
        Content response, or ``None`` for a missing object.

    Raises
    ------
    ValueError
        If no API key is supplied.
    RuntimeError
        If the request is rejected or cannot be completed.
    """
    if not api_key.strip():
        raise ValueError('OpenAlex GROBID downloads require an API key')
    # A content download is the most expensive call OpenAlex bills, so it is
    # the one most worth refusing before it is sent.
    check_budget(api_key)
    return provider.request(
        url,
        label='OpenAlex content',
        limiter=LIMITER,
        params={'api_key': api_key},
        headers=request_headers(),
        session=session,
        timeout=timeout,
        attempts=attempts,
        client_error=lambda answer: _terminal_error(answer, api_key),
        on_response=lambda answer: record_budget(answer, api_key),
    )


def _decoded_content(response: provider.ResponseLike) -> str:
    """Read a content response as text, decompressing it when it is gzipped.

    OpenAlex serves GROBID TEI gzipped, and declares it as a ``Content-Type``
    of ``application/gzip`` rather than a ``Content-Encoding`` of ``gzip``.
    Only the latter is decompressed for us, so reading ``response.text``
    directly yields the compressed bytes decoded as characters. The gzip magic
    number is checked rather than the header, because it is the body itself
    saying what it is.

    Parameters
    ----------
    response : provider.ResponseLike
        Content response to read.

    Returns
    -------
    str
        Decoded document text.
    """
    raw = getattr(response, 'content', b'')
    if isinstance(raw, bytes) and raw[:2] == b'\x1f\x8b':
        try:
            return gzip.decompress(raw).decode('utf-8')
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f'OpenAlex content could not be decompressed: {error}') from error
    return response.text or ''


def full_text_document(
    work: Mapping[str, Any],
    api_key: str,
    session: provider.HTTPClient | None = None,
) -> provider.FullTextDocument:
    """Fetch OpenAlex GROBID TEI and derive text and layout-safe metadata.

    Parameters
    ----------
    work : Mapping[str, Any]
        OpenAlex work record advertising GROBID content.
    api_key : str
        OpenAlex API key used by the metered content endpoint.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    provider.FullTextDocument
        PDF-derived TEI and plain text, or an empty result when unavailable.

    Raises
    ------
    ValueError
        If the API key is absent or the response is not TEI.
    RuntimeError
        If the content request fails.
    """
    url = grobid_xml_url(work)
    if not url:
        return provider.FullTextDocument('')
    response = request_content(url, api_key, session=session)
    if response is None:
        return provider.FullTextDocument('')
    content = _decoded_content(response)
    if not content.strip():
        return provider.FullTextDocument('')
    identifier = work_id(work)
    from paperminertoolkit.corpus.xml_layout import parse_tei_layout
    layout = parse_tei_layout(content, f'openalex:{identifier}', source_identifier=identifier)
    parts = [layout.title]
    parts.extend(block.text for block in layout.iter_text_blocks())
    text = '\n\n'.join(part for part in parts if part).strip()
    if not text:
        return provider.FullTextDocument('')
    return provider.FullTextDocument(
        text=text,
        content=content,
        document_format='tei',
        source_url=url,
        source_identifier=identifier,
        metadata={
            'publisher_native': False,
            'derived_from': 'pdf',
            'parser': 'grobid',
            'estimated_cost_usd': 0.01,
        },
    )
