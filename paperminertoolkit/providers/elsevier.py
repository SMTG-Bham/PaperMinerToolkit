"""Small request helpers for the Elsevier APIs used by PaperMinerToolkit.

This module centralizes Elsevier HTTP details so the rest of the package does
not depend on the unsupported ``elsapy`` wrapper, and maps Elsevier search
records onto PaperMinerToolkit's paper schema.

A key is required, and it travels in the ``X-ELS-APIKey`` header rather than as
a query parameter -- the only header-authenticated source here apart from CORE.
Entitlement is tied to the subscribing institution, so what a key can retrieve
depends on where the request comes from as much as on the key itself: a search
may return a record whose full text the same key cannot fetch.

Elsevier pages a search by handing back a link rather than by an offset the
caller computes, which is why :func:`next_page_url` exists and why there is no
cursor helper.

Elsevier also meters a key by a quota over a period of days, not only by a rate
per second, and it reports what is left of that quota on every authenticated
response. A run that ignores those headers discovers exhaustion as a wall of
refusals partway through, having spent the requests that got it there, so this
module remembers what the last response said and refuses to make the next
request once nothing is left. See :func:`check_quota`.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus, urlsplit

import requests

from paperminertoolkit.providers import base as provider
from paperminertoolkit.corpus.metadata import clean_doi

BASE_URL = 'https://api.elsevier.com/content'
# Elsevier's documented quota headers. They appear on authenticated responses
# only, so an absent one means "nothing learned" rather than "nothing left".
QUOTA_LIMIT_HEADER = 'X-RateLimit-Limit'
QUOTA_REMAINING_HEADER = 'X-RateLimit-Remaining'
QUOTA_RESET_HEADER = 'X-RateLimit-Reset'
# The slowest rate Elsevier documents for any of its APIs, used for a path this
# module does not recognise so an unknown endpoint is never hurried.
ELSEVIER_MIN_INTERVAL = 1 / 2


@dataclass(frozen=True, slots=True)
class ElsevierApi:
    """One Elsevier API's own pace and its own weekly quota.

    Parameters
    ----------
    name : str
        API name as Elsevier's key settings page lists it.
    limiter : provider.RateLimiter
        Pacing window for this API alone.
    """

    name: str
    limiter: provider.RateLimiter


def _api(name: str, per_second: float) -> ElsevierApi:
    """Build one API's entry from its documented requests-per-second rate.

    Parameters
    ----------
    name : str
        API name as Elsevier's key settings page lists it.
    per_second : float
        Requests per second Elsevier documents for it.

    Returns
    -------
    ElsevierApi
        Entry owning that API's pacing window.
    """
    return ElsevierApi(name=name, limiter=provider.RateLimiter(1 / per_second))


# Elsevier states that "quota limits are unique to each API, there is not a
# single global setting for a given APIKey", and the per-second rates differ
# between them too, so both are held per API rather than per host. Pacing every
# path at the fastest of them would exceed the rate on most of them, and one
# shared quota figure would let a healthy allowance on one API clear an
# exhausted one on another. Keyed by the path segment after /content/.
ELSEVIER_APIS: dict[str, ElsevierApi] = {
    'search': _api('Scopus Search', 9),
    'abstract': _api('Abstract Retrieval', 9),
    'article': _api('ScienceDirect Article Retrieval', 10),
    # Object Retrieval serves an article's figures and is not listed among the
    # documented quotas, so it takes the rate of the Scopus APIs rather than
    # the faster article one.
    'object': _api('ScienceDirect Object Retrieval', 9),
}
UNKNOWN_API = _api('Elsevier', 1 / ELSEVIER_MIN_INTERVAL)
# Kept as the name other modules reach for; an unrecognised path lands here.
LIMITER = UNKNOWN_API.limiter


def api_for(url: str) -> ElsevierApi:
    """Identify which Elsevier API one URL belongs to.

    Parameters
    ----------
    url : str
        Elsevier endpoint URL.

    Returns
    -------
    ElsevierApi
        Matching API, or the conservative fallback for a path this module does
        not recognise.
    """
    path = urlsplit(url).path.lower()
    head, _, tail = path.partition('/content/')
    segment = (tail if _ else head).lstrip('/').split('/', 1)[0]
    return ELSEVIER_APIS.get(segment, UNKNOWN_API)


def limiter_for(url: str) -> provider.RateLimiter:
    """Return the pacing window for one Elsevier URL.

    Parameters
    ----------
    url : str
        Elsevier endpoint URL.

    Returns
    -------
    provider.RateLimiter
        The window belonging to that URL's API.
    """
    return api_for(url).limiter


# One remembered quota per API per key, because Elsevier meters them apart.
_quotas: dict[tuple[str, str], provider.Budget] = {}


def record_quota(response: provider.ResponseLike,
                 api_key: str = '',
                 url: str = '') -> provider.Budget | None:
    """Remember what one Elsevier response said about that API's quota.

    Parameters
    ----------
    response : provider.ResponseLike
        Response whose headers should be read.
    api_key : str, default=''
        Key the request was made with, recorded as a digest.
    url : str, default=''
        URL that was requested, which decides the quota it belongs to.

    Returns
    -------
    provider.Budget or None
        Quota now remembered, or ``None`` when the response reported none.
    """
    bucket = (api_for(url).name, provider.fingerprint(api_key))
    headers = getattr(response, 'headers', None)
    remaining = provider.header_int(headers, QUOTA_REMAINING_HEADER)
    if remaining is None:
        return _quotas.get(bucket)
    limit = provider.header_int(headers, QUOTA_LIMIT_HEADER)
    reset = provider.header_int(headers, QUOTA_RESET_HEADER)
    _quotas[bucket] = provider.Budget(remaining=max(remaining, 0),
                                      limit=limit if limit is not None else -1,
                                      reset_at=float(reset) if reset else 0.0,
                                      owner_fingerprint=bucket[1])
    return _quotas[bucket]


def quota_status(url: str = '', api_key: str = '') -> provider.Budget | None:
    """Return the last quota Elsevier reported for one API, if any.

    Parameters
    ----------
    url : str, default=''
        URL identifying the API to report on.
    api_key : str, default=''
        Key the quota belongs to.

    Returns
    -------
    provider.Budget or None
        Remembered quota, or ``None`` when no response has reported one.
    """
    return _quotas.get((api_for(url).name, provider.fingerprint(api_key)))


def reset_quota() -> None:
    """Forget every remembered quota.

    Quota state outlives a single call by design, so a test run or a caller
    starting again with different credentials clears it explicitly.

    Returns
    -------
    None
        The remembered quotas are cleared in place.
    """
    _quotas.clear()


def check_quota(api_key: str = '', url: str = '') -> None:
    """Refuse a request Elsevier has already said it will not answer.

    Elsevier reports each API's remaining allowance on every authenticated
    response. Once one reaches zero, further requests to that API are refused
    until it refills, so making them spends time and learns nothing. Raising
    here reports the exhaustion once, in terms of when it lifts, rather than as
    a run of identical failures.

    Only that API is refused. Elsevier meters each one separately, so a spent
    Scopus Search allowance says nothing about whether article retrieval can
    still be asked, and blocking it would stop work that would have succeeded.

    Nothing is enforced until a response has actually reported a figure, and
    figures observed under one key never gate a request made with another.

    Parameters
    ----------
    api_key : str, default=''
        Key the request is about to be made with.
    url : str, default=''
        URL about to be requested, which decides the quota that applies.

    Returns
    -------
    None
        Nothing is returned when the request may proceed.

    Raises
    ------
    RuntimeError
        If the remembered quota for this API and key is exhausted.
    """
    api = api_for(url)
    quota = _quotas.get((api.name, provider.fingerprint(api_key)))
    if quota is None or not quota.exhausted:
        return
    allowance = f' of {quota.limit}' if quota.limit >= 0 else ''
    refill = f' It refills at {quota.reset_text}.' if quota.reset_text else ''
    raise RuntimeError(
        f"Elsevier reports 0{allowance} requests left on {api.name} for this API key, so this "
        f'request was not sent.{refill} Wait for the allowance to refill, or use a key with a '
        'larger one.'
    )


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
    session: provider.HTTPClient | None = None,
) -> dict[str, Any] | None:
    """Request and decode an Elsevier JSON endpoint.

    Kept as a name callers may already use; :func:`request_json` is the same
    request and is what this delegates to, so there is only one implementation
    to keep paced.

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
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method. Defaults to :mod:`requests`.

    Returns
    -------
    dict[str, Any] or None
        Decoded JSON response body, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If Elsevier has reported that this key's quota is exhausted, the
        request is rejected, or the body is not a JSON object.
    """
    return request_json(url, api_key, params=params, timeout=timeout, session=session)


def get_content(api_key: str,
                url: str,
                accept: str,
                params: Mapping[str, object] | None = None,
                timeout: float = provider.DEFAULT_TIMEOUT,
                session: provider.HTTPClient | None = None) -> provider.ResponseLike:
    """Request raw content from an Elsevier endpoint.

    This is the path a PDF or an abstract arrives by, and it is paced, retried
    and quota-checked exactly as search and metadata are, because it is the
    same host and the same weekly allowance. Every failure, a missing document
    included, is raised rather than returned, so a caller working through
    candidate URLs treats them all the same way.

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
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method. Defaults to :mod:`requests`.

    Returns
    -------
    provider.ResponseLike
        Status-validated raw response.

    Raises
    ------
    RuntimeError
        If Elsevier has reported that this key's quota is exhausted, the
        document is absent, or the request is rejected or keeps failing.
    """
    check_quota(api_key, url)
    response = provider.request(url, label='Elsevier', limiter=limiter_for(url), params=params,
                                headers=api_headers(api_key, accept=accept), session=session,
                                timeout=timeout, missing_ok=False,
                                on_response=lambda answer: record_quota(answer, api_key, url))
    if response is None:  # pragma: no cover - missing_ok=False never returns None
        raise RuntimeError(f'Elsevier returned no content from {url}')
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
        Loaded PaperMinerToolkit settings. Read from disk when omitted.

    Returns
    -------
    str
        API key.

    Raises
    ------
    ValueError
        If no key is configured.
    """
    from paperminertoolkit.settings import load_settings
    settings = settings if settings is not None else load_settings()
    api_key = settings.get('elsevier_api_key') or os.environ.get('ELSEVIER_API_KEY')
    if not api_key:
        raise ValueError('Elsevier API key is not configured. Run pmt config elsevier-key first.')
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
        If Elsevier has reported that this key's quota is exhausted, the
        request is rejected, or all request attempts fail.
    """
    check_quota(api_key, url)
    return provider.request(url, label='Elsevier', limiter=limiter_for(url), params=params,
                            headers=api_headers(api_key, accept=accept), session=session,
                            timeout=timeout, attempts=attempts,
                            on_response=lambda response: record_quota(response, api_key, url))


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
        If Elsevier has reported that this key's quota is exhausted, the
        request is rejected, or the body is not a JSON object.
    """
    check_quota(api_key, url)
    return provider.request_mapping(url, label='Elsevier', limiter=limiter_for(url), params=params,
                                    headers=api_headers(api_key), session=session,
                                    timeout=timeout, attempts=attempts,
                                    on_response=lambda response: record_quota(response, api_key, url))


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
    """Map one Elsevier search record onto PaperMinerToolkit's paper schema.

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

    This lives here rather than in :mod:`paperminertoolkit.settings` so that settings
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


def _local_name(tag: str) -> str:
    """Return an XML tag name without its namespace or prefix."""
    return tag.rsplit('}', 1)[-1].rsplit(':', 1)[-1].lower()


def xml_plain_text(content: str) -> str:
    """Derive readable article prose from native Elsevier XML.

    Parameters
    ----------
    content : str
        Complete Elsevier article XML.

    Returns
    -------
    str
        Titles and paragraphs in document order, excluding figures, tables,
        references, and embedded objects.

    Raises
    ------
    RuntimeError
        If the response is malformed XML or represents an Elsevier error.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise RuntimeError(f'Elsevier returned malformed article XML: {error}') from error
    root_name = _local_name(root.tag)
    if root_name in {'error-response', 'service-error', 'error'}:
        message = ' '.join(''.join(root.itertext()).split())
        raise RuntimeError(f'Elsevier full text is unavailable: {message or "unknown error"}')
    skip = {'bibliography', 'figure', 'fig', 'table', 'table-wrap', 'object', 'attachment'}
    block_tags = {'title', 'section-title', 'para', 'p'}
    blocks: list[str] = []

    def walk(node: ET.Element) -> None:
        """Append prose blocks from one non-skipped XML subtree."""
        name = _local_name(node.tag)
        if name in skip:
            return
        if name in block_tags:
            text = ' '.join(''.join(node.itertext()).split())
            if text and (not blocks or blocks[-1] != text):
                blocks.append(text)
            return
        for child in node:
            walk(child)

    walk(root)
    return '\n\n'.join(blocks)


def full_text_document(
    url: str,
    api_key: str,
    session: provider.HTTPClient | None = None,
) -> provider.FullTextDocument:
    """Retrieve native Elsevier XML and derive plain text from one response.

    Parameters
    ----------
    url : str
        Elsevier article-retrieval endpoint.
    api_key : str
        Configured Elsevier API key.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    provider.FullTextDocument
        Native XML and derived text, or an empty result when no article is
        available.

    Raises
    ------
    RuntimeError
        If the request fails, entitlement is refused, or the XML is malformed.
    """
    response = request(
        url,
        api_key,
        accept='text/xml',
        params={'httpAccept': 'text/xml', 'view': 'FULL'},
        session=session,
    )
    if response is None or not (response.text or '').strip():
        return provider.FullTextDocument('')
    content = response.text
    text = xml_plain_text(content)
    if not text:
        return provider.FullTextDocument('')
    return provider.FullTextDocument(
        text=text,
        content=content,
        document_format='elsevier-xml',
        source_url=url,
        source_identifier=url.rsplit('/', 1)[-1],
        metadata={'publisher_native': True},
    )
