"""Request helpers for the medRxiv API used by PaperMiner.

This module gives medRxiv its own names and constants over the shared
bioRxiv-family client in :mod:`paperminer.providers.rxiv`, which documents the paging,
error, and versioning rules the two archives have in common. What follows is
what is particular to medRxiv.

The service needs neither an API key nor a contact address, so there is nothing
to configure before using it. It publishes no rate limit either, so requests
are paced through one module-level limiter at a rate chosen to be unobtrusive
rather than to satisfy a documented rule.

medRxiv opened in June 2019, so an unscoped walk reaches back only that
far -- shorter than bioRxiv's, and correspondingly cheaper.

medRxiv DOIs carry the ``10.1101`` prefix up to late 2025 and ``10.64898``
afterwards. Both prefixes are shared with bioRxiv, so a prefix says nothing
about which archive holds a preprint; what separates them is the accession
number, which runs eight digits here -- a two-digit year in front of a
six-digit counter -- against bioRxiv's six. Identifiers are therefore
recognized by the shape of the DOI suffix rather than by a prefix list, so a
further prefix change does not strand the older content or need a code change
to accept the newer, and a bioRxiv DOI is not mistaken for one this API can
answer.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeAlias

from paperminer.providers import base as provider, rxiv as _rxiv

BASE_URL = _rxiv.BASE_URL
CATEGORY_BASE_URL = _rxiv.CATEGORY_BASE_URL
WEB_URL = 'https://www.medrxiv.org'
SERVER = 'medrxiv'
PAGE_SIZE = _rxiv.PAGE_SIZE
CATEGORY_PAGE_SIZE = _rxiv.CATEGORY_PAGE_SIZE
CORPUS_START = '2019-06-01'
MEDRXIV_MIN_INTERVAL = _rxiv.RXIV_MIN_INTERVAL
MAX_SCAN_RECORDS = _rxiv.MAX_SCAN_RECORDS
QUERY_PREFIXES = _rxiv.QUERY_PREFIXES
OK_STATUS = _rxiv.OK_STATUS
EMPTY_STATUSES = _rxiv.EMPTY_STATUSES
# A medRxiv accession number is eight digits, against bioRxiv's six, which is
# the only part of the DOI that tells the two archives apart. The upper bound
# leaves room for a ninth digit without reaching down to bioRxiv's width, and
# the trailing lookahead stops a shorter run from matching inside a longer one.
_MEDRXIV_DOI = re.compile(
    r'(?P<doi>10\.\d{4,9}/\d{4}\.\d{2}\.\d{2}\.\d{8,9}(?!\d))(?:v(?P<version>\d+))?',
    re.IGNORECASE,
)
_MedrxivRecord: TypeAlias = dict[str, Any]
LIMITER = provider.RateLimiter(MEDRXIV_MIN_INTERVAL)
SERVER_CONFIG = _rxiv.RxivServer(
    name=SERVER,
    label='medRxiv',
    web_url=WEB_URL,
    web_host='medrxiv.org',
    corpus_start=CORPUS_START,
    doi_pattern=_MEDRXIV_DOI,
    limiter=LIMITER,
    max_scan_records=MAX_SCAN_RECORDS,
)


def request_headers() -> dict[str, str]:
    """Build medRxiv request headers.

    Returns
    -------
    dict[str, str]
        Headers containing the PaperMiner user agent.
    """
    return provider.default_headers()


def details_url(doi: str, version: str = 'na') -> str:
    """Build the URL for a medRxiv details request.

    Parameters
    ----------
    doi : str
        Bare medRxiv DOI.
    version : str, default='na'
        Posted version to request, or ``'na'`` for every version.

    Returns
    -------
    str
        Details endpoint URL.
    """
    return _rxiv.details_url(SERVER_CONFIG, doi, version)


def endpoint(category: str = '') -> tuple[str, int]:
    """Choose the interval host for a scan and report its page length.

    See :func:`paperminer.providers.rxiv.endpoint` for why the two hosts differ and
    which is cheaper for which scan.

    Parameters
    ----------
    category : str, default=''
        medRxiv subject category the scan is restricted to, if any.

    Returns
    -------
    tuple[str, int]
        Base URL to walk, and the number of records that host returns per page.
    """
    return _rxiv.endpoint(category)


def page_size(payload: Mapping[str, Any] | None, default: int = PAGE_SIZE) -> int:
    """Read how many records a medRxiv payload actually returned.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed medRxiv payload.
    default : int, default=PAGE_SIZE
        Page length assumed when the payload reports none.

    Returns
    -------
    int
        Records on this page, or ``default`` when the payload reports none.
    """
    return _rxiv.page_size(payload, default)


def interval_url(start_date: str, end_date: str, cursor: int = 0, category: str = '') -> str:
    """Build the URL for one page of a medRxiv date interval.

    Parameters
    ----------
    start_date : str
        Inclusive interval start as ``YYYY-MM-DD``.
    end_date : str
        Inclusive interval end as ``YYYY-MM-DD``.
    cursor : int, default=0
        Zero-based index of the first record requested.
    category : str, default=''
        medRxiv subject category the scan is restricted to, if any.

    Returns
    -------
    str
        Interval endpoint URL.

    Raises
    ------
    ValueError
        If either bound is not an ISO ``YYYY-MM-DD`` date.
    """
    return _rxiv.interval_url(SERVER_CONFIG, start_date, end_date, cursor, category)


def pdf_url(doi: str, version: str = '') -> str:
    """Build the public PDF location for a medRxiv preprint.

    Parameters
    ----------
    doi : str
        medRxiv DOI, with or without a version suffix.
    version : str, default=''
        Posted version number. Defaults to the version carried by ``doi``, and
        then to the first version.

    Returns
    -------
    str
        PDF URL, or an empty string when no DOI is present.
    """
    return _rxiv.pdf_url(SERVER_CONFIG, doi, version)


def request(
    url: str,
    params: Mapping[str, object] | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = provider.DEFAULT_TIMEOUT,
    attempts: int = provider.DEFAULT_ATTEMPTS,
) -> provider.ResponseLike | None:
    """Request a medRxiv endpoint with courtesy pacing and bounded retries.

    Parameters
    ----------
    url : str
        medRxiv endpoint URL.
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
    return _rxiv.request(SERVER_CONFIG, url, params=params, session=session,
                         timeout=timeout, attempts=attempts)


def request_json(
    url: str,
    params: Mapping[str, object] | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = provider.DEFAULT_TIMEOUT,
    attempts: int = provider.DEFAULT_ATTEMPTS,
) -> _MedrxivRecord | None:
    """Request a medRxiv endpoint and parse its JSON payload.

    Parameters
    ----------
    url : str
        medRxiv endpoint URL.
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
        Parsed payload, or ``None`` when medRxiv holds no matching records.

    Raises
    ------
    RuntimeError
        If the request fails, the payload is not well-formed JSON, or medRxiv
        reports the request as invalid.
    """
    return _rxiv.request_json(SERVER_CONFIG, url, params=params, session=session,
                              timeout=timeout, attempts=attempts)


def total_results(payload: Mapping[str, Any] | None) -> int:
    """Read the total record count a medRxiv interval payload reports.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed medRxiv payload.

    Returns
    -------
    int
        Total number of records in the interval, or ``0`` when none is
        reported.
    """
    return _rxiv.total_results(payload)


def normalize_medrxiv_doi(value: object) -> str:
    """Normalize a medRxiv identifier to its bare, unversioned DOI.

    A DOI, a ``doi:`` or ``medrxiv:`` paper identifier, a resolver URL, and a
    medRxiv content or PDF URL are all accepted. Case is
    folded, because DOIs are case-insensitive and the corpus compares them as
    strings. A bioRxiv DOI is rejected rather than returned, because it names
    a preprint this API cannot answer for.

    Parameters
    ----------
    value : object
        medRxiv DOI, paper identifier, or content URL.

    Returns
    -------
    str
        Bare medRxiv DOI, or an empty string when none is present.
    """
    return _rxiv.normalize_doi(SERVER_CONFIG, value)


def medrxiv_version(value: object) -> str:
    """Extract the posted version number from a versioned medRxiv identifier.

    Parameters
    ----------
    value : object
        medRxiv DOI or content URL, such as ``10.1101/2024.03.01.24303596v2``.

    Returns
    -------
    str
        Version number such as ``2``, or an empty string when unversioned.
    """
    return _rxiv.version_of(SERVER_CONFIG, value)


def record_to_paper(record: Mapping[str, Any]) -> _MedrxivRecord:
    """Map one medRxiv API record onto PaperMiner's paper schema.

    Parameters
    ----------
    record : Mapping[str, Any]
        medRxiv API record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata plus the ``abstract``, ``categories``,
        ``category``, ``primary_category``, ``license``, ``version``,
        ``published_doi``, and ``jatsxml`` extras that the corpus schema does
        not store directly.
    """
    return _rxiv.record_to_paper(SERVER_CONFIG, record)


def parse_records(payload: Mapping[str, Any] | None) -> list[_MedrxivRecord]:
    """Map every record in a medRxiv payload onto the paper schema.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed medRxiv payload, or ``None``.

    Returns
    -------
    list[dict[str, Any]]
        Normalized paper metadata, one mapping per record, in payload order.
    """
    return _rxiv.parse_records(SERVER_CONFIG, payload)


def latest_versions(entries: Sequence[Mapping[str, Any]]) -> list[_MedrxivRecord]:
    """Reduce parsed medRxiv records to one entry per preprint, newest kept.

    Parameters
    ----------
    entries : Sequence[Mapping[str, Any]]
        Parsed records from :func:`parse_records`.

    Returns
    -------
    list[dict[str, Any]]
        One record per preprint, in first-appearance order.
    """
    return _rxiv.latest_versions(SERVER_CONFIG, entries)


def page_cursors(total: int, step: int = PAGE_SIZE) -> Iterator[int]:
    """Yield medRxiv interval page cursors newest first.

    Parameters
    ----------
    total : int
        Total record count the interval reports.
    step : int, default=PAGE_SIZE
        Records per page, from :func:`page_size`.

    Yields
    ------
    int
        Zero-based cursor of each page, from the last page down to zero.
    """
    yield from _rxiv.page_cursors(total, step)


def interval_page(start_date: str,
                  end_date: str,
                  cursor: int = 0,
                  category: str = '',
                  session: provider.HTTPClient | None = None) -> _MedrxivRecord | None:
    """Fetch one page of medRxiv records posted within a date interval.

    Parameters
    ----------
    start_date : str
        Inclusive interval start as ``YYYY-MM-DD``.
    end_date : str
        Inclusive interval end as ``YYYY-MM-DD``.
    cursor : int, default=0
        Zero-based index of the first record requested.
    category : str, default=''
        medRxiv subject category to restrict the interval to.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Parsed payload, or ``None`` once the cursor is past the last record.

    Raises
    ------
    ValueError
        If either bound is not an ISO ``YYYY-MM-DD`` date.
    RuntimeError
        If the request cannot be completed or medRxiv rejects the interval.
    """
    return _rxiv.interval_page(SERVER_CONFIG, start_date, end_date, cursor, category,
                               session=session)


def details(doi: str,
            version: str = 'na',
            session: provider.HTTPClient | None = None) -> _MedrxivRecord | None:
    """Fetch the posted versions of one medRxiv preprint.

    Parameters
    ----------
    doi : str
        medRxiv DOI, with or without a version suffix.
    version : str, default='na'
        Posted version to request, or ``'na'`` for every version.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Parsed payload, or ``None`` when medRxiv holds no such preprint.

    Raises
    ------
    RuntimeError
        If the request cannot be completed or medRxiv rejects the DOI.
    """
    return _rxiv.details(SERVER_CONFIG, doi, version, session=session)


def fetch_doi(doi: str, session: provider.HTTPClient | None = None) -> _MedrxivRecord | None:
    """Fetch the newest posted version of one medRxiv preprint.

    Parameters
    ----------
    doi : str
        medRxiv DOI, with or without a version suffix.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Normalized paper metadata, or ``None`` when medRxiv holds no such
        preprint.

    Raises
    ------
    RuntimeError
        If the request cannot be completed or medRxiv rejects the DOI.
    """
    return _rxiv.fetch_doi(SERVER_CONFIG, doi, session=session)


def resolve_medrxiv_doi(paper: Mapping[str, Any]) -> str:
    """Resolve one paper row's medRxiv DOI from values it already holds.

    This never issues a request. medRxiv publishes no title search, so a row
    that carries neither a medRxiv DOI nor a medRxiv URL cannot be reached at
    all; a published DOI on the row belongs to the journal version and is not
    a medRxiv identifier.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    str
        Bare medRxiv DOI, or an empty string when the row stores none.
    """
    return _rxiv.resolve_doi(SERVER_CONFIG, paper)


def full_text(entry: Mapping[str, Any], session: provider.HTTPClient | None = None) -> str:
    """Fetch and flatten a medRxiv preprint's JATS full text.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Normalized record from :func:`record_to_paper`.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    str
        Article text, or an empty string when no body is available.

    Raises
    ------
    RuntimeError
        If the request cannot be completed or the payload is not well-formed.
    """
    return _rxiv.full_text(SERVER_CONFIG, entry, session=session)


def parse_query(query: str) -> tuple[list[str], dict[str, str]]:
    """Split a medRxiv search phrase into match terms and interval scope.

    Parameters
    ----------
    query : str
        Search phrase, optionally carrying ``category:``, ``from:``, or ``to:``
        scope terms.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        Match terms, and the scope mapping with ``category``, ``from``, and
        ``to`` keys for whichever were supplied.

    Raises
    ------
    ValueError
        If ``from:`` or ``to:`` is not an ISO ``YYYY-MM-DD`` date.
    """
    return _rxiv.parse_query(query)


def matches(entry: Mapping[str, Any], terms: Sequence[str]) -> bool:
    """Report whether a medRxiv record matches every term of a query.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Normalized record from :func:`record_to_paper`.
    terms : Sequence[str]
        Match terms from :func:`parse_query`. An empty sequence matches every
        record.

    Returns
    -------
    bool
        Whether the record matches all terms.
    """
    return _rxiv.matches(entry, terms)
