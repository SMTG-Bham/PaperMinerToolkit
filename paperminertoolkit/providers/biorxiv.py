"""Request helpers for the bioRxiv API used by PaperMinerToolkit.

This module gives bioRxiv its own names and constants over the shared
bioRxiv-family client in :mod:`paperminertoolkit.providers.rxiv`, which documents the paging,
error, and versioning rules the two archives have in common. What follows is
what is particular to bioRxiv.

The service needs neither an API key nor a contact address, so there is nothing
to configure before using it. It publishes no rate limit either, so requests
are paced through one module-level limiter at a rate chosen to be unobtrusive
rather than to satisfy a documented rule.

bioRxiv has been accepting preprints since November 2013 and holds several
times what medRxiv does, so an unscoped walk is correspondingly longer and
:data:`MAX_SCAN_RECORDS` is correspondingly likelier to be what ends it.
Naming a category or a date range is the difference between a search of a
subject and a read of the archive.

bioRxiv DOIs carry the ``10.1101`` prefix up to late 2025 and ``10.64898``
afterwards. Both prefixes are shared with medRxiv, so a prefix says nothing
about which archive holds a preprint; what separates them is the accession
number, which runs six digits here and eight on medRxiv, whose counter is
prefixed with the two-digit year.

Two suffix shapes have to be recognized, not one. Preprints posted before 2018
were issued a bare accession number, as in ``10.1101/060400``, rather than the
dated ``10.1101/2023.12.01.569634`` in use since. Those identifiers were never
reissued, and a preprint that old can still be revised, so the old shape turns
up in a walk of last week's postings as readily as in one of 2014's. medRxiv
opened in 2019 and so has only ever used the dated form, which is why the bare
one is accepted here alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeAlias

from paperminertoolkit.providers import base as provider, rxiv as _rxiv

BASE_URL = _rxiv.BASE_URL
CATEGORY_BASE_URL = _rxiv.CATEGORY_BASE_URL
WEB_URL = 'https://www.biorxiv.org'
SERVER = 'biorxiv'
PAGE_SIZE = _rxiv.PAGE_SIZE
CATEGORY_PAGE_SIZE = _rxiv.CATEGORY_PAGE_SIZE
CORPUS_START = '2013-11-01'
BIORXIV_MIN_INTERVAL = _rxiv.RXIV_MIN_INTERVAL
MAX_SCAN_RECORDS = _rxiv.MAX_SCAN_RECORDS
QUERY_PREFIXES = _rxiv.QUERY_PREFIXES
OK_STATUS = _rxiv.OK_STATUS
EMPTY_STATUSES = _rxiv.EMPTY_STATUSES
# Two suffix shapes, because bioRxiv changed the scheme in 2018 and kept the
# old identifiers: a dated accession, and the bare accession used before it.
# Six digits either way, against medRxiv's eight, which is the only part of a
# dated DOI that tells the two archives apart; the upper bound leaves room for
# a seventh without reaching medRxiv's width, and the trailing lookahead stops
# a shorter run from matching inside a longer one. The bare form is tied to
# ``10.1101`` because that is the only prefix that ever issued it, and it is
# too plain to recognize safely under any prefix.
_BIORXIV_DOI = re.compile(
    r'(?P<doi>10\.\d{4,9}/\d{4}\.\d{2}\.\d{2}\.\d{6,7}(?!\d)'
    r'|10\.1101/\d{6,7}(?!\d))(?:v(?P<version>\d+))?',
    re.IGNORECASE,
)
_BiorxivRecord: TypeAlias = dict[str, Any]
LIMITER = provider.RateLimiter(BIORXIV_MIN_INTERVAL)
SERVER_CONFIG = _rxiv.RxivServer(
    name=SERVER,
    label='bioRxiv',
    web_url=WEB_URL,
    web_host='biorxiv.org',
    corpus_start=CORPUS_START,
    doi_pattern=_BIORXIV_DOI,
    limiter=LIMITER,
    max_scan_records=MAX_SCAN_RECORDS,
)


def request_headers() -> dict[str, str]:
    """Build bioRxiv request headers.

    Returns
    -------
    dict[str, str]
        Headers containing the PaperMinerToolkit user agent.
    """
    return provider.default_headers()


def details_url(doi: str, version: str = 'na') -> str:
    """Build the URL for a bioRxiv details request.

    Parameters
    ----------
    doi : str
        Bare bioRxiv DOI.
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

    See :func:`paperminertoolkit.providers.rxiv.endpoint` for why the two hosts differ and
    which is cheaper for which scan.

    Parameters
    ----------
    category : str, default=''
        bioRxiv subject category the scan is restricted to, if any.

    Returns
    -------
    tuple[str, int]
        Base URL to walk, and the number of records that host returns per page.
    """
    return _rxiv.endpoint(category)


def page_size(payload: Mapping[str, Any] | None, default: int = PAGE_SIZE) -> int:
    """Read how many records a bioRxiv payload actually returned.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed bioRxiv payload.
    default : int, default=PAGE_SIZE
        Page length assumed when the payload reports none.

    Returns
    -------
    int
        Records on this page, or ``default`` when the payload reports none.
    """
    return _rxiv.page_size(payload, default)


def interval_url(start_date: str, end_date: str, cursor: int = 0, category: str = '') -> str:
    """Build the URL for one page of a bioRxiv date interval.

    Parameters
    ----------
    start_date : str
        Inclusive interval start as ``YYYY-MM-DD``.
    end_date : str
        Inclusive interval end as ``YYYY-MM-DD``.
    cursor : int, default=0
        Zero-based index of the first record requested.
    category : str, default=''
        bioRxiv subject category the scan is restricted to, if any.

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
    """Build the public PDF location for a bioRxiv preprint.

    Parameters
    ----------
    doi : str
        bioRxiv DOI, with or without a version suffix.
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
    """Request a bioRxiv endpoint with courtesy pacing and bounded retries.

    Parameters
    ----------
    url : str
        bioRxiv endpoint URL.
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
) -> _BiorxivRecord | None:
    """Request a bioRxiv endpoint and parse its JSON payload.

    Parameters
    ----------
    url : str
        bioRxiv endpoint URL.
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
        Parsed payload, or ``None`` when bioRxiv holds no matching records.

    Raises
    ------
    RuntimeError
        If the request fails, the payload is not well-formed JSON, or bioRxiv
        reports the request as invalid.
    """
    return _rxiv.request_json(SERVER_CONFIG, url, params=params, session=session,
                              timeout=timeout, attempts=attempts)


def total_results(payload: Mapping[str, Any] | None) -> int:
    """Read the total record count a bioRxiv interval payload reports.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed bioRxiv payload.

    Returns
    -------
    int
        Total number of records in the interval, or ``0`` when none is
        reported.
    """
    return _rxiv.total_results(payload)


def normalize_biorxiv_doi(value: object) -> str:
    """Normalize a bioRxiv identifier to its bare, unversioned DOI.

    A DOI, a ``doi:`` or ``biorxiv:`` paper identifier, a resolver URL, and a
    bioRxiv content or PDF URL are all accepted. Both the dated accession scheme and
    the bare one bioRxiv used before 2018 are recognized. Case is
    folded, because DOIs are case-insensitive and the corpus compares them as
    strings. A medRxiv DOI is rejected rather than returned, because it names
    a preprint this API cannot answer for.

    Parameters
    ----------
    value : object
        bioRxiv DOI, paper identifier, or content URL.

    Returns
    -------
    str
        Bare bioRxiv DOI, or an empty string when none is present.
    """
    return _rxiv.normalize_doi(SERVER_CONFIG, value)


def biorxiv_version(value: object) -> str:
    """Extract the posted version number from a versioned bioRxiv identifier.

    Parameters
    ----------
    value : object
        bioRxiv DOI or content URL, such as ``10.1101/2023.12.01.569634v2``.

    Returns
    -------
    str
        Version number such as ``2``, or an empty string when unversioned.
    """
    return _rxiv.version_of(SERVER_CONFIG, value)


def record_to_paper(record: Mapping[str, Any]) -> _BiorxivRecord:
    """Map one bioRxiv API record onto PaperMinerToolkit's paper schema.

    Parameters
    ----------
    record : Mapping[str, Any]
        bioRxiv API record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata plus the ``abstract``, ``categories``,
        ``category``, ``primary_category``, ``license``, ``version``,
        ``published_doi``, and ``jatsxml`` extras that the corpus schema does
        not store directly.
    """
    return _rxiv.record_to_paper(SERVER_CONFIG, record)


def parse_records(payload: Mapping[str, Any] | None) -> list[_BiorxivRecord]:
    """Map every record in a bioRxiv payload onto the paper schema.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed bioRxiv payload, or ``None``.

    Returns
    -------
    list[dict[str, Any]]
        Normalized paper metadata, one mapping per record, in payload order.
    """
    return _rxiv.parse_records(SERVER_CONFIG, payload)


def latest_versions(entries: Sequence[Mapping[str, Any]]) -> list[_BiorxivRecord]:
    """Reduce parsed bioRxiv records to one entry per preprint, newest kept.

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
    """Yield bioRxiv interval page cursors newest first.

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
                  session: provider.HTTPClient | None = None) -> _BiorxivRecord | None:
    """Fetch one page of bioRxiv records posted within a date interval.

    Parameters
    ----------
    start_date : str
        Inclusive interval start as ``YYYY-MM-DD``.
    end_date : str
        Inclusive interval end as ``YYYY-MM-DD``.
    cursor : int, default=0
        Zero-based index of the first record requested.
    category : str, default=''
        bioRxiv subject category to restrict the interval to.
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
        If the request cannot be completed or bioRxiv rejects the interval.
    """
    return _rxiv.interval_page(SERVER_CONFIG, start_date, end_date, cursor, category,
                               session=session)


def details(doi: str,
            version: str = 'na',
            session: provider.HTTPClient | None = None) -> _BiorxivRecord | None:
    """Fetch the posted versions of one bioRxiv preprint.

    Parameters
    ----------
    doi : str
        bioRxiv DOI, with or without a version suffix.
    version : str, default='na'
        Posted version to request, or ``'na'`` for every version.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Parsed payload, or ``None`` when bioRxiv holds no such preprint.

    Raises
    ------
    RuntimeError
        If the request cannot be completed or bioRxiv rejects the DOI.
    """
    return _rxiv.details(SERVER_CONFIG, doi, version, session=session)


def fetch_doi(doi: str, session: provider.HTTPClient | None = None) -> _BiorxivRecord | None:
    """Fetch the newest posted version of one bioRxiv preprint.

    Parameters
    ----------
    doi : str
        bioRxiv DOI, with or without a version suffix.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Normalized paper metadata, or ``None`` when bioRxiv holds no such
        preprint.

    Raises
    ------
    RuntimeError
        If the request cannot be completed or bioRxiv rejects the DOI.
    """
    return _rxiv.fetch_doi(SERVER_CONFIG, doi, session=session)


def resolve_biorxiv_doi(paper: Mapping[str, Any]) -> str:
    """Resolve one paper row's bioRxiv DOI from values it already holds.

    This never issues a request. bioRxiv publishes no title search, so a row
    that carries neither a bioRxiv DOI nor a bioRxiv URL cannot be reached at
    all; a published DOI on the row belongs to the journal version and is not
    a bioRxiv identifier.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    str
        Bare bioRxiv DOI, or an empty string when the row stores none.
    """
    return _rxiv.resolve_doi(SERVER_CONFIG, paper)


def full_text(entry: Mapping[str, Any], session: provider.HTTPClient | None = None) -> str:
    """Fetch and flatten a bioRxiv preprint's JATS full text.

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
    """Split a bioRxiv search phrase into match terms and interval scope.

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
    """Report whether a bioRxiv record matches every term of a query.

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
