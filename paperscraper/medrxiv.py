"""Request helpers for the medRxiv API used by PaperScraper.

This module centralizes medRxiv HTTP details and the mapping from medRxiv API
records onto PaperScraper's paper schema, so search, download, and enrichment
code can share one implementation. The service needs neither an API key nor a
contact address, so there is nothing to configure before using it. It publishes
no rate limit either, so requests are paced through one module-level limiter at
a rate chosen to be unobtrusive rather than to satisfy a documented rule.

Two hosts answer the same routes with ``medrxiv`` as the server segment, and
they differ in ways that decide which one a given scan should use.
``api.medrxiv.org`` returns 100 records per page but ignores the ``category``
parameter, silently answering with the unfiltered set;
``api.biorxiv.org`` applies ``category`` but returns only 30 records per page.
Both number their records identically, so a cursor means the same thing on
either. :func:`endpoint` therefore picks the wider page for an unscoped walk
and the filtering host only when a category was actually asked for, which is
the cheaper of the two in each case.

Because a cursor is an absolute record offset while a page is only as long as
the host chooses, stepping a walk by anything other than that host's page
length silently skips records. :func:`page_size` reads the length the host
actually used rather than trusting the constant, so a change on their side
costs efficiency instead of coverage.

The API reports failure in the body of an HTTP 200 response, as a status string
under ``messages``, so every payload is inspected for that shape before its
records are read. A missing record and an exhausted page walk are reported the
same way as a malformed request, and are told apart here by status text.

medRxiv publishes no search endpoint. A record is reachable by its DOI or by
walking a date interval, which is why :func:`parse_query` and :func:`matches`
exist: :mod:`paperscraper.search` walks intervals and applies the query itself.
Interval pages also return one entry per posted version rather than one per
paper, so :func:`latest_versions` collapses them.

Because that walk is the whole archive unless the query narrows it,
:data:`MAX_SCAN_RECORDS` caps how far one search reads. Without it a term that
matches nothing recent would read every posting medRxiv has ever accepted,
which is the difference between a slow search and one that looks hung.

medRxiv DOIs carry the ``10.1101`` prefix up to late 2025 and ``10.64898``
afterwards. Identifiers are therefore recognized by the shape of the DOI suffix
rather than by a prefix list, so a further prefix change does not strand the
older content or need a code change to accept the newer.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, TypeAlias

import requests

from paperscraper import pubmed
from paperscraper.metadata import clean_doi

BASE_URL = 'https://api.medrxiv.org'
CATEGORY_BASE_URL = 'https://api.biorxiv.org'
WEB_URL = 'https://www.medrxiv.org'
SERVER = 'medrxiv'
USER_AGENT = 'PaperScraper/0.0.1'
PAGE_SIZE = 100
CATEGORY_PAGE_SIZE = 30
CORPUS_START = '2019-06-01'
MEDRXIV_MIN_INTERVAL = 1.0
MAX_SCAN_RECORDS = 20000
DOI_BATCH_SIZE = 1
QUERY_PREFIXES = ('category', 'from', 'to')
OK_STATUS = 'ok'
EMPTY_STATUSES = ('no posts found', 'doi not recognizable')
_MEDRXIV_DOI = re.compile(
    r'(?P<doi>10\.\d{4,9}/\d{4}\.\d{2}\.\d{2}\.\d{6,8})(?:v(?P<version>\d+))?',
    re.IGNORECASE,
)
_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_NOT_AVAILABLE = {'', 'na', 'n/a', 'none', 'null'}
_MedrxivRecord: TypeAlias = dict[str, Any]
_last_request_at = 0.0


class _ResponseLike(Protocol):
    """HTTP response surface used by the medRxiv helpers."""

    status_code: int
    headers: Mapping[str, str]
    text: str

    def json(self) -> Any:
        """Decode the response body as JSON."""
        ...

    def raise_for_status(self) -> None:
        """Raise when the response has an unsuccessful status."""
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


def _throttle(interval: float = MEDRXIV_MIN_INTERVAL) -> None:
    """Sleep until the shared medRxiv request window allows another request.

    A page walk is a long run of requests against one host, so the window is
    module-level state rather than a per-call loop delay.

    Parameters
    ----------
    interval : float, default=MEDRXIV_MIN_INTERVAL
        Minimum seconds required between consecutive requests.

    Returns
    -------
    None
        The shared request window is updated in place.
    """
    global _last_request_at
    now = time.monotonic()
    wait = interval - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
        now = time.monotonic()
    _last_request_at = now


def request_headers() -> dict[str, str]:
    """Build medRxiv request headers.

    Returns
    -------
    dict[str, str]
        Headers containing the PaperScraper user agent.
    """
    return {'User-Agent': USER_AGENT}


def _text(value: object) -> str:
    """Collapse an API value to trimmed text, treating placeholders as absent.

    The API writes ``NA`` rather than an empty string for a field it holds no
    value for, so that spelling has to be read as missing or it would be stored
    as though it were data.

    Parameters
    ----------
    value : object
        Raw API field value.

    Returns
    -------
    str
        Whitespace-collapsed text, or an empty string when the value is absent.
    """
    if value is None:
        return ''
    text = re.sub(r'\s+', ' ', str(value)).strip()
    return '' if text.lower() in _NOT_AVAILABLE else text


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
    return f'{BASE_URL}/details/{SERVER}/{doi}/{version}/json'


def endpoint(category: str = '') -> tuple[str, int]:
    """Choose the interval host for a scan and report its page length.

    Only ``api.biorxiv.org`` applies the ``category`` filter, and it pages in
    thirds of what ``api.medrxiv.org`` returns, so each host is cheaper for a
    different scan: the wider page wins when every record has to be read
    anyway, and the filter wins when it removes most of them first.

    Parameters
    ----------
    category : str, default=''
        medRxiv subject category the scan is restricted to, if any.

    Returns
    -------
    tuple[str, int]
        Base URL to walk, and the number of records that host returns per page.
    """
    if _text(category):
        return CATEGORY_BASE_URL, CATEGORY_PAGE_SIZE
    return BASE_URL, PAGE_SIZE


def page_size(payload: Mapping[str, Any] | None, default: int = PAGE_SIZE) -> int:
    """Read how many records a medRxiv payload actually returned.

    A walk steps its cursor by this rather than by :data:`PAGE_SIZE`, because
    a cursor counts records while a page holds however many the host chose to
    send. Read from the first page of a walk, where a short page can only mean
    the interval itself is shorter than one page.

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
    count = str(_message(payload).get('count') or '').strip()
    return int(count) if count.isdigit() and int(count) > 0 else default


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
        medRxiv subject category the scan is restricted to, if any. Only the
        host that supports filtering is addressed when one is given.

    Returns
    -------
    str
        Interval endpoint URL.

    Raises
    ------
    ValueError
        If either bound is not an ISO ``YYYY-MM-DD`` date.
    """
    for label, value in (('start_date', start_date), ('end_date', end_date)):
        if not _DATE.match(str(value or '')):
            raise ValueError(f'{label} must be an ISO date such as 2024-01-31, got {value!r}')
    base, _ = endpoint(category)
    return f'{base}/details/{SERVER}/{start_date}/{end_date}/{max(int(cursor), 0)}/json'


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
    identifier = normalize_medrxiv_doi(doi)
    if not identifier:
        return ''
    number = _text(version) or medrxiv_version(doi) or '1'
    return f'{WEB_URL}/content/{identifier}v{number}.full.pdf'


def request(
    url: str,
    params: Mapping[str, object] | None = None,
    session: _HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> _ResponseLike | None:
    """Request a medRxiv endpoint with courtesy pacing and bounded retries.

    A 429 means the request rate was too high rather than that a budget is
    gone, so it is retried. Every other client error is terminal and fails at
    once, because retrying it would only spend more requests.

    Parameters
    ----------
    url : str
        medRxiv endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    session : _HTTPClient or None, optional
        HTTP client exposing a ``get`` method. Defaults to :mod:`requests`.
    timeout : int or float, default=60
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    _ResponseLike or None
        Successful response, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If the request is rejected, or all request attempts fail.
    """
    session = session or requests
    headers = request_headers()
    merged = dict(params or {})
    last_error: Exception | None = None
    for attempt in range(attempts):
        _throttle()
        try:
            response = session.get(url, params=merged, headers=headers, timeout=timeout)
            if response.status_code == 404:
                return None
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(f'medRxiv rejected the request with '
                                   f'{response.status_code} from {url}')
            response.raise_for_status()
            return response
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
    raise RuntimeError(f'medRxiv request failed after {attempts} attempts: {last_error}') from last_error


def request_json(
    url: str,
    params: Mapping[str, object] | None = None,
    session: _HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> _MedrxivRecord | None:
    """Request a medRxiv endpoint and parse its JSON payload.

    An unknown DOI and a cursor past the end of an interval both arrive as an
    HTTP 200 carrying a status string, and both mean "nothing here" rather than
    "something went wrong", so they return ``None`` alongside a 404. Any other
    status is a rejected request and raises.

    Parameters
    ----------
    url : str
        medRxiv endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    session : _HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    timeout : int or float, default=60
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
    response = request(url, params=params, session=session, timeout=timeout, attempts=attempts)
    if response is None:
        return None
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(f'medRxiv returned malformed JSON: {error}') from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f'medRxiv returned an unexpected payload of type {type(payload).__name__}')
    status = _status(payload)
    if status.lower() in EMPTY_STATUSES:
        return None
    if status and status.lower() != OK_STATUS:
        raise RuntimeError(f'medRxiv rejected the request: {status}')
    return dict(payload)


def _message(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the first message block of a payload.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed medRxiv payload.

    Returns
    -------
    Mapping[str, Any]
        First message mapping, or an empty mapping when none is present.
    """
    if not payload:
        return {}
    messages = payload.get('messages')
    if isinstance(messages, Sequence) and not isinstance(messages, str):
        for message in messages:
            if isinstance(message, Mapping):
                return message
    return {}


def _status(payload: Mapping[str, Any] | None) -> str:
    """Return the status string a payload reports.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed medRxiv payload.

    Returns
    -------
    str
        Status text, or an empty string when the payload reports none.
    """
    return str(_message(payload).get('status') or '').strip()


def total_results(payload: Mapping[str, Any] | None) -> int:
    """Read the total record count a medRxiv interval payload reports.

    The count includes one entry per posted version rather than one per paper,
    which is what makes it usable as a page-walk bound.

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
    total = str(_message(payload).get('total') or '').strip()
    return int(total) if total.isdigit() else 0


def normalize_medrxiv_doi(value: object) -> str:
    """Normalize a medRxiv identifier to its bare, unversioned DOI.

    A DOI, a ``doi:`` or ``medrxiv:`` paper identifier, a resolver URL, and a
    medRxiv content or PDF URL are all accepted. Case is folded, because DOIs
    are case-insensitive and the corpus compares them as strings.

    Parameters
    ----------
    value : object
        medRxiv DOI, paper identifier, or content URL.

    Returns
    -------
    str
        Bare medRxiv DOI, or an empty string when none is present.
    """
    if value is None:
        return ''
    match = _MEDRXIV_DOI.search(str(value))
    return match.group('doi').lower() if match else ''


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
    if value is None:
        return ''
    match = _MEDRXIV_DOI.search(str(value))
    return match.group('version') or '' if match else ''


def _title_key(value: object) -> str:
    """Create a comparable paper-title key.

    Parameters
    ----------
    value : object
        Paper title value.

    Returns
    -------
    str
        Normalized lower-case title key.
    """
    return re.sub(r'\W+', ' ', str(value or '').lower()).strip()


def _authors(value: object) -> str:
    """Reformat a medRxiv author list into the corpus name order.

    medRxiv lists authors as ``Family, G. I.`` while Crossref, OpenAlex,
    PubMed, and arXiv all supply ``Given Family``. The corpus holds one
    ``authors`` string per paper whatever found it, so the order is flipped
    here to keep rows comparable across providers. A name with no comma is
    passed through, which is what keeps collaboration names intact.

    Parameters
    ----------
    value : object
        Semicolon-separated author list as medRxiv publishes it.

    Returns
    -------
    str
        Author names in record order, ``Given Family`` and semicolon-separated.
    """
    names = []
    for author in _text(value).split(';'):
        name = author.strip().strip(',')
        if not name:
            continue
        family, _, given = name.partition(',')
        names.append(f'{given.strip()} {family.strip()}'.strip() if given.strip() else name)
    return '; '.join(names)


def _categories(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect a record's medRxiv subject category.

    medRxiv files each preprint under exactly one category, so the list holds
    at most one entry. It is returned as a list anyway to match the shape the
    other providers' subject helpers consume.

    Parameters
    ----------
    record : Mapping[str, Any]
        medRxiv API record.

    Returns
    -------
    list[dict[str, Any]]
        Category term with its primary flag, or an empty list.
    """
    category = _text(record.get('category'))
    if not category:
        return []
    return [{'id': category.lower(), 'name': category, 'is_primary': True}]


def record_to_paper(record: Mapping[str, Any]) -> _MedrxivRecord:
    """Map one medRxiv API record onto PaperScraper's paper schema.

    A preprint that has since appeared in a journal reports the published DOI,
    in which case it is used as the paper's DOI and identifier so the row
    merges with the published record rather than duplicating it. The preprint's
    own DOI is kept in ``medrxiv_doi`` either way, because it is what reaches
    this API again later.

    ``journal`` is filled only for a preprint that has not been published.
    Enrichment fills only columns that are still empty, so writing ``medRxiv``
    onto a row that names a published version would permanently mask the
    journal that Crossref holds for it.

    Parameters
    ----------
    record : Mapping[str, Any]
        medRxiv API record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata plus the ``abstract``, ``categories``,
        ``category``, ``license``, ``version``, ``published_doi``, and
        ``jatsxml`` extras that the corpus schema does not store directly.
    """
    medrxiv_doi = normalize_medrxiv_doi(record.get('doi'))
    published = clean_doi(_text(record.get('published')))
    doi = published or medrxiv_doi
    version = _text(record.get('version'))
    categories = _categories(record)
    return {
        'paper_id': f'doi:{doi}' if doi else '',
        'doi': doi,
        'medrxiv_doi': medrxiv_doi,
        'title': _text(record.get('title')),
        'journal': '' if published else 'medRxiv',
        'publication_date': _text(record.get('date')),
        'authors': _authors(record.get('authors')),
        'sources': 'medrxiv',
        'pdf_url': pdf_url(medrxiv_doi, version),
        'metadata_status': 'retrieved',
        'abstract': _text(record.get('abstract')),
        'categories': categories,
        'category': categories[0]['name'] if categories else '',
        'license': _text(record.get('license')),
        'version': version,
        'published_doi': published,
        'jatsxml': _text(record.get('jatsxml')),
    }


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
    if not payload:
        return []
    collection = payload.get('collection')
    if not isinstance(collection, Sequence) or isinstance(collection, str):
        return []
    return [record_to_paper(record) for record in collection if isinstance(record, Mapping)]


def latest_versions(entries: Sequence[Mapping[str, Any]]) -> list[_MedrxivRecord]:
    """Reduce parsed records to one entry per preprint, keeping the newest.

    Both the details and the interval endpoints return one entry per posted
    version, so a paper revised three times arrives three times. The highest
    version wins, and each paper keeps the position of its first appearance so
    the caller's ordering survives.

    ``publication_date`` is the exception: it keeps the earliest date seen, so
    a paper holds the date it first appeared rather than the date of its most
    recent revision. That matches how :mod:`paperscraper.arxiv` dates a
    resubmitted preprint, which is what keeps a date-filtered corpus coherent
    across the two. Both rules hold whichever order the versions arrive in,
    because a search walks the archive newest first while a details request
    returns it oldest first.

    Parameters
    ----------
    entries : Sequence[Mapping[str, Any]]
        Parsed records from :func:`parse_records`.

    Returns
    -------
    list[dict[str, Any]]
        One record per preprint, in first-appearance order.
    """
    best: dict[str, _MedrxivRecord] = {}
    for entry in entries:
        key = str(entry.get('medrxiv_doi') or entry.get('paper_id') or '')
        if not key:
            continue
        current = best.get(key)
        if current is None:
            best[key] = dict(entry)
            continue
        newer, older = ((entry, current) if _version_rank(entry) >= _version_rank(current)
                        else (current, entry))
        merged = {**dict(older), **{field: value for field, value in newer.items() if value}}
        merged['publication_date'] = min(filter(None, (current.get('publication_date'),
                                                       entry.get('publication_date'))),
                                         default='')
        best[key] = merged
    return list(best.values())


def _version_rank(entry: Mapping[str, Any]) -> int:
    """Return a record's posted version as a sortable number.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Parsed medRxiv record.

    Returns
    -------
    int
        Version number, or ``0`` when the record carries none.
    """
    version = _text(entry.get('version'))
    return int(version) if version.isdigit() else 0


def page_cursors(total: int, step: int = PAGE_SIZE) -> Iterator[int]:
    """Yield interval page cursors newest first.

    Interval pages are ordered oldest first with no way to reverse them, so a
    walk that wants the newest preprints first has to start at the last page
    and step backwards. The record count and the page length are both known
    from the first page, which is what makes the last cursor computable
    without walking there.

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
    step = max(int(step), 1)
    if total <= 0:
        return
    for cursor in range((total - 1) // step * step, -1, -step):
        yield cursor


def interval_page(start_date: str,
                  end_date: str,
                  cursor: int = 0,
                  category: str = '',
                  session: _HTTPClient | None = None) -> _MedrxivRecord | None:
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
    session : _HTTPClient or None, optional
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
    params = {'category': _text(category)} if _text(category) else {}
    return request_json(interval_url(start_date, end_date, cursor, category),
                        params=params, session=session)


def details(doi: str,
            version: str = 'na',
            session: _HTTPClient | None = None) -> _MedrxivRecord | None:
    """Fetch the posted versions of one medRxiv preprint.

    Parameters
    ----------
    doi : str
        medRxiv DOI, with or without a version suffix.
    version : str, default='na'
        Posted version to request, or ``'na'`` for every version.
    session : _HTTPClient or None, optional
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
    identifier = normalize_medrxiv_doi(doi)
    if not identifier:
        return None
    return request_json(details_url(identifier, version), session=session)


def fetch_doi(doi: str, session: _HTTPClient | None = None) -> _MedrxivRecord | None:
    """Fetch the newest posted version of one medRxiv preprint.

    Parameters
    ----------
    doi : str
        medRxiv DOI, with or without a version suffix.
    session : _HTTPClient or None, optional
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
    entries = latest_versions(parse_records(details(doi, session=session)))
    return entries[0] if entries else None


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
    identifier = normalize_medrxiv_doi(paper.get('medrxiv_doi'))
    if identifier:
        return identifier
    for column in ['paper_id', 'doi', 'pdf_url']:
        value = str(paper.get(column) or '')
        if column == 'pdf_url' and 'medrxiv.org' not in value.lower():
            continue
        identifier = normalize_medrxiv_doi(value)
        if identifier:
            return identifier
    return ''


def full_text(entry: Mapping[str, Any], session: _HTTPClient | None = None) -> str:
    """Fetch and flatten a medRxiv preprint's JATS full text.

    Every medRxiv record names a JATS document, which is the same format
    PubMed Central serves, so :mod:`paperscraper.pubmed`'s flattener is reused
    rather than reimplemented. Taking the text from JATS rather than from the
    PDF also skips a scrape: the structure this walks is the one medRxiv
    published, not one recovered from a page layout.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Normalized record from :func:`record_to_paper`.
    session : _HTTPClient or None, optional
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
    url = _text(entry.get('jatsxml'))
    if not url:
        return ''
    response = request(url, session=session)
    if response is None:
        return ''
    text = response.text or ''
    if not text.strip():
        return ''
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise RuntimeError(f'medRxiv returned malformed JATS XML: {error}') from error
    title = pubmed._element_text(root.find('.//article-title'))
    body = pubmed._jats_body_text(root)
    if not body:
        return ''
    return f'{title}\n\n{body}'.strip() if title else body


def parse_query(query: str) -> tuple[list[str], dict[str, str]]:
    """Split a search phrase into match terms and interval scope.

    medRxiv publishes no search endpoint, so a query is answered by walking a
    date interval and matching records locally. That walk is the whole corpus
    unless the caller narrows it, so the query string doubles as the place to
    say how far it should reach: ``category:``, ``from:``, and ``to:`` are
    lifted out of the phrase and everything else is left as a match term.
    Quoted runs stay together as one phrase.

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
    terms: list[str] = []
    scope: dict[str, str] = {}
    pattern = re.compile(rf'(?:(?P<field>{"|".join(QUERY_PREFIXES)}):)?(?:"(?P<quoted>[^"]*)"|(?P<bare>\S+))',
                         re.IGNORECASE)
    for match in pattern.finditer(str(query or '')):
        value = match.group('quoted') if match.group('quoted') is not None else match.group('bare')
        value = ' '.join(str(value or '').split())
        if not value:
            continue
        field = (match.group('field') or '').lower()
        if not field:
            terms.append(value)
            continue
        if field in {'from', 'to'} and not _DATE.match(value):
            raise ValueError(f'{field}: must be an ISO date such as 2024-01-31, got {value!r}')
        scope[field] = value
    return terms, scope


def matches(entry: Mapping[str, Any], terms: Sequence[str]) -> bool:
    """Report whether a record matches every term of a query.

    Terms are combined with ``AND`` across the record's title, abstract,
    authors, and category, matching how OpenAlex, PubMed, and arXiv read the
    same phrase so result counts stay comparable across providers. A term
    matches at the start of a word rather than the whole of it, so ``vaccine``
    finds ``vaccines`` and ``covid`` finds ``covid-19``.

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
    haystack = ' '.join(_text(entry.get(field)) for field in
                        ('title', 'abstract', 'authors', 'category'))
    for term in terms:
        pattern = r'\s+'.join(re.escape(word) for word in term.split())
        if not re.search(rf'\b{pattern}', haystack, re.IGNORECASE):
            return False
    return True
