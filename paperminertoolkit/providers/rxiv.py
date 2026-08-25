"""Shared client for the bioRxiv-family preprint APIs.

medRxiv and bioRxiv are one service answering under two host names. They share
their endpoints, their paging rules, their record shape, their error
conventions, and their absence of a search route, and differ only in the web
host that serves a preprint, the path segment that selects an archive, the date
their corpus starts, and the shape of the accession number in their DOIs. This
module holds the implementation once; :mod:`paperminertoolkit.providers.medrxiv` and
:mod:`paperminertoolkit.providers.biorxiv` supply those four differences as a
:class:`RxivServer` and re-export the operations under their own names.

Two hosts answer the same routes, and they differ in ways that decide which one
a given scan should use. ``api.medrxiv.org`` returns 100 records per page but
ignores the ``category`` parameter, silently answering with the unfiltered set;
``api.biorxiv.org`` applies ``category`` but returns only 30 records per page.
Both number their records identically, so a cursor means the same thing on
either, and both serve either archive because the archive is chosen by the path
rather than by the host. :func:`endpoint` therefore picks the wider page for an
unscoped walk and the filtering host only when a category was actually asked
for, which is the cheaper of the two in each case.

Because a cursor is an absolute record offset while a page is only as long as
the host chooses, stepping a walk by anything other than that host's page
length silently skips records. :func:`page_size` reads the length the host
actually used rather than trusting the constant, so a change on their side
costs efficiency instead of coverage.

The API reports failure in the body of an HTTP 200 response, as a status string
under ``messages``, so every payload is inspected for that shape before its
records are read. A missing record and an exhausted page walk are reported the
same way as a malformed request, and are told apart here by status text.

Neither archive publishes a search endpoint. A record is reachable by its DOI
or by walking a date interval, which is why :func:`parse_query` and
:func:`matches` exist: :mod:`paperminertoolkit.workflows.search` walks intervals and applies
the query itself. Interval pages also return one entry per posted version
rather than one per paper, so :func:`latest_versions` collapses them.

Because that walk is the whole archive unless the query narrows it,
:data:`MAX_SCAN_RECORDS` caps how far one search reads. Without it a term that
matches nothing recent would read every posting the archive has ever accepted,
which is the difference between a slow search and one that looks hung.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from paperminertoolkit.providers import base as provider, pubmed
from paperminertoolkit.corpus.metadata import clean_doi

BASE_URL = 'https://api.medrxiv.org'
CATEGORY_BASE_URL = 'https://api.biorxiv.org'
PAGE_SIZE = 100
CATEGORY_PAGE_SIZE = 30
RXIV_MIN_INTERVAL = 1.0
MAX_SCAN_RECORDS = 20000
QUERY_PREFIXES = ('category', 'from', 'to')
OK_STATUS = 'ok'
EMPTY_STATUSES = ('no posts found', 'doi not recognizable')
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
RxivRecord: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class RxivServer:
    """Everything that separates one bioRxiv-family archive from the other.

    Parameters
    ----------
    name : str
        Archive identifier, used as the API path segment, as the corpus
        ``sources`` value, and as the ``<name>_doi`` column stem.
    label : str
        Display name, as it should read in error messages and in ``journal``.
    web_url : str
        Site that serves the preprint's PDF and landing pages.
    web_host : str
        Host substring that identifies one of this archive's content URLs.
    corpus_start : str
        Date the archive began accepting preprints, as ``YYYY-MM-DD``.
    doi_pattern : re.Pattern[str]
        Pattern recognizing this archive's DOIs, with ``doi`` and ``version``
        groups. The two archives share their DOI prefixes, so the accession
        shape is the only thing that tells them apart.
    limiter : provider.RateLimiter
        Pacing window for this archive's host.
    max_scan_records : int, default=MAX_SCAN_RECORDS
        Most records one search reads before it reports a shortfall.
    """

    name: str
    label: str
    web_url: str
    web_host: str
    corpus_start: str
    doi_pattern: re.Pattern[str]
    limiter: provider.RateLimiter
    max_scan_records: int = MAX_SCAN_RECORDS

    @property
    def id_column(self) -> str:
        """Return the corpus column holding this archive's DOI.

        Returns
        -------
        str
            Column name, such as ``medrxiv_doi``.
        """
        return f'{self.name}_doi'


def details_url(server: RxivServer, doi: str, version: str = 'na') -> str:
    """Build the URL for a details request.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    doi : str
        Bare preprint DOI.
    version : str, default='na'
        Posted version to request, or ``'na'`` for every version.

    Returns
    -------
    str
        Details endpoint URL.
    """
    return f'{BASE_URL}/details/{server.name}/{doi}/{version}/json'


def endpoint(category: str = '') -> tuple[str, int]:
    """Choose the interval host for a scan and report its page length.

    Only ``api.biorxiv.org`` applies the ``category`` filter, and it pages in
    thirds of what ``api.medrxiv.org`` returns, so each host is cheaper for a
    different scan: the wider page wins when every record has to be read
    anyway, and the filter wins when it removes most of them first. Both serve
    either archive, so the choice costs nothing in coverage.

    Parameters
    ----------
    category : str, default=''
        Subject category the scan is restricted to, if any.

    Returns
    -------
    tuple[str, int]
        Base URL to walk, and the number of records that host returns per page.
    """
    if provider.clean_text(category):
        return CATEGORY_BASE_URL, CATEGORY_PAGE_SIZE
    return BASE_URL, PAGE_SIZE


def page_size(payload: Mapping[str, Any] | None, default: int = PAGE_SIZE) -> int:
    """Read how many records a payload actually returned.

    A walk steps its cursor by this rather than by :data:`PAGE_SIZE`, because
    a cursor counts records while a page holds however many the host chose to
    send. Read from the first page of a walk, where a short page can only mean
    the interval itself is shorter than one page.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed payload.
    default : int, default=PAGE_SIZE
        Page length assumed when the payload reports none.

    Returns
    -------
    int
        Records on this page, or ``default`` when the payload reports none.
    """
    count = str(_message(payload).get('count') or '').strip()
    return int(count) if count.isdigit() and int(count) > 0 else default


def interval_url(server: RxivServer,
                 start_date: str,
                 end_date: str,
                 cursor: int = 0,
                 category: str = '') -> str:
    """Build the URL for one page of a date interval.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    start_date : str
        Inclusive interval start as ``YYYY-MM-DD``.
    end_date : str
        Inclusive interval end as ``YYYY-MM-DD``.
    cursor : int, default=0
        Zero-based index of the first record requested.
    category : str, default=''
        Subject category the scan is restricted to, if any. Only the host that
        supports filtering is addressed when one is given.

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
        if not DATE_PATTERN.match(str(value or '')):
            raise ValueError(f'{label} must be an ISO date such as 2024-01-31, got {value!r}')
    base, _ = endpoint(category)
    return f'{base}/details/{server.name}/{start_date}/{end_date}/{max(int(cursor), 0)}/json'


def pdf_url(server: RxivServer, doi: str, version: str = '') -> str:
    """Build the public PDF location for a preprint.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    doi : str
        Preprint DOI, with or without a version suffix.
    version : str, default=''
        Posted version number. Defaults to the version carried by ``doi``, and
        then to the first version.

    Returns
    -------
    str
        PDF URL, or an empty string when no DOI is present.
    """
    identifier = normalize_doi(server, doi)
    if not identifier:
        return ''
    number = provider.clean_text(version) or version_of(server, doi) or '1'
    return f'{server.web_url}/content/{identifier}v{number}.full.pdf'


def request(
    server: RxivServer,
    url: str,
    params: Mapping[str, object] | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = provider.DEFAULT_TIMEOUT,
    attempts: int = provider.DEFAULT_ATTEMPTS,
) -> provider.ResponseLike | None:
    """Request an endpoint with courtesy pacing and bounded retries.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    url : str
        Endpoint URL.
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
    return provider.request(url, label=server.label, limiter=server.limiter, params=params,
                            session=session, timeout=timeout, attempts=attempts)


def request_json(
    server: RxivServer,
    url: str,
    params: Mapping[str, object] | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = provider.DEFAULT_TIMEOUT,
    attempts: int = provider.DEFAULT_ATTEMPTS,
) -> RxivRecord | None:
    """Request an endpoint and parse its JSON payload.

    An unknown DOI and a cursor past the end of an interval both arrive as an
    HTTP 200 carrying a status string, and both mean "nothing here" rather than
    "something went wrong", so they return ``None`` alongside a 404. Any other
    status is a rejected request and raises.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    url : str
        Endpoint URL.
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
        Parsed payload, or ``None`` when the archive holds no matching records.

    Raises
    ------
    RuntimeError
        If the request fails, the payload is not well-formed JSON, or the
        archive reports the request as invalid.
    """
    payload = provider.request_mapping(url, label=server.label, limiter=server.limiter,
                                       params=params, session=session, timeout=timeout,
                                       attempts=attempts)
    if payload is None:
        return None
    status = _status(payload)
    if status.lower() in EMPTY_STATUSES:
        return None
    if status and status.lower() != OK_STATUS:
        raise RuntimeError(f'{server.label} rejected the request: {status}')
    return payload


def _message(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the first message block of a payload.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed payload.

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
        Parsed payload.

    Returns
    -------
    str
        Status text, or an empty string when the payload reports none.
    """
    return str(_message(payload).get('status') or '').strip()


def total_results(payload: Mapping[str, Any] | None) -> int:
    """Read the total record count an interval payload reports.

    The count includes one entry per posted version rather than one per paper,
    which is what makes it usable as a page-walk bound.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed payload.

    Returns
    -------
    int
        Total number of records in the interval, or ``0`` when none is
        reported.
    """
    total = str(_message(payload).get('total') or '').strip()
    return int(total) if total.isdigit() else 0


def normalize_doi(server: RxivServer, value: object) -> str:
    """Normalize an identifier to its bare, unversioned DOI.

    A DOI, a ``doi:`` or ``<archive>:`` paper identifier, a resolver URL, and a
    content or PDF URL are all accepted. Case is folded, because DOIs are
    case-insensitive and the corpus compares them as strings. The sibling
    archive's DOI is rejected rather than returned, because it names a preprint
    this archive cannot answer for.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    value : object
        DOI, paper identifier, or content URL.

    Returns
    -------
    str
        Bare DOI, or an empty string when none is present.
    """
    if value is None:
        return ''
    match = server.doi_pattern.search(str(value))
    return match.group('doi').lower() if match else ''


def version_of(server: RxivServer, value: object) -> str:
    """Extract the posted version number from a versioned identifier.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    value : object
        DOI or content URL, such as ``10.1101/2024.03.01.24303596v2``.

    Returns
    -------
    str
        Version number such as ``2``, or an empty string when unversioned.
    """
    if value is None:
        return ''
    match = server.doi_pattern.search(str(value))
    return match.group('version') or '' if match else ''


def _authors(value: object) -> str:
    """Reformat an author list into the corpus name order.

    These archives list authors as ``Family, G. I.`` while Crossref, OpenAlex,
    PubMed, and arXiv all supply ``Given Family``. The corpus holds one
    ``authors`` string per paper whatever found it, so the order is flipped
    here to keep rows comparable across providers. A name with no comma is
    passed through, which is what keeps consortium names intact.

    Parameters
    ----------
    value : object
        Semicolon-separated author list as the archive publishes it.

    Returns
    -------
    str
        Author names in record order, ``Given Family`` and semicolon-separated.
    """
    names = []
    for author in provider.clean_text(value).split(';'):
        name = author.strip().strip(',')
        if not name:
            continue
        family, _, given = name.partition(',')
        names.append(f'{given.strip()} {family.strip()}'.strip() if given.strip() else name)
    return '; '.join(names)


def _categories(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect a record's subject category.

    These archives file each preprint under exactly one category, so the list
    holds at most one entry. It is returned as a list anyway to match the shape
    the other providers' subject helpers consume.

    Parameters
    ----------
    record : Mapping[str, Any]
        API record.

    Returns
    -------
    list[dict[str, Any]]
        Category term with its primary flag, or an empty list.
    """
    category = provider.clean_text(record.get('category'))
    if not category:
        return []
    return [{'id': category.lower(), 'name': category, 'is_primary': True}]


def record_to_paper(server: RxivServer, record: Mapping[str, Any]) -> RxivRecord:
    """Map one API record onto PaperMinerToolkit's paper schema.

    A preprint that has since appeared in a journal reports the published DOI,
    in which case it is used as the paper's DOI and identifier so the row
    merges with the published record rather than duplicating it. The preprint's
    own DOI is kept in the archive's own column either way, because it is what
    reaches this API again later.

    ``journal`` is filled only for a preprint that has not been published.
    Enrichment fills only columns that are still empty, so writing the archive
    name onto a row that names a published version would permanently mask the
    journal that Crossref holds for it.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    record : Mapping[str, Any]
        API record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata plus the ``abstract``, ``categories``,
        ``category``, ``primary_category``, ``license``, ``version``,
        ``published_doi``, and ``jatsxml`` extras that the corpus schema does
        not store directly.
    """
    preprint_doi = normalize_doi(server, record.get('doi'))
    published = clean_doi(provider.clean_text(record.get('published')))
    doi = published or preprint_doi
    version = provider.clean_text(record.get('version'))
    categories = _categories(record)
    return {
        'paper_id': f'doi:{doi}' if doi else '',
        'doi': doi,
        server.id_column: preprint_doi,
        'title': provider.clean_text(record.get('title')),
        'journal': '' if published else server.label,
        'publication_date': provider.clean_text(record.get('date')),
        'authors': _authors(record.get('authors')),
        'sources': server.name,
        'pdf_url': pdf_url(server, preprint_doi, version),
        'metadata_status': 'retrieved',
        'abstract': provider.clean_text(record.get('abstract')),
        'categories': categories,
        'category': categories[0]['name'] if categories else '',
        'primary_category': categories[0]['id'] if categories else '',
        'license': provider.clean_text(record.get('license')),
        'version': version,
        'published_doi': published,
        'jatsxml': provider.clean_text(record.get('jatsxml')),
    }


def parse_records(server: RxivServer, payload: Mapping[str, Any] | None) -> list[RxivRecord]:
    """Map every record in a payload onto the paper schema.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    payload : Mapping[str, Any] or None
        Parsed payload, or ``None``.

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
    return [record_to_paper(server, record) for record in collection
            if isinstance(record, Mapping)]


def latest_versions(server: RxivServer,
                    entries: Sequence[Mapping[str, Any]]) -> list[RxivRecord]:
    """Reduce parsed records to one entry per preprint, keeping the newest.

    Both the details and the interval endpoints return one entry per posted
    version, so a paper revised three times arrives three times. The highest
    version wins, and each paper keeps the position of its first appearance so
    the caller's ordering survives.

    ``publication_date`` is the exception: it keeps the earliest date seen, so
    a paper holds the date it first appeared rather than the date of its most
    recent revision. That matches how :mod:`paperminertoolkit.providers.arxiv` dates a
    resubmitted preprint, which is what keeps a date-filtered corpus coherent
    across the two. Both rules hold whichever order the versions arrive in,
    because a search walks the archive newest first while a details request
    returns it oldest first.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    entries : Sequence[Mapping[str, Any]]
        Parsed records from :func:`parse_records`.

    Returns
    -------
    list[dict[str, Any]]
        One record per preprint, in first-appearance order.
    """
    best: dict[str, RxivRecord] = {}
    for entry in entries:
        key = str(entry.get(server.id_column) or entry.get('paper_id') or '')
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
        Parsed record.

    Returns
    -------
    int
        Version number, or ``0`` when the record carries none.
    """
    version = provider.clean_text(entry.get('version'))
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


def interval_page(server: RxivServer,
                  start_date: str,
                  end_date: str,
                  cursor: int = 0,
                  category: str = '',
                  session: provider.HTTPClient | None = None) -> RxivRecord | None:
    """Fetch one page of records posted within a date interval.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    start_date : str
        Inclusive interval start as ``YYYY-MM-DD``.
    end_date : str
        Inclusive interval end as ``YYYY-MM-DD``.
    cursor : int, default=0
        Zero-based index of the first record requested.
    category : str, default=''
        Subject category to restrict the interval to.
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
        If the request cannot be completed or the archive rejects the interval.
    """
    params = ({'category': provider.clean_text(category)}
              if provider.clean_text(category) else {})
    return request_json(server, interval_url(server, start_date, end_date, cursor, category),
                        params=params, session=session)


def details(server: RxivServer,
            doi: str,
            version: str = 'na',
            session: provider.HTTPClient | None = None) -> RxivRecord | None:
    """Fetch the posted versions of one preprint.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    doi : str
        Preprint DOI, with or without a version suffix.
    version : str, default='na'
        Posted version to request, or ``'na'`` for every version.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Parsed payload, or ``None`` when the archive holds no such preprint.

    Raises
    ------
    RuntimeError
        If the request cannot be completed or the archive rejects the DOI.
    """
    identifier = normalize_doi(server, doi)
    if not identifier:
        return None
    return request_json(server, details_url(server, identifier, version), session=session)


def fetch_doi(server: RxivServer,
              doi: str,
              session: provider.HTTPClient | None = None) -> RxivRecord | None:
    """Fetch the newest posted version of one preprint.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    doi : str
        Preprint DOI, with or without a version suffix.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Normalized paper metadata, or ``None`` when the archive holds no such
        preprint.

    Raises
    ------
    RuntimeError
        If the request cannot be completed or the archive rejects the DOI.
    """
    entries = latest_versions(server, parse_records(
        server, details(server, doi, session=session)))
    return entries[0] if entries else None


def resolve_doi(server: RxivServer, paper: Mapping[str, Any]) -> str:
    """Resolve one paper row's preprint DOI from values it already holds.

    This never issues a request. Neither archive publishes a title search, so a
    row that carries neither its DOI nor one of its URLs cannot be reached at
    all; a published DOI on the row belongs to the journal version and is not a
    preprint identifier.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    str
        Bare preprint DOI, or an empty string when the row stores none.
    """
    identifier = normalize_doi(server, paper.get(server.id_column))
    if identifier:
        return identifier
    for column in ['paper_id', 'doi', 'pdf_url']:
        value = str(paper.get(column) or '')
        if column == 'pdf_url' and server.web_host not in value.lower():
            continue
        identifier = normalize_doi(server, value)
        if identifier:
            return identifier
    return ''


def full_text(server: RxivServer,
              entry: Mapping[str, Any],
              session: provider.HTTPClient | None = None) -> str:
    """Fetch and flatten a preprint's JATS full text.

    Every record names a JATS document, which is the same format PubMed Central
    serves, so :mod:`paperminertoolkit.providers.pubmed`'s flattener is reused rather than
    reimplemented. Taking the text from JATS rather than from the PDF also
    skips a scrape: the structure this walks is the one the archive published,
    not one recovered from a page layout.

    Parameters
    ----------
    server : RxivServer
        Archive being addressed.
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
    url = provider.clean_text(entry.get('jatsxml'))
    if not url:
        return ''
    response = request(server, url, session=session)
    if response is None:
        return ''
    text = response.text or ''
    if not text.strip():
        return ''
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise RuntimeError(f'{server.label} returned malformed JATS XML: {error}') from error
    title = pubmed._element_text(root.find('.//article-title'))
    body = pubmed._jats_body_text(root)
    if not body:
        return ''
    return f'{title}\n\n{body}'.strip() if title else body


def parse_query(query: str) -> tuple[list[str], dict[str, str]]:
    """Split a search phrase into match terms and interval scope.

    These archives publish no search endpoint, so a query is answered by
    walking a date interval and matching records locally. That walk is the
    whole corpus unless the caller narrows it, so the query string doubles as
    the place to say how far it should reach: ``category:``, ``from:``, and
    ``to:`` are lifted out of the phrase and everything else is left as a match
    term. Quoted runs stay together as one phrase.

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
    pattern = re.compile(
        rf'(?:(?P<field>{"|".join(QUERY_PREFIXES)}):)?(?:"(?P<quoted>[^"]*)"|(?P<bare>\S+))',
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
        if field in {'from', 'to'} and not DATE_PATTERN.match(value):
            raise ValueError(f'{field}: must be an ISO date such as 2024-01-31, got {value!r}')
        scope[field] = value
    return terms, scope


def matches(entry: Mapping[str, Any], terms: Sequence[str]) -> bool:
    """Report whether a record matches every term of a query.

    Terms are combined with ``AND`` across the record's title, abstract,
    authors, and category, matching how OpenAlex, PubMed, and arXiv read the
    same phrase so result counts stay comparable across providers. A term
    matches at the start of a word rather than the whole of it, so ``genome``
    finds ``genomes`` and ``covid`` finds ``covid-19``.

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
    haystack = ' '.join(provider.clean_text(entry.get(field)) for field in
                        ('title', 'abstract', 'authors', 'category'))
    for term in terms:
        pattern = r'\s+'.join(re.escape(word) for word in term.split())
        if not re.search(rf'\b{pattern}', haystack, re.IGNORECASE):
            return False
    return True
