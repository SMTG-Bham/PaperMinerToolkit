"""Request helpers for the arXiv API used by PaperScraper.

This module centralizes arXiv HTTP details and the mapping from arXiv Atom
entries onto PaperScraper's paper schema, so search, download, and enrichment
code can share one implementation. arXiv asks that clients leave three seconds
between consecutive requests, counted per IP address, so every request is paced
through one module-level limiter. The service needs neither an API key nor a
contact address, so there is nothing to configure before using it.

arXiv reports a rejected query as an HTTP 200 Atom feed holding a single error
entry rather than as an error status, so each parsed document is inspected for
that shape before its entries are read. arXiv also has no DOI search field:
a record is reachable by its arXiv identifier or by a fielded title query, and
:func:`resolve_arxiv_id` exists to bridge that gap for corpus rows that carry
only a title.

Responses are parsed with :mod:`xml.etree.ElementTree`, which is documented as
vulnerable to entity-expansion attacks. arXiv over HTTPS is a trusted source and
these helpers never resolve an external DTD, so this is safe here; do not point
them at arbitrary user-supplied XML.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from paperscraper import provider
from paperscraper.metadata import clean_doi

BASE_URL = 'https://export.arxiv.org/api/query'
PDF_URL = 'https://arxiv.org/pdf'
MAX_SEARCH_RESULTS = 30000
PAGE_SIZE = 200
ID_BATCH_SIZE = 100
TITLE_MATCH_LIMIT = 10
ARXIV_MIN_INTERVAL = 3.0
SORT_ORDERS = ('relevance', 'lastUpdatedDate', 'submittedDate')
SORT_DIRECTIONS = ('ascending', 'descending')
FIELD_PREFIXES = ('ti', 'au', 'abs', 'co', 'jr', 'cat', 'rn', 'id', 'all')
# Fields arXiv accepts in a search_query but that cannot serve as the default
# for a plain phrase, because they take a range rather than a term. They are
# recognized so a native query carrying one is passed through rather than
# being rewritten into nonsense.
RANGE_FIELDS = ('submittedDate', 'lastUpdatedDate')
ATOM_NS = 'http://www.w3.org/2005/Atom'
ARXIV_NS = 'http://arxiv.org/schemas/atom'
OPENSEARCH_NS = 'http://a9.com/-/spec/opensearch/1.1/'
ERROR_ID_MARKER = 'arxiv.org/api/errors'
# Matched case-insensitively because arXiv resolves identifiers that way and a
# citation may capitalize the archive; the trailing lookahead stops a shorter
# run from matching inside a longer one, as it does for the preprint DOIs.
_ARXIV_ID = re.compile(
    r'(?P<id>[a-z][a-z-]*(?:\.[a-zA-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?!\d)'
    r'(?P<version>v\d+)?',
    re.IGNORECASE,
)
_ArxivRecord: TypeAlias = dict[str, Any]
LIMITER = provider.RateLimiter(ARXIV_MIN_INTERVAL)


def request_headers() -> dict[str, str]:
    """Build arXiv request headers.

    Returns
    -------
    dict[str, str]
        Headers containing the PaperScraper user agent.
    """
    return provider.default_headers()


def request_params(
    search_query: str = '',
    id_list: Sequence[str] = (),
    start: int = 0,
    max_results: int = PAGE_SIZE,
    sort_by: str = '',
    sort_order: str = '',
) -> dict[str, object]:
    """Build arXiv query parameters, omitting the ones left unset.

    Sort values are checked here rather than left to the service, because arXiv
    answers an unknown value with an error feed that costs a request to learn
    nothing the caller could not have been told immediately.

    Parameters
    ----------
    search_query : str, default=''
        Fielded arXiv search expression.
    id_list : Sequence[str], default=()
        arXiv identifiers to fetch instead of, or alongside, a query.
    start : int, default=0
        Zero-based index of the first result requested.
    max_results : int, default=PAGE_SIZE
        Maximum number of results requested in this slice.
    sort_by : str, default=''
        Result ordering, one of :data:`SORT_ORDERS`.
    sort_order : str, default=''
        Sort direction, one of :data:`SORT_DIRECTIONS`.

    Returns
    -------
    dict[str, object]
        Query parameters for the arXiv endpoint.

    Raises
    ------
    ValueError
        If ``sort_by`` or ``sort_order`` is not a value arXiv accepts.
    """
    if sort_by and sort_by not in SORT_ORDERS:
        raise ValueError(f'sort_by must be one of: {", ".join(SORT_ORDERS)}')
    if sort_order and sort_order not in SORT_DIRECTIONS:
        raise ValueError(f'sort_order must be one of: {", ".join(SORT_DIRECTIONS)}')
    params: dict[str, object] = {}
    if search_query:
        params['search_query'] = search_query
    if id_list:
        params['id_list'] = ','.join(id_list)
    if start:
        params['start'] = int(start)
    params['max_results'] = int(max_results)
    if sort_by:
        params['sortBy'] = sort_by
    if sort_order:
        params['sortOrder'] = sort_order
    return params


def _error_text(root: ET.Element) -> str:
    """Extract an arXiv error message from a parsed response document.

    A rejected query still returns HTTP 200, carrying one entry whose
    identifier points at the arXiv error namespace, so the shape of the feed is
    the only signal that the request failed.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Parsed Atom feed root.

    Returns
    -------
    str
        Error message, or an empty string when the feed reports no error.
    """
    entries = root.findall(f'{{{ATOM_NS}}}entry')
    if len(entries) != 1:
        return ''
    identifier = _element_text(entries[0].find(f'{{{ATOM_NS}}}id'))
    if ERROR_ID_MARKER not in identifier:
        return ''
    return _element_text(entries[0].find(f'{{{ATOM_NS}}}summary'))


def request(
    url: str = BASE_URL,
    params: Mapping[str, object] | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = provider.DEFAULT_TIMEOUT,
    attempts: int = provider.DEFAULT_ATTEMPTS,
) -> provider.ResponseLike | None:
    """Request an arXiv endpoint with courtesy pacing and bounded retries.

    A 429 means the courtesy delay was not honoured rather than that a budget
    is gone, so it is retried. Every other client error is terminal and fails
    at once, because retrying it would only spend more requests.

    Parameters
    ----------
    url : str, default=BASE_URL
        arXiv endpoint URL.
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
    return provider.request(url, label='arXiv', limiter=LIMITER, params=params,
                            session=session, timeout=timeout, attempts=attempts)


def request_xml(
    url: str = BASE_URL,
    params: Mapping[str, object] | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = provider.DEFAULT_TIMEOUT,
    attempts: int = provider.DEFAULT_ATTEMPTS,
) -> ET.Element | None:
    """Request an arXiv endpoint and parse its Atom payload.

    arXiv reports a rejected query in the body of an HTTP 200 as a feed holding
    a single error entry, so a parsed feed is inspected before it is returned.

    Parameters
    ----------
    url : str, default=BASE_URL
        arXiv endpoint URL.
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
    xml.etree.ElementTree.Element or None
        Parsed feed root, or ``None`` for a 404 or empty response.

    Raises
    ------
    RuntimeError
        If the request fails, the payload is not well-formed, or the feed
        reports an error.
    """
    root = provider.request_xml(url, label='arXiv', limiter=LIMITER, params=params,
                                session=session, timeout=timeout, attempts=attempts)
    if root is None:
        return None
    error_text = _error_text(root)
    if error_text:
        raise RuntimeError(f'arXiv rejected the request: {error_text}')
    return root


def normalize_arxiv_id(value: object) -> str:
    """Normalize an arXiv identifier to its bare, unversioned form.

    Both identifier schemes are accepted: the current ``2301.12345`` form and
    the pre-2007 ``cond-mat/0501001`` form, with or without an ``arXiv:``
    label, a resolver URL, or a trailing version suffix.

    An old-style identifier is recognized whatever its case, because arXiv
    resolves it that way and a citation may capitalize the archive, and is
    returned in the form arXiv writes it: a lower-case archive and an
    upper-case two-letter subject class, as in ``math.GT/0309136``. Folding
    here is what keeps one preprint from being stored under two identifiers.

    Parameters
    ----------
    value : object
        arXiv identifier, abstract or PDF URL, or other identifier-like value.

    Returns
    -------
    str
        Bare arXiv identifier, or an empty string when none is present.
    """
    if value is None:
        return ''
    text = str(value).strip()
    if not text:
        return ''
    text = re.sub(r'^(?:https?://)?(?:[\w.-]*\.)?arxiv\.org/(?:abs|pdf)/', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^arxiv[:\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\.pdf$', '', text, flags=re.IGNORECASE)
    match = _ARXIV_ID.search(text)
    if not match:
        return ''
    identifier = match.group('id')
    archive, separator, number = identifier.partition('/')
    if not separator:
        return identifier
    subject, dot, subclass = archive.partition('.')
    canonical = f'{subject.lower()}{dot}{subclass.upper()}' if dot else subject.lower()
    return f'{canonical}/{number}'


def arxiv_version(value: object) -> str:
    """Extract the version suffix from an arXiv identifier.

    Parameters
    ----------
    value : object
        arXiv identifier, abstract or PDF URL, or other identifier-like value.

    Returns
    -------
    str
        Version suffix such as ``v2``, or an empty string when unversioned.
    """
    if value is None:
        return ''
    match = _ARXIV_ID.search(str(value))
    return match.group('version') or '' if match else ''


def _element_text(element: ET.Element | None) -> str:
    """Flatten an element's text and inline markup into one plain string.

    arXiv wraps titles and abstracts to a fixed width, so the newlines and
    runs of indentation they arrive with are collapsed to single spaces.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element or None
        Element to flatten.

    Returns
    -------
    str
        Whitespace-collapsed text, or an empty string for a missing element.
    """
    if element is None:
        return ''
    return re.sub(r'\s+', ' ', ''.join(element.itertext())).strip()


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


def _authors(entry: ET.Element) -> str:
    """Format an entry's author list as a semicolon-separated string.

    Parameters
    ----------
    entry : xml.etree.ElementTree.Element
        Atom ``entry`` element.

    Returns
    -------
    str
        Author names in record order.
    """
    names = []
    for author in entry.findall(f'{{{ATOM_NS}}}author'):
        name = _element_text(author.find(f'{{{ATOM_NS}}}name'))
        if name:
            names.append(name)
    return '; '.join(names)


def _publication_date(entry: ET.Element) -> str:
    """Extract the date version one of an entry was submitted.

    The ``published`` timestamp is preferred over ``updated`` so a paper keeps
    the date it first appeared even after the authors post a new version.

    Parameters
    ----------
    entry : xml.etree.ElementTree.Element
        Atom ``entry`` element.

    Returns
    -------
    str
        Publication date as ``YYYY-MM-DD``, or an empty string.
    """
    for tag in ['published', 'updated']:
        match = re.match(r'\d{4}-\d{2}-\d{2}', _element_text(entry.find(f'{{{ATOM_NS}}}{tag}')))
        if match:
            return match.group(0)
    return ''


def _categories(entry: ET.Element) -> list[dict[str, Any]]:
    """Collect an entry's arXiv categories, primary first.

    arXiv repeats the primary category among the plain ``category`` elements,
    so it is emitted once and flagged rather than listed twice.

    Parameters
    ----------
    entry : xml.etree.ElementTree.Element
        Atom ``entry`` element.

    Returns
    -------
    list[dict[str, Any]]
        Category terms with their primary flag, in display order.
    """
    primary = entry.find(f'{{{ARXIV_NS}}}primary_category')
    primary_term = (primary.get('term') or '').strip() if primary is not None else ''
    categories: list[dict[str, Any]] = []
    seen = set()
    if primary_term:
        categories.append({'id': primary_term, 'name': primary_term, 'is_primary': True})
        seen.add(primary_term)
    for category in entry.findall(f'{{{ATOM_NS}}}category'):
        term = (category.get('term') or '').strip()
        if not term or term in seen:
            continue
        seen.add(term)
        categories.append({'id': term, 'name': term, 'is_primary': False})
    return categories


def _pdf_url(entry: ET.Element, arxiv_id: str) -> str:
    """Find an entry's PDF location, falling back to the canonical path.

    Withdrawn papers carry no PDF link, and older records occasionally omit
    one, so the identifier-derived URL stands in rather than leaving the row
    without any location to try.

    Parameters
    ----------
    entry : xml.etree.ElementTree.Element
        Atom ``entry`` element.
    arxiv_id : str
        Normalized arXiv identifier for the entry.

    Returns
    -------
    str
        PDF URL, or an empty string when the entry has no identifier.
    """
    for link in entry.findall(f'{{{ATOM_NS}}}link'):
        href = (link.get('href') or '').strip()
        if link.get('title') == 'pdf' and href:
            return href
    return f'{PDF_URL}/{arxiv_id}' if arxiv_id else ''


def _journal_name(journal_ref: str) -> str:
    """Reduce an arXiv journal reference to the journal name alone.

    ``arxiv:journal_ref`` is a free-text citation such as
    ``Phys. Rev. B 108, 014101 (2023)`` rather than a journal title. Enrichment
    fills only columns that are still empty, so a raw citation string written
    at search time could never be corrected later; the leading name is kept and
    the volume, pages, and year dropped. A reference whose name cannot be
    isolated is returned unchanged rather than emptied, and the unmodified
    string stays on the record as ``journal_ref``.

    Parameters
    ----------
    journal_ref : str
        Free-text journal reference supplied by the submitting author.

    Returns
    -------
    str
        Journal name, or an empty string when no reference was supplied.
    """
    text = ' '.join(str(journal_ref or '').split())
    if not text:
        return ''
    kept = []
    for token in text.replace(',', ' ').split():
        if token[0].isdigit() or token.startswith('('):
            break
        kept.append(token)
    return ' '.join(kept).strip(' ,;:') or text


def entry_to_paper(entry: ET.Element) -> _ArxivRecord:
    """Map one arXiv entry onto PaperScraper's paper schema.

    Authors may register a DOI for the published version of a preprint, in
    which case it is used as the paper identifier so the row merges with the
    published record rather than duplicating it.

    Parameters
    ----------
    entry : xml.etree.ElementTree.Element
        Atom ``entry`` element.

    Returns
    -------
    _ArxivRecord
        Normalized paper metadata plus the ``abstract``, ``categories``,
        ``primary_category``, ``journal_ref``, ``comment``, and ``version``
        extras that the corpus schema does not store directly.
    """
    raw_id = _element_text(entry.find(f'{{{ATOM_NS}}}id'))
    arxiv_id = normalize_arxiv_id(raw_id)
    raw_doi = _element_text(entry.find(f'{{{ARXIV_NS}}}doi'))
    doi = clean_doi(raw_doi) if raw_doi else ''
    if doi:
        paper_id = f'doi:{doi}'
    elif arxiv_id:
        paper_id = f'arxiv:{arxiv_id}'
    else:
        paper_id = ''
    categories = _categories(entry)
    journal_ref = _element_text(entry.find(f'{{{ARXIV_NS}}}journal_ref'))
    return {
        'paper_id': paper_id,
        'doi': doi,
        'arxiv_id': arxiv_id,
        'title': _element_text(entry.find(f'{{{ATOM_NS}}}title')),
        'journal': _journal_name(journal_ref),
        'publication_date': _publication_date(entry),
        'authors': _authors(entry),
        'sources': 'arxiv',
        'pdf_url': _pdf_url(entry, arxiv_id),
        'metadata_status': 'retrieved',
        'abstract': _element_text(entry.find(f'{{{ATOM_NS}}}summary')),
        'categories': categories,
        'primary_category': next((term['id'] for term in categories if term['is_primary']), ''),
        'journal_ref': journal_ref,
        'comment': _element_text(entry.find(f'{{{ARXIV_NS}}}comment')),
        'version': arxiv_version(raw_id),
    }


def parse_entries(root: ET.Element | None) -> list[_ArxivRecord]:
    """Map every entry in an arXiv feed onto the paper schema.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element or None
        Atom ``feed`` root, or ``None``.

    Returns
    -------
    list[_ArxivRecord]
        Normalized paper metadata, one mapping per entry.
    """
    if root is None:
        return []
    entries = root.findall(f'{{{ATOM_NS}}}entry')
    if not entries and root.tag == f'{{{ATOM_NS}}}entry':
        entries = [root]
    return [entry_to_paper(entry) for entry in entries]


def total_results(root: ET.Element | None) -> int:
    """Read the total match count an arXiv feed reports.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element or None
        Atom ``feed`` root, or ``None``.

    Returns
    -------
    int
        Total number of matches, or ``0`` when the feed reports none.
    """
    if root is None:
        return 0
    text = _element_text(root.find(f'{{{OPENSEARCH_NS}}}totalResults'))
    return int(text) if text.isdigit() else 0


def query_expression(query: str, default_field: str = 'all') -> str:
    """Translate a plain search phrase into an arXiv fielded expression.

    ``ps_search`` hands every provider the same bare phrase, but arXiv's
    ``search_query`` is a fielded language whose behaviour for an unprefixed
    multi-word query is undocumented. Terms are therefore joined with ``AND``
    under one field, matching how OpenAlex and PubMed read the same phrase, so
    result counts stay comparable across providers. An expression that already
    carries a field prefix or a boolean operator is passed through untouched,
    which is what lets a caller write a native arXiv query such as
    ``cat:cond-mat.mtrl-sci AND abs:"solid electrolyte"``. A date range counts
    as a field prefix for that purpose, so
    ``submittedDate:[20230101 TO 20240101]`` survives intact rather than being
    split into terms; without that it became
    ``all:submittedDate:[20230101 AND all:TO AND all:20240101]``, which arXiv
    rejects.

    Parameters
    ----------
    query : str
        Plain phrase or native arXiv search expression.
    default_field : str, default='all'
        Field prefix applied to the terms of a plain phrase.

    Returns
    -------
    str
        arXiv search expression, or an empty string for an empty query.

    Raises
    ------
    ValueError
        If ``default_field`` is not a field prefix arXiv accepts.
    """
    if default_field not in FIELD_PREFIXES:
        raise ValueError(f'default_field must be one of: {", ".join(FIELD_PREFIXES)}')
    expression = ' '.join(str(query or '').split())
    if not expression:
        return ''
    if re.search(r'\b(?:AND|OR|ANDNOT)\b', expression):
        return expression
    if re.search(rf'\b(?:{"|".join((*FIELD_PREFIXES, *RANGE_FIELDS))}):', expression):
        return expression
    terms = re.findall(r'"[^"]+"|\S+', expression)
    return ' AND '.join(f'{default_field}:{term}' for term in terms)


def search_page(query: str,
                start: int = 0,
                max_results: int = PAGE_SIZE,
                sort_by: str = 'submittedDate',
                sort_order: str = 'descending',
                session: provider.HTTPClient | None = None) -> ET.Element | None:
    """Fetch one page of arXiv search results.

    Parameters
    ----------
    query : str
        Fielded arXiv search expression.
    start : int, default=0
        Zero-based index of the first result requested.
    max_results : int, default=PAGE_SIZE
        Maximum number of results requested in this slice.
    sort_by : str, default='submittedDate'
        Result ordering, one of :data:`SORT_ORDERS`.
    sort_order : str, default='descending'
        Sort direction, one of :data:`SORT_DIRECTIONS`.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    xml.etree.ElementTree.Element or None
        Parsed feed root, or ``None`` for an empty response.

    Raises
    ------
    RuntimeError
        If the request cannot be completed or arXiv rejects the query.
    """
    params = request_params(search_query=query, start=start, max_results=max_results,
                            sort_by=sort_by, sort_order=sort_order)
    return request_xml(BASE_URL, params=params, session=session)


def fetch_ids(identifiers: Sequence[str],
              session: provider.HTTPClient | None = None) -> ET.Element | None:
    """Fetch the records for one batch of arXiv identifiers.

    Parameters
    ----------
    identifiers : Sequence[str]
        arXiv identifiers, at most :data:`ID_BATCH_SIZE` per call.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    xml.etree.ElementTree.Element or None
        Parsed feed root, or ``None`` when no identifiers were supplied.

    Raises
    ------
    RuntimeError
        If the request cannot be completed or arXiv rejects the identifiers.
    """
    wanted = [normalize_arxiv_id(identifier) for identifier in identifiers]
    wanted = [identifier for identifier in wanted if identifier]
    if not wanted:
        return None
    params = request_params(id_list=wanted, max_results=len(wanted))
    return request_xml(BASE_URL, params=params, session=session)


def find_arxiv_id(title: str, session: provider.HTTPClient | None = None) -> str:
    """Find the arXiv identifier for a paper title.

    arXiv has no DOI search field, so a title query is the only way to reach a
    record from an ordinary corpus row. arXiv matches titles loosely, so a hit
    counts only when its normalized title equals the one searched for.

    Parameters
    ----------
    title : str
        Paper title to look up.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    str
        Bare arXiv identifier, or an empty string when no title matches.

    Raises
    ------
    RuntimeError
        If the request cannot be completed.
    """
    wanted = _title_key(title)
    if not wanted:
        return ''
    quoted = re.sub(r'\W+', ' ', str(title)).strip()
    root = search_page(f'ti:"{quoted}"', max_results=TITLE_MATCH_LIMIT,
                       sort_by='relevance', session=session)
    for entry in parse_entries(root):
        if _title_key(entry.get('title')) == wanted:
            return str(entry.get('arxiv_id') or '')
    return ''


def resolve_arxiv_id(paper: Mapping[str, Any],
                     session: provider.HTTPClient | None = None) -> str:
    """Resolve one paper row's arXiv identifier.

    A stored identifier, an ``arxiv:`` paper identifier, or an arXiv URL
    already on the row is used without a request; otherwise the row's title is
    looked up.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    str
        Bare arXiv identifier, or an empty string when none can be resolved.

    Raises
    ------
    RuntimeError
        If an arXiv request cannot be completed.
    """
    identifier = normalize_arxiv_id(paper.get('arxiv_id'))
    if identifier:
        return identifier
    paper_id = str(paper.get('paper_id') or '')
    if paper_id.startswith('arxiv:'):
        identifier = normalize_arxiv_id(paper_id.split(':', 1)[1])
        if identifier:
            return identifier
    url = str(paper.get('pdf_url') or '')
    if 'arxiv.org' in url.lower():
        identifier = normalize_arxiv_id(url)
        if identifier:
            return identifier
    return find_arxiv_id(str(paper.get('title') or ''), session=session)
