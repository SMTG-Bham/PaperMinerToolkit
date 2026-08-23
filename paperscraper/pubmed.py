"""Request helpers for the NCBI E-utilities API used by PaperScraper.

This module centralizes PubMed and PubMed Central HTTP details and the mapping
from PubMed article records onto PaperScraper's paper schema, so search,
download, and enrichment code can share one implementation. NCBI serves
unauthenticated clients at three requests per second and keyed clients at ten,
counted per IP address across every E-utilities endpoint, so all requests are
paced through one module-level limiter. An API key is optional but raises the
ceiling; a contact email lets NCBI warn before blocking an address.

Responses are parsed with :mod:`xml.etree.ElementTree`, which is documented as
vulnerable to entity-expansion attacks. NCBI over HTTPS is a trusted source and
the parser never resolves the external DTD PubMed declares, so this is safe
here; do not point these helpers at arbitrary user-supplied XML.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from paperscraper import provider
from paperscraper.metadata import clean_doi
from paperscraper.settings import load_settings

BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
ESEARCH_URL = f'{BASE_URL}/esearch.fcgi'
EFETCH_URL = f'{BASE_URL}/efetch.fcgi'
OA_URL = 'https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi'
FTP_PREFIX = 'ftp://ftp.ncbi.nlm.nih.gov/pub/pmc'
HTTPS_PREFIX = 'https://ftp.ncbi.nlm.nih.gov/pub/pmc'
TOOL_NAME = 'PaperScraper'
MAX_SEARCH_RESULTS = 10000
EFETCH_BATCH_SIZE = 200
NCBI_MIN_INTERVAL = 0.34
NCBI_KEYED_MIN_INTERVAL = 0.11
SORT_ORDERS = ('relevance', 'pub_date', 'Author', 'JournalName')
DATE_TYPES = ('pdat', 'mdat', 'edat')
JATS_SKIP_TAGS = frozenset({'table-wrap', 'fig', 'ref-list', 'supplementary-material',
                            'table', 'graphic', 'inline-graphic', 'media'})
_MONTHS = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
    'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
}
_PubMedRecord: TypeAlias = dict[str, Any]
LIMITER = provider.RateLimiter(NCBI_MIN_INTERVAL)


def configured_api_key(settings: Mapping[str, str] | None = None) -> str | None:
    """Return the configured NCBI API key.

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
    return settings.get('ncbi_api_key') or os.environ.get('NCBI_API_KEY')


def configured_email(settings: Mapping[str, str] | None = None) -> str:
    """Return the contact email address sent to NCBI.

    The Crossref address is reused when no NCBI-specific address is configured,
    so users who already identified themselves to Crossref need no new setup.

    Parameters
    ----------
    settings : Mapping[str, str] or None, optional
        Settings mapping to inspect before the environment.

    Returns
    -------
    str
        Contact address, or an empty string when none is configured.
    """
    settings = settings or load_settings()
    return (settings.get('ncbi_email')
            or os.environ.get('NCBI_EMAIL')
            or settings.get('crossref_email')
            or '')


def min_interval(api_key: str | None = None) -> float:
    """Return the minimum seconds between NCBI requests for one credential.

    Parameters
    ----------
    api_key : str or None, optional
        NCBI API key used for the requests being paced.

    Returns
    -------
    float
        Delay honoring three requests per second, or ten with an API key.
    """
    return NCBI_KEYED_MIN_INTERVAL if api_key else NCBI_MIN_INTERVAL


def request_headers() -> dict[str, str]:
    """Build E-utilities request headers.

    Returns
    -------
    dict[str, str]
        Headers containing the PaperScraper user agent.
    """
    return provider.default_headers()


def request_params(
    params: Mapping[str, object] | None = None,
    api_key: str | None = None,
    email: str = '',
) -> dict[str, object]:
    """Copy query parameters and add the tool name, contact address, and key.

    NCBI asks that clients identify themselves with ``tool`` and ``email`` so a
    contact can be warned before an address is blocked. Neither the key nor the
    address is ever invented; each is sent only when configured.

    Parameters
    ----------
    params : Mapping[str, object] or None, optional
        Query parameters to copy.
    api_key : str or None, optional
        NCBI API key to add.
    email : str, default=''
        Contact email address to add when non-empty.

    Returns
    -------
    dict[str, object]
        Copied parameters including ``tool`` and, when available, ``email`` and
        ``api_key``.
    """
    merged = dict(params or {})
    merged['tool'] = TOOL_NAME
    if email:
        merged['email'] = email
    if api_key:
        merged['api_key'] = api_key
    return merged


def _error_text(payload: object) -> str:
    """Extract an E-utilities error message from a decoded response body.

    Parameters
    ----------
    payload : object
        Decoded JSON payload or parsed XML element.

    Returns
    -------
    str
        Error message, or an empty string when the response reports no error.
    """
    if isinstance(payload, ET.Element):
        error = payload.find('.//ERROR')
        return _element_text(error) if error is not None else ''
    if isinstance(payload, Mapping):
        result = payload.get('esearchresult')
        if isinstance(result, Mapping):
            error = result.get('ERROR') or result.get('error')
            if error:
                return str(error)
        error = payload.get('ERROR') or payload.get('error')
        return str(error) if error else ''
    return ''


def request(
    url: str,
    params: Mapping[str, object] | None = None,
    api_key: str | None = None,
    email: str = '',
    session: provider.HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> provider.ResponseLike | None:
    """Request an E-utilities endpoint with NCBI pacing and bounded retries.

    Unlike OpenAlex, a 429 from NCBI means the per-second limit was exceeded
    rather than that a daily budget is gone, so it is retried rather than
    treated as terminal. Every other client error is terminal and fails at
    once, because retrying it would only spend more of the request budget.

    Parameters
    ----------
    url : str
        E-utilities endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method. Defaults to :mod:`requests`.
    timeout : int or float, default=60
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
    return provider.request(url, label='NCBI', limiter=LIMITER,
                            params=request_params(params, api_key, email),
                            session=session, timeout=timeout, attempts=attempts,
                            interval=min_interval(api_key))


def request_json(
    url: str,
    params: Mapping[str, object] | None = None,
    api_key: str | None = None,
    email: str = '',
    session: provider.HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> _PubMedRecord | None:
    """Request an E-utilities endpoint and decode its JSON payload.

    E-utilities reports a rejected query as an error member of a ``200``
    response rather than an unsuccessful status, so the body is inspected
    before it is returned.

    Parameters
    ----------
    url : str
        E-utilities endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    timeout : int or float, default=60
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    _PubMedRecord or None
        Decoded JSON payload, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If the request fails or the payload reports an error.
    """
    response = request(url, params=params, api_key=api_key, email=email,
                       session=session, timeout=timeout, attempts=attempts)
    if response is None:
        return None
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(f'NCBI returned an undecodable JSON payload: {error}') from error
    error_text = _error_text(payload)
    if error_text:
        raise RuntimeError(f'NCBI rejected the request: {error_text}')
    return payload


def request_xml(
    url: str,
    params: Mapping[str, object] | None = None,
    api_key: str | None = None,
    email: str = '',
    session: provider.HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> ET.Element | None:
    """Request an E-utilities endpoint and parse its XML payload.

    Parameters
    ----------
    url : str
        E-utilities endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    timeout : int or float, default=60
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    xml.etree.ElementTree.Element or None
        Parsed document root, or ``None`` for a 404 or empty response.

    Raises
    ------
    RuntimeError
        If the request fails, the payload is not well-formed, or it reports an
        error.
    """
    response = request(url, params=params, api_key=api_key, email=email,
                       session=session, timeout=timeout, attempts=attempts)
    if response is None:
        return None
    text = response.text or ''
    if not text.strip():
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise RuntimeError(f'NCBI returned malformed XML: {error}') from error
    error_text = _error_text(root)
    if error_text:
        raise RuntimeError(f'NCBI rejected the request: {error_text}')
    return root


def _search_params(term: str,
                   sort: str = 'relevance',
                   datetype: str = '',
                   mindate: str = '',
                   maxdate: str = '',
                   reldate: int = 0,
                   db: str = 'pubmed') -> dict[str, object]:
    """Build the shared esearch query parameters.

    Parameters
    ----------
    term : str
        PubMed search expression.
    sort : str, default='relevance'
        Result ordering requested from PubMed.
    datetype : str, default=''
        Date field used by ``mindate``, ``maxdate``, and ``reldate``.
    mindate : str, default=''
        Earliest publication date, as ``YYYY``, ``YYYY/MM``, or ``YYYY/MM/DD``.
    maxdate : str, default=''
        Latest publication date, in the same formats as ``mindate``.
    reldate : int, default=0
        Restrict results to the last ``reldate`` days when positive.
    db : str, default='pubmed'
        Entrez database to search.

    Returns
    -------
    dict[str, object]
        Query parameters excluding paging and history options.
    """
    params: dict[str, object] = {'db': db, 'term': term, 'retmode': 'json'}
    if sort:
        params['sort'] = sort
    if datetype:
        params['datetype'] = datetype
    if mindate:
        params['mindate'] = mindate
    if maxdate:
        params['maxdate'] = maxdate
    if reldate:
        params['reldate'] = int(reldate)
    return params


def esearch(term: str,
            retmax: int = 100,
            retstart: int = 0,
            sort: str = 'relevance',
            datetype: str = '',
            mindate: str = '',
            maxdate: str = '',
            reldate: int = 0,
            db: str = 'pubmed',
            api_key: str | None = None,
            email: str = '',
            session: provider.HTTPClient | None = None) -> tuple[list[str], int]:
    """Run one esearch page and return its identifiers with the match count.

    Parameters
    ----------
    term : str
        PubMed search expression.
    retmax : int, default=100
        Maximum identifiers to return, capped at the E-utilities maximum.
    retstart : int, default=0
        Zero-based index of the first identifier to return.
    sort : str, default='relevance'
        Result ordering requested from PubMed.
    datetype : str, default=''
        Date field used by ``mindate``, ``maxdate``, and ``reldate``.
    mindate : str, default=''
        Earliest publication date.
    maxdate : str, default=''
        Latest publication date.
    reldate : int, default=0
        Restrict results to the last ``reldate`` days when positive.
    db : str, default='pubmed'
        Entrez database to search.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    tuple[list[str], int]
        Identifiers on this page, and the total number of matches.

    Raises
    ------
    RuntimeError
        If the E-utilities request cannot be completed.
    """
    params = _search_params(term, sort, datetype, mindate, maxdate, reldate, db)
    params['retmax'] = max(0, min(int(retmax), MAX_SEARCH_RESULTS))
    params['retstart'] = max(0, int(retstart))
    payload = request_json(ESEARCH_URL, params=params, api_key=api_key,
                           email=email, session=session) or {}
    result = payload.get('esearchresult') or {}
    identifiers = [str(value) for value in result.get('idlist') or []]
    try:
        total = int(result.get('count') or 0)
    except (TypeError, ValueError):
        total = 0
    return identifiers, total


def esearch_history(term: str,
                    sort: str = 'relevance',
                    datetype: str = '',
                    mindate: str = '',
                    maxdate: str = '',
                    reldate: int = 0,
                    db: str = 'pubmed',
                    api_key: str | None = None,
                    email: str = '',
                    session: provider.HTTPClient | None = None) -> tuple[str, str, int]:
    """Run esearch with history and return its stored-set handles.

    Storing the result set server-side lets efetch page through it without
    resending long identifier lists, which is what NCBI recommends and what
    keeps request URLs within length limits.

    Parameters
    ----------
    term : str
        PubMed search expression.
    sort : str, default='relevance'
        Result ordering requested from PubMed.
    datetype : str, default=''
        Date field used by ``mindate``, ``maxdate``, and ``reldate``.
    mindate : str, default=''
        Earliest publication date.
    maxdate : str, default=''
        Latest publication date.
    reldate : int, default=0
        Restrict results to the last ``reldate`` days when positive.
    db : str, default='pubmed'
        Entrez database to search.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    tuple[str, str, int]
        Web environment, query key, and the total number of matches.

    Raises
    ------
    RuntimeError
        If the E-utilities request cannot be completed.
    """
    params = _search_params(term, sort, datetype, mindate, maxdate, reldate, db)
    params['usehistory'] = 'y'
    params['retmax'] = 0
    payload = request_json(ESEARCH_URL, params=params, api_key=api_key,
                           email=email, session=session) or {}
    result = payload.get('esearchresult') or {}
    try:
        total = int(result.get('count') or 0)
    except (TypeError, ValueError):
        total = 0
    return str(result.get('webenv') or ''), str(result.get('querykey') or ''), total


def efetch_history(webenv: str,
                   query_key: str,
                   retstart: int = 0,
                   retmax: int = EFETCH_BATCH_SIZE,
                   db: str = 'pubmed',
                   api_key: str | None = None,
                   email: str = '',
                   session: provider.HTTPClient | None = None) -> ET.Element | None:
    """Fetch one page of records from a stored E-utilities history set.

    Parameters
    ----------
    webenv : str
        Web environment returned by :func:`esearch_history`.
    query_key : str
        Query key returned by :func:`esearch_history`.
    retstart : int, default=0
        Zero-based index of the first record to fetch.
    retmax : int, default=200
        Maximum records to fetch in this request.
    db : str, default='pubmed'
        Entrez database to fetch from.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    xml.etree.ElementTree.Element or None
        Parsed record set, or ``None`` when the page is empty.

    Raises
    ------
    RuntimeError
        If the E-utilities request cannot be completed.
    """
    params = {
        'db': db,
        'WebEnv': webenv,
        'query_key': query_key,
        'retstart': max(0, int(retstart)),
        'retmax': max(1, int(retmax)),
        'retmode': 'xml',
    }
    return request_xml(EFETCH_URL, params=params, api_key=api_key, email=email, session=session)


def efetch_ids(identifiers: Sequence[str],
               db: str = 'pubmed',
               api_key: str | None = None,
               email: str = '',
               session: provider.HTTPClient | None = None) -> ET.Element | None:
    """Fetch records for an explicit identifier list in one efetch request.

    Parameters
    ----------
    identifiers : Sequence[str]
        Entrez identifiers to fetch.
    db : str, default='pubmed'
        Entrez database to fetch from.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    xml.etree.ElementTree.Element or None
        Parsed record set, or ``None`` when no identifiers were supplied.

    Raises
    ------
    RuntimeError
        If the E-utilities request cannot be completed.
    """
    wanted = [str(value).strip() for value in identifiers if str(value).strip()]
    if not wanted:
        return None
    params = {'db': db, 'id': ','.join(wanted), 'retmode': 'xml'}
    return request_xml(EFETCH_URL, params=params, api_key=api_key, email=email, session=session)


def normalize_pmid(value: object) -> str:
    """Normalize a PubMed identifier to its bare digits.

    Parameters
    ----------
    value : object
        PMID, PubMed URL, or other identifier-like value.

    Returns
    -------
    str
        Bare PMID digits, or an empty string when none is present.
    """
    if value is None:
        return ''
    match = re.search(r'\d+', str(value))
    return match.group(0) if match else ''


def normalize_pmcid(value: object) -> str:
    """Normalize a PubMed Central identifier to its ``PMC`` form.

    Parameters
    ----------
    value : object
        PMCID, PMC URL, or other identifier-like value.

    Returns
    -------
    str
        Identifier as ``PMC`` followed by digits, or an empty string.
    """
    if value is None:
        return ''
    match = re.search(r'\d+', str(value))
    return f'PMC{match.group(0)}' if match else ''


def _element_text(element: ET.Element | None) -> str:
    """Flatten an element's text and inline markup into one plain string.

    PubMed wraps formatting such as subscripts and italics in child elements,
    so reading ``element.text`` alone would truncate a title at its first tag.

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


def _abstract_text(parent: ET.Element) -> str:
    """Join an abstract's sections, prefixing any section labels.

    Structured abstracts arrive as sibling ``AbstractText`` elements carrying a
    ``Label``. Only those elements are read, which also excludes the sibling
    ``CopyrightInformation`` notice from the stored abstract.

    Parameters
    ----------
    parent : xml.etree.ElementTree.Element
        Element containing an ``Abstract``.

    Returns
    -------
    str
        Abstract text, or an empty string when the record has no abstract.
    """
    sections = []
    for section in parent.findall('.//Abstract/AbstractText'):
        text = _element_text(section)
        if not text:
            continue
        label = (section.get('Label') or '').strip()
        if label and label.upper() != 'UNLABELLED':
            text = f'{label}: {text}'
        sections.append(text)
    return ' '.join(sections)


def _publication_date(article: ET.Element) -> str:
    """Extract the best available publication date from an article element.

    The electronic ``ArticleDate`` is preferred because it is already a
    year/month/day triple. A free-text ``MedlineDate`` such as ``2019 Jan-Feb``
    contributes only its year, since guessing a month would be worse than
    omitting one.

    Parameters
    ----------
    article : xml.etree.ElementTree.Element
        ``Article`` or ``BookDocument`` element.

    Returns
    -------
    str
        Publication date as ``YYYY``, ``YYYY-MM``, or ``YYYY-MM-DD``.
    """
    for candidate in [article.find('./ArticleDate'), article.find('.//PubDate')]:
        if candidate is None:
            continue
        year = _element_text(candidate.find('./Year'))
        if year:
            parts = [year]
            month = _month_number(_element_text(candidate.find('./Month')))
            if month:
                parts.append(month)
                day = _element_text(candidate.find('./Day'))
                if day.isdigit():
                    parts.append(day.zfill(2))
            return '-'.join(parts)
        medline = _element_text(candidate.find('./MedlineDate'))
        match = re.search(r'\d{4}', medline)
        if match:
            return match.group(0)
    return ''


def _month_number(value: str) -> str:
    """Normalize a PubMed month to two digits.

    PubMed records a month as either an abbreviated name or a number, and both
    forms occur in the same corpus.

    Parameters
    ----------
    value : str
        Month name or number.

    Returns
    -------
    str
        Two-digit month, or an empty string when the value is unusable.
    """
    value = value.strip()
    if value.isdigit():
        number = int(value)
        return f'{number:02d}' if 1 <= number <= 12 else ''
    return _MONTHS.get(value.lower()[:3], '')


def _author_name(author: ET.Element) -> str:
    """Format one author element as a display name.

    Parameters
    ----------
    author : xml.etree.ElementTree.Element
        ``Author`` element.

    Returns
    -------
    str
        Personal or collective name, or an empty string when unusable.
    """
    collective = _element_text(author.find('./CollectiveName'))
    if collective:
        return collective
    family = _element_text(author.find('./LastName'))
    if not family:
        return ''
    given = _element_text(author.find('./ForeName')) or _element_text(author.find('./Initials'))
    return f'{given} {family}'.strip()


def _authors(parent: ET.Element) -> str:
    """Format an article's author list as a semicolon-separated string.

    Parameters
    ----------
    parent : xml.etree.ElementTree.Element
        Element containing an ``AuthorList``.

    Returns
    -------
    str
        Author names in record order.
    """
    names = []
    for author in parent.findall('.//AuthorList/Author'):
        if author.get('ValidYN') == 'N':
            continue
        name = _author_name(author)
        if name:
            names.append(name)
    return '; '.join(names)


def _article_ids(record: ET.Element) -> dict[str, str]:
    """Collect the DOI, PMID, and PMCID recorded for one article.

    Parameters
    ----------
    record : xml.etree.ElementTree.Element
        ``PubmedArticle`` or ``PubmedBookArticle`` element.

    Returns
    -------
    dict[str, str]
        Mapping with ``doi``, ``pmid``, and ``pmcid`` keys, empty when absent.
    """
    ids = {'doi': '', 'pmid': '', 'pmcid': ''}
    # Scoped to the record's own identifier list: a PubmedArticle also carries
    # an ArticleIdList inside every cited reference, and a '//' search would
    # take a reference's DOI for any record that omits its own.
    id_list = record.find('./PubmedData/ArticleIdList')
    if id_list is None:
        id_list = record.find('./PubmedBookData/ArticleIdList')
    for article_id in (id_list if id_list is not None else []):
        id_type = (article_id.get('IdType') or '').lower()
        value = _element_text(article_id)
        if not value:
            continue
        if id_type == 'doi' and not ids['doi']:
            ids['doi'] = clean_doi(value)
        elif id_type in {'pubmed', 'pmid'} and not ids['pmid']:
            ids['pmid'] = normalize_pmid(value)
        elif id_type == 'pmc' and not ids['pmcid']:
            ids['pmcid'] = normalize_pmcid(value)
    if not ids['pmid']:
        ids['pmid'] = normalize_pmid(_element_text(record.find('.//PMID')))
    return ids


def _mesh_terms(record: ET.Element) -> list[dict[str, str]]:
    """Collect MeSH descriptors and qualifiers from one article.

    Parameters
    ----------
    record : xml.etree.ElementTree.Element
        ``PubmedArticle`` element.

    Returns
    -------
    list[dict[str, str]]
        Terms with ``scheme``, ``id``, ``name``, and ``is_primary`` keys.
    """
    terms = []
    for heading in record.findall('.//MeshHeadingList/MeshHeading'):
        for scheme, tag in [('mesh', 'DescriptorName'), ('mesh_qualifier', 'QualifierName')]:
            for element in heading.findall(f'./{tag}'):
                name = _element_text(element)
                if not name:
                    continue
                terms.append({
                    'scheme': scheme,
                    'id': element.get('UI') or name,
                    'name': name,
                    'is_primary': '1' if element.get('MajorTopicYN') == 'Y' else '0',
                })
    return terms


def _publication_types(record: ET.Element) -> list[dict[str, str]]:
    """Collect the publication types recorded for one article.

    Parameters
    ----------
    record : xml.etree.ElementTree.Element
        ``PubmedArticle`` or ``PubmedBookArticle`` element.

    Returns
    -------
    list[dict[str, str]]
        Types with ``id`` and ``name`` keys.
    """
    types = []
    for element in record.findall('.//PublicationTypeList/PublicationType'):
        name = _element_text(element)
        if name:
            types.append({'id': element.get('UI') or name, 'name': name})
    return types


def _keywords(record: ET.Element) -> list[str]:
    """Collect author-supplied keywords from one article.

    Parameters
    ----------
    record : xml.etree.ElementTree.Element
        ``PubmedArticle`` or ``PubmedBookArticle`` element.

    Returns
    -------
    list[str]
        Keyword text in record order.
    """
    return [text for text in (_element_text(element)
                              for element in record.findall('.//KeywordList/Keyword')) if text]


def article_to_paper(record: ET.Element) -> _PubMedRecord:
    """Map one PubMed record onto PaperScraper's paper schema.

    Book records nest their metadata under ``BookDocument`` and have no
    ``Article`` element, so they are read from a different path rather than
    dropped; they carry PMIDs and appear in ordinary PubMed searches.

    Parameters
    ----------
    record : xml.etree.ElementTree.Element
        ``PubmedArticle`` or ``PubmedBookArticle`` element.

    Returns
    -------
    _PubMedRecord
        Normalized paper metadata plus the ``abstract``, ``mesh``,
        ``keywords``, ``publication_types``, and ``article_type`` extras that
        the corpus schema does not store directly.
    """
    is_book = record.tag == 'PubmedBookArticle'
    article = record.find('.//BookDocument') if is_book else record.find('.//Article')
    if article is None:
        article = record
    if is_book:
        title = (_element_text(article.find('./ArticleTitle'))
                 or _element_text(article.find('./Book/BookTitle')))
        journal = _element_text(article.find('./Book/BookTitle'))
    else:
        title = _element_text(article.find('./ArticleTitle'))
        journal = (_element_text(article.find('./Journal/Title'))
                   or _element_text(article.find('./Journal/ISOAbbreviation')))
    ids = _article_ids(record)
    if ids['doi']:
        paper_id = f'doi:{ids["doi"]}'
    elif ids['pmid']:
        paper_id = f'pmid:{ids["pmid"]}'
    else:
        paper_id = ''
    return {
        'paper_id': paper_id,
        'doi': ids['doi'],
        'pmid': ids['pmid'],
        'pmcid': ids['pmcid'],
        'title': title,
        'journal': journal,
        'publication_date': _publication_date(article),
        'authors': _authors(article),
        'sources': 'pubmed',
        'metadata_status': 'retrieved',
        'abstract': _abstract_text(article),
        'mesh': _mesh_terms(record),
        'keywords': _keywords(record),
        'publication_types': _publication_types(record),
        'article_type': 'book' if is_book else 'article',
    }


def parse_articles(root: ET.Element | None) -> list[_PubMedRecord]:
    """Map every record in a PubMed record set onto the paper schema.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element or None
        ``PubmedArticleSet`` root, or ``None``.

    Returns
    -------
    list[_PubMedRecord]
        Normalized paper metadata, one mapping per record.
    """
    if root is None:
        return []
    records = [child for child in root if child.tag in {'PubmedArticle', 'PubmedBookArticle'}]
    if not records and root.tag in {'PubmedArticle', 'PubmedBookArticle'}:
        records = [root]
    return [article_to_paper(record) for record in records]


def find_pmid(doi: str,
              api_key: str | None = None,
              email: str = '',
              session: provider.HTTPClient | None = None) -> str:
    """Look up the PubMed identifier that PubMed records against a DOI.

    The PMC ID converter service is not usable for this: it answers requests
    with ``403``. Searching the ``AID`` article-identifier field stays on the
    E-utilities endpoint that every other request already uses and shares its
    pacing.

    Parameters
    ----------
    doi : str
        DOI to look up.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    str
        Bare PMID digits, or an empty string when PubMed indexes no such DOI.

    Raises
    ------
    RuntimeError
        If the E-utilities request cannot be completed.
    """
    doi = clean_doi(doi) if doi else ''
    if not doi:
        return ''
    identifiers, _ = esearch(f'{doi}[AID]', retmax=1, sort='', api_key=api_key,
                             email=email, session=session)
    return normalize_pmid(identifiers[0]) if identifiers else ''


def resolve_pmid(paper: Mapping[str, Any],
                 api_key: str | None = None,
                 email: str = '',
                 session: provider.HTTPClient | None = None) -> str:
    """Resolve one paper row's PubMed identifier.

    A stored PMID is returned without a request; otherwise the row's DOI is
    looked up in PubMed.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    str
        Bare PMID digits, or an empty string when none can be resolved.

    Raises
    ------
    RuntimeError
        If an E-utilities request cannot be completed.
    """
    pmid = normalize_pmid(paper.get('pmid'))
    if pmid:
        return pmid
    paper_id = str(paper.get('paper_id') or '')
    if paper_id.startswith('pmid:'):
        return normalize_pmid(paper_id.split(':', 1)[1])
    return find_pmid(str(paper.get('doi') or ''), api_key=api_key,
                     email=email, session=session)


def resolve_pmcid(paper: Mapping[str, Any],
                  api_key: str | None = None,
                  email: str = '',
                  session: provider.HTTPClient | None = None) -> str:
    """Resolve one paper row's PubMed Central identifier.

    A record's own identifier list already carries its PMCID, so the PMID is
    resolved first and the identifier is read from that record.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    str
        Identifier as ``PMC`` followed by digits, or an empty string.

    Raises
    ------
    RuntimeError
        If an E-utilities request cannot be completed.
    """
    pmcid = normalize_pmcid(paper.get('pmcid'))
    if pmcid:
        return pmcid
    pmid = resolve_pmid(paper, api_key=api_key, email=email, session=session)
    if not pmid:
        return ''
    root = efetch_ids([pmid], api_key=api_key, email=email, session=session)
    records = [child for child in (root if root is not None else [])
               if child.tag in {'PubmedArticle', 'PubmedBookArticle'}]
    return _article_ids(records[0])['pmcid'] if records else ''


def _https_urls(url: str) -> list[str]:
    """Rewrite an NCBI FTP link to the HTTPS locations that may serve it.

    The open-access service still advertises ``ftp://`` links, which the
    :mod:`requests`-based downloaders cannot fetch. NCBI moved the article
    datasets in 2026 and left the previous tree under a ``deprecated``
    directory that it has announced it will remove, so both the current path
    and that mirror are offered. Whichever one NCBI is serving answers; the
    other returns a cheap 404.

    Parameters
    ----------
    url : str
        Link advertised by the open-access service.

    Returns
    -------
    list[str]
        HTTPS candidates in the order they should be tried, or the original
        value alone when it is not an NCBI FTP link.
    """
    if not url.startswith(FTP_PREFIX):
        return [url]
    path = url[len(FTP_PREFIX):].lstrip('/')
    return [f'{HTTPS_PREFIX}/deprecated/{path}', f'{HTTPS_PREFIX}/{path}']


def oa_package_urls(pmcid: str,
                    session: provider.HTTPClient | None = None,
                    api_key: str | None = None,
                    email: str = '',
                    timeout: float = 60) -> list[str]:
    """List the open-access service's links for a PubMed Central record.

    Only the open-access subset is redistributable, so a record outside it
    yields no links rather than an error worth retrying.

    Parameters
    ----------
    pmcid : str
        PubMed Central identifier.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    timeout : int or float, default=60
        Request timeout in seconds.

    Returns
    -------
    list[str]
        HTTPS links, PDF links first, or an empty list when none are offered.

    Raises
    ------
    RuntimeError
        If the open-access request cannot be completed.
    """
    identifier = normalize_pmcid(pmcid)
    if not identifier:
        return []
    root = request_xml(OA_URL, params={'id': identifier}, api_key=api_key,
                       email=email, session=session, timeout=timeout)
    if root is None:
        return []
    pdfs, packages = [], []
    for link in root.findall('.//record/link'):
        href = (link.get('href') or '').strip()
        if not href:
            continue
        target = pdfs if (link.get('format') or '').lower() == 'pdf' else packages
        target.extend(_https_urls(href))
    return list(dict.fromkeys(pdfs + packages))


def _jats_blocks(node: ET.Element, blocks: list[str]) -> None:
    """Collect prose blocks from a JATS subtree in document order.

    Tables, figures, reference lists, and supplementary material are skipped
    whole so the stored text is prose rather than markup residue.

    Parameters
    ----------
    node : xml.etree.ElementTree.Element
        Subtree to walk.
    blocks : list[str]
        Accumulator appended to in place.

    Returns
    -------
    None
        Blocks are appended to ``blocks``.
    """
    for child in node:
        if child.tag in JATS_SKIP_TAGS:
            continue
        if child.tag in {'title', 'p'}:
            text = _element_text(child)
            if text:
                blocks.append(text)
            continue
        _jats_blocks(child, blocks)


def _jats_body_text(root: ET.Element) -> str:
    """Flatten a JATS body into section and paragraph text.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Parsed JATS article.

    Returns
    -------
    str
        Section titles and paragraphs separated by blank lines.
    """
    body = root.find('.//body')
    if body is None:
        return ''
    blocks: list[str] = []
    _jats_blocks(body, blocks)
    return '\n\n'.join(blocks)


def pmc_full_text(pmcid: str,
                  api_key: str | None = None,
                  email: str = '',
                  session: provider.HTTPClient | None = None) -> str:
    """Fetch and flatten a PubMed Central open-access record to plain text.

    Parameters
    ----------
    pmcid : str
        PubMed Central identifier.
    api_key : str or None, optional
        NCBI API key to attach.
    email : str, default=''
        Contact email address sent with the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    str
        Article text, or an empty string when no body is available.

    Raises
    ------
    RuntimeError
        If the E-utilities request cannot be completed.
    """
    identifier = normalize_pmcid(pmcid)
    if not identifier:
        return ''
    root = efetch_ids([identifier[3:]], db='pmc', api_key=api_key, email=email, session=session)
    if root is None:
        return ''
    title = _element_text(root.find('.//article-title'))
    body = _jats_body_text(root)
    if not body:
        return ''
    return f'{title}\n\n{body}'.strip() if title else body
