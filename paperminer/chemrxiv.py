"""Request helpers for the chemRxiv API used by PaperMiner.

This module centralizes chemRxiv HTTP details and the mapping from chemRxiv API
records onto PaperMiner's paper schema, so search, download, and enrichment
code can share one implementation. The service needs neither an API key nor a
contact address, so there is nothing to configure before using it. It publishes
no rate limit either, so requests are paced through one module-level limiter at
a rate chosen to be unobtrusive rather than to satisfy a documented rule.

Unlike medRxiv and bioRxiv, chemRxiv publishes a real search endpoint: a query
is answered by the server from ``term``, ``skip``, and ``limit`` rather than by
reading the archive and matching locally. That is why this module has no
``matches``, ``page_cursors``, ``interval_page``, or ``MAX_SCAN_RECORDS``, and
why :mod:`paperminer.search` pages chemRxiv the way it pages arXiv instead of
walking it the way it walks the two ``_rxiv`` archives. The ``category:``,
``from:``, and ``to:`` scope terms are still accepted, because the corpus
documents them for preprint sources, but here they are forwarded to the server
as ``categoryIds``, ``searchDateFrom``, and ``searchDateTo`` rather than
applied to records after the fact.

chemRxiv serves PDFs and abstracts but no machine-readable full text, so there
is no ``full_text`` here and chemRxiv is not a member of
:data:`paperminer.download.TEXT_SOURCES`.

**The version suffix is part of a chemRxiv DOI and is never stripped.** This is
the one place where copying :mod:`paperminer.biorxiv` would be wrong. A
bioRxiv version is a URL suffix that is not part of the registered DOI, so
``normalize_biorxiv_doi`` drops it; a chemRxiv version is registered, and
dropping it yields an identifier that does not resolve. Checked against the
registry: ``10.26434/chemrxiv.15007737/v1`` resolves but
``10.26434/chemrxiv.15007737`` is unregistered, while conversely
``10.26434/chemrxiv-2022-w08rh`` resolves and ``10.26434/chemrxiv-2022-w08rh-v1``
is unregistered. What was issued is therefore kept verbatim, and
:func:`chemrxiv_stem` supplies the version-free key needed for grouping.

Five DOI shapes have to be recognized, because chemRxiv has moved between three
hosting platforms and reissued nothing. Across a sample of 800 live DOIs: a
bare dated accession ``10.26434/chemrxiv-2022-w08rh`` (265), the same with a
hyphenated version ``-v4`` (77) or a slashed one ``/v4`` (13), and a numeric
accession carrying a dotted version ``10.26434/chemrxiv.8011268.v1`` (322) or a
slashed one ``10.26434/chemrxiv.15007737/v1`` (123). Identifiers are recognized
by the ``chemrxiv`` token in the suffix rather than by the ``10.26434`` prefix
alone, which is what keeps a prefix change from stranding the older content and
keeps this module from claiming a DOI that belongs to another archive.

chemrxiv.org is fronted by a bot challenge that can refuse a client outright
with an HTTP 403, and it answers such a refusal with an HTML page rather than
JSON. Both shapes are reported as the failure reason, naming the challenge so
the cause is legible; neither is worked around. A 403 is terminal rather than
retried, because repeating a refused request only spends more of them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from paperminer import _rxiv, provider
from paperminer.metadata import clean_doi

BASE_URL = 'https://chemrxiv.org/engage/chemrxiv/public-api/v1'
WEB_URL = 'https://chemrxiv.org'
PAGE_SIZE = 50
MAX_SEARCH_RESULTS = 10000
CHEMRXIV_MIN_INTERVAL = 1.0
DEFAULT_SORT = 'PUBLISHED_DATE_DESC'
QUERY_PREFIXES = _rxiv.QUERY_PREFIXES
# Five suffix shapes across three hosting platforms, told apart by the
# ``chemrxiv`` token rather than by the prefix. The version is captured with
# its separator so it can be written back exactly as it was issued: the dated
# accession takes ``-v4`` or ``/v4``, the numeric one ``.v1`` or ``/v1``, and
# which of them is registered differs per record. The trailing lookahead stops
# a shorter accession from matching inside a longer one.
_CHEMRXIV_DOI = re.compile(
    r'(?P<stem>10\.\d{4,9}/chemrxiv(?:-\d{4}-[a-z0-9]+|\.\d{4,12}))'
    r'(?:(?P<sep>[-./])v(?P<version>\d+))?(?![\w-])',
    re.IGNORECASE,
)
_DATE = _rxiv.DATE_PATTERN
_ChemrxivRecord: TypeAlias = dict[str, Any]
LIMITER = provider.RateLimiter(CHEMRXIV_MIN_INTERVAL)
_categories_cache: list[dict[str, Any]] | None = None


def request_headers() -> dict[str, str]:
    """Build chemRxiv request headers.

    The headers name PaperMiner honestly. chemrxiv.org runs a bot challenge
    that may refuse this client, and that refusal is reported rather than
    disguised, so nothing here is chosen to look like a browser.

    Returns
    -------
    dict[str, str]
        Headers containing the PaperMiner user agent.
    """
    return provider.default_headers()


def _date(value: object) -> str:
    """Reduce an API timestamp to its ISO calendar date.

    chemRxiv dates a posting with a full ISO timestamp while the corpus stores
    a calendar date, so the time of day is dropped rather than stored and
    ignored.

    Parameters
    ----------
    value : object
        Raw API date value, such as ``2022-10-11T00:00:00.000Z``.

    Returns
    -------
    str
        Date as ``YYYY-MM-DD``, or an empty string when none is present.
    """
    text = provider.clean_text(value)
    match = re.match(r'(\d{4}-\d{2}-\d{2})', text)
    return match.group(1) if match else ''


def search_url() -> str:
    """Build the chemRxiv item-search endpoint URL.

    Returns
    -------
    str
        URL of the item search endpoint.
    """
    return f'{BASE_URL}/items'


def item_url(item_id: str) -> str:
    """Build the endpoint URL for one chemRxiv item.

    Parameters
    ----------
    item_id : str
        chemRxiv internal item identifier.

    Returns
    -------
    str
        URL of the single-item endpoint.
    """
    return f'{BASE_URL}/items/{item_id}'


def doi_url(doi: str) -> str:
    """Build the endpoint URL that looks one chemRxiv item up by DOI.

    Parameters
    ----------
    doi : str
        chemRxiv DOI.

    Returns
    -------
    str
        URL of the DOI lookup endpoint.
    """
    return f'{BASE_URL}/items/doi/{doi}'


def categories_url() -> str:
    """Build the chemRxiv category-list endpoint URL.

    Returns
    -------
    str
        URL of the category listing endpoint.
    """
    return f'{BASE_URL}/categories'


def pdf_url(doi: str) -> str:
    """Build the PDF location for a chemRxiv DOI.

    The location is derived from the DOI rather than read from the record's
    asset block. chemRxiv moved hosting platforms and the asset URLs recorded
    against older items point at the previous one, while the DOI-derived path
    is what the registry currently records for every item.

    No version argument is taken, because unlike bioRxiv the version is already
    part of the DOI.

    Parameters
    ----------
    doi : str
        chemRxiv DOI, in any of the shapes :func:`normalize_chemrxiv_doi`
        accepts.

    Returns
    -------
    str
        PDF URL, or an empty string when the value holds no chemRxiv DOI.
    """
    identifier = normalize_chemrxiv_doi(doi)
    return f'{WEB_URL}/doi/pdf/{identifier}' if identifier else ''


def landing_url(doi: str) -> str:
    """Build the landing-page location for a chemRxiv DOI.

    Parameters
    ----------
    doi : str
        chemRxiv DOI.

    Returns
    -------
    str
        Landing page URL, or an empty string when the value holds no chemRxiv
        DOI.
    """
    identifier = normalize_chemrxiv_doi(doi)
    return f'{WEB_URL}/doi/full/{identifier}' if identifier else ''


def request(
    url: str,
    params: Mapping[str, object] | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> provider.ResponseLike | None:
    """Request a chemRxiv endpoint with courtesy pacing and bounded retries.

    A 429 means the request rate was too high rather than that a budget is
    gone, so it is retried. Every other client error is terminal and fails at
    once, because retrying it would only spend more requests.

    A 403 is called out separately. chemrxiv.org is fronted by a bot challenge
    that refuses some clients outright, and a caller who is told only that a
    request was rejected has no way to tell that from a bad query. PaperMiner
    does not attempt to defeat the challenge, so the message says what happened
    and which providers can answer for the same papers instead.

    Parameters
    ----------
    url : str
        chemRxiv endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
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
        If the request is refused by the bot challenge, is otherwise rejected,
        or all request attempts fail.
    """
    def challenge(response: provider.ResponseLike) -> str:
        """Report the bot challenge chemRxiv answers a refused request with.

        Parameters
        ----------
        response : provider.ResponseLike
            Response to classify.

        Returns
        -------
        str
            Failure message for a 403, or an empty string for anything else.
        """
        if response.status_code != 403:
            return ''
        return (f'chemRxiv refused the request with 403 from {url}. chemrxiv.org is '
                f'behind a bot challenge that PaperMiner does not try to bypass; '
                f'the same papers can be reached through the openalex or crossref '
                f'sources.')

    return provider.request(url, label='chemRxiv', limiter=LIMITER, params=params,
                            session=session, timeout=timeout, attempts=attempts,
                            client_error=challenge)


def request_payload(
    url: str,
    params: Mapping[str, object] | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> Any:
    """Request a chemRxiv endpoint and decode its body, whatever its shape.

    The item endpoints answer with an object while the category listing answers
    with a bare array, so the decoded body is returned as it came and the
    caller decides what shape it expected.

    The bot challenge can answer an otherwise successful request with an HTML
    page, which fails to parse as JSON. That is reported as the challenge
    rather than as a malformed body, because the two have different causes and
    only one of them is worth retrying later.

    Parameters
    ----------
    url : str
        chemRxiv endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    timeout : int or float, default=60
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    Any
        Decoded response body, or ``None`` when chemRxiv holds no matching
        record.

    Raises
    ------
    RuntimeError
        If the request fails, the response is a challenge page, or the body is
        not well-formed JSON.
    """
    response = request(url, params=params, session=session, timeout=timeout, attempts=attempts)
    if response is None:
        return None
    try:
        return response.json()
    except ValueError as error:
        if (response.text or '').lstrip()[:1] == '<':
            raise RuntimeError(
                f'chemRxiv returned an HTML challenge page rather than JSON from {url}. '
                f'chemrxiv.org is behind a bot challenge that PaperMiner does not try '
                f'to bypass; the same papers can be reached through the openalex or '
                f'crossref sources.') from error
        raise RuntimeError(f'chemRxiv returned malformed JSON: {error}') from error


def request_json(
    url: str,
    params: Mapping[str, object] | None = None,
    session: provider.HTTPClient | None = None,
    timeout: float = 60,
    attempts: int = 4,
) -> _ChemrxivRecord | None:
    """Request a chemRxiv endpoint that answers with a JSON object.

    Parameters
    ----------
    url : str
        chemRxiv endpoint URL.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    timeout : int or float, default=60
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    dict[str, Any] or None
        Parsed payload, or ``None`` when chemRxiv holds no matching record.

    Raises
    ------
    RuntimeError
        If the request fails, the response is a challenge page, the body is not
        well-formed JSON, or the payload is not an object.
    """
    payload = request_payload(url, params=params, session=session, timeout=timeout,
                              attempts=attempts)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise RuntimeError(f'chemRxiv returned an unexpected payload of type {type(payload).__name__}')
    return dict(payload)


def normalize_chemrxiv_doi(value: object) -> str:
    """Normalize a chemRxiv identifier to the DOI as it was registered.

    A DOI, a ``doi:`` or ``chemrxiv:`` paper identifier, a resolver URL, and a
    chemRxiv landing or PDF URL are all accepted, in each of the five suffix
    shapes chemRxiv has issued. Case is folded, because DOIs are
    case-insensitive and the corpus compares them as strings.

    The version suffix is preserved exactly as it was found, separator
    included, and is never added or removed. Which form is registered differs
    per record: ``10.26434/chemrxiv.15007737/v1`` resolves while
    ``10.26434/chemrxiv.15007737`` does not, and ``10.26434/chemrxiv-2022-w08rh``
    resolves while ``10.26434/chemrxiv-2022-w08rh-v1`` does not. Normalizing
    the version away, as :func:`paperminer.biorxiv.normalize_biorxiv_doi`
    correctly does for an archive whose versions are not registered, would
    strand most of this one. Use :func:`chemrxiv_stem` when a version-free key
    is what is wanted.

    Parameters
    ----------
    value : object
        chemRxiv DOI, paper identifier, or content URL.

    Returns
    -------
    str
        chemRxiv DOI as registered, or an empty string when none is present.
    """
    if value is None:
        return ''
    match = _CHEMRXIV_DOI.search(str(value))
    if not match:
        return ''
    version = match.group('version')
    suffix = f"{match.group('sep')}v{version}" if version else ''
    return f"{match.group('stem')}{suffix}".lower()


def chemrxiv_stem(value: object) -> str:
    """Extract the version-free part of a chemRxiv identifier.

    This is a grouping key, not necessarily a resolvable DOI: for the numeric
    accession scheme the unversioned form is frequently unregistered. Use it to
    recognize two postings as versions of one preprint, and
    :func:`normalize_chemrxiv_doi` for anything that has to resolve.

    Parameters
    ----------
    value : object
        chemRxiv DOI, paper identifier, or content URL.

    Returns
    -------
    str
        Version-free identifier, or an empty string when none is present.
    """
    if value is None:
        return ''
    match = _CHEMRXIV_DOI.search(str(value))
    return match.group('stem').lower() if match else ''


def chemrxiv_version(value: object) -> str:
    """Extract the posted version number from a chemRxiv identifier.

    All three separators chemRxiv has used are read, so ``-v4``, ``/v4``, and
    ``.v4`` are equivalent here even though only one of them is registered for
    any given record.

    Parameters
    ----------
    value : object
        chemRxiv DOI or content URL, such as
        ``10.26434/chemrxiv-2024-bxxhh-v4``.

    Returns
    -------
    str
        Version number such as ``4``, or an empty string when unversioned.
    """
    if value is None:
        return ''
    match = _CHEMRXIV_DOI.search(str(value))
    return match.group('version') or '' if match else ''


def resolve_chemrxiv_doi(paper: Mapping[str, Any]) -> str:
    """Resolve one paper row's chemRxiv DOI from values it already holds.

    This never issues a request. A row that carries neither a chemRxiv DOI nor
    a chemRxiv URL cannot be reached without searching for it by title, which
    is not done here because it costs a request per row and can match the wrong
    record. A published DOI on the row belongs to the journal version and is
    not a chemRxiv identifier.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    str
        chemRxiv DOI as registered, or an empty string when the row stores
        none.
    """
    identifier = normalize_chemrxiv_doi(paper.get('chemrxiv_doi'))
    if identifier:
        return identifier
    for column in ['paper_id', 'doi', 'pdf_url']:
        value = str(paper.get(column) or '')
        if column == 'pdf_url' and 'chemrxiv.org' not in value.lower():
            continue
        identifier = normalize_chemrxiv_doi(value)
        if identifier:
            return identifier
    return ''


def _authors(value: object) -> str:
    """Reformat a chemRxiv author list into the corpus name order.

    chemRxiv supplies given and family names as separate fields, so unlike
    bioRxiv's ``Family, G. I.`` there is nothing to reorder; the two parts are
    joined in the order the rest of the corpus uses. A record that carries
    author names as one string is passed through, which is what keeps a
    consortium name intact.

    Parameters
    ----------
    value : object
        Author list as chemRxiv publishes it.

    Returns
    -------
    str
        Author names in record order, ``Given Family`` and semicolon-separated.
    """
    if isinstance(value, str):
        return provider.clean_text(value)
    if not isinstance(value, Sequence):
        return ''
    names = []
    for author in value:
        if isinstance(author, str):
            name = provider.clean_text(author)
        elif isinstance(author, Mapping):
            given = provider.clean_text(author.get('firstName'))
            family = provider.clean_text(author.get('lastName'))
            name = ' '.join(part for part in (given, family) if part)
        else:
            continue
        if name:
            names.append(name)
    return '; '.join(names)


def _categories(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect a record's chemRxiv subject categories.

    chemRxiv files a preprint under one or more categories, where medRxiv and
    bioRxiv allow exactly one, so this returns every category rather than the
    first. The first is flagged primary, matching the shape the other
    providers' subject helpers produce.

    Parameters
    ----------
    record : Mapping[str, Any]
        chemRxiv API record.

    Returns
    -------
    list[dict[str, Any]]
        Category terms in record order, the first flagged primary.
    """
    raw = record.get('categories')
    if isinstance(raw, Mapping) or isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence):
        return []
    categories = []
    for entry in raw:
        if isinstance(entry, Mapping):
            name = provider.clean_text(entry.get('name'))
            identifier = provider.clean_text(entry.get('id')) or name.lower()
        else:
            name = provider.clean_text(entry)
            identifier = name.lower()
        if not name:
            continue
        categories.append({'id': identifier, 'name': name,
                           'is_primary': not categories})
    return categories


def _keywords(record: Mapping[str, Any]) -> list[str]:
    """Collect a record's author-supplied keywords.

    Parameters
    ----------
    record : Mapping[str, Any]
        chemRxiv API record.

    Returns
    -------
    list[str]
        Keyword strings in record order, with blanks dropped.
    """
    raw = record.get('keywords')
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence):
        return []
    keywords = []
    for entry in raw:
        keyword = provider.clean_text(entry.get('name')) if isinstance(entry, Mapping) else provider.clean_text(entry)
        if keyword:
            keywords.append(keyword)
    return keywords


def _published_doi(record: Mapping[str, Any]) -> str:
    """Read the journal DOI a record names as its version of record.

    chemRxiv reports a preprint's published version under ``vor``. The DOI is
    taken from that block only; nothing is inferred when it is absent, because
    a wrong published DOI keys the row to the wrong paper and Crossref
    enrichment can supply the link correctly later.

    Parameters
    ----------
    record : Mapping[str, Any]
        chemRxiv API record.

    Returns
    -------
    str
        Published DOI, or an empty string when the record names none.
    """
    vor = record.get('vor')
    if not isinstance(vor, Mapping):
        return ''
    for field in ('vorDoi', 'doi', 'url'):
        doi = clean_doi(provider.clean_text(vor.get(field)))
        if doi:
            return doi
    return ''


def _asset_url(record: Mapping[str, Any]) -> str:
    """Read the asset location a record records for its PDF.

    This is kept as provenance rather than used as the download location.
    chemRxiv changed hosting platforms and older records still name the
    previous one, so :func:`pdf_url` derives the location from the DOI and this
    is only a fallback for a record whose DOI-derived path does not serve.

    Parameters
    ----------
    record : Mapping[str, Any]
        chemRxiv API record.

    Returns
    -------
    str
        Asset URL, or an empty string when the record names none.
    """
    asset = record.get('asset')
    if not isinstance(asset, Mapping):
        return ''
    original = asset.get('original')
    if isinstance(original, Mapping):
        return provider.clean_text(original.get('url'))
    return provider.clean_text(asset.get('url'))


def record_to_paper(record: Mapping[str, Any]) -> _ChemrxivRecord:
    """Map one chemRxiv API record onto PaperMiner's paper schema.

    A preprint that has since appeared in a journal names the published DOI
    under ``vor``, in which case it is used as the paper's DOI and identifier
    so the row merges with the published record rather than duplicating it. The
    preprint's own DOI is kept in ``chemrxiv_doi`` either way, because it is
    what reaches this API again later.

    ``journal`` is filled only for a preprint that has not been published.
    Enrichment fills only columns that are still empty, so writing ``chemRxiv``
    onto a row that names a published version would permanently mask the
    journal that Crossref holds for it.

    Parameters
    ----------
    record : Mapping[str, Any]
        chemRxiv API record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata plus the ``abstract``, ``categories``,
        ``category``, ``keywords``, ``license``, ``version``,
        ``chemrxiv_stem``, ``chemrxiv_id``, ``published_doi``, and
        ``asset_url`` extras that the corpus schema does not store directly.
    """
    chemrxiv_doi = normalize_chemrxiv_doi(record.get('doi'))
    published = _published_doi(record)
    doi = published or chemrxiv_doi
    categories = _categories(record)
    license_value = record.get('license')
    license_name = (provider.clean_text(license_value.get('name')) if isinstance(license_value, Mapping)
                    else provider.clean_text(license_value))
    return {
        'paper_id': f'doi:{doi}' if doi else '',
        'doi': doi,
        'chemrxiv_doi': chemrxiv_doi,
        'title': provider.clean_text(record.get('title')),
        'journal': '' if published else 'chemRxiv',
        'publication_date': _date(record.get('publishedDate') or record.get('submittedDate')),
        'authors': _authors(record.get('authors')),
        'sources': 'chemrxiv',
        'pdf_url': pdf_url(chemrxiv_doi),
        'metadata_status': 'retrieved',
        'abstract': provider.clean_text(record.get('abstract')),
        'categories': categories,
        'category': categories[0]['name'] if categories else '',
        'primary_category': categories[0]['id'] if categories else '',
        'keywords': _keywords(record),
        'license': license_name,
        'version': provider.clean_text(record.get('version')) or chemrxiv_version(chemrxiv_doi),
        'chemrxiv_stem': chemrxiv_stem(chemrxiv_doi),
        'chemrxiv_id': provider.clean_text(record.get('id')),
        'published_doi': published,
        'asset_url': _asset_url(record),
    }


def parse_records(payload: Mapping[str, Any] | None) -> list[_ChemrxivRecord]:
    """Map every record in a chemRxiv payload onto the paper schema.

    A search answers with items wrapped under ``itemHits``, while a single-item
    lookup answers with the item itself. Both shapes are read here so one
    parser serves every endpoint.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed chemRxiv payload, or ``None``.

    Returns
    -------
    list[dict[str, Any]]
        Normalized paper metadata, one mapping per record, in payload order.
    """
    if not payload:
        return []
    hits = payload.get('itemHits')
    if hits is None:
        hits = payload.get('items')
    if hits is None:
        item = payload.get('item')
        if isinstance(item, Mapping):
            return [record_to_paper(item)]
        return [record_to_paper(payload)] if payload.get('doi') else []
    if not isinstance(hits, Sequence) or isinstance(hits, str):
        return []
    records = []
    for hit in hits:
        if not isinstance(hit, Mapping):
            continue
        item = hit.get('item')
        records.append(record_to_paper(item if isinstance(item, Mapping) else hit))
    return records


def total_results(payload: Mapping[str, Any] | None) -> int:
    """Read the total result count a chemRxiv search payload reports.

    Parameters
    ----------
    payload : Mapping[str, Any] or None
        Parsed chemRxiv payload.

    Returns
    -------
    int
        Total number of matching items, or ``0`` when none is reported.
    """
    if not payload:
        return 0
    total = str(payload.get('totalCount') or '').strip()
    return int(total) if total.isdigit() else 0


def _version_rank(entry: Mapping[str, Any]) -> int:
    """Return a record's posted version as a sortable number.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Parsed chemRxiv record.

    Returns
    -------
    int
        Version number, or ``0`` when the record carries none.
    """
    return _rxiv._version_rank(entry)


def latest_versions(entries: Sequence[Mapping[str, Any]]) -> list[_ChemrxivRecord]:
    """Reduce parsed records to one entry per preprint, keeping the newest.

    Records are grouped on :func:`chemrxiv_stem` rather than on the DOI,
    because each posted version of a chemRxiv preprint carries a DOI of its
    own; grouping on the DOI would keep every version as a separate paper. The
    highest version wins field by field, only non-empty values overwrite, and
    each paper keeps the position of its first appearance so the caller's
    ordering survives.

    ``publication_date`` is the exception: it keeps the earliest date seen, so
    a paper holds the date it first appeared rather than the date of its most
    recent revision. That matches how :mod:`paperminer.arxiv` and
    :mod:`paperminer.biorxiv` date a revised preprint, which is what keeps a
    date-filtered corpus coherent across them. Both rules hold whichever order
    the versions arrive in.

    Parameters
    ----------
    entries : Sequence[Mapping[str, Any]]
        Parsed records from :func:`parse_records`.

    Returns
    -------
    list[dict[str, Any]]
        One record per preprint, in first-appearance order.
    """
    best: dict[str, _ChemrxivRecord] = {}
    for entry in entries:
        key = str(entry.get('chemrxiv_stem') or chemrxiv_stem(entry.get('chemrxiv_doi'))
                  or entry.get('paper_id') or '')
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


def parse_query(query: str) -> tuple[list[str], dict[str, str]]:
    """Split a chemRxiv search phrase into match terms and query scope.

    chemRxiv answers a search itself rather than being walked, so unlike the
    bioRxiv-family archives the scope is forwarded to the service rather than
    applied here. The grammar is the same one, so it is read by the same parser
    and the three sources stay comparable from a user's point of view.

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


def search_terms(terms: Sequence[str]) -> str:
    """Join query match terms into one chemRxiv search phrase.

    A term holding spaces came from a quoted phrase, so it is re-quoted to
    reach the server as one phrase rather than as separate words.

    Parameters
    ----------
    terms : Sequence[str]
        Match terms from :func:`parse_query`.

    Returns
    -------
    str
        Search phrase for the ``term`` parameter.
    """
    return ' '.join(f'"{term}"' if ' ' in term else term for term in terms if term)


def categories(session: provider.HTTPClient | None = None) -> list[dict[str, Any]]:
    """List the subject categories chemRxiv files preprints under.

    The listing is fetched once and cached for the process, because it changes
    far more slowly than a search runs and every scoped query needs it.

    Parameters
    ----------
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    list[dict[str, Any]]
        Category mappings carrying at least ``id`` and ``name``.

    Raises
    ------
    RuntimeError
        If the request cannot be completed.
    """
    global _categories_cache
    if _categories_cache is not None:
        return _categories_cache
    payload = request_payload(categories_url(), session=session)
    raw: Any = payload
    if isinstance(payload, Mapping):
        for field in ('categories', 'items', 'data'):
            if isinstance(payload.get(field), Sequence):
                raw = payload.get(field)
                break
    found = []
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            name = provider.clean_text(entry.get('name'))
            identifier = provider.clean_text(entry.get('id'))
            if name and identifier:
                found.append({'id': identifier, 'name': name})
    _categories_cache = found
    return found


def reset_categories_cache() -> None:
    """Discard the cached chemRxiv category list.

    The list is fetched once per process because it changes rarely and every
    category-scoped search needs it. A test run wants each case to start from
    the same state, so it is cleared through this rather than by reaching into
    the module global.

    Returns
    -------
    None
        The cache is cleared in place.
    """
    global _categories_cache
    _categories_cache = None


def category_ids(names: Sequence[str], session: provider.HTTPClient | None = None) -> list[str]:
    """Resolve chemRxiv category names to the identifiers the API filters on.

    An unmatched name raises rather than being dropped, because silently
    ignoring a filter returns the unfiltered archive and reads as a search that
    simply matched a lot.

    Parameters
    ----------
    names : Sequence[str]
        Category names as a query's ``category:`` terms spelled them.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    list[str]
        Category identifiers, in the order the names were given.

    Raises
    ------
    ValueError
        If a name matches no chemRxiv category.
    RuntimeError
        If the category listing cannot be fetched.
    """
    wanted = [provider.clean_text(name) for name in names if provider.clean_text(name)]
    if not wanted:
        return []
    listing = categories(session=session)
    by_name = {str(entry['name']).lower(): str(entry['id']) for entry in listing}
    resolved = []
    for name in wanted:
        identifier = by_name.get(name.lower())
        if identifier is None:
            known = ', '.join(sorted(entry['name'] for entry in listing)) or 'none reported'
            raise ValueError(f'category: {name!r} is not a chemRxiv category. Known categories: {known}')
        resolved.append(identifier)
    return resolved


def search_page(
    term: str = '',
    skip: int = 0,
    limit: int = PAGE_SIZE,
    sort: str = DEFAULT_SORT,
    category_id: str = '',
    date_from: str = '',
    date_to: str = '',
    session: provider.HTTPClient | None = None,
) -> _ChemrxivRecord | None:
    """Request one page of chemRxiv search results.

    Parameters
    ----------
    term : str, default=''
        Search phrase. An empty phrase lists the archive in ``sort`` order.
    skip : int, default=0
        Number of matching items to skip before the page starts.
    limit : int, default=PAGE_SIZE
        Maximum number of items in the page.
    sort : str, default=DEFAULT_SORT
        Result ordering requested of the server.
    category_id : str, default=''
        Category identifier to restrict the search to, from
        :func:`category_ids`.
    date_from : str, default=''
        Earliest posting date to include, as ``YYYY-MM-DD``.
    date_to : str, default=''
        Latest posting date to include, as ``YYYY-MM-DD``.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Parsed search payload, or ``None`` when chemRxiv reports no such page.

    Raises
    ------
    RuntimeError
        If the request cannot be completed.
    """
    params: dict[str, object] = {'skip': max(int(skip), 0),
                                 'limit': max(int(limit), 1),
                                 'sort': sort}
    if provider.clean_text(term):
        params['term'] = provider.clean_text(term)
    if provider.clean_text(category_id):
        params['categoryIds'] = provider.clean_text(category_id)
    if provider.clean_text(date_from):
        params['searchDateFrom'] = provider.clean_text(date_from)
    if provider.clean_text(date_to):
        params['searchDateTo'] = provider.clean_text(date_to)
    return request_json(search_url(), params=params, session=session)


def fetch_item(item_id: str, session: provider.HTTPClient | None = None) -> _ChemrxivRecord | None:
    """Fetch one chemRxiv item by its internal identifier.

    Parameters
    ----------
    item_id : str
        chemRxiv internal item identifier.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Normalized paper metadata, or ``None`` when no such item exists.

    Raises
    ------
    RuntimeError
        If the request cannot be completed.
    """
    if not provider.clean_text(item_id):
        return None
    entries = parse_records(request_json(item_url(provider.clean_text(item_id)), session=session))
    return entries[0] if entries else None


def fetch_doi(doi: str, session: provider.HTTPClient | None = None) -> _ChemrxivRecord | None:
    """Fetch one chemRxiv preprint by DOI.

    The DOI lookup route is tried first. When it reports nothing, the DOI is
    put through the search endpoint and the result matched on
    :func:`chemrxiv_stem`, which both covers a record reachable only by search
    and keeps this working if that route is withdrawn. Matching on the stem
    rather than the DOI means a request for one version can be answered by
    another, which is what the caller wants from a function that returns the
    current posting.

    Parameters
    ----------
    doi : str
        chemRxiv DOI, in any of the shapes
        :func:`normalize_chemrxiv_doi` accepts.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict[str, Any] or None
        Newest posting of the preprint, or ``None`` when chemRxiv holds no
        matching record.

    Raises
    ------
    RuntimeError
        If the request cannot be completed.
    """
    identifier = normalize_chemrxiv_doi(doi)
    if not identifier:
        return None
    entries = latest_versions(parse_records(request_json(doi_url(identifier), session=session)))
    if entries:
        return entries[0]
    stem = chemrxiv_stem(identifier)
    found = parse_records(search_page(term=identifier, limit=10, session=session))
    matched = [entry for entry in found if entry.get('chemrxiv_stem') == stem]
    entries = latest_versions(matched)
    return entries[0] if entries else None
