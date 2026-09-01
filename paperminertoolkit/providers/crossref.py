"""Discover an author's works through Crossref and import their metadata.

Crossref serves two pools. A client that names a contact address is routed onto
the polite pool and allowed ten requests per second; one that does not is
served by the public pool at five. Membership costs nothing but the address,
which travels both in the user agent and as a ``mailto`` query parameter, and
the current allowance is announced on every response in ``X-Rate-Limit-Limit``
and ``X-Rate-Limit-Interval``. Every entry point here resolves a contact
address before requesting, so in practice these helpers always qualify for the
polite pool; the public-pool pace is the floor for anything that reaches
Crossref without one.

Requests are issued one at a time through a single module-level limiter, so the
concurrency each pool allows never becomes a factor.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias
from urllib.parse import quote

import pandas as pd
import requests

from paperminertoolkit.corpus.database import connect, find_paper, upsert_paper, upsert_papers
from paperminertoolkit.providers import base as provider
from paperminertoolkit.corpus.metadata import clean_doi
from paperminertoolkit.settings import load_settings


CROSSREF_WORKS_URL = 'https://api.crossref.org/v1/works'
CROSSREF_SINGLE_WORK_URL = 'https://api.crossref.org/works'
CROSSREF_PUBLIC_MIN_INTERVAL = 0.2
CROSSREF_POLITE_MIN_INTERVAL = 0.1
MAX_FILTER_VALUES = 100
BATCHABLE_DOI_PATTERN = re.compile(r'^10\.\d{4,9}/[^\s,]+$')
ORCID_PATTERN = re.compile(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', re.IGNORECASE)
REVIEW_COLUMNS = ['paper_id', 'doi', 'title', 'journal', 'publication_date', 'authors', 'sources']
_CrossrefRecord: TypeAlias = dict[str, Any]
LIMITER = provider.RateLimiter(CROSSREF_PUBLIC_MIN_INTERVAL)


def normalize_orcid(value: str | None) -> str:
    """Normalize and validate an ORCID identifier.

    Parameters
    ----------
    value : str or None
        Bare ORCID or ORCID URL.

    Returns
    -------
    str
        Upper-case, hyphenated ORCID identifier.

    Raises
    ------
    ValueError
        If the identifier format or checksum is invalid.
    """
    orcid = str(value or '').strip().rstrip('/').split('/')[-1].upper()
    if not ORCID_PATTERN.fullmatch(orcid):
        raise ValueError(f'Invalid ORCID: {value}')
    compact = orcid.replace('-', '')
    total = 0
    for digit in compact[:-1]:
        total = (total + int(digit)) * 2
    checksum = (12 - (total % 11)) % 11
    expected = 'X' if checksum == 10 else str(checksum)
    if compact[-1] != expected:
        raise ValueError(f'Invalid ORCID checksum: {value}')
    return orcid


def _normalize_text(value: object) -> str:
    """Normalize human-readable metadata for comparison.

    Parameters
    ----------
    value : object
        Value to normalize as text.

    Returns
    -------
    str
        Lower-case ASCII comparison key.
    """
    value = unicodedata.normalize('NFKD', str(value or ''))
    value = ''.join(character for character in value if not unicodedata.combining(character))
    return re.sub(r'[^a-z0-9]+', ' ', value.lower()).strip()


def _given_names_match(target: object, candidate: object) -> bool:
    """Compare given names while allowing initials.

    Parameters
    ----------
    target : object
        Requested given names.
    candidate : object
        Deposited given names to compare.

    Returns
    -------
    bool
        Whether the normalized names match.
    """
    target_parts = _normalize_text(target).split()
    candidate_parts = _normalize_text(candidate).split()
    if not target_parts or not candidate_parts:
        return False
    if target_parts == candidate_parts:
        return True
    if all(len(part) > 1 for part in target_parts + candidate_parts):
        return False
    return ''.join(part[0] for part in target_parts) == ''.join(part[0] for part in candidate_parts)


def _matching_authors(work: _CrossrefRecord, author_name: str) -> list[_CrossrefRecord]:
    """Find Crossref authors matching a human name.

    Parameters
    ----------
    work : _CrossrefRecord
        Crossref work record containing author metadata.
    author_name : str
        Given and family names to match.

    Returns
    -------
    list[_CrossrefRecord]
        Matching Crossref author records.

    Raises
    ------
    ValueError
        If ``author_name`` does not contain given and family names.
    """
    parts = _normalize_text(author_name).split()
    if len(parts) < 2:
        raise ValueError('Author name must include given and family names.')
    target_given = ' '.join(parts[:-1])
    target_family = parts[-1]
    return [
        author
        for author in work.get('author') or []
        if _normalize_text(author.get('family')) == target_family
        and _given_names_match(target_given, author.get('given'))
    ]


def _author_orcid(author: _CrossrefRecord) -> str:
    """Read a valid ORCID from an author record.

    Parameters
    ----------
    author : _CrossrefRecord
        Crossref author record.

    Returns
    -------
    str
        Normalized ORCID, or an empty string for malformed values.
    """
    try:
        return normalize_orcid(author.get('ORCID'))
    except ValueError:
        return ''


def work_matches_author(
    work: _CrossrefRecord,
    author_name: str | None = None,
    orcid: str | None = None,
    affiliation: str | None = None,
) -> bool:
    """Check whether a work contains the requested author identity.

    Parameters
    ----------
    work : _CrossrefRecord
        Crossref work record.
    author_name : str, optional
        Human-readable author name to match when no ORCID is supplied.
    orcid : str, optional
        ORCID identifier to match exactly.
    affiliation : str, optional
        Affiliation fragment required on a matching author record.

    Returns
    -------
    bool
        Whether the requested identity is present.

    Raises
    ------
    ValueError
        If ``orcid`` is invalid or ``author_name`` lacks both given and family
        names.
    """
    authors = work.get('author') or []
    if orcid:
        expected = normalize_orcid(orcid)
        matches = [
            author for author in authors
            if _author_orcid(author) == expected
        ]
    else:
        matches = _matching_authors(work, author_name)
    if not matches:
        return False
    if not affiliation:
        return True
    affiliation_key = _normalize_text(affiliation)
    return any(
        affiliation_key in _normalize_text(entry.get('name'))
        for author in matches
        for entry in author.get('affiliation') or []
    )


def min_interval(email: str | None = None) -> float:
    """Return the minimum seconds between Crossref requests for one address.

    Parameters
    ----------
    email : str or None, optional
        Contact address advertised on the requests being paced.

    Returns
    -------
    float
        Delay honoring the polite pool's ten requests per second when an
        address is advertised, or the public pool's five when none is.
    """
    email = str(email or '')
    return CROSSREF_POLITE_MIN_INTERVAL if '@' in email else CROSSREF_PUBLIC_MIN_INTERVAL


def _request_page(
    session: provider.HTTPClient,
    params: dict[str, str | int],
    email: str,
    url: str = CROSSREF_WORKS_URL,
    timeout: float = 60,
    attempts: int = 4,
    pace: float | None = None,
) -> _CrossrefRecord:
    """Request one Crossref page with bounded retries.

    Parameters
    ----------
    session : provider.HTTPClient
        HTTP session used for the request.
    params : dict[str, str or int]
        Crossref query parameters.
    email : str
        Contact email included in the user agent.
    url : str, optional
        Crossref endpoint to request.
    timeout : float, optional
        Request timeout in seconds.
    attempts : int, optional
        Maximum number of request attempts.
    pace : float or None, optional
        Pacing override for this request, in seconds. ``None`` paces by the
        pool the contact address qualifies for.

    Returns
    -------
    _CrossrefRecord
        Crossref response message.

    Raises
    ------
    RuntimeError
        If every request attempt fails or returns an invalid payload.
    """
    response = provider.request(url, label='Crossref', limiter=LIMITER, params=params,
                                headers=provider.default_headers(email), session=session,
                                timeout=timeout, attempts=attempts, missing_ok=False,
                                interval=min_interval(email) if pace is None else pace,
                                error_types=(*provider.RETRY_ERRORS, KeyError))
    if response is None:
        raise RuntimeError(f'Crossref request failed after {attempts} attempts: 404')
    return response.json()['message']


def configured_email(settings: Mapping[str, str] | None = None) -> str:
    """Return the configured Crossref contact email.

    Parameters
    ----------
    settings : Mapping[str, str] or None, optional
        Settings mapping to inspect instead of loading the stored settings.

    Returns
    -------
    str
        Configured contact email, or an empty string when none is stored.
    """
    settings = settings if settings is not None else load_settings()
    return str(settings.get('crossref_email') or '')


def resolve_email(email: str | None = None) -> str:
    """Resolve the Crossref contact email, preferring an explicit value.

    Parameters
    ----------
    email : str or None, optional
        Contact email supplied by the caller.

    Returns
    -------
    str
        Resolved contact email.

    Raises
    ------
    ValueError
        If no valid contact email is available.
    """
    resolved = str(email or '').strip() or configured_email()
    if not resolved or '@' not in resolved:
        raise ValueError(
            'A contact email is required for Crossref requests. '
            'Run pmt config crossref-email, set CROSSREF_EMAIL, or pass --email.'
        )
    return resolved


def work_by_doi(doi: str,
                email: str | None = None,
                session: provider.HTTPClient | None = None,
                timeout: float = 60) -> _CrossrefRecord | None:
    """Look up one Crossref work through the single-work route.

    Parameters
    ----------
    doi : str
        DOI to look up.
    email : str or None, optional
        Contact email for Crossref requests.
    session : provider.HTTPClient or None, optional
        HTTP session used for the request.
    timeout : float, default=60
        Request timeout in seconds.

    Returns
    -------
    _CrossrefRecord or None
        Crossref work record, or ``None`` when Crossref does not know the DOI.

    Raises
    ------
    ValueError
        If no contact email is available.
    RuntimeError
        If the Crossref request exhausts its retries.
    """
    doi = clean_doi(doi)
    if not doi:
        return None
    email = resolve_email(email)
    session = session or requests.Session()
    url = f'{CROSSREF_SINGLE_WORK_URL}/{quote(doi, safe="")}'
    try:
        return _request_page(session, {'mailto': email}, email, url=url)
    except RuntimeError:
        return None


def works_by_doi(dois: Sequence[str],
                 email: str | None = None,
                 session: provider.HTTPClient | None = None,
                 batch_size: int = MAX_FILTER_VALUES,
                 pace: float | None = None) -> dict[str, _CrossrefRecord]:
    """Look up Crossref works in paced DOI batches.

    Crossref treats repeated ``doi`` filters as alternatives, returns matches
    in an arbitrary order, and omits DOIs it does not know, so results are
    keyed by cleaned DOI rather than zipped onto the request order. No
    ``select`` is sent because ``language`` is returned by default but is not a
    selectable field. DOIs containing a comma are requested individually
    because the comma separates Crossref filter terms.

    Parameters
    ----------
    dois : Sequence[str]
        DOIs to look up.
    email : str or None, optional
        Contact email for Crossref requests.
    session : provider.HTTPClient or None, optional
        HTTP session used for the requests.
    batch_size : int, default=100
        DOIs requested per Crossref page.
    pace : float or None, optional
        Seconds to wait between consecutive Crossref requests. ``None`` paces
        by the pool the contact address qualifies for.

    Returns
    -------
    dict[str, _CrossrefRecord]
        Crossref work records keyed by cleaned DOI.

    Raises
    ------
    ValueError
        If ``batch_size`` is invalid or no contact email is available.
    RuntimeError
        If a Crossref request exhausts its retries.
    """
    if batch_size < 1 or batch_size > MAX_FILTER_VALUES:
        raise ValueError(f'batch_size must be between 1 and {MAX_FILTER_VALUES}.')
    email = resolve_email(email)
    session = session or requests.Session()
    wanted = list(dict.fromkeys(clean_doi(doi) for doi in dois if clean_doi(doi)))
    batched = [doi for doi in wanted if BATCHABLE_DOI_PATTERN.match(doi)]
    individual = [doi for doi in wanted if not BATCHABLE_DOI_PATTERN.match(doi)]

    works = {}
    for start in range(0, len(batched), batch_size):
        chunk = batched[start:start + batch_size]
        try:
            message = _request_page(
                session,
                {
                    'filter': ','.join(f'doi:{doi}' for doi in chunk),
                    'rows': len(chunk),
                    'mailto': email,
                },
                email,
                pace=pace,
            )
        except RuntimeError:
            # Crossref rejects an entire filter when one value is unusable, so
            # fall back to single-work lookups rather than losing the chunk.
            individual.extend(chunk)
            continue
        for work in message.get('items') or []:
            key = clean_doi(work.get('DOI'))
            if key:
                works[key] = work
    for doi in individual:
        work = work_by_doi(doi, email=email, session=session)
        if work:
            works[clean_doi(work.get('DOI')) or doi] = work
    return works


def author_works(orcid: str | None = None,
                 author_name: str | None = None,
                 affiliation: str | None = None,
                 email: str | None = None,
                 max_results: int | None = 500,
                 page_size: int = 200,
                 session: provider.HTTPClient | None = None) -> list[_CrossrefRecord]:
    """Retrieve DOI-bearing works for one author.

    Parameters
    ----------
    orcid : str, optional
        ORCID identifier used for exact server-side filtering.
    author_name : str, optional
        Given and family names used for search and local filtering.
    affiliation : str, optional
        Affiliation fragment required on the matched author record.
    email : str, optional
        Contact email for Crossref polite-pool requests.
    max_results : int or None, optional
        Maximum accepted works, or ``None`` for no explicit limit.
    page_size : int, optional
        Number of records requested per Crossref page.
    session : provider.HTTPClient or None, optional
        HTTP session, primarily for connection reuse or testing.

    Returns
    -------
    list[_CrossrefRecord]
        Unique matching Crossref work records.

    Raises
    ------
    ValueError
        If identity, contact, or pagination options are invalid.
    RuntimeError
        If a Crossref request exhausts its retries.
    """
    if bool(orcid) == bool(author_name):
        raise ValueError('Provide exactly one of orcid or author_name.')
    if not email or '@' not in email:
        raise ValueError('A contact email is required for Crossref polite-pool requests. '
                         'Run pmt config crossref-email, set CROSSREF_EMAIL, or pass --email.')
    if max_results is not None and max_results < 1:
        raise ValueError('max_results must be a positive integer or None.')
    if page_size < 1 or page_size > 1000:
        raise ValueError('page_size must be between 1 and 1000.')

    session = session or requests.Session()
    params = {'rows': page_size, 'cursor': '*', 'mailto': email}
    if orcid:
        orcid = normalize_orcid(orcid)
        params['filter'] = f'orcid:{orcid}'
    else:
        params['query.author'] = author_name

    works = []
    seen_dois = set()
    while True:
        message = _request_page(session, params, email)
        items = message.get('items') or []
        for work in items:
            doi = str(work.get('DOI') or '').strip().lower()
            if not doi or doi in seen_dois:
                continue
            if not work_matches_author(work, author_name=author_name, orcid=orcid, affiliation=affiliation):
                continue
            seen_dois.add(doi)
            works.append(work)
            if max_results is not None and len(works) >= max_results:
                return works
        if len(items) < page_size:
            break
        next_cursor = message.get('next-cursor')
        if not next_cursor or next_cursor == params['cursor']:
            break
        params['cursor'] = next_cursor
    return works


def _first(value: object) -> str:
    """Select the first non-empty Crossref field value.

    Parameters
    ----------
    value : object
        Scalar or list Crossref field value.

    Returns
    -------
    str
        First non-empty string, or an empty string.
    """
    if isinstance(value, list):
        return next((str(item).strip() for item in value if str(item).strip()), '')
    return str(value or '').strip()


def _publication_date(work: _CrossrefRecord) -> str:
    """Extract the best available Crossref publication date.

    Parameters
    ----------
    work : _CrossrefRecord
        Crossref work record.

    Returns
    -------
    str
        ISO-like publication date, or an empty string.
    """
    for key in ['published-print', 'published-online', 'published', 'issued', 'created']:
        value = work.get(key) or {}
        parts = value.get('date-parts') if isinstance(value, dict) else None
        if parts and parts[0]:
            return '-'.join(f'{int(part):02d}' if index else str(part)
                            for index, part in enumerate(parts[0][:3]))
    return ''


def crossref_work_to_paper(work: _CrossrefRecord) -> dict[str, Any]:
    """Map a Crossref work to the corpus schema.

    Parameters
    ----------
    work : _CrossrefRecord
        Crossref work record.

    Returns
    -------
    dict[str, Any]
        Normalized paper metadata suitable for corpus insertion.
    """
    doi = str(work.get('DOI') or '').strip().lower()
    authors = '; '.join(
        ' '.join(part for part in [str(author.get('given') or '').strip(),
                                   str(author.get('family') or '').strip()] if part)
        for author in work.get('author') or []
    )
    return {
        'paper_id': f'doi:{doi}',
        'doi': doi,
        'title': _first(work.get('title')),
        'journal': _first(work.get('container-title')),
        'publication_date': _publication_date(work),
        'authors': authors,
        'sources': 'crossref',
        'metadata_status': 'retrieved',
        'metadata': work,
    }


def import_author_works(db_path: str | PathLike[str],
                        email: str | None = None,
                        orcid: str | None = None,
                        author_name: str | None = None,
                        affiliation: str | None = None,
                        max_results: int | None = 500,
                        review_csv: str | PathLike[str] | None = None,
                        session: provider.HTTPClient | None = None,
                        enrich: bool = False) -> dict[str, int]:
    """Discover and import an author's Crossref works.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Destination corpus database.
    email : str or None, optional
        Contact email for Crossref requests. Defaults to the stored
        ``crossref_email`` setting.
    orcid : str, optional
        ORCID identifier for exact author discovery.
    author_name : str, optional
        Author name used when no ORCID is supplied.
    affiliation : str, optional
        Affiliation fragment required on the matched author record.
    max_results : int or None, optional
        Maximum number of works to import.
    review_csv : str, os.PathLike[str], or None, optional
        CSV path for a human-readable import review.
    session : provider.HTTPClient or None, optional
        HTTP session used for discovery.
    enrich : bool, default=False
        Whether to supplement imported works with Crossref and OpenAlex
        metadata.

    Returns
    -------
    dict[str, int]
        Counts of found, added, updated, and enriched papers.

    Raises
    ------
    ValueError
        If the author identity, contact email, result limit, or ORCID is
        invalid.
    RuntimeError
        If a Crossref request exhausts its retries or the corpus schema is
        newer than this package supports.
    """
    email = resolve_email(email)
    works = author_works(
        orcid=orcid,
        author_name=author_name,
        affiliation=affiliation,
        email=email,
        max_results=max_results,
        session=session,
    )
    papers = [crossref_work_to_paper(work) for work in works]
    if review_csv:
        review_path = Path(review_csv)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(papers, columns=REVIEW_COLUMNS).to_csv(review_path, index=False)
    with connect(db_path) as conn:
        added, updated = upsert_papers(conn, papers)
        for paper in papers:
            matched = find_paper(conn, paper)
            if matched is None:
                continue
            matched['metadata'] = paper['metadata']
            upsert_paper(conn, matched)
        enriched = 0
        if enrich:
            from paperminertoolkit.workflows.enrichment import enrich_papers

            enriched = enrich_papers(conn, papers, email=email)['succeeded']
    return {'found': len(papers), 'added': added, 'updated': updated, 'enriched': enriched}
