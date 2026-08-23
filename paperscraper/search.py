"""Search Elsevier/Scopus, CORE, OpenAlex, PubMed, arXiv, medRxiv, bioRxiv, and chemRxiv, then merge results into the paper corpus.

The functions here translate provider-specific API responses into PaperScraper's
small public paper schema and append or update rows without duplicating papers
that appear in multiple sources.
"""

from __future__ import annotations

import datetime
import html
import os
import sqlite3
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any
import pandas as pd
import re
import requests
from tqdm import tqdm

from paperscraper import arxiv, biorxiv, chemrxiv, elsevier, medrxiv, openalex, pubmed
from paperscraper.enrichment import enrich_papers
from paperscraper.corpus import PAPER_FIELDS, add_asset, connect, find_paper, normalize_paper, upsert_paper, upsert_papers
from paperscraper.settings import load_settings

SEARCH_SOURCES = {'elsevier', 'core', 'openalex', 'pubmed', 'arxiv', 'medrxiv', 'biorxiv',
                  'chemrxiv', 'all'}
SEARCH_FIELDS = PAPER_FIELDS + ['abstract']


def _elsevier_api_key() -> str:
    """Return the configured Elsevier API key."""
    api_key = load_settings().get('elsevier_api_key')
    if not api_key:
        raise ValueError('Elsevier API key is not configured. Run ps_elsevier_key first.')
    return api_key


def _recast_elsevier_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten common one-item Elsevier list fields into scalar DataFrame values."""
    rows = []
    for record in records:
        row = {}
        for key, value in record.items():
            row[key] = value if key == 'link' else _first(value)
        rows.append(row)
    return pd.DataFrame(rows)


def document_search(query: str,
                    index: str = 'scopus',
                    count: int = 200,
                    get_all: bool = True,
                    search_fields: str = 'TITLE-ABS-KEY') -> pd.DataFrame:
    """Search Elsevier and return raw provider records.

    This is a small replacement for the Elsapy search helper so PaperScraper can
    control pagination and treat ``count`` as a hard result cap.

    Parameters
    ----------
    query : str
        Search expression.
    index : str, default='scopus'
        Elsevier index to search.
    count : int, default=200
        Maximum number of records to return.
    get_all : bool, default=True
        Whether to follow provider pagination links.
    search_fields : str, default='TITLE-ABS-KEY'
        Elsevier fields in which to evaluate ``query``.

    Returns
    -------
    pandas.DataFrame
        Raw Elsevier records capped at ``count`` rows.

    Raises
    ------
    ValueError
        If no Elsevier API key is configured.
    requests.RequestException
        If an Elsevier request fails.
    """
    api_key = _elsevier_api_key()
    max_results = max(int(count), 1)
    page_size = min(max_results, 200)
    index = index.lower()
    url = elsevier.search_url(index, query, page_size, search_fields)
    api_response = elsevier.get_json(api_key, url)
    tot_num_res = int(api_response['search-results']['opensearch:totalResults'])
    target_results = min(tot_num_res, max_results)
    print('Document search is retrieving', target_results, 'of', tot_num_res, 'results.')
    if tot_num_res == 0:
        return pd.DataFrame()
    results = api_response['search-results'].get('entry', [])
    if get_all:
        with tqdm(total=target_results, desc='Searching Scopus', colour='blue') as pbar:
            pbar.update(min(len(results), target_results))
            upper_limit_reached = False
            while (len(results) < target_results) and not upper_limit_reached:
                next_url = None
                for e in api_response['search-results']['link']:
                    if e['@ref'] == 'next':
                        next_url = e['@href']
                if not next_url:
                    break
                api_response = elsevier.get_json(api_key, next_url)
                next_results = api_response['search-results'].get('entry', [])
                remaining = target_results - len(results)
                results += next_results[:remaining]
                if len(results) >= 5000 and index != 'scopus':
                    upper_limit_reached = True
                pbar.update(min(len(next_results), remaining))
    else:
        results = results[:target_results]
    return _recast_elsevier_records(results)


def _first(value: object) -> object:
    """Return the first item from a list-like provider value, or the value itself."""
    if isinstance(value, list):
        return value[0] if value else ''
    return value or ''


def _elsevier_link(value: object) -> object:
    """Return the full-text Elsevier link when present, otherwise the first link value."""
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                ref = str(item.get('@ref') or item.get('ref') or '').lower()
                href = item.get('@href') or item.get('href') or ''
                if href and ('full-text' in ref or 'full-text' in str(href).lower()):
                    return href
        return _elsevier_link(_first(value))
    if isinstance(value, dict):
        return value.get('@href') or value.get('href') or value.get('url') or ''
    return value or ''


def _clean_search_abstract(value: object) -> str:
    """Normalize a search-result abstract to compact plain text."""
    if value is None:
        return ''
    if isinstance(value, list):
        value = ' '.join(str(part) for part in value if part)
    text = html.unescape(str(value))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _abstract_from_search_record(record: Mapping[str, Any]) -> str:
    """Extract an abstract-like field from one provider search record."""
    for key in ['abstract', 'dc:description', 'description', 'dcDescription']:
        abstract = _clean_search_abstract(record.get(key))
        if abstract:
            return abstract
    return ''


def _elsevier_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Elsevier search records into normalized paper rows."""
    rows = []
    for _, paper in results.iterrows():
        row = {
            'paper_id': paper.get('dc:identifier') or paper.get('eid') or '',
            'doi': paper.get('prism:doi') or '',
            'title': paper.get('dc:title') or '',
            'journal': paper.get('prism:publicationName') or '',
            'publication_date': paper.get('prism:coverDate') or '',
            'authors': paper.get('dc:creator') or paper.get('creator') or '',
            'sources': 'elsevier',
            'elsevier_link': _elsevier_link(paper.get('link')),
            'metadata_status': 'retrieved',
        }
        normalized = normalize_paper(row)
        normalized['abstract'] = _abstract_from_search_record(paper)
        rows.append(normalized)
    return pd.DataFrame(rows, columns=SEARCH_FIELDS)


def _core_api_key() -> str | None:
    """Return the configured CORE API key, if one is available."""
    settings = load_settings()
    return settings.get('core_api_key') or os.environ.get('CORE_API_KEY')


def _core_headers() -> dict[str, str]:
    """Build request headers for CORE API calls."""
    api_key = _core_api_key()
    headers = {'User-Agent': 'PaperScraper/0.0.1'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


def _core_download_url(work: Mapping[str, Any]) -> str:
    """Return the best CORE PDF download URL for a work record."""
    download_url = work.get('downloadUrl') or work.get('download_url')
    if download_url:
        return download_url
    core_id = work.get('id')
    if core_id:
        return f'https://api.core.ac.uk/v3/works/{core_id}/download'
    return ''


def _core_authors(work: Mapping[str, Any]) -> str:
    """Format CORE author data as a semicolon-separated author string."""
    authors = work.get('authors') or []
    names = []
    for author in authors:
        if isinstance(author, dict):
            names.append(author.get('name') or author.get('fullName') or '')
        else:
            names.append(str(author))
    return '; '.join(name for name in names if name)


def _core_journal(work: Mapping[str, Any]) -> object:
    """Extract a journal or publisher name from a CORE work record."""
    journal = work.get('journal') or work.get('publisher') or ''
    if isinstance(journal, dict):
        return journal.get('title') or journal.get('name') or ''
    return journal


def _core_date(work: Mapping[str, Any]) -> object:
    """Extract the best available publication date/year from a CORE work record."""
    return work.get('publishedDate') or work.get('published_date') or work.get('yearPublished') or work.get(
        'year') or ''


def _core_rows(works: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert CORE work records into normalized paper rows."""
    rows = []
    for work in works:
        core_id = work.get('id') or ''
        doi = _first(work.get('doi') or work.get('DOI'))
        identifier = f'core:{core_id}' if core_id else (f'doi:{doi}' if doi else '')
        row = {
            'paper_id': identifier,
            'doi': doi,
            'title': _first(work.get('title')),
            'journal': _core_journal(work),
            'publication_date': _core_date(work),
            'authors': _core_authors(work),
            'sources': 'core',
            'core_id': core_id,
            'pdf_url': _core_download_url(work),
            'metadata_status': 'retrieved',
        }
        normalized = normalize_paper(row)
        normalized['abstract'] = _abstract_from_search_record(work)
        rows.append(normalized)
    return pd.DataFrame(rows, columns=SEARCH_FIELDS)


def core_search(query: str, count: int = 200) -> pd.DataFrame:
    """Search CORE works.

    Parameters
    ----------
    query : str
        Search expression.
    count : int, default=200
        Maximum number of records to return.

    Returns
    -------
    pandas.DataFrame
        Normalized paper rows capped at ``count`` records.

    Raises
    ------
    requests.RequestException
        If a CORE request fails.
    """
    url = 'https://api.core.ac.uk/v3/search/works'
    limit = min(max(int(count), 1), 100)
    offset = 0
    works = []
    with tqdm(total=count, desc='Searching CORE', colour='cyan') as pbar:
        while len(works) < count:
            params = {'q': query, 'limit': min(limit, count - len(works)), 'offset': offset}
            response = requests.get(url, headers=_core_headers(), params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            results = payload.get('results') or payload.get('data') or []
            if not results:
                break
            works.extend(results)
            offset += len(results)
            pbar.update(len(results))
            total_hits = payload.get('totalHits') or payload.get('total') or payload.get('count')
            if total_hits is not None and offset >= int(total_hits):
                break
            if len(results) < params['limit']:
                break
    return _core_rows(works)


def _openalex_rows(works: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert OpenAlex work records into normalized paper rows."""
    rows = []
    for work in works:
        normalized = normalize_paper(openalex.work_to_paper(work))
        abstract = openalex.reconstruct_abstract(work.get('abstract_inverted_index'))
        normalized['abstract'] = _clean_search_abstract(abstract)
        rows.append(normalized)
    return pd.DataFrame(rows, columns=SEARCH_FIELDS)


def openalex_search(query: str, count: int = 200) -> pd.DataFrame:
    """Search OpenAlex works.

    Parameters
    ----------
    query : str
        Search expression.
    count : int, default=200
        Maximum number of records to return.

    Returns
    -------
    pandas.DataFrame
        Normalized paper rows capped at ``count`` records.

    Raises
    ------
    RuntimeError
        If an OpenAlex request cannot be completed.
    """
    api_key = openalex.configured_api_key()
    per_page = min(max(int(count), 1), 200)
    params = {'search': query, 'per-page': per_page, 'cursor': '*'}
    works = []
    with tqdm(total=count, desc='Searching OpenAlex', colour='green') as pbar:
        while len(works) < count:
            params['per-page'] = min(per_page, count - len(works))
            payload = openalex.request_json(openalex.WORKS_URL, params=params, api_key=api_key) or {}
            results = payload.get('results') or []
            if not results:
                break
            works.extend(results)
            pbar.update(len(results))
            next_cursor = (payload.get('meta') or {}).get('next_cursor')
            if not next_cursor:
                break
            params['cursor'] = next_cursor
    return _openalex_rows(works)


def _pubmed_rows(articles: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert PubMed article records into normalized paper rows."""
    rows = []
    for article in articles:
        normalized = normalize_paper(article)
        normalized['abstract'] = _clean_search_abstract(article.get('abstract'))
        rows.append(normalized)
    return pd.DataFrame(rows, columns=SEARCH_FIELDS)


def pubmed_search(query: str, count: int = 200) -> pd.DataFrame:
    """Search PubMed records.

    PubMed exposes only the first 10000 matches for any query, so a larger
    corpus needs the query split by date range. The shortfall is printed rather
    than passed over silently.

    Parameters
    ----------
    query : str
        Search expression.
    count : int, default=200
        Maximum number of records to return.

    Returns
    -------
    pandas.DataFrame
        Normalized paper rows capped at ``count`` records.

    Raises
    ------
    RuntimeError
        If a PubMed request cannot be completed.
    """
    api_key = pubmed.configured_api_key()
    email = pubmed.configured_email()
    webenv, query_key, total = pubmed.esearch_history(query, api_key=api_key, email=email)
    reachable = min(total, pubmed.MAX_SEARCH_RESULTS)
    target = min(max(int(count), 0), reachable)
    if total > pubmed.MAX_SEARCH_RESULTS:
        print(f'PubMed matched {total} records but exposes only the first '
              f'{pubmed.MAX_SEARCH_RESULTS}; narrow the query by date to reach the rest.')
    if not target or not webenv or not query_key:
        return _pubmed_rows([])
    articles = []
    with tqdm(total=target, desc='Searching PubMed', colour='magenta') as pbar:
        while len(articles) < target:
            page = pubmed.parse_articles(pubmed.efetch_history(
                webenv,
                query_key,
                retstart=len(articles),
                retmax=min(pubmed.EFETCH_BATCH_SIZE, target - len(articles)),
                api_key=api_key,
                email=email,
            ))
            if not page:
                break
            articles.extend(page)
            pbar.update(len(page))
    return _pubmed_rows(articles[:target])


def _arxiv_rows(entries: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert arXiv entry records into normalized paper rows."""
    rows = []
    for entry in entries:
        normalized = normalize_paper(entry)
        normalized['abstract'] = _clean_search_abstract(entry.get('abstract'))
        rows.append(normalized)
    return pd.DataFrame(rows, columns=SEARCH_FIELDS)


def arxiv_search(query: str, count: int = 200) -> pd.DataFrame:
    """Search arXiv records.

    A plain phrase is translated into arXiv's fielded query language by
    :func:`paperscraper.arxiv.query_expression`; a native arXiv expression is
    used as written. Results are ordered by submission date rather than by
    relevance so that paging is stable, because relevance ordering can shift
    between the requests that make up one page walk. Entries are deduplicated
    by identifier for the same reason: arXiv repeats records across page
    boundaries often enough to inflate a naive count.

    arXiv exposes only the first 30000 matches for any query, so a larger
    corpus needs the query split by category or date. The shortfall is printed
    rather than passed over silently.

    Parameters
    ----------
    query : str
        Search expression, either a plain phrase or a native arXiv query.
    count : int, default=200
        Maximum number of records to return.

    Returns
    -------
    pandas.DataFrame
        Normalized paper rows capped at ``count`` records.

    Raises
    ------
    RuntimeError
        If an arXiv request cannot be completed.
    """
    expression = arxiv.query_expression(query)
    target = max(int(count), 0)
    if not expression or not target:
        return _arxiv_rows([])
    entries: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    reachable = 0
    reported = False
    with tqdm(total=target, desc='Searching arXiv', colour='yellow') as pbar:
        while len(entries) < target:
            root = arxiv.search_page(expression, start=offset,
                                     max_results=min(arxiv.PAGE_SIZE, target - len(entries)))
            page = arxiv.parse_entries(root)
            if not reported:
                total = arxiv.total_results(root)
                if total > arxiv.MAX_SEARCH_RESULTS:
                    print(f'arXiv matched {total} records but exposes only the first '
                          f'{arxiv.MAX_SEARCH_RESULTS}; narrow the query by category or date '
                          f'to reach the rest.')
                reachable = min(total, arxiv.MAX_SEARCH_RESULTS)
                target = min(target, reachable)
                pbar.total = target
                reported = True
            if not page:
                break
            # Advance by what arXiv returned, not by what survived deduplication,
            # so a page of repeats still moves the cursor forward.
            offset += len(page)
            for entry in page:
                identifier = str(entry.get('arxiv_id') or entry.get('paper_id') or '')
                if identifier and identifier in seen:
                    continue
                if identifier:
                    seen.add(identifier)
                entries.append(entry)
                pbar.update(1)
                if len(entries) >= target:
                    break
            # Stop once the walk has passed everything arXiv reports, so a run
            # of duplicate pages cannot keep the loop going indefinitely.
            if offset >= reachable:
                break
    return _arxiv_rows(entries[:target])


def _rxiv_rows(entries: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert preprint-server records into normalized paper rows."""
    rows = []
    for entry in entries:
        normalized = normalize_paper(entry)
        normalized['abstract'] = _clean_search_abstract(entry.get('abstract'))
        rows.append(normalized)
    return pd.DataFrame(rows, columns=SEARCH_FIELDS)


def _medrxiv_rows(entries: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert medRxiv records into normalized paper rows."""
    return _rxiv_rows(entries)


def _biorxiv_rows(entries: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert bioRxiv records into normalized paper rows."""
    return _rxiv_rows(entries)


def _chemrxiv_rows(entries: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert chemRxiv records into normalized paper rows."""
    return _rxiv_rows(entries)


def _rxiv_search(provider: ModuleType,
                 label: str,
                 query: str,
                 count: int = 200,
                 today: str = '') -> pd.DataFrame:
    """Answer a query by walking a preprint server's posting archive.

    medRxiv and bioRxiv are one service answering under two names, with the
    same endpoints, paging rules, record shape, and absence of a search route,
    so one walk serves both and is parameterized by the provider module rather
    than written twice. Everything server-specific -- the hosts, the page
    widths, the archive start, and the scan limit -- is read off ``provider``.

    Parameters
    ----------
    provider : types.ModuleType
        :mod:`paperscraper.medrxiv` or :mod:`paperscraper.biorxiv`.
    label : str
        Server name as it should appear in progress and summary messages.
    query : str
        Search phrase, optionally carrying ``category:``, ``from:``, or ``to:``
        scope terms.
    count : int, default=200
        Maximum number of records to return.
    today : str, default=''
        Interval end used when the query names none, as ``YYYY-MM-DD``.
        Defaults to the current UTC date.

    Returns
    -------
    pandas.DataFrame
        Normalized paper rows capped at ``count`` records.

    Raises
    ------
    ValueError
        If a scope term in ``query`` is not an ISO ``YYYY-MM-DD`` date.
    RuntimeError
        If a request to the server cannot be completed.
    """
    terms, scope = provider.parse_query(query)
    target = max(int(count), 0)
    if not target:
        return _rxiv_rows([])
    category = scope.get('category', '')
    start = scope.get('from', provider.CORPUS_START)
    end = scope.get('to') or today or datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')
    first = provider.interval_page(start, end, category=category)
    total = provider.total_results(first)
    step = provider.page_size(first, provider.endpoint(category)[1])
    if not total:
        return _rxiv_rows([])
    print(f'{label} has no search endpoint, so PaperScraper is reading the {total} postings '
          f'between {start} and {end}'
          f'{f" filed under {category}" if category else ""}, newest first, and matching them '
          f'locally. Add category:, from:, or to: terms to the query to read fewer.')
    matched: list[Mapping[str, Any]] = []
    papers: list[Mapping[str, Any]] = []
    scanned = 0
    unread = 0
    with tqdm(total=target, desc=f'Searching {label}', colour='yellow') as pbar:
        for cursor in provider.page_cursors(total, step):
            # The first page was already fetched to learn the record count, so
            # it is reused rather than requested again when the walk reaches it.
            payload = first if cursor == 0 else provider.interval_page(
                start, end, cursor=cursor, category=category)
            page = provider.parse_records(payload)
            if not page:
                continue
            scanned += len(page)
            # Pages run oldest first with no way to reverse them at the API, so
            # each one is reversed as it arrives to keep the walk newest first.
            matched.extend(entry for entry in reversed(page) if provider.matches(entry, terms))
            # Versions of one preprint are separate records that can fall on
            # different pages, so papers are collapsed against everything
            # matched so far rather than within a page.
            papers = provider.latest_versions(matched)
            pbar.n = min(len(papers), target)
            pbar.refresh()
            if len(papers) >= target:
                break
            if scanned >= provider.MAX_SCAN_RECORDS:
                # Cursors count records, so the cursor of the page that tripped
                # the limit is exactly how many older postings go unread.
                unread = cursor
                break
    print(f'{label} matched {len(papers)} papers in {scanned} postings read.')
    if unread:
        print(f'{unread} older postings in {start} to {end} were left unread after the '
              f'{provider.MAX_SCAN_RECORDS}-posting scan limit; narrow the query with '
              f'category:, from:, or to: terms to reach them.')
    return _rxiv_rows(papers[:target])


def medrxiv_search(query: str, count: int = 200, today: str = '') -> pd.DataFrame:
    """Search medRxiv records.

    medRxiv publishes no search endpoint, so the query is answered by walking
    the posting archive and matching each record locally. The walk runs newest
    first and stops as soon as ``count`` papers match, which keeps an ordinary
    query cheap; a term that matches nothing recent is what makes it expensive,
    so the size of the archive being scanned is printed before the walk starts
    rather than discovered as a hang.

    The query string carries its own scope, because narrowing the walk is the
    only way to bound it: ``category:``, ``from:``, and ``to:`` restrict the
    archive that is read, and everything else is a match term. Terms are
    combined with ``AND`` over each record's title, abstract, authors, and
    category.

    One search reads at most :data:`paperscraper.medrxiv.MAX_SCAN_RECORDS`
    postings. That bound is what keeps ``--source all`` usable: a query aimed
    at another provider's subject matter matches nothing here, and without a
    stop it would read the entire archive before reporting nothing found. The
    shortfall is printed rather than passed over silently.

    Parameters
    ----------
    query : str
        Search phrase, optionally carrying ``category:``, ``from:``, or ``to:``
        scope terms.
    count : int, default=200
        Maximum number of records to return.
    today : str, default=''
        Interval end used when the query names none, as ``YYYY-MM-DD``.
        Defaults to the current UTC date.

    Returns
    -------
    pandas.DataFrame
        Normalized paper rows capped at ``count`` records.

    Raises
    ------
    ValueError
        If a scope term in ``query`` is not an ISO ``YYYY-MM-DD`` date.
    RuntimeError
        If a medRxiv request cannot be completed.
    """
    return _rxiv_search(medrxiv, 'medRxiv', query, count=count, today=today)


def biorxiv_search(query: str, count: int = 200, today: str = '') -> pd.DataFrame:
    """Search bioRxiv records.

    bioRxiv answers a query exactly as medRxiv does, by the same archive walk
    over the same endpoints, so :func:`medrxiv_search` describes the mechanism
    and the ``category:``, ``from:``, and ``to:`` scope terms in full.

    What differs is scale. bioRxiv has been accepting preprints since November
    2013 and holds several times what medRxiv does, so an unscoped walk is
    correspondingly longer and the
    :data:`paperscraper.biorxiv.MAX_SCAN_RECORDS` limit is correspondingly
    likelier to be what ends it. Naming a category or a date range is the
    difference between a search of a subject and a read of the archive.

    Parameters
    ----------
    query : str
        Search phrase, optionally carrying ``category:``, ``from:``, or ``to:``
        scope terms.
    count : int, default=200
        Maximum number of records to return.
    today : str, default=''
        Interval end used when the query names none, as ``YYYY-MM-DD``.
        Defaults to the current UTC date.

    Returns
    -------
    pandas.DataFrame
        Normalized paper rows capped at ``count`` records.

    Raises
    ------
    ValueError
        If a scope term in ``query`` is not an ISO ``YYYY-MM-DD`` date.
    RuntimeError
        If a bioRxiv request cannot be completed.
    """
    return _rxiv_search(biorxiv, 'bioRxiv', query, count=count, today=today)


def chemrxiv_search(query: str, count: int = 200) -> pd.DataFrame:
    """Search chemRxiv records.

    chemRxiv, unlike medRxiv and bioRxiv, publishes a search endpoint, so the
    query is answered by the server rather than by reading the archive and
    matching locally. Paging therefore works the way it does for arXiv: pages
    are requested until ``count`` papers are held, and the cursor advances by
    what the server returned rather than by what survived deduplication, so a
    page of repeats still moves it forward.

    The ``category:``, ``from:``, and ``to:`` scope terms are accepted with the
    same spelling the other preprint sources use, but here they are forwarded
    as ``categoryIds``, ``searchDateFrom``, and ``searchDateTo`` rather than
    applied to records after reading them. Narrowing a chemRxiv query makes the
    server do less work; it is not, as it is for the other two archives, the
    only way to bound what gets read.

    Each posted version of a chemRxiv preprint carries a DOI of its own, so
    records are grouped by :func:`paperscraper.chemrxiv.chemrxiv_stem` and
    reduced to the newest posting. That is done over everything read so far
    rather than per page, because two versions of one preprint can fall on
    either side of a page boundary.

    chemRxiv exposes only the first
    :data:`paperscraper.chemrxiv.MAX_SEARCH_RESULTS` matches for any query, so
    a larger corpus needs the query split by category or date. The shortfall is
    printed rather than passed over silently.

    Parameters
    ----------
    query : str
        Search phrase, optionally carrying ``category:``, ``from:``, or ``to:``
        scope terms.
    count : int, default=200
        Maximum number of records to return.

    Returns
    -------
    pandas.DataFrame
        Normalized paper rows capped at ``count`` records.

    Raises
    ------
    ValueError
        If a scope term in ``query`` is not an ISO ``YYYY-MM-DD`` date, or
        names a category chemRxiv does not file preprints under.
    RuntimeError
        If a chemRxiv request cannot be completed.
    """
    terms, scope = chemrxiv.parse_query(query)
    target = max(int(count), 0)
    if not target:
        return _chemrxiv_rows([])
    term = chemrxiv.search_terms(terms)
    category = scope.get('category', '')
    category_id = chemrxiv.category_ids([category])[0] if category else ''
    entries: list[Mapping[str, Any]] = []
    papers: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    skip = 0
    reachable = 0
    reported = False
    with tqdm(total=target, desc='Searching chemRxiv', colour='yellow') as pbar:
        while len(papers) < target:
            payload = chemrxiv.search_page(term=term, skip=skip,
                                           limit=min(chemrxiv.PAGE_SIZE, target - len(papers)),
                                           category_id=category_id,
                                           date_from=scope.get('from', ''),
                                           date_to=scope.get('to', ''))
            page = chemrxiv.parse_records(payload)
            if not reported:
                total = chemrxiv.total_results(payload)
                if total > chemrxiv.MAX_SEARCH_RESULTS:
                    print(f'chemRxiv matched {total} records but exposes only the first '
                          f'{chemrxiv.MAX_SEARCH_RESULTS}; narrow the query with category:, '
                          f'from:, or to: terms to reach the rest.')
                reachable = min(total, chemrxiv.MAX_SEARCH_RESULTS)
                target = min(target, reachable)
                pbar.total = target
                reported = True
            if not page:
                break
            # Advance by what chemRxiv returned, not by what survived
            # deduplication, so a page of repeats still moves the cursor.
            skip += len(page)
            for entry in page:
                identifier = str(entry.get('paper_id') or entry.get('chemrxiv_doi') or '')
                if identifier and identifier in seen:
                    continue
                if identifier:
                    seen.add(identifier)
                entries.append(entry)
            papers = chemrxiv.latest_versions(entries)
            pbar.n = min(len(papers), target)
            pbar.refresh()
            # Stop once the walk has passed everything chemRxiv reports, so a
            # run of duplicate pages cannot keep the loop going indefinitely.
            if skip >= reachable:
                break
    return _chemrxiv_rows(papers[:target])


def _store_search_abstracts(
    conn: sqlite3.Connection,
    papers: Iterable[dict[str, Any]],
) -> int:
    """Store search-result abstracts as corpus assets and return the stored count."""
    stored = 0
    for paper in papers:
        abstract = _clean_search_abstract(paper.get('abstract'))
        if not abstract:
            continue
        matched = find_paper(conn, paper) or paper
        source = paper.get('sources') or 'search'
        add_asset(conn,
                  matched,
                  abstract,
                  role='abstract',
                  kind='text',
                  mime_type='text/plain',
                  source=source,
                  original_filename='abstract.txt')
        matched['abstract_source'] = source
        matched['abstract_download_status'] = 'succeeded'
        upsert_paper(conn, matched)
        stored += 1
    return stored


def search_for_papers(query: str,
                      db_path: str = 'papers.db',
                      source: str = 'all',
                      count: int = 200,
                      store_abstract: bool = False,
                      enrich: bool = False) -> None:
    """Search providers and merge results into a corpus.

    Parameters
    ----------
    query : str
        Search expression.
    db_path : str, default='papers.db'
        Path to the SQLite paper corpus.
    source : {'all', 'core', 'elsevier', 'openalex', 'pubmed', 'arxiv', 'medrxiv', 'biorxiv', 'chemrxiv'}, default='all'
        Provider or provider set to search.
    count : int, default=200
        Maximum number of records requested from each provider.
    store_abstract : bool, default=False
        Whether to store search-result abstracts as corpus assets.
    enrich : bool, default=False
        Whether to supplement stored rows with metadata from the configured
        enrichment providers.

    Returns
    -------
    None
        Results are written directly to ``db_path``.

    Raises
    ------
    ValueError
        If ``source`` is unsupported or required provider configuration is
        missing.
    requests.RequestException
        If the explicitly selected CORE provider fails.
    RuntimeError
        If the explicitly selected OpenAlex, PubMed, arXiv, medRxiv, bioRxiv, or
        chemRxiv provider fails.
    """
    source = source.lower()
    if source not in SEARCH_SOURCES:
        raise ValueError(f'source must be one of: {", ".join(sorted(SEARCH_SOURCES))}')
    frames = []
    if source in {'elsevier', 'all'}:
        try:
            frames.append(_elsevier_rows(document_search(query, count=count)))
        except Exception as e:
            if source == 'elsevier':
                raise
            print(f'Elsevier search skipped: {e}')
    if source in {'core', 'all'}:
        try:
            frames.append(core_search(query, count=count))
        except requests.RequestException as e:
            if source == 'core':
                raise
            print(f'CORE search skipped: {e}')
    if source in {'openalex', 'all'}:
        try:
            frames.append(openalex_search(query, count=count))
        except Exception as e:
            if source == 'openalex':
                raise
            print(f'OpenAlex search skipped: {e}')
    if source in {'pubmed', 'all'}:
        try:
            frames.append(pubmed_search(query, count=count))
        except Exception as e:
            if source == 'pubmed':
                raise
            print(f'PubMed search skipped: {e}')
    if source in {'arxiv', 'all'}:
        try:
            frames.append(arxiv_search(query, count=count))
        except Exception as e:
            if source == 'arxiv':
                raise
            print(f'arXiv search skipped: {e}')
    if source in {'medrxiv', 'all'}:
        try:
            frames.append(medrxiv_search(query, count=count))
        except Exception as e:
            if source == 'medrxiv':
                raise
            print(f'medRxiv search skipped: {e}')
    if source in {'biorxiv', 'all'}:
        try:
            frames.append(biorxiv_search(query, count=count))
        except Exception as e:
            if source == 'biorxiv':
                raise
            print(f'bioRxiv search skipped: {e}')
    if source in {'chemrxiv', 'all'}:
        try:
            frames.append(chemrxiv_search(query, count=count))
        except Exception as e:
            if source == 'chemrxiv':
                raise
            print(f'chemRxiv search skipped: {e}')

    new_papers = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SEARCH_FIELDS)
    if new_papers.empty:
        print('Document search found 0 new results.')
        return
    with connect(db_path) as conn:
        records = new_papers.to_dict('records')
        added, updated = upsert_papers(conn, records)
        abstract_count = _store_search_abstracts(conn, records) if store_abstract else 0
        enrichment_summary = enrich_papers(conn, records) if enrich else {}
    print(f'Document search found {added} new results and updated {updated} existing rows.')
    if store_abstract:
        print(f'Stored {abstract_count} search-time abstracts.')
    if enrich:
        print(f'Enriched {enrichment_summary.get("succeeded", 0)} papers '
              f'({enrichment_summary.get("partial", 0)} partial, '
              f'{enrichment_summary.get("not_found", 0)} not found).')
