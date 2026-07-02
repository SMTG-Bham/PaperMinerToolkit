"""Search Elsevier/Scopus and CORE, then merge results into the paper corpus.

The functions here translate provider-specific API responses into PaperScraper's
small public paper schema and append or update rows without duplicating papers
that appear in multiple sources.
"""

import os
import pandas as pd
import requests
from tqdm import tqdm

from paperscraper import elsevier
from paperscraper.corpus import PAPER_FIELDS, connect, normalize_paper, upsert_papers
from paperscraper.settings import load_settings

SEARCH_SOURCES = {'elsevier', 'core', 'all'}


def _elsevier_api_key():
    """Return the configured Elsevier API key."""
    api_key = load_settings().get('elsevier_api_key')
    if not api_key:
        raise ValueError('Elsevier API key is not configured. Run ps_elsevier_key first.')
    return api_key


def _recast_elsevier_records(records) -> pd.DataFrame:
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
                    search_fields: str = 'TITLE-ABS-KEY'):
    """
    Search Elsevier/Scopus and return at most ``count`` raw provider records.

    This is a small replacement for the Elsapy search helper so PaperScraper can
    control pagination and treat ``count`` as a hard result cap.
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
        with tqdm(total=target_results, desc='Getting Results', colour='blue') as pbar:
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


def _first(value):
    """Return the first item from a list-like provider value, or the value itself."""
    if isinstance(value, list):
        return value[0] if value else ''
    return value or ''


def _elsevier_link(value):
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


def _elsevier_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Elsevier search records into normalized paper rows."""
    rows = []
    for _, paper in results.iterrows():
        rows.append({
            'paper_id': paper.get('dc:identifier') or paper.get('eid') or '',
            'doi': paper.get('prism:doi') or '',
            'title': paper.get('dc:title') or '',
            'journal': paper.get('prism:publicationName') or '',
            'publication_date': paper.get('prism:coverDate') or '',
            'authors': paper.get('dc:creator') or paper.get('creator') or '',
            'sources': 'elsevier',
            'elsevier_link': _elsevier_link(paper.get('link')),
            'metadata_status': 'retrieved',
        })
    normalized = [normalize_paper(row) for row in rows]
    return pd.DataFrame(normalized, columns=PAPER_FIELDS)


def _core_api_key():
    """Return the configured CORE API key, if one is available."""
    settings = load_settings()
    return settings.get('core_api_key') or os.environ.get('CORE_API_KEY')


def _core_headers():
    """Build request headers for CORE API calls."""
    api_key = _core_api_key()
    headers = {'User-Agent': 'PaperScraper/0.0.1'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


def _core_download_url(work):
    """Return the best CORE PDF download URL for a work record."""
    download_url = work.get('downloadUrl') or work.get('download_url')
    if download_url:
        return download_url
    core_id = work.get('id')
    if core_id:
        return f'https://api.core.ac.uk/v3/works/{core_id}/download'
    return ''


def _core_authors(work):
    """Format CORE author data as a semicolon-separated author string."""
    authors = work.get('authors') or []
    names = []
    for author in authors:
        if isinstance(author, dict):
            names.append(author.get('name') or author.get('fullName') or '')
        else:
            names.append(str(author))
    return '; '.join(name for name in names if name)


def _core_journal(work):
    """Extract a journal or publisher name from a CORE work record."""
    journal = work.get('journal') or work.get('publisher') or ''
    if isinstance(journal, dict):
        return journal.get('title') or journal.get('name') or ''
    return journal


def _core_date(work):
    """Extract the best available publication date/year from a CORE work record."""
    return work.get('publishedDate') or work.get('published_date') or work.get('yearPublished') or work.get(
        'year') or ''


def _core_rows(works) -> pd.DataFrame:
    """Convert CORE work records into normalized paper rows."""
    rows = []
    for work in works:
        core_id = work.get('id') or ''
        doi = _first(work.get('doi') or work.get('DOI'))
        identifier = f'core:{core_id}' if core_id else (f'doi:{doi}' if doi else '')
        rows.append({
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
        })
    normalized = [normalize_paper(row) for row in rows]
    return pd.DataFrame(normalized, columns=PAPER_FIELDS)


def core_search(query: str, count: int = 200):
    """
    Search CORE works and return at most ``count`` normalized paper rows.
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


def search_for_papers(query: str, db_path: str = 'papers.db', source: str = 'all', count: int = 200):
    """
    Search the selected source(s) and merge results into ``db_path``.
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

    new_papers = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAPER_FIELDS)
    if new_papers.empty:
        print('Document search found 0 new results.')
        return
    with connect(db_path) as conn:
        added, updated = upsert_papers(conn, new_papers.to_dict('records'))
    print(f'Document search found {added} new results and updated {updated} existing rows.')
