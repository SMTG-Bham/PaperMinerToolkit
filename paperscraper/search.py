from elsapy.elsclient import ElsClient
from elsapy.utils import recast_df
from paperscraper.pipeline import merge_paper_rows, normalize_paper_columns, read_papers, write_papers
from paperscraper.settings import load_settings
from urllib.parse import quote_plus as url_encode
import pandas as pd
from tqdm import tqdm
import os
import requests


SEARCH_SOURCES = {'elsevier', 'core', 'all'}


def _elsevier_client():
    api_key = load_settings().get('elsevier_api_key')
    if not api_key:
        raise ValueError('Elsevier API key is not configured. Run ps_elsevier_key first.')
    return ElsClient(api_key)


def document_search(query: str,
                    index: str = 'scopus',
                    count: int = 200,
                    get_all: bool = True,
                    search_fields: str = 'TITLE-ABS-KEY'):
    """
    Complete a search for papers. (This was rewritten from the Elsapy package to fix some bugs)
    """
    client = _elsevier_client()
    base_url = 'https://api.elsevier.com/content/search/'
    index = index.lower()
    url = base_url + index
    query = f'{search_fields}({query})'
    url += f'?query={url_encode(query)}'
    count_str = str(count)
    url += f'&count={count_str}'
    if index == 'scopus':
        url += '&cursor=*'
    api_response = client.exec_request(url)
    tot_num_res = int(api_response['search-results']['opensearch:totalResults'])
    print('Document search is retrieving', tot_num_res, 'results.')
    if tot_num_res == 0:
        return pd.DataFrame()
    results = api_response['search-results'].get('entry', [])
    if get_all:
        with tqdm(range(tot_num_res), desc='Getting Results', colour='blue') as pbar:
            num_res = count
            pbar.update(count)
            upper_limit_reached = False
            while (num_res < tot_num_res) and not upper_limit_reached:
                for e in api_response['search-results']['link']:
                    if e['@ref'] == 'next':
                        next_url = e['@href']
                api_response = client.exec_request(next_url)
                results += api_response['search-results']['entry']
                num_res += count
                if num_res >= 5000 and index != 'scopus':
                    upper_limit_reached = True
                if num_res > tot_num_res:
                    count = tot_num_res - num_res + count
                pbar.update(count)
    results_df = recast_df(pd.DataFrame(results))
    return results_df


def _first(value):
    if isinstance(value, list):
        return value[0] if value else ''
    return value or ''


def _elsevier_rows(results: pd.DataFrame) -> pd.DataFrame:
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
            'elsevier_link': paper.get('link') or '',
            'metadata_status': 'retrieved',
        })
    return normalize_paper_columns(pd.DataFrame(rows))


def _core_api_key():
    settings = load_settings()
    return settings.get('core_api_key') or os.environ.get('CORE_API_KEY')


def _core_headers():
    api_key = _core_api_key()
    headers = {'User-Agent': 'PaperScraper/0.0.1'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


def _core_download_url(work):
    download_url = work.get('downloadUrl') or work.get('download_url')
    if download_url:
        return download_url
    core_id = work.get('id')
    if core_id:
        return f'https://api.core.ac.uk/v3/works/{core_id}/download'
    return ''


def _core_authors(work):
    authors = work.get('authors') or []
    names = []
    for author in authors:
        if isinstance(author, dict):
            names.append(author.get('name') or author.get('fullName') or '')
        else:
            names.append(str(author))
    return '; '.join(name for name in names if name)


def _core_journal(work):
    journal = work.get('journal') or work.get('publisher') or ''
    if isinstance(journal, dict):
        return journal.get('title') or journal.get('name') or ''
    return journal


def _core_date(work):
    return work.get('publishedDate') or work.get('published_date') or work.get('yearPublished') or work.get('year') or ''


def _core_rows(works) -> pd.DataFrame:
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
    return normalize_paper_columns(pd.DataFrame(rows))


def core_search(query: str, count: int = 200):
    """
    Search CORE works and return normalized paper rows.
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


def search_for_papers(query: str, papers_path: str = 'papers.csv', source: str = 'all', count: int = 200):
    """
    Search for papers and append new results to the papers database.
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

    new_papers = normalize_paper_columns(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    if new_papers.empty:
        print('Document search found 0 new results.')
        return
    if os.path.isfile(papers_path):
        old_papers = read_papers(papers_path)
        papers, added, updated = merge_paper_rows(old_papers, new_papers)
        print(f'Document search found {added} new results and updated {updated} existing rows.')
        write_papers(papers, papers_path)
    else:
        papers, added, updated = merge_paper_rows(pd.DataFrame(), new_papers)
        print(f'Document search found {added} new results and updated {updated} existing rows.')
        write_papers(papers, papers_path)
