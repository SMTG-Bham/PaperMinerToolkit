from elsapy.elsclient import ElsClient
from elsapy.utils import recast_df
from paperscraper.settings import load_settings
from urllib.parse import quote_plus as url_encode
import pandas as pd
from tqdm import tqdm
import os


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
    results = api_response['search-results']['entry']
    tot_num_res = int(api_response['search-results']['opensearch:totalResults'])
    print('Document search is retrieving', tot_num_res, 'results.')
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


def search_for_papers(query: str, papers_path: str = 'papers.csv'):
    """
    Search for papers and append new results to the papers database.
    """
    new_papers = document_search(query)
    new_papers['status'] = 'retrieved'
    if os.path.isfile(papers_path):
        old_papers = pd.read_csv(papers_path, index_col=0)
        num_old_papers = len(old_papers)
        papers = pd.concat([old_papers, new_papers], ignore_index=True)
        papers.drop_duplicates(subset=range(1, 11), keep='first', inplace=True, ignore_index=True)
        papers.reset_index(drop=True, inplace=True)
        tot_num_papers = len(papers)
        num_new_papers = tot_num_papers - num_old_papers
        print('Document search found', num_new_papers, 'new results.')
        papers.to_csv(papers_path)
    else:
        papers = new_papers
        print('Document search found', len(papers), 'new results.')
        papers.to_csv(papers_path)
