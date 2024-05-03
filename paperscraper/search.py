from paperscraper import SETTINGS
from elsapy.elsclient import ElsClient
from elsapy.utils import recast_df
from urllib.parse import quote_plus as url_encode
import pandas as pd
from tqdm import tqdm
import os
    
## Get Elsevier API key and initialize client
client = ElsClient(SETTINGS.get('elsevier_api_key'))

## Initialize doc search object using ScienceDirect and execute search, retrieving all results
def document_search(query, 
                    index='scopus', 
                    count=200, 
                    get_all=True, 
                    search_fields='TITLE-ABS-KEY'
                    ):
    '''REWRITTEN FROM ELSAPY'''
    base_url = u'https://api.elsevier.com/content/search/'
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
    print ('Document search is retrieving', tot_num_res, 'results.')
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


## Search for papers
def search_for_papers(query, papers_path='papers.csv'):
    new_papers = document_search(query)
    if os.path.isfile(papers_path):
        old_papers = pd.read_csv(papers_path, index_col=0)
        num_old_papers = len(old_papers)
        papers = pd.concat([old_papers, new_papers], ignore_index=True)
        papers.drop_duplicates(keep='first', inplace=True, ignore_index = True)
        tot_num_papers = len(papers)
        num_new_papers = tot_num_papers - num_old_papers
        print('Document search found', num_new_papers, 'new results.')
        papers.to_csv(papers_path)
    else:
        papers = new_papers
        print('Document search found', len(papers), 'new results.')
        papers.to_csv(papers_path)
    if os.path.isfile('papers_to_scrape.csv'):
        papers_to_scrape = pd.read_csv('papers_to_scrape.csv', index_col=0)
        papers = pd.concat([papers, papers_to_scrape], ignore_index=True)
        papers.drop_duplicates(keep='first', inplace=True, ignore_index = True)
    if os.path.isfile('papers_scraped.csv'):
        papers_scraped = pd.read_csv('papers_scraped.csv', index_col=0)
        papers = pd.concat([papers, papers_scraped], ignore_index=True)
        papers.drop_duplicates(keep=False, inplace=True, ignore_index = True)
    papers.to_csv('papers_to_scrape.csv')
