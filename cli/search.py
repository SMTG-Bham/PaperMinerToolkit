import scraper.scraper
from ahocorapy.keywordtree import KeywordTree
import click
import os


cwd = os.getcwd()

@click.command()
@click.argument('query', default='Lithium solid electrolyte', type=click.str)
@click.argument('database', default='scopus', type=click.str)
def search_for_papers(query: str, database: str):
    papers = scraper.scraper.document_search(query, database)
    paper_dois = papers['prism:doi'].tolist()
    scraped_papers_path = f'{cwd}/scraped_papers.txt'
    if os.path.isfile(scraped_papers_path):
        with open(scraped_papers_path, 'r') as scraped_papers_file:
            scraped_papers = scraped_papers_file.read()
        kwtree = KeywordTree(case_insensitive=True)
        for doi in paper_dois:
            kwtree.add(doi)
        kwtree.finalize()
        results = kwtree.search_all(scraped_papers)
        for result in results:
            print(result)
            papers.drop(papers[papers['prism:doi'] == result[0]].index, inplace=True)
    print('doc_srch has', len(papers['prism:doi'].tolist()), 'new results.')
    papers.to_csv('papers.csv')