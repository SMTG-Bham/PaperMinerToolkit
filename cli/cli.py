import click
from scraper.search import search_for_papers
from scraper.download import elsevier_downloader
from scraper.scrape import scrape_papers


@click.command()
@click.argument('query', default='Lithium solid electrolyte', type=str)
@click.argument('database', default='scopus', type=str)
def paper_search(query: str, database: str):
    search_for_papers(query, database)

@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
def elsevier_download(path: str):
    elsevier_downloader(path)

@click.command()
@click.argument('path', default='.', type=click.Path(exists=True))
def scrape(path: str):
    scrape_papers(path)