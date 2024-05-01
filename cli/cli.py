import click
from paperscraper.search import search_for_papers
from paperscraper.download import elsevier_downloader
from paperscraper.scrape import scrape_papers


@click.command()
@click.argument('query', default='Lithium solid electrolyte', type=str)
@click.argument('path', default='papers.csv', type=str)
def paper_search(query: str, path: str):
    search_for_papers(query, path)

@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
def elsevier_download(path: str):
    elsevier_downloader(path)

@click.command()
@click.argument('path', default='.', type=click.Path(exists=True))
def scrape(path: str):
    scrape_papers(path)