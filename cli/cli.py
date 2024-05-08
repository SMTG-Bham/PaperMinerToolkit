import click
from paperscraper.search import search_for_papers
from paperscraper.download import elsevier_downloader
from paperscraper.scrape import scrape_papers
from paperscraper.store import store_results


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

@click.command()
@click.argument('in_file', default='.', type=click.Path(exists=True))
@click.argument('out_file', default='.', type=click.Path(exists=True))
def store(in_file: str, out_file: str):
    store_results(in_file, out_file)