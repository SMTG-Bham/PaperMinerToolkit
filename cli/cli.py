import click
from paperscraper.search import search_for_papers
from paperscraper.download import elsevier_downloader
from paperscraper.scrape import scrape_papers
from paperscraper.store import store_results
from paperscraper.settings import update_elsevier_key, update_openai_key
from paperscraper.utilities import reset, status, sort, shuffle


@click.command()
@click.argument('query', default='Lithium solid electrolyte', type=str)
@click.argument('path', default='papers.csv', type=str)
def paper_search(query: str, path: str):
    search_for_papers(query, path)

@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
@click.argument('dir', default='papers', type=click.Path())
def elsevier_download(path: str, dir: str):
    elsevier_downloader(path, dir)

@click.command()
@click.argument('dir', default='papers', type=click.Path(exists=True))
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
@click.argument('recipe', default='sse', type=str)
def scrape(dir: str, path: str, recipe: str):
    scrape_papers(dir, path, recipe)

@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
@click.argument('in_file', default='temp_scraped_materials.csv', type=click.Path(exists=True))
@click.argument('out_file', default='materials.csv', type=click.Path())
@click.argument('recipe', default='sse', type=str)
def store(path: str, in_file: str, out_file: str, recipe: str):
    store_results(path, in_file, out_file, True, recipe)

def update_elsevier_api_key():
    update_elsevier_key()

def update_openai_api_key():
    update_openai_key()

@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
def reset_scraper(path: str):
    reset(path)

@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
def scraper_status(path: str):
    status(path)

@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
@click.argument('field', default='status', type=str)
@click.option("--ascending", is_flag=True, show_default=True, default=True)
def sort_df(path: str, field: str, ascending: bool):
    sort(path, field, ascending)

@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
def shuffle_papers(path: str):
    shuffle(path)