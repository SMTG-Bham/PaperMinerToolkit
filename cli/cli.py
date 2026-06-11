import click
from paperscraper.search import search_for_papers
from paperscraper.download import elsevier_downloader
from paperscraper.scrape import scrape_papers
from paperscraper.store import store_results
from paperscraper.settings import update_elsevier_key, update_openai_key, update_model_settings
from paperscraper.utilities import reset, status, sort, shuffle


@click.command()
@click.argument('query', default='Lithium solid electrolyte', type=str)
@click.argument('path', default='papers.csv', type=str)
def paper_search(query: str, path: str):
    search_for_papers(query, path)


@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
@click.argument('dir', default='papers', type=click.Path())
@click.option('--format', 'download_format', type=click.Choice(['text', 'pdf', 'both']), default='text', show_default=True)
def elsevier_download(path: str, dir: str, download_format: str):
    elsevier_downloader(path, dir, download_format=download_format)


@click.command()
@click.argument('dir', default='papers', type=click.Path(exists=True))
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
@click.argument('recipe', default='sse', type=str)
@click.option('--content', 'content_mode', type=click.Choice(['text', 'images', 'both']), default='text', show_default=True)
@click.option('--image-dir', default='paper_images', type=click.Path(), show_default=True)
@click.option('--model', default=None, help='Model name override for this scrape run.')
@click.option('--provider', default=None, help='Provider override: openai, anthropic, openai-compatible, local, or hpc.')
@click.option('--base-url', default=None, help='OpenAI-compatible base URL for local/HPC models.')
def scrape(dir: str, path: str, recipe: str, content_mode: str, image_dir: str, model: str | None, provider: str | None, base_url: str | None):
    scrape_papers(dir, path, recipe, content_mode=content_mode, image_dir=image_dir, model=model, provider=provider, base_url=base_url)


@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
@click.argument('in_file', default='temp_scraped_materials.csv', type=click.Path(exists=True))
@click.argument('out_file', default='materials.csv', type=click.Path())
@click.argument('recipe', default='sse', type=str)
@click.option('--assume-yes', is_flag=True, default=False, help='Store converted results without an interactive confirmation prompt.')
def store(path: str, in_file: str, out_file: str, recipe: str, assume_yes: bool):
    store_results(path, in_file, out_file, True, recipe, assume_yes=assume_yes)


def update_elsevier_api_key():
    update_elsevier_key()


def update_openai_api_key():
    update_openai_key()


def update_model_config():
    update_model_settings()


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
@click.option('--ascending', is_flag=True, show_default=True, default=True)
def sort_df(path: str, field: str, ascending: bool):
    sort(path, field, ascending)


@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
def shuffle_papers(path: str):
    shuffle(path)
