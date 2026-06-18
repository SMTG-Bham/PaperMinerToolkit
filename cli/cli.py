import click
from paperscraper.search import search_for_papers
from paperscraper.download import elsevier_downloader
from paperscraper.imports import import_pdfs
from paperscraper.scrape import scrape_papers
from paperscraper.store import store_results
from paperscraper.settings import get_model_profile, infer_model_capabilities, set_model_profile, update_elsevier_key, update_openai_key, update_model_settings
from paperscraper.utilities import reset, status, sort, shuffle


@click.command()
@click.argument('query', default='Lithium solid electrolyte', type=str)
@click.argument('path', default='papers.csv', type=str)
def paper_search(query: str, path: str):
    search_for_papers(query, path)


@click.command()
@click.argument('dir', default='papers', type=click.Path(exists=True, file_okay=False))
@click.argument('path', default='external_papers.csv', type=click.Path())
@click.option('--no-crossref', is_flag=True, default=False, help='Only scrape DOI from PDFs; skip Crossref metadata lookup.')
def import_pdf_folder(dir: str, path: str, no_crossref: bool):
    import_pdfs(dir, path, use_crossref=not no_crossref)


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
@click.option('--mode', type=click.Choice(['text', 'images', 'text-images']), default='text', show_default=True)
@click.option('--image-context', type=click.Choice(['none', 'paper-text']), default='none', show_default=True)
@click.option('--image-dir', default='paper_images', type=click.Path(), show_default=True)
@click.option('--model', default=None, help='Text model name override for this scrape run.')
@click.option('--provider', default=None, help='Text provider override: openai, anthropic, or local.')
@click.option('--base-url', default=None, help='Text model base URL override.')
@click.option('--vision-model', default=None, help='Vision model name override for this scrape run.')
@click.option('--vision-provider', default=None, help='Vision provider override.')
@click.option('--vision-base-url', default=None, help='Vision model base URL override.')
@click.option('--delete-images-after', is_flag=True, default=False, help='Delete extracted images after successful image analysis.')
@click.option('--delete-papers-after', is_flag=True, default=False, help='Delete downloaded paper files after successful scraping.')
def scrape(
    dir: str,
    path: str,
    recipe: str,
    mode: str,
    image_context: str,
    image_dir: str,
    model: str | None,
    provider: str | None,
    base_url: str | None,
    vision_model: str | None,
    vision_provider: str | None,
    vision_base_url: str | None,
    delete_images_after: bool,
    delete_papers_after: bool,
):
    scrape_papers(
        dir,
        path,
        recipe,
        mode=mode,
        image_dir=image_dir,
        image_context=image_context,
        model=model,
        provider=provider,
        base_url=base_url,
        vision_model=vision_model,
        vision_provider=vision_provider,
        vision_base_url=vision_base_url,
        delete_images_after=delete_images_after,
        delete_papers_after=delete_papers_after,
    )


@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
@click.argument('in_file', default='temp_scraped_materials.csv', type=click.Path())
@click.argument('out_file', default='materials.csv', type=click.Path())
@click.argument('recipe', default='sse', type=str)
@click.option('--assume-yes', is_flag=True, default=False, help='Store converted results without an interactive confirmation prompt.')
def store(path: str, in_file: str, out_file: str, recipe: str, assume_yes: bool):
    store_results(path, in_file, out_file, True, recipe, assume_yes=assume_yes)


def update_elsevier_api_key():
    update_elsevier_key()


def update_openai_api_key():
    update_openai_key()


@click.command()
@click.argument('profile', default='text', type=click.Choice(['text', 'vision']))
@click.option('--provider', prompt=True, help='Provider: openai, anthropic, or local.')
@click.option('--model', prompt=True, help='Model name.')
@click.option('--base-url', default=None, help='Base URL for local providers.')
@click.option('--api-key', default=None, help='Provider API key. Leave unset for env/provider defaults.')
@click.option('--capability', 'capabilities', multiple=True, help='Optional override. By default capabilities are inferred from profile/model name.')
@click.option('--temperature', default=0.0, type=float, show_default=True, help='Sampling temperature for model requests.')
@click.option('--top-p', default=1.0, type=float, show_default=True, help='Nucleus sampling probability mass for model requests.')
def model_config(profile: str, provider: str, model: str, base_url: str | None, api_key: str | None, capabilities: tuple[str], temperature: float, top_p: float):
    caps = list(capabilities) or infer_model_capabilities(profile, model)
    set_model_profile(profile, provider, model, base_url, api_key, caps, temperature=temperature, top_p=top_p)
    click.echo(f'Updated {profile} model profile: {provider}/{model} [{", ".join(caps)}] temperature={temperature} top_p={top_p}')


def update_model_config():
    update_model_settings()


@click.command()
def model_status():
    for profile in ['text', 'vision']:
        config = get_model_profile(profile)
        capabilities = ', '.join(config.get('capabilities', []))
        click.echo(f'{profile}: {config.get("provider")}/{config.get("model")} capabilities=[{capabilities}] temperature={config.get("temperature")} top_p={config.get("top_p")} base_url={config.get("base_url")}')


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
@click.argument('field', default='metadata_status', type=str)
@click.option('--ascending', is_flag=True, show_default=True, default=True)
def sort_df(path: str, field: str, ascending: bool):
    sort(path, field, ascending)


@click.command()
@click.argument('path', default='papers.csv', type=click.Path(exists=True))
def shuffle_papers(path: str):
    shuffle(path)
