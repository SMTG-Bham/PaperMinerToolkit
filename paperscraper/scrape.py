from paperscraper.gpt import gpt_scrape, token_length
from paperscraper.models import ModelConfig
import json
import pandas as pd
import math
import os
from tqdm import tqdm
from PyPDF2 import PdfReader
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
CONTENT_MODES = {'text', 'images', 'both'}


def load_recipe(recipe_name: str):
    """
    Load configuration information from the recipes JSON file.
    """
    try:
        with open(str(MODULE_DIR / 'resources' / 'recipes.json'), 'r') as f:
            recipes = json.load(f)
            recipe = recipes[recipe_name.lower()]
    except FileNotFoundError:
        raise FileNotFoundError('The recipes.json file is missing. Please reinstall PaperScraper.')
    except KeyError:
        raise KeyError(f'Recipe called "{recipe_name}" does not exist.')
    except Exception as e:
        raise ValueError('The recipes.json file may be corrupted and cannot not be read. Please reinstall PaperScraper.') from e
    return recipe


def pdf_reader(pdf_path: str):
    """
    Extract text from a PDF.
    """
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text() or ''
        text += page_text
    return text


def extract_pdf_images(pdf_path: str, output_dir: str, prefix: str | None = None):
    """
    Extract embedded images from a PDF using PyMuPDF.
    """
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError('Image extraction requires PyMuPDF. Install the package with: pip install pymupdf') from e

    os.makedirs(output_dir, exist_ok=True)
    prefix = prefix or Path(pdf_path).stem
    saved = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for image_index, image in enumerate(page.get_images(full=True), start=1):
                xref = image[0]
                extracted = doc.extract_image(xref)
                ext = extracted.get('ext', 'png')
                image_bytes = extracted['image']
                filename = f'{prefix}_page-{page_index + 1}_image-{image_index}.{ext}'
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as out_file:
                    out_file.write(image_bytes)
                saved.append(filepath)
    return saved


def _text_chunks(text: str, model_name: str):
    coeff = token_length(text, model_name) / 120000
    if coeff <= 1:
        return [text]
    coeff = math.ceil(coeff)
    char_per_split = math.ceil(len(text) / coeff)
    chunks = []
    for split in range(coeff):
        if (split + 1) * char_per_split >= len(text):
            chunks.append(text[split * char_per_split:])
        else:
            chunks.append(text[split * char_per_split:(split + 1) * char_per_split])
    return chunks


def _paper_filename(papers_dir, files, row):
    scopus_id = row['dc:identifier'].split(':')[-1]
    filenames = [file for file in files if scopus_id in file]
    if filenames == []:
        return None
    return os.path.join(papers_dir, filenames[0])


def scrape_papers(
    papers_dir: str,
    papers_path: str = 'papers.csv',
    recipe: str = 'sse',
    content_mode: str = 'text',
    image_dir: str = 'paper_images',
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
):
    """
    Scrape downloaded papers.

    content_mode controls whether text, embedded PDF images, or both are scraped.
    """
    content_mode = content_mode.lower()
    if content_mode not in CONTENT_MODES:
        raise ValueError(f'content_mode must be one of: {", ".join(sorted(CONTENT_MODES))}')

    scrape_text = content_mode in {'text', 'both'}
    scrape_images = content_mode in {'images', 'both'}
    files = os.listdir(papers_dir)
    papers_df = pd.read_csv(papers_path, index_col=0)
    if 'scraped' in papers_df['status'].value_counts().keys() and scrape_text:
        print('Some scraped papers have not been stored. Would you like to re-scrape these?')
        decision = input('Yes (Y)/ No (N): ')
        if decision.lower() in ['y', 'yes']:
            papers_df['status'].replace('scraped', 'retrieved', inplace=True)
            if os.path.isfile('temp_scraped_materials.csv'):
                os.remove('temp_scraped_materials.csv')
        else:
            print('Scrape cancelled: Please run ps_store before scraping.')
            exit()

    recipe_data = load_recipe(recipe)
    model_config = ModelConfig.from_settings(name=model, provider=provider, base_url=base_url)
    first_material = True
    retrieved_count = papers_df['status'].value_counts().get('retrieved', 0)
    if retrieved_count == 0:
        print('No retrieved papers to scrape.')
        return

    with tqdm(total=retrieved_count, desc='Scraping Papers', colour='green') as pbar:
        for i, row in papers_df.iterrows():
            if row['status'] == 'stored':
                continue
            filename = _paper_filename(papers_dir, files, row)
            if filename is None:
                retrieved_count -= 1
                pbar.total = retrieved_count
                pbar.refresh()
                continue

            extension = Path(filename).suffix.lower()
            if scrape_images:
                if extension != '.pdf':
                    print(f'Skipping image extraction for non-PDF file: {filename}')
                else:
                    scopus_id = row['dc:identifier'].split(':')[-1]
                    extract_pdf_images(filename, image_dir, prefix=scopus_id)

            if scrape_text:
                if extension == '.txt':
                    with open(filename, 'r', encoding='utf-8') as f:
                        text = f.read()
                elif extension == '.pdf':
                    text = pdf_reader(filename)
                    index = text.lower().rfind('references')
                    if index != -1:
                        text = text[:index]
                else:
                    print(f'Skipping unsupported file type: {filename}')
                    pbar.update(1)
                    continue

                for text_chunk in _text_chunks(text, model_config.name):
                    response = gpt_scrape(text_chunk, recipe_data, model_config=model_config)
                    if response == 'None':
                        continue
                    materials = []
                    for material in response:
                        material['Scopus id'] = row['dc:identifier']
                        material['doi'] = row.get('prism:doi')
                        material['Publication date'] = row.get('prism:coverDate')
                        materials.append(material)
                    if materials in [[], '', '""', "''"]:
                        continue
                    if first_material:
                        materials_df = pd.DataFrame(materials)
                        first_material = False
                    else:
                        for material in materials:
                            materials_df.loc[len(materials_df)] = material
                            materials_df.reset_index(drop=True, inplace=True)
                    materials_df.to_csv('temp_scraped_materials.csv')

            papers_df.loc[i, 'status'] = 'scraped'
            papers_df.to_csv(papers_path)
            pbar.update(1)
