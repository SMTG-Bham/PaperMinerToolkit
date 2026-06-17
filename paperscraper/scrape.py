from paperscraper.gpt import gpt_image_scrape, gpt_scrape, token_length
from paperscraper.models import ModelConfig
from paperscraper.pipeline import ensure_pipeline_columns, existing_path, set_status, write_papers
import json
import pandas as pd
import math
import os
from tqdm import tqdm
from PyPDF2 import PdfReader
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
SCRAPE_MODES = {'text', 'images', 'text-images'}
IMAGE_CONTEXT_MODES = {'none', 'paper-text'}


def load_recipe(recipe_name: str):
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
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text() or ''
        text += page_text
    return text


def extract_pdf_images(pdf_path: str, output_dir: str, prefix: str | None = None):
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError('Image analysis requires PyMuPDF. Install the package with: pip install pymupdf') from e

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


def _text_file_for_row(papers_dir, files, row):
    return existing_path(row.get('text_path')) or _legacy_file_for_extension(papers_dir, files, row, '.txt')


def _pdf_file_for_row(papers_dir, files, row):
    return existing_path(row.get('pdf_path')) or _legacy_file_for_extension(papers_dir, files, row, '.pdf')


def _legacy_file_for_extension(papers_dir, files, row, extension):
    scopus_id = row['dc:identifier'].split(':')[-1]
    filenames = [file for file in files if scopus_id in file and file.lower().endswith(extension)]
    if filenames:
        return os.path.join(papers_dir, filenames[0])
    return None


def _append_materials(materials, row, source, source_path):
    output = []
    for material in materials:
        material['Scopus id'] = row['dc:identifier']
        material['doi'] = row.get('prism:doi')
        material['Publication date'] = row.get('prism:coverDate')
        material['Source'] = source
        material['Source path'] = source_path
        output.append(material)
    return output


def _write_materials(materials, first_material):
    if not materials:
        return first_material
    if first_material or not os.path.isfile('temp_scraped_materials.csv'):
        materials_df = pd.DataFrame(materials)
        first_material = False
    else:
        materials_df = pd.read_csv('temp_scraped_materials.csv', index_col=0)
        materials_df = pd.concat([materials_df, pd.DataFrame(materials)], ignore_index=True)
    materials_df.to_csv('temp_scraped_materials.csv')
    return first_material


def _delete_file(path):
    if path and os.path.isfile(path):
        os.remove(path)


def scrape_papers(
    papers_dir: str,
    papers_path: str = 'papers.csv',
    recipe: str = 'sse',
    mode: str = 'text',
    image_dir: str = 'paper_images',
    image_context: str = 'none',
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    vision_model: str | None = None,
    vision_provider: str | None = None,
    vision_base_url: str | None = None,
    delete_images_after: bool = False,
    delete_papers_after: bool = False,
    content_mode: str | None = None,
):
    if content_mode is not None:
        mode = {'both': 'text-images'}.get(content_mode, content_mode)
    mode = mode.lower()
    image_context = image_context.lower()
    if mode not in SCRAPE_MODES:
        raise ValueError(f'mode must be one of: {", ".join(sorted(SCRAPE_MODES))}')
    if image_context not in IMAGE_CONTEXT_MODES:
        raise ValueError(f'image_context must be one of: {", ".join(sorted(IMAGE_CONTEXT_MODES))}')

    scrape_text = mode in {'text', 'text-images'}
    scrape_images = mode in {'images', 'text-images'}
    files = os.listdir(papers_dir)
    papers_df = ensure_pipeline_columns(pd.read_csv(papers_path, index_col=0))
    recipe_data = load_recipe(recipe)
    text_config = ModelConfig.from_profile('text', name=model, provider=provider, base_url=base_url)
    vision_config = None
    if scrape_images:
        vision_config = ModelConfig.from_profile(
            'vision',
            name=vision_model,
            provider=vision_provider,
            base_url=vision_base_url,
        )
        vision_config.require('vision')

    first_material = not os.path.isfile('temp_scraped_materials.csv')
    target_count = len(papers_df)
    with tqdm(total=target_count, desc='Scraping Papers', colour='green') as pbar:
        for i, row in papers_df.iterrows():
            row_materials = []
            text = None
            text_path = _text_file_for_row(papers_dir, files, row)
            pdf_path = _pdf_file_for_row(papers_dir, files, row)

            if scrape_text:
                try:
                    source_path = text_path or pdf_path
                    if not source_path:
                        raise FileNotFoundError('No downloaded text or PDF file found for text scrape.')
                    if source_path.lower().endswith('.txt'):
                        with open(source_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    elif source_path.lower().endswith('.pdf'):
                        text = pdf_reader(source_path)
                        index = text.lower().rfind('references')
                        if index != -1:
                            text = text[:index]
                    for text_chunk in _text_chunks(text, text_config.name):
                        response = gpt_scrape(text_chunk, recipe_data, model_config=text_config)
                        row_materials.extend(_append_materials(response, row, 'text', source_path))
                    set_status(papers_df, i, 'text_scrape_status', 'succeeded')
                except Exception as e:
                    set_status(papers_df, i, 'text_scrape_status', 'failed', str(e))

            if scrape_images:
                image_paths = []
                try:
                    if not pdf_path:
                        raise FileNotFoundError('No downloaded PDF file found for image analysis.')
                    scopus_id = row['dc:identifier'].split(':')[-1]
                    paper_image_dir = os.path.join(image_dir, scopus_id)
                    image_paths = extract_pdf_images(pdf_path, paper_image_dir, prefix=scopus_id)
                    if not image_paths:
                        raise RuntimeError('No embedded images were found in the PDF.')
                    context = None
                    if image_context == 'paper-text':
                        if text is None:
                            text = pdf_reader(pdf_path)
                        context = text
                    image_materials = []
                    for image_path in image_paths:
                        response = gpt_image_scrape([image_path], recipe_data, model_config=vision_config, context=context)
                        image_materials.extend(_append_materials(response, row, 'image', image_path))
                    row_materials.extend(image_materials)
                    papers_df.loc[i, 'image_dir'] = paper_image_dir
                    papers_df.loc[i, 'num_images'] = len(image_paths)
                    set_status(papers_df, i, 'image_scrape_status', 'succeeded')
                    if delete_images_after:
                        for image_path in image_paths:
                            _delete_file(image_path)
                except Exception as e:
                    set_status(papers_df, i, 'image_scrape_status', 'failed', str(e))

            first_material = _write_materials(row_materials, first_material)
            if delete_papers_after:
                if scrape_text and papers_df.loc[i, 'text_scrape_status'] == 'succeeded':
                    _delete_file(text_path)
                if scrape_images and papers_df.loc[i, 'image_scrape_status'] == 'succeeded':
                    _delete_file(pdf_path)
            (papers_df)
            write_papers(papers_df, papers_path)
            pbar.update(1)
