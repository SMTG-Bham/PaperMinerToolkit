"""Run text and image scraping stages over downloaded papers.

This module coordinates document lookup, text chunking, PDF image extraction,
model calls, text/image reconciliation, status updates, and optional cleanup for
the main ``ps_scrape`` command.
"""

from paperscraper.documents import extract_pdf_images, pdf_file_for_row, read_document_text, read_pdf_text, text_file_for_row
from paperscraper.extract import combine_material_records, scrape_images as analyze_images, scrape_text as analyze_text, token_length
from paperscraper.models import ModelConfig
from paperscraper.pipeline import ensure_pipeline_columns, set_status, write_papers
from paperscraper.recipes import load_recipe
import pandas as pd
import re
import math
import os
from tqdm import tqdm

SCRAPE_MODES = {'text', 'images', 'text-images'}
IMAGE_CONTEXT_MODES = {'none', 'paper-text'}
IMAGE_EXTRACTION_MODES = {'auto', 'embedded', 'pages'}


def _text_chunks(text: str, model_name: str):
    """Split long text into chunks sized for the configured model context."""
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


def _append_materials(materials, row, source, source_path):
    """Attach paper metadata and source provenance to extracted material rows."""
    output = []
    for material in materials:
        material['Paper id'] = row['paper_id']
        material['doi'] = row.get('doi')
        material['Publication date'] = row.get('publication_date')
        material['Source'] = source
        material['Source path'] = source_path
        output.append(material)
    return output


def _write_materials(materials, first_material, output_path):
    """Append extracted material rows to the scrape output CSV."""
    if not materials:
        return first_material, 0
    if first_material or not os.path.isfile(output_path):
        materials_df = pd.DataFrame(materials)
        first_material = False
    else:
        materials_df = pd.read_csv(output_path, index_col=0)
        materials_df = pd.concat([materials_df, pd.DataFrame(materials)], ignore_index=True)
    materials_df.to_csv(output_path)
    return first_material, len(materials)


def _delete_file(path):
    """Delete a file path if it exists."""
    if path and os.path.isfile(path):
        os.remove(path)


def _safe_path_part(value):
    """Convert an arbitrary value into a safe path fragment."""
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value or '').strip())
    return safe.strip('._') or 'paper'


def _image_key_for_row(row, pdf_path):
    """Choose a stable image-output key for a paper row."""
    identifier = str(row.get('paper_id') or '')
    if identifier.startswith('doi:') and pdf_path:
        return _safe_path_part(os.path.splitext(os.path.basename(pdf_path))[0])
    return _safe_path_part(identifier.split(':')[-1])


def _image_batches(image_paths, batch_size):
    """Group image paths into batches for vision model requests."""
    batch_size = str(batch_size).strip().lower()
    if batch_size == 'all':
        return [image_paths]
    try:
        size = int(batch_size)
    except (TypeError, ValueError) as e:
        raise ValueError('image_batch_size must be a positive integer or "all"') from e
    if size < 1:
        raise ValueError('image_batch_size must be a positive integer or "all"')
    return [image_paths[index:index + size] for index in range(0, len(image_paths), size)]


def scrape_papers(
    papers_dir: str,
    papers_path: str = 'papers.csv',
    recipe: str = 'sse',
    mode: str = 'text',
    image_dir: str = 'paper_images',
    image_context: str = 'none',
    image_extraction: str = 'auto',
    image_dpi: int = 200,
    image_batch_size: str | int = 1,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    vision_model: str | None = None,
    vision_provider: str | None = None,
    vision_base_url: str | None = None,
    delete_images_after: bool = False,
    delete_papers_after: bool = False,
    output_path: str = 'temp_scraped_materials.csv',
    force: bool = False,
):
    """Scrape downloaded papers with text, images, or both and write material rows."""
    mode = mode.lower()
    image_context = image_context.lower()
    image_extraction = image_extraction.lower()
    if mode not in SCRAPE_MODES:
        raise ValueError(f'mode must be one of: {", ".join(sorted(SCRAPE_MODES))}')
    if image_context not in IMAGE_CONTEXT_MODES:
        raise ValueError(f'image_context must be one of: {", ".join(sorted(IMAGE_CONTEXT_MODES))}')
    if image_extraction not in IMAGE_EXTRACTION_MODES:
        raise ValueError(f'image_extraction must be one of: {", ".join(sorted(IMAGE_EXTRACTION_MODES))}')

    should_scrape_text = mode in {'text', 'text-images'}
    should_scrape_images = mode in {'images', 'text-images'}
    files = os.listdir(papers_dir)
    papers_df = ensure_pipeline_columns(pd.read_csv(papers_path, index_col=0))
    recipe_data = load_recipe(recipe)
    text_config = ModelConfig.from_profile('text', name=model, provider=provider, base_url=base_url)
    vision_config = None
    if should_scrape_images:
        vision_config = ModelConfig.from_profile(
            'vision',
            name=vision_model,
            provider=vision_provider,
            base_url=vision_base_url,
        )
        vision_config.require('vision')

    first_material = not os.path.isfile(output_path)
    target_count = len(papers_df)
    summary = {
        'papers': 0,
        'text_attempted': 0,
        'text_skipped': 0,
        'image_attempted': 0,
        'image_skipped': 0,
        'materials': 0,
    }
    with tqdm(total=target_count, desc='Scraping Papers', colour='green') as pbar:
        for i, row in papers_df.iterrows():
            summary['papers'] += 1
            row_materials = []
            text_materials = []
            image_materials = []
            text_source_path = None
            image_source_paths = []
            text = None
            text_stage_ran = False
            image_stage_ran = False
            text_path = text_file_for_row(papers_dir, files, row)
            pdf_path = pdf_file_for_row(papers_dir, files, row)

            if should_scrape_text:
                if not force and row.get('text_scrape_status') == 'succeeded':
                    summary['text_skipped'] += 1
                else:
                    summary['text_attempted'] += 1
                    text_stage_ran = True
                    try:
                        source_path = text_path or pdf_path
                        if not source_path:
                            raise FileNotFoundError('No downloaded text or PDF file found for text scrape.')
                        text = read_document_text(source_path)
                        text_source_path = source_path
                        for text_chunk in _text_chunks(text, text_config.name):
                            response = analyze_text(text_chunk, recipe_data, model_config=text_config)
                            text_materials.extend(response)
                        papers_df.loc[i, 'num_text_materials'] = len(text_materials)
                        set_status(papers_df, i, 'text_scrape_status', 'succeeded')
                    except Exception as e:
                        set_status(papers_df, i, 'text_scrape_status', 'failed', str(e))

            if should_scrape_images:
                image_paths = []
                if not force and row.get('image_scrape_status') == 'succeeded':
                    summary['image_skipped'] += 1
                else:
                    summary['image_attempted'] += 1
                    image_stage_ran = True
                    try:
                        if not pdf_path:
                            raise FileNotFoundError('No downloaded PDF file found for image analysis.')
                        image_key = _image_key_for_row(row, pdf_path)
                        paper_image_dir = os.path.join(image_dir, image_key)
                        image_paths = extract_pdf_images(
                            pdf_path,
                            paper_image_dir,
                            prefix=image_key,
                            strategy=image_extraction,
                            dpi=image_dpi,
                        )
                        if not image_paths:
                            raise RuntimeError('No PDF images could be extracted or rendered.')
                        context = None
                        if image_context == 'paper-text':
                            if text is None:
                                text = read_pdf_text(pdf_path)
                            context = text
                        for image_batch in _image_batches(image_paths, image_batch_size):
                            response = analyze_images(image_batch, recipe_data, model_config=vision_config, context=context)
                            image_source_paths.extend(image_batch)
                            image_materials.extend(response)
                        papers_df.loc[i, 'image_dir'] = paper_image_dir
                        papers_df.loc[i, 'num_images'] = len(image_paths)
                        papers_df.loc[i, 'num_image_materials'] = len(image_materials)
                        set_status(papers_df, i, 'image_scrape_status', 'succeeded')
                        if delete_images_after:
                            for image_path in image_paths:
                                _delete_file(image_path)
                    except Exception as e:
                        set_status(papers_df, i, 'image_scrape_status', 'failed', str(e))

            if text_materials and image_materials:
                text_source = text_source_path or ''
                image_source = ';'.join(image_source_paths)
                try:
                    combined_materials = combine_material_records(text_materials, image_materials, recipe_data, model_config=text_config)
                    if not combined_materials:
                        raise ValueError('reconciliation returned no material records')
                    row_materials.extend(_append_materials(combined_materials, row, 'text+image', f'{text_source};{image_source}'.strip(';')))
                except Exception as e:
                    papers_df.loc[i, 'last_error'] = f'Combining text and image results failed: {e}'
                    row_materials.extend(_append_materials(text_materials, row, 'text', text_source))
                    row_materials.extend(_append_materials(image_materials, row, 'image', image_source))
            elif text_materials:
                row_materials.extend(_append_materials(text_materials, row, 'text', text_source_path))
            elif image_materials:
                row_materials.extend(_append_materials(image_materials, row, 'image', ';'.join(image_source_paths)))

            first_material, written_count = _write_materials(row_materials, first_material, output_path)
            summary['materials'] += written_count
            if delete_papers_after:
                if text_stage_ran and papers_df.loc[i, 'text_scrape_status'] == 'succeeded':
                    _delete_file(text_path)
                if image_stage_ran and papers_df.loc[i, 'image_scrape_status'] == 'succeeded':
                    _delete_file(pdf_path)
            write_papers(papers_df, papers_path)
            pbar.update(1)

    print(
        f"Scrape complete: {summary['papers']} papers processed, "
        f"{summary['materials']} material rows written to {output_path}."
    )
    if summary['text_skipped'] or summary['image_skipped']:
        print(
            f"Skipped already successful stages: text={summary['text_skipped']}, "
            f"images={summary['image_skipped']}. Use --force to rescrape."
        )
    if summary['materials'] == 0:
        print(f"No new scraped material rows were written to {output_path}.")
