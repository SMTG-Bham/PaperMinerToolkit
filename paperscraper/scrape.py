from paperscraper.documents import extract_pdf_images, pdf_file_for_row, read_document_text, read_pdf_text, text_file_for_row
from paperscraper.extract import scrape_images as analyze_images, scrape_text as analyze_text, token_length
from paperscraper.models import ModelConfig
from paperscraper.pipeline import ensure_pipeline_columns, set_status, write_papers
from paperscraper.recipes import load_recipe
import pandas as pd
import math
import os
from tqdm import tqdm

SCRAPE_MODES = {'text', 'images', 'text-images'}
IMAGE_CONTEXT_MODES = {'none', 'paper-text'}


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


def _write_materials(materials, first_material, output_path):
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
    output_path: str = 'temp_scraped_materials.csv',
    force: bool = False,
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
                        text_materials = []
                        for text_chunk in _text_chunks(text, text_config.name):
                            response = analyze_text(text_chunk, recipe_data, model_config=text_config)
                            text_materials.extend(_append_materials(response, row, 'text', source_path))
                        row_materials.extend(text_materials)
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
                        scopus_id = row['dc:identifier'].split(':')[-1]
                        paper_image_dir = os.path.join(image_dir, scopus_id)
                        image_paths = extract_pdf_images(pdf_path, paper_image_dir, prefix=scopus_id)
                        if not image_paths:
                            raise RuntimeError('No embedded images were found in the PDF.')
                        context = None
                        if image_context == 'paper-text':
                            if text is None:
                                text = read_pdf_text(pdf_path)
                            context = text
                        image_materials = []
                        for image_path in image_paths:
                            response = analyze_images([image_path], recipe_data, model_config=vision_config, context=context)
                            image_materials.extend(_append_materials(response, row, 'image', image_path))
                        row_materials.extend(image_materials)
                        papers_df.loc[i, 'image_dir'] = paper_image_dir
                        papers_df.loc[i, 'num_images'] = len(image_paths)
                        papers_df.loc[i, 'num_image_materials'] = len(image_materials)
                        set_status(papers_df, i, 'image_scrape_status', 'succeeded')
                        if delete_images_after:
                            for image_path in image_paths:
                                _delete_file(image_path)
                    except Exception as e:
                        set_status(papers_df, i, 'image_scrape_status', 'failed', str(e))

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
