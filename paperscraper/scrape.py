"""Run text and image scraping stages over downloaded papers.

This module coordinates document lookup, text chunking, PDF image extraction,
model calls, text/image reconciliation, status updates, and optional cleanup for
the main ``ps_scrape`` command.
"""

import math
import os
import pandas as pd
import random
import re
import sys
import tempfile
from tqdm import tqdm

from paperscraper.compression import compression_config, maybe_compress_text
from paperscraper.documents import (extract_pdf_images,
                                    read_document_text,
                                    read_pdf_text)
from paperscraper.corpus import PIPELINE_COLUMNS, connect, get_asset, paper_rows, upsert_paper
from paperscraper.extract import build_scrape_prompt, combine_material_records, scrape_images, scrape_text, token_length
from paperscraper.filtering import active_filter_stack, current_filter_statuses, filter_expression, filter_overview
from paperscraper.models import ModelConfig
from paperscraper.recipes import load_recipe
from paperscraper.tokenizer import prompt_token_reserve, usable_input_token_limit

SCRAPE_MODES = {'abstract', 'text', 'images', 'text-images'}
IMAGE_CONTEXT_MODES = {'none', 'paper-text'}
IMAGE_EXTRACTION_MODES = {'auto', 'embedded', 'pages'}
SCRAPE_ORDERS = {'corpus', 'random', 'publication-asc', 'publication-desc', 'title', 'paper-id'}


def _text_chunks(text: str, model_config, prompt: str = ''):
    """Split long text into chunks sized for the configured model context."""
    reserve_tokens = prompt_token_reserve(prompt, model_config=model_config, buffer_tokens=500)
    token_budget = usable_input_token_limit(model_config, reserve_tokens=reserve_tokens)
    coeff = token_length(text, model_config=model_config) / token_budget
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


def _record_chunk_plan(row, stage, chunks, model_config, summary):
    """Persist a text chunk count and warn when one input needs several model requests."""
    chunk_count = len(chunks)
    row[f'num_{stage}_chunks'] = chunk_count
    if chunk_count <= 1:
        return

    summary['chunked_inputs'] += 1
    summary['chunk_requests'] += chunk_count
    summary[f'{stage}_chunked'] += 1
    paper_id = row.get('paper_id') or 'unknown paper'
    model_name = getattr(model_config, 'name', None) or 'configured text model'
    input_limit = getattr(model_config, 'input_token_limit', None)
    limit_text = f' ({input_limit} tokens)' if input_limit is not None else ''
    print(
        f'Warning: {stage} for paper {paper_id} was split into {chunk_count} independent model requests '
        f'to fit the configured input limit{limit_text} for {model_name}. Results from separate chunks are not '
        'automatically reconciled and may contain duplicated or incomplete records.',
        file=sys.stderr,
    )


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


def _set_status(paper, column: str, status: str, error: str | None = None):
    """Update a corpus paper status field and optional error text."""
    if column not in PIPELINE_COLUMNS:
        raise KeyError(f'Unknown pipeline status column: {column}')
    paper[column] = status
    if error:
        paper['last_error'] = error
    elif status in {'succeeded', 'stored'}:
        paper['last_error'] = ''


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


def _asset_path(asset, temp_dir, fallback_name):
    """Write a corpus asset to a temporary file and return its path."""
    if asset is None:
        return None
    filename = asset.get('original_filename') or fallback_name
    path = os.path.join(temp_dir, _safe_path_part(filename))
    if not os.path.splitext(path)[1]:
        path += os.path.splitext(fallback_name)[1]
    with open(path, 'wb') as out_file:
        out_file.write(asset['content'])
    return path


def _paper_asset_paths(conn, paper, temp_dir):
    """Materialize corpus abstract, text, and PDF assets for one paper as temporary files."""
    paper_id = paper.get('paper_id')
    abstract_asset = get_asset(conn, paper_id, 'abstract')
    text_asset = get_asset(conn, paper_id, 'text')
    pdf_asset = get_asset(conn, paper_id, 'pdf')
    key = _safe_path_part(paper_id)
    paper_temp_dir = os.path.join(temp_dir, key)
    os.makedirs(paper_temp_dir, exist_ok=True)
    return {
        'abstract': _asset_path(abstract_asset, paper_temp_dir, f'{key}-abstract.txt'),
        'text': _asset_path(text_asset, paper_temp_dir, f'{key}.txt'),
        'pdf': _asset_path(pdf_asset, paper_temp_dir, f'{key}.pdf'),
    }


def _publication_key(paper):
    """Return a stable sort key for publication-date ordering."""
    return str(paper.get('publication_date') or '9999-99-99')


def _select_papers(papers, scrape_order='corpus', scrape_count=None):
    """Order and optionally limit corpus papers for a scrape run."""
    if scrape_order not in SCRAPE_ORDERS:
        raise ValueError(f'scrape_order must be one of: {", ".join(sorted(SCRAPE_ORDERS))}')
    if scrape_count is not None and scrape_count < 1:
        raise ValueError('scrape_count must be a positive integer')
    selected = list(papers)
    if scrape_order == 'random':
        random.shuffle(selected)
    elif scrape_order == 'publication-asc':
        selected.sort(key=_publication_key)
    elif scrape_order == 'publication-desc':
        selected.sort(key=_publication_key, reverse=True)
    elif scrape_order == 'title':
        selected.sort(key=lambda paper: str(paper.get('title') or '').lower())
    elif scrape_order == 'paper-id':
        selected.sort(key=lambda paper: str(paper.get('paper_id') or ''))
    if scrape_count is not None:
        return selected[:scrape_count]
    return selected


def scrape_papers(db_path: str = 'papers.db',
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
                  output_path: str = 'temp_scraped_materials.csv',
                  force: bool = False,
                  ignore_filters: bool = False,
                  scrape_count: int | None = None,
                  scrape_order: str = 'corpus',
                  compression_scope: str = 'none',
                  compression_mode: str = 'auto',
                  compression_ratio: float | str = 'auto',
                  compression_content_detection: bool = True,
                  ):
    """Scrape downloaded papers with text, images, or both and write material rows."""
    mode = mode.lower()
    image_context = image_context.lower()
    image_extraction = image_extraction.lower()
    compression = compression_config(compression_scope,
                                     compression_mode,
                                     ratio=compression_ratio,
                                     content_detection=compression_content_detection)
    if mode not in SCRAPE_MODES:
        raise ValueError(f'mode must be one of: {", ".join(sorted(SCRAPE_MODES))}')
    if image_context not in IMAGE_CONTEXT_MODES:
        raise ValueError(f'image_context must be one of: {", ".join(sorted(IMAGE_CONTEXT_MODES))}')
    scrape_order = str(scrape_order).lower()
    if image_extraction not in IMAGE_EXTRACTION_MODES:
        raise ValueError(f'image_extraction must be one of: {", ".join(sorted(IMAGE_EXTRACTION_MODES))}')
    if scrape_order not in SCRAPE_ORDERS:
        raise ValueError(f'scrape_order must be one of: {", ".join(sorted(SCRAPE_ORDERS))}')
    if scrape_count is not None and scrape_count < 1:
        raise ValueError('scrape_count must be a positive integer')

    should_scrape_abstract = mode == 'abstract'
    should_scrape_text = mode in {'text', 'text-images'}
    should_scrape_images = mode in {'images', 'text-images'}
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
    summary = {
        'papers': 0,
        'abstract_attempted': 0,
        'abstract_skipped': 0,
        'text_attempted': 0,
        'text_skipped': 0,
        'image_attempted': 0,
        'image_skipped': 0,
        'materials': 0,
        'chunked_inputs': 0,
        'chunk_requests': 0,
        'abstract_chunked': 0,
        'text_chunked': 0,
    }
    with connect(db_path) as conn:
        filters = active_filter_stack(conn)
        papers = paper_rows(conn)
        if filters:
            overview = filter_overview(conn)
            counts = overview['counts']
            if ignore_filters:
                print(
                    f'Ignoring active corpus filters for this scrape: {filter_expression(filters)}. '
                    f'Recorded result: included={counts["included"]}, excluded={counts["excluded"]}, '
                    f'unavailable={counts["unavailable"]}.'
                )
            else:
                statuses = current_filter_statuses(conn)
                papers = [paper for paper in papers if statuses.get(paper['paper_id']) == 'included']
                print(
                    f'Applying corpus filters: {filter_expression(filters)}. '
                    f'Final result: included={counts["included"]}, excluded={counts["excluded"]}, '
                    f'unavailable={counts["unavailable"]}.'
                )
        else:
            print('No active corpus filters; all otherwise eligible papers are available for scraping.')
        papers = _select_papers(papers, scrape_order=scrape_order, scrape_count=scrape_count)
        with tempfile.TemporaryDirectory(prefix='paperscraper-scrape-') as temp_dir:
            with tqdm(total=len(papers), desc='Scraping Papers', colour='green') as pbar:
                for row in papers:
                    paths = _paper_asset_paths(conn, row, temp_dir)
                    abstract_path = paths['abstract']
                    text_path = paths['text']
                    pdf_path = paths['pdf']
                    summary['papers'] += 1
                    row_materials = []
                    text_materials = []
                    image_materials = []
                    text_source_path = None
                    image_source_paths = []
                    text = None

                    if should_scrape_abstract:
                        if not force and row.get('abstract_scrape_status') == 'succeeded':
                            summary['abstract_skipped'] += 1
                        else:
                            summary['abstract_attempted'] += 1
                            try:
                                if not abstract_path:
                                    raise FileNotFoundError('No downloaded abstract asset found for abstract scrape.')
                                text = read_document_text(abstract_path)
                                prompt = build_scrape_prompt(recipe_data, source='text')
                                text = maybe_compress_text(text, prompt, text_config, compression)
                                text_chunks = _text_chunks(text, text_config, prompt=prompt)
                                _record_chunk_plan(row, 'abstract', text_chunks, text_config, summary)
                                for text_chunk in text_chunks:
                                    response = scrape_text(text_chunk, recipe_data, model_config=text_config)
                                    text_materials.extend(response)
                                row['num_abstract_materials'] = len(text_materials)
                                _set_status(row, 'abstract_scrape_status', 'succeeded')
                            except Exception as e:
                                _set_status(row, 'abstract_scrape_status', 'failed', str(e))

                    if should_scrape_text:
                        if not force and row.get('text_scrape_status') == 'succeeded':
                            summary['text_skipped'] += 1
                        else:
                            summary['text_attempted'] += 1
                            try:
                                source_path = text_path or pdf_path
                                if not source_path:
                                    raise FileNotFoundError('No downloaded text or PDF asset found for text scrape.')
                                text = read_document_text(source_path)
                                text_source_path = 'corpus:text' if text_path else 'corpus:pdf'
                                prompt = build_scrape_prompt(recipe_data, source='text')
                                text = maybe_compress_text(text, prompt, text_config, compression)
                                text_chunks = _text_chunks(text, text_config, prompt=prompt)
                                _record_chunk_plan(row, 'text', text_chunks, text_config, summary)
                                for text_chunk in text_chunks:
                                    response = scrape_text(text_chunk, recipe_data, model_config=text_config)
                                    text_materials.extend(response)
                                row['num_text_materials'] = len(text_materials)
                                _set_status(row, 'text_scrape_status', 'succeeded')
                            except Exception as e:
                                _set_status(row, 'text_scrape_status', 'failed', str(e))

                    if should_scrape_images:
                        image_paths = []
                        if not force and row.get('image_scrape_status') == 'succeeded':
                            summary['image_skipped'] += 1
                        else:
                            summary['image_attempted'] += 1
                            try:
                                if not pdf_path:
                                    raise FileNotFoundError('No downloaded PDF asset found for image analysis.')
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
                                    prompt = build_scrape_prompt(recipe_data, source='image', with_context=True)
                                    text = maybe_compress_text(text, prompt, vision_config, compression)
                                    context = text
                                for image_batch in _image_batches(image_paths, image_batch_size):
                                    response = scrape_images(image_batch,
                                                             recipe_data,
                                                             model_config=vision_config,
                                                             context=context,
                                                             compression_config=compression)
                                    image_source_paths.extend(image_batch)
                                    image_materials.extend(response)
                                row['image_dir'] = paper_image_dir
                                row['num_images'] = len(image_paths)
                                row['num_image_materials'] = len(image_materials)
                                _set_status(row, 'image_scrape_status', 'succeeded')
                                if delete_images_after:
                                    for image_path in image_paths:
                                        _delete_file(image_path)
                            except Exception as e:
                                _set_status(row, 'image_scrape_status', 'failed', str(e))

                    if text_materials and image_materials:
                        text_source = text_source_path or ''
                        image_source = ';'.join(image_source_paths)
                        try:
                            combined_materials = combine_material_records(text_materials, image_materials, recipe_data,
                                                                          model_config=text_config)
                            if not combined_materials:
                                raise ValueError('reconciliation returned no material records')
                            row_materials.extend(_append_materials(combined_materials, row, 'text+image',
                                                                   f'{text_source};{image_source}'.strip(';')))
                        except Exception as e:
                            row['last_error'] = f'Combining text and image results failed: {e}'
                            row_materials.extend(_append_materials(text_materials, row, 'text', text_source))
                            row_materials.extend(_append_materials(image_materials, row, 'image', image_source))
                    elif text_materials:
                        if should_scrape_abstract:
                            row_materials.extend(_append_materials(text_materials, row, 'abstract', 'corpus:abstract'))
                        else:
                            row_materials.extend(_append_materials(text_materials, row, 'text', text_source_path))
                    elif image_materials:
                        row_materials.extend(_append_materials(image_materials, row, 'image', ';'.join(image_source_paths)))

                    first_material, written_count = _write_materials(row_materials, first_material, output_path)
                    summary['materials'] += written_count
                    upsert_paper(conn, row)
                    pbar.update(1)

    print(
        f"Scrape complete: {summary['papers']} papers processed, "
        f"{summary['materials']} material rows written to {output_path}."
    )
    if summary['abstract_skipped'] or summary['text_skipped'] or summary['image_skipped']:
        print(
            f"Skipped already successful stages: abstracts={summary['abstract_skipped']}, "
            f"text={summary['text_skipped']}, "
            f"images={summary['image_skipped']}. Use --force to rescrape."
        )
    if summary['materials'] == 0:
        print(f"No new scraped material rows were written to {output_path}.")
    if summary['chunked_inputs']:
        input_summary = (
            '1 paper input was'
            if summary['chunked_inputs'] == 1
            else f"{summary['chunked_inputs']} paper inputs were"
        )
        print(
            f"Chunking warning: {input_summary} split into "
            f"{summary['chunk_requests']} independent model requests "
            f"(abstracts={summary['abstract_chunked']}, text={summary['text_chunked']}).",
            file=sys.stderr,
        )
