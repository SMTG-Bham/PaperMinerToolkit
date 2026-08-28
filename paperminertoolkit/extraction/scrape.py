"""Run text and image scraping stages over downloaded papers.

This module coordinates document lookup, text chunking, PDF image extraction,
model calls, text/image reconciliation, status updates, and optional cleanup for
the main ``pmt scrape`` command.
"""

from __future__ import annotations

import math
import os
import pandas as pd
import random
import re
import sqlite3
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from tqdm import tqdm
from typing import Any, TypeAlias

from paperminertoolkit.extraction.compression import compression_config, maybe_compress_text
from paperminertoolkit.corpus.documents import (extract_pdf_images,
                                    read_document_text,
                                    read_pdf_text)
from paperminertoolkit.corpus.database import (PIPELINE_COLUMNS,
                                    connect,
                                    get_asset,
                                    get_figure_assets,
                                    paper_rows,
                                    set_figure_extraction_status,
                                    upsert_paper)
from paperminertoolkit.extraction.extract import build_scrape_prompt, combine_material_records, scrape_images, scrape_text, token_length
from paperminertoolkit.corpus.filtering import active_filter_stack, current_filter_statuses, filter_expression, filter_overview
from paperminertoolkit.extraction.models import ModelConfig
from paperminertoolkit.extraction.recipes import load_recipe
from paperminertoolkit.extraction.tokenizer import _ModelConfigSource, prompt_token_reserve, usable_input_token_limit
from paperminertoolkit.workflows.figures import store_pdf_layout_figures

SCRAPE_MODES = {'abstract', 'text', 'images', 'text-images'}
IMAGE_CONTEXT_MODES = {'none', 'paper-text'}
IMAGE_EXTRACTION_MODES = {'auto', 'embedded', 'pages', 'layout'}
SCRAPE_ORDERS = {'corpus', 'random', 'publication-asc', 'publication-desc', 'title', 'paper-id'}
_Paper: TypeAlias = dict[str, Any]
_Material: TypeAlias = dict[str, Any]


def _text_chunks(text: str, model_config: _ModelConfigSource, prompt: str = '') -> list[str]:
    """Split text into chunks that fit a model context window.

    Parameters
    ----------
    text : str
        Text to split.
    model_config : _ModelConfigSource
        Model configuration used to calculate the input budget.
    prompt : str, optional
        Prompt that shares the model context window with the text.

    Returns
    -------
    list[str]
        One or more character-contiguous text chunks.
    """
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


def _record_chunk_plan(
    row: _Paper,
    stage: str,
    chunks: Sequence[object],
    model_config: _ModelConfigSource,
    summary: dict[str, int],
) -> None:
    """Record a chunk plan and warn about multi-request inputs.

    Parameters
    ----------
    row : _Paper
        Corpus row updated with the chunk count.
    stage : str
        Pipeline stage name used in row and summary keys.
    chunks : Sequence[object]
        Planned model input chunks.
    model_config : _ModelConfigSource
        Model configuration used to describe the input limit.
    summary : dict[str, int]
        Mutable scrape summary updated for multi-chunk inputs.
    """
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


def _append_materials(
    materials: Iterable[_Material],
    row: Mapping[str, Any],
    source: str,
    source_path: str | None,
    figures: Sequence[_FigurePackage] = (),
) -> list[_Material]:
    """Attach paper metadata and provenance to material records.

    Parameters
    ----------
    materials : Iterable[_Material]
        Extracted material records to enrich.
    row : Mapping[str, Any]
        Corpus paper row supplying metadata.
    source : str
        Extraction source label.
    source_path : str or None
        Source path or corpus asset identifier.
    figures : Sequence[_FigurePackage], optional
        Figures that produced these records. Their identities and captions
        are recorded so extracted evidence stays traceable to a figure.

    Returns
    -------
    list[_Material]
        Enriched material records.
    """
    figure_ids = '; '.join(figure.figure_id for figure in figures)
    figure_labels = '; '.join(figure.label or figure.figure_id for figure in figures)
    figure_sources = '; '.join(dict.fromkeys(figure.source for figure in figures if figure.source))
    output = []
    for material in materials:
        material['Paper id'] = row['paper_id']
        material['doi'] = row.get('doi')
        material['Publication date'] = row.get('publication_date')
        material['Source'] = source
        material['Source path'] = source_path
        if figures:
            material['Figure id'] = figure_ids
            material['Figure label'] = figure_labels
            material['Figure source'] = figure_sources
        output.append(material)
    return output


def _write_materials(
    materials: list[_Material],
    first_material: bool,
    output_path: str | PathLike[str],
) -> tuple[bool, int]:
    """Append extracted material records to a CSV file.

    Parameters
    ----------
    materials : list[_Material]
        Material records to write.
    first_material : bool
        Whether this is the first write in the current scrape run.
    output_path : str or os.PathLike[str]
        Destination CSV path.

    Returns
    -------
    tuple[bool, int]
        Updated first-write flag and number of records written.
    """
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


def _delete_file(path: str | PathLike[str] | None) -> None:
    """Delete a file when it exists.

    Parameters
    ----------
    path : str, os.PathLike[str], or None
        File path to delete.
    """
    if path and os.path.isfile(path):
        os.remove(path)


def _set_status(paper: _Paper, column: str, status: str, error: str | None = None) -> None:
    """Update a paper's pipeline status and error message.

    Parameters
    ----------
    paper : _Paper
        Corpus paper row to update.
    column : str
        Pipeline status column.
    status : str
        New stage status.
    error : str, optional
        Error text to store for a failed stage.

    Raises
    ------
    KeyError
        If ``column`` is not a recognized pipeline status column.
    """
    if column not in PIPELINE_COLUMNS:
        raise KeyError(f'Unknown pipeline status column: {column}')
    paper[column] = status
    if error:
        paper['last_error'] = error
    elif status in {'succeeded', 'stored'}:
        paper['last_error'] = ''


def _safe_path_part(value: object) -> str:
    """Convert a value into a safe path fragment.

    Parameters
    ----------
    value : object
        Value to normalize.

    Returns
    -------
    str
        Non-empty path-safe fragment.
    """
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value or '').strip())
    return safe.strip('._') or 'paper'


def _image_key_for_row(
    row: Mapping[str, Any],
    pdf_path: str | PathLike[str] | None,
) -> str:
    """Choose a stable image-output key for a paper.

    Parameters
    ----------
    row : Mapping[str, Any]
        Corpus paper row.
    pdf_path : str, os.PathLike[str], or None
        Materialized PDF path.

    Returns
    -------
    str
        Path-safe image output key.
    """
    identifier = str(row.get('paper_id') or '')
    if identifier.startswith('doi:') and pdf_path:
        return _safe_path_part(os.path.splitext(os.path.basename(pdf_path))[0])
    return _safe_path_part(identifier.split(':')[-1])


def _image_batches(image_paths: list[str], batch_size: int | str) -> list[list[str]]:
    """Group image paths into vision request batches.

    Parameters
    ----------
    image_paths : list of str
        Image paths to batch.
    batch_size : int or str
        Positive batch size, or ``"all"`` for a single batch.

    Returns
    -------
    list[list[str]]
        Ordered image batches.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive or ``"all"``.
    """
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


@dataclass(frozen=True, slots=True)
class _FigurePackage:
    """One figure and the context sent with it to a vision model.

    Parameters
    ----------
    path : str
        Materialized local image path.
    figure_id : str
        Parsed figure identifier used for provenance and checkpointing.
    label : str
        Display label such as ``"Figure 3"``.
    caption : str
        Figure caption.
    source : str
        Provider or detector that supplied the figure.
    """

    path: str
    figure_id: str
    label: str
    caption: str
    source: str

    @property
    def model_label(self) -> str:
        """Return the text describing this figure to the model."""
        parts = [part for part in (self.label, self.caption) if part]
        return ': '.join(parts) if parts else f'Figure {self.figure_id}'


def _figure_packages(
    assets: Iterable[Mapping[str, Any]],
    temp_dir: str | PathLike[str],
    *,
    force: bool,
) -> list[_FigurePackage]:
    """Materialize stored figure assets as vision-request packages.

    Parameters
    ----------
    assets : Iterable[Mapping[str, Any]]
        Figure assets loaded from the corpus with their content.
    temp_dir : str or os.PathLike[str]
        Directory in which to write the figure images.
    force : bool
        Whether to include figures already marked as extracted.

    Returns
    -------
    list[_FigurePackage]
        Packages for figures still awaiting extraction, oldest first.
    """
    os.makedirs(temp_dir, exist_ok=True)
    packages = []
    for asset in reversed(list(assets)):
        metadata = asset.get('metadata') or {}
        if not force and metadata.get('extraction_status') == 'succeeded':
            continue
        figure_id = str(metadata.get('figure_id') or '')
        if not figure_id:
            continue
        filename = asset.get('original_filename') or f'{figure_id}.png'
        path = os.path.join(temp_dir, _safe_path_part(filename))
        if not os.path.splitext(path)[1]:
            path += '.png'
        with open(path, 'wb') as out_file:
            out_file.write(asset['content'])
        packages.append(_FigurePackage(
            path=path,
            figure_id=figure_id,
            label=str(metadata.get('figure_label') or ''),
            caption=str(metadata.get('caption') or ''),
            source=str(asset.get('source') or ''),
        ))
    return packages


def _layout_figure_packages(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    pdf_path: str | None,
    temp_dir: str | PathLike[str],
    *,
    force: bool,
    required: bool,
) -> list[_FigurePackage]:
    """Collect layout-aware figures for one paper, detecting them if needed.

    Structured figures already downloaded into the corpus are preferred. PDF
    layout detection only runs when the corpus holds no figures for the paper,
    so a structured figure and its PDF-rendered duplicate are never both
    analysed.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : Mapping[str, Any]
        Corpus paper row.
    pdf_path : str or None
        Materialized PDF path used for detection fallback.
    temp_dir : str or os.PathLike[str]
        Directory in which to write figure images.
    force : bool
        Whether to reanalyse figures already marked as extracted.
    required : bool
        Whether a detection failure should be raised. An automatic run treats
        an unreadable PDF as "no layout figures" so it can fall back to the
        other image strategies.

    Returns
    -------
    list[_FigurePackage]
        Figure packages awaiting extraction, or an empty list when the paper
        has no layout-aware figures.

    Raises
    ------
    OSError
        If ``required`` and the PDF cannot be read.
    RuntimeError
        If ``required`` and figure detection or rendering fails.
    ValueError
        If ``required`` and the paper or detected figures are unusable.
    """
    paper_id = paper.get('paper_id')
    assets = get_figure_assets(conn, paper_id, include_content=True)
    if not assets and pdf_path:
        try:
            store_pdf_layout_figures(conn, paper, pdf_path)
        except (OSError, RuntimeError, ValueError):
            if required:
                raise
            return []
        assets = get_figure_assets(conn, paper_id, include_content=True)
    return _figure_packages(assets, temp_dir, force=force)


def _asset_path(
    asset: Mapping[str, Any] | None,
    temp_dir: str | PathLike[str],
    fallback_name: str,
) -> str | None:
    """Materialize a corpus asset in a temporary directory.

    Parameters
    ----------
    asset : Mapping[str, Any] or None
        Asset metadata and binary content.
    temp_dir : str or os.PathLike[str]
        Directory in which to write the asset.
    fallback_name : str
        Filename used when the asset has no original filename.

    Returns
    -------
    str or None
        Materialized file path, or ``None`` when no asset is supplied.
    """
    if asset is None:
        return None
    filename = asset.get('original_filename') or fallback_name
    path = os.path.join(temp_dir, _safe_path_part(filename))
    if not os.path.splitext(path)[1]:
        path += os.path.splitext(fallback_name)[1]
    with open(path, 'wb') as out_file:
        out_file.write(asset['content'])
    return path


def _paper_asset_paths(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    temp_dir: str | PathLike[str],
) -> dict[str, str | None]:
    """Materialize one paper's corpus assets as temporary files.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus database connection.
    paper : Mapping[str, Any]
        Corpus paper row identifying the assets.
    temp_dir : str or os.PathLike[str]
        Root temporary directory for materialized assets.

    Returns
    -------
    dict[str, str or None]
        Paths for the paper's abstract, text, and PDF assets.
    """
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


def _publication_key(paper: Mapping[str, Any]) -> str:
    """Build a stable publication-date sort key.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    str
        Publication date, or a value that sorts missing dates last.
    """
    return str(paper.get('publication_date') or '9999-99-99')


def _select_papers(
    papers: Iterable[_Paper],
    scrape_order: str = 'corpus',
    scrape_count: int | None = None,
) -> list[_Paper]:
    """Order and optionally limit papers for a scrape run.

    Parameters
    ----------
    papers : Iterable[_Paper]
        Corpus paper rows.
    scrape_order : str, optional
        Selection order, such as corpus order, random order, or publication
        date.
    scrape_count : int, optional
        Maximum number of papers to return.

    Returns
    -------
    list[_Paper]
        Selected paper rows.

    Raises
    ------
    ValueError
        If the order is unsupported or the count is not positive.
    """
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
                  ) -> None:
    """Scrape corpus papers and write extracted material records.

    Parameters
    ----------
    db_path : str, optional
        Path to the corpus SQLite database.
    recipe : str, optional
        Recipe name or path used to define the extraction schema.
    mode : str, optional
        Extraction source: abstract, text, images, or combined text and images.
    image_dir : str, optional
        Directory for images extracted or rendered from PDFs.
    image_context : str, optional
        Text context policy for image requests.
    image_extraction : str, optional
        Image selection strategy. ``layout`` sends stored structured figures,
        falling back to figures detected from PDF geometry, each accompanied
        by its caption. ``auto`` prefers those layout-aware figures and
        otherwise uses embedded PDF images or rendered pages.
    image_dpi : int, optional
        Resolution used when rendering PDF pages.
    image_batch_size : str or int, optional
        Images per vision request, or ``"all"``.
    model : str, optional
        Text model name override.
    provider : str, optional
        Text model provider override.
    base_url : str, optional
        Text model API base URL override.
    vision_model : str, optional
        Vision model name override.
    vision_provider : str, optional
        Vision model provider override.
    vision_base_url : str, optional
        Vision model API base URL override.
    delete_images_after : bool, optional
        Whether to delete extracted images after successful analysis.
    output_path : str, optional
        Destination CSV path for material records.
    force : bool, optional
        Whether to rerun stages already marked successful.
    ignore_filters : bool, optional
        Whether to scrape papers excluded by active corpus filters.
    scrape_count : int, optional
        Maximum number of selected papers to scrape.
    scrape_order : str, optional
        Ordering applied before the optional paper limit.
    compression_scope : str, optional
        Content types eligible for Headroom compression.
    compression_mode : str, optional
        Compression policy.
    compression_ratio : float or str, optional
        Target compression ratio, or ``"auto"``.
    compression_content_detection : bool, optional
        Whether Headroom should detect content types.

    Raises
    ------
    ValueError
        If a scrape, image, ordering, count, or compression option is invalid.
    """
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
        'layout_figures': 0,
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
        with tempfile.TemporaryDirectory(prefix='paperminertoolkit-scrape-') as temp_dir:
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
                    scraped_figures: list[_FigurePackage] = []
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
                                image_key = _image_key_for_row(row, pdf_path)
                                paper_image_dir = os.path.join(image_dir, image_key)
                                figure_packages = []
                                if image_extraction in {'auto', 'layout'}:
                                    figure_packages = _layout_figure_packages(
                                        conn,
                                        row,
                                        pdf_path,
                                        paper_image_dir,
                                        force=force,
                                        required=image_extraction == 'layout',
                                    )
                                    summary['layout_figures'] += len(figure_packages)
                                if image_extraction == 'layout' and not figure_packages:
                                    raise RuntimeError(
                                        'No layout-aware figures are available for this paper.'
                                    )
                                if figure_packages:
                                    image_paths = [figure.path for figure in figure_packages]
                                else:
                                    if not pdf_path:
                                        raise FileNotFoundError('No downloaded PDF asset found for image analysis.')
                                    image_paths = extract_pdf_images(
                                        pdf_path,
                                        paper_image_dir,
                                        prefix=image_key,
                                        strategy='auto' if image_extraction == 'auto' else image_extraction,
                                        dpi=image_dpi,
                                    )
                                    if not image_paths:
                                        raise RuntimeError('No PDF images could be extracted or rendered.')
                                context = None
                                if image_context == 'paper-text':
                                    if text is None:
                                        if not pdf_path:
                                            raise FileNotFoundError(
                                                'No downloaded PDF asset found for image context.'
                                            )
                                        text = read_pdf_text(pdf_path)
                                    prompt = build_scrape_prompt(recipe_data, source='image', with_context=True)
                                    text = maybe_compress_text(text, prompt, vision_config, compression)
                                    context = text
                                if figure_packages:
                                    for figure_batch in _image_batches(figure_packages, image_batch_size):
                                        batch_paths = [figure.path for figure in figure_batch]
                                        try:
                                            response = scrape_images(
                                                batch_paths,
                                                recipe_data,
                                                model_config=vision_config,
                                                context=context,
                                                compression_config=compression,
                                                image_labels=[
                                                    figure.model_label for figure in figure_batch
                                                ],
                                            )
                                        except Exception as batch_error:
                                            for figure in figure_batch:
                                                set_figure_extraction_status(
                                                    conn,
                                                    row['paper_id'],
                                                    figure.figure_id,
                                                    'failed',
                                                    error=str(batch_error),
                                                )
                                            raise
                                        for figure in figure_batch:
                                            set_figure_extraction_status(
                                                conn,
                                                row['paper_id'],
                                                figure.figure_id,
                                                'succeeded',
                                            )
                                        image_source_paths.extend(batch_paths)
                                        scraped_figures.extend(figure_batch)
                                        image_materials.extend(response)
                                else:
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
                                                                   f'{text_source};{image_source}'.strip(';'),
                                                                   scraped_figures))
                        except Exception as e:
                            row['last_error'] = f'Combining text and image results failed: {e}'
                            row_materials.extend(_append_materials(text_materials, row, 'text', text_source))
                            row_materials.extend(_append_materials(image_materials, row, 'image', image_source,
                                                                   scraped_figures))
                    elif text_materials:
                        if should_scrape_abstract:
                            row_materials.extend(_append_materials(text_materials, row, 'abstract', 'corpus:abstract'))
                        else:
                            row_materials.extend(_append_materials(text_materials, row, 'text', text_source_path))
                    elif image_materials:
                        row_materials.extend(_append_materials(image_materials, row, 'image',
                                                               ';'.join(image_source_paths), scraped_figures))

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
    if summary['layout_figures']:
        print(
            f"Analysed {summary['layout_figures']} layout-aware figures with their captions."
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
