"""Run text and image scraping stages over downloaded papers.

This module coordinates document lookup, text chunking, PDF image extraction,
model calls, text/image reconciliation, status updates, and optional cleanup for
the main ``pmt scrape`` command.
"""

from __future__ import annotations

import csv
import hashlib
import os
import pandas as pd
import random
import re
import sqlite3
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from os import PathLike
from tqdm import tqdm

from paperminertoolkit import _progress as progress_events
from paperminertoolkit._progress import bars_disabled
from typing import Any, TypeAlias

from paperminertoolkit.extraction.compression import CompressionConfig, compression_config, maybe_compress_text
from paperminertoolkit.corpus.tables import STRUCTURED_TABLE_FORMATS, append_tables, pdf_tables_text, structured_tables_text
from paperminertoolkit.corpus.documents import (extract_pdf_images,
                                    read_document_text,
                                    read_pdf_text)
from paperminertoolkit.corpus.database import (connect,
                                    get_asset,
                                    get_figure_assets,
                                    get_structured_documents,
                                    paper_rows,
                                    set_figure_extraction_status,
                                    set_pipeline_status as _set_status,
                                    upsert_paper)
from paperminertoolkit.extraction.extract import (ask_paper_question, build_scrape_prompt, combine_material_records,
                                                  scrape_images, scrape_text, token_length)
from paperminertoolkit.corpus.filtering import active_filter_stack, current_filter_statuses, filter_expression, filter_overview
from paperminertoolkit.extraction.merge import merge_chunk_records
from paperminertoolkit.extraction.models import ModelConfig
from paperminertoolkit.extraction.recipes import load_recipe
from paperminertoolkit.corpus.supplements import get_supplements, set_supplement_extraction_status
from paperminertoolkit.extraction.supplements import supplement_images, supplement_parts, supplement_text
from paperminertoolkit.extraction.tokenizer import _ModelConfigSource, request_token_budget
from paperminertoolkit.extraction.verify import DROPPED_COLUMNS, apply_paper_answer, verify_records
from paperminertoolkit.corpus.pdf_layout import figure_regions
from paperminertoolkit.workflows.figures import store_pdf_layout_figures

SCRAPE_MODES = {'abstract', 'text', 'images', 'text-images'}
IMAGE_CONTEXT_MODES = {'none', 'paper-text'}
IMAGE_EXTRACTION_MODES = {'auto', 'embedded', 'pages', 'layout'}
SCRAPE_ORDERS = {'corpus', 'random', 'publication-asc', 'publication-desc', 'title', 'paper-id'}
_Paper: TypeAlias = dict[str, Any]
_Material: TypeAlias = dict[str, Any]


# A supporting-information file that needs more text requests than this is a
# raw data dump, such as a genome annotation or instrument export, not reported
# results: one was split into 180 requests in a 2026 polymer-degradation run.
MAX_SUPPLEMENT_TEXT_REQUESTS = 20


_CHUNK_SAFETY = 0.9
_CHUNK_OVERLAP = 0.05


def _is_table_block(block: str) -> bool:
    """Return whether a text block contains Markdown table rows."""
    return any(line.startswith('|') for line in block.splitlines())


def _split_block(block: str, max_chars: int) -> list[str]:
    """Split an oversized block on line, then word, boundaries.

    A Markdown table is split between rows, and each continuation repeats
    the table's title and header so its columns stay readable.
    """
    lines = block.splitlines()
    header: list[str] = []
    if _is_table_block(block):
        separator = next((index for index, line in enumerate(lines) if line.startswith('| ---')), None)
        if separator is not None and len('\n'.join(lines[:separator + 1])) < max_chars // 2:
            header = lines[:separator + 1]
            lines = lines[separator + 1:]
    pieces: list[str] = []
    current: list[str] = list(header)
    for line in lines:
        parts = textwrap.wrap(line, max_chars, break_on_hyphens=False) if len(line) > max_chars else [line]
        for part in parts:
            if len(current) > len(header) and len('\n'.join([*current, part])) > max_chars:
                pieces.append('\n'.join(current))
                current = list(header)
            current.append(part)
    if len(current) > len(header):
        pieces.append('\n'.join(current))
    return pieces


def _overlap(block: str, max_chars: int) -> str:
    """Return the tail of a prose block to repeat at the start of the next chunk."""
    if _is_table_block(block):
        return ''
    limit = max(int(max_chars * _CHUNK_OVERLAP), 1)
    if len(block) <= limit:
        return block
    tail = block[-limit:]
    sentence, word = tail.find('. '), tail.find(' ')
    start = sentence + 2 if sentence >= 0 else word + 1
    return tail[start:] if start < len(tail) else tail


def _pack_blocks(blocks: Sequence[str], max_chars: int) -> list[str]:
    """Greedily pack paragraph blocks into chunks with a short overlap."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for block in blocks:
        if current and size + len(block) + 2 > max_chars:
            chunks.append('\n\n'.join(current))
            carry = _overlap(current[-1], max_chars)
            current = [carry] if carry else []
            size = len(carry)
        current.append(block)
        size += len(block) + 2
    if current:
        chunks.append('\n\n'.join(current))
    return chunks
def _text_chunks(text: str, model_config: _ModelConfigSource, prompt: str = '') -> list[str]:
    """Split text into chunks that fit a model context window.

    Text is split between paragraphs (blank lines), never inside a Markdown
    table, and each chunk repeats the end of the previous paragraph so a
    statement cut at the boundary is still seen whole. Chunks that still
    exceed the budget are split again.

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
        One or more text chunks.
    """
    token_budget = request_token_budget(prompt, model_config)
    tokens = token_length(text, model_config=model_config)
    if tokens <= token_budget:
        return [text]
    chars_per_token = len(text) / tokens
    return _fit_chunks(text, token_budget, chars_per_token, model_config)


def _fit_chunks(
    text: str,
    token_budget: int,
    chars_per_token: float,
    model_config: _ModelConfigSource,
    depth: int = 0,
) -> list[str]:
    """Pack text into budget-sized chunks, re-splitting any that overflow.

    Token density is not uniform: a table of numbers can need several times
    the tokens of the same length of prose. A chunk that still overflows is
    split again using its own measured density, with a tighter margin at
    each level.
    """
    max_chars = max(int(token_budget * chars_per_token * _CHUNK_SAFETY ** (depth + 1)), 1)
    blocks: list[str] = []
    for block in (part.strip('\n') for part in re.split(r'\n\s*\n', text)):
        if not block.strip():
            continue
        blocks.extend(_split_block(block, max_chars) if len(block) > max_chars else [block])
    chunks = []
    for chunk in _pack_blocks(blocks, max_chars):
        tokens = token_length(chunk, model_config=model_config)
        if depth < 3 and tokens > token_budget and len(chunk) > 1:
            chunks.extend(_fit_chunks(chunk, token_budget, len(chunk) / tokens, model_config, depth + 1))
        else:
            chunks.append(chunk)
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
        f'to fit the configured input limit{limit_text} for {model_name}. Records from separate chunks are merged by '
        'their identity fields, but a record split across chunks may still be incomplete.',
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


def dropped_path(output_path: str | PathLike[str]) -> str:
    """Return the audit CSV path that accompanies a scrape output.

    Parameters
    ----------
    output_path : str or os.PathLike[str]
        Scrape output CSV.

    Returns
    -------
    str
        ``<output stem>.dropped.csv`` beside the output.
    """
    root, _ = os.path.splitext(os.fspath(output_path))
    return f'{root}.dropped.csv'


def _write_dropped(rows: Sequence[Mapping[str, str]], output_path: str | PathLike[str]) -> int:
    """Append check audit rows beside a scrape output.

    Parameters
    ----------
    rows : Sequence[Mapping[str, str]]
        Audit rows from :func:`verify_records`.
    output_path : str or os.PathLike[str]
        Scrape output CSV.

    Returns
    -------
    int
        Number of rows written.
    """
    if not rows:
        return 0
    path = dropped_path(output_path)
    new_file = not os.path.isfile(path)
    with open(path, 'a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=DROPPED_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


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
        path = _asset_path(asset, temp_dir, f'{figure_id}.png')
        assert path is not None
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


def _paper_tables(
    conn: sqlite3.Connection,
    paper_id: str,
    pdf_path: str | None,
    figure_text: bool = True,
) -> str:
    """Render one paper's tables as Markdown for the text prompt.

    Tables come from the paper's structured XML when it has one in a
    supported format, since plain-text assets are built without them.
    Otherwise ruled tables are detected in the PDF.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus database connection.
    paper_id : str
        Paper whose tables are rendered.
    pdf_path : str or None
        Materialized PDF used when no structured document is stored.
    figure_text : bool, optional
        When ``False``, tables PyMuPDF detects inside a PDF's captioned
        figures (gridded plots such as heat maps) are left out.

    Returns
    -------
    str
        Markdown tables, or an empty string when none are available.
    """
    documents = [
        document
        for document in get_structured_documents(conn, paper_id)
        if str(document['metadata'].get('document_format', '')).lower() in STRUCTURED_TABLE_FORMATS
    ]
    if documents:
        for document in documents:
            tables = structured_tables_text(
                document['content'],
                str(document['metadata']['document_format']),
                paper_id,
            )
            if tables:
                return tables
        return ''
    if not pdf_path:
        return ''
    return pdf_tables_text(pdf_path, skip_regions=() if figure_text else figure_regions(pdf_path))


def paper_source_text(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    paths: Mapping[str, str | None],
    *,
    abstract: bool = False,
    tables: bool = True,
    figure_text: bool = True,
) -> str:
    """Read the text a paper is scraped from.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus database connection.
    paper : Mapping[str, Any]
        Corpus paper row.
    paths : Mapping[str, str or None]
        Materialized asset paths from :func:`_paper_asset_paths`.
    abstract : bool, optional
        Read the abstract instead of the full text.
    tables : bool, optional
        Append the paper's tables to full text.
    figure_text : bool, optional
        Keep text printed inside a PDF's figures. Text assets are unaffected.

    Returns
    -------
    str
        Abstract, or full text (text asset, else PDF) with any tables appended.

    Raises
    ------
    FileNotFoundError
        If the paper has no asset of the requested kind.
    """
    source_path = paths['abstract'] if abstract else paths['text'] or paths['pdf']
    if not source_path:
        if abstract:
            raise FileNotFoundError('No downloaded abstract asset found for abstract scrape.')
        raise FileNotFoundError('No downloaded text or PDF asset found for text scrape.')
    text = read_document_text(source_path, figure_text=figure_text)
    if not abstract and tables:
        text = append_tables(text, _paper_tables(conn, str(paper['paper_id']), paths['pdf'], figure_text=figure_text))
    return text


def _publication_key(paper: Mapping[str, Any]) -> str:
    """Build a stable publication-date sort key.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    str
        Publication date, or an empty string when the date is missing.
    """
    return str(paper.get('publication_date') or '')


def _supplement_status(statuses: Iterable[str]) -> str:
    """Aggregate supplementary member outcomes without hiding partial errors."""
    outcomes = set(statuses) - {'not_applicable'}
    if not outcomes:
        return 'not_applicable'
    if outcomes <= {'succeeded'}:
        return 'succeeded'
    if 'succeeded' in outcomes or 'partial' in outcomes:
        return 'partial'
    if 'failed' in outcomes:
        return 'failed'
    return 'unsupported'


def _scrape_supplements(
    conn: sqlite3.Connection,
    row: _Paper,
    *,
    mode: str,
    recipe: Mapping[str, Any],
    text_config: ModelConfig,
    vision_config: ModelConfig | None,
    compression: CompressionConfig,
    image_dir: str,
    image_dpi: int,
    image_batch_size: str | int,
    image_context: str,
    delete_images_after: bool,
    force: bool,
    output_path: str,
    first_material: bool,
    summary: dict[str, int],
) -> bool:
    """Extract SI independently of the main article with durable provenance.

    Each successful asset member is written before its checkpoint is stored.
    Failed members are retried while successful members are skipped, including
    when SI is enabled after the article itself was already processed.
    """
    assets = get_supplements(conn, row['paper_id'], include_content=True)
    stages = [stage for stage in ('text', 'image')
              if mode == 'text-images' or mode == ('images' if stage == 'image' else stage)]
    outcomes: dict[str, list[str]] = {stage: [] for stage in stages}
    seen_blobs = set()
    for asset in assets:
        if asset['blob_id'] in seen_blobs:
            continue
        seen_blobs.add(asset['blob_id'])
        metadata = asset.get('metadata') or {}
        pending_stages = []
        for stage in stages:
            previous_status = metadata.get(f'{stage}_extraction_status')
            if not force and previous_status in {'succeeded', 'not_applicable'}:
                summary['supplement_skipped'] += 1
                outcomes[stage].append(previous_status)
            else:
                pending_stages.append(stage)
        if not pending_stages:
            continue
        label = str(metadata.get('label') or asset.get('original_filename') or asset['blob_id'])
        source_url = str(metadata.get('source_url') or '')
        try:
            parts = supplement_parts(asset)
            parse_error = ''
        except Exception as error:
            parts = []
            parse_error = str(error)
        for stage in pending_stages:
            summary[f'supplement_{stage}_attempted'] += 1
            members = dict(metadata.get(f'{stage}_extraction_members') or {}) if not force else {}
            errors = []
            if parse_error:
                errors.append(parse_error)
                members = {'': 'failed'}
                summary['supplement_failed'] += 1
            for part in parts:
                key = part.member or part.name
                if not force and members.get(key) == 'succeeded':
                    summary['supplement_skipped'] += 1
                    continue
                if part.kind in {'unsupported', 'zip'}:
                    members[key] = 'unsupported'
                    summary['supplement_unsupported'] += 1
                    errors.append(f'{key}: unsupported supplementary format')
                    continue
                if not (part.supports_text if stage == 'text' else part.supports_images):
                    members[key] = 'not_applicable'
                    continue
                reference = f'corpus:supplement:{asset["blob_id"]}'
                if part.member:
                    reference += f'!{part.member}'
                description = f'Supporting information: {label}\nFile: {part.name}\nAsset: {reference}'
                if source_url:
                    description += f'\nSource URL: {source_url}'
                response_records = []
                image_paths = []
                try:
                    if stage == 'text':
                        text = supplement_text(part)
                        if not text.strip():
                            raise ValueError('No extractable text; use image mode for scanned PDF pages')
                        prompt = build_scrape_prompt(recipe, source='text')
                        text = maybe_compress_text(text, prompt, text_config, compression)
                        chunks = _text_chunks(text, text_config, prompt=f'{prompt}\n{description}')
                        if len(chunks) > MAX_SUPPLEMENT_TEXT_REQUESTS:
                            members[key] = 'unsupported'
                            summary['supplement_unsupported'] += 1
                            errors.append(f'{key}: too large for text extraction ({len(chunks)} requests, '
                                          f'limit {MAX_SUPPLEMENT_TEXT_REQUESTS})')
                            continue
                        if len(chunks) > 1:
                            print(f'Warning: supporting information {reference} was split into '
                                  f'{len(chunks)} independent text requests.', file=sys.stderr)
                        for chunk in chunks:
                            response_records.extend(scrape_text(
                                f'{description}\n\n{chunk}', recipe, model_config=text_config,
                            ))
                    else:
                        member_key = hashlib.sha256(key.encode()).hexdigest()[:16]
                        directory = os.path.join(image_dir, _safe_path_part(row['paper_id']),
                                                 'supplements', _safe_path_part(asset['blob_id']), member_key)
                        image_paths = supplement_images(part, directory, dpi=image_dpi)
                        if not image_paths:
                            raise ValueError('No supplementary images or PDF pages could be read')
                        context = None
                        if image_context == 'paper-text' and part.supports_text:
                            # SI context belongs to its own file, not a different article asset.
                            context = supplement_text(part)
                            prompt = build_scrape_prompt(recipe, source='image', with_context=True)
                            context = maybe_compress_text(context, prompt, vision_config, compression)
                        for batch in _image_batches(image_paths, image_batch_size):
                            response_records.extend(scrape_images(
                                batch, recipe, model_config=vision_config, context=context,
                                compression_config=compression,
                                image_labels=[f'{description}\nPage/image {image_paths.index(path) + 1}'
                                              for path in batch],
                            ))
                except Exception as error:
                    members[key] = 'failed'
                    summary['supplement_failed'] += 1
                    errors.append(f'{key}: {error}')
                    continue
                records = _append_materials(response_records, row, f'supplement-{stage}', reference)
                for record in records:
                    record['Supplement id'] = asset['blob_id']
                    record['Supplement label'] = label
                    record['Supplement filename'] = part.name
                    record['Supplement source URL'] = source_url
                first_material, written = _write_materials(records, first_material, output_path)
                summary['materials'] += written
                summary['supplement_materials'] += written
                members[key] = 'succeeded'
                # Persist each successful member to avoid repeated output on retries.
                set_supplement_extraction_status(conn, row['paper_id'], asset['blob_id'], stage,
                                                'partial', members=members)
                if delete_images_after:
                    for path in image_paths:
                        _delete_file(path)
            status = _supplement_status(members.values())
            error_text = '; '.join(errors)
            set_supplement_extraction_status(conn, row['paper_id'], asset['blob_id'], stage,
                                            status, error=error_text, members=members)
            outcomes[stage].append(status)
            if error_text:
                print(f'Supporting information {asset["blob_id"]} ({stage}): {error_text}', file=sys.stderr)
    for stage, statuses in outcomes.items():
        # Assign directly so successful SI does not clear an unrelated article error.
        row[f'supplement_{stage}_scrape_status'] = _supplement_status(statuses) if assets else 'not_found'
    return first_material


_OUTCOME_KEYS = tuple(f'{stage}_{outcome}' for stage in ('abstract', 'text', 'image')
                      for outcome in ('succeeded', 'failed')) + ('supplement_failed',)


def _paper_outcome(summary: Mapping[str, Any], before: Mapping[str, int]) -> str:
    """Name one paper's outcome from the summary counters it moved.

    Returns ``failed`` if any of its stages or SI files failed, ``succeeded``
    if a stage succeeded, and ``skipped`` when every stage was already done.
    """
    moved = {key for key in _OUTCOME_KEYS if summary[key] != before[key]}
    if any(key.endswith('_failed') for key in moved):
        return 'failed'
    return 'succeeded' if moved else 'skipped'


def _record_failure(summary: dict[str, Any], row: Mapping[str, Any], stage: str,
                    error: Exception) -> None:
    """Count a failed scrape stage and remember which paper it was.

    Parameters
    ----------
    summary : dict[str, Any]
        Run summary to update.
    row : Mapping[str, Any]
        Paper row whose stage failed.
    stage : str
        ``abstract``, ``text``, or ``image``.
    error : Exception
        The failure, recorded as text.
    """
    summary[f'{stage}_failed'] += 1
    if isinstance(error, FileNotFoundError):
        summary['missing_source'] += 1
    summary['failed_papers'].append({
        'paper_id': str(row.get('paper_id') or ''), 'stage': stage, 'error': str(error),
    })


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
        date. Missing publication dates sort last in either direction;
        papers with equal dates retain their corpus order.
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
    elif scrape_order in {'publication-asc', 'publication-desc'}:
        selected.sort(key=_publication_key, reverse=scrape_order == 'publication-desc')
        # A second stable sort keeps undated rows last without reversing ties.
        selected.sort(key=lambda paper: not bool(paper.get('publication_date')))
    elif scrape_order == 'title':
        selected.sort(key=lambda paper: str(paper.get('title') or '').lower())
    elif scrape_order == 'paper-id':
        selected.sort(key=lambda paper: str(paper.get('paper_id') or ''))
    if scrape_count is not None:
        return selected[:scrape_count]
    return selected


@dataclass(frozen=True, slots=True)
class _TextInput:
    """One paper's text prepared for extraction.

    Parameters
    ----------
    source_text : str
        Full source text, with tables, used by the checks.
    text : str
        Source text after any compression; image requests may use it as context.
    chunks : list[str]
        ``text`` split to fit the model's input budget.
    """

    source_text: str
    text: str
    chunks: list[str]


@dataclass(frozen=True, slots=True)
class _TextResult:
    """Records extracted from one paper's text.

    Parameters
    ----------
    text_input : _TextInput
        The prepared input the records came from.
    materials : list[dict[str, Any]]
        Merged and, when requested, checked records.
    removed : list[dict[str, str]]
        Audit rows for items the paper question or checks removed, edited or moved.
    error : Exception or None
        The failure that stopped extraction early, if any. ``materials`` then
        holds the records gathered before it.
    """

    text_input: _TextInput
    materials: list[dict[str, Any]]
    removed: list[dict[str, str]]
    error: Exception | None = None


def _extract_text_records(
    text_input: _TextInput,
    recipe_data: Mapping[str, Any],
    text_config: ModelConfig,
    paper_id: str,
    *,
    verify: bool,
    ask_question: bool,
) -> _TextResult:
    """Run the model requests for one paper's text and post-process the records.

    This touches no database state, so it can run on a worker thread.

    Parameters
    ----------
    text_input : _TextInput
        Prepared text chunks and source text.
    recipe_data : Mapping[str, Any]
        Loaded recipe.
    text_config : ModelConfig
        Text model configuration.
    paper_id : str
        Paper identifier recorded in audit rows.
    verify : bool
        Whether to apply the recipe's checks.
    ask_question : bool
        Whether the recipe's paper question may be asked (full-text scrapes only).

    Returns
    -------
    _TextResult
        Records and audit rows for the paper. A failure part-way through is
        returned in ``error`` with the records gathered before it.
    """
    materials: list[dict[str, Any]] = []
    removed: list[dict[str, str]] = []
    try:
        for text_chunk in text_input.chunks:
            materials.extend(scrape_text(text_chunk, recipe_data, model_config=text_config))
        materials = merge_chunk_records(materials, recipe_data)
        question = (recipe_data.get('checks') or {}).get('paper question')
        if verify and question and ask_question and materials:
            answer = ask_paper_question(text_input.chunks[0], question['question'], model_config=text_config)
            materials, moved = apply_paper_answer(materials, recipe_data, answer, paper_id)
            removed.extend(moved)
        if verify:
            materials, dropped = verify_records(materials, recipe_data, text_input.source_text, paper_id)
            removed.extend(dropped)
    except Exception as error:
        return _TextResult(text_input, materials, removed, error)
    return _TextResult(text_input, materials, removed)


class _TextPipeline:
    """Prepare papers in order and run their text requests on a thread pool.

    Preparation (reading assets from the corpus) stays on the calling thread,
    which owns the database connection, and runs a few papers ahead of the
    paper being written so the model server always has requests queued.

    Parameters
    ----------
    papers : Sequence[dict[str, Any]]
        Papers in scrape order.
    workers : int
        Concurrent model requests.
    asset_paths : Callable
        ``row -> paths`` materializing a paper's assets.
    needs_text : Callable
        ``row -> bool`` deciding whether the paper's text stage runs.
    prepare : Callable
        ``(row, paths) -> _TextInput``.
    extract : Callable
        ``(row, _TextInput) -> _TextResult``, run on a worker thread.
    """

    def __init__(
        self,
        papers: Sequence[dict[str, Any]],
        workers: int,
        asset_paths: Callable[[dict[str, Any]], dict[str, str | None]],
        needs_text: Callable[[dict[str, Any]], bool],
        prepare: Callable[[dict[str, Any], dict[str, str | None]], _TextInput],
        extract: Callable[[dict[str, Any], _TextInput], _TextResult],
    ) -> None:
        """Start the worker pool; nothing is prepared until :meth:`take`."""
        self._papers = papers
        self._ahead = 2 * workers
        self._asset_paths = asset_paths
        self._needs_text = needs_text
        self._prepare = prepare
        self._extract = extract
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='pmt-scrape')
        self._jobs: dict[int, tuple[dict[str, str | None], Future[_TextResult] | None]] = {}
        self._submitted = 0

    def __enter__(self) -> _TextPipeline:
        """Return the pipeline for use in a ``with`` block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Cancel queued requests and wait for running ones to finish."""
        self._executor.shutdown(wait=True, cancel_futures=True)

    def take(self, index: int) -> tuple[dict[str, str | None], Future[_TextResult] | None]:
        """Return a paper's asset paths and pending text result.

        Parameters
        ----------
        index : int
            Position of the paper in scrape order. Papers must be taken in order.

        Returns
        -------
        tuple
            Asset paths, and the future text result or ``None`` when the
            paper's text stage is skipped.
        """
        while self._submitted < len(self._papers) and self._submitted <= index + self._ahead:
            self._submit(self._submitted)
            self._submitted += 1
        return self._jobs.pop(index)

    def _submit(self, index: int) -> None:
        """Prepare one paper and queue its text requests, recording any preparation error."""
        row = self._papers[index]
        paths = self._asset_paths(row)
        future: Future[_TextResult] | None = None
        if self._needs_text(row):
            try:
                text_input = self._prepare(row, paths)
            except Exception as error:
                future = Future()
                future.set_exception(error)
            else:
                future = self._executor.submit(self._extract, row, text_input)
        self._jobs[index] = (paths, future)


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
                  include_supplements: bool = True,
                  paper_ids: Iterable[str] | None = None,
                  tables: bool = True,
                  verify: bool = False,
                  workers: int = 1,
                  figure_text: bool = True,
                  ) -> dict[str, Any]:
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
    include_supplements : bool, default=True
        Whether text and image modes also process downloaded supporting
        information. SI has independent checkpoints and per-file provenance;
        abstract-only scraping never includes SI.
    paper_ids : Iterable[str] or None, optional
        Only consider these canonical paper identifiers. Active corpus
        filters do not apply to papers named this way; ``scrape_order`` and
        ``scrape_count`` still do. ``None`` considers the whole corpus.
    tables : bool, optional
        Whether full-text scrapes append the paper's tables, rendered as
        Markdown from its structured XML or detected in its PDF.
    verify : bool, optional
        Whether to apply the recipe's ``"checks"`` to each paper's records.
        Removed and edited items are listed in ``<output>.dropped.csv``.
    workers : int, optional
        Papers whose text requests run concurrently. A model server such as
        vLLM batches concurrent requests, so values above 1 make abstract and
        text scrapes several times faster. Output order is unchanged. Image
        modes run one request at a time.
    figure_text : bool, optional
        Whether full text read from a PDF keeps the labels and numbers printed
        inside its figures. ``False`` redacts them, so values shown only in a
        figure are neither read by the model nor accepted by the checks.

    Returns
    -------
    dict[str, Any]
        Run summary. Besides the progress counters printed at the end, it
        holds ``papers_selected``, per-stage ``*_attempted``, ``*_skipped``,
        ``*_succeeded`` and ``*_failed`` counts, ``missing_source`` (failed
        stages with no downloaded asset to read), ``failed_papers`` (one
        ``paper_id``, ``stage`` and ``error`` entry per failed stage),
        ``output_path`` and ``elapsed_seconds``.

    Raises
    ------
    ValueError
        If a scrape, image, ordering, count, or compression option is invalid.
    """
    started = time.monotonic()
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
    if workers < 1:
        raise ValueError('workers must be a positive integer')
    if workers > 1 and mode in {'images', 'text-images'}:
        raise ValueError('workers above 1 are only supported for abstract and text scrapes')

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
    summary: dict[str, Any] = {
        'mode': mode,
        'output_path': str(output_path),
        'papers_selected': 0,
        'papers': 0,
        'abstract_attempted': 0,
        'abstract_skipped': 0,
        'abstract_succeeded': 0,
        'abstract_failed': 0,
        'text_attempted': 0,
        'text_skipped': 0,
        'text_succeeded': 0,
        'text_failed': 0,
        'image_attempted': 0,
        'image_skipped': 0,
        'image_succeeded': 0,
        'image_failed': 0,
        'missing_source': 0,
        'failed_papers': [],
        'layout_figures': 0,
        'materials': 0,
        'chunked_inputs': 0,
        'chunk_requests': 0,
        'abstract_chunked': 0,
        'text_chunked': 0,
        'supplement_text_attempted': 0,
        'supplement_image_attempted': 0,
        'supplement_skipped': 0,
        'supplement_failed': 0,
        'supplement_unsupported': 0,
        'supplement_materials': 0,
        'dropped': 0,
    }
    with connect(db_path) as conn:
        filters = active_filter_stack(conn)
        papers = paper_rows(conn)
        if paper_ids is not None:
            selected_ids = set(paper_ids)
            papers = [paper for paper in papers if paper['paper_id'] in selected_ids]
            print(f'Scraping {len(papers)} named papers; active corpus filters do not apply to them.')
        elif filters:
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
        summary['papers_selected'] = len(papers)
        progress_events.emit('run_start', total=len(papers), mode=mode)
        text_stage = 'abstract' if should_scrape_abstract else 'text'

        def needs_text(paper: dict[str, Any]) -> bool:
            """Return whether the paper's abstract or text stage should run."""
            return (should_scrape_abstract or should_scrape_text) and (
                force or paper.get(f'{text_stage}_scrape_status') != 'succeeded'
            )

        def prepare_text(paper: dict[str, Any], paper_paths: dict[str, str | None]) -> _TextInput:
            """Read, compress and chunk one paper's text on the database thread."""
            source = paper_source_text(conn, paper, paper_paths, abstract=should_scrape_abstract, tables=tables,
                                        figure_text=figure_text)
            prompt = build_scrape_prompt(recipe_data, source='text')
            compressed = maybe_compress_text(source, prompt, text_config, compression)
            return _TextInput(source, compressed, _text_chunks(compressed, text_config, prompt=prompt))

        def extract_text(paper: dict[str, Any], text_input: _TextInput) -> _TextResult:
            """Run one paper's model requests and checks."""
            return _extract_text_records(text_input, recipe_data, text_config, str(paper['paper_id']),
                                         verify=verify, ask_question=should_scrape_text)

        with tempfile.TemporaryDirectory(prefix='paperminertoolkit-scrape-') as temp_dir:
            pipeline = _TextPipeline(
                papers, workers, lambda paper: _paper_asset_paths(conn, paper, temp_dir),
                needs_text, prepare_text, extract_text,
            ) if workers > 1 else None
            with tqdm(total=len(papers), desc='Scraping Papers', colour='green',
                      disable=bars_disabled()) as pbar, pipeline or nullcontext():
                for done, row in enumerate(papers, start=1):
                    before = {key: summary[key] for key in _OUTCOME_KEYS}
                    if pipeline:
                        paths, text_job = pipeline.take(done - 1)
                    else:
                        paths, text_job = _paper_asset_paths(conn, row, temp_dir), None
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

                    if should_scrape_abstract or should_scrape_text:
                        stage = text_stage
                        status_column = f'{stage}_scrape_status'
                        if not needs_text(row):
                            summary[f'{stage}_skipped'] += 1
                        else:
                            summary[f'{stage}_attempted'] += 1
                            try:
                                if text_job is not None:
                                    result = text_job.result()
                                else:
                                    result = extract_text(row, prepare_text(row, paths))
                                if should_scrape_text:
                                    text_source_path = 'corpus:text' if text_path else 'corpus:pdf'
                                text = result.text_input.text
                                _record_chunk_plan(row, stage, result.text_input.chunks, text_config, summary)
                                text_materials = result.materials
                                summary['dropped'] += _write_dropped(result.removed, output_path)
                                if result.error is not None:
                                    raise result.error
                                row[f'num_{stage}_materials'] = len(text_materials)
                                _set_status(row, status_column, 'succeeded')
                                summary[f'{stage}_succeeded'] += 1
                            except Exception as e:
                                _set_status(row, status_column, 'failed', str(e))
                                _record_failure(summary, row, stage, e)

                    if should_scrape_images:
                        image_paths = []
                        if not force and row.get('image_scrape_status') == 'succeeded':
                            summary['image_skipped'] += 1
                        else:
                            summary['image_attempted'] += 1
                            previous_images = 0
                            previous_materials = 0
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
                                    if not force:
                                        previous_images = int(row.get('num_images') or 0)
                                        previous_materials = int(row.get('num_image_materials') or 0)
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
                                        image_materials.extend(_append_materials(
                                            response, row, 'image', ';'.join(batch_paths), figure_batch,
                                        ))
                                        row['num_images'] = previous_images + len(image_source_paths)
                                        row['num_image_materials'] = previous_materials + len(image_materials)
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
                                row['num_images'] = previous_images + len(image_source_paths)
                                row['num_image_materials'] = previous_materials + len(image_materials)
                                _set_status(row, 'image_scrape_status', 'succeeded')
                                summary['image_succeeded'] += 1
                                if delete_images_after:
                                    for image_path in image_paths:
                                        _delete_file(image_path)
                            except Exception as e:
                                _set_status(row, 'image_scrape_status', 'failed', str(e))
                                _record_failure(summary, row, 'image', e)

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
                            row_materials.extend(
                                image_materials if scraped_figures else _append_materials(
                                    image_materials, row, 'image', image_source,
                                )
                            )
                    elif text_materials:
                        if should_scrape_abstract:
                            row_materials.extend(_append_materials(text_materials, row, 'abstract', 'corpus:abstract'))
                        else:
                            row_materials.extend(_append_materials(text_materials, row, 'text', text_source_path))
                    elif image_materials:
                        row_materials.extend(
                            image_materials if scraped_figures else _append_materials(
                                image_materials, row, 'image', ';'.join(image_source_paths),
                            )
                        )

                    first_material, written_count = _write_materials(row_materials, first_material, output_path)
                    summary['materials'] += written_count
                    if include_supplements and mode != 'abstract':
                        first_material = _scrape_supplements(
                            conn, row, mode=mode, recipe=recipe_data,
                            text_config=text_config, vision_config=vision_config,
                            compression=compression, image_dir=image_dir, image_dpi=image_dpi,
                            image_batch_size=image_batch_size, image_context=image_context,
                            delete_images_after=delete_images_after, force=force,
                            output_path=output_path, first_material=first_material, summary=summary,
                        )
                    upsert_paper(conn, row)
                    pbar.update(1)
                    progress_events.emit('paper_done', paper_id=row['paper_id'], done=done,
                                         total=len(papers), status=_paper_outcome(summary, before),
                                         materials=len(row_materials))

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
    if summary['dropped']:
        print(f"Checks removed, edited or moved {summary['dropped']} items; see {dropped_path(output_path)}.")
    if summary['materials'] == 0:
        print(f"No new scraped material rows were written to {output_path}.")
    if any(summary[key] for key in ('supplement_text_attempted', 'supplement_image_attempted', 'supplement_skipped')):
        print(
            f"Supporting information: text stages={summary['supplement_text_attempted']}, "
            f"image stages={summary['supplement_image_attempted']}, "
            f"completed stages/members skipped={summary['supplement_skipped']}, "
            f"unsupported file stages={summary['supplement_unsupported']}, "
            f"failed file stages={summary['supplement_failed']}, "
            f"material rows={summary['supplement_materials']}."
        )
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
    summary['elapsed_seconds'] = round(time.monotonic() - started, 3)
    progress_events.emit('run_end', papers=summary['papers'], materials=summary['materials'],
                         failed=len(summary['failed_papers']))
    return summary
