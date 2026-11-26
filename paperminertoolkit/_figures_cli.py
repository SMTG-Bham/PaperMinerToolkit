"""The ``pmt figures`` commands: get, inspect, and export paper figures.

A paper's figures reach the corpus by one of two routes, and each paper keeps
figures from one origin so that image scraping and digitization never see a
figure twice:

- ``figures download`` fetches the publisher's figure images referenced by
  stored structured documents (JATS, Elsevier XML, GROBID TEI). They replace
  any figures previously detected from the paper's PDF.
- ``figures detect`` detects figures from a stored PDF and stores the crops,
  for papers that have no structured figures.

``figures list`` and ``figures export`` read what is stored, and ``figures
render`` runs the detection (or the embedded-image and page strategies) on any
PDF file without a corpus.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

import click

from paperminertoolkit import _output_schemas as out
from paperminertoolkit import _progress as progress_events
from paperminertoolkit._agent import agent_meta, emit_json
from paperminertoolkit._cli_common import (_json_option, _paper_id_options, _progress_option, _quiet,
                                          _selected_paper_ids)
from paperminertoolkit._progress import progress_mode
from paperminertoolkit.corpus import database
from paperminertoolkit.corpus.documents import extract_pdf_images
from paperminertoolkit.corpus.filtering import active_filter_stack, current_filter_statuses
from paperminertoolkit.corpus.pdf_layout import detect_pdf_layout, render_pdf_figures
from paperminertoolkit.workflows.figures import (DETECT_STATUSES, PDF_LAYOUT_SOURCE,
                                                 detect_paper_figures, download_paper_figures)

_EXTENSIONS = {'image/png': 'png', 'image/jpeg': 'jpg', 'image/gif': 'gif', 'image/tiff': 'tif',
               'image/webp': 'webp', 'image/bmp': 'bmp', 'image/svg+xml': 'svg'}

_ignore_filters_option = click.option(
    '--ignore-filters', is_flag=True, default=False,
    help='Process papers that active corpus filters exclude. Named papers (--paper-id) always '
         'bypass filters.',
)


def _safe(value: str) -> str:
    """Turn an identifier into a safe file or directory name."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._') or 'item'


def _selection(conn: sqlite3.Connection, candidates: list[str], named: list[str] | None,
               ignore_filters: bool) -> tuple[list[str], int]:
    """Apply --paper-id or the active filters to candidate papers, as scrape does.

    Returns the selected identifiers in corpus order and how many candidates
    the filters excluded.
    """
    if named is not None:
        wanted = set(named)
        return [paper_id for paper_id in candidates if paper_id in wanted], 0
    if ignore_filters or not active_filter_stack(conn):
        return candidates, 0
    statuses = current_filter_statuses(conn)
    selected = [paper_id for paper_id in candidates if statuses.get(paper_id) == 'included']
    return selected, len(candidates) - len(selected)


def _paper_rows(conn: sqlite3.Connection, paper_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Load full paper rows for the given identifiers, in the given order."""
    wanted = list(paper_ids)
    lookup = set(wanted)
    rows = {row['paper_id']: row for row in database.paper_rows(conn) if row['paper_id'] in lookup}
    return [rows[paper_id] for paper_id in wanted if paper_id in rows]


def _figure_record(asset: dict[str, Any]) -> dict[str, Any]:
    """Describe one stored figure asset without its image bytes."""
    metadata = asset.get('metadata') or {}
    return {
        'paper_id': asset['paper_id'],
        'figure_id': str(metadata.get('figure_id') or ''),
        'label': str(metadata.get('figure_label') or metadata.get('label') or ''),
        'caption': str(metadata.get('caption') or ''),
        'source': str(asset.get('source') or ''),
        'source_url': str(metadata.get('source_url') or ''),
        'mime_type': str(asset.get('mime_type') or ''),
        'bytes': int(asset.get('original_size') or 0),
        'blob_id': str(asset.get('blob_id') or ''),
        'page_numbers': list(metadata.get('page_numbers') or []),
        'region_detected': metadata.get('region_detected'),
        'extraction_status': metadata.get('extraction_status'),
        'digitization_status': metadata.get('digitization_status'),
    }


@agent_meta(effects={'network', 'writes_corpus'}, output=out.FIGURES_DOWNLOAD)
@click.command('download')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True, dir_okay=False))
@_paper_id_options
@_ignore_filters_option
@click.option('--force', is_flag=True, default=False,
              help='Download figures again even when already stored.')
@_progress_option
@_json_option
def figures_download(db_path: str, paper_ids: tuple[str, ...], paper_ids_file: str | None,
                     ignore_filters: bool, force: bool, progress_display: str,
                     json_output: bool) -> None:
    """Download the figure images that papers' structured documents reference.

    Only papers with a stored structured document (JATS, Elsevier XML, or
    GROBID TEI, saved by pmt download) have figure references. Each image is
    checked to be an image and stored with its label and caption; a rerun
    skips images already stored. A paper's figures detected from its PDF are
    removed once its structured figures arrive, so each figure is kept once.
    """
    named = _selected_paper_ids(db_path, paper_ids, paper_ids_file)
    try:
        with closing(database.connect(db_path)) as conn:
            candidates = database.papers_with_role(conn, database.STRUCTURED_DOCUMENT_ROLE)
            selected, filtered = _selection(conn, candidates, named, ignore_filters)
        totals = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'pdf_layout_replaced': 0}
        failures = []
        with _quiet(json_output), progress_mode(progress_display, 'figures download'):
            progress_events.emit('run_start', total=len(selected))
            for done, paper_id in enumerate(selected, start=1):
                result = download_paper_figures(db_path, paper_id, force=force)
                for key in totals:
                    totals[key] += result[key]
                failures.extend({'paper_id': paper_id, 'message': message}
                                for message in result['failures'])
                progress_events.emit('paper_done', paper_id=paper_id, done=done,
                                     total=len(selected), downloaded=result['downloaded'],
                                     failed=result['failed'])
            progress_events.emit('run_end', papers=len(selected), **totals)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    summary = {'papers': len(selected), 'papers_without_structured_documents': (
        None if named is None else len(named) - len(selected)),
        'papers_filtered_out': filtered, **totals, 'failures': failures}
    if json_output:
        emit_json(summary)
        return
    click.echo(f'Figures from structured documents of {len(selected)} papers: '
               f'{totals["downloaded"]} downloaded, {totals["skipped"]} already stored, '
               f'{totals["failed"]} failed; {totals["pdf_layout_replaced"]} PDF-detected figures '
               'replaced.')
    for failure in failures[:10]:
        click.echo(f'  {failure["paper_id"]}: {failure["message"]}', err=True)


@agent_meta(effects={'writes_corpus'}, output=out.FIGURES_DETECT)
@click.command('detect')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True, dir_okay=False))
@_paper_id_options
@_ignore_filters_option
@click.option('--force', is_flag=True, default=False,
              help='Replace figures already detected from the PDF.')
@click.option('--dpi', default=200, type=click.IntRange(min=36, max=1200), show_default=True,
              help='Render resolution of the figure crops.')
@click.option('--padding', default=12.0, type=click.FloatRange(min=0), show_default=True,
              help='PDF points of margin around each detected region.')
@click.option('--min-confidence', 'minimum_confidence', default=0.2,
              type=click.FloatRange(0, 1), show_default=True,
              help='Minimum caption-to-artwork association score; weaker matches render the '
                   'whole page.')
@_progress_option
@_json_option
def figures_detect(db_path: str, paper_ids: tuple[str, ...], paper_ids_file: str | None,
                   ignore_filters: bool, force: bool, dpi: int, padding: float,
                   minimum_confidence: float, progress_display: str, json_output: bool) -> None:
    """Detect figures in papers' stored PDFs and store the crops in the corpus.

    Captions (Figure, Fig., Table) are paired with nearby images or vector
    drawings and the regions rendered; a figure without a confident region
    is stored as its whole page. Papers that already have structured figures
    are skipped, as are papers already detected unless --force. This is what
    pmt scrape --image-extraction layout and pmt digitize do on demand.
    """
    named = _selected_paper_ids(db_path, paper_ids, paper_ids_file)
    statuses = dict.fromkeys(DETECT_STATUSES, 0)
    stored = replaced = 0
    failures = []
    try:
        with closing(database.connect(db_path)) as conn:
            candidates = [row['paper_id'] for row in database.paper_rows(conn)]
            selected, filtered = _selection(conn, candidates, named, ignore_filters)
            papers = _paper_rows(conn, selected)
            with _quiet(json_output), progress_mode(progress_display, 'figures detect'):
                progress_events.emit('run_start', total=len(papers))
                for done, paper in enumerate(papers, start=1):
                    result = detect_paper_figures(conn, paper, force=force, padding=padding, dpi=dpi,
                                                  minimum_confidence=minimum_confidence)
                    statuses[result['status']] += 1
                    stored += result['stored']
                    replaced += result['replaced']
                    failures.extend({'paper_id': paper['paper_id'], 'message': message}
                                    for message in result['failures'])
                    progress_events.emit('paper_done', paper_id=paper['paper_id'], done=done,
                                         total=len(papers), status=result['status'],
                                         stored=result['stored'])
                progress_events.emit('run_end', papers=len(papers), figures_stored=stored)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    summary = {'papers': len(papers), 'papers_filtered_out': filtered, 'statuses': statuses,
               'figures_stored': stored, 'figures_replaced': replaced, 'failures': failures}
    if json_output:
        emit_json(summary)
        return
    click.echo(f'Detected figures in {statuses["detected"]} of {len(papers)} papers: {stored} '
               f'figures stored, {replaced} replaced.')
    click.echo(f'Skipped: {statuses["skipped_structured"]} with structured figures, '
               f'{statuses["skipped_existing"]} already detected, {statuses["no_pdf"]} without a '
               f'PDF; {statuses["none_found"]} had no figure captions; {statuses["failed"]} failed.')
    for failure in failures[:10]:
        click.echo(f'  {failure["paper_id"]}: {failure["message"]}', err=True)


def _stored_figures(conn: sqlite3.Connection, named: list[str] | None, source: str | None,
                    figure_id: str | None = None,
                    include_content: bool = False) -> list[dict[str, Any]]:
    """Load the stored figure assets of the selected papers, in corpus and figure order."""
    papers = database.papers_with_role(conn, database.FIGURE_ASSET_ROLE)
    if named is not None:
        wanted = set(named)
        papers = [paper_id for paper_id in papers if paper_id in wanted]
    assets = []
    for paper_id in papers:
        found = database.get_figure_assets(conn, paper_id, figure_id=figure_id,
                                           include_content=include_content)
        assets.extend(asset for asset in reversed(found)
                      if source is None or asset.get('source') == source)
    return assets


@agent_meta(effects={'read_only'}, output=out.FIGURES_LIST)
@click.command('list')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True, dir_okay=False))
@_paper_id_options
@click.option('--source', default=None,
              help=f'Only figures from this source, such as {PDF_LAYOUT_SOURCE} or a provider name.')
@click.option('--limit', default=100, type=click.IntRange(1, 10000), show_default=True,
              help='Maximum number of figures to return.')
@click.option('--offset', default=0, type=click.IntRange(min=0), show_default=True,
              help='Number of figures to skip.')
@_json_option
def figures_list(db_path: str, paper_ids: tuple[str, ...], paper_ids_file: str | None,
                 source: str | None, limit: int, offset: int, json_output: bool) -> None:
    """List the figures stored in a corpus, with captions and progress.

    Each figure shows its source, whether a region was found (PDF figures),
    and its image-scrape and digitization status. The corpus is opened
    read-only and image bytes are not loaded.
    """
    named = _selected_paper_ids(db_path, paper_ids, paper_ids_file)
    try:
        with database.connect_read_only(db_path) as conn:
            records = [_figure_record(asset) for asset in _stored_figures(conn, named, source)]
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    page = records[offset:offset + limit]
    if json_output:
        emit_json({'figures': page, 'total': len(records), 'limit': limit, 'offset': offset})
        return
    click.echo(f'{len(records)} figures; showing {len(page)} from offset {offset}.')
    for record in page:
        region = {True: 'region', False: 'whole page', None: ''}.get(record['region_detected'], '')
        status = record['extraction_status'] or 'not scraped'
        click.echo(f'{record["paper_id"]} {record["figure_id"]} [{record["source"]}'
                   f'{", " + region if region else ""}; {status}] {record["label"]}: '
                   f'{record["caption"][:80]}')


@agent_meta(effects={'writes_files'}, output=out.FIGURES_EXPORT)
@click.command('export')
@click.argument('db_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('outdir', type=click.Path(file_okay=False))
@_paper_id_options
@click.option('--figure-id', default=None, help='Only the figure with this identifier.')
@click.option('--source', default=None, help='Only figures from this source.')
@click.option('--force', is_flag=True, default=False, help='Overwrite existing files.')
@_json_option
def figures_export(db_path: str, outdir: str, paper_ids: tuple[str, ...],
                   paper_ids_file: str | None, figure_id: str | None, source: str | None,
                   force: bool, json_output: bool) -> None:
    """Write stored figure images to OUTDIR/<paper>/<figure>.<ext>.

    The corpus is opened read-only. Existing files are refused, and nothing
    is written, unless --force is given.
    """
    named = _selected_paper_ids(db_path, paper_ids, paper_ids_file)
    try:
        with database.connect_read_only(db_path) as conn:
            assets = _stored_figures(conn, named, source, figure_id=figure_id, include_content=True)
        planned = []
        for asset in assets:
            record = _figure_record(asset)
            extension = _EXTENSIONS.get(record['mime_type'], 'img')
            name = f'{_safe(record["figure_id"])}.{extension}'
            target = Path(outdir) / _safe(record['paper_id']) / name
            planned.append((record, target, asset['content']))
        existing = [str(target) for _, target, _ in planned if target.exists()]
        if existing and not force:
            raise click.ClickException(f'{len(existing)} output files already exist, such as '
                                       f'{existing[0]}. Use --force to overwrite.')
        files = []
        for record, target, content in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            files.append({'paper_id': record['paper_id'], 'figure_id': record['figure_id'],
                          'label': record['label'], 'source': record['source'],
                          'path': str(target), 'bytes': len(content)})
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json({'outdir': outdir, 'count': len(files), 'files': files})
        return
    click.echo(f'Exported {len(files)} figures to {outdir}.')


@agent_meta(effects={'writes_files'}, output=out.FIGURES_RENDER)
@click.command('render')
@click.argument('pdf_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('outdir', type=click.Path(file_okay=False))
@click.option('--strategy', type=click.Choice(['layout', 'embedded', 'pages']), default='layout',
              show_default=True,
              help='layout: detected figure regions; embedded: every raster image in the PDF; '
                   'pages: one image per page.')
@click.option('--dpi', default=200, type=click.IntRange(min=36, max=1200), show_default=True,
              help='Render resolution.')
@click.option('--padding', default=12.0, type=click.FloatRange(min=0), show_default=True,
              help='PDF points of margin around each detected region (layout).')
@click.option('--min-confidence', 'minimum_confidence', default=0.2,
              type=click.FloatRange(0, 1), show_default=True,
              help='Minimum caption-to-artwork association score (layout).')
@click.option('--prefix', default=None, help='File name prefix; defaults to the PDF name.')
@_json_option
def figures_render(pdf_path: str, outdir: str, strategy: str, dpi: int, padding: float,
                   minimum_confidence: float, prefix: str | None, json_output: bool) -> None:
    """Extract figure images from a PDF file into OUTDIR, without a corpus.

    The strategies are the image sources of pmt scrape --image-extraction:
    layout detects captioned figures and renders their regions (whole pages
    when uncertain), embedded saves the PDF's raster images, and pages
    renders every page.
    """
    try:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        with _quiet(json_output):
            if strategy == 'layout':
                layout = detect_pdf_layout(pdf_path, minimum_confidence=minimum_confidence)
                paths = render_pdf_figures(pdf_path, layout, outdir, prefix=prefix, padding=padding,
                                           dpi=dpi)
                images = [{'path': path, 'figure_id': figure.identifier, 'label': figure.label,
                           'caption': figure.caption, 'page_numbers': list(figure.page_numbers),
                           'region_detected': bool(figure.boxes)}
                          for figure, path in zip(layout.figures, paths)]
                tables = len(layout.tables)
            else:
                images = [{'path': path} for path in extract_pdf_images(
                    pdf_path, outdir, prefix=prefix, strategy=strategy, dpi=dpi)]
                tables = None
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json({'pdf': pdf_path, 'strategy': strategy, 'outdir': outdir, 'count': len(images),
                   'images': images, 'tables_detected': tables})
        return
    click.echo(f'Wrote {len(images)} images to {outdir} ({strategy}).')
    for image in images:
        detail = ''
        if strategy == 'layout':
            detail = f' {image["label"]}' + ('' if image['region_detected'] else ' (whole page)')
        click.echo(f'  {image["path"]}{detail}')


@click.group('figures')
def figures_group() -> None:
    """Download, detect, list, export, and render paper figures."""


figures_group.add_command(figures_download)
figures_group.add_command(figures_detect)
figures_group.add_command(figures_list)
figures_group.add_command(figures_export)
figures_group.add_command(figures_render)
