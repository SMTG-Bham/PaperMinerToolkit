"""The nested ``pmt`` command-line interface for PaperMinerToolkit workflows.

This module maps installed commands to the underlying search, import,
download, scrape, store, configuration, and maintenance functions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import os
import sys
from contextlib import closing, nullcontext, redirect_stdout
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

import click
from paperminertoolkit import _output_schemas as out
from paperminertoolkit._agent import AgentGroup, agent_meta, emit_json
from paperminertoolkit._progress import PROGRESS_MODES, progress_mode
from paperminertoolkit._cli_schema import schema_command
from paperminertoolkit._corpus_cli import corpus_export, corpus_list, corpus_show
from paperminertoolkit._version import __version__
from paperminertoolkit.providers import registry as sources
from paperminertoolkit.corpus.database import (connect,
                                               corpus_copy,
                                               corpus_stats,
                                               enrichment_stats,
                                               search_history)
from paperminertoolkit.providers.crossref import import_author_works
from paperminertoolkit.workflows.search import search_for_papers
from paperminertoolkit.workflows.gather import gather_papers
from paperminertoolkit.probe_cli import probe, report_provider_status
from paperminertoolkit.extraction.compression import COMPRESSION_MODES, COMPRESSION_SCOPES
from paperminertoolkit.extraction.entities import EntityExtractionConfig
from paperminertoolkit.extraction.extract import (build_image_extraction_prompt,
                                                  build_reconciliation_prompt,
                                                  build_text_extraction_prompt)
from paperminertoolkit.extraction.recipes import RECIPES_PATH, load_recipe
from paperminertoolkit.workflows.download import DOWNLOAD_FORMATS, download_papers, plan_downloads
from paperminertoolkit.workflows.supplements import SUPPLEMENT_SOURCES
from paperminertoolkit.workflows.enrichment import enrich_corpus
from paperminertoolkit.workflows.entities import extract_corpus_entities, extract_file_entities
from paperminertoolkit.corpus.filtering import (apply_regex_filter,
                                    apply_topic_filter,
                                    filter_overview,
                                    reset_filters)
from paperminertoolkit.workflows.imports import import_pdfs
from paperminertoolkit.extraction.scrape import SCRAPE_ORDERS, scrape_papers
from paperminertoolkit.extraction.scrape_plan import plan_scrape
from paperminertoolkit.extraction.scrape_progress import STAGES as SCRAPE_PROGRESS_MODES
from paperminertoolkit.extraction.scrape_progress import report as scrape_progress_report
from paperminertoolkit.extraction.scrape_progress import scrape_progress
from paperminertoolkit.extraction.store import store_results
from paperminertoolkit.workflows.digitize import digitize_figures
from paperminertoolkit.workflows.validation import ReviewApp, review_session, saved_review
from paperminertoolkit.workflows.validation import serve as serve_validation
from paperminertoolkit.workflows.validation_scoring import score_review
from paperminertoolkit.workflows.topics import (aggregate_topic_trends,
                                 compare_topic_models,
                                 predict_topic_model,
                                 set_topic_name,
                                 store_topic_model_scores,
                                 stored_topic_models,
                                 topic_descriptions,
                                 train_topic_model)
from paperminertoolkit.settings import (get_model_profile,
                                   DEFAULT_INPUT_TOKEN_LIMIT,
                                   _saved_settings,
                                   infer_model_capabilities,
                                   set_model_profile,
                                   update_anthropic_key,
                                   update_core_key,
                                   update_core_rate,
                                   update_crossref_email,
                                   update_elsevier_key,
                                   update_ncbi_email,
                                   update_ncbi_key,
                                   update_openai_key,
                                   update_openalex_key,
                                   update_unpaywall_email)
from paperminertoolkit.workflows.utilities import progress_summary, reset, status


_json_option = click.option(
    '--json', 'json_output', is_flag=True,
    help='Write JSON to stdout and progress or errors to stderr.',
)


def _quiet(json_output: bool) -> Any:
    """Route workflow chatter to stderr while a command produces JSON."""
    return redirect_stdout(sys.stderr) if json_output else nullcontext()


_progress_option = click.option(
    '--progress', 'progress_display', type=click.Choice(PROGRESS_MODES), default='auto',
    show_default=True,
    help='auto shows progress bars; none hides them; jsonl hides them and writes one JSON '
         'event per line to stderr.',
)

_dry_run_option = click.option(
    '--dry-run', is_flag=True, default=False,
    help='Report what the command would change, without changing the corpus or any file.',
)


def _on_copy(db_path: str, operation: Callable[[str], Mapping[str, Any]]) -> dict[str, Any]:
    """Run a corpus-changing operation on a temporary copy and report its result."""
    with corpus_copy(db_path) as copy:
        return {**operation(str(copy)), 'dry_run': True}


def _paper_id_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Add the shared ``--paper-id`` and ``--paper-ids-file`` selection options."""
    command = click.option(
        '--paper-ids-file', default=None, type=click.Path(exists=True, dir_okay=False),
        help='File of corpus paper IDs, one per line (blank lines and # comments ignored); '
             'combined with --paper-id.',
    )(command)
    return click.option(
        '--paper-id', 'paper_ids', multiple=True,
        help='Only process this corpus paper ID. Repeat for several. Active corpus filters '
             'do not apply to named papers.',
    )(command)


# SQLite limits the number of bound parameters per statement.
_ID_LOOKUP_BATCH = 500


def _selected_paper_ids(db_path: str, paper_ids: tuple[str, ...],
                        ids_file: str | None) -> list[str] | None:
    """Resolve ``--paper-id`` and ``--paper-ids-file`` against the corpus.

    Returns ``None`` when neither option was given. Unknown identifiers are a
    usage error that names them; the corpus is opened read-only.
    """
    if not paper_ids and ids_file is None:
        return None
    requested = [paper_id.strip() for paper_id in paper_ids if paper_id.strip()]
    if ids_file is not None:
        try:
            lines = Path(ids_file).read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise click.BadParameter(str(error), param_hint='--paper-ids-file') from error
        requested.extend(line.strip() for line in lines
                         if line.strip() and not line.lstrip().startswith('#'))
    requested = list(dict.fromkeys(requested))
    if not requested:
        raise click.BadParameter('No paper IDs were given.', param_hint='--paper-id')
    known: set[str] = set()
    try:
        uri = f'{Path(db_path).resolve().as_uri()}?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            for start in range(0, len(requested), _ID_LOOKUP_BATCH):
                batch = requested[start:start + _ID_LOOKUP_BATCH]
                rows = conn.execute(
                    f'SELECT paper_id FROM papers WHERE paper_id IN ({", ".join("?" for _ in batch)})',
                    batch,
                ).fetchall()
                known.update(row[0] for row in rows)
    except sqlite3.Error as error:
        raise click.ClickException(f'Cannot look up paper IDs in {db_path}: {error}') from error
    missing = [paper_id for paper_id in requested if paper_id not in known]
    if missing:
        shown = ', '.join(missing[:20])
        more = f' and {len(missing) - 20} more' if len(missing) > 20 else ''
        raise click.BadParameter(f'Unknown paper ID(s) in {db_path}: {shown}{more}',
                                 param_hint='--paper-id')
    return requested

# A download may fetch a PDF, full text, or an abstract, so its choices are
# the union of the three, kept in PDF order because that is the one a caller
# most often means.
DOWNLOAD_CHOICES = [
    'all',
    *dict.fromkeys((*sources.names(sources.PDF),
                    *sources.names(sources.TEXT),
                    *sources.names(sources.ABSTRACT))),
]


def _format_bytes(size: int) -> str:
    """Format a byte count using compact binary units."""
    value = float(size)
    for unit in ['B', 'KiB', 'MiB', 'GiB']:
        if value < 1024 or unit == 'GiB':
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024


@agent_meta(effects={'network', 'writes_corpus'}, output=out.SEARCH)
@click.command('search')
@click.argument('query', default='Lithium solid electrolyte', type=str)
@click.argument('db_path', default='papers.db', type=click.Path(dir_okay=False))
@click.option('--source',
              type=click.Choice(sources.choices(sources.SEARCH)),
              multiple=True,
              default=('all',),
              show_default=True,
              help='Search source to use. Repeat to choose more than one.')
@click.option('--count',
              default=200,
              type=click.IntRange(min=1),
              show_default=True,
              help='Maximum results to request from each selected source.')
@click.option('--store-abstract',
              is_flag=True,
              default=False,
              help='Store abstracts returned by search providers as corpus assets.')
@click.option('--enrich', 'enrich_metadata',
              is_flag=True,
              default=False,
              help='Supplement newly stored papers with all enrichment providers.')
@click.option('--parallel', is_flag=True, default=False,
              help='Search selected providers concurrently.')
@click.option('--workers', default=None, type=click.IntRange(min=1),
              help='Maximum provider workers; also enables parallel search.')
@_json_option
def paper_search(query: str, db_path: str, source: tuple[str, ...], count: int,
                 store_abstract: bool, enrich_metadata: bool, parallel: bool,
                 workers: int | None, json_output: bool) -> None:
    """Search configured paper sources and merge results into the paper corpus."""
    if not query.strip():
        raise click.BadParameter('Search query must not be blank.', param_hint='QUERY')
    try:
        with redirect_stdout(sys.stderr) if json_output else nullcontext():
            summary = search_for_papers(
                query, db_path, source=source[0] if len(source) == 1 else source, count=count,
                store_abstract=store_abstract, enrich=enrich_metadata,
                parallel=parallel, workers=workers,
            )
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)
    if summary:
        _check_search_status(summary)


def _check_search_status(summary: Mapping[str, Any]) -> None:
    """Report incomplete searches after preserving their available results."""
    if summary['status'] in {'partial', 'failed'}:
        failed = [name for name, result in summary['source_results'].items()
                  if result['status'] == 'failed']
        raise click.ClickException(
            f'Search {summary["status"]}: {", ".join(failed)} failed. '
            'Available results have been saved; inspect pmt corpus searches for details.'
        )


@agent_meta(effects={'network', 'writes_corpus'}, output=out.GATHER)
@click.command('gather')
@click.argument('query')
@click.argument('db_path', default='papers.db', type=click.Path(dir_okay=False))
@click.option('--source', multiple=True,
              type=click.Choice(sources.choices(sources.SEARCH)),
              default=('all',), show_default=True,
              help='Search source to use. Repeat to choose more than one.')
@click.option('--count', default=200, type=click.IntRange(min=1), show_default=True,
              help='Maximum results to request from each selected search source.')
@click.option('--format', 'download_format',
              type=click.Choice(sorted(DOWNLOAD_FORMATS)),
              default='both', show_default=True,
              help='Paper assets to retrieve; both means text and PDF, supplement means SI only.')
@click.option('--download-source', 'download_sources', multiple=True,
              type=click.Choice(DOWNLOAD_CHOICES), default=('all',), show_default=True,
              help='Content provider to try. Repeat to choose more than one.')
@click.option('--abstract/--no-abstract', 'download_abstract',
              default=True, show_default=True,
              help='Store search abstracts and retrieve missing abstracts; ignored for supplement-only runs.')
@click.option('--supplements/--no-supplements', 'download_supplements',
              default=True, show_default=True,
              help='Retrieve supporting information alongside paper assets; the abstract/supplement formats take precedence.')
@click.option('--supplement-source', 'supplement_sources', multiple=True,
              type=click.Choice(['all', *SUPPLEMENT_SOURCES]), default=('all',), show_default=True,
              help='Supporting-information source to use. Repeat to choose more than one.')
@click.option('--enrich', 'enrich_metadata', is_flag=True,
              help='Supplement matching papers with configured enrichment providers.')
@click.option('--parallel', is_flag=True,
              help='Search selected providers concurrently.')
@click.option('--workers', default=None, type=click.IntRange(min=1),
              help='Maximum provider workers; also enables parallel search.')
@click.option('--force', is_flag=True,
              help='Redownload requested assets even when already stored.')
@_progress_option
@click.option('--json', 'json_output', is_flag=True,
              help='Write a JSON summary to stdout and progress to stderr.')
def gather(query: str, db_path: str, source: tuple[str, ...], count: int,
           download_format: str, download_sources: tuple[str, ...],
           download_abstract: bool, download_supplements: bool,
           supplement_sources: tuple[str, ...], enrich_metadata: bool, parallel: bool,
           workers: int | None, force: bool, progress_display: str, json_output: bool) -> None:
    """Search for papers, save metadata, and retrieve assets in one run.

    Only papers matched by this search are considered for download. Existing
    assets are reused unless --force is set. No language model is required.
    A failed search source gives exit status 1 after available results are saved.
    """
    if not query.strip():
        raise click.BadParameter('Search query must not be blank.', param_hint='QUERY')
    try:
        with redirect_stdout(sys.stderr) if json_output else nullcontext(), \
                progress_mode(progress_display, 'gather'):
            summary = gather_papers(
                query, db_path, source=source[0] if len(source) == 1 else source,
                count=count, download_format=download_format,
                download_sources=list(download_sources), download_abstract=download_abstract,
                download_supplements=download_supplements,
                supplement_sources=list(supplement_sources),
                enrich=enrich_metadata, parallel=parallel, workers=workers, force=force,
            )
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)
    else:
        matched = len(summary['search']['paper_ids'])
        click.echo(f'Gathered metadata for {matched} matching papers in {db_path}.')
        if not matched:
            click.echo('No matching papers to download.')
    _check_search_status(summary['search'])


@agent_meta(effects={'network', 'writes_corpus'}, output=out.IMPORT_PDFS)
@click.command('pdfs')
@click.argument('dir', default='papers', type=click.Path(exists=True, file_okay=False))
@click.argument('db_path', default='papers.db', type=click.Path())
@click.option('--no-crossref',
              is_flag=True,
              default=False,
              help='Only scrape DOI from PDFs; skip Crossref metadata lookup.')
@_json_option
def import_pdf_folder(dir: str, db_path: str, no_crossref: bool, json_output: bool) -> None:
    """Import local PDFs into the paper corpus, optionally skipping Crossref lookup."""
    try:
        with _quiet(json_output):
            summary = import_pdfs(dir, db_path, use_crossref=not no_crossref)
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)


@agent_meta(effects={'network', 'writes_corpus', 'writes_files'}, output=out.IMPORT_AUTHOR)
@click.command('author')
@click.argument('db_path', default='papers.db', type=click.Path())
@click.option('--email', default=None,
              help='Contact email sent with Crossref requests. '
                   'Defaults to the stored crossref_email setting.')
@click.option('--orcid', default=None, help='Exact ORCID identifier for the author.')
@click.option('--author', 'author_name', default=None, help='Author name fallback when no ORCID is available.')
@click.option('--affiliation', default=None, help='Require this affiliation on the matching author record.')
@click.option('--max-results', default=500, type=click.IntRange(min=1), show_default=True)
@click.option('--review-csv', default='author_works.csv', type=click.Path(), show_default=True,
              help='CSV summary written for manual review.')
@click.option('--enrich', 'enrich_metadata',
              is_flag=True,
              default=False,
              help='Supplement imported works with all enrichment providers.')
@_json_option
def import_author(db_path: str,
                  email: str | None,
                  orcid: str | None,
                  author_name: str | None,
                  affiliation: str | None,
                  max_results: int,
                  review_csv: str,
                  enrich_metadata: bool,
                  json_output: bool) -> None:
    """Import one author's DOI-bearing Crossref works into a corpus."""
    if bool(orcid) == bool(author_name):
        raise click.UsageError('Provide exactly one of --orcid or --author.')
    try:
        with _quiet(json_output):
            summary = import_author_works(
                db_path,
                email=email,
                orcid=orcid,
                author_name=author_name,
                affiliation=affiliation,
                max_results=max_results,
                review_csv=review_csv,
                enrich=enrich_metadata,
            )
    except (RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json({**summary, 'review_csv': review_csv})
        return
    click.echo(
        f'Crossref found {summary["found"]} matching works: '
        f'{summary["added"]} added and {summary["updated"]} updated in {db_path}.'
    )
    if enrich_metadata:
        click.echo(f'Enriched {summary["enriched"]} imported works.')
    click.echo(f'Review the imported metadata in {review_csv} before downloading papers.')


@agent_meta(effects={'network', 'writes_corpus'}, output=out.DOWNLOAD_OR_PLAN)
@click.command('download')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--format', 'download_format',
              type=click.Choice(sorted(DOWNLOAD_FORMATS)),
              default='both',
              show_default=True)
@click.option('--source', 'sources',
              multiple=True,
              type=click.Choice(DOWNLOAD_CHOICES),
              default=('all',),
              show_default=True,
              help='Content source to use for PDFs and PubMed Central full text. '
                   'Repeat to choose more than one.')
@click.option('--abstract/--no-abstract',
              'download_abstract',
              default=True,
              show_default=True,
              help='Download and store abstracts alongside paper assets; ignored for supplement-only runs.')
@click.option('--supplements/--no-supplements', 'download_supplements',
              default=True, show_default=True,
              help='Retrieve supporting information alongside paper assets; the abstract/supplement formats take precedence.')
@click.option('--supplement-source', 'supplement_sources', multiple=True,
              type=click.Choice(['all', *SUPPLEMENT_SOURCES]), default=('all',), show_default=True,
              help='Supporting-information source to use. Repeat to choose more than one.')
@click.option('--force', is_flag=True, default=False,
              help='Redownload requested assets even when that content type is already stored.')
@_paper_id_options
@_dry_run_option
@_progress_option
@_json_option
def download(
        db_path: str,
        download_format: str,
        sources: tuple[str, ...],
        download_abstract: bool,
        download_supplements: bool,
        supplement_sources: tuple[str, ...],
        force: bool,
        paper_ids: tuple[str, ...],
        paper_ids_file: str | None,
        dry_run: bool,
        progress_display: str,
        json_output: bool,
) -> None:
    """Download abstracts, text, PDFs, and supporting information for corpus rows.

    Supplement-only runs skip abstracts. Abstract-only runs skip supplements.
    With --dry-run nothing is requested: the command lists the assets each
    paper would be asked for, not which providers would supply them.
    """
    selected = _selected_paper_ids(db_path, paper_ids, paper_ids_file)
    selection = {} if selected is None else {'paper_ids': selected, 'respect_filters': False}
    if dry_run:
        try:
            with _quiet(json_output):
                plan = plan_downloads(
                    db_path, download_format=download_format, sources=list(sources),
                    download_abstract=download_abstract,
                    download_supplements=download_supplements,
                    supplement_sources=list(supplement_sources), force=force, **selection)
        except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
            raise click.ClickException(str(error)) from error
        if json_output:
            emit_json(plan)
        else:
            would, stored = plan['would_attempt'], plan['already_stored']
            click.echo(f'Dry run: {plan["papers_considered"]} papers; would request '
                       f'{would["texts"]} texts, {would["pdfs"]} PDFs, {would["abstracts"]} '
                       f'abstracts, and SI discovery for {would["supplement_discovery"]} papers.')
            click.echo(f'Already stored: {stored["texts"]} texts, {stored["pdfs"]} PDFs, '
                       f'{stored["abstracts"]} abstracts, {stored["supplement_files"]} SI files; '
                       f'{plan["no_text_source"]} papers have no reachable text source.')
        return
    try:
        with redirect_stdout(sys.stderr) if json_output else nullcontext(), \
                progress_mode(progress_display, 'download'):
            summary = download_papers(
                db_path,
                download_format=download_format,
                sources=list(sources),
                download_abstract=download_abstract,
                download_supplements=download_supplements,
                supplement_sources=list(supplement_sources),
                force=force,
                **selection,
            )
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)


@agent_meta(effects={'writes_corpus', 'writes_files'}, output=out.DIGITIZE)
@click.command('digitize')
@click.argument('db_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('records_csv', type=click.Path(exists=True, dir_okay=False))
@click.option('--figure-field', required=True,
              help="Column naming the figure that plots each record's result, such as 'Mass loss figure'.")
@click.option('--output', 'output_path', default='digitized_figures.csv', show_default=True,
              type=click.Path(dir_okay=False), help='CSV to append the recovered points to, one row per point.')
@click.option('--only-missing', 'missing_field', default=None,
              help='Only follow records whose value in this column is missing, such as the curve the figure plots.')
@click.option('--previews', 'preview_dir', default=None, type=click.Path(file_okay=False),
              help='Directory for a preview of each recovered figure, its curves drawn over it.')
@click.option('--detect/--no-detect', default=True, show_default=True,
              help='Detect and store figures from the PDF of a paper whose corpus entry holds none.')
@click.option('--limit', default=None, type=click.IntRange(min=1),
              help='Stop after this many figures have been attempted.')
@click.option('--force', is_flag=True, default=False,
              help='Recover figures again even when already recovered.')
@_json_option
def digitize(
        db_path: str,
        records_csv: str,
        figure_field: str,
        output_path: str,
        missing_field: str | None,
        preview_dir: str | None,
        detect: bool,
        limit: int | None,
        force: bool,
        json_output: bool,
) -> None:
    """Recover the curves of the figures that extracted records name, with figrecover.

    Reads each record's figure reference, such as "Figure 4a", finds that
    figure in the corpus (detecting figures from the PDF when the paper has
    none), reads its axes by OCR, and writes every curve's points. Curves
    are named by colour and line style, as figrecover does not read legends.
    Needs figrecover with its OCR extra, installed from GitHub.
    """
    try:
        with redirect_stdout(sys.stderr) if json_output else nullcontext():
            summary = digitize_figures(
                db_path,
                records_csv,
                figure_field,
                output_path,
                missing_field=missing_field,
                preview_dir=preview_dir,
                detect=detect,
                force=force,
                limit=limit,
            )
    except (RuntimeError, KeyError, ValueError, OSError, sqlite3.Error) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        raise click.ClickException(str(message)) from error
    if json_output:
        emit_json(summary)
        return
    click.echo(
        f'Figures requested: {summary["requests"]}; recovered {summary["digitized"]}, '
        f'already recovered {summary["skipped"]}, failed {summary["failed"]}, '
        f'not found {summary["not_found"]}, supplementary {summary["supplementary"]}.'
    )
    click.echo(f'{summary["series"]} curves in {summary["panels"]} panels, '
               f'{summary["points"]} points written to {summary["output"]}.')


@agent_meta(effects={'network', 'writes_corpus'}, output=out.ENRICH)
@click.command('enrich')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--source', 'sources',
              multiple=True,
              type=click.Choice(sources.choices(sources.ENRICH)),
              default=('all',),
              show_default=True,
              help='Metadata source to use. Repeat to choose more than one.')
@click.option('--batch-size',
              default=100,
              type=click.IntRange(1, 100),
              show_default=True,
              help='Papers looked up per provider request.')
@click.option('--limit',
              default=None,
              type=click.IntRange(min=1),
              help='Stop after enriching this many papers.')
@click.option('--force', is_flag=True, default=False,
              help='Re-enrich papers whose enrichment already succeeded.')
@click.option('--retry-failed', is_flag=True, default=False,
              help='Retry papers whose previous enrichment failed.')
@click.option('--refresh-after',
              default=0,
              type=click.IntRange(min=0),
              show_default=True,
              help='Re-enrich papers enriched more than this many days ago. 0 disables refreshing.')
@click.option('--references/--no-references',
              default=True,
              show_default=True,
              help='Store reference lists returned by the selected sources.')
@click.option('--resolve-references', is_flag=True, default=False,
              help='Link stored reference DOIs to papers already in the corpus.')
@click.option('--email',
              default=None,
              help='Contact email for Crossref. Defaults to the stored crossref_email setting.')
@_paper_id_options
@_progress_option
@_json_option
def enrich(db_path: str,
           sources: tuple[str, ...],
           batch_size: int,
           limit: int | None,
           force: bool,
           retry_failed: bool,
           refresh_after: int,
           references: bool,
           resolve_references: bool,
           email: str | None, paper_ids: tuple[str, ...], paper_ids_file: str | None,
           progress_display: str, json_output: bool) -> None:
    """Supplement corpus metadata with the selected provider records.

    Named papers (--paper-id) are still subject to the status rules: use
    --force or --retry-failed to redo succeeded or failed ones.
    """
    selected = _selected_paper_ids(db_path, paper_ids, paper_ids_file)
    selection = {} if selected is None else {'paper_ids': selected}
    try:
        with redirect_stdout(sys.stderr) if json_output else nullcontext(), \
                progress_mode(progress_display, 'enrich'):
            summary = enrich_corpus(
                db_path,
                sources=list(sources),
                batch_size=batch_size,
                limit=limit,
                force=force,
                retry_failed=retry_failed,
                refresh_after=refresh_after,
                references=references,
                resolve_references=resolve_references,
                email=email,
                **selection,
            )
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)
        return
    click.echo(
        f'Enrichment complete: {summary["succeeded"]} enriched, '
        f'{summary["partial"]} partial, {summary["not_found"]} not found.'
    )
    click.echo(
        f'Stored {summary["authors"]} author records, {summary["subjects"]} subject records, '
        f'and {summary["references"]} references.'
    )
    if summary['unresolved']:
        click.echo(
            f'{summary["unresolved"]} papers have no DOI, OpenAlex identifier, PMID, '
            f'arXiv identifier, medRxiv DOI, bioRxiv DOI, or chemRxiv DOI and were '
            f'skipped.',
            err=True,
        )


@agent_meta(effects={'migrates_corpus'}, output=out.CORPUS_STATS)
@click.command('stats')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True, dir_okay=False))
@_json_option
def corpus_status(db_path: str, json_output: bool) -> None:
    """Print storage statistics for the paper corpus."""
    try:
        with closing(connect(db_path)) as conn, conn:
            stats = corpus_stats(conn)
            enrichment = enrichment_stats(conn)
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json({'stats': stats, 'enrichment': enrichment})
        return
    click.echo(f'Corpus: {db_path}')
    click.echo(f'Papers: {stats["papers"]}')
    click.echo(f'Papers with abstracts: {stats["papers_with_abstract"]}')
    click.echo(f'Papers with text: {stats["papers_with_text"]}')
    click.echo(f'Papers with PDFs: {stats["papers_with_pdf"]}')
    click.echo(f'Papers with supporting information: {stats.get("papers_with_supplements", 0)}')
    click.echo(f'Supplement files: {stats.get("supplements", 0)}')
    click.echo(
        f'Papers with structured documents: '
        f'{stats["papers_with_structured_documents"]}'
    )
    click.echo(f'Text scrapes split into chunks: {stats["papers_with_chunked_text"]}')
    click.echo(f'Abstract scrapes split into chunks: {stats["papers_with_chunked_abstracts"]}')
    click.echo(f'Blobs: {stats["blobs"]}')
    click.echo(f'Original size: {_format_bytes(stats["original_size"])}')
    click.echo(f'Stored size: {_format_bytes(stats["stored_size"])}')
    click.echo(f'Storage saved: {stats["savings_fraction"]:.1%}')
    click.echo(f'Papers enriched: {enrichment["papers_succeeded"]}')
    click.echo(f'Papers open access: {enrichment["papers_open_access"]}')
    click.echo(f'Papers retracted: {enrichment["papers_retracted"]}')
    click.echo(f'Author records: {enrichment["author_records"]} '
               f'({enrichment["authors_with_orcid"]} with ORCID)')
    click.echo(f'Subject records: {enrichment["subject_records"]}')
    click.echo(f'Reference records: {enrichment["reference_records"]}')


@agent_meta(effects={'migrates_corpus', 'writes_files'}, output=out.CORPUS_SEARCHES)
@click.command('searches')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True, dir_okay=False))
@click.option('--limit', default=20, type=click.IntRange(min=1), show_default=True,
              help='Maximum number of recent searches to show.')
@click.option('--outfile', default=None, type=click.Path(dir_okay=False),
              help='Write the selected recent search history to a JSON file.')
@_json_option
def corpus_searches(db_path: str, limit: int, outfile: str | None, json_output: bool) -> None:
    """Print the searches recorded in a paper corpus.

    Parameters
    ----------
    db_path : str
        Corpus database to inspect.
    limit : int
        Maximum number of recent searches to display.
    outfile : str or None
        Optional JSON output path. A ``.json`` suffix is appended when absent.
    json_output : bool
        Emit the selected search history as JSON on stdout.
    """
    try:
        with closing(connect(db_path)) as conn, conn:
            searches = search_history(conn, limit=limit)
        if outfile is not None:
            output_path = Path(outfile)
            if output_path.suffix.lower() != '.json':
                output_path = Path(f'{output_path}.json')
            output_path.write_text(f'{json.dumps(searches, indent=2)}\n', encoding='utf-8')
            click.echo(f'Search history written to {output_path}.', err=json_output)
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(searches)
        return
    if outfile is not None:
        return
    click.echo(f'Corpus searches: {db_path}')
    if not searches:
        click.echo('No searches recorded.')
        return
    for item in searches:
        click.echo(
            f'#{item["search_id"]} {item["started_at"]} {item["status"]} | '
            f'source={item["requested_source"]} | requested={item["requested_count"]} | '
            f'parallel={"yes" if item["parallel"] else "no"} | workers={item["workers"]} | '
            f'results={item["result_count"]} | added={item["papers_added"]} | '
            f'updated={item["papers_updated"]}'
        )
        click.echo(f'  {" ".join(item["query"].split())}')
        for source_name, outcome in item['source_results'].items():
            if outcome['status'] == 'failed':
                click.echo(f'  {source_name} failed: {outcome.get("error", "unknown error")}', err=True)


def _echo_filter_overview(db_path: str, overview: Mapping[str, Any]) -> None:
    """Print a compact, explicit summary of the active filter stack."""
    click.echo(f'Corpus filters: {db_path}')
    if not overview['filters']:
        click.echo('Active expression: none')
    else:
        click.echo(f'Active expression: {overview["expression"]}')
        for index, item in enumerate(overview['filters']):
            prefix = 'ROOT' if index == 0 else item['join_operator'].upper()
            definition = item['definition']
            counts = item['counts']
            if item['method'] == 'regex':
                details = (
                    f'regex; {definition["include_mode"]}; '
                    f'fields={",".join(definition["fields"])}; '
                    f'timeout={definition["timeout_ms"]}ms'
                )
            else:
                details = (
                    f'topic; model={definition["model"]}; '
                    f'{definition["include_mode"]}'
                )
            stale = '; STALE' if item.get('stale') else ''
            click.echo(
                f'  {prefix} {item["name"]} [{details}{stale}] '
                f'included={counts["included"]}, excluded={counts["excluded"]}, '
                f'unavailable={counts["unavailable"]}'
            )
    counts = overview['counts']
    click.echo(
        f'Final result: included={counts["included"]}, excluded={counts["excluded"]}, '
        f'unavailable={counts["unavailable"]}'
    )
    if overview['unavailable_reasons']:
        click.echo('Unavailable reasons:')
        ordered = sorted(
            overview['unavailable_reasons'].items(), key=lambda item: (-item[1], item[0])
        )
        for reason, count in ordered[:10]:
            click.echo(f'  {count}: {reason}')
        if len(ordered) > 10:
            click.echo(f'  ... {len(ordered) - 10} more reason combinations stored in the corpus')
    if overview.get('stale_topic_filters'):
        click.echo(
            'Stale topic filters: ' + ', '.join(overview['stale_topic_filters'])
            + '. Run pmt topics store again before scraping.'
        )


def _report_filter_overview(db_path: str, overview: Mapping[str, Any],
                            json_output: bool) -> None:
    """Write a filter overview as JSON or as the human-readable summary."""
    if json_output:
        emit_json(overview)
        return
    if overview.get('dry_run'):
        click.echo('Dry run: the corpus was not changed. It would look like this:')
    _echo_filter_overview(db_path, overview)


@agent_meta(effects={'writes_corpus'}, output=out.FILTER_OVERVIEW)
@click.command('regex')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('rules_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--field', 'fields', multiple=True,
              type=click.Choice(['title', 'abstract', 'full_text']),
              help='Override JSON fields. Repeat to combine fields.')
@click.option('--join', 'join_operator', type=click.Choice(['and', 'or']), default=None,
              help='Join this filter to the preceding active expression.')
@click.option('--replace', is_flag=True, default=False,
              help='Replace and reevaluate an active filter with the same name.')
@click.option('--timeout-ms', type=click.IntRange(min=1), default=None,
              help='Override the per-pattern match timeout from the JSON definition.')
@_dry_run_option
@_json_option
def filter_regex(db_path: str, rules_path: str, fields: tuple[str, ...],
                 join_operator: str | None, replace: bool, timeout_ms: int | None,
                 dry_run: bool, json_output: bool) -> None:
    """Apply a named post-download regex filter to a paper corpus.

    With --dry-run the filter is applied to a temporary copy of the corpus
    and the resulting decisions are reported; the corpus is not changed.
    """
    def apply(path: str) -> Mapping[str, Any]:
        """Apply the filter to the corpus at ``path``."""
        return apply_regex_filter(path, rules_path, fields=fields or None,
                                  join_operator=join_operator, replace=replace,
                                  timeout_ms=timeout_ms)

    try:
        with _quiet(json_output):
            overview = _on_copy(db_path, apply) if dry_run else apply(db_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    _report_filter_overview(db_path, overview, json_output)


@agent_meta(effects={'writes_corpus'}, output=out.FILTER_OVERVIEW)
@click.command('topic')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('rules_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--join', 'join_operator', type=click.Choice(['and', 'or']), default=None,
              help='Join this filter to the preceding active expression.')
@click.option('--replace', is_flag=True, default=False,
              help='Replace and reevaluate an active filter with the same name.')
@_dry_run_option
@_json_option
def filter_topic(db_path: str, rules_path: str,
                 join_operator: str | None, replace: bool, dry_run: bool,
                 json_output: bool) -> None:
    """Apply a named stored-model topic filter to a paper corpus.

    With --dry-run the filter is applied to a temporary copy of the corpus
    and the resulting decisions are reported; the corpus is not changed.
    """
    def apply(path: str) -> Mapping[str, Any]:
        """Apply the filter to the corpus at ``path``."""
        return apply_topic_filter(path, rules_path, join_operator=join_operator, replace=replace)

    try:
        with _quiet(json_output):
            overview = _on_copy(db_path, apply) if dry_run else apply(db_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    _report_filter_overview(db_path, overview, json_output)


@agent_meta(effects={'migrates_corpus'}, output=out.FILTER_OVERVIEW)
@click.command('status')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True, dir_okay=False))
@_json_option
def filter_status(db_path: str, json_output: bool) -> None:
    """Show active corpus filters and their final paper decisions."""
    try:
        with closing(connect(db_path)) as conn, conn:
            overview = filter_overview(conn)
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    _report_filter_overview(db_path, overview, json_output)


@agent_meta(effects={'writes_corpus'}, output=out.FILTER_OVERVIEW)
@click.command('reset')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--name', default=None, help='Remove one active filter by name.')
@click.option('--all', 'all_filters', is_flag=True, default=False,
              help='Remove every active filter.')
@_dry_run_option
@_json_option
def filter_reset(db_path: str, name: str | None, all_filters: bool, dry_run: bool,
                 json_output: bool) -> None:
    """Remove one or all active corpus filters.

    With --dry-run the removal runs on a temporary copy of the corpus and the
    remaining decisions are reported; the corpus is not changed.
    """
    def apply(path: str) -> Mapping[str, Any]:
        """Remove the filters from the corpus at ``path``."""
        return reset_filters(path, name=name, all_filters=all_filters)

    try:
        with _quiet(json_output):
            overview = _on_copy(db_path, apply) if dry_run else apply(db_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    _report_filter_overview(db_path, overview, json_output)


# Reuse option definitions while retaining each command's ordering and help.
# Click decorators construct a separate Option object for each callback.
_topic_field_option = click.option(
    '--field', 'text_fields', multiple=True,
    type=click.Choice(['title', 'abstract', 'text']),
    default=('title', 'abstract'), show_default=True,
    help='Corpus text field to model. Repeat to combine fields.',
)
_topic_min_df_option = partial(
    click.option, '--min-df', default=2, type=click.IntRange(min=1), show_default=True,
)
_topic_max_df_option = partial(
    click.option, '--max-df', default=0.95,
    type=click.FloatRange(min=0.0, max=1.0, min_open=True), show_default=True,
)
_topic_max_features_option = click.option(
    '--max-features', default=20000, type=click.IntRange(min=2), show_default=True,
)
_topic_learning_method_option = click.option(
    '--learning-method', type=click.Choice(['online', 'batch']), default='online', show_default=True,
)
_topic_iterations_option = click.option(
    '--iterations', 'max_iter', default=20, type=click.IntRange(min=1), show_default=True,
)
_topic_top_terms_option = click.option(
    '--top-terms', default=15, type=click.IntRange(min=1), show_default=True,
)
_topic_representative_papers_option = click.option(
    '--representative-papers', default=5, type=click.IntRange(min=1), show_default=True,
)
_topic_stopwords_file_option = click.option(
    '--stopwords-file', default=None, type=click.Path(exists=True, dir_okay=False),
    help='UTF-8 file containing one corpus-specific stopword per line.',
)
_topic_ngram_max_option = partial(
    click.option, '--ngram-max', default=2, type=click.IntRange(min=1, max=2), show_default=True,
)
_topic_batch_size_option = click.option(
    '--batch-size', default=128, type=click.IntRange(min=1), show_default=True,
)
_topic_cache_dir_option = click.option(
    '--cache-dir', default=None, type=click.Path(file_okay=False),
    help='Parent directory for temporary streaming caches.',
)
_topic_evaluation_sample_size_option = click.option(
    '--evaluation-sample-size', default=10000, type=click.IntRange(min=1), show_default=True,
)


@agent_meta(effects={'migrates_corpus', 'writes_files'}, output=out.TOPICS_TRAIN)
@click.command('train')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('model_dir', default='topic_model', type=click.Path())
@click.option('--topics', 'num_topics', default=10, type=click.IntRange(min=2), show_default=True)
@_topic_field_option
@_topic_min_df_option(help='Minimum number of documents containing a retained term.')
@_topic_max_df_option(help='Maximum fraction of documents containing a retained term.')
@_topic_max_features_option
@_topic_learning_method_option
@_topic_iterations_option
@click.option('--random-seed', 'random_state', default=0, type=int, show_default=True)
@_topic_top_terms_option
@_topic_representative_papers_option
@_topic_stopwords_file_option
@_topic_ngram_max_option(help='Largest generated n-gram; use 2 to include bigrams.')
@click.option('--overwrite', is_flag=True, default=False,
              help='Replace known model artifact files in a non-empty model directory.')
@click.option('--streaming/--in-memory', default=True, show_default=True,
              help='Use disk-backed bounded batches or materialize the corpus in memory.')
@_topic_batch_size_option
@_topic_cache_dir_option
@_topic_evaluation_sample_size_option
@_json_option
def topics_train(db_path: str,
                 model_dir: str,
                 num_topics: int,
                 text_fields: tuple[str, ...],
                 min_df: int,
                 max_df: float,
                 max_features: int,
                 learning_method: str,
                 max_iter: int,
                 random_state: int,
                 top_terms: int,
                 representative_papers: int,
                 stopwords_file: str | None,
                 ngram_max: int,
                 overwrite: bool,
                 streaming: bool,
                 batch_size: int,
                 cache_dir: str | None,
                 evaluation_sample_size: int,
                 json_output: bool) -> None:
    """Train an LDA model and write inspectable, manually nameable artifacts."""
    try:
        with _quiet(json_output):
            summary = train_topic_model(
                db_path,
                model_dir,
                num_topics=num_topics,
                text_fields=text_fields,
                min_df=min_df,
                max_df=max_df,
                max_features=max_features,
                learning_method=learning_method,
                max_iter=max_iter,
                random_state=random_state,
                top_terms=top_terms,
                representative_papers=representative_papers,
                stopwords_file=stopwords_file,
                ngram_max=ngram_max,
                overwrite=overwrite,
                emit_warnings=False,
                streaming=streaming,
                batch_size=batch_size,
                cache_dir=cache_dir,
                evaluation_sample_size=evaluation_sample_size,
            )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    for message in summary['report']['warnings']:
        click.echo(f'Warning: {message}', err=True)
    if json_output:
        emit_json(summary)
        return
    click.echo(
        f'Trained {num_topics} topics from {summary["report"]["documents_used"]} papers '
        f'using {summary["report"]["vocabulary_size"]} terms.'
    )
    click.echo(f'Model artifacts: {summary["model_dir"]}')
    click.echo(f'Manual topic review: {summary["model_dir"]}/topics.csv')


@agent_meta(effects={'migrates_corpus', 'writes_files'}, output=out.TOPICS_COMPARE)
@click.command('compare')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('output_dir', default='topic_comparison', type=click.Path())
@click.option('--topics', 'topic_counts', multiple=True, type=click.IntRange(min=2),
              default=(5, 10), show_default=True,
              help='Topic count to train. Repeat to compare several values.')
@click.option('--seed', 'random_states', multiple=True, type=int, default=(0, 1), show_default=True,
              help='Random seed to train. Repeat to assess stability.')
@_topic_field_option
@_topic_min_df_option()
@_topic_max_df_option()
@_topic_max_features_option
@_topic_learning_method_option
@_topic_iterations_option
@_topic_top_terms_option
@_topic_representative_papers_option
@_topic_stopwords_file_option
@_topic_ngram_max_option()
@click.option('--overwrite', is_flag=True, default=False,
              help='Replace known comparison and model artifact files.')
@click.option('--streaming/--in-memory', default=True, show_default=True,
              help='Use one reusable disk-backed corpus cache or in-memory matrices.')
@_topic_batch_size_option
@_topic_cache_dir_option
@_topic_evaluation_sample_size_option
@_json_option
def topics_compare(db_path: str,
                   output_dir: str,
                   topic_counts: tuple[int, ...],
                   random_states: tuple[int, ...],
                   text_fields: tuple[str, ...],
                   min_df: int,
                   max_df: float,
                   max_features: int,
                   learning_method: str,
                   max_iter: int,
                   top_terms: int,
                   representative_papers: int,
                   stopwords_file: str | None,
                   ngram_max: int,
                   overwrite: bool,
                   streaming: bool,
                   batch_size: int,
                   cache_dir: str | None,
                   evaluation_sample_size: int,
                   json_output: bool) -> None:
    """Train several LDA configurations and export comparable diagnostics."""
    try:
        with _quiet(json_output):
            summary = compare_topic_models(
                db_path,
                output_dir,
                topic_counts=topic_counts,
                random_states=random_states,
                text_fields=text_fields,
                min_df=min_df,
                max_df=max_df,
                max_features=max_features,
                learning_method=learning_method,
                max_iter=max_iter,
                top_terms=top_terms,
                representative_papers=representative_papers,
                stopwords_file=stopwords_file,
                ngram_max=ngram_max,
                overwrite=overwrite,
                streaming=streaming,
                batch_size=batch_size,
                cache_dir=cache_dir,
                evaluation_sample_size=evaluation_sample_size,
            )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    warning_messages = dict.fromkeys(
        message
        for model in summary['models']
        for message in model['warnings']
    )
    for message in warning_messages:
        click.echo(f'Warning: {message}', err=True)
    if json_output:
        emit_json(summary)
        return
    click.echo(f'Trained {summary["models_trained"]} comparison models in {summary["output_dir"]}.')
    click.echo(f'Comparison metrics: {summary["comparison_csv"]}')
    click.echo('Inspect each model with pmt topics show before choosing one.')


@agent_meta(effects={'read_only'}, output=out.TOPICS_SHOW)
@click.command('show')
@click.argument('model_dir', default='topic_model', type=click.Path(exists=True, file_okay=False))
@click.option('--representatives', default=3, type=click.IntRange(min=0), show_default=True,
              help='Representative paper titles to print for each topic.')
@_json_option
def topics_show(model_dir: str, representatives: int, json_output: bool) -> None:
    """Print top terms and representative papers for manual topic naming."""
    try:
        with _quiet(json_output):
            topics = topic_descriptions(model_dir)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json({'model_dir': model_dir, 'topics': [
            {**topic, 'representative_papers': list(topic['representative_papers'][:representatives])}
            for topic in topics
        ]})
        return
    for topic in topics:
        name = topic['topic_name'] or '(unnamed)'
        click.echo(f'Topic {topic["topic_id"]}: {name}')
        click.echo(f'  Terms: {", ".join(topic["top_terms"])}')
        for paper in topic['representative_papers'][:representatives]:
            click.echo(f'  - {paper["title"]} ({float(paper["probability"]):.3f})')


@agent_meta(effects={'writes_files'}, output=out.TOPICS_NAME)
@click.command('name')
@click.argument('model_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('topic_id', type=click.IntRange(min=0))
@click.argument('topic_name', type=str)
@_json_option
def topics_name(model_dir: str, topic_id: int, topic_name: str, json_output: bool) -> None:
    """Assign a manual human-readable name to one fitted topic."""
    try:
        with _quiet(json_output):
            set_topic_name(model_dir, topic_id, topic_name)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json({'model_dir': model_dir, 'topic_id': topic_id,
                   'topic_name': topic_name.strip()})
        return
    click.echo(f'Named topic {topic_id}: {topic_name.strip()}')


@agent_meta(effects={'migrates_corpus', 'writes_files'}, output=out.TOPICS_PREDICT)
@click.command('predict')
@click.argument('model_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('output_path', default='paper_topics.csv', type=click.Path())
@click.option('--batch-size', default=128, type=click.IntRange(min=1), show_default=True)
@_json_option
def topics_predict(model_dir: str, db_path: str, output_path: str, batch_size: int,
                   json_output: bool) -> None:
    """Apply a saved LDA model to a corpus and export topic probabilities."""
    try:
        with _quiet(json_output):
            summary = predict_topic_model(
                model_dir, db_path, output_path, batch_size=batch_size
            )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)
        return
    click.echo(
        f'Predicted topics for {summary["papers_predicted"]} of {summary["papers_total"]} papers; '
        f'{summary["papers_without_vocabulary_terms"]} had no model vocabulary terms.'
    )
    click.echo(f'Topic scores: {summary["output_path"]}')


@agent_meta(effects={'writes_files'}, output=out.TOPICS_TRENDS)
@click.command('trends')
@click.argument('model_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', default='topic_trends', type=click.Path())
@click.option('--predictions', 'predictions_path', default=None,
              type=click.Path(exists=True, dir_okay=False),
              help='Long-form prediction CSV; defaults to the model training predictions.')
@click.option('--bin-size', default=1, type=click.IntRange(min=1), show_default=True)
@click.option('--step-size', default=1, type=click.IntRange(min=1), show_default=True)
@click.option('--start-year', default=None, type=int)
@click.option('--end-year', default=None, type=int)
@click.option('--include-partial/--complete-only', default=True, show_default=True)
@click.option('--plot', is_flag=True, default=False,
              help='Write topic_trends_plot.png in the trend output directory.')
@click.option('--plot-file', default=None, type=click.Path(dir_okay=False),
              help='Write a plot to this filename instead; its extension selects the format.')
@click.option('--overwrite', is_flag=True, default=False,
              help='Replace known trend artifacts in a non-empty output directory.')
@_json_option
def topics_trends(model_dir: str, output_dir: str, predictions_path: str | None,
                  bin_size: int, step_size: int, start_year: int | None,
                  end_year: int | None, include_partial: bool, plot: bool,
                  plot_file: str | None,
                  overwrite: bool, json_output: bool) -> None:
    """Aggregate fixed-model topic probabilities over publication-year windows."""
    try:
        with _quiet(json_output):
            summary = aggregate_topic_trends(
                model_dir,
                output_dir,
                predictions_path=predictions_path,
                bin_size=bin_size,
                step_size=step_size,
                start_year=start_year,
                end_year=end_year,
                include_partial=include_partial,
                overwrite=overwrite,
                plot=plot_file or plot,
            )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)
        return
    click.echo(f'Wrote {summary["windows"]} topic trend windows to {summary["trends_csv"]}.')
    if summary['plot_path']:
        click.echo(f'Topic trend plot: {summary["plot_path"]}')
    if summary['papers_missing_or_invalid_date']:
        click.echo(
            f'{summary["papers_missing_or_invalid_date"]} papers had missing or invalid dates.',
            err=True,
        )


@agent_meta(effects={'writes_corpus'}, output=out.TOPICS_STORE)
@click.command('store')
@click.argument('model_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--name', default=None,
              help='Immutable corpus display name; defaults to the model directory name.')
@click.option('--batch-size', default=128, type=click.IntRange(min=1), show_default=True)
@_json_option
def topics_store(model_dir: str, db_path: str, name: str | None, batch_size: int,
                 json_output: bool) -> None:
    """Predict afresh and transactionally store a topic model in the corpus."""
    try:
        with _quiet(json_output):
            summary = store_topic_model_scores(
                model_dir, db_path, name=name, batch_size=batch_size
            )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)
        return
    click.echo(
        f'Stored {summary["name"]} ({summary["model_id"]}): '
        f'{summary["papers_predicted"]} predicted, '
        f'{summary["papers_without_vocabulary_terms"]} without vocabulary terms.'
    )


@agent_meta(effects={'migrates_corpus'}, output=out.TOPICS_MODELS)
@click.command('models')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--batch-size', default=128, type=click.IntRange(min=1), show_default=True)
@_json_option
def topics_models(db_path: str, batch_size: int, json_output: bool) -> None:
    """List topic models and prediction freshness recorded in a corpus."""
    try:
        with _quiet(json_output):
            models = stored_topic_models(db_path, batch_size=batch_size)
    except (FileNotFoundError, OSError, ValueError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json({'models': models})
        return
    if not models:
        click.echo('No topic models are stored in this corpus.')
        return
    for item in models:
        state = 'current' if item['is_current'] else 'STALE'
        click.echo(
            f'{item["name"]} ({item["model_id"]}) [{state}]: '
            f'{item["num_topics"]} topics, '
            f'{item["papers_predicted"]} predicted, '
            f'{item["papers_without_vocabulary_terms"]} without vocabulary terms; '
            f'fields={",".join(item["text_fields"])}'
        )


def _entity_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Apply the shared inference and export options to an entity command."""
    options = [
        click.option('--model', required=True,
                     help='Fine-tuned token-classification checkpoint: Hugging Face ID or local directory.'),
        click.option('--device', default='cpu', show_default=True,
                     help='PyTorch device, for example cpu or cuda:0.'),
        click.option('--batch-size', default=8, type=click.IntRange(min=1), show_default=True,
                     help='Maximum number of text windows per inference batch.'),
        click.option('--stride', default=64, type=click.IntRange(min=0), show_default=True,
                     help='Overlapping tokens between long-document windows.'),
        click.option('--max-length', default=None, type=click.IntRange(min=1),
                     help='Tokens per window, including special tokens; defaults to the model limit.'),
        click.option('--revision', default=None,
                     help='Hugging Face checkpoint revision, such as a commit hash.'),
        click.option('--cache-dir', default=None, type=click.Path(file_okay=False),
                     help='Directory for downloaded model and tokenizer files.'),
        click.option('--local-files-only', is_flag=True,
                     help='Load only local or already cached checkpoint files.'),
        click.option('--min-score', default=0.0, type=click.FloatRange(0, 1), show_default=True,
                     help='Minimum entity confidence to include in the output.'),
        click.option('--label', 'labels', multiple=True,
                     help='Include this checkpoint-specific entity label; repeat for several.'),
        click.option('--overwrite', is_flag=True,
                     help='Replace an existing JSONL output file.'),
        _json_option,
    ]
    for option in reversed(options):
        command = option(command)
    return command


@agent_meta(effects={'migrates_corpus', 'network', 'writes_files'}, output=out.ENTITIES)
@click.command('corpus')
@click.argument('db_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(dir_okay=False))
@click.option('--field', 'text_fields', multiple=True, default=('abstract',),
              type=click.Choice(['title', 'abstract', 'text']), show_default=True,
              help='Corpus text field to annotate; repeat to process several fields separately.')
@_entity_options
def entities_corpus(db_path: str, output_path: str, text_fields: tuple[str, ...],
                    model: str, device: str, batch_size: int, stride: int,
                    max_length: int | None, revision: str | None, cache_dir: str | None,
                    local_files_only: bool, min_score: float, labels: tuple[str, ...],
                    overwrite: bool, json_output: bool) -> None:
    """Extract named entities from stored corpus text into a JSONL file."""
    try:
        config = EntityExtractionConfig(
            model=model, device=device, batch_size=batch_size, stride=stride,
            max_length=max_length, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        with _quiet(json_output):
            summary = extract_corpus_entities(
                db_path, output_path, config, text_fields=text_fields,
                min_score=min_score, labels=labels or None, overwrite=overwrite,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)
        return
    click.echo(
        f'Extracted {summary["entities"]} entities from {summary["documents"]} documents '
        f'across {summary["papers"]} papers.'
    )
    click.echo(f'Entity annotations: {summary["output"]}')


@agent_meta(effects={'network', 'writes_files'}, output=out.ENTITIES)
@click.command('text')
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(dir_okay=False))
@_entity_options
def entities_text(input_path: str, output_path: str, model: str, device: str,
                  batch_size: int, stride: int, max_length: int | None,
                  revision: str | None, cache_dir: str | None, local_files_only: bool,
                  min_score: float, labels: tuple[str, ...], overwrite: bool,
                  json_output: bool) -> None:
    """Extract named entities from one UTF-8 text file into a JSONL file."""
    try:
        config = EntityExtractionConfig(
            model=model, device=device, batch_size=batch_size, stride=stride,
            max_length=max_length, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        with _quiet(json_output):
            summary = extract_file_entities(
                input_path, output_path, config, min_score=min_score,
                labels=labels or None, overwrite=overwrite,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)
        return
    click.echo(f'Extracted {summary["entities"]} entities from {summary["documents"]} documents.')
    click.echo(f'Entity annotations: {summary["output"]}')


@agent_meta(effects={'network', 'paid_model', 'writes_corpus', 'writes_files'}, output=out.SCRAPE)
@click.command('scrape')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('recipe', default='sse', type=str)
@click.option('--mode', type=click.Choice(['abstract', 'text', 'images', 'text-images']), default='text', show_default=True)
@click.option('--supplements/--no-supplements', 'include_supplements',
              default=True, show_default=True,
              help='Include stored supporting information in text and image extraction.')
@click.option('--image-context', type=click.Choice(['none', 'paper-text']), default='none', show_default=True)
@click.option('--image-dir', default='paper_images', type=click.Path(), show_default=True)
@click.option('--image-extraction',
              type=click.Choice(['auto', 'embedded', 'pages', 'layout']),
              default='auto',
              show_default=True,
              help='How to choose images for vision analysis. "layout" sends figures with their captions.')
@click.option('--image-dpi', default=200, type=int, show_default=True,
              help='DPI for rendered page images when page rendering is used.')
@click.option('--image-batch-size', default='1', show_default=True,
              help='Number of images per vision request, or "all".')
@click.option('--model', default=None, help='Text model name override for this scrape run.')
@click.option('--provider', default=None,
              help='Text provider override: openai, anthropic, or local. Pair with --model when changing provider.')
@click.option('--base-url', default=None, help='Text model base URL override.')
@click.option('--vision-model', default=None, help='Vision model name override for this scrape run.')
@click.option('--vision-provider', default=None,
              help='Vision provider override. Pair with --vision-model when changing provider.')
@click.option('--vision-base-url', default=None, help='Vision model base URL override.')
@click.option('--delete-images-after',
              is_flag=True,
              default=False,
              help='Delete extracted images after successful image analysis.')
@click.option('--output', 'output_path',
              default='temp_scraped_materials.csv',
              type=click.Path(),
              show_default=True,
              help='CSV file for newly scraped material rows.')
@click.option('--force',
              is_flag=True,
              default=False,
              help='Rescrape stages even if their status is already succeeded.')
@click.option('--ignore-filters',
              is_flag=True,
              default=False,
              help='Explicitly bypass active corpus filters for this scrape run.')
@click.option('--count',
              'scrape_count',
              default=None,
              type=int,
              help='Maximum number of corpus papers to process in this scrape run.')
@click.option('--order',
              'scrape_order',
              type=click.Choice(sorted(SCRAPE_ORDERS)),
              default='corpus',
              show_default=True,
              help='Order used to select papers for this scrape run.')
@click.option('--compression-scope',
              type=click.Choice(sorted(COMPRESSION_SCOPES)),
              default='none',
              show_default=True,
              help='Inputs to compress with Headroom.')
@click.option('--compression-mode',
              type=click.Choice(sorted(COMPRESSION_MODES)),
              default='auto',
              show_default=True,
              help='When to compress selected inputs.')
@click.option('--compression-ratio',
              default='auto',
              show_default=True,
              help='Target compression ratio, or "auto" to fit the configured input token budget.')
@click.option('--compression-content-detection/--no-compression-content-detection',
              default=True,
              show_default=True,
              help='Use Headroom content detection when compressing inputs.')
@_paper_id_options
@click.option('--dry-run', is_flag=True, default=False,
              help='Report the papers, model requests, and estimated input tokens a run would '
                   'use, without calling a model or changing the corpus.')
@_progress_option
@_json_option
def scrape(
        db_path: str,
        recipe: str,
        mode: str,
        include_supplements: bool,
        image_context: str,
        image_dir: str,
        image_extraction: str,
        image_dpi: int,
        image_batch_size: str,
        model: str | None,
        provider: str | None,
        base_url: str | None,
        vision_model: str | None,
        vision_provider: str | None,
        vision_base_url: str | None,
        delete_images_after: bool,
        output_path: str,
        force: bool,
        ignore_filters: bool,
        scrape_count: int | None,
        scrape_order: str,
        compression_scope: str,
        compression_mode: str,
        compression_ratio: str,
        compression_content_detection: bool,
        paper_ids: tuple[str, ...],
        paper_ids_file: str | None,
        dry_run: bool,
        progress_display: str,
        json_output: bool,
) -> None:
    """Run text and/or image scraping over downloaded papers.

    Exits 0 when the run completes, even if some papers failed: inspect the
    failed counts and failed_papers in the --json summary, or pmt status.
    With --dry-run, nothing is scraped: the command reports what a run with
    the same options would do and how many tokens it would send.
    """
    selected = _selected_paper_ids(db_path, paper_ids, paper_ids_file)
    selection = {} if selected is None else {'paper_ids': selected}
    if dry_run:
        try:
            with _quiet(json_output):
                plan = plan_scrape(
                    db_path, recipe, mode=mode, model=model, provider=provider,
                    base_url=base_url, force=force, ignore_filters=ignore_filters,
                    scrape_count=scrape_count, scrape_order=scrape_order,
                    compression_scope=compression_scope, compression_mode=compression_mode,
                    compression_ratio=compression_ratio,
                    compression_content_detection=compression_content_detection,
                    include_supplements=include_supplements,
                    image_extraction=image_extraction, **selection,
                )
        except (RuntimeError, ValueError, KeyError, OSError, sqlite3.Error) as error:
            message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
            raise click.ClickException(str(message)) from error
        if json_output:
            emit_json(plan)
        else:
            _echo_scrape_plan(plan)
        return
    try:
        with _quiet(json_output), progress_mode(progress_display, 'scrape'):
            summary = scrape_papers(
                db_path,
                recipe,
                mode=mode,
                include_supplements=include_supplements,
                image_dir=image_dir,
                image_context=image_context,
                image_extraction=image_extraction,
                image_dpi=image_dpi,
                image_batch_size=image_batch_size,
                model=model,
                provider=provider,
                base_url=base_url,
                vision_model=vision_model,
                vision_provider=vision_provider,
                vision_base_url=vision_base_url,
                delete_images_after=delete_images_after,
                output_path=output_path,
                force=force,
                ignore_filters=ignore_filters,
                scrape_count=scrape_count,
                scrape_order=scrape_order,
                compression_scope=compression_scope,
                compression_mode=compression_mode,
                compression_ratio=compression_ratio,
                compression_content_detection=compression_content_detection,
                **selection,
            )
    except (RuntimeError, ValueError, KeyError, OSError, sqlite3.Error) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        raise click.ClickException(str(message)) from error
    if json_output:
        emit_json(summary)


@agent_meta(effects={'migrates_corpus'}, output=out.SCRAPE_PROGRESS)
@click.command('scrape-progress')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True, dir_okay=False))
@click.argument('mode', default='text', type=click.Choice(list(SCRAPE_PROGRESS_MODES)))
@_json_option
def scrape_progress_command(db_path: str, mode: str, json_output: bool) -> None:
    """Report how far a scrape has got and what rerunning it would still do.

    Counts only the papers pmt scrape would select (active filters apply),
    separating papers with no downloaded document, which need pmt download,
    from failures a rerun retries. The --json output lists their paper IDs
    for --paper-id. No model is called.
    """
    try:
        with _quiet(json_output):
            progress = scrape_progress(db_path, mode)
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(progress)
    else:
        click.echo(scrape_progress_report(progress))


def _echo_scrape_plan(plan: Mapping[str, Any]) -> None:
    """Print the human-readable summary of a scrape dry run."""
    totals = plan['totals']
    model = plan['model']
    click.echo(f'Dry run: {plan["mode"]} scrape with recipe {plan["recipe"]}; nothing was scraped.')
    click.echo(f'Model: {model["provider"]}/{model["name"]}, input limit {model["input_token_limit"]} '
               f'tokens; token counts from {plan["tokenizer"]}, before any compression.')
    click.echo(f'Papers selected: {plan["papers_selected"]} '
               f'({plan["papers_filtered_out"]} excluded by filters); '
               f'would scrape {totals["papers_to_scrape"]}, skip {totals["skipped_done"]} done, '
               f'and fail {totals["skipped_missing_source"]} with no downloaded source.')
    click.echo(f'Model requests: {totals["requests"]} text requests carrying about '
               f'{totals["input_tokens"]} input tokens; {totals["image_inputs"]} image inputs '
               'not estimated.')
    click.echo(f'Supporting information: {totals["supplement_files_to_scrape"]} files to scrape, '
               f'{totals["supplement_files_skipped"]} already done.')
    for fallback in plan['tokenizer_fallbacks']:
        click.echo(f'Warning: token counts for {fallback} are character estimates.', err=True)
    if plan['overflows']:
        click.echo(f'{len(plan["overflows"])} inputs need more than one request:')
        for item in plan['overflows'][:10]:
            outcome = '' if item['scraped'] else ' (over the limit; would be skipped)'
            click.echo(f'  {item["paper_id"]} {item["item"]}: {item["requests"]} requests, '
                       f'{item["text_tokens"]} tokens{outcome}')
        if len(plan['overflows']) > 10:
            click.echo(f'  ... {len(plan["overflows"]) - 10} more in --json output')


@agent_meta(effects={'paid_model', 'prompts', 'writes_corpus', 'writes_files'}, output=out.STORE)
@click.command('store')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('in_file', default='temp_scraped_materials.csv', type=click.Path())
@click.argument('out_file', default='materials.csv', type=click.Path())
@click.argument('recipe', default='sse', type=str)
@click.option('--assume-yes', is_flag=True, default=False,
              help='Store converted results without an interactive confirmation prompt.')
@_dry_run_option
@_json_option
def store(db_path: str, in_file: str, out_file: str, recipe: str, assume_yes: bool,
          dry_run: bool, json_output: bool) -> None:
    """Store temporary scrape results in the final materials CSV.

    With --dry-run the columns are matched and reported, but no unit is
    converted (no model call), nothing is written, and nothing prompts.
    """
    extra = {'dry_run': True} if dry_run else {}
    try:
        with _quiet(json_output):
            summary = store_results(db_path, in_file, out_file, True, recipe,
                                    assume_yes=assume_yes, **extra)
    except (RuntimeError, ValueError, KeyError, OSError, sqlite3.Error) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        raise click.ClickException(str(message)) from error
    if json_output:
        emit_json(summary)
    elif dry_run:
        click.echo(f'Dry run: would store {summary["rows_written"]} of {summary["rows_read"]} rows '
                   f'in {summary["output_path"]} ({summary["rows_total"]} rows in total), '
                   f'converting {len(summary["converted_columns"])} columns with the model and '
                   f'skipping {len(summary["skipped_columns"])} unmatched columns.')


@agent_meta(effects={'starts_server', 'writes_files'})
@click.command('gui')
@click.option('--validation', 'validation_path', type=click.Path(exists=True, dir_okay=False),
              help='Validation CSV to open immediately.')
@click.option('--scraped', 'scraped_path', type=click.Path(exists=True, dir_okay=False),
              help='Scraped CSV to open immediately.')
@click.option('--recipe', 'recipe_name', help='Bundled recipe name or recipe JSON path.')
@click.option('--output', type=click.Path(dir_okay=False),
              help='Review decisions JSON path; defaults to the local review directory.')
@click.option('--no-browser', is_flag=True, help='Print the local URL without opening a browser.')
def validation_gui(validation_path: str | None, scraped_path: str | None,
                   recipe_name: str | None, output: str | None, no_browser: bool) -> None:
    """Review recipe extraction against a validation CSV in a local browser."""
    supplied = [validation_path, scraped_path, recipe_name]
    if any(supplied) and not all(supplied):
        raise click.UsageError('Supply --validation, --scraped, and --recipe together, or choose all three in the browser.')
    app = ReviewApp(output=Path(output) if output else None)
    if all(supplied):
        try:
            validation_bytes = Path(validation_path).read_bytes()
            scraped_bytes = Path(scraped_path).read_bytes()
            app.load(validation_bytes.decode('utf-8-sig'),
                     scraped_bytes.decode('utf-8-sig'), recipe_name,
                     validation_name=validation_path, scraped_name=scraped_path,
                     validation_sha256=hashlib.sha256(validation_bytes).hexdigest(),
                     scraped_sha256=hashlib.sha256(scraped_bytes).hexdigest())
        except (OSError, ValueError, KeyError) as error:
            raise click.ClickException(str(error)) from error
    serve_validation(app, open_browser=not no_browser)


def _format_metric(value: float | None) -> str:
    """Format a precision, recall, or F1 value, which is null when undefined."""
    return '-' if value is None else f'{value:.3f}'


@agent_meta(effects={'writes_files'}, output=out.VALIDATE_SCORE)
@click.command('score')
@click.option('--review', 'review_path', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Review JSON saved by pmt validate gui; its inputs are restored from the '
                   'local copy the GUI keeps.')
@click.option('--validation', 'validation_path', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Validation CSV.')
@click.option('--scraped', 'scraped_path', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Scraped CSV.')
@click.option('--recipe', 'recipe_name', default=None, help='Bundled recipe name or recipe JSON path.')
@click.option('--decisions', 'decisions_path', default=None,
              type=click.Path(exists=True, dir_okay=False),
              help='Review JSON whose decisions apply to these inputs. Without it only suggested '
                   'pairs are scored, and every leaf stays pending.')
@click.option('--outfile', default=None, type=click.Path(dir_okay=False),
              help='Also write the full score JSON to this file.')
@_json_option
def validation_score(review_path: str | None, validation_path: str | None,
                     scraped_path: str | None, recipe_name: str | None,
                     decisions_path: str | None, outfile: str | None, json_output: bool) -> None:
    """Score a validation review without the browser.

    Give either --review (a review saved by the GUI), or --validation,
    --scraped, and --recipe with an optional --decisions file. Decisions only
    apply to the exact inputs they were made on: a changed CSV or recipe
    changes the fingerprint and is rejected.
    """
    direct = [validation_path, scraped_path, recipe_name]
    if review_path is not None:
        if any(direct) or decisions_path:
            raise click.UsageError('Use --review on its own, or --validation, --scraped, and '
                                   '--recipe without it.')
    elif not all(direct):
        raise click.UsageError('Give --review, or all of --validation, --scraped, and --recipe.')
    try:
        if review_path is not None:
            session, decisions = saved_review(review_path)
        else:
            validation_bytes = Path(validation_path).read_bytes()
            scraped_bytes = Path(scraped_path).read_bytes()
            session = review_session(
                validation_bytes.decode('utf-8-sig'), scraped_bytes.decode('utf-8-sig'), recipe_name,
                validation_name=validation_path, scraped_name=scraped_path,
                validation_sha256=hashlib.sha256(validation_bytes).hexdigest(),
                scraped_sha256=hashlib.sha256(scraped_bytes).hexdigest(),
            )
            decisions = {}
            if decisions_path is not None:
                saved = json.loads(Path(decisions_path).read_text(encoding='utf-8'))
                if saved.get('fingerprint') != session['fingerprint']:
                    raise ValueError(f'{decisions_path} was saved for different inputs; its '
                                     'decisions cannot be applied to these files.')
                decisions = saved.get('decisions', {})
        score = score_review(session, decisions)
        if outfile is not None:
            Path(outfile).write_text(f'{json.dumps(score, indent=2)}\n', encoding='utf-8')
    except (KeyError, OSError, UnicodeDecodeError, ValueError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        raise click.ClickException(str(message)) from error
    if json_output:
        emit_json(score)
        return
    click.echo(f'Recipe {score["recipe"]}: {score["papers_evaluated"]} validation papers, '
               f'{score["pending_pairs"]} pending pairs; complete: {"yes" if score["complete"] else "no"}.')
    if score['missing_metadata']:
        click.echo(f'Missing run metadata: {", ".join(score["missing_metadata"])}.')
    click.echo(f'{"Field":30} {"P":>6} {"R":>6} {"F1":>6} {"TP":>5} {"FP":>5} {"FN":>5} {"pend":>5}')
    for field, metric in score['fields'].items():
        click.echo(f'{field[:30]:30} {_format_metric(metric.get("precision")):>6} '
                   f'{_format_metric(metric.get("recall")):>6} {_format_metric(metric.get("f1")):>6} '
                   f'{metric.get("tp", 0):>5} {metric.get("fp", 0):>5} {metric.get("fn", 0):>5} '
                   f'{metric.get("pending", 0):>5}')
    if outfile is not None:
        click.echo(f'Score written to {outfile}.')


@agent_meta(effects={'read_only'}, output=out.RECIPE_LIST)
@click.command('list')
@_json_option
def recipe_list(json_output: bool) -> None:
    """List bundled recipe names without making model requests."""
    try:
        names = sorted(json.loads(RECIPES_PATH.read_text(encoding='utf-8')))
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(names)
    else:
        click.echo('\n'.join(names))


@agent_meta(effects={'read_only'}, output=out.RECIPE)
@click.command('show')
@click.argument('recipe', default='sse')
def recipe_show(recipe: str) -> None:
    """Write a validated bundled recipe or recipe file as JSON."""
    try:
        loaded = load_recipe(recipe)
    except (KeyError, ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error
    emit_json(loaded)


@agent_meta(effects={'writes_files'})
@click.command('prompt')
@click.argument('recipe', default='sse', type=str)
@click.option(
    '--kind',
    type=click.Choice(['text', 'image', 'image-context', 'reconciliation']),
    default='text',
    show_default=True,
    help='Prompt variant to render.',
)
@click.option('--outfile', default=None, type=click.Path(dir_okay=False),
              help='Write the prompt to this file instead of standard output.')
def recipe_prompt(recipe: str, kind: str, outfile: str | None) -> None:
    """Render the exact LLM system prompt generated from a recipe."""
    try:
        loaded = load_recipe(recipe)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if kind == 'text':
        prompt = build_text_extraction_prompt(loaded)
    elif kind == 'image':
        prompt = build_image_extraction_prompt(loaded)
    elif kind == 'image-context':
        prompt = build_image_extraction_prompt(loaded, with_context=True)
    else:
        prompt = build_reconciliation_prompt(loaded)

    if outfile is None:
        click.echo(prompt)
        return
    try:
        Path(outfile).write_text(prompt + '\n', encoding='utf-8')
    except OSError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f'Prompt written to {outfile}.')


@agent_meta(effects={'writes_files'}, output=out.VALIDATE_TEMPLATE)
@click.command('template')
@click.argument('recipe', type=str)
@click.argument('outfile', type=click.Path(dir_okay=False))
@_json_option
def validation_template(recipe: str, outfile: str, json_output: bool) -> None:
    """Create an empty validation CSV for a bundled or file-based recipe."""
    try:
        loaded = load_recipe(recipe)
        columns = ['Identifier', 'Title', 'DOI']
        existing = {column.casefold() for column in columns}
        columns.extend(
            field for field in loaded['search fields']
            if field.casefold() not in existing
        )
        with Path(outfile).open('w', encoding='utf-8', newline='') as handle:
            csv.writer(handle).writerow(columns)
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json({'recipe': recipe, 'outfile': outfile, 'columns': columns})
        return
    click.echo(f'Validation template written to {outfile}.')


# Each setter: command name, settings key, label, secret, validated online, helper.
_SETTERS = [
    ('elsevier-key', 'elsevier_api_key', 'Elsevier API key', True, True, update_elsevier_key),
    ('core-key', 'core_api_key', 'CORE API key', True, False, update_core_key),
    ('core-rate', 'core_min_interval', 'seconds between CORE requests', False, False,
     update_core_rate),
    ('unpaywall-email', 'unpaywall_email', 'Unpaywall email', False, False, update_unpaywall_email),
    ('crossref-email', 'crossref_email', 'Crossref contact email', False, False,
     update_crossref_email),
    ('openalex-key', 'openalex_api_key', 'OpenAlex API key', True, True, update_openalex_key),
    ('ncbi-key', 'ncbi_api_key', 'NCBI E-utilities API key', True, False, update_ncbi_key),
    ('ncbi-email', 'ncbi_email', 'NCBI E-utilities contact email', False, False, update_ncbi_email),
    ('openai-key', 'openai_api_key', 'OpenAI API key', True, True, update_openai_key),
    ('anthropic-key', 'anthropic_api_key', 'Anthropic API key', True, False, update_anthropic_key),
]


def _value_without_prompt(env_var: str | None, from_stdin: bool) -> tuple[str | None, str]:
    """Read a setting from --value-from-env or --value-stdin, or decide to prompt.

    Returns the value and its source: ``env``, ``stdin``, or ``prompt`` (value
    ``None``). Prompting is refused when stdin is not a terminal, so an
    unattended caller gets a usage error instead of a hang.
    """
    if env_var and from_stdin:
        raise click.UsageError('Use --value-from-env or --value-stdin, not both.')
    if env_var:
        if env_var not in os.environ:
            raise click.BadParameter(f'Environment variable {env_var} is not set.',
                                     param_hint='--value-from-env')
        return os.environ[env_var], 'env'
    if from_stdin:
        line = sys.stdin.readline()
        if not line:
            raise click.BadParameter('Standard input was empty.', param_hint='--value-stdin')
        return line.rstrip('\r\n'), 'stdin'
    if not sys.stdin.isatty():
        raise click.UsageError('No value given and stdin is not a terminal: pass '
                               '--value-from-env VAR or --value-stdin.')
    return None, 'prompt'


def _setter_command(name: str, key: str, label: str, secret: bool, validated: bool,
                    setter: Callable[..., None]) -> click.Command:
    """Build the ``pmt config`` command that saves one provider setting."""
    effects = {'prompts', 'writes_config', *({'network'} if validated else ())}
    checked = ' The value is checked with the provider before it is saved.' if validated else ''

    @agent_meta(effects=effects, output=out.CONFIG_SETTING)
    @click.command(name, help=(
        f'Save the {label}. Without --value-from-env or --value-stdin it prompts, '
        f'which needs a terminal.{checked}'))
    @click.option('--value-from-env', 'env_var', default=None, metavar='VAR',
                  help='Read the value from this environment variable.')
    @click.option('--value-stdin', 'from_stdin', is_flag=True,
                  help='Read the value from the first line of standard input.')
    @_json_option
    def command(env_var: str | None, from_stdin: bool, json_output: bool) -> None:
        """Save one setting from the environment, stdin, or a prompt."""
        value, source = _value_without_prompt(env_var, from_stdin)
        try:
            with _quiet(json_output):
                setter(value=value)
        except (OSError, RuntimeError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        result: dict[str, Any] = {'setting': key, 'secret': secret, 'source': source}
        if not secret:
            try:
                result['value'] = _saved_settings().get(key)
            except (OSError, RuntimeError, ValueError):
                result['value'] = None
        if json_output:
            emit_json(result)
        else:
            click.echo(f'Saved the {label}.')

    return command


@agent_meta(effects={'prompts', 'writes_config'}, output=out.CONFIG_MODEL)
@click.command('model')
@click.argument('profile', default='text', type=click.Choice(['text', 'vision']))
@click.option('--provider', default=None,
              help='Provider: openai, anthropic, or local. Prompted for only at a terminal.')
@click.option('--model', default=None, help='Model name. Prompted for only at a terminal.')
@click.option('--base-url', default=None, help='Base URL for local providers.')
@click.option('--api-key', default=None,
              help='Provider API key. Prefer --api-key-from-env, which keeps the key out of '
                   'the process list. Leave unset for env/provider defaults.')
@click.option('--api-key-from-env', 'api_key_env', default=None, metavar='VAR',
              help='Read the provider API key from this environment variable.')
@click.option('--capability', 'capabilities',
              multiple=True,
              help='Optional override. By default capabilities are inferred from profile/model name.')
@click.option('--temperature',
              default=0.0,
              type=float,
              show_default=True,
              help='Sampling temperature for model requests.')
@click.option('--top-p',
              default=1.0,
              type=float,
              show_default=True,
              help='Nucleus sampling probability mass for model requests.')
@click.option('--input-token-limit',
              default=DEFAULT_INPUT_TOKEN_LIMIT,
              type=int,
              show_default=True,
              help='Maximum input tokens to send to the model before chunking.')
@_json_option
def model_config(profile: str, provider: str | None, model: str | None, base_url: str | None,
                 api_key: str | None, api_key_env: str | None,
                 capabilities: tuple[str, ...], temperature: float, top_p: float,
                 input_token_limit: int, json_output: bool) -> None:
    """Configure a text or vision model profile from CLI options.

    --provider and --model are prompted for only when stdin is a terminal;
    otherwise a missing one is a usage error.
    """
    missing = [option for option, value in [('--provider', provider), ('--model', model)]
               if not value]
    if missing and not sys.stdin.isatty():
        raise click.UsageError(f'Missing {" and ".join(missing)}; stdin is not a terminal, so '
                               'they cannot be prompted for.')
    provider = provider or click.prompt('Provider')
    model = model or click.prompt('Model')
    if api_key and api_key_env:
        raise click.UsageError('Use --api-key or --api-key-from-env, not both.')
    if api_key_env:
        if api_key_env not in os.environ:
            raise click.BadParameter(f'Environment variable {api_key_env} is not set.',
                                     param_hint='--api-key-from-env')
        api_key = os.environ[api_key_env]
    caps = list(capabilities) or infer_model_capabilities(profile, model)
    try:
        set_model_profile(
            profile,
            provider,
            model,
            base_url,
            api_key,
            caps,
            temperature=temperature,
            top_p=top_p,
            input_token_limit=input_token_limit,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json({'profile': profile, 'provider': provider, 'model': model, 'base_url': base_url,
                   'capabilities': caps, 'temperature': temperature, 'top_p': top_p,
                   'input_token_limit': input_token_limit, 'api_key_set': bool(api_key)})
        return
    click.echo(
        f'Updated {profile} model profile: {provider}/{model} [{", ".join(caps)}] temperature={temperature} top_p={top_p} input_token_limit={input_token_limit}')


@agent_meta(effects={'read_only'}, output=out.MODEL_STATUS)
@click.command('status')
@_json_option
def model_status(json_output: bool) -> None:
    """Print configured text and vision model profiles."""
    try:
        profiles = {
            profile: {key: value for key, value in get_model_profile(profile).items()
                      if key != 'api_key'}
            for profile in ['text', 'vision']
        }
    except (RuntimeError, ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(profiles)
        return
    for profile, config in profiles.items():
        capabilities = ', '.join(config.get('capabilities', []))
        click.echo(
            f'{profile}: {config.get("provider")}/{config.get("model")} capabilities=[{capabilities}] temperature={config.get("temperature")} top_p={config.get("top_p")} input_token_limit={config.get("input_token_limit")} base_url={config.get("base_url")}')


@agent_meta(effects={'network'}, output=out.PROVIDERS)
@click.command('providers')
@click.option('--no-probe', is_flag=True,
              help='Report what is configured without making any requests.')
@click.option('--source', 'sources', multiple=True,
              type=click.Choice(['all', *sources.SOURCES]),
              help='Report on one provider. Repeat for several.')
@click.option('--json', 'json_output', is_flag=True,
              help='Emit machine-readable JSON with provider rows and an ok flag.')
def provider_status_command(no_probe: bool, sources: tuple[str, ...],
                            json_output: bool) -> None:
    """Show which providers are set up and which are answering.

    Requests respect each provider's rate limiter. Checking openalex-content
    costs 100 OpenAlex credits when its key is configured. Exits non-zero when
    a configured provider is not answering, so it can gate a script.
    """
    report_provider_status(no_probe, sources, json_output=json_output)


@agent_meta(effects={'writes_corpus'}, output=out.RESET)
@click.command('reset')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@_dry_run_option
@_json_option
def reset_miner(db_path: str, dry_run: bool, json_output: bool) -> None:
    """Reset pipeline statuses in the paper corpus.

    With --dry-run the reset runs on a temporary copy of the corpus, which
    is then deleted, and reports what it would change.
    """
    try:
        with _quiet(json_output):
            summary = _on_copy(db_path, reset) if dry_run else reset(db_path)
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        emit_json(summary)
    else:
        verb = 'Would reset' if dry_run else 'Reset'
        click.echo(f'{verb} {summary["papers_reset"]} papers and '
                   f'{summary["supplement_stages_reset"]} supporting-information stages.')


@agent_meta(effects={'migrates_corpus'}, output=out.STATUS)
@click.command('status')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True, dir_okay=False))
@_json_option
def miner_status(db_path: str, json_output: bool) -> None:
    """Print pipeline progress for the paper corpus."""
    try:
        if json_output:
            emit_json(progress_summary(db_path))
        else:
            status(db_path)
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error


@click.group('pmt', cls=AgentGroup)
@click.version_option(__version__, prog_name='pmt')
def main() -> None:
    """Build, analyse, and extract structured data from paper corpora."""


@click.group('corpus')
def corpus_group() -> None:
    """Inspect and maintain a paper corpus."""


@click.group('filter')
def filter_group() -> None:
    """Apply, inspect, and reset corpus filters."""


@click.group('topics')
def topics_group() -> None:
    """Train, inspect, and apply LDA topic models."""


@click.group('import')
def import_group() -> None:
    """Import existing papers or an author's publication list."""


@click.group('entities')
def entities_group() -> None:
    """Extract named entities with fine-tuned BERT and transformer checkpoints."""


@click.group('config')
def config_group() -> None:
    """Configure model profiles and provider credentials."""


@click.group('recipe')
def recipe_group() -> None:
    """Inspect extraction recipes and their generated prompts."""


@click.group('validate')
def validate_group() -> None:
    """Create validation sets and review extraction results."""


main.add_command(paper_search, 'search')
main.add_command(schema_command)
main.add_command(probe)
main.add_command(gather)
main.add_command(download, 'download')
main.add_command(enrich, 'enrich')
main.add_command(scrape, 'scrape')
main.add_command(scrape_progress_command, 'scrape-progress')
main.add_command(store, 'store')
main.add_command(digitize, 'digitize')
main.add_command(miner_status, 'status')
main.add_command(reset_miner, 'reset')

corpus_group.add_command(corpus_status, 'stats')
corpus_group.add_command(corpus_searches, 'searches')
corpus_group.add_command(corpus_list)
corpus_group.add_command(corpus_show)
corpus_group.add_command(corpus_export)
main.add_command(corpus_group)

filter_group.add_command(filter_regex, 'regex')
filter_group.add_command(filter_topic, 'topic')
filter_group.add_command(filter_status, 'status')
filter_group.add_command(filter_reset, 'reset')
main.add_command(filter_group)

topics_group.add_command(topics_train, 'train')
topics_group.add_command(topics_compare, 'compare')
topics_group.add_command(topics_show, 'show')
topics_group.add_command(topics_name, 'name')
topics_group.add_command(topics_predict, 'predict')
topics_group.add_command(topics_trends, 'trends')
topics_group.add_command(topics_store, 'store')
topics_group.add_command(topics_models, 'models')
main.add_command(topics_group)

entities_group.add_command(entities_corpus, 'corpus')
entities_group.add_command(entities_text, 'text')
main.add_command(entities_group)

import_group.add_command(import_pdf_folder, 'pdfs')
import_group.add_command(import_author, 'author')
main.add_command(import_group)

config_group.add_command(model_config, 'model')
config_group.add_command(model_status, 'status')
config_group.add_command(provider_status_command, 'providers')
for _name, _key, _label, _secret, _validated, _setter in _SETTERS:
    config_group.add_command(_setter_command(_name, _key, _label, _secret, _validated, _setter))
main.add_command(config_group)

recipe_group.add_command(recipe_prompt, 'prompt')
recipe_group.add_command(recipe_list)
recipe_group.add_command(recipe_show)
main.add_command(recipe_group)

validate_group.add_command(validation_gui, 'gui')
validate_group.add_command(validation_template, 'template')
validate_group.add_command(validation_score, 'score')
main.add_command(validate_group)
