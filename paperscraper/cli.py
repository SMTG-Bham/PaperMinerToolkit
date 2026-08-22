"""Command-line entry points for PaperScraper workflows.

This module maps installed ``ps_*`` commands to the underlying search, import,
download, scrape, store, configuration, and maintenance functions.
"""

from __future__ import annotations

from typing import Any, Mapping

import click
from paperscraper.corpus import connect, corpus_stats, enrichment_stats
from paperscraper.crossref import import_author_works
from paperscraper.search import search_for_papers
from paperscraper.compression import COMPRESSION_MODES, COMPRESSION_SCOPES
from paperscraper.download import download_papers
from paperscraper.enrichment import enrich_corpus
from paperscraper.filtering import (apply_regex_filter,
                                    apply_topic_filter,
                                    filter_overview,
                                    reset_filters)
from paperscraper.imports import import_pdfs
from paperscraper.scrape import SCRAPE_ORDERS, scrape_papers
from paperscraper.store import store_results
from paperscraper.topics import (aggregate_topic_trends,
                                 compare_topic_models,
                                 predict_topic_model,
                                 set_topic_name,
                                 store_topic_model_scores,
                                 stored_topic_models,
                                 topic_descriptions,
                                 train_topic_model)
from paperscraper.settings import (get_model_profile,
                                   DEFAULT_INPUT_TOKEN_LIMIT,
                                   infer_model_capabilities,
                                   set_model_profile,
                                   update_anthropic_key,
                                   update_core_key,
                                   update_crossref_email,
                                   update_elsevier_key,
                                   update_ncbi_email,
                                   update_ncbi_key,
                                   update_openai_key,
                                   update_openalex_key,
                                   update_unpaywall_email)
from paperscraper.utilities import reset, status


def _format_bytes(size: int) -> str:
    """Format a byte count using compact binary units."""
    value = float(size)
    for unit in ['B', 'KiB', 'MiB', 'GiB']:
        if value < 1024 or unit == 'GiB':
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024


@click.command()
@click.argument('query', default='Lithium solid electrolyte', type=str)
@click.argument('db_path', default='papers.db', type=click.Path())
@click.option('--source',
              type=click.Choice(['all', 'elsevier', 'core', 'openalex', 'pubmed']),
              default='all',
              show_default=True,
              help='Search source to use.')
@click.option('--count',
              default=200,
              type=int,
              show_default=True,
              help='Maximum results to request from each selected source.')
@click.option('--store-abstract',
              is_flag=True,
              default=False,
              help='Store abstracts returned by search providers as corpus assets.')
@click.option('--enrich', 'enrich_metadata',
              is_flag=True,
              default=False,
              help='Supplement newly stored papers with Crossref and OpenAlex metadata.')
def paper_search(query: str, db_path: str, source: str, count: int,
                 store_abstract: bool, enrich_metadata: bool) -> None:
    """Search configured paper sources and merge results into the paper corpus."""
    search_for_papers(query, db_path, source=source, count=count,
                      store_abstract=store_abstract, enrich=enrich_metadata)


@click.command()
@click.argument('dir', default='papers', type=click.Path(exists=True, file_okay=False))
@click.argument('db_path', default='papers.db', type=click.Path())
@click.option('--no-crossref',
              is_flag=True,
              default=False,
              help='Only scrape DOI from PDFs; skip Crossref metadata lookup.')
def import_pdf_folder(dir: str, db_path: str, no_crossref: bool) -> None:
    """Import local PDFs into the paper corpus, optionally skipping Crossref lookup."""
    import_pdfs(dir, db_path, use_crossref=not no_crossref)


@click.command()
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
              help='Supplement imported works with Crossref and OpenAlex metadata.')
def import_author(db_path: str,
                  email: str | None,
                  orcid: str | None,
                  author_name: str | None,
                  affiliation: str | None,
                  max_results: int,
                  review_csv: str,
                  enrich_metadata: bool) -> None:
    """Import one author's DOI-bearing Crossref works into a corpus."""
    if bool(orcid) == bool(author_name):
        raise click.UsageError('Provide exactly one of --orcid or --author.')
    try:
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
    click.echo(
        f'Crossref found {summary["found"]} matching works: '
        f'{summary["added"]} added and {summary["updated"]} updated in {db_path}.'
    )
    if enrich_metadata:
        click.echo(f'Enriched {summary["enriched"]} imported works.')
    click.echo(f'Review the imported metadata in {review_csv} before downloading papers.')


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--format', 'download_format',
              type=click.Choice(['abstract', 'text', 'pdf', 'both']),
              default='both',
              show_default=True)
@click.option('--source', 'sources',
              multiple=True,
              type=click.Choice(['all', 'unpaywall', 'core', 'elsevier', 'openalex', 'pubmed']),
              default=('all',),
              show_default=True,
              help='Content source to use for PDFs and PubMed Central full text. '
                   'Repeat to choose more than one.')
@click.option('--abstract/--no-abstract',
              'download_abstract',
              default=True,
              show_default=True,
              help='Download and store abstracts alongside requested paper assets.')
@click.option('--force', is_flag=True, default=False,
              help='Redownload requested assets even when that content type is already stored.')
def download(
        db_path: str,
        download_format: str,
        sources: tuple[str, ...],
        download_abstract: bool,
        force: bool,
) -> None:
    """Download abstracts, text, and/or PDFs for rows in the paper corpus."""
    download_papers(
        db_path,
        download_format=download_format,
        sources=list(sources),
        download_abstract=download_abstract,
        force=force,
    )


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--source', 'sources',
              multiple=True,
              type=click.Choice(['all', 'crossref', 'openalex', 'pubmed']),
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
def enrich(db_path: str,
           sources: tuple[str, ...],
           batch_size: int,
           limit: int | None,
           force: bool,
           retry_failed: bool,
           refresh_after: int,
           references: bool,
           resolve_references: bool,
           email: str | None) -> None:
    """Supplement corpus metadata with Crossref and OpenAlex records."""
    try:
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
        )
    except (RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
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
            f'{summary["unresolved"]} papers have no DOI or OpenAlex identifier and were skipped.',
            err=True,
        )


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
def corpus_status(db_path: str) -> None:
    """Print storage statistics for the paper corpus."""
    with connect(db_path) as conn:
        stats = corpus_stats(conn)
        enrichment = enrichment_stats(conn)
    click.echo(f'Corpus: {db_path}')
    click.echo(f'Papers: {stats["papers"]}')
    click.echo(f'Papers with abstracts: {stats["papers_with_abstract"]}')
    click.echo(f'Papers with text: {stats["papers_with_text"]}')
    click.echo(f'Papers with PDFs: {stats["papers_with_pdf"]}')
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
            + '. Run ps_topics_store again before scraping.'
        )


@click.command()
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
def filter_regex(db_path: str, rules_path: str, fields: tuple[str, ...],
                 join_operator: str | None, replace: bool, timeout_ms: int | None) -> None:
    """Apply a named post-download regex filter to a paper corpus."""
    try:
        overview = apply_regex_filter(
            db_path,
            rules_path,
            fields=fields or None,
            join_operator=join_operator,
            replace=replace,
            timeout_ms=timeout_ms,
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    _echo_filter_overview(db_path, overview)


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('rules_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--join', 'join_operator', type=click.Choice(['and', 'or']), default=None,
              help='Join this filter to the preceding active expression.')
@click.option('--replace', is_flag=True, default=False,
              help='Replace and reevaluate an active filter with the same name.')
def filter_topic(db_path: str, rules_path: str,
                 join_operator: str | None, replace: bool) -> None:
    """Apply a named stored-model topic filter to a paper corpus."""
    try:
        overview = apply_topic_filter(
            db_path, rules_path, join_operator=join_operator, replace=replace
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    _echo_filter_overview(db_path, overview)


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
def filter_status(db_path: str) -> None:
    """Show active corpus filters and their final paper decisions."""
    with connect(db_path) as conn:
        overview = filter_overview(conn)
    _echo_filter_overview(db_path, overview)


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--name', default=None, help='Remove one active filter by name.')
@click.option('--all', 'all_filters', is_flag=True, default=False,
              help='Remove every active filter.')
def filter_reset(db_path: str, name: str | None, all_filters: bool) -> None:
    """Remove one or all active corpus filters."""
    try:
        overview = reset_filters(db_path, name=name, all_filters=all_filters)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    _echo_filter_overview(db_path, overview)


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('model_dir', default='topic_model', type=click.Path())
@click.option('--topics', 'num_topics', default=10, type=click.IntRange(min=2), show_default=True)
@click.option('--field', 'text_fields', multiple=True,
              type=click.Choice(['title', 'abstract', 'text']),
              default=('title', 'abstract'), show_default=True,
              help='Corpus text field to model. Repeat to combine fields.')
@click.option('--min-df', default=2, type=click.IntRange(min=1), show_default=True,
              help='Minimum number of documents containing a retained term.')
@click.option('--max-df', default=0.95, type=click.FloatRange(min=0.0, max=1.0, min_open=True),
              show_default=True, help='Maximum fraction of documents containing a retained term.')
@click.option('--max-features', default=20000, type=click.IntRange(min=2), show_default=True)
@click.option('--learning-method', type=click.Choice(['online', 'batch']), default='online', show_default=True)
@click.option('--iterations', 'max_iter', default=20, type=click.IntRange(min=1), show_default=True)
@click.option('--random-seed', 'random_state', default=0, type=int, show_default=True)
@click.option('--top-terms', default=15, type=click.IntRange(min=1), show_default=True)
@click.option('--representative-papers', default=5, type=click.IntRange(min=1), show_default=True)
@click.option('--stopwords-file', default=None, type=click.Path(exists=True, dir_okay=False),
              help='UTF-8 file containing one corpus-specific stopword per line.')
@click.option('--ngram-max', default=2, type=click.IntRange(min=1, max=2), show_default=True,
              help='Largest generated n-gram; use 2 to include bigrams.')
@click.option('--overwrite', is_flag=True, default=False,
              help='Replace known model artifact files in a non-empty model directory.')
@click.option('--streaming/--in-memory', default=True, show_default=True,
              help='Use disk-backed bounded batches or materialize the corpus in memory.')
@click.option('--batch-size', default=128, type=click.IntRange(min=1), show_default=True)
@click.option('--cache-dir', default=None, type=click.Path(file_okay=False),
              help='Parent directory for temporary streaming caches.')
@click.option('--evaluation-sample-size', default=10000,
              type=click.IntRange(min=1), show_default=True)
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
                 evaluation_sample_size: int) -> None:
    """Train an LDA model and write inspectable, manually nameable artifacts."""
    try:
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
    click.echo(
        f'Trained {num_topics} topics from {summary["report"]["documents_used"]} papers '
        f'using {summary["report"]["vocabulary_size"]} terms.'
    )
    click.echo(f'Model artifacts: {summary["model_dir"]}')
    click.echo(f'Manual topic review: {summary["model_dir"]}/topics.csv')


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('output_dir', default='topic_comparison', type=click.Path())
@click.option('--topics', 'topic_counts', multiple=True, type=click.IntRange(min=2),
              default=(5, 10), show_default=True,
              help='Topic count to train. Repeat to compare several values.')
@click.option('--seed', 'random_states', multiple=True, type=int, default=(0, 1), show_default=True,
              help='Random seed to train. Repeat to assess stability.')
@click.option('--field', 'text_fields', multiple=True,
              type=click.Choice(['title', 'abstract', 'text']),
              default=('title', 'abstract'), show_default=True,
              help='Corpus text field to model. Repeat to combine fields.')
@click.option('--min-df', default=2, type=click.IntRange(min=1), show_default=True)
@click.option('--max-df', default=0.95, type=click.FloatRange(min=0.0, max=1.0, min_open=True),
              show_default=True)
@click.option('--max-features', default=20000, type=click.IntRange(min=2), show_default=True)
@click.option('--learning-method', type=click.Choice(['online', 'batch']), default='online', show_default=True)
@click.option('--iterations', 'max_iter', default=20, type=click.IntRange(min=1), show_default=True)
@click.option('--top-terms', default=15, type=click.IntRange(min=1), show_default=True)
@click.option('--representative-papers', default=5, type=click.IntRange(min=1), show_default=True)
@click.option('--stopwords-file', default=None, type=click.Path(exists=True, dir_okay=False),
              help='UTF-8 file containing one corpus-specific stopword per line.')
@click.option('--ngram-max', default=2, type=click.IntRange(min=1, max=2), show_default=True)
@click.option('--overwrite', is_flag=True, default=False,
              help='Replace known comparison and model artifact files.')
@click.option('--streaming/--in-memory', default=True, show_default=True,
              help='Use one reusable disk-backed corpus cache or in-memory matrices.')
@click.option('--batch-size', default=128, type=click.IntRange(min=1), show_default=True)
@click.option('--cache-dir', default=None, type=click.Path(file_okay=False),
              help='Parent directory for temporary streaming caches.')
@click.option('--evaluation-sample-size', default=10000,
              type=click.IntRange(min=1), show_default=True)
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
                   evaluation_sample_size: int) -> None:
    """Train several LDA configurations and export comparable diagnostics."""
    try:
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
    click.echo(f'Trained {summary["models_trained"]} comparison models in {summary["output_dir"]}.')
    click.echo(f'Comparison metrics: {summary["comparison_csv"]}')
    click.echo('Inspect each model with ps_topics_show before choosing one.')


@click.command()
@click.argument('model_dir', default='topic_model', type=click.Path(exists=True, file_okay=False))
@click.option('--representatives', default=3, type=click.IntRange(min=0), show_default=True,
              help='Representative paper titles to print for each topic.')
def topics_show(model_dir: str, representatives: int) -> None:
    """Print top terms and representative papers for manual topic naming."""
    try:
        topics = topic_descriptions(model_dir)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    for topic in topics:
        name = topic['topic_name'] or '(unnamed)'
        click.echo(f'Topic {topic["topic_id"]}: {name}')
        click.echo(f'  Terms: {", ".join(topic["top_terms"])}')
        for paper in topic['representative_papers'][:representatives]:
            click.echo(f'  - {paper["title"]} ({float(paper["probability"]):.3f})')


@click.command()
@click.argument('model_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('topic_id', type=click.IntRange(min=0))
@click.argument('topic_name', type=str)
def topics_name(model_dir: str, topic_id: int, topic_name: str) -> None:
    """Assign a manual human-readable name to one fitted topic."""
    try:
        set_topic_name(model_dir, topic_id, topic_name)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f'Named topic {topic_id}: {topic_name.strip()}')


@click.command()
@click.argument('model_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('output_path', default='paper_topics.csv', type=click.Path())
@click.option('--batch-size', default=128, type=click.IntRange(min=1), show_default=True)
def topics_predict(model_dir: str, db_path: str, output_path: str, batch_size: int) -> None:
    """Apply a saved LDA model to a corpus and export topic probabilities."""
    try:
        summary = predict_topic_model(
            model_dir, db_path, output_path, batch_size=batch_size
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f'Predicted topics for {summary["papers_predicted"]} of {summary["papers_total"]} papers; '
        f'{summary["papers_without_vocabulary_terms"]} had no model vocabulary terms.'
    )
    click.echo(f'Topic scores: {summary["output_path"]}')


@click.command()
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
def topics_trends(model_dir: str, output_dir: str, predictions_path: str | None,
                  bin_size: int, step_size: int, start_year: int | None,
                  end_year: int | None, include_partial: bool, plot: bool,
                  plot_file: str | None,
                  overwrite: bool) -> None:
    """Aggregate fixed-model topic probabilities over publication-year windows."""
    try:
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
    click.echo(f'Wrote {summary["windows"]} topic trend windows to {summary["trends_csv"]}.')
    if summary['plot_path']:
        click.echo(f'Topic trend plot: {summary["plot_path"]}')
    if summary['papers_missing_or_invalid_date']:
        click.echo(
            f'{summary["papers_missing_or_invalid_date"]} papers had missing or invalid dates.',
            err=True,
        )


@click.command()
@click.argument('model_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--name', default=None,
              help='Immutable corpus display name; defaults to the model directory name.')
@click.option('--batch-size', default=128, type=click.IntRange(min=1), show_default=True)
def topics_store(model_dir: str, db_path: str, name: str | None, batch_size: int) -> None:
    """Predict afresh and transactionally store a topic model in the corpus."""
    try:
        summary = store_topic_model_scores(
            model_dir, db_path, name=name, batch_size=batch_size
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f'Stored {summary["name"]} ({summary["model_id"]}): '
        f'{summary["papers_predicted"]} predicted, '
        f'{summary["papers_without_vocabulary_terms"]} without vocabulary terms.'
    )


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.option('--batch-size', default=128, type=click.IntRange(min=1), show_default=True)
def topics_models(db_path: str, batch_size: int) -> None:
    """List topic models and prediction freshness recorded in a corpus."""
    models = stored_topic_models(db_path, batch_size=batch_size)
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


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('recipe', default='sse', type=str)
@click.option('--mode', type=click.Choice(['abstract', 'text', 'images', 'text-images']), default='text', show_default=True)
@click.option('--image-context', type=click.Choice(['none', 'paper-text']), default='none', show_default=True)
@click.option('--image-dir', default='paper_images', type=click.Path(), show_default=True)
@click.option('--image-extraction',
              type=click.Choice(['auto', 'embedded', 'pages']),
              default='auto',
              show_default=True,
              help='How to turn PDFs into images for vision analysis.')
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
def scrape(
        db_path: str,
        recipe: str,
        mode: str,
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
) -> None:
    """Run text and/or image scraping over downloaded papers."""
    scrape_papers(
        db_path,
        recipe,
        mode=mode,
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
    )


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
@click.argument('in_file', default='temp_scraped_materials.csv', type=click.Path())
@click.argument('out_file', default='materials.csv', type=click.Path())
@click.argument('recipe', default='sse', type=str)
@click.option('--assume-yes', is_flag=True, default=False,
              help='Store converted results without an interactive confirmation prompt.')
def store(db_path: str, in_file: str, out_file: str, recipe: str, assume_yes: bool) -> None:
    """Store temporary scrape results in the final materials CSV."""
    store_results(db_path, in_file, out_file, True, recipe, assume_yes=assume_yes)


def update_elsevier_api_key() -> None:
    """Prompt for and save an Elsevier API key."""
    update_elsevier_key()


def update_core_api_key() -> None:
    """Prompt for and save a CORE API key."""
    update_core_key()


def update_unpaywall_api_email() -> None:
    """Prompt for and save an Unpaywall email address."""
    update_unpaywall_email()


def update_crossref_api_email() -> None:
    """Prompt for and save a Crossref contact email address."""
    update_crossref_email()


def update_openalex_api_key() -> None:
    """Prompt for and save an OpenAlex API key."""
    update_openalex_key()


def update_ncbi_api_key() -> None:
    """Prompt for and save an NCBI E-utilities API key."""
    update_ncbi_key()


def update_ncbi_api_email() -> None:
    """Prompt for and save an NCBI E-utilities contact email address."""
    update_ncbi_email()


def update_openai_api_key() -> None:
    """Prompt for and save an OpenAI API key."""
    update_openai_key()


def update_anthropic_api_key() -> None:
    """Prompt for and save an Anthropic API key."""
    update_anthropic_key()


@click.command()
@click.argument('profile', default='text', type=click.Choice(['text', 'vision']))
@click.option('--provider', prompt=True, help='Provider: openai, anthropic, or local.')
@click.option('--model', prompt=True, help='Model name.')
@click.option('--base-url', default=None, help='Base URL for local providers.')
@click.option('--api-key', default=None, help='Provider API key. Leave unset for env/provider defaults.')
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
def model_config(profile: str, provider: str, model: str, base_url: str | None, api_key: str | None,
                 capabilities: tuple[str, ...], temperature: float, top_p: float, input_token_limit: int) -> None:
    """Configure a text or vision model profile from CLI options."""
    caps = list(capabilities) or infer_model_capabilities(profile, model)
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
    click.echo(
        f'Updated {profile} model profile: {provider}/{model} [{", ".join(caps)}] temperature={temperature} top_p={top_p} input_token_limit={input_token_limit}')


@click.command()
def model_status() -> None:
    """Print configured text and vision model profiles."""
    for profile in ['text', 'vision']:
        config = get_model_profile(profile)
        capabilities = ', '.join(config.get('capabilities', []))
        click.echo(
            f'{profile}: {config.get("provider")}/{config.get("model")} capabilities=[{capabilities}] temperature={config.get("temperature")} top_p={config.get("top_p")} input_token_limit={config.get("input_token_limit")} base_url={config.get("base_url")}')


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
def reset_scraper(db_path: str) -> None:
    """Reset pipeline statuses in the paper corpus."""
    reset(db_path)


@click.command()
@click.argument('db_path', default='papers.db', type=click.Path(exists=True))
def scraper_status(db_path: str) -> None:
    """Print pipeline progress for the paper corpus."""
    status(db_path)
