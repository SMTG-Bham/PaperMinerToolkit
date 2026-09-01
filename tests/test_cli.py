"""Test the PaperMinerToolkit command-line entry points."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, NoReturn

import pytest
from click.testing import CliRunner

import paperminertoolkit.cli as cli
import paperminertoolkit.corpus.database as corpus


def test_package_installs_only_the_nested_pmt_entry_point() -> None:
    """Replace the legacy underscore scripts with one discoverable command."""
    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
    assert project['project']['scripts'] == {'pmt': 'paperminertoolkit.cli:main'}


def test_main_command_exposes_discoverable_nested_groups() -> None:
    """Expose one executable whose group help lists the available workflows."""
    runner = CliRunner()

    root_help = runner.invoke(cli.main, ['--help'])
    filter_help = runner.invoke(cli.main, ['filter', '--help'])
    topics_help = runner.invoke(cli.main, ['topics', '--help'])
    recipe_help = runner.invoke(cli.main, ['recipe', '--help'])

    assert root_help.exit_code == 0
    for command in ['search', 'download', 'corpus', 'filter', 'topics', 'import',
                    'config', 'recipe', 'scrape', 'store', 'status', 'reset']:
        assert command in root_help.output
    assert filter_help.exit_code == 0
    assert set(cli.filter_group.commands) == {'regex', 'topic', 'status', 'reset'}
    assert all(command in filter_help.output for command in cli.filter_group.commands)
    assert topics_help.exit_code == 0
    assert set(cli.topics_group.commands) == {
        'train', 'compare', 'show', 'name', 'predict', 'trends', 'store', 'models',
    }
    assert recipe_help.exit_code == 0
    assert 'prompt' in recipe_help.output


def test_nested_groups_register_every_command_at_its_public_path() -> None:
    """Keep the installed command hierarchy from drifting from its design."""
    assert set(cli.corpus_group.commands) == {'stats', 'searches'}
    assert set(cli.import_group.commands) == {'pdfs', 'author'}
    assert set(cli.recipe_group.commands) == {'prompt'}
    assert set(cli.config_group.commands) == {
        'model', 'status', 'elsevier-key', 'core-key', 'core-membership',
        'unpaywall-email', 'crossref-email', 'openalex-key', 'ncbi-key',
        'ncbi-email', 'openai-key', 'anthropic-key',
    }
    for group in [cli.main, cli.corpus_group, cli.filter_group, cli.topics_group,
                  cli.import_group, cli.config_group, cli.recipe_group]:
        for public_name, command in group.commands.items():
            assert command.name == public_name


@pytest.mark.parametrize(
    ('kind', 'builder', 'expected_kwargs'),
    [
        ('text', 'build_text_extraction_prompt', {}),
        ('image', 'build_image_extraction_prompt', {}),
        ('image-context', 'build_image_extraction_prompt', {'with_context': True}),
        ('reconciliation', 'build_reconciliation_prompt', {}),
    ],
)
def test_recipe_prompt_renders_each_prompt_kind(
    kind: str,
    builder: str,
    expected_kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render each recipe prompt without making a model request."""
    recipe = {'recipe': 'data'}
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli, 'load_recipe', lambda name: recipe if name == 'custom' else None)

    def build(loaded: dict[str, str], **kwargs: Any) -> str:
        """Capture the selected recipe and builder options."""
        calls.update({'recipe': loaded, 'kwargs': kwargs})
        return f'{kind} prompt'

    monkeypatch.setattr(cli, builder, build)

    result = CliRunner().invoke(cli.recipe_prompt, ['custom', '--kind', kind])

    assert result.exit_code == 0
    assert result.output == f'{kind} prompt\n'
    assert calls == {'recipe': recipe, 'kwargs': expected_kwargs}


def test_recipe_prompt_writes_an_output_file_and_reports_recipe_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write long prompts to a file and present loading failures cleanly."""
    output = tmp_path / 'prompt.txt'
    monkeypatch.setattr(cli, 'load_recipe', lambda _: {})
    monkeypatch.setattr(cli, 'build_text_extraction_prompt', lambda _: 'rendered prompt')

    result = CliRunner().invoke(cli.recipe_prompt, ['custom', '--outfile', str(output)])

    assert result.exit_code == 0
    assert output.read_text() == 'rendered prompt\n'
    assert result.output == f'Prompt written to {output}.\n'

    monkeypatch.setattr(cli, 'load_recipe', lambda _: (_ for _ in ()).throw(ValueError('bad recipe')))
    invalid = CliRunner().invoke(cli.recipe_prompt, ['custom'])
    assert invalid.exit_code == 1
    assert 'Error: bad recipe' in invalid.output

    monkeypatch.setattr(cli, 'load_recipe', lambda _: {})
    unwritable = CliRunner().invoke(
        cli.recipe_prompt,
        ['custom', '--outfile', str(tmp_path / 'missing' / 'prompt.txt')],
    )
    assert unwritable.exit_code == 1
    assert 'Error:' in unwritable.output


def test_paper_search_passes_query_db_path_source_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delegate paper searches with the requested source and result count."""
    calls = {}
    monkeypatch.setattr(
        cli,
        'search_for_papers',
        lambda query, path, source, count, store_abstract, enrich, parallel, workers: calls.update({
            'query': query,
            'db_path': path,
            'source': source,
            'count': count,
            'store_abstract': store_abstract,
            'enrich': enrich,
            'parallel': parallel,
            'workers': workers,
        }),
    )

    result = CliRunner().invoke(cli.paper_search, ['Lithium solid electrolyte', 'papers.db', '--source', 'core',
                                                   '--count', '10', '--store-abstract', '--enrich',
                                                   '--parallel', '--workers', '2'])

    assert result.exit_code == 0
    assert calls == {
        'query': 'Lithium solid electrolyte',
        'db_path': 'papers.db',
        'source': 'core',
        'count': 10,
        'store_abstract': True,
        'enrich': True,
        'parallel': True,
        'workers': 2,
    }


def test_import_pdf_folder_passes_crossref_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Delegate PDF imports with Crossref lookup disabled."""
    calls = {}
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    monkeypatch.setattr(
        cli,
        'import_pdfs',
        lambda directory, db_path, use_crossref: calls.update({
            'directory': directory,
            'db_path': db_path,
            'use_crossref': use_crossref,
        }),
    )

    result = CliRunner().invoke(cli.import_pdf_folder, [str(papers_dir), 'papers.db', '--no-crossref'])

    assert result.exit_code == 0
    assert calls == {
        'directory': str(papers_dir),
        'db_path': 'papers.db',
        'use_crossref': False,
    }


def test_import_author_validates_identity_and_reports_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Delegate author discovery with exactly one identity selector."""
    calls = {}
    db_path = tmp_path / 'supervisor.db'
    review_path = tmp_path / 'works.csv'
    monkeypatch.setattr(cli, 'import_author_works', lambda *args, **kwargs: (
        calls.update({'args': args, 'kwargs': kwargs})
        or {'found': 3, 'added': 2, 'updated': 1}
    ))

    result = CliRunner().invoke(cli.import_author, [
        str(db_path),
        '--orcid', '0000-0001-2345-6789',
        '--email', 'person@example.ac.uk',
        '--review-csv', str(review_path),
    ])

    assert result.exit_code == 0
    assert calls['args'] == (str(db_path),)
    assert calls['kwargs']['orcid'] == '0000-0001-2345-6789'
    assert calls['kwargs']['author_name'] is None
    assert calls['kwargs']['review_csv'] == str(review_path)
    assert '3 matching works: 2 added and 1 updated' in result.output

    invalid = CliRunner().invoke(cli.import_author, [
        str(db_path), '--orcid', '0000-0001-2345-6789', '--author', 'Jane Smith',
        '--email', 'person@example.ac.uk',
    ])
    assert invalid.exit_code == 2
    assert 'exactly one' in invalid.output


def test_download_passes_format_and_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Delegate downloads with the selected format and sources."""
    calls = {}
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    monkeypatch.setattr(
        cli,
        'download_papers',
        lambda path, download_format, sources, download_abstract, force: calls.update({
            'db_path': path,
            'download_format': download_format,
            'sources': sources,
            'download_abstract': download_abstract,
            'force': force,
        }),
    )

    result = CliRunner().invoke(cli.download, [str(db_path), '--format', 'pdf', '--source', 'core',
                                               '--source', 'unpaywall', '--no-abstract', '--force'])

    assert result.exit_code == 0
    assert calls == {
        'db_path': str(db_path),
        'download_format': 'pdf',
        'sources': ['core', 'unpaywall'],
        'download_abstract': False,
        'force': True,
    }

    result = CliRunner().invoke(cli.download, [str(db_path), '--format', 'abstract'])

    assert result.exit_code == 0
    assert calls['download_format'] == 'abstract'
    assert calls['download_abstract'] is True


def test_search_and_download_source_choices_accept_openalex(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Accept OpenAlex as a search and download source."""
    search_calls = {}
    monkeypatch.setattr(
        cli,
        'search_for_papers',
        lambda query, path, source, count, store_abstract, enrich, parallel, workers: search_calls.update({
            'source': source, 'parallel': parallel, 'workers': workers,
        }),
    )

    result = CliRunner().invoke(cli.paper_search, ['query', 'papers.db', '--source', 'openalex'])

    assert result.exit_code == 0
    assert search_calls['source'] == 'openalex'
    assert search_calls['parallel'] is False
    assert search_calls['workers'] is None

    download_calls = {}
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    monkeypatch.setattr(
        cli,
        'download_papers',
        lambda path, download_format, sources, download_abstract, force: download_calls.update({
            'sources': sources,
            'force': force,
        }),
    )

    result = CliRunner().invoke(cli.download, [str(db_path), '--source', 'openalex'])

    assert result.exit_code == 0
    assert download_calls['sources'] == ['openalex']
    assert download_calls['force'] is False


def test_corpus_status_prints_database_storage_statistics(tmp_path: Path) -> None:
    """Print paper, blob, and storage statistics for a corpus."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.add_asset(
            conn,
            {'paper_id': 'paper:1', 'title': 'Corpus paper'},
            'paper text ' * 100,
            role='text',
            kind='text',
            mime_type='text/plain',
        )
        corpus.add_asset(
            conn,
            {'paper_id': 'paper:1', 'title': 'Corpus paper'},
            'abstract text',
            role='abstract',
            kind='text',
            mime_type='text/plain',
        )
        corpus.add_asset(
            conn,
            {'paper_id': 'paper:2', 'title': 'Corpus paper 2'},
            b'%PDF data',
            role='pdf',
            kind='pdf',
            mime_type='application/pdf',
        )
        corpus.add_structured_document(
            conn,
            {'paper_id': 'paper:1', 'title': 'Corpus paper'},
            '<article/>',
            document_format='jats',
            source='pubmed',
            original_filename='paper.nxml',
        )
        papers = {paper['paper_id']: paper for paper in corpus.paper_rows(conn)}
        papers['paper:1']['num_text_chunks'] = 3
        papers['paper:2']['num_abstract_chunks'] = 2
        corpus.upsert_paper(conn, papers['paper:1'])
        corpus.upsert_paper(conn, papers['paper:2'])

    result = CliRunner().invoke(cli.corpus_status, [str(db_path)])

    assert result.exit_code == 0
    assert f'Corpus: {db_path}' in result.output
    assert 'Papers: 2' in result.output
    assert 'Papers with abstracts: 1' in result.output
    assert 'Papers with text: 1' in result.output
    assert 'Papers with PDFs: 1' in result.output
    assert 'Papers with structured documents: 1' in result.output
    assert 'Text scrapes split into chunks: 1' in result.output
    assert 'Abstract scrapes split into chunks: 1' in result.output
    assert 'Blobs: 4' in result.output
    assert 'Original size:' in result.output
    assert 'Stored size:' in result.output
    assert 'Storage saved:' in result.output


def test_corpus_searches_prints_recent_queries_and_failures(tmp_path: Path) -> None:
    """Show reproducible search settings, outcomes, and an empty-history message."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        empty_result = CliRunner().invoke(cli.corpus_searches, [str(db_path)])
        search_id = corpus.begin_search_run(conn, 'solid   electrolyte', 'all', ['core'], 10)
        corpus.finish_search_run(
            conn,
            search_id,
            'partial',
            {'core': {'status': 'failed', 'result_count': 0, 'error': 'service unavailable'}},
        )

    result = CliRunner().invoke(cli.corpus_searches, [str(db_path), '--limit', '1'])
    json_path = tmp_path / 'searches.json'
    suffixed_path = tmp_path / 'searches-copy.json'
    json_result = CliRunner().invoke(
        cli.corpus_searches,
        [str(db_path), '--limit', '1', '--outfile', str(json_path)],
    )
    suffixed_result = CliRunner().invoke(
        cli.corpus_searches,
        [str(db_path), '--limit', '1', '--outfile', str(tmp_path / 'searches-copy')],
    )

    assert empty_result.exit_code == 0
    assert 'No searches recorded.' in empty_result.output
    assert result.exit_code == 0
    assert f'Corpus searches: {db_path}' in result.output
    assert f'#{search_id}' in result.output
    assert ('partial | source=all | requested=10 | parallel=no | workers=1 | '
            'results=0 | added=0 | updated=0') in result.output
    assert 'solid electrolyte' in result.output
    assert 'core failed: service unavailable' in result.output
    assert json_result.exit_code == 0
    assert f'Search history written to {json_path}.' in json_result.output
    json_rows = json.loads(json_path.read_text())
    assert json_rows[0]['query'] == 'solid   electrolyte'
    assert json_rows[0]['sources'] == ['core']
    assert json_rows[0]['source_results']['core']['error'] == 'service unavailable'
    assert suffixed_result.exit_code == 0
    assert suffixed_path.exists()
    assert json.loads(suffixed_path.read_text()) == json_rows


def test_import_author_translates_service_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present Crossref validation and runtime failures as Click errors."""
    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a representative provider failure."""
        raise RuntimeError('crossref failed')

    monkeypatch.setattr(cli, 'import_author_works', fail)
    result = CliRunner().invoke(cli.import_author, [
        'papers.db', '--author', 'Jane Smith', '--email', 'person@example.org',
    ])
    assert result.exit_code == 1
    assert 'crossref failed' in result.output


def test_filter_overview_prints_topic_details_reasons_and_staleness() -> None:
    """Render topic filters, bounded unavailable reasons, and stale-model advice."""
    reasons = {f'reason {index}': index for index in range(12)}
    overview = {
        'filters': [{
            'name': 'topic-filter', 'method': 'topic', 'join_operator': None,
            'definition': {'model': 'model', 'include_mode': 'all'},
            'counts': {'included': 1, 'excluded': 2, 'unavailable': 3},
            'stale': True,
        }],
        'expression': 'topic-filter',
        'counts': {'included': 1, 'excluded': 2, 'unavailable': 3},
        'unavailable_reasons': reasons,
        'stale_topic_filters': ['topic-filter'],
    }
    runner = CliRunner()
    with runner.isolated_filesystem():
        cli._echo_filter_overview('papers.db', overview)


@pytest.mark.parametrize(
    ('command_name', 'target', 'arguments'),
    [
        ('filter_regex', 'apply_regex_filter', ('db', 'rules')),
        ('filter_topic', 'apply_topic_filter', ('db', 'rules')),
        ('filter_reset', 'reset_filters', ('db', '--all')),
        ('topics_train', 'train_topic_model', ('db', 'model')),
        ('topics_compare', 'compare_topic_models', ('db', 'comparison')),
        ('topics_show', 'topic_descriptions', ('model',)),
        ('topics_name', 'set_topic_name', ('model', '0', 'name')),
        ('topics_predict', 'predict_topic_model', ('model', 'db', 'predictions.csv')),
        ('topics_trends', 'aggregate_topic_trends', ('model', 'trends')),
        ('topics_store', 'store_topic_model_scores', ('model', 'db')),
    ],
)
def test_analysis_commands_translate_domain_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_name: str,
    target: str,
    arguments: tuple[str, ...],
) -> None:
    """Convert analysis-layer exceptions into concise command-line failures."""
    for name in {'db', 'rules'}:
        (tmp_path / name).write_text('{}')
    (tmp_path / 'model').mkdir()

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a representative domain validation error."""
        raise ValueError('domain failed')

    monkeypatch.setattr(cli, target, fail)
    resolved = tuple(str(tmp_path / value) if value in {'db', 'rules', 'model'} else value
                     for value in arguments)
    result = CliRunner().invoke(getattr(cli, command_name), resolved)
    assert result.exit_code == 1
    assert 'domain failed' in result.output


def test_topics_models_reports_an_empty_corpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tell users explicitly when no stored topic models exist."""
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    monkeypatch.setattr(cli, 'stored_topic_models', lambda *args, **kwargs: [])
    result = CliRunner().invoke(cli.topics_models, [str(db_path)])
    assert result.exit_code == 0
    assert 'No topic models' in result.output


def test_topics_train_delegates_options_and_reports_diagnostics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Train the topic model with explicit fields and print corpus warnings."""
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    model_dir = tmp_path / 'model'
    stopwords_path = tmp_path / 'stopwords.txt'
    stopwords_path.write_text('battery\n')
    calls = {}

    def fake_train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Record training arguments and return a diagnostic summary."""
        calls['args'] = args
        calls['kwargs'] = kwargs
        return {
            'model_dir': str(model_dir),
            'report': {
                'documents_used': 120,
                'vocabulary_size': 800,
                'warnings': ['Small topic-model corpus: test warning'],
            },
        }

    monkeypatch.setattr(cli, 'train_topic_model', fake_train)

    result = CliRunner().invoke(cli.topics_train, [
        str(db_path),
        str(model_dir),
        '--topics', '6',
        '--field', 'title',
        '--field', 'text',
        '--min-df', '3',
        '--max-df', '0.9',
        '--max-features', '5000',
        '--learning-method', 'batch',
        '--iterations', '12',
        '--random-seed', '9',
        '--top-terms', '8',
        '--representative-papers', '4',
        '--stopwords-file', str(stopwords_path),
        '--ngram-max', '1',
        '--overwrite',
        '--in-memory',
    ])

    assert result.exit_code == 0
    assert calls['args'] == (str(db_path), str(model_dir))
    assert calls['kwargs'] == {
        'num_topics': 6,
        'text_fields': ('title', 'text'),
        'min_df': 3,
        'max_df': 0.9,
        'max_features': 5000,
        'learning_method': 'batch',
        'max_iter': 12,
        'random_state': 9,
        'top_terms': 8,
        'representative_papers': 4,
        'stopwords_file': str(stopwords_path),
        'ngram_max': 1,
        'overwrite': True,
        'emit_warnings': False,
        'streaming': False,
        'batch_size': 128,
        'cache_dir': None,
        'evaluation_sample_size': 10000,
    }
    assert 'Warning: Small topic-model corpus' in result.output
    assert 'Trained 6 topics from 120 papers using 800 terms.' in result.output


def test_topics_compare_delegates_grid_and_reports_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Train a topic-count and seed grid through the comparison command."""
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    output_dir = tmp_path / 'comparison'
    calls = {}

    def fake_compare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Record comparison arguments and return an output summary."""
        calls['args'] = args
        calls['kwargs'] = kwargs
        return {
            'output_dir': str(output_dir),
            'models_trained': 4,
            'comparison_csv': str(output_dir / 'model_comparison.csv'),
            'models': [{'warnings': ['comparison warning']}],
        }

    monkeypatch.setattr(cli, 'compare_topic_models', fake_compare)

    result = CliRunner().invoke(cli.topics_compare, [
        str(db_path), str(output_dir),
        '--topics', '6', '--topics', '8',
        '--seed', '3', '--seed', '4',
        '--field', 'abstract',
        '--iterations', '7',
        '--ngram-max', '2',
    ])

    assert result.exit_code == 0
    assert calls['args'] == (str(db_path), str(output_dir))
    assert calls['kwargs']['topic_counts'] == (6, 8)
    assert calls['kwargs']['random_states'] == (3, 4)
    assert calls['kwargs']['text_fields'] == ('abstract',)
    assert calls['kwargs']['max_iter'] == 7
    assert calls['kwargs']['ngram_max'] == 2
    assert 'Warning: comparison warning' in result.output
    assert 'Trained 4 comparison models' in result.output


def test_topic_inspection_naming_and_prediction_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Expose manual topic review, naming, and prediction through CLI commands."""
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    calls = []
    monkeypatch.setattr(cli, 'topic_descriptions', lambda _: [{
        'topic_id': 0,
        'topic_name': '',
        'top_terms': ['lithium', 'electrolyte'],
        'representative_papers': [{'title': 'Representative paper', 'probability': '0.75'}],
    }])
    monkeypatch.setattr(
        cli,
        'set_topic_name',
        lambda directory, topic_id, name: calls.append(('name', directory, topic_id, name)),
    )
    monkeypatch.setattr(
        cli,
        'predict_topic_model',
        lambda directory, database, output, batch_size: {
            'papers_total': 10,
            'papers_predicted': 9,
            'papers_without_vocabulary_terms': 1,
            'output_path': output,
        },
    )
    runner = CliRunner()

    shown = runner.invoke(cli.topics_show, [str(model_dir)])
    named = runner.invoke(cli.topics_name, [str(model_dir), '0', 'solid electrolytes'])
    predicted = runner.invoke(cli.topics_predict, [
        str(model_dir), str(db_path), str(tmp_path / 'scores.csv'),
    ])

    assert shown.exit_code == 0
    assert 'Topic 0: (unnamed)' in shown.output
    assert 'Representative paper (0.750)' in shown.output
    assert named.exit_code == 0
    assert calls == [('name', str(model_dir), 0, 'solid electrolytes')]
    assert predicted.exit_code == 0
    assert 'Predicted topics for 9 of 10 papers' in predicted.output


def test_topic_trend_store_and_model_status_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expose trend aggregation and explicit corpus score storage through the CLI."""
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    calls = []
    monkeypatch.setattr(
        cli,
        'aggregate_topic_trends',
        lambda *args, **kwargs: calls.append(('trends', args, kwargs)) or {
            'windows': 4,
            'trends_csv': str(tmp_path / 'trends' / 'topic_trends.csv'),
            'papers_missing_or_invalid_date': 2,
            'plot_path': str(tmp_path / 'trends' / 'topic_trends_plot.png'),
        },
    )
    monkeypatch.setattr(
        cli,
        'store_topic_model_scores',
        lambda *args, **kwargs: calls.append(('store', args, kwargs)) or {
            'name': 'demo',
            'model_id': 'lda:test',
            'papers_predicted': 9,
            'papers_without_vocabulary_terms': 1,
        },
    )
    monkeypatch.setattr(cli, 'stored_topic_models', lambda *args, **kwargs: [{
        'name': 'demo',
        'model_id': 'lda:test',
        'is_current': True,
        'num_topics': 3,
        'papers_predicted': 9,
        'papers_without_vocabulary_terms': 1,
        'text_fields': ['title', 'abstract'],
    }])
    runner = CliRunner()

    trended = runner.invoke(cli.topics_trends, [
        str(model_dir), str(tmp_path / 'trends'),
        '--bin-size', '5', '--step-size', '1', '--start-year', '2000', '--plot',
    ])
    stored = runner.invoke(cli.topics_store, [
        str(model_dir), str(db_path), '--name', 'demo', '--batch-size', '64',
    ])
    shown = runner.invoke(cli.topics_models, [str(db_path)])

    assert trended.exit_code == 0
    assert calls[0][2]['bin_size'] == 5
    assert calls[0][2]['step_size'] == 1
    assert calls[0][2]['plot'] is True
    assert '4 topic trend windows' in trended.output
    assert 'Topic trend plot:' in trended.output
    assert stored.exit_code == 0
    assert calls[1][2] == {'name': 'demo', 'batch_size': 64}
    assert 'Stored demo (lda:test)' in stored.output
    assert shown.exit_code == 0
    assert 'demo (lda:test) [current]' in shown.output

    custom_plot = runner.invoke(cli.topics_trends, [
        str(model_dir), str(tmp_path / 'custom-trends'),
        '--plot-file', 'custom-topic-trends.svg',
    ])
    assert custom_plot.exit_code == 0
    assert calls[2][2]['plot'] == 'custom-topic-trends.svg'


def test_scrape_passes_model_image_cleanup_and_output_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Delegate scraping with model, image, cleanup, and output options."""
    calls = {}
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')

    def fake_scrape_papers(*args: Any, **kwargs: Any) -> None:
        """Record arguments delegated to the scrape workflow."""
        calls['args'] = args
        calls['kwargs'] = kwargs

    monkeypatch.setattr(cli, 'scrape_papers', fake_scrape_papers)

    result = CliRunner().invoke(cli.scrape, [
        str(db_path),
        'custom_recipe',
        '--mode', 'text-images',
        '--image-context', 'paper-text',
        '--image-dir', 'images',
        '--image-extraction', 'pages',
        '--image-dpi', '150',
        '--image-batch-size', 'all',
        '--model', 'text-model',
        '--provider', 'local',
        '--base-url', 'http://localhost:8000/v1',
        '--vision-model', 'vision-model',
        '--vision-provider', 'local',
        '--vision-base-url', 'http://localhost:8000/v1',
        '--delete-images-after',
        '--output', 'scraped.csv',
        '--force',
        '--ignore-filters',
        '--count', '3',
        '--order', 'publication-desc',
        '--compression-scope', 'both',
        '--compression-mode', 'always',
        '--compression-ratio', 'auto',
        '--no-compression-content-detection',
    ])

    assert result.exit_code == 0
    assert calls['args'] == (str(db_path), 'custom_recipe')
    assert calls['kwargs'] == {
        'mode': 'text-images',
        'image_dir': 'images',
        'image_context': 'paper-text',
        'image_extraction': 'pages',
        'image_dpi': 150,
        'image_batch_size': 'all',
        'model': 'text-model',
        'provider': 'local',
        'base_url': 'http://localhost:8000/v1',
        'vision_model': 'vision-model',
        'vision_provider': 'local',
        'vision_base_url': 'http://localhost:8000/v1',
        'delete_images_after': True,
        'output_path': 'scraped.csv',
        'force': True,
        'ignore_filters': True,
        'scrape_count': 3,
        'scrape_order': 'publication-desc',
        'compression_scope': 'both',
        'compression_mode': 'always',
        'compression_ratio': 'auto',
        'compression_content_detection': False,
    }


def test_store_passes_files_recipe_and_assume_yes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Delegate result storage with file, recipe, and confirmation options."""
    calls = {}
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    monkeypatch.setattr(
        cli,
        'store_results',
        lambda db_path, in_file, out_file, unit_conversion, recipe, assume_yes: calls.update({
            'db_path': db_path,
            'in_file': in_file,
            'out_file': out_file,
            'unit_conversion': unit_conversion,
            'recipe': recipe,
            'assume_yes': assume_yes,
        }),
    )

    result = CliRunner().invoke(cli.store, [str(db_path), 'scraped.csv', 'materials.csv', 'sse', '--assume-yes'])

    assert result.exit_code == 0
    assert calls == {
        'db_path': str(db_path),
        'in_file': 'scraped.csv',
        'out_file': 'materials.csv',
        'unit_conversion': True,
        'recipe': 'sse',
        'assume_yes': True,
    }


def test_key_update_entry_points_call_settings_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delegate API configuration entry points to their settings helpers."""
    calls = []
    monkeypatch.setattr(cli, 'update_elsevier_key', lambda: calls.append('elsevier'))
    monkeypatch.setattr(cli, 'update_core_key', lambda: calls.append('core'))
    monkeypatch.setattr(cli, 'update_core_membership', lambda: calls.append('core-membership'))
    monkeypatch.setattr(cli, 'update_unpaywall_email', lambda: calls.append('unpaywall'))
    monkeypatch.setattr(cli, 'update_crossref_email', lambda: calls.append('crossref'))
    monkeypatch.setattr(cli, 'update_openalex_key', lambda: calls.append('openalex'))
    monkeypatch.setattr(cli, 'update_ncbi_key', lambda: calls.append('ncbi-key'))
    monkeypatch.setattr(cli, 'update_ncbi_email', lambda: calls.append('ncbi-email'))
    monkeypatch.setattr(cli, 'update_openai_key', lambda: calls.append('openai'))
    monkeypatch.setattr(cli, 'update_anthropic_key', lambda: calls.append('anthropic'))

    cli.update_elsevier_api_key()
    cli.update_core_api_key()
    cli.update_core_membership_level()
    cli.update_unpaywall_api_email()
    cli.update_crossref_api_email()
    cli.update_openalex_api_key()
    cli.update_ncbi_api_key()
    cli.update_ncbi_api_email()
    cli.update_openai_api_key()
    cli.update_anthropic_api_key()

    assert calls == ['elsevier', 'core', 'core-membership', 'unpaywall', 'crossref',
                     'openalex', 'ncbi-key', 'ncbi-email', 'openai', 'anthropic']


def test_model_config_infers_capabilities_saves_profile_and_prints_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Infer capabilities, save the model profile, and print its summary."""
    calls = {}
    monkeypatch.setattr(cli, 'infer_model_capabilities', lambda profile, model: ['text'])
    monkeypatch.setattr(
        cli,
        'set_model_profile',
        lambda profile, provider, model, base_url, api_key, capabilities, temperature, top_p, input_token_limit: calls.update({
            'profile': profile,
            'provider': provider,
            'model': model,
            'base_url': base_url,
            'api_key': api_key,
            'capabilities': capabilities,
            'temperature': temperature,
            'top_p': top_p,
            'input_token_limit': input_token_limit,
        }),
    )

    result = CliRunner().invoke(cli.model_config, [
        'text',
        '--provider', 'openai',
        '--model', 'gpt-test',
        '--temperature', '0.2',
        '--top-p', '0.9',
        '--input-token-limit', '64000',
    ])

    assert result.exit_code == 0
    assert calls == {
        'profile': 'text',
        'provider': 'openai',
        'model': 'gpt-test',
        'base_url': None,
        'api_key': None,
        'capabilities': ['text'],
        'temperature': 0.2,
        'top_p': 0.9,
        'input_token_limit': 64000,
    }
    assert 'Updated text model profile: openai/gpt-test [text] temperature=0.2 top_p=0.9 input_token_limit=64000' in result.output


def test_model_config_uses_explicit_capabilities_without_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Save explicit model capabilities without running inference."""
    calls = {}
    monkeypatch.setattr(
        cli,
        'infer_model_capabilities',
        lambda *_: (_ for _ in ()).throw(AssertionError('capabilities should not be inferred')),
    )
    monkeypatch.setattr(
        cli,
        'set_model_profile',
        lambda profile, provider, model, base_url, api_key, capabilities, temperature, top_p, input_token_limit: calls.update({
            'capabilities': capabilities,
            'input_token_limit': input_token_limit,
        }),
    )

    result = CliRunner().invoke(cli.model_config, [
        'vision',
        '--provider', 'local',
        '--model', 'vision-model',
        '--base-url', 'http://localhost:8000/v1',
        '--api-key', 'test-key',
        '--capability', 'text',
        '--capability', 'vision',
    ])

    assert result.exit_code == 0
    assert calls['capabilities'] == ['text', 'vision']
    assert calls['input_token_limit'] == cli.DEFAULT_INPUT_TOKEN_LIMIT


def test_model_status_prints_text_and_vision_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Print configured text and vision model profiles."""
    profiles = {
        'text': {
            'provider': 'openai',
            'model': 'gpt-test',
            'capabilities': ['text'],
            'temperature': 0.0,
            'top_p': 1.0,
            'input_token_limit': 32000,
            'base_url': None,
        },
        'vision': {
            'provider': 'local',
            'model': 'vision-test',
            'capabilities': ['text', 'vision'],
            'temperature': 0.1,
            'top_p': 0.8,
            'input_token_limit': 120000,
            'base_url': 'http://localhost:8000/v1',
        },
    }
    monkeypatch.setattr(cli, 'get_model_profile', lambda profile: profiles[profile])

    result = CliRunner().invoke(cli.model_status)

    assert result.exit_code == 0
    assert 'text: openai/gpt-test capabilities=[text] temperature=0.0 top_p=1.0 input_token_limit=32000 base_url=None' in result.output
    assert 'vision: local/vision-test capabilities=[text, vision] temperature=0.1 top_p=0.8 input_token_limit=120000' in result.output


def test_utility_commands_delegate_to_maintenance_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Delegate reset and status commands to maintenance helpers."""
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    calls = []
    monkeypatch.setattr(cli, 'reset', lambda path: calls.append(('reset', path)))
    monkeypatch.setattr(cli, 'status', lambda path: calls.append(('status', path)))

    runner = CliRunner()
    assert runner.invoke(cli.reset_miner, [str(db_path)]).exit_code == 0
    assert runner.invoke(cli.miner_status, [str(db_path)]).exit_code == 0

    assert calls == [
        ('reset', str(db_path)),
        ('status', str(db_path)),
    ]


def test_enrich_passes_sources_batch_size_and_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Forward every pmt enrich option to the enrichment worker."""
    calls = {}
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')

    def fake_enrich_corpus(path: str, **kwargs: Any) -> dict[str, int]:
        """Record the enrichment request and report an empty run."""
        calls.update({'db_path': path, **kwargs})
        return {'succeeded': 0, 'partial': 0, 'not_found': 0, 'unresolved': 0,
                'authors': 0, 'subjects': 0, 'references': 0}

    monkeypatch.setattr(cli, 'enrich_corpus', fake_enrich_corpus)

    result = CliRunner().invoke(cli.enrich, [
        str(db_path), '--source', 'crossref', '--batch-size', '25', '--limit', '10',
        '--force', '--retry-failed', '--refresh-after', '30', '--no-references',
        '--resolve-references', '--email', 'me@example.com',
    ])

    assert result.exit_code == 0
    assert calls['sources'] == ['crossref']
    assert calls['batch_size'] == 25
    assert calls['limit'] == 10
    assert calls['force'] is True
    assert calls['retry_failed'] is True
    assert calls['refresh_after'] == 30
    assert calls['references'] is False
    assert calls['resolve_references'] is True
    assert calls['email'] == 'me@example.com'


def test_enrich_reports_the_run_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Print enrichment counts and warn about skipped papers on stderr."""
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    monkeypatch.setattr(cli, 'enrich_corpus', lambda *_, **__: {
        'succeeded': 8, 'partial': 1, 'not_found': 2, 'unresolved': 3,
        'authors': 40, 'subjects': 20, 'references': 100,
    })

    result = CliRunner().invoke(cli.enrich, [str(db_path)])

    assert result.exit_code == 0
    assert '8 enriched, 1 partial, 2 not found' in result.output
    assert 'Stored 40 author records, 20 subject records, and 100 references.' in result.output
    assert ('3 papers have no DOI, OpenAlex identifier, PMID, arXiv identifier, medRxiv DOI, '
            'bioRxiv DOI, or chemRxiv DOI' in result.output)


def test_enrich_wraps_worker_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Convert provider and validation failures into a CLI error."""
    db_path = tmp_path / 'papers.db'
    db_path.write_text('')

    def failing(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail as an exhausted provider budget would."""
        raise RuntimeError('OpenAlex daily credit budget is exhausted.')

    monkeypatch.setattr(cli, 'enrich_corpus', failing)

    result = CliRunner().invoke(cli.enrich, [str(db_path)])

    assert result.exit_code == 1
    assert 'credit budget is exhausted' in result.output


def test_import_author_accepts_a_missing_email_and_forwards_enrich(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the stored setting supply the email and forward the enrich flag."""
    calls = {}
    monkeypatch.setattr(cli, 'import_author_works', lambda db_path, **kwargs: calls.update(
        {'db_path': db_path, **kwargs}) or {'found': 1, 'added': 1, 'updated': 0, 'enriched': 1})

    result = CliRunner().invoke(cli.import_author, ['papers.db', '--orcid',
                                                    '0000-0002-1825-0097', '--enrich'])

    assert result.exit_code == 0
    assert calls['email'] is None
    assert calls['enrich'] is True
    assert 'Enriched 1 imported works.' in result.output


def test_corpus_status_prints_enrichment_counts(tmp_path: Path) -> None:
    """Report enrichment progress alongside the storage statistics."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'demo:1', 'title': 'Demo'})
        corpus.write_enrichment(
            conn,
            [{**{field: '' for field in corpus.enrichment_update_fields()},
              'paper_id': 'demo:1', 'enrichment_status': 'succeeded',
              'is_oa': 1, 'updated_at': corpus.utc_now()}],
            authors=[{'paper_id': 'demo:1', 'author_position': 0, 'affiliation_rank': 0,
                      'orcid': '0000-0002-1825-0097', 'source': 'openalex'}],
        )

    result = CliRunner().invoke(cli.corpus_status, [str(db_path)])

    assert result.exit_code == 0
    assert 'Papers enriched: 1' in result.output
    assert 'Papers open access: 1' in result.output
    assert 'Author records: 1 (1 with ORCID)' in result.output


def test_search_download_and_enrich_source_choices_accept_arxiv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accept arXiv as a search, download, and enrichment source."""
    search_calls = {}
    monkeypatch.setattr(
        cli,
        'search_for_papers',
        lambda query, path, source, count, store_abstract, enrich, parallel, workers: search_calls.update({
            'source': source, 'parallel': parallel, 'workers': workers,
        }),
    )

    result = CliRunner().invoke(cli.paper_search, ['query', 'papers.db', '--source', 'arxiv'])

    assert result.exit_code == 0
    assert search_calls['source'] == 'arxiv'

    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    download_calls = {}
    monkeypatch.setattr(
        cli,
        'download_papers',
        lambda path, download_format, sources, download_abstract, force: download_calls.update(
            {'sources': sources}),
    )

    result = CliRunner().invoke(cli.download, [str(db_path), '--source', 'arxiv'])

    assert result.exit_code == 0
    assert download_calls['sources'] == ['arxiv']

    enrich_calls = {}
    monkeypatch.setattr(
        cli,
        'enrich_corpus',
        lambda path, **kwargs: enrich_calls.update({'sources': kwargs.get('sources')}) or {
            key: 0 for key in ('succeeded', 'partial', 'not_found', 'unresolved',
                               'failed', 'authors', 'subjects', 'references', 'batches')},
    )

    result = CliRunner().invoke(cli.enrich, [str(db_path), '--source', 'arxiv'])

    assert result.exit_code == 0
    assert enrich_calls['sources'] == ['arxiv']


def test_search_download_and_enrich_source_choices_accept_pubmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accept PubMed as a search, download, and enrichment source."""
    search_calls = {}
    monkeypatch.setattr(
        cli,
        'search_for_papers',
        lambda query, path, source, count, store_abstract, enrich, parallel, workers: search_calls.update({
            'source': source, 'parallel': parallel, 'workers': workers,
        }),
    )

    result = CliRunner().invoke(cli.paper_search, ['query', 'papers.db', '--source', 'pubmed'])

    assert result.exit_code == 0
    assert search_calls['source'] == 'pubmed'

    db_path = tmp_path / 'papers.db'
    db_path.write_text('')
    download_calls = {}
    monkeypatch.setattr(
        cli,
        'download_papers',
        lambda path, download_format, sources, download_abstract, force: download_calls.update(
            {'sources': sources}),
    )

    result = CliRunner().invoke(cli.download, [str(db_path), '--source', 'pubmed'])

    assert result.exit_code == 0
    assert download_calls['sources'] == ['pubmed']

    enrich_calls = {}
    monkeypatch.setattr(
        cli,
        'enrich_corpus',
        lambda path, **kwargs: enrich_calls.update({'sources': kwargs.get('sources')}) or {
            key: 0 for key in ('succeeded', 'partial', 'not_found', 'unresolved',
                               'failed', 'authors', 'subjects', 'references', 'batches')},
    )

    result = CliRunner().invoke(cli.enrich, [str(db_path), '--source', 'pubmed'])

    assert result.exit_code == 0
    assert enrich_calls['sources'] == ['pubmed']


def test_filter_topic_prints_the_successful_overview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass a successful topic-filter result to the shared CLI formatter."""
    db_path = tmp_path / 'papers.db'
    rules_path = tmp_path / 'rules.json'
    db_path.write_text('')
    rules_path.write_text('{}')
    overview = {'counts': {}, 'filters': [], 'stale_topic_filters': []}
    seen = []
    monkeypatch.setattr(cli, 'apply_topic_filter', lambda *args, **kwargs: overview)
    monkeypatch.setattr(cli, '_echo_filter_overview', lambda *args: seen.append(args))
    result = CliRunner().invoke(cli.filter_topic, [str(db_path), str(rules_path)])
    assert result.exit_code == 0
    assert seen == [(str(db_path), overview)]
