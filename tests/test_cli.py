"""Test the PaperScraper command-line entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import paperscraper.cli as cli
import paperscraper.corpus as corpus


def test_paper_search_passes_query_db_path_source_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delegate paper searches with the requested source and result count."""
    calls = {}
    monkeypatch.setattr(
        cli,
        'search_for_papers',
        lambda query, path, source, count, store_abstract: calls.update({
            'query': query,
            'db_path': path,
            'source': source,
            'count': count,
            'store_abstract': store_abstract,
        }),
    )

    result = CliRunner().invoke(cli.paper_search, ['Lithium solid electrolyte', 'papers.db', '--source', 'core',
                                                   '--count', '10', '--store-abstract'])

    assert result.exit_code == 0
    assert calls == {
        'query': 'Lithium solid electrolyte',
        'db_path': 'papers.db',
        'source': 'core',
        'count': 10,
        'store_abstract': True,
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
        lambda query, path, source, count, store_abstract: search_calls.update({'source': source}),
    )

    result = CliRunner().invoke(cli.paper_search, ['query', 'papers.db', '--source', 'openalex'])

    assert result.exit_code == 0
    assert search_calls['source'] == 'openalex'

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
    assert 'Text scrapes split into chunks: 1' in result.output
    assert 'Abstract scrapes split into chunks: 1' in result.output
    assert 'Blobs: 3' in result.output
    assert 'Original size:' in result.output
    assert 'Stored size:' in result.output
    assert 'Storage saved:' in result.output


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
        lambda directory, database, output: {
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


def test_scrape_passes_model_image_cleanup_and_output_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(cli, 'update_unpaywall_email', lambda: calls.append('unpaywall'))
    monkeypatch.setattr(cli, 'update_openalex_key', lambda: calls.append('openalex'))
    monkeypatch.setattr(cli, 'update_openai_key', lambda: calls.append('openai'))
    monkeypatch.setattr(cli, 'update_anthropic_key', lambda: calls.append('anthropic'))

    cli.update_elsevier_api_key()
    cli.update_core_api_key()
    cli.update_unpaywall_api_email()
    cli.update_openalex_api_key()
    cli.update_openai_api_key()
    cli.update_anthropic_api_key()

    assert calls == ['elsevier', 'core', 'unpaywall', 'openalex', 'openai', 'anthropic']


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
    assert runner.invoke(cli.reset_scraper, [str(db_path)]).exit_code == 0
    assert runner.invoke(cli.scraper_status, [str(db_path)]).exit_code == 0

    assert calls == [
        ('reset', str(db_path)),
        ('status', str(db_path)),
    ]
