"""Exercise machine-readable workflow and inspection output for CLI agents."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Any, NoReturn

import pytest
from click.testing import CliRunner

from paperminertoolkit import cli
from paperminertoolkit.corpus import database as corpus

# Complete workflow results: every JSON result is validated against its schema.
SEARCH_SUMMARY = {
    'search_id': 1, 'status': 'completed', 'sources': ['arxiv'], 'source_results': {},
    'paper_ids': ['paper:1'], 'result_count': 1, 'papers_added': 1, 'papers_updated': 0,
    'abstracts_stored': 0,
}
DOWNLOAD_SUMMARY = {
    'texts': 0, 'pdfs': 1, 'abstracts': 0, 'texts_skipped': 0, 'pdfs_skipped': 0,
    'abstracts_skipped': 0, 'supplements': 0, 'supplements_skipped': 0,
    'supplements_failed': 0, 'supplements_unavailable': 0, 'supplement_discovery_failed': 0,
}
ENRICH_SUMMARY = {'succeeded': 1, 'partial': 0, 'not_found': 0, 'unresolved': 0, 'failed': 1,
                  'authors': 2, 'subjects': 0, 'references': 3, 'batches': 1}
SCRAPE_SUMMARY = {
    'mode': 'text', 'output_path': 'scraped.csv', 'papers_selected': 2, 'papers': 2,
    **{f'{stage}_{outcome}': 0 for stage in ('abstract', 'text', 'image')
       for outcome in ('attempted', 'skipped', 'succeeded', 'failed')},
    'text_attempted': 2, 'text_succeeded': 1, 'text_failed': 1, 'missing_source': 1,
    'failed_papers': [{'paper_id': 'p:1', 'stage': 'text', 'error': 'No downloaded text'}],
    'materials': 3, 'layout_figures': 0, 'chunked_inputs': 0, 'chunk_requests': 0,
    'abstract_chunked': 0, 'text_chunked': 0, 'supplement_text_attempted': 0,
    'supplement_image_attempted': 0, 'supplement_skipped': 0, 'supplement_failed': 0,
    'supplement_unsupported': 0, 'supplement_materials': 0, 'elapsed_seconds': 1.5,
}
STORE_SUMMARY = {'status': 'stored', 'input_path': 'in.csv', 'output_path': 'out.csv',
                 'rows_read': 2, 'rows_written': 2, 'rows_total': 2,
                 'converted_columns': [], 'unconverted_columns': ['Name'], 'skipped_columns': [],
                 'papers_marked_stored': 1}
IMPORT_PDFS_SUMMARY = {'files': 1, 'added': 1, 'updated': 0, 'doi_found': 1, 'enriched': 0,
                       'metadata_errors': 0, 'paper_ids': ['external:a']}
def _error(result: Any) -> dict[str, Any]:
    """Return the --json error object a failed command printed on stdout."""
    payload = json.loads(result.stdout)
    assert set(payload) == {'error'}
    assert payload['error']['exit_code'] == result.exit_code
    return payload['error']


SUMMARIES = {'download_papers': DOWNLOAD_SUMMARY, 'enrich_corpus': ENRICH_SUMMARY,
             'scrape_papers': SCRAPE_SUMMARY}


@pytest.mark.parametrize(('command', 'target', 'summary'), [
    ('search', 'search_for_papers', SEARCH_SUMMARY),
    ('download', 'download_papers', DOWNLOAD_SUMMARY),
    ('enrich', 'enrich_corpus', ENRICH_SUMMARY),
])
def test_workflow_json_is_separate_from_progress(
    command: str, target: str, summary: dict[str, Any],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the workflow payload unmodified while routing progress to stderr."""
    def workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Simulate workflow chatter alongside its structured return value."""
        print('Workflow progress')
        return summary

    monkeypatch.setattr(cli, target, workflow)
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    arguments = ['query', str(db_path)] if command == 'search' else [str(db_path)]
    result = CliRunner().invoke(cli.main, [command, *arguments, '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == summary
    assert result.stderr == 'Workflow progress\n'


@pytest.mark.parametrize('search_status', ['partial', 'failed'])
def test_search_json_survives_partial_failure(
    search_status: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let callers recover saved paper IDs even when a search provider fails."""
    summary = {
        **SEARCH_SUMMARY, 'status': search_status, 'paper_ids': ['saved:1'],
        'source_results': {'arxiv': {'status': 'failed'}},
    }
    monkeypatch.setattr(cli, 'search_for_papers', lambda *args, **kwargs: summary)
    result = CliRunner().invoke(cli.main, ['search', 'query', '--json'])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == summary
    assert 'Available results have been saved' in result.stderr


@pytest.mark.parametrize('command', ['download', 'enrich'])
@pytest.mark.parametrize('error', [
    ValueError('Invalid configuration'), RuntimeError('Unsupported schema'),
    OSError('Permission denied'), sqlite3.OperationalError('Database is locked'),
])
def test_workflow_operational_errors_print_one_error_object(
    command: str, error: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report expected failures as a typed JSON error object, with the message on stderr."""
    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise the selected storage or configuration failure."""
        raise error

    monkeypatch.setattr(cli, 'download_papers' if command == 'download' else 'enrich_corpus', fail)
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    result = CliRunner().invoke(cli.main, [command, str(db_path), '--json'])
    assert result.exit_code == 1
    assert _error(result)['type'] == ('io' if isinstance(error, (OSError, sqlite3.Error))
                                      else 'operational')
    assert _error(result)['message'] == str(error)
    assert f'Error: {error}' in result.stderr


def test_json_inspection_reports_real_corpus_state(tmp_path: Path) -> None:
    """Expose storage, failures, search provenance, and filter decisions as data."""
    db_path = tmp_path / 'papers.db'
    with closing(corpus.connect(db_path)) as conn, conn:
        corpus.upsert_paper(conn, {
            'paper_id': 'paper:1', 'title': 'Example',
            'pdf_download_status': 'failed', 'num_text_materials': 3,
        })
        search_id = corpus.begin_search_run(conn, 'query', 'arxiv', ['arxiv'], 1)
        corpus.finish_search_run(conn, search_id, 'completed', {})
    runner = CliRunner()
    stats = runner.invoke(cli.main, ['corpus', 'stats', str(db_path), '--json'])
    assert stats.exit_code == 0, stats.output
    assert json.loads(stats.stdout)['stats']['papers'] == 1
    assert 'enrichment' in json.loads(stats.stdout)

    progress = runner.invoke(cli.main, ['status', str(db_path), '--json'])
    assert progress.exit_code == 0, progress.output
    assert json.loads(progress.stdout)['pipeline']['pdf_download_status'] == {'failed': 1}
    assert json.loads(progress.stdout)['text_materials'] == 3

    outfile = tmp_path / 'history'
    history = runner.invoke(cli.main, [
        'corpus', 'searches', str(db_path), '--json', '--outfile', str(outfile),
    ])
    assert history.exit_code == 0, history.output
    assert json.loads(history.stdout)[0]['search_id'] == search_id
    assert json.loads(outfile.with_suffix('.json').read_text()) == json.loads(history.stdout)
    assert 'Search history written' in history.stderr

    filters = runner.invoke(cli.main, ['filter', 'status', str(db_path), '--json'])
    assert filters.exit_code == 0, filters.output
    assert isinstance(json.loads(filters.stdout), dict)


@pytest.mark.parametrize('command', [
    ['corpus', 'stats'], ['corpus', 'searches'], ['status'], ['filter', 'status'],
])
def test_json_inspection_reports_invalid_databases_cleanly(
    command: list[str], tmp_path: Path,
) -> None:
    """Keep corrupt database diagnostics off stdout and avoid tracebacks."""
    db_path = tmp_path / 'corrupt.db'
    db_path.write_text('not sqlite')
    result = CliRunner().invoke(cli.main, [*command, str(db_path), '--json'])
    assert result.exit_code == 1
    assert _error(result)['type'] == 'io'
    assert 'Error:' in result.stderr


def test_config_json_omits_api_keys_without_mutating_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep effective model credentials out of machine-readable status."""
    profile = {'provider': 'local', 'model': 'example', 'api_key': 'secret-value'}
    monkeypatch.setattr(cli, 'get_model_profile', lambda _: profile)
    result = CliRunner().invoke(cli.main, ['config', 'status', '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        'text': {'provider': 'local', 'model': 'example'},
        'vision': {'provider': 'local', 'model': 'example'},
    }
    assert 'secret-value' not in result.output
    assert profile['api_key'] == 'secret-value'


def test_recipe_discovery_outputs_validated_bundled_and_custom_recipes(tmp_path: Path) -> None:
    """Allow recipe selection and customization without any model calls."""
    runner = CliRunner()
    listed = runner.invoke(cli.main, ['recipe', 'list', '--json'])
    assert listed.exit_code == 0, listed.output
    names = json.loads(listed.stdout)
    assert 'sse' in names
    shown = runner.invoke(cli.main, ['recipe', 'show', 'sse'])
    assert shown.exit_code == 0, shown.output
    recipe = json.loads(shown.stdout)
    assert recipe['record definition'] and recipe['search fields']
    custom = tmp_path / 'recipe.json'
    custom.write_text(shown.stdout)
    assert runner.invoke(cli.main, ['recipe', 'show', str(custom)]).stdout == shown.stdout
    custom.write_text('{}')
    invalid = runner.invoke(cli.main, ['recipe', 'show', str(custom)])
    assert invalid.exit_code == 1
    assert invalid.stdout == ''
    missing = runner.invoke(cli.main, ['recipe', 'show', 'missing-recipe'])
    assert missing.exit_code == 1
    assert missing.stdout == ''


_OVERVIEW = {'filters': [], 'expression': None, 'counts': {'included': 2, 'excluded': 1,
                                                          'unavailable': 0},
             'unavailable_reasons': {}, 'stale_topic_filters': []}


def _topic_inputs(tmp_path: Path) -> dict[str, str]:
    """Create the corpus, rules, model, and text paths the commands require."""
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    rules = tmp_path / 'rules.json'
    rules.write_text('{}', encoding='utf-8')
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    text = tmp_path / 'input.txt'
    text.write_text('LiPON', encoding='utf-8')
    return {'db': str(db_path), 'rules': str(rules), 'model': str(model_dir),
            'text': str(text), 'out': str(tmp_path / 'out')}


@pytest.mark.parametrize(('arguments', 'target', 'returned', 'expected'), [
    (['import', 'author', '{db}', '--orcid', '0000-0001', '--json'], 'import_author_works',
     {'found': 2, 'added': 1, 'updated': 1, 'enriched': 0},
     {'found': 2, 'added': 1, 'updated': 1, 'enriched': 0, 'review_csv': 'author_works.csv'}),
    (['filter', 'regex', '{db}', '{rules}', '--json'], 'apply_regex_filter', _OVERVIEW, _OVERVIEW),
    (['filter', 'topic', '{db}', '{rules}', '--json'], 'apply_topic_filter', _OVERVIEW, _OVERVIEW),
    (['filter', 'reset', '{db}', '--all', '--json'], 'reset_filters', _OVERVIEW, _OVERVIEW),
    (['topics', 'train', '{db}', '{out}', '--json'], 'train_topic_model',
     {'model_dir': 'm', 'report': {'warnings': [], 'documents_used': 3, 'vocabulary_size': 9}},
     None),
    (['topics', 'compare', '{db}', '{out}', '--json'], 'compare_topic_models',
     {'models_trained': 1, 'output_dir': 'c', 'comparison_csv': 'c.csv',
      'models': [{'warnings': []}]}, None),
    (['topics', 'show', '{model}', '--representatives', '1', '--json'], 'topic_descriptions',
     [{'topic_id': 0, 'topic_name': '', 'top_terms': ['li'],
       'representative_papers': [{'title': 'A', 'probability': 0.9},
                                 {'title': 'B', 'probability': 0.8}]}],
     {'model_dir': '{model}', 'topics': [{'topic_id': 0, 'topic_name': '', 'top_terms': ['li'],
                                          'representative_papers': [{'title': 'A',
                                                                     'probability': 0.9}]}]}),
    (['topics', 'name', '{model}', '0', ' Solid electrolytes ', '--json'], 'set_topic_name', None,
     {'model_dir': '{model}', 'topic_id': 0, 'topic_name': 'Solid electrolytes'}),
    (['topics', 'predict', '{model}', '{db}', '{out}', '--json'], 'predict_topic_model',
     {'papers_total': 3, 'papers_predicted': 2, 'papers_without_vocabulary_terms': 1,
      'output_path': 'p.csv'}, None),
    (['topics', 'trends', '{model}', '{out}', '--json'], 'aggregate_topic_trends',
     {'output_dir': 't', 'trends_csv': 't.csv', 'report_json': 't.json', 'windows': 2,
      'papers_missing_or_invalid_date': 0, 'plot_path': None}, None),
    (['topics', 'store', '{model}', '{db}', '--json'], 'store_topic_model_scores',
     {'model_id': 'id', 'name': 'model', 'papers_predicted': 2,
      'papers_without_vocabulary_terms': 0, 'prediction_corpus_fingerprint': 'f'}, None),
    (['topics', 'models', '{db}', '--json'], 'stored_topic_models',
     [{'name': 'model', 'model_id': 'id', 'is_current': True}],
     {'models': [{'name': 'model', 'model_id': 'id', 'is_current': True}]}),
    (['entities', 'corpus', '{db}', '{out}', '--model', 'm', '--json'], 'extract_corpus_entities',
     {'entities': 4, 'documents': 2, 'papers': 2, 'output': 'e.jsonl'}, None),
    (['entities', 'text', '{text}', '{out}', '--model', 'm', '--json'], 'extract_file_entities',
     {'entities': 1, 'documents': 1, 'output': 'e.jsonl'}, None),
])
def test_analysis_commands_emit_their_workflow_results_as_json(
    arguments: list[str], target: str, returned: Any, expected: Any,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose filter, topic, entity, and import summaries without parsing prose."""
    paths = _topic_inputs(tmp_path)

    def workflow(*args: Any, **kwargs: Any) -> Any:
        """Print chatter, as real workflows do, and return the canned result."""
        print('Workflow progress')
        return returned

    monkeypatch.setattr(cli, target, workflow)
    result = CliRunner().invoke(cli.main, [argument.format(**paths) for argument in arguments])
    assert result.exit_code == 0, result.output
    wanted = returned if expected is None else json.loads(json.dumps(expected).replace(
        '{model}', paths['model']))
    assert json.loads(result.stdout) == wanted
    assert result.stderr == 'Workflow progress\n'


def test_validation_template_json_reports_the_written_columns(tmp_path: Path) -> None:
    """Describe the created template so an agent can fill it without reopening it."""
    outfile = tmp_path / 'validation.csv'
    result = CliRunner().invoke(cli.main, ['validate', 'template', 'sse', str(outfile), '--json'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload['outfile'] == str(outfile)
    assert payload['columns'][:3] == ['Identifier', 'Title', 'DOI']
    assert outfile.read_text(encoding='utf-8').strip().split(',') == payload['columns']


def test_topics_models_reports_errors_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn a broken corpus into a clean command error."""
    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise the storage error a damaged corpus produces."""
        raise sqlite3.DatabaseError('file is not a database')

    monkeypatch.setattr(cli, 'stored_topic_models', fail)
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    result = CliRunner().invoke(cli.main, ['topics', 'models', str(db_path), '--json'])
    assert result.exit_code == 1
    assert _error(result)['type'] == 'io'
    assert 'file is not a database' in result.stderr


def test_emit_json_converts_numpy_paths_and_sets() -> None:
    """Serialise the value types that topic summaries carry."""
    import numpy as np
    from paperminertoolkit._agent import _json_default

    assert _json_default(np.float64(0.5)) == 0.5
    assert _json_default(np.array([1, 2])) == [1, 2]
    assert _json_default(Path('a/b')) == str(Path('a/b'))
    assert _json_default({'b', 'a'}) == ['a', 'b']


@pytest.mark.parametrize(('arguments', 'target', 'returned'), [
    (['scrape', '{db}', 'sse', '--json'], 'scrape_papers', SCRAPE_SUMMARY),
    (['store', '{db}', 'in.csv', 'out.csv', 'sse', '--assume-yes', '--json'], 'store_results',
     STORE_SUMMARY),
    (['import', 'pdfs', '{dir}', '{db}', '--json'], 'import_pdfs', IMPORT_PDFS_SUMMARY),
    (['reset', '{db}', '--json'], 'reset',
     {'papers_reset': 3, 'supplement_stages_reset': 0, 'columns': ['text_scrape_status']}),
])
def test_pipeline_commands_emit_their_run_summaries(
    arguments: list[str], target: str, returned: dict[str, Any],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report scrape, store, import, and reset outcomes as data."""
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    pdf_dir = tmp_path / 'pdfs'
    pdf_dir.mkdir()

    def workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Print progress like the real workflow and return its summary."""
        print('Workflow progress')
        return returned

    monkeypatch.setattr(cli, target, workflow)
    result = CliRunner().invoke(cli.main, [
        argument.format(db=db_path, dir=pdf_dir) for argument in arguments])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == returned
    assert result.stderr == 'Workflow progress\n'


@pytest.mark.parametrize('error', [
    ValueError('mode must be one of: abstract'), KeyError('No recipe named nope'),
    RuntimeError('Corpus schema 99 is newer'),
])
def test_scrape_and_store_report_errors_without_tracebacks(
    error: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn configuration and recipe failures into exit status 1 with a message."""
    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise the selected failure."""
        raise error

    db_path = tmp_path / 'papers.db'
    db_path.touch()
    for command, target in [(['scrape', str(db_path), '--json'], 'scrape_papers'),
                            (['store', str(db_path), '--json'], 'store_results')]:
        monkeypatch.setattr(cli, target, fail)
        result = CliRunner().invoke(cli.main, command)
        assert result.exit_code == 1, result.output
        assert _error(result)['type'] == ('not_found' if isinstance(error, KeyError)
                                          else 'operational')
        assert str(error.args[0]) in result.stderr
        assert 'Traceback' not in result.stderr


def _corpus_with(tmp_path: Path, *paper_ids: str) -> Path:
    """Create a real corpus holding the given paper IDs."""
    db_path = tmp_path / 'papers.db'
    with closing(corpus.connect(db_path)) as conn, conn:
        for paper_id in paper_ids:
            corpus.upsert_paper(conn, {'paper_id': paper_id, 'title': paper_id})
    return db_path


@pytest.mark.parametrize(('command', 'target', 'extra'), [
    ('download', 'download_papers', {'respect_filters': False}),
    ('enrich', 'enrich_corpus', {}),
    ('scrape', 'scrape_papers', {}),
])
def test_paper_id_options_restrict_workflows_to_named_papers(
    command: str, target: str, extra: dict[str, Any],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combine, de-duplicate, and forward named papers in the given order."""
    db_path = _corpus_with(tmp_path, 'paper:1', 'paper:2', 'paper:3')
    ids_file = tmp_path / 'ids.txt'
    ids_file.write_text('# retry list\npaper:3\n\npaper:1\n', encoding='utf-8')
    seen: dict[str, Any] = {}

    def workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Record the selection the command forwarded."""
        seen.update(kwargs)
        return SUMMARIES[target]

    monkeypatch.setattr(cli, target, workflow)
    result = CliRunner().invoke(cli.main, [
        command, str(db_path), '--paper-id', 'paper:1', '--paper-ids-file', str(ids_file),
        '--json'])
    assert result.exit_code == 0, result.output
    assert seen['paper_ids'] == ['paper:1', 'paper:3']
    for key, value in extra.items():
        assert seen[key] == value


@pytest.mark.parametrize('command', ['download', 'enrich', 'scrape'])
def test_paper_id_options_are_omitted_when_not_given(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the whole-corpus behaviour untouched without a selection."""
    db_path = _corpus_with(tmp_path, 'paper:1')
    seen: dict[str, Any] = {}
    target = {'download': 'download_papers', 'enrich': 'enrich_corpus',
              'scrape': 'scrape_papers'}[command]
    monkeypatch.setattr(cli, target, lambda *args, **kwargs: seen.update(kwargs) or SUMMARIES[target])
    result = CliRunner().invoke(cli.main, [command, str(db_path), '--json'])
    assert result.exit_code == 0, result.output
    assert 'paper_ids' not in seen and 'respect_filters' not in seen


@pytest.mark.parametrize('command', ['download', 'enrich', 'scrape'])
def test_unknown_paper_ids_are_usage_errors(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name the unknown IDs and never start the workflow."""
    db_path = _corpus_with(tmp_path, 'paper:1')
    target = {'download': 'download_papers', 'enrich': 'enrich_corpus',
              'scrape': 'scrape_papers'}[command]

    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail if the workflow runs."""
        raise AssertionError('workflow ran')

    monkeypatch.setattr(cli, target, forbidden)
    result = CliRunner().invoke(cli.main, [
        command, str(db_path), '--paper-id', 'paper:1', '--paper-id', 'nope:9', '--json'])
    assert result.exit_code == 2
    assert _error(result)['type'] == 'usage'
    assert _error(result)['hint'] == f'Run pmt schema {command} for the accepted parameters.'
    assert 'Unknown paper ID(s)' in result.stderr
    assert 'nope:9' in result.stderr
    assert 'paper:1,' not in result.stderr


def test_an_empty_paper_ids_file_is_a_usage_error(tmp_path: Path) -> None:
    """Refuse a selection file that names nothing rather than process everything."""
    db_path = _corpus_with(tmp_path, 'paper:1')
    ids_file = tmp_path / 'ids.txt'
    ids_file.write_text('# nothing\n\n', encoding='utf-8')
    result = CliRunner().invoke(cli.main, [
        'download', str(db_path), '--paper-ids-file', str(ids_file), '--json'])
    assert result.exit_code == 2
    assert 'No paper IDs were given' in result.stderr


def test_json_errors_cover_usage_not_found_and_unknown_commands(tmp_path: Path) -> None:
    """Type usage errors, unknown commands, and missing papers in the error object."""
    db_path = _corpus_with(tmp_path, 'paper:1')
    runner = CliRunner()
    missing = runner.invoke(cli.main, ['corpus', 'show', str(db_path), 'paper:404', '--json'])
    assert missing.exit_code == 1
    assert _error(missing) == {'type': 'not_found', 'message': 'Paper not found: paper:404',
                               'hint': None, 'exit_code': 1}
    bad_option = runner.invoke(cli.main, ['corpus', 'list', str(db_path), '--limit', '0', '--json'])
    assert bad_option.exit_code == 2
    assert _error(bad_option)['type'] == 'usage'
    assert _error(bad_option)['hint'] == 'Run pmt schema corpus list for the accepted parameters.'
    unknown = runner.invoke(cli.main, ['nonsense', '--json'])
    assert unknown.exit_code == 2
    assert _error(unknown)['hint'] == 'Run pmt schema for the accepted parameters.'
    plain = runner.invoke(cli.main, ['corpus', 'show', str(db_path), 'paper:404'])
    assert plain.stdout == ''


def test_json_errors_are_not_added_after_a_printed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a partial search to exactly one JSON value even though it exits 1."""
    monkeypatch.setattr(cli, 'search_for_papers',
                        lambda *args, **kwargs: {**SEARCH_SUMMARY, 'status': 'partial',
                                                 'source_results': {'arxiv': {'status': 'failed'}}})
    result = CliRunner().invoke(cli.main, ['search', 'query', '--json'])
    assert result.exit_code == 1
    assert json.loads(result.stdout)['status'] == 'partial'


def test_json_errors_redact_configured_credentials(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    """Never repeat a configured key that a provider error happens to include."""
    import paperminertoolkit._agent as agent

    monkeypatch.setattr(agent, '_redaction_values', lambda: ['sk-secret'])

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail with a message that contains the key."""
        raise RuntimeError('GET https://api.example/?key=sk-secret failed')

    monkeypatch.setattr(cli, 'download_papers', fail)
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    result = CliRunner().invoke(cli.main, ['download', str(db_path), '--json'])
    assert 'sk-secret' not in result.stdout
    assert '[redacted]' in _error(result)['message']


def test_unexpected_errors_stay_visible_while_validating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Let a bug surface as an exception in tests, and as an internal error otherwise."""
    from paperminertoolkit._agent import VALIDATE_OUTPUT_ENV

    def crash(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail with an exception the command does not handle."""
        raise ZeroDivisionError('boom')

    monkeypatch.setattr(cli, 'download_papers', crash)
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    validating = CliRunner().invoke(cli.main, ['download', str(db_path), '--json'])
    assert isinstance(validating.exception, ZeroDivisionError)
    assert validating.stdout == ''
    monkeypatch.delenv(VALIDATE_OUTPUT_ENV)
    shipped = CliRunner().invoke(cli.main, ['download', str(db_path), '--json'])
    assert _error(shipped)['type'] == 'internal'
    assert _error(shipped)['message'] == 'ZeroDivisionError: boom'
