"""Exercise machine-readable workflow and inspection output for CLI agents."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, NoReturn

import pytest
from click.testing import CliRunner

from paperminertoolkit import cli
from paperminertoolkit.corpus import database as corpus


@pytest.mark.parametrize(('command', 'target', 'summary'), [
    ('search', 'search_for_papers', {
        'status': 'completed', 'paper_ids': ['paper:1'], 'source_results': {},
    }),
    ('download', 'download_papers', {'pdfs': 1, 'texts': 0, 'abstracts': 0}),
    ('enrich', 'enrich_corpus', {'succeeded': 1, 'partial': 0, 'failed': 1}),
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
        'status': search_status, 'paper_ids': ['saved:1'],
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
def test_workflow_operational_errors_leave_stdout_empty(
    command: str, error: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report expected failures without corrupting the JSON output channel."""
    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise the selected storage or configuration failure."""
        raise error

    monkeypatch.setattr(cli, 'download_papers' if command == 'download' else 'enrich_corpus', fail)
    db_path = tmp_path / 'papers.db'
    db_path.touch()
    result = CliRunner().invoke(cli.main, [command, str(db_path), '--json'])
    assert result.exit_code == 1
    assert result.stdout == ''
    assert f'Error: {error}' in result.stderr


def test_json_inspection_reports_real_corpus_state(tmp_path: Path) -> None:
    """Expose storage, failures, search provenance, and filter decisions as data."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
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
    assert result.stdout == ''
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
    assert result.stdout == ''
    assert 'file is not a database' in result.stderr


def test_emit_json_converts_numpy_paths_and_sets() -> None:
    """Serialise the value types that topic summaries carry."""
    import numpy as np
    from paperminertoolkit._agent import _json_default

    assert _json_default(np.float64(0.5)) == 0.5
    assert _json_default(np.array([1, 2])) == [1, 2]
    assert _json_default(Path('a/b')) == str(Path('a/b'))
    assert _json_default({'b', 'a'}) == ['a', 'b']
