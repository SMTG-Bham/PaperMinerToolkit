"""Exercise the error, empty-result, and plain-text branches of the ``pmt`` commands.

Workflows are replaced with stand-ins, so no test makes a network or model
request; each checks the exit code and the message a caller would see.
"""

from __future__ import annotations

import io
import json
import runpy
import sqlite3
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, NoReturn

import click
import pytest
import requests
from click.testing import CliRunner, Result

import paperminertoolkit._agent as agent
import paperminertoolkit.settings as settings
from paperminertoolkit import cli
from paperminertoolkit._cli_schema import schema_command
from paperminertoolkit._version import __version__
from paperminertoolkit.extraction.postprocess import PostprocessSummary


def _error(result: Result) -> dict[str, Any]:
    """Return the ``--json`` error object a failed command printed on stdout."""
    payload = json.loads(result.stdout)
    assert set(payload) == {'error'}
    assert payload['error']['exit_code'] == result.exit_code
    return payload['error']


def _raise(error: Exception) -> Callable[..., NoReturn]:
    """Return a stand-in workflow that raises ``error`` whenever it is called."""
    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise the configured failure."""
        raise error
    return fail


def _forbid(name: str) -> Callable[..., NoReturn]:
    """Return a stand-in that fails the test if the named workflow starts."""
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail on any call."""
        raise AssertionError(f'{name} ran')
    return forbidden


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create an empty file that stands in for a corpus the stubbed workflow never opens."""
    path = tmp_path / 'papers.db'
    path.touch()
    return path


@pytest.mark.parametrize(('arguments', 'target', 'error', 'error_type'), [
    (['import', 'pdfs', '{dir}', '{db}'], 'import_pdfs',
     OSError('PDF folder is unreadable'), 'io'),
    (['download', '{db}', '--dry-run'], 'plan_downloads',
     sqlite3.OperationalError('database is locked'), 'io'),
    (['scrape', '{db}', '--dry-run'], 'plan_scrape',
     ValueError('mode must be one of: text'), 'operational'),
    (['scrape', '{db}', '--dry-run'], 'plan_scrape',
     KeyError('No recipe named nope'), 'not_found'),
    (['scrape-progress', '{db}'], 'scrape_progress',
     RuntimeError('Corpus schema 99 is newer'), 'operational'),
    (['postprocess', '{db}', 'sse', '{csv}'], 'postprocess_scrape',
     KeyError('No recipe named nope'), 'not_found'),
    (['postprocess', '{db}', 'sse', '{csv}'], 'postprocess_scrape',
     sqlite3.DatabaseError('file is not a database'), 'io'),
    (['reset', '{db}'], 'reset', sqlite3.OperationalError('database is locked'), 'io'),
    (['config', 'model', 'text', '--provider', 'local', '--model', 'm'], 'set_model_profile',
     OSError('Settings directory is read-only'), 'io'),
    (['config', 'status'], 'get_model_profile',
     RuntimeError('Error loading settings'), 'operational'),
])
def test_workflow_failures_are_reported_once_with_their_type(
    arguments: list[str], target: str, error: Exception, error_type: str,
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn a workflow failure into exit status 1, a typed JSON error, and a stderr message."""
    pdf_dir = tmp_path / 'pdfs'
    pdf_dir.mkdir()
    scraped = tmp_path / 'scraped.csv'
    scraped.write_text('', encoding='utf-8')
    monkeypatch.setattr(cli, target, _raise(error))
    result = CliRunner().invoke(cli.main, [
        *(argument.format(db=db_path, dir=pdf_dir, csv=scraped) for argument in arguments),
        '--json'])
    message = str(error.args[0])
    assert result.exit_code == 1
    assert _error(result) == {'type': error_type, 'message': message, 'hint': None,
                              'exit_code': 1}
    assert f'Error: {message}\n' in result.stderr
    assert 'Traceback' not in result.stderr


def test_provider_failures_point_to_the_probe(
    db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Type a failed provider request as a provider error with the readiness hint."""
    monkeypatch.setattr(cli, 'download_papers',
                        _raise(requests.ConnectionError('arxiv.org refused the connection')))
    result = CliRunner().invoke(cli.main, ['download', str(db_path), '--json'])
    assert result.exit_code == 1
    assert _error(result) == {
        'type': 'provider', 'message': 'arxiv.org refused the connection',
        'hint': 'Check provider readiness with pmt probe --no-probe, then pmt probe.',
        'exit_code': 1,
    }


def test_download_dry_run_prints_the_plan_totals(
    db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summarise the requests a download would make and what is already stored."""
    plan = {
        'dry_run': True, 'format': 'both', 'sources': ['all'], 'supplement_sources': ['all'],
        'papers_considered': 3,
        'would_attempt': {'texts': 2, 'pdfs': 1, 'abstracts': 0, 'supplement_discovery': 3},
        'already_stored': {'texts': 1, 'pdfs': 2, 'abstracts': 3, 'supplement_files': 4},
        'no_text_source': 1, 'papers': [],
    }
    monkeypatch.setattr(cli, 'plan_downloads', lambda *args, **kwargs: plan)
    monkeypatch.setattr(cli, 'download_papers', _forbid('download_papers'))
    result = CliRunner().invoke(cli.main, ['download', str(db_path), '--dry-run'])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        'Dry run: 3 papers; would request 2 texts, 1 PDFs, 0 abstracts, and SI discovery '
        'for 3 papers.',
        'Already stored: 1 texts, 2 PDFs, 3 abstracts, 4 SI files; 1 papers have no '
        'reachable text source.',
    ]


def test_scrape_dry_run_warns_about_estimates_and_truncates_long_overflow_lists(
    db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warn on stderr about estimated token counts and list only the first ten overflows."""
    plan = {
        'dry_run': True, 'mode': 'text', 'recipe': 'sse',
        'model': {'provider': 'local', 'name': 'qwen', 'input_token_limit': 8000,
                  'usable_input_tokens': 7000, 'request_text_budget': 6000},
        'tokenizer': 'character estimate', 'tokenizer_fallbacks': ['local/qwen'],
        'prompt_tokens': 500, 'papers_selected': 12, 'papers_filtered_out': 0,
        'totals': {'papers_to_scrape': 11, 'skipped_done': 1, 'skipped_missing_source': 0,
                   'requests': 22, 'input_tokens': 99000, 'image_inputs': 0,
                   'supplement_files_to_scrape': 0, 'supplement_files_skipped': 0},
        'overflows': [{'paper_id': f'p:{index}', 'item': 'text', 'requests': 2,
                       'text_tokens': 9000, 'scraped': index != 0} for index in range(11)],
        'papers': [],
    }
    monkeypatch.setattr(cli, 'plan_scrape', lambda *args, **kwargs: plan)
    monkeypatch.setattr(cli, 'scrape_papers', _forbid('scrape_papers'))
    result = CliRunner().invoke(cli.main, ['scrape', str(db_path), '--dry-run'])
    assert result.exit_code == 0, result.output
    assert result.stderr == 'Warning: token counts for local/qwen are character estimates.\n'
    lines = result.stdout.splitlines()
    assert '11 inputs need more than one request:' in lines
    assert '  p:0 text: 2 requests, 9000 tokens (over the limit; would be skipped)' in lines
    assert '  p:9 text: 2 requests, 9000 tokens' in lines
    assert 'p:10' not in result.stdout
    assert lines[-1] == '  ... 1 more in --json output'


def test_postprocess_json_reports_counts_and_paths(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the post-processing summary as data rather than prose."""
    scraped = tmp_path / 'scraped.csv'
    scraped.write_text('', encoding='utf-8')
    monkeypatch.setattr(cli, 'postprocess_scrape', lambda *args, **kwargs: PostprocessSummary(
        5, 3, 2, ('paper:9',), 'out.csv', 'out.dropped.csv'))
    result = CliRunner().invoke(cli.main, ['postprocess', str(db_path), 'sse', str(scraped),
                                           '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        'records_in': 5, 'records_out': 3, 'audit_rows': 2, 'unchecked_papers': ['paper:9'],
        'output_path': 'out.csv', 'dropped_path': 'out.dropped.csv',
    }


def test_store_dry_run_prints_what_it_would_store(
    db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Describe the rows and columns a store would write, and ask the workflow for a dry run."""
    seen: dict[str, Any] = {}

    def fake_store(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Record the options and return a dry-run summary."""
        seen.update(kwargs)
        return {'status': 'dry_run', 'dry_run': True, 'input_path': 'in.csv',
                'output_path': 'out.csv', 'rows_read': 3, 'rows_written': 2, 'rows_total': 5,
                'converted_columns': ['Conductivity'], 'unconverted_columns': ['Name'],
                'skipped_columns': ['Unknown', 'Other'], 'papers_marked_stored': 0}

    monkeypatch.setattr(cli, 'store_results', fake_store)
    result = CliRunner().invoke(cli.main, ['store', str(db_path), 'in.csv', 'out.csv', 'sse',
                                           '--dry-run'])
    assert result.exit_code == 0, result.output
    assert seen == {'assume_yes': False, 'dry_run': True}
    assert result.stdout == ('Dry run: would store 2 of 3 rows in out.csv (5 rows in total), '
                             'converting 1 columns with the model and skipping 2 unmatched '
                             'columns.\n')


def test_validation_score_names_missing_run_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List the run details a saved review lacks before the per-field table."""
    review = tmp_path / 'review.json'
    review.write_text('{}', encoding='utf-8')
    score = {'fingerprint': 'f', 'sources': {}, 'recipe': 'sse', 'run': {},
             'papers_evaluated': 2, 'pending_pairs': 1, 'missing_metadata': ['provider', 'model'],
             'fields': {'Name': {'precision': 1.0, 'recall': 0.5, 'f1': None, 'tp': 1, 'fp': 0,
                                 'fn': 1, 'pending': 1}},
             'complete': False}
    monkeypatch.setattr(cli, 'saved_review', lambda path: ({'fingerprint': 'f'}, {}))
    monkeypatch.setattr(cli, 'score_review', lambda session, decisions: score)
    result = CliRunner().invoke(cli.main, ['validate', 'score', '--review', str(review)])
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[:2] == ['Recipe sse: 2 validation papers, 1 pending pairs; complete: no.',
                         'Missing run metadata: provider, model.']
    assert lines[-1].split() == ['Name', '1.000', '0.500', '-', '1', '0', '1', '1']


@pytest.mark.parametrize(('content', 'error_type'), [
    (None, 'not_found'),
    ('{not json', 'operational'),
])
def test_recipe_list_reports_an_unreadable_recipe_file(
    content: str | None, error_type: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a missing or damaged bundled recipe file as a command error."""
    recipes = tmp_path / 'recipes.json'
    if content is not None:
        recipes.write_text(content, encoding='utf-8')
    monkeypatch.setattr(cli, 'RECIPES_PATH', recipes)
    result = CliRunner().invoke(cli.main, ['recipe', 'list', '--json'])
    assert result.exit_code == 1
    assert _error(result)['type'] == error_type
    assert result.stderr == f'Error: {_error(result)["message"]}\n'


def test_recipe_list_prints_one_bundled_name_per_line() -> None:
    """Print every bundled recipe name, sorted, one per line."""
    result = CliRunner().invoke(cli.main, ['recipe', 'list'])
    assert result.exit_code == 0, result.output
    names = result.stdout.splitlines()
    assert 'sse' in names
    assert names == sorted(json.loads(cli.RECIPES_PATH.read_text(encoding='utf-8')))


class _Terminal(io.StringIO):
    """Standard input that reports itself as an interactive terminal."""

    def isatty(self) -> bool:
        """Claim a terminal, as an interactive shell's stdin would."""
        return True


def test_setter_prompts_when_stdin_is_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Fall back to the interactive prompt at a terminal and report it as the value's source."""
    # CliRunner swaps in a non-terminal stdin, so run the command without it.
    monkeypatch.setattr(sys, 'stdin', _Terminal('reader@example.org\n'))
    cli.main.main(['config', 'unpaywall-email', '--json'], standalone_mode=False)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {'setting': 'unpaywall_email', 'secret': False,
                                        'source': 'prompt', 'value': 'reader@example.org'}
    assert 'Enter Unpaywall email: ' in captured.err
    assert settings._saved_settings()['unpaywall_email'] == 'reader@example.org'


def test_setter_reports_an_unknown_value_when_the_settings_cannot_be_reread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm a saved setting even when reading it back fails, reporting its value as null."""
    monkeypatch.setattr(cli, '_saved_settings', _raise(OSError('settings file vanished')))
    result = CliRunner().invoke(cli.main, ['config', 'crossref-email', '--value-stdin', '--json'],
                                input='team@example.org\n')
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {'setting': 'crossref_email', 'secret': False,
                                         'source': 'stdin', 'value': None}
    assert settings._saved_settings()['crossref_email'] == 'team@example.org'


@pytest.mark.parametrize(('content', 'problem'), [
    ('{not json', 'Error loading'),
    ('["not", "an", "object"]', 'expected a JSON object'),
])
def test_setters_leave_an_unreadable_settings_file_untouched(
    content: str, problem: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse to overwrite a damaged settings file and still describe the failure as JSON."""
    path = Path(settings.SETTINGS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    monkeypatch.setenv('PMT_TEST_CONTACT_EMAIL', 'team@example.org')
    result = CliRunner().invoke(cli.main, ['config', 'crossref-email',
                                           '--value-from-env', 'PMT_TEST_CONTACT_EMAIL', '--json'])
    assert result.exit_code == 1
    error = _error(result)
    assert error['type'] == 'operational'
    assert problem in error['message']
    assert settings.SETTINGS_FILE in error['message']
    assert path.read_text(encoding='utf-8') == content
    # Credential redaction cannot read the file either and must not mask the error.
    assert agent._redaction_values() == []


@pytest.mark.parametrize(('options', 'message'), [
    (['--api-key', 'sk-inline', '--api-key-from-env', 'PMT_TEST_MODEL_KEY'],
     'Use --api-key or --api-key-from-env, not both.'),
    (['--api-key-from-env', 'PMT_TEST_UNSET_MODEL_KEY'],
     'Environment variable PMT_TEST_UNSET_MODEL_KEY is not set.'),
])
def test_config_model_rejects_conflicting_or_missing_key_sources(
    options: list[str], message: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an ambiguous or absent API key source before saving the profile."""
    monkeypatch.setenv('PMT_TEST_MODEL_KEY', 'sk-from-env')
    monkeypatch.delenv('PMT_TEST_UNSET_MODEL_KEY', raising=False)
    monkeypatch.setattr(cli, 'set_model_profile', _forbid('set_model_profile'))
    result = CliRunner().invoke(cli.main, ['config', 'model', 'text', '--provider', 'openai',
                                           '--model', 'gpt-test', *options, '--json'])
    assert result.exit_code == 2
    assert _error(result)['type'] == 'usage'
    assert message in _error(result)['message']
    assert message in result.stderr


def test_mcp_announces_its_tools_on_stderr_and_answers_on_stdout() -> None:
    """Keep stdout for JSON-RPC responses and report the number of tools on stderr."""
    messages = [{'jsonrpc': '2.0', 'id': 1, 'method': 'ping'},
                {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'}]
    result = CliRunner().invoke(cli.main, ['mcp'],
                                input=''.join(f'{json.dumps(message)}\n' for message in messages))
    assert result.exit_code == 0, result.output
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert responses[0] == {'jsonrpc': '2.0', 'id': 1, 'result': {}}
    names = {tool['name'] for tool in responses[1]['result']['tools']}
    assert 'corpus_list' in names and 'download' not in names
    assert result.stderr == (f'pmt MCP server: {len(names)} tools; waiting for JSON-RPC '
                             'on stdin.\n')


def test_json_after_a_double_dash_is_an_argument_not_a_request_for_json(tmp_path: Path) -> None:
    """Leave the error object out when --json follows the ``--`` separator."""
    result = CliRunner().invoke(cli.main, ['download', '--', '--json'])
    assert result.exit_code == 2
    assert result.stdout == ''
    assert '--json' in result.stderr and 'does not exist' in result.stderr
    missing = CliRunner().invoke(cli.main, ['download', str(tmp_path / 'missing.db'), '--json'])
    assert missing.exit_code == 2
    assert _error(missing)['type'] == 'usage'


def test_error_objects_that_break_their_schema_fail_loudly_in_tests(
    db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise instead of printing an error object that drifts from the published schema."""
    monkeypatch.setattr(agent, 'error_payload',
                        lambda error, path: {'error': {'type': 'io', 'message': str(error)}})
    monkeypatch.setattr(cli, 'reset', _raise(sqlite3.OperationalError('database is locked')))
    result = CliRunner().invoke(cli.main, ['reset', str(db_path), '--json'])
    assert result.exit_code == 1
    assert isinstance(result.exception, AssertionError)
    assert "missing required key 'exit_code'" in str(result.exception)
    assert result.stdout == ''


def test_json_default_falls_back_to_the_string_form() -> None:
    """Encode values without a NumPy-style conversion, such as dates, as their text."""
    assert agent._json_default(date(2026, 1, 2)) == '2026-01-02'


def test_an_undecodable_paper_ids_file_is_a_usage_error(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name the selection file when it is not UTF-8 text, and never start the workflow."""
    ids_file = tmp_path / 'ids.txt'
    ids_file.write_bytes(b'paper:1\n\xff\xfe\n')
    monkeypatch.setattr(cli, 'download_papers', _forbid('download_papers'))
    result = CliRunner().invoke(cli.main, ['download', str(db_path), '--paper-ids-file',
                                           str(ids_file), '--json'])
    assert result.exit_code == 2
    assert _error(result)['type'] == 'usage'
    assert '--paper-ids-file' in result.stderr
    assert "'utf-8' codec can't decode" in _error(result)['message']


def test_paper_ids_cannot_be_looked_up_in_a_file_that_is_not_a_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an unreadable corpus while resolving --paper-id, and never start the workflow."""
    db = tmp_path / 'papers.db'
    db.write_text('not a sqlite database', encoding='utf-8')
    monkeypatch.setattr(cli, 'download_papers', _forbid('download_papers'))
    result = CliRunner().invoke(cli.main, ['download', str(db), '--paper-id', 'paper:1',
                                           '--json'])
    assert result.exit_code == 1
    error = _error(result)
    assert error['type'] == 'io'
    assert error['message'].startswith(f'Cannot look up paper IDs in {db}: ')
    assert 'not a database' in error['message']


def test_schema_lists_a_set_default_in_sorted_order() -> None:
    """Describe a set-valued default as a stable, sorted JSON list."""
    root = click.Group('temporary')
    root.add_command(schema_command)
    root.add_command(click.Command('child', params=[
        click.Option(['--effects'], default=frozenset({'writes_files', 'network'}))]))
    result = CliRunner().invoke(root, ['schema', 'child'])
    assert result.exit_code == 0, result.output
    effects = next(parameter for parameter in json.loads(result.stdout)['command']['parameters']
                   if parameter['name'] == 'effects')
    assert effects['default'] == ['network', 'writes_files']


def test_module_entry_point_runs_the_cli_in_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Run ``python -m paperminertoolkit`` in this process and exit cleanly after --version."""
    monkeypatch.setattr(sys, 'argv', ['pmt', '--version'])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module('paperminertoolkit', run_name='__main__')
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
