"""Configure credentials and model profiles without an interactive terminal."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import paperminertoolkit.settings as settings
from paperminertoolkit import cli


def test_setter_reads_the_environment_and_never_prints_the_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save a key named by --value-from-env and report only where it came from."""
    monkeypatch.setenv('MY_CORE_KEY', 'core-secret-123')
    result = CliRunner().invoke(cli.main, ['config', 'core-key', '--value-from-env', 'MY_CORE_KEY',
                                           '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {'setting': 'core_api_key', 'secret': True, 'source': 'env'}
    assert 'core-secret-123' not in result.output
    assert settings._saved_settings()['core_api_key'] == 'core-secret-123'


def test_setter_without_a_value_fails_instead_of_prompting() -> None:
    """Refuse to wait for input that an unattended caller will never send."""
    result = CliRunner().invoke(cli.main, ['config', 'crossref-email', '--json'])
    assert result.exit_code == 2
    error = json.loads(result.stdout)['error']
    assert error['type'] == 'usage'
    assert '--value-from-env VAR or --value-stdin' in error['message']
    assert 'crossref_email' not in settings._saved_settings()


@pytest.mark.parametrize(('arguments', 'message'), [
    (['--value-from-env', 'X', '--value-stdin'], 'not both'),
    (['--value-from-env', 'PMT_TEST_UNSET_VARIABLE'], 'PMT_TEST_UNSET_VARIABLE is not set'),
    (['--value-stdin'], 'Standard input was empty'),
])
def test_setter_rejects_conflicting_or_missing_sources(arguments: list[str], message: str) -> None:
    """Name the problem with the value source as a usage error."""
    result = CliRunner().invoke(cli.main, ['config', 'ncbi-key', *arguments], input='')
    assert result.exit_code == 2
    assert message in result.stderr


def test_setter_reports_a_rejected_key_and_saves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn a failed provider check into a command error, not a traceback."""
    monkeypatch.setattr(settings, '_check_openai_api_key', lambda _: False)
    result = CliRunner().invoke(cli.main, ['config', 'openai-key', '--value-stdin', '--json'],
                                input='sk-wrong\n')
    assert result.exit_code == 1
    assert json.loads(result.stdout)['error']['message'] == 'OpenAI API key is invalid.'
    assert 'sk-wrong' not in result.output
    assert 'openai_api_key' not in settings._saved_settings()


def test_core_rate_accepts_a_blank_value_to_restore_the_default() -> None:
    """Keep the prompt's blank-means-default rule for piped values."""
    runner = CliRunner()
    assert runner.invoke(cli.main, ['config', 'core-rate', '--value-stdin'], input='3\n').exit_code == 0
    assert settings._saved_settings()['core_min_interval'] == 3.0
    result = runner.invoke(cli.main, ['config', 'core-rate', '--value-stdin', '--json'], input='\n')
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['value'] is None
    assert 'core_min_interval' not in settings._saved_settings()


def test_config_model_needs_its_options_without_a_terminal() -> None:
    """Report the missing options rather than prompting on a closed stdin."""
    result = CliRunner().invoke(cli.main, ['config', 'model', 'text', '--provider', 'local'])
    assert result.exit_code == 2
    assert 'Missing --model' in result.stderr


def test_config_model_reads_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Save a profile non-interactively and report it without the key."""
    monkeypatch.setenv('PMT_TEST_MODEL_KEY', 'model-secret')
    result = CliRunner().invoke(cli.main, [
        'config', 'model', 'text', '--provider', 'openai', '--model', 'gpt-test',
        '--api-key-from-env', 'PMT_TEST_MODEL_KEY', '--json'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload['provider'] == 'openai' and payload['model'] == 'gpt-test'
    assert payload['api_key_set'] is True
    assert 'model-secret' not in result.output
    assert settings._saved_settings()['model_profiles']['text']['api_key'] == 'model-secret'


def test_config_model_saves_and_reports_reasoning_settings() -> None:
    """Save a local profile's reasoning effort and thinking switch and echo them as JSON."""
    result = CliRunner().invoke(cli.main, [
        'config', 'model', 'text', '--provider', 'local', '--model', 'Qwen/Qwen3-8B',
        '--base-url', 'http://localhost:8000/v1', '--reasoning-effort', 'low', '--no-thinking',
        '--json'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert (payload['reasoning_effort'], payload['thinking']) == ('low', False)
    profile = settings._saved_settings()['model_profiles']['text']
    assert (profile['reasoning_effort'], profile['thinking']) == ('low', False)


def test_config_model_checks_reasoning_settings_before_reading_the_key() -> None:
    """Reject a reasoning setting the provider cannot take without saving anything."""
    result = CliRunner().invoke(cli.main, [
        'config', 'model', 'text', '--provider', 'claude-code', '--model', 'sonnet',
        '--reasoning-effort', 'high', '--api-key-from-env', 'PMT_UNSET_MODEL_KEY'])
    assert result.exit_code == 2
    assert 'not supported for provider "claude-code"' in result.stderr
    assert 'model_profiles' not in settings._saved_settings()
