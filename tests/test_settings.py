"""Tests for paperminertoolkit.settings.

This module tests configuration parsing, environment overrides, profile
inference, and interactive key/profile update helpers. Live API validation
tests are marked as network tests and read the user's real configuration only
when explicitly requested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

import paperminertoolkit.settings as settings


API_ENV_KEYS = [
    'ELSEVIER_API_KEY',
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'CORE_API_KEY',
    'UNPAYWALL_EMAIL',
    'OPENALEX_API_KEY',
    'CROSSREF_EMAIL',
    'NCBI_API_KEY',
    'NCBI_EMAIL',
]

MODEL_ENV_KEYS = [
    'PAPERMINERTOOLKIT_MODEL_PROVIDER',
    'PAPERMINERTOOLKIT_MODEL_NAME',
    'PAPERMINERTOOLKIT_MODEL_BASE_URL',
    'PAPERMINERTOOLKIT_MODEL_API_KEY',
    'PAPERMINERTOOLKIT_MODEL_CAPABILITIES',
    'PAPERMINERTOOLKIT_MODEL_TEMPERATURE',
    'PAPERMINERTOOLKIT_MODEL_TOP_P',
    'PAPERMINERTOOLKIT_VISION_MODEL_PROVIDER',
    'PAPERMINERTOOLKIT_VISION_MODEL_NAME',
    'PAPERMINERTOOLKIT_VISION_MODEL_BASE_URL',
    'PAPERMINERTOOLKIT_VISION_MODEL_API_KEY',
    'PAPERMINERTOOLKIT_VISION_MODEL_CAPABILITIES',
    'PAPERMINERTOOLKIT_VISION_MODEL_TEMPERATURE',
    'PAPERMINERTOOLKIT_VISION_MODEL_TOP_P',
]


@pytest.fixture
def isolated_settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point settings reads/writes at a temporary config file and clear env overrides."""
    settings_path = tmp_path / 'paperminertoolkitrc.json'
    monkeypatch.setattr(settings, 'SETTINGS_FILE', str(settings_path))
    for key in API_ENV_KEYS + MODEL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return settings_path


def test_float_setting_uses_defaults_and_converts_values() -> None:
    """Convert numeric settings and use defaults for missing values."""
    assert settings._float_setting(None, 0.5) == 0.5
    assert settings._float_setting('', 0.5) == 0.5
    assert settings._float_setting('0.25', 0.5) == 0.25


def test_capabilities_normalizes_missing_strings_and_lists() -> None:
    """Normalize missing, string, and list capability settings."""
    assert settings._capabilities(None) == ['text']
    assert settings._capabilities('', ['text', 'vision']) == ['text', 'vision']
    assert settings._capabilities('text, vision') == ['text', 'vision']
    assert settings._capabilities(['text']) == ['text']


def test_merge_profile_applies_defaults_and_coerces_types() -> None:
    """Merge profile defaults while coercing configured value types."""
    merged = settings._merge_profile(
        settings.DEFAULT_MODEL_PROFILE,
        {
            'model': 'custom-model',
            'capabilities': 'text,vision',
            'temperature': '0.2',
            'top_p': '0.9',
            'input_token_limit': '64000',
        },
    )

    assert merged['provider'] == 'openai'
    assert merged['model'] == 'custom-model'
    assert merged['capabilities'] == ['text', 'vision']
    assert merged['temperature'] == 0.2
    assert merged['top_p'] == 0.9
    assert merged['input_token_limit'] == 64000


def test_merge_profile_defaults_missing_input_token_limit() -> None:
    """Default an empty model input token limit."""
    merged = settings._merge_profile(
        settings.DEFAULT_MODEL_PROFILE,
        {'input_token_limit': ''},
    )

    assert merged['input_token_limit'] == settings.DEFAULT_INPUT_TOKEN_LIMIT


def test_env_profile_collects_only_defined_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect only defined environment values for a model profile."""
    monkeypatch.setenv('PAPERMINERTOOLKIT_MODEL_PROVIDER', 'local')
    monkeypatch.setenv('PAPERMINERTOOLKIT_MODEL_NAME', 'qwen')
    monkeypatch.setenv('PAPERMINERTOOLKIT_MODEL_INPUT_TOKEN_LIMIT', '120000')

    profile = settings._env_profile('PAPERMINERTOOLKIT_MODEL_')

    assert profile == {'provider': 'local', 'model': 'qwen', 'input_token_limit': '120000'}


def test_load_settings_returns_defaults_when_config_file_is_missing(isolated_settings_file: Path) -> None:
    """Load default model profiles when the config file is missing."""
    loaded = settings.load_settings()

    assert loaded['model_profiles']['text']['capabilities'] == ['text']
    assert loaded['model_profiles']['vision']['capabilities'] == ['text', 'vision']


def test_load_settings_merges_file_and_environment_overrides(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply environment overrides after values loaded from the config file."""
    isolated_settings_file.write_text(json.dumps({
        'elsevier_api_key': 'file-elsevier',
        'model_profiles': {
            'text': {
                'provider': 'openai',
                'model': 'file-model',
                'temperature': '0.1',
            }
        },
    }))
    monkeypatch.setenv('ELSEVIER_API_KEY', 'env-elsevier')
    monkeypatch.setenv('PAPERMINERTOOLKIT_MODEL_PROVIDER', 'local')
    monkeypatch.setenv('PAPERMINERTOOLKIT_MODEL_NAME', 'env-model')
    monkeypatch.setenv('PAPERMINERTOOLKIT_MODEL_BASE_URL', 'http://127.0.0.1:8000/v1')
    monkeypatch.setenv('PAPERMINERTOOLKIT_MODEL_CAPABILITIES', 'text,vision')

    loaded = settings.load_settings()

    assert loaded['elsevier_api_key'] == 'env-elsevier'
    assert loaded['model_profiles']['text']['provider'] == 'local'
    assert loaded['model_profiles']['text']['model'] == 'env-model'
    assert loaded['model_profiles']['text']['base_url'] == 'http://127.0.0.1:8000/v1'
    assert loaded['model_profiles']['text']['capabilities'] == ['text', 'vision']
    assert loaded['model_profiles']['text']['temperature'] == 0.1


def test_load_settings_reports_invalid_config_file(isolated_settings_file: Path) -> None:
    """Report invalid JSON with the config file path."""
    isolated_settings_file.write_text('{not-json')

    with pytest.raises(RuntimeError, match=str(isolated_settings_file)):
        settings.load_settings()


def test_load_settings_applies_all_api_environment_overrides(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Override every stored service credential from the environment."""
    isolated_settings_file.write_text(json.dumps({
        'elsevier_api_key': 'file-elsevier',
        'openai_api_key': 'file-openai',
        'anthropic_api_key': 'file-anthropic',
        'core_api_key': 'file-core',
        'core_min_interval': 2.0,
        'unpaywall_email': 'file@example.com',
        'openalex_api_key': 'file-openalex',
        'crossref_email': 'file-crossref@example.com',
        'ncbi_api_key': 'file-ncbi',
        'ncbi_email': 'file-ncbi@example.com',
    }))
    monkeypatch.setenv('ELSEVIER_API_KEY', 'env-elsevier')
    monkeypatch.setenv('OPENAI_API_KEY', 'env-openai')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'env-anthropic')
    monkeypatch.setenv('CORE_API_KEY', 'env-core')
    monkeypatch.setenv('CORE_MIN_INTERVAL', '0.25')
    monkeypatch.setenv('UNPAYWALL_EMAIL', 'env@example.com')
    monkeypatch.setenv('OPENALEX_API_KEY', 'env-openalex')
    monkeypatch.setenv('CROSSREF_EMAIL', 'env-crossref@example.com')
    monkeypatch.setenv('NCBI_API_KEY', 'env-ncbi')
    monkeypatch.setenv('NCBI_EMAIL', 'env-ncbi@example.com')

    loaded = settings.load_settings()

    assert loaded['elsevier_api_key'] == 'env-elsevier'
    assert loaded['openai_api_key'] == 'env-openai'
    assert loaded['anthropic_api_key'] == 'env-anthropic'
    assert loaded['core_api_key'] == 'env-core'
    assert loaded['core_min_interval'] == '0.25'
    assert loaded['unpaywall_email'] == 'env@example.com'
    assert loaded['openalex_api_key'] == 'env-openalex'
    assert loaded['crossref_email'] == 'env-crossref@example.com'
    assert loaded['ncbi_api_key'] == 'env-ncbi'
    assert loaded['ncbi_email'] == 'env-ncbi@example.com'


def test_load_settings_applies_vision_model_environment_overrides(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Override the complete vision model profile from the environment."""
    monkeypatch.setenv('PAPERMINERTOOLKIT_VISION_MODEL_PROVIDER', 'local')
    monkeypatch.setenv('PAPERMINERTOOLKIT_VISION_MODEL_NAME', 'qwen-vl')
    monkeypatch.setenv('PAPERMINERTOOLKIT_VISION_MODEL_BASE_URL', 'http://127.0.0.1:8000/v1')
    monkeypatch.setenv('PAPERMINERTOOLKIT_VISION_MODEL_API_KEY', 'local-key')
    monkeypatch.setenv('PAPERMINERTOOLKIT_VISION_MODEL_CAPABILITIES', 'text,vision')
    monkeypatch.setenv('PAPERMINERTOOLKIT_VISION_MODEL_TEMPERATURE', '0.4')
    monkeypatch.setenv('PAPERMINERTOOLKIT_VISION_MODEL_TOP_P', '0.7')
    monkeypatch.setenv('PAPERMINERTOOLKIT_VISION_MODEL_INPUT_TOKEN_LIMIT', '96000')

    profile = settings.load_settings()['model_profiles']['vision']

    assert profile['provider'] == 'local'
    assert profile['model'] == 'qwen-vl'
    assert profile['base_url'] == 'http://127.0.0.1:8000/v1'
    assert profile['api_key'] == 'local-key'
    assert profile['capabilities'] == ['text', 'vision']
    assert profile['temperature'] == 0.4
    assert profile['top_p'] == 0.7
    assert profile['input_token_limit'] == 96000


def test_save_settings_writes_json_to_config_file(isolated_settings_file: Path) -> None:
    """Write supplied settings as JSON to the config file."""
    settings._save_settings({'core_api_key': 'core-key'})

    assert isolated_settings_file.is_file()
    assert json.loads(isolated_settings_file.read_text()) == {'core_api_key': 'core-key'}


def test_get_model_profile_returns_profile_and_rejects_missing_profile(isolated_settings_file: Path) -> None:
    """Return named model profiles and reject unknown names."""
    assert settings.get_model_profile('text')['provider'] == 'openai'

    with pytest.raises(KeyError):
        settings.get_model_profile('missing')


def test_infer_model_capabilities_detects_vision_models() -> None:
    """Infer vision capability from the profile or model name."""
    assert settings.infer_model_capabilities('text', 'gpt-4') == ['text']
    assert settings.infer_model_capabilities('vision', 'gpt-4') == ['text', 'vision']
    assert settings.infer_model_capabilities('text', 'Qwen/Qwen3-VL-30B') == ['text', 'vision']


def test_set_model_profile_persists_profile_values(isolated_settings_file: Path) -> None:
    """Persist every configured model profile value."""
    settings.set_model_profile(
        'text',
        'local',
        'qwen',
        base_url='http://127.0.0.1:8000/v1',
        api_key='not-needed',
        capabilities=['text'],
        temperature=0.3,
        top_p=0.8,
        input_token_limit=120000,
    )

    loaded = settings.load_settings()
    profile = loaded['model_profiles']['text']

    assert profile['provider'] == 'local'
    assert profile['model'] == 'qwen'
    assert profile['base_url'] == 'http://127.0.0.1:8000/v1'
    assert profile['api_key'] == 'not-needed'
    assert profile['capabilities'] == ['text']
    assert profile['temperature'] == 0.3
    assert profile['top_p'] == 0.8
    assert profile['input_token_limit'] == 120000


def test_update_anthropic_key_prompts_and_saves_key(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Prompt for and save an Anthropic API key."""
    settings._save_settings({'anthropic_api_key': 'old-anthropic-key'})
    monkeypatch.setattr('builtins.input', lambda _: 'anthropic-key')

    settings.update_anthropic_key()

    output = capsys.readouterr().out
    assert 'Current Anthropic API key: old-...-key' in output
    assert settings.load_settings()['anthropic_api_key'] == 'anthropic-key'


def test_check_openai_api_key_returns_true_for_valid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept an OpenAI key when listing models succeeds."""

    class FakeModels:
        """Provide a successful fake models resource."""

        def list(self) -> list[object]:
            """Return an empty model listing."""
            return []

    class FakeOpenAI:
        """Provide a fake authenticated OpenAI client."""

        def __init__(self, api_key: str) -> None:
            """Store the API key and expose the fake models resource."""
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr(settings.openai, 'OpenAI', FakeOpenAI)

    assert settings._check_openai_api_key('placeholder-openai-key') is True


def test_check_openai_api_key_returns_false_for_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an OpenAI key when listing models fails authentication."""

    class FakeAuthenticationError(Exception):
        """Represent a fake OpenAI authentication failure."""

        pass

    class FakeModels:
        """Provide a model resource that rejects authentication."""

        def list(self) -> NoReturn:
            """Raise the fake authentication error."""
            raise FakeAuthenticationError()

    class FakeOpenAI:
        """Provide a fake unauthenticated OpenAI client."""

        def __init__(self, api_key: str) -> None:
            """Store the API key and expose the failing models resource."""
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr(settings.openai, 'AuthenticationError', FakeAuthenticationError)
    monkeypatch.setattr(settings.openai, 'OpenAI', FakeOpenAI)

    assert settings._check_openai_api_key('placeholder-openai-key') is False


def test_update_openai_key_prompts_validates_and_saves_key(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Prompt for, validate, and save an OpenAI API key."""
    settings._save_settings({'openai_api_key': 'old-openai-key'})
    monkeypatch.setattr('builtins.input', lambda _: 'openai-key')
    monkeypatch.setattr(settings, '_check_openai_api_key', lambda api_key: api_key == 'openai-key')

    settings.update_openai_key()

    output = capsys.readouterr().out
    assert 'Current OpenAI API key: old-...-key' in output
    assert settings.load_settings()['openai_api_key'] == 'openai-key'


def test_update_openai_key_rejects_invalid_key(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject and avoid saving an invalid OpenAI API key."""
    monkeypatch.setattr('builtins.input', lambda _: 'bad-openai-key')
    monkeypatch.setattr(settings, '_check_openai_api_key', lambda _: False)

    with pytest.raises(ValueError, match='OpenAI API key is invalid'):
        settings.update_openai_key()

    assert 'openai_api_key' not in settings.load_settings()


def test_update_core_key_prompts_and_saves_key(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Prompt for and save a CORE API key."""
    settings._save_settings({'core_api_key': 'old-core-key'})
    monkeypatch.setattr('builtins.input', lambda _: 'core-key')

    settings.update_core_key()

    output = capsys.readouterr().out
    assert 'Current CORE API key: old-...-key' in output
    assert settings.load_settings()['core_api_key'] == 'core-key'


def test_update_core_rate_saves_a_granted_pace(
    isolated_settings_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Save the seconds CORE granted between requests, and show the default.

    CORE publishes no figure for the faster paces it grants, so the pace is
    entered directly and the prompt states the free allowance it replaces.
    """
    monkeypatch.setattr('builtins.input', lambda _: '0.2')

    settings.update_core_rate()

    output = capsys.readouterr().out
    assert settings.load_settings()['core_min_interval'] == 0.2
    assert '5 requests per 10 seconds' in output
    assert '2.0 seconds between requests' in output


def test_update_core_rate_clears_the_pace_and_rejects_bad_values(
    isolated_settings_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return to the free allowance on a blank entry and refuse a bad pace."""
    settings._save_settings({'core_min_interval': 0.2})
    monkeypatch.setattr('builtins.input', lambda _: '')
    settings.update_core_rate()
    assert 'core_min_interval' not in settings.load_settings()

    monkeypatch.setattr('builtins.input', lambda _: 'quickly')
    with pytest.raises(ValueError, match='must be a number of seconds'):
        settings.update_core_rate()

    monkeypatch.setattr('builtins.input', lambda _: '0')
    with pytest.raises(ValueError, match='greater than zero'):
        settings.update_core_rate()


def test_check_elsevier_api_key_delegates_to_the_elsevier_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate a key through the Elsevier module rather than inline here.

    settings used to import paperminertoolkit.providers.elsevier at module level for this one
    function, which was the single edge stopping the Elsevier client from
    importing settings the way every other source module does.
    """
    import paperminertoolkit.providers.elsevier as elsevier

    calls = []
    monkeypatch.setattr(elsevier, 'check_api_key',
                        lambda key, **_: calls.append(key) or True)
    assert settings._check_elsevier_api_key('placeholder-elsevier-key') is True
    assert calls == ['placeholder-elsevier-key']

    monkeypatch.setattr(elsevier, 'check_api_key', lambda *_, **__: False)
    assert settings._check_elsevier_api_key('bad-key') is False


def test_update_elsevier_key_prompts_validates_and_saves_key(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Prompt for, validate, and save an Elsevier API key."""
    settings._save_settings({'elsevier_api_key': 'old-elsevier-key'})
    monkeypatch.setattr('builtins.input', lambda _: 'elsevier-key')
    monkeypatch.setattr(settings, '_check_elsevier_api_key', lambda api_key: api_key == 'elsevier-key')

    settings.update_elsevier_key()

    output = capsys.readouterr().out
    assert 'Current Elsevier API key: old-...-key' in output
    assert settings.load_settings()['elsevier_api_key'] == 'elsevier-key'


def test_update_elsevier_key_rejects_invalid_key(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject and avoid saving an invalid Elsevier API key."""
    monkeypatch.setattr('builtins.input', lambda _: 'bad-elsevier-key')
    monkeypatch.setattr(settings, '_check_elsevier_api_key', lambda _: False)

    with pytest.raises(ValueError, match='Elsevier API key is invalid'):
        settings.update_elsevier_key()

    assert 'elsevier_api_key' not in settings.load_settings()


def test_update_unpaywall_email_validates_and_saves_email(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Save a valid Unpaywall email and reject an invalid one."""
    settings._save_settings({'unpaywall_email': 'old@example.com'})
    monkeypatch.setattr('builtins.input', lambda _: 'person@example.com')
    settings.update_unpaywall_email()
    output = capsys.readouterr().out
    assert 'Current Unpaywall email: old@example.com' in output
    assert settings.load_settings()['unpaywall_email'] == 'person@example.com'

    monkeypatch.setattr('builtins.input', lambda _: 'not-an-email')
    with pytest.raises(ValueError):
        settings.update_unpaywall_email()


def test_update_crossref_email_validates_and_saves_email(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Save a valid Crossref email and reject an invalid one."""
    settings._save_settings({'crossref_email': 'old@example.com'})
    monkeypatch.setattr('builtins.input', lambda _: 'person@example.com')
    settings.update_crossref_email()
    output = capsys.readouterr().out
    assert 'Current Crossref email: old@example.com' in output
    assert settings.load_settings()['crossref_email'] == 'person@example.com'

    monkeypatch.setattr('builtins.input', lambda _: 'not-an-email')
    with pytest.raises(ValueError):
        settings.update_crossref_email()


def test_load_settings_reads_crossref_email_from_the_settings_file(isolated_settings_file: Path) -> None:
    """Read a stored Crossref email through the settings-file whitelist."""
    isolated_settings_file.write_text(json.dumps({'crossref_email': 'file@example.com'}))

    assert settings.load_settings()['crossref_email'] == 'file@example.com'


def test_update_openalex_key_validates_and_saves_key(isolated_settings_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Save a valid OpenAlex key and reject an invalid one."""
    settings._save_settings({'openalex_api_key': 'old-openalex-key'})
    monkeypatch.setattr('builtins.input', lambda _: 'openalex-key')
    monkeypatch.setattr(settings, '_check_openalex_api_key', lambda api_key: api_key == 'openalex-key')

    settings.update_openalex_key()

    output = capsys.readouterr().out
    assert 'Current OpenAlex API key: old-...-key' in output
    assert settings.load_settings()['openalex_api_key'] == 'openalex-key'

    monkeypatch.setattr('builtins.input', lambda _: 'wrong-key')
    with pytest.raises(ValueError):
        settings.update_openalex_key()


def test_check_openalex_api_key_only_rejects_an_explicit_401(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject only explicit OpenAlex authentication failures."""

    class FakeResponse:
        """Provide a response with a configurable status code."""

        def __init__(self, status_code: int) -> None:
            """Store the response status code."""
            self.status_code = status_code

    monkeypatch.setattr(settings.requests, 'get', lambda *_, **__: FakeResponse(401))
    assert settings._check_openalex_api_key('bad-key') is False

    monkeypatch.setattr(settings.requests, 'get', lambda *_, **__: FakeResponse(200))
    assert settings._check_openalex_api_key('good-key') is True


def test_display_helpers_and_ncbi_updates_cover_optional_values(
    isolated_settings_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mask short secrets and save optional NCBI identity settings."""
    assert settings._mask_secret('short') == '********'
    settings._show_current_setting({}, 'email', 'email', secret=False)
    assert 'Current email: not set' in capsys.readouterr().out

    key_settings: dict[str, object] = {}
    email_settings: dict[str, object] = {}
    answers = iter(['ncbi-key', 'person@example.org'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    settings.update_ncbi_key(key_settings)
    settings.update_ncbi_email(email_settings)
    assert key_settings['ncbi_api_key'] == 'ncbi-key'
    assert email_settings['ncbi_email'] == 'person@example.org'

    loaded: dict[str, object] = {}
    monkeypatch.setattr(settings, 'load_settings', lambda: loaded)
    monkeypatch.setattr('builtins.input', lambda _: 'loaded-key')
    settings.update_ncbi_key()
    assert loaded['ncbi_api_key'] == 'loaded-key'
    monkeypatch.setattr('builtins.input', lambda _: 'loaded@example.org')
    settings.update_ncbi_email()
    assert loaded['ncbi_email'] == 'loaded@example.org'

    monkeypatch.setattr('builtins.input', lambda _: 'invalid')
    with pytest.raises(ValueError, match='valid email'):
        settings.update_ncbi_email({})

    def unreachable(*_: object, **__: object) -> NoReturn:
        """Simulate an unreachable OpenAlex API."""
        raise settings.requests.ConnectionError('offline')

    monkeypatch.setattr(settings.requests, 'get', unreachable)
    assert settings._check_openalex_api_key('good-key') is True


@pytest.mark.network
def test_check_openai_api_key_validates_configured_key() -> None:
    """Validate the configured OpenAI key against the live models API."""
    loaded = settings.load_settings()
    api_key = loaded.get('openai_api_key')

    assert api_key, 'Set openai_api_key in ~/.config/.paperminertoolkitrc.json or OPENAI_API_KEY before running network tests.'
    assert settings._check_openai_api_key(api_key) is True


@pytest.mark.network
def test_check_elsevier_api_key_validates_configured_key() -> None:
    """Validate the configured Elsevier key against live Scopus search."""
    loaded = settings.load_settings()
    api_key = loaded.get('elsevier_api_key')

    assert api_key, 'Set elsevier_api_key in ~/.config/.paperminertoolkitrc.json or ELSEVIER_API_KEY before running network tests.'
    assert settings._check_elsevier_api_key(api_key) is True
