"""Tests for paperscraper.settings.

This module tests configuration parsing, environment overrides, profile
inference, and interactive key/profile update helpers. Live API validation
tests are marked as network tests and read the user's real configuration only
when explicitly requested.
"""

import json

import pytest

import paperscraper.settings as settings


API_ENV_KEYS = [
    'ELSEVIER_API_KEY',
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'CORE_API_KEY',
    'UNPAYWALL_EMAIL',
]

MODEL_ENV_KEYS = [
    'PAPERSCRAPER_MODEL_PROVIDER',
    'PAPERSCRAPER_MODEL_NAME',
    'PAPERSCRAPER_MODEL_BASE_URL',
    'PAPERSCRAPER_MODEL_API_KEY',
    'PAPERSCRAPER_MODEL_CAPABILITIES',
    'PAPERSCRAPER_MODEL_TEMPERATURE',
    'PAPERSCRAPER_MODEL_TOP_P',
    'PAPERSCRAPER_VISION_MODEL_PROVIDER',
    'PAPERSCRAPER_VISION_MODEL_NAME',
    'PAPERSCRAPER_VISION_MODEL_BASE_URL',
    'PAPERSCRAPER_VISION_MODEL_API_KEY',
    'PAPERSCRAPER_VISION_MODEL_CAPABILITIES',
    'PAPERSCRAPER_VISION_MODEL_TEMPERATURE',
    'PAPERSCRAPER_VISION_MODEL_TOP_P',
]


@pytest.fixture
def isolated_settings_file(tmp_path, monkeypatch):
    """Point settings reads/writes at a temporary config file and clear env overrides."""
    settings_path = tmp_path / 'pscraperrc.json'
    monkeypatch.setattr(settings, 'SETTINGS_FILE', str(settings_path))
    for key in API_ENV_KEYS + MODEL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return settings_path


def test_float_setting_uses_defaults_and_converts_values():
    """
    Test conversion of optional float settings.

    This function performs the following steps:
    1. Passes None and an empty string to `_float_setting`.
    2. Passes a numeric string to `_float_setting`.
    3. Compares each output to the expected default or converted float.

    Asserts:
        - Missing values return the supplied default.
        - Numeric strings are converted to floats.
    """
    assert settings._float_setting(None, 0.5) == 0.5
    assert settings._float_setting('', 0.5) == 0.5
    assert settings._float_setting('0.25', 0.5) == 0.25


def test_capabilities_normalizes_missing_strings_and_lists():
    """
    Test normalization of model capability settings.

    This function performs the following steps:
    1. Passes missing values to `_capabilities`.
    2. Passes a comma-separated capability string.
    3. Passes an already-normalized list.

    Asserts:
        - Missing values default to text or the supplied default.
        - Comma-separated strings become trimmed lists.
        - Lists are returned unchanged.
    """
    assert settings._capabilities(None) == ['text']
    assert settings._capabilities('', ['text', 'vision']) == ['text', 'vision']
    assert settings._capabilities('text, vision') == ['text', 'vision']
    assert settings._capabilities(['text']) == ['text']


def test_merge_profile_applies_defaults_and_coerces_types():
    """
    Test merging of stored model profile values with defaults.

    This function performs the following steps:
    1. Defines a configured profile with string capabilities and numeric strings.
    2. Merges it with the default model profile.
    3. Checks inherited and coerced fields.

    Asserts:
        - Configured values override defaults.
        - Missing values are inherited from defaults.
        - Capabilities and generation parameters are normalized.
    """
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


def test_merge_profile_defaults_missing_input_token_limit():
    """
    Test defaulting of missing model input token limits.

    This function performs the following steps:
    1. Defines a configured profile with an empty input token limit.
    2. Merges it with the default model profile.
    3. Reads the normalized input token limit from the merged profile.

    Asserts:
        - Empty input token limits are replaced with the package default.
    """
    merged = settings._merge_profile(
        settings.DEFAULT_MODEL_PROFILE,
        {'input_token_limit': ''},
    )

    assert merged['input_token_limit'] == settings.DEFAULT_INPUT_TOKEN_LIMIT


def test_env_profile_collects_only_defined_values(monkeypatch):
    """
    Test model profile environment variable collection.

    This function performs the following steps:
    1. Sets selected environment variables for one model profile prefix.
    2. Leaves unrelated profile variables unset.
    3. Calls `_env_profile` with the selected prefix.

    Asserts:
        - Defined environment values are returned.
        - Undefined values are omitted from the result.
    """
    monkeypatch.setenv('PAPERSCRAPER_MODEL_PROVIDER', 'local')
    monkeypatch.setenv('PAPERSCRAPER_MODEL_NAME', 'qwen')
    monkeypatch.setenv('PAPERSCRAPER_MODEL_INPUT_TOKEN_LIMIT', '120000')

    profile = settings._env_profile('PAPERSCRAPER_MODEL_')

    assert profile == {'provider': 'local', 'model': 'qwen', 'input_token_limit': '120000'}


def test_load_settings_returns_defaults_when_config_file_is_missing(isolated_settings_file):
    """
    Test loading settings with no user config file.

    This function performs the following steps:
    1. Points `SETTINGS_FILE` at a path that does not exist.
    2. Calls `load_settings`.
    3. Checks the default text and vision model profiles.

    Asserts:
        - Default model profiles are present.
        - Text defaults to text capability only.
        - Vision defaults to text and vision capabilities.
    """
    loaded = settings.load_settings()

    assert loaded['model_profiles']['text']['capabilities'] == ['text']
    assert loaded['model_profiles']['vision']['capabilities'] == ['text', 'vision']


def test_load_settings_merges_file_and_environment_overrides(isolated_settings_file, monkeypatch):
    """
    Test settings precedence between config file values and environment variables.

    This function performs the following steps:
    1. Writes API keys and a text model profile to the temporary config file.
    2. Sets environment overrides for API keys and the text model profile.
    3. Calls `load_settings`.

    Asserts:
        - Environment API keys override file API keys.
        - Environment model settings override stored model profile values.
        - Non-overridden profile fields are preserved or defaulted.
    """
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
    monkeypatch.setenv('PAPERSCRAPER_MODEL_PROVIDER', 'local')
    monkeypatch.setenv('PAPERSCRAPER_MODEL_NAME', 'env-model')
    monkeypatch.setenv('PAPERSCRAPER_MODEL_BASE_URL', 'http://127.0.0.1:8000/v1')
    monkeypatch.setenv('PAPERSCRAPER_MODEL_CAPABILITIES', 'text,vision')

    loaded = settings.load_settings()

    assert loaded['elsevier_api_key'] == 'env-elsevier'
    assert loaded['model_profiles']['text']['provider'] == 'local'
    assert loaded['model_profiles']['text']['model'] == 'env-model'
    assert loaded['model_profiles']['text']['base_url'] == 'http://127.0.0.1:8000/v1'
    assert loaded['model_profiles']['text']['capabilities'] == ['text', 'vision']
    assert loaded['model_profiles']['text']['temperature'] == 0.1


def test_load_settings_reports_invalid_config_file(isolated_settings_file):
    """
    Test error handling for an unreadable settings file.

    This function performs the following steps:
    1. Writes invalid JSON to the temporary settings file.
    2. Calls `load_settings`.
    3. Captures the expected runtime error.

    Asserts:
        - Invalid JSON is reported as a `RuntimeError`.
        - The error message includes the settings file path.
    """
    isolated_settings_file.write_text('{not-json')

    with pytest.raises(RuntimeError, match=str(isolated_settings_file)):
        settings.load_settings()


def test_load_settings_applies_all_api_environment_overrides(isolated_settings_file, monkeypatch):
    """
    Test environment overrides for all stored service credentials.

    This function performs the following steps:
    1. Writes file-based API settings to the temporary settings file.
    2. Sets environment variables for each supported API setting.
    3. Calls `load_settings`.

    Asserts:
        - Elsevier, OpenAI, Anthropic, CORE, and Unpaywall environment values override file values.
    """
    isolated_settings_file.write_text(json.dumps({
        'elsevier_api_key': 'file-elsevier',
        'openai_api_key': 'file-openai',
        'anthropic_api_key': 'file-anthropic',
        'core_api_key': 'file-core',
        'unpaywall_email': 'file@example.com',
    }))
    monkeypatch.setenv('ELSEVIER_API_KEY', 'env-elsevier')
    monkeypatch.setenv('OPENAI_API_KEY', 'env-openai')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'env-anthropic')
    monkeypatch.setenv('CORE_API_KEY', 'env-core')
    monkeypatch.setenv('UNPAYWALL_EMAIL', 'env@example.com')

    loaded = settings.load_settings()

    assert loaded['elsevier_api_key'] == 'env-elsevier'
    assert loaded['openai_api_key'] == 'env-openai'
    assert loaded['anthropic_api_key'] == 'env-anthropic'
    assert loaded['core_api_key'] == 'env-core'
    assert loaded['unpaywall_email'] == 'env@example.com'


def test_load_settings_applies_vision_model_environment_overrides(isolated_settings_file, monkeypatch):
    """
    Test environment overrides for the vision model profile.

    This function performs the following steps:
    1. Sets environment variables for the vision model profile.
    2. Calls `load_settings`.
    3. Reads the merged vision profile.

    Asserts:
        - Vision model provider, model, base URL, API key, capabilities, temperature, and top-p are overridden.
    """
    monkeypatch.setenv('PAPERSCRAPER_VISION_MODEL_PROVIDER', 'local')
    monkeypatch.setenv('PAPERSCRAPER_VISION_MODEL_NAME', 'qwen-vl')
    monkeypatch.setenv('PAPERSCRAPER_VISION_MODEL_BASE_URL', 'http://127.0.0.1:8000/v1')
    monkeypatch.setenv('PAPERSCRAPER_VISION_MODEL_API_KEY', 'local-key')
    monkeypatch.setenv('PAPERSCRAPER_VISION_MODEL_CAPABILITIES', 'text,vision')
    monkeypatch.setenv('PAPERSCRAPER_VISION_MODEL_TEMPERATURE', '0.4')
    monkeypatch.setenv('PAPERSCRAPER_VISION_MODEL_TOP_P', '0.7')
    monkeypatch.setenv('PAPERSCRAPER_VISION_MODEL_INPUT_TOKEN_LIMIT', '96000')

    profile = settings.load_settings()['model_profiles']['vision']

    assert profile['provider'] == 'local'
    assert profile['model'] == 'qwen-vl'
    assert profile['base_url'] == 'http://127.0.0.1:8000/v1'
    assert profile['api_key'] == 'local-key'
    assert profile['capabilities'] == ['text', 'vision']
    assert profile['temperature'] == 0.4
    assert profile['top_p'] == 0.7
    assert profile['input_token_limit'] == 96000


def test_save_settings_writes_json_to_config_file(isolated_settings_file):
    """
    Test writing settings to disk.

    This function performs the following steps:
    1. Saves a settings dictionary with `save_settings`.
    2. Reads the temporary config file as JSON.
    3. Compares the saved content to the original settings.

    Asserts:
        - The settings file is created.
        - Saved JSON preserves the supplied key/value pairs.
    """
    settings.save_settings({'core_api_key': 'core-key'})

    assert isolated_settings_file.is_file()
    assert json.loads(isolated_settings_file.read_text()) == {'core_api_key': 'core-key'}


def test_get_model_profile_returns_profile_and_rejects_missing_profile(isolated_settings_file):
    """
    Test access to named model profiles.

    This function performs the following steps:
    1. Loads the default text model profile.
    2. Requests a non-existent model profile.
    3. Captures the expected exception.

    Asserts:
        - The text profile is returned.
        - Missing profile names raise `KeyError`.
    """
    assert settings.get_model_profile('text')['provider'] == 'openai'

    with pytest.raises(KeyError):
        settings.get_model_profile('missing')


def test_infer_model_capabilities_detects_vision_models():
    """
    Test model capability inference from profile and model name.

    This function performs the following steps:
    1. Infers capabilities for a normal text profile.
    2. Infers capabilities for a vision profile.
    3. Infers capabilities from a model name containing a vision marker.

    Asserts:
        - Text profiles default to text-only capability.
        - Vision profiles include text and vision capabilities.
        - Vision-like model names include text and vision capabilities.
    """
    assert settings.infer_model_capabilities('text', 'gpt-4') == ['text']
    assert settings.infer_model_capabilities('vision', 'gpt-4') == ['text', 'vision']
    assert settings.infer_model_capabilities('text', 'Qwen/Qwen3-VL-30B') == ['text', 'vision']


def test_set_model_profile_persists_profile_values(isolated_settings_file):
    """
    Test storing a model profile.

    This function performs the following steps:
    1. Calls `set_model_profile` with a local model configuration.
    2. Reloads settings from the temporary config file.
    3. Checks that the stored profile values were persisted.

    Asserts:
        - Provider, model, base URL, API key, capabilities, and generation settings are saved.
    """
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


def test_update_anthropic_key_prompts_and_saves_key(isolated_settings_file, monkeypatch, capsys):
    """
    Test interactive Anthropic API key storage.

    This function performs the following steps:
    1. Replaces `input` with a fake Anthropic key response.
    2. Calls `update_anthropic_key`.
    3. Reloads settings from the temporary config file.

    Asserts:
        - The entered Anthropic API key is saved.
    """
    settings.save_settings({'anthropic_api_key': 'old-anthropic-key'})
    monkeypatch.setattr('builtins.input', lambda _: 'anthropic-key')

    settings.update_anthropic_key()

    output = capsys.readouterr().out
    assert 'Current Anthropic API key: old-...-key' in output
    assert settings.load_settings()['anthropic_api_key'] == 'anthropic-key'


def test_check_openai_api_key_returns_true_for_valid_key(monkeypatch):
    """
    Test successful OpenAI API key validation without calling the live API.

    This function performs the following steps:
    1. Replaces the OpenAI client with a local fake client.
    2. Calls `check_openai_api_key` with a placeholder key.
    3. Checks the validation result.

    Asserts:
        - A client whose model listing succeeds returns `True`.
    """

    class FakeModels:
        def list(self):
            return []

    class FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr(settings.openai, 'OpenAI', FakeOpenAI)

    assert settings.check_openai_api_key('placeholder-openai-key') is True


def test_check_openai_api_key_returns_false_for_invalid_key(monkeypatch):
    """
    Test failed OpenAI API key validation without calling the live API.

    This function performs the following steps:
    1. Replaces the OpenAI authentication error with a local exception.
    2. Replaces the OpenAI client with a local fake client that raises that exception.
    3. Calls `check_openai_api_key` with a placeholder key.

    Asserts:
        - A client whose model listing raises an authentication error returns `False`.
    """

    class FakeAuthenticationError(Exception):
        pass

    class FakeModels:
        def list(self):
            raise FakeAuthenticationError()

    class FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr(settings.openai, 'AuthenticationError', FakeAuthenticationError)
    monkeypatch.setattr(settings.openai, 'OpenAI', FakeOpenAI)

    assert settings.check_openai_api_key('placeholder-openai-key') is False


def test_update_openai_key_prompts_validates_and_saves_key(isolated_settings_file, monkeypatch, capsys):
    """
    Test interactive OpenAI API key storage after successful validation.

    This function performs the following steps:
    1. Replaces `input` with a fake OpenAI key response.
    2. Replaces OpenAI API key validation with a successful local result.
    3. Calls `update_openai_key` and reloads the temporary settings file.

    Asserts:
        - The entered OpenAI API key is saved after validation succeeds.
    """
    settings.save_settings({'openai_api_key': 'old-openai-key'})
    monkeypatch.setattr('builtins.input', lambda _: 'openai-key')
    monkeypatch.setattr(settings, 'check_openai_api_key', lambda api_key: api_key == 'openai-key')

    settings.update_openai_key()

    output = capsys.readouterr().out
    assert 'Current OpenAI API key: old-...-key' in output
    assert settings.load_settings()['openai_api_key'] == 'openai-key'


def test_update_openai_key_rejects_invalid_key(isolated_settings_file, monkeypatch):
    """
    Test interactive OpenAI API key storage after failed validation.

    This function performs the following steps:
    1. Replaces `input` with a fake OpenAI key response.
    2. Replaces OpenAI API key validation with a failed local result.
    3. Calls `update_openai_key` and captures the expected error.

    Asserts:
        - A failed OpenAI API key validation raises `ValueError`.
        - The invalid OpenAI API key is not saved.
    """
    monkeypatch.setattr('builtins.input', lambda _: 'bad-openai-key')
    monkeypatch.setattr(settings, 'check_openai_api_key', lambda _: False)

    with pytest.raises(ValueError, match='OpenAI API key is invalid'):
        settings.update_openai_key()

    assert 'openai_api_key' not in settings.load_settings()


def test_update_core_key_prompts_and_saves_key(isolated_settings_file, monkeypatch, capsys):
    """
    Test interactive CORE API key storage.

    This function performs the following steps:
    1. Replaces `input` with a fake CORE key response.
    2. Calls `update_core_key`.
    3. Reloads settings from the temporary config file.

    Asserts:
        - The entered CORE API key is saved.
    """
    settings.save_settings({'core_api_key': 'old-core-key'})
    monkeypatch.setattr('builtins.input', lambda _: 'core-key')

    settings.update_core_key()

    output = capsys.readouterr().out
    assert 'Current CORE API key: old-...-key' in output
    assert settings.load_settings()['core_api_key'] == 'core-key'


def test_check_elsevier_api_key_returns_true_for_valid_key(monkeypatch):
    """
    Test successful Elsevier API key validation without calling the live API.

    This function performs the following steps:
    1. Replaces the Elsevier JSON request helper with a local fake.
    2. Calls `check_elsevier_api_key` with a placeholder key.
    3. Checks the validation result.

    Asserts:
        - A client whose request succeeds returns `True`.
    """
    calls = {}

    def fake_get_json(api_key, url):
        calls['api_key'] = api_key
        calls['url'] = url
        return {'ok': True}

    monkeypatch.setattr(settings.elsevier, 'get_json', fake_get_json)

    assert settings.check_elsevier_api_key('placeholder-elsevier-key') is True
    assert calls['api_key'] == 'placeholder-elsevier-key'
    assert 'content/search/scopus' in calls['url']


def test_check_elsevier_api_key_returns_false_for_invalid_key(monkeypatch):
    """
    Test failed Elsevier API key validation without calling the live API.

    This function performs the following steps:
    1. Replaces the Elsevier JSON request helper with a local fake that raises an HTTP error.
    2. Calls `check_elsevier_api_key` with a placeholder key.
    3. Checks the validation result.

    Asserts:
        - A client whose request raises an HTTP error returns `False`.
    """
    monkeypatch.setattr(
        settings.elsevier,
        'get_json',
        lambda *_, **__: (_ for _ in ()).throw(settings.requests.HTTPError('invalid key')),
    )

    assert settings.check_elsevier_api_key('placeholder-elsevier-key') is False


def test_update_elsevier_key_prompts_validates_and_saves_key(isolated_settings_file, monkeypatch, capsys):
    """
    Test interactive Elsevier API key storage after successful validation.

    This function performs the following steps:
    1. Replaces `input` with a fake Elsevier key response.
    2. Replaces Elsevier API key validation with a successful local result.
    3. Calls `update_elsevier_key` and reloads the temporary settings file.

    Asserts:
        - The entered Elsevier API key is saved after validation succeeds.
    """
    settings.save_settings({'elsevier_api_key': 'old-elsevier-key'})
    monkeypatch.setattr('builtins.input', lambda _: 'elsevier-key')
    monkeypatch.setattr(settings, 'check_elsevier_api_key', lambda api_key: api_key == 'elsevier-key')

    settings.update_elsevier_key()

    output = capsys.readouterr().out
    assert 'Current Elsevier API key: old-...-key' in output
    assert settings.load_settings()['elsevier_api_key'] == 'elsevier-key'


def test_update_elsevier_key_rejects_invalid_key(isolated_settings_file, monkeypatch):
    """
    Test interactive Elsevier API key storage after failed validation.

    This function performs the following steps:
    1. Replaces `input` with a fake Elsevier key response.
    2. Replaces Elsevier API key validation with a failed local result.
    3. Calls `update_elsevier_key` and captures the expected error.

    Asserts:
        - A failed Elsevier API key validation raises `ValueError`.
        - The invalid Elsevier API key is not saved.
    """
    monkeypatch.setattr('builtins.input', lambda _: 'bad-elsevier-key')
    monkeypatch.setattr(settings, 'check_elsevier_api_key', lambda _: False)

    with pytest.raises(ValueError, match='Elsevier API key is invalid'):
        settings.update_elsevier_key()

    assert 'elsevier_api_key' not in settings.load_settings()


def test_update_unpaywall_email_validates_and_saves_email(isolated_settings_file, monkeypatch, capsys):
    """
    Test interactive Unpaywall email validation and storage.

    This function performs the following steps:
    1. Enters a valid email and saves it with `update_unpaywall_email`.
    2. Enters an invalid email in a second call.
    3. Captures the expected validation error.

    Asserts:
        - A valid email address is saved.
        - An invalid email address raises `ValueError`.
    """
    settings.save_settings({'unpaywall_email': 'old@example.com'})
    monkeypatch.setattr('builtins.input', lambda _: 'person@example.com')
    settings.update_unpaywall_email()
    output = capsys.readouterr().out
    assert 'Current Unpaywall email: old@example.com' in output
    assert settings.load_settings()['unpaywall_email'] == 'person@example.com'

    monkeypatch.setattr('builtins.input', lambda _: 'not-an-email')
    with pytest.raises(ValueError):
        settings.update_unpaywall_email()


@pytest.mark.network
def test_check_openai_api_key_validates_configured_key():
    """
    Test live OpenAI API key validation.

    This function performs the following steps:
    1. Loads the user's real PaperScraper settings.
    2. Reads the configured OpenAI API key.
    3. Validates the key against the OpenAI models API.

    Asserts:
        - An OpenAI API key is configured.
        - The configured OpenAI API key authenticates successfully.
    """
    loaded = settings.load_settings()
    api_key = loaded.get('openai_api_key')

    assert api_key, 'Set openai_api_key in ~/.config/.pscraperrc.json or OPENAI_API_KEY before running network tests.'
    assert settings.check_openai_api_key(api_key) is True


@pytest.mark.network
def test_check_elsevier_api_key_validates_configured_key():
    """
    Test live Elsevier API key validation.

    This function performs the following steps:
    1. Loads the user's real PaperScraper settings.
    2. Reads the configured Elsevier API key.
    3. Validates the key against a minimal Elsevier Scopus request.

    Asserts:
        - An Elsevier API key is configured.
        - The configured Elsevier API key authenticates successfully.
    """
    loaded = settings.load_settings()
    api_key = loaded.get('elsevier_api_key')

    assert api_key, 'Set elsevier_api_key in ~/.config/.pscraperrc.json or ELSEVIER_API_KEY before running network tests.'
    assert settings.check_elsevier_api_key(api_key) is True
