import os
import types

import pytest

import paperscraper.settings as settings
import paperscraper.tokenizer as tokenizer


def config(provider='openai', name='test-model', api_key='test-key', base_url=None, input_token_limit=32000):
    """Return a minimal model config for tokenizer tests."""
    return types.SimpleNamespace(
        provider=provider,
        name=name,
        api_key=api_key,
        base_url=base_url,
        input_token_limit=input_token_limit,
    )


def live_anthropic_config():
    """Build an Anthropic model config from the user's real PaperScraper settings."""
    loaded = settings.load_settings()
    profiles = loaded.get('model_profiles', {})
    anthropic_profile = next(
        (profile for profile in profiles.values() if str(profile.get('provider', '')).lower() == 'anthropic'),
        {},
    )
    api_key = anthropic_profile.get('api_key') or loaded.get('anthropic_api_key')
    model = os.environ.get('PAPERSCRAPER_ANTHROPIC_TEST_MODEL') or anthropic_profile.get('model')
    base_url = anthropic_profile.get('base_url')
    return config(provider='anthropic', name=model, api_key=api_key, base_url=base_url)


def test_conservative_token_estimate_handles_missing_and_text_values():
    """
    Test conservative token estimates.

    This function performs the following steps:
    1. Counts tokens for missing and empty values.
    2. Counts tokens for a short text value.
    3. Compares the result to the conservative character heuristic.

    Asserts:
        - Missing and empty values return zero tokens.
        - Text values are estimated with roughly one token per three characters.
    """
    assert tokenizer.conservative_token_estimate(None) == 0
    assert tokenizer.conservative_token_estimate('') == 0
    assert tokenizer.conservative_token_estimate('abcdefghij') == 4


def test_openai_token_count_selects_model_encoding_and_falls_back(monkeypatch):
    """
    Test OpenAI token counting with automatic tokenizer selection and fallbacks.

    This function performs the following steps:
    1. Replaces `tiktoken.encoding_for_model` with a fake model-specific encoding.
    2. Counts tokens for an OpenAI model.
    3. Replaces model-specific lookup with a failure and provides a fallback encoding.

    Asserts:
        - OpenAI token counting asks tiktoken for the configured model encoding.
        - The fallback encoding is used when the model-specific lookup fails.
    """
    calls = {}

    class FakeEncoding:
        def __init__(self, separator=' '):
            self.separator = separator

        def encode(self, text):
            return text.split(self.separator)

    def fake_encoding_for_model(model):
        calls['model'] = model
        return FakeEncoding()

    monkeypatch.setattr(tokenizer.tiktoken, 'encoding_for_model', fake_encoding_for_model)
    assert tokenizer.openai_token_count('one two three', 'gpt-test') == 3
    assert calls['model'] == 'gpt-test'

    monkeypatch.setattr(
        tokenizer.tiktoken,
        'encoding_for_model',
        lambda _: (_ for _ in ()).throw(KeyError('unknown model')),
    )
    fallback_calls = {}

    def fake_get_encoding(name):
        fallback_calls['name'] = name
        return FakeEncoding(separator='|')

    monkeypatch.setattr(tokenizer.tiktoken, 'get_encoding', fake_get_encoding)
    assert tokenizer.openai_token_count('one|two', 'unknown-model') == 2
    assert fallback_calls['name'] == 'o200k_base'


def test_anthropic_token_count_uses_count_tokens_endpoint(monkeypatch):
    """
    Test Anthropic token counting through the Messages count_tokens endpoint.

    This function performs the following steps:
    1. Replaces HTTP POST with a fake response that records request details.
    2. Counts tokens for an Anthropic model config.
    3. Reads the recorded request payload and headers.

    Asserts:
        - The Anthropic count_tokens endpoint is called.
        - Model, messages, API key, version, and base URL are passed through.
        - The returned `input_tokens` value is used.
    """
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            calls['raised'] = True

        def json(self):
            return {'input_tokens': 12}

    def fake_post(url, headers, json, timeout):
        calls['url'] = url
        calls['headers'] = headers
        calls['json'] = json
        calls['timeout'] = timeout
        return FakeResponse()

    monkeypatch.setattr(tokenizer.requests, 'post', fake_post)

    count = tokenizer.anthropic_token_count(
        'paper text',
        config(provider='anthropic', name='claude-test', api_key='anthropic-key', base_url='https://anthropic.local'),
    )

    assert count == 12
    assert calls['url'] == 'https://anthropic.local/v1/messages/count_tokens'
    assert calls['headers']['x-api-key'] == 'anthropic-key'
    assert calls['headers']['anthropic-version'] == tokenizer.ANTHROPIC_VERSION
    assert calls['json'] == {
        'model': 'claude-test',
        'messages': [{'role': 'user', 'content': 'paper text'}],
    }
    assert calls['timeout'] == 60
    assert calls['raised'] is True


def test_anthropic_token_count_requires_api_key():
    """
    Test Anthropic token counting without an API key.

    This function performs the following steps:
    1. Builds an Anthropic model config without an API key.
    2. Calls `anthropic_token_count`.
    3. Captures the expected exception.

    Asserts:
        - Missing Anthropic API keys raise `ValueError`.
    """
    with pytest.raises(ValueError, match='requires an API key'):
        tokenizer.anthropic_token_count('paper text', config(provider='anthropic', api_key=None))


@pytest.mark.network
def test_anthropic_token_count_uses_real_count_tokens_api():
    """
    Test live Anthropic token counting with the user's configured credentials.

    This function performs the following steps:
    1. Loads the user's real PaperScraper settings.
    2. Builds an Anthropic model config from an Anthropic profile or test-model override.
    3. Calls Anthropic's live Messages count_tokens endpoint.

    Asserts:
        - An Anthropic API key is configured.
        - An Anthropic model name is configured.
        - The live count_tokens endpoint returns a positive integer token count.
    """
    model_config = live_anthropic_config()

    assert model_config.api_key, (
        'Set anthropic_api_key in ~/.config/.pscraperrc.json or ANTHROPIC_API_KEY before running network tests.'
    )
    assert model_config.name, (
        'Configure an Anthropic model profile or set PAPERSCRAPER_ANTHROPIC_TEST_MODEL before running network tests.'
    )
    count = tokenizer.anthropic_token_count('Count these paper-scraping tokens.', model_config)

    assert isinstance(count, int)
    assert count > 0


def test_transformers_token_count_uses_auto_tokenizer(monkeypatch):
    """
    Test local-model token counting with a Hugging Face tokenizer.

    This function performs the following steps:
    1. Clears the tokenizer cache.
    2. Replaces dynamic imports with a fake `transformers` module.
    3. Counts tokens for a local model.

    Asserts:
        - `AutoTokenizer.from_pretrained` is called for the configured model.
        - The tokenizer's encoded token count is returned.
    """
    calls = {}
    tokenizer._auto_tokenizer.cache_clear()

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            calls['text'] = text
            calls['add_special_tokens'] = add_special_tokens
            return [1, 2, 3]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model):
            calls['model'] = model
            return FakeTokenizer()

    fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    monkeypatch.setattr(tokenizer.importlib, 'import_module', lambda name: fake_transformers)

    assert tokenizer.transformers_token_count('local text', 'Qwen/Qwen3') == 3
    assert calls == {
        'model': 'Qwen/Qwen3',
        'text': 'local text',
        'add_special_tokens': False,
    }


def test_count_text_tokens_routes_by_provider_and_falls_back(monkeypatch):
    """
    Test provider-aware token count routing and fallback behavior.

    This function performs the following steps:
    1. Replaces each provider-specific tokenizer with local fakes.
    2. Counts tokens for OpenAI, Anthropic, local, unknown, and non-string inputs.
    3. Makes provider-specific tokenizers fail to exercise fallback estimates.

    Asserts:
        - Provider-specific token counters are selected from the model config.
        - Unknown providers use the conservative estimate.
        - Failed provider tokenizers fall back to the conservative estimate.
        - Non-string inputs return zero tokens.
    """
    monkeypatch.setattr(tokenizer, 'openai_token_count', lambda text, model: 5)
    monkeypatch.setattr(tokenizer, 'anthropic_token_count', lambda text, model_config: 6)
    monkeypatch.setattr(tokenizer, 'transformers_token_count', lambda text, model: 7)

    assert tokenizer.count_text_tokens('text', config(provider='openai')) == 5
    assert tokenizer.count_text_tokens('text', config(provider='anthropic')) == 6
    assert tokenizer.count_text_tokens('text', config(provider='local')) == 7
    assert tokenizer.count_text_tokens('abcdef', config(provider='other')) == 2
    assert tokenizer.count_text_tokens(None, config(provider='openai')) == 0

    monkeypatch.setattr(
        tokenizer,
        'transformers_token_count',
        lambda *_: (_ for _ in ()).throw(ImportError('missing transformers')),
    )
    assert tokenizer.count_text_tokens('abcdefghij', config(provider='local')) == 4

    monkeypatch.setattr(
        tokenizer,
        'openai_token_count',
        lambda *_: (_ for _ in ()).throw(KeyError('unknown model')),
    )
    assert tokenizer.count_text_tokens('abcdefghij', config(provider='openai')) == 4

    monkeypatch.setattr(
        tokenizer,
        'anthropic_token_count',
        lambda *_: (_ for _ in ()).throw(ValueError('missing key')),
    )
    assert tokenizer.count_text_tokens('abcdefghij', config(provider='anthropic')) == 4


def test_usable_input_token_limit_uses_configured_limit_with_reserve_and_fallback():
    """
    Test usable input token budget calculation.

    This function performs the following steps:
    1. Builds model configs with valid and invalid input token limits.
    2. Calculates usable budgets with reserved prompt space.
    3. Calculates a budget whose reserve would otherwise make it too small.

    Asserts:
        - Configured input token limits are reduced by reserved tokens.
        - Invalid configured values fall back to the default.
        - The returned value does not go below the minimum budget.
    """
    assert tokenizer.usable_input_token_limit(config(input_token_limit=64000), reserve_tokens=2000) == 62000
    assert tokenizer.usable_input_token_limit(config(input_token_limit='bad'), reserve_tokens=2000) == (
        tokenizer.DEFAULT_INPUT_TOKEN_LIMIT - 2000
    )
    assert tokenizer.usable_input_token_limit(config(input_token_limit=1200), reserve_tokens=1000, minimum=500) == 500


def test_prompt_token_reserve_counts_prompt_and_adds_buffer(monkeypatch):
    """
    Test prompt reserve calculation.

    This function performs the following steps:
    1. Replaces text token counting with a deterministic fake.
    2. Calculates reserve tokens for a prompt and model config.
    3. Calculates reserve tokens for a tiny prompt with a larger minimum.

    Asserts:
        - Prompt token count and buffer tokens are added together.
        - The minimum reserve is respected.
        - The model config is passed to token counting.
    """
    calls = []

    def fake_count(text, model_config=None):
        calls.append((text, model_config))
        return len(text.split())

    model_config = config(provider='openai')
    monkeypatch.setattr(tokenizer, 'count_text_tokens', fake_count)

    assert tokenizer.prompt_token_reserve('one two three', model_config=model_config, buffer_tokens=7, minimum=0) == 10
    assert tokenizer.prompt_token_reserve('one', model_config=model_config, buffer_tokens=0, minimum=5) == 5
    assert calls == [('one two three', model_config), ('one', model_config)]
