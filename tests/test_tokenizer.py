"""Test provider-aware token counting and budget calculations."""

from __future__ import annotations

import os
import types
from typing import Any

import pytest

import paperminer.settings as settings
import paperminer.tokenizer as tokenizer


def config(
    provider: str = 'openai',
    name: str | None = 'test-model',
    api_key: str | None = 'test-key',
    base_url: str | None = None,
    input_token_limit: int | str = 32000,
) -> types.SimpleNamespace:
    """Return a minimal model config for tokenizer tests."""
    return types.SimpleNamespace(
        provider=provider,
        name=name,
        api_key=api_key,
        base_url=base_url,
        input_token_limit=input_token_limit,
    )


def live_anthropic_config() -> types.SimpleNamespace:
    """Build an Anthropic model config from the user's real PaperMiner settings."""
    loaded = settings.load_settings()
    profiles = loaded.get('model_profiles', {})
    anthropic_profile = next(
        (profile for profile in profiles.values() if str(profile.get('provider', '')).lower() == 'anthropic'),
        {},
    )
    api_key = anthropic_profile.get('api_key') or loaded.get('anthropic_api_key')
    model = os.environ.get('PAPERMINER_ANTHROPIC_TEST_MODEL') or anthropic_profile.get('model')
    base_url = anthropic_profile.get('base_url')
    return config(provider='anthropic', name=model, api_key=api_key, base_url=base_url)


def test_conservative_token_estimate_handles_missing_and_text_values() -> None:
    """Estimate text conservatively and map missing values to zero."""
    assert tokenizer.conservative_token_estimate(None) == 0
    assert tokenizer.conservative_token_estimate('') == 0
    assert tokenizer.conservative_token_estimate('abcdefghij') == 4


def test_openai_token_count_selects_model_encoding_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the model encoding and fall back for unknown OpenAI models."""
    calls = {}

    class FakeEncoding:
        """Encode text by splitting it on a configured separator."""

        def __init__(self, separator: str = ' ') -> None:
            """Store the token separator."""
            self.separator = separator

        def encode(self, text: str) -> list[str]:
            """Split text into fake tokens."""
            return text.split(self.separator)

    def fake_encoding_for_model(model: str) -> FakeEncoding:
        """Record the model and return its fake encoding."""
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

    def fake_get_encoding(name: str) -> FakeEncoding:
        """Record and return the fallback encoding."""
        fallback_calls['name'] = name
        return FakeEncoding(separator='|')

    monkeypatch.setattr(tokenizer.tiktoken, 'get_encoding', fake_get_encoding)
    assert tokenizer.openai_token_count('one|two', 'unknown-model') == 2
    assert fallback_calls['name'] == 'o200k_base'


def test_anthropic_token_count_uses_count_tokens_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call Anthropic's Messages token-counting endpoint."""
    calls = {}

    class FakeResponse:
        """Provide a successful Anthropic token-count response."""

        def raise_for_status(self) -> None:
            """Record that the response status was checked."""
            calls['raised'] = True

        def json(self) -> dict[str, int]:
            """Return the fake input token count."""
            return {'input_tokens': 12}

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        """Record an Anthropic request and return the fake response."""
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


def test_anthropic_token_count_requires_api_key() -> None:
    """Require an API key for Anthropic token counting."""
    with pytest.raises(ValueError, match='requires an API key'):
        tokenizer.anthropic_token_count('paper text', config(provider='anthropic', api_key=None))


@pytest.mark.network
def test_anthropic_token_count_uses_real_count_tokens_api() -> None:
    """Count tokens with the live Anthropic API and configured credentials."""
    model_config = live_anthropic_config()

    assert model_config.api_key, (
        'Set anthropic_api_key in ~/.config/.pscraperrc.json or ANTHROPIC_API_KEY before running network tests.'
    )
    assert model_config.name, (
        'Configure an Anthropic model profile or set PAPERMINER_ANTHROPIC_TEST_MODEL before running network tests.'
    )
    count = tokenizer.anthropic_token_count('Count these paper-scraping tokens.', model_config)

    assert isinstance(count, int)
    assert count > 0


def test_transformers_token_count_uses_auto_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count local-model tokens with a Hugging Face tokenizer."""
    calls = {}
    tokenizer._auto_tokenizer.cache_clear()

    class FakeTokenizer:
        """Provide deterministic encoded tokens."""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            """Record encoding arguments and return three tokens."""
            calls['text'] = text
            calls['add_special_tokens'] = add_special_tokens
            return [1, 2, 3]

    class FakeAutoTokenizer:
        """Provide a fake pretrained tokenizer factory."""

        @staticmethod
        def from_pretrained(model: str) -> FakeTokenizer:
            """Record the model and return a fake tokenizer."""
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


def test_count_text_tokens_routes_by_provider_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route token counts by provider and fall back on failures."""
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


def test_usable_input_token_limit_uses_configured_limit_with_reserve_and_fallback() -> None:
    """Reserve tokens from valid, invalid, and minimal input budgets."""
    assert tokenizer.usable_input_token_limit(config(input_token_limit=64000), reserve_tokens=2000) == 62000
    assert tokenizer.usable_input_token_limit(config(input_token_limit='bad'), reserve_tokens=2000) == (
        tokenizer.DEFAULT_INPUT_TOKEN_LIMIT - 2000
    )
    assert tokenizer.usable_input_token_limit(config(input_token_limit=1200), reserve_tokens=1000, minimum=500) == 500


def test_prompt_token_reserve_counts_prompt_and_adds_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add a safety buffer to prompt tokens while respecting a minimum."""
    calls = []

    def fake_count(
        text: str,
        model_config: types.SimpleNamespace | None = None,
    ) -> int:
        """Record the model config and count whitespace-delimited words."""
        calls.append((text, model_config))
        return len(text.split())

    model_config = config(provider='openai')
    monkeypatch.setattr(tokenizer, 'count_text_tokens', fake_count)

    assert tokenizer.prompt_token_reserve('one two three', model_config=model_config, buffer_tokens=7, minimum=0) == 10
    assert tokenizer.prompt_token_reserve('one', model_config=model_config, buffer_tokens=0, minimum=5) == 5
    assert calls == [('one two three', model_config), ('one', model_config)]
