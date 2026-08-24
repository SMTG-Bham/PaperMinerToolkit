"""Provider-aware token counting for model context management.

This module chooses the best available token counter for the configured model:
OpenAI models use ``tiktoken``, Anthropic models can use Claude's token counting
endpoint, local models use Hugging Face tokenizers, and all
paths fall back to a conservative character-based estimate.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from functools import lru_cache
from typing import Protocol, TypeAlias

import requests
import tiktoken

from paperminer.settings import DEFAULT_MODEL
from paperminer.settings import DEFAULT_INPUT_TOKEN_LIMIT

ANTHROPIC_VERSION = '2023-06-01'
DEFAULT_ANTHROPIC_BASE_URL = 'https://api.anthropic.com'


class _ModelConfigLike(Protocol):
    """Structural type required by provider-aware token helpers."""

    provider: str
    name: str
    api_key: str | None
    base_url: str | None
    input_token_limit: int


_ModelConfigSource: TypeAlias = _ModelConfigLike | Mapping[str, object]


class _TokenizerLike(Protocol):
    """Structural type returned by dynamically loaded tokenizers."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text into integer token identifiers."""
        ...


def conservative_token_estimate(text: str) -> int:
    """Estimate tokens when no model tokenizer is available.

    Parameters
    ----------
    text : str
        Text to estimate.

    Returns
    -------
    int
        Zero for missing or empty input; otherwise one token per three
        characters, rounded up.
    """
    if not isinstance(text, str) or text == '':
        return 0
    return max(1, math.ceil(len(text) / 3))


def _provider_name(model_config: _ModelConfigSource | None = None, provider: str | None = None) -> str:
    """Resolve and normalize a provider name.

    Parameters
    ----------
    model_config : _ModelConfigSource or None, optional
        Object that may expose a ``provider`` attribute.
    provider : str, optional
        Explicit provider name, which takes precedence over ``model_config``.

    Returns
    -------
    str
        Lowercase provider name with underscores replaced by hyphens.
    """
    value = provider or getattr(model_config, 'provider', '') or ''
    return str(value).lower().replace('_', '-')


def _model_name(model_config: _ModelConfigSource | None = None, model: str | None = None) -> str:
    """Resolve a model name from explicit input or configuration.

    Parameters
    ----------
    model_config : _ModelConfigSource or None, optional
        Object that may expose a ``name`` attribute.
    model : str, optional
        Explicit model name, which takes precedence over ``model_config``.

    Returns
    -------
    str
        Resolved model name, falling back to the package default.
    """
    return model or getattr(model_config, 'name', None) or DEFAULT_MODEL


def openai_token_count(text: str, model: str | None = None) -> int:
    """Count text tokens with an OpenAI tokenizer.

    Parameters
    ----------
    text : str
        Text to tokenize.
    model : str, optional
        OpenAI model used to select an encoding. Unknown models use
        ``o200k_base``.

    Returns
    -------
    int
        Number of encoded tokens.
    """
    model = model or DEFAULT_MODEL
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding('o200k_base')
    return len(encoding.encode(text))


def anthropic_token_count(text: str, model_config: _ModelConfigSource) -> int:
    """Count text tokens with Anthropic's Messages API.

    Parameters
    ----------
    text : str
        Text to count as a single user message.
    model_config : _ModelConfigSource
        Configuration exposing model, API key, and optional base URL
        attributes.

    Returns
    -------
    int
        Input token count reported by Anthropic.

    Raises
    ------
    ValueError
        If the configuration has no Anthropic API key.
    requests.RequestException
        If the token-counting request fails.
    """
    api_key = getattr(model_config, 'api_key', None)
    if not api_key:
        raise ValueError('Anthropic token counting requires an API key.')
    model = _model_name(model_config)
    base_url = getattr(model_config, 'base_url', None) or DEFAULT_ANTHROPIC_BASE_URL
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': text}],
    }
    headers = {
        'x-api-key': api_key,
        'anthropic-version': ANTHROPIC_VERSION,
        'content-type': 'application/json',
    }
    response = requests.post(
        f'{base_url.rstrip("/")}/v1/messages/count_tokens',
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return int(response.json()['input_tokens'])


@lru_cache(maxsize=16)
def _auto_tokenizer(model: str) -> _TokenizerLike:
    """Load and cache a Hugging Face tokenizer.

    Parameters
    ----------
    model : str
        Hugging Face model identifier or local model path.

    Returns
    -------
    _TokenizerLike
        Loaded tokenizer instance.
    """
    transformers = importlib.import_module('transformers')
    return transformers.AutoTokenizer.from_pretrained(model)


def transformers_token_count(text: str, model: str) -> int:
    """Count text tokens with a Hugging Face tokenizer.

    Parameters
    ----------
    text : str
        Text to tokenize.
    model : str
        Hugging Face model identifier or local model path.

    Returns
    -------
    int
        Number of encoded tokens, excluding special tokens.
    """
    tokenizer = _auto_tokenizer(model)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def count_text_tokens(
    text: str,
    model_config: _ModelConfigSource | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> int:
    """Count text tokens with the best available provider tokenizer.

    Parameters
    ----------
    text : str
        Text to tokenize. Non-string values return zero.
    model_config : _ModelConfigSource or None, optional
        Model configuration used to resolve provider credentials and model
        details.
    model : str, optional
        Explicit model name override.
    provider : str, optional
        Explicit provider override.

    Returns
    -------
    int
        Provider token count, or a conservative estimate when tokenization
        fails or the provider is unknown.
    """
    if not isinstance(text, str):
        return 0
    provider_name = _provider_name(model_config, provider)
    model_name = _model_name(model_config, model)
    if provider_name in {'', 'openai'}:
        try:
            return openai_token_count(text, model_name)
        except Exception:
            return conservative_token_estimate(text)
    if provider_name == 'anthropic':
        try:
            return anthropic_token_count(text, model_config)
        except Exception:
            return conservative_token_estimate(text)
    if provider_name == 'local':
        try:
            return transformers_token_count(text, model_name)
        except Exception:
            return conservative_token_estimate(text)
    return conservative_token_estimate(text)


def usable_input_token_limit(
    model_config: _ModelConfigSource | None = None,
    reserve_tokens: int = 2000,
    minimum: int = 1000,
) -> int:
    """Calculate the usable model input token budget.

    Parameters
    ----------
    model_config : _ModelConfigSource or None, optional
        Configuration that may expose an ``input_token_limit`` attribute.
    reserve_tokens : int, default=2000
        Tokens reserved for prompts, metadata, and output framing.
    minimum : int, default=1000
        Minimum budget returned after applying the reserve.

    Returns
    -------
    int
        Usable input token budget.
    """
    configured = getattr(model_config, 'input_token_limit', DEFAULT_INPUT_TOKEN_LIMIT)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_INPUT_TOKEN_LIMIT
    return max(minimum, configured - int(reserve_tokens))


def prompt_token_reserve(prompt: str,
                         model_config: _ModelConfigSource | None = None,
                         buffer_tokens: int = 500,
                         minimum: int = 500) -> int:
    """Calculate a token reserve for a static prompt.

    Parameters
    ----------
    prompt : str
        Static prompt text included with model input.
    model_config : _ModelConfigSource or None, optional
        Model configuration passed to provider-aware token counting.
    buffer_tokens : int, default=500
        Safety buffer added to the prompt token count.
    minimum : int, default=500
        Minimum reserve returned.

    Returns
    -------
    int
        Prompt token count plus the buffer, bounded by ``minimum``.
    """
    prompt_tokens = count_text_tokens(prompt or '', model_config=model_config)
    return max(minimum, prompt_tokens + int(buffer_tokens))
