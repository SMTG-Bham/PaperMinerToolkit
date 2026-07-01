"""Provider-aware token counting for model context management.

This module chooses the best available token counter for the configured model:
OpenAI models use ``tiktoken``, Anthropic models can use Claude's token counting
endpoint, local models use Hugging Face tokenizers, and all
paths fall back to a conservative character-based estimate.
"""

import importlib
import math
from functools import lru_cache

import requests
import tiktoken

from paperscraper.settings import DEFAULT_MODEL
from paperscraper.settings import DEFAULT_INPUT_TOKEN_LIMIT

ANTHROPIC_VERSION = '2023-06-01'
DEFAULT_ANTHROPIC_BASE_URL = 'https://api.anthropic.com'


def conservative_token_estimate(text: str) -> int:
    """Estimate tokens conservatively when no model tokenizer is available."""
    if not isinstance(text, str) or text == '':
        return 0
    return max(1, math.ceil(len(text) / 3))


def _provider_name(model_config=None, provider: str | None = None) -> str:
    """Return a normalized provider name from explicit input or a model config."""
    value = provider or getattr(model_config, 'provider', '') or ''
    return str(value).lower().replace('_', '-')


def _model_name(model_config=None, model: str | None = None) -> str:
    """Return the selected model name from explicit input or a model config."""
    return model or getattr(model_config, 'name', None) or DEFAULT_MODEL


def openai_token_count(text: str, model: str | None = None) -> int:
    """Count text tokens with the OpenAI tokenizer selected for ``model``."""
    model = model or DEFAULT_MODEL
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding('o200k_base')
    return len(encoding.encode(text))


def anthropic_token_count(text: str, model_config) -> int:
    """Count text tokens with Anthropic's Messages token counting endpoint."""
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
def _auto_tokenizer(model: str):
    """Load and cache a Hugging Face tokenizer for a local model."""
    transformers = importlib.import_module('transformers')
    return transformers.AutoTokenizer.from_pretrained(model)


def transformers_token_count(text: str, model: str) -> int:
    """Count text tokens with ``transformers.AutoTokenizer`` when available."""
    tokenizer = _auto_tokenizer(model)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def count_text_tokens(text: str, model_config=None, model: str | None = None, provider: str | None = None) -> int:
    """Count text tokens using the best tokenizer available for the model."""
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


def usable_input_token_limit(model_config=None, reserve_tokens: int = 2000, minimum: int = 1000) -> int:
    """Return a model input budget after reserving room for prompts and metadata."""
    configured = getattr(model_config, 'input_token_limit', DEFAULT_INPUT_TOKEN_LIMIT)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_INPUT_TOKEN_LIMIT
    return max(minimum, configured - int(reserve_tokens))


def prompt_token_reserve(prompt: str,
                         model_config=None,
                         buffer_tokens: int = 500,
                         minimum: int = 500) -> int:
    """Estimate tokens to reserve for static prompts plus a small safety buffer."""
    prompt_tokens = count_text_tokens(prompt or '', model_config=model_config)
    return max(minimum, prompt_tokens + int(buffer_tokens))
