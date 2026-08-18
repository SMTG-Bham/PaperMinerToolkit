"""Model configuration and provider clients for text and vision requests.

This module hides provider differences behind a small client interface used by
the extraction code. It supports OpenAI Responses, Anthropic Messages, and
OpenAI-compatible local chat servers.
"""

import base64
import mimetypes
import openai
import requests
from dataclasses import dataclass, field
from typing import Any

from paperscraper.compression import CompressionConfig, maybe_compress_image_messages
from paperscraper.settings import DEFAULT_INPUT_TOKEN_LIMIT, DEFAULT_MODEL, get_model_profile, load_settings

OPENAI_FIXED_SAMPLING_PREFIXES = ('gpt-5', 'o1', 'o3', 'o4')


class ModelCapabilityError(ValueError):
    """Raised when a configured model is asked to handle an unsupported input type."""
    pass


@dataclass
class ModelConfig:
    """Configuration for one text or vision model profile."""

    provider: str = 'openai'
    name: str = DEFAULT_MODEL
    base_url: str | None = None
    api_key: str | None = None
    capabilities: set[str] = field(default_factory=lambda: {'text'})
    temperature: float = 0
    top_p: float = 1
    input_token_limit: int = DEFAULT_INPUT_TOKEN_LIMIT

    @classmethod
    def from_profile(cls, profile: str = 'text', **overrides):
        """Build a model config from a named settings profile plus overrides."""
        configured = get_model_profile(profile)
        capabilities = overrides.get('capabilities', configured.get('capabilities', ['text']))
        if isinstance(capabilities, str):
            capabilities = [cap.strip() for cap in capabilities.split(',') if cap.strip()]
        settings = load_settings()
        configured_provider = configured.get('provider') or 'openai'
        provider = overrides.get('provider') or configured_provider
        provider_changed = provider.lower().replace('_', '-') != configured_provider.lower().replace('_', '-')
        name = overrides.get('name') or overrides.get('model')
        if provider_changed and not name:
            raise ValueError('Changing model provider requires an explicit model name.')
        provider_key = f'{provider.lower()}_api_key'
        profile_api_key = None if provider_changed else configured.get('api_key')
        profile_base_url = None if provider_changed else configured.get('base_url')
        return cls(
            provider=provider,
            name=name or configured.get('model') or DEFAULT_MODEL,
            base_url=overrides.get('base_url') or profile_base_url,
            api_key=overrides.get('api_key') or profile_api_key or settings.get(provider_key),
            capabilities=set(capabilities or ['text']),
            temperature=float(overrides.get('temperature', configured.get('temperature', 0))),
            top_p=float(overrides.get('top_p', configured.get('top_p', 1))),
            input_token_limit=int(
                overrides.get('input_token_limit', configured.get('input_token_limit', DEFAULT_INPUT_TOKEN_LIMIT))
            ),
        )

    def generation_args(self):
        """Return provider generation parameters shared across request types."""
        provider = self.provider.lower().replace('_', '-')
        model = self.name.lower()
        if provider == 'openai' and model.startswith(OPENAI_FIXED_SAMPLING_PREFIXES):
            return {}
        args = {'temperature': self.temperature, 'top_p': self.top_p}
        if provider == 'anthropic':
            args.pop('top_p')
        return args

    def require(self, capability: str):
        """Raise when this config lacks the requested model capability."""
        if capability not in self.capabilities:
            caps = ', '.join(sorted(self.capabilities)) or 'none'
            raise ModelCapabilityError(
                f'Model "{self.name}" is configured for [{caps}] and cannot handle {capability} inputs.'
            )


def image_to_data_url(path: str):
    """Encode a local image as a data URL suitable for model API payloads."""
    mime_type = mimetypes.guess_type(path)[0] or 'image/png'
    with open(path, 'rb') as image_file:
        encoded = base64.b64encode(image_file.read()).decode('ascii')
    return f'data:{mime_type};base64,{encoded}'


class BaseModelClient:
    """Abstract interface implemented by concrete model provider clients."""

    def __init__(self, config: ModelConfig):
        """Store the model configuration used for provider requests."""
        self.config = config

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        """Send a text-only request and return the model text response."""
        raise NotImplementedError

    def query_with_images(self,
                          prompt: str,
                          image_paths: list[str],
                          context: str | None = None,
                          max_output_tokens: int = 10000,
                          compression_config: CompressionConfig | None = None) -> str:
        """Send a vision request with local images and return the text response."""
        raise NotImplementedError


class OpenAIResponsesClient(BaseModelClient):
    """Client for OpenAI's Responses API."""

    def __init__(self, config: ModelConfig):
        """Create an OpenAI SDK client using the supplied model config."""
        super().__init__(config)
        kwargs = {}
        if config.api_key:
            kwargs['api_key'] = config.api_key
        if config.base_url:
            kwargs['base_url'] = config.base_url
        self.client = openai.OpenAI(**kwargs)

    def _response_text(self, response):
        """Extract text from an OpenAI Responses API response object."""
        if getattr(response, 'output_text', None):
            return response.output_text
        for output in getattr(response, 'output', []):
            for content in getattr(output, 'content', []):
                text = getattr(content, 'text', None)
                if text:
                    return text
        raise RuntimeError('Model response did not contain text output.')

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        """Send a text prompt through OpenAI Responses."""
        self.config.require('text')
        try:
            response = self.client.responses.create(
                model=self.config.name,
                input=messages,
                max_output_tokens=max_output_tokens,
                **self.config.generation_args(),
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f'OpenAI request failed: {e}') from e
        return self._response_text(response)

    def query_with_images(self,
                          prompt: str,
                          image_paths: list[str],
                          context: str | None = None,
                          max_output_tokens: int = 10000,
                          compression_config: CompressionConfig | None = None) -> str:
        """Send image inputs through OpenAI Responses."""
        self.config.require('vision')
        content = [{'type': 'input_text', 'text': prompt}]
        if context:
            content.append({'type': 'input_text', 'text': context})
        for image_path in image_paths:
            content.append({'type': 'input_image', 'image_url': image_to_data_url(image_path)})
        messages = [{'role': 'user', 'content': content}]
        messages = maybe_compress_image_messages(
            messages,
            image_paths,
            prompt,
            context,
            self.config,
            compression_config,
        )
        try:
            response = self.client.responses.create(
                model=self.config.name,
                input=messages,
                max_output_tokens=max_output_tokens,
                **self.config.generation_args(),
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f'OpenAI vision request failed: {e}') from e
        return self._response_text(response)


class AnthropicMessagesClient(BaseModelClient):
    """Client for Anthropic's Messages API."""

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        """Send a text prompt through Anthropic Messages."""
        self.config.require('text')
        if not self.config.api_key:
            raise ValueError('Anthropic provider requires an API key in the model profile or settings.')
        system = ''
        anthropic_messages = []
        for message in messages:
            if message.get('role') == 'system':
                system += message.get('content', '')
            else:
                anthropic_messages.append({
                    'role': message.get('role', 'user'),
                    'content': message.get('content', ''),
                })
        return self._request(system, anthropic_messages, max_output_tokens)

    def query_with_images(self,
                          prompt: str,
                          image_paths: list[str],
                          context: str | None = None,
                          max_output_tokens: int = 10000,
                          compression_config: CompressionConfig | None = None) -> str:
        """Send image inputs through Anthropic Messages."""
        self.config.require('vision')
        content = [{'type': 'text', 'text': prompt}]
        if context:
            content.append({'type': 'text', 'text': context})
        for image_path in image_paths:
            mime_type = mimetypes.guess_type(image_path)[0] or 'image/png'
            with open(image_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('ascii')
            content.append({
                'type': 'image',
                'source': {'type': 'base64', 'media_type': mime_type, 'data': encoded},
            })
        messages = [{'role': 'user', 'content': content}]
        messages = maybe_compress_image_messages(
            messages,
            image_paths,
            prompt,
            context,
            self.config,
            compression_config,
        )
        return self._request('', messages, max_output_tokens)

    def _request(self, system, anthropic_messages, max_output_tokens):
        """Post a prepared Anthropic Messages payload and return joined text."""
        if not self.config.api_key:
            raise ValueError('Anthropic provider requires an API key in the model profile or settings.')
        payload = {
            'model': self.config.name,
            'max_tokens': max_output_tokens,
            'messages': anthropic_messages,
            **self.config.generation_args(),
        }
        if system:
            payload['system'] = system
        headers = {
            'x-api-key': self.config.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        base_url = (self.config.base_url or 'https://api.anthropic.com').rstrip('/')
        endpoint = f'{base_url}/messages' if base_url.endswith('/v1') else f'{base_url}/v1/messages'
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
        except requests.HTTPError as e:
            detail = _anthropic_error_detail(e.response)
            suffix = f': {detail}' if detail else ''
            raise RuntimeError(f'Anthropic request failed: {e}{suffix}') from e
        except requests.RequestException as e:
            raise RuntimeError(f'Anthropic request failed: {e}') from e
        data = response.json()
        chunks = [block.get('text', '') for block in data.get('content', []) if block.get('type') == 'text']
        return ''.join(chunks)


def _anthropic_error_detail(response):
    """Return a concise provider error message from an Anthropic HTTP response."""
    if response is None:
        return ''
    try:
        data = response.json()
    except (requests.JSONDecodeError, ValueError):
        data = None
    if isinstance(data, dict):
        error = data.get('error')
        if isinstance(error, dict) and error.get('message'):
            return str(error['message'])[:1000]
    return str(getattr(response, 'text', '') or '').strip()[:1000]


class OpenAICompatibleChatClient(BaseModelClient):
    """Client for local or third-party OpenAI-compatible chat servers."""

    def __init__(self, config: ModelConfig):
        """Create an OpenAI-compatible SDK client from config values."""
        super().__init__(config)
        kwargs = {'api_key': config.api_key or 'not-needed'}
        if config.base_url:
            kwargs['base_url'] = config.base_url
        self.client = openai.OpenAI(**kwargs)

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        """Send a text chat completion request."""
        self.config.require('text')
        return self._chat(messages, max_output_tokens)

    def query_with_images(self,
                          prompt: str,
                          image_paths: list[str],
                          context: str | None = None,
                          max_output_tokens: int = 10000,
                          compression_config: CompressionConfig | None = None) -> str:
        """Send image inputs as OpenAI-compatible chat content."""
        self.config.require('vision')
        content = [{'type': 'text', 'text': prompt}]
        if context:
            content.append({'type': 'text', 'text': context})
        for image_path in image_paths:
            content.append({'type': 'image_url', 'image_url': {'url': image_to_data_url(image_path)}})
        messages = [{'role': 'user', 'content': content}]
        messages = maybe_compress_image_messages(
            messages,
            image_paths,
            prompt,
            context,
            self.config,
            compression_config,
        )
        return self._chat(messages, max_output_tokens)

    def _chat(self, messages, max_output_tokens):
        """Call the chat completions endpoint and return the first message text."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                messages=messages,
                max_tokens=max_output_tokens,
                **self.config.generation_args(),
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f'Local model request failed: {e}') from e
        return response.choices[0].message.content or ''


def get_model_client(config: ModelConfig | None = None):
    """Return the concrete provider client for a model config."""
    config = config or ModelConfig.from_profile('text')
    provider = config.provider.lower().replace('_', '-')
    if provider == 'openai':
        return OpenAIResponsesClient(config)
    if provider == 'anthropic':
        return AnthropicMessagesClient(config)
    if provider == 'local':
        if not config.base_url:
            raise ValueError(f'Provider "{config.provider}" requires a base URL.')
        return OpenAICompatibleChatClient(config)
    raise ValueError(f'Unknown model provider: {config.provider}')


def query_text(messages: list[dict[str, Any]],
               config: ModelConfig | None = None,
               max_output_tokens: int = 10000) -> str:
    """Send a text request through the configured model provider."""
    return get_model_client(config).query(messages, max_output_tokens=max_output_tokens)


def query_images(prompt: str,
                 image_paths: list[str],
                 config: ModelConfig | None = None,
                 context: str | None = None,
                 max_output_tokens: int = 10000,
                 compression_config: CompressionConfig | None = None) -> str:
    """Send an image request through the configured vision model provider."""
    return get_model_client(config).query_with_images(prompt, image_paths, context=context,
                                                      max_output_tokens=max_output_tokens,
                                                      compression_config=compression_config)
