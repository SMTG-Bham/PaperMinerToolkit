from dataclasses import dataclass, field
from typing import Any

import openai
import requests

from paperscraper.settings import load_settings


class ModelCapabilityError(ValueError):
    pass


@dataclass
class ModelConfig:
    provider: str = 'openai'
    name: str = 'gpt-5-mini'
    base_url: str | None = None
    api_key: str | None = None
    capabilities: set[str] = field(default_factory=lambda: {'text'})

    @classmethod
    def from_settings(cls, **overrides):
        settings = load_settings()
        capabilities = overrides.get('capabilities', settings.get('model_capabilities', ['text']))
        if isinstance(capabilities, str):
            capabilities = [cap.strip() for cap in capabilities.split(',') if cap.strip()]
        return cls(
            provider=overrides.get('provider') or settings.get('model_provider') or 'openai',
            name=overrides.get('name') or overrides.get('model') or settings.get('model_name') or 'gpt-5-mini',
            base_url=overrides.get('base_url') or settings.get('model_base_url'),
            api_key=overrides.get('api_key') or settings.get('model_api_key') or settings.get('openai_api_key'),
            capabilities=set(capabilities or ['text']),
        )

    def require(self, capability: str):
        if capability not in self.capabilities:
            caps = ', '.join(sorted(self.capabilities)) or 'none'
            raise ModelCapabilityError(
                f'Model "{self.name}" is configured for [{caps}] and cannot handle {capability} inputs.'
            )


class BaseModelClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        raise NotImplementedError


class OpenAIResponsesClient(BaseModelClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        kwargs = {}
        if config.api_key:
            kwargs['api_key'] = config.api_key
        if config.base_url:
            kwargs['base_url'] = config.base_url
        self.client = openai.OpenAI(**kwargs)

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        self.config.require('text')
        try:
            response = self.client.responses.create(
                model=self.config.name,
                input=messages,
                max_output_tokens=max_output_tokens,
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f'OpenAI request failed: {e}') from e
        if getattr(response, 'output_text', None):
            return response.output_text
        for output in getattr(response, 'output', []):
            for content in getattr(output, 'content', []):
                text = getattr(content, 'text', None)
                if text:
                    return text
        raise RuntimeError('Model response did not contain text output.')


class AnthropicMessagesClient(BaseModelClient):
    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        self.config.require('text')
        if not self.config.api_key:
            raise ValueError('Anthropic provider requires model_api_key in settings.')
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
        payload = {
            'model': self.config.name,
            'max_tokens': max_output_tokens,
            'messages': anthropic_messages,
        }
        if system:
            payload['system'] = system
        headers = {
            'x-api-key': self.config.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        base_url = self.config.base_url or 'https://api.anthropic.com'
        try:
            response = requests.post(f'{base_url.rstrip("/")}/v1/messages', headers=headers, json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f'Anthropic request failed: {e}') from e
        data = response.json()
        chunks = [block.get('text', '') for block in data.get('content', []) if block.get('type') == 'text']
        return ''.join(chunks)


class OpenAICompatibleChatClient(BaseModelClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        kwargs = {'api_key': config.api_key or 'not-needed'}
        if config.base_url:
            kwargs['base_url'] = config.base_url
        self.client = openai.OpenAI(**kwargs)

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        self.config.require('text')
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                messages=messages,
                max_tokens=max_output_tokens,
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f'OpenAI-compatible request failed: {e}') from e
        return response.choices[0].message.content or ''


def get_model_client(config: ModelConfig | None = None):
    config = config or ModelConfig.from_settings()
    provider = config.provider.lower().replace('_', '-')
    if provider == 'openai':
        return OpenAIResponsesClient(config)
    if provider in {'anthropic', 'claude'}:
        return AnthropicMessagesClient(config)
    if provider in {'openai-compatible', 'local', 'hpc'}:
        if not config.base_url:
            raise ValueError(f'Provider "{config.provider}" requires a base URL.')
        return OpenAICompatibleChatClient(config)
    raise ValueError(f'Unknown model provider: {config.provider}')


def query_text(messages: list[dict[str, Any]], config: ModelConfig | None = None, max_output_tokens: int = 10000) -> str:
    return get_model_client(config).query(messages, max_output_tokens=max_output_tokens)
