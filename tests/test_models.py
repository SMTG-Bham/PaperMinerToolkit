"""Unit tests for paperminer.models.

This module tests model configuration, provider selection, image encoding,
OpenAI Responses clients, Anthropic Messages clients, OpenAI-compatible chat
clients, and public query helpers without calling live model APIs.
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from pathlib import Path
from typing import Any, NoReturn

import pytest

from paperminer.compression import CompressionConfig
import paperminer.models as models


def test_anthropic_error_detail_handles_every_response_shape() -> None:
    """Prefer structured provider errors and fall back to bounded response text."""
    assert models._anthropic_error_detail(None) == ''

    class InvalidJSON:
        """Response double whose body is not valid JSON."""

        text = ' plain failure '

        def json(self) -> Any:
            """Raise a response decoding error."""
            raise ValueError('bad json')

    assert models._anthropic_error_detail(InvalidJSON()) == 'plain failure'
    response = SimpleNamespace(json=lambda: {'error': {'message': 'structured failure'}}, text='ignored')
    assert models._anthropic_error_detail(response) == 'structured failure'


def text_config(**overrides: Any) -> models.ModelConfig:
    """Return a text-capable model config for model unit tests."""
    values = {
        'provider': 'openai',
        'name': 'test-model',
        'api_key': 'test-key',
        'capabilities': {'text'},
        'temperature': 0.2,
        'top_p': 0.9,
    }
    values.update(overrides)
    return models.ModelConfig(**values)


def test_model_config_from_profile_merges_settings_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test profile settings, provider keys, and explicit overrides."""
    monkeypatch.setattr(models, 'get_model_profile', lambda profile: {
        'provider': 'anthropic',
        'model': 'profile-model',
        'base_url': 'https://profile.example',
        'capabilities': 'text, vision',
        'temperature': '0.1',
        'top_p': '0.8',
        'input_token_limit': '64000',
    })
    monkeypatch.setattr(models, 'load_settings', lambda: {'anthropic_api_key': 'settings-key'})

    config = models.ModelConfig.from_profile('vision', model='override-model', temperature='0.3')

    assert config.provider == 'anthropic'
    assert config.name == 'override-model'
    assert config.base_url == 'https://profile.example'
    assert config.api_key == 'settings-key'
    assert config.capabilities == {'text', 'vision'}
    assert config.temperature == 0.3
    assert config.top_p == 0.8
    assert config.input_token_limit == 64000


def test_model_config_provider_override_drops_profile_specific_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not send a provider override through another provider's URL or API key."""
    monkeypatch.setattr(models, 'get_model_profile', lambda profile: {
        'provider': 'local',
        'model': 'qwen',
        'base_url': 'http://127.0.0.1:8000/v1',
        'api_key': 'local-key',
        'capabilities': ['text'],
        'temperature': 0,
        'top_p': 1,
        'input_token_limit': 32000,
    })
    monkeypatch.setattr(models, 'load_settings', lambda: {'anthropic_api_key': 'anthropic-key'})

    config = models.ModelConfig.from_profile(
        'text',
        provider='anthropic',
        model='claude-test',
    )

    assert config.provider == 'anthropic'
    assert config.name == 'claude-test'
    assert config.base_url is None
    assert config.api_key == 'anthropic-key'

    with pytest.raises(ValueError, match='explicit model name'):
        models.ModelConfig.from_profile('text', provider='anthropic')


def test_model_config_generation_args_and_require() -> None:
    """Test generation arguments and capability validation."""
    config = text_config()

    assert config.generation_args() == {'temperature': 0.2, 'top_p': 0.9}
    assert text_config(provider='anthropic').generation_args() == {'temperature': 0.2}
    assert text_config(name='gpt-5.6-terra').generation_args() == {}
    assert text_config(provider='local', name='gpt-5.6-terra').generation_args() == {
        'temperature': 0.2,
        'top_p': 0.9,
    }
    config.require('text')
    with pytest.raises(models.ModelCapabilityError, match='cannot handle vision inputs'):
        config.require('vision')


def test_image_to_data_url_encodes_file_with_guessed_mime_type(tmp_path: Path) -> None:
    """Test local image encoding with an inferred MIME type."""
    image_path = tmp_path / 'image.png'
    image_path.write_bytes(b'image bytes')

    assert models._image_to_data_url(str(image_path)) == 'data:image/png;base64,aW1hZ2UgYnl0ZXM='


def test_base_model_client_methods_are_abstract() -> None:
    """Test that base model request methods are abstract."""
    client = models.BaseModelClient(text_config())

    with pytest.raises(NotImplementedError):
        client.query([])
    with pytest.raises(NotImplementedError):
        client.query_with_images('prompt', [])


def test_openai_responses_client_extracts_text_from_response_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test text extraction from supported OpenAI response shapes."""

    class FakeOpenAI:
        """Capture OpenAI client initialization options."""

        def __init__(self, **kwargs: Any) -> None:
            """Store client initialization options."""
            self.kwargs = kwargs

    monkeypatch.setattr(models.openai, 'OpenAI', FakeOpenAI)
    client = models.OpenAIResponsesClient(text_config(base_url='https://api.example'))

    assert client.client.kwargs == {'api_key': 'test-key', 'base_url': 'https://api.example'}
    assert client._response_text(types.SimpleNamespace(output_text='direct text')) == 'direct text'
    nested = types.SimpleNamespace(output=[
        types.SimpleNamespace(content=[types.SimpleNamespace(text='nested text')])
    ])
    assert client._response_text(nested) == 'nested text'
    with pytest.raises(RuntimeError, match='did not contain text'):
        client._response_text(types.SimpleNamespace(output=[]))


def test_openai_responses_client_queries_text_and_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test OpenAI Responses text, image, and error paths."""

    class FakeResponses:
        """Capture Responses API calls and optionally raise errors."""

        def __init__(self) -> None:
            """Initialize response call tracking."""
            self.calls = []
            self.raise_error = False

        def create(self, **kwargs: Any) -> types.SimpleNamespace:
            """Record a request and return or raise a fake response."""
            self.calls.append(kwargs)
            if self.raise_error:
                raise models.openai.OpenAIError('bad request')
            return types.SimpleNamespace(output_text='model text')

    class FakeOpenAI:
        """Expose a shared fake Responses API resource."""

        responses = FakeResponses()

        def __init__(self, **kwargs: Any) -> None:
            """Attach the shared Responses API resource."""
            self.responses = FakeOpenAI.responses

    monkeypatch.setattr(models.openai, 'OpenAI', FakeOpenAI)
    image_path = tmp_path / 'image.png'
    image_path.write_bytes(b'image bytes')
    client = models.OpenAIResponsesClient(text_config(capabilities={'text', 'vision'}))

    assert client.query([{'role': 'user', 'content': 'hello'}], max_output_tokens=5) == 'model text'
    assert FakeOpenAI.responses.calls[0]['max_output_tokens'] == 5
    assert FakeOpenAI.responses.calls[0]['temperature'] == 0.2

    assert client.query_with_images('look', [str(image_path)], context='context', max_output_tokens=7) == 'model text'
    image_call = FakeOpenAI.responses.calls[1]
    content = image_call['input'][0]['content']
    assert content[0] == {'type': 'input_text', 'text': 'look'}
    assert content[1] == {'type': 'input_text', 'text': 'context'}
    assert content[2]['type'] == 'input_image'

    FakeOpenAI.responses.raise_error = True
    with pytest.raises(RuntimeError, match='OpenAI request failed'):
        client.query([{'role': 'user', 'content': 'hello'}])
    with pytest.raises(RuntimeError, match='OpenAI vision request failed'):
        client.query_with_images('look', [str(image_path)])


def test_openai_vision_client_applies_image_compression_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that OpenAI vision requests apply image compression."""
    calls = {}

    class FakeResponses:
        """Capture a Responses API request."""

        def create(self, **kwargs: Any) -> types.SimpleNamespace:
            """Record a request and return model text."""
            calls['request'] = kwargs
            return types.SimpleNamespace(output_text='model text')

    class FakeOpenAI:
        """Expose a fake Responses API resource."""

        def __init__(self, **kwargs: Any) -> None:
            """Initialize a fake Responses API resource."""
            self.responses = FakeResponses()

    def fake_compress(
        messages: list[dict[str, Any]],
        image_paths: list[str],
        prompt: str,
        context: str,
        model_config: models.ModelConfig,
        compression_config: CompressionConfig,
    ) -> list[dict[str, Any]]:
        """Record compression inputs and return a compressed payload."""
        calls['compression'] = {
            'messages': messages,
            'image_paths': image_paths,
            'prompt': prompt,
            'context': context,
            'model_config': model_config,
            'compression_config': compression_config,
        }
        return [{'role': 'user', 'content': [{'type': 'compressed_image'}]}]

    monkeypatch.setattr(models.openai, 'OpenAI', FakeOpenAI)
    monkeypatch.setattr(models, 'maybe_compress_image_messages', fake_compress)
    image_path = tmp_path / 'image.png'
    image_path.write_bytes(b'image bytes')
    config = text_config(capabilities={'text', 'vision'})
    compression_config = CompressionConfig(scope='images', mode='always')
    client = models.OpenAIResponsesClient(config)

    assert client.query_with_images('look', [str(image_path)], context='context',
                                    compression_config=compression_config) == 'model text'
    assert calls['request']['input'] == [{'role': 'user', 'content': [{'type': 'compressed_image'}]}]
    assert calls['compression']['image_paths'] == [str(image_path)]
    assert calls['compression']['prompt'] == 'look'
    assert calls['compression']['context'] == 'context'
    assert calls['compression']['model_config'] is config
    assert calls['compression']['compression_config'] is compression_config


def test_anthropic_messages_client_queries_text_and_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test Anthropic text and image request construction."""

    class FakeResponse:
        """Provide a successful Anthropic HTTP response."""

        def raise_for_status(self) -> None:
            """Accept the fake HTTP status."""
            return None

        def json(self) -> dict[str, Any]:
            """Return two Anthropic text content blocks."""
            return {'content': [{'type': 'text', 'text': 'hello'}, {'type': 'text', 'text': ' world'}]}

    calls = []

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        """Record an HTTP request and return a successful response."""
        calls.append({'url': url, 'headers': headers, 'json': json, 'timeout': timeout})
        return FakeResponse()

    monkeypatch.setattr(models.requests, 'post', fake_post)
    image_path = tmp_path / 'image.jpg'
    image_path.write_bytes(b'image bytes')
    config = text_config(provider='anthropic', base_url='https://anthropic.example', capabilities={'text', 'vision'})
    client = models.AnthropicMessagesClient(config)

    assert client.query([
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'question'},
    ]) == 'hello world'
    assert calls[0]['url'] == 'https://anthropic.example/v1/messages'
    assert calls[0]['json']['system'] == 'system'
    assert calls[0]['json']['messages'] == [{'role': 'user', 'content': 'question'}]
    assert calls[0]['json']['temperature'] == 0.2
    assert 'top_p' not in calls[0]['json']

    assert client.query_with_images('look', [str(image_path)], context='context') == 'hello world'
    content = calls[1]['json']['messages'][0]['content']
    assert content[0] == {'type': 'text', 'text': 'look'}
    assert content[1] == {'type': 'text', 'text': 'context'}
    assert content[2]['type'] == 'image'
    assert content[2]['source']['media_type'] == 'image/jpeg'


def test_anthropic_messages_client_handles_versioned_base_url_and_error_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid duplicate API versions and retain Anthropic validation messages."""
    calls = []

    class ErrorResponse:
        """Provide an Anthropic validation error response."""

        text = ''

        def raise_for_status(self) -> NoReturn:
            """Raise an HTTP error associated with this response."""
            raise models.requests.HTTPError('400 Client Error', response=self)

        def json(self) -> dict[str, Any]:
            """Return structured Anthropic validation detail."""
            return {'error': {'type': 'invalid_request_error', 'message': 'invalid sampling parameters'}}

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> ErrorResponse:
        """Record an HTTP request and return a validation error."""
        calls.append(url)
        return ErrorResponse()

    monkeypatch.setattr(models.requests, 'post', fake_post)
    client = models.AnthropicMessagesClient(text_config(
        provider='anthropic',
        base_url='https://anthropic.example/v1/',
    ))

    with pytest.raises(RuntimeError, match='invalid sampling parameters'):
        client.query([{'role': 'user', 'content': 'question'}])
    assert calls == ['https://anthropic.example/v1/messages']


def test_anthropic_messages_client_requires_key_and_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Anthropic API key validation and request error wrapping."""
    no_key_client = models.AnthropicMessagesClient(text_config(provider='anthropic', api_key=None))

    with pytest.raises(ValueError, match='Anthropic provider requires an API key'):
        no_key_client.query([{'role': 'user', 'content': 'question'}])
    with pytest.raises(ValueError, match='Anthropic provider requires an API key'):
        no_key_client._request('', [], 10)

    monkeypatch.setattr(
        models.requests,
        'post',
        lambda *_, **__: (_ for _ in ()).throw(models.requests.RequestException('network down')),
    )
    client = models.AnthropicMessagesClient(text_config(provider='anthropic'))

    with pytest.raises(RuntimeError, match='Anthropic request failed'):
        client.query([{'role': 'user', 'content': 'question'}])


def test_openai_compatible_chat_client_queries_text_and_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test OpenAI-compatible text, image, and error paths."""

    class FakeCompletions:
        """Capture chat completion calls and optionally raise errors."""

        def __init__(self) -> None:
            """Initialize completion call tracking."""
            self.calls = []
            self.raise_error = False

        def create(self, **kwargs: Any) -> types.SimpleNamespace:
            """Record a request and return or raise a fake completion."""
            self.calls.append(kwargs)
            if self.raise_error:
                raise models.openai.OpenAIError('bad request')
            message = types.SimpleNamespace(content='chat text')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        """Expose a shared fake chat completions resource."""

        completions = FakeCompletions()

        def __init__(self, **kwargs: Any) -> None:
            """Store options and attach the fake chat resource."""
            self.kwargs = kwargs
            self.chat = types.SimpleNamespace(completions=FakeOpenAI.completions)

    monkeypatch.setattr(models.openai, 'OpenAI', FakeOpenAI)
    image_path = tmp_path / 'image.png'
    image_path.write_bytes(b'image bytes')
    config = text_config(provider='local', base_url='http://127.0.0.1:8000/v1', api_key=None, capabilities={'text', 'vision'})
    client = models.OpenAICompatibleChatClient(config)

    assert client.client.kwargs == {'api_key': 'not-needed', 'base_url': 'http://127.0.0.1:8000/v1'}
    assert client.query([{'role': 'user', 'content': 'hello'}], max_output_tokens=5) == 'chat text'
    assert FakeOpenAI.completions.calls[0]['max_tokens'] == 5
    assert FakeOpenAI.completions.calls[0]['temperature'] == 0.2

    assert client.query_with_images('look', [str(image_path)], context='context') == 'chat text'
    content = FakeOpenAI.completions.calls[1]['messages'][0]['content']
    assert content[0] == {'type': 'text', 'text': 'look'}
    assert content[1] == {'type': 'text', 'text': 'context'}
    assert content[2]['type'] == 'image_url'

    FakeOpenAI.completions.raise_error = True
    with pytest.raises(RuntimeError, match='Local model request failed'):
        client.query([{'role': 'user', 'content': 'hello'}])


def test_get_model_client_selects_providers_and_validates_local_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test provider selection and local base URL validation."""
    monkeypatch.setattr(models.openai, 'OpenAI', lambda **_: types.SimpleNamespace())

    assert isinstance(models.get_model_client(text_config(provider='openai')), models.OpenAIResponsesClient)
    assert isinstance(models.get_model_client(text_config(provider='anthropic')), models.AnthropicMessagesClient)
    assert isinstance(
        models.get_model_client(text_config(provider='local', base_url='http://127.0.0.1:8000/v1')),
        models.OpenAICompatibleChatClient,
    )
    with pytest.raises(ValueError, match='requires a base URL'):
        models.get_model_client(text_config(provider='local', base_url=None))
    with pytest.raises(ValueError, match='Unknown model provider'):
        models.get_model_client(text_config(provider='unknown'))


def test_query_helpers_delegate_to_selected_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that public query helpers delegate to the selected client."""

    class FakeClient:
        """Provide deterministic text and image query methods."""

        def query(
            self,
            messages: list[dict[str, Any]],
            max_output_tokens: int,
        ) -> str:
            """Validate a text request and return a result."""
            assert messages == [{'role': 'user', 'content': 'hello'}]
            assert max_output_tokens == 5
            return 'text result'

        def query_with_images(
            self,
            prompt: str,
            image_paths: list[str],
            context: str,
            max_output_tokens: int,
            compression_config: CompressionConfig | None = None,
        ) -> str:
            """Validate an image request and return a result."""
            assert prompt == 'look'
            assert image_paths == ['image.png']
            assert context == 'context'
            assert max_output_tokens == 7
            assert compression_config is None
            return 'image result'

    monkeypatch.setattr(models, 'get_model_client', lambda config=None: FakeClient())

    assert models.query_text([{'role': 'user', 'content': 'hello'}], max_output_tokens=5) == 'text result'
    assert models.query_images('look', ['image.png'], context='context', max_output_tokens=7) == 'image result'
