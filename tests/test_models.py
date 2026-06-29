"""Unit tests for paperscraper.models.

This module tests model configuration, provider selection, image encoding,
OpenAI Responses clients, Anthropic Messages clients, OpenAI-compatible chat
clients, and public query helpers without calling live model APIs.
"""

import types

import pytest

import paperscraper.models as models


def text_config(**overrides):
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


def test_model_config_from_profile_merges_settings_and_overrides(monkeypatch):
    """
    Test model config construction from a named settings profile.

    This function performs the following steps:
    1. Replaces model profile loading with a configured Anthropic profile.
    2. Replaces settings loading with provider-level API keys.
    3. Builds a model config with selected overrides.

    Asserts:
        - Overrides take precedence over profile values.
        - Provider API keys are pulled from settings when the profile lacks an API key.
        - Capability strings are normalized to a set.
        - Temperature and top-p values are converted to floats.
    """
    monkeypatch.setattr(models, 'get_model_profile', lambda profile: {
        'provider': 'anthropic',
        'model': 'profile-model',
        'base_url': 'https://profile.example',
        'capabilities': 'text, vision',
        'temperature': '0.1',
        'top_p': '0.8',
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


def test_model_config_generation_args_and_require():
    """
    Test generation arguments and capability validation.

    This function performs the following steps:
    1. Creates a text-only model config.
    2. Reads generation arguments.
    3. Requires supported and unsupported capabilities.

    Asserts:
        - Generation arguments include temperature and top-p.
        - Supported capabilities do not raise.
        - Unsupported capabilities raise `ModelCapabilityError`.
    """
    config = text_config()

    assert config.generation_args() == {'temperature': 0.2, 'top_p': 0.9}
    config.require('text')
    with pytest.raises(models.ModelCapabilityError, match='cannot handle vision inputs'):
        config.require('vision')


def test_image_to_data_url_encodes_file_with_guessed_mime_type(tmp_path):
    """
    Test local image encoding to data URL format.

    This function performs the following steps:
    1. Writes a temporary PNG file.
    2. Converts it with `image_to_data_url`.
    3. Checks the returned data URL.

    Asserts:
        - The MIME type is included.
        - The file bytes are base64 encoded.
    """
    image_path = tmp_path / 'image.png'
    image_path.write_bytes(b'image bytes')

    assert models.image_to_data_url(str(image_path)) == 'data:image/png;base64,aW1hZ2UgYnl0ZXM='


def test_base_model_client_methods_are_abstract():
    """
    Test abstract base model client methods.

    This function performs the following steps:
    1. Creates a base model client.
    2. Calls `query`.
    3. Calls `query_with_images`.

    Asserts:
        - Base text queries raise `NotImplementedError`.
        - Base image queries raise `NotImplementedError`.
    """
    client = models.BaseModelClient(text_config())

    with pytest.raises(NotImplementedError):
        client.query([])
    with pytest.raises(NotImplementedError):
        client.query_with_images('prompt', [])


def test_openai_responses_client_extracts_text_from_response_shapes(monkeypatch):
    """
    Test OpenAI Responses text extraction.

    This function performs the following steps:
    1. Replaces the OpenAI SDK client with a local fake.
    2. Extracts text from `output_text`.
    3. Extracts text from nested output content and from an empty response.

    Asserts:
        - `output_text` is preferred.
        - Nested text content is supported.
        - Responses without text raise `RuntimeError`.
    """

    class FakeOpenAI:
        def __init__(self, **kwargs):
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


def test_openai_responses_client_queries_text_and_images(monkeypatch, tmp_path):
    """
    Test OpenAI Responses text and image requests.

    This function performs the following steps:
    1. Replaces the OpenAI SDK client with a fake responses client.
    2. Sends a text query.
    3. Sends an image query with context.

    Asserts:
        - Text queries call the Responses API with generation arguments.
        - Image queries include prompt, context, and encoded image content.
        - OpenAI SDK errors are wrapped in `RuntimeError`.
    """

    class FakeResponses:
        def __init__(self):
            self.calls = []
            self.raise_error = False

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if self.raise_error:
                raise models.openai.OpenAIError('bad request')
            return types.SimpleNamespace(output_text='model text')

    class FakeOpenAI:
        responses = FakeResponses()

        def __init__(self, **kwargs):
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


def test_anthropic_messages_client_queries_text_and_images(monkeypatch, tmp_path):
    """
    Test Anthropic text and image request construction.

    This function performs the following steps:
    1. Replaces HTTP POST with a local fake response.
    2. Sends a text request with system and user messages.
    3. Sends an image request with context.

    Asserts:
        - System messages are moved into the Anthropic system field.
        - Image requests include base64 image payloads.
        - Text chunks are joined into the final response.
    """

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {'content': [{'type': 'text', 'text': 'hello'}, {'type': 'text', 'text': ' world'}]}

    calls = []

    def fake_post(url, headers, json, timeout):
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

    assert client.query_with_images('look', [str(image_path)], context='context') == 'hello world'
    content = calls[1]['json']['messages'][0]['content']
    assert content[0] == {'type': 'text', 'text': 'look'}
    assert content[1] == {'type': 'text', 'text': 'context'}
    assert content[2]['type'] == 'image'
    assert content[2]['source']['media_type'] == 'image/jpeg'


def test_anthropic_messages_client_requires_key_and_wraps_errors(monkeypatch):
    """
    Test Anthropic API key validation and request error wrapping.

    This function performs the following steps:
    1. Creates an Anthropic client without an API key.
    2. Calls text request helpers.
    3. Replaces HTTP POST with a request exception and retries with an API key.

    Asserts:
        - Missing Anthropic API keys raise `ValueError`.
        - HTTP request errors are wrapped in `RuntimeError`.
    """
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


def test_openai_compatible_chat_client_queries_text_and_images(monkeypatch, tmp_path):
    """
    Test OpenAI-compatible chat client text and image requests.

    This function performs the following steps:
    1. Replaces the OpenAI SDK client with a fake chat completions client.
    2. Sends text and image requests.
    3. Replaces the fake client with one that raises an OpenAI error.

    Asserts:
        - Text and image requests call chat completions with generation arguments.
        - Image requests include encoded image URLs.
        - OpenAI-compatible request errors are wrapped in `RuntimeError`.
    """

    class FakeCompletions:
        def __init__(self):
            self.calls = []
            self.raise_error = False

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if self.raise_error:
                raise models.openai.OpenAIError('bad request')
            message = types.SimpleNamespace(content='chat text')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        completions = FakeCompletions()

        def __init__(self, **kwargs):
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


def test_get_model_client_selects_providers_and_validates_local_base_url(monkeypatch):
    """
    Test provider client selection.

    This function performs the following steps:
    1. Requests OpenAI, Anthropic, and local clients.
    2. Requests a local client without a base URL.
    3. Requests an unknown provider.

    Asserts:
        - Known providers return the expected client class.
        - Local providers require a base URL.
        - Unknown providers raise `ValueError`.
    """
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


def test_query_helpers_delegate_to_selected_client(monkeypatch):
    """
    Test public query helper delegation.

    This function performs the following steps:
    1. Replaces `get_model_client` with a fake client.
    2. Calls `query_text`.
    3. Calls `query_images`.

    Asserts:
        - Text queries delegate to the selected client's `query` method.
        - Image queries delegate to the selected client's `query_with_images` method.
    """

    class FakeClient:
        def query(self, messages, max_output_tokens):
            assert messages == [{'role': 'user', 'content': 'hello'}]
            assert max_output_tokens == 5
            return 'text result'

        def query_with_images(self, prompt, image_paths, context, max_output_tokens):
            assert prompt == 'look'
            assert image_paths == ['image.png']
            assert context == 'context'
            assert max_output_tokens == 7
            return 'image result'

    monkeypatch.setattr(models, 'get_model_client', lambda config=None: FakeClient())

    assert models.query_text([{'role': 'user', 'content': 'hello'}], max_output_tokens=5) == 'text result'
    assert models.query_images('look', ['image.png'], context='context', max_output_tokens=7) == 'image result'
