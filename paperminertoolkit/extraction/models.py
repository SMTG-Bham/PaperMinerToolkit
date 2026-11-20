"""Model configuration and provider clients for text and vision requests.

This module hides provider differences behind a small client interface used by
the extraction code. It supports OpenAI Responses, Anthropic Messages, and
OpenAI-compatible local chat servers.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import openai
import os
import requests
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self, TypedDict, Unpack

from paperminertoolkit.extraction._anthropic import messages_url, request_headers
from paperminertoolkit.extraction.compression import CompressionConfig, maybe_compress_image_messages
from paperminertoolkit.settings import (DEFAULT_INPUT_TOKEN_LIMIT, DEFAULT_MODEL, DEFAULT_OUTPUT_TOKEN_LIMIT,
                                        DEFAULT_REQUEST_TIMEOUT, get_model_profile, load_settings,
                                        reasoning_effort_setting, thinking_setting)

OPENAI_FIXED_SAMPLING_PREFIXES = ('gpt-5', 'o1', 'o3', 'o4')


class _ModelOverrides(TypedDict, total=False):
    """Describe explicit overrides accepted by a model profile."""

    provider: str
    name: str
    model: str
    base_url: str | None
    api_key: str | None
    capabilities: str | list[str] | set[str] | tuple[str, ...]
    temperature: float | int | str
    top_p: float | int | str
    input_token_limit: int | str
    output_token_limit: int | str
    request_timeout: float | int | str
    reasoning_effort: str | None
    thinking: bool | str | None


class ModelCapabilityError(ValueError):
    """Raised when a configured model is asked to handle an unsupported input type."""

    pass


@dataclass
class ModelConfig:
    """Configure one text or vision model profile.

    Attributes
    ----------
    provider : str
        Model provider name.
    name : str
        Provider model identifier.
    base_url : str | None
        Optional provider API base URL.
    api_key : str | None
        Optional provider API key.
    capabilities : set[str]
        Supported input capabilities, such as ``"text"`` and ``"vision"``.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling probability.
    input_token_limit : int
        Maximum model input size in tokens.
    output_token_limit : int
        Maximum tokens generated per extraction request.
    request_timeout : float
        Seconds to wait for one request to an OpenAI-compatible server.
    reasoning_effort : str | None
        Reasoning effort sent with each request, or ``None`` for the model's
        default.
    thinking : bool | None
        Whether a hybrid model thinks before answering (OpenAI-compatible
        servers only), or ``None`` for the model's default.
    """

    provider: str = 'openai'
    name: str = DEFAULT_MODEL
    base_url: str | None = None
    api_key: str | None = None
    capabilities: set[str] = field(default_factory=lambda: {'text'})
    temperature: float = 0
    top_p: float = 1
    input_token_limit: int = DEFAULT_INPUT_TOKEN_LIMIT
    output_token_limit: int = DEFAULT_OUTPUT_TOKEN_LIMIT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    reasoning_effort: str | None = None
    thinking: bool | None = None

    @classmethod
    def from_profile(cls, profile: str = 'text', **overrides: Unpack[_ModelOverrides]) -> Self:
        """Build a configuration from a settings profile.

        Parameters
        ----------
        profile : str, optional
            Settings profile to load.
        **overrides : Unpack[_ModelOverrides]
            Explicit configuration values that replace profile settings.

        Returns
        -------
        ModelConfig
            Resolved model configuration.

        Raises
        ------
        ValueError
            If the provider changes without an explicit model name or a
            numeric setting cannot be converted.
        """
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
        # Reasoning settings are provider-specific too: a profile's effort or thinking switch
        # for one provider must not be sent to another (e.g. a separate recipe-optimiser model).
        profile_effort = None if provider_changed else configured.get('reasoning_effort')
        profile_thinking = None if provider_changed else configured.get('thinking')
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
            output_token_limit=int(
                overrides.get('output_token_limit', configured.get('output_token_limit', DEFAULT_OUTPUT_TOKEN_LIMIT))
            ),
            request_timeout=float(
                overrides.get('request_timeout', configured.get('request_timeout', DEFAULT_REQUEST_TIMEOUT))
            ),
            reasoning_effort=reasoning_effort_setting(overrides.get('reasoning_effort', profile_effort)),
            thinking=thinking_setting(overrides.get('thinking', profile_thinking)),
        )

    def generation_args(self) -> dict[str, float]:
        """Build provider-specific generation parameters.

        Returns
        -------
        dict[str, float]
            Supported sampling arguments for the configured provider and
            model.
        """
        provider = self.provider.lower().replace('_', '-')
        model = self.name.lower()
        if provider == 'openai' and model.startswith(OPENAI_FIXED_SAMPLING_PREFIXES):
            return {}
        args = {'temperature': self.temperature, 'top_p': self.top_p}
        if provider == 'anthropic':
            args.pop('top_p')
        return args

    def reasoning_args(self) -> dict[str, Any]:
        """Build provider-specific reasoning parameters.

        Returns
        -------
        dict[str, Any]
            Request arguments that set the reasoning effort and thinking
            switch, or an empty dict when both use the model's default.
            OpenAI-compatible servers get ``reasoning_effort`` and, for
            ``thinking``, ``chat_template_kwargs.enable_thinking``; OpenAI's
            Responses API gets ``reasoning.effort``.

        Raises
        ------
        ValueError
            If the provider does not support a configured setting.
        """
        provider = self.provider.lower().replace('_', '-')
        if self.reasoning_effort is None and self.thinking is None:
            return {}
        if provider == 'local':
            args: dict[str, Any] = {}
            if self.reasoning_effort is not None:
                args['reasoning_effort'] = self.reasoning_effort
            if self.thinking is not None:
                args['extra_body'] = {'chat_template_kwargs': {'enable_thinking': self.thinking}}
            return args
        if provider == 'openai':
            if self.thinking is not None:
                raise ValueError('thinking is only supported for local OpenAI-compatible servers; '
                                 'set reasoning_effort for OpenAI models.')
            return {'reasoning': {'effort': self.reasoning_effort}}
        raise ValueError(f'Reasoning settings are not supported for provider "{self.provider}".')

    def require(self, capability: str) -> None:
        """Require a model input capability.

        Parameters
        ----------
        capability : str
            Capability required by the pending request.

        Raises
        ------
        ModelCapabilityError
            If the configured model does not declare the capability.
        """
        if capability not in self.capabilities:
            caps = ', '.join(sorted(self.capabilities)) or 'none'
            raise ModelCapabilityError(
                f'Model "{self.name}" is configured for [{caps}] and cannot handle {capability} inputs.'
            )


def _image_to_data_url(path: str) -> str:
    """Encode a local image as a data URL.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    str
        MIME-typed base64 data URL.
    """
    mime_type = mimetypes.guess_type(path)[0] or 'image/png'
    with open(path, 'rb') as image_file:
        encoded = base64.b64encode(image_file.read()).decode('ascii')
    return f'data:{mime_type};base64,{encoded}'


def _image_label(image_labels: Sequence[str] | None, index: int) -> str:
    """Return the label describing one image in a vision request.

    Parameters
    ----------
    image_labels : Sequence[str] or None
        Per-image descriptions aligned with the request's image order.
    index : int
        Position of the image within the request.

    Returns
    -------
    str
        Label text, or an empty string when the image has no description.
    """
    if image_labels is None or index >= len(image_labels):
        return ''
    return str(image_labels[index]).strip()


class BaseModelClient:
    """Abstract interface implemented by concrete model provider clients."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize a model client.

        Parameters
        ----------
        config : ModelConfig
            Configuration used for provider requests.

        Raises
        ------
        ValueError
            If the provider does not support the configured reasoning
            settings.
        """
        self.config = config
        config.reasoning_args()

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        """Send a text-only request.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Provider-neutral chat messages.
        max_output_tokens : int, optional
            Maximum number of tokens to generate.

        Returns
        -------
        str
            Model response text.

        Raises
        ------
        NotImplementedError
            Always; subclasses must implement text requests.
        """
        raise NotImplementedError

    def query_with_images(self,
                          prompt: str,
                          image_paths: list[str],
                          context: str | None = None,
                          max_output_tokens: int = 10000,
                          compression_config: CompressionConfig | None = None,
                          image_labels: Sequence[str] | None = None) -> str:
        """Send a vision request with local images.

        Parameters
        ----------
        prompt : str
            Instructions for image analysis.
        image_paths : list[str]
            Local images to include in the request.
        context : str, optional
            Additional text context.
        max_output_tokens : int, optional
            Maximum number of tokens to generate.
        compression_config : CompressionConfig | None, optional
            Compression policy for the provider payload.
        image_labels : Sequence[str] or None, optional
            Per-image descriptions, such as a figure label and caption, sent
            immediately before their image so the model can attribute
            findings to a specific figure.

        Returns
        -------
        str
            Model response text.

        Raises
        ------
        NotImplementedError
            Always; subclasses must implement vision requests.
        """
        raise NotImplementedError


class OpenAIResponsesClient(BaseModelClient):
    """Client for OpenAI's Responses API."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize an OpenAI Responses client.

        Parameters
        ----------
        config : ModelConfig
            OpenAI model and connection configuration.
        """
        super().__init__(config)
        kwargs: dict[str, Any] = {'timeout': config.request_timeout}
        if config.api_key:
            kwargs['api_key'] = config.api_key
        if config.base_url:
            kwargs['base_url'] = config.base_url
        self.client = openai.OpenAI(**kwargs)

    def _response_text(self, response: object) -> str:
        """Extract text from an OpenAI response.

        Parameters
        ----------
        response : object
            Responses API result object.

        Returns
        -------
        str
            First available response text.

        Raises
        ------
        RuntimeError
            If the response contains no text output.
        """
        if getattr(response, 'output_text', None):
            return response.output_text
        for output in getattr(response, 'output', []):
            for content in getattr(output, 'content', []):
                text = getattr(content, 'text', None)
                if text:
                    return text
        raise RuntimeError('Model response did not contain text output.')

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        """Send a text request through OpenAI Responses.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Responses API input messages.
        max_output_tokens : int, optional
            Maximum number of tokens to generate.

        Returns
        -------
        str
            Model response text.

        Raises
        ------
        ModelCapabilityError
            If the model lacks text capability.
        RuntimeError
            If the OpenAI request fails or returns no text.
        """
        self.config.require('text')
        try:
            response = self.client.responses.create(
                model=self.config.name,
                input=messages,
                max_output_tokens=max_output_tokens,
                **self.config.generation_args(),
                **self.config.reasoning_args(),
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f'OpenAI request failed: {e}') from e
        return self._response_text(response)

    def query_with_images(self,
                          prompt: str,
                          image_paths: list[str],
                          context: str | None = None,
                          max_output_tokens: int = 10000,
                          compression_config: CompressionConfig | None = None,
                          image_labels: Sequence[str] | None = None) -> str:
        """Send an image request through OpenAI Responses.

        Parameters
        ----------
        prompt : str
            Instructions for image analysis.
        image_paths : list[str]
            Local images to include in the request.
        context : str, optional
            Additional text context.
        max_output_tokens : int, optional
            Maximum number of tokens to generate.
        compression_config : CompressionConfig | None, optional
            Compression policy for the provider payload.
        image_labels : Sequence[str] or None, optional
            Per-image descriptions sent immediately before their image.

        Returns
        -------
        str
            Model response text.

        Raises
        ------
        ModelCapabilityError
            If the model lacks vision capability.
        RuntimeError
            If the OpenAI request fails or returns no text.
        """
        self.config.require('vision')
        content = [{'type': 'input_text', 'text': prompt}]
        if context:
            content.append({'type': 'input_text', 'text': context})
        for index, image_path in enumerate(image_paths):
            label = _image_label(image_labels, index)
            if label:
                content.append({'type': 'input_text', 'text': label})
            content.append({'type': 'input_image', 'image_url': _image_to_data_url(image_path)})
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
                **self.config.reasoning_args(),
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f'OpenAI vision request failed: {e}') from e
        return self._response_text(response)


class AnthropicMessagesClient(BaseModelClient):
    """Client for Anthropic's Messages API."""

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        """Send a text request through Anthropic Messages.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Provider-neutral chat messages.
        max_output_tokens : int, optional
            Maximum number of tokens to generate.

        Returns
        -------
        str
            Joined text from the Anthropic response.

        Raises
        ------
        ModelCapabilityError
            If the model lacks text capability.
        ValueError
            If no Anthropic API key is configured.
        RuntimeError
            If the provider request fails.
        """
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
                          compression_config: CompressionConfig | None = None,
                          image_labels: Sequence[str] | None = None) -> str:
        """Send an image request through Anthropic Messages.

        Parameters
        ----------
        prompt : str
            Instructions for image analysis.
        image_paths : list[str]
            Local images to encode in the request.
        context : str, optional
            Additional text context.
        max_output_tokens : int, optional
            Maximum number of tokens to generate.
        compression_config : CompressionConfig | None, optional
            Compression policy for the provider payload.
        image_labels : Sequence[str] or None, optional
            Per-image descriptions sent immediately before their image.

        Returns
        -------
        str
            Joined text from the Anthropic response.

        Raises
        ------
        ModelCapabilityError
            If the model lacks vision capability.
        ValueError
            If no Anthropic API key is configured.
        RuntimeError
            If the provider request fails.
        """
        self.config.require('vision')
        content = [{'type': 'text', 'text': prompt}]
        if context:
            content.append({'type': 'text', 'text': context})
        for index, image_path in enumerate(image_paths):
            label = _image_label(image_labels, index)
            if label:
                content.append({'type': 'text', 'text': label})
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

    def _request(
        self,
        system: str,
        anthropic_messages: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> str:
        """Post a prepared Anthropic Messages request.

        Parameters
        ----------
        system : str
            System instructions for the request.
        anthropic_messages : list[dict[str, Any]]
            Anthropic-formatted conversation messages.
        max_output_tokens : int
            Maximum number of tokens to generate.

        Returns
        -------
        str
            Joined text blocks from the response.

        Raises
        ------
        ValueError
            If no Anthropic API key is configured.
        RuntimeError
            If the HTTP request fails.
        """
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
        headers = request_headers(self.config.api_key)
        endpoint = messages_url(self.config.base_url)
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


def _anthropic_error_detail(response: requests.Response | None) -> str:
    """Extract a concise Anthropic error message.

    Parameters
    ----------
    response : requests.Response | None
        Failed HTTP response.

    Returns
    -------
    str
        Provider error detail truncated to 1,000 characters.
    """
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

    def __init__(self, config: ModelConfig) -> None:
        """Initialize an OpenAI-compatible chat client.

        Parameters
        ----------
        config : ModelConfig
            Model and connection configuration.
        """
        super().__init__(config)
        kwargs: dict[str, Any] = {'api_key': config.api_key or 'not-needed', 'timeout': config.request_timeout}
        if config.base_url:
            kwargs['base_url'] = config.base_url
        self.client = openai.OpenAI(**kwargs)

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        """Send a text chat completion request.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Chat completion messages.
        max_output_tokens : int, optional
            Maximum number of tokens to generate.

        Returns
        -------
        str
            Model response text.

        Raises
        ------
        ModelCapabilityError
            If the model lacks text capability.
        RuntimeError
            If the provider request fails.
        """
        self.config.require('text')
        return self._chat(messages, max_output_tokens)

    def query_with_images(self,
                          prompt: str,
                          image_paths: list[str],
                          context: str | None = None,
                          max_output_tokens: int = 10000,
                          compression_config: CompressionConfig | None = None,
                          image_labels: Sequence[str] | None = None) -> str:
        """Send images as OpenAI-compatible chat content.

        Parameters
        ----------
        prompt : str
            Instructions for image analysis.
        image_paths : list[str]
            Local images to encode in the request.
        context : str, optional
            Additional text context.
        max_output_tokens : int, optional
            Maximum number of tokens to generate.
        compression_config : CompressionConfig | None, optional
            Compression policy for the provider payload.
        image_labels : Sequence[str] or None, optional
            Per-image descriptions sent immediately before their image.

        Returns
        -------
        str
            Model response text.

        Raises
        ------
        ModelCapabilityError
            If the model lacks vision capability.
        RuntimeError
            If the provider request fails.
        """
        self.config.require('vision')
        content = [{'type': 'text', 'text': prompt}]
        if context:
            content.append({'type': 'text', 'text': context})
        for index, image_path in enumerate(image_paths):
            label = _image_label(image_labels, index)
            if label:
                content.append({'type': 'text', 'text': label})
            content.append({'type': 'image_url', 'image_url': {'url': _image_to_data_url(image_path)}})
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

    def _chat(self, messages: list[dict[str, Any]], max_output_tokens: int) -> str:
        """Call the chat completions endpoint.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Provider-formatted chat messages.
        max_output_tokens : int
            Maximum number of tokens to generate.

        Returns
        -------
        str
            First response message text, or an empty string.

        Raises
        ------
        RuntimeError
            If the provider request fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                messages=messages,
                max_tokens=max_output_tokens,
                **self.config.generation_args(),
                **self.config.reasoning_args(),
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f'Local model request failed: {e}') from e
        return response.choices[0].message.content or ''


class ClaudeCodeClient(BaseModelClient):
    """Client that sends text requests through a logged-in Claude Code CLI.

    This uses ``claude -p`` with every tool disabled, so requests are billed to
    the Claude Code login rather than an API key. The executable is read from
    ``PAPERMINERTOOLKIT_CLAUDE_CODE_BIN`` or found on ``PATH`` as ``claude``.
    Each request may run for at most the configured ``request_timeout``.
    """

    def _executable(self) -> str:
        """Locate the Claude Code executable.

        Returns
        -------
        str
            Path to the ``claude`` executable.

        Raises
        ------
        ValueError
            If no executable can be found.
        """
        executable = os.environ.get('PAPERMINERTOOLKIT_CLAUDE_CODE_BIN') or shutil.which('claude')
        if not executable:
            raise ValueError('Provider "claude-code" needs the claude CLI on PATH or '
                             'PAPERMINERTOOLKIT_CLAUDE_CODE_BIN set to its path.')
        return executable

    def query(self, messages: list[dict[str, Any]], max_output_tokens: int = 10000) -> str:
        """Send a text-only request through ``claude -p``.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Provider-neutral chat messages. System messages become the system
            prompt and the remaining messages are sent as the user prompt.
        max_output_tokens : int, optional
            Unused; Claude Code applies its own output limit.

        Returns
        -------
        str
            Model response text.

        Raises
        ------
        RuntimeError
            If the CLI fails, times out, or reports an error result.
        """
        system = '\n\n'.join(str(m['content']) for m in messages if m.get('role') == 'system')
        prompt = '\n\n'.join(str(m['content']) for m in messages if m.get('role') != 'system')
        command = [self._executable(), '-p', '--output-format', 'json', '--tools', '',
                   '--strict-mcp-config', '--no-session-persistence', '--model', self.config.name]
        if system:
            command += ['--system-prompt', system]
        with tempfile.TemporaryDirectory(prefix='paperminertoolkit-claude-') as cwd:
            try:
                completed = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                           encoding='utf-8', cwd=cwd, timeout=self.config.request_timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError('Claude Code request timed out.') from error
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            detail = (completed.stderr or completed.stdout).strip()[:500]
            raise RuntimeError(f'Claude Code returned no JSON result (exit {completed.returncode}): {detail}') from error
        if result.get('is_error') or completed.returncode:
            raise RuntimeError(f'Claude Code request failed: {str(result.get("result"))[:500]}')
        return str(result.get('result') or '')

    def query_with_images(self,
                          prompt: str,
                          image_paths: list[str],
                          context: str | None = None,
                          max_output_tokens: int = 10000,
                          compression_config: CompressionConfig | None = None,
                          image_labels: Sequence[str] | None = None) -> str:
        """Reject image requests, which this client does not support.

        Raises
        ------
        ModelCapabilityError
            Always.
        """
        raise ModelCapabilityError('Provider "claude-code" supports text requests only.')


def get_model_client(config: ModelConfig | None = None) -> BaseModelClient:
    """Create the provider client for a model configuration.

    Parameters
    ----------
    config : ModelConfig | None, optional
        Model configuration. The text profile is used by default.

    Returns
    -------
    BaseModelClient
        Client implementation for the configured provider.

    Raises
    ------
    ValueError
        If the provider is unknown or a local provider lacks a base URL.
    """
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
    if provider == 'claude-code':
        return ClaudeCodeClient(config)
    raise ValueError(f'Unknown model provider: {config.provider}')


def query_text(messages: list[dict[str, Any]],
               config: ModelConfig | None = None,
               max_output_tokens: int = 10000) -> str:
    """Send a text request through a configured provider.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        Provider-neutral chat messages.
    config : ModelConfig | None, optional
        Model configuration. The text profile is used by default.
    max_output_tokens : int, optional
        Maximum number of tokens to generate.

    Returns
    -------
    str
        Model response text.
    """
    return get_model_client(config).query(messages, max_output_tokens=max_output_tokens)


def query_images(prompt: str,
                 image_paths: list[str],
                 config: ModelConfig | None = None,
                 context: str | None = None,
                 max_output_tokens: int = 10000,
                 compression_config: CompressionConfig | None = None,
                 image_labels: Sequence[str] | None = None) -> str:
    """Send an image request through a configured provider.

    Parameters
    ----------
    prompt : str
        Instructions for image analysis.
    image_paths : list[str]
        Local images to include in the request.
    config : ModelConfig | None, optional
        Vision model configuration.
    context : str, optional
        Additional text context.
    max_output_tokens : int, optional
        Maximum number of tokens to generate.
    compression_config : CompressionConfig | None, optional
        Compression policy for the provider payload.
    image_labels : Sequence[str] or None, optional
        Per-image descriptions sent immediately before their image.

    Returns
    -------
    str
        Model response text.
    """
    return get_model_client(config).query_with_images(prompt, image_paths, context=context,
                                                      max_output_tokens=max_output_tokens,
                                                      compression_config=compression_config,
                                                      image_labels=image_labels)
