"""Test compression policies, token estimates, and Headroom integration."""

from __future__ import annotations

import builtins
from pathlib import Path
import sys
import types
from typing import Any

import pytest

import paperscraper.compression as compression

DATA_DIR = Path(__file__).resolve().parent / 'data'
IMAGE_DIR = DATA_DIR / 'images'


def model_config(input_token_limit: int = 100) -> types.SimpleNamespace:
    """Return a minimal model config for compression tests."""
    return types.SimpleNamespace(
        provider='openai',
        name='test-model',
        input_token_limit=input_token_limit,
    )


def provider_config(provider: str) -> types.SimpleNamespace:
    """Return a minimal provider-specific model config for compression tests."""
    cfg = model_config()
    cfg.provider = provider
    return cfg


def test_compression_config_normalizes_options_and_rejects_invalid_values() -> None:
    """Test compression option normalization and validation."""
    config = compression.CompressionConfig(scope='Both', mode='Always', ratio='0.5', content_detection=False)

    assert config.scope == 'both'
    assert config.mode == 'always'
    assert config.ratio == 0.5
    assert config.content_detection is False
    assert compression.compression_config('text', 'auto', ratio='auto') == compression.CompressionConfig(
        scope='text',
        mode='auto',
        ratio='auto',
    )
    assert config.includes_text() is True
    assert config.includes_images() is True
    with pytest.raises(ValueError, match='compression_scope must be one of'):
        compression.CompressionConfig(scope='tables')
    with pytest.raises(ValueError, match='compression_mode must be one of'):
        compression.CompressionConfig(mode='sometimes')
    with pytest.raises(ValueError, match='compression_ratio must be'):
        compression.CompressionConfig(ratio=0)
    with pytest.raises(ValueError, match='compression_ratio must be'):
        compression.CompressionConfig(ratio=1.1)


def test_text_compression_decision_uses_scope_mode_and_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test text compression decisions across scopes, modes, and budgets."""
    cfg = model_config()
    monkeypatch.setattr(compression, '_request_token_budget', lambda prompt, model_config: 10)
    monkeypatch.setattr(compression, 'count_text_tokens', lambda text, model_config=None: len(text.split()))

    assert compression.should_compress_text('one two three', 'prompt', cfg, compression.CompressionConfig()) is False
    assert compression.should_compress_text(
        'one two three',
        'prompt',
        cfg,
        compression.CompressionConfig(scope='text', mode='always'),
    ) is True
    assert compression.should_compress_text(
        'one two three',
        'prompt',
        cfg,
        compression.CompressionConfig(scope='text', mode='auto'),
    ) is False
    assert compression.should_compress_text(
        ' '.join(str(index) for index in range(11)),
        'prompt',
        cfg,
        compression.CompressionConfig(scope='text', mode='auto'),
    ) is True


def test_request_token_budget_reserves_prompt_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that request budgets reserve prompt tokens."""
    calls = {}
    cfg = model_config()
    monkeypatch.setattr(compression, 'prompt_token_reserve', lambda prompt, model_config=None, buffer_tokens=500: calls.update({
        'prompt': prompt,
        'reserve_model_config': model_config,
        'buffer_tokens': buffer_tokens,
    }) or 25)
    monkeypatch.setattr(compression, 'usable_input_token_limit', lambda model_config=None, reserve_tokens=0: calls.update({
        'limit_model_config': model_config,
        'reserve_tokens': reserve_tokens,
    }) or 75)

    assert compression._request_token_budget('extract prompt', cfg) == 75
    assert calls == {
        'prompt': 'extract prompt',
        'reserve_model_config': cfg,
        'buffer_tokens': 500,
        'limit_model_config': cfg,
        'reserve_tokens': 25,
    }


def test_ideal_compression_ratio_uses_fixed_values_or_auto_budget() -> None:
    """Test fixed and automatically calculated compression ratios."""
    assert compression.ideal_compression_ratio(1000, 100, compression.CompressionConfig(ratio=0.4)) == 0.4
    assert compression.ideal_compression_ratio(0, 100, compression.CompressionConfig(ratio='auto')) == 1.0
    assert compression.ideal_compression_ratio(100, 200, compression.CompressionConfig(ratio='auto')) == 1.0
    assert compression.ideal_compression_ratio(200, 100, compression.CompressionConfig(ratio='auto')) == 0.475
    assert compression.ideal_compression_ratio(100000, 100, compression.CompressionConfig(ratio='auto')) == (
        compression.MIN_COMPRESSION_RATIO
    )


def test_image_compression_decision_uses_scope_mode_and_estimated_request_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test image compression decisions across scopes, modes, and budgets."""
    cfg = model_config()
    monkeypatch.setattr(compression, '_request_token_budget', lambda prompt, model_config: 1500)
    monkeypatch.setattr(compression, 'count_text_tokens', lambda text, model_config=None: len(text.split()))
    monkeypatch.setattr(compression, 'estimate_image_tokens', lambda image_paths, model_config=None: len(image_paths) * 1000)

    assert compression.should_compress_images(['image.png'], 'prompt', '', cfg, compression.CompressionConfig()) is False
    assert compression.should_compress_images(
        [],
        'prompt',
        '',
        cfg,
        compression.CompressionConfig(scope='images', mode='always'),
    ) is False
    assert compression.should_compress_images(
        ['image.png'],
        'prompt',
        '',
        cfg,
        compression.CompressionConfig(scope='images', mode='always'),
    ) is True
    assert compression.should_compress_images(
        ['image.png'],
        'prompt',
        'short context',
        cfg,
        compression.CompressionConfig(scope='images', mode='auto'),
    ) is False
    assert compression.should_compress_images(
        ['one.png', 'two.png'],
        'prompt',
        'short context',
        cfg,
        compression.CompressionConfig(scope='images', mode='auto'),
    ) is True


def test_estimate_image_tokens_uses_provider_specific_dimension_estimates() -> None:
    """Test provider-specific image token estimates."""
    small = str(IMAGE_DIR / '512x512.png')
    large = str(IMAGE_DIR / '1024x1024.png')
    missing = str(IMAGE_DIR / 'missing.png')

    assert compression.estimate_image_tokens([small], provider_config('openai')) == 255
    assert compression.estimate_image_tokens([large], provider_config('local')) == 765
    assert compression.estimate_image_tokens([small], provider_config('anthropic')) == 350
    assert compression.estimate_image_tokens([small], provider_config('other')) == 1000
    assert compression.estimate_image_tokens([missing], provider_config('openai')) == (
        compression.FALLBACK_IMAGE_TOKEN_ESTIMATE
    )


def test_image_size_reads_png_dimensions_without_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PNG header dimension parsing when Pillow is unavailable."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Block Pillow imports and delegate all others."""
        if name == 'PIL' or name.startswith('PIL.'):
            raise ImportError('missing pillow')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    assert compression._image_size(str(IMAGE_DIR / '12x34.png')) == (12, 34)
    assert compression._image_size(str(IMAGE_DIR / 'missing.png')) is None


def test_image_size_uses_pillow_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Pillow-backed image dimension reading."""

    class FakeImage:
        """Represent an image context manager with fixed dimensions."""

        size = (20, 30)

        def __enter__(self) -> FakeImage:
            """Return the fake image from its context manager."""
            return self

        def __exit__(self, *_: Any) -> bool:
            """Leave the fake image context without suppressing errors."""
            return False

    class FakeImageModule:
        """Provide the subset of ``PIL.Image`` used by compression."""

        @staticmethod
        def open(image_path: str) -> FakeImage:
            """Validate the path and return a fake image."""
            assert image_path == 'image.png'
            return FakeImage()

    fake_pil = types.ModuleType('PIL')
    fake_pil.Image = FakeImageModule
    monkeypatch.setitem(sys.modules, 'PIL', fake_pil)

    assert compression._image_size('image.png') == (20, 30)


def test_image_size_returns_none_for_readable_non_png_files(tmp_path: Path) -> None:
    """Test that readable non-PNG files have no inferred dimensions."""
    text_path = tmp_path / 'not-an-image.txt'
    text_path.write_text('not an image')

    assert compression._image_size(str(text_path)) is None


def test_compress_content_uses_headroom_universal_compressor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test integration with Headroom's universal compressor."""
    calls = {}

    class FakeUniversalCompressorConfig:
        """Capture universal compressor configuration options."""

        def __init__(self, **kwargs: Any) -> None:
            """Store configuration options for assertions."""
            self.kwargs = kwargs
            calls['config'] = kwargs

    class FakeUniversalCompressor:
        """Capture content passed to a universal compressor."""

        def __init__(self, config: FakeUniversalCompressorConfig) -> None:
            """Record the compressor configuration."""
            calls['compressor_config'] = config

        def compress(self, content: Any) -> types.SimpleNamespace:
            """Record content and return a compressed result shape."""
            calls['content'] = content
            return types.SimpleNamespace(compressed='compressed content')

    fake_headroom = types.ModuleType('headroom')
    fake_headroom_compression = types.ModuleType('headroom.compression')
    fake_headroom_compression.UniversalCompressor = FakeUniversalCompressor
    fake_headroom_compression.UniversalCompressorConfig = FakeUniversalCompressorConfig
    monkeypatch.setitem(sys.modules, 'headroom', fake_headroom)
    monkeypatch.setitem(sys.modules, 'headroom.compression', fake_headroom_compression)

    assert compression.compress_content(
        'full paper text',
        prompt='extract materials',
        compression_ratio=0.5,
        content_detection=False,
    ) == 'compressed content'
    assert calls['config'] == {
        'compression_ratio_target': 0.5,
        'use_magika': False,
        'use_entropy_preservation': True,
        'ccr_enabled': False,
    }
    assert calls['compressor_config'].kwargs == calls['config']
    assert calls['content'] == 'full paper text'


@pytest.mark.slow
@pytest.mark.filterwarnings('ignore:builtin type SwigPyPacked has no __module__ attribute:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:builtin type SwigPyObject has no __module__ attribute:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:builtin type swigvarlink has no __module__ attribute:DeprecationWarning')
def test_compress_content_uses_real_headroom_universal_compressor() -> None:
    """Test real Headroom compression through the project wrapper."""
    headroom_compression = pytest.importorskip('headroom.compression')
    if not hasattr(headroom_compression, 'UniversalCompressor'):
        pytest.skip('Headroom universal compressor is not installed.')
    text = (
        'Lithium solid electrolyte conductivity was measured at room temperature. '
        'The lithium solid electrolyte conductivity was reported as 1e-3 S cm^-1. '
        'The same lithium solid electrolyte sample was tested repeatedly under identical conditions. '
    ) * 6

    compressed = compression.compress_content(text, prompt='Extract lithium solid electrolyte materials data.')

    assert isinstance(compressed, str)
    assert compressed.strip()
    assert len(compressed) <= len(text)


def test_compressed_content_handles_common_result_shapes() -> None:
    """Test content extraction from common Headroom result shapes."""
    message_payload = [{'role': 'user', 'content': []}]
    assert compression._compressed_content('already compressed') == 'already compressed'
    assert compression._compressed_content(message_payload) == message_payload
    assert compression._compressed_content(types.SimpleNamespace(content='object content')) == 'object content'
    assert compression._compressed_content({'output': 'dict output'}) == 'dict output'
    assert compression._compressed_content(42) == '42'


def test_maybe_compress_text_returns_original_or_compressed_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test conditional text compression results."""
    cfg = model_config()
    policy = compression.CompressionConfig(scope='text')
    monkeypatch.setattr(compression, 'should_compress_text', lambda *args: False)

    assert compression.maybe_compress_text('paper text', 'prompt', cfg, policy) == 'paper text'

    monkeypatch.setattr(compression, 'should_compress_text', lambda *args: True)
    monkeypatch.setattr(
        compression,
        'count_text_tokens',
        lambda text, model_config=None: 400,
    )
    monkeypatch.setattr(compression, '_request_token_budget', lambda prompt, model_config: 200)
    monkeypatch.setattr(
        compression,
        'compress_content',
        lambda text, prompt='', compression_ratio=0.5, content_detection=True: (
            f'compressed {text} with {prompt} ratio={compression_ratio} detection={content_detection}'
        ),
    )
    assert compression.maybe_compress_text('paper text', 'prompt', cfg, policy) == (
        'compressed paper text with prompt ratio=0.475 detection=True'
    )


def test_compress_content_reports_missing_headroom_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the error reported when Headroom is unavailable."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Block Headroom imports and delegate all others."""
        if name == 'headroom' or name.startswith('headroom.'):
            raise ImportError('missing headroom')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    with pytest.raises(RuntimeError, match='Headroom universal compression requires'):
        compression.compress_content('paper text')


def test_maybe_compress_image_messages_returns_original_or_compressed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test conditional image message compression results."""
    messages = [{'role': 'user', 'content': []}]
    cfg = model_config()
    policy = compression.CompressionConfig(scope='images')
    monkeypatch.setattr(compression, 'should_compress_images', lambda *args: False)

    assert compression.maybe_compress_image_messages(messages, ['image.png'], 'prompt', '', cfg, policy) is messages
    assert compression.maybe_compress_image_messages(messages, ['image.png'], 'prompt', '', cfg, None) is messages

    monkeypatch.setattr(compression, 'should_compress_images', lambda *args: True)
    monkeypatch.setattr(compression, 'count_text_tokens', lambda text, model_config=None: 100)
    monkeypatch.setattr(compression, 'estimate_image_tokens', lambda image_paths, model_config=None: 300)
    monkeypatch.setattr(compression, '_request_token_budget', lambda prompt, model_config: 200)
    monkeypatch.setattr(compression, 'compress_content', lambda payload, prompt='',
                        compression_ratio=0.5, content_detection=True: [{
        'payload': payload,
        'prompt': prompt,
        'ratio': compression_ratio,
        'content_detection': content_detection,
    }])
    assert compression.maybe_compress_image_messages(messages, ['image.png'], 'prompt', '', cfg, policy) == [{
        'payload': messages,
        'prompt': 'prompt',
        'ratio': 0.475,
        'content_detection': True,
    }]
