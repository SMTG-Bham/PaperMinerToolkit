import builtins
from pathlib import Path
import sys
import types

import pytest

import paperscraper.compression as compression

DATA_DIR = Path(__file__).resolve().parent / 'data'
IMAGE_DIR = DATA_DIR / 'images'


def model_config(input_token_limit=100):
    """Return a minimal model config for compression tests."""
    return types.SimpleNamespace(
        provider='openai',
        name='test-model',
        input_token_limit=input_token_limit,
    )


def provider_config(provider):
    """Return a minimal provider-specific model config for compression tests."""
    cfg = model_config()
    cfg.provider = provider
    return cfg


def test_compression_config_normalizes_options_and_rejects_invalid_values():
    """
    Test compression configuration validation.

    This function performs the following steps:
    1. Builds a compression config with mixed-case options.
    2. Attempts to build configs with invalid scope and mode values.
    3. Reads scope membership helpers from the valid config.

    Asserts:
        - Scope and mode values are normalized.
        - Compression ratio and content detection options are normalized.
        - Text and image membership helpers reflect the selected scope.
        - Invalid options raise helpful `ValueError` messages.
    """
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


def test_text_compression_decision_uses_scope_mode_and_token_budget(monkeypatch):
    """
    Test text compression policy decisions.

    This function performs the following steps:
    1. Builds compression configs for disabled, always-on, and automatic text compression.
    2. Replaces token counting and budget helpers with deterministic local fakes.
    3. Checks decisions for short and long text inputs.

    Asserts:
        - Disabled text compression returns `False`.
        - Always-on text compression returns `True` for non-empty text.
        - Automatic text compression compares text tokens against the usable input budget.
    """
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


def test_request_token_budget_reserves_prompt_tokens(monkeypatch):
    """
    Test compression request budget calculation.

    This function performs the following steps:
    1. Replaces prompt reserve and usable budget helpers with deterministic local fakes.
    2. Calls `_request_token_budget` with a prompt and model config.
    3. Reads the helper calls.

    Asserts:
        - Prompt reserve is calculated with the configured model.
        - The usable budget receives the reserve token count.
        - The usable budget value is returned.
    """
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


def test_ideal_compression_ratio_uses_fixed_values_or_auto_budget():
    """
    Test compression ratio selection.

    This function performs the following steps:
    1. Builds compression configs with fixed and automatic ratios.
    2. Calculates ratios for inputs below, near, and far above the token budget.
    3. Reads the returned compression ratios.

    Asserts:
        - Fixed ratios pass through unchanged.
        - Automatic ratios keep content unchanged when the request fits.
        - Automatic ratios scale oversized requests to the usable budget with a safety margin.
        - Automatic ratios do not go below the configured minimum.
    """
    assert compression.ideal_compression_ratio(1000, 100, compression.CompressionConfig(ratio=0.4)) == 0.4
    assert compression.ideal_compression_ratio(0, 100, compression.CompressionConfig(ratio='auto')) == 1.0
    assert compression.ideal_compression_ratio(100, 200, compression.CompressionConfig(ratio='auto')) == 1.0
    assert compression.ideal_compression_ratio(200, 100, compression.CompressionConfig(ratio='auto')) == 0.475
    assert compression.ideal_compression_ratio(100000, 100, compression.CompressionConfig(ratio='auto')) == (
        compression.MIN_COMPRESSION_RATIO
    )


def test_image_compression_decision_uses_scope_mode_and_estimated_request_tokens(monkeypatch):
    """
    Test image compression policy decisions.

    This function performs the following steps:
    1. Builds compression configs for disabled, always-on, and automatic image compression.
    2. Replaces token counting and budget helpers with deterministic local fakes.
    3. Checks decisions for empty, small, and large image requests.

    Asserts:
        - Disabled or empty image compression requests return `False`.
        - Always-on image compression returns `True` when image paths are present.
        - Automatic image compression estimates image and text context tokens against the usable input budget.
    """
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


def test_estimate_image_tokens_uses_provider_specific_dimension_estimates():
    """
    Test provider-specific image token estimates.

    This function performs the following steps:
    1. Reads image dimensions from PNG fixtures in `tests/data/images`.
    2. Estimates image tokens for OpenAI, Anthropic, unknown, and unreadable images.
    3. Compares the estimates to provider-specific formulas.

    Asserts:
        - OpenAI-style providers estimate one base cost plus 512px image tiles.
        - Anthropic estimates from image pixel area.
        - Unknown providers use the more conservative dimension estimate.
        - Unreadable images use the fallback estimate.
    """
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


def test_image_size_reads_dimensions_from_local_image():
    """
    Test local image dimension reading.

    This function performs the following steps:
    1. Reads a PNG fixture from `tests/data/images`.
    2. Reads the image dimensions with `_image_size`.
    3. Reads dimensions from a missing image path.

    Asserts:
        - Local image dimensions are returned as a `(width, height)` tuple.
        - Unreadable paths return `None`.
    """
    assert compression._image_size(str(IMAGE_DIR / '12x34.png')) == (12, 34)
    assert compression._image_size(str(IMAGE_DIR / 'missing.png')) is None


def test_image_size_uses_pillow_when_available(monkeypatch):
    """
    Test Pillow-backed image dimension reading.

    This function performs the following steps:
    1. Installs a fake `PIL.Image` module in `sys.modules`.
    2. Reads dimensions from a fake image path.
    3. Checks the returned image size.

    Asserts:
        - The Pillow image reader path returns the image object's size.
    """
    class FakeImage:
        size = (20, 30)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class FakeImageModule:
        @staticmethod
        def open(image_path):
            assert image_path == 'image.png'
            return FakeImage()

    fake_pil = types.ModuleType('PIL')
    fake_pil.Image = FakeImageModule
    monkeypatch.setitem(sys.modules, 'PIL', fake_pil)

    assert compression._image_size('image.png') == (20, 30)


def test_image_size_returns_none_for_readable_non_png_files(tmp_path):
    """
    Test image dimension handling for readable unsupported files.

    This function performs the following steps:
    1. Writes a readable non-PNG file.
    2. Reads dimensions with `_image_size`.
    3. Checks the returned value.

    Asserts:
        - Readable files without supported image headers return `None`.
    """
    text_path = tmp_path / 'not-an-image.txt'
    text_path.write_text('not an image')

    assert compression._image_size(str(text_path)) is None


def test_compress_content_uses_headroom_universal_compressor(monkeypatch):
    """
    Test Headroom universal compressor integration.

    This function performs the following steps:
    1. Installs fake `headroom` and `headroom.compression` modules in `sys.modules`.
    2. Calls `compress_content` with a prompt context.
    3. Reads the fake compressor calls.

    Asserts:
        - Headroom's `UniversalCompressorConfig` is built with PaperScraper options.
        - Headroom's `UniversalCompressor` is instantiated and called.
        - The compressed content from the Headroom result is returned.
    """
    calls = {}

    class FakeUniversalCompressorConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls['config'] = kwargs

    class FakeUniversalCompressor:
        def __init__(self, config):
            calls['compressor_config'] = config

        def compress(self, content):
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
def test_compress_content_uses_real_headroom_universal_compressor():
    """
    Test real Headroom universal compression through the PaperScraper wrapper.

    This function performs the following steps:
    1. Skips the test when Headroom's universal compressor is not installed.
    2. Compresses a small repetitive paper-like text with the real Headroom compressor.
    3. Reads the returned compressed text.

    Asserts:
        - The real Headroom integration returns a non-empty string.
        - The returned value is not longer than the original input text.
    """
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


def test_compressed_content_handles_common_result_shapes():
    """
    Test compressed content extraction from Headroom result shapes.

    This function performs the following steps:
    1. Passes string, list, object, dictionary, and fallback values to `_compressed_content`.
    2. Reads the extracted content value from each result shape.
    3. Compares the outputs to expected strings.

    Asserts:
        - Strings and lists are returned unchanged.
        - Object and dictionary compressed/content fields are preferred.
        - Unknown result shapes are converted to strings.
    """
    message_payload = [{'role': 'user', 'content': []}]
    assert compression._compressed_content('already compressed') == 'already compressed'
    assert compression._compressed_content(message_payload) == message_payload
    assert compression._compressed_content(types.SimpleNamespace(content='object content')) == 'object content'
    assert compression._compressed_content({'output': 'dict output'}) == 'dict output'
    assert compression._compressed_content(42) == '42'


def test_maybe_compress_text_returns_original_or_compressed_value(monkeypatch):
    """
    Test conditional text compression wrapper.

    This function performs the following steps:
    1. Replaces the text compression decision helper with a fake returning `False`.
    2. Calls `maybe_compress_text`.
    3. Replaces the decision helper with a fake returning `True` and compression with a fake output.

    Asserts:
        - Text is returned unchanged when compression is not needed.
        - Compressed text is returned when compression is requested.
    """
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


def test_compress_content_reports_missing_headroom_package(monkeypatch):
    """
    Test missing Headroom universal dependency errors.

    This function performs the following steps:
    1. Replaces Python imports with a fake that raises for `headroom`.
    2. Calls `compress_content`.
    3. Captures the expected exception.

    Asserts:
        - Missing Headroom universal dependencies raise `RuntimeError`.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'headroom':
            raise ImportError('missing headroom')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    with pytest.raises(RuntimeError, match='Headroom universal compression requires'):
        compression.compress_content('paper text')


def test_maybe_compress_image_messages_returns_original_or_compressed_payload(monkeypatch):
    """
    Test conditional image compression wrapper.

    This function performs the following steps:
    1. Replaces the image compression decision helper with a fake returning `False`.
    2. Calls `maybe_compress_image_messages`.
    3. Replaces the decision helper with a fake returning `True` and compression with a fake payload.

    Asserts:
        - Messages are returned unchanged when compression is not needed.
        - Compressed messages are returned when compression is requested.
    """
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
