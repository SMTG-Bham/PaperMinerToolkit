"""Configure and apply optional Headroom context compression.

This module keeps Headroom integration behind small helper functions so scrape
and model code can decide when to compress without depending on Headroom's
objects directly.
"""

import math
from dataclasses import dataclass

from paperscraper.tokenizer import count_text_tokens, prompt_token_reserve, usable_input_token_limit

COMPRESSION_SCOPES = {'none', 'text', 'images', 'both'}
COMPRESSION_MODES = {'auto', 'always'}
FALLBACK_IMAGE_TOKEN_ESTIMATE = 1000
MIN_COMPRESSION_RATIO = 0.05
AUTO_RATIO_SAFETY_FACTOR = 0.95


@dataclass(frozen=True)
class CompressionConfig:
    """User-selected Headroom compression settings for a scrape run."""

    scope: str = 'none'
    mode: str = 'auto'
    ratio: float | str = 'auto'
    content_detection: bool = True

    def __post_init__(self):
        """Normalize and validate compression options after initialization."""
        scope = (self.scope or 'none').lower()
        mode = (self.mode or 'auto').lower()
        ratio = self.ratio
        if scope not in COMPRESSION_SCOPES:
            raise ValueError(f'compression_scope must be one of: {", ".join(sorted(COMPRESSION_SCOPES))}')
        if mode not in COMPRESSION_MODES:
            raise ValueError(f'compression_mode must be one of: {", ".join(sorted(COMPRESSION_MODES))}')
        if isinstance(ratio, str):
            ratio = ratio.strip().lower()
            if ratio != 'auto':
                ratio = float(ratio)
        if ratio != 'auto':
            ratio = float(ratio)
            if ratio <= 0 or ratio > 1:
                raise ValueError('compression_ratio must be "auto" or a number greater than 0 and less than or equal to 1.')
        object.__setattr__(self, 'scope', scope)
        object.__setattr__(self, 'mode', mode)
        object.__setattr__(self, 'ratio', ratio)
        object.__setattr__(self, 'content_detection', bool(self.content_detection))

    def includes_text(self):
        """Return whether paper text or text context should be compressed."""
        return self.scope in {'text', 'both'}

    def includes_images(self):
        """Return whether image payloads should be compressed."""
        return self.scope in {'images', 'both'}


def compression_config(scope: str = 'none',
                       mode: str = 'auto',
                       ratio: float | str = 'auto',
                       content_detection: bool = True) -> CompressionConfig:
    """Build a normalized compression config from CLI or function options."""
    return CompressionConfig(scope=scope, mode=mode, ratio=ratio, content_detection=content_detection)


def _request_token_budget(prompt, model_config):
    """Return the usable input budget after reserving space for the prompt."""
    reserve_tokens = prompt_token_reserve(prompt, model_config=model_config, buffer_tokens=500)
    return usable_input_token_limit(model_config, reserve_tokens=reserve_tokens)


def ideal_compression_ratio(input_tokens, token_budget, config: CompressionConfig):
    """Return the target compression ratio for an estimated request size."""
    if config.ratio != 'auto':
        return config.ratio
    if input_tokens <= 0:
        return 1.0
    ratio = (token_budget / input_tokens) * AUTO_RATIO_SAFETY_FACTOR
    return max(MIN_COMPRESSION_RATIO, min(1.0, ratio))


def should_compress_text(text, prompt, model_config, config: CompressionConfig):
    """Decide whether a text input should be compressed before model analysis."""
    if not config.includes_text() or not isinstance(text, str) or text == '':
        return False
    if config.mode == 'always':
        return True
    return count_text_tokens(text, model_config=model_config) > _request_token_budget(prompt, model_config)


def should_compress_images(image_paths, prompt, context, model_config, config: CompressionConfig):
    """Decide whether an image request should be compressed before model analysis."""
    if not config.includes_images() or not image_paths:
        return False
    if config.mode == 'always':
        return True
    text_tokens = count_text_tokens(context or '', model_config=model_config)
    image_tokens = estimate_image_tokens(image_paths, model_config)
    return text_tokens + image_tokens > _request_token_budget(prompt, model_config)


def _image_size(image_path):
    """Return image dimensions for a local file or ``None`` when unreadable."""
    try:
        from PIL import Image
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        pass
    try:
        with open(image_path, 'rb') as image_file:
            header = image_file.read(24)
        if header.startswith(b'\x89PNG\r\n\x1a\n') and header[12:16] == b'IHDR':
            return int.from_bytes(header[16:20], 'big'), int.from_bytes(header[20:24], 'big')
    except Exception:
        return None
    return None


def estimate_image_tokens(image_paths, model_config=None):
    """Estimate vision-model input tokens from image dimensions and provider."""
    provider = str(getattr(model_config, 'provider', '') or '').lower().replace('_', '-')
    total = 0
    for image_path in image_paths or []:
        size = _image_size(image_path)
        if not size:
            total += FALLBACK_IMAGE_TOKEN_ESTIMATE
            continue
        width, height = size
        if provider == 'anthropic':
            total += max(1, math.ceil(width * height / 750))
        elif provider in {'openai', 'local'}:
            tiles = math.ceil(width / 512) * math.ceil(height / 512)
            total += 85 + 170 * tiles
        else:
            total += max(FALLBACK_IMAGE_TOKEN_ESTIMATE, math.ceil(width * height / 750))
    return total


def _compressed_content(result):
    """Extract compressed content from common Headroom result shapes."""
    if isinstance(result, (str, list, tuple)):
        return result
    for name in ['compressed', 'content', 'text', 'output']:
        value = getattr(result, name, None)
        if value is not None:
            return value
    if isinstance(result, dict):
        for name in ['compressed', 'content', 'text', 'output']:
            value = result.get(name)
            if value is not None:
                return value
    return str(result)


def _universal_compressor(compression_ratio=0.5, content_detection=True):
    """Create a Headroom universal compressor with PaperScraper safety defaults."""
    try:
        from headroom.compression import UniversalCompressor, UniversalCompressorConfig
    except ImportError as e:
        raise RuntimeError('Headroom universal compression requires the headroom-ai package.') from e
    config = UniversalCompressorConfig(
        compression_ratio_target=compression_ratio,
        use_magika=content_detection,
        use_entropy_preservation=True,
        ccr_enabled=False,
    )
    return UniversalCompressor(config)


def compress_content(content, prompt='', compression_ratio=0.5, content_detection=True):
    """Compress text or provider-shaped message payloads with Headroom's universal compressor."""
    compressor = _universal_compressor(
        compression_ratio=compression_ratio,
        content_detection=content_detection,
    )
    result = compressor.compress(content)
    compressed = _compressed_content(result)
    return compressed if compressed else content


def maybe_compress_text(text, prompt, model_config, config: CompressionConfig):
    """Compress text only when the configured compression policy asks for it."""
    if should_compress_text(text, prompt, model_config, config):
        input_tokens = count_text_tokens(text, model_config=model_config)
        token_budget = _request_token_budget(prompt, model_config)
        ratio = ideal_compression_ratio(input_tokens, token_budget, config)
        return compress_content(text,
                                prompt=prompt,
                                compression_ratio=ratio,
                                content_detection=config.content_detection)
    return text


def maybe_compress_image_messages(messages,
                                  image_paths,
                                  prompt,
                                  context,
                                  model_config,
                                  config: CompressionConfig | None):
    """Compress image messages only when the configured policy asks for it."""
    if config is None or not should_compress_images(image_paths, prompt, context, model_config, config):
        return messages
    input_tokens = count_text_tokens(context or '', model_config=model_config) + estimate_image_tokens(
        image_paths,
        model_config,
    )
    token_budget = _request_token_budget(prompt, model_config)
    ratio = ideal_compression_ratio(input_tokens, token_budget, config)
    return compress_content(messages,
                            prompt=prompt,
                            compression_ratio=ratio,
                            content_detection=config.content_detection)
