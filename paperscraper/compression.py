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
    """Store Headroom compression settings for a scrape run.

    Attributes
    ----------
    scope : str
        Content to compress: ``"none"``, ``"text"``, ``"images"``, or
        ``"both"``.
    mode : str
        Compression policy, either ``"auto"`` or ``"always"``.
    ratio : float or str
        Target compression ratio, or ``"auto"`` to derive one from the token
        budget.
    content_detection : bool
        Whether Headroom should detect content types with Magika.
    """

    scope: str = 'none'
    mode: str = 'auto'
    ratio: float | str = 'auto'
    content_detection: bool = True

    def __post_init__(self):
        """Normalize and validate compression options after initialization.

        Raises
        ------
        ValueError
            If the scope, mode, or compression ratio is invalid.
        """
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
        """Return whether paper text or text context should be compressed.

        Returns
        -------
        bool
            Whether the configured scope includes text.
        """
        return self.scope in {'text', 'both'}

    def includes_images(self):
        """Return whether image payloads should be compressed.

        Returns
        -------
        bool
            Whether the configured scope includes images.
        """
        return self.scope in {'images', 'both'}


def compression_config(scope: str = 'none',
                       mode: str = 'auto',
                       ratio: float | str = 'auto',
                       content_detection: bool = True) -> CompressionConfig:
    """Build a normalized compression configuration.

    Parameters
    ----------
    scope : str, optional
        Content to compress.
    mode : str, optional
        Compression policy.
    ratio : float or str, optional
        Target compression ratio, or ``"auto"``.
    content_detection : bool, optional
        Whether to enable Headroom content detection.

    Returns
    -------
    CompressionConfig
        Validated compression settings.

    Raises
    ------
    ValueError
        If any option is invalid.
    """
    return CompressionConfig(scope=scope, mode=mode, ratio=ratio, content_detection=content_detection)


def _request_token_budget(prompt, model_config):
    """Calculate the input budget remaining after prompt reservation.

    Parameters
    ----------
    prompt : str
        Prompt that shares the model context window with the content.
    model_config : object
        Model configuration used to determine token limits.

    Returns
    -------
    int
        Number of tokens available for content.
    """
    reserve_tokens = prompt_token_reserve(prompt, model_config=model_config, buffer_tokens=500)
    return usable_input_token_limit(model_config, reserve_tokens=reserve_tokens)


def ideal_compression_ratio(input_tokens, token_budget, config: CompressionConfig):
    """Calculate the target compression ratio for a request.

    Parameters
    ----------
    input_tokens : int
        Estimated number of input tokens.
    token_budget : int
        Number of tokens available for input content.
    config : CompressionConfig
        Compression settings.

    Returns
    -------
    float
        Configured or automatically calculated compression ratio.
    """
    if config.ratio != 'auto':
        return config.ratio
    if input_tokens <= 0:
        return 1.0
    ratio = (token_budget / input_tokens) * AUTO_RATIO_SAFETY_FACTOR
    return max(MIN_COMPRESSION_RATIO, min(1.0, ratio))


def should_compress_text(text, prompt, model_config, config: CompressionConfig):
    """Determine whether text should be compressed before analysis.

    Parameters
    ----------
    text : object
        Candidate text content.
    prompt : str
        Prompt that accompanies the content.
    model_config : object
        Model configuration used for token estimation.
    config : CompressionConfig
        Compression settings.

    Returns
    -------
    bool
        Whether the text should be compressed.
    """
    if not config.includes_text() or not isinstance(text, str) or text == '':
        return False
    if config.mode == 'always':
        return True
    return count_text_tokens(text, model_config=model_config) > _request_token_budget(prompt, model_config)


def should_compress_images(image_paths, prompt, context, model_config, config: CompressionConfig):
    """Determine whether an image request should be compressed.

    Parameters
    ----------
    image_paths : sequence of str
        Local image paths included in the request.
    prompt : str
        Prompt that accompanies the images.
    context : str or None
        Optional text context included in the request.
    model_config : object
        Model configuration used for token estimation.
    config : CompressionConfig
        Compression settings.

    Returns
    -------
    bool
        Whether the image request should be compressed.
    """
    if not config.includes_images() or not image_paths:
        return False
    if config.mode == 'always':
        return True
    text_tokens = count_text_tokens(context or '', model_config=model_config)
    image_tokens = estimate_image_tokens(image_paths, model_config)
    return text_tokens + image_tokens > _request_token_budget(prompt, model_config)


def _image_size(image_path):
    """Read the dimensions of a local image.

    Parameters
    ----------
    image_path : str or path-like
        Path to an image file.

    Returns
    -------
    tuple of int or None
        Image width and height, or ``None`` when dimensions cannot be read.
    """
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
    """Estimate image input tokens for a model provider.

    Parameters
    ----------
    image_paths : sequence of str
        Local image paths included in the request.
    model_config : object, optional
        Model configuration identifying the provider.

    Returns
    -------
    int
        Estimated number of image input tokens.
    """
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
    """Extract content from common Headroom result shapes.

    Parameters
    ----------
    result : object
        Value returned by a Headroom compressor.

    Returns
    -------
    object
        Extracted compressed content, or the string form of an unknown result.
    """
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
    """Create a Headroom compressor with safe defaults.

    Parameters
    ----------
    compression_ratio : float, optional
        Target output-to-input size ratio.
    content_detection : bool, optional
        Whether Headroom should detect content types with Magika.

    Returns
    -------
    headroom.compression.UniversalCompressor
        Configured universal compressor.

    Raises
    ------
    RuntimeError
        If the optional ``headroom-ai`` package is unavailable.
    """
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
    """Compress content with Headroom's universal compressor.

    Parameters
    ----------
    content : object
        Text or provider-shaped message payload to compress.
    prompt : str, optional
        Prompt associated with the content. Reserved for compressor adapters.
    compression_ratio : float, optional
        Target output-to-input size ratio.
    content_detection : bool, optional
        Whether Headroom should detect content types with Magika.

    Returns
    -------
    object
        Compressed content, or the original content when compression is empty.

    Raises
    ------
    RuntimeError
        If the optional ``headroom-ai`` package is unavailable.
    """
    compressor = _universal_compressor(
        compression_ratio=compression_ratio,
        content_detection=content_detection,
    )
    result = compressor.compress(content)
    compressed = _compressed_content(result)
    return compressed if compressed else content


def maybe_compress_text(text, prompt, model_config, config: CompressionConfig):
    """Apply compression to text when required by the configured policy.

    Parameters
    ----------
    text : str
        Text to evaluate and optionally compress.
    prompt : str
        Prompt that accompanies the text.
    model_config : object
        Model configuration used for token estimation.
    config : CompressionConfig
        Compression settings.

    Returns
    -------
    object
        Compressed content or the original text.
    """
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
    """Apply compression to image messages when required by policy.

    Parameters
    ----------
    messages : list
        Provider-shaped messages containing image content.
    image_paths : sequence of str
        Local paths for images represented in ``messages``.
    prompt : str
        Prompt that accompanies the images.
    context : str or None
        Optional text context included in the request.
    model_config : object
        Model configuration used for token estimation.
    config : CompressionConfig or None
        Compression settings, or ``None`` to disable compression.

    Returns
    -------
    object
        Compressed messages or the original message list.
    """
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
