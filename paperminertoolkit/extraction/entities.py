"""Extract labeled spans with locally executed token-classification models.

Fine-tuned Hugging Face BERT-family checkpoints supply the entity vocabulary.
Base encoders and masked-language models need token-classification fine-tuning
before they can be used here. Model loading is deferred until the first input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict


class Entity(TypedDict):
    """An entity with an exact source span and a model confidence score."""

    label: str
    text: str
    score: float
    start: int
    end: int


@dataclass(frozen=True)
class EntityExtractionConfig:
    """Configure a local Hugging Face token-classification checkpoint.

    Parameters
    ----------
    model : str
        Fine-tuned model repository ID or local checkpoint directory.
    device : str, default='cpu'
        PyTorch inference device, for example ``'cpu'`` or ``'cuda:0'``.
    batch_size : int, default=8
        Number of token windows processed per inference batch.
    stride : int, default=64
        Number of overlapping content tokens between consecutive windows.
    max_length : int or None, default=None
        Window length including special tokens. The smallest finite model or
        tokenizer limit is used by default; explicit values may lower it.
    revision : str or None, default=None
        Optional Hugging Face revision, tag, or commit ID.
    cache_dir : str or None, default=None
        Optional directory for downloaded model and tokenizer files.
    local_files_only : bool, default=False
        Require all model files to be available locally.
    """

    model: str
    device: str = 'cpu'
    batch_size: int = 8
    stride: int = 64
    max_length: int | None = None
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        """Reject invalid configuration before model files are accessed."""
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('model must identify a fine-tuned token-classification checkpoint.')
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError('device must be a non-empty PyTorch device string.')
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        if isinstance(self.stride, bool) or not isinstance(self.stride, int) or self.stride < 0:
            raise ValueError('stride must be a non-negative integer.')
        if self.max_length is not None and (
            isinstance(self.max_length, bool)
            or not isinstance(self.max_length, int)
            or self.max_length < 1
        ):
            raise ValueError('max_length must be a positive integer or None.')


def _load_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    """Load optional local inference dependencies with installation guidance.

    Returns
    -------
    tuple
        PyTorch, automatic config/tokenizer/model loaders, and pipeline factory.

    Raises
    ------
    ImportError
        If the local inference dependencies cannot be imported.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError as error:
        raise ImportError(
            'BERT entity extraction requires the local inference dependencies. '
            'Install them with: pip install "paperminertoolkit[bert]"'
        ) from error
    return torch, AutoConfig, AutoTokenizer, AutoModelForTokenClassification, pipeline


def _window_length(config: EntityExtractionConfig, tokenizer: Any, model_config: Any) -> int:
    """Resolve a finite window size and validate overlap against content space.

    Parameters
    ----------
    config : EntityExtractionConfig
        Requested inference configuration.
    tokenizer : Any
        Loaded fast tokenizer.
    model_config : Any
        Loaded Hugging Face checkpoint configuration.

    Returns
    -------
    int
        Safe token window size including special tokens.

    Raises
    ------
    ValueError
        If a finite safe window cannot be selected or the overlap is too large.
    """
    limits = [
        value for value in (
            getattr(tokenizer, 'model_max_length', None),
            getattr(model_config, 'max_position_embeddings', None),
        )
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 1_000_000_000
    ]
    limit = min(limits) if limits else None
    if config.max_length is not None:
        if limit is not None and config.max_length > limit:
            raise ValueError(f'max_length={config.max_length} exceeds the checkpoint token limit of {limit}.')
        length = config.max_length
    elif limit is not None:
        length = limit
    else:
        raise ValueError('The checkpoint has no finite token limit; specify max_length explicitly.')
    content_length = length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_length <= 0:
        raise ValueError('max_length must leave room for content after tokenizer special tokens.')
    if config.stride >= content_length:
        raise ValueError(
            f'stride={config.stride} must be smaller than the content window of {content_length} tokens; '
            'reduce stride or increase max_length within the checkpoint token limit.'
        )
    return length


class TransformerEntityExtractor:
    """Run a fine-tuned token classifier and retain exact source offsets.

    Long inputs are processed in overlapping tokenizer windows. Hugging Face's
    ``simple`` aggregation joins compatible BIO labels and resolves overlapping
    predictions across windows. Labels come from the checkpoint's ``id2label``
    mapping; a checkpoint using generic ``LABEL_0`` names retains those names.

    Parameters
    ----------
    config : EntityExtractionConfig
        Local model and inference settings.
    """

    def __init__(self, config: EntityExtractionConfig) -> None:
        """Store settings without importing PyTorch or downloading a model.

        Parameters
        ----------
        config : EntityExtractionConfig
            Model and inference settings.
        """
        self.config = config
        self._pipeline: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        """Initialize and cache a validated token-classification pipeline.

        Raises
        ------
        ValueError
            If the checkpoint is a base model, has uninitialized weights,
            requires a slow tokenizer, or cannot support the selected windows.
        ImportError
            If optional local inference dependencies are unavailable.
        """
        if self._pipeline is not None:
            return
        torch, auto_config, auto_tokenizer, auto_model, pipeline = _load_dependencies()
        options = {
            'revision': self.config.revision,
            'cache_dir': self.config.cache_dir,
            'local_files_only': self.config.local_files_only,
            'trust_remote_code': False,
        }
        checkpoint_config = auto_config.from_pretrained(self.config.model, **options)
        architectures = getattr(checkpoint_config, 'architectures', None) or []
        if architectures and not any(name.endswith('ForTokenClassification') for name in architectures):
            raise ValueError(
                f'Checkpoint "{self.config.model}" declares {", ".join(architectures)}, '
                'not a fine-tuned token-classification model. MatSciBERT and polyBERT base checkpoints '
                'must first be fine-tuned on labeled entities; select the saved fine-tuned checkpoint.'
            )
        tokenizer = auto_tokenizer.from_pretrained(self.config.model, use_fast=True, **options)
        if not tokenizer.is_fast:
            raise ValueError('Entity extraction requires a fast tokenizer for exact offsets and long-text windows.')
        tokenizer.model_max_length = _window_length(self.config, tokenizer, checkpoint_config)
        model, loading_info = auto_model.from_pretrained(
            self.config.model,
            config=checkpoint_config,
            output_loading_info=True,
            **options,
        )
        if loading_info.get('missing_keys') or loading_info.get('mismatched_keys') or loading_info.get('error_msgs'):
            raise ValueError(
                f'Checkpoint "{self.config.model}" has missing or incompatible token-classifier weights. '
                'Use a complete saved model fine-tuned for token classification; randomly initialized '
                'entity predictions are not supported.'
            )
        model.eval()
        self._pipeline = pipeline(
            'token-classification',
            model=model,
            tokenizer=tokenizer,
            device=self.config.device,
            aggregation_strategy='simple',
            stride=self.config.stride,
        )
        self._torch = torch

    def extract(self, texts: Sequence[str] | str) -> list[list[Entity]]:
        """Extract entities from texts without changing their case or content.

        Parameters
        ----------
        texts : sequence of str or str
            Texts to classify, or a single text. Character offsets are relative
            to each original input, with an inclusive start and exclusive end.

        Returns
        -------
        list of list of Entity
            One list per input, including for a single string. Each entity
            contains ``label``, exact source ``text``, ``score``, ``start``,
            and ``end``. Empty or whitespace-only inputs produce empty lists.

        Raises
        ------
        TypeError
            If an input is not a string or a sequence of strings.
        ValueError
            If model configuration, weights, or returned spans are invalid.
        ImportError
            If local inference dependencies are unavailable.
        """
        if isinstance(texts, str):
            inputs = [texts]
        elif isinstance(texts, Sequence):
            inputs = list(texts)
        else:
            raise TypeError('texts must be a string or a sequence of strings.')
        if any(not isinstance(text, str) for text in inputs):
            raise TypeError('Every entity extraction input must be a string.')
        results: list[list[Entity]] = [[] for _ in inputs]
        positions = [index for index, text in enumerate(inputs) if text.strip()]
        if not positions:
            return results
        self._load()
        with self._torch.inference_mode():
            predictions = self._pipeline(
                [inputs[index] for index in positions],
                batch_size=self.config.batch_size,
                num_workers=0,
            )
        if len(predictions) != len(positions):
            raise ValueError('The token classifier returned an unexpected number of input results.')
        for position, entities in zip(positions, predictions, strict=True):
            text = inputs[position]
            for entity in entities:
                start, end = int(entity['start']), int(entity['end'])
                score = float(entity['score'])
                if not 0 <= start < end <= len(text) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('The token classifier returned an invalid entity span or confidence score.')
                results[position].append({
                    'label': str(entity['entity_group']),
                    'text': text[start:end],
                    'score': score,
                    'start': start,
                    'end': end,
                })
        return results
