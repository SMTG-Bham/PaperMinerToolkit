"""Test local token-classification inference and exact source entity spans."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import paperminertoolkit.extraction.entities as entities


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install an offline backend that records local model-loading arguments."""
    checkpoint = SimpleNamespace(architectures=['BertForTokenClassification'], max_position_embeddings=512)
    tokenizer = SimpleNamespace(is_fast=True, model_max_length=512, num_special_tokens_to_add=lambda pair: 2)
    model = SimpleNamespace(eval=Mock())
    inference = Mock(return_value=[[]])
    torch = SimpleNamespace(inference_mode=Mock(side_effect=nullcontext))
    config_loader = SimpleNamespace(from_pretrained=Mock(return_value=checkpoint))
    tokenizer_loader = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    model_loader = SimpleNamespace(from_pretrained=Mock(return_value=(model, {})))
    factory = Mock(return_value=inference)
    dependencies = Mock(return_value=(torch, config_loader, tokenizer_loader, model_loader, factory))
    monkeypatch.setattr(entities, '_load_dependencies', dependencies)
    return SimpleNamespace(
        checkpoint=checkpoint, tokenizer=tokenizer, model=model, inference=inference,
        torch=torch, config_loader=config_loader, tokenizer_loader=tokenizer_loader,
        model_loader=model_loader, factory=factory, dependencies=dependencies,
    )


def test_loading_is_lazy_and_empty_inputs_do_not_load(backend: SimpleNamespace) -> None:
    """Avoid model downloads for construction and empty input batches."""
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/ner'))
    assert extractor.extract([]) == []
    assert extractor.extract(['', ' \n']) == [[], []]
    backend.dependencies.assert_not_called()


def test_extract_preserves_source_spans_and_reuses_model(backend: SimpleNamespace) -> None:
    """Use source character slices instead of normalized tokenizer words."""
    text = '  Fe₂O₃ and [*]CC[*]'
    backend.inference.return_value = [[{
        'entity_group': 'MATERIAL', 'word': 'wrong token text', 'score': 0.9, 'start': 2, 'end': 7,
    }]]
    config = entities.EntityExtractionConfig(
        'test/ner', batch_size=3, revision='pinned', cache_dir='/tmp/model-cache', local_files_only=True,
    )
    extractor = entities.TransformerEntityExtractor(config)
    expected = [{'label': 'MATERIAL', 'text': 'Fe₂O₃', 'score': 0.9, 'start': 2, 'end': 7}]
    assert extractor.extract(['', text, ' ']) == [[], expected, []]
    assert extractor.extract(text) == [expected]
    backend.inference.assert_called_with([text], batch_size=3, num_workers=0)
    backend.dependencies.assert_called_once()
    backend.model.eval.assert_called_once()
    assert backend.torch.inference_mode.call_count == 2
    backend.config_loader.from_pretrained.assert_called_once_with(
        'test/ner', revision='pinned', cache_dir='/tmp/model-cache', local_files_only=True, trust_remote_code=False,
    )
    backend.factory.assert_called_once_with(
        'token-classification', model=backend.model, tokenizer=backend.tokenizer,
        device='cpu', aggregation_strategy='simple', stride=64,
    )


@pytest.mark.parametrize('architecture', ['BertForMaskedLM', 'BertModel', 'DebertaV2Model'])
def test_base_checkpoints_rejected_before_weights(backend: SimpleNamespace, architecture: str) -> None:
    """Prevent arbitrary untrained classification heads on base checkpoints."""
    backend.checkpoint.architectures = [architecture]
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/base'))
    with pytest.raises(ValueError, match='fine-tuned token-classification'):
        extractor.extract('Fe')
    backend.tokenizer_loader.from_pretrained.assert_not_called()
    backend.model_loader.from_pretrained.assert_not_called()


@pytest.mark.parametrize('loading_info', [
    {'missing_keys': ['classifier.weight', 'classifier.bias']},
    {'mismatched_keys': [('classifier.weight', (2, 3), (4, 3))]},
    {'error_msgs': ['invalid checkpoint']},
])
def test_incomplete_weights_rejected(backend: SimpleNamespace, loading_info: dict[str, Any]) -> None:
    """Reject partially initialized models even when metadata claims NER."""
    backend.model_loader.from_pretrained.return_value = backend.model, loading_info
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/ner'))
    with pytest.raises(ValueError, match='missing or incompatible'):
        extractor.extract('Fe')
    backend.factory.assert_not_called()


def test_missing_architecture_metadata_still_checks_weights(backend: SimpleNamespace) -> None:
    """Allow older complete checkpoints that omitted architecture metadata."""
    backend.checkpoint.architectures = None
    backend.model_loader.from_pretrained.return_value = backend.model, {'missing_keys': ['classifier.weight']}
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/ner'))
    with pytest.raises(ValueError, match='missing or incompatible'):
        extractor.extract('Fe')


def test_fast_tokenizer_required(backend: SimpleNamespace) -> None:
    """Require reliable offsets and overflow windows before loading weights."""
    backend.tokenizer.is_fast = False
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/ner'))
    with pytest.raises(ValueError, match='fast tokenizer'):
        extractor.extract('Fe')
    backend.model_loader.from_pretrained.assert_not_called()


def test_finite_model_limit_replaces_tokenizer_sentinel(backend: SimpleNamespace) -> None:
    """Bound overflow windows even when the tokenizer has an unknown limit."""
    backend.tokenizer.model_max_length = 10 ** 30
    backend.checkpoint.max_position_embeddings = 128
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/ner'))
    assert extractor.extract('Fe') == [[]]
    assert backend.tokenizer.model_max_length == 128


@pytest.mark.parametrize(('max_length', 'stride', 'message'), [
    (513, 64, 'exceeds'), (64, 64, 'content window'), (2, 0, 'special tokens'),
])
def test_invalid_windows_rejected(
    backend: SimpleNamespace, max_length: int, stride: int, message: str,
) -> None:
    """Validate context limits and content overlap before model execution."""
    config = entities.EntityExtractionConfig('test/ner', max_length=max_length, stride=stride)
    with pytest.raises(ValueError, match=message):
        entities.TransformerEntityExtractor(config).extract('Fe')
    backend.model_loader.from_pretrained.assert_not_called()


def test_explicit_finite_limit_required_when_metadata_has_none(backend: SimpleNamespace) -> None:
    """Never silently drop text when no safe overflow window is known."""
    backend.tokenizer.model_max_length = 10 ** 30
    backend.checkpoint.max_position_embeddings = None
    with pytest.raises(ValueError, match='no finite token limit'):
        entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/ner')).extract('Fe')
    config = entities.EntityExtractionConfig('test/ner', max_length=128)
    assert entities.TransformerEntityExtractor(config).extract('Fe') == [[]]
    assert backend.tokenizer.model_max_length == 128


@pytest.mark.parametrize('overrides', [
    {'model': ''}, {'device': ''}, {'batch_size': 0}, {'batch_size': True},
    {'stride': -1}, {'stride': 0.5}, {'max_length': 0}, {'max_length': True},
])
def test_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    """Fail on invalid inference controls without loading dependencies."""
    options = {'model': 'test/ner', **overrides}
    with pytest.raises(ValueError):
        entities.EntityExtractionConfig(**options)


@pytest.mark.parametrize('texts', [None, 2, ['Fe', None], b'Fe'])
def test_extract_rejects_non_string_inputs(backend: SimpleNamespace, texts: Any) -> None:
    """Reject malformed batches before attempting a download."""
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/ner'))
    with pytest.raises(TypeError, match='string'):
        extractor.extract(texts)
    backend.dependencies.assert_not_called()


@pytest.mark.parametrize(('start', 'end', 'score'), [(-1, 2, 0.9), (0, 4, 0.9), (0, 0, 0.9), (0, 2, float('nan'))])
def test_invalid_model_spans_rejected(backend: SimpleNamespace, start: int, end: int, score: float) -> None:
    """Refuse corrupt offsets and non-finite confidence instead of exporting them."""
    backend.inference.return_value = [[{'entity_group': 'MATERIAL', 'start': start, 'end': end, 'score': score}]]
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/ner'))
    with pytest.raises(ValueError, match='invalid entity span'):
        extractor.extract('Fe')


def test_prediction_count_must_match_non_empty_inputs(backend: SimpleNamespace) -> None:
    """Refuse to attribute entities when the classifier drops or adds an input."""
    backend.inference.return_value = [[], []]
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig('test/ner'))
    with pytest.raises(ValueError, match='unexpected number of input results'):
        extractor.extract(['', 'Fe'])
    backend.inference.assert_called_once_with(['Fe'], batch_size=8, num_workers=0)


def test_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain how to install local inference support when PyTorch is absent."""
    import sys

    monkeypatch.setitem(sys.modules, 'torch', None)
    with pytest.raises(ImportError, match=r'paperminertoolkit\[bert\]'):
        entities._load_dependencies()


@pytest.fixture(params=['bert', 'deberta-v2'])
def tiny_checkpoint(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Save a deterministic tiny BERT-family token classifier without downloads."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    vocabulary = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'Fe': 5, 'O': 6}
    backend = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocab=vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)],
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, model_max_length=8, pad_token='[PAD]', unk_token='[UNK]',
        cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]',
    )
    config_type, model_type = (
        (transformers.BertConfig, transformers.BertForTokenClassification)
        if request.param == 'bert'
        else (transformers.DebertaV2Config, transformers.DebertaV2ForTokenClassification)
    )
    config = config_type(
        vocab_size=len(vocabulary), hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=8,
        id2label={0: 'O', 1: 'B-MATERIAL', 2: 'I-MATERIAL'},
        label2id={'O': 0, 'B-MATERIAL': 1, 'I-MATERIAL': 2},
    )
    model = model_type(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    return tmp_path


def test_real_tiny_models_cover_long_inputs_and_preserve_offsets(tiny_checkpoint: Path) -> None:
    """Exercise real tokenizer overflow, padding, inference, and overlap removal."""
    config = entities.EntityExtractionConfig(str(tiny_checkpoint), stride=2, batch_size=2, local_files_only=True)
    extractor = entities.TransformerEntityExtractor(config)
    text = 'Fe O Fe O Fe O Fe O Fe O Fe O'
    predicted = extractor.extract([text, 'Fe', ''])
    assert len(predicted) == 3
    assert len(predicted[0]) == 12
    assert [row['text'] for row in predicted[0]] == text.split()
    assert predicted[0][-1]['end'] == len(text)
    assert len({(row['start'], row['end']) for row in predicted[0]}) == 12
    assert all(row['text'] == text[row['start']:row['end']] for row in predicted[0])
    assert all(row['label'] == 'MATERIAL' and row['score'] > 0.99 for row in predicted[0])
    assert predicted[1][0]['text'] == 'Fe'
    assert predicted[2] == []
    assert extractor.extract('Fe') == [predicted[1]]


def test_real_base_config_rejected_without_weights(tmp_path: Path) -> None:
    """Reject actual base-model metadata without requiring checkpoint weights."""
    pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    config = transformers.BertConfig(architectures=['BertForMaskedLM'])
    config.save_pretrained(tmp_path)
    extractor = entities.TransformerEntityExtractor(entities.EntityExtractionConfig(str(tmp_path), local_files_only=True))
    with pytest.raises(ValueError, match='fine-tuned token-classification'):
        extractor.extract('Fe')


def test_real_missing_classifier_weights_rejected(tiny_checkpoint: Path) -> None:
    """Detect an untrained head even if checkpoint metadata claims NER."""
    safetensors = pytest.importorskip('safetensors.torch')
    weights = tiny_checkpoint / 'model.safetensors'
    state = safetensors.load_file(weights)
    safetensors.save_file({key: value for key, value in state.items() if not key.startswith('classifier.')}, weights)
    config = entities.EntityExtractionConfig(str(tiny_checkpoint), stride=2, local_files_only=True)
    with pytest.raises(ValueError, match='missing or incompatible'):
        entities.TransformerEntityExtractor(config).extract('Fe')
