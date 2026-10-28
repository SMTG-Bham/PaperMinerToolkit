"""Test source-preserving entity workflows without downloading model weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from paperminertoolkit.corpus import database as corpus
from paperminertoolkit.extraction.entities import EntityExtractionConfig
from paperminertoolkit.workflows import entities


class RecordingExtractor:
    """Record inference batches and return explicitly supplied entity predictions."""

    instances: list[RecordingExtractor] = []
    predictions: dict[str, list[dict[str, Any]]] = {}
    fail_on_call: int | None = None

    def __init__(self, config: EntityExtractionConfig) -> None:
        """Record the requested checkpoint and inference configuration."""
        self.config = config
        self.batches: list[list[str]] = []
        self.instances.append(self)

    def extract(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        """Return configured predictions, optionally failing partway through export."""
        self.batches.append(list(texts))
        if len(self.batches) == self.fail_on_call:
            raise RuntimeError('Model inference failed')
        return [self.predictions.get(text, []) for text in texts]


@pytest.fixture
def extractor(monkeypatch: pytest.MonkeyPatch) -> type[RecordingExtractor]:
    """Replace optional transformer inference with a resettable recording double."""
    monkeypatch.setattr(RecordingExtractor, 'instances', [])
    monkeypatch.setattr(RecordingExtractor, 'predictions', {})
    monkeypatch.setattr(RecordingExtractor, 'fail_on_call', None)
    monkeypatch.setattr(entities, 'TransformerEntityExtractor', RecordingExtractor)
    return RecordingExtractor


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a workflow JSONL export into document records."""
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def span(text: str, value: str, label: str = 'MATERIAL', score: float = 0.9) -> dict[str, Any]:
    """Construct a source-aligned entity prediction for a known substring."""
    start = text.index(value)
    return {'label': label, 'text': value, 'start': start, 'end': start + len(value), 'score': score}


def test_corpus_preserves_raw_fields_latest_assets_and_provenance(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
) -> None:
    """Batch separate fields, preserve chemical text, and identify the newest assets."""
    db_path = tmp_path / 'papers.db'
    output = tmp_path / 'entities.jsonl'
    title = '  Ni alloys '
    abstract = 'PSMILES: *CC(=O)O*\nFe₂O₃ improves.'
    full_text = '  Ni\r\nMixed CASE and spaces.\n'
    paper = {'paper_id': 'a', 'doi': '10.1000/a', 'title': title}
    with corpus.connect(db_path) as conn:
        corpus.add_asset(conn, paper, 'Obsolete abstract', 'abstract', 'text', 'text/plain', source='old')
        abstract_blob = corpus.add_asset(
            conn, paper, abstract, 'abstract', 'text', 'text/plain', source='publisher',
        )
        text_blob = corpus.add_asset(
            conn, paper, full_text, 'text', 'text', 'text/plain', source='pdf',
        )
        corpus.upsert_paper(conn, {'paper_id': 'b', 'title': 'No known entities'})
        corpus.add_asset(
            conn, {'paper_id': 'c'}, ' \r\n\t', 'abstract', 'text', 'text/plain',
        )
    extractor.predictions = {
        title: [span(title, 'Ni')],
        abstract: [span(abstract, '*CC(=O)O*', 'POLYMER')],
        full_text: [span(full_text, 'Ni')],
    }
    config = EntityExtractionConfig(model='test/materials-ner', revision='pinned', batch_size=2)

    summary = entities.extract_corpus_entities(
        db_path, output, config, text_fields=('title', 'abstract', 'text', 'abstract'),
    )

    assert summary == {'papers': 2, 'documents': 4, 'entities': 3, 'output': str(output)}
    assert extractor.instances[0].config is config
    assert extractor.instances[0].batches == [[title, abstract], [full_text, 'No known entities']]
    rows = read_rows(output)
    assert [row['field'] for row in rows] == ['title', 'abstract', 'text', 'title']
    assert [row['text'] for row in rows] == [title, abstract, full_text, 'No known entities']
    assert [row['source'] for row in rows] == ['metadata', 'publisher', 'pdf', 'metadata']
    assert [row['blob_id'] for row in rows] == [None, abstract_blob, text_blob, None]
    assert [row['paper_id'] for row in rows] == ['a', 'a', 'a', 'b']
    assert rows[0]['doi'] == '10.1000/a'
    assert rows[-1]['doi'] == ''
    assert rows[-1]['entities'] == []
    for row in rows:
        assert row['model'] == config.model
        assert row['revision'] == 'pinned'
        assert row['extraction'] == {**asdict(config), 'min_score': 0.0, 'labels': None}
        assert row['text_sha256'] == hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
        for entity in row['entities']:
            assert row['text'][entity['start']:entity['end']] == entity['text']
    assert rows[0]['entities'][0]['start'] == rows[2]['entities'][0]['start'] == 2


def test_file_preserves_psmiles_unicode_and_original_newlines(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
) -> None:
    """Keep byte-decoded source text and report character offsets for plain files."""
    source = tmp_path / 'polymers.txt'
    output = tmp_path / 'entities.jsonl'
    text = '  α-phase\r\nPSMILES: *c1ccccc1*\r\n'
    source.write_bytes(text.encode('utf-8'))
    extractor.predictions[text] = [span(text, '*c1ccccc1*', 'POLYMER')]

    summary = entities.extract_file_entities(source, output, EntityExtractionConfig(model='local-ner'))

    assert summary == {'papers': 0, 'documents': 1, 'entities': 1, 'output': str(output)}
    assert extractor.instances[0].batches == [[text]]
    row = read_rows(output)[0]
    assert row['text'] == text
    assert row['source'] == str(source.resolve())
    assert row['field'] == 'text'
    assert row['text_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert row['entities'] == extractor.predictions[text]


def test_filters_are_inclusive_and_retain_empty_prediction_documents(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
) -> None:
    """Apply score and label filters while retaining rows with no surviving spans."""
    source = tmp_path / 'input.txt'
    output = tmp_path / 'entities.jsonl'
    text = 'Ni Fe Cu Li'
    source.write_text(text)
    extractor.predictions[text] = [
        span(text, 'Ni', score=0.8), span(text, 'Fe', score=0.79),
        span(text, 'Cu', 'PROPERTY', 0.99), span(text, 'Li', score=0.95),
    ]
    config = EntityExtractionConfig(model='local-ner')

    summary = entities.extract_file_entities(
        source, output, config, min_score=0.8, labels=('MATERIAL',),
    )

    row = read_rows(output)[0]
    assert summary['entities'] == 2
    assert [entity['text'] for entity in row['entities']] == ['Ni', 'Li']
    assert row['extraction']['min_score'] == 0.8
    assert row['extraction']['labels'] == ['MATERIAL']

    summary = entities.extract_file_entities(
        source, output, config, labels=('UNPREDICTED',), overwrite=True,
    )
    assert summary['documents'] == 1
    assert summary['entities'] == 0
    assert read_rows(output)[0]['entities'] == []


def test_empty_corpus_skips_inference_and_exports_an_empty_file(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
) -> None:
    """Skip papers with absent or blank selected fields without calling inference."""
    db_path = tmp_path / 'papers.db'
    output = tmp_path / 'entities.jsonl'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'missing'})
        corpus.add_asset(conn, {'paper_id': 'blank'}, ' \n\t', 'abstract', 'text', 'text/plain')

    summary = entities.extract_corpus_entities(db_path, output, EntityExtractionConfig(model='ner'))

    assert summary['papers'] == summary['documents'] == summary['entities'] == 0
    assert output.read_bytes() == b''
    assert extractor.instances[0].batches == []


def test_existing_output_is_protected_and_overwrite_is_explicit(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
) -> None:
    """Refuse to clobber an existing artifact until overwrite is requested."""
    source = tmp_path / 'input.txt'
    source.write_text('Ni')
    output = tmp_path / 'entities.jsonl'
    output.write_text('previous export')
    config = EntityExtractionConfig(model='ner')

    with pytest.raises(FileExistsError, match='Output already exists'):
        entities.extract_file_entities(source, output, config)

    assert output.read_text() == 'previous export'
    assert extractor.instances == []
    entities.extract_file_entities(source, output, config, overwrite=True)
    assert read_rows(output)[0]['text'] == 'Ni'
    assert list(tmp_path.glob('.entities.jsonl.*.tmp')) == []


@pytest.mark.parametrize('existing_output', [False, True])
def test_inference_failure_never_publishes_partial_output(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
    existing_output: bool,
) -> None:
    """Clean temporary files and preserve previous output after a later batch fails."""
    db_path = tmp_path / 'papers.db'
    output = tmp_path / 'entities.jsonl'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'a', 'title': 'Ni'})
        corpus.upsert_paper(conn, {'paper_id': 'b', 'title': 'Fe'})
    if existing_output:
        output.write_text('previous export')
    extractor.fail_on_call = 2

    with pytest.raises(RuntimeError, match='Model inference failed'):
        entities.extract_corpus_entities(
            db_path, output, EntityExtractionConfig(model='ner', batch_size=1),
            text_fields=('title',), overwrite=existing_output,
        )

    assert extractor.instances[0].batches == [['Ni'], ['Fe']]
    assert list(tmp_path.glob('.entities.jsonl.*.tmp')) == []
    if existing_output:
        assert output.read_text() == 'previous export'
    else:
        assert not output.exists()


def test_prediction_count_mismatch_never_publishes_output(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse to pair entities with documents when the model drops a result."""
    db_path = tmp_path / 'papers.db'
    output = tmp_path / 'entities.jsonl'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'a', 'title': 'Ni'})
    monkeypatch.setattr(extractor, 'extract', lambda self, texts: [])

    with pytest.raises(RuntimeError, match='different number of results'):
        entities.extract_corpus_entities(
            db_path, output, EntityExtractionConfig(model='ner'), text_fields=('title',),
        )

    assert not output.exists()
    assert list(tmp_path.glob('.entities.jsonl.*.tmp')) == []


@pytest.mark.parametrize('as_corpus', [False, True])
@pytest.mark.parametrize('hardlink', [False, True])
def test_input_cannot_be_replaced_even_through_a_hardlink(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
    as_corpus: bool,
    hardlink: bool,
) -> None:
    """Reject output aliases of both text files and corpus databases before inference."""
    source = tmp_path / 'input'
    if as_corpus:
        with corpus.connect(source) as conn:
            corpus.upsert_paper(conn, {'paper_id': 'a', 'title': 'Ni'})
    else:
        source.write_text('Ni')
    before = source.read_bytes()
    output = tmp_path / 'alias' if hardlink else source
    if hardlink:
        os.link(source, output)
    workflow = entities.extract_corpus_entities if as_corpus else entities.extract_file_entities

    with pytest.raises(ValueError, match='must not replace the input'):
        workflow(source, output, EntityExtractionConfig(model='ner'), overwrite=True)

    assert source.read_bytes() == before
    assert extractor.instances == []


@pytest.mark.parametrize('fields', [(), ('body',), ('title', 'body'), 'abstract'])
def test_invalid_corpus_fields_fail_before_inference(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
    fields: Sequence[str],
) -> None:
    """Reject empty selections, unknown fields, and accidental string sequences."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path):
        pass
    output = tmp_path / 'entities.jsonl'

    with pytest.raises(ValueError, match='field'):
        entities.extract_corpus_entities(
            db_path, output, EntityExtractionConfig(model='ner'), text_fields=fields,
        )

    assert not output.exists()
    assert extractor.instances == []


@pytest.mark.parametrize('min_score', [-0.1, 1.1, float('nan'), float('inf')])
def test_invalid_score_filters_fail_before_inference(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
    min_score: float,
) -> None:
    """Require a finite confidence threshold within the probability range."""
    source = tmp_path / 'input.txt'
    source.write_text('Ni')
    output = tmp_path / 'entities.jsonl'

    with pytest.raises(ValueError, match='min_score'):
        entities.extract_file_entities(source, output, EntityExtractionConfig(model='ner'), min_score=min_score)

    assert not output.exists()
    assert extractor.instances == []


@pytest.mark.parametrize('labels', ['MATERIAL', ('',), (' ',), (42,)])
def test_invalid_label_filters_fail_before_inference(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
    labels: Any,
) -> None:
    """Require explicit nonempty label strings in a collection."""
    source = tmp_path / 'input.txt'
    source.write_text('Ni')
    output = tmp_path / 'entities.jsonl'

    with pytest.raises(ValueError, match='labels'):
        entities.extract_file_entities(source, output, EntityExtractionConfig(model='ner'), labels=labels)

    assert not output.exists()
    assert extractor.instances == []


def test_missing_corpus_is_not_created(
    tmp_path: Path,
    extractor: type[RecordingExtractor],
) -> None:
    """Reject a nonexistent corpus without creating an empty SQLite database."""
    db_path = tmp_path / 'missing.db'
    output = tmp_path / 'entities.jsonl'

    with pytest.raises(FileNotFoundError, match='Corpus does not exist'):
        entities.extract_corpus_entities(db_path, output, EntityExtractionConfig(model='ner'))

    assert not db_path.exists()
    assert not output.exists()
    assert extractor.instances == []
