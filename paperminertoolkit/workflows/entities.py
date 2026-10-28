"""Extract source-aligned entities from corpus fields or UTF-8 text files."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from paperminertoolkit.corpus.database import connect, get_asset
from paperminertoolkit.extraction.entities import EntityExtractionConfig, TransformerEntityExtractor


def _corpus_documents(db_path: Path, fields: Sequence[str], batch_size: int) -> Iterator[dict[str, Any]]:
    """Yield raw corpus fields in paper-ID order without altering their text."""
    with closing(connect(db_path)) as conn:
        cursor = conn.execute('SELECT paper_id, doi, title FROM papers ORDER BY paper_id')
        while rows := cursor.fetchmany(batch_size):
            for paper in rows:
                for field in fields:
                    source = 'metadata'
                    blob_id = None
                    if field == 'title':
                        text = paper['title'] or ''
                    else:
                        asset = get_asset(conn, paper['paper_id'], field)
                        if asset is None:
                            continue
                        text = asset['content'].decode('utf-8')
                        source = asset['source']
                        blob_id = asset['blob_id']
                    if text.strip():
                        yield {
                            'corpus': str(db_path.resolve()),
                            'paper_id': paper['paper_id'], 'doi': paper['doi'] or '',
                            'field': field, 'source': source, 'blob_id': blob_id, 'text': text,
                        }


def _write_entities(
    documents: Iterable[dict[str, Any]],
    input_path: Path,
    output_path: str | os.PathLike[str],
    config: EntityExtractionConfig,
    min_score: float,
    labels: Sequence[str] | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Batch entity inference and atomically publish a provenance-bearing JSONL file."""
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError('min_score must be between 0 and 1.')
    if isinstance(labels, str) or labels is not None and any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError('labels must be a sequence of non-empty label strings.')
    selected_labels = set(labels) if labels else None
    output = Path(output_path)
    if output.resolve() == input_path.resolve() or (
        output.exists() and os.path.samefile(output, input_path)
    ):
        raise ValueError('The entity output must not replace the input file or corpus.')
    if output.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {output}. Pass overwrite=True to replace it.')

    extractor = TransformerEntityExtractor(config)
    summary: dict[str, Any] = {'papers': 0, 'documents': 0, 'entities': 0, 'output': str(output)}
    last_paper_id = None
    iterator = iter(documents)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=output.parent, prefix=f'.{output.name}.',
                                         suffix='.tmp', delete=False) as handle:
            temp_path = Path(handle.name)
            while True:
                batch = []
                for _ in range(config.batch_size):
                    document = next(iterator, None)
                    if document is None:
                        break
                    batch.append(document)
                if not batch:
                    break
                predictions = extractor.extract([document['text'] for document in batch])
                if len(predictions) != len(batch):
                    raise RuntimeError('Entity model returned a different number of results than inputs.')
                for document, entities in zip(batch, predictions):
                    kept = [entity for entity in entities if entity['score'] >= min_score
                            and (selected_labels is None or entity['label'] in selected_labels)]
                    row = {
                        **document,
                        'text_sha256': hashlib.sha256(document['text'].encode('utf-8')).hexdigest(),
                        'model': config.model, 'revision': config.revision,
                        'extraction': {**asdict(config), 'min_score': min_score,
                                       'labels': sorted(selected_labels) if selected_labels else None},
                        'entities': kept,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    summary['documents'] += 1
                    summary['entities'] += len(kept)
                    paper_id = document.get('paper_id')
                    if paper_id is not None and paper_id != last_paper_id:
                        summary['papers'] += 1
                        last_paper_id = paper_id
        if overwrite:
            os.replace(temp_path, output)
        else:
            # A hard link publishes the complete file without clobbering a
            # destination another process may have created during inference.
            os.link(temp_path, output)
    finally:
        if hasattr(iterator, 'close'):
            iterator.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return summary


def extract_corpus_entities(
    db_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    config: EntityExtractionConfig,
    *,
    text_fields: Sequence[str] = ('abstract',),
    min_score: float = 0.0,
    labels: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract entities from each selected raw corpus field into JSONL.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Existing SQLite paper corpus. All papers are considered.
    output_path : str or os.PathLike[str]
        Destination JSONL file in an existing directory.
    config : EntityExtractionConfig
        Fine-tuned token-classification checkpoint and inference settings.
    text_fields : Sequence[str], default=('abstract',)
        Fields to process separately: ``title``, ``abstract``, or ``text``.
        The newest stored asset is used; missing or blank fields are skipped.
    min_score : float, default=0.0
        Minimum entity confidence, inclusively between zero and one.
    labels : Sequence[str] or None, optional
        Retain only these checkpoint-specific entity labels, or all labels.
    overwrite : bool, default=False
        Replace an existing output after successful extraction.

    Returns
    -------
    dict[str, Any]
        Counts of papers, documents (individual fields), and entities, plus
        the output path. Each JSONL row includes the exact source text, model
        configuration, source identity, text hash, and entities. Entity spans
        use zero-based, end-exclusive character offsets within that row's text.

    Raises
    ------
    ValueError
        If fields, filters, or the output destination are invalid.
    OSError
        If an input is absent, output already exists, or file access fails.
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f'Corpus does not exist: {path}')
    if isinstance(text_fields, str):
        raise ValueError('text_fields must be a sequence of field names.')
    fields = tuple(dict.fromkeys(text_fields))
    if not fields or set(fields) - {'title', 'abstract', 'text'}:
        raise ValueError('Select at least one corpus field from: title, abstract, text.')
    return _write_entities(_corpus_documents(path, fields, config.batch_size), path,
                           output_path, config, min_score, labels, overwrite)


def extract_file_entities(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    config: EntityExtractionConfig,
    *,
    min_score: float = 0.0,
    labels: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract entities from one UTF-8 text file, preserving original offsets.

    Parameters
    ----------
    input_path : str or os.PathLike[str]
        UTF-8 text file treated as one document, including its original newlines.
    output_path : str or os.PathLike[str]
        Destination JSONL file in an existing directory.
    config : EntityExtractionConfig
        Fine-tuned token-classification checkpoint and inference settings.
    min_score : float, default=0.0
        Inclusive minimum confidence between zero and one.
    labels : Sequence[str] or None, optional
        Checkpoint-specific labels to keep; omit to retain all labels.
    overwrite : bool, default=False
        Replace an existing output after successful extraction.

    Returns
    -------
    dict[str, Any]
        Document and entity counts, output path, and ``papers=0``. The JSONL
        row contains source text, provenance, and end-exclusive entity spans.

    Raises
    ------
    ValueError
        If the text is not UTF-8 or filters or output destination are invalid.
    OSError
        If input or output cannot be accessed or the output already exists.
    """
    path = Path(input_path)
    text = path.read_bytes().decode('utf-8')
    documents = [{'source': str(path.resolve()), 'field': 'text', 'text': text}]
    return _write_entities(documents, path, output_path, config, min_score, labels, overwrite)
