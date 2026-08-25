"""Train, inspect, name, and apply reproducible LDA topic models.

Topic models are stored outside the paper corpus as versioned artifacts. The
corpus remains the source of paper metadata and text assets; predictions are
exported in long CSV form and can be explicitly registered for filtering.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import html
import json
import math
import re
import sqlite3
import tempfile
import time
import unicodedata
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from itertools import combinations, groupby
from os import PathLike
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import joblib
import numpy as np
import sklearn
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import spmatrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from paperminer.corpus.database import connect, get_asset
from paperminer.corpus.documents import trim_reference_section


SUPPORTED_TEXT_FIELDS = {'title', 'abstract', 'text'}
MODEL_FILENAME = 'lda_model.joblib'
VECTORIZER_FILENAME = 'vectorizer.joblib'
CONFIG_FILENAME = 'config.json'
REPORT_FILENAME = 'training_report.json'
FINGERPRINT_FILENAME = 'corpus_fingerprint.json'
TOPICS_FILENAME = 'topics.csv'
TOPIC_NAMES_FILENAME = 'topic_names.json'
REPRESENTATIVES_FILENAME = 'representative_papers.csv'
PREDICTIONS_FILENAME = 'paper_topics.csv'
STOPWORDS_FILENAME = 'stopwords.txt'
COMPARISON_CSV_FILENAME = 'model_comparison.csv'
COMPARISON_JSON_FILENAME = 'model_comparison.json'
TRENDS_CSV_FILENAME = 'topic_trends.csv'
TRENDS_REPORT_FILENAME = 'trend_report.json'
TRENDS_PLOT_FILENAME = 'topic_trends_plot.png'
ARTIFACT_VERSION = 2
TOKEN_PATTERN = r'(?u)\b[a-z][a-z0-9]{1,}\b'
TOKEN_RE = re.compile(TOKEN_PATTERN)
URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)
DOI_RE = re.compile(r'\b10\.\d{4,9}/\S+', re.IGNORECASE)
DEFAULT_BATCH_SIZE = 128
DEFAULT_EVALUATION_SAMPLE_SIZE = 10000


class _TopicDocument(TypedDict):
    """Normalized corpus document used by topic-model workflows."""

    paper_id: str
    doi: str
    title: str
    publication_date: str
    text: str
    token_count: int


class _TopicRow(TypedDict):
    """Exportable topic description."""

    topic_id: int
    topic_name: str
    top_terms: list[str]
    representative_papers: NotRequired[list[dict[str, str]]]


class TopicAnalyzer:
    """Generate domain-aware features for topic models.

    Parameters
    ----------
    domain_stopwords : iterable of str, optional
        Domain-specific words to omit as standalone features.
    ngram_max : int, default=2
        Maximum n-gram size to emit.

    Attributes
    ----------
    domain_stopwords : frozenset of str
        Normalized collection of standalone features to omit.
    ngram_max : int
        Maximum configured n-gram size.
    """

    def __init__(self, domain_stopwords: Iterable[str] = (), ngram_max: int = 2) -> None:
        """Initialize the topic analyzer.

        Parameters
        ----------
        domain_stopwords : Iterable[str], optional
            Domain-specific words to omit as standalone features.
        ngram_max : int, default=2
            Maximum n-gram size to emit.
        """
        self.domain_stopwords = frozenset(domain_stopwords)
        self.ngram_max = ngram_max

    def __call__(self, document: str) -> list[str]:
        """Extract topic-model features from a document.

        Parameters
        ----------
        document : str
            Normalized document text to analyze.

        Returns
        -------
        list[str]
            Unigram and optional bigram features.
        """
        features = []
        for segment in re.split(r'[.!?;:\n]+', document):
            tokens = [token for token in TOKEN_RE.findall(segment) if token not in ENGLISH_STOP_WORDS]
            features.extend(token for token in tokens if token not in self.domain_stopwords)
            if self.ngram_max >= 2:
                for left, right in zip(tokens, tokens[1:]):
                    if left in self.domain_stopwords and right in self.domain_stopwords:
                        continue
                    features.append(f'{left}_{right}')
        return features


def _utc_now() -> str:
    """Return a stable UTC timestamp for model metadata.

    Returns
    -------
    str
        ISO-8601 UTC timestamp with second precision.
    """
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def normalize_topic_text(value: object) -> str:
    """Normalize text while preserving words and chemical formulae.

    Parameters
    ----------
    value : object
        Document text or a value coercible to text.

    Returns
    -------
    str
        Unicode-normalized, case-folded plain text without URLs or DOIs.
    """
    if value is None:
        return ''
    text = html.unescape(str(value))
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = URL_RE.sub(' ', text)
    text = DOI_RE.sub(' ', text)
    text = text.casefold()
    return re.sub(r'\s+', ' ', text).strip()


def load_domain_stopwords(stopwords_file: str | PathLike[str] | None) -> list[str]:
    """Load corpus-specific stopwords from a text file.

    Parameters
    ----------
    stopwords_file : str, os.PathLike[str], or None
        File containing one word per non-comment line, or ``None``.

    Returns
    -------
    list[str]
        Sorted unique normalized stopwords.

    Raises
    ------
    OSError
        If the stopword file cannot be read.
    ValueError
        If a non-comment line does not contain exactly one normalized word.
    """
    if stopwords_file is None:
        return []
    path = Path(stopwords_file)
    words = []
    for line_number, raw_line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), start=1):
        value = raw_line.split('#', 1)[0].strip()
        if not value:
            continue
        normalized = normalize_topic_text(value)
        tokens = TOKEN_RE.findall(normalized)
        if len(tokens) != 1 or tokens[0] != normalized:
            raise ValueError(
                f'{path}:{line_number}: expected exactly one word per line; found {value!r}.'
            )
        words.append(tokens[0])
    return sorted(set(words))


def _validate_text_fields(text_fields: Iterable[str]) -> tuple[str, ...]:
    """Validate and deduplicate topic-model text fields.

    Parameters
    ----------
    text_fields : Iterable[str]
        Requested corpus text fields.

    Returns
    -------
    tuple[str, ...]
        Unique fields in first-seen order.

    Raises
    ------
    ValueError
        If no fields are supplied or any field is unsupported.
    """
    fields = tuple(dict.fromkeys(text_fields))
    if not fields:
        raise ValueError('At least one topic-model text field is required.')
    unsupported = set(fields) - SUPPORTED_TEXT_FIELDS
    if unsupported:
        raise ValueError(f'Unsupported topic-model text fields: {", ".join(sorted(unsupported))}')
    return fields


def _topic_document(
    paper: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any] | None],
    fields: Iterable[str],
) -> _TopicDocument:
    """Build one normalized topic document from metadata and loaded assets.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    assets : Mapping[str, Mapping[str, Any] or None]
        Newest stored asset for each requested content role.
    fields : Iterable[str]
        Metadata and asset fields to combine.

    Returns
    -------
    _TopicDocument
        Paper metadata, normalized modeling text, and token count.
    """
    pieces = []
    for field in fields:
        if field == 'title':
            value = paper.get('title') or ''
        else:
            asset = assets.get(field)
            value = asset['content'].decode('utf-8', errors='replace') if asset else ''
            if field == 'text' and value:
                value = trim_reference_section(value)
        if str(value).strip():
            pieces.append(str(value))
    text = normalize_topic_text(' '.join(pieces))
    return {
        'paper_id': paper['paper_id'],
        'doi': paper.get('doi') or '',
        'title': paper.get('title') or '',
        'publication_date': paper.get('publication_date') or '',
        'text': text,
        'token_count': len(TOKEN_RE.findall(text)),
    }


def iter_topic_document_batches(
    db_path: str | PathLike[str],
    text_fields: Iterable[str] = ('title', 'abstract'),
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[list[_TopicDocument]]:
    """Yield stable, bounded batches of normalized corpus documents.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the SQLite paper corpus.
    text_fields : Iterable[str], default=('title', 'abstract')
        Metadata and asset fields to combine into each document.
    batch_size : int, default=128
        Maximum number of papers loaded per batch.

    Yields
    ------
    list[_TopicDocument]
        Normalized documents ordered by paper ID.

    Raises
    ------
    ValueError
        If a requested text field is unsupported or ``batch_size`` is not
        positive.
    """
    fields = _validate_text_fields(text_fields)
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    asset_roles = [field for field in fields if field in {'abstract', 'text'}]
    with connect(db_path) as conn:
        last_paper_id = ''
        while True:
            rows = conn.execute(
                'SELECT * FROM papers WHERE paper_id > ? ORDER BY paper_id LIMIT ?',
                (last_paper_id, batch_size),
            ).fetchall()
            if not rows:
                break
            documents = []
            for row in rows:
                paper = dict(row)
                assets = {
                    role: get_asset(conn, paper['paper_id'], role)
                    for role in asset_roles
                }
                documents.append(_topic_document(paper, assets, fields))
            yield documents
            last_paper_id = rows[-1]['paper_id']


def load_topic_documents(
    db_path: str | PathLike[str],
    text_fields: Iterable[str] = ('title', 'abstract'),
) -> list[_TopicDocument]:
    """Load normalized topic-model documents from a corpus.

    Parameters
    ----------
    db_path : str or pathlib.Path
        Path to the SQLite paper corpus.
    text_fields : Iterable[str], default=('title', 'abstract')
        Metadata and asset fields to combine into each document.

    Returns
    -------
    list[_TopicDocument]
        Paper metadata, normalized text, and token counts.

    Raises
    ------
    ValueError
        If any requested text field is unsupported.
    """
    fields = _validate_text_fields(text_fields)
    return [
        document
        for batch in iter_topic_document_batches(db_path, fields)
        for document in batch
    ]


def _median(values: Sequence[int | float]) -> int | float:
    """Calculate the median of numeric values.

    Parameters
    ----------
    values : Sequence[int or float]
        Values whose median is required.

    Returns
    -------
    int or float
        Median value, or zero for an empty sequence.
    """
    if not values:
        return 0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def assess_topic_corpus(
    documents: Sequence[_TopicDocument],
    num_topics: int,
) -> dict[str, Any]:
    """Assess corpus quality for LDA topic modeling.

    Parameters
    ----------
    documents : Sequence[_TopicDocument]
        Topic documents containing ``token_count`` values.
    num_topics : int
        Requested number of latent topics.

    Returns
    -------
    dict[str, Any]
        Corpus counts, length diagnostics, and heuristic warnings.

    Raises
    ------
    ValueError
        If fewer than two topics are requested or too few usable documents
        exist.
    """
    if num_topics < 2:
        raise ValueError('num_topics must be at least 2.')
    usable = [document for document in documents if document['token_count'] > 0]
    if len(usable) < num_topics:
        raise ValueError(
            f'LDA requires at least as many usable documents as topics; '
            f'found {len(usable)} documents for {num_topics} topics.'
        )

    document_count = len(documents)
    empty_count = document_count - len(usable)
    token_counts = [document['token_count'] for document in usable]
    warning_messages = []
    recommended_documents = max(500, 50 * num_topics)
    if len(usable) < recommended_documents:
        warning_messages.append(
            f'Small topic-model corpus: {len(usable)} usable documents for {num_topics} topics. '
            f'Consider at least {recommended_documents} documents (50 per topic, minimum 500) '
            'for more stable topics.'
        )
    median_tokens = _median(token_counts)
    if median_tokens < 50:
        warning_messages.append(
            f'Short topic-model documents: median usable length is {median_tokens:g} tokens; '
            'topic quality may be poor below roughly 50 tokens.'
        )
    if document_count and empty_count / document_count > 0.1:
        warning_messages.append(
            f'{empty_count} of {document_count} documents contain no usable text '
            'and will be excluded from training.'
        )
    return {
        'documents_total': document_count,
        'documents_usable_before_vectorization': len(usable),
        'documents_empty': empty_count,
        'median_tokens': median_tokens,
        'documents_per_topic': len(usable) / num_topics,
        'warnings': warning_messages,
    }


def _emit_warnings(messages: Iterable[str]) -> None:
    """Emit corpus diagnostics through Python's warnings interface.

    Parameters
    ----------
    messages : Iterable[str]
        Warning messages to emit.
    """
    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=3)


def _corpus_fingerprint(documents: Iterable[_TopicDocument]) -> str:
    """Build a deterministic fingerprint for topic documents.

    Parameters
    ----------
    documents : Iterable[_TopicDocument]
        Documents containing paper IDs and normalized text.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item['paper_id']):
        digest.update(document['paper_id'].encode('utf-8'))
        digest.update(b'\0')
        digest.update(hashlib.sha256(document['text'].encode('utf-8')).digest())
        digest.update(b'\0')
    return digest.hexdigest()


def _topic_rows(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_terms: int,
) -> list[_TopicRow]:
    """Extract the highest-weighted terms from each fitted topic.

    Parameters
    ----------
    model : sklearn.decomposition.LatentDirichletAllocation
        Fitted LDA model.
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        Fitted feature vectorizer.
    top_terms : int
        Maximum terms to retain per topic.

    Returns
    -------
    list[_TopicRow]
        Topic IDs, blank manual names, and weighted terms.
    """
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for topic_id, weights in enumerate(model.components_):
        indices = weights.argsort()[::-1][:top_terms]
        rows.append({
            'topic_id': topic_id,
            'topic_name': '',
            'top_terms': [str(feature_names[index]) for index in indices],
        })
    return rows


def _write_json(path: str | PathLike[str], value: Any) -> None:
    """Write deterministic, readable JSON metadata.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Destination JSON path.
    value : Any
        JSON-serializable value to write.

    Raises
    ------
    OSError
        If the destination cannot be written.
    TypeError
        If ``value`` is not JSON serializable.
    """
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _write_topics(
    path: str | PathLike[str],
    topic_rows: Iterable[_TopicRow],
) -> None:
    """Write manually nameable topic summaries.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Destination CSV path.
    topic_rows : Iterable[_TopicRow]
        Topic IDs, names, and top-term lists.

    Raises
    ------
    OSError
        If the destination cannot be written.
    """
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['topic_id', 'topic_name', 'top_terms'])
        writer.writeheader()
        for row in topic_rows:
            writer.writerow({
                'topic_id': row['topic_id'],
                'topic_name': row.get('topic_name', ''),
                'top_terms': '; '.join(row['top_terms']),
            })


def _refresh_csv_topic_names(
    path: str | PathLike[str],
    names: Mapping[str, str],
) -> None:
    """Refresh names in an existing topic artifact CSV.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Artifact CSV to update when it exists.
    names : Mapping[str, str]
        Manual names keyed by string topic ID.

    Raises
    ------
    OSError
        If the artifact cannot be read, rewritten, or replaced.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or 'topic_name' not in fieldnames:
        return
    for row in rows:
        topic_id = row.get('topic_id', '')
        if topic_id != '':
            row['topic_name'] = names.get(str(topic_id), '')
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _topic_names(
    model_dir: str | PathLike[str],
    num_topics: int,
) -> dict[str, str]:
    """Load manual topic names from a model artifact.

    Parameters
    ----------
    model_dir : str or os.PathLike[str]
        Topic-model artifact directory.
    num_topics : int
        Number of topic IDs to include.

    Returns
    -------
    dict[str, str]
        Manual names keyed by topic ID, with missing names left blank.

    Raises
    ------
    OSError
        If the topic-name file cannot be read.
    json.JSONDecodeError
        If the topic-name file contains invalid JSON.
    """
    path = Path(model_dir) / TOPIC_NAMES_FILENAME
    if path.exists():
        values = json.loads(path.read_text(encoding='utf-8'))
    else:
        values = {}
    return {str(topic_id): str(values.get(str(topic_id), '')) for topic_id in range(num_topics)}


def _write_predictions(
    path: str | PathLike[str],
    documents: Sequence[_TopicDocument],
    distributions: Iterable[NDArray[np.float64]],
    names: Mapping[str, str],
    included_indices: Iterable[int],
) -> None:
    """Write long-form topic predictions and skipped-paper states.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Destination CSV path.
    documents : Sequence[_TopicDocument]
        Paper documents in output order.
    distributions : Iterable[numpy.ndarray]
        Topic probability vectors for included documents.
    names : Mapping[str, str]
        Manual names keyed by string topic ID.
    included_indices : Iterable[int]
        Document indices corresponding to ``distributions``.

    Raises
    ------
    OSError
        If the destination cannot be written.
    """
    distribution_by_index = dict(zip(included_indices, distributions))
    fieldnames = [
        'paper_id', 'doi', 'title', 'publication_date', 'topic_id',
        'topic_name', 'probability', 'is_dominant', 'status',
    ]
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for document_index, document in enumerate(documents):
            distribution = distribution_by_index.get(document_index)
            base = {key: document[key] for key in ['paper_id', 'doi', 'title', 'publication_date']}
            if distribution is None:
                writer.writerow({**base, 'status': 'no_vocabulary_terms'})
                continue
            dominant_topic = int(distribution.argmax())
            for topic_id, probability in enumerate(distribution):
                writer.writerow({
                    **base,
                    'topic_id': topic_id,
                    'topic_name': names[str(topic_id)],
                    'probability': f'{float(probability):.12g}',
                    'is_dominant': topic_id == dominant_topic,
                    'status': 'predicted',
                })


def _write_representatives(
    path: str | PathLike[str],
    documents: Sequence[_TopicDocument],
    distributions: NDArray[np.float64],
    names: Mapping[str, str],
    count: int,
) -> None:
    """Write representative papers for manual topic interpretation.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Destination CSV path.
    documents : Sequence[_TopicDocument]
        Documents corresponding to distribution rows.
    distributions : numpy.ndarray
        Per-document topic probabilities.
    names : Mapping[str, str]
        Manual names keyed by string topic ID.
    count : int
        Maximum representative papers per topic.

    Raises
    ------
    OSError
        If the destination cannot be written.
    """
    fieldnames = ['topic_id', 'topic_name', 'rank', 'paper_id', 'probability', 'title', 'publication_date']
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for topic_id in range(distributions.shape[1]):
            ranked = distributions[:, topic_id].argsort()[::-1][:count]
            for rank, document_index in enumerate(ranked, start=1):
                document = documents[int(document_index)]
                writer.writerow({
                    'topic_id': topic_id,
                    'topic_name': names[str(topic_id)],
                    'rank': rank,
                    'paper_id': document['paper_id'],
                    'probability': f'{float(distributions[document_index, topic_id]):.12g}',
                    'title': document['title'],
                    'publication_date': document['publication_date'],
                })


def _prepare_output_directory(
    output_dir: str | PathLike[str],
    overwrite: bool,
) -> Path:
    """Prepare a topic-model artifact directory.

    Parameters
    ----------
    output_dir : str or os.PathLike[str]
        Artifact directory to create or reuse.
    overwrite : bool
        Whether a nonempty directory may be reused.

    Returns
    -------
    pathlib.Path
        Prepared artifact directory.

    Raises
    ------
    OSError
        If the directory cannot be inspected or created.
    ValueError
        If the directory is nonempty and ``overwrite`` is false.
    """
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ValueError(f'Model directory is not empty: {path}. Pass overwrite=True to replace model files.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _streaming_corpus_report(
    prepared: Mapping[str, Any],
    num_topics: int,
) -> dict[str, Any]:
    """Build topic-count-specific diagnostics from a prepared corpus cache.

    Parameters
    ----------
    prepared : Mapping[str, Any]
        Metadata produced while constructing the streaming corpus cache.
    num_topics : int
        Requested number of latent topics.

    Returns
    -------
    dict[str, Any]
        Corpus counts, timing data, vocabulary size, and quality warnings.

    Raises
    ------
    ValueError
        If too few usable or vectorized documents remain for ``num_topics``.
    """
    usable = prepared['documents_usable_before_vectorization']
    used = prepared['documents_used']
    if usable < num_topics:
        raise ValueError(
            f'LDA requires at least as many usable documents as topics; '
            f'found {usable} documents for {num_topics} topics.'
        )
    if used < num_topics:
        raise ValueError(
            f'Only {used} documents contain retained vocabulary for {num_topics} topics. '
            'Lower min_df, raise max_df, or reduce the topic count.'
        )
    warnings_list = []
    recommended = max(500, 50 * num_topics)
    if usable < recommended:
        warnings_list.append(
            f'Small topic-model corpus: {usable} usable documents for {num_topics} topics. '
            f'Consider at least {recommended} documents (50 per topic, minimum 500) '
            'for more stable topics.'
        )
    if prepared['median_tokens'] < 50:
        warnings_list.append(
            f'Short topic-model documents: median usable length is '
            f'{prepared["median_tokens"]:g} tokens; topic quality may be poor below roughly 50 tokens.'
        )
    total = prepared['documents_total']
    empty = prepared['documents_empty']
    if total and empty / total > 0.1:
        warnings_list.append(
            f'{empty} of {total} documents contain no usable text and will be excluded from training.'
        )
    without_vocabulary = usable - used
    if prepared['vocabulary_size'] < 2 * num_topics:
        warnings_list.append(
            f'Small topic vocabulary: {prepared["vocabulary_size"]} retained terms for '
            f'{num_topics} topics. Consider lowering min_df, raising max_features, '
            'or reducing the topic count.'
        )
    if without_vocabulary:
        warnings_list.append(
            f'{without_vocabulary} usable documents contained no terms from the retained '
            'vocabulary and were excluded.'
        )
    return {
        'documents_total': total,
        'documents_usable_before_vectorization': usable,
        'documents_empty': empty,
        'median_tokens': prepared['median_tokens'],
        'documents_per_topic': usable / num_topics,
        'documents_used': used,
        'documents_without_vocabulary_terms': without_vocabulary,
        'vocabulary_size': prepared['vocabulary_size'],
        'vectorization_seconds': prepared['preparation_seconds'],
        'cache_size_bytes': prepared['cache_size_bytes'],
        'warnings': warnings_list,
    }


def _prepare_streaming_corpus(
    db_path: str | PathLike[str],
    text_fields: Iterable[str],
    domain_stopwords: Iterable[str],
    ngram_max: int,
    min_df: int,
    max_df: float,
    max_features: int,
    batch_size: int,
    evaluation_sample_size: int,
    work_dir: str | PathLike[str],
) -> dict[str, Any]:
    """Build a deterministic vocabulary and disk-backed sparse batch cache.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the source SQLite paper corpus.
    text_fields : Iterable[str]
        Corpus fields combined into each modeling document.
    domain_stopwords : Iterable[str]
        Normalized corpus-specific words excluded from features.
    ngram_max : {1, 2}
        Maximum feature n-gram size.
    min_df : int
        Minimum document frequency for retained features.
    max_df : float
        Maximum document-frequency fraction for retained features.
    max_features : int
        Maximum retained vocabulary size.
    batch_size : int
        Maximum papers processed in each disk-backed batch.
    evaluation_sample_size : int
        Maximum deterministic sample size used for fit metrics.
    work_dir : str or os.PathLike[str]
        Temporary directory in which to store counts and sparse matrices.

    Returns
    -------
    dict[str, Any]
        Cache paths, fitted vectorizer, corpus diagnostics, and fingerprint.

    Raises
    ------
    OSError
        If cache files cannot be created or written.
    ValueError
        If no usable documents or vocabulary terms remain.
    """
    started = time.perf_counter()
    text_fields = _validate_text_fields(text_fields)
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    counts_path = work_path / 'vocabulary_counts.db'
    analyzer = TopicAnalyzer(domain_stopwords=domain_stopwords, ngram_max=ngram_max)
    digest = hashlib.sha256()
    token_counts = []
    document_count = 0

    with sqlite3.connect(counts_path) as vocab_conn:
        vocab_conn.execute(
            'CREATE TABLE term_counts ('
            'term TEXT PRIMARY KEY, term_frequency INTEGER NOT NULL, '
            'document_frequency INTEGER NOT NULL)'
        )
        for documents in iter_topic_document_batches(db_path, text_fields, batch_size):
            batch_tf = Counter()
            batch_df = Counter()
            for document in documents:
                document_count += 1
                digest.update(document['paper_id'].encode('utf-8'))
                digest.update(b'\0')
                digest.update(hashlib.sha256(document['text'].encode('utf-8')).digest())
                digest.update(b'\0')
                if document['token_count'] <= 0:
                    continue
                token_counts.append(document['token_count'])
                counts = Counter(analyzer(document['text']))
                batch_tf.update(counts)
                batch_df.update(counts.keys())
            vocab_conn.executemany(
                """
                INSERT INTO term_counts (term, term_frequency, document_frequency)
                VALUES (?, ?, ?)
                ON CONFLICT(term) DO UPDATE SET
                    term_frequency = term_frequency + excluded.term_frequency,
                    document_frequency = document_frequency + excluded.document_frequency
                """,
                ((term, frequency, batch_df[term]) for term, frequency in batch_tf.items()),
            )
            vocab_conn.commit()

        usable_count = len(token_counts)
        if not usable_count:
            raise ValueError('The corpus contains no usable topic-model text.')
        if min_df > usable_count:
            raise ValueError(f'min_df={min_df} exceeds the {usable_count} usable documents.')
        max_documents = max_df * usable_count
        rows = vocab_conn.execute(
            """
            SELECT term FROM term_counts
            WHERE document_frequency >= ? AND document_frequency <= ?
            ORDER BY term_frequency DESC, term ASC
            LIMIT ?
            """,
            (min_df, max_documents, max_features),
        ).fetchall()
    selected_terms = sorted(row[0] for row in rows)
    if not selected_terms:
        raise ValueError(
            'Could not build the topic vocabulary: no terms remain after document-frequency filtering.'
        )
    vocabulary = {term: index for index, term in enumerate(selected_terms)}
    vectorizer = CountVectorizer(analyzer=analyzer, vocabulary=vocabulary, dtype='int64')

    batches = []
    documents_used = 0
    evaluation_heap = []
    sequence = 0
    for batch_number, documents in enumerate(
            iter_topic_document_batches(db_path, text_fields, batch_size)):
        matrix = vectorizer.transform(document['text'] for document in documents).tocsr()
        matrix_path = work_path / f'batch-{batch_number:08d}.npz'
        metadata_path = work_path / f'batch-{batch_number:08d}.json'
        sparse.save_npz(matrix_path, matrix, compressed=True)
        metadata = [
            {key: document[key] for key in [
                'paper_id', 'doi', 'title', 'publication_date', 'token_count'
            ]}
            for document in documents
        ]
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        batches.append((matrix_path, metadata_path))
        nonzero_rows = np.flatnonzero(matrix.getnnz(axis=1) > 0)
        documents_used += len(nonzero_rows)
        for row_index in nonzero_rows:
            if evaluation_sample_size <= 0:
                continue
            hash_value = int.from_bytes(
                hashlib.sha256(documents[int(row_index)]['paper_id'].encode('utf-8')).digest()[:8],
                'big',
            )
            entry = (-hash_value, sequence, matrix.getrow(int(row_index)))
            sequence += 1
            if len(evaluation_heap) < evaluation_sample_size:
                heapq.heappush(evaluation_heap, entry)
            elif hash_value < -evaluation_heap[0][0]:
                heapq.heapreplace(evaluation_heap, entry)

    if evaluation_heap:
        evaluation_rows = [entry[2] for entry in sorted(evaluation_heap, reverse=True)]
        evaluation_matrix = sparse.vstack(evaluation_rows, format='csr')
    else:
        evaluation_matrix = sparse.csr_matrix((0, len(vocabulary)), dtype='int64')
    evaluation_path = work_path / 'evaluation.npz'
    sparse.save_npz(evaluation_path, evaluation_matrix, compressed=True)
    cache_size_bytes = sum(
        path.stat().st_size for path in work_path.iterdir() if path.is_file()
    )
    return {
        'batches': batches,
        'vectorizer': vectorizer,
        'evaluation_path': evaluation_path,
        'documents_total': document_count,
        'documents_usable_before_vectorization': usable_count,
        'documents_empty': document_count - usable_count,
        'median_tokens': _median(token_counts),
        'documents_used': documents_used,
        'vocabulary_size': len(vocabulary),
        'preparation_seconds': time.perf_counter() - started,
        'cache_size_bytes': cache_size_bytes,
        'fingerprint': {
            'algorithm': 'sha256-paper-id-and-normalized-text-v2',
            'sha256': digest.hexdigest(),
            'documents': document_count,
            'text_fields': list(text_fields),
        },
    }


def _cached_batches(
    prepared: Mapping[str, Any],
) -> Iterator[tuple[spmatrix, list[dict[str, Any]]]]:
    """Load sparse document matrices and metadata from a prepared cache.

    Parameters
    ----------
    prepared : Mapping[str, Any]
        Streaming cache metadata containing matrix and metadata paths.

    Yields
    ------
    tuple[scipy.sparse.spmatrix, list[dict[str, Any]]]
        Sparse document-term matrix and aligned paper metadata.
    """
    for matrix_path, metadata_path in prepared['batches']:
        yield sparse.load_npz(matrix_path), json.loads(metadata_path.read_text(encoding='utf-8'))


def _model_identifier(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    config: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
) -> str:
    """Build an immutable deterministic identifier for a fitted model.

    Parameters
    ----------
    model : sklearn.decomposition.LatentDirichletAllocation
        Fitted LDA estimator.
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        Vectorizer containing the ordered model vocabulary.
    config : Mapping[str, Any]
        Model configuration; volatile identity fields are ignored.
    fingerprint : Mapping[str, Any]
        Fingerprint of the normalized training corpus.

    Returns
    -------
    str
        Stable ``lda:`` identifier derived from configuration, corpus,
        vocabulary, and fitted components.
    """
    digest = hashlib.sha256()
    stable_config = {key: value for key, value in config.items() if key not in {'created_at', 'model_id'}}
    digest.update(json.dumps(stable_config, sort_keys=True).encode('utf-8'))
    digest.update(fingerprint['sha256'].encode('ascii'))
    digest.update('\0'.join(vectorizer.get_feature_names_out()).encode('utf-8'))
    digest.update(np.asarray(model.components_, dtype='<f8').tobytes())
    return f'lda:{digest.hexdigest()[:24]}'


def _write_streamed_outputs(
    prepared: Mapping[str, Any],
    model: LatentDirichletAllocation,
    names: Mapping[str, str],
    predictions_path: str | PathLike[str],
    representatives_path: str | PathLike[str],
    representative_count: int,
) -> dict[str, Any]:
    """Write predictions and bounded representatives from cached batches.

    Parameters
    ----------
    prepared : Mapping[str, Any]
        Streaming cache metadata.
    model : sklearn.decomposition.LatentDirichletAllocation
        Fitted topic model.
    names : Mapping[str, str]
        Manual topic names keyed by string topic ID.
    predictions_path : str or os.PathLike[str]
        Destination long-form prediction CSV.
    representatives_path : str or os.PathLike[str]
        Destination representative-paper CSV.
    representative_count : int
        Maximum papers retained for each topic.

    Returns
    -------
    dict[str, Any]
        Prediction coverage and dominant-topic balance metrics.
    """
    prediction_fields = [
        'paper_id', 'doi', 'title', 'publication_date', 'topic_id',
        'topic_name', 'probability', 'is_dominant', 'status',
    ]
    representative_heaps = {topic_id: [] for topic_id in range(model.n_components)}
    dominant_counts = [0] * model.n_components
    predicted = 0
    sequence = 0
    with Path(predictions_path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for matrix, documents in _cached_batches(prepared):
            included = np.flatnonzero(matrix.getnnz(axis=1) > 0)
            distributions = model.transform(matrix[included]) if len(included) else np.empty((0, model.n_components))
            distribution_by_row = dict(zip(included.tolist(), distributions))
            for row_index, document in enumerate(documents):
                base = {key: document[key] for key in ['paper_id', 'doi', 'title', 'publication_date']}
                distribution = distribution_by_row.get(row_index)
                if distribution is None:
                    writer.writerow({**base, 'status': 'no_vocabulary_terms'})
                    continue
                predicted += 1
                dominant_topic = int(distribution.argmax())
                dominant_counts[dominant_topic] += 1
                for topic_id, probability in enumerate(distribution):
                    probability = float(probability)
                    writer.writerow({
                        **base,
                        'topic_id': topic_id,
                        'topic_name': names[str(topic_id)],
                        'probability': f'{probability:.12g}',
                        'is_dominant': topic_id == dominant_topic,
                        'status': 'predicted',
                    })
                    item = (probability, sequence, document)
                    heap = representative_heaps[topic_id]
                    if len(heap) < representative_count:
                        heapq.heappush(heap, item)
                    elif probability > heap[0][0]:
                        heapq.heapreplace(heap, item)
                sequence += 1

    fields = ['topic_id', 'topic_name', 'rank', 'paper_id', 'probability', 'title', 'publication_date']
    with Path(representatives_path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for topic_id in range(model.n_components):
            ranked = sorted(representative_heaps[topic_id], reverse=True)
            for rank, (probability, _, document) in enumerate(ranked, start=1):
                writer.writerow({
                    'topic_id': topic_id,
                    'topic_name': names[str(topic_id)],
                    'rank': rank,
                    'paper_id': document['paper_id'],
                    'probability': f'{probability:.12g}',
                    'title': document['title'],
                    'publication_date': document['publication_date'],
                })
    proportions = [count / predicted for count in dominant_counts if count] if predicted else []
    entropy = -sum(value * math.log(value) for value in proportions)
    balance = entropy / math.log(model.n_components) if model.n_components > 1 and predicted else 0
    return {
        'dominant_topic_counts': dominant_counts,
        'dominant_topic_balance': balance,
        'smallest_dominant_topic': min(dominant_counts),
        'largest_dominant_topic': max(dominant_counts),
        'papers_predicted': predicted,
    }


def _model_quality_metrics(
    model: LatentDirichletAllocation,
    matrix: spmatrix,
    distributions: NDArray[np.float64],
    topic_rows: Sequence[_TopicRow],
) -> dict[str, Any]:
    """Calculate diagnostics for a fitted topic model.

    Parameters
    ----------
    model : sklearn.decomposition.LatentDirichletAllocation
        Fitted LDA model.
    matrix : scipy.sparse.spmatrix
        Document-term matrix used to fit the model.
    distributions : numpy.ndarray
        Per-document topic probabilities.
    topic_rows : Sequence[_TopicRow]
        Extracted topic terms.

    Returns
    -------
    dict[str, Any]
        Fit, topic-diversity, and dominant-topic balance metrics.
    """
    dominant_topics = distributions.argmax(axis=1)
    dominant_counts = [int((dominant_topics == topic_id).sum()) for topic_id in range(distributions.shape[1])]
    proportions = [count / len(distributions) for count in dominant_counts if count]
    entropy = -sum(proportion * math.log(proportion) for proportion in proportions)
    normalized_entropy = entropy / math.log(distributions.shape[1]) if distributions.shape[1] > 1 else 0
    all_top_terms = [term for topic in topic_rows for term in topic['top_terms']]
    topic_diversity = len(set(all_top_terms)) / len(all_top_terms) if all_top_terms else 0
    return {
        'perplexity': float(model.perplexity(matrix)),
        'log_likelihood': float(model.score(matrix)),
        'topic_diversity': topic_diversity,
        'dominant_topic_counts': dominant_counts,
        'dominant_topic_balance': normalized_entropy,
        'smallest_dominant_topic': min(dominant_counts),
        'largest_dominant_topic': max(dominant_counts),
    }


def _train_streaming_topic_model(
    output_dir: str | PathLike[str],
    prepared: Mapping[str, Any],
    num_topics: int,
    text_fields: Iterable[str],
    min_df: int,
    max_df: float,
    max_features: int,
    max_iter: int,
    random_state: int,
    top_terms: int,
    representative_papers: int,
    domain_stopwords: list[str],
    ngram_max: int,
    overwrite: bool,
    emit_warnings: bool,
    batch_size: int,
    evaluation_sample_size: int,
) -> dict[str, Any]:
    """Train and persist online LDA from a reusable streaming cache.

    Parameters
    ----------
    output_dir : str or os.PathLike[str]
        Destination model artifact directory.
    prepared : Mapping[str, Any]
        Prepared streaming corpus cache and diagnostics.
    num_topics : int
        Number of latent topics.
    text_fields : Iterable[str]
        Corpus fields represented by the model.
    min_df : int
        Minimum feature document frequency.
    max_df : float
        Maximum feature document-frequency fraction.
    max_features : int
        Maximum vocabulary size.
    max_iter : int
        Number of complete passes over cached batches.
    random_state : int
        Model initialization seed.
    top_terms : int
        Terms exported for each topic.
    representative_papers : int
        Representative papers exported for each topic.
    domain_stopwords : list[str]
        Normalized corpus-specific stopwords.
    ngram_max : {1, 2}
        Maximum feature n-gram size.
    overwrite : bool
        Whether known artifacts may be replaced.
    emit_warnings : bool
        Whether to emit heuristic corpus warnings.
    batch_size : int
        Documents processed per online batch.
    evaluation_sample_size : int
        Maximum documents used for fit metrics.

    Returns
    -------
    dict[str, Any]
        Artifact paths, configuration, diagnostics, fingerprint, and topics.
    """
    report = _streaming_corpus_report(prepared, num_topics)
    if emit_warnings:
        _emit_warnings(report['warnings'])
    model = LatentDirichletAllocation(
        n_components=num_topics,
        learning_method='online',
        max_iter=1,
        batch_size=batch_size,
        total_samples=prepared['documents_used'],
        random_state=random_state,
    )
    fitting_started = time.perf_counter()
    for _ in range(max_iter):
        for matrix, _metadata in _cached_batches(prepared):
            included = matrix.getnnz(axis=1) > 0
            if included.any():
                model.partial_fit(matrix[included])
    report['fitting_seconds'] = time.perf_counter() - fitting_started

    vectorizer = prepared['vectorizer']
    topic_rows = _topic_rows(model, vectorizer, top_terms)
    all_top_terms = [term for topic in topic_rows for term in topic['top_terms']]
    report['topic_diversity'] = (
        len(set(all_top_terms)) / len(all_top_terms) if all_top_terms else 0
    )
    evaluation_matrix = sparse.load_npz(prepared['evaluation_path'])
    report['evaluation_documents'] = evaluation_matrix.shape[0]
    report['metrics_scope'] = (
        'complete' if evaluation_matrix.shape[0] == prepared['documents_used'] else 'sample'
    )
    if evaluation_matrix.shape[0]:
        report['perplexity'] = float(model.perplexity(evaluation_matrix))
        report['log_likelihood'] = float(model.score(evaluation_matrix))
    else:
        report['perplexity'] = None
        report['log_likelihood'] = None

    config = {
        'artifact_version': ARTIFACT_VERSION,
        'created_at': _utc_now(),
        'num_topics': num_topics,
        'text_fields': list(text_fields),
        'min_df': min_df,
        'max_df': max_df,
        'max_features': max_features,
        'learning_method': 'online',
        'max_iter': max_iter,
        'random_state': random_state,
        'top_terms': top_terms,
        'representative_papers': representative_papers,
        'domain_stopwords': domain_stopwords,
        'ngram_max': ngram_max,
        'streaming': True,
        'batch_size': batch_size,
        'evaluation_sample_size': evaluation_sample_size,
        'sklearn_version': sklearn.__version__,
    }
    fingerprint = prepared['fingerprint']
    config['model_id'] = _model_identifier(model, vectorizer, config, fingerprint)
    names = {str(topic_id): '' for topic_id in range(num_topics)}
    output_path = _prepare_output_directory(output_dir, overwrite)

    inference_metrics = _write_streamed_outputs(
        prepared,
        model,
        names,
        output_path / PREDICTIONS_FILENAME,
        output_path / REPRESENTATIVES_FILENAME,
        representative_papers,
    )
    report.update(inference_metrics)
    joblib.dump(model, output_path / MODEL_FILENAME)
    joblib.dump(vectorizer, output_path / VECTORIZER_FILENAME)
    _write_json(output_path / CONFIG_FILENAME, config)
    _write_json(output_path / REPORT_FILENAME, report)
    _write_json(output_path / FINGERPRINT_FILENAME, fingerprint)
    _write_json(output_path / TOPIC_NAMES_FILENAME, names)
    (output_path / STOPWORDS_FILENAME).write_text(
        ''.join(f'{word}\n' for word in domain_stopwords), encoding='utf-8'
    )
    _write_topics(output_path / TOPICS_FILENAME, topic_rows)
    return {
        'model_dir': str(output_path),
        'config': config,
        'report': report,
        'fingerprint': fingerprint,
        'topics': topic_rows,
        'predictions_path': str(output_path / PREDICTIONS_FILENAME),
    }


def train_topic_model(db_path: str | PathLike[str],
                      output_dir: str | PathLike[str],
                      num_topics: int = 10,
                      text_fields: Iterable[str] = ('title', 'abstract'),
                      min_df: int = 2,
                      max_df: float = 0.95,
                      max_features: int = 20000,
                      learning_method: str = 'online',
                      max_iter: int = 20,
                      random_state: int = 0,
                      top_terms: int = 15,
                      representative_papers: int = 5,
                      stopwords_file: str | PathLike[str] | None = None,
                      ngram_max: int = 2,
                      overwrite: bool = False,
                      emit_warnings: bool = True,
                      documents: Sequence[_TopicDocument] | None = None,
                      streaming: bool = True,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      cache_dir: str | PathLike[str] | None = None,
                      evaluation_sample_size: int = DEFAULT_EVALUATION_SAMPLE_SIZE,
                      _prepared_streaming: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Train and persist an LDA model and its inspection artifacts.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the source SQLite paper corpus.
    output_dir : str or os.PathLike[str]
        Directory in which to store model artifacts.
    num_topics : int, default=10
        Number of latent topics to fit.
    text_fields : Iterable[str], default=('title', 'abstract')
        Corpus fields combined into each modeling document.
    min_df : int, default=2
        Minimum number of documents in which a feature must occur.
    max_df : float, default=0.95
        Maximum fraction of documents in which a feature may occur.
    max_features : int, default=20000
        Maximum retained vocabulary size.
    learning_method : {'batch', 'online'}, default='online'
        Scikit-learn LDA learning strategy. Streaming mode requires ``online``.
    max_iter : int, default=20
        Training iterations or complete passes over streaming batches.
    random_state : int, default=0
        Seed controlling model initialization.
    top_terms : int, default=15
        Number of terms exported for each topic.
    representative_papers : int, default=5
        Number of high-probability papers exported per topic.
    stopwords_file : str, os.PathLike[str], or None, optional
        File containing domain-specific stopwords.
    ngram_max : {1, 2}, default=2
        Maximum feature n-gram size.
    overwrite : bool, default=False
        Whether to reuse a nonempty artifact directory.
    emit_warnings : bool, default=True
        Whether to emit heuristic corpus-quality warnings.
    documents : Sequence[_TopicDocument] or None, optional
        Preloaded documents used instead of reading ``db_path`` in in-memory
        mode.
    streaming : bool, default=True
        Whether to train from bounded disk-backed document batches.
    batch_size : int, default=128
        Maximum documents processed in each streaming batch.
    cache_dir : str, os.PathLike[str], or None, optional
        Parent directory for the temporary streaming cache.
    evaluation_sample_size : int, default=10000
        Maximum deterministic document sample used for streaming fit metrics.
    _prepared_streaming : Mapping[str, Any] or None, optional
        Internal reusable cache supplied by model comparison.

    Returns
    -------
    dict[str, Any]
        Artifact paths, configuration, quality report, fingerprint, and topics.

    Raises
    ------
    OSError
        If an input, cache, or artifact file cannot be accessed.
    ValueError
        If configuration, corpus size, or retained vocabulary is unsuitable.
    """
    fields = _validate_text_fields(text_fields)
    if learning_method not in {'online', 'batch'}:
        raise ValueError('learning_method must be one of: online, batch')
    if min_df < 1:
        raise ValueError('min_df must be at least 1.')
    if not 0 < max_df <= 1:
        raise ValueError('max_df must be greater than 0 and at most 1.')
    if max_features < num_topics:
        raise ValueError('max_features must be at least num_topics.')
    if ngram_max not in {1, 2}:
        raise ValueError('ngram_max must be either 1 or 2.')
    if max_iter < 1 or top_terms < 1 or representative_papers < 1:
        raise ValueError('max_iter, top_terms, and representative_papers must be positive.')
    if batch_size < 1 or evaluation_sample_size < 1:
        raise ValueError('batch_size and evaluation_sample_size must be positive.')

    domain_stopwords = load_domain_stopwords(stopwords_file)
    if streaming:
        if learning_method != 'online':
            raise ValueError('Streaming training requires learning_method="online"; use in-memory mode for batch LDA.')
        if documents is not None:
            raise ValueError('Explicit documents are only supported by in-memory training.')
        if _prepared_streaming is not None:
            return _train_streaming_topic_model(
                output_dir, _prepared_streaming, num_topics, fields, min_df, max_df,
                max_features, max_iter, random_state, top_terms,
                representative_papers, domain_stopwords, ngram_max, overwrite,
                emit_warnings, batch_size, evaluation_sample_size,
            )
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix='paperminer-topics-', dir=cache_dir) as temporary_dir:
            prepared = _prepare_streaming_corpus(
                db_path, fields, domain_stopwords, ngram_max, min_df, max_df,
                max_features, batch_size, evaluation_sample_size, temporary_dir,
            )
            return _train_streaming_topic_model(
                output_dir, prepared, num_topics, fields, min_df, max_df,
                max_features, max_iter, random_state, top_terms,
                representative_papers, domain_stopwords, ngram_max, overwrite,
                emit_warnings, batch_size, evaluation_sample_size,
            )

    documents = documents if documents is not None else load_topic_documents(db_path, fields)
    report = assess_topic_corpus(documents, num_topics)
    usable_documents = [document for document in documents if document['token_count'] > 0]
    if min_df > len(usable_documents):
        raise ValueError(f'min_df={min_df} exceeds the {len(usable_documents)} usable documents.')

    vectorizer = CountVectorizer(
        analyzer=TopicAnalyzer(domain_stopwords=domain_stopwords, ngram_max=ngram_max),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        dtype='int64',
    )
    vectorization_started = time.perf_counter()
    try:
        matrix = vectorizer.fit_transform(document['text'] for document in usable_documents)
    except ValueError as error:
        raise ValueError(f'Could not build the topic vocabulary: {error}') from error
    included_mask = matrix.getnnz(axis=1) > 0
    included_indices = [index for index, included in enumerate(included_mask) if included]
    training_documents = [usable_documents[index] for index in included_indices]
    matrix = matrix[included_mask]
    if len(training_documents) < num_topics:
        raise ValueError(
            f'Only {len(training_documents)} documents contain retained vocabulary for {num_topics} topics. '
            'Lower min_df, raise max_df, or reduce the topic count.'
        )

    feature_count = len(vectorizer.get_feature_names_out())
    report['documents_used'] = len(training_documents)
    report['documents_without_vocabulary_terms'] = len(usable_documents) - len(training_documents)
    report['vocabulary_size'] = feature_count
    report['vectorization_seconds'] = time.perf_counter() - vectorization_started
    if feature_count < 2 * num_topics:
        report['warnings'].append(
            f'Small topic vocabulary: {feature_count} retained terms for {num_topics} topics. '
            'Consider lowering min_df, raising max_features, or reducing the topic count.'
        )
    if report['documents_without_vocabulary_terms']:
        report['warnings'].append(
            f'{report["documents_without_vocabulary_terms"]} usable documents contained no terms '
            'from the retained vocabulary and were excluded.'
        )
    if emit_warnings:
        _emit_warnings(report['warnings'])

    model = LatentDirichletAllocation(
        n_components=num_topics,
        learning_method=learning_method,
        max_iter=max_iter,
        random_state=random_state,
    )
    fitting_started = time.perf_counter()
    distributions = model.fit_transform(matrix)
    report['fitting_seconds'] = time.perf_counter() - fitting_started
    topic_rows = _topic_rows(model, vectorizer, top_terms)
    report.update(_model_quality_metrics(model, matrix, distributions, topic_rows))
    config = {
        'artifact_version': ARTIFACT_VERSION,
        'created_at': _utc_now(),
        'num_topics': num_topics,
        'text_fields': list(fields),
        'min_df': min_df,
        'max_df': max_df,
        'max_features': max_features,
        'learning_method': learning_method,
        'max_iter': max_iter,
        'random_state': random_state,
        'top_terms': top_terms,
        'representative_papers': representative_papers,
        'domain_stopwords': domain_stopwords,
        'ngram_max': ngram_max,
        'streaming': False,
        'batch_size': batch_size,
        'evaluation_sample_size': evaluation_sample_size,
        'sklearn_version': sklearn.__version__,
    }
    fingerprint = {
        'algorithm': 'sha256-paper-id-and-normalized-text-v2',
        'sha256': _corpus_fingerprint(documents),
        'documents': len(documents),
        'text_fields': list(fields),
    }
    config['model_id'] = _model_identifier(model, vectorizer, config, fingerprint)
    names = {str(topic_id): '' for topic_id in range(num_topics)}
    output_path = _prepare_output_directory(output_dir, overwrite)

    joblib.dump(model, output_path / MODEL_FILENAME)
    joblib.dump(vectorizer, output_path / VECTORIZER_FILENAME)
    _write_json(output_path / CONFIG_FILENAME, config)
    _write_json(output_path / REPORT_FILENAME, report)
    _write_json(output_path / FINGERPRINT_FILENAME, fingerprint)
    _write_json(output_path / TOPIC_NAMES_FILENAME, names)
    (output_path / STOPWORDS_FILENAME).write_text(
        ''.join(f'{word}\n' for word in domain_stopwords),
        encoding='utf-8',
    )
    _write_topics(output_path / TOPICS_FILENAME, topic_rows)
    _write_representatives(
        output_path / REPRESENTATIVES_FILENAME,
        training_documents,
        distributions,
        names,
        representative_papers,
    )
    training_index_by_paper_id = {
        document['paper_id']: index for index, document in enumerate(training_documents)
    }
    all_included_indices = [
        index for index, document in enumerate(documents)
        if document['paper_id'] in training_index_by_paper_id
    ]
    ordered_distributions = [
        distributions[training_index_by_paper_id[documents[index]['paper_id']]]
        for index in all_included_indices
    ]
    _write_predictions(
        output_path / PREDICTIONS_FILENAME,
        documents,
        ordered_distributions,
        names,
        all_included_indices,
    )
    return {
        'model_dir': str(output_path),
        'config': config,
        'report': report,
        'fingerprint': fingerprint,
        'topics': topic_rows,
        'predictions_path': str(output_path / PREDICTIONS_FILENAME),
    }


def load_topic_model(
    model_dir: str | PathLike[str],
) -> tuple[LatentDirichletAllocation, CountVectorizer, dict[str, Any], dict[str, str]]:
    """Load and validate a trusted local LDA artifact.

    Parameters
    ----------
    model_dir : str or os.PathLike[str]
        Topic-model artifact directory.

    Returns
    -------
    tuple[LatentDirichletAllocation, CountVectorizer, dict[str, Any], dict[str, str]]
        Fitted LDA model, fitted vectorizer, configuration, and manual names.

    Raises
    ------
    FileNotFoundError
        If the model configuration is missing.
    OSError
        If an artifact cannot be read.
    ValueError
        If the artifact format version is unsupported.
    json.JSONDecodeError
        If JSON metadata is invalid.

    Notes
    -----
    Joblib artifacts can execute code while loading and must come from a
    trusted source.
    """
    path = Path(model_dir)
    config_path = path / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f'Missing topic model configuration: {config_path}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('artifact_version') != ARTIFACT_VERSION:
        raise ValueError(
            f'Unsupported topic model artifact version: {config.get("artifact_version")}; '
            f'expected {ARTIFACT_VERSION}.'
        )
    model = joblib.load(path / MODEL_FILENAME)
    vectorizer = joblib.load(path / VECTORIZER_FILENAME)
    names = _topic_names(path, config['num_topics'])
    return model, vectorizer, config, names


def topic_descriptions(model_dir: str | PathLike[str]) -> list[_TopicRow]:
    """Load descriptions and representative papers for each topic.

    Parameters
    ----------
    model_dir : str or os.PathLike[str]
        Topic-model artifact directory.

    Returns
    -------
    list[_TopicRow]
        Topic terms, manual names, and representative paper rows.

    Raises
    ------
    OSError
        If required artifacts cannot be read.
    ValueError
        If the artifact format version is unsupported.
    """
    model, vectorizer, config, names = load_topic_model(model_dir)
    topics = _topic_rows(model, vectorizer, config['top_terms'])
    representatives = {topic_id: [] for topic_id in range(config['num_topics'])}
    path = Path(model_dir) / REPRESENTATIVES_FILENAME
    if path.exists():
        with path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                representatives[int(row['topic_id'])].append(row)
    for topic in topics:
        topic['topic_name'] = names[str(topic['topic_id'])]
        topic['representative_papers'] = representatives[topic['topic_id']]
    return topics


def set_topic_name(
    model_dir: str | PathLike[str],
    topic_id: int,
    topic_name: str,
) -> dict[str, str]:
    """Set a manual topic name and refresh artifact exports.

    Parameters
    ----------
    model_dir : str or os.PathLike[str]
        Topic-model artifact directory.
    topic_id : int
        Zero-based topic identifier.
    topic_name : str
        Nonempty manual topic name.

    Returns
    -------
    dict[str, str]
        Updated names keyed by string topic ID.

    Raises
    ------
    OSError
        If model artifacts cannot be read or rewritten.
    ValueError
        If the artifact is unsupported, the ID is out of range, or the name is
        empty.
    """
    model, vectorizer, config, names = load_topic_model(model_dir)
    if topic_id < 0 or topic_id >= config['num_topics']:
        raise ValueError(f'topic_id must be between 0 and {config["num_topics"] - 1}.')
    name = str(topic_name).strip()
    if not name:
        raise ValueError('topic_name must not be empty.')
    names[str(topic_id)] = name
    path = Path(model_dir)
    _write_json(path / TOPIC_NAMES_FILENAME, names)
    rows = _topic_rows(model, vectorizer, config['top_terms'])
    for row in rows:
        row['topic_name'] = names[str(row['topic_id'])]
    _write_topics(path / TOPICS_FILENAME, rows)
    _refresh_csv_topic_names(path / REPRESENTATIVES_FILENAME, names)
    _refresh_csv_topic_names(path / PREDICTIONS_FILENAME, names)
    return names


def topic_corpus_fingerprint(
    db_path: str | PathLike[str],
    text_fields: Iterable[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Fingerprint normalized topic inputs for stale-score detection.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the SQLite paper corpus.
    text_fields : Iterable[str]
        Fields normalized into each fingerprinted document.
    batch_size : int, default=128
        Maximum papers loaded per batch.

    Returns
    -------
    dict[str, Any]
        Fingerprint algorithm, SHA-256 digest, document count, and text fields.
    """
    fields = _validate_text_fields(text_fields)
    digest = hashlib.sha256()
    documents = 0
    for batch in iter_topic_document_batches(db_path, fields, batch_size):
        for document in batch:
            documents += 1
            digest.update(document['paper_id'].encode('utf-8'))
            digest.update(b'\0')
            digest.update(hashlib.sha256(document['text'].encode('utf-8')).digest())
            digest.update(b'\0')
    return {
        'algorithm': 'sha256-paper-id-and-normalized-text-v2',
        'sha256': digest.hexdigest(),
        'documents': documents,
        'text_fields': list(fields),
    }


def _iter_topic_predictions(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    config: Mapping[str, Any],
    db_path: str | PathLike[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Yield fresh prediction records in bounded corpus batches.

    Parameters
    ----------
    model : sklearn.decomposition.LatentDirichletAllocation
        Fitted model used for inference.
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        Saved training vectorizer.
    config : Mapping[str, Any]
        Saved model configuration.
    db_path : str or os.PathLike[str]
        Corpus whose papers are predicted.
    batch_size : int, default=128
        Maximum papers inferred at once.

    Yields
    ------
    dict[str, Any]
        Document metadata, prediction status, topic distribution, dominant
        topic, and normalized-document fingerprint.
    """
    for documents in iter_topic_document_batches(db_path, config['text_fields'], batch_size):
        matrix = vectorizer.transform(document['text'] for document in documents).tocsr()
        included = np.flatnonzero(matrix.getnnz(axis=1) > 0)
        distributions = model.transform(matrix[included]) if len(included) else np.empty((0, config['num_topics']))
        by_row = dict(zip(included.tolist(), distributions))
        for row_index, document in enumerate(documents):
            distribution = by_row.get(row_index)
            yield {
                'document': document,
                'status': 'predicted' if distribution is not None else 'no_vocabulary_terms',
                'distribution': distribution,
                'dominant_topic': int(distribution.argmax()) if distribution is not None else None,
                'document_fingerprint': hashlib.sha256(
                    document['text'].encode('utf-8')
                ).hexdigest(),
            }


def predict_topic_model(
    model_dir: str | PathLike[str],
    db_path: str | PathLike[str],
    output_path: str | PathLike[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int | str]:
    """Apply a saved topic model and export long-form scores.

    Parameters
    ----------
    model_dir : str or os.PathLike[str]
        Topic-model artifact directory.
    db_path : str or os.PathLike[str]
        Path to the SQLite paper corpus to score.
    output_path : str or os.PathLike[str]
        Destination prediction CSV.
    batch_size : int, default=128
        Maximum papers inferred at once.

    Returns
    -------
    dict[str, int or str]
        Counts of total, predicted, and skipped papers plus the output path.

    Raises
    ------
    OSError
        If model artifacts, the corpus, or output path cannot be accessed.
    ValueError
        If the artifact is unsupported or the corpus contains no papers.
    """
    model, vectorizer, config, names = load_topic_model(model_dir)
    fields = [
        'paper_id', 'doi', 'title', 'publication_date', 'topic_id',
        'topic_name', 'probability', 'is_dominant', 'status',
    ]
    total = 0
    predicted = 0
    with Path(output_path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prediction in _iter_topic_predictions(
                model, vectorizer, config, db_path, batch_size):
            total += 1
            document = prediction['document']
            base = {key: document[key] for key in ['paper_id', 'doi', 'title', 'publication_date']}
            if prediction['status'] != 'predicted':
                writer.writerow({**base, 'status': prediction['status']})
                continue
            predicted += 1
            for topic_id, probability in enumerate(prediction['distribution']):
                writer.writerow({
                    **base,
                    'topic_id': topic_id,
                    'topic_name': names[str(topic_id)],
                    'probability': f'{float(probability):.12g}',
                    'is_dominant': topic_id == prediction['dominant_topic'],
                    'status': 'predicted',
                })
    if not total:
        raise ValueError('The corpus contains no papers to predict.')
    return {
        'papers_total': total,
        'papers_predicted': predicted,
        'papers_without_vocabulary_terms': total - predicted,
        'output_path': str(output_path),
    }


def store_topic_model_scores(
    model_dir: str | PathLike[str],
    db_path: str | PathLike[str],
    name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Predict and transactionally store one immutable model run.

    Parameters
    ----------
    model_dir : str or os.PathLike[str]
        Topic-model artifact directory.
    db_path : str or os.PathLike[str]
        Corpus in which to store model metadata and paper scores.
    name : str or None, optional
        Stable display name; defaults to the model directory name.
    batch_size : int, default=128
        Maximum papers inferred at once.

    Returns
    -------
    dict[str, Any]
        Model identity, prediction counts, and prediction fingerprint.

    Raises
    ------
    OSError
        If model artifacts or the corpus cannot be accessed.
    ValueError
        If the model name conflicts with an existing immutable identity.
    """
    model, vectorizer, config, names = load_topic_model(model_dir)
    model_id = config['model_id']
    resolved_name = str(name or Path(model_dir).name).strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', resolved_name):
        raise ValueError('Model name must use only letters, numbers, dots, underscores, and hyphens.')
    prediction_fingerprint = topic_corpus_fingerprint(
        db_path, config['text_fields'], batch_size
    )
    training_fingerprint = json.loads(
        (Path(model_dir) / FINGERPRINT_FILENAME).read_text(encoding='utf-8')
    )
    topic_rows = topic_descriptions(model_dir)
    predicted_at = _utc_now()
    predicted_count = 0
    missing_count = 0

    with tempfile.TemporaryDirectory(prefix='paperminer-topic-store-') as temporary_dir:
        staging_path = Path(temporary_dir) / 'predictions.db'
        with sqlite3.connect(staging_path) as staging:
            staging.executescript(
                """
                CREATE TABLE predictions (
                    paper_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    dominant_topic_id INTEGER, document_fingerprint TEXT NOT NULL
                );
                CREATE TABLE scores (
                    paper_id TEXT NOT NULL, topic_id INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    PRIMARY KEY (paper_id, topic_id)
                );
                """
            )
            for prediction in _iter_topic_predictions(
                    model, vectorizer, config, db_path, batch_size):
                document = prediction['document']
                staging.execute(
                    'INSERT INTO predictions VALUES (?, ?, ?, ?)',
                    (
                        document['paper_id'], prediction['status'],
                        prediction['dominant_topic'], prediction['document_fingerprint'],
                    ),
                )
                if prediction['status'] == 'predicted':
                    predicted_count += 1
                    staging.executemany(
                        'INSERT INTO scores VALUES (?, ?, ?)',
                        (
                            (document['paper_id'], topic_id, float(probability))
                            for topic_id, probability in enumerate(prediction['distribution'])
                        ),
                    )
                else:
                    missing_count += 1
            staging.commit()

        with connect(db_path) as conn, sqlite3.connect(staging_path) as staging:
            existing_name = conn.execute(
                'SELECT model_id FROM topic_models WHERE name = ?', (resolved_name,)
            ).fetchone()
            if existing_name is not None and existing_name['model_id'] != model_id:
                raise ValueError(
                    f'Topic model name {resolved_name!r} already refers to a different model.'
                )
            existing_model = conn.execute(
                'SELECT name FROM topic_models WHERE model_id = ?', (model_id,)
            ).fetchone()
            if existing_model is not None and existing_model['name'] != resolved_name:
                raise ValueError(
                    f'Model {model_id} is already stored as {existing_model["name"]!r}.'
                )
            now = _utc_now()
            try:
                conn.execute('BEGIN')
                conn.execute(
                    """
                    INSERT INTO topic_models (
                        model_id, name, artifact_path, artifact_version, config_json,
                        training_corpus_fingerprint, prediction_corpus_fingerprint,
                        text_fields_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(model_id) DO UPDATE SET
                        artifact_path=excluded.artifact_path,
                        config_json=excluded.config_json,
                        prediction_corpus_fingerprint=excluded.prediction_corpus_fingerprint,
                        text_fields_json=excluded.text_fields_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        model_id, resolved_name, str(Path(model_dir).resolve()),
                        config['artifact_version'], json.dumps(config, sort_keys=True),
                        training_fingerprint['sha256'], prediction_fingerprint['sha256'],
                        json.dumps(config['text_fields']), now, now,
                    ),
                )
                conn.execute('DELETE FROM topic_definitions WHERE model_id = ?', (model_id,))
                conn.executemany(
                    'INSERT INTO topic_definitions VALUES (?, ?, ?, ?)',
                    (
                        (
                            model_id, row['topic_id'], names[str(row['topic_id'])],
                            json.dumps(row['top_terms']),
                        )
                        for row in topic_rows
                    ),
                )
                conn.execute('DELETE FROM paper_topic_predictions WHERE model_id = ?', (model_id,))
                prediction_rows = staging.execute(
                    'SELECT paper_id, status, dominant_topic_id, document_fingerprint FROM predictions'
                )
                while batch := prediction_rows.fetchmany(1000):
                    conn.executemany(
                        'INSERT INTO paper_topic_predictions VALUES (?, ?, ?, ?, ?, ?)',
                        (
                            (model_id, row[0], row[1], row[2], row[3], predicted_at)
                            for row in batch
                        ),
                    )
                score_rows = staging.execute(
                    'SELECT paper_id, topic_id, probability FROM scores'
                )
                while batch := score_rows.fetchmany(5000):
                    conn.executemany(
                        'INSERT INTO paper_topic_scores VALUES (?, ?, ?, ?)',
                        ((model_id, row[0], row[1], row[2]) for row in batch),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    try:
        from paperminer.corpus.filtering import refresh_topic_filters
        refresh_topic_filters(db_path, model_id)
    except ImportError:
        pass
    return {
        'model_id': model_id,
        'name': resolved_name,
        'papers_predicted': predicted_count,
        'papers_without_vocabulary_terms': missing_count,
        'prediction_corpus_fingerprint': prediction_fingerprint['sha256'],
    }


def stored_topic_models(
    db_path: str | PathLike[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """List stored topic models with coverage and freshness information.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the SQLite paper corpus.
    batch_size : int, default=128
        Maximum papers loaded while calculating current fingerprints.

    Returns
    -------
    list[dict[str, Any]]
        Stored model rows augmented with topic count, prediction coverage,
        text fields, and current/stale state.
    """
    with connect(db_path) as conn:
        models = [dict(row) for row in conn.execute(
            'SELECT * FROM topic_models ORDER BY name'
        ).fetchall()]
        for item in models:
            counts = dict(conn.execute(
                'SELECT status, COUNT(*) FROM paper_topic_predictions '
                'WHERE model_id = ? GROUP BY status', (item['model_id'],)
            ).fetchall())
            item['papers_predicted'] = counts.get('predicted', 0)
            item['papers_without_vocabulary_terms'] = counts.get('no_vocabulary_terms', 0)
            item['num_topics'] = conn.execute(
                'SELECT COUNT(*) FROM topic_definitions WHERE model_id = ?',
                (item['model_id'],),
            ).fetchone()[0]
            item['text_fields'] = json.loads(item['text_fields_json'])
    fingerprints = {}
    for item in models:
        key = tuple(item['text_fields'])
        if key not in fingerprints:
            fingerprints[key] = topic_corpus_fingerprint(db_path, key, batch_size)['sha256']
        item['is_current'] = fingerprints[key] == item['prediction_corpus_fingerprint']
    return models


def _prediction_papers(
    predictions_path: str | PathLike[str],
) -> Iterator[dict[str, Any]]:
    """Group a long-form prediction CSV into complete paper records.

    Parameters
    ----------
    predictions_path : str or os.PathLike[str]
        Long-form topic prediction CSV ordered by paper.

    Yields
    ------
    dict[str, Any]
        Paper ID, publication date, status, complete probability mapping, and
        dominant topic ID.
    """
    with Path(predictions_path).open(encoding='utf-8', newline='') as handle:
        rows = csv.DictReader(handle)
        for paper_id, grouped_rows in groupby(rows, key=lambda row: row['paper_id']):
            grouped = list(grouped_rows)
            first = grouped[0]
            status = first.get('status') or ''
            probabilities = {
                int(row['topic_id']): float(row['probability'])
                for row in grouped
                if row.get('status') == 'predicted' and row.get('topic_id') != ''
            }
            dominant = next(
                (int(row['topic_id']) for row in grouped
                 if str(row.get('is_dominant')).lower() in {'true', '1'}),
                None,
            )
            yield {
                'paper_id': paper_id,
                'publication_date': first.get('publication_date') or '',
                'status': status,
                'probabilities': probabilities,
                'dominant_topic': dominant,
            }


def _publication_year(value: object) -> int | None:
    """Extract a plausible four-digit publication year.

    Parameters
    ----------
    value : object
        Publication-date value from corpus metadata.

    Returns
    -------
    int or None
        Year from 1000 through 2999, or ``None`` when no valid year exists.
    """
    match = re.search(r'(?<!\d)(\d{4})(?!\d)', str(value or ''))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1000 <= year <= 2999 else None


def plot_topic_trends(
    trends_path: str | PathLike[str],
    report_path: str | PathLike[str],
    output_path: str | PathLike[str] | None = None,
) -> str:
    """Plot topic prevalence and paper coverage from trend artifacts.

    Parameters
    ----------
    trends_path : str or os.PathLike[str]
        Topic trend CSV produced by :func:`aggregate_topic_trends`.
    report_path : str or os.PathLike[str]
        JSON report describing the aggregation windows and corpus coverage.
    output_path : str, os.PathLike[str], or None, optional
        Destination image. A relative path is resolved inside the trend output
        directory, and a missing suffix defaults to PNG.

    Returns
    -------
    str
        Path to the generated plot.

    Raises
    ------
    OSError
        If trend artifacts cannot be read or the plot cannot be written.
    RuntimeError
        If Matplotlib is unavailable.
    ValueError
        If the trend CSV has no usable topic rows.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, PercentFormatter
    except ImportError as error:
        raise RuntimeError(
            'Plotting topic trends requires matplotlib. Install PaperMiner dependencies '
            'and rerun pm topics trends with --plot.'
        ) from error

    trends_path = Path(trends_path)
    report_path = Path(report_path)
    destination = (
        trends_path.parent / TRENDS_PLOT_FILENAME
        if output_path is None else Path(output_path)
    )
    if output_path is not None and not destination.is_absolute():
        destination = trends_path.parent / destination
    if not destination.suffix:
        destination = destination.with_suffix('.png')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with trends_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Topic trend CSV contains no rows: {trends_path}')
    report = json.loads(report_path.read_text(encoding='utf-8'))

    topic_series = {}
    window_counts = {}
    for row in rows:
        if row.get('mean_probability', '') == '':
            continue
        topic_id = int(row['topic_id'])
        item = {
            'window_start': int(row['window_start']),
            'window_end': int(row['window_end']),
            'is_partial': str(row['is_partial']).lower() in {'true', '1'},
            'mean_probability': float(row['mean_probability']),
            'papers_total': int(row['papers_total']),
            'topic_name': row.get('topic_name') or '',
        }
        topic_series.setdefault(topic_id, []).append(item)
        window_counts[item['window_start']] = item['papers_total']
    if not topic_series:
        raise ValueError(f'Topic trend CSV contains no predicted topic values: {trends_path}')

    figure, (axis, count_axis) = plt.subplots(
        2, 1, figsize=(13.5, 8.2), sharex=True,
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.08},
    )
    color_map = plt.get_cmap('tab10')
    maximum_probability = 0.0
    first_partial = None
    for topic_id in sorted(topic_series):
        series = sorted(topic_series[topic_id], key=lambda item: item['window_start'])
        maximum_probability = max(
            maximum_probability,
            max(item['mean_probability'] for item in series),
        )
        label = (
            f'{topic_id}: {series[0]["topic_name"]}'
            if series[0]['topic_name'] else f'Topic {topic_id}'
        )
        partial_index = next(
            (index for index, item in enumerate(series) if item['is_partial']),
            len(series),
        )
        complete = series[:partial_index]
        color = color_map(topic_id % 10)
        if complete:
            axis.plot(
                [item['window_start'] for item in complete],
                [item['mean_probability'] for item in complete],
                color=color, linewidth=2.3, marker='o', markersize=4.5,
                label=label,
            )
        else:
            axis.plot([], [], color=color, linewidth=2.3, marker='o', label=label)
        if partial_index < len(series):
            partial = series[max(0, partial_index - 1):]
            first_partial = min(
                first_partial if first_partial is not None else partial[0]['window_start'],
                series[partial_index]['window_start'],
            )
            axis.plot(
                [item['window_start'] for item in partial],
                [item['mean_probability'] for item in partial],
                color=color, linewidth=2.0, marker='o', markersize=4,
                linestyle='--', alpha=0.72,
            )

    starts = sorted(window_counts)
    count_axis.bar(
        starts,
        [window_counts[start] for start in starts],
        width=0.72,
        color=['#BBBBBB' if first_partial is not None and start >= first_partial else '#777777'
               for start in starts],
        edgecolor='white',
        linewidth=0.5,
    )
    if first_partial is not None:
        for plot_axis in (axis, count_axis):
            plot_axis.axvspan(
                first_partial - 0.5, max(starts) + 0.5,
                color='#E5E5E5', alpha=0.55, zorder=0,
            )

    for plot_axis in (axis, count_axis):
        plot_axis.grid(axis='y', color='#D9D9D9', linewidth=0.8, alpha=0.75)
        plot_axis.spines[['top', 'right']].set_visible(False)
    bin_size = int(report['bin_size'])
    step_size = int(report['step_size'])
    annual = bin_size == 1 and step_size == 1
    axis.set_title(
        'Annual topic prevalence' if annual else 'Topic prevalence over time',
        loc='left', fontsize=17, weight='bold', pad=18,
    )
    subtitle = (
        'Mean LDA probability by publication year'
        if annual else
        f'Mean LDA probability in {bin_size}-year windows with a {step_size}-year step'
    )
    if first_partial is not None:
        subtitle += '; dashed lines are partial windows'
    axis.text(
        0, 1.015, subtitle, transform=axis.transAxes,
        fontsize=10.5, color='#444444', va='bottom',
    )
    axis.set_ylabel('Mean topic probability')
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylim(0, min(1.0, max(0.4, math.ceil(maximum_probability * 20) / 20 + 0.05)))
    axis.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9.5)
    count_axis.set_ylabel('Papers')
    count_axis.set_xlabel('Publication year' if annual else 'Start year of time window')
    count_axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    count_axis.set_xticks(starts)
    count_axis.tick_params(axis='x', rotation=45)
    count_axis.text(
        1.0, -0.78,
        f'{report["papers_total"]} papers; observed publication years '
        f'{report["observed_start_year"]}–{report["observed_end_year"]}.',
        transform=count_axis.transAxes, ha='right', va='top',
        fontsize=9, color='#555555',
    )
    figure.subplots_adjust(left=0.08, right=0.72, top=0.88, bottom=0.18)

    figure.savefig(
        destination,
        dpi=180 if destination.suffix.lower() == '.png' else None,
        bbox_inches='tight',
    )
    plt.close(figure)
    return str(destination)


def aggregate_topic_trends(
    model_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
    predictions_path: str | PathLike[str] | None = None,
    bin_size: int = 1,
    step_size: int = 1,
    start_year: int | None = None,
    end_year: int | None = None,
    include_partial: bool = True,
    overwrite: bool = False,
    plot: bool | str | PathLike[str] = False,
) -> dict[str, Any]:
    """Aggregate fixed-model topic probabilities into time windows.

    Parameters
    ----------
    model_dir : str or os.PathLike[str]
        Topic-model artifact directory supplying configuration and names.
    output_dir : str or os.PathLike[str]
        Destination directory for trend artifacts.
    predictions_path : str, os.PathLike[str], or None, optional
        Long-form predictions; defaults to the model's training predictions.
    bin_size : int, default=1
        Width of each publication-year window.
    step_size : int, default=1
        Years between consecutive window starts.
    start_year : int or None, optional
        First window start; defaults to the earliest observed year.
    end_year : int or None, optional
        Final observed year included in the configured range.
    include_partial : bool, default=True
        Whether to include trailing windows extending past ``end_year``.
    overwrite : bool, default=False
        Whether known artifacts may be replaced in a nonempty directory.
    plot : bool, str, or os.PathLike[str], default=False
        Generate the default PNG when true, or write to the supplied filename.

    Returns
    -------
    dict[str, Any]
        Artifact paths, window count, missing-date count, and optional plot path.

    Raises
    ------
    FileNotFoundError
        If the prediction CSV is unavailable.
    OSError
        If model or trend artifacts cannot be accessed.
    ValueError
        If window settings or publication dates are unsuitable.
    """
    if bin_size < 1 or step_size < 1:
        raise ValueError('bin_size and step_size must be positive.')
    if step_size > bin_size:
        raise ValueError('step_size must not exceed bin_size.')
    _model, _vectorizer, config, names = load_topic_model(model_dir)
    predictions_path = Path(predictions_path or Path(model_dir) / PREDICTIONS_FILENAME)
    if not predictions_path.exists():
        raise FileNotFoundError(f'Missing topic predictions: {predictions_path}')

    observed_years = []
    missing_dates = 0
    paper_count = 0
    for paper in _prediction_papers(predictions_path):
        paper_count += 1
        year = _publication_year(paper['publication_date'])
        if year is None:
            missing_dates += 1
        else:
            observed_years.append(year)
    if not observed_years:
        raise ValueError('Topic predictions contain no valid publication years.')
    observed_min = min(observed_years)
    observed_max = max(observed_years)
    resolved_start = observed_min if start_year is None else int(start_year)
    resolved_end = observed_max if end_year is None else int(end_year)
    if resolved_start > resolved_end:
        raise ValueError('start_year must not exceed end_year.')

    windows = []
    for window_start in range(resolved_start, resolved_end + 1, step_size):
        window_end = window_start + bin_size - 1
        partial = window_end > resolved_end
        if partial and not include_partial:
            continue
        windows.append((window_start, window_end, partial))
    window_totals = {start: 0 for start, _, _ in windows}
    window_predicted = {start: 0 for start, _, _ in windows}
    aggregates = {
        (start, topic_id): {'probability_sum': 0.0, 'dominant_papers': 0}
        for start, _, _ in windows
        for topic_id in range(config['num_topics'])
    }
    for paper in _prediction_papers(predictions_path):
        year = _publication_year(paper['publication_date'])
        if year is None:
            continue
        matching = [window for window in windows if window[0] <= year <= window[1]]
        for window_start, _, _ in matching:
            window_totals[window_start] += 1
            if paper['status'] != 'predicted' or not paper['probabilities']:
                continue
            window_predicted[window_start] += 1
            for topic_id in range(config['num_topics']):
                aggregate = aggregates[(window_start, topic_id)]
                aggregate['probability_sum'] += paper['probabilities'].get(topic_id, 0.0)
                if paper['dominant_topic'] == topic_id:
                    aggregate['dominant_papers'] += 1

    output_path = _prepare_output_directory(output_dir, overwrite=overwrite)
    trend_path = output_path / TRENDS_CSV_FILENAME
    fields = [
        'window_start', 'window_end', 'is_partial', 'topic_id', 'topic_name',
        'papers_total', 'papers_predicted', 'mean_probability', 'expected_papers',
        'dominant_papers', 'dominant_share',
    ]
    with trend_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for window_start, window_end, partial in windows:
            predicted = window_predicted[window_start]
            for topic_id in range(config['num_topics']):
                aggregate = aggregates[(window_start, topic_id)]
                probability_sum = aggregate['probability_sum']
                writer.writerow({
                    'window_start': window_start,
                    'window_end': window_end,
                    'is_partial': partial,
                    'topic_id': topic_id,
                    'topic_name': names[str(topic_id)],
                    'papers_total': window_totals[window_start],
                    'papers_predicted': predicted,
                    'mean_probability': f'{probability_sum / predicted:.12g}' if predicted else '',
                    'expected_papers': f'{probability_sum:.12g}' if predicted else '',
                    'dominant_papers': aggregate['dominant_papers'] if predicted else '',
                    'dominant_share': (
                        f'{aggregate["dominant_papers"] / predicted:.12g}' if predicted else ''
                    ),
                })
    report = {
        'created_at': _utc_now(),
        'model_id': config['model_id'],
        'predictions_path': str(predictions_path),
        'bin_size': bin_size,
        'step_size': step_size,
        'start_year': resolved_start,
        'end_year': resolved_end,
        'observed_start_year': observed_min,
        'observed_end_year': observed_max,
        'include_partial': include_partial,
        'papers_total': paper_count,
        'papers_missing_or_invalid_date': missing_dates,
        'windows': len(windows),
    }
    report_path = output_path / TRENDS_REPORT_FILENAME
    _write_json(report_path, report)
    requested_plot_path = None
    if plot:
        requested_plot_path = None if plot is True else plot
    plot_path = (
        plot_topic_trends(trend_path, report_path, requested_plot_path)
        if plot else None
    )
    return {
        'output_dir': str(output_path),
        'trends_csv': str(trend_path),
        'report_json': str(report_path),
        'windows': len(windows),
        'papers_missing_or_invalid_date': missing_dates,
        'plot_path': plot_path,
    }


def _topic_set_similarity(
    left_topics: Sequence[Mapping[str, Any]],
    right_topics: Sequence[Mapping[str, Any]],
) -> float:
    """Calculate symmetric best-match similarity between topic sets.

    Parameters
    ----------
    left_topics : Sequence[Mapping[str, Any]]
        First topic collection containing ``top_terms`` lists.
    right_topics : Sequence[Mapping[str, Any]]
        Second topic collection containing ``top_terms`` lists.

    Returns
    -------
    float
        Symmetric mean best-match Jaccard similarity from zero to one.
    """
    left_sets = [set(topic['top_terms']) for topic in left_topics]
    right_sets = [set(topic['top_terms']) for topic in right_topics]

    def directional(source: Sequence[set[str]], targets: Sequence[set[str]]) -> float:
        """Calculate mean best-match similarity in one direction.

        Parameters
        ----------
        source : Sequence[set[str]]
            Topic-term sets to score.
        targets : Sequence[set[str]]
            Candidate topic-term sets for each source topic.

        Returns
        -------
        float
            Mean best-match Jaccard similarity, or zero for no source topics.
        """
        scores = []
        for source_terms in source:
            candidates = [
                len(source_terms & target_terms) / len(source_terms | target_terms)
                for target_terms in targets
                if source_terms or target_terms
            ]
            scores.append(max(candidates, default=0))
        return sum(scores) / len(scores) if scores else 0

    return (directional(left_sets, right_sets) + directional(right_sets, left_sets)) / 2


def _write_model_comparison(
    path: str | PathLike[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write quality metrics for a collection of comparison models.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Destination comparison CSV path.
    rows : Iterable[Mapping[str, Any]]
        Per-model configuration and quality metrics.

    Raises
    ------
    OSError
        If the destination cannot be written.
    """
    fieldnames = [
        'num_topics', 'random_state', 'model_dir', 'documents_used', 'vocabulary_size',
        'perplexity', 'log_likelihood', 'topic_diversity', 'dominant_topic_balance',
        'smallest_dominant_topic', 'largest_dominant_topic', 'mean_seed_stability',
        'vectorization_seconds', 'fitting_seconds', 'cache_size_bytes', 'warnings',
    ]
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                'warnings': ' | '.join(row['warnings']),
            })


def compare_topic_models(db_path: str | PathLike[str],
                         output_dir: str | PathLike[str],
                         topic_counts: Iterable[int] = (5, 10),
                         random_states: Iterable[int] = (0, 1),
                         text_fields: Iterable[str] = ('title', 'abstract'),
                         min_df: int = 2,
                         max_df: float = 0.95,
                         max_features: int = 20000,
                         learning_method: str = 'online',
                         max_iter: int = 20,
                         top_terms: int = 15,
                         representative_papers: int = 5,
                         stopwords_file: str | PathLike[str] | None = None,
                         ngram_max: int = 2,
                         overwrite: bool = False,
                         streaming: bool = True,
                         batch_size: int = DEFAULT_BATCH_SIZE,
                         cache_dir: str | PathLike[str] | None = None,
                         evaluation_sample_size: int = DEFAULT_EVALUATION_SAMPLE_SIZE) -> dict[str, Any]:
    """Train and compare topic counts and random seeds on one corpus.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the source SQLite paper corpus.
    output_dir : str or os.PathLike[str]
        Directory for comparison models and reports.
    topic_counts : Iterable[int], default=(5, 10)
        Distinct topic counts to evaluate.
    random_states : Iterable[int], default=(0, 1)
        Distinct model seeds to evaluate.
    text_fields : Iterable[str], default=('title', 'abstract')
        Corpus fields combined into each modeling document.
    min_df : int, default=2
        Minimum number of documents in which a feature must occur.
    max_df : float, default=0.95
        Maximum fraction of documents in which a feature may occur.
    max_features : int, default=20000
        Maximum retained vocabulary size.
    learning_method : {'batch', 'online'}, default='online'
        Scikit-learn LDA learning strategy.
    max_iter : int, default=20
        Maximum LDA training iterations.
    top_terms : int, default=15
        Number of terms exported for each topic.
    representative_papers : int, default=5
        Number of high-probability papers exported per topic.
    stopwords_file : str, os.PathLike[str], or None, optional
        File containing domain-specific stopwords.
    ngram_max : {1, 2}, default=2
        Maximum feature n-gram size.
    overwrite : bool, default=False
        Whether nonempty comparison directories may be reused.
    streaming : bool, default=True
        Whether every model trains from one reusable disk-backed corpus cache.
    batch_size : int, default=128
        Maximum documents processed in each streaming batch.
    cache_dir : str, os.PathLike[str], or None, optional
        Parent directory for the temporary comparison cache.
    evaluation_sample_size : int, default=10000
        Maximum deterministic document sample used for streaming fit metrics.

    Returns
    -------
    dict[str, Any]
        Comparison paths, model count, and per-model metrics.

    Raises
    ------
    OSError
        If inputs or artifacts cannot be read or written.
    ValueError
        If the comparison grid or a training configuration is invalid.
    """
    counts = tuple(dict.fromkeys(int(value) for value in topic_counts))
    seeds = tuple(dict.fromkeys(int(value) for value in random_states))
    if not counts or any(value < 2 for value in counts):
        raise ValueError('topic_counts must contain values of at least 2.')
    if not seeds:
        raise ValueError('At least one random state is required.')
    if len(counts) * len(seeds) < 2:
        raise ValueError('Model comparison requires at least two topic-count and seed combinations.')
    if streaming and learning_method != 'online':
        raise ValueError('Streaming comparison requires learning_method="online"; use in-memory mode for batch LDA.')

    fields = _validate_text_fields(text_fields)
    comparison_path = _prepare_output_directory(output_dir, overwrite)
    model_records = []
    summaries = []
    temporary = None
    try:
        if streaming:
            if cache_dir is not None:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
            temporary = tempfile.TemporaryDirectory(
                prefix='paperminer-topics-compare-', dir=cache_dir
            )
            prepared = _prepare_streaming_corpus(
                db_path, fields, load_domain_stopwords(stopwords_file), ngram_max,
                min_df, max_df, max_features, batch_size,
                evaluation_sample_size, temporary.name,
            )
            documents = None
        else:
            prepared = None
            documents = load_topic_documents(db_path, fields)
        for num_topics in counts:
            for random_state in seeds:
                model_dir = comparison_path / f'topics-{num_topics}_seed-{random_state}'
                summary = train_topic_model(
                    db_path,
                    model_dir,
                    num_topics=num_topics,
                    text_fields=fields,
                    min_df=min_df,
                    max_df=max_df,
                    max_features=max_features,
                    learning_method=learning_method,
                    max_iter=max_iter,
                    random_state=random_state,
                    top_terms=top_terms,
                    representative_papers=representative_papers,
                    stopwords_file=stopwords_file,
                    ngram_max=ngram_max,
                    overwrite=overwrite,
                    emit_warnings=False,
                    documents=documents,
                    streaming=streaming,
                    batch_size=batch_size,
                    cache_dir=cache_dir,
                    evaluation_sample_size=evaluation_sample_size,
                    _prepared_streaming=prepared,
                )
                report = summary['report']
                record = {
                    'num_topics': num_topics,
                    'random_state': random_state,
                    'model_dir': str(model_dir),
                    'documents_used': report['documents_used'],
                    'vocabulary_size': report['vocabulary_size'],
                    'perplexity': report['perplexity'],
                    'log_likelihood': report['log_likelihood'],
                    'topic_diversity': report['topic_diversity'],
                    'dominant_topic_balance': report['dominant_topic_balance'],
                    'smallest_dominant_topic': report['smallest_dominant_topic'],
                    'largest_dominant_topic': report['largest_dominant_topic'],
                    'mean_seed_stability': '',
                    'vectorization_seconds': report['vectorization_seconds'],
                    'fitting_seconds': report['fitting_seconds'],
                    'cache_size_bytes': report.get('cache_size_bytes', ''),
                    'warnings': report['warnings'],
                }
                model_records.append(record)
                summaries.append(summary)
    finally:
        if temporary is not None:
            temporary.cleanup()

    for num_topics in counts:
        matching = [summary for summary in summaries if summary['config']['num_topics'] == num_topics]
        pair_scores = [
            _topic_set_similarity(left['topics'], right['topics'])
            for left, right in combinations(matching, 2)
        ]
        mean_stability = sum(pair_scores) / len(pair_scores) if pair_scores else ''
        for record in model_records:
            if record['num_topics'] == num_topics:
                record['mean_seed_stability'] = mean_stability

    _write_model_comparison(comparison_path / COMPARISON_CSV_FILENAME, model_records)
    comparison_report = {
        'created_at': _utc_now(),
        'topic_counts': list(counts),
        'random_states': list(seeds),
        'streaming': streaming,
        'batch_size': batch_size,
        'models': model_records,
    }
    _write_json(comparison_path / COMPARISON_JSON_FILENAME, comparison_report)
    return {
        'output_dir': str(comparison_path),
        'models_trained': len(model_records),
        'models': model_records,
        'comparison_csv': str(comparison_path / COMPARISON_CSV_FILENAME),
        'comparison_json': str(comparison_path / COMPARISON_JSON_FILENAME),
    }
