"""Train, inspect, name, and apply reproducible LDA topic models.

Topic models are stored outside the paper corpus as versioned artifacts. The
corpus remains the source of paper metadata and text assets, while predictions
are exported in a long CSV format suitable for later trend and filter stages.
"""

import csv
import hashlib
import html
import json
import math
import re
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from paperscraper.corpus import connect, latest_assets, paper_rows


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
ARTIFACT_VERSION = 1
TOKEN_PATTERN = r'(?u)\b[a-z][a-z0-9]{1,}\b'
TOKEN_RE = re.compile(TOKEN_PATTERN)
URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)
DOI_RE = re.compile(r'\b10\.\d{4,9}/\S+', re.IGNORECASE)


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

    def __init__(self, domain_stopwords=(), ngram_max=2):
        """Initialize the topic analyzer.

        Parameters
        ----------
        domain_stopwords : iterable of str, optional
            Domain-specific words to omit as standalone features.
        ngram_max : int, default=2
            Maximum n-gram size to emit.
        """
        self.domain_stopwords = frozenset(domain_stopwords)
        self.ngram_max = ngram_max

    def __call__(self, document):
        """Extract topic-model features from a document.

        Parameters
        ----------
        document : str
            Normalized document text to analyze.

        Returns
        -------
        list of str
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


def _utc_now():
    """Return a stable UTC timestamp for model metadata.

    Returns
    -------
    str
        ISO-8601 UTC timestamp with second precision.
    """
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def normalize_topic_text(value):
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


def load_domain_stopwords(stopwords_file):
    """Load corpus-specific stopwords from a text file.

    Parameters
    ----------
    stopwords_file : str, pathlib.Path, or None
        File containing one word per non-comment line, or ``None``.

    Returns
    -------
    list of str
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


def _validate_text_fields(text_fields):
    """Validate and deduplicate topic-model text fields.

    Parameters
    ----------
    text_fields : iterable of str
        Requested corpus text fields.

    Returns
    -------
    tuple of str
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


def load_topic_documents(db_path, text_fields=('title', 'abstract')):
    """Load normalized topic-model documents from a corpus.

    Parameters
    ----------
    db_path : str or pathlib.Path
        Path to the SQLite paper corpus.
    text_fields : iterable of str, default=('title', 'abstract')
        Metadata and asset fields to combine into each document.

    Returns
    -------
    list of dict
        Paper metadata, normalized text, and token counts.

    Raises
    ------
    ValueError
        If any requested text field is unsupported.
    """
    fields = _validate_text_fields(text_fields)
    asset_roles = [field for field in fields if field in {'abstract', 'text'}]
    with connect(db_path) as conn:
        papers = paper_rows(conn)
        assets = latest_assets(conn, asset_roles)

    documents = []
    for paper in papers:
        pieces = []
        for field in fields:
            if field == 'title':
                value = paper.get('title') or ''
            else:
                asset = assets.get((paper['paper_id'], field))
                value = asset['content'].decode('utf-8', errors='replace') if asset else ''
            if str(value).strip():
                pieces.append(str(value))
        text = normalize_topic_text(' '.join(pieces))
        documents.append({
            'paper_id': paper['paper_id'],
            'doi': paper.get('doi') or '',
            'title': paper.get('title') or '',
            'publication_date': paper.get('publication_date') or '',
            'text': text,
            'token_count': len(TOKEN_RE.findall(text)),
        })
    return documents


def _median(values):
    """Calculate the median of numeric values.

    Parameters
    ----------
    values : sequence of numbers
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


def assess_topic_corpus(documents, num_topics):
    """Assess corpus quality for LDA topic modeling.

    Parameters
    ----------
    documents : sequence of dict
        Topic documents containing ``token_count`` values.
    num_topics : int
        Requested number of latent topics.

    Returns
    -------
    dict
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


def _emit_warnings(messages):
    """Emit corpus diagnostics through Python's warnings interface.

    Parameters
    ----------
    messages : iterable of str
        Warning messages to emit.
    """
    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=3)


def _corpus_fingerprint(documents):
    """Build a deterministic fingerprint for topic documents.

    Parameters
    ----------
    documents : iterable of dict
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


def _topic_rows(model, vectorizer, top_terms):
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
    list of dict
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


def _write_json(path, value):
    """Write deterministic, readable JSON metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination JSON path.
    value : object
        JSON-serializable value to write.

    Raises
    ------
    OSError
        If the destination cannot be written.
    TypeError
        If ``value`` is not JSON serializable.
    """
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _write_topics(path, topic_rows):
    """Write manually nameable topic summaries.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV path.
    topic_rows : iterable of dict
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


def _refresh_csv_topic_names(path, names):
    """Refresh names in an existing topic artifact CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Artifact CSV to update when it exists.
    names : mapping of str to str
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


def _topic_names(model_dir, num_topics):
    """Load manual topic names from a model artifact.

    Parameters
    ----------
    model_dir : str or pathlib.Path
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


def _write_predictions(path, documents, distributions, names, included_indices):
    """Write long-form topic predictions and skipped-paper states.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV path.
    documents : sequence of dict
        Paper documents in output order.
    distributions : iterable of array-like
        Topic probability vectors for included documents.
    names : mapping of str to str
        Manual names keyed by string topic ID.
    included_indices : iterable of int
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


def _write_representatives(path, documents, distributions, names, count):
    """Write representative papers for manual topic interpretation.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV path.
    documents : sequence of dict
        Documents corresponding to distribution rows.
    distributions : numpy.ndarray
        Per-document topic probabilities.
    names : mapping of str to str
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


def _prepare_output_directory(output_dir, overwrite):
    """Prepare a topic-model artifact directory.

    Parameters
    ----------
    output_dir : str or pathlib.Path
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


def _model_quality_metrics(model, matrix, distributions, topic_rows):
    """Calculate diagnostics for a fitted topic model.

    Parameters
    ----------
    model : sklearn.decomposition.LatentDirichletAllocation
        Fitted LDA model.
    matrix : scipy.sparse.spmatrix
        Document-term matrix used to fit the model.
    distributions : numpy.ndarray
        Per-document topic probabilities.
    topic_rows : sequence of dict
        Extracted topic terms.

    Returns
    -------
    dict
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


def train_topic_model(db_path,
                      output_dir,
                      num_topics=10,
                      text_fields=('title', 'abstract'),
                      min_df=2,
                      max_df=0.95,
                      max_features=20000,
                      learning_method='online',
                      max_iter=20,
                      random_state=0,
                      top_terms=15,
                      representative_papers=5,
                      stopwords_file=None,
                      ngram_max=2,
                      overwrite=False,
                      emit_warnings=True,
                      documents=None):
    """Train and persist an LDA model and its inspection artifacts.

    Parameters
    ----------
    db_path : str or pathlib.Path
        Path to the source SQLite paper corpus.
    output_dir : str or pathlib.Path
        Directory in which to store the model artifacts.
    num_topics : int, default=10
        Number of latent topics to fit.
    text_fields : iterable of str, default=('title', 'abstract')
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
    random_state : int, default=0
        Seed controlling model initialization.
    top_terms : int, default=15
        Number of terms exported for each topic.
    representative_papers : int, default=5
        Number of high-probability papers exported per topic.
    stopwords_file : str, pathlib.Path, or None, optional
        File containing domain-specific stopwords.
    ngram_max : {1, 2}, default=2
        Maximum feature n-gram size.
    overwrite : bool, default=False
        Whether to reuse a nonempty artifact directory.
    emit_warnings : bool, default=True
        Whether to emit heuristic corpus-quality warnings.
    documents : sequence of dict or None, optional
        Preloaded documents used instead of reading ``db_path``.

    Returns
    -------
    dict
        Artifact paths, configuration, quality report, fingerprint, and topics.

    Raises
    ------
    OSError
        If an input or artifact file cannot be read or written.
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

    domain_stopwords = load_domain_stopwords(stopwords_file)
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
        'sklearn_version': sklearn.__version__,
    }
    fingerprint = {
        'algorithm': 'sha256-paper-id-and-normalized-text-v1',
        'sha256': _corpus_fingerprint(training_documents),
        'documents': len(training_documents),
    }
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


def load_topic_model(model_dir):
    """Load and validate a trusted local LDA artifact.

    Parameters
    ----------
    model_dir : str or pathlib.Path
        Topic-model artifact directory.

    Returns
    -------
    tuple
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


def topic_descriptions(model_dir):
    """Load descriptions and representative papers for each topic.

    Parameters
    ----------
    model_dir : str or pathlib.Path
        Topic-model artifact directory.

    Returns
    -------
    list of dict
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


def set_topic_name(model_dir, topic_id, topic_name):
    """Set a manual topic name and refresh artifact exports.

    Parameters
    ----------
    model_dir : str or pathlib.Path
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


def predict_topic_model(model_dir, db_path, output_path):
    """Apply a saved topic model and export long-form scores.

    Parameters
    ----------
    model_dir : str or pathlib.Path
        Topic-model artifact directory.
    db_path : str or pathlib.Path
        Path to the SQLite paper corpus to score.
    output_path : str or pathlib.Path
        Destination prediction CSV.

    Returns
    -------
    dict
        Counts of total, predicted, and skipped papers plus the output path.

    Raises
    ------
    OSError
        If model artifacts or the prediction destination cannot be accessed.
    ValueError
        If the model artifact is unsupported or the corpus contains no papers.
    """
    model, vectorizer, config, names = load_topic_model(model_dir)
    documents = load_topic_documents(db_path, config['text_fields'])
    if not documents:
        raise ValueError('The corpus contains no papers to predict.')
    matrix = vectorizer.transform(document['text'] for document in documents)
    included_mask = matrix.getnnz(axis=1) > 0
    included_indices = [index for index, included in enumerate(included_mask) if included]
    distributions = model.transform(matrix[included_mask]) if included_indices else []
    _write_predictions(output_path, documents, distributions, names, included_indices)
    return {
        'papers_total': len(documents),
        'papers_predicted': len(included_indices),
        'papers_without_vocabulary_terms': len(documents) - len(included_indices),
        'output_path': str(output_path),
    }


def _topic_set_similarity(left_topics, right_topics):
    """Calculate symmetric best-match similarity between topic sets.

    Parameters
    ----------
    left_topics : sequence of dict
        First topic collection containing ``top_terms`` lists.
    right_topics : sequence of dict
        Second topic collection containing ``top_terms`` lists.

    Returns
    -------
    float
        Symmetric mean best-match Jaccard similarity from zero to one.
    """
    left_sets = [set(topic['top_terms']) for topic in left_topics]
    right_sets = [set(topic['top_terms']) for topic in right_topics]

    def directional(source, targets):
        """Calculate mean best-match similarity in one direction.

        Parameters
        ----------
        source : sequence of set
            Topic-term sets to score.
        targets : sequence of set
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


def _write_model_comparison(path, rows):
    """Write quality metrics for a collection of comparison models.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination comparison CSV path.
    rows : iterable of dict
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
        'vectorization_seconds', 'fitting_seconds', 'warnings',
    ]
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                'warnings': ' | '.join(row['warnings']),
            })


def compare_topic_models(db_path,
                         output_dir,
                         topic_counts=(5, 10),
                         random_states=(0, 1),
                         text_fields=('title', 'abstract'),
                         min_df=2,
                         max_df=0.95,
                         max_features=20000,
                         learning_method='online',
                         max_iter=20,
                         top_terms=15,
                         representative_papers=5,
                         stopwords_file=None,
                         ngram_max=2,
                         overwrite=False):
    """Train and compare topic counts and random seeds on one corpus.

    Parameters
    ----------
    db_path : str or pathlib.Path
        Path to the source SQLite paper corpus.
    output_dir : str or pathlib.Path
        Directory for comparison models and reports.
    topic_counts : iterable of int, default=(5, 10)
        Distinct topic counts to evaluate.
    random_states : iterable of int, default=(0, 1)
        Distinct model seeds to evaluate.
    text_fields : iterable of str, default=('title', 'abstract')
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
    stopwords_file : str, pathlib.Path, or None, optional
        File containing domain-specific stopwords.
    ngram_max : {1, 2}, default=2
        Maximum feature n-gram size.
    overwrite : bool, default=False
        Whether nonempty comparison directories may be reused.

    Returns
    -------
    dict
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

    fields = _validate_text_fields(text_fields)
    documents = load_topic_documents(db_path, fields)
    comparison_path = _prepare_output_directory(output_dir, overwrite)
    model_records = []
    summaries = []
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
                'warnings': report['warnings'],
            }
            model_records.append(record)
            summaries.append(summary)

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
