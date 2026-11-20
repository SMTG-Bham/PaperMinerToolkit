"""Lock down topic export formats and numerical diagnostics independently of fitting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from paperminertoolkit.workflows import topics


def test_topic_prediction_export_contract_across_all_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep UTF-8, quoting, precision, first-topic ties and missing-vocabulary rows."""
    documents = [
        {'paper_id': 'α', 'doi': '10.1/a', 'title': 'A, title', 'publication_date': '2020'},
        {'paper_id': 'empty', 'doi': '', 'title': 'No terms', 'publication_date': ''},
        {'paper_id': 'b', 'doi': '', 'title': 'Battery', 'publication_date': '2021'},
    ]
    distributions = np.array([[0.5, 0.5], [1 / 3, 2 / 3]])
    names = {'0': 'Ions', '1': 'Cells'}
    expected = (
        'paper_id,doi,title,publication_date,topic_id,topic_name,probability,is_dominant,status\r\n'
        'α,10.1/a,"A, title",2020,0,Ions,0.5,True,predicted\r\n'
        'α,10.1/a,"A, title",2020,1,Cells,0.5,False,predicted\r\n'
        'empty,,No terms,,,,,,no_vocabulary_terms\r\n'
        'b,,Battery,2021,0,Ions,0.333333333333,False,predicted\r\n'
        'b,,Battery,2021,1,Cells,0.666666666667,True,predicted\r\n'
    ).encode('utf-8')
    in_memory = tmp_path / 'memory.csv'
    topics._write_predictions(in_memory, documents, distributions, names, [0, 2])

    class FixedModel:
        """Return known probabilities so serialization tests do not depend on LDA."""

        n_components = 2

        def transform(self, matrix: sparse.spmatrix) -> np.ndarray:
            """Assert that the empty-vocabulary row was excluded before inference."""
            assert matrix.shape == (2, 1)
            return distributions

    model = FixedModel()
    monkeypatch.setattr(topics, '_cached_batches', lambda prepared: iter([
        (sparse.csr_matrix([[1], [0], [1]]), documents),
    ]))
    streamed = tmp_path / 'streamed.csv'
    metrics = topics._write_streamed_outputs(
        {}, model, names, streamed, tmp_path / 'representatives.csv', 1,
    )
    assert metrics['dominant_topic_counts'] == [1, 1]
    assert metrics['dominant_topic_balance'] == 1
    assert metrics['papers_predicted'] == 2

    monkeypatch.setattr(topics, 'load_topic_model', lambda path: (model, None, {}, names))
    monkeypatch.setattr(topics, '_iter_topic_predictions', lambda *args: iter([
        {'document': document, 'distribution': distribution}
        for document, distribution in zip(documents, [distributions[0], None, distributions[1]])
    ]))
    predicted = tmp_path / 'predicted.csv'
    summary = topics.predict_topic_model(tmp_path, tmp_path / 'unused.db', predicted)
    assert summary['papers_predicted'] == 2
    assert summary['papers_without_vocabulary_terms'] == 1
    for path in (in_memory, streamed, predicted):
        assert path.read_bytes() == expected


def test_representative_selection_retains_distinct_tie_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep NumPy ranking for in-memory output and first retained streaming ties."""
    documents = [
        {'paper_id': name, 'title': name, 'doi': '', 'publication_date': ''}
        for name in ('first', 'second')
    ]
    distributions = np.array([[0.5, 0.5], [0.5, 0.5]])
    names = {'0': '', '1': ''}

    class TiedModel:
        """Provide a tie at the representative cutoff."""

        n_components = 2

        def transform(self, matrix: sparse.spmatrix) -> np.ndarray:
            """Return tied probabilities for both cached documents."""
            return distributions

    monkeypatch.setattr(topics, '_cached_batches', lambda prepared: iter([
        (sparse.csr_matrix([[1], [1]]), documents),
    ]))
    streamed = tmp_path / 'streamed.csv'
    topics._write_streamed_outputs({}, TiedModel(), names, tmp_path / 'scores.csv', streamed, 1)
    in_memory = tmp_path / 'memory.csv'
    topics._write_representatives(in_memory, documents, distributions, names, 1)
    with streamed.open() as handle:
        assert [row['paper_id'] for row in csv.DictReader(handle)] == ['first', 'first']
    with in_memory.open() as handle:
        assert [row['paper_id'] for row in csv.DictReader(handle)] == ['second', 'second']


def test_topic_fingerprint_retains_version_two_encoding_and_id_order() -> None:
    """Preserve a pre-refactor digest including Unicode and empty normalized text."""
    documents: list[dict[str, Any]] = [
        {'paper_id': 'β', 'text': 'cathode café'}, {'paper_id': 'a', 'text': ''},
    ]
    expected = '1350a1d660948ec5a3c9f30f4433e52bf6fc821a661dea4aacf1d115892c8f9e'
    assert topics._corpus_fingerprint(documents) == expected
    assert topics._corpus_fingerprint(reversed(documents)) == expected


@pytest.mark.parametrize(('counts', 'balance'), [([0, 0], 0), ([8, 0], 0), ([4, 4], 1), ([3, 1], 0.8112781244591328)])
def test_dominant_topic_entropy_has_known_values(counts: list[int], balance: float) -> None:
    """Check dimensionless normalized entropy against independently known values."""
    metrics = topics._dominant_topic_metrics(counts)
    assert metrics['dominant_topic_balance'] == pytest.approx(balance)
    assert metrics['dominant_topic_counts'] == counts
    assert metrics['smallest_dominant_topic'] == min(counts)
    assert metrics['largest_dominant_topic'] == max(counts)


@pytest.mark.parametrize('streaming', [0, np.bool_(False), 1, np.bool_(True)])
def test_training_mode_coercion_preserves_immutable_model_identity(
    tmp_path: Path, streaming: int | np.bool_,
) -> None:
    """Serialize mode as a Python bool as the original separate branches did."""
    from tests.workflows.test_topics import build_topic_corpus

    db_path = tmp_path / 'papers.db'
    build_topic_corpus(db_path, papers_per_theme=3)
    options = {
        'num_topics': 3, 'min_df': 1, 'max_df': 1.0, 'max_iter': 1,
        'batch_size': 3, 'evaluation_sample_size': 4, 'emit_warnings': False,
    }
    reference = topics.train_topic_model(
        db_path, tmp_path / 'reference', streaming=bool(streaming), **options,
    )
    candidate = topics.train_topic_model(
        db_path, tmp_path / 'candidate', streaming=streaming, **options,
    )
    assert candidate['config']['streaming'] is bool(streaming)
    assert candidate['config']['model_id'] == reference['config']['model_id']
    assert candidate['fingerprint'] == reference['fingerprint']
