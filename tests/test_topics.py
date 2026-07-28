"""Tests for reproducible LDA training, inspection, naming, and prediction."""

import csv
import json
from collections import defaultdict

import pytest

from paperscraper import corpus, topics


THEME_TEXTS = [
    'lithium solid electrolyte ionic conductivity garnet battery interface transport',
    'photocatalysis catalyst hydrogen oxygen reaction sunlight water semiconductor surface',
    'flexible transistor electronics synaptic computing polymer device circuit reservoir',
]


def build_topic_corpus(db_path, papers_per_theme=8):
    """Create a small corpus with three deliberately distinct themes."""
    with corpus.connect(db_path) as conn:
        for theme_id, theme_text in enumerate(THEME_TEXTS):
            for index in range(papers_per_theme):
                paper = {
                    'paper_id': f'theme:{theme_id}:{index}',
                    'doi': f'10.1000/{theme_id}.{index}',
                    'title': f'{theme_text.split()[0].title()} study {index}',
                    'publication_date': f'{2000 + theme_id * 10 + index % 5}-01-01',
                }
                corpus.add_asset(
                    conn,
                    paper,
                    f'{theme_text}. {theme_text}. Detailed scientific investigation and measurements.',
                    role='abstract',
                    kind='text',
                    mime_type='text/plain',
                )


def train_small_model(db_path, model_dir):
    """Train a fast deterministic model for topic tests."""
    return topics.train_topic_model(
        db_path,
        model_dir,
        num_topics=3,
        text_fields=('title', 'abstract'),
        min_df=1,
        max_df=1.0,
        max_features=1000,
        learning_method='batch',
        max_iter=5,
        random_state=7,
        top_terms=6,
        representative_papers=2,
    )


def test_normalize_topic_text_removes_markup_urls_and_dois():
    """Normalize boilerplate while retaining words and chemical formulae."""
    value = '<p>Li₇La₃Zr₂O₁₂ transport</p> https://example.test 10.1000/example'

    normalized = topics.normalize_topic_text(value)

    assert normalized == 'li7la3zr2o12 transport'


def test_load_topic_documents_combines_selected_metadata_and_assets(tmp_path):
    """Build one normalized document from a title and its latest abstract asset."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        paper = {'paper_id': 'paper:1', 'title': 'Solid Electrolyte', 'publication_date': '2024'}
        corpus.add_asset(conn, paper, 'Lithium conductivity', role='abstract', kind='text', mime_type='text/plain')
        corpus.add_asset(conn, paper, 'Full text words', role='text', kind='text', mime_type='text/plain')

    documents = topics.load_topic_documents(db_path, ('title', 'abstract'))

    assert len(documents) == 1
    assert documents[0]['paper_id'] == 'paper:1'
    assert documents[0]['text'] == 'solid electrolyte lithium conductivity'
    assert 'full text' not in documents[0]['text']


def test_train_topic_model_writes_reusable_manual_inspection_artifacts(tmp_path):
    """Train LDA, persist its complete artifact, and export normalized probabilities."""
    db_path = tmp_path / 'papers.db'
    model_dir = tmp_path / 'model'
    build_topic_corpus(db_path)

    with pytest.warns(UserWarning) as warning_records:
        summary = train_small_model(db_path, model_dir)

    assert any('Small topic-model corpus' in str(record.message) for record in warning_records)

    expected_files = {
        topics.MODEL_FILENAME,
        topics.VECTORIZER_FILENAME,
        topics.CONFIG_FILENAME,
        topics.REPORT_FILENAME,
        topics.FINGERPRINT_FILENAME,
        topics.TOPICS_FILENAME,
        topics.TOPIC_NAMES_FILENAME,
        topics.REPRESENTATIVES_FILENAME,
        topics.PREDICTIONS_FILENAME,
    }
    assert expected_files <= {path.name for path in model_dir.iterdir()}
    assert summary['report']['documents_used'] == 24
    assert summary['report']['vocabulary_size'] > 3
    assert len(summary['fingerprint']['sha256']) == 64

    descriptions = topics.topic_descriptions(model_dir)
    assert len(descriptions) == 3
    assert all(len(topic['top_terms']) == 6 for topic in descriptions)
    assert all(len(topic['representative_papers']) == 2 for topic in descriptions)
    assert all(topic['topic_name'] == '' for topic in descriptions)

    probabilities = defaultdict(float)
    with (model_dir / topics.PREDICTIONS_FILENAME).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24 * 3
    for row in rows:
        probabilities[row['paper_id']] += float(row['probability'])
    assert all(total == pytest.approx(1.0) for total in probabilities.values())


def test_set_topic_name_updates_manual_metadata_and_existing_exports(tmp_path):
    """Manual names remain separate from the fitted model and propagate to exports."""
    db_path = tmp_path / 'papers.db'
    model_dir = tmp_path / 'model'
    build_topic_corpus(db_path)
    with pytest.warns(UserWarning):
        train_small_model(db_path, model_dir)

    names = topics.set_topic_name(model_dir, 0, 'solid electrolyte research')

    assert names['0'] == 'solid electrolyte research'
    assert topics.topic_descriptions(model_dir)[0]['topic_name'] == 'solid electrolyte research'
    stored_names = json.loads((model_dir / topics.TOPIC_NAMES_FILENAME).read_text())
    assert stored_names['0'] == 'solid electrolyte research'
    with (model_dir / topics.PREDICTIONS_FILENAME).open(encoding='utf-8', newline='') as handle:
        topic_zero_rows = [row for row in csv.DictReader(handle) if row['topic_id'] == '0']
    assert topic_zero_rows
    assert all(row['topic_name'] == 'solid electrolyte research' for row in topic_zero_rows)


def test_predict_topic_model_marks_documents_without_known_vocabulary(tmp_path):
    """Prediction exports distinguish scored papers from out-of-vocabulary papers."""
    training_db = tmp_path / 'training.db'
    prediction_db = tmp_path / 'prediction.db'
    model_dir = tmp_path / 'model'
    output_path = tmp_path / 'predictions.csv'
    build_topic_corpus(training_db)
    with pytest.warns(UserWarning):
        train_small_model(training_db, model_dir)
    with corpus.connect(prediction_db) as conn:
        corpus.add_asset(
            conn,
            {'paper_id': 'known', 'title': 'Lithium electrolyte'},
            'solid ionic conductivity garnet',
            role='abstract',
            kind='text',
            mime_type='text/plain',
        )
        corpus.upsert_paper(conn, {'paper_id': 'unknown', 'title': 'quasar nebula cosmology'})

    summary = topics.predict_topic_model(model_dir, prediction_db, output_path)

    assert summary == {
        'papers_total': 2,
        'papers_predicted': 1,
        'papers_without_vocabulary_terms': 1,
        'output_path': str(output_path),
    }
    with output_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len([row for row in rows if row['paper_id'] == 'known']) == 3
    assert [row['status'] for row in rows if row['paper_id'] == 'unknown'] == ['no_vocabulary_terms']


def test_topic_training_rejects_invalid_corpora_and_accidental_overwrite(tmp_path):
    """Fail clearly for too many topics and non-empty artifact directories."""
    db_path = tmp_path / 'papers.db'
    model_dir = tmp_path / 'model'
    build_topic_corpus(db_path, papers_per_theme=1)

    with pytest.raises(ValueError, match='at least as many usable documents'):
        topics.train_topic_model(db_path, model_dir, num_topics=4, min_df=1)

    build_topic_corpus(db_path, papers_per_theme=4)
    model_dir.mkdir()
    (model_dir / 'keep.txt').write_text('existing')
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match='not empty'):
            topics.train_topic_model(
                db_path,
                model_dir,
                num_topics=3,
                min_df=1,
                max_df=1.0,
                max_iter=2,
            )
