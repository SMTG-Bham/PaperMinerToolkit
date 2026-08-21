"""Tests for reproducible LDA training, inspection, naming, and prediction."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from paperscraper import corpus, filtering, topics


THEME_TEXTS = [
    'lithium solid electrolyte ionic conductivity garnet battery interface transport',
    'photocatalysis catalyst hydrogen oxygen reaction sunlight water semiconductor surface',
    'flexible transistor electronics synaptic computing polymer device circuit reservoir',
]


def build_topic_corpus(db_path: Path, papers_per_theme: int = 8) -> None:
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


def train_small_model(db_path: Path, model_dir: Path) -> dict[str, Any]:
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
        streaming=False,
    )


def test_normalize_topic_text_removes_markup_urls_and_dois() -> None:
    """Normalize boilerplate while retaining words and chemical formulae."""
    value = '<p>Li₇La₃Zr₂O₁₂ transport</p> https://example.test 10.1000/example'

    normalized = topics.normalize_topic_text(value)

    assert normalized == 'li7la3zr2o12 transport'


def test_domain_stopwords_preserve_meaningful_bigrams(tmp_path: Path) -> None:
    """Suppress generic unigrams without removing phrases that contain them."""
    stopwords_path = tmp_path / 'stopwords.txt'
    stopwords_path.write_text(
        '# Corpus-specific generic terms\nLithium\nbattery\nstudy\nperformance\nlithium\n',
        encoding='utf-8',
    )

    stopwords = topics.load_domain_stopwords(stopwords_path)
    analyzer = topics.TopicAnalyzer(stopwords, ngram_max=2)
    features = analyzer('lithium metal battery performance. ionic conductivity and solid electrolyte.')

    assert stopwords == ['battery', 'lithium', 'performance', 'study']
    assert 'lithium' not in features
    assert 'battery' not in features
    assert 'performance' not in features
    assert 'lithium_metal' in features
    assert 'ionic_conductivity' in features
    assert 'solid_electrolyte' in features
    assert 'battery_performance' not in features


def test_domain_stopwords_require_one_word_per_line(tmp_path: Path) -> None:
    """Reject ambiguous multi-word stopword entries with a useful location."""
    path = tmp_path / 'stopwords.txt'
    path.write_text('solid electrolyte\n', encoding='utf-8')

    with pytest.raises(ValueError, match=r'stopwords\.txt:1: expected exactly one word'):
        topics.load_domain_stopwords(path)


def test_load_topic_documents_combines_selected_metadata_and_assets(tmp_path: Path) -> None:
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


def test_train_topic_model_writes_reusable_manual_inspection_artifacts(tmp_path: Path) -> None:
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
        topics.STOPWORDS_FILENAME,
    }
    assert expected_files <= {path.name for path in model_dir.iterdir()}
    assert summary['report']['documents_used'] == 24
    assert summary['report']['vocabulary_size'] > 3
    assert summary['report']['perplexity'] > 0
    assert 0 < summary['report']['topic_diversity'] <= 1
    assert 0 <= summary['report']['dominant_topic_balance'] <= 1
    assert summary['report']['vectorization_seconds'] >= 0
    assert summary['report']['fitting_seconds'] >= 0
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


def test_set_topic_name_updates_manual_metadata_and_existing_exports(tmp_path: Path) -> None:
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


def test_predict_topic_model_marks_documents_without_known_vocabulary(tmp_path: Path) -> None:
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


def test_topic_training_rejects_invalid_corpora_and_accidental_overwrite(tmp_path: Path) -> None:
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


def test_compare_topic_models_exports_counts_seeds_metrics_and_stability(tmp_path: Path) -> None:
    """Train a comparison grid from one corpus and summarize model diagnostics."""
    db_path = tmp_path / 'papers.db'
    output_dir = tmp_path / 'comparison'
    stopwords_path = tmp_path / 'stopwords.txt'
    stopwords_path.write_text('study\n', encoding='utf-8')
    build_topic_corpus(db_path, papers_per_theme=4)

    summary = topics.compare_topic_models(
        db_path,
        output_dir,
        topic_counts=(2, 3),
        random_states=(0, 1),
        text_fields=('title', 'abstract'),
        min_df=1,
        max_df=1.0,
        max_features=500,
        learning_method='batch',
        max_iter=2,
        top_terms=5,
        representative_papers=1,
        stopwords_file=stopwords_path,
        ngram_max=2,
        streaming=False,
    )

    assert summary['models_trained'] == 4
    assert (output_dir / topics.COMPARISON_CSV_FILENAME).exists()
    assert (output_dir / topics.COMPARISON_JSON_FILENAME).exists()
    assert {(row['num_topics'], row['random_state']) for row in summary['models']} == {
        (2, 0), (2, 1), (3, 0), (3, 1),
    }
    assert all(0 <= row['mean_seed_stability'] <= 1 for row in summary['models'])
    assert all(0 < row['topic_diversity'] <= 1 for row in summary['models'])
    assert all((output_dir / f'topics-{row["num_topics"]}_seed-{row["random_state"]}').is_dir()
               for row in summary['models'])


def test_compare_topic_models_requires_multiple_configurations(tmp_path: Path) -> None:
    """Reject a comparison request that would train only one model."""
    with pytest.raises(ValueError, match='at least two'):
        topics.compare_topic_models(
            tmp_path / 'missing.db',
            tmp_path / 'comparison',
            topic_counts=(3,),
            random_states=(0,),
        )


def test_streaming_training_writes_version_two_artifacts_and_cleans_cache(tmp_path: Path) -> None:
    """Train online from bounded disk batches and remove the temporary cache."""
    db_path = tmp_path / 'papers.db'
    model_dir = tmp_path / 'streaming-model'
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    build_topic_corpus(db_path, papers_per_theme=4)

    with pytest.warns(UserWarning):
        summary = topics.train_topic_model(
            db_path,
            model_dir,
            num_topics=3,
            min_df=1,
            max_df=1.0,
            max_features=500,
            max_iter=2,
            random_state=4,
            top_terms=5,
            representative_papers=2,
            streaming=True,
            batch_size=3,
            cache_dir=cache_dir,
            evaluation_sample_size=5,
        )

    assert summary['config']['artifact_version'] == 2
    assert summary['config']['streaming'] is True
    assert summary['config']['model_id'].startswith('lda:')
    assert summary['report']['documents_used'] == 12
    assert summary['report']['evaluation_documents'] == 5
    assert summary['report']['metrics_scope'] == 'sample'
    assert not list(cache_dir.iterdir())
    with (model_dir / topics.PREDICTIONS_FILENAME).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    totals = defaultdict(float)
    for row in rows:
        totals[row['paper_id']] += float(row['probability'])
    assert len(totals) == 12
    assert all(value == pytest.approx(1.0) for value in totals.values())


def test_streaming_comparison_prepares_the_corpus_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse one disk-backed vocabulary and matrix cache across comparison models."""
    db_path = tmp_path / 'papers.db'
    output_dir = tmp_path / 'comparison'
    build_topic_corpus(db_path, papers_per_theme=3)
    original_prepare = topics._prepare_streaming_corpus
    calls = []

    def counted_prepare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Count and delegate streaming corpus preparation."""
        calls.append(1)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(topics, '_prepare_streaming_corpus', counted_prepare)
    summary = topics.compare_topic_models(
        db_path,
        output_dir,
        topic_counts=(2,),
        random_states=(0, 1),
        min_df=1,
        max_df=1.0,
        max_features=300,
        max_iter=1,
        top_terms=4,
        representative_papers=1,
        streaming=True,
        batch_size=3,
        evaluation_sample_size=4,
    )

    assert calls == [1]
    assert summary['models_trained'] == 2
    assert all(row['mean_seed_stability'] != '' for row in summary['models'])


def test_topic_trends_support_fixed_and_overlapping_windows(tmp_path: Path) -> None:
    """Aggregate probabilities into fixed-width and overlapping year windows."""
    db_path = tmp_path / 'papers.db'
    model_dir = tmp_path / 'model'
    build_topic_corpus(db_path, papers_per_theme=5)
    with pytest.warns(UserWarning):
        train_small_model(db_path, model_dir)

    fixed = topics.aggregate_topic_trends(
        model_dir,
        tmp_path / 'fixed',
        bin_size=5,
        step_size=5,
        start_year=2000,
        end_year=2024,
        plot=True,
    )
    rolling = topics.aggregate_topic_trends(
        model_dir,
        tmp_path / 'rolling',
        bin_size=5,
        step_size=1,
        start_year=2000,
        end_year=2006,
    )

    assert fixed['windows'] == 5
    with open(fixed['trends_csv'], encoding='utf-8', newline='') as handle:
        fixed_rows = list(csv.DictReader(handle))
    assert len(fixed_rows) == 5 * 3
    assert {row['window_start'] for row in fixed_rows} == {'2000', '2005', '2010', '2015', '2020'}
    assert Path(fixed['plot_path']).name == 'topic_trends_plot.png'
    assert Path(fixed['plot_path']).stat().st_size > 0
    assert rolling['windows'] == 7
    with open(rolling['trends_csv'], encoding='utf-8', newline='') as handle:
        rolling_rows = list(csv.DictReader(handle))
    assert any(row['is_partial'] == 'True' for row in rolling_rows)
    with pytest.raises(ValueError, match='must not exceed'):
        topics.aggregate_topic_trends(
            model_dir, tmp_path / 'invalid', bin_size=2, step_size=3
        )


def test_topic_store_predicts_fresh_scores_and_reports_staleness(tmp_path: Path) -> None:
    """Store normalized scores transactionally and detect subsequent corpus changes."""
    db_path = tmp_path / 'papers.db'
    model_dir = tmp_path / 'model'
    build_topic_corpus(db_path, papers_per_theme=3)
    with pytest.warns(UserWarning):
        train_small_model(db_path, model_dir)

    summary = topics.store_topic_model_scores(
        model_dir, db_path, name='demo-model', batch_size=2
    )

    assert summary['papers_predicted'] == 9
    with corpus.connect(db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM topic_models').fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM topic_definitions').fetchone()[0] == 3
        assert conn.execute('SELECT COUNT(*) FROM paper_topic_predictions').fetchone()[0] == 9
        assert conn.execute('SELECT COUNT(*) FROM paper_topic_scores').fetchone()[0] == 27
    assert topics.stored_topic_models(db_path)[0]['is_current'] is True
    rules_path = tmp_path / 'topic-filter.json'
    rules_path.write_text(json.dumps({
        'name': 'stored-topic-filter',
        'model': 'demo-model',
        'include': [{
            'name': 'topic-zero', 'topic_id': 0, 'min_probability': 0.0,
        }],
    }))
    filtering.apply_topic_filter(db_path, rules_path)

    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'new', 'title': 'unseen quasar terminology'})
    assert topics.stored_topic_models(db_path)[0]['is_current'] is False
    topics.store_topic_model_scores(model_dir, db_path, name='demo-model', batch_size=2)
    with corpus.connect(db_path) as conn:
        overview = filtering.filter_overview(conn)
    assert overview['stale_topic_filters'] == []
    assert sum(overview['counts'].values()) == 10
