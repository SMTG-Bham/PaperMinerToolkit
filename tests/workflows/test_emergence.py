"""Tests for embedding-cluster emergence analysis."""

from __future__ import annotations

import builtins
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from paperminertoolkit.corpus import database as corpus
from paperminertoolkit.workflows import emergence

# Each theme has its own words; a paper uses three of them, so no single word
# reaches the 40% document frequency above which labels drop a term.
THEMES = {
    'established': ['garnet', 'zirconate', 'sintering', 'densification', 'tantalum', 'cubic'],
    'declining': ['ceria', 'zirconia', 'yttria', 'cathodic', 'oxygen', 'fuelcell'],
    'emerging': ['oxyhalide', 'amorphous', 'chlorides', 'oxychloride', 'glassy', 'niobium'],
    'steady': ['polymer', 'polyether', 'plasticiser', 'oligomer', 'membrane', 'casting'],
}
FILLER = ('we measured the ionic transport in several prepared samples and '
          'compared them with earlier reports from other groups')


def _theme_shares(year: int) -> dict[str, float]:
    """Return each theme's share of the corpus in one year."""
    emerging = 0.0 if year < 2019 else min(0.3, 0.05 * (year - 2018))
    declining = max(0.02, 0.45 - 0.03 * (year - 2010))
    established = 0.3
    return {'established': established, 'declining': declining, 'emerging': emerging,
            'steady': max(0.0, 1.0 - established - declining - emerging)}


def build_emergence_corpus(db_path: Path, start: int = 2010, end: int = 2025) -> int:
    """Create a corpus whose themes grow, hold, and shrink in known ways.

    Parameters
    ----------
    db_path : pathlib.Path
        Corpus to create.
    start : int, default=2010
        First publication year.
    end : int, default=2025
        Last publication year.

    Returns
    -------
    int
        Number of papers written.
    """
    written = 0
    with corpus.connect(db_path) as conn:
        for year in range(start, end + 1):
            total = 30 + 4 * (year - start)
            for theme, share in _theme_shares(year).items():
                words = THEMES[theme]
                for index in range(round(total * share)):
                    chosen = [words[(index + offset) % len(words)] for offset in (0, 2, 3)]
                    paper = {
                        'paper_id': f'doi:10.1000/{theme}.{year}.{index}',
                        'doi': f'10.1000/{theme}.{year}.{index}',
                        'title': f'{chosen[0].title()} {chosen[1]} report {year}-{index}',
                        'publication_date': f'{year}-06-01',
                    }
                    corpus.add_asset(conn, paper, f'Abstract: {" ".join(chosen)} {FILLER}.',
                                     role='abstract', kind='text', mime_type='text/plain')
                    written += 1
    return written


def theme_embedder(texts: Sequence[str]) -> np.ndarray:
    """Embed a text as its theme direction plus deterministic per-text noise.

    Parameters
    ----------
    texts : Sequence[str]
        Texts naming one theme word.

    Returns
    -------
    numpy.ndarray
        One row per text.
    """
    rows = []
    for text in texts:
        lowered = text.lower()
        theme = next(index for index, words in enumerate(THEMES.values())
                     if any(word in lowered for word in words))
        seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
        noise = np.random.default_rng(seed).normal(scale=0.08, size=12)
        noise[theme] += 1.0
        rows.append(noise)
    return np.array(rows)


SMALL = emergence.EmergenceSettings(niche_size=20, topic_size=80, areas=3, backtest_cutoffs=2,
                                    novelty_min_prior=20)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact as dictionaries."""
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def test_growth_trend_recovers_known_share_growth() -> None:
    """Fit the yearly share multiplier and report flat or empty series as such."""
    years = np.arange(2018, 2026)
    totals = np.full(len(years), 1000)
    growing = np.round(10 * 1.5 ** (years - 2018))
    slope, se = emergence.growth_trend(growing, totals, years)
    assert math.exp(slope) == pytest.approx(1.5, abs=0.05)
    assert slope / se > 5
    slope, se = emergence.growth_trend(np.full(len(years), 50), totals, years)
    assert abs(slope / se) < 2
    assert all(math.isnan(value) for value in emergence.growth_trend(
        np.array([0, 1, 0]), np.array([10, 10, 10]), np.array([2020, 2021, 2022])))
    burst = np.array([0, 0, 0, 0, 0, 0, 0, 40])
    slope, se = emergence.growth_trend(burst, totals, years)
    assert math.isfinite(slope) and slope <= 3.0


@pytest.mark.parametrize(('slope', 'curvature', 'shape', 'turning'), [
    (0.5, 0.05, 'accelerating', ''),
    (0.5, -0.05, 'growth slowing', ''),
    (-0.3, -0.1, 'peaked', 2020),
    (0.3, 0.1, 'rebounding', 2020),
    (-0.5, 0.05, 'decline easing', ''),
    (-0.5, -0.05, 'decline steepening', ''),
    (0.1, 0.0, 'steady', ''),
])
def test_trend_shape_names_how_share_growth_changed(slope: float, curvature: float, shape: str,
                                                    turning: int | str) -> None:
    """Recover the shape, turning year, and end growth of known curved share trends."""
    years = np.arange(2018, 2026)
    t = years - years.mean()
    counts = np.round(400 * np.exp(slope * t + curvature * t ** 2))
    result = emergence.trend_shape(counts, np.full(len(years), 20000), years)
    assert result['shape'] == shape
    assert result['shape'] in emergence.SHAPES
    assert result['turning_year'] == turning
    assert result['end_growth'] == pytest.approx(math.exp(slope + 2 * curvature * t[-1]), rel=0.02)


def test_trend_shape_leaves_sparse_series_unlabelled() -> None:
    """Give no shape when too few papers leave nothing to fit."""
    years = np.arange(2018, 2026)
    result = emergence.trend_shape(np.array([0, 1, 0, 1, 0, 0, 1, 0]),
                                   np.full(len(years), 100), years)
    assert result['shape'] == '' and math.isnan(result['curvature_z'])


def test_niche_spread_tells_gathered_terms_from_scattered_ones() -> None:
    """Score a term held by one niche high on both measures, and an even one at zero."""
    niches = np.repeat(np.arange(4), 25)
    gathered = (niches == 2).astype(float)
    concentration, dispersion = emergence._niche_spread(
        sparse.csr_matrix(np.column_stack([gathered, np.ones(100)])), niches)
    assert concentration == pytest.approx([1.0, 0.0])
    assert dispersion == pytest.approx([25.0, 0.0])


def test_topic_term_rows_find_what_rises_and_fades_inside_a_topic() -> None:
    """Name a topic's rising and fading field terms, but not wording that is spread out."""
    terms = np.array(['newphase', 'oldphase', 'common', 'scattered'])
    # "scattered" appears in six recent topic-0 papers, each in a different niche: rare enough
    # to look concentrated, but spread as chance would spread it.
    scattered = {(2023, 10), (2023, 12), (2024, 14), (2024, 16), (2025, 18), (2025, 1)}
    rows, years, topics, niches = [], [], [], []
    for year in range(2018, 2026):
        step = year - 2018
        for topic in (0, 1):
            for index in range(20):
                # In topic 0, newphase grows from 0 to 14 of 20 papers a year and oldphase
                # shrinks from 16 to 2; half of every topic's papers say "common".
                used = [topic == 0 and index < 2 * step, topic == 0 and index >= 2 * step + 4,
                        index % 2 == 0, topic == 0 and (year, index) in scattered]
                rows.append([int(flag) for flag in used])
                years.append(year)
                topics.append(topic)
                niches.append(10 * topic + index // 2)
    D = sparse.csr_matrix(np.array(rows))

    found = emergence._topic_term_rows(D, terms, np.array(years), np.array(topics),
                                       np.array(niches), 2025, emergence.EmergenceSettings())

    assert {(row['topic'], row['direction'], row['term']) for row in found} == {
        (0, 'rising', 'newphase'), (0, 'fading', 'oldphase')}
    rising = next(row for row in found if row['direction'] == 'rising')
    assert rising['growth_z'] >= 2 and rising['lift'] == pytest.approx(2.0)
    assert rising['counts_by_year'] == '0 2 4 6 8 10 12 14'
    concentration, dispersion = emergence._niche_spread(D[:, [3]], np.array(niches))
    assert concentration[0] >= emergence.TOPIC_TERM_CONCENTRATION
    assert dispersion[0] < emergence.TOPIC_TERM_DISPERSION


def test_classify_trend_labels_each_status() -> None:
    """Label emerging, growing, stable, declining, and too-small clusters."""
    assert emergence.classify_trend(3.0, 2.0, 10) == 'emerging'
    assert emergence.classify_trend(3.0, 1.2, 10) == 'growing'
    assert emergence.classify_trend(0.5, 1.0, 10) == 'stable'
    assert emergence.classify_trend(-2.5, 0.5, 10) == 'declining'
    assert emergence.classify_trend(math.nan, 1.0, 10) == 'too small'
    assert emergence.classify_trend(3.0, 2.0, 2) == 'too small'


def test_settings_reject_impossible_windows_and_sizes() -> None:
    """Refuse windows and granularities that cannot define an analysis."""
    with pytest.raises(ValueError, match='growth_window'):
        emergence.EmergenceSettings(growth_window=2).validate()
    with pytest.raises(ValueError, match='recent_window'):
        emergence.EmergenceSettings(growth_window=4, recent_window=4).validate()
    with pytest.raises(ValueError, match='niche_size'):
        emergence.EmergenceSettings(niche_size=300, topic_size=250).validate()
    with pytest.raises(ValueError, match='areas'):
        emergence.EmergenceSettings(areas=1).validate()
    with pytest.raises(ValueError, match='backtest_cutoffs'):
        emergence.EmergenceSettings(backtest_cutoffs=-1).validate()
    with pytest.raises(ValueError, match='timeline_years'):
        emergence.EmergenceSettings(timeline_years=-1).validate()
    with pytest.raises(ValueError, match='horizon'):
        emergence.EmergenceSettings(horizon=0).validate()


def test_clean_abstract_removes_heading_markup_and_publisher_furniture() -> None:
    """Strip text that would cluster papers by publisher instead of subject."""
    raw = ('Abstract: <p>Halide &amp; oxide electrolytes.</p> Download: Download high-res image '
           '(213KB) Download: Download full-size image Display Omitted')
    assert emergence.clean_abstract(raw) == 'Halide & oxide electrolytes.'
    assert emergence.clean_abstract(None) == ''


def test_screen_documents_drops_patents_foreign_text_and_duplicates() -> None:
    """Keep one record per title and year and drop patent or non-English text."""
    def document(paper_id: str, title: str, abstract: str, year: int = 2020) -> emergence.EmergenceDocument:
        """Build one candidate document."""
        return {'paper_id': paper_id, 'doi': '', 'title': title, 'abstract': abstract,
                'year': year, 'cited_by_count': math.nan}

    english = 'the conductivity of the electrolyte was measured at room temperature and it was high'
    german = ' '.join(['Leitfähigkeit Elektrolyt Messung Temperatur Probe'] * 5)
    kept, dropped = emergence.screen_documents([
        document('a', 'Same title', ''),
        document('b', 'Same  Title!', english),
        document('c', 'Same title', english, year=2021),
        document('d', 'A cell', 'In one embodiment the present invention provides a cell.'),
        document('e', 'Ein Elektrolyt', german),
    ])
    assert [item['paper_id'] for item in kept] == ['b', 'c']
    assert dropped == {'patent_like': 1, 'non_english': 1, 'duplicate_title': 1}


def test_embed_documents_caches_vectors_and_embeds_repeats_once(tmp_path: Path) -> None:
    """Embed each distinct uncached text once and reuse the cache afterwards."""
    calls: list[list[str]] = []

    def recording_embedder(texts: Sequence[str]) -> np.ndarray:
        """Record the texts asked for and return fixed vectors."""
        calls.append(list(texts))
        return np.array([[3.0, 4.0] for _ in texts])

    cache = tmp_path / 'cache.npz'
    vectors, computed = emergence.embed_documents(['a', 'b', 'a'], cache, model_name='m',
                                                  embedder=recording_embedder)
    assert computed == 2 and calls == [['a', 'b']]
    assert np.allclose(vectors, [[0.6, 0.8]] * 3)
    vectors, computed = emergence.embed_documents(['b', 'a'], cache, model_name='m',
                                                  embedder=recording_embedder)
    assert computed == 0 and len(calls) == 1
    _, computed = emergence.embed_documents(['a'], cache, model_name='other',
                                            embedder=recording_embedder)
    assert computed == 1


def test_transformer_embedder_explains_how_to_install_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Name the optional dependency when PyTorch is missing."""
    original_import = builtins.__import__

    def import_without_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        """Fail to import torch and import everything else normally."""
        if name == 'torch':
            raise ImportError('No module named torch')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_without_torch)
    with pytest.raises(RuntimeError, match=r'paperminertoolkit\[bert\]'):
        emergence.transformer_embedder('any/model')


def test_analyze_emergence_maps_known_growth_and_decline(tmp_path: Path) -> None:
    """Find the emerging theme, label the declining one, and write every artifact."""
    db_path = tmp_path / 'papers.db'
    written = build_emergence_corpus(db_path)
    output = tmp_path / 'emergence'

    summary = emergence.analyze_emergence(db_path, output, settings=SMALL, end_year=2025,
                                          embedder=theme_embedder,
                                          watch_terms=('oxyhalide', 'absent'))

    assert summary['papers_read'] == written
    assert summary['papers_used'] == written
    assert summary['areas'] == 3
    for name in (emergence.REPORT_FILENAME, emergence.SUMMARY_FILENAME, emergence.AREAS_FILENAME,
                 emergence.TOPICS_FILENAME, emergence.NICHES_FILENAME, emergence.PAPERS_FILENAME,
                 emergence.TERMS_FILENAME, emergence.WATCH_FILENAME, emergence.BACKTEST_FILENAME,
                 emergence.COVERAGE_FILENAME, emergence.SERIES_FILENAME,
                 emergence.TIMELINE_FILENAME, emergence.TOPIC_TERMS_FILENAME,
                 emergence.EMBEDDINGS_FILENAME):
        assert (output / name).exists(), name
    plots = sorted((output / emergence.PLOTS_DIRNAME).iterdir())
    assert [path.name for path in plots] == ['areas.png', 'niches.png', 'timeline.png',
                                             'topics.png', 'watch_terms.png']
    assert all(path.read_bytes()[:4] == b'\x89PNG' for path in plots)
    assert sorted(summary['plots']) == [str(path) for path in plots]
    assert 'NaN' not in (output / emergence.SUMMARY_FILENAME).read_text()
    assert json.loads((output / emergence.SUMMARY_FILENAME).read_text())['end_year'] == 2025

    def theme_of(label_terms: str) -> str:
        """Name the theme whose words label a cluster."""
        return next(theme for theme, words in THEMES.items()
                    if any(word in label_terms for word in words))

    # Niches split a theme at random with respect to year, so a niche of a
    # stable theme can drift; the top of the ranking and the topics cannot.
    niches = _read_csv(output / emergence.NICHES_FILENAME)
    assert theme_of(niches[0]['label_terms']) == 'emerging'
    assert any(theme_of(niche['label_terms']) == 'emerging' for niche in summary['flagged_niches'])
    status: dict[str, set[str]] = {}
    for topic in _read_csv(output / emergence.TOPICS_FILENAME):
        status.setdefault(theme_of(topic['label_terms']), set()).add(topic['status'])
    assert 'emerging' in status['emerging']
    assert 'declining' in status['declining']
    assert 'emerging' not in status['established']

    watch = {row['term']: row for row in _read_csv(output / emergence.WATCH_FILENAME)}
    assert watch['oxyhalide']['2015'] == '0' and int(watch['oxyhalide']['2025']) > 0
    assert watch['absent']['docs'] == '0'
    backtest = _read_csv(output / emergence.BACKTEST_FILENAME)
    assert backtest[-1]['cutoff'] == 'pooled'
    report = (output / emergence.REPORT_FILENAME).read_text(encoding='utf-8')
    for heading in ('## Field map', '## Emerging niches', '## Emergence timeline',
                    '## Emerging terms', '## Backtest', '## Watched terms', '## Coverage by year'):
        assert heading in report
    for path in plots:
        assert f']({emergence.PLOTS_DIRNAME}/{path.name})' in report
    # The synthetic corpus has no citation counts, so the impact column is left out.
    assert '| impact |' not in report and 'pmt enrich' in report
    field_map = report.split('## Emerging niches')[0]
    for topic in _read_csv(output / emergence.TOPICS_FILENAME):
        nearest = topic['central_titles'].split(' || ')[0]
        assert f'- Topic {topic["cluster"]}: {nearest}' in field_map
        assert topic['shape'] in (*emergence.SHAPES, '')
        assert len(topic['share_sparkline']) == 16

    # Every paper sits in one area, so the areas' yearly papers add up to the corpus.
    series = _read_csv(output / emergence.SERIES_FILENAME)
    for year in range(2010, 2026):
        rows = [row for row in series if row['level'] == 'area' and row['year'] == str(year)]
        assert sum(int(row['papers']) for row in rows) == int(rows[0]['corpus_papers'])

    # The emerging theme was flagged before the analysis year; the declining one never was.
    first = {theme_of(niche['label_terms']): set() for niche in niches}
    for niche in niches:
        if niche['first_flagged']:
            first[theme_of(niche['label_terms'])].add(int(niche['first_flagged']))
    assert first['emerging'] and min(first['emerging']) < 2025
    assert not first['declining']
    # A niche flagged now was seen at the latest by now.
    assert all(niche['first_seen'] != '' and niche['first_seen'] <= 2025
               for niche in summary['flagged_niches'])
    timeline = _read_csv(output / emergence.TIMELINE_FILENAME)
    assert {row['year'] for row in timeline} <= {str(year) for year in range(2019, 2026)}
    latest = [row for row in timeline if row['year'] == '2025']
    assert all(row['earlier_niche'] == row['niche'] and row['overlap'] == '1.000' for row in latest)


def test_analyze_emergence_reuses_the_cache_and_protects_earlier_output(tmp_path: Path) -> None:
    """Refuse to overwrite silently, then rerun from cached embeddings with citations."""
    db_path = tmp_path / 'papers.db'
    build_emergence_corpus(db_path)
    output = tmp_path / 'emergence'
    first = emergence.analyze_emergence(db_path, output, settings=SMALL, end_year=2025,
                                        embedder=theme_embedder)
    assert first['papers_embedded_now'] == first['papers_used']

    def refuse(texts: Sequence[str]) -> np.ndarray:
        """Fail if any text needs embedding again."""
        raise AssertionError(f'{len(texts)} texts were embedded again')

    with pytest.raises(ValueError, match='already holds an analysis'):
        emergence.analyze_emergence(db_path, output, settings=SMALL, embedder=refuse)
    conn = corpus.connect(db_path)
    conn.execute('UPDATE papers SET cited_by_count = 3')
    conn.commit()
    conn.close()
    second = emergence.analyze_emergence(db_path, output, settings=SMALL, end_year=2025,
                                         embedder=refuse, overwrite=True, plots=False)
    assert second['papers_embedded_now'] == 0
    assert second['niches'] == first['niches']
    report = (output / emergence.REPORT_FILENAME).read_text(encoding='utf-8')
    assert '| impact |' in report
    # Without plots the report embeds no figures, even with an earlier run's plots beside it.
    assert second['plots'] == [] and '](plots/' not in report


def test_analyze_emergence_rejects_corpora_that_cannot_show_change(tmp_path: Path) -> None:
    """Explain that a one- or two-year corpus has no history to compare against."""
    short = tmp_path / 'short.db'
    build_emergence_corpus(short, start=2024, end=2025)
    with pytest.raises(ValueError, match='cannot show change over time'):
        emergence.analyze_emergence(short, tmp_path / 'out', settings=SMALL, end_year=2025,
                                    embedder=theme_embedder)
    empty = tmp_path / 'empty.db'
    corpus.connect(empty).close()
    with pytest.raises(ValueError, match='at least 50'):
        emergence.analyze_emergence(empty, tmp_path / 'out2', settings=SMALL,
                                    embedder=theme_embedder)


def test_load_emergence_documents_can_keep_title_only_papers(tmp_path: Path) -> None:
    """Drop undated papers and, unless asked, papers without an abstract."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'doi:10.1/a', 'title': 'Title only',
                                   'publication_date': '2020'})
        corpus.upsert_paper(conn, {'paper_id': 'doi:10.1/b', 'title': 'Undated'})
    documents, counts = emergence.load_emergence_documents(db_path)
    assert documents == [] and counts['no_abstract'] == 1 and counts['no_year'] == 1
    documents, _ = emergence.load_emergence_documents(db_path, require_abstract=False)
    assert [document['title'] for document in documents] == ['Title only']
    assert emergence._embedding_text(documents[0]) == 'Title only'
