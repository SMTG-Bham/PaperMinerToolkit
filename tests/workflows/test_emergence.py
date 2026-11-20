"""Tests for embedding-cluster emergence analysis."""

from __future__ import annotations

import builtins
import contextlib
import csv
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    assert [path.name for path in plots] == ['area_growth.png', 'areas.png', 'niche_growth.png',
                                             'niches.png', 'timeline.png', 'topic_growth.png',
                                             'watch_terms.png']
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


def test_load_emergence_documents_drops_dated_papers_without_any_text(tmp_path: Path) -> None:
    """Count a dated paper whose title is only markup and which has no abstract as textless."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'doi:10.1/c', 'title': '<sup></sup>',
                                   'publication_date': '2021'})
    documents, counts = emergence.load_emergence_documents(db_path, require_abstract=False)
    assert documents == [] and counts['no_text'] == 1


class _FakeTensor:
    """Wrap a NumPy array in the few tensor methods the transformer embedder calls."""

    def __init__(self, array: np.ndarray) -> None:
        """Hold the wrapped array.

        Parameters
        ----------
        array : numpy.ndarray
            Values of the tensor.
        """
        self.array = array

    def __getitem__(self, index: Any) -> _FakeTensor:
        """Index the wrapped array."""
        return _FakeTensor(self.array[index])

    def float(self) -> _FakeTensor:
        """Return the tensor unchanged, as it already holds floats."""
        return self

    def cpu(self) -> _FakeTensor:
        """Return the tensor unchanged, as it already lives in host memory."""
        return self

    def numpy(self) -> np.ndarray:
        """Return the wrapped array."""
        return self.array


class _FakeBatch(dict):
    """Tokenizer output that can be moved to a device."""

    def to(self, device: str) -> _FakeBatch:
        """Return the batch unchanged; devices are not modelled."""
        return self


class _FakeEncoder:
    """Stand in for a Hugging Face encoder whose CLS vector is ``[length, 1, 0]``."""

    config = SimpleNamespace(hidden_size=3)

    def __init__(self) -> None:
        """Start on no device."""
        self.device = ''

    def to(self, device: str) -> _FakeEncoder:
        """Record the device the model is moved to."""
        self.device = device
        return self

    def eval(self) -> _FakeEncoder:
        """Return the model; there is no training mode to leave."""
        return self

    def __call__(self, lengths: list[int]) -> SimpleNamespace:
        """Return hidden states for two tokens whose first is ``[length, 1, 0]``.

        Parameters
        ----------
        lengths : list[int]
            Length of each text in the batch, as the fake tokenizer encodes it.

        Returns
        -------
        types.SimpleNamespace
            Output with a ``last_hidden_state`` of shape (batch, 2, 3).
        """
        hidden = np.zeros((len(lengths), 2, 3))
        hidden[:, 0, 0] = lengths
        hidden[:, 0, 1] = 1.0
        return SimpleNamespace(last_hidden_state=_FakeTensor(hidden))


def test_transformer_embedder_encodes_length_sorted_batches_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch texts by length on the CPU when CUDA is absent, and return vectors in input order."""
    batches: list[list[str]] = []
    loaded: list[str] = []
    model = _FakeEncoder()

    def tokenize(texts: list[str], **options: Any) -> _FakeBatch:
        """Record a batch and encode each text as its length."""
        batches.append(list(texts))
        assert options['padding'] and options['truncation'] and options['return_tensors'] == 'pt'
        return _FakeBatch(lengths=[len(text) for text in texts])

    def load_tokenizer(name: str) -> Callable[..., _FakeBatch]:
        """Record the tokenizer asked for and return the fake one."""
        loaded.append(name)
        return tokenize

    def load_model(name: str) -> _FakeEncoder:
        """Record the model asked for and return the fake one."""
        loaded.append(name)
        return model

    # Small stand-ins, so the embedding loop runs without PyTorch or a model download.
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.inference_mode = contextlib.nullcontext
    transformers = ModuleType('transformers')
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=load_tokenizer)
    transformers.AutoModel = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)

    embed = emergence.transformer_embedder('fake/model', batch_size=2)
    vectors = embed(['ccc', 'a', 'bb'])

    assert loaded == ['fake/model', 'fake/model'] and model.device == 'cpu'
    assert batches == [['a', 'bb'], ['ccc']]
    assert vectors.tolist() == [[3.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]


def test_fits_give_up_when_every_paper_falls_in_one_year() -> None:
    """Return no trend, shape, or curve when a single year leaves the fit singular."""
    years = np.full(5, 2020)
    counts, totals = np.full(5, 4), np.full(5, 100)
    assert emergence._fit_counts(counts.astype(float), totals.astype(float), np.zeros(5),
                                 degree=1) is None
    assert all(math.isnan(value) for value in emergence.growth_trend(counts, totals, years))
    assert emergence.trend_shape(counts, totals, years)['shape'] == ''
    assert emergence._trend_curve(counts, totals, years, 'peaked') is None


def test_trend_shape_needs_a_curvature_error_to_judge_the_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give no shape when the fitted curvature has a zero standard error."""
    monkeypatch.setattr(emergence, '_fit_counts',
                        lambda *_, **__: (np.array([0.0, 0.1, 0.05]), np.zeros((3, 3))))
    years = np.arange(2018, 2026)
    result = emergence.trend_shape(np.full(len(years), 10), np.full(len(years), 100), years)
    assert result['shape'] == '' and math.isnan(result['curvature_z'])


def test_trend_curve_draws_the_fitted_share_or_nothing_for_a_sparse_series() -> None:
    """Return the fitted yearly share, and no curve when too few papers leave nothing to fit."""
    years = np.array([2020, 2021, 2022])
    curve = emergence._trend_curve(np.array([2, 4, 8]), np.full(3, 100), years, '')
    assert curve == pytest.approx([0.02, 0.04, 0.08], rel=1e-4)
    assert emergence._trend_curve(np.array([0, 1, 0]), np.full(3, 100), years, '') is None


def test_topic_term_rows_skip_topics_without_papers_in_the_window() -> None:
    """Pass over a topic whose papers all predate the growth window."""
    years = np.array([2005, 2006, 2007, *range(2018, 2026)])
    topics = np.array([1, 1, 1, *[0] * 8])
    # One term used evenly by every paper: neither topic has anything rising or fading.
    D = sparse.csr_matrix(np.ones((len(years), 1)))
    rows = emergence._topic_term_rows(D, np.array(['garnet']), years, topics, topics, 2025,
                                      emergence.EmergenceSettings())
    assert rows == []


def test_cutoff_snapshot_needs_enough_papers_and_years_before_the_cutoff() -> None:
    """Skip a cutoff with too few papers to cluster or too few years to fit growth."""
    years = np.repeat(np.arange(2016, 2026), 3)
    X = np.random.default_rng(0).normal(size=(len(years), 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    novelty = np.full(len(years), np.nan)
    # 4 * niche_size = 80 papers are needed; there are 30.
    assert emergence._cutoff_snapshot(X, years, novelty, 2025,
                                      emergence.EmergenceSettings(niche_size=20)) is None
    small = emergence.EmergenceSettings(niche_size=5)
    # 21 papers precede 2022, enough for 20, but they span 7 years, fewer than growth_window.
    assert emergence._cutoff_snapshot(X, years, novelty, 2022, small) is None
    snapshot = emergence._cutoff_snapshot(X, years, novelty, 2025, small)
    assert snapshot is not None and len(snapshot['labels']) == len(years)


def test_backtest_skips_cutoffs_without_later_papers_or_a_snapshot() -> None:
    """Pass over a cutoff with too few later papers and one that could not be re-clustered."""
    settings = emergence.EmergenceSettings(niche_size=5, backtest_cutoffs=2)
    years = np.array([2015] * 10 + [2025] * 5)
    asked: list[int] = []

    def no_snapshot(cutoff: int) -> dict[str, Any] | None:
        """Record the cutoff and report that the corpus could not be re-clustered then."""
        asked.append(cutoff)
        return None

    # Cutoff 2021 has no papers in 2022-2024; cutoff 2022 has the five from 2025.
    summaries, pooled = emergence._backtest(np.zeros((len(years), 2)), years, 2025, settings,
                                            no_snapshot)
    assert (summaries, pooled) == ([], [])
    assert asked == [2022]


def test_sparkline_draws_a_series_without_papers_flat() -> None:
    """Draw an all-zero series at the lowest block instead of dividing by zero."""
    assert emergence._sparkline([1.0, 2.0]) == '▅█'
    assert emergence._sparkline([0.0, 0.0, 0.0]) == '▁▁▁'
    assert emergence._sparkline([]) == ''


def _cluster_row(cluster: int, **fields: Any) -> dict[str, Any]:
    """Build a cluster row with the fields the report and plots read.

    Parameters
    ----------
    cluster : int
        Cluster identifier.
    **fields : Any
        Values replacing the defaults.

    Returns
    -------
    dict[str, Any]
        One area, topic, or niche row.
    """
    row: dict[str, Any] = {
        'cluster': cluster, 'area': 0, 'topic': 0, 'label_terms': 'oxyhalide; glassy; niobium',
        'dated_papers': 40, 'recent_papers': 12, 'recent_share': 0.2,
        'share_growth_per_year': 1.3, 'growth_z': 3.0, 'recency_lift': 1.8, 'status': 'emerging',
        'shape': 'steady', 'turning_year': '', 'share_sparkline': '▁▄█', 'flag': False,
        'central_titles': 'Oxyhalide glass (2025) || Glassy niobium (2024)',
        'rising_terms': '', 'fading_terms': '',
    }
    row.update(fields)
    return row


def test_report_lists_terms_rising_and_fading_inside_a_topic() -> None:
    """Write a topic's changing terms under it, and no timeline section when none was run."""
    summary = {'end_year': 2025, 'db_path': 'papers.db', 'papers_used': 40, 'papers_read': 41,
               'dropped': {'no_year': 1}, 'embedding_model': 'stub'}
    topic = _cluster_row(0, rising_terms='oxyhalide; glassy', fading_terms='ceria')
    report = emergence._report(summary, [_cluster_row(0)], [topic], [], [], [], [], [], [], 2018,
                               emergence.EmergenceSettings(), 20, {})
    assert '  - rising: oxyhalide, glassy; fading: ceria' in report
    assert '## Emergence timeline' not in report


def test_plots_are_skipped_when_there_is_nothing_to_draw() -> None:
    """Draw no figure for no niches, topics without a fitted trend, or a skipped timeline."""
    span = np.arange(2016, 2026)
    window = span[-8:]
    table = np.zeros((1, len(span)), dtype=np.int64)
    assert emergence._plot_niches([], table, span, np.full(len(span), 10), window) is None
    assert emergence._plot_growth([_cluster_row(0, share_growth_per_year=math.nan)], window,
                                  'topic') is None
    assert emergence._plot_timeline([_cluster_row(0)], [], 20) is None


def test_growth_plots_name_every_area_the_top_niches_and_the_fastest_topics() -> None:
    """Name all areas, the top-ranked eligible niches, and the three fastest topics each way."""
    rows = [_cluster_row(index, growth_z=z, eligible=index != 1, flag=index == 0)
            for index, z in enumerate([5.0, 4.0, 3.0, 2.5, -1.0, -2.5, -3.0, -4.0, -5.0])]
    assert emergence._growth_naming(rows, 'area', 10)[0] == set(range(9))
    assert emergence._growth_naming(rows, 'niche', 2)[0] == {0, 2}
    assert emergence._growth_naming(rows, 'topic', 10)[0] == {0, 1, 2, 6, 7, 8}
    window = np.arange(2018, 2026)
    niches = emergence._plot_growth(rows, window, 'niche', 2)
    # Niches name only the top-ranked, without the numbers of the rest or interval lines.
    assert [text.get_text() for text in niches.axes[0].texts] == ['0 oxyhalide', '2 oxyhalide']
    assert len(niches.axes[0].lines) == 1  # the zero line
    legend = [text.get_text() for text in niches.axes[0].get_legend().get_texts()]
    assert legend[-1] == 'flagged as emerging'
    areas = emergence._plot_growth(rows, window, 'area')
    assert any('Each dot is an area' in text.get_text() for text in areas.texts)
    assert len(areas.axes[0].texts) == 9
    # Every row sits on one point: the four rising areas, labelled on the right, step from just
    # above it to below, further above, and further below, rather than printing over each other.
    assert [text.xyann[1] for text in areas.axes[0].texts][:5] == [3.0, -12.375, 12.375, -21.75,
                                                                  3.0]


def test_share_panels_hide_spare_axes_and_watched_terms_share_a_legend() -> None:
    """Hide the unused panels of a part-filled grid, and name several watched terms in a legend."""
    span = np.arange(2016, 2026)
    table = np.array([np.arange(len(span)) + 2 * cluster for cluster in range(5)])
    table[4] = 0  # An area without papers in the span has no trend to fit.
    figure = emergence._plot_areas([_cluster_row(cluster) for cluster in range(5)], table, span,
                                   np.full(len(span), 100), span[-8:])
    assert [axis.get_visible() for axis in figure.axes] == [True] * 5 + [False] * 3

    watch = [{'term': 'oxyhalide', 'docs': 5, '2024': 2, '2025': 3},
             {'term': 'glassy', 'docs': 3, '2024': 1, '2025': 2},
             {'term': 'absent', 'docs': 0, '2024': 0, '2025': 0}]
    coverage = [{'year': 2024, 'papers': 10}, {'year': 2025, 'papers': 12}]
    figure = emergence._plot_watch(watch, coverage, np.arange(2024, 2026))
    legend = figure.axes[0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ['oxyhalide', 'glassy']


def test_draw_plots_explains_how_to_get_matplotlib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Name the missing dependency and the option that skips the plots."""
    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    span = np.arange(2018, 2026)
    with pytest.raises(RuntimeError, match='requires matplotlib.*--no-plots'):
        emergence._draw_plots(tmp_path, span, np.ones(len(span), dtype=np.int64), {}, [], [], [],
                              [], [], [], 2025, emergence.EmergenceSettings(), 20)


def test_analyze_emergence_caps_the_corpus_and_leaves_out_the_current_year(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a corpus too large for one Ward tree, and by default stop before this year."""
    db_path = tmp_path / 'papers.db'
    this_year = time.gmtime().tm_year
    build_emergence_corpus(db_path, start=2020, end=2021)
    build_emergence_corpus(db_path, start=this_year, end=this_year)
    with monkeypatch.context() as patch:
        patch.setattr(emergence, 'MAX_DOCUMENTS', emergence.MIN_DOCUMENTS)
        with pytest.raises(ValueError, match=f'exceed the {emergence.MIN_DOCUMENTS} that one Ward'):
            emergence.analyze_emergence(db_path, tmp_path / 'capped', settings=SMALL,
                                        embedder=theme_embedder)
    # This year is incomplete, so the default analysis year is last year and only 2020 and
    # 2021 count, too few to fit growth.
    with pytest.raises(ValueError, match=f'spans 2 publication years up to {this_year - 1};'):
        emergence.analyze_emergence(db_path, tmp_path / 'default', settings=SMALL,
                                    embedder=theme_embedder)
