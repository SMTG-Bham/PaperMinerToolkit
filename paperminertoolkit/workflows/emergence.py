"""Find emerging research areas in a corpus from embedded abstracts.

LDA trends average one fixed set of broad topics over time, so a small new
field is absorbed into an older topic and never appears. This workflow instead
embeds each title and abstract, builds one Ward clustering tree, and cuts it at
three depths: a dozen *areas*, a few dozen *topics*, and a few hundred small
*niches*. Every cluster is then scored on how its share of the corpus changes:

``growth``
    Slope of a quasi-Poisson fit to the cluster's yearly paper counts with the
    corpus size as an offset, so it measures a growing *share*, not a growing
    field. Reported as the yearly share multiplier and a z statistic.
``recency lift``
    The cluster's fraction of papers from the last few years, divided by the
    corpus fraction from those years.
``novelty``
    How unlike every earlier paper a niche's first papers were. It is reported
    but not scored: on a solid-electrolyte corpus it correlated negatively with
    later growth.

Areas and topics are labelled emerging, growing, stable, or declining. Niches
are ranked by the mean percentile of growth and recency lift, and the ranking
is backtested: the analysis is repeated on papers up to an earlier year and
compared with what grew afterwards. The scores measure momentum, so a field is
found once it is measurably growing, not before.

The corpus must sample every publication year the same way. A relevance-ranked
search over-represents some years, which then look like growth.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import textwrap
import time
import unicodedata
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from tqdm import tqdm

from paperminertoolkit.corpus.database import connect, get_asset, utc_now
from paperminertoolkit.workflows.topics import load_domain_stopwords

DEFAULT_EMBEDDING_MODEL = 'allenai/specter2_base'
REPORT_FILENAME = 'report.md'
SUMMARY_FILENAME = 'summary.json'
AREAS_FILENAME = 'areas.csv'
TOPICS_FILENAME = 'topics.csv'
NICHES_FILENAME = 'niches.csv'
PAPERS_FILENAME = 'paper_clusters.csv'
TERMS_FILENAME = 'emerging_terms.csv'
WATCH_FILENAME = 'watch_terms.csv'
BACKTEST_FILENAME = 'backtest.csv'
BACKTEST_NICHES_FILENAME = 'backtest_niches.csv'
COVERAGE_FILENAME = 'coverage_by_year.csv'
SERIES_FILENAME = 'cluster_years.csv'
TIMELINE_FILENAME = 'timeline.csv'
TOPIC_TERMS_FILENAME = 'topic_terms.csv'
PLOTS_DIRNAME = 'plots'
EMBEDDINGS_FILENAME = 'embeddings.npz'
STATUSES = ('emerging', 'growing', 'stable', 'declining', 'too small')
SHAPES = ('accelerating', 'growth slowing', 'peaked', 'rebounding', 'decline easing',
          'decline steepening', 'steady')
# Yearly share series and sparklines cover at most this many years up to the analysis year.
SERIES_YEARS = 25
# A niche is traced back to an earlier year only once this many of its papers existed then.
LINEAGE_MIN_PAPERS = 3
# A term rising (fading) inside a topic must be this many times as common in the topic's recent
# (earlier) papers as in the corpus's.
TOPIC_TERM_LIFT = 1.5
# ...and be field vocabulary: gathered in a few niches by both measures of _niche_spread.
TOPIC_TERM_CONCENTRATION = 0.25
TOPIC_TERM_DISPERSION = 2.0
_SPARK = '▁▂▃▄▅▆▇█'
# Plot colours: chart surface (white, so figures drop into papers and slides as they are) and
# ink, a blue/red pair for rising and falling share, a sequential blue ramp for percentiles,
# and categorical slots taken in this order.
_SURFACE, _INK, _INK2, _MUTED, _AXIS, _BAND, _GRID = (
    '#ffffff', '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#f0efec', '#e1e0d9')
_RISE, _FALL = '#2a78d6', '#e34948'
_RAMP = ('#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b')
_SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
_PLOT_STYLE = {
    'figure.facecolor': _SURFACE, 'axes.facecolor': _SURFACE, 'savefig.facecolor': _SURFACE,
    'axes.edgecolor': _AXIS, 'axes.labelcolor': _INK2, 'axes.titlecolor': _INK,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True, 'grid.color': _GRID,
    'grid.linewidth': 0.6, 'xtick.color': _MUTED, 'ytick.color': _MUTED, 'xtick.labelcolor': _INK2,
    'ytick.labelcolor': _INK2, 'text.color': _INK, 'font.size': 9, 'axes.titlesize': 9,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.frameon': False, 'legend.fontsize': 8,
}
# Ward linkage holds the condensed distance matrix in memory: n(n-1)/2 doubles,
# about 10 GB at this size.
MAX_DOCUMENTS = 50000
MIN_DOCUMENTS = 50
TOKEN_PATTERN = r'(?u)\b[a-z][a-z0-9]{1,}\b'
_YEAR_RE = re.compile(r'(?:19|20)\d{2}')
_TAG_RE = re.compile(r'<[^>]+>')
_WORD_RE = re.compile(r'[a-z]+')
_ABSTRACT_PREFIX_RE = re.compile(r'^\s*(?:abstract|summary)\s*[:.\-]?\s+', re.IGNORECASE)
# Publisher furniture that OpenAlex and others leave inside abstracts. Left in,
# it pulls papers from one publisher into a cluster of their own.
_BOILERPLATE_RE = re.compile(
    r'download:?\s+download\s+(?:high[- ]res(?:olution)?|full[- ]size)\s+image(?:\s*\(\d+\s*[km]b\))?'
    r'|download\s+(?:high[- ]res(?:olution)?\s+image|powerpoint\s+slide)'
    r'|high[- ]resolution\s+image|display\s+omitted|graphical\s+abstract|abstract\s+image',
    re.IGNORECASE,
)
_PATENT_RE = re.compile(
    r'\b(?:the\s+(?:present\s+)?invention|embodiments?|wherein|disclosed\s+herein)\b',
    re.IGNORECASE,
)
_GENERIC_STOPWORDS = frozenset({
    'study', 'studies', 'studied', 'result', 'results', 'using', 'used', 'use', 'based',
    'high', 'higher', 'performance', 'paper', 'work', 'report', 'reported', 'reports',
    'show', 'shows', 'shown', 'showed', 'demonstrate', 'demonstrated', 'demonstrates',
    'novel', 'new', 'approach', 'method', 'methods', 'proposed', 'propose', 'investigated',
    'investigate', 'significantly', 'significant', 'excellent', 'enhanced', 'improved',
    'improve', 'effect', 'effects', 'different', 'various', 'compared', 'obtained',
    'present', 'presented', 'presents', 'findings', 'potential', 'promising',
    'respectively', 'herein', 'achieve', 'achieved', 'achieving', 'exhibits', 'exhibit',
    'exhibited', 'mah', 'cm', 'ma', 'mv', 'wh', 'kg', 'mg', 'nm', 'ms', 'rt', 'vs',
    'good', 'great', 'large', 'low', 'lower', 'key', 'important', 'provide', 'provides',
    'provided', 'including', 'furthermore', 'moreover', 'thus', 'also', 'et', 'al', 'doi',
    'figure', 'fig', 'formula', 'text', 'ensuremath', 'mathrm', 'sub', 'sup',
})

Embedder = Callable[[Sequence[str]], NDArray[np.float64]]


class EmergenceDocument(TypedDict):
    """One corpus paper prepared for emergence analysis."""

    paper_id: str
    doi: str
    title: str
    abstract: str
    year: int
    cited_by_count: float


@dataclass(frozen=True)
class EmergenceSettings:
    """Time windows, cluster granularity, and thresholds of an analysis.

    Parameters
    ----------
    growth_window : int, default=8
        Years ending at the analysis year over which share growth is fitted.
    recent_window : int, default=3
        Final years counted as recent for recency lift.
    niche_size : int, default=40
        Target mean papers per niche, the finest clusters.
    topic_size : int, default=250
        Target mean papers per topic.
    areas : int, default=12
        Number of areas, the coarsest clusters.
    min_papers : int, default=8
        Dated papers a niche needs before it is scored.
    min_recent : int, default=3
        Recent papers a niche needs before it is scored.
    min_term_recent : int, default=5
        Recent documents a term needs before it is scored.
    novelty_neighbours : int, default=5
        Earlier papers compared with each paper to measure novelty.
    novelty_min_prior : int, default=100
        Earlier papers required before novelty is measured for a year.
    horizon : int, default=3
        Years after each backtest cutoff over which growth is checked.
    backtest_cutoffs : int, default=4
        Number of backtest cutoff years; ``0`` skips the backtest.
    backtest_top : int, default=5
        Top-ranked niches compared with the rest in each backtest.
    timeline_years : int, default=6
        Earlier years, ending the year before the analysis year, at which the
        corpus is re-clustered to date when each niche was first flagged;
        ``0`` skips the timeline.
    """

    growth_window: int = 8
    recent_window: int = 3
    niche_size: int = 40
    topic_size: int = 250
    areas: int = 12
    min_papers: int = 8
    min_recent: int = 3
    min_term_recent: int = 5
    novelty_neighbours: int = 5
    novelty_min_prior: int = 100
    horizon: int = 3
    backtest_cutoffs: int = 4
    backtest_top: int = 5
    timeline_years: int = 6

    def validate(self) -> None:
        """Reject settings that cannot define a meaningful analysis.

        Raises
        ------
        ValueError
            If a window, size, or threshold is out of range.
        """
        positive = ('growth_window', 'recent_window', 'niche_size', 'topic_size', 'areas',
                    'min_papers', 'min_recent', 'min_term_recent', 'novelty_neighbours',
                    'horizon', 'backtest_top')
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be at least 1.')
        if self.growth_window < 3:
            raise ValueError('growth_window must be at least 3 years to fit a trend.')
        if self.recent_window >= self.growth_window:
            raise ValueError('recent_window must be shorter than growth_window.')
        if self.niche_size >= self.topic_size:
            raise ValueError('niche_size must be smaller than topic_size.')
        if self.areas < 2:
            raise ValueError('areas must be at least 2.')
        if self.backtest_cutoffs < 0 or self.novelty_min_prior < 0 or self.timeline_years < 0:
            raise ValueError('backtest_cutoffs, novelty_min_prior and timeline_years must not '
                             'be negative.')


# --------------------------------------------------------------------------- documents


def _clean_text(value: object) -> str:
    """Return unescaped, tag-free, whitespace-normalised text."""
    if value is None:
        return ''
    text = html.unescape(str(value))
    text = unicodedata.normalize('NFKC', text)
    text = _TAG_RE.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_abstract(value: object) -> str:
    """Strip markup, an "Abstract" heading, and publisher boilerplate.

    Parameters
    ----------
    value : object
        Abstract text or a value coercible to text.

    Returns
    -------
    str
        Abstract text without markup, heading, or image-download furniture.
    """
    text = _ABSTRACT_PREFIX_RE.sub('', _clean_text(value))
    return re.sub(r'\s+', ' ', _BOILERPLATE_RE.sub(' ', text)).strip()


def _publication_year(value: object) -> int:
    """Return the first plausible year in a date string, or 0."""
    match = _YEAR_RE.search(str(value or ''))
    return int(match.group(0)) if match else 0


def screen_documents(
    documents: Sequence[EmergenceDocument],
) -> tuple[list[EmergenceDocument], dict[str, int]]:
    """Drop patent-like and non-English records and repeated titles.

    Patents filed as articles and non-English abstracts form tight clusters
    of their own, and a duplicated record inflates one year's count.

    Parameters
    ----------
    documents : Sequence[EmergenceDocument]
        Candidate documents.

    Returns
    -------
    tuple of list[EmergenceDocument] and dict[str, int]
        Kept documents in paper-ID order, and counts of each reason a
        document was dropped.
    """
    dropped = {'patent_like': 0, 'non_english': 0, 'duplicate_title': 0}
    kept: list[EmergenceDocument] = []
    seen: set[tuple[str, int]] = set()
    # Documents with abstracts come first so a duplicate keeps the fuller record.
    for document in sorted(documents, key=lambda item: (not item['abstract'], item['paper_id'])):
        abstract = document['abstract']
        if _PATENT_RE.search(abstract):
            dropped['patent_like'] += 1
            continue
        words = _WORD_RE.findall(abstract.lower())
        if len(words) >= 20 and sum(word in ENGLISH_STOP_WORDS for word in words) / len(words) < 0.15:
            dropped['non_english'] += 1
            continue
        key = (re.sub(r'[^a-z0-9]', '', document['title'].lower()), document['year'])
        if key[0] and key in seen:
            dropped['duplicate_title'] += 1
            continue
        seen.add(key)
        kept.append(document)
    kept.sort(key=lambda item: item['paper_id'])
    return kept, dropped


def load_emergence_documents(
    db_path: str | PathLike[str],
    require_abstract: bool = True,
) -> tuple[list[EmergenceDocument], dict[str, int]]:
    """Load dated titles and abstracts from a corpus.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the SQLite paper corpus.
    require_abstract : bool, default=True
        Whether to drop papers without an abstract instead of using the title
        alone. Title-only embeddings are noisier, and in testing they weakened
        the backtest.

    Returns
    -------
    tuple of list[EmergenceDocument] and dict[str, int]
        Documents in paper-ID order and counts of papers read and dropped.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            'SELECT paper_id, doi, title, publication_date, cited_by_count '
            'FROM papers ORDER BY paper_id'
        ).fetchall()
        documents: list[EmergenceDocument] = []
        counts = {'papers_read': len(rows), 'no_year': 0, 'no_abstract': 0, 'no_text': 0}
        for row in rows:
            asset = get_asset(conn, row['paper_id'], 'abstract')
            abstract = clean_abstract(
                asset['content'].decode('utf-8', errors='replace') if asset else ''
            )
            title = _clean_text(row['title'])
            year = _publication_year(row['publication_date'])
            if not year:
                counts['no_year'] += 1
                continue
            if require_abstract and not abstract:
                counts['no_abstract'] += 1
                continue
            if not (title or abstract):
                counts['no_text'] += 1
                continue
            cites = row['cited_by_count']
            documents.append({
                'paper_id': row['paper_id'],
                'doi': row['doi'] or '',
                'title': title,
                'abstract': abstract,
                'year': year,
                'cited_by_count': math.nan if cites is None else float(cites),
            })
    finally:
        conn.close()
    documents, dropped = screen_documents(documents)
    counts.update(dropped)
    return documents, counts


# --------------------------------------------------------------------------- embeddings


def transformer_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    device: str | None = None,
) -> Embedder:
    """Build a Hugging Face encoder that embeds texts by CLS pooling.

    Parameters
    ----------
    model_name : str, default='allenai/specter2_base'
        Hugging Face model identifier. SPECTER2 is trained on citation links
        between scientific papers, so nearby vectors are related papers.
    batch_size : int, default=32
        Texts encoded per forward pass.
    device : str or None, optional
        Torch device; defaults to CUDA when available, otherwise CPU.

    Returns
    -------
    Callable[[Sequence[str]], numpy.ndarray]
        Function returning one row per input text.

    Raises
    ------
    RuntimeError
        If PyTorch is not installed.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            'Embedding papers requires PyTorch. Install it with '
            "pip install 'paperminertoolkit[bert]'."
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    resolved_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(model_name).to(resolved_device).eval()

    def embed(texts: Sequence[str]) -> NDArray[np.float64]:
        """Encode texts in length-sorted batches to limit padding.

        Parameters
        ----------
        texts : Sequence[str]
            Texts to encode.

        Returns
        -------
        numpy.ndarray
            CLS vectors, one row per text in input order.
        """
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        vectors = np.zeros((len(texts), model.config.hidden_size))
        with torch.inference_mode():
            for start in tqdm(range(0, len(order), batch_size), desc='Embedding papers',
                              unit='batch'):
                chunk = order[start:start + batch_size]
                batch = tokenizer([texts[index] for index in chunk], padding=True,
                                  truncation=True, max_length=512,
                                  return_tensors='pt').to(resolved_device)
                output = model(**batch).last_hidden_state[:, 0]
                vectors[chunk] = output.float().cpu().numpy()
        return vectors

    return embed


def embed_documents(
    texts: Sequence[str],
    cache_path: str | PathLike[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Embedder | None = None,
    batch_size: int = 32,
) -> tuple[NDArray[np.float64], int]:
    """Embed texts, reusing vectors cached under the model and text.

    Parameters
    ----------
    texts : Sequence[str]
        Texts to embed.
    cache_path : str or os.PathLike[str]
        ``.npz`` cache read before and rewritten after embedding.
    model_name : str, default='allenai/specter2_base'
        Model identifier, part of each cache key.
    embedder : Callable or None, optional
        Embedding function; :func:`transformer_embedder` is built only when
        some text is not cached.
    batch_size : int, default=32
        Batch size for the default embedder.

    Returns
    -------
    tuple of numpy.ndarray and int
        Unit-length vectors, one row per text, and the number computed now.
    """
    path = Path(cache_path)
    keys = [hashlib.sha1(f'{model_name}\0{text}'.encode()).hexdigest() for text in texts]
    cached: dict[str, NDArray[np.float64]] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            cached = dict(zip(data['keys'].tolist(), data['vectors']))
    # One index per uncached key, so repeated texts are embedded once.
    missing: dict[str, int] = {}
    for index, key in enumerate(keys):
        if key not in cached:
            missing.setdefault(key, index)
    if missing:
        embed = embedder or transformer_embedder(model_name, batch_size=batch_size)
        vectors = np.asarray(embed([texts[index] for index in missing.values()]), dtype=np.float64)
        for key, vector in zip(missing, vectors):
            cached[key] = vector
        all_keys = list(cached)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            np.savez(handle, keys=np.array(all_keys),
                     vectors=np.stack([cached[key] for key in all_keys]))
    matrix = np.stack([cached[key] for key in keys]).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0), len(missing)


def _embedding_text(document: EmergenceDocument) -> str:
    """Join title and abstract the way SPECTER was trained to read them."""
    if not document['abstract']:
        return document['title']
    return f'{document["title"]}[SEP]{document["abstract"]}'


# --------------------------------------------------------------------------- statistics


def growth_trend(
    counts: NDArray[Any],
    totals: NDArray[Any],
    years: NDArray[Any],
) -> tuple[float, float]:
    """Fit the yearly growth of a share with a quasi-Poisson model.

    The model is ``log E[count] = a + b * year + log(total)``, so ``b`` is the
    log of the yearly multiplier of the share. Its standard error is inflated
    by the Pearson dispersion when the counts vary more than Poisson.

    Parameters
    ----------
    counts : numpy.ndarray
        Papers in the cluster each year.
    totals : numpy.ndarray
        Papers in the corpus each year.
    years : numpy.ndarray
        The years.

    Returns
    -------
    tuple of float and float
        Slope ``b`` and its standard error; both NaN when fewer than three
        papers or years leave nothing to fit.
    """
    mask = np.asarray(totals) > 0
    y = np.asarray(counts, dtype=float)[mask]
    n = np.asarray(totals, dtype=float)[mask]
    t = np.asarray(years, dtype=float)[mask]
    if y.sum() < 3 or len(y) < 3:
        return math.nan, math.nan
    fit = _fit_counts(y, n, t - t.mean(), degree=1)
    if fit is None:
        return math.nan, math.nan
    beta, covariance = fit
    return float(beta[1]), float(math.sqrt(max(covariance[1, 1], 0.0)))


def _fit_counts(
    y: NDArray[np.float64],
    n: NDArray[np.float64],
    t: NDArray[np.float64],
    degree: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Fit ``log E[y] = poly(t) + log(n)`` by quasi-Poisson regression.

    Returns the coefficients, lowest power first, and their covariance
    inflated by the Pearson dispersion, or ``None`` when the fit is singular.
    """
    design = np.column_stack([t ** power for power in range(degree + 1)])
    offset = np.log(n)
    beta = np.zeros(degree + 1)
    beta[0] = math.log(y.sum() / n.sum())
    # Damped steps and clipped coefficients (a slope of 20-fold a year) keep a
    # cluster that appears only in the last year finite.
    limits = np.array([np.inf, 3.0, 1.0][:degree + 1])
    try:
        for _ in range(100):
            mu = np.exp(design @ beta + offset)
            step = np.linalg.solve(design.T @ (design * mu[:, None]), design.T @ (y - mu))
            step *= min(1.0, 1.0 / max(float(np.abs(step).max()), 1e-12))
            beta = np.clip(beta + step, -limits, limits)
            if np.abs(step).max() < 1e-9:
                break
        mu = np.exp(design @ beta + offset)
        covariance = np.linalg.inv(design.T @ (design * mu[:, None]))
    except np.linalg.LinAlgError:
        return None
    dispersion = max(1.0, float(((y - mu) ** 2 / mu).sum()) / max(len(y) - len(beta), 1))
    return beta, covariance * dispersion


def trend_shape(
    counts: NDArray[Any],
    totals: NDArray[Any],
    years: NDArray[Any],
) -> dict[str, Any]:
    """Describe whether a share's yearly growth sped up or slowed down.

    The model is ``log E[count] = a + b * t + c * t**2 + log(total)`` with
    ``t`` the centred year, a quasi-Poisson fit like :func:`growth_trend`.
    Its slope ``b + 2 * c * t`` gives the share's yearly multiplier at the
    start and end of the window, and the curvature ``c``, over its standard
    error, decides the shape:

    - curvature z >= 2: ``accelerating``, ``rebounding`` (falling at the
      start, rising at the end), or ``decline easing`` (still falling);
    - curvature z <= -2: ``growth slowing`` (still rising), ``peaked``
      (rising at the start, falling at the end), or ``decline steepening``;
    - otherwise ``steady``.

    Parameters
    ----------
    counts : numpy.ndarray
        Papers in the cluster each year.
    totals : numpy.ndarray
        Papers in the corpus each year.
    years : numpy.ndarray
        The years.

    Returns
    -------
    dict[str, Any]
        ``start_growth`` and ``end_growth``, the fitted yearly share
        multipliers in the first and last year; ``curvature_z``;
        ``turning_year``, the year the fitted share peaked or bottomed out, for
        those shapes, else ``''``; and ``shape``, ``''`` when fewer than five
        papers or years leave nothing to fit.
    """
    shape: dict[str, Any] = {'start_growth': math.nan, 'end_growth': math.nan,
                             'curvature_z': math.nan, 'turning_year': '', 'shape': ''}
    mask = np.asarray(totals) > 0
    y = np.asarray(counts, dtype=float)[mask]
    n = np.asarray(totals, dtype=float)[mask]
    t = np.asarray(years, dtype=float)[mask]
    if y.sum() < 5 or len(y) < 5:
        return shape
    centre = float(t.mean())
    t = t - centre
    fit = _fit_counts(y, n, t, degree=2)
    if fit is None:
        return shape
    (_, b, c), covariance = fit
    se = math.sqrt(max(covariance[2, 2], 0.0))
    if se <= 0:
        return shape
    z = float(c / se)
    start = math.exp(b + 2 * c * t.min())
    end = math.exp(b + 2 * c * t.max())
    label = 'steady'
    if z >= 2:
        label = 'decline easing' if end < 1 else 'rebounding' if start < 1 else 'accelerating'
    elif z <= -2:
        label = 'growth slowing' if end > 1 else 'peaked' if start > 1 else 'decline steepening'
    shape.update(start_growth=start, end_growth=end, curvature_z=z, shape=label)
    if label in ('peaked', 'rebounding'):
        # The slope changes sign inside the window, at the vertex of the parabola.
        shape['turning_year'] = int(round(centre - b / (2 * c)))
    return shape


def classify_trend(growth_z: float, recency_lift: float, recent_papers: int) -> str:
    """Label how a cluster's share of the corpus is changing.

    Parameters
    ----------
    growth_z : float
        Share-growth slope divided by its standard error.
    recency_lift : float
        Recent fraction of the cluster over the recent fraction of the corpus.
    recent_papers : int
        Cluster papers in the recent window.

    Returns
    -------
    str
        ``emerging`` (growth z >= 2 and recency lift >= 1.5), ``growing``
        (growth z >= 2), ``declining`` (growth z <= -2), ``stable``, or
        ``too small``.
    """
    if not math.isfinite(growth_z) or recent_papers < 3:
        return 'too small'
    if growth_z >= 2.0 and recency_lift >= 1.5:
        return 'emerging'
    if growth_z >= 2.0:
        return 'growing'
    if growth_z <= -2.0:
        return 'declining'
    return 'stable'


def _percentile_rank(values: Sequence[float]) -> NDArray[np.float64]:
    """Rank values onto [0, 1], placing NaN lowest."""
    array = np.asarray(values, dtype=float)
    filled = np.where(np.isfinite(array), array, -np.inf)
    return (rankdata(filled, method='average') - 1) / max(len(array) - 1, 1)


def _paper_novelty(X: NDArray[np.float64], years: NDArray[np.int64], neighbours: int,
                   min_prior: int) -> NDArray[np.float64]:
    """Return each paper's distance from its nearest earlier papers, z-scored by year."""
    novelty = np.full(len(X), np.nan)
    for year in np.unique(years):
        rows = np.where(years == year)[0]
        prior = np.where(years < year)[0]
        if len(prior) < max(min_prior, neighbours):
            continue
        similarity = X[rows] @ X[prior].T
        nearest = np.partition(similarity, -neighbours, axis=1)[:, -neighbours:]
        novelty[rows] = 1.0 - nearest.mean(axis=1)
    z = np.full(len(X), np.nan)
    for year in np.unique(years):
        rows = np.where((years == year) & np.isfinite(novelty))[0]
        if len(rows) >= 5:
            spread = novelty[rows].std()
            z[rows] = (novelty[rows] - novelty[rows].mean()) / spread if spread > 0 else 0.0
    return z


def _normalised_citations(cites: NDArray[np.float64], years: NDArray[np.int64]) -> NDArray[np.float64]:
    """Divide citation counts by the median of papers from the same year."""
    out = np.full(len(cites), np.nan)
    for year in np.unique(years):
        rows = np.where((years == year) & np.isfinite(cites))[0]
        if len(rows) >= 5:
            out[rows] = cites[rows] / max(float(np.median(cites[rows])), 1.0)
    return out


# --------------------------------------------------------------------------- clusters


def _cut(tree: NDArray[np.float64], clusters: int) -> NDArray[np.int64]:
    """Cut a linkage tree into at most ``clusters`` zero-based clusters."""
    return fcluster(tree, t=clusters, criterion='maxclust').astype(np.int64) - 1


def _cluster_rows(
    labels: NDArray[np.int64],
    X: NDArray[np.float64],
    years: NDArray[np.int64],
    novelty_z: NDArray[np.float64],
    cites: NDArray[np.float64],
    end_year: int,
    settings: EmergenceSettings,
) -> tuple[list[dict[str, Any]], dict[int, NDArray[np.float64]]]:
    """Measure growth, recency, novelty, and impact for every cluster.

    Parameters
    ----------
    labels : numpy.ndarray
        Cluster of each paper.
    X : numpy.ndarray
        Unit-length paper embeddings.
    years : numpy.ndarray
        Publication year of each paper.
    novelty_z : numpy.ndarray
        Within-year novelty z score of each paper.
    cites : numpy.ndarray
        Year-normalised citation count of each paper.
    end_year : int
        Last year analysed; later papers are ignored for timing.
    settings : EmergenceSettings
        Windows and thresholds.

    Returns
    -------
    tuple of list[dict[str, Any]] and dict[int, numpy.ndarray]
        One row per cluster, and each cluster's unit-length centroid.
    """
    valid = years <= end_year
    window = np.arange(end_year - settings.growth_window + 1, end_year + 1)
    totals = np.array([np.count_nonzero(valid & (years == year)) for year in window])
    recent = valid & (years > end_year - settings.recent_window)
    recent_share = recent.sum() / max(valid.sum(), 1)
    rows: list[dict[str, Any]] = []
    centroids: dict[int, NDArray[np.float64]] = {}
    for cluster_id in np.unique(labels):
        members = labels == cluster_id
        centroid = X[members].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids[int(cluster_id)] = centroid
        dated = members & valid
        size = int(dated.sum())
        recent_papers = int((dated & recent).sum())
        counts = np.array([np.count_nonzero(dated & (years == year)) for year in window])
        slope, se = growth_trend(counts, totals, window)
        growth_z = slope / se if math.isfinite(slope) and se > 0 else math.nan
        shape = trend_shape(counts, totals, window)
        member_years = years[dated]
        novel = np.where(dated & np.isfinite(novelty_z))[0]
        if len(novel) >= 3:
            novel = novel[np.argsort(years[novel], kind='stable')]
            pioneer = float(novelty_z[novel[:max(3, len(novel) // 3)]].mean())
        else:
            pioneer = math.nan
        spike = False
        if size >= 6:
            values, value_counts = np.unique(member_years, return_counts=True)
            peak = int(np.argmax(value_counts))
            # Most papers in one year before the last is a batch (a patent
            # family, a special issue, duplicates), not a trend.
            spike = int(values[peak]) < end_year and value_counts[peak] / size >= 0.5
        impact = cites[members]
        rows.append({
            'cluster': int(cluster_id),
            'papers': int(members.sum()),
            'dated_papers': size,
            'recent_papers': recent_papers,
            'first_year_p10': int(np.percentile(member_years, 10)) if size else '',
            'median_year': float(np.median(member_years)) if size else math.nan,
            'share_growth_per_year': math.exp(slope) if math.isfinite(slope) else math.nan,
            'growth_z': growth_z,
            'recency_lift': (recent_papers / size) / recent_share if size and recent_share else math.nan,
            'pioneer_novelty': pioneer,
            'coherence': float((X[members] @ centroid).mean()),
            'impact': float(np.nanmedian(impact)) if np.isfinite(impact).any() else math.nan,
            'counts_by_year': ' '.join(str(int(count)) for count in counts),
            'shape': shape['shape'],
            'turning_year': shape['turning_year'],
            'start_growth': shape['start_growth'],
            'end_growth': shape['end_growth'],
            'curvature_z': shape['curvature_z'],
            'spike': bool(spike),
            'eligible': (size >= settings.min_papers and recent_papers >= settings.min_recent
                         and not spike),
        })
    eligible = [row for row in rows if row['eligible']]
    if eligible:
        ranks = [_percentile_rank([row[key] for row in eligible])
                 for key in ('growth_z', 'recency_lift')]
        for index, row in enumerate(eligible):
            row['score'] = float(np.mean([rank[index] for rank in ranks]))
            row['flag'] = bool(row['growth_z'] >= 2.0 and row['recency_lift'] >= 1.5)
    for row in rows:
        row.setdefault('score', math.nan)
        row.setdefault('flag', False)
        row['status'] = classify_trend(row['growth_z'], row['recency_lift'], row['recent_papers'])
        if row['status'] == 'too small':
            row['shape'], row['turning_year'] = '', ''
    return rows, centroids


def _document_terms(texts: Sequence[str], stopwords: set[str]) -> tuple[sparse.csr_matrix, NDArray[Any]]:
    """Count uni- and bigrams, dropping words found in most documents.

    Without ``max_df`` the search query's own words head every cluster label.
    """
    vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
                                 stop_words=sorted(stopwords), min_df=3, max_df=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        matrix = vectorizer.fit_transform(texts).tocsr()
    return matrix, vectorizer.get_feature_names_out()


def _membership(labels: NDArray[np.int64]) -> tuple[sparse.csr_matrix, NDArray[np.int64]]:
    """Return a cluster-by-document indicator matrix and its cluster IDs."""
    ids = np.unique(labels)
    index = np.searchsorted(ids, labels)
    matrix = sparse.csr_matrix((np.ones(len(labels)), (index, np.arange(len(labels)))),
                               shape=(len(ids), len(labels)))
    return matrix, ids


def _cluster_labels(D: sparse.csr_matrix, labels: NDArray[np.int64], terms: NDArray[Any],
                    top_n: int) -> dict[int, list[str]]:
    """Label clusters by class-based TF-IDF, as BERTopic does."""
    membership, ids = _membership(labels)
    M = np.asarray((membership @ D).todense(), dtype=float)
    tf = M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)
    idf = np.log(1.0 + (M.sum() / len(ids)) / np.maximum(M.sum(axis=0), 1.0))
    weights = tf * idf
    return {
        int(cluster_id): [str(terms[j]) for j in np.argsort(-weights[i])[:top_n] if M[i, j] > 0]
        for i, cluster_id in enumerate(ids)
    }


def _niche_spread(binary: sparse.spmatrix,
                  niches: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure how far each term's documents gather in a few niches.

    Parameters
    ----------
    binary : scipy.sparse.spmatrix
        Document-by-term indicator matrix.
    niches : numpy.ndarray
        Niche of each document.

    Returns
    -------
    tuple of numpy.ndarray and numpy.ndarray
        *Concentration*, one minus the entropy of each term's documents over
        niches divided by its maximum, ``log(niches)``; and *dispersion*,
        Pearson's chi-square of the term's niche counts against niche sizes
        over its degrees of freedom. A term spread at random has a dispersion
        near 1 however common it is, while its concentration rises as it
        gets rarer; a common term gathered mildly has a high dispersion but a
        low concentration.
    """
    membership, ids = _membership(niches)
    sizes = np.asarray(membership.sum(axis=1)).ravel()
    counts = sparse.csr_matrix(membership @ binary)
    docs = np.maximum(np.asarray(counts.sum(axis=0)).ravel(), 1.0)
    shares = (counts @ sparse.diags(1.0 / docs)).tocsr()
    shares.data = shares.data * np.log(shares.data)
    concentration = 1.0 + np.asarray(shares.sum(axis=0)).ravel() / math.log(max(len(ids), 2))
    # Pearson's sum of (observed - expected)^2 / expected, with expected = docs * size / total,
    # equals total / docs * sum(observed^2 / size) - docs.
    squares = sparse.diags(1.0 / sizes) @ counts.multiply(counts)
    chi_square = sizes.sum() / docs * np.asarray(squares.sum(axis=0)).ravel() - docs
    return concentration, chi_square / max(len(ids) - 1, 1)


def _term_rows(
    D: sparse.csr_matrix,
    terms: NDArray[Any],
    years: NDArray[np.int64],
    niches: NDArray[np.int64],
    end_year: int,
    watch_terms: Sequence[str],
    settings: EmergenceSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score growing terms and count watched terms by year.

    Terms are ranked by growth z times their concentration, one minus the
    normalised entropy of the term over niches. Phrasing that is spreading
    through every abstract ("enables", "strategy") grows too, but evenly.
    """
    valid = np.where(years <= end_year)[0]
    all_years = np.arange(int(years[valid].min()), end_year + 1)
    by_year = sparse.csr_matrix(
        (np.ones(len(valid)), (years[valid] - all_years[0], valid)),
        shape=(len(all_years), len(years)),
    )
    binary = (D > 0).astype(np.float64).tocsc()
    counts = np.asarray((by_year @ binary).todense())
    totals = np.asarray(by_year.sum(axis=1)).ravel()
    window = all_years > end_year - settings.growth_window
    recent = all_years > end_year - settings.recent_window
    recent_share = totals[recent].sum() / max(totals.sum(), 1)
    docs_total = counts.sum(axis=0)
    docs_recent = counts[recent].sum(axis=0)
    candidates = np.where(docs_recent >= settings.min_term_recent)[0]
    concentration, _ = _niche_spread(binary[:, candidates], niches)
    rows = []
    for position, j in enumerate(candidates):
        slope, se = growth_trend(counts[window, j], totals[window], all_years[window])
        growth_z = slope / se if math.isfinite(slope) and se > 0 else math.nan
        rows.append({
            'term': str(terms[j]),
            'docs': int(docs_total[j]),
            'recent_docs': int(docs_recent[j]),
            'first_year': int(all_years[np.argmax(counts[:, j] > 0)]),
            'share_growth_per_year': math.exp(slope) if math.isfinite(slope) else math.nan,
            'growth_z': growth_z,
            'concentration': float(concentration[position]),
            'term_score': growth_z * float(concentration[position]),
            'recency_lift': (docs_recent[j] / docs_total[j]) / recent_share if recent_share else math.nan,
            'counts_by_year': ' '.join(str(int(count)) for count in counts[window, j]),
        })
    rows.sort(key=lambda row: -row['term_score'] if math.isfinite(row['term_score']) else math.inf)
    lookup = {str(term): j for j, term in enumerate(terms)}
    watch_rows = []
    for term in watch_terms:
        j = lookup.get(term.lower())
        row: dict[str, Any] = {'term': term, 'docs': int(docs_total[j]) if j is not None else 0}
        for i, year in enumerate(all_years):
            row[str(int(year))] = int(counts[i, j]) if j is not None else ''
        watch_rows.append(row)
    return rows, watch_rows


def _topic_term_rows(
    D: sparse.csr_matrix,
    terms: NDArray[Any],
    years: NDArray[np.int64],
    topics: NDArray[np.int64],
    niches: NDArray[np.int64],
    end_year: int,
    settings: EmergenceSettings,
    keep: int = 10,
) -> list[dict[str, Any]]:
    """Find the terms rising or fading inside each topic over the growth window.

    A term's yearly count of topic papers using it is fitted against the
    topic's yearly paper count, the share model used for clusters, so the
    trend is the term's share of the topic's papers. A rising term must also be
    at least :data:`TOPIC_TERM_LIFT` times as common in the topic's recent
    papers as in the corpus's, and a fading one in its earlier papers. Only
    field vocabulary counts: terms gathered in a few niches, with a
    concentration of at least :data:`TOPIC_TERM_CONCENTRATION` and a
    dispersion of at least :data:`TOPIC_TERM_DISPERSION` (see
    :func:`_niche_spread`). Without it, the drift in academic wording
    ("crucial", "insights", "comprehensive") rises inside most topics.

    Parameters
    ----------
    D : scipy.sparse.csr_matrix
        Document-term counts.
    terms : numpy.ndarray
        Term of each column of ``D``.
    years : numpy.ndarray
        Publication year of each document.
    topics : numpy.ndarray
        Topic of each document.
    niches : numpy.ndarray
        Niche of each document, over which field vocabulary is judged.
    end_year : int
        Last year of the growth window.
    settings : EmergenceSettings
        Windows, and the documents a term needs in the topic.
    keep : int, default=10
        Rising and fading terms kept per topic.

    Returns
    -------
    list[dict[str, Any]]
        Up to ``keep`` rising and ``keep`` fading terms per topic, strongest
        first.
    """
    window = np.arange(end_year - settings.growth_window + 1, end_year + 1)
    inside = np.where((years >= window[0]) & (years <= end_year))[0]
    binary = (D[inside] > 0).astype(np.float64).tocsr()
    position = years[inside] - window[0]
    recent = window > end_year - settings.recent_window
    by_year = sparse.csr_matrix((np.ones(len(inside)), (position, np.arange(len(inside)))),
                                shape=(len(window), len(inside)))
    corpus = np.asarray((by_year @ binary).todense())
    corpus_totals = np.bincount(position, minlength=len(window)).astype(float)
    corpus_rate = {part: corpus[mask].sum(axis=0) / max(corpus_totals[mask].sum(), 1.0)
                   for part, mask in (('rising', recent), ('fading', ~recent))}
    field = np.zeros(len(terms), dtype=bool)
    common = np.where(corpus.sum(axis=0) >= settings.min_term_recent)[0]
    if len(common):
        concentration, dispersion = _niche_spread((D[:, common] > 0).astype(np.float64), niches)
        field[common] = ((concentration >= TOPIC_TERM_CONCENTRATION)
                         & (dispersion >= TOPIC_TERM_DISPERSION))
    topic_of = topics[inside]
    rows: list[dict[str, Any]] = []
    for topic in np.unique(topics):
        members = np.where(topic_of == topic)[0]
        if not len(members):
            continue
        counts = np.asarray((by_year[:, members] @ binary[members]).todense())
        totals = np.bincount(position[members], minlength=len(window)).astype(float)
        rate = {part: counts[mask].sum(axis=0) / max(totals[mask].sum(), 1.0)
                for part, mask in (('rising', recent), ('fading', ~recent))}
        for direction, mask, sign in (('rising', recent, 1.0), ('fading', ~recent, -1.0)):
            docs = counts[mask].sum(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                lift = rate[direction] / corpus_rate[direction]
            other = 'fading' if direction == 'rising' else 'rising'
            candidates = np.where(field & (docs >= settings.min_term_recent)
                                  & (lift >= TOPIC_TERM_LIFT) & (rate[direction] > rate[other]))[0]
            found = []
            for j in candidates:
                slope, se = growth_trend(counts[:, j], totals, window)
                if not (math.isfinite(slope) and se > 0) or sign * slope / se < 2.0:
                    continue
                found.append({
                    'topic': int(topic), 'direction': direction, 'term': str(terms[j]),
                    'topic_docs': int(counts[:, j].sum()), 'recent_docs': int(counts[recent, j].sum()),
                    'share_growth_per_year': math.exp(slope), 'growth_z': slope / se,
                    'lift': float(lift[j]),
                    'counts_by_year': ' '.join(str(int(count)) for count in counts[:, j]),
                })
            found.sort(key=lambda row: -abs(row['growth_z']))
            rows += found[:keep]
    return rows


# --------------------------------------------------------------------------- backtest


def _rank_correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Spearman correlation of each signal with later share change."""
    target = np.array([row['future_log_ratio'] for row in rows], dtype=float)
    out = {}
    for key, name in (('score', 'rho_score'), ('growth_z', 'rho_growth_z'),
                      ('recency_lift', 'rho_recency_lift'),
                      ('pioneer_novelty', 'rho_novelty'), ('dated_papers', 'rho_size')):
        values = np.array([row[key] for row in rows], dtype=float)
        mask = np.isfinite(values) & np.isfinite(target)
        rho = math.nan
        if mask.sum() >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rho = float(spearmanr(values[mask], target[mask])[0])
        out[name] = rho
    return out


_Snapshot = dict[str, Any]


def _cutoff_snapshot(
    X: NDArray[np.float64],
    years: NDArray[np.int64],
    novelty_z: NDArray[np.float64],
    cutoff: int,
    settings: EmergenceSettings,
) -> _Snapshot | None:
    """Re-cluster and score the papers published up to ``cutoff``.

    This is the niche analysis as it would have run in ``cutoff``. Novelty only
    uses earlier papers, so it is known then; citations are not, and are left
    out. Returns ``None`` when too few papers or years precede the cutoff.
    """
    past = np.where(years <= cutoff)[0]
    if (len(past) < 4 * settings.niche_size
            or len(np.unique(years[past])) < settings.growth_window):
        return None
    tree = linkage(X[past], method='ward')
    labels = _cut(tree, max(2, round(len(past) / settings.niche_size)))
    rows, centroids = _cluster_rows(labels, X[past], years[past], novelty_z[past],
                                    np.full(len(past), np.nan), cutoff, settings)
    return {'past': past, 'labels': labels, 'rows': rows, 'centroids': centroids}


def _eligible_percentiles(rows: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """Rank the scored clusters' scores from 0 (lowest) to 100 (highest)."""
    eligible = [row for row in rows if row['eligible'] and math.isfinite(row['score'])]
    ranks = _percentile_rank([row['score'] for row in eligible])
    return {int(row['cluster']): float(rank * 100) for row, rank in zip(eligible, ranks)}


def _timeline(
    X: NDArray[np.float64],
    niches: NDArray[np.int64],
    years: NDArray[np.int64],
    novelty_z: NDArray[np.float64],
    niche_rows: Sequence[dict[str, Any]],
    end_year: int,
    settings: EmergenceSettings,
    snapshot: Callable[[int], _Snapshot | None],
) -> list[dict[str, Any]]:
    """Date when each niche was first flagged and when the analysis could have seen it.

    For each of the ``timeline_years`` years before ``end_year``, and for
    ``end_year`` itself, two things are recorded:

    - Today's niches are scored on their papers published up to that year, as
      the analysis would have scored them then (``percentile``, ``flag``).
      This uses today's cluster boundaries, which were not known then.
    - The corpus is re-clustered as it stood that year, and each niche is
      linked to the niche of that year holding most of its papers, once
      :data:`LINEAGE_MIN_PAPERS` of them existed (``earlier_niche``,
      ``earlier_percentile``, ``earlier_flag``). ``overlap`` is the fraction of
      the niche's papers that earlier niche held, and ``share`` the fraction
      of the earlier niche they made up.

    Sets ``first_flagged`` and ``last_flagged`` from the first, and
    ``first_seen`` and ``first_seen_share`` (the first year a flagged earlier
    niche held most of the niche's papers) from the second, on every niche row.
    """
    current = {int(row['cluster']): row for row in niche_rows}
    no_cites = np.full(len(years), np.nan)
    entries: list[dict[str, Any]] = []
    for cutoff in range(end_year - settings.timeline_years, end_year + 1):
        own_rows: Sequence[Mapping[str, Any]] = (
            niche_rows if cutoff == end_year
            else _cluster_rows(niches, X, years, novelty_z, no_cites, cutoff, settings)[0])
        own = {int(row['cluster']): row for row in own_rows}
        own_percentiles = _eligible_percentiles(own_rows)
        snap = snapshot(cutoff) if cutoff < end_year else None
        label_then = np.full(len(years), -1, dtype=np.int64)
        rows_then: dict[int, Mapping[str, Any]] = {}
        percentiles_then: dict[int, float] = {}
        if snap is not None:
            label_then[snap['past']] = snap['labels']
            rows_then = {int(row['cluster']): row for row in snap['rows']}
            percentiles_then = _eligible_percentiles(snap['rows'])
        for niche_id in current:
            row = own[niche_id]
            if not row['dated_papers']:
                continue
            entry: dict[str, Any] = {
                'niche': niche_id, 'year': cutoff, 'papers_then': row['dated_papers'],
                'growth_z': row['growth_z'], 'recency_lift': row['recency_lift'],
                'percentile': own_percentiles.get(niche_id, math.nan),
                'eligible': row['eligible'], 'flag': row['flag'], 'earlier_niche': '',
                'overlap': math.nan, 'share': math.nan, 'earlier_percentile': math.nan,
                'earlier_flag': '',
            }
            if cutoff == end_year:
                entry.update(earlier_niche=niche_id, overlap=1.0, share=1.0,
                             earlier_percentile=entry['percentile'], earlier_flag=row['flag'])
            else:
                then = label_then[(niches == niche_id) & (label_then >= 0)]
                if len(then) >= LINEAGE_MIN_PAPERS:
                    held = np.bincount(then)
                    ancestor = int(held.argmax())
                    earlier = rows_then[ancestor]
                    entry.update(
                        earlier_niche=ancestor, overlap=float(held[ancestor] / len(then)),
                        share=float(held[ancestor] / max(earlier['dated_papers'], 1)),
                        earlier_percentile=percentiles_then.get(ancestor, math.nan),
                        earlier_flag=earlier['flag'])
            entries.append(entry)
    for niche_id, row in current.items():
        history = [entry for entry in entries if entry['niche'] == niche_id]
        flagged = [entry['year'] for entry in history if entry['flag']]
        seen = [entry for entry in history if entry['earlier_flag'] and entry['overlap'] >= 0.5]
        row['first_flagged'] = min(flagged) if flagged else ''
        row['last_flagged'] = max(flagged) if flagged else ''
        row['first_seen'] = seen[0]['year'] if seen else ''
        row['first_seen_share'] = seen[0]['share'] if seen else math.nan
    return sorted(entries, key=lambda entry: (entry['niche'], entry['year']))


def _backtest(
    X: NDArray[np.float64],
    years: NDArray[np.int64],
    end_year: int,
    settings: EmergenceSettings,
    snapshot: Callable[[int], _Snapshot | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check whether niche scores at earlier cutoffs predicted later growth.

    For each cutoff year the papers up to it are re-clustered and scored. The
    papers of the following ``horizon`` years are assigned to the nearest
    niche, and the log change of each niche's share from the recent window to
    the horizon is correlated with its score.
    """
    last = end_year - settings.horizon
    summaries: list[dict[str, Any]] = []
    pooled: list[dict[str, Any]] = []
    for cutoff in range(last - settings.backtest_cutoffs + 1, last + 1):
        future = np.where((years > cutoff) & (years <= cutoff + settings.horizon))[0]
        if len(future) < settings.niche_size:
            continue
        snap = snapshot(cutoff)
        if snap is None:
            continue
        past, labels, rows, centroids = (snap['past'], snap['labels'], snap['rows'],
                                         snap['centroids'])
        ids = np.array(sorted(centroids))
        assigned = ids[np.argmax(X[future] @ np.stack([centroids[i] for i in ids]).T, axis=1)]
        recent = years[past] > cutoff - settings.recent_window
        eligible = [row for row in rows if row['eligible']]
        for row in eligible:
            previous = np.count_nonzero((labels == row['cluster']) & recent)
            later = np.count_nonzero(assigned == row['cluster'])
            row['future_log_ratio'] = (math.log((later + 0.5) / len(future))
                                       - math.log((previous + 0.5) / max(int(recent.sum()), 1)))
            row['cutoff'] = cutoff
        pooled.extend(eligible)
        ranked = sorted(eligible, key=lambda row: -row['score'])
        top = [row['future_log_ratio'] for row in ranked[:settings.backtest_top]]
        rest = [row['future_log_ratio'] for row in ranked[settings.backtest_top:]]
        summary: dict[str, Any] = {'cutoff': cutoff, 'past_papers': len(past),
                                   'future_papers': len(future), 'eligible_niches': len(eligible)}
        summary.update(_rank_correlations(eligible))
        summary['top_mean_future_log_ratio'] = float(np.mean(top)) if top else math.nan
        summary['rest_mean_future_log_ratio'] = float(np.mean(rest)) if rest else math.nan
        summaries.append(summary)
    if pooled:
        total: dict[str, Any] = {'cutoff': 'pooled', 'past_papers': '', 'future_papers': '',
                                 'eligible_niches': len(pooled)}
        total.update(_rank_correlations(pooled))
        total['top_mean_future_log_ratio'] = math.nan
        total['rest_mean_future_log_ratio'] = math.nan
        summaries.append(total)
    return summaries, pooled


# --------------------------------------------------------------------------- output


def _yearly_counts(labels: NDArray[np.int64], years: NDArray[np.int64],
                   span: NDArray[np.int64]) -> NDArray[np.int64]:
    """Count each cluster's papers in each year of ``span``, one row per cluster id."""
    inside = (years >= span[0]) & (years <= span[-1])
    table = np.zeros((int(labels.max()) + 1, len(span)), dtype=np.int64)
    np.add.at(table, (labels[inside], years[inside] - span[0]), 1)
    return table


def _sparkline(values: Sequence[float]) -> str:
    """Draw values as eighth-height blocks scaled to their maximum."""
    array = np.asarray(values, dtype=float)
    peak = float(array.max()) if len(array) else 0.0
    if peak <= 0:
        return _SPARK[0] * len(array)
    return ''.join(_SPARK[min(7, int(value / peak * 8))] for value in array)


def _shape_text(row: Mapping[str, Any]) -> str:
    """Name a cluster's trend shape, with the turning year when it has one."""
    return f'{row["shape"]} {row["turning_year"]}'.strip()


def _format(value: object, digits: int = 2) -> str:
    """Format a table cell, leaving NaN blank."""
    if isinstance(value, float):
        return f'{value:.{digits}f}' if math.isfinite(value) else ''
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    """Write rows to CSV with the given or first row's columns."""
    columns = list(fields or (rows[0].keys() if rows else []))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a Markdown table."""
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    lines += ['| ' + ' | '.join(_format(cell).replace('|', '/') for cell in row) + ' |'
              for row in rows]
    return '\n'.join(lines)


def _timeline_section(niches: Sequence[Mapping[str, Any]], timeline: Sequence[Mapping[str, Any]],
                      top: int, span: str, figure: Sequence[str]) -> list[str]:
    """Render the flag history of today's top niches and of niches flagged only earlier."""
    if not timeline:
        return []
    years = sorted({int(entry['year']) for entry in timeline})
    cells = {(entry['niche'], entry['year']):
             (_format(entry['percentile'], 0) + ('*' if entry['flag'] else '')
              if entry['eligible'] else '-')
             for entry in timeline}

    def seen(niche: Mapping[str, Any]) -> str:
        """Give the year a flagged niche of its time first held most of this one."""
        if niche['first_seen'] == '':
            return ''
        return f'{niche["first_seen"]} ({niche["first_seen_share"]:.0%})'

    def terms(niche: Mapping[str, Any]) -> str:
        """Shorten a niche label to its first four terms."""
        return '; '.join(niche['label_terms'].split('; ')[:4])

    shown = [niche for rank, niche in enumerate(niches) if rank < top or niche['flag']]
    faded = sorted((niche for niche in niches if niche['last_flagged'] != '' and not niche['flag']),
                   key=lambda niche: -niche['last_flagged'])
    lines = [
        '## Emergence timeline', '',
        'Each year column scores today\'s niches on the papers published up to that year, as '
        'the analysis would have scored them then, and gives the percentile among the niches '
        'ranked that year (100 = top); * marks a flag as emerging and - a niche too small to '
        'rank. These columns use today\'s clusters, which were not known then. *First seen* is '
        'the realistic date: the corpus is re-clustered as it stood in each year, and it is the '
        'first year a flagged niche of that time held most of this niche\'s papers. The '
        'percentage is how much of that flagged niche this one made up; a small one means it '
        f'was seen only inside a broader field. Years before {years[0]} were not examined. '
        f'Rows are the top {top} niches and every niche flagged now, in rank order; '
        f'`{TIMELINE_FILENAME}` has every niche.', '',
        _markdown_table(
            ['niche', 'terms', 'status now', 'first flagged', 'first seen',
             *(str(year) for year in years)],
            [[niche['cluster'], terms(niche), niche['status'], niche['first_flagged'],
              seen(niche), *(cells.get((niche['cluster'], year), '') for year in years)]
             for niche in shown]),
        '',
        *figure,
    ]
    if faded:
        lines += [
            '### Flagged before, not now', '',
            'Niches flagged in an earlier year but not in the latest, most recently flagged '
            'first: fields that peaked, or whose growth slowed.', '',
            _markdown_table(
                ['niche', 'terms', 'first flagged', 'last flagged', 'status now', 'shape', span],
                [[niche['cluster'], terms(niche), niche['first_flagged'], niche['last_flagged'],
                  niche['status'], _shape_text(niche), niche['share_sparkline']]
                 for niche in faded[:top]]),
            '',
        ]
    return lines


def _report(summary: Mapping[str, Any], areas: Sequence[Mapping[str, Any]],
            topics: Sequence[Mapping[str, Any]], niches: Sequence[Mapping[str, Any]],
            terms: Sequence[Mapping[str, Any]], watch: Sequence[Mapping[str, Any]],
            backtest: Sequence[Mapping[str, Any]], coverage: Sequence[Mapping[str, Any]],
            timeline: Sequence[Mapping[str, Any]], series_start: int,
            settings: EmergenceSettings, top: int, plots: Mapping[str, str]) -> str:
    """Render the Markdown report: field map, niches, timeline, terms, backtest, coverage."""
    end_year = summary['end_year']
    window = f'{end_year - settings.growth_window + 1}-{end_year}'
    span = f'share {series_start}-{end_year}'

    def figure(name: str, caption: str) -> list[str]:
        """Embed a drawn figure, or nothing when it was not drawn."""
        return [f'![{caption}]({plots[name]})', ''] if name in plots else []

    lines = [
        '# Emerging fields', '',
        f'- Corpus: `{summary["db_path"]}`; {summary["papers_used"]} papers used of '
        f'{summary["papers_read"]} read (dropped: {summary["dropped"]})',
        f'- Embedding model: `{summary["embedding_model"]}`; one Ward tree cut into '
        f'{len(areas)} areas, {len(topics)} topics and {len(niches)} niches',
        f'- Analysed up to {end_year}; growth fitted over {window}; recent = last '
        f'{settings.recent_window} years',
        '', '## Field map', '',
        f'Status compares each cluster\'s share of the corpus over {window}: emerging = growth z '
        '>= 2 and recency lift >= 1.5; growing = growth z >= 2; declining = growth z <= -2; '
        'stable otherwise. "Recent" is the cluster\'s share of all recent papers. Flagged '
        'niches are listed inside their topic. Below each table, every topic shows the three '
        'papers nearest its centre; an area is named by its largest part, so read its topics.',
        '',
        f'Shape says whether share growth sped up or slowed over {window}, from a curved fit: '
        'accelerating, growth slowing, peaked (rising, then falling, with the turning year), '
        'rebounding, decline easing, decline steepening, or steady. The share column draws '
        f'each year\'s share of the corpus from {series_start}, scaled to the cluster\'s own '
        f'peak; `{SERIES_FILENAME}` holds the numbers.',
        '',
        'Under each topic, *rising* and *fading* name what is changing inside it: terms whose '
        f'share of the topic\'s papers grew or shrank over {window} (growth z >= 2) and that are '
        f'at least {TOPIC_TERM_LIFT:g} times as common in the topic\'s recent (earlier) papers as '
        'in the corpus\'s. Only field vocabulary counts, terms whose papers gather in a few '
        'niches, so wording that drifts everywhere ("crucial", "insights") is left out. '
        f'`{TOPIC_TERMS_FILENAME}` has up to ten of each.',
        '',
        *figure('areas', 'Share of the corpus by area'),
        *figure('area_growth', 'Area size and share growth'),
        *figure('topic_growth', 'Topic size and share growth'),
    ]
    flagged: dict[int, list[str]] = {}
    for niche in niches:
        if niche['flag']:
            label = '; '.join(niche['label_terms'].split('; ')[:3])
            flagged.setdefault(niche['topic'], []).append(f'{niche["cluster"]}: {label}')
    for area in areas:
        area_topics = [topic for topic in topics if topic['area'] == area['cluster']]
        lines += [
            f'### Area {area["cluster"]}: {area["label_terms"]}', '',
            f'{area["dated_papers"]} papers, {area["recent_share"]:.0%} of recent papers, '
            f'share x{_format(area["share_growth_per_year"])}/yr, {area["status"]}, '
            f'{_shape_text(area)}; {span}: {area["share_sparkline"]}', '',
            _markdown_table(
                ['topic', 'papers', 'recent', 'share x/yr', 'growth z', 'status', 'shape', span,
                 'terms', 'flagged niches'],
                [[topic['cluster'], topic['dated_papers'], f'{topic["recent_share"]:.1%}',
                  topic['share_growth_per_year'], topic['growth_z'], topic['status'],
                  _shape_text(topic), topic['share_sparkline'], topic['label_terms'],
                  ' / '.join(flagged.get(topic['cluster'], []))]
                 for topic in area_topics]),
            '',
        ]
        for topic in area_topics:
            lines.append(f'- Topic {topic["cluster"]}: '
                         f'{"; ".join(topic["central_titles"].split(" || "))}')
            change = [f'{direction}: {topic[f"{direction}_terms"].replace("; ", ", ")}'
                      for direction in ('rising', 'fading') if topic[f'{direction}_terms']]
            if change:
                lines.append(f'  - {"; ".join(change)}')
        lines.append('')
    has_impact = any(math.isfinite(niche['impact']) for niche in niches)
    impact_note = ('Novelty and impact are shown but not scored.' if has_impact else
                   'Novelty is shown but not scored. Impact needs citation counts, which '
                   '`pmt enrich` adds.')
    lines += [
        '## Emerging niches', '',
        'Ranked by the mean percentile of growth z and recency lift among niches with '
        f'at least {settings.min_papers} dated and {settings.min_recent} recent papers; '
        'a niche with most papers in one earlier year is a one-off spike and is not ranked. '
        f'{impact_note}', '',
        _markdown_table(
            ['#', 'niche', 'topic', 'papers', 'recent', 'p10 year', 'share x/yr', 'growth z',
             'recency lift', 'novelty'] + (['impact'] if has_impact else [])
            + ['flag', 'first flagged', 'shape', f'counts {window}', span, 'terms'],
            [[rank + 1, niche['cluster'], niche['topic'], niche['dated_papers'],
              niche['recent_papers'], niche['first_year_p10'], niche['share_growth_per_year'],
              niche['growth_z'], niche['recency_lift'], niche['pioneer_novelty']]
             + ([niche['impact']] if has_impact else [])
             + ['yes' if niche['flag'] else '', niche.get('first_flagged', ''),
                _shape_text(niche), niche['counts_by_year'], niche['share_sparkline'],
                niche['label_terms']]
             for rank, niche in enumerate(niches[:top])]),
        '',
        *figure('niches', 'Share of the top niches'),
        *figure('niche_growth', 'Niche size and share growth'),
        '### Details', '',
    ]
    for niche in niches[:min(top, 10)]:
        lines += [f'**Niche {niche["cluster"]}**: {niche["label_terms"]}', '',
                  f'- Topic {niche["topic"]}: {niche["topic_terms"]}',
                  f'- Central: {niche["central_titles"]}',
                  f'- Newest: {niche["newest_titles"]}', '']
    lines += _timeline_section(niches, timeline, top, span, figure('timeline', 'Emergence timeline'))
    lines += [
        '## Emerging terms', '',
        'Ranked by growth z times concentration (one minus the normalised entropy of the '
        'term over niches), so field vocabulary outranks phrasing that is spreading '
        'through every abstract.', '',
        _markdown_table(
            ['term', 'docs', 'recent', 'first year', 'share x/yr', 'growth z', 'concentration',
             'recency lift', f'counts {window}'],
            [[term['term'], term['docs'], term['recent_docs'], term['first_year'],
              term['share_growth_per_year'], term['growth_z'], term['concentration'],
              term['recency_lift'], term['counts_by_year']] for term in terms[:30]]),
        '',
    ]
    if watch:
        shown = [str(year) for year in range(end_year - 11, end_year + 1)]
        lines += ['## Watched terms (documents per year)', '',
                  _markdown_table(['term', *shown],
                                  [[row['term'], *(row.get(year, '') for year in shown)]
                                   for row in watch]), '',
                  *figure('watch_terms', 'Watched terms')]
    if backtest:
        lines += [
            f'## Backtest ({settings.horizon}-year horizon)', '',
            'Spearman correlation of each signal at the cutoff with the log change in the '
            'niche\'s share over the following years. "top" and "rest" are mean log share '
            f'changes of the {settings.backtest_top} top-ranked niches and of the others.', '',
            _markdown_table(
                ['cutoff', 'niches', 'score', 'growth z', 'recency', 'novelty', 'size', 'top',
                 'rest'],
                [[row['cutoff'], row['eligible_niches'], row['rho_score'], row['rho_growth_z'],
                  row['rho_recency_lift'], row['rho_novelty'], row['rho_size'],
                  row['top_mean_future_log_ratio'], row['rest_mean_future_log_ratio']]
                 for row in backtest]),
            '',
        ]
    lines += ['## Coverage by year', '',
              _markdown_table(['year', 'papers used'],
                              [[row['year'], row['papers']] for row in coverage]), '']
    return '\n'.join(lines)


# --------------------------------------------------------------------------- plots


def _trend_curve(counts: NDArray[Any], totals: NDArray[Any], years: NDArray[Any],
                 shape: str) -> NDArray[np.float64] | None:
    """Return the fitted yearly share: the curved fit when a shape was named, else the straight one."""
    mask = np.asarray(totals) > 0
    y = np.asarray(counts, dtype=float)[mask]
    if y.sum() < 3 or len(y) < 3:
        return None
    t = np.asarray(years, dtype=float)
    centre = float(t[mask].mean())
    degree = 1 if shape in ('', 'steady') else 2
    fit = _fit_counts(y, np.asarray(totals, dtype=float)[mask], t[mask] - centre, degree)
    if fit is None:
        return None
    return np.exp(sum(beta * (t - centre) ** power for power, beta in enumerate(fit[0])))


def _shorten(text: str, limit: int) -> str:
    """Cut text to ``limit`` characters, marking the cut with an ellipsis."""
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def _first_terms(row: Mapping[str, Any], count: int = 2) -> str:
    """Return a cluster's first label terms."""
    return ', '.join(row['label_terms'].split('; ')[:count])


def _panel_note(row: Mapping[str, Any]) -> str:
    """Summarise a cluster's status, shape, and yearly share change for a plot panel."""
    parts = [row['status'], _shape_text(row)]
    if math.isfinite(row['share_growth_per_year']):
        parts.append(f'×{row["share_growth_per_year"]:.2f}/yr')
    return ', '.join(part for part in parts if part)


def _figure(width: float, height: float, title: str, subtitle: str) -> Any:
    """Make a figure with a left-aligned title and wrapped subtitle in its top inch."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height))
    figure.text(0.3 / width, 1 - 0.15 / height, title, fontsize=12.5, fontweight='bold', va='top')
    figure.text(0.3 / width, 1 - 0.45 / height, textwrap.fill(subtitle, int(width * 15.5)),
                fontsize=8.5, color=_INK2, va='top', linespacing=1.45)
    return figure


def _share_panels(rows: Sequence[Mapping[str, Any]], table: NDArray[np.int64],
                  span: NDArray[np.int64], totals: NDArray[np.int64],
                  window: NDArray[np.int64], title: str, subtitle: str,
                  name: Callable[[Mapping[str, Any]], str],
                  note: Callable[[Mapping[str, Any]], str]) -> tuple[Any, list[Any]]:
    """Draw one panel per cluster: its yearly share, the growth window, and the fitted trend."""
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    columns = min(4, len(rows))
    lines = math.ceil(len(rows) / columns)
    panel, header = 2.25, 1.05
    width, height = 3.2 * columns + 0.3, header + lines * panel + 0.1
    figure = _figure(width, height, title, subtitle)
    axes = figure.subplots(lines, columns, squeeze=False).ravel()
    figure.subplots_adjust(left=0.55 / width, right=1 - 0.2 / width,
                           top=1 - (header + 0.45) / height, bottom=0.35 / height,
                           hspace=0.8 / (panel - 0.8), wspace=0.6 / 2.6)
    for axis in axes[len(rows):]:
        axis.set_visible(False)
    inside = (span >= window[0]) & (span <= window[-1])
    for axis, row in zip(axes, rows):
        counts = table[row['cluster']]
        shares = counts / np.maximum(totals, 1)
        fit = _trend_curve(counts[inside], totals[inside], span[inside], row['shape'])
        axis.axvspan(window[0] - 0.5, window[-1] + 0.5, color=_BAND, linewidth=0, zorder=0)
        axis.plot(span, shares, color=_RISE, linewidth=1.8, zorder=3)
        peak = float(shares.max())
        if fit is not None:
            axis.plot(span[inside], fit, color=_INK, linewidth=1.0, zorder=4)
            peak = max(peak, float(fit.max()))
        axis.set_xlim(span[0] - 0.5, span[-1] + 0.5)
        axis.set_ylim(0, peak * 1.15 if peak > 0 else 1.0)
        axis.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(3))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_title(_shorten(name(row), 38), pad=14)
        axis.text(0, 1.03, _shorten(note(row), 50), transform=axis.transAxes, fontsize=7.5,
                  color=_INK2, va='bottom')
    return figure, list(axes[:len(rows)])


def _plot_areas(areas: Sequence[Mapping[str, Any]], table: NDArray[np.int64],
                span: NDArray[np.int64], totals: NDArray[np.int64],
                window: NDArray[np.int64]) -> Any:
    """Draw each area's yearly share of the corpus."""
    figure, _ = _share_panels(
        areas, table, span, totals, window, 'Share of the corpus by area',
        f'Each year\'s share of papers in the area. Shaded: the growth window {window[0]}-'
        f'{window[-1]}, with the fitted trend, curved when a shape was found. Each panel has its '
        'own scale; areas are ordered by recent papers, as in the field map.',
        lambda row: f'Area {row["cluster"]}: {_first_terms(row)}', _panel_note)
    return figure


def _plot_niches(niches: Sequence[Mapping[str, Any]], table: NDArray[np.int64],
                 span: NDArray[np.int64], totals: NDArray[np.int64],
                 window: NDArray[np.int64]) -> Any | None:
    """Draw the top niches' yearly shares with the years they were first seen and flagged."""
    if not niches:
        return None
    figure, axes = _share_panels(
        niches, table, span, totals, window, f'Share of the top {len(niches)} niches',
        'Each year\'s share of papers in the highest-ranked niches. Shaded: the growth window, '
        'with the fitted trend. Vertical lines mark the years from the emergence timeline: '
        'seen inside a flagged niche, and flagged on the niche\'s own papers. Each panel has its '
        'own scale.',
        lambda row: f'Niche {row["cluster"]}: {_first_terms(row)}',
        lambda row: f'{row["dated_papers"]} papers, {_panel_note(row)}')
    for axis, niche in zip(axes, niches):
        events: dict[int, list[str]] = {}
        for key, label in (('first_seen', 'seen'), ('first_flagged', 'flagged')):
            if niche.get(key, '') != '':
                events.setdefault(int(niche[key]), []).append(label)
        # Labels of two different years are staggered so that neighbouring years stay legible.
        for height, (year, labels) in zip((0.97, 0.85), sorted(events.items())):
            right = year > span[-1] - 4
            axis.axvline(year, color=_INK2, linewidth=0.8, zorder=2)
            text = ' & '.join(labels)
            axis.text(year, height, f'{text} ' if right else f' {text}', fontsize=6.5,
                      color=_INK2, transform=axis.get_xaxis_transform(), va='top',
                      ha='right' if right else 'left',
                      bbox={'boxstyle': 'square,pad=0.1', 'facecolor': _BAND, 'linewidth': 0})
    return figure


def _place_labels(axis: Any, labels: Sequence[tuple[float, float, str, bool, float, str]]) -> None:
    """Label points, moving a label off any label placed before it.

    Each label first goes just above its point, on its right or left. Where
    that box would overlap an earlier label, it tries below the point, then
    further above and below, and keeps the first free place, or the last tried.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes holding the points, with its scales and limits already set.
    labels : Sequence[tuple[float, float, str, bool, float, str]]
        ``(x, y, text, left, fontsize, colour)`` for each label, in priority order.
    """
    axis.get_xlim(), axis.get_ylim()  # settle autoscaled limits before measuring
    points = 72 / axis.figure.dpi
    placed: list[tuple[float, float, float, float]] = []
    for x, y, text, left, size, colour in labels:
        px, py = (value * points for value in axis.transData.transform((x, y)))
        # Rough text extent in points: DejaVu Sans averages about 0.6 em a character.
        width, height = 0.6 * size * len(text), 1.25 * size
        start = px - 5 - width if left else px + 5
        for rise in (3.0, -3.0 - height, 3.0 + height, -3.0 - 2 * height):
            box = (start, py + rise, start + width, py + rise + height)
            if not any(box[0] < other[2] and other[0] < box[2]
                       and box[1] < other[3] and other[1] < box[3] for other in placed):
                break
        placed.append(box)
        axis.annotate(text, (x, y), xytext=(-5 if left else 5, rise), textcoords='offset points',
                      ha='right' if left else 'left', va='bottom', fontsize=size, color=colour,
                      zorder=5)


def _growth_naming(rows: Sequence[Mapping[str, Any]], level: str,
                   named_count: int) -> tuple[set[int], str]:
    """Choose the clusters a growth plot names, and say how in its subtitle.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Plotted cluster rows of one level, in report order.
    level : str
        ``'area'``, ``'topic'``, or ``'niche'``.
    named_count : int
        Top-ranked niches to name.

    Returns
    -------
    tuple[set[int], str]
        Clusters to name, and the sentence of the subtitle describing them.
    """
    if level == 'area':
        return ({row['cluster'] for row in rows},
                "Numbers are the field map's areas, each named by its first term.")
    if level == 'niche':
        ranked = [row['cluster'] for row in rows if row.get('eligible')][:named_count]
        return (set(ranked),
                f'Ringed: flagged as emerging. The {len(ranked)} top-ranked niches are named, '
                'with their number in the niche table.')
    finite = [row for row in rows if math.isfinite(row['growth_z'])]
    named = ({row['cluster'] for row in sorted(finite, key=lambda row: -row['growth_z'])[:3]
              if row['growth_z'] >= 2}
             | {row['cluster'] for row in sorted(finite, key=lambda row: row['growth_z'])[:3]
                if row['growth_z'] <= -2})
    return named, ("Numbers are the field map's topics; the three fastest rising and falling "
                   'are named.')


def _plot_growth(clusters: Sequence[Mapping[str, Any]], window: NDArray[np.int64], level: str,
                 named_count: int = 10) -> Any | None:
    """Plot each cluster's yearly share change against its size, coloured by direction.

    Parameters
    ----------
    clusters : Sequence[Mapping[str, Any]]
        Cluster rows of one tree level, in report order.
    window : numpy.ndarray
        Years of the growth window.
    level : str
        ``'area'``, ``'topic'``, or ``'niche'``. Areas are all named. Topics name
        the three fastest rising and falling. Niches leave out the 95% intervals,
        which would hide hundreds of dots, ring the flagged niches, and name the
        ``named_count`` top-ranked.
    named_count : int, default=10
        Top-ranked niches to name.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or ``None`` when no cluster has a fitted trend.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator

    rows = [row for row in clusters
            if math.isfinite(row['share_growth_per_year']) and row['dated_papers']]
    if not rows:
        return None
    niche = level == 'niche'
    named, naming = _growth_naming(rows, level, named_count)
    interval = ' against its papers' if niche else ', with a 95% interval, against its papers'
    width, height = 9.0, 6.6
    figure = _figure(
        width, height, f'{level.capitalize()} size and share growth',
        f'Each dot is {"an" if level == "area" else "a"} {level}: the yearly change in its share '
        f'of the corpus over {window[0]}-{window[-1]}{interval}. Blue: share rising '
        f'(growth z ≥ 2); red: falling (z ≤ −2); grey: no significant change. {naming}')
    figure.subplots_adjust(left=0.8 / width, right=1 - 0.3 / width, top=1 - 1.2 / height,
                           bottom=0.65 / height)
    axis = figure.add_subplot()
    axis.grid(True, axis='both')
    axis.axvline(0, color=_AXIS, linewidth=1.0, zorder=1)
    xs: list[float] = []
    for row in rows:
        slope, z, papers = math.log(row['share_growth_per_year']), row['growth_z'], row['dated_papers']
        colour = _RISE if z >= 2 else _FALL if z <= -2 else _MUTED
        x = 100 * (row['share_growth_per_year'] - 1)
        if not niche and math.isfinite(z) and z != 0:
            se = abs(slope / z)
            axis.plot([100 * math.expm1(slope - 1.96 * se), 100 * math.expm1(slope + 1.96 * se)],
                      [papers, papers], color=colour, alpha=0.35, linewidth=1.2,
                      solid_capstyle='butt', zorder=2)
        ringed = niche and bool(row.get('flag'))
        axis.scatter([x], [papers], s=20 if niche else 42, color=colour,
                     edgecolors=_INK if ringed else _SURFACE,
                     linewidths=1.0 if ringed else 0.6 if niche else 1.2,
                     zorder=4 if ringed else 3)
        xs.append(x)
    pad = max(2.0, 0.12 * (max(xs) - min(xs)))
    axis.set_xlim(min(xs) - pad, max(xs) + pad)
    axis.set_yscale('log')
    # Named clusters are labelled first, on their outer side, away from the crowd near zero.
    _place_labels(axis, [
        (x, row['dated_papers'], f'{row["cluster"]} {_shorten(_first_terms(row, 1), 24)}',
         row['growth_z'] < 0, 7.5, _INK)
        for x, row in zip(xs, rows) if row['cluster'] in named
    ] + ([] if niche else [
        (x, row['dated_papers'], str(row['cluster']), False, 6.5, _INK2)
        for x, row in zip(xs, rows) if row['cluster'] not in named
    ]))
    axis.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    axis.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: ''))
    axis.set_xlabel('Share change per year (%)')
    axis.set_ylabel('Papers (log scale)')
    handles = [Line2D([], [], marker='o', linestyle='', markersize=7, color=colour, label=label)
               for colour, label in ((_RISE, 'share rising'), (_FALL, 'share falling'),
                                     (_MUTED, 'no significant change'))]
    if niche:
        handles.append(Line2D([], [], marker='o', linestyle='', markersize=7, color=_RISE,
                              markeredgecolor=_INK, label='flagged as emerging'))
    axis.legend(handles=handles, loc='best')
    return figure


def _plot_timeline(niches: Sequence[Mapping[str, Any]], timeline: Sequence[Mapping[str, Any]],
                   top: int) -> Any | None:
    """Draw the timeline table: yearly percentiles, flags, and first-seen years."""
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Rectangle

    shown = [niche for rank, niche in enumerate(niches) if rank < top or niche['flag']]
    if not timeline or not shown:
        return None
    years = sorted({int(entry['year']) for entry in timeline})
    entries = {(entry['niche'], entry['year']): entry for entry in timeline}
    grid = np.full((len(shown), len(years)), np.nan)
    flags: list[tuple[int, int]] = []
    for r, niche in enumerate(shown):
        for c, year in enumerate(years):
            entry = entries.get((niche['cluster'], year))
            if entry is not None and entry['eligible']:
                grid[r, c] = entry['percentile']
                if entry['flag']:
                    flags.append((c, r))
    width, height = 9.0, 1.75 + 0.26 * len(shown)
    figure = _figure(
        width, height, 'Emergence timeline',
        'Today\'s top niches and every niche flagged now, scored on the papers published up to '
        'each year: the percentile among the niches ranked that year (grey: too small to rank). '
        'Dot: flagged as emerging. Box: first seen, the first year a flagged niche of that time '
        'held most of its papers.')
    bottom, top_margin = 0.4 / height, 1.25 / height
    figure.subplots_adjust(left=3.0 / width, right=1 - 1.1 / width, top=1 - top_margin,
                           bottom=bottom)
    axis = figure.add_subplot()
    edges_x, edges_y = np.arange(len(years) + 1) - 0.5, np.arange(len(shown) + 1) - 0.5
    axis.pcolormesh(edges_x, edges_y, np.zeros_like(grid), cmap=ListedColormap([_BAND]),
                    edgecolors=_SURFACE, linewidth=1.5)
    mesh = axis.pcolormesh(edges_x, edges_y, np.ma.masked_invalid(grid),
                           cmap=LinearSegmentedColormap.from_list('percentile', _RAMP),
                           vmin=0, vmax=100, edgecolors=_SURFACE, linewidth=1.5)
    if flags:
        axis.scatter([c for c, _ in flags], [r for _, r in flags], s=16, color='white',
                     edgecolors=_INK, linewidths=0.7, zorder=3)
    for r, niche in enumerate(shown):
        if niche.get('first_seen', '') in years:
            axis.add_patch(Rectangle((years.index(niche['first_seen']) - 0.43, r - 0.41), 0.86,
                                     0.82, fill=False, edgecolor=_INK, linewidth=1.3, zorder=4))
    axis.set_xlim(-0.5, len(years) - 0.5)
    axis.set_ylim(len(shown) - 0.5, -0.5)
    axis.set_xticks(range(len(years)), [str(year) for year in years])
    axis.set_yticks(range(len(shown)),
                    [_shorten(f'{niche["cluster"]}  {_first_terms(niche)}', 44) for niche in shown])
    axis.tick_params(length=0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    scale = figure.add_axes((1 - 0.85 / width, bottom, 0.12 / width, 1 - top_margin - bottom))
    colorbar = figure.colorbar(mesh, cax=scale, ticks=[0, 50, 100])
    colorbar.outline.set_visible(False)
    colorbar.set_label('percentile', color=_INK2)
    return figure


def _plot_watch(watch: Sequence[Mapping[str, Any]], coverage: Sequence[Mapping[str, Any]],
                span: NDArray[np.int64]) -> Any | None:
    """Plot the share of each year's papers that use each watched term."""
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    rows = [row for row in watch if row['docs']][:len(_SERIES)]
    if not rows:
        return None
    totals = {int(row['year']): int(row['papers']) for row in coverage}
    labelled = len(rows) <= 4
    width, height = 9.0, 5.0
    figure = _figure(width, height, 'Watched terms',
                     'Share of each year\'s papers whose title or abstract uses the term.')
    figure.subplots_adjust(left=0.75 / width, right=1 - (1.5 if labelled else 0.3) / width,
                           top=1 - 1.25 / height, bottom=0.45 / height)
    axis = figure.add_subplot()
    for colour, row in zip(_SERIES, rows):
        shares = np.array([int(row.get(str(year)) or 0) / totals[int(year)]
                           if totals.get(int(year)) else np.nan for year in span])
        axis.plot(span, shares, color=colour, linewidth=2.0, label=row['term'])
        if labelled:
            axis.annotate(row['term'], (span[-1], shares[-1]), xytext=(6, 0),
                          textcoords='offset points', va='center', fontsize=8, color=_INK2)
    axis.set_xlim(span[0] - 0.5, span[-1] + 0.5)
    axis.set_ylim(bottom=0)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    if len(rows) > 1:
        axis.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=min(4, len(rows)))
    return figure


def _draw_plots(output: Path, span: NDArray[np.int64], totals: NDArray[np.int64],
                tables: Mapping[str, NDArray[np.int64]], areas: Sequence[Mapping[str, Any]],
                topics: Sequence[Mapping[str, Any]], niches: Sequence[Mapping[str, Any]],
                timeline: Sequence[Mapping[str, Any]], watch: Sequence[Mapping[str, Any]],
                coverage: Sequence[Mapping[str, Any]], end_year: int,
                settings: EmergenceSettings, top: int) -> dict[str, str]:
    """Draw the report's figures into ``plots/``.

    Parameters
    ----------
    output : pathlib.Path
        Output directory of the analysis.
    span : numpy.ndarray
        Years of the share series.
    totals : numpy.ndarray
        Corpus papers in each year of ``span``.
    tables : Mapping[str, numpy.ndarray]
        Papers of each cluster in each year of ``span``, by level.
    areas, topics, niches : Sequence[Mapping[str, Any]]
        Cluster rows, in report order.
    timeline : Sequence[Mapping[str, Any]]
        Timeline entries; empty when the timeline was skipped.
    watch : Sequence[Mapping[str, Any]]
        Yearly document counts of the watched terms.
    coverage : Sequence[Mapping[str, Any]]
        Corpus papers per year.
    end_year : int
        Analysis year.
    settings : EmergenceSettings
        Windows of the analysis.
    top : int
        Niches listed in the report.

    Returns
    -------
    dict[str, str]
        Path of each drawn figure relative to the output directory, by name.

    Raises
    ------
    RuntimeError
        If Matplotlib is unavailable.
    """
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError('Plotting the emergence report requires matplotlib. Install '
                           'PaperMinerToolkit dependencies, or skip the plots with '
                           '--no-plots.') from error
    window = np.arange(end_year - settings.growth_window + 1, end_year + 1)
    directory = output / PLOTS_DIRNAME
    directory.mkdir(exist_ok=True)
    drawn: dict[str, str] = {}
    with matplotlib.rc_context(_PLOT_STYLE):
        for name, figure in (
            ('areas', _plot_areas(areas, tables['area'], span, totals, window)),
            ('area_growth', _plot_growth(areas, window, 'area')),
            ('topic_growth', _plot_growth(topics, window, 'topic')),
            ('niches', _plot_niches(niches[:min(top, 12)], tables['niche'], span, totals, window)),
            ('niche_growth', _plot_growth(niches, window, 'niche', min(top, 10))),
            ('timeline', _plot_timeline(niches, timeline, top)),
            ('watch_terms', _plot_watch(watch, coverage, span)),
        ):
            path = directory / f'{name}.png'
            if figure is None:
                path.unlink(missing_ok=True)
                continue
            figure.savefig(path, dpi=150)
            drawn[name] = f'{PLOTS_DIRNAME}/{name}.png'
    return drawn


# --------------------------------------------------------------------------- workflow


def _prepare_output_directory(output_dir: str | PathLike[str], overwrite: bool) -> Path:
    """Create the output directory, refusing to mix runs unless asked."""
    path = Path(output_dir)
    known = {REPORT_FILENAME, SUMMARY_FILENAME, NICHES_FILENAME}
    if path.exists() and not overwrite and any(child.name in known for child in path.iterdir()):
        raise ValueError(f'Output directory already holds an analysis: {path}. '
                         'Pass overwrite=True to replace it.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze_emergence(
    db_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    settings: EmergenceSettings | None = None,
    end_year: int | None = None,
    require_abstract: bool = True,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: str | PathLike[str] | None = None,
    embedder: Embedder | None = None,
    batch_size: int = 32,
    stopwords_file: str | PathLike[str] | None = None,
    watch_terms: Sequence[str] = (),
    top: int = 20,
    overwrite: bool = False,
    plots: bool = True,
) -> dict[str, Any]:
    """Map a corpus into areas, topics, and niches and find what is emerging.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Path to the SQLite paper corpus.
    output_dir : str or os.PathLike[str]
        Directory for the report, tables, and default embedding cache.
    settings : EmergenceSettings or None, optional
        Windows, granularity, and thresholds; defaults to
        :class:`EmergenceSettings`.
    end_year : int or None, optional
        Last complete publication year. Defaults to the latest year in the
        corpus before the current calendar year.
    require_abstract : bool, default=True
        Whether to drop papers without an abstract instead of embedding the
        title alone.
    embedding_model : str, default='allenai/specter2_base'
        Hugging Face model used when ``embedder`` is not supplied.
    embedding_cache : str, os.PathLike[str], or None, optional
        ``.npz`` embedding cache; defaults to ``embeddings.npz`` in
        ``output_dir``. Share one cache between runs on related corpora.
    embedder : Callable or None, optional
        Function mapping texts to vectors, used instead of the default model.
    batch_size : int, default=32
        Texts per batch for the default embedder.
    stopwords_file : str, os.PathLike[str], or None, optional
        Extra words to drop from cluster labels and terms, one per line.
    watch_terms : Sequence[str], default=()
        Terms whose yearly document counts are reported.
    top : int, default=20
        Niches listed in the report.
    overwrite : bool, default=False
        Whether to replace a previous analysis in ``output_dir``.
    plots : bool, default=True
        Whether to draw the report's figures into ``plots/``.

    Returns
    -------
    dict[str, Any]
        Paths to the artifacts, corpus counts, the analysis year, topic
        status counts, flagged niches, and the pooled backtest correlation.

    Raises
    ------
    OSError
        If the corpus or output directory cannot be accessed.
    RuntimeError
        If the default embedder is needed and PyTorch is not installed, or
        plots are requested and Matplotlib is not.
    ValueError
        If the settings are invalid, the corpus is too small, or it spans
        too few years.
    """
    settings = settings or EmergenceSettings()
    settings.validate()
    started = time.perf_counter()
    output = _prepare_output_directory(output_dir, overwrite)
    documents, counts = load_emergence_documents(db_path, require_abstract=require_abstract)
    if len(documents) < MIN_DOCUMENTS:
        raise ValueError(f'Emergence analysis needs at least {MIN_DOCUMENTS} dated papers; '
                         f'the corpus provides {len(documents)}.')
    if len(documents) > MAX_DOCUMENTS:
        raise ValueError(f'{len(documents)} papers exceed the {MAX_DOCUMENTS} that one Ward tree '
                         'can hold in memory; sample the corpus evenly across years.')
    years = np.array([document['year'] for document in documents], dtype=np.int64)
    if end_year is None:
        end_year = int(min(years.max(), time.gmtime().tm_year - 1))
    complete_years = np.unique(years[years <= end_year])
    if len(complete_years) < settings.growth_window:
        raise ValueError(
            f'The corpus spans {len(complete_years)} publication years up to {end_year}; '
            f'emergence needs at least growth_window={settings.growth_window}. A corpus from one '
            'or two years cannot show change over time.'
        )

    cache = Path(embedding_cache) if embedding_cache else output / EMBEDDINGS_FILENAME
    X, embedded = embed_documents([_embedding_text(document) for document in documents], cache,
                                  model_name=embedding_model, embedder=embedder,
                                  batch_size=batch_size)
    novelty_z = _paper_novelty(X, years, settings.novelty_neighbours, settings.novelty_min_prior)
    cites = _normalised_citations(
        np.array([document['cited_by_count'] for document in documents], dtype=float), years)
    stopwords = set(ENGLISH_STOP_WORDS) | _GENERIC_STOPWORDS | set(load_domain_stopwords(stopwords_file))
    D, terms = _document_terms(
        [f'{document["title"]}. {document["abstract"]}'.lower() for document in documents],
        stopwords)

    tree = linkage(X, method='ward')
    niches = _cut(tree, max(10, round(len(documents) / settings.niche_size)))
    topics = _cut(tree, max(2 * settings.areas, round(len(documents) / settings.topic_size)))
    areas = _cut(tree, settings.areas)
    recent_total = max(int(np.count_nonzero((years <= end_year)
                                            & (years > end_year - settings.recent_window))), 1)
    levels: dict[str, list[dict[str, Any]]] = {}
    for name, labels, top_terms in (('niche', niches, 8), ('topic', topics, 6), ('area', areas, 5)):
        rows, centroids = _cluster_rows(labels, X, years, novelty_z, cites, end_year, settings)
        names = _cluster_labels(D, labels, terms, top_terms)
        for row in rows:
            members = np.where(labels == row['cluster'])[0]
            closest = members[np.argsort(-(X[members] @ centroids[row['cluster']]))[:3]]
            newest = members[np.argsort(-years[members], kind='stable')[:2]]
            row['recent_share'] = row['recent_papers'] / recent_total
            row['label_terms'] = '; '.join(names[row['cluster']])
            row['topic'] = int(np.bincount(topics[members]).argmax())
            row['area'] = int(np.bincount(areas[members]).argmax())
            row['central_titles'] = ' || '.join(
                f'{documents[i]["title"]} ({years[i]})' for i in closest)
            row['newest_titles'] = ' || '.join(
                f'{documents[i]["title"]} ({years[i]})' for i in newest)
        levels[name] = rows
    topic_names = {row['cluster']: row['label_terms'] for row in levels['topic']}
    for row in levels['niche']:
        row['topic_terms'] = topic_names[row['topic']]
    topic_term_rows = _topic_term_rows(D, terms, years, topics, niches, end_year, settings)
    for row in levels['topic']:
        for direction in ('rising', 'fading'):
            row[f'{direction}_terms'] = '; '.join(
                [item['term'] for item in topic_term_rows
                 if item['topic'] == row['cluster'] and item['direction'] == direction][:4])

    span = np.arange(max(int(years.min()),
                         end_year - max(SERIES_YEARS, settings.growth_window) + 1), end_year + 1)
    span_totals = np.array([np.count_nonzero(years == year) for year in span])
    series_rows: list[dict[str, Any]] = []
    tables: dict[str, NDArray[np.int64]] = {}
    for name, labels in (('area', areas), ('topic', topics), ('niche', niches)):
        table = tables[name] = _yearly_counts(labels, years, span)
        for row in levels[name]:
            yearly = table[row['cluster']]
            shares = yearly / np.maximum(span_totals, 1)
            row['share_sparkline'] = _sparkline(shares)
            series_rows += [{'level': name, 'cluster': row['cluster'], 'year': int(year),
                             'papers': int(count), 'corpus_papers': int(total),
                             'share': _format(float(share), 5)}
                            for year, count, total, share in zip(span, yearly, span_totals, shares)]

    snapshots: dict[int, _Snapshot | None] = {}

    def snapshot(cutoff: int) -> _Snapshot | None:
        """Re-cluster the corpus as it stood in ``cutoff``, once per cutoff."""
        if cutoff not in snapshots:
            snapshots[cutoff] = _cutoff_snapshot(X, years, novelty_z, cutoff, settings)
        return snapshots[cutoff]

    timeline_rows = (_timeline(X, niches, years, novelty_z, levels['niche'], end_year, settings,
                               snapshot)
                     if settings.timeline_years else [])
    niche_rows = sorted(levels['niche'],
                        key=lambda row: (not math.isfinite(row['score']), -np.nan_to_num(row['score'])))
    topic_rows = sorted(levels['topic'], key=lambda row: -row['recent_papers'])
    area_rows = sorted(levels['area'], key=lambda row: -row['recent_papers'])

    term_rows, watch_rows = _term_rows(D, terms, years, niches, end_year, watch_terms, settings)
    backtest_rows, pooled = (_backtest(X, years, end_year, settings, snapshot)
                             if settings.backtest_cutoffs else ([], []))
    coverage = [{'year': int(year), 'papers': int(count)}
                for year, count in zip(*np.unique(years, return_counts=True))]

    _write_csv(output / AREAS_FILENAME, area_rows)
    _write_csv(output / TOPICS_FILENAME, topic_rows)
    _write_csv(output / NICHES_FILENAME, niche_rows)
    _write_csv(output / PAPERS_FILENAME, [
        {'paper_id': document['paper_id'], 'doi': document['doi'], 'year': document['year'],
         'title': document['title'], 'niche': int(niches[i]), 'topic': int(topics[i]),
         'area': int(areas[i]), 'novelty_z': _format(float(novelty_z[i]), 3)}
        for i, document in enumerate(documents)
    ])
    _write_csv(output / TERMS_FILENAME, term_rows)
    _write_csv(output / COVERAGE_FILENAME, coverage)
    _write_csv(output / SERIES_FILENAME, series_rows)
    _write_csv(output / TOPIC_TERMS_FILENAME, topic_term_rows,
               ['topic', 'direction', 'term', 'topic_docs', 'recent_docs',
                'share_growth_per_year', 'growth_z', 'lift', 'counts_by_year'])
    if timeline_rows:
        _write_csv(output / TIMELINE_FILENAME, [
            {**entry, **{key: _format(float(entry[key]), 3)
                         for key in ('growth_z', 'recency_lift', 'percentile', 'overlap', 'share',
                                     'earlier_percentile')}}
            for entry in timeline_rows])
    if watch_rows:
        _write_csv(output / WATCH_FILENAME, watch_rows)
    if backtest_rows:
        _write_csv(output / BACKTEST_FILENAME, backtest_rows)
        _write_csv(output / BACKTEST_NICHES_FILENAME, pooled,
                   ['cutoff', 'cluster', 'dated_papers', 'recent_papers', 'growth_z',
                    'recency_lift', 'pioneer_novelty', 'score', 'future_log_ratio',
                    'counts_by_year'])
    pooled_row = next((row for row in backtest_rows if row['cutoff'] == 'pooled'), None)
    figures = (_draw_plots(output, span, span_totals, tables, area_rows, topic_rows, niche_rows,
                           timeline_rows, watch_rows, coverage, end_year, settings, top)
               if plots else {})
    summary: dict[str, Any] = {
        'created_at': utc_now(),
        'db_path': str(db_path),
        'output_dir': str(output),
        'embedding_model': embedding_model,
        'embedding_cache': str(cache),
        'papers_embedded_now': embedded,
        'papers_read': counts['papers_read'],
        'papers_used': len(documents),
        'dropped': {key: value for key, value in counts.items() if key != 'papers_read'},
        'require_abstract': require_abstract,
        'end_year': end_year,
        'settings': asdict(settings),
        'areas': len(area_rows),
        'topics': len(topic_rows),
        'niches': len(niche_rows),
        'topic_status': {status: sum(row['status'] == status for row in topic_rows)
                         for status in STATUSES},
        'flagged_niches': [
            {'niche': row['cluster'], 'topic': row['topic'], 'papers': row['dated_papers'],
             'growth_z': row['growth_z'], 'recency_lift': row['recency_lift'],
             'shape': row['shape'], 'first_flagged': row.get('first_flagged', ''),
             'first_seen': row.get('first_seen', ''), 'label_terms': row['label_terms']}
            for row in niche_rows if row['flag']
        ],
        'backtest_rho_score': pooled_row['rho_score'] if pooled_row else math.nan,
        'report': str(output / REPORT_FILENAME),
        'plots': [str(output / path) for path in figures.values()],
    }
    summary['seconds'] = round(time.perf_counter() - started, 1)
    (output / REPORT_FILENAME).write_text(
        _report(summary, area_rows, topic_rows, niche_rows, term_rows, watch_rows, backtest_rows,
                coverage, timeline_rows, int(span[0]), settings, top, figures),
        encoding='utf-8',
    )
    (output / SUMMARY_FILENAME).write_text(json.dumps(json_safe(summary), indent=2),
                                           encoding='utf-8')
    return summary


def json_safe(value: Any) -> Any:
    """Replace NaN and infinity with ``None`` so the summary is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
