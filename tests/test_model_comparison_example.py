"""Exercise the Gaudi model-comparison summariser on synthetic run directories."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from examples.sol_gaudi_model_comparison import summarize_comparison as summary

MODELS = (
    '# label\trepo\ttp\tclass\n'
    'tiny\torg/tiny\t1\tsmall\n'
    'big\torg/big\t4\tlarge\n'
)


def _write_run(
    run_dir: Path,
    papers: list[tuple[str, str, str, int | None]],
    records: list[dict[str, str]],
    history: list[dict[str, object]],
) -> None:
    """Create a run directory with a corpus copy, scraped CSV and job history.

    Parameters
    ----------
    run_dir : pathlib.Path
        ``runs/<label>`` to create.
    papers : list[tuple[str, str, str, int or None]]
        ``(paper_id, text_scrape_status, updated_at, num_text_chunks)`` rows.
    records : list[dict[str, str]]
        Scraped rows; an empty list writes no CSV.
    history : list[dict[str, object]]
        Job records to append to ``run_history.jsonl``.
    """
    run_dir.mkdir(parents=True)
    con = sqlite3.connect(run_dir / 'papers.db')
    con.execute(
        'CREATE TABLE papers (paper_id TEXT PRIMARY KEY, text_scrape_status TEXT, last_error TEXT,'
        ' updated_at TEXT, num_text_chunks INTEGER)',
    )
    con.executemany(
        'INSERT INTO papers VALUES (?, ?, ?, ?, ?)',
        [
            (paper, status, 'No downloaded text' if status == 'failed' else None, stamp, chunks)
            for paper, status, stamp, chunks in papers
        ],
    )
    con.commit()
    con.close()
    if records:
        pd.DataFrame(records).to_csv(run_dir / 'scraped_band_gap_validation.csv')
    with (run_dir / 'run_history.jsonl').open('w', encoding='utf-8') as handle:
        for entry in history:
            handle.write(json.dumps({'mode': 'text', **entry}) + '\n')


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    """Build two runs that share two succeeded papers and disagree on one of them."""
    (tmp_path / 'models.tsv').write_text(MODELS, encoding='utf-8')
    gap = json.dumps([{'value': '1.34 eV', 'method_or_source': 'UV-vis', 'gap_type': 'optical', 'conditions': 'None'}])
    other = json.dumps([{'value': '2.0 eV', 'method_or_source': 'HSE06', 'gap_type': 'indirect', 'conditions': 'None'}])
    # tiny: two jobs that record the scrape phase separately. p1 finishes 600 s
    # into the first window, p3 fails quickly, p2 finishes 600 s into the second.
    _write_run(
        tmp_path / 'runs' / 'tiny',
        [
            ('p1', 'succeeded', '2026-10-26T10:20:00+00:00', 2),
            ('p3', 'failed', '2026-10-26T10:21:00+00:00', None),
            ('p2', 'succeeded', '2026-10-26T11:15:00+00:00', 1),
        ],
        [
            {'Material system': 'Cs2SnI6', 'Band gap': gap, 'All band gaps': gap,
             'Cited literature band gaps': '', 'Paper id': 'p1'},
            {'Material system': 'MAPbI3', 'Band gap': other, 'All band gaps': 'not a list',
             'Cited literature band gaps': '', 'Paper id': 'p2'},
        ],
        [
            {'started': '2026-10-26T10:00:00Z', 'scrape_started': '2026-10-26T10:10:00Z',
             'finished': '2026-10-26T10:40:00Z', 'elapsed_s': 2400, 'server_startup_s': 600,
             'scrape_elapsed_s': 1800},
            {'started': '2026-10-26T11:00:00Z', 'scrape_started': '2026-10-26T11:05:00Z',
             'finished': '2026-10-26T11:35:00Z', 'elapsed_s': 2100, 'server_startup_s': 300,
             'scrape_elapsed_s': 1800},
        ],
    )
    # big: one job recorded before the scrape phase was split out, so timing
    # falls back to the whole job and every paper is 1800 s apart.
    _write_run(
        tmp_path / 'runs' / 'big',
        [
            ('p1', 'succeeded', '2026-10-26T09:30:00+00:00', None),
            ('p2', 'succeeded', '2026-10-26T10:00:00+00:00', None),
            ('p3', 'succeeded', '2026-10-26T10:30:00+00:00', None),
        ],
        [
            {'Material system': 'Cs2 Sn I6', 'Band gap': gap, 'All band gaps': gap,
             'Cited literature band gaps': other, 'Paper id': 'p1'},
            {'Material system': 'CsPbBr3', 'Band gap': gap, 'All band gaps': gap,
             'Cited literature band gaps': '', 'Paper id': 'p2'},
            {'Material system': 'ZnO', 'Band gap': '[]', 'All band gaps': '[]',
             'Cited literature band gaps': '', 'Paper id': 'p3'},
        ],
        [{'started': '2026-10-26T09:00:00Z', 'finished': '2026-10-26T11:00:00Z', 'elapsed_s': 7200}],
    )
    return tmp_path


def test_summary_reports_coverage_fill_rates_and_throughput(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Count outcomes from the corpus copy and decode list cells from the CSV."""
    code = summary.main(['--models', str(runs / 'models.tsv'), '--runs-dir', str(runs / 'runs')])
    assert code == 0
    table = pd.read_csv(runs / 'runs' / 'comparison_summary.csv').set_index('label')
    tiny, big = table.loc['tiny'], table.loc['big']
    assert (tiny['succeeded'], tiny['failed'], tiny['untouched']) == (2, 1, 0)
    assert tiny['jobs'] == 2 and tiny['wall_hours'] == 1.25 and tiny['scrape_hours'] == 1.0
    assert tiny['papers_per_hour'] == 2.0
    assert tiny['records'] == 2 and tiny['invalid_list_cells'] == 1
    assert tiny['band_gap_fill'] == 1.0 and tiny['all_band_gaps_fill'] == 0.5
    assert tiny['top_error'] == '[1x] No downloaded text'
    assert big['records'] == 3 and big['band_gap_fill'] == pytest.approx(2 / 3, abs=1e-3)
    assert big['cited_literature_band_gaps_fill'] == pytest.approx(1 / 3, abs=1e-3)
    assert 'succeeded_ids' not in table.columns
    out = capsys.readouterr().out
    assert '## Timing' in out and '## Pairwise agreement on shared papers' in out
    assert '| tiny | big | 2 |' in out


def test_timing_splits_startup_from_scrape_and_times_each_paper(runs: Path) -> None:
    """Average start-up over jobs, time papers from the corpus stamps, and fall back for old records."""
    summary.main(['--models', str(runs / 'models.tsv'), '--runs-dir', str(runs / 'runs')])
    table = pd.read_csv(runs / 'runs' / 'comparison_summary.csv').set_index('label')
    tiny, big = table.loc['tiny'], table.loc['big']
    assert tiny['server_startup_s'] == 450
    assert tiny['timed_papers'] == 2
    assert (tiny['paper_s_median'], tiny['paper_s_mean'], tiny['paper_s_p90'], tiny['paper_s_max']) == (600,) * 4
    assert tiny['chunks_per_paper'] == 1.5 and tiny['s_per_chunk'] == 400.0
    assert pd.isna(big['server_startup_s']) and big['scrape_hours'] == 2.0
    assert big['papers_per_hour'] == 1.5 and big['timed_papers'] == 3
    assert big['paper_s_median'] == 1800 and big['paper_s_max'] == 1800
    assert big['chunks_per_paper'] == 0.0 and pd.isna(big['s_per_chunk'])


def test_paper_durations_ignore_stamps_outside_the_job_windows(tmp_path: Path) -> None:
    """Skip papers stamped before or after any job and papers that did not succeed."""
    run_dir = tmp_path / 'run'
    _write_run(
        run_dir,
        [
            ('old', 'succeeded', '2026-10-25T10:00:00+00:00', 1),
            ('in', 'succeeded', '2026-10-26T10:05:00+00:00', 1),
            ('fail', 'failed', '2026-10-26T10:06:00+00:00', None),
            ('late', 'succeeded', '2026-10-26T12:00:00+00:00', 1),
        ],
        [],
        [{'scrape_started': '2026-10-26T10:00:00Z', 'finished': '2026-10-26T10:30:00Z'}],
    )
    history = [json.loads(line) for line in (run_dir / 'run_history.jsonl').read_text(encoding='utf-8').splitlines()]
    assert summary.paper_durations(run_dir / 'papers.db', 'text', history) == ([300.0], 1)


@pytest.mark.parametrize(
    ('values', 'fraction', 'expected'),
    [([], 0.9, None), ([5.0], 0.9, 5.0), ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 0.9, 9.0)],
)
def test_percentile_uses_nearest_rank(values: list[float], fraction: float, expected: float | None) -> None:
    """Return None for no sample and the nearest-rank value otherwise."""
    assert summary.percentile(values, fraction) == expected


def test_agreement_ignores_spacing_and_scores_shared_papers_only(runs: Path) -> None:
    """Treat Cs2SnI6 and Cs2 Sn I6 as one material and score only papers both models finished."""
    summary.main(['--models', str(runs / 'models.tsv'), '--runs-dir', str(runs / 'runs')])
    pairs = pd.read_csv(runs / 'runs' / 'comparison_agreement.csv')
    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert pair['shared_papers'] == 2
    # p1 agrees fully on material and value; p2 disagrees on both.
    assert pair['material_jaccard'] == 0.5
    assert pair['principal_gap_jaccard'] == 0.5


@pytest.mark.parametrize(
    ('cell', 'expected'),
    [
        ('', []),
        ('None', []),
        ('[]', []),
        ('[{"value": "1.1 eV"}]', [{'value': '1.1 eV'}]),
        ("[{'value': '1.1 eV'}]", [{'value': '1.1 eV'}]),
        ('{"value": "1.1 eV"}', [{'value': '1.1 eV'}]),
        ('1.1 eV', None),
        ('["1.1 eV"]', None),
    ],
)
def test_parse_items_accepts_json_and_literals_and_rejects_scalars(cell: str, expected: object) -> None:
    """Decode JSON and Python literals, and flag populated cells that are not item lists."""
    assert summary.parse_items(cell) == expected


def test_gap_values_round_first_number_and_skip_unparseable() -> None:
    """Keep the first number of each value at two decimals and drop values without one."""
    items = [{'value': '~1.346 eV'}, {'value': '2 eV (direct)'}, {'value': 'None'}]
    assert summary.gap_values(items) == {1.35, 2.0}


def test_missing_runs_exit_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Fail clearly when no run directory exists for any lineup entry."""
    (tmp_path / 'models.tsv').write_text(MODELS, encoding='utf-8')
    assert summary.main(['--models', str(tmp_path / 'models.tsv'), '--runs-dir', str(tmp_path / 'runs')]) == 1
    assert 'No run directories' in capsys.readouterr().err
