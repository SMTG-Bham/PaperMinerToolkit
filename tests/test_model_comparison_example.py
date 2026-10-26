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
    run_dir: Path, statuses: dict[str, str], records: list[dict[str, str]], elapsed: list[int],
) -> None:
    """Create a run directory with a corpus copy, scraped CSV and job history."""
    run_dir.mkdir(parents=True)
    con = sqlite3.connect(run_dir / 'papers.db')
    con.execute('CREATE TABLE papers (paper_id TEXT PRIMARY KEY, text_scrape_status TEXT, last_error TEXT)')
    con.executemany(
        'INSERT INTO papers VALUES (?, ?, ?)',
        [(paper, status, 'No downloaded text' if status == 'failed' else None) for paper, status in statuses.items()],
    )
    con.commit()
    con.close()
    if records:
        pd.DataFrame(records).to_csv(run_dir / 'scraped_band_gap_validation.csv')
    with (run_dir / 'run_history.jsonl').open('w', encoding='utf-8') as handle:
        for seconds in elapsed:
            handle.write(json.dumps({'mode': 'text', 'elapsed_s': seconds}) + '\n')


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    """Build two runs that share two succeeded papers and disagree on one of them."""
    (tmp_path / 'models.tsv').write_text(MODELS, encoding='utf-8')
    gap = json.dumps([{'value': '1.34 eV', 'method_or_source': 'UV-vis', 'gap_type': 'optical', 'conditions': 'None'}])
    other = json.dumps([{'value': '2.0 eV', 'method_or_source': 'HSE06', 'gap_type': 'indirect', 'conditions': 'None'}])
    _write_run(
        tmp_path / 'runs' / 'tiny',
        {'p1': 'succeeded', 'p2': 'succeeded', 'p3': 'failed'},
        [
            {'Material system': 'Cs2SnI6', 'Band gap': gap, 'All band gaps': gap,
             'Cited literature band gaps': '', 'Paper id': 'p1'},
            {'Material system': 'MAPbI3', 'Band gap': other, 'All band gaps': 'not a list',
             'Cited literature band gaps': '', 'Paper id': 'p2'},
        ],
        [1800, 1800],
    )
    _write_run(
        tmp_path / 'runs' / 'big',
        {'p1': 'succeeded', 'p2': 'succeeded', 'p3': 'succeeded'},
        [
            {'Material system': 'Cs2 Sn I6', 'Band gap': gap, 'All band gaps': gap,
             'Cited literature band gaps': other, 'Paper id': 'p1'},
            {'Material system': 'CsPbBr3', 'Band gap': gap, 'All band gaps': gap,
             'Cited literature band gaps': '', 'Paper id': 'p2'},
            {'Material system': 'ZnO', 'Band gap': '[]', 'All band gaps': '[]',
             'Cited literature band gaps': '', 'Paper id': 'p3'},
        ],
        [7200],
    )
    return tmp_path


def test_summary_reports_coverage_fill_rates_and_throughput(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Count outcomes from the corpus copy and decode list cells from the CSV."""
    code = summary.main(['--models', str(runs / 'models.tsv'), '--runs-dir', str(runs / 'runs')])
    assert code == 0
    table = pd.read_csv(runs / 'runs' / 'comparison_summary.csv').set_index('label')
    tiny, big = table.loc['tiny'], table.loc['big']
    assert (tiny['succeeded'], tiny['failed'], tiny['untouched']) == (2, 1, 0)
    assert tiny['jobs'] == 2 and tiny['wall_hours'] == 1.0 and tiny['papers_per_hour'] == 2.0
    assert tiny['records'] == 2 and tiny['invalid_list_cells'] == 1
    assert tiny['band_gap_fill'] == 1.0 and tiny['all_band_gaps_fill'] == 0.5
    assert tiny['top_error'] == '[1x] No downloaded text'
    assert big['records'] == 3 and big['band_gap_fill'] == pytest.approx(2 / 3, abs=1e-3)
    assert big['cited_literature_band_gaps_fill'] == pytest.approx(1 / 3, abs=1e-3)
    assert 'succeeded_ids' not in table.columns
    out = capsys.readouterr().out
    assert '## Pairwise agreement on shared papers' in out
    assert '| tiny | big | 2 |' in out


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
