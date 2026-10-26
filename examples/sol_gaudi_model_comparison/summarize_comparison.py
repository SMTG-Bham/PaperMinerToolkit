"""Summarise band-gap extraction runs from several models into comparable tables.

Each run directory written by ``scrape_model.sbatch`` holds the model's copy of
the corpus, its scraped CSV and a ``run_history.jsonl`` of job records. This
script reports, per model, how many papers were scraped, how many records and
band-gap items came back, how often each recipe field was filled, how many list
cells were not valid JSON, and timing: server start-up, scrape hours, seconds
per paper and per text chunk. It then measures pairwise agreement between
models on the materials and principal band-gap values they extracted from the
same papers. It calls no model and modifies nothing under the run directories;
run it on a login node once the jobs have finished.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

IDENTITY_FIELD = 'Material system'
PRINCIPAL_FIELD = 'Band gap'
LIST_FIELDS = ('Band gap', 'All band gaps', 'Cited literature band gaps')
NUMBER = re.compile(r'[-+]?\d+(?:\.\d+)?')
MISSING = {'', 'none', 'null', 'nan', '[]'}


def load_models(path: Path) -> list[dict[str, str]]:
    """Read the lineup from a models.tsv file.

    Parameters
    ----------
    path : pathlib.Path
        Tab-separated file with ``label``, ``repo_id``, ``tensor_parallel`` and
        ``class`` columns; blank lines and ``#`` comments are skipped.

    Returns
    -------
    list[dict[str, str]]
        One mapping per model in file order.
    """
    models = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        label, repo, tensor_parallel, size_class = line.split()[:4]
        models.append({'label': label, 'model': repo, 'tensor_parallel': tensor_parallel, 'class': size_class})
    return models


def parse_items(cell: object) -> list[dict[str, Any]] | None:
    """Decode a list-valued recipe cell.

    Parameters
    ----------
    cell : object
        Raw CSV cell holding a JSON or Python-literal list of gap items.

    Returns
    -------
    list[dict[str, Any]] or None
        The decoded items, an empty list for a missing value, or ``None`` when
        the cell is populated but not a list of mappings.
    """
    text = str(cell or '').strip()
    if text.casefold() in MISSING:
        return []
    for decoder in (json.loads, ast.literal_eval):
        try:
            value = decoder(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
        return None
    return None


def gap_values(items: list[dict[str, Any]]) -> set[float]:
    """Extract the numeric band-gap values from decoded gap items.

    Parameters
    ----------
    items : list[dict[str, Any]]
        Items with a ``value`` key such as ``'1.34 eV'`` or ``'~2.1'``.

    Returns
    -------
    set[float]
        The first number in each value, rounded to two decimals.
    """
    values = set()
    for item in items:
        match = NUMBER.search(str(item.get('value', '')))
        if match:
            values.add(round(float(match.group()), 2))
    return values


def normalise_material(name: object) -> str:
    """Reduce a material name to a comparison key.

    Parameters
    ----------
    name : object
        Material system cell.

    Returns
    -------
    str
        Lower-cased alphanumeric characters only, so ``Cs2SnI6`` and
        ``Cs₂SnI₆``-style spacing or punctuation differences do not count as
        disagreements.
    """
    return re.sub(r'[^0-9a-z]+', '', str(name or '').casefold())


def paper_outcomes(db_path: Path, mode: str) -> dict[str, Any]:
    """Count scrape outcomes recorded in a run's corpus copy.

    Parameters
    ----------
    db_path : pathlib.Path
        The model's ``papers.db``.
    mode : str
        Scrape mode, which selects the status column.

    Returns
    -------
    dict[str, Any]
        Paper totals, succeeded, failed and untouched counts, the set of
        succeeded paper ids and the most common recorded error.
    """
    column = 'abstract_scrape_status' if mode == 'abstract' else 'text_scrape_status'
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        total = con.execute('SELECT COUNT(*) FROM papers').fetchone()[0]
        # The column name comes from the fixed mapping above, never from input.
        succeeded = {row[0] for row in con.execute(
            f"SELECT paper_id FROM papers WHERE COALESCE({column}, '') = 'succeeded'",
        )}
        failed = con.execute(f"SELECT COUNT(*) FROM papers WHERE COALESCE({column}, '') = 'failed'").fetchone()[0]
        error = con.execute(
            "SELECT substr(last_error, 1, 80), COUNT(*) FROM papers WHERE COALESCE(last_error, '') != ''"
            ' GROUP BY 1 ORDER BY 2 DESC LIMIT 1',
        ).fetchone()
    finally:
        con.close()
    return {
        'papers': total, 'succeeded': len(succeeded), 'failed': failed,
        'untouched': total - len(succeeded) - failed, 'succeeded_ids': succeeded,
        'top_error': f'[{error[1]}x] {error[0]}' if error else '',
    }


def _timestamp(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp as written by the toolkit or the job script.

    Parameters
    ----------
    value : object
        ``2026-10-26T13:40:10+00:00`` or ``2026-10-26T13:40:10Z``; anything
        else gives ``None``.

    Returns
    -------
    datetime.datetime or None
        A timezone-aware instant in UTC.
    """
    text = str(value or '').strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def paper_durations(db_path: Path, mode: str, history: list[dict[str, Any]]) -> tuple[list[float], int]:
    """Estimate how long each succeeded paper took, from the corpus timestamps.

    The scrape commits each paper as it finishes and stamps ``updated_at``, and
    papers are processed one after another, so the gap between consecutive
    stamps inside a job's scrape window is that paper's duration. The first
    paper of a window is measured from the moment vLLM became ready.

    Parameters
    ----------
    db_path : pathlib.Path
        The model's ``papers.db``.
    mode : str
        Scrape mode, which selects the status column.
    history : list[dict[str, Any]]
        Job records with ``scrape_started`` (or ``started``) and ``finished``.

    Returns
    -------
    tuple[list[float], int]
        Seconds per succeeded paper, and the total text chunks those papers
        used.
    """
    column = 'abstract_scrape_status' if mode == 'abstract' else 'text_scrape_status'
    chunk_column = 'num_abstract_chunks' if mode == 'abstract' else 'num_text_chunks'
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        columns = {row[1] for row in con.execute('PRAGMA table_info(papers)')}
        chunk_select = chunk_column if chunk_column in columns else 'NULL'
        # Column names come from the fixed mappings above, never from input.
        rows = con.execute(
            f"SELECT updated_at, COALESCE({column}, ''), {chunk_select} FROM papers"
            " WHERE updated_at IS NOT NULL ORDER BY updated_at",
        ).fetchall()
    finally:
        con.close()
    windows = []
    for entry in history:
        begin = _timestamp(entry.get('scrape_started') or entry.get('started'))
        end = _timestamp(entry.get('finished'))
        if begin and end:
            windows.append((begin, end))

    durations: list[float] = []
    chunks = 0
    for begin, end in windows:
        previous = begin
        for stamp, status, chunk_count in rows:
            moment = _timestamp(stamp)
            if moment is None or moment < begin or moment > end:
                continue
            if status == 'succeeded':
                durations.append((moment - previous).total_seconds())
                chunks += int(chunk_count or 0)
            previous = moment
    return durations, chunks


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile.

    Parameters
    ----------
    values : list[float]
        Sample.
    fraction : float
        Percentile as a fraction, such as ``0.9``.

    Returns
    -------
    float or None
        The value, or ``None`` for an empty sample.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered)) - 1))
    return ordered[index]


def load_records(csv_path: Path) -> pd.DataFrame:
    """Read a scraped CSV as strings.

    Parameters
    ----------
    csv_path : pathlib.Path
        ``scraped_<recipe>.csv`` written by ``pmt scrape``.

    Returns
    -------
    pandas.DataFrame
        The records, or an empty frame when the scrape wrote no rows.
    """
    if not csv_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(csv_path, index_col=0, dtype=str, keep_default_na=False)


def summarise_run(model: dict[str, str], run_dir: Path, recipe: str) -> dict[str, Any]:
    """Build one summary row for a model's run directory.

    Parameters
    ----------
    model : dict[str, str]
        Lineup entry with ``label``, ``model``, ``tensor_parallel`` and ``class``.
    run_dir : pathlib.Path
        ``runs/<label>``.
    recipe : str
        Recipe name, which names the scraped CSV.

    Returns
    -------
    dict[str, Any]
        Coverage, record, fill-rate, validity and timing figures, plus the
        succeeded paper ids and per-paper extractions used for agreement.
    """
    history_path = run_dir / 'run_history.jsonl'
    history = []
    if history_path.is_file():
        history = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    mode = history[-1]['mode'] if history else 'text'
    hours = sum(entry.get('elapsed_s', 0) for entry in history) / 3600
    # Jobs written before the scrape phase was recorded separately fall back
    # to the whole job time, which then includes model loading.
    scrape_hours = sum(entry.get('scrape_elapsed_s', entry.get('elapsed_s', 0)) for entry in history) / 3600
    startups = [entry['server_startup_s'] for entry in history if entry.get('server_startup_s') is not None]
    db_path = run_dir / 'papers.db'
    outcomes = paper_outcomes(db_path, mode) if db_path.is_file() else {
        'papers': 0, 'succeeded': 0, 'failed': 0, 'untouched': 0, 'succeeded_ids': set(), 'top_error': '',
    }
    records = load_records(run_dir / f'scraped_{recipe}.csv')
    durations, chunks = paper_durations(db_path, mode, history) if db_path.is_file() else ([], 0)

    row: dict[str, Any] = {
        'label': model['label'], 'class': model['class'], 'model': model['model'],
        'tensor_parallel': model['tensor_parallel'], 'jobs': len(history),
        'wall_hours': round(hours, 2), 'papers': outcomes['papers'],
        'succeeded': outcomes['succeeded'], 'failed': outcomes['failed'],
        'untouched': outcomes['untouched'], 'records': len(records),
        'papers_with_records': records['Paper id'].nunique() if 'Paper id' in records else 0,
        'invalid_list_cells': 0, 'top_error': outcomes['top_error'],
        'server_startup_s': round(statistics.mean(startups)) if startups else None,
        'scrape_hours': round(scrape_hours, 2),
        'papers_per_hour': round(outcomes['succeeded'] / scrape_hours, 1) if scrape_hours else None,
        'timed_papers': len(durations),
        'paper_s_median': round(statistics.median(durations)) if durations else None,
        'paper_s_mean': round(statistics.mean(durations)) if durations else None,
        'paper_s_p90': round(percentile(durations, 0.9)) if durations else None,
        'paper_s_max': round(max(durations)) if durations else None,
        'chunks_per_paper': round(chunks / len(durations), 2) if durations else None,
        's_per_chunk': round(sum(durations) / chunks, 1) if chunks else None,
    }
    row['records_per_paper'] = round(len(records) / outcomes['succeeded'], 2) if outcomes['succeeded'] else None

    per_paper: dict[str, dict[str, set[Any]]] = {}
    item_counts = {field: 0 for field in LIST_FIELDS}
    filled = {field: 0 for field in LIST_FIELDS}
    for _, record in records.iterrows():
        paper = per_paper.setdefault(record.get('Paper id', ''), {'materials': set(), 'gaps': set()})
        paper['materials'].add(normalise_material(record.get(IDENTITY_FIELD, '')))
        for field in LIST_FIELDS:
            items = parse_items(record.get(field, ''))
            if items is None:
                row['invalid_list_cells'] += 1
                continue
            if items:
                filled[field] += 1
                item_counts[field] += len(items)
            if field == PRINCIPAL_FIELD:
                paper['gaps'].update(gap_values(items))
    for field in LIST_FIELDS:
        key = field.casefold().replace(' ', '_')
        row[f'{key}_fill'] = round(filled[field] / len(records), 3) if len(records) else None
        row[f'{key}_items_per_record'] = round(item_counts[field] / len(records), 2) if len(records) else None
    row['succeeded_ids'] = outcomes['succeeded_ids']
    row['per_paper'] = per_paper
    return row


def jaccard(left: set[Any], right: set[Any]) -> float:
    """Return the Jaccard similarity of two sets, with two empty sets counting as agreement.

    Parameters
    ----------
    left : set[Any]
        First set.
    right : set[Any]
        Second set.

    Returns
    -------
    float
        ``|left ∩ right| / |left ∪ right|``, or 1.0 when both are empty.
    """
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def agreement(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every pair of models on the papers both scraped successfully.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Summary rows from :func:`summarise_run`.

    Returns
    -------
    list[dict[str, Any]]
        One row per pair: shared paper count, mean material-name Jaccard and
        mean principal-gap-value Jaccard across those papers.
    """
    pairs = []
    for left, right in combinations(rows, 2):
        shared = sorted(left['succeeded_ids'] & right['succeeded_ids'])
        materials = []
        gaps = []
        for paper_id in shared:
            first = left['per_paper'].get(paper_id, {'materials': set(), 'gaps': set()})
            second = right['per_paper'].get(paper_id, {'materials': set(), 'gaps': set()})
            materials.append(jaccard(first['materials'], second['materials']))
            gaps.append(jaccard(first['gaps'], second['gaps']))
        pairs.append({
            'left': left['label'], 'right': right['label'], 'shared_papers': len(shared),
            'material_jaccard': round(sum(materials) / len(shared), 3) if shared else None,
            'principal_gap_jaccard': round(sum(gaps) / len(shared), 3) if shared else None,
        })
    return pairs


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as a GitHub-flavoured Markdown table.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Records to render.
    columns : list[str]
        Keys to show, in order.

    Returns
    -------
    str
        The table, with ``None`` shown as a dash.
    """
    def cell(value: object) -> str:
        """Format one cell."""
        return '-' if value is None else str(value)

    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    lines.extend('| ' + ' | '.join(cell(row.get(column)) for column in columns) + ' |' for row in rows)
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    """Summarise the runs and write the comparison tables.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments; ``None`` reads :data:`sys.argv`.

    Returns
    -------
    int
        Zero on success, one when no run directory was found.
    """
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--models', type=Path, default=here / 'models.tsv', help='Lineup file (default: models.tsv).')
    parser.add_argument('--runs-dir', type=Path, default=here / 'runs', help='Directory holding runs/<label>/.')
    parser.add_argument('--recipe', default='band_gap_validation', help='Recipe name used by the scrapes.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Where to write the CSVs (default: runs dir).')
    args = parser.parse_args(argv)

    rows = [
        summarise_run(model, args.runs_dir / model['label'], args.recipe)
        for model in load_models(args.models)
        if (args.runs_dir / model['label']).is_dir()
    ]
    if not rows:
        print(f'No run directories under {args.runs_dir}.', file=sys.stderr)
        return 1
    pairs = agreement(rows)

    output_dir = args.output_dir or args.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    public = [{key: value for key, value in row.items() if key not in {'succeeded_ids', 'per_paper'}} for row in rows]
    pd.DataFrame(public).to_csv(output_dir / 'comparison_summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(output_dir / 'comparison_agreement.csv', index=False)

    print('## Coverage\n')
    print(markdown_table(public, [
        'label', 'class', 'tensor_parallel', 'jobs', 'papers', 'succeeded', 'failed', 'untouched', 'top_error',
    ]))
    print('\n## Timing\n')
    print(markdown_table(public, [
        'label', 'wall_hours', 'server_startup_s', 'scrape_hours', 'papers_per_hour', 'timed_papers',
        'paper_s_median', 'paper_s_mean', 'paper_s_p90', 'paper_s_max', 'chunks_per_paper', 's_per_chunk',
    ]))
    print('\n## Records\n')
    print(markdown_table(public, [
        'label', 'records', 'papers_with_records', 'records_per_paper', 'band_gap_fill',
        'all_band_gaps_fill', 'all_band_gaps_items_per_record', 'cited_literature_band_gaps_fill',
        'invalid_list_cells',
    ]))
    print('\n## Pairwise agreement on shared papers\n')
    print(markdown_table(pairs, ['left', 'right', 'shared_papers', 'material_jaccard', 'principal_gap_jaccard']))
    print(f'\nTables written to {output_dir / "comparison_summary.csv"} and {output_dir / "comparison_agreement.csv"}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
