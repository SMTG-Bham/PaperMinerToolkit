"""Merge and check the records of an existing scrape output.

``pmt scrape`` merges chunk records and, with ``--verify``, applies the
recipe's checks as it goes. This module applies the same steps to a CSV that
was scraped earlier, reading each paper's source text from the corpus, so the
effect of the steps can be measured without scraping again.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any

import pandas as pd

from paperminertoolkit.corpus.database import connect, paper_rows
from paperminertoolkit.extraction.merge import merge_chunk_records
from paperminertoolkit.extraction.recipes import load_recipe
from paperminertoolkit.extraction.scrape import _paper_asset_paths, _write_dropped, dropped_path, paper_source_text
from paperminertoolkit.extraction.verify import verify_records

_CHECKED_SOURCES = {'text': False, 'abstract': True}


@dataclass(frozen=True, slots=True)
class PostprocessSummary:
    """Summarize one post-processing run.

    Parameters
    ----------
    records_in : int
        Records read from the input CSV.
    records_out : int
        Records written to the output CSV.
    audit_rows : int
        Items removed or edited by the checks.
    unchecked_papers : tuple[str, ...]
        Papers whose source text could not be read, so their records were
        merged but not checked.
    output_path : str
        Output CSV path.
    dropped_path : str
        Audit CSV path.
    """

    records_in: int
    records_out: int
    audit_rows: int
    unchecked_papers: tuple[str, ...]
    output_path: str
    dropped_path: str


def _parse_cell(value: object) -> object:
    """Turn a stored list or object cell back into Python values."""
    if not isinstance(value, str) or not value.lstrip().startswith(('[', '{')):
        return value
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(value)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return value


def _records(frame: pd.DataFrame, fields: list[str]) -> list[dict[str, Any]]:
    """Read CSV rows as records, parsing structured recipe-field cells."""
    records = frame.to_dict(orient='records')
    for record in records:
        for field in fields:
            if field in record:
                record[field] = _parse_cell(record[field])
    return records


def default_output_path(in_path: str | PathLike[str]) -> str:
    """Return ``<input stem>.postprocessed.csv`` beside the input.

    Parameters
    ----------
    in_path : str or os.PathLike[str]
        Scrape output CSV.

    Returns
    -------
    str
        Default post-processed CSV path.
    """
    root, _ = os.path.splitext(os.fspath(in_path))
    return f'{root}.postprocessed.csv'


def postprocess_scrape(
    db_path: str | PathLike[str],
    recipe: str,
    in_path: str | PathLike[str],
    output_path: str | PathLike[str] | None = None,
    *,
    merge: bool = True,
    verify: bool = True,
    tables: bool = True,
    figure_text: bool = True,
) -> PostprocessSummary:
    """Merge and check an existing scrape output against the corpus.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Corpus database the CSV was scraped from.
    recipe : str
        Recipe name or path used for the scrape.
    in_path : str or os.PathLike[str]
        Scrape output CSV.
    output_path : str or os.PathLike[str] or None, optional
        Destination CSV. Defaults to ``<input stem>.postprocessed.csv``.
    merge : bool, optional
        Merge records with the same identity within each paper and source.
    verify : bool, optional
        Apply the recipe's ``"checks"`` to text and abstract records.
    tables : bool, optional
        Include the paper's tables in the source text used by the checks.
    figure_text : bool, optional
        Keep text printed inside PDF figures in the source text used by the checks.

    Returns
    -------
    PostprocessSummary
        Counts and paths for the run.
    """
    recipe_data = load_recipe(recipe)
    out_path = os.fspath(output_path or default_output_path(in_path))
    audit_path = dropped_path(out_path)
    if os.path.isfile(audit_path):
        os.remove(audit_path)
    frame = pd.read_csv(in_path, index_col=0)
    records = _records(frame, list(recipe_data['search fields']))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get('Paper id', '')), str(record.get('Source', '')))
        groups.setdefault(key, []).append(record)

    output: list[dict[str, Any]] = []
    audit_rows = 0
    unchecked: list[str] = []
    with connect(db_path) as conn, tempfile.TemporaryDirectory(prefix='paperminertoolkit-postprocess-') as temp_dir:
        papers: Mapping[str, Mapping[str, Any]] = {str(row['paper_id']): row for row in paper_rows(conn)}
        texts: dict[tuple[str, bool], str | None] = {}
        for (paper_id, source), group in groups.items():
            kept = merge_chunk_records(group, recipe_data) if merge else group
            if verify and source in _CHECKED_SOURCES and paper_id in papers:
                abstract = _CHECKED_SOURCES[source]
                if (paper_id, abstract) not in texts:
                    try:
                        paths = _paper_asset_paths(conn, papers[paper_id], temp_dir)
                        texts[paper_id, abstract] = paper_source_text(
                            conn, papers[paper_id], paths, abstract=abstract, tables=tables,
                            figure_text=figure_text,
                        )
                    except (FileNotFoundError, OSError, ValueError):
                        texts[paper_id, abstract] = None
                source_text = texts[paper_id, abstract]
                if source_text is None:
                    unchecked.append(paper_id)
                else:
                    kept, dropped = verify_records(kept, recipe_data, source_text, paper_id)
                    audit_rows += _write_dropped(dropped, out_path)
            elif verify and source in _CHECKED_SOURCES:
                unchecked.append(paper_id)
            output.extend(kept)

    pd.DataFrame(output, columns=frame.columns).to_csv(out_path)
    return PostprocessSummary(
        records_in=len(records),
        records_out=len(output),
        audit_rows=audit_rows,
        unchecked_papers=tuple(dict.fromkeys(unchecked)),
        output_path=out_path,
        dropped_path=audit_path,
    )
