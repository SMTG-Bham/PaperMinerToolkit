"""Store scraped material rows into the final materials CSV.

The storage step maps temporary scrape columns to recipe fields, optionally
converts units, appends accepted rows to the output file, and marks papers as
stored once their scrape results have been persisted.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Collection
from os import PathLike
from pathlib import Path

import pandas as pd

from paperminertoolkit.corpus.database import connect, paper_rows, upsert_paper
from paperminertoolkit.extraction.extract import convert_units
from paperminertoolkit.extraction.models import ModelConfig
from paperminertoolkit.extraction.recipes import canonical_match, field_columns, load_recipe


def _temporary_path(directory: Path, prefix: str, suffix: str) -> str:
    """Create and close a unique temporary file.

    Parameters
    ----------
    directory : pathlib.Path
        Directory in which to create the file.
    prefix : str
        Temporary filename prefix.
    suffix : str
        Temporary filename suffix.

    Returns
    -------
    str
        Path to the temporary file.

    Raises
    ------
    OSError
        If the temporary file cannot be created or its descriptor cannot be
        closed.
    """
    descriptor, path = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=suffix)
    os.close(descriptor)
    return path


def _write_csv_atomically(data: pd.DataFrame, output_path: Path) -> None:
    """Write a CSV before atomically replacing its destination.

    Parameters
    ----------
    data : pandas.DataFrame
        Tabular data to serialize.
    output_path : pathlib.Path
        Destination CSV path.

    Raises
    ------
    OSError
        If the temporary CSV cannot be written or atomically moved into place.
    """
    temp_path = _temporary_path(output_path.parent, f'.{output_path.name}.', '.tmp')
    try:
        data.to_csv(temp_path)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _stored_paper_ids(data: pd.DataFrame) -> set[str]:
    """Extract non-empty paper identifiers from stored rows.

    Parameters
    ----------
    data : pandas.DataFrame
        Material rows that must contain a ``Paper id`` column.

    Returns
    -------
    set[str]
        Unique non-empty paper identifiers.

    Raises
    ------
    ValueError
        If the required column is missing or contains no non-empty identifiers.
    """
    if 'Paper id' not in data.columns:
        raise ValueError('Scraped materials must contain a "Paper id" column.')
    paper_ids = {
        str(value).strip()
        for value in data['Paper id']
        if pd.notna(value) and str(value).strip()
    }
    if not paper_ids:
        raise ValueError('Scraped materials must contain at least one non-empty "Paper id" value.')
    return paper_ids


def _mark_stored_papers(
    db_path: str | PathLike[str],
    paper_ids: Collection[str],
) -> None:
    """Mark successfully scraped papers as stored.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Corpus database to update.
    paper_ids : Collection[str]
        Paper identifiers represented by the stored batch.

    Raises
    ------
    RuntimeError
        If the corpus schema is newer than this package supports.
    """
    with connect(db_path) as conn:
        for paper in paper_rows(conn):
            if paper['paper_id'] not in paper_ids:
                continue
            if not any(
                paper.get(column) == 'succeeded'
                for column in ['abstract_scrape_status', 'text_scrape_status', 'image_scrape_status']
            ):
                continue
            paper['store_status'] = 'stored'
            upsert_paper(conn, paper)


def store_results(db_path: str | PathLike[str] = 'papers.db',
                  in_filepath: str | PathLike[str] = 'temp_scraped_materials.csv',
                  out_filepath: str | PathLike[str] = 'materials.csv',
                  unit_conversion: bool = True,
                  recipe: str = 'sse',
                  assume_yes: bool = False,
                  model_config: ModelConfig | None = None,
                  ) -> None:
    """Convert and append temporary scrape results.

    Parameters
    ----------
    db_path : str or os.PathLike[str], optional
        Corpus database whose paper statuses should be updated.
    in_filepath : str or os.PathLike[str], optional
        Temporary scraped-materials CSV to consume.
    out_filepath : str or os.PathLike[str], optional
        Final materials CSV to create or append.
    unit_conversion : bool, optional
        Whether to convert recipe fields with configured units.
    recipe : str, optional
        Bundled recipe name or recipe JSON path.
    assume_yes : bool, optional
        Whether to accept conversions and skip unmatched columns without a prompt.
    model_config : ModelConfig or None, optional
        Model configuration forwarded to unit conversion.

    Raises
    ------
    ValueError
        If input and output paths match, the recipe is invalid, or stored paper
        identifiers are invalid.
    RuntimeError
        If an unmatched column is encountered in interactive mode or the
        corpus schema is newer than this package supports.
    FileNotFoundError
        If the bundled recipe resource or an output parent directory is
        missing.
    KeyError
        If no bundled or standalone recipe matches ``recipe``.
    OSError
        If an input, preview, or output file cannot be accessed.
    """
    input_path = Path(in_filepath)
    output_path = Path(out_filepath)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output materials paths must be different.')
    if not input_path.is_file():
        print(f'No scraped materials file found at {in_filepath}. Nothing new to store.')
        return
    in_file = pd.read_csv(input_path, index_col=0)
    if in_file.empty:
        print(f'Scraped materials file {in_filepath} is empty. Nothing new to store.')
        return
    recipe = load_recipe(recipe)
    if output_path.is_file():
        out_file = pd.read_csv(output_path, index_col=0)
        columns = list(out_file.keys())
    else:
        out_file = None
        columns = field_columns(recipe)

    out_data = pd.DataFrame()
    for series_name, series in in_file.items():
        match = canonical_match(series_name, columns, recipe)
        if match is None:
            if assume_yes:
                print(f'Skipping unmatched column in noninteractive mode: {series_name}')
                continue
            raise RuntimeError(f'{series_name} did not match with a field in the recipe or output file.')
        split_field = match.split(' [')
        if len(split_field) == 2 and unit_conversion:
            print(f'{series_name} column was matched with {match} and will be converted.')
            split_field[1] = split_field[1].replace(']', '')
            converted_series = convert_units(
                series,
                field=split_field[0],
                unit=split_field[1],
                model_config=model_config,
            )
        else:
            print(f'{series_name} column was matched with {match} and will not be converted.')
            converted_series = series
        out_data[match] = converted_series

    paper_ids = _stored_paper_ids(out_data)
    preview_path = _temporary_path(output_path.parent, '.paperminertoolkit-converted-', '.csv')
    try:
        out_data.to_csv(preview_path)
        if assume_yes:
            decision = 'yes'
        else:
            print(f'\nOutput data has been saved to {preview_path} temporarily. Are you happy with these conversions?')
            decision = input('Yes (Y) / No (N): ')
        if decision.lower() not in {'y', 'yes'}:
            return

        if out_file is not None:
            materials_df = pd.concat([out_file, out_data], ignore_index=True)
        else:
            materials_df = out_data.reset_index(drop=True)
        materials_df.drop_duplicates(ignore_index=True, inplace=True)
        _write_csv_atomically(materials_df, output_path)
        _mark_stored_papers(db_path, paper_ids)
        input_path.unlink()
    finally:
        if os.path.exists(preview_path):
            os.remove(preview_path)
