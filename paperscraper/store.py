"""Store scraped material rows into the final materials CSV.

The storage step maps temporary scrape columns to recipe fields, optionally
converts units, appends accepted rows to the output file, and marks papers as
stored once their scrape results have been persisted.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd

from paperscraper.corpus import connect, paper_rows, upsert_paper
from paperscraper.extract import convert_units
from paperscraper.recipes import canonical_match, field_columns, load_recipe


def _temporary_path(directory: Path, prefix: str, suffix: str):
    """Create and close a unique temporary file in ``directory``."""
    descriptor, path = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=suffix)
    os.close(descriptor)
    return path


def _write_csv_atomically(data: pd.DataFrame, output_path: Path):
    """Replace ``output_path`` only after a complete CSV has been written."""
    temp_path = _temporary_path(output_path.parent, f'.{output_path.name}.', '.tmp')
    try:
        data.to_csv(temp_path)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _stored_paper_ids(data: pd.DataFrame):
    """Return the non-empty paper identifiers represented in stored rows."""
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


def _mark_stored_papers(db_path, paper_ids):
    """Mark only successfully scraped papers represented by the stored batch."""
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


def store_results(db_path='papers.db',
                  in_filepath='temp_scraped_materials.csv',
                  out_filepath='materials.csv',
                  unit_conversion=True,
                  recipe='sse',
                  assume_yes=False,
                  model_config=None,
                  ):
    """Convert and append temporary scrape results to the materials database."""
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
    preview_path = _temporary_path(output_path.parent, '.paperscraper-converted-', '.csv')
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
