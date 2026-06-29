"""Unit tests for paperscraper.store.

This module tests storing temporary scraped material rows into the final
materials CSV, including missing inputs, empty inputs, column matching, optional
unit conversion, append behavior, user confirmation, and paper status updates.
"""

import os

import pandas as pd
import pytest

import paperscraper.store as store


def sample_recipe():
    """Return a minimal recipe for store unit tests."""
    return {
        'material type': 'test material',
        'search fields': {
            'Name': {'aliases': []},
            'Conductivity': {'unit': 'S cm^-1', 'aliases': []},
        },
    }


def write_papers_csv(path):
    """Write a minimal papers CSV for store unit tests."""
    pd.DataFrame([
        {
            'paper_id': 'paper-1',
            'text_scrape_status': 'succeeded',
            'image_scrape_status': 'pending',
            'store_status': 'pending',
        },
        {
            'paper_id': 'paper-2',
            'text_scrape_status': 'pending',
            'image_scrape_status': 'pending',
            'store_status': 'pending',
        },
    ]).to_csv(path)


def test_store_results_reports_missing_scraped_materials_file(tmp_path, capsys):
    """
    Test storing results when the temporary scraped materials file is missing.

    This function performs the following steps:
    1. Builds paths for missing scraped materials and output materials files.
    2. Calls `store_results`.
    3. Captures the printed output.

    Asserts:
        - A helpful message is printed.
        - No output materials file is created.
    """
    in_path = tmp_path / 'missing.csv'
    out_path = tmp_path / 'materials.csv'

    store.store_results(in_filepath=str(in_path), out_filepath=str(out_path))

    output = capsys.readouterr().out
    assert f'No scraped materials file found at {in_path}' in output
    assert not out_path.exists()


def test_store_results_reports_empty_scraped_materials_file(tmp_path, capsys):
    """
    Test storing results when the temporary scraped materials file is empty.

    This function performs the following steps:
    1. Writes an empty scraped materials CSV.
    2. Calls `store_results`.
    3. Captures the printed output.

    Asserts:
        - A helpful message is printed.
        - No output materials file is created.
    """
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    pd.DataFrame(columns=['Name']).to_csv(in_path)

    store.store_results(in_filepath=str(in_path), out_filepath=str(out_path))

    output = capsys.readouterr().out
    assert f'Scraped materials file {in_path} is empty' in output
    assert not out_path.exists()


def test_store_results_converts_units_skips_unmatched_columns_and_updates_papers(tmp_path, monkeypatch, capsys):
    """
    Test storing converted results in noninteractive mode.

    This function performs the following steps:
    1. Writes scraped materials with a matched name column, a unit-bearing conductivity column, and an unknown column.
    2. Replaces recipe loading and unit conversion with deterministic local helpers.
    3. Calls `store_results` with `assume_yes=True`.
    4. Reloads the materials and papers CSV files.

    Asserts:
        - The unknown column is skipped in noninteractive mode.
        - Unit-bearing columns are converted when unit conversion is enabled.
        - The temporary scraped materials file is removed after storing.
        - Papers with succeeded scrape statuses are marked as stored.
    """
    monkeypatch.chdir(tmp_path)
    papers_path = tmp_path / 'papers.csv'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_csv(papers_path)
    pd.DataFrame({
        'Name': ['LLZO'],
        'Conductivity': ['1e-3'],
        'Unknown': ['skip me'],
    }).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())

    def fake_convert_units(series, field, unit, model_config=None):
        assert field == 'Conductivity'
        assert unit == 'S cm^-1'
        assert model_config == {'provider': 'test'}
        return series.map(lambda value: f'{value} converted')

    monkeypatch.setattr(store, 'convert_units', fake_convert_units)

    store.store_results(
        papers_path=str(papers_path),
        in_filepath=str(in_path),
        out_filepath=str(out_path),
        assume_yes=True,
        model_config={'provider': 'test'},
    )

    materials = pd.read_csv(out_path, index_col=0)
    papers = pd.read_csv(papers_path, index_col=0)
    output = capsys.readouterr().out
    assert materials.columns.tolist() == ['Name', 'Conductivity [S cm^-1]']
    assert materials.loc[0, 'Name'] == 'LLZO'
    assert materials.loc[0, 'Conductivity [S cm^-1]'] == '0.001 converted'
    assert 'Skipping unmatched column in noninteractive mode: Unknown' in output
    assert not in_path.exists()
    assert papers['store_status'].tolist() == ['stored', 'pending']
    assert not (tmp_path / 'temp_converted_materials.csv').exists()


def test_store_results_raises_for_unmatched_columns_in_interactive_mode(tmp_path, monkeypatch):
    """
    Test storing results with an unmatched column in interactive mode.

    This function performs the following steps:
    1. Writes scraped materials with an unmatched column.
    2. Replaces recipe loading with a minimal recipe.
    3. Calls `store_results` without `assume_yes`.

    Asserts:
        - An unmatched scraped materials column raises `RuntimeError`.
        - The original scraped materials file is left in place.
    """
    papers_path = tmp_path / 'papers.csv'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_csv(papers_path)
    pd.DataFrame({'Unknown': ['value']}).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())

    with pytest.raises(RuntimeError, match='did not match'):
        store.store_results(
            papers_path=str(papers_path),
            in_filepath=str(in_path),
            out_filepath=str(out_path),
            assume_yes=False,
        )

    assert in_path.exists()


def test_store_results_appends_existing_materials_without_unit_conversion(tmp_path, monkeypatch):
    """
    Test appending to an existing materials CSV without unit conversion.

    This function performs the following steps:
    1. Writes existing materials, new scraped materials, and a papers CSV.
    2. Replaces recipe loading with a minimal recipe.
    3. Calls `store_results` with unit conversion disabled.
    4. Reloads the output materials CSV.

    Asserts:
        - New rows are appended after existing material rows.
        - Unit-bearing columns are not converted when unit conversion is disabled.
        - The output index is reset after appending.
    """
    monkeypatch.chdir(tmp_path)
    papers_path = tmp_path / 'papers.csv'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_csv(papers_path)
    pd.DataFrame({'Name': ['Existing'], 'Conductivity [S cm^-1]': ['old']}).to_csv(out_path)
    pd.DataFrame({'Name': ['New'], 'Conductivity': ['new']}).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())

    store.store_results(
        papers_path=str(papers_path),
        in_filepath=str(in_path),
        out_filepath=str(out_path),
        unit_conversion=False,
        assume_yes=True,
    )

    materials = pd.read_csv(out_path, index_col=0)
    assert materials.index.tolist() == [0, 1]
    assert materials['Name'].tolist() == ['Existing', 'New']
    assert materials['Conductivity [S cm^-1]'].tolist() == ['old', 'new']


def test_store_results_keeps_files_when_user_rejects_conversions(tmp_path, monkeypatch):
    """
    Test storing results when the user rejects the temporary converted data.

    This function performs the following steps:
    1. Writes scraped materials and a papers CSV.
    2. Replaces recipe loading and interactive input.
    3. Calls `store_results` and rejects the conversion prompt.

    Asserts:
        - No materials CSV is written.
        - The scraped materials file remains available.
        - The temporary converted materials file is removed.
    """
    monkeypatch.chdir(tmp_path)
    papers_path = tmp_path / 'papers.csv'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_csv(papers_path)
    pd.DataFrame({'Name': ['LLZO']}).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())
    monkeypatch.setattr('builtins.input', lambda _: 'no')

    store.store_results(
        papers_path=str(papers_path),
        in_filepath=str(in_path),
        out_filepath=str(out_path),
        assume_yes=False,
    )

    assert not out_path.exists()
    assert in_path.exists()
    assert not (tmp_path / 'temp_converted_materials.csv').exists()
