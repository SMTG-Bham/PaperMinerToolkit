"""Unit tests for paperminertoolkit.extraction.store.

This module tests storing temporary scraped material rows into the final
materials CSV, including missing inputs, empty inputs, column matching, optional
unit conversion, append behavior, user confirmation, and paper status updates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import paperminertoolkit.corpus.database as corpus
import paperminertoolkit.extraction.store as store


def sample_recipe() -> dict[str, Any]:
    """Return a minimal recipe for store unit tests."""
    return {
        'record definition': {
            'subject': 'test materials',
            'singular': 'material',
            'plural': 'materials',
            'unit': 'a distinct test material',
            'identity fields': ['Name'],
        },
        'search fields': {
            'Name': {'aliases': []},
            'Conductivity': {'unit': 'S cm^-1', 'aliases': []},
        },
    }


def test_helpers_reject_empty_ids_and_skip_unscraped_papers(tmp_path: Path) -> None:
    """Require useful paper IDs and avoid marking papers without successful scrapes."""
    with pytest.raises(ValueError, match='non-empty'):
        store._stored_paper_ids(pd.DataFrame({'Paper id': ['', None]}))
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'p', 'title': 'paper'})
    store._mark_stored_papers(db_path, {'p'})
    with corpus.connect(db_path) as conn:
        assert corpus.paper_rows(conn)[0]['store_status'] == 'pending'


def write_papers_corpus(path: Path) -> None:
    """Write a minimal paper corpus for store unit tests."""
    with corpus.connect(path) as conn:
        for row in [
            {
                'paper_id': 'paper-1',
                'text_scrape_status': 'succeeded',
                'image_scrape_status': 'pending',
                'store_status': 'pending',
            },
            {
                'paper_id': 'paper-2',
                'abstract_scrape_status': 'succeeded',
                'text_scrape_status': 'pending',
                'image_scrape_status': 'pending',
                'store_status': 'pending',
            },
            {
                'paper_id': 'paper-3',
                'text_scrape_status': 'pending',
                'image_scrape_status': 'pending',
                'store_status': 'pending',
            },
        ]:
            corpus.upsert_paper(conn, row)


def read_papers_corpus(path: Path) -> pd.DataFrame:
    """Read paper rows from a test corpus."""
    with corpus.connect(path) as conn:
        return pd.DataFrame(corpus.paper_rows(conn))


def test_store_results_reports_missing_scraped_materials_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test storing results when the temporary scraped materials file is missing."""
    in_path = tmp_path / 'missing.csv'
    out_path = tmp_path / 'materials.csv'

    store.store_results(in_filepath=str(in_path), out_filepath=str(out_path))

    output = capsys.readouterr().out
    assert f'No scraped materials file found at {in_path}' in output
    assert not out_path.exists()


def test_store_results_reports_empty_scraped_materials_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test storing results when the temporary scraped materials file is empty."""
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    pd.DataFrame(columns=['Name']).to_csv(in_path)

    store.store_results(in_filepath=str(in_path), out_filepath=str(out_path))

    output = capsys.readouterr().out
    assert f'Scraped materials file {in_path} is empty' in output
    assert not out_path.exists()


def test_store_results_converts_units_skips_unmatched_columns_and_updates_papers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test storing converted results in noninteractive mode."""
    monkeypatch.chdir(tmp_path)
    papers_path = tmp_path / 'papers.db'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_corpus(papers_path)
    pd.DataFrame({
        'Paper id': ['paper-1'],
        'Name': ['LLZO'],
        'Conductivity': ['1e-3'],
        'Unknown': ['skip me'],
    }).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())

    def fake_convert_units(
        series: pd.Series,
        field: str,
        unit: str,
        model_config: dict[str, str] | None = None,
    ) -> pd.Series:
        """Validate conversion arguments and return deterministic values."""
        assert field == 'Conductivity'
        assert unit == 'S cm^-1'
        assert model_config == {'provider': 'test'}
        return series.map(lambda value: f'{value} converted')

    monkeypatch.setattr(store, 'convert_units', fake_convert_units)

    store.store_results(
        db_path=str(papers_path),
        in_filepath=str(in_path),
        out_filepath=str(out_path),
        assume_yes=True,
        model_config={'provider': 'test'},
    )

    materials = pd.read_csv(out_path, index_col=0)
    papers = read_papers_corpus(papers_path)
    output = capsys.readouterr().out
    assert materials.columns.tolist() == ['Paper id', 'Name', 'Conductivity [S cm^-1]']
    assert materials.loc[0, 'Paper id'] == 'paper-1'
    assert materials.loc[0, 'Name'] == 'LLZO'
    assert materials.loc[0, 'Conductivity [S cm^-1]'] == '0.001 converted'
    assert 'Skipping unmatched column in noninteractive mode: Unknown' in output
    assert not in_path.exists()
    assert papers['store_status'].tolist() == ['stored', 'pending', 'pending']
    assert not list(tmp_path.glob('.paperminertoolkit-converted-*'))


def test_store_results_raises_for_unmatched_columns_in_interactive_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test storing results with an unmatched column in interactive mode."""
    papers_path = tmp_path / 'papers.db'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_corpus(papers_path)
    pd.DataFrame({'Unknown': ['value']}).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())

    with pytest.raises(RuntimeError, match='did not match'):
        store.store_results(
            db_path=str(papers_path),
            in_filepath=str(in_path),
            out_filepath=str(out_path),
            assume_yes=False,
        )

    assert in_path.exists()


def test_store_results_appends_existing_materials_without_unit_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test appending to an existing materials CSV without unit conversion."""
    monkeypatch.chdir(tmp_path)
    papers_path = tmp_path / 'papers.db'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_corpus(papers_path)
    pd.DataFrame({
        'Paper id': ['paper-1'],
        'Name': ['Existing'],
        'Conductivity [S cm^-1]': ['old'],
    }).to_csv(out_path)
    pd.DataFrame({'Paper id': ['paper-2'], 'Name': ['New'], 'Conductivity': ['new']}).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())

    store.store_results(
        db_path=str(papers_path),
        in_filepath=str(in_path),
        out_filepath=str(out_path),
        unit_conversion=False,
        assume_yes=True,
    )

    materials = pd.read_csv(out_path, index_col=0)
    assert materials.index.tolist() == [0, 1]
    assert materials['Paper id'].tolist() == ['paper-1', 'paper-2']
    assert materials['Name'].tolist() == ['Existing', 'New']
    assert materials['Conductivity [S cm^-1]'].tolist() == ['old', 'new']


def test_store_results_keeps_files_when_user_rejects_conversions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test storing results when the user rejects the temporary converted data."""
    monkeypatch.chdir(tmp_path)
    papers_path = tmp_path / 'papers.db'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_corpus(papers_path)
    pd.DataFrame({'Paper id': ['paper-1'], 'Name': ['LLZO']}).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())
    monkeypatch.setattr('builtins.input', lambda _: 'no')

    store.store_results(
        db_path=str(papers_path),
        in_filepath=str(in_path),
        out_filepath=str(out_path),
        assume_yes=False,
    )

    assert not out_path.exists()
    assert in_path.exists()
    assert not list(tmp_path.glob('.paperminertoolkit-converted-*'))


def test_store_results_requires_paper_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject batches that cannot be tied to corpus records."""
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    pd.DataFrame({'Name': ['LLZO']}).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())

    with pytest.raises(ValueError, match='Paper id'):
        store.store_results(
            in_filepath=str(in_path),
            out_filepath=str(out_path),
            assume_yes=True,
        )

    assert in_path.exists()
    assert not out_path.exists()


def test_store_results_is_retry_safe_for_identical_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not append a duplicate when an identical stored batch is retried."""
    papers_path = tmp_path / 'papers.db'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_corpus(papers_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())
    batch = pd.DataFrame({'Paper id': ['paper-1'], 'Name': ['LLZO'], 'Conductivity': ['1e-3']})

    for _ in range(2):
        batch.to_csv(in_path)
        store.store_results(
            db_path=str(papers_path),
            in_filepath=str(in_path),
            out_filepath=str(out_path),
            unit_conversion=False,
            assume_yes=True,
        )

    materials = pd.read_csv(out_path, index_col=0)
    assert len(materials) == 1
    assert materials.loc[0, 'Paper id'] == 'paper-1'


def test_store_results_preserves_existing_output_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the existing output, input batch, and corpus state intact on replacement failure."""
    papers_path = tmp_path / 'papers.db'
    in_path = tmp_path / 'scraped.csv'
    out_path = tmp_path / 'materials.csv'
    write_papers_corpus(papers_path)
    existing = pd.DataFrame({'Paper id': ['paper-3'], 'Name': ['Existing'], 'Conductivity [S cm^-1]': ['old']})
    existing.to_csv(out_path)
    pd.DataFrame({'Paper id': ['paper-1'], 'Name': ['New'], 'Conductivity': ['new']}).to_csv(in_path)
    monkeypatch.setattr(store, 'load_recipe', lambda _: sample_recipe())
    monkeypatch.setattr(store.os, 'replace', lambda *_: (_ for _ in ()).throw(OSError('replace failed')))

    with pytest.raises(OSError, match='replace failed'):
        store.store_results(
            db_path=str(papers_path),
            in_filepath=str(in_path),
            out_filepath=str(out_path),
            unit_conversion=False,
            assume_yes=True,
        )

    assert pd.read_csv(out_path, index_col=0).equals(existing)
    assert in_path.exists()
    assert read_papers_corpus(papers_path)['store_status'].tolist() == ['pending', 'pending', 'pending']
    assert not list(tmp_path.glob('.materials.csv.*'))
    assert not list(tmp_path.glob('.paperminertoolkit-converted-*'))


def test_store_results_rejects_identical_input_and_output_paths(tmp_path: Path) -> None:
    """Prevent the successful-store cleanup from deleting the output file."""
    path = tmp_path / 'materials.csv'
    pd.DataFrame({'Paper id': ['paper-1'], 'Name': ['LLZO']}).to_csv(path)

    with pytest.raises(ValueError, match='must be different'):
        store.store_results(in_filepath=str(path), out_filepath=str(path), assume_yes=True)

    assert path.exists()
