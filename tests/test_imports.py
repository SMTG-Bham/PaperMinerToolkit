"""Unit tests for paperscraper.imports.

This module tests importing local PDF files into the papers CSV, including
folder validation, empty-folder handling, metadata mapping, and merging imported
PDFs with existing paper rows.
"""

import importlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

imports = importlib.import_module('paperscraper.imports')

DATA_DIR = Path(__file__).parent / 'data'
FIXTURE_PDF = DATA_DIR / 'disorder-driven_fast_na_transport_oxychlorides.pdf'
FIXTURE_DOI = '10.1002/aenm.70977'


def test_import_pdfs_rejects_missing_directory(tmp_path):
    """
    Test validation of the PDF import directory.

    This function performs the following steps:
    1. Builds a path to a directory that does not exist.
    2. Calls `import_pdfs` with that missing directory.
    3. Captures the expected exception.

    Asserts:
        - A missing PDF directory raises `NotADirectoryError`.
    """
    missing_dir = tmp_path / 'missing'

    with pytest.raises(NotADirectoryError):
        imports.import_pdfs(str(missing_dir))


def test_import_pdfs_rejects_directory_without_pdfs(tmp_path):
    """
    Test PDF import behavior for an empty directory.

    This function performs the following steps:
    1. Creates an empty temporary directory.
    2. Calls `import_pdfs` with that directory.
    3. Captures the expected exception.

    Asserts:
        - A directory without PDF files raises `RuntimeError`.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()

    with pytest.raises(RuntimeError, match='No PDF files found'):
        imports.import_pdfs(str(papers_dir))


def test_import_pdfs_writes_imported_rows_with_crossref_metadata(tmp_path, capsys):
    """
    Test importing a fixture PDF into a new papers CSV with Crossref enrichment.

    This function performs the following steps:
    1. Copies the fixture PDF from the test data folder into a temporary PDF directory.
    2. Calls `import_pdfs` with Crossref enrichment enabled.
    3. Reloads the written papers CSV.

    Asserts:
        - One row is written for the fixture PDF.
        - The DOI and Crossref-enriched metadata are preserved.
        - The imported PDF is marked as an external source with a downloaded PDF.
        - The import summary reports added rows, DOI count, and Crossref enrichment count.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    pdf_path = papers_dir / FIXTURE_PDF.name
    shutil.copy(FIXTURE_PDF, pdf_path)
    papers_path = tmp_path / 'external_papers.csv'

    imports.import_pdfs(str(papers_dir), papers_path=str(papers_path), use_crossref=True)

    imported = pd.read_csv(papers_path, index_col=0)
    output = capsys.readouterr().out
    row = imported.iloc[0]
    assert len(imported) == 1
    assert row['paper_id'] == f'external:{FIXTURE_PDF.stem}'
    assert row['doi'] == FIXTURE_DOI
    assert 'Disorder-Driven' in row['title']
    assert 'Fast Na' in row['title']
    assert row['metadata_status'] == 'enriched'
    assert row['sources'] == 'external'
    assert row['pdf_download_status'] == 'succeeded'
    assert row['pdf_path'] == str(pdf_path)
    assert '1 added' in output
    assert '1 DOI found' in output
    assert '1 enriched via Crossref' in output


def test_import_pdfs_merges_fixture_pdf_with_existing_paper_rows(tmp_path, capsys):
    """
    Test importing a fixture PDF that matches an existing paper row.

    This function performs the following steps:
    1. Writes an existing papers CSV containing a row with a matching DOI.
    2. Copies the fixture PDF from the test data folder into a temporary PDF directory.
    3. Calls `import_pdfs` with Crossref enrichment enabled and reloads the merged papers CSV.

    Asserts:
        - The matching PDF updates the existing row instead of adding a duplicate.
        - Existing metadata is preserved when already populated.
        - Empty pipeline fields are filled from the imported PDF row.
        - The import summary reports one matched existing row.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    pdf_path = papers_dir / FIXTURE_PDF.name
    shutil.copy(FIXTURE_PDF, pdf_path)
    papers_path = tmp_path / 'external_papers.csv'
    pd.DataFrame([{
        'paper_id': 'scopus:1',
        'doi': FIXTURE_DOI,
        'title': 'Existing Title',
        'sources': 'scopus',
        'metadata_status': 'retrieved',
        'pdf_download_status': 'pending',
        'pdf_path': '',
    }]).to_csv(papers_path)

    imports.import_pdfs(str(papers_dir), papers_path=str(papers_path), use_crossref=True)

    merged = pd.read_csv(papers_path, index_col=0)
    output = capsys.readouterr().out
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row['paper_id'] == 'scopus:1'
    assert row['title'] == 'Existing Title'
    assert isinstance(row['journal'], str)
    assert row['journal'] != ''
    assert row['sources'] == 'scopus;external'
    assert row['pdf_download_status'] == 'succeeded'
    assert row['pdf_path'] == str(pdf_path)
    assert '0 added' in output
    assert '1 matched existing rows' in output
