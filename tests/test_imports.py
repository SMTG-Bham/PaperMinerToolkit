"""Unit tests for paperscraper.imports.

This module tests importing local PDF files into the paper corpus, including
folder validation, empty-folder handling, metadata mapping, and merging imported
PDFs with existing corpus rows.
"""

import shutil
from pathlib import Path

import pytest

import paperscraper.corpus as corpus
import paperscraper.imports as imports

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


def test_import_pdfs_writes_imported_rows_with_metadata(tmp_path, monkeypatch, capsys):
    """
    Test importing a fixture PDF into a new corpus database with metadata.

    This function performs the following steps:
    1. Copies the fixture PDF from the test data folder into a temporary PDF directory.
    2. Replaces PDF metadata extraction with deterministic metadata.
    3. Calls `import_pdfs` and reloads the written corpus rows and PDF asset.

    Asserts:
        - One row is written for the fixture PDF.
        - The extracted metadata is preserved.
        - The imported PDF is marked as an external source with a downloaded PDF in the corpus.
        - The import summary reports added rows, DOI count, and Crossref enrichment count.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    pdf_path = papers_dir / FIXTURE_PDF.name
    shutil.copy(FIXTURE_PDF, pdf_path)
    db_path = tmp_path / 'papers.db'
    monkeypatch.setattr(
        imports,
        'metadata_from_pdf',
        lambda path, use_crossref: ({
            'doi': FIXTURE_DOI,
            'title': 'Disorder-Driven Fast Na Transport',
            'journal': 'Advanced Energy Materials',
        }, 'enriched', ''),
    )

    imports.import_pdfs(str(papers_dir), db_path=str(db_path), use_crossref=True)

    with corpus.connect(db_path) as conn:
        imported = corpus.paper_rows(conn)
        asset = corpus.get_asset(conn, f'external:{FIXTURE_PDF.stem}', 'pdf')
    output = capsys.readouterr().out
    row = imported[0]
    assert len(imported) == 1
    assert row['paper_id'] == f'external:{FIXTURE_PDF.stem}'
    assert row['doi'] == FIXTURE_DOI
    assert 'Disorder-Driven' in row['title']
    assert 'Fast Na' in row['title']
    assert row['metadata_status'] == 'enriched'
    assert row['sources'] == 'external'
    assert row['pdf_download_status'] == 'succeeded'
    assert row['pdf_path'] == ''
    assert row['pdf_source'] == 'external'
    assert asset['content'] == pdf_path.read_bytes()
    assert '1 added' in output
    assert '1 DOI found' in output
    assert '1 enriched via Crossref' in output


def test_import_pdfs_merges_fixture_pdf_with_existing_corpus_rows(tmp_path, monkeypatch, capsys):
    """
    Test importing a fixture PDF that matches an existing corpus row.

    This function performs the following steps:
    1. Writes an existing corpus row with a matching DOI.
    2. Copies the fixture PDF from the test data folder into a temporary PDF directory.
    3. Replaces PDF metadata extraction with deterministic metadata.
    4. Calls `import_pdfs` and reloads the merged corpus rows and PDF asset.

    Asserts:
        - The matching PDF updates the existing row instead of adding a duplicate.
        - Existing metadata is preserved when already populated.
        - Empty corpus fields are filled from the imported PDF row.
        - The imported PDF asset is linked to the existing paper id.
        - The import summary reports one matched existing row.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    pdf_path = papers_dir / FIXTURE_PDF.name
    shutil.copy(FIXTURE_PDF, pdf_path)
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {
            'paper_id': 'scopus:1',
            'doi': FIXTURE_DOI,
            'title': 'Existing Title',
            'sources': 'scopus',
            'metadata_status': 'retrieved',
            'pdf_download_status': 'pending',
            'pdf_path': '',
        })
    monkeypatch.setattr(
        imports,
        'metadata_from_pdf',
        lambda path, use_crossref: ({
            'doi': FIXTURE_DOI,
            'title': 'Imported Title',
            'journal': 'Imported Journal',
        }, 'enriched', ''),
    )

    imports.import_pdfs(str(papers_dir), db_path=str(db_path), use_crossref=True)

    with corpus.connect(db_path) as conn:
        merged = corpus.paper_rows(conn)
        asset = corpus.get_asset(conn, 'scopus:1', 'pdf')
    output = capsys.readouterr().out
    assert len(merged) == 1
    row = merged[0]
    assert row['paper_id'] == 'scopus:1'
    assert row['title'] == 'Existing Title'
    assert row['journal'] == 'Imported Journal'
    assert row['sources'] == 'scopus;external'
    assert row['pdf_download_status'] == 'succeeded'
    assert row['pdf_path'] == ''
    assert row['pdf_source'] == 'external'
    assert asset['content'] == pdf_path.read_bytes()
    assert '0 added' in output
    assert '1 matched existing rows' in output
