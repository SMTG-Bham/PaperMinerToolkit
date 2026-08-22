"""Unit tests for paperscraper.imports.

This module tests importing local PDF files into the paper corpus, including
folder validation, empty-folder handling, metadata mapping, and merging imported
PDFs with existing corpus rows.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import paperscraper.corpus as corpus
import paperscraper.imports as imports

DATA_DIR = Path(__file__).parent / 'data'
FIXTURE_PDF = DATA_DIR / 'disorder-driven_fast_na_transport_oxychlorides.pdf'
FIXTURE_DOI = '10.1002/aenm.70977'


def test_import_pdfs_rejects_missing_directory(tmp_path: Path) -> None:
    """Test validation of the PDF import directory."""
    missing_dir = tmp_path / 'missing'

    with pytest.raises(NotADirectoryError):
        imports.import_pdfs(str(missing_dir))


def test_import_pdfs_rejects_directory_without_pdfs(tmp_path: Path) -> None:
    """Test PDF import behavior for an empty directory."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()

    with pytest.raises(RuntimeError, match='No PDF files found'):
        imports.import_pdfs(str(papers_dir))


def test_import_pdfs_writes_imported_rows_with_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test importing a fixture PDF into a new corpus database with metadata."""
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


def test_import_pdfs_merges_fixture_pdf_with_existing_corpus_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test importing a fixture PDF that matches an existing corpus row."""
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


def test_import_pdfs_writes_publisher_and_work_type_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store the Crossref fields the importer used to fetch and discard."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    shutil.copy(FIXTURE_PDF, papers_dir / FIXTURE_PDF.name)
    db_path = tmp_path / 'papers.db'
    message = {
        'DOI': FIXTURE_DOI,
        'publisher': 'Wiley',
        'type': 'journal-article',
        'volume': '14',
        'issue': '7',
        'page': '2470977',
        'language': 'en',
        'issn-type': [{'value': '1614-6832', 'type': 'print'}],
        'author': [{'given': 'Jane A.', 'family': 'Smith',
                    'ORCID': 'https://orcid.org/0000-0002-1825-0097'}],
        'reference': [{'key': 'ref1', 'DOI': '10.1234/cited'}],
    }
    monkeypatch.setattr(
        imports,
        'metadata_from_pdf',
        lambda path, use_crossref: ({
            'doi': FIXTURE_DOI,
            'title': 'Disorder-Driven Fast Na Transport',
            'publisher': 'Wiley',
            'work_type': 'journal-article',
            'crossref_message': message,
        }, 'enriched', ''),
    )

    imports.import_pdfs(str(papers_dir), db_path=str(db_path), use_crossref=True)

    with corpus.connect(db_path) as conn:
        row = corpus.paper_rows(conn)[0]
        authors = corpus.paper_authors(conn, row['paper_id'])
        references = corpus.paper_references(conn, row['paper_id'])

    assert row['publisher'] == 'Wiley'
    assert row['work_type'] == 'journal-article'
    assert row['volume'] == '14'
    assert row['issue'] == '7'
    assert row['pages'] == '2470977'
    assert row['issn'] == '1614-6832'
    assert row['language'] == 'en'
    assert row['enrichment_sources'] == 'crossref'
    assert authors[0]['orcid'] == '0000-0002-1825-0097'
    assert references[0]['referenced_doi'] == '10.1234/cited'


def test_import_pdfs_keeps_the_raw_message_out_of_the_paper_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never leak the raw Crossref payload into a corpus paper column."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    shutil.copy(FIXTURE_PDF, papers_dir / FIXTURE_PDF.name)
    db_path = tmp_path / 'papers.db'
    monkeypatch.setattr(
        imports,
        'metadata_from_pdf',
        lambda path, use_crossref: ({'doi': FIXTURE_DOI,
                                     'crossref_message': {'DOI': FIXTURE_DOI}}, 'enriched', ''),
    )

    imports.import_pdfs(str(papers_dir), db_path=str(db_path), use_crossref=True)

    with corpus.connect(db_path) as conn:
        row = corpus.paper_rows(conn)[0]

    assert 'crossref_message' not in row
