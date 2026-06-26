"""Unit tests for paperscraper.metadata.

This module tests DOI cleanup and extraction from text, embedded PDF metadata,
PDF page text fallback, Crossref date formatting, and Crossref metadata
normalization.
"""

import importlib
from pathlib import Path

metadata = importlib.import_module('paperscraper.metadata')

DATA_DIR = Path(__file__).parent / 'data'
FIXTURE_PDF = DATA_DIR / 'disorder-driven_fast_na_transport_oxychlorides.pdf'


def test_clean_doi_removes_trailing_punctuation_and_page_count_artifacts():
    """
    Test DOI cleanup for punctuation and PDF page-count artifacts.

    This function performs the following steps:
    1. Defines DOI strings with trailing punctuation and a glued page-count suffix.
    2. Cleans each DOI with `clean_doi`.
    3. Compares the cleaned values to the expected DOI.

    Asserts:
        - Trailing punctuation is removed.
        - Glued page-count suffixes such as `1of15` are removed.
    """
    assert metadata.clean_doi('10.1002/aenm.70977.') == '10.1002/aenm.70977'
    assert metadata.clean_doi('10.1002/aenm.709771of15') == '10.1002/aenm.70977'


def test_extract_doi_from_text_finds_first_doi():
    """
    Test DOI extraction from plain text.

    This function performs the following steps:
    1. Defines text containing a DOI.
    2. Extracts the DOI with `extract_doi_from_text`.
    3. Extracts a DOI from text with no DOI.

    Asserts:
        - The DOI in the text is returned.
        - Text without a DOI returns None.
    """
    assert metadata.extract_doi_from_text('See doi 10.1002/aenm.70977.') == '10.1002/aenm.70977'
    assert metadata.extract_doi_from_text('No DOI here.') is None


def test_extract_doi_from_pdf_metadata_reads_fixture_article_doi():
    """
    Test DOI extraction from embedded PDF metadata.

    This function performs the following steps:
    1. Opens the fixture PDF from the test data folder.
    2. Extracts the DOI from embedded metadata.
    3. Compares it to the expected article DOI.

    Asserts:
        - The article DOI stored in PDF metadata is returned.
    """
    assert metadata.extract_doi_from_pdf_metadata(str(FIXTURE_PDF)) == '10.1002/aenm.70977'


def test_extract_doi_from_pdf_prefers_metadata_before_text(monkeypatch):
    """
    Test DOI extraction preference for PDF metadata over page text.

    This function performs the following steps:
    1. Replaces PDF metadata extraction with a deterministic DOI.
    2. Replaces PDF text reading with a function that would fail if called.
    3. Calls `extract_doi_from_pdf`.

    Asserts:
        - The metadata DOI is returned.
        - Page text is not read when metadata contains a DOI.
    """
    monkeypatch.setattr(metadata, 'extract_doi_from_pdf_metadata', lambda _: '10.1234/metadata')

    def fail_read_pdf_text(_):
        raise AssertionError('PDF text should not be read when metadata has a DOI.')

    monkeypatch.setattr(metadata, 'read_pdf_text', fail_read_pdf_text)

    assert metadata.extract_doi_from_pdf('paper.pdf') == '10.1234/metadata'


def test_extract_doi_from_pdf_falls_back_to_page_text(monkeypatch):
    """
    Test DOI extraction fallback from PDF metadata to page text.

    This function performs the following steps:
    1. Replaces PDF metadata extraction with no DOI.
    2. Replaces PDF text reading with text containing a DOI.
    3. Calls `extract_doi_from_pdf`.

    Asserts:
        - The DOI from page text is returned when metadata has no DOI.
    """
    monkeypatch.setattr(metadata, 'extract_doi_from_pdf_metadata', lambda _: None)
    monkeypatch.setattr(metadata, 'read_pdf_text', lambda _: 'Text DOI 10.1234/text.')

    assert metadata.extract_doi_from_pdf('paper.pdf') == '10.1234/text'


def test_metadata_from_pdf_handles_missing_doi(monkeypatch):
    """
    Test metadata extraction when a PDF contains no DOI.

    This function performs the following steps:
    1. Replaces PDF DOI extraction with a no-DOI result.
    2. Calls `metadata_from_pdf` with Crossref enrichment enabled.
    3. Checks the returned metadata, status, and error message.

    Asserts:
        - No metadata fields are returned.
        - The import status remains `imported`.
        - The error message explains that no DOI was found.
    """
    monkeypatch.setattr(metadata, 'extract_doi_from_pdf', lambda _: None)

    paper_metadata, status, error = metadata.metadata_from_pdf('paper.pdf', use_crossref=True)

    assert paper_metadata == {}
    assert status == 'imported'
    assert error == 'No DOI found in PDF metadata or text.'
