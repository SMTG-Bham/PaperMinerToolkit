"""Unit tests for paperscraper.metadata.

This module tests DOI cleanup and extraction from text, embedded PDF metadata,
PDF page text fallback, Crossref date formatting, and Crossref metadata
normalization.
"""

from pathlib import Path

import paperscraper.metadata as metadata

DATA_DIR = Path(__file__).parent / 'data'
FIXTURE_PDF = DATA_DIR / 'disorder-driven_fast_na_transport_oxychlorides.pdf'


def test_clean_doi_removes_trailing_punctuation():
    """
    Test DOI cleanup for trailing punctuation.

    This function performs the following steps:
    1. Defines a DOI string with trailing punctuation.
    2. Cleans the DOI with `clean_doi`.
    3. Compares the cleaned value to the expected DOI.

    Asserts:
        - Trailing punctuation is removed.
    """
    assert metadata.clean_doi('10.1002/aenm.70977.') == '10.1002/aenm.70977'


def test_clean_doi_normalizes_labels_resolver_urls_and_case():
    """Test canonicalization of common DOI presentation formats."""
    assert metadata.clean_doi('DOI: 10.1234/EXAMPLE') == '10.1234/example'
    assert metadata.clean_doi('https://doi.org/10.1234/EXAMPLE%2BONE?source=pdf') == '10.1234/example+one'


def test_clean_doi_preserves_balanced_suffix_delimiters():
    """Test that sentence delimiters are removed without damaging a DOI suffix."""
    assert metadata.clean_doi('10.1234/example(test)).') == '10.1234/example(test)'


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
    assert metadata.extract_doi_from_text('See https://doi.org/10.1234/EXAMPLE%2BONE?source=pdf') == (
        '10.1234/example+one'
    )
    assert metadata.extract_doi_from_text('No DOI here.') is None


def test_normalize_metadata_text_flattens_unicode_punctuation_and_super_subscripts():
    """
    Test metadata text normalization for punctuation and super/subscript content.

    This function performs the following steps:
    1. Defines metadata text with Unicode hyphens, superscript characters, subscript characters, and HTML tags.
    2. Normalizes the text with `normalize_metadata_text`.
    3. Compares the normalized text to a plain-text value.

    Asserts:
        - Unicode punctuation is converted to plain punctuation.
        - Superscript and subscript characters are converted to normal characters.
        - HTML superscript and subscript tags are removed while preserving their text.
    """
    text = 'Disorder‐Driven Na⁺ <sup>+</sup> Transport in Li₁₀GeP₂S₁₂ &amp; oxides'

    assert metadata.normalize_metadata_text(text) == 'Disorder-Driven Na+ + Transport in Li10GeP2S12 & oxides'


def test_normalize_metadata_text_handles_missing_and_uncommon_punctuation():
    """
    Test metadata text normalization for missing values and uncommon Unicode punctuation.

    This function performs the following steps:
    1. Normalizes a missing value.
    2. Normalizes text containing non-ASCII quote punctuation.
    3. Normalizes text containing non-ASCII punctuation without a direct ASCII mapping.

    Asserts:
        - Missing values normalize to an empty string.
        - Unicode quote punctuation is converted to plain quotes.
        - Unmapped Unicode punctuation is removed when no ASCII approximation exists.
        - Non-punctuation Unicode characters are preserved.
    """
    assert metadata.normalize_metadata_text(None) == ''
    assert metadata.normalize_metadata_text('A‹quoted› title') == 'A"quoted" title'
    assert metadata.normalize_metadata_text('Charge⁺ carrier') == 'Charge+ carrier'


def test_normalize_punctuation_char_covers_fallback_branches():
    """
    Test direct normalization of uncommon punctuation characters.

    This function performs the following steps:
    1. Normalizes one dash-like character.
    2. Normalizes one apostrophe-like character.
    3. Normalizes one generic punctuation character.
    4. Normalizes one non-punctuation Unicode character.

    Asserts:
        - Dash-like characters normalize to a hyphen.
        - Apostrophe-like characters normalize to a plain apostrophe.
        - Generic punctuation with no ASCII decomposition normalizes to an empty string.
        - Non-punctuation Unicode characters are preserved.
    """
    assert metadata._normalize_punctuation_char('﹣') == '-'
    assert metadata._normalize_punctuation_char('＇') == "'"
    assert metadata._normalize_punctuation_char('。') == ''
    assert metadata._normalize_punctuation_char('β') == 'β'


def test_extract_dois_from_text_ranks_candidates_by_frequency():
    """
    Test ranking DOI candidates found in extracted PDF text.

    This function performs the following steps:
    1. Defines text containing one layout-corrupted DOI candidate and two clean DOI candidates.
    2. Extracts ranked DOI candidates with `extract_dois_from_text`.
    3. Compares the ranked candidates to the expected order.

    Asserts:
        - The repeated clean DOI candidate is ranked before the one-off corrupted candidate.
    """
    text = (
        'Header DOI 10.1002/aenm.709771of15 '
        'Body DOI 10.1002/aenm.70977 '
        'Footer DOI 10.1002/aenm.70977'
    )

    assert metadata.extract_dois_from_text(text) == ['10.1002/aenm.70977', '10.1002/aenm.709771of15']


def test_extract_dois_from_text_deduplicates_case_insensitively():
    """Test that differently cased forms of one DOI count as one candidate."""
    text = 'First 10.1234/EXAMPLE then 10.1234/example and finally 10.9999/other.'

    assert metadata.extract_dois_from_text(text) == ['10.1234/example', '10.9999/other']


def test_extract_dois_from_text_handles_pdf_text_artifacts():
    """Test DOI extraction across common invisible and line-wrap artifacts."""
    text = (
        'Prefix wrap 10.1234/\nwrapped '
        'soft hyphen 10.1234/soft\u00ad\nhyphen '
        'zero width 10.1234/zero\u200bwidth '
        'Unicode dash 10.1234/unicode\u2010dash'
    )

    assert metadata.extract_dois_from_text(text) == [
        '10.1234/wrapped',
        '10.1234/softhyphen',
        '10.1234/zerowidth',
        '10.1234/unicode-dash',
    ]


def test_extract_dois_from_text_supports_legacy_crossref_formats():
    """Test valid legacy publisher DOI forms excluded by the modern pattern."""
    wiley = '10.1002/(SICI)1099-0844(199912)17:4<290::AID-CBF849>3.0.CO;2-P'
    taylor_and_francis = '10.1207/S15327965PLI1503&4_01'

    assert metadata.extract_dois_from_text(f'{wiley}. {taylor_and_francis}.') == [
        wiley.casefold(),
        taylor_and_francis.casefold(),
    ]
    assert metadata.extract_doi_from_text(taylor_and_francis.replace('&', '&amp;')) == taylor_and_francis.casefold()


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


def test_extract_doi_from_pdf_metadata_returns_none_when_no_metadata_doi(monkeypatch):
    """
    Test PDF metadata DOI extraction when no metadata fields contain a DOI.

    This function performs the following steps:
    1. Replaces `PdfReader` with a reader whose metadata contains no DOI.
    2. Calls `extract_doi_from_pdf_metadata`.
    3. Checks the returned value.

    Asserts:
        - Missing DOI metadata returns None.
    """

    class FakeReader:
        metadata = {'/Title': 'A PDF without a DOI'}

        def __init__(self, _):
            return None

    monkeypatch.setattr(metadata, 'PdfReader', FakeReader)

    assert metadata.extract_doi_from_pdf_metadata('paper.pdf') is None


def test_extract_doi_from_pdf_prefers_metadata_before_text(monkeypatch):
    """
    Test DOI extraction preference for PDF metadata over page text.

    This function performs the following steps:
    1. Replaces PDF DOI candidate extraction with a metadata candidate followed by a text candidate.
    2. Calls `extract_doi_from_pdf`.
    3. Compares the returned DOI to the metadata candidate.

    Asserts:
        - The metadata DOI is returned.
        - The text DOI candidate is not selected ahead of the metadata DOI candidate.
    """
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: ['10.1234/metadata', '10.1234/text'])

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
    monkeypatch.setattr(metadata, 'extract_dois_from_pdf_metadata', lambda _: [])
    monkeypatch.setattr(metadata, 'read_pdf_text', lambda _: 'Text DOI 10.1234/text.')

    assert metadata.extract_doi_from_pdf('paper.pdf') == '10.1234/text'


def test_extract_doi_from_pdf_returns_none_when_no_candidates(monkeypatch):
    """
    Test PDF DOI extraction when neither metadata nor text contains candidates.

    This function performs the following steps:
    1. Replaces PDF DOI candidate extraction with an empty list.
    2. Calls `extract_doi_from_pdf`.
    3. Checks the returned value.

    Asserts:
        - Missing metadata and text DOI candidates return None.
    """
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: [])

    assert metadata.extract_doi_from_pdf('paper.pdf') is None


def test_date_from_parts_handles_missing_and_short_dates():
    """
    Test Crossref date formatting for missing and short date parts.

    This function performs the following steps:
    1. Passes missing and empty date-parts values to `_date_from_parts`.
    2. Passes year-only and year-month date-parts values to `_date_from_parts`.
    3. Compares each result to the expected formatted value.

    Asserts:
        - Missing date parts return an empty string.
        - Year-only dates format as a four-digit year.
        - Year-month dates format as `YYYY-MM`.
    """
    assert metadata._date_from_parts(None) == ''
    assert metadata._date_from_parts([[]]) == ''
    assert metadata._date_from_parts([[2024]]) == '2024'
    assert metadata._date_from_parts([[2024, 2]]) == '2024-02'


def test_published_date_returns_empty_string_when_no_dates_are_available():
    """
    Test publication date selection when Crossref has no usable dates.

    This function performs the following steps:
    1. Defines an empty Crossref message.
    2. Calls `_published_date`.
    3. Checks the returned value.

    Asserts:
        - Messages without publication dates return an empty string.
    """
    assert metadata._published_date({}) == ''


def test_get_crossref_metadata_normalizes_text_fields(monkeypatch):
    """
    Test normalization of Crossref text fields.

    This function performs the following steps:
    1. Replaces the Crossref HTTP response with a local response object.
    2. Returns Crossref metadata containing Unicode punctuation and HTML superscript tags.
    3. Calls `get_crossref_metadata`.

    Asserts:
        - Title, journal, and publisher text fields are normalized.
        - DOI and publication date fields are preserved.
    """

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'message': {
                    'DOI': '10.1234/example',
                    'published-online': {'date-parts': [[2024, 2, 3]]},
                    'title': ['Disorder‐Driven Na<sup>+</sup> Transport'],
                    'container-title': ['Advanced Energy Materials'],
                    'type': 'journal-article',
                    'publisher': 'Publisher\u2019s Name',
                }
            }

    monkeypatch.setattr(metadata.requests, 'get', lambda *_, **__: FakeResponse())

    crossref_metadata = metadata.get_crossref_metadata('10.1234/example')

    assert crossref_metadata['doi'] == '10.1234/example'
    assert crossref_metadata['publication_date'] == '2024-02-03'
    assert crossref_metadata['title'] == 'Disorder-Driven Na+ Transport'
    assert crossref_metadata['journal'] == 'Advanced Energy Materials'
    assert crossref_metadata['crossref_publisher'] == "Publisher's Name"


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
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: [])

    paper_metadata, status, error = metadata.metadata_from_pdf('paper.pdf', use_crossref=True)

    assert paper_metadata == {}
    assert status == 'imported'
    assert error == 'No DOI found in PDF metadata or text.'


def test_metadata_from_pdf_handles_pdf_read_errors(monkeypatch):
    """
    Test metadata extraction when PDF reading fails.

    This function performs the following steps:
    1. Replaces PDF DOI candidate extraction with a function that raises an error.
    2. Calls `metadata_from_pdf`.
    3. Checks the returned metadata, status, and error message.

    Asserts:
        - No metadata fields are returned.
        - The import status remains `imported`.
        - The error message includes the PDF read failure.
    """
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: (_ for _ in ()).throw(RuntimeError('bad pdf')))

    paper_metadata, status, error = metadata.metadata_from_pdf('paper.pdf', use_crossref=True)

    assert paper_metadata == {}
    assert status == 'imported'
    assert 'Could not read PDF metadata text: bad pdf' == error


def test_metadata_from_pdf_returns_basic_metadata_without_crossref(monkeypatch):
    """
    Test metadata extraction when Crossref enrichment is disabled.

    This function performs the following steps:
    1. Replaces PDF DOI candidate extraction with one DOI candidate.
    2. Calls `metadata_from_pdf` with Crossref enrichment disabled.
    3. Checks the returned metadata, status, and error message.

    Asserts:
        - The first DOI candidate is returned as basic metadata.
        - The metadata status is `doi_found`.
        - No error message is returned.
    """
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: ['10.1234/basic'])

    paper_metadata, status, error = metadata.metadata_from_pdf('paper.pdf', use_crossref=False)

    assert paper_metadata == {'doi': '10.1234/basic'}
    assert status == 'doi_found'
    assert error == ''


def test_metadata_from_pdf_validates_ranked_doi_candidates_with_crossref(monkeypatch):
    """
    Test Crossref validation across ranked DOI candidates.

    This function performs the following steps:
    1. Replaces PDF DOI candidate extraction with a bad candidate followed by a valid candidate.
    2. Replaces Crossref metadata lookup with a function that rejects the bad candidate.
    3. Calls `metadata_from_pdf` with Crossref enrichment enabled.

    Asserts:
        - The failed first DOI candidate is skipped.
        - The first Crossref-valid DOI candidate is returned with enriched metadata.
        - The metadata status is `enriched`.
    """
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: ['10.1002/aenm.709771of15', '10.1002/aenm.70977'])

    def fake_get_crossref_metadata(doi):
        if doi == '10.1002/aenm.709771of15':
            raise metadata.requests.HTTPError('not found')
        return {
            'doi': doi,
            'publication_date': '2026-04-23',
            'title': 'Disorder-Driven Fast Na Transport',
            'journal': 'Advanced Energy Materials',
        }

    monkeypatch.setattr(metadata, 'get_crossref_metadata', fake_get_crossref_metadata)

    paper_metadata, status, error = metadata.metadata_from_pdf('paper.pdf', use_crossref=True)

    assert paper_metadata['doi'] == '10.1002/aenm.70977'
    assert paper_metadata['title'] == 'Disorder-Driven Fast Na Transport'
    assert status == 'enriched'
    assert error == ''


def test_metadata_from_pdf_reports_when_all_crossref_candidates_fail(monkeypatch):
    """
    Test Crossref failure reporting when every DOI candidate fails validation.

    This function performs the following steps:
    1. Replaces PDF DOI candidate extraction with two DOI candidates.
    2. Replaces Crossref metadata lookup with a function that always raises an HTTP error.
    3. Calls `metadata_from_pdf` with Crossref enrichment enabled.

    Asserts:
        - The first DOI candidate is kept as basic metadata.
        - The metadata status is `doi_found`.
        - The error message includes the failed candidate list.
    """
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: ['10.1234/first', '10.1234/second'])
    monkeypatch.setattr(
        metadata,
        'get_crossref_metadata',
        lambda _: (_ for _ in ()).throw(metadata.requests.HTTPError('not found')),
    )

    paper_metadata, status, error = metadata.metadata_from_pdf('paper.pdf', use_crossref=True)

    assert paper_metadata == {'doi': '10.1234/first'}
    assert status == 'doi_found'
    assert '10.1234/first, 10.1234/second' in error
