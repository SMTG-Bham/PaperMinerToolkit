"""Unit tests for paperscraper.metadata.

This module tests DOI cleanup and extraction from text, embedded PDF metadata,
PDF page text fallback, Crossref date formatting, and Crossref metadata
normalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import paperscraper.corpus as corpus
import paperscraper.crossref as crossref
import paperscraper.metadata as metadata

DATA_DIR = Path(__file__).parent / 'data'
FIXTURE_PDF = DATA_DIR / 'disorder-driven_fast_na_transport_oxychlorides.pdf'


def test_clean_doi_removes_trailing_punctuation() -> None:
    """Remove trailing punctuation from a DOI."""
    assert metadata.clean_doi('10.1002/aenm.70977.') == '10.1002/aenm.70977'


def test_clean_doi_normalizes_labels_resolver_urls_and_case() -> None:
    """Test canonicalization of common DOI presentation formats."""
    assert metadata.clean_doi('DOI: 10.1234/EXAMPLE') == '10.1234/example'
    assert metadata.clean_doi('https://doi.org/10.1234/EXAMPLE%2BONE?source=pdf') == '10.1234/example+one'


def test_clean_doi_preserves_balanced_suffix_delimiters() -> None:
    """Test that sentence delimiters are removed without damaging a DOI suffix."""
    assert metadata.clean_doi('10.1234/example(test)).') == '10.1234/example(test)'


def test_extract_doi_from_text_finds_first_doi() -> None:
    """Extract the first DOI from text or return ``None``."""
    assert metadata.extract_doi_from_text('See doi 10.1002/aenm.70977.') == '10.1002/aenm.70977'
    assert metadata.extract_doi_from_text('See https://doi.org/10.1234/EXAMPLE%2BONE?source=pdf') == (
        '10.1234/example+one'
    )
    assert metadata.extract_doi_from_text('No DOI here.') is None


def test_normalize_metadata_text_flattens_unicode_punctuation_and_super_subscripts() -> None:
    """Flatten Unicode punctuation and super/subscripts in metadata text."""
    text = 'Disorder‐Driven Na⁺ <sup>+</sup> Transport in Li₁₀GeP₂S₁₂ &amp; oxides'

    assert metadata.normalize_metadata_text(text) == 'Disorder-Driven Na+ + Transport in Li10GeP2S12 & oxides'


def test_normalize_metadata_text_handles_missing_and_uncommon_punctuation() -> None:
    """Normalize missing values and uncommon Unicode punctuation."""
    assert metadata.normalize_metadata_text(None) == ''
    assert metadata.normalize_metadata_text('A‹quoted› title') == 'A"quoted" title'
    assert metadata.normalize_metadata_text('Charge⁺ carrier') == 'Charge+ carrier'


def test_normalize_punctuation_char_covers_fallback_branches() -> None:
    """Normalize uncommon punctuation through each fallback branch."""
    assert metadata._normalize_punctuation_char('﹣') == '-'
    assert metadata._normalize_punctuation_char('＇') == "'"
    assert metadata._normalize_punctuation_char('。') == ''
    assert metadata._normalize_punctuation_char('β') == 'β'


def test_extract_dois_from_text_ranks_candidates_by_frequency() -> None:
    """Rank repeated DOI candidates ahead of one-off artifacts."""
    text = (
        'Header DOI 10.1002/aenm.709771of15 '
        'Body DOI 10.1002/aenm.70977 '
        'Footer DOI 10.1002/aenm.70977'
    )

    assert metadata.extract_dois_from_text(text) == ['10.1002/aenm.70977', '10.1002/aenm.709771of15']


def test_extract_dois_from_text_deduplicates_case_insensitively() -> None:
    """Test that differently cased forms of one DOI count as one candidate."""
    text = 'First 10.1234/EXAMPLE then 10.1234/example and finally 10.9999/other.'

    assert metadata.extract_dois_from_text(text) == ['10.1234/example', '10.9999/other']


def test_extract_dois_from_text_handles_pdf_text_artifacts() -> None:
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


def test_extract_dois_from_text_supports_legacy_crossref_formats() -> None:
    """Test valid legacy publisher DOI forms excluded by the modern pattern."""
    wiley = '10.1002/(SICI)1099-0844(199912)17:4<290::AID-CBF849>3.0.CO;2-P'
    taylor_and_francis = '10.1207/S15327965PLI1503&4_01'

    assert metadata.extract_dois_from_text(f'{wiley}. {taylor_and_francis}.') == [
        wiley.casefold(),
        taylor_and_francis.casefold(),
    ]
    assert metadata.extract_doi_from_text(taylor_and_francis.replace('&', '&amp;')) == taylor_and_francis.casefold()


def test_extract_doi_from_pdf_metadata_reads_fixture_article_doi() -> None:
    """Extract the article DOI embedded in the fixture PDF metadata."""
    assert metadata.extract_doi_from_pdf_metadata(str(FIXTURE_PDF)) == '10.1002/aenm.70977'


def test_extract_doi_from_pdf_metadata_returns_none_when_no_metadata_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return ``None`` when PDF metadata contains no DOI."""

    class FakeReader:
        """Provide PDF metadata without a DOI."""

        metadata = {'/Title': 'A PDF without a DOI'}

        def __init__(self, _: str) -> None:
            """Initialize the fake reader without reading a file."""
            return None

    monkeypatch.setattr(metadata, 'PdfReader', FakeReader)

    assert metadata.extract_doi_from_pdf_metadata('paper.pdf') is None


def test_extract_doi_from_pdf_prefers_metadata_before_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefer a metadata DOI over a page-text candidate."""
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: ['10.1234/metadata', '10.1234/text'])

    assert metadata.extract_doi_from_pdf('paper.pdf') == '10.1234/metadata'


def test_extract_doi_from_pdf_falls_back_to_page_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to page text when PDF metadata has no DOI."""
    monkeypatch.setattr(metadata, 'extract_dois_from_pdf_metadata', lambda _: [])
    monkeypatch.setattr(metadata, 'read_pdf_text', lambda _: 'Text DOI 10.1234/text.')

    assert metadata.extract_doi_from_pdf('paper.pdf') == '10.1234/text'


def test_extract_doi_from_pdf_returns_none_when_no_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return ``None`` when a PDF contains no DOI candidates."""
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: [])

    assert metadata.extract_doi_from_pdf('paper.pdf') is None


def test_date_from_parts_handles_missing_and_short_dates() -> None:
    """Format missing, year-only, and year-month Crossref dates."""
    assert metadata._date_from_parts(None) == ''
    assert metadata._date_from_parts([[]]) == ''
    assert metadata._date_from_parts([[2024]]) == '2024'
    assert metadata._date_from_parts([[2024, 2]]) == '2024-02'


def test_published_date_returns_empty_string_when_no_dates_are_available() -> None:
    """Return an empty publication date when Crossref has no dates."""
    assert metadata._published_date({}) == ''


def test_get_crossref_metadata_normalizes_text_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalize Crossref text while preserving DOI and date fields."""

    class FakeResponse:
        """Provide a successful Crossref metadata response."""

        def raise_for_status(self) -> None:
            """Accept the fake response status."""
            return None

        def json(self) -> dict[str, Any]:
            """Return representative Crossref response data."""
            return {
                'message': {
                    'DOI': '10.1234/example',
                    'published-online': {'date-parts': [[2024, 2, 3]]},
                    'title': ['Disorder‐Driven Na<sup>+</sup> Transport'],
                    'container-title': ['Advanced Energy Materials'],
                    'type': 'journal-article',
                    'publisher': 'Publisher\u2019s Name',
                    'volume': '14',
                    'issue': '7',
                    'page': '54-58',
                    'language': 'en',
                    'issn-type': [
                        {'value': '1614-6840', 'type': 'electronic'},
                        {'value': '1614-6832', 'type': 'print'},
                    ],
                }
            }

    monkeypatch.setattr(crossref, 'work_by_doi',
                        lambda *_, **__: FakeResponse().json()['message'])

    crossref_metadata = metadata.get_crossref_metadata('10.1234/example', email='me@example.com')

    assert crossref_metadata['doi'] == '10.1234/example'
    assert crossref_metadata['publication_date'] == '2024-02-03'
    assert crossref_metadata['title'] == 'Disorder-Driven Na+ Transport'
    assert crossref_metadata['journal'] == 'Advanced Energy Materials'
    assert crossref_metadata['publisher'] == "Publisher's Name"
    assert crossref_metadata['work_type'] == 'journal-article'
    assert crossref_metadata['volume'] == '14'
    assert crossref_metadata['issue'] == '7'
    assert crossref_metadata['pages'] == '54-58'
    assert crossref_metadata['language'] == 'en'
    assert crossref_metadata['issn'] == '1614-6832;1614-6840'
    assert crossref_metadata['crossref_message']['DOI'] == '10.1234/example'


def test_get_crossref_metadata_returns_only_known_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return corpus columns and the raw payload, and nothing else."""

    class FakeResponse:
        """Provide a minimal Crossref metadata response."""

        def raise_for_status(self) -> None:
            """Accept the fake response status."""
            return None

        def json(self) -> dict[str, Any]:
            """Return a minimal Crossref message."""
            return {'message': {'DOI': '10.1234/example'}}

    monkeypatch.setattr(crossref, 'work_by_doi',
                        lambda *_, **__: FakeResponse().json()['message'])

    keys = set(metadata.get_crossref_metadata('10.1234/example', email='me@example.com'))

    known = set(corpus.PAPER_FIELDS) | set(corpus.ENRICHMENT_COLUMNS) | {'crossref_message'}
    assert keys <= known
    assert 'crossref_type' not in keys
    assert 'crossref_publisher' not in keys



def test_metadata_from_pdf_handles_missing_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report an imported PDF that contains no DOI."""
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: [])

    paper_metadata, status, error = metadata.metadata_from_pdf('paper.pdf', use_crossref=True)

    assert paper_metadata == {}
    assert status == 'imported'
    assert error == 'No DOI found in PDF metadata or text.'


def test_metadata_from_pdf_handles_pdf_read_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report PDF read errors without returning metadata."""
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: (_ for _ in ()).throw(RuntimeError('bad pdf')))

    paper_metadata, status, error = metadata.metadata_from_pdf('paper.pdf', use_crossref=True)

    assert paper_metadata == {}
    assert status == 'imported'
    assert 'Could not read PDF metadata text: bad pdf' == error


def test_metadata_from_pdf_returns_basic_metadata_without_crossref(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return basic DOI metadata when Crossref enrichment is disabled."""
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: ['10.1234/basic'])

    paper_metadata, status, error = metadata.metadata_from_pdf('paper.pdf', use_crossref=False)

    assert paper_metadata == {'doi': '10.1234/basic'}
    assert status == 'doi_found'
    assert error == ''


def test_metadata_from_pdf_validates_ranked_doi_candidates_with_crossref(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enrich the first Crossref-valid DOI candidate."""
    monkeypatch.setattr(metadata, 'doi_candidates_from_pdf', lambda _: ['10.1002/aenm.709771of15', '10.1002/aenm.70977'])

    def fake_get_crossref_metadata(doi: str) -> dict[str, str]:
        """Reject the malformed candidate and enrich the valid DOI."""
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


def test_metadata_from_pdf_reports_when_all_crossref_candidates_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report all DOI candidates when Crossref enrichment fails."""
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
