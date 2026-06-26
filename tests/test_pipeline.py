"""Unit tests for paperscraper.pipeline.

This module tests the helpers that maintain the papers CSV schema, merge paper
records, update pipeline status, and resolve local file paths.
"""

import os

import pandas as pd
import pytest

from paperscraper.pipeline import (
    PAPER_COLUMNS,
    PIPELINE_COLUMNS,
    _clean_doi,
    _has_value,
    _merge_sources,
    _title_key,
    _year,
    ensure_pipeline_columns,
    existing_path,
    merge_paper_rows,
    normalize_paper_columns,
    read_papers,
    set_status,
    write_papers,
)


def test_has_value_rejects_empty_and_missing_values():
    """
    Test value detection for empty, missing, and meaningful values.

    This function performs the following steps:
    1. Passes empty strings, whitespace, None, and NaN to `_has_value`.
    2. Passes a non-empty DOI-like string to `_has_value`.
    3. Compares each result to the expected truth value.

    Asserts:
        - Empty or missing values return False.
        - A meaningful string returns True.
    """
    assert not _has_value('')
    assert not _has_value('   ')
    assert not _has_value(None)
    assert not _has_value(float('nan'))
    assert _has_value('10.1234/example')


def test_clean_doi_normalizes_common_doi_formats():
    """
    Test DOI normalization for common provider formats.

    This function performs the following steps:
    1. Defines DOI strings with a `doi:` prefix, DOI URL prefix, mixed casing, whitespace, and trailing punctuation.
    2. Normalizes each DOI with `_clean_doi`.
    3. Compares the normalized values to the canonical lower-case DOI.

    Asserts:
        - Prefixes and URL wrappers are removed.
        - DOI casing is normalized.
        - Empty DOI values return an empty string.
    """
    assert _clean_doi('doi:10.1234/ABC.') == '10.1234/abc'
    assert _clean_doi('https://doi.org/10.1234/ABC.') == '10.1234/abc'
    assert _clean_doi('  10.1234/ABC  ') == '10.1234/abc'
    assert _clean_doi('') == ''


def test_title_key_normalizes_case_and_punctuation():
    """
    Test title-key normalization for duplicate matching.

    This function performs the following steps:
    1. Defines a title with mixed casing and punctuation.
    2. Converts it to a comparison key with `_title_key`.
    3. Checks how `_title_key` handles a missing title.

    Asserts:
        - Punctuation is converted to spacing.
        - Casing is normalized to lower case.
        - Missing titles return an empty key.
    """
    assert _title_key('A Solid-Electrolyte Paper!') == 'a solid electrolyte paper'
    assert _title_key(None) == ''


def test_year_extracts_first_four_digit_year():
    """
    Test publication-year extraction from date-like values.

    This function performs the following steps:
    1. Defines a full date, a prose string containing a year, and an invalid date string.
    2. Extracts years with `_year`.
    3. Compares the extracted values to expected four-digit years or an empty string.

    Asserts:
        - Full dates return their year.
        - Text containing a year returns that year.
        - Values without a year return an empty string.
    """
    assert _year('2024-01-31') == '2024'
    assert _year('published in 1999') == '1999'
    assert _year('no date') == ''


def test_ensure_pipeline_columns_adds_defaults_and_coerces_counts():
    """
    Test that missing pipeline columns are added with normalized defaults.

    This function performs the following steps:
    1. Builds a DataFrame with partial pipeline fields and mixed count values.
    2. Calls `ensure_pipeline_columns`.
    3. Checks that status defaults and numeric coercion were applied.

    Asserts:
        - Existing metadata values are preserved.
        - Empty status values become `pending`.
        - Missing status fields are added as `pending`.
        - Count fields are converted to integers with invalid values set to zero.
    """
    raw = pd.DataFrame([
        {
            'paper_id': 'paper:1',
            'metadata_status': '',
            'num_images': '3',
            'num_text_materials': None,
            'num_image_materials': 'not-a-number',
        }
    ])

    normalized = ensure_pipeline_columns(raw)

    assert normalized.loc[0, 'paper_id'] == 'paper:1'
    assert normalized.loc[0, 'metadata_status'] == 'pending'
    assert normalized.loc[0, 'text_download_status'] == 'pending'
    assert normalized.loc[0, 'num_images'] == 3
    assert normalized.loc[0, 'num_text_materials'] == 0
    assert normalized.loc[0, 'num_image_materials'] == 0


def test_merge_paper_rows_deduplicates_by_doi_and_keeps_schema():
    """
    Test DOI-based deduplication and schema preservation during paper merges.

    This function performs the following steps:
    1. Creates one Elsevier-style row and one CORE-style row with the same DOI.
    2. Merges the incoming row into the existing row with `merge_paper_rows`.
    3. Checks merge counts, merged source provenance, carried metadata, and column order.

    Asserts:
        - No new row is added.
        - One existing row is updated.
        - Source names are merged without duplication.
        - CORE metadata is copied into the merged row.
        - The output schema matches the public paper columns plus pipeline columns.
    """
    existing = pd.DataFrame([
        {
            'paper_id': 'elsevier:1',
            'doi': '10.1234/ABC',
            'title': 'A Solid Electrolyte Paper',
            'publication_date': '2024-01-01',
            'sources': 'elsevier',
            'elsevier_link': 'full-text-link',
        }
    ])
    incoming = pd.DataFrame([
        {
            'paper_id': 'core:99',
            'doi': 'https://doi.org/10.1234/abc.',
            'title': 'A Solid Electrolyte Paper',
            'publication_date': '2024',
            'sources': 'core',
            'core_id': '99',
            'pdf_url': 'https://example.test/paper.pdf',
        }
    ])

    merged, added, updated = merge_paper_rows(existing, incoming)

    assert added == 0
    assert updated == 1
    assert len(merged) == 1
    assert merged.loc[0, 'sources'] == 'elsevier;core'
    assert merged.loc[0, 'core_id'] == '99'
    assert merged.loc[0, 'pdf_url'] == 'https://example.test/paper.pdf'
    assert list(merged.columns) == PAPER_COLUMNS + list(PIPELINE_COLUMNS)


def test_normalize_paper_columns_drops_unknown_columns_and_adds_defaults():
    """
    Test papers CSV schema normalization.

    This function performs the following steps:
    1. Builds a DataFrame with valid paper fields and an extra provider field.
    2. Normalizes it with `normalize_paper_columns`.
    3. Checks that only public paper and pipeline columns remain.

    Asserts:
        - Unknown provider fields are dropped.
        - Pipeline defaults are added.
        - Column order matches the declared schema.
    """
    raw = pd.DataFrame([
        {
            'paper_id': 'paper:1',
            'doi': '10.1234/test',
            'extra_provider_field': 'not kept',
        }
    ])

    normalized = normalize_paper_columns(raw)

    assert 'extra_provider_field' not in normalized.columns
    assert normalized.loc[0, 'metadata_status'] == 'pending'
    assert normalized.loc[0, 'num_images'] == 0
    assert list(normalized.columns) == PAPER_COLUMNS + list(PIPELINE_COLUMNS)


def test_merge_sources_preserves_order_and_removes_duplicates():
    """
    Test source provenance merging.

    This function performs the following steps:
    1. Combines overlapping semicolon-separated source strings.
    2. Combines an empty source string with a non-empty source string.
    3. Compares both outputs to the expected compact source lists.

    Asserts:
        - Source order follows first appearance.
        - Duplicate source names are removed.
        - Empty inputs do not add separators.
    """
    assert _merge_sources('elsevier;core', 'core;unpaywall') == 'elsevier;core;unpaywall'
    assert _merge_sources('', 'core') == 'core'


def test_merge_paper_rows_matches_by_paper_id_when_doi_is_missing():
    """
    Test paper-ID matching when DOI metadata is unavailable.

    This function performs the following steps:
    1. Creates an existing row and an incoming row with the same `paper_id`.
    2. Leaves DOI values absent from both rows.
    3. Merges the rows and checks the update result.

    Asserts:
        - No new row is added.
        - One existing row is updated.
        - Existing metadata is preserved.
        - New file-path metadata is copied into the existing row.
    """
    existing = pd.DataFrame([{'paper_id': 'external:paper-a', 'title': 'First title'}])
    incoming = pd.DataFrame([{'paper_id': 'external:paper-a', 'pdf_path': 'papers/paper-a.pdf'}])

    merged, added, updated = merge_paper_rows(existing, incoming)

    assert added == 0
    assert updated == 1
    assert len(merged) == 1
    assert merged.loc[0, 'title'] == 'First title'
    assert merged.loc[0, 'pdf_path'] == 'papers/paper-a.pdf'


def test_merge_paper_rows_matches_by_core_id():
    """
    Test CORE-ID matching during paper merges.

    This function performs the following steps:
    1. Creates two rows with different paper IDs but the same CORE ID.
    2. Merges the incoming row into the existing table.
    3. Checks that the CORE ID matched the existing row.

    Asserts:
        - No new row is added.
        - One existing row is updated.
        - Incoming PDF metadata is copied into the matched row.
    """
    existing = pd.DataFrame([{'paper_id': 'core:1', 'core_id': '1', 'title': 'Old'}])
    incoming = pd.DataFrame([{'paper_id': 'core:updated', 'core_id': '1', 'pdf_url': 'https://example.test/core.pdf'}])

    merged, added, updated = merge_paper_rows(existing, incoming)

    assert added == 0
    assert updated == 1
    assert merged.loc[0, 'pdf_url'] == 'https://example.test/core.pdf'


def test_merge_paper_rows_matches_by_title_and_year():
    """
    Test title-and-year matching for rows without shared identifiers.

    This function performs the following steps:
    1. Creates two rows with different paper IDs and no DOI or CORE ID match.
    2. Gives the rows equivalent titles with different punctuation and matching years.
    3. Merges the rows and checks the duplicate detection result.

    Asserts:
        - The incoming row updates the existing row.
        - No new row is added.
        - The merged table contains a single paper.
    """
    existing = pd.DataFrame([
        {'paper_id': 'source:a', 'title': 'A Solid Electrolyte Paper', 'publication_date': '2024-01-01'}
    ])
    incoming = pd.DataFrame([
        {'paper_id': 'source:b', 'title': 'A solid-electrolyte paper!', 'publication_date': '2024'}
    ])

    merged, added, updated = merge_paper_rows(existing, incoming)

    assert added == 0
    assert updated == 1
    assert len(merged) == 1


def test_merge_paper_rows_adds_new_rows_when_no_match_exists():
    """
    Test appending incoming papers when no duplicate match exists.

    This function performs the following steps:
    1. Creates an existing paper row.
    2. Creates an incoming paper row with a distinct DOI and paper ID.
    3. Merges the rows and checks the add/update counts.

    Asserts:
        - One new row is added.
        - No existing row is updated.
        - The merged table contains both papers.
    """
    existing = pd.DataFrame([{'paper_id': 'paper:1', 'doi': '10.1/one'}])
    incoming = pd.DataFrame([{'paper_id': 'paper:2', 'doi': '10.1/two'}])

    merged, added, updated = merge_paper_rows(existing, incoming)

    assert added == 1
    assert updated == 0
    assert len(merged) == 2


def test_read_and_write_papers_round_trip_normalized_schema(tmp_path):
    """
    Test papers CSV read/write round-tripping.

    This function performs the following steps:
    1. Creates a temporary papers CSV path.
    2. Writes a DataFrame with public fields and an unexpected field.
    3. Reads the CSV back through `read_papers`.

    Asserts:
        - Unexpected fields are not preserved.
        - Public paper fields round-trip correctly.
        - The reloaded DataFrame uses the normalized schema.
    """
    papers_path = tmp_path / 'papers.csv'
    raw = pd.DataFrame([{'paper_id': 'paper:1', 'doi': '10.1234/test', 'unexpected': 'drop me'}])

    write_papers(raw, str(papers_path))
    reloaded = read_papers(str(papers_path))

    assert 'unexpected' not in reloaded.columns
    assert reloaded.loc[0, 'paper_id'] == 'paper:1'
    assert list(reloaded.columns) == PAPER_COLUMNS + list(PIPELINE_COLUMNS)


def test_set_status_records_errors_and_clears_success_errors():
    """
    Test status updates with failure and success transitions.

    This function performs the following steps:
    1. Creates a normalized papers DataFrame.
    2. Records a failed PDF download status with an error message.
    3. Records a successful PDF download status.

    Asserts:
        - The failure status is written.
        - The failure error message is recorded.
        - The success status is written.
        - The prior error is cleared after success.
    """
    papers = ensure_pipeline_columns(pd.DataFrame([{'paper_id': 'paper:1'}]))

    set_status(papers, 0, 'pdf_download_status', 'failed', 'download failed')
    assert papers.loc[0, 'pdf_download_status'] == 'failed'
    assert papers.loc[0, 'last_error'] == 'download failed'

    set_status(papers, 0, 'pdf_download_status', 'succeeded')
    assert papers.loc[0, 'pdf_download_status'] == 'succeeded'
    assert papers.loc[0, 'last_error'] == ''


def test_set_status_rejects_unknown_status_columns():
    """
    Test validation of pipeline status column names.

    This function performs the following steps:
    1. Creates a normalized papers DataFrame.
    2. Attempts to set a status on an unknown column name.
    3. Captures the expected exception.

    Asserts:
        - A `KeyError` is raised for unknown status columns.
    """
    papers = ensure_pipeline_columns(pd.DataFrame([{'paper_id': 'paper:1'}]))

    with pytest.raises(KeyError):
        set_status(papers, 0, 'not_a_status', 'failed')


def test_existing_path_returns_only_existing_files(tmp_path):
    """
    Test existing-path detection for file path values.

    This function performs the following steps:
    1. Creates a temporary file.
    2. Checks the helper with the existing file, a missing file, and None.
    3. Confirms the returned path still points to a real file.

    Asserts:
        - Existing files are returned unchanged.
        - Missing files return None.
        - Non-string or empty values return None.
    """
    file_path = tmp_path / 'paper.pdf'
    file_path.write_text('placeholder')

    assert existing_path(str(file_path)) == str(file_path)
    assert existing_path(str(tmp_path / 'missing.pdf')) is None
    assert existing_path(None) is None
    assert os.path.isfile(existing_path(str(file_path)))
