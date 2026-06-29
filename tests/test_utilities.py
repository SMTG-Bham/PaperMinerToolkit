"""Unit tests for paperscraper.utilities.

This module tests the small CSV maintenance helpers used by the command-line
interface to reset pipeline state, print progress, sort rows, and shuffle rows.
"""


import pandas as pd

from paperscraper.pipeline import PIPELINE_COLUMNS

import paperscraper.utilities as utilities


def test_reset_restores_pipeline_defaults_and_marks_metadata_retrieved(tmp_path):
    """
    Test resetting pipeline columns in a papers CSV.

    This function performs the following steps:
    1. Writes a temporary papers CSV with non-default pipeline statuses and counts.
    2. Calls `reset` on the temporary CSV path.
    3. Reloads the CSV and checks the reset pipeline values.

    Asserts:
        - Metadata status is set to `retrieved`.
        - Other pipeline columns are reset to their configured defaults.
        - Existing paper metadata is preserved.
    """
    papers_path = tmp_path / 'papers.csv'
    pd.DataFrame([{
        'paper_id': 'paper-1',
        'title': 'Lithium solid electrolyte',
        'metadata_status': 'failed',
        'text_download_status': 'succeeded',
        'pdf_download_status': 'failed',
        'num_images': 4,
        'num_text_materials': 3,
        'last_error': 'previous failure',
    }]).to_csv(papers_path)

    utilities.reset(str(papers_path))

    reset_df = pd.read_csv(papers_path, index_col=0)
    row = reset_df.iloc[0]
    assert row['paper_id'] == 'paper-1'
    assert row['title'] == 'Lithium solid electrolyte'
    assert row['metadata_status'] == 'retrieved'
    for column, default in PIPELINE_COLUMNS.items():
        if column == 'metadata_status':
            continue
        if default == '':
            assert pd.isna(row[column]) or row[column] == ''
        else:
            assert row[column] == default


def test_status_prints_pipeline_progress_summary(tmp_path, capsys):
    """
    Test printing a progress summary for a papers CSV.

    This function performs the following steps:
    1. Writes a temporary papers CSV with representative pipeline states.
    2. Calls `status` on the temporary CSV path.
    3. Captures the printed output.

    Asserts:
        - The total paper count is printed.
        - Succeeded and failed status counts are printed.
        - Extracted text and image material totals are printed.
    """
    papers_path = tmp_path / 'papers.csv'
    pd.DataFrame([
        {
            'paper_id': 'paper-1',
            'metadata_status': 'retrieved',
            'text_download_status': 'succeeded',
            'pdf_download_status': 'succeeded',
            'text_scrape_status': 'succeeded',
            'image_scrape_status': 'failed',
            'store_status': 'stored',
            'num_text_materials': 2,
            'num_image_materials': 1,
        },
        {
            'paper_id': 'paper-2',
            'metadata_status': 'retrieved',
            'text_download_status': 'failed',
            'pdf_download_status': 'failed',
            'text_scrape_status': 'failed',
            'image_scrape_status': 'succeeded',
            'store_status': 'pending',
            'num_text_materials': 3,
            'num_image_materials': 4,
        },
    ]).to_csv(papers_path)

    utilities.status(str(papers_path))

    output = capsys.readouterr().out
    assert 'PaperScraper Progress Summary' in output
    assert 'Total papers: 2' in output
    assert 'Metadata retrieved: 2' in output
    assert 'Text downloaded: 1' in output
    assert 'Failed PDF downloads: 1' in output
    assert 'Text material rows extracted: 5' in output
    assert 'Image material rows extracted: 5' in output


def test_sort_orders_rows_by_field_and_resets_index(tmp_path):
    """
    Test sorting a papers CSV by a selected field.

    This function performs the following steps:
    1. Writes a temporary papers CSV with titles out of order.
    2. Calls `sort` using the title field.
    3. Reloads the sorted CSV.

    Asserts:
        - Rows are sorted by the requested field.
        - The saved CSV index is reset after sorting.
    """
    papers_path = tmp_path / 'papers.csv'
    pd.DataFrame([
        {'paper_id': 'paper-2', 'title': 'Zeta'},
        {'paper_id': 'paper-1', 'title': 'Alpha'},
        {'paper_id': 'paper-3', 'title': 'Gamma'},
    ]).to_csv(papers_path)

    utilities.sort(str(papers_path), field='title')

    sorted_df = pd.read_csv(papers_path, index_col=0)
    assert sorted_df['paper_id'].tolist() == ['paper-1', 'paper-3', 'paper-2']
    assert sorted_df.index.tolist() == [0, 1, 2]


def test_shuffle_writes_sampled_row_order(tmp_path, monkeypatch):
    """
    Test shuffling a papers CSV.

    This function performs the following steps:
    1. Writes a temporary papers CSV with three rows.
    2. Replaces DataFrame sampling with a deterministic row order.
    3. Calls `shuffle` and reloads the CSV.

    Asserts:
        - The shuffled CSV uses the sampled row order.
        - The saved CSV index is reset after shuffling.
    """
    papers_path = tmp_path / 'papers.csv'
    pd.DataFrame([
        {'paper_id': 'paper-1', 'title': 'Alpha'},
        {'paper_id': 'paper-2', 'title': 'Beta'},
        {'paper_id': 'paper-3', 'title': 'Gamma'},
    ]).to_csv(papers_path)

    def deterministic_sample(self, frac):
        assert frac == 1
        return self.iloc[[2, 0, 1]]

    monkeypatch.setattr(pd.DataFrame, 'sample', deterministic_sample)

    utilities.shuffle(str(papers_path))

    shuffled_df = pd.read_csv(papers_path, index_col=0)
    assert shuffled_df['paper_id'].tolist() == ['paper-3', 'paper-1', 'paper-2']
    assert shuffled_df.index.tolist() == [0, 1, 2]
