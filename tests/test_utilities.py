"""Unit tests for paperscraper.utilities.

This module tests the small SQLite corpus maintenance helpers used by the
command-line interface to reset pipeline state and print progress.
"""

import paperscraper.corpus as corpus
import paperscraper.utilities as utilities
from paperscraper.corpus import PIPELINE_COLUMNS


def test_reset_restores_pipeline_defaults_and_marks_metadata_retrieved(tmp_path):
    """Test resetting pipeline columns in a corpus database."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {
            'paper_id': 'paper-1',
            'title': 'Lithium solid electrolyte',
            'metadata_status': 'failed',
            'text_download_status': 'succeeded',
            'pdf_download_status': 'failed',
            'num_images': 4,
            'num_text_materials': 3,
            'last_error': 'previous failure',
        })

    utilities.reset(str(db_path))

    with corpus.connect(db_path) as conn:
        row = corpus.paper_rows(conn)[0]
    assert row['paper_id'] == 'paper-1'
    assert row['title'] == 'Lithium solid electrolyte'
    assert row['metadata_status'] == 'retrieved'
    for column, default in PIPELINE_COLUMNS.items():
        if column == 'metadata_status':
            continue
        assert row[column] == default


def test_status_prints_pipeline_progress_summary(tmp_path, capsys):
    """Test printing a progress summary for a corpus database."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {
            'paper_id': 'paper-1',
            'metadata_status': 'retrieved',
            'text_download_status': 'succeeded',
            'pdf_download_status': 'succeeded',
            'text_scrape_status': 'succeeded',
            'image_scrape_status': 'failed',
            'store_status': 'stored',
            'num_text_materials': 2,
            'num_image_materials': 1,
        })
        corpus.upsert_paper(conn, {
            'paper_id': 'paper-2',
            'metadata_status': 'retrieved',
            'text_download_status': 'failed',
            'pdf_download_status': 'failed',
            'text_scrape_status': 'failed',
            'image_scrape_status': 'succeeded',
            'store_status': 'pending',
            'num_text_materials': 3,
            'num_image_materials': 4,
        })

    utilities.status(str(db_path))

    output = capsys.readouterr().out
    assert 'PaperScraper Progress Summary' in output
    assert 'Total papers: 2' in output
    assert 'Metadata retrieved: 2' in output
    assert 'Text downloaded: 1' in output
    assert 'Failed PDF downloads: 1' in output
    assert 'Text material rows extracted: 5' in output
    assert 'Image material rows extracted: 5' in output
