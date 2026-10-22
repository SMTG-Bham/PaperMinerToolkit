"""Small maintenance utilities for the PaperMinerToolkit corpus.

These helpers back CLI commands for resetting pipeline status and printing a
progress summary for the SQLite paper corpus.
"""

from __future__ import annotations

from os import PathLike
from contextlib import closing
from typing import Any

from paperminertoolkit.corpus.database import (PIPELINE_COLUMNS,
                                               connect,
                                               corpus_stats,
                                               paper_rows,
                                               upsert_paper)


def reset(db_path: str | PathLike[str] = 'papers.db') -> None:
    """Reset all corpus pipeline columns.

    Parameters
    ----------
    db_path : str or os.PathLike[str], optional
        Corpus database to reset.

    Raises
    ------
    RuntimeError
        If the corpus schema is newer than this package supports.
    """
    with connect(db_path) as conn:
        for paper in paper_rows(conn):
            for column, default in PIPELINE_COLUMNS.items():
                paper[column] = default
            paper['metadata_status'] = 'retrieved'
            upsert_paper(conn, paper)


def progress_summary(db_path: str | PathLike[str] = 'papers.db') -> dict[str, Any]:
    """Collect pipeline status counts without loading all paper metadata.

    Parameters
    ----------
    db_path : str or os.PathLike[str], optional
        Corpus database to summarize. Opens and migrates the schema as needed.

    Returns
    -------
    dict[str, Any]
        Paper count, structured-document coverage, counts by pipeline status,
        and extracted text/image material totals.
    """
    with closing(connect(db_path)) as conn, conn:
        stats = corpus_stats(conn)
        pipeline = {
            column: {row[0]: row[1] for row in conn.execute(
                f'SELECT {column}, COUNT(*) FROM papers GROUP BY {column}'
            )}
            for column in PIPELINE_COLUMNS if column.endswith('_status')
        }
        totals = conn.execute(
            'SELECT COALESCE(SUM(num_text_materials), 0), '
            'COALESCE(SUM(num_image_materials), 0) FROM papers'
        ).fetchone()
    return {
        'papers': stats['papers'],
        'papers_with_structured_documents': stats['papers_with_structured_documents'],
        'pipeline': pipeline,
        'text_materials': totals[0],
        'image_materials': totals[1],
    }


def status(db_path: str | PathLike[str] = 'papers.db') -> None:
    """Print a compact corpus progress summary.

    Parameters
    ----------
    db_path : str or os.PathLike[str], optional
        Corpus database to summarize.

    Raises
    ------
    RuntimeError
        If the corpus schema is newer than this package supports.
    """
    summary = progress_summary(db_path)
    print('\nPaperMinerToolkit Progress Summary')
    print('---------------------------')
    print(f'Total papers: {summary["papers"]}')
    print(f'Papers with structured documents: {summary["papers_with_structured_documents"]}')
    rows = [
        ('Metadata retrieved', 'metadata_status', 'retrieved'),
        ('Metadata enriched', 'enrichment_status', 'succeeded'),
        ('Text downloaded', 'text_download_status', 'succeeded'),
        ('PDFs downloaded', 'pdf_download_status', 'succeeded'),
        ('Text scraped', 'text_scrape_status', 'succeeded'),
        ('Images scraped', 'image_scrape_status', 'succeeded'),
        ('Stored', 'store_status', 'stored'),
        ('Failed text downloads', 'text_download_status', 'failed'),
        ('Failed PDF downloads', 'pdf_download_status', 'failed'),
        ('Failed text scrapes', 'text_scrape_status', 'failed'),
        ('Failed image scrapes', 'image_scrape_status', 'failed'),
    ]
    for label, column, value in rows:
        count = summary['pipeline'][column].get(value, 0)
        print(f'{label}: {count}')
    text_materials = summary['text_materials']
    image_materials = summary['image_materials']
    print(f'Text material rows extracted: {text_materials}')
    print(f'Image material rows extracted: {image_materials}')
    print('---------------------------\n')
