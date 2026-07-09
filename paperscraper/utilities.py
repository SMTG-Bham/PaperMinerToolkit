"""Small maintenance utilities for the PaperScraper corpus.

These helpers back CLI commands for resetting pipeline status and printing a
progress summary for the SQLite paper corpus.
"""

from paperscraper.corpus import PIPELINE_COLUMNS, connect, paper_rows, upsert_paper


def reset(db_path: str = 'papers.db'):
    """Reset all pipeline columns in a corpus database to their initial statuses."""
    with connect(db_path) as conn:
        for paper in paper_rows(conn):
            for column, default in PIPELINE_COLUMNS.items():
                paper[column] = default
            paper['metadata_status'] = 'retrieved'
            upsert_paper(conn, paper)


def status(db_path: str = 'papers.db'):
    """Print a compact progress summary for a corpus database."""
    with connect(db_path) as conn:
        papers = paper_rows(conn)
    print('\nPaperScraper Progress Summary')
    print('---------------------------')
    print(f'Total papers: {len(papers)}')
    rows = [
        ('Metadata retrieved', 'metadata_status', 'retrieved'),
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
        count = sum(1 for paper in papers if paper.get(column) == value)
        print(f'{label}: {count}')
    text_materials = sum(int(paper.get('num_text_materials') or 0) for paper in papers)
    image_materials = sum(int(paper.get('num_image_materials') or 0) for paper in papers)
    print(f'Text material rows extracted: {text_materials}')
    print(f'Image material rows extracted: {image_materials}')
    print('---------------------------\n')
