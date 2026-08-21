"""Import local PDFs into a PaperScraper SQLite corpus.

The import flow scans each PDF for DOI metadata, optionally enriches rows with
Crossref, stores the PDF bytes as corpus assets, and merges imported files with
existing paper rows so local PDFs can join the same corpus as searched papers.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from paperscraper.corpus import add_asset, connect, find_paper, normalize_paper, upsert_papers
from paperscraper.metadata import metadata_from_pdf


def import_pdfs(
    papers_dir: str | PathLike[str],
    db_path: str | PathLike[str] = 'papers.db',
    use_crossref: bool = True,
) -> None:
    """Import local PDF files into a paper corpus.

    Parameters
    ----------
    papers_dir : str or os.PathLike[str]
        Directory containing the PDF files to import.
    db_path : str or os.PathLike[str], optional
        Path to the destination corpus database.
    use_crossref : bool, optional
        Whether to enrich discovered DOI metadata with Crossref.

    Raises
    ------
    NotADirectoryError
        If ``papers_dir`` is not a directory.
    RuntimeError
        If the directory contains no PDF files or the corpus schema is newer
        than this package supports.
    """
    pdf_dir = Path(papers_dir)
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f'{papers_dir} is not a directory.')
    rows = []
    for index, pdf_path in enumerate(sorted(pdf_dir.glob('*.pdf'))):
        stem = pdf_path.stem
        metadata, metadata_status, metadata_error = metadata_from_pdf(str(pdf_path), use_crossref=use_crossref)
        row = {
            'paper_id': f'external:{stem}',
            'doi': '',
            'publication_date': '',
            'sources': 'external',
            'metadata_status': metadata_status,
            'pdf_download_status': 'succeeded',
            'pdf_path': '',
            'pdf_source': 'external',
            'text_download_status': 'pending',
            'text_scrape_status': 'pending',
            'image_scrape_status': 'pending',
            'store_status': 'pending',
            'text_path': '',
            'image_dir': '',
            'num_images': 0,
            'num_text_materials': 0,
            'num_image_materials': 0,
            'last_error': metadata_error,
        }
        row.update({key: value for key, value in metadata.items() if value})
        row['_pdf_path'] = pdf_path
        rows.append(row)
    if not rows:
        raise RuntimeError(f'No PDF files found in {papers_dir}.')

    paper_rows = [normalize_paper({key: value for key, value in row.items() if key != '_pdf_path'}) for row in rows]
    with connect(db_path) as conn:
        added, updated = upsert_papers(conn, paper_rows)
        for row in rows:
            paper = {key: value for key, value in row.items() if key != '_pdf_path'}
            matched = find_paper(conn, paper) or paper
            add_asset(
                conn,
                matched,
                row['_pdf_path'],
                role='pdf',
                kind='pdf',
                mime_type='application/pdf',
                source='external',
                original_filename=row['_pdf_path'].name,
            )
    enriched = sum(1 for row in paper_rows if row['metadata_status'] == 'enriched')
    doi_found = sum(1 for row in paper_rows if row['doi'])
    print(
        f'Imported {len(paper_rows)} PDFs into {db_path} '
        f'({added} added, {updated} matched existing rows, {doi_found} DOI found, {enriched} enriched via Crossref).'
    )
