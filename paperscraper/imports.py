from pathlib import Path

import pandas as pd

from paperscraper.metadata import metadata_from_pdf
from paperscraper.pipeline import ensure_pipeline_columns, write_papers


def import_pdfs(papers_dir: str, papers_path: str = 'external_papers.csv', use_crossref: bool = True):
    pdf_dir = Path(papers_dir)
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f'{papers_dir} is not a directory.')
    rows = []
    for index, pdf_path in enumerate(sorted(pdf_dir.glob('*.pdf'))):
        stem = pdf_path.stem
        metadata, metadata_status, metadata_error = metadata_from_pdf(str(pdf_path), use_crossref=use_crossref)
        row = {
            'dc:identifier': f'external:{stem}',
            'prism:doi': '',
            'prism:coverDate': '',
            'metadata_status': metadata_status,
            'pdf_download_status': 'succeeded',
            'pdf_path': str(pdf_path),
            'text_download_status': 'pending',
            'text_scrape_status': 'pending',
            'image_scrape_status': 'pending',
            'store_status': 'pending',
            'text_path': '',
            'image_dir': '',
            'num_images': 0,
            'last_error': metadata_error,
        }
        row.update({key: value for key, value in metadata.items() if value})
        rows.append(row)
    if not rows:
        raise RuntimeError(f'No PDF files found in {papers_dir}.')
    papers_df = ensure_pipeline_columns(pd.DataFrame(rows))
    write_papers(papers_df, papers_path)
    enriched = int((papers_df['metadata_status'] == 'enriched').sum())
    doi_found = int((papers_df['prism:doi'] != '').sum()) if 'prism:doi' in papers_df.columns else 0
    print(f'Imported {len(papers_df)} PDFs into {papers_path} ({doi_found} DOI found, {enriched} enriched via Crossref).')
