from pathlib import Path

import pandas as pd

from paperscraper.metadata import metadata_from_pdf
from paperscraper.pipeline import ensure_pipeline_columns, merge_paper_rows, normalize_paper_columns, write_papers


def import_pdfs(papers_dir: str, papers_path: str = 'external_papers.csv', use_crossref: bool = True):
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
            'pdf_path': str(pdf_path),
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
        rows.append(row)
    if not rows:
        raise RuntimeError(f'No PDF files found in {papers_dir}.')

    imported_df = normalize_paper_columns(pd.DataFrame(rows))
    if Path(papers_path).is_file():
        papers_df = ensure_pipeline_columns(pd.read_csv(papers_path, index_col=0))
    else:
        papers_df = ensure_pipeline_columns(pd.DataFrame())

    papers_df, added, updated = merge_paper_rows(papers_df, imported_df)
    write_papers(papers_df, papers_path)
    enriched = int((imported_df['metadata_status'] == 'enriched').sum())
    doi_found = int((imported_df['doi'] != '').sum()) if 'doi' in imported_df.columns else 0
    print(
        f'Imported {len(imported_df)} PDFs into {papers_path} '
        f'({added} added, {updated} matched existing rows, {doi_found} DOI found, {enriched} enriched via Crossref).'
    )
