from pathlib import Path

import pandas as pd

from paperscraper.metadata import metadata_from_pdf
from paperscraper.pipeline import ensure_pipeline_columns, write_papers


def _has_value(value):
    return not pd.isna(value) and str(value) != ''


def _matching_existing_index(papers_df, row):
    doi = row.get('prism:doi')
    if _has_value(doi) and 'prism:doi' in papers_df.columns:
        matches = papers_df.index[papers_df['prism:doi'].astype(str).str.lower() == str(doi).lower()].tolist()
        if matches:
            return matches[0]

    identifier = row.get('dc:identifier')
    if _has_value(identifier) and 'dc:identifier' in papers_df.columns:
        matches = papers_df.index[papers_df['dc:identifier'].astype(str) == str(identifier)].tolist()
        if matches:
            return matches[0]
    return None


def _merge_imported_pdf_row(papers_df, index, row):
    always_update = {
        'pdf_path',
        'pdf_download_status',
        'metadata_status',
        'last_error',
    }
    for column, value in row.items():
        if not _has_value(value):
            continue
        if column not in papers_df.columns:
            papers_df[column] = ''
        current = papers_df.at[index, column]
        if column in always_update or not _has_value(current):
            papers_df.at[index, column] = value
    return papers_df


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
            'num_text_materials': 0,
            'num_image_materials': 0,
            'last_error': metadata_error,
        }
        row.update({key: value for key, value in metadata.items() if value})
        rows.append(row)
    if not rows:
        raise RuntimeError(f'No PDF files found in {papers_dir}.')

    imported_df = ensure_pipeline_columns(pd.DataFrame(rows))
    if Path(papers_path).is_file():
        papers_df = ensure_pipeline_columns(pd.read_csv(papers_path, index_col=0))
    else:
        papers_df = ensure_pipeline_columns(pd.DataFrame())

    added = 0
    updated = 0
    for row in imported_df.to_dict(orient='records'):
        match_index = _matching_existing_index(papers_df, row) if not papers_df.empty else None
        if match_index is None:
            papers_df = pd.concat([papers_df, pd.DataFrame([row])], ignore_index=True)
            added += 1
        else:
            papers_df = _merge_imported_pdf_row(papers_df, match_index, row)
            updated += 1

    papers_df = ensure_pipeline_columns(papers_df)
    write_papers(papers_df, papers_path)
    enriched = int((imported_df['metadata_status'] == 'enriched').sum())
    doi_found = int((imported_df['prism:doi'] != '').sum()) if 'prism:doi' in imported_df.columns else 0
    print(
        f'Imported {len(imported_df)} PDFs into {papers_path} '
        f'({added} added, {updated} matched existing rows, {doi_found} DOI found, {enriched} enriched via Crossref).'
    )
