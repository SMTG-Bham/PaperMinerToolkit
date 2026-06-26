"""Define the papers CSV schema and shared helpers for pipeline state.

This module keeps paper metadata columns small and user-facing, adds missing
pipeline status columns, reads/writes normalized CSVs, and merges duplicate
paper rows from multiple search or import sources.
"""

import os
import pandas as pd
import re

PAPER_COLUMNS = [
    'paper_id',
    'doi',
    'title',
    'journal',
    'publication_date',
    'authors',
    'sources',
    'core_id',
    'pdf_url',
    'pdf_source',
    'text_source',
    'elsevier_link',
]

PIPELINE_COLUMNS = {
    'metadata_status': 'pending',
    'text_download_status': 'pending',
    'pdf_download_status': 'pending',
    'text_scrape_status': 'pending',
    'image_scrape_status': 'pending',
    'store_status': 'pending',
    'text_path': '',
    'pdf_path': '',
    'image_dir': '',
    'num_images': 0,
    'num_text_materials': 0,
    'num_image_materials': 0,
    'last_error': '',
}


def _has_value(value) -> bool:
    """Return whether a DataFrame cell contains a meaningful non-empty value."""
    return not pd.isna(value) and str(value).strip() != ''


def _clean_doi(value):
    """Normalize DOI-like values for reliable duplicate matching."""
    if not _has_value(value):
        return ''
    doi = str(value).strip()
    if doi.lower().startswith('doi:'):
        doi = doi[4:]
    if doi.lower().startswith('https://doi.org/'):
        doi = doi[16:]
    return doi.strip().rstrip('.').lower()


def _title_key(value):
    """Create a case-insensitive comparable key from a paper title."""
    if not _has_value(value):
        return ''
    return re.sub(r'\W+', ' ', str(value).lower()).strip()


def _year(value):
    """Extract a four-digit year from a date-like value."""
    if not _has_value(value):
        return ''
    match = re.search(r'\d{4}', str(value))
    return match.group(0) if match else ''


def ensure_pipeline_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all pipeline status/path/count columns present."""
    df = df.copy()
    for column, default in PIPELINE_COLUMNS.items():
        if column not in df.columns:
            df[column] = default
    string_columns = [
        'metadata_status',
        'text_download_status',
        'pdf_download_status',
        'text_scrape_status',
        'image_scrape_status',
        'store_status',
        'text_path',
        'pdf_path',
        'image_dir',
        'last_error',
    ]
    for column in string_columns:
        df[column] = df[column].fillna('').astype('object')
        if column.endswith('_status'):
            df.loc[df[column] == '', column] = 'pending'
    for column in ['num_images', 'num_text_materials', 'num_image_materials']:
        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype('int64')
    return df


def normalize_paper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with only public paper columns plus pipeline columns."""
    df = ensure_pipeline_columns(df)
    for column in PAPER_COLUMNS:
        if column not in df.columns:
            df[column] = ''
    ordered = PAPER_COLUMNS + [column for column in PIPELINE_COLUMNS if column not in PAPER_COLUMNS]
    return df[ordered].copy()


def _merge_sources(current, incoming):
    """Combine semicolon-separated source names while preserving first-seen order."""
    values = []
    for value in [current, incoming]:
        if not _has_value(value):
            continue
        values.extend(part.strip() for part in str(value).split(';') if part.strip())
    return ';'.join(dict.fromkeys(values))


def _merge_row(target: pd.DataFrame, index, row: pd.Series):
    """Merge non-empty values from ``row`` into an existing paper row."""
    for column, value in row.items():
        if column not in target.columns:
            target[column] = ''
        if not _has_value(value):
            continue
        current = target.at[index, column]
        if column == 'sources':
            target.at[index, column] = _merge_sources(current, value)
        elif column.endswith('_status'):
            if not _has_value(current) or current == 'pending':
                target.at[index, column] = value
        elif column == 'last_error':
            if _has_value(value) and not _has_value(current):
                target.at[index, column] = value
        elif not _has_value(current):
            target.at[index, column] = value
    return target


def _matching_existing_index(papers_df: pd.DataFrame, row: pd.Series):
    """Find the existing row that matches ``row`` by DOI, IDs, or title/year."""
    doi = _clean_doi(row.get('doi'))
    if doi and 'doi' in papers_df.columns:
        normalized = papers_df['doi'].map(_clean_doi)
        matches = papers_df.index[normalized == doi].tolist()
        if matches:
            return matches[0]

    for column in ['paper_id', 'core_id']:
        value = row.get(column)
        if _has_value(value) and column in papers_df.columns:
            matches = papers_df.index[papers_df[column].astype(str) == str(value)].tolist()
            if matches:
                return matches[0]

    title = _title_key(row.get('title'))
    year = _year(row.get('publication_date'))
    if title and year and {'title', 'publication_date'}.issubset(papers_df.columns):
        matches = papers_df.index[
            (papers_df['title'].map(_title_key) == title)
            & (papers_df['publication_date'].map(_year) == year)
            ].tolist()
        if matches:
            return matches[0]
    return None


def merge_paper_rows(existing: pd.DataFrame, incoming: pd.DataFrame):
    """Merge incoming paper records into an existing papers table.

    Returns the normalized merged DataFrame plus counts of added and updated
    rows.
    """
    papers = normalize_paper_columns(existing if existing is not None else pd.DataFrame())
    incoming = normalize_paper_columns(incoming if incoming is not None else pd.DataFrame())
    added = 0
    updated = 0
    for _, row in incoming.iterrows():
        match_index = _matching_existing_index(papers, row) if not papers.empty else None
        if match_index is None:
            papers = pd.concat([papers, pd.DataFrame([row])], ignore_index=True)
            added += 1
        else:
            papers = _merge_row(papers, match_index, row)
            updated += 1
    return normalize_paper_columns(papers), added, updated


def read_papers(path: str) -> pd.DataFrame:
    """Read a papers CSV and normalize it to the current schema."""
    return normalize_paper_columns(pd.read_csv(path, index_col=0))


def write_papers(df: pd.DataFrame, path: str):
    """Write a papers DataFrame using the current normalized CSV schema."""
    normalize_paper_columns(df).to_csv(path)


def set_status(df: pd.DataFrame,
               index,
               column: str,
               status: str,
               error: str | None = None):
    """Update a pipeline status column and optionally record or clear an error."""
    if column not in PIPELINE_COLUMNS:
        raise KeyError(f'Unknown pipeline status column: {column}')
    df.loc[index, column] = status
    if error:
        df.loc[index, 'last_error'] = error
    elif 'last_error' in df.columns and status in {'succeeded', 'stored'}:
        df.loc[index, 'last_error'] = ''


def existing_path(value) -> str | None:
    """Return ``value`` when it points to an existing file, otherwise ``None``."""
    if isinstance(value, str) and value and os.path.isfile(value):
        return value
    return None
