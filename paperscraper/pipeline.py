import os

import pandas as pd

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


def ensure_pipeline_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def read_papers(path: str) -> pd.DataFrame:
    return ensure_pipeline_columns(pd.read_csv(path, index_col=0))


def write_papers(df: pd.DataFrame, path: str):
    ensure_pipeline_columns(df).to_csv(path)


def set_status(df: pd.DataFrame, index, column: str, status: str, error: str | None = None):
    if column not in PIPELINE_COLUMNS:
        raise KeyError(f'Unknown pipeline status column: {column}')
    df.loc[index, column] = status
    if error:
        df.loc[index, 'last_error'] = error
    elif 'last_error' in df.columns and status in {'succeeded', 'stored'}:
        df.loc[index, 'last_error'] = ''


def existing_path(value) -> str | None:
    if isinstance(value, str) and value and os.path.isfile(value):
        return value
    return None
