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
    'last_error': '',
}


def ensure_pipeline_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column, default in PIPELINE_COLUMNS.items():
        if column not in df.columns:
            df[column] = default
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
