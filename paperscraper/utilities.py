import pandas as pd

from paperscraper.pipeline import PIPELINE_COLUMNS, ensure_pipeline_columns, write_papers


def reset(papers_path: str = 'papers.csv'):
    papers_df = ensure_pipeline_columns(pd.read_csv(papers_path, index_col=0))
    for column, default in PIPELINE_COLUMNS.items():
        papers_df[column] = default
    papers_df['metadata_status'] = 'retrieved'
    write_papers(papers_df, papers_path)


def status(papers_path: str = 'papers.csv'):
    papers_df = ensure_pipeline_columns(pd.read_csv(papers_path, index_col=0))
    print('\nPaperScraper Progress Summary')
    print('---------------------------')
    print(f'Total papers: {len(papers_df)}')
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
        count = int((papers_df[column] == value).sum()) if column in papers_df else 0
        print(f'{label}: {count}')
    text_materials = int(papers_df['num_text_materials'].sum()) if 'num_text_materials' in papers_df else 0
    image_materials = int(papers_df['num_image_materials'].sum()) if 'num_image_materials' in papers_df else 0
    print(f'Text material rows extracted: {text_materials}')
    print(f'Image material rows extracted: {image_materials}')
    print('---------------------------\n')


def sort(path: str = 'papers.csv', field: str = 'metadata_status', ascending: bool = True):
    papers_df = ensure_pipeline_columns(pd.read_csv(path, index_col=0))
    papers_df.sort_values(by=field, ascending=ascending, inplace=True)
    papers_df.reset_index(drop=True, inplace=True)
    write_papers(papers_df, path)


def shuffle(path: str = 'papers.csv'):
    papers_df = ensure_pipeline_columns(pd.read_csv(path, index_col=0))
    papers_df = papers_df.sample(frac=1).reset_index(drop=True)
    write_papers(papers_df, path)
