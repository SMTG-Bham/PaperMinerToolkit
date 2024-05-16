import pandas as pd

def reset(papers_path: str='papers.csv'):
    papers_df = pd.read_csv(papers_path, index_col=0)
    papers_df['status'] = 'retrieved'
    papers_df.to_csv(papers_path)

def status(papers_path: str='papers.csv'):
    papers_df = pd.read_csv(papers_path, index_col=0)
    total = str(len(papers_df))
    print(f'Total: {total}')
    retrieved_count = str(papers_df['status'].value_counts()['retrieved'])
    print(f'Retrieved: {retrieved_count}')
    scraped_count = str(papers_df['status'].value_counts()['scraped'])
    print(f'Scraped: {scraped_count}')
    stored_count = str(papers_df['status'].value_counts()['stored'])
    print(f'Stored: {stored_count}')