import pandas as pd

def reset(papers_path: str='papers.csv'):
    papers_df = pd.read_csv(papers_path, index_col=0)
    papers_df['status'] = 'retrieved'
    papers_df.to_csv(papers_path)

def status(papers_path: str='papers.csv'):
    papers_df = pd.read_csv(papers_path, index_col=0)
    print('\nPaperScraper Progress Summary')
    print('---------------------------')
    total = str(len(papers_df))
    print(f'Total: {total}')
    status_counts = {'Retrieved':0, 'Scraped':0, 'Stored':0}
    for status in status_counts.keys():
        try:
            count = str(papers_df['status'].value_counts()[status.lower()])
        except KeyError:
            count = 0
        finally:
            status_counts[status] = count
            print(f'{status}: {status_counts[status]}')
    print('---------------------------\n')

def sort(path: str='papers.csv', field: str='status', ascending: bool=True):
    papers_df = pd.read_csv(path, index_col=0)
    papers_df.sort_values(by=field, ascending=ascending, inplace=True)
    papers_df.reset_index(drop=True, inplace=True)
    papers_df.to_csv(path)

def shuffle(path: str='papers.csv'):
    papers_df = pd.read_csv(path, index_col=0)
    papers_df = papers_df.sample(frac=1).reset_index(drop=True)
    papers_df.to_csv(path)