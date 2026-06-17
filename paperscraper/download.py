from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc
from paperscraper.pipeline import ensure_pipeline_columns, set_status, write_papers
from paperscraper.settings import load_settings
from urllib.parse import quote
import json
import os
import pandas as pd
import requests
from tqdm import tqdm

DOWNLOAD_FORMATS = {'text', 'pdf', 'both'}


def _elsevier_client():
    api_key = load_settings().get('elsevier_api_key')
    if not api_key:
        raise ValueError('Elsevier API key is not configured. Run ps_elsevier_key first.')
    return ElsClient(api_key)


def retrieve_document(uri):
    os.makedirs('data', exist_ok=True)
    for file in os.listdir('data'):
        os.remove(os.path.join('data', file))
    doi_doc = FullDoc(uri=uri)
    if doi_doc.read(_elsevier_client()):
        doi_doc.write()
    else:
        print('Read document failed.')


def json_to_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    text = doc['originalText']
    if type(text) == dict:
        return 'failed'
    return text


def elsevier_string_formatter(text: str):
    if text.count('Acknowledgements') == 2:
        text = text.split('Acknowledgements')[1]
    elif text.count('References') == 2:
        text = text.split('References')[1]
    if 'amazonaws.com/' in text:
        text = text.split('amazonaws.com/')[-1]
        text = text[text.find(' '):]
    return text


def _full_text_uri(paper):
    link = paper.get('link')
    if not isinstance(link, str) or 'full-text' not in link:
        return None
    parts = link.split("'")
    if len(parts) >= 2:
        return parts[-2]
    return None


def _download_text(paper, filepath):
    uri = _full_text_uri(paper)
    if not uri:
        return False
    retrieve_document(uri)
    files = os.listdir('data')
    if len(files) < 1:
        return False
    temp_file = os.path.join('data', files[0])
    text = json_to_text(temp_file)
    if text == 'failed':
        return False
    formatted_text = elsevier_string_formatter(text)
    with open(filepath, 'w', encoding='utf-8') as out_file:
        out_file.write(formatted_text)
    return True


def _pdf_urls(paper):
    urls = []
    doi = paper.get('prism:doi')
    if pd.notna(doi) and str(doi).strip():
        urls.append(f'https://api.elsevier.com/content/article/doi/{quote(str(doi), safe="")}')
    pii = paper.get('pii') or paper.get('prism:pii')
    if pd.notna(pii) and str(pii).strip():
        urls.append(f'https://api.elsevier.com/content/article/pii/{quote(str(pii), safe="")}')
    uri = _full_text_uri(paper)
    if uri:
        urls.append(uri)
    return urls


def _download_pdf(paper, filepath):
    api_key = load_settings().get('elsevier_api_key')
    if not api_key:
        raise ValueError('Elsevier API key is not configured. Run ps_elsevier_key first.')
    headers = {
        'X-ELS-APIKey': api_key,
        'Accept': 'application/pdf',
    }
    params = {'httpAccept': 'application/pdf'}
    last_error = None
    for url in _pdf_urls(paper):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)
            if response.status_code >= 400:
                last_error = f'{response.status_code} from {url}'
                continue
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type or response.content.startswith(b'%PDF'):
                with open(filepath, 'wb') as out_file:
                    out_file.write(response.content)
                return True
            last_error = f'non-PDF response from {url}'
        except requests.RequestException as e:
            last_error = str(e)
    if last_error:
        print(f'PDF download failed for {paper.get("dc:identifier")}: {last_error}')
    return False


def elsevier_downloader(papers_path='papers.csv', download_dir='papers', download_format='text'):
    download_format = download_format.lower()
    if download_format not in DOWNLOAD_FORMATS:
        raise ValueError(f'download_format must be one of: {", ".join(sorted(DOWNLOAD_FORMATS))}')
    os.makedirs(download_dir, exist_ok=True)
    papers = ensure_pipeline_columns(pd.read_csv(papers_path, index_col=0))
    if 'link' not in papers.columns:
        raise RuntimeError(f'{papers_path} does not contain a link column. Search returned no downloadable Elsevier results.')
    elsevier_papers = papers[papers['link'].astype(str).str.contains('full-text', na=False)]
    if elsevier_papers.empty:
        raise RuntimeError(f'{papers_path} does not contain any Elsevier full-text links to download.')
    with tqdm(total=len(elsevier_papers['link']), desc='Downloading Papers', colour='#A020F0') as pbar:
        for index, paper in elsevier_papers.iterrows():
            filename = paper['dc:identifier'].split(':')[-1]
            if download_format in {'text', 'both'}:
                text_filepath = os.path.join(download_dir, f'{filename}.txt')
                try:
                    if os.path.isfile(text_filepath) or _download_text(paper, text_filepath):
                        papers.loc[index, 'text_path'] = text_filepath
                        set_status(papers, index, 'text_download_status', 'succeeded')
                    else:
                        set_status(papers, index, 'text_download_status', 'failed', 'Elsevier text download failed')
                except Exception as e:
                    set_status(papers, index, 'text_download_status', 'failed', str(e))
            if download_format in {'pdf', 'both'}:
                pdf_filepath = os.path.join(download_dir, f'{filename}.pdf')
                try:
                    if os.path.isfile(pdf_filepath) or _download_pdf(paper, pdf_filepath):
                        papers.loc[index, 'pdf_path'] = pdf_filepath
                        set_status(papers, index, 'pdf_download_status', 'succeeded')
                    else:
                        set_status(papers, index, 'pdf_download_status', 'failed', 'Elsevier PDF download failed')
                except Exception as e:
                    set_status(papers, index, 'pdf_download_status', 'failed', str(e))
            (papers)
            write_papers(papers, papers_path)
            pbar.update(1)
