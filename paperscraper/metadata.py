import re
from urllib.parse import quote

import requests

from paperscraper.documents import read_pdf_text


DOI_PATTERN = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
TRAILING_PUNCTUATION = '.),;:]}'


def clean_doi(value: str):
    doi = value.strip().strip(TRAILING_PUNCTUATION)
    return doi.rstrip('.')


def extract_doi_from_text(text: str):
    normalized = re.sub(r'\s+', ' ', text or '')
    match = DOI_PATTERN.search(normalized)
    if not match:
        return None
    return clean_doi(match.group(0))


def extract_doi_from_pdf(pdf_path: str):
    return extract_doi_from_text(read_pdf_text(pdf_path))


def _date_from_parts(parts):
    if not parts:
        return ''
    date = parts[0]
    if not date:
        return ''
    if len(date) >= 3:
        return f'{date[0]:04d}-{date[1]:02d}-{date[2]:02d}'
    if len(date) == 2:
        return f'{date[0]:04d}-{date[1]:02d}'
    return f'{date[0]:04d}'


def _published_date(message):
    for key in ['published-print', 'published-online', 'published', 'issued', 'created']:
        date = _date_from_parts(message.get(key, {}).get('date-parts'))
        if date:
            return date
    return ''


def get_crossref_metadata(doi: str, timeout: int = 30):
    url = f'https://api.crossref.org/works/{quote(doi, safe="")}'
    headers = {'User-Agent': 'PaperScraper/0.0.1 (https://github.com/SMTG-Bham/PaperScraper)'}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    message = response.json().get('message', {})
    return {
        'dc:identifier': f'doi:{message.get("DOI", doi)}',
        'prism:doi': message.get('DOI', doi),
        'prism:coverDate': _published_date(message),
        'dc:title': (message.get('title') or [''])[0],
        'prism:publicationName': (message.get('container-title') or [''])[0],
        'crossref_type': message.get('type', ''),
        'crossref_publisher': message.get('publisher', ''),
    }


def metadata_from_pdf(pdf_path: str, use_crossref: bool = True):
    try:
        doi = extract_doi_from_pdf(pdf_path)
    except Exception as e:
        return {}, 'imported', f'Could not read PDF metadata text: {e}'
    if not doi:
        return {}, 'imported', 'No DOI found in PDF text.'
    metadata = {'prism:doi': doi, 'dc:identifier': f'doi:{doi}'}
    if not use_crossref:
        return metadata, 'doi_found', ''
    try:
        metadata.update(get_crossref_metadata(doi))
    except requests.RequestException as e:
        return metadata, 'doi_found', f'Crossref lookup failed for DOI {doi}: {e}'
    return metadata, 'enriched', ''
