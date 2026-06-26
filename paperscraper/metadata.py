"""Extract and enrich paper metadata from DOI text and Crossref.

The helpers here read DOI values from raw text/PDFs, normalize them, and use
Crossref to populate the public paper metadata fields used in ``papers.csv``.
"""

import re
import requests
from urllib.parse import quote

from pypdf import PdfReader

from paperscraper.documents import read_pdf_text

DOI_PATTERN = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
TRAILING_PUNCTUATION = '.),;:]}'


def clean_doi(value: str):
    """Trim common trailing punctuation from a DOI string."""
    doi = value.strip().strip(TRAILING_PUNCTUATION)
    return doi.rstrip('.')


def extract_doi_from_text(text: str):
    """Find and clean the first DOI-like value in a block of text."""
    normalized = re.sub(r'\s+', ' ', text or '')
    match = DOI_PATTERN.search(normalized)
    if not match:
        return None
    return clean_doi(match.group(0))


def extract_doi_from_pdf_metadata(pdf_path: str):
    """Extract a DOI from embedded PDF metadata fields when available."""
    reader = PdfReader(pdf_path)
    metadata = reader.metadata or {}
    preferred_fields = [
        '/WPS-ARTICLEDOI',
        '/prism:doi',
        '/doi',
        '/DOI',
        '/dc:identifier',
        '/Subject',
    ]
    for field in preferred_fields:
        value = metadata.get(field)
        if not value:
            continue
        doi = extract_doi_from_text(str(value))
        if doi and '(issn)' not in doi.lower():
            return doi
    for key, value in metadata.items():
        if 'journaldoi' in str(key).lower():
            continue
        doi = extract_doi_from_text(str(value))
        if doi and '(issn)' not in doi.lower():
            return doi
    return None


def extract_doi_from_pdf(pdf_path: str):
    """Extract a DOI from PDF metadata first, then fall back to page text."""
    doi = extract_doi_from_pdf_metadata(pdf_path)
    if doi:
        return doi
    return extract_doi_from_text(read_pdf_text(pdf_path))


def _date_from_parts(parts):
    """Format Crossref date-parts arrays as ISO-like date strings."""
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
    """Pick the best available publication date from a Crossref message."""
    for key in ['published-print', 'published-online', 'published', 'issued', 'created']:
        date = _date_from_parts(message.get(key, {}).get('date-parts'))
        if date:
            return date
    return ''


def get_crossref_metadata(doi: str, timeout: int = 30):
    """Fetch normalized paper metadata for a DOI from Crossref."""
    url = f'https://api.crossref.org/works/{quote(doi, safe="")}'
    headers = {'User-Agent': 'PaperScraper/0.0.1 (https://github.com/SMTG-Bham/PaperScraper)'}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    message = response.json().get('message', {})
    return {
        'doi': message.get('DOI', doi),
        'publication_date': _published_date(message),
        'title': (message.get('title') or [''])[0],
        'journal': (message.get('container-title') or [''])[0],
        'crossref_type': message.get('type', ''),
        'crossref_publisher': message.get('publisher', ''),
    }


def metadata_from_pdf(pdf_path: str, use_crossref: bool = True):
    """Extract DOI metadata from a PDF and optionally enrich it with Crossref."""
    try:
        doi = extract_doi_from_pdf(pdf_path)
    except Exception as e:
        return {}, 'imported', f'Could not read PDF metadata text: {e}'
    if not doi:
        return {}, 'imported', 'No DOI found in PDF metadata or text.'
    metadata = {'doi': doi}
    if not use_crossref:
        return metadata, 'doi_found', ''
    try:
        metadata.update(get_crossref_metadata(doi))
    except requests.RequestException as e:
        return metadata, 'doi_found', f'Crossref lookup failed for DOI {doi}: {e}'
    return metadata, 'enriched', ''
