"""Extract and enrich paper metadata from DOI text and Crossref.

The helpers here read DOI values from raw text/PDFs, normalize them, and use
Crossref to populate the public paper metadata fields used in ``papers.csv``.
"""

import html
import re
import requests
import unicodedata
from collections import Counter
from urllib.parse import quote

from pypdf import PdfReader

from paperscraper.documents import read_pdf_text

DOI_PATTERN = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
TRAILING_PUNCTUATION = '.),;:]}'
UNICODE_PUNCTUATION = str.maketrans({
    '\u2010': '-',
    '\u2011': '-',
    '\u2012': '-',
    '\u2013': '-',
    '\u2014': '-',
    '\u2015': '-',
    '\u2212': '-',
    '\u2018': "'",
    '\u2019': "'",
    '\u201a': "'",
    '\u201b': "'",
    '\u201c': '"',
    '\u201d': '"',
    '\u201e': '"',
    '\u201f': '"',
    '\u2026': '...',
})


def clean_doi(value: str):
    """Trim common trailing punctuation from a DOI string."""
    doi = value.strip().strip(TRAILING_PUNCTUATION)
    return doi.rstrip('.')


def normalize_metadata_text(value):
    """Normalize metadata text by flattening HTML, Unicode punctuation, and super/subscripts."""
    if value is None:
        return ''
    text = html.unescape(str(value))
    text = re.sub(r'</?(?:sub|sup)>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(UNICODE_PUNCTUATION)
    text = ''.join(_normalize_punctuation_char(character) for character in text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _normalize_punctuation_char(character):
    """Return an ASCII approximation for remaining Unicode punctuation."""
    if ord(character) < 128:
        return character
    category = unicodedata.category(character)
    name = unicodedata.name(character, '').lower()
    if category == 'Pd' or 'dash' in name or 'hyphen' in name or 'minus' in name:
        return '-'
    if 'apostrophe' in name or 'single quotation' in name:
        return "'"
    if 'quotation mark' in name or 'double quotation' in name:
        return '"'
    if category.startswith('P'):
        return unicodedata.normalize('NFKD', character).encode('ascii', 'ignore').decode('ascii')
    return character


def _is_article_doi(doi: str):
    """Return whether a DOI candidate looks like an article DOI rather than journal metadata."""
    return bool(doi) and '(issn)' not in doi.lower()


def _rank_doi_candidates(candidates):
    """Return DOI candidates ranked by frequency while preserving first-seen order for ties."""
    cleaned = []
    for candidate in candidates:
        doi = clean_doi(candidate)
        if _is_article_doi(doi):
            cleaned.append(doi)
    counts = Counter(cleaned)
    first_seen = {}
    for index, doi in enumerate(cleaned):
        first_seen.setdefault(doi, index)
    return sorted(counts, key=lambda doi: (-counts[doi], first_seen[doi]))


def extract_dois_from_text(text: str):
    """Find DOI-like values in a block of text ranked by frequency."""
    normalized = re.sub(r'\s+', ' ', text or '')
    return _rank_doi_candidates(match.group(0) for match in DOI_PATTERN.finditer(normalized))


def extract_doi_from_text(text: str):
    """Find the most likely DOI-like value in a block of text."""
    candidates = extract_dois_from_text(text)
    if not candidates:
        return None
    return candidates[0]


def extract_dois_from_pdf_metadata(pdf_path: str):
    """Extract DOI candidates from embedded PDF metadata fields when available."""
    reader = PdfReader(pdf_path)
    metadata = reader.metadata or {}
    candidates = []
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
        candidates.extend(extract_dois_from_text(str(value)))
    for key, value in metadata.items():
        if 'journaldoi' in str(key).lower():
            continue
        candidates.extend(extract_dois_from_text(str(value)))
    return list(dict.fromkeys(candidates))


def extract_doi_from_pdf_metadata(pdf_path: str):
    """Extract the best DOI from embedded PDF metadata fields when available."""
    candidates = extract_dois_from_pdf_metadata(pdf_path)
    if not candidates:
        return None
    return candidates[0]


def doi_candidates_from_pdf(pdf_path: str):
    """Return metadata-first DOI candidates, followed by ranked text candidates."""
    candidates = extract_dois_from_pdf_metadata(pdf_path)
    text_candidates = extract_dois_from_text(read_pdf_text(pdf_path))
    candidates.extend(candidate for candidate in text_candidates if candidate not in candidates)
    return candidates


def extract_doi_from_pdf(pdf_path: str):
    """Extract a DOI from PDF metadata first, then fall back to page text."""
    candidates = doi_candidates_from_pdf(pdf_path)
    if not candidates:
        return None
    return candidates[0]


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
        'title': normalize_metadata_text((message.get('title') or [''])[0]),
        'journal': normalize_metadata_text((message.get('container-title') or [''])[0]),
        'crossref_type': message.get('type', ''),
        'crossref_publisher': normalize_metadata_text(message.get('publisher', '')),
    }


def metadata_from_pdf(pdf_path: str, use_crossref: bool = True):
    """Extract DOI metadata from a PDF and optionally enrich it with Crossref."""
    try:
        doi_candidates = doi_candidates_from_pdf(pdf_path)
    except Exception as e:
        return {}, 'imported', f'Could not read PDF metadata text: {e}'
    if not doi_candidates:
        return {}, 'imported', 'No DOI found in PDF metadata or text.'
    doi = doi_candidates[0]
    metadata = {'doi': doi}
    if not use_crossref:
        return metadata, 'doi_found', ''
    last_error = None
    for candidate in doi_candidates:
        try:
            enriched = get_crossref_metadata(candidate)
        except requests.RequestException as e:
            last_error = e
            continue
        metadata = {'doi': candidate}
        metadata.update(enriched)
        return metadata, 'enriched', ''
    return metadata, 'doi_found', f'Crossref lookup failed for DOI candidates {", ".join(doi_candidates)}: {last_error}'
