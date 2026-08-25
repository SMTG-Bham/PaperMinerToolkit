"""Extract and enrich paper metadata from DOI text and Crossref.

The helpers here read DOI values from raw text/PDFs, normalize them, and use
Crossref to populate the public paper metadata fields stored in the corpus.
"""

from __future__ import annotations

import html
import re
import requests
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import Any, Literal
from urllib.parse import unquote

from pypdf import PdfReader

from paperminer.corpus.documents import read_pdf_text

MODERN_DOI_PATTERN = r'10\.\d{4,9}/(?:[-._;()/:A-Z0-9]|%[0-9A-F]{2})+'
# These two older publisher formats contain characters excluded by the modern
# Crossref pattern. Keep them narrow to avoid treating arbitrary URLs as DOIs.
LEGACY_DOI_PATTERNS = (
    r'10\.1002/\S+',
    r'10\.1207/[\w\d]+&\d+_\d+',
)
DOI_PRESENTATION_PREFIX = r'(?:(?:https?://(?:dx\.)?doi\.org/)|(?:doi\s*:\s*))?'
DOI_PATTERN = re.compile(
    DOI_PRESENTATION_PREFIX + '(?:' + '|'.join((*LEGACY_DOI_PATTERNS, MODERN_DOI_PATTERN)) + ')',
    re.IGNORECASE,
)
DOI_URL_PREFIX = re.compile(r'^https?://(?:dx\.)?doi\.org/', re.IGNORECASE)
DOI_LABEL_PREFIX = re.compile(r'^doi\s*:\s*', re.IGNORECASE)
TRAILING_SENTENCE_PUNCTUATION = '.,;:'
TRAILING_DELIMITERS = {')': '(', ']': '[', '}': '{', '>': '<'}
INVISIBLE_CHARACTERS = str.maketrans('', '', '\u200b\u200c\u200d\u2060\ufeff')
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


def clean_doi(value: object) -> str:
    """Canonicalize a DOI presentation value.

    Parameters
    ----------
    value : object
        Plain DOI, labelled DOI, or DOI resolver URL.

    Returns
    -------
    str
        Case-folded DOI without presentation prefixes or trailing sentence
        punctuation.
    """
    doi = unicodedata.normalize('NFKC', html.unescape(str(value or ''))).translate(UNICODE_PUNCTUATION)
    doi = doi.translate(INVISIBLE_CHARACTERS).strip()

    url_prefix = DOI_URL_PREFIX.match(doi)
    if url_prefix:
        doi = unquote(doi[url_prefix.end():])
        doi = re.split(r'[?#]', doi, maxsplit=1)[0]
    else:
        doi = DOI_LABEL_PREFIX.sub('', doi, count=1)
    doi = doi.strip()

    while doi:
        final_character = doi[-1]
        if final_character in TRAILING_SENTENCE_PUNCTUATION or final_character in {'"', "'"}:
            doi = doi[:-1]
            continue
        opening_character = TRAILING_DELIMITERS.get(final_character)
        if opening_character and doi.count(final_character) > doi.count(opening_character):
            doi = doi[:-1]
            continue
        break
    return doi.casefold()


def _normalize_metadata_text(value: object) -> str:
    """Normalize text from a metadata provider.

    Parameters
    ----------
    value : object
        Metadata value to convert to plain text. ``None`` represents missing
        text.

    Returns
    -------
    str
        Whitespace-normalized text with HTML removed and Unicode punctuation
        and super/subscripts flattened.
    """
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


def _normalize_punctuation_char(character: str) -> str:
    """Approximate a Unicode punctuation character in ASCII.

    Parameters
    ----------
    character : str
        Single character to normalize.

    Returns
    -------
    str
        ASCII approximation when available, an empty string for unsupported
        punctuation, or the original non-punctuation character.
    """
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


def _is_article_doi(doi: str) -> bool:
    """Check whether a DOI candidate appears to identify an article.

    Parameters
    ----------
    doi : str
        DOI candidate.

    Returns
    -------
    bool
        ``True`` for a non-empty DOI that does not identify ISSN metadata.
    """
    return bool(doi) and '(issn)' not in doi.lower()


def _rank_doi_candidates(candidates: Iterable[str]) -> list[str]:
    """Rank canonical DOI candidates by frequency.

    Parameters
    ----------
    candidates : Iterable[str]
        DOI presentation values to clean and rank.

    Returns
    -------
    list[str]
        Unique article DOI candidates ordered by descending frequency, with
        first appearance breaking ties.
    """
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


def extract_dois_from_text(text: str) -> list[str]:
    """Extract DOI candidates from text.

    Parameters
    ----------
    text : str
        Text that may contain DOI values and common PDF extraction artifacts.

    Returns
    -------
    list[str]
        Canonical DOI candidates ranked by frequency.
    """
    normalized = unicodedata.normalize('NFKC', html.unescape(str(text or ''))).translate(UNICODE_PUNCTUATION)
    normalized = normalized.translate(INVISIBLE_CHARACTERS)
    # A soft hyphen marks a word-wrap rather than part of the identifier. PDF
    # extractors may retain both it and the following line break.
    normalized = re.sub(r'\u00ad[ \t]*(?:\r?\n)?[ \t]*', '', normalized)
    # Joining immediately after the mandatory prefix slash is unambiguous. Do
    # not join arbitrary DOI-looking lines: a DOI can legitimately end there.
    normalized = re.sub(r'(10\.\d{4,9}/)[ \t]*\r?\n[ \t]*', r'\1', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s+', ' ', normalized)
    return _rank_doi_candidates(match.group(0) for match in DOI_PATTERN.finditer(normalized))


def extract_doi_from_text(text: str) -> str | None:
    """Extract the most likely DOI from text.

    Parameters
    ----------
    text : str
        Text that may contain a DOI.

    Returns
    -------
    str or None
        Highest-ranked canonical DOI, or ``None`` when no candidate exists.
    """
    candidates = extract_dois_from_text(text)
    if not candidates:
        return None
    return candidates[0]


def extract_dois_from_pdf_metadata(pdf_path: str | PathLike[str]) -> list[str]:
    """Extract DOI candidates from embedded PDF metadata.

    Parameters
    ----------
    pdf_path : str or os.PathLike[str]
        Path to the PDF file.

    Returns
    -------
    list[str]
        Unique DOI candidates in metadata field order.
    """
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


def extract_doi_from_pdf_metadata(pdf_path: str | PathLike[str]) -> str | None:
    """Extract the best DOI from embedded PDF metadata.

    Parameters
    ----------
    pdf_path : str or os.PathLike[str]
        Path to the PDF file.

    Returns
    -------
    str or None
        First metadata DOI candidate, or ``None`` when no candidate exists.
    """
    candidates = extract_dois_from_pdf_metadata(pdf_path)
    if not candidates:
        return None
    return candidates[0]


def doi_candidates_from_pdf(pdf_path: str | PathLike[str]) -> list[str]:
    """Collect metadata-first DOI candidates from a PDF.

    Parameters
    ----------
    pdf_path : str or os.PathLike[str]
        Path to the PDF file.

    Returns
    -------
    list[str]
        Unique metadata candidates followed by ranked page-text candidates.
    """
    candidates = extract_dois_from_pdf_metadata(pdf_path)
    text_candidates = extract_dois_from_text(read_pdf_text(pdf_path))
    candidates.extend(candidate for candidate in text_candidates if candidate not in candidates)
    return candidates


def extract_doi_from_pdf(pdf_path: str | PathLike[str]) -> str | None:
    """Extract the most likely DOI from a PDF.

    Parameters
    ----------
    pdf_path : str or os.PathLike[str]
        Path to the PDF file.

    Returns
    -------
    str or None
        First metadata or page-text candidate, or ``None`` when none exists.
    """
    candidates = doi_candidates_from_pdf(pdf_path)
    if not candidates:
        return None
    return candidates[0]


def _date_from_parts(parts: Sequence[Sequence[int]] | None) -> str:
    """Format a Crossref ``date-parts`` value.

    Parameters
    ----------
    parts : Sequence[Sequence[int]] or None
        Crossref date components, with the first nested sequence representing
        the date.

    Returns
    -------
    str
        Date formatted as ``YYYY``, ``YYYY-MM``, or ``YYYY-MM-DD``; an empty
        string when no date is available.
    """
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


def _published_date(message: Mapping[str, Any]) -> str:
    """Select the best publication date from a Crossref message.

    Parameters
    ----------
    message : Mapping[str, Any]
        Crossref work message containing optional date fields.

    Returns
    -------
    str
        First available publication date in preference order, or an empty
        string.
    """
    for key in ['published-print', 'published-online', 'published', 'issued', 'created']:
        date = _date_from_parts(message.get(key, {}).get('date-parts'))
        if date:
            return date
    return ''


def _crossref_issn(message: Mapping[str, Any]) -> str:
    """Join a Crossref work's ISSNs, print issues first.

    Parameters
    ----------
    message : Mapping[str, Any]
        Crossref work message.

    Returns
    -------
    str
        Semicolon-separated ISSNs, or an empty string when none are deposited.
    """
    typed = message.get('issn-type') or []
    ordered = [entry.get('value') for entry in typed if (entry or {}).get('type') == 'print']
    ordered.extend(entry.get('value') for entry in typed if (entry or {}).get('type') != 'print')
    if not typed:
        ordered = list(message.get('ISSN') or [])
    return ';'.join(dict.fromkeys(str(value).strip() for value in ordered if value))


def _crossref_pages(message: Mapping[str, Any]) -> str:
    """Read a Crossref work's page range or article number.

    Journals that number articles rather than pages, such as those published by
    the American Physical Society, deposit ``article-number`` and no ``page``,
    so the article number is used as the locator when no range is present.

    Parameters
    ----------
    message : Mapping[str, Any]
        Crossref work message.

    Returns
    -------
    str
        Page range or article number as deposited, or an empty string.
    """
    return str(message.get('page') or message.get('article-number') or '').strip()


def crossref_fields(message: Mapping[str, Any], doi: str = '') -> dict[str, Any]:
    """Map a Crossref work message onto corpus column names.

    Every returned key is either a corpus column or the explicitly named raw
    payload, so no fetched value is silently discarded downstream.

    Parameters
    ----------
    message : Mapping[str, Any]
        Crossref work message.
    doi : str, default=''
        DOI used when the message does not carry one.

    Returns
    -------
    dict[str, Any]
        Corpus metadata fields plus the raw ``crossref_message``.
    """
    return {
        'doi': message.get('DOI', doi),
        'publication_date': _published_date(message),
        'title': _normalize_metadata_text((message.get('title') or [''])[0]),
        'journal': _normalize_metadata_text((message.get('container-title') or [''])[0]),
        'publisher': _normalize_metadata_text(message.get('publisher', '')),
        'work_type': str(message.get('type') or ''),
        'volume': str(message.get('volume') or ''),
        'issue': str(message.get('issue') or ''),
        'pages': _crossref_pages(message),
        'issn': _crossref_issn(message),
        'language': str(message.get('language') or ''),
        'crossref_message': dict(message),
    }


def get_crossref_metadata(doi: str, timeout: int = 30, email: str | None = None) -> dict[str, Any]:
    """Fetch normalized paper metadata from Crossref.

    The contact address is sent both as the ``mailto`` query parameter and in
    the user agent, which is what Crossref asks automated clients to do.

    Parameters
    ----------
    doi : str
        DOI to look up.
    timeout : int, default=30
        HTTP request timeout in seconds.
    email : str or None, optional
        Contact email. Defaults to the stored ``crossref_email`` setting.

    Returns
    -------
    dict[str, Any]
        Corpus metadata columns plus the raw ``crossref_message`` payload.

    Raises
    ------
    ValueError
        If no contact email is available.
    RuntimeError
        If the Crossref request exhausts its retries.
    """
    from paperminer.providers.crossref import work_by_doi

    work = work_by_doi(doi, email=email or None, timeout=timeout)
    return crossref_fields(work or {}, doi)


def metadata_from_pdf(
    pdf_path: str | PathLike[str],
    use_crossref: bool = True,
) -> tuple[dict[str, Any], Literal['imported', 'doi_found', 'enriched'], str]:
    """Extract and optionally enrich DOI metadata from a PDF.

    Parameters
    ----------
    pdf_path : str or os.PathLike[str]
        Path to the PDF file.
    use_crossref : bool, default=True
        Whether to validate candidates and fetch bibliographic metadata from
        Crossref.

    Returns
    -------
    metadata : dict[str, Any]
        Extracted metadata, empty when the PDF cannot be read or has no DOI.
    status : {'imported', 'doi_found', 'enriched'}
        Furthest successful metadata stage.
    error : str
        Human-readable failure details, or an empty string after success.
    """
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
