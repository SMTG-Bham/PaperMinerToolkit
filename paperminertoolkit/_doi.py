"""Canonical DOI presentation handling shared by corpus storage and metadata."""

from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import unquote

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
