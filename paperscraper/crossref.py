"""Discover an author's works through Crossref and import their metadata."""

import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from paperscraper.corpus import connect, find_paper, upsert_paper, upsert_papers


CROSSREF_WORKS_URL = 'https://api.crossref.org/v1/works'
ORCID_PATTERN = re.compile(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', re.IGNORECASE)
REVIEW_COLUMNS = ['paper_id', 'doi', 'title', 'journal', 'publication_date', 'authors', 'sources']


def normalize_orcid(value: str):
    """Return a bare, validated ORCID identifier."""
    orcid = str(value or '').strip().rstrip('/').split('/')[-1].upper()
    if not ORCID_PATTERN.fullmatch(orcid):
        raise ValueError(f'Invalid ORCID: {value}')
    compact = orcid.replace('-', '')
    total = 0
    for digit in compact[:-1]:
        total = (total + int(digit)) * 2
    checksum = (12 - (total % 11)) % 11
    expected = 'X' if checksum == 10 else str(checksum)
    if compact[-1] != expected:
        raise ValueError(f'Invalid ORCID checksum: {value}')
    return orcid


def _normalize_text(value):
    """Create a comparable lower-case key from human-readable metadata."""
    value = unicodedata.normalize('NFKD', str(value or ''))
    value = ''.join(character for character in value if not unicodedata.combining(character))
    return re.sub(r'[^a-z0-9]+', ' ', value.lower()).strip()


def _given_names_match(target, candidate):
    """Match full given names exactly, allowing initials on either side."""
    target_parts = _normalize_text(target).split()
    candidate_parts = _normalize_text(candidate).split()
    if not target_parts or not candidate_parts:
        return False
    if target_parts == candidate_parts:
        return True
    if all(len(part) > 1 for part in target_parts + candidate_parts):
        return False
    return ''.join(part[0] for part in target_parts) == ''.join(part[0] for part in candidate_parts)


def _matching_authors(work, author_name):
    """Return Crossref author records matching a supplied human name."""
    parts = _normalize_text(author_name).split()
    if len(parts) < 2:
        raise ValueError('Author name must include given and family names.')
    target_given = ' '.join(parts[:-1])
    target_family = parts[-1]
    return [
        author
        for author in work.get('author') or []
        if _normalize_text(author.get('family')) == target_family
        and _given_names_match(target_given, author.get('given'))
    ]


def _author_orcid(author):
    """Return a normalized author ORCID, ignoring malformed deposited values."""
    try:
        return normalize_orcid(author.get('ORCID'))
    except ValueError:
        return ''


def work_matches_author(work, author_name=None, orcid=None, affiliation=None):
    """Return whether a Crossref work contains the requested author identity."""
    authors = work.get('author') or []
    if orcid:
        expected = normalize_orcid(orcid)
        matches = [
            author for author in authors
            if _author_orcid(author) == expected
        ]
    else:
        matches = _matching_authors(work, author_name)
    if not matches:
        return False
    if not affiliation:
        return True
    affiliation_key = _normalize_text(affiliation)
    return any(
        affiliation_key in _normalize_text(entry.get('name'))
        for author in matches
        for entry in author.get('affiliation') or []
    )


def _request_page(session, params, email, timeout=60, attempts=4):
    """Request one Crossref page with bounded retry/backoff behavior."""
    headers = {'User-Agent': f'PaperScraper/0.0.1 (mailto:{email})'}
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.get(CROSSREF_WORKS_URL, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()['message']
        except (requests.RequestException, KeyError, ValueError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            retry_after = getattr(getattr(error, 'response', None), 'headers', {}).get('Retry-After')
            try:
                delay = min(max(float(retry_after), 0), 60) if retry_after else 2 ** attempt
            except (TypeError, ValueError):
                delay = 2 ** attempt
            time.sleep(delay)
    raise RuntimeError(f'Crossref request failed after {attempts} attempts: {last_error}') from last_error


def author_works(orcid=None,
                 author_name=None,
                 affiliation=None,
                 email=None,
                 max_results=500,
                 page_size=200,
                 session=None):
    """Retrieve DOI-bearing Crossref works for one author identity."""
    if bool(orcid) == bool(author_name):
        raise ValueError('Provide exactly one of orcid or author_name.')
    if not email or '@' not in email:
        raise ValueError('A contact email is required for Crossref polite-pool requests.')
    if max_results is not None and max_results < 1:
        raise ValueError('max_results must be a positive integer or None.')
    if page_size < 1 or page_size > 1000:
        raise ValueError('page_size must be between 1 and 1000.')

    session = session or requests.Session()
    params = {'rows': page_size, 'cursor': '*', 'mailto': email}
    if orcid:
        orcid = normalize_orcid(orcid)
        params['filter'] = f'orcid:{orcid}'
    else:
        params['query.author'] = author_name

    works = []
    seen_dois = set()
    while True:
        message = _request_page(session, params, email)
        items = message.get('items') or []
        for work in items:
            doi = str(work.get('DOI') or '').strip().lower()
            if not doi or doi in seen_dois:
                continue
            if not work_matches_author(work, author_name=author_name, orcid=orcid, affiliation=affiliation):
                continue
            seen_dois.add(doi)
            works.append(work)
            if max_results is not None and len(works) >= max_results:
                return works
        if len(items) < page_size:
            break
        next_cursor = message.get('next-cursor')
        if not next_cursor or next_cursor == params['cursor']:
            break
        params['cursor'] = next_cursor
    return works


def _first(value):
    """Return the first non-empty value from a Crossref list-like field."""
    if isinstance(value, list):
        return next((str(item).strip() for item in value if str(item).strip()), '')
    return str(value or '').strip()


def _publication_date(work):
    """Return the best available Crossref publication date as ISO-like text."""
    for key in ['published-print', 'published-online', 'published', 'issued', 'created']:
        value = work.get(key) or {}
        parts = value.get('date-parts') if isinstance(value, dict) else None
        if parts and parts[0]:
            return '-'.join(f'{int(part):02d}' if index else str(part)
                            for index, part in enumerate(parts[0][:3]))
    return ''


def crossref_work_to_paper(work):
    """Map one Crossref work into the PaperScraper corpus schema."""
    doi = str(work.get('DOI') or '').strip().lower()
    authors = '; '.join(
        ' '.join(part for part in [str(author.get('given') or '').strip(),
                                   str(author.get('family') or '').strip()] if part)
        for author in work.get('author') or []
    )
    return {
        'paper_id': f'doi:{doi}',
        'doi': doi,
        'title': _first(work.get('title')),
        'journal': _first(work.get('container-title')),
        'publication_date': _publication_date(work),
        'authors': authors,
        'sources': 'crossref',
        'metadata_status': 'retrieved',
        'metadata': work,
    }


def import_author_works(db_path,
                        email,
                        orcid=None,
                        author_name=None,
                        affiliation=None,
                        max_results=500,
                        review_csv=None,
                        session=None):
    """Discover an author's Crossref works, write a review CSV, and upsert a corpus."""
    works = author_works(
        orcid=orcid,
        author_name=author_name,
        affiliation=affiliation,
        email=email,
        max_results=max_results,
        session=session,
    )
    papers = [crossref_work_to_paper(work) for work in works]
    if review_csv:
        review_path = Path(review_csv)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(papers, columns=REVIEW_COLUMNS).to_csv(review_path, index=False)
    with connect(db_path) as conn:
        added, updated = upsert_papers(conn, papers)
        for paper in papers:
            matched = find_paper(conn, paper)
            if matched is None:
                continue
            matched['metadata'] = paper['metadata']
            upsert_paper(conn, matched)
    return {'found': len(papers), 'added': added, 'updated': updated}
