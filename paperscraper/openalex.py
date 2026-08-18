"""Small request helpers for the OpenAlex API used by PaperScraper.

This module centralizes OpenAlex HTTP details and the mapping from OpenAlex
work records onto PaperScraper's paper schema so search and download code can
share one implementation. OpenAlex needs no API key; configuring a contact
email joins the faster polite pool.
"""

import os
import time
from urllib.parse import quote

import requests

from paperscraper.metadata import clean_doi
from paperscraper.settings import load_settings

BASE_URL = 'https://api.openalex.org'
WORKS_URL = f'{BASE_URL}/works'
USER_AGENT = 'PaperScraper/0.0.1'


def configured_email(settings=None):
    """Return the polite-pool contact email for OpenAlex requests, if any."""
    settings = settings or load_settings()
    return (settings.get('openalex_email')
            or os.environ.get('OPENALEX_EMAIL')
            or settings.get('unpaywall_email')
            or os.environ.get('UNPAYWALL_EMAIL'))


def request_headers(email=None):
    """Build OpenAlex request headers, joining the polite pool when an email is set."""
    if email:
        return {'User-Agent': f'{USER_AGENT} (mailto:{email})'}
    return {'User-Agent': USER_AGENT}


def request_json(url, params=None, email=None, session=None, timeout=60, attempts=4):
    """Request an OpenAlex endpoint with bounded retry/backoff behavior.

    Returns the decoded JSON payload, or ``None`` for a 404 response so
    single-work lookups can miss without retrying.
    """
    session = session or requests
    headers = request_headers(email)
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params or {}, headers=headers, timeout=timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            retry_after = getattr(getattr(error, 'response', None), 'headers', {}).get('Retry-After')
            try:
                delay = min(max(float(retry_after), 0), 60) if retry_after else 2 ** attempt
            except (TypeError, ValueError):
                delay = 2 ** attempt
            time.sleep(delay)
    raise RuntimeError(f'OpenAlex request failed after {attempts} attempts: {last_error}') from last_error


def work_url(identifier):
    """Build a single-work URL from a W-identifier or ``doi:``-prefixed DOI."""
    return f'{WORKS_URL}/{quote(str(identifier), safe=":/")}'


def get_work(identifier, email=None, session=None):
    """Fetch one OpenAlex work record, returning ``None`` when it does not exist."""
    return request_json(work_url(identifier), email=email, session=session)


def work_id(work):
    """Extract the short W-identifier from an OpenAlex work record."""
    identifier = str(work.get('id') or '')
    return identifier.rstrip('/').rsplit('/', 1)[-1] if identifier else ''


def reconstruct_abstract(inverted_index):
    """Rebuild plain abstract text from an OpenAlex inverted index."""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ''
    positions = [(position, token)
                 for token, indexes in inverted_index.items()
                 for position in indexes or []]
    return ' '.join(token for _, token in sorted(positions))


def work_to_paper(work):
    """Map an OpenAlex work record onto PaperScraper's paper schema."""
    doi = clean_doi(work.get('doi')) if work.get('doi') else ''
    identifier = work_id(work)
    if doi:
        paper_id = f'doi:{doi}'
    elif identifier:
        paper_id = f'openalex:{identifier}'
    else:
        paper_id = ''
    authors = '; '.join(
        author for author in (
            ((authorship or {}).get('author') or {}).get('display_name')
            for authorship in work.get('authorships') or []
        ) if author
    )
    journal = ((work.get('primary_location') or {}).get('source') or {}).get('display_name')
    return {
        'paper_id': paper_id,
        'doi': doi,
        'title': work.get('title') or work.get('display_name') or '',
        'journal': journal or '',
        'publication_date': work.get('publication_date') or str(work.get('publication_year') or ''),
        'authors': authors,
        'sources': 'openalex',
        'pdf_url': (work.get('best_oa_location') or {}).get('pdf_url') or '',
        'metadata_status': 'retrieved',
    }


def pdf_candidates(work):
    """Return candidate PDF URLs for a work, most authoritative first."""
    candidates = [(work.get('best_oa_location') or {}).get('pdf_url')]
    for location in work.get('locations') or []:
        candidates.append((location or {}).get('pdf_url'))
    candidates.append((work.get('open_access') or {}).get('oa_url'))
    return list(dict.fromkeys(url for url in candidates if url))
