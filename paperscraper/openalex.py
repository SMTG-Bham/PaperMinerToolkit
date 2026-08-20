"""Small request helpers for the OpenAlex API used by PaperScraper.

This module centralizes OpenAlex HTTP details and the mapping from OpenAlex
work records onto PaperScraper's paper schema so search and download code can
share one implementation. OpenAlex meters access against a daily credit budget;
requests without an API key still work but draw on a much smaller budget.
"""

import os
import time
from urllib.parse import quote

import requests

from paperscraper.metadata import clean_doi
from paperscraper.settings import load_settings

BASE_URL = 'https://api.openalex.org'
WORKS_URL = f'{BASE_URL}/works'
RATE_LIMIT_URL = f'{BASE_URL}/rate-limit'
USER_AGENT = 'PaperScraper/0.0.1'


def configured_api_key(settings=None):
    """Return the configured OpenAlex API key.

    Parameters
    ----------
    settings : dict or None, optional
        Settings mapping to inspect before the environment.

    Returns
    -------
    str or None
        Configured API key, or ``None`` when no key is available.
    """
    settings = settings or load_settings()
    return settings.get('openalex_api_key') or os.environ.get('OPENALEX_API_KEY')


def request_headers():
    """Build OpenAlex request headers.

    Returns
    -------
    dict[str, str]
        Headers containing the PaperScraper user agent.
    """
    return {'User-Agent': USER_AGENT}


def request_params(params=None, api_key=None):
    """Copy query parameters and add an API key when configured.

    Requests without a key are served from a much smaller daily credit budget,
    so the key is attached whenever it is available but never invented.

    Parameters
    ----------
    params : dict or None, optional
        Query parameters to copy.
    api_key : str or None, optional
        OpenAlex API key to add.

    Returns
    -------
    dict
        Copied parameters, optionally including ``api_key``.
    """
    merged = dict(params or {})
    if api_key:
        merged['api_key'] = api_key
    return merged


def _budget_error(response):
    """Describe an exhausted OpenAlex credit budget using the rate-limit headers."""
    reset = response.headers.get('X-RateLimit-Reset')
    try:
        wait = f' Budget resets in {round(float(reset) / 3600, 1)} hours.' if reset else ''
    except (TypeError, ValueError):
        wait = ''
    return ('OpenAlex daily credit budget is exhausted.'
            f'{wait} Configure an API key with ps_openalex_key or OPENALEX_API_KEY '
            'to raise the budget.')


def request_json(url, params=None, api_key=None, session=None, timeout=60, attempts=4):
    """Request an OpenAlex endpoint with bounded retry/backoff behavior.

    Parameters
    ----------
    url : str
        OpenAlex endpoint URL.
    params : dict or None, optional
        Query parameters for the request.
    api_key : str or None, optional
        OpenAlex API key to attach.
    session : module or object or None, optional
        HTTP client exposing a ``get`` method. Defaults to :mod:`requests`.
    timeout : int or float, default=60
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    dict or None
        Decoded JSON payload, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If the API key is rejected, the credit budget is exhausted, or all
        request attempts fail.
    """
    session = session or requests
    headers = request_headers()
    params = request_params(params, api_key)
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 404:
                return None
            if response.status_code == 401:
                raise RuntimeError('OpenAlex rejected the API key. Set a valid key with '
                                   'ps_openalex_key or OPENALEX_API_KEY, or unset it to use '
                                   'the smaller keyless budget.')
            if response.status_code == 429:
                raise RuntimeError(_budget_error(response))
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
    """Build a single-work URL.

    Parameters
    ----------
    identifier : str
        OpenAlex W-identifier or ``doi:``-prefixed DOI.

    Returns
    -------
    str
        Encoded OpenAlex work URL.
    """
    return f'{WORKS_URL}/{quote(str(identifier), safe=":/")}'


def get_work(identifier, api_key=None, session=None):
    """Fetch one OpenAlex work record.

    Parameters
    ----------
    identifier : str
        OpenAlex W-identifier or ``doi:``-prefixed DOI.
    api_key : str or None, optional
        OpenAlex API key to attach.
    session : module or object or None, optional
        HTTP client exposing a ``get`` method.

    Returns
    -------
    dict or None
        Work record, or ``None`` when the work does not exist.

    Raises
    ------
    RuntimeError
        If the OpenAlex request cannot be completed.
    """
    return request_json(work_url(identifier), api_key=api_key, session=session)


def work_id(work):
    """Extract the short W-identifier from an OpenAlex work record.

    Parameters
    ----------
    work : dict
        OpenAlex work record.

    Returns
    -------
    str
        Short W-identifier, or an empty string when unavailable.
    """
    identifier = str(work.get('id') or '')
    return identifier.rstrip('/').rsplit('/', 1)[-1] if identifier else ''


def reconstruct_abstract(inverted_index):
    """Rebuild abstract text from an OpenAlex inverted index.

    Parameters
    ----------
    inverted_index : dict or None
        Mapping of tokens to their positions in the abstract.

    Returns
    -------
    str
        Reconstructed abstract, or an empty string for a missing index.
    """
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ''
    positions = [(position, token)
                 for token, indexes in inverted_index.items()
                 for position in indexes or []]
    return ' '.join(token for _, token in sorted(positions))


def work_to_paper(work):
    """Map an OpenAlex work onto PaperScraper's paper schema.

    Parameters
    ----------
    work : dict
        OpenAlex work record.

    Returns
    -------
    dict
        Normalized paper metadata.
    """
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
    """Return candidate PDF URLs for an OpenAlex work.

    Parameters
    ----------
    work : dict
        OpenAlex work record.

    Returns
    -------
    list[str]
        Deduplicated PDF candidates, most authoritative first.
    """
    candidates = [(work.get('best_oa_location') or {}).get('pdf_url')]
    for location in work.get('locations') or []:
        candidates.append((location or {}).get('pdf_url'))
    candidates.append((work.get('open_access') or {}).get('oa_url'))
    return list(dict.fromkeys(url for url in candidates if url))
