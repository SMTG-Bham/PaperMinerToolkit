"""Small request helpers for the Elsevier APIs used by PaperScraper.

This module centralizes Elsevier HTTP details so the rest of the package does
not depend on the unsupported ``elsapy`` wrapper.
"""

from urllib.parse import quote_plus as url_encode

import requests

BASE_URL = 'https://api.elsevier.com/content'
USER_AGENT = 'PaperScraper/0.0.1'


def api_headers(api_key: str, accept: str = 'application/json') -> dict[str, str]:
    """Build standard Elsevier API headers for a configured API key."""
    return {
        'X-ELS-APIKey': api_key,
        'Accept': accept,
        'User-Agent': USER_AGENT,
    }


def get_json(api_key: str, url: str, params: dict | None = None, timeout: int = 60) -> dict:
    """Request an Elsevier JSON endpoint and return the decoded response."""
    response = requests.get(url, headers=api_headers(api_key), params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_content(api_key: str,
                url: str,
                accept: str,
                params: dict | None = None,
                timeout: int = 60) -> requests.Response:
    """Request an Elsevier endpoint and return the raw response after status validation."""
    response = requests.get(url, headers=api_headers(api_key, accept=accept), params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response


def search_url(index: str, query: str, count: int, search_fields: str) -> str:
    """Build an Elsevier search URL for the selected index and query."""
    index = index.lower()
    provider_query = f'{search_fields}({query})'
    url = f'{BASE_URL}/search/{index}?query={url_encode(provider_query)}&count={count}'
    if index == 'scopus':
        url += '&cursor=*'
    return url


def article_url_from_doi(doi: str) -> str:
    """Build an Elsevier article retrieval URL from a DOI."""
    return f'{BASE_URL}/article/doi/{url_encode(str(doi))}'
