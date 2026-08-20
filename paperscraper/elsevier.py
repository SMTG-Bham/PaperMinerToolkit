"""Small request helpers for the Elsevier APIs used by PaperScraper.

This module centralizes Elsevier HTTP details so the rest of the package does
not depend on the unsupported ``elsapy`` wrapper.
"""

from urllib.parse import quote_plus as url_encode

import requests

BASE_URL = 'https://api.elsevier.com/content'
USER_AGENT = 'PaperScraper/0.0.1'


def api_headers(api_key: str, accept: str = 'application/json') -> dict[str, str]:
    """Build standard Elsevier API headers.

    Parameters
    ----------
    api_key : str
        Configured Elsevier API key.
    accept : str, default='application/json'
        Media type requested from Elsevier.

    Returns
    -------
    dict[str, str]
        Headers containing the API key, media type, and user agent.
    """
    return {
        'X-ELS-APIKey': api_key,
        'Accept': accept,
        'User-Agent': USER_AGENT,
    }


def get_json(api_key: str, url: str, params: dict | None = None, timeout: int = 60) -> dict:
    """Request and decode an Elsevier JSON endpoint.

    Parameters
    ----------
    api_key : str
        Configured Elsevier API key.
    url : str
        Endpoint URL to request.
    params : dict or None, optional
        Query parameters for the request.
    timeout : int, default=60
        Request timeout in seconds.

    Returns
    -------
    dict
        Decoded JSON response body.

    Raises
    ------
    requests.RequestException
        If the request fails or the response has an error status.
    """
    response = requests.get(url, headers=api_headers(api_key), params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_content(api_key: str,
                url: str,
                accept: str,
                params: dict | None = None,
                timeout: int = 60) -> requests.Response:
    """Request raw content from an Elsevier endpoint.

    Parameters
    ----------
    api_key : str
        Configured Elsevier API key.
    url : str
        Endpoint URL to request.
    accept : str
        Media type requested from Elsevier.
    params : dict or None, optional
        Query parameters for the request.
    timeout : int, default=60
        Request timeout in seconds.

    Returns
    -------
    requests.Response
        Status-validated raw response.

    Raises
    ------
    requests.RequestException
        If the request fails or the response has an error status.
    """
    response = requests.get(url, headers=api_headers(api_key, accept=accept), params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response


def search_url(index: str, query: str, count: int, search_fields: str) -> str:
    """Build an Elsevier search URL.

    Parameters
    ----------
    index : str
        Elsevier index to search, such as ``scopus``.
    query : str
        Search expression.
    count : int
        Number of records requested per page.
    search_fields : str
        Elsevier field expression that wraps ``query``.

    Returns
    -------
    str
        Encoded Elsevier search URL.
    """
    index = index.lower()
    provider_query = f'{search_fields}({query})'
    url = f'{BASE_URL}/search/{index}?query={url_encode(provider_query)}&count={count}'
    if index == 'scopus':
        url += '&cursor=*'
    return url


def article_url_from_doi(doi: str) -> str:
    """Build an Elsevier article retrieval URL.

    Parameters
    ----------
    doi : str
        DOI identifying the article.

    Returns
    -------
    str
        Encoded Elsevier article URL.
    """
    return f'{BASE_URL}/article/doi/{url_encode(str(doi))}'
