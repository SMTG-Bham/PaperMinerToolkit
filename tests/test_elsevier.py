import pytest

import paperscraper.elsevier as elsevier


def test_api_headers_include_key_accept_and_user_agent():
    """
    Test Elsevier API header construction.

    This function performs the following steps:
    1. Builds default JSON headers.
    2. Builds PDF headers with an explicit accept value.
    3. Checks the returned dictionaries.

    Asserts:
        - The API key is sent using the Elsevier header name.
        - The accept header defaults to JSON.
        - Custom accept headers are preserved.
        - A PaperScraper user agent is included.
    """
    assert elsevier.api_headers('elsevier-key') == {
        'X-ELS-APIKey': 'elsevier-key',
        'Accept': 'application/json',
        'User-Agent': 'PaperScraper/0.0.1',
    }
    assert elsevier.api_headers('elsevier-key', accept='application/pdf')['Accept'] == 'application/pdf'


def test_get_json_requests_elsevier_json_and_raises_status(monkeypatch):
    """
    Test JSON requests through the Elsevier helper.

    This function performs the following steps:
    1. Replaces HTTP GET with a fake response object.
    2. Calls `get_json` with params and timeout values.
    3. Checks the recorded request and returned payload.

    Asserts:
        - Requests include Elsevier headers, params, and timeout.
        - HTTP status validation is called.
        - The decoded JSON payload is returned.
    """
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            calls['raised'] = True

        def json(self):
            return {'ok': True}

    def fake_get(url, headers, params, timeout):
        calls['url'] = url
        calls['headers'] = headers
        calls['params'] = params
        calls['timeout'] = timeout
        return FakeResponse()

    monkeypatch.setattr(elsevier.requests, 'get', fake_get)

    assert elsevier.get_json('elsevier-key', 'https://example.com/search', params={'count': 1}, timeout=5) == {
        'ok': True,
    }
    assert calls['url'] == 'https://example.com/search'
    assert calls['headers']['X-ELS-APIKey'] == 'elsevier-key'
    assert calls['params'] == {'count': 1}
    assert calls['timeout'] == 5
    assert calls['raised'] is True


def test_get_content_requests_elsevier_raw_response(monkeypatch):
    """
    Test raw content requests through the Elsevier helper.

    This function performs the following steps:
    1. Replaces HTTP GET with a fake response object.
    2. Calls `get_content` with a PDF accept header.
    3. Checks the recorded request and returned response.

    Asserts:
        - Requests include the custom accept header.
        - HTTP status validation is called.
        - The raw response object is returned.
    """
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            calls['raised'] = True

    response = FakeResponse()

    def fake_get(url, headers, params, timeout):
        calls['headers'] = headers
        calls['params'] = params
        return response

    monkeypatch.setattr(elsevier.requests, 'get', fake_get)

    assert elsevier.get_content('elsevier-key', 'https://example.com/article', 'application/pdf') is response
    assert calls['headers']['Accept'] == 'application/pdf'
    assert calls['params'] == {}
    assert calls['raised'] is True


def test_elsevier_url_builders_quote_query_and_doi_values():
    """
    Test Elsevier URL builders.

    This function performs the following steps:
    1. Builds a Scopus search URL.
    2. Builds a non-Scopus search URL.
    3. Builds an article URL from a DOI containing reserved characters.

    Asserts:
        - Search terms are wrapped in the selected search field.
        - Scopus search URLs include cursor pagination.
        - Non-Scopus search URLs omit cursor pagination.
        - DOI values are URL encoded for article retrieval.
    """
    scopus_url = elsevier.search_url('scopus', 'solid electrolyte', 10, 'TITLE-ABS-KEY')
    article_search_url = elsevier.search_url('article', 'solid electrolyte', 10, 'TITLE')

    assert 'content/search/scopus' in scopus_url
    assert 'query=TITLE-ABS-KEY%28solid+electrolyte%29' in scopus_url
    assert '&count=10' in scopus_url
    assert '&cursor=*' in scopus_url
    assert 'query=TITLE%28solid+electrolyte%29' in article_search_url
    assert '&cursor=*' not in article_search_url
    assert elsevier.article_url_from_doi('10.1234/a b') == (
        'https://api.elsevier.com/content/article/doi/10.1234%2Fa+b'
    )
