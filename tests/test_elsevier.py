"""Test Elsevier request helpers and URL construction."""

import pytest

import paperscraper.elsevier as elsevier


def test_api_headers_include_key_accept_and_user_agent():
    """API headers include key accept and user agent."""
    assert elsevier.api_headers('elsevier-key') == {
        'X-ELS-APIKey': 'elsevier-key',
        'Accept': 'application/json',
        'User-Agent': 'PaperScraper/0.0.1',
    }
    assert elsevier.api_headers('elsevier-key', accept='application/pdf')['Accept'] == 'application/pdf'


def test_get_json_requests_elsevier_json_and_raises_status(monkeypatch):
    """Get JSON requests Elsevier JSON and raises status."""
    calls = {}

    class FakeResponse:
        """Provide a response test double."""

        def raise_for_status(self):
            """Validate the prepared response status."""
            calls['raised'] = True

        def json(self):
            """Return the prepared JSON payload."""
            return {'ok': True}

    def fake_get(url, headers, params, timeout):
        """Provide a fake HTTP GET implementation."""
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
    """Get content requests Elsevier raw response."""
    calls = {}

    class FakeResponse:
        """Provide a response test double."""

        def raise_for_status(self):
            """Validate the prepared response status."""
            calls['raised'] = True

    response = FakeResponse()

    def fake_get(url, headers, params, timeout):
        """Provide a fake HTTP GET implementation."""
        calls['headers'] = headers
        calls['params'] = params
        return response

    monkeypatch.setattr(elsevier.requests, 'get', fake_get)

    assert elsevier.get_content('elsevier-key', 'https://example.com/article', 'application/pdf') is response
    assert calls['headers']['Accept'] == 'application/pdf'
    assert calls['params'] == {}
    assert calls['raised'] is True


def test_elsevier_url_builders_quote_query_and_doi_values():
    """Elsevier URL builders quote query and DOI values."""
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
