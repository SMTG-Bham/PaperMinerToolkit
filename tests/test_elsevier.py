"""Test Elsevier request helpers and URL construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

import paperscraper.elsevier as elsevier
from paperscraper import provider

from tests.doubles import FakeResponse, FakeSession


def test_api_headers_include_key_accept_and_user_agent() -> None:
    """API headers include key accept and user agent."""
    assert elsevier.api_headers('elsevier-key') == {
        'X-ELS-APIKey': 'elsevier-key',
        'Accept': 'application/json',
        'User-Agent': 'PaperMiner/0.0.1',
    }
    assert elsevier.api_headers('elsevier-key', accept='application/pdf')['Accept'] == 'application/pdf'


def test_get_json_requests_elsevier_json_and_raises_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Get JSON requests Elsevier JSON and raises status."""
    calls = {}

    class FakeResponse:
        """Provide a response test double."""

        def raise_for_status(self) -> None:
            """Validate the prepared response status."""
            calls['raised'] = True

        def json(self) -> dict[str, bool]:
            """Return the prepared JSON payload."""
            return {'ok': True}

    def fake_get(
        url: str,
        headers: Mapping[str, str],
        params: Mapping[str, Any],
        timeout: float,
    ) -> FakeResponse:
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


def test_get_content_requests_elsevier_raw_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Get content requests Elsevier raw response."""
    calls = {}

    class FakeResponse:
        """Provide a response test double."""

        def raise_for_status(self) -> None:
            """Validate the prepared response status."""
            calls['raised'] = True

    response = FakeResponse()

    def fake_get(
        url: str,
        headers: Mapping[str, str],
        params: Mapping[str, Any],
        timeout: float,
    ) -> FakeResponse:
        """Provide a fake HTTP GET implementation."""
        calls['headers'] = headers
        calls['params'] = params
        return response

    monkeypatch.setattr(elsevier.requests, 'get', fake_get)

    assert elsevier.get_content('elsevier-key', 'https://example.com/article', 'application/pdf') is response
    assert calls['headers']['Accept'] == 'application/pdf'
    assert calls['params'] == {}
    assert calls['raised'] is True


def test_elsevier_url_builders_quote_query_and_doi_values() -> None:
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


def test_configured_api_key_prefers_settings_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the key in the module, not separately in search and download."""
    monkeypatch.delenv('ELSEVIER_API_KEY', raising=False)
    assert elsevier.configured_api_key({'elsevier_api_key': 'stored'}) == 'stored'

    monkeypatch.setenv('ELSEVIER_API_KEY', 'from-env')
    assert elsevier.configured_api_key({}) == 'from-env'

    monkeypatch.delenv('ELSEVIER_API_KEY', raising=False)
    with pytest.raises(ValueError, match='Elsevier API key is not configured'):
        elsevier.configured_api_key({})


def test_request_json_reports_a_rejection_as_a_runtime_error() -> None:
    """Fail like every other source rather than leaking a requests error.

    Leaking requests.RequestException is what forced search_for_papers to catch
    a different exception type for this source than for the others.
    """
    session = FakeSession([FakeResponse(status_code=401)])
    with pytest.raises(RuntimeError, match='Elsevier rejected the request with 401'):
        elsevier.request_json(elsevier.BASE_URL, 'key', session=session)
    assert len(session.calls) == 1


def test_request_json_sends_the_key_as_a_header() -> None:
    """Authenticate through X-ELS-APIKey, not through the query string."""
    session = FakeSession([FakeResponse(payload={'search-results': {}})])
    elsevier.request_json(elsevier.BASE_URL, 'elsevier-key', session=session)

    assert session.calls[0]['headers']['X-ELS-APIKey'] == 'elsevier-key'
    assert session.calls[0]['headers']['User-Agent'] == provider.USER_AGENT
    assert session.calls[0]['params'] == {}


def test_search_payload_helpers_read_the_envelope() -> None:
    """Read the count, the entries, and the next link out of one payload."""
    payload = {'search-results': {
        'opensearch:totalResults': '58',
        'entry': [{'dc:title': 'One'}, {'dc:title': 'Two'}, 'not a record'],
        'link': [{'@ref': 'self', '@href': 'here'}, {'@ref': 'next', '@href': 'there'}],
    }}

    assert elsevier.total_results(payload) == 58
    assert [record['dc:title'] for record in elsevier.parse_records(payload)] == ['One', 'Two']
    assert elsevier.next_page_url(payload) == 'there'

    last_page = {'search-results': {'link': [{'@ref': 'self', '@href': 'here'}]}}
    assert elsevier.next_page_url(last_page) == ''
    assert elsevier.total_results({'search-results': {}}) == 0
    assert elsevier.parse_records(None) == []
    assert elsevier.next_page_url(None) == ''


def test_record_to_paper_maps_a_scopus_record_onto_the_schema() -> None:
    """Map an Elsevier record here rather than inside the search module."""
    record = elsevier.record_to_paper({
        'dc:identifier': 'SCOPUS_ID:85001',
        'prism:doi': 'https://doi.org/10.1016/j.example.2024.01.001',
        'dc:title': 'A   subscription   article',
        'prism:publicationName': 'Journal of Examples',
        'prism:coverDate': '2024-03-01',
        'dc:creator': 'A. Author',
        'dc:description': 'An abstract.',
        'link': [{'@ref': 'scopus', '@href': 'https://www.scopus.com/x'},
                 {'@ref': 'full-text',
                  '@href': f'{elsevier.BASE_URL}/article/eid/1-s2.0-0001'}],
    })

    assert record['paper_id'] == 'SCOPUS_ID:85001'
    assert record['doi'] == '10.1016/j.example.2024.01.001'
    assert record['title'] == 'A subscription article'
    assert record['journal'] == 'Journal of Examples'
    assert record['publication_date'] == '2024-03-01'
    assert record['authors'] == 'A. Author'
    assert record['sources'] == 'elsevier'
    assert record['abstract'] == 'An abstract.'
    assert record['elsevier_link'] == f'{elsevier.BASE_URL}/article/eid/1-s2.0-0001'


def test_record_to_paper_falls_back_to_the_eid_and_survives_a_sparse_record() -> None:
    """Identify a record by its EID when no Scopus identifier is present."""
    assert elsevier.record_to_paper({'eid': '2-s2.0-1'})['paper_id'] == '2-s2.0-1'
    sparse = elsevier.record_to_paper({})
    assert sparse['paper_id'] == ''
    assert sparse['doi'] == ''
    assert sparse['elsevier_link'] == ''


def test_full_text_uri_ignores_links_that_are_not_the_article_route() -> None:
    """Take only a link the article endpoint will actually answer for."""
    article = f'{elsevier.BASE_URL}/article/doi/10.1/x'
    assert elsevier.full_text_uri({'link': article}) == article
    assert elsevier.full_text_uri({'elsevier_link': article}) == article
    assert elsevier.full_text_uri({'link': [{'@href': 'https://www.scopus.com/x'}]}) == ''
    assert elsevier.full_text_uri({}) == ''


def test_check_api_key_reports_acceptance_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Answer yes or no, so a settings prompt need not handle an exception."""
    monkeypatch.setattr(elsevier, 'request_json', lambda *_, **__: {'search-results': {}})
    assert elsevier.check_api_key('good-key') is True

    monkeypatch.setattr(
        elsevier, 'request_json',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('Elsevier rejected the request')))
    assert elsevier.check_api_key('bad-key') is False
