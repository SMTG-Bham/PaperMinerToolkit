"""Unit tests for CORE API request helpers and CORE record mapping."""

from __future__ import annotations

from typing import Any

import pytest

import paperminer.providers.core as core
from paperminer.providers import base as provider

from tests.doubles import FakeResponse, FakeSession


def work(core_id: str = '123', **overrides: Any) -> dict[str, Any]:
    """Return a CORE work record for the tests to map.

    Parameters
    ----------
    core_id : str, default='123'
        Identifier CORE issued for the work.
    **overrides : Any
        Fields to replace on the record.

    Returns
    -------
    dict[str, Any]
        CORE work record.
    """
    record = {
        'id': core_id,
        'doi': '10.1234/example',
        'title': 'A repository deposit',
        'authors': [{'name': 'A. Author'}, {'fullName': 'B. Author'}, 'C. Author', {'name': ''}],
        'journal': {'title': 'Journal Title'},
        'yearPublished': 2024,
        'abstract': 'An  abstract   with spacing.',
        'downloadUrl': 'https://repository.example/paper.pdf',
    }
    record.update(overrides)
    return record


def test_configured_api_key_prefers_settings_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the stored key first, and fall back to the environment variable."""
    monkeypatch.delenv('CORE_API_KEY', raising=False)
    assert core.configured_api_key({'core_api_key': 'stored'}) == 'stored'
    assert core.configured_api_key({}) is None

    monkeypatch.setenv('CORE_API_KEY', 'from-env')
    assert core.configured_api_key({}) == 'from-env'
    assert core.configured_api_key({'core_api_key': 'stored'}) == 'stored'


def test_request_headers_carry_the_key_as_a_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the key in the Authorization header, not as a query parameter."""
    monkeypatch.setattr(core, 'configured_api_key', lambda: None)
    assert core.request_headers() == {'User-Agent': provider.USER_AGENT}

    monkeypatch.setattr(core, 'configured_api_key', lambda: 'core-key')
    headers = core.request_headers()
    assert headers['Authorization'] == 'Bearer core-key'
    assert headers['User-Agent'] == provider.USER_AGENT
    assert core.request_headers('explicit')['Authorization'] == 'Bearer explicit'


def test_work_and_download_urls_are_built_from_one_base() -> None:
    """Build every CORE URL from the single base rather than from literals."""
    assert core.work_url('123') == f'{core.WORKS_URL}/123'
    assert core.work_url(' 123 ') == f'{core.WORKS_URL}/123'
    assert core.work_url('a/b') == f'{core.WORKS_URL}/a%2Fb'
    assert core.work_url('') == ''
    assert core.work_url(None) == ''

    assert core.download_url({'downloadUrl': 'https://example.test/pdf'}) == (
        'https://example.test/pdf')
    assert core.download_url({'id': '123'}) == f'{core.WORKS_URL}/123/download'
    assert core.download_url({}) == ''


def test_work_to_paper_maps_a_repository_record_onto_the_schema() -> None:
    """Key a CORE record by its own identifier, which is what reaches the API."""
    record = core.work_to_paper(work())

    assert record['paper_id'] == 'core:123'
    assert record['core_id'] == '123'
    assert record['doi'] == '10.1234/example'
    assert record['sources'] == 'core'
    assert record['title'] == 'A repository deposit'
    assert record['journal'] == 'Journal Title'
    assert record['publication_date'] == '2024'
    assert record['authors'] == 'A. Author; B. Author; C. Author'
    assert record['abstract'] == 'An abstract with spacing.'
    assert record['pdf_url'] == 'https://repository.example/paper.pdf'
    assert record['metadata_status'] == 'retrieved'


def test_work_to_paper_falls_back_to_a_doi_and_then_to_nothing() -> None:
    """Identify a record however little it carries, without raising."""
    assert core.work_to_paper(work(core_id=''))['paper_id'] == 'doi:10.1234/example'
    assert core.work_to_paper({'doi': None, 'id': ''})['paper_id'] == ''

    sparse = core.work_to_paper({'id': '9'})
    assert sparse['title'] == ''
    assert sparse['authors'] == ''
    assert sparse['journal'] == ''
    assert sparse['publication_date'] == ''


def test_journal_falls_back_to_the_publisher_for_a_repository_deposit() -> None:
    """Name the closest thing to a venue when no journal is recorded."""
    assert core._journal({'journal': {'title': 'Journal Title'}}) == 'Journal Title'
    assert core._journal({'journal': {'name': 'By Name'}}) == 'By Name'
    assert core._journal({'publisher': 'A University'}) == 'A University'
    assert core._journal({}) == ''


def test_publication_date_prefers_a_full_date_over_a_bare_year() -> None:
    """Take the most precise date the record carries."""
    assert core._publication_date({'publishedDate': '2024-03-04', 'year': 2020}) == '2024-03-04'
    assert core._publication_date({'published_date': '2024-03-04'}) == '2024-03-04'
    assert core._publication_date({'yearPublished': 2024}) == '2024'
    assert core._publication_date({'year': 2020}) == '2020'
    assert core._publication_date({}) == ''


def test_parse_records_reads_both_envelopes_and_tolerates_nonsense() -> None:
    """Accept either key CORE has answered under, and refuse anything else."""
    assert core.parse_records({'results': [work()]})[0]['id'] == '123'
    assert core.parse_records({'data': [work()]})[0]['id'] == '123'
    assert core.parse_records({'results': []}) == []
    assert core.parse_records({'results': 'not a list'}) == []
    assert core.parse_records(None) == []


def test_total_results_reads_any_of_the_count_fields() -> None:
    """Read the hit count whichever name the payload gives it."""
    assert core.total_results({'totalHits': 101}) == 101
    assert core.total_results({'total': '42'}) == 42
    assert core.total_results({'count': 7}) == 7
    assert core.total_results({'totalHits': 'many'}) == 0
    assert core.total_results({}) == 0
    assert core.total_results(None) == 0


def test_search_page_caps_the_page_and_floors_the_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ask for a page CORE will actually serve, whatever the caller requested."""
    session = FakeSession([FakeResponse(payload={'results': []}) for _ in range(3)])
    monkeypatch.setattr(core, 'configured_api_key', lambda: 'core-key')

    core.search_page('garnet', limit=500, offset=-5, session=session)
    core.search_page('garnet', limit=0, offset=10, session=session)
    core.search_page('garnet', session=session)

    assert session.calls[0]['params'] == {'q': 'garnet', 'limit': core.PAGE_SIZE, 'offset': 0}
    assert session.calls[1]['params'] == {'q': 'garnet', 'limit': 1, 'offset': 10}
    assert session.calls[2]['params'] == {'q': 'garnet', 'limit': core.PAGE_SIZE, 'offset': 0}
    assert session.calls[0]['url'] == core.SEARCH_URL
    assert session.calls[0]['headers']['Authorization'] == 'Bearer core-key'


def test_get_work_returns_none_without_an_identifier() -> None:
    """Skip the request entirely when the row names no CORE work."""
    session = FakeSession([])
    assert core.get_work('', session=session) is None
    assert core.get_work(None, session=session) is None
    assert session.calls == []


def test_request_json_reports_a_rejected_request_with_the_provider_name() -> None:
    """Fail a client error once, naming CORE rather than leaking an HTTP error."""
    session = FakeSession([FakeResponse(status_code=401)])
    with pytest.raises(RuntimeError, match='CORE rejected the request with 401'):
        core.request_json(core.SEARCH_URL, session=session)
    assert len(session.calls) == 1


def test_request_json_reads_a_missing_work_as_nothing_found() -> None:
    """Read a 404 as an absent record rather than as a failure."""
    session = FakeSession([FakeResponse(status_code=404)])
    assert core.request_json(core.work_url('123'), session=session) is None


def test_resolve_core_id_reads_a_stored_identifier_without_a_request() -> None:
    """Take the identifier from the row, trimmed, or report that there is none."""
    assert core.resolve_core_id({'core_id': ' 123 '}) == '123'
    assert core.resolve_core_id({'core_id': ''}) == ''
    assert core.resolve_core_id({}) == ''
