"""Unit tests for OpenAlex request helpers and work-record mapping."""

from __future__ import annotations

from typing import Any

import pytest

import paperminer.providers.openalex as openalex
from paperminer.providers import base as provider

from tests.doubles import FakeResponse, FakeSession


def work(
    doi: str | None = 'https://doi.org/10.1234/Example.One',
    identifier: str = 'https://openalex.org/W123',
) -> dict[str, Any]:
    """Return a minimal OpenAlex work record."""
    return {
        'id': identifier,
        'doi': doi,
        'title': 'Solid electrolyte interphase',
        'display_name': 'Solid electrolyte interphase',
        'publication_date': '2024-03-07',
        'publication_year': 2024,
        'authorships': [
            {'author': {'display_name': 'Jane A. Smith'}},
            {'author': {'display_name': 'Wei Chen'}},
        ],
        'primary_location': {'source': {'display_name': 'Test Journal'}},
        'best_oa_location': {'pdf_url': 'https://example.org/best.pdf'},
        'locations': [
            {'pdf_url': 'https://example.org/best.pdf'},
            {'pdf_url': 'https://example.org/repo.pdf'},
            {'pdf_url': None},
        ],
        'open_access': {'oa_url': 'https://example.org/landing'},
        'abstract_inverted_index': {'Despite': [0], 'growth': [2], 'the': [1]},
    }


def test_reconstruct_abstract_orders_tokens_and_handles_missing_index() -> None:
    """Rebuild readable text from position lists and tolerate absent indexes."""
    assert openalex.reconstruct_abstract({'Despite': [0], 'growth': [2], 'the': [1]}) == 'Despite the growth'
    assert openalex.reconstruct_abstract({'is': [1, 3], 'It': [0], 'good': [2]}) == 'It is good is'
    assert openalex.reconstruct_abstract(None) == ''
    assert openalex.reconstruct_abstract({}) == ''


def test_work_to_paper_maps_and_cleans_fields() -> None:
    """Clean URL-form DOIs and flatten nested OpenAlex fields into paper columns."""
    paper = openalex.work_to_paper(work())
    assert paper['paper_id'] == 'doi:10.1234/example.one'
    assert paper['doi'] == '10.1234/example.one'
    assert paper['title'] == 'Solid electrolyte interphase'
    assert paper['journal'] == 'Test Journal'
    assert paper['publication_date'] == '2024-03-07'
    assert paper['authors'] == 'Jane A. Smith; Wei Chen'
    assert paper['sources'] == 'openalex'
    assert paper['pdf_url'] == 'https://example.org/best.pdf'
    assert paper['metadata_status'] == 'retrieved'


def test_work_to_paper_without_doi_uses_openalex_identifier() -> None:
    """Fall back to the W-identifier and empty fields when data is missing."""
    record = work(doi=None)
    record['primary_location'] = None
    record['authorships'] = None
    record['best_oa_location'] = None
    record['publication_date'] = None

    paper = openalex.work_to_paper(record)

    assert paper['paper_id'] == 'openalex:W123'
    assert paper['doi'] == ''
    assert paper['journal'] == ''
    assert paper['authors'] == ''
    assert paper['pdf_url'] == ''
    assert paper['publication_date'] == '2024'
    assert openalex.work_to_paper({})['paper_id'] == ''


def test_request_json_honors_retry_after_and_backoff_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep for Retry-After seconds when numeric and exponential backoff otherwise."""
    sleeps = []
    monkeypatch.setattr(provider.time, 'sleep', sleeps.append)

    unavailable = FakeResponse(payload={}, status_code=503)
    unavailable.headers['Retry-After'] = '7'
    session = FakeSession([unavailable, FakeResponse(payload={'ok': True})])
    assert openalex.request_json(openalex.WORKS_URL, session=session) == {'ok': True}

    malformed = FakeResponse(payload={}, status_code=503)
    malformed.headers['Retry-After'] = 'soon'
    session = FakeSession([malformed, FakeResponse(payload={'ok': True})])
    assert openalex.request_json(openalex.WORKS_URL, session=session) == {'ok': True}

    # Pacing sleeps are interleaved now that OpenAlex has a courtesy delay, so
    # check for the backoff values rather than for the whole sequence.
    assert 7.0 in sleeps
    assert 1 in sleeps


def test_request_json_returns_none_for_missing_work() -> None:
    """Treat a 404 as a terminal miss instead of retrying."""
    session = FakeSession([FakeResponse(payload={'error': 'not found'}, status_code=404)])
    assert openalex.request_json(openalex.WORKS_URL, session=session) is None
    assert len(session.calls) == 1


def test_request_json_raises_after_exhausting_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface a RuntimeError containing the last error once retries run out."""
    monkeypatch.setattr(provider.time, 'sleep', lambda delay: None)
    session = FakeSession([FakeResponse(payload={}, status_code=500), FakeResponse(payload={}, status_code=500)])
    with pytest.raises(RuntimeError, match='OpenAlex request failed after 2 attempts'):
        openalex.request_json(openalex.WORKS_URL, session=session, attempts=2)


def test_get_work_builds_doi_and_id_urls() -> None:
    """Request single works by DOI or W-identifier, sending the key when configured."""
    session = FakeSession([FakeResponse(payload=work()), FakeResponse(payload=work())])

    openalex.get_work('doi:10.1234/example.one', api_key='oa-key', session=session)
    openalex.get_work('W123', session=session)

    assert session.calls[0]['url'] == 'https://api.openalex.org/works/doi:10.1234/example.one'
    assert session.calls[0]['params']['api_key'] == 'oa-key'
    assert session.calls[0]['headers']['User-Agent'] == 'PaperMiner/0.0.1'
    assert session.calls[1]['url'] == 'https://api.openalex.org/works/W123'
    assert 'api_key' not in session.calls[1]['params']


def test_request_params_adds_api_key_only_when_configured() -> None:
    """Copy caller parameters and attach the API key only when one is available."""
    params = {'search': 'solid electrolyte'}

    assert openalex.request_params(params, 'oa-key') == {'search': 'solid electrolyte', 'api_key': 'oa-key'}
    assert openalex.request_params(params, None) == {'search': 'solid electrolyte'}
    assert openalex.request_params(params, '') == {'search': 'solid electrolyte'}
    assert openalex.request_params(None, 'oa-key') == {'api_key': 'oa-key'}
    assert params == {'search': 'solid electrolyte'}


def test_request_json_raises_immediately_for_a_rejected_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast on 401 rather than retrying a key OpenAlex will keep rejecting."""
    monkeypatch.setattr(provider.time, 'sleep', lambda delay: None)
    session = FakeSession([FakeResponse(payload={'error': 'Invalid or missing API key'}, status_code=401)])

    with pytest.raises(RuntimeError, match='OpenAlex rejected the API key'):
        openalex.request_json(openalex.WORKS_URL, api_key='bad-key', session=session)

    assert len(session.calls) == 1


def test_request_json_raises_immediately_when_the_credit_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail fast on 429 and report the reset window, which outlasts any backoff."""
    monkeypatch.setattr(provider.time, 'sleep', lambda delay: None)
    exhausted = FakeResponse(payload={}, status_code=429)
    exhausted.headers['X-RateLimit-Reset'] = '32841'
    session = FakeSession([exhausted])

    with pytest.raises(RuntimeError, match='daily credit budget is exhausted') as excinfo:
        openalex.request_json(openalex.WORKS_URL, session=session)

    assert 'resets in 9.1 hours' in str(excinfo.value)
    assert len(session.calls) == 1

    unlabelled = FakeResponse(payload={}, status_code=429)
    unlabelled.headers['X-RateLimit-Reset'] = 'soon'
    session = FakeSession([unlabelled])
    with pytest.raises(RuntimeError, match='daily credit budget is exhausted'):
        openalex.request_json(openalex.WORKS_URL, session=session)


def test_pdf_candidates_orders_and_deduplicates_urls() -> None:
    """Prefer the best OA location, then other locations, then the OA landing URL."""
    assert openalex.pdf_candidates(work()) == [
        'https://example.org/best.pdf',
        'https://example.org/repo.pdf',
        'https://example.org/landing',
    ]
    assert openalex.pdf_candidates({}) == []


def test_configured_api_key_prefers_settings_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the OpenAlex API key from settings before the environment."""
    monkeypatch.delenv('OPENALEX_API_KEY', raising=False)
    assert openalex.configured_api_key({'openalex_api_key': 'settings-key'}) == 'settings-key'
    assert openalex.configured_api_key({'elsevier_api_key': 'key'}) is None
    monkeypatch.setenv('OPENALEX_API_KEY', 'env-key')
    assert openalex.configured_api_key({'openalex_api_key': 'settings-key'}) == 'settings-key'
    assert openalex.configured_api_key({'elsevier_api_key': 'key'}) == 'env-key'


@pytest.mark.network
def test_get_work_uses_real_openalex_api() -> None:
    """Fetch one known open-access work from the live OpenAlex API."""
    record = openalex.get_work('doi:10.1371/journal.pone.0000308', api_key=openalex.configured_api_key())
    assert record is not None
    assert record['doi'] == 'https://doi.org/10.1371/journal.pone.0000308'


def test_request_params_adds_mailto_only_when_supplied() -> None:
    """Attach the contact address to query parameters only when it is set."""
    assert openalex.request_params({'filter': 'doi:10.1/a'}) == {'filter': 'doi:10.1/a'}
    assert openalex.request_params({}, 'key', 'me@example.com') == {
        'api_key': 'key',
        'mailto': 'me@example.com',
    }


def test_works_page_builds_an_or_filter_and_pins_the_page_size() -> None:
    """Request an OR-joined filter with a page size covering every value."""
    session = FakeSession([FakeResponse(payload={'results': [work()]})])

    results = openalex.works_page('10.1/a|10.1/b', per_page=2, session=session,
                                  mailto='me@example.com')

    params = session.calls[0]['params']
    assert params['filter'] == 'doi:10.1/a|10.1/b'
    assert params['per-page'] == 2
    assert params['mailto'] == 'me@example.com'
    assert params['select'] == ','.join(openalex.WORK_SELECT_FIELDS)
    assert len(results) == 1


def test_work_select_fields_are_root_level_only() -> None:
    """Request only root-level fields, which is all OpenAlex select accepts."""
    assert all('.' not in field for field in openalex.WORK_SELECT_FIELDS)
    assert {'open_access', 'biblio', 'primary_location', 'authorships'} <= set(openalex.WORK_SELECT_FIELDS)


def test_works_batch_keys_results_by_clean_doi_and_reports_misses() -> None:
    """Key batch results by cleaned DOI regardless of the response order."""
    session = FakeSession([FakeResponse(payload={'results': [
        work(doi='https://doi.org/10.1234/Second'),
        work(doi='https://doi.org/10.1234/First'),
    ]})])

    works = openalex.works_batch(['10.1234/first', '10.1234/second', '10.1234/missing'],
                                 session=session)

    assert set(works) == {'10.1234/first', '10.1234/second'}
    assert session.calls[0]['params']['filter'] == 'doi:10.1234/first|10.1234/second|10.1234/missing'


def test_works_batch_chunks_requests_at_the_openalex_maximum() -> None:
    """Split more than one hundred identifiers across several requests."""
    session = FakeSession([FakeResponse(payload={'results': []}), FakeResponse(payload={'results': []})])

    openalex.works_batch([f'10.1234/paper{index}' for index in range(101)], session=session)

    assert len(session.calls) == 2
    assert session.calls[0]['params']['per-page'] == openalex.MAX_FILTER_VALUES
    assert session.calls[1]['params']['per-page'] == 1


def test_works_batch_uses_the_identifier_filter_for_openalex_ids() -> None:
    """Key results by short identifier when filtering on OpenAlex ids."""
    session = FakeSession([FakeResponse(payload={'results': [work(doi=None)]})])

    works = openalex.works_batch(['W123'], filter_name='ids.openalex', session=session)

    assert set(works) == {'W123'}
    assert session.calls[0]['params']['filter'] == 'ids.openalex:W123'


def test_works_batch_rejects_an_invalid_batch_size() -> None:
    """Reject a non-positive batch size before issuing any request."""
    session = FakeSession([])

    with pytest.raises(ValueError):
        openalex.works_batch(['10.1/a'], batch_size=0, session=session)

    assert session.calls == []
