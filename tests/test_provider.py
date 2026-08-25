"""Unit tests for the HTTP core the data-source clients share."""

from __future__ import annotations

from typing import NoReturn
import xml.etree.ElementTree as ET

import pytest
import requests

import paperminer.arxiv as arxiv
import paperminer.biorxiv as biorxiv
import paperminer.core as core
import paperminer.elsevier as elsevier
import paperminer.medrxiv as medrxiv
import paperminer.crossref as crossref
import paperminer.openalex as openalex
import paperminer.pubmed as pubmed
from paperminer import provider
from paperminer import _rxiv

from tests.doubles import FakeResponse, FakeSession


def test_provider_wrappers_delegate_and_handle_sparse_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise shared wrappers against empty and sparse provider responses."""
    monkeypatch.setattr(_rxiv.provider, 'request_mapping', lambda *args, **kwargs: None)
    assert _rxiv.request_json(biorxiv.SERVER_CONFIG, 'url') is None
    monkeypatch.setattr(_rxiv, 'request', lambda *args, **kwargs: None)
    assert _rxiv.full_text(biorxiv.SERVER_CONFIG, {'jatsxml': 'url'}) == ''
    assert _rxiv.parse_query('""')[0] == []
    sentinel = object()
    monkeypatch.setattr(_rxiv, 'request', lambda *args, **kwargs: sentinel)
    assert biorxiv.request('url') is sentinel
    assert medrxiv.request('url') is sentinel
    entry = ET.Element(f'{{{arxiv.ATOM_NS}}}entry')
    assert arxiv._publication_date(entry) == ''
    assert arxiv.entry_to_paper(entry)['paper_id'] == ''
    assert len(arxiv.parse_entries(entry)) == 1
    monkeypatch.setattr(core, 'request_json', lambda *args, **kwargs: {'id': 'work'})
    assert core.get_work('1') == {'id': 'work'}
    assert core._first(['', None]) == ''
    monkeypatch.setattr(provider, 'request', lambda *args, **kwargs: None)
    assert provider.request_xml('url', label='test', limiter=provider.RateLimiter(1)) is None
    monkeypatch.setattr(elsevier.provider, 'request', lambda *args, **kwargs: sentinel)
    assert elsevier.request('url', 'api-key') is sentinel
    assert elsevier._link_values(1) == []


def limiter() -> provider.RateLimiter:
    """Return an unpaced limiter for tests that are not about pacing.

    Returns
    -------
    provider.RateLimiter
        Limiter that never delays.
    """
    return provider.RateLimiter(0.0)


def test_user_agent_carries_the_package_version_and_an_optional_address() -> None:
    """Build both the plain and the polite-pool user agent from one version."""
    assert provider.USER_AGENT == f'PaperMiner/{provider.__version__}'
    assert provider.user_agent() == provider.USER_AGENT
    assert provider.user_agent('me@example.com') == f'{provider.USER_AGENT} (mailto:me@example.com)'
    assert provider.default_headers() == {'User-Agent': provider.USER_AGENT}


def test_each_provider_owns_its_own_pacing_window() -> None:
    """Keep one source's courtesy delay from holding up another's requests.

    arXiv asks for three seconds between requests and PubMed allows three a
    second, so a shared window would pace every source at the slowest one.
    """
    limiters = [arxiv.LIMITER, pubmed.LIMITER, openalex.LIMITER, crossref.LIMITER]
    assert len({id(one) for one in limiters}) == len(limiters)
    assert arxiv.LIMITER.min_interval == arxiv.ARXIV_MIN_INTERVAL
    assert pubmed.LIMITER.min_interval == pubmed.NCBI_MIN_INTERVAL
    assert arxiv.LIMITER.min_interval != pubmed.LIMITER.min_interval


def test_waiting_on_one_limiter_does_not_delay_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advance only the window that was actually used."""
    clock = {'now': 100.0}
    sleeps: list[float] = []
    monkeypatch.setattr(provider.time, 'monotonic', lambda: clock['now'])
    monkeypatch.setattr(provider.time, 'sleep', sleeps.append)

    slow, fast = provider.RateLimiter(3.0), provider.RateLimiter(0.1)
    slow.wait()
    slow.wait()
    assert sleeps == [3.0]

    fast.wait()
    assert sleeps == [3.0]


def test_wait_accepts_a_per_call_interval_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let a provider whose pace depends on a credential vary it per request."""
    clock = {'now': 100.0}
    sleeps: list[float] = []
    monkeypatch.setattr(provider.time, 'monotonic', lambda: clock['now'])
    monkeypatch.setattr(provider.time, 'sleep', sleeps.append)

    paced = provider.RateLimiter(0.34)
    paced.wait()
    paced.wait(0.11)
    assert sleeps == [pytest.approx(0.11)]


def test_reset_limiters_reopens_every_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear pacing state so one caller's requests cannot delay the next."""
    clock = {'now': 100.0}
    sleeps: list[float] = []
    monkeypatch.setattr(provider.time, 'monotonic', lambda: clock['now'])
    monkeypatch.setattr(provider.time, 'sleep', sleeps.append)

    paced = provider.RateLimiter(2.0)
    paced.wait()
    provider.reset_limiters()
    paced.wait()
    assert sleeps == []


def test_request_returns_none_for_a_missing_record() -> None:
    """Read a 404 as an absent record rather than as a failure."""
    session = FakeSession([FakeResponse(status_code=404)])
    assert provider.request('https://example.test', label='Test', limiter=limiter(),
                            session=session) is None
    assert len(session.calls) == 1


def test_request_can_treat_a_missing_record_as_a_rejection() -> None:
    """Fail a 404 for a provider whose caller reads absence from the failure."""
    session = FakeSession([FakeResponse(status_code=404)])
    with pytest.raises(RuntimeError, match='Test rejected the request with 404'):
        provider.request('https://example.test', label='Test', limiter=limiter(),
                         session=session, missing_ok=False)
    assert len(session.calls) == 1


def test_request_fails_at_once_on_a_client_error_other_than_a_rate_limit() -> None:
    """Spend one attempt on a terminal client error instead of retrying it."""
    session = FakeSession([FakeResponse(status_code=400)])
    with pytest.raises(RuntimeError, match='Test rejected the request with 400'):
        provider.request('https://example.test', label='Test', limiter=limiter(),
                         session=session)
    assert len(session.calls) == 1


def test_request_does_not_retry_a_client_error_raised_by_the_session() -> None:
    """Treat an HTTP error carrying a terminal 4xx response like a returned 4xx."""
    response = FakeResponse(status_code=400)
    error = requests.HTTPError('bad request', response=response)

    class RaisingSession:
        """Raise the prepared HTTP error and count calls."""

        def __init__(self) -> None:
            """Start with no calls."""
            self.calls = 0

        def get(self, *args: object, **kwargs: object) -> NoReturn:
            """Raise the terminal response error."""
            self.calls += 1
            raise error

    session = RaisingSession()
    with pytest.raises(RuntimeError, match='failed after'):
        provider.request(
            'https://example.test', label='Test', limiter=limiter(), session=session,
        )
    assert session.calls == 1


def test_request_retries_a_rate_limited_response_and_honours_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait the advertised interval and retry rather than giving up on a 429."""
    sleeps: list[float] = []
    monkeypatch.setattr(provider.time, 'sleep', sleeps.append)
    session = FakeSession([
        FakeResponse(status_code=429, headers={'Retry-After': '5'}),
        FakeResponse(text='ok'),
    ])

    assert provider.request('https://example.test', label='Test', limiter=limiter(),
                            session=session) is not None
    assert sleeps == [5.0]
    assert len(session.calls) == 2


def test_request_clamps_and_ignores_an_unusable_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to backoff for a header that is absent, unparsable, or absurd."""
    sleeps: list[float] = []
    monkeypatch.setattr(provider.time, 'sleep', sleeps.append)

    for header, expected in [({'Retry-After': 'soon'}, 1.0),
                             ({'Retry-After': '86400'}, provider.MAX_RETRY_DELAY),
                             ({}, 1.0)]:
        sleeps.clear()
        session = FakeSession([FakeResponse(status_code=503, headers=header),
                               FakeResponse(text='ok')])
        assert provider.request('https://example.test', label='Test', limiter=limiter(),
                                session=session) is not None
        assert sleeps == [expected]


def test_request_reports_how_many_attempts_were_spent() -> None:
    """Name the attempt count so an exhausted request is not mistaken for one."""
    session = FakeSession([FakeResponse(status_code=500) for _ in range(4)])
    with pytest.raises(RuntimeError, match='Test request failed after 4 attempts'):
        provider.request('https://example.test', label='Test', limiter=limiter(),
                         session=session)
    assert len(session.calls) == 4


def test_request_consults_the_terminal_hook_before_the_shared_rule() -> None:
    """Let a provider fail a status the shared rule would have retried."""
    def budget_gone(response: provider.ResponseLike) -> str:
        """Treat a rate limit as an exhausted budget.

        Parameters
        ----------
        response : provider.ResponseLike
            Response to classify.

        Returns
        -------
        str
            Failure message for a 429, or an empty string otherwise.
        """
        return 'budget exhausted' if response.status_code == 429 else ''

    session = FakeSession([FakeResponse(status_code=429)])
    with pytest.raises(RuntimeError, match='budget exhausted'):
        provider.request('https://example.test', label='Test', limiter=limiter(),
                         session=session, client_error=budget_gone)
    assert len(session.calls) == 1


def test_request_accepts_extra_error_types_as_retryable() -> None:
    """Retry a decode failure a provider counts as a failed attempt."""
    class Broken(FakeResponse):
        """Response whose body cannot be read as the caller expects."""

        def json(self) -> dict[str, str]:
            """Fail the way a truncated payload would.

            Returns
            -------
            dict[str, str]
                Never returned.

            Raises
            ------
            KeyError
                Always.
            """
            raise KeyError('message')

    session = FakeSession([Broken(), FakeResponse(payload={'ok': True})])
    with pytest.raises(KeyError):
        provider.request_payload('https://example.test', label='Test', limiter=limiter(),
                                 session=session).get('message')


def test_request_payload_reports_a_malformed_body_with_the_provider_name() -> None:
    """Name the provider in a decode failure so the source is unambiguous."""
    session = FakeSession([FakeResponse(text='not json')])
    with pytest.raises(RuntimeError, match='Test returned malformed JSON'):
        provider.request_payload('https://example.test', label='Test', limiter=limiter(),
                                 session=session)


def test_request_payload_prefers_a_provider_specific_decode_message() -> None:
    """Let a provider explain a body that is not JSON for a known reason."""
    def challenge(response: provider.ResponseLike, error: ValueError) -> str:
        """Recognize a challenge page served in place of JSON.

        Parameters
        ----------
        response : provider.ResponseLike
            Response whose body failed to decode.
        error : ValueError
            The decoding failure.

        Returns
        -------
        str
            Failure message when the body is HTML, empty otherwise.
        """
        return 'challenge page' if response.text.startswith('<') else ''

    session = FakeSession([FakeResponse(text='<html></html>')])
    with pytest.raises(RuntimeError, match='challenge page'):
        provider.request_payload('https://example.test', label='Test', limiter=limiter(),
                                 session=session, decode_error=challenge)


def test_request_mapping_rejects_a_payload_that_is_not_an_object() -> None:
    """Fail a list where an object was expected rather than passing it on."""
    session = FakeSession([FakeResponse(payload=[1, 2, 3])])
    with pytest.raises(RuntimeError, match='unexpected payload of type list'):
        provider.request_mapping('https://example.test', label='Test', limiter=limiter(),
                                 session=session)


def test_request_xml_skips_an_empty_body_and_reports_a_malformed_one() -> None:
    """Read a blank body as nothing found and a broken one as a failure."""
    session = FakeSession([FakeResponse(text='   ')])
    assert provider.request_xml('https://example.test', label='Test', limiter=limiter(),
                                session=session) is None

    session = FakeSession([FakeResponse(text='<feed>')])
    with pytest.raises(RuntimeError, match='Test returned malformed XML'):
        provider.request_xml('https://example.test', label='Test', limiter=limiter(),
                             session=session)


def test_request_defaults_to_the_requests_module_when_no_session_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to a plain request so a caller need not supply a client."""
    seen: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        """Record the request and answer it.

        Parameters
        ----------
        url : str
            Endpoint requested.
        **kwargs : object
            Request arguments.

        Returns
        -------
        FakeResponse
            A successful response.
        """
        seen.update({'url': url, **kwargs})
        return FakeResponse(text='ok')

    monkeypatch.setattr(requests, 'get', fake_get)
    assert provider.request('https://example.test', label='Test', limiter=limiter()) is not None
    assert seen['url'] == 'https://example.test'
    assert seen['headers'] == provider.default_headers()


def test_chunked_splits_a_sequence_and_tolerates_a_zero_size() -> None:
    """Batch values in order without dropping any or looping forever."""
    assert list(provider.chunked(['a', 'b', 'c'], 2)) == [['a', 'b'], ['c']]
    assert list(provider.chunked([], 5)) == []
    assert list(provider.chunked(['a', 'b'], 0)) == [['a'], ['b']]


def test_clean_text_collapses_whitespace_and_drops_placeholders() -> None:
    """Report a provider's absent-value placeholder as empty rather than as text."""
    assert provider.clean_text('  many   spaces \n here ') == 'many spaces here'
    assert provider.clean_text(None) == ''
    for placeholder in ['NA', 'n/a', 'None', 'null', '']:
        assert provider.clean_text(placeholder) == ''
