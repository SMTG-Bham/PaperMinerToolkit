"""Unit tests for Crossref author discovery and corpus import."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd
import pytest
import requests

import paperminer.corpus.database as corpus
import paperminer.providers.crossref as crossref
import paperminer.workflows.enrichment as enrichment
from paperminer.providers import base as provider

from tests.doubles import FakeResponse


def test_author_validation_and_safe_paging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate author options and stop safely on duplicate records and cursors."""
    assert not crossref._given_names_match('', 'Jane')
    assert not crossref._given_names_match('Jane', 'Janet')
    with pytest.raises(ValueError, match='given and family'):
        crossref._matching_authors({}, 'Madonna')
    assert crossref._author_orcid({'ORCID': 'invalid'}) == ''
    monkeypatch.setattr(crossref.provider, 'request', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='failed after'):
        crossref._request_page(None, {}, 'person@example.org')
    for kwargs, message in [({}, 'exactly one'), ({'author_name': 'Jane Smith', 'email': ''}, 'contact email'), ({'author_name': 'Jane Smith', 'email': 'a@b.com', 'max_results': 0}, 'positive'), ({'author_name': 'Jane Smith', 'email': 'a@b.com', 'page_size': 0}, 'between')]:
        with pytest.raises(ValueError, match=message):
            crossref.author_works(**kwargs)
    monkeypatch.setattr(crossref, 'work_matches_author', lambda *args, **kwargs: True)
    monkeypatch.setattr(crossref, '_request_page', lambda *args, **kwargs: {'items': [{'DOI': ''}, {'DOI': '10.1/A'}, {'DOI': '10.1/A'}], 'next-cursor': '*'})
    assert len(crossref.author_works(author_name='Ada Lovelace', email='ada@example.org', page_size=3)) == 1
    monkeypatch.setattr(crossref, '_request_page', lambda *args, **kwargs: {'items': [{'DOI': '10.1/A'}]})
    assert len(crossref.author_works(author_name='Ada Lovelace', email='ada@example.org', max_results=1)) == 1


def work(
    doi: str,
    given: str = 'Jane A.',
    family: str = 'Smith',
    orcid: str = '0000-0001-2345-6789',
    affiliation: str = 'Example University',
) -> dict[str, Any]:
    """Return a minimal Crossref work record."""
    return {
        'DOI': doi,
        'title': [f'Title for {doi}'],
        'container-title': ['Test Journal'],
        'published-online': {'date-parts': [[2024, 3, 7]]},
        'author': [{
            'given': given,
            'family': family,
            'ORCID': f'https://orcid.org/{orcid}',
            'affiliation': [{'name': affiliation}],
        }],
    }


class FakeSession:
    """Return prepared Crossref pages and record request arguments.

    Crossref answers every route with a ``message`` envelope, so this takes the
    messages themselves and wraps each one, keeping the tests readable.

    Parameters
    ----------
    messages : Iterable[dict[str, Any]]
        Crossref messages to hand out, one per ``get`` call.
    """

    def __init__(self, messages: Iterable[dict[str, Any]]) -> None:
        """Store the prepared messages.

        Parameters
        ----------
        messages : Iterable[dict[str, Any]]
            Crossref messages to hand out, one per ``get`` call.

        Returns
        -------
        None
            The double is initialized in place.
        """
        self.messages = iter(messages)
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: dict[str, str | int],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        """Record a GET request and return the next prepared response.

        Parameters
        ----------
        url : str
            Endpoint the client asked for.
        params : dict[str, str or int]
            Query parameters the client sent.
        headers : dict[str, str]
            Headers the client sent.
        timeout : float
            Timeout the client asked for.

        Returns
        -------
        FakeResponse
            The next prepared message, wrapped in a Crossref envelope.
        """
        self.calls.append({
            'url': url,
            'params': params.copy(),
            'headers': headers.copy(),
            'timeout': timeout,
        })
        return FakeResponse(payload={'message': next(self.messages)})



def test_normalize_orcid_accepts_urls_and_rejects_invalid_values() -> None:
    """Normalize canonical ORCID URLs without accepting malformed identifiers."""
    assert crossref.normalize_orcid('https://orcid.org/0000-0001-2345-6789') == '0000-0001-2345-6789'
    with pytest.raises(ValueError, match='Invalid ORCID'):
        crossref.normalize_orcid('1234')
    with pytest.raises(ValueError, match='checksum'):
        crossref.normalize_orcid('0000-0001-2345-678X')


def test_request_page_ignores_malformed_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to exponential backoff when Retry-After is not numeric."""
    response = requests.Response()
    response.status_code = 429
    response.headers['Retry-After'] = 'not-a-number'

    class RetrySession:
        """Fail the first request before returning a successful response."""

        def __init__(self) -> None:
            """Initialize the request counter."""
            self.calls = 0

        def get(self, *args: object, **kwargs: object) -> FakeResponse:
            """Raise once, then return an empty successful response."""
            self.calls += 1
            if self.calls == 1:
                raise requests.HTTPError('rate limited', response=response)
            return FakeResponse(payload={'message': {'items': []}})

    sleeps = []
    monkeypatch.setattr(provider.time, 'sleep', sleeps.append)
    message = crossref._request_page(RetrySession(), {}, 'person@example.ac.uk', attempts=2)

    assert message == {'items': []}
    assert 1 in sleeps


def test_work_matches_exact_orcid_or_conservative_name_and_affiliation() -> None:
    """Match exact identities while rejecting different names and affiliations."""
    record = work('10.1/example')
    assert crossref.work_matches_author(record, orcid='0000-0001-2345-6789')
    assert crossref.work_matches_author(record, author_name='Jane A Smith')
    assert crossref.work_matches_author(record, author_name='J A Smith')
    assert crossref.work_matches_author(record, author_name='Jane A Smith', affiliation='Example')
    assert not crossref.work_matches_author(record, author_name='James Smith')
    assert not crossref.work_matches_author(record, author_name='Jane A Smith', affiliation='Elsewhere')


def test_author_works_uses_orcid_filter_polite_pool_and_cursor_pagination() -> None:
    """Retrieve all unique DOI records over multiple Crossref cursor pages."""
    session = FakeSession([
        {'items': [work('10.1/one'), work('10.1/two')], 'next-cursor': 'next'},
        {'items': [work('10.1/three')], 'next-cursor': 'unused'},
    ])

    records = crossref.author_works(
        orcid='0000-0001-2345-6789',
        email='person@example.ac.uk',
        page_size=2,
        session=session,
    )

    assert [record['DOI'] for record in records] == ['10.1/one', '10.1/two', '10.1/three']
    assert session.calls[0]['params']['filter'] == 'orcid:0000-0001-2345-6789'
    assert session.calls[0]['params']['cursor'] == '*'
    assert session.calls[1]['params']['cursor'] == 'next'
    assert session.calls[0]['params']['mailto'] == 'person@example.ac.uk'
    assert 'mailto:person@example.ac.uk' in session.calls[0]['headers']['User-Agent']


def test_author_works_name_query_filters_false_positive_candidates() -> None:
    """Post-filter Crossref's fuzzy author query before accepting records."""
    session = FakeSession([{
        'items': [work('10.1/right'), work('10.1/wrong', given='James')],
        'next-cursor': 'unused',
    }])

    records = crossref.author_works(
        author_name='Jane A Smith',
        email='person@example.ac.uk',
        page_size=10,
        session=session,
    )

    assert [record['DOI'] for record in records] == ['10.1/right']
    assert session.calls[0]['params']['query.author'] == 'Jane A Smith'


def test_import_author_works_writes_review_csv_and_corpus(tmp_path: Path) -> None:
    """Map discovered works into both a reviewable CSV and SQLite corpus."""
    session = FakeSession([{'items': [work('10.1/one')], 'next-cursor': 'unused'}])
    db_path = tmp_path / 'supervisor.db'
    review_path = tmp_path / 'review' / 'supervisor_works.csv'

    summary = crossref.import_author_works(
        db_path,
        email='person@example.ac.uk',
        orcid='0000-0001-2345-6789',
        review_csv=review_path,
        session=session,
    )

    with corpus.connect(db_path) as connection:
        papers = corpus.paper_rows(connection)
    review = pd.read_csv(review_path)
    assert summary == {'found': 1, 'added': 1, 'updated': 0, 'enriched': 0}
    assert papers[0]['paper_id'] == 'doi:10.1/one'
    assert papers[0]['title'] == 'Title for 10.1/one'
    assert papers[0]['publication_date'] == '2024-03-07'
    assert papers[0]['metadata_status'] == 'retrieved'
    assert json.loads(papers[0]['metadata_json'])['DOI'] == '10.1/one'
    assert review['doi'].tolist() == ['10.1/one']


def test_import_author_works_tolerates_a_lost_match_and_can_enrich(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue after a reconciliation miss and report optional enrichment counts."""
    monkeypatch.setattr(crossref, 'author_works', lambda **kwargs: [work('10.1/one')])
    monkeypatch.setattr(crossref, 'find_paper', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        enrichment, 'enrich_papers',
        lambda *args, **kwargs: {'succeeded': 1},
    )
    summary = crossref.import_author_works(
        tmp_path / 'papers.db', email='person@example.org',
        author_name='Jane Smith', enrich=True,
    )
    assert summary == {'found': 1, 'added': 1, 'updated': 0, 'enriched': 1}


def test_configured_email_reads_the_stored_setting() -> None:
    """Read the Crossref contact email from a supplied settings mapping."""
    assert crossref.configured_email({'crossref_email': 'me@example.com'}) == 'me@example.com'
    assert crossref.configured_email({}) == ''


def test_resolve_email_prefers_an_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefer an explicit address, fall back to settings, then fail clearly."""
    monkeypatch.setattr(crossref, 'configured_email', lambda settings=None: 'stored@example.com')
    assert crossref.resolve_email('explicit@example.com') == 'explicit@example.com'
    assert crossref.resolve_email(None) == 'stored@example.com'

    monkeypatch.setattr(crossref, 'configured_email', lambda settings=None: '')
    with pytest.raises(ValueError, match='pm config crossref-email'):
        crossref.resolve_email(None)


def test_request_page_accepts_an_alternate_url() -> None:
    """Send a request to the single-work route through the shared retry loop."""
    session = FakeSession([work('10.1234/one')])

    message = crossref._request_page(session, {'mailto': 'me@example.com'},
                                     'me@example.com', url='https://api.crossref.org/works/10.1234%2Fone')

    assert message['DOI'] == '10.1234/one'
    assert session.calls[0]['url'].endswith('10.1234%2Fone')


def test_works_by_doi_builds_a_comma_filter_with_rows_and_mailto() -> None:
    """Batch DOIs into one filter with a row count covering every value."""
    session = FakeSession([{'items': [work('10.1234/two'), work('10.1234/one')]}])

    works = crossref.works_by_doi(['10.1234/one', '10.1234/two'],
                                  email='me@example.com', session=session, pace=0)

    params = session.calls[0]['params']
    assert params['filter'] == 'doi:10.1234/one,doi:10.1234/two'
    assert params['rows'] == 2
    assert params['mailto'] == 'me@example.com'
    assert set(works) == {'10.1234/one', '10.1234/two'}


def test_works_by_doi_sends_no_select_so_language_is_returned() -> None:
    """Omit select entirely because language is returned but not selectable."""
    session = FakeSession([{'items': []}])

    crossref.works_by_doi(['10.1234/one'], email='me@example.com', session=session, pace=0)

    assert 'select' not in session.calls[0]['params']


def test_works_by_doi_routes_comma_bearing_dois_to_the_single_work_route() -> None:
    """Keep DOIs containing the filter separator out of the batch request."""
    session = FakeSession([{'items': [work('10.1234/plain')]}, work('10.1234/with,comma')])

    works = crossref.works_by_doi(['10.1234/plain', '10.1234/with,comma'],
                                  email='me@example.com', session=session, pace=0)

    assert session.calls[0]['params']['filter'] == 'doi:10.1234/plain'
    assert session.calls[1]['url'].endswith('10.1234%2Fwith%2Ccomma')
    assert set(works) == {'10.1234/plain', '10.1234/with,comma'}


def test_works_by_doi_sleeps_between_consecutive_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pace consecutive Crossref requests without sleeping before the first."""
    sleeps = []
    monkeypatch.setattr(provider.time, 'sleep', sleeps.append)
    session = FakeSession([{'items': []}, {'items': []}])

    crossref.works_by_doi(['10.1234/one', '10.1234/two'], email='me@example.com',
                          session=session, batch_size=1)

    assert sleeps == [pytest.approx(crossref.CROSSREF_MIN_INTERVAL, abs=1e-3)]


def test_works_by_doi_rejects_an_invalid_batch_size() -> None:
    """Reject a batch size outside the supported range before requesting."""
    session = FakeSession([])

    with pytest.raises(ValueError):
        crossref.works_by_doi(['10.1234/one'], email='me@example.com',
                              session=session, batch_size=0)

    assert session.calls == []


def test_work_by_doi_returns_none_for_an_unknown_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return None when the single-work route exhausts its retries."""
    def failing(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail as an exhausted Crossref request would."""
        raise RuntimeError('Crossref request failed after 4 attempts: 404')

    monkeypatch.setattr(crossref, '_request_page', failing)

    assert crossref.work_by_doi('10.1234/missing', email='me@example.com',
                                session=FakeSession([])) is None


def test_work_by_doi_ignores_an_empty_doi() -> None:
    """Skip the request entirely when no usable DOI is supplied."""
    session = FakeSession([])

    assert crossref.work_by_doi('', email='me@example.com', session=session) is None
    assert session.calls == []


def test_request_page_does_not_retry_a_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give up immediately on a deterministic 4xx rather than retrying it."""
    sleeps: list[float] = []
    monkeypatch.setattr(provider.time, 'sleep', sleeps.append)

    class BadRequestResponse:
        """Reject every request with an HTTP 400."""

        status_code = 400
        headers: dict[str, str] = {}

        def raise_for_status(self) -> NoReturn:
            """Raise the client error Crossref returns for a bad filter."""
            raise requests.HTTPError('400 Client Error', response=self)

        def json(self) -> dict[str, Any]:
            """Never reached for a failing response."""
            return {}

    class BadSession:
        """Return an HTTP 400 for every request and count the attempts."""

        def __init__(self) -> None:
            """Start with no recorded attempts."""
            self.attempts = 0

        def get(self, url: str, *, params: Any, headers: Any, timeout: float) -> BadRequestResponse:
            """Record the attempt and return the failing response."""
            self.attempts += 1
            return BadRequestResponse()

    session = BadSession()
    with pytest.raises(RuntimeError):
        crossref._request_page(session, {'mailto': 'me@example.com'}, 'me@example.com')

    assert session.attempts == 1
    assert sleeps == []


def test_works_by_doi_routes_unusable_dois_away_from_the_batch() -> None:
    """Keep a malformed DOI out of the filter that Crossref would reject."""
    session = FakeSession([{'items': [work('10.1234/valid')]}, work('10.1/bad')])

    crossref.works_by_doi(['10.1234/valid', '10.1/bad'],
                          email='me@example.com', session=session, pace=0)

    assert session.calls[0]['params']['filter'] == 'doi:10.1234/valid'
    assert session.calls[1]['url'].endswith('10.1%2Fbad')


def test_works_by_doi_falls_back_to_single_lookups_when_a_batch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover the whole chunk individually when Crossref rejects the filter."""
    calls: list[dict[str, Any]] = []

    def flaky_request_page(session: Any, params: Any, email: str, url: str = '',
                           **kwargs: Any) -> dict[str, Any]:
        """Fail the batched filter but answer the single-work route."""
        calls.append({'params': dict(params), 'url': url})
        if 'filter' in params:
            raise RuntimeError('Crossref request failed after 4 attempts: 400')
        return work('10.1234/one')

    monkeypatch.setattr(crossref, '_request_page', flaky_request_page)

    works = crossref.works_by_doi(['10.1234/one'], email='me@example.com',
                                  session=FakeSession([]), pace=0)

    assert set(works) == {'10.1234/one'}
    assert len(calls) == 2
