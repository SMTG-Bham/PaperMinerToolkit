"""Unit tests for Crossref author discovery and corpus import."""

import json
import pandas as pd
import pytest
import requests

import paperscraper.corpus as corpus
import paperscraper.crossref as crossref


def work(doi, given='Jane A.', family='Smith', orcid='0000-0001-2345-6789', affiliation='Example University'):
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


class FakeResponse:
    """Successful requests response containing one prepared Crossref message."""

    def __init__(self, message):
        """Store a prepared Crossref response message."""
        self.message = message

    def raise_for_status(self):
        """Represent a successful HTTP status check."""
        return None

    def json(self):
        """Return the prepared message as a response payload."""
        return {'message': self.message}


class FakeSession:
    """Return prepared Crossref pages and record request arguments."""

    def __init__(self, messages):
        """Initialize the session with prepared response messages."""
        self.messages = iter(messages)
        self.calls = []

    def get(self, url, params, headers, timeout):
        """Record a GET request and return the next prepared response."""
        self.calls.append({
            'url': url,
            'params': params.copy(),
            'headers': headers.copy(),
            'timeout': timeout,
        })
        return FakeResponse(next(self.messages))


def test_normalize_orcid_accepts_urls_and_rejects_invalid_values():
    """Normalize canonical ORCID URLs without accepting malformed identifiers."""
    assert crossref.normalize_orcid('https://orcid.org/0000-0001-2345-6789') == '0000-0001-2345-6789'
    with pytest.raises(ValueError, match='Invalid ORCID'):
        crossref.normalize_orcid('1234')
    with pytest.raises(ValueError, match='checksum'):
        crossref.normalize_orcid('0000-0001-2345-678X')


def test_request_page_ignores_malformed_retry_after(monkeypatch):
    """Fall back to exponential backoff when Retry-After is not numeric."""
    response = requests.Response()
    response.status_code = 429
    response.headers['Retry-After'] = 'not-a-number'

    class RetrySession:
        """Fail the first request before returning a successful response."""

        def __init__(self):
            """Initialize the request counter."""
            self.calls = 0

        def get(self, *args, **kwargs):
            """Raise once, then return an empty successful response."""
            self.calls += 1
            if self.calls == 1:
                raise requests.HTTPError('rate limited', response=response)
            return FakeResponse({'items': []})

    sleeps = []
    monkeypatch.setattr(crossref.time, 'sleep', sleeps.append)
    message = crossref._request_page(RetrySession(), {}, 'person@example.ac.uk', attempts=2)

    assert message == {'items': []}
    assert sleeps == [1]


def test_work_matches_exact_orcid_or_conservative_name_and_affiliation():
    """Match exact identities while rejecting different names and affiliations."""
    record = work('10.1/example')
    assert crossref.work_matches_author(record, orcid='0000-0001-2345-6789')
    assert crossref.work_matches_author(record, author_name='Jane A Smith')
    assert crossref.work_matches_author(record, author_name='J A Smith')
    assert crossref.work_matches_author(record, author_name='Jane A Smith', affiliation='Example')
    assert not crossref.work_matches_author(record, author_name='James Smith')
    assert not crossref.work_matches_author(record, author_name='Jane A Smith', affiliation='Elsewhere')


def test_author_works_uses_orcid_filter_polite_pool_and_cursor_pagination():
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


def test_author_works_name_query_filters_false_positive_candidates():
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


def test_import_author_works_writes_review_csv_and_corpus(tmp_path):
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
    assert summary == {'found': 1, 'added': 1, 'updated': 0}
    assert papers[0]['paper_id'] == 'doi:10.1/one'
    assert papers[0]['title'] == 'Title for 10.1/one'
    assert papers[0]['publication_date'] == '2024-03-07'
    assert papers[0]['metadata_status'] == 'retrieved'
    assert json.loads(papers[0]['metadata_json'])['DOI'] == '10.1/one'
    assert review['doi'].tolist() == ['10.1/one']
