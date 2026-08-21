"""Unit tests for paperscraper.search.

This module tests provider-specific search helpers, CORE and Elsevier row
normalization, pagination behavior, request headers, and merging search results
into the paper corpus.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import pandas as pd
import pytest

import paperscraper.corpus as corpus
import paperscraper.search as search


def test_elsevier_api_key_requires_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elsevier API key requires configured API key."""
    monkeypatch.setattr(search, 'load_settings', lambda: {})

    with pytest.raises(ValueError, match='Elsevier API key is not configured'):
        search._elsevier_api_key()


def test_elsevier_api_key_returns_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elsevier API key returns configured API key."""
    monkeypatch.setattr(search, 'load_settings', lambda: {'elsevier_api_key': 'elsevier-key'})

    assert search._elsevier_api_key() == 'elsevier-key'


def test_document_search_caps_count_and_paginates_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Document search caps count and paginates results."""
    urls = []

    def fake_get_json(api_key: str, url: str) -> dict[str, Any]:
        """Provide fake JSON retrieval for this test."""
        assert api_key == 'elsevier-key'
        urls.append(url)
        if len(urls) == 1:
            return {
                'search-results': {
                    'opensearch:totalResults': '3',
                    'entry': [{'dc:title': 'first'}],
                    'link': [{'@ref': 'next', '@href': 'next-url'}],
                }
            }
        return {
            'search-results': {
                'opensearch:totalResults': '3',
                'entry': [{'dc:title': 'second'}, {'dc:title': 'third'}],
                'link': [],
            }
        }

    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the test double."""
            self.updates = []

        def __enter__(self) -> Self:
            """Enter the test-double context."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Exit the test-double context."""
            return False

        def update(self, value: int) -> None:
            """Record a progress update."""
            self.updates.append(value)

    monkeypatch.setattr(search, '_elsevier_api_key', lambda: 'elsevier-key')
    monkeypatch.setattr(search.elsevier, 'get_json', fake_get_json)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    results = search.document_search('solid electrolyte', count=2, get_all=True)

    output = capsys.readouterr().out
    assert results['dc:title'].tolist() == ['first', 'second']
    assert 'retrieving 2 of 3 results' in output
    assert '&count=2' in urls[0]
    assert urls[1] == 'next-url'


def test_document_search_returns_empty_dataframe_for_zero_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document search returns empty dataframe for zero results."""
    monkeypatch.setattr(search, '_elsevier_api_key', lambda: 'elsevier-key')
    monkeypatch.setattr(
        search.elsevier,
        'get_json',
        lambda *_: {'search-results': {'opensearch:totalResults': '0', 'entry': [], 'link': []}},
    )

    assert search.document_search('missing').empty


def test_document_search_without_get_all_returns_first_page_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document search without get all returns first page slice."""
    monkeypatch.setattr(search, '_elsevier_api_key', lambda: 'elsevier-key')
    monkeypatch.setattr(
        search.elsevier,
        'get_json',
        lambda *_: {
            'search-results': {
                'opensearch:totalResults': '3',
                'entry': [{'dc:title': 'first'}, {'dc:title': 'second'}, {'dc:title': 'third'}],
                'link': [],
            }
        },
    )

    results = search.document_search('solid electrolyte', count=2, get_all=False)

    assert results['dc:title'].tolist() == ['first', 'second']


def test_document_search_stops_when_next_link_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document search stops when next link is missing."""
    calls = []

    def fake_get_json(*_: object) -> dict[str, Any]:
        """Provide fake JSON retrieval for this test."""
        calls.append(True)
        return {
            'search-results': {
                'opensearch:totalResults': '3',
                'entry': [{'dc:title': 'first'}],
                'link': [],
            }
        }

    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the test double."""
            return None

        def __enter__(self) -> Self:
            """Enter the test-double context."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Exit the test-double context."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
            return None

    monkeypatch.setattr(search, '_elsevier_api_key', lambda: 'elsevier-key')
    monkeypatch.setattr(search.elsevier, 'get_json', fake_get_json)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    results = search.document_search('solid electrolyte', count=3, get_all=True)

    assert results['dc:title'].tolist() == ['first']
    assert len(calls) == 1


def test_document_search_stops_non_scopus_searches_at_provider_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document search stops non-Scopus searches at the provider limit."""
    first_page = [{'dc:title': f'paper {index}'} for index in range(4999)]

    calls = []

    def fake_get_json(*_: object) -> dict[str, Any]:
        """Provide fake JSON retrieval for this test."""
        calls.append(True)
        if len(calls) == 1:
            return {
                'search-results': {
                    'opensearch:totalResults': '6000',
                    'entry': first_page,
                    'link': [{'@ref': 'next', '@href': 'next-url'}],
                }
            }
        return {
            'search-results': {
                'opensearch:totalResults': '6000',
                'entry': [{'dc:title': 'paper 4999'}, {'dc:title': 'paper 5000'}],
                'link': [{'@ref': 'next', '@href': 'unused-next-url'}],
            }
        }

    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the test double."""
            return None

        def __enter__(self) -> Self:
            """Enter the test-double context."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Exit the test-double context."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
            return None

    monkeypatch.setattr(search, '_elsevier_api_key', lambda: 'elsevier-key')
    monkeypatch.setattr(search.elsevier, 'get_json', fake_get_json)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    results = search.document_search('solid electrolyte', index='article', count=6000, get_all=True)

    assert len(results) == 5001
    assert len(calls) == 2


def test_first_returns_first_list_item_or_scalar_value() -> None:
    """First returns first list item or scalar value."""
    assert search._first(['10.1234/example']) == '10.1234/example'
    assert search._first([]) == ''
    assert search._first('value') == 'value'
    assert search._first(None) == ''


def test_elsevier_rows_normalizes_provider_records() -> None:
    """Elsevier rows normalize provider records."""
    raw = pd.DataFrame([{
        'dc:identifier': 'SCOPUS_ID:1',
        'prism:doi': '10.1234/example',
        'dc:title': 'Paper title',
        'prism:publicationName': 'Journal',
        'prism:coverDate': '2024-01-01',
        'dc:creator': 'Author',
        'dc:description': '<p>Elsevier abstract</p>',
        'link': [
            {'@ref': 'self', '@href': 'self-link'},
            {'@ref': 'full-text', '@href': 'full-text-link'},
        ],
    }])

    rows = search._elsevier_rows(raw)
    row = rows.iloc[0]

    assert row['paper_id'] == 'SCOPUS_ID:1'
    assert row['doi'] == '10.1234/example'
    assert row['title'] == 'Paper title'
    assert row['journal'] == 'Journal'
    assert row['authors'] == 'Author'
    assert row['sources'] == 'elsevier'
    assert row['elsevier_link'] == 'full-text-link'
    assert row['abstract'] == 'Elsevier abstract'
    assert row['metadata_status'] == 'retrieved'


def test_recast_elsevier_records_preserves_link_lists_for_full_text_selection() -> None:
    """Recasting Elsevier records preserves links used for full-text selection."""
    recast = search._recast_elsevier_records([{
        'prism:doi': ['10.1234/example'],
        'link': [
            {'@ref': 'self', '@href': 'self-link'},
            {'@ref': 'full-text', '@href': 'full-text-link'},
        ],
    }])
    rows = search._elsevier_rows(recast)

    assert recast.loc[0, 'prism:doi'] == '10.1234/example'
    assert rows.loc[0, 'elsevier_link'] == 'full-text-link'


def test_core_headers_include_optional_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    """CORE headers include optional authorization."""
    monkeypatch.setattr(search, '_core_api_key', lambda: None)
    assert search._core_headers() == {'User-Agent': 'PaperScraper/0.0.1'}

    monkeypatch.setattr(search, '_core_api_key', lambda: 'core-key')
    assert search._core_headers()['Authorization'] == 'Bearer core-key'


def test_core_field_helpers_extract_download_authors_journal_and_date() -> None:
    """CORE field helpers extract download authors journal and date."""
    work = {
        'id': '123',
        'authors': [{'name': 'A. Author'}, {'fullName': 'B. Author'}, 'C. Author', {'name': ''}],
        'journal': {'title': 'Journal Title'},
        'year': 2024,
    }

    assert search._core_download_url({'downloadUrl': 'https://example.com/pdf'}) == 'https://example.com/pdf'
    assert search._core_download_url(work) == 'https://api.core.ac.uk/v3/works/123/download'
    assert search._core_download_url({}) == ''
    assert search._core_authors(work) == 'A. Author; B. Author; C. Author'
    assert search._core_journal(work) == 'Journal Title'
    assert search._core_journal({'publisher': 'Publisher'}) == 'Publisher'
    assert search._core_date(work) == 2024


def test_core_rows_normalizes_work_records() -> None:
    """CORE rows normalize work records."""
    rows = search._core_rows([
        {
            'id': '123',
            'doi': ['10.1234/core'],
            'title': ['Core title'],
            'journal': {'name': 'Core Journal'},
            'publishedDate': '2023-01-01',
            'authors': [{'name': 'A. Author'}],
            'abstract': '<p>Core abstract</p>',
        },
        {
            'DOI': '10.1234/no-id',
            'title': 'No ID title',
        },
    ])

    assert rows.loc[0, 'paper_id'] == 'core:123'
    assert rows.loc[0, 'doi'] == '10.1234/core'
    assert rows.loc[0, 'title'] == 'Core title'
    assert rows.loc[0, 'journal'] == 'Core Journal'
    assert rows.loc[0, 'pdf_url'] == 'https://api.core.ac.uk/v3/works/123/download'
    assert rows.loc[0, 'abstract'] == 'Core abstract'
    assert rows.loc[0, 'metadata_status'] == 'retrieved'
    assert rows.loc[1, 'paper_id'] == 'doi:10.1234/no-id'


def test_openalex_rows_normalizes_work_records() -> None:
    """OpenAlex rows normalize work records."""
    rows = search._openalex_rows([
        {
            'id': 'https://openalex.org/W123',
            'doi': 'https://doi.org/10.1234/OpenAlex',
            'title': 'OpenAlex title',
            'publication_date': '2024-03-07',
            'authorships': [{'author': {'display_name': 'A. Author'}}],
            'primary_location': {'source': {'display_name': 'OpenAlex Journal'}},
            'best_oa_location': {'pdf_url': 'https://example.org/paper.pdf'},
            'abstract_inverted_index': {'Second': [1], 'First': [0]},
        },
        {
            'id': 'https://openalex.org/W456',
            'title': 'No DOI title',
        },
    ])

    assert list(rows.columns) == search.SEARCH_FIELDS
    assert rows.loc[0, 'paper_id'] == 'doi:10.1234/openalex'
    assert rows.loc[0, 'doi'] == '10.1234/openalex'
    assert rows.loc[0, 'title'] == 'OpenAlex title'
    assert rows.loc[0, 'journal'] == 'OpenAlex Journal'
    assert rows.loc[0, 'authors'] == 'A. Author'
    assert rows.loc[0, 'pdf_url'] == 'https://example.org/paper.pdf'
    assert rows.loc[0, 'abstract'] == 'First Second'
    assert rows.loc[0, 'sources'] == 'openalex'
    assert rows.loc[0, 'metadata_status'] == 'retrieved'
    assert rows.loc[1, 'paper_id'] == 'openalex:W456'
    assert rows.loc[1, 'abstract'] == ''


def test_core_search_paginates_and_stops_at_total_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    """CORE search paginates and stops at total hits."""
    class FakeResponse:
        """Provide a response test double."""

        def __init__(self, payload: dict[str, Any]) -> None:
            """Initialize the test double."""
            self.payload = payload

        def raise_for_status(self) -> None:
            """Validate the prepared response status."""
            return None

        def json(self) -> dict[str, Any]:
            """Return the prepared JSON payload."""
            return self.payload

    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the test double."""
            return None

        def __enter__(self) -> Self:
            """Enter the test-double context."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Exit the test-double context."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
            return None

    calls = []

    first_page = [{'id': str(index), 'title': f'paper {index}'} for index in range(100)]

    def fake_get(
        url: str,
        headers: Mapping[str, str],
        params: dict[str, Any],
        timeout: float,
    ) -> FakeResponse:
        """Provide a fake HTTP GET implementation."""
        calls.append(params.copy())
        if len(calls) == 1:
            return FakeResponse({'results': first_page, 'totalHits': 101})
        return FakeResponse({'results': [{'id': '100', 'title': 'paper 100'}], 'totalHits': 101})

    monkeypatch.setattr(search, '_core_headers', lambda: {'User-Agent': 'test'})
    monkeypatch.setattr(search.requests, 'get', fake_get)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    rows = search.core_search('solid electrolyte', count=101)

    assert calls == [
        {'q': 'solid electrolyte', 'limit': 100, 'offset': 0},
        {'q': 'solid electrolyte', 'limit': 1, 'offset': 100},
    ]
    assert len(rows) == 101
    assert rows['paper_id'].iloc[0] == 'core:0'
    assert rows['paper_id'].iloc[-1] == 'core:100'


def test_core_search_stops_when_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """CORE search stops when no results."""
    class FakeResponse:
        """Provide a response test double."""

        def raise_for_status(self) -> None:
            """Validate the prepared response status."""
            return None

        def json(self) -> dict[str, list[object]]:
            """Return the prepared JSON payload."""
            return {'results': []}

    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the test double."""
            return None

        def __enter__(self) -> Self:
            """Enter the test-double context."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Exit the test-double context."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
            return None

    monkeypatch.setattr(search.requests, 'get', lambda *_, **__: FakeResponse())
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    assert search.core_search('missing', count=3).empty


def test_core_search_stops_when_page_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """CORE search stops when page is short."""
    class FakeResponse:
        """Provide a response test double."""

        def raise_for_status(self) -> None:
            """Validate the prepared response status."""
            return None

        def json(self) -> dict[str, list[dict[str, str]]]:
            """Return the prepared JSON payload."""
            return {'results': [{'id': '1', 'title': 'one'}]}

    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the test double."""
            return None

        def __enter__(self) -> Self:
            """Enter the test-double context."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Exit the test-double context."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
            return None

    calls = []

    def fake_get(*args: object, **kwargs: Any) -> FakeResponse:
        """Provide a fake HTTP GET implementation."""
        calls.append(kwargs['params'])
        return FakeResponse()

    monkeypatch.setattr(search.requests, 'get', fake_get)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    rows = search.core_search('solid electrolyte', count=3)

    assert rows['paper_id'].tolist() == ['core:1']
    assert len(calls) == 1


def test_openalex_search_paginates_with_cursor_and_stops_at_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAlex search paginates with cursor and stops at count."""
    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the test double."""
            return None

        def __enter__(self) -> Self:
            """Enter the test-double context."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Exit the test-double context."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
            return None

    calls = []

    first_page = [{'id': f'https://openalex.org/W{index}', 'title': f'paper {index}'} for index in range(200)]

    def fake_request_json(
        url: str,
        params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        **_: object,
    ) -> dict[str, Any]:
        """Provide fake OpenAlex JSON responses for this test."""
        calls.append({'url': url, 'params': dict(params), 'api_key': api_key})
        if len(calls) == 1:
            return {'results': first_page, 'meta': {'next_cursor': 'cursor-2'}}
        return {'results': [{'id': 'https://openalex.org/W200', 'title': 'paper 200'}],
                'meta': {'next_cursor': 'cursor-3'}}

    monkeypatch.setattr(search.openalex, 'request_json', fake_request_json)
    monkeypatch.setattr(search.openalex, 'configured_api_key', lambda: 'oa-key')
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    rows = search.openalex_search('solid electrolyte', count=201)

    assert calls[0]['url'] == search.openalex.WORKS_URL
    assert calls[0]['params']['search'] == 'solid electrolyte'
    assert calls[0]['params']['per-page'] == 200
    assert calls[0]['params']['cursor'] == '*'
    assert calls[0]['api_key'] == 'oa-key'
    assert calls[1]['api_key'] == 'oa-key'
    assert calls[1]['params']['per-page'] == 1
    assert calls[1]['params']['cursor'] == 'cursor-2'
    assert len(rows) == 201
    assert rows['paper_id'].iloc[0] == 'openalex:W0'
    assert rows['paper_id'].iloc[-1] == 'openalex:W200'


def test_openalex_search_stops_without_next_cursor_and_omits_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAlex search stops without next cursor and omits API key."""
    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the test double."""
            return None

        def __enter__(self) -> Self:
            """Enter the test-double context."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Exit the test-double context."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
            return None

    calls = []

    def fake_request_json(
        url: str,
        params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        **_: object,
    ) -> dict[str, Any]:
        """Provide fake OpenAlex JSON responses for this test."""
        calls.append({'params': dict(params), 'api_key': api_key})
        return {'results': [{'id': 'https://openalex.org/W1', 'title': 'only'}], 'meta': {}}

    monkeypatch.setattr(search.openalex, 'request_json', fake_request_json)
    monkeypatch.setattr(search.openalex, 'configured_api_key', lambda: None)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    rows = search.openalex_search('query', count=50)

    assert len(calls) == 1
    assert 'api_key' not in calls[0]['params']
    assert calls[0]['api_key'] is None
    assert rows['paper_id'].tolist() == ['openalex:W1']


def test_search_for_papers_rejects_invalid_source() -> None:
    """Search for papers rejects invalid source."""
    with pytest.raises(ValueError, match='source must be one of'):
        search.search_for_papers('query', source='bad')


def test_search_for_papers_merges_and_writes_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search for papers merges and writes results."""
    db_path = tmp_path / 'papers.db'
    rows = search._elsevier_rows(pd.DataFrame([{
        'dc:identifier': 'SCOPUS_ID:1',
        'prism:doi': '10.1234/new',
        'dc:title': 'New paper',
    }]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)

    search.search_for_papers('query', db_path=str(db_path), source='elsevier', count=1)

    with corpus.connect(db_path) as conn:
        written = corpus.paper_rows(conn)
    output = capsys.readouterr().out
    assert written[0]['paper_id'] == 'SCOPUS_ID:1'
    assert '1 new results and updated 0 existing rows' in output


def test_search_for_papers_merges_into_existing_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search for papers merges into existing corpus."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {
            'paper_id': 'core:old',
            'doi': '10.1234/existing',
            'title': 'Existing title',
            'sources': 'core',
            'metadata_status': 'retrieved',
        })
    incoming = search._core_rows([{
        'id': 'new',
        'doi': '10.1234/existing',
        'title': 'Incoming title',
        'journal': 'Updated Journal',
    }])
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: incoming)

    search.search_for_papers('query', db_path=str(db_path), source='core', count=1)

    with corpus.connect(db_path) as conn:
        written = corpus.paper_rows(conn)
    output = capsys.readouterr().out
    assert len(written) == 1
    assert written[0]['title'] == 'Existing title'
    assert written[0]['journal'] == 'Updated Journal'
    assert '0 new results and updated 1 existing rows' in output


def test_search_for_papers_stores_search_time_abstract_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search for papers stores search time abstract assets."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {
            'paper_id': 'existing:paper',
            'doi': '10.1234/existing',
            'title': 'Existing title',
        })
    incoming = search._core_rows([{
        'id': 'core-new',
        'doi': '10.1234/existing',
        'title': 'Incoming title',
        'abstract': '<p>Search abstract</p>',
    }])
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: incoming)

    search.search_for_papers('query', db_path=str(db_path), source='core', count=1, store_abstract=True)

    with corpus.connect(db_path) as conn:
        rows = corpus.paper_rows(conn)
        abstract = corpus.get_asset(conn, 'existing:paper', 'abstract')
    output = capsys.readouterr().out
    assert len(rows) == 1
    assert rows[0]['paper_id'] == 'existing:paper'
    assert rows[0]['abstract_source'] == 'core'
    assert rows[0]['abstract_download_status'] == 'succeeded'
    assert abstract['content'] == b'Search abstract'
    assert 'Stored 1 search-time abstracts.' in output


def test_search_for_papers_reports_zero_results_when_sources_are_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search for papers reports zero results when sources are empty."""
    db_path = tmp_path / 'papers.db'
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: pd.DataFrame())

    search.search_for_papers('query', db_path=str(db_path), source='core', count=1)

    assert 'Document search found 0 new results.' in capsys.readouterr().out
    assert not db_path.exists()


def test_search_for_papers_skips_failed_source_for_all_but_raises_for_selected_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search for papers skips failed source for all but raises for selected source."""
    db_path = tmp_path / 'papers.db'
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: (_ for _ in ()).throw(RuntimeError('elsevier down')))
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: search._core_rows([{'id': '1', 'title': 'Core paper'}]))
    monkeypatch.setattr(search, 'openalex_search', lambda *_, **__: search._openalex_rows([]))

    search.search_for_papers('query', db_path=str(db_path), source='all', count=1)

    assert 'Elsevier search skipped: elsevier down' in capsys.readouterr().out
    assert db_path.exists()

    with pytest.raises(RuntimeError, match='elsevier down'):
        search.search_for_papers('query', source='elsevier', count=1)


def test_search_for_papers_skips_failed_core_for_all_but_raises_for_core(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search for papers skips failed CORE for all but raises for CORE."""
    db_path = tmp_path / 'papers.db'
    rows = search._elsevier_rows(pd.DataFrame([{'dc:identifier': 'SCOPUS_ID:1', 'dc:title': 'Elsevier paper'}]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)
    monkeypatch.setattr(search, 'openalex_search', lambda *_, **__: search._openalex_rows([]))
    monkeypatch.setattr(
        search,
        'core_search',
        lambda *_, **__: (_ for _ in ()).throw(search.requests.RequestException('core down')),
    )

    search.search_for_papers('query', db_path=str(db_path), source='all', count=1)

    assert 'CORE search skipped: core down' in capsys.readouterr().out
    assert db_path.exists()

    with pytest.raises(search.requests.RequestException, match='core down'):
        search.search_for_papers('query', source='core', count=1)


def test_search_for_papers_skips_failed_openalex_for_all_but_raises_for_openalex(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search for papers skips failed OpenAlex for all but raises for OpenAlex."""
    db_path = tmp_path / 'papers.db'
    rows = search._elsevier_rows(pd.DataFrame([{'dc:identifier': 'SCOPUS_ID:1', 'dc:title': 'Elsevier paper'}]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: search._core_rows([]))
    monkeypatch.setattr(
        search,
        'openalex_search',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('openalex down')),
    )

    search.search_for_papers('query', db_path=str(db_path), source='all', count=1)

    assert 'OpenAlex search skipped: openalex down' in capsys.readouterr().out
    assert db_path.exists()

    with pytest.raises(RuntimeError, match='openalex down'):
        search.search_for_papers('query', source='openalex', count=1)


@pytest.mark.network
def test_document_search_uses_real_elsevier_api_when_configured() -> None:
    """Document search uses real Elsevier API when configured."""
    loaded = search.load_settings()

    assert loaded.get('elsevier_api_key'), 'Set elsevier_api_key or ELSEVIER_API_KEY before running network tests.'
    assert len(search.document_search('solid electrolyte', count=1, get_all=False)) <= 1


@pytest.mark.network
def test_core_search_uses_real_core_api_when_configured() -> None:
    """CORE search uses real CORE API when configured."""
    assert search._core_api_key(), 'Set core_api_key or CORE_API_KEY before running network tests.'
    assert len(search.core_search('solid electrolyte', count=1)) <= 1


@pytest.mark.network
def test_openalex_search_uses_real_openalex_api() -> None:
    """OpenAlex search uses real OpenAlex API."""
    rows = search.openalex_search('solid electrolyte', count=1)

    assert len(rows) <= 1
    assert rows.empty or rows.loc[0, 'sources'] == 'openalex'
