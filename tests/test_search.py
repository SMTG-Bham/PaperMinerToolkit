"""Unit tests for paperscraper.search.

This module tests provider-specific search helpers, CORE and Elsevier row
normalization, pagination behavior, request headers, and merging search results
into the paper corpus.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn, Self

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

    monkeypatch.setattr(search, 'medrxiv_search', lambda *_, **__: search._medrxiv_rows([]))
    monkeypatch.setattr(search, 'biorxiv_search', lambda *_, **__: search._biorxiv_rows([]))

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

    monkeypatch.setattr(search, 'medrxiv_search', lambda *_, **__: search._medrxiv_rows([]))
    monkeypatch.setattr(search, 'biorxiv_search', lambda *_, **__: search._biorxiv_rows([]))

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

    monkeypatch.setattr(search, 'medrxiv_search', lambda *_, **__: search._medrxiv_rows([]))
    monkeypatch.setattr(search, 'biorxiv_search', lambda *_, **__: search._biorxiv_rows([]))

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


def test_search_for_papers_enriches_new_rows_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Enrich stored rows through the real search hook when --enrich is set."""
    db_path = tmp_path / 'papers.db'
    rows = search._elsevier_rows(pd.DataFrame([{
        'dc:identifier': 'SCOPUS_ID:1',
        'prism:doi': '10.1234/new',
        'dc:title': 'New paper',
    }]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)

    calls = {}

    def fake_enrich_papers(conn: Any, records: Any, **kwargs: Any) -> dict[str, int]:
        """Record the enrichment call and report one success."""
        calls['records'] = list(records)
        return {'succeeded': 1, 'partial': 0, 'not_found': 0}

    monkeypatch.setattr(search, 'enrich_papers', fake_enrich_papers)

    search.search_for_papers('query', db_path=str(db_path), source='elsevier',
                             count=1, enrich=True)

    output = capsys.readouterr().out
    assert len(calls['records']) == 1
    assert 'Enriched 1 papers' in output


def test_search_for_papers_skips_enrichment_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Leave enrichment untouched unless it is explicitly requested."""
    db_path = tmp_path / 'papers.db'
    rows = search._elsevier_rows(pd.DataFrame([{
        'dc:identifier': 'SCOPUS_ID:1',
        'prism:doi': '10.1234/new',
        'dc:title': 'New paper',
    }]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)

    def fail_enrich_papers(conn: Any, records: Any, **kwargs: Any) -> NoReturn:
        """Fail if enrichment runs without being requested."""
        raise AssertionError('enrichment must not run by default')

    monkeypatch.setattr(search, 'enrich_papers', fail_enrich_papers)

    search.search_for_papers('query', db_path=str(db_path), source='elsevier', count=1)

    assert 'Enriched' not in capsys.readouterr().out


def pubmed_articles() -> list[dict[str, Any]]:
    """Return PubMed article mappings covering a DOI-less and a normal record."""
    return [
        {'paper_id': 'doi:10.1234/one', 'doi': '10.1234/one', 'pmid': '1', 'pmcid': 'PMC1',
         'title': 'Garnet electrolytes', 'journal': 'Test Journal',
         'publication_date': '2024-03-07', 'authors': 'Jane A Smith',
         'sources': 'pubmed', 'metadata_status': 'retrieved',
         'abstract': '<p>BACKGROUND: Solid   electrolytes  matter.</p>'},
        {'paper_id': 'pmid:2', 'doi': '', 'pmid': '2', 'pmcid': '',
         'title': 'An older paper', 'journal': 'Old Journal', 'publication_date': '2019',
         'authors': 'J Doe', 'sources': 'pubmed', 'metadata_status': 'retrieved',
         'abstract': ''},
    ]


def test_pubmed_rows_normalize_records_and_clean_abstracts() -> None:
    """Map PubMed records onto the shared search columns and clean abstract markup."""
    rows = search._pubmed_rows(pubmed_articles())

    assert list(rows.columns) == search.SEARCH_FIELDS
    assert rows['paper_id'].tolist() == ['doi:10.1234/one', 'pmid:2']
    assert rows['pmid'].tolist() == ['1', '2']
    assert rows['pmcid'].tolist() == ['PMC1', '']
    assert rows['sources'].tolist() == ['pubmed', 'pubmed']
    assert rows.loc[0, 'abstract'] == 'BACKGROUND: Solid electrolytes matter.'
    assert rows.loc[1, 'abstract'] == ''
    assert search._pubmed_rows([]).empty


def test_pubmed_search_pages_efetch_and_stops_at_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Page a stored result set and stop once the requested count is reached."""
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

    pages = []

    def fake_esearch_history(query: str, **kwargs: Any) -> tuple[str, str, int]:
        """Provide a fake stored-search implementation."""
        return 'WE1', '1', 500

    def fake_efetch_history(webenv: str, query_key: str, **kwargs: Any) -> object:
        """Provide a fake record-page implementation."""
        pages.append((kwargs['retstart'], kwargs['retmax']))
        return object()

    def fake_parse_articles(_: object) -> list[dict[str, Any]]:
        """Return one page worth of article mappings."""
        start = len(pages) - 1
        return [{'paper_id': f'pmid:{start * 200 + index}', 'pmid': str(start * 200 + index),
                 'sources': 'pubmed'} for index in range(pages[-1][1])]

    monkeypatch.setattr(search.pubmed, 'esearch_history', fake_esearch_history)
    monkeypatch.setattr(search.pubmed, 'efetch_history', fake_efetch_history)
    monkeypatch.setattr(search.pubmed, 'parse_articles', fake_parse_articles)
    monkeypatch.setattr(search.pubmed, 'configured_api_key', lambda *_, **__: None)
    monkeypatch.setattr(search.pubmed, 'configured_email', lambda *_, **__: '')
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    rows = search.pubmed_search('lithium', count=250)

    assert len(rows) == 250
    assert pages == [(0, 200), (200, 50)]


def test_pubmed_search_reports_the_ten_thousand_result_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Warn when a query matches more records than PubMed will ever return."""
    monkeypatch.setattr(search.pubmed, 'esearch_history', lambda *_, **__: ('', '', 240000))
    monkeypatch.setattr(search.pubmed, 'configured_api_key', lambda *_, **__: None)
    monkeypatch.setattr(search.pubmed, 'configured_email', lambda *_, **__: '')

    rows = search.pubmed_search('lithium', count=10)

    assert rows.empty
    assert 'PubMed matched 240000 records but exposes only the first 10000' in capsys.readouterr().out


def test_search_for_papers_skips_failed_pubmed_for_all_but_raises_for_pubmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Skip a failing PubMed provider under all but surface it when selected."""
    db_path = tmp_path / 'papers.db'
    rows = search._elsevier_rows(pd.DataFrame([{'dc:identifier': 'SCOPUS_ID:1', 'dc:title': 'Elsevier paper'}]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: search._core_rows([]))
    monkeypatch.setattr(search, 'openalex_search', lambda *_, **__: search._openalex_rows([]))
    monkeypatch.setattr(
        search,
        'pubmed_search',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('pubmed down')),
    )

    monkeypatch.setattr(search, 'medrxiv_search', lambda *_, **__: search._medrxiv_rows([]))
    monkeypatch.setattr(search, 'biorxiv_search', lambda *_, **__: search._biorxiv_rows([]))

    search.search_for_papers('query', db_path=str(db_path), source='all', count=1)

    assert 'PubMed search skipped: pubmed down' in capsys.readouterr().out
    assert db_path.exists()

    with pytest.raises(RuntimeError, match='pubmed down'):
        search.search_for_papers('query', source='pubmed', count=1)


def arxiv_entries(count: int, start: int = 0) -> list[dict[str, Any]]:
    """Return mapped arXiv entries numbered from an offset."""
    return [{'paper_id': f'arxiv:2301.{start + index:05d}',
             'arxiv_id': f'2301.{start + index:05d}',
             'title': f'Preprint {start + index}',
             'abstract': f'  Abstract {start + index}\n  wrapped.  ',
             'sources': 'arxiv'} for index in range(count)]


def test_arxiv_rows_normalize_records_and_clean_abstracts() -> None:
    """Frame arXiv entries on the search schema with compacted abstracts."""
    rows = search._arxiv_rows(arxiv_entries(2))

    assert list(rows.columns) == search.SEARCH_FIELDS
    assert rows.loc[0, 'sources'] == 'arxiv'
    assert rows.loc[0, 'arxiv_id'] == '2301.00000'
    assert rows.loc[0, 'abstract'] == 'Abstract 0 wrapped.'
    assert search._arxiv_rows([]).empty


def test_arxiv_search_pages_and_stops_at_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Walk arXiv pages until the requested count is filled."""
    class FakeTqdm:
        """Record progress-bar interactions without rendering anything."""

        def __init__(self, *_: object, **__: object) -> None:
            """Initialize the test double."""
            self.total = 0

        def __enter__(self) -> Self:
            """Enter the test-double context."""
            return self

        def __exit__(self, *_: object) -> bool:
            """Exit the test-double context."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
            return None

    pages = []

    def fake_search_page(query: str, **kwargs: Any) -> object:
        """Record the requested window and return a page marker."""
        pages.append((kwargs['start'], kwargs['max_results']))
        return object()

    def fake_parse_entries(_: object) -> list[dict[str, Any]]:
        """Return one page worth of entry mappings."""
        start, size = pages[-1]
        return arxiv_entries(size, start=start)

    monkeypatch.setattr(search.arxiv, 'search_page', fake_search_page)
    monkeypatch.setattr(search.arxiv, 'parse_entries', fake_parse_entries)
    monkeypatch.setattr(search.arxiv, 'total_results', lambda _: 500)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    rows = search.arxiv_search('lithium', count=250)

    assert len(rows) == 250
    assert pages == [(0, 200), (200, 50)]


def test_arxiv_search_deduplicates_entries_repeated_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep each identifier once and still advance past a page of repeats."""
    pages = []

    def fake_search_page(query: str, **kwargs: Any) -> object:
        """Record the requested window and return a page marker."""
        pages.append((kwargs['start'], kwargs['max_results']))
        return object()

    def fake_parse_entries(_: object) -> list[dict[str, Any]]:
        """Return pages that overlap by one entry, then run dry."""
        return [arxiv_entries(3), arxiv_entries(3, start=2), []][len(pages) - 1]

    monkeypatch.setattr(search.arxiv, 'search_page', fake_search_page)
    monkeypatch.setattr(search.arxiv, 'parse_entries', fake_parse_entries)
    monkeypatch.setattr(search.arxiv, 'total_results', lambda _: 500)

    rows = search.arxiv_search('lithium', count=10)

    # Five distinct entries across two overlapping pages, and the cursor moved by
    # what arXiv returned rather than by what survived, so the walk terminates.
    assert len(rows) == 5
    assert [start for start, _ in pages] == [0, 3, 6]


def test_arxiv_search_stops_once_it_has_paged_past_every_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End the walk at the reported match total instead of looping on repeats."""
    pages = []

    def fake_search_page(query: str, **kwargs: Any) -> object:
        """Record the requested window and return a page marker."""
        pages.append((kwargs['start'], kwargs['max_results']))
        return object()

    monkeypatch.setattr(search.arxiv, 'search_page', fake_search_page)
    monkeypatch.setattr(search.arxiv, 'parse_entries', lambda _: arxiv_entries(2))
    monkeypatch.setattr(search.arxiv, 'total_results', lambda _: 4)

    rows = search.arxiv_search('lithium', count=50)

    assert len(rows) == 2
    assert len(pages) == 2


def test_arxiv_search_reports_the_deep_paging_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Warn when a query matches more records than arXiv will page through."""
    monkeypatch.setattr(search.arxiv, 'search_page', lambda *_, **__: object())
    monkeypatch.setattr(search.arxiv, 'parse_entries', lambda _: arxiv_entries(1))
    monkeypatch.setattr(search.arxiv, 'total_results', lambda _: 240000)

    rows = search.arxiv_search('lithium', count=1)

    assert len(rows) == 1
    assert 'arXiv matched 240000 records but exposes only the first 30000' in capsys.readouterr().out


def test_arxiv_search_returns_no_rows_for_an_empty_query() -> None:
    """Skip the request entirely when the query reduces to nothing."""
    assert search.arxiv_search('   ', count=10).empty
    assert search.arxiv_search('lithium', count=0).empty


def test_search_for_papers_skips_failed_arxiv_for_all_but_raises_for_arxiv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Skip a failing arXiv provider under all but surface it when selected."""
    db_path = tmp_path / 'papers.db'
    rows = search._elsevier_rows(pd.DataFrame([{'dc:identifier': 'SCOPUS_ID:1', 'dc:title': 'Elsevier paper'}]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: search._core_rows([]))
    monkeypatch.setattr(search, 'openalex_search', lambda *_, **__: search._openalex_rows([]))
    monkeypatch.setattr(search, 'pubmed_search', lambda *_, **__: search._pubmed_rows([]))
    monkeypatch.setattr(
        search,
        'arxiv_search',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('arxiv down')),
    )

    monkeypatch.setattr(search, 'medrxiv_search', lambda *_, **__: search._medrxiv_rows([]))
    monkeypatch.setattr(search, 'biorxiv_search', lambda *_, **__: search._biorxiv_rows([]))

    search.search_for_papers('query', db_path=str(db_path), source='all', count=1)

    assert 'arXiv search skipped: arxiv down' in capsys.readouterr().out
    assert db_path.exists()

    with pytest.raises(RuntimeError, match='arxiv down'):
        search.search_for_papers('query', source='arxiv', count=1)


def medrxiv_records(count: int, start: int = 0, version: str = '1',
                    date: str = '2024-03-01') -> list[dict[str, Any]]:
    """Return mapped medRxiv records numbered from an offset."""
    return [{'paper_id': f'doi:10.1101/2024.03.01.2430{start + index:04d}',
             'doi': f'10.1101/2024.03.01.2430{start + index:04d}',
             'medrxiv_doi': f'10.1101/2024.03.01.2430{start + index:04d}',
             'title': f'Preprint {start + index} on vaccines',
             'abstract': f'  Abstract {start + index}\n  wrapped.  ',
             'publication_date': date,
             'version': version,
             'category': 'infectious diseases',
             'sources': 'medrxiv'} for index in range(count)]


def stub_medrxiv_walk(monkeypatch: pytest.MonkeyPatch,
                      pages: list[list[dict[str, Any]]],
                      total: int | None = None,
                      step: int = 100) -> list[dict[str, Any]]:
    """Serve prepared interval pages and record the cursors requested.

    Pages are indexed by cursor so a walk that revisits the first page reads
    the same records, which is what the reuse of the count-probing request
    depends on.
    """
    calls: list[dict[str, Any]] = []
    by_cursor = {index * step: page for index, page in enumerate(pages)}

    def fake_interval_page(start: str, end: str, cursor: int = 0,
                           category: str = '', **_: Any) -> object:
        """Record the requested window and return a page marker."""
        calls.append({'start': start, 'end': end, 'cursor': cursor, 'category': category})
        return {'cursor': cursor}

    monkeypatch.setattr(search.medrxiv, 'interval_page', fake_interval_page)
    monkeypatch.setattr(search.medrxiv, 'parse_records',
                        lambda payload: by_cursor.get(payload['cursor'], []))
    monkeypatch.setattr(search.medrxiv, 'total_results',
                        lambda _: total if total is not None else step * len(pages))
    monkeypatch.setattr(search.medrxiv, 'page_size', lambda *_, **__: step)
    return calls


def test_medrxiv_rows_normalize_records_and_clean_abstracts() -> None:
    """Frame medRxiv records on the search schema with compacted abstracts."""
    rows = search._medrxiv_rows(medrxiv_records(2))

    assert list(rows.columns) == search.SEARCH_FIELDS
    assert rows.loc[0, 'sources'] == 'medrxiv'
    assert rows.loc[0, 'medrxiv_doi'] == '10.1101/2024.03.01.24300000'
    assert rows.loc[0, 'abstract'] == 'Abstract 0 wrapped.'
    assert search._medrxiv_rows([]).empty


def test_medrxiv_search_walks_pages_newest_first_and_stops_at_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the last page first and stop once the requested count matches."""
    pages = [medrxiv_records(2, start=index * 2) for index in range(4)]
    calls = stub_medrxiv_walk(monkeypatch, pages, total=400)

    rows = search.medrxiv_search('vaccines from:2024-01-01 to:2024-12-31', count=3)

    assert len(rows) == 3
    # The count probe runs at cursor zero, then the walk starts at the last page.
    assert [call['cursor'] for call in calls] == [0, 300, 200]
    # Records within a page are reversed, so the newest posting leads.
    assert rows.loc[0, 'medrxiv_doi'] == '10.1101/2024.03.01.24300007'


def test_medrxiv_search_reuses_the_page_it_fetched_to_learn_the_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spend one request on the first page rather than fetching it twice."""
    calls = stub_medrxiv_walk(monkeypatch, [medrxiv_records(2)], total=2)

    rows = search.medrxiv_search('vaccines', count=50)

    assert len(rows) == 2
    assert [call['cursor'] for call in calls] == [0]


def test_medrxiv_search_collapses_versions_that_fall_on_different_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count a revised preprint once even when its postings are pages apart."""
    pages = [medrxiv_records(1, version='1', date='2024-01-05'),
             medrxiv_records(1, version='2', date='2024-06-05')]
    stub_medrxiv_walk(monkeypatch, pages, total=200)

    rows = search.medrxiv_search('vaccines', count=5)

    assert len(rows) == 1
    # The walk met the newest posting first, and the earlier one dated it.
    assert rows.loc[0, 'publication_date'] == '2024-01-05'


def test_medrxiv_search_applies_the_scope_terms_to_the_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the interval and category the query named to the API."""
    calls = stub_medrxiv_walk(monkeypatch, [medrxiv_records(1)], total=1, step=30)

    search.medrxiv_search('vaccines category:"Infectious Diseases" from:2024-02-01 to:2024-02-29',
                          count=1)

    assert calls[0] == {'start': '2024-02-01', 'end': '2024-02-29',
                        'cursor': 0, 'category': 'Infectious Diseases'}


def test_medrxiv_search_defaults_the_interval_to_the_whole_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read from the first medRxiv posting to today when the query says nothing."""
    calls = stub_medrxiv_walk(monkeypatch, [medrxiv_records(1)], total=1)

    search.medrxiv_search('vaccines', count=1, today='2026-08-22')

    assert calls[0]['start'] == search.medrxiv.CORPUS_START
    assert calls[0]['end'] == '2026-08-22'
    assert calls[0]['category'] == ''


def test_medrxiv_search_matches_terms_rather_than_returning_the_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep only the postings that carry every term of the query."""
    stub_medrxiv_walk(monkeypatch, [medrxiv_records(3)], total=3)

    assert len(search.medrxiv_search('vaccines', count=10)) == 3
    assert search.medrxiv_search('lithium electrolyte', count=10).empty


def test_medrxiv_search_stops_at_the_scan_limit_and_reports_the_shortfall(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End a fruitless walk at the scan limit instead of reading the archive."""
    monkeypatch.setattr(search.medrxiv, 'MAX_SCAN_RECORDS', 200)
    pages = [medrxiv_records(100, start=index * 100) for index in range(10)]
    calls = stub_medrxiv_walk(monkeypatch, pages, total=1000)

    rows = search.medrxiv_search('lithium', count=5)

    assert rows.empty
    # Two pages of a ten-page archive were read before the limit stopped it.
    assert len(calls) == 3
    output = capsys.readouterr().out
    assert 'medRxiv matched 0 papers in 200 postings read.' in output
    assert '800 older postings in' in output
    assert '200-posting scan limit' in output


def test_medrxiv_search_announces_the_scan_before_it_starts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Say how much is about to be read rather than appearing to hang."""
    stub_medrxiv_walk(monkeypatch, [medrxiv_records(1)], total=1)

    search.medrxiv_search('vaccines category:oncology from:2024-01-01 to:2024-12-31', count=1)

    output = capsys.readouterr().out
    assert 'medRxiv has no search endpoint' in output
    assert 'reading the 1 postings between 2024-01-01 and 2024-12-31 filed under oncology' in output


def test_medrxiv_search_returns_no_rows_for_an_empty_or_unmatched_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip the walk when the count is zero or the interval holds nothing."""
    assert search.medrxiv_search('vaccines', count=0).empty

    stub_medrxiv_walk(monkeypatch, [], total=0)
    assert search.medrxiv_search('vaccines', count=10).empty


def test_medrxiv_search_rejects_a_scope_date_it_cannot_use() -> None:
    """Refuse a malformed interval before spending any request on it."""
    with pytest.raises(ValueError, match='from: must be an ISO date'):
        search.medrxiv_search('vaccines from:january', count=1)


def test_search_for_papers_skips_failed_medrxiv_for_all_but_raises_for_medrxiv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Skip a failing medRxiv provider under all but surface it when selected."""
    db_path = tmp_path / 'papers.db'
    rows = search._elsevier_rows(pd.DataFrame([{'dc:identifier': 'SCOPUS_ID:1', 'dc:title': 'Elsevier paper'}]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: search._core_rows([]))
    monkeypatch.setattr(search, 'openalex_search', lambda *_, **__: search._openalex_rows([]))
    monkeypatch.setattr(search, 'pubmed_search', lambda *_, **__: search._pubmed_rows([]))
    monkeypatch.setattr(search, 'arxiv_search', lambda *_, **__: search._arxiv_rows([]))
    monkeypatch.setattr(
        search,
        'medrxiv_search',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('medrxiv down')),
    )
    monkeypatch.setattr(search, 'biorxiv_search', lambda *_, **__: search._biorxiv_rows([]))

    search.search_for_papers('query', db_path=str(db_path), source='all', count=1)

    assert 'medRxiv search skipped: medrxiv down' in capsys.readouterr().out
    assert db_path.exists()

    with pytest.raises(RuntimeError, match='medrxiv down'):
        search.search_for_papers('query', source='medrxiv', count=1)


def biorxiv_records(count: int, start: int = 0, version: str = '1',
                    date: str = '2024-03-01') -> list[dict[str, Any]]:
    """Return mapped bioRxiv records numbered from an offset."""
    return [{'paper_id': f'doi:10.1101/2024.03.01.58{start + index:04d}',
             'doi': f'10.1101/2024.03.01.58{start + index:04d}',
             'biorxiv_doi': f'10.1101/2024.03.01.58{start + index:04d}',
             'title': f'Preprint {start + index} on genomes',
             'abstract': f'  Abstract {start + index}\n  wrapped.  ',
             'publication_date': date,
             'version': version,
             'category': 'neuroscience',
             'sources': 'biorxiv'} for index in range(count)]


def stub_biorxiv_walk(monkeypatch: pytest.MonkeyPatch,
                      pages: list[list[dict[str, Any]]],
                      total: int | None = None,
                      step: int = 100) -> list[dict[str, Any]]:
    """Serve prepared bioRxiv interval pages and record the cursors requested."""
    calls: list[dict[str, Any]] = []
    by_cursor = {index * step: page for index, page in enumerate(pages)}

    def fake_interval_page(start: str, end: str, cursor: int = 0,
                           category: str = '', **_: Any) -> object:
        """Record the requested window and return a page marker."""
        calls.append({'start': start, 'end': end, 'cursor': cursor, 'category': category})
        return {'cursor': cursor}

    monkeypatch.setattr(search.biorxiv, 'interval_page', fake_interval_page)
    monkeypatch.setattr(search.biorxiv, 'parse_records',
                        lambda payload: by_cursor.get(payload['cursor'], []))
    monkeypatch.setattr(search.biorxiv, 'total_results',
                        lambda _: total if total is not None else step * len(pages))
    monkeypatch.setattr(search.biorxiv, 'page_size', lambda *_, **__: step)
    return calls


def test_biorxiv_rows_normalize_records_and_clean_abstracts() -> None:
    """Frame bioRxiv records on the search schema with compacted abstracts."""
    rows = search._biorxiv_rows(biorxiv_records(2))

    assert list(rows.columns) == search.SEARCH_FIELDS
    assert rows.loc[0, 'sources'] == 'biorxiv'
    assert rows.loc[0, 'biorxiv_doi'] == '10.1101/2024.03.01.580000'
    assert rows.loc[0, 'abstract'] == 'Abstract 0 wrapped.'
    assert search._biorxiv_rows([]).empty


def test_biorxiv_search_walks_pages_newest_first_and_stops_at_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the last page first and stop once the requested count matches."""
    pages = [biorxiv_records(2, start=index * 2) for index in range(4)]
    calls = stub_biorxiv_walk(monkeypatch, pages, total=400)

    rows = search.biorxiv_search('genomes from:2024-01-01 to:2024-12-31', count=3)

    assert len(rows) == 3
    # The count probe runs at cursor zero, then the walk starts at the last page.
    assert [call['cursor'] for call in calls] == [0, 300, 200]
    # Records within a page are reversed, so the newest posting leads.
    assert rows.loc[0, 'biorxiv_doi'] == '10.1101/2024.03.01.580007'


def test_biorxiv_search_collapses_versions_that_fall_on_different_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count a revised preprint once even when its postings are pages apart."""
    pages = [biorxiv_records(1, version='1', date='2024-01-05'),
             biorxiv_records(1, version='2', date='2024-06-05')]
    stub_biorxiv_walk(monkeypatch, pages, total=200)

    rows = search.biorxiv_search('genomes', count=5)

    assert len(rows) == 1
    # The walk met the newest posting first, and the earlier one dated it.
    assert rows.loc[0, 'publication_date'] == '2024-01-05'


def test_biorxiv_search_defaults_the_interval_to_the_whole_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read from the first bioRxiv posting to today when the query says nothing.

    bioRxiv opened in 2013, six years before medRxiv, so the unscoped walk it
    defaults to is the longer of the two by a wide margin.
    """
    calls = stub_biorxiv_walk(monkeypatch, [biorxiv_records(1)], total=1)

    search.biorxiv_search('genomes', count=1, today='2026-08-22')

    assert calls[0]['start'] == search.biorxiv.CORPUS_START == '2013-11-01'
    assert calls[0]['end'] == '2026-08-22'
    assert calls[0]['category'] == ''


def test_biorxiv_search_applies_the_scope_terms_to_the_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the interval and category the query named to the API."""
    calls = stub_biorxiv_walk(monkeypatch, [biorxiv_records(1)], total=1, step=30)

    search.biorxiv_search('genomes category:"Developmental Biology" '
                          'from:2024-02-01 to:2024-02-29', count=1)

    assert calls[0] == {'start': '2024-02-01', 'end': '2024-02-29',
                        'cursor': 0, 'category': 'Developmental Biology'}


def test_biorxiv_search_reads_biorxiv_rather_than_medrxiv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind the shared archive walk to the server the caller actually asked for.

    One walk serves both preprint servers, so the binding is the only thing
    keeping a bioRxiv search off medRxiv's archive.
    """
    def unreachable(*_: Any, **__: Any) -> None:
        """Fail the test if the medRxiv module is touched by a bioRxiv search."""
        raise AssertionError('a bioRxiv search must not read the medRxiv archive')

    monkeypatch.setattr(search.medrxiv, 'interval_page', unreachable)
    stub_biorxiv_walk(monkeypatch, [biorxiv_records(1)], total=1)

    rows = search.biorxiv_search('genomes', count=1)

    assert rows.loc[0, 'sources'] == 'biorxiv'


def test_biorxiv_search_stops_at_the_scan_limit_and_reports_the_shortfall(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End a fruitless walk at the scan limit instead of reading the archive."""
    monkeypatch.setattr(search.biorxiv, 'MAX_SCAN_RECORDS', 200)
    pages = [biorxiv_records(100, start=index * 100) for index in range(10)]
    calls = stub_biorxiv_walk(monkeypatch, pages, total=1000)

    rows = search.biorxiv_search('lithium', count=5)

    assert rows.empty
    # Two pages of a ten-page archive were read before the limit stopped it.
    assert len(calls) == 3
    output = capsys.readouterr().out
    assert 'bioRxiv matched 0 papers in 200 postings read.' in output
    assert '800 older postings in' in output
    assert '200-posting scan limit' in output


def test_biorxiv_search_announces_the_scan_before_it_starts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Say how much is about to be read rather than appearing to hang."""
    stub_biorxiv_walk(monkeypatch, [biorxiv_records(1)], total=1)

    search.biorxiv_search('genomes category:genomics from:2024-01-01 to:2024-12-31', count=1)

    output = capsys.readouterr().out
    assert 'bioRxiv has no search endpoint' in output
    assert 'reading the 1 postings between 2024-01-01 and 2024-12-31 filed under genomics' in output


def test_biorxiv_search_returns_no_rows_for_an_empty_or_unmatched_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip the walk when the count is zero or the interval holds nothing."""
    assert search.biorxiv_search('genomes', count=0).empty

    stub_biorxiv_walk(monkeypatch, [], total=0)
    assert search.biorxiv_search('genomes', count=10).empty


def test_biorxiv_search_rejects_a_scope_date_it_cannot_use() -> None:
    """Refuse a malformed interval before spending any request on it."""
    with pytest.raises(ValueError, match='from: must be an ISO date'):
        search.biorxiv_search('genomes from:january', count=1)


def test_search_for_papers_skips_failed_biorxiv_for_all_but_raises_for_biorxiv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Skip a failing bioRxiv provider under all but surface it when selected."""
    db_path = tmp_path / 'papers.db'
    rows = search._elsevier_rows(pd.DataFrame([{'dc:identifier': 'SCOPUS_ID:1',
                                                'dc:title': 'Elsevier paper'}]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: search._core_rows([]))
    monkeypatch.setattr(search, 'openalex_search', lambda *_, **__: search._openalex_rows([]))
    monkeypatch.setattr(search, 'pubmed_search', lambda *_, **__: search._pubmed_rows([]))
    monkeypatch.setattr(search, 'arxiv_search', lambda *_, **__: search._arxiv_rows([]))
    monkeypatch.setattr(search, 'medrxiv_search', lambda *_, **__: search._medrxiv_rows([]))
    monkeypatch.setattr(
        search,
        'biorxiv_search',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('biorxiv down')),
    )

    search.search_for_papers('query', db_path=str(db_path), source='all', count=1)

    assert 'bioRxiv search skipped: biorxiv down' in capsys.readouterr().out
    assert db_path.exists()

    with pytest.raises(RuntimeError, match='biorxiv down'):
        search.search_for_papers('query', source='biorxiv', count=1)


@pytest.mark.network
def test_biorxiv_search_uses_the_real_api() -> None:
    """Search the real bioRxiv archive over a small interval."""
    rows = search.biorxiv_search('cell category:neuroscience '
                                 'from:2024-03-01 to:2024-03-02', count=1)

    assert len(rows) <= 1
    assert list(rows.columns) == search.SEARCH_FIELDS


@pytest.mark.network
def test_medrxiv_search_uses_the_real_api() -> None:
    """Search the real medRxiv archive over a small interval."""
    rows = search.medrxiv_search('covid category:"infectious diseases" '
                                 'from:2024-03-01 to:2024-03-05', count=1)

    assert len(rows) <= 1
    assert list(rows.columns) == search.SEARCH_FIELDS


@pytest.mark.network
def test_arxiv_search_uses_the_real_api() -> None:
    """Search the real arXiv query service."""
    rows = search.arxiv_search('cat:cond-mat.mtrl-sci', count=1)

    assert len(rows) <= 1
    assert rows.empty or rows.loc[0, 'sources'] == 'arxiv'


@pytest.mark.network
def test_pubmed_search_uses_the_real_eutilities_api() -> None:
    """PubMed search uses the real E-utilities service."""
    rows = search.pubmed_search('solid electrolyte', count=1)

    assert len(rows) <= 1
    assert rows.empty or rows.loc[0, 'sources'] == 'pubmed'
