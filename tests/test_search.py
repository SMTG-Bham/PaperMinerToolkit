"""Unit tests for paperscraper.search.

This module tests provider-specific search helpers, CORE and Elsevier row
normalization, pagination behavior, request headers, and merging search results
into the papers CSV.
"""

import importlib

import pandas as pd
import pytest

search = importlib.import_module('paperscraper.search')


def test_elsevier_client_requires_configured_api_key(monkeypatch):
    """
    Test Elsevier client creation without a configured API key.

    This function performs the following steps:
    1. Replaces settings loading with an empty settings dictionary.
    2. Calls `_elsevier_client`.
    3. Captures the expected exception.

    Asserts:
        - Missing Elsevier API keys raise `ValueError`.
    """
    monkeypatch.setattr(search, 'load_settings', lambda: {})

    with pytest.raises(ValueError, match='Elsevier API key is not configured'):
        search._elsevier_client()


def test_elsevier_client_uses_configured_api_key(monkeypatch):
    """
    Test Elsevier client creation with a configured API key.

    This function performs the following steps:
    1. Replaces settings loading with an Elsevier API key.
    2. Replaces `ElsClient` with a fake client class.
    3. Calls `_elsevier_client`.

    Asserts:
        - The configured API key is passed to `ElsClient`.
        - The created client is returned.
    """

    class FakeElsClient:
        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr(search, 'load_settings', lambda: {'elsevier_api_key': 'elsevier-key'})
    monkeypatch.setattr(search, 'ElsClient', FakeElsClient)

    client = search._elsevier_client()

    assert isinstance(client, FakeElsClient)
    assert client.api_key == 'elsevier-key'


def test_document_search_caps_count_and_paginates_results(monkeypatch, capsys):
    """
    Test Elsevier document search pagination and count limiting.

    This function performs the following steps:
    1. Replaces the Elsevier client with a fake paginated client.
    2. Replaces tqdm and recast helpers with simple local helpers.
    3. Calls `document_search` with a count lower than total available results.

    Asserts:
        - Returned results are capped at the requested count.
        - The first request uses the capped page size.
        - The next page is requested.
    """

    class FakeClient:
        def __init__(self):
            self.urls = []

        def exec_request(self, url):
            self.urls.append(url)
            if len(self.urls) == 1:
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
        def __init__(self, *args, **kwargs):
            self.updates = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, value):
            self.updates.append(value)

    client = FakeClient()
    monkeypatch.setattr(search, '_elsevier_client', lambda: client)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)
    monkeypatch.setattr(search, 'recast_df', lambda df: df)

    results = search.document_search('solid electrolyte', count=2, get_all=True)

    output = capsys.readouterr().out
    assert results['dc:title'].tolist() == ['first', 'second']
    assert 'retrieving 2 of 3 results' in output
    assert '&count=2' in client.urls[0]
    assert client.urls[1] == 'next-url'


def test_document_search_returns_empty_dataframe_for_zero_results(monkeypatch):
    """
    Test Elsevier document search with no provider results.

    This function performs the following steps:
    1. Replaces the Elsevier client with a fake client returning zero total results.
    2. Calls `document_search`.
    3. Checks the returned DataFrame.

    Asserts:
        - Zero provider results return an empty DataFrame.
    """

    class FakeClient:
        def exec_request(self, _):
            return {'search-results': {'opensearch:totalResults': '0', 'entry': [], 'link': []}}

    monkeypatch.setattr(search, '_elsevier_client', lambda: FakeClient())

    assert search.document_search('missing').empty


def test_document_search_without_get_all_returns_first_page_slice(monkeypatch):
    """
    Test Elsevier document search without pagination.

    This function performs the following steps:
    1. Replaces the Elsevier client with a fake first-page response.
    2. Replaces recasting with an identity helper.
    3. Calls `document_search` with `get_all=False`.

    Asserts:
        - Only the requested number of first-page results are returned.
    """

    class FakeClient:
        def exec_request(self, _):
            return {
                'search-results': {
                    'opensearch:totalResults': '3',
                    'entry': [{'dc:title': 'first'}, {'dc:title': 'second'}, {'dc:title': 'third'}],
                    'link': [],
                }
            }

    monkeypatch.setattr(search, '_elsevier_client', lambda: FakeClient())
    monkeypatch.setattr(search, 'recast_df', lambda df: df)

    results = search.document_search('solid electrolyte', count=2, get_all=False)

    assert results['dc:title'].tolist() == ['first', 'second']


def test_document_search_stops_when_next_link_is_missing(monkeypatch):
    """
    Test Elsevier document search pagination when the provider omits a next link.

    This function performs the following steps:
    1. Replaces the Elsevier client with a fake response containing fewer results than requested.
    2. Omits a next-page link from the fake response.
    3. Calls `document_search` with pagination enabled.

    Asserts:
        - The available first-page results are returned.
        - Search stops without requesting another page.
    """

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def exec_request(self, _):
            self.calls += 1
            return {
                'search-results': {
                    'opensearch:totalResults': '3',
                    'entry': [{'dc:title': 'first'}],
                    'link': [],
                }
            }

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    client = FakeClient()
    monkeypatch.setattr(search, '_elsevier_client', lambda: client)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)
    monkeypatch.setattr(search, 'recast_df', lambda df: df)

    results = search.document_search('solid electrolyte', count=3, get_all=True)

    assert results['dc:title'].tolist() == ['first']
    assert client.calls == 1


def test_document_search_stops_non_scopus_searches_at_provider_limit(monkeypatch):
    """
    Test Elsevier document search stop behavior for non-Scopus indexes.

    This function performs the following steps:
    1. Replaces the Elsevier client with a fake non-Scopus paginated response.
    2. Starts with 4,999 results and returns two more results on the next page.
    3. Calls `document_search` for a non-Scopus index.

    Asserts:
        - Search stops after crossing the non-Scopus provider limit.
        - The returned DataFrame includes the page that crosses that limit.
    """

    first_page = [{'dc:title': f'paper {index}'} for index in range(4999)]

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def exec_request(self, _):
            self.calls += 1
            if self.calls == 1:
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
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    client = FakeClient()
    monkeypatch.setattr(search, '_elsevier_client', lambda: client)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)
    monkeypatch.setattr(search, 'recast_df', lambda df: df)

    results = search.document_search('solid electrolyte', index='article', count=6000, get_all=True)

    assert len(results) == 5001
    assert client.calls == 2


def test_first_returns_first_list_item_or_scalar_value():
    """
    Test provider value normalization to a scalar.

    This function performs the following steps:
    1. Passes a populated list to `_first`.
    2. Passes an empty list to `_first`.
    3. Passes a scalar value and a missing value to `_first`.

    Asserts:
        - Populated lists return their first item.
        - Empty lists and missing values return an empty string.
        - Scalar values are returned unchanged.
    """
    assert search._first(['10.1234/example']) == '10.1234/example'
    assert search._first([]) == ''
    assert search._first('value') == 'value'
    assert search._first(None) == ''


def test_elsevier_rows_normalizes_provider_records():
    """
    Test Elsevier search row normalization.

    This function performs the following steps:
    1. Builds a raw Elsevier search results DataFrame.
    2. Converts it with `_elsevier_rows`.
    3. Reads the normalized row values.

    Asserts:
        - Provider-specific fields map to public paper columns.
        - Elsevier rows are marked with source and retrieved metadata status.
    """
    raw = pd.DataFrame([{
        'dc:identifier': 'SCOPUS_ID:1',
        'prism:doi': '10.1234/example',
        'dc:title': 'Paper title',
        'prism:publicationName': 'Journal',
        'prism:coverDate': '2024-01-01',
        'dc:creator': 'Author',
        'link': 'full-text-link',
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
    assert row['metadata_status'] == 'retrieved'


def test_core_headers_include_optional_authorization(monkeypatch):
    """
    Test CORE API request header construction.

    This function performs the following steps:
    1. Replaces CORE API key lookup with no key.
    2. Builds CORE headers.
    3. Replaces CORE API key lookup with a configured key and rebuilds headers.

    Asserts:
        - User-Agent is always included.
        - Authorization is only included when a CORE API key is available.
    """
    monkeypatch.setattr(search, '_core_api_key', lambda: None)
    assert search._core_headers() == {'User-Agent': 'PaperScraper/0.0.1'}

    monkeypatch.setattr(search, '_core_api_key', lambda: 'core-key')
    assert search._core_headers()['Authorization'] == 'Bearer core-key'


def test_core_field_helpers_extract_download_authors_journal_and_date():
    """
    Test CORE field extraction helpers.

    This function performs the following steps:
    1. Builds CORE work records with alternative field shapes.
    2. Extracts download URLs, authors, journal names, and dates.
    3. Compares the extracted values to expected strings.

    Asserts:
        - CORE download URLs use explicit URLs before generated API URLs.
        - Author names are joined with semicolons.
        - Journal dictionaries and scalar publisher fields are supported.
        - Publication dates fall back across supported date fields.
    """
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


def test_core_rows_normalizes_work_records():
    """
    Test CORE work row normalization.

    This function performs the following steps:
    1. Builds CORE work records with IDs, DOI lists, titles, and journal metadata.
    2. Converts them with `_core_rows`.
    3. Reads the normalized row values.

    Asserts:
        - CORE IDs and DOI-only records produce stable paper IDs.
        - CORE fields map to public paper columns.
        - CORE rows include PDF URLs and retrieved metadata status.
    """
    rows = search._core_rows([
        {
            'id': '123',
            'doi': ['10.1234/core'],
            'title': ['Core title'],
            'journal': {'name': 'Core Journal'},
            'publishedDate': '2023-01-01',
            'authors': [{'name': 'A. Author'}],
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
    assert rows.loc[0, 'metadata_status'] == 'retrieved'
    assert rows.loc[1, 'paper_id'] == 'doi:10.1234/no-id'


def test_core_search_paginates_and_stops_at_total_hits(monkeypatch):
    """
    Test CORE search pagination and stop conditions.

    This function performs the following steps:
    1. Replaces CORE request headers and HTTP GET calls with local fakes.
    2. Replaces tqdm with a local fake progress object.
    3. Calls `core_search` for more than one page.

    Asserts:
        - CORE requests use the expected limit and offset values.
        - Results from multiple pages are returned.
        - Pagination stops when total hits are reached.
    """

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    calls = []

    first_page = [{'id': str(index), 'title': f'paper {index}'} for index in range(100)]

    def fake_get(url, headers, params, timeout):
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


def test_core_search_stops_when_no_results(monkeypatch):
    """
    Test CORE search stop behavior when no results are returned.

    This function performs the following steps:
    1. Replaces CORE HTTP GET calls with an empty result payload.
    2. Replaces tqdm with a local fake progress object.
    3. Calls `core_search`.

    Asserts:
        - Empty CORE result pages return an empty normalized DataFrame.
    """

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {'results': []}

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    monkeypatch.setattr(search.requests, 'get', lambda *_, **__: FakeResponse())
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    assert search.core_search('missing', count=3).empty


def test_core_search_stops_when_page_is_short(monkeypatch):
    """
    Test CORE search stop behavior when a page has fewer results than requested.

    This function performs the following steps:
    1. Replaces CORE HTTP GET calls with a short result page.
    2. Replaces tqdm with a local fake progress object.
    3. Calls `core_search`.

    Asserts:
        - The short result page is returned.
        - CORE search stops after the short page.
    """

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {'results': [{'id': '1', 'title': 'one'}]}

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs['params'])
        return FakeResponse()

    monkeypatch.setattr(search.requests, 'get', fake_get)
    monkeypatch.setattr(search, 'tqdm', FakeTqdm)

    rows = search.core_search('solid electrolyte', count=3)

    assert rows['paper_id'].tolist() == ['core:1']
    assert len(calls) == 1


def test_search_for_papers_rejects_invalid_source():
    """
    Test validation of search source names.

    This function performs the following steps:
    1. Calls `search_for_papers` with an invalid source.
    2. Captures the expected exception.

    Asserts:
        - Invalid source names raise `ValueError`.
    """
    with pytest.raises(ValueError, match='source must be one of'):
        search.search_for_papers('query', source='bad')


def test_search_for_papers_merges_and_writes_results(tmp_path, monkeypatch, capsys):
    """
    Test merging search results into a papers CSV.

    This function performs the following steps:
    1. Replaces Elsevier search with a normalized one-row DataFrame.
    2. Calls `search_for_papers`.
    3. Reloads the written papers CSV.

    Asserts:
        - New search results are written to the requested papers CSV.
        - The summary reports one added result.
    """
    papers_path = tmp_path / 'papers.csv'
    rows = search._elsevier_rows(pd.DataFrame([{
        'dc:identifier': 'SCOPUS_ID:1',
        'prism:doi': '10.1234/new',
        'dc:title': 'New paper',
    }]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)

    search.search_for_papers('query', papers_path=str(papers_path), source='elsevier', count=1)

    written = pd.read_csv(papers_path, index_col=0)
    output = capsys.readouterr().out
    assert written.loc[0, 'paper_id'] == 'SCOPUS_ID:1'
    assert '1 new results and updated 0 existing rows' in output


def test_search_for_papers_merges_into_existing_papers_file(tmp_path, monkeypatch, capsys):
    """
    Test merging search results into an existing papers CSV.

    This function performs the following steps:
    1. Writes an existing papers CSV with one DOI.
    2. Replaces CORE search with a row using the same DOI and additional fields.
    3. Calls `search_for_papers`.

    Asserts:
        - The existing row is updated instead of duplicated.
        - The summary reports one updated row.
    """
    papers_path = tmp_path / 'papers.csv'
    search._core_rows([{
        'id': 'old',
        'doi': '10.1234/existing',
        'title': 'Existing title',
    }]).to_csv(papers_path)
    incoming = search._core_rows([{
        'id': 'new',
        'doi': '10.1234/existing',
        'title': 'Incoming title',
        'journal': 'Updated Journal',
    }])
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: incoming)

    search.search_for_papers('query', papers_path=str(papers_path), source='core', count=1)

    written = pd.read_csv(papers_path, index_col=0)
    output = capsys.readouterr().out
    assert len(written) == 1
    assert written.loc[0, 'title'] == 'Existing title'
    assert written.loc[0, 'journal'] == 'Updated Journal'
    assert '0 new results and updated 1 existing rows' in output


def test_search_for_papers_reports_zero_results_when_sources_are_empty(tmp_path, monkeypatch, capsys):
    """
    Test search output when all selected sources return no rows.

    This function performs the following steps:
    1. Replaces CORE search with an empty DataFrame.
    2. Calls `search_for_papers`.
    3. Captures the printed output.

    Asserts:
        - Empty search results print a zero-results message.
        - No papers CSV is written.
    """
    papers_path = tmp_path / 'papers.csv'
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: pd.DataFrame())

    search.search_for_papers('query', papers_path=str(papers_path), source='core', count=1)

    assert 'Document search found 0 new results.' in capsys.readouterr().out
    assert not papers_path.exists()


def test_search_for_papers_skips_failed_source_for_all_but_raises_for_selected_source(monkeypatch, tmp_path, capsys):
    """
    Test source failure handling during paper search.

    This function performs the following steps:
    1. Replaces Elsevier search with a failure and CORE search with one successful result.
    2. Calls `search_for_papers` with all sources.
    3. Calls `search_for_papers` with the failing source selected directly.

    Asserts:
        - Failed sources are skipped when searching all sources.
        - Directly selected failed sources re-raise their error.
        - Successful fallback source results are still written.
    """
    papers_path = tmp_path / 'papers.csv'
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: (_ for _ in ()).throw(RuntimeError('elsevier down')))
    monkeypatch.setattr(search, 'core_search', lambda *_, **__: search._core_rows([{'id': '1', 'title': 'Core paper'}]))

    search.search_for_papers('query', papers_path=str(papers_path), source='all', count=1)

    assert 'Elsevier search skipped: elsevier down' in capsys.readouterr().out
    assert papers_path.exists()

    with pytest.raises(RuntimeError, match='elsevier down'):
        search.search_for_papers('query', source='elsevier', count=1)


def test_search_for_papers_skips_failed_core_for_all_but_raises_for_core(monkeypatch, tmp_path, capsys):
    """
    Test CORE failure handling during paper search.

    This function performs the following steps:
    1. Replaces Elsevier search with one successful result.
    2. Replaces CORE search with a request failure.
    3. Calls `search_for_papers` with all sources and with CORE selected directly.

    Asserts:
        - Failed CORE searches are skipped when searching all sources.
        - Directly selected failed CORE searches re-raise their request error.
        - Successful fallback Elsevier results are still written.
    """
    papers_path = tmp_path / 'papers.csv'
    rows = search._elsevier_rows(pd.DataFrame([{'dc:identifier': 'SCOPUS_ID:1', 'dc:title': 'Elsevier paper'}]))
    monkeypatch.setattr(search, 'document_search', lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(search, '_elsevier_rows', lambda _: rows)
    monkeypatch.setattr(
        search,
        'core_search',
        lambda *_, **__: (_ for _ in ()).throw(search.requests.RequestException('core down')),
    )

    search.search_for_papers('query', papers_path=str(papers_path), source='all', count=1)

    assert 'CORE search skipped: core down' in capsys.readouterr().out
    assert papers_path.exists()

    with pytest.raises(search.requests.RequestException, match='core down'):
        search.search_for_papers('query', source='core', count=1)


@pytest.mark.network
def test_document_search_uses_real_elsevier_api_when_configured():
    """
    Test live Elsevier search with the user's configured API key.

    This function performs the following steps:
    1. Loads the user's real PaperScraper settings.
    2. Verifies an Elsevier API key is configured.
    3. Runs a one-result Elsevier document search.

    Asserts:
        - An Elsevier API key is configured.
        - The live search returns no more than one result.
    """
    loaded = search.load_settings()

    assert loaded.get('elsevier_api_key'), 'Set elsevier_api_key or ELSEVIER_API_KEY before running network tests.'
    assert len(search.document_search('solid electrolyte', count=1, get_all=False)) <= 1


@pytest.mark.network
def test_core_search_uses_real_core_api_when_configured():
    """
    Test live CORE search with the user's configured API key.

    This function performs the following steps:
    1. Loads the user's real PaperScraper settings and environment.
    2. Verifies a CORE API key is configured.
    3. Runs a one-result CORE search.

    Asserts:
        - A CORE API key is configured.
        - The live search returns no more than one result.
    """
    assert search._core_api_key(), 'Set core_api_key or CORE_API_KEY before running network tests.'
    assert len(search.core_search('solid electrolyte', count=1)) <= 1
