"""Unit tests for paperscraper.download.

This module tests download helper behavior without calling live APIs, including
Elsevier text/PDF helpers, open-access PDF source selection, filename creation,
configured source resolution, and corpus status updates.
"""

import json
import os

import pytest

import paperscraper.corpus as corpus
import paperscraper.download as download


def write_corpus(db_path, rows):
    """Write paper rows to a temporary test corpus."""
    with corpus.connect(db_path) as conn:
        for row in rows:
            corpus.upsert_paper(conn, row)


def read_corpus(db_path):
    """Read paper rows from a temporary test corpus."""
    with corpus.connect(db_path) as conn:
        return corpus.paper_rows(conn)


def test_elsevier_api_key_requires_and_uses_configured_api_key(monkeypatch):
    """
    Test Elsevier API key lookup from settings.

    This function performs the following steps:
    1. Replaces settings loading with no Elsevier API key.
    2. Calls `_elsevier_api_key` and captures the expected error.
    3. Replaces settings loading with a configured Elsevier API key.

    Asserts:
        - Missing Elsevier API keys raise `ValueError`.
        - Configured Elsevier API keys are returned.
    """
    monkeypatch.setattr(download, 'load_settings', lambda: {})
    with pytest.raises(ValueError, match='Elsevier API key is not configured'):
        download._elsevier_api_key()

    monkeypatch.setattr(download, 'load_settings', lambda: {'elsevier_api_key': 'elsevier-key'})
    assert download._elsevier_api_key() == 'elsevier-key'


def test_retrieve_document_clears_data_folder_and_writes_successful_document(tmp_path, monkeypatch):
    """
    Test Elsevier document retrieval into the temporary data folder.

    This function performs the following steps:
    1. Creates an existing file in a temporary data folder.
    2. Replaces the Elsevier content request helper with a local fake.
    3. Calls `retrieve_document`.

    Asserts:
        - Existing files in the data folder are removed.
        - Successful document reads write a new document file.
    """
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'old.json').write_text('old')

    class FakeResponse:
        def json(self):
            return {'originalText': 'new'}

    def fake_get_content(api_key, uri, accept, params):
        assert api_key == 'elsevier-key'
        assert uri == 'full-text-uri'
        assert accept == 'application/json'
        assert params == {'httpAccept': 'application/json'}
        return FakeResponse()

    monkeypatch.setattr(download, '_elsevier_api_key', lambda: 'elsevier-key')
    monkeypatch.setattr(download.elsevier, 'get_content', fake_get_content)

    download.retrieve_document('full-text-uri')

    assert not (data_dir / 'old.json').exists()
    assert json.loads((data_dir / 'elsevier_document.json').read_text()) == {'originalText': 'new'}


def test_json_to_text_and_elsevier_string_formatter(tmp_path):
    """
    Test Elsevier JSON text extraction and wrapper cleanup.

    This function performs the following steps:
    1. Writes JSON files containing valid text, nested valid text, and a failed text dictionary.
    2. Reads the files with `json_to_text`.
    3. Formats Elsevier text with duplicate section wrappers and an AWS URL wrapper.

    Asserts:
        - String `originalText` values are returned.
        - Dictionary `originalText` values return `failed`.
        - Duplicate wrapper sections and AWS prefixes are removed.
    """
    text_path = tmp_path / 'text.json'
    nested_text_path = tmp_path / 'nested_text.json'
    failed_path = tmp_path / 'failed.json'
    text_path.write_text(json.dumps({'originalText': 'paper text'}))
    nested_text_path.write_text(json.dumps({'full-text-retrieval-response': {'originalText': 'nested text'}}))
    failed_path.write_text(json.dumps({'originalText': {'bad': 'text'}}))

    assert download.json_to_text(str(text_path)) == 'paper text'
    assert download.json_to_text(str(nested_text_path)) == 'nested text'
    assert download.json_to_text(str(failed_path)) == 'failed'
    assert download.elsevier_string_formatter('A Acknowledgements clean Acknowledgements') == ' clean '
    assert download.elsevier_string_formatter('A References clean References') == ' clean '
    assert download.elsevier_string_formatter('prefix amazonaws.com/key paper text') == ' paper text'


def test_full_text_uri_and_download_text_success_and_failure(tmp_path, monkeypatch):
    """
    Test Elsevier full-text URI extraction and text download behavior.

    This function performs the following steps:
    1. Extracts full-text URIs from valid and invalid paper links.
    2. Replaces document retrieval with a helper that writes a temporary JSON document.
    3. Downloads text to a target file and checks failure paths.

    Asserts:
        - Full-text links produce a URI.
        - Missing full-text links return None.
        - Successful text downloads write formatted text.
        - Failed provider text returns False.
    """
    monkeypatch.chdir(tmp_path)
    paper = {'elsevier_link': "some 'full-text-uri' full-text"}
    assert download._full_text_uri(paper) == 'full-text-uri'
    assert download._full_text_uri({'elsevier_link': 'abstract-only'}) is None
    assert download._full_text_uri({
        'elsevier_link': 'https://api.elsevier.com/content/article/eid/1-s2.0-S1005030226004123',
    }) == 'https://api.elsevier.com/content/article/eid/1-s2.0-S1005030226004123'
    assert download._full_text_uri({
        'elsevier_link': 'https://api.elsevier.com/content/abstract/scopus_id/105042507561',
    }) is None
    assert download._full_text_uri({'doi': '10.1234/a b', 'elsevier_link': 'abstract-only'}) == (
        'https://api.elsevier.com/content/article/doi/10.1234%2Fa+b'
    )
    assert download._full_text_uri({
        'elsevier_link': [
            {'@ref': 'self', '@href': 'self-link'},
            {'@ref': 'full-text', '@href': 'full-text-link'},
        ],
    }) == 'full-text-link'
    assert download._full_text_uri({
        'elsevier_link': '[{"@ref": "full-text", "@href": "json-full-text-link"}]',
    }) == 'json-full-text-link'

    def fake_retrieve_document(_):
        os.makedirs('data', exist_ok=True)
        with open('data/doc.json', 'w', encoding='utf-8') as f:
            json.dump({'originalText': 'downloaded text'}, f)

    monkeypatch.setattr(download, 'retrieve_document', fake_retrieve_document)
    out_path = tmp_path / 'paper.txt'
    assert download._download_text(paper, str(out_path)) is True
    assert out_path.read_text() == 'downloaded text'

    monkeypatch.setattr(download, 'json_to_text', lambda _: 'failed')
    assert download._download_text(paper, str(out_path)) is False
    assert download._download_text({'elsevier_link': ''}, str(out_path)) is False


def test_pdf_urls_and_safe_filename_build_expected_values():
    """
    Test PDF URL and safe filename creation.

    This function performs the following steps:
    1. Builds Elsevier PDF URLs from DOI and full-text URI values.
    2. Builds safe filenames from DOI, CORE ID, paper ID, and empty rows.
    3. Compares outputs to expected strings.

    Asserts:
        - DOI values are URL-quoted for Elsevier PDF endpoints.
        - Full-text URIs are included when present.
        - Filename stems are sanitized and fall back in priority order.
    """
    paper = {'doi': '10.1234/a b', 'elsevier_link': "x 'uri' full-text"}

    assert download._pdf_urls(paper) == ['https://api.elsevier.com/content/article/doi/10.1234%2Fa+b', 'uri']
    assert download._safe_filename({'doi': '10.1234/a b'}) == '10.1234_a_b'
    assert download._safe_filename({'doi': '', 'core_id': 'core/1'}) == 'core_1'
    assert download._safe_filename({'doi': '', 'core_id': '', 'paper_id': 'paper:1'}) == 'paper_1'
    assert download._safe_filename({}) == 'paper'


def test_download_url_to_pdf_saves_only_pdf_responses(tmp_path, monkeypatch):
    """
    Test generic URL-to-PDF download behavior.

    This function performs the following steps:
    1. Calls `_download_url_to_pdf` with a missing URL.
    2. Replaces HTTP GET with PDF, non-PDF, error status, and request exception responses.
    3. Checks the returned success and error details.

    Asserts:
        - Missing URLs fail.
        - PDF-like responses are written to disk.
        - Non-PDF, HTTP error, and request exception responses fail with messages.
    """

    class FakeResponse:
        def __init__(self, status_code=200, content=b'%PDF data', content_type='application/pdf'):
            self.status_code = status_code
            self.content = content
            self.headers = {'Content-Type': content_type}

    out_path = tmp_path / 'paper.pdf'
    assert download._download_url_to_pdf('', str(out_path)) == (False, 'missing URL')

    monkeypatch.setattr(download.requests, 'get', lambda *_, **__: FakeResponse())
    assert download._download_url_to_pdf('https://example.com/pdf', str(out_path)) == (True, '')
    assert out_path.read_bytes() == b'%PDF data'

    monkeypatch.setattr(download.requests, 'get', lambda *_, **__: FakeResponse(content=b'html', content_type='text/html'))
    ok, error = download._download_url_to_pdf('https://example.com/html', str(out_path))
    assert ok is False
    assert 'non-PDF response' in error

    monkeypatch.setattr(download.requests, 'get', lambda *_, **__: FakeResponse(status_code=404))
    ok, error = download._download_url_to_pdf('https://example.com/missing', str(out_path))
    assert ok is False
    assert '404 from' in error

    monkeypatch.setattr(
        download.requests,
        'get',
        lambda *_, **__: (_ for _ in ()).throw(download.requests.RequestException('network down')),
    )
    assert download._download_url_to_pdf('https://example.com/error', str(out_path)) == (False, 'network down')


def test_download_unpaywall_pdf_handles_missing_config_and_pdf_candidates(tmp_path, monkeypatch):
    """
    Test Unpaywall PDF download behavior.

    This function performs the following steps:
    1. Checks missing DOI and missing email paths.
    2. Replaces Unpaywall metadata lookup with PDF candidates.
    3. Replaces URL download helper so the second candidate succeeds.

    Asserts:
        - Missing DOI and missing email fail with useful messages.
        - Candidate PDF URLs are tried in order without duplicates.
        - The successful PDF candidate URL is returned.
    """

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                'best_oa_location': {'url_for_pdf': 'https://example.com/one.pdf'},
                'oa_locations': [
                    {'url_for_pdf': 'https://example.com/one.pdf'},
                    {'url_for_pdf': 'https://example.com/two.pdf'},
                ],
            }

    monkeypatch.setattr(download, '_unpaywall_email', lambda settings=None: None)
    assert download._download_unpaywall_pdf({'doi': ''}, str(tmp_path / 'paper.pdf')) == (False, 'missing DOI')
    assert 'Unpaywall email is not configured' in download._download_unpaywall_pdf(
        {'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')
    )[1]

    tried = []
    monkeypatch.setattr(download, '_unpaywall_email', lambda settings=None: 'person@example.com')
    monkeypatch.setattr(download.requests, 'get', lambda *_, **__: FakeResponse())

    def fake_download(url, filepath):
        tried.append(url)
        return (url.endswith('two.pdf'), 'failed')

    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download)

    assert download._download_unpaywall_pdf({'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')) == (
        True,
        'https://example.com/two.pdf',
    )
    assert tried == ['https://example.com/one.pdf', 'https://example.com/two.pdf']


def test_core_headers_and_core_pdf_download(tmp_path, monkeypatch):
    """
    Test CORE headers and PDF download source selection.

    This function performs the following steps:
    1. Builds CORE headers without and with an API key.
    2. Replaces URL-to-PDF download with a local fake.
    3. Downloads a CORE PDF from stored URL and generated CORE ID URL paths.

    Asserts:
        - CORE authorization is included only when configured.
        - Stored PDF URLs are tried before generated CORE download URLs.
        - Missing CORE download URLs fail.
    """
    monkeypatch.setattr(download, 'load_settings', lambda: {})
    monkeypatch.delenv('CORE_API_KEY', raising=False)
    assert download._core_headers() == {'User-Agent': 'PaperScraper/0.0.1'}

    monkeypatch.setenv('CORE_API_KEY', 'core-key')
    assert download._core_headers()['Authorization'] == 'Bearer core-key'

    tried = []

    def fake_download(url, filepath, headers=None):
        tried.append((url, headers))
        return True, ''

    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download)
    assert download._download_core_pdf({'pdf_url': 'https://example.com/pdf', 'core_id': '123'}, str(tmp_path / 'paper.pdf')) == (
        True,
        'https://example.com/pdf',
    )
    assert tried[0][0] == 'https://example.com/pdf'
    assert tried[0][1]['Authorization'] == 'Bearer core-key'

    tried.clear()
    assert download._download_core_pdf({'pdf_url': '', 'core_id': 'abc 123'}, str(tmp_path / 'paper.pdf')) == (
        True,
        'https://api.core.ac.uk/v3/works/abc%20123/download',
    )
    assert download._download_core_pdf({'pdf_url': '', 'core_id': ''}, str(tmp_path / 'paper.pdf')) == (
        False,
        'no CORE download URL found',
    )


def test_abstract_helpers_clean_provider_text_and_try_sources(monkeypatch):
    """
    Test abstract cleanup and provider source selection.

    This function performs the following steps:
    1. Cleans HTML, entities, whitespace, and list values from abstract text.
    2. Extracts abstract text from nested provider payloads.
    3. Replaces CORE and Elsevier abstract fetchers with local helpers.
    4. Calls `_download_abstract` for CORE, Elsevier, and missing-source rows.

    Asserts:
        - Provider abstract text is normalized to plain text.
        - CORE abstracts are preferred when a CORE ID is available.
        - Elsevier abstracts are used when configured and CORE is unavailable.
        - Missing sources return a useful failure.
    """
    assert download._clean_abstract(' A&nbsp;<b>solid</b>\n electrolyte ') == 'A solid electrolyte'
    assert download._clean_abstract(['First', 'Second']) == 'First Second'
    assert download._abstract_from_mapping({'outer': {'dc:description': '<p>Nested abstract</p>'}}) == 'Nested abstract'

    calls = []
    monkeypatch.setattr(download, '_download_core_abstract', lambda paper: calls.append('core') or (True, 'core', 'CORE abstract'))
    monkeypatch.setattr(download, '_download_elsevier_abstract',
                        lambda paper: calls.append('elsevier') or (True, 'elsevier', 'Elsevier abstract'))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)

    assert download._download_abstract({'core_id': '123'}) == (True, 'core', 'CORE abstract')
    assert download._download_abstract({'doi': '10.1234/example'}) == (True, 'elsevier', 'Elsevier abstract')

    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    assert download._download_abstract({'paper_id': 'missing'}) == (False, 'no abstract source available', '')
    assert calls == ['core', 'elsevier']


def test_core_and_elsevier_abstract_downloads_parse_provider_payloads(monkeypatch):
    """
    Test CORE and Elsevier abstract metadata requests.

    This function performs the following steps:
    1. Replaces CORE HTTP requests with successful, missing, and error responses.
    2. Replaces Elsevier content requests with successful and missing-abstract responses.
    3. Calls the provider-specific abstract download helpers.

    Asserts:
        - CORE and Elsevier abstract payloads are parsed into plain text.
        - Missing IDs, missing abstracts, and HTTP errors return failure details.
    """

    class FakeResponse:
        def __init__(self, payload=None, status_code=200):
            self.payload = payload or {}
            self.status_code = status_code

        def json(self):
            return self.payload

    core_calls = []

    def fake_get(url, headers, timeout):
        core_calls.append((url, headers, timeout))
        return FakeResponse({'abstract': '<p>CORE abstract</p>'})

    monkeypatch.setattr(download.requests, 'get', fake_get)
    monkeypatch.setattr(download, '_core_headers', lambda: {'Authorization': 'Bearer core-key'})

    assert download._download_core_abstract({'core_id': 'abc 123'}) == (True, 'core', 'CORE abstract')
    assert core_calls == [(
        'https://api.core.ac.uk/v3/works/abc%20123',
        {'Authorization': 'Bearer core-key'},
        60,
    )]
    assert download._download_core_abstract({'core_id': ''}) == (False, 'missing CORE ID', '')

    monkeypatch.setattr(download.requests, 'get', lambda *_, **__: FakeResponse(status_code=404))
    assert download._download_core_abstract({'core_id': '123'}) == (False, '404 from CORE', '')

    monkeypatch.setattr(download, '_elsevier_api_key', lambda: 'elsevier-key')
    monkeypatch.setattr(
        download.elsevier,
        'get_content',
        lambda api_key, url, accept, params: FakeResponse({'abstracts-retrieval-response': {
            'coredata': {'dc:description': '<p>Elsevier abstract</p>'},
        }}),
    )
    assert download._download_elsevier_abstract({
        'doi': '10.1234/example',
        'elsevier_link': 'https://api.elsevier.com/content/abstract/scopus_id/1',
    }) == (True, 'elsevier', 'Elsevier abstract')

    monkeypatch.setattr(download.elsevier, 'get_content', lambda *_, **__: FakeResponse({'no': 'abstract'}))
    ok, error, abstract = download._download_elsevier_abstract({'doi': '10.1234/example'})
    assert ok is False
    assert 'no abstract in response' in error
    assert abstract == ''


def test_configured_sources_resolves_all_deduplicates_and_rejects_invalid(monkeypatch):
    """
    Test PDF source configuration resolution.

    This function performs the following steps:
    1. Resolves `all` with configured Unpaywall, CORE, and Elsevier settings.
    2. Resolves an explicit list containing duplicate sources.
    3. Resolves an invalid source.

    Asserts:
        - `all` expands only to configured sources.
        - Explicit source lists are de-duplicated while preserving order.
        - Invalid source names raise `ValueError`.
    """
    monkeypatch.setattr(
        download,
        'load_settings',
        lambda: {
            'unpaywall_email': 'person@example.com',
            'core_api_key': 'core-key',
            'elsevier_api_key': 'elsevier-key',
        },
    )

    assert download._configured_sources(['all']) == ['unpaywall', 'core', 'elsevier']
    assert download._configured_sources(['core', 'core', 'unpaywall']) == ['core', 'unpaywall']
    with pytest.raises(ValueError, match='download source must be one of'):
        download._configured_sources(['bad'])


def test_download_pdf_from_sources_handles_existing_success_and_failures(tmp_path, monkeypatch):
    """
    Test trying PDF download sources in order.

    This function performs the following steps:
    1. Replaces source downloaders with failing and successful local helpers.
    2. Calls `_download_pdf_from_sources`.
    3. Replaces all source downloaders with failing helpers.

    Asserts:
        - Sources are tried until one succeeds.
        - Errors are collected when all sources fail.
    """
    pdf_path = tmp_path / 'paper.pdf'
    monkeypatch.setattr(download, '_download_unpaywall_pdf', lambda *_: (False, 'no oa pdf'))
    monkeypatch.setattr(download, '_download_core_pdf', lambda *_: (True, 'https://core/pdf'))
    assert download._download_pdf_from_sources({}, str(pdf_path), ['unpaywall', 'core']) == (
        True,
        'core',
        'https://core/pdf',
    )

    monkeypatch.setattr(download, '_download_core_pdf', lambda *_: (False, 'core failed'))
    ok, error, detail = download._download_pdf_from_sources({}, str(pdf_path), ['unpaywall', 'core'])
    assert ok is False
    assert 'unpaywall: no oa pdf' in error
    assert 'core: core failed' in error
    assert detail == ''


def test_should_try_elsevier_text_detects_full_text_links():
    """
    Test detection of Elsevier full-text availability.

    This function performs the following steps:
    1. Checks rows with a full-text Elsevier link and DOI fallback.
    2. Checks rows with abstract-only and missing links.

    Asserts:
        - Full-text links and DOI rows return True.
        - Missing or non-full-text links return False.
    """
    assert download._should_try_elsevier_text({'elsevier_link': "has 'full-text-link' full-text"}) is True
    assert download._should_try_elsevier_text({
        'elsevier_link': 'https://api.elsevier.com/content/article/eid/1-s2.0-S1005030226004123',
    }) is True
    assert download._should_try_elsevier_text({'doi': '10.1234/example', 'elsevier_link': 'abstract only'}) is True
    assert download._should_try_elsevier_text({'elsevier_link': 'abstract only'}) is False
    assert download._should_try_elsevier_text({'elsevier_link': None}) is False


def test_download_papers_validates_configuration(tmp_path, monkeypatch):
    """
    Test download configuration validation.

    This function performs the following steps:
    1. Calls `download_papers` with an invalid format.
    2. Disables Elsevier text availability and requests text downloads.
    3. Disables all PDF sources and requests PDF downloads.

    Asserts:
        - Invalid download formats raise `ValueError`.
        - Text downloads require an Elsevier API key.
        - PDF downloads require at least one configured PDF source.
    """
    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'paper-1'}])

    with pytest.raises(ValueError, match='download_format must be one of'):
        download.download_papers(str(db_path), download_format='bad')

    monkeypatch.setattr(download, '_configured_sources', lambda _: [])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    with pytest.raises(ValueError, match='Elsevier text download requires'):
        download.download_papers(str(db_path), download_format='text')

    with pytest.raises(ValueError, match='No PDF download sources are configured'):
        download.download_papers(str(db_path), download_format='pdf')


def test_download_papers_updates_text_and_pdf_statuses(tmp_path, monkeypatch, capsys):
    """
    Test downloading text and PDFs for papers in a corpus database.

    This function performs the following steps:
    1. Writes a corpus database with one Elsevier full-text row.
    2. Replaces configured source checks and download helpers with local fakes.
    3. Calls `download_papers` for both text and PDF downloads.

    Asserts:
        - Text and PDF paths are left empty because corpus storage is authoritative.
        - Text and PDF statuses are marked as succeeded.
        - PDF source URL details are copied back to `pdf_url`.
        - Downloaded text and PDF files are stored as corpus assets.
    """

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{
        'paper_id': 'paper:1',
        'doi': '10.1234/example',
        'elsevier_link': 'has full-text link',
    }])
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['unpaywall'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)

    def fake_download_text(paper, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('text')
        return True

    def fake_download_pdf_from_sources(paper, filepath, sources):
        with open(filepath, 'wb') as f:
            f.write(b'%PDF text')
        return True, 'unpaywall', 'https://oa/pdf'

    monkeypatch.setattr(download, '_download_text', fake_download_text)
    monkeypatch.setattr(download, '_download_pdf_from_sources', fake_download_pdf_from_sources)

    download.download_papers(str(db_path), download_format='both', download_abstract=False)

    papers = read_corpus(db_path)
    output = capsys.readouterr().out
    with corpus.connect(db_path) as conn:
        text_asset = corpus.get_asset(conn, 'paper:1', 'text')
        pdf_asset = corpus.get_asset(conn, 'paper:1', 'pdf')
    assert papers[0]['text_download_status'] == 'succeeded'
    assert papers[0]['pdf_download_status'] == 'succeeded'
    assert papers[0]['text_source'] == 'elsevier'
    assert papers[0]['pdf_source'] == 'unpaywall'
    assert papers[0]['pdf_url'] == 'https://oa/pdf'
    assert papers[0]['text_path'] == ''
    assert papers[0]['pdf_path'] == ''
    assert not (tmp_path / 'downloads').exists()
    assert text_asset['content'] == b'text'
    assert pdf_asset['content'].startswith(b'%PDF')
    assert 'Download complete: 1 text files, 1 PDFs, 0 abstracts downloaded.' in output


def test_download_papers_downloads_abstract_by_default(tmp_path, monkeypatch, capsys):
    """
    Test default abstract downloading during corpus downloads.

    This function performs the following steps:
    1. Writes a corpus database with one paper row.
    2. Replaces abstract downloading and progress reporting with local helpers.
    3. Calls `download_papers` without passing an abstract option.
    4. Reads the updated paper row and stored abstract asset.

    Asserts:
        - Abstract downloading runs by default.
        - Abstract status and source are recorded on the paper row.
        - Abstract text is stored as a compressed corpus asset.
    """

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'paper:abstract'}])
    monkeypatch.setattr(download, '_configured_sources', lambda _: [])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(download, '_download_abstract', lambda paper: (True, 'core', 'abstract text'))

    download.download_papers(str(db_path), download_format='text')

    papers = read_corpus(db_path)
    output = capsys.readouterr().out
    with corpus.connect(db_path) as conn:
        abstract_asset = corpus.get_asset(conn, 'paper:abstract', 'abstract')
    assert papers[0]['abstract_download_status'] == 'succeeded'
    assert papers[0]['abstract_source'] == 'core'
    assert abstract_asset['content'] == b'abstract text'
    assert 'Download complete: 0 text files, 0 PDFs, 1 abstracts downloaded.' in output


def test_download_papers_skips_abstract_download_when_asset_already_exists(tmp_path, monkeypatch, capsys):
    """
    Test avoiding duplicate abstract downloads.

    This function performs the following steps:
    1. Writes a corpus database with one paper and an existing abstract asset.
    2. Replaces abstract downloading with a helper that fails if called.
    3. Calls `download_papers` with default abstract downloading enabled.
    4. Reads the paper row and abstract asset.

    Asserts:
        - Existing abstract assets prevent provider abstract downloads.
        - The existing abstract asset remains unchanged.
        - Abstract status is marked succeeded and keeps the existing source.
    """

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.add_asset(
            conn,
            {'paper_id': 'paper:abstract', 'abstract_source': 'search'},
            'search abstract text',
            role='abstract',
            kind='text',
            mime_type='text/plain',
            source='search',
        )
    monkeypatch.setattr(download, '_configured_sources', lambda _: [])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(
        download,
        '_download_abstract',
        lambda paper: (_ for _ in ()).throw(AssertionError('abstract should not be downloaded twice')),
    )

    download.download_papers(str(db_path), download_format='text')

    papers = read_corpus(db_path)
    output = capsys.readouterr().out
    with corpus.connect(db_path) as conn:
        abstract_asset = corpus.get_asset(conn, 'paper:abstract', 'abstract')
    assert papers[0]['abstract_download_status'] == 'succeeded'
    assert papers[0]['abstract_source'] == 'search'
    assert abstract_asset['content'] == b'search abstract text'
    assert 'Download complete: 0 text files, 0 PDFs, 0 abstracts downloaded.' in output


def test_retrieve_document_reports_failed_read(tmp_path, monkeypatch, capsys):
    """
    Test Elsevier document retrieval when the provider read fails.

    This function performs the following steps:
    1. Replaces the Elsevier content request helper with a fake request failure.
    2. Calls `retrieve_document`.
    3. Captures the printed output.

    Asserts:
        - A failed document read prints a failure message.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(download, '_elsevier_api_key', lambda: 'elsevier-key')
    monkeypatch.setattr(
        download.elsevier,
        'get_content',
        lambda *_, **__: (_ for _ in ()).throw(download.requests.RequestException('read failed')),
    )

    download.retrieve_document('full-text-uri')

    assert 'Read document failed.' in capsys.readouterr().out


def test_full_text_uri_and_download_text_handle_malformed_or_empty_retrieval(tmp_path, monkeypatch):
    """
    Test malformed full-text links and empty text retrieval folders.

    This function performs the following steps:
    1. Extracts a URI from a malformed full-text link.
    2. Replaces document retrieval with a helper that creates an empty data folder.
    3. Calls `_download_text`.

    Asserts:
        - Malformed full-text links return None.
        - Text downloads fail when no temporary document is retrieved.
    """
    monkeypatch.chdir(tmp_path)
    assert download._full_text_uri({'elsevier_link': 'full-text without quoted uri'}) is None

    def fake_retrieve_document(_):
        os.makedirs('data', exist_ok=True)

    monkeypatch.setattr(download, 'retrieve_document', fake_retrieve_document)

    assert download._download_text({'elsevier_link': "x 'uri' full-text"}, str(tmp_path / 'paper.txt')) is False


def test_download_pdf_requires_key_and_handles_success_and_failures(tmp_path, monkeypatch, capsys):
    """
    Test Elsevier PDF download behavior.

    This function performs the following steps:
    1. Calls `_download_pdf` without an Elsevier API key.
    2. Replaces settings and HTTP GET calls with local fake responses.
    3. Exercises HTTP error, request exception, non-PDF, and PDF success paths.

    Asserts:
        - Missing Elsevier API keys raise `ValueError`.
        - PDF responses are written to disk and return True.
        - Failed candidate responses return False and print the last error.
    """

    class FakeResponse:
        def __init__(self, status_code=200, content=b'%PDF data', content_type='application/pdf'):
            self.status_code = status_code
            self.content = content
            self.headers = {'Content-Type': content_type}

    out_path = tmp_path / 'paper.pdf'
    monkeypatch.setattr(download, 'load_settings', lambda: {})
    with pytest.raises(ValueError, match='Elsevier API key is not configured'):
        download._download_pdf({'doi': '10.1234/example'}, str(out_path))

    monkeypatch.setattr(download, 'load_settings', lambda: {'elsevier_api_key': 'elsevier-key'})
    monkeypatch.setattr(download, '_pdf_urls', lambda _: ['bad-status', 'bad-request', 'bad-content', 'good-pdf'])

    def fake_get_content(api_key, url, accept, params):
        assert api_key == 'elsevier-key'
        assert accept == 'application/pdf'
        assert params == {'httpAccept': 'application/pdf'}
        if url == 'bad-status':
            error = download.requests.HTTPError('forbidden')
            error.response = FakeResponse(status_code=403)
            raise error
        if url == 'bad-request':
            raise download.requests.RequestException('network down')
        if url == 'bad-content':
            return FakeResponse(content=b'html', content_type='text/html')
        return FakeResponse()

    monkeypatch.setattr(download.elsevier, 'get_content', fake_get_content)
    assert download._download_pdf({'paper_id': 'paper-1'}, str(out_path)) is True
    assert out_path.read_bytes() == b'%PDF data'

    monkeypatch.setattr(
        download.elsevier,
        'get_content',
        lambda *_, **__: FakeResponse(content=b'html', content_type='text/html'),
    )
    assert download._download_pdf({'paper_id': 'paper-1'}, str(out_path)) is False
    assert 'non-PDF response' in capsys.readouterr().out


def test_download_unpaywall_pdf_handles_metadata_errors_and_missing_candidates(tmp_path, monkeypatch):
    """
    Test Unpaywall metadata error and no-candidate paths.

    This function performs the following steps:
    1. Replaces Unpaywall email lookup with a configured email.
    2. Replaces metadata requests with an HTTP error response and a request exception.
    3. Replaces metadata requests with a response containing no PDF URLs.

    Asserts:
        - HTTP error responses return an Unpaywall status message.
        - Request exceptions return the exception message.
        - Metadata without PDF URLs returns a no-URL message.
    """

    class ErrorResponse:
        status_code = 500

    class EmptyResponse:
        status_code = 200

        def json(self):
            return {'best_oa_location': None, 'oa_locations': []}

    monkeypatch.setattr(download, '_unpaywall_email', lambda settings=None: 'person@example.com')
    monkeypatch.setattr(download.requests, 'get', lambda *_, **__: ErrorResponse())
    assert download._download_unpaywall_pdf({'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')) == (
        False,
        '500 from Unpaywall',
    )

    monkeypatch.setattr(
        download.requests,
        'get',
        lambda *_, **__: (_ for _ in ()).throw(download.requests.RequestException('network down')),
    )
    assert download._download_unpaywall_pdf({'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')) == (
        False,
        'network down',
    )

    monkeypatch.setattr(download.requests, 'get', lambda *_, **__: EmptyResponse())
    assert download._download_unpaywall_pdf({'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')) == (
        False,
        'no Unpaywall PDF URL found',
    )


def test_download_core_pdf_returns_last_error_when_candidates_fail(tmp_path, monkeypatch):
    """
    Test CORE PDF download failure details.

    This function performs the following steps:
    1. Replaces URL-to-PDF download with a helper that always fails.
    2. Calls `_download_core_pdf` with stored URL and CORE ID candidates.
    3. Checks the returned error message.

    Asserts:
        - The final failed candidate error is returned.
    """
    monkeypatch.setattr(download, '_download_url_to_pdf', lambda *_, **__: (False, 'candidate failed'))

    assert download._download_core_pdf(
        {'pdf_url': 'https://example.com/pdf', 'core_id': '123'},
        str(tmp_path / 'paper.pdf'),
    ) == (False, 'candidate failed')


def test_elsevier_configured_reads_settings(monkeypatch):
    """
    Test Elsevier configuration detection.

    This function performs the following steps:
    1. Replaces settings loading with no Elsevier API key.
    2. Replaces settings loading with an Elsevier API key.
    3. Calls `_elsevier_configured` for both cases.

    Asserts:
        - Missing Elsevier API keys return False.
        - Configured Elsevier API keys return True.
    """
    monkeypatch.setattr(download, 'load_settings', lambda: {})
    assert download._elsevier_configured() is False

    monkeypatch.setattr(download, 'load_settings', lambda: {'elsevier_api_key': 'elsevier-key'})
    assert download._elsevier_configured() is True


def test_download_pdf_from_sources_collects_exceptions(tmp_path, monkeypatch):
    """
    Test PDF source download exception handling.

    This function performs the following steps:
    1. Replaces the Unpaywall downloader with a helper that raises an exception.
    2. Calls `_download_pdf_from_sources`.
    3. Checks the aggregated error string.

    Asserts:
        - Exceptions from individual source downloaders are captured as source errors.
    """
    monkeypatch.setattr(download, '_download_unpaywall_pdf', lambda *_: (_ for _ in ()).throw(RuntimeError('boom')))

    ok, error, detail = download._download_pdf_from_sources({}, str(tmp_path / 'paper.pdf'), ['unpaywall'])

    assert ok is False
    assert error == 'unpaywall: boom'
    assert detail == ''


def test_download_papers_records_text_and_pdf_failures(tmp_path, monkeypatch):
    """
    Test failed text and PDF downloads in the main download loop.

    This function performs the following steps:
    1. Writes a corpus database with one Elsevier full-text row.
    2. Replaces download helpers with failing local helpers.
    3. Calls `download_papers` for both text and PDF downloads.

    Asserts:
        - Failed text downloads are marked failed.
        - Failed PDF downloads are marked failed with the source error.
    """

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{
        'paper_id': 'paper:1',
        'doi': '10.1234/example',
        'elsevier_link': 'has full-text link',
    }])
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['unpaywall'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(download, '_download_text', lambda *_: False)
    monkeypatch.setattr(download, '_download_pdf_from_sources', lambda *_: (False, 'no pdf', ''))

    download.download_papers(str(db_path), download_format='both', download_abstract=False)

    papers = read_corpus(db_path)
    assert papers[0]['text_download_status'] == 'failed'
    assert papers[0]['pdf_download_status'] == 'failed'
    assert papers[0]['last_error'] == 'no pdf'


def test_download_papers_records_initial_text_download_exception(tmp_path, monkeypatch):
    """
    Test exception handling during the initial Elsevier text download attempt.

    This function performs the following steps:
    1. Writes a corpus database with one Elsevier full-text row.
    2. Replaces text downloading with a helper that raises an exception.
    3. Calls `download_papers` for text downloads.

    Asserts:
        - The text download status is marked failed.
        - The exception message is recorded as the last error.
    """

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{
        'paper_id': 'paper:text-error',
        'doi': '10.1234/text-error',
        'elsevier_link': 'has full-text link',
    }])
    monkeypatch.setattr(download, '_configured_sources', lambda _: [])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(download, '_download_text', lambda *_: (_ for _ in ()).throw(RuntimeError('text exploded')))

    download.download_papers(str(db_path), download_format='text', download_abstract=False)

    papers = read_corpus(db_path)
    assert papers[0]['text_download_status'] == 'failed'
    assert papers[0]['last_error'] == 'text exploded'


def test_download_papers_records_download_exceptions_and_elsevier_text_after_oa_pdf(tmp_path, monkeypatch):
    """
    Test exception handling and Elsevier text retrieval after an open-access PDF succeeds.

    This function performs the following steps:
    1. Writes corpus rows covering text exception, PDF exception, and OA PDF success rows.
    2. Replaces download helpers with local helpers that raise or succeed by row.
    3. Calls `download_papers` for PDF downloads.

    Asserts:
        - PDF exceptions are recorded as failed PDF downloads.
        - Open-access PDF success can trigger Elsevier text retrieval even in PDF mode.
        - Text exceptions after open-access PDF success are recorded as failed text downloads.
    """

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [
        {'paper_id': 'paper:pdf-error', 'doi': '10.1234/pdf-error'},
        {'paper_id': 'paper:oa-text', 'doi': '10.1234/oa-text', 'elsevier_link': 'has full-text link'},
    ])
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['core'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)

    def fake_download_pdf_from_sources(paper, filepath, sources):
        if paper['paper_id'] == 'paper:pdf-error':
            raise RuntimeError('pdf exploded')
        with open(filepath, 'wb') as f:
            f.write(b'%PDF data')
        return True, 'core', 'https://core/pdf'

    def fake_download_text(paper, filepath):
        raise RuntimeError('text exploded')

    monkeypatch.setattr(download, '_download_pdf_from_sources', fake_download_pdf_from_sources)
    monkeypatch.setattr(download, '_download_text', fake_download_text)

    download.download_papers(str(db_path), download_format='pdf', download_abstract=False)

    papers = read_corpus(db_path)
    assert papers[0]['pdf_download_status'] == 'failed'
    assert papers[0]['last_error'] == 'pdf exploded'
    assert papers[1]['pdf_download_status'] == 'succeeded'
    assert papers[1]['text_download_status'] == 'failed'
    assert papers[1]['last_error'] == 'text exploded'


def test_download_papers_downloads_elsevier_text_after_oa_pdf_success(tmp_path, monkeypatch):
    """
    Test successful Elsevier text retrieval after an open-access PDF download.

    This function performs the following steps:
    1. Writes a corpus database with one Elsevier full-text row.
    2. Replaces PDF and text download helpers with successful local helpers.
    3. Calls `download_papers` for PDF downloads.

    Asserts:
        - The PDF download is marked succeeded.
        - The follow-up Elsevier text download is marked succeeded.
        - The text source is recorded while the file path remains empty.
    """

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def update(self, _):
            return None

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{
        'paper_id': 'paper:oa-text',
        'doi': '10.1234/oa-text',
        'elsevier_link': 'has full-text link',
    }])
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['core'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)

    def fake_download_pdf_from_sources(paper, filepath, sources):
        with open(filepath, 'wb') as f:
            f.write(b'%PDF data')
        return True, 'core', 'https://core/pdf'

    monkeypatch.setattr(download, '_download_pdf_from_sources', fake_download_pdf_from_sources)

    def fake_download_text(paper, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('elsevier text')
        return True

    monkeypatch.setattr(download, '_download_text', fake_download_text)

    download.download_papers(str(db_path), download_format='pdf', download_abstract=False)

    papers = read_corpus(db_path)
    assert papers[0]['pdf_download_status'] == 'succeeded'
    assert papers[0]['text_download_status'] == 'succeeded'
    assert papers[0]['text_source'] == 'elsevier'
    assert papers[0]['text_path'] == ''


@pytest.mark.network
def test_download_unpaywall_pdf_uses_real_api(tmp_path):
    """
    Test real Unpaywall metadata lookup and PDF download.

    This function performs the following steps:
    1. Loads the user's configured Unpaywall email.
    2. Calls `_download_unpaywall_pdf` for a known open-access DOI.
    3. Reads the downloaded file header.

    Asserts:
        - An Unpaywall email is configured.
        - The Unpaywall API locates and downloads a PDF.
        - The downloaded file starts with a PDF header.
    """
    assert download._unpaywall_email(), (
        'Set unpaywall_email in ~/.config/.pscraperrc.json or UNPAYWALL_EMAIL before running network tests.'
    )
    pdf_path = tmp_path / 'unpaywall.pdf'

    ok, detail = download._download_unpaywall_pdf({'doi': '10.1371/journal.pone.0000308'}, str(pdf_path))

    assert ok, detail
    assert pdf_path.read_bytes().startswith(b'%PDF')


@pytest.mark.network
def test_download_core_pdf_uses_real_api_when_configured(tmp_path):
    """
    Test real CORE search and PDF download.

    This function performs the following steps:
    1. Verifies that a CORE API key is configured.
    2. Uses CORE search to find a small set of candidate works.
    3. Attempts to download the first available PDF candidate through `_download_core_pdf`.

    Asserts:
        - A CORE API key is configured.
        - At least one CORE candidate can be downloaded as a PDF.
        - The downloaded file starts with a PDF header.
    """
    assert download._core_headers().get('Authorization'), (
        'Set core_api_key in ~/.config/.pscraperrc.json or CORE_API_KEY before running network tests.'
    )
    from paperscraper.search import core_search

    candidates = core_search('solid electrolyte', count=5)
    last_error = 'no CORE candidates were returned'
    for _, paper in candidates.iterrows():
        pdf_path = tmp_path / f'{download._safe_filename(paper)}.pdf'
        ok, detail = download._download_core_pdf(paper, str(pdf_path))
        if ok:
            assert pdf_path.read_bytes().startswith(b'%PDF')
            return
        last_error = detail
    pytest.fail(f'No CORE PDF candidate downloaded successfully: {last_error}')


@pytest.mark.network
def test_download_elsevier_pdf_uses_real_api_when_entitled(tmp_path):
    """
    Test real Elsevier PDF download when the configured account is entitled.

    This function performs the following steps:
    1. Verifies that an Elsevier API key is configured.
    2. Attempts to download a known Elsevier article PDF.
    3. Checks the downloaded file when the account has PDF entitlement.

    Asserts:
        - An Elsevier API key is configured.
        - Entitled accounts download a valid PDF.
    """
    assert download.load_settings().get('elsevier_api_key'), (
        'Set elsevier_api_key in ~/.config/.pscraperrc.json or ELSEVIER_API_KEY before running network tests.'
    )
    pdf_path = tmp_path / 'elsevier.pdf'
    paper = {
        'paper_id': 'elsevier-live',
        'doi': '10.1016/j.ssi.2012.10.014',
        'elsevier_link': '',
    }

    ok = download._download_pdf(paper, str(pdf_path))
    if not ok:
        pytest.skip('Elsevier API key is configured, but this account/DOI did not return a PDF entitlement.')
    assert pdf_path.read_bytes().startswith(b'%PDF')
