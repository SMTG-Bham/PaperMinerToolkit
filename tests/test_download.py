"""Unit tests for paperscraper.download.

This module tests download helper behavior without calling live APIs, including
Elsevier text/PDF helpers, open-access PDF source selection, filename creation,
configured source resolution, and corpus status updates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
from typing import Any, Self

import pytest

import paperscraper.corpus as corpus
import paperscraper.download as download


def write_corpus(db_path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write paper rows to a temporary test corpus."""
    with corpus.connect(db_path) as conn:
        for row in rows:
            corpus.upsert_paper(conn, row)


def read_corpus(db_path: str | Path) -> list[dict[str, Any]]:
    """Read paper rows from a temporary test corpus."""
    with corpus.connect(db_path) as conn:
        return corpus.paper_rows(conn)


def test_elsevier_api_key_requires_and_uses_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elsevier API key requires and uses configured API key."""
    monkeypatch.setattr(download, 'load_settings', lambda: {})
    with pytest.raises(ValueError, match='Elsevier API key is not configured'):
        download._elsevier_api_key()

    monkeypatch.setattr(download, 'load_settings', lambda: {'elsevier_api_key': 'elsevier-key'})
    assert download._elsevier_api_key() == 'elsevier-key'


def test_retrieve_document_clears_data_folder_and_writes_successful_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieve document clears data folder and writes successful document."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'old.json').write_text('old')

    class FakeResponse:
        """Provide a response test double."""

        def json(self) -> dict[str, str]:
            """Return the prepared JSON payload."""
            return {'originalText': 'new'}

    def fake_get_content(
        api_key: str,
        uri: str,
        accept: str,
        params: Mapping[str, str],
    ) -> FakeResponse:
        """Provide fake content retrieval for this test."""
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


def test_json_to_text_and_elsevier_string_formatter(tmp_path: Path) -> None:
    """JSON to text and Elsevier string formatter."""
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


def test_full_text_uri_and_download_text_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full text URI and download text success and failure."""
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

    def fake_retrieve_document(_: str) -> None:
        """Provide fake document retrieval for this test."""
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


def test_pdf_urls_and_safe_filename_build_expected_values() -> None:
    """PDF URLs and safe filename build expected values."""
    paper = {'doi': '10.1234/a b', 'elsevier_link': "x 'uri' full-text"}

    assert download._pdf_urls(paper) == ['https://api.elsevier.com/content/article/doi/10.1234%2Fa+b', 'uri']
    assert download._safe_filename({'doi': '10.1234/a b'}) == '10.1234_a_b'
    assert download._safe_filename({'doi': '', 'core_id': 'core/1'}) == 'core_1'
    assert download._safe_filename({'doi': '', 'core_id': '', 'paper_id': 'paper:1'}) == 'paper_1'
    assert download._safe_filename({}) == 'paper'


def test_download_url_to_pdf_saves_only_pdf_responses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Download URL to PDF saves only PDF responses."""
    class FakeResponse:
        """Provide a response test double."""

        def __init__(
            self,
            status_code: int = 200,
            content: bytes = b'%PDF data',
            content_type: str = 'application/pdf',
        ) -> None:
            """Initialize the test double."""
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


def test_download_unpaywall_pdf_handles_missing_config_and_pdf_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download Unpaywall PDF handles missing config and PDF candidates."""
    class FakeResponse:
        """Provide a response test double."""

        status_code = 200

        def json(self) -> dict[str, Any]:
            """Return the prepared JSON payload."""
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

    def fake_download(url: str, filepath: str) -> tuple[bool, str]:
        """Provide a fake download implementation."""
        tried.append(url)
        return (url.endswith('two.pdf'), 'failed')

    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download)

    assert download._download_unpaywall_pdf({'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')) == (
        True,
        'https://example.com/two.pdf',
    )
    assert tried == ['https://example.com/one.pdf', 'https://example.com/two.pdf']


def test_core_headers_and_core_pdf_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CORE headers and CORE PDF download."""
    monkeypatch.setattr(download, 'load_settings', lambda: {})
    monkeypatch.delenv('CORE_API_KEY', raising=False)
    assert download._core_headers() == {'User-Agent': 'PaperScraper/0.0.1'}

    monkeypatch.setenv('CORE_API_KEY', 'core-key')
    assert download._core_headers()['Authorization'] == 'Bearer core-key'

    tried = []

    def fake_download(
        url: str,
        filepath: str,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[bool, str]:
        """Provide a fake download implementation."""
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


def test_download_openalex_pdf_handles_missing_doi_lookup_failure_and_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download OpenAlex PDF handles missing DOI lookup failure and candidates."""
    pdf_path = str(tmp_path / 'paper.pdf')
    assert download._download_openalex_pdf({'doi': ''}, pdf_path) == (False, 'missing DOI')

    monkeypatch.setattr(
        download.openalex,
        'get_work',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('rate limited')),
    )
    assert download._download_openalex_pdf({'doi': '10.1234/example'}, pdf_path) == (False, 'rate limited')

    monkeypatch.setattr(download.openalex, 'get_work', lambda *_, **__: None)
    assert download._download_openalex_pdf({'doi': '10.9999/missing'}, pdf_path) == (
        False,
        'no OpenAlex work found for doi:10.9999/missing',
    )

    monkeypatch.setattr(download.openalex, 'get_work', lambda *_, **__: {'id': 'https://openalex.org/W1'})
    assert download._download_openalex_pdf({'doi': '10.1234/example'}, pdf_path) == (
        False,
        'no OpenAlex PDF URL found',
    )

    identifiers = []

    def fake_get_work(identifier: str, **_: object) -> dict[str, Any]:
        """Provide fake work retrieval for this test."""
        identifiers.append(identifier)
        return {
            'best_oa_location': {'pdf_url': 'https://example.com/one.pdf'},
            'locations': [{'pdf_url': 'https://example.com/two.pdf'}],
            'open_access': {'oa_url': 'https://example.com/landing'},
        }

    tried = []

    def fake_download(url: str, filepath: str) -> tuple[bool, str]:
        """Provide a fake download implementation."""
        tried.append(url)
        return (url.endswith('two.pdf'), 'failed')

    monkeypatch.setattr(download.openalex, 'get_work', fake_get_work)
    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download)

    assert download._download_openalex_pdf({'paper_id': 'openalex:W123'}, pdf_path) == (
        True,
        'https://example.com/two.pdf',
    )
    assert identifiers == ['W123']
    assert tried == ['https://example.com/one.pdf', 'https://example.com/two.pdf']


def test_abstract_helpers_clean_provider_text_and_try_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Abstract helpers clean provider text and try sources."""
    assert download._clean_abstract(' A&nbsp;<b>solid</b>\n electrolyte ') == 'A solid electrolyte'
    assert download._clean_abstract(['First', 'Second']) == 'First Second'
    assert download._abstract_from_mapping({'outer': {'dc:description': '<p>Nested abstract</p>'}}) == 'Nested abstract'

    calls = []
    monkeypatch.setattr(download, '_download_openalex_abstract',
                        lambda paper: calls.append('openalex') or (True, 'openalex', 'OpenAlex abstract'))
    monkeypatch.setattr(download, '_download_core_abstract', lambda paper: calls.append('core') or (True, 'core', 'CORE abstract'))
    monkeypatch.setattr(download, '_download_elsevier_abstract',
                        lambda paper: calls.append('elsevier') or (True, 'elsevier', 'Elsevier abstract'))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)

    assert download._download_abstract({'core_id': '123'}) == (True, 'core', 'CORE abstract')
    assert download._download_abstract({'doi': '10.1234/example'}) == (
        True, 'openalex', 'OpenAlex abstract'
    )

    monkeypatch.setattr(download, '_download_openalex_abstract',
                        lambda paper: calls.append('openalex-miss') or (False, 'missing', ''))
    assert download._download_abstract({'doi': '10.1234/example', 'core_id': '456'}) == (
        True, 'core', 'CORE abstract'
    )

    monkeypatch.setattr(download, '_download_core_abstract',
                        lambda paper: calls.append('core-miss') or (False, 'missing', ''))
    assert download._download_abstract({'doi': '10.1234/example', 'core_id': '456'}) == (
        True, 'elsevier', 'Elsevier abstract'
    )

    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    assert download._download_abstract({'paper_id': 'missing'}) == (False, 'no abstract source available', '')
    assert calls == ['core', 'openalex', 'openalex-miss', 'core',
                     'openalex-miss', 'core-miss', 'elsevier']


def test_download_openalex_abstract_reconstructs_inverted_index(monkeypatch):
    """Fetch and reconstruct OpenAlex abstracts using DOI or work identifiers."""
    assert download._download_openalex_abstract({'paper_id': 'missing'}) == (
        False, 'missing DOI or OpenAlex ID', ''
    )

    monkeypatch.setattr(
        download.openalex,
        'get_work',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('rate limited')),
    )
    assert download._download_openalex_abstract({'doi': '10.1234/example'}) == (
        False, 'rate limited', ''
    )

    calls = []

    def fake_get_work(identifier, api_key=None):
        calls.append((identifier, api_key))
        return {
            'abstract_inverted_index': {
                'conductivity': [2],
                'Ionic': [0],
                'improves': [1],
            },
        }

    monkeypatch.setattr(download.openalex, 'configured_api_key', lambda: 'openalex-key')
    monkeypatch.setattr(download.openalex, 'get_work', fake_get_work)
    assert download._download_openalex_abstract({'paper_id': 'openalex:W123'}) == (
        True, 'openalex', 'Ionic improves conductivity'
    )
    assert calls == [('W123', 'openalex-key')]

    monkeypatch.setattr(download.openalex, 'get_work', lambda *_, **__: None)
    assert download._download_openalex_abstract({'doi': '10.9999/missing'}) == (
        False, 'no OpenAlex work found for doi:10.9999/missing', ''
    )

    monkeypatch.setattr(
        download.openalex,
        'get_work',
        lambda *_, **__: {'id': 'https://openalex.org/W456'},
    )
    assert download._download_openalex_abstract({'doi': '10.1234/no-abstract'}) == (
        False, 'no OpenAlex abstract found', ''
    )


def test_core_and_elsevier_abstract_downloads_parse_provider_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """CORE and Elsevier abstract downloads parse provider payloads."""
    class FakeResponse:
        """Provide a response test double."""

        def __init__(self, payload: dict[str, Any] | None = None, status_code: int = 200) -> None:
            """Initialize the test double."""
            self.payload = payload or {}
            self.status_code = status_code

        def json(self) -> dict[str, Any]:
            """Return the prepared JSON payload."""
            return self.payload

    core_calls = []

    def fake_get(url: str, headers: Mapping[str, str], timeout: float) -> FakeResponse:
        """Provide a fake HTTP GET implementation."""
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


def test_configured_sources_resolves_all_deduplicates_and_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured sources resolve ``all``, deduplicate, and reject invalid values."""
    monkeypatch.setattr(
        download,
        'load_settings',
        lambda: {
            'unpaywall_email': 'person@example.com',
            'core_api_key': 'core-key',
            'elsevier_api_key': 'elsevier-key',
        },
    )

    assert download._configured_sources(['all']) == ['unpaywall', 'openalex', 'core', 'elsevier']
    assert download._configured_sources(['core', 'core', 'unpaywall']) == ['core', 'unpaywall']
    with pytest.raises(ValueError, match='download source must be one of'):
        download._configured_sources(['bad'])

    monkeypatch.setattr(download, 'load_settings', lambda: {})
    monkeypatch.delenv('UNPAYWALL_EMAIL', raising=False)
    monkeypatch.delenv('CORE_API_KEY', raising=False)
    assert download._configured_sources(['all']) == ['openalex']


def test_download_pdf_from_sources_handles_existing_success_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download PDF from sources handles existing success and failures."""
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


def test_should_try_elsevier_text_detects_full_text_links() -> None:
    """Elsevier text eligibility detects full-text links and DOI fallbacks."""
    assert download._should_try_elsevier_text({'elsevier_link': "has 'full-text-link' full-text"}) is True
    assert download._should_try_elsevier_text({
        'elsevier_link': 'https://api.elsevier.com/content/article/eid/1-s2.0-S1005030226004123',
    }) is True
    assert download._should_try_elsevier_text({'doi': '10.1234/example', 'elsevier_link': 'abstract only'}) is True
    assert download._should_try_elsevier_text({'elsevier_link': 'abstract only'}) is False
    assert download._should_try_elsevier_text({'elsevier_link': None}) is False


def test_download_papers_validates_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Download papers validates configuration."""
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


def test_download_papers_updates_text_and_pdf_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Download papers updates text and PDF statuses."""
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

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{
        'paper_id': 'paper:1',
        'doi': '10.1234/example',
        'elsevier_link': 'has full-text link',
    }])
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['unpaywall'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)

    def fake_download_text(paper: Mapping[str, Any], filepath: str) -> bool:
        """Provide fake text download behavior for this test."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('text')
        return True

    def fake_download_pdf_from_sources(
        paper: Mapping[str, Any],
        filepath: str,
        sources: list[str],
    ) -> tuple[bool, str, str]:
        """Provide fake PDF download behavior for this test."""
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


def test_download_papers_persists_openalex_pdf_source_and_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download papers persists OpenAlex PDF source and URL."""
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

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'doi:10.1234/example', 'doi': '10.1234/example'}])
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['openalex'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)

    def fake_download_pdf_from_sources(
        paper: Mapping[str, Any],
        filepath: str,
        sources: list[str],
    ) -> tuple[bool, str, str]:
        """Provide fake PDF download behavior for this test."""
        assert sources == ['openalex']
        with open(filepath, 'wb') as f:
            f.write(b'%PDF text')
        return True, 'openalex', 'https://example.org/openalex.pdf'

    monkeypatch.setattr(download, '_download_pdf_from_sources', fake_download_pdf_from_sources)

    download.download_papers(str(db_path), download_format='pdf', download_abstract=False)

    papers = read_corpus(db_path)
    with corpus.connect(db_path) as conn:
        pdf_asset = corpus.get_asset(conn, 'doi:10.1234/example', 'pdf')
    assert papers[0]['pdf_download_status'] == 'succeeded'
    assert papers[0]['pdf_source'] == 'openalex'
    assert papers[0]['pdf_url'] == 'https://example.org/openalex.pdf'
    assert pdf_asset['content'].startswith(b'%PDF')


def test_download_papers_downloads_abstract_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Download papers downloads abstract by default."""
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


def test_download_papers_supports_abstract_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Abstract-only downloads must not attempt text or PDF retrieval."""
    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'paper:abstract-only'}])
    monkeypatch.setattr(download, '_configured_sources', lambda _: [])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    monkeypatch.setattr(
        download,
        '_download_abstract',
        lambda paper: (True, 'openalex', 'Only the abstract should be downloaded.'),
    )
    monkeypatch.setattr(
        download,
        '_download_text',
        lambda *args, **kwargs: pytest.fail('text download was attempted'),
    )
    monkeypatch.setattr(
        download,
        '_download_pdf_from_sources',
        lambda *args, **kwargs: pytest.fail('PDF download was attempted'),
    )

    download.download_papers(str(db_path), download_format='abstract')

    with corpus.connect(db_path) as conn:
        abstract_asset = corpus.get_asset(conn, 'paper:abstract-only', 'abstract')
    assert abstract_asset['content'] == b'Only the abstract should be downloaded.'
    output = capsys.readouterr().out
    assert 'Download complete: 0 text files, 0 PDFs, 1 abstracts downloaded.' in output


def test_download_papers_skips_abstract_download_when_asset_already_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Download papers skips abstract download when asset already exists."""
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
    assert 'Skipped existing corpus assets: 0 text files, 0 PDFs, 1 abstracts.' in output


def test_download_papers_skips_every_requested_existing_content_type(tmp_path, monkeypatch, capsys):
    """Do not call providers for abstract, text, or PDF assets already in the corpus."""

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
    paper = {'paper_id': 'paper:complete', 'doi': '10.1234/complete'}
    with corpus.connect(db_path) as conn:
        corpus.add_asset(conn, paper, 'stored abstract', role='abstract', kind='text',
                         mime_type='text/plain', source='search')
        corpus.add_asset(conn, paper, 'stored text', role='text', kind='text',
                         mime_type='text/plain', source='elsevier')
        corpus.add_asset(conn, paper, b'%PDF stored', role='pdf', kind='pdf',
                         mime_type='application/pdf', source='openalex')

    monkeypatch.setattr(download, '_configured_sources', lambda _: [])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(
        download,
        '_download_abstract',
        lambda *_: (_ for _ in ()).throw(AssertionError('abstract provider called')),
    )
    monkeypatch.setattr(
        download,
        '_download_text',
        lambda *_: (_ for _ in ()).throw(AssertionError('text provider called')),
    )
    monkeypatch.setattr(
        download,
        '_download_pdf_from_sources',
        lambda *_: (_ for _ in ()).throw(AssertionError('PDF provider called')),
    )

    download.download_papers(str(db_path), download_format='both')

    paper = read_corpus(db_path)[0]
    output = capsys.readouterr().out
    assert paper['abstract_download_status'] == 'succeeded'
    assert paper['text_download_status'] == 'succeeded'
    assert paper['pdf_download_status'] == 'succeeded'
    assert paper['abstract_source'] == 'search'
    assert paper['text_source'] == 'elsevier'
    assert paper['pdf_source'] == 'openalex'
    assert 'Skipped existing corpus assets: 1 text files, 1 PDFs, 1 abstracts.' in output


def test_download_papers_existing_text_needs_no_elsevier_key(tmp_path, monkeypatch):
    """Allow a text-only rerun to skip stored text without provider configuration."""
    db_path = tmp_path / 'papers.db'
    paper = {'paper_id': 'paper:text', 'doi': '10.1234/text'}
    with corpus.connect(db_path) as conn:
        corpus.add_asset(conn, paper, 'stored text', role='text', kind='text',
                         mime_type='text/plain', source='elsevier')

    monkeypatch.setattr(download, '_configured_sources', lambda _: [])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    download.download_papers(
        str(db_path), download_format='text', download_abstract=False
    )

    assert read_corpus(db_path)[0]['text_download_status'] == 'succeeded'


def test_download_papers_force_redownloads_existing_content(tmp_path, monkeypatch, capsys):
    """The force option refreshes every requested content role despite stored assets."""

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
    paper = {'paper_id': 'paper:refresh', 'doi': '10.1234/refresh'}
    with corpus.connect(db_path) as conn:
        corpus.add_asset(conn, paper, 'old abstract', role='abstract', kind='text',
                         mime_type='text/plain', source='old')
        corpus.add_asset(conn, paper, 'old text', role='text', kind='text',
                         mime_type='text/plain', source='old')
        corpus.add_asset(conn, paper, b'%PDF old', role='pdf', kind='pdf',
                         mime_type='application/pdf', source='old')

    calls = []

    def fake_text(_paper, filepath):
        calls.append('text')
        with open(filepath, 'w', encoding='utf-8') as out_file:
            out_file.write('new text')
        return True

    def fake_pdf(_paper, filepath, sources):
        calls.append(('pdf', sources))
        with open(filepath, 'wb') as out_file:
            out_file.write(b'%PDF new')
        return True, 'openalex', 'https://example.org/new.pdf'

    monkeypatch.setattr(download, '_configured_sources', lambda _: ['openalex'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(
        download,
        '_download_abstract',
        lambda *_: calls.append('abstract') or (True, 'openalex', 'new abstract'),
    )
    monkeypatch.setattr(download, '_download_text', fake_text)
    monkeypatch.setattr(download, '_download_pdf_from_sources', fake_pdf)

    download.download_papers(str(db_path), download_format='both', force=True)

    with corpus.connect(db_path) as conn:
        abstract_asset = corpus.get_asset(conn, 'paper:refresh', 'abstract')
        text_asset = corpus.get_asset(conn, 'paper:refresh', 'text')
        pdf_asset = corpus.get_asset(conn, 'paper:refresh', 'pdf')
    output = capsys.readouterr().out
    assert calls == ['abstract', 'text', ('pdf', ['openalex'])]
    assert abstract_asset['content'] == b'new abstract'
    assert text_asset['content'] == b'new text'
    assert pdf_asset['content'] == b'%PDF new'
    assert 'Download complete: 1 text files, 1 PDFs, 1 abstracts downloaded.' in output
    assert 'Skipped existing corpus assets' not in output


def test_retrieve_document_reports_failed_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Retrieve document reports failed read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(download, '_elsevier_api_key', lambda: 'elsevier-key')
    monkeypatch.setattr(
        download.elsevier,
        'get_content',
        lambda *_, **__: (_ for _ in ()).throw(download.requests.RequestException('read failed')),
    )

    download.retrieve_document('full-text-uri')

    assert 'Read document failed.' in capsys.readouterr().out


def test_full_text_uri_and_download_text_handle_malformed_or_empty_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full text URI and download text handle malformed or empty retrieval."""
    monkeypatch.chdir(tmp_path)
    assert download._full_text_uri({'elsevier_link': 'full-text without quoted uri'}) is None

    def fake_retrieve_document(_: str) -> None:
        """Provide fake document retrieval for this test."""
        os.makedirs('data', exist_ok=True)

    monkeypatch.setattr(download, 'retrieve_document', fake_retrieve_document)

    assert download._download_text({'elsevier_link': "x 'uri' full-text"}, str(tmp_path / 'paper.txt')) is False


def test_download_pdf_requires_key_and_handles_success_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Download PDF requires key and handles success and failures."""
    class FakeResponse:
        """Provide a response test double."""

        def __init__(
            self,
            status_code: int = 200,
            content: bytes = b'%PDF data',
            content_type: str = 'application/pdf',
        ) -> None:
            """Initialize the test double."""
            self.status_code = status_code
            self.content = content
            self.headers = {'Content-Type': content_type}

    out_path = tmp_path / 'paper.pdf'
    monkeypatch.setattr(download, 'load_settings', lambda: {})
    with pytest.raises(ValueError, match='Elsevier API key is not configured'):
        download._download_pdf({'doi': '10.1234/example'}, str(out_path))

    monkeypatch.setattr(download, 'load_settings', lambda: {'elsevier_api_key': 'elsevier-key'})
    monkeypatch.setattr(download, '_pdf_urls', lambda _: ['bad-status', 'bad-request', 'bad-content', 'good-pdf'])

    def fake_get_content(
        api_key: str,
        url: str,
        accept: str,
        params: Mapping[str, str],
    ) -> FakeResponse:
        """Provide fake content retrieval for this test."""
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


def test_download_unpaywall_pdf_handles_metadata_errors_and_missing_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download Unpaywall PDF handles metadata errors and missing candidates."""
    class ErrorResponse:
        """Provide a response test double that raises an error."""

        status_code = 500

    class EmptyResponse:
        """Provide a response test double with no metadata."""

        status_code = 200

        def json(self) -> dict[str, Any]:
            """Return the prepared JSON payload."""
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


def test_download_core_pdf_returns_last_error_when_candidates_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download CORE PDF returns last error when candidates fail."""
    monkeypatch.setattr(download, '_download_url_to_pdf', lambda *_, **__: (False, 'candidate failed'))

    assert download._download_core_pdf(
        {'pdf_url': 'https://example.com/pdf', 'core_id': '123'},
        str(tmp_path / 'paper.pdf'),
    ) == (False, 'candidate failed')


def test_elsevier_configured_reads_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elsevier configured reads settings."""
    monkeypatch.setattr(download, 'load_settings', lambda: {})
    assert download._elsevier_configured() is False

    monkeypatch.setattr(download, 'load_settings', lambda: {'elsevier_api_key': 'elsevier-key'})
    assert download._elsevier_configured() is True


def test_download_pdf_from_sources_collects_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download PDF from sources collects exceptions."""
    monkeypatch.setattr(download, '_download_unpaywall_pdf', lambda *_: (_ for _ in ()).throw(RuntimeError('boom')))

    ok, error, detail = download._download_pdf_from_sources({}, str(tmp_path / 'paper.pdf'), ['unpaywall'])

    assert ok is False
    assert error == 'unpaywall: boom'
    assert detail == ''


def test_download_papers_records_text_and_pdf_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download papers records text and PDF failures."""
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


def test_download_papers_records_initial_text_download_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download papers records initial text download exception."""
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


def test_download_papers_records_download_exceptions_and_elsevier_text_after_oa_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download papers records download exceptions and Elsevier text after OA PDF."""
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

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [
        {'paper_id': 'paper:pdf-error', 'doi': '10.1234/pdf-error'},
        {'paper_id': 'paper:oa-text', 'doi': '10.1234/oa-text', 'elsevier_link': 'has full-text link'},
    ])
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['core'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)

    def fake_download_pdf_from_sources(
        paper: Mapping[str, Any],
        filepath: str,
        sources: list[str],
    ) -> tuple[bool, str, str]:
        """Provide fake PDF download behavior for this test."""
        if paper['paper_id'] == 'paper:pdf-error':
            raise RuntimeError('pdf exploded')
        with open(filepath, 'wb') as f:
            f.write(b'%PDF data')
        return True, 'core', 'https://core/pdf'

    def fake_download_text(paper: Mapping[str, Any], filepath: str) -> bool:
        """Provide fake text download behavior for this test."""
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


def test_download_papers_downloads_elsevier_text_after_oa_pdf_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download papers downloads Elsevier text after OA PDF success."""
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

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{
        'paper_id': 'paper:oa-text',
        'doi': '10.1234/oa-text',
        'elsevier_link': 'has full-text link',
    }])
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['core'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)

    def fake_download_pdf_from_sources(
        paper: Mapping[str, Any],
        filepath: str,
        sources: list[str],
    ) -> tuple[bool, str, str]:
        """Provide fake PDF download behavior for this test."""
        with open(filepath, 'wb') as f:
            f.write(b'%PDF data')
        return True, 'core', 'https://core/pdf'

    monkeypatch.setattr(download, '_download_pdf_from_sources', fake_download_pdf_from_sources)

    def fake_download_text(paper: Mapping[str, Any], filepath: str) -> bool:
        """Provide fake text download behavior for this test."""
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
def test_download_unpaywall_pdf_uses_real_api(tmp_path: Path) -> None:
    """Download Unpaywall PDF uses real API."""
    assert download._unpaywall_email(), (
        'Set unpaywall_email in ~/.config/.pscraperrc.json or UNPAYWALL_EMAIL before running network tests.'
    )
    pdf_path = tmp_path / 'unpaywall.pdf'

    ok, detail = download._download_unpaywall_pdf({'doi': '10.1371/journal.pone.0000308'}, str(pdf_path))

    assert ok, detail
    assert pdf_path.read_bytes().startswith(b'%PDF')


@pytest.mark.network
def test_download_openalex_pdf_uses_real_api(tmp_path: Path) -> None:
    """Download OpenAlex PDF uses real API."""
    from paperscraper import openalex

    payload = openalex.request_json(openalex.WORKS_URL,
                                    params={
                                        'search': 'solid electrolyte',
                                        'filter': 'open_access.is_oa:true',
                                        'per-page': 5,
                                    },
                                    api_key=openalex.configured_api_key())
    candidates = [openalex.work_to_paper(work) for work in payload.get('results') or []]
    last_error = 'no OpenAlex candidates were returned'
    for index, paper in enumerate(candidates):
        pdf_path = tmp_path / f'openalex-{index}.pdf'
        ok, detail = download._download_openalex_pdf(paper, str(pdf_path))
        if ok:
            assert pdf_path.read_bytes().startswith(b'%PDF')
            return
        last_error = detail
    pytest.fail(f'No OpenAlex PDF candidate downloaded successfully: {last_error}')


@pytest.mark.network
def test_download_core_pdf_uses_real_api_when_configured(tmp_path: Path) -> None:
    """Download CORE PDF uses real API when configured."""
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
def test_download_elsevier_pdf_uses_real_api_when_entitled(tmp_path: Path) -> None:
    """Download Elsevier PDF uses real API when entitled."""
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
