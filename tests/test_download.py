"""Unit tests for paperminertoolkit.workflows.download.

This module tests download helper behavior without calling live APIs, including
Elsevier text/PDF helpers, open-access PDF source selection, filename creation,
configured source resolution, and corpus status updates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, Self

import pytest
import requests

import paperminertoolkit.corpus.database as corpus
import paperminertoolkit.workflows.download as download


def write_corpus(db_path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write paper rows to a temporary test corpus."""
    with corpus.connect(db_path) as conn:
        for row in rows:
            corpus.upsert_paper(conn, row)


def read_corpus(db_path: str | Path) -> list[dict[str, Any]]:
    """Read paper rows from a temporary test corpus."""
    with corpus.connect(db_path) as conn:
        return corpus.paper_rows(conn)


def jats_download(text: str = 'Title\n\nBody text.') -> download.provider.FullTextDocument:
    """Return one structured full-text result for download workflow tests."""
    return download.provider.FullTextDocument(
        text=text,
        content='<article><body><p>Body text.</p></body></article>',
        document_format='jats',
        source_url='https://example.test/article.xml',
        source_identifier='article-1',
    )


def test_download_parsers_cover_unusual_link_and_abstract_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize dict links, malformed serialized links, nested abstracts, and direct URLs."""
    assert download._link_values({'@href': 'url'}) == [{'@href': 'url'}]
    assert download._link_values('{broken') == ['{broken']
    assert download._full_text_uri_from_link_value(1) is None
    endpoint = 'https://api.elsevier.com/content/article/doi/10.1/x'
    assert download._full_text_uri_from_link_value(endpoint) == endpoint
    assert download._full_text_uri_from_link_value('https://example/full-text') == (
        'https://example/full-text'
    )
    assert download._abstract_from_mapping([{}, {'nested': {'description': 'Abstract'}}]) == 'Abstract'
    urls = download._elsevier_abstract_urls({
        'paper_id': 'SCOPUS_ID:123',
        'elsevier_link': {'@ref': 'abstract', '@href': 'https://example/abstract'},
    })
    assert urls == [
        'https://example/abstract',
        'https://api.elsevier.com/content/abstract/scopus_id/123',
    ]
    with pytest.raises(KeyError, match='Unknown pipeline'):
        download._set_status({}, 'unknown', 'failed')
    monkeypatch.setattr(download.unpaywall, 'configured_email', lambda settings=None: '')
    assert download._unpaywall_email({}) is None


def test_pubmed_download_helpers_report_resolution_and_service_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return actionable errors for PMC and PubMed lookup failures."""
    monkeypatch.setattr(download, '_pubmed_credentials', lambda: (None, ''))
    monkeypatch.setattr(
        download.pubmed, 'resolve_pmcid',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('resolve failed')),
    )
    assert download._download_pubmed_pdf({}, tmp_path / 'paper.pdf') == (False, 'resolve failed')
    assert download._download_pmc_text({}, tmp_path / 'paper.txt') == (
        False, 'resolve failed', None)

    monkeypatch.setattr(download.pubmed, 'resolve_pmcid', lambda *args, **kwargs: 'PMC1')
    monkeypatch.setattr(
        download.pubmed, 'oa_package_urls',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('oa failed')),
    )
    assert download._download_pubmed_pdf({}, tmp_path / 'paper.pdf') == (False, 'oa failed')
    monkeypatch.setattr(
        download.pubmed, 'pmc_full_text_document',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('text failed')),
    )
    assert download._download_pmc_text({}, tmp_path / 'paper.txt') == (
        False, 'text failed', None)

    monkeypatch.setattr(
        download.pubmed, 'resolve_pmid',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('abstract failed')),
    )
    assert download._download_pubmed_abstract({}) == (False, 'abstract failed', '')


@pytest.mark.parametrize(
    ('record_name', 'url_name', 'expected'),
    [
        ('_medrxiv_record', 'medrxiv', 'missing medRxiv DOI'),
        ('_biorxiv_record', 'biorxiv', 'missing bioRxiv DOI'),
    ],
)
def test_preprint_pdf_helpers_report_missing_provider_urls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_name: str,
    url_name: str,
    expected: str,
) -> None:
    """Report a missing DOI when a preprint record cannot produce a PDF URL."""
    monkeypatch.setattr(download, record_name, lambda paper: ({}, ''))
    monkeypatch.setattr(getattr(download, url_name), 'pdf_url', lambda *args: '')
    function = getattr(download, f'_download_{url_name}_pdf')
    assert function({}, tmp_path / 'paper.pdf') == (False, expected)


@pytest.mark.parametrize(
    ('record_name', 'module_name', 'function_name', 'message'),
    [
        ('_medrxiv_record', 'medrxiv', '_download_medrxiv_text', 'med text failed'),
        ('_biorxiv_record', 'biorxiv', '_download_biorxiv_text', 'bio text failed'),
    ],
)
def test_preprint_text_helpers_report_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_name: str,
    module_name: str,
    function_name: str,
    message: str,
) -> None:
    """Preserve provider errors raised while downloading preprint full text."""
    monkeypatch.setattr(download, record_name, lambda paper: ({'doi': 'x'}, ''))

    def fail(entry: object) -> download.provider.FullTextDocument:
        """Raise the configured provider failure."""
        raise RuntimeError(message)

    monkeypatch.setattr(getattr(download, module_name), 'full_text_document', fail)
    assert getattr(download, function_name)({}, tmp_path / 'paper.txt') == (
        False, message, None)


def test_chemrxiv_and_arxiv_helpers_report_empty_or_failed_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail clearly when preprint identifiers or abstract responses are unusable."""
    monkeypatch.setattr(download, '_chemrxiv_record', lambda paper: ({}, ''))
    monkeypatch.setattr(download.chemrxiv, 'pdf_url', lambda value: '')
    assert download._download_chemrxiv_pdf({}, tmp_path / 'paper.pdf') == (
        False, 'missing chemRxiv DOI'
    )
    monkeypatch.setattr(download, '_arxiv_identifier', lambda paper: '1234.5')
    monkeypatch.setattr(
        download.arxiv, 'fetch_ids',
        lambda values: (_ for _ in ()).throw(RuntimeError('arxiv failed')),
    )
    assert download._download_arxiv_abstract({}) == (False, 'arxiv failed', '')
    monkeypatch.setattr(download.arxiv, 'fetch_ids', lambda values: '<feed/>')
    monkeypatch.setattr(download.arxiv, 'parse_entries', lambda value: [{'abstract': ''}])
    assert download._download_arxiv_abstract({}) == (
        False, 'no arXiv abstract found for 1234.5', ''
    )


def test_elsevier_abstract_errors_and_invalid_asset_role(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Handle absent endpoints, HTTP failures, connection failures, and invalid roles."""
    assert download._download_elsevier_abstract({}) == (
        False, 'missing Elsevier abstract URL', ''
    )
    monkeypatch.setattr(download.elsevier, 'configured_api_key', lambda: 'key')
    errors = [
        requests.HTTPError(response=SimpleNamespace(status_code=404)),
        requests.ConnectionError('offline'),
    ]

    def fail(*args: Any, **kwargs: Any) -> Any:
        """Raise successive Elsevier request failures."""
        raise errors.pop(0)

    monkeypatch.setattr(download.elsevier, 'get_content', fail)
    paper = {'elsevier_link': [
        'https://api.elsevier.com/content/abstract/doi/one',
        'https://api.elsevier.com/content/abstract/doi/two',
    ]}
    assert download._download_elsevier_abstract(paper) == (False, 'offline', '')
    with pytest.raises(ValueError, match='role must be'):
        download._store_downloaded_asset(None, {}, tmp_path / 'x', 'image', 'source')



def test_json_to_text_and_elsevier_string_formatter() -> None:
    """Read text out of a full-text payload and tidy what comes back."""
    assert download.elsevier.full_text({'originalText': 'paper text'}) == 'paper text'
    assert download.elsevier.full_text(
        {'full-text-retrieval-response': {'originalText': 'nested text'}}) == 'nested text'
    # Structured XML rather than text is not usable, and is reported as absent.
    assert download.elsevier.full_text({'originalText': {'bad': 'text'}}) == ''
    assert download.elsevier.full_text({}) == ''
    assert download.elsevier.full_text(None) == ''
    assert download._elsevier_string_formatter('A Acknowledgements clean Acknowledgements') == ' clean '
    assert download._elsevier_string_formatter('A References clean References') == ' clean '
    assert download._elsevier_string_formatter('prefix amazonaws.com/key paper text') == ' paper text'


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

    monkeypatch.setattr(download.elsevier, 'configured_api_key', lambda *_: 'elsevier-key')
    document = download.provider.FullTextDocument(
        'downloaded text',
        '<article><body><p>downloaded text</p></body></article>',
        'elsevier-xml',
    )
    monkeypatch.setattr(download.elsevier, 'full_text_document',
                        lambda *_, **__: document)
    out_path = tmp_path / 'paper.txt'
    assert download._download_text(paper, str(out_path)) is True
    assert out_path.read_text() == 'downloaded text'

    # Nothing is written to the working directory any more.
    assert not (tmp_path / 'data').exists()

    monkeypatch.setattr(download.elsevier, 'full_text_document',
                        lambda *_, **__: download.provider.FullTextDocument(''))
    assert download._download_text(paper, str(out_path)) is False
    assert download._download_text({'elsevier_link': ''}, str(out_path)) is False


def test_download_elsevier_text_returns_native_document_and_clear_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return the XML-bearing provider result from the registered text handler."""
    filepath = tmp_path / 'paper.txt'
    assert download._download_elsevier_text({}, filepath) == (
        False, 'missing Elsevier full-text URL', None)
    monkeypatch.setattr(download.elsevier, 'configured_api_key', lambda: 'key')
    monkeypatch.setattr(download.elsevier, 'full_text_document',
                        lambda *_, **__: download.provider.FullTextDocument(''))
    paper = {'doi': '10.1/x'}
    assert download._download_elsevier_text(paper, filepath) == (
        False, 'Elsevier text download failed', None)

    document = download.provider.FullTextDocument(
        'Derived text.', '<article/>', 'elsevier-xml', source_identifier='10.1/x',
    )
    monkeypatch.setattr(download.elsevier, 'full_text_document',
                        lambda *_, **__: document)
    assert download._download_elsevier_text(paper, filepath) == (True, '', document)
    assert filepath.read_text(encoding='utf-8') == 'Derived text.'


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
    record = {
        'best_oa_location': {'url_for_pdf': 'https://example.com/one.pdf'},
        'oa_locations': [
            {'url_for_pdf': 'https://example.com/one.pdf'},
            {'url_for_pdf': 'https://example.com/two.pdf'},
        ],
    }

    monkeypatch.setattr(download.unpaywall, 'configured_email', lambda *_: '')
    assert download._download_unpaywall_pdf({'doi': ''}, str(tmp_path / 'paper.pdf')) == (False, 'missing DOI')
    assert 'Unpaywall email is not configured' in download._download_unpaywall_pdf(
        {'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')
    )[1]

    tried = []
    monkeypatch.setattr(download.unpaywall, 'configured_email', lambda *_: 'person@example.com')
    monkeypatch.setattr(download.unpaywall, 'get_work', lambda *_, **__: record)

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
    monkeypatch.setattr(download.core, 'configured_api_key', lambda *_: 'core-key')

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
        f'{download.core.WORKS_URL}/abc%20123/download',
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


def test_download_openalex_abstract_reconstructs_inverted_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def fake_get_work(identifier: str, api_key: str | None = None) -> dict[str, Any]:
        """Return deterministic OpenAlex metadata for the requested work."""
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


def test_download_openalex_tei_reports_failures_and_writes_derived_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use GROBID only with a key and preserve the provider result on success."""
    filepath = tmp_path / 'paper.txt'
    assert download._download_openalex_tei_text({}, filepath) == (
        False, 'missing DOI or OpenAlex ID', None)
    monkeypatch.setattr(download.openalex, 'configured_api_key', lambda: None)
    assert download._download_openalex_tei_text({'doi': '10.1/x'}, filepath) == (
        False, 'OpenAlex GROBID downloads require an API key', None)

    monkeypatch.setattr(download.openalex, 'configured_api_key', lambda: 'key')
    monkeypatch.setattr(
        download.openalex,
        'get_work',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('budget exhausted')),
    )
    assert download._download_openalex_tei_text({'doi': '10.1/x'}, filepath) == (
        False, 'budget exhausted', None)
    monkeypatch.setattr(download.openalex, 'get_work', lambda *_, **__: None)
    assert download._download_openalex_tei_text({'doi': '10.1/x'}, filepath) == (
        False, 'no OpenAlex work found for doi:10.1/x', None)

    monkeypatch.setattr(download.openalex, 'get_work', lambda *_, **__: {'id': 'W1'})
    monkeypatch.setattr(download.openalex, 'full_text_document',
                        lambda *_, **__: download.provider.FullTextDocument(''))
    assert download._download_openalex_tei_text({'doi': '10.1/x'}, filepath) == (
        False, 'no OpenAlex GROBID XML found for doi:10.1/x', None)

    document = download.provider.FullTextDocument(
        'Derived GROBID text.',
        '<TEI/>',
        'tei',
        'https://content.openalex.org/works/W1.grobid-xml',
        'W1',
        metadata={'publisher_native': False, 'estimated_cost_usd': 0.01},
    )
    monkeypatch.setattr(download.openalex, 'full_text_document',
                        lambda *_, **__: document)
    assert download._download_openalex_tei_text({'doi': '10.1/x'}, filepath) == (
        True, '', document)
    assert filepath.read_text(encoding='utf-8') == 'Derived GROBID text.'


def test_openalex_tei_respects_existing_text_and_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Skip paid fallback for existing text unless the user explicitly forces it."""
    db_path = tmp_path / 'papers.db'
    paper = {'paper_id': 'doi:10.1/x', 'doi': '10.1/x'}
    with corpus.connect(db_path) as conn:
        corpus.add_asset(conn, paper, 'existing text', role='text', kind='text',
                         mime_type='text/plain', source='pdf')
    calls = []

    def fake_openalex(
        row: Mapping[str, Any],
        filepath: str | os.PathLike[str],
    ) -> tuple[bool, str, download.provider.FullTextDocument]:
        """Record the paid fallback and return PDF-derived TEI."""
        calls.append(row['paper_id'])
        Path(filepath).write_text('new text', encoding='utf-8')
        document = download.provider.FullTextDocument(
            'new text', '<TEI/>', 'tei', source_identifier='W1',
            metadata={'publisher_native': False, 'estimated_cost_usd': 0.01},
        )
        return True, '', document

    monkeypatch.setattr(download, '_download_openalex_tei_text', fake_openalex)
    with corpus.connect(db_path) as conn:
        row = corpus.paper_rows(conn)[0]
        download._download_paper(conn, row, tmp_path, 'text', ['openalex'],
                                 download_abstract=False)
        assert corpus.get_structured_documents(conn, row['paper_id']) == []
        download._download_paper(conn, row, tmp_path, 'text', ['openalex'],
                                 download_abstract=False, force=True)
        documents = corpus.get_structured_documents(conn, row['paper_id'])

    assert calls == ['doi:10.1/x']
    assert documents[0]['metadata']['publisher_native'] is False
    assert documents[0]['metadata']['estimated_cost_usd'] == 0.01


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

    def fake_get_work(core_id: object, **_: object) -> dict[str, Any]:
        """Record the work asked for and answer with an abstract.

        Parameters
        ----------
        core_id : object
            CORE identifier requested.
        **_ : object
            Credential and session arguments, unused.

        Returns
        -------
        dict[str, Any]
            A work record carrying an abstract.
        """
        core_calls.append(core_id)
        return {'abstract': '<p>CORE abstract</p>'}

    monkeypatch.setattr(download.core, 'get_work', fake_get_work)

    assert download._download_core_abstract({'core_id': 'abc 123'}) == (True, 'core', 'CORE abstract')
    assert core_calls == ['abc 123']
    assert download._download_core_abstract({'core_id': ''}) == (False, 'missing CORE ID', '')

    monkeypatch.setattr(download.core, 'get_work', lambda *_, **__: None)
    assert download._download_core_abstract({'core_id': '123'}) == (
        False, 'no CORE abstract found', '')

    def refuse(*_: object, **__: object) -> dict[str, Any]:
        """Fail the way a rejected CORE request does.

        Parameters
        ----------
        *_ : object
            Positional arguments, unused.
        **__ : object
            Keyword arguments, unused.

        Returns
        -------
        dict[str, Any]
            Never returned.

        Raises
        ------
        RuntimeError
            Always.
        """
        raise RuntimeError('CORE rejected the request with 404')

    monkeypatch.setattr(download.core, 'get_work', refuse)
    assert download._download_core_abstract({'core_id': '123'}) == (
        False, 'CORE rejected the request with 404', '')

    monkeypatch.setattr(download.elsevier, 'configured_api_key', lambda *_: 'elsevier-key')
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

    assert download._configured_sources(['all']) == ['unpaywall', 'openalex', 'core', 'elsevier',
                                                     'pubmed', 'medrxiv', 'biorxiv', 'chemrxiv',
                                                     'arxiv']
    assert download._configured_sources(['core', 'core', 'unpaywall']) == ['core', 'unpaywall']
    with pytest.raises(ValueError, match='download source must be one of'):
        download._configured_sources(['bad'])

    monkeypatch.setattr(download, 'load_settings', lambda: {})
    monkeypatch.delenv('UNPAYWALL_EMAIL', raising=False)
    monkeypatch.delenv('CORE_API_KEY', raising=False)
    assert download._configured_sources(['all']) == ['openalex', 'pubmed', 'medrxiv', 'biorxiv',
                                                     'chemrxiv', 'arxiv']


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


def test_should_try_medrxiv_text_needs_the_doi_medrxiv_issued() -> None:
    """Offer medRxiv text only for a row medRxiv can actually be asked about."""
    assert download._should_try_medrxiv_text(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596'}) is True
    assert download._should_try_medrxiv_text({'doi': '10.1038/s41467-021-21444-5'}) is False
    assert download._should_try_medrxiv_text({}) is False


def test_download_papers_accepts_medrxiv_as_a_full_text_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat medRxiv as satisfying the text-format precondition, like PubMed."""
    def fake_medrxiv_text(paper: Mapping[str, Any], filepath: object) -> tuple[bool, str]:
        """Write the text a successful medRxiv fetch would have written."""
        with open(filepath, 'w', encoding='utf-8') as out_file:
            out_file.write('Full text.')
        return True, ''

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'doi:10.1101/2024.03.01.24303596',
                            'medrxiv_doi': '10.1101/2024.03.01.24303596'}])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    monkeypatch.setattr(download, '_download_medrxiv_text', fake_medrxiv_text)
    monkeypatch.setattr(download, '_download_abstract', lambda *_, **__: (False, 'no abstract', ''))

    download.download_papers(str(db_path), download_format='text', sources=['medrxiv'])

    row = read_corpus(db_path)[0]
    assert row['text_download_status'] == 'succeeded'
    assert row['text_source'] == 'medrxiv'

    # Without a full-text provider the precondition still refuses the run, and
    # now names the preprint servers among the ways to satisfy it.
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['arxiv'])
    with pytest.raises(ValueError, match='--source medrxiv or --source biorxiv'):
        download.download_papers(str(db_path), download_format='text', force=True)


def test_download_paper_attempts_medrxiv_text_for_a_medrxiv_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach the text sources for a row only medRxiv can supply text for."""
    attempts = []
    monkeypatch.setattr(download, '_download_abstract', lambda *_, **__: (False, 'no abstract', ''))

    def fake_text_sources(paper: Mapping[str, Any], filepath: object,
                          sources: object) -> tuple[bool, str, str]:
        """Record the attempt and write the text a real provider would."""
        attempts.append(dict(paper))
        with open(filepath, 'w', encoding='utf-8') as out_file:
            out_file.write('Full text.')
        return True, 'medrxiv', ''

    monkeypatch.setattr(download, '_download_text_from_sources', fake_text_sources)
    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'doi:10.1101/2024.03.01.24303596'}])

    with download.connect(db_path) as conn:
        paper = {'paper_id': 'doi:10.1101/2024.03.01.24303596',
                 'medrxiv_doi': '10.1101/2024.03.01.24303596'}
        summary = download._download_paper(conn, paper, tmp_path, 'text', ['medrxiv'],
                                           download_abstract=False)

    assert len(attempts) == 1
    assert summary['texts'] == 1
    assert paper['text_source'] == 'medrxiv'

    # A row with no medRxiv DOI has no text provider at all, so nothing is tried.
    attempts.clear()
    with download.connect(db_path) as conn:
        download._download_paper(conn, {'paper_id': 'doi:10.1234/x', 'doi': '10.1234/x'},
                                 tmp_path, 'text', ['medrxiv'], download_abstract=False)
    assert attempts == []


def test_download_paper_records_unexpected_abstract_and_text_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep processing a paper when an individual downloader raises unexpectedly."""
    db_path = tmp_path / 'papers.db'
    paper = {
        'paper_id': 'doi:10.1101/2024.03.01.24303596',
        'medrxiv_doi': '10.1101/2024.03.01.24303596',
    }
    write_corpus(db_path, [paper])

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        """Simulate an unexpected provider implementation failure."""
        raise RuntimeError('provider exploded')

    monkeypatch.setattr(download, '_download_abstract', fail)
    monkeypatch.setattr(download, '_download_text_from_sources', fail)
    with download.connect(db_path) as conn:
        download._download_paper(
            conn, paper, tmp_path, 'text', ['medrxiv'], download_abstract=True
        )
    assert paper['abstract_download_status'] == 'failed'
    assert paper['text_download_status'] == 'failed'
    assert paper['last_error'] == 'provider exploded'


def test_pubmed_download_helpers_keep_the_last_failure_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a failed OA package URL and a missing PMC identifier precisely."""
    monkeypatch.setattr(download, '_pubmed_credentials', lambda: ('', 'person@example.org'))
    monkeypatch.setattr(download.pubmed, 'resolve_pmcid', lambda *args, **kwargs: 'PMC1')
    monkeypatch.setattr(download.pubmed, 'oa_package_urls', lambda *args, **kwargs: ['paper.pdf'])
    monkeypatch.setattr(download, '_download_url_to_pdf', lambda *args, **kwargs: (False, 'bad PDF'))
    assert download._download_pubmed_pdf({}, tmp_path / 'paper.pdf') == (False, 'bad PDF')
    monkeypatch.setattr(download.pubmed, 'resolve_pmcid', lambda *args, **kwargs: '')
    assert download._download_pmc_text({}, tmp_path / 'paper.txt') == (
        False, 'missing PMC ID', None)


def test_download_abstract_honours_the_requested_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ask only the providers the caller selected.

    The source selection reached PDF and full-text retrieval but never got as
    far as _download_abstract, so a run scoped to one provider still queried
    all eight in a fixed order.
    """
    asked = []

    def record(name: str) -> object:
        """Return a downloader that records the attempt and fails.

        Parameters
        ----------
        name : str
            Source the downloader stands in for.

        Returns
        -------
        object
            Downloader recording its call.
        """
        def downloader(_: Mapping[str, Any]) -> tuple[bool, str, str]:
            """Record one attempt and report no abstract.

            Parameters
            ----------
            _ : Mapping[str, Any]
                Corpus paper row, unused.

            Returns
            -------
            tuple[bool, str, str]
                Always a failure, so the loop moves on.
            """
            asked.append(name)
            return False, 'nothing here', ''
        return downloader

    for name in download.registry.names(download.registry.ABSTRACT):
        downloader = download.registry.SOURCES[name].abstract_handler.rsplit(':', 1)[1]
        monkeypatch.setattr(download, downloader, record(name))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)

    paper = {'doi': '10.1234/x', 'arxiv_id': '2301.01234', 'core_id': '7',
             'medrxiv_doi': '10.1101/2024.03.01.24303596'}

    download._download_abstract(dict(paper), ['arxiv'])
    assert asked == ['arxiv']

    asked.clear()
    download._download_abstract(dict(paper), ['medrxiv', 'core'])
    assert asked == ['medrxiv', 'core']

    # No selection still means every reachable source, in registry order.
    asked.clear()
    download._download_abstract(dict(paper))
    assert asked == ['openalex', 'pubmed', 'medrxiv', 'arxiv', 'core', 'elsevier']


def test_download_papers_validates_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Download papers validates configuration."""
    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'paper-1'}])

    with pytest.raises(ValueError, match='download_format must be one of'):
        download.download_papers(str(db_path), download_format='bad')

    monkeypatch.setattr(download, '_configured_sources', lambda _: [])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    with pytest.raises(ValueError, match='Text download requires an Elsevier API key'):
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
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['unpaywall', 'elsevier'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)

    def fake_download_text(
        paper: Mapping[str, Any],
        filepath: str,
    ) -> tuple[bool, str, None]:
        """Provide fake text download behavior for this test."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('text')
        return True, '', None

    def fake_download_pdf_from_sources(
        paper: Mapping[str, Any],
        filepath: str,
        sources: list[str],
    ) -> tuple[bool, str, str]:
        """Provide fake PDF download behavior for this test."""
        with open(filepath, 'wb') as f:
            f.write(b'%PDF text')
        return True, 'unpaywall', 'https://oa/pdf'

    monkeypatch.setattr(download, '_download_elsevier_text', fake_download_text)
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
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['elsevier'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(download, '_download_abstract', lambda paper, sources=None: (True, 'core', 'abstract text'))

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
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['elsevier'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    monkeypatch.setattr(
        download,
        '_download_abstract', lambda paper, sources=None: (True, 'openalex', 'Only the abstract should be downloaded.'),
    )
    monkeypatch.setattr(
        download,
        '_download_elsevier_text',
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
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['elsevier'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(
        download,
        '_download_abstract', lambda paper, sources=None: (_ for _ in ()).throw(AssertionError('abstract should not be downloaded twice')),
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


def test_download_papers_skips_every_requested_existing_content_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Do not call providers for abstract, text, or PDF assets already in the corpus."""

    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept and ignore progress-bar options."""
            return None

        def __enter__(self) -> Self:
            """Enter the progress-bar context."""
            return self

        def __exit__(self, *_: Any) -> bool:
            """Exit the progress-bar context without suppressing errors."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
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

    monkeypatch.setattr(download, '_configured_sources', lambda _: ['elsevier'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(
        download,
        '_download_abstract', lambda *_, **__: (_ for _ in ()).throw(AssertionError('abstract provider called')),
    )
    monkeypatch.setattr(
        download,
        '_download_elsevier_text',
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


def test_download_papers_existing_text_needs_no_elsevier_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow a text-only rerun to skip stored text without provider configuration."""
    db_path = tmp_path / 'papers.db'
    paper = {'paper_id': 'paper:text', 'doi': '10.1234/text'}
    with corpus.connect(db_path) as conn:
        corpus.add_asset(conn, paper, 'stored text', role='text', kind='text',
                         mime_type='text/plain', source='elsevier')

    monkeypatch.setattr(download, '_configured_sources', lambda _: ['elsevier'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    download.download_papers(
        str(db_path), download_format='text', download_abstract=False
    )

    assert read_corpus(db_path)[0]['text_download_status'] == 'succeeded'


def test_download_papers_force_redownloads_existing_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The force option refreshes every requested content role despite stored assets."""

    class FakeTqdm:
        """Provide a progress-bar test double."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept and ignore progress-bar options."""
            return None

        def __enter__(self) -> Self:
            """Enter the progress-bar context."""
            return self

        def __exit__(self, *_: Any) -> bool:
            """Exit the progress-bar context without suppressing errors."""
            return False

        def update(self, _: int) -> None:
            """Ignore a progress update."""
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

    def fake_text(
        _paper: Mapping[str, Any],
        filepath: str | os.PathLike[str],
    ) -> tuple[bool, str, None]:
        """Write replacement full text for a forced download."""
        calls.append('text')
        with open(filepath, 'w', encoding='utf-8') as out_file:
            out_file.write('new text')
        return True, '', None

    def fake_pdf(
        _paper: Mapping[str, Any],
        filepath: str | os.PathLike[str],
        sources: Iterable[str],
    ) -> tuple[bool, str, str]:
        """Write a replacement PDF for a forced download."""
        calls.append(('pdf', sources))
        with open(filepath, 'wb') as out_file:
            out_file.write(b'%PDF new')
        return True, 'openalex', 'https://example.org/new.pdf'

    monkeypatch.setattr(download, '_configured_sources', lambda _: ['openalex', 'elsevier'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(
        download,
        '_download_abstract', lambda *_, **__: calls.append('abstract') or (True, 'openalex', 'new abstract'),
    )
    monkeypatch.setattr(download, '_download_elsevier_text', fake_text)
    monkeypatch.setattr(download, '_download_pdf_from_sources', fake_pdf)

    download.download_papers(str(db_path), download_format='both', force=True)

    with corpus.connect(db_path) as conn:
        abstract_asset = corpus.get_asset(conn, 'paper:refresh', 'abstract')
        text_asset = corpus.get_asset(conn, 'paper:refresh', 'text')
        pdf_asset = corpus.get_asset(conn, 'paper:refresh', 'pdf')
    output = capsys.readouterr().out
    assert calls == ['abstract', 'text', ('pdf', ['openalex', 'elsevier'])]
    assert abstract_asset['content'] == b'new abstract'
    assert text_asset['content'] == b'new text'
    assert pdf_asset['content'] == b'%PDF new'
    assert 'Download complete: 1 text files, 1 PDFs, 1 abstracts downloaded.' in output
    assert 'Skipped existing corpus assets' not in output



def test_full_text_uri_and_download_text_handle_malformed_or_empty_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full text URI and download text handle malformed or empty retrieval."""
    monkeypatch.chdir(tmp_path)
    assert download._full_text_uri({'elsevier_link': 'full-text without quoted uri'}) is None

    monkeypatch.setattr(download.elsevier, 'configured_api_key', lambda *_: 'elsevier-key')
    monkeypatch.setattr(download.elsevier, 'full_text_document',
                        lambda *_, **__: download.provider.FullTextDocument(''))

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
    monkeypatch.setattr(download.elsevier, 'configured_api_key',
                        lambda *_: (_ for _ in ()).throw(
                            ValueError('Elsevier API key is not configured.')))
    with pytest.raises(ValueError, match='Elsevier API key is not configured'):
        download._download_pdf({'doi': '10.1234/example'}, str(out_path))

    monkeypatch.setattr(download.elsevier, 'configured_api_key', lambda *_: 'elsevier-key')
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
    def refuse(*_: object, **__: object) -> dict[str, Any]:
        """Fail the way a rejected Unpaywall request does.

        Parameters
        ----------
        *_ : object
            Positional arguments, unused.
        **__ : object
            Keyword arguments, unused.

        Returns
        -------
        dict[str, Any]
            Never returned.

        Raises
        ------
        RuntimeError
            Always.
        """
        raise RuntimeError('Unpaywall rejected the request with 500')

    monkeypatch.setattr(download.unpaywall, 'configured_email', lambda *_: 'person@example.com')
    monkeypatch.setattr(download.unpaywall, 'get_work', refuse)
    assert download._download_unpaywall_pdf({'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')) == (
        False,
        'Unpaywall rejected the request with 500',
    )

    monkeypatch.setattr(
        download.unpaywall, 'get_work',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('network down')),
    )
    assert download._download_unpaywall_pdf({'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')) == (
        False,
        'network down',
    )

    monkeypatch.setattr(download.unpaywall, 'get_work',
                        lambda *_, **__: {'best_oa_location': None, 'oa_locations': []})
    assert download._download_unpaywall_pdf({'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')) == (
        False,
        'no Unpaywall PDF URL found',
    )

    # An unknown DOI is not a failure, but there is nothing to download either.
    monkeypatch.setattr(download.unpaywall, 'get_work', lambda *_, **__: None)
    assert download._download_unpaywall_pdf({'doi': '10.1234/example'}, str(tmp_path / 'paper.pdf')) == (
        False,
        'Unpaywall knows nothing of this DOI',
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
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['unpaywall', 'elsevier'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(download, '_download_elsevier_text',
                        lambda *_: (False, 'Elsevier text download failed', None))
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
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['elsevier'])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: True)
    monkeypatch.setattr(download, 'tqdm', FakeTqdm)
    monkeypatch.setattr(
        download,
        '_download_elsevier_text',
        lambda *_: (_ for _ in ()).throw(RuntimeError('text exploded')),
    )

    download.download_papers(str(db_path), download_format='text', download_abstract=False)

    papers = read_corpus(db_path)
    assert papers[0]['text_download_status'] == 'failed'
    assert papers[0]['last_error'] == 'elsevier: text exploded'


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
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['core', 'elsevier'])
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

    def fake_download_text(
        paper: Mapping[str, Any],
        filepath: str,
    ) -> tuple[bool, str, None]:
        """Provide fake text download behavior for this test."""
        raise RuntimeError('text exploded')

    monkeypatch.setattr(download, '_download_pdf_from_sources', fake_download_pdf_from_sources)
    monkeypatch.setattr(download, '_download_elsevier_text', fake_download_text)

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
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['core', 'elsevier'])
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

    def fake_download_text(
        paper: Mapping[str, Any],
        filepath: str,
    ) -> tuple[bool, str, None]:
        """Provide fake text download behavior for this test."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('elsevier text')
        return True, '', None

    monkeypatch.setattr(download, '_download_elsevier_text', fake_download_text)

    download.download_papers(str(db_path), download_format='pdf', download_abstract=False)

    papers = read_corpus(db_path)
    assert papers[0]['pdf_download_status'] == 'succeeded'
    assert papers[0]['text_download_status'] == 'succeeded'
    assert papers[0]['text_source'] == 'elsevier'
    assert papers[0]['text_path'] == ''


def test_oa_pdf_followup_records_an_unavailable_elsevier_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Record the registered handler's failure after an open-access PDF succeeds."""
    db_path = tmp_path / 'papers.db'
    paper = {
        'paper_id': 'doi:10.1/x',
        'doi': '10.1/x',
        'elsevier_link': 'has full-text link',
    }
    write_corpus(db_path, [paper])

    def pdf_success(
        row: Mapping[str, Any],
        filepath: str | os.PathLike[str],
        sources: Iterable[str],
    ) -> tuple[bool, str, str]:
        """Write an open-access PDF."""
        Path(filepath).write_bytes(b'%PDF test')
        return True, 'core', 'https://core.example/paper.pdf'

    monkeypatch.setattr(download, '_download_pdf_from_sources', pdf_success)
    monkeypatch.setattr(download, '_download_elsevier_text',
                        lambda *_: (False, 'not entitled', None))
    with corpus.connect(db_path) as conn:
        row = corpus.paper_rows(conn)[0]
        download._download_paper(
            conn, row, tmp_path, 'pdf', ['core', 'elsevier'], download_abstract=False,
        )

    assert row['pdf_download_status'] == 'succeeded'
    assert row['text_download_status'] == 'failed'
    assert row['last_error'] == 'not entitled'


@pytest.mark.network
def test_download_unpaywall_pdf_uses_real_api(tmp_path: Path) -> None:
    """Download Unpaywall PDF uses real API."""
    assert download._unpaywall_email(), (
        'Set unpaywall_email in ~/.config/.paperminertoolkitrc.json or UNPAYWALL_EMAIL before running network tests.'
    )
    pdf_path = tmp_path / 'unpaywall.pdf'

    ok, detail = download._download_unpaywall_pdf({'doi': '10.1371/journal.pone.0000308'}, str(pdf_path))

    assert ok, detail
    assert pdf_path.read_bytes().startswith(b'%PDF')


@pytest.mark.network
def test_download_openalex_pdf_uses_real_api(tmp_path: Path) -> None:
    """Download OpenAlex PDF uses real API."""
    from paperminertoolkit.providers import openalex

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
        'Set core_api_key in ~/.config/.paperminertoolkitrc.json or CORE_API_KEY before running network tests.'
    )
    from paperminertoolkit.workflows.search import core_search

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
        'Set elsevier_api_key in ~/.config/.paperminertoolkitrc.json or ELSEVIER_API_KEY before running network tests.'
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


def test_pubmed_identifier_reads_stored_values_without_a_request() -> None:
    """Resolve a PubMed identifier from the corpus row alone."""
    assert download._pubmed_identifier({'pmid': '31234567'}) == '31234567'
    assert download._pubmed_identifier({'paper_id': 'pmid:31234567'}) == '31234567'
    assert download._pubmed_identifier({'paper_id': 'doi:10.1234/x', 'doi': '10.1234/x'}) is None
    assert download._pubmed_identifier({}) is None


def test_download_pubmed_abstract_uses_a_stored_pmid_and_records_a_resolved_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch by stored PMID, and write back an identifier resolved from a DOI."""
    monkeypatch.setattr(download, '_pubmed_credentials', lambda: (None, ''))
    monkeypatch.setattr(download.pubmed, 'efetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(download.pubmed, 'parse_articles',
                        lambda _: [{'abstract': '<i>Structured</i>   abstract.'}])

    paper = {'pmid': '31234567'}
    monkeypatch.setattr(download.pubmed, 'resolve_pmid', lambda row, **__: str(row.get('pmid') or ''))
    assert download._download_pubmed_abstract(paper) == (True, 'pubmed', 'Structured abstract.')

    paper = {'doi': '10.1234/x'}
    monkeypatch.setattr(download.pubmed, 'resolve_pmid', lambda *_, **__: '99')
    assert download._download_pubmed_abstract(paper)[:2] == (True, 'pubmed')
    assert paper['pmid'] == '99'


def test_download_pubmed_abstract_reports_missing_and_empty_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an unresolvable row and a record that carries no abstract."""
    monkeypatch.setattr(download, '_pubmed_credentials', lambda: (None, ''))
    monkeypatch.setattr(download.pubmed, 'resolve_pmid', lambda *_, **__: '')
    assert download._download_pubmed_abstract({}) == (False, 'missing PMID', '')

    monkeypatch.setattr(download.pubmed, 'resolve_pmid', lambda *_, **__: '7')
    monkeypatch.setattr(download.pubmed, 'efetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(download.pubmed, 'parse_articles', lambda _: [])
    assert download._download_pubmed_abstract({'pmid': '7'}) == (
        False, 'no PubMed record found for 7', '')

    monkeypatch.setattr(download.pubmed, 'parse_articles', lambda _: [{'abstract': ''}])
    assert download._download_pubmed_abstract({'pmid': '7'}) == (
        False, 'no PubMed abstract found for 7', '')


def test_download_abstract_tries_pubmed_between_openalex_and_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall through OpenAlex to PubMed before reaching CORE."""
    calls = []

    def fake_openalex(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
        """Provide a fake OpenAlex abstract lookup."""
        calls.append('openalex')
        return False, 'no OpenAlex abstract found', ''

    def fake_pubmed(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
        """Provide a fake PubMed abstract lookup."""
        calls.append('pubmed')
        return True, 'pubmed', 'From PubMed.'

    def fake_core(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
        """Provide a fake CORE abstract lookup."""
        calls.append('core')
        return False, 'no CORE abstract found', ''

    monkeypatch.setattr(download, '_download_openalex_abstract', fake_openalex)
    monkeypatch.setattr(download, '_download_pubmed_abstract', fake_pubmed)
    monkeypatch.setattr(download, '_download_core_abstract', fake_core)
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    result = download._download_abstract({'doi': '10.1234/x', 'core_id': '5'})

    assert result == (True, 'pubmed', 'From PubMed.')
    assert calls == ['openalex', 'pubmed']


def medrxiv_entry(**overrides: Any) -> dict[str, Any]:
    """Return a mapped medRxiv record for the download helpers."""
    entry = {
        'medrxiv_doi': '10.1101/2024.03.01.24303596',
        'version': '2',
        'abstract': '  A trial\n  abstract. ',
        'jatsxml': 'https://www.medrxiv.org/content/early/2024/03/04/x.source.xml',
    }
    entry.update(overrides)
    return entry


def test_medrxiv_identifier_reads_stored_values_without_a_request() -> None:
    """Recognise every stored form of a medRxiv DOI, and nothing else."""
    assert download._medrxiv_identifier(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596v2'}) == '10.1101/2024.03.01.24303596'
    assert download._medrxiv_identifier(
        {'paper_id': 'doi:10.64898/2026.08.05.26359794'}) == '10.64898/2026.08.05.26359794'
    assert download._medrxiv_identifier(
        {'pdf_url': 'https://www.medrxiv.org/content/10.1101/2020.09.09.20191205v1.full.pdf'}
    ) == '10.1101/2020.09.09.20191205'
    assert download._medrxiv_identifier({}) is None
    # A published DOI names the journal version, which medRxiv does not index.
    assert download._medrxiv_identifier({'doi': '10.1038/s41467-021-21444-5'}) is None


def test_download_medrxiv_pdf_asks_for_the_version_the_record_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fetch the posted version's PDF and report a row that carries no DOI."""
    attempted = []

    def fake_download_url_to_pdf(url: str, filepath: object,
                                 headers: object = None) -> tuple[bool, str]:
        """Provide a fake PDF fetch that records the attempted URL."""
        attempted.append(url)
        return True, ''

    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download_url_to_pdf)
    monkeypatch.setattr(download.medrxiv, 'fetch_doi', lambda *_, **__: medrxiv_entry())

    ok, detail = download._download_medrxiv_pdf(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596'}, tmp_path / 'out.pdf')

    expected = ('https://www.medrxiv.org/content/10.1101/2024.03.01.24303596v2.full.pdf')
    assert (ok, detail) == (True, expected)
    assert attempted == [expected]

    assert download._download_medrxiv_pdf({}, tmp_path / 'out.pdf') == (
        False, 'missing medRxiv DOI')


def test_download_medrxiv_pdf_reports_a_refused_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Surface the reason a PDF was refused rather than working around it."""
    monkeypatch.setattr(download.medrxiv, 'fetch_doi', lambda *_, **__: medrxiv_entry())
    monkeypatch.setattr(download, '_download_url_to_pdf',
                        lambda *_, **__: (False, 'HTTP 403 for medrxiv.org'))

    assert download._download_medrxiv_pdf(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596'}, tmp_path / 'out.pdf') == (
        False, 'HTTP 403 for medrxiv.org')


def test_download_medrxiv_text_writes_the_flattened_jats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Store the JATS full text medRxiv publishes beside every record."""
    monkeypatch.setattr(download.medrxiv, 'fetch_doi', lambda *_, **__: medrxiv_entry())
    document = jats_download()
    monkeypatch.setattr(download.medrxiv, 'full_text_document',
                        lambda *_, **__: document)
    filepath = tmp_path / 'out.txt'

    assert download._download_medrxiv_text(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596'}, filepath) == (
        True, '', document)
    assert filepath.read_text(encoding='utf-8') == 'Title\n\nBody text.'

    monkeypatch.setattr(download.medrxiv, 'full_text_document',
                        lambda *_, **__: download.provider.FullTextDocument(''))
    assert download._download_medrxiv_text(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596'}, filepath) == (
        False, 'no medRxiv full text for 10.1101/2024.03.01.24303596', None)

    assert download._download_medrxiv_text({}, filepath) == (
        False, 'missing medRxiv DOI', None)


def test_download_medrxiv_helpers_report_a_failed_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pass a request failure through as the reason rather than raising."""
    def fail(*_: object, **__: object) -> None:
        """Raise the error a failed medRxiv request produces."""
        raise RuntimeError('medRxiv request failed after 4 attempts')

    monkeypatch.setattr(download.medrxiv, 'fetch_doi', fail)
    row = {'medrxiv_doi': '10.1101/2024.03.01.24303596'}

    assert download._download_medrxiv_abstract(row) == (
        False, 'medRxiv request failed after 4 attempts', '')
    assert download._download_medrxiv_text(row, tmp_path / 'out.txt') == (
        False, 'medRxiv request failed after 4 attempts', None)

    monkeypatch.setattr(download.medrxiv, 'fetch_doi', lambda *_, **__: None)
    assert download._download_medrxiv_abstract(row) == (
        False, 'no medRxiv record found for 10.1101/2024.03.01.24303596', '')


def test_download_medrxiv_abstract_uses_the_stored_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the record's abstract, compacted, and report an empty one."""
    monkeypatch.setattr(download.medrxiv, 'fetch_doi', lambda *_, **__: medrxiv_entry())
    assert download._download_medrxiv_abstract(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596'}) == (True, 'medrxiv', 'A trial abstract.')

    monkeypatch.setattr(download.medrxiv, 'fetch_doi',
                        lambda *_, **__: medrxiv_entry(abstract=''))
    assert download._download_medrxiv_abstract(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596'}) == (
        False, 'no medRxiv abstract found for 10.1101/2024.03.01.24303596', '')

    assert download._download_medrxiv_abstract({}) == (False, 'missing medRxiv DOI', '')


def test_download_abstract_tries_medrxiv_between_pubmed_and_arxiv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall through OpenAlex and PubMed to medRxiv before reaching arXiv."""
    calls = []

    def record(name: str, ok: bool) -> object:
        """Build a fake abstract lookup that records that it ran."""
        def lookup(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
            """Record the provider and return the prepared outcome."""
            calls.append(name)
            return (True, name, f'From {name}.') if ok else (False, f'no {name} abstract', '')
        return lookup

    monkeypatch.setattr(download, '_download_openalex_abstract', record('openalex', False))
    monkeypatch.setattr(download, '_download_pubmed_abstract', record('pubmed', False))
    monkeypatch.setattr(download, '_download_medrxiv_abstract', record('medrxiv', True))
    monkeypatch.setattr(download, '_download_arxiv_abstract', record('arxiv', True))
    monkeypatch.setattr(download, '_download_core_abstract', record('core', False))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    result = download._download_abstract(
        {'doi': '10.1234/x', 'medrxiv_doi': '10.1101/2024.03.01.24303596',
         'arxiv_id': '2301.12345', 'core_id': '5'})

    assert result == (True, 'medrxiv', 'From medrxiv.')
    assert calls == ['openalex', 'pubmed', 'medrxiv']


def test_download_abstract_skips_medrxiv_without_an_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never spend a medRxiv lookup on a row with no medRxiv DOI."""
    def unreachable(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
        """Fail the test if the medRxiv lookup is attempted."""
        raise AssertionError('medRxiv should not be queried without an identifier')

    monkeypatch.setattr(download, '_download_openalex_abstract',
                        lambda _: (False, 'no openalex abstract', ''))
    monkeypatch.setattr(download, '_download_pubmed_abstract',
                        lambda _: (False, 'no pubmed abstract', ''))
    monkeypatch.setattr(download, '_download_medrxiv_abstract', unreachable)
    monkeypatch.setattr(download, '_download_arxiv_abstract',
                        lambda _: (False, 'no arxiv abstract', ''))
    monkeypatch.setattr(download, '_download_core_abstract',
                        lambda _: (False, 'no core abstract', ''))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    ok, reason, abstract = download._download_abstract({'doi': '10.1234/x'})

    assert ok is False
    assert 'medrxiv' not in reason


def test_download_text_from_sources_falls_through_pubmed_to_medrxiv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Offer medRxiv full text once PubMed Central has nothing to give."""
    monkeypatch.setattr(download, '_should_try_pmc_text', lambda _: True)
    monkeypatch.setattr(download, '_download_pmc_text', lambda *_: (False, 'missing PMC ID'))
    monkeypatch.setattr(download, '_download_medrxiv_text', lambda *_: (True, ''))

    row = {'medrxiv_doi': '10.1101/2024.03.01.24303596'}
    assert download._download_text_from_sources(
        row, tmp_path / 'out.txt', ['pubmed', 'medrxiv']) == (True, 'medrxiv', None)

    # A row with no medRxiv DOI never reaches the provider at all.
    ok, reason, _ = download._download_text_from_sources(
        {'doi': '10.1234/x'}, tmp_path / 'out.txt', ['pubmed', 'medrxiv'])
    assert ok is False
    assert reason == 'pubmed: missing PMC ID'


def test_download_text_from_sources_reports_a_failed_medrxiv_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Name medRxiv in the joined reason when its text lookup fails."""
    monkeypatch.setattr(download, '_should_try_pmc_text', lambda _: False)
    monkeypatch.setattr(download, '_download_medrxiv_text',
                        lambda *_: (False, 'no medRxiv full text for 10.1101/x'))

    ok, reason, _ = download._download_text_from_sources(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596'}, tmp_path / 'out.txt', ['medrxiv'])

    assert ok is False
    assert reason == 'medrxiv: no medRxiv full text for 10.1101/x'


def test_download_pdf_from_sources_can_select_medrxiv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Route a medRxiv PDF request through the registered downloader."""
    monkeypatch.setattr(download, '_download_medrxiv_pdf',
                        lambda *_: (True, 'https://www.medrxiv.org/content/x.full.pdf'))

    assert download._download_pdf_from_sources({}, tmp_path / 'out.pdf', ['medrxiv']) == (
        True, 'medrxiv', 'https://www.medrxiv.org/content/x.full.pdf')


def biorxiv_entry(**overrides: Any) -> dict[str, Any]:
    """Return a mapped bioRxiv record for the download helpers."""
    entry = {
        'biorxiv_doi': '10.1101/2023.12.01.569634',
        'version': '2',
        'abstract': '  A memory\n  abstract. ',
        'jatsxml': 'https://www.biorxiv.org/content/early/2023/12/03/x.source.xml',
    }
    entry.update(overrides)
    return entry


def test_biorxiv_identifier_reads_stored_values_without_a_request() -> None:
    """Recognise every stored form of a bioRxiv DOI, and nothing else."""
    assert download._biorxiv_identifier(
        {'biorxiv_doi': '10.1101/2023.12.01.569634v2'}) == '10.1101/2023.12.01.569634'
    assert download._biorxiv_identifier(
        {'paper_id': 'doi:10.64898/2026.08.07.742070'}) == '10.64898/2026.08.07.742070'
    assert download._biorxiv_identifier(
        {'pdf_url': 'https://www.biorxiv.org/content/10.1101/060400v1.full.pdf'}
    ) == '10.1101/060400'
    assert download._biorxiv_identifier({}) is None
    # A published DOI names the journal version, which bioRxiv does not index.
    assert download._biorxiv_identifier({'doi': '10.7554/elife.94191.3'}) is None
    # A medRxiv row belongs to the other archive and is not reachable here.
    assert download._biorxiv_identifier(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596'}) is None


def test_download_biorxiv_pdf_asks_for_the_version_the_record_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fetch the posted version's PDF and report a row that carries no DOI."""
    attempted = []

    def fake_download_url_to_pdf(url: str, filepath: object,
                                 headers: object = None) -> tuple[bool, str]:
        """Provide a fake PDF fetch that records the attempted URL."""
        attempted.append(url)
        return True, ''

    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download_url_to_pdf)
    monkeypatch.setattr(download.biorxiv, 'fetch_doi', lambda *_, **__: biorxiv_entry())

    ok, detail = download._download_biorxiv_pdf(
        {'biorxiv_doi': '10.1101/2023.12.01.569634'}, tmp_path / 'out.pdf')

    expected = 'https://www.biorxiv.org/content/10.1101/2023.12.01.569634v2.full.pdf'
    assert (ok, detail) == (True, expected)
    assert attempted == [expected]

    assert download._download_biorxiv_pdf({}, tmp_path / 'out.pdf') == (
        False, 'missing bioRxiv DOI')


def test_download_biorxiv_pdf_reports_a_refused_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Surface the reason a PDF was refused rather than working around it."""
    monkeypatch.setattr(download.biorxiv, 'fetch_doi', lambda *_, **__: biorxiv_entry())
    monkeypatch.setattr(download, '_download_url_to_pdf',
                        lambda *_, **__: (False, 'HTTP 403 for biorxiv.org'))

    assert download._download_biorxiv_pdf(
        {'biorxiv_doi': '10.1101/2023.12.01.569634'}, tmp_path / 'out.pdf') == (
        False, 'HTTP 403 for biorxiv.org')


def test_download_biorxiv_text_writes_the_flattened_jats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Store the JATS full text bioRxiv publishes beside every record."""
    monkeypatch.setattr(download.biorxiv, 'fetch_doi', lambda *_, **__: biorxiv_entry())
    document = jats_download()
    monkeypatch.setattr(download.biorxiv, 'full_text_document',
                        lambda *_, **__: document)
    filepath = tmp_path / 'out.txt'

    assert download._download_biorxiv_text(
        {'biorxiv_doi': '10.1101/2023.12.01.569634'}, filepath) == (
        True, '', document)
    assert filepath.read_text(encoding='utf-8') == 'Title\n\nBody text.'

    monkeypatch.setattr(download.biorxiv, 'full_text_document',
                        lambda *_, **__: download.provider.FullTextDocument(''))
    assert download._download_biorxiv_text(
        {'biorxiv_doi': '10.1101/2023.12.01.569634'}, filepath) == (
        False, 'no bioRxiv full text for 10.1101/2023.12.01.569634', None)

    assert download._download_biorxiv_text({}, filepath) == (
        False, 'missing bioRxiv DOI', None)


def test_download_biorxiv_helpers_report_a_failed_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pass a request failure through as the reason rather than raising."""
    def fail(*_: object, **__: object) -> None:
        """Raise the error a failed bioRxiv request produces."""
        raise RuntimeError('bioRxiv request failed after 4 attempts')

    monkeypatch.setattr(download.biorxiv, 'fetch_doi', fail)
    row = {'biorxiv_doi': '10.1101/2023.12.01.569634'}

    assert download._download_biorxiv_abstract(row) == (
        False, 'bioRxiv request failed after 4 attempts', '')
    assert download._download_biorxiv_text(row, tmp_path / 'out.txt') == (
        False, 'bioRxiv request failed after 4 attempts', None)

    monkeypatch.setattr(download.biorxiv, 'fetch_doi', lambda *_, **__: None)
    assert download._download_biorxiv_abstract(row) == (
        False, 'no bioRxiv record found for 10.1101/2023.12.01.569634', '')


def test_download_biorxiv_abstract_uses_the_stored_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the record's abstract, compacted, and report an empty one."""
    monkeypatch.setattr(download.biorxiv, 'fetch_doi', lambda *_, **__: biorxiv_entry())
    assert download._download_biorxiv_abstract(
        {'biorxiv_doi': '10.1101/2023.12.01.569634'}) == (True, 'biorxiv', 'A memory abstract.')

    monkeypatch.setattr(download.biorxiv, 'fetch_doi',
                        lambda *_, **__: biorxiv_entry(abstract=''))
    assert download._download_biorxiv_abstract(
        {'biorxiv_doi': '10.1101/2023.12.01.569634'}) == (
        False, 'no bioRxiv abstract found for 10.1101/2023.12.01.569634', '')

    assert download._download_biorxiv_abstract({}) == (False, 'missing bioRxiv DOI', '')


def test_download_abstract_tries_biorxiv_between_medrxiv_and_arxiv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall through to bioRxiv once medRxiv has declined, before reaching arXiv."""
    calls = []

    def record(name: str, ok: bool) -> object:
        """Build a fake abstract lookup that records that it ran."""
        def lookup(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
            """Record the provider and return the prepared outcome."""
            calls.append(name)
            return (True, name, f'From {name}.') if ok else (False, f'no {name} abstract', '')
        return lookup

    monkeypatch.setattr(download, '_download_openalex_abstract', record('openalex', False))
    monkeypatch.setattr(download, '_download_pubmed_abstract', record('pubmed', False))
    monkeypatch.setattr(download, '_download_medrxiv_abstract', record('medrxiv', False))
    monkeypatch.setattr(download, '_download_biorxiv_abstract', record('biorxiv', True))
    monkeypatch.setattr(download, '_download_arxiv_abstract', record('arxiv', True))
    monkeypatch.setattr(download, '_download_core_abstract', record('core', False))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    result = download._download_abstract(
        {'doi': '10.1234/x', 'medrxiv_doi': '10.1101/2024.03.01.24303596',
         'biorxiv_doi': '10.1101/2023.12.01.569634',
         'arxiv_id': '2301.12345', 'core_id': '5'})

    assert result == (True, 'biorxiv', 'From biorxiv.')
    assert calls == ['openalex', 'pubmed', 'medrxiv', 'biorxiv']


def test_download_abstract_skips_biorxiv_without_an_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never spend a bioRxiv lookup on a row with no bioRxiv DOI."""
    def unreachable(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
        """Fail the test if the bioRxiv lookup is attempted."""
        raise AssertionError('bioRxiv should not be queried without an identifier')

    monkeypatch.setattr(download, '_download_openalex_abstract',
                        lambda _: (False, 'no openalex abstract', ''))
    monkeypatch.setattr(download, '_download_pubmed_abstract',
                        lambda _: (False, 'no pubmed abstract', ''))
    monkeypatch.setattr(download, '_download_medrxiv_abstract',
                        lambda _: (False, 'no medrxiv abstract', ''))
    monkeypatch.setattr(download, '_download_biorxiv_abstract', unreachable)
    monkeypatch.setattr(download, '_download_arxiv_abstract',
                        lambda _: (False, 'no arxiv abstract', ''))
    monkeypatch.setattr(download, '_download_core_abstract',
                        lambda _: (False, 'no core abstract', ''))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    # The row carries a medRxiv DOI, which must not be read as a bioRxiv one.
    ok, reason, _ = download._download_abstract(
        {'doi': '10.1234/x', 'medrxiv_doi': '10.1101/2024.03.01.24303596'})

    assert ok is False
    assert 'biorxiv' not in reason


def test_download_text_from_sources_falls_through_medrxiv_to_biorxiv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Offer bioRxiv full text once the earlier providers have nothing to give."""
    monkeypatch.setattr(download, '_should_try_pmc_text', lambda _: False)
    monkeypatch.setattr(download, '_download_biorxiv_text', lambda *_: (True, ''))

    row = {'biorxiv_doi': '10.1101/2023.12.01.569634'}
    assert download._download_text_from_sources(
        row, tmp_path / 'out.txt', ['medrxiv', 'biorxiv']) == (True, 'biorxiv', None)

    # A row with no bioRxiv DOI never reaches the provider at all.
    ok, reason, _ = download._download_text_from_sources(
        {'doi': '10.1234/x'}, tmp_path / 'out.txt', ['medrxiv', 'biorxiv'])
    assert ok is False
    assert reason == 'no full-text source available'


def test_download_text_from_sources_reports_a_failed_biorxiv_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Name bioRxiv in the joined reason when its text lookup fails."""
    monkeypatch.setattr(download, '_should_try_pmc_text', lambda _: False)
    monkeypatch.setattr(download, '_download_biorxiv_text',
                        lambda *_: (False, 'no bioRxiv full text for 10.1101/x'))

    ok, reason, _ = download._download_text_from_sources(
        {'biorxiv_doi': '10.1101/2023.12.01.569634'}, tmp_path / 'out.txt', ['biorxiv'])

    assert ok is False
    assert reason == 'biorxiv: no bioRxiv full text for 10.1101/x'


def test_download_pdf_from_sources_can_select_biorxiv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Route a bioRxiv PDF request through the registered downloader."""
    monkeypatch.setattr(download, '_download_biorxiv_pdf',
                        lambda *_: (True, 'https://www.biorxiv.org/content/x.full.pdf'))

    assert download._download_pdf_from_sources({}, tmp_path / 'out.pdf', ['biorxiv']) == (
        True, 'biorxiv', 'https://www.biorxiv.org/content/x.full.pdf')


def test_download_papers_accepts_biorxiv_as_a_full_text_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat bioRxiv as satisfying the text-format precondition, like medRxiv."""
    def fake_biorxiv_text(paper: Mapping[str, Any], filepath: object) -> tuple[bool, str]:
        """Write the text a successful bioRxiv fetch would have written."""
        with open(filepath, 'w', encoding='utf-8') as out_file:
            out_file.write('Full text.')
        return True, ''

    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'doi:10.1101/2023.12.01.569634',
                            'biorxiv_doi': '10.1101/2023.12.01.569634'}])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    monkeypatch.setattr(download, '_download_biorxiv_text', fake_biorxiv_text)
    monkeypatch.setattr(download, '_download_abstract', lambda *_, **__: (False, 'no abstract', ''))

    download.download_papers(str(db_path), download_format='text', sources=['biorxiv'])

    row = read_corpus(db_path)[0]
    assert row['text_download_status'] == 'succeeded'
    assert row['text_source'] == 'biorxiv'


def test_arxiv_identifier_reads_stored_values_without_a_request() -> None:
    """Recognise every stored form of an arXiv identifier, and nothing else."""
    assert download._arxiv_identifier({'arxiv_id': '2301.12345v2'}) == '2301.12345'
    assert download._arxiv_identifier(
        {'paper_id': 'arxiv:cond-mat/0501001'}) == 'cond-mat/0501001'
    assert download._arxiv_identifier(
        {'pdf_url': 'https://arxiv.org/pdf/2405.00001v1'}) == '2405.00001'
    assert download._arxiv_identifier({}) is None
    # A DOI is not a route into arXiv, so a DOI-only row stays unresolved.
    assert download._arxiv_identifier({'doi': '10.1234/x'}) is None
    assert download._arxiv_identifier({'pdf_url': 'https://example.com/a.pdf'}) is None


def test_download_arxiv_pdf_builds_the_canonical_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fetch the identifier's PDF and report a row that carries none."""
    attempted = []

    def fake_download_url_to_pdf(url: str, filepath: object,
                                 headers: object = None) -> tuple[bool, str]:
        """Provide a fake PDF fetch that records the attempted URL."""
        attempted.append(url)
        return True, ''

    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download_url_to_pdf)

    ok, detail = download._download_arxiv_pdf({'arxiv_id': '2301.12345'}, tmp_path / 'out.pdf')
    assert (ok, detail) == (True, 'https://arxiv.org/pdf/2301.12345')
    assert attempted == ['https://arxiv.org/pdf/2301.12345']

    # An old-style identifier keeps the slash that separates its archive name.
    download._download_arxiv_pdf({'arxiv_id': 'cond-mat/0501001'}, tmp_path / 'out.pdf')
    assert attempted[-1] == 'https://arxiv.org/pdf/cond-mat/0501001'

    assert download._download_arxiv_pdf({}, tmp_path / 'out.pdf') == (False, 'missing arXiv ID')


def test_download_arxiv_abstract_uses_the_stored_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the entry's summary, and report an identifier that finds nothing."""
    monkeypatch.setattr(download.arxiv, 'fetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(download.arxiv, 'parse_entries',
                        lambda _: [{'abstract': '  A garnet\n  abstract. '}])

    assert download._download_arxiv_abstract({'arxiv_id': '2301.12345'}) == (
        True, 'arxiv', 'A garnet abstract.')

    monkeypatch.setattr(download.arxiv, 'parse_entries', lambda _: [])
    assert download._download_arxiv_abstract({'arxiv_id': '2301.12345'}) == (
        False, 'no arXiv record found for 2301.12345', '')

    assert download._download_arxiv_abstract({}) == (False, 'missing arXiv ID', '')


def test_download_abstract_tries_arxiv_between_pubmed_and_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall through OpenAlex and PubMed to arXiv before reaching CORE."""
    calls = []

    def record(name: str, ok: bool) -> object:
        """Build a fake abstract lookup that records that it ran."""
        def lookup(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
            """Record the provider and return the prepared outcome."""
            calls.append(name)
            return (True, name, f'From {name}.') if ok else (False, f'no {name} abstract', '')
        return lookup

    monkeypatch.setattr(download, '_download_openalex_abstract', record('openalex', False))
    monkeypatch.setattr(download, '_download_pubmed_abstract', record('pubmed', False))
    monkeypatch.setattr(download, '_download_arxiv_abstract', record('arxiv', True))
    monkeypatch.setattr(download, '_download_core_abstract', record('core', False))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    result = download._download_abstract(
        {'doi': '10.1234/x', 'arxiv_id': '2301.12345', 'core_id': '5'})

    assert result == (True, 'arxiv', 'From arxiv.')
    assert calls == ['openalex', 'pubmed', 'arxiv']


def test_download_abstract_skips_arxiv_without_an_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never spend an arXiv lookup on a row with no arXiv identifier."""
    monkeypatch.setattr(download, '_download_openalex_abstract',
                        lambda _: (False, 'no OpenAlex abstract found', ''))
    monkeypatch.setattr(download, '_download_pubmed_abstract',
                        lambda _: (False, 'no PubMed abstract found', ''))
    monkeypatch.setattr(download, '_download_arxiv_abstract',
                        lambda _: pytest.fail('arXiv must not be queried'))
    monkeypatch.setattr(download, '_download_core_abstract', lambda _: (True, 'core', 'From CORE.'))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    assert download._download_abstract({'doi': '10.1/x', 'core_id': '5'}) == (
        True, 'core', 'From CORE.')


def test_download_abstract_skips_pubmed_without_a_pmid_or_doi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never spend a PubMed lookup on a row with nothing to resolve from."""
    monkeypatch.setattr(download, '_download_openalex_abstract',
                        lambda _: (False, 'no OpenAlex abstract found', ''))
    monkeypatch.setattr(download, '_download_pubmed_abstract',
                        lambda _: pytest.fail('PubMed must not be queried'))
    monkeypatch.setattr(download, '_download_core_abstract', lambda _: (True, 'core', 'From CORE.'))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    assert download._download_abstract({'core_id': '5'}) == (True, 'core', 'From CORE.')


def test_download_pdf_from_sources_dispatches_every_configured_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Look up a downloader for every name the source set accepts."""
    monkeypatch.setattr(download, '_download_unpaywall_pdf', lambda *_: (False, 'no'))
    monkeypatch.setattr(download, '_download_openalex_pdf', lambda *_: (False, 'no'))
    monkeypatch.setattr(download, '_download_core_pdf', lambda *_: (False, 'no'))
    monkeypatch.setattr(download, '_download_pdf', lambda *_: False)
    monkeypatch.setattr(download, '_download_pubmed_pdf', lambda *_: (True, 'https://pmc/a.pdf'))
    monkeypatch.setattr(download, '_download_arxiv_pdf', lambda *_: (False, 'no'))

    ok, source, url = download._download_pdf_from_sources(
        {'doi': '10.1234/x'}, tmp_path / 'out.pdf', sorted(download.DOWNLOAD_SOURCES))

    assert (ok, source, url) == (True, 'pubmed', 'https://pmc/a.pdf')


def test_download_pubmed_pdf_uses_only_open_access_pdf_links(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Try the offered PDF links and ignore package archives."""
    attempted = []

    def fake_download_url_to_pdf(url: str, filepath: object, headers: object = None) -> tuple[bool, str]:
        """Provide a fake PDF fetch that records each attempted URL."""
        attempted.append(url)
        return url.endswith('.pdf'), 'not a PDF'

    monkeypatch.setattr(download, '_pubmed_credentials', lambda: (None, ''))
    monkeypatch.setattr(download.pubmed, 'resolve_pmcid', lambda *_, **__: 'PMC1')
    monkeypatch.setattr(download.pubmed, 'oa_package_urls',
                        lambda *_, **__: ['https://ftp.ncbi.nlm.nih.gov/a.tar.gz',
                                          'https://ftp.ncbi.nlm.nih.gov/a.pdf'])
    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download_url_to_pdf)

    assert download._download_pubmed_pdf({'pmcid': 'PMC1'}, tmp_path / 'out.pdf') == (
        True, 'https://ftp.ncbi.nlm.nih.gov/a.pdf')
    assert attempted == ['https://ftp.ncbi.nlm.nih.gov/a.pdf']

    monkeypatch.setattr(download.pubmed, 'oa_package_urls', lambda *_, **__: [])
    assert download._download_pubmed_pdf({'pmcid': 'PMC1'}, tmp_path / 'out.pdf') == (
        False, 'no open-access PDF offered for PMC1')

    monkeypatch.setattr(download.pubmed, 'resolve_pmcid', lambda *_, **__: '')
    assert download._download_pubmed_pdf({}, tmp_path / 'out.pdf') == (False, 'missing PMC ID')


def test_download_pmc_text_writes_open_access_full_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Write PMC full text to disk and report an absent record cleanly."""
    monkeypatch.setattr(download, '_pubmed_credentials', lambda: (None, ''))
    monkeypatch.setattr(download.pubmed, 'resolve_pmcid', lambda *_, **__: 'PMC1')
    document = jats_download('Full text body.')
    monkeypatch.setattr(download.pubmed, 'pmc_full_text_document',
                        lambda *_, **__: document)

    filepath = tmp_path / 'paper.txt'
    assert download._download_pmc_text({'pmcid': 'PMC1'}, filepath) == (
        True, '', document)
    assert filepath.read_text(encoding='utf-8') == 'Full text body.'

    monkeypatch.setattr(download.pubmed, 'pmc_full_text_document',
                        lambda *_, **__: download.provider.FullTextDocument(''))
    assert download._download_pmc_text({'pmcid': 'PMC1'}, filepath) == (
        False, 'no open-access full text for PMC1', None)


def test_download_paper_stores_plain_text_and_original_jats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist both representations returned by one full-text acquisition."""
    db_path = tmp_path / 'papers.db'
    paper = {'paper_id': 'pmcid:PMC1', 'pmcid': 'PMC1'}
    write_corpus(db_path, [paper])
    document = jats_download('Derived text.')

    def download_text(
        row: Mapping[str, Any],
        filepath: str | os.PathLike[str],
        sources: Iterable[str],
    ) -> tuple[bool, str, download.provider.FullTextDocument]:
        """Write derived text and return the single acquired JATS document."""
        Path(filepath).write_text(document.text, encoding='utf-8')
        return True, 'pubmed', document

    monkeypatch.setattr(download, '_download_text_from_sources', download_text)
    monkeypatch.setattr(download, '_source_reachable', lambda *args: True)

    with corpus.connect(db_path) as conn:
        summary = download._download_paper(
            conn,
            paper,
            tmp_path,
            'text',
            ['pubmed'],
            download_abstract=False,
        )
        text_asset = corpus.get_asset(conn, paper['paper_id'], 'text')
        documents = corpus.get_structured_documents(conn, paper['paper_id'])

    assert summary['texts'] == 1
    assert text_asset is not None and text_asset['content'] == b'Derived text.'
    assert len(documents) == 1
    assert documents[0]['content'].decode() == document.content
    assert documents[0]['source'] == 'pubmed'
    assert documents[0]['metadata']['document_format'] == 'jats'
    assert documents[0]['metadata']['source_url'] == document.source_url


def test_download_text_from_sources_falls_back_from_elsevier_to_pmc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Try Elsevier first, then PMC, and join the reasons when both fail."""
    monkeypatch.setattr(download, '_should_try_elsevier_text', lambda _: True)
    monkeypatch.setattr(download, '_download_elsevier_text',
                        lambda *_: (_ for _ in ()).throw(RuntimeError('elsevier down')))
    monkeypatch.setattr(download, '_download_pmc_text', lambda *_: (True, ''))

    filepath = tmp_path / 'paper.txt'
    assert download._download_text_from_sources(
        {'pmid': '1'}, filepath, ['elsevier', 'pubmed']) == (True, 'pubmed', None)

    monkeypatch.setattr(download, '_download_pmc_text', lambda *_: (False, 'not open access'))
    ok, detail, _ = download._download_text_from_sources(
        {'pmid': '1'}, filepath, ['elsevier', 'pubmed'])
    assert ok is False
    assert detail == 'elsevier: elsevier down; pubmed: not open access'

    # Elsevier used to be tried whatever the selection said, because it was
    # gated on a key rather than named as a source.
    ok, detail, _ = download._download_text_from_sources({'pmid': '1'}, filepath, ['pubmed'])
    assert detail == 'pubmed: not open access'

    assert download._download_text_from_sources({}, filepath, []) == (
        False, 'no full-text source available', None)


def test_download_papers_allows_text_without_elsevier_when_pubmed_is_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept a text run without an Elsevier key once PubMed can supply text."""
    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'pmid:1', 'pmid': '1'}])
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)
    monkeypatch.setattr(download, '_configured_sources', lambda _: ['pubmed'])
    monkeypatch.setattr(download, '_download_pmc_text',
                        lambda _, filepath: (Path(filepath).write_text('Body.', encoding='utf-8'), (True, ''))[1])

    download.download_papers(str(db_path), download_format='text', download_abstract=False)

    papers = read_corpus(db_path)
    assert papers[0]['text_download_status'] == 'succeeded'
    assert papers[0]['text_source'] == 'pubmed'


def chemrxiv_entry(**overrides: Any) -> dict[str, Any]:
    """Return a mapped chemRxiv record for the download helpers."""
    entry = {
        'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1',
        'version': '1',
        'abstract': '  A catalysis\n  abstract. ',
        'asset_url': 'https://chemrxiv.org/engage/assets/old.pdf',
    }
    entry.update(overrides)
    return entry


def test_chemrxiv_identifier_reads_stored_values_without_a_request() -> None:
    """Recover the chemRxiv DOI from the row, keeping its version suffix."""
    assert download._chemrxiv_identifier(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'}) == '10.26434/chemrxiv.15007737/v1'
    assert download._chemrxiv_identifier(
        {'paper_id': 'doi:10.26434/chemrxiv-2024-bxxhh-v4'}) == '10.26434/chemrxiv-2024-bxxhh-v4'
    assert download._chemrxiv_identifier({'doi': '10.1101/2023.12.01.569634'}) is None
    assert download._chemrxiv_identifier({'doi': '10.1234/journal'}) is None


def test_download_chemrxiv_pdf_asks_for_the_location_the_doi_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prefer the DOI-derived location over the record's stale asset URL."""
    requested = []
    monkeypatch.setattr(download.chemrxiv, 'fetch_doi', lambda *_, **__: chemrxiv_entry())

    def fake_download(url: str, filepath: Any, headers: Any = None) -> tuple[bool, str]:
        """Record the URL requested and report success."""
        requested.append(url)
        return True, ''

    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download)
    ok, source = download._download_chemrxiv_pdf(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'}, tmp_path / 'paper.pdf')

    assert ok
    assert source == 'https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15007737/v1'
    assert requested == ['https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15007737/v1']


def test_download_chemrxiv_pdf_falls_back_to_the_recorded_asset_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Try the record's own asset location when the DOI path does not serve."""
    requested = []
    monkeypatch.setattr(download.chemrxiv, 'fetch_doi', lambda *_, **__: chemrxiv_entry())

    def fake_download(url: str, filepath: Any, headers: Any = None) -> tuple[bool, str]:
        """Fail the DOI-derived location and accept the asset location."""
        requested.append(url)
        return (False, '404 from doi path') if '/doi/pdf/' in url else (True, '')

    monkeypatch.setattr(download, '_download_url_to_pdf', fake_download)
    ok, source = download._download_chemrxiv_pdf(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'}, tmp_path / 'paper.pdf')

    assert ok
    assert source == 'https://chemrxiv.org/engage/assets/old.pdf'
    assert len(requested) == 2


def test_download_chemrxiv_pdf_reports_a_refused_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Report a bot-challenge refusal as the failure reason.

    chemrxiv.org can refuse a client outright. PaperMinerToolkit does not work
    around that, so the refusal has to reach the caller as the reason.
    """
    monkeypatch.setattr(download.chemrxiv, 'fetch_doi', lambda *_, **__: chemrxiv_entry())
    monkeypatch.setattr(download, '_download_url_to_pdf',
                        lambda *_, **__: (False, '403 from chemrxiv.org'))
    ok, error = download._download_chemrxiv_pdf(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'}, tmp_path / 'paper.pdf')

    assert not ok
    assert '403 from chemrxiv.org' in error


def test_download_chemrxiv_helpers_report_a_failed_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Turn a missing identifier, an absent record, and an error into reasons."""
    ok, error = download._download_chemrxiv_pdf({'doi': '10.1234/journal'}, tmp_path / 'p.pdf')
    assert (ok, error) == (False, 'missing chemRxiv DOI')

    monkeypatch.setattr(download.chemrxiv, 'fetch_doi', lambda *_, **__: None)
    ok, error = download._download_chemrxiv_abstract(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'})[:2]
    assert not ok
    assert 'no chemRxiv record found' in error

    def refuse(*_: Any, **__: Any) -> None:
        """Fail the lookup the way a refused request does."""
        raise RuntimeError('chemRxiv refused the request with 403')

    monkeypatch.setattr(download.chemrxiv, 'fetch_doi', refuse)
    ok, error = download._download_chemrxiv_abstract(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'})[:2]
    assert not ok
    assert '403' in error


def test_download_chemrxiv_abstract_uses_the_stored_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the compacted abstract chemRxiv holds for the row."""
    monkeypatch.setattr(download.chemrxiv, 'fetch_doi', lambda *_, **__: chemrxiv_entry())
    ok, source, abstract = download._download_chemrxiv_abstract(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'})

    assert (ok, source) == (True, 'chemrxiv')
    assert abstract == 'A catalysis abstract.'

    monkeypatch.setattr(download.chemrxiv, 'fetch_doi',
                        lambda *_, **__: chemrxiv_entry(abstract=''))
    ok, error, abstract = download._download_chemrxiv_abstract(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'})
    assert not ok
    assert 'no chemRxiv abstract found' in error


def test_download_abstract_tries_chemrxiv_between_biorxiv_and_arxiv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall through to chemRxiv once bioRxiv has declined, before reaching arXiv."""
    calls = []

    def record(name: str, ok: bool) -> object:
        """Build a fake abstract lookup that records that it ran."""
        def lookup(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
            """Record the provider and return the prepared outcome."""
            calls.append(name)
            return (True, name, f'From {name}.') if ok else (False, f'no {name} abstract', '')
        return lookup

    monkeypatch.setattr(download, '_download_openalex_abstract', record('openalex', False))
    monkeypatch.setattr(download, '_download_pubmed_abstract', record('pubmed', False))
    monkeypatch.setattr(download, '_download_medrxiv_abstract', record('medrxiv', False))
    monkeypatch.setattr(download, '_download_biorxiv_abstract', record('biorxiv', False))
    monkeypatch.setattr(download, '_download_chemrxiv_abstract', record('chemrxiv', True))
    monkeypatch.setattr(download, '_download_arxiv_abstract', record('arxiv', True))
    monkeypatch.setattr(download, '_download_core_abstract', record('core', False))
    monkeypatch.setattr(download, '_elsevier_configured', lambda: False)

    result = download._download_abstract(
        {'doi': '10.1234/x', 'biorxiv_doi': '10.1101/2023.12.01.569634',
         'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1',
         'arxiv_id': '2301.12345', 'core_id': '5'})

    assert result == (True, 'chemrxiv', 'From chemrxiv.')
    assert calls == ['openalex', 'pubmed', 'biorxiv', 'chemrxiv']


def test_download_pdf_from_sources_can_select_chemrxiv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Route a PDF download through chemRxiv when it is the chosen source."""
    monkeypatch.setattr(download.chemrxiv, 'fetch_doi', lambda *_, **__: chemrxiv_entry())
    monkeypatch.setattr(download, '_download_url_to_pdf', lambda *_, **__: (True, ''))
    ok, source, url = download._download_pdf_from_sources(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'}, tmp_path / 'p.pdf', ['chemrxiv'])

    assert (ok, source) == (True, 'chemrxiv')
    assert url == 'https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15007737/v1'


def test_chemrxiv_is_not_a_full_text_source() -> None:
    """Confirm chemRxiv is offered for PDFs and abstracts but not for text.

    chemRxiv publishes no machine-readable full text, so naming it in the text
    sources would promise a document that cannot be fetched.
    """
    assert 'chemrxiv' in download.DOWNLOAD_SOURCES
    assert 'chemrxiv' not in download.TEXT_SOURCES
    assert not hasattr(download, '_download_chemrxiv_text')
