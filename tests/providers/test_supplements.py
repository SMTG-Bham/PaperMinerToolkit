"""Offline supplementary discovery and bounded HTTP acquisition regression tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
import requests

from paperminertoolkit.providers import supplements
from tests.doubles import FakeResponse, FakeSession


class StreamResponse(FakeResponse):
    """Add closeable streaming bodies to the shared response double."""

    def __init__(self, content: bytes = b'', **kwargs: Any) -> None:
        """Prepare raw bytes or encode a supplied JSON payload."""
        if 'payload' in kwargs:
            content = json.dumps(kwargs['payload']).encode()
        super().__init__(content=content, **kwargs)
        self.closed = False

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yield small chunks so tests exercise running byte limits."""
        for offset in range(0, len(self.content), min(chunk_size, 4)):
            yield self.content[offset:offset + min(chunk_size, 4)]

    def close(self) -> None:
        """Record stream closure for error-path resource checks."""
        self.closed = True


class StreamSession(FakeSession):
    """Add generic requests to the shared ordered-response session."""

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        """Record each request and return its prepared response or exception."""
        self.calls.append({'method': method, 'url': url, **kwargs})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve test hostnames to a public IP without touching the network."""
    monkeypatch.setattr(supplements.socket, 'getaddrinfo', lambda *_args, **_kwargs: [(2, 1, 6, '', ('93.184.216.34', 443))])


def document(xml: str, url: str = 'https://publisher.example/article.xml', kind: str = 'jats') -> dict[str, Any]:
    """Construct the structured asset shape consumed by discovery."""
    return {'content': xml.encode(), 'metadata': {'source_url': url, 'document_format': kind}}


def test_jats_discovers_only_explicit_supplements_and_inline_files() -> None:
    """Keep supplementary PDFs and datasets while excluding article graphics."""
    xml = '''<article xmlns:xlink="http://www.w3.org/1999/xlink"><body>
        <fig><graphic xlink:href="figure.pdf"/></fig>
        <supplementary-material><label>Supporting experiments</label>
          <media xlink:href="../si/data.csv"/>
          <ext-link xlink:href="https://repo.example/s1.pdf#view"/>
        </supplementary-material>
        <inline-supplementary-material xlink:href="s2.zip"/>
        </body></article>'''
    result = supplements.discover_structured({}, [document(xml)])
    assert [candidate.url for candidate in result.candidates] == [
        'https://publisher.example/si/data.csv', 'https://repo.example/s1.pdf',
        'https://publisher.example/s2.zip',
    ]
    assert result.errors == []
    assert result.candidates[0].label == 'Supporting experiments'


def test_relative_jats_from_shared_efetch_is_not_resolved_as_a_file() -> None:
    """E-utilities is a shared retrieval endpoint, not a media directory."""
    xml = '<article><supplementary-material href="s1.pdf"/></article>'
    result = supplements.discover_structured({}, [document(xml, 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi')])
    assert result.candidates == []


def test_elsevier_supplement_uses_recorded_object_reference() -> None:
    """Resolve internal Elsevier locators through article identifiers."""
    xml = '''<article xmlns:ce="urn:ce" xmlns:xlink="http://www.w3.org/1999/xlink">
      <eid>1-s2.0-S123456</eid><ce:supplementary-material>
      <ce:label>Appendix A</ce:label><ce:link locator="mmc1" xlink:href="pii:S123456/mmc1"/>
      </ce:supplementary-material></article>'''
    result = supplements.discover_structured({}, [document(xml, kind='elsevier')])
    assert result.candidates[0].url == 'https://api.elsevier.com/content/object/eid/1-s2.0-S123456-mmc1'


def test_supplied_urls_are_explicit_and_malformed_xml_is_an_error() -> None:
    """Keep imported links while reporting a failed structured discovery."""
    result = supplements.discover_structured(
        {'supplementary_urls': '[{"url":"https://repo.example/s.pdf","label":"SI"}]'},
        [document('<broken')],
    )
    assert result.candidates[0].url == 'https://repo.example/s.pdf'
    assert result.candidates[0].label == 'SI'
    assert 'malformed' in result.errors[0]


def test_pmc_cloud_matches_jats_files_and_ignores_unrelated_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Download only files explicitly identified as SI in the newest JATS."""
    base = 'https://pmc-oa-opendata.s3.amazonaws.com/PMC1.2/'
    monkeypatch.setattr(supplements.pubmed, 'resolve_pmcid', lambda *_args, **_kwargs: 'PMC1')
    monkeypatch.setattr(supplements.pubmed, 'pmc_cloud_files', lambda *_args, **_kwargs: {
        'PMC1.2.xml': base + 'PMC1.2.xml', 'PMC1.2.pdf': base + 'PMC1.2.pdf',
        'figure.jpg': base + 'figure.jpg', 's1.pdf': base + 's1.pdf',
    })
    session = StreamSession([StreamResponse(b'<article><supplementary-material href="s1.pdf"/><supplementary-material href="withheld.zip"/></article>')])
    result = supplements.discover_pubmed({'pmcid': 'PMC1'}, session=session)
    assert [candidate.url for candidate in result.candidates] == [base + 's1.pdf']
    assert result.errors == []
    assert len(session.calls) == 1


def test_figshare_rejects_unrelated_search_hits_and_validates_exact_parent() -> None:
    """A DOI mentioned only in an unrelated record must never attach its files."""
    file = {'id': 5, 'name': 'si.pdf', 'download_url': 'https://figshare.com/ndownloader/files/5'}
    session = StreamSession([
        StreamResponse(payload=[{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}]),
        StreamResponse(payload={'resource_doi': '10.1234/paper-extra', 'files': [file]}),
        StreamResponse(payload={'resource_doi': 'https://doi.org/10.1234/PAPER', 'files': [file]}),
        StreamResponse(payload={'related_materials': [{'identifier': '10.1234/paper', 'relation': 'IsReferencedBy'}], 'files': [file]}),
        StreamResponse(payload={'related_materials': [{'identifier': '10.1234/paper', 'relation': 'IsSupplementTo'}], 'files': [{**file, 'id': 6, 'download_url': 'https://figshare.com/ndownloader/files/6'}]}),
    ])
    result = supplements.discover_figshare({'doi': '10.1234/paper'}, session=session)
    assert [candidate.metadata['figshare_article_id'] for candidate in result.candidates] == ['2', '4']
    assert session.calls[0]['method'] == 'POST'
    assert session.calls[0]['json']['search_for'] == '"10.1234/paper"'
    assert result.errors == []


def test_figshare_acs_doi_suffix_must_match_whole_identifier() -> None:
    """ACS SI suffix matching cannot select a different paper's longer DOI."""
    assert supplements._related_doi({'doi': '10.1021/acs.123.s001'}, '10.1021/acs.123')
    assert not supplements._related_doi({'doi': '10.1021/acs.1234.s001'}, '10.1021/acs.123')


def test_publisher_keeps_explicit_supporting_files_and_resolves_redirect_base() -> None:
    """Follow landing redirects and reject ordinary article/figure links."""
    html = b'''<html><body><a href="article.pdf">Download PDF</a>
      <a href="/suppdata/a/b/s1.pdf">PDF</a>
      <a href="data.zip">Supporting information</a>
      <a href="figure.png">Figure 1</a><a href="#supplementary">Supplementary</a>
      </body></html>'''
    session = StreamSession([
        StreamResponse(status_code=302, headers={'Location': 'https://publisher.example/paper/'}),
        StreamResponse(html, headers={'Content-Type': 'text/html'}),
    ])
    result = supplements.discover_publisher({'doi': '10.1234/paper'}, session=session)
    assert [candidate.url for candidate in result.candidates] == [
        'https://publisher.example/suppdata/a/b/s1.pdf', 'https://publisher.example/paper/data.zip',
    ]
    assert all(call['allow_redirects'] is False for call in session.calls)


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'ftp://host.example/a', 'https://user:secret@host.example/a', 'https://localhost/s.pdf'])
def test_rejects_nonpublic_url_forms_before_request(url: str) -> None:
    """Do not issue requests to non-web URLs or credential/local destinations."""
    session = StreamSession([])
    with pytest.raises(ValueError):
        supplements.download_supplement(supplements.SupplementCandidate(url, 'structured'), session)
    assert session.calls == []


def test_rejects_redirect_to_private_ip_before_following(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every redirect before requesting it, including IP literals."""
    def resolve(host: str, *_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        """Resolve a public first host and a private redirected target."""
        return [(2, 1, 6, '', ('127.0.0.1' if host == '127.0.0.1' else '93.184.216.34', 443))]
    monkeypatch.setattr(supplements.socket, 'getaddrinfo', resolve)
    response = StreamResponse(status_code=302, headers={'Location': 'http://127.0.0.1/private'})
    session = StreamSession([response])
    with pytest.raises(ValueError, match='non-public'):
        supplements.download_supplement(supplements.SupplementCandidate('https://host.example/s', 'publisher'), session)
    assert len(session.calls) == 1
    assert response.closed


@pytest.mark.parametrize('headers,content', [
    ({'Content-Type': 'text/html'}, b'challenge'),
    ({'Content-Type': 'application/pdf'}, b'<!DOCTYPE html><html>denied</html>'),
    ({'Content-Type': 'application/pdf'}, b'not a pdf'),
])
def test_error_pages_are_not_stored_as_pdf(headers: Mapping[str, str], content: bytes) -> None:
    """Inspect both response type and file signature when publishers return 200."""
    response = StreamResponse(content, headers=headers)
    with pytest.raises(RuntimeError):
        supplements.download_supplement(supplements.SupplementCandidate('https://host.example/s.pdf', 'publisher', 's.pdf'), StreamSession([response]))
    assert response.closed


def test_stream_limit_is_enforced_without_trusting_content_length() -> None:
    """Bound chunked or falsely labelled responses and close oversized streams."""
    response = StreamResponse(b'123456789')
    with pytest.raises(ValueError, match='byte limit'):
        supplements._read_url('https://host.example/large', session=StreamSession([response]), max_bytes=8)
    assert response.closed


def test_retry_timeout_and_throttle_then_validate_integrity() -> None:
    """Transient network failures and rate limits restart a checked download."""
    data = b'%PDF-1.7 supporting experiments'
    candidate = supplements.SupplementCandidate('https://host.example/s.pdf', 'figshare', 's.pdf', metadata={
        'expected_size': len(data), 'expected_md5': hashlib.md5(data).hexdigest(),
    })
    session = StreamSession([
        requests.Timeout('timed out'),
        StreamResponse(status_code=429, headers={'Retry-After': '0'}),
        StreamResponse(data, headers={'Content-Disposition': 'attachment; filename="../../SI.pdf"'}),
    ])
    downloaded = supplements.download_supplement(candidate, session)
    assert downloaded.filename == 'SI.pdf'
    assert downloaded.content == data
    assert downloaded.mime_type == 'application/pdf'
    assert len(session.calls) == 3


def test_figshare_expected_size_mismatch_reports_damage() -> None:
    """A successful response cannot hide a truncated repository file."""
    candidate = supplements.SupplementCandidate('https://host.example/s.csv', 'figshare', 's.csv', metadata={'expected_size': 100})
    with pytest.raises(RuntimeError, match='size'):
        supplements.download_supplement(candidate, StreamSession([StreamResponse(b'a,b\n1,2')]))


def test_discovery_uses_registry_and_keeps_errors_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed providers remain errors even when another discovers a file."""
    candidate = supplements.SupplementCandidate('https://host.example/s.pdf', 'structured')
    monkeypatch.setattr(supplements, 'discover_structured', lambda *_args, **_kwargs: supplements.DiscoveryResult([candidate]))
    monkeypatch.setattr(supplements, 'discover_pubmed', lambda *_args, **_kwargs: supplements.DiscoveryResult(errors=['pubmed unavailable']))
    monkeypatch.setattr(supplements, 'discover_publisher', lambda *_args, **_kwargs: pytest.fail('publisher is a fallback'))
    result = supplements.discover_supplements({}, sources=['structured', 'pubmed', 'publisher'])
    assert result.candidates == [candidate]
    assert result.errors == ['pubmed unavailable']


def test_elsevier_attachment_label_marks_an_explicit_supplement() -> None:
    """Supplement-labelled Elsevier attachments can be outside supplementary tags."""
    xml = '''<article><eid>1-s2.0-123</eid><attachment>
      <label>Supplementary material</label><e-component><link locator="mmc1"/>
      </e-component></attachment><attachment><label>Graphical abstract</label>
      <link locator="gr1"/></attachment></article>'''
    result = supplements.discover_structured({}, [document(xml, kind='elsevier')])
    assert [candidate.url for candidate in result.candidates] == [
        'https://api.elsevier.com/content/object/eid/1-s2.0-123-mmc1',
    ]


def test_shared_provider_limiters_apply_to_si_and_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preprint SI observes the same seven-second window as article downloads."""
    waits: list[Any] = []
    monkeypatch.setattr(supplements.base.RateLimiter, 'wait', lambda self: waits.append(self))
    session = StreamSession([
        StreamResponse(status_code=302, headers={'Location': 'https://www.biorxiv.org/content/s.pdf'}),
        StreamResponse(b'%PDF-1.7'),
    ])
    supplements.download_supplement(supplements.SupplementCandidate('https://files.example/s.pdf', 'publisher'), session)
    assert waits == [supplements.LIMITER, supplements.rxiv.CONTENT_LIMITER]
    assert supplements.supplement_limiter('https://www.medrxiv.org/content/s.pdf') is supplements.rxiv.CONTENT_LIMITER
    assert supplements.supplement_limiter('https://pmc-oa-opendata.s3.amazonaws.com/PMC1.1/s.pdf') is supplements.pubmed.CLOUD_LIMITER
    assert supplements.supplement_limiter('https://api.elsevier.com/content/object/eid/1') is supplements.elsevier.limiter_for('https://api.elsevier.com/content/object/eid/1')


def test_structured_supplements_preserve_known_licenses_and_article_ids() -> None:
    """SI provenance records the parent article's explicit licensing metadata."""
    xml = '''<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>
        <article-id pub-id-type="doi">10.1234/paper</article-id>
        <permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/">
        <license-p>CC BY 4.0</license-p></license></permissions></article-meta></front>
        <supplementary-material xlink:href="s1.pdf"/></article>'''
    item = document(xml)
    item['metadata']['source_identifier'] = 'PMC1'
    result = supplements.discover_structured({}, [item])
    assert result.candidates[0].metadata['license']['url'].endswith('/by/4.0/')
    assert result.candidates[0].metadata['article_identifiers']['doi'] == '10.1234/paper'
    assert result.candidates[0].metadata['source_identifier'] == 'PMC1'
