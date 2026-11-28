"""Offline supplementary discovery and bounded HTTP acquisition regression tests."""

from __future__ import annotations

import hashlib
import json
import socket
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, NoReturn

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


class ClosingSession(StreamSession):
    """Stand in for the fresh ``requests.Session`` a caller-less read creates."""

    def __init__(self, responses: Iterable[FakeResponse]) -> None:
        """Start trusting environment proxies, as a new requests session does."""
        super().__init__(responses)
        self.trust_env = True
        self.closed = False

    def close(self) -> None:
        """Record that the owned session was released."""
        self.closed = True


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


def test_malformed_port_is_rejected_before_request() -> None:
    """An unparseable port is reported as an invalid URL, not sent anywhere."""
    session = StreamSession([])
    with pytest.raises(ValueError, match='invalid supplementary URL'):
        supplements.download_supplement(supplements.SupplementCandidate('https://host.example:99999/s.pdf', 'publisher'), session)
    assert session.calls == []


def test_unresolvable_host_is_reported_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DNS failure becomes a resolution error rather than an attempted request."""
    def unresolvable(*_args: Any, **_kwargs: Any) -> NoReturn:
        """Fail resolution the way an unknown hostname does."""
        raise socket.gaierror('Name or service not known')
    monkeypatch.setattr(supplements.socket, 'getaddrinfo', unresolvable)
    session = StreamSession([])
    with pytest.raises(RuntimeError, match='cannot resolve supplementary host missing.example'):
        supplements.download_supplement(supplements.SupplementCandidate('https://missing.example/s.pdf', 'publisher'), session)
    assert session.calls == []


@pytest.mark.parametrize('value', ['CON.txt', 'lpt1', 'aux.tar.gz', '../..', '%2F%2E%2E'])
def test_safe_filename_replaces_reserved_and_empty_names(value: str) -> None:
    """Windows device names and names emptied by sanitizing use the fallback."""
    assert supplements.safe_filename(value) == 'supplement.bin'
    assert supplements.safe_filename(value, fallback='mmc1.pdf') == 'mmc1.pdf'


def test_safe_filename_truncates_long_names_but_keeps_extension() -> None:
    """Overlong names are cut to 180 characters without losing the file type."""
    name = supplements.safe_filename('a' * 300 + '.xlsx')
    assert len(name) == 180
    assert name == 'a' * 175 + '.xlsx'


def test_elsevier_objects_carry_key_and_stop_at_exhausted_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send the key only to Elsevier objects and refuse once their quota is spent."""
    monkeypatch.setenv('ELSEVIER_API_KEY', 'test-elsevier-key')
    for other in ('http://api.elsevier.com/content/object/eid/1', 'https://api.elsevier.com/content/article/doi/1',
                  'https://mirror.example/content/object/eid/1'):
        assert 'X-ELS-APIKey' not in supplements._headers(other)
    url = 'https://api.elsevier.com/content/object/eid/1-s2.0-S123-mmc1.pdf'
    candidate = supplements.SupplementCandidate(url, 'structured', 'mmc1.pdf')
    session = StreamSession([StreamResponse(b'%PDF-1.7', headers={'X-RateLimit-Remaining': '0', 'X-RateLimit-Limit': '500'})])
    downloaded = supplements.download_supplement(candidate, session)
    assert downloaded.content == b'%PDF-1.7'
    assert session.calls[0]['headers']['X-ELS-APIKey'] == 'test-elsevier-key'
    assert supplements.elsevier.quota_status(url, 'test-elsevier-key').remaining == 0
    with pytest.raises(RuntimeError, match='reports 0 of 500 requests left'):
        supplements.download_supplement(candidate, session)
    assert len(session.calls) == 1


def test_elsevier_objects_are_requested_without_a_key_when_none_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send an unauthenticated request rather than stop the run when no Elsevier key is set."""
    def no_key() -> str:
        """Fail as the settings lookup does when no key is stored or exported."""
        raise ValueError('Elsevier API key is not configured.')

    monkeypatch.setattr(supplements.elsevier, 'configured_api_key', no_key)
    url = 'https://api.elsevier.com/content/object/eid/1-s2.0-S456-mmc2.pdf'
    assert 'X-ELS-APIKey' not in supplements._headers(url)
    candidate = supplements.SupplementCandidate(url, 'structured', 'mmc2.pdf')
    session = StreamSession([StreamResponse(b'%PDF-1.7')])
    assert supplements.download_supplement(candidate, session).content == b'%PDF-1.7'
    assert 'X-ELS-APIKey' not in session.calls[0]['headers']


def test_owned_session_ignores_environment_proxies_and_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read without a caller session disables proxy trust and always closes it."""
    owned = [ClosingSession([StreamResponse(b'{"ok": true}')]), ClosingSession([StreamResponse(status_code=404)])]
    created = iter(owned)
    monkeypatch.setattr(supplements.requests, 'Session', lambda: next(created))
    assert supplements._json_url('https://host.example/meta.json') == {'ok': True}
    with pytest.raises(RuntimeError, match='HTTP 404'):
        supplements._json_url('https://host.example/missing.json')
    assert [(session.trust_env, session.closed) for session in owned] == [(False, True), (False, True)]
    assert owned[0].calls[0]['method'] == 'GET'


@pytest.mark.parametrize('kwargs,error,message', [
    ({'status_code': 404}, RuntimeError, 'HTTP 404 from https://host.example/s'),
    ({'status_code': 204}, RuntimeError, 'unexpected HTTP 204'),
    ({'status_code': 302}, RuntimeError, 'missing location'),
    ({'content': b'123456789', 'headers': {'Content-Length': '9'}}, ValueError, 'exceeds 8 byte limit'),
])
def test_unusable_responses_fail_without_retry(kwargs: dict[str, Any], error: type[Exception], message: str) -> None:
    """Permanent errors, odd statuses, bare redirects, and declared oversize stop at once."""
    response = StreamResponse(**kwargs)
    session = StreamSession([response])
    with pytest.raises(error, match=message):
        supplements._read_url('https://host.example/s', session=session, max_bytes=8)
    assert len(session.calls) == 1
    assert response.closed


def test_redirect_chain_is_bounded() -> None:
    """Stop after the redirect budget rather than following a loop forever."""
    responses = [StreamResponse(status_code=307, headers={'Location': f'/hop{index}'}) for index in range(supplements.MAX_REDIRECTS + 1)]
    session = StreamSession(responses)
    with pytest.raises(RuntimeError, match='redirect limit exceeded'):
        supplements._read_url('https://host.example/start', session=session)
    assert [call['url'] for call in session.calls] == ['https://host.example/start'] + [
        f'https://host.example/hop{index}' for index in range(supplements.MAX_REDIRECTS)]
    assert all(response.closed for response in responses)


def test_truncated_responses_are_retried_then_reported() -> None:
    """A body shorter than its Content-Length is retried, then reported as failed."""
    responses = [StreamResponse(b'%PDF-1', headers={'Content-Length': '64'}) for _ in range(3)]
    session = StreamSession(responses)
    with pytest.raises(RuntimeError, match='request failed: incomplete supplementary response'):
        supplements._read_url('https://host.example/s.pdf', session=session)
    assert len(session.calls) == 3
    assert all(response.closed for response in responses)


def test_compressed_response_is_not_held_to_its_encoded_length() -> None:
    """Content-Length counts encoded bytes, so a decoded body may differ in size."""
    session = StreamSession([StreamResponse(b'decoded body', headers={'Content-Length': '5', 'Content-Encoding': 'gzip'})])
    content, headers, final_url = supplements._read_url('https://host.example/s.txt', session=session)
    assert content == b'decoded body'
    assert headers['content-encoding'] == 'gzip'
    assert final_url == 'https://host.example/s.txt'


@pytest.mark.parametrize('xml,source_url,expected', [
    ('<article><eid>1-s2.0-S1</eid><supplementary-material><link locator="mmc1"/></supplementary-material>'
     '<object id="mmc1" href="https://cdn.example/mmc1.pdf"/></article>',
     'https://publisher.example/a.xml', ['https://cdn.example/mmc1.pdf']),
    ('<article><supplementary-material href="ftp://ftp.example/s1.zip"/>'
     '<supplementary-material href="mailto:author@example.org"/></article>',
     'https://publisher.example/a.xml', []),
    ('<article><supplementary-material><link locator="mmc1"/></supplementary-material></article>',
     'https://publisher.example/a.xml', []),
    ('<article><supplementary-material href="s1.pdf"/></article>', 'file:///data/article.xml', []),
], ids=['linked-object-href', 'non-web-scheme', 'locator-without-eid', 'relative-to-local-file'])
def test_structured_references_resolve_only_to_web_urls(xml: str, source_url: str, expected: list[str]) -> None:
    """Follow explicit object links, and drop references with no public web address."""
    result = supplements.discover_structured({}, [document(xml, source_url, kind='elsevier')])
    assert [candidate.url for candidate in result.candidates] == expected
    assert result.errors == []


def test_structured_skips_non_xml_documents_and_reports_oversized_xml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only XML formats are parsed, and an oversized one is an error, not a parse."""
    monkeypatch.setattr(supplements, 'MAX_METADATA_BYTES', 64)
    stored_pdf = document('<article><supplementary-material href="https://repo.example/s1.pdf"/></article>', kind='pdf')
    oversized = document('<article>' + ' ' * 100 + '<supplementary-material href="s1.pdf"/></article>')
    result = supplements.discover_structured({}, [stored_pdf, oversized])
    assert result.candidates == []
    assert result.errors == ['structured: supplementary XML exceeds metadata size limit']


def test_supplied_url_fields_accept_plain_strings_and_ignore_malformed_json() -> None:
    """Imported URLs may be strings, mappings, or lists; malformed JSON adds nothing."""
    paper = {
        'supplementary_urls': '[not json',
        'supplement_urls': [{'href': 'https://repo.example/a.csv', 'filename': 'a.csv'}, 'https://repo.example/b.pdf', 42],
        'supplementary_url': '   ',
        'supplement_url': 'https://repo.example/c.zip#download',
    }
    result = supplements.discover_structured(paper)
    assert [(item.url, item.filename, item.metadata['discovery_field']) for item in result.candidates] == [
        ('https://repo.example/a.csv', 'a.csv', 'supplement_urls'),
        ('https://repo.example/b.pdf', '', 'supplement_urls'),
        ('https://repo.example/c.zip', '', 'supplement_url'),
    ]
    assert result.errors == []


def test_pubmed_without_pmcid_or_versioned_xml_requests_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No PMCID, or a cloud listing without that article's JATS, ends discovery quietly."""
    session = StreamSession([])
    monkeypatch.setattr(supplements.pubmed, 'resolve_pmcid', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(supplements.pubmed, 'pmc_cloud_files', lambda *_args, **_kwargs: pytest.fail('no PMCID to list'))
    assert supplements.discover_pubmed({'doi': '10.1234/paper'}, session=session) == supplements.DiscoveryResult()
    cloud = 'https://pmc-oa-opendata.s3.amazonaws.com/'
    monkeypatch.setattr(supplements.pubmed, 'resolve_pmcid', lambda *_args, **_kwargs: 'PMC1')
    monkeypatch.setattr(supplements.pubmed, 'pmc_cloud_files', lambda *_args, **_kwargs: {
        'PMC1.2.pdf': cloud + 'PMC1.2/PMC1.2.pdf', 'PMC10.1.xml': cloud + 'PMC10.1/PMC10.1.xml',
    })
    assert supplements.discover_pubmed({'doi': '10.1234/paper'}, session=session) == supplements.DiscoveryResult()
    assert session.calls == []


def test_pubmed_keeps_absolute_links_absent_from_cloud_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit external SI link survives even though PMC does not host it."""
    xml_url = 'https://pmc-oa-opendata.s3.amazonaws.com/PMC1.2/PMC1.2.xml'
    monkeypatch.setattr(supplements.pubmed, 'resolve_pmcid', lambda *_args, **_kwargs: 'PMC1')
    monkeypatch.setattr(supplements.pubmed, 'pmc_cloud_files', lambda *_args, **_kwargs: {'PMC1.2.xml': xml_url})
    session = StreamSession([StreamResponse(b'<article><supplementary-material href="https://repo.example/data.zip"/></article>')])
    result = supplements.discover_pubmed({'pmcid': 'PMC1'}, session=session)
    assert [candidate.url for candidate in result.candidates] == ['https://repo.example/data.zip']
    assert result.candidates[0].metadata['pmcid'] == 'PMC1'
    assert result.candidates[0].metadata['pmc_version'] == 'PMC1.2'
    assert result.notes == []
    assert result.errors == []


def test_pubmed_reports_failed_cloud_xml_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused JATS download is a discovery error, not an empty result."""
    xml_url = 'https://pmc-oa-opendata.s3.amazonaws.com/PMC1.2/PMC1.2.xml'
    monkeypatch.setattr(supplements.pubmed, 'resolve_pmcid', lambda *_args, **_kwargs: 'PMC1')
    monkeypatch.setattr(supplements.pubmed, 'pmc_cloud_files', lambda *_args, **_kwargs: {'PMC1.2.xml': xml_url})
    result = supplements.discover_pubmed({'pmcid': 'PMC1'}, session=StreamSession([StreamResponse(status_code=403)]))
    assert result.candidates == []
    assert result.errors == [f'pubmed: HTTP 403 from {xml_url}']


def test_figshare_without_doi_sends_no_request() -> None:
    """Figshare matching needs the parent DOI, so none means no search."""
    session = StreamSession([])
    assert supplements.discover_figshare({'doi': '  '}, session=session) == supplements.DiscoveryResult()
    assert session.calls == []


@pytest.mark.parametrize('kwargs,message', [
    ({'payload': {'message': 'Internal error'}}, 'figshare: Figshare search returned a non-list response'),
    ({'content': b'<html>busy</html>'}, f'figshare: malformed supplementary JSON from {supplements.FIGSHARE_URL}/articles/search'),
])
def test_figshare_unusable_search_response_is_an_error(kwargs: dict[str, Any], message: str) -> None:
    """A search that is not a JSON list cannot support a claim of no files."""
    result = supplements.discover_figshare({'doi': '10.1234/paper'}, session=StreamSession([StreamResponse(**kwargs)]))
    assert result.candidates == []
    assert result.errors == [message]


def test_figshare_skips_unusable_entries_and_files_and_reports_bad_items() -> None:
    """Ignore malformed hits and link-only files, and record per-item failures."""
    good = {'id': 11, 'name': 'si.csv', 'download_url': 'https://figshare.com/ndownloader/files/11',
            'size': 7, 'computed_md5': 'a' * 32}
    session = StreamSession([
        StreamResponse(payload=['junk', {'id': 'abc'}, {'title': 'no id'}, {'id': 7}, {'id': 7}, {'id': 8}, {'id': 9}]),
        StreamResponse(payload={
            'title': 'Supporting data', 'doi': '10.6084/m9.figshare.7', 'license': {'name': 'CC BY 4.0'},
            'related_materials': ['not a mapping', {'identifier': 'https://doi.org/10.1234/PAPER', 'relation_type': 'IsSupplementTo'}],
            'files': ['junk', {'id': 9, 'name': 'external', 'download_url': 'https://example.org/x', 'is_link_only': True},
                      {'id': 10, 'name': 'no-url.csv'}, good],
        }),
        StreamResponse(payload=['not', 'an', 'object']),
        StreamResponse(b'not json'),
    ])
    result = supplements.discover_figshare({'doi': '10.1234/paper'}, session=session)
    assert [candidate.url for candidate in result.candidates] == ['https://figshare.com/ndownloader/files/11']
    metadata = result.candidates[0].metadata
    assert (metadata['figshare_article_id'], metadata['figshare_file_id'], metadata['expected_size']) == ('7', 11, 7)
    assert metadata['expected_md5'] == 'a' * 32
    assert metadata['item_doi'] == '10.6084/m9.figshare.7'
    assert result.candidates[0].label == 'Supporting data'
    assert result.errors == [
        'figshare item 8: Figshare article returned a non-object response',
        f'figshare item 9: malformed supplementary JSON from {supplements.FIGSHARE_URL}/articles/9',
    ]
    assert [call['url'] for call in session.calls] == [f'{supplements.FIGSHARE_URL}/articles/search'] + [
        f'{supplements.FIGSHARE_URL}/articles/{identifier}' for identifier in (7, 8, 9)]


def test_figshare_marks_discovery_incomplete_after_five_full_pages() -> None:
    """Five full search pages are the cap, so later hits are reported as unseen."""
    page = [{'id': 1}] * 100
    article = {'resource_doi': '10.1234/paper',
               'files': [{'id': 2, 'name': 's.pdf', 'download_url': 'https://figshare.com/ndownloader/files/2'}]}
    session = StreamSession([StreamResponse(payload=page), StreamResponse(payload=article),
                             *(StreamResponse(payload=page) for _ in range(4))])
    result = supplements.discover_figshare({'doi': '10.1234/paper'}, session=session)
    assert [candidate.url for candidate in result.candidates] == ['https://figshare.com/ndownloader/files/2']
    assert result.errors == ['figshare: search exceeded 500 results; discovery is incomplete']
    assert [call['json']['page'] for call in session.calls if call['method'] == 'POST'] == [1, 2, 3, 4, 5]
    assert len(session.calls) == 6


def test_publisher_without_url_or_doi_requests_nothing() -> None:
    """With no landing page and no DOI there is nothing to read."""
    session = StreamSession([])
    assert supplements.discover_publisher({'title': 'Untitled'}, session=session) == supplements.DiscoveryResult()
    assert session.calls == []


def test_publisher_reads_citation_supplementary_meta_tags() -> None:
    """Publisher citation meta tags name SI files explicitly, even without anchors."""
    html = (b'<html><head><meta name="citation_supplementary_material_url" content="/doi/suppl/10.1234/paper/suppl_file/si.pdf">'
            b'<meta name="citation_pdf_url" content="/doi/pdf/10.1234/paper"></head><body></body></html>')
    session = StreamSession([StreamResponse(html)])
    result = supplements.discover_publisher({'landing_page_url': 'https://publisher.example/doi/10.1234/paper'}, session=session)
    assert [(item.url, item.filename, item.label) for item in result.candidates] == [
        ('https://publisher.example/doi/suppl/10.1234/paper/suppl_file/si.pdf', 'si.pdf', 'Supplementary material'),
    ]
    assert result.errors == []


@pytest.mark.parametrize('content,headers,errors', [
    (b'  \n ', {'Content-Type': 'text/html'}, ['publisher: empty publisher landing page']),
    (b'%PDF-1.7 <a href="si.pdf">Supporting information</a>', {'Content-Type': 'application/pdf'}, []),
    (b'<html><title>Just a moment...</title><div id="cf-chl-widget"></div><a href="si.pdf">Supplementary</a></html>',
     {'Content-Type': 'text/html'}, ['publisher: publisher returned a browser or access challenge']),
], ids=['empty', 'not-html', 'challenge'])
def test_publisher_pages_that_are_not_articles_yield_no_links(content: bytes, headers: dict[str, str], errors: list[str]) -> None:
    """Empty, non-HTML, and challenge pages are never mined for SI links."""
    session = StreamSession([StreamResponse(content, headers=headers)])
    result = supplements.discover_publisher({'landing_page_url': 'https://publisher.example/article/1'}, session=session)
    assert result.candidates == []
    assert result.errors == errors


def test_discovery_rejects_unknown_sources() -> None:
    """Misspelled source names fail loudly instead of silently discovering less."""
    with pytest.raises(ValueError, match='unknown supplementary sources: nope, zenodo'):
        supplements.discover_supplements({}, sources=['structured', 'zenodo', 'nope'])


def test_discovery_caps_candidates_and_reports_truncation() -> None:
    """More than the file cap keeps the first files and marks the result incomplete."""
    urls = [f'https://repo.example/s{index}.pdf' for index in range(supplements.MAX_FILES + 1)]
    result = supplements.discover_supplements({'supplementary_urls': urls}, sources=['structured'])
    assert [candidate.url for candidate in result.candidates] == urls[:supplements.MAX_FILES]
    assert result.errors == [f'supplement discovery exceeded {supplements.MAX_FILES} files; result is incomplete']


def test_oversized_repository_file_is_refused_before_request() -> None:
    """A declared size above the cap is refused without downloading anything."""
    session = StreamSession([])
    candidate = supplements.SupplementCandidate('https://figshare.com/ndownloader/files/1', 'figshare', 'big.zip',
                                                metadata={'expected_size': supplements.MAX_FILE_BYTES + 1})
    with pytest.raises(ValueError, match='byte limit'):
        supplements.download_supplement(candidate, session)
    assert session.calls == []


@pytest.mark.parametrize('filename,content,metadata,message', [
    ('s.csv', b'', {}, 'empty supplementary file'),
    ('s.zip', b'%PDF-1.7', {}, 'invalid supplementary ZIP signature'),
    ('s.xlsx', b'not a workbook', {}, 'invalid supplementary ZIP signature'),
    ('s.csv', b'a,b\n1,2\n', {'expected_md5': '0' * 32}, 'checksum does not match'),
], ids=['empty', 'zip-not-zip', 'xlsx-not-zip', 'md5-mismatch'])
def test_empty_or_damaged_files_are_refused(filename: str, content: bytes, metadata: dict[str, Any], message: str) -> None:
    """Empty bodies, wrong archive signatures, and checksum mismatches are not stored."""
    candidate = supplements.SupplementCandidate(f'https://host.example/{filename}', 'figshare', filename, metadata=metadata)
    with pytest.raises(RuntimeError, match=message):
        supplements.download_supplement(candidate, StreamSession([StreamResponse(content)]))


def test_pdf_without_extension_is_named_and_typed_as_pdf() -> None:
    """A repository download URL without a suffix still yields a .pdf name."""
    session = StreamSession([StreamResponse(b'%PDF-1.7 data', headers={'Content-Type': 'application/octet-stream'})])
    downloaded = supplements.download_supplement(
        supplements.SupplementCandidate('https://figshare.com/ndownloader/files/12345', 'figshare'), session)
    assert downloaded.filename == '12345.pdf'
    assert downloaded.mime_type == 'application/pdf'
    assert downloaded.source_url == 'https://figshare.com/ndownloader/files/12345'


@pytest.mark.parametrize('filename,headers,mime', [
    ('notes.txt', {}, 'text/plain'),
    ('results.pmtdata', {'Content-Type': 'binary/octet-stream'}, 'application/octet-stream'),
    ('table.pmtdata', {'Content-Type': 'Text/Tab-Separated-Values; charset=utf-8'}, 'text/tab-separated-values'),
], ids=['guessed', 'unknown-extension', 'declared'])
def test_generic_or_missing_content_type_is_guessed_from_filename(filename: str, headers: dict[str, str], mime: str) -> None:
    """Generic binary types defer to the filename; a specific declared type is kept."""
    candidate = supplements.SupplementCandidate(f'https://host.example/{filename}', 'publisher', filename)
    downloaded = supplements.download_supplement(candidate, StreamSession([StreamResponse(b'col\nvalue\n', headers=headers)]))
    assert downloaded.filename == filename
    assert downloaded.mime_type == mime
