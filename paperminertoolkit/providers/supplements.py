"""Discover explicitly linked supporting information and fetch bounded public files.

Discovery uses article XML, PMC's versioned cloud objects, Figshare records
linked to the exact parent DOI, and explicit links on publisher article pages.
No archive is unpacked here and no supplementary filename is guessed.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import mimetypes
import re
import socket
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from email.message import Message
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urldefrag, urljoin, urlsplit

import requests

from paperminertoolkit.providers import base, elsevier, pubmed, registry, rxiv

MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_FILES = 100
MAX_REDIRECTS = 5
FIGSHARE_URL = 'https://api.figshare.com/v2'
LIMITER = base.RateLimiter(1.0)
_SUPPLEMENT_WORDS = re.compile(r'supplement(?:ary|al)?|supporting[\s_-]*(?:information|material|data)', re.I)
_FILE_EXTENSION = re.compile(r'\.(?:pdf|zip|gz|tgz|tar|docx?|xlsx?|pptx?|csv|tsv|txt|xml|json|cif|mol|sdf|xyz|mp4|avi|mov|png|jpe?g|tiff?)(?:$|[?#])', re.I)
_SUPPLEMENT_TAGS = {'inline-supplementary-material', 'supplementary-material', 'supplemental-material', 'supplement', 'supplementary-file', 'supplemental-file'}


@dataclass
class SupplementCandidate:
    """An explicit supporting file reference and its acquisition provenance."""

    url: str
    source: str
    filename: str = ''
    label: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Discovered files plus errors that prevent claiming discovery is complete."""

    candidates: list[SupplementCandidate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class DownloadedSupplement:
    """Validated file bytes with the final response URL and safe display name."""

    content: bytes
    filename: str
    mime_type: str
    source_url: str


def canonical_url(url: str) -> str:
    """Strip an irrelevant URL fragment without altering signed query strings."""
    return urldefrag(str(url).strip())[0]


def _validate_url(url: str) -> str:
    """Reject non-web, credential-bearing, and non-public network destinations."""
    url = canonical_url(url)
    try:
        parts = urlsplit(url)
        port = parts.port or (443 if parts.scheme == 'https' else 80)
    except ValueError as error:
        raise ValueError('invalid supplementary URL') from error
    host = (parts.hostname or '').rstrip('.').lower()
    if (parts.scheme not in {'http', 'https'} or not host or parts.username is not None
            or parts.password is not None or any(ord(char) < 32 for char in url)
            or host == 'localhost' or host.endswith(('.localhost', '.local', '.internal'))):
        raise ValueError('supplementary URLs must address a public HTTP(S) host')
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
    except OSError as error:
        raise RuntimeError(f'cannot resolve supplementary host {host}: {error}') from error
    if not addresses or any(not ipaddress.ip_address(value.split('%')[0]).is_global for value in addresses):
        raise ValueError(f'non-public supplementary host rejected: {host}')
    return url


def safe_filename(value: str, fallback: str = 'supplement.bin') -> str:
    """Remove path traversal, control characters, and platform-reserved names."""
    value = unquote(str(value)).replace('\\', '/').rsplit('/', 1)[-1]
    value = re.sub(r'[^\w.() -]', '_', value, flags=re.UNICODE).strip(' .')
    if not value or value.split('.', 1)[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}:
        value = fallback
    if len(value) > 180:
        suffix = PurePosixPath(value).suffix[:20]
        value = value[:180 - len(suffix)] + suffix
    return value


def _headers(url: str) -> dict[str, str]:
    """Attach Elsevier credentials only to that provider's object endpoint."""
    headers = {'User-Agent': base.user_agent(), 'Accept': '*/*'}
    parts = urlsplit(url)
    if parts.scheme == 'https' and parts.hostname == 'api.elsevier.com' and parts.path.startswith('/content/object/'):
        key = elsevier.configured_api_key()
        if key:
            headers.update(elsevier.api_headers(key, accept='*/*'))
    return headers



def supplement_limiter(url: str) -> base.RateLimiter:
    """Share each provider's pacing window across article and SI requests.

    Parameters
    ----------
    url : str
        URL about to be requested, including redirect targets.

    Returns
    -------
    base.RateLimiter
        Existing provider limiter or the general one-request-per-second window.
    """
    host = (urlsplit(url).hostname or '').lower().rstrip('.')
    if host in {'biorxiv.org', 'www.biorxiv.org', 'medrxiv.org', 'www.medrxiv.org'}:
        return rxiv.CONTENT_LIMITER
    if host == 'pmc-oa-opendata.s3.amazonaws.com':
        return pubmed.CLOUD_LIMITER
    if host == 'api.elsevier.com':
        return elsevier.limiter_for(url)
    return LIMITER

def _read_url(
    url: str,
    *,
    session: Any = None,
    max_bytes: int = MAX_METADATA_BYTES,
    json_body: Mapping[str, Any] | None = None,
) -> tuple[bytes, Mapping[str, str], str]:
    """Read a bounded response, checking every redirect and retrying transient errors.

    A private destination is refused before any request is sent. Each response
    stream closes even on a size error; retries restart the whole bounded read.
    """
    client = session if session is not None else requests.Session()
    if session is None:
        client.trust_env = False
    try:
        for attempt in range(3):
            current = url
            body = json_body
            try:
                for hop in range(MAX_REDIRECTS + 1):
                    current = _validate_url(current)
                    headers = _headers(current)
                    is_elsevier = urlsplit(current).hostname == 'api.elsevier.com'
                    api_key = headers.get('X-ELS-APIKey', '')
                    if is_elsevier:
                        elsevier.check_quota(api_key, current)
                    supplement_limiter(current).wait()
                    response = client.request(
                        'POST' if body is not None else 'GET', current,
                        json=body, headers=headers, timeout=(15, 60),
                        stream=True, allow_redirects=False,
                    )
                    try:
                        if is_elsevier:
                            elsevier.record_quota(response, api_key, current)
                        status = response.status_code
                        if status in {301, 302, 303, 307, 308}:
                            if hop == MAX_REDIRECTS or not response.headers.get('Location'):
                                raise RuntimeError('supplementary redirect limit exceeded or missing location')
                            current = urljoin(current, response.headers['Location'])
                            if status in {301, 302, 303}:
                                body = None
                            continue
                        if status in {408, 429, 500, 502, 503, 504} and attempt < 2:
                            retry = response.headers.get('Retry-After', '')
                            time.sleep(min(float(retry), 30) if retry.isdigit() else 2 ** attempt)
                            break
                        if status >= 400:
                            raise RuntimeError(f'HTTP {status} from {current}')
                        if status != 200:
                            raise RuntimeError(f'unexpected HTTP {status} from {current}')
                        length = response.headers.get('Content-Length', '')
                        if length.isdigit() and int(length) > max_bytes:
                            raise ValueError(f'supplementary response exceeds {max_bytes} byte limit')
                        content = bytearray()
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                content.extend(chunk)
                                if len(content) > max_bytes:
                                    raise ValueError(f'supplementary response exceeds {max_bytes} byte limit')
                        if length.isdigit() and not response.headers.get('Content-Encoding') and len(content) != int(length):
                            raise requests.ConnectionError('incomplete supplementary response')
                        return bytes(content), requests.structures.CaseInsensitiveDict(response.headers), current
                    finally:
                        response.close()
            except requests.RequestException as error:
                if attempt == 2:
                    raise RuntimeError(f'supplementary request failed: {error}') from error
                time.sleep(2 ** attempt)
        raise RuntimeError('supplementary request retries exhausted')
    finally:
        if session is None:
            client.close()


def _json_url(url: str, session: Any = None, body: Mapping[str, Any] | None = None) -> Any:
    """Read a size-limited JSON response and report malformed metadata."""
    content, _, _ = _read_url(url, session=session, json_body=body)
    try:
        return json.loads(content)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError(f'malformed supplementary JSON from {url}') from error


def _local(value: str) -> str:
    """Return a namespace-free XML element or attribute name."""
    return value.rsplit('}', 1)[-1].lower()


def _attribute(element: ET.Element, *names: str) -> str:
    """Read the first named XML attribute regardless of namespace."""
    for name in names:
        for key, value in element.attrib.items():
            if _local(key) == name:
                return value.strip()
    return ''


def _xml_candidates(
    content: str | bytes,
    source_url: str,
    source: str,
    files: Mapping[str, str] | None = None,
    unavailable: list[str] | None = None,
) -> list[SupplementCandidate]:
    """Resolve only explicit supplementary XML elements, excluding article graphics."""
    if len(content) > MAX_METADATA_BYTES:
        raise ValueError('supplementary XML exceeds metadata size limit')
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise RuntimeError(f'malformed supplementary XML: {error}') from error
    objects = {_attribute(node, 'id', 'ref'): node for node in root.iter()
               if _local(node.tag) in {'object', 'attachment', 'e-component'}}
    eid = next((''.join(node.itertext()).strip() for node in root.iter() if _local(node.tag) == 'eid'), '')
    provenance: dict[str, Any] = {}
    article_meta = next((node for node in root.iter() if _local(node.tag) == 'article-meta'), None)
    if article_meta is not None:
        identifiers = {_attribute(node, 'pub-id-type'): ''.join(node.itertext()).strip()
                       for node in article_meta if _local(node.tag) == 'article-id'}
        if identifiers:
            provenance['article_identifiers'] = identifiers
        license_node = next((node for node in article_meta.iter() if _local(node.tag) == 'license'), None)
        if license_node is not None:
            provenance['license'] = {'url': _attribute(license_node, 'href'),
                                     'text': ' '.join(' '.join(license_node.itertext()).split())[:2000]}
    candidates = []
    for element in root.iter():
        tag = _local(element.tag)
        explicit = tag in _SUPPLEMENT_TAGS or (
            tag in {'media', 'object', 'ext-link', 'attachment', 'e-component'} and _SUPPLEMENT_WORDS.search(
                ' '.join(element.attrib.values()) + ' ' + ' '.join(element.itertext())))
        if not explicit:
            continue
        label = ' '.join(' '.join(element.itertext()).split())[:500]
        for node in element.iter():
            reference = _attribute(node, 'href', 'url', 'src', 'locator')
            if not reference or reference.startswith('#'):
                continue
            locator = _attribute(node, 'locator')
            linked = objects.get(locator or reference)
            if linked is not None:
                reference = next((_attribute(child, 'href', 'url') for child in linked.iter()
                                  if _attribute(child, 'href', 'url')), reference)
            parsed = urlsplit(reference)
            if parsed.scheme and parsed.scheme not in {'http', 'https', 'pii'}:
                continue
            filename = unquote(PurePosixPath(parsed.path).name)
            if files is not None:
                # PMC packages may omit unreleased/embargoed SI. A cloud key
                # must match the XML reference; figures and the main PDF are
                # never inferred to be SI merely because they share a prefix.
                match = files.get(reference) or files.get(filename)
                if match:
                    url = match
                elif parsed.scheme in {'http', 'https'}:
                    url = reference
                else:
                    if unavailable is not None:
                        unavailable.append(f'PMC does not distribute referenced supplementary file: {reference}')
                    continue
            elif reference.startswith('pii:') or (locator and '/' not in reference and '.' not in reference):
                if not eid:
                    continue
                token = locator or reference.rsplit('/', 1)[-1]
                url = f'https://api.elsevier.com/content/object/eid/{quote(eid, safe="")}-{quote(token, safe="")}'
            else:
                if not parsed.scheme and (not source_url or 'eutils.ncbi.nlm.nih.gov' in urlsplit(source_url).netloc):
                    continue
                url = urljoin(source_url, reference)
            if urlsplit(url).scheme not in {'http', 'https'}:
                continue
            candidates.append(SupplementCandidate(
                canonical_url(url), source, filename, label,
                {**provenance, 'discovery_url': source_url, 'reference': reference},
            ))
    return _unique(candidates)


def _unique(candidates: Iterable[SupplementCandidate]) -> list[SupplementCandidate]:
    """Deduplicate explicit URLs while preserving provider preference order."""
    return list({canonical_url(candidate.url): candidate for candidate in reversed(list(candidates))}.values())[::-1]


def _supplied_urls(value: Any) -> list[Mapping[str, Any]]:
    """Normalize explicitly named supplementary URL metadata from imports."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith(('[', '{')):
            try:
                return _supplied_urls(json.loads(value))
            except ValueError:
                return []
        return [{'url': value}] if value else []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _supplied_urls(child)]
    return []


def discover_structured(
    paper: Mapping[str, Any], documents: Iterable[Mapping[str, Any]] = (), session: Any = None,
) -> DiscoveryResult:
    """Find explicit supplementary URLs in stored XML and imported metadata.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Paper metadata, optionally with ``supplementary_urls``,
        ``supplement_urls``, or ``supplementary_url`` values.
    documents : iterable of mappings, optional
        Stored original XML assets with content and source URL metadata.
    session : object or None, optional
        Accepted for the common discovery interface; unused.

    Returns
    -------
    DiscoveryResult
        Explicit file references and any malformed-document errors.
    """
    result = DiscoveryResult()
    for document in documents:
        metadata = document.get('metadata') or {}
        if metadata.get('document_format') not in {'jats', 'elsevier', 'elsevier-xml', 'xml'}:
            continue
        try:
            candidates = _xml_candidates(
                document.get('content') or b'', metadata.get('source_url', ''), 'structured',
            )
            for candidate in candidates:
                for key in ('source_identifier', 'document_format', 'license', 'version'):
                    if metadata.get(key):
                        candidate.metadata.setdefault(key, metadata[key])
                candidate.metadata['document_source'] = document.get('source', '')
            result.candidates.extend(candidates)
        except (RuntimeError, ValueError) as error:
            result.errors.append(f'structured: {error}')
    for field_name in ('supplementary_urls', 'supplement_urls', 'supplementary_url', 'supplement_url'):
        for item in _supplied_urls(paper.get(field_name)):
            url = str(item.get('url') or item.get('href') or '').strip()
            if url:
                result.candidates.append(SupplementCandidate(
                    canonical_url(url), 'structured', str(item.get('filename') or ''),
                    str(item.get('label') or ''), {'discovery_field': field_name},
                ))
    result.candidates = _unique(result.candidates)
    return result


def discover_pubmed(
    paper: Mapping[str, Any], documents: Iterable[Mapping[str, Any]] = (), session: Any = None,
) -> DiscoveryResult:
    """Resolve a paper's newest PMC cloud JATS and collect its supplementary links.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Paper metadata with PMCID, PMID, or DOI.
    documents : iterable of mappings, optional
        Available for the common discovery interface; unused.
    session : object or None, optional
        HTTP client for metadata and file requests.

    Returns
    -------
    DiscoveryResult
        Explicit references offered by the article's own cloud version.
    """
    result = DiscoveryResult()
    try:
        pmcid = pubmed.resolve_pmcid(paper, api_key=pubmed.configured_api_key(),
                                   email=pubmed.configured_email(), session=session)
        if not pmcid:
            return result
        files = pubmed.pmc_cloud_files(pmcid, session=session)
        xml_url = next((url for name, url in files.items()
                        if re.fullmatch(rf'{re.escape(pmcid)}\.\d+\.xml', name, re.I)), '')
        if not xml_url:
            return result
        content, _, _ = _read_url(xml_url, session=session)
        result.candidates = _xml_candidates(content, xml_url, 'pubmed', files, result.notes)
        for candidate in result.candidates:
            candidate.metadata['pmcid'] = pmcid
            candidate.metadata['pmc_version'] = PurePosixPath(urlsplit(xml_url).path).parent.name
    except (RuntimeError, ValueError, requests.RequestException) as error:
        result.errors.append(f'pubmed: {error}')
    return result


def _doi(value: Any) -> str:
    """Normalize DOI prefixes and case while retaining the complete suffix."""
    value = unquote(str(value or '').strip()).lower()
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', value).strip()


def _related_doi(article: Mapping[str, Any], doi: str) -> bool:
    """Require an exact parent DOI association, never just a title search hit."""
    if _doi(article.get('resource_doi')) == doi:
        return True
    for material in article.get('related_materials') or []:
        if not isinstance(material, Mapping):
            continue
        relation = str(material.get('relation') or material.get('relation_type') or '').lower()
        if relation in {'issupplementto', 'ispartof', 'isdescribedby'} and _doi(material.get('identifier') or material.get('link')) == doi:
            return True
    # ACS uses the parent DOI plus .sNNN for SI item DOIs. Match the full
    # suffix, not a shared prefix such as jacs.4c123 versus jacs.4c12345.
    return doi.startswith('10.1021/') and bool(re.fullmatch(re.escape(doi) + r'\.s\d+', _doi(article.get('doi'))))


def discover_figshare(
    paper: Mapping[str, Any], documents: Iterable[Mapping[str, Any]] = (), session: Any = None,
) -> DiscoveryResult:
    """Find public Figshare files whose metadata identifies the exact parent DOI.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Paper metadata containing a DOI.
    documents : iterable of mappings, optional
        Accepted for the common interface; unused.
    session : object or None, optional
        HTTP client for public Figshare metadata.

    Returns
    -------
    DiscoveryResult
        Public files with parent DOI, item DOI, license, and checksum metadata.
    """
    result = DiscoveryResult()
    doi = _doi(paper.get('doi'))
    if not doi:
        return result
    try:
        # A quoted exact DOI search also finds modern related_materials
        # records, which replaced resource_doi. Each detail is checked below.
        seen: set[str] = set()
        for page in range(1, 6):
            entries = _json_url(f'{FIGSHARE_URL}/articles/search', session,
                                {'search_for': f'"{doi}"', 'page': page, 'page_size': 100})
            if not isinstance(entries, list):
                raise RuntimeError('Figshare search returned a non-list response')
            for entry in entries:
                if not isinstance(entry, Mapping) or not str(entry.get('id', '')).isdigit():
                    continue
                identifier = str(entry['id'])
                if identifier in seen:
                    continue
                seen.add(identifier)
                try:
                    article = _json_url(f'{FIGSHARE_URL}/articles/{identifier}', session)
                    if not isinstance(article, Mapping):
                        raise RuntimeError('Figshare article returned a non-object response')
                    if not _related_doi(article, doi):
                        continue
                    for file in article.get('files') or []:
                        if not isinstance(file, Mapping) or file.get('is_link_only') or not file.get('download_url'):
                            continue
                        result.candidates.append(SupplementCandidate(
                            str(file['download_url']), 'figshare', str(file.get('name') or ''),
                            str(article.get('title') or ''),
                            {'figshare_article_id': identifier, 'figshare_file_id': file.get('id'),
                             'related_doi': doi, 'item_doi': article.get('doi', ''),
                             'license': article.get('license'), 'expected_size': file.get('size'),
                             'expected_md5': file.get('computed_md5') or file.get('supplied_md5'),
                             'discovery_url': f'{FIGSHARE_URL}/articles/{identifier}'},
                        ))
                except (RuntimeError, ValueError) as error:
                    result.errors.append(f'figshare item {identifier}: {error}')
            if len(entries) < 100:
                break
        else:
            result.errors.append('figshare: search exceeded 500 results; discovery is incomplete')
    except (RuntimeError, ValueError) as error:
        result.errors.append(f'figshare: {error}')
    result.candidates = _unique(result.candidates)
    return result


class _SupplementLinks(HTMLParser):
    """Collect links explicitly labelled as supplementary files in article HTML."""

    def __init__(self) -> None:
        """Initialize the anchor stack and detected references."""
        super().__init__(convert_charrefs=True)
        self.anchor: dict[str, str] | None = None
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Start an anchor or read an explicit supplementary citation meta tag."""
        values = {key: value or '' for key, value in attrs}
        if tag == 'a':
            self.anchor = {'url': values.get('href', ''), 'label': '',
                           'hint': ' '.join(values.get(key, '') for key in ('title', 'class', 'id', 'data-test'))}
        elif tag == 'meta' and values.get('name', '').lower() in {'citation_supplementary_material_url', 'citation_supplement_url'}:
            self.links.append((values.get('content', ''), 'Supplementary material'))

    def handle_data(self, data: str) -> None:
        """Accumulate the visible anchor label, including nested markup."""
        if self.anchor is not None:
            self.anchor['label'] += data

    def handle_endtag(self, tag: str) -> None:
        """Keep direct file anchors carrying explicit supplementary context."""
        if tag != 'a' or self.anchor is None:
            return
        anchor = self.anchor
        self.anchor = None
        url = anchor['url']
        text = ' '.join((anchor['label'], anchor['hint'], unquote(url)))
        file_link = _FILE_EXTENSION.search(url) or re.search(r'/(?:suppl_file|suppdata|attachment|ndownloader/files)/', url, re.I)
        explicit_route = re.search(r'/(?:suppl_file|suppdata|supplementary)/', url, re.I)
        if (_SUPPLEMENT_WORDS.search(text) or explicit_route) and file_link:
            self.links.append((url, ' '.join(anchor['label'].split())))


def discover_publisher(
    paper: Mapping[str, Any], documents: Iterable[Mapping[str, Any]] = (), session: Any = None,
) -> DiscoveryResult:
    """Read a publisher landing page and retain explicitly labelled SI file links.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Paper metadata with DOI or an article landing-page URL.
    documents : iterable of mappings, optional
        Accepted for the common interface; unused.
    session : object or None, optional
        HTTP client for the article page.

    Returns
    -------
    DiscoveryResult
        Explicit supplementary links, or an error when the page is refused.
    """
    result = DiscoveryResult()
    doi = _doi(paper.get('doi'))
    url = next((str(paper[key]) for key in ('landing_page_url', 'article_url', 'url') if paper.get(key)), '')
    if not url and doi:
        url = f'https://doi.org/{quote(doi, safe="/()")}'
    if not url:
        return result
    try:
        content, headers, final_url = _read_url(url, session=session)
        if not content.strip():
            raise RuntimeError('empty publisher landing page')
        if 'html' not in headers.get('Content-Type', '').lower() and not re.search(br'<(?:html|head|body)\b', content[:4096], re.I):
            return result
        if re.search(br'captcha|cf-chl-|verify (?:that )?you are human|checking your browser', content[:131072], re.I):
            raise RuntimeError('publisher returned a browser or access challenge')
        parser = _SupplementLinks()
        parser.feed(content.decode('utf-8', errors='replace'))
        for reference, label in parser.links:
            if reference and not reference.startswith('#'):
                resolved = urljoin(final_url, reference)
                result.candidates.append(SupplementCandidate(
                    canonical_url(resolved), 'publisher', unquote(PurePosixPath(urlsplit(resolved).path).name),
                    label, {'discovery_url': final_url},
                ))
    except (RuntimeError, ValueError) as error:
        result.errors.append(f'publisher: {error}')
    result.candidates = _unique(result.candidates)
    return result


def discover_supplements(
    paper: Mapping[str, Any], documents: Iterable[Mapping[str, Any]] = (),
    sources: Iterable[str] | None = None, session: Any = None,
) -> DiscoveryResult:
    """Combine selected discovery sources, using publisher HTML as a final fallback.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Owning paper metadata.
    documents : iterable of mappings, optional
        Stored structured documents belonging to the paper.
    sources : iterable of str or None, optional
        Selected source names, defaulting to every supported source.
    session : object or None, optional
        HTTP client used by the discovery providers.

    Returns
    -------
    DiscoveryResult
        Unique explicit file candidates and independent discovery errors.
    """
    names = registry.names(registry.SUPPLEMENT)
    selected = list(sources) if sources is not None else list(names)
    unknown = set(selected) - set(names)
    if unknown:
        raise ValueError(f'unknown supplementary sources: {", ".join(sorted(unknown))}')
    result = DiscoveryResult()
    documents = list(documents)
    for name in names:
        if name not in selected or (name == 'publisher' and result.candidates):
            continue
        handler = registry.resolve_handler(name, registry.SUPPLEMENT)
        found = handler(paper, documents=documents, session=session)
        result.candidates.extend(found.candidates)
        result.errors.extend(found.errors)
        result.notes.extend(found.notes)
    result.candidates = _unique(result.candidates)
    if len(result.candidates) > MAX_FILES:
        result.candidates = result.candidates[:MAX_FILES]
        result.errors.append(f'supplement discovery exceeded {MAX_FILES} files; result is incomplete')
    return result


def download_supplement(candidate: SupplementCandidate, session: Any = None) -> DownloadedSupplement:
    """Fetch one bounded file and refuse HTML errors, truncation, or invalid PDFs.

    Parameters
    ----------
    candidate : SupplementCandidate
        Explicit file reference and optional expected size/checksum.
    session : object or None, optional
        HTTP client exposing ``request`` and streaming responses.

    Returns
    -------
    DownloadedSupplement
        Original bytes, safe name, MIME type, and final URL.

    Raises
    ------
    RuntimeError
        If the request fails or the response is an error page or damaged file.
    ValueError
        If a destination is unsafe or the response exceeds the size limit.
    """
    expected = candidate.metadata.get('expected_size')
    if expected is not None and str(expected).isdigit() and int(expected) > MAX_FILE_BYTES:
        raise ValueError(f'supplement exceeds {MAX_FILE_BYTES} byte limit')
    content, headers, final_url = _read_url(candidate.url, session=session, max_bytes=MAX_FILE_BYTES)
    if not content:
        raise RuntimeError('empty supplementary file')
    mime = headers.get('Content-Type', '').split(';')[0].strip().lower()
    prefix = content.lstrip()[:4096].lower()
    if 'html' in mime or re.search(br'<(?:!doctype\s+html|html|head|body)\b', prefix):
        raise RuntimeError('HTML response received instead of a supplementary file')
    disposition = Message()
    disposition['content-disposition'] = headers.get('Content-Disposition', '')
    filename = safe_filename(disposition.get_filename() or candidate.filename or PurePosixPath(urlsplit(final_url).path).name)
    if filename.lower().endswith('.pdf') and not content.startswith(b'%PDF'):
        raise RuntimeError('invalid supplementary PDF signature')
    if filename.lower().endswith(('.zip', '.docx', '.xlsx', '.pptx')) and not content.startswith(b'PK'):
        raise RuntimeError('invalid supplementary ZIP signature')
    if expected is not None and str(expected).isdigit() and len(content) != int(expected):
        raise RuntimeError('supplementary file size does not match repository metadata')
    md5 = candidate.metadata.get('expected_md5')
    if md5 and re.fullmatch(r'[a-fA-F0-9]{32}', str(md5)) and hashlib.md5(content, usedforsecurity=False).hexdigest() != str(md5).lower():
        raise RuntimeError('supplementary file checksum does not match repository metadata')
    if content.startswith(b'%PDF'):
        mime = 'application/pdf'
        if not PurePosixPath(filename).suffix:
            filename += '.pdf'
    elif not mime or mime in {'application/octet-stream', 'binary/octet-stream'}:
        mime = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    return DownloadedSupplement(content, filename, mime, final_url)
