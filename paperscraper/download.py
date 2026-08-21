"""Download paper text and PDFs from configured open-access and publisher sources.

This module powers ``ps_download``. It can fetch Elsevier full text when
available, try PDFs from Unpaywall, OpenAlex, CORE, and Elsevier, and update
per-paper download status in the SQLite paper corpus after each row.
"""

from __future__ import annotations

import ast
import html
import json
import os
import re
import requests
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from typing import Any, TypeAlias
from urllib.parse import quote

from paperscraper import elsevier, openalex
from paperscraper.corpus import (PIPELINE_COLUMNS, add_asset, connect,
                                get_asset_metadata, paper_rows, upsert_paper)
from paperscraper.settings import load_settings

DOWNLOAD_FORMATS = {'text', 'pdf', 'both'}
DOWNLOAD_SOURCES = {'unpaywall', 'core', 'elsevier', 'openalex'}
_Paper: TypeAlias = dict[str, Any]


def _elsevier_api_key() -> str:
    """Return the configured Elsevier API key."""
    api_key = load_settings().get('elsevier_api_key')
    if not api_key:
        raise ValueError('Elsevier API key is not configured. Run ps_elsevier_key first.')
    return api_key


def retrieve_document(uri: str) -> None:
    """Retrieve an Elsevier full-text document.

    The decoded response is written to ``data/elsevier_document.json`` after
    existing files in the directory are removed.

    Parameters
    ----------
    uri : str
        Elsevier full-text endpoint URL.

    Returns
    -------
    None
        The document is written to the local ``data`` directory. If the
        request fails, an error is printed and no document is written.

    Raises
    ------
    ValueError
        If no Elsevier API key is configured.
    OSError
        If the output directory or its files cannot be created, removed, or
        written.
    """
    os.makedirs('data', exist_ok=True)
    for file in os.listdir('data'):
        os.remove(os.path.join('data', file))
    try:
        response = elsevier.get_content(
            _elsevier_api_key(),
            uri,
            accept='application/json',
            params={'httpAccept': 'application/json'},
        )
    except requests.RequestException:
        print('Read document failed.')
        return
    with open(os.path.join('data', 'elsevier_document.json'), 'w', encoding='utf-8') as out_file:
        json.dump(response.json(), out_file)


def json_to_text(filepath: str | PathLike[str]) -> str:
    """Read original text from an Elsevier JSON document.

    Parameters
    ----------
    filepath : str or os.PathLike[str]
        Path to the downloaded JSON document.

    Returns
    -------
    str
        Original text, or ``'failed'`` when text is missing or structured.

    Raises
    ------
    OSError
        If the document cannot be read.
    json.JSONDecodeError
        If the document is not valid JSON.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    text = doc.get('originalText')
    if text is None:
        text = (doc.get('full-text-retrieval-response') or {}).get('originalText')
    if type(text) == dict:
        return 'failed'
    return text or 'failed'


def elsevier_string_formatter(text: str) -> str:
    """Clean wrapper artifacts from Elsevier original text.

    Parameters
    ----------
    text : str
        Raw ``originalText`` value.

    Returns
    -------
    str
        Cleaned article text.
    """
    if text.count('Acknowledgements') == 2:
        text = text.split('Acknowledgements')[1]
    elif text.count('References') == 2:
        text = text.split('References')[1]
    if 'amazonaws.com/' in text:
        text = text.split('amazonaws.com/')[-1]
        text = text[text.find(' '):]
    return text


def _full_text_uri(paper: Mapping[str, Any]) -> str | None:
    """Extract an Elsevier full-text URI or DOI retrieval URL from a paper row."""
    link = paper.get('elsevier_link')
    for value in _link_values(link):
        uri = _full_text_uri_from_link_value(value)
        if uri:
            return uri
    doi = paper.get('doi')
    if _has_value(doi):
        return elsevier.article_url_from_doi(str(doi))
    return None


def _link_values(link: object) -> list[Any]:
    """Return one or more Elsevier link values from strings, lists, or dictionaries."""
    if isinstance(link, list):
        return link
    if isinstance(link, dict):
        return [link]
    if not isinstance(link, str) or not link.strip():
        return []
    text = link.strip()
    if text[0] in '[{':
        for parser in (json.loads, ast.literal_eval):
            try:
                return _link_values(parser(text))
            except (ValueError, SyntaxError, TypeError):
                continue
    return [text]


def _full_text_uri_from_link_value(value: object) -> str | None:
    """Extract a full-text URI from one Elsevier link object or string."""
    if isinstance(value, dict):
        ref = str(value.get('@ref') or value.get('ref') or '').lower()
        href = value.get('@href') or value.get('href') or value.get('url') or ''
        if href and ('full-text' in ref or _is_elsevier_article_endpoint(str(href))):
            return str(href)
        return None
    if not isinstance(value, str):
        return None
    if _is_elsevier_article_endpoint(value):
        return value
    if 'full-text' not in value.lower():
        return None
    quoted = [part for part in re.findall(r"""['"]([^'"]+)['"]""", value) if part.lower() != 'full-text']
    if quoted:
        return quoted[-1]
    if value.startswith('http'):
        return value
    return None


def _is_elsevier_article_endpoint(value: str) -> bool:
    """Return whether a URL is an Elsevier article retrieval endpoint."""
    return 'api.elsevier.com/content/article/' in value.lower()


def _has_value(value: object) -> bool:
    """Return whether a paper field contains a meaningful non-empty value."""
    return value is not None and str(value).strip() != ''


def _set_status(paper: _Paper, column: str, status: str, error: str | None = None) -> None:
    """Update a corpus paper status field and optional error text."""
    if column not in PIPELINE_COLUMNS:
        raise KeyError(f'Unknown pipeline status column: {column}')
    paper[column] = status
    if error:
        paper['last_error'] = error
    elif status in {'succeeded', 'stored'}:
        paper['last_error'] = ''


def _download_text(paper: Mapping[str, Any], filepath: str | PathLike[str]) -> bool:
    """Download Elsevier full text for one paper row to ``filepath``."""
    uri = _full_text_uri(paper)
    if not uri:
        return False
    retrieve_document(uri)
    files = os.listdir('data')
    if len(files) < 1:
        return False
    temp_file = os.path.join('data', files[0])
    text = json_to_text(temp_file)
    if text == 'failed':
        return False
    formatted_text = elsevier_string_formatter(text)
    with open(filepath, 'w', encoding='utf-8') as out_file:
        out_file.write(formatted_text)
    return True


def _pdf_urls(paper: Mapping[str, Any]) -> list[str]:
    """Build Elsevier PDF endpoint candidates for a normalized paper row."""
    urls = []
    doi = paper.get('doi')
    if _has_value(doi):
        urls.append(elsevier.article_url_from_doi(str(doi)))
    uri = _full_text_uri(paper)
    if uri:
        urls.append(uri)
    return urls


def _download_pdf(paper: Mapping[str, Any], filepath: str | PathLike[str]) -> bool:
    """Try to download an Elsevier PDF for one paper row."""
    api_key = _elsevier_api_key()
    params = {'httpAccept': 'application/pdf'}
    last_error = None
    for url in _pdf_urls(paper):
        try:
            response = elsevier.get_content(api_key, url, accept='application/pdf', params=params)
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type or response.content.startswith(b'%PDF'):
                with open(filepath, 'wb') as out_file:
                    out_file.write(response.content)
                return True
            last_error = f'non-PDF response from {url}'
        except requests.HTTPError as e:
            response = getattr(e, 'response', None)
            status_code = response.status_code if response is not None else 'HTTP error'
            last_error = f'{status_code} from {url}'
        except requests.RequestException as e:
            last_error = str(e)
    if last_error:
        print(f'PDF download failed for {paper.get("paper_id")}: {last_error}')
    return False


def _safe_filename(paper: Mapping[str, Any]) -> str:
    """Create a filesystem-safe filename stem for a paper row."""
    for column in ['doi', 'core_id', 'paper_id']:
        value = paper.get(column)
        if _has_value(value):
            safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())
            safe = safe.strip('._')
            if safe:
                return safe
    return 'paper'


def _unpaywall_email(settings: Mapping[str, str] | None = None) -> str | None:
    """Return the configured email address used for Unpaywall requests."""
    settings = settings or load_settings()
    return settings.get('unpaywall_email') or os.environ.get('UNPAYWALL_EMAIL')


def _download_url_to_pdf(
    url: str | None,
    filepath: str | PathLike[str],
    headers: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Fetch a URL and save it only when the response appears to be a PDF."""
    if not url:
        return False, 'missing URL'
    try:
        response = requests.get(url, headers=headers or {}, timeout=60, allow_redirects=True)
        if response.status_code >= 400:
            return False, f'{response.status_code} from {url}'
        content_type = response.headers.get('Content-Type', '').lower()
        if 'pdf' not in content_type and not response.content.startswith(b'%PDF'):
            return False, f'non-PDF response from {url}'
        with open(filepath, 'wb') as out_file:
            out_file.write(response.content)
        return True, ''
    except requests.RequestException as e:
        return False, str(e)


def _download_unpaywall_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Use Unpaywall metadata to locate and download an open-access PDF."""
    doi = paper.get('doi')
    if not _has_value(doi):
        return False, 'missing DOI'
    api_url = f'https://api.unpaywall.org/v2/{quote(str(doi).strip(), safe="")}'
    try:
        email = _unpaywall_email()
        if not email:
            return False, 'Unpaywall email is not configured. Run ps_unpaywall_email first.'
        response = requests.get(api_url, params={'email': email}, timeout=60)
        if response.status_code >= 400:
            return False, f'{response.status_code} from Unpaywall'
        metadata = response.json()
    except requests.RequestException as e:
        return False, str(e)
    candidates = []
    best = metadata.get('best_oa_location') or {}
    candidates.append(best.get('url_for_pdf'))
    for location in metadata.get('oa_locations') or []:
        candidates.append(location.get('url_for_pdf'))
    for url in dict.fromkeys(url for url in candidates if url):
        ok, error = _download_url_to_pdf(url, filepath)
        if ok:
            return True, url
        last_error = error
    return False, locals().get('last_error', 'no Unpaywall PDF URL found')


def _core_headers() -> dict[str, str]:
    """Build request headers for CORE downloads."""
    settings = load_settings()
    api_key = settings.get('core_api_key') or os.environ.get('CORE_API_KEY')
    headers = {'User-Agent': 'PaperScraper/0.0.1'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


def _download_core_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Download a PDF through CORE using a stored PDF URL or CORE work ID."""
    urls = []
    url = paper.get('pdf_url')
    if _has_value(url):
        urls.append(str(url).strip())
    core_id = paper.get('core_id')
    if _has_value(core_id):
        urls.append(f'https://api.core.ac.uk/v3/works/{quote(str(core_id).strip(), safe="")}/download')
    last_error = 'no CORE download URL found'
    for candidate in dict.fromkeys(urls):
        ok, error = _download_url_to_pdf(candidate, filepath, headers=_core_headers())
        if ok:
            return True, candidate
        last_error = error
    return False, last_error


def _openalex_identifier(paper: Mapping[str, Any]) -> str | None:
    """Return the DOI or OpenAlex work identifier usable for a work lookup."""
    doi = paper.get('doi')
    paper_id = str(paper.get('paper_id') or '')
    if _has_value(doi):
        return f'doi:{str(doi).strip()}'
    if paper_id.startswith('openalex:'):
        return paper_id.split(':', 1)[1]
    return None


def _download_openalex_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Use OpenAlex metadata to locate and download an open-access PDF."""
    identifier = _openalex_identifier(paper)
    if identifier is None:
        return False, 'missing DOI'
    try:
        work = openalex.get_work(identifier, api_key=openalex.configured_api_key())
    except RuntimeError as e:
        return False, str(e)
    if not work:
        return False, f'no OpenAlex work found for {identifier}'
    last_error = 'no OpenAlex PDF URL found'
    for url in openalex.pdf_candidates(work):
        ok, error = _download_url_to_pdf(url, filepath)
        if ok:
            return True, url
        last_error = error
    return False, last_error


def _download_openalex_abstract(
    paper: Mapping[str, Any],
) -> tuple[bool, str, str]:
    """Fetch and reconstruct an abstract from OpenAlex work metadata."""
    identifier = _openalex_identifier(paper)
    if identifier is None:
        return False, 'missing DOI or OpenAlex ID', ''
    try:
        work = openalex.get_work(identifier, api_key=openalex.configured_api_key())
    except RuntimeError as error:
        return False, str(error), ''
    if not work:
        return False, f'no OpenAlex work found for {identifier}', ''
    abstract = _clean_abstract(
        openalex.reconstruct_abstract(work.get('abstract_inverted_index'))
    )
    if abstract:
        return True, 'openalex', abstract
    return False, 'no OpenAlex abstract found', ''


def _clean_abstract(value: object) -> str:
    """Normalize provider abstract text to compact plain text."""
    if not _has_value(value):
        return ''
    if isinstance(value, list):
        value = ' '.join(str(part) for part in value if _has_value(part))
    text = html.unescape(str(value))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _abstract_from_mapping(value: object) -> str:
    """Return the first abstract-like text from a nested provider mapping."""
    if isinstance(value, dict):
        for key in ['abstract', 'dc:description', 'description', 'dcDescription']:
            abstract = _clean_abstract(value.get(key))
            if abstract:
                return abstract
        for child in value.values():
            abstract = _abstract_from_mapping(child)
            if abstract:
                return abstract
    elif isinstance(value, list):
        for child in value:
            abstract = _abstract_from_mapping(child)
            if abstract:
                return abstract
    return ''


def _core_work_url(paper: Mapping[str, Any]) -> str | None:
    """Return a CORE work metadata URL for a paper row when possible."""
    core_id = paper.get('core_id')
    if not _has_value(core_id):
        return None
    return f'https://api.core.ac.uk/v3/works/{quote(str(core_id).strip(), safe="")}'


def _download_core_abstract(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
    """Fetch and return a CORE abstract for one paper row."""
    url = _core_work_url(paper)
    if not url:
        return False, 'missing CORE ID', ''
    try:
        response = requests.get(url, headers=_core_headers(), timeout=60)
        if response.status_code >= 400:
            return False, f'{response.status_code} from CORE', ''
        abstract = _abstract_from_mapping(response.json())
    except requests.RequestException as e:
        return False, str(e), ''
    if abstract:
        return True, 'core', abstract
    return False, 'no CORE abstract found', ''


def _elsevier_abstract_urls(paper: Mapping[str, Any]) -> list[str]:
    """Build Elsevier abstract endpoint candidates for a paper row."""
    urls = []
    link = paper.get('elsevier_link')
    for value in _link_values(link):
        if isinstance(value, dict):
            href = value.get('@href') or value.get('href') or value.get('url') or ''
            ref = str(value.get('@ref') or value.get('ref') or '').lower()
            if href and ('abstract' in ref or '/content/abstract/' in str(href).lower()):
                urls.append(str(href))
        elif isinstance(value, str) and '/content/abstract/' in value.lower():
            urls.append(value)
    paper_id = str(paper.get('paper_id') or '')
    if paper_id.startswith('SCOPUS_ID:'):
        urls.append(f'https://api.elsevier.com/content/abstract/scopus_id/{quote(paper_id.split(":", 1)[1])}')
    doi = paper.get('doi')
    if _has_value(doi):
        urls.append(f'https://api.elsevier.com/content/abstract/doi/{quote(str(doi).strip(), safe="")}')
    return list(dict.fromkeys(urls))


def _download_elsevier_abstract(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
    """Fetch and return an Elsevier abstract for one paper row."""
    urls = _elsevier_abstract_urls(paper)
    if not urls:
        return False, 'missing Elsevier abstract URL', ''
    api_key = _elsevier_api_key()
    last_error = 'no Elsevier abstract found'
    for url in urls:
        try:
            response = elsevier.get_content(api_key, url, accept='application/json', params={'httpAccept': 'application/json'})
            abstract = _abstract_from_mapping(response.json())
            if abstract:
                return True, 'elsevier', abstract
            last_error = f'no abstract in response from {url}'
        except requests.HTTPError as e:
            response = getattr(e, 'response', None)
            status_code = response.status_code if response is not None else 'HTTP error'
            last_error = f'{status_code} from {url}'
        except requests.RequestException as e:
            last_error = str(e)
    return False, last_error, ''


def _download_abstract(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
    """Fetch abstract text from available metadata providers."""
    errors = []
    if _openalex_identifier(paper) is not None:
        ok, source, abstract = _download_openalex_abstract(paper)
        if ok:
            return ok, source, abstract
        errors.append(f'openalex: {source}')
    if _has_value(paper.get('core_id')):
        ok, source, abstract = _download_core_abstract(paper)
        if ok:
            return ok, source, abstract
        errors.append(f'core: {source}')
    if _elsevier_configured():
        ok, source, abstract = _download_elsevier_abstract(paper)
        if ok:
            return ok, source, abstract
        errors.append(f'elsevier: {source}')
    return False, '; '.join(errors) or 'no abstract source available', ''


def _configured_sources(sources: Iterable[str] | None) -> list[str]:
    """Resolve requested PDF sources, expanding ``all`` to configured providers."""
    if not sources or 'all' in sources:
        settings = load_settings()
        enabled = []
        if _unpaywall_email(settings):
            enabled.append('unpaywall')
        enabled.append('openalex')
        if settings.get('core_api_key') or os.environ.get('CORE_API_KEY'):
            enabled.append('core')
        if settings.get('elsevier_api_key'):
            enabled.append('elsevier')
        return enabled
    invalid = set(sources) - DOWNLOAD_SOURCES
    if invalid:
        raise ValueError(f'download source must be one of: all, {", ".join(sorted(DOWNLOAD_SOURCES))}')
    return list(dict.fromkeys(sources))


def _elsevier_configured() -> bool:
    """Return whether an Elsevier API key is available for downloads."""
    return bool(load_settings().get('elsevier_api_key'))


def _download_pdf_from_sources(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
    sources: Iterable[str],
) -> tuple[bool, str, str]:
    """Try configured PDF sources in order and return success/source details."""
    downloader_by_source = {
        'unpaywall': _download_unpaywall_pdf,
        'openalex': _download_openalex_pdf,
        'core': _download_core_pdf,
        'elsevier': lambda row, path: (_download_pdf(row, path), 'Elsevier PDF download failed'),
    }
    errors = []
    for source in sources:
        downloader = downloader_by_source[source]
        try:
            ok, detail = downloader(paper, filepath)
        except Exception as e:
            ok, detail = False, str(e)
        if ok:
            return True, source, detail
        errors.append(f'{source}: {detail}')
    return False, '; '.join(errors), ''


def _should_try_elsevier_text(paper: Mapping[str, Any]) -> bool:
    """Return whether a paper row advertises Elsevier full text."""
    return _full_text_uri(paper) is not None


def _store_downloaded_asset(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
    role: str,
    source: str,
) -> None:
    """Store a downloaded text or PDF file in the corpus blob store."""
    if role in {'text', 'abstract'}:
        kind = 'text'
        mime_type = 'text/plain'
    elif role == 'pdf':
        kind = 'pdf'
        mime_type = 'application/pdf'
    else:
        raise ValueError('role must be text, abstract, or pdf')
    add_asset(
        conn,
        paper,
        Path(filepath),
        role=role,
        kind=kind,
        mime_type=mime_type,
        source=source,
        original_filename=os.path.basename(filepath),
    )


def download_papers(db_path: str | PathLike[str] = 'papers.db',
                    download_format: str = 'text',
                    sources: Iterable[str] | None = None,
                    download_abstract: bool = True,
                    force: bool = False) -> None:
    """Download paper assets and update a corpus in place.

    Parameters
    ----------
    db_path : str or os.PathLike[str], default='papers.db'
        Path to the SQLite paper corpus.
    download_format : {'both', 'pdf', 'text'}, default='text'
        Primary asset types to download.
    sources : Iterable[str] or None, optional
        Ordered PDF providers to try. ``all`` expands to configured providers.
    download_abstract : bool, default=True
        Whether to retrieve and store abstracts.
    force : bool, default=False
        Redownload requested asset types even when they are already stored.

    Returns
    -------
    None
        Assets and status fields are written directly to the corpus.

    Raises
    ------
    ValueError
        If the format or source is invalid, or required provider configuration
        is unavailable.
    """
    download_format = download_format.lower()
    if download_format not in DOWNLOAD_FORMATS:
        raise ValueError(f'download_format must be one of: {", ".join(sorted(DOWNLOAD_FORMATS))}')
    sources = _configured_sources(sources or ['all'])
    elsevier_text_available = _elsevier_configured()
    with connect(db_path) as conn:
        papers = paper_rows(conn)
        if download_format == 'text' and not elsevier_text_available:
            missing_text = force or any(
                get_asset_metadata(conn, paper['paper_id'], 'text') is None
                for paper in papers
            )
            if missing_text:
                raise ValueError(
                    'Elsevier text download requires an Elsevier API key. Run ps_elsevier_key first.'
                )
        if download_format in {'pdf', 'both'} and not sources:
            missing_pdf = force or any(
                get_asset_metadata(conn, paper['paper_id'], 'pdf') is None
                for paper in papers
            )
            if missing_pdf:
                raise ValueError(
                    'No PDF download sources are configured. Set an Unpaywall email, '
                    'CORE API key, or Elsevier API key.'
                )
        summary = {
            'texts': 0,
            'pdfs': 0,
            'abstracts': 0,
            'texts_skipped': 0,
            'pdfs_skipped': 0,
            'abstracts_skipped': 0,
        }
        with tempfile.TemporaryDirectory(prefix='paperscraper-download-') as download_dir:
            with tqdm(total=len(papers), desc='Downloading Papers', colour='#A020F0') as pbar:
                for paper in papers:
                    paper_summary = _download_paper(conn,
                                                    paper,
                                                    download_dir,
                                                    download_format,
                                                    sources,
                                                    elsevier_text_available,
                                                    download_abstract=download_abstract,
                                                    force=force)
                    for key, value in paper_summary.items():
                        summary[key] += value
                    pbar.update(1)
    print(
        f"Download complete: {summary['texts']} text files, "
        f"{summary['pdfs']} PDFs, {summary['abstracts']} abstracts downloaded."
    )
    skipped = summary['texts_skipped'] + summary['pdfs_skipped'] + summary['abstracts_skipped']
    if skipped:
        print(
            f"Skipped existing corpus assets: {summary['texts_skipped']} text files, "
            f"{summary['pdfs_skipped']} PDFs, {summary['abstracts_skipped']} abstracts. "
            "Use --force to redownload them."
        )


def _download_paper(
    conn: sqlite3.Connection,
    paper: _Paper,
    download_dir: str | PathLike[str],
    download_format: str,
    sources: Iterable[str],
    elsevier_text_available: bool,
    download_abstract: bool = True,
    force: bool = False,
) -> dict[str, int]:
    """Download requested assets for one corpus paper row."""
    filename = _safe_filename(paper)
    summary = {
        'texts': 0,
        'pdfs': 0,
        'abstracts': 0,
        'texts_skipped': 0,
        'pdfs_skipped': 0,
        'abstracts_skipped': 0,
    }
    existing_assets = {
        role: get_asset_metadata(conn, paper.get('paper_id'), role)
        for role in ['abstract', 'text', 'pdf']
    }
    text_requested = download_format in {'text', 'both'}
    pdf_requested = download_format in {'pdf', 'both'}
    text_attempt_needed = (
        text_requested
        and elsevier_text_available
        and _should_try_elsevier_text(paper)
        and (force or existing_assets['text'] is None)
    )
    pdf_attempt_needed = pdf_requested and (force or existing_assets['pdf'] is None)
    pdf_succeeded_from_oa = False

    if download_abstract:
        existing_abstract = existing_assets['abstract']
        if existing_abstract is not None and not force:
            paper['abstract_source'] = existing_abstract.get('source') or paper.get('abstract_source')
            _set_status(paper, 'abstract_download_status', 'succeeded')
            summary['abstracts_skipped'] += 1
        else:
            abstract_filepath = os.path.join(download_dir, f'{filename}-abstract.txt')
            try:
                ok, source_or_error, abstract = _download_abstract(paper)
                if not ok:
                    _set_status(paper, 'abstract_download_status', 'failed', source_or_error)
                    abstract = ''
                if abstract:
                    with open(abstract_filepath, 'w', encoding='utf-8') as out_file:
                        out_file.write(abstract)
                    paper['abstract_source'] = source_or_error
                    _set_status(paper, 'abstract_download_status', 'succeeded')
                    _store_downloaded_asset(conn, paper, abstract_filepath, role='abstract', source=source_or_error)
                    summary['abstracts'] += 1
            except Exception as e:
                _set_status(paper, 'abstract_download_status', 'failed', str(e))

    if text_requested and existing_assets['text'] is not None and not force:
        paper['text_source'] = existing_assets['text'].get('source') or paper.get('text_source')
        _set_status(paper, 'text_download_status', 'succeeded')
        summary['texts_skipped'] += 1

    if text_attempt_needed:
        text_filepath = os.path.join(download_dir, f'{filename}.txt')
        try:
            if os.path.isfile(text_filepath) or _download_text(paper, text_filepath):
                paper['text_path'] = ''
                paper['text_source'] = 'elsevier'
                _set_status(paper, 'text_download_status', 'succeeded')
                _store_downloaded_asset(conn, paper, text_filepath, role='text', source='elsevier')
                summary['texts'] += 1
            else:
                _set_status(paper, 'text_download_status', 'failed', 'Elsevier text download failed')
        except Exception as e:
            _set_status(paper, 'text_download_status', 'failed', str(e))

    if pdf_requested and existing_assets['pdf'] is not None and not force:
        paper['pdf_source'] = existing_assets['pdf'].get('source') or paper.get('pdf_source')
        _set_status(paper, 'pdf_download_status', 'succeeded')
        summary['pdfs_skipped'] += 1

    if pdf_attempt_needed:
        pdf_filepath = os.path.join(download_dir, f'{filename}.pdf')
        try:
            ok, source_or_error, source_url = _download_pdf_from_sources(paper, pdf_filepath, sources)
            if ok:
                paper['pdf_path'] = ''
                paper['pdf_source'] = source_or_error
                if source_or_error in {'unpaywall', 'openalex', 'core'} and source_url:
                    paper['pdf_url'] = source_url
                _set_status(paper, 'pdf_download_status', 'succeeded')
                _store_downloaded_asset(conn, paper, pdf_filepath, role='pdf', source=source_or_error)
                summary['pdfs'] += 1
                pdf_succeeded_from_oa = source_or_error in {'unpaywall', 'openalex', 'core'}
            else:
                _set_status(paper, 'pdf_download_status', 'failed', source_or_error)
        except Exception as e:
            _set_status(paper, 'pdf_download_status', 'failed', str(e))

    if (
        pdf_succeeded_from_oa
        and download_format == 'pdf'
        and not text_attempt_needed
        and existing_assets['text'] is None
        and elsevier_text_available
        and _should_try_elsevier_text(paper)
    ):
        text_filepath = os.path.join(download_dir, f'{filename}.txt')
        try:
            if os.path.isfile(text_filepath) or _download_text(paper, text_filepath):
                paper['text_path'] = ''
                paper['text_source'] = 'elsevier'
                _set_status(paper, 'text_download_status', 'succeeded')
                _store_downloaded_asset(conn, paper, text_filepath, role='text', source='elsevier')
                summary['texts'] += 1
        except Exception as e:
            _set_status(paper, 'text_download_status', 'failed', str(e))
    upsert_paper(conn, paper)
    return summary
