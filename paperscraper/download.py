"""Download paper text and PDFs from configured open-access and publisher sources.

This module powers ``ps_download``. It can fetch Elsevier full text when
available, try PDFs from Unpaywall, CORE, and Elsevier, and update per-paper
download status in the SQLite paper corpus after each row.
"""

import ast
import json
import os
import re
import requests
import tempfile
from pathlib import Path
from tqdm import tqdm
from urllib.parse import quote

from paperscraper import elsevier
from paperscraper.corpus import PIPELINE_COLUMNS, add_asset, connect, paper_rows, upsert_paper
from paperscraper.settings import load_settings

DOWNLOAD_FORMATS = {'text', 'pdf', 'both'}
DOWNLOAD_SOURCES = {'unpaywall', 'core', 'elsevier'}


def _elsevier_api_key():
    """Return the configured Elsevier API key."""
    api_key = load_settings().get('elsevier_api_key')
    if not api_key:
        raise ValueError('Elsevier API key is not configured. Run ps_elsevier_key first.')
    return api_key


def retrieve_document(uri):
    """Retrieve an Elsevier full-text document into the temporary ``data`` folder."""
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


def json_to_text(filepath):
    """Read an Elsevier JSON document and return its original text content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    text = doc.get('originalText')
    if text is None:
        text = (doc.get('full-text-retrieval-response') or {}).get('originalText')
    if type(text) == dict:
        return 'failed'
    return text or 'failed'


def elsevier_string_formatter(text: str):
    """Clean common wrapper artifacts from Elsevier originalText output."""
    if text.count('Acknowledgements') == 2:
        text = text.split('Acknowledgements')[1]
    elif text.count('References') == 2:
        text = text.split('References')[1]
    if 'amazonaws.com/' in text:
        text = text.split('amazonaws.com/')[-1]
        text = text[text.find(' '):]
    return text


def _full_text_uri(paper):
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


def _link_values(link):
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


def _full_text_uri_from_link_value(value):
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


def _is_elsevier_article_endpoint(value):
    """Return whether a URL is an Elsevier article retrieval endpoint."""
    return 'api.elsevier.com/content/article/' in value.lower()


def _has_value(value):
    """Return whether a paper field contains a meaningful non-empty value."""
    return value is not None and str(value).strip() != ''


def _set_status(paper, column: str, status: str, error: str | None = None):
    """Update a corpus paper status field and optional error text."""
    if column not in PIPELINE_COLUMNS:
        raise KeyError(f'Unknown pipeline status column: {column}')
    paper[column] = status
    if error:
        paper['last_error'] = error
    elif status in {'succeeded', 'stored'}:
        paper['last_error'] = ''


def _download_text(paper, filepath):
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


def _pdf_urls(paper):
    """Build Elsevier PDF endpoint candidates for a normalized paper row."""
    urls = []
    doi = paper.get('doi')
    if _has_value(doi):
        urls.append(elsevier.article_url_from_doi(str(doi)))
    uri = _full_text_uri(paper)
    if uri:
        urls.append(uri)
    return urls


def _download_pdf(paper, filepath):
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


def _safe_filename(paper):
    """Create a filesystem-safe filename stem for a paper row."""
    for column in ['doi', 'core_id', 'paper_id']:
        value = paper.get(column)
        if _has_value(value):
            safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())
            safe = safe.strip('._')
            if safe:
                return safe
    return 'paper'


def _unpaywall_email(settings=None):
    """Return the configured email address used for Unpaywall requests."""
    settings = settings or load_settings()
    return settings.get('unpaywall_email') or os.environ.get('UNPAYWALL_EMAIL')


def _download_url_to_pdf(url, filepath, headers=None):
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


def _download_unpaywall_pdf(paper, filepath):
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


def _core_headers():
    """Build request headers for CORE downloads."""
    settings = load_settings()
    api_key = settings.get('core_api_key') or os.environ.get('CORE_API_KEY')
    headers = {'User-Agent': 'PaperScraper/0.0.1'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


def _download_core_pdf(paper, filepath):
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


def _configured_sources(sources):
    """Resolve requested PDF sources, expanding ``all`` to configured providers."""
    if not sources or 'all' in sources:
        settings = load_settings()
        enabled = []
        if _unpaywall_email(settings):
            enabled.append('unpaywall')
        if settings.get('core_api_key') or os.environ.get('CORE_API_KEY'):
            enabled.append('core')
        if settings.get('elsevier_api_key'):
            enabled.append('elsevier')
        return enabled
    invalid = set(sources) - DOWNLOAD_SOURCES
    if invalid:
        raise ValueError(f'download source must be one of: all, {", ".join(sorted(DOWNLOAD_SOURCES))}')
    return list(dict.fromkeys(sources))


def _elsevier_configured():
    """Return whether an Elsevier API key is available for downloads."""
    return bool(load_settings().get('elsevier_api_key'))


def _download_pdf_from_sources(paper, filepath, sources):
    """Try configured PDF sources in order and return success/source details."""
    downloader_by_source = {
        'unpaywall': _download_unpaywall_pdf,
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


def _should_try_elsevier_text(paper):
    """Return whether a paper row advertises Elsevier full text."""
    return _full_text_uri(paper) is not None


def _store_downloaded_asset(conn, paper, filepath, role, source):
    """Store a downloaded text or PDF file in the corpus blob store."""
    if role == 'text':
        kind = 'text'
        mime_type = 'text/plain'
    elif role == 'pdf':
        kind = 'pdf'
        mime_type = 'application/pdf'
    else:
        raise ValueError('role must be text or pdf')
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


def download_papers(db_path='papers.db',
                    download_format='text',
                    sources=None):
    """Download requested paper assets and update the corpus database in place."""
    download_format = download_format.lower()
    if download_format not in DOWNLOAD_FORMATS:
        raise ValueError(f'download_format must be one of: {", ".join(sorted(DOWNLOAD_FORMATS))}')
    sources = _configured_sources(sources or ['all'])
    elsevier_text_available = _elsevier_configured()
    if download_format == 'text' and not elsevier_text_available:
        raise ValueError('Elsevier text download requires an Elsevier API key. Run ps_elsevier_key first.')
    if download_format in {'pdf', 'both'} and not sources:
        raise ValueError(
            'No PDF download sources are configured. Set an Unpaywall email, CORE API key, or Elsevier API key.')
    with connect(db_path) as conn:
        papers = paper_rows(conn)
        with tempfile.TemporaryDirectory(prefix='paperscraper-download-') as download_dir:
            with tqdm(total=len(papers), desc='Downloading Papers', colour='#A020F0') as pbar:
                for paper in papers:
                    _download_paper(conn, paper, download_dir, download_format, sources, elsevier_text_available)
                    pbar.update(1)


def _download_paper(conn, paper, download_dir, download_format, sources, elsevier_text_available):
    """Download requested assets for one corpus paper row."""
    filename = _safe_filename(paper)
    text_attempt_needed = (
        download_format in {'text', 'both'}
        and elsevier_text_available
        and _should_try_elsevier_text(paper)
    )
    pdf_attempt_needed = download_format in {'pdf', 'both'}
    pdf_succeeded_from_oa = False

    if text_attempt_needed:
        text_filepath = os.path.join(download_dir, f'{filename}.txt')
        try:
            if os.path.isfile(text_filepath) or _download_text(paper, text_filepath):
                paper['text_path'] = ''
                paper['text_source'] = 'elsevier'
                _set_status(paper, 'text_download_status', 'succeeded')
                _store_downloaded_asset(conn, paper, text_filepath, role='text', source='elsevier')
            else:
                _set_status(paper, 'text_download_status', 'failed', 'Elsevier text download failed')
        except Exception as e:
            _set_status(paper, 'text_download_status', 'failed', str(e))

    if pdf_attempt_needed:
        pdf_filepath = os.path.join(download_dir, f'{filename}.pdf')
        try:
            ok, source_or_error, source_url = _download_pdf_from_sources(paper, pdf_filepath, sources)
            if ok:
                paper['pdf_path'] = ''
                paper['pdf_source'] = source_or_error
                if source_or_error in {'unpaywall', 'core'} and source_url:
                    paper['pdf_url'] = source_url
                _set_status(paper, 'pdf_download_status', 'succeeded')
                _store_downloaded_asset(conn, paper, pdf_filepath, role='pdf', source=source_or_error)
                pdf_succeeded_from_oa = source_or_error in {'unpaywall', 'core'}
            else:
                _set_status(paper, 'pdf_download_status', 'failed', source_or_error)
        except Exception as e:
            _set_status(paper, 'pdf_download_status', 'failed', str(e))

    if (
        pdf_succeeded_from_oa
        and not text_attempt_needed
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
        except Exception as e:
            _set_status(paper, 'text_download_status', 'failed', str(e))
    upsert_paper(conn, paper)
