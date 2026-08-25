"""Download paper text and PDFs from configured open-access and publisher sources.

This module powers ``pm_download``. It can fetch Elsevier, PubMed Central
open-access, medRxiv, and bioRxiv full text, try PDFs from Unpaywall, OpenAlex,
CORE, Elsevier, PubMed Central, medRxiv, bioRxiv, and arXiv, and update
per-paper download status in the SQLite paper corpus after each row. arXiv
serves PDFs and abstracts but no full text, because it publishes no
machine-readable full-text format. medRxiv and bioRxiv publish both: their PDFs
and their JATS full text, the latter being the same format PubMed Central
serves, so text from either needs no PDF scrape.
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
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from typing import Any, TypeAlias
from urllib.parse import quote

from paperminer.providers import (arxiv, biorxiv, chemrxiv, core, elsevier, medrxiv,
                          openalex, pubmed, unpaywall)
from paperminer.corpus.database import (PIPELINE_COLUMNS, add_asset, connect,
                                get_asset_metadata, paper_rows, upsert_paper)
from paperminer.providers import registry
from paperminer.settings import load_settings

DOWNLOAD_FORMATS = {'abstract', 'text', 'pdf', 'both'}
DOWNLOAD_SOURCES = {*registry.names(registry.PDF), *registry.names(registry.TEXT),
                    *registry.names(registry.ABSTRACT)}
# Providers that serve a machine-readable full text rather than only a PDF.
TEXT_SOURCES = set(registry.names(registry.TEXT))
_Paper: TypeAlias = dict[str, Any]


def _elsevier_string_formatter(text: str) -> str:
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
    """Download Elsevier full text for one paper row to ``filepath``.

    The payload is read in memory and written straight to the destination. It
    used to be staged through a ``data`` directory created in the working
    directory, every file of which was deleted first -- so a download run wiped
    whatever a user happened to keep in ``./data``.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded text.

    Returns
    -------
    bool
        Whether text was retrieved and written.
    """
    uri = _full_text_uri(paper)
    if not uri:
        return False
    payload = elsevier.request_json(uri, elsevier.configured_api_key(),
                                    params={'httpAccept': 'application/json'})
    text = elsevier.full_text(payload)
    if not text:
        return False
    with open(filepath, 'w', encoding='utf-8') as out_file:
        out_file.write(_elsevier_string_formatter(text))
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
    api_key = elsevier.configured_api_key()
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
    """Return the configured email address used for Unpaywall requests.

    Parameters
    ----------
    settings : Mapping[str, str] or None, optional
        Loaded PaperMiner settings.

    Returns
    -------
    str or None
        Contact address, or ``None`` when none is configured.
    """
    return unpaywall.configured_email(settings) or None


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
    """Use Unpaywall metadata to locate and download an open-access PDF.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded PDF.

    Returns
    -------
    tuple[bool, str]
        Success flag, and the URL used or the failure reason.
    """
    doi = paper.get('doi')
    if not _has_value(doi):
        return False, 'missing DOI'
    try:
        metadata = unpaywall.get_work(doi)
    except (ValueError, RuntimeError) as e:
        return False, str(e)
    if not metadata:
        return False, 'Unpaywall knows nothing of this DOI'
    last_error = 'no Unpaywall PDF URL found'
    for url in unpaywall.pdf_candidates(metadata):
        ok, error = _download_url_to_pdf(url, filepath)
        if ok:
            return True, url
        last_error = error
    return False, last_error


def _download_core_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Download a PDF through CORE using a stored PDF URL or CORE work ID."""
    urls = []
    url = paper.get('pdf_url')
    if _has_value(url):
        urls.append(str(url).strip())
    core_id = core.resolve_core_id(paper)
    if core_id:
        urls.append(f'{core.work_url(core_id)}/download')
    last_error = 'no CORE download URL found'
    for candidate in dict.fromkeys(urls):
        ok, error = _download_url_to_pdf(candidate, filepath, headers=core.request_headers())
        if ok:
            return True, candidate
        last_error = error
    return False, last_error


def _openalex_identifier(paper: Mapping[str, Any]) -> str | None:
    """Resolve an identifier accepted by the OpenAlex works endpoint.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row containing a DOI or an ``openalex:`` paper ID.

    Returns
    -------
    str or None
        A DOI-qualified identifier or OpenAlex work ID, or ``None`` when the
        paper cannot be looked up through OpenAlex.
    """
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


def _pubmed_identifier(paper: Mapping[str, Any]) -> str | None:
    """Resolve a stored PubMed identifier for a paper row.

    Only values already held in the corpus are considered, so this never issues
    a request. Rows without a PMID can still be resolved from their DOI by
    :func:`paperminer.providers.pubmed.resolve_pmid`, at the cost of one lookup.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row containing a PMID or a ``pmid:`` paper ID.

    Returns
    -------
    str or None
        Bare PMID digits, or ``None`` when the row stores no PubMed identifier.
    """
    pmid = pubmed.normalize_pmid(paper.get('pmid')) if _has_value(paper.get('pmid')) else ''
    if pmid:
        return pmid
    paper_id = str(paper.get('paper_id') or '')
    if paper_id.startswith('pmid:'):
        return pubmed.normalize_pmid(paper_id.split(':', 1)[1]) or None
    return None


def _pubmed_credentials() -> tuple[str | None, str]:
    """Return the configured NCBI API key and contact address.

    Returns
    -------
    tuple[str or None, str]
        API key, which may be ``None``, and contact address, which may be empty.
    """
    settings = load_settings()
    return pubmed.configured_api_key(settings), pubmed.configured_email(settings)


def _download_pubmed_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Download an open-access PDF through the PubMed Central OA service.

    Only the open-access subset is redistributable, so a paper outside it
    reports that no PDF is offered rather than failing.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded PDF.

    Returns
    -------
    tuple[bool, str]
        Success flag, and the source URL or a failure reason.
    """
    api_key, email = _pubmed_credentials()
    try:
        pmcid = pubmed.resolve_pmcid(paper, api_key=api_key, email=email)
    except RuntimeError as e:
        return False, str(e)
    if not pmcid:
        return False, 'missing PMC ID'
    try:
        urls = pubmed.oa_package_urls(pmcid, api_key=api_key, email=email)
    except RuntimeError as e:
        return False, str(e)
    last_error = f'no open-access PDF offered for {pmcid}'
    for url in urls:
        if not url.lower().endswith('.pdf'):
            continue
        ok, error = _download_url_to_pdf(url, filepath)
        if ok:
            return True, url
        last_error = error
    return False, last_error


def _should_try_pmc_text(paper: Mapping[str, Any]) -> bool:
    """Return whether a paper row can be looked up in PubMed Central.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    bool
        Whether the row carries a PMC ID, PMID, or DOI to resolve from.
    """
    return any(_has_value(paper.get(column)) for column in ['pmcid', 'pmid', 'doi'])


def _download_pmc_text(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Write PubMed Central open-access full text for one paper row.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    filepath : str or os.PathLike[str]
        Destination path for the extracted text.

    Returns
    -------
    tuple[bool, str]
        Success flag, and an empty string or a failure reason.

    Raises
    ------
    OSError
        If the text cannot be written.
    """
    api_key, email = _pubmed_credentials()
    try:
        pmcid = pubmed.resolve_pmcid(paper, api_key=api_key, email=email)
        if not pmcid:
            return False, 'missing PMC ID'
        text = pubmed.pmc_full_text(pmcid, api_key=api_key, email=email)
    except RuntimeError as e:
        return False, str(e)
    if not text:
        return False, f'no open-access full text for {pmcid}'
    with open(filepath, 'w', encoding='utf-8') as out_file:
        out_file.write(text)
    return True, ''


def _download_pubmed_abstract(
    paper: MutableMapping[str, Any],
) -> tuple[bool, str, str]:
    """Fetch and return a PubMed abstract for one paper row.

    A stored PMID is used directly. A row that has only a DOI is resolved
    through the PMC ID converter, and the resolved identifier is recorded on
    ``paper`` so the corpus keeps it and later runs take the direct path.

    Parameters
    ----------
    paper : MutableMapping[str, Any]
        Corpus paper row containing a PMID or a DOI. A resolved PMID is stored
        back into this mapping.

    Returns
    -------
    tuple[bool, str, str]
        Success flag, provider name or failure reason, and normalized abstract
        text. The text is empty when retrieval fails.
    """
    api_key, email = _pubmed_credentials()
    try:
        pmid = pubmed.resolve_pmid(paper, api_key=api_key, email=email)
        if not pmid:
            return False, 'missing PMID', ''
        paper['pmid'] = pmid
        articles = pubmed.parse_articles(
            pubmed.efetch_ids([pmid], api_key=api_key, email=email)
        )
    except RuntimeError as e:
        return False, str(e), ''
    if not articles:
        return False, f'no PubMed record found for {pmid}', ''
    abstract = _clean_abstract(articles[0].get('abstract'))
    if abstract:
        return True, 'pubmed', abstract
    return False, f'no PubMed abstract found for {pmid}', ''


def _medrxiv_identifier(paper: Mapping[str, Any]) -> str | None:
    """Resolve a stored medRxiv DOI for a paper row.

    Only values already held in the corpus are considered, so this never issues
    a request. medRxiv publishes no title search, so there is no lookup to fall
    back on for a row that carries neither a medRxiv DOI nor a medRxiv URL.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row containing a medRxiv DOI, a medRxiv paper ID, or a
        medRxiv content URL.

    Returns
    -------
    str or None
        Bare medRxiv DOI, or ``None`` when the row stores none.
    """
    return medrxiv.resolve_medrxiv_doi(paper) or None


def _medrxiv_record(paper: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str]:
    """Fetch the newest posted version of a row's medRxiv preprint.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a medRxiv DOI.

    Returns
    -------
    tuple[Mapping[str, Any] or None, str]
        Normalized record, or ``None`` and a failure reason.
    """
    identifier = _medrxiv_identifier(paper)
    if identifier is None:
        return None, 'missing medRxiv DOI'
    try:
        entry = medrxiv.fetch_doi(identifier)
    except RuntimeError as e:
        return None, str(e)
    if entry is None:
        return None, f'no medRxiv record found for {identifier}'
    return entry, ''


def _download_medrxiv_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Download a paper's PDF from medRxiv.

    The version is read from the record rather than assumed, because a
    preprint's PDF is served per posted version and the newest one is the
    document the corpus row describes. medRxiv fronts these files with a bot
    challenge that can refuse a client outright; a refusal is reported as the
    failure reason rather than worked around, and full text remains reachable
    through the medRxiv text source when it happens.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a medRxiv DOI.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded PDF.

    Returns
    -------
    tuple[bool, str]
        Success flag, and the source URL or a failure reason.
    """
    entry, error = _medrxiv_record(paper)
    if entry is None:
        return False, error
    url = medrxiv.pdf_url(str(entry.get('medrxiv_doi') or ''), str(entry.get('version') or ''))
    if not url:
        return False, 'missing medRxiv DOI'
    ok, error = _download_url_to_pdf(url, filepath, headers=medrxiv.request_headers())
    return (True, url) if ok else (False, error)


def _download_medrxiv_text(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Write medRxiv JATS full text for one paper row.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a medRxiv DOI.
    filepath : str or os.PathLike[str]
        Destination path for the extracted text.

    Returns
    -------
    tuple[bool, str]
        Success flag, and an empty string or a failure reason.

    Raises
    ------
    OSError
        If the text cannot be written.
    """
    entry, error = _medrxiv_record(paper)
    if entry is None:
        return False, error
    try:
        text = medrxiv.full_text(entry)
    except RuntimeError as e:
        return False, str(e)
    if not text:
        return False, f'no medRxiv full text for {entry.get("medrxiv_doi")}'
    with open(filepath, 'w', encoding='utf-8') as out_file:
        out_file.write(text)
    return True, ''


def _download_medrxiv_abstract(
    paper: Mapping[str, Any],
) -> tuple[bool, str, str]:
    """Fetch and return a medRxiv abstract for one paper row.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a medRxiv DOI.

    Returns
    -------
    tuple[bool, str, str]
        Success flag, provider name or failure reason, and normalized abstract
        text. The text is empty when retrieval fails.
    """
    entry, error = _medrxiv_record(paper)
    if entry is None:
        return False, error, ''
    abstract = _clean_abstract(entry.get('abstract'))
    if abstract:
        return True, 'medrxiv', abstract
    return False, f'no medRxiv abstract found for {entry.get("medrxiv_doi")}', ''


def _biorxiv_identifier(paper: Mapping[str, Any]) -> str | None:
    """Resolve a stored bioRxiv DOI for a paper row.

    Only values already held in the corpus are considered, so this never issues
    a request. bioRxiv publishes no title search, so there is no lookup to fall
    back on for a row that carries neither a bioRxiv DOI nor a bioRxiv URL.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row containing a bioRxiv DOI, a bioRxiv paper ID, or a
        bioRxiv content URL.

    Returns
    -------
    str or None
        Bare bioRxiv DOI, or ``None`` when the row stores none.
    """
    return biorxiv.resolve_biorxiv_doi(paper) or None


def _biorxiv_record(paper: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str]:
    """Fetch the newest posted version of a row's bioRxiv preprint.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a bioRxiv DOI.

    Returns
    -------
    tuple[Mapping[str, Any] or None, str]
        Normalized record, or ``None`` and a failure reason.
    """
    identifier = _biorxiv_identifier(paper)
    if identifier is None:
        return None, 'missing bioRxiv DOI'
    try:
        entry = biorxiv.fetch_doi(identifier)
    except RuntimeError as e:
        return None, str(e)
    if entry is None:
        return None, f'no bioRxiv record found for {identifier}'
    return entry, ''


def _download_biorxiv_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Download a paper's PDF from bioRxiv.

    The version is read from the record rather than assumed, because a
    preprint's PDF is served per posted version and the newest one is the
    document the corpus row describes. bioRxiv fronts these files with the same
    bot challenge medRxiv uses, and it can refuse a client outright; a refusal
    is reported as the failure reason rather than worked around, and full text
    remains reachable through the bioRxiv text source when it happens.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a bioRxiv DOI.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded PDF.

    Returns
    -------
    tuple[bool, str]
        Success flag, and the source URL or a failure reason.
    """
    entry, error = _biorxiv_record(paper)
    if entry is None:
        return False, error
    url = biorxiv.pdf_url(str(entry.get('biorxiv_doi') or ''), str(entry.get('version') or ''))
    if not url:
        return False, 'missing bioRxiv DOI'
    ok, error = _download_url_to_pdf(url, filepath, headers=biorxiv.request_headers())
    return (True, url) if ok else (False, error)


def _download_biorxiv_text(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Write bioRxiv JATS full text for one paper row.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a bioRxiv DOI.
    filepath : str or os.PathLike[str]
        Destination path for the extracted text.

    Returns
    -------
    tuple[bool, str]
        Success flag, and an empty string or a failure reason.

    Raises
    ------
    OSError
        If the text cannot be written.
    """
    entry, error = _biorxiv_record(paper)
    if entry is None:
        return False, error
    try:
        text = biorxiv.full_text(entry)
    except RuntimeError as e:
        return False, str(e)
    if not text:
        return False, f'no bioRxiv full text for {entry.get("biorxiv_doi")}'
    with open(filepath, 'w', encoding='utf-8') as out_file:
        out_file.write(text)
    return True, ''


def _download_biorxiv_abstract(
    paper: Mapping[str, Any],
) -> tuple[bool, str, str]:
    """Fetch and return a bioRxiv abstract for one paper row.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a bioRxiv DOI.

    Returns
    -------
    tuple[bool, str, str]
        Success flag, provider name or failure reason, and normalized abstract
        text. The text is empty when retrieval fails.
    """
    entry, error = _biorxiv_record(paper)
    if entry is None:
        return False, error, ''
    abstract = _clean_abstract(entry.get('abstract'))
    if abstract:
        return True, 'biorxiv', abstract
    return False, f'no bioRxiv abstract found for {entry.get("biorxiv_doi")}', ''


def _chemrxiv_identifier(paper: Mapping[str, Any]) -> str | None:
    """Resolve a stored chemRxiv DOI for a paper row.

    Only values already held in the corpus are considered, so this never issues
    a request. chemRxiv does publish a title search, but it is not used here:
    it costs a request per row and can return a different preprint with a
    similar title, which would attach the wrong document to the row.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row containing a chemRxiv DOI, a chemRxiv paper ID, or a
        chemRxiv content URL.

    Returns
    -------
    str or None
        chemRxiv DOI as registered, or ``None`` when the row stores none.
    """
    return chemrxiv.resolve_chemrxiv_doi(paper) or None


def _chemrxiv_record(paper: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str]:
    """Fetch the newest posted version of a row's chemRxiv preprint.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a chemRxiv DOI.

    Returns
    -------
    tuple[Mapping[str, Any] or None, str]
        Normalized record, or ``None`` and a failure reason.
    """
    identifier = _chemrxiv_identifier(paper)
    if identifier is None:
        return None, 'missing chemRxiv DOI'
    try:
        entry = chemrxiv.fetch_doi(identifier)
    except RuntimeError as e:
        return None, str(e)
    if entry is None:
        return None, f'no chemRxiv record found for {identifier}'
    return entry, ''


def _download_chemrxiv_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Download a paper's PDF from chemRxiv.

    The location is derived from the DOI, which is what the registry currently
    records; the record's own asset URL is tried second, because chemRxiv
    changed hosting platforms and older records still name the previous one.

    chemrxiv.org fronts these files with a bot challenge that can refuse a
    client outright. A refusal is reported as the failure reason rather than
    worked around, and the paper stays reachable through the other open-access
    PDF sources when it happens.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a chemRxiv DOI.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded PDF.

    Returns
    -------
    tuple[bool, str]
        Success flag, and the source URL on success or a failure reason.
    """
    entry, error = _chemrxiv_record(paper)
    if entry is None:
        return False, error
    identifier = str(entry.get('chemrxiv_doi') or '')
    candidates = [chemrxiv.pdf_url(identifier), str(entry.get('asset_url') or '')]
    errors = []
    for url in [candidate for candidate in candidates if candidate]:
        ok, failure = _download_url_to_pdf(url, filepath, headers=chemrxiv.request_headers())
        if ok:
            return True, url
        errors.append(failure)
    if not errors:
        return False, 'missing chemRxiv DOI'
    return False, '; '.join(errors)


def _download_chemrxiv_abstract(
    paper: Mapping[str, Any],
) -> tuple[bool, str, str]:
    """Fetch and return a chemRxiv abstract for one paper row.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying a chemRxiv DOI.

    Returns
    -------
    tuple[bool, str, str]
        Success flag, provider name or failure reason, and normalized abstract
        text. The text is empty when retrieval fails.
    """
    entry, error = _chemrxiv_record(paper)
    if entry is None:
        return False, error, ''
    abstract = _clean_abstract(entry.get('abstract'))
    if abstract:
        return True, 'chemrxiv', abstract
    return False, f'no chemRxiv abstract found for {entry.get("chemrxiv_doi")}', ''


def _arxiv_identifier(paper: Mapping[str, Any]) -> str | None:
    """Resolve a stored arXiv identifier for a paper row.

    Only values already held in the corpus are considered, so this never issues
    a request. arXiv has no DOI lookup, so the only alternative would be a
    title search, which costs one paced request per paper and can match the
    wrong record; :func:`paperminer.providers.arxiv.resolve_arxiv_id` offers that
    deliberately as an opt-in rather than running it on every download.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row containing an arXiv ID, an ``arxiv:`` paper ID, or an
        arXiv PDF URL.

    Returns
    -------
    str or None
        Bare arXiv identifier, or ``None`` when the row stores none.
    """
    identifier = arxiv.normalize_arxiv_id(paper.get('arxiv_id')) if _has_value(paper.get('arxiv_id')) else ''
    if identifier:
        return identifier
    paper_id = str(paper.get('paper_id') or '')
    if paper_id.startswith('arxiv:'):
        identifier = arxiv.normalize_arxiv_id(paper_id.split(':', 1)[1])
        if identifier:
            return identifier
    url = str(paper.get('pdf_url') or '')
    if 'arxiv.org' in url.lower():
        return arxiv.normalize_arxiv_id(url) or None
    return None


def _download_arxiv_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Download a paper's PDF from arXiv.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying an arXiv identifier.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded PDF.

    Returns
    -------
    tuple[bool, str]
        Success flag, and the source URL or a failure reason.
    """
    identifier = _arxiv_identifier(paper)
    if identifier is None:
        return False, 'missing arXiv ID'
    url = f'{arxiv.PDF_URL}/{quote(identifier, safe="/")}'
    ok, error = _download_url_to_pdf(url, filepath, headers=arxiv.request_headers())
    return (True, url) if ok else (False, error)


def _download_arxiv_abstract(
    paper: Mapping[str, Any],
) -> tuple[bool, str, str]:
    """Fetch and return an arXiv abstract for one paper row.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row carrying an arXiv identifier.

    Returns
    -------
    tuple[bool, str, str]
        Success flag, provider name or failure reason, and normalized abstract
        text. The text is empty when retrieval fails.
    """
    identifier = _arxiv_identifier(paper)
    if identifier is None:
        return False, 'missing arXiv ID', ''
    try:
        entries = arxiv.parse_entries(arxiv.fetch_ids([identifier]))
    except RuntimeError as e:
        return False, str(e), ''
    if not entries:
        return False, f'no arXiv record found for {identifier}', ''
    abstract = _clean_abstract(entries[0].get('abstract'))
    if abstract:
        return True, 'arxiv', abstract
    return False, f'no arXiv abstract found for {identifier}', ''


def _download_openalex_abstract(
    paper: Mapping[str, Any],
) -> tuple[bool, str, str]:
    """Fetch and reconstruct an abstract from OpenAlex work metadata.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row containing a DOI or OpenAlex work identifier.

    Returns
    -------
    tuple[bool, str, str]
        Success flag, provider name or failure reason, and normalized abstract
        text. The text is empty when retrieval fails or no abstract is stored.
    """
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


def _download_core_abstract(paper: Mapping[str, Any]) -> tuple[bool, str, str]:
    """Fetch and return a CORE abstract for one paper row.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    tuple[bool, str, str]
        Success flag, provider name or failure reason, and the abstract text.
    """
    core_id = core.resolve_core_id(paper)
    if not core_id:
        return False, 'missing CORE ID', ''
    try:
        work = core.get_work(core_id)
    except RuntimeError as e:
        return False, str(e), ''
    abstract = _abstract_from_mapping(work) if work else ''
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
    api_key = elsevier.configured_api_key()
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


def _reachable(answer: object) -> bool:
    """Read an identifier helper's answer as a yes or a no.

    Parameters
    ----------
    answer : object
        Either a bool from a reachability predicate, or an identifier -- or
        ``None`` -- from one of the ``_<source>_identifier`` helpers.

    Returns
    -------
    bool
        Whether the source can be asked about this paper.
    """
    return answer is not False and answer is not None


def _source_reachable(name: str, capability: str, paper: Mapping[str, Any]) -> bool:
    """Report whether a registered source can be asked for one paper asset."""
    predicate = registry.resolve_reachability(name, capability)
    return predicate is None or _reachable(predicate(paper))


def _pubmed_abstract_reachable(paper: Mapping[str, Any]) -> bool:
    """Report whether PubMed can be asked for this paper's abstract.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    bool
        Whether the row carries a PubMed identifier or a DOI to resolve one
        from.
    """
    return _pubmed_identifier(paper) is not None or _has_value(paper.get('doi'))


def _core_abstract_reachable(paper: Mapping[str, Any]) -> bool:
    """Report whether CORE can be asked for this paper's abstract.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    bool
        Whether the row carries a CORE identifier.
    """
    return _has_value(paper.get('core_id'))


def _elsevier_abstract_reachable(paper: Mapping[str, Any]) -> bool:
    """Report whether Elsevier can be asked for this paper's abstract.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row. Unused; Elsevier is reachable for any row once a key
        is configured, because the lookup goes by DOI.

    Returns
    -------
    bool
        Whether an Elsevier API key is configured.
    """
    return _elsevier_configured()


def _download_abstract(paper: MutableMapping[str, Any],
                       sources: Iterable[str] | None = None) -> tuple[bool, str, str]:
    """Fetch abstract text from the requested metadata providers.

    Sources are tried in the registry's abstract order, skipping any the row
    cannot reach and any the caller did not ask for. Until now the caller's
    ``--source`` selection was ignored here entirely, so a run scoped to one
    provider still queried all eight.

    A PubMed identifier resolved from a DOI is recorded on ``paper`` so the
    corpus keeps it after the caller upserts the row.

    Parameters
    ----------
    paper : MutableMapping[str, Any]
        Corpus paper row.
    sources : Iterable[str] or None, optional
        Resolved provider names for this run. Defaults to every source that
        serves abstracts.

    Returns
    -------
    tuple[bool, str, str]
        Success flag, provider name or joined failure reasons, and the abstract
        text.
    """
    requested = set(sources) if sources is not None else set(registry.names(registry.ABSTRACT))
    errors = []
    for name in registry.names(registry.ABSTRACT):
        if name not in requested:
            continue
        if not _source_reachable(name, registry.ABSTRACT, paper):
            continue
        downloader = registry.resolve_handler(name, registry.ABSTRACT)
        ok, source, abstract = downloader(paper)
        if ok:
            return ok, source, abstract
        errors.append(f'{name}: {source}')
    return False, '; '.join(errors) or 'no abstract source available', ''


def _configured_sources(requested: Iterable[str] | None) -> list[str]:
    """Resolve requested download sources, dropping ones with no credential.

    An unscoped run skips a source whose credential is not configured, because
    a request it cannot authenticate is a wasted round trip. An explicitly
    named source is kept whatever its credential, so the failure says what is
    missing rather than silently doing nothing.

    Parameters
    ----------
    requested : Iterable[str] or None
        Requested source names, or ``None`` for every configured source.

    Returns
    -------
    list[str]
        Source names in the registry's download order.

    Raises
    ------
    ValueError
        If a requested name is not a download source.
    """
    asked = list(requested or [])
    resolved = registry.resolve_names(asked, registry.PDF, preserve_order=True,
                                     label='download')
    if asked and 'all' not in {str(name).strip().lower() for name in asked}:
        return resolved
    settings = load_settings()
    return [name for name in resolved if _source_configured(name, settings)]


def _source_configured(name: str, settings: Mapping[str, str]) -> bool:
    """Report whether a source that needs a credential has one.

    Only a required credential gates a source. OpenAlex and PubMed both answer
    without a key -- the key buys a larger budget -- so an unscoped run still
    asks them.

    Parameters
    ----------
    name : str
        Registry source name.
    settings : Mapping[str, str]
        Loaded PaperMiner settings.

    Returns
    -------
    bool
        Whether the source can be used without further configuration.
    """
    entry = registry.SOURCES[name]
    if not entry.credential_required:
        return True
    return bool(settings.get(entry.credential) or os.environ.get(entry.credential_env))


def _elsevier_configured() -> bool:
    """Return whether an Elsevier API key is available for downloads."""
    return bool(load_settings().get('elsevier_api_key'))


def _download_elsevier_pdf(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> tuple[bool, str]:
    """Fetch a paper's PDF through the Elsevier article route.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded PDF.

    Returns
    -------
    tuple[bool, str]
        Success flag, and the failure reason when unsuccessful.
    """
    return _download_pdf(paper, filepath), 'Elsevier PDF download failed'


def _pdf_downloader(source: str) -> Callable[..., tuple[bool, str]]:
    """Return the PDF downloader for one source.

    Parameters
    ----------
    source : str
        Registry source name.

    Returns
    -------
    Callable[..., tuple[bool, str]]
        Downloader taking a paper row and a destination path.
    """
    return registry.resolve_handler(source, registry.PDF)


def _download_pdf_from_sources(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
    sources: Iterable[str],
) -> tuple[bool, str, str]:
    """Try configured PDF sources in order and return success/source details."""
    errors = []
    for source in sources:
        downloader = _pdf_downloader(source)
        try:
            ok, detail = downloader(paper, filepath)
        except Exception as e:
            ok, detail = False, str(e)
        if ok:
            return True, source, detail
        errors.append(f'{source}: {detail}')
    return False, '; '.join(errors), ''


def _download_elsevier_text(paper: Mapping[str, Any],
                            filepath: str | PathLike[str]) -> tuple[bool, str]:
    """Fetch a paper's full text through the Elsevier article route.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded text.

    Returns
    -------
    tuple[bool, str]
        Success flag, and the failure reason when unsuccessful.
    """
    if os.path.isfile(filepath) or _download_text(paper, filepath):
        return True, ''
    return False, 'Elsevier text download failed'


def _download_text_from_sources(
    paper: Mapping[str, Any],
    filepath: str | PathLike[str],
    sources: Iterable[str],
) -> tuple[bool, str, str]:
    """Try each requested full-text source in order and report the outcome.

    Elsevier is one of these sources rather than a separate flag threaded
    through three call sites, so ``--source elsevier --format text`` means what
    it says and ``--source pubmed`` no longer also tries Elsevier.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    filepath : str or os.PathLike[str]
        Destination path for the downloaded text.
    sources : Iterable[str]
        Resolved provider names for this run.

    Returns
    -------
    tuple[bool, str, str]
        Success flag, provider name or joined failure reasons, and an empty
        string reserved for a source URL.
    """
    requested = set(sources)
    errors = []
    for name in registry.names(registry.TEXT):
        if name not in requested:
            continue
        if not _source_reachable(name, registry.TEXT, paper):
            continue
        try:
            downloader = registry.resolve_handler(name, registry.TEXT)
            ok, detail = downloader(paper, filepath)
            if ok:
                return True, name, ''
            errors.append(f'{name}: {detail}')
        except Exception as e:
            errors.append(f'{name}: {e}')
    return False, '; '.join(errors) or 'no full-text source available', ''


def _should_try_elsevier_text(paper: Mapping[str, Any]) -> bool:
    """Return whether a paper row advertises Elsevier full text."""
    return _full_text_uri(paper) is not None


def _should_try_medrxiv_text(paper: Mapping[str, Any]) -> bool:
    """Return whether a paper row can be looked up on medRxiv.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    bool
        Whether the row carries a medRxiv DOI to fetch a JATS document with.
    """
    return _medrxiv_identifier(paper) is not None


def _should_try_biorxiv_text(paper: Mapping[str, Any]) -> bool:
    """Return whether a paper row can be looked up on bioRxiv.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.

    Returns
    -------
    bool
        Whether the row carries a bioRxiv DOI to fetch a JATS document with.
    """
    return _biorxiv_identifier(paper) is not None


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
    download_format : {'abstract', 'both', 'pdf', 'text'}, default='text'
        Primary asset types to download.
    sources : Iterable[str] or None, optional
        Ordered content providers to try. ``all`` expands to the configured
        providers. The selection orders PDF retrieval and also decides whether
        PubMed Central is consulted for full text.
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
    with connect(db_path) as conn:
        papers = paper_rows(conn)
        if download_format == 'text' and not TEXT_SOURCES.intersection(sources):
            missing_text = force or any(
                get_asset_metadata(conn, paper['paper_id'], 'text') is None
                for paper in papers
            )
            if missing_text:
                raise ValueError(
                    'Text download requires an Elsevier API key, the pubmed source, or a '
                    'preprint-server source. Run pm_elsevier_key first, pass --source pubmed '
                    'for PubMed Central open-access full text, or pass --source medrxiv or '
                    '--source biorxiv for preprint JATS full text.'
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
        with tempfile.TemporaryDirectory(prefix='paperminer-download-') as download_dir:
            with tqdm(total=len(papers), desc='Downloading Papers', colour='#A020F0') as pbar:
                for paper in papers:
                    paper_summary = _download_paper(conn,
                                                    paper,
                                                    download_dir,
                                                    download_format,
                                                    sources,
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
    requested = set(sources)
    text_available = any(
        name in requested and _source_reachable(name, registry.TEXT, paper)
        for name in registry.names(registry.TEXT)
    )
    text_attempt_needed = (
        text_requested
        and text_available
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
                ok, source_or_error, abstract = _download_abstract(paper, sources)
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
            ok, text_source_or_error, _ = _download_text_from_sources(
                paper, text_filepath, sources)
            if ok:
                paper['text_path'] = ''
                paper['text_source'] = text_source_or_error
                _set_status(paper, 'text_download_status', 'succeeded')
                _store_downloaded_asset(conn, paper, text_filepath, role='text',
                                        source=text_source_or_error)
                summary['texts'] += 1
            else:
                _set_status(paper, 'text_download_status', 'failed', text_source_or_error)
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
                if source_or_error in registry.open_access_names() and source_url:
                    paper['pdf_url'] = source_url
                _set_status(paper, 'pdf_download_status', 'succeeded')
                _store_downloaded_asset(conn, paper, pdf_filepath, role='pdf', source=source_or_error)
                summary['pdfs'] += 1
                pdf_succeeded_from_oa = source_or_error in registry.open_access_names()
            else:
                _set_status(paper, 'pdf_download_status', 'failed', source_or_error)
        except Exception as e:
            _set_status(paper, 'pdf_download_status', 'failed', str(e))

    if (
        pdf_succeeded_from_oa
        and download_format == 'pdf'
        and not text_attempt_needed
        and existing_assets['text'] is None
        and 'elsevier' in requested
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
