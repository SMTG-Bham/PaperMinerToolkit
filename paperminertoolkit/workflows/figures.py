"""Download figure images referenced by stored structured documents.

The workflow parses preserved JATS, Elsevier XML, or GROBID TEI layouts,
resolves their graphic references, validates downloaded image payloads, and
stores them as content-addressed corpus assets. Per-figure source provenance
makes an interrupted run resumable without weakening global blob
deduplication.

Papers with no structured document can instead contribute figures detected
from PDF geometry, which are stored through the same corpus asset model so
downstream extraction treats both origins identically.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from paperminertoolkit.corpus import database
from paperminertoolkit.corpus.layout import DocumentLayout, Figure, Graphic
from paperminertoolkit.corpus.pdf_layout import detect_pdf_layout, render_pdf_figures
from paperminertoolkit.corpus.xml_layout import (
    parse_elsevier_layout,
    parse_jats_layout,
    parse_tei_layout,
)
from paperminertoolkit.providers import base as provider
from paperminertoolkit.providers import elsevier

FIGURE_MIN_INTERVAL = 0.2
DEFAULT_MAX_IMAGE_BYTES = 50 * 1024 * 1024
PDF_LAYOUT_SOURCE = 'pdf-layout'
FIGURE_LIMITER = provider.RateLimiter(FIGURE_MIN_INTERVAL)
_ALLOWED_MIME_TYPES = frozenset({
    'image/gif',
    'image/jpeg',
    'image/png',
    'image/svg+xml',
    'image/tiff',
    'image/webp',
})
_MIME_EXTENSIONS = {
    'image/gif': '.gif',
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/svg+xml': '.svg',
    'image/tiff': '.tiff',
    'image/webp': '.webp',
}
_PARSERS = {
    'jats': parse_jats_layout,
    'elsevier-xml': parse_elsevier_layout,
    'tei': parse_tei_layout,
}


@dataclass(frozen=True, slots=True)
class FigureDownloadSummary:
    """Summarize one structured-figure download run.

    Parameters
    ----------
    downloaded : int
        Image links stored successfully.
    skipped : int
        Completed figure URLs skipped during a resumed run.
    failed : int
        Graphic references that could not be parsed or downloaded.
    failures : tuple[str, ...]
        Human-readable per-document or per-graphic failure messages.
    """

    downloaded: int
    skipped: int
    failed: int
    failures: tuple[str, ...] = ()


def _safe_image_url(uri: str, base_url: str) -> str:
    """Resolve a graphic reference and reject unsafe URL forms."""
    reference = uri.strip()
    base = base_url.strip()
    if not reference:
        raise ValueError('graphic reference is empty')
    resolved = urljoin(base, reference)
    parts = urlsplit(resolved)
    if parts.scheme.lower() not in {'http', 'https'}:
        raise ValueError('figure URLs must use HTTP or HTTPS')
    if parts.username is not None or parts.password is not None:
        raise ValueError('figure URLs must not contain credentials')
    hostname = (parts.hostname or '').rstrip('.').lower()
    if not hostname:
        raise ValueError('figure URL has no host')
    if hostname == 'localhost' or hostname.endswith(('.localhost', '.local')):
        raise ValueError('figure URL resolves to a local host')
    try:
        address = ipaddress.ip_address(hostname.strip('[]'))
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError('figure URL must use a public network address')
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, parts.query, ''))


def _sniff_mime_type(content: bytes) -> str:
    """Identify supported image bytes from their signature."""
    if content.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if content.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if content.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if content.startswith((b'II*\x00', b'MM\x00*')):
        return 'image/tiff'
    if len(content) >= 12 and content.startswith(b'RIFF') and content[8:12] == b'WEBP':
        return 'image/webp'
    prefix = content[:4096].lstrip().lower()
    if prefix.startswith(b'<?xml'):
        closing = prefix.find(b'?>')
        prefix = prefix[closing + 2:].lstrip() if closing >= 0 else prefix
    if prefix.startswith(b'<svg'):
        return 'image/svg+xml'
    return ''


def _validated_mime_type(content: bytes, declared_types: tuple[str, ...]) -> str:
    """Return a validated image MIME type or reject an unexpected payload."""
    sniffed = _sniff_mime_type(content)
    if not sniffed:
        raise ValueError('response is not a supported image payload')
    normalized = {
        value.split(';', 1)[0].strip().lower().replace('image/jpg', 'image/jpeg')
        for value in declared_types
        if value.strip()
    }
    declared_images = normalized & _ALLOWED_MIME_TYPES
    if declared_images and sniffed not in declared_images:
        expected = ', '.join(sorted(declared_images))
        raise ValueError(f'image signature does not match declared MIME type {expected}')
    if sniffed == 'image/svg+xml':
        lowered = content.lower()
        if b'<script' in lowered or b'<!entity' in lowered:
            raise ValueError('active or entity-bearing SVG payloads are not accepted')
    return sniffed


def _response_content(response: provider.ResponseLike, max_bytes: int) -> bytes:
    """Read and size-check a response body exposed by ``requests``."""
    length_header = response.headers.get('Content-Length', '')
    if length_header:
        try:
            declared_length = int(length_header)
        except ValueError:
            declared_length = 0
        if declared_length > max_bytes:
            raise ValueError(f'image exceeds the {max_bytes}-byte limit')
    content = getattr(response, 'content', b'')
    if not isinstance(content, bytes) or not content:
        raise ValueError('image response is empty')
    if len(content) > max_bytes:
        raise ValueError(f'image exceeds the {max_bytes}-byte limit')
    return content


def _request_headers(source: str) -> dict[str, str]:
    """Build image request headers, including Elsevier credentials when configured."""
    if source.lower() == 'elsevier':
        try:
            return elsevier.api_headers(elsevier.configured_api_key(), accept='image/*')
        except ValueError:
            pass
    headers = provider.default_headers()
    headers['Accept'] = 'image/*'
    return headers


def _asset_filename(figure: Figure, graphic: Graphic, url: str, mime_type: str) -> str:
    """Build a stable filename that cannot collide across graphic URLs."""
    identifier = graphic.identifier or figure.identifier
    safe_identifier = re.sub(r'[^A-Za-z0-9._-]+', '-', identifier).strip('.-') or 'figure'
    source_name = PurePosixPath(unquote(urlsplit(url).path)).name
    stem = PurePosixPath(source_name).stem if source_name else safe_identifier
    safe_stem = re.sub(r'[^A-Za-z0-9._-]+', '-', stem).strip('.-') or safe_identifier
    url_digest = hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]
    return f'{safe_identifier}-{safe_stem}-{url_digest}{_MIME_EXTENSIONS[mime_type]}'


def _layout_from_asset(asset: Mapping[str, Any], paper_id: str) -> DocumentLayout:
    """Parse one stored structured-document asset into the common layout model."""
    metadata = asset['metadata']
    document_format = str(metadata.get('document_format', '')).lower()
    parser = _PARSERS.get(document_format)
    if parser is None:
        raise ValueError(f'unsupported structured-document format: {document_format or "missing"}')
    content = asset['content']
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    source_identifier = str(metadata.get('source_identifier', ''))
    if document_format == 'jats':
        return parser(
            content,
            paper_id,
            source=str(asset['source']),
            source_identifier=source_identifier,
        )
    return parser(content, paper_id, source_identifier=source_identifier)


def _download_graphic(
    graphic: Graphic,
    *,
    requested_url: str,
    source: str,
    session: provider.HTTPClient | None,
    max_bytes: int,
) -> tuple[bytes, str, str]:
    """Download and validate one graphic, including its final redirect URL."""
    response = provider.request(
        requested_url,
        label=f'{source} figure',
        limiter=FIGURE_LIMITER,
        headers=_request_headers(source),
        session=session,
        missing_ok=False,
    )
    if response is None:  # pragma: no cover - missing_ok=False guarantees this
        raise RuntimeError(f'{source} figure returned no response')
    final_url = _safe_image_url(str(getattr(response, 'url', requested_url)), requested_url)
    content = _response_content(response, max_bytes)
    mime_type = _validated_mime_type(
        content,
        (response.headers.get('Content-Type', ''), graphic.mime_type),
    )
    return content, final_url, mime_type


def download_structured_figures(
    db_path: str | PathLike[str],
    paper_id: str,
    *,
    force: bool = False,
    session: provider.HTTPClient | None = None,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> FigureDownloadSummary:
    """Download every figure image referenced by a paper's structured documents.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        PaperMinerToolkit corpus database.
    paper_id : str
        Corpus paper identifier to process.
    force : bool, default=False
        Request and replace completed figure links instead of resuming around
        them.
    session : provider.HTTPClient or None, optional
        Injectable HTTP client. Defaults to :mod:`requests`.
    max_bytes : int, default=52428800
        Maximum accepted uncompressed response size per image.

    Returns
    -------
    FigureDownloadSummary
        Counts and failure messages for the run.

    Raises
    ------
    ValueError
        If the paper is absent or ``max_bytes`` is not positive.
    """
    if max_bytes <= 0:
        raise ValueError('max_bytes must be positive')
    downloaded = 0
    skipped = 0
    failures: list[str] = []
    with database.connect(db_path) as conn:
        paper = next((row for row in database.paper_rows(conn) if row['paper_id'] == paper_id), None)
        if paper is None:
            raise ValueError(f'paper not found in corpus: {paper_id}')
        assets = database.get_structured_documents(conn, paper_id)
        for asset in assets:
            try:
                layout = _layout_from_asset(asset, paper_id)
            except (UnicodeDecodeError, ValueError) as error:
                failures.append(f'{asset["source"]} structured document: {error}')
                continue
            metadata = asset['metadata']
            base_url = str(metadata.get('source_url', ''))
            source = str(asset['source'])
            license_name = str(metadata.get('license') or paper.get('license') or '')
            for figure in layout.figures:
                for graphic in figure.graphics:
                    try:
                        requested_url = _safe_image_url(graphic.uri, base_url)
                        if not force and database.has_figure_asset(
                            conn,
                            paper_id,
                            requested_url,
                            figure_id=figure.identifier,
                        ):
                            skipped += 1
                            continue
                        content, final_url, mime_type = _download_graphic(
                            graphic,
                            requested_url=requested_url,
                            source=source,
                            session=session,
                            max_bytes=max_bytes,
                        )
                        database.add_figure_asset(
                            conn,
                            paper,
                            content,
                            figure_id=figure.identifier,
                            caption=figure.caption,
                            source=source,
                            source_url=final_url,
                            mime_type=mime_type,
                            original_filename=_asset_filename(
                                figure, graphic, requested_url, mime_type,
                            ),
                            license=license_name,
                            metadata={
                                'requested_url': requested_url,
                                'graphic_id': graphic.identifier,
                                'figure_label': figure.label,
                                'section_id': figure.section_id,
                                'reference_sentences': [
                                    reference.text for reference in figure.reference_sentences
                                ],
                            },
                        )
                        downloaded += 1
                    except (RuntimeError, ValueError) as error:
                        label = figure.label or figure.identifier
                        failures.append(f'{label} ({graphic.uri or "missing URI"}): {error}')
    return FigureDownloadSummary(downloaded, skipped, len(failures), tuple(failures))


def pdf_layout_figure_url(paper_id: str, figure_id: str) -> str:
    """Build the stable identity recorded for one PDF-detected figure.

    Figures detected from PDF geometry have no download URL, so a
    deterministic value derived from the paper and figure identifiers takes
    its place. This keeps resume checks and provenance identical to
    downloaded structured figures.

    Parameters
    ----------
    paper_id : str
        Owning corpus paper identifier.
    figure_id : str
        Figure identifier assigned by the PDF layout detector.

    Returns
    -------
    str
        Stable ``pdf-layout://`` identity for the figure.
    """
    return f'{PDF_LAYOUT_SOURCE}://{paper_id}/{figure_id}'


def store_pdf_layout_figures(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    pdf_path: str | PathLike[str],
    *,
    force: bool = False,
    padding: float = 12.0,
    dpi: int = 200,
    minimum_confidence: float = 0.2,
) -> FigureDownloadSummary:
    """Detect figures in a PDF and store them as corpus figure assets.

    This is the fallback for papers with no structured document. Detected
    regions are rendered once and stored through the same content-addressed
    asset model as downloaded structured figures, so an interrupted run
    resumes per figure and later extraction treats both origins alike.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : Mapping[str, Any]
        Owning corpus paper row.
    pdf_path : str or os.PathLike[str]
        Source PDF for the paper.
    force : bool, default=False
        Replace figures already stored for this paper instead of resuming
        around them.
    padding : float, default=12.0
        PDF-point padding applied around a confident figure region.
    dpi : int, default=200
        Render resolution for detected regions.
    minimum_confidence : float, default=0.2
        Minimum geometry-association score for a detected region.

    Returns
    -------
    FigureDownloadSummary
        Counts and failure messages for the run.

    Raises
    ------
    ValueError
        If the paper has no identifier.
    """
    paper_id = str(paper.get('paper_id') or '').strip()
    if not paper_id:
        raise ValueError('paper must carry a paper_id')
    layout = detect_pdf_layout(pdf_path, document_id=paper_id, minimum_confidence=minimum_confidence)
    if not layout.figures:
        return FigureDownloadSummary(0, 0, 0)
    stored = 0
    skipped = 0
    failures: list[str] = []
    render_dir = tempfile.mkdtemp(prefix='pmt-pdf-layout-')
    try:
        rendered = render_pdf_figures(pdf_path, layout, render_dir, padding=padding, dpi=dpi)
        license_name = str(paper.get('license') or '')
        for figure, image_path in zip(layout.figures, rendered):
            source_url = pdf_layout_figure_url(paper_id, figure.identifier)
            try:
                if not force and database.has_figure_asset(
                    conn,
                    paper_id,
                    source_url,
                    figure_id=figure.identifier,
                ):
                    skipped += 1
                    continue
                with open(image_path, 'rb') as image_file:
                    content = image_file.read()
                database.add_figure_asset(
                    conn,
                    paper,
                    content,
                    figure_id=figure.identifier,
                    caption=figure.caption,
                    source=PDF_LAYOUT_SOURCE,
                    source_url=source_url,
                    mime_type='image/png',
                    original_filename=Path(image_path).name,
                    license=license_name,
                    metadata={
                        'requested_url': source_url,
                        'figure_label': figure.label,
                        'section_id': figure.section_id,
                        'page_numbers': list(figure.page_numbers),
                        'region_detected': bool(figure.boxes),
                    },
                )
                stored += 1
            except (OSError, ValueError) as error:
                label = figure.label or figure.identifier
                failures.append(f'{label}: {error}')
    finally:
        shutil.rmtree(render_dir, ignore_errors=True)
    return FigureDownloadSummary(stored, skipped, len(failures), tuple(failures))
