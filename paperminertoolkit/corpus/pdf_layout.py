"""Detect and render figure regions in PDF-only scientific papers.

The detector uses PyMuPDF's native text and drawing geometry. It deliberately
does not attempt panel segmentation: captions are associated with nearby raster
images or vector-drawing clusters, while uncertain matches retain their caption
provenance and use complete-page rendering.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

from paperminertoolkit.corpus.documents import _load_fitz
from paperminertoolkit.corpus.layout import (
    BoundingBox,
    DocumentLayout,
    Figure,
    LayoutProvenance,
    Table,
)

_CAPTION_RE = re.compile(
    r'^\s*(?P<label>(?P<kind>fig(?:ure)?\.?|table)\s+[A-Za-z]?\d+[A-Za-z]?)'
    r'\s*(?:[.:\-–—]\s*|\s+)(?P<caption>.*)$',
    re.IGNORECASE,
)


def _block_text(block: Mapping[str, Any]) -> str:
    """Flatten one PyMuPDF text block into normalized text."""
    pieces = []
    for line in block.get('lines', []):
        if not isinstance(line, Mapping):
            continue
        for span in line.get('spans', []):
            if isinstance(span, Mapping) and span.get('text'):
                pieces.append(str(span['text']))
    return ' '.join(' '.join(pieces).split())


def _text_blocks(page: Any, page_number: int) -> list[tuple[str, BoundingBox]]:
    """Read ordered text blocks and their PDF-point geometry from one page."""
    payload = page.get_text('dict')
    raw_blocks = payload.get('blocks', []) if isinstance(payload, Mapping) else []
    blocks = []
    for block in raw_blocks:
        if not isinstance(block, Mapping) or block.get('type', 0) != 0:
            continue
        bbox = block.get('bbox')
        text = _block_text(block)
        if not text or not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        blocks.append((text, BoundingBox(page_number, *bbox)))
    return sorted(blocks, key=lambda item: (item[1].y0, item[1].x0))


def _horizontal_overlap(first: BoundingBox, second: BoundingBox) -> float:
    """Return horizontal overlap as a fraction of the narrower box."""
    overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    denominator = min(first.width, second.width)
    return overlap / denominator if denominator > 0 else 0.0


def _join_caption_blocks(
    blocks: Sequence[tuple[str, BoundingBox]],
    max_gap: float,
) -> list[tuple[str, str, str, tuple[BoundingBox, ...]]]:
    """Join wrapped caption blocks while respecting column boundaries."""
    captions = []
    consumed: set[int] = set()
    for index, (text, box) in enumerate(blocks):
        if index in consumed:
            continue
        match = _CAPTION_RE.match(text)
        if match is None:
            continue
        label = match.group('label').strip()
        kind = 'table' if match.group('kind').lower().startswith('table') else 'figure'
        parts = [match.group('caption').strip()]
        boxes = [box]
        previous = box
        for next_index in range(index + 1, min(len(blocks), index + 9)):
            next_text, next_box = blocks[next_index]
            gap = next_box.y0 - previous.y1
            overlap = _horizontal_overlap(box, next_box)
            if gap > max_gap:
                break
            if next_index in consumed or overlap < 0.4:
                continue
            if _CAPTION_RE.match(next_text) or gap < -2:
                break
            parts.append(next_text)
            boxes.append(next_box)
            consumed.add(next_index)
            previous = next_box
        caption = ' '.join(part for part in parts if part).strip()
        captions.append((kind, label, caption, tuple(boxes)))
    return captions


def _rect_box(page_number: int, value: object) -> BoundingBox | None:
    """Convert a PyMuPDF rectangle-like value into a valid bounding box."""
    try:
        coordinates = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(coordinates) != 4:
        return None
    box = BoundingBox(page_number, *coordinates)
    return box if box.width >= 20 and box.height >= 20 else None


def _visual_regions(page: Any, page_number: int) -> tuple[BoundingBox, ...]:
    """Collect raster-image boxes and vector-drawing clusters from one page."""
    regions: list[BoundingBox] = []
    try:
        image_info = page.get_image_info(xrefs=True)
    except (AttributeError, TypeError):
        image_info = []
    for image in image_info:
        if isinstance(image, Mapping):
            box = _rect_box(page_number, image.get('bbox'))
            if box is not None:
                regions.append(box)
    try:
        clusters = page.cluster_drawings()
    except (AttributeError, TypeError, ValueError):
        clusters = []
    for cluster in clusters:
        box = _rect_box(page_number, cluster)
        if box is not None and box not in regions:
            regions.append(box)
    return tuple(regions)


def _associated_region(
    caption_boxes: Sequence[BoundingBox],
    candidates: Iterable[BoundingBox],
    page_height: float,
    kind: str,
    minimum_confidence: float,
) -> BoundingBox | None:
    """Choose the most plausible nearby visual region for one caption."""
    caption = caption_boxes[0]
    caption_end = caption_boxes[-1]
    max_distance = max(page_height * 0.35, 1.0)
    scored = []
    for candidate in candidates:
        overlap = max(_horizontal_overlap(box, candidate) for box in caption_boxes)
        if overlap < 0.2:
            continue
        if candidate.y1 <= caption.y0:
            distance = caption.y0 - candidate.y1
            direction_bonus = 1.0 if kind == 'figure' else 0.85
        elif candidate.y0 >= caption_end.y1:
            distance = candidate.y0 - caption_end.y1
            direction_bonus = 1.0 if kind == 'table' else 0.85
        else:
            continue
        if distance > max_distance:
            continue
        confidence = overlap * (1 - distance / max_distance) * direction_bonus
        scored.append((confidence, -distance, candidate))
    if not scored:
        return None
    confidence, _, candidate = max(scored, key=lambda item: (item[0], item[1]))
    return candidate if confidence >= minimum_confidence else None


def detect_pdf_layout(
    pdf_path: str | PathLike[str],
    document_id: str = '',
    *,
    caption_gap: float = 12.0,
    minimum_confidence: float = 0.2,
) -> DocumentLayout:
    """Detect figures, tables, captions, and source geometry in a PDF.

    Parameters
    ----------
    pdf_path : str or os.PathLike[str]
        Source PDF.
    document_id : str, optional
        Corpus paper identifier. The filename stem is used when omitted.
    caption_gap : float, default=12.0
        Maximum vertical gap, in PDF points, for joining wrapped caption blocks.
    minimum_confidence : float, default=0.2
        Minimum geometry-association score. Lower-confidence elements retain
        caption boxes but have no region box, which signals page fallback.

    Returns
    -------
    DocumentLayout
        Provider-neutral PDF layout with figures and tables in page order.

    Raises
    ------
    ValueError
        If caption or confidence thresholds are invalid.
    RuntimeError
        If PyMuPDF is unavailable.
    """
    if caption_gap < 0:
        raise ValueError('caption_gap must be non-negative')
    if not 0 <= minimum_confidence <= 1:
        raise ValueError('minimum_confidence must be between 0 and 1')
    path = os.fspath(pdf_path)
    identifier = document_id.strip() or Path(path).stem
    figures: list[Figure] = []
    tables: list[Table] = []
    fitz = _load_fitz()
    with fitz.open(path) as document:
        for page_index in range(len(document)):
            page = document[page_index]
            page_number = page_index + 1
            blocks = _text_blocks(page, page_number)
            regions = _visual_regions(page, page_number)
            page_height = float(page.rect.height)
            for kind, label, caption, caption_boxes in _join_caption_blocks(
                blocks,
                caption_gap,
            ):
                region = _associated_region(
                    caption_boxes,
                    regions,
                    page_height,
                    kind,
                    minimum_confidence,
                )
                boxes = (region,) if region is not None else ()
                if kind == 'figure':
                    figures.append(Figure(
                        identifier=f'pdf-figure-{len(figures) + 1}',
                        label=label,
                        caption=caption,
                        boxes=boxes,
                        caption_boxes=caption_boxes,
                    ))
                else:
                    tables.append(Table(
                        identifier=f'pdf-table-{len(tables) + 1}',
                        label=label,
                        caption=caption,
                        boxes=boxes,
                        caption_boxes=caption_boxes,
                    ))
    return DocumentLayout(
        document_id=identifier,
        provenance=LayoutProvenance(
            source='pdf',
            document_format='pdf',
            parser='pymupdf-layout',
            source_identifier=path,
        ),
        figures=tuple(figures),
        tables=tuple(tables),
    )


def _render_box(
    page: Any,
    box: BoundingBox | None,
    padding: float,
    fitz: Any,
    caption_boxes: Sequence[BoundingBox] = (),
) -> object:
    """Build a page-clamped PyMuPDF clip rectangle that excludes the caption.

    Padding is clamped on whichever side the caption sits so the rendered
    crop never bleeds into caption text, even when the caption sits closer
    to the region than ``padding``.
    """
    if box is None:
        return page.rect
    top = max(float(page.rect.y0), box.y0 - padding)
    bottom = min(float(page.rect.y1), box.y1 + padding)
    if caption_boxes:
        caption_top = min(caption.y0 for caption in caption_boxes)
        caption_bottom = max(caption.y1 for caption in caption_boxes)
        if caption_top >= box.y1:
            bottom = min(bottom, caption_top)
        elif caption_bottom <= box.y0:
            top = max(top, caption_bottom)
    return fitz.Rect(
        max(float(page.rect.x0), box.x0 - padding),
        top,
        min(float(page.rect.x1), box.x1 + padding),
        bottom,
    )


def render_pdf_figures(
    pdf_path: str | PathLike[str],
    layout: DocumentLayout,
    output_dir: str | PathLike[str],
    *,
    prefix: str | None = None,
    padding: float = 12.0,
    dpi: int = 200,
) -> list[str]:
    """Render detected figure regions, falling back to complete pages.

    Parameters
    ----------
    pdf_path : str or os.PathLike[str]
        Source PDF corresponding to ``layout``.
    layout : DocumentLayout
        PDF layout returned by :func:`detect_pdf_layout`.
    output_dir : str or os.PathLike[str]
        Destination directory.
    prefix : str or None, optional
        Output filename prefix. The PDF stem is used when omitted.
    padding : float, default=12.0
        PDF-point padding around confident regions.
    dpi : int, default=200
        Render resolution.

    Returns
    -------
    list[str]
        Rendered PNG paths in figure order. An uncertain figure produces a
        complete-page image using its caption page.

    Raises
    ------
    ValueError
        If padding or resolution is invalid, or a figure has no page provenance.
    """
    if padding < 0:
        raise ValueError('padding must be non-negative')
    if dpi < 1:
        raise ValueError('dpi must be positive')
    path = os.fspath(pdf_path)
    destination = os.fspath(output_dir)
    os.makedirs(destination, exist_ok=True)
    stem = prefix or Path(path).stem
    fitz = _load_fitz()
    saved = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(path) as document:
        for index, figure in enumerate(layout.figures, start=1):
            region = figure.boxes[0] if figure.boxes else None
            pages = figure.page_numbers
            if not pages:
                raise ValueError(f'figure {figure.identifier!r} has no page provenance')
            page_number = region.page if region is not None else pages[0]
            page = document[page_number - 1]
            clip = _render_box(page, region, padding, fitz, figure.caption_boxes)
            pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            filename = f'{stem}_figure-{index}_page-{page_number}.png'
            filepath = os.path.join(destination, filename)
            pixmap.save(filepath)
            saved.append(filepath)
    return saved
