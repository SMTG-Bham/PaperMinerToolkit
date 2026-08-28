"""Tests for PDF figure/caption detection and layout-aware rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from paperminertoolkit.corpus import pdf_layout
from paperminertoolkit.corpus.layout import BoundingBox, DocumentLayout, Figure, LayoutProvenance


def _write_layout_pdf(path: Path, *, include_drawing: bool = True) -> None:
    """Create a deterministic two-column PDF with figure and table captions."""
    fitz = pytest.importorskip('fitz')
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    if include_drawing:
        page.draw_rect(fitz.Rect(40, 80, 270, 250), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
        page.draw_rect(fitz.Rect(330, 330, 560, 520), color=(0, 0, 0))
    page.insert_textbox(
        fitz.Rect(40, 260, 270, 282),
        'Figure 1. Conductivity map for the sample.',
        fontsize=9,
    )
    page.insert_textbox(
        fitz.Rect(40, 282, 270, 304),
        'Colours identify the measured phases.',
        fontsize=9,
    )
    page.insert_textbox(
        fitz.Rect(330, 300, 560, 322),
        'Table 1. Measured conductivity values.',
        fontsize=9,
    )
    document.save(path)
    document.close()


def test_join_caption_blocks_preserves_double_column_reading_groups() -> None:
    """Join wrapped captions without absorbing the neighbouring column."""
    blocks = [
        ('Figure 1. Left caption.', BoundingBox(1, 20, 100, 280, 110)),
        ('Figure 2. Right caption.', BoundingBox(1, 320, 100, 580, 110)),
        ('Left continuation.', BoundingBox(1, 20, 112, 280, 122)),
        ('Right continuation.', BoundingBox(1, 320, 112, 580, 122)),
    ]

    captions = pdf_layout._join_caption_blocks(blocks, max_gap=4)

    assert [caption[2] for caption in captions] == [
        'Left caption. Left continuation.',
        'Right caption. Right continuation.',
    ]
    assert all(len(caption[3]) == 2 for caption in captions)


def test_pdf_layout_helpers_ignore_malformed_blocks_and_unusable_geometry() -> None:
    """Exercise defensive parsing for provider-shaped PyMuPDF payloads."""
    class TextPage:
        """Return prepared text dictionaries."""

        def __init__(self, payload: object) -> None:
            """Store the prepared text payload."""
            self.payload = payload

        def get_text(self, mode: str) -> object:
            """Return the prepared payload."""
            assert mode == 'dict'
            return self.payload

    payload = {'blocks': [
        'bad block',
        {'type': 1, 'bbox': (0, 0, 10, 10)},
        {'type': 0, 'bbox': (0, 0, 100, 20), 'lines': ['bad line']},
        {'type': 0, 'bbox': (0, 0, 100),
         'lines': [{'spans': [{'text': 'bad bbox'}]}]},
        {'type': 0, 'bbox': (0, 20, 100, 40),
         'lines': [{'spans': [{'text': '  valid '}, {'missing': 'text'}]}]},
    ]}
    assert pdf_layout._text_blocks(TextPage(payload), 1)[0][0] == 'valid'
    assert pdf_layout._text_blocks(TextPage([]), 1) == []

    assert pdf_layout._rect_box(1, None) is None
    assert pdf_layout._rect_box(1, (0, 0, 10)) is None
    assert pdf_layout._rect_box(1, (0, 0, 10, 10)) is None

    class BrokenGeometryPage:
        """Raise from optional PyMuPDF geometry methods."""

        def get_image_info(self, *, xrefs: bool) -> object:
            """Reject image inspection."""
            raise TypeError(xrefs)

        def cluster_drawings(self) -> object:
            """Reject drawing inspection."""
            raise ValueError('broken drawings')

    assert pdf_layout._visual_regions(BrokenGeometryPage(), 1) == ()

    class GeometryPage:
        """Return duplicate, malformed, raster, and vector regions."""

        def get_image_info(self, *, xrefs: bool) -> list[object]:
            """Return one usable raster box and one malformed entry."""
            assert xrefs is True
            return [{'bbox': (0, 0, 100, 100)}, 'bad image']

        def cluster_drawings(self) -> list[object]:
            """Return a duplicate and one distinct vector box."""
            return [(0, 0, 100, 100), (120, 0, 220, 100)]

    assert len(pdf_layout._visual_regions(GeometryPage(), 2)) == 2


def test_caption_and_region_scoring_rejects_implausible_neighbours() -> None:
    """Stop captions correctly and reject weak, overlapping, or distant regions."""
    caption = BoundingBox(1, 20, 100, 280, 112)
    blocks = [
        ('Body text.', BoundingBox(1, 20, 80, 280, 92)),
        ('Figure 1. Caption.', caption),
        ('Figure 2. Next.', BoundingBox(1, 20, 113, 280, 125)),
        ('Figure 3. Last.', BoundingBox(1, 20, 200, 280, 212)),
    ]
    captions = pdf_layout._join_caption_blocks(blocks, max_gap=4)
    assert [item[1] for item in captions] == ['Figure 1', 'Figure 2', 'Figure 3']

    inside = BoundingBox(1, 20, 102, 280, 160)
    distant = BoundingBox(1, 20, 400, 280, 500)
    wrong_column = BoundingBox(1, 400, 20, 580, 90)
    assert pdf_layout._associated_region(
        [caption], [inside, distant, wrong_column], 600, 'figure', 0.1,
    ) is None
    nearby = BoundingBox(1, 20, 20, 280, 90)
    assert pdf_layout._associated_region(
        [caption], [nearby], 600, 'table', 0.99,
    ) is None


def test_detect_pdf_layout_finds_captioned_regions_in_two_columns(tmp_path: Path) -> None:
    """Associate figure and table captions with nearby vector regions."""
    pdf_path = tmp_path / 'layout.pdf'
    _write_layout_pdf(pdf_path)

    layout = pdf_layout.detect_pdf_layout(pdf_path, document_id='paper:1')

    assert layout.document_id == 'paper:1'
    assert layout.provenance.parser == 'pymupdf-layout'
    assert len(layout.figures) == 1
    assert layout.figures[0].label == 'Figure 1'
    assert layout.figures[0].caption == (
        'Conductivity map for the sample. Colours identify the measured phases.'
    )
    assert layout.figures[0].boxes[0].page == 1
    assert len(layout.tables) == 1
    assert layout.tables[0].label == 'Table 1'
    assert layout.tables[0].boxes[0].page == 1


def test_render_pdf_figures_clips_confident_regions_and_falls_back_to_pages(
    tmp_path: Path,
) -> None:
    """Render detected geometry and use a full page when no region is trusted."""
    pdf_path = tmp_path / 'layout.pdf'
    _write_layout_pdf(pdf_path)
    layout = pdf_layout.detect_pdf_layout(pdf_path)
    detected_path = Path(pdf_layout.render_pdf_figures(
        pdf_path,
        layout,
        tmp_path / 'detected',
        padding=5,
        dpi=72,
    )[0])
    with Image.open(detected_path) as image:
        detected_width = image.width

    uncertain = DocumentLayout(
        'paper:uncertain',
        LayoutProvenance('pdf', 'pdf', 'pymupdf-layout'),
        figures=[Figure(
            'f1',
            label='Figure 1',
            caption='Uncertain figure.',
            caption_boxes=[BoundingBox(1, 40, 260, 270, 282)],
        )],
    )
    page_path = Path(pdf_layout.render_pdf_figures(
        pdf_path,
        uncertain,
        tmp_path / 'fallback',
        dpi=72,
    )[0])
    with Image.open(page_path) as image:
        page_width = image.width

    assert detected_path.name == 'layout_figure-1_page-1.png'
    assert detected_width < page_width
    assert page_width == 600


def test_pdf_layout_validates_options_and_page_provenance(tmp_path: Path) -> None:
    """Reject invalid thresholds, rendering options, and unlocated figures."""
    pdf_path = tmp_path / 'layout.pdf'
    _write_layout_pdf(pdf_path, include_drawing=False)
    with pytest.raises(ValueError, match='caption_gap'):
        pdf_layout.detect_pdf_layout(pdf_path, caption_gap=-1)
    with pytest.raises(ValueError, match='minimum_confidence'):
        pdf_layout.detect_pdf_layout(pdf_path, minimum_confidence=2)

    layout = DocumentLayout(
        'paper:1',
        LayoutProvenance('pdf', 'pdf', 'pymupdf-layout'),
        figures=[Figure('unlocated')],
    )
    with pytest.raises(ValueError, match='padding'):
        pdf_layout.render_pdf_figures(pdf_path, layout, tmp_path, padding=-1)
    with pytest.raises(ValueError, match='dpi'):
        pdf_layout.render_pdf_figures(pdf_path, layout, tmp_path, dpi=0)
    with pytest.raises(ValueError, match='no page provenance'):
        pdf_layout.render_pdf_figures(pdf_path, layout, tmp_path)
