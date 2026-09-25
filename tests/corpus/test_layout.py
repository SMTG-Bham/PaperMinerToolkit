"""Tests for provider-neutral scientific-document layout models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from paperminertoolkit.corpus.layout import (BoundingBox,
                                             CoordinateSpace,
                                             DocumentLayout,
                                             Figure,
                                             Graphic,
                                             LayoutProvenance,
                                             ReferenceSentence,
                                             Section,
                                             Table,
                                             TextBlock)


def test_bounding_boxes_normalize_coordinates_and_report_dimensions() -> None:
    """Normalize coordinate values and retain their explicit native space."""
    box = BoundingBox(2, 10, 20.5, 110, 70.5, 'pixels')

    assert box.page == 2
    assert box.coordinate_space is CoordinateSpace.PIXELS
    assert box.width == 100
    assert box.height == 50
    assert all(isinstance(value, float) for value in (box.x0, box.y0, box.x1, box.y1))
    assert BoundingBox(1, 0, 0, 1, 1).coordinate_space is CoordinateSpace.PDF_POINTS


@pytest.mark.parametrize('page', [0, -1, 1.5, True])
def test_bounding_boxes_require_positive_integer_pages(page: object) -> None:
    """Reject page values that cannot identify a one-based document page."""
    with pytest.raises(ValueError, match='positive one-based integer'):
        BoundingBox(page, 0, 0, 1, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ('coordinates', 'message'),
    [
        ((0, 0, float('inf'), 1), 'must be finite'),
        ((2, 0, 1, 1), 'must not precede'),
        ((0, 2, 1, 1), 'must not precede'),
    ],
)
def test_bounding_boxes_reject_invalid_coordinates(
    coordinates: tuple[float, float, float, float],
    message: str,
) -> None:
    """Reject non-finite and reversed coordinate ranges."""
    with pytest.raises(ValueError, match=message):
        BoundingBox(1, *coordinates)


def test_bounding_boxes_validate_coordinate_spaces_and_normalized_ranges() -> None:
    """Reject unknown spaces and normalized coordinates outside zero to one."""
    with pytest.raises(ValueError, match='coordinate_space must be one of'):
        BoundingBox(1, 0, 0, 1, 1, 'centimetres')
    with pytest.raises(ValueError, match='normalized coordinates'):
        BoundingBox(1, -0.1, 0, 1, 1, CoordinateSpace.NORMALIZED)

    normalized = BoundingBox(1, 0, 0.25, 1, 0.75, CoordinateSpace.NORMALIZED)
    assert normalized.coordinate_space is CoordinateSpace.NORMALIZED


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'source': ' ', 'document_format': 'jats', 'parser': 'jats'}, 'source'),
        ({'source': 'pubmed', 'document_format': ' ', 'parser': 'jats'}, 'document_format'),
        ({'source': 'pubmed', 'document_format': 'jats', 'parser': ' '}, 'parser'),
    ],
)
def test_layout_provenance_requires_complete_parser_context(
    kwargs: dict[str, str],
    message: str,
) -> None:
    """Require the provenance fields needed to interpret a layout."""
    with pytest.raises(ValueError, match=f'{message} must not be empty'):
        LayoutProvenance(**kwargs)


def test_layout_provenance_normalizes_labels() -> None:
    """Normalize format and parser labels while preserving the source name."""
    provenance = LayoutProvenance(
        source=' PubMed Central ',
        document_format=' JATS ',
        parser=' JATS-XML ',
        source_identifier=' PMC123 ',
    )

    assert provenance == LayoutProvenance('PubMed Central', 'jats', 'jats-xml', 'PMC123')


def test_document_layout_freezes_sequences_and_traverses_nested_sections() -> None:
    """Traverse immutable section and block sequences in document order."""
    introduction = TextBlock('Introduction text', reading_order=0, section_id='s1')
    methods = TextBlock(
        'Methods text',
        block_type='paragraph',
        reading_order=1,
        section_id='s1.1',
        boxes=[BoundingBox(2, 10, 10, 200, 40)],
    )
    child = Section('s1.1', 'Methods', blocks=[methods])
    root = Section('s1', 'Introduction', blocks=[introduction], children=[child])
    layout = DocumentLayout(
        'doi:10.1/example',
        LayoutProvenance('elsevier', 'elsevier-xml', 'elsevier-xml'),
        title='Example paper',
        sections=[root],
    )

    assert isinstance(layout.sections, tuple)
    assert isinstance(root.blocks, tuple)
    assert isinstance(root.children, tuple)
    assert isinstance(methods.boxes, tuple)
    assert [section.identifier for section in layout.iter_sections()] == ['s1', 's1.1']
    assert [block.text for block in layout.iter_text_blocks()] == [
        'Introduction text', 'Methods text',
    ]
    assert [section.identifier for section in root.iter_sections()] == ['s1', 's1.1']
    assert [block.text for block in root.iter_text_blocks()] == [
        'Introduction text', 'Methods text',
    ]
    with pytest.raises(FrozenInstanceError):
        layout.title = 'Changed'  # type: ignore[misc]


def test_figures_freeze_components_and_aggregate_source_pages() -> None:
    """Collect pages from figure, caption, graphic, and reference regions."""
    figure_box = BoundingBox(3, 10, 20, 300, 400)
    caption_box = BoundingBox(3, 10, 405, 300, 450)
    graphic = Graphic(
        identifier='g1',
        uri='https://example.test/figure.jpg',
        mime_type='image/jpeg',
        boxes=[BoundingBox(4, 0, 0, 100, 100, 'pixels')],
    )
    reference = ReferenceSentence(
        'Figure 1 shows the result.',
        section_id='results',
        boxes=[BoundingBox(2, 10, 10, 200, 30)],
    )
    figure = Figure(
        'fig1',
        label='Figure 1',
        caption='Example figure.',
        section_id='results',
        boxes=[figure_box],
        caption_boxes=[caption_box],
        graphics=[graphic],
        reference_sentences=[reference],
    )

    assert isinstance(figure.boxes, tuple)
    assert isinstance(figure.caption_boxes, tuple)
    assert isinstance(figure.graphics, tuple)
    assert isinstance(figure.reference_sentences, tuple)
    assert isinstance(graphic.boxes, tuple)
    assert isinstance(reference.boxes, tuple)
    assert figure.page_numbers == (2, 3, 4)
    assert Figure('empty').page_numbers == ()


def test_tables_freeze_components_and_aggregate_source_pages() -> None:
    """Retain serialized tables and collect their distinct source pages."""
    reference = ReferenceSentence(
        'Values are listed in Table 1.',
        boxes=[BoundingBox(5, 0, 0, 1, 1, 'normalized')],
    )
    table = Table(
        'table1',
        label='Table 1',
        caption='Measured values.',
        section_id='results',
        boxes=[BoundingBox(6, 20, 50, 500, 700)],
        caption_boxes=[BoundingBox(6, 20, 20, 500, 45)],
        content='<table><tr><td>1</td></tr></table>',
        content_format='jats-xml',
        reference_sentences=[reference],
    )
    layout = DocumentLayout(
        'paper:1',
        LayoutProvenance('pubmed', 'jats', 'jats-xml'),
        figures=[Figure('fig1')],
        tables=[table],
    )

    assert table.page_numbers == (5, 6)
    assert isinstance(table.boxes, tuple)
    assert isinstance(table.caption_boxes, tuple)
    assert isinstance(table.reference_sentences, tuple)
    assert Table('empty').page_numbers == ()
    assert isinstance(layout.figures, tuple)
    assert isinstance(layout.tables, tuple)
