"""Represent provider-neutral scientific-document layouts.

The immutable models in this module are shared contracts for structured XML,
repository JATS, derived TEI, and PDF geometry parsers. They retain native
coordinate systems and source provenance without depending on one provider's
markup vocabulary.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum


class CoordinateSpace(StrEnum):
    """Coordinate systems supported by document layout elements."""

    PDF_POINTS = 'pdf-points'
    PIXELS = 'pixels'
    NORMALIZED = 'normalized'


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Locate a rectangular region on one document page.

    Parameters
    ----------
    page : int
        One-based document page number.
    x0, y0, x1, y1 : float
        Native corner coordinates with the origin defined by
        ``coordinate_space``.
    coordinate_space : CoordinateSpace or str, default=CoordinateSpace.PDF_POINTS
        Units and origin convention used by the coordinates.

    Raises
    ------
    ValueError
        If the page or coordinates do not describe a finite, ordered box, or
        normalized coordinates fall outside zero to one.
    """

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: CoordinateSpace | str = CoordinateSpace.PDF_POINTS

    def __post_init__(self) -> None:
        """Normalize and validate the bounding box."""
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1:
            raise ValueError('page must be a positive one-based integer')
        coordinates = tuple(float(value) for value in (self.x0, self.y0, self.x1, self.y1))
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError('bounding-box coordinates must be finite')
        x0, y0, x1, y1 = coordinates
        if x1 < x0 or y1 < y0:
            raise ValueError('bounding-box end coordinates must not precede start coordinates')
        try:
            coordinate_space = CoordinateSpace(self.coordinate_space)
        except ValueError as error:
            choices = ', '.join(space.value for space in CoordinateSpace)
            raise ValueError(f'coordinate_space must be one of: {choices}') from error
        if coordinate_space is CoordinateSpace.NORMALIZED and any(
            value < 0 or value > 1 for value in coordinates
        ):
            raise ValueError('normalized coordinates must be between 0 and 1')
        object.__setattr__(self, 'x0', x0)
        object.__setattr__(self, 'y0', y0)
        object.__setattr__(self, 'x1', x1)
        object.__setattr__(self, 'y1', y1)
        object.__setattr__(self, 'coordinate_space', coordinate_space)

    @property
    def width(self) -> float:
        """Return the box width in its native coordinate space."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Return the box height in its native coordinate space."""
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class LayoutProvenance:
    """Describe where a parsed document layout came from.

    Parameters
    ----------
    source : str
        Provider, publisher, repository, or local acquisition source.
    document_format : str
        Structured or rendered input format, such as ``"jats"`` or ``"pdf"``.
    parser : str
        Parser or detection method that produced the layout.
    source_identifier : str, optional
        Provider-native document identifier or source asset identifier.

    Raises
    ------
    ValueError
        If source, document format, or parser is empty.
    """

    source: str
    document_format: str
    parser: str
    source_identifier: str = ''

    def __post_init__(self) -> None:
        """Normalize required provenance labels."""
        values = {
            'source': self.source.strip(),
            'document_format': self.document_format.strip().lower(),
            'parser': self.parser.strip().lower(),
            'source_identifier': self.source_identifier.strip(),
        }
        for field_name in ('source', 'document_format', 'parser'):
            if not values[field_name]:
                raise ValueError(f'{field_name} must not be empty')
        for field_name, value in values.items():
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class TextBlock:
    """Represent one ordered block of document text.

    Parameters
    ----------
    text : str
        Block text.
    block_type : str, default='paragraph'
        Open block classification such as paragraph, heading, or list item.
    reading_order : int or None, optional
        Parser-assigned reading position.
    section_id : str, optional
        Identifier of the containing section.
    boxes : tuple[BoundingBox, ...], optional
        Source regions contributing to this block.
    """

    text: str
    block_type: str = 'paragraph'
    reading_order: int | None = None
    section_id: str = ''
    boxes: tuple[BoundingBox, ...] = ()

    def __post_init__(self) -> None:
        """Freeze sequence inputs supplied by parsers."""
        object.__setattr__(self, 'boxes', tuple(self.boxes))


@dataclass(frozen=True, slots=True)
class Section:
    """Represent a nested document section.

    Parameters
    ----------
    identifier : str
        Stable provider or parser-assigned section identifier.
    title : str, optional
        Section heading.
    blocks : tuple[TextBlock, ...], optional
        Text blocks directly contained by this section.
    children : tuple[Section, ...], optional
        Nested subsections in document order.
    """

    identifier: str
    title: str = ''
    blocks: tuple[TextBlock, ...] = ()
    children: tuple[Section, ...] = ()

    def __post_init__(self) -> None:
        """Freeze block and child-section sequences."""
        object.__setattr__(self, 'blocks', tuple(self.blocks))
        object.__setattr__(self, 'children', tuple(self.children))

    def iter_sections(self) -> Iterator[Section]:
        """Yield this section and its descendants in document order."""
        yield self
        for child in self.children:
            yield from child.iter_sections()

    def iter_text_blocks(self) -> Iterator[TextBlock]:
        """Yield this section's blocks followed by descendant blocks."""
        yield from self.blocks
        for child in self.children:
            yield from child.iter_text_blocks()


@dataclass(frozen=True, slots=True)
class ReferenceSentence:
    """Represent prose that refers to a figure or table.

    Parameters
    ----------
    text : str
        Sentence text.
    section_id : str, optional
        Identifier of the containing section.
    boxes : tuple[BoundingBox, ...], optional
        Source regions contributing to the sentence.
    """

    text: str
    section_id: str = ''
    boxes: tuple[BoundingBox, ...] = ()

    def __post_init__(self) -> None:
        """Freeze bounding-box inputs."""
        object.__setattr__(self, 'boxes', tuple(self.boxes))


@dataclass(frozen=True, slots=True)
class Graphic:
    """Represent one graphical resource belonging to a figure.

    Parameters
    ----------
    identifier : str, optional
        Provider or parser-assigned graphic identifier.
    uri : str, optional
        Source URL, relative reference, or local asset identifier.
    mime_type : str, optional
        Declared media type when available.
    boxes : tuple[BoundingBox, ...], optional
        Graphic locations in the source document.
    """

    identifier: str = ''
    uri: str = ''
    mime_type: str = ''
    boxes: tuple[BoundingBox, ...] = ()

    def __post_init__(self) -> None:
        """Freeze bounding-box inputs."""
        object.__setattr__(self, 'boxes', tuple(self.boxes))


def _page_numbers(*box_groups: tuple[BoundingBox, ...]) -> tuple[int, ...]:
    """Collect sorted unique page numbers from bounding-box groups."""
    return tuple(sorted({box.page for boxes in box_groups for box in boxes}))


@dataclass(frozen=True, slots=True)
class Figure:
    """Represent a figure and its caption, graphics, and textual references.

    Parameters
    ----------
    identifier : str
        Stable provider or parser-assigned figure identifier.
    label : str, optional
        Display label such as ``"Figure 3"``.
    caption : str, optional
        Complete figure caption.
    section_id : str, optional
        Identifier of the section containing the figure.
    boxes : tuple[BoundingBox, ...], optional
        Complete figure-region locations.
    caption_boxes : tuple[BoundingBox, ...], optional
        Caption text locations.
    graphics : tuple[Graphic, ...], optional
        Graphical resources associated with the figure.
    reference_sentences : tuple[ReferenceSentence, ...], optional
        Paper sentences that refer to the figure.
    """

    identifier: str
    label: str = ''
    caption: str = ''
    section_id: str = ''
    boxes: tuple[BoundingBox, ...] = ()
    caption_boxes: tuple[BoundingBox, ...] = ()
    graphics: tuple[Graphic, ...] = ()
    reference_sentences: tuple[ReferenceSentence, ...] = ()

    def __post_init__(self) -> None:
        """Freeze figure component sequences."""
        object.__setattr__(self, 'boxes', tuple(self.boxes))
        object.__setattr__(self, 'caption_boxes', tuple(self.caption_boxes))
        object.__setattr__(self, 'graphics', tuple(self.graphics))
        object.__setattr__(self, 'reference_sentences', tuple(self.reference_sentences))

    @property
    def page_numbers(self) -> tuple[int, ...]:
        """Return every page referenced by this figure in ascending order."""
        groups = [self.boxes, self.caption_boxes]
        groups.extend(graphic.boxes for graphic in self.graphics)
        groups.extend(reference.boxes for reference in self.reference_sentences)
        return _page_numbers(*groups)


@dataclass(frozen=True, slots=True)
class Table:
    """Represent a table and its source content and textual references.

    Parameters
    ----------
    identifier : str
        Stable provider or parser-assigned table identifier.
    label : str, optional
        Display label such as ``"Table 2"``.
    caption : str, optional
        Complete table caption.
    section_id : str, optional
        Identifier of the section containing the table.
    boxes : tuple[BoundingBox, ...], optional
        Complete table-region locations.
    caption_boxes : tuple[BoundingBox, ...], optional
        Caption text locations.
    content : str, optional
        Serialized or flattened table content.
    content_format : str, optional
        Format of ``content``, such as XML, HTML, or text.
    reference_sentences : tuple[ReferenceSentence, ...], optional
        Paper sentences that refer to the table.
    """

    identifier: str
    label: str = ''
    caption: str = ''
    section_id: str = ''
    boxes: tuple[BoundingBox, ...] = ()
    caption_boxes: tuple[BoundingBox, ...] = ()
    content: str = ''
    content_format: str = ''
    reference_sentences: tuple[ReferenceSentence, ...] = ()

    def __post_init__(self) -> None:
        """Freeze table component sequences."""
        object.__setattr__(self, 'boxes', tuple(self.boxes))
        object.__setattr__(self, 'caption_boxes', tuple(self.caption_boxes))
        object.__setattr__(self, 'reference_sentences', tuple(self.reference_sentences))

    @property
    def page_numbers(self) -> tuple[int, ...]:
        """Return every page referenced by this table in ascending order."""
        groups = [self.boxes, self.caption_boxes]
        groups.extend(reference.boxes for reference in self.reference_sentences)
        return _page_numbers(*groups)


@dataclass(frozen=True, slots=True)
class DocumentLayout:
    """Represent one parsed scientific document.

    Parameters
    ----------
    document_id : str
        Stable document identifier, normally a corpus paper identifier.
    provenance : LayoutProvenance
        Source and parser information for this layout.
    title : str, optional
        Document title.
    sections : tuple[Section, ...], optional
        Top-level sections in document order.
    figures : tuple[Figure, ...], optional
        Document-level figures linked to sections by identifier.
    tables : tuple[Table, ...], optional
        Document-level tables linked to sections by identifier.
    """

    document_id: str
    provenance: LayoutProvenance
    title: str = ''
    sections: tuple[Section, ...] = ()
    figures: tuple[Figure, ...] = ()
    tables: tuple[Table, ...] = ()

    def __post_init__(self) -> None:
        """Freeze document component sequences."""
        object.__setattr__(self, 'sections', tuple(self.sections))
        object.__setattr__(self, 'figures', tuple(self.figures))
        object.__setattr__(self, 'tables', tuple(self.tables))

    def iter_sections(self) -> Iterator[Section]:
        """Yield all sections depth-first in document order."""
        for section in self.sections:
            yield from section.iter_sections()

    def iter_text_blocks(self) -> Iterator[TextBlock]:
        """Yield all text blocks depth-first in document order."""
        for section in self.sections:
            yield from section.iter_text_blocks()
