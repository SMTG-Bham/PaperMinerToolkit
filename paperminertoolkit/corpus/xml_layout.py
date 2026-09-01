"""Normalize publisher and repository XML into document layout models."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import replace
from urllib.parse import urlsplit

from paperminertoolkit.corpus.layout import (
    BoundingBox,
    DocumentLayout,
    Figure,
    Graphic,
    LayoutProvenance,
    ReferenceSentence,
    Section,
    Table,
    TextBlock,
)

_SECTION_TAGS = {'sec', 'section', 'div'}
_PARAGRAPH_TAGS = {'p', 'para', 'simple-para'}
_FIGURE_TAGS = {'fig', 'figure'}
_TABLE_TAGS = {'table-wrap', 'table'}
_GRAPHIC_TAGS = {'graphic', 'media', 'link'}
_REFERENCE_TAGS = {'xref', 'cross-ref', 'ref'}


def _local_name(tag: str) -> str:
    """Return a lower-case XML name without namespace or prefix."""
    return tag.rsplit('}', 1)[-1].rsplit(':', 1)[-1].lower()


def _text(element: ET.Element | None) -> str:
    """Flatten one XML subtree into normalized text."""
    if element is None:
        return ''
    return ' '.join(''.join(element.itertext()).split())


def _attribute(element: ET.Element, *names: str) -> str:
    """Read the first non-empty attribute by namespace-insensitive name."""
    wanted = {name.lower() for name in names}
    for key, value in element.attrib.items():
        if _local_name(key) in wanted and str(value).strip():
            return str(value).strip()
    return ''


def _first_descendant(element: ET.Element, names: Iterable[str]) -> ET.Element | None:
    """Return the first descendant whose local tag is in ``names``."""
    wanted = set(names)
    return next((node for node in element.iter() if _local_name(node.tag) in wanted), None)


def _direct_child(element: ET.Element, names: Iterable[str]) -> ET.Element | None:
    """Return the first direct child whose local tag is in ``names``."""
    wanted = set(names)
    return next((child for child in element if _local_name(child.tag) in wanted), None)


def _document_title(root: ET.Element) -> str:
    """Find an article title without accidentally choosing a section title."""
    article_title = _first_descendant(root, {'article-title'})
    if article_title is not None:
        return _text(article_title)
    for container_name in ('coredata', 'title-group', 'titlestmt', 'head'):
        container = _first_descendant(root, {container_name})
        if container is None:
            continue
        title = _first_descendant(container, {'title'})
        if title is not None:
            return _text(title)
    return ''


def _caption(element: ET.Element) -> str:
    """Extract a figure or table caption while excluding its display label."""
    caption = _first_descendant(element, {'caption', 'figdesc'})
    return _text(caption)


def _label(element: ET.Element) -> str:
    """Extract a direct or nested display label."""
    return _text(_first_descendant(element, {'label', 'head'}))


def _coordinate_boxes(element: ET.Element) -> tuple[BoundingBox, ...]:
    """Decode GROBID ``page,x,y,width,height`` coordinate groups."""
    value = _attribute(element, 'coords', 'coordinates')
    if not value:
        return ()
    boxes = []
    for group in value.split(';'):
        try:
            page_text, x_text, y_text, width_text, height_text = group.split(',')
            page = int(page_text)
            x, y = float(x_text), float(y_text)
            width, height = float(width_text), float(height_text)
            boxes.append(BoundingBox(page, x, y, x + width, y + height))
        except (TypeError, ValueError):
            continue
    return tuple(boxes)


def _graphic_resources(element: ET.Element) -> tuple[Graphic, ...]:
    """Extract linked graphical resources from a figure element."""
    graphics = []
    seen = set()
    for node in element.iter():
        if _local_name(node.tag) not in _GRAPHIC_TAGS:
            continue
        uri = _attribute(node, 'href', 'locator', 'src', 'url')
        if not uri or uri in seen:
            continue
        seen.add(uri)
        mime_type = _attribute(node, 'mime-type', 'mimetype')
        subtype = _attribute(node, 'mime-subtype')
        if mime_type and subtype and '/' not in mime_type:
            mime_type = f'{mime_type}/{subtype}'
        graphics.append(Graphic(
            identifier=_attribute(node, 'id') or f'graphic-{len(graphics) + 1}',
            uri=uri,
            mime_type=mime_type,
            boxes=_coordinate_boxes(node),
        ))
    return tuple(graphics)


def _parse_figure(element: ET.Element, section_id: str, index: int) -> Figure:
    """Normalize one figure, tolerating absent identifiers and captions."""
    graphics = _graphic_resources(element)
    boxes = _coordinate_boxes(element) or tuple(
        box for graphic in graphics for box in graphic.boxes
    )
    return Figure(
        identifier=_attribute(element, 'id') or f'figure-{index}',
        label=_label(element),
        caption=_caption(element),
        section_id=section_id,
        boxes=boxes,
        graphics=graphics,
    )


def _parse_table(element: ET.Element, section_id: str, index: int, format_name: str) -> Table:
    """Normalize one table and retain its serialized source content."""
    return Table(
        identifier=_attribute(element, 'id') or f'table-{index}',
        label=_label(element),
        caption=_caption(element),
        section_id=section_id,
        boxes=_coordinate_boxes(element),
        content=ET.tostring(element, encoding='unicode'),
        content_format=format_name,
    )


def _collect_objects(
    node: ET.Element,
    section_id: str,
    format_name: str,
    figures: list[Figure],
    tables: list[Table],
) -> None:
    """Collect figures and tables recursively without duplicating wrappers."""
    name = _local_name(node.tag)
    current_section = section_id
    if name in _SECTION_TAGS:
        current_section = _attribute(node, 'id') or section_id
    if name in _FIGURE_TAGS and _attribute(node, 'type').lower() == 'table':
        tables.append(_parse_table(node, current_section, len(tables) + 1, format_name))
        return
    if name in _FIGURE_TAGS:
        figures.append(_parse_figure(node, current_section, len(figures) + 1))
        return
    if name in _TABLE_TAGS:
        tables.append(_parse_table(node, current_section, len(tables) + 1, format_name))
        return
    for child in node:
        _collect_objects(child, current_section, format_name, figures, tables)


def _section_identifier(element: ET.Element, path: tuple[int, ...]) -> str:
    """Return a source identifier or stable structural fallback."""
    return _attribute(element, 'id') or 'section-' + '-'.join(str(part) for part in path)


def _parse_section(
    element: ET.Element,
    path: tuple[int, ...],
    reading_order: list[int],
) -> Section:
    """Parse one nested section while keeping direct paragraphs in order."""
    identifier = _section_identifier(element, path)
    title = _text(_direct_child(element, {'title', 'section-title', 'head'}))
    blocks = []
    children = []
    child_number = 0
    for child in element:
        name = _local_name(child.tag)
        if name in _SECTION_TAGS:
            child_number += 1
            children.append(_parse_section(child, (*path, child_number), reading_order))
        elif name in _PARAGRAPH_TAGS:
            text = _text(child)
            if text:
                blocks.append(TextBlock(
                    text=text,
                    reading_order=reading_order[0],
                    section_id=identifier,
                ))
                reading_order[0] += 1
    return Section(identifier, title=title, blocks=tuple(blocks), children=tuple(children))


def _sections(root: ET.Element) -> tuple[Section, ...]:
    """Parse top-level sections and preserve unsectioned body paragraphs."""
    body = _first_descendant(root, {'body'})
    if body is None:
        body = _first_descendant(root, {'originaltext'})
    if body is None:
        body = root
    reading_order = [0]
    sections = []
    loose_blocks = []
    section_number = 0

    def top_sections(container: ET.Element) -> Iterable[ET.Element]:
        """Yield sections not nested inside another section."""
        for node in container:
            if _local_name(node.tag) in _SECTION_TAGS:
                yield node
            else:
                yield from top_sections(node)

    for child in body:
        name = _local_name(child.tag)
        if name in _SECTION_TAGS:
            section_number += 1
            sections.append(_parse_section(child, (section_number,), reading_order))
        elif name in _PARAGRAPH_TAGS:
            text = _text(child)
            if text:
                loose_blocks.append(TextBlock(text, reading_order=reading_order[0],
                                              section_id='body'))
                reading_order[0] += 1
    if not sections:
        for section in top_sections(body):
            section_number += 1
            sections.append(_parse_section(section, (section_number,), reading_order))
    if loose_blocks:
        sections.insert(0, Section('body', blocks=tuple(loose_blocks)))
    return tuple(sections)


def _reference_targets(paragraph: ET.Element) -> tuple[str, ...]:
    """Return figure/table identifiers referenced by one paragraph."""
    targets = []
    for node in paragraph.iter():
        if _local_name(node.tag) not in _REFERENCE_TAGS:
            continue
        reference_type = _attribute(node, 'ref-type', 'ref-type-name', 'type').lower()
        target = _attribute(node, 'rid', 'refid', 'target')
        if target and (not reference_type or reference_type in {'fig', 'figure', 'table'}):
            targets.extend(part.lstrip('#') for part in re.split(r'\s+', target) if part)
    return tuple(dict.fromkeys(targets))


def _reference_sentences(root: ET.Element) -> dict[str, tuple[ReferenceSentence, ...]]:
    """Collect paragraphs that explicitly cross-reference figures or tables."""
    references: dict[str, list[ReferenceSentence]] = {}

    def walk(node: ET.Element, section_id: str = '') -> None:
        """Walk prose while carrying its nearest section identifier."""
        name = _local_name(node.tag)
        if name in _SECTION_TAGS:
            section_id = _attribute(node, 'id') or section_id
        if name in _PARAGRAPH_TAGS:
            text = _text(node)
            if text:
                for target in _reference_targets(node):
                    references.setdefault(target, []).append(ReferenceSentence(
                        text=text,
                        section_id=section_id,
                    ))
            return
        for child in node:
            walk(child, section_id)

    walk(root)
    return {target: tuple(items) for target, items in references.items()}


def _attach_references(
    figures: Iterable[Figure],
    tables: Iterable[Table],
    references: dict[str, tuple[ReferenceSentence, ...]],
) -> tuple[tuple[Figure, ...], tuple[Table, ...]]:
    """Attach collected reference prose to matching immutable objects."""
    linked_figures = tuple(replace(
        figure,
        reference_sentences=references.get(figure.identifier, ()),
    ) for figure in figures)
    linked_tables = tuple(replace(
        table,
        reference_sentences=references.get(table.identifier, ()),
    ) for table in tables)
    return linked_figures, linked_tables


def _parse_xml_layout(
    content: str,
    document_id: str,
    source: str,
    source_identifier: str,
    format_name: str,
    parser_name: str,
) -> DocumentLayout:
    """Parse one supported XML vocabulary into the common layout contract."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise ValueError(f'malformed {format_name} document: {error}') from error
    figures: list[Figure] = []
    tables: list[Table] = []
    _collect_objects(root, '', format_name, figures, tables)
    figures_out, tables_out = _attach_references(
        figures,
        tables,
        _reference_sentences(root),
    )
    return DocumentLayout(
        document_id=document_id,
        provenance=LayoutProvenance(source, format_name, parser_name, source_identifier),
        title=_document_title(root),
        sections=_sections(root),
        figures=figures_out,
        tables=tables_out,
    )


def parse_jats_layout(
    content: str,
    document_id: str,
    *,
    source: str = 'repository',
    source_identifier: str = '',
    source_url: str = '',
) -> DocumentLayout:
    """Parse repository or publisher JATS into a document layout.

    Parameters
    ----------
    content : str
        Complete JATS XML document.
    document_id : str
        Owning corpus paper identifier.
    source : str, default='repository'
        Repository or publisher that supplied the JATS.
    source_identifier : str, optional
        Provider-native document identifier.
    source_url : str, optional
        URL the document was served from. bioRxiv and medRxiv name their
        figures with internal tokens that resolve against nothing, so their
        real image URLs are derived from this together with each figure's
        display slug. Other repositories are unaffected.

    Returns
    -------
    DocumentLayout
        Normalized section, figure, table, graphic, and reference structure.

    Raises
    ------
    ValueError
        If the document is not well-formed XML.
    """
    layout = _parse_xml_layout(
        content,
        document_id,
        source,
        source_identifier,
        'jats',
        'jats-xml',
    )
    if not source_url:
        return layout
    root = ET.fromstring(content)  # already validated well-formed above
    figures = _resolve_rxiv_graphics(layout.figures, source_url, _figure_slugs(root))
    return replace(layout, figures=figures) if figures != layout.figures else layout


_ELSEVIER_OBJECT_URL = 'https://api.elsevier.com/content/object/eid/{eid}-{locator}.jpg'
# bioRxiv and medRxiv reference their figures as internal tokens such as
# "339747v4_fig1", which resolve against nothing. Their content sites serve the
# image under the figure's HighWire slug instead, beneath a path that repeats
# the archive name and the posting date the source document was published at:
#     .../content/early/2019/05/10/339747.source.xml   (the JATS)
#     .../content/biorxiv/early/2019/05/10/339747/F1.large.jpg   (its Figure 1)
# Everything needed is therefore in the source URL and the document itself.
_RXIV_SOURCE_PATH = re.compile(
    r'^/content/early/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/(?P<slug>[^/]+?)'
    r'(?:\.source\.xml)?$'
)
_RXIV_ARCHIVES = ('biorxiv', 'medrxiv')


def _highwire_slug(element: ET.Element) -> str:
    """Read a HighWire display slug, such as ``"F1"``, from one element.

    Parameters
    ----------
    element : ET.Element
        Figure or table element to inspect.

    Returns
    -------
    str
        Slug from a ``sub-type="slug"`` object identifier, falling back to a
        prefixed ``id`` attribute, or an empty string when neither is present.
    """
    for child in element:
        if _local_name(child.tag) == 'object-id' and _attribute(child, 'sub-type') == 'slug':
            return _text(child)
    for key, value in element.attrib.items():
        # A prefixed or namespaced id, never the element's own plain "id",
        # which is the publisher identifier rather than the display slug.
        if key != 'id' and _local_name(key) == 'id' and str(value).strip():
            return str(value).strip()
    return ''


def _rxiv_figure_url(source_url: str, slug: str) -> str:
    """Build a bioRxiv or medRxiv image URL from the JATS source URL.

    Parameters
    ----------
    source_url : str
        URL the archive served the JATS document from.
    slug : str
        Figure's HighWire slug, such as ``"F1"``.

    Returns
    -------
    str
        Image URL, or an empty string when the source URL is not one of these
        archives' content paths or the slug is missing.
    """
    if not slug:
        return ''
    parts = urlsplit(source_url)
    host = (parts.hostname or '').lower()
    archive = next((name for name in _RXIV_ARCHIVES if host.endswith(f'{name}.org')), '')
    match = _RXIV_SOURCE_PATH.match(parts.path)
    if not archive or match is None:
        return ''
    return (f'{parts.scheme}://{parts.netloc}/content/{archive}/early/'
            f'{match["year"]}/{match["month"]}/{match["day"]}/{match["slug"]}/{slug}.large.jpg')


def _figure_slugs(root: ET.Element) -> dict[str, str]:
    """Map each figure's own identifier to its HighWire display slug.

    Parameters
    ----------
    root : ET.Element
        Parsed document root.

    Returns
    -------
    dict[str, str]
        Figure identifier to slug, for figures that declare both.
    """
    slugs = {}
    for node in root.iter():
        if _local_name(node.tag) not in _FIGURE_TAGS:
            continue
        identifier = _attribute(node, 'id')
        slug = _highwire_slug(node)
        if identifier and slug:
            slugs[identifier] = slug
    return slugs


def _resolve_rxiv_graphics(
    figures: tuple[Figure, ...],
    source_url: str,
    slugs: Mapping[str, str],
) -> tuple[Figure, ...]:
    """Point bioRxiv and medRxiv figure graphics at their real image URLs.

    Their JATS names each graphic with an internal token that resolves against
    nothing, so a graphic that is not already an absolute URL is replaced by
    the archive's image URL for that figure's slug.
    """
    resolved = []
    for figure in figures:
        url = _rxiv_figure_url(source_url, slugs.get(figure.identifier, ''))
        graphics = tuple(
            replace(graphic, uri=url)
            if url and not graphic.uri.startswith(('http://', 'https://'))
            else graphic
            for graphic in figure.graphics
        )
        resolved.append(replace(figure, graphics=graphics) if graphics != figure.graphics else figure)
    return tuple(resolved)


def _is_bare_elsevier_locator(value: str) -> bool:
    """Return whether a ``ce:link`` locator is an internal reference token.

    Native Elsevier XML embeds figure graphics as bare tokens such as
    ``"gr1"`` rather than a resolvable path or URL. Retrieving the image
    requires combining this token with the article's own ``eid`` through
    Elsevier's object-retrieval endpoint; it is not a relative reference that
    ``urljoin`` against the article URL can resolve.
    """
    return bool(value) and '/' not in value and '.' not in value


def _resolve_elsevier_object_urls(figures: tuple[Figure, ...], eid: str) -> tuple[Figure, ...]:
    """Rewrite bare Elsevier locator tokens into object-retrieval URLs."""
    if not eid:
        return figures
    resolved = []
    for figure in figures:
        graphics = tuple(
            replace(graphic, uri=_ELSEVIER_OBJECT_URL.format(eid=eid, locator=graphic.uri))
            if _is_bare_elsevier_locator(graphic.uri) else graphic
            for graphic in figure.graphics
        )
        resolved.append(replace(figure, graphics=graphics) if graphics != figure.graphics else figure)
    return tuple(resolved)


def parse_elsevier_layout(
    content: str,
    document_id: str,
    *,
    source_identifier: str = '',
) -> DocumentLayout:
    """Parse native Elsevier article XML into a document layout.

    Bare ``ce:link`` locator tokens (Elsevier's native graphic reference
    form, such as ``"gr1"``) are rewritten into absolute object-retrieval
    URLs using the article's own ``eid``, since they are not paths that
    resolve against the article endpoint.

    Parameters
    ----------
    content : str
        Complete Elsevier full-article XML document.
    document_id : str
        Owning corpus paper identifier.
    source_identifier : str, optional
        Elsevier article identifier or endpoint.

    Returns
    -------
    DocumentLayout
        Normalized section, figure, table, graphic, and reference structure,
        with graphic URIs resolvable to real image downloads where the
        document's own ``eid`` is available.

    Raises
    ------
    ValueError
        If the document is not well-formed XML.
    """
    layout = _parse_xml_layout(
        content,
        document_id,
        'elsevier',
        source_identifier,
        'elsevier-xml',
        'elsevier-xml',
    )
    root = ET.fromstring(content)  # already validated well-formed above
    eid = _text(_first_descendant(root, {'eid'}))
    figures = _resolve_elsevier_object_urls(layout.figures, eid)
    return replace(layout, figures=figures) if figures != layout.figures else layout


def parse_tei_layout(
    content: str,
    document_id: str,
    *,
    source_identifier: str = '',
) -> DocumentLayout:
    """Parse OpenAlex GROBID TEI into a PDF-derived document layout.

    Parameters
    ----------
    content : str
        Complete GROBID TEI XML document.
    document_id : str
        Owning corpus paper identifier.
    source_identifier : str, optional
        OpenAlex work identifier or content URL.

    Returns
    -------
    DocumentLayout
        PDF-derived sections, figures, captions, tables, coordinates, and
        references normalized into the common layout model.

    Raises
    ------
    ValueError
        If the document is malformed or its root is not TEI.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise ValueError(f'malformed tei document: {error}') from error
    if _local_name(root.tag) != 'tei':
        raise ValueError('OpenAlex GROBID content must have a TEI root element')
    return _parse_xml_layout(
        content,
        document_id,
        'openalex-grobid',
        source_identifier,
        'tei',
        'grobid-tei',
    )
