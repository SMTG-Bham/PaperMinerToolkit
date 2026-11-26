"""Render document tables as Markdown text for language-model extraction.

Plain-text assets are built without tables, and PDF text extraction flattens
table rows into a stream of cells. These helpers recover each table as a
Markdown grid, so a model can read a value together with its row and column.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from os import PathLike
from typing import Any

from paperminertoolkit._xml import element_text, local_name
from paperminertoolkit.corpus.layout import BoundingBox, DocumentLayout, Table
from paperminertoolkit.corpus.xml_layout import parse_elsevier_layout, parse_jats_layout, parse_tei_layout

TABLES_HEADING = '=== TABLES ==='
_PARSERS: dict[str, Callable[..., DocumentLayout]] = {
    'jats': parse_jats_layout,
    'elsevier-xml': parse_elsevier_layout,
    'tei': parse_tei_layout,
}
STRUCTURED_TABLE_FORMATS = frozenset(_PARSERS)
_ROW_TAGS = {'row', 'tr'}
_CELL_TAGS = {'entry', 'td', 'th', 'cell'}
_HEAD_TAGS = {'thead'}
_FOOTNOTE_TAGS = {'table-wrap-foot', 'table-footnote', 'legend'}
_SKIP_TAGS = {'caption', 'label', 'title', 'colgroup'} | _FOOTNOTE_TAGS


def _cell(text: str | None) -> str:
    """Collapse whitespace and escape Markdown column separators."""
    return ' '.join((text or '').split()).replace('|', '\\|')


def _span(element: ET.Element, *names: str) -> int:
    """Read a positive integer span attribute, defaulting to one."""
    for name in names:
        value = element.get(name)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    return 1


def _column_names(table: ET.Element) -> dict[str, int]:
    """Map CALS ``colspec`` names to zero-based column indexes."""
    names: dict[str, int] = {}
    for spec in (node for node in table.iter() if local_name(node.tag) == 'colspec'):
        name = spec.get('colname')
        number = spec.get('colnum')
        if name:
            names[name] = int(number) - 1 if number and number.isdigit() else len(names)
    return names


def _source_rows(node: ET.Element, in_head: bool = False) -> Iterator[tuple[ET.Element, bool]]:
    """Yield table rows in document order with whether they are header rows."""
    for child in node:
        name = local_name(child.tag)
        if name in _SKIP_TAGS:
            continue
        if name in _ROW_TAGS:
            yield child, in_head or child.get('role') == 'label'
        else:
            yield from _source_rows(child, in_head or name in _HEAD_TAGS)


def _grid(table: ET.Element) -> list[tuple[list[str], bool]]:
    """Expand row and column spans into a rectangular grid of cell texts."""
    columns = _column_names(table)
    carried: dict[int, tuple[str, int]] = {}
    rows: list[tuple[list[str], bool]] = []
    for row, is_head in _source_rows(table):
        cells: dict[int, str] = {}
        column = 0

        def take_carried(index: int) -> None:
            """Fill one column still covered by a row-spanning cell from an earlier row."""
            text, remaining = carried.pop(index)
            cells[index] = text
            if remaining > 1:
                carried[index] = (text, remaining - 1)

        def fill_carried() -> None:
            """Fill the covered columns from the current position onwards."""
            nonlocal column
            while column in carried:
                take_carried(column)
                column += 1

        for entry in (child for child in row if local_name(child.tag) in _CELL_TAGS):
            fill_carried()
            start = columns.get(entry.get('namest') or entry.get('colname') or '', column)
            end = columns.get(entry.get('nameend') or '', start)
            width = max(end - start + 1, _span(entry, 'colspan', 'cols'))
            height = _span(entry, 'rowspan', 'rows')
            if entry.get('morerows', '').isdigit():
                height = int(entry.get('morerows', '0')) + 1
            text = element_text(entry)
            for offset in range(width):
                cells[start + offset] = text
                if height > 1:
                    carried[start + offset] = (text, height - 1)
            column = start + width
        for index in sorted(index for index in carried if index >= column):
            take_carried(index)
        if cells:
            width = max(cells) + 1
            rows.append(([cells.get(index, '') for index in range(width)], is_head))
    return rows


def _merge_header(rows: list[list[str]]) -> list[str]:
    """Join stacked header rows column by column without repeating words."""
    width = max(len(row) for row in rows)
    merged = []
    for index in range(width):
        parts: list[str] = []
        for row in rows:
            text = row[index] if index < len(row) else ''
            if text and text not in parts:
                parts.append(text)
        merged.append(' '.join(parts))
    return merged


def markdown_table(rows: list[list[str]], header_rows: int = 1) -> str:
    """Format a cell grid as a Markdown table.

    Parameters
    ----------
    rows : list[list[str]]
        Table rows, each a list of cell texts.
    header_rows : int, default=1
        Number of leading rows merged into the Markdown header.

    Returns
    -------
    str
        Markdown table, or an empty string when there are no non-empty cells.
    """
    rows = [[_cell(cell) for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ''
    header_rows = min(max(header_rows, 1), len(rows))
    header = _merge_header(rows[:header_rows])
    body = rows[header_rows:]
    width = max(len(header), *(len(row) for row in body)) if body else len(header)
    lines = [header + [''] * (width - len(header)), ['---'] * width]
    lines += [row + [''] * (width - len(row)) for row in body]
    return '\n'.join('| ' + ' | '.join(row) + ' |' for row in lines)


def _footnotes(table: ET.Element) -> str:
    """Collect table footnotes and legends that explain symbols or methods."""
    notes = [
        element_text(node) for node in table.iter() if local_name(node.tag) in _FOOTNOTE_TAGS and node is not table
    ]
    return ' '.join(note for note in notes if note)


def table_markdown(table: Table) -> str:
    """Render one parsed XML table with its label, caption and footnotes.

    Parameters
    ----------
    table : Table
        Table parsed from a structured document, with its XML in ``content``.

    Returns
    -------
    str
        Markdown block, or an empty string when the table has no cell text.
    """
    try:
        element = ET.fromstring(table.content)
    except ET.ParseError:
        return ''
    grid = _grid(element)
    header_rows = sum(1 for _, is_head in grid if is_head) or 1
    body = markdown_table([cells for cells, _ in grid], header_rows)
    if not body:
        return ''
    label = next((element_text(child) for child in element if local_name(child.tag) == 'label'), '')
    title = ' '.join(part for part in (label or 'Table', table.caption) if part)
    notes = _footnotes(element)
    return '\n'.join(part for part in (title, body, f'Notes: {notes}' if notes else '') if part)


def structured_tables_text(content: str | bytes, document_format: str, document_id: str = '') -> str:
    """Render every table of a structured XML document as Markdown.

    Parameters
    ----------
    content : str or bytes
        Complete structured document.
    document_format : str
        Stored format name: ``"jats"``, ``"elsevier-xml"`` or ``"tei"``.
    document_id : str, optional
        Paper identifier recorded in the parsed layout.

    Returns
    -------
    str
        Tables separated by blank lines, or an empty string for unsupported
        formats, malformed XML or documents without tables.
    """
    parser = _PARSERS.get(document_format.strip().lower())
    if parser is None:
        return ''
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    try:
        layout = parser(content, document_id)
    except ValueError:
        return ''
    blocks = [table_markdown(table) for table in layout.tables]
    return '\n\n'.join(block for block in blocks if block)


def _words_in(words: list[tuple[Any, ...]], box: tuple[float, float, float, float] | None) -> str:
    """Join the words whose centres fall inside a table cell, in reading order.

    Cell text is rebuilt from positioned words rather than taken from the
    content stream, because some publisher fonts draw decimal points after
    the digits (``"204 ."`` for ``"2.04"``).
    """
    if box is None:
        return ''
    x0, y0, x1, y1 = box
    inside = [
        word[4] for word in words if x0 <= (word[0] + word[2]) / 2 <= x1 and y0 <= (word[1] + word[3]) / 2 <= y1
    ]
    return ' '.join(inside)


def pdf_tables_text(path: str | PathLike[str], skip_regions: Sequence[BoundingBox] = ()) -> str:
    """Detect ruled tables in a PDF and render them as Markdown.

    Parameters
    ----------
    path : str or os.PathLike[str]
        PDF file to scan.
    skip_regions : Sequence[BoundingBox], optional
        Regions, such as figures, whose detected tables are left out. A table
        is skipped when its centre lies in a region on the same page; PyMuPDF
        can report gridded figures such as heat maps as tables.

    Returns
    -------
    str
        Tables in page order, each headed ``Table (page n)``, or an empty
        string when none are found or the PDF cannot be read.
    """
    import pymupdf

    blocks = []
    try:
        with pymupdf.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                found: Any = page.find_tables()
                if not found.tables:
                    continue
                words = page.get_text('words', sort=True)
                for table in found.tables:
                    x0, y0, x1, y1 = table.bbox
                    centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
                    if any(region.page == page_number and region.x0 <= centre_x <= region.x1
                           and region.y0 <= centre_y <= region.y1 for region in skip_regions):
                        continue
                    rows = [[_words_in(words, cell) for cell in row.cells] for row in table.rows]
                    body = markdown_table(rows)
                    if body and len(rows) > 1:
                        blocks.append(f'Table (page {page_number})\n{body}')
    except (RuntimeError, ValueError):
        return ''
    return '\n\n'.join(blocks)


def append_tables(text: str, tables: str) -> str:
    """Append a Markdown tables section to document text when tables exist.

    Parameters
    ----------
    text : str
        Document text.
    tables : str
        Rendered tables, possibly empty.

    Returns
    -------
    str
        ``text`` followed by a ``=== TABLES ===`` section, or ``text`` unchanged.
    """
    if not tables.strip():
        return text
    return f'{text.rstrip()}\n\n{TABLES_HEADING}\n\n{tables.strip()}\n'
