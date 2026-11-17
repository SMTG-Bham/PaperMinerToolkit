"""Test Markdown rendering of structured-document and PDF tables."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from paperminertoolkit.corpus import tables
from paperminertoolkit.corpus.layout import Table

JATS = """<article><body><sec id="s1"><p>See Table 1.</p>
<table-wrap id="t1"><label>Table 1</label><caption><title>Computed gaps<xref rid="fa">a</xref></title></caption>
<table><colgroup><col/><col/><col/></colgroup>
<thead>
<tr><th rowspan="2">Material</th><th colspan="2">Gap [eV]</th></tr>
<tr><th>indirect</th><th>direct</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">ZrTaN<sub>3</sub></td><td>0.45</td><td>1.06</td></tr>
<tr><td>A|B</td><td>2.13</td></tr>
<tr><td></td><td></td><td></td></tr>
</tbody></table>
<table-wrap-foot><fn id="fa"><label>a</label><p>HSE06 values.</p></fn></table-wrap-foot>
</table-wrap></sec></body></article>"""

ELSEVIER = """<full-text-retrieval-response xmlns:ce="http://www.elsevier.com/xml/common/dtd"
 xmlns:cals="http://www.elsevier.com/xml/common/cals/dtd">
<ce:sections><ce:section id="s1"><ce:para>Text.</ce:para>
<ce:table id="tbl1"><ce:label>Table 2</ce:label><ce:caption><ce:simple-para>Optical gaps.</ce:simple-para></ce:caption>
<cals:tgroup cols="3">
<cals:colspec colname="c1"/><cals:colspec colname="c2"/><cals:colspec colname="c3" colnum="3"/>
<cals:thead><cals:row><cals:entry>Sample</cals:entry><cals:entry namest="c2" nameend="c3">Gap (eV)</cals:entry></cals:row></cals:thead>
<cals:tbody>
<cals:row><cals:entry morerows="1">Cs2SnI6</cals:entry><cals:entry>1.35</cals:entry><cals:entry>RT</cals:entry></cals:row>
<cals:row><cals:entry colname="c2">1.30</cals:entry><cals:entry>77 K</cals:entry></cals:row>
</cals:tbody></cals:tgroup>
<ce:table-footnote><ce:note-para>Tauc analysis.</ce:note-para></ce:table-footnote>
</ce:table>
<ce:table id="tbl2"><ce:label>Table 3</ce:label><cals:tgroup cols="1"><cals:tbody></cals:tbody></cals:tgroup></ce:table>
</ce:section></ce:sections></full-text-retrieval-response>"""

TEI = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div>
<figure type="table" xml:id="tab_0"><head>Table 1</head><figDesc>Band gaps.</figDesc>
<table><row role="label"><cell>Phase</cell><cell>Gap</cell></row>
<row><cell cols="2">alpha 1.5 eV</cell></row><row><cell>beta</cell><cell rows="0">2.1</cell></row></table>
</figure></div></body></text></TEI>"""


def test_jats_tables_expand_spans_merge_headers_and_keep_footnotes() -> None:
    """Render row and column spans as a rectangular Markdown grid."""
    text = tables.structured_tables_text(JATS, 'jats', 'paper-1')

    assert text.splitlines() == [
        'Table 1 Computed gapsa',
        '| Material | Gap [eV] indirect | Gap [eV] direct |',
        '| --- | --- | --- |',
        '| ZrTaN3 | 0.45 | 1.06 |',
        '| ZrTaN3 | A\\|B | 2.13 |',
        'Notes: aHSE06 values.',
    ]


def test_elsevier_cals_tables_use_column_names_and_morerows() -> None:
    """Map CALS column names, spans and continuation rows onto the grid."""
    text = tables.structured_tables_text(ELSEVIER.encode(), 'Elsevier-XML')

    assert text.splitlines() == [
        'Table 2 Optical gaps.',
        '| Sample | Gap (eV) | Gap (eV) |',
        '| --- | --- | --- |',
        '| Cs2SnI6 | 1.35 | RT |',
        '| Cs2SnI6 | 1.30 | 77 K |',
        'Notes: Tauc analysis.',
    ]


def test_tei_tables_use_label_rows_and_cell_spans() -> None:
    """Treat TEI label rows as the header and expand ``cols``."""
    text = tables.structured_tables_text(TEI, 'tei')

    assert '| Phase | Gap |' in text
    assert '| alpha 1.5 eV | alpha 1.5 eV |' in text
    assert '| beta | 2.1 |' in text


def test_structured_tables_text_ignores_unsupported_or_malformed_documents() -> None:
    """Return nothing for unknown formats, malformed XML and empty tables."""
    assert tables.structured_tables_text(JATS, 'html') == ''
    assert tables.structured_tables_text('<article>', 'jats') == ''
    assert tables.table_markdown(Table('t', content='<table')) == ''
    assert tables.table_markdown(Table('t', content='<table><tr><td/></tr></table>')) == ''
    untitled = tables.table_markdown(Table('t', content='<table><tr><td>a</td></tr><tr><td>1</td></tr></table>'))
    assert untitled.splitlines()[0] == 'Table'


def test_markdown_table_pads_rows_and_handles_header_only_tables() -> None:
    """Pad ragged rows to the widest row and accept a header without a body."""
    assert tables.markdown_table([]) == ''
    assert tables.markdown_table([['a', 'b']]) == '| a | b |\n| --- | --- |'
    assert tables.markdown_table([['a'], ['1', '2']], header_rows=0).splitlines() == [
        '| a |  |',
        '| --- | --- |',
        '| 1 | 2 |',
    ]


def test_row_spans_continue_past_short_rows() -> None:
    """Carry a row-spanning cell into a later row that ends before its column."""
    content = ('<table><tr><th>a</th><th>b</th><th>c</th></tr>'
               '<tr><td>1</td><td>2</td><td rowspan="3">x</td></tr><tr><td>3</td></tr><tr><td>4</td></tr></table>')

    text = tables.table_markdown(Table('t', content=content))

    assert text.splitlines()[3:] == ['| 1 | 2 | x |', '| 3 |  | x |', '| 4 |  | x |']


def test_column_names_fall_back_to_declaration_order() -> None:
    """Number CALS column specs without ``colnum`` by their position."""
    import xml.etree.ElementTree as ET

    table = ET.fromstring('<t><colspec colname="a"/><colspec/><colspec colname="b" colnum="x"/></t>')
    assert tables._column_names(table) == {'a': 0, 'b': 1}


def _ruled_table_pdf(path: Path) -> None:
    """Write a one-page PDF with a ruled two-by-three table."""
    document = pymupdf.open()
    page = document.new_page()
    cells = [['Material', 'Gap (eV)'], ['LaWN3', '2.04'], ['ZrTaN3', '0.45']]
    for row_index, row in enumerate(cells):
        for column_index, value in enumerate(row):
            rect = pymupdf.Rect(72 + 120 * column_index, 72 + 24 * row_index, 192 + 120 * column_index,
                                96 + 24 * row_index)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 7), value, fontsize=10)
    document.save(path)
    document.close()


def test_pdf_tables_text_reads_ruled_tables_by_word_position(tmp_path: Path) -> None:
    """Detect a ruled PDF table and rebuild its cells from positioned words."""
    pdf = tmp_path / 'table.pdf'
    _ruled_table_pdf(pdf)
    blank = tmp_path / 'blank.pdf'
    document = pymupdf.open()
    document.new_page()
    document.save(blank)
    document.close()

    text = tables.pdf_tables_text(pdf)

    assert text.splitlines()[0] == 'Table (page 1)'
    assert '| LaWN3 | 2.04 |' in text
    assert '| ZrTaN3 | 0.45 |' in text
    assert tables.pdf_tables_text(blank) == ''


def test_pdf_tables_text_returns_nothing_for_unreadable_files(tmp_path: Path) -> None:
    """Treat a file PyMuPDF cannot open as having no tables."""
    broken = tmp_path / 'broken.pdf'
    broken.write_bytes(b'not a pdf')

    assert tables.pdf_tables_text(broken) == ''


def test_words_in_skips_merged_cells_and_words_outside_the_cell() -> None:
    """Keep only words whose centres fall inside the cell box."""
    words = [(0, 0, 10, 10, 'in'), (50, 50, 60, 60, 'out')]

    assert tables._words_in(words, None) == ''
    assert tables._words_in(words, (0, 0, 20, 20)) == 'in'


@pytest.mark.parametrize(('text', 'found', 'expected'), [
    ('Body', '', 'Body'),
    ('Body\n', '| a |', 'Body\n\n=== TABLES ===\n\n| a |\n'),
])
def test_append_tables_adds_a_section_only_when_tables_exist(text: str, found: str, expected: str) -> None:
    """Append a headed tables section after the document text."""
    assert tables.append_tables(text, found) == expected
