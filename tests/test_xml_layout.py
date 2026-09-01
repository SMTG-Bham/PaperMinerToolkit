"""Tests for JATS and Elsevier XML layout normalization."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from paperminertoolkit.corpus.layout import BoundingBox, Figure, Graphic, Section, Table
from paperminertoolkit.corpus.xml_layout import (
    parse_elsevier_layout,
    parse_jats_layout,
    parse_tei_layout,
)

DATA_DIR = Path(__file__).resolve().parent / 'data'
# Real documents fetched from the archives, trimmed to their figures. See
# tests/data/README.md for provenance and licensing.
REAL_JATS = {
    'biorxiv': ('biorxiv_2023.03.30.534894v4.jats.xml',
                'https://www.biorxiv.org/content/early/2024/08/02/2023.03.30.534894.source.xml'),
    'medrxiv': ('medrxiv_2024.05.31.24307874v1.jats.xml',
                'https://www.medrxiv.org/content/early/2024/06/01/2024.05.31.24307874.source.xml'),
}


JATS = '''<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta><title-group><article-title>JATS example</article-title>
  </title-group></article-meta></front>
  <body>
    <p>Unsectioned introduction.</p>
    <sec id="results"><title>Results</title>
      <p>As shown in <xref ref-type="fig" rid="fig1">Figure 1</xref> and
      <xref ref-type="table" rid="tab1">Table 1</xref>, values increased.</p>
      <fig id="fig1"><label>Figure 1</label><caption><p>Measured values.</p></caption>
        <graphic id="g1" xlink:href="figures/f1.tif" mimetype="image" mime-subtype="tiff"/>
        <graphic xlink:href="figures/f1.tif"/>
      </fig>
      <fig><graphic xlink:href="figures/unlabelled.png"/></fig>
      <table-wrap id="tab1"><label>Table 1</label><caption><p>Tabulated values.</p></caption>
        <table><tr><td>1</td></tr></table>
      </table-wrap>
      <sec><title>Nested</title><p>Nested prose.</p></sec>
    </sec>
  </body>
</article>'''


ELSEVIER = '''<full-text-retrieval-response
  xmlns:ce="http://www.elsevier.com/xml/common/dtd"
  xmlns:xlink="http://www.w3.org/1999/xlink">
  <coredata><title>Elsevier example</title></coredata>
  <originalText><body><ce:sections><ce:section id="s1">
    <ce:section-title>Methods</ce:section-title>
    <ce:para>See <ce:cross-ref refid="fA">Figure 2</ce:cross-ref>.</ce:para>
    <ce:figure id="fA"><ce:label>Figure 2</ce:label>
      <ce:caption><ce:simple-para>Instrument layout.</ce:simple-para></ce:caption>
      <ce:link locator="https://cdn.example/fA.svg" mime-type="image/svg+xml"/>
    </ce:figure>
    <ce:table id="tA"><ce:label>Table 2</ce:label>
      <ce:caption><ce:simple-para>Settings.</ce:simple-para></ce:caption>
      <row><entry>A</entry></row>
    </ce:table>
  </ce:section></ce:sections></body></originalText>
</full-text-retrieval-response>'''


TEI = '''<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><titleStmt><title>GROBID example</title></titleStmt></fileDesc></teiHeader>
  <text><body><div xml:id="d1"><head>Results</head>
    <p>See <ref type="figure" target="#fig_1">Figure 1</ref>.</p>
    <figure xml:id="fig_1"><head>Figure 1</head><figDesc>PDF-derived plot.</figDesc>
      <graphic url="figure-1.png" type="image/png" coords="2,10,20,200,100;bad"/>
    </figure>
    <figure type="table" xml:id="tab_1" coords="3,30,40,300,120">
      <head>Table 1</head><figDesc>PDF-derived table.</figDesc><table><row><cell>A</cell></row></table>
    </figure>
  </div></body></text>
</TEI>'''


def test_parse_jats_layout_retains_hierarchy_objects_and_references() -> None:
    """Normalize JATS sections, graphics, tables, and in-text references."""
    layout = parse_jats_layout(
        JATS,
        'pmcid:PMC1',
        source='pubmed',
        source_identifier='PMC1',
    )

    assert layout.title == 'JATS example'
    assert layout.provenance.source == 'pubmed'
    assert layout.provenance.document_format == 'jats'
    assert [section.identifier for section in layout.sections] == ['body', 'results']
    assert layout.sections[0].blocks[0].text == 'Unsectioned introduction.'
    assert layout.sections[1].children[0].identifier == 'section-1-1'
    assert [block.reading_order for block in layout.iter_text_blocks()] == [0, 1, 2]

    figure = layout.figures[0]
    assert isinstance(figure, Figure)
    assert figure.identifier == 'fig1'
    assert figure.label == 'Figure 1'
    assert figure.caption == 'Measured values.'
    assert figure.graphics == (
        Graphic('g1', 'figures/f1.tif', 'image/tiff'),
    )
    assert figure.reference_sentences[0].section_id == 'results'
    assert 'values increased' in figure.reference_sentences[0].text
    assert layout.figures[1].identifier == 'figure-2'
    assert layout.figures[1].caption == ''

    table = layout.tables[0]
    assert isinstance(table, Table)
    assert table.identifier == 'tab1'
    assert table.caption == 'Tabulated values.'
    assert table.content_format == 'jats'
    assert '<table' in table.content
    assert table.reference_sentences[0].section_id == 'results'


def test_parse_elsevier_layout_produces_the_same_model_types() -> None:
    """Normalize namespaced Elsevier XML through the common layout contract."""
    layout = parse_elsevier_layout(
        ELSEVIER,
        'doi:10.1/example',
        source_identifier='1-s2.0-example',
    )

    assert layout.title == 'Elsevier example'
    assert layout.provenance.source == 'elsevier'
    assert layout.provenance.document_format == 'elsevier-xml'
    assert layout.provenance.source_identifier == '1-s2.0-example'
    assert isinstance(layout.sections[0], Section)
    assert layout.sections[0].title == 'Methods'
    assert layout.sections[0].blocks[0].text == 'See Figure 2.'
    assert isinstance(layout.figures[0], Figure)
    assert layout.figures[0].graphics == (
        Graphic('graphic-1', 'https://cdn.example/fA.svg', 'image/svg+xml'),
    )
    assert layout.figures[0].reference_sentences[0].text == 'See Figure 2.'
    assert isinstance(layout.tables[0], Table)
    assert layout.tables[0].content_format == 'elsevier-xml'


def test_parse_elsevier_layout_resolves_bare_locators_using_the_document_eid() -> None:
    """Rewrite native ``gr1``-style locators into object-retrieval URLs.

    Elsevier's own native XML embeds figure graphics as bare internal
    reference tokens (``locator="gr1"``) rather than a path or URL. They are
    not resolvable by joining them against the article endpoint; retrieving
    the image requires the article's own ``eid`` plus Elsevier's
    object-retrieval endpoint convention.
    """
    with_eid = '''<full-text-retrieval-response xmlns:ce="http://www.elsevier.com/xml/common/dtd">
      <coredata><eid>1-s2.0-S000000000000X</eid></coredata>
      <originalText><body><ce:sections><ce:section id="s1">
        <ce:figure id="fA"><ce:label>Figure 1</ce:label>
          <ce:link locator="gr1"/>
          <ce:link locator="https://cdn.example/already-absolute.jpg"/>
        </ce:figure>
      </ce:section></ce:sections></body></originalText>
    </full-text-retrieval-response>'''
    layout = parse_elsevier_layout(with_eid, 'paper:eid')
    uris = [graphic.uri for graphic in layout.figures[0].graphics]
    assert uris == [
        'https://api.elsevier.com/content/object/eid/1-s2.0-S000000000000X-gr1.jpg',
        'https://cdn.example/already-absolute.jpg',
    ]

    without_eid = '''<full-text-retrieval-response xmlns:ce="http://www.elsevier.com/xml/common/dtd">
      <originalText><body><ce:sections><ce:section id="s1">
        <ce:figure id="fA"><ce:label>Figure 1</ce:label><ce:link locator="gr1"/></ce:figure>
      </ce:section></ce:sections></body></originalText>
    </full-text-retrieval-response>'''
    unresolved = parse_elsevier_layout(without_eid, 'paper:no-eid')
    assert unresolved.figures[0].graphics[0].uri == 'gr1'


def test_xml_layout_tolerates_sparse_elements_and_rejects_broken_xml() -> None:
    """Use stable fallbacks for sparse objects but reject malformed documents."""
    sparse = '''<article><body><sec><p>Text without references.</p>
      <fig><caption/></fig><table-wrap><table/></table-wrap></sec></body></article>'''
    layout = parse_jats_layout(sparse, 'paper:sparse')

    assert layout.title == ''
    assert layout.sections[0].identifier == 'section-1'
    assert layout.figures[0].identifier == 'figure-1'
    assert layout.figures[0].graphics == ()
    assert layout.figures[0].reference_sentences == ()
    assert layout.tables[0].identifier == 'table-1'
    assert layout.tables[0].reference_sentences == ()

    original_text = parse_elsevier_layout(
        '<article><originalText><section id="s"><para>Text.</para></section>'
        '</originalText></article>',
        'paper:original-text',
    )
    root_sections = parse_jats_layout(
        '<article><sec id="s"><p>Text.</p></sec></article>',
        'paper:root-sections',
    )
    assert original_text.sections[0].identifier == 's'
    assert root_sections.sections[0].identifier == 's'

    with pytest.raises(ValueError, match='malformed jats document'):
        parse_jats_layout('<article', 'paper:broken')
    with pytest.raises(ValueError, match='malformed elsevier-xml document'):
        parse_elsevier_layout('<article', 'paper:broken')


def test_parse_tei_layout_marks_pdf_derived_structure_and_coordinates() -> None:
    """Normalize GROBID TEI sections, references, graphics, and coordinate groups."""
    layout = parse_tei_layout(TEI, 'openalex:W1', source_identifier='W1')

    assert layout.title == 'GROBID example'
    assert layout.provenance.source == 'openalex-grobid'
    assert layout.provenance.document_format == 'tei'
    assert layout.provenance.parser == 'grobid-tei'
    assert layout.sections[0].identifier == 'd1'
    assert layout.sections[0].title == 'Results'
    assert layout.figures[0].caption == 'PDF-derived plot.'
    assert layout.figures[0].boxes[0] == BoundingBox(2, 10, 20, 210, 120)
    assert layout.figures[0].graphics[0].uri == 'figure-1.png'
    assert layout.figures[0].reference_sentences[0].text == 'See Figure 1.'
    assert layout.tables[0].identifier == 'tab_1'
    assert layout.tables[0].boxes[0] == BoundingBox(3, 30, 40, 330, 160)

    with pytest.raises(ValueError, match='malformed tei document'):
        parse_tei_layout('<TEI', 'paper:broken')
    with pytest.raises(ValueError, match='TEI root'):
        parse_tei_layout('<article/>', 'paper:not-tei')


RXIV_JATS = '''<article xmlns:xlink="http://www.w3.org/1999/xlink"
  xmlns:hwp="http://schema.highwire.org/Journal">
  <front><article-meta>
    <article-id pub-id-type="doi">10.1101/339747</article-id>
    <article-id pub-id-type="other" hwp:sub-type="slug">339747</article-id>
    <title-group><article-title>Anoxic conditioning</article-title></title-group>
  </article-meta></front>
  <body><sec id="results"><title>Results</title>
    <fig id="fig1" hwp:id="F1">
      <object-id pub-id-type="other" hwp:sub-type="pisa">biorxiv;339747v4/FIG1</object-id>
      <object-id pub-id-type="other" hwp:sub-type="slug">F1</object-id>
      <label>Fig. 1</label><caption><p>Direct viable counting.</p></caption>
      <graphic xlink:href="339747v4_fig1"/>
    </fig>
    <fig id="fig2" hwp:id="F2">
      <object-id pub-id-type="other" hwp:sub-type="slug">F2</object-id>
      <label>Fig. 2</label><caption><p>Oxygen intolerance.</p></caption>
      <graphic xlink:href="339747v4_fig2"/>
    </fig>
    <fig id="fig3">
      <label>Fig. 3</label><caption><p>No slug declared.</p></caption>
      <graphic xlink:href="339747v4_fig3"/>
    </fig>
    <fig id="fig4" hwp:id="F4">
      <object-id pub-id-type="other" hwp:sub-type="slug">F4</object-id>
      <label>Fig. 4</label><caption><p>Already absolute.</p></caption>
      <graphic xlink:href="https://cdn.example.org/kept.png"/>
    </fig>
    <fig id="fig5" hwp:id="F5">
      <label>Fig. 5</label><caption><p>Slug only on the attribute.</p></caption>
      <graphic xlink:href="339747v4_fig5"/>
    </fig>
  </sec></body>
</article>'''


def test_parse_jats_layout_resolves_rxiv_figure_tokens_from_the_source_url() -> None:
    """Turn bioRxiv's internal figure tokens into its real image URLs.

    bioRxiv and medRxiv name each graphic with a token such as
    ``339747v4_fig1`` that resolves against nothing. The archives serve the
    image under the figure's display slug, on a path derived from the posting
    date and article slug in the document's own source URL.
    """
    source_url = 'https://www.biorxiv.org/content/early/2019/05/10/339747.source.xml'
    layout = parse_jats_layout(RXIV_JATS, 'paper:rxiv', source='biorxiv', source_url=source_url)
    uris = [figure.graphics[0].uri for figure in layout.figures]

    assert uris[0] == (
        'https://www.biorxiv.org/content/biorxiv/early/2019/05/10/339747/F1.large.jpg'
    )
    assert uris[1] == (
        'https://www.biorxiv.org/content/biorxiv/early/2019/05/10/339747/F2.large.jpg'
    )
    # A figure declaring no slug cannot be resolved, and keeps its raw token.
    assert uris[2] == '339747v4_fig3'
    # A graphic that is already a URL is never rewritten.
    assert uris[3] == 'https://cdn.example.org/kept.png'
    # A figure carrying only a prefixed id attribute still yields its slug.
    assert uris[4] == (
        'https://www.biorxiv.org/content/biorxiv/early/2019/05/10/339747/F5.large.jpg'
    )

    # medRxiv uses its own archive segment.
    medrxiv = parse_jats_layout(
        RXIV_JATS, 'paper:rxiv', source='medrxiv',
        source_url='https://www.medrxiv.org/content/early/2024/01/02/12345678.source.xml',
    )
    assert medrxiv.figures[0].graphics[0].uri == (
        'https://www.medrxiv.org/content/medrxiv/early/2024/01/02/12345678/F1.large.jpg'
    )


def test_parse_jats_layout_leaves_other_repositories_untouched() -> None:
    """Resolve tokens only for the archives that use them.

    PMC names its graphics with real filenames that resolve against the
    article's own URL, so they must not be rewritten.
    """
    unchanged = parse_jats_layout(RXIV_JATS, 'paper:rxiv', source='biorxiv')
    assert unchanged.figures[0].graphics[0].uri == '339747v4_fig1'

    pmc = parse_jats_layout(
        RXIV_JATS, 'paper:pmc', source='pubmed',
        source_url='https://pmc-oa-opendata.s3.amazonaws.com/PMC1.1/PMC1.1.xml',
    )
    assert pmc.figures[0].graphics[0].uri == '339747v4_fig1'

    # A content URL that is not a recognised posting path resolves nothing.
    odd = parse_jats_layout(
        RXIV_JATS, 'paper:rxiv', source='biorxiv',
        source_url='https://www.biorxiv.org/some/other/path.xml',
    )
    assert odd.figures[0].graphics[0].uri == '339747v4_fig1'


@pytest.mark.parametrize('archive', sorted(REAL_JATS))
def test_real_archive_jats_resolves_every_figure_to_a_downloadable_url(archive: str) -> None:
    """Parse markup the archives actually publish, not an imitation of it.

    The synthetic fixtures in this module were written from the documented
    shape of JATS. These are the real thing, trimmed to their figures, and they
    carry details no hand-written sample would have thought to include: the
    HighWire ``hwp:`` namespace, three ``object-id`` values per figure of which
    only one is the display slug, and graphic references that are internal
    tokens resolving against nothing.

    Parameters
    ----------
    archive : str
        Archive whose real document is under test.
    """
    filename, source_url = REAL_JATS[archive]
    layout = parse_jats_layout((DATA_DIR / filename).read_text(encoding='utf-8'),
                               'doi:10.1101/example', source=archive,
                               source_identifier='10.1101/example', source_url=source_url)

    assert layout.figures, 'no figures parsed from a real archive document'
    for figure in layout.figures:
        assert figure.caption.strip(), f'{figure.identifier} lost its caption'
        assert figure.graphics, f'{figure.identifier} resolved to no graphic'
        for graphic in figure.graphics:
            # The token in the document is "24307874v1_fig1" and resolves
            # against nothing; what comes out must be the archive's image URL.
            assert graphic.uri.startswith(f'https://www.{archive}.org/content/{archive}/early/')
            assert graphic.uri.endswith('.large.jpg')


def test_real_multi_panel_figures_do_not_repeat_one_image() -> None:
    """Keep one graphic where a figure names several tokens for one image.

    A multi-panel medRxiv figure names one token per panel -- ``_fig4``,
    ``_fig4a``, ``_fig4b`` -- but the archive publishes no per-panel URL, so
    all three resolve to the same figure-level image. Emitting the graphic
    three times would download the same bytes three times, and these archives
    ask seven seconds between requests, so each repeat is seven seconds spent
    fetching something already held.
    """
    filename, source_url = REAL_JATS['medrxiv']
    content = (DATA_DIR / filename).read_text(encoding='utf-8')
    layout = parse_jats_layout(content, 'doi:10.1101/example', source='medrxiv',
                               source_identifier='10.1101/example', source_url=source_url)

    named = {
        re.search(r'id="([^"]+)"', element).group(1): len(re.findall(r'<graphic\b', element))
        for element in re.findall(r'<fig\b.*?</fig>', content, re.S)
    }
    parsed = {figure.identifier: len(figure.graphics) for figure in layout.figures}

    # The fixture has to still contain the case, or this proves nothing.
    assert max(named.values()) > 1, 'fixture no longer covers a multi-panel figure'
    assert named == {'fig1': 1, 'fig2': 1, 'fig3': 2, 'fig4': 3}
    # However many panels a figure names, one image is what can be fetched.
    assert parsed == dict.fromkeys(named, 1)
