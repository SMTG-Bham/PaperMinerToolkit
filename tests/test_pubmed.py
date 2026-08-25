"""Unit tests for NCBI E-utilities request helpers and PubMed record mapping."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import pytest

import paperminertoolkit.providers.pubmed as pubmed
from paperminertoolkit.providers import base as provider

from tests.doubles import FakeResponse, FakeSession


def test_article_mapping_falls_back_to_the_record_root() -> None:
    """Map a sparse record even when its usual Article child is absent."""
    assert pubmed.article_to_paper(ET.Element('PubmedArticle'))['paper_id'] == ''


def test_pubmed_defensive_request_and_parser_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover empty responses, invalid payloads, option guards, and sparse XML records."""
    monkeypatch.setattr(pubmed.provider, 'default_headers', lambda: {'X': 'header'})
    assert pubmed.request_headers() == {'X': 'header'}
    assert pubmed._error_text(object()) == ''

    monkeypatch.setattr(pubmed, 'request', lambda *args, **kwargs: None)
    assert pubmed.request_json('url') is None
    assert pubmed.request_xml('url') is None

    class InvalidJSON:
        """Response double with an undecodable JSON body."""

        def json(self) -> Any:
            """Raise a JSON-like decoding failure."""
            raise ValueError('bad json')

    monkeypatch.setattr(pubmed, 'request', lambda *args, **kwargs: InvalidJSON())
    with pytest.raises(RuntimeError, match='undecodable JSON'):
        pubmed.request_json('url')

    with pytest.raises(ValueError, match='sort must be'):
        pubmed._search_params('term', 'bad', '', '', '', 0, 'pubmed')
    with pytest.raises(ValueError, match='datetype must be'):
        pubmed._search_params('term', '', 'bad', '', '', 0, 'pubmed')

    monkeypatch.setattr(
        pubmed, 'request_json',
        lambda *args, **kwargs: {'esearchresult': {'count': 'not-a-number'}},
    )
    assert pubmed.esearch('term') == ([], 0)
    assert pubmed.esearch_history('term') == ('', '', 0)

    sparse = ET.fromstring(
        '<PubmedArticle><MedlineCitation><Article>'
        '<Abstract><AbstractText/></Abstract>'
        '<AuthorList><Author/></AuthorList>'
        '</Article></MedlineCitation>'
        '<PubmedData><ArticleIdList><ArticleId IdType="doi"/></ArticleIdList></PubmedData>'
        '<MeshHeadingList><MeshHeading><DescriptorName/></MeshHeading></MeshHeadingList>'
        '</PubmedArticle>'
    )
    paper = pubmed.article_to_paper(sparse)
    assert paper['paper_id'] == ''
    assert paper['abstract'] == ''
    assert paper['authors'] == ''
    assert paper['mesh'] == []
    assert pubmed.parse_articles(sparse) == [paper]


def test_pubmed_open_access_helpers_handle_empty_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return empty results for absent identifiers, records, links, and PMC bodies."""
    assert pubmed.oa_package_urls('') == []
    monkeypatch.setattr(pubmed, 'request_xml', lambda *args, **kwargs: None)
    assert pubmed.oa_package_urls('PMC1') == []
    root = ET.fromstring('<oa><record><link href="" format="pdf"/></record></oa>')
    monkeypatch.setattr(pubmed, 'request_xml', lambda *args, **kwargs: root)
    assert pubmed.oa_package_urls('PMC1') == []
    monkeypatch.setattr(pubmed, 'efetch_ids', lambda *args, **kwargs: None)
    assert pubmed.pmc_full_text('PMC1') == ''


def article_set() -> str:
    """Return a PubmedArticleSet exercising every field the parser consumes.

    The three records cover, in order, a fully populated journal article with
    inline title markup and a labelled abstract, a DOI-less record carrying a
    free-text ``MedlineDate``, and a book record with no ``Article`` element.
    """
    return '''<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">31234567</PMID>
      <Article>
        <Journal>
          <Title>Journal of Solid State Chemistry</Title>
          <ISOAbbreviation>J Solid State Chem</ISOAbbreviation>
          <JournalIssue>
            <PubDate><Year>2024</Year><Month>Apr</Month><Day>1</Day></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Conductivity of Li<sub>7</sub>La<sub>3</sub>Zr<sub>2</sub>O<sub>12</sub></ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Solid electrolytes matter.</AbstractText>
          <AbstractText Label="METHODS" NlmCategory="METHODS">We measured impedance.</AbstractText>
          <AbstractText Label="UNLABELLED">Conductivity reached 1 mS/cm.</AbstractText>
          <CopyrightInformation>Copyright 2024 Example Press.</CopyrightInformation>
        </Abstract>
        <AuthorList>
          <Author ValidYN="Y"><LastName>Smith</LastName><ForeName>Jane A</ForeName><Initials>JA</Initials></Author>
          <Author ValidYN="Y"><CollectiveName>Solid State Consortium</CollectiveName></Author>
          <Author ValidYN="N"><LastName>Ghost</LastName><ForeName>Removed</ForeName></Author>
        </AuthorList>
        <ArticleDate DateType="Electronic"><Year>2024</Year><Month>03</Month><Day>07</Day></ArticleDate>
        <PublicationTypeList>
          <PublicationType UI="D016428">Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName UI="D007854" MajorTopicYN="Y">Lithium</DescriptorName>
          <QualifierName UI="Q000032" MajorTopicYN="N">analysis</QualifierName>
        </MeshHeading>
      </MeshHeadingList>
      <KeywordList Owner="NOTNLM"><Keyword MajorTopicYN="N">garnet</Keyword></KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">31234567</ArticleId>
        <ArticleId IdType="doi">10.1016/J.JSSC.2024.01.001</ArticleId>
        <ArticleId IdType="pmc">PMC9876543</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">10101010</PMID>
      <Article>
        <Journal>
          <Title>Old Journal</Title>
          <JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>An older paper without a DOI</ArticleTitle>
        <Abstract><AbstractText>One flat paragraph.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Doe</LastName><Initials>J</Initials></Author></AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList><ArticleId IdType="pubmed">10101010</ArticleId></ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedBookArticle>
    <BookDocument>
      <PMID Version="1">29262034</PMID>
      <Book>
        <BookTitle book="statpearls">StatPearls</BookTitle>
        <PubDate><Year>2023</Year><Month>Jan</Month></PubDate>
      </Book>
      <ArticleTitle book="statpearls">Lithium Toxicity</ArticleTitle>
      <Abstract><AbstractText>Lithium has a narrow therapeutic index.</AbstractText></Abstract>
      <AuthorList Type="authors">
        <Author><LastName>Roe</LastName><ForeName>Ann</ForeName></Author>
      </AuthorList>
      <PublicationTypeList><PublicationType UI="D016454">Review</PublicationType></PublicationTypeList>
    </BookDocument>
    <PubmedBookData>
      <ArticleIdList>
        <ArticleId IdType="bookaccession">NBK1234</ArticleId>
        <ArticleId IdType="pubmed">29262034</ArticleId>
      </ArticleIdList>
    </PubmedBookData>
  </PubmedBookArticle>
</PubmedArticleSet>
'''


def jats_article() -> str:
    """Return a PMC open-access record with prose, a table, and a reference list."""
    return '''<?xml version="1.0" ?>
<pmc-articleset><article>
  <front><article-meta><title-group>
    <article-title>Garnet electrolytes</article-title>
  </title-group></article-meta></front>
  <body>
    <sec><title>Introduction</title>
      <p>Garnets conduct lithium at <italic>room</italic> temperature.</p>
      <table-wrap><label>Table 1</label><table><tr><td>Do not extract me</td></tr></table></table-wrap>
    </sec>
    <sec><title>Results</title>
      <p>Conductivity reached 1 mS/cm.</p>
      <fig><caption><p>Skip this caption.</p></caption></fig>
    </sec>
  </body>
  <back><ref-list><ref><p>Skip this reference.</p></ref></ref-list></back>
</article></pmc-articleset>
'''


def parsed_articles() -> list[dict[str, Any]]:
    """Return the three shared fixture records mapped onto the paper schema."""
    return pubmed.parse_articles(ET.fromstring(article_set()))


def test_element_text_flattens_inline_markup_and_missing_elements() -> None:
    """Keep subscript content in a title instead of truncating at the first tag."""
    article = parsed_articles()[0]
    assert article['title'] == 'Conductivity of Li7La3Zr2O12'
    assert pubmed._element_text(None) == ''
    assert pubmed._element_text(ET.fromstring('<t>  a\n  b </t>')) == 'a b'


def test_abstract_text_labels_sections_and_drops_copyright() -> None:
    """Prefix real section labels, skip the placeholder label, and omit copyright."""
    abstract = parsed_articles()[0]['abstract']
    assert abstract == ('BACKGROUND: Solid electrolytes matter. '
                        'METHODS: We measured impedance. '
                        'Conductivity reached 1 mS/cm.')
    assert 'Copyright' not in abstract
    assert parsed_articles()[1]['abstract'] == 'One flat paragraph.'


def test_publication_date_prefers_article_date_then_pubdate_then_medline_year() -> None:
    """Take the electronic date first and reduce a free-text date to its year."""
    articles = parsed_articles()
    assert articles[0]['publication_date'] == '2024-03-07'
    assert articles[1]['publication_date'] == '2019'
    assert articles[2]['publication_date'] == '2023-01'


def test_month_number_accepts_names_and_numbers() -> None:
    """Normalize both month spellings PubMed uses and reject impossible values."""
    assert pubmed._month_number('Jan') == '01'
    assert pubmed._month_number('december') == '12'
    assert pubmed._month_number('3') == '03'
    assert pubmed._month_number('13') == ''
    assert pubmed._month_number('') == ''


def test_authors_format_personal_and_collective_names() -> None:
    """Join personal and collective names and drop authors marked invalid."""
    articles = parsed_articles()
    assert articles[0]['authors'] == 'Jane A Smith; Solid State Consortium'
    assert 'Removed Ghost' not in articles[0]['authors']
    assert articles[1]['authors'] == 'J Doe'


def test_article_to_paper_falls_back_to_a_pubmed_paper_id_without_a_doi() -> None:
    """Key a DOI-less record on its PMID and clean the DOI when one is present."""
    articles = parsed_articles()
    assert articles[0]['paper_id'] == 'doi:10.1016/j.jssc.2024.01.001'
    assert articles[0]['doi'] == '10.1016/j.jssc.2024.01.001'
    assert articles[0]['pmid'] == '31234567'
    assert articles[0]['pmcid'] == 'PMC9876543'
    assert articles[1]['paper_id'] == 'pmid:10101010'
    assert articles[1]['doi'] == ''
    assert articles[1]['pmcid'] == ''


def test_parse_articles_maps_book_records_without_dropping_them() -> None:
    """Read a book record from its BookDocument rather than skipping it."""
    articles = parsed_articles()
    assert len(articles) == 3
    book = articles[2]
    assert book['article_type'] == 'book'
    assert book['title'] == 'Lithium Toxicity'
    assert book['journal'] == 'StatPearls'
    assert book['paper_id'] == 'pmid:29262034'
    assert book['authors'] == 'Ann Roe'
    assert pubmed.parse_articles(None) == []


def test_article_to_paper_collects_mesh_keywords_and_publication_types() -> None:
    """Split descriptors, qualifiers, keywords, and types into separate lists."""
    article = parsed_articles()[0]
    assert article['mesh'] == [
        {'scheme': 'mesh', 'id': 'D007854', 'name': 'Lithium', 'is_primary': '1'},
        {'scheme': 'mesh_qualifier', 'id': 'Q000032', 'name': 'analysis', 'is_primary': '0'},
    ]
    assert article['keywords'] == ['garnet']
    assert article['publication_types'] == [{'id': 'D016428', 'name': 'Journal Article'}]
    assert article['sources'] == 'pubmed'
    assert article['metadata_status'] == 'retrieved'


def test_normalize_identifiers_extract_digits_and_prefix_pmc() -> None:
    """Reduce identifier URLs to the bare forms the corpus stores."""
    assert pubmed.normalize_pmid('https://pubmed.ncbi.nlm.nih.gov/31234567/') == '31234567'
    assert pubmed.normalize_pmid(None) == ''
    assert pubmed.normalize_pmid('none') == ''
    assert pubmed.normalize_pmcid('9876543') == 'PMC9876543'
    assert pubmed.normalize_pmcid('PMC9876543') == 'PMC9876543'
    assert pubmed.normalize_pmcid(None) == ''


def test_request_params_add_tool_and_omit_absent_credentials() -> None:
    """Always identify the tool but never invent a key or contact address."""
    assert pubmed.request_params({'db': 'pubmed'}) == {'db': 'pubmed', 'tool': 'PaperMinerToolkit'}
    assert pubmed.request_params(None, 'key', 'person@example.com') == {
        'tool': 'PaperMinerToolkit', 'email': 'person@example.com', 'api_key': 'key'}


def test_min_interval_switches_on_the_presence_of_an_api_key() -> None:
    """Pace keyless clients at three requests per second and keyed ones at ten."""
    assert pubmed.min_interval(None) == pubmed.NCBI_MIN_INTERVAL
    assert pubmed.min_interval('key') == pubmed.NCBI_KEYED_MIN_INTERVAL


def test_request_paces_consecutive_calls_across_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep between requests using one shared window for every endpoint."""
    sleeps: list[float] = []
    clock = {'now': 100.0}
    monkeypatch.setattr(provider.time, 'sleep', lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(provider.time, 'monotonic', lambda: clock['now'])

    session = FakeSession([FakeResponse(text='<a/>') for _ in range(3)])
    for _ in range(3):
        pubmed.request(pubmed.ESEARCH_URL, session=session)
    assert sleeps == pytest.approx([pubmed.NCBI_MIN_INTERVAL] * 2)

    # The window carries over to a different endpoint and credential, so under a
    # frozen clock the first keyed request waits too rather than starting fresh.
    sleeps.clear()
    session = FakeSession([FakeResponse(text='<a/>') for _ in range(2)])
    for _ in range(2):
        pubmed.request(pubmed.EFETCH_URL, api_key='key', session=session)
    assert sleeps == pytest.approx([pubmed.NCBI_KEYED_MIN_INTERVAL] * 2)


def test_request_returns_none_for_a_missing_record() -> None:
    """Treat a 404 as an absent record rather than an error."""
    session = FakeSession([FakeResponse(status_code=404)])
    assert pubmed.request(pubmed.EFETCH_URL, session=session) is None


def test_request_retries_a_rate_limited_response_and_honours_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry a per-second rate limit instead of failing, waiting as instructed."""
    sleeps: list[float] = []
    monkeypatch.setattr(provider.time, 'sleep', lambda seconds: sleeps.append(seconds))
    session = FakeSession([
        FakeResponse(status_code=429, headers={'Retry-After': '5'}),
        FakeResponse(text='<ok/>'),
    ])
    response = pubmed.request(pubmed.ESEARCH_URL, session=session)
    assert response is not None
    assert response.text == '<ok/>'
    assert 5 in sleeps


def test_request_raises_after_exhausting_attempts() -> None:
    """Retry a server error and report the last one once every attempt failed."""
    session = FakeSession([FakeResponse(status_code=500) for _ in range(4)])
    with pytest.raises(RuntimeError, match='NCBI request failed after 4 attempts'):
        pubmed.request(pubmed.ESEARCH_URL, session=session)
    assert len(session.calls) == 4


def test_request_fails_immediately_on_a_client_error_other_than_a_rate_limit() -> None:
    """Spend one request on a rejection instead of retrying a terminal status."""
    session = FakeSession([FakeResponse(status_code=403)])
    with pytest.raises(RuntimeError, match='NCBI rejected the request with 403'):
        pubmed.request(pubmed.ESEARCH_URL, session=session)
    assert len(session.calls) == 1


def test_request_json_raises_on_an_error_member_of_a_successful_response() -> None:
    """Surface a rejected query that E-utilities reports inside a 200 response."""
    session = FakeSession([FakeResponse(payload={'esearchresult': {'ERROR': 'Invalid field'}})])
    with pytest.raises(RuntimeError, match='NCBI rejected the request: Invalid field'):
        pubmed.request_json(pubmed.ESEARCH_URL, session=session)


def test_request_xml_raises_on_an_error_element_and_malformed_bodies() -> None:
    """Reject an error document and an unparseable body, and tolerate an empty one."""
    session = FakeSession([FakeResponse(text='<eFetchResult><ERROR>ID list is empty</ERROR></eFetchResult>')])
    with pytest.raises(RuntimeError, match='ID list is empty'):
        pubmed.request_xml(pubmed.EFETCH_URL, session=session)

    session = FakeSession([FakeResponse(text='<broken')])
    with pytest.raises(RuntimeError, match='malformed XML'):
        pubmed.request_xml(pubmed.EFETCH_URL, session=session)

    session = FakeSession([FakeResponse(text='   ')])
    assert pubmed.request_xml(pubmed.EFETCH_URL, session=session) is None


def test_esearch_returns_identifiers_with_the_total_match_count() -> None:
    """Read the identifier list and count from one esearch page."""
    session = FakeSession([FakeResponse(payload={'esearchresult': {'count': '42', 'idlist': ['1', '2']}})])
    identifiers, total = pubmed.esearch('lithium', retmax=2, session=session)
    assert identifiers == ['1', '2']
    assert total == 42
    params = session.calls[0]['params']
    assert params['db'] == 'pubmed'
    assert params['term'] == 'lithium'
    assert params['retmode'] == 'json'
    assert params['sort'] == 'relevance'
    assert params['retmax'] == 2


def test_esearch_caps_retmax_and_forwards_date_filters() -> None:
    """Clamp the page size to the E-utilities maximum and pass date filters on."""
    session = FakeSession([FakeResponse(payload={'esearchresult': {'count': '0', 'idlist': []}})])
    pubmed.esearch('lithium', retmax=999999, datetype='pdat', mindate='2020',
                   maxdate='2024/12/31', reldate=30, session=session)
    params = session.calls[0]['params']
    assert params['retmax'] == pubmed.MAX_SEARCH_RESULTS
    assert params['datetype'] == 'pdat'
    assert params['mindate'] == '2020'
    assert params['maxdate'] == '2024/12/31'
    assert params['reldate'] == 30


def test_esearch_history_returns_webenv_query_key_and_count() -> None:
    """Store the result set server-side and return its handles."""
    session = FakeSession([FakeResponse(payload={
        'esearchresult': {'count': '3', 'webenv': 'WE1', 'querykey': '1'}})])
    webenv, query_key, total = pubmed.esearch_history('lithium', session=session)
    assert (webenv, query_key, total) == ('WE1', '1', 3)
    assert session.calls[0]['params']['usehistory'] == 'y'
    assert session.calls[0]['params']['retmax'] == 0


def test_efetch_history_pages_the_stored_set_with_retstart_and_retmax() -> None:
    """Request one page of a stored set rather than resending identifiers."""
    session = FakeSession([FakeResponse(text=article_set())])
    root = pubmed.efetch_history('WE1', '1', retstart=200, retmax=100, session=session)
    assert root is not None
    params = session.calls[0]['params']
    assert params['WebEnv'] == 'WE1'
    assert params['query_key'] == '1'
    assert params['retstart'] == 200
    assert params['retmax'] == 100
    assert params['retmode'] == 'xml'
    assert 'id' not in params


def test_efetch_ids_joins_identifiers_and_skips_an_empty_list() -> None:
    """Send one comma-joined identifier list and never request an empty set."""
    session = FakeSession([FakeResponse(text=article_set())])
    assert pubmed.efetch_ids(['31234567', ' 10101010 '], session=session) is not None
    assert session.calls[0]['params']['id'] == '31234567,10101010'
    assert pubmed.efetch_ids([' ', ''], session=session) is None


def test_find_pmid_searches_the_article_identifier_field() -> None:
    """Resolve a DOI through esearch rather than the rejected converter service."""
    session = FakeSession([FakeResponse(payload={'esearchresult': {'count': '1', 'idlist': ['77']}})])
    assert pubmed.find_pmid('10.1234/A', session=session) == '77'
    assert session.calls[0]['url'] == pubmed.ESEARCH_URL
    assert session.calls[0]['params']['term'] == '10.1234/a[AID]'
    assert session.calls[0]['params']['retmax'] == 1

    session = FakeSession([FakeResponse(payload={'esearchresult': {'count': '0', 'idlist': []}})])
    assert pubmed.find_pmid('10.1234/b', session=session) == ''
    assert pubmed.find_pmid('', session=FakeSession([])) == ''


def test_resolve_pmid_prefers_stored_values_over_a_lookup() -> None:
    """Skip the lookup entirely when the row already identifies the record."""
    session = FakeSession([])
    assert pubmed.resolve_pmid({'pmid': '31234567'}, session=session) == '31234567'
    assert pubmed.resolve_pmid({'paper_id': 'pmid:22'}, session=session) == '22'
    assert pubmed.resolve_pmid({}, session=session) == ''
    assert session.calls == []

    session = FakeSession([FakeResponse(payload={'esearchresult': {'count': '1', 'idlist': ['77']}})])
    assert pubmed.resolve_pmid({'doi': '10.1234/a'}, session=session) == '77'


def test_resolve_pmcid_reads_the_identifier_from_the_pubmed_record() -> None:
    """Return a stored PMC identifier or take it from the resolved record."""
    session = FakeSession([])
    assert pubmed.resolve_pmcid({'pmcid': 'PMC1'}, session=session) == 'PMC1'
    assert pubmed.resolve_pmcid({}, session=session) == ''
    assert session.calls == []

    session = FakeSession([FakeResponse(text=article_set())])
    assert pubmed.resolve_pmcid({'pmid': '31234567'}, session=session) == 'PMC9876543'
    assert session.calls[0]['url'] == pubmed.EFETCH_URL

    session = FakeSession([FakeResponse(text='<PubmedArticleSet/>')])
    assert pubmed.resolve_pmcid({'pmid': '99'}, session=session) == ''


def test_oa_package_urls_order_pdfs_first_and_rewrite_ftp_links() -> None:
    """Offer PDF links before packages and hand back links requests can fetch."""
    payload = ('<OA><records><record>'
               '<link format="tgz" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa/a.tar.gz"/>'
               '<link format="pdf" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa/a.pdf"/>'
               '</record></records></OA>')
    session = FakeSession([FakeResponse(text=payload)])
    assert pubmed.oa_package_urls('PMC9876543', session=session) == [
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa/a.pdf',
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa/a.pdf',
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa/a.tar.gz',
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa/a.tar.gz',
    ]
    assert session.calls[0]['params']['id'] == 'PMC9876543'
    assert pubmed.oa_package_urls('', session=FakeSession([])) == []


def test_https_urls_offer_both_dataset_locations_and_pass_others_through() -> None:
    """Cover the relocated dataset tree and its temporary mirror."""
    assert pubmed._https_urls('ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/a/b.pdf') == [
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_pdf/a/b.pdf',
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/a/b.pdf',
    ]
    assert pubmed._https_urls('https://example.org/a.pdf') == ['https://example.org/a.pdf']


def test_oa_package_urls_return_nothing_for_a_record_outside_the_oa_subset() -> None:
    """Report a closed-access record as offering no links, not as a failure."""
    session = FakeSession([FakeResponse(text='<OA><records/></OA>')])
    assert pubmed.oa_package_urls('PMC1', session=session) == []


def test_pmc_full_text_flattens_prose_and_skips_tables_figures_and_references() -> None:
    """Keep section titles and paragraphs while dropping non-prose blocks."""
    session = FakeSession([FakeResponse(text=jats_article())])
    text = pubmed.pmc_full_text('PMC9876543', session=session)
    assert text.startswith('Garnet electrolytes')
    assert 'Introduction' in text
    assert 'Garnets conduct lithium at room temperature.' in text
    assert 'Conductivity reached 1 mS/cm.' in text
    assert 'Do not extract me' not in text
    assert 'Skip this caption.' not in text
    assert 'Skip this reference.' not in text
    assert session.calls[0]['params']['db'] == 'pmc'
    assert session.calls[0]['params']['id'] == '9876543'


def test_pmc_full_text_returns_empty_for_a_missing_identifier_or_body() -> None:
    """Return no text when there is nothing to fetch or no body to flatten."""
    assert pubmed.pmc_full_text('', session=FakeSession([])) == ''
    session = FakeSession([FakeResponse(text='<pmc-articleset><article/></pmc-articleset>')])
    assert pubmed.pmc_full_text('PMC1', session=session) == ''


def test_configured_credentials_fall_back_through_settings_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the NCBI settings first and reuse the Crossref address as a fallback."""
    monkeypatch.delenv('NCBI_API_KEY', raising=False)
    monkeypatch.delenv('NCBI_EMAIL', raising=False)
    assert pubmed.configured_api_key({'ncbi_api_key': 'key'}) == 'key'
    assert pubmed.configured_api_key({}) is None
    monkeypatch.setenv('NCBI_API_KEY', 'env-key')
    assert pubmed.configured_api_key({}) == 'env-key'

    assert pubmed.configured_email({'ncbi_email': 'ncbi@example.com'}) == 'ncbi@example.com'
    assert pubmed.configured_email({'crossref_email': 'cr@example.com'}) == 'cr@example.com'
    assert pubmed.configured_email({}) == ''


@pytest.mark.network
def test_pubmed_returns_a_known_record_from_the_live_api() -> None:
    """Fetch a stable PubMed record from the live E-utilities service."""
    articles = pubmed.parse_articles(pubmed.efetch_ids(['31234567']))
    assert len(articles) == 1
    assert articles[0]['pmid'] == '31234567'
    assert articles[0]['title']


@pytest.mark.network
def test_resolve_pmcid_reaches_the_live_service_for_an_open_access_paper() -> None:
    """Resolve a DOI to its PMC identifier through the live E-utilities service."""
    assert pubmed.resolve_pmcid({'doi': '10.1039/d3sc03514j'}) == 'PMC10530773'


def test_article_ids_ignore_the_identifiers_of_cited_references() -> None:
    """Read a record's own identifiers even when its references carry their own."""
    record = ET.fromstring('''<PubmedArticle>
      <MedlineCitation><PMID>1</PMID></MedlineCitation>
      <PubmedData>
        <ArticleIdList>
          <ArticleId IdType="pubmed">1</ArticleId>
          <ArticleId IdType="doi">10.1234/own</ArticleId>
          <ArticleId IdType="pmc">PMC1</ArticleId>
        </ArticleIdList>
        <ReferenceList><Reference><ArticleIdList>
          <ArticleId IdType="doi">10.9999/cited</ArticleId>
          <ArticleId IdType="pmc">PMC999</ArticleId>
        </ArticleIdList></Reference></ReferenceList>
      </PubmedData>
    </PubmedArticle>''')

    assert pubmed._article_ids(record) == {'doi': '10.1234/own', 'pmid': '1', 'pmcid': 'PMC1'}


def test_article_ids_do_not_borrow_a_reference_doi_when_the_record_has_none() -> None:
    """Leave the DOI empty rather than taking one from a cited reference."""
    record = ET.fromstring('''<PubmedArticle>
      <MedlineCitation><PMID>2</PMID></MedlineCitation>
      <PubmedData>
        <ReferenceList><Reference><ArticleIdList>
          <ArticleId IdType="doi">10.9999/cited</ArticleId>
        </ArticleIdList></Reference></ReferenceList>
      </PubmedData>
    </PubmedArticle>''')

    assert pubmed._article_ids(record) == {'doi': '', 'pmid': '2', 'pmcid': ''}
