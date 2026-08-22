"""Unit tests for arXiv API request helpers and arXiv record mapping."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from typing import Any

import pytest
import requests

import paperscraper.arxiv as arxiv


def feed() -> str:
    """Return an Atom feed holding the four shared fixture entries.

    The entries are chosen so that one of each awkward shape is covered: a
    fully populated modern record whose title is line-wrapped the way arXiv
    wraps it and whose PDF link is not the first link; a pre-2007 identifier
    with no DOI and no journal reference; a degenerate entry with no authors,
    no PDF link and no primary category; and an entry whose title carries an
    escaped entity.
    """
    return '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>4213</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <title>Conductivity of a garnet
      solid electrolyte</title>
    <summary>  We report the ionic conductivity
      of a garnet framework.
    </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Grace Hopper</name></author>
    <published>2023-01-30T18:00:00Z</published>
    <updated>2023-03-02T09:15:00Z</updated>
    <arxiv:doi>10.1103/PhysRevB.108.014101</arxiv:doi>
    <arxiv:journal_ref>Phys. Rev. B 108, 014101 (2023)</arxiv:journal_ref>
    <arxiv:comment>12 pages, 5 figures</arxiv:comment>
    <arxiv:primary_category term="cond-mat.mtrl-sci"/>
    <category term="cond-mat.mtrl-sci"/>
    <category term="physics.chem-ph"/>
    <link href="http://arxiv.org/abs/2301.12345v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2301.12345v2" rel="related" title="pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cond-mat/0501001v1</id>
    <title>An early lattice study</title>
    <summary>An older preprint.</summary>
    <author><name>Alan Turing</name></author>
    <published>2005-01-03T12:00:00Z</published>
    <arxiv:primary_category term="cond-mat.str-el"/>
    <category term="cond-mat.str-el"/>
    <link href="http://arxiv.org/pdf/cond-mat/0501001v1" rel="related" title="pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2405.00001v1</id>
    <title>A withdrawn note</title>
    <summary/>
    <published>2024-05-01T00:00:00Z</published>
    <arxiv:comment>This submission has been withdrawn</arxiv:comment>
    <link href="http://arxiv.org/abs/2405.00001v1" rel="alternate" type="text/html"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2210.09999v1</id>
    <title>Spin &amp; charge order</title>
    <summary>Entities survive the mapping.</summary>
    <author><name>Emmy Noether</name></author>
    <published>2022-10-18T07:00:00Z</published>
    <arxiv:journal_ref>Nature 615, 1 (2023)</arxiv:journal_ref>
    <arxiv:primary_category term="cond-mat.supr-con"/>
    <category term="cond-mat.supr-con"/>
    <link href="http://arxiv.org/pdf/2210.09999v1" rel="related" title="pdf"/>
  </entry>
</feed>'''


def error_feed() -> str:
    """Return the HTTP 200 feed arXiv sends in place of an error status."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_bogus</id>
    <title>Error</title>
    <summary>incorrect id format for bogus</summary>
  </entry>
</feed>'''


def empty_feed() -> str:
    """Return a feed reporting no matches at all."""
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            '<opensearch:totalResults>0</opensearch:totalResults></feed>')


def parsed_entries() -> list[dict[str, Any]]:
    """Return the four shared fixture entries mapped onto the paper schema."""
    return arxiv.parse_entries(ET.fromstring(feed()))


class FakeResponse:
    """Prepared arXiv response with a configurable status code and body."""

    def __init__(self,
                 text: str = '',
                 status_code: int = 200,
                 headers: Mapping[str, str] | None = None) -> None:
        """Initialize the response test double."""
        self.text = text
        self.status_code = status_code
        self.headers = dict(headers or {})

    def raise_for_status(self) -> None:
        """Validate the prepared response status."""
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error', response=self)


class FakeSession:
    """Return prepared arXiv responses and record request arguments."""

    def __init__(self, responses: Iterable[FakeResponse]) -> None:
        """Initialize the session with prepared responses."""
        self.responses = iter(responses)
        self.calls = []

    def get(
        self,
        url: str,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> FakeResponse:
        """Return the next prepared response and record the request."""
        self.calls.append({
            'url': url,
            'params': dict(params),
            'headers': dict(headers),
            'timeout': timeout,
        })
        return next(self.responses)


@pytest.fixture(autouse=True)
def reset_pacer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the shared request window and silence pacing sleeps."""
    monkeypatch.setattr(arxiv, '_last_request_at', 0.0, raising=False)
    monkeypatch.setattr(arxiv.time, 'sleep', lambda _: None)


def test_element_text_collapses_the_line_wrapping_arxiv_applies() -> None:
    """Join a title arXiv wrapped across lines back into one string."""
    entry = parsed_entries()[0]
    assert entry['title'] == 'Conductivity of a garnet solid electrolyte'
    assert entry['abstract'] == 'We report the ionic conductivity of a garnet framework.'
    assert arxiv._element_text(None) == ''


def test_entry_to_paper_prefers_a_deposited_doi_for_the_paper_id() -> None:
    """Use the author-deposited DOI so a preprint merges with its published row."""
    entry = parsed_entries()[0]
    assert entry['doi'] == '10.1103/physrevb.108.014101'
    assert entry['paper_id'] == 'doi:10.1103/physrevb.108.014101'
    assert entry['arxiv_id'] == '2301.12345'
    assert entry['version'] == 'v2'
    assert entry['sources'] == 'arxiv'
    assert entry['authors'] == 'Ada Lovelace; Grace Hopper'
    assert entry['publication_date'] == '2023-01-30'
    assert entry['comment'] == '12 pages, 5 figures'


def test_entry_to_paper_falls_back_to_an_arxiv_paper_id_without_a_doi() -> None:
    """Key a DOI-less preprint by its arXiv identifier instead."""
    entry = parsed_entries()[1]
    assert entry['doi'] == ''
    assert entry['paper_id'] == 'arxiv:cond-mat/0501001'
    assert entry['arxiv_id'] == 'cond-mat/0501001'
    assert entry['journal'] == ''
    assert entry['authors'] == 'Alan Turing'


def test_entry_to_paper_takes_the_pdf_link_rather_than_the_first_link() -> None:
    """Skip the alternate HTML link that arXiv lists before the PDF one."""
    entries = parsed_entries()
    assert entries[0]['pdf_url'] == 'http://arxiv.org/pdf/2301.12345v2'
    # The degenerate entry offers only an HTML link, so the URL is constructed.
    assert entries[2]['pdf_url'] == f'{arxiv.PDF_URL}/2405.00001'


def test_entry_to_paper_survives_an_entry_missing_authors_and_a_category() -> None:
    """Return empty values instead of raising on a sparsely populated entry."""
    entry = parsed_entries()[2]
    assert entry['authors'] == ''
    assert entry['abstract'] == ''
    assert entry['categories'] == []
    assert entry['primary_category'] == ''
    assert entry['paper_id'] == 'arxiv:2405.00001'


def test_entry_to_paper_lists_the_primary_category_once_and_first() -> None:
    """Flag the primary category rather than repeating it among the others."""
    entry = parsed_entries()[0]
    assert entry['primary_category'] == 'cond-mat.mtrl-sci'
    assert [term['id'] for term in entry['categories']] == ['cond-mat.mtrl-sci',
                                                            'physics.chem-ph']
    assert [term['is_primary'] for term in entry['categories']] == [True, False]


def test_entry_to_paper_unescapes_entities_in_a_title() -> None:
    """Decode an escaped ampersand rather than leaving the entity in place."""
    assert parsed_entries()[3]['title'] == 'Spin & charge order'


def test_journal_name_trims_the_citation_around_a_journal_reference() -> None:
    """Keep the journal name and drop the volume, pages, and year."""
    assert parsed_entries()[0]['journal'] == 'Phys. Rev. B'
    assert parsed_entries()[0]['journal_ref'] == 'Phys. Rev. B 108, 014101 (2023)'
    assert parsed_entries()[3]['journal'] == 'Nature'
    assert arxiv._journal_name('') == ''
    # A reference that opens with a number has no name to isolate, so it stands.
    assert arxiv._journal_name('2023 Conference Proceedings') == '2023 Conference Proceedings'


def test_parse_entries_maps_every_entry_and_tolerates_none() -> None:
    """Return one mapping per entry, and an empty list for a missing document."""
    assert len(parsed_entries()) == 4
    assert arxiv.parse_entries(None) == []
    assert arxiv.parse_entries(ET.fromstring(empty_feed())) == []


def test_total_results_reads_the_opensearch_count() -> None:
    """Read the match total arXiv reports alongside the page of entries."""
    assert arxiv.total_results(ET.fromstring(feed())) == 4213
    assert arxiv.total_results(ET.fromstring(empty_feed())) == 0
    assert arxiv.total_results(None) == 0


def test_normalize_arxiv_id_strips_versions_labels_and_urls() -> None:
    """Reduce every identifier presentation to one bare, unversioned form."""
    assert arxiv.normalize_arxiv_id('http://arxiv.org/abs/2301.12345v2') == '2301.12345'
    assert arxiv.normalize_arxiv_id('https://arxiv.org/pdf/2301.12345v2.pdf') == '2301.12345'
    assert arxiv.normalize_arxiv_id('arXiv:2301.12345') == '2301.12345'
    assert arxiv.normalize_arxiv_id('2301.12345v11') == '2301.12345'
    assert arxiv.normalize_arxiv_id('cond-mat/0501001v1') == 'cond-mat/0501001'
    # An old-style subject class is case-sensitive and must survive intact.
    assert arxiv.normalize_arxiv_id('http://arxiv.org/abs/math.GT/0309136') == 'math.GT/0309136'
    assert arxiv.normalize_arxiv_id('') == ''
    assert arxiv.normalize_arxiv_id(None) == ''
    assert arxiv.normalize_arxiv_id('not an identifier') == ''


def test_arxiv_version_reads_the_suffix_when_one_is_present() -> None:
    """Report the version suffix separately from the bare identifier."""
    assert arxiv.arxiv_version('http://arxiv.org/abs/2301.12345v2') == 'v2'
    assert arxiv.arxiv_version('2301.12345') == ''
    assert arxiv.arxiv_version(None) == ''


def test_query_expression_wraps_a_plain_phrase_and_passes_native_queries_through() -> None:
    """Field-qualify a bare phrase but leave an arXiv expression untouched."""
    assert arxiv.query_expression('Lithium solid electrolyte') == (
        'all:Lithium AND all:solid AND all:electrolyte')
    assert arxiv.query_expression('"solid electrolyte" garnet') == (
        'all:"solid electrolyte" AND all:garnet')
    native = 'cat:cond-mat.mtrl-sci AND abs:"solid electrolyte"'
    assert arxiv.query_expression(native) == native
    assert arxiv.query_expression('ti:garnet') == 'ti:garnet'
    assert arxiv.query_expression('   ') == ''
    with pytest.raises(ValueError, match='default_field must be one of'):
        arxiv.query_expression('garnet', default_field='nope')


def test_request_params_omit_unset_values_and_reject_unknown_sorts() -> None:
    """Send only the parameters the caller set, and refuse invalid sort values."""
    assert arxiv.request_params(search_query='all:garnet') == {
        'search_query': 'all:garnet', 'max_results': arxiv.PAGE_SIZE}
    assert arxiv.request_params(id_list=['2301.12345', 'cond-mat/0501001'],
                                max_results=2)['id_list'] == '2301.12345,cond-mat/0501001'
    with pytest.raises(ValueError, match='sort_by must be one of'):
        arxiv.request_params(search_query='a', sort_by='nope')
    with pytest.raises(ValueError, match='sort_order must be one of'):
        arxiv.request_params(search_query='a', sort_order='sideways')


def test_request_paces_consecutive_calls_with_the_courtesy_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sleep the documented delay between requests using one shared window."""
    sleeps: list[float] = []
    clock = {'now': 100.0}
    monkeypatch.setattr(arxiv.time, 'sleep', lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(arxiv.time, 'monotonic', lambda: clock['now'])

    session = FakeSession([FakeResponse(text='<feed/>') for _ in range(3)])
    for _ in range(3):
        arxiv.request(session=session)
    assert sleeps == pytest.approx([arxiv.ARXIV_MIN_INTERVAL] * 2)


def test_request_returns_none_for_a_missing_document() -> None:
    """Treat a 404 as an absent record rather than as a failure."""
    session = FakeSession([FakeResponse(status_code=404)])
    assert arxiv.request(session=session) is None
    assert len(session.calls) == 1


def test_request_retries_a_rate_limited_response_and_honours_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait the advertised interval and retry rather than giving up on a 429."""
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv.time, 'sleep', lambda seconds: sleeps.append(seconds))
    session = FakeSession([
        FakeResponse(status_code=429, headers={'Retry-After': '5'}),
        FakeResponse(text='<feed/>'),
    ])

    assert arxiv.request(session=session) is not None
    assert len(session.calls) == 2
    assert 5 in sleeps


def test_request_raises_after_exhausting_every_attempt() -> None:
    """Report how many attempts were spent before the request was abandoned."""
    session = FakeSession([FakeResponse(status_code=500) for _ in range(4)])
    with pytest.raises(RuntimeError, match='arXiv request failed after 4 attempts'):
        arxiv.request(session=session)
    assert len(session.calls) == 4


def test_request_fails_immediately_on_a_client_error_other_than_a_rate_limit() -> None:
    """Spend one attempt on a terminal client error instead of retrying it."""
    session = FakeSession([FakeResponse(status_code=400)])
    with pytest.raises(RuntimeError, match='arXiv rejected the request with 400'):
        arxiv.request(session=session)
    assert len(session.calls) == 1


def test_request_xml_raises_on_the_error_feed_arxiv_returns_with_a_200() -> None:
    """Detect a rejected query that arrives dressed as a successful response."""
    session = FakeSession([FakeResponse(text=error_feed())])
    with pytest.raises(RuntimeError, match='incorrect id format for bogus'):
        arxiv.request_xml(session=session)


def test_request_xml_reports_malformed_bodies_and_skips_empty_ones() -> None:
    """Raise on unparseable XML but treat a blank body as no document."""
    session = FakeSession([FakeResponse(text='<broken')])
    with pytest.raises(RuntimeError, match='arXiv returned malformed XML'):
        arxiv.request_xml(session=session)

    session = FakeSession([FakeResponse(text='   ')])
    assert arxiv.request_xml(session=session) is None


def test_search_page_sends_the_paging_and_ordering_parameters() -> None:
    """Request a stable submission-date ordering with an explicit page window."""
    session = FakeSession([FakeResponse(text=feed())])
    root = arxiv.search_page('all:garnet', start=50, max_results=25, session=session)

    assert arxiv.total_results(root) == 4213
    assert session.calls[0]['url'] == arxiv.BASE_URL
    assert session.calls[0]['params'] == {
        'search_query': 'all:garnet',
        'start': 50,
        'max_results': 25,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
    }
    assert session.calls[0]['headers']['User-Agent'] == arxiv.USER_AGENT


def test_fetch_ids_normalizes_identifiers_and_skips_an_empty_batch() -> None:
    """Send a comma-joined id list, and make no request when none survive."""
    session = FakeSession([FakeResponse(text=feed())])
    arxiv.fetch_ids(['http://arxiv.org/abs/2301.12345v2', 'cond-mat/0501001v1'],
                    session=session)

    assert session.calls[0]['params']['id_list'] == '2301.12345,cond-mat/0501001'
    assert 'search_query' not in session.calls[0]['params']

    session = FakeSession([])
    assert arxiv.fetch_ids([' ', ''], session=session) is None
    assert session.calls == []


def test_find_arxiv_id_requires_the_title_to_match_the_record_it_found() -> None:
    """Reject arXiv's loose title match unless the titles are actually equal."""
    session = FakeSession([FakeResponse(text=feed())])
    assert arxiv.find_arxiv_id('An early lattice study', session=session) == 'cond-mat/0501001'
    assert session.calls[0]['params']['search_query'] == 'ti:"An early lattice study"'

    session = FakeSession([FakeResponse(text=feed())])
    assert arxiv.find_arxiv_id('A completely different paper', session=session) == ''

    session = FakeSession([])
    assert arxiv.find_arxiv_id('', session=session) == ''
    assert session.calls == []


def test_resolve_arxiv_id_prefers_stored_values_over_a_lookup() -> None:
    """Read an identifier already on the row without spending a request."""
    session = FakeSession([])
    assert arxiv.resolve_arxiv_id({'arxiv_id': '2301.12345v2'}, session=session) == '2301.12345'
    assert arxiv.resolve_arxiv_id({'paper_id': 'arxiv:cond-mat/0501001'},
                                  session=session) == 'cond-mat/0501001'
    assert arxiv.resolve_arxiv_id({'pdf_url': 'https://arxiv.org/pdf/2405.00001v1'},
                                  session=session) == '2405.00001'
    assert session.calls == []


def test_resolve_arxiv_id_falls_back_to_a_title_search() -> None:
    """Look a row up by title only when it stores no identifier at all."""
    session = FakeSession([FakeResponse(text=feed())])
    assert arxiv.resolve_arxiv_id({'title': 'An early lattice study'},
                                  session=session) == 'cond-mat/0501001'
    assert len(session.calls) == 1


@pytest.mark.network
def test_arxiv_returns_a_known_record_from_the_live_api() -> None:
    """Fetch a stable arXiv record from the live API service."""
    entries = arxiv.parse_entries(arxiv.fetch_ids(['1706.03762']))
    assert len(entries) == 1
    assert entries[0]['arxiv_id'] == '1706.03762'
    assert entries[0]['title'] == 'Attention Is All You Need'


@pytest.mark.network
def test_arxiv_search_reaches_the_live_service_for_a_category_query() -> None:
    """Run a fielded category query against the live arXiv service."""
    root = arxiv.search_page('cat:cond-mat.mtrl-sci', max_results=5)
    entries = arxiv.parse_entries(root)
    assert len(entries) == 5
    assert arxiv.total_results(root) > 0
    assert all(entry['arxiv_id'] for entry in entries)
