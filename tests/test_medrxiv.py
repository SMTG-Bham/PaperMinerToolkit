"""Unit tests for medRxiv API request helpers and medRxiv record mapping."""

from __future__ import annotations

import json
from typing import Any

import pytest

import paperminer.medrxiv as medrxiv
from paperminer import _rxiv, provider

from tests.doubles import FakeResponse, FakeSession


def collection() -> list[dict[str, Any]]:
    """Return the five shared fixture records.

    The records are chosen so that one of each awkward shape is covered: two
    postings of one preprint that has since been published, so the published
    DOI, the version collapse, and the first-posting date all have something to
    act on; an unpublished preprint whose absent fields arrive as the ``NA``
    string medRxiv writes instead of an empty one; a record carrying the newer
    ``10.64898`` DOI prefix; and a degenerate record with almost nothing on it.
    """
    return [
        {
            'doi': '10.1101/2020.09.09.20191205',
            'title': 'Evolution of immunity to SARS-CoV-2',
            'authors': 'Wheatley, A. K.; Juno, J. A.; Kent, S. J.',
            'author_corresponding': 'Stephen J Kent',
            'author_corresponding_institution': 'University of Melbourne',
            'date': '2020-09-10',
            'version': '1',
            'type': 'PUBLISHAHEADOFPRINT',
            'license': 'cc_no',
            'category': 'Infectious Diseases',
            'jatsxml': 'https://www.medrxiv.org/content/early/2020/09/10/2020.09.09.20191205.source.xml',
            'abstract': 'The durability of\n  infection-induced immunity.',
            'published': '10.1038/s41467-021-21444-5',
            'server': 'medRxiv',
        },
        {
            'doi': '10.1101/2020.09.09.20191205',
            'title': 'Evolution of immunity to SARS-CoV-2',
            'authors': 'Wheatley, A. K.; Juno, J. A.; Kent, S. J.',
            'date': '2020-09-11',
            'version': '2',
            'license': 'cc_no',
            'category': 'Infectious Diseases',
            'jatsxml': 'https://www.medrxiv.org/content/early/2020/09/11/2020.09.09.20191205.source.xml',
            'abstract': 'The durability of infection-induced immunity, revised.',
            'published': '10.1038/s41467-021-21444-5',
            'server': 'medRxiv',
        },
        {
            'doi': '10.1101/2024.03.01.24303596',
            'title': 'Ultrasound-guided blocks for renal cancer resection',
            'authors': 'Xu, Guangmin; Li, W.',
            'date': '2024-03-04',
            'version': '1',
            'license': 'cc_by',
            'category': 'Health Policy',
            'jatsxml': 'https://www.medrxiv.org/content/early/2024/03/04/2024.03.01.24303596.source.xml',
            'abstract': 'A single-centre randomized controlled trial.',
            'funder': 'NA',
            'published': 'NA',
            'server': 'medRxiv',
        },
        {
            'doi': '10.64898/2026.08.05.26359794',
            'title': 'Machine learning triage in emergency departments',
            'authors': 'Okonkwo, N.',
            'date': '2026-08-06',
            'version': '3',
            'license': 'cc_by_nd',
            'category': 'Health Informatics',
            'jatsxml': 'https://www.medrxiv.org/content/early/2026/08/06/2026.08.05.26359794.source.xml',
            'abstract': 'Triage models evaluated prospectively.',
            'published': 'NA',
            'server': 'medRxiv',
        },
        {
            'doi': '10.1101/2023.01.02.23000001',
            'title': 'A sparsely described posting',
            'date': '2023-01-05',
            'version': '1',
            'server': 'medRxiv',
        },
    ]


def interval_payload(cursor: int = 0, total: int = 5) -> str:
    """Return an interval payload wrapping the shared fixture records."""
    return json.dumps({
        'messages': [{'status': 'ok', 'interval': '2024-01-01:2024-12-31', 'cursor': cursor,
                      'count': len(collection()), 'count_new_papers': 4, 'total': str(total)}],
        'collection': collection(),
    })


def empty_payload(status: str = 'no posts found') -> str:
    """Return a payload reporting that medRxiv holds nothing to return."""
    return json.dumps({'messages': [{'status': status}], 'collection': []})


def parsed_records() -> list[dict[str, Any]]:
    """Return the shared fixture records mapped onto the paper schema."""
    return medrxiv.parse_records(json.loads(interval_payload()))


def test_record_to_paper_prefers_the_published_doi_for_the_paper_id() -> None:
    """Use the published DOI so a preprint merges with its published row."""
    record = parsed_records()[0]
    assert record['doi'] == '10.1038/s41467-021-21444-5'
    assert record['paper_id'] == 'doi:10.1038/s41467-021-21444-5'
    assert record['medrxiv_doi'] == '10.1101/2020.09.09.20191205'
    assert record['published_doi'] == '10.1038/s41467-021-21444-5'
    assert record['sources'] == 'medrxiv'
    assert record['publication_date'] == '2020-09-10'


def test_record_to_paper_leaves_the_journal_empty_for_a_published_preprint() -> None:
    """Withhold a venue that would mask the journal Crossref holds."""
    published, unpublished = parsed_records()[0], parsed_records()[2]
    assert published['journal'] == ''
    assert unpublished['journal'] == 'medRxiv'
    assert unpublished['paper_id'] == 'doi:10.1101/2024.03.01.24303596'
    assert unpublished['published_doi'] == ''


def test_record_to_paper_reads_the_na_placeholder_as_an_absent_value() -> None:
    """Treat medRxiv's ``NA`` spelling as missing rather than as data."""
    assert provider.clean_text('NA') == ''
    assert provider.clean_text('n/a') == ''
    assert provider.clean_text(None) == ''
    assert provider.clean_text('  spaced   out  ') == 'spaced out'
    assert parsed_records()[3]['published_doi'] == ''


def test_record_to_paper_flips_author_names_into_the_corpus_order() -> None:
    """Rewrite ``Family, G.`` as ``G. Family`` to match the other providers."""
    assert parsed_records()[0]['authors'] == 'A. K. Wheatley; J. A. Juno; S. J. Kent'
    assert _rxiv._authors('Okonkwo, N.') == 'N. Okonkwo'
    # A name with no comma is a collaboration rather than a person, so it stands.
    assert _rxiv._authors('The RECOVERY Collaborative Group') == (
        'The RECOVERY Collaborative Group')
    assert _rxiv._authors('') == ''


def test_record_to_paper_builds_a_versioned_pdf_location() -> None:
    """Point at the posted version the record describes, not at version one."""
    assert parsed_records()[3]['pdf_url'] == (
        f'{medrxiv.WEB_URL}/content/10.64898/2026.08.05.26359794v3.full.pdf')
    assert medrxiv.pdf_url('10.1101/2024.03.01.24303596') == (
        f'{medrxiv.WEB_URL}/content/10.1101/2024.03.01.24303596v1.full.pdf')
    assert medrxiv.pdf_url('10.1101/2024.03.01.24303596v4') == (
        f'{medrxiv.WEB_URL}/content/10.1101/2024.03.01.24303596v4.full.pdf')
    assert medrxiv.pdf_url('') == ''


def test_record_to_paper_survives_a_record_missing_almost_every_field() -> None:
    """Return empty values instead of raising on a sparsely populated record."""
    record = parsed_records()[4]
    assert record['authors'] == ''
    assert record['abstract'] == ''
    assert record['categories'] == []
    assert record['category'] == ''
    assert record['license'] == ''
    assert record['paper_id'] == 'doi:10.1101/2023.01.02.23000001'


def test_categories_carry_the_single_primary_subject_medrxiv_files_under() -> None:
    """Emit one flagged category, keyed lower-case and displayed as written."""
    assert parsed_records()[0]['categories'] == [
        {'id': 'infectious diseases', 'name': 'Infectious Diseases', 'is_primary': True}]
    assert parsed_records()[0]['category'] == 'Infectious Diseases'


def test_parse_records_maps_every_record_and_tolerates_none() -> None:
    """Return one mapping per record, and an empty list for a missing payload."""
    assert len(parsed_records()) == 5
    assert medrxiv.parse_records(None) == []
    assert medrxiv.parse_records(json.loads(empty_payload())) == []
    assert medrxiv.parse_records({'collection': 'not a list'}) == []


def test_latest_versions_keeps_the_newest_posting_and_the_first_date() -> None:
    """Collapse the versions of one preprint without moving its posting date."""
    papers = medrxiv.latest_versions(parsed_records())

    assert len(papers) == 4
    revised = papers[0]
    assert revised['version'] == '2'
    assert revised['abstract'].endswith('revised.')
    assert revised['publication_date'] == '2020-09-10'
    # Ordering follows first appearance, so the caller's page order survives.
    assert [paper['medrxiv_doi'] for paper in papers][1:] == [
        '10.1101/2024.03.01.24303596',
        '10.64898/2026.08.05.26359794',
        '10.1101/2023.01.02.23000001',
    ]
    assert medrxiv.latest_versions([]) == []


def test_latest_versions_collapses_the_same_whichever_order_versions_arrive() -> None:
    """Apply both rules to a newest-first walk as well as an oldest-first fetch."""
    ascending = medrxiv.latest_versions(parsed_records()[:2])
    descending = medrxiv.latest_versions(list(reversed(parsed_records()[:2])))

    assert ascending == descending
    assert descending[0]['version'] == '2'
    assert descending[0]['publication_date'] == '2020-09-10'


def test_latest_versions_does_not_let_a_sparse_revision_blank_a_field() -> None:
    """Keep a value the earlier posting stated when the later one omits it."""
    first = medrxiv.record_to_paper({'doi': '10.1101/2024.01.01.24300001', 'version': '1',
                                     'date': '2024-01-02', 'license': 'cc_by',
                                     'category': 'Oncology', 'title': 'A trial'})
    revised = medrxiv.record_to_paper({'doi': '10.1101/2024.01.01.24300001', 'version': '2',
                                       'date': '2024-02-02', 'title': 'A trial, revised'})
    merged = medrxiv.latest_versions([first, revised])[0]

    assert merged['version'] == '2'
    assert merged['title'] == 'A trial, revised'
    assert merged['license'] == 'cc_by'
    assert merged['category'] == 'Oncology'


def test_total_results_and_page_size_read_the_message_block() -> None:
    """Read the record count and page length medRxiv reports alongside a page."""
    payload = json.loads(interval_payload(total=1281))
    assert medrxiv.total_results(payload) == 1281
    assert medrxiv.page_size(payload) == 5
    assert medrxiv.total_results(None) == 0
    assert medrxiv.total_results(json.loads(empty_payload())) == 0
    # A payload reporting no usable length falls back rather than stepping zero.
    assert medrxiv.page_size(json.loads(empty_payload()), default=30) == 30
    assert medrxiv.page_size(None) == medrxiv.PAGE_SIZE


def test_page_cursors_walks_pages_from_the_last_one_back_to_zero() -> None:
    """Start at the final page so the newest postings are read first."""
    assert list(medrxiv.page_cursors(1281, 100)) == list(range(1200, -1, -100))
    assert list(medrxiv.page_cursors(250, 30)) == list(range(240, -1, -30))
    # An exact multiple must not produce an empty page past the end.
    assert list(medrxiv.page_cursors(200, 100)) == [100, 0]
    assert list(medrxiv.page_cursors(1, 100)) == [0]
    assert list(medrxiv.page_cursors(0, 100)) == []
    assert list(medrxiv.page_cursors(50, 0)) == list(range(49, -1, -1))


def test_normalize_medrxiv_doi_accepts_both_prefixes_urls_and_versions() -> None:
    """Reduce every identifier presentation to one bare, unversioned DOI."""
    assert medrxiv.normalize_medrxiv_doi('10.1101/2024.03.01.24303596v2') == (
        '10.1101/2024.03.01.24303596')
    assert medrxiv.normalize_medrxiv_doi('doi:10.64898/2026.08.05.26359794') == (
        '10.64898/2026.08.05.26359794')
    assert medrxiv.normalize_medrxiv_doi(
        'https://www.medrxiv.org/content/10.1101/2020.09.09.20191205v2.full.pdf') == (
        '10.1101/2020.09.09.20191205')
    assert medrxiv.normalize_medrxiv_doi('https://doi.org/10.1101/2024.03.01.24303596') == (
        '10.1101/2024.03.01.24303596')
    assert medrxiv.normalize_medrxiv_doi('10.1038/s41467-021-21444-5') == ''
    assert medrxiv.normalize_medrxiv_doi('') == ''
    assert medrxiv.normalize_medrxiv_doi(None) == ''


def test_medrxiv_version_reads_the_suffix_when_one_is_present() -> None:
    """Report the version number separately from the bare DOI."""
    assert medrxiv.medrxiv_version('10.1101/2024.03.01.24303596v2') == '2'
    assert medrxiv.medrxiv_version('10.1101/2024.03.01.24303596') == ''
    assert medrxiv.medrxiv_version(None) == ''


def test_endpoint_trades_page_width_for_the_category_filter() -> None:
    """Address the filtering host only when a category is actually wanted."""
    assert medrxiv.endpoint() == (medrxiv.BASE_URL, medrxiv.PAGE_SIZE)
    assert medrxiv.endpoint('  ') == (medrxiv.BASE_URL, medrxiv.PAGE_SIZE)
    assert medrxiv.endpoint('oncology') == (medrxiv.CATEGORY_BASE_URL,
                                            medrxiv.CATEGORY_PAGE_SIZE)


def test_interval_url_rejects_a_bound_that_is_not_an_iso_date() -> None:
    """Refuse a malformed interval here rather than spending a request on it."""
    assert medrxiv.interval_url('2024-01-01', '2024-12-31', 200) == (
        f'{medrxiv.BASE_URL}/details/medrxiv/2024-01-01/2024-12-31/200/json')
    assert medrxiv.interval_url('2024-01-01', '2024-12-31', -5).endswith('/0/json')
    with pytest.raises(ValueError, match='start_date must be an ISO date'):
        medrxiv.interval_url('last week', '2024-12-31')
    with pytest.raises(ValueError, match='end_date must be an ISO date'):
        medrxiv.interval_url('2024-01-01', '')







def test_request_json_reads_an_absent_record_out_of_a_200_response() -> None:
    """Treat the statuses that mean "nothing here" as an empty result."""
    for status in ['no posts found', 'DOI not recognizable']:
        session = FakeSession([FakeResponse(text=empty_payload(status))])
        assert medrxiv.request_json(medrxiv.BASE_URL, session=session) is None


def test_request_json_raises_on_a_rejection_dressed_as_a_200_response() -> None:
    """Detect a rejected request that arrives with a successful status code."""
    session = FakeSession([FakeResponse(text=empty_payload('Both dates must be in yyyy-mm-dd'))])
    with pytest.raises(RuntimeError, match='Both dates must be in yyyy-mm-dd'):
        medrxiv.request_json(medrxiv.BASE_URL, session=session)


def test_request_json_reports_malformed_and_unexpected_bodies() -> None:
    """Raise on a body that is not the JSON object the API documents."""
    session = FakeSession([FakeResponse(text='{broken')])
    with pytest.raises(RuntimeError, match='medRxiv returned malformed JSON'):
        medrxiv.request_json(medrxiv.BASE_URL, session=session)

    session = FakeSession([FakeResponse(text='[1, 2, 3]')])
    with pytest.raises(RuntimeError, match='unexpected payload of type list'):
        medrxiv.request_json(medrxiv.BASE_URL, session=session)


def test_interval_page_sends_the_category_to_the_host_that_applies_it() -> None:
    """Address the filtering host and pass the category it needs."""
    session = FakeSession([FakeResponse(text=interval_payload())])
    medrxiv.interval_page('2024-01-01', '2024-12-31', cursor=60, category='Oncology',
                          session=session)

    assert session.calls[0]['url'] == (
        f'{medrxiv.CATEGORY_BASE_URL}/details/medrxiv/2024-01-01/2024-12-31/60/json')
    assert session.calls[0]['params'] == {'category': 'Oncology'}
    assert session.calls[0]['headers']['User-Agent'] == provider.USER_AGENT

    session = FakeSession([FakeResponse(text=interval_payload())])
    medrxiv.interval_page('2024-01-01', '2024-12-31', session=session)
    assert session.calls[0]['url'].startswith(medrxiv.BASE_URL)
    assert session.calls[0]['params'] == {}


def test_details_requests_every_version_and_skips_an_unusable_doi() -> None:
    """Ask for the full version history, and make no request without a DOI."""
    session = FakeSession([FakeResponse(text=interval_payload())])
    medrxiv.details('https://www.medrxiv.org/content/10.1101/2020.09.09.20191205v2.full.pdf',
                    session=session)

    assert session.calls[0]['url'] == (
        f'{medrxiv.BASE_URL}/details/medrxiv/10.1101/2020.09.09.20191205/na/json')

    session = FakeSession([])
    assert medrxiv.details('10.1038/s41467-021-21444-5', session=session) is None
    assert session.calls == []


def test_fetch_doi_returns_the_newest_posting_of_one_preprint() -> None:
    """Collapse the returned versions into the single record callers expect."""
    session = FakeSession([FakeResponse(text=interval_payload())])
    record = medrxiv.fetch_doi('10.1101/2020.09.09.20191205', session=session)

    assert record is not None
    assert record['version'] == '2'
    assert record['medrxiv_doi'] == '10.1101/2020.09.09.20191205'

    session = FakeSession([FakeResponse(text=empty_payload('DOI not recognizable'))])
    assert medrxiv.fetch_doi('10.1101/2020.09.09.20191205', session=session) is None


def test_resolve_medrxiv_doi_reads_stored_values_without_a_request() -> None:
    """Recover the identifier from whichever column already carries it."""
    assert medrxiv.resolve_medrxiv_doi(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596v2'}) == '10.1101/2024.03.01.24303596'
    assert medrxiv.resolve_medrxiv_doi(
        {'paper_id': 'doi:10.64898/2026.08.05.26359794'}) == '10.64898/2026.08.05.26359794'
    assert medrxiv.resolve_medrxiv_doi(
        {'doi': '10.1101/2023.01.02.23000001'}) == '10.1101/2023.01.02.23000001'
    assert medrxiv.resolve_medrxiv_doi(
        {'pdf_url': 'https://www.medrxiv.org/content/10.1101/2020.09.09.20191205v1.full.pdf'}
    ) == '10.1101/2020.09.09.20191205'
    # A published DOI names the journal version, which medRxiv does not index.
    assert medrxiv.resolve_medrxiv_doi({'doi': '10.1038/s41467-021-21444-5'}) == ''
    # A PDF hosted elsewhere is not a medRxiv location even if a DOI matches.
    assert medrxiv.resolve_medrxiv_doi(
        {'pdf_url': 'https://example.org/10.1101/2020.09.09.20191205.pdf'}) == ''
    assert medrxiv.resolve_medrxiv_doi({}) == ''


def test_full_text_flattens_the_jats_document_medrxiv_publishes() -> None:
    """Return the article title and prose, skipping tables and references."""
    jats = ('<article><front><article-meta><title-group>'
            '<article-title>Evolution of immunity</article-title>'
            '</title-group></article-meta></front><body>'
            '<sec><title>Results</title><p>Neutralising activity was widespread.</p>'
            '<table-wrap><p>Table residue</p></table-wrap></sec>'
            '<sec><title>Discussion</title><p>Immunity persisted.</p></sec>'
            '</body><back><ref-list><ref><p>A citation</p></ref></ref-list></back></article>')
    session = FakeSession([FakeResponse(text=jats)])
    text = medrxiv.full_text({'jatsxml': 'https://www.medrxiv.org/x.source.xml'}, session=session)

    assert text.startswith('Evolution of immunity')
    assert 'Neutralising activity was widespread.' in text
    assert 'Immunity persisted.' in text
    assert 'Table residue' not in text
    assert 'A citation' not in text


def test_full_text_reports_a_broken_document_and_skips_an_absent_one() -> None:
    """Raise on unparseable JATS but treat a bodyless record as no text."""
    session = FakeSession([])
    assert medrxiv.full_text({}, session=session) == ''
    assert session.calls == []

    session = FakeSession([FakeResponse(text='<article')])
    with pytest.raises(RuntimeError, match='medRxiv returned malformed JATS XML'):
        medrxiv.full_text({'jatsxml': 'https://www.medrxiv.org/x.source.xml'}, session=session)

    session = FakeSession([FakeResponse(text='<article><front/></article>')])
    assert medrxiv.full_text({'jatsxml': 'https://www.medrxiv.org/x.source.xml'},
                             session=session) == ''

    session = FakeSession([FakeResponse(text='   ')])
    assert medrxiv.full_text({'jatsxml': 'https://www.medrxiv.org/x.source.xml'},
                             session=session) == ''


def test_parse_query_lifts_the_scope_terms_out_of_the_phrase() -> None:
    """Separate the walk's bounds from the words that have to match."""
    terms, scope = medrxiv.parse_query(
        '"vaccine hesitancy" uptake category:"Infectious Diseases" '
        'from:2024-01-01 to:2024-06-30')

    assert terms == ['vaccine hesitancy', 'uptake']
    assert scope == {'category': 'Infectious Diseases', 'from': '2024-01-01', 'to': '2024-06-30'}
    assert medrxiv.parse_query('CATEGORY:oncology') == ([], {'category': 'oncology'})
    assert medrxiv.parse_query('   ') == ([], {})


def test_parse_query_rejects_a_bound_that_is_not_an_iso_date() -> None:
    """Refuse a date the interval endpoint would reject anyway."""
    with pytest.raises(ValueError, match='from: must be an ISO date'):
        medrxiv.parse_query('covid from:last-year')
    with pytest.raises(ValueError, match='to: must be an ISO date'):
        medrxiv.parse_query('covid to:2024')


def test_matches_combines_terms_with_and_across_the_record_text() -> None:
    """Require every term, and read the whole record rather than the title."""
    record = parsed_records()[0]

    assert medrxiv.matches(record, ['immunity'])
    assert medrxiv.matches(record, ['immunity', 'SARS-CoV-2'])
    # The abstract, the authors, and the category are searched alongside it.
    assert medrxiv.matches(record, ['durability'])
    assert medrxiv.matches(record, ['Wheatley'])
    assert medrxiv.matches(record, ['infectious'])
    assert not medrxiv.matches(record, ['immunity', 'lithium'])
    # An empty query matches everything, which is what a scope-only walk wants.
    assert medrxiv.matches(record, [])


def test_matches_reads_a_term_as_a_word_prefix_rather_than_a_substring() -> None:
    """Find plurals and hyphenated forms without matching mid-word noise."""
    record = medrxiv.record_to_paper({
        'doi': '10.1101/2024.01.01.24300001',
        'title': 'Vaccines and covid-19 outcomes',
        'abstract': 'Reported outcomes.',
        'date': '2024-01-02',
        'version': '1',
    })

    assert medrxiv.matches(record, ['vaccine'])
    assert medrxiv.matches(record, ['covid'])
    assert medrxiv.matches(record, ['covid-19'])
    # 'come' appears inside 'outcomes' but never starts a word there.
    assert not medrxiv.matches(record, ['come'])


@pytest.mark.network
def test_medrxiv_returns_a_known_record_from_the_live_api() -> None:
    """Fetch a stable medRxiv record from the live API service."""
    record = medrxiv.fetch_doi('10.1101/2020.09.09.20191205')
    assert record is not None
    assert record['medrxiv_doi'] == '10.1101/2020.09.09.20191205'
    assert record['title'] == 'Evolution of immunity to SARS-CoV-2'
    assert record['published_doi'] == '10.1038/s41467-021-21444-5'


@pytest.mark.network
def test_medrxiv_interval_walk_reaches_the_live_service() -> None:
    """Page one live date interval and confirm the walk covers every record."""
    first = medrxiv.interval_page('2024-03-01', '2024-03-03')
    total = medrxiv.total_results(first)
    step = medrxiv.page_size(first)
    assert total > 0

    seen = []
    for cursor in medrxiv.page_cursors(total, step):
        payload = first if cursor == 0 else medrxiv.interval_page('2024-03-01', '2024-03-03',
                                                                  cursor=cursor)
        seen.extend(medrxiv.parse_records(payload))
    assert len(seen) == total
    assert all(record['medrxiv_doi'] for record in seen)


@pytest.mark.network
def test_medrxiv_category_filter_is_applied_by_the_live_service() -> None:
    """Confirm the filtering host actually narrows the interval it returns."""
    scoped = medrxiv.interval_page('2024-03-01', '2024-03-05', category='infectious diseases')
    records = medrxiv.parse_records(scoped)
    assert records
    assert {record['category'] for record in records} == {'infectious diseases'}
