"""Unit tests for bioRxiv API request helpers and bioRxiv record mapping."""

from __future__ import annotations

import json
from typing import Any

import pytest

import paperminer.biorxiv as biorxiv
from paperminer import _rxiv, provider
import paperminer.medrxiv as medrxiv

from tests.doubles import FakeResponse, FakeSession


def collection() -> list[dict[str, Any]]:
    """Return the five shared fixture records.

    The records are chosen so that one of each awkward shape is covered: two
    postings of one preprint that has since been published, so the published
    DOI, the version collapse, and the first-posting date all have something to
    act on; an unpublished preprint whose absent fields arrive as the ``NA``
    string bioRxiv writes instead of an empty one; a record carrying the newer
    ``10.64898`` DOI prefix; and a degenerate record with almost nothing on it.
    """
    return [
        {
            'doi': '10.1101/2023.12.01.569634',
            'title': 'Visual working memories are abstractions of percepts',
            'authors': 'Duan, Z.; Curtis, C. E.',
            'author_corresponding': 'Clayton E Curtis',
            'author_corresponding_institution': 'NYU',
            'date': '2023-12-03',
            'version': '1',
            'type': 'new results',
            'license': 'cc_by_nc_nd',
            'category': 'Neuroscience',
            'jatsxml': 'https://www.biorxiv.org/content/early/2023/12/03/2023.12.01.569634.source.xml',
            'abstract': 'Decoding the orientation\n  of memorized gratings.',
            'published': '10.7554/elife.94191.3',
            'server': 'bioRxiv',
        },
        {
            'doi': '10.1101/2023.12.01.569634',
            'title': 'Visual working memories are abstractions of percepts',
            'authors': 'Duan, Z.; Curtis, C. E.',
            'date': '2024-03-01',
            'version': '2',
            'license': 'cc_by_nc_nd',
            'category': 'Neuroscience',
            'jatsxml': 'https://www.biorxiv.org/content/early/2024/03/01/2023.12.01.569634.source.xml',
            'abstract': 'Decoding the orientation of memorized gratings, revised.',
            'published': '10.7554/elife.94191.3',
            'server': 'bioRxiv',
        },
        {
            'doi': '10.1101/2024.03.01.583596',
            'title': 'Chromatin accessibility during limb regeneration',
            'authors': 'Ferreira, M.; Okonkwo, N.',
            'date': '2024-03-04',
            'version': '1',
            'license': 'cc_by',
            'category': 'Developmental Biology',
            'jatsxml': 'https://www.biorxiv.org/content/early/2024/03/04/2024.03.01.583596.source.xml',
            'abstract': 'Regenerating blastema profiled by ATAC-seq.',
            'funder': 'NA',
            'published': 'NA',
            'server': 'bioRxiv',
        },
        {
            'doi': '10.64898/2026.08.07.742070',
            'title': 'Colorimetric hydrogel dressing for wound pH monitoring',
            'authors': 'Matoori, S.',
            'date': '2026-08-10',
            'version': '3',
            'license': 'cc_no',
            'category': 'Bioengineering',
            'jatsxml': 'https://www.biorxiv.org/content/early/2026/08/10/2026.08.07.742070.source.xml',
            'abstract': 'Point-of-care wound monitoring with a smartphone detector.',
            'published': 'NA',
            'server': 'bioRxiv',
        },
        {
            'doi': '10.1101/2023.01.02.522001',
            'title': 'A sparsely described posting',
            'date': '2023-01-05',
            'version': '1',
            'server': 'bioRxiv',
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
    """Return a payload reporting that bioRxiv holds nothing to return."""
    return json.dumps({'messages': [{'status': status}], 'collection': []})


def parsed_records() -> list[dict[str, Any]]:
    """Return the shared fixture records mapped onto the paper schema."""
    return biorxiv.parse_records(json.loads(interval_payload()))


def test_record_to_paper_prefers_the_published_doi_for_the_paper_id() -> None:
    """Key a published preprint by the journal DOI so the rows merge."""
    record = biorxiv.record_to_paper(collection()[0])

    assert record['paper_id'] == 'doi:10.7554/elife.94191.3'
    assert record['doi'] == '10.7554/elife.94191.3'
    assert record['published_doi'] == '10.7554/elife.94191.3'
    assert record['biorxiv_doi'] == '10.1101/2023.12.01.569634'
    assert record['sources'] == 'biorxiv'
    assert record['metadata_status'] == 'retrieved'


def test_record_to_paper_leaves_the_journal_empty_for_a_published_preprint() -> None:
    """Avoid masking the journal Crossref holds for the published version."""
    published = biorxiv.record_to_paper(collection()[0])
    unpublished = biorxiv.record_to_paper(collection()[2])

    assert published['journal'] == ''
    assert unpublished['journal'] == 'bioRxiv'
    assert unpublished['paper_id'] == 'doi:10.1101/2024.03.01.583596'


def test_record_to_paper_reads_the_na_placeholder_as_an_absent_value() -> None:
    """Store the NA spelling bioRxiv writes as missing rather than as data."""
    record = biorxiv.record_to_paper(collection()[2])

    assert record['published_doi'] == ''
    assert provider.clean_text('NA') == ''
    assert provider.clean_text('n/a') == ''
    assert provider.clean_text(None) == ''
    assert provider.clean_text('  spaced   out  ') == 'spaced out'


def test_record_to_paper_flips_author_names_into_the_corpus_order() -> None:
    """Rewrite Family, G. I. as Given Family so rows compare across providers."""
    record = biorxiv.record_to_paper(collection()[0])

    assert record['authors'] == 'Z. Duan; C. E. Curtis'
    assert _rxiv._authors('Okonkwo, N.') == 'N. Okonkwo'
    # A name with no comma is a consortium rather than a person, and survives.
    assert _rxiv._authors('The ENCODE Project Consortium') == (
        'The ENCODE Project Consortium')
    assert _rxiv._authors('') == ''


def test_record_to_paper_builds_a_versioned_pdf_location() -> None:
    """Point the PDF URL at the posted version the record describes."""
    record = biorxiv.record_to_paper(collection()[3])

    assert record['pdf_url'] == (
        f'{biorxiv.WEB_URL}/content/10.64898/2026.08.07.742070v3.full.pdf')
    assert biorxiv.pdf_url('10.1101/2023.12.01.569634') == (
        f'{biorxiv.WEB_URL}/content/10.1101/2023.12.01.569634v1.full.pdf')
    assert biorxiv.pdf_url('10.1101/2023.12.01.569634v4') == (
        f'{biorxiv.WEB_URL}/content/10.1101/2023.12.01.569634v4.full.pdf')
    assert biorxiv.pdf_url('') == ''


def test_record_to_paper_survives_a_record_missing_almost_every_field() -> None:
    """Map a sparse posting without inventing values for what it omits."""
    record = biorxiv.record_to_paper(collection()[4])

    assert record['paper_id'] == 'doi:10.1101/2023.01.02.522001'
    assert record['authors'] == ''
    assert record['abstract'] == ''
    assert record['category'] == ''
    assert record['categories'] == []


def test_categories_carry_the_single_primary_subject_biorxiv_files_under() -> None:
    """Return the one category as a list, flagged primary, keyed lower-case."""
    record = biorxiv.record_to_paper(collection()[0])

    assert record['category'] == 'Neuroscience'
    assert record['categories'] == [
        {'id': 'neuroscience', 'name': 'Neuroscience', 'is_primary': True}]


def test_parse_records_maps_every_record_and_tolerates_none() -> None:
    """Map each collection entry, and treat an absent payload as no records."""
    assert len(parsed_records()) == 5
    assert biorxiv.parse_records(None) == []
    assert biorxiv.parse_records({'collection': 'not a list'}) == []
    assert biorxiv.parse_records(json.loads(empty_payload())) == []


def test_latest_versions_keeps_the_newest_posting_and_the_first_date() -> None:
    """Collapse a revised preprint onto its newest version and earliest date."""
    entries = biorxiv.latest_versions(parsed_records())

    assert len(entries) == 4
    revised = entries[0]
    assert revised['version'] == '2'
    assert revised['abstract'].endswith('revised.')
    # The paper is dated when it first appeared, not when it was last revised.
    assert revised['publication_date'] == '2023-12-03'
    assert [entry['biorxiv_doi'] for entry in entries] == [
        '10.1101/2023.12.01.569634',
        '10.1101/2024.03.01.583596',
        '10.64898/2026.08.07.742070',
        '10.1101/2023.01.02.522001',
    ]


def test_latest_versions_collapses_the_same_whichever_order_versions_arrive() -> None:
    """Reach one answer from a details walk and from a reversed search walk."""
    forward = biorxiv.latest_versions(parsed_records()[:2])
    backward = biorxiv.latest_versions(list(reversed(parsed_records()[:2])))

    assert forward[0]['version'] == backward[0]['version'] == '2'
    assert forward[0]['publication_date'] == backward[0]['publication_date'] == '2023-12-03'


def test_latest_versions_does_not_let_a_sparse_revision_blank_a_field() -> None:
    """Keep a value the earlier posting supplied when the newer one omits it."""
    first = biorxiv.record_to_paper({'doi': '10.1101/2024.01.01.570001', 'version': '1',
                                     'date': '2024-01-02', 'license': 'cc_by',
                                     'category': 'Genomics'})
    revised = biorxiv.record_to_paper({'doi': '10.1101/2024.01.01.570001', 'version': '2',
                                       'date': '2024-02-02'})

    merged = biorxiv.latest_versions([first, revised])[0]
    assert merged['version'] == '2'
    assert merged['license'] == 'cc_by'
    assert merged['category'] == 'Genomics'
    assert merged['publication_date'] == '2024-01-02'
    assert biorxiv.latest_versions([{'title': 'no identifier'}]) == []


def test_total_results_and_page_size_read_the_message_block() -> None:
    """Read the record count and page length bioRxiv reports alongside a page."""
    payload = json.loads(interval_payload(total=1281))
    assert biorxiv.total_results(payload) == 1281
    assert biorxiv.page_size(payload) == 5
    assert biorxiv.total_results(None) == 0
    assert biorxiv.total_results(json.loads(empty_payload())) == 0
    # A payload reporting no usable length falls back rather than stepping zero.
    assert biorxiv.page_size(json.loads(empty_payload()), default=30) == 30
    assert biorxiv.page_size(None) == biorxiv.PAGE_SIZE


def test_page_cursors_walks_pages_from_the_last_one_back_to_zero() -> None:
    """Start at the final page so the newest postings are read first."""
    assert list(biorxiv.page_cursors(1281, 100)) == list(range(1200, -1, -100))
    assert list(biorxiv.page_cursors(250, 30)) == list(range(240, -1, -30))
    # An exact multiple must not produce an empty page past the end.
    assert list(biorxiv.page_cursors(200, 100)) == [100, 0]
    assert list(biorxiv.page_cursors(1, 100)) == [0]
    assert list(biorxiv.page_cursors(0, 100)) == []


def test_normalize_biorxiv_doi_accepts_both_prefixes_urls_and_versions() -> None:
    """Reduce every identifier presentation to one bare, unversioned DOI."""
    assert biorxiv.normalize_biorxiv_doi('10.1101/2023.12.01.569634v2') == (
        '10.1101/2023.12.01.569634')
    assert biorxiv.normalize_biorxiv_doi('doi:10.64898/2026.08.07.742070') == (
        '10.64898/2026.08.07.742070')
    assert biorxiv.normalize_biorxiv_doi(
        'https://www.biorxiv.org/content/10.1101/2023.12.01.569634v2.full.pdf') == (
        '10.1101/2023.12.01.569634')
    assert biorxiv.normalize_biorxiv_doi('https://doi.org/10.1101/2024.03.01.583596') == (
        '10.1101/2024.03.01.583596')
    assert biorxiv.normalize_biorxiv_doi('10.7554/elife.94191.3') == ''
    assert biorxiv.normalize_biorxiv_doi('') == ''
    assert biorxiv.normalize_biorxiv_doi(None) == ''


def test_normalize_biorxiv_doi_accepts_the_pre_2018_bare_accession() -> None:
    """Recognize the DOI shape bioRxiv issued before it dated its accessions.

    Those identifiers were never reissued, and an old preprint can still be
    revised, so the shape appears in a walk of recent postings too. It is
    accepted only under ``10.1101``, the sole prefix that ever issued it, since
    a bare six-digit suffix is too plain to claim under any prefix at all.
    """
    assert biorxiv.normalize_biorxiv_doi('10.1101/060400') == '10.1101/060400'
    assert biorxiv.normalize_biorxiv_doi('10.1101/000075') == '10.1101/000075'
    assert biorxiv.normalize_biorxiv_doi('10.1101/060400v2') == '10.1101/060400'
    assert biorxiv.biorxiv_version('10.1101/060400v2') == '2'
    assert biorxiv.normalize_biorxiv_doi(
        'https://www.biorxiv.org/content/10.1101/060400v2.full.pdf') == '10.1101/060400'
    assert biorxiv.pdf_url('10.1101/060400', '2') == (
        f'{biorxiv.WEB_URL}/content/10.1101/060400v2.full.pdf')
    assert biorxiv.resolve_biorxiv_doi({'biorxiv_doi': '10.1101/060400'}) == '10.1101/060400'
    # The bare form belongs to the old prefix alone.
    assert biorxiv.normalize_biorxiv_doi('10.64898/060400') == ''
    assert biorxiv.normalize_biorxiv_doi('10.1234/567890') == ''


def test_normalize_biorxiv_doi_rejects_the_publishers_other_dois() -> None:
    """Leave Cold Spring Harbor's journal DOIs, which share the prefix, alone."""
    for value in ['10.1101/gr.123456.789', '10.1101/gad.123456.789',
                  '10.1101/cshperspect.a041234', '10.1101/pdb.prot5678']:
        assert biorxiv.normalize_biorxiv_doi(value) == ''


def test_the_two_preprint_servers_do_not_claim_each_others_dois() -> None:
    """Route each archive's DOIs to it alone, on the accession-number width.

    Both servers issue DOIs under ``10.1101`` and now ``10.64898``, so nothing
    but the accession number separates them: six digits here against medRxiv's
    year-prefixed eight. Were either pattern to accept the other's, every row
    from one archive would spend a paced request asking the other for a record
    it has never held.
    """
    for value in ['10.1101/2023.12.01.569634', '10.64898/2026.08.07.742070',
                  '10.1101/2023.12.01.569634v2', '10.1101/060400']:
        assert biorxiv.normalize_biorxiv_doi(value)
        assert medrxiv.normalize_medrxiv_doi(value) == ''

    for value in ['10.1101/2024.03.01.24303596', '10.64898/2026.08.05.26359794',
                  '10.1101/2024.03.01.24303596v2']:
        assert medrxiv.normalize_medrxiv_doi(value)
        assert biorxiv.normalize_biorxiv_doi(value) == ''


def test_biorxiv_version_reads_the_suffix_when_one_is_present() -> None:
    """Report the version number separately from the bare DOI."""
    assert biorxiv.biorxiv_version('10.1101/2023.12.01.569634v2') == '2'
    assert biorxiv.biorxiv_version('10.1101/2023.12.01.569634') == ''
    assert biorxiv.biorxiv_version(None) == ''


def test_endpoint_trades_page_width_for_the_category_filter() -> None:
    """Address the filtering host only when a category is actually wanted."""
    assert biorxiv.endpoint() == (biorxiv.BASE_URL, biorxiv.PAGE_SIZE)
    assert biorxiv.endpoint('  ') == (biorxiv.BASE_URL, biorxiv.PAGE_SIZE)
    assert biorxiv.endpoint('neuroscience') == (biorxiv.CATEGORY_BASE_URL,
                                                biorxiv.CATEGORY_PAGE_SIZE)
    # The wider host is the one named for the other archive; the server segment
    # of the path, not the host, is what selects bioRxiv's content.
    assert biorxiv.BASE_URL != biorxiv.CATEGORY_BASE_URL


def test_interval_url_rejects_a_bound_that_is_not_an_iso_date() -> None:
    """Refuse a malformed interval here rather than spending a request on it."""
    assert biorxiv.interval_url('2024-01-01', '2024-12-31', 200) == (
        f'{biorxiv.BASE_URL}/details/biorxiv/2024-01-01/2024-12-31/200/json')
    assert biorxiv.interval_url('2024-01-01', '2024-12-31', -5).endswith('/0/json')
    with pytest.raises(ValueError, match='start_date must be an ISO date'):
        biorxiv.interval_url('last week', '2024-12-31')
    with pytest.raises(ValueError, match='end_date must be an ISO date'):
        biorxiv.interval_url('2024-01-01', '')







def test_request_json_reads_an_absent_record_out_of_a_200_response() -> None:
    """Treat the statuses that mean "nothing here" as an empty result."""
    for status in ['no posts found', 'DOI not recognizable']:
        session = FakeSession([FakeResponse(text=empty_payload(status))])
        assert biorxiv.request_json(biorxiv.BASE_URL, session=session) is None


def test_request_json_raises_on_a_rejection_dressed_as_a_200_response() -> None:
    """Detect a rejected request that arrives with a successful status code."""
    session = FakeSession([FakeResponse(text=empty_payload('Both dates must be in yyyy-mm-dd'))])
    with pytest.raises(RuntimeError, match='Both dates must be in yyyy-mm-dd'):
        biorxiv.request_json(biorxiv.BASE_URL, session=session)


def test_request_json_reports_malformed_and_unexpected_bodies() -> None:
    """Raise on a body that is not the JSON object the API documents."""
    session = FakeSession([FakeResponse(text='{broken')])
    with pytest.raises(RuntimeError, match='bioRxiv returned malformed JSON'):
        biorxiv.request_json(biorxiv.BASE_URL, session=session)

    session = FakeSession([FakeResponse(text='[1, 2, 3]')])
    with pytest.raises(RuntimeError, match='unexpected payload of type list'):
        biorxiv.request_json(biorxiv.BASE_URL, session=session)


def test_interval_page_sends_the_category_to_the_host_that_applies_it() -> None:
    """Address the filtering host and pass the category it needs."""
    session = FakeSession([FakeResponse(text=interval_payload())])
    biorxiv.interval_page('2024-01-01', '2024-12-31', cursor=60, category='Neuroscience',
                          session=session)

    assert session.calls[0]['url'] == (
        f'{biorxiv.CATEGORY_BASE_URL}/details/biorxiv/2024-01-01/2024-12-31/60/json')
    assert session.calls[0]['params'] == {'category': 'Neuroscience'}
    assert session.calls[0]['headers']['User-Agent'] == provider.USER_AGENT

    session = FakeSession([FakeResponse(text=interval_payload())])
    biorxiv.interval_page('2024-01-01', '2024-12-31', session=session)
    assert session.calls[0]['url'].startswith(biorxiv.BASE_URL)
    assert session.calls[0]['params'] == {}


def test_details_requests_every_version_and_skips_an_unusable_doi() -> None:
    """Ask for the full version history, and make no request without a DOI."""
    session = FakeSession([FakeResponse(text=interval_payload())])
    biorxiv.details('https://www.biorxiv.org/content/10.1101/2023.12.01.569634v2.full.pdf',
                    session=session)

    assert session.calls[0]['url'] == (
        f'{biorxiv.BASE_URL}/details/biorxiv/10.1101/2023.12.01.569634/na/json')

    session = FakeSession([])
    assert biorxiv.details('10.7554/elife.94191.3', session=session) is None
    assert session.calls == []
    # A medRxiv DOI is not addressable here, and costs no request to refuse.
    assert biorxiv.details('10.1101/2024.03.01.24303596', session=session) is None
    assert session.calls == []


def test_fetch_doi_returns_the_newest_posting_of_one_preprint() -> None:
    """Collapse the returned versions into the single record callers expect."""
    session = FakeSession([FakeResponse(text=interval_payload())])
    record = biorxiv.fetch_doi('10.1101/2023.12.01.569634', session=session)

    assert record is not None
    assert record['version'] == '2'
    assert record['biorxiv_doi'] == '10.1101/2023.12.01.569634'

    session = FakeSession([FakeResponse(text=empty_payload('DOI not recognizable'))])
    assert biorxiv.fetch_doi('10.1101/2023.12.01.569634', session=session) is None


def test_resolve_biorxiv_doi_reads_stored_values_without_a_request() -> None:
    """Recover the identifier from whichever column already carries it."""
    assert biorxiv.resolve_biorxiv_doi(
        {'biorxiv_doi': '10.1101/2023.12.01.569634v2'}) == '10.1101/2023.12.01.569634'
    assert biorxiv.resolve_biorxiv_doi(
        {'paper_id': 'doi:10.64898/2026.08.07.742070'}) == '10.64898/2026.08.07.742070'
    assert biorxiv.resolve_biorxiv_doi(
        {'doi': '10.1101/2023.01.02.522001'}) == '10.1101/2023.01.02.522001'
    assert biorxiv.resolve_biorxiv_doi(
        {'pdf_url': 'https://www.biorxiv.org/content/10.1101/2023.12.01.569634v1.full.pdf'}
    ) == '10.1101/2023.12.01.569634'
    # A published DOI names the journal version, which bioRxiv does not index.
    assert biorxiv.resolve_biorxiv_doi({'doi': '10.7554/elife.94191.3'}) == ''
    # A PDF hosted elsewhere is not a bioRxiv location even if a DOI matches.
    assert biorxiv.resolve_biorxiv_doi(
        {'pdf_url': 'https://example.org/10.1101/2023.12.01.569634.pdf'}) == ''
    # A medRxiv row is not reachable here whichever column carries its DOI.
    assert biorxiv.resolve_biorxiv_doi(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596',
         'doi': '10.1101/2024.03.01.24303596'}) == ''
    assert biorxiv.resolve_biorxiv_doi({}) == ''


def test_full_text_flattens_the_jats_document_biorxiv_publishes() -> None:
    """Return the article title and prose, skipping tables and references."""
    jats = ('<article><front><article-meta><title-group>'
            '<article-title>Visual working memories</article-title>'
            '</title-group></article-meta></front><body>'
            '<sec><title>Results</title><p>Decoding tracked the aperture edges.</p>'
            '<table-wrap><p>Table residue</p></table-wrap></sec>'
            '<sec><title>Discussion</title><p>Memories are abstractions.</p></sec>'
            '</body><back><ref-list><ref><p>A citation</p></ref></ref-list></back></article>')
    session = FakeSession([FakeResponse(text=jats)])
    text = biorxiv.full_text({'jatsxml': 'https://www.biorxiv.org/x.source.xml'}, session=session)

    assert text.startswith('Visual working memories')
    assert 'Decoding tracked the aperture edges.' in text
    assert 'Memories are abstractions.' in text
    assert 'Table residue' not in text
    assert 'A citation' not in text


def test_full_text_reports_a_broken_document_and_skips_an_absent_one() -> None:
    """Raise on unparseable JATS but treat a bodyless record as no text."""
    session = FakeSession([])
    assert biorxiv.full_text({}, session=session) == ''
    assert session.calls == []

    session = FakeSession([FakeResponse(text='<article')])
    with pytest.raises(RuntimeError, match='bioRxiv returned malformed JATS XML'):
        biorxiv.full_text({'jatsxml': 'https://www.biorxiv.org/x.source.xml'}, session=session)

    session = FakeSession([FakeResponse(text='<article><front/></article>')])
    assert biorxiv.full_text({'jatsxml': 'https://www.biorxiv.org/x.source.xml'},
                             session=session) == ''

    session = FakeSession([FakeResponse(text='   ')])
    assert biorxiv.full_text({'jatsxml': 'https://www.biorxiv.org/x.source.xml'},
                             session=session) == ''


def test_parse_query_lifts_the_scope_terms_out_of_the_phrase() -> None:
    """Separate the walk's bounds from the words that have to match."""
    terms, scope = biorxiv.parse_query(
        '"gene regulation" enhancer category:"Developmental Biology" '
        'from:2024-01-01 to:2024-06-30')

    assert terms == ['gene regulation', 'enhancer']
    assert scope == {'category': 'Developmental Biology', 'from': '2024-01-01',
                     'to': '2024-06-30'}
    assert biorxiv.parse_query('CATEGORY:genomics') == ([], {'category': 'genomics'})
    assert biorxiv.parse_query('   ') == ([], {})


def test_parse_query_rejects_a_bound_that_is_not_an_iso_date() -> None:
    """Refuse a date the interval endpoint would reject anyway."""
    with pytest.raises(ValueError, match='from: must be an ISO date'):
        biorxiv.parse_query('crispr from:last-year')
    with pytest.raises(ValueError, match='to: must be an ISO date'):
        biorxiv.parse_query('crispr to:2024')


def test_matches_combines_terms_with_and_across_the_record_text() -> None:
    """Require every term, and read the whole record rather than the title."""
    record = parsed_records()[0]

    assert biorxiv.matches(record, ['memories'])
    assert biorxiv.matches(record, ['memories', 'percepts'])
    # The abstract, the authors, and the category are searched alongside it.
    assert biorxiv.matches(record, ['gratings'])
    assert biorxiv.matches(record, ['Curtis'])
    assert biorxiv.matches(record, ['neuroscience'])
    assert not biorxiv.matches(record, ['memories', 'lithium'])
    # An empty query matches everything, which is what a scope-only walk wants.
    assert biorxiv.matches(record, [])


def test_matches_reads_a_term_as_a_word_prefix_rather_than_a_substring() -> None:
    """Find plurals and hyphenated forms without matching mid-word noise."""
    record = biorxiv.record_to_paper({
        'doi': '10.1101/2024.01.01.570001',
        'title': 'Genomes and crispr-cas9 outcomes',
        'abstract': 'Reported outcomes.',
        'date': '2024-01-02',
        'version': '1',
    })

    assert biorxiv.matches(record, ['genome'])
    assert biorxiv.matches(record, ['crispr'])
    assert biorxiv.matches(record, ['crispr-cas9'])
    # 'come' appears inside 'outcomes' but never starts a word there.
    assert not biorxiv.matches(record, ['come'])


@pytest.mark.network
def test_biorxiv_returns_a_known_record_from_the_live_api() -> None:
    """Fetch a stable bioRxiv record from the live API service."""
    record = biorxiv.fetch_doi('10.1101/2023.12.01.569634')
    assert record is not None
    assert record['biorxiv_doi'] == '10.1101/2023.12.01.569634'
    assert record['title'] == 'Visual working memories are abstractions of percepts'
    assert record['published_doi'] == '10.7554/elife.94191.3'


@pytest.mark.network
def test_biorxiv_interval_walk_reaches_the_live_service() -> None:
    """Page one live date interval and confirm the walk covers every record."""
    first = biorxiv.interval_page('2024-03-01', '2024-03-02')
    total = biorxiv.total_results(first)
    step = biorxiv.page_size(first)
    assert total > 0

    seen = []
    for cursor in biorxiv.page_cursors(total, step):
        payload = first if cursor == 0 else biorxiv.interval_page('2024-03-01', '2024-03-02',
                                                                  cursor=cursor)
        seen.extend(biorxiv.parse_records(payload))
    assert len(seen) == total
    assert all(record['biorxiv_doi'] for record in seen)


@pytest.mark.network
def test_biorxiv_category_filter_is_applied_by_the_live_service() -> None:
    """Confirm the filtering host actually narrows the interval it returns."""
    scoped = biorxiv.interval_page('2024-03-01', '2024-03-02', category='neuroscience')
    records = biorxiv.parse_records(scoped)
    assert records
    assert {record['category'] for record in records} == {'neuroscience'}


@pytest.mark.network
def test_the_live_archive_issues_the_doi_shape_the_module_recognizes() -> None:
    """Confirm live bioRxiv DOIs route here and never to medRxiv."""
    records = biorxiv.parse_records(biorxiv.interval_page('2026-08-10', '2026-08-11'))
    assert records

    for record in records:
        assert record['biorxiv_doi']
        assert medrxiv.normalize_medrxiv_doi(record['biorxiv_doi']) == ''
