"""Unit tests for chemRxiv API request helpers and chemRxiv record mapping."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.request import Request, urlopen

import pytest

import paperminertoolkit.providers.biorxiv as biorxiv
import paperminertoolkit.providers.chemrxiv as chemrxiv
from paperminertoolkit.providers import base as provider
import paperminertoolkit.providers.medrxiv as medrxiv

from tests.doubles import FakeResponse, FakeSession


def items() -> list[dict[str, Any]]:
    """Build the chemRxiv API records the mapping tests read.

    The set covers a preprint that has since been published, two postings of
    one unpublished preprint, a record filed under several categories, and a
    record holding almost nothing, so the mapping is exercised against each
    shape the archive actually returns.
    """
    return [
        {
            'id': '6341a223ea6a22bb990ecdc8',
            'doi': '10.26434/chemrxiv-2022-w08rh',
            'title': 'Online Protein Unfolding Characterized by Ion Mobility',
            'abstract': '  An unfolding\n  abstract.  ',
            'publishedDate': '2022-10-11T00:00:00.000Z',
            'version': '1',
            'authors': [
                {'firstName': 'Rebecca', 'lastName': 'Cain'},
                {'firstName': 'Ada', 'lastName': 'Lovelace'},
            ],
            'categories': [{'id': 'cat-analytical', 'name': 'Analytical Chemistry'}],
            'keywords': ['mass spectrometry', 'protein folding'],
            'license': {'name': 'CC BY 4.0'},
            'vor': {'vorDoi': '10.1007/s00216-022-04501-w'},
            'asset': {'original': {'url': 'https://chemrxiv.org/engage/assets/old.pdf'}},
        },
        {
            'id': 'item-two-v1',
            'doi': '10.26434/chemrxiv.15007737/v1',
            'title': 'A catalysis preprint',
            'abstract': 'First posting.',
            'publishedDate': '2026-08-01T00:00:00.000Z',
            'version': '1',
            'authors': [{'firstName': 'Grace', 'lastName': 'Hopper'}],
            'categories': [{'id': 'cat-catalysis', 'name': 'Catalysis'}],
            'license': {'name': 'CC BY-NC 4.0'},
            'vor': None,
        },
        {
            'id': 'item-two-v2',
            'doi': '10.26434/chemrxiv.15007737/v2',
            'title': 'A catalysis preprint, revised',
            'abstract': 'Second posting.',
            'publishedDate': '2026-08-21T00:00:00.000Z',
            'version': '2',
            'authors': [{'firstName': 'Grace', 'lastName': 'Hopper'}],
            'categories': [{'id': 'cat-catalysis', 'name': 'Catalysis'}],
            'license': {'name': 'CC BY-NC 4.0'},
        },
        {
            'id': 'item-three',
            'doi': '10.26434/chemrxiv-2024-bxxhh-v4',
            'title': 'A preprint filed under several categories',
            'abstract': 'Multi-category posting.',
            'submittedDate': '2024-05-02T13:45:00.000Z',
            'version': '4',
            'authors': [{'firstName': 'Dorothy', 'lastName': 'Hodgkin'}],
            'categories': [
                {'id': 'cat-organic', 'name': 'Organic Chemistry'},
                {'id': 'cat-theory', 'name': 'Theoretical and Computational Chemistry'},
            ],
            'keywords': ['crystallography'],
        },
        {'id': 'item-bare', 'doi': '10.26434/chemrxiv.8011268.v1'},
    ]


def search_payload(total: int = 5, records: Iterable[Mapping[str, Any]] | None = None) -> str:
    """Serialize a chemRxiv search response around the prepared records."""
    hits = [{'item': item} for item in (items() if records is None else records)]
    return json.dumps({'totalCount': total, 'itemHits': hits})


def empty_payload() -> str:
    """Serialize a chemRxiv search response holding no records."""
    return json.dumps({'totalCount': 0, 'itemHits': []})


def parsed_records() -> list[dict[str, Any]]:
    """Map the prepared chemRxiv records onto the paper schema."""
    return chemrxiv.parse_records(json.loads(search_payload()))


def test_normalize_chemrxiv_doi_preserves_the_version_the_registry_issued() -> None:
    """Keep the version suffix, because it is part of the registered DOI.

    Unlike bioRxiv, whose ``v2`` is a URL suffix outside the DOI, chemRxiv
    registers the version. Checked against the registry:
    ``10.26434/chemrxiv.15007737/v1`` resolves while
    ``10.26434/chemrxiv.15007737`` is unregistered, and
    ``10.26434/chemrxiv.8011268.v1`` resolves while the bare form only
    redirects. Normalizing the version away would strand most of the archive,
    so this test exists to stop that being reintroduced for consistency with
    :mod:`paperminertoolkit.providers.biorxiv`.
    """
    for doi in ['10.26434/chemrxiv.15007737/v1', '10.26434/chemrxiv.8011268.v1',
                '10.26434/chemrxiv-2025-0dxhw/v4']:
        assert chemrxiv.normalize_chemrxiv_doi(doi) == doi


def test_normalize_chemrxiv_doi_accepts_the_hyphen_version_engage_issued() -> None:
    """Read a hyphen-separated version without folding it into the accession.

    This is the shape a greedy accession pattern silently swallows, and it is
    the second most common versioned form in the live archive.
    """
    assert chemrxiv.normalize_chemrxiv_doi('10.26434/chemrxiv-2024-bxxhh-v4') == \
        '10.26434/chemrxiv-2024-bxxhh-v4'
    assert chemrxiv.chemrxiv_stem('10.26434/chemrxiv-2024-bxxhh-v4') == \
        '10.26434/chemrxiv-2024-bxxhh'
    assert chemrxiv.chemrxiv_version('10.26434/chemrxiv-2024-bxxhh-v4') == '4'


def test_normalize_chemrxiv_doi_accepts_every_shape_url_and_label() -> None:
    """Accept all five DOI shapes however they are presented."""
    assert chemrxiv.normalize_chemrxiv_doi('10.26434/chemrxiv-2022-w08rh') == \
        '10.26434/chemrxiv-2022-w08rh'
    assert chemrxiv.normalize_chemrxiv_doi('doi:10.26434/chemrxiv-2022-w08rh') == \
        '10.26434/chemrxiv-2022-w08rh'
    assert chemrxiv.normalize_chemrxiv_doi('chemrxiv:10.26434/chemrxiv.15007737/v1') == \
        '10.26434/chemrxiv.15007737/v1'
    assert chemrxiv.normalize_chemrxiv_doi('https://doi.org/10.26434/chemrxiv-2024-bxxhh-v4') == \
        '10.26434/chemrxiv-2024-bxxhh-v4'
    assert chemrxiv.normalize_chemrxiv_doi(
        'https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.8011268.v1') == \
        '10.26434/chemrxiv.8011268.v1'
    assert chemrxiv.normalize_chemrxiv_doi('  10.26434/CHEMRXIV-2022-W08RH  ') == \
        '10.26434/chemrxiv-2022-w08rh'
    assert chemrxiv.normalize_chemrxiv_doi(None) == ''
    assert chemrxiv.normalize_chemrxiv_doi('') == ''


def test_chemrxiv_stem_is_a_grouping_key_rather_than_a_resolvable_doi() -> None:
    """Strip the version for grouping, knowing the result may not resolve.

    ``10.26434/chemrxiv.15007737`` is unregistered, so the stem is only ever a
    key for recognizing two postings as one preprint.
    """
    assert chemrxiv.chemrxiv_stem('10.26434/chemrxiv.15007737/v1') == '10.26434/chemrxiv.15007737'
    assert chemrxiv.chemrxiv_stem('10.26434/chemrxiv.15007737/v2') == '10.26434/chemrxiv.15007737'
    assert chemrxiv.chemrxiv_stem('10.26434/chemrxiv-2022-w08rh') == '10.26434/chemrxiv-2022-w08rh'
    assert chemrxiv.chemrxiv_stem('not a doi') == ''


def test_chemrxiv_version_reads_every_separator_the_archive_has_used() -> None:
    """Read the version from the dotted, hyphenated, and slashed forms."""
    assert chemrxiv.chemrxiv_version('10.26434/chemrxiv.8011268.v1') == '1'
    assert chemrxiv.chemrxiv_version('10.26434/chemrxiv-2024-bxxhh-v4') == '4'
    assert chemrxiv.chemrxiv_version('10.26434/chemrxiv-2025-rq1vl/v10') == '10'
    assert chemrxiv.chemrxiv_version('10.26434/chemrxiv-2022-w08rh') == ''


def test_normalize_chemrxiv_doi_rejects_the_publishers_other_dois() -> None:
    """Reject a DOI that carries no chemRxiv accession token."""
    for value in ['10.1021/jacs.1c00001', '10.26434/something-else-2024', '10.26434/',
                  'https://chemrxiv.org/doi/full/']:
        assert chemrxiv.normalize_chemrxiv_doi(value) == ''


def test_the_three_preprint_servers_do_not_claim_each_others_dois() -> None:
    """Confirm chemRxiv, bioRxiv, and medRxiv identifiers stay disjoint.

    chemRxiv is recognized by the ``chemrxiv`` token rather than by its
    ``10.26434`` prefix, which is what keeps this true without any defensive
    change to the other two modules.
    """
    for value in ['10.26434/chemrxiv-2022-w08rh', '10.26434/chemrxiv.15007737/v1',
                  '10.26434/chemrxiv-2024-bxxhh-v4', '10.26434/chemrxiv.8011268.v1']:
        assert chemrxiv.normalize_chemrxiv_doi(value)
        assert biorxiv.normalize_biorxiv_doi(value) == ''
        assert medrxiv.normalize_medrxiv_doi(value) == ''

    for value in ['10.1101/2023.12.01.569634', '10.1101/060400',
                  '10.1101/2024.03.01.24303596', '10.64898/2026.08.05.26359794']:
        assert chemrxiv.normalize_chemrxiv_doi(value) == ''


def test_resolve_chemrxiv_doi_reads_stored_values_without_a_request() -> None:
    """Recover the identifier from whichever column the row happens to hold."""
    assert chemrxiv.resolve_chemrxiv_doi(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'}) == '10.26434/chemrxiv.15007737/v1'
    assert chemrxiv.resolve_chemrxiv_doi(
        {'paper_id': 'doi:10.26434/chemrxiv-2022-w08rh'}) == '10.26434/chemrxiv-2022-w08rh'
    assert chemrxiv.resolve_chemrxiv_doi(
        {'doi': '10.26434/chemrxiv-2024-bxxhh-v4'}) == '10.26434/chemrxiv-2024-bxxhh-v4'
    assert chemrxiv.resolve_chemrxiv_doi(
        {'pdf_url': 'https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.8011268.v1'}) == \
        '10.26434/chemrxiv.8011268.v1'
    # A published DOI names the journal version, which chemRxiv cannot answer for.
    assert chemrxiv.resolve_chemrxiv_doi({'doi': '10.1007/s00216-022-04501-w'}) == ''
    assert chemrxiv.resolve_chemrxiv_doi({}) == ''


def test_pdf_url_is_derived_from_the_doi_rather_than_the_asset_block() -> None:
    """Build the PDF location from the DOI the registry currently records."""
    assert chemrxiv.pdf_url('10.26434/chemrxiv.15007737/v1') == \
        'https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15007737/v1'
    assert chemrxiv.landing_url('10.26434/chemrxiv-2022-w08rh') == \
        'https://chemrxiv.org/doi/full/10.26434/chemrxiv-2022-w08rh'
    assert chemrxiv.pdf_url('') == ''
    assert chemrxiv.landing_url('nonsense') == ''


def test_record_to_paper_prefers_the_published_doi_for_the_paper_id() -> None:
    """Key a published preprint by the journal DOI so the rows merge."""
    record = chemrxiv.record_to_paper(items()[0])

    assert record['paper_id'] == 'doi:10.1007/s00216-022-04501-w'
    assert record['doi'] == '10.1007/s00216-022-04501-w'
    assert record['published_doi'] == '10.1007/s00216-022-04501-w'
    assert record['chemrxiv_doi'] == '10.26434/chemrxiv-2022-w08rh'


def test_record_to_paper_leaves_the_journal_empty_for_a_published_preprint() -> None:
    """Withhold the archive name so enrichment can fill the real journal."""
    published = chemrxiv.record_to_paper(items()[0])
    unpublished = chemrxiv.record_to_paper(items()[1])

    assert published['journal'] == ''
    assert unpublished['journal'] == 'chemRxiv'
    assert unpublished['paper_id'] == 'doi:10.26434/chemrxiv.15007737/v1'


def test_record_to_paper_joins_author_names_in_the_corpus_order() -> None:
    """Combine the split name fields into ``Given Family`` order."""
    record = chemrxiv.record_to_paper(items()[0])

    assert record['authors'] == 'Rebecca Cain; Ada Lovelace'


def test_record_to_paper_reduces_a_timestamp_to_a_calendar_date() -> None:
    """Store the posting date without the time of day."""
    assert chemrxiv.record_to_paper(items()[0])['publication_date'] == '2022-10-11'
    # The submitted date stands in when no published date is given.
    assert chemrxiv.record_to_paper(items()[3])['publication_date'] == '2024-05-02'


def test_record_to_paper_builds_the_pdf_location_from_the_doi() -> None:
    """Prefer the DOI-derived location over the record's stale asset URL."""
    record = chemrxiv.record_to_paper(items()[0])

    assert record['pdf_url'] == 'https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2022-w08rh'
    assert record['asset_url'] == 'https://chemrxiv.org/engage/assets/old.pdf'


def test_categories_carry_every_subject_chemrxiv_files_a_preprint_under() -> None:
    """Keep all categories, flagging only the first as primary."""
    record = chemrxiv.record_to_paper(items()[3])

    assert [category['name'] for category in record['categories']] == [
        'Organic Chemistry', 'Theoretical and Computational Chemistry']
    assert [category['is_primary'] for category in record['categories']] == [True, False]
    assert record['category'] == 'Organic Chemistry'
    assert record['keywords'] == ['crystallography']


def test_record_to_paper_survives_a_record_missing_almost_every_field() -> None:
    """Invent nothing for a record that carries only an identifier."""
    record = chemrxiv.record_to_paper(items()[4])

    assert record['chemrxiv_doi'] == '10.26434/chemrxiv.8011268.v1'
    assert record['title'] == ''
    assert record['authors'] == ''
    assert record['categories'] == []
    assert record['published_doi'] == ''
    assert record['journal'] == 'chemRxiv'
    # The version falls back to the one carried inside the DOI.
    assert record['version'] == '1'


def test_parse_records_reads_both_the_search_and_single_item_shapes() -> None:
    """Map records whether they arrive wrapped in hits or on their own."""
    assert len(parsed_records()) == 5
    assert chemrxiv.parse_records(None) == []
    assert chemrxiv.parse_records({'totalCount': 0, 'itemHits': []}) == []

    single = chemrxiv.parse_records({'item': items()[0]})
    assert len(single) == 1 and single[0]['chemrxiv_doi'] == '10.26434/chemrxiv-2022-w08rh'
    bare = chemrxiv.parse_records(items()[4])
    assert len(bare) == 1 and bare[0]['chemrxiv_doi'] == '10.26434/chemrxiv.8011268.v1'


def test_total_results_reads_the_reported_match_count() -> None:
    """Read the total the search payload reports."""
    assert chemrxiv.total_results(json.loads(search_payload(total=417))) == 417
    assert chemrxiv.total_results(json.loads(empty_payload())) == 0
    assert chemrxiv.total_results(None) == 0


def test_latest_versions_groups_on_the_stem_and_keeps_the_first_date() -> None:
    """Collapse two postings of one preprint, keeping the earliest date.

    Grouping is on the stem rather than the DOI, because each posted version of
    a chemRxiv preprint carries a DOI of its own.
    """
    collapsed = chemrxiv.latest_versions(parsed_records())

    assert len(collapsed) == 4
    revised = next(entry for entry in collapsed
                   if entry['chemrxiv_stem'] == '10.26434/chemrxiv.15007737')
    assert revised['version'] == '2'
    assert revised['title'] == 'A catalysis preprint, revised'
    assert revised['publication_date'] == '2026-08-01'


def test_latest_versions_collapses_the_same_whichever_order_versions_arrive() -> None:
    """Reach the same record from a newest-first and an oldest-first page."""
    forward = chemrxiv.latest_versions(parsed_records())
    backward = chemrxiv.latest_versions(list(reversed(parsed_records())))

    def keyed(entries: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
        """Index collapsed entries by stem for an order-independent compare."""
        return {entry['chemrxiv_stem']: (entry['version'], entry['publication_date'])
                for entry in entries}

    assert keyed(forward) == keyed(backward)


def test_latest_versions_does_not_let_a_sparse_revision_blank_a_field() -> None:
    """Overwrite only with values the newer posting actually carries."""
    first, second = parsed_records()[1], dict(parsed_records()[2])
    second['abstract'] = ''
    collapsed = chemrxiv.latest_versions([first, second])

    assert len(collapsed) == 1
    assert collapsed[0]['version'] == '2'
    assert collapsed[0]['abstract'] == 'First posting.'


def test_request_paces_consecutive_calls_with_the_courtesy_delay() -> None:
    """Space consecutive requests through the shared window."""
    slept: list[float] = []
    session = FakeSession([FakeResponse(search_payload()), FakeResponse(search_payload())])
    original = provider.time.sleep
    try:
        provider.time.sleep = slept.append
        chemrxiv.request(chemrxiv.search_url(), session=session)
        chemrxiv.request(chemrxiv.search_url(), session=session)
    finally:
        provider.time.sleep = original

    assert len(slept) == 1
    assert 0 < slept[0] <= chemrxiv.CHEMRXIV_MIN_INTERVAL
    assert session.calls[0]['headers'] == {'User-Agent': provider.USER_AGENT}


def test_request_returns_none_for_a_missing_document() -> None:
    """Read a 404 as an absent record rather than as a failure."""
    session = FakeSession([FakeResponse('', status_code=404)])

    assert chemrxiv.request(chemrxiv.doi_url('10.26434/chemrxiv-2022-w08rh'),
                            session=session) is None


def test_request_reports_the_bot_challenge_and_does_not_retry_it() -> None:
    """Fail a 403 once, naming the challenge and an alternative source.

    A refusal is not retried, because repeating a refused request only spends
    more of them, and it is named explicitly so it cannot be mistaken for a bad
    query. PaperMinerToolkit does not attempt to defeat the challenge.
    """
    session = FakeSession([FakeResponse('<!DOCTYPE html>', status_code=403)])

    with pytest.raises(RuntimeError, match='bot challenge'):
        chemrxiv.request(chemrxiv.search_url(), session=session)
    assert len(session.calls) == 1


def test_request_retries_a_rate_limited_response_and_honours_retry_after() -> None:
    """Retry a 429 after the delay the response asks for."""
    slept: list[float] = []
    session = FakeSession([
        FakeResponse('', status_code=429, headers={'Retry-After': '7'}),
        FakeResponse(search_payload()),
    ])
    original = provider.time.sleep
    try:
        provider.time.sleep = slept.append
        response = chemrxiv.request(chemrxiv.search_url(), session=session)
    finally:
        provider.time.sleep = original

    assert response is not None
    assert 7 in slept
    assert len(session.calls) == 2


def test_request_fails_immediately_on_a_client_error_other_than_a_rate_limit() -> None:
    """Treat a 400 as terminal rather than spending further attempts."""
    session = FakeSession([FakeResponse('', status_code=400)])

    with pytest.raises(RuntimeError, match='rejected the request with 400'):
        chemrxiv.request(chemrxiv.search_url(), session=session)
    assert len(session.calls) == 1


def test_request_raises_after_exhausting_every_attempt() -> None:
    """Report a run of failed attempts once the budget is gone."""
    session = FakeSession([FakeResponse('', status_code=500) for _ in range(4)])

    with pytest.raises(RuntimeError, match='failed after 4 attempts'):
        chemrxiv.request(chemrxiv.search_url(), session=session)
    assert len(session.calls) == 4


def test_request_json_reports_an_html_challenge_page_rather_than_bad_json() -> None:
    """Name the challenge when a 200 carries an HTML body.

    The challenge can answer an otherwise successful request with its own page,
    which is a different cause from a truncated body and is worth telling
    apart.
    """
    session = FakeSession([FakeResponse('<!DOCTYPE html><html>Just a moment...</html>')])

    with pytest.raises(RuntimeError, match='HTML challenge page'):
        chemrxiv.request_json(chemrxiv.search_url(), session=session)


def test_request_json_reports_malformed_and_unexpected_bodies() -> None:
    """Separate a broken body and a wrongly-typed payload from a challenge."""
    session = FakeSession([FakeResponse('{"totalCount": ')])
    with pytest.raises(RuntimeError, match='malformed JSON'):
        chemrxiv.request_json(chemrxiv.search_url(), session=session)

    session = FakeSession([FakeResponse('[1, 2, 3]')])
    with pytest.raises(RuntimeError, match='unexpected payload of type list'):
        chemrxiv.request_json(chemrxiv.search_url(), session=session)


def test_search_page_sends_the_scope_the_caller_asked_for() -> None:
    """Forward the term, window, and filters as query parameters."""
    session = FakeSession([FakeResponse(search_payload())])
    chemrxiv.search_page(term='catalysis', skip=50, limit=25, category_id='cat-catalysis',
                         date_from='2024-01-01', date_to='2024-12-31', session=session)

    call = session.calls[0]
    assert call['url'] == f'{chemrxiv.BASE_URL}/items'
    assert call['params'] == {'skip': 50, 'limit': 25, 'sort': chemrxiv.DEFAULT_SORT,
                              'term': 'catalysis', 'categoryIds': 'cat-catalysis',
                              'searchDateFrom': '2024-01-01', 'searchDateTo': '2024-12-31'}


def test_search_page_omits_the_filters_a_query_does_not_name() -> None:
    """Leave an unset filter out rather than sending an empty one."""
    session = FakeSession([FakeResponse(search_payload())])
    chemrxiv.search_page(session=session)

    assert session.calls[0]['params'] == {'skip': 0, 'limit': chemrxiv.PAGE_SIZE,
                                          'sort': chemrxiv.DEFAULT_SORT}


def test_category_ids_resolves_names_and_rejects_an_unknown_one() -> None:
    """Map a category name to its identifier, raising on one that is unknown.

    An unmatched name raises rather than being dropped, because silently
    ignoring the filter returns the unfiltered archive.
    """
    listing = json.dumps([{'id': 'cat-catalysis', 'name': 'Catalysis'},
                          {'id': 'cat-organic', 'name': 'Organic Chemistry'}])
    session = FakeSession([FakeResponse(listing)])

    assert chemrxiv.category_ids(['catalysis'], session=session) == ['cat-catalysis']
    # The listing is cached, so a second lookup issues no further request.
    assert chemrxiv.category_ids(['Organic Chemistry'], session=session) == ['cat-organic']
    assert len(session.calls) == 1

    with pytest.raises(ValueError, match='not a chemRxiv category'):
        chemrxiv.category_ids(['Astrology'], session=session)
    assert chemrxiv.category_ids([], session=session) == []


def test_fetch_doi_returns_the_newest_posting_of_one_preprint() -> None:
    """Collapse the returned postings and answer with the newest."""
    records = [items()[1], items()[2]]
    session = FakeSession([FakeResponse(search_payload(total=2, records=records))])
    entry = chemrxiv.fetch_doi('10.26434/chemrxiv.15007737/v1', session=session)

    assert entry is not None
    assert entry['version'] == '2'
    assert session.calls[0]['url'].endswith('/items/doi/10.26434/chemrxiv.15007737/v1')


def test_fetch_doi_falls_back_to_the_search_endpoint_on_a_missing_route() -> None:
    """Search for the DOI when the lookup route reports nothing.

    The fallback matches on the stem, so it keeps working if the DOI route is
    withdrawn, and it answers a request for one version with the current one.
    """
    records = [items()[2]]
    session = FakeSession([FakeResponse('', status_code=404),
                           FakeResponse(search_payload(total=1, records=records))])
    entry = chemrxiv.fetch_doi('10.26434/chemrxiv.15007737/v1', session=session)

    assert entry is not None
    assert entry['version'] == '2'
    assert session.calls[1]['params']['term'] == '10.26434/chemrxiv.15007737/v1'


def test_fetch_doi_skips_a_value_holding_no_chemrxiv_identifier() -> None:
    """Spend no request on a row chemRxiv could not answer for."""
    session = FakeSession([])

    assert chemrxiv.fetch_doi('10.1101/2023.12.01.569634', session=session) is None
    assert session.calls == []


def test_parse_query_lifts_the_scope_terms_out_of_the_phrase() -> None:
    """Separate match terms from the scope the server will apply."""
    terms, scope = chemrxiv.parse_query(
        'photocatalysis "singlet oxygen" category:Catalysis from:2024-01-01 to:2024-12-31')

    assert terms == ['photocatalysis', 'singlet oxygen']
    assert scope == {'category': 'Catalysis', 'from': '2024-01-01', 'to': '2024-12-31'}
    assert chemrxiv.search_terms(terms) == 'photocatalysis "singlet oxygen"'


def test_parse_query_rejects_a_bound_that_is_not_an_iso_date() -> None:
    """Refuse a scope date the API could not use."""
    with pytest.raises(ValueError, match='must be an ISO date'):
        chemrxiv.parse_query('catalysis from:last-tuesday')


def test_the_module_publishes_no_archive_walk_or_full_text_surface() -> None:
    """Confirm chemRxiv answers by search rather than by an archive walk.

    The absent names are the ones :func:`paperminertoolkit.workflows.search._rxiv_search`
    requires. chemRxiv has a search endpoint and no machine-readable full text,
    so implementing them would mean reading the archive to answer a query the
    server already answers, and promising a text source that does not exist.
    """
    for name in ['matches', 'page_cursors', 'interval_page', 'endpoint', 'page_size',
                 'CORPUS_START', 'MAX_SCAN_RECORDS', 'full_text']:
        assert not hasattr(chemrxiv, name)


def test_chemrxiv_mapping_helpers_accept_sparse_and_mixed_api_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize the permissive shapes returned by several chemRxiv API versions."""
    assert chemrxiv.request_headers()['User-Agent']
    assert chemrxiv.item_url('item') == f'{chemrxiv.BASE_URL}/items/item'
    assert chemrxiv.chemrxiv_stem(None) == ''
    assert chemrxiv.chemrxiv_version(None) == ''
    assert chemrxiv._authors(' Ada  Lovelace ') == 'Ada Lovelace'
    assert chemrxiv._authors(['Grace Hopper', {'firstName': 'Dorothy', 'lastName': 'Hodgkin'}, 1]) == (
        'Grace Hopper; Dorothy Hodgkin'
    )
    assert chemrxiv._categories({'categories': 'Catalysis'}) == [
        {'id': 'catalysis', 'name': 'Catalysis', 'is_primary': True}
    ]
    assert chemrxiv._categories({'categories': ['', {'name': 'Theory'}]}) == [
        {'id': 'theory', 'name': 'Theory', 'is_primary': True}
    ]
    assert chemrxiv._keywords({'keywords': ' catalysis '}) == ['catalysis']
    assert chemrxiv._published_doi({'vor': {}}) == ''
    assert chemrxiv._asset_url({'asset': {'url': 'https://example.org/paper.pdf'}}).endswith('paper.pdf')

    assert chemrxiv.parse_records({'itemHits': 'invalid'}) == []
    assert len(chemrxiv.parse_records({'itemHits': [None, {'item': items()[0]}]})) == 1
    assert chemrxiv.latest_versions([{}]) == []

    monkeypatch.setattr(
        chemrxiv,
        'request_payload',
        lambda *args, **kwargs: {'data': [None, {'id': 'c1', 'name': 'Catalysis'}]},
    )
    chemrxiv.reset_categories_cache()
    assert chemrxiv.categories() == [{'id': 'c1', 'name': 'Catalysis'}]


def test_fetch_item_handles_blank_missing_and_present_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid blank item requests and return the first normalized API record."""
    assert chemrxiv.fetch_item(' ') is None
    monkeypatch.setattr(chemrxiv, 'request_json', lambda *args, **kwargs: {'itemHits': []})
    assert chemrxiv.fetch_item('missing') is None
    monkeypatch.setattr(chemrxiv, 'request_json', lambda *args, **kwargs: {'item': items()[0]})
    assert chemrxiv.fetch_item('present')['chemrxiv_doi'] == '10.26434/chemrxiv-2022-w08rh'


@pytest.fixture
def live_chemrxiv() -> None:
    """Skip a live test when the chemRxiv API cannot be reached.

    chemrxiv.org sits behind a bot challenge that refuses some clients
    outright, and the archive has since moved hosting platforms, so the API may
    answer, refuse, or no longer exist depending on where the suite runs.
    Skipping says which of those happened; failing would report only that a
    request did not succeed, and would not distinguish an unreachable service
    from a mapping that has drifted.
    """
    try:
        payload = chemrxiv.search_page(term='water', limit=1)
    except RuntimeError as error:
        pytest.skip(f'chemRxiv API unreachable: {error}')
    if not chemrxiv.parse_records(payload):
        pytest.skip('chemRxiv API returned no records')


@pytest.mark.network
@pytest.mark.usefixtures('live_chemrxiv')
def test_chemrxiv_returns_a_known_record_from_the_live_api() -> None:
    """Fetch a stable chemRxiv record and check the published DOI it names."""
    record = chemrxiv.fetch_doi('10.26434/chemrxiv-2022-w08rh')
    assert record is not None
    assert record['chemrxiv_doi'] == '10.26434/chemrxiv-2022-w08rh'
    assert record['published_doi'] == '10.1007/s00216-022-04501-w'
    assert record['title']


@pytest.mark.network
@pytest.mark.usefixtures('live_chemrxiv')
def test_the_live_search_endpoint_pages_and_reports_a_total() -> None:
    """Page the live search endpoint and confirm the pages do not overlap."""
    first = chemrxiv.search_page(term='catalysis', skip=0, limit=5)
    assert chemrxiv.total_results(first) > 0

    second = chemrxiv.search_page(term='catalysis', skip=5, limit=5)
    first_dois = {record['chemrxiv_doi'] for record in chemrxiv.parse_records(first)}
    second_dois = {record['chemrxiv_doi'] for record in chemrxiv.parse_records(second)}
    assert first_dois and second_dois
    assert not first_dois & second_dois


@pytest.mark.network
@pytest.mark.usefixtures('live_chemrxiv')
def test_the_live_category_filter_narrows_the_result_set() -> None:
    """Confirm the live service applies the category filter it is sent."""
    listing = chemrxiv.categories()
    assert listing

    name = listing[0]['name']
    scoped = chemrxiv.search_page(skip=0, limit=5,
                                  category_id=chemrxiv.category_ids([name])[0])
    records = chemrxiv.parse_records(scoped)
    assert records
    for record in records:
        assert name in {category['name'] for category in record['categories']}


@pytest.mark.network
@pytest.mark.usefixtures('live_chemrxiv')
def test_the_live_records_carry_every_field_the_mapping_reads() -> None:
    """Catch mapping drift by checking a live record fills the core columns."""
    records = chemrxiv.parse_records(chemrxiv.search_page(term='synthesis', limit=10))
    assert records

    for record in records:
        assert record['chemrxiv_doi']
        assert record['title']
        assert record['publication_date']
    assert any(record['authors'] for record in records)
    assert any(record['categories'] for record in records)


@pytest.mark.network
def test_the_live_registry_still_issues_the_doi_shapes_the_module_reads() -> None:
    """Confirm live chemRxiv DOIs all parse here and never as bio/medRxiv.

    chemrxiv.org can refuse this client outright, but Crossref cannot, so the
    DOI shapes stay checkable even while the archive's own API is unreachable.
    That matters because the shapes are the part most likely to change quietly:
    chemRxiv has issued five of them across three hosting platforms, and the
    version suffix is registered rather than decorative.
    """
    url = ('https://api.crossref.org/works?filter=prefix:10.26434&rows=200&select=DOI'
           '&mailto=paperminertoolkit@example.com')
    with urlopen(Request(url, headers={'User-Agent': provider.USER_AGENT}), timeout=60) as handle:
        dois = [item['DOI'] for item in json.load(handle)['message']['items']]
    assert len(dois) > 100

    for doi in dois:
        assert chemrxiv.normalize_chemrxiv_doi(doi) == doi.lower(), doi
        assert biorxiv.normalize_biorxiv_doi(doi) == ''
        assert medrxiv.normalize_medrxiv_doi(doi) == ''
    assert {chemrxiv.chemrxiv_version(doi) != '' for doi in dois} == {True, False}
