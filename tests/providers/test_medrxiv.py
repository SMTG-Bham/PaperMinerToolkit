"""Unit tests for medRxiv API request helpers and medRxiv record mapping."""

from __future__ import annotations

import json
from typing import Any

import pytest

import paperminertoolkit.providers.medrxiv as medrxiv

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
    assert parsed_records()[3]['published_doi'] == ''


def test_record_to_paper_flips_author_names_into_the_corpus_order() -> None:
    """Rewrite ``Family, G.`` as ``G. Family`` to match the other providers."""
    assert parsed_records()[0]['authors'] == 'A. K. Wheatley; J. A. Juno; S. J. Kent'


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
