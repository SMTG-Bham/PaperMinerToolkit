"""Unit tests for the shared bioRxiv-family client.

The two archives that use it have their own suites for their own DOI shapes and
record data. What is tested here is the behaviour that does not vary between
them, and the wiring that makes each archive's module address its own service.
"""

from __future__ import annotations

import json
from copy import deepcopy
from types import ModuleType
from typing import Any

import pytest

import paperminertoolkit.providers.biorxiv as biorxiv
import paperminertoolkit.providers.chemrxiv as chemrxiv
import paperminertoolkit.providers.medrxiv as medrxiv
from paperminertoolkit.providers import base as provider, rxiv as _rxiv

from tests.doubles import FakeResponse, FakeSession

ARCHIVES = [medrxiv, biorxiv]
ARCHIVE_IDS = ['medrxiv', 'biorxiv']


@pytest.mark.parametrize('archive', [*ARCHIVES, chemrxiv], ids=[*ARCHIVE_IDS, 'chemrxiv'])
def test_version_collapse_preserves_order_sparse_fields_and_inputs(archive: ModuleType) -> None:
    """Keep first appearance and earlier values when a revision omits them."""
    entries = [
        {'paper_id': 'first', 'version': '1', 'publication_date': '2024-01-02',
         'title': 'Original', 'count': 3, 'flag': True, 'categories': ['Science']},
        {'paper_id': 'second', 'version': '1'},
        {'paper_id': 'first', 'version': '2', 'publication_date': '2024-03-01',
         'title': 'Revised', 'count': 0, 'flag': False, 'categories': []},
        {'title': 'Unidentifiable'},
    ]
    original = deepcopy(entries)

    result = archive.latest_versions(entries)

    assert result == [
        {'paper_id': 'first', 'version': '2', 'publication_date': '2024-01-02',
         'title': 'Revised', 'count': 3, 'flag': True, 'categories': ['Science']},
        {'paper_id': 'second', 'version': '1'},
    ]
    assert entries == original
    assert result[1] is not entries[1]


@pytest.mark.parametrize('archive', [*ARCHIVES, chemrxiv], ids=[*ARCHIVE_IDS, 'chemrxiv'])
@pytest.mark.parametrize(('earlier_version', 'later_version'), [
    ('2', '2'), (None, 'unavailable'), ('NA', 0),
])
def test_version_ties_favor_later_input_without_inventing_dates(
    archive: ModuleType,
    earlier_version: object,
    later_version: object,
) -> None:
    """Use input order to break ties, including missing or nonnumeric versions."""
    result = archive.latest_versions([
        {'paper_id': 'one', 'version': earlier_version, 'title': 'Earlier'},
        {'paper_id': 'one', 'version': later_version, 'title': 'Later'},
    ])

    assert len(result) == 1
    assert result[0]['title'] == 'Later'
    assert result[0]['publication_date'] == ''
    assert result[0]['version'] == (later_version or earlier_version)


def test_message_ignores_non_mapping_entries() -> None:
    """Ignore malformed message-list entries."""
    assert _rxiv._message({'messages': ['not-a-mapping']}) == {}


def payload(status: str = 'ok', total: str = '5', count: int = 5) -> str:
    """Return an interval payload carrying only its message block.

    Parameters
    ----------
    status : str, default='ok'
        Status string the archive reports.
    total : str, default='5'
        Total record count for the interval.
    count : int, default=5
        Records the page actually carries.

    Returns
    -------
    str
        Encoded payload.
    """
    return json.dumps({'messages': [{'status': status, 'total': total, 'count': count}],
                       'collection': []})


@pytest.mark.parametrize('archive', ARCHIVES, ids=ARCHIVE_IDS)
def test_each_archive_addresses_its_own_service(archive: Any) -> None:
    """Point each module's config at its own path segment, host, and start date."""
    config = archive.SERVER_CONFIG
    assert config.name == archive.SERVER
    assert config.web_url == archive.WEB_URL
    assert config.corpus_start == archive.CORPUS_START
    assert config.id_column == f'{archive.SERVER}_doi'
    assert config.limiter is archive.LIMITER
    assert archive.details_url('10.1101/2024.01.01.24300001').startswith(
        f'{_rxiv.BASE_URL}/details/{archive.SERVER}/')
    assert archive.interval_url('2024-01-01', '2024-12-31').startswith(
        f'{_rxiv.BASE_URL}/details/{archive.SERVER}/')


def test_the_two_archives_share_one_implementation_but_not_one_window() -> None:
    """Keep the shared client from pacing both archives against one window."""
    assert medrxiv.SERVER_CONFIG is not biorxiv.SERVER_CONFIG
    assert medrxiv.LIMITER is not biorxiv.LIMITER
    assert medrxiv.SERVER_CONFIG.doi_pattern is not biorxiv.SERVER_CONFIG.doi_pattern
    # Both hosts answer for both archives, so the base URLs are deliberately equal.
    assert medrxiv.BASE_URL == biorxiv.BASE_URL
    assert medrxiv.CATEGORY_BASE_URL == biorxiv.CATEGORY_BASE_URL
    # The filtering and wider-page hosts differ for both archives.
    assert biorxiv.BASE_URL != biorxiv.CATEGORY_BASE_URL


@pytest.mark.parametrize('archive', [_rxiv, *ARCHIVES], ids=['shared', *ARCHIVE_IDS])
def test_endpoint_trades_page_width_for_the_category_filter(archive: ModuleType) -> None:
    """Address the filtering host only when a category is actually wanted."""
    assert archive.endpoint() == (archive.BASE_URL, archive.PAGE_SIZE)
    assert archive.endpoint('  ') == (archive.BASE_URL, archive.PAGE_SIZE)
    assert archive.endpoint('oncology') == (archive.CATEGORY_BASE_URL, archive.CATEGORY_PAGE_SIZE)
    assert archive.endpoint('neuroscience') == (archive.CATEGORY_BASE_URL, archive.CATEGORY_PAGE_SIZE)


@pytest.mark.parametrize('archive', [_rxiv, *ARCHIVES], ids=['shared', *ARCHIVE_IDS])
def test_page_size_reads_the_length_the_host_actually_sent(archive: ModuleType) -> None:
    """Step a walk by the page the host chose rather than by the constant."""
    assert archive.page_size(json.loads(payload(count=30))) == 30
    assert archive.page_size(json.loads(payload(count=0)), default=30) == 30
    assert archive.page_size(None) == archive.PAGE_SIZE
    assert archive.page_size(json.loads(payload(count=5))) == 5
    assert archive.page_size({'messages': [{'status': 'no posts found'}]}, default=30) == 30


@pytest.mark.parametrize('archive', [_rxiv, *ARCHIVES], ids=['shared', *ARCHIVE_IDS])
def test_total_results_reads_the_message_block(archive: ModuleType) -> None:
    """Read the interval's record count, tolerating a payload that has none."""
    assert archive.total_results(json.loads(payload(total='1281'))) == 1281
    assert archive.total_results(json.loads(payload(total='not a number'))) == 0
    assert archive.total_results(None) == 0
    assert archive.total_results({'messages': [{'status': 'no posts found'}]}) == 0


@pytest.mark.parametrize('archive', [_rxiv, *ARCHIVES], ids=['shared', *ARCHIVE_IDS])
def test_page_cursors_walks_pages_from_the_last_one_back_to_zero(archive: ModuleType) -> None:
    """Start at the final page so the newest postings are read first."""
    assert list(archive.page_cursors(1281, 100)) == list(range(1200, -1, -100))
    assert list(archive.page_cursors(250, 30)) == list(range(240, -1, -30))
    # An exact multiple must not produce an empty page past the end.
    assert list(archive.page_cursors(200, 100)) == [100, 0]
    assert list(archive.page_cursors(1, 100)) == [0]
    assert list(archive.page_cursors(0, 100)) == []
    assert list(archive.page_cursors(50, 0)) == list(range(49, -1, -1))


@pytest.mark.parametrize('archive', ARCHIVES, ids=ARCHIVE_IDS)
def test_request_json_reads_an_absent_record_out_of_a_200_response(archive: Any) -> None:
    """Report the statuses that mean nothing found as nothing rather than failure."""
    for status in ['no posts found', 'doi not recognizable', 'DOI not recognizable']:
        session = FakeSession([FakeResponse(text=payload(status=status))])
        assert archive.request_json(_rxiv.BASE_URL, session=session) is None


@pytest.mark.parametrize('archive', ARCHIVES, ids=ARCHIVE_IDS)
@pytest.mark.parametrize('status', ['bad interval', 'Both dates must be in yyyy-mm-dd'])
def test_request_json_raises_on_a_rejection_dressed_as_a_200_response(
    archive: ModuleType, status: str,
) -> None:
    """Name the archive and explain a rejected request in an HTTP 200 body."""
    session = FakeSession([FakeResponse(text=payload(status=status))])
    with pytest.raises(RuntimeError, match=f'{archive.SERVER_CONFIG.label} rejected the request') as error:
        archive.request_json(_rxiv.BASE_URL, session=session)
    assert status in str(error.value)


@pytest.mark.parametrize('archive,query,terms,scope,uppercase_category', [
    pytest.param(
        _rxiv, '"gene therapy" crispr category:Genomics from:2024-01-01 to:2024-12-31',
        ['gene therapy', 'crispr'],
        {'category': 'Genomics', 'from': '2024-01-01', 'to': '2024-12-31'},
        'genomics', id='shared',
    ),
    pytest.param(
        medrxiv, '"vaccine hesitancy" uptake category:"Infectious Diseases" '
        'from:2024-01-01 to:2024-06-30',
        ['vaccine hesitancy', 'uptake'],
        {'category': 'Infectious Diseases', 'from': '2024-01-01', 'to': '2024-06-30'},
        'oncology', id='medrxiv',
    ),
    pytest.param(
        biorxiv, '"gene regulation" enhancer category:"Developmental Biology" '
        'from:2024-01-01 to:2024-06-30',
        ['gene regulation', 'enhancer'],
        {'category': 'Developmental Biology', 'from': '2024-01-01', 'to': '2024-06-30'},
        'genomics', id='biorxiv',
    ),
])
def test_parse_query_lifts_the_scope_terms_out_of_the_phrase(
    archive: ModuleType,
    query: str,
    terms: list[str],
    scope: dict[str, str],
    uppercase_category: str,
) -> None:
    """Separate scope and words while preserving quoted and case-insensitive keys."""
    assert archive.parse_query(query) == (terms, scope)
    assert archive.parse_query(f'CATEGORY:{uppercase_category}') == ([], {'category': uppercase_category})
    assert archive.parse_query('') == ([], {})
    assert archive.parse_query('   ') == ([], {})


@pytest.mark.parametrize('archive,query', [
    pytest.param(_rxiv, 'crispr from:last-week', id='shared'),
    pytest.param(medrxiv, 'covid from:last-year', id='medrxiv'),
    pytest.param(biorxiv, 'crispr from:last-year', id='biorxiv'),
])
def test_parse_query_rejects_a_bound_that_is_not_an_iso_date(archive: ModuleType, query: str) -> None:
    """Refuse a malformed scope here rather than spending a walk on it."""
    with pytest.raises(ValueError, match='from: must be an ISO date'):
        archive.parse_query(query)
    with pytest.raises(ValueError, match='to: must be an ISO date'):
        archive.parse_query(f'{query.split()[0]} to:2024')


def test_matches_combines_terms_with_and_across_the_record_text() -> None:
    """Require every term, searching the fields a reader would expect."""
    entry = {'title': 'Genome editing in maize', 'abstract': 'A CRISPR-Cas9 protocol.',
             'authors': 'N. Okonkwo', 'category': 'Plant Biology'}
    assert _rxiv.matches(entry, [])
    assert _rxiv.matches(entry, ['genome', 'crispr'])
    assert _rxiv.matches(entry, ['okonkwo'])
    assert _rxiv.matches(entry, ['plant biology'])
    assert not _rxiv.matches(entry, ['genome', 'proteomics'])


def test_matches_reads_a_term_as_a_word_prefix_rather_than_a_substring() -> None:
    """Find a plural from its singular without matching mid-word."""
    entry = {'title': 'Genomes and transcriptomes', 'abstract': '', 'authors': '',
             'category': ''}
    assert _rxiv.matches(entry, ['genome'])
    assert not _rxiv.matches(entry, ['nome'])


def test_authors_flip_into_the_corpus_name_order() -> None:
    """Rewrite ``Family, G.`` as ``G. Family`` to match the other providers."""
    assert _rxiv._authors('Okonkwo, N.') == 'N. Okonkwo'
    assert _rxiv._authors('Wheatley, A. K.; Juno, J. A.') == 'A. K. Wheatley; J. A. Juno'
    # A name with no comma is a consortium rather than a person, so it stands.
    assert _rxiv._authors('The ENCODE Project Consortium') == 'The ENCODE Project Consortium'
    assert _rxiv._authors('The RECOVERY Collaborative Group') == 'The RECOVERY Collaborative Group'
    assert _rxiv._authors('') == ''


def test_categories_carry_the_single_primary_subject_the_archives_file_under() -> None:
    """Emit one flagged category, keyed lower-case and displayed as written."""
    assert _rxiv._categories({'category': 'Infectious Diseases'}) == [
        {'id': 'infectious diseases', 'name': 'Infectious Diseases', 'is_primary': True}]
    assert _rxiv._categories({}) == []
    assert _rxiv._categories({'category': 'NA'}) == []


def test_limiter_for_separates_the_api_host_from_the_content_host() -> None:
    """Pace content requests by the crawl delay the archives publish.

    The metadata API and the content site share a registrable domain but
    advertise different limits, so the API host must not inherit the slower
    content window.
    """
    for server in (biorxiv.SERVER_CONFIG, medrxiv.SERVER_CONFIG):
        api_url = f'https://api.{server.web_host}/details/{server.name}/10.1101/1'
        assert _rxiv.limiter_for(server, api_url) is server.limiter

        for content_url in (
            f'{server.web_url}/content/early/2019/05/10/339747.source.xml',
            f'{server.web_url}/content/10.1101/339747v4.full.pdf',
            f'{server.web_url}/content/biorxiv/early/2019/05/10/339747/F1.large.jpg',
        ):
            assert _rxiv.limiter_for(server, content_url) is _rxiv.CONTENT_LIMITER

    assert _rxiv.CONTENT_LIMITER.min_interval == _rxiv.RXIV_CONTENT_MIN_INTERVAL == 7.0
    assert _rxiv.limiter_for(biorxiv.SERVER_CONFIG, 'not a url') is biorxiv.SERVER_CONFIG.limiter


@pytest.mark.parametrize('archive,server', [
    pytest.param(medrxiv, 'medrxiv', id='medrxiv'),
    pytest.param(biorxiv, 'biorxiv', id='biorxiv'),
])
def test_interval_url_rejects_a_bound_that_is_not_an_iso_date(archive: ModuleType, server: str) -> None:
    """Preserve the archive path and refuse malformed dates before requesting."""
    assert archive.interval_url('2024-01-01', '2024-12-31', 200) == (
        f'{archive.BASE_URL}/details/{server}/2024-01-01/2024-12-31/200/json')
    assert archive.interval_url('2024-01-01', '2024-12-31', -5).endswith('/0/json')
    with pytest.raises(ValueError, match='start_date must be an ISO date'):
        archive.interval_url('last week', '2024-12-31')
    with pytest.raises(ValueError, match='end_date must be an ISO date'):
        archive.interval_url('2024-01-01', '')


@pytest.mark.parametrize('archive,label', [
    pytest.param(medrxiv, 'medRxiv', id='medrxiv'),
    pytest.param(biorxiv, 'bioRxiv', id='biorxiv'),
])
def test_request_json_reports_malformed_and_unexpected_bodies(archive: ModuleType, label: str) -> None:
    """Retain the archive-specific error for malformed and non-object JSON."""
    session = FakeSession([FakeResponse(text='{broken')])
    with pytest.raises(RuntimeError, match=f'{label} returned malformed JSON'):
        archive.request_json(archive.BASE_URL, session=session)

    session = FakeSession([FakeResponse(text='[1, 2, 3]')])
    with pytest.raises(RuntimeError, match='unexpected payload of type list'):
        archive.request_json(archive.BASE_URL, session=session)


@pytest.mark.parametrize('archive,server,category', [
    pytest.param(medrxiv, 'medrxiv', 'Oncology', id='medrxiv'),
    pytest.param(biorxiv, 'biorxiv', 'Neuroscience', id='biorxiv'),
])
def test_interval_page_sends_the_category_to_the_host_that_applies_it(
    archive: ModuleType, server: str, category: str,
) -> None:
    """Keep archive path, filter host, exact category, and identifying headers."""
    session = FakeSession([FakeResponse(text=payload())])
    archive.interval_page('2024-01-01', '2024-12-31', cursor=60, category=category,
                          session=session)

    assert session.calls[0]['url'] == (
        f'{archive.CATEGORY_BASE_URL}/details/{server}/2024-01-01/2024-12-31/60/json')
    assert session.calls[0]['params'] == {'category': category}
    assert session.calls[0]['headers']['User-Agent'] == provider.USER_AGENT

    session = FakeSession([FakeResponse(text=payload())])
    archive.interval_page('2024-01-01', '2024-12-31', session=session)
    assert session.calls[0]['url'].startswith(archive.BASE_URL)
    assert session.calls[0]['params'] == {}


@pytest.mark.parametrize('archive,server,title,results,discussion', [
    pytest.param(
        medrxiv, 'medrxiv', 'Evolution of immunity',
        'Neutralising activity was widespread.', 'Immunity persisted.', id='medrxiv',
    ),
    pytest.param(
        biorxiv, 'biorxiv', 'Visual working memories',
        'Decoding tracked the aperture edges.', 'Memories are abstractions.', id='biorxiv',
    ),
])
def test_full_text_flattens_jats_and_preserves_the_structured_document(
    archive: ModuleType, server: str, title: str, results: str, discussion: str,
) -> None:
    """Retain native content and provenance while omitting tables and references."""
    jats = ('<article><front><article-meta><title-group>'
            f'<article-title>{title}</article-title>'
            '</title-group></article-meta></front><body>'
            f'<sec><title>Results</title><p>{results}</p>'
            '<table-wrap><p>Table residue</p></table-wrap></sec>'
            f'<sec><title>Discussion</title><p>{discussion}</p></sec>'
            '</body><back><ref-list><ref><p>A citation</p></ref></ref-list></back></article>')
    xml_url = f'https://www.{server}.org/x.source.xml'
    session = FakeSession([FakeResponse(text=jats)])
    text = archive.full_text({'jatsxml': xml_url}, session=session)

    assert text.startswith(title)
    assert results in text
    assert discussion in text
    assert 'Table residue' not in text
    assert 'A citation' not in text

    session = FakeSession([FakeResponse(text=jats)])
    document = archive.full_text_document(
        {f'{server}_doi': '10.1101/x', 'jatsxml': xml_url}, session=session,
    )
    assert document.content == jats
    assert document.document_format == 'jats'
    assert document.source_identifier == '10.1101/x'
    assert len(session.calls) == 1


@pytest.mark.parametrize('archive,server,label', [
    pytest.param(medrxiv, 'medrxiv', 'medRxiv', id='medrxiv'),
    pytest.param(biorxiv, 'biorxiv', 'bioRxiv', id='biorxiv'),
])
def test_full_text_reports_a_broken_document_and_skips_an_absent_one(
    archive: ModuleType, server: str, label: str,
) -> None:
    """Raise on unparseable JATS but treat a bodyless record as no text."""
    record = {'jatsxml': f'https://www.{server}.org/x.source.xml'}
    session = FakeSession([])
    assert archive.full_text({}, session=session) == ''
    assert session.calls == []

    session = FakeSession([FakeResponse(text='<article')])
    with pytest.raises(RuntimeError, match=f'{label} returned malformed JATS XML'):
        archive.full_text(record, session=session)

    session = FakeSession([FakeResponse(text='<article><front/></article>')])
    assert archive.full_text(record, session=session) == ''

    session = FakeSession([FakeResponse(text='   ')])
    assert archive.full_text(record, session=session) == ''
