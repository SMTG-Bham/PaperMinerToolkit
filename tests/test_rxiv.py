"""Unit tests for the shared bioRxiv-family client.

The two archives that use it have their own suites for their own DOI shapes and
record data. What is tested here is the behaviour that does not vary between
them, and the wiring that makes each archive's module address its own service.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import paperminer.biorxiv as biorxiv
import paperminer.medrxiv as medrxiv
from paperminer import _rxiv

from tests.doubles import FakeResponse, FakeSession

ARCHIVES = [medrxiv, biorxiv]
ARCHIVE_IDS = ['medrxiv', 'biorxiv']


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


def test_endpoint_trades_page_width_for_the_category_filter() -> None:
    """Address the filtering host only when a category is actually wanted."""
    assert _rxiv.endpoint() == (_rxiv.BASE_URL, _rxiv.PAGE_SIZE)
    assert _rxiv.endpoint('  ') == (_rxiv.BASE_URL, _rxiv.PAGE_SIZE)
    assert _rxiv.endpoint('oncology') == (_rxiv.CATEGORY_BASE_URL, _rxiv.CATEGORY_PAGE_SIZE)


def test_page_size_reads_the_length_the_host_actually_sent() -> None:
    """Step a walk by the page the host chose rather than by the constant."""
    assert _rxiv.page_size(json.loads(payload(count=30))) == 30
    assert _rxiv.page_size(json.loads(payload(count=0)), default=30) == 30
    assert _rxiv.page_size(None) == _rxiv.PAGE_SIZE


def test_total_results_reads_the_message_block() -> None:
    """Read the interval's record count, tolerating a payload that has none."""
    assert _rxiv.total_results(json.loads(payload(total='1281'))) == 1281
    assert _rxiv.total_results(json.loads(payload(total='not a number'))) == 0
    assert _rxiv.total_results(None) == 0


def test_page_cursors_walks_pages_from_the_last_one_back_to_zero() -> None:
    """Start at the final page so the newest postings are read first."""
    assert list(_rxiv.page_cursors(1281, 100)) == list(range(1200, -1, -100))
    assert list(_rxiv.page_cursors(250, 30)) == list(range(240, -1, -30))
    # An exact multiple must not produce an empty page past the end.
    assert list(_rxiv.page_cursors(200, 100)) == [100, 0]
    assert list(_rxiv.page_cursors(1, 100)) == [0]
    assert list(_rxiv.page_cursors(0, 100)) == []
    assert list(_rxiv.page_cursors(50, 0)) == list(range(49, -1, -1))


@pytest.mark.parametrize('archive', ARCHIVES, ids=ARCHIVE_IDS)
def test_request_json_reads_an_absent_record_out_of_a_200_response(archive: Any) -> None:
    """Report the statuses that mean nothing found as nothing rather than failure."""
    for status in _rxiv.EMPTY_STATUSES:
        session = FakeSession([FakeResponse(text=payload(status=status))])
        assert archive.request_json(_rxiv.BASE_URL, session=session) is None


@pytest.mark.parametrize('archive', ARCHIVES, ids=ARCHIVE_IDS)
def test_request_json_raises_on_a_rejection_dressed_as_a_200_response(archive: Any) -> None:
    """Name the archive when it reports a rejected request in the body."""
    session = FakeSession([FakeResponse(text=payload(status='bad interval'))])
    with pytest.raises(RuntimeError, match=f'{archive.SERVER_CONFIG.label} rejected the request'):
        archive.request_json(_rxiv.BASE_URL, session=session)


def test_parse_query_lifts_the_scope_terms_out_of_the_phrase() -> None:
    """Separate the walk's scope from the terms records are matched against."""
    terms, scope = _rxiv.parse_query(
        '"gene therapy" crispr category:Genomics from:2024-01-01 to:2024-12-31')
    assert terms == ['gene therapy', 'crispr']
    assert scope == {'category': 'Genomics', 'from': '2024-01-01', 'to': '2024-12-31'}
    assert _rxiv.parse_query('') == ([], {})


def test_parse_query_rejects_a_bound_that_is_not_an_iso_date() -> None:
    """Refuse a malformed scope here rather than spending a walk on it."""
    with pytest.raises(ValueError, match='from: must be an ISO date'):
        _rxiv.parse_query('crispr from:last-week')
    with pytest.raises(ValueError, match='to: must be an ISO date'):
        _rxiv.parse_query('crispr to:2024')


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
    assert _rxiv._authors('') == ''


def test_categories_carry_the_single_primary_subject_the_archives_file_under() -> None:
    """Emit one flagged category, keyed lower-case and displayed as written."""
    assert _rxiv._categories({'category': 'Infectious Diseases'}) == [
        {'id': 'infectious diseases', 'name': 'Infectious Diseases', 'is_primary': True}]
    assert _rxiv._categories({}) == []
    assert _rxiv._categories({'category': 'NA'}) == []
