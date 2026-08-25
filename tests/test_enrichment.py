"""Unit tests for Crossref and OpenAlex metadata enrichment.

This module tests provider request construction, the field precedence rules
between the two providers, the structured author, subject, and reference rows,
and the batched orchestration that keeps enrichment idempotent and resumable.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import requests

import paperminer.corpus as corpus
import paperminer.enrichment as enrichment


def crossref_work(doi: str = '10.1234/example') -> dict[str, Any]:
    """Return a Crossref work message exercising every consumed field."""
    return {
        'DOI': doi,
        'title': ['Deposited title'],
        'container-title': ['Deposited Journal'],
        'published-print': {'date-parts': [[2024, 3, 7]]},
        'published-online': {'date-parts': [[2024, 2, 1]]},
        'publisher': 'Deposited Publisher',
        'type': 'journal-article',
        'volume': '12',
        'issue': '4',
        'page': '100-110',
        'language': 'en',
        'issn-type': [
            {'value': '2000-0002', 'type': 'electronic'},
            {'value': '2000-0001', 'type': 'print'},
        ],
        'is-referenced-by-count': 11,
        'references-count': 2,
        'license': [
            {'content-version': 'tdm', 'URL': 'https://example.org/text-mining'},
            {'content-version': 'vor', 'URL': 'https://creativecommons.org/licenses/by/4.0'},
        ],
        'author': [
            {'given': 'Jane A.', 'family': 'Smith', 'sequence': 'first',
             'ORCID': 'https://orcid.org/0000-0002-1825-0097',
             'affiliation': [{'name': 'Example University'}]},
            {'given': 'Wei', 'family': 'Chen', 'sequence': 'additional', 'affiliation': []},
        ],
        'reference': [
            {'key': 'ref1', 'DOI': '10.1234/cited-one', 'article-title': 'Cited one'},
            {'key': 'ref2', 'unstructured': 'Someone et al., Journal, 2019'},
        ],
    }


def openalex_work(doi: str | None = 'https://doi.org/10.1234/Example',
                  identifier: str = 'https://openalex.org/W123') -> dict[str, Any]:
    """Return an OpenAlex work record exercising every consumed field."""
    return {
        'id': identifier,
        'doi': doi,
        'ids': {'openalex': identifier, 'pmid': 'https://pubmed.ncbi.nlm.nih.gov/1'},
        'title': 'Derived title',
        'display_name': 'Derived title',
        'publication_date': '2024-02-01',
        'publication_year': 2024,
        'language': 'fr',
        'type': 'article',
        'biblio': {'volume': '99', 'issue': '9', 'first_page': '1', 'last_page': '9'},
        'primary_location': {
            'license': 'cc-by-nc',
            'source': {
                'display_name': 'Derived Journal',
                'host_organization_name': 'Derived Publisher',
                'issn': ['2000-0003'],
                'issn_l': '2000-0001',
            },
        },
        'best_oa_location': {'license': 'cc-by'},
        'open_access': {'is_oa': True, 'oa_status': 'gold'},
        'authorships': [
            {
                'author_position': 'first',
                'is_corresponding': True,
                'author': {'id': 'https://openalex.org/A1', 'display_name': 'Jane A. Smith',
                           'orcid': 'https://orcid.org/0000-0002-1825-0097'},
                'institutions': [
                    {'display_name': 'Example University', 'ror': 'https://ror.org/03vek6s52',
                     'country_code': 'GB'},
                    {'display_name': 'Second Institute', 'ror': 'https://ror.org/0080fxk18',
                     'country_code': 'US'},
                ],
                'raw_affiliation_strings': ['Example University, UK', 'Second Institute, USA'],
            },
            {
                'author_position': 'last',
                'author': {'id': 'https://openalex.org/A2', 'display_name': 'Wei Chen',
                           'orcid': None},
                'institutions': [],
                'raw_affiliation_strings': ['Third Lab'],
            },
        ],
        'cited_by_count': 42,
        'referenced_works_count': 3,
        'referenced_works': ['https://openalex.org/W9', 'https://openalex.org/W8'],
        'is_retracted': False,
        'primary_topic': {
            'id': 'https://openalex.org/T1', 'display_name': 'Batteries',
            'subfield': {'id': 'https://openalex.org/subfields/2500', 'display_name': 'Materials'},
            'field': {'id': 'https://openalex.org/fields/25', 'display_name': 'Materials Science'},
            'domain': {'id': 'https://openalex.org/domains/3', 'display_name': 'Physical Sciences'},
        },
        'topics': [
            {'id': 'https://openalex.org/T1', 'display_name': 'Batteries', 'score': 0.99,
             'field': {'display_name': 'Materials Science'},
             'domain': {'display_name': 'Physical Sciences'}},
            {'id': 'https://openalex.org/T2', 'display_name': 'Electrolytes', 'score': 0.8},
        ],
        'concepts': [{'id': 'https://openalex.org/C1', 'display_name': 'Chemistry',
                      'score': 0.6, 'level': 0}],
        'keywords': [{'id': 'https://openalex.org/keywords/sei', 'display_name': 'SEI',
                      'score': 0.5}],
        'sustainable_development_goals': [
            {'id': 'https://metadata.un.org/sdg/7', 'display_name': 'Affordable energy',
             'score': 0.4},
        ],
    }


class FakeResponse:
    """Prepared OpenAlex JSON response with a configurable status code."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        """Store the prepared payload and status."""
        self.payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        """Validate the prepared response status."""
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error', response=self)

    def json(self) -> dict[str, Any]:
        """Return the prepared JSON payload."""
        return self.payload


class FakeOpenAlexSession:
    """Return prepared OpenAlex responses and record request arguments."""

    def __init__(self, responses: Iterable[FakeResponse]) -> None:
        """Initialize the session with prepared responses."""
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: Mapping[str, Any], headers: Mapping[str, str],
            timeout: float) -> FakeResponse:
        """Record the request and return the next prepared response."""
        self.calls.append({'url': url, 'params': dict(params), 'timeout': timeout})
        return next(self.responses, FakeResponse({'results': []}))


class FakeCrossrefResponse:
    """Successful Crossref response wrapping one prepared message."""

    def __init__(self, message: dict[str, Any], status_code: int = 200) -> None:
        """Store the prepared Crossref message."""
        self.message = message
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        """Represent a successful HTTP status check."""
        return None

    def json(self) -> dict[str, Any]:
        """Return the prepared message as a response payload."""
        return {'message': self.message}


class FakeCrossrefSession:
    """Return prepared Crossref pages and record request arguments."""

    def __init__(self, messages: Iterable[dict[str, Any]]) -> None:
        """Initialize the session with prepared response messages."""
        self.messages = iter(messages)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: Mapping[str, Any], headers: Mapping[str, str],
            timeout: float) -> FakeCrossrefResponse:
        """Record the request and return the next prepared response."""
        self.calls.append({'url': url, 'params': dict(params)})
        return FakeCrossrefResponse(next(self.messages, {'items': []}))


@contextlib.contextmanager
def open_corpus(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a corpus connection that is committed and then closed."""
    conn = corpus.connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def seed_corpus(db_path: Path, papers: Iterable[Mapping[str, Any]]) -> None:
    """Insert paper rows into a fresh corpus."""
    with open_corpus(db_path) as conn:
        for paper in papers:
            corpus.upsert_paper(conn, paper)


def test_configured_sources_expands_all_and_rejects_unknown_providers() -> None:
    """Expand the all sentinel and reject an unsupported provider name."""
    assert enrichment._configured_sources(None) == ['crossref', 'openalex', 'pubmed', 'arxiv',
                                                   'medrxiv', 'biorxiv', 'chemrxiv']
    assert enrichment._configured_sources(['all']) == ['crossref', 'openalex', 'pubmed', 'arxiv',
                                                      'medrxiv', 'biorxiv', 'chemrxiv']
    assert enrichment._configured_sources(['openalex']) == ['openalex']
    with pytest.raises(ValueError):
        enrichment._configured_sources(['scopus'])


def test_partition_candidates_splits_by_available_lookup_key() -> None:
    """Route papers by DOI, by OpenAlex identifier, or to the unresolved set."""
    by_doi, by_openalex, unresolved = enrichment._partition_candidates([
        {'paper_id': 'doi:10.1234/a', 'doi': '10.1234/A'},
        {'paper_id': 'openalex:W7', 'doi': '', 'openalex_id': ''},
        {'paper_id': 'core:5', 'doi': '', 'openalex_id': 'https://openalex.org/W8'},
        {'paper_id': 'external:scan', 'doi': '', 'openalex_id': ''},
    ])

    assert by_doi == {'10.1234/a': 'doi:10.1234/a'}
    assert by_openalex == {'W7': 'openalex:W7', 'W8': 'core:5'}
    assert unresolved == ['external:scan']


def test_crossref_fields_map_core_bibliographic_values() -> None:
    """Map the deposited bibliographic record onto corpus column names."""
    mapped = enrichment._crossref_fields(crossref_work())

    assert mapped['publisher'] == 'Deposited Publisher'
    assert mapped['work_type'] == 'journal-article'
    assert mapped['volume'] == '12'
    assert mapped['issue'] == '4'
    assert mapped['pages'] == '100-110'
    assert mapped['language'] == 'en'
    assert mapped['authors'] == 'Jane A. Smith; Wei Chen'
    assert mapped['referenced_works_count'] == 2


def test_crossref_fields_order_typed_issns_with_print_first() -> None:
    """Put the print ISSN first regardless of the deposited order."""
    assert enrichment._crossref_fields(crossref_work())['issn'] == '2000-0001;2000-0002'


def test_crossref_fields_ignore_text_mining_licences() -> None:
    """Skip a text-mining grant and keep the version-of-record licence."""
    work = crossref_work()
    assert enrichment._crossref_fields(work)['license'] == 'https://creativecommons.org/licenses/by/4.0'

    work['license'] = [{'content-version': 'tdm', 'URL': 'https://example.org/text-mining'}]
    assert enrichment._crossref_fields(work)['license'] == ''


def test_crossref_retraction_requires_a_retraction_update_type() -> None:
    """Treat only a retraction update as a retraction, never a correction."""
    work = crossref_work()
    work['updated-by'] = [{'type': 'correction', 'DOI': '10.1234/correction'}]
    assert enrichment._crossref_retraction(work) == (False, '')

    work['updated-by'].append({'type': 'retraction', 'DOI': '10.1234/Retraction'})
    assert enrichment._crossref_retraction(work) == (True, '10.1234/retraction')


def test_openalex_fields_map_impact_and_open_access_values() -> None:
    """Map the derived impact, access, and identifier fields from OpenAlex."""
    mapped = enrichment._openalex_fields(openalex_work())

    assert mapped['openalex_id'] == 'W123'
    assert mapped['doi'] == '10.1234/example'
    assert mapped['cited_by_count'] == 42
    assert mapped['referenced_works_count'] == 3
    assert mapped['is_oa'] == 1
    assert mapped['oa_status'] == 'gold'
    assert mapped['license'] == 'cc-by'
    assert mapped['issn_l'] == '2000-0001'
    assert mapped['pages'] == '1-9'


def test_openalex_fields_tolerate_a_null_primary_location() -> None:
    """Survive the null locations OpenAlex returns for venue-less works."""
    work = openalex_work()
    work['primary_location'] = None
    work['best_oa_location'] = None
    work['open_access'] = None

    mapped = enrichment._openalex_fields(work)

    assert mapped['journal'] == ''
    assert mapped['publisher'] == ''
    assert mapped['issn'] == ''
    assert mapped['license'] == ''
    assert mapped['is_oa'] == 0


def test_merge_prefers_crossref_bibliography_and_openalex_impact() -> None:
    """Apply the documented precedence between the two providers."""
    merged = enrichment._merge_fields('doi:10.1234/example', crossref_work(), openalex_work(),
                                     ['crossref', 'openalex'])

    assert merged['publisher'] == 'Deposited Publisher'
    assert merged['work_type'] == 'journal-article'
    assert merged['volume'] == '12'
    assert merged['language'] == 'en'
    assert merged['issn'] == '2000-0001;2000-0002'
    assert merged['cited_by_count'] == 42
    assert merged['oa_status'] == 'gold'
    assert merged['license'] == 'cc-by'
    assert merged['issn_l'] == '2000-0001'
    assert merged['openalex_id'] == 'W123'
    assert merged['enrichment_sources'] == 'crossref;openalex'
    assert merged['enrichment_status'] == 'succeeded'


def test_merge_falls_back_to_openalex_when_crossref_is_missing() -> None:
    """Use derived values and report a partial result when Crossref misses."""
    merged = enrichment._merge_fields('doi:10.1234/example', None, openalex_work(),
                                     ['crossref', 'openalex'])

    assert merged['publisher'] == 'Derived Publisher'
    assert merged['language'] == 'fr'
    assert merged['enrichment_sources'] == 'openalex'
    assert merged['enrichment_status'] == 'partial'


def test_merge_flags_a_retraction_reported_by_either_provider() -> None:
    """Flag a retraction when either provider reports one."""
    retracted_crossref = crossref_work()
    retracted_crossref['updated-by'] = [{'type': 'retraction', 'DOI': '10.1234/r'}]
    retracted_openalex = openalex_work()
    retracted_openalex['is_retracted'] = True

    both = ['crossref', 'openalex']
    assert enrichment._merge_fields('p', retracted_crossref, openalex_work(), both)['is_retracted'] == 1
    assert enrichment._merge_fields('p', crossref_work(), retracted_openalex, both)['is_retracted'] == 1
    assert enrichment._merge_fields('p', crossref_work(), openalex_work(), both)['is_retracted'] == 0


def test_merge_records_both_citation_counts_separately() -> None:
    """Keep the two differently scoped citation counts from being reconciled."""
    merged = enrichment._merge_fields('p', crossref_work(), openalex_work(),
                                     ['crossref', 'openalex'])
    provenance = merged['enrichment_json']

    assert merged['cited_by_count'] == 42
    assert provenance['crossref']['is_referenced_by_count'] == 11
    assert provenance['openalex']['cited_by_count'] == 42
    assert provenance['crossref']['published_print'] == '2024-03-07'
    assert provenance['crossref']['published_online'] == '2024-02-01'


def test_merge_reports_not_found_when_no_provider_has_the_work() -> None:
    """Report a miss without an enrichment timestamp when nothing is found."""
    merged = enrichment._merge_fields('p', None, None, ['crossref', 'openalex'])

    assert merged['enrichment_status'] == 'not_found'
    assert merged['enrichment_sources'] == ''
    assert merged['enriched_at'] == ''


def test_merge_records_provider_failures_separately_from_not_found() -> None:
    """Distinguish an unavailable provider from a successful empty lookup."""
    failed = enrichment._merge_fields(
        'p', None, None, ['crossref'], provider_errors={'crossref': 'service unavailable'})
    partial = enrichment._merge_fields(
        'p', None, openalex_work(), ['crossref', 'openalex'],
        provider_errors={'crossref': 'service unavailable'})

    assert failed['enrichment_status'] == 'failed'
    assert failed['enrichment_json']['provider_errors'] == {'crossref': 'service unavailable'}
    assert partial['enrichment_status'] == 'partial'
    assert partial['enrichment_json']['provider_errors'] == {'crossref': 'service unavailable'}


def test_author_rows_emit_one_row_per_affiliation() -> None:
    """Emit a row per author and affiliation so institutions stay joinable."""
    rows = enrichment._author_rows('p', crossref_work(), openalex_work())

    first = [row for row in rows if row['author_position'] == 0]
    assert [row['affiliation_rank'] for row in first] == [0, 1]
    assert [row['institution_ror'] for row in first] == ['03vek6s52', '0080fxk18']
    assert first[0]['affiliation'] == 'Example University, UK'
    assert first[0]['is_corresponding'] == 1
    assert first[0]['position_label'] == 'first'

    second = [row for row in rows if row['author_position'] == 1]
    assert len(second) == 1
    assert second[0]['affiliation'] == 'Third Lab'
    assert second[0]['institution_ror'] == ''


def test_author_rows_take_names_from_crossref_and_identifiers_from_openalex() -> None:
    """Combine the deposited name split with the disambiguated identifiers."""
    rows = enrichment._author_rows('p', crossref_work(), openalex_work())

    assert rows[0]['display_name'] == 'Jane A. Smith'
    assert rows[0]['given_name'] == 'Jane A.'
    assert rows[0]['family_name'] == 'Smith'
    assert rows[0]['orcid'] == '0000-0002-1825-0097'
    assert rows[0]['openalex_author_id'] == 'A1'
    assert rows[0]['source'] == 'openalex'


def test_author_rows_tolerate_an_invalid_orcid() -> None:
    """Drop a malformed ORCID rather than aborting the whole batch."""
    work = openalex_work()
    work['authorships'][0]['author']['orcid'] = 'not-an-orcid'
    deposited = crossref_work()
    deposited['author'][0]['ORCID'] = 'also-invalid'

    assert enrichment._author_rows('p', deposited, work)[0]['orcid'] == ''


def test_author_rows_extend_a_truncated_openalex_list_from_crossref() -> None:
    """Continue past the OpenAlex author list using the deposited authors."""
    work = openalex_work()
    work['authorships'] = work['authorships'][:1]

    rows = enrichment._author_rows('p', crossref_work(), work)

    assert [row['author_position'] for row in rows] == [0, 0, 1]
    assert rows[-1]['source'] == 'crossref'
    assert rows[-1]['display_name'] == 'Wei Chen'


def test_author_rows_use_crossref_alone_when_openalex_has_no_record() -> None:
    """Fall back to the deposited author list with a Crossref provenance."""
    rows = enrichment._author_rows('p', crossref_work(), None)

    assert [row['source'] for row in rows] == ['crossref', 'crossref']
    assert [row['position_label'] for row in rows] == ['first', 'last']
    assert rows[0]['affiliation'] == 'Example University'


def test_subject_rows_mark_the_primary_topic_and_cover_every_scheme() -> None:
    """Store topics, hierarchy rollups, concepts, keywords, and SDGs."""
    rows = enrichment._subject_rows('p', openalex_work())
    schemes = {row['scheme'] for row in rows}

    assert schemes == {'topic', 'subfield', 'field', 'domain', 'concept', 'keyword', 'sdg'}
    primary = [row for row in rows if row['scheme'] == 'topic' and row['is_primary']]
    assert [row['subject_id'] for row in primary] == ['T1']
    assert [row['score'] for row in rows if row['subject_id'] == 'T1'] == [0.99]
    assert [row['level'] for row in rows if row['scheme'] == 'concept'] == [0]


def test_subject_rows_are_empty_without_an_openalex_record() -> None:
    """Return no subjects when OpenAlex has no record, as Crossref has none."""
    assert enrichment._subject_rows('p', None) == []


def test_reference_rows_keep_each_provider_separate() -> None:
    """Store both reference lists side by side without merging them."""
    rows = enrichment._reference_rows('p', crossref_work(), openalex_work())

    from_crossref = [row for row in rows if row['source'] == 'crossref']
    from_openalex = [row for row in rows if row['source'] == 'openalex']
    assert [row['referenced_doi'] for row in from_crossref] == ['10.1234/cited-one', '']
    assert from_crossref[0]['referenced_title'] == 'Cited one'
    assert from_crossref[1]['unstructured'] == 'Someone et al., Journal, 2019'
    assert [row['referenced_openalex_id'] for row in from_openalex] == ['W9', 'W8']


def enrich(db_path: Path, crossref_messages: Iterable[dict[str, Any]],
           openalex_payloads: Iterable[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Run enrich_corpus against prepared provider responses.

    Parameters
    ----------
    db_path : pathlib.Path
        Corpus to enrich.
    crossref_messages : Iterable[dict[str, Any]]
        Prepared Crossref response messages.
    openalex_payloads : Iterable[dict[str, Any]]
        Prepared OpenAlex response payloads.
    **kwargs : Any
        Extra keyword arguments forwarded to enrich_corpus.

    Returns
    -------
    dict[str, Any]
        Run summary plus both session doubles under ``crossref`` and ``openalex``.
    """
    crossref_session = FakeCrossrefSession(crossref_messages)
    openalex_session = FakeOpenAlexSession(FakeResponse(payload) for payload in openalex_payloads)
    kwargs.setdefault('sources', ['crossref', 'openalex'])
    summary = enrichment.enrich_corpus(
        db_path,
        email='me@example.com',
        api_key='',
        pace=0,
        crossref_session=crossref_session,
        openalex_session=openalex_session,
        **kwargs,
    )
    return {**summary, 'crossref': crossref_session, 'openalex': openalex_session}


def test_enrich_corpus_writes_every_enrichment_column(tmp_path: Path) -> None:
    """Populate each enrichment column from the two provider records."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'title': '', 'sources': 'seed'}])

    summary = enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}])

    with open_corpus(db_path) as conn:
        row = corpus.paper_rows(conn)[0]

    assert summary['succeeded'] == 1
    assert row['enrichment_status'] == 'succeeded'
    assert row['publisher'] == 'Deposited Publisher'
    assert row['work_type'] == 'journal-article'
    assert row['volume'] == '12'
    assert row['issue'] == '4'
    assert row['pages'] == '100-110'
    assert row['issn'] == '2000-0001;2000-0002'
    assert row['issn_l'] == '2000-0001'
    assert row['language'] == 'en'
    assert row['is_oa'] == 1
    assert row['oa_status'] == 'gold'
    assert row['license'] == 'cc-by'
    assert row['is_retracted'] == 0
    assert row['cited_by_count'] == 42
    assert row['referenced_works_count'] == 3
    assert row['openalex_id'] == 'W123'
    assert row['enrichment_sources'] == 'crossref;openalex'
    assert json.loads(row['enrichment_json'])['crossref']['is_referenced_by_count'] == 11


def test_enrich_corpus_preserves_curated_core_columns(tmp_path: Path) -> None:
    """Fill an empty core column but never overwrite a populated one."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'title': 'Curated title',
                           'journal': '', 'sources': 'seed'}])

    enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}])

    with open_corpus(db_path) as conn:
        row = corpus.paper_rows(conn)[0]

    assert row['title'] == 'Curated title'
    assert row['journal'] == 'Deposited Journal'


def test_enrich_corpus_leaves_metadata_json_untouched(tmp_path: Path) -> None:
    """Never write the raw provider payload column reserved for imports."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed',
                           'metadata': {'kept': True}}])

    enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}])

    with open_corpus(db_path) as conn:
        row = corpus.paper_rows(conn)[0]

    assert json.loads(row['metadata_json']) == {'kept': True}


def test_enrich_corpus_marks_papers_without_lookup_keys_unresolved(tmp_path: Path) -> None:
    """Skip papers with no lookup key without spending a single request."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'paper_id': 'external:scan', 'title': 'Scanned', 'sources': 'external'}])

    summary = enrich(db_path, [], [])

    with open_corpus(db_path) as conn:
        row = corpus.paper_rows(conn)[0]

    assert summary['unresolved'] == 1
    assert row['enrichment_status'] == 'unresolved'
    assert summary['crossref'].calls == []
    assert summary['openalex'].calls == []


def test_enrich_corpus_marks_missing_records_not_found(tmp_path: Path) -> None:
    """Record a miss when neither provider knows the DOI."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/missing', 'sources': 'seed'}])

    summary = enrich(db_path, [{'items': []}], [{'results': []}])

    with open_corpus(db_path) as conn:
        row = corpus.paper_rows(conn)[0]

    assert summary['not_found'] == 1
    assert row['enrichment_status'] == 'not_found'


def test_enrich_corpus_reconciles_reordered_and_partial_batches(tmp_path: Path) -> None:
    """Key provider responses by DOI rather than by request order."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/one', 'sources': 'seed'},
                          {'doi': '10.1234/two', 'sources': 'seed'}])

    summary = enrich(
        db_path,
        [{'items': [crossref_work('10.1234/two'), crossref_work('10.1234/one')]}],
        [{'results': [openalex_work(doi='https://doi.org/10.1234/two')]}],
    )

    with open_corpus(db_path) as conn:
        rows = {row['doi']: row for row in corpus.paper_rows(conn)}

    assert summary['succeeded'] == 1
    assert summary['partial'] == 1
    assert rows['10.1234/two']['enrichment_sources'] == 'crossref;openalex'
    assert rows['10.1234/one']['enrichment_sources'] == 'crossref'


def test_enrich_corpus_normalizes_mixed_case_dois(tmp_path: Path) -> None:
    """Match a stored mixed-case DOI against the lowercase provider response."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1103/PhysRevB.94.035105', 'sources': 'seed'}])

    summary = enrich(db_path,
                     [{'items': [crossref_work('10.1103/physrevb.94.035105')]}],
                     [{'results': []}])

    assert summary['succeeded'] == 0
    assert summary['partial'] == 1
    assert summary['crossref'].calls[0]['params']['filter'] == 'doi:10.1103/physrevb.94.035105'


def test_enrich_corpus_is_idempotent_on_a_second_run(tmp_path: Path) -> None:
    """Spend no requests re-running enrichment over an enriched corpus."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed'}])
    enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}])

    second = enrich(db_path, [], [])

    assert second['succeeded'] == 0
    assert second['crossref'].calls == []
    assert second['openalex'].calls == []


def test_enrich_corpus_replaces_rather_than_duplicates_child_rows(tmp_path: Path) -> None:
    """Replace child rows on a forced re-run instead of appending them."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed'}])
    enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}])

    with open_corpus(db_path) as conn:
        first = len(corpus.paper_authors(conn, 'doi:10.1234/example'))

    enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}], force=True)

    with open_corpus(db_path) as conn:
        assert len(corpus.paper_authors(conn, 'doi:10.1234/example')) == first


def test_enrich_corpus_refreshes_citation_counts_on_force(tmp_path: Path) -> None:
    """Overwrite a time-varying value when enrichment is forced."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed'}])
    enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}])

    updated = openalex_work()
    updated['cited_by_count'] = 99
    enrich(db_path, [{'items': [crossref_work()]}], [{'results': [updated]}], force=True)

    with open_corpus(db_path) as conn:
        assert corpus.paper_rows(conn)[0]['cited_by_count'] == 99


def test_enrich_corpus_resumes_after_a_limit(tmp_path: Path) -> None:
    """Stop at the requested limit and continue from there on the next run."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': f'10.1234/paper{index}', 'sources': 'seed'} for index in range(3)])

    first = enrich(db_path, [{'items': [crossref_work('10.1234/paper0')]}],
                   [{'results': []}], limit=1, batch_size=1)
    second = enrich(db_path,
                    [{'items': [crossref_work('10.1234/paper1')]},
                     {'items': [crossref_work('10.1234/paper2')]}],
                    [{'results': []}, {'results': []}], batch_size=1)

    with open_corpus(db_path) as conn:
        statuses = [row['enrichment_status'] for row in corpus.paper_rows(conn)]

    assert first['partial'] == 1
    assert second['partial'] == 2
    assert statuses == ['partial', 'partial', 'partial']


def test_enrich_corpus_isolates_a_provider_budget_error(tmp_path: Path) -> None:
    """Keep other providers working when one exhausts its budget."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': f'10.1234/paper{index}', 'sources': 'seed'} for index in range(2)])

    crossref_session = FakeCrossrefSession([{'items': [crossref_work('10.1234/paper0')]}])
    openalex_session = FakeOpenAlexSession([FakeResponse({'results': []}),
                                            FakeResponse({}, status_code=429)])

    summary = enrichment.enrich_corpus(
        db_path, sources=['crossref', 'openalex'], email='me@example.com', api_key='', pace=0,
        batch_size=1, crossref_session=crossref_session, openalex_session=openalex_session)

    with open_corpus(db_path) as conn:
        statuses = [row['enrichment_status'] for row in corpus.paper_rows(conn)]

    assert statuses == ['partial', 'failed']
    assert summary['partial'] == 1
    assert summary['failed'] == 1


def test_enrich_corpus_matches_doi_less_papers_by_openalex_identifier(tmp_path: Path) -> None:
    """Look up a DOI-less paper through its OpenAlex identifier."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'paper_id': 'openalex:W123', 'title': 'No DOI', 'sources': 'openalex'}])

    summary = enrich(db_path, [], [{'results': [openalex_work(doi=None)]}], sources=['openalex'])

    with open_corpus(db_path) as conn:
        row = corpus.paper_rows(conn)[0]

    assert summary['succeeded'] == 1
    assert row['openalex_id'] == 'W123'
    assert summary['openalex'].calls[0]['params']['filter'] == 'ids.openalex:W123'


def test_reset_clears_enrichment_status_but_keeps_enrichment_data(tmp_path: Path) -> None:
    """Re-arm the enrichment stage without discarding enrichment values."""
    import paperminer.utilities as utilities

    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed'}])
    enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}])

    utilities.reset(db_path)

    with open_corpus(db_path) as conn:
        row = corpus.paper_rows(conn)[0]
        authors = corpus.paper_authors(conn, 'doi:10.1234/example')

    assert row['enrichment_status'] == 'pending'
    assert row['publisher'] == 'Deposited Publisher'
    assert row['cited_by_count'] == 42
    assert authors


def test_enrich_from_crossref_message_writes_import_metadata(tmp_path: Path) -> None:
    """Store the metadata a PDF import already fetched from Crossref."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'paper_id': 'external:scan', 'doi': '10.1234/example',
                           'sources': 'external'}])

    with open_corpus(db_path) as conn:
        enrichment._enrich_from_crossref_message(conn, 'external:scan', crossref_work())
        row = corpus.paper_rows(conn)[0]
        references = corpus.paper_references(conn, 'external:scan')

    assert row['publisher'] == 'Deposited Publisher'
    assert row['work_type'] == 'journal-article'
    assert row['enrichment_sources'] == 'crossref'
    assert len(references) == 2


def test_resolve_reference_targets_links_in_corpus_dois(tmp_path: Path) -> None:
    """Link a stored reference DOI to the corpus paper that carries it."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed'},
                          {'doi': '10.1234/cited-one', 'sources': 'seed'}])

    with open_corpus(db_path) as conn:
        enrichment._enrich_from_crossref_message(conn, 'doi:10.1234/example', crossref_work())
        enrichment._resolve_reference_targets(conn)
        references = corpus.paper_references(conn, 'doi:10.1234/example')

    assert references[0]['referenced_paper_id'] == 'doi:10.1234/cited-one'
    assert references[1]['referenced_paper_id'] == ''


def test_enrich_corpus_rejects_an_invalid_batch_size(tmp_path: Path) -> None:
    """Reject an out-of-range batch size before opening the corpus."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed'}])

    with pytest.raises(ValueError, match='batch_size'):
        enrichment.enrich_corpus(db_path, email='me@example.com', api_key='', batch_size=0)


def test_enrich_papers_resolves_rows_merged_under_another_identifier(tmp_path: Path) -> None:
    """Enrich the stored row even when discovery merged it under another id."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'paper_id': 'core:7', 'doi': '10.1234/example', 'sources': 'core'}])

    crossref_session = FakeCrossrefSession([{'items': [crossref_work()]}])
    openalex_session = FakeOpenAlexSession([FakeResponse({'results': [openalex_work()]})])

    with open_corpus(db_path) as conn:
        summary = enrichment.enrich_papers(
            conn,
            [{'paper_id': 'doi:10.1234/example', 'doi': '10.1234/example'}],
            email='me@example.com', api_key='', pace=0,
            crossref_session=crossref_session, openalex_session=openalex_session,
        )
        row = corpus.paper_rows(conn)[0]

    assert summary['succeeded'] == 1
    assert row['paper_id'] == 'core:7'
    assert row['publisher'] == 'Deposited Publisher'


def test_enrich_corpus_reselects_stale_rows_with_refresh_after(tmp_path: Path) -> None:
    """Re-enrich a succeeded paper once its enrichment is old enough."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed'}])
    enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}])

    with open_corpus(db_path) as conn:
        conn.execute("UPDATE papers SET enriched_at = '2020-01-01T00:00:00+00:00'")

    fresh = enrich(db_path, [{'items': [crossref_work()]}], [{'results': [openalex_work()]}],
                   refresh_after=1)

    assert fresh['succeeded'] == 1


def test_resolve_reference_dois_deduplicates_identifiers_across_papers(tmp_path: Path) -> None:
    """Resolve each cited work once even when several papers cite it."""
    db_path = tmp_path / 'corpus.db'
    seed_corpus(db_path, [{'doi': '10.1234/one', 'sources': 'seed'},
                          {'doi': '10.1234/two', 'sources': 'seed'}])
    with open_corpus(db_path) as conn:
        for paper_id in ('doi:10.1234/one', 'doi:10.1234/two'):
            corpus.write_enrichment(
                conn,
                [{**{field: '' for field in corpus.enrichment_update_fields()},
                  'paper_id': paper_id, 'enrichment_status': 'succeeded',
                  'updated_at': corpus.utc_now()}],
                references=[{'paper_id': paper_id, 'source': 'openalex',
                             'reference_rank': 0, 'referenced_openalex_id': 'W9'}],
            )

    session = FakeOpenAlexSession([FakeResponse({'results': [
        {'id': 'https://openalex.org/W9', 'doi': 'https://doi.org/10.1234/Cited'},
    ]})])
    updated = enrichment.resolve_reference_dois(db_path, api_key='', session=session)

    with open_corpus(db_path) as conn:
        references = corpus.paper_references(conn, 'doi:10.1234/one')

    assert len(session.calls) == 1
    assert session.calls[0]['params']['filter'] == 'ids.openalex:W9'
    assert updated == 2
    assert references[0]['referenced_doi'] == '10.1234/cited'


def pubmed_article(pmid: str = '31234567',
                   doi: str = '10.1234/pubmed-one') -> dict[str, Any]:
    """Return a PubMed article mapping exercising every consumed field."""
    return {
        'paper_id': f'doi:{doi}' if doi else f'pmid:{pmid}',
        'doi': doi,
        'pmid': pmid,
        'pmcid': 'PMC9876543',
        'title': 'PubMed title',
        'journal': 'PubMed Journal',
        'publication_date': '2024-03-07',
        'authors': 'Jane A Smith',
        'sources': 'pubmed',
        'abstract': 'An abstract.',
        'mesh': [
            {'scheme': 'mesh', 'id': 'D007854', 'name': 'Lithium', 'is_primary': '1'},
            {'scheme': 'mesh_qualifier', 'id': 'Q000032', 'name': 'analysis', 'is_primary': '0'},
        ],
        'keywords': ['garnet'],
        'publication_types': [{'id': 'D016428', 'name': 'Journal Article'}],
        'article_type': 'article',
    }


def test_openalex_fields_backfill_pubmed_identifiers_from_the_ids_block() -> None:
    """Recover a PMID that OpenAlex already reports rather than requesting it."""
    fields = enrichment._openalex_fields(openalex_work())

    assert fields['pmid'] == '1'
    assert enrichment._openalex_fields({})['pmid'] == ''
    assert enrichment._openalex_fields(
        {'ids': {'pmcid': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC55'}})['pmcid'] == 'PMC55'


def test_pubmed_candidates_key_rows_by_their_pubmed_identifier() -> None:
    """Read a PMID from the stored column or from a PubMed paper identifier."""
    candidates = [
        {'paper_id': 'doi:10.1234/a', 'pmid': '11'},
        {'paper_id': 'pmid:22'},
        {'paper_id': 'doi:10.1234/c'},
        {'paper_id': ''},
    ]

    assert enrichment._pubmed_candidates(candidates) == {'11': 'doi:10.1234/a', '22': 'pmid:22'}


def test_pubmed_fields_map_the_shared_enrichment_columns() -> None:
    """Contribute the bibliographic columns PubMed can fill."""
    fields = enrichment._pubmed_fields(pubmed_article())

    assert fields['doi'] == '10.1234/pubmed-one'
    assert fields['title'] == 'PubMed title'
    assert fields['journal'] == 'PubMed Journal'
    assert fields['publication_date'] == '2024-03-07'
    assert fields['pmid'] == '31234567'
    assert fields['pmcid'] == 'PMC9876543'
    assert fields['work_type'] == 'Journal Article'


def test_pubmed_subject_rows_split_descriptors_qualifiers_types_and_keywords() -> None:
    """Keep each controlled vocabulary in its own scheme and drop duplicates."""
    rows = enrichment._pubmed_subject_rows('doi:10.1234/pubmed-one', pubmed_article())

    assert [(row['scheme'], row['subject_id'], row['display_name']) for row in rows] == [
        ('mesh', 'D007854', 'Lithium'),
        ('mesh_qualifier', 'Q000032', 'analysis'),
        ('publication_type', 'D016428', 'Journal Article'),
        ('mesh_keyword', 'garnet', 'garnet'),
    ]
    assert {row['source'] for row in rows} == {'pubmed'}
    assert rows[0]['is_primary'] == 1
    assert rows[1]['is_primary'] == 0
    assert enrichment._pubmed_subject_rows('doi:10.1234/x', None) == []


def test_pubmed_keywords_use_a_scheme_distinct_from_openalex_keywords() -> None:
    """Keep both providers' keywords addressable under the composite key."""
    openalex_schemes = {row['scheme'] for row in enrichment._subject_rows('p', openalex_work())}
    pubmed_schemes = {row['scheme'] for row in enrichment._pubmed_subject_rows('p', pubmed_article())}

    assert 'keyword' in openalex_schemes
    assert 'mesh_keyword' in pubmed_schemes
    assert not openalex_schemes & pubmed_schemes


def arxiv_entry(doi: str = '10.1234/arxiv-one') -> dict[str, Any]:
    """Return an arXiv entry mapping shaped like arxiv.entry_to_paper output."""
    return {
        'paper_id': f'doi:{doi}' if doi else 'arxiv:2301.12345',
        'doi': doi,
        'arxiv_id': '2301.12345',
        'title': 'arXiv title',
        'journal': 'Phys. Rev. B',
        'journal_ref': 'Phys. Rev. B 108, 014101 (2023)',
        'publication_date': '2023-01-30',
        'authors': 'Ada Lovelace; Grace Hopper',
        'version': 'v2',
        'comment': '12 pages',
        'primary_category': 'cond-mat.mtrl-sci',
        'categories': [
            {'id': 'cond-mat.mtrl-sci', 'name': 'cond-mat.mtrl-sci', 'is_primary': True},
            {'id': 'physics.chem-ph', 'name': 'physics.chem-ph', 'is_primary': False},
        ],
    }


def test_arxiv_candidates_key_rows_by_their_arxiv_identifier() -> None:
    """Read an arXiv ID from the stored column or from an arXiv paper identifier."""
    candidates = [
        {'paper_id': 'doi:10.1234/a', 'arxiv_id': '2301.00001v3'},
        {'paper_id': 'arxiv:cond-mat/0501001'},
        {'paper_id': 'doi:10.1234/c'},
        {'paper_id': ''},
    ]

    assert enrichment._arxiv_candidates(candidates) == {
        '2301.00001': 'doi:10.1234/a',
        'cond-mat/0501001': 'arxiv:cond-mat/0501001',
    }


def test_arxiv_fields_map_the_shared_enrichment_columns() -> None:
    """Contribute the bibliographic columns arXiv can fill, plus its open access."""
    fields = enrichment._arxiv_fields(arxiv_entry())

    assert fields['doi'] == '10.1234/arxiv-one'
    assert fields['title'] == 'arXiv title'
    assert fields['journal'] == 'Phys. Rev. B'
    assert fields['publication_date'] == '2023-01-30'
    assert fields['arxiv_id'] == '2301.12345'
    assert fields['work_type'] == 'preprint'
    assert (fields['is_oa'], fields['oa_status']) == (1, 'green')


def test_arxiv_subject_rows_flag_the_primary_category() -> None:
    """Emit one row per category and mark the primary one."""
    rows = enrichment._arxiv_subject_rows('p', arxiv_entry())

    assert [(row['scheme'], row['subject_id'], row['is_primary']) for row in rows] == [
        ('arxiv_category', 'cond-mat.mtrl-sci', 1),
        ('arxiv_category', 'physics.chem-ph', 0),
    ]
    assert {row['source'] for row in rows} == {'arxiv'}
    assert enrichment._arxiv_subject_rows('p', None) == []


def test_arxiv_categories_use_a_scheme_distinct_from_the_other_providers() -> None:
    """Keep arXiv categories off the primary key OpenAlex and PubMed share."""
    openalex_schemes = {row['scheme'] for row in enrichment._subject_rows('p', openalex_work())}
    pubmed_schemes = {row['scheme'] for row in enrichment._pubmed_subject_rows('p', pubmed_article())}
    arxiv_schemes = {row['scheme'] for row in enrichment._arxiv_subject_rows('p', arxiv_entry())}

    assert arxiv_schemes == {'arxiv_category'}
    assert not arxiv_schemes & openalex_schemes
    assert not arxiv_schemes & pubmed_schemes


def test_merge_fields_ranks_arxiv_last_and_records_it_as_a_found_source() -> None:
    """Let a version-of-record value win, and fall back to arXiv when none exists."""
    update = enrichment._merge_fields('p', crossref_work(), openalex_work(),
                                     ['crossref', 'openalex', 'arxiv'],
                                     None, arxiv_entry())

    assert 'arxiv' in update['enrichment_sources'].split(';')
    assert update['arxiv_id'] == '2301.12345'
    # Crossref deposits the journal of record, so arXiv's citation string loses.
    assert update['journal'] == crossref_work()['container-title'][0]

    # With only arXiv, its values fill the row and its open access is asserted.
    solo = enrichment._merge_fields('p', None, None, ['arxiv'], None, arxiv_entry())
    assert solo['enrichment_status'] == 'succeeded'
    assert solo['enrichment_sources'] == 'arxiv'
    assert solo['journal'] == 'Phys. Rev. B'
    assert (solo['is_oa'], solo['oa_status']) == (1, 'green')
    assert solo['enrichment_json']['arxiv']['primary_category'] == 'cond-mat.mtrl-sci'


def test_enrich_batch_enriches_an_arxiv_only_row_and_stores_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a row that has only an arXiv identifier and store its categories."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'arxiv:2301.12345', 'arxiv_id': '2301.12345',
                           'title': 'Seeded'}])
    monkeypatch.setattr(enrichment.arxiv, 'fetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(enrichment.arxiv, 'parse_entries', lambda _: [arxiv_entry(doi='')])

    with open_corpus(db_path) as conn:
        candidates = corpus.enrichment_candidates(conn)
        summary = enrichment._enrich_batch(conn, candidates, ['arxiv'], '')
        subjects = conn.execute('SELECT scheme, source FROM paper_subjects').fetchall()
        rows = corpus.paper_rows(conn)

    assert summary['unresolved'] == 0
    assert summary['succeeded'] == 1
    assert {row['scheme'] for row in subjects} == {'arxiv_category'}
    assert rows[0]['arxiv_id'] == '2301.12345'
    assert rows[0]['enrichment_sources'] == 'arxiv'
    assert rows[0]['is_oa'] == 1


def test_enrich_batch_resolves_rows_only_pubmed_or_arxiv_can_reach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report neither identifier-only row as unresolved when both providers run."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [
        {'paper_id': 'pmid:31234567', 'pmid': '31234567', 'title': 'PubMed only'},
        {'paper_id': 'arxiv:2301.12345', 'arxiv_id': '2301.12345', 'title': 'arXiv only'},
        {'paper_id': 'paper:abc', 'title': 'Nothing to resolve from'},
    ])
    monkeypatch.setattr(enrichment.pubmed, 'efetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(enrichment.pubmed, 'parse_articles', lambda _: [pubmed_article(doi='')])
    monkeypatch.setattr(enrichment.arxiv, 'fetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(enrichment.arxiv, 'parse_entries', lambda _: [arxiv_entry(doi='')])

    with open_corpus(db_path) as conn:
        candidates = corpus.enrichment_candidates(conn)
        summary = enrichment._enrich_batch(conn, candidates, ['pubmed', 'arxiv'], '')

    assert summary['succeeded'] == 2
    assert summary['unresolved'] == 1


def test_enrich_batch_does_not_count_unreachable_openalex_as_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mark a DOI-less arXiv record successful when OpenAlex was never queried."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [
        {'paper_id': 'arxiv:2301.12345', 'arxiv_id': '2301.12345', 'title': 'arXiv only'},
    ])
    monkeypatch.setattr(enrichment.arxiv, 'fetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(enrichment.arxiv, 'parse_entries', lambda _: [arxiv_entry(doi='')])

    def unreachable(*_: object, **__: object) -> None:
        """Fail if OpenAlex is queried without a DOI or OpenAlex identifier."""
        raise AssertionError('OpenAlex is not reachable for this row')

    monkeypatch.setattr(enrichment.openalex, 'works_batch', unreachable)
    with open_corpus(db_path) as conn:
        summary = enrichment._enrich_batch(
            conn, corpus.enrichment_candidates(conn), ['openalex', 'arxiv'], '')
        row = corpus.paper_rows(conn)[0]

    assert summary['succeeded'] == 1
    assert summary['partial'] == 0
    assert row['enrichment_status'] == 'succeeded'


def test_enrich_batch_raises_when_the_only_provider_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep fail-fast behavior for an explicitly selected single provider."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed'}])
    monkeypatch.setattr(
        enrichment.openalex,
        'works_batch',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('OpenAlex unavailable')),
    )

    with open_corpus(db_path) as conn, pytest.raises(RuntimeError, match='OpenAlex unavailable'):
        enrichment._enrich_batch(
            conn, corpus.enrichment_candidates(conn), ['openalex'], '')


def test_enrich_batch_preserves_child_rows_from_a_failed_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not treat an outage as evidence that stored provider data vanished."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'doi': '10.1234/example', 'sources': 'seed'}])
    monkeypatch.setattr(
        enrichment.openalex,
        'works_batch',
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError('OpenAlex unavailable')),
    )

    with open_corpus(db_path) as conn:
        paper_id = corpus.paper_rows(conn)[0]['paper_id']
        corpus.write_enrichment(
            conn,
            [{**{field: '' for field in corpus.enrichment_update_fields()},
              'paper_id': paper_id, 'openalex_id': 'W123', 'cited_by_count': 42,
              'is_oa': 1, 'oa_status': 'gold', 'enrichment_status': 'pending',
              'updated_at': ''}],
            subjects=[{'paper_id': paper_id, 'scheme': 'topic', 'subject_id': 'T1',
                       'display_name': 'Stored topic', 'source': 'openalex'}],
            sources=['openalex'],
        )
        summary = enrichment._enrich_batch(
            conn, corpus.enrichment_candidates(conn), ['crossref', 'openalex'],
            'me@example.com',
            crossref_session=FakeCrossrefSession([{'items': [crossref_work()]}]), pace=0)
        subjects = conn.execute(
            'SELECT source, display_name FROM paper_subjects').fetchall()
        paper = corpus.paper_rows(conn)[0]

    assert summary['partial'] == 1
    assert [(row['source'], row['display_name']) for row in subjects] == [
        ('openalex', 'Stored topic')]
    assert paper['openalex_id'] == 'W123'
    assert paper['cited_by_count'] == 42
    assert paper['is_oa'] == 1
    assert paper['oa_status'] == 'gold'


def test_enrich_batch_keeps_other_provider_subjects_when_only_arxiv_is_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave another provider's child rows intact when enriching from arXiv."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1234/arxiv-one', 'doi': '10.1234/arxiv-one',
                           'arxiv_id': '2301.12345'}])
    monkeypatch.setattr(enrichment.arxiv, 'fetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(enrichment.arxiv, 'parse_entries', lambda _: [arxiv_entry()])

    with open_corpus(db_path) as conn:
        corpus.write_enrichment(
            conn,
            [{**{field: '' for field in corpus.enrichment_update_fields()},
              'paper_id': 'doi:10.1234/arxiv-one', 'enrichment_status': 'pending',
              'updated_at': ''}],
            subjects=[{'paper_id': 'doi:10.1234/arxiv-one', 'scheme': 'topic',
                       'subject_id': 'T1', 'display_name': 'Batteries', 'source': 'openalex'}],
            sources=['openalex'])
        enrichment._enrich_batch(conn, corpus.enrichment_candidates(conn, statuses=('pending',)),
                                ['arxiv'], '')
        sources = {row['source'] for row in
                   conn.execute('SELECT source FROM paper_subjects').fetchall()}

    assert sources == {'openalex', 'arxiv'}


def medrxiv_entry(published: str = '10.1234/medrxiv-one') -> dict[str, Any]:
    """Return a medRxiv record shaped like medrxiv.record_to_paper output."""
    return {
        'paper_id': f'doi:{published}' if published else 'doi:10.1101/2024.03.01.24303596',
        'doi': published or '10.1101/2024.03.01.24303596',
        'medrxiv_doi': '10.1101/2024.03.01.24303596',
        'title': 'medRxiv title',
        'journal': '' if published else 'medRxiv',
        'publication_date': '2024-03-04',
        'authors': 'Guangmin Xu',
        'version': '2',
        'license': 'cc_by',
        'category': 'Health Policy',
        'published_doi': published,
        'jatsxml': 'https://www.medrxiv.org/content/early/2024/03/04/x.source.xml',
        'categories': [{'id': 'health policy', 'name': 'Health Policy', 'is_primary': True}],
    }


def test_medrxiv_candidates_key_rows_by_the_doi_medrxiv_issued() -> None:
    """Read a medRxiv DOI from the stored column, the paper ID, or the DOI."""
    candidates = [
        {'paper_id': 'doi:10.1234/a', 'medrxiv_doi': '10.1101/2024.03.01.24303596v2'},
        {'paper_id': 'doi:10.64898/2026.08.05.26359794'},
        {'paper_id': 'doi:10.1234/b', 'doi': '10.1234/b'},
        {'paper_id': '', 'medrxiv_doi': '10.1101/2024.03.01.24303597'},
    ]

    assert enrichment._medrxiv_candidates(candidates) == {
        '10.1101/2024.03.01.24303596': 'doi:10.1234/a',
        '10.64898/2026.08.05.26359794': 'doi:10.64898/2026.08.05.26359794',
    }


def test_medrxiv_fields_map_the_shared_enrichment_columns() -> None:
    """Assert open access and keep the preprint DOI beside the published one."""
    fields = enrichment._medrxiv_fields(medrxiv_entry())

    assert fields['doi'] == '10.1234/medrxiv-one'
    assert fields['medrxiv_doi'] == '10.1101/2024.03.01.24303596'
    assert fields['license'] == 'cc_by'
    assert fields['is_oa'] == 1
    assert fields['oa_status'] == 'green'
    # A published paper is no longer a preprint, whatever medRxiv still hosts.
    assert fields['work_type'] == ''
    assert enrichment._medrxiv_fields(medrxiv_entry(published=''))['work_type'] == 'preprint'
    assert enrichment._medrxiv_fields(medrxiv_entry(published=''))['journal'] == 'medRxiv'


def test_medrxiv_subject_rows_use_a_scheme_distinct_from_the_other_providers() -> None:
    """Flag the single category medRxiv files a preprint under."""
    rows = enrichment._medrxiv_subject_rows('doi:10.1234/medrxiv-one', medrxiv_entry())

    assert len(rows) == 1
    assert rows[0]['scheme'] == 'medrxiv_category'
    assert rows[0]['subject_id'] == 'health policy'
    assert rows[0]['display_name'] == 'Health Policy'
    assert rows[0]['is_primary'] == 1
    assert rows[0]['source'] == 'medrxiv'
    assert enrichment._medrxiv_subject_rows('doi:x', None) == []

    schemes = {row['scheme'] for row in
               enrichment._arxiv_subject_rows('doi:x', arxiv_entry())
               + enrichment._medrxiv_subject_rows('doi:x', medrxiv_entry())}
    assert schemes == {'arxiv_category', 'medrxiv_category'}


def test_merge_fields_ranks_medrxiv_below_the_other_providers() -> None:
    """Fill only what no better-placed provider supplied for the same column."""
    update = enrichment._merge_fields('doi:10.1234/medrxiv-one', None, None,
                                     ['crossref', 'medrxiv'], None, None, medrxiv_entry())

    assert update['enrichment_status'] == 'partial'
    assert update['enrichment_sources'] == 'medrxiv'
    assert update['medrxiv_doi'] == '10.1101/2024.03.01.24303596'
    assert update['title'] == 'medRxiv title'
    assert update['is_oa'] == 1
    assert update['oa_status'] == 'green'
    # OPENALEX_ONLY blanks the licence, so medRxiv's own statement fills it.
    assert update['license'] == 'cc_by'
    assert update['enrichment_json']['medrxiv']['category'] == 'Health Policy'
    assert update['enrichment_json']['medrxiv']['published_doi'] == '10.1234/medrxiv-one'


def test_merge_fields_lets_a_better_placed_provider_win_over_medrxiv() -> None:
    """Keep Crossref's title and licence when both providers answered."""
    crossref = {'DOI': '10.1234/medrxiv-one', 'title': ['Crossref title'],
                'container-title': ['The Lancet'], 'type': 'journal-article'}
    openalex = {'id': 'https://openalex.org/W1',
                'open_access': {'is_oa': True, 'oa_status': 'gold'},
                'best_oa_location': {'license': 'cc-by-4.0'}}
    update = enrichment._merge_fields('doi:10.1234/medrxiv-one', crossref, openalex,
                                     ['crossref', 'openalex', 'medrxiv'], None, None,
                                     medrxiv_entry())

    assert update['title'] == 'Crossref title'
    assert update['journal'] == 'The Lancet'
    assert update['license'] == 'cc-by-4.0'
    assert update['oa_status'] == 'gold'
    # medRxiv still contributes the identifier none of the others carry.
    assert update['medrxiv_doi'] == '10.1101/2024.03.01.24303596'
    assert update['enrichment_sources'] == 'crossref;openalex;medrxiv'


def test_enrich_batch_enriches_a_medrxiv_only_row_and_stores_its_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a row that has only a medRxiv DOI and store its subject row."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1101/2024.03.01.24303596',
                           'medrxiv_doi': '10.1101/2024.03.01.24303596', 'title': 'Seeded'}])
    monkeypatch.setattr(enrichment.medrxiv, 'fetch_doi',
                        lambda *_, **__: medrxiv_entry(published=''))

    with open_corpus(db_path) as conn:
        candidates = corpus.enrichment_candidates(conn)
        summary = enrichment._enrich_batch(conn, candidates, ['medrxiv'], '')
        subjects = conn.execute('SELECT scheme, source FROM paper_subjects').fetchall()
        rows = corpus.paper_rows(conn)

    assert summary['unresolved'] == 0
    assert summary['succeeded'] == 1
    assert {row['scheme'] for row in subjects} == {'medrxiv_category'}
    assert rows[0]['medrxiv_doi'] == '10.1101/2024.03.01.24303596'
    assert rows[0]['enrichment_sources'] == 'medrxiv'
    assert rows[0]['is_oa'] == 1


def test_enrich_batch_skips_medrxiv_for_a_row_carrying_only_a_published_doi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spend no request on a DOI medRxiv has no way to look up."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1234/journal-only', 'doi': '10.1234/journal-only'}])

    def unreachable(*_: object, **__: object) -> None:
        """Fail the test if medRxiv is queried for a non-medRxiv DOI."""
        raise AssertionError('medRxiv should not be queried for a published DOI')

    monkeypatch.setattr(enrichment.medrxiv, 'fetch_doi', unreachable)

    with open_corpus(db_path) as conn:
        summary = enrichment._enrich_batch(conn, corpus.enrichment_candidates(conn),
                                          ['medrxiv'], '')

    assert summary['not_found'] == 1


def test_enrich_batch_keeps_other_provider_subjects_when_only_medrxiv_is_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave another provider's child rows intact when enriching from medRxiv."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1234/medrxiv-one', 'doi': '10.1234/medrxiv-one',
                           'medrxiv_doi': '10.1101/2024.03.01.24303596'}])
    monkeypatch.setattr(enrichment.medrxiv, 'fetch_doi', lambda *_, **__: medrxiv_entry())

    with open_corpus(db_path) as conn:
        corpus.write_enrichment(
            conn,
            [{**{field: '' for field in corpus.enrichment_update_fields()},
              'paper_id': 'doi:10.1234/medrxiv-one', 'enrichment_status': 'pending',
              'updated_at': ''}],
            subjects=[{'paper_id': 'doi:10.1234/medrxiv-one', 'scheme': 'topic',
                       'subject_id': 'T1', 'display_name': 'Vaccines', 'source': 'openalex'}],
            sources=['openalex'])
        enrichment._enrich_batch(conn, corpus.enrichment_candidates(conn, statuses=('pending',)),
                                ['medrxiv'], '')
        sources = {row['source'] for row in
                   conn.execute('SELECT source FROM paper_subjects').fetchall()}

    assert sources == {'openalex', 'medrxiv'}


def biorxiv_entry(published: str = '10.1234/biorxiv-one') -> dict[str, Any]:
    """Return a bioRxiv record shaped like biorxiv.record_to_paper output."""
    return {
        'paper_id': f'doi:{published}' if published else 'doi:10.1101/2023.12.01.569634',
        'doi': published or '10.1101/2023.12.01.569634',
        'biorxiv_doi': '10.1101/2023.12.01.569634',
        'title': 'bioRxiv title',
        'journal': '' if published else 'bioRxiv',
        'publication_date': '2023-12-03',
        'authors': 'Clayton Curtis',
        'version': '2',
        'license': 'cc_by_nc_nd',
        'category': 'Neuroscience',
        'published_doi': published,
        'jatsxml': 'https://www.biorxiv.org/content/early/2023/12/03/x.source.xml',
        'categories': [{'id': 'neuroscience', 'name': 'Neuroscience', 'is_primary': True}],
    }


def test_biorxiv_candidates_key_rows_by_the_doi_biorxiv_issued() -> None:
    """Read a bioRxiv DOI from the stored column, the paper ID, or the DOI."""
    candidates = [
        {'paper_id': 'doi:10.1234/a', 'biorxiv_doi': '10.1101/2023.12.01.569634v2'},
        {'paper_id': 'doi:10.64898/2026.08.07.742070'},
        {'paper_id': 'doi:10.1234/b', 'doi': '10.1234/b'},
        {'paper_id': '', 'biorxiv_doi': '10.1101/2023.12.01.569635'},
        # A medRxiv row is not a bioRxiv row, whichever column carries its DOI.
        {'paper_id': 'doi:10.1234/c', 'medrxiv_doi': '10.1101/2024.03.01.24303596'},
    ]

    assert enrichment._biorxiv_candidates(candidates) == {
        '10.1101/2023.12.01.569634': 'doi:10.1234/a',
        '10.64898/2026.08.07.742070': 'doi:10.64898/2026.08.07.742070',
    }


def test_biorxiv_fields_map_the_shared_enrichment_columns() -> None:
    """Assert open access and keep the preprint DOI beside the published one."""
    fields = enrichment._biorxiv_fields(biorxiv_entry())

    assert fields['doi'] == '10.1234/biorxiv-one'
    assert fields['biorxiv_doi'] == '10.1101/2023.12.01.569634'
    assert fields['license'] == 'cc_by_nc_nd'
    assert fields['is_oa'] == 1
    assert fields['oa_status'] == 'green'
    # A published paper is no longer a preprint, whatever bioRxiv still hosts.
    assert fields['work_type'] == ''
    assert enrichment._biorxiv_fields(biorxiv_entry(published=''))['work_type'] == 'preprint'
    assert enrichment._biorxiv_fields(biorxiv_entry(published=''))['journal'] == 'bioRxiv'


def test_biorxiv_subject_rows_use_a_scheme_distinct_from_the_other_providers() -> None:
    """Flag the single category bioRxiv files a preprint under."""
    rows = enrichment._biorxiv_subject_rows('doi:10.1234/biorxiv-one', biorxiv_entry())

    assert len(rows) == 1
    assert rows[0]['scheme'] == 'biorxiv_category'
    assert rows[0]['subject_id'] == 'neuroscience'
    assert rows[0]['display_name'] == 'Neuroscience'
    assert rows[0]['is_primary'] == 1
    assert rows[0]['source'] == 'biorxiv'
    assert enrichment._biorxiv_subject_rows('doi:x', None) == []

    # The two archives classify under different lists, so their schemes must
    # differ or one would overwrite the other on the subjects primary key.
    schemes = {row['scheme'] for row in
               enrichment._arxiv_subject_rows('doi:x', arxiv_entry())
               + enrichment._medrxiv_subject_rows('doi:x', medrxiv_entry())
               + enrichment._biorxiv_subject_rows('doi:x', biorxiv_entry())}
    assert schemes == {'arxiv_category', 'medrxiv_category', 'biorxiv_category'}


def test_merge_fields_ranks_biorxiv_below_the_other_providers() -> None:
    """Fill only what no better-placed provider supplied for the same column."""
    update = enrichment._merge_fields('doi:10.1234/biorxiv-one', None, None,
                                     ['crossref', 'biorxiv'], None, None, None,
                                     biorxiv_entry())

    assert update['enrichment_status'] == 'partial'
    assert update['enrichment_sources'] == 'biorxiv'
    assert update['biorxiv_doi'] == '10.1101/2023.12.01.569634'
    assert update['title'] == 'bioRxiv title'
    assert update['is_oa'] == 1
    assert update['oa_status'] == 'green'
    # OPENALEX_ONLY blanks the licence, so bioRxiv's own statement fills it.
    assert update['license'] == 'cc_by_nc_nd'
    assert update['enrichment_json']['biorxiv']['category'] == 'Neuroscience'
    assert update['enrichment_json']['biorxiv']['published_doi'] == '10.1234/biorxiv-one'


def test_merge_fields_lets_a_better_placed_provider_win_over_biorxiv() -> None:
    """Keep Crossref's title and licence when both providers answered."""
    crossref = {'DOI': '10.1234/biorxiv-one', 'title': ['Crossref title'],
                'container-title': ['eLife'], 'type': 'journal-article'}
    openalex = {'id': 'https://openalex.org/W1',
                'open_access': {'is_oa': True, 'oa_status': 'gold'},
                'best_oa_location': {'license': 'cc-by-4.0'}}
    update = enrichment._merge_fields('doi:10.1234/biorxiv-one', crossref, openalex,
                                     ['crossref', 'openalex', 'biorxiv'], None, None, None,
                                     biorxiv_entry())

    assert update['title'] == 'Crossref title'
    assert update['journal'] == 'eLife'
    assert update['license'] == 'cc-by-4.0'
    assert update['oa_status'] == 'gold'
    # bioRxiv still contributes the identifier none of the others carry.
    assert update['biorxiv_doi'] == '10.1101/2023.12.01.569634'
    assert update['enrichment_sources'] == 'crossref;openalex;biorxiv'


def test_merge_fields_keeps_both_preprint_servers_apart_on_one_row() -> None:
    """Record each archive's DOI in its own column when a row reaches both."""
    update = enrichment._merge_fields('doi:10.1234/both', None, None,
                                     ['medrxiv', 'biorxiv'], None, None,
                                     medrxiv_entry(published='10.1234/both'),
                                     biorxiv_entry(published='10.1234/both'))

    assert update['medrxiv_doi'] == '10.1101/2024.03.01.24303596'
    assert update['biorxiv_doi'] == '10.1101/2023.12.01.569634'
    assert update['enrichment_sources'] == 'medrxiv;biorxiv'
    assert set(update['enrichment_json']) >= {'medrxiv', 'biorxiv'}


def test_enrich_batch_enriches_a_biorxiv_only_row_and_stores_its_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a row that has only a bioRxiv DOI and store its subject row."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1101/2023.12.01.569634',
                           'biorxiv_doi': '10.1101/2023.12.01.569634', 'title': 'Seeded'}])
    monkeypatch.setattr(enrichment.biorxiv, 'fetch_doi',
                        lambda *_, **__: biorxiv_entry(published=''))

    with open_corpus(db_path) as conn:
        candidates = corpus.enrichment_candidates(conn)
        summary = enrichment._enrich_batch(conn, candidates, ['biorxiv'], '')
        subjects = conn.execute('SELECT scheme, source FROM paper_subjects').fetchall()
        rows = corpus.paper_rows(conn)

    assert summary['unresolved'] == 0
    assert summary['succeeded'] == 1
    assert {row['scheme'] for row in subjects} == {'biorxiv_category'}
    assert rows[0]['biorxiv_doi'] == '10.1101/2023.12.01.569634'
    assert rows[0]['enrichment_sources'] == 'biorxiv'
    assert rows[0]['is_oa'] == 1


def test_enrich_batch_does_not_ask_either_preprint_server_for_the_others_doi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route each preprint DOI to the one archive that can answer for it.

    Both servers share a DOI prefix and one paced API, so a row misrouted to
    the wrong one costs a request per paper and returns nothing.
    """
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1101/2023.12.01.569634',
                           'biorxiv_doi': '10.1101/2023.12.01.569634'}])

    def unreachable(*_: object, **__: object) -> None:
        """Fail the test if medRxiv is queried for a bioRxiv DOI."""
        raise AssertionError('medRxiv should not be queried for a bioRxiv DOI')

    monkeypatch.setattr(enrichment.medrxiv, 'fetch_doi', unreachable)
    monkeypatch.setattr(enrichment.biorxiv, 'fetch_doi',
                        lambda *_, **__: biorxiv_entry(published=''))

    with open_corpus(db_path) as conn:
        summary = enrichment._enrich_batch(conn, corpus.enrichment_candidates(conn),
                                          ['medrxiv', 'biorxiv'], '')

    assert summary['succeeded'] == 1


def test_enrich_batch_skips_biorxiv_for_a_row_carrying_only_a_published_doi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spend no request on a DOI bioRxiv has no way to look up."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1234/journal-only', 'doi': '10.1234/journal-only'}])

    def unreachable(*_: object, **__: object) -> None:
        """Fail the test if bioRxiv is queried for a non-bioRxiv DOI."""
        raise AssertionError('bioRxiv should not be queried for a published DOI')

    monkeypatch.setattr(enrichment.biorxiv, 'fetch_doi', unreachable)

    with open_corpus(db_path) as conn:
        summary = enrichment._enrich_batch(conn, corpus.enrichment_candidates(conn),
                                          ['biorxiv'], '')

    assert summary['not_found'] == 1


def test_merge_fields_records_pubmed_as_a_found_source() -> None:
    """Count PubMed towards the enrichment status and store its provenance."""
    update = enrichment._merge_fields('doi:10.1234/pubmed-one', None, None,
                                     ['pubmed'], pubmed_article())

    assert update['enrichment_status'] == 'succeeded'
    assert update['enrichment_sources'] == 'pubmed'
    assert update['pmid'] == '31234567'
    assert update['pmcid'] == 'PMC9876543'
    assert update['enrichment_json']['pubmed']['mesh_count'] == 2

    partial = enrichment._merge_fields('doi:10.1234/pubmed-one', None, None,
                                      ['crossref', 'pubmed'], pubmed_article())
    assert partial['enrichment_status'] == 'partial'


def test_enrich_batch_enriches_a_pubmed_only_row_and_stores_mesh(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve a row that has only a PMID and store its MeSH terms."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'pmid:31234567', 'pmid': '31234567', 'title': 'Seeded'}])
    monkeypatch.setattr(enrichment.pubmed, 'efetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(enrichment.pubmed, 'parse_articles', lambda _: [pubmed_article(doi='')])

    with open_corpus(db_path) as conn:
        candidates = corpus.enrichment_candidates(conn)
        summary = enrichment._enrich_batch(conn, candidates, ['pubmed'], '')
        subjects = conn.execute('SELECT scheme, source FROM paper_subjects').fetchall()
        rows = corpus.paper_rows(conn)

    assert summary['unresolved'] == 0
    assert summary['succeeded'] == 1
    assert {row['scheme'] for row in subjects} == {'mesh', 'mesh_qualifier',
                                                   'publication_type', 'mesh_keyword'}
    assert rows[0]['pmcid'] == 'PMC9876543'
    assert rows[0]['enrichment_sources'] == 'pubmed'


def test_enrich_batch_keeps_openalex_subjects_when_only_pubmed_is_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave another provider's child rows intact when enriching from PubMed."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1234/pubmed-one', 'doi': '10.1234/pubmed-one',
                           'pmid': '31234567'}])
    monkeypatch.setattr(enrichment.pubmed, 'efetch_ids', lambda *_, **__: object())
    monkeypatch.setattr(enrichment.pubmed, 'parse_articles', lambda _: [pubmed_article()])

    with open_corpus(db_path) as conn:
        corpus.write_enrichment(
            conn,
            [{**{field: '' for field in corpus.enrichment_update_fields()},
              'paper_id': 'doi:10.1234/pubmed-one', 'enrichment_status': 'pending',
              'updated_at': ''}],
            subjects=[{'paper_id': 'doi:10.1234/pubmed-one', 'scheme': 'topic',
                       'subject_id': 'T1', 'display_name': 'Batteries', 'source': 'openalex'}],
            sources=['openalex'])
        enrichment._enrich_batch(conn, corpus.enrichment_candidates(conn, statuses=('pending',)),
                                ['pubmed'], '')
        sources = {row['source'] for row in
                   conn.execute('SELECT source FROM paper_subjects').fetchall()}

    assert sources == {'openalex', 'pubmed'}


def chemrxiv_entry(published: str = '10.1234/chemrxiv-one') -> dict[str, Any]:
    """Return a chemRxiv record shaped like chemrxiv.record_to_paper output."""
    return {
        'paper_id': f'doi:{published}' if published else 'doi:10.26434/chemrxiv.15007737/v1',
        'doi': published or '10.26434/chemrxiv.15007737/v1',
        'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1',
        'chemrxiv_stem': '10.26434/chemrxiv.15007737',
        'title': 'chemRxiv title',
        'journal': '' if published else 'chemRxiv',
        'publication_date': '2026-08-01',
        'authors': 'Grace Hopper',
        'version': '1',
        'license': 'CC BY 4.0',
        'category': 'Catalysis',
        'categories': [{'id': 'cat-catalysis', 'name': 'Catalysis', 'is_primary': True},
                       {'id': 'cat-organic', 'name': 'Organic Chemistry', 'is_primary': False}],
        'keywords': ['photocatalysis'],
        'published_doi': published,
        'asset_url': 'https://chemrxiv.org/engage/assets/old.pdf',
    }


def test_chemrxiv_candidates_key_rows_by_the_doi_chemrxiv_issued() -> None:
    """Index candidate rows by their chemRxiv DOI, version suffix included."""
    by_chemrxiv = enrichment._chemrxiv_candidates([
        {'paper_id': 'doi:a', 'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1'},
        {'paper_id': 'doi:b', 'doi': '10.1234/journal-only'},
        {'paper_id': '', 'chemrxiv_doi': '10.26434/chemrxiv-2022-w08rh'},
    ])

    assert by_chemrxiv == {'10.26434/chemrxiv.15007737/v1': 'doi:a'}


def test_chemrxiv_fields_map_the_shared_enrichment_columns() -> None:
    """Map a chemRxiv record onto the columns enrichment merges."""
    fields = enrichment._chemrxiv_fields(chemrxiv_entry())

    assert fields['doi'] == '10.1234/chemrxiv-one'
    assert fields['chemrxiv_doi'] == '10.26434/chemrxiv.15007737/v1'
    assert fields['work_type'] == ''
    assert fields['is_oa'] == 1
    assert fields['oa_status'] == 'green'
    assert fields['license'] == 'CC BY 4.0'

    unpublished = enrichment._chemrxiv_fields(chemrxiv_entry(published=''))
    assert unpublished['work_type'] == 'preprint'
    assert unpublished['doi'] == '10.26434/chemrxiv.15007737/v1'
    assert unpublished['journal'] == 'chemRxiv'


def test_chemrxiv_subject_rows_use_schemes_distinct_from_every_provider() -> None:
    """Emit a row per category plus the keywords, under their own schemes."""
    rows = enrichment._chemrxiv_subject_rows('doi:10.1234/chemrxiv-one', chemrxiv_entry())

    categories = [row for row in rows if row['scheme'] == 'chemrxiv_category']
    keywords = [row for row in rows if row['scheme'] == 'chemrxiv_keyword']
    assert [row['display_name'] for row in categories] == ['Catalysis', 'Organic Chemistry']
    assert [row['is_primary'] for row in categories] == [1, 0]
    assert [row['subject_id'] for row in keywords] == ['photocatalysis']
    assert {row['source'] for row in rows} == {'chemrxiv'}
    assert enrichment._chemrxiv_subject_rows('doi:x', None) == []

    # Every provider classifies under its own list, so the schemes must stay
    # disjoint or one would overwrite another on the subjects primary key.
    schemes = {row['scheme'] for row in
               enrichment._arxiv_subject_rows('doi:x', arxiv_entry())
               + enrichment._medrxiv_subject_rows('doi:x', medrxiv_entry())
               + enrichment._biorxiv_subject_rows('doi:x', biorxiv_entry())
               + enrichment._chemrxiv_subject_rows('doi:x', chemrxiv_entry())}
    assert schemes == {'arxiv_category', 'medrxiv_category', 'biorxiv_category',
                       'chemrxiv_category', 'chemrxiv_keyword'}


def test_merge_fields_ranks_chemrxiv_below_the_other_providers() -> None:
    """Let chemRxiv fill a column no better-placed provider supplied."""
    update = enrichment._merge_fields('doi:10.1234/chemrxiv-one', None, None, ['chemrxiv'],
                                     None, None, None, None, chemrxiv_entry())

    assert update['title'] == 'chemRxiv title'
    assert update['chemrxiv_doi'] == '10.26434/chemrxiv.15007737/v1'
    assert update['is_oa'] == 1
    assert update['oa_status'] == 'green'
    assert update['license'] == 'CC BY 4.0'
    assert update['enrichment_sources'] == 'chemrxiv'


def test_merge_fields_lets_a_better_placed_provider_win_over_chemrxiv() -> None:
    """Prefer bioRxiv's values but keep the chemRxiv identifier alongside."""
    update = enrichment._merge_fields('doi:x', None, None, ['biorxiv', 'chemrxiv'],
                                     None, None, None, biorxiv_entry(published=''),
                                     chemrxiv_entry(published=''))

    assert update['title'] == 'bioRxiv title'
    assert update['biorxiv_doi'] == '10.1101/2023.12.01.569634'
    assert update['chemrxiv_doi'] == '10.26434/chemrxiv.15007737/v1'
    assert update['enrichment_sources'] == 'biorxiv;chemrxiv'


def test_enrich_batch_enriches_a_chemrxiv_only_row_and_stores_its_terms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a row that has only a chemRxiv DOI and store its subject rows."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.26434/chemrxiv.15007737/v1',
                           'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1',
                           'title': 'Seeded'}])
    monkeypatch.setattr(enrichment.chemrxiv, 'fetch_doi',
                        lambda *_, **__: chemrxiv_entry(published=''))

    with open_corpus(db_path) as conn:
        candidates = corpus.enrichment_candidates(conn)
        summary = enrichment._enrich_batch(conn, candidates, ['chemrxiv'], '')
        subjects = conn.execute('SELECT scheme, source FROM paper_subjects').fetchall()
        rows = corpus.paper_rows(conn)

    assert summary['unresolved'] == 0
    assert summary['succeeded'] == 1
    assert {row['scheme'] for row in subjects} == {'chemrxiv_category', 'chemrxiv_keyword'}
    # The version suffix is part of the registered DOI and must survive a round trip.
    assert rows[0]['chemrxiv_doi'] == '10.26434/chemrxiv.15007737/v1'
    assert rows[0]['enrichment_sources'] == 'chemrxiv'
    assert rows[0]['is_oa'] == 1


def test_enrich_batch_does_not_ask_chemrxiv_for_another_archives_doi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spend no chemRxiv request on a bioRxiv row, or the reverse."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1101/2023.12.01.569634',
                           'biorxiv_doi': '10.1101/2023.12.01.569634', 'title': 'Seeded'}])

    def unreachable(*_: Any, **__: Any) -> None:
        """Fail the test if chemRxiv is queried for another archive's row."""
        raise AssertionError('chemRxiv was asked for a DOI it does not hold')

    monkeypatch.setattr(enrichment.chemrxiv, 'fetch_doi', unreachable)
    monkeypatch.setattr(enrichment.biorxiv, 'fetch_doi',
                        lambda *_, **__: biorxiv_entry(published=''))

    with open_corpus(db_path) as conn:
        candidates = corpus.enrichment_candidates(conn)
        summary = enrichment._enrich_batch(conn, candidates, ['biorxiv', 'chemrxiv'], '')

    assert summary['succeeded'] == 1


def test_enrich_batch_skips_chemrxiv_for_a_row_carrying_only_a_published_doi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave a journal-only row to the providers that can answer for it."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'doi:10.1234/journal-only',
                           'doi': '10.1234/journal-only', 'title': 'Seeded'}])

    def unreachable(*_: Any, **__: Any) -> None:
        """Fail the test if chemRxiv is queried for a published DOI."""
        raise AssertionError('chemRxiv was asked for a published DOI')

    monkeypatch.setattr(enrichment.chemrxiv, 'fetch_doi', unreachable)

    with open_corpus(db_path) as conn:
        candidates = corpus.enrichment_candidates(conn)
        enrichment._enrich_batch(conn, candidates, ['chemrxiv'], '')


def test_enrichment_helpers_ignore_blank_and_duplicate_subject_data(
    tmp_path: Path,
) -> None:
    """Skip unusable identifiers while retaining unique provider subjects."""
    assert enrichment._partition_candidates([{}, {'paper_id': ''}]) == ({}, {}, [])
    assert enrichment._pubmed_subject_rows('p', {'mesh': [{'id': ''}, {'id': ''}]}) == []
    assert enrichment._arxiv_subject_rows('p', {'categories': [{'id': ''}]}) == []
    assert enrichment._medrxiv_subject_rows('p', {'categories': [{'id': ''}]}) == []
    assert enrichment._biorxiv_subject_rows('p', {'categories': [{'id': ''}]}) == []
    assert enrichment._chemrxiv_subject_rows(
        'p', {'categories': [{'id': ''}], 'keywords': ['']}
    ) == []
    subjects = enrichment._subject_rows('p', {
        'primary_topic': {},
        'topics': [
            {'id': 'https://openalex.org/T1', 'display_name': 'One'},
            {'id': 'https://openalex.org/T1', 'display_name': 'Duplicate'},
            {},
        ],
    })
    assert len(subjects) == 1

    context = enrichment._FetchContext(
        dois=[], identifiers=[], email='', api_key=None,
        openalex_session=None, crossref_session=None, pace=0,
    )
    assert enrichment._fetch_crossref(context) == {}
    assert enrichment._json_text('already-json') == 'already-json'
    with corpus.connect(tmp_path / 'empty.db') as conn:
        assert enrichment._enrich_batch(conn, [], ['crossref'], '')['succeeded'] == 0
        enrichment._enrich_from_crossref_message(conn, '', {})


def test_enrichment_entry_points_use_optional_pubmed_identity_and_reference_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve optional NCBI settings and run reference linking on request."""
    db_path = tmp_path / 'papers.db'
    seed_corpus(db_path, [{'paper_id': 'p', 'title': 'Paper'}])
    monkeypatch.setattr(enrichment.pubmed, 'configured_email', lambda: 'ncbi@example.org')
    monkeypatch.setattr(enrichment.pubmed, 'configured_api_key', lambda: 'ncbi-key')
    monkeypatch.setattr(enrichment.openalex, 'configured_api_key', lambda: '')
    with open_corpus(db_path) as conn:
        summary = enrichment.enrich_papers(conn, [], sources=['pubmed'])
    assert summary['succeeded'] == 0
    assert enrichment._selected_statuses(False, True) == ('pending', 'failed')

    resolved = []
    monkeypatch.setattr(enrichment, '_resolve_reference_targets', lambda conn: resolved.append(True))
    enrichment.enrich_corpus(
        db_path, sources=['pubmed'], resolve_references=True,
        email='', pubmed_api_key=None,
    )
    assert resolved == [True]

    with open_corpus(db_path) as conn:
        conn.execute(
            "INSERT INTO paper_references (paper_id, source, reference_rank, referenced_openalex_id) "
            "VALUES ('p', 'openalex', 0, 'W404')"
        )
        conn.commit()
    monkeypatch.setattr(enrichment.openalex, 'works_batch', lambda *args, **kwargs: {})
    assert enrichment.resolve_reference_dois(db_path) == 0
