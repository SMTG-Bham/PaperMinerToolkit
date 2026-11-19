"""Unit tests for the corpus paper index used by bulk stores.

The index replaces a scan of every stored row per lookup, so these tests hold it
to the scan's answers: the same matches under every duplicate rule, the same
earliest row when several match, and the same merges in ``upsert_papers``.
"""

from __future__ import annotations

import itertools
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import paperminertoolkit.corpus.database as corpus
from tests.corpus_helpers import ScanIndex, corpus_snapshot, open_corpus


# Rows that exercise each rule of ``_papers_match`` and the normalization behind it:
# DOI prefixes, case, and trailing dots; identifiers given as numbers, padded, or
# missing; punctuated, accented, and dateless titles; and non-ASCII digits in years.
TRICKY_ROWS: list[dict[str, Any]] = [
    {},
    {'paper_id': 'doi:10.1/abc', 'doi': '10.1/ABC'},
    {'paper_id': 'a', 'doi': 'https://doi.org/10.1/abc.'},
    {'paper_id': 'b', 'doi': 'doi:https://doi.org/10.1/abc'},
    {'paper_id': 'c', 'doi': ' DOI:10.1/abc '},
    {'paper_id': 'd', 'doi': 'HTTPS://DOI.ORG/10.1/ABC'},
    {'paper_id': 'e', 'doi': 'https://doi.org/doi:10.1/abc'},
    {'paper_id': 'f', 'doi': '   ', 'pmid': 12345},
    {'paper_id': 'g', 'doi': float('nan'), 'pmid': '12345'},
    {'paper_id': 'h', 'pmid': ' 12345'},
    {'paper_id': 'i', 'pmcid': 'PMC1', 'arxiv_id': '2101.00001'},
    {'paper_id': 'j', 'arxiv_id': '2101.00001v2', 'pmcid': 'pmc1'},
    {'paper_id': 'k', 'core_id': 'core-9', 'medrxiv_doi': '10.1101/2020.01.01.1'},
    {'paper_id': 'l', 'biorxiv_doi': '10.1101/2020.01.01.1', 'chemrxiv_doi': '10.26434/x'},
    {'paper_id': 'core-9', 'chemrxiv_doi': '10.26434/X'},
    {'paper_id': 'm', 'title': 'Lithium–ion: Batteries!', 'publication_date': '2020-05-01'},
    {'paper_id': 'n', 'title': 'lithium ion batteries', 'publication_date': '2020'},
    {'paper_id': 'o', 'title': 'Lithium ion batteries', 'publication_date': '2021'},
    {'paper_id': 'p', 'title': 'Lithium ion batteries', 'publication_date': 'n.d.'},
    {'paper_id': 'q', 'title': '!!!', 'publication_date': '2020'},
    {'paper_id': 'r', 'title': 'Ünïcödé Tïtle',
     'publication_date': '٢٠٢٠'},
    {'paper_id': 's', 'title': 'unicode title', 'publication_date': '2020'},
    {'paper_id': 'None', 'doi': None, 'title': None},
    {'paper_id': 't', 'doi': '10.1/abc', 'title': 'Lithium ion batteries',
     'publication_date': '2021-01-01', 'pmid': '999'},
]


def test_match_keys_agree_with_papers_match_on_every_pair() -> None:
    """Share a key exactly when the pairwise rule says two rows match."""
    outcomes = []
    for existing, raw in itertools.product(TRICKY_ROWS, repeat=2):
        for incoming in (raw, corpus.normalize_paper(raw)):
            expected = corpus._papers_match(existing, incoming)
            keys = corpus._stored_match_keys(existing) & corpus._incoming_match_keys(incoming)
            assert bool(keys) == expected, (existing, incoming, keys)
            outcomes.append(expected)

    assert outcomes.count(True) > len(TRICKY_ROWS) * 2
    assert outcomes.count(False) > outcomes.count(True)


def store_without_merging(conn: sqlite3.Connection, rows: list[dict[str, Any]], suffix: str) -> None:
    """Insert rows as they are, so several stored rows can match one paper.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    rows : list[dict[str, Any]]
        Paper rows to insert, each under its own identifier.
    suffix : str
        Text appended to every identifier to keep repeated inserts distinct.
    """
    for number, row in enumerate(rows):
        paper = dict(row)
        paper['paper_id'] = f'{paper.get("paper_id", "row")}{suffix}{number}'
        corpus.upsert_paper(conn, paper)


def test_paper_index_returns_the_earliest_matching_row_like_a_scan(tmp_path: Path) -> None:
    """Pick the same stored row as the scan, including when several rows match."""
    with open_corpus(tmp_path / 'papers.db') as conn:
        store_without_merging(conn, TRICKY_ROWS, '-first-')
        store_without_merging(conn, list(reversed(TRICKY_ROWS)), '-second-')
        index = corpus.PaperIndex(conn)
        probes = TRICKY_ROWS + [
            {'paper_id': 'b-first-3'},
            {'pmid': 12345.0},
            {'title': 'LITHIUM ION BATTERIES', 'publication_date': 'May 2021'},
            {'doi': 'nothing/stored'},
        ]
        found = [index.find(probe) for probe in probes]
        scanned = [corpus._find_existing_paper(conn, probe) for probe in probes]

    assert found == scanned
    assert sum(row is not None for row in found) > len(TRICKY_ROWS) // 2
    assert found[1]['paper_id'] == 'doi:10.1/abc-first-1'
    assert found[-1] is None


def test_paper_index_returns_rows_the_caller_can_change(tmp_path: Path) -> None:
    """Return a fresh row each time, so editing one does not alter the next."""
    with open_corpus(tmp_path / 'papers.db') as conn:
        corpus.upsert_paper(conn, {'paper_id': 'p1', 'doi': '10.1/one'})
        index = corpus.PaperIndex(conn)
        first = index.find({'doi': '10.1/one'})
        first['abstract_source'] = 'edited'
        second = index.find({'doi': '10.1/one'})

    assert second is not None
    assert second['abstract_source'] == ''


def test_paper_index_follows_refreshed_rows_and_rejects_deleted_ones(tmp_path: Path) -> None:
    """Re-index a changed row on refresh and refuse to answer from a deleted one."""
    with open_corpus(tmp_path / 'papers.db') as conn:
        corpus.upsert_paper(conn, {'paper_id': 'p1', 'title': 'Old title', 'publication_date': '2020'})
        index = corpus.PaperIndex(conn)
        conn.execute("UPDATE papers SET doi = '10.1/new', title = 'New title' WHERE paper_id = 'p1'")
        stale = index.find({'doi': '10.1/new'})
        index.refresh('p1')
        index.refresh('unknown')
        by_doi = index.find({'doi': '10.1/new'})
        by_old_title = index.find({'title': 'Old title', 'publication_date': '2020'})
        conn.execute("DELETE FROM papers WHERE paper_id = 'p1'")
        with pytest.raises(RuntimeError, match='stale'):
            index.find({'doi': '10.1/new'})

    assert stale is None
    assert by_doi is not None
    assert by_doi['paper_id'] == 'p1'
    assert by_old_title is None


# A stream of provider records in which every duplicate rule fires, including a
# record that only matches through a DOI an earlier record added to its row.
DUPLICATE_STREAM: list[dict[str, Any]] = [
    {'paper_id': 'openalex:W1', 'doi': '10.1/A', 'title': 'First paper',
     'publication_date': '2019-01-01', 'sources': 'openalex'},
    {'paper_id': 'pubmed:1', 'doi': 'https://doi.org/10.1/a', 'pmid': '1',
     'journal': 'Journal One', 'sources': 'pubmed'},
    {'paper_id': 'pmid:77', 'pmid': '77', 'title': 'Second paper', 'sources': 'pubmed'},
    {'paper_id': 'arxiv:2101.0001', 'arxiv_id': '2101.0001', 'pmid': '77', 'sources': 'arxiv'},
    {'paper_id': 'core:5', 'core_id': '5', 'title': 'Graphene: a review',
     'publication_date': '2019', 'sources': 'core'},
    {'paper_id': 'openalex:W6', 'doi': '10.1/b', 'title': 'graphene a review',
     'publication_date': '2019-05-01', 'sources': 'openalex'},
    {'paper_id': 'crossref:7', 'doi': 'DOI:10.1/B', 'authors': 'A. Author', 'sources': 'crossref'},
    {'paper_id': 'core:8', 'core_id': '8', 'title': 'Graphene: a review',
     'publication_date': '2020', 'sources': 'core'},
    {'paper_id': 'openalex:W1', 'journal': 'Ignored Journal', 'sources': 'openalex'},
    {'title': 'Untitled preprint', 'authors': 'B. Author', 'sources': 'chemrxiv'},
    {'title': 'untitled preprint', 'authors': 'B. Author', 'sources': 'chemrxiv'},
]


def test_upsert_papers_merges_duplicates_as_the_scan_did(tmp_path: Path) -> None:
    """Merge by DOI, identifier, and title with year into the rows a scan chose."""
    with open_corpus(tmp_path / 'indexed.db') as conn:
        counts = corpus.upsert_papers(conn, DUPLICATE_STREAM)
        indexed = corpus_snapshot(conn)
        rows = corpus.paper_rows(conn)
    with open_corpus(tmp_path / 'scanned.db') as conn:
        scan_counts = corpus.upsert_papers(conn, DUPLICATE_STREAM, index=ScanIndex(conn))
        scanned = corpus_snapshot(conn)

    assert counts == scan_counts == (6, 5)
    assert indexed == scanned
    assert [row['paper_id'] for row in rows] == [
        'openalex:W1', 'pmid:77', 'core:5', 'core:8',
        corpus.normalize_paper(DUPLICATE_STREAM[9])['paper_id'],
        corpus.normalize_paper(DUPLICATE_STREAM[10])['paper_id'],
    ]
    assert rows[0]['journal'] == 'Journal One'
    assert rows[0]['sources'] == 'openalex;pubmed'
    assert rows[1]['arxiv_id'] == '2101.0001'
    assert rows[2]['doi'] == '10.1/b'
    assert rows[2]['authors'] == 'A. Author'
    assert rows[2]['sources'] == 'core;openalex;crossref'


def test_upsert_papers_merges_into_rows_stored_before_the_call(tmp_path: Path) -> None:
    """Index rows already in the corpus as well as rows the call writes."""
    with open_corpus(tmp_path / 'papers.db') as conn:
        corpus.upsert_paper(conn, {'paper_id': 'old:1', 'title': 'Graphene: A Review',
                                   'publication_date': '2019-12-31', 'sources': 'old'})
        counts = corpus.upsert_papers(conn, DUPLICATE_STREAM)
        rows = corpus.paper_rows(conn)

    assert counts == (5, 6)
    assert rows[0]['paper_id'] == 'old:1'
    assert rows[0]['core_id'] == '5'
    assert rows[0]['doi'] == '10.1/b'
    assert 'core:5' not in {row['paper_id'] for row in rows}


def test_upsert_papers_without_rows_commits_nothing(tmp_path: Path) -> None:
    """Leave a caller's open transaction alone when there is nothing to write."""
    with open_corpus(tmp_path / 'papers.db') as conn:
        corpus.upsert_paper(conn, {'paper_id': 'p1'}, commit=False)
        assert corpus.upsert_papers(conn, []) == (0, 0)
        assert conn.in_transaction
        corpus.upsert_papers(conn, [{'paper_id': 'p2'}], commit=False)
        assert conn.in_transaction
        corpus.upsert_papers(conn, [{'paper_id': 'p3'}])
        assert not conn.in_transaction
