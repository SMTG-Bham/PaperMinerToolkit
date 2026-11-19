"""Tests for storing search results in the corpus.

A search merges every provider result into the corpus, stores its abstract, and
links it to the search run. These tests check that the indexed store makes the
same merges as the corpus scan it replaced, writes in one transaction, and does
an amount of work that grows linearly with the number of results.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import paperminertoolkit.corpus.database as corpus
import paperminertoolkit.workflows.search as search
from tests.corpus_helpers import ScanIndex, corpus_snapshot, open_corpus


EXISTING_PAPER = {
    'paper_id': 'old:1',
    'title': 'Existing Paper',
    'publication_date': '2018-06-01',
    'sources': 'old',
}


def duplicate_results() -> dict[str, pd.DataFrame]:
    """Return two providers' results with duplicates within and across them.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Search rows for ``openalex`` and ``pubmed``. They repeat papers by
        identifier, by DOI written several ways, by PubMed ID, and by title and
        year, including a stored paper and an abstract shared by two records.
    """
    return {
        'openalex': search._paper_rows([
            {'paper_id': 'openalex:W1', 'doi': '10.1/A', 'title': 'First paper',
             'publication_date': '2019-01-01', 'sources': 'openalex',
             'abstract': '<p>First abstract</p>'},
            {'paper_id': 'openalex:W2', 'doi': '10.1/b', 'title': 'Graphene: a review',
             'publication_date': '2019-05-01', 'sources': 'openalex',
             'abstract': 'Graphene abstract'},
            {'paper_id': 'openalex:W3', 'title': 'Second paper', 'publication_date': '2021',
             'pmid': '77', 'sources': 'openalex', 'abstract': ''},
            {'paper_id': 'openalex:W1', 'doi': '10.1/a', 'sources': 'openalex',
             'abstract': 'Repeated abstract'},
            {'paper_id': 'openalex:W4', 'title': 'Existing paper', 'publication_date': '2018',
             'sources': 'openalex', 'abstract': 'Abstract for existing'},
        ]),
        'pubmed': search._paper_rows([
            {'paper_id': 'doi:10.1/a', 'doi': 'https://doi.org/10.1/A', 'pmid': '1',
             'journal': 'Journal One', 'sources': 'pubmed', 'abstract': 'PubMed abstract'},
            {'paper_id': 'pmid:77', 'pmid': '77', 'title': 'Second paper',
             'publication_date': '2021-02-02', 'sources': 'pubmed',
             'abstract': 'Second abstract'},
            {'paper_id': 'pmid:88', 'pmid': '88', 'doi': 'DOI:10.1/B', 'sources': 'pubmed',
             'abstract': 'Graphene abstract'},
            {'paper_id': 'pmid:99', 'pmid': '99', 'title': 'Unrelated',
             'publication_date': '2020', 'sources': 'pubmed'},
        ]),
    }


def stub_providers(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]) -> None:
    """Make each provider search return a fixed frame.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the provider lookup.
    frames : dict[str, pandas.DataFrame]
        Search rows to return, keyed by provider name.
    """
    def provider_search(name: str) -> Any:
        """Return a search function for one provider.

        Parameters
        ----------
        name : str
            Provider name.

        Returns
        -------
        Any
            Function answering any query with that provider's frame.
        """
        def run(query: str, count: int) -> pd.DataFrame:
            """Return a copy of the provider's fixed frame.

            Parameters
            ----------
            query : str
                Ignored search expression.
            count : int
                Ignored result limit.

            Returns
            -------
            pandas.DataFrame
                The provider's search rows.
            """
            return frames[name].copy()

        return run

    monkeypatch.setattr(search, '_source_search', provider_search)


def search_into_seeded_corpus(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Search the duplicate results into a corpus that already holds one paper.

    Parameters
    ----------
    db_path : pathlib.Path
        New corpus database to create.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub the providers.

    Returns
    -------
    dict[str, Any]
        Search summary returned by ``search_for_papers``.
    """
    with open_corpus(db_path) as conn:
        corpus.upsert_paper(conn, EXISTING_PAPER)
    stub_providers(monkeypatch, duplicate_results())
    return search.search_for_papers('query', db_path=str(db_path),
                                    source=['openalex', 'pubmed'], count=10,
                                    store_abstract=True)


def test_search_store_makes_the_same_merges_as_the_corpus_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store, merge, and link every result exactly as scanning the corpus did."""
    indexed = search_into_seeded_corpus(tmp_path / 'indexed.db', monkeypatch)
    monkeypatch.setattr(search, 'PaperIndex', ScanIndex)
    scanned = search_into_seeded_corpus(tmp_path / 'scanned.db', monkeypatch)
    with open_corpus(tmp_path / 'indexed.db') as conn:
        indexed_rows = corpus_snapshot(conn)
        papers = corpus.paper_rows(conn)
        links = {paper['paper_id']: corpus.paper_search_history(conn, paper['paper_id'])
                 for paper in papers}
        abstracts = {paper['paper_id']: corpus.get_asset(conn, paper['paper_id'], 'abstract')
                     for paper in papers}
    with open_corpus(tmp_path / 'scanned.db') as conn:
        scanned_rows = corpus_snapshot(conn)

    assert indexed == scanned
    assert indexed_rows == scanned_rows
    assert (indexed['papers_added'], indexed['papers_updated']) == (4, 5)
    assert indexed['abstracts_stored'] == 7
    assert indexed['paper_ids'] == ['openalex:W1', 'openalex:W2', 'openalex:W3', 'old:1', 'pmid:99']
    assert [paper['paper_id'] for paper in papers] == [
        'old:1', 'openalex:W1', 'openalex:W2', 'openalex:W3', 'pmid:99',
    ]
    by_id = {paper['paper_id']: paper for paper in papers}
    assert by_id['openalex:W1']['pmid'] == '1'
    assert by_id['openalex:W1']['journal'] == 'Journal One'
    assert by_id['openalex:W2']['pmid'] == '88'
    assert by_id['old:1']['sources'] == 'old;openalex'
    assert by_id['old:1']['abstract_source'] == 'openalex'
    assert by_id['openalex:W3']['abstract_source'] == 'pubmed'
    assert [(link['source'], link['result_rank']) for link in links['openalex:W1']] == [
        ('openalex', 0), ('pubmed', 0),
    ]
    assert [(link['source'], link['result_rank']) for link in links['openalex:W2']] == [
        ('openalex', 1), ('pubmed', 2),
    ]
    assert abstracts['old:1']['content'] == b'Abstract for existing'
    assert abstracts['pmid:99'] is None
    assert len(indexed_rows['blobs']) == 6


def test_a_failed_search_store_leaves_no_partial_papers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Roll back the merged papers when storing their abstracts fails part-way."""
    db_path = tmp_path / 'papers.db'
    stub_providers(monkeypatch, duplicate_results())
    stored = []
    real_add_asset = search.add_asset

    def fail_on_second_asset(*args: Any, **kwargs: Any) -> str:
        """Store one abstract, then fail.

        Parameters
        ----------
        *args : Any
            Positional arguments for ``add_asset``.
        **kwargs : Any
            Keyword arguments for ``add_asset``.

        Returns
        -------
        str
            Identifier of the first stored blob.

        Raises
        ------
        RuntimeError
            On the second abstract.
        """
        if stored:
            raise RuntimeError('disk full')
        stored.append(real_add_asset(*args, **kwargs))
        return stored[-1]

    monkeypatch.setattr(search, 'add_asset', fail_on_second_asset)

    with pytest.raises(RuntimeError, match='disk full'):
        search.search_for_papers('query', db_path=str(db_path), source=['openalex', 'pubmed'],
                                 count=10, store_abstract=True)

    with open_corpus(db_path) as conn:
        rows = corpus_snapshot(conn)
        run = corpus.search_history(conn)[0]
    assert rows == {'papers': [], 'paper_assets': [], 'blobs': [], 'paper_search_results': []}
    assert run['status'] == 'failed'
    assert run['source_results']['corpus']['error'] == 'disk full'


def test_search_store_commits_before_enrichment_and_reindexes_after_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let enrichment open its own transaction, then resolve results against its writes."""
    db_path = tmp_path / 'papers.db'
    with open_corpus(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'old:1', 'title': 'Unrelated', 'sources': 'old'})
    stub_providers(monkeypatch, {'openalex': search._paper_rows([{
        'paper_id': 'openalex:W9', 'doi': '10.9/x', 'title': 'Different',
        'sources': 'openalex', 'abstract': 'An abstract',
    }])})

    def enrich_older_row(conn: sqlite3.Connection, records: Any, **kwargs: Any) -> dict[str, int]:
        """Give the older row the new result's DOI in an explicit transaction.

        Parameters
        ----------
        conn : sqlite3.Connection
            Search connection.
        records : Any
            Search records, unused.
        **kwargs : Any
            Enrichment options, unused.

        Returns
        -------
        dict[str, int]
            Enrichment summary reporting one success.
        """
        conn.execute('BEGIN')
        conn.execute("UPDATE papers SET doi = '10.9/x' WHERE paper_id = 'old:1'")
        conn.commit()
        return {'succeeded': 1}

    monkeypatch.setattr(search, 'enrich_papers', enrich_older_row)

    summary = search.search_for_papers('query', db_path=str(db_path), source='openalex',
                                       count=1, store_abstract=True, enrich=True)

    with open_corpus(db_path) as conn:
        old_links = corpus.paper_search_history(conn, 'old:1')
        new_links = corpus.paper_search_history(conn, 'openalex:W9')
        abstract = corpus.get_asset(conn, 'openalex:W9', 'abstract')
    assert summary['status'] == 'completed'
    assert summary['paper_ids'] == ['old:1']
    assert [link['source'] for link in old_links] == ['openalex']
    assert new_links == []
    assert abstract['content'] == b'An abstract'


def distinct_results(count: int) -> pd.DataFrame:
    """Return search rows for distinct papers, each with an abstract.

    Parameters
    ----------
    count : int
        Number of papers.

    Returns
    -------
    pandas.DataFrame
        OpenAlex-style search rows with no duplicates.
    """
    return search._paper_rows(
        {'paper_id': f'openalex:W{number}', 'doi': f'10.5555/{number}',
         'title': f'Paper number {number}', 'publication_date': f'{2000 + number % 25}-01-01',
         'sources': 'openalex', 'abstract': f'Abstract of paper {number}.'}
        for number in range(count)
    )


def measure_search_store(db_path: Path, monkeypatch: pytest.MonkeyPatch, count: int) -> dict[str, int]:
    """Search distinct papers into a new corpus and count the work it takes.

    The counts come from SQLite and from the matching code rather than from a
    clock, so they are the same on every run and machine.

    Parameters
    ----------
    db_path : pathlib.Path
        New corpus database to create.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to stub providers and instrument connections.
    count : int
        Number of distinct search results.

    Returns
    -------
    dict[str, int]
        ``steps``, thousands of SQLite virtual-machine instructions run;
        ``commits``; ``scans``, statements reading the whole ``papers`` table;
        and ``comparisons``, rows compared one by one by ``_papers_match``.
    """
    work = {'steps': 0, 'commits': 0, 'scans': 0, 'comparisons': 0}
    whole_table = re.compile(r'\bFROM\s+papers\b(?!\s+WHERE)')
    real_papers_match = corpus._papers_match

    def tick() -> int:
        """Count a thousand instructions and let the statement continue.

        Returns
        -------
        int
            Zero, so SQLite does not interrupt the statement.
        """
        work['steps'] += 1
        return 0

    def trace(statement: str) -> None:
        """Count commits and statements that read every stored paper.

        Parameters
        ----------
        statement : str
            SQL statement SQLite is about to run.
        """
        if statement.strip().upper() == 'COMMIT':
            work['commits'] += 1
        elif whole_table.search(statement):
            work['scans'] += 1

    def counting_connect(path: str) -> sqlite3.Connection:
        """Open a corpus connection that reports its work.

        Parameters
        ----------
        path : str
            Corpus database path.

        Returns
        -------
        sqlite3.Connection
            Instrumented corpus connection.
        """
        conn = corpus.connect(path)
        conn.set_progress_handler(tick, 1000)
        conn.set_trace_callback(trace)
        return conn

    def counting_papers_match(existing: Any, incoming: Any) -> bool:
        """Count one pairwise row comparison.

        Parameters
        ----------
        existing : Any
            Stored paper row.
        incoming : Any
            Incoming paper row.

        Returns
        -------
        bool
            Whether the rows match.
        """
        work['comparisons'] += 1
        return real_papers_match(existing, incoming)

    stub_providers(monkeypatch, {'openalex': distinct_results(count)})
    monkeypatch.setattr(search, 'connect', counting_connect)
    monkeypatch.setattr(corpus, '_papers_match', counting_papers_match)
    summary = search.search_for_papers('query', db_path=str(db_path), source='openalex',
                                       count=count, store_abstract=True)
    monkeypatch.setattr(corpus, '_papers_match', real_papers_match)
    assert (summary['papers_added'], summary['abstracts_stored']) == (count, count)
    return work


def test_search_store_work_grows_linearly_with_the_result_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do four times the work for four times the results, in a fixed number of commits."""
    small = measure_search_store(tmp_path / 'small.db', monkeypatch, 80)
    large = measure_search_store(tmp_path / 'large.db', monkeypatch, 320)

    assert small['comparisons'] == large['comparisons'] == 0
    assert small['scans'] == large['scans'] == 1
    assert small['commits'] == large['commits']
    assert large['steps'] < 5 * small['steps']


@pytest.mark.slow
def test_search_store_work_measure_detects_a_scanning_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Show that the work measure grows quadratically when every lookup scans.

    This checks the measure behind the linear-growth test rather than the
    store, and scanning makes it slow, so it runs only with ``-m slow``.
    """
    monkeypatch.setattr(search, 'PaperIndex', ScanIndex)
    small = measure_search_store(tmp_path / 'small.db', monkeypatch, 80)
    large = measure_search_store(tmp_path / 'large.db', monkeypatch, 320)

    assert large['comparisons'] > 12 * small['comparisons']
    assert large['steps'] > 6 * small['steps']
