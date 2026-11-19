"""Shared corpus readers and connection lifetime helpers for tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from paperminertoolkit.corpus import database as corpus


class ScanIndex:
    """Stand in for :class:`~paperminertoolkit.corpus.database.PaperIndex` with a scan.

    Every lookup scans the stored rows as matching did before the index
    existed, so a store run through this class is the reference the indexed
    store must reproduce.

    Parameters
    ----------
    conn : sqlite3.Connection
        Corpus connection to scan.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Keep the connection to scan.

        Parameters
        ----------
        conn : sqlite3.Connection
            Corpus connection to scan.
        """
        self.conn = conn

    def find(self, paper: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return the first stored row matching a paper by scanning them all.

        Parameters
        ----------
        paper : collections.abc.Mapping[str, Any]
            Paper metadata to match.

        Returns
        -------
        dict[str, Any] or None
            Matching corpus row, or ``None``.
        """
        return corpus._find_existing_paper(self.conn, paper)

    def refresh(self, paper_id: str) -> None:
        """Ignore a write, which the next scan reads anyway.

        Parameters
        ----------
        paper_id : str
            Identifier of the paper that was written.
        """


def corpus_snapshot(conn: sqlite3.Connection) -> dict[str, list[tuple[Any, ...]]]:
    """Read every stored paper, asset, blob, and search link without timestamps.

    Parameters
    ----------
    conn : sqlite3.Connection
        Corpus connection to read.

    Returns
    -------
    dict[str, list[tuple[Any, ...]]]
        Rows of each table in storage order, with ``created_at`` and
        ``updated_at`` dropped and blob content decompressed, so that two stores
        made at different times compare equal when they wrote the same data.
        Compressed bytes are not compared because gzip records the time it ran.
    """
    tables = ['papers', 'paper_assets', 'blobs', 'paper_search_results']
    snapshot = {}
    for table in tables:
        rows = []
        for row in conn.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall():
            values = {column: row[column] for column in row.keys()
                      if column not in {'created_at', 'updated_at', 'stored_size'}}
            if table == 'blobs':
                values['content'] = corpus._decompress(values['content'], values['compression'])
            rows.append(tuple(values.values()))
        snapshot[table] = rows
    return snapshot


@contextmanager
def open_corpus(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a corpus connection that is committed and then closed.

    Parameters
    ----------
    db_path : pathlib.Path
        Corpus database to open.

    Yields
    ------
    sqlite3.Connection
        Open corpus connection, rolled back if the caller raises.
    """
    conn = corpus.connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def read_corpus_frame(path: Path) -> pd.DataFrame:
    """Read paper rows from a test corpus as a DataFrame.

    Parameters
    ----------
    path : pathlib.Path
        Corpus database to read.

    Returns
    -------
    pandas.DataFrame
        All stored paper rows in corpus order.
    """
    with open_corpus(path) as conn:
        return pd.DataFrame(corpus.paper_rows(conn))
