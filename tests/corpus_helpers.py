"""Shared corpus readers and connection lifetime helpers for tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from paperminertoolkit.corpus import database as corpus


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
