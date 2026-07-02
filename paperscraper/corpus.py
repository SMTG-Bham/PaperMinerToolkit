"""Store a compressed SQLite corpus of downloaded papers.

This module provides a small standalone corpus layer for PaperScraper. It stores
paper metadata, deduplicated text/PDF blobs, and paper-to-blob links in SQLite
without changing the current CSV commands or materials workflow.
"""

import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SUPPORTED_COMPRESSIONS = {'none', 'gzip'}


def utc_now():
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def connect(db_path: str | Path):
    """Open a SQLite corpus database and initialize its schema."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_corpus(conn)
    return conn


def init_corpus(conn):
    """Create corpus tables and indexes when they do not already exist."""
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            doi TEXT,
            title TEXT,
            journal TEXT,
            publication_date TEXT,
            authors TEXT,
            sources TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS blobs (
            blob_id TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            compression TEXT NOT NULL,
            original_size INTEGER NOT NULL,
            stored_size INTEGER NOT NULL,
            content BLOB NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_assets (
            paper_id TEXT NOT NULL,
            blob_id TEXT NOT NULL,
            role TEXT NOT NULL,
            source TEXT,
            original_filename TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (paper_id, role, source, original_filename),
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE,
            FOREIGN KEY (blob_id) REFERENCES blobs(blob_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
        CREATE INDEX IF NOT EXISTS idx_blobs_sha256 ON blobs(sha256);
        CREATE INDEX IF NOT EXISTS idx_assets_paper_role ON paper_assets(paper_id, role);
        """
    )
    conn.commit()


def _json_dumps(value):
    """Serialize metadata dictionaries to deterministic JSON text."""
    if value is None or value == '':
        return '{}'
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def upsert_paper(conn, paper: dict[str, Any]):
    """Insert or update one paper metadata row."""
    now = utc_now()
    paper_id = str(paper['paper_id'])
    existing = conn.execute('SELECT created_at FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()
    created_at = existing['created_at'] if existing else now
    conn.execute(
        """
        INSERT INTO papers (
            paper_id, doi, title, journal, publication_date, authors, sources,
            metadata_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            doi = excluded.doi,
            title = excluded.title,
            journal = excluded.journal,
            publication_date = excluded.publication_date,
            authors = excluded.authors,
            sources = excluded.sources,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at
        """,
        (
            paper_id,
            paper.get('doi'),
            paper.get('title'),
            paper.get('journal'),
            paper.get('publication_date'),
            paper.get('authors'),
            paper.get('sources'),
            _json_dumps(paper.get('metadata')),
            created_at,
            now,
        ),
    )
    conn.commit()
    return paper_id


def _prepare_content(content):
    """Return byte content from text, bytes, or path-like inputs."""
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode('utf-8')
    if isinstance(content, Path):
        return content.read_bytes()
    return bytes(content)


def _compress(content, compression):
    """Compress byte content with the requested corpus storage codec."""
    if compression not in SUPPORTED_COMPRESSIONS:
        raise ValueError(f'compression must be one of: {", ".join(sorted(SUPPORTED_COMPRESSIONS))}')
    if compression == 'gzip':
        return gzip.compress(content)
    return content


def _decompress(content, compression):
    """Decompress corpus blob bytes according to their stored codec."""
    if compression == 'gzip':
        return gzip.decompress(content)
    if compression == 'none':
        return content
    raise ValueError(f'Unsupported blob compression: {compression}')


def store_blob(conn,
               content,
               kind: str,
               mime_type: str,
               compression: str = 'gzip'):
    """Store deduplicated blob content and return its blob id."""
    raw = _prepare_content(content)
    sha256 = hashlib.sha256(raw).hexdigest()
    existing = conn.execute('SELECT blob_id FROM blobs WHERE sha256 = ?', (sha256,)).fetchone()
    if existing:
        return existing['blob_id']
    stored = _compress(raw, compression)
    blob_id = f'{kind}:{sha256}'
    conn.execute(
        """
        INSERT INTO blobs (
            blob_id, sha256, kind, mime_type, compression, original_size,
            stored_size, content, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            blob_id,
            sha256,
            kind,
            mime_type,
            compression,
            len(raw),
            len(stored),
            stored,
            utc_now(),
        ),
    )
    conn.commit()
    return blob_id


def link_asset(conn,
               paper_id: str,
               blob_id: str,
               role: str,
               source: str = '',
               original_filename: str = ''):
    """Link a stored blob to a paper as a text or PDF asset."""
    conn.execute(
        """
        INSERT OR REPLACE INTO paper_assets (
            paper_id, blob_id, role, source, original_filename, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (paper_id, blob_id, role, source, original_filename, utc_now()),
    )
    conn.commit()


def add_asset(conn,
              paper: dict[str, Any],
              content,
              role: str,
              kind: str,
              mime_type: str,
              source: str = '',
              original_filename: str = '',
              compression: str = 'gzip'):
    """Upsert a paper, store one blob, link them, and return the blob id."""
    paper_id = upsert_paper(conn, paper)
    blob_id = store_blob(conn, content, kind=kind, mime_type=mime_type, compression=compression)
    link_asset(conn, paper_id, blob_id, role=role, source=source, original_filename=original_filename)
    return blob_id


def get_asset(conn, paper_id: str, role: str):
    """Return the newest linked asset row and decompressed content for a paper role."""
    row = conn.execute(
        """
        SELECT
            p.paper_id, p.doi, p.title, a.role, a.source, a.original_filename,
            b.blob_id, b.kind, b.mime_type, b.compression, b.original_size,
            b.stored_size, b.content
        FROM paper_assets AS a
        JOIN papers AS p ON p.paper_id = a.paper_id
        JOIN blobs AS b ON b.blob_id = a.blob_id
        WHERE a.paper_id = ? AND a.role = ?
        ORDER BY a.created_at DESC
        LIMIT 1
        """,
        (paper_id, role),
    ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data['content'] = _decompress(data['content'], data['compression'])
    return data


def corpus_stats(conn):
    """Return high-level paper, blob, and storage statistics for the corpus."""
    paper_count = conn.execute('SELECT COUNT(*) FROM papers').fetchone()[0]
    blob_count = conn.execute('SELECT COUNT(*) FROM blobs').fetchone()[0]
    sizes = conn.execute('SELECT COALESCE(SUM(original_size), 0), COALESCE(SUM(stored_size), 0) FROM blobs').fetchone()
    original_size, stored_size = sizes
    savings = 0 if original_size == 0 else 1 - (stored_size / original_size)
    return {
        'papers': paper_count,
        'blobs': blob_count,
        'original_size': original_size,
        'stored_size': stored_size,
        'savings_fraction': savings,
    }
