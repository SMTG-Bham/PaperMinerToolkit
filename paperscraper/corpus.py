"""Store a compressed SQLite corpus of downloaded papers.

This module provides a small standalone corpus layer for PaperScraper. It stores
paper metadata, pipeline state, deduplicated text/PDF blobs, and paper-to-blob
links in SQLite without including the extracted materials table.
"""

import gzip
import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SUPPORTED_COMPRESSIONS = {'none', 'gzip'}
PAPER_COLUMNS = [
    'paper_id',
    'doi',
    'title',
    'journal',
    'publication_date',
    'authors',
    'sources',
    'core_id',
    'pdf_url',
    'pdf_source',
    'text_source',
    'elsevier_link',
]
PIPELINE_COLUMNS = {
    'metadata_status': 'pending',
    'text_download_status': 'pending',
    'pdf_download_status': 'pending',
    'text_scrape_status': 'pending',
    'image_scrape_status': 'pending',
    'store_status': 'pending',
    'text_path': '',
    'pdf_path': '',
    'image_dir': '',
    'num_images': 0,
    'num_text_materials': 0,
    'num_image_materials': 0,
    'last_error': '',
}
PAPER_FIELDS = PAPER_COLUMNS + [column for column in PIPELINE_COLUMNS if column not in PAPER_COLUMNS]


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
    paper_columns = ',\n            '.join(
        f'{column} {_paper_column_type(column)}' for column in PAPER_FIELDS if column != 'paper_id'
    )
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            {paper_columns},
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


def _paper_column_type(column):
    """Return the SQLite storage type for a paper metadata or state column."""
    if column in {'num_images', 'num_text_materials', 'num_image_materials'}:
        return 'INTEGER'
    return 'TEXT'


def _has_value(value) -> bool:
    """Return whether a value contains meaningful non-empty content."""
    if value is None:
        return False
    try:
        if math.isnan(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ''


def _clean_doi(value):
    """Normalize DOI-like values for reliable duplicate matching."""
    if not _has_value(value):
        return ''
    doi = str(value).strip()
    if doi.lower().startswith('doi:'):
        doi = doi[4:]
    if doi.lower().startswith('https://doi.org/'):
        doi = doi[16:]
    return doi.strip().rstrip('.').lower()


def _title_key(value):
    """Create a case-insensitive comparable key from a paper title."""
    if not _has_value(value):
        return ''
    return re.sub(r'\W+', ' ', str(value).lower()).strip()


def _year(value):
    """Extract a four-digit year from a date-like value."""
    if not _has_value(value):
        return ''
    match = re.search(r'\d{4}', str(value))
    return match.group(0) if match else ''


def _merge_sources(current, incoming):
    """Combine semicolon-separated source names while preserving first-seen order."""
    values = []
    for value in [current, incoming]:
        if not _has_value(value):
            continue
        values.extend(part.strip() for part in str(value).split(';') if part.strip())
    return ';'.join(dict.fromkeys(values))


def _fallback_paper_id(paper):
    """Build a stable paper id when a provider does not supply one."""
    doi = _clean_doi(paper.get('doi'))
    if doi:
        return f'doi:{doi}'
    core_id = paper.get('core_id')
    if _has_value(core_id):
        return f'core:{core_id}'
    seed = '|'.join(str(paper.get(column) or '') for column in ['title', 'publication_date', 'authors'])
    return f'paper:{hashlib.sha1(seed.encode("utf-8")).hexdigest()}'


def normalize_paper(paper: dict[str, Any]):
    """Normalize one paper dictionary to the corpus paper schema."""
    normalized = {column: '' for column in PAPER_COLUMNS}
    normalized.update(PIPELINE_COLUMNS)
    for column in PAPER_FIELDS:
        value = paper.get(column)
        if _has_value(value):
            normalized[column] = int(value) if _paper_column_type(column) == 'INTEGER' else str(value)
    if not _has_value(normalized['paper_id']):
        normalized['paper_id'] = _fallback_paper_id(normalized)
    return normalized


def _merge_paper(existing, incoming):
    """Merge one incoming normalized paper into an existing corpus paper row."""
    merged = normalize_paper(existing)
    incoming = normalize_paper(incoming)
    merged['paper_id'] = existing['paper_id']
    if _has_value(existing.get('metadata_json')):
        merged['metadata_json'] = existing.get('metadata_json')
    elif _has_value(incoming.get('metadata_json')):
        merged['metadata_json'] = incoming.get('metadata_json')
    for column, value in incoming.items():
        if column == 'paper_id' or not _has_value(value):
            continue
        current = merged.get(column)
        if column == 'sources':
            merged[column] = _merge_sources(current, value)
        elif _paper_column_type(column) == 'INTEGER':
            if int(current or 0) == 0:
                merged[column] = int(value)
        elif column.endswith('_status'):
            if not _has_value(current) or current == 'pending':
                merged[column] = value
        elif column == 'last_error':
            if _has_value(value) and not _has_value(current):
                merged[column] = value
        elif not _has_value(current):
            merged[column] = value
    return merged


def _papers_match(existing, incoming):
    """Return whether two paper rows describe the same publication."""
    existing_doi = _clean_doi(existing.get('doi'))
    incoming_doi = _clean_doi(incoming.get('doi'))
    if existing_doi and incoming_doi and existing_doi == incoming_doi:
        return True
    for column in ['paper_id', 'core_id']:
        if _has_value(existing.get(column)) and str(existing.get(column)) == str(incoming.get(column)):
            return True
    existing_title = _title_key(existing.get('title'))
    incoming_title = _title_key(incoming.get('title'))
    existing_year = _year(existing.get('publication_date'))
    incoming_year = _year(incoming.get('publication_date'))
    return bool(existing_title and incoming_title and existing_title == incoming_title
                and existing_year and incoming_year and existing_year == incoming_year)


def _find_existing_paper(conn, paper):
    """Return the first corpus paper row matching ``paper``, or ``None``."""
    incoming = normalize_paper(paper)
    for existing in paper_rows(conn):
        if _papers_match(existing, incoming):
            return existing
    return None


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
    metadata = paper.get('metadata') if 'metadata' in paper else paper.get('metadata_json')
    paper = normalize_paper(paper)
    paper_id = str(paper['paper_id'])
    existing = conn.execute('SELECT created_at FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()
    created_at = existing['created_at'] if existing else now
    insert_columns = PAPER_FIELDS + ['metadata_json', 'created_at', 'updated_at']
    update_columns = [column for column in insert_columns if column not in {'paper_id', 'created_at'}]
    placeholders = ', '.join('?' for _ in insert_columns)
    column_sql = ', '.join(insert_columns)
    update_sql = ', '.join(f'{column} = excluded.{column}' for column in update_columns)
    values = [paper.get(column) for column in PAPER_FIELDS]
    values.extend([_json_dumps(metadata), created_at, now])
    conn.execute(
        f"""
        INSERT INTO papers ({column_sql})
        VALUES ({placeholders})
        ON CONFLICT(paper_id) DO UPDATE SET
            {update_sql}
        """,
        values,
    )
    conn.commit()
    return paper_id


def upsert_papers(conn, papers):
    """Merge paper dictionaries into the corpus and return added/updated counts."""
    added = 0
    updated = 0
    for paper in papers:
        incoming = normalize_paper(paper)
        existing = _find_existing_paper(conn, incoming)
        if existing is None:
            upsert_paper(conn, incoming)
            added += 1
        else:
            upsert_paper(conn, _merge_paper(existing, incoming))
            updated += 1
    return added, updated


def paper_rows(conn):
    """Return corpus paper rows as dictionaries ordered by insertion order."""
    return [
        dict(row)
        for row in conn.execute('SELECT * FROM papers ORDER BY rowid').fetchall()
    ]


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
