"""Store a compressed SQLite corpus of downloaded papers.

This module provides a small standalone corpus layer for PaperScraper. It stores
paper metadata, pipeline state, deduplicated text/PDF blobs, and paper-to-blob
links in SQLite without including the extracted materials table.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias


SCHEMA_VERSION = 10
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
    'abstract_source',
    'elsevier_link',
    'pmid',
    'pmcid',
    'arxiv_id',
    'medrxiv_doi',
    'biorxiv_doi',
    'chemrxiv_doi',
]
PIPELINE_COLUMNS = {
    'metadata_status': 'pending',
    'abstract_download_status': 'pending',
    'text_download_status': 'pending',
    'pdf_download_status': 'pending',
    'text_scrape_status': 'pending',
    'abstract_scrape_status': 'pending',
    'image_scrape_status': 'pending',
    'store_status': 'pending',
    'enrichment_status': 'pending',
    'text_path': '',
    'pdf_path': '',
    'image_dir': '',
    'num_images': 0,
    'num_text_materials': 0,
    'num_abstract_materials': 0,
    'num_image_materials': 0,
    'num_text_chunks': None,
    'num_abstract_chunks': None,
    'last_error': '',
}
PAPER_FIELDS = PAPER_COLUMNS + [column for column in PIPELINE_COLUMNS if column not in PAPER_COLUMNS]
ENRICHMENT_COLUMNS = {
    'openalex_id': 'TEXT',
    'publisher': 'TEXT',
    'work_type': 'TEXT',
    'volume': 'TEXT',
    'issue': 'TEXT',
    'pages': 'TEXT',
    'issn': 'TEXT',
    'issn_l': 'TEXT',
    'language': 'TEXT',
    'is_oa': 'INTEGER',
    'oa_status': 'TEXT',
    'license': 'TEXT',
    'is_retracted': 'INTEGER',
    'cited_by_count': 'INTEGER',
    'referenced_works_count': 'INTEGER',
    'enrichment_sources': 'TEXT',
    'enrichment_json': 'TEXT',
    'enriched_at': 'TEXT',
}
ENRICHMENT_FILL_COLUMNS = ['doi', 'title', 'journal', 'publication_date', 'authors',
                           'pmid', 'pmcid', 'arxiv_id', 'medrxiv_doi', 'biorxiv_doi',
                           'chemrxiv_doi']
AUTHOR_COLUMNS = [
    'paper_id', 'author_position', 'affiliation_rank', 'position_label', 'display_name',
    'given_name', 'family_name', 'orcid', 'is_corresponding', 'affiliation',
    'institution_name', 'institution_ror', 'institution_country', 'openalex_author_id', 'source',
]
SUBJECT_COLUMNS = [
    'paper_id', 'scheme', 'subject_id', 'display_name', 'score', 'subject_rank',
    'level', 'is_primary', 'parent_field', 'parent_domain', 'source',
]
REFERENCE_COLUMNS = [
    'paper_id', 'source', 'reference_rank', 'referenced_doi', 'referenced_openalex_id',
    'referenced_paper_id', 'referenced_title', 'unstructured',
]
ENRICHMENT_CHILD_TABLES = {
    'paper_authors': AUTHOR_COLUMNS,
    'paper_subjects': SUBJECT_COLUMNS,
    'paper_references': REFERENCE_COLUMNS,
}
ENRICHMENT_STATUSES = frozenset({
    'pending',
    'succeeded',
    'partial',
    'not_found',
    'unresolved',
    'failed',
})
_Paper: TypeAlias = dict[str, Any]
_PaperInput: TypeAlias = Mapping[str, Any]
_BlobContent: TypeAlias = str | bytes | Path | Iterable[int]
_Asset: TypeAlias = dict[str, Any]


def utc_now() -> str:
    """Return the current UTC timestamp.

    Returns
    -------
    str
        ISO-8601 timestamp with second precision.
    """
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def connect(db_path: str | PathLike[str]) -> sqlite3.Connection:
    """Open and initialize a corpus database.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        SQLite database path.

    Returns
    -------
    sqlite3.Connection
        Configured connection with dictionary-like rows.

    Raises
    ------
    RuntimeError
        If the database schema is newer than this package supports.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_corpus(conn)
    return conn


def init_corpus(conn: sqlite3.Connection) -> None:
    """Create or migrate the corpus schema.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection to initialize.

    Raises
    ------
    RuntimeError
        If the stored schema version is newer than the supported version.
    """
    conn.execute('PRAGMA foreign_keys = ON')
    current_version = conn.execute('PRAGMA user_version').fetchone()[0]
    if current_version > SCHEMA_VERSION:
        raise RuntimeError(
            f'Corpus schema version {current_version} is newer than supported version {SCHEMA_VERSION}.'
        )
    paper_columns = ',\n            '.join(
        f'{column} {column_type}' for column, column_type in _expected_paper_columns().items()
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

        CREATE TABLE IF NOT EXISTS corpus_filters (
            filter_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            method TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            definition_json TEXT NOT NULL,
            stack_position INTEGER NOT NULL,
            join_operator TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (join_operator IS NULL OR join_operator IN ('and', 'or'))
        );

        CREATE TABLE IF NOT EXISTS paper_filter_results (
            filter_id INTEGER NOT NULL,
            paper_id TEXT NOT NULL,
            status TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            unavailable_reason TEXT NOT NULL DEFAULT '',
            evaluated_at TEXT NOT NULL,
            PRIMARY KEY (filter_id, paper_id),
            FOREIGN KEY (filter_id) REFERENCES corpus_filters(filter_id) ON DELETE CASCADE,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE,
            CHECK (status IN ('included', 'excluded', 'unavailable'))
        );

        CREATE TABLE IF NOT EXISTS paper_filter_state (
            paper_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            expression TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE,
            CHECK (status IN ('included', 'excluded', 'unavailable'))
        );

        CREATE INDEX IF NOT EXISTS idx_corpus_filters_position
            ON corpus_filters(stack_position);
        CREATE INDEX IF NOT EXISTS idx_filter_results_status
            ON paper_filter_results(filter_id, status);
        CREATE INDEX IF NOT EXISTS idx_filter_state_status
            ON paper_filter_state(status);

        CREATE TABLE IF NOT EXISTS topic_models (
            model_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            artifact_path TEXT NOT NULL,
            artifact_version INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            training_corpus_fingerprint TEXT NOT NULL,
            prediction_corpus_fingerprint TEXT NOT NULL,
            text_fields_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS topic_definitions (
            model_id TEXT NOT NULL,
            topic_id INTEGER NOT NULL,
            topic_name TEXT NOT NULL DEFAULT '',
            top_terms_json TEXT NOT NULL,
            PRIMARY KEY (model_id, topic_id),
            FOREIGN KEY (model_id) REFERENCES topic_models(model_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS paper_topic_predictions (
            model_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            status TEXT NOT NULL,
            dominant_topic_id INTEGER,
            document_fingerprint TEXT NOT NULL,
            predicted_at TEXT NOT NULL,
            PRIMARY KEY (model_id, paper_id),
            FOREIGN KEY (model_id) REFERENCES topic_models(model_id) ON DELETE CASCADE,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE,
            CHECK (status IN ('predicted', 'no_vocabulary_terms'))
        );

        CREATE TABLE IF NOT EXISTS paper_topic_scores (
            model_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            topic_id INTEGER NOT NULL,
            probability REAL NOT NULL,
            PRIMARY KEY (model_id, paper_id, topic_id),
            FOREIGN KEY (model_id, paper_id)
                REFERENCES paper_topic_predictions(model_id, paper_id) ON DELETE CASCADE,
            FOREIGN KEY (model_id, topic_id)
                REFERENCES topic_definitions(model_id, topic_id) ON DELETE CASCADE,
            CHECK (probability >= 0.0 AND probability <= 1.0)
        );

        CREATE INDEX IF NOT EXISTS idx_topic_predictions_status
            ON paper_topic_predictions(model_id, status);
        CREATE INDEX IF NOT EXISTS idx_topic_scores_topic_probability
            ON paper_topic_scores(model_id, topic_id, probability);

        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_id TEXT NOT NULL,
            author_position INTEGER NOT NULL,
            affiliation_rank INTEGER NOT NULL DEFAULT 0,
            position_label TEXT NOT NULL DEFAULT '',
            display_name TEXT NOT NULL DEFAULT '',
            given_name TEXT NOT NULL DEFAULT '',
            family_name TEXT NOT NULL DEFAULT '',
            orcid TEXT NOT NULL DEFAULT '',
            is_corresponding INTEGER NOT NULL DEFAULT 0,
            affiliation TEXT NOT NULL DEFAULT '',
            institution_name TEXT NOT NULL DEFAULT '',
            institution_ror TEXT NOT NULL DEFAULT '',
            institution_country TEXT NOT NULL DEFAULT '',
            openalex_author_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL,
            PRIMARY KEY (paper_id, author_position, affiliation_rank),
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS paper_subjects (
            paper_id TEXT NOT NULL,
            scheme TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            score REAL,
            subject_rank INTEGER NOT NULL DEFAULT 0,
            level INTEGER,
            is_primary INTEGER NOT NULL DEFAULT 0,
            parent_field TEXT NOT NULL DEFAULT '',
            parent_domain TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'openalex',
            PRIMARY KEY (paper_id, scheme, subject_id),
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS paper_references (
            paper_id TEXT NOT NULL,
            source TEXT NOT NULL,
            reference_rank INTEGER NOT NULL,
            referenced_doi TEXT NOT NULL DEFAULT '',
            referenced_openalex_id TEXT NOT NULL DEFAULT '',
            referenced_paper_id TEXT NOT NULL DEFAULT '',
            referenced_title TEXT NOT NULL DEFAULT '',
            unstructured TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (paper_id, source, reference_rank),
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_paper_authors_orcid ON paper_authors(orcid);
        CREATE INDEX IF NOT EXISTS idx_paper_authors_ror ON paper_authors(institution_ror);
        CREATE INDEX IF NOT EXISTS idx_paper_subjects_subject ON paper_subjects(scheme, subject_id);
        CREATE INDEX IF NOT EXISTS idx_paper_references_doi ON paper_references(referenced_doi);
        CREATE INDEX IF NOT EXISTS idx_paper_references_openalex
            ON paper_references(referenced_openalex_id);
        """
    )
    existing_columns = {
        row['name'] if isinstance(row, sqlite3.Row) else row[1]
        for row in conn.execute('PRAGMA table_info(papers)').fetchall()
    }
    added_columns = []
    for column, column_type in _expected_paper_columns().items():
        if column in existing_columns:
            continue
        conn.execute(f'ALTER TABLE papers ADD COLUMN {column} {column_type}')
        added_columns.append(column)
    if 'enrichment_status' in added_columns:
        conn.execute(
            "UPDATE papers SET enrichment_status = 'pending' WHERE enrichment_status IS NULL"
        )
    conn.execute('CREATE INDEX IF NOT EXISTS idx_papers_enrichment ON papers(enrichment_status)')
    conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
    conn.commit()


def _paper_column_type(column: str) -> str:
    """Resolve the SQLite type for a paper column.

    Parameters
    ----------
    column : str
        Paper metadata or pipeline-state column name.

    Returns
    -------
    str
        SQLite type name.
    """
    if column in {
        'num_images',
        'num_text_materials',
        'num_abstract_materials',
        'num_image_materials',
        'num_text_chunks',
        'num_abstract_chunks',
    }:
        return 'INTEGER'
    return 'TEXT'


def _expected_paper_columns() -> dict[str, str]:
    """Map every column the papers table must have to its SQLite type.

    The mapping drives both the ``CREATE TABLE`` statement and the migration
    that reconciles an existing database, so a new metadata or enrichment
    column is added in one place only.

    Returns
    -------
    dict[str, str]
        Column name to declared SQLite type, excluding the primary key.
    """
    columns = {column: _paper_column_type(column) for column in PAPER_FIELDS if column != 'paper_id'}
    columns.update(ENRICHMENT_COLUMNS)
    return columns


def _has_value(value: object) -> bool:
    """Check whether a value contains meaningful content.

    Parameters
    ----------
    value : object
        Candidate scalar value.

    Returns
    -------
    bool
        ``True`` for non-empty, non-NaN values.
    """
    if value is None:
        return False
    try:
        if math.isnan(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ''


def _clean_doi(value: object) -> str:
    """Normalize a DOI-like value for matching.

    Parameters
    ----------
    value : object
        DOI, DOI URL, or prefixed DOI value.

    Returns
    -------
    str
        Lower-case bare DOI, or an empty string.
    """
    if not _has_value(value):
        return ''
    doi = str(value).strip()
    if doi.lower().startswith('doi:'):
        doi = doi[4:]
    if doi.lower().startswith('https://doi.org/'):
        doi = doi[16:]
    return doi.strip().rstrip('.').lower()


def _title_key(value: object) -> str:
    """Create a comparable paper-title key.

    Parameters
    ----------
    value : object
        Paper title value.

    Returns
    -------
    str
        Normalized lower-case title key.
    """
    if not _has_value(value):
        return ''
    return re.sub(r'\W+', ' ', str(value).lower()).strip()


def _year(value: object) -> str:
    """Extract a year from a date-like value.

    Parameters
    ----------
    value : object
        Date-like input value.

    Returns
    -------
    str
        First four-digit year, or an empty string.
    """
    if not _has_value(value):
        return ''
    match = re.search(r'\d{4}', str(value))
    return match.group(0) if match else ''


def _merge_sources(current: object, incoming: object) -> str:
    """Combine semicolon-separated source names.

    Parameters
    ----------
    current : object
        Existing source names.
    incoming : object
        New source names to merge.

    Returns
    -------
    str
        Deduplicated sources in first-seen order.
    """
    values = []
    for value in [current, incoming]:
        if not _has_value(value):
            continue
        values.extend(part.strip() for part in str(value).split(';') if part.strip())
    return ';'.join(dict.fromkeys(values))


def _fallback_paper_id(paper: _PaperInput) -> str:
    """Build a stable fallback paper identifier.

    Parameters
    ----------
    paper : _PaperInput
        Normalized paper metadata.

    Returns
    -------
    str
        DOI-, PubMed-, arXiv-, medRxiv-, bioRxiv-, chemRxiv-, CORE-, or
        content-derived paper identifier.
    """
    doi = _clean_doi(paper.get('doi'))
    if doi:
        return f'doi:{doi}'
    pmid = paper.get('pmid')
    if _has_value(pmid):
        return f'pmid:{pmid}'
    arxiv_id = paper.get('arxiv_id')
    if _has_value(arxiv_id):
        return f'arxiv:{arxiv_id}'
    medrxiv_doi = paper.get('medrxiv_doi')
    if _has_value(medrxiv_doi):
        return f'doi:{_clean_doi(medrxiv_doi) or medrxiv_doi}'
    biorxiv_doi = paper.get('biorxiv_doi')
    if _has_value(biorxiv_doi):
        return f'doi:{_clean_doi(biorxiv_doi) or biorxiv_doi}'
    chemrxiv_doi = paper.get('chemrxiv_doi')
    if _has_value(chemrxiv_doi):
        return f'doi:{_clean_doi(chemrxiv_doi) or chemrxiv_doi}'
    core_id = paper.get('core_id')
    if _has_value(core_id):
        return f'core:{core_id}'
    seed = '|'.join(str(paper.get(column) or '') for column in ['title', 'publication_date', 'authors'])
    return f'paper:{hashlib.sha1(seed.encode("utf-8")).hexdigest()}'


def normalize_paper(paper: _PaperInput) -> _Paper:
    """Normalize a paper mapping to the corpus schema.

    Parameters
    ----------
    paper : _PaperInput
        Provider or user-supplied paper metadata.

    Returns
    -------
    _Paper
        Complete normalized paper row with pipeline defaults.
    """
    normalized = {column: '' for column in PAPER_COLUMNS}
    normalized.update(PIPELINE_COLUMNS)
    for column in PAPER_FIELDS:
        value = paper.get(column)
        if _has_value(value):
            normalized[column] = int(value) if _paper_column_type(column) == 'INTEGER' else str(value)
    if not _has_value(normalized['paper_id']):
        normalized['paper_id'] = _fallback_paper_id(normalized)
    return normalized


def _merge_paper(existing: _PaperInput, incoming: _PaperInput) -> _Paper:
    """Merge incoming metadata into an existing paper row.

    Parameters
    ----------
    existing : _PaperInput
        Existing corpus paper row.
    incoming : _PaperInput
        Incoming paper metadata.

    Returns
    -------
    _Paper
        Merged paper row that preserves established values.
    """
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


def _papers_match(existing: _PaperInput, incoming: _PaperInput) -> bool:
    """Check whether two rows describe the same publication.

    Parameters
    ----------
    existing : _PaperInput
        Existing corpus paper row.
    incoming : _PaperInput
        Incoming paper metadata.

    Returns
    -------
    bool
        Whether stable identifiers or title and year match.
    """
    existing_doi = _clean_doi(existing.get('doi'))
    incoming_doi = _clean_doi(incoming.get('doi'))
    if existing_doi and incoming_doi and existing_doi == incoming_doi:
        return True
    for column in ['paper_id', 'core_id', 'pmid', 'pmcid', 'arxiv_id', 'medrxiv_doi',
                   'biorxiv_doi', 'chemrxiv_doi']:
        if _has_value(existing.get(column)) and str(existing.get(column)) == str(incoming.get(column)):
            return True
    existing_title = _title_key(existing.get('title'))
    incoming_title = _title_key(incoming.get('title'))
    existing_year = _year(existing.get('publication_date'))
    incoming_year = _year(incoming.get('publication_date'))
    return bool(existing_title and incoming_title and existing_title == incoming_title
                and existing_year and incoming_year and existing_year == incoming_year)


def _find_existing_paper(conn: sqlite3.Connection, paper: _PaperInput) -> _Paper | None:
    """Find the first corpus row matching a paper.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : _PaperInput
        Paper metadata to match.

    Returns
    -------
    _Paper or None
        Matching corpus row, or ``None``.
    """
    incoming = normalize_paper(paper)
    for existing in paper_rows(conn):
        if _papers_match(existing, incoming):
            return existing
    return None


def find_paper(conn: sqlite3.Connection, paper: _PaperInput) -> _Paper | None:
    """Find a corpus paper matching supplied metadata.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : _PaperInput
        Paper metadata to match.

    Returns
    -------
    _Paper or None
        Matching corpus row, or ``None``.
    """
    return _find_existing_paper(conn, paper)


def _json_dumps(value: Any) -> str:
    """Serialize metadata as deterministic JSON text.

    Parameters
    ----------
    value : object
        Metadata mapping, existing JSON text, or empty value.

    Returns
    -------
    str
        Serialized JSON text.
    """
    if value is None or value == '':
        return '{}'
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def upsert_paper(conn: sqlite3.Connection, paper: _PaperInput) -> str:
    """Insert or update one paper row.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : _PaperInput
        Paper metadata and optional pipeline state.

    Returns
    -------
    str
        Identifier of the inserted or updated paper.
    """
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


def upsert_papers(conn: sqlite3.Connection, papers: Iterable[_PaperInput]) -> tuple[int, int]:
    """Merge multiple papers into the corpus.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    papers : Iterable[_PaperInput]
        Paper mappings to insert or merge.

    Returns
    -------
    tuple[int, int]
        Counts of added and updated papers.
    """
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


def paper_rows(conn: sqlite3.Connection) -> list[_Paper]:
    """Load corpus paper rows in insertion order.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.

    Returns
    -------
    list[_Paper]
        Paper rows in their stored representation.
    """
    return [
        dict(row)
        for row in conn.execute('SELECT * FROM papers ORDER BY rowid').fetchall()
    ]


def _enrichment_update_sql() -> str:
    """Build the parameterized UPDATE statement used to write enrichment values.

    Core bibliographic columns are filled only when currently empty, mirroring
    the fill-if-empty rule in :func:`_merge_paper`. Enrichment columns are
    overwritten because citation counts and retraction flags change over time.

    Returns
    -------
    str
        UPDATE statement using named parameters.
    """
    assignments = [f"{column} = COALESCE(NULLIF({column}, ''), :{column})"
                   for column in ENRICHMENT_FILL_COLUMNS]
    assignments.extend(f'{column} = :{column}' for column in ENRICHMENT_COLUMNS)
    assignments.append('enrichment_status = :enrichment_status')
    assignments.append('updated_at = :updated_at')
    return f"UPDATE papers SET {', '.join(assignments)} WHERE paper_id = :paper_id"


def enrichment_update_fields() -> list[str]:
    """List every named parameter the enrichment UPDATE statement expects.

    Returns
    -------
    list[str]
        Parameter names, including ``paper_id`` and ``updated_at``.
    """
    return (ENRICHMENT_FILL_COLUMNS + list(ENRICHMENT_COLUMNS)
            + ['enrichment_status', 'updated_at', 'paper_id'])


def enrichment_candidates(conn: sqlite3.Connection,
                          statuses: Iterable[str] = ('pending',),
                          after_rowid: int = 0,
                          limit: int = 100,
                          refreshed_before: str = '') -> list[_Paper]:
    """Return the next page of papers needing enrichment.

    Results are keyset-paginated by ``rowid`` so an interrupted run resumes
    without extra state and terminates even when a selected status is unchanged
    by the run.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    statuses : Iterable[str], default=('pending',)
        Enrichment statuses selected for processing.
    after_rowid : int, default=0
        Exclusive lower bound on the ``rowid`` of returned rows.
    limit : int, default=100
        Maximum number of rows to return.
    refreshed_before : str, default=''
        Also select succeeded rows enriched before this ISO-8601 timestamp.

    Returns
    -------
    list[_Paper]
        Candidate rows including their ``rowid``.
    """
    statuses = list(statuses)
    placeholders = ', '.join('?' for _ in statuses) or "''"
    rows = conn.execute(
        f"""
        SELECT rowid AS rowid, paper_id, doi, title, journal, publication_date, authors,
               openalex_id, pmid, pmcid, arxiv_id, medrxiv_doi, biorxiv_doi,
               chemrxiv_doi,
               enrichment_status, enriched_at
        FROM papers
        WHERE rowid > ?
          AND (enrichment_status IN ({placeholders})
               OR (? <> ''
                   AND enrichment_status = 'succeeded'
                   AND enriched_at IS NOT NULL
                   AND enriched_at <> ''
                   AND enriched_at < ?))
        ORDER BY rowid
        LIMIT ?
        """,
        [after_rowid, *statuses, refreshed_before, refreshed_before, limit],
    ).fetchall()
    return [dict(row) for row in rows]


def _child_column_defaults(conn: sqlite3.Connection, table: str) -> dict[str, Any]:
    """Read the declared default for every column of an enrichment child table.

    Defaults are read from the live schema so a caller may omit any column
    without tripping its ``NOT NULL`` constraint, and so this helper cannot
    drift from the table definition.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    table : str
        Enrichment child table name.

    Returns
    -------
    dict[str, Any]
        Column name to the value used when a row omits that column.
    """
    defaults = {}
    for row in conn.execute(f'PRAGMA table_info({table})').fetchall():
        name = row['name'] if isinstance(row, sqlite3.Row) else row[1]
        not_null = row['notnull'] if isinstance(row, sqlite3.Row) else row[3]
        declared = row['dflt_value'] if isinstance(row, sqlite3.Row) else row[4]
        if declared is None:
            defaults[name] = None if not not_null else ''
            continue
        declared = str(declared)
        if declared.startswith("'") and declared.endswith("'"):
            defaults[name] = declared[1:-1]
        else:
            try:
                defaults[name] = int(declared)
            except ValueError:
                defaults[name] = declared
    return defaults


def write_enrichment(conn: sqlite3.Connection,
                     updates: Iterable[_PaperInput],
                     authors: Iterable[_PaperInput] = (),
                     subjects: Iterable[_PaperInput] = (),
                     references: Iterable[_PaperInput] = (),
                     sources: Iterable[str] = ()) -> int:
    """Write one batch of enrichment updates and child rows in a transaction.

    Child rows are replaced per paper rather than merged, so re-running
    enrichment cannot leave stale authors, subjects, or references behind. The
    replacement is scoped to ``sources`` so enriching from one provider leaves
    another provider's rows for the same paper intact.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    updates : Iterable[_PaperInput]
        One mapping per paper containing every enrichment update parameter.
    authors : Iterable[_PaperInput], optional
        Rows for the ``paper_authors`` table.
    subjects : Iterable[_PaperInput], optional
        Rows for the ``paper_subjects`` table.
    references : Iterable[_PaperInput], optional
        Rows for the ``paper_references`` table.
    sources : Iterable[str], optional
        Providers these rows came from. Only their rows are replaced; an empty
        value replaces every child row for the papers being updated.

    Returns
    -------
    int
        Number of papers updated.
    """
    updates = [dict(update) for update in updates]
    if not updates:
        return 0
    child_rows = {
        'paper_authors': [dict(row) for row in authors],
        'paper_subjects': [dict(row) for row in subjects],
        'paper_references': [dict(row) for row in references],
    }
    scoped = [str(source) for source in sources if str(source)]
    if scoped:
        placeholders = ', '.join('?' for _ in scoped)
        delete_clause = f'WHERE paper_id = ? AND source IN ({placeholders})'
        delete_parameters = [(str(update['paper_id']), *scoped) for update in updates]
    else:
        delete_clause = 'WHERE paper_id = ?'
        delete_parameters = [(str(update['paper_id']),) for update in updates]
    try:
        conn.execute('BEGIN')
        conn.executemany(_enrichment_update_sql(), updates)
        for table, rows in child_rows.items():
            conn.executemany(f'DELETE FROM {table} {delete_clause}', delete_parameters)
            columns = ENRICHMENT_CHILD_TABLES[table]
            defaults = _child_column_defaults(conn, table)
            placeholders = ', '.join(f':{column}' for column in columns)
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                [
                    {
                        column: defaults[column] if row.get(column) is None else row[column]
                        for column in columns
                    }
                    for row in rows
                ],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(updates)


def set_enrichment_status(conn: sqlite3.Connection,
                          paper_ids: Iterable[str],
                          status: str) -> int:
    """Set the enrichment status for papers that produced no enrichment payload.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_ids : Iterable[str]
        Papers to update.
    status : str
        Enrichment status to store.

    Returns
    -------
    int
        Number of papers updated.

    Raises
    ------
    ValueError
        If ``status`` is not a supported enrichment status.
    """
    if status not in ENRICHMENT_STATUSES:
        raise ValueError(f'enrichment status must be one of: {", ".join(sorted(ENRICHMENT_STATUSES))}')
    paper_ids = [(status, utc_now(), str(paper_id)) for paper_id in paper_ids]
    if not paper_ids:
        return 0
    conn.executemany(
        'UPDATE papers SET enrichment_status = ?, updated_at = ? WHERE paper_id = ?',
        paper_ids,
    )
    conn.commit()
    return len(paper_ids)


def paper_authors(conn: sqlite3.Connection, paper_id: str) -> list[_Paper]:
    """Return one paper's structured authors.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_id : str
        Paper identifier to read.

    Returns
    -------
    list[_Paper]
        Author rows ordered by position and affiliation rank.
    """
    rows = conn.execute(
        'SELECT * FROM paper_authors WHERE paper_id = ? ORDER BY author_position, affiliation_rank',
        (paper_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def paper_subjects(conn: sqlite3.Connection, paper_id: str, scheme: str = '') -> list[_Paper]:
    """Return one paper's subjects.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_id : str
        Paper identifier to read.
    scheme : str, default=''
        Restrict results to a single subject scheme when non-empty.

    Returns
    -------
    list[_Paper]
        Subject rows ordered by scheme and rank.
    """
    query = 'SELECT * FROM paper_subjects WHERE paper_id = ?'
    params: list[Any] = [paper_id]
    if scheme:
        query += ' AND scheme = ?'
        params.append(scheme)
    rows = conn.execute(f'{query} ORDER BY scheme, subject_rank', params).fetchall()
    return [dict(row) for row in rows]


def paper_references(conn: sqlite3.Connection, paper_id: str, source: str = '') -> list[_Paper]:
    """Return one paper's references.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_id : str
        Paper identifier to read.
    source : str, default=''
        Restrict results to one provider when non-empty.

    Returns
    -------
    list[_Paper]
        Reference rows ordered by source and rank.
    """
    query = 'SELECT * FROM paper_references WHERE paper_id = ?'
    params: list[Any] = [paper_id]
    if source:
        query += ' AND source = ?'
        params.append(source)
    rows = conn.execute(f'{query} ORDER BY source, reference_rank', params).fetchall()
    return [dict(row) for row in rows]


def enrichment_stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Summarize enrichment progress and stored enrichment rows.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.

    Returns
    -------
    dict[str, int]
        Per-status paper counts and child-table row totals.
    """
    stats = {
        f'papers_{status}': conn.execute(
            'SELECT COUNT(*) FROM papers WHERE enrichment_status = ?', (status,)
        ).fetchone()[0]
        for status in sorted(ENRICHMENT_STATUSES)
    }
    stats['papers_open_access'] = conn.execute('SELECT COUNT(*) FROM papers WHERE is_oa = 1').fetchone()[0]
    stats['papers_retracted'] = conn.execute(
        'SELECT COUNT(*) FROM papers WHERE is_retracted = 1').fetchone()[0]
    stats['author_records'] = conn.execute('SELECT COUNT(*) FROM paper_authors').fetchone()[0]
    stats['authors_with_orcid'] = conn.execute(
        "SELECT COUNT(*) FROM paper_authors WHERE orcid <> ''").fetchone()[0]
    stats['subject_records'] = conn.execute('SELECT COUNT(*) FROM paper_subjects').fetchone()[0]
    stats['reference_records'] = conn.execute('SELECT COUNT(*) FROM paper_references').fetchone()[0]
    return stats


def _prepare_content(content: _BlobContent) -> bytes:
    """Convert supported content inputs to bytes.

    Parameters
    ----------
    content : str, bytes, pathlib.Path, or Iterable[int]
        Text, raw bytes, file path, or byte-like iterable.

    Returns
    -------
    bytes
        Prepared binary content.
    """
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode('utf-8')
    if isinstance(content, Path):
        return content.read_bytes()
    return bytes(content)


def _compress(content: bytes, compression: str) -> bytes:
    """Compress content with a corpus storage codec.

    Parameters
    ----------
    content : bytes
        Raw content to compress.
    compression : str
        Compression codec name.

    Returns
    -------
    bytes
        Stored content bytes.

    Raises
    ------
    ValueError
        If the codec is unsupported.
    """
    if compression not in SUPPORTED_COMPRESSIONS:
        raise ValueError(f'compression must be one of: {", ".join(sorted(SUPPORTED_COMPRESSIONS))}')
    if compression == 'gzip':
        return gzip.compress(content)
    return content


def _decompress(content: bytes, compression: str) -> bytes:
    """Decompress stored corpus content.

    Parameters
    ----------
    content : bytes
        Stored content bytes.
    compression : str
        Compression codec name.

    Returns
    -------
    bytes
        Decompressed content.

    Raises
    ------
    ValueError
        If the codec is unsupported.
    """
    if compression == 'gzip':
        return gzip.decompress(content)
    if compression == 'none':
        return content
    raise ValueError(f'Unsupported blob compression: {compression}')


def store_blob(conn: sqlite3.Connection,
               content: _BlobContent,
               kind: str,
               mime_type: str,
               compression: str = 'gzip') -> str:
    """Store a deduplicated content blob.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    content : _BlobContent
        Content accepted by :func:`_prepare_content`.
    kind : str
        Logical content kind, such as ``"text"`` or ``"pdf"``.
    mime_type : str
        MIME type of the uncompressed content.
    compression : str, optional
        Storage compression codec.

    Returns
    -------
    str
        Stable content-addressed blob identifier.

    Raises
    ------
    ValueError
        If the compression codec is unsupported.
    """
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


def link_asset(conn: sqlite3.Connection,
               paper_id: str,
               blob_id: str,
               role: str,
               source: str = '',
               original_filename: str = '') -> None:
    """Link a stored blob to a paper asset role.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_id : str
        Identifier of the owning paper.
    blob_id : str
        Identifier of the stored content blob.
    role : str
        Asset role, such as ``"abstract"``, ``"text"``, or ``"pdf"``.
    source : str, optional
        Provider or acquisition source.
    original_filename : str, optional
        Original asset filename.
    """
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


def add_asset(conn: sqlite3.Connection,
              paper: _PaperInput,
              content: _BlobContent,
              role: str,
              kind: str,
              mime_type: str,
              source: str = '',
              original_filename: str = '',
              compression: str = 'gzip') -> str:
    """Store and link an asset for a paper.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : _PaperInput
        Owning paper metadata.
    content : _BlobContent
        Asset content accepted by :func:`_prepare_content`.
    role : str
        Asset role within the paper.
    kind : str
        Logical content kind.
    mime_type : str
        MIME type of the uncompressed content.
    source : str, optional
        Provider or acquisition source.
    original_filename : str, optional
        Original asset filename.
    compression : str, optional
        Storage compression codec.

    Returns
    -------
    str
        Identifier of the linked content blob.
    """
    paper_id = upsert_paper(conn, paper)
    blob_id = store_blob(conn, content, kind=kind, mime_type=mime_type, compression=compression)
    link_asset(conn, paper_id, blob_id, role=role, source=source, original_filename=original_filename)
    return blob_id


def get_asset(conn: sqlite3.Connection, paper_id: str, role: str) -> _Asset | None:
    """Load the newest asset for a paper role.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_id : str
        Owning paper identifier.
    role : str
        Asset role to retrieve.

    Returns
    -------
    _Asset or None
        Asset metadata with decompressed ``content``, or ``None``.
    """
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
        ORDER BY a.created_at DESC, a.rowid DESC
        LIMIT 1
        """,
        (paper_id, role),
    ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data['content'] = _decompress(data['content'], data['compression'])
    return data


def get_asset_metadata(
    conn: sqlite3.Connection,
    paper_id: str,
    role: str,
) -> _Asset | None:
    """Load metadata for the newest linked asset without its content.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_id : str
        Owning paper identifier.
    role : str
        Asset role to retrieve.

    Returns
    -------
    _Asset or None
        Asset metadata, or ``None`` when the role has not been stored.
    """
    row = conn.execute(
        """
        SELECT
            p.paper_id, p.doi, p.title, a.role, a.source, a.original_filename,
            b.blob_id, b.kind, b.mime_type, b.original_size, b.stored_size,
            a.created_at
        FROM paper_assets AS a
        JOIN papers AS p ON p.paper_id = a.paper_id
        JOIN blobs AS b ON b.blob_id = a.blob_id
        WHERE a.paper_id = ? AND a.role = ?
        ORDER BY a.created_at DESC, a.rowid DESC
        LIMIT 1
        """,
        (paper_id, role),
    ).fetchone()
    return dict(row) if row is not None else None


def latest_assets(
    conn: sqlite3.Connection,
    roles: Iterable[str],
) -> dict[tuple[str, str], _Asset]:
    """Load the newest requested assets in bulk.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    roles : Iterable[str]
        Asset roles to retrieve.

    Returns
    -------
    dict[tuple[str, str], _Asset]
        ``(paper_id, role)`` keys mapped to decompressed asset rows.
    """
    roles = tuple(dict.fromkeys(roles))
    if not roles:
        return {}
    placeholders = ', '.join('?' for _ in roles)
    rows = conn.execute(
        f"""
        SELECT paper_id, role, source, original_filename, compression, content
        FROM (
            SELECT
                a.paper_id, a.role, a.source, a.original_filename,
                a.created_at, a.rowid AS asset_rowid,
                b.compression, b.content,
                ROW_NUMBER() OVER (
                    PARTITION BY a.paper_id, a.role
                    ORDER BY a.created_at DESC, a.rowid DESC
                ) AS asset_rank
            FROM paper_assets AS a
            JOIN blobs AS b ON b.blob_id = a.blob_id
            WHERE a.role IN ({placeholders})
        )
        WHERE asset_rank = 1
        """,
        roles,
    ).fetchall()
    assets = {}
    for row in rows:
        asset = dict(row)
        asset['content'] = _decompress(asset['content'], asset.pop('compression'))
        assets[(asset['paper_id'], asset['role'])] = asset
    return assets


def corpus_stats(conn: sqlite3.Connection) -> dict[str, int | float]:
    """Calculate corpus storage statistics.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.

    Returns
    -------
    dict[str, int or float]
        Paper, asset, blob-size, and compression-savings statistics.
    """
    paper_count = conn.execute('SELECT COUNT(*) FROM papers').fetchone()[0]
    blob_count = conn.execute('SELECT COUNT(*) FROM blobs').fetchone()[0]
    asset_counts = {
        role: conn.execute(
            'SELECT COUNT(DISTINCT paper_id) FROM paper_assets WHERE role = ?',
            (role,),
        ).fetchone()[0]
        for role in ['abstract', 'text', 'pdf']
    }
    sizes = conn.execute('SELECT COALESCE(SUM(original_size), 0), COALESCE(SUM(stored_size), 0) FROM blobs').fetchone()
    chunked_text = conn.execute('SELECT COUNT(*) FROM papers WHERE num_text_chunks > 1').fetchone()[0]
    chunked_abstracts = conn.execute('SELECT COUNT(*) FROM papers WHERE num_abstract_chunks > 1').fetchone()[0]
    original_size, stored_size = sizes
    savings = 0 if original_size == 0 else 1 - (stored_size / original_size)
    return {
        'papers': paper_count,
        'papers_with_abstract': asset_counts['abstract'],
        'papers_with_text': asset_counts['text'],
        'papers_with_pdf': asset_counts['pdf'],
        'papers_with_chunked_text': chunked_text,
        'papers_with_chunked_abstracts': chunked_abstracts,
        'blobs': blob_count,
        'original_size': original_size,
        'stored_size': stored_size,
        'savings_fraction': savings,
    }
