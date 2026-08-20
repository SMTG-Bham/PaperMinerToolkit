"""Unit tests for paperscraper.corpus.

This module tests the standalone SQLite corpus layer for storing paper metadata,
compressed blobs, deduplicated content, paper asset links, and corpus storage
statistics without touching the command-line workflow.
"""

from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import paperscraper.corpus as corpus


def sample_paper(paper_id: str = 'demo:1') -> dict[str, Any]:
    """Return a minimal paper metadata dictionary for corpus tests."""
    return {
        'paper_id': paper_id,
        'doi': f'10.1000/{paper_id}',
        'title': f'Title {paper_id}',
        'journal': 'Journal of Demo Storage',
        'publication_date': '2026-01-01',
        'authors': 'A. Author',
        'sources': 'demo',
        'metadata': {'rank': 1},
    }


def test_corpus_stores_deduplicated_compressed_assets_and_reads_them_back(tmp_path: Path) -> None:
    """Test SQLite corpus storage for paper assets."""
    db_path = tmp_path / 'corpus.db'
    text = 'Lithium solid electrolyte storage demo. ' * 50

    with corpus.connect(db_path) as conn:
        first_blob = corpus.add_asset(
            conn,
            sample_paper('demo:1'),
            text,
            role='text',
            kind='text',
            mime_type='text/plain',
            source='demo',
            original_filename='demo_1.txt',
        )
        second_blob = corpus.add_asset(
            conn,
            sample_paper('demo:2'),
            text,
            role='text',
            kind='text',
            mime_type='text/plain',
            source='demo',
            original_filename='demo_2.txt',
        )
        asset = corpus.get_asset(conn, 'demo:1', 'text')
        stats = corpus.corpus_stats(conn)

    assert first_blob == second_blob
    assert asset['content'] == text.encode('utf-8')
    assert asset['mime_type'] == 'text/plain'
    assert stats['papers'] == 2
    assert stats['papers_with_text'] == 2
    assert stats['papers_with_pdf'] == 0
    assert stats['papers_with_abstract'] == 0
    assert stats['papers_with_chunked_text'] == 0
    assert stats['papers_with_chunked_abstracts'] == 0
    assert stats['blobs'] == 1
    assert stats['stored_size'] < stats['original_size']
    assert stats['savings_fraction'] > 0


def test_corpus_migrates_version_one_chunk_counts_without_losing_rows(tmp_path: Path) -> None:
    """Add nullable chunk-count columns to an existing version-one corpus."""
    db_path = tmp_path / 'legacy.db'
    legacy_fields = [
        field
        for field in corpus.PAPER_FIELDS
        if field not in {'num_text_chunks', 'num_abstract_chunks'}
    ]
    legacy_columns = ',\n'.join(
        f'{field} {corpus._paper_column_type(field)}'
        for field in legacy_fields
        if field != 'paper_id'
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            f"""
            PRAGMA user_version = 1;
            CREATE TABLE papers (
                paper_id TEXT PRIMARY KEY,
                {legacy_columns},
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO papers (paper_id, title, metadata_json, created_at, updated_at)
            VALUES ('legacy:1', 'Legacy paper', '{{}}', '2026-01-01', '2026-01-01');
            """
        )

    with corpus.connect(db_path) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()}
        rows = corpus.paper_rows(conn)
        tables = {
            row['name']
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert version == 3
    assert {'num_text_chunks', 'num_abstract_chunks'} <= columns
    assert len(rows) == 1
    assert rows[0]['title'] == 'Legacy paper'
    assert rows[0]['num_text_chunks'] is None
    assert rows[0]['num_abstract_chunks'] is None
    assert {'corpus_filters', 'paper_filter_results', 'paper_filter_state'} <= tables


def test_corpus_supports_uncompressed_blobs_and_missing_assets(tmp_path: Path) -> None:
    """Test uncompressed blob storage and missing asset lookup."""
    pdf = b'%PDF-1.4\n% dummy pdf\n'

    with corpus.connect(tmp_path / 'corpus.db') as conn:
        corpus.add_asset(
            conn,
            sample_paper('demo:pdf'),
            pdf,
            role='pdf',
            kind='pdf',
            mime_type='application/pdf',
            source='demo',
            original_filename='demo.pdf',
            compression='none',
        )
        asset = corpus.get_asset(conn, 'demo:pdf', 'pdf')
        missing = corpus.get_asset(conn, 'demo:pdf', 'text')

    assert asset['content'] == pdf
    assert asset['compression'] == 'none'
    assert missing is None


def test_latest_assets_bulk_loads_only_the_newest_requested_roles(tmp_path: Path) -> None:
    """Load the latest asset per paper and role without returning unrelated roles."""
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        paper = sample_paper('demo:assets')
        corpus.add_asset(conn, paper, 'older abstract', role='abstract', kind='text', mime_type='text/plain')
        corpus.add_asset(
            conn,
            paper,
            'newer abstract',
            role='abstract',
            kind='text',
            mime_type='text/plain',
            source='second',
        )
        corpus.add_asset(conn, paper, 'full text', role='text', kind='text', mime_type='text/plain')
        assets = corpus.latest_assets(conn, ['abstract'])

    assert set(assets) == {('demo:assets', 'abstract')}
    assert assets[('demo:assets', 'abstract')]['content'] == b'newer abstract'
    assert assets[('demo:assets', 'abstract')]['source'] == 'second'


def test_corpus_rejects_unknown_compression_and_decompression_codecs(tmp_path: Path) -> None:
    """Test corpus compression validation."""
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        with pytest.raises(ValueError, match='compression must be one of'):
            corpus.store_blob(conn, b'data', kind='text', mime_type='text/plain', compression='brotli')
    with pytest.raises(ValueError, match='Unsupported blob compression'):
        corpus._decompress(gzip.compress(b'data'), 'brotli')


def test_corpus_serializes_metadata_and_prepares_path_or_iterable_content(tmp_path: Path) -> None:
    """Test corpus helper conversion branches."""
    text_path = tmp_path / 'paper.txt'
    text_path.write_text('paper text')

    assert corpus._json_dumps(None) == '{}'
    assert corpus._json_dumps('{"already": true}') == '{"already": true}'
    assert corpus._prepare_content(Path(text_path)) == b'paper text'
    assert corpus._prepare_content([65, 66, 67]) == b'ABC'


def test_corpus_merges_duplicate_papers_and_preserves_existing_values(tmp_path: Path) -> None:
    """Test corpus paper upserts and duplicate merging."""
    with corpus.connect(tmp_path / 'papers.db') as conn:
        paper_id = corpus.upsert_paper(conn, {
            'paper_id': 'elsevier:1',
            'doi': '10.1000/demo',
            'title': 'Existing title',
            'sources': 'elsevier',
            'metadata': {'provider': 'elsevier'},
        })
        added, updated = corpus.upsert_papers(conn, [{
            'paper_id': 'core:1',
            'doi': 'https://doi.org/10.1000/demo',
            'title': 'Incoming title',
            'journal': 'Corpus Journal',
            'sources': 'core',
            'metadata_status': 'retrieved',
        }])
        rows = corpus.paper_rows(conn)

    assert paper_id == 'elsevier:1'
    assert (added, updated) == (0, 1)
    assert len(rows) == 1
    assert rows[0]['paper_id'] == 'elsevier:1'
    assert rows[0]['title'] == 'Existing title'
    assert rows[0]['journal'] == 'Corpus Journal'
    assert rows[0]['sources'] == 'elsevier;core'
    assert rows[0]['metadata_json'] == '{"provider": "elsevier"}'


def test_corpus_builds_fallback_ids_and_matches_by_title_year(tmp_path: Path) -> None:
    """Test fallback paper IDs and title/year duplicate matching."""
    with corpus.connect(tmp_path / 'papers.db') as conn:
        added, updated = corpus.upsert_papers(conn, [{
            'title': 'Lithium Solid Electrolyte',
            'publication_date': '2026-03-01',
            'authors': 'A. Author',
            'sources': 'elsevier',
        }, {
            'title': 'lithium: solid electrolyte',
            'publication_date': '2026',
            'journal': 'Matched Journal',
            'num_images': '3',
            'sources': 'core',
        }])
        rows = corpus.paper_rows(conn)

    assert (added, updated) == (1, 1)
    assert len(rows) == 1
    assert rows[0]['paper_id'].startswith('paper:')
    assert rows[0]['journal'] == 'Matched Journal'
    assert rows[0]['num_images'] == 3
