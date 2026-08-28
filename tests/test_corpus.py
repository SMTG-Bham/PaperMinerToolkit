"""Unit tests for paperminertoolkit.corpus.database.

This module tests the standalone SQLite corpus layer for storing paper metadata,
compressed blobs, deduplicated content, paper asset links, and corpus storage
statistics without touching the command-line workflow.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import paperminertoolkit.corpus.database as corpus


V4_PAPER_FIELDS = [
    'paper_id', 'doi', 'title', 'journal', 'publication_date', 'authors', 'sources',
    'core_id', 'pdf_url', 'pdf_source', 'text_source', 'abstract_source', 'elsevier_link',
    'metadata_status', 'abstract_download_status', 'text_download_status',
    'pdf_download_status', 'text_scrape_status', 'abstract_scrape_status',
    'image_scrape_status', 'store_status', 'text_path', 'pdf_path', 'image_dir',
    'num_images', 'num_text_materials', 'num_abstract_materials', 'num_image_materials',
    'num_text_chunks', 'num_abstract_chunks', 'last_error',
]


def test_defensive_helpers_and_empty_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover schema guards, normalization edges, defaults, and empty batches."""
    conn = sqlite3.connect(':memory:')
    conn.execute(f'PRAGMA user_version = {corpus.SCHEMA_VERSION + 1}')
    with pytest.raises(RuntimeError, match='newer than supported'):
        corpus.init_corpus(conn)
    conn.close()
    assert corpus._has_value(float('nan')) is False
    assert corpus._clean_doi('doi:10.1/ABC.') == '10.1/abc'
    monkeypatch.setattr(corpus, 'normalize_paper', lambda paper: dict(paper))
    merged = corpus._merge_paper(
        {'paper_id': 'p', 'title': 'title', 'metadata_json': ''},
        {'paper_id': 'p', 'metadata_json': '{"source": true}', 'last_error': 'failed'},
    )
    assert merged['metadata_json'] == '{"source": true}'
    assert merged['last_error'] == 'failed'
    with corpus.connect(':memory:') as conn:
        conn.execute("CREATE TABLE defaults (value TEXT DEFAULT CURRENT_TIMESTAMP)")
        assert corpus._child_column_defaults(conn, 'defaults')['value'] == 'CURRENT_TIMESTAMP'
        assert corpus.write_enrichment(conn, []) == 0
        assert corpus.set_enrichment_status(conn, [], 'pending') == 0
        assert corpus.latest_assets(conn, []) == {}
V1_PAPER_FIELDS = [
    field for field in V4_PAPER_FIELDS
    if field not in {'num_text_chunks', 'num_abstract_chunks'}
]


@contextlib.contextmanager
def open_corpus(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a corpus connection that is committed and then closed.

    Parameters
    ----------
    db_path : pathlib.Path
        Corpus database to open.

    Yields
    ------
    sqlite3.Connection
        Open corpus connection.
    """
    conn = corpus.connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def write_legacy_corpus(db_path: Path, fields: list[str], version: int) -> None:
    """Create a corpus database frozen at an earlier schema version.

    Parameters
    ----------
    db_path : pathlib.Path
        Destination database path.
    fields : list[str]
        Paper columns the legacy schema declared, including ``paper_id``.
    version : int
        Schema version stamped on the legacy database.
    """
    legacy_columns = ',\n'.join(
        f'{field} {corpus._paper_column_type(field)}'
        for field in fields
        if field != 'paper_id'
    )
    with contextlib.closing(sqlite3.connect(db_path)) as conn, conn:
        conn.executescript(
            f"""
            PRAGMA user_version = {version};
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
            metadata={'language': 'en'},
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
        asset_metadata = corpus.get_asset_metadata(conn, 'demo:1', 'text')
        missing_metadata = corpus.get_asset_metadata(conn, 'demo:1', 'pdf')
        stats = corpus.corpus_stats(conn)

    assert first_blob == second_blob
    assert asset['content'] == text.encode('utf-8')
    assert asset['mime_type'] == 'text/plain'
    assert asset_metadata['blob_id'] == first_blob
    assert asset_metadata['role'] == 'text'
    assert asset['metadata'] == {'language': 'en'}
    assert asset_metadata['metadata'] == {'language': 'en'}
    assert 'content' not in asset_metadata
    assert missing_metadata is None
    assert stats['papers'] == 2
    assert stats['papers_with_text'] == 2
    assert stats['papers_with_pdf'] == 0
    assert stats['papers_with_abstract'] == 0
    assert stats['papers_with_structured_documents'] == 0
    assert stats['papers_with_figure_assets'] == 0
    assert stats['papers_with_chunked_text'] == 0
    assert stats['papers_with_chunked_abstracts'] == 0
    assert stats['blobs'] == 1
    assert stats['stored_size'] < stats['original_size']
    assert stats['savings_fraction'] > 0


def test_corpus_stores_and_filters_structured_documents(tmp_path: Path) -> None:
    """Store raw structured documents with extensible provenance metadata."""
    db_path = tmp_path / 'structured.db'
    paper = sample_paper('demo:structured')
    jats = b'<article><body><fig id="fig1"/></body></article>'
    tei = b'<TEI><text><figure xml:id="fig1"/></text></TEI>'

    with corpus.connect(db_path) as conn:
        jats_blob = corpus.add_structured_document(
            conn,
            paper,
            jats,
            document_format=' JATS ',
            source='pubmed',
            original_filename='article.nxml',
            metadata={'license': 'cc-by', 'document_format': 'wrong'},
        )
        duplicate_blob = corpus.add_structured_document(
            conn,
            sample_paper('demo:duplicate'),
            jats,
            document_format='jats',
            source='biorxiv',
            original_filename='article.xml',
        )
        corpus.add_structured_document(
            conn,
            paper,
            tei,
            document_format='tei',
            source='openalex',
            original_filename='article.tei.xml',
            mime_type='text/xml',
        )

        documents = corpus.get_structured_documents(conn, 'demo:structured')
        jats_documents = corpus.get_structured_documents(
            conn,
            'demo:structured',
            source='pubmed',
            document_format='JATS',
        )
        stats = corpus.corpus_stats(conn)

        assert corpus.has_structured_document(conn, 'demo:structured') is True
        assert corpus.has_structured_document(
            conn, 'demo:structured', source='openalex', document_format='tei',
        ) is True
        assert corpus.has_structured_document(
            conn, 'demo:structured', source='openalex', document_format='jats',
        ) is False
        assert corpus.has_structured_document(conn, 'missing') is False

    assert duplicate_blob == jats_blob
    assert len(documents) == 2
    assert documents[0]['metadata']['document_format'] == 'tei'
    assert documents[0]['mime_type'] == 'text/xml'
    assert documents[0]['content'] == tei
    assert len(jats_documents) == 1
    assert jats_documents[0]['role'] == corpus.STRUCTURED_DOCUMENT_ROLE
    assert jats_documents[0]['kind'] == 'structured-document'
    assert jats_documents[0]['mime_type'] == 'application/xml'
    assert jats_documents[0]['metadata'] == {
        'document_format': 'jats',
        'license': 'cc-by',
    }
    assert jats_documents[0]['content'] == jats
    assert stats['papers_with_structured_documents'] == 2


def test_structured_documents_require_format_and_source(tmp_path: Path) -> None:
    """Reject structured assets without the provenance needed for reuse."""
    with corpus.connect(tmp_path / 'structured.db') as conn:
        with pytest.raises(ValueError, match='document_format must not be empty'):
            corpus.add_structured_document(
                conn, sample_paper(), '<article/>', document_format=' ', source='pubmed',
            )
        with pytest.raises(ValueError, match='source must not be empty'):
            corpus.add_structured_document(
                conn, sample_paper(), '<article/>', document_format='jats', source=' ',
            )


def test_corpus_stores_linked_and_deduplicated_figure_assets(tmp_path: Path) -> None:
    """Keep figure provenance on links while deduplicating identical image bytes."""
    image = b'\x89PNG\r\n\x1a\nshared image'
    with corpus.connect(tmp_path / 'figures.db') as conn:
        first_blob = corpus.add_figure_asset(
            conn,
            sample_paper('demo:figure'),
            image,
            figure_id='fig-1',
            caption='Crystal structure.',
            source='pubmed',
            source_url='https://cdn.example/fig-1.png',
            mime_type='image/png; charset=binary',
            original_filename='fig-1.png',
            license='CC BY 4.0',
            metadata={'requested_url': 'https://repo.example/images/fig-1'},
        )
        second_blob = corpus.add_figure_asset(
            conn,
            sample_paper('demo:figure'),
            image,
            figure_id='fig-2',
            caption='The same image reused.',
            source='pubmed',
            source_url='https://cdn.example/fig-2.png',
            mime_type='image/png',
            original_filename='fig-2.png',
        )
        assets = corpus.get_figure_assets(
            conn, 'demo:figure', figure_id='fig-1', include_content=True,
        )
        stats = corpus.corpus_stats(conn)

        assert corpus.has_figure_asset(
            conn, 'demo:figure', 'https://repo.example/images/fig-1', figure_id='fig-1',
        ) is True
        assert corpus.has_figure_asset(
            conn, 'demo:figure', 'https://cdn.example/fig-1.png', figure_id='fig-1',
        ) is True
        assert corpus.has_figure_asset(
            conn, 'demo:figure', 'https://repo.example/images/fig-1', figure_id='fig-2',
        ) is False
        assert corpus.has_figure_asset(conn, 'demo:figure', '') is False

    assert first_blob == second_blob
    assert len(assets) == 1
    assert assets[0]['content'] == image
    assert assets[0]['sha256'] == hashlib.sha256(image).hexdigest()
    assert assets[0]['kind'] == 'image'
    assert assets[0]['mime_type'] == 'image/png'
    assert assets[0]['metadata']['figure_id'] == 'fig-1'
    assert assets[0]['metadata']['caption'] == 'Crystal structure.'
    assert assets[0]['metadata']['license'] == 'CC BY 4.0'
    assert stats['papers_with_figure_assets'] == 1


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'figure_id': ' '}, 'figure_id must not be empty'),
        ({'source': ' '}, 'source must not be empty'),
        ({'source_url': ' '}, 'source_url must not be empty'),
        ({'mime_type': 'text/html'}, 'mime_type must identify an image'),
    ],
)
def test_figure_assets_require_image_provenance(
    tmp_path: Path,
    kwargs: dict[str, str],
    message: str,
) -> None:
    """Reject figure assets whose required identity or media type is absent."""
    arguments = {
        'figure_id': 'fig-1',
        'caption': '',
        'source': 'pubmed',
        'source_url': 'https://example.org/fig.png',
        'mime_type': 'image/png',
        **kwargs,
    }
    with corpus.connect(tmp_path / 'figures.db') as conn:
        with pytest.raises(ValueError, match=message):
            corpus.add_figure_asset(conn, sample_paper(), b'image', **arguments)


def test_corpus_migrates_version_eleven_asset_metadata(tmp_path: Path) -> None:
    """Add asset metadata to a version-eleven corpus without losing links."""
    db_path = tmp_path / 'v11.db'
    write_legacy_corpus(db_path, V4_PAPER_FIELDS, version=11)
    with contextlib.closing(sqlite3.connect(db_path)) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE blobs (
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
            CREATE TABLE paper_assets (
                paper_id TEXT NOT NULL,
                blob_id TEXT NOT NULL,
                role TEXT NOT NULL,
                source TEXT,
                original_filename TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (paper_id, role, source, original_filename)
            );
            INSERT INTO blobs VALUES (
                'text:legacy', 'legacy-sha', 'text', 'text/plain', 'none',
                6, 6, X'6c6567616379', '2026-01-01'
            );
            INSERT INTO paper_assets VALUES (
                'legacy:1', 'text:legacy', 'text', 'legacy', 'legacy.txt', '2026-01-01'
            );
            """
        )

    with corpus.connect(db_path) as conn:
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(paper_assets)')}
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        asset = corpus.get_asset_metadata(conn, 'legacy:1', 'text')

    assert version == corpus.SCHEMA_VERSION == 12
    assert 'metadata_json' in columns
    assert asset is not None
    assert asset['blob_id'] == 'text:legacy'
    assert asset['metadata'] == {}


def test_corpus_migrates_version_one_chunk_counts_without_losing_rows(tmp_path: Path) -> None:
    """Add nullable chunk-count columns to an existing version-one corpus."""
    db_path = tmp_path / 'legacy.db'
    write_legacy_corpus(db_path, V1_PAPER_FIELDS, version=1)

    with open_corpus(db_path) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()}
        rows = corpus.paper_rows(conn)
        tables = {
            row['name']
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert version == corpus.SCHEMA_VERSION
    assert {'num_text_chunks', 'num_abstract_chunks'} <= columns
    assert len(rows) == 1
    assert rows[0]['title'] == 'Legacy paper'
    assert rows[0]['num_text_chunks'] is None
    assert rows[0]['num_abstract_chunks'] is None
    assert {
        'corpus_filters', 'paper_filter_results', 'paper_filter_state',
        'topic_models', 'topic_definitions', 'paper_topic_predictions',
        'paper_topic_scores', 'search_runs', 'paper_search_results',
    } <= tables


def test_corpus_migrates_version_four_enrichment_columns_without_losing_rows(tmp_path: Path) -> None:
    """Add every enrichment column to an existing version-four corpus."""
    db_path = tmp_path / 'v4.db'
    write_legacy_corpus(db_path, V4_PAPER_FIELDS, version=4)

    with open_corpus(db_path) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()}
        rows = corpus.paper_rows(conn)

    assert version == corpus.SCHEMA_VERSION
    assert set(corpus.ENRICHMENT_COLUMNS) <= columns
    assert 'enrichment_status' in columns
    assert len(rows) == 1
    assert rows[0]['title'] == 'Legacy paper'
    assert rows[0]['enrichment_status'] == 'pending'
    assert rows[0]['publisher'] is None
    assert rows[0]['cited_by_count'] is None


def test_migrated_corpus_accepts_writes(tmp_path: Path) -> None:
    """Write to a migrated version-four corpus without a missing-column error."""
    db_path = tmp_path / 'v4.db'
    write_legacy_corpus(db_path, V4_PAPER_FIELDS, version=4)

    with open_corpus(db_path) as conn:
        corpus.upsert_paper(conn, sample_paper())
        rows = corpus.paper_rows(conn)

    assert len(rows) == 2
    assert {row['paper_id'] for row in rows} == {'legacy:1', 'demo:1'}


def test_migrated_corpus_creates_the_enrichment_index(tmp_path: Path) -> None:
    """Create the enrichment status index after the migration adds its column."""
    db_path = tmp_path / 'v4.db'
    write_legacy_corpus(db_path, V4_PAPER_FIELDS, version=4)

    with open_corpus(db_path) as conn:
        indexes = {
            row['name']
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
        }

    assert 'idx_papers_enrichment' in indexes


def test_corpus_migration_is_idempotent(tmp_path: Path) -> None:
    """Reopen a migrated corpus without altering its columns again."""
    db_path = tmp_path / 'v4.db'
    write_legacy_corpus(db_path, V4_PAPER_FIELDS, version=4)

    with open_corpus(db_path) as conn:
        first = [row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()]
    with open_corpus(db_path) as conn:
        second = [row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()]

    assert first == second


def test_corpus_adds_parallel_fields_to_early_search_history_schema(tmp_path: Path) -> None:
    """Add opt-in parallel settings to a prerelease version-eleven corpus."""
    db_path = tmp_path / 'early-v11.db'
    with contextlib.closing(sqlite3.connect(db_path)) as conn, conn:
        conn.executescript(
            """
            PRAGMA user_version = 11;
            CREATE TABLE search_runs (
                search_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                requested_source TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                requested_count INTEGER NOT NULL,
                store_abstract INTEGER NOT NULL,
                enrich INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_count INTEGER NOT NULL DEFAULT 0,
                papers_added INTEGER NOT NULL DEFAULT 0,
                papers_updated INTEGER NOT NULL DEFAULT 0,
                abstracts_stored INTEGER NOT NULL DEFAULT 0,
                source_results_json TEXT NOT NULL DEFAULT '{}',
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            """
        )

    with open_corpus(db_path) as conn:
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(search_runs)')}

    assert {'parallel', 'workers'} <= columns


def test_enrichment_columns_are_absent_from_paper_fields() -> None:
    """Keep enrichment columns outside the normalized paper field set."""
    assert not set(corpus.ENRICHMENT_COLUMNS) & set(corpus.PAPER_FIELDS)
    assert 'enrichment_status' in corpus.PIPELINE_COLUMNS
    assert set(corpus.ENRICHMENT_COLUMNS) <= set(corpus._expected_paper_columns())


def test_upsert_paper_does_not_clear_enrichment_columns(tmp_path: Path) -> None:
    """Preserve enrichment values through every ordinary corpus write path."""
    db_path = tmp_path / 'corpus.db'
    with open_corpus(db_path) as conn:
        corpus.upsert_paper(conn, sample_paper())
        conn.execute(
            "UPDATE papers SET publisher = 'Demo Press', cited_by_count = 7 WHERE paper_id = 'demo:1'"
        )
        conn.commit()

        stored = corpus.paper_rows(conn)[0]
        stored['pdf_download_status'] = 'succeeded'
        corpus.upsert_paper(conn, stored)
        corpus.upsert_papers(conn, [{'doi': '10.1000/demo:1', 'title': 'Other', 'sources': 'core'}])
        corpus.upsert_paper(conn, corpus.normalize_paper({'doi': '10.1000/demo:1'}))

        refreshed = corpus.paper_rows(conn)[0]

    assert refreshed['publisher'] == 'Demo Press'
    assert refreshed['cited_by_count'] == 7


def test_corpus_supports_uncompressed_blobs_and_missing_assets(tmp_path: Path) -> None:
    """Test uncompressed blob storage and missing asset lookup."""
    pdf = b'%PDF-1.4\n% dummy pdf\n'

    with open_corpus(tmp_path / 'corpus.db') as conn:
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
    with open_corpus(tmp_path / 'corpus.db') as conn:
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
            metadata={'version': 2},
        )
        corpus.add_asset(conn, paper, 'full text', role='text', kind='text', mime_type='text/plain')
        assets = corpus.latest_assets(conn, ['abstract'])

    assert set(assets) == {('demo:assets', 'abstract')}
    assert assets[('demo:assets', 'abstract')]['content'] == b'newer abstract'
    assert assets[('demo:assets', 'abstract')]['source'] == 'second'
    assert assets[('demo:assets', 'abstract')]['metadata'] == {'version': 2}


def test_corpus_rejects_unknown_compression_and_decompression_codecs(tmp_path: Path) -> None:
    """Test corpus compression validation."""
    with open_corpus(tmp_path / 'corpus.db') as conn:
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
    with open_corpus(tmp_path / 'papers.db') as conn:
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
    with open_corpus(tmp_path / 'papers.db') as conn:
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


def enrichment_update(paper_id: str, **values: Any) -> dict[str, Any]:
    """Build a complete enrichment update mapping with empty defaults.

    Parameters
    ----------
    paper_id : str
        Paper the update applies to.
    **values : Any
        Enrichment fields overriding the empty defaults.

    Returns
    -------
    dict[str, Any]
        Mapping covering every enrichment update parameter.
    """
    update = {field: '' for field in corpus.enrichment_update_fields()}
    update['paper_id'] = paper_id
    update['enrichment_status'] = 'succeeded'
    update['updated_at'] = corpus.utc_now()
    update.update(values)
    return update


def test_enrichment_candidates_paginates_by_rowid(tmp_path: Path) -> None:
    """Page pending enrichment candidates by rowid without repeating rows."""
    with open_corpus(tmp_path / 'corpus.db') as conn:
        for index in range(3):
            corpus.upsert_paper(conn, sample_paper(f'demo:{index}'))

        first = corpus.enrichment_candidates(conn, limit=2)
        second = corpus.enrichment_candidates(conn, after_rowid=first[-1]['rowid'], limit=2)

    assert [row['paper_id'] for row in first] == ['demo:0', 'demo:1']
    assert [row['paper_id'] for row in second] == ['demo:2']


def test_enrichment_candidates_ignore_succeeded_rows_by_default(tmp_path: Path) -> None:
    """Skip already enriched papers unless their status is requested."""
    with open_corpus(tmp_path / 'corpus.db') as conn:
        corpus.upsert_paper(conn, sample_paper('demo:1'))
        corpus.write_enrichment(conn, [enrichment_update('demo:1', enriched_at='2026-01-01T00:00:00+00:00')])

        pending = corpus.enrichment_candidates(conn)
        forced = corpus.enrichment_candidates(conn, statuses=('pending', 'succeeded'))
        stale = corpus.enrichment_candidates(conn, refreshed_before='2026-06-01T00:00:00+00:00')

    assert pending == []
    assert [row['paper_id'] for row in forced] == ['demo:1']
    assert [row['paper_id'] for row in stale] == ['demo:1']


def test_write_enrichment_fills_only_empty_core_columns(tmp_path: Path) -> None:
    """Fill empty core columns while leaving populated ones untouched."""
    with open_corpus(tmp_path / 'corpus.db') as conn:
        corpus.upsert_paper(conn, {'paper_id': 'demo:1', 'title': 'Curated title'})
        corpus.write_enrichment(conn, [enrichment_update(
            'demo:1', title='Provider title', journal='Provider journal',
            publisher='Demo Press', cited_by_count=42, is_oa=1,
        )])
        row = corpus.paper_rows(conn)[0]

    assert row['title'] == 'Curated title'
    assert row['journal'] == 'Provider journal'
    assert row['publisher'] == 'Demo Press'
    assert row['cited_by_count'] == 42
    assert row['enrichment_status'] == 'succeeded'


def test_write_enrichment_replaces_rather_than_duplicates_child_rows(tmp_path: Path) -> None:
    """Replace a paper's child rows on every enrichment write."""
    author = {'paper_id': 'demo:1', 'author_position': 0, 'affiliation_rank': 0,
              'display_name': 'A. Author', 'orcid': '0000-0002-1825-0097', 'source': 'openalex'}
    subject = {'paper_id': 'demo:1', 'scheme': 'topic', 'subject_id': 'T1',
               'display_name': 'Batteries', 'score': 0.9, 'is_primary': 1, 'source': 'openalex'}
    reference = {'paper_id': 'demo:1', 'source': 'crossref', 'reference_rank': 0,
                 'referenced_doi': '10.1000/ref'}

    with open_corpus(tmp_path / 'corpus.db') as conn:
        corpus.upsert_paper(conn, sample_paper('demo:1'))
        corpus.write_enrichment(conn, [enrichment_update('demo:1')],
                                authors=[author], subjects=[subject], references=[reference])
        corpus.write_enrichment(conn, [enrichment_update('demo:1')],
                                authors=[author], subjects=[subject], references=[reference])

        authors = corpus.paper_authors(conn, 'demo:1')
        subjects = corpus.paper_subjects(conn, 'demo:1', scheme='topic')
        references = corpus.paper_references(conn, 'demo:1', source='crossref')

    assert len(authors) == 1
    assert authors[0]['orcid'] == '0000-0002-1825-0097'
    assert authors[0]['position_label'] == ''
    assert [subject['score'] for subject in subjects] == [0.9]
    assert [reference['referenced_doi'] for reference in references] == ['10.1000/ref']


def test_write_enrichment_rolls_back_a_failed_batch(tmp_path: Path) -> None:
    """Leave the corpus unchanged when a child insert fails mid-batch."""
    with open_corpus(tmp_path / 'corpus.db') as conn:
        corpus.upsert_paper(conn, sample_paper('demo:1'))
        with pytest.raises(sqlite3.IntegrityError):
            corpus.write_enrichment(
                conn,
                [enrichment_update('demo:1', publisher='Demo Press')],
                authors=[{'paper_id': 'missing:1', 'author_position': 0,
                          'affiliation_rank': 0, 'source': 'openalex'}],
            )
        row = corpus.paper_rows(conn)[0]

    assert row['publisher'] is None
    assert row['enrichment_status'] == 'pending'


def test_child_rows_cascade_when_a_paper_is_deleted(tmp_path: Path) -> None:
    """Remove enrichment child rows when their paper is deleted."""
    with open_corpus(tmp_path / 'corpus.db') as conn:
        corpus.upsert_paper(conn, sample_paper('demo:1'))
        corpus.write_enrichment(
            conn,
            [enrichment_update('demo:1')],
            authors=[{'paper_id': 'demo:1', 'author_position': 0, 'affiliation_rank': 0,
                      'display_name': 'A. Author', 'source': 'openalex'}],
        )
        conn.execute("DELETE FROM papers WHERE paper_id = 'demo:1'")
        conn.commit()

        assert corpus.paper_authors(conn, 'demo:1') == []


def test_set_enrichment_status_rejects_unknown_values(tmp_path: Path) -> None:
    """Store a supported enrichment status and reject anything else."""
    with open_corpus(tmp_path / 'corpus.db') as conn:
        corpus.upsert_paper(conn, sample_paper('demo:1'))
        assert corpus.set_enrichment_status(conn, ['demo:1'], 'unresolved') == 1
        assert corpus.paper_rows(conn)[0]['enrichment_status'] == 'unresolved'
        with pytest.raises(ValueError):
            corpus.set_enrichment_status(conn, ['demo:1'], 'nonsense')


def test_enrichment_stats_reports_status_and_child_counts(tmp_path: Path) -> None:
    """Report per-status paper counts alongside stored enrichment rows."""
    with open_corpus(tmp_path / 'corpus.db') as conn:
        corpus.upsert_paper(conn, sample_paper('demo:1'))
        corpus.upsert_paper(conn, sample_paper('demo:2'))
        corpus.write_enrichment(
            conn,
            [enrichment_update('demo:1', is_oa=1, is_retracted=1)],
            authors=[{'paper_id': 'demo:1', 'author_position': 0, 'affiliation_rank': 0,
                      'orcid': '0000-0002-1825-0097', 'source': 'openalex'}],
            subjects=[{'paper_id': 'demo:1', 'scheme': 'sdg', 'subject_id': '7',
                       'source': 'openalex'}],
            references=[{'paper_id': 'demo:1', 'source': 'openalex', 'reference_rank': 0,
                         'referenced_openalex_id': 'W1'}],
        )
        stats = corpus.enrichment_stats(conn)

    assert stats['papers_succeeded'] == 1
    assert stats['papers_pending'] == 1
    assert stats['papers_open_access'] == 1
    assert stats['papers_retracted'] == 1
    assert stats['author_records'] == 1
    assert stats['authors_with_orcid'] == 1
    assert stats['subject_records'] == 1
    assert stats['reference_records'] == 1


def test_search_history_records_runs_outcomes_and_paper_provenance(tmp_path: Path) -> None:
    """Persist search settings, provider outcomes, and contributing paper links."""
    db_path = tmp_path / 'searches.db'
    with open_corpus(db_path) as conn:
        search_id = corpus.begin_search_run(
            conn,
            'solid electrolyte',
            'all',
            ['core', 'openalex'],
            25,
            store_abstract=True,
            enrich=False,
            parallel=True,
            workers=2,
        )
        corpus.upsert_paper(conn, sample_paper())
        assert corpus.add_search_result(conn, search_id, sample_paper(), 'core', 3) is True
        assert corpus.add_search_result(conn, search_id, {'title': 'missing'}, 'openalex', 0) is False
        corpus.finish_search_run(
            conn,
            search_id,
            'partial',
            {
                'core': {'status': 'completed', 'result_count': 1},
                'openalex': {'status': 'failed', 'result_count': 0, 'error': 'unavailable'},
            },
            result_count=1,
            papers_added=1,
            abstracts_stored=1,
        )
        history = corpus.search_history(conn)
        limited = corpus.search_history(conn, limit=0)
        paper_history = corpus.paper_search_history(conn, 'demo:1')

        with pytest.raises(ValueError, match='status must be'):
            corpus.finish_search_run(conn, search_id, 'running', {})

    assert limited == []
    assert len(history) == 1
    assert history[0]['search_id'] == search_id
    assert history[0]['query'] == 'solid electrolyte'
    assert history[0]['sources'] == ['core', 'openalex']
    assert history[0]['store_abstract'] is True
    assert history[0]['enrich'] is False
    assert history[0]['parallel'] is True
    assert history[0]['workers'] == 2
    assert history[0]['status'] == 'partial'
    assert history[0]['papers_added'] == 1
    assert history[0]['source_results']['openalex']['error'] == 'unavailable'
    assert paper_history == [{
        'search_id': search_id,
        'source': 'core',
        'result_rank': 3,
        'query': 'solid electrolyte',
        'started_at': history[0]['started_at'],
    }]


V5_PAPER_FIELDS = V4_PAPER_FIELDS + ['enrichment_status']
V6_PAPER_FIELDS = V5_PAPER_FIELDS + ['pmid', 'pmcid']
V7_PAPER_FIELDS = V6_PAPER_FIELDS + ['arxiv_id']
V8_PAPER_FIELDS = V7_PAPER_FIELDS + ['medrxiv_doi']
V9_PAPER_FIELDS = V8_PAPER_FIELDS + ['biorxiv_doi']


def test_corpus_migrates_version_five_pubmed_columns_without_losing_rows(tmp_path: Path) -> None:
    """Add the PubMed identifier columns to an existing version-five corpus."""
    db_path = tmp_path / 'v5.db'
    write_legacy_corpus(db_path, V5_PAPER_FIELDS, version=5)

    with open_corpus(db_path) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()}
        rows = corpus.paper_rows(conn)

    assert version == corpus.SCHEMA_VERSION == 12
    assert {'pmid', 'pmcid'} <= columns
    assert len(rows) == 1
    assert rows[0]['title'] == 'Legacy paper'
    assert rows[0]['pmid'] is None
    assert rows[0]['pmcid'] is None


def test_corpus_migrates_version_six_arxiv_column_without_losing_rows(tmp_path: Path) -> None:
    """Add the arXiv identifier column to an existing version-six corpus."""
    db_path = tmp_path / 'v6.db'
    write_legacy_corpus(db_path, V6_PAPER_FIELDS, version=6)

    with open_corpus(db_path) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()}
        rows = corpus.paper_rows(conn)

    assert version == corpus.SCHEMA_VERSION == 12
    assert 'arxiv_id' in columns
    assert len(rows) == 1
    assert rows[0]['title'] == 'Legacy paper'
    assert rows[0]['arxiv_id'] is None


def test_corpus_migrates_version_seven_medrxiv_column_without_losing_rows(tmp_path: Path) -> None:
    """Add the medRxiv DOI column to an existing version-seven corpus."""
    db_path = tmp_path / 'v7.db'
    write_legacy_corpus(db_path, V7_PAPER_FIELDS, version=7)

    with open_corpus(db_path) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()}
        rows = corpus.paper_rows(conn)

    assert version == corpus.SCHEMA_VERSION == 12
    assert 'medrxiv_doi' in columns
    assert len(rows) == 1
    assert rows[0]['title'] == 'Legacy paper'
    assert rows[0]['medrxiv_doi'] is None


def test_corpus_migrates_version_eight_biorxiv_column_without_losing_rows(tmp_path: Path) -> None:
    """Add the bioRxiv DOI column to an existing version-eight corpus."""
    db_path = tmp_path / 'v8.db'
    write_legacy_corpus(db_path, V8_PAPER_FIELDS, version=8)

    with open_corpus(db_path) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()}
        rows = corpus.paper_rows(conn)

    assert version == corpus.SCHEMA_VERSION == 12
    assert 'biorxiv_doi' in columns
    assert len(rows) == 1
    assert rows[0]['title'] == 'Legacy paper'
    assert rows[0]['biorxiv_doi'] is None


def test_corpus_migrates_version_nine_chemrxiv_column_without_losing_rows(tmp_path: Path) -> None:
    """Add the chemRxiv DOI column to an existing version-nine corpus."""
    db_path = tmp_path / 'v9.db'
    write_legacy_corpus(db_path, V9_PAPER_FIELDS, version=9)

    with open_corpus(db_path) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(papers)').fetchall()}
        rows = corpus.paper_rows(conn)

    assert version == corpus.SCHEMA_VERSION == 12
    assert 'chemrxiv_doi' in columns
    assert len(rows) == 1
    assert rows[0]['title'] == 'Legacy paper'
    assert rows[0]['chemrxiv_doi'] is None


def test_fallback_paper_id_prefers_doi_then_pmid_then_arxiv_then_core() -> None:
    """Choose the most portable identifier available for a paper without one."""
    assert corpus._fallback_paper_id({'doi': '10.1/x', 'pmid': '1', 'core_id': '2'}) == 'doi:10.1/x'
    assert corpus._fallback_paper_id({'pmid': '31234567', 'core_id': '2'}) == 'pmid:31234567'
    assert corpus._fallback_paper_id(
        {'arxiv_id': '2301.12345', 'core_id': '2'}) == 'arxiv:2301.12345'
    assert corpus._fallback_paper_id(
        {'medrxiv_doi': '10.1101/2024.03.01.24303596', 'core_id': '2'}
    ) == 'doi:10.1101/2024.03.01.24303596'
    assert corpus._fallback_paper_id(
        {'biorxiv_doi': '10.1101/2023.12.01.569634', 'core_id': '2'}
    ) == 'doi:10.1101/2023.12.01.569634'
    # The version suffix is part of a chemRxiv DOI, so it must survive here too.
    assert corpus._fallback_paper_id(
        {'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1', 'core_id': '2'}
    ) == 'doi:10.26434/chemrxiv.15007737/v1'
    assert corpus._fallback_paper_id({'core_id': '2'}) == 'core:2'
    assert corpus._fallback_paper_id({'title': 'A paper'}).startswith('paper:')


def test_papers_match_on_a_shared_pubmed_identifier() -> None:
    """Treat rows sharing a PubMed identifier as the same publication."""
    existing = {'paper_id': 'pmid:31234567', 'pmid': '31234567', 'title': 'One'}
    incoming = {'paper_id': 'doi:10.1/x', 'pmid': '31234567', 'title': 'Different title'}

    assert corpus._papers_match(existing, incoming)
    assert corpus._papers_match({'pmcid': 'PMC1', 'paper_id': 'a'}, {'pmcid': 'PMC1', 'paper_id': 'b'})


def test_papers_match_on_a_shared_arxiv_identifier() -> None:
    """Treat rows sharing an arXiv identifier as the same publication."""
    existing = {'paper_id': 'arxiv:2301.12345', 'arxiv_id': '2301.12345', 'title': 'One'}
    incoming = {'paper_id': 'doi:10.1/x', 'arxiv_id': '2301.12345', 'title': 'Different title'}

    assert corpus._papers_match(existing, incoming)


def test_papers_match_on_a_shared_medrxiv_identifier() -> None:
    """Treat rows sharing a medRxiv DOI as the same publication."""
    existing = {'paper_id': 'doi:10.1101/2024.03.01.24303596',
                'medrxiv_doi': '10.1101/2024.03.01.24303596', 'title': 'One'}
    incoming = {'paper_id': 'doi:10.1/x', 'medrxiv_doi': '10.1101/2024.03.01.24303596',
                'title': 'Different title'}

    assert corpus._papers_match(existing, incoming)


def test_papers_match_on_a_shared_biorxiv_identifier() -> None:
    """Treat rows sharing a bioRxiv DOI as the same publication."""
    existing = {'paper_id': 'doi:10.1101/2023.12.01.569634',
                'biorxiv_doi': '10.1101/2023.12.01.569634', 'title': 'One'}
    incoming = {'paper_id': 'doi:10.1/x', 'biorxiv_doi': '10.1101/2023.12.01.569634',
                'title': 'Different title'}

    assert corpus._papers_match(existing, incoming)


def test_papers_match_on_a_shared_chemrxiv_identifier() -> None:
    """Treat rows sharing a chemRxiv DOI as the same publication."""
    existing = {'paper_id': 'doi:10.26434/chemrxiv.15007737/v1',
                'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1', 'title': 'One'}
    incoming = {'paper_id': 'doi:10.1/x', 'chemrxiv_doi': '10.26434/chemrxiv.15007737/v1',
                'title': 'Different title'}

    assert corpus._papers_match(existing, incoming)


def test_upsert_merges_an_arxiv_preprint_into_a_matching_doi_row(tmp_path: Path) -> None:
    """Fold a preprint carrying a deposited DOI into the published row."""
    db_path = tmp_path / 'papers.db'
    published = {'paper_id': 'doi:10.1/x', 'doi': '10.1/x', 'title': 'Garnet conductivity',
                 'journal': 'Phys. Rev. B', 'publication_date': '2024-02-01',
                 'sources': 'openalex'}
    preprint = {'paper_id': 'doi:10.1/x', 'doi': '10.1/x', 'title': 'Garnet conductivity',
                'arxiv_id': '2301.12345', 'publication_date': '2023-01-30',
                'pdf_url': 'https://arxiv.org/pdf/2301.12345', 'sources': 'arxiv'}

    with open_corpus(db_path) as conn:
        corpus.upsert_papers(conn, [published])
        added, updated = corpus.upsert_papers(conn, [preprint])
        rows = corpus.paper_rows(conn)

    assert (added, updated) == (0, 1)
    assert len(rows) == 1
    assert rows[0]['paper_id'] == 'doi:10.1/x'
    assert rows[0]['arxiv_id'] == '2301.12345'
    assert rows[0]['sources'] == 'openalex;arxiv'
    # The published journal and date win; the preprint only fills what was empty.
    assert rows[0]['journal'] == 'Phys. Rev. B'
    assert rows[0]['publication_date'] == '2024-02-01'


def test_upsert_keeps_a_preprint_separate_when_its_year_differs(tmp_path: Path) -> None:
    """Record the accepted limit: a DOI-less preprint posted in an earlier year.

    Without a deposited DOI the only remaining rule is title and year, so a
    preprint that crossed a calendar boundary before publication is stored as
    its own row. This is asserted so the behaviour stays a known trade-off
    rather than becoming an unnoticed regression.
    """
    db_path = tmp_path / 'papers.db'
    published = {'paper_id': 'doi:10.1/x', 'doi': '10.1/x', 'title': 'Garnet conductivity',
                 'publication_date': '2024-02-01', 'sources': 'openalex'}
    preprint = {'paper_id': 'arxiv:2301.12345', 'arxiv_id': '2301.12345',
                'title': 'Garnet conductivity', 'publication_date': '2023-01-30',
                'sources': 'arxiv'}

    with open_corpus(db_path) as conn:
        corpus.upsert_papers(conn, [published])
        added, updated = corpus.upsert_papers(conn, [preprint])
        rows = corpus.paper_rows(conn)

    assert (added, updated) == (1, 0)
    assert len(rows) == 2
    assert not corpus._papers_match({'pmid': '1', 'paper_id': 'a'}, {'pmid': '2', 'paper_id': 'b'})


def test_upsert_merges_a_pubmed_row_into_a_matching_doi_row(tmp_path: Path) -> None:
    """Merge a PubMed record into an existing row rather than duplicating it."""
    db_path = tmp_path / 'papers.db'
    with open_corpus(db_path) as conn:
        corpus.upsert_papers(conn, [{'paper_id': 'doi:10.1234/x', 'doi': '10.1234/x',
                                     'sources': 'openalex', 'title': 'Shared paper'}])
        added, updated = corpus.upsert_papers(conn, [
            {'paper_id': 'pmid:31234567', 'doi': '10.1234/x', 'pmid': '31234567',
             'pmcid': 'PMC9876543', 'sources': 'pubmed', 'title': 'Shared paper'}])
        rows = corpus.paper_rows(conn)

    assert (added, updated) == (0, 1)
    assert len(rows) == 1
    assert rows[0]['paper_id'] == 'doi:10.1234/x'
    assert rows[0]['pmid'] == '31234567'
    assert rows[0]['pmcid'] == 'PMC9876543'
    assert 'pubmed' in rows[0]['sources']


def test_write_enrichment_scopes_child_deletes_to_the_queried_sources(tmp_path: Path) -> None:
    """Replace only the queried provider's child rows and keep the others."""
    db_path = tmp_path / 'papers.db'

    def update(paper_id: str) -> dict[str, Any]:
        """Return a minimal enrichment update for one paper."""
        fields = {field: '' for field in corpus.enrichment_update_fields()}
        fields.update({'paper_id': paper_id, 'enrichment_status': 'succeeded', 'updated_at': ''})
        return fields

    def subject(paper_id: str, scheme: str, subject_id: str, source: str) -> dict[str, Any]:
        """Return a minimal paper_subjects row."""
        return {'paper_id': paper_id, 'scheme': scheme, 'subject_id': subject_id,
                'display_name': subject_id, 'source': source}

    with open_corpus(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'doi:10.1/x', 'doi': '10.1/x'})
        corpus.write_enrichment(conn, [update('doi:10.1/x')],
                                subjects=[subject('doi:10.1/x', 'topic', 'T1', 'openalex')],
                                sources=['openalex'])
        corpus.write_enrichment(conn, [update('doi:10.1/x')],
                                subjects=[subject('doi:10.1/x', 'mesh', 'D1', 'pubmed')],
                                sources=['pubmed'])
        scoped = {(row['scheme'], row['source'])
                  for row in conn.execute('SELECT scheme, source FROM paper_subjects').fetchall()}

        corpus.write_enrichment(conn, [update('doi:10.1/x')],
                                subjects=[subject('doi:10.1/x', 'topic', 'T2', 'openalex')])
        unscoped = {(row['scheme'], row['source'])
                    for row in conn.execute('SELECT scheme, source FROM paper_subjects').fetchall()}

    assert scoped == {('topic', 'openalex'), ('mesh', 'pubmed')}
    assert unscoped == {('topic', 'openalex')}
