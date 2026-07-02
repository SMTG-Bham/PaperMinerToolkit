"""Unit tests for paperscraper.corpus.

This module tests the standalone SQLite corpus layer for storing paper metadata,
compressed blobs, deduplicated content, paper asset links, and corpus storage
statistics without touching the command-line workflow.
"""

import gzip
from pathlib import Path

import pytest

import paperscraper.corpus as corpus


def sample_paper(paper_id='demo:1'):
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


def test_corpus_stores_deduplicated_compressed_assets_and_reads_them_back(tmp_path):
    """
    Test SQLite corpus storage for paper assets.

    This function performs the following steps:
    1. Creates a temporary SQLite corpus database.
    2. Stores two paper text assets with identical content.
    3. Reads back one stored text asset and corpus statistics.

    Asserts:
        - Both paper rows are stored.
        - Identical content is deduplicated into one blob.
        - Stored gzip content is smaller than the original repeated text.
        - Readback content is decompressed to the original bytes.
    """
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
    assert stats['blobs'] == 1
    assert stats['stored_size'] < stats['original_size']
    assert stats['savings_fraction'] > 0


def test_corpus_supports_uncompressed_blobs_and_missing_assets(tmp_path):
    """
    Test uncompressed blob storage and missing asset lookup.

    This function performs the following steps:
    1. Creates a temporary SQLite corpus database.
    2. Stores a dummy PDF blob without compression.
    3. Reads back the PDF asset and asks for a missing text asset.

    Asserts:
        - Uncompressed blobs are read back unchanged.
        - Missing paper-role assets return `None`.
    """
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


def test_corpus_rejects_unknown_compression_and_decompression_codecs(tmp_path):
    """
    Test corpus compression validation.

    This function performs the following steps:
    1. Creates a temporary SQLite corpus database.
    2. Attempts to store a blob with an unsupported compression codec.
    3. Attempts to decompress bytes with an unsupported codec.

    Asserts:
        - Unknown storage compression codecs raise `ValueError`.
        - Unknown decompression codecs raise `ValueError`.
    """
    with corpus.connect(tmp_path / 'corpus.db') as conn:
        with pytest.raises(ValueError, match='compression must be one of'):
            corpus.store_blob(conn, b'data', kind='text', mime_type='text/plain', compression='brotli')
    with pytest.raises(ValueError, match='Unsupported blob compression'):
        corpus._decompress(gzip.compress(b'data'), 'brotli')


def test_corpus_serializes_metadata_and_prepares_path_or_iterable_content(tmp_path):
    """
    Test corpus helper conversion branches.

    This function performs the following steps:
    1. Serializes missing and pre-serialized metadata values.
    2. Writes a temporary text file and prepares it as blob content.
    3. Prepares byte values from an iterable of integer byte values.

    Asserts:
        - Missing metadata becomes an empty JSON object.
        - Existing JSON text is passed through unchanged.
        - Path inputs are read as bytes.
        - Iterable byte values are converted to bytes.
    """
    text_path = tmp_path / 'paper.txt'
    text_path.write_text('paper text')

    assert corpus._json_dumps(None) == '{}'
    assert corpus._json_dumps('{"already": true}') == '{"already": true}'
    assert corpus._prepare_content(Path(text_path)) == b'paper text'
    assert corpus._prepare_content([65, 66, 67]) == b'ABC'
