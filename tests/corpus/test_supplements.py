"""Verify supplementary storage, schema migration, and resumable extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperminertoolkit.corpus import database as corpus
from paperminertoolkit.corpus.supplements import (
    add_supplement, get_supplements, set_supplement_extraction_status,
)
from paperminertoolkit.workflows.utilities import progress_summary, reset


def test_supplements_preserve_article_and_distinguish_equal_filenames(tmp_path: Path) -> None:
    """Keep every attachment without replacing article text or a namesake file."""
    with corpus.connect(tmp_path / 'papers.db') as conn:
        paper = {'paper_id': 'paper:1', 'text_scrape_status': 'succeeded'}
        corpus.add_asset(conn, paper, 'article', 'text', 'text', 'text/plain')
        for index in range(2):
            add_supplement(conn, paper, f'table {index}', source='publisher',
                           source_url=f'https://example.org/{index}/data.csv',
                           original_filename='data.csv', mime_type='text/csv')
        assets = get_supplements(conn, paper['paper_id'])
        assert len(assets) == 2
        assert {a['content'] for a in assets} == {b'table 0', b'table 1'}
        assert len({a['original_filename'] for a in assets}) == 2
        assert corpus.get_asset(conn, paper['paper_id'], 'text')['content'] == b'article'
        assert corpus.paper_rows(conn)[0]['text_scrape_status'] == 'succeeded'
        assert all('content' not in asset for asset in get_supplements(
            conn, paper['paper_id'], include_content=False))
        stats = corpus.corpus_stats(conn)
        assert stats['papers_with_supplements'] == 1
        assert stats['supplements'] == 2


def test_refresh_keeps_unchanged_checkpoints_and_resets_changed_attachment(tmp_path: Path) -> None:
    """Preserve completed SI work unless a file at its source actually changes."""
    with corpus.connect(tmp_path / 'papers.db') as conn:
        paper = {'paper_id': 'paper:1'}
        first = add_supplement(conn, paper, b'first', source_url='https://example.org/si.txt')
        second = add_supplement(conn, paper, b'second', source_url='https://example.org/other.txt')
        for blob in (first, second):
            assert set_supplement_extraction_status(
                conn, paper['paper_id'], blob, 'text', 'succeeded', members={'part': 'succeeded'})
        assert add_supplement(conn, paper, b'first',
                              source_url='https://example.org/si.txt') == first
        assert get_supplements(conn, paper['paper_id'])[1]['metadata']['text_extraction_status'] == 'succeeded'
        refreshed = add_supplement(conn, paper, b'changed',
                                  source_url='https://example.org/si.txt')
        assert refreshed != first
        states = {a['blob_id']: a['metadata'] for a in get_supplements(conn, paper['paper_id'])}
        assert len(states) == 2
        assert states[refreshed]['text_extraction_status'] == 'pending'
        assert 'text_extraction_members' not in states[refreshed]
        assert states[second]['text_extraction_status'] == 'succeeded'


def test_si_progress_and_reset_include_per_file_checkpoints(tmp_path: Path) -> None:
    """Reset both aggregate SI status and individual file checkpoints."""
    path = tmp_path / 'papers.db'
    with corpus.connect(path) as conn:
        paper = {'paper_id': 'paper:1', 'supplement_download_status': 'partial'}
        blob = add_supplement(conn, paper, b'text', original_filename='si.txt')
        set_supplement_extraction_status(conn, paper['paper_id'], blob, 'text', 'failed',
                                        error='temporary model failure', members={'a': 'succeeded'})
    summary = progress_summary(path)
    assert summary['papers_with_supplements'] == summary['supplements'] == 1
    assert summary['pipeline']['supplement_download_status']['partial'] == 1
    reset(path)
    with corpus.connect(path) as conn:
        metadata = get_supplements(conn, paper['paper_id'])[0]['metadata']
        assert metadata['text_extraction_status'] == 'pending'
        assert metadata['text_extraction_error'] == ''
        assert metadata['text_extraction_members'] == {}
        assert corpus.paper_rows(conn)[0]['supplement_download_status'] == 'pending'


def test_schema_migration_preserves_article_assets_and_adds_si_stages(tmp_path: Path) -> None:
    """Upgrade a previous corpus without changing its paper or article bytes."""
    path = tmp_path / 'papers.db'
    with corpus.connect(path) as conn:
        corpus.add_asset(conn, {'paper_id': 'old'}, b'article', 'text', 'text', 'text/plain')
        for column in ('supplement_download_status', 'supplement_text_scrape_status',
                       'supplement_image_scrape_status'):
            conn.execute(f'ALTER TABLE papers DROP COLUMN {column}')
        conn.execute('PRAGMA user_version = 12')
        conn.commit()
    with corpus.connect(path) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == corpus.SCHEMA_VERSION
        assert corpus.get_asset(conn, 'old', 'text')['content'] == b'article'
        assert corpus.paper_rows(conn)[0]['supplement_download_status'] == 'pending'


def test_checkpoint_rejects_invalid_stage_and_ignores_missing_assets(tmp_path: Path) -> None:
    """Reject malformed updates without creating phantom supplementary files."""
    with corpus.connect(tmp_path / 'papers.db') as conn:
        assert not set_supplement_extraction_status(conn, 'missing', 'missing', 'text', 'failed')
        with pytest.raises(ValueError, match='stage'):
            set_supplement_extraction_status(conn, 'missing', 'missing', 'unknown', 'failed')
