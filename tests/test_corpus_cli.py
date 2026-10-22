"""Exercise read-only corpus inspection and safe asset export through Click."""

from __future__ import annotations

import gzip
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from paperminertoolkit import _corpus_cli as cli
from paperminertoolkit.corpus import database as corpus
from paperminertoolkit.corpus.supplements import add_supplement
from tests.corpus_helpers import open_corpus


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    """Seed unsorted papers, compressed assets, and multi-provider provenance."""
    path = tmp_path / 'corpus ?# café.db'
    with open_corpus(path) as conn:
        for suffix in ['z', 'a', 'm']:
            paper = {'paper_id': f'paper:{suffix}', 'title': f'Paper {suffix}',
                     'metadata': {'provider_value': suffix},
                     'text_download_status': 'succeeded' if suffix == 'a' else 'pending'}
            corpus.upsert_paper(conn, paper)
            if suffix == 'a':
                for role, content, filename in [
                    ('abstract', 'Résumé of a paper.', 'abstract.txt'),
                    ('text', 'Earlier text.', 'old.txt'),
                    ('text', 'Latest text: αβ\nwith another line.\n', 'new.txt'),
                    ('pdf', b'%PDF-1.7\n\x00\xffbinary\n', 'paper.pdf'),
                ]:
                    corpus.add_asset(conn, paper, content, role, role,
                                     'application/pdf' if role == 'pdf' else 'text/plain',
                                     source='test', original_filename=filename,
                                     metadata={'fixture': True})
        conn.execute('UPDATE papers SET enrichment_json = ? WHERE paper_id = ?',
                     ('{"provider": "test"}', 'paper:a'))
        search_id = corpus.begin_search_run(conn, 'demo', 'all', ['arxiv', 'core'], 2)
        for rank, paper_id in enumerate(['paper:a', 'paper:z']):
            for source in ['arxiv', 'core']:
                corpus.add_search_result(conn, search_id, {'paper_id': paper_id}, source, rank)
        corpus.finish_search_run(conn, search_id, 'completed', {}, result_count=4)
    return path


def test_list_paginates_sorted_rows_without_large_metadata(corpus_path: Path) -> None:
    """Use deterministic SQL pagination and retain metadata/status scalar fields."""
    runner = CliRunner()
    result = runner.invoke(cli.corpus_list, [str(corpus_path), '--json', '--limit', '1'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert result.stderr == ''
    assert payload['total'] == 3
    assert payload['offset'] == 0
    assert payload['limit'] == 1
    assert [paper['paper_id'] for paper in payload['papers']] == ['paper:a']
    assert payload['papers'][0]['text_download_status'] == 'succeeded'
    assert 'metadata_json' not in payload['papers'][0]
    assert 'enrichment_json' not in payload['papers'][0]

    second = runner.invoke(cli.corpus_list,
                           [str(corpus_path), '--json', '--offset', '1', '--limit', '1'])
    assert json.loads(second.stdout)['papers'][0]['paper_id'] == 'paper:m'
    empty = runner.invoke(cli.corpus_list, [str(corpus_path), '--json', '--offset', '10'])
    assert json.loads(empty.stdout) == {'papers': [], 'total': 3, 'limit': 20, 'offset': 10}


def test_list_limits_results_to_unique_search_matches(corpus_path: Path) -> None:
    """A paper matched by two providers occupies one row in a search selection."""
    runner = CliRunner()
    selected = runner.invoke(cli.corpus_list, [str(corpus_path), '--search-id', '1', '--json'])
    assert selected.exit_code == 0, selected.output
    payload = json.loads(selected.stdout)
    assert payload['total'] == 2
    assert [paper['paper_id'] for paper in payload['papers']] == ['paper:a', 'paper:z']
    missing = runner.invoke(cli.corpus_list, [str(corpus_path), '--search-id', '999', '--json'])
    assert missing.exit_code == 1
    assert missing.stdout == ''
    assert 'Search not found' in missing.stderr


@pytest.mark.parametrize('arguments', [
    ['--limit', '0'], ['--limit', '1001'], ['--offset', '-1'], ['--search-id', '0'],
])
def test_list_rejects_invalid_bounds(corpus_path: Path, arguments: list[str]) -> None:
    """Validate pagination and search identifiers before querying the corpus."""
    result = CliRunner().invoke(cli.corpus_list, [str(corpus_path), '--json', *arguments])
    assert result.exit_code == 2
    assert result.stdout == ''


def test_show_exposes_decoded_metadata_latest_assets_and_provenance(corpus_path: Path) -> None:
    """Inspection returns structured metadata without decompressed content bytes."""
    result = CliRunner().invoke(cli.corpus_show, [str(corpus_path), 'paper:a', '--json'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload) == {'paper', 'assets', 'searches', 'authors', 'subjects', 'references'}
    assert payload['paper']['metadata'] == {'provider_value': 'a'}
    assert payload['paper']['enrichment'] == {'provider': 'test'}
    assert 'metadata_json' not in payload['paper']
    assert [asset['role'] for asset in payload['assets']] == ['abstract', 'pdf', 'text']
    assert payload['assets'][-1]['original_filename'] == 'new.txt'
    assert payload['assets'][-1]['metadata'] == {'fixture': True}
    assert all('content' not in asset for asset in payload['assets'])
    assert len(payload['searches']) == 2
    assert {search['source'] for search in payload['searches']} == {'arxiv', 'core'}
    assert payload['authors'] == payload['subjects'] == payload['references'] == []


def test_human_inspection_and_missing_paper(corpus_path: Path) -> None:
    """Keep useful human output and clean errors for unrecognized identifiers."""
    runner = CliRunner()
    listing = runner.invoke(cli.corpus_list, [str(corpus_path)])
    assert listing.exit_code == 0
    assert 'paper:a' in listing.stdout
    shown = runner.invoke(cli.corpus_show, [str(corpus_path), 'paper:a'])
    assert shown.exit_code == 0, shown.output
    assert 'Title: Paper a' in shown.stdout
    assert 'text:' in shown.stdout
    missing = runner.invoke(cli.corpus_show, [str(corpus_path), 'unknown', '--json'])
    assert missing.exit_code == 1
    assert missing.stdout == ''
    assert 'Paper not found: unknown' in missing.stderr


def _supplements(corpus_path: Path) -> list[str]:
    """Store two independently identified SI files for the existing fixture paper."""
    with open_corpus(corpus_path) as conn:
        return [add_supplement(
            conn, {'paper_id': 'paper:a'}, content, source='test',
            source_url=f'https://example.test/{filename}', original_filename=filename,
            mime_type='text/plain', metadata={'label': f'Supplement {index}'},
        ) for index, (filename, content) in enumerate([
            ('supporting-1.txt', b'SI one'), ('supporting-2.txt', b'SI two'),
        ], 1)]


def test_show_lists_every_supplement_with_export_identifiers(corpus_path: Path) -> None:
    """Expose every SI link and checkpoint without loading its binary contents."""
    identifiers = _supplements(corpus_path)
    before = corpus_path.read_bytes()
    runner = CliRunner()
    result = runner.invoke(cli.corpus_show, [str(corpus_path), 'paper:a', '--json'])
    assert result.exit_code == 0, result.output
    assets = [asset for asset in json.loads(result.stdout)['assets'] if asset['role'] == 'supplement']
    assert [asset['blob_id'] for asset in assets] == identifiers
    assert [asset['metadata']['label'] for asset in assets] == ['Supplement 1', 'Supplement 2']
    assert all('content' not in asset for asset in assets)
    human = runner.invoke(cli.corpus_show, [str(corpus_path), 'paper:a'])
    assert human.exit_code == 0, human.output
    assert all(identifier in human.stdout for identifier in identifiers)
    assert 'supporting-1.txt' in human.stdout and 'supporting-2.txt' in human.stdout
    assert corpus_path.read_bytes() == before


def test_export_supplement_requires_unambiguous_selection(
    corpus_path: Path, tmp_path: Path,
) -> None:
    """Reject ambiguous and foreign identifiers before replacing output files."""
    identifiers = _supplements(corpus_path)
    output = tmp_path / 'si.txt'
    output.write_bytes(b'preserve output')
    before = corpus_path.read_bytes()
    runner = CliRunner()
    arguments = [str(corpus_path), 'paper:a', str(output), '--role', 'supplement', '--force', '--json']
    ambiguous = runner.invoke(cli.corpus_export, arguments)
    assert ambiguous.exit_code == 1
    assert '--asset-id' in ambiguous.stderr
    assert output.read_bytes() == b'preserve output'
    missing = runner.invoke(cli.corpus_export, [*arguments, '--asset-id', 'missing'])
    assert missing.exit_code == 1
    assert 'No stored supplement missing' in missing.stderr
    assert output.read_bytes() == b'preserve output'
    selected = runner.invoke(cli.corpus_export, [*arguments, '--asset-id', identifiers[1]])
    assert selected.exit_code == 0, selected.output
    assert output.read_bytes() == b'SI two'
    assert json.loads(selected.stdout)['asset_id'] == identifiers[1]
    assert json.loads(selected.stdout)['original_filename'] == 'supporting-2.txt'
    assert corpus_path.read_bytes() == before


def test_export_supplement_ignores_corrupt_unselected_files(
    corpus_path: Path, tmp_path: Path,
) -> None:
    """Decompress only the selected file, allowing unaffected SI to be recovered."""
    identifiers = _supplements(corpus_path)
    with open_corpus(corpus_path) as conn:
        conn.execute('UPDATE blobs SET content = ? WHERE blob_id = ?', (b'bad gzip', identifiers[0]))
    output = tmp_path / 'si.txt'
    result = CliRunner().invoke(cli.corpus_export, [
        str(corpus_path), 'paper:a', str(output), '--role', 'supplement',
        '--asset-id', identifiers[1],
    ])
    assert result.exit_code == 0, result.output
    assert output.read_bytes() == b'SI two'


def test_export_single_supplement_allows_duplicate_links_to_identical_content(
    corpus_path: Path, tmp_path: Path,
) -> None:
    """No selector is needed when every SI link refers to the same stored bytes."""
    with open_corpus(corpus_path) as conn:
        for source in ['publisher', 'figshare']:
            add_supplement(conn, {'paper_id': 'paper:a'}, b'identical SI', source=source,
                           original_filename='si.txt', mime_type='text/plain')
    output = tmp_path / 'si.txt'
    result = CliRunner().invoke(cli.corpus_export, [
        str(corpus_path), 'paper:a', str(output), '--role', 'supplement',
    ])
    assert result.exit_code == 0, result.output
    assert output.read_bytes() == b'identical SI'


def test_export_asset_selector_requires_supplement_role(corpus_path: Path, tmp_path: Path) -> None:
    """Reject a selector that would otherwise be silently ignored for an article asset."""
    output = tmp_path / 'text.txt'
    result = CliRunner().invoke(cli.corpus_export, [
        str(corpus_path), 'paper:a', str(output), '--asset-id', 'some-id',
    ])
    assert result.exit_code == 2
    assert '--asset-id requires --role supplement' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(('role', 'expected'), [
    ('abstract', 'Résumé of a paper.'.encode()),
    ('text', 'Latest text: αβ\nwith another line.\n'.encode()),
    ('pdf', b'%PDF-1.7\n\x00\xffbinary\n'),
])
def test_export_preserves_exact_latest_asset_bytes(
    corpus_path: Path, tmp_path: Path, role: str, expected: bytes,
) -> None:
    """Export compressed Unicode or binary assets as their exact original bytes."""
    output = tmp_path / f'export.{role}'
    result = CliRunner().invoke(cli.corpus_export,
                               [str(corpus_path), 'paper:a', str(output), '--role', role, '--json'])
    assert result.exit_code == 0, result.output
    assert output.read_bytes() == expected
    assert json.loads(result.stdout) == {
        'paper_id': 'paper:a', 'role': role, 'path': str(output), 'bytes': len(expected),
    }
    assert result.stderr == ''


def test_export_requires_force_and_valid_asset_before_overwriting(
    corpus_path: Path, tmp_path: Path,
) -> None:
    """Existing output survives refusal and lookup failures, even with force."""
    output = tmp_path / 'existing.txt'
    output.write_bytes(b'keep this')
    runner = CliRunner()
    refused = runner.invoke(cli.corpus_export, [str(corpus_path), 'paper:a', str(output), '--json'])
    assert refused.exit_code == 1
    assert '--force' in refused.stderr
    assert refused.stdout == ''
    assert output.read_bytes() == b'keep this'

    for paper_id, expected_error in [('paper:z', 'No stored text asset'), ('unknown', 'Paper not found')]:
        failed = runner.invoke(cli.corpus_export,
                               [str(corpus_path), paper_id, str(output), '--force', '--json'])
        assert failed.exit_code == 1
        assert expected_error in failed.stderr
        assert failed.stdout == ''
        assert output.read_bytes() == b'keep this'

    overwritten = runner.invoke(cli.corpus_export,
                                [str(corpus_path), 'paper:a', str(output), '--force'])
    assert overwritten.exit_code == 0, overwritten.output
    assert output.read_text() == 'Latest text: αβ\nwith another line.\n'
    assert 'Exported text' in overwritten.stdout


@pytest.mark.parametrize('damage', ['header', 'truncated', 'deflate'])
def test_corrupt_compressed_asset_does_not_overwrite_existing_output(
    corpus_path: Path, tmp_path: Path, damage: str,
) -> None:
    """Treat gzip/header/deflate failures as CLI errors before touching a forced target."""
    compressed = gzip.compress(b'original asset content')
    damaged = {
        'header': b'not a gzip file',
        'truncated': compressed[:-5],
        'deflate': compressed[:10] + b'\x07' + compressed[-8:],
    }[damage]
    with open_corpus(corpus_path) as conn:
        asset = corpus.get_asset_metadata(conn, 'paper:a', 'text')
        conn.execute('UPDATE blobs SET content = ? WHERE blob_id = ?',
                     (damaged, asset['blob_id']))
    destination = tmp_path / 'existing.txt'
    destination.write_bytes(b'keep existing output')
    result = CliRunner().invoke(cli.corpus_export,
                               [str(corpus_path), 'paper:a', str(destination), '--force', '--json'])
    assert result.exit_code == 1
    assert result.stdout == ''
    assert 'Error:' in result.stderr
    assert destination.read_bytes() == b'keep existing output'


@pytest.mark.parametrize('alias_kind', ['same', 'symlink', 'hardlink'])
def test_export_cannot_overwrite_source_database(
    corpus_path: Path, tmp_path: Path, alias_kind: str,
) -> None:
    """Reject the source corpus itself and both kinds of filesystem alias."""
    destination = corpus_path
    if alias_kind != 'same':
        destination = tmp_path / 'alias.db'
        if alias_kind == 'symlink':
            destination.symlink_to(corpus_path)
        else:
            destination.hardlink_to(corpus_path)
    original = corpus_path.read_bytes()
    result = CliRunner().invoke(cli.corpus_export,
                               [str(corpus_path), 'paper:a', str(destination), '--force', '--json'])
    assert result.exit_code == 1
    assert 'must not be the source corpus' in result.stderr
    assert result.stdout == ''
    assert corpus_path.read_bytes() == original


@pytest.mark.parametrize('suffix', ['-wal', '-shm', '-journal'])
@pytest.mark.parametrize('alias_kind', ['same', 'symlink', 'hardlink'])
def test_export_cannot_overwrite_database_companions(
    corpus_path: Path, tmp_path: Path, suffix: str, alias_kind: str,
) -> None:
    """Protect SQLite companion files and aliases before opening the database."""
    companion = Path(f'{corpus_path}{suffix}')
    companion.write_bytes(b'preserve database companion')
    destination = companion
    if alias_kind != 'same':
        destination = tmp_path / 'companion-alias'
        if alias_kind == 'symlink':
            destination.symlink_to(companion)
        else:
            destination.hardlink_to(companion)
    original = corpus_path.read_bytes()
    result = CliRunner().invoke(cli.corpus_export,
                               [str(corpus_path), 'paper:a', str(destination), '--force', '--json'])
    assert result.exit_code == 1
    assert result.stdout == ''
    assert 'must not be the source corpus' in result.stderr
    assert companion.read_bytes() == b'preserve database companion'
    assert corpus_path.read_bytes() == original


def test_export_reports_unwritable_path_without_modifying_corpus(
    corpus_path: Path, tmp_path: Path,
) -> None:
    """An unavailable output directory produces a CLI error without a partial export."""
    output = tmp_path / 'missing' / 'asset.txt'
    result = CliRunner().invoke(cli.corpus_export,
                               [str(corpus_path), 'paper:a', str(output), '--json'])
    assert result.exit_code == 1
    assert result.stdout == ''
    assert 'Error:' in result.stderr
    assert not output.exists()


def test_inspection_and_export_preserve_database_bytes(corpus_path: Path, tmp_path: Path) -> None:
    """All successful commands leave the corpus contents and modification time intact."""
    before = corpus_path.read_bytes()
    modified_at = corpus_path.stat().st_mtime_ns
    for command, arguments in [
        (cli.corpus_list, [str(corpus_path), '--json']),
        (cli.corpus_show, [str(corpus_path), 'paper:a', '--json']),
        (cli.corpus_export, [str(corpus_path), 'paper:a', str(tmp_path / 'export.txt'), '--json']),
    ]:
        result = CliRunner().invoke(command, arguments)
        assert result.exit_code == 0, result.output
        assert corpus_path.read_bytes() == before
        assert corpus_path.stat().st_mtime_ns == modified_at


def test_inspection_connection_enforces_read_only_and_closes(corpus_path: Path) -> None:
    """The inspection connection rejects SQL writes and closes after use."""
    with cli._open_corpus(str(corpus_path)) as conn:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            conn.execute('DELETE FROM papers')
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        conn.execute('SELECT 1')


def test_inspection_connection_holds_a_consistent_read_snapshot(corpus_path: Path) -> None:
    """Concurrent WAL updates do not change the records midway through inspection."""
    with closing(sqlite3.connect(corpus_path)) as setup:
        setup.execute('PRAGMA journal_mode = WAL')
    with cli._open_corpus(str(corpus_path)) as reader:
        original = reader.execute('SELECT title FROM papers WHERE paper_id = ?',
                                  ('paper:a',)).fetchone()[0]
        with closing(sqlite3.connect(corpus_path)) as writer, writer:
            writer.execute('UPDATE papers SET title = ? WHERE paper_id = ?',
                           ('Concurrent change', 'paper:a'))
        assert reader.execute('SELECT title FROM papers WHERE paper_id = ?',
                              ('paper:a',)).fetchone()[0] == original
    with cli._open_corpus(str(corpus_path)) as reader:
        assert reader.execute('SELECT title FROM papers WHERE paper_id = ?',
                              ('paper:a',)).fetchone()[0] == 'Concurrent change'


@pytest.mark.parametrize(('command', 'suffix'), [
    (cli.corpus_list, []),
    (cli.corpus_show, ['paper:a']),
    (cli.corpus_export, ['paper:a', 'export.txt']),
])
def test_missing_database_does_not_create_a_file(
    tmp_path: Path, command: click.Command, suffix: list[str],
) -> None:
    """Missing database arguments fail during parsing, before SQLite can create files."""
    missing = tmp_path / 'missing.db'
    result = CliRunner().invoke(command, [str(missing), *suffix, '--json'])
    assert result.exit_code == 2
    assert result.stdout == ''
    assert not missing.exists()


@pytest.mark.parametrize(('kind', 'expected_error'), [
    ('corrupt', 'not a database'),
    ('unrelated', 'Not a PaperMinerToolkit corpus'),
    ('older', 'Migrate it with a corpus-writing workflow'),
    ('newer', 'newer than supported'),
    ('incomplete', 'Incomplete PaperMinerToolkit corpus schema'),
])
def test_invalid_database_fails_without_migration_or_writes(
    tmp_path: Path, kind: str, expected_error: str,
) -> None:
    """Corrupt, unrelated, old, new, and incomplete schemas remain unchanged."""
    path = tmp_path / f'{kind}.db'
    if kind == 'corrupt':
        path.write_bytes(b'This is not a SQLite database.')
    elif kind == 'unrelated':
        with sqlite3.connect(path) as conn:
            conn.execute('CREATE TABLE unrelated (id INTEGER)')
        conn.close()
    else:
        with open_corpus(path) as conn:
            if kind == 'incomplete':
                conn.execute('DROP TABLE paper_authors')
            else:
                version = corpus.SCHEMA_VERSION + (1 if kind == 'newer' else -1)
                conn.execute(f'PRAGMA user_version = {version}')
    before = path.read_bytes()
    result = CliRunner().invoke(cli.corpus_list, [str(path), '--json'])
    assert result.exit_code == 1
    assert result.stdout == ''
    assert expected_error in result.stderr
    assert path.read_bytes() == before
