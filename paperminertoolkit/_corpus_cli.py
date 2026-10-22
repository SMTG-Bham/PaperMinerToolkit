"""Read-only paper inspection and explicit asset export for the CLI."""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

import click

from paperminertoolkit.corpus.database import (
    ENRICHMENT_COLUMNS,
    PAPER_FIELDS,
    SCHEMA_VERSION,
    get_asset,
    get_asset_metadata,
    paper_authors,
    paper_references,
    paper_search_history,
    paper_subjects,
)


_LIST_COLUMNS = [
    *PAPER_FIELDS,
    *(column for column in ENRICHMENT_COLUMNS if column != 'enrichment_json'),
    'created_at',
    'updated_at',
]
_REQUIRED_TABLES = {
    'papers', 'blobs', 'paper_assets', 'search_runs', 'paper_search_results',
    'paper_authors', 'paper_subjects', 'paper_references',
}


@contextmanager
def _open_corpus(db_path: str) -> Iterator[sqlite3.Connection]:
    """Open a current corpus without creating, migrating, or writing its files.

    Parameters
    ----------
    db_path : str
        Existing corpus database path.

    Yields
    ------
    sqlite3.Connection
        Read-only connection with dictionary-like rows.

    Raises
    ------
    click.ClickException
        If the file cannot be read or has an incompatible corpus schema.
    """
    try:
        uri = Path(db_path).resolve().as_uri() + '?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )}
            if not {'papers', 'blobs', 'paper_assets'} <= tables:
                raise click.ClickException(f'Not a PaperMinerToolkit corpus: {db_path}')
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if version > SCHEMA_VERSION:
                raise click.ClickException(
                    f'Corpus schema version {version} is newer than supported version '
                    f'{SCHEMA_VERSION}. Upgrade PaperMinerToolkit to inspect this corpus.'
                )
            if version < SCHEMA_VERSION:
                raise click.ClickException(
                    f'Corpus schema version {version} is older than required version '
                    f'{SCHEMA_VERSION}. Migrate it with a corpus-writing workflow '
                    'before using read-only inspection.'
                )
            if not _REQUIRED_TABLES <= tables:
                raise click.ClickException(f'Incomplete PaperMinerToolkit corpus schema: {db_path}')
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(papers)')}
            if not {*_LIST_COLUMNS, 'metadata_json', 'enrichment_json'} <= columns:
                raise click.ClickException(f'Incomplete PaperMinerToolkit corpus schema: {db_path}')
            conn.execute('BEGIN')
            yield conn
    except (EOFError, OSError, RuntimeError, ValueError, sqlite3.Error, zlib.error) as error:
        raise click.ClickException(str(error)) from error


def _paper(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any]:
    """Read one exact canonical identifier or report a missing paper."""
    row = conn.execute('SELECT * FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()
    if row is None:
        raise click.ClickException(f'Paper not found: {paper_id}')
    paper = dict(row)
    for column, key in [('metadata_json', 'metadata'), ('enrichment_json', 'enrichment')]:
        paper[key] = json.loads(paper.pop(column) or '{}')
    return paper


@click.command('list')
@click.argument('db_path', default='papers.db', type=click.Path(exists=True, dir_okay=False))
@click.option('--limit', default=20, type=click.IntRange(1, 1000), show_default=True,
              help='Maximum paper rows to return, up to 1000.')
@click.option('--offset', default=0, type=click.IntRange(min=0), show_default=True,
              help='Number of paper rows to skip in paper-ID order.')
@click.option('--search-id', default=None, type=click.IntRange(min=1),
              help='Only papers matched by this recorded search.')
@click.option('--json', 'json_output', is_flag=True,
              help='Write paper rows and pagination metadata as JSON.')
def corpus_list(db_path: str, limit: int, offset: int,
                search_id: int | None, json_output: bool) -> None:
    """List paper metadata and pipeline state in canonical paper-ID order.

    The corpus is opened read-only. Large raw metadata payloads and binary
    assets are omitted; use show for full metadata and export for content.
    """
    with _open_corpus(db_path) as conn:
        where = ''
        parameters: tuple[int, ...] = ()
        if search_id is not None:
            if conn.execute('SELECT 1 FROM search_runs WHERE search_id = ?',
                            (search_id,)).fetchone() is None:
                raise click.ClickException(f'Search not found: {search_id}')
            where = (' WHERE EXISTS (SELECT 1 FROM paper_search_results AS results '
                     'WHERE results.paper_id = papers.paper_id AND results.search_id = ?)')
            parameters = (search_id,)
        total = conn.execute(f'SELECT COUNT(*) FROM papers{where}', parameters).fetchone()[0]
        columns = ', '.join(f'papers.{column}' for column in _LIST_COLUMNS)
        rows = conn.execute(
            f'SELECT {columns} FROM papers{where} ORDER BY papers.paper_id LIMIT ? OFFSET ?',
            (*parameters, limit, offset),
        ).fetchall()
        papers = [dict(row) for row in rows]
    if json_output:
        click.echo(json.dumps({'papers': papers, 'total': total,
                               'limit': limit, 'offset': offset}, indent=2))
        return
    click.echo(f'Corpus: {db_path} | {total} papers | offset={offset} limit={limit}')
    for paper in papers:
        title = ' '.join(str(paper['title'] or '').split())
        click.echo(f'{paper["paper_id"]}\t{paper["publication_date"] or ""}\t{title}')
    if not papers:
        click.echo('No papers in this page.')


@click.command('show')
@click.argument('db_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('paper_id')
@click.option('--json', 'json_output', is_flag=True,
              help='Write complete paper metadata, asset metadata, and provenance as JSON.')
def corpus_show(db_path: str, paper_id: str, json_output: bool) -> None:
    """Inspect one canonical paper ID, its latest assets, and provenance.

    Asset metadata describes the newest stored asset for each role, matching
    export. Content bytes are excluded. No provider requests or writes occur.
    """
    with _open_corpus(db_path) as conn:
        paper = _paper(conn, paper_id)
        roles = conn.execute(
            'SELECT DISTINCT role FROM paper_assets WHERE paper_id = ? ORDER BY role',
            (paper_id,),
        ).fetchall()
        result = {
            'paper': paper,
            'assets': [get_asset_metadata(conn, paper_id, row['role']) for row in roles],
            'searches': paper_search_history(conn, paper_id),
            'authors': paper_authors(conn, paper_id),
            'subjects': paper_subjects(conn, paper_id),
            'references': paper_references(conn, paper_id),
        }
    if json_output:
        click.echo(json.dumps(result, indent=2))
        return
    click.echo(f'Paper: {paper_id}')
    click.echo(f'Title: {paper["title"] or ""}')
    click.echo(f'DOI: {paper["doi"] or ""}')
    click.echo(f'Publication date: {paper["publication_date"] or ""}')
    click.echo(f'Authors: {paper["authors"] or ""}')
    for field, value in paper.items():
        if field.endswith('_status') or field == 'last_error':
            click.echo(f'{field}: {value or ""}')
    click.echo('Assets:')
    for asset in result['assets']:
        click.echo(f'  {asset["role"]}: {asset["original_size"]} bytes '
                   f'({asset["mime_type"]}; source={asset["source"] or "unknown"})')
    click.echo(f'Contributing searches: {len(result["searches"])}')


@click.command('export')
@click.argument('db_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('paper_id')
@click.argument('outfile', type=click.Path(dir_okay=False))
@click.option('--role', default='text', type=click.Choice(['abstract', 'text', 'pdf']),
              show_default=True, help='Stored asset to export, using its newest version.')
@click.option('--force', is_flag=True, help='Overwrite an existing output file.')
@click.option('--json', 'json_output', is_flag=True,
              help='Write the exported path, paper ID, role, and byte count as JSON.')
def corpus_export(db_path: str, paper_id: str, outfile: str, role: str,
                  force: bool, json_output: bool) -> None:
    """Export one stored asset to OUTFILE without changing the corpus.

    Decompressed bytes are written exactly as stored. A missing paper or asset
    fails before writing the output. Existing files require --force; the source
    corpus and its SQLite journal files can never be overwritten, including
    through symlinks or hard links.
    """
    try:
        destination = Path(outfile).absolute()
        source = Path(db_path).resolve()
        protected = [source, *(Path(f'{source}{suffix}')
                               for suffix in ('-wal', '-shm', '-journal'))]
        if any(destination.resolve() == path.resolve() or (
            destination.exists() and path.exists() and destination.samefile(path)
        ) for path in protected):
            raise click.ClickException(
                'The export destination must not be the source corpus or its SQLite journal files.'
            )
        if destination.exists() and not force:
            raise click.ClickException(f'Output file already exists: {outfile}. Use --force to overwrite.')
        with _open_corpus(db_path) as conn:
            _paper(conn, paper_id)
            asset = get_asset(conn, paper_id, role)
            if asset is None:
                raise click.ClickException(f'No stored {role} asset for paper: {paper_id}')
            content = asset['content']
        with destination.open('wb' if force else 'xb') as output:
            output.write(content)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        click.echo(json.dumps({'paper_id': paper_id, 'role': role,
                               'path': str(destination), 'bytes': len(content)}, indent=2))
    else:
        click.echo(f'Exported {role} for {paper_id} to {destination} ({len(content)} bytes).')
