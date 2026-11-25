"""Options and helpers shared by the ``pmt`` command modules.

They live apart from :mod:`paperminertoolkit.cli` so that command groups
defined in their own modules can use them without a circular import.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Callable, Mapping
from contextlib import closing, nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import click

from paperminertoolkit._progress import PROGRESS_MODES
from paperminertoolkit.corpus.database import corpus_copy


_json_option = click.option(
    '--json', 'json_output', is_flag=True,
    help='Write JSON to stdout and progress or errors to stderr.',
)


def _quiet(json_output: bool) -> Any:
    """Route workflow chatter to stderr while a command produces JSON."""
    return redirect_stdout(sys.stderr) if json_output else nullcontext()


_progress_option = click.option(
    '--progress', 'progress_display', type=click.Choice(PROGRESS_MODES), default='auto',
    show_default=True,
    help='auto shows progress bars; none hides them; jsonl hides them and writes one JSON '
         'event per line to stderr.',
)

_dry_run_option = click.option(
    '--dry-run', is_flag=True, default=False,
    help='Report what the command would change, without changing the corpus or any file.',
)


def _on_copy(db_path: str, operation: Callable[[str], Mapping[str, Any]]) -> dict[str, Any]:
    """Run a corpus-changing operation on a temporary copy and report its result."""
    with corpus_copy(db_path) as copy:
        return {**operation(str(copy)), 'dry_run': True}


def _paper_id_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Add the shared ``--paper-id`` and ``--paper-ids-file`` selection options."""
    command = click.option(
        '--paper-ids-file', default=None, type=click.Path(exists=True, dir_okay=False),
        help='File of corpus paper IDs, one per line (blank lines and # comments ignored); '
             'combined with --paper-id.',
    )(command)
    return click.option(
        '--paper-id', 'paper_ids', multiple=True,
        help='Only process this corpus paper ID. Repeat for several. Active corpus filters '
             'do not apply to named papers.',
    )(command)


# SQLite limits the number of bound parameters per statement.
_ID_LOOKUP_BATCH = 500


def _selected_paper_ids(db_path: str, paper_ids: tuple[str, ...],
                        ids_file: str | None) -> list[str] | None:
    """Resolve ``--paper-id`` and ``--paper-ids-file`` against the corpus.

    Returns ``None`` when neither option was given. Unknown identifiers are a
    usage error that names them; the corpus is opened read-only.
    """
    if not paper_ids and ids_file is None:
        return None
    requested = [paper_id.strip() for paper_id in paper_ids if paper_id.strip()]
    if ids_file is not None:
        try:
            lines = Path(ids_file).read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise click.BadParameter(str(error), param_hint='--paper-ids-file') from error
        requested.extend(line.strip() for line in lines
                         if line.strip() and not line.lstrip().startswith('#'))
    requested = list(dict.fromkeys(requested))
    if not requested:
        raise click.BadParameter('No paper IDs were given.', param_hint='--paper-id')
    known: set[str] = set()
    try:
        uri = f'{Path(db_path).resolve().as_uri()}?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            for start in range(0, len(requested), _ID_LOOKUP_BATCH):
                batch = requested[start:start + _ID_LOOKUP_BATCH]
                rows = conn.execute(
                    f'SELECT paper_id FROM papers WHERE paper_id IN ({", ".join("?" for _ in batch)})',
                    batch,
                ).fetchall()
                known.update(row[0] for row in rows)
    except sqlite3.Error as error:
        raise click.ClickException(f'Cannot look up paper IDs in {db_path}: {error}') from error
    missing = [paper_id for paper_id in requested if paper_id not in known]
    if missing:
        shown = ', '.join(missing[:20])
        more = f' and {len(missing) - 20} more' if len(missing) > 20 else ''
        raise click.BadParameter(f'Unknown paper ID(s) in {db_path}: {shown}{more}',
                                 param_hint='--paper-id')
    return requested

# A download may fetch a PDF, full text, or an abstract, so its choices are
# the union of the three, kept in PDF order because that is the one a caller
# most often means.
