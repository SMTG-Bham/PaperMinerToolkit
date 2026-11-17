"""Machine-readable progress events for long-running workflows.

Workflows call :func:`emit` at milestones such as one paper finishing. It does
nothing unless a command has turned events on with :func:`jsonl_progress`,
which writes each event as one JSON line to stderr and hides tqdm progress
bars, so the event stream is not interleaved with bar redraws.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_SINK: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    'paperminertoolkit_progress_sink', default=None)
_BARS_DISABLED: ContextVar[bool] = ContextVar('paperminertoolkit_bars_disabled', default=False)

#: ``--progress`` choices: ``auto`` shows tqdm bars, ``none`` hides them, and
#: ``jsonl`` hides them and writes JSON events to stderr.
PROGRESS_MODES = ('auto', 'none', 'jsonl')


def emit(event: str, **fields: Any) -> None:
    """Report one progress event, if a command asked for events.

    Parameters
    ----------
    event : str
        Event name, such as ``run_start``, ``paper_done``, or ``run_end``.
    **fields : Any
        JSON-compatible event fields.
    """
    sink = _SINK.get()
    if sink is not None:
        sink({'event': event, **fields})


def bars_disabled() -> bool:
    """Return whether tqdm progress bars should be hidden.

    Returns
    -------
    bool
        True under ``--progress none`` and ``--progress jsonl``.
    """
    return _BARS_DISABLED.get()


@contextmanager
def progress_mode(mode: str, command: str) -> Iterator[None]:
    """Apply a ``--progress`` mode for the duration of a command.

    Under ``jsonl`` every event is one line on stderr: a JSON object with
    ``event``, ``command``, and ``elapsed_s`` (seconds since the command
    started) plus the event's own fields. An exception that ends the command
    is reported as an ``error`` event before it propagates.

    Parameters
    ----------
    mode : str
        One of :data:`PROGRESS_MODES`.
    command : str
        Command name recorded on each event.

    Yields
    ------
    None
        Control returns to the command with the mode applied.

    Raises
    ------
    ValueError
        If ``mode`` is unknown.
    """
    if mode not in PROGRESS_MODES:
        raise ValueError(f'progress mode must be one of: {", ".join(PROGRESS_MODES)}')
    started = time.monotonic()

    def write(record: dict[str, Any]) -> None:
        """Write one event as a JSON line to the current stderr."""
        line = {'event': record.pop('event'), 'command': command,
                'elapsed_s': round(time.monotonic() - started, 3), **record}
        sys.stderr.write(json.dumps(line, default=str) + '\n')
        sys.stderr.flush()

    sink_token = _SINK.set(write if mode == 'jsonl' else None)
    bars_token = _BARS_DISABLED.set(mode != 'auto')
    try:
        yield
    except Exception as error:
        emit('error', message=str(error), error_class=type(error).__name__)
        raise
    finally:
        _BARS_DISABLED.reset(bars_token)
        _SINK.reset(sink_token)
