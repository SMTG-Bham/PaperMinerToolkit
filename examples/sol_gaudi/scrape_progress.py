"""Report how far a corpus scrape has got and what resubmitting the job would still do.

``scrape_gaudi.sbatch`` runs this after every scrape::

    python scrape_progress.py <papers.db> <mode> [--json]

The counting lives in :mod:`paperminertoolkit.extraction.scrape_progress`,
which ``pmt scrape-progress`` also uses; this wrapper only adds the
resubmission command for the Gaudi batch job.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from paperminertoolkit.extraction import scrape_progress as _progress

scrape_progress = _progress.scrape_progress


def report(progress: dict[str, Any], db_path: str) -> str:
    """Describe scrape progress, naming the batch job that continues it.

    Parameters
    ----------
    progress : dict[str, Any]
        Output of :func:`scrape_progress`.
    db_path : str
        Corpus database path, quoted in the resubmission command.

    Returns
    -------
    str
        Multi-line report.
    """
    return _progress.report(progress, rerun=f'sbatch --export=ALL,PMT_DB={db_path} scrape_gaudi.sbatch')


def main() -> None:
    """Print the progress report for the database and mode on the command line."""
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != '--json'):
        sys.exit('usage: python scrape_progress.py <papers.db> <mode> [--json]')
    progress = scrape_progress(sys.argv[1], sys.argv[2])
    print(json.dumps(progress, indent=2) if len(sys.argv) == 4 else report(progress, sys.argv[1]))


if __name__ == '__main__':
    main()
