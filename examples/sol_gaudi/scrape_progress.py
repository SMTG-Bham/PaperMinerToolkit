"""Report how far a corpus scrape has got and what resubmitting the job would still do.

``scrape_gaudi.sbatch`` runs this after every scrape::

    python scrape_progress.py <papers.db> <mode>

A paper counts only if ``pmt scrape`` would select it: with active corpus
filters, only the papers they include. Within that scope a paper has
succeeded, has no downloaded text or PDF (a resubmission fails it again at
once, so it needs ``pmt download`` instead), has failed for another reason (a
resubmission retries it), or has not been tried yet. Supporting-information
files are counted the same way, since a resubmission retries every failed or
partial file even when its paper already succeeded.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

from paperminertoolkit.corpus.database import connect, paper_rows
from paperminertoolkit.corpus.filtering import active_filter_stack, current_filter_statuses
from paperminertoolkit.corpus.supplements import get_supplements

STAGES = {
    'abstract': ['abstract'],
    'text': ['text'],
    'images': ['image'],
    'text-images': ['text', 'image'],
}
# pmt scrape records these when a paper has nothing to read; retrying cannot help.
NO_ASSET_ERRORS = ('No downloaded text or PDF asset', 'No downloaded abstract asset')
RETRIED_SUPPLEMENT_STATUSES = {'failed', 'partial', 'pending', None}


def scrape_progress(db_path: str, mode: str) -> dict[str, Any]:
    """Count scrape outcomes for the papers ``pmt scrape`` would select.

    Parameters
    ----------
    db_path : str
        Corpus database path.
    mode : str
        Scrape mode: ``abstract``, ``text``, ``images`` or ``text-images``.

    Returns
    -------
    dict[str, Any]
        ``scope`` (papers selected), ``out_of_scope`` (filtered out),
        ``filtered`` (whether filters are active), ``stages`` (per stage:
        ``succeeded``, ``no_asset``, ``failed`` and ``not_attempted``),
        ``supplements`` (per stage, file counts by status) and ``remaining``
        (work a resubmission would still attempt).

    Raises
    ------
    ValueError
        If `mode` is not a scrape mode.
    """
    if mode not in STAGES:
        raise ValueError(f'mode must be one of: {", ".join(STAGES)}')
    conn = connect(db_path)
    try:
        papers = paper_rows(conn)
        filtered = bool(active_filter_stack(conn))
        statuses = current_filter_statuses(conn) if filtered else {}
        selected = [paper for paper in papers if not filtered or statuses.get(paper['paper_id']) == 'included']
        stages = {}
        for stage in STAGES[mode]:
            counts = Counter()
            for paper in selected:
                status = paper.get(f'{stage}_scrape_status')
                if status == 'succeeded':
                    counts['succeeded'] += 1
                elif status == 'failed' and str(paper.get('last_error') or '').startswith(NO_ASSET_ERRORS):
                    counts['no_asset'] += 1
                elif status == 'failed':
                    counts['failed'] += 1
                else:
                    counts['not_attempted'] += 1
            stages[stage] = {key: counts[key] for key in ('succeeded', 'no_asset', 'failed', 'not_attempted')}
        supplements = {stage: Counter() for stage in STAGES[mode] if stage != 'abstract'}
        for paper in selected if supplements else []:
            for asset in get_supplements(conn, paper['paper_id'], include_content=False):
                for stage, counts in supplements.items():
                    counts[str(asset['metadata'].get(f'{stage}_extraction_status'))] += 1
    finally:
        conn.close()
    remaining = max((counts['failed'] + counts['not_attempted'] for counts in stages.values()), default=0)
    remaining_supplements = max(
        (sum(n for status, n in counts.items() if status in {str(s) for s in RETRIED_SUPPLEMENT_STATUSES})
         for counts in supplements.values()), default=0)
    return {
        'scope': len(selected),
        'out_of_scope': len(papers) - len(selected),
        'filtered': filtered,
        'stages': stages,
        'supplements': {stage: dict(counts) for stage, counts in supplements.items()},
        'remaining': {'papers': remaining, 'supplement_files': remaining_supplements},
    }


def _count(n: int, noun: str) -> str:
    """Return a count with its noun, plural unless the count is one."""
    return f'{n} {noun}' if n == 1 else f'{n} {noun}s'


def report(progress: dict[str, Any], db_path: str) -> str:
    """Describe scrape progress and the next step in plain text.

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
    lines = []
    if progress['filtered']:
        lines.append(f'{progress["scope"]} papers are included by the active corpus filters; '
                     f'{progress["out_of_scope"]} excluded or unavailable papers are never scraped.')
    else:
        lines.append(f'{progress["scope"]} papers; no corpus filters are active.')
    no_asset = 0
    for stage, counts in progress['stages'].items():
        lines.append(f'{stage}_scrape_status: {counts["succeeded"]} succeeded, '
                     f'{counts["no_asset"]} without a downloaded document, {counts["failed"]} failed, '
                     f'{counts["not_attempted"]} not yet attempted.')
        no_asset = max(no_asset, counts['no_asset'])
    for stage, counts in progress['supplements'].items():
        if counts:
            summary = ', '.join(f'{n} {status}' for status, n in sorted(counts.items()))
            lines.append(f'Supporting-information files ({stage}): {summary}.')
    papers, files = progress['remaining']['papers'], progress['remaining']['supplement_files']
    lines.append('')
    if papers or files:
        lines.append(f'A resubmission would retry {_count(papers, "paper")} and '
                     f'{_count(files, "supporting-information file")}. '
                     'The scrape skips what already succeeded, so continue it by submitting the same job again:')
        lines.append(f'  sbatch --export=ALL,PMT_DB={db_path} scrape_gaudi.sbatch')
    else:
        lines.append('Corpus fully scraped: every selected paper with a document has been scraped.')
    if no_asset:
        lines.append(f'{_count(no_asset, "selected paper")} without downloaded text or PDF; scraping cannot reach them '
                     'until pmt download finds one.')
    return '\n'.join(lines)


def main() -> None:
    """Print the progress report for the database and mode on the command line."""
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != '--json'):
        sys.exit('usage: python scrape_progress.py <papers.db> <mode> [--json]')
    progress = scrape_progress(sys.argv[1], sys.argv[2])
    print(json.dumps(progress, indent=2) if len(sys.argv) == 4 else report(progress, sys.argv[1]))


if __name__ == '__main__':
    main()
