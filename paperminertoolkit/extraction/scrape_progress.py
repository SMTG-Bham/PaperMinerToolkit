"""Report how far a corpus scrape has got and what rerunning it would still do.

A paper counts only if ``pmt scrape`` would select it: with active corpus
filters, only the papers they include. Within that scope a paper has
succeeded, has no downloaded text or PDF (a rerun fails it again at once, so
it needs ``pmt download`` instead), has failed for another reason (a rerun
retries it), or has not been tried yet. Supporting-information files are
counted the same way, since a rerun retries every failed or partial file even
when its paper already succeeded.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from os import PathLike
from typing import Any

from paperminertoolkit.corpus.database import connect, paper_rows
from paperminertoolkit.corpus.filtering import active_filter_stack, current_filter_statuses
from paperminertoolkit.corpus.supplements import get_supplements

#: Corpus status columns each scrape mode writes.
STAGES = {
    'abstract': ['abstract'],
    'text': ['text'],
    'images': ['image'],
    'text-images': ['text', 'image'],
}
# pmt scrape records these when a paper has nothing to read; retrying cannot help.
NO_ASSET_ERRORS = ('No downloaded text or PDF asset', 'No downloaded abstract asset')
RETRIED_SUPPLEMENT_STATUSES = {'failed', 'partial', 'pending', 'None'}


def scrape_progress(db_path: str | PathLike[str], mode: str) -> dict[str, Any]:
    """Count scrape outcomes for the papers ``pmt scrape`` would select.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Corpus database path.
    mode : str
        Scrape mode: ``abstract``, ``text``, ``images`` or ``text-images``.

    Returns
    -------
    dict[str, Any]
        ``mode``, ``scope`` (papers selected), ``out_of_scope`` (filtered
        out), ``filtered`` (whether filters are active), ``stages`` (per stage:
        ``succeeded``, ``no_asset``, ``failed`` and ``not_attempted``),
        ``supplements`` (per stage, file counts by status), ``remaining``
        (work a rerun would still attempt), and per stage the
        ``failed_papers`` a rerun retries and the ``no_asset_papers`` that need
        a download first, as paper IDs in corpus order.

    Raises
    ------
    ValueError
        If `mode` is not a scrape mode.
    RuntimeError
        If the corpus schema is newer than this package supports.
    """
    if mode not in STAGES:
        raise ValueError(f'mode must be one of: {", ".join(STAGES)}')
    with closing(connect(db_path)) as conn:
        papers = paper_rows(conn)
        filtered = bool(active_filter_stack(conn))
        statuses = current_filter_statuses(conn) if filtered else {}
        selected = [paper for paper in papers
                    if not filtered or statuses.get(paper['paper_id']) == 'included']
        stages = {}
        failed_papers: dict[str, list[str]] = {}
        no_asset_papers: dict[str, list[str]] = {}
        for stage in STAGES[mode]:
            counts: Counter[str] = Counter()
            failed_papers[stage], no_asset_papers[stage] = [], []
            for paper in selected:
                status = paper.get(f'{stage}_scrape_status')
                if status == 'succeeded':
                    counts['succeeded'] += 1
                elif status == 'failed' and str(paper.get('last_error') or '').startswith(NO_ASSET_ERRORS):
                    counts['no_asset'] += 1
                    no_asset_papers[stage].append(paper['paper_id'])
                elif status == 'failed':
                    counts['failed'] += 1
                    failed_papers[stage].append(paper['paper_id'])
                else:
                    counts['not_attempted'] += 1
            stages[stage] = {key: counts[key]
                             for key in ('succeeded', 'no_asset', 'failed', 'not_attempted')}
        supplements: dict[str, Counter[str]] = {
            stage: Counter() for stage in STAGES[mode] if stage != 'abstract'}
        for paper in selected if supplements else []:
            for asset in get_supplements(conn, paper['paper_id'], include_content=False):
                for stage, file_counts in supplements.items():
                    file_counts[str(asset['metadata'].get(f'{stage}_extraction_status'))] += 1
    remaining = max((counts['failed'] + counts['not_attempted'] for counts in stages.values()),
                    default=0)
    remaining_supplements = max(
        (sum(n for status, n in file_counts.items() if status in RETRIED_SUPPLEMENT_STATUSES)
         for file_counts in supplements.values()), default=0)
    return {
        'mode': mode,
        'scope': len(selected),
        'out_of_scope': len(papers) - len(selected),
        'filtered': filtered,
        'stages': stages,
        'supplements': {stage: dict(file_counts) for stage, file_counts in supplements.items()},
        'remaining': {'papers': remaining, 'supplement_files': remaining_supplements},
        'failed_papers': failed_papers,
        'no_asset_papers': no_asset_papers,
    }


def _count(n: int, noun: str) -> str:
    """Return a count with its noun, plural unless the count is one."""
    return f'{n} {noun}' if n == 1 else f'{n} {noun}s'


def report(progress: dict[str, Any], rerun: str | None = None) -> str:
    """Describe scrape progress and the next step in plain text.

    Parameters
    ----------
    progress : dict[str, Any]
        Output of :func:`scrape_progress`.
    rerun : str or None, optional
        Command that continues the scrape, shown when work remains. Without
        it the report says to run the same ``pmt scrape`` command again.

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
        retry = (f'A rerun would retry {_count(papers, "paper")} and '
                 f'{_count(files, "supporting-information file")}. '
                 'The scrape skips what already succeeded, so continue it by ')
        if rerun:
            lines.append(retry + 'submitting the same job again:')
            lines.append(f'  {rerun}')
        else:
            lines.append(retry + 'running the same pmt scrape command again. To retry only the '
                         'failed papers, pass their IDs (failed_papers in --json) with --paper-id.')
    else:
        lines.append('Corpus fully scraped: every selected paper with a document has been scraped.')
    if no_asset:
        lines.append(f'{_count(no_asset, "selected paper")} without downloaded text or PDF; scraping '
                     'cannot reach them until pmt download finds one.')
    return '\n'.join(lines)
