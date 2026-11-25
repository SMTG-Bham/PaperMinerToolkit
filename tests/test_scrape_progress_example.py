"""Check that the Gaudi example's progress report names its batch job."""

from __future__ import annotations

from examples.sol_gaudi import scrape_progress as example
from paperminertoolkit.extraction import scrape_progress as progress


def test_example_report_resubmits_the_batch_job() -> None:
    """Point at scrape_gaudi.sbatch instead of a bare pmt scrape rerun."""
    result = {
        'scope': 2, 'out_of_scope': 0, 'filtered': False,
        'stages': {'text': {'succeeded': 1, 'no_asset': 0, 'failed': 1, 'not_attempted': 0}},
        'supplements': {'text': {}}, 'remaining': {'papers': 1, 'supplement_files': 0},
    }
    text = example.report(result, '/data/papers.db')
    assert 'sbatch --export=ALL,PMT_DB=/data/papers.db scrape_gaudi.sbatch' in text
    assert example.scrape_progress is progress.scrape_progress
