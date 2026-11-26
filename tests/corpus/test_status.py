"""Verify shared in-memory corpus pipeline status updates."""

from __future__ import annotations

import pytest

from paperminertoolkit.corpus.database import set_pipeline_status


@pytest.mark.parametrize(
    ('status', 'error', 'expected_error'),
    [
        ('failed', 'new failure', 'new failure'),
        ('failed', None, 'previous failure'),
        ('pending', '', 'previous failure'),
        ('succeeded', None, ''),
        ('stored', '', ''),
        ('succeeded', 'new failure', 'new failure'),
    ],
)
def test_pipeline_status_preserves_error_precedence(
    status: str,
    error: str | None,
    expected_error: str,
) -> None:
    """Keep download and scrape status/error mutations identical."""
    paper = {'text_scrape_status': 'pending', 'last_error': 'previous failure'}

    assert set_pipeline_status(paper, 'text_scrape_status', status, error) is None

    assert paper == {'text_scrape_status': status, 'last_error': expected_error}


def test_pipeline_status_rejects_unknown_columns_without_mutation() -> None:
    """Reject an unknown column before changing either status or error."""
    paper = {'last_error': 'previous failure'}

    with pytest.raises(KeyError, match='Unknown pipeline status column: unknown'):
        set_pipeline_status(paper, 'unknown', 'failed', 'new failure')

    assert paper == {'last_error': 'previous failure'}
