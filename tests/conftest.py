"""Session-wide fixtures for the PaperMiner test suite."""

from __future__ import annotations

import pytest

from paperminer.providers import base as provider, chemrxiv


@pytest.fixture(autouse=True)
def reset_limiters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset shared provider state and silence pacing and backoff sleeps.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the sleep the shared client would otherwise do.

    Returns
    -------
    None
        The limiters and the chemRxiv category cache are reset for the test.
    """
    provider.reset_limiters()
    chemrxiv.reset_categories_cache()
    monkeypatch.setattr(provider.time, 'sleep', lambda _: None)
