"""Session-wide fixtures for the PaperMinerToolkit test suite."""

from __future__ import annotations

import pytest

from paperminertoolkit.providers import base as provider, chemrxiv, elsevier


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
        The limiters, the chemRxiv category cache, and the remembered Elsevier
        quota are reset for the test.
    """
    provider.reset_limiters()
    chemrxiv.reset_categories_cache()
    elsevier.reset_quota()
    monkeypatch.setattr(provider.time, 'sleep', lambda _: None)
