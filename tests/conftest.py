"""Session-wide fixtures for the PaperMinerToolkit test suite."""

from __future__ import annotations

import pytest

from paperminertoolkit._agent import VALIDATE_OUTPUT_ENV
from paperminertoolkit.providers import base as provider, chemrxiv, elsevier, openalex


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
        quota and OpenAlex budget are reset for the test.
    """
    provider.reset_limiters()
    chemrxiv.reset_categories_cache()
    elsevier.reset_quota()
    openalex.reset_budget()
    monkeypatch.setattr(provider.time, 'sleep', lambda _: None)


@pytest.fixture(autouse=True)
def validate_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check every JSON result a command prints against its declared output schema.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to set the validation switch for the test.

    Returns
    -------
    None
        ``emit_json`` raises when a result does not match the schema, which
        fails the command and the test that ran it.
    """
    monkeypatch.setenv(VALIDATE_OUTPUT_ENV, '1')
