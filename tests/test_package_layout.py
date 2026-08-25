"""Tests for the domain-oriented Python package layout."""

from __future__ import annotations

import importlib

import pytest


PUBLIC_MODULES = (
    'paperminertoolkit.corpus.database',
    'paperminertoolkit.corpus.documents',
    'paperminertoolkit.corpus.filtering',
    'paperminertoolkit.corpus.metadata',
    'paperminertoolkit.extraction.compression',
    'paperminertoolkit.extraction.extract',
    'paperminertoolkit.extraction.models',
    'paperminertoolkit.extraction.recipes',
    'paperminertoolkit.extraction.scrape',
    'paperminertoolkit.extraction.store',
    'paperminertoolkit.extraction.tokenizer',
    'paperminertoolkit.providers.arxiv',
    'paperminertoolkit.providers.base',
    'paperminertoolkit.providers.biorxiv',
    'paperminertoolkit.providers.chemrxiv',
    'paperminertoolkit.providers.core',
    'paperminertoolkit.providers.crossref',
    'paperminertoolkit.providers.elsevier',
    'paperminertoolkit.providers.medrxiv',
    'paperminertoolkit.providers.openalex',
    'paperminertoolkit.providers.pubmed',
    'paperminertoolkit.providers.registry',
    'paperminertoolkit.providers.rxiv',
    'paperminertoolkit.providers.unpaywall',
    'paperminertoolkit.workflows.download',
    'paperminertoolkit.workflows.enrichment',
    'paperminertoolkit.workflows.imports',
    'paperminertoolkit.workflows.search',
    'paperminertoolkit.workflows.topics',
    'paperminertoolkit.workflows.utilities',
)


@pytest.mark.parametrize('module_name', PUBLIC_MODULES)
def test_public_domain_modules_are_importable(module_name: str) -> None:
    """Keep each documented module reachable at its new public path."""
    assert importlib.import_module(module_name).__name__ == module_name
