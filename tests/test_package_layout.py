"""Tests for the domain-oriented Python package layout."""

from __future__ import annotations

import importlib

import pytest


PUBLIC_MODULES = (
    'paperminer.corpus.database',
    'paperminer.corpus.documents',
    'paperminer.corpus.filtering',
    'paperminer.corpus.metadata',
    'paperminer.extraction.compression',
    'paperminer.extraction.extract',
    'paperminer.extraction.models',
    'paperminer.extraction.recipes',
    'paperminer.extraction.scrape',
    'paperminer.extraction.store',
    'paperminer.extraction.tokenizer',
    'paperminer.providers.arxiv',
    'paperminer.providers.base',
    'paperminer.providers.biorxiv',
    'paperminer.providers.chemrxiv',
    'paperminer.providers.core',
    'paperminer.providers.crossref',
    'paperminer.providers.elsevier',
    'paperminer.providers.medrxiv',
    'paperminer.providers.openalex',
    'paperminer.providers.pubmed',
    'paperminer.providers.registry',
    'paperminer.providers.rxiv',
    'paperminer.providers.unpaywall',
    'paperminer.workflows.download',
    'paperminer.workflows.enrichment',
    'paperminer.workflows.imports',
    'paperminer.workflows.search',
    'paperminer.workflows.topics',
    'paperminer.workflows.utilities',
)


@pytest.mark.parametrize('module_name', PUBLIC_MODULES)
def test_public_domain_modules_are_importable(module_name: str) -> None:
    """Keep each documented module reachable at its new public path."""
    assert importlib.import_module(module_name).__name__ == module_name
