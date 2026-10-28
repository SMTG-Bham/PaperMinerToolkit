"""Test common search-row assembly through each provider adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from paperminertoolkit.workflows import search


@pytest.mark.parametrize('assemble', [
    search._pubmed_rows, search._arxiv_rows, search._rxiv_rows,
    search._medrxiv_rows, search._biorxiv_rows, search._chemrxiv_rows,
])
def test_mapped_search_rows_keep_order_columns_and_clean_abstracts(
    assemble: Callable[[Iterable[Mapping[str, Any]]], pd.DataFrame],
) -> None:
    """Normalize a one-shot input without changing records or empty schemas."""
    entries = [
        {'paper_id': 'pmid:2', 'title': 'Second', 'abstract': ['<b>Some</b>', '&amp; text']},
        {'paper_id': 'pmid:1', 'title': 'First', 'abstract': None},
    ]
    original = deepcopy(entries)

    result = assemble(iter(entries))

    assert result.columns.tolist() == search.SEARCH_FIELDS
    assert result['paper_id'].tolist() == ['pmid:2', 'pmid:1']
    assert result['abstract'].tolist() == ['Some & text', '']
    assert entries == original
    empty = assemble(iter(()))
    assert empty.empty
    assert empty.columns.tolist() == search.SEARCH_FIELDS


def test_core_and_openalex_search_rows_map_native_abstracts() -> None:
    """Apply each provider's native mapping before common abstract cleaning."""
    core_rows = search._core_rows(iter([
        {'id': '1', 'abstract': ['<b>A</b>  deposit.'], 'title': ['Repository paper']},
    ]))
    openalex_rows = search._openalex_rows(iter([
        {'id': 'https://openalex.org/W2', 'abstract_inverted_index': {
            '<b>First</b>': [0], '&amp;': [1], 'last.': [2],
        }},
    ]))

    assert core_rows['paper_id'].tolist() == ['core:1']
    assert core_rows['abstract'].tolist() == ['A deposit.']
    assert openalex_rows['paper_id'].tolist() == ['openalex:W2']
    assert openalex_rows['abstract'].tolist() == ['First & last.']
    assert search._core_rows([]).columns.tolist() == search.SEARCH_FIELDS
    assert search._openalex_rows([]).columns.tolist() == search.SEARCH_FIELDS
