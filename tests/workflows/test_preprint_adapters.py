"""Contract checks for shared download and enrichment preprint operations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from paperminertoolkit.providers import registry
from paperminertoolkit.providers.base import FullTextDocument
from paperminertoolkit.workflows import download, enrichment


PREPRINTS = [
    ('medrxiv', '10.1101/2024.03.01.24303596', 'v2'),
    ('biorxiv', '10.1101/2023.12.01.569634', 'v2'),
    ('chemrxiv', '10.26434/chemrxiv.15007737', '/v2'),
]


@pytest.mark.parametrize(('source', 'stem', 'suffix'), PREPRINTS)
def test_preprint_fields_preserve_native_and_published_doi_rules(
    source: str, stem: str, suffix: str,
) -> None:
    """Keep archive URL versions distinct from chemRxiv's registered version."""
    fields = getattr(enrichment, f'_{source}_fields')
    column = f'{source}_doi'
    entry = {column: f'https://doi.org/{stem}{suffix}', 'title': '  A title  '}
    before = deepcopy(entry)
    result = fields(entry)
    assert result['doi'] == stem + suffix
    assert result[column] == (stem + suffix if source == 'chemrxiv' else stem)
    assert result['title'] == 'A title'
    assert result['work_type'] == 'preprint'
    assert (result['is_oa'], result['oa_status']) == (1, 'green')
    assert entry == before
    entry['published_doi'] = 'https://doi.org/10.1234/PUBLISHED'
    assert fields(entry)['doi'] == '10.1234/published'
    assert fields(entry)['work_type'] == ''


@pytest.mark.parametrize(('source', 'stem', 'suffix'), PREPRINTS)
def test_preprint_candidate_index_retains_last_row_and_requires_paper_id(
    source: str, stem: str, suffix: str,
) -> None:
    """Skip absent IDs and retain the last candidate for one native DOI."""
    column = f'{source}_doi'
    candidates = [
        {'paper_id': '', column: stem + suffix},
        {'paper_id': 'first', column: stem + suffix},
        {'paper_id': 'missing'},
        {'paper_id': 'last', column: stem + suffix},
    ]
    expected = stem + suffix if source == 'chemrxiv' else stem
    assert getattr(enrichment, f'_{source}_candidates')(candidates) == {expected: 'last'}


@pytest.mark.parametrize('source', ['medrxiv', 'biorxiv', 'chemrxiv'])
def test_preprint_subject_rows_preserve_duplicates_rank_gaps_and_keywords(source: str) -> None:
    """Retain category ordering and chemRxiv keywords without arXiv deduplication."""
    entry = {'categories': [
        {'id': ''},
        {'id': ' A ', 'name': '', 'is_primary': True},
        {'id': 'A', 'name': 'Alias', 'is_primary': False},
    ], 'keywords': ['', ' Catalysis ', 'CATALYSIS']}
    before = deepcopy(entry)
    rows = getattr(enrichment, f'_{source}_subject_rows')('paper', entry)
    assert [(row['subject_id'], row['subject_rank'], row['is_primary'])
            for row in rows[:2]] == [('A', 1, 1), ('A', 2, 0)]
    assert [row['display_name'] for row in rows[:2]] == ['A', 'Alias']
    assert all(row['scheme'] == f'{source}_category' for row in rows[:2])
    if source == 'chemrxiv':
        assert [(row['subject_id'], row['subject_rank']) for row in rows[2:]] == [
            ('catalysis', 1), ('catalysis', 2)]
    else:
        assert len(rows) == 2
    assert entry == before


@pytest.mark.parametrize('source', ['medrxiv', 'biorxiv', 'chemrxiv'])
def test_preprint_empty_provenance_retains_provider_specific_schema(source: str) -> None:
    """An empty fetched record differs from an absent record in provenance."""
    provenance = enrichment._provenance(None, None, **{f'{source}_entry': {}})
    fields = provenance[source]
    assert fields[f'{source}_doi'] == ''
    assert fields['published_doi'] == ''
    if source == 'chemrxiv':
        assert fields['keywords'] == []
        assert fields['asset_url'] == ''
        assert 'jatsxml' not in fields
    else:
        assert fields['jatsxml'] == ''
        assert 'keywords' not in fields


@pytest.mark.parametrize(('source', 'stem', 'suffix'), PREPRINTS)
def test_preprint_record_distinguishes_missing_and_empty_mapping(
    monkeypatch: pytest.MonkeyPatch, source: str, stem: str, suffix: str,
) -> None:
    """Retain an empty successful mapping while reporting an absent response."""
    client = registry.resolve(source)
    record = getattr(download, f'_{source}_record')
    paper = {f'{source}_doi': stem + suffix}
    monkeypatch.setattr(client, 'fetch_doi', lambda _: {})
    assert record(paper) == ({}, '')
    monkeypatch.setattr(client, 'fetch_doi', lambda _: None)
    identifier = stem + suffix if source == 'chemrxiv' else stem
    assert record(paper) == (
        None, f'no {registry.SOURCES[source].label} record found for {identifier}')
    assert record({}) == (None, f'missing {registry.SOURCES[source].label} DOI')


@pytest.mark.parametrize('source', ['medrxiv', 'biorxiv'])
def test_rxiv_text_preserves_original_document_and_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str,
) -> None:
    """Retain the structured source and propagate filesystem failures."""
    entry: dict[str, Any] = {f'{source}_doi': '10.1101/example'}
    document = FullTextDocument(text='Text μm²', content='<article/>', document_format='jats')
    monkeypatch.setattr(download, f'_{source}_record', lambda _: (entry, ''))
    monkeypatch.setattr(registry.resolve(source), 'full_text_document', lambda _: document)
    download_text = getattr(download, f'_download_{source}_text')
    destination = tmp_path / 'article.txt'
    success, reason, original = download_text({}, destination)
    assert (success, reason) == (True, '')
    assert original is document
    assert destination.read_text(encoding='utf-8') == document.text
    with pytest.raises(OSError):
        download_text({}, tmp_path)
