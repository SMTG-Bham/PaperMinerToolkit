"""Characterize shared metadata primitives and deliberately distinct adapters."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from paperminertoolkit.corpus import metadata
from paperminertoolkit.providers import crossref, openalex, pubmed
from paperminertoolkit.workflows import download, enrichment, search


@pytest.mark.parametrize(('value', 'searched', 'downloaded'), [
    (None, '', ''),
    (float('nan'), 'nan', 'nan'),
    ([0, False, None, ''], '', '0 False'),
    (['<b>α</b>', '&amp;', 'μm²'], 'α & μm²', 'α & μm²'),
    ('&lt;i&gt;H₂O&lt;/i&gt;\n β', 'H₂O β', 'H₂O β'),
])
def test_abstract_cleaners_share_markup_but_keep_list_coercion(
    value: object, searched: str, downloaded: str,
) -> None:
    """Preserve Unicode and the two workflow entry points' falsey-value rules."""
    assert search._clean_search_abstract(value) == searched
    assert download._clean_abstract(value) == downloaded


def test_openalex_adapters_share_authors_but_retain_field_whitespace() -> None:
    """Use one author rendering policy without broadening field normalization."""
    work = {
        'title': '  A title  ', 'publication_date': ' 2025-01-01 ',
        'primary_location': {'source': {'display_name': ' Journal '}},
        'authorships': [None, {}, {'author': None}, {'author': {'display_name': ''}},
                       {'author': {'display_name': ' Alice '}},
                       {'author': {'display_name': 'Βeta'}}],
    }
    paper = openalex.work_to_paper(work)
    fields = enrichment._openalex_fields(work)
    assert paper['authors'] == fields['authors'] == ' Alice ; Βeta'
    assert paper['title'] == '  A title  '
    assert fields['title'] == 'A title'
    assert paper['journal'] == ' Journal '
    assert fields['journal'] == 'Journal'
    assert paper['publication_date'] == ' 2025-01-01 '
    assert fields['publication_date'] == '2025-01-01'


def test_crossref_date_adapters_keep_existing_coercion_and_container_rules() -> None:
    """Different year padding, numeric coercion, and mapping support remain explicit."""
    assert crossref._publication_date({'issued': {'date-parts': [[7, 1]]}}) == '7-01'
    assert metadata._date_from_parts([[7, 1]]) == '0007-01'
    assert enrichment._date_parts({'date-parts': [['7', '1']]}) == '0007-01'
    assert crossref._publication_date({'issued': {'date-parts': [['7', '1']]}}) == '7-01'
    with pytest.raises(ValueError):
        metadata._date_from_parts([['7', '1']])
    container = MappingProxyType({'date-parts': [[2025, 1]]})
    assert crossref._publication_date({'issued': container}) == ''
    assert metadata._published_date({'issued': container}) == '2025-01'
    assert enrichment._date_parts(container) == '2025-01'


def test_crossref_text_adapters_keep_selection_and_scientific_normalization() -> None:
    """Author import selects nonempty titles; metadata takes and normalizes the first."""
    work: dict[str, Any] = {
        'DOI': 'https://doi.org/10.1234/ABC',
        'title': ['', '<i>H₂O</i>'],
        'container-title': ['Journal'],
    }
    imported = crossref.crossref_work_to_paper(work)
    fields = metadata.crossref_fields(work)
    assert imported['title'] == '<i>H₂O</i>'
    assert fields['title'] == ''
    assert imported['doi'] == 'https://doi.org/10.1234/abc'
    assert fields['doi'] == 'https://doi.org/10.1234/ABC'
    assert metadata.clean_doi(work['DOI']) == '10.1234/abc'
    work['title'] = ['<i>H₂O</i>']
    assert metadata.crossref_fields(work)['title'] == 'H2O'
    work['author'] = [{'given': 0, 'family': False}, {}]
    assert crossref.crossref_work_to_paper(work)['authors'] == '; '
    assert enrichment._crossref_fields(work)['authors'] == '0 False; '


def test_contact_resolution_preserves_provider_and_workflow_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NCBI and enrichment intentionally prioritize different stored addresses."""
    settings = {'ncbi_email': 'ncbi@example.test', 'crossref_email': 'crossref@example.test'}
    monkeypatch.setattr(pubmed, 'load_settings', lambda: settings)
    monkeypatch.setattr(crossref, 'load_settings', lambda: settings)
    monkeypatch.setenv('NCBI_EMAIL', 'environment@example.test')
    assert pubmed.configured_email() == 'ncbi@example.test'
    assert pubmed.configured_email({'crossref_email': 'other@example.test'}) == (
        'environment@example.test')
    assert enrichment._contact_address(['crossref', 'pubmed'], None) == 'crossref@example.test'
    assert enrichment._contact_address(['pubmed'], None) == 'ncbi@example.test'
    assert enrichment._contact_address(['crossref', 'pubmed'], ' explicit@example.test ') == (
        'explicit@example.test')
    assert pubmed.configured_email({}) == 'ncbi@example.test'
    assert crossref.configured_email({}) == ''
