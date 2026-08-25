"""Assert every source maps a paper onto the same record shape.

Each source module has its own suite for its own fields. What is checked here
is the part that is supposed to be identical across all of them, because that
is what search, enrichment, and download read without knowing which source a
record came from.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any

import pytest

import paperminer.providers.arxiv as arxiv
import paperminer.providers.biorxiv as biorxiv
import paperminer.providers.chemrxiv as chemrxiv
import paperminer.providers.core as core
import paperminer.providers.crossref as crossref
import paperminer.providers.elsevier as elsevier
import paperminer.providers.medrxiv as medrxiv
import paperminer.providers.openalex as openalex
import paperminer.providers.pubmed as pubmed
from paperminer.providers import registry as sources
from paperminer.corpus.database import ENRICHMENT_COLUMNS, PAPER_FIELDS

# The keys every source must supply, whatever it is and whatever it holds.
CORE_KEYS = ('paper_id', 'doi', 'title', 'journal', 'publication_date',
             'authors', 'sources', 'metadata_status')
# The keys a preprint server must supply, so that a consumer can read a
# preprint record without asking which archive it came from.
PREPRINT_KEYS = ('abstract', 'categories', 'category', 'primary_category',
                 'published_doi', 'version', 'pdf_url')


def records() -> dict[str, dict[str, Any]]:
    """Map one record per source onto the paper schema.

    Returns
    -------
    dict[str, dict[str, Any]]
        Normalized record keyed by source name.
    """
    return {
        'arxiv': arxiv.entry_to_paper(ET.fromstring(
            '<entry xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:arxiv="http://arxiv.org/schemas/atom">'
            '<id>http://arxiv.org/abs/2301.01234v1</id><title>A preprint</title>'
            '<summary>An abstract.</summary><published>2023-01-05T00:00:00Z</published>'
            '<category term="cond-mat.mtrl-sci"/></entry>')),
        'medrxiv': medrxiv.record_to_paper({
            'doi': '10.1101/2024.03.01.24303596', 'title': 'A preprint',
            'date': '2024-03-04', 'version': '1', 'category': 'Health Policy'}),
        'biorxiv': biorxiv.record_to_paper({
            'doi': '10.1101/2023.12.01.569634', 'title': 'A preprint',
            'date': '2023-12-02', 'version': '1', 'category': 'Neuroscience'}),
        'chemrxiv': chemrxiv.record_to_paper({
            'doi': '10.26434/chemrxiv-2022-w08rh', 'title': 'A preprint',
            'publishedDate': '2022-05-01', 'version': '1'}),
        'pubmed': pubmed.article_to_paper(ET.fromstring(
            '<PubmedArticle><MedlineCitation><PMID>1</PMID><Article>'
            '<ArticleTitle>An article</ArticleTitle></Article>'
            '</MedlineCitation></PubmedArticle>')),
        'openalex': openalex.work_to_paper({'id': 'https://openalex.org/W1',
                                            'title': 'A work'}),
        'crossref': crossref.crossref_work_to_paper({'DOI': '10.1234/x',
                                                     'title': ['A work']}),
        'core': core.work_to_paper({'id': '1', 'title': 'A deposit'}),
        'elsevier': elsevier.record_to_paper({'dc:identifier': 'SCOPUS_ID:1',
                                              'dc:title': 'An article'}),
    }


@pytest.mark.parametrize('name', sorted(records()))
def test_every_source_supplies_the_core_record_keys(name: str) -> None:
    """Require the keys a consumer reads without knowing the source."""
    record = records()[name]
    missing = [key for key in CORE_KEYS if key not in record]
    assert not missing, f'{name} omits {missing}'
    assert record['sources'] == name
    assert record['metadata_status'] == 'retrieved'


@pytest.mark.parametrize('name', ['arxiv', 'medrxiv', 'biorxiv', 'chemrxiv'])
def test_every_preprint_source_supplies_the_preprint_keys(name: str) -> None:
    """Require the four preprint servers to agree on their extra keys.

    arXiv used to emit primary_category holding a category id while the other
    three emitted category holding a name, so the same concept reached
    enrichment under two names with two kinds of value.
    """
    record = records()[name]
    missing = [key for key in PREPRINT_KEYS if key not in record]
    assert not missing, f'{name} omits {missing}'
    assert isinstance(record['categories'], list)
    assert isinstance(record['category'], str)
    assert isinstance(record['primary_category'], str)


@pytest.mark.parametrize('name', ['arxiv', 'medrxiv', 'biorxiv', 'chemrxiv'])
def test_the_primary_category_matches_the_flagged_term(name: str) -> None:
    """Take both category keys from the same term, not from different ones."""
    record = records()[name]
    primary = next((term for term in record['categories'] if term['is_primary']), None)
    if primary is None:
        assert record['category'] == ''
        assert record['primary_category'] == ''
        return
    assert record['category'] == primary['name']
    assert record['primary_category'] == primary['id']


@pytest.mark.parametrize('name', sorted(records()))
def test_a_source_identifies_its_own_paper_id(name: str) -> None:
    """Give every record a usable identifier rather than an empty one."""
    assert records()[name]['paper_id']


def test_source_identifier_columns_are_corpus_columns() -> None:
    """Keep the registry's identifier columns to ones the corpus can store.

    Most sit on the papers table; OpenAlex's is written by enrichment, so both
    column sets count.
    """
    storable = set(PAPER_FIELDS) | set(ENRICHMENT_COLUMNS)
    for column in sources.identifier_columns():
        assert column in storable, f'{column} is not a corpus column'


@pytest.mark.parametrize('name', ['arxiv', 'medrxiv', 'biorxiv', 'chemrxiv',
                                  'pubmed', 'core', 'elsevier', 'openalex'])
def test_every_abstract_bearing_source_supplies_the_key(name: str) -> None:
    """Offer an abstract key from each source a download can ask for one from.

    OpenAlex stores its abstract as an inverted index and used to leave the
    reconstruction to whichever caller wanted one, so the key was absent here
    while every other source supplied it.
    """
    assert 'abstract' in records()[name]


def test_a_record_survives_being_written_to_the_corpus_schema() -> None:
    """Keep every mapper's output loadable, extras and all."""
    from paperminer.corpus.database import normalize_paper

    for name, record in records().items():
        normalized = normalize_paper(record)
        assert normalized['sources'] == name
        assert set(normalized) >= set(PAPER_FIELDS)
        # Round-trips as JSON, which is what enrichment stores it as.
        json.dumps(record, default=str)
