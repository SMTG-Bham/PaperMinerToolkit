"""Supplement corpus metadata from bibliographic and preprint providers.

Discovery and download populate only a handful of bibliographic fields. This
module fills in the rest: publisher, work type, volume/issue/pages, ISSNs and
language from Crossref; citation counts, open-access status, licence and
subject classification from OpenAlex; and structured authors, subjects and
reference lists in the corpus child tables.

Crossref is treated as authoritative for the metadata a publisher deposits
against the DOI, and OpenAlex for everything it derives that no publisher
deposits. Where the two disagree, both values are recorded in
``papers.enrichment_json`` so nothing is silently discarded.

arXiv ranks below those two, because its record describes a preprint rather
than the version of record. It contributes what no other provider holds: the
arXiv subject taxonomy, the author-deposited DOI that lets a preprint-only row
reach Crossref and OpenAlex on a later pass, and the fact that the paper is
freely readable. Unlike the other three it cannot be reached from a DOI at all,
because arXiv exposes no DOI search field, so it enriches only rows that
already carry an arXiv identifier.

medRxiv and bioRxiv rank last, for the same reason as arXiv and one more: a
preprint row that names a published version already holds that version's DOI,
so Crossref and OpenAlex describe the paper better than the preprint record
does. What only they hold is their own subject category, the preprint's licence,
and the link between the two DOIs. Like arXiv both are reachable only from an
identifier they issued themselves, which here is the preprint DOI rather than
the published one. They are separate sources rather than one because they are
separate archives: a DOI belongs to exactly one of them, and asking the wrong
one costs a request and returns nothing.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from os import PathLike
from typing import Any, TypeAlias

from tqdm import tqdm

from paperminer.providers import arxiv
from paperminer.providers import base as provider
from paperminer.providers import registry
from paperminer.providers import biorxiv
from paperminer.providers import chemrxiv
from paperminer.providers import crossref as crossref_client
from paperminer.providers import medrxiv
from paperminer.providers import openalex
from paperminer.providers import pubmed
from paperminer.corpus.database import (connect,
                                 enrichment_candidates,
                                 enrichment_update_fields,
                                 find_paper,
                                 paper_rows,
                                 set_enrichment_status,
                                 utc_now,
                                 write_enrichment)
from paperminer.corpus.metadata import clean_doi, crossref_fields as crossref_metadata_fields

ENRICHMENT_SOURCES = registry.names(registry.ENRICH)
MAX_BATCH_SIZE = 100
_Record: TypeAlias = dict[str, Any]
_Fields: TypeAlias = dict[str, Any]


def _configured_sources(sources: Sequence[str] | None) -> list[str]:
    """Resolve requested enrichment providers, expanding ``all``.

    Parameters
    ----------
    sources : Sequence[str] or None
        Requested provider names, or ``None`` for every provider.

    Returns
    -------
    list[str]
        Provider names in a stable order.

    Raises
    ------
    ValueError
        If a requested provider is not supported.
    """
    return registry.resolve_names(sources, registry.ENRICH, label='enrichment')


def _short_openalex_id(value: object) -> str:
    """Reduce an OpenAlex entity URL to its short identifier."""
    identifier = str(value or '').strip().rstrip('/')
    return identifier.rsplit('/', 1)[-1] if identifier else ''


def _text(value: object) -> str:
    """Normalize an optional provider value to stripped text."""
    return '' if value is None else str(value).strip()


def _partition_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Split candidates by the lookup key each one can be resolved with.

    Parameters
    ----------
    candidates : Sequence[Mapping[str, Any]]
        Candidate corpus rows carrying ``paper_id``, ``doi`` and ``openalex_id``.

    Returns
    -------
    by_doi : dict[str, str]
        Cleaned DOI to paper identifier.
    by_openalex : dict[str, str]
        Short OpenAlex identifier to paper identifier, for DOI-less rows.
    unresolved : list[str]
        Paper identifiers with no usable lookup key.
    """
    by_doi = {}
    by_openalex = {}
    unresolved = []
    for candidate in candidates:
        paper_id = str(candidate.get('paper_id') or '')
        if not paper_id:
            continue
        doi = clean_doi(candidate.get('doi'))
        identifier = _short_openalex_id(candidate.get('openalex_id'))
        if not identifier and paper_id.startswith('openalex:'):
            identifier = paper_id.split(':', 1)[1]
        if doi:
            by_doi[doi] = paper_id
        elif identifier:
            by_openalex[identifier] = paper_id
        else:
            unresolved.append(paper_id)
    return by_doi, by_openalex, unresolved


def _pubmed_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map each candidate's PubMed identifier to its paper identifier.

    This is a parallel accessor rather than a fourth
    :func:`_partition_candidates` bucket, because a PMID is read from the same
    candidate row that already supplies the DOI and OpenAlex keys.

    Parameters
    ----------
    candidates : Sequence[Mapping[str, Any]]
        Candidate corpus rows carrying ``paper_id`` and ``pmid``.

    Returns
    -------
    dict[str, str]
        Bare PMID to paper identifier, for rows that carry one.
    """
    by_pmid = {}
    for candidate in candidates:
        paper_id = str(candidate.get('paper_id') or '')
        if not paper_id:
            continue
        pmid = pubmed.normalize_pmid(candidate.get('pmid'))
        if not pmid and paper_id.startswith('pmid:'):
            pmid = pubmed.normalize_pmid(paper_id.split(':', 1)[1])
        if pmid:
            by_pmid[pmid] = paper_id
    return by_pmid


def _pubmed_fields(article: Mapping[str, Any]) -> _Fields:
    """Map one PubMed article onto the shared enrichment field set.

    Parameters
    ----------
    article : Mapping[str, Any]
        Article mapping produced by :func:`paperminer.providers.pubmed.article_to_paper`.

    Returns
    -------
    dict[str, Any]
        Enrichment fields contributed by PubMed.
    """
    types = article.get('publication_types') or []
    return {
        'doi': clean_doi(article.get('doi')) if article.get('doi') else '',
        'title': _text(article.get('title')),
        'journal': _text(article.get('journal')),
        'publication_date': _text(article.get('publication_date')),
        'authors': _text(article.get('authors')),
        'pmid': pubmed.normalize_pmid(article.get('pmid')),
        'pmcid': pubmed.normalize_pmcid(article.get('pmcid')),
        'work_type': _text((types[0] or {}).get('name')) if types else '',
    }


def _pubmed_subject_rows(paper_id: str, article: Mapping[str, Any] | None) -> list[_Record]:
    """Build ``paper_subjects`` rows from a PubMed article's controlled terms.

    MeSH descriptors, MeSH qualifiers, publication types and author keywords
    are kept in separate schemes. PubMed keywords use ``mesh_keyword`` rather
    than ``keyword`` so they stay distinguishable from the OpenAlex keywords
    that share the ``(paper_id, scheme, subject_id)`` primary key.

    Parameters
    ----------
    paper_id : str
        Paper the rows belong to.
    article : Mapping[str, Any] or None
        Article mapping produced by :func:`paperminer.providers.pubmed.article_to_paper`.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the ``paper_subjects`` table.
    """
    if not article:
        return []
    rows: list[_Record] = []
    for rank, term in enumerate(article.get('mesh') or []):
        rows.append({
            'paper_id': paper_id, 'scheme': term.get('scheme') or 'mesh',
            'subject_id': term.get('id'), 'display_name': _text(term.get('name')),
            'subject_rank': rank, 'is_primary': int(term.get('is_primary') == '1'),
            'source': 'pubmed',
        })
    for rank, entry in enumerate(article.get('publication_types') or []):
        rows.append({
            'paper_id': paper_id, 'scheme': 'publication_type',
            'subject_id': entry.get('id'), 'display_name': _text(entry.get('name')),
            'subject_rank': rank, 'source': 'pubmed',
        })
    for rank, keyword in enumerate(article.get('keywords') or []):
        rows.append({
            'paper_id': paper_id, 'scheme': 'mesh_keyword', 'subject_id': keyword,
            'display_name': keyword, 'subject_rank': rank, 'source': 'pubmed',
        })
    seen = set()
    unique = []
    for row in rows:
        key = (row['scheme'], row['subject_id'])
        if not row['subject_id'] or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _arxiv_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map each candidate's arXiv identifier to its paper identifier.

    Unlike :func:`_pubmed_candidates`, which can fall back to resolving a PMID
    from a DOI, this can only ever see rows that already carry an arXiv
    identifier: arXiv publishes no DOI search field, so a DOI-only row is
    unreachable no matter how many requests are spent.

    Parameters
    ----------
    candidates : Sequence[Mapping[str, Any]]
        Candidate corpus rows carrying ``paper_id`` and ``arxiv_id``.

    Returns
    -------
    dict[str, str]
        Bare arXiv identifier to paper identifier, for rows that carry one.
    """
    by_arxiv = {}
    for candidate in candidates:
        paper_id = str(candidate.get('paper_id') or '')
        if not paper_id:
            continue
        identifier = arxiv.normalize_arxiv_id(candidate.get('arxiv_id'))
        if not identifier and paper_id.startswith('arxiv:'):
            identifier = arxiv.normalize_arxiv_id(paper_id.split(':', 1)[1])
        if identifier:
            by_arxiv[identifier] = paper_id
    return by_arxiv


def _arxiv_fields(entry: Mapping[str, Any]) -> _Fields:
    """Map one arXiv entry onto the shared enrichment field set.

    Every arXiv paper is freely readable, so ``is_oa`` and ``oa_status`` are
    asserted here rather than left at the OpenAlex default, which would record
    a plainly false value for a row OpenAlex has never seen.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Entry mapping produced by :func:`paperminer.providers.arxiv.entry_to_paper`.

    Returns
    -------
    dict[str, Any]
        Enrichment fields contributed by arXiv.
    """
    published = clean_doi(entry.get('published_doi')) if entry.get('published_doi') else ''
    return {
        'doi': clean_doi(entry.get('doi')) if entry.get('doi') else '',
        'title': _text(entry.get('title')),
        'journal': _text(entry.get('journal')),
        'publication_date': _text(entry.get('publication_date')),
        'authors': _text(entry.get('authors')),
        'arxiv_id': arxiv.normalize_arxiv_id(entry.get('arxiv_id')),
        'work_type': '' if published else 'preprint',
        'is_oa': 1,
        'oa_status': 'green',
    }


def _arxiv_subject_rows(paper_id: str, entry: Mapping[str, Any] | None) -> list[_Record]:
    """Build ``paper_subjects`` rows from an arXiv entry's categories.

    Categories use the ``arxiv_category`` scheme, which is disjoint from every
    scheme OpenAlex and PubMed write, so the three providers cannot collide on
    the ``(paper_id, scheme, subject_id)`` primary key. The primary category is
    flagged rather than given a scheme of its own, matching how a major MeSH
    descriptor is marked.

    Parameters
    ----------
    paper_id : str
        Paper the rows belong to.
    entry : Mapping[str, Any] or None
        Entry mapping produced by :func:`paperminer.providers.arxiv.entry_to_paper`.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the ``paper_subjects`` table.
    """
    if not entry:
        return []
    rows: list[_Record] = []
    seen = set()
    for rank, category in enumerate(entry.get('categories') or []):
        term = _text(category.get('id'))
        if not term or term in seen:
            continue
        seen.add(term)
        rows.append({
            'paper_id': paper_id, 'scheme': 'arxiv_category', 'subject_id': term,
            'display_name': _text(category.get('name')) or term, 'subject_rank': rank,
            'is_primary': int(bool(category.get('is_primary'))), 'source': 'arxiv',
        })
    return rows


def _medrxiv_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map each candidate's medRxiv DOI to its paper identifier.

    Like :func:`_arxiv_candidates`, this can only see rows that already carry
    the identifier medRxiv issued. A row's ``doi`` is not a usable substitute
    once the preprint has been published, because it then names the journal
    version, which medRxiv does not index.

    Parameters
    ----------
    candidates : Sequence[Mapping[str, Any]]
        Candidate corpus rows carrying ``paper_id`` and ``medrxiv_doi``.

    Returns
    -------
    dict[str, str]
        Bare medRxiv DOI to paper identifier, for rows that carry one.
    """
    by_medrxiv = {}
    for candidate in candidates:
        paper_id = str(candidate.get('paper_id') or '')
        if not paper_id:
            continue
        identifier = medrxiv.resolve_medrxiv_doi(candidate)
        if identifier:
            by_medrxiv[identifier] = paper_id
    return by_medrxiv


def _medrxiv_fields(entry: Mapping[str, Any]) -> _Fields:
    """Map one medRxiv record onto the shared enrichment field set.

    Every medRxiv preprint is freely readable, so ``is_oa`` and ``oa_status``
    are asserted here rather than left at the OpenAlex default, which would
    record a plainly false value for a row OpenAlex has never seen.

    ``work_type`` is only ``preprint`` while the paper is unpublished. Once
    medRxiv names a published version, the corpus row describes that version,
    and calling it a preprint would contradict the DOI stored beside it.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Record mapping produced by :func:`paperminer.providers.medrxiv.record_to_paper`.

    Returns
    -------
    dict[str, Any]
        Enrichment fields contributed by medRxiv.
    """
    published = clean_doi(entry.get('published_doi')) if entry.get('published_doi') else ''
    return {
        'doi': published or clean_doi(entry.get('medrxiv_doi') or ''),
        'title': _text(entry.get('title')),
        'journal': _text(entry.get('journal')),
        'publication_date': _text(entry.get('publication_date')),
        'authors': _text(entry.get('authors')),
        'medrxiv_doi': medrxiv.normalize_medrxiv_doi(entry.get('medrxiv_doi')),
        'work_type': '' if published else 'preprint',
        'license': _text(entry.get('license')),
        'is_oa': 1,
        'oa_status': 'green',
    }


def _medrxiv_subject_rows(paper_id: str, entry: Mapping[str, Any] | None) -> list[_Record]:
    """Build ``paper_subjects`` rows from a medRxiv record's category.

    Categories use the ``medrxiv_category`` scheme, which is disjoint from
    every scheme the other providers write, so they cannot collide on the
    ``(paper_id, scheme, subject_id)`` primary key. medRxiv files each preprint
    under exactly one category, so the single row is flagged primary.

    Parameters
    ----------
    paper_id : str
        Paper the rows belong to.
    entry : Mapping[str, Any] or None
        Record mapping produced by
        :func:`paperminer.providers.medrxiv.record_to_paper`.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the ``paper_subjects`` table.
    """
    if not entry:
        return []
    rows: list[_Record] = []
    for rank, category in enumerate(entry.get('categories') or []):
        term = _text(category.get('id'))
        if not term:
            continue
        rows.append({
            'paper_id': paper_id, 'scheme': 'medrxiv_category', 'subject_id': term,
            'display_name': _text(category.get('name')) or term, 'subject_rank': rank,
            'is_primary': int(bool(category.get('is_primary'))), 'source': 'medrxiv',
        })
    return rows


def _biorxiv_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map each candidate's bioRxiv DOI to its paper identifier.

    Like :func:`_medrxiv_candidates`, this can only see rows that already carry
    the identifier bioRxiv issued. A row's ``doi`` is not a usable substitute
    once the preprint has been published, because it then names the journal
    version, which bioRxiv does not index.

    Parameters
    ----------
    candidates : Sequence[Mapping[str, Any]]
        Candidate corpus rows carrying ``paper_id`` and ``biorxiv_doi``.

    Returns
    -------
    dict[str, str]
        Bare bioRxiv DOI to paper identifier, for rows that carry one.
    """
    by_biorxiv = {}
    for candidate in candidates:
        paper_id = str(candidate.get('paper_id') or '')
        if not paper_id:
            continue
        identifier = biorxiv.resolve_biorxiv_doi(candidate)
        if identifier:
            by_biorxiv[identifier] = paper_id
    return by_biorxiv


def _biorxiv_fields(entry: Mapping[str, Any]) -> _Fields:
    """Map one bioRxiv record onto the shared enrichment field set.

    Every bioRxiv preprint is freely readable, so ``is_oa`` and ``oa_status``
    are asserted here rather than left at the OpenAlex default, which would
    record a plainly false value for a row OpenAlex has never seen.

    ``work_type`` is only ``preprint`` while the paper is unpublished. Once
    bioRxiv names a published version, the corpus row describes that version,
    and calling it a preprint would contradict the DOI stored beside it.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Record mapping produced by :func:`paperminer.providers.biorxiv.record_to_paper`.

    Returns
    -------
    dict[str, Any]
        Enrichment fields contributed by bioRxiv.
    """
    published = clean_doi(entry.get('published_doi')) if entry.get('published_doi') else ''
    return {
        'doi': published or clean_doi(entry.get('biorxiv_doi') or ''),
        'title': _text(entry.get('title')),
        'journal': _text(entry.get('journal')),
        'publication_date': _text(entry.get('publication_date')),
        'authors': _text(entry.get('authors')),
        'biorxiv_doi': biorxiv.normalize_biorxiv_doi(entry.get('biorxiv_doi')),
        'work_type': '' if published else 'preprint',
        'license': _text(entry.get('license')),
        'is_oa': 1,
        'oa_status': 'green',
    }


def _biorxiv_subject_rows(paper_id: str, entry: Mapping[str, Any] | None) -> list[_Record]:
    """Build ``paper_subjects`` rows from a bioRxiv record's category.

    Categories use the ``biorxiv_category`` scheme, which is disjoint from
    every scheme the other providers write -- medRxiv's included, because the
    two archives classify under different subject lists -- so they cannot
    collide on the ``(paper_id, scheme, subject_id)`` primary key. bioRxiv files
    each preprint under exactly one category, so the single row is flagged
    primary.

    Parameters
    ----------
    paper_id : str
        Paper the rows belong to.
    entry : Mapping[str, Any] or None
        Record mapping produced by
        :func:`paperminer.providers.biorxiv.record_to_paper`.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the ``paper_subjects`` table.
    """
    if not entry:
        return []
    rows: list[_Record] = []
    for rank, category in enumerate(entry.get('categories') or []):
        term = _text(category.get('id'))
        if not term:
            continue
        rows.append({
            'paper_id': paper_id, 'scheme': 'biorxiv_category', 'subject_id': term,
            'display_name': _text(category.get('name')) or term, 'subject_rank': rank,
            'is_primary': int(bool(category.get('is_primary'))), 'source': 'biorxiv',
        })
    return rows


def _openalex_fields(work: Mapping[str, Any]) -> _Fields:
    """Map one OpenAlex work onto the shared enrichment field set.

    Parameters
    ----------
    work : Mapping[str, Any]
        OpenAlex work record.

    Returns
    -------
    dict[str, Any]
        Enrichment fields contributed by OpenAlex.
    """
    primary = work.get('primary_location') or {}
    best = work.get('best_oa_location') or {}
    source = primary.get('source') or {}
    biblio = work.get('biblio') or {}
    access = work.get('open_access') or {}
    identifiers = work.get('ids') or {}
    pages = [_text(biblio.get('first_page')), _text(biblio.get('last_page'))]
    authors = '; '.join(
        name for name in (
            ((authorship or {}).get('author') or {}).get('display_name')
            for authorship in work.get('authorships') or []
        ) if name
    )
    return {
        'openalex_id': openalex.work_id(work),
        'doi': clean_doi(work.get('doi')),
        'title': _text(work.get('title') or work.get('display_name')),
        'journal': _text(source.get('display_name')),
        'publication_date': _text(work.get('publication_date') or work.get('publication_year')),
        'authors': authors,
        'publisher': _text(source.get('host_organization_name')),
        'work_type': _text(work.get('type')),
        'volume': _text(biblio.get('volume')),
        'issue': _text(biblio.get('issue')),
        'pages': '-'.join(part for part in pages if part),
        'issn': ';'.join(_text(value) for value in source.get('issn') or [] if _text(value)),
        'issn_l': _text(source.get('issn_l')),
        'language': _text(work.get('language')),
        'is_oa': int(bool(access.get('is_oa'))),
        'oa_status': _text(access.get('oa_status')),
        'license': _text(best.get('license') or primary.get('license')),
        'is_retracted': int(bool(work.get('is_retracted'))),
        'cited_by_count': work.get('cited_by_count'),
        'referenced_works_count': work.get('referenced_works_count'),
        'pmid': pubmed.normalize_pmid(identifiers.get('pmid')),
        'pmcid': pubmed.normalize_pmcid(identifiers.get('pmcid')),
    }


def _crossref_retraction(work: Mapping[str, Any]) -> tuple[bool, str]:
    """Detect a retraction notice on a Crossref work.

    ``updated-by`` also carries corrections, errata and expressions of concern,
    so the update type is checked rather than the presence of the array.

    Parameters
    ----------
    work : Mapping[str, Any]
        Crossref work message.

    Returns
    -------
    retracted : bool
        Whether a retraction notice references this work.
    notice_doi : str
        DOI of the retraction notice, or an empty string.
    """
    for update in work.get('updated-by') or []:
        if str((update or {}).get('type') or '').lower() == 'retraction':
            return True, clean_doi(update.get('DOI'))
    return False, ''


def _crossref_fields(work: Mapping[str, Any]) -> _Fields:
    """Map one Crossref work onto the shared enrichment field set.

    Parameters
    ----------
    work : Mapping[str, Any]
        Crossref work message.

    Returns
    -------
    dict[str, Any]
        Enrichment fields contributed by Crossref.
    """
    mapped = crossref_metadata_fields(work)
    mapped.pop('crossref_message', None)
    retracted, _ = _crossref_retraction(work)
    mapped.update({
        'doi': clean_doi(mapped.get('doi')),
        'authors': '; '.join(
            ' '.join(part for part in [_text(author.get('given')), _text(author.get('family'))] if part)
            for author in work.get('author') or []
        ),
        'is_retracted': int(retracted),
        'referenced_works_count': work.get('references-count'),
        'license': _crossref_license(work),
    })
    return mapped


def _crossref_license(work: Mapping[str, Any]) -> str:
    """Read a Crossref licence URL, ignoring text-mining-only grants."""
    for entry in work.get('license') or []:
        if str((entry or {}).get('content-version') or '').lower() in {'vor', 'am'}:
            return _text(entry.get('URL'))
    return ''


def _chemrxiv_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map each candidate's chemRxiv DOI to its paper identifier.

    Like the other preprint servers, this can only see rows that already carry
    the identifier chemRxiv issued. A row's ``doi`` is not a usable substitute
    once the preprint has been published, because it then names the journal
    version, which chemRxiv does not index.

    Parameters
    ----------
    candidates : Sequence[Mapping[str, Any]]
        Candidate corpus rows carrying ``paper_id`` and ``chemrxiv_doi``.

    Returns
    -------
    dict[str, str]
        chemRxiv DOI to paper identifier, for rows that carry one.
    """
    by_chemrxiv = {}
    for candidate in candidates:
        paper_id = str(candidate.get('paper_id') or '')
        if not paper_id:
            continue
        identifier = chemrxiv.resolve_chemrxiv_doi(candidate)
        if identifier:
            by_chemrxiv[identifier] = paper_id
    return by_chemrxiv


def _chemrxiv_fields(entry: Mapping[str, Any]) -> _Fields:
    """Map one chemRxiv record onto the shared enrichment field set.

    Every chemRxiv preprint is freely readable, so ``is_oa`` and ``oa_status``
    are asserted here rather than left at the OpenAlex default, which would
    record a plainly false value for a row OpenAlex has never seen.

    ``work_type`` is only ``preprint`` while the paper is unpublished. Once
    chemRxiv names a published version, the corpus row describes that version,
    and calling it a preprint would contradict the DOI stored beside it.

    The chemRxiv DOI keeps the version suffix it was issued with, which is why
    it is normalized by :func:`paperminer.providers.chemrxiv.normalize_chemrxiv_doi`
    rather than by :func:`paperminer.corpus.metadata.clean_doi` alone.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Record mapping produced by
        :func:`paperminer.providers.chemrxiv.record_to_paper`.

    Returns
    -------
    dict[str, Any]
        Enrichment fields contributed by chemRxiv.
    """
    published = clean_doi(entry.get('published_doi')) if entry.get('published_doi') else ''
    identifier = chemrxiv.normalize_chemrxiv_doi(entry.get('chemrxiv_doi'))
    return {
        'doi': published or identifier,
        'title': _text(entry.get('title')),
        'journal': _text(entry.get('journal')),
        'publication_date': _text(entry.get('publication_date')),
        'authors': _text(entry.get('authors')),
        'chemrxiv_doi': identifier,
        'work_type': '' if published else 'preprint',
        'license': _text(entry.get('license')),
        'is_oa': 1,
        'oa_status': 'green',
    }


def _chemrxiv_subject_rows(paper_id: str, entry: Mapping[str, Any] | None) -> list[_Record]:
    """Build ``paper_subjects`` rows from a chemRxiv record's terms.

    Two schemes are written. ``chemrxiv_category`` carries the subject
    categories chemRxiv files the preprint under, of which there may be several
    where medRxiv and bioRxiv allow exactly one, so only the first is flagged
    primary. ``chemrxiv_keyword`` carries the author-supplied keywords, which
    the other preprint servers do not publish. Both are disjoint from every
    scheme the other providers write, so they cannot collide on the
    ``(paper_id, scheme, subject_id)`` primary key.

    Parameters
    ----------
    paper_id : str
        Paper the rows belong to.
    entry : Mapping[str, Any] or None
        Record mapping produced by
        :func:`paperminer.providers.chemrxiv.record_to_paper`.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the ``paper_subjects`` table.
    """
    if not entry:
        return []
    rows: list[_Record] = []
    for rank, category in enumerate(entry.get('categories') or []):
        term = _text(category.get('id'))
        if not term:
            continue
        rows.append({
            'paper_id': paper_id, 'scheme': 'chemrxiv_category', 'subject_id': term,
            'display_name': _text(category.get('name')) or term, 'subject_rank': rank,
            'is_primary': int(bool(category.get('is_primary'))), 'source': 'chemrxiv',
        })
    for rank, keyword in enumerate(entry.get('keywords') or []):
        term = _text(keyword)
        if not term:
            continue
        rows.append({
            'paper_id': paper_id, 'scheme': 'chemrxiv_keyword', 'subject_id': term.lower(),
            'display_name': term, 'subject_rank': rank,
            'is_primary': 0, 'source': 'chemrxiv',
        })
    return rows


def _provenance(crossref: Mapping[str, Any] | None,
                openalex_work: Mapping[str, Any] | None,
                pubmed_article: Mapping[str, Any] | None = None,
                arxiv_entry: Mapping[str, Any] | None = None,
                medrxiv_entry: Mapping[str, Any] | None = None,
                biorxiv_entry: Mapping[str, Any] | None = None,
                chemrxiv_entry: Mapping[str, Any] | None = None) -> _Record:
    """Build the trimmed provenance document stored in ``enrichment_json``."""
    provenance: _Record = {'fetched_at': utc_now()}
    if crossref is not None:
        retracted, notice = _crossref_retraction(crossref)
        provenance['crossref'] = {
            'publisher': _text(crossref.get('publisher')),
            'is_referenced_by_count': crossref.get('is-referenced-by-count'),
            'references_count': crossref.get('references-count'),
            'issn_type': crossref.get('issn-type') or [],
            'license': crossref.get('license') or [],
            'published_print': _date_parts(crossref.get('published-print')),
            'published_online': _date_parts(crossref.get('published-online')),
            'is_retracted': retracted,
            'retraction_doi': notice,
        }
    if openalex_work is not None:
        primary = openalex_work.get('primary_location') or {}
        source = primary.get('source') or {}
        provenance['openalex'] = {
            'id': openalex.work_id(openalex_work),
            'ids': openalex_work.get('ids') or {},
            'publisher': _text(source.get('host_organization_name')),
            'publication_date': _text(openalex_work.get('publication_date')),
            'cited_by_count': openalex_work.get('cited_by_count'),
            'is_authors_truncated': bool(openalex_work.get('is_authors_truncated')),
            'primary_location_license': _text(primary.get('license')),
            'best_oa_location_license': _text((openalex_work.get('best_oa_location') or {}).get('license')),
        }
    if pubmed_article is not None:
        provenance['pubmed'] = {
            'pmid': pubmed.normalize_pmid(pubmed_article.get('pmid')),
            'pmcid': pubmed.normalize_pmcid(pubmed_article.get('pmcid')),
            'article_type': _text(pubmed_article.get('article_type')),
            'publication_types': [_text(entry.get('name'))
                                  for entry in pubmed_article.get('publication_types') or []],
            'mesh_count': len(pubmed_article.get('mesh') or []),
        }
    if arxiv_entry is not None:
        provenance['arxiv'] = {
            'arxiv_id': arxiv.normalize_arxiv_id(arxiv_entry.get('arxiv_id')),
            'version': _text(arxiv_entry.get('version')),
            'primary_category': _text(arxiv_entry.get('primary_category')),
            'categories': [_text(term.get('id'))
                           for term in arxiv_entry.get('categories') or []],
            'journal_ref': _text(arxiv_entry.get('journal_ref')),
            'comment': _text(arxiv_entry.get('comment')),
        }
    if medrxiv_entry is not None:
        provenance['medrxiv'] = {
            'medrxiv_doi': medrxiv.normalize_medrxiv_doi(medrxiv_entry.get('medrxiv_doi')),
            'version': _text(medrxiv_entry.get('version')),
            'category': _text(medrxiv_entry.get('category')),
            'license': _text(medrxiv_entry.get('license')),
            'published_doi': _text(medrxiv_entry.get('published_doi')),
            'jatsxml': _text(medrxiv_entry.get('jatsxml')),
        }
    if biorxiv_entry is not None:
        provenance['biorxiv'] = {
            'biorxiv_doi': biorxiv.normalize_biorxiv_doi(biorxiv_entry.get('biorxiv_doi')),
            'version': _text(biorxiv_entry.get('version')),
            'category': _text(biorxiv_entry.get('category')),
            'license': _text(biorxiv_entry.get('license')),
            'published_doi': _text(biorxiv_entry.get('published_doi')),
            'jatsxml': _text(biorxiv_entry.get('jatsxml')),
        }
    if chemrxiv_entry is not None:
        # No jatsxml: chemRxiv serves PDFs and abstracts but no full text.
        provenance['chemrxiv'] = {
            'chemrxiv_doi': chemrxiv.normalize_chemrxiv_doi(chemrxiv_entry.get('chemrxiv_doi')),
            'version': _text(chemrxiv_entry.get('version')),
            'category': _text(chemrxiv_entry.get('category')),
            'keywords': list(chemrxiv_entry.get('keywords') or []),
            'license': _text(chemrxiv_entry.get('license')),
            'published_doi': _text(chemrxiv_entry.get('published_doi')),
            'asset_url': _text(chemrxiv_entry.get('asset_url')),
        }
    return provenance


def _date_parts(value: object) -> str:
    """Format a Crossref ``date-parts`` container as an ISO-like date."""
    parts = (value or {}).get('date-parts') if isinstance(value, Mapping) else None
    if not parts or not parts[0]:
        return ''
    date = [int(part) for part in parts[0][:3]]
    return '-'.join(f'{part:02d}' if index else f'{part:04d}' for index, part in enumerate(date))


CROSSREF_PREFERRED = ('publisher', 'work_type', 'volume', 'issue', 'pages', 'issn', 'language')
OPENALEX_ONLY = ('openalex_id', 'issn_l', 'is_oa', 'oa_status', 'license', 'cited_by_count')
FILL_COLUMNS = ('doi', 'title', 'journal', 'publication_date', 'authors', 'pmid', 'pmcid',
                'arxiv_id', 'medrxiv_doi', 'biorxiv_doi', 'chemrxiv_doi')
PRESERVED_ON_PROVIDER_ERROR = (CROSSREF_PREFERRED + OPENALEX_ONLY
                               + ('referenced_works_count', 'is_retracted'))


def _merge_fields(paper_id: str,
                 crossref: Mapping[str, Any] | None,
                 openalex_work: Mapping[str, Any] | None,
                 requested: Sequence[str],
                 pubmed_article: Mapping[str, Any] | None = None,
                 arxiv_entry: Mapping[str, Any] | None = None,
                 medrxiv_entry: Mapping[str, Any] | None = None,
                 biorxiv_entry: Mapping[str, Any] | None = None,
                 chemrxiv_entry: Mapping[str, Any] | None = None,
                 provider_errors: Mapping[str, str] | None = None) -> _Record:
    """Apply the provider precedence rules and build one paper's update.

    Parameters
    ----------
    paper_id : str
        Paper the update applies to.
    crossref : Mapping[str, Any] or None
        Crossref work message, or ``None`` when Crossref had no record.
    openalex_work : Mapping[str, Any] or None
        OpenAlex work record, or ``None`` when OpenAlex had no record.
    requested : Sequence[str]
        Providers that were queried for this paper.
    pubmed_article : Mapping[str, Any] or None, optional
        PubMed article mapping, or ``None`` when PubMed had no record.
    arxiv_entry : Mapping[str, Any] or None, optional
        arXiv entry mapping, or ``None`` when arXiv had no record.
    medrxiv_entry : Mapping[str, Any] or None, optional
        medRxiv record mapping, or ``None`` when medRxiv had no record.
    biorxiv_entry : Mapping[str, Any] or None, optional
        bioRxiv record mapping, or ``None`` when bioRxiv had no record.
    chemrxiv_entry : Mapping[str, Any] or None, optional
        chemRxiv record mapping, or ``None`` when chemRxiv had no record.
    provider_errors : Mapping[str, str] or None, optional
        Failures from providers that were applicable to this paper. These make
        an otherwise empty result ``failed`` rather than ``not_found``.

    Returns
    -------
    dict[str, Any]
        Mapping covering every enrichment update parameter.
    """
    from_crossref = _crossref_fields(crossref) if crossref is not None else {}
    from_openalex = _openalex_fields(openalex_work) if openalex_work is not None else {}
    from_pubmed = _pubmed_fields(pubmed_article) if pubmed_article is not None else {}
    from_arxiv = _arxiv_fields(arxiv_entry) if arxiv_entry is not None else {}
    from_medrxiv = _medrxiv_fields(medrxiv_entry) if medrxiv_entry is not None else {}
    from_biorxiv = _biorxiv_fields(biorxiv_entry) if biorxiv_entry is not None else {}
    from_chemrxiv = _chemrxiv_fields(chemrxiv_entry) if chemrxiv_entry is not None else {}
    update = {field: '' for field in enrichment_update_fields()}

    for column in FILL_COLUMNS + CROSSREF_PREFERRED:
        update[column] = (from_crossref.get(column)
                          or from_openalex.get(column)
                          or from_pubmed.get(column)
                          or from_arxiv.get(column)
                          or from_medrxiv.get(column)
                          or from_biorxiv.get(column)
                          or from_chemrxiv.get(column)
                          or '')
    for column in OPENALEX_ONLY:
        update[column] = from_openalex.get(column) if from_openalex.get(column) is not None else ''
    update['referenced_works_count'] = (from_openalex.get('referenced_works_count')
                                        or from_crossref.get('referenced_works_count') or 0)
    update['is_retracted'] = int(bool(from_crossref.get('is_retracted'))
                                 or bool(from_openalex.get('is_retracted')))
    update['is_oa'] = int(bool(from_openalex.get('is_oa'))
                          or bool(from_arxiv.get('is_oa'))
                          or bool(from_medrxiv.get('is_oa'))
                          or bool(from_biorxiv.get('is_oa'))
                          or bool(from_chemrxiv.get('is_oa')))
    for preprint in (from_arxiv, from_medrxiv, from_biorxiv, from_chemrxiv):
        if not update['oa_status'] and preprint.get('oa_status'):
            update['oa_status'] = preprint['oa_status']
    # OPENALEX_ONLY blanks the licence for a row OpenAlex has no record of,
    # which would discard the one the preprint server states on the posting.
    for preprint in (from_medrxiv, from_biorxiv, from_chemrxiv):
        if not update['license'] and preprint.get('license'):
            update['license'] = preprint['license']
    update['cited_by_count'] = from_openalex.get('cited_by_count') or 0

    found = [source for source, record in (('crossref', crossref),
                                           ('openalex', openalex_work),
                                           ('pubmed', pubmed_article),
                                           ('arxiv', arxiv_entry),
                                           ('medrxiv', medrxiv_entry),
                                           ('biorxiv', biorxiv_entry),
                                           ('chemrxiv', chemrxiv_entry))
             if record is not None]
    provider_errors = dict(provider_errors or {})
    if not found and provider_errors:
        status = 'failed'
    elif not found:
        status = 'not_found'
    elif len(found) == len(requested):
        status = 'succeeded'
    else:
        status = 'partial'

    now = utc_now()
    provenance = _provenance(crossref, openalex_work, pubmed_article, arxiv_entry,
                             medrxiv_entry, biorxiv_entry, chemrxiv_entry)
    if provider_errors:
        provenance['provider_errors'] = provider_errors
    update.update({
        'paper_id': paper_id,
        'enrichment_sources': ';'.join(found),
        'enrichment_json': provenance,
        'enriched_at': now if found else '',
        'enrichment_status': status,
        'updated_at': now,
    })
    return update


def _preserve_stored_values(update: _Record, stored: Mapping[str, Any]) -> None:
    """Keep non-empty enrichment values when a partial refresh cannot replace them.

    A zero is a valid fresh value, but during a provider failure it is also the
    default produced for missing counts and flags. Keeping the previous non-zero
    value until every requested provider completes avoids turning an outage into
    an apparent metadata change.
    """
    for column in PRESERVED_ON_PROVIDER_ERROR:
        if update.get(column) in ('', None, 0) and stored[column] not in ('', None, 0):
            update[column] = stored[column]


def _position_label(index: int, total: int, authorship: Mapping[str, Any] | None) -> str:
    """Choose a shared author-position label across both providers."""
    if authorship and _text(authorship.get('author_position')):
        return _text(authorship.get('author_position'))
    if index == 0:
        return 'first'
    return 'last' if index == total - 1 else 'middle'


def _safe_orcid(value: object) -> str:
    """Normalize an ORCID, returning an empty string for malformed values."""
    try:
        return crossref_client.normalize_orcid(value)
    except ValueError:
        return ''


def _author_rows(paper_id: str,
                crossref: Mapping[str, Any] | None,
                openalex_work: Mapping[str, Any] | None) -> list[_Record]:
    """Build one ``paper_authors`` row per author and affiliation.

    OpenAlex authorships are the spine because they alone carry ORCIDs,
    disambiguated author identifiers and ROR-matched institutions. Crossref
    supplies the deposited given/family split by ordinal, and extends the list
    when OpenAlex reports a truncated author set.

    Parameters
    ----------
    paper_id : str
        Paper the rows belong to.
    crossref : Mapping[str, Any] or None
        Crossref work message.
    openalex_work : Mapping[str, Any] or None
        OpenAlex work record.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the ``paper_authors`` table.
    """
    authorships = list((openalex_work or {}).get('authorships') or [])
    crossref_authors = list((crossref or {}).get('author') or [])
    rows: list[_Record] = []

    for index, authorship in enumerate(authorships):
        author = (authorship or {}).get('author') or {}
        deposited = crossref_authors[index] if index < len(crossref_authors) else {}
        institutions = list(authorship.get('institutions') or [])
        raw_affiliations = list(authorship.get('raw_affiliation_strings') or [])
        base = {
            'paper_id': paper_id,
            'author_position': index,
            'position_label': _position_label(index, len(authorships), authorship),
            'display_name': _text(author.get('display_name')),
            'given_name': _text(deposited.get('given')),
            'family_name': _text(deposited.get('family')),
            'orcid': _safe_orcid(author.get('orcid')) or _safe_orcid(deposited.get('ORCID')),
            'is_corresponding': int(bool(authorship.get('is_corresponding'))),
            'openalex_author_id': _short_openalex_id(author.get('id')),
            'institution_name': '',
            'institution_ror': '',
            'institution_country': '',
            'source': 'openalex',
        }
        if not institutions:
            rows.append({**base, 'affiliation_rank': 0,
                         'affiliation': raw_affiliations[0] if raw_affiliations else ''})
            continue
        for rank, institution in enumerate(institutions):
            institution = institution or {}
            rows.append({
                **base,
                'affiliation_rank': rank,
                'affiliation': raw_affiliations[rank] if rank < len(raw_affiliations) else '',
                'institution_name': _text(institution.get('display_name')),
                'institution_ror': _short_openalex_id(institution.get('ror')),
                'institution_country': _text(institution.get('country_code')),
            })

    for index in range(len(authorships), len(crossref_authors)):
        deposited = crossref_authors[index] or {}
        affiliations = list(deposited.get('affiliation') or [])
        base = {
            'paper_id': paper_id,
            'author_position': index,
            'position_label': _position_label(index, len(crossref_authors), None),
            'display_name': ' '.join(part for part in [_text(deposited.get('given')),
                                                       _text(deposited.get('family'))] if part),
            'given_name': _text(deposited.get('given')),
            'family_name': _text(deposited.get('family')),
            'orcid': _safe_orcid(deposited.get('ORCID')),
            'is_corresponding': 0,
            'openalex_author_id': '',
            'institution_name': '',
            'institution_ror': '',
            'institution_country': '',
            'source': 'crossref',
        }
        if not affiliations:
            rows.append({**base, 'affiliation_rank': 0, 'affiliation': ''})
            continue
        for rank, affiliation in enumerate(affiliations):
            rows.append({**base, 'affiliation_rank': rank,
                         'affiliation': _text((affiliation or {}).get('name'))})
    return rows


def _subject_rows(paper_id: str, openalex_work: Mapping[str, Any] | None) -> list[_Record]:
    """Build ``paper_subjects`` rows from an OpenAlex work.

    Crossref retired subject assignment and returns an empty list, so subjects
    come from OpenAlex only. Topics, concepts, keywords, sustainable
    development goals and the topic hierarchy are kept in separate schemes.

    Parameters
    ----------
    paper_id : str
        Paper the rows belong to.
    openalex_work : Mapping[str, Any] or None
        OpenAlex work record.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the ``paper_subjects`` table.
    """
    if not openalex_work:
        return []
    rows: list[_Record] = []
    primary_id = _short_openalex_id((openalex_work.get('primary_topic') or {}).get('id'))

    for rank, topic in enumerate(openalex_work.get('topics') or []):
        topic = topic or {}
        identifier = _short_openalex_id(topic.get('id'))
        if not identifier:
            continue
        rows.append({
            'paper_id': paper_id, 'scheme': 'topic', 'subject_id': identifier,
            'display_name': _text(topic.get('display_name')), 'score': topic.get('score'),
            'subject_rank': rank, 'is_primary': int(identifier == primary_id),
            'parent_field': _text((topic.get('field') or {}).get('display_name')),
            'parent_domain': _text((topic.get('domain') or {}).get('display_name')),
            'source': 'openalex',
        })

    for level_name in ('subfield', 'field', 'domain'):
        entry = (openalex_work.get('primary_topic') or {}).get(level_name) or {}
        identifier = _short_openalex_id(entry.get('id'))
        if identifier:
            rows.append({
                'paper_id': paper_id, 'scheme': level_name, 'subject_id': identifier,
                'display_name': _text(entry.get('display_name')), 'is_primary': 1,
                'source': 'openalex',
            })

    for rank, concept in enumerate(openalex_work.get('concepts') or []):
        concept = concept or {}
        identifier = _short_openalex_id(concept.get('id'))
        if identifier:
            rows.append({
                'paper_id': paper_id, 'scheme': 'concept', 'subject_id': identifier,
                'display_name': _text(concept.get('display_name')), 'score': concept.get('score'),
                'subject_rank': rank, 'level': concept.get('level'), 'source': 'openalex',
            })

    for rank, keyword in enumerate(openalex_work.get('keywords') or []):
        keyword = keyword or {}
        identifier = _short_openalex_id(keyword.get('id')) or _text(keyword.get('display_name'))
        if identifier:
            rows.append({
                'paper_id': paper_id, 'scheme': 'keyword', 'subject_id': identifier,
                'display_name': _text(keyword.get('display_name')), 'score': keyword.get('score'),
                'subject_rank': rank, 'source': 'openalex',
            })

    for rank, goal in enumerate(openalex_work.get('sustainable_development_goals') or []):
        goal = goal or {}
        identifier = _short_openalex_id(goal.get('id'))
        if identifier:
            rows.append({
                'paper_id': paper_id, 'scheme': 'sdg', 'subject_id': identifier,
                'display_name': _text(goal.get('display_name')), 'score': goal.get('score'),
                'subject_rank': rank, 'source': 'openalex',
            })

    seen = set()
    unique = []
    for row in rows:
        key = (row['scheme'], row['subject_id'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _reference_rows(paper_id: str,
                   crossref: Mapping[str, Any] | None,
                   openalex_work: Mapping[str, Any] | None) -> list[_Record]:
    """Build ``paper_references`` rows from both providers.

    The two reference lists are stored side by side rather than merged:
    Crossref carries publisher-asserted DOIs, while OpenAlex covers works whose
    publisher deposited no reference list but yields its own identifiers.

    Parameters
    ----------
    paper_id : str
        Paper the rows belong to.
    crossref : Mapping[str, Any] or None
        Crossref work message.
    openalex_work : Mapping[str, Any] or None
        OpenAlex work record.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the ``paper_references`` table.
    """
    rows: list[_Record] = []
    for rank, reference in enumerate((crossref or {}).get('reference') or []):
        reference = reference or {}
        rows.append({
            'paper_id': paper_id, 'source': 'crossref', 'reference_rank': rank,
            'referenced_doi': clean_doi(reference.get('DOI')),
            'referenced_title': _text(reference.get('article-title')
                                      or reference.get('volume-title')
                                      or reference.get('journal-title')),
            'unstructured': _text(reference.get('unstructured')),
        })
    for rank, referenced in enumerate((openalex_work or {}).get('referenced_works') or []):
        identifier = _short_openalex_id(referenced)
        if identifier:
            rows.append({
                'paper_id': paper_id, 'source': 'openalex', 'reference_rank': rank,
                'referenced_openalex_id': identifier,
            })
    return rows


@dataclass(frozen=True)
class _FetchContext:
    """Inputs shared by the registry-backed enrichment fetch handlers."""

    dois: Sequence[str]
    identifiers: Sequence[str]
    email: str
    api_key: str | None
    openalex_session: provider.HTTPClient | None
    crossref_session: provider.HTTPClient | None
    pace: float
    pmids: Sequence[str] = ()
    pubmed_session: provider.HTTPClient | None = None
    pubmed_api_key: str | None = None
    arxiv_ids: Sequence[str] = ()
    arxiv_session: provider.HTTPClient | None = None
    medrxiv_dois: Sequence[str] = ()
    medrxiv_session: provider.HTTPClient | None = None
    biorxiv_dois: Sequence[str] = ()
    biorxiv_session: provider.HTTPClient | None = None
    chemrxiv_dois: Sequence[str] = ()
    chemrxiv_session: provider.HTTPClient | None = None


@dataclass
class _FetchResult:
    """Records and provider failures produced while enriching one batch."""

    crossref_works: dict[str, _Record] = field(default_factory=dict)
    openalex_by_doi: dict[str, _Record] = field(default_factory=dict)
    openalex_by_id: dict[str, _Record] = field(default_factory=dict)
    pubmed_by_pmid: dict[str, _Record] = field(default_factory=dict)
    arxiv_by_id: dict[str, _Record] = field(default_factory=dict)
    medrxiv_by_doi: dict[str, _Record] = field(default_factory=dict)
    biorxiv_by_doi: dict[str, _Record] = field(default_factory=dict)
    chemrxiv_by_doi: dict[str, _Record] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def _fetch_crossref(context: _FetchContext) -> dict[str, dict[str, _Record]]:
    """Fetch Crossref records for one enrichment batch."""
    if not context.dois:
        return {}
    return {'crossref_works': crossref_client.works_by_doi(
        context.dois, email=context.email, session=context.crossref_session,
        pace=context.pace)}


def _fetch_openalex(context: _FetchContext) -> dict[str, dict[str, _Record]]:
    """Fetch OpenAlex records for one enrichment batch."""
    records: dict[str, dict[str, _Record]] = {}
    if context.dois:
        records['openalex_by_doi'] = openalex.works_batch(
            context.dois, api_key=context.api_key, session=context.openalex_session,
            mailto=context.email)
    if context.identifiers:
        records['openalex_by_id'] = openalex.works_batch(
            context.identifiers, filter_name='ids.openalex', api_key=context.api_key,
            session=context.openalex_session, mailto=context.email)
    return records


def _fetch_pubmed(context: _FetchContext) -> dict[str, dict[str, _Record]]:
    """Fetch PubMed records for one enrichment batch."""
    records: dict[str, _Record] = {}
    for chunk in provider.chunked(list(context.pmids), pubmed.EFETCH_BATCH_SIZE):
        articles = pubmed.parse_articles(pubmed.efetch_ids(
            chunk, api_key=context.pubmed_api_key, email=context.email,
            session=context.pubmed_session))
        for article in articles:
            pmid = pubmed.normalize_pmid(article.get('pmid'))
            if pmid:
                records[pmid] = article
    return {'pubmed_by_pmid': records} if records else {}


def _fetch_arxiv(context: _FetchContext) -> dict[str, dict[str, _Record]]:
    """Fetch arXiv records for one enrichment batch."""
    records: dict[str, _Record] = {}
    for chunk in provider.chunked(list(context.arxiv_ids), arxiv.ID_BATCH_SIZE):
        for entry in arxiv.parse_entries(arxiv.fetch_ids(chunk, session=context.arxiv_session)):
            identifier = arxiv.normalize_arxiv_id(entry.get('arxiv_id'))
            if identifier:
                records[identifier] = entry
    return {'arxiv_by_id': records} if records else {}


def _fetch_rxiv(identifiers: Sequence[str],
                 fetcher: Callable[..., Mapping[str, Any] | None],
                 session: provider.HTTPClient | None,
                 bucket: str) -> dict[str, dict[str, _Record]]:
    """Fetch DOI-addressed preprint records for one enrichment batch."""
    records: dict[str, _Record] = {}
    for identifier in identifiers:
        entry = fetcher(identifier, session=session)
        if entry is not None:
            records[identifier] = entry
    return {bucket: records} if records else {}


def _fetch_medrxiv(context: _FetchContext) -> dict[str, dict[str, _Record]]:
    """Fetch medRxiv records for one enrichment batch."""
    return _fetch_rxiv(context.medrxiv_dois, medrxiv.fetch_doi,
                       context.medrxiv_session, 'medrxiv_by_doi')


def _fetch_biorxiv(context: _FetchContext) -> dict[str, dict[str, _Record]]:
    """Fetch bioRxiv records for one enrichment batch."""
    return _fetch_rxiv(context.biorxiv_dois, biorxiv.fetch_doi,
                       context.biorxiv_session, 'biorxiv_by_doi')


def _fetch_chemrxiv(context: _FetchContext) -> dict[str, dict[str, _Record]]:
    """Fetch chemRxiv records for one enrichment batch."""
    return _fetch_rxiv(context.chemrxiv_dois, chemrxiv.fetch_doi,
                       context.chemrxiv_session, 'chemrxiv_by_doi')


def _fetch(source_names: Sequence[str], context: _FetchContext) -> _FetchResult:
    """Fetch one batch, isolating failures when several providers were selected."""
    result = _FetchResult()
    for name in source_names:
        try:
            fetched = registry.resolve_handler(name, registry.ENRICH)(context)
        except Exception as error:
            if len(source_names) == 1:
                raise
            result.errors[name] = str(error)
            print(f'{registry.SOURCES[name].label} enrichment skipped: {error}')
            continue
        for bucket, records in fetched.items():
            getattr(result, bucket).update(records)
    return result


def _enrich_batch(conn: sqlite3.Connection,
                 candidates: Sequence[Mapping[str, Any]],
                 sources: Sequence[str],
                 email: str,
                 api_key: str | None = None,
                 references: bool = True,
                 openalex_session: provider.HTTPClient | None = None,
                 crossref_session: provider.HTTPClient | None = None,
                 pace: float = crossref_client.CROSSREF_MIN_INTERVAL,
                 pubmed_session: provider.HTTPClient | None = None,
                 pubmed_api_key: str | None = None,
                 arxiv_session: provider.HTTPClient | None = None,
                 medrxiv_session: provider.HTTPClient | None = None,
                 biorxiv_session: provider.HTTPClient | None = None,
                 chemrxiv_session: provider.HTTPClient | None = None) -> dict[str, int]:
    """Fetch, map and store enrichment for one batch of papers.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    candidates : Sequence[Mapping[str, Any]]
        Candidate corpus rows to enrich.
    sources : Sequence[str]
        Providers to query.
    email : str
        Contact email sent to both providers.
    api_key : str or None, optional
        OpenAlex API key to attach.
    references : bool, default=True
        Whether to store reference lists.
    openalex_session : provider.HTTPClient or None, optional
        HTTP client used for OpenAlex requests.
    crossref_session : provider.HTTPClient or None, optional
        HTTP session used for Crossref requests.
    pace : float, optional
        Seconds to wait between consecutive Crossref requests.
    pubmed_session : provider.HTTPClient or None, optional
        HTTP client used for PubMed requests.
    pubmed_api_key : str or None, optional
        NCBI API key to attach to PubMed requests.
    arxiv_session : provider.HTTPClient or None, optional
        HTTP client used for arXiv requests.
    medrxiv_session : provider.HTTPClient or None, optional
        HTTP client used for medRxiv requests.
    biorxiv_session : provider.HTTPClient or None, optional
        HTTP client used for bioRxiv requests.
    chemrxiv_session : provider.HTTPClient or None, optional
        HTTP client used for chemRxiv requests.

    Returns
    -------
    dict[str, int]
        Counts of each resulting status and of stored child rows.
    """
    summary = {status: 0 for status in
               ('succeeded', 'partial', 'not_found', 'unresolved', 'failed')}
    summary.update({'authors': 0, 'subjects': 0, 'references': 0})
    if not candidates:
        return summary

    by_doi, by_openalex, unresolved = _partition_candidates(candidates)
    by_pmid = _pubmed_candidates(candidates) if 'pubmed' in sources else {}
    by_arxiv = _arxiv_candidates(candidates) if 'arxiv' in sources else {}
    by_medrxiv = _medrxiv_candidates(candidates) if 'medrxiv' in sources else {}
    by_biorxiv = _biorxiv_candidates(candidates) if 'biorxiv' in sources else {}
    by_chemrxiv = _chemrxiv_candidates(candidates) if 'chemrxiv' in sources else {}
    openalex_papers = set(by_openalex.values())
    pubmed_papers = set(by_pmid.values())
    arxiv_papers = set(by_arxiv.values())
    medrxiv_papers = set(by_medrxiv.values())
    biorxiv_papers = set(by_biorxiv.values())
    chemrxiv_papers = set(by_chemrxiv.values())
    # A row known only to PubMed, arXiv, medRxiv, bioRxiv, or chemRxiv carries no DOI or
    # OpenAlex ID, so _partition_candidates reports it unresolved. It is
    # resolvable whenever the provider that already knows its identifier is
    # being queried.
    identifier_papers = (pubmed_papers | arxiv_papers | medrxiv_papers | biorxiv_papers
                         | chemrxiv_papers)
    identifier_only = [paper_id for paper_id in unresolved if paper_id in identifier_papers]
    unresolved = [paper_id for paper_id in unresolved if paper_id not in identifier_papers]
    if unresolved:
        set_enrichment_status(conn, unresolved, 'unresolved')
        summary['unresolved'] += len(unresolved)
    if not by_doi and not by_openalex and not identifier_only:
        return summary

    fetched = _fetch(sources, _FetchContext(
        dois=list(by_doi), identifiers=list(by_openalex), email=email, api_key=api_key,
        openalex_session=openalex_session, crossref_session=crossref_session, pace=pace,
        pmids=list(by_pmid), pubmed_session=pubmed_session, pubmed_api_key=pubmed_api_key,
        arxiv_ids=list(by_arxiv), arxiv_session=arxiv_session,
        medrxiv_dois=list(by_medrxiv), medrxiv_session=medrxiv_session,
        biorxiv_dois=list(by_biorxiv), biorxiv_session=biorxiv_session,
        chemrxiv_dois=list(by_chemrxiv), chemrxiv_session=chemrxiv_session))
    pubmed_by_paper = {paper_id: fetched.pubmed_by_pmid[pmid]
                       for pmid, paper_id in by_pmid.items() if pmid in fetched.pubmed_by_pmid}
    arxiv_by_paper = {paper_id: fetched.arxiv_by_id[identifier]
                      for identifier, paper_id in by_arxiv.items()
                      if identifier in fetched.arxiv_by_id}
    medrxiv_by_paper = {paper_id: fetched.medrxiv_by_doi[identifier]
                        for identifier, paper_id in by_medrxiv.items()
                        if identifier in fetched.medrxiv_by_doi}
    biorxiv_by_paper = {paper_id: fetched.biorxiv_by_doi[identifier]
                        for identifier, paper_id in by_biorxiv.items()
                        if identifier in fetched.biorxiv_by_doi}
    chemrxiv_by_paper = {paper_id: fetched.chemrxiv_by_doi[identifier]
                         for identifier, paper_id in by_chemrxiv.items()
                         if identifier in fetched.chemrxiv_by_doi}

    targets = [(paper_id, fetched.crossref_works.get(key),
                fetched.openalex_by_doi.get(key), True)
               for key, paper_id in by_doi.items()]
    targets += [(paper_id, None, fetched.openalex_by_id.get(key), False)
                for key, paper_id in by_openalex.items()]
    targets += [(paper_id, None, None, False) for paper_id in identifier_only]

    updates, authors, subjects, reference_records = [], [], [], []
    for paper_id, crossref_work, openalex_work, keyed_by_doi in targets:
        pubmed_article = pubmed_by_paper.get(paper_id)
        arxiv_entry = arxiv_by_paper.get(paper_id)
        medrxiv_entry = medrxiv_by_paper.get(paper_id)
        biorxiv_entry = biorxiv_by_paper.get(paper_id)
        chemrxiv_entry = chemrxiv_by_paper.get(paper_id)
        requested = [source for source in sources
                     if (source == 'crossref' and keyed_by_doi)
                     or (source == 'openalex'
                         and (keyed_by_doi or paper_id in openalex_papers))
                     or (source == 'pubmed' and paper_id in pubmed_papers)
                     or (source == 'arxiv' and paper_id in arxiv_papers)
                     or (source == 'medrxiv' and paper_id in medrxiv_papers)
                     or (source == 'biorxiv' and paper_id in biorxiv_papers)
                     or (source == 'chemrxiv' and paper_id in chemrxiv_papers)]
        errors = {source: fetched.errors[source]
                  for source in requested if source in fetched.errors}
        update = _merge_fields(paper_id, crossref_work, openalex_work, requested,
                              pubmed_article, arxiv_entry, medrxiv_entry, biorxiv_entry,
                              chemrxiv_entry, provider_errors=errors)
        if errors:
            stored = conn.execute(
                'SELECT * FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()
            if stored is not None:
                _preserve_stored_values(update, stored)
        updates.append(update)
        summary[update['enrichment_status']] += 1
        if update['enrichment_status'] in {'failed', 'not_found'}:
            continue
        authors.extend(_author_rows(paper_id, crossref_work, openalex_work))
        subjects.extend(_subject_rows(paper_id, openalex_work))
        subjects.extend(_pubmed_subject_rows(paper_id, pubmed_article))
        subjects.extend(_arxiv_subject_rows(paper_id, arxiv_entry))
        subjects.extend(_medrxiv_subject_rows(paper_id, medrxiv_entry))
        subjects.extend(_biorxiv_subject_rows(paper_id, biorxiv_entry))
        subjects.extend(_chemrxiv_subject_rows(paper_id, chemrxiv_entry))
        if references:
            reference_records.extend(_reference_rows(paper_id, crossref_work, openalex_work))

    for update in updates:
        update['enrichment_json'] = _json_text(update['enrichment_json'])
    # A provider outage is not evidence that its previously stored child rows
    # disappeared. Replace rows only for providers that completed this batch.
    completed_sources = [source for source in sources if source not in fetched.errors]
    write_enrichment(conn, updates, authors, subjects, reference_records,
                     sources=completed_sources)
    summary['authors'] += len(authors)
    summary['subjects'] += len(subjects)
    summary['references'] += len(reference_records)
    return summary


def _json_text(value: object) -> str:
    """Serialize a provenance document the way the corpus stores JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, sort_keys=True, default=str)


def _enrich_from_crossref_message(conn: sqlite3.Connection,
                                 paper_id: str,
                                 message: Mapping[str, Any]) -> None:
    """Store enrichment for one paper from an already-fetched Crossref work.

    This is the write path that keeps ``pm import pdfs`` from discarding the
    publisher, work type, volume, issue, pages, ISSN and language it already
    fetched while resolving a PDF's DOI.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_id : str
        Paper the Crossref record belongs to.
    message : Mapping[str, Any]
        Crossref work message.
    """
    if not paper_id or not message:
        return
    update = _merge_fields(paper_id, message, None, ['crossref'])
    update['enrichment_json'] = _json_text(update['enrichment_json'])
    write_enrichment(conn,
                     [update],
                     _author_rows(paper_id, message, None),
                     [],
                     _reference_rows(paper_id, message, None))


def enrich_papers(conn: sqlite3.Connection,
                  papers: Iterable[Mapping[str, Any]],
                  sources: Sequence[str] | None = None,
                  batch_size: int = MAX_BATCH_SIZE,
                  references: bool = True,
                  email: str | None = None,
                  api_key: str | None = None,
                  openalex_session: provider.HTTPClient | None = None,
                  crossref_session: provider.HTTPClient | None = None,
                  pace: float = crossref_client.CROSSREF_MIN_INTERVAL,
                  pubmed_session: provider.HTTPClient | None = None,
                  pubmed_api_key: str | None = None,
                  arxiv_session: provider.HTTPClient | None = None,
                  medrxiv_session: provider.HTTPClient | None = None,
                  biorxiv_session: provider.HTTPClient | None = None,
                  chemrxiv_session: provider.HTTPClient | None = None) -> dict[str, int]:
    """Enrich specific papers on an already-open corpus connection.

    Incoming rows are resolved against the corpus first, because discovery
    merges matching records and the stored row may carry a different paper
    identifier than the provider row that produced it.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    papers : Iterable[Mapping[str, Any]]
        Paper rows to enrich.
    sources : Sequence[str] or None, optional
        Providers to query. ``None`` selects every provider.
    batch_size : int, default=100
        Papers looked up per provider request.
    references : bool, default=True
        Whether to store reference lists.
    email : str or None, optional
        Contact email. Defaults to the stored Crossref setting.
    api_key : str or None, optional
        OpenAlex API key. Defaults to the configured key.
    openalex_session : provider.HTTPClient or None, optional
        HTTP client used for OpenAlex requests.
    crossref_session : provider.HTTPClient or None, optional
        HTTP session used for Crossref requests.
    pace : float, optional
        Seconds to wait between consecutive Crossref requests.
    pubmed_session : provider.HTTPClient or None, optional
        HTTP client used for PubMed requests.
    pubmed_api_key : str or None, optional
        NCBI API key. Defaults to the configured key.
    arxiv_session : provider.HTTPClient or None, optional
        HTTP client used for arXiv requests.
    medrxiv_session : provider.HTTPClient or None, optional
        HTTP client used for medRxiv requests.
    biorxiv_session : provider.HTTPClient or None, optional
        HTTP client used for bioRxiv requests.
    chemrxiv_session : provider.HTTPClient or None, optional
        HTTP client used for chemRxiv requests.

    Returns
    -------
    dict[str, int]
        Counts of each resulting status and of stored child rows.
    """
    sources = _configured_sources(sources)
    email = crossref_client.resolve_email(email) if 'crossref' in sources else (email or '')
    if not email and 'pubmed' in sources:
        email = pubmed.configured_email()
    api_key = api_key if api_key is not None else openalex.configured_api_key()
    if pubmed_api_key is None and 'pubmed' in sources:
        pubmed_api_key = pubmed.configured_api_key()

    resolved: dict[str, _Record] = {}
    stored = {row['paper_id']: row for row in paper_rows(conn)}
    for paper in papers:
        paper_id = str(paper.get('paper_id') or '')
        row = stored.get(paper_id) or find_paper(conn, paper)
        if row:
            resolved[str(row['paper_id'])] = row

    summary = {status: 0 for status in
               ('succeeded', 'partial', 'not_found', 'unresolved', 'failed')}
    summary.update({'authors': 0, 'subjects': 0, 'references': 0})
    candidates = list(resolved.values())
    batch_size = max(1, min(batch_size, MAX_BATCH_SIZE))
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        for key, value in _enrich_batch(conn, batch, sources, email, api_key, references,
                                       openalex_session, crossref_session, pace,
                                       pubmed_session, pubmed_api_key,
                                       arxiv_session, medrxiv_session,
                                       biorxiv_session, chemrxiv_session).items():
            summary[key] += value
    return summary


def _selected_statuses(force: bool, retry_failed: bool) -> tuple[str, ...]:
    """Choose which enrichment statuses a run re-processes."""
    statuses = ['pending']
    if force:
        statuses.extend(['succeeded', 'partial', 'not_found'])
    if retry_failed:
        statuses.append('failed')
    return tuple(statuses)


def enrich_corpus(db_path: str | PathLike[str] = 'papers.db',
                  sources: Sequence[str] | None = None,
                  batch_size: int = MAX_BATCH_SIZE,
                  limit: int | None = None,
                  force: bool = False,
                  retry_failed: bool = False,
                  refresh_after: int = 0,
                  references: bool = True,
                  resolve_references: bool = False,
                  email: str | None = None,
                  api_key: str | None = None,
                  openalex_session: provider.HTTPClient | None = None,
                  crossref_session: provider.HTTPClient | None = None,
                  pace: float = crossref_client.CROSSREF_MIN_INTERVAL,
                  pubmed_session: provider.HTTPClient | None = None,
                  pubmed_api_key: str | None = None,
                  arxiv_session: provider.HTTPClient | None = None,
                  medrxiv_session: provider.HTTPClient | None = None,
                  biorxiv_session: provider.HTTPClient | None = None,
                  chemrxiv_session: provider.HTTPClient | None = None) -> dict[str, int]:
    """Supplement every candidate paper in a corpus with provider metadata.

    Progress is committed after each batch, so an interrupted or budget-limited
    run keeps the work it already did and a later run resumes from the first
    paper that is still pending.

    Parameters
    ----------
    db_path : str or os.PathLike[str], default='papers.db'
        Path to the SQLite paper corpus.
    sources : Sequence[str] or None, optional
        Providers to query. ``None`` selects every provider.
    batch_size : int, default=100
        Papers looked up per provider request.
    limit : int or None, optional
        Stop after enriching this many papers.
    force : bool, default=False
        Re-enrich papers whose enrichment already succeeded.
    retry_failed : bool, default=False
        Retry papers whose previous enrichment failed.
    refresh_after : int, default=0
        Re-enrich succeeded papers older than this many days. ``0`` disables it.
    references : bool, default=True
        Whether to store reference lists.
    resolve_references : bool, default=False
        Whether to resolve OpenAlex reference identifiers to DOIs afterwards.
    email : str or None, optional
        Contact email. Defaults to the stored Crossref setting.
    api_key : str or None, optional
        OpenAlex API key. Defaults to the configured key.
    openalex_session : provider.HTTPClient or None, optional
        HTTP client used for OpenAlex requests.
    crossref_session : provider.HTTPClient or None, optional
        HTTP session used for Crossref requests.
    pace : float, optional
        Seconds to wait between consecutive Crossref requests.
    pubmed_session : provider.HTTPClient or None, optional
        HTTP client used for PubMed requests.
    pubmed_api_key : str or None, optional
        NCBI API key. Defaults to the configured key.
    arxiv_session : provider.HTTPClient or None, optional
        HTTP client used for arXiv requests.
    medrxiv_session : provider.HTTPClient or None, optional
        HTTP client used for medRxiv requests.
    biorxiv_session : provider.HTTPClient or None, optional
        HTTP client used for bioRxiv requests.
    chemrxiv_session : provider.HTTPClient or None, optional
        HTTP client used for chemRxiv requests.

    Returns
    -------
    dict[str, int]
        Counts of each resulting status and of stored child rows.

    Raises
    ------
    ValueError
        If the provider selection, batch size, or contact email is invalid.
    RuntimeError
        If a provider request cannot be completed.
    """
    sources = _configured_sources(sources)
    if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise ValueError(f'batch_size must be between 1 and {MAX_BATCH_SIZE}.')
    email = crossref_client.resolve_email(email) if 'crossref' in sources else (email or '')
    if not email and 'pubmed' in sources:
        email = pubmed.configured_email()
    api_key = api_key if api_key is not None else openalex.configured_api_key()
    if pubmed_api_key is None and 'pubmed' in sources:
        pubmed_api_key = pubmed.configured_api_key()
    statuses = _selected_statuses(force, retry_failed)
    refreshed_before = ''
    if refresh_after > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=refresh_after)
        refreshed_before = cutoff.isoformat(timespec='seconds')

    summary = {status: 0 for status in
               ('succeeded', 'partial', 'not_found', 'unresolved', 'failed')}
    summary.update({'authors': 0, 'subjects': 0, 'references': 0, 'batches': 0})

    with connect(db_path) as conn:
        remaining = limit
        after_rowid = 0
        with tqdm(total=limit, desc='Enriching Papers', colour='#FFA500') as progress:
            while remaining is None or remaining > 0:
                page = batch_size if remaining is None else min(batch_size, remaining)
                candidates = enrichment_candidates(conn, statuses=statuses,
                                                   after_rowid=after_rowid, limit=page,
                                                   refreshed_before=refreshed_before)
                if not candidates:
                    break
                after_rowid = int(candidates[-1]['rowid'])
                counts = _enrich_batch(conn, candidates, sources, email, api_key, references,
                                      openalex_session, crossref_session, pace,
                                      pubmed_session, pubmed_api_key, arxiv_session,
                                      medrxiv_session, biorxiv_session, chemrxiv_session)
                for key, value in counts.items():
                    summary[key] += value
                summary['batches'] += 1
                progress.update(len(candidates))
                if remaining is not None:
                    remaining -= len(candidates)
        if resolve_references:
            _resolve_reference_targets(conn)
    return summary


def resolve_reference_dois(db_path: str | PathLike[str] = 'papers.db',
                           batch_size: int = MAX_BATCH_SIZE,
                           api_key: str | None = None,
                           session: provider.HTTPClient | None = None,
                           mailto: str = '') -> int:
    """Resolve stored OpenAlex reference identifiers to DOIs.

    Identifiers are deduplicated across the whole corpus first, so a work cited
    by many papers costs one lookup rather than one lookup per citing paper.

    Parameters
    ----------
    db_path : str or os.PathLike[str], default='papers.db'
        Path to the SQLite paper corpus.
    batch_size : int, default=100
        Identifiers looked up per OpenAlex request.
    api_key : str or None, optional
        OpenAlex API key. Defaults to the configured key.
    session : provider.HTTPClient or None, optional
        HTTP client used for OpenAlex requests.
    mailto : str, default=''
        Contact email sent with the requests.

    Returns
    -------
    int
        Number of reference rows given a DOI.
    """
    api_key = api_key if api_key is not None else openalex.configured_api_key()
    updated = 0
    with connect(db_path) as conn:
        pending = [row[0] for row in conn.execute(
            "SELECT DISTINCT referenced_openalex_id FROM paper_references "
            "WHERE referenced_openalex_id <> '' AND referenced_doi = ''"
        ).fetchall()]
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            works = openalex.works_batch(chunk, filter_name='ids.openalex',
                                         select=('id', 'doi'), api_key=api_key,
                                         session=session, mailto=mailto)
            resolved = [(clean_doi(work.get('doi')), identifier)
                        for identifier, work in works.items() if clean_doi(work.get('doi'))]
            if not resolved:
                continue
            cursor = conn.executemany(
                'UPDATE paper_references SET referenced_doi = ? WHERE referenced_openalex_id = ?',
                resolved,
            )
            updated += cursor.rowcount if cursor.rowcount > 0 else 0
            conn.commit()
        _resolve_reference_targets(conn)
    return updated


def _resolve_reference_targets(conn: sqlite3.Connection) -> int:
    """Link referenced DOIs to papers already present in the corpus.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.

    Returns
    -------
    int
        Number of reference rows linked to a corpus paper.
    """
    cursor = conn.execute(
        """
        UPDATE paper_references
        SET referenced_paper_id = COALESCE(
            (SELECT papers.paper_id FROM papers WHERE papers.doi = paper_references.referenced_doi),
            ''
        )
        WHERE referenced_doi <> ''
        """
    )
    conn.commit()
    return cursor.rowcount if cursor.rowcount > 0 else 0
