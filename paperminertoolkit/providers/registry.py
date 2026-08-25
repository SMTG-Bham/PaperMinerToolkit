"""The data sources PaperMinerToolkit knows about, and what each can be used for.

A source name used to be written out in eight places: a set in
:mod:`paperminertoolkit.workflows.search`, a tuple in :mod:`paperminertoolkit.workflows.enrichment`, two sets
in :mod:`paperminertoolkit.workflows.download`, three hand-typed ``click.Choice`` lists in
:mod:`paperminertoolkit.cli`, and a literal open-access set written twice inside one
function. They disagreed -- on whether ``all`` was a member, on ordering, and on
whether a name was lower-cased before being matched -- and adding a source meant
finding all eight. This module is the one place instead.

An entry names a source, says what it can be used for, and records the
credential it needs and the corpus column that identifies a paper to it. The
module it belongs to is stored as a dotted path and imported only when asked
for, so importing the registry costs nothing and cannot cycle back through
:mod:`paperminertoolkit.settings`, which several source modules import.

The registry also owns the operation targets used by search, enrichment, and
asset download. Those callables are resolved at use time, so the orchestration
modules do not maintain parallel dispatch dictionaries and replacing a handler
for testing still takes effect immediately.

Order is data here, not statement order. Each capability keeps its own sequence,
because they genuinely differ and each was chosen: a search puts arXiv last so a
published record wins over its preprint, while a PDF download puts arXiv last
for the different reason that it is the least likely to hold the version of
record. Those five orders are declared explicitly, and a test asserts each one
covers exactly the sources carrying that capability, so a source added to a
capability without being placed in its order fails loudly.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

SEARCH = 'search'
ENRICH = 'enrich'
PDF = 'pdf'
TEXT = 'text'
ABSTRACT = 'abstract'
CAPABILITIES = (SEARCH, ENRICH, PDF, TEXT, ABSTRACT)


@dataclass(frozen=True)
class Source:
    """One data source and the capabilities PaperMinerToolkit can use it for.

    Parameters
    ----------
    name : str
        Machine name, lower-case. This is the CLI value, the corpus ``sources``
        value, and the key into :data:`SOURCES`.
    label : str
        Display name, as it should read in progress bars and error messages.
    module : str
        Dotted path of the client module, imported on demand by :func:`resolve`.
    capabilities : frozenset[str]
        Subset of :data:`CAPABILITIES` this source can serve.
    identifier_column : str, default=''
        Corpus column holding this source's own identifier for a paper, when it
        has one.
    credential : str, default=''
        Settings key holding the credential this source needs, if any.
    credential_env : str, default=''
        Environment variable that also supplies that credential.
    setup_command : str, default=''
        Console script that stores the credential.
    credential_required : bool, default=False
        Whether the source is unusable without that credential. An optional
        credential raises a rate limit or a quota; a required one is the
        difference between the source answering and not.
    open_access : bool, default=False
        Whether a PDF from this source is openly licensed. A closed-access
        download is not re-advertised as the paper's public PDF URL.
    search_handler : str, default=''
        Dotted ``module:function`` target implementing search.
    enrich_handler : str, default=''
        Dotted target implementing enrichment fetches.
    pdf_handler : str, default=''
        Dotted target implementing PDF download.
    text_handler : str, default=''
        Dotted target implementing machine-readable full-text download.
    text_reachable : str, default=''
        Dotted predicate deciding whether full text can be requested for a row.
    abstract_handler : str, default=''
        Dotted target implementing abstract download.
    abstract_reachable : str, default=''
        Dotted predicate deciding whether an abstract can be requested for a row.
    """

    name: str
    label: str
    module: str
    capabilities: frozenset[str]
    identifier_column: str = ''
    credential: str = ''
    credential_env: str = ''
    setup_command: str = ''
    credential_required: bool = False
    open_access: bool = False
    search_handler: str = ''
    enrich_handler: str = ''
    pdf_handler: str = ''
    text_handler: str = ''
    text_reachable: str = ''
    abstract_handler: str = ''
    abstract_reachable: str = ''

    def has(self, capability: str) -> bool:
        """Report whether this source offers one capability.

        Parameters
        ----------
        capability : str
            One of :data:`CAPABILITIES`.

        Returns
        -------
        bool
            Whether the source can be used that way.
        """
        return capability in self.capabilities

    def handler(self, capability: str) -> str:
        """Return the operation target implementing one capability.

        Parameters
        ----------
        capability : str
            One of :data:`CAPABILITIES`.

        Returns
        -------
        str
            Dotted ``module:function`` target, or an empty string when the
            source does not implement the capability.
        """
        attribute = {
            SEARCH: 'search_handler',
            ENRICH: 'enrich_handler',
            PDF: 'pdf_handler',
            TEXT: 'text_handler',
            ABSTRACT: 'abstract_handler',
        }.get(capability)
        if attribute is None:
            raise ValueError(f'capability must be one of: {", ".join(CAPABILITIES)}')
        return str(getattr(self, attribute))


def _source(name: str,
            label: str,
            module: str,
            capabilities: Iterable[str],
            **rest: object) -> Source:
    """Build one registry entry.

    Parameters
    ----------
    name : str
        Machine name.
    label : str
        Display name.
    module : str
        Dotted module path, relative to the package.
    capabilities : Iterable[str]
        Capabilities the source offers.
    **rest : object
        Further :class:`Source` fields.

    Returns
    -------
    Source
        The registry entry.
    """
    return Source(name=name, label=label, module=f'paperminertoolkit.providers.{module}',
                  capabilities=frozenset(capabilities), **rest)  # type: ignore[arg-type]


SOURCES: dict[str, Source] = {
    entry.name: entry for entry in (
        _source('crossref', 'Crossref', 'crossref', [ENRICH],
                credential='crossref_email', credential_env='CROSSREF_EMAIL',
                setup_command='pmt config crossref-email',
                enrich_handler='paperminertoolkit.workflows.enrichment:_fetch_crossref'),
        _source('openalex', 'OpenAlex', 'openalex', [SEARCH, ENRICH, PDF, ABSTRACT],
                identifier_column='openalex_id', credential='openalex_api_key',
                credential_env='OPENALEX_API_KEY', setup_command='pmt config openalex-key',
                open_access=True,
                search_handler='paperminertoolkit.workflows.search:openalex_search',
                enrich_handler='paperminertoolkit.workflows.enrichment:_fetch_openalex',
                pdf_handler='paperminertoolkit.workflows.download:_download_openalex_pdf',
                abstract_handler='paperminertoolkit.workflows.download:_download_openalex_abstract',
                abstract_reachable='paperminertoolkit.workflows.download:_openalex_identifier'),
        _source('pubmed', 'PubMed', 'pubmed', [SEARCH, ENRICH, PDF, TEXT, ABSTRACT],
                identifier_column='pmid', credential='ncbi_api_key',
                credential_env='NCBI_API_KEY', setup_command='pmt config ncbi-key',
                search_handler='paperminertoolkit.workflows.search:pubmed_search',
                enrich_handler='paperminertoolkit.workflows.enrichment:_fetch_pubmed',
                pdf_handler='paperminertoolkit.workflows.download:_download_pubmed_pdf',
                text_handler='paperminertoolkit.workflows.download:_download_pmc_text',
                text_reachable='paperminertoolkit.workflows.download:_should_try_pmc_text',
                abstract_handler='paperminertoolkit.workflows.download:_download_pubmed_abstract',
                abstract_reachable='paperminertoolkit.workflows.download:_pubmed_abstract_reachable'),
        _source('elsevier', 'Elsevier', 'elsevier', [SEARCH, PDF, TEXT, ABSTRACT],
                identifier_column='elsevier_link', credential='elsevier_api_key',
                credential_env='ELSEVIER_API_KEY', setup_command='pmt config elsevier-key',
                credential_required=True,
                search_handler='paperminertoolkit.workflows.search:_elsevier_search',
                pdf_handler='paperminertoolkit.workflows.download:_download_elsevier_pdf',
                text_handler='paperminertoolkit.workflows.download:_download_elsevier_text',
                text_reachable='paperminertoolkit.workflows.download:_should_try_elsevier_text',
                abstract_handler='paperminertoolkit.workflows.download:_download_elsevier_abstract',
                abstract_reachable='paperminertoolkit.workflows.download:_elsevier_abstract_reachable'),
        _source('core', 'CORE', 'core', [SEARCH, PDF, ABSTRACT],
                identifier_column='core_id', credential='core_api_key',
                credential_env='CORE_API_KEY', setup_command='pmt config core-key',
                credential_required=True, open_access=True,
                search_handler='paperminertoolkit.workflows.search:core_search',
                pdf_handler='paperminertoolkit.workflows.download:_download_core_pdf',
                abstract_handler='paperminertoolkit.workflows.download:_download_core_abstract',
                abstract_reachable='paperminertoolkit.workflows.download:_core_abstract_reachable'),
        _source('unpaywall', 'Unpaywall', 'unpaywall', [PDF],
                credential='unpaywall_email', credential_env='UNPAYWALL_EMAIL',
                setup_command='pmt config unpaywall-email', credential_required=True,
                open_access=True,
                pdf_handler='paperminertoolkit.workflows.download:_download_unpaywall_pdf'),
        _source('arxiv', 'arXiv', 'arxiv', [SEARCH, ENRICH, PDF, ABSTRACT],
                identifier_column='arxiv_id', open_access=True,
                search_handler='paperminertoolkit.workflows.search:arxiv_search',
                enrich_handler='paperminertoolkit.workflows.enrichment:_fetch_arxiv',
                pdf_handler='paperminertoolkit.workflows.download:_download_arxiv_pdf',
                abstract_handler='paperminertoolkit.workflows.download:_download_arxiv_abstract',
                abstract_reachable='paperminertoolkit.workflows.download:_arxiv_identifier'),
        _source('medrxiv', 'medRxiv', 'medrxiv', [SEARCH, ENRICH, PDF, TEXT, ABSTRACT],
                identifier_column='medrxiv_doi', open_access=True,
                search_handler='paperminertoolkit.workflows.search:medrxiv_search',
                enrich_handler='paperminertoolkit.workflows.enrichment:_fetch_medrxiv',
                pdf_handler='paperminertoolkit.workflows.download:_download_medrxiv_pdf',
                text_handler='paperminertoolkit.workflows.download:_download_medrxiv_text',
                text_reachable='paperminertoolkit.workflows.download:_should_try_medrxiv_text',
                abstract_handler='paperminertoolkit.workflows.download:_download_medrxiv_abstract',
                abstract_reachable='paperminertoolkit.workflows.download:_medrxiv_identifier'),
        _source('biorxiv', 'bioRxiv', 'biorxiv', [SEARCH, ENRICH, PDF, TEXT, ABSTRACT],
                identifier_column='biorxiv_doi', open_access=True,
                search_handler='paperminertoolkit.workflows.search:biorxiv_search',
                enrich_handler='paperminertoolkit.workflows.enrichment:_fetch_biorxiv',
                pdf_handler='paperminertoolkit.workflows.download:_download_biorxiv_pdf',
                text_handler='paperminertoolkit.workflows.download:_download_biorxiv_text',
                text_reachable='paperminertoolkit.workflows.download:_should_try_biorxiv_text',
                abstract_handler='paperminertoolkit.workflows.download:_download_biorxiv_abstract',
                abstract_reachable='paperminertoolkit.workflows.download:_biorxiv_identifier'),
        _source('chemrxiv', 'chemRxiv', 'chemrxiv', [SEARCH, ENRICH, PDF, ABSTRACT],
                identifier_column='chemrxiv_doi', open_access=True,
                search_handler='paperminertoolkit.workflows.search:chemrxiv_search',
                enrich_handler='paperminertoolkit.workflows.enrichment:_fetch_chemrxiv',
                pdf_handler='paperminertoolkit.workflows.download:_download_chemrxiv_pdf',
                abstract_handler='paperminertoolkit.workflows.download:_download_chemrxiv_abstract',
                abstract_reachable='paperminertoolkit.workflows.download:_chemrxiv_identifier'),
    )
}

# One order per capability, each copied from the sequence the pipeline used
# before the registry existed. They differ deliberately.
#
# Search puts arXiv last because a published record should win over its
# preprint. PDF download leads with the open-access resolvers, which are both
# free and most likely to hold something. Abstracts lead with the providers that
# serve one directly from metadata already in hand. Text lists only the four
# sources that serve machine-readable full text rather than a PDF. Enrichment
# order is also the field precedence: Crossref is the registration authority, so
# it wins, and the preprint servers fill in behind everything else.
SEARCH_ORDER = ('elsevier', 'core', 'openalex', 'pubmed', 'arxiv',
                'medrxiv', 'biorxiv', 'chemrxiv')
ENRICH_ORDER = ('crossref', 'openalex', 'pubmed', 'arxiv', 'medrxiv', 'biorxiv', 'chemrxiv')
PDF_ORDER = ('unpaywall', 'openalex', 'core', 'elsevier', 'pubmed',
             'medrxiv', 'biorxiv', 'chemrxiv', 'arxiv')
TEXT_ORDER = ('elsevier', 'pubmed', 'medrxiv', 'biorxiv')
ABSTRACT_ORDER = ('openalex', 'pubmed', 'medrxiv', 'biorxiv', 'chemrxiv',
                  'arxiv', 'core', 'elsevier')
ORDERS: dict[str, tuple[str, ...]] = {
    SEARCH: SEARCH_ORDER,
    ENRICH: ENRICH_ORDER,
    PDF: PDF_ORDER,
    TEXT: TEXT_ORDER,
    ABSTRACT: ABSTRACT_ORDER,
}
# Client modules are imported once, on first use.
_MODULES: dict[str, ModuleType] = {}


def resolve(name: str) -> ModuleType:
    """Import and return the client module for one source.

    Parameters
    ----------
    name : str
        Source name.

    Returns
    -------
    types.ModuleType
        The source's client module.

    Raises
    ------
    KeyError
        If no source goes by that name.
    """
    if name not in _MODULES:
        _MODULES[name] = importlib.import_module(SOURCES[name].module)
    return _MODULES[name]


def resolve_handler(name: str, capability: str) -> Callable[..., Any]:
    """Resolve the callable implementing a source capability.

    The target is looked up on every call after importing its module. This keeps
    imports lazy at the registry boundary while allowing tests and applications
    to replace a handler dynamically.

    Parameters
    ----------
    name : str
        Registry source name.
    capability : str
        One of :data:`CAPABILITIES`.

    Returns
    -------
    Callable[..., Any]
        Operation callable registered for the source.

    Raises
    ------
    KeyError
        If no source goes by ``name``.
    ValueError
        If the source does not implement the capability or its target is malformed.
    TypeError
        If the registered target is not callable.
    """
    target = SOURCES[name].handler(capability)
    if not target:
        raise ValueError(f'{name} does not implement {capability}')
    module_name, separator, attribute = target.partition(':')
    if not separator or not module_name or not attribute:
        raise ValueError(f'invalid {capability} handler for {name}: {target!r}')
    handler = getattr(importlib.import_module(module_name), attribute)
    if not callable(handler):
        raise TypeError(f'{target} is not callable')
    return handler


def resolve_reachability(name: str, capability: str) -> Callable[..., Any] | None:
    """Resolve an optional per-row reachability predicate.

    Parameters
    ----------
    name : str
        Registry source name.
    capability : {'abstract', 'text'}
        Capability whose predicate is requested.

    Returns
    -------
    callable or None
        Registered predicate, or ``None`` when no predicate is required.

    Raises
    ------
    ValueError
        If the capability is not abstract or text, or the target is malformed.
    """
    if capability == TEXT:
        target = SOURCES[name].text_reachable
    elif capability == ABSTRACT:
        target = SOURCES[name].abstract_reachable
    else:
        raise ValueError('reachability is defined only for text and abstract capabilities')
    if not target:
        return None
    module_name, separator, attribute = target.partition(':')
    if not separator or not module_name or not attribute:
        raise ValueError(f'invalid {capability} reachability predicate for {name}: {target!r}')
    predicate = getattr(importlib.import_module(module_name), attribute)
    if not callable(predicate):
        raise TypeError(f'{target} is not callable')
    return predicate


def names(capability: str) -> tuple[str, ...]:
    """Return the sources offering one capability, in that capability's order.

    Parameters
    ----------
    capability : str
        One of :data:`CAPABILITIES`.

    Returns
    -------
    tuple[str, ...]
        Source names, ordered.

    Raises
    ------
    ValueError
        If ``capability`` is not one PaperMinerToolkit knows.
    """
    if capability not in ORDERS:
        raise ValueError(f'capability must be one of: {", ".join(CAPABILITIES)}')
    return ORDERS[capability]


def labels(requested: Iterable[str]) -> list[str]:
    """Return the display names of the given sources.

    Parameters
    ----------
    requested : Iterable[str]
        Source names.

    Returns
    -------
    list[str]
        Display names, in the order given.
    """
    return [SOURCES[name].label for name in requested]


def choices(capability: str) -> list[str]:
    """Return the CLI choice values for one capability, with ``all`` first.

    Parameters
    ----------
    capability : str
        One of :data:`CAPABILITIES`.

    Returns
    -------
    list[str]
        Accepted ``--source`` values.
    """
    return ['all', *names(capability)]


def identifier_columns() -> list[str]:
    """Return every corpus column that holds a source's own paper identifier.

    Returns
    -------
    list[str]
        Column names, in registry order.
    """
    return [entry.identifier_column for entry in SOURCES.values() if entry.identifier_column]


def open_access_names() -> frozenset[str]:
    """Return the sources whose PDFs are openly licensed.

    A PDF fetched from one of these may be re-advertised as the paper's public
    location; one fetched from a subscription route may not.

    Returns
    -------
    frozenset[str]
        Source names.
    """
    return frozenset(name for name, entry in SOURCES.items() if entry.open_access)


def resolve_names(requested: Iterable[str] | None,
                  capability: str,
                  preserve_order: bool = False,
                  label: str = '') -> list[str]:
    """Lower-case, validate, and expand ``all`` for one capability.

    This is the only place a ``--source`` selection is interpreted, so the three
    pipelines agree on what they accept and on how they say no.

    Parameters
    ----------
    requested : Iterable[str] or None
        Requested source names. ``None``, empty, or a selection containing
        ``all`` expands to every source with the capability.
    capability : str
        One of :data:`CAPABILITIES`.
    preserve_order : bool, default=False
        Whether an explicit selection keeps the order it was written in. A
        download tries sources in the order asked for, because naming several
        is a statement of preference; enrichment does not, because its order is
        a field-precedence rule rather than a choice. An expanded ``all`` always
        takes the capability's own order.
    label : str, default=''
        Word for the selection in a rejection message, when the capability's
        own name is not what a user typed. ``--source`` on a download selects a
        PDF source but reads as a download source.

    Returns
    -------
    list[str]
        Source names, without duplicates.

    Raises
    ------
    ValueError
        If a requested name is not a source with that capability.
    """
    available = names(capability)
    asked = [str(name).strip().lower() for name in requested or [] if str(name).strip()]
    if not asked or 'all' in asked:
        return list(available)
    unknown = [name for name in asked if name not in available]
    if unknown:
        raise ValueError(f'{label or capability} source must be one of: '
                         f'all, {", ".join(available)}')
    if preserve_order:
        return list(dict.fromkeys(asked))
    return [name for name in available if name in set(asked)]
