"""Unit tests for the data-source registry."""

from __future__ import annotations

import pytest

import paperscraper.cli as cli
import paperscraper.download as download
import paperscraper.enrichment as enrichment
import paperscraper.search as search
from paperscraper import sources


@pytest.mark.parametrize('capability', sources.CAPABILITIES)
def test_each_capability_order_covers_exactly_its_sources(capability: str) -> None:
    """Keep an order and its capability from drifting apart.

    A source added to a capability without being placed in that capability's
    order would otherwise be silently skipped by every pipeline that reads it.
    """
    ordered = sources.names(capability)
    declared = {name for name, entry in sources.SOURCES.items() if entry.has(capability)}

    assert set(ordered) == declared
    assert len(ordered) == len(set(ordered))


@pytest.mark.parametrize('capability', sources.CAPABILITIES)
def test_each_declared_capability_resolves_to_a_handler(capability: str) -> None:
    """Make the registry the executable source of pipeline dispatch."""
    for name in sources.names(capability):
        assert callable(sources.resolve_handler(name, capability))


@pytest.mark.parametrize('capability', [sources.TEXT, sources.ABSTRACT])
def test_asset_reachability_predicates_resolve_from_the_registry(capability: str) -> None:
    """Keep per-row source gating beside the corresponding asset handler."""
    for name in sources.names(capability):
        assert callable(sources.resolve_reachability(name, capability))


def test_every_source_declares_at_least_one_capability() -> None:
    """Reject a registry entry nothing can ever use."""
    for name, entry in sources.SOURCES.items():
        assert entry.capabilities, f'{name} declares no capability'
        assert entry.capabilities <= set(sources.CAPABILITIES)
        assert entry.name == name
        assert entry.label
        assert entry.module.startswith('paperscraper.')


def test_a_credential_bearing_source_names_its_setup_command() -> None:
    """Keep the credential, its environment variable, and its command together."""
    for entry in sources.SOURCES.values():
        if entry.credential:
            assert entry.credential_env, f'{entry.name} has no environment variable'
            assert entry.setup_command, f'{entry.name} has no setup command'
        else:
            assert not entry.credential_env
            assert not entry.setup_command


def test_resolve_names_expands_all_into_the_capability_order() -> None:
    """Answer an unscoped request with every source, in the declared order."""
    for capability in sources.CAPABILITIES:
        expected = list(sources.names(capability))
        assert sources.resolve_names(None, capability) == expected
        assert sources.resolve_names([], capability) == expected
        assert sources.resolve_names(['all'], capability) == expected
        assert sources.resolve_names(['all', 'openalex'], capability) == expected


def test_resolve_names_lower_cases_deduplicates_and_reorders() -> None:
    """Read a selection the same way whichever pipeline was handed it."""
    assert sources.resolve_names(['ARXIV', 'OpenAlex'], sources.SEARCH) == ['openalex', 'arxiv']
    assert sources.resolve_names(['arxiv', 'arxiv'], sources.SEARCH) == ['arxiv']
    assert sources.resolve_names([' arxiv '], sources.SEARCH) == ['arxiv']


def test_resolve_names_rejects_a_source_that_lacks_the_capability() -> None:
    """Refuse a name that exists but cannot do what was asked of it."""
    with pytest.raises(ValueError, match='search source must be one of'):
        sources.resolve_names(['crossref'], sources.SEARCH)
    with pytest.raises(ValueError, match='enrich source must be one of'):
        sources.resolve_names(['unpaywall'], sources.ENRICH)
    with pytest.raises(ValueError, match='text source must be one of'):
        sources.resolve_names(['arxiv'], sources.TEXT)

    with pytest.raises(ValueError, match='does not implement search'):
        sources.resolve_handler('crossref', sources.SEARCH)


def test_names_rejects_a_capability_that_does_not_exist() -> None:
    """Fail a typo rather than answering with an empty source list."""
    with pytest.raises(ValueError, match='capability must be one of'):
        sources.names('fulltext')


def test_choices_offer_all_first_for_the_command_line() -> None:
    """Present the default first, as the CLI options already did."""
    for capability in sources.CAPABILITIES:
        assert sources.choices(capability) == ['all', *sources.names(capability)]


def test_resolve_imports_each_client_module_once() -> None:
    """Return the real module, and the same object on a second request."""
    assert sources.resolve('arxiv') is sources.resolve('arxiv')
    assert sources.resolve('medrxiv').SERVER == 'medrxiv'
    assert sources.resolve('chemrxiv').WEB_URL.endswith('chemrxiv.org')


def test_open_access_names_match_the_sources_that_declare_it() -> None:
    """Keep the open-access set derived rather than written out again."""
    assert sources.open_access_names() == frozenset(
        name for name, entry in sources.SOURCES.items() if entry.open_access)
    assert 'elsevier' not in sources.open_access_names()
    assert 'unpaywall' in sources.open_access_names()


def test_identifier_columns_are_unique_and_named_for_their_source() -> None:
    """Give each source at most one corpus column, and no two the same one."""
    columns = sources.identifier_columns()
    assert len(columns) == len(set(columns))


def _choice_values(command: object, option: str) -> list[str]:
    """Return the accepted values of one CLI option.

    Parameters
    ----------
    command : object
        Click command to inspect.
    option : str
        Long option name, without the leading dashes.

    Returns
    -------
    list[str]
        Accepted values.
    """
    for parameter in command.params:
        if f'--{option}' in parameter.opts:
            return list(parameter.type.choices)
    raise AssertionError(f'--{option} not found on {command.name}')


def test_the_pipelines_and_the_cli_read_the_same_source_lists() -> None:
    """Keep one registry from drifting apart across the places that consume it.

    These lists used to be written out separately in search, enrichment,
    download, and three times in the CLI, and they disagreed.
    """
    assert search.SEARCH_SOURCES == {'all', *sources.names(sources.SEARCH)}
    assert enrichment.ENRICHMENT_SOURCES == sources.names(sources.ENRICH)
    assert _choice_values(cli.paper_search, 'source') == sources.choices(sources.SEARCH)
    assert _choice_values(cli.enrich, 'source') == sources.choices(sources.ENRICH)
    assert set(_choice_values(cli.download, 'source')) == {'all', *download.DOWNLOAD_SOURCES}
