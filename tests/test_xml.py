"""Scientific text and namespace contracts for shared XML primitives."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from paperminertoolkit._xml import element_text, local_name


@pytest.mark.parametrize(('tag', 'name'), [
    ('{https://example.test/ns}TITLE', 'title'),
    ('ce:Para', 'para'),
    ('plain', 'plain'),
    ('', ''),
])
def test_local_name_handles_namespaces_and_prefixes(tag: str, name: str) -> None:
    """Compare XML names regardless of source namespace representation."""
    assert local_name(tag) == name


@pytest.mark.parametrize(('xml', 'expected'), [
    ('<p>H<sub>2</sub>O and μm<sup>2</sup></p>', 'H2O and μm2'),
    ('<p>  α\n\t<italic>β</italic>\u00a0γ </p>', 'α β γ'),
    ('<p><b>first</b>tail<i>last</i></p>', 'firsttaillast'),
    ('<p/>', ''),
])
def test_element_text_preserves_inline_adjacency_and_unicode(xml: str, expected: str) -> None:
    """Normalize only existing whitespace while preserving scientific symbols."""
    assert element_text(ET.fromstring(xml)) == expected
    assert element_text(None) == ''
