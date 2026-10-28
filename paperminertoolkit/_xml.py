"""Small XML primitives shared by provider and corpus document readers."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def local_name(tag: str) -> str:
    """Return a lower-case XML name without its namespace or prefix.

    Parameters
    ----------
    tag : str
        Element or attribute name.

    Returns
    -------
    str
        Lower-case local name.
    """
    return tag.rsplit('}', 1)[-1].rsplit(':', 1)[-1].lower()


def element_text(element: ET.Element | None) -> str:
    """Flatten XML text and tails without adding spaces around inline tags.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element or None
        Subtree to flatten. Adjacent scientific symbols retain their original
        spacing, including text inside superscript and subscript elements.

    Returns
    -------
    str
        Whitespace-collapsed text, or an empty string for a missing element.
    """
    if element is None:
        return ''
    return ' '.join(''.join(element.itertext()).split())
