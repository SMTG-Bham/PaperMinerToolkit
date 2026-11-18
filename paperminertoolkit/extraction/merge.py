"""Merge the records that separate text chunks return for one paper.

A long paper is sent to the model in several chunks, and each chunk returns
its own records. The same material then appears once per chunk, often with
the same values repeated. Merging is deterministic: records are grouped by
their recipe identity fields, list items that state the same thing are kept
once, and other fields keep their first reported value. A record that only
repeats an earlier one under a qualified name, such as "Cs3Bi2I9 (thin film)"
after "Cs3Bi2I9 (powder)" with the same values, is dropped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from paperminertoolkit.extraction.values import is_missing, same_value, text_key


def identity_fields(recipe: Mapping[str, Any]) -> list[str]:
    """Return the fields that identify one record of a recipe.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Loaded extraction recipe.

    Returns
    -------
    list[str]
        The record definition's identity fields, or the first search field.
    """
    fields = recipe.get('record definition', {}).get('identity fields') or []
    return list(fields) or list(recipe['search fields'])[:1]


def record_key(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Build a normalized identity key, or ``""`` when no identity is reported.

    Parameters
    ----------
    record : Mapping[str, Any]
        Extracted record.
    fields : Sequence[str]
        Identity fields.

    Returns
    -------
    str
        Key that ignores case, punctuation and word order.
    """
    parts = [text_key(record.get(field)) for field in fields if not is_missing(record.get(field))]
    return ' | '.join(parts) if len(parts) == len(fields) else ''


def same_item(first: object, second: object) -> bool:
    """Return whether two list items state the same thing.

    Dictionary items agree when every key is missing on one side or agrees
    on both, so ``{"value": "1.5 eV", "method": "None"}`` duplicates
    ``{"value": "1.50 eV", "method": "PBE"}``. Other items are compared as
    values.

    Parameters
    ----------
    first, second : object
        Items to compare.

    Returns
    -------
    bool
        ``True`` when the items are duplicates.
    """
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        shared = False
        for key in set(first) | set(second):
            a, b = first.get(key), second.get(key)
            if is_missing(a) or is_missing(b):
                continue
            if not same_value(a, b):
                return False
            shared = True
        return shared
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return False
    return same_value(first, second)


def _populated(item: object) -> int:
    """Count an item's reported parts, so the fuller duplicate is kept."""
    if isinstance(item, Mapping):
        return sum(not is_missing(value) for value in item.values())
    return int(not is_missing(item))


def merge_items(items: Sequence[object]) -> list[object]:
    """Drop duplicate list items, keeping the fullest copy in first position.

    Parameters
    ----------
    items : Sequence[object]
        List-field items, possibly from several records.

    Returns
    -------
    list[object]
        Distinct items in order of first appearance.
    """
    kept: list[object] = []
    for item in items:
        if is_missing(item):
            continue
        for index, existing in enumerate(kept):
            if same_item(existing, item):
                if _populated(item) > _populated(existing):
                    kept[index] = item
                break
        else:
            kept.append(item)
    return kept


def _merge_into(target: dict[str, Any], record: Mapping[str, Any]) -> None:
    """Fold one record into another with the same identity."""
    for field, value in record.items():
        current = target.get(field)
        if isinstance(current, list) or isinstance(value, list):
            current_items = current if isinstance(current, list) else ([] if is_missing(current) else [current])
            new_items = value if isinstance(value, list) else ([] if is_missing(value) else [value])
            target[field] = merge_items([*current_items, *new_items])
        elif field not in target or (is_missing(current) and not is_missing(value)):
            target[field] = value


def field_items(value: object) -> list[object]:
    """Return a field's items as a list, without missing values."""
    items = value if isinstance(value, list) else [value]
    return [item for item in items if not is_missing(item)]


def _base_name(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Return the identity's first word, e.g. ``"cs3bi2i9"`` for ``"Cs3Bi2I9 (thin film)"``."""
    text = ' '.join(str(record.get(field)) for field in fields if not is_missing(record.get(field)))
    match = re.match(r'\s*([^\s(\[,;]+)', text)
    return match.group(1).casefold() if match else ''


def same_content(first: Mapping[str, Any], second: Mapping[str, Any], fields: Sequence[str]) -> bool:
    """Return whether two records report the same items in every given field.

    Parameters
    ----------
    first, second : Mapping[str, Any]
        Records to compare.
    fields : Sequence[str]
        Fields to compare, normally the recipe's non-identity search fields.

    Returns
    -------
    bool
        ``True`` when at least one field is reported and every field holds
        the same items (see :func:`same_item`) on both sides.
    """
    reported = False
    for field in fields:
        a, b = field_items(first.get(field)), field_items(second.get(field))
        if not a and not b:
            continue
        if (len(a) != len(b) or not all(any(same_item(x, y) for y in b) for x in a)
                or not all(any(same_item(y, x) for x in a) for y in b)):
            return False
        reported = True
    return reported


def merge_chunk_records(records: Sequence[Mapping[str, Any]], recipe: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Merge records that describe the same subject.

    Parameters
    ----------
    records : Sequence[Mapping[str, Any]]
        Records from one paper, typically from several text chunks.
    recipe : Mapping[str, Any]
        Loaded extraction recipe supplying the identity fields.

    Returns
    -------
    list[dict[str, Any]]
        One record per identity, in order of first appearance. Records with
        no reported identity are kept as they are, with duplicate list items
        removed. A record whose identity starts with the same word as an
        earlier record's and whose other fields hold the same items is
        dropped as a repeat.
    """
    fields = identity_fields(recipe)
    merged: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_key(record, fields)
        if key and key in by_key:
            _merge_into(by_key[key], record)
            continue
        copy: dict[str, Any] = {}
        _merge_into(copy, record)
        merged.append(copy)
        if key:
            by_key[key] = copy
    content = [field for field in recipe['search fields'] if field not in fields]
    kept: list[dict[str, Any]] = []
    for record in merged:
        base = _base_name(record, fields)
        if base and any(_base_name(earlier, fields) == base and same_content(earlier, record, content)
                        for earlier in kept):
            continue
        kept.append(record)
    return kept
