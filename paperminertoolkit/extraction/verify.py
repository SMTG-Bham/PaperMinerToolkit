"""Check extracted records against their source text and recipe rules.

The checks are declared per recipe under ``"checks"`` and run only when a
scrape or post-processing step asks for them (``pmt scrape --verify``,
``pmt postprocess``). Each removed or edited item is reported, so an audit
file can show exactly what was dropped and why.

Recipe keys
-----------
``"fields"``
    Fields to check. Defaults to every non-identity search field.
``"value key"``
    Item key holding the value in list-of-object fields. Defaults to ``"value"``.
``"values in source"``
    Drop items whose numbers are not printed in the source text.
``"tolerance"``
    Relative tolerance for ``"values in source"``. Defaults to ``0.005``.
``"unit"``
    Unit, or list of units, values may be written in; drop values written with
    any other unit (unitless values are kept), e.g. ``["eV", "meV"]``.
``"exclude terms"``
    Drop items whose text mentions one of these terms.
``"exclude terms keys"``
    Item keys searched for excluded terms. Defaults to every key; limit it to
    the parts that say what a value is, since a method description may name
    an excluded quantity only as context.
``"unsupported terms"``
    Map a term to the phrases that justify it, e.g. ``{"SOC": ["spin-orbit"]}``.
    When the source contains none of the phrases, the term is removed from
    the item's descriptive parts.
``"drop empty"``
    Drop items without a value and records left with no reported fields.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from paperminertoolkit.extraction.merge import identity_fields
from paperminertoolkit.extraction.values import is_missing, numbers, printed_in

DEFAULT_SOURCE_TOLERANCE = 0.005
_CHECK_TYPES: dict[str, type | tuple[type, ...]] = {
    'fields': list,
    'value key': str,
    'values in source': bool,
    'tolerance': (int, float),
    'unit': (str, list),
    'exclude terms': list,
    'exclude terms keys': list,
    'unsupported terms': dict,
    'drop empty': bool,
}
DROPPED_COLUMNS = ['Paper id', 'Record', 'Field', 'Item', 'Check', 'Action', 'Detail']
_DASHES = dict.fromkeys([*range(0x2010, 0x2016), 0x2212], '-')
# Words that join two numbers in a range, e.g. '3.9 to 4.66 eV', rather than units.
_RANGE_WORDS = frozenset({'to', 'and', 'or'})


def validate_checks(checks: object, source: str, search_fields: Mapping[str, Any]) -> None:
    """Validate a recipe's ``"checks"`` block.

    Parameters
    ----------
    checks : object
        Value of the recipe's ``"checks"`` key.
    source : str
        Human-readable recipe source used in errors.
    search_fields : Mapping[str, Any]
        The recipe's search fields.

    Raises
    ------
    ValueError
        If the block is not an object, has unknown keys, wrong value types, or
        names unknown fields.
    """
    if not isinstance(checks, dict):
        raise ValueError(f'Recipe in {source} "checks" must be a JSON object.')
    unknown = sorted(set(checks) - set(_CHECK_TYPES))
    if unknown:
        raise ValueError(f'Recipe in {source} "checks" has unknown key(s): {", ".join(unknown)}.')
    for key, value in checks.items():
        expected = _CHECK_TYPES[key]
        if not isinstance(value, expected) or (expected != bool and isinstance(value, bool)):
            raise ValueError(f'Recipe in {source} "checks" key "{key}" has the wrong type.')
    for key in ('fields', 'unit', 'exclude terms', 'exclude terms keys'):
        values = checks.get(key, [])
        if any(not isinstance(item, str) or not item.strip() for item in ([values] if isinstance(values, str) else values)):
            raise ValueError(f'Recipe in {source} "checks" key "{key}" must be a list of strings.')
    unknown_fields = [field for field in checks.get('fields', []) if field not in search_fields]
    if unknown_fields:
        raise ValueError(f'Recipe in {source} "checks" names unknown field(s): {", ".join(unknown_fields)}.')
    for term, phrases in checks.get('unsupported terms', {}).items():
        if not term.strip() or not isinstance(phrases, list) or any(not isinstance(p, str) for p in phrases):
            raise ValueError(
                f'Recipe in {source} "checks" "unsupported terms" must map terms to lists of phrases.'
            )


def _fold(text: object) -> str:
    """Normalize text for term matching: case, compatibility forms and dashes."""
    folded = unicodedata.normalize('NFKC', str(text)).casefold()
    return folded.translate(_DASHES)


def _mentions(text: str, term: str) -> bool:
    """Return whether folded text contains a term as whole words."""
    return re.search(rf'(?<!\w){re.escape(_fold(term))}(?!\w)', text) is not None


def _remove_term(text: str, term: str) -> str:
    """Remove a term and the separator before it, e.g. ``"HSE06+SOC"`` -> ``"HSE06"``."""
    pattern = rf'\s*(?:[+;,/&]|\bwith\b|\band\b|\bincluding\b)?\s*(?<!\w){re.escape(term)}(?!\w)'
    cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return cleaned.strip(' ;,+/&') or 'None'


class _Audit:
    """Collect dropped and edited items for one record."""

    def __init__(self, paper_id: str, record: str) -> None:
        """Start an empty audit for one paper's record."""
        self.paper_id = paper_id
        self.record = record
        self.rows: list[dict[str, str]] = []

    def add(self, field: str, item: object, check: str, action: str, detail: str) -> None:
        """Record one dropped or edited item."""
        self.rows.append({
            'Paper id': self.paper_id,
            'Record': self.record,
            'Field': field,
            'Item': json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item,
            'Check': check,
            'Action': action,
            'Detail': detail,
        })


class _Checker:
    """Apply one recipe's checks to records from one source text."""

    def __init__(self, checks: Mapping[str, Any], source_text: str) -> None:
        """Prepare the source's numbers, folded text and unsupported terms."""
        self.checks = checks
        self.value_key = checks.get('value key', 'value')
        self.tolerance = float(checks.get('tolerance', DEFAULT_SOURCE_TOLERANCE))
        self.source = _fold(source_text)
        self.source_numbers = numbers(source_text)
        units = checks.get('unit') or []
        self.unit_names = [units] if isinstance(units, str) else list(units)
        self.units = {_fold(unit) for unit in self.unit_names}
        self.unsupported = [
            term for term, phrases in checks.get('unsupported terms', {}).items()
            if not any(_mentions(self.source, phrase) for phrase in [term, *phrases])
        ]

    def _value(self, item: object) -> object:
        """Return the value part of an item."""
        return item.get(self.value_key) if isinstance(item, Mapping) else item

    def _drop_reason(self, item: object) -> tuple[str, str] | None:
        """Return the check that rejects an item, with a detail, or ``None``."""
        value = self._value(item)
        if is_missing(value):
            return ('drop empty', 'no value') if self.checks.get('drop empty') else None
        keys = self.checks.get('exclude terms keys')
        if isinstance(item, Mapping):
            parts = [part for key, part in item.items() if keys is None or key in keys]
        else:
            parts = [item]
        text = ' '.join(_fold(part) for part in parts if not is_missing(part))
        for term in self.checks.get('exclude terms', []):
            if _mentions(text, term):
                return 'exclude terms', f'mentions "{term}"'
        if self.units:
            units = [word for word in re.findall(r'\d\s*([^\W\d_]+)', _fold(value)) if word not in _RANGE_WORDS]
            if units and units[0] not in self.units:
                return 'unit', f'not in {" or ".join(self.unit_names)}'
        if self.checks.get('values in source') and numbers(value):
            if not printed_in(value, self.source_numbers, self.tolerance):
                return 'values in source', 'number not printed in the source text'
        return None

    def _edit(self, item: object, field: str, audit: _Audit) -> object:
        """Remove unsupported terms from an item's descriptive parts."""
        if not self.unsupported or not isinstance(item, Mapping):
            return item
        edited = dict(item)
        for key, part in item.items():
            if key == self.value_key or not isinstance(part, str):
                continue
            for term in self.unsupported:
                if _mentions(_fold(edited[key]), term):
                    edited[key] = _remove_term(edited[key], term)
        if edited != item:
            audit.add(field, item, 'unsupported terms', 'edited', json.dumps(edited, ensure_ascii=False))
        return edited

    def check_field(self, field: str, value: Any, audit: _Audit) -> Any:
        """Return a field value with rejected items removed and others edited."""
        items = value if isinstance(value, list) else [value]
        kept = []
        for item in items:
            reason = self._drop_reason(item)
            if reason:
                audit.add(field, item, reason[0], 'dropped', reason[1])
                continue
            kept.append(self._edit(item, field, audit))
        if isinstance(value, list):
            return kept
        return kept[0] if kept else 'None'


def verify_records(
    records: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any],
    source_text: str,
    paper_id: str = '',
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Apply a recipe's checks to one paper's records.

    Parameters
    ----------
    records : Sequence[Mapping[str, Any]]
        Records extracted from one paper.
    recipe : Mapping[str, Any]
        Loaded recipe. Without a ``"checks"`` block, records pass unchanged.
    source_text : str
        Full text the records were extracted from, including tables.
    paper_id : str, optional
        Paper identifier written to the audit rows.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, str]]]
        Kept records (copies) and audit rows with columns ``DROPPED_COLUMNS``.
    """
    checks = recipe.get('checks') or {}
    kept_records = [copy.deepcopy(dict(record)) for record in records]
    if not checks:
        return kept_records, []
    identities = identity_fields(recipe)
    fields = checks.get('fields') or [field for field in recipe['search fields'] if field not in identities]
    checker = _Checker(checks, source_text)
    kept: list[dict[str, Any]] = []
    audit_rows: list[dict[str, str]] = []
    for record in kept_records:
        name = ' | '.join(str(record.get(field, '')) for field in identities)
        audit = _Audit(paper_id or str(record.get('Paper id', '')), name)
        for field in fields:
            if field in record and not is_missing(record[field]):
                record[field] = checker.check_field(field, record[field], audit)
        audit_rows.extend(audit.rows)
        if checks.get('drop empty') and all(is_missing(record.get(field)) for field in fields):
            audit_rows.append({
                'Paper id': audit.paper_id,
                'Record': name,
                'Field': '',
                'Item': '',
                'Check': 'drop empty',
                'Action': 'dropped',
                'Detail': 'record has no reported fields',
            })
            continue
        kept.append(record)
    return kept, audit_rows
