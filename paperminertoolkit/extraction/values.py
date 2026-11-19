"""Compare extracted values by their printed numbers and normalized text."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable

NUMBER = re.compile(r'\d+(?:\.\d+)?')
MISSING_VALUES = frozenset({'', 'none', 'nan', 'null', 'n/a', 'na', '[]', '{}'})
DEFAULT_TOLERANCE = 0.02


def is_missing(value: object) -> bool:
    """Return whether a cell or item part holds no reported value.

    Parameters
    ----------
    value : object
        Extracted value.

    Returns
    -------
    bool
        ``True`` for ``None``, NaN, empty containers and missing-value tokens.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return not value
    return str(value).strip().casefold() in MISSING_VALUES


def numbers(value: object) -> list[float]:
    """Return the unsigned decimal numbers printed in a value.

    Parameters
    ----------
    value : object
        Text such as ``"1.2-1.4 eV"``.

    Returns
    -------
    list[float]
        Numbers in order of appearance, e.g. ``[1.2, 1.4]``.
    """
    text = unicodedata.normalize('NFKC', str(value)).replace('·', '.')
    return [float(number) for number in NUMBER.findall(text)]


def close(first: float, second: float, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return whether two numbers agree within a relative tolerance.

    Parameters
    ----------
    first, second : float
        Numbers to compare.
    tolerance : float, default=0.02
        Allowed difference relative to the larger magnitude.

    Returns
    -------
    bool
        ``True`` when the numbers agree.
    """
    return abs(first - second) <= tolerance * max(abs(first), abs(second), 1e-9)


def printed_in(value: object, source_numbers: Iterable[float], tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return whether every number of a value appears among a source's numbers.

    Parameters
    ----------
    value : object
        Extracted value such as ``"2.04 eV"``.
    source_numbers : Iterable[float]
        Numbers printed in the source text.
    tolerance : float, default=0.02
        Relative tolerance for each match.

    Returns
    -------
    bool
        ``True`` when the value has numbers and each one is printed.
    """
    wanted = numbers(value)
    printed = list(source_numbers)
    return bool(wanted) and all(any(close(number, other, tolerance) for other in printed) for number in wanted)


def text_key(value: object) -> str:
    """Normalize text for equality: case, accents, punctuation and token order.

    Parameters
    ----------
    value : object
        Text to normalize.

    Returns
    -------
    str
        Sorted lower-case word tokens joined by spaces.
    """
    text = unicodedata.normalize('NFKC', str(value)).casefold()
    return ' '.join(sorted(re.findall(r'\w+(?:\.\w+)*', text)))


def same_value(first: object, second: object, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return whether two values state the same thing.

    Values with numbers agree when their numbers match pairwise within the
    tolerance and their remaining words (units, qualifiers) are equal.
    Other values are compared with :func:`text_key`.

    Parameters
    ----------
    first, second : object
        Values to compare.
    tolerance : float, default=0.02
        Relative tolerance for numbers.

    Returns
    -------
    bool
        ``True`` when the values agree.
    """
    first_numbers, second_numbers = numbers(first), numbers(second)
    if first_numbers or second_numbers:
        if len(first_numbers) != len(second_numbers):
            return False
        if not all(close(a, b, tolerance) for a, b in zip(first_numbers, second_numbers, strict=True)):
            return False
        return text_key(NUMBER.sub(' ', str(first))) == text_key(NUMBER.sub(' ', str(second)))
    return text_key(first) == text_key(second)
