"""Small, pure-Python library for Roman numeral conversion and validation.

Public API:
    to_roman(n)      -- convert integer 1..3999 to a Roman numeral string
    to_arabic(s)     -- convert a valid Roman numeral string to an integer
    is_valid(s)      -- bool: is `s` a valid Roman numeral? (with exact canonical form)
    is_ascii(s)      -- bool: does `s` use only ASCII characters?
    roman_range()    -- generator of all (number, numeral) pairs 1..3999
    RomanError       -- exception base for all errors raised here
    ValidationError  -- raised for invalid numerals
    RangeError       -- raised for integers outside 1..3999

Canonical Roman numerals (form used here) use subtractive notation for
4, 9, 40, 90, 400, 900 and are strictly non-increasing in value elsewhere.
"""

from __future__ import annotations

__all__ = [
    "to_roman",
    "to_arabic",
    "is_valid",
    "is_ascii",
    "roman_range",
    "RomanError",
    "ValidationError",
    "RangeError",
]

MIN_VALUE = 1
MAX_VALUE = 3999


class RomanError(Exception):
    """Base class for all errors raised by the roman module."""


class ValidationError(RomanError, ValueError):
    """Raised when a string is not a valid Roman numeral."""


class RangeError(RomanError, ValueError):
    """Raised when an integer is outside the supported range 1..3999."""


# Value -> numeral pairs, ordered from largest to smallest value.
_ROMAN_PAIRS = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)

_ROMAN_CHAR_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def is_ascii(s: str) -> bool:
    """Return True if `s` contains only ASCII characters."""
    return len(s.encode("ascii", errors="ignore")) == len(s)


def to_roman(n: int) -> str:
    """Convert the integer `n` (1..3999) to its canonical Roman numeral string."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"expected int, got {type(n).__name__}")
    if n < MIN_VALUE or n > MAX_VALUE:
        raise RangeError(f"value {n} out of range {MIN_VALUE}..{MAX_VALUE}")
    parts: list[str] = []
    remaining = n
    for value, symbol in _ROMAN_PAIRS:
        count, remaining = divmod(remaining, value)
        parts.append(symbol * count)
    return "".join(parts)


def to_arabic(s: str) -> int:
    """Convert a canonical Roman numeral string to its integer value."""
    if not isinstance(s, str):
        raise TypeError(f"expected str, got {type(s).__name__}")
    if not s:
        raise ValidationError("empty string is not a Roman numeral")
    if not is_ascii(s):
        raise ValidationError("non-ASCII characters in Roman numeral")
    upper = s.upper()
    total = 0
    prev_value = 0
    for ch in reversed(upper):
        if ch not in _ROMAN_CHAR_VALUES:
            raise ValidationError(f"invalid Roman numeral character: {ch!r}")
        value = _ROMAN_CHAR_VALUES[ch]
        if value < prev_value:
            total -= value
        else:
            total += value
            prev_value = value
    if to_roman(total) != upper:
        raise ValidationError(f"{s!r} is not in canonical Roman numeral form")
    return total


def is_valid(s: str) -> bool:
    """Return True if `s` is a canonical Roman numeral in the range 1..3999."""
    if not isinstance(s, str) or not s:
        return False
    try:
        to_arabic(s)
    except RomanError:
        return False
    return True


def roman_range():
    """Yield (number, numeral) pairs for every value in 1..3999."""
    n = MIN_VALUE
    while n <= MAX_VALUE:
        yield n, to_roman(n)
        n += 1
