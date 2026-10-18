"""Unit conversion library for length, mass and temperature.

Supports parsing strings like '5 km', '2.5 kg', '-3 degC', and converting
between compatible units with validation and optional rounding.
"""

from __future__ import annotations

_LENGTH: dict[str, float] = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0}
_MASS: dict[str, float] = {"mg": 1e-6, "g": 0.001, "kg": 1.0, "t": 1000.0}
_TEMPERATURE: set[str] = {"degC", "degF", "K"}

_FACTORS: dict[str, dict[str, float]] = {"length": _LENGTH, "mass": _MASS}
_COMPACT: dict[str, str] = {}
for _cat, _tbl in _FACTORS.items():
    for _u in _tbl:
        _COMPACT[_u] = _cat
for _u in _TEMPERATURE:
    _COMPACT[_u] = "temperature"


class UnitError(ValueError):
    """Raised for unknown units or incompatible conversions."""


def known_units() -> list[str]:
    """Return all supported unit symbols, sorted by category then symbol."""
    out: list[str] = []
    for cat in ("length", "mass"):
        out.extend(sorted(_FACTORS[cat]))
    out.extend(sorted(_TEMPERATURE))
    return out


def category(unit: str) -> str:
    """Return the category ('length', 'mass', 'temperature') of *unit*.

    Raises UnitError if the unit is unknown.
    """
    if not isinstance(unit, str):
        raise UnitError(f"unit must be a string, got {type(unit).__name__}")
    cat = _COMPACT.get(unit.strip())
    if cat is None:
        raise UnitError(f"unknown unit: {unit!r}")
    return cat


def to_base(value: float, unit: str) -> float:
    """Convert *value* in *unit* to the base SI unit of its category."""
    cat = category(unit)
    if cat == "temperature":
        if unit == "degC":
            return value + 273.15
        if unit == "degF":
            return (value + 459.67) * 5.0 / 9.0
        return value
    return value * _FACTORS[cat][unit]


def from_base(value: float, unit: str) -> float:
    """Convert *value* from the base SI unit to *unit*."""
    cat = category(unit)
    if cat == "temperature":
        if unit == "degC":
            return value - 273.15
        if unit == "degF":
            return value * 9.0 / 5.0 - 459.67
        return value
    return value / _FACTORS[cat][unit]


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert *value* from *from_unit* to *to_unit*.

    Raises UnitError if the units belong to different categories.
    """
    cat_from = category(from_unit)
    cat_to = category(to_unit)
    if cat_from != cat_to:
        raise UnitError(f"cannot convert {from_unit!r} ({cat_from}) to {to_unit!r} ({cat_to})")
    base = to_base(value, from_unit)
    return from_base(base, to_unit)


def parse(text: str) -> tuple[float, str]:
    """Parse a string like '5 km' or '-2.5 degC' into (value, unit).

    The value and unit must be separated by whitespace.
    Raises ValueError on malformed input.
    """
    if not isinstance(text, str):
        raise ValueError(f"expected a string, got {type(text).__name__}")
    parts = text.strip().split()
    if len(parts) != 2:
        raise ValueError(f"expected '<value> <unit>', got {text!r}")
    raw_value, raw_unit = parts
    try:
        val = float(raw_value)
    except ValueError:
        raise ValueError(f"invalid number: {raw_value!r}") from None
    if not raw_unit.isidentifier():
        raise ValueError(f"invalid unit: {raw_unit!r}")
    return val, raw_unit


def smart_convert(text: str, to_unit: str, digits: int | None = None) -> float:
    """Parse *text* as '<value> <unit>' and convert to *to_unit*.

    Round to *digits* decimal places if provided (0 or more).
    """
    if digits is not None and digits < 0:
        raise ValueError("digits must be non-negative")
    val, unit = parse(text)
    result = convert(val, unit, to_unit)
    if digits is not None:
        result = round(result, digits)
    return result
