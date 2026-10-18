"""Case conversion and detection utilities for identifiers.

Supports four common identifier naming conventions:

* ``snake_case``        -- e.g. ``user_name``
* ``camelCase``         -- e.g. ``userName``
* ``PascalCase``        -- e.g. ``UserName``
* ``kebab-case``        -- e.g. ``user-name``

The module provides a single entry point :func:`convert` and convenience
wrappers, plus :func:`detect` which inspects a string and guesses which
convention it follows.
"""

from __future__ import annotations

import re
from typing import Literal

CaseStyle = Literal["snake", "camel", "pascal", "kebab"]

__all__ = [
    "CaseStyle",
    "to_snake",
    "to_camel",
    "to_pascal",
    "to_kebab",
    "convert",
    "detect",
    "split_words",
]

_WORD_DENORMAL_RE = re.compile(r"([A-Z]+)([A-Z][a-z])|([a-z0-9])([A-Z])|[-_]+")


def split_words(identifier: str) -> list[str]:
    """Split *identifier* into its constituent lower-case words.

    Raises ``ValueError`` when *identifier* contains no alphanumeric
    characters at all.
    """
    normalized = _WORD_DENORMAL_RE.sub(
        lambda m: (m.group(1) or m.group(3) or "") + " " + (m.group(2) or m.group(4) or ""),
        identifier,
    )
    parts = [p.lower() for p in normalized.split() if p]
    if not parts:
        raise ValueError(f"cannot split words from {identifier!r}")
    return parts


def to_snake(identifier: str) -> str:
    """Convert *identifier* to ``snake_case``."""
    return "_".join(split_words(identifier))


def to_kebab(identifier: str) -> str:
    """Convert *identifier* to ``kebab-case``."""
    return "-".join(split_words(identifier))


def to_pascal(identifier: str) -> str:
    """Convert *identifier* to ``PascalCase``."""
    return "".join(w.capitalize() for w in split_words(identifier))


def to_camel(identifier: str) -> str:
    """Convert *identifier* to ``camelCase``."""
    words = split_words(identifier)
    return words[0] + "".join(w.capitalize() for w in words[1:])


def convert(identifier: str, target: CaseStyle, source: CaseStyle | None = None) -> str:
    """Convert *identifier* from *source* style to *target* style.

    When *source* is ``None`` the style is inferred with :func:`detect`.
    Raises ``ValueError`` for unknown style names or empty identifiers.
    """
    valid = {"snake", "camel", "pascal", "kebab"}
    if target not in valid:
        raise ValueError(f"unknown case style {target!r}")
    if source is not None and source not in valid:
        raise ValueError(f"unknown case style {source!r}")
    if not identifier or not identifier.strip():
        raise ValueError("identifier must be non-empty")
    dispatch = {
        "snake": to_snake,
        "kebab": to_kebab,
        "pascal": to_pascal,
        "camel": to_camel,
    }
    return dispatch[target](identifier)


def detect(identifier: str) -> CaseStyle:
    """Guess which naming convention *identifier* follows.

    Rules, checked in order:

    1. contains ``-`` -> ``kebab``
    2. contains ``_`` -> ``snake``
    3. starts with an uppercase letter -> ``pascal``
    4. otherwise -> ``camel``

    ``ValueError`` is raised for strings without any letter.
    """
    if not identifier:
        raise ValueError("identifier must be non-empty")
    if "-" in identifier:
        return "kebab"
    if "_" in identifier:
        return "snake"
    first = identifier[0]
    if not first.isalpha():
        raise ValueError(f"identifier {identifier!r} has no leading letter")
    if first.isupper():
        return "pascal"
    return "camel"
