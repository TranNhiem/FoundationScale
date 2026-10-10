"""Evidence pointers: ``<artifact path>#<json field>`` resolved to a measured value on disk.

The pointer form is exact: one ``#``, a path to a JSON file plus a field. The
field walks nested object keys and array indexes separated by ``.``
(``eval_report.json#benchmarks.0.score``). Anything that cannot be walked to a
non-null value raises :class:`EvidenceError` — an unresolvable pointer demotes
its claim, because absence of evidence is never PASS and this package never
invents a measurement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """An evidence pointer is malformed or names no measured value on disk."""


def walk_field(node: Any, field: str) -> Any:
    """Walk dotted ``field`` (object keys, array indexes) out of the loaded JSON document."""
    current = node
    for segment in field.split("."):
        if isinstance(current, dict):
            if segment not in current:
                raise EvidenceError(f"evidence field {field!r} does not resolve at key {segment!r}")
            current = current[segment]
        elif isinstance(current, list):
            if not segment.isdigit() or int(segment) >= len(current):
                raise EvidenceError(f"evidence field {field!r} does not resolve at index {segment!r}")
            current = current[int(segment)]
        else:
            raise EvidenceError(f"evidence field {field!r} does not resolve below {type(current).__name__}")
    return current


def resolve_evidence(pointer: str) -> Any:
    """Return the measured value ``<path>#<field>`` names, or raise :class:`EvidenceError`."""
    if not isinstance(pointer, str) or pointer.count("#") != 1:
        raise EvidenceError(f"malformed evidence pointer {pointer!r}: expected <path>#<field>")
    raw_path, field = pointer.split("#")
    if not raw_path.strip() or not field.strip():
        raise EvidenceError(f"malformed evidence pointer {pointer!r}: expected <path>#<field>")
    try:
        with Path(raw_path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as exc:
        raise EvidenceError(f"evidence file {raw_path!r} is not on disk: {exc}") from exc
    except ValueError as exc:
        raise EvidenceError(f"evidence file {raw_path!r} is not JSON: {exc}") from exc
    value = walk_field(document, field)
    if value is None:
        raise EvidenceError(f"evidence pointer {pointer!r} names no measured value (null on disk)")
    return value
