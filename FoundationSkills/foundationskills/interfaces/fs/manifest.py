"""Generic reader for FoundationScale's self-declared capability manifests (A5.2).

A FoundationScale module declares what it can do; the probe never trusts that declaration. Each
entry names a zero-arg ``verify`` callable (the measurement) and how to ``compare`` it with what was
``declared``. One row per entry, status:

  verified      the measurement ran and the comparison holds
  contradicted  it ran and the comparison fails -- the MEASURED value is what the artifact reports
  unmeasured    nothing could be measured (no verify, it raised, bad entry, manifest import failed)

Discovery is the union of the entry-point group ``foundationscale.capabilities`` (installed FS) and
``capability_manifest.py`` files under ``foundationscale.__path__`` (FS on PYTHONPATH has no
distribution metadata, so entry points alone would find nothing there). Nothing raises: a broken
manifest is an unmeasured row with its reason. Spec: artifacts/research/a52_capability_manifest.md.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

GROUP = "foundationscale.capabilities"
FILENAME = "capability_manifest.py"
COMPARES = ("equal", "subset", "superset")
VERIFIED, CONTRADICTED, UNMEASURED = "verified", "contradicted", "unmeasured"


def _row(
    key: str, module: str, declared: Any, measured: Any, status: str, reason: str
) -> dict[str, Any]:
    return {
        "key": key,
        "module": module,
        "declared": _jsonable(declared),
        "measured": _jsonable(measured),
        "status": status,
        "reason": reason,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(v) for v in value]
        return sorted(items, key=str) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _sources() -> tuple[dict[str, Callable[[], Any]], list[dict[str, Any]]]:
    """module name -> loader of its MANIFEST, plus rows for discovery failures."""
    try:
        import foundationscale  # type: ignore  # noqa: PLC0415
    except Exception as exc:
        reason = f"import foundationscale failed: {type(exc).__name__}: {exc}"
        return {}, [_row("<discovery>", "foundationscale", None, None, UNMEASURED, reason)]
    sources: dict[str, Callable[[], Any]] = {}
    for base in getattr(foundationscale, "__path__", []):
        root = Path(base)
        for path in sorted(root.rglob(FILENAME)):
            if "__pycache__" in path.parts:
                continue
            name = ".".join(("foundationscale", *path.relative_to(root).with_suffix("").parts))
            sources.setdefault(name, lambda n=name: importlib.import_module(n).MANIFEST)
    try:
        points = list(metadata.entry_points(group=GROUP))
    except Exception:
        points = []
    for point in points:
        sources.setdefault(point.value.split(":")[0], point.load)
    return sources, []


def _resolve(target: Any) -> Callable[[], Any]:
    if callable(target):
        return target
    if not isinstance(target, str) or ":" not in target:
        raise ValueError(f"verify must be 'module:callable', got {target!r}")
    module_name, _, attr = target.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"verify target {target!r} is not callable")
    return obj


def _holds(compare: str, declared: Any, measured: Any) -> bool:
    if compare == "equal":
        if isinstance(declared, (list, tuple, set)) and isinstance(measured, (list, tuple, set)):
            return set(map(str, declared)) == set(map(str, measured))
        return bool(declared == measured)
    got, want = set(map(str, measured)), set(map(str, declared))
    return bool(got) and got <= want if compare == "subset" else got >= want


def _entry_row(module: str, entry: Any, seen: set[str]) -> dict[str, Any]:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("key"), str) or not entry["key"]:
        return _row(
            "<malformed>",
            module,
            None,
            None,
            UNMEASURED,
            f"entry is not a mapping with a key: {entry!r}"[:300],
        )
    key, declared, compare = entry["key"], entry.get("declared"), entry.get("compare", "equal")
    if key in seen:
        return _row(
            key,
            module,
            declared,
            None,
            UNMEASURED,
            "duplicate key: an earlier manifest already declared it",
        )
    seen.add(key)
    if compare not in COMPARES:
        return _row(
            key,
            module,
            declared,
            None,
            UNMEASURED,
            f"compare must be one of {COMPARES}, got {compare!r}",
        )
    if entry.get("verify") is None:
        return _row(
            key,
            module,
            declared,
            None,
            UNMEASURED,
            "no verify callable: a declaration alone is not a measurement",
        )
    try:
        measured = _resolve(entry["verify"])()
    except Exception as exc:
        return _row(
            key,
            module,
            declared,
            None,
            UNMEASURED,
            f"verify failed: {type(exc).__name__}: {exc}"[:300],
        )
    try:
        ok = _holds(compare, declared, measured)
    except TypeError as exc:
        return _row(key, module, declared, measured, UNMEASURED, f"cannot compare: {exc}")
    if ok:
        return _row(key, module, declared, measured, VERIFIED, "")
    return _row(
        key,
        module,
        declared,
        measured,
        CONTRADICTED,
        f"declared {_jsonable(declared)!r} but measured {_jsonable(measured)!r} ({compare})",
    )


def read() -> list[dict[str, Any]]:
    """Every manifest entry FS declares, each verified by its own measurement."""
    sources, rows = _sources()
    seen: set[str] = set()
    for module, load in sorted(sources.items()):
        try:
            loaded = load()
            entries = loaded() if callable(loaded) else loaded
            if not isinstance(entries, (list, tuple)):
                raise TypeError(f"MANIFEST must be a list, got {type(entries).__name__}")
        except Exception as exc:
            rows.append(
                _row(
                    "<manifest>",
                    module,
                    None,
                    None,
                    UNMEASURED,
                    f"manifest failed to load: {type(exc).__name__}: {exc}"[:300],
                )
            )
            continue
        rows.extend(_entry_row(module, entry, seen) for entry in entries)
    return rows


def exit_code(rows: list[dict[str, Any]]) -> int:
    """Precedence: any unmeasured (or no rows) -> 95, else any contradicted -> 5, else 0."""
    if not rows or any(r["status"] == UNMEASURED for r in rows):
        return 95
    return 5 if any(r["status"] == CONTRADICTED for r in rows) else 0
