"""eval_report.json: build the verdict, validate against the artifact schema, write atomically."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from foundationskills.core.artifacts import _atomic_write_json, _load_artifact_schema
from foundationskills.core.provenance import sha256_file
from foundationskills.core.schema import SchemaError, validate

# Per-benchmark statuses; "skipped" (configured but not executed/reported) is a failure.
FAILING = frozenset({"red", "skipped"})
UNMEASURED = frozenset({"unmeasured"})


def verdict(statuses: Iterable[str], limited: bool) -> str:
    """RED on any breach/skip; else UNMEASURED on any gap or a --limit run; else PASS. Empty is UNMEASURED."""
    seen = list(statuses)
    if any(s in FAILING for s in seen):
        return "RED"
    if not seen or limited or any(s in UNMEASURED for s in seen):
        return "UNMEASURED"
    return "PASS"


def validate_report(report: dict[str, Any]) -> list[str]:
    return validate(report, _load_artifact_schema("eval_report"))


def write_report(path: str | Path, report: dict[str, Any]) -> str:
    """Validate then atomically write ``report``; return its sha256."""
    errors = validate_report(report)
    if errors:
        raise SchemaError("eval_report invalid: " + "; ".join(errors))
    target = Path(path)
    _atomic_write_json(target, report)
    return sha256_file(target)
