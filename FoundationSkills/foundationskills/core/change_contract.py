from __future__ import annotations
"""Governed-harness mutation gate (atlas P15 §11): any diff that touches protected
code must declare what it changes and which invariants it preserves. Parsing is
tolerant and check() reports every problem as a string; it never raises."""
import re
from dataclasses import dataclass
from typing import Iterable

REQUIRED_INVARIANTS = ("0/5/95/96 exits", "typed-back hash", "real-GPU budget")
PROTECTED_PREFIXES = (
    "FoundationSkills/foundationskills/skills/",
    "FoundationSkills/foundationskills/core/",
    "FoundationSkills/foundationskills/interfaces/fs/",
    "FoundationSkills/foundationskills/vetting/",
)
FIELDS = ("Component", "Failure-Mode", "Predicted-Gain", "Preserves", "Falsifier", "Rollback")

HEADER = "Change-Contract:"
_HEX_COMMIT = re.compile(r"[0-9a-fA-F]{7,40}")


@dataclass(frozen=True)
class ChangeContract:
    component: str
    failure_mode: str
    predicted_gain: str
    preserves: tuple[str, ...]
    falsifier: str
    rollback: str


def needs_contract(paths: Iterable[str]) -> bool:
    """True when any touched file is protected; tests and .md files are exempt."""
    for path in paths:
        p = str(path).replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        if "/tests/" in p or p.endswith(".md"):
            continue
        if p.startswith(PROTECTED_PREFIXES):
            return True
    return False


def _block(message: str) -> dict[str, str] | None:
    """Key -> value map of the Change-Contract block, None when the header is absent."""
    lines = str(message).splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == HEADER), None)
    if start is None:
        return None
    values: dict[str, str] = {}
    for line in lines[start + 1:]:
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        key = key.strip()
        if sep and key in FIELDS and key not in values:
            values[key] = value.strip()
    return values


def parse_contract(message: str) -> ChangeContract | None:
    """Parse the contract block; None when it is absent or a field is missing/empty."""
    values = _block(message)
    if values is None or any(not values.get(field, "").strip() for field in FIELDS):
        return None
    preserves = tuple(p.strip() for p in values["Preserves"].split(";") if p.strip())
    return ChangeContract(
        component=values["Component"],
        failure_mode=values["Failure-Mode"],
        predicted_gain=values["Predicted-Gain"],
        preserves=preserves,
        falsifier=values["Falsifier"],
        rollback=values["Rollback"],
    )


def check(message: str, paths: Iterable[str]) -> list[str]:
    """Problems with the diff's change contract; [] when no contract is required."""
    problems: list[str] = []
    try:
        if not needs_contract(paths):
            return problems
        values = _block(message)
        if values is None:
            problems.append("missing Change-Contract block")
            return problems
        for field in FIELDS:
            if not values.get(field, "").strip():
                problems.append("missing Change-Contract field: " + field)
        preserves = values.get("Preserves", "").strip()
        if preserves:
            for invariant in REQUIRED_INVARIANTS:
                if invariant.lower() not in preserves.lower():
                    problems.append("Preserves must list: " + invariant)
        rollback = values.get("Rollback", "").strip()
        if rollback and _HEX_COMMIT.fullmatch(rollback) is None:
            problems.append("Rollback must be a 7-40 char hex commit")
    except Exception:  # defensive: this gate must never crash the harness
        return problems
    return problems
