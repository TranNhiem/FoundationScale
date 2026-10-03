"""Deterministic proposer: rank ideas from the ideas catalog (read-only apart from the catalog read)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from foundationskills.skills.auto_research.campaign import AXIS_PATHS, axis_value_fits
from foundationskills.skills.auto_research.ledger import canonical, sha256_hex

GAIN_WEIGHTS = {"s": 1, "m": 2, "l": 3}
DROP_REASONS = frozenset({"out_of_axes", "duplicate_delta", "not_in_axes", "missing_current"})
BASELINE_CARD = "baseline"
_MULT = re.compile(r"^\s*x\s*([0-9]*\.?[0-9]+)\s*$")


def load_catalog() -> list[dict[str, Any]]:
    """The ideas catalog (``ideas_catalog.yaml`` next to this module)."""
    raw = yaml.safe_load(Path(__file__).with_name("ideas_catalog.yaml").read_text(encoding="utf-8"))
    return [dict(entry) for entry in (raw.get("ideas") or [])]


def resolve_delta(
    delta: dict[str, Any], current: dict[str, Any], spec: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Resolve "x0.5"/"x2" deltas against ``current``; return (delta, drop_reason)."""
    axes = {str(a.get("key") or ""): dict(a) for a in (spec.get("axes") or [])}
    resolved: dict[str, Any] = {}
    for key, value in (delta or {}).items():
        if key not in AXIS_PATHS or key not in axes:
            return {}, "not_in_axes"
        match = _MULT.match(value) if isinstance(value, str) else None
        if match:
            if key not in current:
                return {}, "missing_current"
            new = float(current[key]) * float(match.group(1))
            resolved[key] = int(round(new)) if AXIS_PATHS[key] == "int" else new
        else:
            resolved[key] = value
    for key, value in resolved.items():
        if not axis_value_fits(axes[key], value):
            return {}, "out_of_axes"
    return resolved, None


def _symptom_hits(phrases: list[str], symptoms: list[str]) -> int:
    hits = 0
    for phrase in phrases:
        needle = str(phrase).strip().lower()
        for symptom in symptoms:
            text = str(symptom).strip().lower()
            if text and needle and (needle in text or text in needle):
                hits += 1
                break
    return hits


def evaluate(
    spec: dict[str, Any], ledger_results: list[dict[str, Any]], current: dict[str, Any], symptoms: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every catalog idea; return (live cards, dropped cards) ordered score desc, complexity asc, id asc."""
    results = [dict(r) for r in (ledger_results or [])]
    baseline_repeats = int((spec.get("seeds") or {}).get("baseline_repeats", 3))
    measured = [
        r for r in results if r.get("role") == "baseline" and r.get("status") == "ok" and not r.get("limited")
    ]
    if len(measured) < baseline_repeats:
        card = {
            "idea": BASELINE_CARD,
            "delta": {},
            "score": 0.0,
            "reason": f"run the calibrated baseline first ({len(measured)}/{baseline_repeats} ok repeats)",
        }
        return [card], []

    seen = {
        sha256_hex(canonical(dict(r["delta"])))
        for r in results
        if isinstance(r.get("delta"), dict) and r["delta"]
    }
    live: list[tuple[dict[str, Any], int]] = []
    dropped: list[tuple[dict[str, Any], int]] = []
    for idea in load_catalog():
        complexity = int(idea.get("complexity", 1))
        delta, drop = resolve_delta(dict(idea.get("delta") or {}), current, spec)
        if drop is None and sha256_hex(canonical(delta)) in seen:
            drop = "duplicate_delta"
        hits = _symptom_hits(list(idea.get("applies_when") or []), list(symptoms or []))
        gain = GAIN_WEIGHTS.get(str(idea.get("prior_gain")), 1)
        score = round(gain * (1 + hits) - 0.1 * complexity, 4)
        card = {
            "idea": str(idea.get("id")),
            "delta": delta,
            "score": score,
            "reason": drop if drop else f"prior={idea.get('prior_gain')} matched {hits} symptom(s)",
        }
        (dropped if drop else live).append((card, complexity))
    order = lambda row: (-row[0]["score"], row[1], row[0]["idea"])
    return [card for card, _ in sorted(live, key=order)], [card for card, _ in sorted(dropped, key=order)]


def propose(
    spec: dict[str, Any],
    ledger_results: list[dict[str, Any]],
    current: dict[str, Any],
    symptoms: list[str],
    k: int = 3,
) -> list[dict[str, Any]]:
    """Top-``k`` live idea cards; pruned ideas are dropped (see evaluate() for the drop reasons)."""
    live, _dropped = evaluate(spec, ledger_results, current, symptoms)
    return live[: max(0, int(k))]
