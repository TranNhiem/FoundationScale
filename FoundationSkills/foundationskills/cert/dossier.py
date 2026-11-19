"""Certification dossier for a harness run-set.

Records which must-fire fixtures actually fired on valid (non-infra) runs and how far the
score swings across harness configurations (card 2608.21382 "no neutral harness"): a
dossier never reaches PASS without a measured fragility band across harness configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from foundationskills.core import harness

SCHEMA_ID = "fs.cert_dossier.v1"


@dataclass(frozen=True)
class SuiteResult:
    """One must-fire fixture outcome: expected vs fired rules, verdict, run validity."""

    fixture_id: str
    rule_ids_expected: tuple[str, ...]
    rule_ids_fired: tuple[str, ...]
    verdict: str  # PASS/RED/UNMEASURED/REFUSED
    validity: str = "valid"  # "valid" | "infra"


def valid_results(results: Iterable[SuiteResult]) -> list[SuiteResult]:
    """Results on genuine runs; infra failures are dropped before scoring. Raises on unknown validity."""
    kept: list[SuiteResult] = []
    for result in results:
        if result.validity == "infra":
            continue
        if result.validity != "valid":
            raise ValueError(f"{result.fixture_id}: unknown validity {result.validity!r}")
        kept.append(result)
    return kept


def beacon_ok(results: Iterable[SuiteResult]) -> bool:
    """True when at least one valid run exists and each fired all of its expected rules."""
    kept = valid_results(results)
    return bool(kept) and all(
        set(r.rule_ids_expected) <= set(r.rule_ids_fired) for r in kept
    )


def robust_band(grid: Mapping[str, float] | None) -> dict[str, Any] | None:
    """Score spread across harness configs; None when fragility was not measured at all."""
    values = list((grid or {}).values())
    if len(values) < 2:
        return None
    low, high = min(values), max(values)
    return {"configs": len(values), "min": low, "max": high, "spread": high - low}


def _beacon_misses(kept: Iterable[SuiteResult]) -> list[str]:
    """Reasons naming the valid fixtures whose expected must-fire rules never fired."""
    reasons = []
    for result in kept:
        missing = [r for r in result.rule_ids_expected if r not in result.rule_ids_fired]
        if missing:
            reasons.append(f"fixture {result.fixture_id}: expected rules never fired: {', '.join(missing)}")
    return reasons


def build_dossier(
    manifest_payload: dict,
    results: Iterable[SuiteResult],
    grid: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble the certification dossier with verdict and reasons; never PASS unmeasured."""
    all_results = list(results)
    kept = valid_results(all_results)
    n_infra = sum(1 for r in all_results if r.validity == "infra")
    problems = harness.validate_manifest(manifest_payload)
    ok = beacon_ok(kept)
    band = robust_band(grid or {})
    passed = sum(1 for r in kept if r.verdict == "PASS")

    reasons: list[str] = []
    if problems:
        verdict = "REFUSED"
        reasons = [f"manifest: {problem}" for problem in problems]
    elif not kept:
        verdict = "UNMEASURED"
        reasons = ["no valid results"]
    elif not ok:
        verdict = "RED"
        reasons = _beacon_misses(kept)
    elif band is None:
        verdict = "UNMEASURED"
        reasons = ["fragility not measured: fewer than 2 harness configs"]
    else:
        verdict = "PASS"

    return {
        "schema": SCHEMA_ID,
        "manifest_hash": harness.manifest_hash(manifest_payload) if isinstance(manifest_payload, dict) else None,
        "manifest_problems": problems,
        "n_results": len(all_results),
        "n_valid": len(kept),
        "n_infra": n_infra,
        "beacon_ok": ok,
        "pass_rate": (passed / len(kept)) if kept else None,
        "robust_band": band,
        "verdict": verdict,
        "reasons": reasons,
    }
