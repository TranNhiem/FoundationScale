"""Pure acceptance statistics for auto_research: noise floor, tau bands and verdicts (no I/O)."""
from __future__ import annotations

import math
import statistics
from typing import Any

SIGNS = {"max": 1.0, "min": -1.0}
ROBUST_N = 3
ROBUST_SCALE = 1.4826


def _point(result: dict[str, Any], metric: str) -> dict[str, float] | None:
    """The measured point for ``metric``; crashes and limited runs are never evidence."""
    if result.get("status") != "ok" or result.get("limited"):
        return None
    point = (result.get("metrics") or {}).get(metric)
    if not isinstance(point, dict) or point.get("value") is None:
        return None
    se = point.get("se")
    return {"value": float(point["value"]), "se": float(se) if se is not None else 0.0}


def values(results: list[dict[str, Any]], metric: str) -> list[float]:
    """Measured values for ``metric`` over the ok, non-limited results (crashes excluded)."""
    points = [_point(row, metric) for row in results]
    return [p["value"] for p in points if p is not None]


def noise_floor(baseline_results: list[dict[str, Any]], metric: str, baseline_repeats: int = 3) -> float | None:
    """Sample std of the baseline repeats (1.4826*MAD when n == 3); None when the floor is uncalibrated."""
    pts = values(baseline_results, metric)
    if len(pts) < max(2, int(baseline_repeats)):
        return None
    std = float(statistics.stdev(pts))
    if len(pts) == ROBUST_N:
        med = statistics.median(pts)
        robust = ROBUST_SCALE * statistics.median([abs(v - med) for v in pts])
        if robust > 0:
            return float(robust)
    return std


def paired(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], metric: str
) -> list[tuple[dict[str, float], dict[str, float]]]:
    """Seed-matched (reference, candidate) point pairs over measured results."""
    refs: dict[Any, dict[str, float]] = {}
    for row in baseline:
        point = _point(row, metric)
        if point is not None:
            refs[row.get("seed")] = point
    pairs: list[tuple[dict[str, float], dict[str, float]]] = []
    for row in candidate:
        point = _point(row, metric)
        if point is not None and row.get("seed") in refs:
            pairs.append((refs[row.get("seed")], point))
    return pairs


def _mean_sq(pairs: list[tuple[dict[str, float], dict[str, float]]], index: int) -> float:
    return statistics.fmean([point[index]["se"] ** 2 for point in pairs]) if pairs else 0.0


def decide(
    spec: dict[str, Any], baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    """Statistical verdict for one candidate against the baseline (seed-paired, tau-bounded, guardrailed)."""
    objective = dict(spec.get("objective") or {})
    confirm = dict(spec.get("confirm") or {})
    seeds = dict(spec.get("seeds") or {})
    fingerprint = str(dict(spec.get("eval_policy") or {}).get("fingerprint") or "")
    sign = SIGNS.get(str(objective.get("direction") or "max"), 1.0)
    k = float(confirm.get("k", 2.0))
    rel = float(confirm.get("noise_floor_rel", 0.005))
    guard_eps = float(confirm.get("guard_abs_epsilon", 0.01))
    guards = [str(g) for g in (confirm.get("guardrails") or [])]
    guard_dirs = {str(n): str(m) for n, m in dict(confirm.get("guardrail_directions") or {}).items()}
    confirm_repeats = int(seeds.get("confirm_repeats", 3))
    baseline_repeats = int(seeds.get("baseline_repeats", 3))

    reasons: list[str] = []
    unmeasured: list[str] = []
    for row in [*baseline, *candidate]:
        tag = f"{row.get('trial')}:{row.get('seed')}"
        if row.get("status") == "crash":
            reasons.append(f"crash_excluded:{tag}")  # crashes are counted, never evidence
            continue
        if row.get("limited"):
            unmeasured.append(f"limited_run:{tag}")
            continue
        if str(row.get("eval_policy_fingerprint") or "") != fingerprint:
            unmeasured.append(f"fingerprint_mismatch:{tag}")
        if metric not in (row.get("metrics") or {}):
            unmeasured.append(f"missing_metric:{tag}:{metric}")

    pairs = paired(baseline, candidate, metric)
    if len(pairs) < confirm_repeats:
        unmeasured.append(f"n_pairs={len(pairs)}<confirm_repeats={confirm_repeats}")
    floor = noise_floor(baseline, metric, baseline_repeats)
    if floor is None:
        unmeasured.append(
            f"uncalibrated_noise_floor:{len(values(baseline, metric))}/{baseline_repeats} ok baseline repeats"
        )

    deltas = [sign * (cand["value"] - ref["value"]) for ref, cand in pairs]
    mean_delta = statistics.fmean(deltas) if deltas else None
    mean_ref = statistics.fmean([ref["value"] for ref, _ in pairs]) if pairs else 0.0
    se_band = k * math.sqrt(_mean_sq(pairs, 0) + _mean_sq(pairs, 1))
    tau = max(floor or 0.0, rel * abs(mean_ref), se_band)

    breaches: list[str] = []
    for guard in guards:
        gpairs = paired(baseline, candidate, guard)
        if len(gpairs) < 2:
            unmeasured.append(f"guardrail_unmeasured:{guard}:{len(gpairs)} pairs")
            continue
        # The drop sign is the GUARD metric's own direction (higher-is-better "max" by default,
        # confirm.guardrail_directions overrides); it is never the objective's outcome sign.
        gsign = SIGNS.get(guard_dirs.get(guard, "max"), 1.0)
        drop = statistics.fmean([gsign * (ref["value"] - cand["value"]) for ref, cand in gpairs])
        band = max(guard_eps, k * math.sqrt(_mean_sq(gpairs, 0) + _mean_sq(gpairs, 1)))
        if drop > band:
            breaches.append(f"guardrail:{guard}")
    reasons.extend(breaches)

    simpler = any(bool(row.get("simpler")) for row in candidate)
    if unmeasured:  # unmeasured evidence dominates: there is nothing to declare a regression from
        reasons.extend(unmeasured)
        verdict = "unmeasured"
    elif breaches:
        verdict = "rejected_regress"
    elif mean_delta is not None and mean_delta > tau:
        verdict = "accepted_gain"
        reasons.append(f"mean_delta {mean_delta:.6g} > tau {tau:.6g}")
    elif mean_delta is not None and mean_delta < -tau:
        verdict = "rejected_regress"
        reasons.append(f"mean_delta {mean_delta:.6g} < -tau {-tau:.6g}")
    elif simpler:
        verdict = "accepted_flat"
        reasons.append("|mean_delta| <= tau with a simpler candidate")
    else:
        verdict = "no_gain"
        reasons.append("|mean_delta| <= tau and the candidate is not simpler")
    return {"verdict": verdict, "mean_delta": mean_delta, "tau": tau, "n_pairs": len(pairs), "reasons": reasons}
