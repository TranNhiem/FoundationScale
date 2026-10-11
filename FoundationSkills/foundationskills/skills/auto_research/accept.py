"""Pure acceptance statistics for auto_research: noise floor, tau bands and verdicts (no I/O)."""
from __future__ import annotations

import math
import statistics
from typing import Any

SIGNS = {"max": 1.0, "min": -1.0}
ROBUST_N = 3
ROBUST_SCALE = 1.4826
RL_GUARD = "rl_measured_fraction"


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
        if row.get("status") == "unmeasured":
            unmeasured.append(f"manifest_unmeasured:{tag}")  # counted, never evidence (M6, D2)
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

    breaches, guard_unmeasured = _guard_checks(
        baseline, candidate, guards, guard_dirs, k, guard_eps, rl_floor=spec.get("rl_min_measured_fraction")
    )
    unmeasured.extend(guard_unmeasured)
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


def _guard_checks(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    guards: list[str],
    guard_dirs: dict[str, str],
    k: float,
    guard_eps: float,
    rl_floor: Any = None,
) -> tuple[list[str], list[str]]:
    """Guardrail bands shared by :func:`decide` and :func:`decide_multi`: ``(breaches, unmeasured markers)``.

    ``rl_measured_fraction`` is judged against the declared ``rl_floor`` on the candidate's own rows (M6, D6
    revised): the reference is usually the untrained base model, which has no RL fraction to pair with.
    """
    breaches: list[str] = []
    unmeasured: list[str] = []
    for guard in guards:
        if guard == RL_GUARD and rl_floor is not None:
            measured = values(candidate, guard)
            if len(measured) < 2:
                unmeasured.append(f"guardrail_unmeasured:{guard}:{len(measured)} candidate rows")
            elif min(measured) < float(rl_floor):
                breaches.append(f"guardrail:{guard}:below_floor:{min(measured):.6g}<{float(rl_floor):.6g}")
            continue
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
    return breaches, unmeasured


def decide_multi(
    spec: dict[str, Any],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    """Statistical verdict for one candidate over every ``spec.objectives`` entry (M5a, A.5).

    Each objective is classified over the COMMON seed set ``J`` (seeds measured on both sides for
    *every* objective); a win here with a loss there is ``tradeoff`` — a decision value that is
    never claimable.  An objective that cannot be measured is ``None``-valued with class
    ``unmeasured`` — never ``0``.
    """
    objectives = [dict(o) for o in (spec.get("objectives") or [])]
    confirm = dict(spec.get("confirm") or {})
    seeds = dict(spec.get("seeds") or {})
    fingerprint = str(dict(spec.get("eval_policy") or {}).get("fingerprint") or "")
    k = float(confirm.get("k", 2.0))
    rel = float(confirm.get("noise_floor_rel", 0.005))
    guard_eps = float(confirm.get("guard_abs_epsilon", 0.01))
    guards = [str(g) for g in (confirm.get("guardrails") or [])]
    guard_dirs = {str(n): str(m) for n, m in dict(confirm.get("guardrail_directions") or {}).items()}
    confirm_repeats = int(seeds.get("confirm_repeats", 3))
    baseline_repeats = int(seeds.get("baseline_repeats", 3))
    metrics = [str(o.get("metric")) for o in objectives]

    reasons: list[str] = []
    unmeasured: list[str] = []
    for row in [*baseline, *candidate]:
        tag = f"{row.get('trial')}:{row.get('seed')}"
        if row.get("status") == "crash":
            reasons.append(f"crash_excluded:{tag}")  # crashes are counted, never evidence
            continue
        if row.get("status") == "unmeasured":
            unmeasured.append(f"manifest_unmeasured:{tag}")  # counted, never evidence (M6, D2)
            continue
        if row.get("limited"):
            unmeasured.append(f"limited_run:{tag}")
            continue
        if str(row.get("eval_policy_fingerprint") or "") != fingerprint:
            unmeasured.append(f"fingerprint_mismatch:{tag}")
        row_metrics = row.get("metrics") or {}
        for metric in metrics:
            node = row_metrics.get(metric)
            if not isinstance(node, dict) or node.get("value") is None:
                unmeasured.append(f"objective_metric_absent:{metric}:{tag}")

    ref_index: dict[str, dict[Any, dict[str, float]]] = {}
    cand_index: dict[str, dict[Any, dict[str, float]]] = {}
    common: set[Any] = set()
    for step, metric in enumerate(metrics):
        ref_index[metric] = {}
        cand_index[metric] = {}
        for row in baseline:
            point = _point(row, metric)
            if point is not None:
                ref_index[metric][row.get("seed")] = point
        for row in candidate:
            point = _point(row, metric)
            if point is not None:
                cand_index[metric][row.get("seed")] = point
        both = set(ref_index[metric]) & set(cand_index[metric])
        common = both if step == 0 else common & both
    journal = sorted(common, key=lambda seed: (type(seed).__name__, str(seed)))  # J, deterministic
    n_pairs = len(journal)
    if n_pairs < confirm_repeats:
        unmeasured.append(f"n_pairs_below_confirm:{n_pairs}/{confirm_repeats}")

    per_objective: dict[str, Any] = {}
    wins: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    summaries: list[str] = []
    for obj in objectives:
        metric = str(obj.get("metric"))
        sign = SIGNS.get(str(obj.get("direction") or "max"), 1.0)
        pairs = [(ref_index[metric][seed], cand_index[metric][seed]) for seed in journal]
        floor_j = noise_floor(baseline, metric, baseline_repeats)
        if floor_j is None:
            unmeasured.append(f"noise_floor_uncalibrated:{metric}")
        if not pairs or floor_j is None or n_pairs < confirm_repeats:
            # missing / insufficient evidence: nothing is asserted for this objective, never a zero
            per_objective[metric] = {"mean_delta": None, "tau": None, "class": "unmeasured", "n_pairs": len(pairs)}
            continue
        deltas = [sign * (cand["value"] - ref["value"]) for ref, cand in pairs]
        delta_j = statistics.fmean(deltas)
        mean_ref = statistics.fmean([ref["value"] for ref, _ in pairs])
        tau_j = max(floor_j, rel * abs(mean_ref), k * math.sqrt(_mean_sq(pairs, 0) + _mean_sq(pairs, 1)))
        if delta_j > tau_j:
            class_j = "win"
            wins.append(metric)
        elif delta_j < -tau_j:
            class_j = "loss"
            losses.append(metric)
        else:
            class_j = "tie"  # exact +-tau_j is a tie (matches the decide() strictness)
            ties.append(metric)
        per_objective[metric] = {"mean_delta": delta_j, "tau": tau_j, "class": class_j, "n_pairs": len(pairs)}
        summaries.append(f"{metric}: delta {delta_j:.6g} vs tau {tau_j:.6g} -> {class_j}")

    breaches, guard_unmeasured = _guard_checks(
        baseline, candidate, guards, guard_dirs, k, guard_eps, rl_floor=spec.get("rl_min_measured_fraction")
    )
    unmeasured.extend(guard_unmeasured)
    reasons.extend(breaches)

    simpler = any(bool(row.get("simpler")) for row in candidate)
    reasons.extend(summaries)
    if unmeasured:  # unmeasured evidence dominates: there is nothing to declare a regression from
        reasons.extend(unmeasured)
        verdict = "unmeasured"
    elif breaches:  # a guardrail veto beats even a dominating candidate
        verdict = "rejected_regress"
    elif wins and not losses:
        verdict = "accepted_gain"  # tau-weakly dominates the baseline
        reasons.append(f"objectives win (dominates): {','.join(sorted(wins))}")
    elif wins and losses:
        verdict = "tradeoff"  # a decision value, never claimable (AR-RS-008 / AR-HO-008)
        reasons.append(
            f"objectives win/loss (tradeoff): win {','.join(sorted(wins))} loss {','.join(sorted(losses))}"
        )
    elif losses and not wins:
        verdict = "rejected_regress"  # the baseline tau-weakly dominates the candidate
        reasons.append(f"objectives loss (regressed): {','.join(sorted(losses))}")
    elif simpler:
        verdict = "accepted_flat"
        reasons.append("|mean_delta| <= tau for every objective with a simpler candidate")
    else:
        verdict = "no_gain"
        reasons.append("|mean_delta| <= tau for every objective and the candidate is not simpler")

    primary = per_objective[metrics[0]] if metrics else {"mean_delta": None, "tau": None}
    return {
        "verdict": verdict,
        "mean_delta": primary["mean_delta"],
        "tau": primary["tau"],
        "n_pairs": n_pairs,
        "reasons": reasons,
        "per_objective": per_objective,
        "wins": sorted(wins),
        "losses": sorted(losses),
        "ties": sorted(ties),
    }


def frontier(decisions: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Report-only tau-weak Pareto frontier over measured decisions: ``(entries, drops)``.

    Trial A is dominated by B when every objective delta of B is >= -t below A's and at least one
    is > t above it, with t = max(tau_A, tau_B) per objective.  Presentation only: never on
    accept/claim.  Inputs are not mutated.
    """
    drops: list[str] = []
    measured: list[str] = []
    for trial in sorted(decisions):
        decision = decisions[trial]
        if not isinstance(decision, dict) or decision.get("verdict") == "unmeasured" or not decision.get("per_objective"):
            drops.append(f"frontier_excluded:{trial}:unmeasured")
            continue
        measured.append(trial)

    index: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for trial in measured:
        deltas: dict[str, float] = {}
        taus: dict[str, float] = {}
        for name, entry in (decisions[trial].get("per_objective") or {}).items():
            node = dict(entry or {})
            if node.get("mean_delta") is None:
                continue  # unmeasured objectives never take part in dominance
            deltas[str(name)] = float(node["mean_delta"])
            taus[str(name)] = float(node.get("tau") or 0.0)
        index[trial] = (deltas, taus)

    entries: list[dict[str, Any]] = []
    for trial in measured:
        delta_a, tau_a = index[trial]
        dominated = False
        for other in measured:
            if other == trial:
                continue
            delta_b, tau_b = index[other]
            if not delta_a or any(name not in delta_b for name in delta_a):
                continue
            edges = [delta_b[name] - delta_a[name] for name in sorted(delta_a)]
            bands = [max(tau_a.get(name, 0.0), tau_b.get(name, 0.0)) for name in sorted(delta_a)]
            if all(edge >= -band for edge, band in zip(edges, bands)) and any(
                edge > band for edge, band in zip(edges, bands)
            ):
                dominated = True
                break
        if dominated:
            drops.append(f"frontier_excluded:{trial}:dominated")
            continue
        decision = decisions[trial]
        entries.append(
            {
                "trial": trial,
                "class": decision.get("verdict"),
                "wins": list(decision.get("wins") or []),
                "losses": list(decision.get("losses") or []),
            }
        )
    entries.sort(key=lambda entry: (-len(entry["wins"]), len(entry["losses"]), str(entry["trial"])))
    return entries, drops
