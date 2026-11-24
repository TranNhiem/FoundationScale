"""C9: build the admission evidence for one candidate from measured reports, never from hand-typed numbers.

Metrics come from behaviour-eval reports (incumbent: the package's SKILL.md; candidate: the same cases run with
``--skill-md-override``) and from routing-eval reports (per-package trigger recall in the gated arm). The evidence is
``measured`` only when every report is measured and the candidate reports really measured the candidate text.
Aligned case pairs and the candidate's activation-beacon count feed the admission paired-significance and beacon gates.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Mapping


def with_skill(report: Mapping[str, Any], key: str) -> float | None:
    value = ((report.get("results") or {}).get("with_skill") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def trigger_recall(report: Mapping[str, Any], package: str) -> float | None:
    """Share of the gated arm's routing calls expecting ``package`` that picked it (None when there are none)."""
    arm = report.get("gated_arm") or "full"
    calls = [c for c in report.get("calls") or []
             if c.get("arm") == arm and c.get("expected") == package and c.get("measured") in (True, "True")]
    if not calls:
        return None
    return sum(1 for c in calls if c.get("correct") in (True, "True")) / len(calls)


def eval_calls(*reports: Mapping[str, Any]) -> int:
    return sum(len(r.get("calls") or []) for r in reports)


def _case_pass_value(call: Mapping[str, Any]) -> float:
    graded = int(call.get("items_graded") or 0)
    met = int(call.get("items_met") or 0)
    return 1.0 if graded > 0 and met == graded else 0.0


def paired_cases(incumbent_report: Mapping[str, Any], candidate_report: Mapping[str, Any], package: str,
                 arm: str = "with_skill") -> list[list[float]]:
    """Return aligned incumbent/candidate case outcomes, ordered by ``(case_id, rep)``."""
    def keyed_calls(report: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
        calls = {}
        for call in report.get("calls") or []:
            if (call.get("arm") == arm and call.get("package") == package
                    and call.get("measured") in (True, "True")):
                key = (str(call.get("case_id")), int(call.get("rep")))
                calls[key] = call
        return calls

    incumbent_calls = keyed_calls(incumbent_report)
    candidate_calls = keyed_calls(candidate_report)
    keys = sorted(incumbent_calls.keys() & candidate_calls.keys())
    return [[_case_pass_value(incumbent_calls[key]), _case_pass_value(candidate_calls[key])] for key in keys]


def beacon_fired(report: Mapping[str, Any], condition: str, package: str, arm: str = "with_skill") -> int:
    """Count measured calls for the arm/package whose answers contain ``condition``."""
    needle = condition.lower()
    return sum(
        1 for call in report.get("calls") or []
        if (call.get("arm") == arm and call.get("package") == package
            and call.get("measured") in (True, "True")
            and needle in str(call.get("answer") or "").lower())
    )


def paired_signflip_p(pairs: list[list[float]]) -> tuple[float, float]:
    """Return mean candidate-minus-incumbent gain and its one-sided exact/Monte-Carlo sign-flip p-value."""
    differences = [float(candidate) - float(incumbent) for incumbent, candidate in pairs]
    mean_gain = sum(differences) / len(differences) if differences else 0.0
    nonzero = [difference for difference in differences if difference != 0]
    if not nonzero:
        return mean_gain, 1.0

    observed = sum(nonzero)
    if len(nonzero) <= 16:
        assignments = itertools.product((-1.0, 1.0), repeat=len(nonzero))
        hits = sum(
            1 for signs in assignments
            if sum(value * sign for value, sign in zip(nonzero, signs)) >= observed
        )
        return mean_gain, hits / (1 << len(nonzero))

    draws = 20000
    rng = random.Random(0)
    hits = 0
    for _ in range(draws):
        total = sum(value * rng.choice((-1.0, 1.0)) for value in nonzero)
        if total >= observed:
            hits += 1
    return mean_gain, (hits + 1) / (draws + 1)


def build_evidence(package: str, incumbent: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]],
                   target: str = "behaviour_case_pass", beacon: str | None = None) -> dict[str, Any]:
    """``incumbent``/``candidate`` map "routing_behaviour", "oracle" and "routing" to their reports."""
    def metrics(side: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
        out = {"behaviour_case_pass": with_skill(side.get("routing_behaviour") or {}, "case_pass_rate"),
               "oracle_item_pass": with_skill(side.get("oracle") or {}, "item_pass_rate"),
               "routing_recall": trigger_recall(side.get("routing") or {}, package)}
        return {k: v for k, v in out.items() if v is not None}

    reports = [*incumbent.values(), *candidate.values()]
    measured = (all(str(r.get("verdict")) != "unmeasured" for r in reports)
                and all(package in (r.get("skill_md_override") or {})
                        for key, r in candidate.items() if key != "routing"))
    inc, cand = metrics(incumbent), metrics(candidate)
    calls = eval_calls(*candidate.values())
    gain = cand[target] - inc[target] if target in cand and target in inc else None
    evidence = {"measured": bool(measured), "target": target, "incumbent": inc, "candidate": cand,
                "candidate_eval_calls": calls,
                "gain_per_100_calls": round(100 * gain / calls, 4) if gain is not None and calls else None,
                "paired": {"target": target,
                           "pairs": paired_cases(incumbent.get("routing_behaviour") or {},
                                                 candidate.get("routing_behaviour") or {}, package)}}
    if isinstance(beacon, str) and beacon:
        evidence["beacon"] = {
            "condition": beacon,
            "fired": beacon_fired(candidate.get("routing_behaviour") or {}, beacon, package),
        }
    return evidence
