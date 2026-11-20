"""C9: build the admission evidence for one candidate from measured reports, never from hand-typed numbers.

Metrics come from behaviour-eval reports (incumbent: the package's SKILL.md; candidate: the same cases run with
``--skill-md-override``) and from routing-eval reports (per-package trigger recall in the gated arm). The evidence is
``measured`` only when every report is measured and the candidate reports really measured the candidate text.
"""

from __future__ import annotations

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


def build_evidence(package: str, incumbent: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]],
                   target: str = "behaviour_case_pass") -> dict[str, Any]:
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
    return {"measured": bool(measured), "target": target, "incumbent": inc, "candidate": cand,
            "candidate_eval_calls": calls,
            "gain_per_100_calls": round(100 * gain / calls, 4) if gain is not None and calls else None}
