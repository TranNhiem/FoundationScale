"""A2.2 keep-only-if rule: paired with-skill A/B of two behaviour reports of one package.

A writing change to a SKILL.md is kept only when the with-skill item pass rate rises AND the paired
(by case) bootstrap interval of the with-skill deltas excludes 0 upwards; uplift against the
ablation is not the question here (that is :mod:`foundationskills.agent.skill_impact`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from foundationskills.agent.skill_impact import (
    ARMS,
    MIN_REPS,
    SkillImpactRefused,
    _bootstrap_ci,
    _group_calls,
    _rate,
    _require_index,
    _require_report,
)


def ab_compare(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """A2.2 keep-only-if rule for one writing change: pair each case's WITH-SKILL item pass rate
    between the report measured before the change and the one measured after it (same package,
    same case set, same models and reps), bootstrap the paired deltas, and keep the change only
    when the with-skill rate rises AND the interval excludes 0 upwards. Anything that could make
    the comparison unfair refuses; a missing
    cell makes the verdict ``unmeasured``, never ``keep``."""
    for name, report in (("before", before), ("after", after)):
        _require_report(report)
        if len(_require_index(report)) != 1:
            raise SkillImpactRefused(
                f"ab_compare: the {name} report must measure exactly one package"
            )
    entry_b, entry_a = _require_index(before)[0], _require_index(after)[0]
    if entry_b.get("package") != entry_a.get("package"):
        raise SkillImpactRefused(
            "ab_compare: package differs "
            f"({entry_b.get('package')!r} vs {entry_a.get('package')!r})"
        )
    same = ("cases_sha256", "evals_sha256")
    for key in same:
        if entry_b.get(key) != entry_a.get(key):
            raise SkillImpactRefused(
                f"ab_compare: {key} differs: the two reports measured different cases"
            )
    for key in ("cases", "agent_model", "judge_model", "reps", "agent_temperature"):
        if before.get(key) != after.get(key):
            raise SkillImpactRefused(
                f"ab_compare: {key} differs ({before.get(key)!r} vs {after.get(key)!r})"
            )
    if entry_b.get("skill_bundle_sha256") == entry_a.get("skill_bundle_sha256"):
        raise SkillImpactRefused(
            "ab_compare: both reports measured the same SKILL.md bundle: nothing changed"
        )
    name = str(entry_b.get("name"))
    rates, unmeasured = {}, 0
    totals = {}
    for label, report in (("before", before), ("after", after)):
        cells, missing = _group_calls(report["calls"], [name])
        unmeasured += sum(
            1 for c in report["calls"] if c.get("arm") == ARMS[0] and not c.get("measured")
        )
        rates[label] = {
            cid: _rate(arms[ARMS[0]]["met"], arms[ARMS[0]]["graded"])
            for cid, arms in cells[name].items()
            if ARMS[0] in arms
        }
        met = sum(arms[ARMS[0]]["met"] for arms in cells[name].values() if ARMS[0] in arms)
        graded = sum(arms[ARMS[0]]["graded"] for arms in cells[name].values() if ARMS[0] in arms)
        totals[label] = _rate(met, graded)
    paired = sorted(
        c
        for c in rates["before"]
        if rates["before"][c] is not None and rates["after"].get(c) is not None
    )
    deltas = [rates["after"][c] - rates["before"][c] for c in paired]
    ci_low, ci_high = _bootstrap_ci(deltas, n_boot=n_boot, seed=seed, level=level)
    reasons = []
    if int(before.get("reps") or 0) < MIN_REPS:
        reasons.append(f"reps < {MIN_REPS}")
    if unmeasured:
        reasons.append(f"{unmeasured} unmeasured with_skill call(s)")
    if ci_low is None or ci_high is None:
        reasons.append("fewer than 2 paired cases")
    rise = (
        None
        if totals["before"] is None or totals["after"] is None
        else totals["after"] - totals["before"]
    )
    if reasons:
        verdict = "unmeasured"
    elif rise is not None and rise > 0 and ci_low > 0:
        verdict = "keep"
    else:
        verdict = "reject"
    return {
        "schema": "fskills.skill_ab/1",
        "package": name,
        "cases": before.get("cases"),
        "skill_bundle_before": entry_b.get("skill_bundle_sha256"),
        "skill_bundle_after": entry_a.get("skill_bundle_sha256"),
        "with_before": totals["before"],
        "with_after": totals["after"],
        "rise": rise,
        "mean_delta": (math.fsum(deltas) / len(deltas)) if deltas else None,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_paired": len(paired),
        "verdict": verdict,
        "reasons": reasons,
    }
