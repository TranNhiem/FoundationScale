"""``skill-behaviour`` suite (A4.5): behaviour-eval reports as a benchmark with measured-only denominators.

The behaviour harness (``fskills behaviour-eval``) asks an agent model to plan every case with and without the
skill in context, and a judge grades each call item by item. This suite reads those JSON reports and re-scores the
gated arm (``with_skill`` unless the report names another one):
- cell = one call. ``pass`` when every graded item was met, ``fail`` when some were not, ``unmeasured`` when the
  report marks the call unmeasured (agent or judge error, empty reply). Unmeasured calls never count as 0;
- score per package = items met / items graded over measured calls, the harness's own ``item_pass_rate``
  (agent/behaviour_eval.py). Each recomputed rate is cross-checked against the report's ``results`` block, and a
  disagreement refuses: a re-score that does not reproduce its source measures nothing;
- refused-fixture items = the oracle cases of BLOCK input rules, whose correct answer is REFUSED. A suite run must
  include at least one measured such call, so a routing-only set (no oracle cases) is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUITE = "skill-behaviour"


class SkillBehaviourRefused(ValueError):
    """The reports cannot carry a score (unreadable, no gated-arm calls, no refused-fixture item, no reproduction)."""


def _load(path: str) -> dict[str, Any]:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillBehaviourRefused(f"{path}: unreadable behaviour report ({exc})") from exc
    if not isinstance(report, dict) or not isinstance(report.get("calls"), list):
        raise SkillBehaviourRefused(f"{path}: not a behaviour-eval report (no calls list)")
    return report


def score(reports: list[str], max_unmeasured_fraction: float = 0.0) -> dict[str, Any]:
    """The skill-behaviour suite report over ``reports``; raises SkillBehaviourRefused."""
    packages: dict[str, dict[str, Any]] = {}
    refused_fixtures = 0
    cells = unmeasured = 0
    for path in reports:
        report = _load(path)
        arm = str(report.get("gated_arm") or "with_skill")
        calls = [c for c in report["calls"] if isinstance(c, dict) and c.get("arm") == arm]
        if not calls:
            raise SkillBehaviourRefused(f"{path}: no {arm!r} calls")
        mine: dict[str, dict[str, Any]] = {}
        for call in calls:
            pkg = mine.setdefault(str(call.get("package")), {"calls": 0, "measured": 0, "items_met": 0,
                                                              "items_graded": 0, "passes": 0, "unmeasured": 0})
            pkg["calls"] += 1
            cells += 1
            if not call.get("measured"):
                pkg["unmeasured"] += 1
                unmeasured += 1
                continue
            met, graded = int(call.get("items_met") or 0), int(call.get("items_graded") or 0)
            pkg["measured"] += 1
            pkg["items_met"] += met
            pkg["items_graded"] += graded
            pkg["passes"] += int(graded > 0 and met == graded)
            if call.get("phase") == "input" and call.get("severity") == "BLOCK":
                refused_fixtures += 1
        source = ((report.get("results") or {}).get(arm) or {}).get("packages") or {}
        for name, pkg in mine.items():
            rate = pkg["items_met"] / pkg["items_graded"] if pkg["items_graded"] else None
            expected = (source.get(name) or {}).get("item_pass_rate")
            if expected is not None and (rate is None or abs(rate - float(expected)) > 1e-9):
                raise SkillBehaviourRefused(f"{path}: {name} item_pass_rate {rate} does not reproduce the report's "
                                            f"{expected}")
            key = f"{name}:{report.get('cases') or 'cases'}"
            packages[key] = {**pkg, "item_pass_rate": None if rate is None else round(rate, 4),
                             "case_pass_rate": round(pkg["passes"] / pkg["measured"], 4) if pkg["measured"] else None,
                             "source": str(path), "arm": arm}
    if not refused_fixtures:
        raise SkillBehaviourRefused("no measured refused-fixture item (an oracle case of a BLOCK input rule): a "
                                    "suite that never asks for a refusal cannot show one is recognised")
    measured_rates = [p["item_pass_rate"] for p in packages.values() if p["item_pass_rate"] is not None]
    fraction = unmeasured / cells if cells else 1.0
    verdict = "UNMEASURED" if (fraction > max_unmeasured_fraction or not measured_rates) else "PASS"
    met = sum(p["items_met"] for p in packages.values())
    graded = sum(p["items_graded"] for p in packages.values())
    return {"suite": SUITE, "verdict": verdict, "item_pass_rate": round(met / graded, 4) if graded else None,
            "packages": packages, "refused_fixture_calls": refused_fixtures, "cells": cells,
            "unmeasured_cells": unmeasured, "unmeasured_fraction": round(fraction, 4),
            "max_unmeasured_fraction": max_unmeasured_fraction}
