"""``harbor`` suite (A4.5): score Harbor job directories with measured-only denominators and negative controls.

Harbor (the Skill2Env paper's harness, paper:336) writes one ``<job>/<task>__<id>/result.json`` per trial. A trial
counts only when its directory holds ``agent/`` or ``verifier/``, the rule B2.2 registered in
``score_b22.py``. Each trial is a cell:
- ``pass``: the verifier returned numeric rewards and every one == 1;
- ``fail``: numeric rewards, not all 1. This includes AgentTimeoutError with a reward: the agent ran out of time and
  the verifier measured the result;
- ``unmeasured``: no numeric reward (an exception before or inside the verifier, i.e. infra). An abstention is
  never folded in as 0.

Primary score: pass@1 = mean over tasks of passes / measured trials. A task with no measured trial is excluded and
listed. Also reported, labelled ``paper_denominator``: B2.2's registered rule (reward-less trials and trial-less
tasks count 0), so a published comparison stays reproducible. CIs are a bootstrap over tasks (10,000 resamples,
seed 0, as registered).

Negative controls (the refused-fixture items): trials of a ``nop`` agent over the same tasks. Their correct outcome
is non-pass. A task whose control passes has a verifier that cannot discriminate; it is dropped from the score and
counted. Missing controls, or controls none of which measured a non-pass, refuse the suite: a scorer that cannot
say "no" measures nothing.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SUITE = "harbor"
BOOTSTRAP, SEED = 10_000, 0


class HarborSuiteRefused(ValueError):
    """The job directories cannot carry a score (no trials, no usable negative controls, unreadable task list)."""


def _cells(job_dirs: Iterable[str | Path], agents: set[str] | None = None) -> dict[str, list[str]]:
    """task -> trial cells; ``agents`` (when given) collects the agent name each trial records."""
    cells: dict[str, list[str]] = defaultdict(list)
    for job in job_dirs:
        root = Path(job)
        if not root.is_dir():
            raise HarborSuiteRefused(f"harbor job directory {job} does not exist")
        for result in sorted(root.rglob("result.json")):
            trial = result.parent
            if not ((trial / "agent").is_dir() or (trial / "verifier").is_dir()):
                continue  # a job-level summary, not a trial (score_b22.py rule)
            try:
                record = json.loads(result.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                cells[trial.name.split("__")[0]].append("unmeasured")
                continue
            if agents is not None:
                agent = (record.get("agent_info") or {}).get("name") or ((record.get("config") or {}).get("agent") or {}).get("name")
                agents.add(str(agent))
            task = str(record.get("task_name") or "").split("/")[-1] or trial.name.split("__")[0]
            rewards = (record.get("verifier_result") or {}).get("rewards") or {}
            numbers = [v for v in rewards.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            cells[task].append("unmeasured" if not numbers else "pass" if all(v == 1 for v in numbers) else "fail")
    return dict(cells)


def _bootstrap(values: list[float]) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(SEED)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(BOOTSTRAP))
    return [round(100 * means[int(0.025 * BOOTSTRAP)], 2), round(100 * means[int(0.975 * BOOTSTRAP) - 1], 2)]


def score(job_dirs: list[str], control_dirs: list[str], tasks: list[str] | None = None,
          max_unmeasured_fraction: float = 0.0) -> dict[str, Any]:
    """The harbor suite report; raises HarborSuiteRefused. ``tasks`` fixes the task universe (else: tasks seen)."""
    cells = _cells(job_dirs)
    if not cells:
        raise HarborSuiteRefused(f"no Harbor trials (result.json beside agent/ or verifier/) under {job_dirs}")
    control_agents: set[str] = set()
    controls = _cells(control_dirs, control_agents) if control_dirs else {}
    if controls and control_agents != {"nop"}:
        # a solving agent (oracle, pi) passed as "controls" would drop every task it solves and still print a number
        raise HarborSuiteRefused(f"negative-control trials must come from the nop agent; found {sorted(control_agents)}")
    measured_controls = {t: [c for c in cs if c != "unmeasured"] for t, cs in controls.items()}
    if not any(measured_controls.values()):
        raise HarborSuiteRefused("no measured negative-control trial (a nop agent over the same tasks): a suite "
                                 "that cannot show a non-pass cannot vouch for its verifier")
    non_discriminating = sorted(t for t, cs in measured_controls.items() if "pass" in cs)
    if all("pass" in cs for cs in measured_controls.values() if cs):
        raise HarborSuiteRefused(f"every measured negative control passed ({len(non_discriminating)} tasks): "
                                 "the verifier cannot discriminate")
    universe = sorted(tasks) if tasks else sorted(cells)
    rows, primary, paper = [], [], []
    for task in universe:
        mine = cells.get(task, [])
        measured = [c for c in mine if c != "unmeasured"]
        passes = mine.count("pass")
        paper.append(passes / len(mine) if mine else 0.0)  # B2.2's registered denominator, kept for comparison
        dropped = task in non_discriminating
        if measured and not dropped:
            primary.append(passes / len(measured))
        rows.append({"task": task, "trials": len(mine), "measured": len(measured), "passes": passes,
                     "unmeasured": len(mine) - len(measured), "control": controls.get(task),
                     "dropped": "non_discriminating_verifier" if dropped else (None if measured else "no_measured_trial")})
    trials = sum(r["trials"] for r in rows)
    unmeasured = sum(r["unmeasured"] for r in rows) + sum(1 for r in rows if r["trials"] == 0)
    cells_total = trials + sum(1 for r in rows if r["trials"] == 0)
    fraction = unmeasured / cells_total if cells_total else 1.0
    verdict = "UNMEASURED" if (fraction > max_unmeasured_fraction or not primary) else "PASS"
    return {
        "suite": SUITE, "verdict": verdict,
        "pass_at_1": round(100 * sum(primary) / len(primary), 2) if primary else None,
        "ci95": _bootstrap(primary),
        "tasks_scored": len(primary), "tasks_total": len(universe),
        "paper_denominator": {"pass_at_1": round(100 * sum(paper) / len(paper), 2), "ci95": _bootstrap(paper),
                              "rule": "reward-less trials and trial-less tasks count 0 (B2.2 registered scorer)"},
        "unmeasured_cells": unmeasured, "cells": cells_total, "unmeasured_fraction": round(fraction, 4),
        "max_unmeasured_fraction": max_unmeasured_fraction,
        "negative_controls": {"tasks": len(controls), "measured": sum(bool(v) for v in measured_controls.values()),
                              "non_discriminating": non_discriminating},
        "tasks": rows,
    }
