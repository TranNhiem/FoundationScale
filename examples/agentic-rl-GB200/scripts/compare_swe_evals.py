"""Paired task-clustered bootstrap over two eval_swe --dump files (reward + resolve);
see compare_evals.py for the general per-task tool.

    python compare_swe_evals.py --base A.dump.jsonl --cand B.dump.jsonl [--resamples 10000]
        [--seed 0] [--out cmp.json]

Pairs the two runs on tasks present (and measured) in both, averages each task's
samples first (the task is the
unit), and reports the mean candidate-minus-base difference with a 95% bootstrap CI over tasks
for the partial-credit
reward and for binary resolve (reward == 1). An abstained sample (reward null) is dropped
from its task's mean, never
counted as 0; a task with no measured sample on either side is excluded and counted. Verdict
per metric: "better" if
the CI is above 0, "worse" if below, else "no detectable difference"; fewer than 20
paired tasks is "unmeasured".
Exit 95 when any verdict is unmeasured, else 0 (it reports, it does not gate).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

MIN_PAIRED = 20  # fewer paired tasks give a degenerate bootstrap; never a verdict


def per_task(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("reward") is not None:
            out[str(row["session"]).rsplit("#", 1)[0]].append(float(row["reward"]))
    return out


def bootstrap(diffs: list[float], resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def compare(base: dict, cand: dict, resamples: int, seed: int) -> dict:
    tasks = sorted(set(base) & set(cand))
    result = {
        "paired_tasks": len(tasks),
        "base_only": len(set(base) - set(cand)),
        "cand_only": len(set(cand) - set(base)),
    }
    for metric, f in (("reward", lambda v: v), ("resolve", lambda v: 1.0 if v == 1.0 else 0.0)):
        b = [sum(map(f, base[t])) / len(base[t]) for t in tasks]
        c = [sum(map(f, cand[t])) / len(cand[t]) for t in tasks]
        d = [y - x for x, y in zip(b, c, strict=True)]
        if len(d) < MIN_PAIRED:
            result[metric] = {
                "verdict": "unmeasured",
                "reason": f"{len(d)} paired tasks < {MIN_PAIRED}",
            }
            continue
        lo, hi = bootstrap(d, resamples, seed)
        result[metric] = {
            "base": sum(b) / len(b),
            "cand": sum(c) / len(c),
            "diff": sum(d) / len(d),
            "ci95": [lo, hi],
            "verdict": "better" if lo > 0 else "worse" if hi < 0 else "no detectable difference",
        }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = compare(per_task(a.base), per_task(a.cand), a.resamples, a.seed)
    text = json.dumps(res, indent=2)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    # report-only tool, but an unmeasured comparison must not look like PASS to automation
    # (0/5/95/96 doctrine)
    return (
        95
        if any(res.get(m, {}).get("verdict") == "unmeasured" for m in ("reward", "resolve"))
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
