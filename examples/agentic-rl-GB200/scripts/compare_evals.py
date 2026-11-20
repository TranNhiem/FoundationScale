"""Compare two evaluations of the same benchmark: paired bootstrap over tasks (ARL-G13).

    python compare_evals.py A_per_task.jsonl B_per_task.jsonl [--resamples 10000] [--k 1 4]

Each file is the ``--per-task-out`` JSONL of ``eval_math.py`` (one ``{"uid", "scores"}`` per task),
or several files joined with ``+`` (e.g. three seeds: ``s10.jsonl+s11.jsonl+s12.jsonl``), whose
samples are pooled per task. With ``--seeds`` the '+'-joined files are instead independent
training seeds and the CI is a hierarchical (seed, then task) bootstrap -- use this for any
algorithm A/B: a task-only CI over one seed per arm calls training-seed noise a finding.
Prints one JSON line: per-side means, ``diff = B - A``, its bootstrap
CI over tasks, the fraction of resamples at or below zero, and pass@k per side.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from foundationscale.agentic_rl.eval_stats import (
    EvalStatsRefusal,
    paired_bootstrap,
    pass_at_k,
    seed_bootstrap,
)


def load(spec: str) -> dict[str, list[float | None]]:
    pooled: dict[str, list[float | None]] = {}
    for path in spec.split("+"):
        with Path(path).open() as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    pooled.setdefault(row["uid"], []).extend(row["scores"])
    return pooled


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--k", type=int, nargs="*", default=[1])
    p.add_argument(
        "--seeds",
        action="store_true",
        help="treat '+'-joined files as independent SEEDS (hierarchical seed+task bootstrap)",
    )
    args = p.parse_args()
    if args.seeds:
        a_runs = [load(path) for path in args.a.split("+")]
        b_runs = [load(path) for path in args.b.split("+")]
        cmp = seed_bootstrap(a_runs, b_runs, resamples=args.resamples, confidence=args.confidence)
        print(json.dumps(dataclasses.asdict(cmp)))
        return
    a, b = load(args.a), load(args.b)
    result = dataclasses.asdict(
        paired_bootstrap(a, b, resamples=args.resamples, confidence=args.confidence)
    )
    for k in args.k:
        for side, data in (("a", a), ("b", b)):
            try:
                result[f"pass@{k}_{side}"] = pass_at_k(data, k)
            except EvalStatsRefusal:
                result[f"pass@{k}_{side}"] = None
    print(json.dumps(result))


if __name__ == "__main__":
    main()
