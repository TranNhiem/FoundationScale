"""Evaluation statistics for agentic-RL benchmarks: pass@k and paired comparisons (ARL-G13).

Benchmarks here are small (AIME25 has 30 problems), so a difference between two checkpoints is
only a finding once it clears the sampling noise. This module is torch-free and stdlib-only.

* :func:`pass_at_k` -- the unbiased estimator of Chen et al. (2021, arXiv:2107.03374):
  ``1 - C(n-c, k) / C(n, k)`` per task, averaged over tasks.
* :func:`paired_bootstrap` -- the difference of per-task means between two evaluations of the SAME
  tasks, with a percentile bootstrap confidence interval obtained by resampling TASKS (the unit
  the benchmark samples), and the fraction of resamples at or below zero.

Per-task input is ``{uid: [score, ...]}``; ``None`` scores are abstentions and are dropped from
that task's mean (a task with no measured score is dropped from the comparison and counted).
WHAT IS NOT CLAIMED: that a CI excluding zero makes a difference practically important, or that
tasks are independent draws from the population of interest -- it is the noise floor of THIS set.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class EvalStatsRefusal(ValueError):
    """Inputs that cannot support the requested statistic."""


def _measured(scores: Sequence[float | None]) -> list[float]:
    out: list[float] = []
    for score in scores:
        if score is None:
            continue
        if type(score) is bool or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise EvalStatsRefusal(f"score {score!r}: must be None or a finite real number")
        out.append(float(score))
    return out


def pass_at_k(per_task: Mapping[str, Sequence[float | None]], k: int) -> float:
    """Mean unbiased pass@k over tasks; a sample "passes" when its score is >= 1.0.

    Refuses k < 1 and any task with fewer than k measured samples (the estimator needs n >= k).
    """
    if type(k) is not int or k < 1:
        raise EvalStatsRefusal(f"k={k!r}: must be an int >= 1")
    if not per_task:
        raise EvalStatsRefusal("pass_at_k: no tasks")
    values: list[float] = []
    for uid, scores in per_task.items():
        measured = _measured(scores)
        n, c = len(measured), sum(1 for s in measured if s >= 1.0)
        if n < k:
            raise EvalStatsRefusal(f"task {uid!r}: {n} measured sample(s) < k={k}")
        values.append(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k))
    return statistics.fmean(values)


@dataclass(frozen=True)
class PairedComparison:
    """``mean_b - mean_a`` over the shared tasks, with a bootstrap CI over tasks."""

    n_tasks: int
    n_dropped: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_le_zero: float  # fraction of bootstrap resamples with diff <= 0
    confidence: float
    resamples: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def paired_bootstrap(
    a: Mapping[str, Sequence[float | None]],
    b: Mapping[str, Sequence[float | None]],
    *,
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Compare two evaluations of the same task set (``b - a``), resampling tasks.

    Refuses differing task sets (a paired comparison over different tasks is not paired), fewer
    than 2 shared measured tasks, or nonsensical ``resamples``/``confidence``.
    """
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise EvalStatsRefusal(
            f"task sets differ: {len(only_a)} only in A (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in B (e.g. {only_b[:3]})"
        )
    if type(resamples) is not int or resamples < 100:
        raise EvalStatsRefusal(f"resamples={resamples!r}: must be an int >= 100")
    if not 0.5 < confidence < 1.0:
        raise EvalStatsRefusal(f"confidence={confidence!r}: must lie in (0.5, 1.0)")
    pairs: list[tuple[float, float]] = []
    dropped = 0
    for uid in sorted(a):
        ma, mb = _measured(a[uid]), _measured(b[uid])
        if not ma or not mb:
            dropped += 1
            continue
        pairs.append((statistics.fmean(ma), statistics.fmean(mb)))
    if len(pairs) < 2:
        raise EvalStatsRefusal(f"{len(pairs)} shared measured task(s): need >= 2")
    deltas = [y - x for x, y in pairs]
    rng = random.Random(seed)
    n = len(deltas)
    boot = sorted(statistics.fmean(rng.choices(deltas, k=n)) for _ in range(resamples))
    alpha = (1.0 - confidence) / 2.0
    low = boot[max(0, math.floor(alpha * resamples))]
    high = boot[min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1)]
    return PairedComparison(
        n_tasks=n,
        n_dropped=dropped,
        mean_a=statistics.fmean(x for x, _ in pairs),
        mean_b=statistics.fmean(y for _, y in pairs),
        diff=statistics.fmean(deltas),
        ci_low=low,
        ci_high=high,
        p_le_zero=sum(1 for d in boot if d <= 0.0) / resamples,
        confidence=confidence,
        resamples=resamples,
    )
