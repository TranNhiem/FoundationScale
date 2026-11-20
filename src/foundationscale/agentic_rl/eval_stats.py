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


@dataclass(frozen=True)
class SeedComparison:
    """``mean_b - mean_a`` across independently trained seeds, with a hierarchical CI."""

    seeds_a: int
    seeds_b: int
    n_tasks: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_le_zero: float
    seed_means_a: tuple[float, ...]
    seed_means_b: tuple[float, ...]
    confidence: float
    resamples: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def seed_bootstrap(
    a_runs: Sequence[Mapping[str, Sequence[float | None]]],
    b_runs: Sequence[Mapping[str, Sequence[float | None]]],
    *,
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> SeedComparison:
    """Compare two ARMS, each a list of independently trained seeds evaluated on the same tasks.

    Each bootstrap draw resamples seeds within each arm (with replacement), then resamples the
    shared TASKS once (the same task draw for both arms, keeping the comparison paired over tasks),
    and takes ``mean over drawn seeds and drawn tasks`` per arm. The CI therefore covers training-
    seed variance as well as evaluation noise: GB200 math baselines showed two seeds of ONE
    algorithm differing by +2.1 points on MATH-500 with a task-only CI excluding zero, so a
    task-only bootstrap over one seed per arm would call seed noise a finding.

    Refuses fewer than 2 seeds per arm (no seed variance can be estimated from one) and seeds whose
    task sets differ. Tasks with no measured score in ANY seed of either arm are dropped.
    """
    if len(a_runs) < 2 or len(b_runs) < 2:
        raise EvalStatsRefusal(
            f"{len(a_runs)} seed(s) in A and {len(b_runs)} in B: need >= 2 per arm to "
            f"estimate seed variance"
        )
    if type(resamples) is not int or resamples < 100:
        raise EvalStatsRefusal(f"resamples={resamples!r}: must be an int >= 100")
    if not 0.5 < confidence < 1.0:
        raise EvalStatsRefusal(f"confidence={confidence!r}: must lie in (0.5, 1.0)")
    runs = list(a_runs) + list(b_runs)
    tasks = set(runs[0])
    for index, run in enumerate(runs[1:], start=1):
        if set(run) != tasks:
            raise EvalStatsRefusal(f"seed {index}: task set differs from seed 0")
    # per-seed per-task means; a task is kept only if measured in every seed of both arms
    means: list[dict[str, float]] = []
    for run in runs:
        means.append({uid: statistics.fmean(m) for uid in run if (m := _measured(run[uid]))})
    shared = sorted(uid for uid in tasks if all(uid in m for m in means))
    if len(shared) < 2:
        raise EvalStatsRefusal(f"{len(shared)} task(s) measured in every seed: need >= 2")
    a_means, b_means = means[: len(a_runs)], means[len(a_runs) :]

    def arm_mean(arm: Sequence[dict[str, float]], picks: Sequence[str]) -> float:
        return statistics.fmean(statistics.fmean(m[uid] for uid in picks) for m in arm)

    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(resamples):
        picks = rng.choices(shared, k=len(shared))
        draw_a = rng.choices(a_means, k=len(a_means))
        draw_b = rng.choices(b_means, k=len(b_means))
        boot.append(arm_mean(draw_b, picks) - arm_mean(draw_a, picks))
    boot.sort()
    alpha = (1.0 - confidence) / 2.0
    mean_a, mean_b = arm_mean(a_means, shared), arm_mean(b_means, shared)
    return SeedComparison(
        seeds_a=len(a_runs),
        seeds_b=len(b_runs),
        n_tasks=len(shared),
        mean_a=mean_a,
        mean_b=mean_b,
        diff=mean_b - mean_a,
        ci_low=boot[max(0, math.floor(alpha * resamples))],
        ci_high=boot[min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1)],
        p_le_zero=sum(1 for d in boot if d <= 0.0) / resamples,
        seed_means_a=tuple(statistics.fmean(m[uid] for uid in shared) for m in a_means),
        seed_means_b=tuple(statistics.fmean(m[uid] for uid in shared) for m in b_means),
        confidence=confidence,
        resamples=resamples,
    )
