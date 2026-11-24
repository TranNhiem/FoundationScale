"""Frontier curriculum: sample tasks where GRPO has gradient (ARL-20).

A GRPO group whose samples all score the same gets zero centred advantage. On GB200 math about
half of all groups were like that -- tasks the policy always solves or never solves. Dynamic
sampling (``rollout.dynamic_sampling="refill"``) discards such groups AFTER paying for their
rollouts; a curriculum avoids drawing those tasks in the first place.

:class:`DifficultyRegistry` keeps, per task uid, how many measured rollouts it has seen and their
summed reward (each reward clipped to [0, 1], so a binary reward counts successes and partial
credit counts fractionally). Its estimate is the Beta(1, 1) posterior mean
``p = (s + 1) / (n + 2)``.

:class:`FrontierTaskSource` wraps a :class:`~foundationscale.agentic_rl.tasks.TaskSource` and draws
each step's tasks WITHOUT replacement with weight ``max(floor, 4 p (1 - p))``. That weight is 1 for
an unseen task (p = 1/2) and for any task near a 50% pass rate, and falls toward the floor as a task
becomes reliably solved or reliably failed. A task observed ``min_obs`` or more times with every
reward 0 (dead) or every reward 1 (solved) sits at exactly the floor, so it is still revisited
occasionally and the estimate can recover when the policy changes. Draws are deterministic given
``(seed, step)`` and the registry state (Efraimidis-Spirakis keys from a seeded RNG).

WHAT IS NOT CLAIMED: that frontier sampling improves held-out accuracy (the GB200 run measures
that), or that the registry survives a restart -- like the rest of the rollout host it is
in-memory; a resumed run starts with an empty registry, i.e. uniform sampling.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from foundationscale.agentic_rl.harness.base import EpisodeTask
from foundationscale.agentic_rl.tasks import TaskSource, TaskSourceRefusal


@dataclass
class DifficultyRegistry:
    """Per-task measured rollout counts and clipped reward sums."""

    counts: dict[str, int] = field(default_factory=dict)
    sums: dict[str, float] = field(default_factory=dict)

    def update(self, uid: str, reward: float | None) -> None:
        """Record one rollout of ``uid``. ``None`` (abstained: infra, unscored) is ignored --
        an unmeasured rollout says nothing about the task's difficulty."""
        if reward is None:
            return
        if (
            type(reward) is bool
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
        ):
            raise TaskSourceRefusal(
                f"DifficultyRegistry.update: reward {reward!r} for {uid!r} must be None or a "
                f"finite real number"
            )
        self.counts[uid] = self.counts.get(uid, 0) + 1
        self.sums[uid] = self.sums.get(uid, 0.0) + min(1.0, max(0.0, float(reward)))

    def pass_rate(self, uid: str) -> float:
        """Beta(1, 1) posterior mean: 1/2 for an unseen task."""
        return (self.sums.get(uid, 0.0) + 1.0) / (self.counts.get(uid, 0) + 2.0)

    def saturated(self, uid: str, min_obs: int) -> bool:
        """Observed ``min_obs``+ times with every reward 0 or every reward 1."""
        n = self.counts.get(uid, 0)
        if n < min_obs:
            return False
        s = self.sums.get(uid, 0.0)
        return s == 0.0 or s == float(n)


@dataclass
class FrontierTaskSource:
    """Draws tasks near the policy's pass-rate frontier (see the module docstring)."""

    base: TaskSource
    seed: int = 0
    min_obs: int = 4
    floor: float = 0.05
    registry: DifficultyRegistry = field(default_factory=DifficultyRegistry)

    def __post_init__(self) -> None:
        where = "FrontierTaskSource"
        if not isinstance(self.base, TaskSource):
            raise TaskSourceRefusal(f"{where}: field 'base' must be a TaskSource")
        if type(self.min_obs) is not int or self.min_obs < 1:
            raise TaskSourceRefusal(f"{where}: field 'min_obs' is {self.min_obs!r}: must be >= 1")
        if (
            type(self.floor) is bool
            or not isinstance(self.floor, (int, float))
            or not 0.0 < float(self.floor) <= 1.0
        ):
            raise TaskSourceRefusal(
                f"{where}: field 'floor' is {self.floor!r}: it must lie in (0, 1] -- a zero floor "
                f"would never revisit a solved or dead task, so its estimate could not recover"
            )

    def __len__(self) -> int:
        return len(self.base)

    def weight(self, uid: str) -> float:
        if self.registry.saturated(uid, self.min_obs):
            return float(self.floor)
        p = self.registry.pass_rate(uid)
        return max(float(self.floor), 4.0 * p * (1.0 - p))

    def batch(self, step: int, n: int) -> tuple[EpisodeTask, ...]:
        """``n`` distinct tasks for ``step``, weighted toward the frontier; deterministic given
        ``(seed, step)`` and the registry. More tasks than the pool falls back to the base
        source's with-replacement draw (a frontier cannot be distinct past the pool size)."""
        if type(step) is not int or step < 0 or type(n) is not int or n < 1:
            raise TaskSourceRefusal(
                f"FrontierTaskSource.batch: step={step!r}, n={n!r}: step must be an int >= 0 and "
                f"n an int >= 1"
            )
        pool = self.base.tasks
        if n > len(pool):
            return self.base.batch(step, n)
        rng = random.Random(f"frontier:{self.seed}:{step}")
        keyed = [(rng.random() ** (1.0 / self.weight(task.uid)), i) for i, task in enumerate(pool)]
        keyed.sort(reverse=True)
        return tuple(pool[i] for _, i in keyed[:n])

    def observe(self, trajectories: Iterable[Any]) -> None:
        """Feed every rollout's measured reward into the registry (abstentions are ignored)."""
        for trajectory in trajectories:
            self.registry.update(trajectory.uid, trajectory.reward)

    def stats(self) -> dict[str, float | int]:
        """Registry summary for the ``[agentic-rl:curriculum]`` line."""
        seen = [uid for uid in self.registry.counts]
        rates = [self.registry.pass_rate(uid) for uid in seen]
        return {
            "pool": len(self.base),
            "tasks_seen": len(seen),
            "frontier": sum(1 for p in rates if 0.2 <= p <= 0.8),
            "saturated": sum(1 for uid in seen if self.registry.saturated(uid, self.min_obs)),
            "mean_pass_rate": round(sum(rates) / len(rates), 4) if rates else 0.5,
        }
