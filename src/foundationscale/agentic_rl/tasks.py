"""``TaskSource``: a JSONL file of episode tasks, sampled deterministically per step.

Each line is a JSON object ``{"uid": <str>, "messages": [{"role": <str>,
"content": <str>}, ...], "metadata": {...}}``; ``metadata`` is optional and
defaults to ``{}``. Every row is parsed into a
``harness.base.EpisodeTask`` eagerly, at construction, so a malformed file
fails fast rather than mid-training-step. ``EpisodeTask.session_id`` is set to
the row's own ``uid`` -- a stable per-task placeholder identity -- since the
JSONL schema carries no per-attempt session id; the rollout host assigns the
real per-attempt session id (``f"{uid}#{k}"`` for the k-th of ``group_size``
attempts) itself when it builds each attempt's ``contracts.Trajectory``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from foundationscale.agentic_rl.harness.base import EpisodeTask, HarnessRefusal

__all__ = (
    "TaskSource",
    "TaskSourceRefusal",
)


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


class TaskSourceRefusal(ValueError):
    """A malformed JSONL row, an empty task file, or a bad ``batch`` argument.

    Every row-level message names the source path and the 1-based line number.
    """


def _parse_row(path: str, line_number: int, raw_line: str) -> EpisodeTask | None:
    """Parse one JSONL line into an ``EpisodeTask``, or ``None`` for a blank line."""
    where = f"TaskSource({path!r}) line {line_number}"
    stripped = raw_line.strip()
    if not stripped:
        return None
    try:
        row = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise TaskSourceRefusal(f"{where}: invalid JSON ({exc})") from exc
    if not isinstance(row, dict):
        raise TaskSourceRefusal(f"{where}: is {_described(row)}, not a JSON object")
    if "uid" not in row:
        raise TaskSourceRefusal(f"{where}: is missing key 'uid'")
    uid = row["uid"]
    if "messages" not in row:
        raise TaskSourceRefusal(f"{where}: is missing key 'messages'")
    metadata = row.get("metadata", {})
    try:
        return EpisodeTask(uid=uid, session_id=uid, messages=row["messages"], metadata=metadata)
    except HarnessRefusal as exc:
        raise TaskSourceRefusal(f"{where}: {exc}") from exc


def _load_tasks(path: str) -> tuple[EpisodeTask, ...]:
    tasks: list[EpisodeTask] = []
    seen_uids: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            task = _parse_row(path, line_number, raw_line)
            if task is None:
                continue
            first_seen = seen_uids.get(task.uid)
            if first_seen is not None:
                raise TaskSourceRefusal(
                    f"TaskSource({path!r}) line {line_number}: uid {task.uid!r} was "
                    f"already read at line {first_seen}: each uid must be unique -- a "
                    f"repeated uid would make 'uid' ambiguous as a prompt-group key"
                )
            seen_uids[task.uid] = line_number
            tasks.append(task)
    if not tasks:
        raise TaskSourceRefusal(f"TaskSource({path!r}): contains 0 usable rows")
    return tuple(tasks)


@dataclass(frozen=True)
class TaskSource:
    """Reads ``path`` (a JSONL file) into an immutable pool of ``EpisodeTask``, and
    draws a deterministic ``n``-task sample for each training ``step``.

    ``seed`` plus ``step`` alone determine ``batch(step, n)``'s result (no
    sequential/replay state), so a training run can resume at any step and draw
    the exact same batch it would have drawn the first time through.
    """

    path: str
    seed: int = 0
    _tasks: tuple[EpisodeTask, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        where = "TaskSource"
        if not isinstance(self.path, str) or not self.path:
            raise TaskSourceRefusal(
                f"{where}: field 'path' is {_described(self.path)}: it must be a non-empty str"
            )
        if type(self.seed) is not int:
            raise TaskSourceRefusal(
                f"{where}: field 'seed' is {_described(self.seed)}: it must be a real int "
                f"(bool excluded)"
            )
        object.__setattr__(self, "_tasks", _load_tasks(self.path))

    def __len__(self) -> int:
        return len(self._tasks)

    def batch(self, step: int, n: int) -> tuple[EpisodeTask, ...]:
        """``n`` tasks for ``step``, deterministic given ``(self.seed, step, n)``.

        Samples WITHOUT replacement when ``n <= len(self)``, else WITH replacement
        (there are not enough distinct tasks to avoid it) -- both paths draw from
        the same seeded ``random.Random``, so the choice is a property of the pool
        size, never of the caller.
        """
        where = "TaskSource.batch"
        if type(step) is not int or step < 0:
            raise TaskSourceRefusal(
                f"{where}: parameter 'step' is {_described(step)}: it must be a real int >= 0"
            )
        if type(n) is not int or n < 1:
            raise TaskSourceRefusal(
                f"{where}: parameter 'n' is {_described(n)}: it must be a real int >= 1"
            )
        rng = random.Random(f"{self.seed}:{step}")
        pool_size = len(self._tasks)
        if n <= pool_size:
            indices = rng.sample(range(pool_size), n)
        else:
            indices = [rng.randrange(pool_size) for _ in range(n)]
        return tuple(self._tasks[index] for index in indices)


@dataclass(frozen=True)  # frozen like its parent; the per-task state dict itself is mutated in place
class CurriculumTaskSource(TaskSource):
    """``TaskSource`` with an online learnability curriculum (atlas v2 adoption B; opt-in via ``rollout.curriculum``).

    Zero-variance groups carry no GRPO advantage (the trainer zeroes them), so after each step ``observe`` records,
    per task, whether its group had reward spread and its EMA reward. ``batch`` then draws ``informative_share`` of
    the tasks from those whose last measured group had spread or partial progress, and the rest uniformly
    from the active pool (hard and unseen tasks keep coverage). A task flat for ``quarantine_after`` consecutive
    measured groups is quarantined for ``reprobe_every`` steps, then re-probed. A group with no measured reward
    (infra abstention) is never evidence of difficulty and changes nothing.

    Unlike ``TaskSource``, a batch depends on what was observed, so a resumed run does not replay the same draws.
    """

    informative_share: float = 0.8
    quarantine_after: int = 3
    reprobe_every: int = 20
    _state: dict = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.informative_share <= 1.0 or self.quarantine_after < 1 or self.reprobe_every < 1:
            raise TaskSourceRefusal(
                f"CurriculumTaskSource: informative_share={self.informative_share!r} must be in [0, 1], "
                f"quarantine_after={self.quarantine_after!r} and reprobe_every={self.reprobe_every!r} >= 1"
            )

    def batch(self, step: int, n: int) -> tuple[EpisodeTask, ...]:
        base = super().batch(step, n)  # validates step/n; also the draw when nothing has been observed yet
        if not self._state:
            return base
        rng = random.Random(f"{self.seed}:{step}:curriculum")
        active = [i for i, t in enumerate(self._tasks) if self._state.get(t.uid, {}).get("quarantined_until", -1) <= step]
        if not active:
            active = list(range(len(self._tasks)))
        informative = [i for i in active if self._state.get(self._tasks[i].uid, {}).get("informative")]
        k = min(len(informative), round(self.informative_share * n))
        chosen = rng.sample(informative, k) if k else []
        rest = [i for i in active if i not in set(chosen)]
        need = n - len(chosen)
        chosen += rng.sample(rest, need) if need <= len(rest) else [rng.choice(active) for _ in range(need)]
        return tuple(self._tasks[i] for i in chosen)

    def observe(self, step: int, trajectories) -> dict:
        """Update per-task state from one step's trajectories; returns a small summary for the run log."""
        groups: dict[str, list[float]] = {}
        for t in trajectories:
            if t.reward is not None:
                groups.setdefault(t.uid, []).append(float(t.reward))
        quarantined = 0
        for uid, rewards in groups.items():
            s = self._state.setdefault(uid, {"ema": 0.0, "flat": 0, "quarantined_until": -1})
            mean = sum(rewards) / len(rewards)
            spread = len(rewards) > 1 and max(rewards) > min(rewards)
            s["ema"] = 0.5 * s["ema"] + 0.5 * mean
            s["informative"] = bool(spread or 0.0 < mean < 1.0)
            s["flat"] = 0 if spread else s["flat"] + 1
            if s["flat"] >= self.quarantine_after:
                s["quarantined_until"], s["flat"] = step + 1 + self.reprobe_every, 0
                quarantined += 1
        pool = self._state.values()
        return {"rollout_step": step, "observed": len(groups), "newly_quarantined": quarantined,
                "informative": sum(1 for s in pool if s.get("informative")),
                "quarantined": sum(1 for s in pool if s["quarantined_until"] > step), "seen": len(self._state)}
