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
