"""The declared plan of one LIBERO evaluation: the suite, the tasks, the trials, how episodes start.

A success rate is only comparable to a published number when the episodes behind it were declared
before the first rollout. :class:`EpisodePlan` is that declaration: the LIBERO ``suite``, the
``task_ids`` of it to evaluate, the ``trials_per_task`` per task, the ``seed`` of the run, how an
episode starts (``init``), how many settle steps of the dummy action come first
(``num_steps_wait``) and how many control steps one episode may take (``max_steps``).
:meth:`EpisodePlan.episodes` expands the plan into the ``(task_id, trial_index)`` list the runner
executes, in task-major order, so a re-run of the same plan replays exactly the same episodes.

Nothing here guesses a value. ``init`` has no default: ``"suite_init_states"`` replays the
benchmark's recorded init states, the openpi protocol, while ``"seeded_reset"`` seeds the env with
``seed + trial`` and resets it, the GR00T protocol, and the two start different episodes -- so the
caller declares which one the evaluation runs. The one resolution this module makes is
``max_steps=None``, which becomes the suite's cap in :data:`LIBERO_MAX_STEPS` at construction, so
the budget every episode runs under is an int here and never a choice made mid-rollout.

Every field that is not what its declaration describes is an :class:`EvalPlanError` naming the
offending value: a suite :data:`LIBERO_MAX_STEPS` carries no cap for (the episodes would run under
no declared budget), a ``task_ids`` that is not a tuple of ints, is empty (a success rate over no
episodes), repeats a task (its trials would be counted twice) or holds a negative id (an index that
cannot name a task), a ``trials_per_task`` below 1 (again no episodes), a non-int ``seed``, an
``init`` outside :data:`INIT_MODES`, a negative ``num_steps_wait`` (there are no settle steps
before step 0 to take away) and a ``max_steps`` below 1 (an episode that may never step cannot
succeed). A task id the suite does not carry is not refused here -- this module holds no benchmark
to ask -- it is refused when the runner resolves the task.

This module is pure stdlib: it declares a plan, it does not roll one out, so it needs neither numpy
nor a simulator or an upstream package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeGuard

__all__ = [
    "INIT_MODES",
    "LIBERO_MAX_STEPS",
    "EpisodePlan",
    "EvalPlanError",
]


class EvalPlanError(ValueError):
    """An episode plan declares an evaluation that cannot run the way it declares it."""


LIBERO_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
INIT_MODES: tuple[str, ...] = ("suite_init_states", "seeded_reset")


@dataclass(frozen=True)
class EpisodePlan:
    """One declared evaluation: which episodes run, and how each of them starts.

    ``suite`` is a key of :data:`LIBERO_MAX_STEPS`; ``task_ids`` are the task indices of that suite,
    each evaluated ``trials_per_task`` times; ``seed`` is the base seed of the run. ``init`` is the
    episode-start protocol, one of :data:`INIT_MODES`, with no default: ``"suite_init_states"``
    replays the benchmark's recorded init states (openpi's protocol) and ``"seeded_reset"`` seeds
    the env with ``seed + trial`` and resets it (GR00T's protocol). ``num_steps_wait`` is the number
    of settle steps an episode takes with the dummy action ``[0] * 6 + [-1]`` before the policy
    acts -- 10 under the openpi protocol, 0 under GR00T's -- and ``max_steps`` is the control-step
    budget of one episode, ``None`` taking the suite's cap from :data:`LIBERO_MAX_STEPS`.

    The dataclass is frozen: a plan that changes between episodes is not a declaration. Every field
    is validated at construction and ``max_steps`` is resolved there, so an instance that exists is
    an evaluation that can run; see :meth:`__post_init__` for what is refused and why.
    """

    suite: str
    task_ids: tuple[int, ...]
    trials_per_task: int
    seed: int
    init: Literal["suite_init_states", "seeded_reset"]
    num_steps_wait: int
    max_steps: int | None = None

    def __post_init__(self) -> None:
        """Validate every field, then resolve ``max_steps``, or raise :class:`EvalPlanError`.

        Refuses a ``suite`` that is not a string or that :data:`LIBERO_MAX_STEPS` has no cap for
        (the episodes would run under no declared budget); a ``task_ids`` that is not a tuple, is
        empty (the report would hold a success rate over no episodes), repeats a task id (its
        trials would be counted twice) or holds a non-int or negative id (an index that cannot name
        a task of the suite); a ``trials_per_task`` that is not an int >= 1 (again, no episodes to
        score); a ``seed`` that is not an int (it is added to a trial index and handed to
        ``env.seed``); an ``init`` outside :data:`INIT_MODES` (the two protocols start different
        episodes and there is no default to fall back on); a ``num_steps_wait`` that is not an int
        >= 0 (there are no settle steps before step 0 to take away); and a ``max_steps`` that is
        neither ``None`` nor an int >= 1 (an episode that may never step cannot succeed). Every
        refusal names the offending value.
        """
        if not isinstance(self.suite, str):
            raise EvalPlanError(
                f"plan suite is {self.suite!r}, a {type(self.suite).__name__}; expected one of "
                f"{sorted(LIBERO_MAX_STEPS)}"
            )
        if self.suite not in LIBERO_MAX_STEPS:
            raise EvalPlanError(
                f"plan suite {self.suite!r} is unknown; expected one of {sorted(LIBERO_MAX_STEPS)}"
            )
        if not isinstance(self.task_ids, tuple):
            raise EvalPlanError(
                f"plan task_ids is {self.task_ids!r}, a {type(self.task_ids).__name__}; expected a "
                "tuple of task ids"
            )
        if not self.task_ids:
            raise EvalPlanError(
                f"plan task_ids is {self.task_ids!r}; expected a non-empty tuple of task ids -- an "
                "empty plan would report a success rate over no episodes"
            )
        seen: set[int] = set()
        for position, task_id in enumerate(self.task_ids):
            if not _is_int(task_id):
                raise EvalPlanError(
                    f"plan task_ids[{position}] is {task_id!r}, a {type(task_id).__name__}; "
                    "expected an int task id >= 0"
                )
            if task_id < 0:
                raise EvalPlanError(
                    f"plan task_ids[{position}] is {task_id!r}; expected a task id >= 0 -- a "
                    "negative index cannot name a task of the suite"
                )
            if task_id in seen:
                raise EvalPlanError(
                    f"plan task_ids[{position}] is {task_id!r}, a duplicate of an earlier id in "
                    f"{self.task_ids!r}; expected unique task ids, or that task's trials would be "
                    "counted twice in the report"
                )
            seen.add(task_id)
        if not _is_int(self.trials_per_task):
            raise EvalPlanError(
                f"plan trials_per_task is {self.trials_per_task!r}, a "
                f"{type(self.trials_per_task).__name__}; expected an int >= 1"
            )
        if self.trials_per_task < 1:
            raise EvalPlanError(
                f"plan trials_per_task is {self.trials_per_task!r}; expected an int >= 1 -- zero "
                "trials would report a success rate over no episodes"
            )
        if not _is_int(self.seed):
            raise EvalPlanError(
                f"plan seed is {self.seed!r}, a {type(self.seed).__name__}; expected an int -- it "
                "is added to a trial index and handed to env.seed"
            )
        if self.init not in INIT_MODES:
            raise EvalPlanError(
                f"plan init is {self.init!r}; expected one of {list(INIT_MODES)} -- there is no "
                "default, 'suite_init_states' replays the benchmark's init states and "
                "'seeded_reset' seeds and resets the env, and the two start different episodes"
            )
        if not _is_int(self.num_steps_wait):
            raise EvalPlanError(
                f"plan num_steps_wait is {self.num_steps_wait!r}, a "
                f"{type(self.num_steps_wait).__name__}; expected an int >= 0"
            )
        if self.num_steps_wait < 0:
            raise EvalPlanError(
                f"plan num_steps_wait is {self.num_steps_wait!r}; expected an int >= 0 -- "
                "there are no settle steps before step 0 to take away"
            )
        budget = self.max_steps
        if budget is None:
            object.__setattr__(self, "max_steps", LIBERO_MAX_STEPS[self.suite])
        else:
            if not _is_int(budget):
                raise EvalPlanError(
                    f"plan max_steps is {budget!r}, a {type(budget).__name__}; expected an int "
                    ">= 1 or None (None takes the suite's cap from LIBERO_MAX_STEPS)"
                )
            if budget < 1:
                raise EvalPlanError(
                    f"plan max_steps is {budget!r}; expected an int >= 1 or None -- an episode "
                    "that may never step cannot succeed"
                )

    def episodes(self) -> list[tuple[int, int]]:
        """Every ``(task_id, trial_index)`` of this plan, in task-major order.

        Task-major order runs every trial of ``task_ids[0]`` before the first trial of
        ``task_ids[1]``, with trial indices ``0 .. trials_per_task - 1`` inside a task. This is the
        order the runner executes and the report lists, so the same plan always replays the same
        episodes in the same order.
        """
        episodes: list[tuple[int, int]] = []
        for task_id in self.task_ids:
            for trial in range(self.trials_per_task):
                episodes.append((task_id, trial))
        return episodes


# -- fail-closed helpers --------------------------------------------------------------------


def _is_int(value: Any) -> TypeGuard[int]:
    """``True`` for a real int: ``bool`` is refused, it is not a task id or a step count."""
    return isinstance(value, int) and not isinstance(value, bool)
