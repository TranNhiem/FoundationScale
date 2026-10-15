"""FS-owned LIBERO evaluation: the episode plan, the rollout loop, the success flag, the record.

Evaluation contract #4 of plan v2. :func:`run_libero` walks every ``(task_id, trial)``
:class:`~foundationscale.vla.eval.plan.EpisodePlan` declares, in task-major order, drives one
:class:`~foundationscale.vla.eval.protocol.PolicyAdapter` through LIBERO's
``OffScreenRenderEnv``, and writes exactly one :class:`EpisodeRecord` per episode. The episodes
that fail are written too: an exception inside an episode and an action chunk the policy cannot
legally return are both recorded with ``error`` and ``success=False``, and both still count in
the denominator. A harness that dropped them would publish a success rate over only the episodes
that happened to work.

Nothing here guesses a value. The suite, the task ids, the trials, the seed, the init mode, the
settle steps and the step budget all come from the plan -- ``make_env`` overrides how the env is
constructed (tests inject a fake one), never what is run. The settle action is exactly
``[0] * 6 + [-1]``, the openpi/GR00T convention, and the loop runs ``max_steps +
num_steps_wait`` env steps so the settle steps never eat the policy's budget. ``init`` picks
between the two protocols and has no default: ``"suite_init_states"`` restores the states the
suite stores (the openpi protocol), ``"seeded_reset"`` re-seeds and resets (the GR00T protocol).

Refused, and why. Before anything runs, a ``plan`` that is not an :class:`EpisodePlan`, a
``policy`` that is not a ``PolicyAdapter` (no name, no ``replan_steps >= 1``, no
``reset``/``act``), a ``resolution`` that is not an int >= 1, and a ``make_env`` that is not
callable are :class:`LiberoRunnerError`. Inside an episode an action chunk that is not
``[k >= replan_steps, 7]`` is refused rather than truncated or padded -- a short chunk would
silently change the replan cadence the plan declares and a wrong width would step the env with a
malformed action -- and the refusal is recorded on that episode. A suite the LIBERO benchmark
dict does not carry, a task the benchmark cannot answer for, a task with no ``language``, an
init-state list that is empty, and an ``env.step`` result that is not LIBERO's
``(obs, reward, done, info)`` are all refused naming what came back: guessing which slot holds
``done`` would score the episode on a reward or an info dict. :func:`within_published` refuses a
``published_trials`` below 1 and a ``published_successes`` outside ``[0, published_trials]``: a
rate with no denominator, or a numerator larger than its denominator, is not a measurement to
compare against. :meth:`EvalReport.rate` and :meth:`EvalReport.wilson_ci` refuse a report with
no episodes, because a rate over zero trials is not 0.0 (that would claim a measured failure),
it is undefined.

``libero`` is imported function-locally, inside the helpers that need it: the core install has
none of it, and this module has to import on a machine that only runs the tests.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

from foundationscale.vla.eval.plan import LIBERO_MAX_STEPS, EpisodePlan
from foundationscale.vla.eval.protocol import PolicyAdapter

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "EpisodeRecord",
    "EvalReport",
    "LiberoEnv",
    "LiberoRunnerError",
    "run_libero",
    "within_published",
]

_ACTION_DIM = 7
# The settle action openpi and GR00T both step while the scene comes to rest: no motion, gripper
# open. It is copied per step, never mutated in place.
_DUMMY_ACTION: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


class LiberoRunnerError(ValueError):
    """A run, an episode, or the published numbers it is compared against, is not valid."""


class LiberoEnv(Protocol):
    """The slice of LIBERO's ``OffScreenRenderEnv`` the rollout loop drives.

    ``step`` answers LIBERO's ``(obs, reward, done, info)``. A gymnasium-style
    ``(obs, reward, terminated, truncated, info)`` is read too, as
    ``done = terminated or truncated``; anything else is refused naming what came back.
    """

    def seed(self, seed: int) -> Any:
        """Seed the resets that follow; called before every ``seeded_reset`` episode."""
        ...

    def reset(self) -> Mapping[str, Any]:
        """Reset the scene and return the raw observation the policy is shown."""
        ...

    def set_init_state(self, state: Any) -> Mapping[str, Any]:
        """Restore a stored init ``state`` and return the raw observation the policy is shown."""
        ...

    def step(self, action: Sequence[float]) -> Any:
        """Apply one 7-D ``action`` and answer the episode result."""
        ...

    def close(self) -> None:
        """Release the off-screen renderer."""
        ...


@dataclass(frozen=True)
class EpisodeRecord:
    """One finished episode: which one it was, whether it succeeded, and why it did not.

    ``steps`` counts the ``env.step`` calls the episode made, settle steps included, and
    ``inferences`` the ``policy.act`` calls that answered with a chunk -- a chunk that was then
    refused still counts, an ``act`` that raised does not. ``error`` is ``None`` for an episode
    that ran to a verdict and ``str(exc)`` for one that failed, and a failed episode is always
    ``success=False``.
    """

    task_id: int
    trial: int
    success: bool
    steps: int
    inferences: int
    wall_s: float
    error: str | None


@dataclass(frozen=True)
class EvalReport:
    """Every episode of one plan under one policy, and the rates that fall out of it.

    ``episodes`` is in the order :meth:`~foundationscale.vla.eval.plan.EpisodePlan.episodes`
    declares (task-major), one record per episode including the failures, so :meth:`trials` is
    the plan's episode count however badly the run went.
    """

    plan: EpisodePlan
    policy: str
    episodes: tuple[EpisodeRecord, ...]

    def successes(self) -> int:
        """The episodes that ended with ``done``: the numerator of the success rate."""
        return sum(1 for episode in self.episodes if episode.success)

    def trials(self) -> int:
        """The episodes the report holds, failures included: the denominator never shrinks."""
        return len(self.episodes)

    def rate(self) -> float:
        """``successes() / trials()``, the measured success rate of this run.

        Refuses a report with no episodes: a rate over zero trials is not 0.0 (that would claim a
        measured failure) and not 1.0, it is undefined.
        """
        trials = self.trials()
        if trials < 1:
            raise LiberoRunnerError(
                f"the report holds {trials} episode(s); a success rate needs at least one trial, "
                "so an empty report has no rate to report"
            )
        return self.successes() / trials

    def per_task(self) -> dict[int, tuple[int, int]]:
        """``task_id -> (successes, trials)``, in the order the tasks were run."""
        counts: dict[int, tuple[int, int]] = {}
        for episode in self.episodes:
            wins, trials = counts.get(episode.task_id, (0, 0))
            counts[episode.task_id] = (wins + int(episode.success), trials + 1)
        return counts

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        """The Wilson score interval of the success rate at ``z`` standard deviations.

        ``(low, high)`` bounds the binomial rate without the normal approximation's habit of
        running below 0 or above 1 at small ``trials`` -- which is exactly the regime a LIBERO
        evaluation is in. ``z`` is refused unless it is a finite float above 0, and a report with
        no episodes is refused for the same reason as :meth:`rate`.
        """
        trials = self.trials()
        if trials < 1:
            raise LiberoRunnerError(
                f"the report holds {trials} episode(s); a Wilson interval needs at least one "
                "trial to bound"
            )
        if not _is_number(z) or not math.isfinite(z) or z <= 0.0:
            raise LiberoRunnerError(
                f"z is {z!r}; expected a finite float > 0 (1.96 is the two-sided 95% normal "
                "quantile), the width of the interval to report"
            )
        p = self.successes() / trials
        z2 = z * z
        denominator = 1.0 + z2 / trials
        centre = (p + z2 / (2.0 * trials)) / denominator
        half = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
        return (max(0.0, centre - half / denominator), min(1.0, centre + half / denominator))

    def to_json(self) -> dict[str, Any]:
        """The result record: the plan, the policy, every episode, and the measured rates.

        The plan is written out field by field rather than repr'd, so the record names the suite,
        the seed, the init mode and the step budget it was produced under. An empty report is
        refused here too, through :meth:`rate`: it has no rates to record.
        """
        return {
            "plan": {
                "suite": self.plan.suite,
                "task_ids": list(self.plan.task_ids),
                "trials_per_task": self.plan.trials_per_task,
                "seed": self.plan.seed,
                "init": self.plan.init,
                "num_steps_wait": self.plan.num_steps_wait,
                "max_steps": self.plan.max_steps,
            },
            "policy": self.policy,
            "episodes": [
                {
                    "task_id": episode.task_id,
                    "trial": episode.trial,
                    "success": episode.success,
                    "steps": episode.steps,
                    "inferences": episode.inferences,
                    "wall_s": episode.wall_s,
                    "error": episode.error,
                }
                for episode in self.episodes
            ],
            "successes": self.successes(),
            "trials": self.trials(),
            "rate": self.rate(),
            "wilson_ci": list(self.wilson_ci()),
        }


def run_libero(
    plan: EpisodePlan,
    policy: PolicyAdapter,
    *,
    make_env: Callable[[Any, int, int], LiberoEnv] | None = None,
    resolution: int = 256,
) -> EvalReport:
    """Every episode of ``plan``, rolled out through ``policy``, as one :class:`EvalReport`.

    The benchmark is ``libero.libero.benchmark.get_benchmark_dict()[plan.suite]()``, the task is
    ``benchmark.get_task(task_id)``, and the env is ``OffScreenRenderEnv`` over the task's BDDL
    file at ``resolution`` x ``resolution``, seeded with ``plan.seed`` -- unless ``make_env``
    overrides that construction as ``make_env(task, resolution, plan.seed)``, which is how the
    tests inject a fake env. ``make_env`` overrides construction only: the suite, the tasks, the
    trials, the seed, the init mode and the budgets always come from the plan.

    Each episode calls ``policy.reset()``, resets the env, and then initialises it the way
    ``plan.init`` declares: ``"suite_init_states"`` restores
    ``init_states[trial % len(init_states)]`` with ``env.set_init_state`` (the openpi protocol),
    ``"seeded_reset"`` calls ``env.seed(plan.seed + trial)`` and resets again (the GR00T
    protocol). The loop then runs ``max_steps + num_steps_wait`` env steps: the first
    ``num_steps_wait`` step the dummy action ``[0] * 6 + [-1]``, the rest step
    ``policy.act(obs, task.language)`` in chunks of ``policy.replan_steps`` actions, and
    ``done`` ends the episode as a success.

    Refuses a ``plan`` that is not an :class:`EpisodePlan`, a ``policy`` that is not a
    :class:`~foundationscale.vla.eval.protocol.PolicyAdapter`, a ``resolution`` that is not an
    int >= 1, and a ``make_env`` that is neither callable nor ``None``, naming each value. Inside
    an episode a chunk that is not ``[k >= replan_steps, 7]`` is refused (never truncated,
    padded or skipped) and recorded on that episode, and any exception is recorded the same way:
    the episode fails and still counts, so the denominator never shrinks. A task whose env cannot
    be built fails all of that task's episodes rather than vanishing from the report, and
    ``env.close()`` runs per task whatever happened.
    """
    if not isinstance(plan, EpisodePlan):
        raise LiberoRunnerError(
            f"plan is {plan!r} (a {type(plan).__name__}); expected an EpisodePlan, the declared "
            "suite, task ids, trials, seed, init mode and step budget"
        )
    if not _is_adapter(policy):
        raise LiberoRunnerError(
            f"policy is {policy!r} (a {type(policy).__name__}); expected a PolicyAdapter with a "
            "non-empty name, replan_steps >= 1 and callable reset()/act()"
        )
    if not _is_int(resolution) or resolution < 1:
        raise LiberoRunnerError(
            f"resolution is {resolution!r}; expected an int >= 1, the square camera size in "
            "pixels the frames are rendered at"
        )
    if make_env is not None and not callable(make_env):
        raise LiberoRunnerError(
            f"make_env is {make_env!r} (a {type(make_env).__name__}); expected a callable "
            "(task, resolution, seed) -> env, or None to build LIBERO's OffScreenRenderEnv"
        )

    max_steps = _max_steps(plan)
    benchmark = _benchmark(plan.suite)
    records: list[EpisodeRecord] = []
    for task_id in plan.task_ids:
        records.extend(
            _run_task(
                plan,
                policy,
                task_id,
                benchmark=benchmark,
                make_env=make_env,
                resolution=resolution,
                max_steps=max_steps,
            )
        )
    return EvalReport(plan=plan, policy=policy.name, episodes=tuple(records))


def within_published(
    report: EvalReport,
    published_successes: int,
    published_trials: int,
    z: float = 1.96,
) -> bool:
    """``True`` when ``report``'s rate is within the two-sample binomial margin of the published.

    The margin is ``z * sqrt(p(1-p)/n_published + p(1-p)/n_measured)`` with ``p`` the published
    rate -- the same rule the VLA registry fixes its tolerances with
    (:func:`foundationscale.upstream.vla_models.binomial_tolerance`), so the harness and the
    registry never disagree on whether a run reproduces. Both runs are samples: a one-sample
    test that only asks whether the published rate falls inside OUR interval ignores the
    published run's own noise and rejects honest reproductions (measured on GB200: GR00T 190/200
    against 195/200 fails that test and passes this one). A degenerate published rate (0 or 1)
    has no binomial spread, so the pooled rate of both runs is used for ``p`` instead. The
    comparison is inclusive.

    Refuses a ``report`` that is not an :class:`EvalReport`, a ``published_trials`` that is not
    an int >= 1, and a ``published_successes`` that is not an int in ``[0, published_trials]``,
    naming each value: a rate with no denominator, or a numerator larger than its denominator,
    is not a measurement to compare against. ``z`` is refused by
    :meth:`EvalReport.wilson_ci`.
    """
    if not isinstance(report, EvalReport):
        raise LiberoRunnerError(
            f"report is {report!r} (a {type(report).__name__}); expected an EvalReport, the run "
            "the published numbers are compared against"
        )
    if not _is_int(published_trials) or published_trials < 1:
        raise LiberoRunnerError(
            f"published_trials is {published_trials!r}; expected an int >= 1, the published "
            "denominator"
        )
    if not _is_int(published_successes):
        raise LiberoRunnerError(
            f"published_successes is {published_successes!r}; expected an int in "
            f"[0, {published_trials}] (published_trials)"
        )
    if not 0 <= published_successes <= published_trials:
        raise LiberoRunnerError(
            f"published_successes is {published_successes}; expected an int in "
            f"[0, {published_trials}] (published_trials), a count cannot exceed its denominator"
        )
    if not _is_number(z) or not math.isfinite(z) or z <= 0.0:
        raise LiberoRunnerError(f"z is {z!r}; expected a finite float > 0")
    measured_trials = report.trials()
    if measured_trials < 1:
        raise LiberoRunnerError("the report holds no episodes; there is no rate to compare")
    published = published_successes / published_trials
    p = published
    if p in (0.0, 1.0):
        p = (published_successes + report.successes()) / (published_trials + measured_trials)
    margin = z * math.sqrt(p * (1.0 - p) / published_trials + p * (1.0 - p) / measured_trials)
    return abs(report.rate() - published) <= margin + 1e-12


# -- the rollout, one task and one episode at a time -------------------------------------


def _run_task(
    plan: EpisodePlan,
    policy: PolicyAdapter,
    task_id: int,
    *,
    benchmark: Any,
    make_env: Callable[[Any, int, int], LiberoEnv] | None,
    resolution: int,
    max_steps: int,
) -> list[EpisodeRecord]:
    """Every episode of ``task_id``, with the task's env built once and closed at the end.

    A failure while setting the task up -- a task id the benchmark cannot answer for, an env that
    cannot be constructed, an init-state list the suite does not carry -- is recorded as a
    failure for each episode of the task that has no record yet, rather than raised: the
    denominator never shrinks, so a broken task cannot quietly disappear from the rate and make
    it look better. ``env.close()`` runs in ``finally``, whatever happened.
    """
    started = time.perf_counter()
    env: LiberoEnv | None = None
    records: list[EpisodeRecord] = []
    try:
        task = _task(benchmark, task_id)
        env = (
            make_env(task, resolution, plan.seed)
            if make_env is not None
            else _make_env(task, resolution, plan.seed)
        )
        init_states = _init_states(benchmark, task_id) if plan.init == "suite_init_states" else None
        for trial in range(plan.trials_per_task):
            records.append(
                _run_episode(
                    plan,
                    policy,
                    task,
                    env,
                    task_id,
                    trial,
                    init_states,
                    max_steps=max_steps,
                )
            )
    except Exception as exc:  # noqa: BLE001
        wall_s = time.perf_counter() - started
        for trial in range(len(records), plan.trials_per_task):
            records.append(
                EpisodeRecord(
                    task_id=task_id,
                    trial=trial,
                    success=False,
                    steps=0,
                    inferences=0,
                    wall_s=wall_s,
                    error=str(exc),
                )
            )
    finally:
        if env is not None:
            env.close()
    return records


def _run_episode(
    plan: EpisodePlan,
    policy: PolicyAdapter,
    task: Any,
    env: LiberoEnv,
    task_id: int,
    trial: int,
    init_states: Any,
    max_steps: int,
) -> EpisodeRecord:
    """One rollout of ``task_id``/``trial``, recorded as one :class:`EpisodeRecord` always.

    The episode initialises the way ``plan.init`` declares, settles for ``num_steps_wait`` steps
    on the dummy action, then steps ``policy.act`` chunks of ``replan_steps`` actions until
    ``done`` or the ``max_steps + num_steps_wait`` budget. Any exception -- a policy that raises,
    a chunk that is not ``[k >= replan_steps, 7]``, an env that answers nonsense -- is recorded
    as ``error=str(exc)`` with ``success=False``, at the step and inference counts the episode
    had reached. Nothing is swallowed and nothing is skipped: the episode still counts.
    """
    started = time.perf_counter()
    steps = 0
    inferences = 0
    success = False
    try:
        policy.reset()
        obs = env.reset()
        if plan.init == "suite_init_states":
            obs = env.set_init_state(_init_state(init_states, trial))
        else:
            env.seed(plan.seed + trial)
            obs = env.reset()
        queue: list[list[float]] = []
        for step in range(max_steps + plan.num_steps_wait):
            if step < plan.num_steps_wait:
                action = list(_DUMMY_ACTION)
            else:
                if not queue:
                    raw = policy.act(obs, _task_text(task))
                    inferences += 1
                    queue = _queue(_checked_chunk(raw, policy), policy.replan_steps)
                action = queue.pop(0)
            obs, succeeded, ended = _step(env, action)
            steps += 1
            if ended:
                success = succeeded
                break
    except Exception as exc:  # noqa: BLE001
        return EpisodeRecord(
            task_id=task_id,
            trial=trial,
            success=False,
            steps=steps,
            inferences=inferences,
            wall_s=time.perf_counter() - started,
            error=str(exc),
        )
    return EpisodeRecord(
        task_id=task_id,
        trial=trial,
        success=success,
        steps=steps,
        inferences=inferences,
        wall_s=time.perf_counter() - started,
        error=None,
    )


def _queue(chunk: np.ndarray, replan_steps: int) -> list[list[float]]:
    """The first ``replan_steps`` rows of ``chunk`` as plain 7-float actions, in order.

    Only ``replan_steps`` rows are taken: the rest of the chunk is the policy's plan for later,
    and stepping it would execute actions this episode's replan cadence never asked for.
    """
    return [[float(value) for value in row] for row in chunk[:replan_steps]]


def _step(env: LiberoEnv, action: Sequence[float]) -> tuple[Mapping[str, Any], bool, bool]:
    """``(obs, success, ended)`` from one ``env.step(action)``.

    LIBERO answers ``(obs, reward, done, info)`` and its ``done`` means the task succeeded. A
    gymnasium-style ``(obs, reward, terminated, truncated, info)`` succeeds only on ``terminated``:
    ``truncated`` ends the episode as a FAILURE -- scoring a time-limit cut as success would inflate
    every rate. Anything else is refused naming what came back and how many items it carried:
    guessing which slot holds ``done`` would score the episode on a reward or an info dict.
    """
    result = env.step(action)
    items = _step_items(result)
    if len(items) == 4:
        obs, _reward, done, _info = items
        return obs, bool(done), bool(done)
    if len(items) == 5:
        obs, _reward, terminated, truncated, _info = items
        return obs, bool(terminated), bool(terminated) or bool(truncated)
    raise LiberoRunnerError(
        f"env.step returned {result!r}, a {len(items)}-item {type(result).__name__}; "
        "expected (obs, reward, done, info) or "
        "(obs, reward, terminated, truncated, info)"
    )


# -- fail-closed helpers -----------------------------------------------------------------


def _checked_chunk(chunk: Any, policy: PolicyAdapter) -> np.ndarray:
    """``chunk`` as a ``[k, 7]`` numeric array, or a refusal naming its shape and dtype.

    Refuses anything that is not ``[k >= replan_steps, 7]``: a shorter chunk would silently
    change the replan cadence the plan declares (the episode would re-infer early), a wrong width
    would step the env with a malformed action, and a non-numeric chunk cannot be stepped at all.
    The chunk is never truncated to fit, never padded, and never skipped.
    """
    import numpy as np  # noqa: PLC0415

    try:
        array = np.asarray(chunk)
    except (TypeError, ValueError) as exc:
        raise LiberoRunnerError(
            f"policy {policy.name!r} action chunk of {type(chunk).__name__} is not an array "
            f"({exc}); expected [k, {_ACTION_DIM}]"
        ) from exc
    if array.ndim != 2 or array.shape[1] != _ACTION_DIM:
        raise LiberoRunnerError(
            f"policy {policy.name!r} returned an action chunk of shape {array.shape}; expected "
            f"[k, {_ACTION_DIM}] (k >= replan_steps {policy.replan_steps}), one env-ready action "
            "per row"
        )
    if not np.issubdtype(array.dtype, np.integer) and not np.issubdtype(array.dtype, np.floating):
        raise LiberoRunnerError(
            f"policy {policy.name!r} returned an action chunk of dtype {array.dtype}; expected an "
            "integer or floating dtype the env can step with"
        )
    if array.shape[0] < policy.replan_steps:
        raise LiberoRunnerError(
            f"policy {policy.name!r} returned an action chunk of {array.shape[0]} step(s); "
            f"expected at least replan_steps {policy.replan_steps}, the actions one inference "
            "executes"
        )
    return array


def _step_items(result: Any) -> tuple[Any, ...]:
    """``env.step``'s answer as an ordered tuple of items, or a refusal naming what came back."""
    if isinstance(result, (str, bytes, bytearray, Mapping)):
        raise LiberoRunnerError(
            f"env.step returned {result!r} (a {type(result).__name__}); expected "
            "(obs, reward, done, info), an ordered result"
        )
    try:
        return tuple(result)
    except TypeError as exc:
        raise LiberoRunnerError(
            f"env.step returned {result!r} (a {type(result).__name__}); expected "
            f"(obs, reward, done, info), an ordered result ({exc})"
        ) from exc


def _task_text(task: Any) -> str:
    """The instruction ``task.language``, the prompt the policy is asked.

    Refuses a task that carries no non-empty instruction naming what it does carry: the prompt is
    the task, and an empty one would ask the policy for nothing in particular.
    """
    text = getattr(task, "language", None)
    if not isinstance(text, str) or not text:
        raise LiberoRunnerError(
            f"task carries language {text!r}; expected the non-empty instruction a LIBERO task "
            "declares, the prompt the policy is asked"
        )
    return text


def _init_state(init_states: Any, trial: int) -> Any:
    """The init state ``trial`` restores: ``init_states[trial % len(init_states)]``.

    Refuses an init-state list that is not sized or holds no state, naming what came back: with
    no state to restore there is no episode to run, and inventing one (or skipping the episode)
    would evaluate something other than the plan declares.
    """
    try:
        count = len(init_states)
    except TypeError as exc:
        raise LiberoRunnerError(
            f"the suite's init states are {init_states!r} (a {type(init_states).__name__}); "
            "expected a sized list of states, one per stored reset"
        ) from exc
    if count < 1:
        raise LiberoRunnerError(
            f"the suite's init states hold {count} state(s); expected at least one, the states "
            "'suite_init_states' episodes restore in trial order"
        )
    return init_states[trial % count]


def _init_states(benchmark: Any, task_id: int) -> Any:
    """The init states the suite stores for ``task_id``, via ``get_task_init_states(task_id)``.

    LIBERO's benchmark takes the task's INDEX here, not the task object (measured on GB200: passing
    the object raises ``list indices must be integers``).

    Refuses a benchmark that cannot answer naming what it is: ``"suite_init_states"`` is the
    openpi protocol, which restores exactly the states the suite stores, so there is nothing to
    restore without them.
    """
    getter = getattr(benchmark, "get_task_init_states", None)
    if not callable(getter):
        raise LiberoRunnerError(
            f"benchmark {benchmark!r} carries no get_task_init_states(task_id); the "
            "'suite_init_states' init mode restores the states the suite stores, so there is "
            "nothing to restore without them"
        )
    return getter(task_id)


def _task(benchmark: Any, task_id: int) -> Any:
    """``benchmark.get_task(task_id)``, the task whose BDDL file and instruction are evaluated.

    Refuses a benchmark with no ``get_task``, and a ``task_id`` the benchmark cannot answer for,
    naming the id: a task is its BDDL file and its instruction, so a wrong one is a different
    episode than the plan declares.
    """
    getter = getattr(benchmark, "get_task", None)
    if not callable(getter):
        raise LiberoRunnerError(
            f"benchmark {benchmark!r} carries no get_task(task_id); expected the LIBERO "
            "benchmark API, the source of each task's BDDL file and instruction"
        )
    try:
        return getter(task_id)
    except Exception as exc:  # noqa: BLE001
        raise LiberoRunnerError(
            f"the suite carries no task {task_id} ({exc}); expected a task id the benchmark "
            "declares"
        ) from exc


def _benchmark(suite: str) -> Any:
    """The ``libero`` benchmark object behind ``suite``, built from the benchmark dict.

    ``libero`` is imported here, function-locally: the core install has none of it, and a machine
    that only runs the tests still has to import this module. A suite the dict does not carry is
    refused naming the suites it does -- a near-name would evaluate a different benchmark than
    the plan declares -- and an unimportable ``libero`` is refused naming the import error rather
    than silently reaching for another backend.
    """
    try:
        from libero.libero.benchmark import (  # noqa: PLC0415
            get_benchmark_dict,
        )
    except ImportError as exc:
        raise LiberoRunnerError(
            f"suite {suite!r} cannot be opened: 'libero' is not importable ({exc}); install it "
            "to run a LIBERO evaluation"
        ) from exc
    benchmarks = get_benchmark_dict()
    if suite not in benchmarks:
        raise LiberoRunnerError(
            f"suite {suite!r} is not one of the LIBERO benchmarks {sorted(benchmarks)}; expected "
            "a suite the benchmark dict carries"
        )
    return benchmarks[suite]()


def _make_env(task: Any, resolution: int, seed: int) -> LiberoEnv:
    """LIBERO's ``OffScreenRenderEnv`` for ``task``, at ``resolution`` and seeded with ``seed``.

    ``libero`` is imported here, function-locally: the core install has none of it. The scene
    file is resolved the way LIBERO's own task objects describe it -- ``problem_folder`` and
    ``bddl_file`` under ``get_libero_path("bddl_files")`` (as openpi's LIBERO client does) -- and a
    task missing either is refused naming what it carries: there is no scene to render without it.
    """
    from pathlib import Path  # noqa: PLC0415

    from libero.libero import get_libero_path  # noqa: PLC0415
    from libero.libero.envs import (  # noqa: PLC0415
        OffScreenRenderEnv,
    )

    folder = getattr(task, "problem_folder", None)
    bddl = getattr(task, "bddl_file", None)
    if not isinstance(folder, str) or not folder or not isinstance(bddl, str) or not bddl:
        raise LiberoRunnerError(
            f"task {getattr(task, 'language', None)!r} carries problem_folder {folder!r} and "
            f"bddl_file {bddl!r}; expected both, the BDDL scene OffScreenRenderEnv renders"
        )
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / folder / bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _max_steps(plan: EpisodePlan) -> int:
    """The policy step budget of one episode: ``plan.max_steps``, or the suite's published one.

    ``None`` means the suite's entry in :data:`~foundationscale.vla.eval.plan.LIBERO_MAX_STEPS`;
    anything else is refused unless it is an int >= 1, and a suite that constant does not carry
    is refused naming the suites it does rather than falling back on a guessed budget.
    """
    if plan.max_steps is not None:
        if not _is_int(plan.max_steps) or plan.max_steps < 1:
            raise LiberoRunnerError(
                f"plan max_steps is {plan.max_steps!r}; expected an int >= 1, or None for "
                f"LIBERO_MAX_STEPS[{plan.suite!r}]"
            )
        return int(plan.max_steps)
    budget = LIBERO_MAX_STEPS.get(plan.suite)
    if budget is None:
        raise LiberoRunnerError(
            f"plan suite {plan.suite!r} carries no published budget in LIBERO_MAX_STEPS "
            f"{sorted(LIBERO_MAX_STEPS)}; max_steps is None, so there is no step budget to run"
        )
    return budget


def _is_int(value: Any) -> TypeGuard[int]:
    """``True`` for a real int: ``bool`` is refused, it is not a count or a step budget."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> TypeGuard[float]:
    """``True`` for a real number: ``bool`` is refused, it is not a normal quantile."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_adapter(value: Any) -> TypeGuard[PolicyAdapter]:
    """``True`` for something that speaks :class:`PolicyAdapter`, checked field by field.

    A name, a ``replan_steps`` of at least 1 and callable ``reset``/``act`` are what the rollout
    loop uses, so each is checked rather than assumed: a policy missing one would fail halfway
    through an episode and lose the episodes before it.
    """
    replan_steps = getattr(value, "replan_steps", None)
    return (
        isinstance(getattr(value, "name", None), str)
        and bool(getattr(value, "name", None))
        and _is_int(replan_steps)
        and replan_steps >= 1
        and callable(getattr(value, "reset", None))
        and callable(getattr(value, "act", None))
    )
