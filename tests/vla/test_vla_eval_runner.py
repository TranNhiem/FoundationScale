"""Tests for the FS-owned LIBERO evaluation harness: the plan, the rollout loop and the report.

CPU-only and dependency-free: no ``libero``, ``robosuite`` or upstream package is installed or
imported. The runner reaches LIBERO through ``run_libero(make_env=...)`` (a fake env) and through
the module-level benchmark lookup (a fake benchmark, monkeypatched where the runner asks for it),
and the policy is a fake :class:`PolicyAdapter`. Covered: plan refusals, episode order, the dummy
settle steps, the replan cadence and inference count, success and the ``max_steps`` budget, short
chunks and crashed episodes counted as failures, both init protocols, the Wilson interval against
a hand-computed value, and ``within_published`` (two-sample rule) both ways.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import FrozenInstanceError

import pytest

from foundationscale.vla.eval import libero_runner
from foundationscale.vla.eval.libero_runner import (
    EpisodeRecord,
    EvalReport,
    LiberoRunnerError,
    run_libero,
    within_published,
)
from foundationscale.vla.eval.plan import INIT_MODES, LIBERO_MAX_STEPS, EpisodePlan, EvalPlanError
from foundationscale.vla.eval.protocol import PolicyAdapter

DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def make_chunk(rows: int, width: int = 7) -> list[list[float]]:
    """A distinguishable ``[rows, width]`` action chunk: row ``r`` is ``[10r, 10r + 1, ...]``."""
    return [[float(10 * row + column) for column in range(width)] for row in range(rows)]


class FakeEnv:
    """A deterministic stand-in for LIBERO's ``OffScreenRenderEnv``: no renderer, no mujoco."""

    def __init__(
        self,
        *,
        done_at: int | None = None,
        truncated_at: int | None = None,
        step_mode: str = "classic",
        bad_step_at: int | None = None,
        bad_step_value: object = (None, 0.0, False),
        raise_at: int | None = None,
    ) -> None:
        self.done_at = done_at
        self.truncated_at = truncated_at
        self.step_mode = step_mode
        self.bad_step_at = bad_step_at
        self.bad_step_value = bad_step_value
        self.raise_at = raise_at
        self.seeds: list[int] = []
        self.reset_calls = 0
        self.init_states: list[object] = []
        self.actions: list[list[float]] = []
        self.closed = 0
        self.task = None
        self.resolution = None
        self.seed_arg = None

    def seed(self, seed: int) -> tuple[int, ...]:
        self.seeds.append(seed)
        return (seed,)

    def reset(self) -> dict[str, str]:
        self.reset_calls += 1
        return {"obs": f"reset-{self.reset_calls}"}

    def set_init_state(self, state: object) -> dict[str, str]:
        self.init_states.append(state)
        return {"obs": f"state-{state}"}

    def step(self, action: object) -> object:
        self.actions.append([float(value) for value in action])  # type: ignore[union-attr]
        count = len(self.actions)
        if self.raise_at == count:
            raise RuntimeError("the fake renderer fell over")
        if self.bad_step_at == count:
            return self.bad_step_value
        done = self.done_at is not None and count >= self.done_at
        truncated = self.truncated_at is not None and count >= self.truncated_at
        obs = {"obs": f"step-{count}"}
        if self.step_mode == "gymnasium":
            return (obs, 1.0, done, truncated, {})
        return (obs, 1.0, done, {})

    def close(self) -> None:
        self.closed += 1


class FakePolicy:
    """A fake adapter: replays declared chunks, or raises, and counts its calls."""

    def __init__(
        self,
        *,
        name: str = "fake-policy",
        replan_steps: int = 2,
        chunks: list[object] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.name = name
        self.replan_steps = replan_steps
        self._chunks = list(chunks or [])
        self.raises = raises
        self.reset_calls = 0
        self.calls: list[tuple[object, str]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, raw_obs: object, task_text: str) -> object:
        self.calls.append((raw_obs, task_text))
        if self.raises is not None:
            raise self.raises
        return self._chunks[min(len(self.calls) - 1, len(self._chunks) - 1)]


class FakeTask:
    def __init__(
        self, language: str = "pick up the bowl", bddl: str = "bowl.bddl", folder: str = "tasks"
    ) -> None:
        # Real LIBERO Task objects carry problem_folder + bddl_file (not a full path).
        self.language = language
        self.problem_folder = folder
        self.bddl_file = bddl


class FakeBenchmark:
    """A fake ``libero`` benchmark: ``get_task`` and ``get_task_init_states``."""

    def __init__(
        self, tasks: dict[int, FakeTask], init_states: dict[int, list] | None = None
    ) -> None:
        self.tasks = dict(tasks)
        self.init_states = dict(init_states or {})
        self._ids = {id(task): task_id for task_id, task in self.tasks.items()}

    def get_task(self, task_id: int) -> FakeTask:
        return self.tasks[task_id]

    def get_task_init_states(self, task_id: int) -> list:
        # Real LIBERO takes the task INDEX (an int), not the task object.
        assert isinstance(task_id, int) and not isinstance(task_id, bool)
        return self.init_states[task_id]


class FakeHarness:
    """Installs a fake benchmark where the runner looks one up; records the envs it builds."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self.benchmark: object = None
        self.suites: list[str] = []
        self.envs: list[FakeEnv] = []

    def install(self, benchmark: object, env_factory: object = FakeEnv) -> object:
        self.benchmark = benchmark
        self._monkeypatch.setattr(libero_runner, "_benchmark", self._lookup)

        def make_env(task: object, resolution: int, seed: int) -> FakeEnv:
            env = env_factory()  # type: ignore[operator]
            env.task, env.resolution, env.seed_arg = task, resolution, seed
            self.envs.append(env)
            return env

        return make_env

    def _lookup(self, suite: str) -> object:
        self.suites.append(suite)
        return self.benchmark


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> FakeHarness:
    return FakeHarness(monkeypatch)


@pytest.fixture
def fake_libero_envs(monkeypatch: pytest.MonkeyPatch) -> list[FakeEnv]:
    """A fake ``libero.libero.envs`` in ``sys.modules``: the default construction path."""
    built: list[FakeEnv] = []

    class OffScreenRenderEnv(FakeEnv):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(done_at=1)
            self.kwargs = kwargs
            built.append(self)

    module = types.ModuleType("libero.libero.envs")
    module.OffScreenRenderEnv = OffScreenRenderEnv
    libero_libero = types.ModuleType("libero.libero")
    libero_libero.get_libero_path = lambda key: f"/libero/{key}"  # LIBERO's own path resolver
    monkeypatch.setitem(sys.modules, "libero", types.ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", module)
    return built


def build_plan(**overrides: object) -> EpisodePlan:
    fields: dict[str, object] = {
        "suite": "libero_spatial",
        "task_ids": (0,),
        "trials_per_task": 1,
        "seed": 7,
        "init": "seeded_reset",
        "num_steps_wait": 0,
        "max_steps": 4,
    }
    fields.update(overrides)
    return EpisodePlan(**fields)  # type: ignore[arg-type]


def record(
    task_id: int = 0,
    trial: int = 0,
    success: bool = False,
    steps: int = 1,
    inferences: int = 1,
    wall_s: float = 0.25,
    error: str | None = None,
) -> EpisodeRecord:
    return EpisodeRecord(task_id, trial, success, steps, inferences, wall_s, error)


def build_report(*records: EpisodeRecord, declared: EpisodePlan | None = None) -> EvalReport:
    return EvalReport(plan=declared or build_plan(), policy="fake-policy", episodes=tuple(records))


# -- plan -------------------------------------------------------------------------------


class TestEpisodePlan:
    def test_episodes_expand_in_task_major_order(self) -> None:
        declared = build_plan(task_ids=(3, 1), trials_per_task=3)
        assert declared.episodes() == [(3, 0), (3, 1), (3, 2), (1, 0), (1, 1), (1, 2)]

    def test_max_steps_none_takes_the_suite_cap(self) -> None:
        for suite, cap in LIBERO_MAX_STEPS.items():
            assert build_plan(suite=suite, max_steps=None).max_steps == cap

    def test_an_explicit_max_steps_is_kept_and_the_plan_is_frozen(self) -> None:
        declared = build_plan(max_steps=17)
        assert declared.max_steps == 17
        with pytest.raises(FrozenInstanceError):
            declared.seed = 8  # type: ignore[misc]

    def test_the_two_init_modes_are_the_two_protocols(self) -> None:
        assert set(INIT_MODES) == {"suite_init_states", "seeded_reset"}
        assert issubclass(EvalPlanError, ValueError)

    @pytest.mark.parametrize(
        "bad",
        [
            {"suite": "libero_unknown"},
            {"suite": 3},
            {"suite": None},
            {"task_ids": [0]},
            {"task_ids": ()},
            {"task_ids": (0, 0)},
            {"task_ids": (0, -1)},
            {"task_ids": (0, "1")},
            {"task_ids": (True,)},
            {"trials_per_task": 0},
            {"trials_per_task": -2},
            {"trials_per_task": 1.0},
            {"trials_per_task": True},
            {"seed": 1.5},
            {"seed": "7"},
            {"seed": False},
            {"init": "reset"},
            {"init": None},
            {"init": "Suite_Init_States"},
            {"num_steps_wait": -1},
            {"num_steps_wait": 2.0},
            {"max_steps": 0},
            {"max_steps": -3},
            {"max_steps": "4"},
        ],
    )
    def test_every_invalid_field_is_refused(self, bad: dict) -> None:
        with pytest.raises(EvalPlanError):
            build_plan(**bad)


# -- run_libero: what is refused before anything rolls out ------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        types.SimpleNamespace(name="p", replan_steps=1),
        types.SimpleNamespace(name="", replan_steps=1, reset=None, act=None),
        types.SimpleNamespace(name="p", replan_steps=0, reset=lambda: None, act=lambda o, t: None),
        types.SimpleNamespace(
            name="p", replan_steps=True, reset=lambda: None, act=lambda o, t: None
        ),
        types.SimpleNamespace(name=7, replan_steps=1, reset=lambda: None, act=lambda o, t: None),
        None,
    ],
    ids=["no-methods", "empty-name", "zero-replan", "bool-replan", "non-str-name", "none"],
)
def test_a_policy_that_does_not_speak_the_adapter_protocol_is_refused(policy: object) -> None:
    with pytest.raises(LiberoRunnerError, match="expected a PolicyAdapter"):
        run_libero(build_plan(), policy, make_env=lambda task, resolution, seed: FakeEnv())  # type: ignore[arg-type]


def test_a_fake_policy_is_a_policy_adapter() -> None:
    assert isinstance(FakePolicy(), PolicyAdapter)


def test_plan_resolution_and_make_env_are_refused_by_name() -> None:
    policy = FakePolicy(chunks=[make_chunk(1)])
    make_env = lambda task, resolution, seed: FakeEnv()  # noqa: E731
    with pytest.raises(LiberoRunnerError, match="expected an EpisodePlan"):
        run_libero({"suite": "libero_spatial"}, policy, make_env=make_env)  # type: ignore[arg-type]
    for resolution in (0, -1, 256.0, True, "256"):
        with pytest.raises(LiberoRunnerError, match="resolution is"):
            run_libero(build_plan(), policy, make_env=make_env, resolution=resolution)  # type: ignore[arg-type]
    for bad_maker in (5, "make"):
        with pytest.raises(LiberoRunnerError, match="make_env is"):
            run_libero(build_plan(), policy, make_env=bad_maker)  # type: ignore[arg-type]


# -- the rollout: settle steps, replan cadence, success, budget --------------------------


class TestRollout:
    def test_settle_steps_step_the_dummy_action_without_eating_the_budget(self, harness) -> None:
        declared = build_plan(num_steps_wait=3, max_steps=4)
        make_env = harness.install(FakeBenchmark({0: FakeTask()}), lambda: FakeEnv())
        policy = FakePolicy(replan_steps=2, chunks=[make_chunk(5)])
        episode = run_libero(declared, policy, make_env=make_env).episodes[0]
        rows = make_chunk(5)
        assert harness.envs[0].actions[:3] == [DUMMY] * 3
        assert harness.envs[0].actions[3:] == [rows[0], rows[1], rows[0], rows[1]]
        assert (episode.steps, episode.inferences) == (7, 2)  # 3 settle + 4 policy steps
        assert episode.success is False and episode.error is None
        assert policy.reset_calls == 1

    def test_only_replan_steps_rows_of_each_chunk_are_executed(self, harness) -> None:
        declared = build_plan(num_steps_wait=0, max_steps=6)
        make_env = harness.install(FakeBenchmark({0: FakeTask()}), lambda: FakeEnv())
        policy = FakePolicy(replan_steps=2, chunks=[make_chunk(5)])
        episode = run_libero(declared, policy, make_env=make_env).episodes[0]
        rows = make_chunk(5)
        assert harness.envs[0].actions == [rows[0], rows[1]] * 3
        assert (episode.inferences, episode.steps) == (3, 6)
        assert len(policy.calls) == 3
        assert policy.calls[0] == ({"obs": "reset-2"}, "pick up the bowl")

    def test_done_ends_the_episode_as_a_success(self, harness) -> None:
        declared = build_plan(num_steps_wait=1, max_steps=10)
        make_env = harness.install(FakeBenchmark({0: FakeTask()}), lambda: FakeEnv(done_at=2))
        policy = FakePolicy(replan_steps=1, chunks=[make_chunk(4)])
        episode = run_libero(declared, policy, make_env=make_env).episodes[0]
        assert (episode.success, episode.steps, episode.inferences, episode.error) == (
            True,
            2,
            1,
            None,
        )
        assert harness.envs[0].actions == [DUMMY, make_chunk(4)[0]]

    def test_running_out_of_max_steps_fails_without_an_error(self, harness) -> None:
        declared = build_plan(num_steps_wait=2, max_steps=3)
        make_env = harness.install(FakeBenchmark({0: FakeTask()}), lambda: FakeEnv())
        policy = FakePolicy(replan_steps=1, chunks=[make_chunk(1)])
        episode = run_libero(declared, policy, make_env=make_env).episodes[0]
        assert (episode.success, episode.error) == (False, None)
        assert (episode.steps, episode.inferences) == (5, 3)

    def test_a_gymnasium_truncation_ends_the_episode_as_a_failure(self, harness) -> None:
        declared = build_plan(num_steps_wait=0, max_steps=3)
        make_env = harness.install(
            FakeBenchmark({0: FakeTask()}), lambda: FakeEnv(step_mode="gymnasium", truncated_at=1)
        )
        episode = run_libero(
            declared, FakePolicy(replan_steps=1, chunks=[make_chunk(1)]), make_env=make_env
        )
        assert (episode.episodes[0].success, episode.episodes[0].steps) == (False, 1)

    @pytest.mark.parametrize(
        ("bad_step_value", "needle"),
        [({"reward": 1}, "env.step returned"), ((None, 0.0, False), "3-item")],
        ids=["mapping", "three-tuple"],
    )
    def test_a_step_result_that_is_not_ordered_is_refused(
        self, harness, bad_step_value, needle
    ) -> None:
        declared = build_plan(num_steps_wait=0, max_steps=4)
        make_env = harness.install(
            FakeBenchmark({0: FakeTask()}),
            lambda: FakeEnv(bad_step_at=2, bad_step_value=bad_step_value),
        )
        result = run_libero(
            declared, FakePolicy(replan_steps=1, chunks=[make_chunk(4)]), make_env=make_env
        )
        episode = result.episodes[0]
        assert episode.success is False and episode.steps == 1
        assert needle in episode.error
        assert result.trials() == 1  # the refused episode still counts


# -- both init protocols ----------------------------------------------------------------


class TestInitModes:
    def test_suite_init_states_restores_the_stored_states_in_trial_order(self, harness) -> None:
        declared = build_plan(init="suite_init_states", trials_per_task=3, max_steps=1)
        benchmark = FakeBenchmark({0: FakeTask()}, {0: ["s0", "s1"]})
        make_env = harness.install(benchmark, lambda: FakeEnv(done_at=1))
        run_libero(declared, FakePolicy(replan_steps=1, chunks=[make_chunk(1)]), make_env=make_env)
        env = harness.envs[0]
        assert env.init_states == ["s0", "s1", "s0"]
        assert env.reset_calls == 3 and env.seeds == []
        assert [call[0] for call in FakePolicy(replan_steps=1).calls] == []

    def test_seeded_reset_seeds_seed_plus_trial_and_resets(self, harness) -> None:
        declared = build_plan(init="seeded_reset", seed=100, trials_per_task=2, max_steps=1)
        make_env = harness.install(FakeBenchmark({0: FakeTask()}), lambda: FakeEnv(done_at=1))
        policy = FakePolicy(replan_steps=1, chunks=[make_chunk(1)])
        run_libero(declared, policy, make_env=make_env)
        env = harness.envs[0]
        assert env.seeds == [100, 101]
        assert env.reset_calls == 4 and env.init_states == []
        assert [call[0] for call in policy.calls] == [{"obs": "reset-2"}, {"obs": "reset-4"}]

    def test_make_env_receives_task_resolution_and_seed_and_envs_close(self, harness) -> None:
        task = FakeTask()
        declared = build_plan(task_ids=(0, 1), trials_per_task=1, max_steps=1)
        make_env = harness.install(
            FakeBenchmark({0: task, 1: FakeTask("stack the blocks")}), lambda: FakeEnv(done_at=1)
        )
        result = run_libero(
            declared,
            FakePolicy(replan_steps=1, chunks=[make_chunk(1)]),
            make_env=make_env,
            resolution=64,
        )
        assert [(r.task_id, r.trial) for r in result.episodes] == declared.episodes()
        assert harness.suites == ["libero_spatial"]
        assert [(env.task, env.resolution, env.seed_arg) for env in harness.envs] == [
            (task, 64, 7),
            (harness.envs[1].task, 64, 7),
        ]
        assert [env.closed for env in harness.envs] == [1, 1]
        assert result.policy == "fake-policy"


# -- failures are recorded, never dropped ------------------------------------------------


class TestFailedEpisodes:
    def test_a_short_chunk_is_refused_and_counts_as_a_failure(self, harness) -> None:
        make_env = harness.install(FakeBenchmark({0: FakeTask()}), lambda: FakeEnv())
        policy = FakePolicy(name="shorty", replan_steps=3, chunks=[make_chunk(2)])
        result = run_libero(build_plan(max_steps=5), policy, make_env=make_env)
        episode = result.episodes[0]
        assert (episode.success, episode.steps, episode.inferences) == (False, 0, 1)
        assert "2 step(s)" in episode.error and "replan_steps 3" in episode.error
        assert result.trials() == 1

    @pytest.mark.parametrize(
        ("chunk_value", "needle"),
        [(make_chunk(3, width=6), "shape"), ([["a"] * 7] * 3, "dtype")],
        ids=["wrong-width", "non-numeric"],
    )
    def test_a_malformed_chunk_is_refused(self, harness, chunk_value, needle) -> None:
        make_env = harness.install(FakeBenchmark({0: FakeTask()}), lambda: FakeEnv())
        policy = FakePolicy(name="bad", replan_steps=2, chunks=[chunk_value])
        episode = run_libero(build_plan(max_steps=5), policy, make_env=make_env).episodes[0]
        assert episode.success is False and needle in episode.error

    def test_a_policy_that_raises_is_recorded_as_a_failed_trial(self, harness) -> None:
        make_env = harness.install(FakeBenchmark({0: FakeTask()}), lambda: FakeEnv())
        policy = FakePolicy(raises=RuntimeError("backend down"))
        result = run_libero(build_plan(max_steps=5), policy, make_env=make_env)
        episode = result.episodes[0]
        assert (episode.success, episode.error, episode.inferences) == (False, "backend down", 0)
        assert result.trials() == 1 and result.rate() == 0.0

    def test_an_env_that_raises_mid_episode_keeps_the_step_count(self, harness) -> None:
        make_env = harness.install(FakeBenchmark({0: FakeTask()}), lambda: FakeEnv(raise_at=3))
        policy = FakePolicy(replan_steps=1, chunks=[make_chunk(5)])
        episode = run_libero(build_plan(max_steps=5), policy, make_env=make_env).episodes[0]
        assert (episode.steps, episode.inferences) == (2, 3)
        assert episode.success is False and "fell over" in episode.error

    def test_a_task_the_benchmark_cannot_answer_for_fails_only_its_episodes(self, harness) -> None:
        class Flaky(FakeBenchmark):
            def get_task(self, task_id: int) -> FakeTask:
                if task_id == 1:
                    raise KeyError("not in this suite")
                return super().get_task(task_id)

        make_env = harness.install(
            Flaky({0: FakeTask(), 1: FakeTask()}), lambda: FakeEnv(done_at=1)
        )
        declared = build_plan(task_ids=(0, 1), trials_per_task=2, max_steps=1)
        result = run_libero(
            declared, FakePolicy(replan_steps=1, chunks=[make_chunk(1)]), make_env=make_env
        )
        assert result.trials() == 4  # the denominator never shrinks
        errors = [r.error for r in result.episodes]
        assert errors[:2] == [None, None]
        assert all(e is not None and "the suite carries no task 1" in e for e in errors[2:])
        assert [env.closed for env in harness.envs] == [1]

    @pytest.mark.parametrize(
        ("benchmark", "needle"),
        [
            (FakeBenchmark({0: FakeTask()}, {0: []}), "hold 0 state(s)"),
            (types.SimpleNamespace(get_task=lambda task_id: FakeTask()), "get_task_init_states"),
        ],
        ids=["no-init-states", "no-init-state-getter"],
    )
    def test_a_suite_without_init_states_fails_the_episodes_it_declared(
        self, harness, benchmark, needle
    ) -> None:
        make_env = harness.install(benchmark, lambda: FakeEnv())
        declared = build_plan(init="suite_init_states", trials_per_task=2, max_steps=1)
        result = run_libero(
            declared, FakePolicy(replan_steps=1, chunks=[make_chunk(1)]), make_env=make_env
        )
        assert result.trials() == 2
        assert [needle in r.error for r in result.episodes] == [True, True]

    def test_a_task_without_an_instruction_fails_the_episode(self, harness) -> None:
        make_env = harness.install(FakeBenchmark({0: FakeTask(language="")}), lambda: FakeEnv())
        episode = run_libero(
            build_plan(max_steps=2), FakePolicy(chunks=[make_chunk(1)]), make_env=make_env
        )
        assert episode.episodes[0].success is False
        assert "carries language" in episode.episodes[0].error


# -- the report --------------------------------------------------------------------------


class TestEvalReport:
    def test_counts_rate_and_per_task(self) -> None:
        result = build_report(record(0, 0, True), record(0, 1, False), record(1, 0, True))
        assert (result.successes(), result.trials()) == (2, 3)
        assert result.rate() == pytest.approx(2 / 3)
        assert result.per_task() == {0: (1, 2), 1: (1, 1)}
        assert list(result.per_task()) == [0, 1]

    def test_an_empty_report_has_no_rate_no_interval_and_no_record(self) -> None:
        empty = build_report()
        for call in (empty.rate, empty.wilson_ci, empty.to_json):
            with pytest.raises(LiberoRunnerError, match="at least one"):
                call()

    def test_wilson_ci_matches_a_hand_computed_interval(self) -> None:
        result = build_report(record(success=True), record(trial=1, success=False))
        # p = 1/2, n = 2, z = 1: centre = (0.5 + 1/4) / 1.5 = 0.5 and
        # half = sqrt(0.25/2 + 1/16) / 1.5 = (sqrt(3)/4) / 1.5 = sqrt(3)/6.
        assert result.wilson_ci(z=1.0) == pytest.approx(
            (0.5 - math.sqrt(3.0) / 6.0, 0.5 + math.sqrt(3.0) / 6.0), abs=1e-12
        )
        # the default 95% interval of the same run: centre 0.5, half 1.18430/2.9208 = 0.405471
        assert result.wilson_ci() == pytest.approx((0.0945287, 0.9054713), abs=1e-6)

    @pytest.mark.parametrize("z", [0, -1.0, float("inf"), float("nan"), "1.96", True])
    def test_wilson_ci_refuses_a_z_that_is_not_a_finite_positive_number(self, z) -> None:
        with pytest.raises(LiberoRunnerError, match="z is"):
            build_report(record()).wilson_ci(z)

    def test_to_json_records_plan_policy_episodes_and_rates(self) -> None:
        declared = build_plan(
            task_ids=(2, 5),
            trials_per_task=1,
            seed=11,
            init="suite_init_states",
            num_steps_wait=10,
            max_steps=9,
        )
        result = build_report(record(2, 0, True), record(5, 0, False), declared=declared)
        payload = result.to_json()
        assert payload["plan"] == {
            "suite": "libero_spatial",
            "task_ids": [2, 5],
            "trials_per_task": 1,
            "seed": 11,
            "init": "suite_init_states",
            "num_steps_wait": 10,
            "max_steps": 9,
        }
        assert payload["policy"] == "fake-policy"
        assert (payload["successes"], payload["trials"], payload["rate"]) == (1, 2, 0.5)
        assert payload["wilson_ci"] == list(result.wilson_ci())
        assert payload["episodes"][0] == {
            "task_id": 2,
            "trial": 0,
            "success": True,
            "steps": 1,
            "inferences": 1,
            "wall_s": 0.25,
            "error": None,
        }


# -- the tolerance rule of plan v2 -------------------------------------------------------


class TestWithinPublished:
    def test_gr00t_calibration_reproduces_under_the_two_sample_rule(self) -> None:
        """Measured on GB200: GR00T 190/200 through the FS harness vs the published 195/200.

        The one-sample test (is 0.975 inside OUR Wilson interval?) rejects it -- the interval
        tops out at 0.9726 -- while the two-sample margin the registry uses (3.06 points at
        n = 200 vs 200) accepts it. The harness must agree with the registry.
        """
        result = build_report(
            *[record(trial=i, success=i >= 10) for i in range(200)]  # 190 successes of 200
        )
        assert result.successes() == 190
        assert result.wilson_ci()[1] < 0.975  # the one-sample rule would refuse
        assert within_published(result, 195, 200) is True

    def test_a_real_regression_is_outside_the_margin(self) -> None:
        result = build_report(*[record(trial=i, success=i >= 50) for i in range(200)])  # 75%
        assert within_published(result, 195, 200) is False

    def test_identical_rates_are_within_and_the_bound_is_inclusive(self) -> None:
        result = build_report(record(success=True), record(trial=1, success=False))
        assert within_published(result, 1, 2) is True  # 0.5 vs 0.5
        # p = 0.5, n = 2 vs 2, z = 1: margin sqrt(0.125 + 0.125) = 0.5
        assert within_published(result, 0, 2, z=1.0) is False  # p = 0 -> pooled p = 0.25
        assert within_published(result, 2, 4, z=1.0) is True

    def test_a_degenerate_published_rate_uses_the_pooled_rate(self) -> None:
        # published 10/10 (p = 1, no binomial spread); ours 9/10 -> pooled p = 0.95
        result = build_report(*[record(trial=i, success=i != 0) for i in range(10)])
        assert within_published(result, 10, 10) is True
        result = build_report(*[record(trial=i, success=i >= 5) for i in range(10)])  # 50%
        assert within_published(result, 10, 10) is False

    @pytest.mark.parametrize(
        ("published_successes", "published_trials"),
        [(1, 0), (1, -5), (1, "10"), (1, True), (-1, 10), (11, 10), (1.5, 10), (True, 10)],
    )
    def test_impossible_published_numbers_are_refused(
        self, published_successes: object, published_trials: object
    ) -> None:
        with pytest.raises(LiberoRunnerError):
            within_published(build_report(record()), published_successes, published_trials)  # type: ignore[arg-type]

    def test_a_report_that_is_not_a_report_is_refused(self) -> None:
        with pytest.raises(LiberoRunnerError, match="expected an EvalReport"):
            within_published({"rate": 0.5}, 1, 2)  # type: ignore[arg-type]


# -- where the runner looks up the benchmark ---------------------------------------------


def _install_fake_benchmark_module(monkeypatch: pytest.MonkeyPatch, factory: object) -> None:
    module = types.ModuleType("libero.libero.benchmark")
    module.get_benchmark_dict = factory  # type: ignore[attr-defined]
    libero_libero = sys.modules.get("libero.libero") or types.ModuleType("libero.libero")
    if not hasattr(libero_libero, "get_libero_path"):
        libero_libero.get_libero_path = lambda key: f"/libero/{key}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero", types.ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", module)


def test_the_benchmark_comes_from_liberos_benchmark_dict(monkeypatch) -> None:
    calls: list[bool] = []

    class Bench(FakeBenchmark):
        def __init__(self) -> None:
            super().__init__({0: FakeTask()})

    def get_benchmark_dict() -> dict:
        calls.append(True)
        return {"libero_spatial": Bench}

    _install_fake_benchmark_module(monkeypatch, get_benchmark_dict)
    declared = build_plan(max_steps=1)
    result = run_libero(
        declared,
        FakePolicy(replan_steps=1, chunks=[make_chunk(1)]),
        make_env=lambda t, r, s: FakeEnv(done_at=1),
    )
    assert (result.successes(), result.trials()) == (1, 1)
    assert calls == [True]


def test_a_suite_the_benchmark_dict_does_not_carry_is_refused(monkeypatch) -> None:
    _install_fake_benchmark_module(monkeypatch, lambda: {"libero_spatial": FakeBenchmark})
    with pytest.raises(LiberoRunnerError, match="is not one of the LIBERO benchmarks"):
        run_libero(
            build_plan(suite="libero_object"),
            FakePolicy(chunks=[make_chunk(1)]),
            make_env=lambda t, r, s: FakeEnv(),
        )


def test_an_unimportable_libero_is_refused_by_name(monkeypatch) -> None:
    for name in ("libero", "libero.libero", "libero.libero.benchmark"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(LiberoRunnerError, match="is not importable"):
        run_libero(
            build_plan(),
            FakePolicy(chunks=[make_chunk(1)]),
            make_env=lambda t, r, s: FakeEnv(),
        )


def test_the_default_env_is_offscreen_render_env_over_the_task_bddl(
    monkeypatch, fake_libero_envs
) -> None:
    _install_fake_benchmark_module(monkeypatch, lambda: {"libero_spatial": FakeBenchmark})
    monkeypatch.setitem(
        sys.modules, "libero.libero.benchmark", types.ModuleType("libero.libero.benchmark")
    )
    task = FakeTask()
    monkeypatch.setattr(libero_runner, "_benchmark", lambda suite: FakeBenchmark({0: task}))
    result = run_libero(
        build_plan(max_steps=1), FakePolicy(replan_steps=1, chunks=[make_chunk(1)]), resolution=64
    )
    assert result.successes() == 1
    assert fake_libero_envs[0].kwargs == {
        "bddl_file_name": "/libero/bddl_files/tasks/bowl.bddl",
        "camera_heights": 64,
        "camera_widths": 64,
    }
    # seeded once at construction, and again per trial under the GR00T "seeded_reset" protocol
    assert fake_libero_envs[0].seeds[0] == 7 and fake_libero_envs[0].closed == 1


def test_a_task_without_a_bddl_file_fails_its_episodes(monkeypatch, fake_libero_envs) -> None:
    monkeypatch.setattr(
        libero_runner, "_benchmark", lambda suite: FakeBenchmark({0: FakeTask(bddl="")})
    )
    result = run_libero(
        build_plan(trials_per_task=2, max_steps=1),
        FakePolicy(replan_steps=1, chunks=[make_chunk(1)]),
    )
    assert result.trials() == 2
    assert ["bddl_file" in r.error for r in result.episodes] == [True, True]
