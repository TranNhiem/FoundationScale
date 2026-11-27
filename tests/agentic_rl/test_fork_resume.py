"""Tests for ARL-18 slice 4: response-boundary checkpoints and fork resumption."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from foundationscale.agentic_rl.contracts import Termination, Trajectory, TrajectoryRefusal
from foundationscale.agentic_rl.forking import (
    ForkBoundary,
    ForkRefusal,
    check_fork,
    resume_fork,
)
from foundationscale.agentic_rl.fs_trace import (
    ForkPoint,
    SpanKind,
    fold,
    fork_prefix,
    trace_from_trajectory,
)
from foundationscale.agentic_rl.harness.base import (
    EpisodeBudget,
    EpisodeCheckpoint,
    EpisodeOutcome,
    EpisodeTask,
    Generation,
    HarnessRefusal,
    SamplingParams,
)
from foundationscale.agentic_rl.harness.native_tool_loop import (
    NativeToolLoop,
    NativeToolLoopRefusal,
)


def _load_helpers() -> ModuleType:
    path = Path(__file__).with_name("test_native_tool_loop.py")
    spec = importlib.util.spec_from_file_location("_native_tool_loop_test_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_helpers()
ToyTokenizer: Any = _HELPERS.ToyTokenizer
ScriptedClient: Any = _HELPERS.ScriptedClient
FakeEnvironment: Any = _HELPERS.FakeEnvironment
_bash_call_text: Any = _HELPERS._bash_call_text
_submit_call_text: Any = _HELPERS._submit_call_text
_generation: Any = _HELPERS._generation
_task: Any = _HELPERS._task
_loop: Any = _HELPERS._loop
_budget: Any = _HELPERS._budget
_exec_result: Any = _HELPERS._exec_result


class CountingTokenizer(ToyTokenizer):
    """ToyTokenizer that counts exactly the renders requested by the loop."""

    def __init__(self) -> None:
        super().__init__()
        self.render_calls = 0

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.render_calls += 1
        rendered: list[int] = super().render(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )
        return rendered


class RecordingClient:
    """Wrap ScriptedClient and record the exact context of every generate call."""

    def __init__(
        self,
        scripted: Any,
        *,
        on_generate: Callable[[], None] | None = None,
    ) -> None:
        self._scripted = scripted
        self._on_generate = on_generate
        self.prompts: list[tuple[int, ...]] = []

    async def generate(
        self,
        prompt_ids: Sequence[int],
        sampling: SamplingParams,
    ) -> Generation:
        self.prompts.append(tuple(prompt_ids))
        if self._on_generate is not None:
            self._on_generate()
        generation: Generation = await self._scripted.generate(prompt_ids, sampling)
        return generation


@dataclass(frozen=True)
class _ParentEpisode:
    outcome: EpisodeOutcome
    checkpoint: EpisodeCheckpoint
    generations: tuple[Generation, ...]
    prompt_render_calls: int


def _parent_episode() -> _ParentEpisode:
    """Run bash, bash, submit and return the boundary before generation two."""
    tokenizer = CountingTokenizer()
    generations = (
        _generation(_bash_call_text("echo one")),
        _generation(_bash_call_text("echo two")),
        _generation(_submit_call_text("finished")),
    )
    outputs = {"echo one": "one\n", "echo two": "two\n"}
    env = FakeEnvironment(
        execute_fn=lambda command: _exec_result(outputs[command]),
    )
    checkpoints: list[EpisodeCheckpoint] = []
    render_counts: list[int] = []
    client = RecordingClient(
        ScriptedClient(generations),
        on_generate=lambda: render_counts.append(tokenizer.render_calls),
    )

    async def collect(checkpoint: EpisodeCheckpoint) -> None:
        checkpoints.append(checkpoint)

    loop: NativeToolLoop = _loop(tokenizer=tokenizer)
    outcome: EpisodeOutcome = asyncio.run(
        loop.run_episode(
            _task(),
            client=client,
            env=env,
            budget=_budget(),
            checkpoint_hook=collect,
        )
    )
    assert outcome.termination is Termination.STOP
    assert outcome.submitted_answer == "finished"
    assert env.commands == ["echo one", "echo two"]
    assert len(checkpoints) == 3
    assert checkpoints[1].steps_used == 1
    assert len(checkpoints[1].turns) == 2
    assert render_counts
    return _ParentEpisode(
        outcome=outcome,
        checkpoint=checkpoints[1],
        generations=generations,
        prompt_render_calls=render_counts[0],
    )


def _remaining_environment() -> Any:
    remaining_results = {"echo two": "two\n"}
    return FakeEnvironment(
        execute_fn=lambda command: _exec_result(remaining_results[command]),
    )


def test_resume_is_equivalent_and_does_not_replay_tools() -> None:
    parent = _parent_episode()
    client = RecordingClient(ScriptedClient(parent.generations[1:]))
    env = _remaining_environment()
    loop: NativeToolLoop = _loop()

    resumed: EpisodeOutcome = asyncio.run(
        loop.run_episode(
            _task(),
            client=client,
            env=env,
            budget=_budget(),
            resume=parent.checkpoint,
        )
    )

    assert resumed.trajectory_turns == parent.outcome.trajectory_turns
    assert resumed.prompt_turns == parent.outcome.prompt_turns
    assert resumed.termination == parent.outcome.termination
    assert resumed.submitted_answer == parent.outcome.submitted_answer
    assert client.prompts[0] == parent.checkpoint.token_ids
    assert len(client.prompts) == 2
    assert env.commands == ["echo two"]


def test_resume_does_not_render_prompt_before_first_generation() -> None:
    parent = _parent_episode()
    tokenizer = CountingTokenizer()
    render_counts: list[int] = []
    client = RecordingClient(
        ScriptedClient(parent.generations[1:]),
        on_generate=lambda: render_counts.append(tokenizer.render_calls),
    )
    loop: NativeToolLoop = _loop(tokenizer=tokenizer)

    outcome: EpisodeOutcome = asyncio.run(
        loop.run_episode(
            _task(),
            client=client,
            env=_remaining_environment(),
            budget=_budget(),
            resume=parent.checkpoint,
        )
    )

    assert outcome.termination is Termination.STOP
    assert parent.prompt_render_calls > 0
    assert render_counts[0] == 0
    assert render_counts[0] < parent.prompt_render_calls
    assert len(render_counts) == 2
    assert render_counts[1] > 0


def test_resume_continues_the_step_budget() -> None:
    parent = _parent_episode()
    checkpoint = parent.checkpoint
    client = RecordingClient(
        ScriptedClient([_generation(_bash_call_text("echo capped"))]),
    )
    env = FakeEnvironment(
        execute_fn=lambda command: _exec_result(f"{command}\n"),
    )
    loop: NativeToolLoop = _loop()
    budget: EpisodeBudget = _budget(step_limit=checkpoint.steps_used + 1)

    outcome: EpisodeOutcome = asyncio.run(
        loop.run_episode(
            _task(),
            client=client,
            env=env,
            budget=budget,
            resume=checkpoint,
        )
    )

    assert outcome.termination is Termination.STEP_LIMIT
    assert outcome.submitted_answer is None
    assert len(client.prompts) == 1
    assert env.commands == ["echo capped"]


def test_resume_skips_starting_hook_and_propagates_later_hook_errors() -> None:
    parent = _parent_episode()
    calls: list[EpisodeCheckpoint] = []

    async def exploding_hook(checkpoint: EpisodeCheckpoint) -> None:
        calls.append(checkpoint)
        raise RuntimeError("checkpoint hook failed")

    client = RecordingClient(ScriptedClient(parent.generations[1:]))
    env = _remaining_environment()
    loop: NativeToolLoop = _loop()

    with pytest.raises(RuntimeError, match="checkpoint hook failed"):
        asyncio.run(
            loop.run_episode(
                _task(),
                client=client,
                env=env,
                budget=_budget(),
                checkpoint_hook=exploding_hook,
                resume=parent.checkpoint,
            )
        )

    assert len(calls) == 1
    assert calls[0].steps_used == parent.checkpoint.steps_used + 1
    assert len(calls[0].turns) == len(parent.checkpoint.turns) + 2
    assert len(client.prompts) == 1
    assert env.commands == ["echo two"]


def test_resume_refuses_a_checkpoint_for_another_task() -> None:
    parent = _parent_episode()
    wrong_task = dataclasses.replace(parent.checkpoint, task_uid="another-task")
    client = RecordingClient(ScriptedClient([]))
    loop: NativeToolLoop = _loop()

    with pytest.raises(NativeToolLoopRefusal, match="checkpoint is for task"):
        asyncio.run(
            loop.run_episode(
                _task(),
                client=client,
                env=FakeEnvironment(),
                budget=_budget(),
                resume=wrong_task,
            )
        )

    assert client.prompts == []


def test_checkpoint_refuses_empty_prompt_and_assistant_boundary() -> None:
    checkpoint = _parent_episode().checkpoint

    with pytest.raises(HarnessRefusal, match="prompt_turns.*empty"):
        dataclasses.replace(checkpoint, prompt_turns=())

    with pytest.raises(HarnessRefusal, match="last turn is an assistant turn"):
        dataclasses.replace(checkpoint, turns=checkpoint.turns[:1])


@pytest.mark.parametrize("field", ["cursor", "steps_used"])
def test_checkpoint_refuses_negative_counters(field: str) -> None:
    checkpoint = _parent_episode().checkpoint

    with pytest.raises(HarnessRefusal, match=rf"{field}.*>= 0"):
        if field == "cursor":
            dataclasses.replace(checkpoint, cursor=-1)
        else:
            dataclasses.replace(checkpoint, steps_used=-1)


@pytest.mark.parametrize(
    "elapsed_s",
    [-0.01, float("nan"), float("inf"), True, "not-a-number"],
)
def test_checkpoint_refuses_bad_elapsed_time(elapsed_s: Any) -> None:
    checkpoint = _parent_episode().checkpoint

    with pytest.raises(HarnessRefusal, match="elapsed_s.*finite real >= 0"):
        dataclasses.replace(checkpoint, elapsed_s=elapsed_s)


@pytest.mark.parametrize("task_uid", ["", None, 17])
def test_checkpoint_refuses_bad_task_uid(task_uid: Any) -> None:
    checkpoint = _parent_episode().checkpoint

    with pytest.raises(HarnessRefusal, match="task_uid.*non-empty str"):
        dataclasses.replace(checkpoint, task_uid=task_uid)


def _fork_parts(checkpoint: EpisodeCheckpoint) -> tuple[ForkPoint, ForkBoundary]:
    point = ForkPoint(
        parent_trace_id="trace-parent",
        parent_branch_id="branch-parent",
        branch_id="branch-fork",
        prompt_token_ids=checkpoint.token_ids,
        turns_kept=len(checkpoint.turns),
    )
    boundary = ForkBoundary(
        snapshot_id="snapshot-1",
        env_signature="env-signature-1",
        checkpoint=checkpoint,
        parent_trace_id=point.parent_trace_id,
        parent_branch_id=point.parent_branch_id,
    )
    return point, boundary


def test_check_fork_accepts_exact_boundary_and_refuses_each_mismatch() -> None:
    point, boundary = _fork_parts(_parent_episode().checkpoint)
    check_fork(point, boundary)

    changed_ids = (*point.prompt_token_ids[:-1], point.prompt_token_ids[-1] + 1)
    mismatches = (
        (
            dataclasses.replace(point, parent_trace_id="trace-other"),
            "parent_trace_id",
        ),
        (
            dataclasses.replace(point, parent_branch_id="branch-other"),
            "parent_branch_id",
        ),
        (
            dataclasses.replace(point, branch_id=point.parent_branch_id),
            "equals its parent branch",
        ),
        (
            dataclasses.replace(point, turns_kept=point.turns_kept + 1),
            "turns_kept",
        ),
        (
            dataclasses.replace(point, prompt_token_ids=changed_ids),
            "token prefix",
        ),
    )
    for bad_point, match in mismatches:
        with pytest.raises(ForkRefusal, match=match):
            check_fork(bad_point, boundary)


@dataclass(frozen=True)
class _FakeSnapshot:
    env_signature: str


class _FakeSnapshotStore:
    def __init__(self, env_signature: str) -> None:
        self._env_signature = env_signature
        self.requested_ids: list[str] = []

    def get(self, snapshot_id: str) -> _FakeSnapshot:
        self.requested_ids.append(snapshot_id)
        return _FakeSnapshot(env_signature=self._env_signature)


class _FakeRestoreEnvironment:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.restore_calls: list[tuple[Any, str]] = []

    async def restore(self, store: Any, snapshot_id: str) -> None:
        self._events.append("restore")
        self.restore_calls.append((store, snapshot_id))


class _RecordingHarness:
    def __init__(self, outcome: EpisodeOutcome, events: list[str]) -> None:
        self._outcome = outcome
        self._events = events
        self.run_calls = 0
        self.received_resume: EpisodeCheckpoint | None = None

    async def run_episode(
        self,
        task: EpisodeTask,
        *,
        client: Any,
        env: Any,
        budget: EpisodeBudget,
        resume: EpisodeCheckpoint | None = None,
    ) -> EpisodeOutcome:
        del task, client, env, budget
        self._events.append("run_episode")
        self.run_calls += 1
        self.received_resume = resume
        return self._outcome


def test_resume_fork_restores_before_running_and_adds_metadata() -> None:
    parent = _parent_episode()
    point, boundary = _fork_parts(parent.checkpoint)
    events: list[str] = []
    env = _FakeRestoreEnvironment(events)
    store = _FakeSnapshotStore(boundary.env_signature)
    harness = _RecordingHarness(parent.outcome, events)

    outcome: EpisodeOutcome = asyncio.run(
        resume_fork(
            point,
            boundary,
            harness=cast(Any, harness),
            task=_task(),
            client=ScriptedClient([]),
            env=cast(Any, env),
            budget=_budget(),
            store=cast(Any, store),
        )
    )

    assert events == ["restore", "run_episode"]
    assert env.restore_calls == [(store, boundary.snapshot_id)]
    assert store.requested_ids == [boundary.snapshot_id]
    assert harness.run_calls == 1
    assert harness.received_resume is boundary.checkpoint
    assert outcome.metadata["fork_parent_trace_id"] == point.parent_trace_id
    assert outcome.metadata["fork_parent_branch_id"] == point.parent_branch_id
    assert outcome.metadata["fork_branch_id"] == point.branch_id
    assert outcome.metadata["fork_turns_kept"] == str(point.turns_kept)


def test_resume_fork_refuses_signature_mismatch_without_restore() -> None:
    parent = _parent_episode()
    point, boundary = _fork_parts(parent.checkpoint)
    events: list[str] = []
    env = _FakeRestoreEnvironment(events)
    store = _FakeSnapshotStore("different-signature")
    harness = _RecordingHarness(parent.outcome, events)

    with pytest.raises(ForkRefusal, match="records signature"):
        asyncio.run(
            resume_fork(
                point,
                boundary,
                harness=cast(Any, harness),
                task=_task(),
                client=ScriptedClient([]),
                env=cast(Any, env),
                budget=_budget(),
                store=cast(Any, store),
            )
        )

    assert env.restore_calls == []
    assert harness.run_calls == 0
    assert events == []


def test_resume_fork_refuses_task_uid_mismatch() -> None:
    parent = _parent_episode()
    point, boundary = _fork_parts(parent.checkpoint)
    events: list[str] = []
    env = _FakeRestoreEnvironment(events)
    store = _FakeSnapshotStore(boundary.env_signature)
    harness = _RecordingHarness(parent.outcome, events)

    with pytest.raises(ForkRefusal, match="is not the checkpoint's task"):
        asyncio.run(
            resume_fork(
                point,
                boundary,
                harness=cast(Any, harness),
                task=_task(uid="another-task"),
                client=ScriptedClient([]),
                env=cast(Any, env),
                budget=_budget(),
                store=cast(Any, store),
            )
        )

    assert store.requested_ids == []
    assert env.restore_calls == []
    assert harness.run_calls == 0


# --- ARL-19: the kept fork prefix gets no policy-gradient credit -----------------------


def _forked_outcome() -> tuple[EpisodeOutcome, EpisodeCheckpoint]:
    parent = _parent_episode()
    point, boundary = _fork_parts(parent.checkpoint)
    events: list[str] = []
    outcome = asyncio.run(
        resume_fork(
            point,
            boundary,
            harness=cast(Any, _RecordingHarness(parent.outcome, events)),
            task=_task(),
            client=ScriptedClient([]),
            env=cast(Any, _FakeRestoreEnvironment(events)),
            budget=_budget(),
            store=cast(Any, _FakeSnapshotStore(boundary.env_signature)),
        )
    )
    return outcome, parent.checkpoint


def _as_trajectory(outcome: EpisodeOutcome) -> Trajectory:
    return Trajectory(
        uid="t",
        session_id="t#fork",
        harness="native_tool_loop",
        prompt_turns=outcome.prompt_turns,
        turns=outcome.trajectory_turns,
        reward=1.0,
        abstention_reason=None,
        termination=outcome.termination,
        policy_version_min=0,
        policy_version_max=0,
    )


def test_resume_fork_withholds_credit_from_the_kept_prefix_only() -> None:
    outcome, checkpoint = _forked_outcome()
    kept = len(checkpoint.turns)
    assert kept > 0
    prefix, rest = outcome.trajectory_turns[:kept], outcome.trajectory_turns[kept:]
    assert all(not t.credited for t in prefix if t.generated)
    assert all(t.credited for t in prefix if not t.generated)
    assert all(t.credited for t in rest)
    # provenance is untouched: same tokens, still generated, same logprobs
    assert [(t.token_ids, t.generated, t.logprobs) for t in prefix] == [
        (t.token_ids, t.generated, t.logprobs) for t in checkpoint.turns
    ]
    traj = _as_trajectory(outcome)
    mask = traj.loss_mask()
    prefix_len = sum(len(t.token_ids) for t in prefix)
    assert set(mask[:prefix_len]) == {0}
    assert 1 in mask[prefix_len:]
    assert traj.rollout_logprobs()[:prefix_len] == tuple(
        lp for t in prefix for lp in (t.logprobs or (None,) * len(t.token_ids))
    )


def test_resume_fork_refuses_an_outcome_that_does_not_start_with_the_prefix() -> None:
    parent = _parent_episode()
    point, boundary = _fork_parts(parent.checkpoint)
    events: list[str] = []
    wrong = dataclasses.replace(
        parent.outcome, trajectory_turns=parent.outcome.trajectory_turns[1:]
    )
    with pytest.raises(ForkRefusal, match="kept prefix cannot be identified"):
        asyncio.run(
            resume_fork(
                point,
                boundary,
                harness=cast(Any, _RecordingHarness(wrong, events)),
                task=_task(),
                client=ScriptedClient([]),
                env=cast(Any, _FakeRestoreEnvironment(events)),
                budget=_budget(),
                store=cast(Any, _FakeSnapshotStore(boundary.env_signature)),
            )
        )


def test_turn_credited_is_a_strict_bool_and_only_withheld_from_generated_turns() -> None:
    outcome, _ = _forked_outcome()
    generated = next(t for t in outcome.trajectory_turns if t.generated)
    injected = next(t for t in outcome.trajectory_turns if not t.generated)
    with pytest.raises(TrajectoryRefusal, match="credited"):
        dataclasses.replace(generated, credited=cast(Any, 1))
    with pytest.raises(TrajectoryRefusal, match="non-generated turn"):
        dataclasses.replace(injected, credited=False)


def test_trace_round_trip_keeps_credit_and_leaves_credited_traces_unchanged() -> None:
    outcome, checkpoint = _forked_outcome()
    traj = _as_trajectory(outcome)
    spans = trace_from_trajectory(traj, run_id="r", rollout_step=0)
    assert fold(spans) == traj
    turn_data = [s.data for s in spans if s.kind is SpanKind.TURN]
    assert any(d.get("credited") is False for d in turn_data)
    assert all("credited" not in d for d in turn_data if d["phase"] == "prompt")
    fork_prefix(spans, turn_index=traj.turns[-1].index)  # tolerates the optional key

    full = _as_trajectory(_parent_episode().outcome)
    plain = trace_from_trajectory(full, run_id="r", rollout_step=0)
    assert all("credited" not in s.data for s in plain if s.kind is SpanKind.TURN)
    assert fold(plain) == full


def test_trajectory_v2_refuses_to_lift_an_uncredited_fork() -> None:
    from foundationscale.agentic_rl import trajectory_v2

    outcome, _ = _forked_outcome()
    unused = cast(Any, None)  # the refusal fires before any metadata is read
    with pytest.raises(trajectory_v2.TrajectoryV2Refusal, match="uncredited turns"):
        trajectory_v2.from_v1(
            _as_trajectory(outcome), model=unused, harness=unused, env=unused, gateway=unused
        )


def test_checkpoint_refuses_a_cursor_past_its_messages() -> None:
    checkpoint = _parent_episode().checkpoint
    with pytest.raises(HarnessRefusal, match="cursor"):
        dataclasses.replace(checkpoint, cursor=len(checkpoint.messages) + 1)
