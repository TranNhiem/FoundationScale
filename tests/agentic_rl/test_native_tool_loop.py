"""Deterministic tests for NativeToolLoop, slice 3's multi-turn tool-calling harness.

Claimed here (11 scenarios, no network, no skips):
  * a bash call followed by a submit call ends the episode with the submitted answer
  * Trajectory.loss_mask() is 1 exactly on generated ASSISTANT tokens
  * an unregistered tool name records ToolCall.parse_error and yields an error
    observation without ending the episode
  * budget.step_limit steps with no other termination ends Termination.STEP_LIMIT
  * a real response-budget squeeze (not finish_reason="length") ends Termination.LENGTH
  * finish_reason "length" ends Termination.LENGTH
  * finish_reason "abort" ends Termination.ABORTED
  * EnvInfraError from env.execute ends Termination.INFRA with an "infra:" reason
  * EngineInfraError from client.generate ends Termination.INFRA with an
    "infra:engine_<kind>" reason
  * max_observation_chars middle-truncates the observation with a "chars elided" marker
  * a non-prefix-stable tokenizer is refused with NativeToolLoopRefusal
  * an exhausted episode wall clock ends Termination.TIMEOUT before any generation

NOT claimed: no real tokenizer, no real subprocess, no real network. Those live in
test_envs_local.py and test_tools.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from foundationscale.agentic_rl import markup
from foundationscale.agentic_rl.contracts import SegmentKind, Termination, Trajectory, Turn
from foundationscale.agentic_rl.engines import EngineInfraError
from foundationscale.agentic_rl.envs.base import EnvInfraError, ExecResult
from foundationscale.agentic_rl.harness.base import (
    EpisodeBudget,
    EpisodeOutcome,
    EpisodeTask,
    Generation,
    GenerationClient,
    HarnessRefusal,
    SamplingParams,
    to_trajectory,
)
from foundationscale.agentic_rl.harness.native_tool_loop import (
    NativeToolLoop,
    NativeToolLoopRefusal,
)
from foundationscale.agentic_rl.tools import ToolSpec, default_tools


class ToyTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
    ) -> list[int]:
        parts: list[str] = []
        if tools:
            parts.append("<tools>")
        for message in messages:
            parts.append(f"<{message['role']}>")
            parts.extend(str(message["content"]).split())
        if add_generation_prompt:
            parts.append("<assistant>")
        return [self._id(part) for part in parts]

    def decode(self, ids: Sequence[int]) -> str:
        inverse = {value: key for key, value in self._vocab.items()}
        return " ".join(inverse[i] for i in ids)


class LyingTokenizer(ToyTokenizer):
    """A tokenizer whose render is NOT prefix-stable: it stamps the current total
    message count into its very first token, so two cumulative renders over a
    growing message list never share a common prefix -- unlike prepending the same
    constant to every render (which stays prefix-stable), this actually changes an
    EARLY, already-emitted token as later messages are added.
    """

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
    ) -> list[int]:
        marker = self._id(f"<count={len(messages)}>")
        rest = super().render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        return [marker, *rest]


class ScriptedClient:
    def __init__(self, generations: Sequence[Generation]) -> None:
        self._generations = list(generations)
        self._calls = 0

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        assert self._calls < len(self._generations), "scripted client ran out of turns"
        generation = self._generations[self._calls]
        self._calls += 1
        return generation


class ExplodingEngineClient:
    """A GenerationClient that returns each of ``generations`` in order, then raises
    EngineInfraError on every call after they are exhausted.
    """

    def __init__(
        self,
        generations: Sequence[Generation] = (),
        *,
        kind: str = "transport",
        message: str = "connection reset",
    ) -> None:
        self._generations = list(generations)
        self._calls = 0
        self._kind = kind
        self._message = message

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        if self._calls < len(self._generations):
            generation = self._generations[self._calls]
            self._calls += 1
            return generation
        raise EngineInfraError(self._kind, self._message)


def _bash_call_text(command: str) -> str:
    return (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "bash"
        + markup.TAG_CLOSE
        + markup.PARAMETER_OPEN_PREFIX
        + "command"
        + markup.TAG_CLOSE
        + command
        + markup.PARAMETER_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )


def _submit_call_text(answer: str) -> str:
    return (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "submit"
        + markup.TAG_CLOSE
        + markup.PARAMETER_OPEN_PREFIX
        + "answer"
        + markup.TAG_CLOSE
        + answer
        + markup.PARAMETER_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )


def _unknown_call_text() -> str:
    return (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "frobnicate"
        + markup.TAG_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )


def _argless_call_text(name: str) -> str:
    """A tool call for ``name`` with NO parameters at all (empty ``arguments``)."""
    return (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + name
        + markup.TAG_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )


def _tools_without_required(name: str, param: str) -> dict[str, ToolSpec]:
    """``default_tools()`` with ``name``'s ToolSpec rebuilt to declare NO required
    parameters -- so ``validate_call`` cannot catch a missing ``param`` itself,
    exercising the harness's own hardcoded fallback check for "submit"/"bash".
    """
    tools = dict(default_tools())
    original = tools[name]
    tools[name] = ToolSpec(
        name=original.name,
        description=original.description,
        parameters=original.parameters,
        required=(),
    )
    return tools


_TOKEN_COUNTER_START = 10_000


def _generation(
    text: str, *, finish_reason: str = "stop", n_tokens: int | None = None
) -> Generation:
    # Generated token ids are intentionally in a disjoint id range from the
    # ToyTokenizer's own vocabulary (which starts at 0) -- they represent the
    # engine's own sampled ids, never re-derived from a chat-template render.
    count = n_tokens if n_tokens is not None else max(1, len(text.split()))
    global _TOKEN_COUNTER_START
    ids = tuple(range(_TOKEN_COUNTER_START, _TOKEN_COUNTER_START + count))
    _TOKEN_COUNTER_START += count
    return Generation(token_ids=ids, logprobs=None, finish_reason=finish_reason, text=text)  # type: ignore[arg-type]


class FakeEnvironment:
    def __init__(
        self,
        *,
        execute_fn: Callable[[str], ExecResult] | None = None,
        episode_timeout_s: float = 3600.0,
    ) -> None:
        self._execute_fn = execute_fn
        self._episode_timeout_s = episode_timeout_s
        self._started = False
        self._closed = False
        self.commands: list[str] = []

    async def start(self) -> None:
        self._started = True

    async def execute(self, command: str, *, timeout_s: float | None = None) -> ExecResult:
        self.commands.append(command)
        if self._execute_fn is None:
            return ExecResult(
                stdout="", stderr="", exit_code=0, timed_out=False, truncated_bytes=0, seconds=0.0
            )
        return self._execute_fn(command)

    async def close(self) -> None:
        self._closed = True

    @property
    def workdir(self) -> str:
        return "/fake"

    @property
    def episode_timeout_s(self) -> float:
        return self._episode_timeout_s


def _run(
    loop: NativeToolLoop,
    task: EpisodeTask,
    client: GenerationClient,
    env: FakeEnvironment,
    budget: EpisodeBudget,
) -> EpisodeOutcome:
    return asyncio.run(loop.run_episode(task, client=client, env=env, budget=budget))


def _task(**overrides: Any) -> EpisodeTask:
    fields: dict[str, Any] = {
        "uid": "u1",
        "session_id": "s1",
        "messages": (
            {"role": "system", "content": "you are terse"},
            {"role": "user", "content": "do the thing"},
        ),
        "metadata": {},
    }
    fields.update(overrides)
    return EpisodeTask(**fields)


def _loop(**overrides: Any) -> NativeToolLoop:
    from foundationscale.agentic_rl.tools import QwenXmlToolCallParser

    fields: dict[str, Any] = {
        "tools": default_tools(),
        "parser": QwenXmlToolCallParser(),
        "tokenizer": ToyTokenizer(),
        "sampling": SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=64,
        ),
    }
    fields.update(overrides)
    return NativeToolLoop(**fields)


def _budget(**overrides: Any) -> EpisodeBudget:
    fields: dict[str, Any] = {
        "step_limit": 5,
        "max_response_tokens": 4096,
        "max_observation_chars": 256,
    }
    fields.update(overrides)
    return EpisodeBudget(**fields)


def _exec_result(stdout: str) -> ExecResult:
    return ExecResult(
        stdout=stdout, stderr="", exit_code=0, timed_out=False, truncated_bytes=0, seconds=0.01
    )


def _bash_then_submit_scenario() -> tuple[NativeToolLoop, EpisodeOutcome, FakeEnvironment]:
    tokenizer = ToyTokenizer()
    loop = _loop(tokenizer=tokenizer)
    client = ScriptedClient(
        [_generation(_bash_call_text("echo hi")), _generation(_submit_call_text("done"))]
    )
    env = FakeEnvironment(execute_fn=lambda cmd: _exec_result("hi\n"))
    outcome = _run(loop, _task(), client, env, _budget())
    return loop, outcome, env


def test_multi_turn_bash_episode_ends_via_submit() -> None:
    _, outcome, env = _bash_then_submit_scenario()
    assert outcome.termination is Termination.STOP
    assert outcome.submitted_answer == "done"
    assert env.commands == ["echo hi"]
    assert len(outcome.trajectory_turns) == 3
    assert [turn.kind for turn in outcome.trajectory_turns] == [
        SegmentKind.ASSISTANT,
        SegmentKind.TOOL_RESULT,
        SegmentKind.ASSISTANT,
    ]
    indices = [turn.index for turn in outcome.trajectory_turns]
    assert indices == sorted(indices)
    assert all(a < b for a, b in zip(indices, indices[1:], strict=False))
    trajectory = to_trajectory(
        outcome,
        uid="u1",
        session_id="s1",
        harness="native_tool_loop",
        reward=1.0,
        abstention_reason=None,
    )
    assert isinstance(trajectory, Trajectory)


def test_loss_mask_is_one_exactly_on_generated_tokens() -> None:
    _, outcome, _ = _bash_then_submit_scenario()
    trajectory = to_trajectory(
        outcome,
        uid="u1",
        session_id="s1",
        harness="native_tool_loop",
        reward=1.0,
        abstention_reason=None,
    )
    mask = trajectory.loss_mask()
    assert len(mask) == len(trajectory.response_token_ids)
    expected: list[int] = []
    for turn in outcome.trajectory_turns:
        expected.extend([1 if turn.kind is SegmentKind.ASSISTANT else 0] * len(turn.token_ids))
    assert mask == tuple(expected)
    assert sum(mask) == sum(
        len(turn.token_ids)
        for turn in outcome.trajectory_turns
        if turn.kind is SegmentKind.ASSISTANT
    )
    assert all(
        mask[
            sum(len(t.token_ids) for t in outcome.trajectory_turns[: i + 1]) - len(turn.token_ids) :
        ][: len(turn.token_ids)]
        == tuple([0] * len(turn.token_ids))
        for i, turn in enumerate(outcome.trajectory_turns)
        if turn.kind is SegmentKind.TOOL_RESULT
    )


def test_invalid_tool_call_records_parse_error_and_error_observation() -> None:
    tokenizer = ToyTokenizer()
    loop = _loop(tokenizer=tokenizer)
    client = ScriptedClient(
        [_generation(_unknown_call_text()), _generation(_submit_call_text("ok"))]
    )
    env = FakeEnvironment()
    outcome = _run(loop, _task(), client, env, _budget())
    first = outcome.trajectory_turns[0]
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].parse_error is not None
    assert env.commands == []
    second = outcome.trajectory_turns[1]
    assert second.kind is SegmentKind.TOOL_RESULT
    decoded = tokenizer.decode(second.token_ids)
    assert "frobnicate" in decoded
    assert outcome.termination is Termination.STOP
    assert outcome.submitted_answer == "ok"


def test_submit_call_missing_answer_is_reported_not_a_crash() -> None:
    # Regression: when the configured ToolSpec does not declare "answer" required,
    # validate_call() reports no error, and the old code indexed
    # valid_submit.arguments["answer"] directly -- a bare KeyError crashing
    # run_episode instead of a model-attributable parse error.
    tokenizer = ToyTokenizer()
    loop = _loop(tokenizer=tokenizer, tools=_tools_without_required("submit", "answer"))
    client = ScriptedClient(
        [_generation(_argless_call_text("submit")), _generation(_submit_call_text("done"))]
    )
    env = FakeEnvironment()
    outcome = _run(loop, _task(), client, env, _budget())
    first = outcome.trajectory_turns[0]
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].parse_error is not None
    assert "answer" in first.tool_calls[0].parse_error
    second = outcome.trajectory_turns[1]
    assert second.kind is SegmentKind.TOOL_RESULT
    decoded = tokenizer.decode(second.token_ids)
    assert "answer" in decoded
    assert outcome.termination is Termination.STOP
    assert outcome.submitted_answer == "done"


def test_bash_call_missing_command_is_reported_not_a_crash() -> None:
    # Same regression as above, for "bash"'s "command" argument (the other
    # directly-indexed site).
    tokenizer = ToyTokenizer()
    loop = _loop(tokenizer=tokenizer, tools=_tools_without_required("bash", "command"))
    client = ScriptedClient(
        [_generation(_argless_call_text("bash")), _generation(_submit_call_text("done"))]
    )
    env = FakeEnvironment()
    outcome = _run(loop, _task(), client, env, _budget())
    first = outcome.trajectory_turns[0]
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].parse_error is not None
    assert "command" in first.tool_calls[0].parse_error
    assert env.commands == []
    second = outcome.trajectory_turns[1]
    assert second.kind is SegmentKind.TOOL_RESULT
    decoded = tokenizer.decode(second.token_ids)
    assert "command" in decoded
    assert outcome.termination is Termination.STOP
    assert outcome.submitted_answer == "done"


def test_to_trajectory_extra_metadata_merges_over_outcome_metadata() -> None:
    _, outcome, _ = _bash_then_submit_scenario()
    outcome_with_metadata = EpisodeOutcome(
        trajectory_turns=outcome.trajectory_turns,
        prompt_turns=outcome.prompt_turns,
        termination=outcome.termination,
        submitted_answer=outcome.submitted_answer,
        abstention_reason=outcome.abstention_reason,
        metadata={"a": "1", "b": "2"},
    )
    trajectory = to_trajectory(
        outcome_with_metadata,
        uid="u1",
        session_id="s1",
        harness="native_tool_loop",
        reward=1.0,
        abstention_reason=None,
        extra_metadata={"reward_note": "no_abc", "a": "overridden"},
    )
    assert dict(trajectory.metadata) == {"a": "overridden", "b": "2", "reward_note": "no_abc"}


def test_to_trajectory_without_extra_metadata_keeps_outcome_metadata_unchanged() -> None:
    _, outcome, _ = _bash_then_submit_scenario()
    outcome_with_metadata = EpisodeOutcome(
        trajectory_turns=outcome.trajectory_turns,
        prompt_turns=outcome.prompt_turns,
        termination=outcome.termination,
        submitted_answer=outcome.submitted_answer,
        abstention_reason=outcome.abstention_reason,
        metadata={"a": "1"},
    )
    trajectory = to_trajectory(
        outcome_with_metadata,
        uid="u1",
        session_id="s1",
        harness="native_tool_loop",
        reward=1.0,
        abstention_reason=None,
    )
    assert dict(trajectory.metadata) == {"a": "1"}


def test_step_limit_reached() -> None:
    loop = _loop()
    client = ScriptedClient([_generation(_bash_call_text("true")) for _ in range(3)])
    env = FakeEnvironment()
    outcome = _run(loop, _task(), client, env, _budget(step_limit=3))
    assert outcome.termination is Termination.STEP_LIMIT
    assert len(outcome.trajectory_turns) == 6
    assert [turn.kind for turn in outcome.trajectory_turns] == [
        SegmentKind.ASSISTANT,
        SegmentKind.TOOL_RESULT,
    ] * 3


def test_response_budget_exhausted_yields_length() -> None:
    loop = _loop()
    client = ScriptedClient([_generation(_bash_call_text("true"), n_tokens=8) for _ in range(6)])
    env = FakeEnvironment()
    outcome = _run(
        loop,
        _task(),
        client,
        env,
        _budget(step_limit=50, max_response_tokens=10, max_observation_chars=256),
    )
    assert outcome.termination is Termination.LENGTH
    assert len(outcome.trajectory_turns) >= 1
    trajectory = to_trajectory(
        outcome,
        uid="u1",
        session_id="s1",
        harness="native_tool_loop",
        reward=None,
        abstention_reason="budget_exhausted",
    )
    assert isinstance(trajectory, Trajectory)


def test_finish_reason_length_ends_episode() -> None:
    loop = _loop()
    client = ScriptedClient(
        [_generation("partial output with no tool call", finish_reason="length")]
    )
    env = FakeEnvironment()
    outcome = _run(loop, _task(), client, env, _budget())
    assert outcome.termination is Termination.LENGTH
    assert len(outcome.trajectory_turns) == 1


def test_engine_abort_yields_aborted() -> None:
    loop = _loop()
    client = ScriptedClient([_generation("I was cut off", finish_reason="abort")])
    env = FakeEnvironment()
    outcome = _run(loop, _task(), client, env, _budget())
    assert outcome.termination is Termination.ABORTED
    assert outcome.submitted_answer is None
    assert outcome.abstention_reason is None
    trajectory = to_trajectory(
        outcome,
        uid="u1",
        session_id="s1",
        harness="native_tool_loop",
        reward=None,
        abstention_reason="model_aborted",
    )
    assert isinstance(trajectory, Trajectory)


def test_env_infra_error_yields_infra_with_prefixed_reason() -> None:
    def _explode(command: str) -> ExecResult:
        raise EnvInfraError("transport", "connection reset")

    loop = _loop()
    client = ScriptedClient([_generation(_bash_call_text("boom"))])
    env = FakeEnvironment(execute_fn=_explode)
    outcome = _run(loop, _task(), client, env, _budget())
    assert outcome.termination is Termination.INFRA
    assert outcome.abstention_reason == "infra:transport"
    assert outcome.submitted_answer is None
    trajectory = to_trajectory(
        outcome,
        uid="u1",
        session_id="s1",
        harness="native_tool_loop",
        reward=None,
        abstention_reason=outcome.abstention_reason,
    )
    assert isinstance(trajectory, Trajectory)
    assert trajectory.is_infra is True
    assert all(flag == 0 for flag in trajectory.loss_mask())


def test_engine_infra_error_yields_infra_with_engine_prefixed_reason() -> None:
    # The FIRST generate() call succeeds (a bash call, so an assistant turn is
    # appended and the tool executes normally); the SECOND raises EngineInfraError.
    # This mirrors how the existing EnvInfraError test is shaped: by the time the
    # infra error fires, trajectory_turns is already non-empty, which is required
    # for to_trajectory() to build a Trajectory at all (contracts.Trajectory
    # refuses an empty 'turns' tuple under every termination, including INFRA).
    loop = _loop()
    client = ExplodingEngineClient(
        [_generation(_bash_call_text("echo hi"))], kind="transport", message="connection reset"
    )
    env = FakeEnvironment()
    outcome = _run(loop, _task(), client, env, _budget())
    assert outcome.termination is Termination.INFRA
    assert outcome.abstention_reason == "infra:engine_transport"
    assert outcome.submitted_answer is None
    assert len(outcome.trajectory_turns) >= 1
    trajectory = to_trajectory(
        outcome,
        uid="u1",
        session_id="s1",
        harness="native_tool_loop",
        reward=None,
        abstention_reason=outcome.abstention_reason,
    )
    assert isinstance(trajectory, Trajectory)
    assert trajectory.is_infra is True


def test_observation_is_truncated_with_elided_marker() -> None:
    tokenizer = ToyTokenizer()
    loop = _loop(tokenizer=tokenizer)
    client = ScriptedClient(
        [_generation(_bash_call_text("yes")), _generation(_submit_call_text("done"))]
    )
    env = FakeEnvironment(execute_fn=lambda cmd: _exec_result("x" * 2000))
    outcome = _run(
        loop, _task(), client, env, _budget(max_response_tokens=8192, max_observation_chars=64)
    )
    decoded = tokenizer.decode(outcome.trajectory_turns[1].token_ids)
    assert "chars elided" in decoded
    assert len(decoded) < 2000


def test_non_prefix_stable_tokenizer_is_refused() -> None:
    loop = _loop(tokenizer=LyingTokenizer())
    client = ScriptedClient(
        [_generation(_bash_call_text("true")), _generation(_submit_call_text("done"))]
    )
    env = FakeEnvironment()
    with pytest.raises(NativeToolLoopRefusal):
        _run(loop, _task(), client, env, _budget())


def test_episode_timeout_yields_timeout_termination() -> None:
    loop = _loop()
    client = ScriptedClient([_generation(_submit_call_text("never"))])
    env = FakeEnvironment(episode_timeout_s=0.0)
    outcome = _run(loop, _task(), client, env, _budget())
    assert outcome.termination is Termination.TIMEOUT
    assert len(outcome.trajectory_turns) == 0
    with pytest.raises(ValueError):
        to_trajectory(
            outcome,
            uid="u1",
            session_id="s1",
            harness="native_tool_loop",
            reward=None,
            abstention_reason="timed_out",
        )


# --------------------------------------------------- coverage additions below


def test_sampling_params_temperature_must_be_real_number() -> None:
    with pytest.raises(HarnessRefusal):
        SamplingParams(
            temperature=True,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=16,
        )


def test_sampling_params_temperature_must_be_finite() -> None:
    with pytest.raises(HarnessRefusal):
        SamplingParams(
            temperature=float("nan"),
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=16,
        )


def test_sampling_params_top_p_out_of_range() -> None:
    with pytest.raises(HarnessRefusal):
        SamplingParams(
            temperature=1.0,
            top_p=0.0,
            top_k=0,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=16,
        )


def test_sampling_params_top_k_must_be_real_int() -> None:
    with pytest.raises(HarnessRefusal):
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=True,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=16,
        )


def test_sampling_params_top_k_negative() -> None:
    with pytest.raises(HarnessRefusal):
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=16,
        )


def test_sampling_params_min_p_out_of_range() -> None:
    with pytest.raises(HarnessRefusal):
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=1.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=16,
        )


def test_sampling_params_repetition_penalty_non_positive() -> None:
    with pytest.raises(HarnessRefusal):
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=0.0,
            max_new_tokens=16,
        )


def test_sampling_params_max_new_tokens_must_be_real_int() -> None:
    with pytest.raises(HarnessRefusal):
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=1.5,
        )


def test_sampling_params_max_new_tokens_below_one() -> None:
    with pytest.raises(HarnessRefusal):
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=0,
        )


def test_generation_token_ids_must_be_sequence() -> None:
    with pytest.raises(HarnessRefusal):
        Generation(token_ids=5, logprobs=None, finish_reason="stop", text="x")  # type: ignore[arg-type]


def test_generation_token_ids_element_must_be_int() -> None:
    with pytest.raises(HarnessRefusal):
        Generation(token_ids=(1, True), logprobs=None, finish_reason="stop", text="x")


def test_generation_logprobs_length_mismatch() -> None:
    with pytest.raises(HarnessRefusal):
        Generation(token_ids=(1, 2), logprobs=(0.1,), finish_reason="stop", text="x")


def test_generation_logprobs_element_must_be_number() -> None:
    with pytest.raises(HarnessRefusal):
        Generation(token_ids=(1,), logprobs=(True,), finish_reason="stop", text="x")


def test_generation_logprobs_element_must_be_finite() -> None:
    with pytest.raises(HarnessRefusal):
        Generation(token_ids=(1,), logprobs=(float("inf"),), finish_reason="stop", text="x")


def test_generation_logprobs_must_be_sequence() -> None:
    with pytest.raises(HarnessRefusal):
        Generation(token_ids=(1,), logprobs=3, finish_reason="stop", text="x")  # type: ignore[arg-type]


def test_generation_finish_reason_unknown() -> None:
    with pytest.raises(HarnessRefusal):
        Generation(token_ids=(1,), logprobs=None, finish_reason="oops", text="x")  # type: ignore[arg-type]


def test_generation_text_must_be_str() -> None:
    with pytest.raises(HarnessRefusal):
        Generation(token_ids=(1,), logprobs=None, finish_reason="stop", text=7)  # type: ignore[arg-type]


def test_episode_task_uid_must_be_non_empty_str() -> None:
    with pytest.raises(HarnessRefusal):
        _task(uid="")


def test_episode_task_session_id_must_be_str() -> None:
    with pytest.raises(HarnessRefusal):
        _task(session_id=3)  # type: ignore[arg-type]


def test_episode_task_messages_must_be_sequence() -> None:
    with pytest.raises(HarnessRefusal):
        _task(messages=5)  # type: ignore[arg-type]


def test_episode_task_messages_must_be_non_empty() -> None:
    with pytest.raises(HarnessRefusal):
        _task(messages=())


def test_episode_task_message_must_be_mapping() -> None:
    with pytest.raises(HarnessRefusal):
        _task(messages=("nope",))  # type: ignore[arg-type]


def test_episode_task_message_missing_role() -> None:
    with pytest.raises(HarnessRefusal):
        _task(messages=({"content": "hi"},))


def test_episode_task_message_role_must_be_non_empty_str() -> None:
    with pytest.raises(HarnessRefusal):
        _task(messages=({"role": "", "content": "hi"},))


def test_episode_task_message_role_must_be_str() -> None:
    with pytest.raises(HarnessRefusal):
        _task(messages=({"role": 5, "content": "hi"},))  # type: ignore[arg-type]


def test_episode_task_message_role_assistant_forbidden() -> None:
    with pytest.raises(HarnessRefusal):
        _task(messages=({"role": "assistant", "content": "hi"},))


def test_episode_task_message_missing_content() -> None:
    with pytest.raises(HarnessRefusal):
        _task(messages=({"role": "user"},))


def test_episode_task_message_content_must_be_str() -> None:
    with pytest.raises(HarnessRefusal):
        _task(messages=({"role": "user", "content": 5},))  # type: ignore[arg-type]


def test_episode_task_metadata_must_be_mapping() -> None:
    with pytest.raises(HarnessRefusal):
        _task(metadata=[("a", "b")])  # type: ignore[arg-type]


def test_episode_task_metadata_key_must_be_str() -> None:
    with pytest.raises(HarnessRefusal):
        _task(metadata={1: "b"})  # type: ignore[dict-item]


def test_episode_task_metadata_value_must_be_str() -> None:
    with pytest.raises(HarnessRefusal):
        _task(metadata={"a": 2})  # type: ignore[dict-item]


def test_episode_budget_step_limit_must_be_real_int() -> None:
    with pytest.raises(HarnessRefusal):
        _budget(step_limit=True)


def test_episode_budget_step_limit_below_one() -> None:
    with pytest.raises(HarnessRefusal):
        _budget(step_limit=0)


def test_episode_budget_max_response_tokens_must_be_real_int() -> None:
    with pytest.raises(HarnessRefusal):
        _budget(max_response_tokens=1.5)


def test_episode_budget_max_response_tokens_below_one() -> None:
    with pytest.raises(HarnessRefusal):
        _budget(max_response_tokens=0)


def test_episode_budget_max_observation_chars_below_64() -> None:
    with pytest.raises(HarnessRefusal):
        _budget(max_observation_chars=63)


def test_episode_budget_max_observation_chars_must_be_real_int() -> None:
    with pytest.raises(HarnessRefusal):
        _budget(max_observation_chars=64.5)


def _prompt_turn() -> Turn:
    return Turn(index=0, kind=SegmentKind.USER, token_ids=(1,), generated=False, logprobs=None)


def _generated_turn() -> Turn:
    return Turn(
        index=1, kind=SegmentKind.ASSISTANT, token_ids=(2,), generated=True, logprobs=(0.0,)
    )


def test_episode_outcome_trajectory_turns_element_must_be_turn() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=("nope",),  # type: ignore[arg-type]
            prompt_turns=(),
            termination=Termination.STOP,
            submitted_answer=None,
            abstention_reason=None,
        )


def test_episode_outcome_trajectory_turns_must_be_sequence() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=5,  # type: ignore[arg-type]
            prompt_turns=(),
            termination=Termination.STOP,
            submitted_answer=None,
            abstention_reason=None,
        )


def test_episode_outcome_prompt_turns_element_must_be_turn() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=("nope",),  # type: ignore[arg-type]
            termination=Termination.STOP,
            submitted_answer=None,
            abstention_reason=None,
        )


def test_episode_outcome_prompt_turns_must_not_be_generated() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(_generated_turn(),),
            termination=Termination.STOP,
            submitted_answer=None,
            abstention_reason=None,
        )


def test_episode_outcome_termination_must_be_termination() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(),
            termination="stop",  # type: ignore[arg-type]
            submitted_answer=None,
            abstention_reason=None,
        )


def test_episode_outcome_submitted_answer_must_be_str_or_none() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(),
            termination=Termination.STOP,
            submitted_answer=5,  # type: ignore[arg-type]
            abstention_reason=None,
        )


def test_episode_outcome_infra_requires_abstention_reason() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(),
            termination=Termination.INFRA,
            submitted_answer=None,
            abstention_reason=None,
        )


def test_episode_outcome_infra_abstention_reason_must_be_non_empty() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(),
            termination=Termination.INFRA,
            submitted_answer=None,
            abstention_reason="",
        )


def test_episode_outcome_infra_abstention_reason_needs_infra_prefix() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(),
            termination=Termination.INFRA,
            submitted_answer=None,
            abstention_reason="transport",
        )


def test_episode_outcome_non_infra_rejects_abstention_reason() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(),
            termination=Termination.STOP,
            submitted_answer=None,
            abstention_reason="unsure",
        )


def test_episode_outcome_metadata_must_be_mapping() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(),
            termination=Termination.STOP,
            submitted_answer=None,
            abstention_reason=None,
            metadata=[("a", "b")],  # type: ignore[arg-type]
        )


def test_episode_outcome_metadata_key_must_be_str() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(),
            termination=Termination.STOP,
            submitted_answer=None,
            abstention_reason=None,
            metadata={1: "b"},  # type: ignore[dict-item]
        )


def test_episode_outcome_metadata_value_must_be_str() -> None:
    with pytest.raises(HarnessRefusal):
        EpisodeOutcome(
            trajectory_turns=(),
            prompt_turns=(),
            termination=Termination.STOP,
            submitted_answer=None,
            abstention_reason=None,
            metadata={"a": 2},  # type: ignore[dict-item]
        )


def test_to_trajectory_propagates_trajectory_refusal() -> None:
    outcome = EpisodeOutcome(
        trajectory_turns=(_generated_turn(),),
        prompt_turns=(_prompt_turn(),),
        termination=Termination.STOP,
        submitted_answer=None,
        abstention_reason=None,
    )
    with pytest.raises(ValueError):
        to_trajectory(
            outcome,
            uid="u1",
            session_id="s1",
            harness="h",
            reward=1.0,
            abstention_reason="both",
        )


def test_to_trajectory_declares_policy_versions() -> None:
    outcome = EpisodeOutcome(
        trajectory_turns=(_generated_turn(),),
        prompt_turns=(_prompt_turn(),),
        termination=Termination.STOP,
        submitted_answer=None,
        abstention_reason=None,
    )
    trajectory = to_trajectory(
        outcome,
        uid="u1",
        session_id="s1",
        harness="h",
        reward=1.0,
        abstention_reason=None,
        versions=(2, 5),
    )
    assert trajectory.policy_version_min == 2
    assert trajectory.policy_version_max == 5


def test_to_trajectory_infra_outcome_without_reward() -> None:
    # Trajectory.turns must be non-empty structurally (prompt + response sides both
    # present) even under INFRA, which only exempts the separate "at least one
    # GENERATED turn" rule -- loss_mask() still zeroes it out either way.
    outcome = EpisodeOutcome(
        trajectory_turns=(_generated_turn(),),
        prompt_turns=(_prompt_turn(),),
        termination=Termination.INFRA,
        submitted_answer=None,
        abstention_reason="infra:transport",
    )
    trajectory = to_trajectory(
        outcome,
        uid="u1",
        session_id="s1",
        harness="h",
        reward=None,
        abstention_reason=outcome.abstention_reason,
    )
    assert trajectory.is_infra is True


def test_native_tool_loop_rejects_empty_tools() -> None:
    with pytest.raises(NativeToolLoopRefusal):
        _loop(tools={})


def test_native_tool_loop_rejects_empty_name() -> None:
    with pytest.raises(NativeToolLoopRefusal):
        _loop(name="")


def test_native_tool_loop_rejects_non_str_name() -> None:
    with pytest.raises(NativeToolLoopRefusal):
        _loop(name=5)  # type: ignore[arg-type]


def test_native_tool_loop_rejects_empty_observation_role() -> None:
    with pytest.raises(NativeToolLoopRefusal):
        _loop(observation_role="")


def test_native_tool_loop_rejects_non_str_observation_role() -> None:
    with pytest.raises(NativeToolLoopRefusal):
        _loop(observation_role=5)  # type: ignore[arg-type]


def test_exec_result_timed_out_marker_in_observation() -> None:
    tokenizer = ToyTokenizer()
    loop = _loop(tokenizer=tokenizer)
    client = ScriptedClient(
        [_generation(_bash_call_text("sleep")), _generation(_submit_call_text("done"))]
    )

    def _timed_out(command: str) -> ExecResult:
        return ExecResult(
            stdout="partial",
            stderr="",
            exit_code=None,
            timed_out=True,
            truncated_bytes=0,
            seconds=1.0,
        )

    env = FakeEnvironment(execute_fn=_timed_out)
    outcome = _run(loop, _task(), client, env, _budget())
    decoded = tokenizer.decode(outcome.trajectory_turns[1].token_ids)
    assert "timed out" in decoded


def test_exec_result_truncated_bytes_marker_in_observation() -> None:
    tokenizer = ToyTokenizer()
    loop = _loop(tokenizer=tokenizer)
    client = ScriptedClient(
        [_generation(_bash_call_text("big")), _generation(_submit_call_text("done"))]
    )

    def _truncated(command: str) -> ExecResult:
        return ExecResult(
            stdout="head",
            stderr="",
            exit_code=0,
            timed_out=False,
            truncated_bytes=123,
            seconds=0.1,
        )

    env = FakeEnvironment(execute_fn=_truncated)
    outcome = _run(loop, _task(), client, env, _budget())
    decoded = tokenizer.decode(outcome.trajectory_turns[1].token_ids)
    assert "bytes elided" in decoded


def test_response_budget_exhausted_before_first_token() -> None:
    loop = _loop()
    client = ScriptedClient([])
    env = FakeEnvironment()
    outcome = _run(
        loop,
        _task(),
        client,
        env,
        _budget(step_limit=5, max_response_tokens=1, max_observation_chars=256),
    )
    assert outcome.termination is Termination.LENGTH
    assert len(outcome.trajectory_turns) == 0


def test_registered_tool_without_execution_binding_reports_error() -> None:
    from foundationscale.agentic_rl.tools import QwenXmlToolCallParser

    tokenizer = ToyTokenizer()
    noop = ToolSpec(name="noop", description="does nothing", parameters={})
    tools = {**default_tools(), "noop": noop}
    loop = _loop(tools=tools, tokenizer=tokenizer, parser=QwenXmlToolCallParser())
    noop_call = (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "noop"
        + markup.TAG_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )
    client = ScriptedClient([_generation(noop_call), _generation(_submit_call_text("ok"))])
    env = FakeEnvironment()
    outcome = _run(loop, _task(), client, env, _budget())
    observation = outcome.trajectory_turns[1]
    assert observation.kind is SegmentKind.TOOL_RESULT
    decoded = tokenizer.decode(observation.token_ids)
    assert "no execution binding" in decoded
    assert outcome.termination is Termination.STOP


def test_nameless_call_markup_is_recorded_not_a_crash() -> None:
    # Regression from the first GPU smoke run: the policy emitted call markup
    # with no function name; building ToolCall(name="") aborted the whole run.
    tokenizer = ToyTokenizer()
    loop = _loop(tokenizer=tokenizer)
    nameless = (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + markup.TAG_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )
    client = ScriptedClient([_generation(nameless), _generation(_submit_call_text("ok"))])
    env = FakeEnvironment()
    outcome = _run(loop, _task(), client, env, _budget())
    first = outcome.trajectory_turns[0]
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].name is None
    assert first.tool_calls[0].parse_error
    assert env.commands == []
    assert outcome.termination is Termination.STOP
