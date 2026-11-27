"""CPU-only tests for ARL-17 slice 2 context compaction and segment outcomes."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from foundationscale.agentic_rl.contracts import (
    SEGMENT_SEPARATOR,
    SegmentKind,
    Termination,
    Turn,
)
from foundationscale.agentic_rl.harness.base import (
    EpisodeCheckpoint,
    EpisodeOutcome,
    Generation,
    GenerationClient,
    HarnessRefusal,
    SamplingParams,
    SegmentOutcome,
    segment_trajectories,
    to_trajectory,
)
from foundationscale.agentic_rl.harness.native_tool_loop import (
    SUMMARIZE_INSTRUCTION,
    SUMMARY_PREFIX,
    CompactionPolicy,
    NativeToolLoopRefusal,
    _compacted_messages,
)

# tests/ is not a package. Load the sibling module explicitly so its deterministic
# tokenizer, engine, environment, and scenario builders remain the single source of truth.
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_arl17_native_tool_loop_helpers",
    Path(__file__).with_name("test_native_tool_loop.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError("could not load test_native_tool_loop.py")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPERS
_HELPER_SPEC.loader.exec_module(_HELPERS)

ToyTokenizer = _HELPERS.ToyTokenizer
ScriptedClient = _HELPERS.ScriptedClient
ExplodingEngineClient = _HELPERS.ExplodingEngineClient
FakeEnvironment = _HELPERS.FakeEnvironment
_bash_call_text = _HELPERS._bash_call_text
_submit_call_text = _HELPERS._submit_call_text
_generation = _HELPERS._generation
_task = _HELPERS._task
_loop = _HELPERS._loop
_budget = _HELPERS._budget
_exec_result = _HELPERS._exec_result
_run = _HELPERS._run


class RecordingClient:
    """ScriptedClient plus the exact contexts and sampling parameters it received."""

    def __init__(self, generations: Sequence[Generation]) -> None:
        self._client: GenerationClient = ScriptedClient(generations)
        self.contexts: list[tuple[int, ...]] = []
        self.samplings: list[SamplingParams] = []
        self.generations: list[Generation] = []

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        self.contexts.append(tuple(prompt_ids))
        self.samplings.append(sampling)
        generation = await self._client.generate(prompt_ids, sampling)
        self.generations.append(generation)
        return generation


def _compaction(mode: str, *, max_segments: int = 4) -> CompactionPolicy:
    # keep_turns=0: a compacted toy prompt (task + summary) is shorter than the context it
    # replaces, so every compaction shrinks and none trips the no-shrink stop.
    return CompactionPolicy(
        mode=mode,
        trigger_tokens=1,
        keep_turns=0,
        max_segments=max_segments,
        summary_max_tokens=16,
    )


def _tokens(turns: Sequence[Turn]) -> tuple[int, ...]:
    return tuple(token_id for turn in turns for token_id in turn.token_ids)


def _all_turns(outcome: EpisodeOutcome) -> Iterator[Turn]:
    for segment in outcome.segments:
        yield from segment.prompt_turns
        yield from segment.trajectory_turns
    yield from outcome.prompt_turns
    yield from outcome.trajectory_turns


def _normalized(text: str) -> str:
    """ToyTokenizer renders whitespace-separated words and decodes with single spaces."""
    return " ".join(text.split())


def test_compaction_none_is_identical_to_an_unmodified_loop() -> None:
    generations = [
        _generation(_bash_call_text("echo hi")),
        _generation(_submit_call_text("done")),
    ]

    baseline = _run(
        _loop(),
        _task(),
        ScriptedClient(generations),
        FakeEnvironment(execute_fn=lambda command: _exec_result("hi\n")),
        _budget(),
    )
    explicit_none = _run(
        dataclasses.replace(_loop(), compaction=None),
        _task(),
        ScriptedClient(generations),
        FakeEnvironment(execute_fn=lambda command: _exec_result("hi\n")),
        _budget(),
    )

    assert explicit_none.segments == ()
    assert explicit_none == baseline


def test_external_compaction_is_off_record_and_prompts_match_client_contexts() -> None:
    tokenizer = ToyTokenizer()
    summary = _generation("offline summary for the next segment")
    client = RecordingClient(
        [
            _generation(_bash_call_text("echo hi")),
            summary,
            _generation(_submit_call_text("done")),
        ]
    )
    loop = _loop(tokenizer=tokenizer, compaction=_compaction("external"))
    env = FakeEnvironment(execute_fn=lambda command: _exec_result("hi\n"))

    outcome = _run(loop, _task(), client, env, _budget())

    assert outcome.termination is Termination.STOP
    assert len(outcome.segments) == 1
    assert len(client.contexts) == 3
    assert env.commands == ["echo hi"]

    # The summary used a separate generation and never became any kind of turn.
    assert all(turn.token_ids != summary.token_ids for turn in _all_turns(outcome))
    assert client.samplings[1].max_new_tokens == 16
    assert _normalized(SUMMARIZE_INSTRUCTION) in tokenizer.decode(client.contexts[1])

    # Every prompt is non-generated, and its tokens are exactly what the engine saw.
    prompts = [outcome.prompt_turns]
    prompts.extend(segment.prompt_turns for segment in outcome.segments)
    for prompt_turns in prompts:
        assert prompt_turns
        assert not any(turn.generated for turn in prompt_turns)
    assert client.contexts[0] == _tokens(outcome.segments[0].prompt_turns)
    assert client.contexts[2] == _tokens(outcome.prompt_turns)

    rendered_new_prompt = tokenizer.decode(client.contexts[2])
    task_and_summary = f"do the thing\n\n{SUMMARY_PREFIX}{summary.text}"
    assert _normalized(task_and_summary) in rendered_new_prompt


def test_policy_summary_is_trained_and_its_tool_call_is_not_executed() -> None:
    tokenizer = ToyTokenizer()
    summary = _generation(
        "policy summary before an illegal call " + _bash_call_text("echo MUST_NOT_RUN")
    )
    client = RecordingClient(
        [
            _generation(_bash_call_text("echo first")),
            summary,
            _generation(_submit_call_text("done")),
        ]
    )
    loop = _loop(tokenizer=tokenizer, compaction=_compaction("policy"))
    env = FakeEnvironment(execute_fn=lambda command: _exec_result("first\n"))

    outcome = _run(loop, _task(), client, env, _budget())

    assert outcome.termination is Termination.STOP
    assert len(outcome.segments) == 1
    closing = outcome.segments[0]
    summary_turn = closing.trajectory_turns[-1]
    assert summary_turn.generated
    assert summary_turn.kind is SegmentKind.ASSISTANT
    assert summary_turn.token_ids == summary.token_ids

    observation = closing.trajectory_turns[-2]
    assert not observation.generated
    assert _normalized(SUMMARIZE_INSTRUCTION) in tokenizer.decode(observation.token_ids)

    # Only the original policy tool call reached the environment.
    assert env.commands == ["echo first"]
    assert len(client.contexts) == 3
    assert client.contexts[2] == _tokens(outcome.prompt_turns)


def test_max_segments_caps_external_compaction() -> None:
    client = RecordingClient(
        [
            _generation(_bash_call_text("echo zero")),
            _generation("first external summary"),
            _generation(_bash_call_text("echo one")),
            _generation(_bash_call_text("echo two")),
            _generation(_submit_call_text("done")),
        ]
    )
    loop = _loop(compaction=_compaction("external", max_segments=2))
    env = FakeEnvironment(execute_fn=lambda command: _exec_result("ok\n"))

    outcome = _run(loop, _task(), client, env, _budget())

    assert outcome.termination is Termination.STOP
    assert len(outcome.segments) == 1
    assert env.commands == ["echo zero", "echo one", "echo two"]


def test_step_budget_is_shared_across_segments() -> None:
    # step_limit=4: no compaction before step 0 (nothing generated) or before the LAST step
    # (it would leave an empty final segment), so steps 1 and 2 compact: 2 closed segments.
    policy_generations = [_generation(_bash_call_text(f"echo {index}")) for index in range(4)]
    summaries = [_generation(f"summary {index}") for index in range(2)]
    client = RecordingClient(
        [
            policy_generations[0],
            summaries[0],
            policy_generations[1],
            summaries[1],
            policy_generations[2],
            policy_generations[3],
        ]
    )
    budget = _budget(step_limit=4)
    env = FakeEnvironment(execute_fn=lambda command: _exec_result("ok " * 60))

    outcome = _run(
        _loop(compaction=_compaction("external")),
        _task(),
        client,
        env,
        budget,
    )

    policy_ids = {generation.token_ids for generation in policy_generations}
    policy_generation_count = sum(
        generation.token_ids in policy_ids for generation in client.generations
    )
    assert outcome.termination is Termination.STEP_LIMIT
    assert len(outcome.segments) == 2
    assert len(client.generations) == 6  # four policy steps plus two summaries
    assert policy_generation_count == 4 == budget.step_limit


def test_resume_with_compaction_is_refused() -> None:
    checkpoints: list[EpisodeCheckpoint] = []

    async def capture(checkpoint: EpisodeCheckpoint) -> None:
        checkpoints.append(checkpoint)

    outcome = asyncio.run(
        _loop().run_episode(
            _task(),
            client=ScriptedClient(
                [
                    _generation(_bash_call_text("echo hi")),
                    _generation(_submit_call_text("done")),
                ]
            ),
            env=FakeEnvironment(execute_fn=lambda command: _exec_result("hi\n")),
            budget=_budget(),
            checkpoint_hook=capture,
        )
    )
    assert outcome.termination is Termination.STOP
    assert checkpoints

    compacted = _loop(compaction=_compaction("external"))
    with pytest.raises(NativeToolLoopRefusal, match="checkpoints .resume"):
        asyncio.run(
            compacted.run_episode(
                _task(),
                client=ScriptedClient([]),
                env=FakeEnvironment(),
                budget=_budget(),
                resume=checkpoints[-1],
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "internal"},
        {"trigger_tokens": 0},
        {"keep_turns": -1},
        {"max_segments": 1},
        {"trigger_tokens": True},
        {"keep_turns": False},
        {"summary_max_tokens": 0},
        {"summary_max_tokens": True},
    ],
)
def test_compaction_policy_refuses_bad_values(overrides: dict[str, object]) -> None:
    values: dict[str, Any] = {
        "mode": "external",
        "trigger_tokens": 10,
        "keep_turns": 2,
        "max_segments": 4,
        "summary_max_tokens": 16,
    }
    values.update(overrides)
    with pytest.raises(NativeToolLoopRefusal, match="CompactionPolicy"):
        CompactionPolicy(**values)


def test_compacted_messages_system_summary_tail_and_missing_user() -> None:
    history: list[dict[str, Any]] = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "first action"},
        {"role": "user", "content": "first observation"},
        {"role": "assistant", "content": "second action"},
        {"role": "user", "content": "second observation"},
    ]

    compacted = _compacted_messages(history, "remember this", keep_turns=3)

    assert [message["role"] for message in compacted] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert compacted[0] == history[0]
    assert compacted[1]["content"] == (f"do the thing\n\n{SUMMARY_PREFIX}remember this")
    assert compacted[2:] == history[-2:]
    assert compacted[2]["role"] == "assistant"
    # The input messages are copied rather than mutated.
    assert history[1]["content"] == "do the thing"

    no_tail = _compacted_messages(history, "remember this", keep_turns=0)
    assert [message["role"] for message in no_tail] == ["system", "user"]
    assert SUMMARY_PREFIX in str(no_tail[1]["content"])

    with pytest.raises(NativeToolLoopRefusal, match="no user message"):
        _compacted_messages(
            ({"role": "assistant", "content": "no task here"},),
            "remember this",
            keep_turns=2,
        )


def test_external_summary_engine_failure_is_an_infra_episode() -> None:
    client = ExplodingEngineClient(
        [_generation(_bash_call_text("echo hi"))],
        kind="transport",
        message="summary connection failed",
    )
    outcome = _run(
        _loop(compaction=_compaction("external")),
        _task(),
        client,
        FakeEnvironment(execute_fn=lambda command: _exec_result("hi\n")),
        _budget(),
    )

    assert outcome.termination is Termination.INFRA
    assert outcome.abstention_reason == "infra:engine_transport"
    assert outcome.segments == ()


def test_empty_post_compaction_segment_reopens_the_last_closed_segment() -> None:
    # The summary succeeds, but the first policy generation in the fresh segment fails.
    # The empty fresh segment must not become the outcome's final segment.
    client = ExplodingEngineClient(
        [
            _generation(_bash_call_text("echo hi")),
            _generation("summary before engine failure"),
        ],
        kind="transport",
    )
    outcome = _run(
        _loop(compaction=_compaction("external")),
        _task(),
        client,
        FakeEnvironment(execute_fn=lambda command: _exec_result("hi\n")),
        _budget(),
    )

    assert outcome.termination is Termination.INFRA
    assert outcome.segments == ()
    assert any(turn.generated for turn in outcome.trajectory_turns)


def test_segment_outcome_and_episode_segments_validation() -> None:
    outcome = _run(
        _loop(),
        _task(),
        ScriptedClient(
            [
                _generation(_bash_call_text("echo hi")),
                _generation(_submit_call_text("done")),
            ]
        ),
        FakeEnvironment(execute_fn=lambda command: _exec_result("hi\n")),
        _budget(),
    )
    generated = next(turn for turn in outcome.trajectory_turns if turn.generated)

    with pytest.raises(HarnessRefusal, match="prompt_turns"):
        SegmentOutcome(prompt_turns=(), trajectory_turns=(generated,))
    with pytest.raises(HarnessRefusal, match="prompt_turns"):
        SegmentOutcome(prompt_turns=(generated,), trajectory_turns=(generated,))
    with pytest.raises(HarnessRefusal, match="no generated turn"):
        SegmentOutcome(
            prompt_turns=outcome.prompt_turns,
            trajectory_turns=outcome.prompt_turns,
        )
    with pytest.raises(HarnessRefusal, match="segments"):
        dataclasses.replace(
            outcome,
            segments=cast(tuple[SegmentOutcome, ...], ("not a segment",)),
        )


def test_segment_trajectories_expands_rows_and_preserves_reward() -> None:
    session_id = "compact-session"
    client = RecordingClient(
        [
            _generation(_bash_call_text("echo zero")),
            _generation("summary zero"),
            _generation(_bash_call_text("echo one")),
            _generation("summary one"),
            _generation(_submit_call_text("done")),
        ]
    )
    outcome = _run(
        _loop(compaction=_compaction("external", max_segments=3)),
        _task(session_id=session_id),
        client,
        FakeEnvironment(execute_fn=lambda command: _exec_result("ok " * 60)),
        _budget(),
    )
    assert len(outcome.segments) == 2

    final = to_trajectory(
        outcome,
        harness="native_tool_loop",
        uid="u9",
        session_id=session_id,
        reward=0.75,
        abstention_reason=None,
        versions=(7, 7),
        extra_metadata={"base": "yes"},
    )
    rows = segment_trajectories(outcome, final)

    assert len(rows) == 3
    assert [row.session_id for row in rows] == [
        f"{session_id}{SEGMENT_SEPARATOR}{index}" for index in range(3)
    ]
    assert {row.reward for row in rows} == {0.75}
    assert [row.metadata["segment"] for row in rows] == ["0", "1", "2"]
    assert all(row.metadata["segments"] == "3" for row in rows)
    assert all(row.metadata["base"] == "yes" for row in rows)

    for index, segment in enumerate(outcome.segments):
        assert rows[index].prompt_turns == segment.prompt_turns
        assert rows[index].turns == segment.trajectory_turns
    assert rows[-1].prompt_turns == outcome.prompt_turns
    assert rows[-1].turns == outcome.trajectory_turns

    never_compacted = dataclasses.replace(outcome, segments=())
    assert segment_trajectories(never_compacted, final) == (final,)


def test_checkpoint_hook_with_compaction_is_refused() -> None:
    async def capture(checkpoint: EpisodeCheckpoint) -> None:
        del checkpoint

    with pytest.raises(NativeToolLoopRefusal, match="checkpoint"):
        asyncio.run(
            _loop(compaction=_compaction("external")).run_episode(
                _task(),
                client=ScriptedClient([_generation(_submit_call_text("done"))]),
                env=FakeEnvironment(execute_fn=lambda command: _exec_result("hi\n")),
                budget=_budget(),
                checkpoint_hook=capture,
            )
        )


def test_episode_token_cap_bounds_the_whole_episode_including_summaries() -> None:
    # Control-matched budget: the cap covers every segment's response tokens AND the external
    # summary tokens, so compaction never gets a fresh budget per segment.
    policy = CompactionPolicy(
        mode="external", trigger_tokens=1, keep_turns=0, max_segments=8, episode_tokens=12
    )
    client = RecordingClient([_generation(_bash_call_text(f"echo {i}")) for i in range(20)])
    outcome = _run(
        _loop(compaction=policy),
        _task(),
        client,
        FakeEnvironment(execute_fn=lambda command: _exec_result("ok\n")),
        _budget(step_limit=20),
    )
    assert outcome.termination is Termination.LENGTH
    assert len(client.generations) < 20  # the cap, not the step limit, ended the episode
    response = sum(
        len(t.token_ids) for seg in (*outcome.segments, outcome) for t in seg.trajectory_turns
    )
    summary_tokens = sum(len(g.token_ids) for g in client.generations) - sum(
        len(t.token_ids)
        for seg in (*outcome.segments, outcome)
        for t in seg.trajectory_turns
        if t.generated
    )
    assert response + summary_tokens >= 12  # the budget really was consumed


def test_episode_tokens_is_validated() -> None:
    for bad in (0, -3, True, 1.5):
        with pytest.raises(NativeToolLoopRefusal, match="episode_tokens"):
            CompactionPolicy(mode="external", trigger_tokens=1, episode_tokens=bad)  # type: ignore[arg-type]


def test_a_summary_that_spends_the_rest_of_the_episode_budget_ends_it() -> None:
    first = _generation(_bash_call_text("echo 0"))
    long_summary = _generation("summary " * 40)
    policy = CompactionPolicy(
        mode="external",
        trigger_tokens=1,
        keep_turns=0,
        max_segments=8,
        episode_tokens=len(first.token_ids) + 8,
    )
    client = RecordingClient([first, long_summary, _generation(_submit_call_text("x"))])
    outcome = _run(
        _loop(compaction=policy),
        _task(),
        client,
        FakeEnvironment(execute_fn=lambda command: _exec_result("ok\n")),
        _budget(step_limit=10),
    )
    assert outcome.termination is Termination.LENGTH
    assert len(client.generations) == 2  # the policy never generated after the summary


def test_a_compaction_that_does_not_shrink_the_context_stops_compacting() -> None:
    # keep_turns large + a long summary: the "compacted" prompt is no shorter than before, so
    # the loop compacts once and then stops instead of re-triggering every step.
    policy = CompactionPolicy(mode="external", trigger_tokens=1, keep_turns=20, max_segments=8)
    script: list[Generation] = []
    for i in range(6):
        script += [_generation(_bash_call_text(f"echo {i}")), _generation("summary " * 30)]
    client = RecordingClient(script)
    outcome = _run(
        _loop(compaction=policy),
        _task(),
        client,
        FakeEnvironment(execute_fn=lambda command: _exec_result("ok\n")),
        _budget(step_limit=5),
    )
    assert len(outcome.segments) == 1
