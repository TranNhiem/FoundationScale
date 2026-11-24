import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl.contracts import (
    SegmentKind,
    Termination,
    ToolCall,
    Trajectory,
    Turn,
)
from foundationscale.agentic_rl.fs_trace import (
    TRACE_SCHEMA_VERSION,
    ForkPoint,
    Span,
    SpanKind,
    TraceRefusal,
    TraceWriter,
    fold,
    fork_prefix,
    read_trace,
    trace_from_trajectory,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_episode(path: Path, spans: tuple[Span, ...]) -> None:
    TraceWriter(path).write_episode(spans)


@pytest.fixture
def rich_trajectory() -> Trajectory:
    prompt = Turn(
        index=0,
        kind=SegmentKind.SYSTEM,
        token_ids=(1, 2, 3),
        generated=False,
        logprobs=None,
    )
    assistant = Turn(
        index=1,
        kind=SegmentKind.ASSISTANT,
        token_ids=(10, 11, 12, 13),
        generated=True,
        logprobs=(-0.25, -0.5, -0.75, -1.0),
        tool_calls=(
            ToolCall(
                name="lookup",
                arguments='{"q": "arl", "n": 2}',
                call_id="call-0001",
                parse_error=None,
            ),
            ToolCall(
                name=None,
                arguments="{not json",
                call_id=None,
                parse_error="call markup has no function name",
            ),
        ),
    )
    tool_result = Turn(
        index=2,
        kind=SegmentKind.TOOL_RESULT,
        token_ids=(20, 21),
        generated=False,
        logprobs=None,
    )
    final = Turn(
        index=3,
        kind=SegmentKind.ASSISTANT,
        token_ids=(30, 31),
        generated=True,
        logprobs=None,
        repetition_hit=True,
    )
    return Trajectory(
        uid="uid-7",
        session_id="session-42",
        harness="harness-a",
        prompt_turns=(prompt,),
        turns=(assistant, tool_result, final),
        reward=0.75,
        abstention_reason=None,
        termination=Termination.STOP,
        policy_version_min=3,
        policy_version_max=4,
        metadata={"host": "worker-7", "attempt": "one"},
    )


@pytest.fixture
def infra_trajectory() -> Trajectory:
    return Trajectory(
        uid="uid-infra",
        session_id="session-infra",
        harness="harness-b",
        prompt_turns=(),
        turns=(),
        reward=None,
        abstention_reason="infra: environment never started",
        termination=Termination.INFRA,
        policy_version_min=None,
        policy_version_max=None,
        metadata={"pod": "worker-a"},
    )


def test_trace_layout_verifier_reward_and_exact_fold_parity(
    rich_trajectory: Trajectory,
) -> None:
    verifier_metadata = {"status": "passed", "detail": "exact"}
    spans = trace_from_trajectory(
        rich_trajectory,
        run_id="run-layout",
        rollout_step=9,
        verifier_metadata=verifier_metadata,
        reward_note="scored by the exact checker",
    )
    verifier_metadata["status"] = "mutated-after-tracing"

    assert isinstance(spans, tuple)
    assert [span.seq for span in spans] == list(range(len(spans)))
    assert [span.kind for span in spans] == [
        SpanKind.SESSION,
        SpanKind.TURN,
        SpanKind.TURN,
        SpanKind.TOOL_CALL,
        SpanKind.TOOL_CALL,
        SpanKind.TURN,
        SpanKind.TURN,
        SpanKind.VERIFIER_RUN,
        SpanKind.REWARD,
    ]
    session = spans[0]
    assert session.parent_span_id is None
    assert session.data == {
        "uid": "uid-7",
        "session_id": "session-42",
        "harness": "harness-a",
        "termination": "stop",
        "reward": 0.75,
        "abstention_reason": None,
        "policy_version_min": 3,
        "policy_version_max": 4,
        "metadata": {"host": "worker-7", "attempt": "one"},
    }

    turn_spans = [span for span in spans if span.kind is SpanKind.TURN]
    assert [(span.data["phase"], span.data["index"]) for span in turn_spans] == [
        ("prompt", 0),
        ("response", 1),
        ("response", 2),
        ("response", 3),
    ]
    assert turn_spans[0].data["token_ids"] == [1, 2, 3]
    assert turn_spans[0].data["logprobs"] is None
    assert turn_spans[1].data["logprobs"] == [-0.25, -0.5, -0.75, -1.0]
    assert turn_spans[-1].data["repetition_hit"] is True
    assert all(span.parent_span_id == session.span_id for span in turn_spans)

    tool_spans = [span for span in spans if span.kind is SpanKind.TOOL_CALL]
    assert [span.data["k"] for span in tool_spans] == [0, 1]
    assert all(span.parent_span_id == turn_spans[1].span_id for span in tool_spans)
    assert tool_spans[0].data["arguments"] == '{"q": "arl", "n": 2}'
    assert tool_spans[0].data["call_id"] == "call-0001"
    assert tool_spans[0].data["parse_error"] is None
    assert tool_spans[1].data["name"] is None
    assert tool_spans[1].data["arguments"] == "{not json"
    assert tool_spans[1].data["parse_error"] == "call markup has no function name"

    verifier = spans[-2]
    assert verifier.parent_span_id == session.span_id
    assert verifier.data == {
        "checked": {"status": "passed", "detail": "exact"},
        "target_span_id": session.span_id,
    }
    reward = spans[-1]
    assert reward.parent_span_id == session.span_id
    assert reward.data == {
        "value": 0.75,
        "abstention_reason": None,
        "note": "scored by the exact checker",
        "target_span_id": session.span_id,
        "verifier_span_id": verifier.span_id,
    }
    assert fold(spans) == rich_trajectory


def test_infra_zero_turn_trace_has_exact_fold_parity(
    infra_trajectory: Trajectory,
) -> None:
    spans = trace_from_trajectory(
        infra_trajectory,
        run_id="run-infra",
        rollout_step=0,
    )

    assert [span.kind for span in spans] == [SpanKind.SESSION, SpanKind.REWARD]
    assert spans[0].data["reward"] is None
    assert spans[0].data["abstention_reason"] == "infra: environment never started"
    assert spans[-1].data["value"] is None
    assert spans[-1].data["verifier_span_id"] is None
    assert fold(spans) == infra_trajectory


def test_ids_are_deterministic_and_invocation_changes_trace_id(
    rich_trajectory: Trajectory,
) -> None:
    first = trace_from_trajectory(
        rich_trajectory,
        run_id="run-A",
        rollout_step=17,
        invocation=0,
    )
    second = trace_from_trajectory(
        rich_trajectory,
        run_id="run-A",
        rollout_step=17,
        invocation=0,
    )
    assert first == second

    expected_trace_id = "tr_" + _digest(f"run-A\0{17}\0{rich_trajectory.session_id}\0{0}")[:32]
    expected_branch_id = "br_" + _digest(expected_trace_id + "\0main")[:32]
    assert all(span.trace_id == expected_trace_id for span in first)
    assert all(span.branch_id == expected_branch_id for span in first)

    def expected_span_id(logical_key: str) -> str:
        material = expected_trace_id + "\0" + expected_branch_id + "\0" + logical_key
        return "sp_" + _digest(material)[:32]

    assert first[0].span_id == expected_span_id("session")
    assert first[1].span_id == expected_span_id("turn:0")
    assert first[3].span_id == expected_span_id("tool_call:1:0")
    assert first[4].span_id == expected_span_id("tool_call:1:1")
    assert first[-1].span_id == expected_span_id("reward")

    other_invocation = trace_from_trajectory(
        rich_trajectory,
        run_id="run-A",
        rollout_step=17,
        invocation=1,
    )
    assert other_invocation[0].trace_id != expected_trace_id
    assert other_invocation[0].branch_id != expected_branch_id


def test_span_json_is_canonical_round_trips_and_refuses_nan(
    rich_trajectory: Trajectory,
) -> None:
    spans = trace_from_trajectory(
        rich_trajectory,
        run_id="run-json",
        rollout_step=2,
    )
    for span in spans:
        encoded = span.to_json()
        assert span.schema_version == TRACE_SCHEMA_VERSION
        assert "NaN" not in encoded
        assert Span.from_json(encoded) == span
        assert encoded == json.dumps(
            json.loads(encoded),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    nan_span = replace(spans[-1], data={"value": float("nan")})
    with pytest.raises(ValueError):
        nan_span.to_json()


def test_writer_and_reader_round_trip_preserves_file_order(
    tmp_path: Path,
    rich_trajectory: Trajectory,
) -> None:
    first = trace_from_trajectory(
        rich_trajectory,
        run_id="run-io",
        rollout_step=5,
        invocation=0,
    )
    second = trace_from_trajectory(
        rich_trajectory,
        run_id="run-io",
        rollout_step=5,
        invocation=1,
    )
    path = tmp_path / "episodes.jsonl"
    writer = TraceWriter(path)
    writer.write_episode(first)
    writer.write_episode(second)

    grouped = read_trace(path)
    assert list(grouped) == [first[0].trace_id, second[0].trace_id]
    assert grouped[first[0].trace_id] == first
    assert grouped[second[0].trace_id] == second
    assert path.read_bytes().endswith(b"\n")


def test_sequence_gap_is_refused_when_folding_reader_output(
    tmp_path: Path,
    rich_trajectory: Trajectory,
) -> None:
    spans = trace_from_trajectory(
        rich_trajectory,
        run_id="run-gap",
        rollout_step=1,
    )
    path = tmp_path / "gap.jsonl"
    _write_episode(path, spans[:1] + spans[2:])

    loaded = read_trace(path)[spans[0].trace_id]
    with pytest.raises(TraceRefusal) as caught:
        fold(loaded)
    assert "seq" in str(caught.value).lower()


def test_unknown_kind_is_refused_with_its_line_number(
    tmp_path: Path,
    rich_trajectory: Trajectory,
) -> None:
    spans = trace_from_trajectory(
        rich_trajectory,
        run_id="run-kind",
        rollout_step=1,
    )
    malformed = json.loads(spans[1].to_json())
    malformed["kind"] = "not-a-span-kind"
    path = tmp_path / "unknown-kind.jsonl"
    path.write_text(
        spans[0].to_json() + "\n" + json.dumps(malformed) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TraceRefusal) as caught:
        read_trace(path)
    message = str(caught.value)
    assert "2" in message
    assert "kind" in message.lower()
    assert "not-a-span-kind" in message


def test_dangling_parent_is_refused_when_folding_reader_output(
    tmp_path: Path,
    rich_trajectory: Trajectory,
) -> None:
    spans = trace_from_trajectory(
        rich_trajectory,
        run_id="run-parent",
        rollout_step=1,
    )
    dangling = (spans[0], replace(spans[1], parent_span_id="sp_missing")) + spans[2:]
    path = tmp_path / "dangling-parent.jsonl"
    _write_episode(path, dangling)

    loaded = read_trace(path)[spans[0].trace_id]
    with pytest.raises(TraceRefusal) as caught:
        fold(loaded)
    message = str(caught.value)
    assert "parent" in message.lower()
    assert "sp_missing" in message


def test_partial_tail_requires_opt_in_and_must_be_unterminated(
    tmp_path: Path,
    rich_trajectory: Trajectory,
) -> None:
    spans = trace_from_trajectory(
        rich_trajectory,
        run_id="run-tail",
        rollout_step=1,
    )
    path = tmp_path / "partial-tail.jsonl"
    tail = '{"schema_version":' + json.dumps(TRACE_SCHEMA_VERSION)
    path.write_text(spans[0].to_json() + "\n" + tail, encoding="utf-8")

    with pytest.raises(TraceRefusal) as caught:
        read_trace(path)
    assert "2" in str(caught.value)

    grouped = read_trace(path, allow_partial_tail=True)
    assert grouped == {spans[0].trace_id: (spans[0],)}

    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(TraceRefusal) as caught:
        read_trace(path, allow_partial_tail=True)
    assert "2" in str(caught.value)


def test_writer_size_refusal_names_record_and_limit_sizes(
    tmp_path: Path,
    rich_trajectory: Trajectory,
) -> None:
    spans = trace_from_trajectory(
        rich_trajectory,
        run_id="run-size",
        rollout_step=1,
    )
    path = tmp_path / "oversize.jsonl"
    writer = TraceWriter(path, max_record_bytes=48)

    with pytest.raises(TraceRefusal) as caught:
        writer.write_episode(spans[:1])

    message = str(caught.value)
    record_size = len(spans[0].to_json().encode("utf-8"))
    assert "48" in message
    assert str(record_size) in message or str(record_size + 1) in message
    assert not path.exists() or path.read_bytes() == b""


def test_fork_prefix_tokens_identity_and_turn_count(
    rich_trajectory: Trajectory,
) -> None:
    spans = trace_from_trajectory(
        rich_trajectory,
        run_id="run-fork",
        rollout_step=8,
    )
    point = fork_prefix(spans, turn_index=2)

    assert isinstance(point, ForkPoint)
    assert point.parent_trace_id == spans[0].trace_id
    assert point.parent_branch_id == spans[0].branch_id
    expected_branch = "br_" + _digest(f"{spans[0].trace_id}\0fork\0{2}")[:32]
    assert point.branch_id == expected_branch
    assert point.prompt_token_ids == (1, 2, 3, 10, 11, 12, 13)
    assert point.turns_kept == 1


@pytest.mark.parametrize("turn_index", [0, 99])
def test_fork_prefix_refuses_non_response_turn_indices(
    rich_trajectory: Trajectory,
    turn_index: int,
) -> None:
    spans = trace_from_trajectory(
        rich_trajectory,
        run_id="run-fork-refusal",
        rollout_step=8,
    )

    with pytest.raises(TraceRefusal) as caught:
        fork_prefix(spans, turn_index=turn_index)
    message = str(caught.value)
    assert "turn_index" in message
    assert str(turn_index) in message


def test_rollout_host_writes_one_foldable_trace_per_trajectory(tmp_path: Path) -> None:
    """ARL-21 wiring: rollout.trace_dir writes fs.trace/v1 traces that fold back exactly."""
    from foundationscale.agentic_rl.fs_trace import fold, read_trace

    try:
        import test_engine_recovery as fakes  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - package-style collection
        from tests.agentic_rl import test_engine_recovery as fakes  # type: ignore[no-redef]

    trace_dir = tmp_path / "traces" / "run-a"
    host = fakes._host(
        tmp_path,
        uids=["taskA"],
        client_factory=lambda i: fakes.SubmitClient("xy"),
        trace_dir=str(trace_dir),
        gates=False,
    )
    host.rollout(3)
    traces = read_trace(trace_dir / "step_000003.jsonl")
    assert len(traces) == 2  # tasks_per_step=1 x group_size=2
    folded = sorted((fold(spans) for spans in traces.values()), key=lambda t: t.session_id)
    assert [t.session_id for t in folded] == ["taskA#0", "taskA#1"]
    assert all(t.reward is not None for t in folded)


@pytest.mark.parametrize("bad", ["", 7])
def test_rollout_host_refuses_an_empty_or_non_string_trace_dir(tmp_path: Path, bad: Any) -> None:
    from foundationscale.agentic_rl.rollout_host import RolloutHostRefusal

    try:
        import test_engine_recovery as fakes  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - package-style collection
        from tests.agentic_rl import test_engine_recovery as fakes  # type: ignore[no-redef]

    with pytest.raises(RolloutHostRefusal, match="trace_dir"):
        fakes._host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda i: fakes.SubmitClient("x"),
            trace_dir=bad,
        )
