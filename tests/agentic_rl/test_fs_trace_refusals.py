import json
from dataclasses import replace
from pathlib import Path

import pytest

from foundationscale.agentic_rl.contracts import (
    SegmentKind,
    Termination,
    ToolCall,
    Trajectory,
    Turn,
)
from foundationscale.agentic_rl.fs_trace import (
    Span,
    SpanKind,
    TraceRefusal,
    TraceWriter,
    fold,
    fork_prefix,
    read_trace,
    trace_from_trajectory,
)


def _trajectory() -> Trajectory:
    prompt = Turn(
        index=0,
        kind=SegmentKind.SYSTEM,
        token_ids=(1, 2),
        generated=False,
        logprobs=None,
    )
    response = Turn(
        index=1,
        kind=SegmentKind.ASSISTANT,
        token_ids=(3, 4),
        generated=True,
        logprobs=(-0.1, -0.2),
        tool_calls=(
            ToolCall(
                name="lookup",
                arguments="{}",
                call_id="call-1",
                parse_error=None,
            ),
        ),
    )
    return Trajectory(
        uid="uid-refusals",
        session_id="session-refusals",
        harness="harness-test",
        prompt_turns=(prompt,),
        turns=(response,),
        reward=1.0,
        abstention_reason=None,
        termination=Termination.STOP,
        policy_version_min=1,
        policy_version_max=1,
        metadata={"worker": "a"},
    )


def _spans() -> tuple[Span, ...]:
    return trace_from_trajectory(
        _trajectory(),
        run_id="run-refusals",
        rollout_step=0,
        verifier_metadata={"status": "passed"},
    )


def _data(span: Span, **changes: object) -> Span:
    data = dict(span.data)
    data.update(changes)
    return replace(span, data=data)


def _refusal(fragment: str, operation, *args, **kwargs) -> None:
    with pytest.raises(TraceRefusal) as caught:
        operation(*args, **kwargs)
    assert fragment in str(caught.value)


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"trace_id": ""}, "non-empty strings"),
        ({"schema_version": "fs.trace/v0"}, "schema_version"),
        ({"seq": True}, "bool is not an int"),
        ({"kind": "session"}, "legal set"),
        ({"data": []}, "mappings with string keys"),
        ({"data": {1: "bad"}}, "legal keys are strings"),
    ],
)
def test_span_field_refusals(changes: dict[str, object], fragment: str) -> None:
    _refusal(fragment, replace, _spans()[0], **changes)


def test_span_from_json_refusals() -> None:
    _refusal("JSON strings", Span.from_json, None)
    _refusal("JSON constant", Span.from_json, "NaN")
    _refusal("top-level value", Span.from_json, "[]")
    _refusal("missing=", Span.from_json, "{}")

    payload = json.loads(_spans()[0].to_json())
    payload["trace_id"] = ""
    _refusal("Span.from_json(): span", Span.from_json, json.dumps(payload))


def test_trace_from_trajectory_argument_refusals() -> None:
    traj = _trajectory()
    _refusal(
        "Trajectory instances",
        trace_from_trajectory,
        object(),
        run_id="run",
        rollout_step=0,
    )
    _refusal(
        "rollout_step",
        trace_from_trajectory,
        traj,
        run_id="run",
        rollout_step=True,
    )
    _refusal(
        "invocation",
        trace_from_trajectory,
        traj,
        run_id="run",
        rollout_step=0,
        invocation=-1,
    )
    _refusal(
        "verifier_metadata",
        trace_from_trajectory,
        traj,
        run_id="run",
        rollout_step=0,
        verifier_metadata=[],
    )
    _refusal(
        "reward_note",
        trace_from_trajectory,
        traj,
        run_id="run",
        rollout_step=0,
        reward_note=1,
    )


def test_validated_span_sequence_refusals() -> None:
    spans = _spans()
    _refusal("iterables of Span", fold, 1)
    _refusal("empty", fold, ())
    _refusal("position 0", fold, (object(),))
    _refusal("position 1", fold, (spans[0], object()))
    _refusal("seq 0", fold, (replace(spans[0], kind=SpanKind.TURN),) + spans[1:])

    wrong_trace = (spans[0], replace(spans[1], trace_id="tr_other")) + spans[2:]
    _refusal("trace_id", fold, wrong_trace)
    wrong_branch = (spans[0], replace(spans[1], branch_id="br_other")) + spans[2:]
    _refusal("branch_id", fold, wrong_branch)
    duplicate = (spans[0], replace(spans[1], span_id=spans[0].span_id)) + spans[2:]
    _refusal("unique", fold, duplicate)

    rooted = (replace(spans[0], parent_span_id="sp_other"),) + spans[1:]
    _refusal("SESSION", fold, rooted)
    two_sessions = (spans[0], replace(spans[1], kind=SpanKind.SESSION)) + spans[2:]
    _refusal("SESSION spans", fold, two_sessions)


def test_fold_exact_data_and_target_refusals() -> None:
    spans = _spans()
    empty_session = (replace(spans[0], data={}),) + spans[1:]
    _refusal("missing keys", fold, empty_session)

    bad_target = spans[:4] + (_data(spans[4], target_span_id=None), spans[5])
    _refusal("target_span_id", fold, bad_target)
    missing_target = spans[:5] + (_data(spans[5], target_span_id="sp_missing"),)
    _refusal("no span with that id", fold, missing_target)


def test_fold_turn_and_tool_call_refusals() -> None:
    spans = _spans()
    wrong_turn_parent = (
        spans[:2] + (replace(spans[2], parent_span_id=spans[1].span_id),) + spans[3:]
    )
    _refusal("only legal value is the SESSION", fold, wrong_turn_parent)

    bad_phase = (spans[0], _data(spans[1], phase="other")) + spans[2:]
    _refusal("legal set", fold, bad_phase)
    bad_tool_parent = spans[:3] + (replace(spans[3], parent_span_id=spans[0].span_id),)
    bad_tool_parent += spans[4:]
    _refusal("earlier TURN", fold, bad_tool_parent)

    bad_index = spans[:3] + (_data(spans[3], k=1),) + spans[4:]
    _refusal("next zero-based", fold, bad_index)
    bad_call = spans[:3] + (_data(spans[3], arguments=7),) + spans[4:]
    _refusal("legal ToolCall", fold, bad_call)


def test_fold_verifier_and_reward_refusals() -> None:
    spans = _spans()
    bad_verifier_parent = spans[:4] + (
        replace(spans[4], parent_span_id=spans[2].span_id),
        spans[5],
    )
    _refusal("VERIFIER_RUN", fold, bad_verifier_parent)
    bad_checked = spans[:4] + (_data(spans[4], checked=[]), spans[5])
    _refusal("checked", fold, bad_checked)

    bad_reward_parent = spans[:5] + (replace(spans[5], parent_span_id=spans[4].span_id),)
    _refusal("REWARD", fold, bad_reward_parent)
    bad_verifier_id = spans[:5] + (_data(spans[5], verifier_span_id="sp_missing"),)
    _refusal("verifier_span_id", fold, bad_verifier_id)
    wrong_kind = spans[:5] + (_data(spans[5], verifier_span_id=spans[1].span_id),)
    _refusal("legal kind", fold, wrong_kind)


def test_fold_reconstruction_refusals() -> None:
    spans = _spans()
    bad_kind = (spans[0], _data(spans[1], kind="unknown")) + spans[2:]
    _refusal("legal set", fold, bad_kind)
    bad_tokens = (spans[0], _data(spans[1], token_ids=(1, 2))) + spans[2:]
    _refusal("token_ids", fold, bad_tokens)
    bad_logprobs = (spans[0], _data(spans[1], logprobs={})) + spans[2:]
    _refusal("logprobs", fold, bad_logprobs)
    bad_turn = (spans[0], _data(spans[1], index=-1)) + spans[2:]
    _refusal("legal Turn", fold, bad_turn)

    bad_termination = (_data(spans[0], termination="finished"),) + spans[1:]
    _refusal("termination", fold, bad_termination)
    bad_metadata = (_data(spans[0], metadata=[]),) + spans[1:]
    _refusal("metadata", fold, bad_metadata)
    bad_trajectory = (_data(spans[0], uid=None),) + spans[1:]
    _refusal("legal Trajectory", fold, bad_trajectory)


def test_fork_prefix_refusals() -> None:
    spans = _spans()
    bad_phase = (spans[0], _data(spans[1], phase="other")) + spans[2:]
    _refusal("fork_prefix(): TURN", fork_prefix, bad_phase, turn_index=1)

    bad_index = spans[:2] + (_data(spans[2], index=-1),) + spans[3:]
    _refusal("real ints >= 0", fork_prefix, bad_index, turn_index=1)
    bad_tokens = (spans[0], _data(spans[1], token_ids=(1, 2))) + spans[2:]
    _refusal("legal JSON representation", fork_prefix, bad_tokens, turn_index=1)
    bad_entry = (spans[0], _data(spans[1], token_ids=[1, "x"])) + spans[2:]
    _refusal("entry 1", fork_prefix, bad_entry, turn_index=1)


def test_writer_and_reader_refusals(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    _refusal("max_record_bytes", TraceWriter, path, 0)
    writer = TraceWriter(path)
    _refusal("iterables of Span", writer.write_episode, 1)
    _refusal("record 1", writer.write_episode, [_spans()[0], "bad"])

    _refusal("allow_partial_tail", read_trace, path, allow_partial_tail=1)
    path.write_bytes(b"\xff\n")
    _refusal("not valid UTF-8", read_trace, path)
