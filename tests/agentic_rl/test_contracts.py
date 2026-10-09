"""Tests for the agentic RL trajectory contracts (the torch-free contract plane).

WHAT IS CLAIMED: shaped tool calls, turns and trajectories construct and keep
their text and logprobs verbatim; every refusal path names the failing field
(and the trajectory where the construction knows it, plus both sides of any
count); the multi-turn positive control flattens with loss_mask 1 exactly on
sampled ASSISTANT/TOOL_CALL tokens and 0 on tool results; per-token columns
line up one entry per response token; a None reward survives flatten as None;
the contract plane and the package carry no engine import.

WHAT IS NOT CLAIMED: no harness, engine, tokenizer or renderer is exercised --
every token id here is synthetic and every trajectory is built by hand.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import foundationscale.agentic_rl as pkg
from foundationscale.agentic_rl import contracts, token_trace
from foundationscale.agentic_rl.contracts import (
    DECLARED_COLUMNS,
    SegmentKind,
    Termination,
    ToolCall,
    Trajectory,
    TrajectoryRefusal,
    Turn,
    flatten,
    group_by_uid,
)


def _tool_call(**overrides: Any) -> ToolCall:
    fields: dict[str, Any] = {
        "name": "search",
        "arguments": '{"q": "paris"}',
        "call_id": "call-1",
        "parse_error": None,
    }
    fields.update(overrides)
    return ToolCall(**fields)


def _turn(**overrides: Any) -> Turn:
    fields: dict[str, Any] = {
        "index": 2,
        "kind": SegmentKind.ASSISTANT,
        "token_ids": (6, 7),
        "generated": True,
        "logprobs": (-0.1, -0.2),
    }
    fields.update(overrides)
    return Turn(**fields)


def _prompt_turns() -> tuple[Turn, ...]:
    return (
        Turn(
            index=0,
            kind=SegmentKind.SYSTEM,
            token_ids=(1, 2, 3),
            generated=False,
            logprobs=None,
        ),
        Turn(
            index=1,
            kind=SegmentKind.USER,
            token_ids=(4, 5),
            generated=False,
            logprobs=None,
        ),
    )


def _response_turns() -> tuple[Turn, ...]:
    return (
        Turn(
            index=2,
            kind=SegmentKind.ASSISTANT,
            token_ids=(6, 7, 8, 9),
            generated=True,
            logprobs=(-0.1, -0.2, -0.3, -0.4),
            tool_calls=(_tool_call(),),
        ),
        Turn(
            index=3,
            kind=SegmentKind.TOOL_RESULT,
            token_ids=(10, 11),
            generated=False,
            logprobs=None,
        ),
        Turn(
            index=4,
            kind=SegmentKind.ASSISTANT,
            token_ids=(12, 13),
            generated=True,
            logprobs=None,
            repetition_hit=True,
        ),
    )


def _fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "uid": "group-1",
        "session_id": "attempt-1",
        "harness": "harness/one",
        "prompt_turns": _prompt_turns(),
        "turns": _response_turns(),
        "reward": 1.5,
        "abstention_reason": None,
        "termination": Termination.STOP,
        "policy_version_min": 0,
        "policy_version_max": 3,
    }
    fields.update(overrides)
    return fields


def _trajectory(**overrides: Any) -> Trajectory:
    return Trajectory(**_fields(**overrides))


# ---------------------------------------------------------------------------
# MUST-style positive controls
# ---------------------------------------------------------------------------


def test_must_multi_turn_trajectory_flattens_and_supervises_only_sampled_turns() -> None:
    trajectory = _trajectory(reward=None, abstention_reason="abstained: no answer found")

    batch = flatten((trajectory,))

    assert tuple(batch.columns) == DECLARED_COLUMNS
    assert batch.required == DECLARED_COLUMNS
    assert len(batch) == 1
    assert batch.column("prompt_token_ids")[0] == (1, 2, 3, 4, 5)
    assert batch.column("prompt_ids")[0] == "group-1"
    assert batch.column("response_ids")[0] == (6, 7, 8, 9, 10, 11, 12, 13)
    # 4 sampled assistant tokens, 2 tool-result tokens, 2 sampled assistant tokens.
    assert batch.column("loss_mask")[0] == (1, 1, 1, 1, 0, 0, 1, 1)
    assert batch.column("segment_kind")[0] == (
        SegmentKind.ASSISTANT,
        SegmentKind.ASSISTANT,
        SegmentKind.ASSISTANT,
        SegmentKind.ASSISTANT,
        SegmentKind.TOOL_RESULT,
        SegmentKind.TOOL_RESULT,
        SegmentKind.ASSISTANT,
        SegmentKind.ASSISTANT,
    )
    assert batch.column("turn_index")[0] == (2, 2, 2, 2, 3, 3, 4, 4)
    assert batch.column("repetition_mask")[0] == (0, 0, 0, 0, 0, 0, 1, 1)
    assert batch.column("tool_call_error_mask")[0] == (0, 0, 0, 0, 0, 0, 0, 0)
    assert batch.column("rollout_logprobs")[0] == (
        -0.1,
        -0.2,
        -0.3,
        -0.4,
        None,
        None,
        None,
        None,
    )
    assert trajectory.supervised_token_count == 6


def test_must_per_token_columns_line_up_one_entry_per_response_token() -> None:
    batch = flatten((_trajectory(),))

    response = batch.column("response_ids")[0]
    assert len(response) == 8
    for name in (
        "loss_mask",
        "rollout_logprobs",
        "turn_index",
        "segment_kind",
        "tool_call_error_mask",
        "repetition_mask",
    ):
        assert len(batch.column(name)[0]) == len(response), name


def test_must_reward_none_survives_flatten_as_none() -> None:
    abstained = _trajectory(reward=None, abstention_reason="abstained: uncertain")
    scored = _trajectory(session_id="attempt-2", reward=0.0, abstention_reason=None)

    batch = flatten((abstained, scored))

    assert batch.column("reward") == (None, 0.0)
    assert batch.column("abstention_reason") == ("abstained: uncertain", None)
    assert batch.column("reward")[0] is None


def test_bool_is_not_an_int_index_or_token_id() -> None:
    Turn(index=1, kind=SegmentKind.USER, token_ids=(1,), generated=False, logprobs=None)

    with pytest.raises(TrajectoryRefusal, match="field 'index'") as excinfo:
        _turn(index=True)
    assert "is not True" in str(excinfo.value)

    with pytest.raises(TrajectoryRefusal, match="field 'token_ids'") as excinfo:
        _turn(token_ids=(1, True))
    assert "entry 1" in str(excinfo.value)
    assert "is not True" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


def test_tool_call_constructs_and_keeps_raw_arguments_verbatim() -> None:
    raw = '{"b": 1, "a": 2}'

    call = _tool_call(arguments=raw, parse_error="invalid JSON in arguments")

    assert call.arguments == raw
    assert call.parse_error == "invalid JSON in arguments"
    assert _tool_call(call_id=None, parse_error=None).call_id is None


def test_tool_call_name_must_be_a_non_empty_str() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'name'"):
        _tool_call(name="")
    with pytest.raises(TrajectoryRefusal, match="field 'name'"):
        _tool_call(name=7)


def test_tool_call_name_may_be_none_only_as_a_recorded_parse_failure() -> None:
    unnamed = _tool_call(name=None, parse_error="call markup carries no function name")
    assert unnamed.name is None
    with pytest.raises(TrajectoryRefusal, match="field 'parse_error'"):
        _tool_call(name=None, parse_error=None)


def test_tool_call_arguments_must_be_the_raw_text() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'arguments'"):
        _tool_call(arguments={"q": 1})


def test_tool_call_call_id_is_none_or_a_non_empty_str() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'call_id'"):
        _tool_call(call_id="")
    with pytest.raises(TrajectoryRefusal, match="field 'call_id'"):
        _tool_call(call_id=5)


def test_tool_call_parse_error_is_none_or_a_non_empty_str() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'parse_error'"):
        _tool_call(parse_error="")
    with pytest.raises(TrajectoryRefusal, match="field 'parse_error'"):
        _tool_call(parse_error=5)


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------


def test_turn_freezes_its_sequences_and_keeps_logprobs_verbatim() -> None:
    turn = _turn(token_ids=[1, 2], logprobs=[-0.5, -0.25])

    assert turn.token_ids == (1, 2)
    assert turn.logprobs == (-0.5, -0.25)
    assert turn.tool_calls == ()
    assert turn.repetition_hit is False


def test_turn_index_must_be_a_real_non_negative_int() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'index'"):
        _turn(index="2")
    with pytest.raises(TrajectoryRefusal, match="field 'index'"):
        _turn(index=-1)


def test_turn_kind_must_be_a_segment_kind() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'kind'"):
        _turn(kind="assistant")


def test_turn_token_ids_must_be_real_non_negative_ints() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'token_ids'"):
        _turn(token_ids=(1, 1.5))
    with pytest.raises(TrajectoryRefusal, match="field 'token_ids'"):
        _turn(token_ids=(-3,))


def test_turn_token_ids_must_be_a_non_empty_sequence() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'token_ids'"):
        _turn(token_ids=())
    with pytest.raises(TrajectoryRefusal, match="field 'token_ids'"):
        _turn(token_ids="ab")


def test_turn_generated_must_be_a_bool() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'generated'"):
        _turn(generated=1)
    with pytest.raises(TrajectoryRefusal, match="field 'generated'"):
        _turn(generated="false")


def test_turn_logprobs_are_lawful_only_on_generated_turns() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'logprobs'") as excinfo:
        _turn(generated=False, logprobs=(-0.1, -0.2))
    assert "non-generated" in str(excinfo.value)


def test_turn_logprobs_must_match_the_token_count_exactly() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'logprobs'") as short:
        _turn(token_ids=(6, 7), logprobs=(-0.1,))
    assert "carries 1 entries but field 'token_ids' carries 2" in str(short.value)

    with pytest.raises(TrajectoryRefusal, match="field 'logprobs'") as long:
        _turn(token_ids=(6, 7), logprobs=(-0.1, -0.2, -0.3))
    assert "carries 3 entries but field 'token_ids' carries 2" in str(long.value)


def test_turn_logprobs_must_be_finite_reals() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'logprobs'"):
        _turn(token_ids=(6, 7), logprobs=(-0.1, "x"))
    with pytest.raises(TrajectoryRefusal, match="field 'logprobs'") as excinfo:
        _turn(token_ids=(6, 7), logprobs=(float("inf"), -0.2))
    assert "finite" in str(excinfo.value)


def test_turn_tool_calls_are_lawful_only_on_generated_assistant_or_tool_call_turns() -> None:
    _turn(
        kind=SegmentKind.TOOL_CALL,
        generated=True,
        tool_calls=(_tool_call(),),
    )
    with pytest.raises(TrajectoryRefusal, match="field 'tool_calls'"):
        _turn(
            kind=SegmentKind.USER,
            generated=False,
            logprobs=None,
            tool_calls=(_tool_call(),),
        )
    with pytest.raises(TrajectoryRefusal, match="field 'tool_calls'"):
        _turn(
            kind=SegmentKind.ASSISTANT,
            generated=False,
            logprobs=None,
            tool_calls=(_tool_call(),),
        )


def test_turn_tool_call_entries_must_be_tool_calls() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'tool_calls'"):
        _turn(tool_calls=("search",))


def test_turn_repetition_hit_must_be_a_bool() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'repetition_hit'"):
        _turn(repetition_hit="yes")


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


def test_trajectory_names_must_be_non_empty_strs() -> None:
    for name in ("uid", "session_id", "harness"):
        with pytest.raises(TrajectoryRefusal, match=name) as excinfo:
            _trajectory(**{name: ""})
        assert name in str(excinfo.value)
    with pytest.raises(TrajectoryRefusal, match="uid"):
        _trajectory(uid=3)


def test_trajectory_prompt_turns_must_exist_and_never_be_generated() -> None:
    with pytest.raises(TrajectoryRefusal, match="prompt_turns"):
        _trajectory(prompt_turns=())
    with pytest.raises(TrajectoryRefusal, match="prompt_turns"):
        _trajectory(
            prompt_turns=(
                Turn(
                    index=0,
                    kind=SegmentKind.USER,
                    token_ids=(1, 2),
                    generated=True,
                    logprobs=None,
                ),
            )
        )


def test_trajectory_turn_entries_must_be_turns() -> None:
    with pytest.raises(TrajectoryRefusal, match="prompt_turns"):
        _trajectory(prompt_turns=("user",))
    with pytest.raises(TrajectoryRefusal, match="turns"):
        _trajectory(turns=("assistant",))


def test_trajectory_turns_must_exist() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'turns'"):
        _trajectory(turns=())


def test_zero_turn_trajectory_is_lawful_only_under_infra() -> None:
    # A run that failed before the harness ever rendered a prompt (e.g. the
    # environment never started) has nothing real on either side; both
    # prompt_turns and turns may be () exactly under Termination.INFRA.
    trajectory = _trajectory(
        prompt_turns=(),
        turns=(),
        reward=None,
        abstention_reason="infra:EnvInfraError",
        termination=Termination.INFRA,
    )
    assert trajectory.is_infra is True
    assert trajectory.prompt_turns == ()
    assert trajectory.turns == ()
    assert trajectory.response_token_ids == ()
    assert trajectory.loss_mask() == ()
    assert trajectory.supervised_token_count == 0

    # Outside INFRA, both sequences are still required non-empty, unchanged --
    # the allowance is INFRA-only and is not extended to ABORTED, which has its
    # own (separate) exemption from the generated-response requirement.
    with pytest.raises(TrajectoryRefusal, match="field 'prompt_turns'"):
        _trajectory(prompt_turns=(), turns=(), termination=Termination.STOP)
    with pytest.raises(TrajectoryRefusal, match="field 'turns'"):
        _trajectory(
            turns=(),
            termination=Termination.ABORTED,
            reward=None,
            abstention_reason="aborted: cancelled",
        )


def test_zero_turn_trajectory_flattens_with_empty_per_token_columns() -> None:
    trajectory = _trajectory(
        prompt_turns=(),
        turns=(),
        reward=None,
        abstention_reason="infra:EnvInfraError",
        termination=Termination.INFRA,
    )
    batch = flatten([trajectory])
    assert len(batch) == 1
    assert batch.columns["response_ids"][0] == ()
    assert batch.columns["loss_mask"][0] == ()
    assert batch.columns["prompt_token_ids"][0] == ()
    assert batch.columns["reward"][0] is None


def test_trajectory_turn_indices_are_strictly_increasing() -> None:
    with pytest.raises(TrajectoryRefusal, match="index order") as duplicated:
        _trajectory(
            turns=(
                _turn(index=4, token_ids=(6, 7)),
                _turn(index=4, token_ids=(8, 9)),
            )
        )
    assert "index 4 follows index 4" in str(duplicated.value)
    assert "position 3" in str(duplicated.value)

    with pytest.raises(TrajectoryRefusal, match="index order"):
        _trajectory(turns=(_turn(index=0, token_ids=(6, 7)),))


def test_trajectory_reward_and_abstention_form_a_biconditional() -> None:
    _trajectory(reward=None, abstention_reason="abstained: not sure")
    _trajectory(reward=0.0, abstention_reason=None)

    with pytest.raises(TrajectoryRefusal) as both_none:
        _trajectory(reward=None, abstention_reason=None)
    message = str(both_none.value)
    assert "field 'reward'" in message
    assert "field 'abstention_reason'" in message

    with pytest.raises(TrajectoryRefusal) as both_set:
        _trajectory(reward=1.0, abstention_reason="abstained: not sure")
    message = str(both_set.value)
    assert "field 'reward'" in message
    assert "field 'abstention_reason'" in message


def test_trajectory_abstention_reason_must_be_a_non_empty_str() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'abstention_reason'"):
        _trajectory(reward=None, abstention_reason="")
    with pytest.raises(TrajectoryRefusal, match="field 'abstention_reason'"):
        _trajectory(reward=None, abstention_reason=5)


def test_trajectory_reward_must_be_a_finite_real() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'reward'"):
        _trajectory(reward="1.0")
    with pytest.raises(TrajectoryRefusal, match="field 'reward'") as excinfo:
        _trajectory(reward=float("nan"))
    assert "finite" in str(excinfo.value)


def test_trajectory_termination_must_be_a_termination() -> None:
    with pytest.raises(TrajectoryRefusal, match="field 'termination'"):
        _trajectory(termination="stop")


def test_infra_trajectory_carries_the_reason_and_never_a_reward() -> None:
    trajectory = _trajectory(
        termination=Termination.INFRA,
        reward=None,
        abstention_reason="infra:pod_lost",
    )

    assert trajectory.is_infra is True
    assert trajectory.abstention_reason == "infra:pod_lost"

    with pytest.raises(TrajectoryRefusal) as scored:
        _trajectory(
            termination=Termination.INFRA,
            reward=0.0,
            abstention_reason=None,
        )
    assert "field 'reward'" in str(scored.value)
    assert "INFRA" in str(scored.value)

    with pytest.raises(TrajectoryRefusal) as unmarked:
        _trajectory(
            termination=Termination.INFRA,
            reward=None,
            abstention_reason="pod lost",
        )
    assert "field 'abstention_reason'" in str(unmarked.value)
    assert "infra:" in str(unmarked.value)


def test_trajectory_policy_versions_are_declared_together() -> None:
    with pytest.raises(TrajectoryRefusal) as half:
        _trajectory(policy_version_min=2, policy_version_max=None)
    message = str(half.value)
    assert "field 'policy_version_min'" in message
    assert "field 'policy_version_max'" in message

    assert _trajectory(policy_version_min=None, policy_version_max=None) is not None


def test_trajectory_policy_versions_must_be_real_non_negative_ints_in_order() -> None:
    with pytest.raises(TrajectoryRefusal) as bad_type:
        _trajectory(policy_version_min=True, policy_version_max=3)
    assert "field 'policy_version_min'" in str(bad_type.value)
    assert "is not True" in str(bad_type.value)

    with pytest.raises(TrajectoryRefusal, match="field 'policy_version_min'"):
        _trajectory(policy_version_min=-1, policy_version_max=3)

    with pytest.raises(TrajectoryRefusal) as backwards:
        _trajectory(policy_version_min=4, policy_version_max=3)
    message = str(backwards.value)
    assert "field 'policy_version_min'" in message
    assert "field 'policy_version_max'" in message
    assert "4" in message and "3" in message


def test_trajectory_requires_a_generated_response_unless_fault_or_abort() -> None:
    idle = (
        Turn(index=2, kind=SegmentKind.ASSISTANT, token_ids=(6, 7), generated=False, logprobs=None),
    )
    with pytest.raises(TrajectoryRefusal, match="field 'turns'") as excinfo:
        _trajectory(turns=idle)
    assert "0 generated turns of 1" in str(excinfo.value)

    _trajectory(
        turns=idle,
        termination=Termination.ABORTED,
        reward=None,
        abstention_reason="abstained: cancelled",
    )
    _trajectory(
        turns=idle,
        termination=Termination.INFRA,
        reward=None,
        abstention_reason="infra:pod_lost",
    )


def test_trajectory_metadata_is_a_read_only_str_mapping() -> None:
    trajectory = _trajectory(metadata={"seed": "7"})

    assert trajectory.metadata == {"seed": "7"}
    with pytest.raises(TypeError):
        trajectory.metadata["seed"] = "8"


def test_trajectory_metadata_entries_must_be_named_strs() -> None:
    with pytest.raises(TrajectoryRefusal, match="metadata"):
        _trajectory(metadata=(('"k"', "v"),))
    with pytest.raises(TrajectoryRefusal, match="metadata"):
        _trajectory(metadata={"": "v"})
    with pytest.raises(TrajectoryRefusal, match="metadata"):
        _trajectory(metadata={3: "v"})
    with pytest.raises(TrajectoryRefusal, match="metadata"):
        _trajectory(metadata={"seed": 7})


def test_trajectory_hash_stands_on_uid_and_session_id() -> None:
    first = _trajectory()
    second = _trajectory()

    assert hash(first) == hash(second)
    assert hash(first) == hash((first.uid, first.session_id))


def test_trajectory_masks_stay_per_turn_and_align_with_the_response() -> None:
    first = Turn(
        index=2,
        kind=SegmentKind.ASSISTANT,
        token_ids=(6, 7),
        generated=True,
        logprobs=None,
        tool_calls=(_tool_call(parse_error="invalid JSON in arguments"),),
    )
    second = Turn(
        index=3,
        kind=SegmentKind.TOOL_CALL,
        token_ids=(8,),
        generated=True,
        logprobs=(-0.25,),
        tool_calls=(
            _tool_call(
                name="lookup",
                arguments='{"id": 1}',
                call_id=None,
                parse_error=None,
            ),
        ),
    )
    trajectory = _trajectory(
        turns=(first, second),
        termination=Termination.LENGTH,
    )

    assert trajectory.response_token_ids == (6, 7, 8)
    assert trajectory.loss_mask() == (1, 1, 1)
    assert trajectory.tool_call_error_mask() == (1, 1, 0)
    assert trajectory.repetition_mask() == (0, 0, 0)
    assert trajectory.rollout_logprobs() == (None, None, -0.25)
    assert trajectory.turn_index_per_token() == (2, 2, 3)
    assert trajectory.segment_kind_per_token() == (
        SegmentKind.ASSISTANT,
        SegmentKind.ASSISTANT,
        SegmentKind.TOOL_CALL,
    )
    assert trajectory.supervised_token_count == 3
    assert trajectory.is_infra is False


# ---------------------------------------------------------------------------
# flatten / group_by_uid
# ---------------------------------------------------------------------------


def test_flatten_refuses_empty_input_naming_the_all_empty_shape() -> None:
    with pytest.raises(TrajectoryRefusal) as excinfo:
        flatten(())
    assert "all([])" in str(excinfo.value)


def test_flatten_refuses_non_trajectory_rows() -> None:
    with pytest.raises(TrajectoryRefusal, match="element 1"):
        flatten((_trajectory(), "attempt"))


def test_flatten_refuses_duplicate_attempts_naming_both_rows() -> None:
    trajectories = (_trajectory(), _trajectory())

    with pytest.raises(TrajectoryRefusal) as excinfo:
        flatten(trajectories)

    message = str(excinfo.value)
    assert "rows 0 and 1" in message
    assert "'group-1'" in message
    assert "'attempt-1'" in message


def test_flatten_refuses_per_token_columns_of_different_lengths() -> None:
    class _OverlongLossMask(Trajectory):
        """Stand-in for adapter code that returns a mask measured to its own length."""

        def loss_mask(self) -> tuple[int, ...]:
            return (1, 0, 1, 1)

    trajectory = _OverlongLossMask(**_fields())

    with pytest.raises(TrajectoryRefusal) as excinfo:
        flatten((trajectory,))

    message = str(excinfo.value)
    assert "'loss_mask'" in message
    assert "4 entries" in message
    assert "8 token(s)" in message


def test_group_by_uid_preserves_first_seen_order_and_stays_read_only() -> None:
    first = _trajectory(uid="group-a")
    second = _trajectory(uid="group-b")
    third = _trajectory(uid="group-a", session_id="attempt-2")

    grouped = group_by_uid((first, second, third))

    assert list(grouped) == ["group-a", "group-b"]
    assert grouped["group-a"] == (first, third)
    assert grouped["group-b"] == (second,)
    with pytest.raises(TypeError):
        grouped["group-c"] = ()


def test_group_by_uid_accepts_an_empty_sequence() -> None:
    assert dict(group_by_uid(())) == {}


def test_group_by_uid_refuses_non_trajectory_entries() -> None:
    with pytest.raises(TrajectoryRefusal, match="element 0"):
        group_by_uid(("attempt",))


# ---------------------------------------------------------------------------
# plane hygiene
# ---------------------------------------------------------------------------


def _assert_torch_free(module: ModuleType) -> None:
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "torch", alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "torch", node.module


def test_the_contract_plane_stays_torch_free() -> None:
    for module in (pkg, contracts, token_trace):
        _assert_torch_free(module)


def test_the_package_exports_the_contract_plane_names_sorted() -> None:
    assert pkg.__all__ == tuple(sorted(pkg.__all__))
    for name in pkg.__all__:
        assert getattr(pkg, name) is not None, name
    assert "ExperienceBatch" not in pkg.__all__


def test_flatten_group_key_can_split_baselines_per_harness() -> None:
    first = _trajectory()
    second = dataclasses.replace(first, session_id="attempt-2", harness="other")
    plain = flatten((first, second))
    split = flatten((first, second), group_by_harness=True)
    assert plain.column("prompt_ids") == ("group-1", "group-1")
    assert split.column("prompt_ids") == (
        f"group-1::{first.harness}",
        "group-1::other",
    )
    with pytest.raises(TrajectoryRefusal, match="group_by_harness"):
        flatten((first,), group_by_harness=1)  # type: ignore[arg-type]
