"""Tests for the v2 agentic RL trajectory plane (binding API contract only).

WHAT IS CLAIMED: every refusal the v2 API declares fires as TrajectoryV2Refusal
and names both counts where a count is compared -- bool-as-int, the per-mask
logprob and policy-version rules, span coverage, routed-experts length (R5),
reward value and TOKEN_SPAN span, and the schema version; build_segments merges
exact prefixes and EOT splices, fragments otherwise, never fabricates routed
experts and never rewrites a token; the status table, the reward reducers, the
JSON round trip, the v1 bridges and flatten_v2 behave exactly as declared.

WHAT IS NOT CLAIMED: no implementation detail beyond the binding API -- every
token id here is synthetic and every object is built by hand.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import pytest

from foundationscale.agentic_rl import contracts
from foundationscale.agentic_rl.trajectory_v2 import (
    SCHEMA_VERSION,
    ApiFormat,
    DiscardedGeneration,
    EnvKind,
    EnvMeta,
    Episode,
    EpisodeStatus,
    Fidelity,
    FidelityStatus,
    GatewayMeta,
    Generation,
    HarnessMeta,
    ModelMeta,
    ParseVerdict,
    RewardEvent,
    RewardScope,
    SandboxMeta,
    Segment,
    Termination,
    Timings,
    ToolCallV2,
    TrajectoryV2Refusal,
    build_segments,
    episode_reward,
    flatten_v2,
    from_json,
    from_v1,
    is_trainable,
    to_json,
    to_v1,
    validate_episode,
)

# ---------------------------------------------------------------------------
# helper builders
# ---------------------------------------------------------------------------


def _experts(count: int, *, start: int = 0) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Build `count` per-token routed-expert rows (two MoE layers, top-k ids)."""
    return tuple(((start + i, start + i + 1), (start + i + 2,)) for i in range(count))


def _tool_call_v2(**overrides: Any) -> ToolCallV2:
    """Build a parsed-fine tool call, overriding any field wholesale."""
    fields: dict[str, Any] = {
        "call_id": "call-1",
        "name": "search",
        "arguments_raw": '{"q": 1}',
        "parse_verdict": ParseVerdict.OBSERVED_ACCEPTED,
        "parse_error": None,
    }
    fields.update(overrides)
    return ToolCallV2(**fields)


def _generation(**overrides: Any) -> Generation:
    """Build a two-output-token generation over a three-token prompt."""
    fields: dict[str, Any] = {
        "generation_id": "gen-0",
        "prompt_ids": (1, 2, 3),
        "output_ids": (4, 5),
        "logprobs": (-0.1, -0.2),
        "policy_version": 3,
    }
    fields.update(overrides)
    return Generation(**fields)


def _spans_for(mask: Sequence[int]) -> tuple[tuple[int, int, str], ...]:
    """Build one generation span per contiguous run of mask-1 tokens."""
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    for index, bit in enumerate(mask):
        if bit == 1 and start is None:
            start = index
        elif bit == 0 and start is not None:
            spans.append((start, index, f"gen-{len(spans)}"))
            start = None
    if start is not None:
        spans.append((start, len(mask), f"gen-{len(spans)}"))
    return tuple(spans)


def _segment(**overrides: Any) -> Segment:
    """Build a valid fully-trainable two-token segment, overriding any field wholesale."""
    fields: dict[str, Any] = {
        "segment_index": 0,
        "prompt_ids": (1, 2, 3),
        "response_ids": (4, 5),
        "loss_mask": (1, 1),
        "logprobs": (-0.1, -0.2),
        "policy_version": (3, 3),
        "generation_spans": ((0, 2, "gen-0"),),
    }
    fields.update(overrides)
    return Segment(**fields)


def _segment_from_mask(
    mask: Sequence[int],
    logprobs: Sequence[float],
    versions: Sequence[int],
    **overrides: Any,
) -> Segment:
    """Build a segment over len(mask) response tokens with the given per-token rules."""
    fields: dict[str, Any] = {
        "segment_index": 0,
        "prompt_ids": (1, 2),
        "response_ids": tuple(range(10, 10 + len(mask))),
        "loss_mask": tuple(mask),
        "logprobs": tuple(logprobs),
        "policy_version": tuple(versions),
        "generation_spans": _spans_for(mask),
    }
    fields.update(overrides)
    return Segment(**fields)


def _two_segments() -> tuple[Segment, ...]:
    """Build a two-segment stream whose second segment mixes masked and trainable tokens."""
    return (
        _segment(),
        _segment(
            segment_index=1,
            prompt_ids=(9, 9),
            response_ids=(8, 8, 7),
            loss_mask=(1, 0, 1),
            logprobs=(-0.3, math.nan, -0.4),
            policy_version=(4, -1, 4),
            generation_spans=((0, 1, "gen-1"), (2, 3, "gen-2")),
        ),
    )


def _reward_event(**overrides: Any) -> RewardEvent:
    """Build one measured EPISODE-scope reward event."""
    fields: dict[str, Any] = {
        "source": "grader",
        "verifier_id": "verifier/one",
        "verifier_version": "1",
        "value": 1.5,
        "scope": RewardScope.EPISODE,
    }
    fields.update(overrides)
    return RewardEvent(**fields)


def _model() -> ModelMeta:
    return ModelMeta(name="model/one", version=0, checkpoint_sha="sha-1")


def _harness() -> HarnessMeta:
    return HarnessMeta(name="harness/one", version="1")


def _env() -> EnvMeta:
    return EnvMeta(kind=EnvKind.FS_LOCAL, task_id="task-1")


def _gateway() -> GatewayMeta:
    return GatewayMeta(api_format=ApiFormat.CHAT_COMPLETIONS, session_id="session-1")


def _episode(**overrides: Any) -> Episode:
    """Build a valid one-segment OK episode, overriding any field wholesale."""
    fields: dict[str, Any] = {
        "episode_id": "ep-1",
        "group_id": "group-1",
        "task_prompt_ids": (1, 2, 3),
        "status": EpisodeStatus.OK,
        "termination": Termination.STOP,
        "segments": (_segment(),),
        "reward_events": (),
        "model": _model(),
        "harness": _harness(),
        "env": _env(),
        "gateway": _gateway(),
    }
    fields.update(overrides)
    return Episode(**fields)


def _status_episode(status: EpisodeStatus, **overrides: Any) -> Episode:
    """Build an episode whose status, termination and fidelity agree with each other."""
    fields: dict[str, Any] = {
        "status": status,
        "termination": {
            EpisodeStatus.OK: Termination.STOP,
            EpisodeStatus.TIMEOUT: Termination.TURN_LIMIT,
            EpisodeStatus.CONTEXT_OVERFLOW: Termination.LENGTH,
        }.get(status, Termination.ERROR),
    }
    if status is EpisodeStatus.FIDELITY_MISMATCH:
        fields["fidelity"] = Fidelity(status=FidelityStatus.MISMATCH, detail="replay drift")
    fields.update(overrides)
    return _episode(**fields)


def _v1_trajectory(**overrides: Any) -> contracts.Trajectory:
    """Build a minimal valid v1 Trajectory through the provided contracts plane."""
    fields: dict[str, Any] = {
        "uid": "group-1",
        "session_id": "attempt-1",
        "harness": "harness/one",
        "prompt_turns": (
            contracts.Turn(
                index=0,
                kind=contracts.SegmentKind.USER,
                token_ids=(1, 2, 3),
                generated=False,
                logprobs=None,
            ),
        ),
        "turns": (
            contracts.Turn(
                index=1,
                kind=contracts.SegmentKind.ASSISTANT,
                token_ids=(4, 5),
                generated=True,
                logprobs=(-0.1, -0.2),
            ),
        ),
        "reward": 1.5,
        "abstention_reason": None,
        "termination": contracts.Termination.STOP,
        "policy_version_min": 0,
        "policy_version_max": 0,
    }
    fields.update(overrides)
    return contracts.Trajectory(**fields)


def _same(left: Any, right: Any) -> bool:
    """Compare two structures field by field, treating NaN as equal to NaN."""
    if (
        isinstance(left, float)
        and isinstance(right, float)
        and (math.isnan(left) or math.isnan(right))
    ):
        return math.isnan(left) and math.isnan(right)
    if dataclasses.is_dataclass(left) and dataclasses.is_dataclass(right):
        return type(left) is type(right) and all(
            _same(getattr(left, f.name), getattr(right, f.name)) for f in dataclasses.fields(left)
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    return bool(left == right)


def _leaves(value: Any) -> list[Any]:
    """Flatten a JSON-shaped payload into its keys and scalars."""
    if isinstance(value, Mapping):
        out: list[Any] = []
        for key, item in value.items():
            out.append(key)
            out.extend(_leaves(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_leaves(item))
        return out
    return [value]


# ---------------------------------------------------------------------------
# refusals: counts, bools, and the shape rules
# ---------------------------------------------------------------------------


def test_refusal_is_a_value_error() -> None:
    """TrajectoryV2Refusal derives from ValueError, so a caller may catch either."""
    assert issubclass(TrajectoryV2Refusal, ValueError)


def test_segment_count_mismatches_refuse_naming_both_counts() -> None:
    """A count comparison names both sides -- k of n -- and never pads or truncates."""
    for field_name, value in (
        ("loss_mask", (1, 1, 1)),
        ("logprobs", (-0.1, -0.2, -0.3)),
        ("policy_version", (3, 3, 3)),
    ):
        with pytest.raises(TrajectoryV2Refusal) as excinfo:
            _segment(**{field_name: value})
        message = str(excinfo.value)
        assert "3" in message and "2" in message, field_name


def test_a_bool_is_never_accepted_where_an_int_is_required() -> None:
    """FS doctrine "1 is not True": a bool is refused in every int-bearing field."""
    with pytest.raises(TrajectoryV2Refusal):
        _generation(prompt_ids=(True, 2))
    with pytest.raises(TrajectoryV2Refusal):
        _generation(policy_version=True)
    with pytest.raises(TrajectoryV2Refusal):
        _segment(segment_index=True)
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, True), (-0.1, -0.2), (3, 3))
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, 1), (-0.1, -0.2), (True, 3))
    with pytest.raises(TrajectoryV2Refusal):
        _reward_event(value=True)


def test_generation_identity_and_tokens_must_be_non_empty_non_negative_ints() -> None:
    """A generation names itself and carries real token ids, never text or a float."""
    with pytest.raises(TrajectoryV2Refusal):
        _generation(generation_id="")
    with pytest.raises(TrajectoryV2Refusal):
        _generation(prompt_ids=())
    with pytest.raises(TrajectoryV2Refusal):
        _generation(output_ids=())
    with pytest.raises(TrajectoryV2Refusal):
        _generation(prompt_ids=(1, -2))
    with pytest.raises(TrajectoryV2Refusal):
        _generation(output_ids=(1, 2.5))


def test_generation_logprobs_are_exactly_one_per_output_token() -> None:
    """One engine-reported logprob per output token, and the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal) as excinfo:
        _generation(logprobs=(-0.1,))
    message = str(excinfo.value)
    assert "1" in message and "2" in message


def test_generation_logprobs_are_all_finite() -> None:
    """Generation logprobs are engine-reported and finite -- nan and inf are refused."""
    with pytest.raises(TrajectoryV2Refusal):
        _generation(logprobs=(math.nan, -0.2))
    with pytest.raises(TrajectoryV2Refusal):
        _generation(logprobs=(-0.1, float("inf")))


def test_generation_policy_version_is_a_non_negative_int() -> None:
    """A generation names the policy that sampled it, starting at version 0."""
    with pytest.raises(TrajectoryV2Refusal):
        _generation(policy_version=-1)


def test_generation_routed_experts_length_is_refused_naming_both_counts() -> None:
    """R5: experts carry one row per prompt+output token or are refused with both counts."""
    with pytest.raises(TrajectoryV2Refusal) as excinfo:
        _generation(routed_experts=_experts(4))
    message = str(excinfo.value)
    assert "4" in message and "5" in message
    assert _generation(routed_experts=_experts(5)).routed_experts is not None


def test_segment_routed_experts_length_is_refused_naming_both_counts() -> None:
    """R5: a segment's experts cover prompt+response exactly, or name both counts."""
    with pytest.raises(TrajectoryV2Refusal) as excinfo:
        _segment(routed_experts=_experts(4))
    message = str(excinfo.value)
    assert "4" in message and "5" in message
    assert _segment(routed_experts=_experts(5)).routed_experts is not None


def test_trainable_positions_carry_finite_logprobs() -> None:
    """R4: a mask-1 position states a finite logprob -- nan and inf are refused there."""
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, 1), (math.nan, -0.2), (3, 3))
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, 1), (-0.1, float("inf")), (3, 3))
    assert _segment_from_mask((1, 1), (-0.1, -0.2), (3, 3)).logprobs == (-0.1, -0.2)


def test_masked_positions_carry_exactly_nan_logprobs() -> None:
    """R4: a mask-0 position is unmeasured -- exactly math.nan, never a finite filler."""
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, 0), (-0.1, -0.5), (3, -1))
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, 0), (-0.1, float("inf")), (3, -1))
    segment = _segment_from_mask((1, 0), (-0.1, math.nan), (3, -1))
    assert math.isnan(segment.logprobs[1])


def test_policy_version_is_minus_one_exactly_on_masked_tokens() -> None:
    """R4: masked tokens name no policy -- exactly -1 at mask 0 and >= 0 at mask 1."""
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, 0), (-0.1, math.nan), (3, 0))
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, 0), (-0.1, math.nan), (3, -2))
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, 1), (-0.1, -0.2), (3, -1))
    with pytest.raises(TrajectoryV2Refusal):
        _segment_from_mask((1, 1), (-0.1, -0.2), (3, -3))
    assert _segment_from_mask((1, 0), (-0.1, math.nan), (3, -1)).policy_version == (3, -1)


def test_generation_spans_cover_exactly_the_trainable_tokens() -> None:
    """Spans are end-exclusive, ascending, non-overlapping and cover exactly the mask-1 tokens."""
    good = _segment_from_mask((1, 0, 1), (-0.1, math.nan, -0.2), (3, -1, 3))
    assert good.generation_spans == ((0, 1, "gen-0"), (2, 3, "gen-1"))
    for spans in (
        (),  # trainable tokens left uncovered
        ((0, 3, "gen-0"),),  # covers a masked token
        ((0, 2, "gen-0"), (2, 3, "gen-1")),  # covers a masked token
        ((0, 2, "gen-0"), (1, 3, "gen-1")),  # overlaps
        ((2, 3, "gen-1"), (0, 1, "gen-0")),  # descending
        ((0, 1, "gen-0"), (2, 4, "gen-1")),  # end runs past the response
        ((0, 0, "gen-0"), (2, 3, "gen-1")),  # zero width leaves token 0 uncovered
    ):
        with pytest.raises(TrajectoryV2Refusal):
            _segment_from_mask(
                (1, 0, 1), (-0.1, math.nan, -0.2), (3, -1, 3), generation_spans=spans
            )


def test_num_trainable_tokens_counts_the_mask_ones() -> None:
    """The trainable count is the mask sum, and 0 is a real count of nothing to train on."""
    assert _segment().num_trainable_tokens == 2
    assert (
        _segment_from_mask((1, 0, 0), (-0.1, math.nan, math.nan), (3, -1, -1)).num_trainable_tokens
        == 1
    )


def test_tool_call_v2_parse_error_is_none_or_a_non_empty_str() -> None:
    """None means parsed fine and a non-empty string names the model-attributable error."""
    with pytest.raises(TrajectoryV2Refusal):
        _tool_call_v2(parse_error="")
    assert _tool_call_v2(parse_error="invalid JSON").parse_error == "invalid JSON"
    assert _tool_call_v2().parse_error is None


def test_tool_call_v2_synthesized_ids_require_a_call_id() -> None:
    """A synthesized call id must name the invocation it invented, and the origin is one of two."""
    with pytest.raises(TrajectoryV2Refusal):
        _tool_call_v2(call_id_origin="traced")
    with pytest.raises(TrajectoryV2Refusal):
        _tool_call_v2(call_id=None, call_id_origin="synthesized")
    with pytest.raises(TrajectoryV2Refusal):
        _tool_call_v2(call_id="", call_id_origin="synthesized")
    assert _tool_call_v2(call_id=None).call_id is None


def test_reward_event_value_is_finite_or_none_and_never_a_bool() -> None:
    """A reward is a finite scalar or None -- the abstention -- and a bool is refused."""
    for value in (True, False, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(TrajectoryV2Refusal):
            _reward_event(value=value)
    assert _reward_event(value=None).value is None
    assert _reward_event(value=0.0).value == 0.0


def test_reward_event_token_span_scope_requires_exactly_one_span() -> None:
    """A span is required iff the scope is TOKEN_SPAN, and then 0 <= start < end."""
    with pytest.raises(TrajectoryV2Refusal):
        _reward_event(scope=RewardScope.TOKEN_SPAN, span=None)
    with pytest.raises(TrajectoryV2Refusal):
        _reward_event(scope=RewardScope.EPISODE, span=(0, 1))
    with pytest.raises(TrajectoryV2Refusal):
        _reward_event(scope=RewardScope.TOKEN_SPAN, span=(1, 1))
    with pytest.raises(TrajectoryV2Refusal):
        _reward_event(scope=RewardScope.TOKEN_SPAN, span=(-1, 2))
    assert _reward_event(scope=RewardScope.TOKEN_SPAN, span=(0, 2)).span == (0, 2)


def test_reward_event_names_and_timestamps_are_stated() -> None:
    """Every reward event names its source and verifier, and its timestamp is non-negative."""
    for name in ("source", "verifier_id", "verifier_version"):
        with pytest.raises(TrajectoryV2Refusal):
            _reward_event(**{name: ""})
    with pytest.raises(TrajectoryV2Refusal):
        _reward_event(at_ts=-0.5)


def test_episode_schema_version_must_match_the_declared_version() -> None:
    """An episode declares exactly SCHEMA_VERSION, so no reader guesses the layout."""
    assert _episode().schema_version == SCHEMA_VERSION
    with pytest.raises(TrajectoryV2Refusal):
        _episode(schema_version="fs.trajectory/v1")


def test_episode_identity_fields_must_be_non_empty() -> None:
    """Episode, group and task prompt all name something -- absence of a name is not a name."""
    for name in ("episode_id", "group_id"):
        with pytest.raises(TrajectoryV2Refusal):
            _episode(**{name: ""})
    with pytest.raises(TrajectoryV2Refusal):
        _episode(task_prompt_ids=())


def test_episode_segment_indices_must_match_their_positions() -> None:
    """segment_index is the segment's position, or no per-token column says where it sits."""
    with pytest.raises(TrajectoryV2Refusal):
        _episode(
            segments=(
                _segment(),
                _segment(
                    segment_index=2,
                    prompt_ids=(9,),
                    response_ids=(8,),
                    loss_mask=(1,),
                    logprobs=(-0.5,),
                    policy_version=(4,),
                    generation_spans=((0, 1, "gen-1"),),
                ),
            )
        )


def test_validate_episode_requires_a_trainable_token_where_the_status_promises_one() -> None:
    """OK and trainable TIMEOUT stand on >= 1 trainable token; other statuses require none."""
    untrainable = dict(
        loss_mask=(0, 0),
        logprobs=(math.nan, math.nan),
        policy_version=(-1, -1),
        generation_spans=(),
    )
    with pytest.raises(TrajectoryV2Refusal):
        validate_episode(_episode(segments=(_segment(**untrainable),)))
    with pytest.raises(TrajectoryV2Refusal):
        validate_episode(
            _status_episode(
                EpisodeStatus.TIMEOUT,
                trainable_on_timeout=True,
                segments=(_segment(**untrainable),),
            )
        )
    validate_episode(
        _status_episode(
            EpisodeStatus.TIMEOUT, trainable_on_timeout=False, segments=(_segment(**untrainable),)
        )
    )
    validate_episode(
        _status_episode(EpisodeStatus.INFRA_ERROR, segments=(_segment(**untrainable),))
    )


def test_validate_episode_ties_mismatch_fidelity_to_the_mismatch_status() -> None:
    """A MISMATCH fidelity claim is only lawful on a FIDELITY_MISMATCH episode."""
    with pytest.raises(TrajectoryV2Refusal):
        validate_episode(
            _episode(fidelity=Fidelity(status=FidelityStatus.MISMATCH, detail="drift"))
        )
    validate_episode(_status_episode(EpisodeStatus.FIDELITY_MISMATCH))


# ---------------------------------------------------------------------------
# build_segments
# ---------------------------------------------------------------------------


def test_build_segments_opens_segment_zero_on_the_first_generation() -> None:
    """Gen 0 opens segment 0: its prompt, its output, all of it trainable."""
    generation = _generation(
        prompt_ids=(1, 2, 3), output_ids=(4, 5), logprobs=(-0.1, -0.2), policy_version=3
    )

    segments, fragmentation_count = build_segments((generation,))

    assert fragmentation_count == 0
    assert len(segments) == 1
    assert segments[0].prompt_ids == (1, 2, 3)
    assert segments[0].response_ids == (4, 5)
    assert segments[0].loss_mask == (1, 1)
    assert segments[0].generation_spans == ((0, 2, "gen-0"),)


def test_build_segments_merges_an_exact_prefix_continuation() -> None:
    """An exact token-prefix continuation merges, masking the context delta as untrainable."""
    first = _generation(
        prompt_ids=(1, 2, 3),
        output_ids=(4, 5),
        logprobs=(-0.1, -0.2),
        policy_version=3,
        generation_id="gen-0",
    )
    second = _generation(
        prompt_ids=(1, 2, 3, 4, 5, 6, 7),
        output_ids=(8, 9),
        logprobs=(-0.3, -0.4),
        policy_version=4,
        generation_id="gen-1",
    )

    segments, fragmentation_count = build_segments((first, second))

    assert fragmentation_count == 0
    assert len(segments) == 1
    segment = segments[0]
    assert segment.prompt_ids == (1, 2, 3)
    assert segment.response_ids == (4, 5, 6, 7, 8, 9)
    assert segment.loss_mask == (1, 1, 0, 0, 1, 1)
    assert segment.policy_version == (3, 3, -1, -1, 4, 4)
    assert segment.logprobs[:2] == (-0.1, -0.2)
    assert math.isnan(segment.logprobs[2]) and math.isnan(segment.logprobs[3])
    assert segment.logprobs[4:] == (-0.3, -0.4)
    assert segment.generation_spans == ((0, 2, "gen-0"), (4, 6, "gen-1"))
    assert segment.num_trainable_tokens == 4


def test_build_segments_merges_an_eot_splice_continuation() -> None:
    """An EOT splice merges across differing end-of-turn tokens and masks only the delta."""
    eot = frozenset({77, 99})
    first = _generation(
        prompt_ids=(1, 2, 3),
        output_ids=(4, 99),
        logprobs=(-0.1, -0.2),
        policy_version=3,
        generation_id="gen-0",
    )
    second = _generation(
        prompt_ids=(1, 2, 3, 4, 77, 88),
        output_ids=(7, 8),
        logprobs=(-0.3, -0.4),
        policy_version=4,
        generation_id="gen-1",
    )

    segments, fragmentation_count = build_segments((first, second), eot_token_ids=eot)

    assert fragmentation_count == 0
    assert len(segments) == 1
    segment = segments[0]
    assert segment.response_ids == (4, 99, 88, 7, 8)
    assert segment.loss_mask == (1, 1, 0, 1, 1)
    assert segment.policy_version == (3, 3, -1, 4, 4)
    assert segment.generation_spans == ((0, 2, "gen-0"), (3, 5, "gen-1"))
    assert build_segments((first, second))[1] == 1


def test_build_segments_opens_a_new_segment_and_counts_fragmentation() -> None:
    """A non-prefix continuation -- including a short partial prefix -- opens a new segment."""
    first = _generation(prompt_ids=(1, 2, 3), output_ids=(4, 5), generation_id="gen-0")
    second = _generation(
        prompt_ids=(9, 10),
        output_ids=(11,),
        logprobs=(-0.5,),
        policy_version=4,
        generation_id="gen-1",
    )
    third = _generation(
        prompt_ids=(1, 2),
        output_ids=(12,),
        logprobs=(-0.6,),
        policy_version=5,
        generation_id="gen-2",
    )

    segments, fragmentation_count = build_segments((first, second, third))

    assert fragmentation_count == 2
    assert [segment.segment_index for segment in segments] == [0, 1, 2]
    assert segments[1].prompt_ids == (9, 10)
    assert segments[1].response_ids == (11,)
    assert segments[1].loss_mask == (1,)
    assert segments[1].generation_spans == ((0, 1, "gen-1"),)
    assert segments[2].prompt_ids == (1, 2)


def test_build_segments_drops_routed_experts_when_one_generation_lacks_them() -> None:
    """R5: experts are carried only when every generation reports them -- never fabricated."""
    head = _generation(prompt_ids=(1, 2, 3), output_ids=(4, 5), generation_id="gen-0")
    tail = _generation(
        prompt_ids=(1, 2, 3, 4, 5, 6),
        output_ids=(7,),
        logprobs=(-0.3,),
        policy_version=4,
        generation_id="gen-1",
        routed_experts=_experts(7),
    )
    rich_head = _generation(
        prompt_ids=(1, 2, 3),
        output_ids=(4, 5),
        generation_id="gen-0",
        routed_experts=_experts(5),
    )

    bare_tail = _generation(
        prompt_ids=(1, 2, 3, 4, 5, 6),
        output_ids=(7,),
        logprobs=(-0.3,),
        policy_version=4,
        generation_id="gen-1",
    )
    # Both pairs MERGE (each tail's prompt extends head prompt+response), so one segment
    # holds a generation without routing and the segment's routing must be None.
    assert build_segments((rich_head, bare_tail))[0][0].routed_experts is None
    assert build_segments((head, tail))[0][0].routed_experts is None


def test_build_segments_carries_routed_experts_when_every_generation_reports_them() -> None:
    """R5: prefix-consistent experts from every generation are carried over prompt+response."""
    rows = _experts(5)
    first = _generation(
        prompt_ids=(1, 2, 3),
        output_ids=(4, 5),
        generation_id="gen-0",
        routed_experts=rows,
    )
    second = _generation(
        prompt_ids=(1, 2, 3, 4, 5, 6),
        output_ids=(7,),
        logprobs=(-0.3,),
        policy_version=4,
        generation_id="gen-1",
        routed_experts=rows + _experts(2, start=100),
    )

    segments, _ = build_segments((first, second))

    assert segments[0].routed_experts is not None
    assert len(segments[0].routed_experts) == 7


def test_build_segments_refuses_an_empty_generation_sequence() -> None:
    """No generation means no segment, no mask and no count -- the call is refused."""
    with pytest.raises(TrajectoryV2Refusal):
        build_segments(())


def test_build_segments_keeps_every_token_verbatim() -> None:
    """Tokens are evidence: prompt and output ids survive a merge byte for byte."""
    first = _generation(
        prompt_ids=(0, 7, 4294967295),
        output_ids=(9, 0),
        logprobs=(-0.1, -0.2),
        generation_id="gen-0",
    )
    second = _generation(
        prompt_ids=(0, 7, 4294967295, 9, 0, 5),
        output_ids=(6,),
        logprobs=(-0.3,),
        policy_version=4,
        generation_id="gen-1",
    )

    segments, _ = build_segments((first, second))

    assert segments[0].prompt_ids == first.prompt_ids
    assert segments[0].response_ids == first.output_ids + (5,) + second.output_ids


# ---------------------------------------------------------------------------
# is_trainable
# ---------------------------------------------------------------------------


def test_is_trainable_follows_the_status_table() -> None:
    """D6: only OK trains outright; every fault status is non-trainable."""
    assert is_trainable(_status_episode(EpisodeStatus.OK)) is True
    for status in (
        EpisodeStatus.INFRA_ERROR,
        EpisodeStatus.SANDBOX_ERROR,
        EpisodeStatus.HARNESS_ERROR,
        EpisodeStatus.CONTEXT_OVERFLOW,
        EpisodeStatus.FIDELITY_MISMATCH,
        EpisodeStatus.VERIFIER_ERROR,
    ):
        assert is_trainable(_status_episode(status)) is False, status


def test_is_trainable_timeout_depends_on_trainable_on_timeout() -> None:
    """D6: TIMEOUT trains only when the episode declares trainable_on_timeout."""
    assert is_trainable(_status_episode(EpisodeStatus.TIMEOUT, trainable_on_timeout=True)) is True
    assert is_trainable(_status_episode(EpisodeStatus.TIMEOUT, trainable_on_timeout=False)) is False


# ---------------------------------------------------------------------------
# episode_reward
# ---------------------------------------------------------------------------


def test_episode_reward_single_returns_none_without_an_episode_event() -> None:
    """Zero EPISODE-scope events is an unmeasured score -- None, never 0.0."""
    assert episode_reward(_episode(reward_events=())) is None
    assert (
        episode_reward(_episode(reward_events=(_reward_event(scope=RewardScope.SEGMENT),))) is None
    )


def test_episode_reward_single_returns_the_one_episode_event_value() -> None:
    """One EPISODE-scope event states the score, and a measured 0.0 stays 0.0."""
    assert episode_reward(_episode(reward_events=(_reward_event(value=1.5),))) == 1.5
    assert episode_reward(_episode(reward_events=(_reward_event(value=0.0),))) == 0.0


def test_episode_reward_single_refuses_two_episode_events_naming_the_count() -> None:
    """Two EPISODE-scope events name no single score, and the refusal names the count 2."""
    events = (_reward_event(value=1.0), _reward_event(value=2.0, at_ts=1.0))
    with pytest.raises(TrajectoryV2Refusal) as excinfo:
        episode_reward(_episode(reward_events=events))
    assert "2" in str(excinfo.value)


def test_episode_reward_single_returns_none_for_an_abstained_event() -> None:
    """An abstaining EPISODE event carries value None and yields None, not a zero."""
    assert episode_reward(_episode(reward_events=(_reward_event(value=None),))) is None


def test_episode_reward_last_by_ts_takes_the_latest_episode_event() -> None:
    """last_by_ts orders the EPISODE events by at_ts and keeps the newest value."""
    events = (_reward_event(value=1.0, at_ts=5.0), _reward_event(value=2.0, at_ts=1.0))
    assert episode_reward(_episode(reward_events=events), reducer="last_by_ts") == 1.0


def test_episode_reward_max_ignores_none_values() -> None:
    """max reads only measured values, and all-None abstentions reduce to None."""
    events = (
        _reward_event(value=None),
        _reward_event(value=1.0, at_ts=1.0),
        _reward_event(value=3.0, at_ts=2.0),
        _reward_event(value=2.0, at_ts=3.0),
    )
    assert episode_reward(_episode(reward_events=events), reducer="max") == 3.0
    assert (
        episode_reward(
            _episode(reward_events=(_reward_event(value=None), _reward_event(value=None))),
            reducer="max",
        )
        is None
    )


# ---------------------------------------------------------------------------
# to_json / from_json
# ---------------------------------------------------------------------------


def test_to_json_keeps_nan_out_of_the_payload_and_enums_as_values() -> None:
    """NaN logprobs serialise as None and enums as their values, so the payload is plain data."""
    payload = to_json(_episode(segments=(_segment_from_mask((1, 0), (-0.1, math.nan), (3, -1)),)))

    leaves = _leaves(payload)
    assert isinstance(payload, dict)
    assert not any(isinstance(value, float) and math.isnan(value) for value in leaves)
    assert not any(isinstance(value, Enum) for value in leaves)
    assert None in leaves
    assert "ok" in leaves


def test_to_json_from_json_round_trips_exactly_including_nan_logprobs() -> None:
    """from_json is the exact inverse of to_json, NaN masked logprobs included."""
    episode = _episode(
        segments=(
            _segment_from_mask((1, 0), (-0.1, math.nan), (3, -1), routed_experts=_experts(4)),
        ),
        reward_events=(
            _reward_event(
                scope=RewardScope.TOKEN_SPAN,
                span=(0, 1),
                value=None,
                at_ts=2.5,
                delayed=True,
                details={"note": "abstained"},
            ),
        ),
        discarded=(
            DiscardedGeneration(
                generation=_generation(generation_id="gen-9", routed_experts=_experts(5)),
                reason="spliced away by the merge",
            ),
        ),
        sandbox=SandboxMeta(network_policy="none", cpu=1.5, exit_code=0),
        timings=Timings(queue_ms=1.0, init_ms=None, run_ms=2.0, reward_ms=0.0),
        fidelity=Fidelity(status=FidelityStatus.UNVERIFIED),
        trainable_on_timeout=True,
        tool_schema_sha256="abc",
    )

    restored = from_json(to_json(episode))

    assert _same(restored, episode)
    assert to_json(restored) == to_json(episode)


# ---------------------------------------------------------------------------
# v1 bridges
# ---------------------------------------------------------------------------


def test_from_v1_marks_the_fidelity_unverified_and_records_one_episode_reward() -> None:
    """R11: a converted v1 trajectory is one segment of UNVERIFIED fidelity with one reward."""
    episode = from_v1(
        _v1_trajectory(),
        model=_model(),
        harness=_harness(),
        env=_env(),
        gateway=_gateway(),
    )

    assert episode.fidelity.status is FidelityStatus.UNVERIFIED
    assert episode.group_id == "group-1"
    assert episode.task_prompt_ids == (1, 2, 3)
    assert len(episode.segments) == 1
    assert len(episode.reward_events) == 1
    assert episode.reward_events[0].scope is RewardScope.EPISODE
    assert episode_reward(episode) == 1.5


def test_to_v1_refuses_an_unobserved_parse_verdict() -> None:
    """R1: v1 states a parse verdict, so an UNOBSERVED one cannot be carried across."""
    episode = _episode(
        discarded=(
            DiscardedGeneration(
                generation=_generation(
                    generation_id="gen-9",
                    tool_calls=(_tool_call_v2(parse_verdict=ParseVerdict.UNOBSERVED),),
                ),
                reason="spliced away by the merge",
            ),
        )
    )

    with pytest.raises(TrajectoryV2Refusal):
        to_v1(episode)


def test_to_v1_refuses_a_multi_segment_episode() -> None:
    """A v1 trajectory is one stream, so a multi-segment episode must go through flatten_v2."""
    assert isinstance(to_v1(_episode()), contracts.Trajectory)
    with pytest.raises(TrajectoryV2Refusal):
        to_v1(_episode(segments=_two_segments()))


# ---------------------------------------------------------------------------
# flatten_v2
# ---------------------------------------------------------------------------


def test_flatten_v2_emits_one_row_per_segment_with_extras_aligned_per_row() -> None:
    """R10: a two-segment episode yields two rows and one extras entry per row."""
    segments = _two_segments()

    batch, extras = flatten_v2((_episode(segments=segments),))

    assert set(extras) == {"policy_version", "segment_index", "episode_id"}
    assert len(batch) == 2
    assert list(extras["segment_index"]) == [0, 1]
    assert list(extras["episode_id"]) == ["ep-1", "ep-1"]
    assert tuple(extras["policy_version"][0]) == segments[0].policy_version
    # R7: segment 1's prompt is rewritten history, so it rides on the response side as
    # non-generated context (mask 0, version -1); segment 0's prompt IS the task prompt.
    ctx = segments[1].prompt_ids
    assert tuple(extras["policy_version"][1]) == (-1,) * len(ctx) + segments[1].policy_version
    assert batch.column("response_ids") == (
        segments[0].response_ids,
        ctx + segments[1].response_ids,
    )
    assert batch.column("loss_mask") == (
        segments[0].loss_mask,
        (0,) * len(ctx) + segments[1].loss_mask,
    )


def test_flatten_v2_masks_non_trainable_rows_without_dropping_them() -> None:
    """R10: a non-trainable episode keeps its rows with an all-0 loss mask and no reward."""
    trainable = _episode()
    broken = _status_episode(EpisodeStatus.INFRA_ERROR, episode_id="ep-2")

    batch, extras = flatten_v2((trainable, broken))

    assert len(batch) == 2
    assert list(extras["episode_id"]) == ["ep-1", "ep-2"]
    assert batch.column("loss_mask")[0] == (1, 1)
    assert batch.column("loss_mask")[1] == (0, 0)
    assert batch.column("reward")[1] is None


def test_flatten_v2_never_marks_merged_context_as_generated() -> None:
    """A merged segment's tool-result span stays non-generated with no 0.0 logprob filler."""
    head = _generation(prompt_ids=(1, 2, 3), output_ids=(4, 5), generation_id="gen-0")
    tail = _generation(
        prompt_ids=(1, 2, 3, 4, 5, 6),
        output_ids=(7,),
        logprobs=(-0.3,),
        policy_version=4,
        generation_id="gen-1",
    )
    (segment,), fragments = build_segments((head, tail))
    assert fragments == 0
    assert segment.loss_mask == (1, 1, 0, 1)

    batch, _extras = flatten_v2((_episode(segments=(segment,)),))

    assert batch.column("loss_mask") == ((1, 1, 0, 1),)
    (rollout,) = batch.column("rollout_logprobs")
    assert rollout[2] is None
    assert all(value is not None for index, value in enumerate(rollout) if index != 2)


def test_flatten_v2_policy_bounds_come_from_sampled_tokens_only() -> None:
    """policy_version_min/max ignore the -1 placed on context tokens."""
    head = _generation(prompt_ids=(1, 2, 3), output_ids=(4, 5), generation_id="gen-0")
    tail = _generation(
        prompt_ids=(1, 2, 3, 4, 5, 6),
        output_ids=(7,),
        logprobs=(-0.3,),
        policy_version=4,
        generation_id="gen-1",
    )
    (segment,), _ = build_segments((head, tail))
    _batch, extras = flatten_v2((_episode(segments=(segment,)),))
    sampled = [
        v for v, m in zip(extras["policy_version"][0], segment.loss_mask, strict=True) if m == 1
    ]
    assert min(sampled) >= 0


def test_from_v1_keeps_a_v1_timeout_visible_and_trainable() -> None:
    """A scored v1 TIMEOUT row becomes v2 TIMEOUT that still trains, never a silent OK."""
    episode = from_v1(
        _v1_trajectory(termination=contracts.Termination.TIMEOUT),
        model=_model(),
        harness=_harness(),
        env=_env(),
        gateway=_gateway(),
    )

    assert episode.status is EpisodeStatus.TIMEOUT
    assert episode.trainable_on_timeout is True
    assert is_trainable(episode)


def test_freeze_mapping_refuses_a_non_mapping_field() -> None:
    """A Mapping field must be a Mapping -- a list is refused naming the field."""
    with pytest.raises(TrajectoryV2Refusal, match="sampling_requested: expected a Mapping"):
        _generation(sampling_requested=[("temperature", 0.7)])


def test_as_tuple_refuses_a_non_sequence_field() -> None:
    """A sequence field must be a sequence -- an int is refused naming the field."""
    with pytest.raises(TrajectoryV2Refusal, match="prompt_ids: expected a sequence"):
        _generation(prompt_ids=7)


def test_as_tuple_refuses_a_string_where_a_sequence_is_required() -> None:
    """A str is not a token sequence -- it is refused as a sequence field."""
    with pytest.raises(TrajectoryV2Refusal, match="output_ids: expected a sequence"):
        _generation(output_ids="ab")


def test_tool_call_name_must_be_none_or_non_empty() -> None:
    """A tool call name is None or a non-empty string -- an empty name is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="ToolCallV2.name: expected None or non-empty"):
        _tool_call_v2(name="")


def test_tool_call_arguments_raw_must_be_none_or_str() -> None:
    """arguments_raw is None or a str -- a dict of parsed arguments is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="arguments_raw: expected None or str"):
        _tool_call_v2(arguments_raw={"q": 1})


def test_tool_call_parse_verdict_must_be_a_parse_verdict() -> None:
    """parse_verdict is a ParseVerdict enum member -- a bare string is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="parse_verdict: expected ParseVerdict"):
        _tool_call_v2(parse_verdict="observed_accepted")


def test_tool_call_parse_error_must_be_none_or_str() -> None:
    """parse_error is None or a str -- an int code is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="parse_error: expected None or str"):
        _tool_call_v2(parse_error=12)


def test_generation_finish_reason_must_be_none_or_str() -> None:
    """finish_reason is None or a str -- an enum is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="finish_reason: expected None or str"):
        _generation(finish_reason=42)


def test_generation_tool_calls_must_all_be_tool_call_v2() -> None:
    """Every tool call entry is a ToolCallV2 -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="tool_calls: 1 of 2 entries are not ToolCallV2"):
        _generation(tool_calls=(_tool_call_v2(), object()))


def test_generation_routed_experts_row_count_must_cover_prompt_plus_output() -> None:
    """R5: routed_experts rows cover prompt+output exactly -- both counts are named."""
    with pytest.raises(TrajectoryV2Refusal, match="routed_experts: 4 of 5 rows"):
        _generation(routed_experts=_experts(4))


def test_generation_routed_experts_expert_ids_must_be_non_negative_ints() -> None:
    """Expert ids are ints >= 0 -- a negative id is refused naming both counts."""
    rows = (((-1, 2), (3,)),) + _experts(4, start=10)
    with pytest.raises(TrajectoryV2Refusal, match="routed_experts\\[0\\]\\[0\\]: 1 of 2"):
        _generation(routed_experts=rows)


def test_segment_prompt_ids_must_be_non_empty() -> None:
    """A segment prompt is non-empty -- 0 of 0 tokens is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="prompt_ids: expected non-empty tuple"):
        _segment(prompt_ids=())


def test_segment_prompt_ids_tokens_must_be_non_negative_ints() -> None:
    """Segment prompt tokens are ints >= 0 -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="prompt_ids: 1 of 3 tokens are not ints"):
        _segment(prompt_ids=(1, -2, 3))


def test_segment_response_ids_must_be_non_empty() -> None:
    """A segment response is non-empty -- 0 of 0 tokens is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="response_ids: expected non-empty tuple"):
        _segment(response_ids=())


def test_segment_response_ids_tokens_must_be_non_negative_ints() -> None:
    """Segment response tokens are ints >= 0 -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="response_ids: 1 of 2 tokens"):
        _segment(response_ids=(4, -5))


def test_segment_loss_mask_length_must_match_response_ids() -> None:
    """len(loss_mask) equals len(response_ids) -- both counts are named."""
    with pytest.raises(TrajectoryV2Refusal, match="loss_mask: 3 of 2 entries"):
        _segment(loss_mask=(1, 1, 1))


def test_segment_loss_mask_entries_must_be_zero_or_one_ints() -> None:
    """A loss mask entry is a 0/1 int -- a bool is refused as not an int here."""
    with pytest.raises(TrajectoryV2Refusal, match="loss_mask: 1 of 2 entries are not 0/1"):
        _segment(loss_mask=(1, True))


def test_segment_logprobs_length_must_match_response_ids() -> None:
    """len(logprobs) equals len(response_ids) -- both counts are named."""
    with pytest.raises(TrajectoryV2Refusal, match="logprobs: 1 of 2 entries"):
        _segment(logprobs=(-0.1,))


def test_segment_logprobs_r4_rule_violation_names_both_counts() -> None:
    """R4: finite where mask==1, NaN where mask==0 -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="logprobs: 1 of 2 entries violate R4"):
        _segment_from_mask((1, 0), (-0.1, -0.5), (3, -1))


def test_segment_policy_version_length_must_match_response_ids() -> None:
    """len(policy_version) equals len(response_ids) -- both counts are named."""
    with pytest.raises(TrajectoryV2Refusal, match="policy_version: 3 of 2 entries"):
        _segment(policy_version=(3, 3, 3))


def test_segment_policy_version_r4_rule_violation_names_both_counts() -> None:
    """R4: >= 0 where mask==1, exactly -1 where mask==0 -- both counts are named."""
    with pytest.raises(TrajectoryV2Refusal, match="policy_version: 1 of 2 entries violate R4"):
        _segment_from_mask((1, 0), (-0.1, math.nan), (3, 0))


def test_segment_generation_span_must_be_a_three_tuple() -> None:
    """A generation span is (start, end, generation_id) -- a two-tuple is refused."""
    with pytest.raises(
        TrajectoryV2Refusal, match="generation_spans\\[0\\]: expected \\(start, end"
    ):
        _segment(generation_spans=((0, 2),))


def test_segment_generation_span_bounds_must_be_ints() -> None:
    """Span start and end are ints -- float bounds are refused."""
    with pytest.raises(TrajectoryV2Refusal, match="start/end must be ints"):
        _segment(generation_spans=((0.0, 2.0, "gen-0"),))


def test_segment_generation_span_bounds_must_be_ordered_within_the_response() -> None:
    """A span satisfies 0 <= start < end <= n -- an inverted span is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="expected 0 <= start < end <= 2"):
        _segment(generation_spans=((2, 1, "gen-0"),))


def test_segment_generation_spans_must_be_ascending_and_non_overlapping() -> None:
    """Spans ascend without overlap -- a descending start is refused naming both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="spans must be ascending and non-overlapping"):
        _segment_from_mask(
            (1, 1, 1, 1),
            (-0.1, -0.2, -0.3, -0.4),
            (3, 3, 3, 3),
            generation_spans=((0, 3, "gen-0"), (2, 4, "gen-1")),
        )


def test_segment_generation_span_generation_id_must_be_non_empty() -> None:
    """A span names its generation -- an empty generation_id is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="generation_id: expected non-empty str"):
        _segment(generation_spans=((0, 2, ""),))


def test_segment_generation_spans_cover_exactly_the_mask_one_positions() -> None:
    """Spans cover exactly the trainable positions -- both counts are named."""
    with pytest.raises(TrajectoryV2Refusal, match="cover exactly the mask==1 positions"):
        _segment_from_mask((1, 0, 1), (-0.1, math.nan, -0.2), (3, -1, 3), generation_spans=())


def test_segment_routed_experts_expert_ids_must_be_non_negative_ints() -> None:
    """Segment expert ids are ints >= 0 -- a float id is refused naming both counts."""
    rows = (((1.5, 2), (3,)),) + _experts(4, start=10)
    with pytest.raises(TrajectoryV2Refusal, match="routed_experts\\[0\\]\\[0\\]: 1 of 2"):
        _segment(routed_experts=rows)


def test_discarded_generation_must_hold_a_generation() -> None:
    """A discarded entry holds a Generation -- a bare object is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="generation: expected Generation"):
        DiscardedGeneration(generation=object(), reason="spliced away")


def test_discarded_generation_reason_must_be_non_empty() -> None:
    """A discarded generation names its reason -- an empty reason is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="reason: expected non-empty str"):
        DiscardedGeneration(generation=_generation(), reason="")


def test_reward_event_scope_must_be_a_reward_scope() -> None:
    """scope is a RewardScope enum member -- a bare string is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="scope: expected RewardScope"):
        _reward_event(scope="episode")


def test_reward_event_span_must_be_exactly_two_ints() -> None:
    """A TOKEN_SPAN span is exactly (start, end) -- a three-tuple is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="span: expected \\(start, end\\)"):
        _reward_event(scope=RewardScope.TOKEN_SPAN, span=(0, 1, 2))


def test_reward_event_span_bounds_must_be_ints() -> None:
    """Span bounds are ints -- float bounds are refused naming both values."""
    with pytest.raises(TrajectoryV2Refusal, match="span: start/end must be ints"):
        _reward_event(scope=RewardScope.TOKEN_SPAN, span=(0.0, 1.0))


def test_reward_event_delayed_must_be_a_bool() -> None:
    """delayed is a bool -- the int 0 is refused (bool is not int here)."""
    with pytest.raises(TrajectoryV2Refusal, match="delayed: expected bool"):
        _reward_event(delayed=0)


def test_model_meta_name_must_be_non_empty() -> None:
    """A model names itself -- an empty name is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="ModelMeta.name: expected non-empty str"):
        ModelMeta(name="", version=0, checkpoint_sha="sha-1")


def test_model_meta_version_must_be_a_non_negative_int() -> None:
    """A model version is an int >= 0 -- a bool is refused naming the value."""
    with pytest.raises(TrajectoryV2Refusal, match="ModelMeta.version: expected int >= 0"):
        ModelMeta(name="m", version=True, checkpoint_sha="sha-1")


def test_model_meta_checkpoint_sha_must_be_non_empty() -> None:
    """A model names its checkpoint -- an empty sha is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="checkpoint_sha: expected non-empty str"):
        ModelMeta(name="m", version=0, checkpoint_sha="")


def test_harness_meta_name_must_be_non_empty() -> None:
    """A harness names itself -- an empty name is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="HarnessMeta.name: expected non-empty str"):
        HarnessMeta(name="", version="1")


def test_harness_meta_version_must_be_non_empty() -> None:
    """A harness names its version -- an empty version is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="HarnessMeta.version: expected non-empty str"):
        HarnessMeta(name="h", version="")


def test_harness_meta_commit_must_be_none_or_non_empty() -> None:
    """A harness commit is None or non-empty -- an empty commit is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="commit: expected None or non-empty str"):
        HarnessMeta(name="h", version="1", commit="")


def test_harness_meta_license_must_be_none_or_non_empty() -> None:
    """A harness license is None or non-empty -- an empty license is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="license: expected None or non-empty str"):
        HarnessMeta(name="h", version="1", license="")


def test_env_meta_kind_must_be_an_env_kind() -> None:
    """kind is an EnvKind enum member -- a bare string is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="EnvMeta.kind: expected EnvKind"):
        EnvMeta(kind="fs_local", task_id="task-1")


def test_env_meta_task_id_must_be_non_empty() -> None:
    """An env names its task -- an empty task_id is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="task_id: expected non-empty str"):
        EnvMeta(kind=EnvKind.FS_LOCAL, task_id="")


def test_env_meta_task_version_must_be_none_or_non_empty() -> None:
    """A task version is None or non-empty -- an empty version is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="task_version: expected None or non-empty str"):
        EnvMeta(kind=EnvKind.FS_LOCAL, task_id="task-1", task_version="")


def test_env_meta_image_digest_must_be_none_or_non_empty() -> None:
    """An image digest is None or non-empty -- an empty digest is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="image_digest: expected None or non-empty str"):
        EnvMeta(kind=EnvKind.FS_LOCAL, task_id="task-1", image_digest="")


def test_sandbox_meta_network_policy_must_be_non_empty() -> None:
    """A sandbox states its network policy -- an empty policy is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="network_policy: expected non-empty str"):
        SandboxMeta(network_policy="")


def test_sandbox_meta_node_must_be_none_or_non_empty() -> None:
    """A sandbox node is None or non-empty -- an empty node is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="node: expected None or non-empty str"):
        SandboxMeta(network_policy="none", node="")


def test_sandbox_meta_cpu_must_be_none_or_non_negative_finite() -> None:
    """A cpu measurement is None or finite >= 0 -- nan and negatives are refused."""
    for cpu in (float("nan"), -0.5, True):
        with pytest.raises(TrajectoryV2Refusal, match="cpu: expected None or finite float"):
            SandboxMeta(network_policy="none", cpu=cpu)


def test_sandbox_meta_mem_mb_must_be_none_or_non_negative_int() -> None:
    """A memory measurement is None or an int >= 0 -- a float is refused naming the value."""
    with pytest.raises(TrajectoryV2Refusal, match="mem_mb: expected None or int >= 0"):
        SandboxMeta(network_policy="none", mem_mb=1.5)


def test_sandbox_meta_started_at_must_be_none_or_non_negative_finite() -> None:
    """A start timestamp is None or finite >= 0 -- a negative value is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="started_at: expected None or finite float"):
        SandboxMeta(network_policy="none", started_at=-1.0)


def test_sandbox_meta_ended_at_must_be_none_or_non_negative_finite() -> None:
    """An end timestamp is None or finite >= 0 -- inf is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="ended_at: expected None or finite float"):
        SandboxMeta(network_policy="none", ended_at=float("inf"))


def test_sandbox_meta_exit_code_must_be_none_or_int() -> None:
    """An exit code is None or an int -- a bool is refused naming the value."""
    with pytest.raises(TrajectoryV2Refusal, match="exit_code: expected None or int"):
        SandboxMeta(network_policy="none", exit_code=True)


def test_gateway_meta_api_format_must_be_an_api_format() -> None:
    """api_format is an ApiFormat enum member -- a bare string is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="api_format: expected ApiFormat"):
        GatewayMeta(api_format="chat_completions", session_id="s")


def test_gateway_meta_session_id_must_be_non_empty() -> None:
    """A gateway names its session -- an empty session_id is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="session_id: expected non-empty str"):
        GatewayMeta(api_format=ApiFormat.CHAT_COMPLETIONS, session_id="")


def test_timings_fields_must_be_none_or_non_negative_finite() -> None:
    """Every timing is None or finite >= 0 -- a negative run_ms is refused naming the field."""
    with pytest.raises(TrajectoryV2Refusal, match="Timings.run_ms: expected None or finite float"):
        Timings(run_ms=-1.0)


def test_fidelity_status_must_be_a_fidelity_status() -> None:
    """A fidelity status is a FidelityStatus enum member -- a bare string is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Fidelity.status: expected FidelityStatus"):
        Fidelity(status="verified")


def test_fidelity_detail_must_be_none_or_str() -> None:
    """A fidelity detail is None or a str -- a dict is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Fidelity.detail: expected None or str"):
        Fidelity(status=FidelityStatus.VERIFIED, detail={"why": "drift"})


def test_episode_task_prompt_ids_tokens_must_be_non_negative_ints() -> None:
    """Task prompt tokens are ints >= 0 -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="task_prompt_ids: 1 of 3 tokens"):
        _episode(task_prompt_ids=(1, -2, 3))


def test_episode_status_must_be_an_episode_status() -> None:
    """status is an EpisodeStatus enum member -- a bare string is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Episode.status: expected EpisodeStatus"):
        _episode(status="ok")


def test_episode_termination_must_be_a_termination() -> None:
    """termination is a Termination enum member -- a bare string is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Episode.termination: expected Termination"):
        _episode(termination="stop")


def test_episode_segments_must_all_be_segments() -> None:
    """Every segment entry is a Segment -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="segments: 1 of 2 entries are not Segment"):
        _episode(segments=(_segment(), object()))


def test_episode_reward_events_must_all_be_reward_events() -> None:
    """Every reward entry is a RewardEvent -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="reward_events: 1 of 1 entries"):
        _episode(reward_events=(object(),))


def test_episode_discarded_must_all_be_discarded_generations() -> None:
    """Every discarded entry is a DiscardedGeneration -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="discarded: 1 of 1 entries"):
        _episode(discarded=(object(),))


def test_episode_model_must_be_model_meta() -> None:
    """The model field holds a ModelMeta -- a dict is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Episode.model: expected ModelMeta"):
        _episode(model={"name": "m"})


def test_episode_harness_must_be_harness_meta() -> None:
    """The harness field holds a HarnessMeta -- a dict is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Episode.harness: expected HarnessMeta"):
        _episode(harness={"name": "h"})


def test_episode_env_must_be_env_meta() -> None:
    """The env field holds an EnvMeta -- a dict is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Episode.env: expected EnvMeta"):
        _episode(env={"task_id": "t"})


def test_episode_gateway_must_be_gateway_meta() -> None:
    """The gateway field holds a GatewayMeta -- a dict is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Episode.gateway: expected GatewayMeta"):
        _episode(gateway={"session_id": "s"})


def test_episode_sandbox_must_be_sandbox_meta_or_none() -> None:
    """The sandbox field holds a SandboxMeta or None -- a dict is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="sandbox: expected SandboxMeta or None"):
        _episode(sandbox={"network_policy": "none"})


def test_episode_timings_must_be_timings() -> None:
    """The timings field holds a Timings -- a dict is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Episode.timings: expected Timings"):
        _episode(timings={"run_ms": 1.0})


def test_episode_fidelity_must_be_fidelity() -> None:
    """The fidelity field holds a Fidelity -- a dict is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="Episode.fidelity: expected Fidelity"):
        _episode(fidelity={"status": "verified"})


def test_episode_trainable_on_timeout_must_be_a_bool() -> None:
    """trainable_on_timeout is a bool -- the int 0 is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="trainable_on_timeout: expected bool"):
        _episode(trainable_on_timeout=0)


def test_episode_tool_schema_sha256_must_be_none_or_non_empty() -> None:
    """A tool schema digest is None or non-empty -- an empty digest is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="tool_schema_sha256: expected None or non-empty"):
        _episode(tool_schema_sha256="")


def test_build_segments_generations_must_all_be_generations() -> None:
    """Every generation entry is a Generation -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="generations: 1 of 2 entries are not Generation"):
        build_segments((_generation(), object()))


def test_build_segments_eot_token_ids_must_be_a_frozenset() -> None:
    """eot_token_ids is a frozenset -- a plain set is refused naming its type."""
    with pytest.raises(TrajectoryV2Refusal, match="eot_token_ids: expected frozenset"):
        build_segments((_generation(),), eot_token_ids={77})


def test_build_segments_eot_token_ids_entries_must_be_non_negative_ints() -> None:
    """EOT token ids are ints >= 0 -- a negative id is refused naming both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="eot_token_ids: 1 of 2 entries"):
        build_segments((_generation(),), eot_token_ids=frozenset({77, -1}))


def test_build_segments_drops_experts_on_layer_count_mismatch() -> None:
    """R5: a layer-count mismatch across merged generations drops the segment's experts."""
    head = _generation(
        prompt_ids=(1, 2, 3),
        output_ids=(4, 5),
        generation_id="gen-0",
        routed_experts=_experts(5),
    )
    tail_rows = tuple(((i, i + 1),) for i in range(7))
    tail = _generation(
        prompt_ids=(1, 2, 3, 4, 5, 6),
        output_ids=(7,),
        logprobs=(-0.3,),
        policy_version=4,
        generation_id="gen-1",
        routed_experts=tail_rows,
    )

    (segment,), _ = build_segments((head, tail))

    assert segment.routed_experts is None


def test_build_segments_drops_experts_on_prefix_inconsistency() -> None:
    """R5: experts disagreeing on the shared prefix are dropped, never reconciled."""
    head = _generation(
        prompt_ids=(1, 2, 3),
        output_ids=(4, 5),
        generation_id="gen-0",
        routed_experts=_experts(5),
    )
    tail = _generation(
        prompt_ids=(1, 2, 3, 4, 5, 6),
        output_ids=(7,),
        logprobs=(-0.3,),
        policy_version=4,
        generation_id="gen-1",
        routed_experts=_experts(7, start=500),
    )

    (segment,), _ = build_segments((head, tail))

    assert segment.routed_experts is None


def test_validate_episode_refuses_a_non_episode() -> None:
    """validate_episode takes an Episode -- a dict is refused naming its type."""
    with pytest.raises(TrajectoryV2Refusal, match="validate_episode.ep: expected Episode"):
        validate_episode({"episode_id": "ep-1"})


def test_validate_episode_refuses_duplicate_generation_ids_across_segments() -> None:
    """Generation ids are unique across segments -- the refusal names both locations."""
    segments = (
        _segment(),
        _segment(
            segment_index=1,
            prompt_ids=(9,),
            response_ids=(8,),
            loss_mask=(1,),
            logprobs=(-0.5,),
            policy_version=(4,),
            generation_spans=((0, 1, "gen-0"),),
        ),
    )
    with pytest.raises(TrajectoryV2Refusal, match="generation id 'gen-0' appears in"):
        validate_episode(_episode(segments=segments))


def test_validate_episode_refuses_a_generation_id_shared_with_a_discarded_entry() -> None:
    """A discarded generation id never collides with a span's id -- both locations are named."""
    episode = _episode(
        discarded=(
            DiscardedGeneration(generation=_generation(generation_id="gen-0"), reason="spliced"),
        )
    )
    with pytest.raises(TrajectoryV2Refusal, match="discarded\\[0\\]"):
        validate_episode(episode)


def test_is_trainable_refuses_a_non_episode() -> None:
    """is_trainable takes an Episode -- a list is refused naming its type."""
    with pytest.raises(TrajectoryV2Refusal, match="is_trainable.ep: expected Episode"):
        is_trainable([])


def test_episode_reward_refuses_a_non_episode() -> None:
    """episode_reward takes an Episode -- a string is refused naming its type."""
    with pytest.raises(TrajectoryV2Refusal, match="episode_reward.ep: expected Episode"):
        episode_reward("ep-1")


def test_episode_reward_refuses_a_non_string_reducer() -> None:
    """The reducer is a str -- a callable is refused naming its type."""
    with pytest.raises(TrajectoryV2Refusal, match="reducer: expected str"):
        episode_reward(_episode(), reducer=max)


def test_episode_reward_refuses_an_unknown_reducer() -> None:
    """Only the three declared reducers exist -- 'median' is refused naming the name."""
    with pytest.raises(TrajectoryV2Refusal, match="unknown reducer 'median'"):
        episode_reward(_episode(), reducer="median")


def test_episode_reward_last_by_ts_returns_none_without_events() -> None:
    """last_by_ts with no EPISODE event is unmeasured -- None, never 0.0."""
    assert episode_reward(_episode(reward_events=()), reducer="last_by_ts") is None


def test_episode_reward_last_by_ts_breaks_ties_toward_the_last_event() -> None:
    """Equal timestamps resolve to the last occurrence in the event order."""
    events = (_reward_event(value=1.0, at_ts=2.0), _reward_event(value=2.0, at_ts=2.0))
    assert episode_reward(_episode(reward_events=events), reducer="last_by_ts") == 2.0


def test_episode_reward_max_returns_none_without_measured_values() -> None:
    """max with no EPISODE events is unmeasured -- None, never 0.0."""
    assert episode_reward(_episode(reward_events=()), reducer="max") is None


def test_to_json_refuses_a_non_episode() -> None:
    """to_json takes an Episode -- an int is refused naming its type."""
    with pytest.raises(TrajectoryV2Refusal, match="to_json.ep: expected Episode"):
        to_json(3)


def test_to_json_serialises_experts_as_plain_nested_lists() -> None:
    """Routed experts serialise to plain nested lists of ints, never tuples."""
    episode = _episode(segments=(_segment(routed_experts=_experts(5)),))
    rows = to_json(episode)["segments"][0]["routed_experts"]
    assert isinstance(rows, list)
    assert all(isinstance(layer, list) for row in rows for layer in row)


def test_from_json_refuses_a_non_mapping() -> None:
    """from_json takes a Mapping -- a list is refused naming its type."""
    with pytest.raises(TrajectoryV2Refusal, match="from_json.d: expected a Mapping"):
        from_json([("episode_id", "ep-1")])


def test_from_json_restores_tool_calls_with_a_default_model_origin() -> None:
    """A payload without call_id_origin restores a tool call with origin 'model'."""
    payload = to_json(_episode())
    payload["discarded"] = [
        {
            "generation": {
                "generation_id": "gen-9",
                "prompt_ids": [1, 2, 3],
                "output_ids": [4, 5],
                "logprobs": [-0.1, -0.2],
                "policy_version": 3,
                "finish_reason": None,
                "sampling_requested": {},
                "sampling_overrides": {},
                "tool_calls": [
                    {
                        "call_id": "call-1",
                        "name": "search",
                        "arguments_raw": "{}",
                        "parse_verdict": "observed_accepted",
                        "parse_error": None,
                    }
                ],
                "routed_experts": None,
            },
            "reason": "spliced away",
        }
    ]

    restored = from_json(payload)

    (discarded,) = restored.discarded
    (tool_call,) = discarded.generation.tool_calls
    assert tool_call.call_id_origin == "model"


def test_from_json_restores_a_none_masked_logprob_as_nan() -> None:
    """A None logprob at a mask-0 position restores to math.nan, never 0.0."""
    payload = to_json(_episode(segments=(_segment_from_mask((1, 0), (-0.1, math.nan), (3, -1)),)))
    assert payload["segments"][0]["logprobs"][1] is None

    restored = from_json(payload)

    assert math.isnan(restored.segments[0].logprobs[1])


def test_from_v1_refuses_a_non_trajectory() -> None:
    """from_v1 takes a v1 Trajectory -- a dict is refused naming its type."""
    with pytest.raises(TrajectoryV2Refusal, match="from_v1.traj: expected contracts.Trajectory"):
        from_v1(
            {"uid": "group-1"}, model=_model(), harness=_harness(), env=_env(), gateway=_gateway()
        )


def test_from_v1_masks_unsupervised_turns_with_nan_and_minus_one() -> None:
    """A non-generated v1 turn contributes mask 0, NaN logprobs and policy version -1."""
    trajectory = _v1_trajectory(
        turns=(
            contracts.Turn(
                index=1,
                kind=contracts.SegmentKind.OBSERVATION,
                token_ids=(4,),
                generated=False,
                logprobs=None,
            ),
            contracts.Turn(
                index=2,
                kind=contracts.SegmentKind.ASSISTANT,
                token_ids=(5,),
                generated=True,
                logprobs=(-0.2,),
            ),
        )
    )

    episode = from_v1(
        trajectory, model=_model(), harness=_harness(), env=_env(), gateway=_gateway()
    )

    segment = episode.segments[0]
    assert segment.loss_mask == (0, 1)
    assert math.isnan(segment.logprobs[0])
    assert segment.policy_version == (-1, 0)


def test_from_v1_maps_infra_and_aborted_terminations_to_infra_error() -> None:
    """R11: v1 INFRA and ABORTED both become INFRA_ERROR with their own termination."""
    infra = from_v1(
        _v1_trajectory(
            termination=contracts.Termination.INFRA,
            reward=None,
            abstention_reason="infra: sandbox lost",
        ),
        model=_model(),
        harness=_harness(),
        env=_env(),
        gateway=_gateway(),
    )
    aborted = from_v1(
        _v1_trajectory(
            termination=contracts.Termination.ABORTED,
            reward=None,
            abstention_reason="infra: aborted before a sample",
        ),
        model=_model(),
        harness=_harness(),
        env=_env(),
        gateway=_gateway(),
    )

    assert (infra.status, infra.termination) == (EpisodeStatus.INFRA_ERROR, Termination.ERROR)
    assert (aborted.status, aborted.termination) == (
        EpisodeStatus.INFRA_ERROR,
        Termination.ABORTED,
    )


def test_from_v1_maps_length_and_step_limit_terminations() -> None:
    """R11: v1 LENGTH and STEP_LIMIT map to v2 LENGTH and TURN_LIMIT on an OK episode."""
    length = from_v1(
        _v1_trajectory(termination=contracts.Termination.LENGTH),
        model=_model(),
        harness=_harness(),
        env=_env(),
        gateway=_gateway(),
    )
    step = from_v1(
        _v1_trajectory(termination=contracts.Termination.STEP_LIMIT),
        model=_model(),
        harness=_harness(),
        env=_env(),
        gateway=_gateway(),
    )

    assert (length.status, length.termination) == (EpisodeStatus.OK, Termination.LENGTH)
    assert (step.status, step.termination) == (EpisodeStatus.OK, Termination.TURN_LIMIT)


def test_from_v1_records_no_reward_event_for_a_v1_abstention() -> None:
    """A v1 abstention becomes no reward event at all -- never a 0.0 placeholder."""
    episode = from_v1(
        _v1_trajectory(reward=None, abstention_reason="no verifier"),
        model=_model(),
        harness=_harness(),
        env=_env(),
        gateway=_gateway(),
    )

    assert episode.reward_events == ()
    assert episode_reward(episode) is None


def test_to_v1_refuses_a_non_episode() -> None:
    """to_v1 takes an Episode -- a tuple is refused naming its type."""
    with pytest.raises(TrajectoryV2Refusal, match="to_v1.ep: expected Episode"):
        to_v1(())


def test_to_v1_maps_each_trainable_termination_onto_a_v1_termination() -> None:
    """R11: STOP, LENGTH and TURN_LIMIT map to v1 STOP, LENGTH and STEP_LIMIT."""
    for termination, expected in (
        (Termination.STOP, contracts.Termination.STOP),
        (Termination.LENGTH, contracts.Termination.LENGTH),
        (Termination.TURN_LIMIT, contracts.Termination.STEP_LIMIT),
        (Termination.ABORTED, contracts.Termination.ABORTED),
        (Termination.ERROR, contracts.Termination.STOP),
    ):
        episode = _episode(termination=termination)
        assert to_v1(episode).termination is expected, termination


def test_to_v1_maps_non_trainable_episodes_to_infra_with_an_abstention() -> None:
    """A non-trainable episode becomes v1 INFRA or ABORTED with no reward and a reason."""
    infra = to_v1(_status_episode(EpisodeStatus.INFRA_ERROR))
    aborted = to_v1(_status_episode(EpisodeStatus.INFRA_ERROR, termination=Termination.ABORTED))

    assert infra.termination is contracts.Termination.INFRA
    assert infra.reward is None
    assert infra.abstention_reason.startswith("infra:")
    assert aborted.termination is contracts.Termination.ABORTED
    assert aborted.abstention_reason == "infra: aborted before a sample"


def test_to_v1_states_an_abstention_when_no_episode_reward_is_measured() -> None:
    """A trainable episode without an EPISODE reward abstains rather than scoring 0.0."""
    trajectory = to_v1(_episode(reward_events=()))

    assert trajectory.reward is None
    assert trajectory.abstention_reason == "v2: no EPISODE-scope reward event"


def test_to_v1_policy_bounds_are_none_on_an_all_context_segment() -> None:
    """With no sampled tokens the policy bounds are None, never 0."""
    segment = _segment_from_mask((0, 0), (math.nan, math.nan), (-1, -1))
    trajectory = to_v1(
        _episode(
            segments=(segment,), status=EpisodeStatus.INFRA_ERROR, termination=Termination.ERROR
        )
    )

    assert trajectory.policy_version_min is None
    assert trajectory.policy_version_max is None


def test_flatten_v2_refuses_an_empty_episode_sequence() -> None:
    """No episodes means no rows -- 0 of 0 episodes is refused."""
    with pytest.raises(TrajectoryV2Refusal, match="episodes: expected a non-empty sequence"):
        flatten_v2(())


def test_flatten_v2_refuses_non_episode_entries_naming_both_counts() -> None:
    """Every entry is an Episode -- the refusal names both counts."""
    with pytest.raises(TrajectoryV2Refusal, match="episodes: 1 of 2 entries are not Episode"):
        flatten_v2((_episode(), "ep-2"))


def test_flatten_v2_maps_each_trainable_termination_onto_a_v1_termination() -> None:
    """R10: each segment row carries the episode's mapped v1 termination."""
    for termination, expected in (
        (Termination.STOP, contracts.Termination.STOP),
        (Termination.LENGTH, contracts.Termination.LENGTH),
        (Termination.TURN_LIMIT, contracts.Termination.STEP_LIMIT),
        (Termination.ABORTED, contracts.Termination.ABORTED),
        (Termination.ERROR, contracts.Termination.STOP),
    ):
        batch, _ = flatten_v2((_episode(termination=termination),))
        assert batch.column("termination")[0] is expected, termination


def test_flatten_v2_marks_non_trainable_aborted_rows_as_aborted_with_an_abstention() -> None:
    """A non-trainable ABORTED episode keeps its row as v1 ABORTED with an infra reason."""
    episode = _status_episode(
        EpisodeStatus.INFRA_ERROR,
        termination=Termination.ABORTED,
        segments=(_segment_from_mask((0, 0), (math.nan, math.nan), (-1, -1)),),
    )

    batch, _ = flatten_v2((episode,))

    assert batch.column("termination")[0] is contracts.Termination.ABORTED
    assert batch.column("abstention_reason")[0] == "infra: aborted before a sample"
    assert batch.column("reward")[0] is None


def test_flatten_v2_states_an_abstention_when_no_episode_reward_is_measured() -> None:
    """A trainable row without an EPISODE reward abstains rather than scoring 0.0."""
    batch, _ = flatten_v2((_episode(reward_events=()),))

    assert batch.column("reward")[0] is None
    assert batch.column("abstention_reason")[0] == "v2: no EPISODE-scope reward event"


def test_flatten_v2_policy_bounds_are_none_on_an_all_context_segment() -> None:
    """With no sampled tokens the row's policy bounds are None, never 0."""
    segment = _segment_from_mask((0, 0), (math.nan, math.nan), (-1, -1))
    episode = _episode(
        segments=(segment,), status=EpisodeStatus.INFRA_ERROR, termination=Termination.ERROR
    )

    batch, _ = flatten_v2((episode,))

    assert batch.column("policy_version_min")[0] is None
    assert batch.column("policy_version_max")[0] is None


def test_flatten_v2_splits_the_response_into_maximal_mask_runs() -> None:
    """Mask runs become alternating generated and observation turns in order."""
    segment = _segment_from_mask((1, 0, 1), (-0.1, math.nan, -0.2), (3, -1, 3))
    # The task prompt equals the segment prompt, so R7 adds no context prefix here.
    batch, _ = flatten_v2((_episode(segments=(segment,), task_prompt_ids=segment.prompt_ids),))

    (rollout,) = batch.column("rollout_logprobs")
    assert rollout == (-0.1, None, -0.2)
    assert batch.column("loss_mask")[0] == (1, 0, 1)


def test_from_v1_refuses_to_invent_a_policy_version() -> None:
    """A v1 row with generated tokens but no recorded policy version is refused, never 0."""
    with pytest.raises(TrajectoryV2Refusal, match="never invents one"):
        from_v1(
            _v1_trajectory(policy_version_min=None, policy_version_max=None),
            model=_model(),
            harness=_harness(),
            env=_env(),
            gateway=_gateway(),
        )
