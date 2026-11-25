"""Deterministic, torch-free trace records for FoundationScale agentic trajectories.

A trace is an append-only sequence of immutable spans. Its identifiers are derived
with SHA-256, its JSON representation is canonical, and folding a trace produced by
:func:`trace_from_trajectory` reconstructs the original trajectory without repairing
or reinterpreting model output. Malformed traces are refused with the offending
field, value, and legal alternatives named wherever the trace supplies them.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, Protocol

from .contracts import SegmentKind, Termination, ToolCall, Trajectory, Turn

__all__ = (
    "LiveTraceWriter",
    "LoopObserver",
    "last_open_operation",
    "read_live",
    "trace_id_for_session",
    "TRACE_SCHEMA_VERSION",
    "ForkPoint",
    "Span",
    "SpanKind",
    "TraceRefusal",
    "TraceWriter",
    "fold",
    "fork_prefix",
    "read_trace",
    "trace_from_trajectory",
)

TRACE_SCHEMA_VERSION = "fs.trace/v1"


class SpanKind(str, Enum):
    """The five records that make up a trajectory trace."""

    SESSION = "session"
    TURN = "turn"
    TOOL_CALL = "tool_call"
    VERIFIER_RUN = "verifier_run"
    REWARD = "reward"


class TraceRefusal(ValueError):
    """Raised when a span, trace, trace file, or fork request violates its contract."""


def _is_int(value: object) -> bool:
    return type(value) is int


def _described(value: object) -> str:
    return f"{value!r} ({type(value).__name__})"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _non_empty_text(value: object, *, field_name: str, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TraceRefusal(
            f"{where}: field {field_name!r} is {_described(value)}: "
            f"the legal values are non-empty strings"
        )
    return value


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(
        f"JSON constant {value!r} is not legal in a trace: NaN and infinities "
        f"are not measurements this format carries"
    )


@dataclass(frozen=True)
class Span:
    """One immutable trace record."""

    schema_version: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    branch_id: str
    seq: int
    kind: SpanKind
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        where = f"span span_id={self.span_id!r}"
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise TraceRefusal(
                f"{where}: field 'schema_version' is {_described(self.schema_version)}: "
                f"the legal set is {{{TRACE_SCHEMA_VERSION!r}}}"
            )
        _non_empty_text(self.trace_id, field_name="trace_id", where=where)
        _non_empty_text(self.span_id, field_name="span_id", where=where)
        _non_empty_text(self.branch_id, field_name="branch_id", where=where)
        if self.parent_span_id is not None:
            _non_empty_text(
                self.parent_span_id,
                field_name="parent_span_id",
                where=where,
            )
        if not _is_int(self.seq) or self.seq < 0:
            raise TraceRefusal(
                f"{where}: field 'seq' is {_described(self.seq)}: the legal values "
                f"are real ints >= 0, and a bool is not an int here"
            )
        kind: object = self.kind  # read through object: callers build spans from JSON
        if not isinstance(kind, SpanKind):
            legal = tuple(member.value for member in SpanKind)
            raise TraceRefusal(
                f"{where}: field 'kind' is {_described(kind)}: the legal set is {legal!r}"
            )
        if not isinstance(self.data, Mapping):
            raise TraceRefusal(
                f"{where}: field 'data' is {_described(self.data)}: "
                f"the legal values are mappings with string keys"
            )
        copied = dict(self.data)
        for key in copied:
            if not isinstance(key, str):
                raise TraceRefusal(
                    f"{where}: field 'data' has key {_described(key)}: the legal keys are strings"
                )
        object.__setattr__(self, "data", MappingProxyType(copied))

    def to_json(self) -> str:
        """Serialize this span as canonical JSON, rejecting NaN and infinities."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "branch_id": self.branch_id,
            "seq": self.seq,
            "kind": self.kind.value,
            "data": dict(self.data),
        }
        try:
            return json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TraceRefusal(
                f"Span.to_json(): span {self.span_id!r} field 'data' is not legal "
                f"finite JSON data: {exc}"
            ) from exc

    @staticmethod
    def from_json(line: str) -> Span:
        """Parse and fully validate one JSON span record."""

        if not isinstance(line, str):
            raise TraceRefusal(
                f"Span.from_json(): field 'line' is {_described(line)}: "
                f"the legal values are JSON strings"
            )
        try:
            payload = json.loads(line, parse_constant=_reject_json_constant)
        except (TypeError, ValueError) as exc:
            raise TraceRefusal(f"Span.from_json(): malformed JSON record: {exc}") from exc
        if not isinstance(payload, dict):
            raise TraceRefusal(
                f"Span.from_json(): record is {_described(payload)}: "
                f"the legal top-level value is a JSON object"
            )
        expected = {
            "schema_version",
            "trace_id",
            "span_id",
            "parent_span_id",
            "branch_id",
            "seq",
            "kind",
            "data",
        }
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise TraceRefusal(
                f"Span.from_json(): record fields differ from the legal set "
                f"{sorted(expected)!r}: missing={missing!r}, unknown={unknown!r}"
            )
        raw_kind = payload["kind"]
        try:
            kind = SpanKind(raw_kind)
        except (TypeError, ValueError) as exc:
            legal = tuple(member.value for member in SpanKind)
            raise TraceRefusal(
                f"Span.from_json(): field 'kind' is {_described(raw_kind)}: "
                f"the legal set is {legal!r}"
            ) from exc
        try:
            return Span(
                schema_version=payload["schema_version"],
                trace_id=payload["trace_id"],
                span_id=payload["span_id"],
                parent_span_id=payload["parent_span_id"],
                branch_id=payload["branch_id"],
                seq=payload["seq"],
                kind=kind,
                data=payload["data"],
            )
        except TraceRefusal as exc:
            raise TraceRefusal(f"Span.from_json(): {exc}") from exc


def _make_span(
    *,
    trace_id: str,
    branch_id: str,
    seq: int,
    parent_span_id: str | None,
    kind: SpanKind,
    logical_key: str,
    data: Mapping[str, Any],
) -> Span:
    span_id = "sp_" + _digest(trace_id + "\0" + branch_id + "\0" + logical_key)
    return Span(
        schema_version=TRACE_SCHEMA_VERSION,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        branch_id=branch_id,
        seq=seq,
        kind=kind,
        data=data,
    )


def trace_from_trajectory(
    traj: Trajectory,
    *,
    run_id: str,
    rollout_step: int,
    invocation: int = 0,
    verifier_metadata: Mapping[str, Any] | None = None,
    reward_note: str | None = None,
) -> tuple[Span, ...]:
    """Encode one trajectory in deterministic session, turn, and outcome spans."""

    if not isinstance(traj, Trajectory):
        raise TraceRefusal(
            f"trace_from_trajectory(): field 'traj' is {_described(traj)}: "
            f"the legal values are Trajectory instances"
        )
    _non_empty_text(
        run_id,
        field_name="run_id",
        where="trace_from_trajectory()",
    )
    if not _is_int(rollout_step) or rollout_step < 0:
        raise TraceRefusal(
            f"trace_from_trajectory(): field 'rollout_step' is "
            f"{_described(rollout_step)}: the legal values are real ints >= 0"
        )
    if not _is_int(invocation) or invocation < 0:
        raise TraceRefusal(
            f"trace_from_trajectory(): field 'invocation' is {_described(invocation)}: "
            f"the legal values are real ints >= 0"
        )
    if verifier_metadata is not None and not isinstance(verifier_metadata, Mapping):
        raise TraceRefusal(
            f"trace_from_trajectory(): field 'verifier_metadata' is "
            f"{_described(verifier_metadata)}: the legal values are a mapping or None"
        )
    if reward_note is not None and not isinstance(reward_note, str):
        raise TraceRefusal(
            f"trace_from_trajectory(): field 'reward_note' is {_described(reward_note)}: "
            f"the legal values are a string or None"
        )

    trace_id = trace_id_for_session(
        run_id=run_id, rollout_step=rollout_step, session_id=traj.session_id, invocation=invocation
    )
    branch_id = "br_" + _digest(trace_id + "\0main")
    session_span = _make_span(
        trace_id=trace_id,
        branch_id=branch_id,
        seq=0,
        parent_span_id=None,
        kind=SpanKind.SESSION,
        logical_key="session",
        data={
            "uid": traj.uid,
            "session_id": traj.session_id,
            "harness": traj.harness,
            "termination": traj.termination.value,
            "reward": traj.reward,
            "abstention_reason": traj.abstention_reason,
            "policy_version_min": traj.policy_version_min,
            "policy_version_max": traj.policy_version_max,
            "metadata": dict(traj.metadata),
        },
    )
    spans: list[Span] = [session_span]

    for phase, turns in (("prompt", traj.prompt_turns), ("response", traj.turns)):
        for turn in turns:
            turn_span = _make_span(
                trace_id=trace_id,
                branch_id=branch_id,
                seq=len(spans),
                parent_span_id=session_span.span_id,
                kind=SpanKind.TURN,
                logical_key=f"turn:{turn.index}",
                data={
                    "phase": phase,
                    "index": turn.index,
                    "kind": turn.kind.value,
                    "token_ids": list(turn.token_ids),
                    "generated": turn.generated,
                    "logprobs": (None if turn.logprobs is None else list(turn.logprobs)),
                    "repetition_hit": turn.repetition_hit,
                    # ARL-19: written only when False, so every uncredited-free trace (all of
                    # them before forks existed) keeps byte-identical spans and span ids.
                    **({} if turn.credited else {"credited": False}),
                },
            )
            spans.append(turn_span)
            for call_index, call in enumerate(turn.tool_calls):
                spans.append(
                    _make_span(
                        trace_id=trace_id,
                        branch_id=branch_id,
                        seq=len(spans),
                        parent_span_id=turn_span.span_id,
                        kind=SpanKind.TOOL_CALL,
                        logical_key=f"tool_call:{turn.index}:{call_index}",
                        data={
                            "k": call_index,
                            "name": call.name,
                            "arguments": call.arguments,
                            "call_id": call.call_id,
                            "parse_error": call.parse_error,
                        },
                    )
                )

    verifier_span_id: str | None = None
    if verifier_metadata is not None:
        verifier_span = _make_span(
            trace_id=trace_id,
            branch_id=branch_id,
            seq=len(spans),
            parent_span_id=session_span.span_id,
            kind=SpanKind.VERIFIER_RUN,
            logical_key="verifier",
            data={
                "checked": dict(verifier_metadata),
                "target_span_id": session_span.span_id,
            },
        )
        spans.append(verifier_span)
        verifier_span_id = verifier_span.span_id

    spans.append(
        _make_span(
            trace_id=trace_id,
            branch_id=branch_id,
            seq=len(spans),
            parent_span_id=session_span.span_id,
            kind=SpanKind.REWARD,
            logical_key="reward",
            data={
                "value": traj.reward,
                "abstention_reason": traj.abstention_reason,
                "note": reward_note,
                "target_span_id": session_span.span_id,
                "verifier_span_id": verifier_span_id,
            },
        )
    )
    return tuple(spans)


def _validated_spans(
    spans: Iterable[Span],
    *,
    operation: str,
) -> tuple[Span, ...]:
    try:
        ordered = tuple(spans)
    except TypeError as exc:
        raise TraceRefusal(
            f"{operation}: field 'spans' is {_described(spans)}: "
            f"the legal values are iterables of Span instances"
        ) from exc
    if not ordered:
        raise TraceRefusal(f"{operation}: field 'spans' is empty: a trace is non-empty")

    first = ordered[0]
    if not isinstance(first, Span):
        raise TraceRefusal(
            f"{operation}: span at position 0 is {_described(first)}: "
            f"the legal values are Span instances"
        )
    trace_id = first.trace_id
    branch_id = first.branch_id
    seen_ids: set[str] = set()
    session_count = 0

    for position, span in enumerate(ordered):
        if not isinstance(span, Span):
            raise TraceRefusal(
                f"{operation}: span at position {position} is {_described(span)}: "
                f"the legal values are Span instances"
            )
        if span.seq != position:
            raise TraceRefusal(
                f"{operation}: field 'seq' at position {position} is {span.seq}: "
                f"the legal value is {position}; seq must be contiguous from 0"
            )
        if position == 0 and span.kind is not SpanKind.SESSION:
            raise TraceRefusal(
                f"{operation}: field 'kind' at seq 0 is {span.kind.value!r}: "
                f"the legal value is {SpanKind.SESSION.value!r}"
            )
        if span.kind is SpanKind.SESSION:
            session_count += 1
        if span.trace_id != trace_id:
            raise TraceRefusal(
                f"{operation}: field 'trace_id' at seq {span.seq} is {span.trace_id!r}: "
                f"the only legal value in this trace is {trace_id!r}"
            )
        if span.branch_id != branch_id:
            raise TraceRefusal(
                f"{operation}: field 'branch_id' at seq {span.seq} is {span.branch_id!r}: "
                f"the only legal value in this trace is {branch_id!r}"
            )
        if span.span_id in seen_ids:
            raise TraceRefusal(
                f"{operation}: field 'span_id' at seq {span.seq} is {span.span_id!r}: "
                f"that id already occurs earlier; span ids must be unique"
            )
        if position == 0:
            if span.parent_span_id is not None:
                raise TraceRefusal(
                    f"{operation}: field 'parent_span_id' on the SESSION is "
                    f"{span.parent_span_id!r}: the only legal value is None"
                )
        elif span.parent_span_id is None or span.parent_span_id not in seen_ids:
            raise TraceRefusal(
                f"{operation}: field 'parent_span_id' at seq {span.seq} is "
                f"{span.parent_span_id!r}: the legal values are ids of earlier spans"
            )
        seen_ids.add(span.span_id)

    if session_count != 1:
        raise TraceRefusal(
            f"{operation}: trace carries {session_count} SESSION spans: "
            f"the legal count is exactly 1, first in the trace"
        )
    return ordered


def _exact_data(
    span: Span,
    expected: tuple[str, ...],
    *,
    operation: str,
    optional: tuple[str, ...] = (),
) -> None:
    wanted = set(expected)
    actual = set(span.data) - set(optional)
    if actual != wanted:
        raise TraceRefusal(
            f"{operation}: span {span.span_id!r} kind {span.kind.value!r} field "
            f"'data' has missing keys {sorted(wanted - actual)!r} and unknown "
            f"keys {sorted(actual - wanted)!r}: the legal key set is {sorted(wanted)!r}"
        )


def _check_target(
    span: Span,
    *,
    field_name: str,
    spans_by_id: Mapping[str, Span],
) -> None:
    target = span.data[field_name]
    if not isinstance(target, str) or not target:
        raise TraceRefusal(
            f"fold(): span {span.span_id!r} field {field_name!r} is "
            f"{_described(target)}: the legal values are non-existing-free, "
            f"non-empty span id strings"
        )
    if target not in spans_by_id:
        raise TraceRefusal(
            f"fold(): span {span.span_id!r} field {field_name!r} is {target!r}: "
            f"no span with that id exists; the legal values are ids in this trace"
        )


def fold(spans: Iterable[Span]) -> Trajectory:
    """Validate one trace and reconstruct its original :class:`Trajectory`."""

    ordered = _validated_spans(spans, operation="fold()")
    session = ordered[0]
    _exact_data(
        session,
        (
            "uid",
            "session_id",
            "harness",
            "termination",
            "reward",
            "abstention_reason",
            "policy_version_min",
            "policy_version_max",
            "metadata",
        ),
        operation="fold()",
    )

    turn_order: list[str] = []
    turn_spans: dict[str, Span] = {}
    turn_phases: dict[str, str] = {}
    calls_by_turn: dict[str, list[ToolCall]] = {}
    verifier_spans: list[Span] = []
    reward_spans: list[Span] = []

    for span in ordered[1:]:
        if span.kind is SpanKind.TURN:
            if span.parent_span_id != session.span_id:
                raise TraceRefusal(
                    f"fold(): TURN span {span.span_id!r} field 'parent_span_id' is "
                    f"{span.parent_span_id!r}: the only legal value is the SESSION "
                    f"span id {session.span_id!r}"
                )
            _exact_data(
                span,
                (
                    "phase",
                    "index",
                    "kind",
                    "token_ids",
                    "generated",
                    "logprobs",
                    "repetition_hit",
                ),
                operation="fold()",
                optional=("credited",),
            )
            phase = span.data["phase"]
            if phase not in ("prompt", "response"):
                raise TraceRefusal(
                    f"fold(): TURN span {span.span_id!r} field 'phase' is "
                    f"{_described(phase)}: the legal set is {{'prompt', 'response'}}"
                )
            turn_order.append(span.span_id)
            turn_spans[span.span_id] = span
            turn_phases[span.span_id] = phase
            calls_by_turn[span.span_id] = []
        elif span.kind is SpanKind.TOOL_CALL:
            _exact_data(
                span,
                ("k", "name", "arguments", "call_id", "parse_error"),
                operation="fold()",
            )
            parent_id = span.parent_span_id
            if parent_id not in turn_spans:
                raise TraceRefusal(
                    f"fold(): TOOL_CALL span {span.span_id!r} field 'parent_span_id' "
                    f"is {parent_id!r}: the legal values are earlier TURN span ids"
                )
            calls = calls_by_turn[parent_id]
            call_index = span.data["k"]
            if not _is_int(call_index) or call_index != len(calls):
                raise TraceRefusal(
                    f"fold(): TOOL_CALL span {span.span_id!r} field 'k' is "
                    f"{_described(call_index)}: the legal value is {len(calls)}, "
                    f"the next zero-based call index on its parent TURN"
                )
            try:
                call = ToolCall(
                    name=span.data["name"],
                    arguments=span.data["arguments"],
                    call_id=span.data["call_id"],
                    parse_error=span.data["parse_error"],
                )
            except (TypeError, ValueError) as exc:
                raise TraceRefusal(
                    f"fold(): TOOL_CALL span {span.span_id!r} field 'data' does "
                    f"not reconstruct a legal ToolCall: {exc}"
                ) from exc
            calls.append(call)
        elif span.kind is SpanKind.VERIFIER_RUN:
            if span.parent_span_id != session.span_id:
                raise TraceRefusal(
                    f"fold(): VERIFIER_RUN span {span.span_id!r} field "
                    f"'parent_span_id' is {span.parent_span_id!r}: the only legal "
                    f"value is the SESSION span id {session.span_id!r}"
                )
            _exact_data(
                span,
                ("checked", "target_span_id"),
                operation="fold()",
            )
            if not isinstance(span.data["checked"], Mapping):
                raise TraceRefusal(
                    f"fold(): VERIFIER_RUN span {span.span_id!r} field 'checked' is "
                    f"{_described(span.data['checked'])}: the legal values are mappings"
                )
            verifier_spans.append(span)
        elif span.kind is SpanKind.REWARD:
            if span.parent_span_id != session.span_id:
                raise TraceRefusal(
                    f"fold(): REWARD span {span.span_id!r} field 'parent_span_id' "
                    f"is {span.parent_span_id!r}: the only legal value is the "
                    f"SESSION span id {session.span_id!r}"
                )
            _exact_data(
                span,
                (
                    "value",
                    "abstention_reason",
                    "note",
                    "target_span_id",
                    "verifier_span_id",
                ),
                operation="fold()",
            )
            reward_spans.append(span)

    spans_by_id = {span.span_id: span for span in ordered}
    for span in verifier_spans + reward_spans:
        _check_target(span, field_name="target_span_id", spans_by_id=spans_by_id)
    for span in reward_spans:
        verifier_id = span.data["verifier_span_id"]
        if verifier_id is None:
            continue
        if not isinstance(verifier_id, str) or verifier_id not in spans_by_id:
            raise TraceRefusal(
                f"fold(): REWARD span {span.span_id!r} field 'verifier_span_id' is "
                f"{_described(verifier_id)}: the legal values are None or the id "
                f"of a VERIFIER_RUN span in this trace"
            )
        if spans_by_id[verifier_id].kind is not SpanKind.VERIFIER_RUN:
            raise TraceRefusal(
                f"fold(): REWARD span {span.span_id!r} field 'verifier_span_id' "
                f"names {verifier_id!r}, whose kind is "
                f"{spans_by_id[verifier_id].kind.value!r}: the legal kind is "
                f"{SpanKind.VERIFIER_RUN.value!r}"
            )

    prompt_turns: list[Turn] = []
    response_turns: list[Turn] = []
    for span_id in turn_order:
        span = turn_spans[span_id]
        raw_kind = span.data["kind"]
        try:
            segment_kind = SegmentKind(raw_kind)
        except (TypeError, ValueError) as exc:
            legal = tuple(member.value for member in SegmentKind)
            raise TraceRefusal(
                f"fold(): TURN span {span_id!r} field 'kind' is "
                f"{_described(raw_kind)}: the legal set is {legal!r}"
            ) from exc
        token_ids = span.data["token_ids"]
        if not isinstance(token_ids, list):
            raise TraceRefusal(
                f"fold(): TURN span {span_id!r} field 'token_ids' is "
                f"{_described(token_ids)}: the legal JSON representation is a list"
            )
        raw_logprobs = span.data["logprobs"]
        if raw_logprobs is not None and not isinstance(raw_logprobs, list):
            raise TraceRefusal(
                f"fold(): TURN span {span_id!r} field 'logprobs' is "
                f"{_described(raw_logprobs)}: the legal values are a list or None"
            )
        logprobs = None if raw_logprobs is None else tuple(raw_logprobs)
        try:
            turn = Turn(
                index=span.data["index"],
                kind=segment_kind,
                token_ids=tuple(token_ids),
                generated=span.data["generated"],
                logprobs=logprobs,
                tool_calls=tuple(calls_by_turn[span_id]),
                repetition_hit=span.data["repetition_hit"],
                credited=span.data.get("credited", True),
            )
        except (TypeError, ValueError) as exc:
            raise TraceRefusal(
                f"fold(): TURN span {span_id!r} field 'data' does not "
                f"reconstruct a legal Turn: {exc}"
            ) from exc
        if turn_phases[span_id] == "prompt":
            prompt_turns.append(turn)
        else:
            response_turns.append(turn)

    raw_termination = session.data["termination"]
    try:
        termination = Termination(raw_termination)
    except (TypeError, ValueError) as exc:
        legal = tuple(member.value for member in Termination)
        raise TraceRefusal(
            f"fold(): SESSION field 'termination' is {_described(raw_termination)}: "
            f"the legal set is {legal!r}"
        ) from exc
    metadata = session.data["metadata"]
    if not isinstance(metadata, Mapping):
        raise TraceRefusal(
            f"fold(): SESSION field 'metadata' is {_described(metadata)}: "
            f"the legal values are mappings"
        )
    try:
        return Trajectory(
            uid=session.data["uid"],
            session_id=session.data["session_id"],
            harness=session.data["harness"],
            prompt_turns=tuple(prompt_turns),
            turns=tuple(response_turns),
            reward=session.data["reward"],
            abstention_reason=session.data["abstention_reason"],
            termination=termination,
            policy_version_min=session.data["policy_version_min"],
            policy_version_max=session.data["policy_version_max"],
            metadata=dict(metadata),
        )
    except (TypeError, ValueError) as exc:
        raise TraceRefusal(f"fold(): spans do not reconstruct a legal Trajectory: {exc}") from exc


@dataclass(frozen=True)
class ForkPoint:
    """The identity and token prefix from which a later response turn may fork."""

    parent_trace_id: str
    parent_branch_id: str
    branch_id: str
    prompt_token_ids: tuple[int, ...]
    turns_kept: int


def fork_prefix(spans: Iterable[Span], *, turn_index: int) -> ForkPoint:
    """Describe, without executing, a fork immediately before a response turn."""

    ordered = _validated_spans(spans, operation="fork_prefix()")
    response_indices: list[int] = []
    for span in ordered:
        if span.kind is not SpanKind.TURN:
            continue
        _exact_data(
            span,
            (
                "phase",
                "index",
                "kind",
                "token_ids",
                "generated",
                "logprobs",
                "repetition_hit",
            ),
            operation="fork_prefix()",
            optional=("credited",),
        )
        phase = span.data["phase"]
        if phase not in ("prompt", "response"):
            raise TraceRefusal(
                f"fork_prefix(): TURN span {span.span_id!r} field 'phase' is "
                f"{_described(phase)}: the legal set is {{'prompt', 'response'}}"
            )
        if phase == "response":
            index = span.data["index"]
            if not _is_int(index) or index < 0:
                raise TraceRefusal(
                    f"fork_prefix(): TURN span {span.span_id!r} field 'index' is "
                    f"{_described(index)}: the legal values are real ints >= 0"
                )
            response_indices.append(index)

    legal_indices = tuple(sorted(set(response_indices)))
    if not _is_int(turn_index) or turn_index not in response_indices:
        raise TraceRefusal(
            f"fork_prefix(): field 'turn_index' is {_described(turn_index)}: "
            f"the legal set of response turn indices is {legal_indices!r}"
        )

    tokens: list[int] = []
    turns_kept = 0
    for span in ordered:
        if span.kind is not SpanKind.TURN:
            continue
        phase = span.data["phase"]
        index = span.data["index"]
        included = phase == "prompt" or (phase == "response" and index < turn_index)
        if not included:
            continue
        token_ids = span.data["token_ids"]
        if not isinstance(token_ids, list):
            raise TraceRefusal(
                f"fork_prefix(): TURN span {span.span_id!r} field 'token_ids' is "
                f"{_described(token_ids)}: the legal JSON representation is a list"
            )
        for position, token_id in enumerate(token_ids):
            if not _is_int(token_id) or token_id < 0:
                raise TraceRefusal(
                    f"fork_prefix(): TURN span {span.span_id!r} field 'token_ids' "
                    f"entry {position} is {_described(token_id)}: the legal values "
                    f"are real ints >= 0"
                )
        tokens.extend(token_ids)
        if phase == "response":
            turns_kept += 1

    trace_id = ordered[0].trace_id
    branch_id = "br_" + _digest(f"{trace_id}\0fork\0{turn_index}")
    return ForkPoint(
        parent_trace_id=trace_id,
        parent_branch_id=ordered[0].branch_id,
        branch_id=branch_id,
        prompt_token_ids=tuple(tokens),
        turns_kept=turns_kept,
    )


class TraceWriter:
    """Append canonical span records, fsyncing once at the end of each episode."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        max_record_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if not _is_int(max_record_bytes) or max_record_bytes <= 0:
            raise TraceRefusal(
                f"TraceWriter(): field 'max_record_bytes' is "
                f"{_described(max_record_bytes)}: the legal values are real ints > 0"
            )
        self.path = os.fspath(path)
        self.max_record_bytes = max_record_bytes

    def write_episode(self, spans: Iterable[Span]) -> None:
        """Append one episode, refusing oversize records before opening the file."""

        records: list[str] = []
        try:
            iterator = iter(spans)
        except TypeError as exc:
            raise TraceRefusal(
                f"TraceWriter.write_episode(): field 'spans' is {_described(spans)}: "
                f"the legal values are iterables of Span instances"
            ) from exc
        for position, span in enumerate(iterator):
            if not isinstance(span, Span):
                raise TraceRefusal(
                    f"TraceWriter.write_episode(): record {position} is "
                    f"{_described(span)}: the legal values are Span instances"
                )
            record = span.to_json() + "\n"
            size = len(record.encode("utf-8"))
            if size > self.max_record_bytes:
                raise TraceRefusal(
                    f"TraceWriter.write_episode(): record {position} is {size} bytes "
                    f"but field 'max_record_bytes' is {self.max_record_bytes} bytes: "
                    f"a record over that limit is refused"
                )
            records.append(record)

        with Path(self.path).open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())


def read_trace(
    path: str | os.PathLike[str],
    *,
    allow_partial_tail: bool = False,
) -> dict[str, tuple[Span, ...]]:
    """Read spans grouped by trace id, preserving file order within each group."""

    if type(allow_partial_tail) is not bool:
        raise TraceRefusal(
            f"read_trace(): field 'allow_partial_tail' is "
            f"{_described(allow_partial_tail)}: the legal values are True and False"
        )
    with Path(path).open("rb") as handle:
        contents = handle.read()
    raw_lines = contents.splitlines(keepends=True)
    grouped: dict[str, list[Span]] = {}

    for offset, raw_line in enumerate(raw_lines):
        line_number = offset + 1
        terminated = raw_line.endswith((b"\n", b"\r"))
        is_final = offset == len(raw_lines) - 1
        if is_final and not terminated:
            if allow_partial_tail:
                break
            raise TraceRefusal(
                f"read_trace(): line {line_number} is an unterminated final line: "
                f"pass allow_partial_tail=True to ignore only that partial tail"
            )
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TraceRefusal(
                f"read_trace(): line {line_number} is not valid UTF-8: {exc}"
            ) from exc
        try:
            span = Span.from_json(line)
        except TraceRefusal as exc:
            raise TraceRefusal(f"read_trace(): malformed line {line_number}: {exc}") from exc
        grouped.setdefault(span.trace_id, []).append(span)

    return {trace_id: tuple(group) for trace_id, group in grouped.items()}


def trace_id_for_session(
    *,
    run_id: str,
    rollout_step: int,
    session_id: str,
    invocation: int = 0,
) -> str:
    """Derive the trace id used by trace_from_trajectory for one session."""
    identity = f"{run_id}\0{rollout_step}\0{session_id}\0{invocation}"
    return "tr_" + _digest(identity)


class LoopObserver(Protocol):
    """Synchronous notifications for operations whose start must survive a crash."""

    def generation_started(self, turn_index: int, policy_version: int | None) -> None: ...

    def generation_finished(
        self, turn_index: int, finish_reason: str | None, n_tokens: int
    ) -> None: ...

    def tool_started(self, turn_index: int, call_index: int, name: str | None) -> None: ...

    def tool_finished(
        self,
        turn_index: int,
        call_index: int,
        exit_code: int | None,
        timed_out: bool,
    ) -> None: ...


class LiveTraceWriter:
    """Append and durably flush one JSON record for every loop operation."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        trace_id: str,
        session_id: str,
    ) -> None:
        self.path = os.fspath(path)
        self.trace_id = trace_id
        self.session_id = session_id
        self._seq = 0

    def _append(self, event: str, turn_index: int, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "schema_version": "fs.trace.live/v1",
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "seq": self._seq,
            "event": event,
            "turn_index": turn_index,
            **fields,
            "at_ts": time.time(),
        }
        record = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with Path(self.path).open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._seq += 1

    def generation_started(self, turn_index: int, policy_version: int | None) -> None:
        self._append("generation_started", turn_index, policy_version=policy_version)

    def generation_finished(
        self, turn_index: int, finish_reason: str | None, n_tokens: int
    ) -> None:
        self._append(
            "generation_finished",
            turn_index,
            finish_reason=finish_reason,
            n_tokens=n_tokens,
        )

    def tool_started(self, turn_index: int, call_index: int, name: str | None) -> None:
        self._append("tool_started", turn_index, call_index=call_index, name=name)

    def tool_finished(
        self,
        turn_index: int,
        call_index: int,
        exit_code: int | None,
        timed_out: bool,
    ) -> None:
        self._append(
            "tool_finished",
            turn_index,
            call_index=call_index,
            exit_code=exit_code,
            timed_out=timed_out,
        )


def read_live(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read live events, ignoring only a possibly torn, unterminated final line."""

    with Path(path).open("rb") as handle:
        contents = handle.read()
    raw_lines = contents.splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    for offset, raw_line in enumerate(raw_lines):
        line_number = offset + 1
        terminated = raw_line.endswith((b"\n", b"\r"))
        if offset == len(raw_lines) - 1 and not terminated:
            break
        try:
            line = raw_line.decode("utf-8")
            payload = json.loads(line, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, TypeError, ValueError) as exc:
            raise TraceRefusal(f"read_live(): malformed line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise TraceRefusal(
                f"read_live(): line {line_number} is {_described(payload)}: "
                "the legal top-level value is a JSON object"
            )
        events.append(payload)
    return events


def last_open_operation(events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the latest started operation that has no matching finished record."""

    open_operations: dict[tuple[object, ...], tuple[int, dict[str, Any]]] = {}
    for position, event in enumerate(events):
        event_name = event.get("event")
        if not isinstance(event_name, str):
            continue
        operation, separator, phase = event_name.rpartition("_")
        if not separator or phase not in ("started", "finished"):
            continue
        if operation not in ("generation", "tool"):
            continue
        call_index = event.get("call_index") if operation == "tool" else None
        key = (operation, event.get("turn_index"), call_index)
        if phase == "started":
            open_operations[key] = (position, event)
        else:
            open_operations.pop(key, None)
    if not open_operations:
        return None
    return max(open_operations.values(), key=lambda item: item[0])[1]
