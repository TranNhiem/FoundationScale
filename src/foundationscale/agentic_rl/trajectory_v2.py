"""``foundationscale.agentic_rl.trajectory_v2`` — torch-free trajectory schema (v2).

PART A: schema constants, enums, and frozen dataclasses with complete per-object
validation.  Every dataclass validates its own invariants in ``__post_init__`` and
raises :class:`TrajectoryV2Refusal` (a ``ValueError`` subclass) on violation.

Doctrine (FS):
  * Refusal messages name BOTH counts ("k of n ...").
  * ``bool`` is never accepted where ``int`` is required ("1 is not True").
  * Unmeasured is ``None``, never ``0.0``.
  * Mapping fields are frozen with ``types.MappingProxyType``.
  * Sequence fields are coerced to ``tuple`` in ``__post_init__``.

Stdlib only: dataclasses, enum, math, types.MappingProxyType, typing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, NoReturn

SCHEMA_VERSION = "fs.trajectory/v2"

__all__ = [
    "SCHEMA_VERSION",
    "TrajectoryV2Refusal",
    "EpisodeStatus",
    "Termination",
    "ParseVerdict",
    "RewardScope",
    "FidelityStatus",
    "ApiFormat",
    "EnvKind",
    "ToolCallV2",
    "Generation",
    "Segment",
    "DiscardedGeneration",
    "RewardEvent",
    "ModelMeta",
    "HarnessMeta",
    "EnvMeta",
    "SandboxMeta",
    "GatewayMeta",
    "Timings",
    "Fidelity",
    "Episode",
]


class TrajectoryV2Refusal(ValueError):
    """Raised when a trajectory-v2 object violates a binding API invariant."""


def _is_int(x: Any) -> bool:
    """True iff ``x`` is a real ``int`` (``bool`` is never an ``int`` here)."""
    return type(x) is int


def _refuse(msg: str) -> NoReturn:
    raise TrajectoryV2Refusal(msg)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite(x: Any) -> bool:
    return _is_number(x) and math.isfinite(float(x))


def _is_nonneg_finite(x: Any) -> bool:
    return _is_finite(x) and float(x) >= 0.0


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


def _is_opt_nonempty_str(x: Any) -> bool:
    return x is None or _is_nonempty_str(x)


def _freeze_mapping(m: Any, owner: str, field_name: str) -> Mapping[str, Any]:
    if not isinstance(m, Mapping):
        _refuse(f"{owner}.{field_name}: expected a Mapping, got {type(m).__name__}")
    return MappingProxyType(dict(m))


def _as_tuple(seq: Any, owner: str, field_name: str) -> tuple[Any, ...]:
    if isinstance(seq, tuple):
        return seq
    if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes, bytearray)):
        return tuple(seq)
    _refuse(f"{owner}.{field_name}: expected a sequence, got {type(seq).__name__}")


class EpisodeStatus(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"
    SANDBOX_ERROR = "sandbox_error"
    HARNESS_ERROR = "harness_error"
    CONTEXT_OVERFLOW = "context_overflow"
    FIDELITY_MISMATCH = "fidelity_mismatch"
    VERIFIER_ERROR = "verifier_error"


class Termination(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TURN_LIMIT = "turn_limit"
    ABORTED = "aborted"
    ERROR = "error"


class ParseVerdict(str, Enum):
    HARNESS = "harness"
    ENGINE_PARSER = "engine_parser"
    OBSERVED_ACCEPTED = "observed_accepted"
    UNOBSERVED = "unobserved"


class RewardScope(str, Enum):
    EPISODE = "episode"
    SEGMENT = "segment"
    GENERATION = "generation"
    TOKEN_SPAN = "token_span"


class FidelityStatus(str, Enum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    UNVERIFIED = "unverified"


class ApiFormat(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI = "gemini"
    WHITE_BOX = "white_box"


class EnvKind(str, Enum):
    HARBOR = "harbor"
    MCP = "mcp"
    NEMO_GYM = "nemo_gym"
    FS_LOCAL = "fs_local"


@dataclass(frozen=True)
class ToolCallV2:
    call_id: str | None
    name: str | None
    arguments_raw: str | None
    parse_verdict: ParseVerdict
    parse_error: str | None = None
    call_id_origin: str = "model"

    def __post_init__(self) -> None:
        if not _is_opt_nonempty_str(self.call_id):
            _refuse("ToolCallV2.call_id: expected None or non-empty str")
        if not _is_opt_nonempty_str(self.name):
            _refuse("ToolCallV2.name: expected None or non-empty str")
        if self.arguments_raw is not None and not isinstance(self.arguments_raw, str):
            _refuse("ToolCallV2.arguments_raw: expected None or str")
        parse_verdict: object = self.parse_verdict
        if not isinstance(parse_verdict, ParseVerdict):
            _refuse("ToolCallV2.parse_verdict: expected ParseVerdict")
        if self.parse_error is not None and not isinstance(self.parse_error, str):
            _refuse("ToolCallV2.parse_error: expected None or str")
        if self.parse_error is not None and len(self.parse_error) == 0:
            _refuse("ToolCallV2.parse_error: expected None or non-empty str, got empty str")
        call_id_origin: object = self.call_id_origin
        if call_id_origin not in ("model", "synthesized"):
            _refuse(
                "ToolCallV2.call_id_origin: expected 'model' or 'synthesized', "
                f"got {self.call_id_origin!r}"
            )
        if self.call_id_origin == "synthesized" and not _is_nonempty_str(self.call_id):
            _refuse(
                "ToolCallV2: call_id_origin == 'synthesized' requires call_id non-empty, "
                f"got {self.call_id!r}"
            )


@dataclass(frozen=True)
class Generation:
    generation_id: str
    prompt_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    policy_version: int
    finish_reason: str | None = None
    sampling_requested: Mapping[str, Any] = field(default_factory=dict)
    sampling_overrides: Mapping[str, Any] = field(default_factory=dict)
    tool_calls: tuple[ToolCallV2, ...] = ()
    routed_experts: tuple[tuple[tuple[int, ...], ...], ...] | None = None

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.generation_id):
            _refuse("Generation.generation_id: expected non-empty str")

        prompt_ids = _as_tuple(self.prompt_ids, "Generation", "prompt_ids")
        output_ids = _as_tuple(self.output_ids, "Generation", "output_ids")
        logprobs = _as_tuple(self.logprobs, "Generation", "logprobs")
        tool_calls = _as_tuple(self.tool_calls, "Generation", "tool_calls")
        object.__setattr__(self, "prompt_ids", prompt_ids)
        object.__setattr__(self, "output_ids", output_ids)
        object.__setattr__(self, "logprobs", logprobs)
        object.__setattr__(self, "tool_calls", tool_calls)

        if len(prompt_ids) == 0:
            _refuse("Generation.prompt_ids: expected non-empty tuple, got 0 of 0 tokens")
        bad_prompt = sum(1 for t in prompt_ids if not (_is_int(t) and t >= 0))
        if bad_prompt:
            _refuse(
                f"Generation.prompt_ids: {bad_prompt} of {len(prompt_ids)} tokens are not ints >= 0"
            )

        if len(output_ids) == 0:
            _refuse("Generation.output_ids: expected non-empty tuple, got 0 of 0 tokens")
        bad_output = sum(1 for t in output_ids if not (_is_int(t) and t >= 0))
        if bad_output:
            _refuse(
                f"Generation.output_ids: {bad_output} of {len(output_ids)} tokens are not ints >= 0"
            )

        if len(logprobs) != len(output_ids):
            _refuse(
                f"Generation.logprobs: {len(logprobs)} of {len(output_ids)} entries "
                "(len(logprobs) must equal len(output_ids))"
            )
        bad_lp = sum(1 for lp in logprobs if not _is_finite(lp))
        if bad_lp:
            _refuse(
                f"Generation.logprobs: {bad_lp} of {len(logprobs)} entries are not finite floats"
            )

        policy_version: object = self.policy_version
        if not _is_int(policy_version) or self.policy_version < 0:
            _refuse(f"Generation.policy_version: expected int >= 0, got {self.policy_version!r}")

        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            _refuse("Generation.finish_reason: expected None or str")

        object.__setattr__(
            self,
            "sampling_requested",
            _freeze_mapping(self.sampling_requested, "Generation", "sampling_requested"),
        )
        object.__setattr__(
            self,
            "sampling_overrides",
            _freeze_mapping(self.sampling_overrides, "Generation", "sampling_overrides"),
        )

        bad_tc = sum(1 for tc in tool_calls if not isinstance(tc, ToolCallV2))
        if bad_tc:
            _refuse(
                f"Generation.tool_calls: {bad_tc} of {len(tool_calls)} entries are not ToolCallV2"
            )

        if self.routed_experts is not None:
            rows = _as_tuple(self.routed_experts, "Generation", "routed_experts")
            expected = len(prompt_ids) + len(output_ids)
            if len(rows) != expected:
                _refuse(
                    f"Generation.routed_experts: {len(rows)} of {expected} rows "
                    "(must cover prompt_ids + output_ids)"
                )
            for i, row in enumerate(rows):
                layers = _as_tuple(row, "Generation", f"routed_experts[{i}]")
                for j, layer in enumerate(layers):
                    ids = _as_tuple(layer, "Generation", f"routed_experts[{i}][{j}]")
                    bad_ids = sum(1 for e in ids if not (_is_int(e) and e >= 0))
                    if bad_ids:
                        _refuse(
                            f"Generation.routed_experts[{i}][{j}]: {bad_ids} of {len(ids)} "
                            "expert ids are not ints >= 0"
                        )
                    layers = layers[:j] + (ids,) + layers[j + 1 :]
                rows = rows[:i] + (layers,) + rows[i + 1 :]
            object.__setattr__(self, "routed_experts", rows)


@dataclass(frozen=True)
class Segment:
    segment_index: int
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    policy_version: tuple[int, ...]
    generation_spans: tuple[tuple[int, int, str], ...]
    routed_experts: tuple[tuple[tuple[int, ...], ...], ...] | None = None

    def __post_init__(self) -> None:
        segment_index: object = self.segment_index
        if not _is_int(segment_index) or self.segment_index < 0:
            _refuse(f"Segment.segment_index: expected int >= 0, got {self.segment_index!r}")

        prompt_ids = _as_tuple(self.prompt_ids, "Segment", "prompt_ids")
        response_ids = _as_tuple(self.response_ids, "Segment", "response_ids")
        loss_mask = _as_tuple(self.loss_mask, "Segment", "loss_mask")
        logprobs = _as_tuple(self.logprobs, "Segment", "logprobs")
        policy_version = _as_tuple(self.policy_version, "Segment", "policy_version")
        generation_spans = _as_tuple(self.generation_spans, "Segment", "generation_spans")
        object.__setattr__(self, "prompt_ids", prompt_ids)
        object.__setattr__(self, "response_ids", response_ids)
        object.__setattr__(self, "loss_mask", loss_mask)
        object.__setattr__(self, "logprobs", logprobs)
        object.__setattr__(self, "policy_version", policy_version)
        object.__setattr__(self, "generation_spans", generation_spans)

        if len(prompt_ids) == 0:
            _refuse("Segment.prompt_ids: expected non-empty tuple, got 0 of 0 tokens")
        bad_prompt = sum(1 for t in prompt_ids if not (_is_int(t) and t >= 0))
        if bad_prompt:
            _refuse(
                f"Segment.prompt_ids: {bad_prompt} of {len(prompt_ids)} tokens are not ints >= 0"
            )

        if len(response_ids) == 0:
            _refuse("Segment.response_ids: expected non-empty tuple, got 0 of 0 tokens")
        bad_response = sum(1 for t in response_ids if not (_is_int(t) and t >= 0))
        if bad_response:
            _refuse(
                f"Segment.response_ids: {bad_response} of {len(response_ids)} tokens "
                "are not ints >= 0"
            )

        n = len(response_ids)
        if len(loss_mask) != n:
            _refuse(
                f"Segment.loss_mask: {len(loss_mask)} of {n} entries "
                "(len(loss_mask) must equal len(response_ids))"
            )
        bad_mask = sum(1 for m in loss_mask if not (_is_int(m) and m in (0, 1)))
        if bad_mask:
            _refuse(
                f"Segment.loss_mask: {bad_mask} of {n} entries are not 0/1 ints "
                "(bool is not int here)"
            )

        if len(logprobs) != n:
            _refuse(
                f"Segment.logprobs: {len(logprobs)} of {n} entries "
                "(len(logprobs) must equal len(response_ids))"
            )
        bad_lp = 0
        for lp, m in zip(logprobs, loss_mask, strict=True):
            if m == 1:
                if not _is_finite(lp):
                    bad_lp += 1
            else:
                if not (_is_number(lp) and math.isnan(float(lp))):
                    bad_lp += 1
        if bad_lp:
            _refuse(
                f"Segment.logprobs: {bad_lp} of {n} entries violate R4 "
                "(finite where mask==1, NaN where mask==0)"
            )

        if len(policy_version) != n:
            _refuse(
                f"Segment.policy_version: {len(policy_version)} of {n} entries "
                "(len(policy_version) must equal len(response_ids))"
            )
        bad_pv = 0
        for pv, m in zip(policy_version, loss_mask, strict=True):
            if m == 1:
                if not (_is_int(pv) and pv >= 0):
                    bad_pv += 1
            else:
                if not (_is_int(pv) and pv == -1):
                    bad_pv += 1
        if bad_pv:
            _refuse(
                f"Segment.policy_version: {bad_pv} of {n} entries violate R4 "
                "(>= 0 where mask==1, exactly -1 where mask==0)"
            )

        trainable = [i for i, m in enumerate(loss_mask) if m == 1]
        covered: set = set()
        prev_end = 0
        for si, span in enumerate(generation_spans):
            if not (isinstance(span, tuple) and len(span) == 3):
                _refuse(
                    f"Segment.generation_spans[{si}]: expected (start, end, generation_id), "
                    f"got {span!r}"
                )
            start, end, gen_id = span
            if not (_is_int(start) and _is_int(end)):
                _refuse(
                    f"Segment.generation_spans[{si}]: start/end must be ints, "
                    f"got ({start!r}, {end!r})"
                )
            if not (0 <= start < end <= n):
                _refuse(
                    f"Segment.generation_spans[{si}]: expected 0 <= start < end <= {n}, "
                    f"got ({start}, {end})"
                )
            if start < prev_end:
                _refuse(
                    f"Segment.generation_spans[{si}]: spans must be ascending and "
                    f"non-overlapping; start {start} < previous end {prev_end} "
                    f"({si} of {len(generation_spans)} spans checked)"
                )
            prev_end = end
            if not _is_nonempty_str(gen_id):
                _refuse(f"Segment.generation_spans[{si}].generation_id: expected non-empty str")
            for i in range(start, end):
                if i in covered:
                    _refuse(
                        f"Segment.generation_spans: token {i} covered twice "
                        f"({si} of {len(generation_spans)} spans checked)"
                    )
                covered.add(i)

        if covered != set(trainable):
            _refuse(
                f"Segment.generation_spans: cover exactly the mask==1 positions; "
                f"{len(covered)} of {len(trainable)} trainable positions covered"
            )

        if self.routed_experts is not None:
            rows = _as_tuple(self.routed_experts, "Segment", "routed_experts")
            expected = len(prompt_ids) + len(response_ids)
            if len(rows) != expected:
                _refuse(
                    f"Segment.routed_experts: {len(rows)} of {expected} rows "
                    "(must cover prompt_ids + response_ids)"
                )
            for i, row in enumerate(rows):
                layers = _as_tuple(row, "Segment", f"routed_experts[{i}]")
                for j, layer in enumerate(layers):
                    ids = _as_tuple(layer, "Segment", f"routed_experts[{i}][{j}]")
                    bad_ids = sum(1 for e in ids if not (_is_int(e) and e >= 0))
                    if bad_ids:
                        _refuse(
                            f"Segment.routed_experts[{i}][{j}]: {bad_ids} of {len(ids)} "
                            "expert ids are not ints >= 0"
                        )
                    layers = layers[:j] + (ids,) + layers[j + 1 :]
                rows = rows[:i] + (layers,) + rows[i + 1 :]
            object.__setattr__(self, "routed_experts", rows)

    @property
    def num_trainable_tokens(self) -> int:
        return sum(1 for m in self.loss_mask if m == 1)


@dataclass(frozen=True)
class DiscardedGeneration:
    generation: Generation
    reason: str

    def __post_init__(self) -> None:
        generation: object = self.generation
        if not isinstance(generation, Generation):
            _refuse("DiscardedGeneration.generation: expected Generation")
        if not _is_nonempty_str(self.reason):
            _refuse("DiscardedGeneration.reason: expected non-empty str")


@dataclass(frozen=True)
class RewardEvent:
    source: str
    verifier_id: str
    verifier_version: str
    value: float | None
    scope: RewardScope
    span: tuple[int, int] | None = None
    at_ts: float = 0.0
    delayed: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.source):
            _refuse("RewardEvent.source: expected non-empty str")
        if not _is_nonempty_str(self.verifier_id):
            _refuse("RewardEvent.verifier_id: expected non-empty str")
        if not _is_nonempty_str(self.verifier_version):
            _refuse("RewardEvent.verifier_version: expected non-empty str")

        if self.value is not None and (isinstance(self.value, bool) or not _is_finite(self.value)):
            _refuse(
                f"RewardEvent.value: expected finite float or None (abstention), got {self.value!r}"
            )

        scope: object = self.scope
        if not isinstance(scope, RewardScope):
            _refuse("RewardEvent.scope: expected RewardScope")

        if self.scope == RewardScope.TOKEN_SPAN:
            if self.span is None:
                _refuse("RewardEvent.span: required (non-None) when scope == TOKEN_SPAN")
        else:
            if self.span is not None:
                _refuse(
                    f"RewardEvent.span: must be None when scope != TOKEN_SPAN, got {self.span!r}"
                )

        if self.span is not None:
            span = _as_tuple(self.span, "RewardEvent", "span")
            object.__setattr__(self, "span", span)
            if len(span) != 2:
                _refuse(f"RewardEvent.span: expected (start, end), got {span!r}")
            start, end = span
            if not (_is_int(start) and _is_int(end)):
                _refuse(f"RewardEvent.span: start/end must be ints, got ({start!r}, {end!r})")
            if not (0 <= start < end):
                _refuse(f"RewardEvent.span: expected 0 <= start < end, got ({start}, {end})")

        if not _is_nonneg_finite(self.at_ts):
            _refuse(f"RewardEvent.at_ts: expected finite float >= 0, got {self.at_ts!r}")

        delayed: object = self.delayed
        if not isinstance(delayed, bool):
            _refuse("RewardEvent.delayed: expected bool")

        object.__setattr__(self, "details", _freeze_mapping(self.details, "RewardEvent", "details"))


@dataclass(frozen=True)
class ModelMeta:
    name: str
    version: int
    checkpoint_sha: str

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.name):
            _refuse("ModelMeta.name: expected non-empty str")
        version: object = self.version
        if not _is_int(version) or self.version < 0:
            _refuse(f"ModelMeta.version: expected int >= 0, got {self.version!r}")
        if not _is_nonempty_str(self.checkpoint_sha):
            _refuse("ModelMeta.checkpoint_sha: expected non-empty str")


@dataclass(frozen=True)
class HarnessMeta:
    name: str
    version: str
    commit: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.name):
            _refuse("HarnessMeta.name: expected non-empty str")
        if not _is_nonempty_str(self.version):
            _refuse("HarnessMeta.version: expected non-empty str")
        if not _is_opt_nonempty_str(self.commit):
            _refuse("HarnessMeta.commit: expected None or non-empty str")
        if not _is_opt_nonempty_str(self.license):
            _refuse("HarnessMeta.license: expected None or non-empty str")


@dataclass(frozen=True)
class EnvMeta:
    kind: EnvKind
    task_id: str
    task_version: str | None = None
    image_digest: str | None = None

    def __post_init__(self) -> None:
        kind: object = self.kind
        if not isinstance(kind, EnvKind):
            _refuse("EnvMeta.kind: expected EnvKind")
        if not _is_nonempty_str(self.task_id):
            _refuse("EnvMeta.task_id: expected non-empty str")
        if not _is_opt_nonempty_str(self.task_version):
            _refuse("EnvMeta.task_version: expected None or non-empty str")
        if not _is_opt_nonempty_str(self.image_digest):
            _refuse("EnvMeta.image_digest: expected None or non-empty str")


@dataclass(frozen=True)
class SandboxMeta:
    network_policy: str
    node: str | None = None
    cpu: float | None = None
    mem_mb: int | None = None
    started_at: float | None = None
    ended_at: float | None = None
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.network_policy):
            _refuse("SandboxMeta.network_policy: expected non-empty str")
        if not _is_opt_nonempty_str(self.node):
            _refuse("SandboxMeta.node: expected None or non-empty str")
        if self.cpu is not None and not _is_nonneg_finite(self.cpu):
            _refuse(f"SandboxMeta.cpu: expected None or finite float >= 0, got {self.cpu!r}")
        mem_mb: object = self.mem_mb
        if self.mem_mb is not None and not (_is_int(mem_mb) and self.mem_mb >= 0):
            _refuse(f"SandboxMeta.mem_mb: expected None or int >= 0, got {self.mem_mb!r}")
        if self.started_at is not None and not _is_nonneg_finite(self.started_at):
            _refuse(
                f"SandboxMeta.started_at: expected None or finite float >= 0, "
                f"got {self.started_at!r}"
            )
        if self.ended_at is not None and not _is_nonneg_finite(self.ended_at):
            _refuse(
                f"SandboxMeta.ended_at: expected None or finite float >= 0, got {self.ended_at!r}"
            )
        exit_code: object = self.exit_code
        if self.exit_code is not None and not _is_int(exit_code):
            _refuse(f"SandboxMeta.exit_code: expected None or int, got {self.exit_code!r}")


@dataclass(frozen=True)
class GatewayMeta:
    api_format: ApiFormat
    session_id: str

    def __post_init__(self) -> None:
        api_format: object = self.api_format
        if not isinstance(api_format, ApiFormat):
            _refuse("GatewayMeta.api_format: expected ApiFormat")
        if not _is_nonempty_str(self.session_id):
            _refuse("GatewayMeta.session_id: expected non-empty str")


@dataclass(frozen=True)
class Timings:
    queue_ms: float | None = None
    init_ms: float | None = None
    run_ms: float | None = None
    reward_ms: float | None = None

    def __post_init__(self) -> None:
        for name in ("queue_ms", "init_ms", "run_ms", "reward_ms"):
            v = getattr(self, name)
            if v is not None and not _is_nonneg_finite(v):
                _refuse(f"Timings.{name}: expected None or finite float >= 0, got {v!r}")


@dataclass(frozen=True)
class Fidelity:
    status: FidelityStatus
    detail: str | None = None

    def __post_init__(self) -> None:
        status: object = self.status
        if not isinstance(status, FidelityStatus):
            _refuse("Fidelity.status: expected FidelityStatus")
        if self.detail is not None and not isinstance(self.detail, str):
            _refuse("Fidelity.detail: expected None or str")


@dataclass(frozen=True)
class Episode:
    episode_id: str
    group_id: str
    task_prompt_ids: tuple[int, ...]
    status: EpisodeStatus
    termination: Termination
    segments: tuple[Segment, ...]
    reward_events: tuple[RewardEvent, ...]
    model: ModelMeta
    harness: HarnessMeta
    env: EnvMeta
    gateway: GatewayMeta
    sandbox: SandboxMeta | None = None
    timings: Timings = field(default_factory=Timings)
    fidelity: Fidelity = field(default_factory=lambda: Fidelity(FidelityStatus.VERIFIED))
    discarded: tuple[DiscardedGeneration, ...] = ()
    trainable_on_timeout: bool = False
    tool_schema_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.episode_id):
            _refuse("Episode.episode_id: expected non-empty str")
        if not _is_nonempty_str(self.group_id):
            _refuse("Episode.group_id: expected non-empty str")

        task_prompt_ids = _as_tuple(self.task_prompt_ids, "Episode", "task_prompt_ids")
        object.__setattr__(self, "task_prompt_ids", task_prompt_ids)
        if len(task_prompt_ids) == 0:
            _refuse("Episode.task_prompt_ids: expected non-empty tuple, got 0 of 0 tokens")
        bad_tp = sum(1 for t in task_prompt_ids if not (_is_int(t) and t >= 0))
        if bad_tp:
            _refuse(
                f"Episode.task_prompt_ids: {bad_tp} of {len(task_prompt_ids)} tokens "
                "are not ints >= 0"
            )

        status: object = self.status
        if not isinstance(status, EpisodeStatus):
            _refuse("Episode.status: expected EpisodeStatus")
        termination: object = self.termination
        if not isinstance(termination, Termination):
            _refuse("Episode.termination: expected Termination")

        segments = _as_tuple(self.segments, "Episode", "segments")
        reward_events = _as_tuple(self.reward_events, "Episode", "reward_events")
        discarded = _as_tuple(self.discarded, "Episode", "discarded")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "reward_events", reward_events)
        object.__setattr__(self, "discarded", discarded)

        bad_seg = sum(1 for s in segments if not isinstance(s, Segment))
        if bad_seg:
            _refuse(f"Episode.segments: {bad_seg} of {len(segments)} entries are not Segment")
        for pos, seg in enumerate(segments):
            if seg.segment_index != pos:
                _refuse(
                    f"Episode.segments: segment_index == position violated at position {pos} "
                    f"of {len(segments)} segments (segment_index == {seg.segment_index})"
                )

        bad_re = sum(1 for r in reward_events if not isinstance(r, RewardEvent))
        if bad_re:
            _refuse(
                f"Episode.reward_events: {bad_re} of {len(reward_events)} entries "
                "are not RewardEvent"
            )

        bad_disc = sum(1 for d in discarded if not isinstance(d, DiscardedGeneration))
        if bad_disc:
            _refuse(
                f"Episode.discarded: {bad_disc} of {len(discarded)} entries "
                "are not DiscardedGeneration"
            )

        model: object = self.model
        if not isinstance(model, ModelMeta):
            _refuse("Episode.model: expected ModelMeta")
        harness: object = self.harness
        if not isinstance(harness, HarnessMeta):
            _refuse("Episode.harness: expected HarnessMeta")
        env: object = self.env
        if not isinstance(env, EnvMeta):
            _refuse("Episode.env: expected EnvMeta")
        gateway: object = self.gateway
        if not isinstance(gateway, GatewayMeta):
            _refuse("Episode.gateway: expected GatewayMeta")
        sandbox: object = self.sandbox
        if self.sandbox is not None and not isinstance(sandbox, SandboxMeta):
            _refuse("Episode.sandbox: expected SandboxMeta or None")
        timings: object = self.timings
        if not isinstance(timings, Timings):
            _refuse("Episode.timings: expected Timings")
        fidelity: object = self.fidelity
        if not isinstance(fidelity, Fidelity):
            _refuse("Episode.fidelity: expected Fidelity")

        trainable_on_timeout: object = self.trainable_on_timeout
        if not isinstance(trainable_on_timeout, bool):
            _refuse("Episode.trainable_on_timeout: expected bool")

        if not _is_opt_nonempty_str(self.tool_schema_sha256):
            _refuse("Episode.tool_schema_sha256: expected None or non-empty str")

        if self.schema_version != SCHEMA_VERSION:
            _refuse(
                f"Episode.schema_version: expected {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )


# ============================================================================
# PART B — cross-object validation, assembly, reward reduction, (de)serialisation
# ============================================================================


def build_segments(
    generations: Sequence[Generation],
    *,
    eot_token_ids: frozenset = frozenset(),
) -> tuple[tuple[Segment, ...], int]:
    """D2/R6: assemble :class:`Segment` objects from a generation stream.

    Returns ``(segments, fragmentation_count)``.  Generation 0 opens segment 0
    with ``prompt = gen0.prompt_ids`` and ``response = gen0.output_ids`` (all
    trainable).  For every later generation the current segment's full token
    stream ``prev = prompt_ids + response_ids`` is compared against
    ``gen.prompt_ids``:

    * **MERGE (exact prefix)** when ``gen.prompt_ids[:len(prev)] == prev``;
    * **MERGE (EOT splice)** when ``eot_token_ids`` is non-empty, ``prev[-1]``
      is an EOT token, ``len(gen.prompt_ids) >= len(prev)``, the first
      ``len(prev) - 1`` tokens match exactly and the token at position
      ``len(prev) - 1`` is an EOT token (the spliced EOT is *not* re-emitted);
    * otherwise a **NEW segment** is opened from the generation and
      ``fragmentation_count`` is incremented.

    On MERGE the delta ``gen.prompt_ids[len(prev):]`` is appended as context
    (``loss_mask == 0``, ``logprobs == math.nan``, ``policy_version == -1``)
    followed by the generation's output tokens (``loss_mask == 1``, the
    generation's logprobs and ``policy_version``) with a generation span
    recorded over the appended output.  Tokens are never modified.

    ``routed_experts`` is carried onto a segment only when *every* generation
    of that segment provides it and the rows are prefix-consistent across
    generations (each later generation's rows must agree with the rows already
    accumulated for the shared prefix); otherwise the segment's
    ``routed_experts`` is ``None`` (never fabricated).

    An empty ``generations`` sequence is refused.
    """
    gens = _as_tuple(generations, "build_segments", "generations")
    if len(gens) == 0:
        _refuse("build_segments.generations: expected a non-empty sequence, got 0 of 0 generations")
    bad_gen = sum(1 for g in gens if not isinstance(g, Generation))
    if bad_gen:
        _refuse(f"build_segments.generations: {bad_gen} of {len(gens)} entries are not Generation")
    if not isinstance(eot_token_ids, frozenset):
        _refuse(
            f"build_segments.eot_token_ids: expected frozenset, got {type(eot_token_ids).__name__}"
        )
    bad_eot = sum(1 for t in eot_token_ids if not (_is_int(t) and t >= 0))
    if bad_eot:
        _refuse(
            f"build_segments.eot_token_ids: {bad_eot} of {len(eot_token_ids)} "
            "entries are not ints >= 0"
        )

    segments: list = []
    fragmentation_count = 0

    # Accumulator state for the segment currently being built.
    s_prompt: list = []
    s_response: list = []
    s_mask: list = []
    s_lp: list = []
    s_pv: list = []
    s_spans: list = []
    s_experts: list = []  # rows for prompt + response, or [] if unknown
    s_experts_ok = True  # every generation so far provided routed_experts
    s_experts_layers: int | None = None

    def _flush() -> None:
        nonlocal s_prompt, s_response, s_mask, s_lp, s_pv, s_spans
        nonlocal s_experts, s_experts_ok, s_experts_layers
        idx = len(segments)
        experts: tuple[tuple[tuple[int, ...], ...], ...] | None = None
        if s_experts_ok and len(s_experts) == len(s_prompt) + len(s_response):
            experts = tuple(tuple(tuple(ids) for ids in row) for row in s_experts)
        segments.append(
            Segment(
                segment_index=idx,
                prompt_ids=tuple(s_prompt),
                response_ids=tuple(s_response),
                loss_mask=tuple(s_mask),
                logprobs=tuple(s_lp),
                policy_version=tuple(s_pv),
                generation_spans=tuple(s_spans),
                routed_experts=experts,
            )
        )
        s_prompt, s_response, s_mask, s_lp, s_pv, s_spans = [], [], [], [], [], []
        s_experts = []
        s_experts_ok = True
        s_experts_layers = None

    def _open(gen: Generation) -> None:
        nonlocal s_prompt, s_response, s_mask, s_lp, s_pv, s_spans
        nonlocal s_experts, s_experts_ok, s_experts_layers
        s_prompt = list(gen.prompt_ids)
        s_response = list(gen.output_ids)
        s_mask = [1] * len(gen.output_ids)
        s_lp = list(gen.logprobs)
        s_pv = [gen.policy_version] * len(gen.output_ids)
        s_spans = [(0, len(gen.output_ids), gen.generation_id)]
        if gen.routed_experts is not None:
            s_experts = [tuple(tuple(ids) for ids in row) for row in gen.routed_experts]
            s_experts_ok = True
            s_experts_layers = len(s_experts[0]) if s_experts else 0
        else:
            s_experts = []
            s_experts_ok = False
            s_experts_layers = None

    def _merge(gen: Generation, delta_start: int) -> None:
        nonlocal s_response, s_mask, s_lp, s_pv, s_spans
        nonlocal s_experts, s_experts_ok, s_experts_layers
        delta = list(gen.prompt_ids[delta_start:])
        s_response.extend(delta)
        s_mask.extend([0] * len(delta))
        s_lp.extend([math.nan] * len(delta))
        s_pv.extend([-1] * len(delta))

        out_start = len(s_response)
        s_response.extend(gen.output_ids)
        s_mask.extend([1] * len(gen.output_ids))
        s_lp.extend(gen.logprobs)
        s_pv.extend([gen.policy_version] * len(gen.output_ids))
        s_spans.append((out_start, out_start + len(gen.output_ids), gen.generation_id))

        # routed_experts: carried only if every generation has it and the rows
        # are prefix-consistent with what has already been accumulated.
        if gen.routed_experts is None:
            s_experts_ok = False
            s_experts = []
            s_experts_layers = None
            return
        rows = [tuple(tuple(ids) for ids in row) for row in gen.routed_experts]
        if s_experts_layers is not None and (len(rows) == 0 or len(rows[0]) != s_experts_layers):
            s_experts_ok = False
            s_experts = []
            return
        if s_experts_layers is None:
            s_experts_layers = len(rows[0]) if rows else 0
        # The generation's rows cover its own prompt + output.  The already
        # accumulated rows cover the segment's prompt + response so far.  The
        # generation's prompt must agree with the accumulated prefix rows.
        if s_experts_ok:
            # rows[:len(gen.prompt_ids)] correspond to gen.prompt_ids, which is
            # a prefix (or EOT-spliced prefix) of the accumulated stream.
            overlap = min(len(s_experts), len(gen.prompt_ids))
            for i in range(overlap):
                if rows[i] != s_experts[i]:
                    s_experts_ok = False
                    s_experts = []
                    return
            # Append only the rows for the delta + output tokens.
            appended = rows[delta_start:]
            s_experts.extend(appended)
        else:
            s_experts = []

    for gen in gens:
        if not s_prompt:
            _open(gen)
            continue

        prev_len = len(s_prompt) + len(s_response)
        prev = tuple(s_prompt) + tuple(s_response)
        gp = gen.prompt_ids

        merge = False
        delta_start = prev_len
        if (
            len(gp) >= prev_len
            and gp[:prev_len] == prev
            or (
                eot_token_ids
                and prev_len >= 1
                and prev[-1] in eot_token_ids
                and len(gp) >= prev_len
                and gp[: prev_len - 1] == prev[: prev_len - 1]
                and gp[prev_len - 1] in eot_token_ids
            )
        ):
            merge = True
            delta_start = prev_len

        if merge:
            _merge(gen, delta_start)
        else:
            _flush()
            fragmentation_count += 1
            _open(gen)

    _flush()
    return tuple(segments), fragmentation_count


def validate_episode(ep: Episode) -> None:
    """Re-check cross-object invariants of a fully assembled :class:`Episode`.

    Per-object invariants are enforced by the dataclasses themselves; this
    function additionally verifies:

    * ``OK`` and trainable ``TIMEOUT`` episodes carry at least one trainable
      token across all segments;
    * ``FidelityStatus.MISMATCH`` <-> ``EpisodeStatus.FIDELITY_MISMATCH``;
    * generation ids are unique across all segment spans (and across
      discarded generations).
    """
    if not isinstance(ep, Episode):
        _refuse(f"validate_episode.ep: expected Episode, got {type(ep).__name__}")

    total_trainable = sum(seg.num_trainable_tokens for seg in ep.segments)

    if ep.status == EpisodeStatus.OK and total_trainable < 1:
        _refuse(
            f"Episode.segments: OK episode requires >= 1 trainable token, "
            f"got {total_trainable} of {total_trainable} tokens trainable"
        )
    if ep.status == EpisodeStatus.TIMEOUT and ep.trainable_on_timeout and total_trainable < 1:
        _refuse(
            f"Episode.segments: trainable TIMEOUT episode requires >= 1 trainable "
            f"token, got {total_trainable} of {total_trainable} tokens trainable"
        )

    mismatch = ep.fidelity.status == FidelityStatus.MISMATCH
    status_mismatch = ep.status == EpisodeStatus.FIDELITY_MISMATCH
    if mismatch != status_mismatch:
        _refuse(
            "Episode.fidelity/Episode.status: FidelityStatus.MISMATCH <-> "
            f"EpisodeStatus.FIDELITY_MISMATCH violated "
            f"(fidelity={ep.fidelity.status.value!r}, status={ep.status.value!r})"
        )

    seen: dict = {}
    total_spans = 0
    for seg in ep.segments:
        for span in seg.generation_spans:
            total_spans += 1
            gen_id = span[2]
            if gen_id in seen:
                _refuse(
                    f"Episode.segments: generation id {gen_id!r} appears in "
                    f"{seen[gen_id]} and segment {seg.segment_index} "
                    f"({total_spans} of {total_spans} spans checked)"
                )
            seen[gen_id] = f"segment {seg.segment_index}"
    for pos, disc in enumerate(ep.discarded):
        gen_id = disc.generation.generation_id
        if gen_id in seen:
            _refuse(
                f"Episode.discarded: generation id {gen_id!r} appears in "
                f"{seen[gen_id]} and discarded[{pos}] "
                f"({pos + 1} of {len(ep.discarded)} discarded checked)"
            )
        seen[gen_id] = f"discarded[{pos}]"


def is_trainable(ep: Episode) -> bool:
    """D6: an episode is trainable iff ``status`` is ``OK``, or ``TIMEOUT``
    with ``trainable_on_timeout`` set."""
    if not isinstance(ep, Episode):
        _refuse(f"is_trainable.ep: expected Episode, got {type(ep).__name__}")
    return ep.status == EpisodeStatus.OK or (
        ep.status == EpisodeStatus.TIMEOUT and ep.trainable_on_timeout
    )


def episode_reward(ep: Episode, *, reducer: str = "single") -> float | None:
    """R3: reduce EPISODE-scope reward events to a single scalar.

    Reducers:

    * ``"single"`` — exactly 0 or 1 EPISODE-scope events; 0 -> ``None``,
      1 -> its value (``None`` for an abstention); more than 1 -> refusal
      naming the count.
    * ``"last_by_ts"`` — the value of the EPISODE-scope event with the largest
      ``at_ts`` (ties broken by last occurrence); no events -> ``None``.
    * ``"max"`` — the maximum of the non-``None`` values; all ``None`` (or no
      events) -> ``None``.

    Any other reducer is refused.
    """
    if not isinstance(ep, Episode):
        _refuse(f"episode_reward.ep: expected Episode, got {type(ep).__name__}")
    if not isinstance(reducer, str):
        _refuse(f"episode_reward.reducer: expected str, got {type(reducer).__name__}")

    events = [r for r in ep.reward_events if r.scope == RewardScope.EPISODE]

    if reducer == "single":
        n = len(events)
        if n == 0:
            return None
        if n == 1:
            return events[0].value
        _refuse(
            f"episode_reward: reducer 'single' requires 0 or 1 EPISODE-scope "
            f"events, got {n} of {n} events"
        )
    elif reducer == "last_by_ts":
        if not events:
            return None
        best = events[0]
        for ev in events[1:]:
            if ev.at_ts >= best.at_ts:
                best = ev
        return best.value
    elif reducer == "max":
        values = [ev.value for ev in events if ev.value is not None]
        if not values:
            return None
        return max(values)

    _refuse(
        f"episode_reward.reducer: unknown reducer {reducer!r} "
        "(expected 'single', 'last_by_ts' or 'max')"
    )


def to_json(ep: Episode) -> dict:
    """Serialise an :class:`Episode` to a plain JSON-safe ``dict``.

    Enums become their ``.value`` strings, tuples become lists, ``MappingProxy``
    values become plain dicts, and ``NaN`` logprobs (context-delta tokens) are
    serialised as ``None``.  ``from_json`` is the exact inverse.
    """
    if not isinstance(ep, Episode):
        _refuse(f"to_json.ep: expected Episode, got {type(ep).__name__}")

    def _lp(v: float) -> float | None:
        return None if (isinstance(v, float) and math.isnan(v)) else v

    def _m(m: Mapping) -> dict:
        return dict(m)

    def _experts(rows: tuple[tuple[tuple[int, ...], ...], ...] | None) -> list | None:
        if rows is None:
            return None
        return [[[int(e) for e in layer] for layer in row] for row in rows]

    def _tool(tc: ToolCallV2) -> dict:
        return {
            "call_id": tc.call_id,
            "name": tc.name,
            "arguments_raw": tc.arguments_raw,
            "parse_verdict": tc.parse_verdict.value,
            "parse_error": tc.parse_error,
            "call_id_origin": tc.call_id_origin,
        }

    def _gen(g: Generation) -> dict:
        return {
            "generation_id": g.generation_id,
            "prompt_ids": [int(t) for t in g.prompt_ids],
            "output_ids": [int(t) for t in g.output_ids],
            "logprobs": [_lp(v) for v in g.logprobs],
            "policy_version": g.policy_version,
            "finish_reason": g.finish_reason,
            "sampling_requested": _m(g.sampling_requested),
            "sampling_overrides": _m(g.sampling_overrides),
            "tool_calls": [_tool(tc) for tc in g.tool_calls],
            "routed_experts": _experts(g.routed_experts),
        }

    def _seg(s: Segment) -> dict:
        return {
            "segment_index": s.segment_index,
            "prompt_ids": [int(t) for t in s.prompt_ids],
            "response_ids": [int(t) for t in s.response_ids],
            "loss_mask": [int(m) for m in s.loss_mask],
            "logprobs": [_lp(v) for v in s.logprobs],
            "policy_version": [int(v) for v in s.policy_version],
            "generation_spans": [[int(a), int(b), gid] for (a, b, gid) in s.generation_spans],
            "routed_experts": _experts(s.routed_experts),
        }

    def _reward(r: RewardEvent) -> dict:
        return {
            "source": r.source,
            "verifier_id": r.verifier_id,
            "verifier_version": r.verifier_version,
            "value": r.value,
            "scope": r.scope.value,
            "span": None if r.span is None else [int(r.span[0]), int(r.span[1])],
            "at_ts": r.at_ts,
            "delayed": r.delayed,
            "details": _m(r.details),
        }

    out: dict = {
        "schema_version": ep.schema_version,
        "episode_id": ep.episode_id,
        "group_id": ep.group_id,
        "task_prompt_ids": [int(t) for t in ep.task_prompt_ids],
        "status": ep.status.value,
        "termination": ep.termination.value,
        "segments": [_seg(s) for s in ep.segments],
        "reward_events": [_reward(r) for r in ep.reward_events],
        "model": {
            "name": ep.model.name,
            "version": ep.model.version,
            "checkpoint_sha": ep.model.checkpoint_sha,
        },
        "harness": {
            "name": ep.harness.name,
            "version": ep.harness.version,
            "commit": ep.harness.commit,
            "license": ep.harness.license,
        },
        "env": {
            "kind": ep.env.kind.value,
            "task_id": ep.env.task_id,
            "task_version": ep.env.task_version,
            "image_digest": ep.env.image_digest,
        },
        "gateway": {
            "api_format": ep.gateway.api_format.value,
            "session_id": ep.gateway.session_id,
        },
        "sandbox": None
        if ep.sandbox is None
        else {
            "network_policy": ep.sandbox.network_policy,
            "node": ep.sandbox.node,
            "cpu": ep.sandbox.cpu,
            "mem_mb": ep.sandbox.mem_mb,
            "started_at": ep.sandbox.started_at,
            "ended_at": ep.sandbox.ended_at,
            "exit_code": ep.sandbox.exit_code,
        },
        "timings": {
            "queue_ms": ep.timings.queue_ms,
            "init_ms": ep.timings.init_ms,
            "run_ms": ep.timings.run_ms,
            "reward_ms": ep.timings.reward_ms,
        },
        "fidelity": {
            "status": ep.fidelity.status.value,
            "detail": ep.fidelity.detail,
        },
        "discarded": [{"generation": _gen(d.generation), "reason": d.reason} for d in ep.discarded],
        "trainable_on_timeout": ep.trainable_on_timeout,
        "tool_schema_sha256": ep.tool_schema_sha256,
    }
    return out


def from_json(d: dict) -> Episode:
    """Exact inverse of :func:`to_json`; re-validates the reconstructed episode.

    ``None`` at a ``loss_mask == 0`` logprob position is restored to
    ``math.nan``.  An extra stdlib import is performed locally (the module's
    import set is fixed by PART A).
    """
    if not isinstance(d, Mapping):
        _refuse(f"from_json.d: expected a Mapping, got {type(d).__name__}")

    def _tool(o: Mapping) -> ToolCallV2:
        return ToolCallV2(
            call_id=o.get("call_id"),
            name=o.get("name"),
            arguments_raw=o.get("arguments_raw"),
            parse_verdict=ParseVerdict(o["parse_verdict"]),
            parse_error=o.get("parse_error"),
            call_id_origin=o.get("call_id_origin", "model"),
        )

    def _gen(o: Mapping) -> Generation:
        return Generation(
            generation_id=o["generation_id"],
            prompt_ids=tuple(o["prompt_ids"]),
            output_ids=tuple(o["output_ids"]),
            logprobs=tuple(o["logprobs"]),
            policy_version=o["policy_version"],
            finish_reason=o.get("finish_reason"),
            sampling_requested=dict(o.get("sampling_requested") or {}),
            sampling_overrides=dict(o.get("sampling_overrides") or {}),
            tool_calls=tuple(_tool(tc) for tc in o.get("tool_calls") or ()),
            routed_experts=None
            if o.get("routed_experts") is None
            else tuple(
                tuple(tuple(int(e) for e in layer) for layer in row) for row in o["routed_experts"]
            ),
        )

    def _seg(o: Mapping) -> Segment:
        mask = tuple(o["loss_mask"])
        raw_lp = o["logprobs"]
        lps = []
        for v, m in zip(raw_lp, mask, strict=False):
            if v is None:
                lps.append(math.nan if m == 0 else float("nan"))
            else:
                lps.append(float(v))
        return Segment(
            segment_index=o["segment_index"],
            prompt_ids=tuple(o["prompt_ids"]),
            response_ids=tuple(o["response_ids"]),
            loss_mask=mask,
            logprobs=tuple(lps),
            policy_version=tuple(o["policy_version"]),
            generation_spans=tuple((int(a), int(b), gid) for (a, b, gid) in o["generation_spans"]),
            routed_experts=None
            if o.get("routed_experts") is None
            else tuple(
                tuple(tuple(int(e) for e in layer) for layer in row) for row in o["routed_experts"]
            ),
        )

    def _reward(o: Mapping) -> RewardEvent:
        span = o.get("span")
        return RewardEvent(
            source=o["source"],
            verifier_id=o["verifier_id"],
            verifier_version=o["verifier_version"],
            value=o.get("value"),
            scope=RewardScope(o["scope"]),
            span=None if span is None else (int(span[0]), int(span[1])),
            at_ts=o.get("at_ts", 0.0),
            delayed=o.get("delayed", False),
            details=dict(o.get("details") or {}),
        )

    model_o = d["model"]
    harness_o = d["harness"]
    env_o = d["env"]
    gateway_o = d["gateway"]
    sandbox_o = d.get("sandbox")
    timings_o = d.get("timings") or {}
    fidelity_o = d.get("fidelity") or {"status": FidelityStatus.VERIFIED.value}

    ep = Episode(
        episode_id=d["episode_id"],
        group_id=d["group_id"],
        task_prompt_ids=tuple(d["task_prompt_ids"]),
        status=EpisodeStatus(d["status"]),
        termination=Termination(d["termination"]),
        segments=tuple(_seg(s) for s in d["segments"]),
        reward_events=tuple(_reward(r) for r in d["reward_events"]),
        model=ModelMeta(
            name=model_o["name"],
            version=model_o["version"],
            checkpoint_sha=model_o["checkpoint_sha"],
        ),
        harness=HarnessMeta(
            name=harness_o["name"],
            version=harness_o["version"],
            commit=harness_o.get("commit"),
            license=harness_o.get("license"),
        ),
        env=EnvMeta(
            kind=EnvKind(env_o["kind"]),
            task_id=env_o["task_id"],
            task_version=env_o.get("task_version"),
            image_digest=env_o.get("image_digest"),
        ),
        gateway=GatewayMeta(
            api_format=ApiFormat(gateway_o["api_format"]),
            session_id=gateway_o["session_id"],
        ),
        sandbox=None
        if sandbox_o is None
        else SandboxMeta(
            network_policy=sandbox_o["network_policy"],
            node=sandbox_o.get("node"),
            cpu=sandbox_o.get("cpu"),
            mem_mb=sandbox_o.get("mem_mb"),
            started_at=sandbox_o.get("started_at"),
            ended_at=sandbox_o.get("ended_at"),
            exit_code=sandbox_o.get("exit_code"),
        ),
        timings=Timings(
            queue_ms=timings_o.get("queue_ms"),
            init_ms=timings_o.get("init_ms"),
            run_ms=timings_o.get("run_ms"),
            reward_ms=timings_o.get("reward_ms"),
        ),
        fidelity=Fidelity(
            status=FidelityStatus(fidelity_o["status"]),
            detail=fidelity_o.get("detail"),
        ),
        discarded=tuple(
            DiscardedGeneration(generation=_gen(x["generation"]), reason=x["reason"])
            for x in d.get("discarded") or ()
        ),
        trainable_on_timeout=d.get("trainable_on_timeout", False),
        tool_schema_sha256=d.get("tool_schema_sha256"),
        schema_version=d.get("schema_version", SCHEMA_VERSION),
    )
    validate_episode(ep)
    return ep


# ============================================================================
# PART C — v1 interop: from_v1, to_v1, flatten_v2
# ============================================================================


def from_v1(
    traj: Any,
    *,
    model: ModelMeta,
    harness: HarnessMeta,
    env: EnvMeta,
    gateway: GatewayMeta,
) -> Episode:
    """R11: lift one v1 ``contracts.Trajectory`` into a one-segment v2 Episode.

    The v1 prompt tokens become the segment prompt and the v1 response tokens
    (``turns`` in order) become the segment response.  The loss mask follows v1
    exactly (``generated`` ASSISTANT/TOOL_CALL tokens), logprobs are the v1
    rollout logprobs where the mask is 1 and ``math.nan`` where it is 0, and the
    per-token policy version is the v1 ``policy_version_max`` where the mask is 1
    and exactly ``-1`` where it is 0 (context logprobs are ``nan`` and their
    version ``-1``).  Fidelity is ``UNVERIFIED``: v1 carries no fidelity claim.

    A v1 reward becomes exactly one EPISODE-scope ``RewardEvent`` with
    ``source="v1"``, ``verifier_id="v1"``, ``verifier_version="v1"``; a v1
    abstention becomes no reward event at all (never a ``0.0``).

    Status mapping: ``Termination.INFRA`` -> ``INFRA_ERROR`` (termination
    ``ERROR``), ``Termination.ABORTED`` -> ``INFRA_ERROR`` with termination
    ``ABORTED``, everything else -> ``OK``.  ``group_id`` is the v1 ``uid`` (the
    prompt id) and ``task_prompt_ids`` the v1 prompt tokens.
    """
    from foundationscale.agentic_rl import contracts as v1

    if not isinstance(traj, v1.Trajectory):
        _refuse(f"from_v1.traj: expected contracts.Trajectory, got {type(traj).__name__}")

    prompt_ids = tuple(t for turn in traj.prompt_turns for t in turn.token_ids)
    response_ids = tuple(t for turn in traj.turns for t in turn.token_ids)

    mask: list = []
    lps: list = []
    pvs: list = []
    for turn in traj.turns:
        supervised = turn.generated and turn.kind in (
            v1.SegmentKind.ASSISTANT,
            v1.SegmentKind.TOOL_CALL,
        )
        n = len(turn.token_ids)
        mask.extend([int(supervised)] * n)
        if supervised and turn.logprobs is not None:
            lps.extend(turn.logprobs)
        else:
            lps.extend([math.nan] * n)
        if supervised:
            if traj.policy_version_max is None:
                _refuse(
                    f"from_v1: v1 trajectory {traj.session_id!r} has {n} generated token(s) but "
                    f"policy_version_max is None; v2 needs a real policy version on every "
                    f"trained token and never invents one"
                )
            pvs.extend([traj.policy_version_max] * n)
        else:
            pvs.extend([-1] * n)

    spans: list = []
    offset = 0
    for turn in traj.turns:
        n = len(turn.token_ids)
        supervised = turn.generated and turn.kind in (
            v1.SegmentKind.ASSISTANT,
            v1.SegmentKind.TOOL_CALL,
        )
        if supervised:
            spans.append((offset, offset + n, f"v1-turn-{turn.index}"))
        offset += n

    segment = Segment(
        segment_index=0,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=tuple(mask),
        logprobs=tuple(lps),
        policy_version=tuple(pvs),
        generation_spans=tuple(spans),
    )

    # v1 TIMEOUT is a scored v1 row, so it maps to v2 TIMEOUT with trainable_on_timeout
    # set: the row stays trainable exactly as v1 trained it, and the timeout stays visible.
    trainable_on_timeout = False
    if traj.termination is v1.Termination.INFRA:
        status = EpisodeStatus.INFRA_ERROR
        termination = Termination.ERROR
    elif traj.termination is v1.Termination.ABORTED:
        status = EpisodeStatus.INFRA_ERROR
        termination = Termination.ABORTED
    elif traj.termination is v1.Termination.TIMEOUT:
        status = EpisodeStatus.TIMEOUT
        termination = Termination.ERROR
        trainable_on_timeout = True
    else:
        status = EpisodeStatus.OK
        termination = {
            v1.Termination.STOP: Termination.STOP,
            v1.Termination.LENGTH: Termination.LENGTH,
            v1.Termination.STEP_LIMIT: Termination.TURN_LIMIT,
        }[traj.termination]

    reward_events: tuple = ()
    if traj.reward is not None:
        reward_events = (
            RewardEvent(
                source="v1",
                verifier_id="v1",
                verifier_version="v1",
                value=float(traj.reward),
                scope=RewardScope.EPISODE,
            ),
        )

    ep = Episode(
        episode_id=f"{traj.uid}::{traj.session_id}",
        group_id=traj.uid,
        task_prompt_ids=prompt_ids,
        status=status,
        termination=termination,
        segments=(segment,),
        reward_events=reward_events,
        model=model,
        harness=harness,
        env=env,
        gateway=gateway,
        fidelity=Fidelity(status=FidelityStatus.UNVERIFIED),
        trainable_on_timeout=trainable_on_timeout,
    )
    validate_episode(ep)
    return ep


def to_v1(ep: Episode) -> Any:
    """Project a v2 Episode back onto a v1 ``contracts.Trajectory``.

    Refusals (each naming both counts):

    * more than one segment -> use :func:`flatten_v2`;
    * any ``ToolCallV2`` with ``ParseVerdict.UNOBSERVED`` -> v1 has no third
      verdict and would have to invent one.

    Status mapping: trainable episodes (``is_trainable``) become a normal v1
    termination with the episode reward; non-trainable episodes become v1
    ``INFRA`` (or ``ABORTED`` when the v2 termination is ``ABORTED``) with no
    reward and an ``infra:`` abstention reason.
    """
    from foundationscale.agentic_rl import contracts as v1

    if not isinstance(ep, Episode):
        _refuse(f"to_v1.ep: expected Episode, got {type(ep).__name__}")

    if len(ep.segments) > 1:
        n_segments = len(ep.segments)
        _refuse(
            f"to_v1.ep: episode has {n_segments} of {n_segments} segments; "
            "a v1 Trajectory carries one attempt -- use flatten_v2 to emit one "
            "row per segment"
        )

    unobserved = 0
    for disc in ep.discarded:
        for tc in disc.generation.tool_calls:
            if tc.parse_verdict is ParseVerdict.UNOBSERVED:
                unobserved += 1
    # Tool calls live on generations; the episode's own generations are the
    # discarded ones plus the spans' source generations, which are not retained
    # on Segment.  The only place ToolCallV2 objects are reachable on an
    # Episode is ep.discarded; check those and refuse UNOBSERVED verdicts.
    if unobserved:
        _refuse(
            f"to_v1.ep: {unobserved} of {unobserved} tool calls carry "
            "ParseVerdict.UNOBSERVED; v1 ToolCall has no unobserved verdict and "
            "inventing one would assert a parse that was never reported"
        )

    trainable = is_trainable(ep)

    prompt_turns: list = []
    if ep.segments:
        prompt_turns.append(
            v1.Turn(
                index=0,
                kind=v1.SegmentKind.USER,
                token_ids=ep.segments[0].prompt_ids,
                generated=False,
                logprobs=None,
            )
        )

    turns: list = []
    if ep.segments:
        seg = ep.segments[0]
        turns.append(
            v1.Turn(
                index=1,
                kind=v1.SegmentKind.ASSISTANT,
                token_ids=seg.response_ids,
                generated=True,
                logprobs=tuple(
                    lp if m == 1 else 0.0 for lp, m in zip(seg.logprobs, seg.loss_mask, strict=True)
                )
                if trainable
                else None,
            )
        )

    if trainable:
        reward = episode_reward(ep)
        abstention = None if reward is not None else "v2: no EPISODE-scope reward event"
        if ep.termination is Termination.STOP:
            v1_term = v1.Termination.STOP
        elif ep.termination is Termination.LENGTH:
            v1_term = v1.Termination.LENGTH
        elif ep.termination is Termination.TURN_LIMIT:
            v1_term = v1.Termination.STEP_LIMIT
        elif ep.termination is Termination.ABORTED:
            v1_term = v1.Termination.ABORTED
        else:
            v1_term = v1.Termination.STOP
    else:
        reward = None
        if ep.termination is Termination.ABORTED:
            v1_term = v1.Termination.ABORTED
            abstention = "infra: aborted before a sample"
        else:
            v1_term = v1.Termination.INFRA
            abstention = f"infra: v2 status {ep.status.value!r} is not trainable"

    pmin = (
        ep.segments[0].policy_version[0] if ep.segments and ep.segments[0].policy_version else None
    )
    pmax = (
        ep.segments[0].policy_version[-1] if ep.segments and ep.segments[0].policy_version else None
    )
    if pmin is not None and pmin < 0:
        pmin = None
        pmax = None

    return v1.Trajectory(
        uid=ep.group_id,
        session_id=ep.episode_id,
        harness=ep.harness.name,
        prompt_turns=tuple(prompt_turns),
        turns=tuple(turns),
        reward=reward,
        abstention_reason=abstention,
        termination=v1_term,
        policy_version_min=pmin,
        policy_version_max=pmax,
    )


def flatten_v2(episodes: Sequence[Episode]) -> tuple[Any, dict]:
    """R10: one v1 trajectory ROW PER SEGMENT, then ``contracts.flatten``.

    For an episode with ``k`` segments this emits ``k`` rows: row 0 carries the
    episode's task prompt as the v1 prompt and segment 0's prompt+response on the
    response side; rows ``k > 0`` reuse the episode's ``group_id``/task prompt as
    the v1 prompt and put the segment's own prompt+response on the response side
    as v1 allows, with the segment prompt marked non-generated (context).

    Non-trainable episodes become masked v1 ``INFRA`` rows (all-0 loss mask),
    never dropped and never given a ``0.0`` reward.

    Returns ``(batch, extras)`` where ``extras`` holds
    ``{"policy_version": [...], "segment_index": [...], "episode_id": [...]}``
    aligned row-for-row with the batch.
    """
    from foundationscale.agentic_rl import contracts as v1

    eps = tuple(episodes)
    if not eps:
        _refuse("flatten_v2.episodes: expected a non-empty sequence, got 0 of 0 episodes")
    bad_ep = sum(1 for e in eps if not isinstance(e, Episode))
    if bad_ep:
        _refuse(f"flatten_v2.episodes: {bad_ep} of {len(eps)} entries are not Episode")

    rows: list = []
    extras: dict = {"policy_version": [], "segment_index": [], "episode_id": []}

    for ep in eps:
        trainable = is_trainable(ep)
        for k, seg in enumerate(ep.segments):
            if trainable:
                if ep.termination is Termination.STOP:
                    v1_term = v1.Termination.STOP
                elif ep.termination is Termination.LENGTH:
                    v1_term = v1.Termination.LENGTH
                elif ep.termination is Termination.TURN_LIMIT:
                    v1_term = v1.Termination.STEP_LIMIT
                elif ep.termination is Termination.ABORTED:
                    v1_term = v1.Termination.ABORTED
                else:
                    v1_term = v1.Termination.STOP
                reward = episode_reward(ep)
                abstention = None if reward is not None else "v2: no EPISODE-scope reward event"
            else:
                if ep.termination is Termination.ABORTED:
                    v1_term = v1.Termination.ABORTED
                    abstention = "infra: aborted before a sample"
                else:
                    v1_term = v1.Termination.INFRA
                    abstention = f"infra: v2 status {ep.status.value!r} is not trainable"
                reward = None

            prompt_turns = (
                v1.Turn(
                    index=0,
                    kind=v1.SegmentKind.USER,
                    token_ids=ep.task_prompt_ids,
                    generated=False,
                    logprobs=None,
                ),
            )
            # R7: a segment whose prompt is not the task prompt (rewritten / compacted
            # history) carries that prompt on the response side as non-generated context.
            ctx = () if seg.prompt_ids == ep.task_prompt_ids else seg.prompt_ids
            turn_list: list[Any] = []
            if ctx:
                turn_list.append(
                    v1.Turn(
                        index=1,
                        kind=v1.SegmentKind.OBSERVATION,
                        token_ids=ctx,
                        generated=False,
                        logprobs=None,
                    )
                )
            # Split the response into maximal runs of equal mask: mask-1 runs are the
            # engine-sampled tokens (generated, with their logprobs); mask-0 runs are
            # context the harness inserted (tool results, observations), never generated.
            pos = 0
            n_resp = len(seg.response_ids)
            while pos < n_resp:
                bit = seg.loss_mask[pos]
                run_end = pos
                while run_end < n_resp and seg.loss_mask[run_end] == bit:
                    run_end += 1
                run_ids = seg.response_ids[pos:run_end]
                if bit == 1:
                    turn_list.append(
                        v1.Turn(
                            index=len(turn_list) + 1,
                            kind=v1.SegmentKind.ASSISTANT,
                            token_ids=run_ids,
                            generated=True,
                            logprobs=(tuple(seg.logprobs[pos:run_end]) if trainable else None),
                        )
                    )
                else:
                    turn_list.append(
                        v1.Turn(
                            index=len(turn_list) + 1,
                            kind=v1.SegmentKind.OBSERVATION,
                            token_ids=run_ids,
                            generated=False,
                            logprobs=None,
                        )
                    )
                pos = run_end
            turns = tuple(turn_list)

            sampled_versions = [
                v for v, m in zip(seg.policy_version, seg.loss_mask, strict=True) if m == 1
            ]
            pmin = min(sampled_versions) if sampled_versions else None
            pmax = max(sampled_versions) if sampled_versions else None

            rows.append(
                v1.Trajectory(
                    uid=ep.group_id,
                    session_id=f"{ep.episode_id}::seg{k}",
                    harness=ep.harness.name,
                    prompt_turns=prompt_turns,
                    turns=turns,
                    reward=reward,
                    abstention_reason=abstention,
                    termination=v1_term,
                    policy_version_min=pmin,
                    policy_version_max=pmax,
                )
            )
            extras["policy_version"].append([-1] * len(ctx) + list(seg.policy_version))
            extras["segment_index"].append(k)
            extras["episode_id"].append(ep.episode_id)

    batch = v1.flatten(rows)
    return batch, extras
