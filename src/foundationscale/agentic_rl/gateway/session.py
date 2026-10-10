"""``foundationscale.agentic_rl.gateway.session`` — session state and episode assembly.

Binding API (P0.2b+c).  Stdlib only plus the two provided modules
(:mod:`foundationscale.agentic_rl.gateway.wire` and
:mod:`foundationscale.agentic_rl.trajectory_v2`); nothing here is copied from
them.

Doctrine (FS):

* One :class:`threading.Lock` in :class:`SessionStore` guards every mutation and
  lookup of session state.
* Frozen dataclasses where specified (:class:`SamplingContract`,
  :class:`EngineCall`); :class:`Session` is a plain mutable dataclass.
* ``bool`` is checked with ``type(x) is bool`` BEFORE any ``int`` check.
* Refusal messages name BOTH counts ("k of n ...").
* No environment variables and no ``time.time()`` calls: callers pass every
  timestamp in.
"""

from __future__ import annotations

import math
import secrets
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, NoReturn

from foundationscale.agentic_rl.gateway.wire import (
    CanonicalToolCall,
    SamplingRequest,
    WireFormat,
)
from foundationscale.agentic_rl.trajectory_v2 import (
    ApiFormat,
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
    SandboxMeta,
    Termination,
    Timings,
    ToolCallV2,
    build_segments,
    validate_episode,
)

__all__ = [
    "SessionRefusal",
    "SessionNotFound",
    "SessionStoreFull",
    "SessionClosed",
    "Mode",
    "SamplingContract",
    "EngineCall",
    "Session",
    "SessionStore",
    "api_format_for",
    "build_episode",
]


class SessionRefusal(ValueError):
    """Raised on bad session input (HTTP 400 later).

    Messages name what was expected and what arrived, with counts where
    relevant.
    """


class SessionNotFound(KeyError):
    """Raised when a session id or api key does not resolve (HTTP 404 later)."""


class SessionStoreFull(RuntimeError):
    """Raised when the store or a session is at capacity (HTTP 503 later).

    ``record_store_full`` / R12: a full store is NEVER a silent drop.
    """


class SessionClosed(RuntimeError):
    """Raised on operations against a closed session (HTTP 409 later)."""


class Mode(str, Enum):
    """Whether the session's samples are used for training or evaluation."""

    TRAIN = "train"
    EVAL = "eval"


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _refuse(expected: str, got: Any) -> NoReturn:
    """Raise a refusal naming what was expected and what arrived."""

    raise SessionRefusal(f"expected {expected}, got {got!r}")


def _is_int(x: Any) -> bool:
    """True iff ``x`` is a real ``int`` (``bool`` is never an ``int`` here)."""

    return type(x) is int


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite(x: Any) -> bool:
    return _is_number(x) and math.isfinite(float(x))


def _is_nonneg_finite(x: Any) -> bool:
    return _is_finite(x) and float(x) >= 0.0


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


def _freeze_mapping(m: Any, owner: str, field_name: str) -> Mapping[str, Any]:
    if not isinstance(m, Mapping):
        _refuse(f"{owner}.{field_name} to be a Mapping", m)
    return MappingProxyType(dict(m))


def _as_tuple(seq: Any, owner: str, field_name: str) -> tuple[Any, ...]:
    if isinstance(seq, tuple):
        return seq
    if isinstance(seq, (list, frozenset, set)):
        return tuple(seq)
    _refuse(f"{owner}.{field_name} to be a sequence", seq)


# ---------------------------------------------------------------------------
# SamplingContract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingContract:
    """D3 / R13: the gateway's sampling policy for one session.

    ``temperature`` is finite and ``>= 0`` and is applied in TRAIN mode.
    ``max_tokens_cap`` caps the client's ``max_tokens`` (``min(client, cap)``);
    ``None`` keeps the client value.
    """

    mode: Mode
    temperature: float
    max_tokens_cap: int | None = None

    def __post_init__(self) -> None:
        mode: object = self.mode
        if not isinstance(mode, Mode):
            _refuse("SamplingContract.mode to be a Mode", self.mode)
        if not _is_finite(self.temperature) or float(self.temperature) < 0.0:
            _refuse(
                "SamplingContract.temperature to be a finite float >= 0",
                self.temperature,
            )
        cap: object = self.max_tokens_cap
        if cap is not None and not (type(cap) is int and cap >= 1):
            _refuse(
                "SamplingContract.max_tokens_cap to be None or an int >= 1",
                self.max_tokens_cap,
            )

    def _capped_max_tokens(self, req: SamplingRequest) -> int | None:
        client_max = req.max_tokens
        cap = self.max_tokens_cap
        if cap is None:
            return client_max
        if client_max is None:
            return cap
        return min(client_max, cap)

    def enforce(self, req: SamplingRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply this contract to a client :class:`SamplingRequest`.

        Returns ``(engine_params, overrides)``.

        TRAIN: ``engine_params`` is ``{"temperature": self.temperature,
        "top_p": 1.0, "top_k": -1, "min_p": 0.0, "n": 1,
        "max_tokens": capped, "stop": list(req.stop)}`` plus ``"seed":
        req.seed`` only when the client sent a seed.

        EVAL: the client's ``temperature``/``top_p``/``top_k``/``n`` are kept
        when given (defaults ``temperature=self.temperature``, ``top_p=1.0``,
        ``top_k=-1``, ``n=1``); ``max_tokens`` is capped the same way.

        ``overrides`` maps ``field -> {"requested": v_client, "applied":
        v_engine}`` for every field whose applied value differs from what the
        client sent.  A field the client did NOT send is never an override.
        """

        if not isinstance(req, SamplingRequest):
            _refuse("SamplingContract.enforce.req to be a SamplingRequest", req)

        capped = self._capped_max_tokens(req)
        overrides: dict[str, Any] = {}

        if self.mode is Mode.TRAIN:
            engine_params: dict[str, Any] = {
                "temperature": self.temperature,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "n": 1,
                "max_tokens": capped,
                "stop": list(req.stop),
            }
            if req.seed is not None:
                engine_params["seed"] = req.seed

            applied_by_field: dict[str, Any] = {
                "temperature": self.temperature,
                "top_p": 1.0,
                "top_k": -1,
                "n": 1,
                "max_tokens": capped,
                "stop": list(req.stop),
            }
            requested_by_field: dict[str, Any] = {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "top_k": req.top_k,
                "n": req.n,
                "max_tokens": req.max_tokens,
                "stop": list(req.stop),
            }
            for name, requested in requested_by_field.items():
                if requested is None:
                    continue
                applied = applied_by_field[name]
                if applied != requested:
                    overrides[name] = {"requested": requested, "applied": applied}
        else:
            temperature = self.temperature if req.temperature is None else req.temperature
            top_p = 1.0 if req.top_p is None else req.top_p
            top_k = -1 if req.top_k is None else req.top_k
            n = 1 if req.n is None else req.n
            engine_params = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "n": n,
                "max_tokens": capped,
                "stop": list(req.stop),
            }
            if req.seed is not None:
                engine_params["seed"] = req.seed

            applied_by_field = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "n": n,
                "max_tokens": capped,
                "stop": list(req.stop),
            }
            requested_by_field = {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "top_k": req.top_k,
                "n": req.n,
                "max_tokens": req.max_tokens,
                "stop": list(req.stop),
            }
            for name, requested in requested_by_field.items():
                if requested is None:
                    continue
                applied = applied_by_field[name]
                if applied != requested:
                    overrides[name] = {"requested": requested, "applied": applied}

        return engine_params, overrides


# ---------------------------------------------------------------------------
# EngineCall
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineCall:
    """One engine round-trip, engine-reported tokens only.

    ``tool_calls`` are the tool calls as parsed by the ENGINE tool parser and
    become :class:`ToolCallV2` objects with
    :attr:`ParseVerdict.ENGINE_PARSER` and the canonical call's
    ``call_id_origin``.
    """

    generation_id: str
    api_format: WireFormat
    prompt_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    policy_version: int
    finish_reason: str | None
    sampling_requested: Mapping[str, Any]
    sampling_overrides: Mapping[str, Any]
    tool_calls: tuple[CanonicalToolCall, ...] = ()
    routed_experts: tuple[tuple[tuple[int, ...], ...], ...] | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.generation_id):
            _refuse("EngineCall.generation_id to be a non-empty str", self.generation_id)
        api_format: object = self.api_format
        if not isinstance(api_format, WireFormat):
            _refuse("EngineCall.api_format to be a WireFormat", self.api_format)

        prompt_ids = _as_tuple(self.prompt_ids, "EngineCall", "prompt_ids")
        output_ids = _as_tuple(self.output_ids, "EngineCall", "output_ids")
        logprobs = _as_tuple(self.logprobs, "EngineCall", "logprobs")
        tool_calls = _as_tuple(self.tool_calls, "EngineCall", "tool_calls")
        object.__setattr__(self, "prompt_ids", prompt_ids)
        object.__setattr__(self, "output_ids", output_ids)
        object.__setattr__(self, "logprobs", logprobs)
        object.__setattr__(self, "tool_calls", tool_calls)

        if len(prompt_ids) == 0:
            _refuse("EngineCall.prompt_ids to be a non-empty tuple, got 0 of 0 tokens", 0)
        bad_prompt = sum(1 for t in prompt_ids if not (_is_int(t) and t >= 0))
        if bad_prompt:
            _refuse(
                f"EngineCall.prompt_ids: {bad_prompt} of {len(prompt_ids)} tokens to be ints >= 0",
                bad_prompt,
            )

        if len(output_ids) == 0:
            _refuse("EngineCall.output_ids to be a non-empty tuple, got 0 of 0 tokens", 0)
        bad_output = sum(1 for t in output_ids if not (_is_int(t) and t >= 0))
        if bad_output:
            _refuse(
                f"EngineCall.output_ids: {bad_output} of {len(output_ids)} tokens to be ints >= 0",
                bad_output,
            )

        if len(logprobs) != len(output_ids):
            _refuse(
                f"EngineCall.logprobs: {len(logprobs)} of {len(output_ids)} entries "
                "(len(logprobs) must equal len(output_ids))",
                len(logprobs),
            )
        bad_lp = sum(1 for lp in logprobs if not _is_finite(lp))
        if bad_lp:
            _refuse(
                f"EngineCall.logprobs: {bad_lp} of {len(logprobs)} entries to be finite floats",
                bad_lp,
            )

        policy_version: object = self.policy_version
        if not (_is_int(policy_version) and self.policy_version >= 0):
            _refuse(
                "EngineCall.policy_version to be an int >= 0",
                self.policy_version,
            )

        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            _refuse("EngineCall.finish_reason to be None or str", self.finish_reason)

        object.__setattr__(
            self,
            "sampling_requested",
            _freeze_mapping(self.sampling_requested, "EngineCall", "sampling_requested"),
        )
        object.__setattr__(
            self,
            "sampling_overrides",
            _freeze_mapping(self.sampling_overrides, "EngineCall", "sampling_overrides"),
        )

        bad_tc = sum(1 for tc in tool_calls if not isinstance(tc, CanonicalToolCall))
        if bad_tc:
            _refuse(
                f"EngineCall.tool_calls: {bad_tc} of {len(tool_calls)} entries "
                "to be CanonicalToolCall",
                bad_tc,
            )

        if self.routed_experts is not None:
            rows = _as_tuple(self.routed_experts, "EngineCall", "routed_experts")
            expected = len(prompt_ids) + len(output_ids)
            if len(rows) != expected:
                _refuse(
                    f"EngineCall.routed_experts: {len(rows)} of {expected} rows "
                    "(must cover prompt_ids + output_ids)",
                    len(rows),
                )
            for i, row in enumerate(rows):
                layers = _as_tuple(row, "EngineCall", f"routed_experts[{i}]")
                for j, layer in enumerate(layers):
                    ids = _as_tuple(layer, "EngineCall", f"routed_experts[{i}][{j}]")
                    bad_ids = sum(1 for e in ids if not (_is_int(e) and e >= 0))
                    if bad_ids:
                        _refuse(
                            f"EngineCall.routed_experts[{i}][{j}]: {bad_ids} of {len(ids)} "
                            "expert ids to be ints >= 0",
                            bad_ids,
                        )
                    layers = layers[:j] + (ids,) + layers[j + 1 :]
                rows = rows[:i] + (layers,) + rows[i + 1 :]
            object.__setattr__(self, "routed_experts", rows)

        if self.latency_ms is not None and not _is_nonneg_finite(self.latency_ms):
            _refuse(
                "EngineCall.latency_ms to be None or a finite float >= 0",
                self.latency_ms,
            )

    def to_generation(self) -> Generation:
        """Project this call onto a trajectory-v2 :class:`Generation`.

        Tool calls become :class:`ToolCallV2` with
        ``parse_verdict=ParseVerdict.ENGINE_PARSER`` and ``call_id_origin``
        taken from the canonical call.
        """

        tool_calls = tuple(
            ToolCallV2(
                call_id=call.call_id,
                name=call.name,
                arguments_raw=call.arguments_raw,
                parse_verdict=ParseVerdict.ENGINE_PARSER,
                call_id_origin=call.call_id_origin,
            )
            for call in self.tool_calls
        )
        return Generation(
            generation_id=self.generation_id,
            prompt_ids=self.prompt_ids,
            output_ids=self.output_ids,
            logprobs=self.logprobs,
            policy_version=self.policy_version,
            finish_reason=self.finish_reason,
            sampling_requested=self.sampling_requested,
            sampling_overrides=self.sampling_overrides,
            tool_calls=tool_calls,
            routed_experts=self.routed_experts,
        )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """One gateway session: mutable, guarded by the owning store's lock."""

    session_id: str
    api_key: str
    mode: Mode
    created_at: float
    group_id: str
    task_prompt_ids: tuple[int, ...] | None
    calls: list[EngineCall] = field(default_factory=list)
    reward_events: list[RewardEvent] = field(default_factory=list)
    closed: bool = False
    status: EpisodeStatus | None = None
    termination: Termination | None = None


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """Thread-safe store of :class:`Session` objects.

    A single :class:`threading.Lock` guards every mutation and lookup.
    """

    def __init__(self, *, max_sessions: int, max_calls_per_session: int) -> None:
        if not (_is_int(max_sessions) and max_sessions >= 1):
            _refuse("SessionStore.max_sessions to be an int >= 1", max_sessions)
        if not (_is_int(max_calls_per_session) and max_calls_per_session >= 1):
            _refuse(
                "SessionStore.max_calls_per_session to be an int >= 1",
                max_calls_per_session,
            )
        self._max_sessions = max_sessions
        self._max_calls_per_session = max_calls_per_session
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._by_api_key: dict[str, str] = {}

    def create(
        self,
        *,
        mode: Mode,
        group_id: str,
        created_at: float,
        session_id: str | None = None,
    ) -> Session:
        """Create a session.

        ``session_id`` defaults to ``secrets.token_hex(16)``; the api key is
        ``"fs-sess-" + secrets.token_urlsafe(24)``.  A duplicate id is a
        :class:`SessionRefusal`; being at capacity is a
        :class:`SessionStoreFull`.
        """

        if not isinstance(mode, Mode):
            _refuse("SessionStore.create.mode to be a Mode", mode)
        if not _is_nonempty_str(group_id):
            _refuse("SessionStore.create.group_id to be a non-empty str", group_id)
        if not _is_nonneg_finite(created_at):
            _refuse(
                "SessionStore.create.created_at to be a finite float >= 0",
                created_at,
            )
        if session_id is not None and not _is_nonempty_str(session_id):
            _refuse(
                "SessionStore.create.session_id to be None or a non-empty str",
                session_id,
            )

        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise SessionStoreFull(
                    f"record_store_full: session store is at capacity "
                    f"({len(self._sessions)} of {self._max_sessions} sessions); "
                    "remove a session before creating another"
                )
            sid = session_id if session_id is not None else secrets.token_hex(16)
            if sid in self._sessions:
                _refuse(
                    f"SessionStore.create.session_id to be unused "
                    f"({len(self._sessions)} of {self._max_sessions} sessions in use)",
                    sid,
                )
            api_key = "fs-sess-" + secrets.token_urlsafe(24)
            session = Session(
                session_id=sid,
                api_key=api_key,
                mode=mode,
                created_at=float(created_at),
                group_id=group_id,
                task_prompt_ids=None,
            )
            self._sessions[sid] = session
            self._by_api_key[api_key] = sid
            return session

    def get(self, session_id: str) -> Session:
        """Look up a session by id (:class:`SessionNotFound` when absent)."""

        if not _is_nonempty_str(session_id):
            _refuse("SessionStore.get.session_id to be a non-empty str", session_id)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(
                    f"expected session {session_id!r} to exist "
                    f"({len(self._sessions)} of {self._max_sessions} sessions in store)"
                )
            return session

    def by_api_key(self, api_key: str) -> Session:
        """Look up a session by api key.

        Closed sessions have their api key revoked, so they are
        :class:`SessionNotFound` here.
        """

        if not _is_nonempty_str(api_key):
            _refuse("SessionStore.by_api_key.api_key to be a non-empty str", api_key)
        with self._lock:
            sid = self._by_api_key.get(api_key)
            if sid is None:
                raise SessionNotFound(
                    f"expected api key {api_key!r} to resolve to a live session "
                    f"({len(self._by_api_key)} of {self._max_sessions} keys live)"
                )
            session = self._sessions.get(sid)
            if session is None:
                raise SessionNotFound(
                    f"expected api key {api_key!r} to resolve to a live session "
                    f"({len(self._by_api_key)} of {self._max_sessions} keys live)"
                )
            return session

    def record_call(self, session_id: str, call: EngineCall) -> None:
        """Append one :class:`EngineCall` to a session.

        :class:`SessionClosed` when the session is closed;
        :class:`SessionStoreFull` when the session already holds
        ``max_calls_per_session`` calls (R12: never a silent drop);
        :class:`SessionRefusal` on a duplicate ``generation_id`` within the
        session.  The FIRST call fixes ``task_prompt_ids`` to its
        ``prompt_ids``.
        """

        if not _is_nonempty_str(session_id):
            _refuse("SessionStore.record_call.session_id to be a non-empty str", session_id)
        if not isinstance(call, EngineCall):
            _refuse("SessionStore.record_call.call to be an EngineCall", call)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(
                    f"expected session {session_id!r} to exist "
                    f"({len(self._sessions)} of {self._max_sessions} sessions in store)"
                )
            if session.closed:
                raise SessionClosed(
                    f"expected session {session_id!r} to be open, got a closed session "
                    f"with {len(session.calls)} of {self._max_calls_per_session} calls"
                )
            if len(session.calls) >= self._max_calls_per_session:
                raise SessionStoreFull(
                    f"record_store_full: session {session_id!r} is at capacity "
                    f"({len(session.calls)} of {self._max_calls_per_session} calls); "
                    "the call was refused, not dropped"
                )
            duplicates = sum(1 for c in session.calls if c.generation_id == call.generation_id)
            if duplicates:
                _refuse(
                    f"SessionStore.record_call.call.generation_id to be unique within "
                    f"session {session_id!r} ({duplicates} of {len(session.calls)} "
                    f"recorded calls already use {call.generation_id!r})",
                    call.generation_id,
                )
            if session.task_prompt_ids is None:
                session.task_prompt_ids = call.prompt_ids
            session.calls.append(call)

    def add_reward(self, session_id: str, event: RewardEvent) -> None:
        """Append one :class:`RewardEvent` to a session.

        Allowed AFTER close (delayed rewards, D5).
        """

        if not _is_nonempty_str(session_id):
            _refuse("SessionStore.add_reward.session_id to be a non-empty str", session_id)
        if not isinstance(event, RewardEvent):
            _refuse("SessionStore.add_reward.event to be a RewardEvent", event)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(
                    f"expected session {session_id!r} to exist "
                    f"({len(self._sessions)} of {self._max_sessions} sessions in store)"
                )
            session.reward_events.append(event)

    def close(
        self,
        session_id: str,
        *,
        status: EpisodeStatus,
        termination: Termination,
    ) -> Session:
        """Close a session and revoke its api key.

        Closing twice is an idempotent refusal: :class:`SessionClosed`.
        """

        if not _is_nonempty_str(session_id):
            _refuse("SessionStore.close.session_id to be a non-empty str", session_id)
        if not isinstance(status, EpisodeStatus):
            _refuse("SessionStore.close.status to be an EpisodeStatus", status)
        if not isinstance(termination, Termination):
            _refuse("SessionStore.close.termination to be a Termination", termination)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(
                    f"expected session {session_id!r} to exist "
                    f"({len(self._sessions)} of {self._max_sessions} sessions in store)"
                )
            if session.closed:
                raise SessionClosed(
                    f"expected session {session_id!r} to be open before close, got a "
                    f"session already closed with status {session.status!r} "
                    f"({len(session.calls)} of {self._max_calls_per_session} calls)"
                )
            session.closed = True
            session.status = status
            session.termination = termination
            self._by_api_key.pop(session.api_key, None)
            return session

    def remove(self, session_id: str) -> None:
        """Remove a session, freeing capacity (:class:`SessionNotFound` when absent)."""

        if not _is_nonempty_str(session_id):
            _refuse("SessionStore.remove.session_id to be a non-empty str", session_id)
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                raise SessionNotFound(
                    f"expected session {session_id!r} to exist "
                    f"({len(self._sessions) + 1} of {self._max_sessions} sessions in store)"
                )
            self._by_api_key.pop(session.api_key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


# ---------------------------------------------------------------------------
# build_episode
# ---------------------------------------------------------------------------

_API_FORMAT_BY_VALUE: dict[str, ApiFormat] = {
    "chat_completions": ApiFormat.CHAT_COMPLETIONS,
    "responses": ApiFormat.RESPONSES,
    "anthropic_messages": ApiFormat.ANTHROPIC_MESSAGES,
    "gemini": ApiFormat.GEMINI,
}


def api_format_for(fmt: WireFormat) -> ApiFormat:
    """Map a :class:`WireFormat` to its :class:`ApiFormat` BY VALUE NAME."""

    if not isinstance(fmt, WireFormat):
        _refuse("api_format_for.fmt to be a WireFormat", fmt)
    return _API_FORMAT_BY_VALUE[fmt.value]


def build_episode(
    session: Session,
    *,
    model: ModelMeta,
    harness: HarnessMeta,
    env: EnvMeta,
    sandbox: SandboxMeta | None = None,
    timings: Timings | None = None,
    fidelity: Fidelity | None = None,
    eot_token_ids: frozenset[int] = frozenset(),
    trainable_on_timeout: bool = False,
    tool_schema_sha256: str | None = None,
) -> Episode:
    """Assemble a trajectory-v2 :class:`Episode` from a closed :class:`Session`.

    The session must be closed (:class:`SessionRefusal` otherwise).  A session
    with ZERO calls is allowed only for non-trainable statuses (INFRA/SANDBOX/
    HARNESS errors); since v2 forbids an episode with zero segments, such a
    session is REFUSED with a message telling the caller to record the attempt
    as an abstention elsewhere.

    Otherwise: ``generations = [c.to_generation() for c in calls]``;
    ``segments, _ = build_segments(generations, eot_token_ids=...)``;
    ``gateway = GatewayMeta(api_format=<ApiFormat of the session's calls>,
    session_id)`` (mixed formats -> :class:`SessionRefusal`); ``task_prompt_ids``
    from the session; ``reward_events`` from the session; then
    :func:`validate_episode`.
    """

    if not isinstance(session, Session):
        _refuse("build_episode.session to be a Session", session)
    if not isinstance(model, ModelMeta):
        _refuse("build_episode.model to be a ModelMeta", model)
    if not isinstance(harness, HarnessMeta):
        _refuse("build_episode.harness to be a HarnessMeta", harness)
    if not isinstance(env, EnvMeta):
        _refuse("build_episode.env to be an EnvMeta", env)
    if sandbox is not None and not isinstance(sandbox, SandboxMeta):
        _refuse("build_episode.sandbox to be a SandboxMeta or None", sandbox)
    # None means the default (immutable, but a call in a default argument is evaluated once).
    timings = Timings() if timings is None else timings
    fidelity = Fidelity(FidelityStatus.VERIFIED) if fidelity is None else fidelity
    if not isinstance(timings, Timings):
        _refuse("build_episode.timings to be a Timings", timings)
    if not isinstance(fidelity, Fidelity):
        _refuse("build_episode.fidelity to be a Fidelity", fidelity)
    if not isinstance(eot_token_ids, frozenset):
        _refuse(
            "build_episode.eot_token_ids to be a frozenset",
            type(eot_token_ids).__name__,
        )
    bad_eot = sum(1 for t in eot_token_ids if not (_is_int(t) and t >= 0))
    if bad_eot:
        _refuse(
            f"build_episode.eot_token_ids: {bad_eot} of {len(eot_token_ids)} entries "
            "to be ints >= 0",
            bad_eot,
        )
    if type(trainable_on_timeout) is not bool:
        _refuse("build_episode.trainable_on_timeout to be a bool", trainable_on_timeout)
    if tool_schema_sha256 is not None and not _is_nonempty_str(tool_schema_sha256):
        _refuse(
            "build_episode.tool_schema_sha256 to be None or a non-empty str",
            tool_schema_sha256,
        )

    if not session.closed:
        _refuse(
            f"build_episode.session to be closed before episode assembly "
            f"({len(session.calls)} calls recorded on an open session)",
            session.closed,
        )
    if session.status is None or session.termination is None:
        _refuse(
            "build_episode.session to carry both status and termination after close "
            f"(status={session.status!r}, termination={session.termination!r})",
            session.status,
        )

    calls = tuple(session.calls)
    if len(calls) == 0:
        _refuse(
            "build_episode.session to carry at least 1 of 1 engine calls when an "
            "Episode is built; a zero-call attempt cannot become an Episode with "
            "zero segments (v2 forbids it) -- record the attempt as an abstention "
            "elsewhere instead",
            0,
        )

    api_formats = {api_format_for(call.api_format) for call in calls}
    if len(api_formats) != 1:
        _refuse(
            f"build_episode.session to use exactly 1 of {len(api_formats)} api formats "
            f"across its {len(calls)} calls (mixed formats: "
            f"{sorted(f.value for f in api_formats)!r})",
            len(api_formats),
        )
    api_format = next(iter(api_formats))

    generations = tuple(call.to_generation() for call in calls)
    segments, _fragmentation = build_segments(generations, eot_token_ids=eot_token_ids)

    task_prompt_ids = session.task_prompt_ids
    if task_prompt_ids is None:
        _refuse(
            "build_episode.session.task_prompt_ids to be fixed by its first call "
            f"({len(calls)} calls recorded)",
            task_prompt_ids,
        )

    episode = Episode(
        episode_id=session.session_id,
        group_id=session.group_id,
        task_prompt_ids=task_prompt_ids,
        status=session.status,
        termination=session.termination,
        segments=segments,
        reward_events=tuple(session.reward_events),
        model=model,
        harness=harness,
        env=env,
        gateway=GatewayMeta(api_format=api_format, session_id=session.session_id),
        sandbox=sandbox,
        timings=timings,
        fidelity=fidelity,
        trainable_on_timeout=trainable_on_timeout,
        tool_schema_sha256=tool_schema_sha256,
    )
    validate_episode(episode)
    return episode
