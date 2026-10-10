"""``foundationscale.agentic_rl.gateway.server`` — the HTTP gateway.

Binding API (P0.2d).  Stdlib only plus the provided modules
(:mod:`foundationscale.agentic_rl.gateway.wire`,
:mod:`foundationscale.agentic_rl.gateway.session`,
:mod:`foundationscale.agentic_rl.trajectory_v2`,
:mod:`foundationscale.agentic_rl.harness.base` and
:mod:`foundationscale.agentic_rl.tools`); nothing here is copied from them.

Doctrine (FS):

* :meth:`GatewayCore.handle` is PURE: it routes one request from
  ``(method, path, headers, body, now)`` to a :class:`GatewayResponse` and never
  touches a socket, a clock or the environment.  ``serve`` is a thin
  :class:`http.server.BaseHTTPRequestHandler` wrapper around it.
* The model-weight swap lock is a readers-writer lock built from one
  :class:`threading.Condition`: model calls take the READ side (many concurrent
  calls), ``/fs/v1/weights/begin`` takes the WRITE side (new model calls wait
  while in-flight calls finish first).
* The global inflight limit is a :class:`threading.BoundedSemaphore` acquired
  NON-BLOCKINGLY; when no slot is free the request is refused with ``429`` and
  ``Retry-After: 1``.
* Request bodies and API keys are NEVER logged or echoed in error messages.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hmac
import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, NoReturn
from urllib.parse import parse_qs, urlsplit

from foundationscale.agentic_rl.gateway.session import (
    EngineCall,
    Mode,
    SamplingContract,
    Session,
    SessionClosed,
    SessionNotFound,
    SessionRefusal,
    SessionStore,
    SessionStoreFull,
    build_episode,
)
from foundationscale.agentic_rl.gateway.wire import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResult,
    CanonicalTool,
    CanonicalToolCall,
    WireRefusal,
    detect_format,
    parse_request,
    render_response,
    render_stream,
)
from foundationscale.agentic_rl.harness.base import (
    ChatTokenizer,
    Generation,
    GenerationClient,
    SamplingParams,
)
from foundationscale.agentic_rl.tools import ParsedCall, ToolCallParser
from foundationscale.agentic_rl.trajectory_v2 import (
    EnvKind,
    EnvMeta,
    EpisodeStatus,
    HarnessMeta,
    ModelMeta,
    RewardEvent,
    RewardScope,
    Termination,
)
from foundationscale.agentic_rl.trajectory_v2 import to_json as episode_to_json

__all__ = [
    "GatewayConfig",
    "GatewayResponse",
    "GatewayCore",
    "ReadersWriterLock",
    "serve",
]


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _refuse(expected: str, got: Any) -> NoReturn:
    """Raise a :class:`SessionRefusal` naming what was expected and what arrived."""

    raise SessionRefusal(f"expected {expected}, got {got!r}")


def _is_int(x: Any) -> bool:
    return type(x) is int


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Readers-writer lock (threading.Condition)
# ---------------------------------------------------------------------------


class ReadersWriterLock:
    """A readers-writer lock built from one :class:`threading.Condition`.

    Many readers may hold the lock concurrently; a writer holds it exclusively
    and waits for in-flight readers to drain first.  Writers are preferred over
    new readers once one is waiting, so ``weights/begin`` cannot be starved.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer or self._waiting_writers > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# GatewayConfig / GatewayResponse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatewayConfig:
    """Static configuration for one gateway instance."""

    served_model_name: str
    sampling: SamplingContract
    admin_key: str
    max_sessions: int = 1024
    max_calls_per_session: int = 512
    max_inflight: int = 64
    base_url: str = "http://127.0.0.1:0"

    def __post_init__(self) -> None:
        if not _is_nonempty_str(self.served_model_name):
            _refuse(
                "GatewayConfig.served_model_name to be a non-empty str",
                self.served_model_name,
            )
        if not isinstance(self.sampling, SamplingContract):
            _refuse("GatewayConfig.sampling to be a SamplingContract", self.sampling)
        if not _is_nonempty_str(self.admin_key):
            _refuse("GatewayConfig.admin_key to be a non-empty str", self.admin_key)
        if not (_is_int(self.max_sessions) and self.max_sessions >= 1):
            _refuse("GatewayConfig.max_sessions to be an int >= 1", self.max_sessions)
        if not (_is_int(self.max_calls_per_session) and self.max_calls_per_session >= 1):
            _refuse(
                "GatewayConfig.max_calls_per_session to be an int >= 1",
                self.max_calls_per_session,
            )
        if not (_is_int(self.max_inflight) and self.max_inflight >= 1):
            _refuse("GatewayConfig.max_inflight to be an int >= 1", self.max_inflight)
        if not _is_nonempty_str(self.base_url):
            _refuse("GatewayConfig.base_url to be a non-empty str", self.base_url)


@dataclass(frozen=True)
class GatewayResponse:
    """One routed HTTP response: JSON body or SSE frames."""

    status: int
    headers: Mapping[str, str]
    json_body: Any | None = None
    sse: tuple[str, ...] | None = None


# ---------------------------------------------------------------------------
# GatewayCore
# ---------------------------------------------------------------------------

_JSON_HEADERS: Mapping[str, str] = {"Content-Type": "application/json"}
_SSE_HEADERS: Mapping[str, str] = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
}

_MODEL_PATH_RE = re.compile(
    r"^/v1beta/models/(?P<model>[^/:]+):(?P<action>generateContent|streamGenerateContent)$"
)
_SESSION_PREFIX_RE = re.compile(r"^/sess/(?P<sid>[^/]+)(?P<rest>/.*)$")

_MODEL_CALL_SUFFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages",
)


def _error_body(error_type: str, message: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": error_type}
    if message is not None:
        payload["message"] = message
    return {"error": payload}


def _json_response(status: int, body: Any) -> GatewayResponse:
    return GatewayResponse(status=status, headers=dict(_JSON_HEADERS), json_body=body)


def _error_response(status: int, error_type: str, message: str | None = None) -> GatewayResponse:
    return _json_response(status, _error_body(error_type, message))


def _sse_response(status: int, frames: Sequence[str]) -> GatewayResponse:
    return GatewayResponse(
        status=status,
        headers=dict(_SSE_HEADERS),
        sse=tuple(frames),
    )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return value
    return None


def _bearer_token(headers: Mapping[str, str]) -> str | None:
    """Return the ``Bearer`` token from ``Authorization`` when present."""

    raw = _header(headers, "authorization")
    if raw is None:
        return None
    parts = raw.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _client_api_key(headers: Mapping[str, str], query: Mapping[str, list[str]]) -> str | None:
    """Resolve a session API key from headers or a Gemini ``?key=`` query."""

    for name in ("x-api-key", "x-goog-api-key"):
        value = _header(headers, name)
        if value:
            return value.strip()
    bearer = _bearer_token(headers)
    if bearer is not None:
        return bearer
    values = query.get("key")
    if values:
        candidate = values[0]
        if candidate:
            return candidate
    return None


def _openai_messages(messages: Sequence[CanonicalMessage]) -> list[dict[str, Any]]:
    """Render canonical messages as OpenAI-style dicts for the chat template."""

    out: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_raw},
                }
                for call in msg.tool_calls
            ]
        if msg.tool_call_id is not None:
            item["tool_call_id"] = msg.tool_call_id
        if msg.name is not None:
            item["name"] = msg.name
        out.append(item)
    return out


def _openai_tools(tools: Sequence[CanonicalTool]) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }
        for tool in tools
    ]


def _sampling_params(engine_params: Mapping[str, Any]) -> SamplingParams:
    """Build engine :class:`SamplingParams` from the enforced engine params."""

    max_tokens = engine_params.get("max_tokens")
    if max_tokens is None:
        max_tokens = 512
    top_k = engine_params.get("top_k")
    if top_k is None or top_k < 0:
        top_k = 0
    return SamplingParams(
        temperature=float(engine_params.get("temperature", 1.0)),
        top_p=float(engine_params.get("top_p", 1.0)),
        top_k=int(top_k),
        min_p=float(engine_params.get("min_p", 0.0)),
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_new_tokens=int(max_tokens),
    )


class GatewayCore:
    """Pure request router for the agentic-RL gateway."""

    def __init__(
        self,
        config: GatewayConfig,
        *,
        client: GenerationClient,
        tokenizer: ChatTokenizer,
        parser: ToolCallParser,
        decode: Callable[[Sequence[int]], str],
    ) -> None:
        if not isinstance(config, GatewayConfig):
            _refuse("GatewayCore.config to be a GatewayConfig", config)
        self.config = config
        self.client = client
        self.tokenizer = tokenizer
        self.parser = parser
        self.decode = decode
        self.store = SessionStore(
            max_sessions=config.max_sessions,
            max_calls_per_session=config.max_calls_per_session,
        )
        self._swap = ReadersWriterLock()
        self._swap_open = False
        self._policy_version = 0
        self._inflight = threading.BoundedSemaphore(config.max_inflight)

    # -- policy version ----------------------------------------------------

    @property
    def policy_version(self) -> int:
        """Current policy version (read under the swap read-lock)."""

        self._swap.acquire_read()
        try:
            return self._policy_version
        finally:
            self._swap.release_read()

    # -- routing ------------------------------------------------------------

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        now: float,
    ) -> GatewayResponse:
        """Route one HTTP request (pure: no sockets)."""

        split = urlsplit(path)
        raw_path = split.path
        try:
            query = parse_qs(split.query, keep_blank_values=True)
        except ValueError:
            return _error_response(400, "invalid_request", "malformed query string")

        prefix = _SESSION_PREFIX_RE.match(raw_path)
        if prefix is not None:
            session_id = prefix.group("sid")
            route_path = prefix.group("rest")
            return self._handle_model_call(
                method=method,
                path=route_path,
                headers=headers,
                body=body,
                now=now,
                query=query,
                session_id=session_id,
            )

        if raw_path.startswith("/fs/v1/"):
            return self._handle_control(method, raw_path, headers, body, now)

        return self._handle_model_call(
            method=method,
            path=raw_path,
            headers=headers,
            body=body,
            now=now,
            query=query,
            session_id=None,
        )

    # -- control plane ------------------------------------------------------

    def _handle_control(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        now: float,
    ) -> GatewayResponse:
        if path == "/fs/v1/health" and method == "GET":
            return _json_response(
                200,
                {
                    "ok": True,
                    "sessions": len(self.store),
                    "policy_version": self.policy_version,
                },
            )

        if not self._authorized_admin(headers):
            return _error_response(401, "unauthorized")

        if path == "/fs/v1/sessions" and method == "POST":
            return self._create_session(body, now)
        if path == "/fs/v1/weights/begin" and method == "POST":
            self._swap.acquire_write()
            self._swap_open = True
            return _json_response(200, {"ok": True})
        if path == "/fs/v1/weights/commit" and method == "POST":
            return self._commit_weights(body)

        match = re.fullmatch(
            r"/fs/v1/sessions/(?P<sid>[^/]+)/(?P<op>reward|end|trajectory)",
            path,
        )
        if match is not None and method == "POST":
            sid = match.group("sid")
            op = match.group("op")
            if op == "reward":
                return self._add_reward(sid, body)
            if op == "end":
                return self._end_session(sid, body)
            return self._trajectory(sid, body)

        return _error_response(404, "not_found")

    def _authorized_admin(self, headers: Mapping[str, str]) -> bool:
        token = _bearer_token(headers)
        if token is None:
            return False
        return hmac.compare_digest(token, self.config.admin_key)

    def _create_session(self, body: bytes, now: float) -> GatewayResponse:
        payload = self._load_json(body)
        if payload is None:
            return _error_response(400, "invalid_request", "body must be a JSON object")
        mode_raw = payload.get("mode")
        if mode_raw == "train":
            mode = Mode.TRAIN
        elif mode_raw == "eval":
            mode = Mode.EVAL
        else:
            return _error_response(
                400, "invalid_request", "expected 'mode' to be 'train' or 'eval'"
            )
        group_id = payload.get("group_id")
        if not _is_nonempty_str(group_id):
            return _error_response(
                400, "invalid_request", "expected 'group_id' to be a non-empty string"
            )
        session_id = payload.get("session_id")
        if session_id is not None and not _is_nonempty_str(session_id):
            return _error_response(
                400, "invalid_request", "expected 'session_id' to be a non-empty string"
            )
        try:
            session = self.store.create(
                mode=mode,
                group_id=str(group_id),
                created_at=now,
                session_id=session_id,
            )
        except SessionStoreFull as exc:
            return _error_response(503, "record_store_full", str(exc))
        except SessionRefusal as exc:
            return _error_response(400, "invalid_request", str(exc))

        base = self.config.base_url.rstrip("/")
        sid = session.session_id
        return _json_response(
            201,
            {
                "session_id": sid,
                "api_key": session.api_key,
                "base_urls": {
                    "chat_completions": f"{base}/sess/{sid}/v1",
                    "responses": f"{base}/sess/{sid}/v1",
                    "anthropic_messages": f"{base}/sess/{sid}",
                    "gemini": f"{base}/sess/{sid}",
                },
            },
        )

    def _add_reward(self, session_id: str, body: bytes) -> GatewayResponse:
        payload = self._load_json(body)
        if payload is None:
            return _error_response(400, "invalid_request", "body must be a JSON object")
        try:
            fields = dict(payload)
            if "scope" in fields:
                fields["scope"] = RewardScope(fields["scope"])  # JSON carries the value
            if isinstance(fields.get("span"), list):
                fields["span"] = tuple(fields["span"])
            event = RewardEvent(**fields)
        except (TypeError, ValueError) as exc:
            return _error_response(400, "invalid_request", str(exc))
        try:
            self.store.add_reward(session_id, event)
        except SessionNotFound as exc:
            return _error_response(404, "not_found", str(exc))
        except SessionRefusal as exc:
            return _error_response(400, "invalid_request", str(exc))
        return _json_response(200, {"ok": True})

    def _end_session(self, session_id: str, body: bytes) -> GatewayResponse:
        payload = self._load_json(body)
        if payload is None:
            return _error_response(400, "invalid_request", "body must be a JSON object")
        try:
            status = EpisodeStatus(payload.get("status"))
            termination = Termination(payload.get("termination"))
        except (TypeError, ValueError) as exc:
            return _error_response(400, "invalid_request", str(exc))
        try:
            self.store.close(session_id, status=status, termination=termination)
        except SessionNotFound as exc:
            return _error_response(404, "not_found", str(exc))
        except SessionClosed as exc:
            return _error_response(409, "session_closed", str(exc))
        except SessionRefusal as exc:
            return _error_response(400, "invalid_request", str(exc))
        return _json_response(200, {"ok": True})

    def _trajectory(self, session_id: str, body: bytes) -> GatewayResponse:
        payload = self._load_json(body)
        if payload is None:
            return _error_response(400, "invalid_request", "body must be a JSON object")
        try:
            model = ModelMeta(**payload.get("model", {}))
            harness = HarnessMeta(**payload.get("harness", {}))
            env_raw = dict(payload.get("env", {}))
            if "kind" in env_raw:
                env_raw["kind"] = EnvKind(env_raw["kind"])  # JSON carries the enum's value
            env = EnvMeta(**env_raw)
        except (TypeError, ValueError) as exc:
            return _error_response(400, "invalid_request", str(exc))
        try:
            session = self.store.get(session_id)
        except SessionNotFound as exc:
            return _error_response(404, "not_found", str(exc))
        try:
            episode = build_episode(session, model=model, harness=harness, env=env)
        except SessionRefusal as exc:
            return _error_response(400, "invalid_request", str(exc))
        return _json_response(200, episode_to_json(episode))

    def _commit_weights(self, body: bytes) -> GatewayResponse:
        if not self._swap_open:
            # A commit without its begin would change the version with no lock held.
            return _error_response(409, "no_swap_in_progress", "POST /fs/v1/weights/begin first")
        payload = self._load_json(body)
        if payload is None:
            self._swap_open = False
            self._swap.release_write()
            return _error_response(400, "invalid_request", "body must be a JSON object")
        version = payload.get("version")
        if not (type(version) is int and version > self._policy_version):
            self._swap_open = False
            self._swap.release_write()
            return _error_response(
                400,
                "invalid_request",
                "expected 'version' to be an int greater than the current policy version",
            )
        self._policy_version = version
        self._swap_open = False
        self._swap.release_write()
        return _json_response(200, {"ok": True, "policy_version": version})

    # -- model calls --------------------------------------------------------

    def _handle_model_call(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        now: float,
        query: Mapping[str, list[str]],
        session_id: str | None,
    ) -> GatewayResponse:
        if method == "GET" and path.endswith("/v1/models"):
            return _json_response(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.config.served_model_name,
                            "object": "model",
                            "owned_by": "foundationscale",
                        }
                    ],
                },
            )
        if method != "POST":
            return _error_response(404, "not_found")

        session = self._resolve_session(headers, query, session_id)
        if isinstance(session, GatewayResponse):
            return session

        payload = self._load_json(body)
        if payload is None:
            return _error_response(400, "invalid_request", "body must be a JSON object")

        try:
            fmt = detect_format(path, headers, payload)
            request = parse_request(fmt, payload)
        except WireRefusal as exc:
            return _error_response(400, "invalid_request", str(exc))

        if session.mode is Mode.TRAIN and request.has_images:
            return _error_response(
                400,
                "modality_not_declared",
                "expected a text-only request in TRAIN mode, got image or file content",
            )

        try:
            engine_params, overrides = self.config.sampling.enforce(request.sampling)
        except SessionRefusal as exc:
            return _error_response(400, "invalid_request", str(exc))

        messages = _openai_messages(request.messages)
        tools = _openai_tools(request.tools)
        try:
            prompt_ids = self.tokenizer.render(
                messages,
                tools=tools,
                add_generation_prompt=True,
            )
        except Exception as exc:  # noqa: BLE001 - engine/template failures are 502
            return _error_response(502, "engine_error", str(exc))

        if not self._inflight.acquire(blocking=False):
            return GatewayResponse(
                status=429,
                headers={"Content-Type": "application/json", "Retry-After": "1"},
                json_body=_error_body(
                    "rate_limited",
                    "expected a free inflight slot, got the global engine-call limit reached",
                ),
            )

        self._swap.acquire_read()
        try:
            version = self._policy_version
            try:
                generation = asyncio.run(
                    self.client.generate(prompt_ids, _sampling_params(engine_params))
                )
            except Exception as exc:  # noqa: BLE001 - engine failures are 502
                return _error_response(502, "engine_error", str(exc))
        finally:
            self._swap.release_read()
            self._inflight.release()

        return self._finish_call(
            session=session,
            request=request,
            overrides=overrides,
            prompt_ids=prompt_ids,
            generation=generation,
            version=version,
            now=now,
        )

    def _finish_call(
        self,
        *,
        session: Session,
        request: CanonicalRequest,
        overrides: Mapping[str, Any],
        prompt_ids: Sequence[int],
        generation: Generation,
        version: int,
        now: float,
    ) -> GatewayResponse:
        # D3: ids and logprobs come only from the engine; a call that lacks them, or that the
        # engine aborted, is an engine error -- never recorded, never filled in.
        if generation.logprobs is None or len(generation.logprobs) != len(generation.token_ids):
            return _error_response(
                502,
                "engine_error",
                f"engine returned {0 if generation.logprobs is None else len(generation.logprobs)} "
                f"logprobs for {len(generation.token_ids)} sampled tokens",
            )
        if generation.finish_reason == "abort":
            return _error_response(502, "engine_error", "engine aborted the generation")
        text = self.decode(generation.token_ids)
        parsed, remainder = self.parser.parse(text)

        tool_calls: list[CanonicalToolCall] = []
        for index, call in enumerate(parsed):
            if not isinstance(call, ParsedCall) or call.error is not None:
                continue
            arguments = call.arguments if call.arguments is not None else {}
            tool_calls.append(
                CanonicalToolCall(
                    call_id=f"call_{generation_id(session)}_{index}",
                    name=call.name,
                    arguments_raw=_compact_json(arguments),
                    call_id_origin="model",
                )
            )

        if tool_calls:
            finish_reason = "tool_calls"
            # Valid calls travel as structured tool_calls; only the leftover prose and the raw
            # markup of calls that failed to parse stay in the visible content.
            failed_raw = "".join(
                c.raw for c in parsed if isinstance(c, ParsedCall) and c.error is not None
            )
            content: str | None = (remainder + failed_raw) or None
        else:
            finish_reason = generation.finish_reason
            content = text
        logprobs = generation.logprobs
        requested = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(request.sampling).items()
            if v is not None and v != ()
        }

        generation_id_value = generation_id(session)
        engine_call = EngineCall(
            generation_id=generation_id_value,
            api_format=request.format,
            prompt_ids=tuple(prompt_ids),
            output_ids=tuple(generation.token_ids),
            logprobs=tuple(logprobs),
            policy_version=version,
            finish_reason=finish_reason,
            sampling_requested=requested,
            sampling_overrides=dict(overrides),
            tool_calls=tuple(tool_calls),
            latency_ms=None,
        )
        try:
            self.store.record_call(session.session_id, engine_call)
        except SessionStoreFull as exc:
            return _error_response(503, "record_store_full", str(exc))
        except (SessionClosed, SessionNotFound, SessionRefusal) as exc:
            return _error_response(409, "session_closed", str(exc))

        result = CanonicalResult(
            text=content,
            tool_calls=tuple(tool_calls),
            reasoning=None,
            finish_reason=finish_reason,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generation.token_ids),
            response_id=generation_id_value,
            created=int(now),
        )

        if request.stream:
            frames = render_stream(request.format, request, result)
            return _sse_response(200, frames)
        return _json_response(200, render_response(request.format, request, result))

    # -- shared helpers -----------------------------------------------------

    def _resolve_session(
        self,
        headers: Mapping[str, str],
        query: Mapping[str, list[str]],
        session_id: str | None,
    ) -> Session | GatewayResponse:
        key = _client_api_key(headers, query)
        if session_id is not None:
            try:
                session = self.store.get(session_id)
            except SessionNotFound as exc:
                return _error_response(404, "not_found", str(exc))
            if key is not None and not hmac.compare_digest(key, session.api_key):
                return _error_response(401, "unauthorized")
            if session.closed:
                return _error_response(409, "session_closed")
            return session

        if key is None:
            return _error_response(401, "unauthorized")
        try:
            session = self.store.by_api_key(key)
        except SessionNotFound as exc:
            return _error_response(404, "not_found", str(exc))
        if session.closed:
            return _error_response(409, "session_closed")
        return session

    def _load_json(self, body: bytes) -> dict[str, Any] | None:
        if not body:
            return None
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload


def generation_id(session: Session) -> str:
    """Stable id for the next engine call of ``session`` (``gen_<n>``)."""

    return f"gen_{len(session.calls)}"


# ---------------------------------------------------------------------------
# serve()
# ---------------------------------------------------------------------------


class _GatewayHandler(BaseHTTPRequestHandler):
    """Thin :class:`BaseHTTPRequestHandler` wrapper around :meth:`GatewayCore.handle`."""

    core: GatewayCore
    server_version = "FoundationScaleGateway/1.0"
    sys_version = ""

    def _dispatch(self) -> None:
        length_raw = self.headers.get("Content-Length", "0")
        try:
            length = int(length_raw)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""
        headers = {str(k): str(v) for k, v in self.headers.items()}
        response = self.core.handle(self.command, self.path, headers, body, _now())
        self._write(response)

    def _write(self, response: GatewayResponse) -> None:
        self.send_response(response.status)
        for name, value in response.headers.items():
            self.send_header(name, value)
        if response.sse is not None:
            payload = "".join(response.sse).encode("utf-8")
        else:
            payload = json.dumps(
                response.json_body, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: ARG002 - stdlib signature
        """Never log request bodies or keys."""

        return


def _now() -> float:
    import time

    return time.time()


def serve(core: GatewayCore, host: str, port: int) -> ThreadingHTTPServer:
    """Bind (port 0 = ephemeral) and return the server.

    The caller runs ``serve_forever`` in a thread.  The handler maps each
    request to :meth:`GatewayCore.handle` and writes JSON or SSE frames, never
    logging bodies or keys.
    """

    if not isinstance(core, GatewayCore):
        _refuse("serve.core to be a GatewayCore", core)
    if not _is_nonempty_str(host):
        _refuse("serve.host to be a non-empty str", host)
    if not (_is_int(port) and 0 <= port <= 65535):
        _refuse("serve.port to be an int in [0, 65535]", port)

    handler = type("_BoundGatewayHandler", (_GatewayHandler,), {"core": core})
    return ThreadingHTTPServer((host, port), handler)
