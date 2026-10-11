"""Tests for foundationscale.agentic_rl.gateway.server."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from foundationscale.agentic_rl.gateway.server import (
    GatewayConfig,
    GatewayCore,
    GatewayResponse,
    serve,
)
from foundationscale.agentic_rl.gateway.session import Mode, SamplingContract
from foundationscale.agentic_rl.gateway.wire import CanonicalMessage  # noqa: F401
from foundationscale.agentic_rl.harness.base import Generation, SamplingParams
from foundationscale.agentic_rl.tools import ParsedCall

ADMIN_KEY = "admin-secret-key"
_ISSUED_KEYS: dict[str, str] = {}
SESSION_ID = "sess-1"


def make_sampling() -> SamplingParams:
    """Build a fully valid SamplingContract-compatible SamplingParams."""
    return SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_new_tokens=128,
    )


def make_config(**overrides: Any) -> GatewayConfig:
    """Build a GatewayConfig with sane defaults and optional overrides."""
    fields: dict[str, Any] = {
        "served_model_name": "fs-model",
        "sampling": SamplingContract(mode=Mode.TRAIN, temperature=0.7, max_tokens_cap=128),
        "admin_key": ADMIN_KEY,
        "base_url": "http://127.0.0.1:0",
    }
    fields.update(overrides)
    return GatewayConfig(**fields)


class FakeTokenizer:
    """Fake ChatTokenizer with deterministic len-based ids and call recording."""

    def __init__(self) -> None:
        self.render_calls: list[dict[str, Any]] = []

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
    ) -> list[int]:
        """Return one id per message plus one id per content character."""
        self.render_calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": [dict(t) for t in tools] if tools is not None else None,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        ids: list[int] = []
        for message in messages:
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True)
            ids.append(len(message))
            ids.extend(len(part) for part in content)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Return a fixed decoded text."""
        return "decoded-text"


class FakeGenerationClient:
    """Fake async GenerationClient returning a fixed Generation."""

    def __init__(self, generation: Generation | None = None, exc: Exception | None = None) -> None:
        self.generation = generation or Generation(
            token_ids=(11, 12, 13),
            logprobs=(-0.1, -0.2, -0.3),
            finish_reason="stop",
            text="decoded-text",
        )
        self.exc = exc
        self.calls: list[tuple[tuple[int, ...], SamplingParams]] = []

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        """Record the call and return the fixed Generation or raise."""
        self.calls.append((tuple(prompt_ids), sampling))
        if self.exc is not None:
            raise self.exc
        return self.generation


class FakeParser:
    """Fake ToolCallParser returning a fixed (ParsedCall list, content) pair."""

    def __init__(
        self,
        calls: Sequence[ParsedCall] = (),
        content: str = "plain content",
    ) -> None:
        self.calls = list(calls)
        self.content = content
        self.parse_calls: list[str] = []

    def parse(self, text: str) -> tuple[list[ParsedCall], str]:
        """Record the text and return the fixed parse result."""
        self.parse_calls.append(text)
        return list(self.calls), self.content


def decode_fixed(ids: Sequence[int]) -> str:
    """Return a fixed decoded text for any id sequence."""
    return "decoded-text"


def make_core(
    *,
    client: FakeGenerationClient | None = None,
    tokenizer: FakeTokenizer | None = None,
    parser: FakeParser | None = None,
    decode: Any = decode_fixed,
    config: GatewayConfig | None = None,
) -> tuple[GatewayCore, FakeGenerationClient, FakeTokenizer, FakeParser]:
    """Build a GatewayCore wired to fakes and return it alongside the fakes."""
    tok = tokenizer or FakeTokenizer()
    gen = client or FakeGenerationClient()
    par = parser or FakeParser()
    core = GatewayCore(
        config or make_config(),
        client=gen,
        tokenizer=tok,
        parser=par,
        decode=decode,
    )
    return core, gen, tok, par


def create_session(
    core: GatewayCore,
    *,
    mode: str = "train",
    session_id: str = SESSION_ID,
) -> dict[str, Any]:
    """Create a session through the admin API and return its JSON body."""
    response = core.handle(
        "POST",
        "/fs/v1/sessions",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"mode": mode, "group_id": "g1", "session_id": session_id}).encode(),
        0.0,
    )
    assert response.status == 201
    assert response.json_body is not None
    body = dict(response.json_body)
    _ISSUED_KEYS[session_id] = body["api_key"]
    return body


def key_for(session_id: str = SESSION_ID) -> str:
    """The API key the gateway issued for ``session_id`` (keys are never caller-chosen)."""
    return _ISSUED_KEYS[session_id]


def chat_body(messages: Sequence[Mapping[str, Any]] | None = None, **extra: Any) -> bytes:
    """Build a minimal OpenAI chat completion request body."""
    payload: dict[str, Any] = {
        "model": "fs-model",
        "messages": list(messages) if messages is not None else [{"role": "user", "content": "hi"}],
    }
    payload.update(extra)
    return json.dumps(payload).encode()


def test_admin_auth_401_without_key() -> None:
    """Control plane rejects requests without an admin bearer token."""
    core, _, _, _ = make_core()
    response = core.handle("POST", "/fs/v1/sessions", {}, b"{}", 0.0)
    assert response.status == 401


def test_create_session_returns_base_urls() -> None:
    """Session creation returns the session id, api key, and base URLs."""
    core, _, _, _ = make_core()
    body = create_session(core)
    assert body["session_id"] == SESSION_ID
    assert isinstance(body["api_key"], str) and body["api_key"]
    base = body["base_urls"]
    assert base["chat_completions"].endswith(f"/sess/{SESSION_ID}/v1")
    assert base["responses"].endswith(f"/sess/{SESSION_ID}/v1")
    assert base["anthropic_messages"].endswith(f"/sess/{SESSION_ID}")
    assert base["gemini"].endswith(f"/sess/{SESSION_ID}")


def test_chat_completion_records_engine_ids_and_policy_version() -> None:
    """Chat completion returns chat.completion JSON and records engine ids/logprobs."""
    core, gen, tok, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert response.status == 200
    body = response.json_body
    assert body is not None
    assert body["object"] == "chat.completion"
    assert body["model"] == "fs-model"
    assert len(gen.calls) == 1
    recorded = core.store.get(SESSION_ID).calls
    assert len(recorded) == 1
    call = recorded[0]
    assert tuple(call.output_ids) == (11, 12, 13)
    assert tuple(call.logprobs or ()) == (-0.1, -0.2, -0.3)
    assert call.policy_version == 0
    assert tok.render_calls and tok.render_calls[0]["add_generation_prompt"] is True


def test_key_scoped_call_via_authorization_bearer() -> None:
    """Key-scoped chat completion resolves the session from the bearer api key."""
    core, gen, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        "/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert response.status == 200
    assert response.json_body is not None
    assert response.json_body["object"] == "chat.completion"
    assert len(gen.calls) == 1


def test_x_api_key_for_anthropic_messages() -> None:
    """Anthropic messages path resolves the session from the x-api-key header."""
    core, gen, _, _ = make_core()
    create_session(core)
    body = json.dumps(
        {"model": "fs-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    ).encode()
    response = core.handle(
        "POST",
        "/v1/messages",
        {"x-api-key": key_for()},
        body,
        0.0,
    )
    assert response.status == 200
    assert response.json_body is not None
    assert response.json_body["type"] == "message"
    assert len(gen.calls) == 1


def test_gemini_path_with_key_query() -> None:
    """Gemini generateContent resolves the session from the ?key= query parameter."""
    core, gen, _, _ = make_core()
    create_session(core)
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}).encode()
    response = core.handle(
        "POST",
        f"/v1beta/models/fs-model:generateContent?key={key_for()}",
        {},
        body,
        0.0,
    )
    assert response.status == 200
    assert response.json_body is not None
    assert "candidates" in response.json_body
    assert len(gen.calls) == 1


def test_streaming_returns_sse_frames() -> None:
    """Streaming chat completion returns SSE frames in the response."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(stream=True),
        0.0,
    )
    assert response.status == 200
    assert response.sse is not None
    assert len(response.sse) >= 1
    assert all(frame.startswith("data:") for frame in response.sse)
    assert response.headers.get("Content-Type") == "text/event-stream"


def test_train_mode_image_request_400_modality_not_declared() -> None:
    """A train-mode request with images is rejected with modality_not_declared."""
    core, gen, _, _ = make_core()
    create_session(core, mode="train")
    body = json.dumps(
        {
            "model": "fs-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                    ],
                }
            ],
        }
    ).encode()
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        body,
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "modality_not_declared"
    assert not gen.calls


def test_parsed_tool_call_finish_reason_tool_calls_and_arguments_json() -> None:
    """A parsed tool call yields finish_reason tool_calls and JSON arguments."""
    parsed = ParsedCall(name="lookup", arguments={"q": "x"}, raw="raw", error=None)
    core, _, _, _ = make_core(parser=FakeParser(calls=[parsed], content=""))
    create_session(core)
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert response.status == 200
    body = response.json_body
    assert body is not None
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["function"]["name"] == "lookup"
    assert json.loads(tc["function"]["arguments"]) == {"q": "x"}
    assert tc["id"].startswith("call_")


def test_parser_error_stays_content() -> None:
    """A ParsedCall with an error is not returned as a tool call and stays in content."""
    bad = ParsedCall(name="", arguments=None, raw="raw", error="unterminated tool_call")
    core, _, _, _ = make_core(parser=FakeParser(calls=[bad], content="leftover text"))
    create_session(core)
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert response.status == 200
    body = response.json_body
    assert body is not None
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    # No call parsed cleanly, so the full decoded text (incl. the bad markup) stays content.
    assert choice["message"]["content"] == "decoded-text"
    assert not choice["message"].get("tool_calls")


def test_engine_exception_502_and_nothing_recorded() -> None:
    """An engine exception returns 502 engine_error and records no call."""
    core, gen, _, _ = make_core(client=FakeGenerationClient(exc=RuntimeError("boom")))
    create_session(core)
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert response.status == 502
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "engine_error"
    assert not core.store.get(SESSION_ID).calls
    assert len(gen.calls) == 1


def test_inflight_limit_429_with_retry_after() -> None:
    """A second call while the only inflight slot is busy gets 429 with Retry-After: 1."""
    nested: list[GatewayResponse] = []

    class ReentrantClient(FakeGenerationClient):
        """Issues a nested gateway call from inside its own (slot-holding) engine call."""

        async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
            nested.append(
                core.handle(
                    "POST",
                    f"/sess/{SESSION_ID}/v1/chat/completions",
                    {"Authorization": f"Bearer {key_for()}"},
                    chat_body(),
                    0.0,
                )
            )
            return await super().generate(prompt_ids, sampling)

    core, _, _, _ = make_core(client=ReentrantClient(), config=make_config(max_inflight=1))
    create_session(core)
    outer = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert outer.status == 200
    (inner,) = nested
    assert inner.status == 429
    assert inner.headers.get("Retry-After") == "1"


def test_closed_session_409() -> None:
    """Model calls to a closed session are rejected with 409."""
    core, _, _, _ = make_core()
    create_session(core)
    core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/end",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"status": "ok", "termination": "stop"}).encode(),
        0.0,
    )
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert response.status == 409


def test_end_and_trajectory_returns_episode_v2() -> None:
    """Ending a session and requesting its trajectory returns an Episode JSON."""
    core, _, _, _ = make_core()
    create_session(core)
    core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    end = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/end",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"status": "ok", "termination": "stop"}).encode(),
        0.0,
    )
    assert end.status == 200
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/trajectory",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps(
            {
                "model": {"name": "fs-model", "version": 0, "checkpoint_sha": "abc123"},
                "harness": {"name": "fake-harness", "version": "0"},
                "env": {"kind": "fs_local", "task_id": "t1"},
            }
        ).encode(),
        0.0,
    )
    assert response.status == 200
    body = response.json_body
    assert body is not None
    assert body["schema_version"] == "fs.trajectory/v2"


def test_weights_begin_commit_bumps_version_seen_by_next_call() -> None:
    """Weights begin/commit bumps the policy version recorded by the next call."""
    core, _, _, _ = make_core()
    create_session(core)
    begin = core.handle(
        "POST", "/fs/v1/weights/begin", {"Authorization": f"Bearer {ADMIN_KEY}"}, b"", 0.0
    )
    assert begin.status == 200
    commit = core.handle(
        "POST",
        "/fs/v1/weights/commit",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"version": 3}).encode(),
        0.0,
    )
    assert commit.status == 200
    assert core.policy_version == 3
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert response.status == 200
    recorded = core.store.get(SESSION_ID).calls
    assert recorded and recorded[0].policy_version == 3


def test_weights_commit_non_increasing_version_400() -> None:
    """Committing a non-increasing weights version is rejected with 400."""
    core, _, _, _ = make_core()
    core.handle("POST", "/fs/v1/weights/begin", {"Authorization": f"Bearer {ADMIN_KEY}"}, b"", 0.0)
    core.handle(
        "POST",
        "/fs/v1/weights/commit",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"version": 5}).encode(),
        0.0,
    )
    core.handle("POST", "/fs/v1/weights/begin", {"Authorization": f"Bearer {ADMIN_KEY}"}, b"", 0.0)
    response = core.handle(
        "POST",
        "/fs/v1/weights/commit",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"version": 5}).encode(),
        0.0,
    )
    assert response.status == 400


def test_health_works_without_auth() -> None:
    """The health endpoint responds without any authorization header."""
    core, _, _, _ = make_core()
    response = core.handle("GET", "/fs/v1/health", {}, b"", 0.0)
    assert response.status == 200
    body = response.json_body
    assert body is not None
    assert body["ok"] is True
    assert body["policy_version"] == 0
    assert body["sessions"] == 0


def test_unknown_path_404() -> None:
    """An unknown path returns 404 not_found."""
    core, _, _, _ = make_core()
    response = core.handle("GET", "/nope", {}, b"", 0.0)
    assert response.status == 404
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "not_found"


def test_http_smoke_serve_chat_completion_and_shutdown() -> None:
    """Serve the core over real HTTP, complete one chat call, then shut down."""
    core, gen, _, _ = make_core()
    create_session(core)
    server = serve(core, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        request = urllib.request.Request(
            f"http://{host}:{port}/sess/{SESSION_ID}/v1/chat/completions",
            data=chat_body(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key_for()}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode())
        assert payload["object"] == "chat.completion"
        assert len(gen.calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _chat(core: GatewayCore, body: bytes | None = None) -> GatewayResponse:
    """One path-scoped chat completion on the default session."""
    return core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        body if body is not None else chat_body(),
        0.0,
    )


def test_missing_engine_logprobs_is_an_engine_error_and_nothing_is_recorded() -> None:
    """D3: logprobs are never filled in -- an engine reply without them is a 502, not a row."""
    no_lp = Generation(token_ids=(11, 12), logprobs=None, finish_reason="stop", text="x")
    core, _, _, _ = make_core(client=FakeGenerationClient(generation=no_lp))
    create_session(core)
    response = _chat(core)
    assert response.status == 502
    assert core.store.get(SESSION_ID).calls == []


def test_engine_abort_is_an_engine_error_not_a_clean_stop() -> None:
    """An aborted generation is not recorded as a finished sample."""
    aborted = Generation(token_ids=(11,), logprobs=(-0.5,), finish_reason="abort", text="x")
    core, _, _, _ = make_core(client=FakeGenerationClient(generation=aborted))
    create_session(core)
    assert _chat(core).status == 502
    assert core.store.get(SESSION_ID).calls == []


def test_valid_tool_call_markup_is_not_duplicated_into_content() -> None:
    """With a parsed call, content is the leftover prose plus only the raw of failed calls."""
    good = ParsedCall(
        name="get_weather", arguments={"city": "Paris"}, raw="<call>ok</call>", error=None
    )
    bad = ParsedCall(name="", arguments=None, raw="<call>broken", error="unterminated call")
    core, _, _, _ = make_core(parser=FakeParser(calls=[good, bad], content="Checking. "))
    create_session(core)
    message = _chat(core).json_body["choices"][0]["message"]
    assert message["content"] == "Checking. <call>broken"
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"


def test_recorded_sampling_requested_is_what_the_client_sent() -> None:
    """sampling_requested holds the client's values; the applied sampler differs and is audited."""
    core, _, _, _ = make_core()
    create_session(core)
    _chat(core, chat_body(temperature=1.3, top_p=0.5))
    (call,) = core.store.get(SESSION_ID).calls
    assert call.sampling_requested["temperature"] == 1.3
    assert call.sampling_requested["top_p"] == 0.5
    assert call.sampling_overrides["top_p"] == {"requested": 0.5, "applied": 1.0}


def test_weights_commit_without_begin_is_refused() -> None:
    """A commit with no open swap would change the version with no lock held: 409."""
    core, _, _, _ = make_core()
    response = core.handle(
        "POST",
        "/fs/v1/weights/commit",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"version": 5}).encode(),
        0.0,
    )
    assert response.status == 409


def test_create_session_non_object_body_400() -> None:
    """A non-object create-session body is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    response = core.handle(
        "POST", "/fs/v1/sessions", {"Authorization": f"Bearer {ADMIN_KEY}"}, b"[]", 0.0
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_create_session_bad_mode_400() -> None:
    """An unknown session mode is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    response = core.handle(
        "POST",
        "/fs/v1/sessions",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"mode": "bogus", "group_id": "g1"}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_create_session_empty_group_id_400() -> None:
    """An empty group_id is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    response = core.handle(
        "POST",
        "/fs/v1/sessions",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"mode": "train", "group_id": ""}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_create_session_bad_session_id_400() -> None:
    """A non-string session_id is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    response = core.handle(
        "POST",
        "/fs/v1/sessions",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"mode": "train", "group_id": "g1", "session_id": 5}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_create_session_store_full_503() -> None:
    """A full session store refuses creation with 503 record_store_full."""
    core, _, _, _ = make_core(config=make_config(max_sessions=1))
    create_session(core, session_id="sess-a")
    response = core.handle(
        "POST",
        "/fs/v1/sessions",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"mode": "train", "group_id": "g1", "session_id": "sess-b"}).encode(),
        0.0,
    )
    assert response.status == 503
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "record_store_full"


def test_create_session_duplicate_id_400() -> None:
    """A duplicate session id is refused by the store with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core, session_id="sess-dup")
    response = core.handle(
        "POST",
        "/fs/v1/sessions",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"mode": "train", "group_id": "g1", "session_id": "sess-dup"}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_add_reward_non_object_body_400() -> None:
    """A non-object reward body is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/reward",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        b"not-json",
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_add_reward_bad_event_400() -> None:
    """A reward payload that is not a valid RewardEvent is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/reward",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"value": "not-a-float"}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_add_reward_refused_400() -> None:
    """A reward the store refuses is reported as 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/reward",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"value": 1.0, "step": -1}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_end_session_non_object_body_400() -> None:
    """A non-object end-session body is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/end",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        b"null",
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_end_session_bad_status_400() -> None:
    """An unknown episode status is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/end",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"status": "bogus", "termination": "stop"}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_end_session_bad_termination_400() -> None:
    """An unknown termination kind is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/end",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"status": "ok", "termination": "bogus"}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_end_session_unknown_session_404() -> None:
    """Ending an unknown session returns 404 not_found."""
    core, _, _, _ = make_core()
    response = core.handle(
        "POST",
        "/fs/v1/sessions/ghost/end",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"status": "ok", "termination": "stop"}).encode(),
        0.0,
    )
    assert response.status == 404
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "not_found"


def test_end_session_twice_409() -> None:
    """Ending an already closed session returns 409 session_closed."""
    core, _, _, _ = make_core()
    create_session(core)
    body = json.dumps({"status": "ok", "termination": "stop"}).encode()
    headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
    first = core.handle("POST", f"/fs/v1/sessions/{SESSION_ID}/end", headers, body, 0.0)
    assert first.status == 200
    second = core.handle("POST", f"/fs/v1/sessions/{SESSION_ID}/end", headers, body, 0.0)
    assert second.status == 409
    assert second.json_body is not None
    assert second.json_body["error"]["type"] == "session_closed"


def test_end_session_refused_400() -> None:
    """A close the store refuses is reported as 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/end",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"status": "error", "termination": "stop"}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_trajectory_non_object_body_400() -> None:
    """A non-object trajectory body is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/trajectory",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        b"[]",
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_trajectory_bad_harness_meta_400() -> None:
    """An invalid HarnessMeta is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/trajectory",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps(
            {
                "model": {"name": "m", "version": 0, "checkpoint_sha": "abc"},
                "harness": {"name": "h"},
            }
        ).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_trajectory_bad_env_kind_400() -> None:
    """An unknown EnvKind value is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/trajectory",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps(
            {
                "model": {"name": "m", "version": 0, "checkpoint_sha": "abc"},
                "harness": {"name": "h", "version": "0"},
                "env": {"kind": "bogus", "task_id": "t1"},
            }
        ).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_trajectory_unknown_session_404() -> None:
    """A trajectory request for an unknown session returns 404 not_found."""
    core, _, _, _ = make_core()
    response = core.handle(
        "POST",
        "/fs/v1/sessions/ghost/trajectory",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps(
            {
                "model": {"name": "m", "version": 0, "checkpoint_sha": "abc"},
                "harness": {"name": "h", "version": "0"},
                "env": {"kind": "fs_local", "task_id": "t1"},
            }
        ).encode(),
        0.0,
    )
    assert response.status == 404
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "not_found"


def test_trajectory_build_refusal_400() -> None:
    """A build_episode refusal is reported as 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/trajectory",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps(
            {
                "model": {"name": "m", "version": 0, "checkpoint_sha": "abc"},
                "harness": {"name": "h", "version": "0"},
                "env": {"kind": "fs_local", "task_id": "t1"},
            }
        ).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_commit_weights_non_object_body_400() -> None:
    """A non-object weights commit body is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    core.handle("POST", "/fs/v1/weights/begin", {"Authorization": f"Bearer {ADMIN_KEY}"}, b"", 0.0)
    response = core.handle(
        "POST", "/fs/v1/weights/commit", {"Authorization": f"Bearer {ADMIN_KEY}"}, b"oops", 0.0
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_commit_weights_non_int_version_400() -> None:
    """A non-int weights version is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    core.handle("POST", "/fs/v1/weights/begin", {"Authorization": f"Bearer {ADMIN_KEY}"}, b"", 0.0)
    response = core.handle(
        "POST",
        "/fs/v1/weights/commit",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"version": "3"}).encode(),
        0.0,
    )
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_model_call_non_post_404() -> None:
    """A non-POST model call path is rejected with 404 not_found."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "GET",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        b"",
        0.0,
    )
    assert response.status == 404
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "not_found"


def test_model_call_non_object_body_400() -> None:
    """A non-object model call body is rejected with 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = _chat(core, b"not-json")
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_model_call_wire_refusal_400() -> None:
    """A wire-level parse refusal is reported as 400 invalid_request."""
    core, _, _, _ = make_core()
    create_session(core)
    response = _chat(core, json.dumps({"model": "fs-model"}).encode())
    assert response.status == 400
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "invalid_request"


def test_tokenizer_failure_502() -> None:
    """A tokenizer/template failure is reported as 502 engine_error."""

    class BoomTokenizer(FakeTokenizer):
        def render(self, messages, *, tools, add_generation_prompt):  # type: ignore[override]
            raise RuntimeError("template boom")

    core, _, _, _ = make_core(tokenizer=BoomTokenizer())
    create_session(core)
    response = _chat(core)
    assert response.status == 502
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "engine_error"


def test_record_call_store_full_503() -> None:
    """A call that exceeds the per-session call cap is refused with 503 record_store_full."""
    core, _, _, _ = make_core(config=make_config(max_calls_per_session=1))
    create_session(core)
    assert _chat(core).status == 200
    response = _chat(core)
    assert response.status == 503
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "record_store_full"


def test_record_call_on_closed_session_409() -> None:
    """Recording a call on a session closed mid-flight is reported as 409 session_closed."""
    closed = threading.Event()

    class ClosingClient(FakeGenerationClient):
        async def generate(self, prompt_ids, sampling):  # type: ignore[override]
            core.handle(
                "POST",
                f"/fs/v1/sessions/{SESSION_ID}/end",
                {"Authorization": f"Bearer {ADMIN_KEY}"},
                json.dumps({"status": "ok", "termination": "stop"}).encode(),
                0.0,
            )
            closed.set()
            return await super().generate(prompt_ids, sampling)

    core, _, _, _ = make_core(client=ClosingClient())
    create_session(core)
    response = _chat(core)
    assert closed.is_set()
    assert response.status == 409
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "session_closed"


def test_session_scoped_unknown_session_404() -> None:
    """A path-scoped call for an unknown session returns 404 not_found."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        "/sess/ghost/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert response.status == 404
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "not_found"


def test_session_scoped_wrong_key_401() -> None:
    """A path-scoped call with the wrong api key returns 401 unauthorized."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": "Bearer wrong-key"},
        chat_body(),
        0.0,
    )
    assert response.status == 401
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "unauthorized"


def test_session_scoped_closed_session_409() -> None:
    """A path-scoped call to a closed session returns 409 session_closed."""
    core, _, _, _ = make_core()
    create_session(core)
    core.handle(
        "POST",
        f"/fs/v1/sessions/{SESSION_ID}/end",
        {"Authorization": f"Bearer {ADMIN_KEY}"},
        json.dumps({"status": "ok", "termination": "stop"}).encode(),
        0.0,
    )
    response = core.handle(
        "POST",
        f"/sess/{SESSION_ID}/v1/chat/completions",
        {"Authorization": f"Bearer {key_for()}"},
        chat_body(),
        0.0,
    )
    assert response.status == 409
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "session_closed"


def test_key_scoped_missing_key_401() -> None:
    """A key-scoped call without any api key returns 401 unauthorized."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle("POST", "/v1/chat/completions", {}, chat_body(), 0.0)
    assert response.status == 401
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "unauthorized"


def test_key_scoped_unknown_key_404() -> None:
    """A key-scoped call with an unknown api key returns 404 not_found."""
    core, _, _, _ = make_core()
    create_session(core)
    response = core.handle(
        "POST", "/v1/chat/completions", {"Authorization": "Bearer ghost-key"}, chat_body(), 0.0
    )
    assert response.status == 404
    assert response.json_body is not None
    assert response.json_body["error"]["type"] == "not_found"


def test_load_json_rejects_non_object_and_bad_utf8() -> None:
    """_load_json returns None for non-objects and undecodable bodies."""
    core, _, _, _ = make_core()
    assert core._load_json(b"") is None
    assert core._load_json(b"[1, 2]") is None
    assert core._load_json(b"\xff\xfe") is None
    assert core._load_json(b"{oops") is None
    assert core._load_json(b"{}") == {}


def test_handler_bad_content_length_reads_no_body() -> None:
    """A non-integer Content-Length is treated as zero and the request still routes."""
    core, _, _, _ = make_core()
    server = serve(core, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        request = urllib.request.Request(
            f"http://{host}:{port}/fs/v1/health",
            headers={"Content-Length": "not-a-number"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode())
        assert payload["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_handler_writes_sse_payload() -> None:
    """The HTTP handler writes SSE frames as the response payload."""
    core, _, _, _ = make_core()
    create_session(core)
    server = serve(core, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        request = urllib.request.Request(
            f"http://{host}:{port}/sess/{SESSION_ID}/v1/chat/completions",
            data=chat_body(stream=True),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key_for()}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode()
        assert raw.startswith("data:")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_handler_log_message_is_silent() -> None:
    """The handler's log_message never logs request bodies or keys."""
    core, _, _, _ = make_core()
    server = serve(core, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        request = urllib.request.Request(
            f"http://{host}:{port}/fs/v1/health", headers={}, method="GET"
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_valid_reward_event_is_accepted_and_reaches_the_episode() -> None:
    """A JSON reward with scope given as its string value is stored as a RewardScope event."""
    core, _, _, _ = make_core()
    create_session(core)
    assert _chat(core).status == 200
    reward = {
        "source": "verifier",
        "verifier_id": "unit",
        "verifier_version": "1",
        "value": 1.0,
        "scope": "episode",
    }
    admin = {"Authorization": f"Bearer {ADMIN_KEY}"}
    r = core.handle(
        "POST", f"/fs/v1/sessions/{SESSION_ID}/reward", admin, json.dumps(reward).encode(), 0.0
    )
    assert r.status == 200, r.json_body
    end = {"status": "ok", "termination": "stop"}
    assert (
        core.handle(
            "POST", f"/fs/v1/sessions/{SESSION_ID}/end", admin, json.dumps(end).encode(), 0.0
        ).status
        == 200
    )
    meta = {
        "model": {"name": "fs-model", "version": 0, "checkpoint_sha": "abc123"},
        "harness": {"name": "fake-harness", "version": "0"},
        "env": {"kind": "fs_local", "task_id": "t1"},
    }
    episode = core.handle(
        "POST", f"/fs/v1/sessions/{SESSION_ID}/trajectory", admin, json.dumps(meta).encode(), 0.0
    ).json_body
    assert episode["reward_events"][0]["scope"] == "episode"
    assert episode["reward_events"][0]["value"] == 1.0


def test_serve_replaces_a_port_zero_placeholder_with_the_bound_port() -> None:
    """Session base_urls point at the live port, never at the port-0 placeholder."""
    core, _, _, _ = make_core()
    server = serve(core, "127.0.0.1", 0)
    try:
        port = server.server_address[1]
        body = create_session(core)
        assert body["base_urls"]["chat_completions"].startswith(f"http://127.0.0.1:{port}/")
    finally:
        server.server_close()


def _two_turns(core: GatewayCore, assistant_echo: str) -> tuple[Any, Any]:
    """Turn 1, then turn 2 echoing the assistant reply as ``assistant_echo`` plus a new user msg."""
    create_session(core)
    first = [{"role": "user", "content": "hi"}]
    assert _chat(core, chat_body(first)).status == 200
    second = [
        *first,
        {"role": "assistant", "content": assistant_echo},
        {"role": "user", "content": "more"},
    ]
    assert _chat(core, chat_body(second)).status == 200
    calls = core.store.get(SESSION_ID).calls
    return calls[0], calls[1]


def test_echoed_history_continues_on_the_cached_sampled_tokens() -> None:
    """Turn 2's prompt is turn-1 prompt + sampled output + only the new message's delta."""
    core, _, _, _ = make_core()
    one, two = _two_turns(core, "decoded-text")
    prefix = one.prompt_ids + one.output_ids
    assert two.prompt_ids[: len(prefix)] == prefix
    assert len(two.prompt_ids) > len(prefix)


def test_rewritten_history_falls_back_to_a_full_render() -> None:
    """If the client edits the assistant turn, the gateway renders afresh (a new segment)."""
    core, _, _, _ = make_core()
    one, two = _two_turns(core, "something the model never said")
    prefix = one.prompt_ids + one.output_ids
    assert two.prompt_ids[: len(prefix)] != prefix


def test_reasoning_is_split_out_and_an_echo_without_it_still_continues() -> None:
    """<think> goes to reasoning_content; a client that drops it from history still continues."""
    core, _, _, _ = make_core(decode=lambda ids: "<think>plan it</think>Answer")
    create_session(core)
    first = [{"role": "user", "content": "hi"}]
    message = _chat(core, chat_body(first)).json_body["choices"][0]["message"]
    assert message["content"] == "Answer"
    assert message["reasoning_content"] == "plan it"
    second = [*first, {"role": "assistant", "content": "Answer"}, {"role": "user", "content": "go"}]
    assert _chat(core, chat_body(second)).status == 200
    one, two = core.store.get(SESSION_ID).calls
    assert (
        two.prompt_ids[: len(one.prompt_ids) + len(one.output_ids)]
        == one.prompt_ids + one.output_ids
    )


def test_end_of_turn_marker_never_reaches_the_client() -> None:
    """A decoded trailing <|im_end|> is stripped from content (the sampled ids keep it)."""
    core, _, _, _ = make_core(decode=lambda ids: "Answer<|im_end|>")
    create_session(core)
    message = _chat(core).json_body["choices"][0]["message"]
    assert message["content"] == "Answer"
    (call,) = core.store.get(SESSION_ID).calls
    assert call.output_ids == (11, 12, 13)
