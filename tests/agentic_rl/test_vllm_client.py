"""Tests for ``engines.vllm.VLLMClient`` against an in-process fake vLLM
OpenAI-compatible HTTP server (a background thread, 127.0.0.1, ephemeral port).

WHAT IS CLAIMED: the request body ``generate()`` sends matches the module's
documented UNPROBED ASSUMPTIONS (model, prompt, max_tokens, sampling fields,
top_k==0 -> -1, return_tokens_as_token_ids/return_token_ids/skip_special_tokens);
a well-formed response parses into a ``Generation`` with the exact token ids,
logprobs and finish reason, via EITHER the ``choices[0].token_ids`` path or the
``choices[0].logprobs.tokens`` ``"token_id:<int>"`` path; every shape violation
raises ``EngineInfraError(kind="bad_response")`` naming the problem; a non-2xx
status raises ``kind="http_<code>"``; a connection refusal raises
``kind="transport"``; a slow handler past ``timeout_s`` raises ``kind="timeout"``;
``health()`` never raises; ``reload_weights`` refuses without
``reload_mode="collective_rpc"`` and posts to ``/collective_rpc`` with it;
``update_weights_from_disk`` is a thin alias of ``reload_weights``; ``sleep``/
``wake_up`` reach their documented endpoints without raising on a 200.

WHAT IS NOT CLAIMED: no real vLLM server is involved, and no test imports the
``vllm`` package (not installed in this environment, and this plane must never
import it).
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from foundationscale.agentic_rl.engines import EngineInfraError, EngineRefusal
from foundationscale.agentic_rl.engines.vllm import VLLMClient
from foundationscale.agentic_rl.harness.base import Generation, SamplingParams


def _sampling(**overrides: Any) -> SamplingParams:
    fields: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 8,
    }
    fields.update(overrides)
    return SamplingParams(**fields)


def _default_completions_response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "text": "hello world",
                "token_ids": [101, 102, 103],
                "logprobs": {"token_logprobs": [-0.1, -0.2, -0.3]},
                "finish_reason": "stop",
            }
        ]
    }


class _Server(ThreadingHTTPServer):
    last_completions_body: dict[str, Any] | None = None
    last_collective_rpc_body: dict[str, Any] | None = None
    last_sleep_path: str | None = None
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    script: dict[str, Any] = {}

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_raw(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_redirect(self, status: int, location: str) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's own naming
        if self.path == "/health":
            # Real vLLM answers /health with 200 and an EMPTY body (probed
            # 2026-10-08); a JSON body here once hid a client that rejected it.
            status = self.script.get("health_status", 200)
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's own naming
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        body: Any = None
        if raw:
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                body = None
        assert isinstance(self.server, _Server)
        if self.path == "/v1/completions":
            sleep_s = self.script.get("completions_sleep_s")
            if sleep_s:
                time.sleep(sleep_s)
            self.server.last_completions_body = body
            if "completions_redirect_to" in self.script:
                self._send_redirect(
                    self.script.get("completions_status", 307),
                    self.script["completions_redirect_to"],
                )
                return
            if "completions_raw_body" in self.script:
                self._send_raw(
                    self.script.get("completions_status", 200), self.script["completions_raw_body"]
                )
                return
            response = self.script.get("completions_response", _default_completions_response())
            self._send_json(self.script.get("completions_status", 200), response)
            return
        if self.path == "/collective_rpc":
            self.server.last_collective_rpc_body = body
            self._send_json(self.script.get("collective_rpc_status", 200), {"result": [True]})
            return
        if self.path.startswith("/sleep"):
            self.server.last_sleep_path = self.path
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/wake_up":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass


@pytest.fixture
def server() -> Generator[_Server, None, None]:
    srv = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        thread.join(timeout=5)
        srv.server_close()


@pytest.fixture(autouse=True)
def _reset_script() -> Generator[None, None, None]:
    _Handler.script = {}
    yield
    _Handler.script = {}


def _client(server: _Server, *, timeout_s: float = 5.0, **overrides: Any) -> VLLMClient:
    fields: dict[str, Any] = {
        "base_url": f"http://127.0.0.1:{server.server_port}",
        "timeout_s": timeout_s,
        "served_model_name": "served-model",
    }
    fields.update(overrides)
    return VLLMClient(**fields)


def _closed_port_client(*, timeout_s: float = 2.0) -> VLLMClient:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return VLLMClient(
        base_url=f"http://127.0.0.1:{port}", timeout_s=timeout_s, served_model_name="m"
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_client_strips_trailing_slash_from_base_url() -> None:
    client = VLLMClient(base_url="http://127.0.0.1:9999/", timeout_s=1.0, served_model_name="m")
    assert client.base_url == "http://127.0.0.1:9999"


@pytest.mark.parametrize("base_url", ["", 5])
def test_client_rejects_bad_base_url(base_url: Any) -> None:
    with pytest.raises(EngineRefusal):
        VLLMClient(base_url=base_url, timeout_s=1.0, served_model_name="m")


@pytest.mark.parametrize(
    "base_url",
    ["file:///etc/passwd", "ftp://example.com", "127.0.0.1:9999", "not a url", "http://"],
)
def test_client_rejects_non_http_base_url(base_url: str) -> None:
    with pytest.raises(EngineRefusal):
        VLLMClient(base_url=base_url, timeout_s=1.0, served_model_name="m")


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, True, float("inf"), float("nan"), "1"])
def test_client_rejects_bad_timeout(timeout_s: Any) -> None:
    with pytest.raises(EngineRefusal):
        VLLMClient(base_url="http://127.0.0.1:9999", timeout_s=timeout_s, served_model_name="m")


@pytest.mark.parametrize("served_model_name", ["", 5, None])
def test_client_rejects_bad_served_model_name(served_model_name: Any) -> None:
    with pytest.raises(EngineRefusal):
        VLLMClient(
            base_url="http://127.0.0.1:9999", timeout_s=1.0, served_model_name=served_model_name
        )


def test_client_rejects_an_undeclared_reload_mode() -> None:
    with pytest.raises(EngineRefusal):
        VLLMClient(
            base_url="http://127.0.0.1:9999",
            timeout_s=1.0,
            served_model_name="m",
            reload_mode="rpc_v2",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# generate(): happy path and request shape
# ---------------------------------------------------------------------------


def test_generate_happy_path_parses_response_via_token_ids_field(server: _Server) -> None:
    client = _client(server)
    generation = asyncio.run(client.generate((10, 20), _sampling()))
    assert isinstance(generation, Generation)
    assert generation.token_ids == (101, 102, 103)
    assert generation.logprobs == (-0.1, -0.2, -0.3)
    assert generation.finish_reason == "stop"
    assert generation.text == "hello world"


def test_generate_falls_back_to_logprobs_tokens_when_token_ids_absent(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [
            {
                "text": "hi",
                "logprobs": {
                    "tokens": ["token_id:5", "token_id:6"],
                    "token_logprobs": [-0.5, -0.6],
                },
                "finish_reason": "length",
            }
        ]
    }
    client = _client(server)
    generation = asyncio.run(client.generate((1,), _sampling()))
    assert generation.token_ids == (5, 6)
    assert generation.logprobs == (-0.5, -0.6)
    assert generation.finish_reason == "length"


def test_generate_sends_documented_request_shape(server: _Server) -> None:
    client = _client(server)
    asyncio.run(client.generate((7, 8, 9), _sampling(top_p=0.9, max_new_tokens=16)))
    body = server.last_completions_body
    assert body is not None
    assert body["model"] == "served-model"
    assert body["prompt"] == [7, 8, 9]
    assert body["max_tokens"] == 16
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.9
    assert body["min_p"] == 0.0
    assert body["presence_penalty"] == 0.0
    assert body["repetition_penalty"] == 1.0
    assert body["logprobs"] == 1
    assert body["return_tokens_as_token_ids"] is True
    assert body["return_token_ids"] is True
    assert body["skip_special_tokens"] is False


def test_generate_translates_top_k_zero_to_minus_one(server: _Server) -> None:
    client = _client(server)
    asyncio.run(client.generate((1,), _sampling(top_k=0)))
    assert server.last_completions_body is not None
    assert server.last_completions_body["top_k"] == -1


def test_generate_passes_nonzero_top_k_through_unchanged(server: _Server) -> None:
    client = _client(server)
    asyncio.run(client.generate((1,), _sampling(top_k=40)))
    assert server.last_completions_body is not None
    assert server.last_completions_body["top_k"] == 40


# ---------------------------------------------------------------------------
# generate(): bad_response shape violations
# ---------------------------------------------------------------------------


def test_generate_missing_choices_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_response"] = {}
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_empty_choices_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_response"] = {"choices": []}
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_non_dict_choice_entry_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_response"] = {"choices": ["not a dict"]}
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_no_token_ids_and_no_logprobs_dict_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_response"] = {"choices": [{"text": "x", "finish_reason": "stop"}]}
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"
    assert "logprobs.tokens" in str(excinfo.value)


def test_generate_logprobs_tokens_not_a_list_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [{"text": "x", "logprobs": {"tokens": "nope"}, "finish_reason": "stop"}]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_token_ids_present_but_logprobs_missing_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [{"text": "x", "token_ids": [1], "finish_reason": "stop"}]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"
    assert "choices[0].logprobs" in str(excinfo.value)


def test_generate_token_logprobs_missing_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [{"text": "x", "token_ids": [1], "logprobs": {}, "finish_reason": "stop"}]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"
    assert "token_logprobs" in str(excinfo.value)


def test_generate_bool_token_id_is_refused_even_if_numerically_equal(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [
            {
                "text": "x",
                "token_ids": [True],
                "logprobs": {"token_logprobs": [-0.1]},
                "finish_reason": "stop",
            }
        ]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_malformed_token_id_entry_in_logprobs_tokens(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [
            {
                "text": "x",
                "logprobs": {"tokens": ["not-prefixed"], "token_logprobs": [-0.1]},
                "finish_reason": "stop",
            }
        ]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_non_int_after_token_id_prefix_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [
            {
                "text": "x",
                "logprobs": {"tokens": ["token_id:not-an-int"], "token_logprobs": [-0.1]},
                "finish_reason": "stop",
            }
        ]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_length_mismatch_names_both_counts(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [
            {
                "text": "x",
                "token_ids": [1, 2, 3],
                "logprobs": {"token_logprobs": [-0.1, -0.2]},
                "finish_reason": "stop",
            }
        ]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"
    assert "3" in str(excinfo.value)
    assert "2" in str(excinfo.value)


def test_generate_bool_logprob_is_refused(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [
            {
                "text": "x",
                "token_ids": [1],
                "logprobs": {"token_logprobs": [True]},
                "finish_reason": "stop",
            }
        ]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_non_finite_logprob_is_refused(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [
            {
                "text": "x",
                "token_ids": [1],
                "logprobs": {"token_logprobs": [float("nan")]},
                "finish_reason": "stop",
            }
        ]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


@pytest.mark.parametrize("finish_reason", ["abort", None, "content_filter", "bogus"])
def test_generate_unrecognised_finish_reason_becomes_abort(
    server: _Server, finish_reason: Any
) -> None:
    _Handler.script["completions_response"] = {
        "choices": [
            {
                "text": "x",
                "token_ids": [1],
                "logprobs": {"token_logprobs": [-0.1]},
                "finish_reason": finish_reason,
            }
        ]
    }
    client = _client(server)
    generation = asyncio.run(client.generate((1,), _sampling()))
    assert generation.finish_reason == "abort"


def test_generate_missing_text_field(server: _Server) -> None:
    _Handler.script["completions_response"] = {
        "choices": [
            {"token_ids": [1], "logprobs": {"token_logprobs": [-0.1]}, "finish_reason": "stop"}
        ]
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_non_json_body_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_raw_body"] = b"not json at all"
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_non_dict_body_is_bad_response(server: _Server) -> None:
    _Handler.script["completions_raw_body"] = b"[1, 2, 3]"
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


# ---------------------------------------------------------------------------
# generate(): transport-level failures
# ---------------------------------------------------------------------------


def test_generate_http_error_status_names_the_code(server: _Server) -> None:
    _Handler.script["completions_status"] = 500
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "http_500"


def test_generate_connection_refused_is_transport() -> None:
    client = _closed_port_client(timeout_s=2.0)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "transport"


def test_generate_slow_server_times_out(server: _Server) -> None:
    _Handler.script["completions_sleep_s"] = 0.5
    client = _client(server, timeout_s=0.05)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "timeout"


def test_generate_refuses_a_3xx_redirect_instead_of_following_it(server: _Server) -> None:
    # Security regression: urllib follows 3xx redirects by default, which could
    # resend the prompt to a different host (or a non-http scheme) the caller
    # never declared. "http://127.0.0.1:1/evil" is unreachable (port 1 is a
    # privileged port nothing here listens on) -- if the client ever followed
    # it, this would raise kind="transport" (or time out), not kind="http_307";
    # it must never even attempt to connect there.
    _Handler.script["completions_status"] = 307
    _Handler.script["completions_redirect_to"] = "http://127.0.0.1:1/evil"
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "http_307"


# ---------------------------------------------------------------------------
# Control calls
# ---------------------------------------------------------------------------


def test_health_true_on_200(server: _Server) -> None:
    assert _client(server).health() is True


def test_health_false_on_non_200(server: _Server) -> None:
    _Handler.script["health_status"] = 503
    assert _client(server).health() is False


def test_health_false_on_connection_refused() -> None:
    assert _closed_port_client(timeout_s=1.0).health() is False


def test_reload_weights_refuses_without_an_explicit_reload_mode(server: _Server) -> None:
    client = _client(server)
    with pytest.raises(EngineRefusal, match="reload_mode"):
        client.reload_weights("/tmp/ckpt")


def test_reload_weights_posts_collective_rpc_when_declared(server: _Server) -> None:
    client = _client(server, reload_mode="collective_rpc")
    client.reload_weights("/tmp/ckpt/step_1")
    assert server.last_collective_rpc_body == {
        "method": "reload_weights",
        "kwargs": {"weights_path": "/tmp/ckpt/step_1"},
    }


def test_reload_weights_rejects_empty_path(server: _Server) -> None:
    client = _client(server, reload_mode="collective_rpc")
    with pytest.raises(EngineRefusal):
        client.reload_weights("")


def test_update_weights_from_disk_is_an_alias_of_reload_weights(server: _Server) -> None:
    client = _client(server, reload_mode="collective_rpc")
    client.update_weights_from_disk("/tmp/ckpt/step_2")
    assert server.last_collective_rpc_body == {
        "method": "reload_weights",
        "kwargs": {"weights_path": "/tmp/ckpt/step_2"},
    }


def test_sleep_posts_the_level_as_a_query_param(server: _Server) -> None:
    client = _client(server)
    client.sleep(1)
    assert server.last_sleep_path == "/sleep?level=1"


@pytest.mark.parametrize("level", [-1, True, 1.5, "1"])
def test_sleep_rejects_a_bad_level(server: _Server, level: Any) -> None:
    client = _client(server)
    with pytest.raises(EngineRefusal):
        client.sleep(level)


def test_wake_up_does_not_raise_on_200(server: _Server) -> None:
    _client(server).wake_up()


def test_health_accepts_the_real_servers_empty_200_body(server: _Server) -> None:
    # Regression from the first GPU smoke run: 983 health probes got HTTP 200
    # with an empty body and every one was scored unhealthy.
    client = _client(server)
    assert client.health() is True
