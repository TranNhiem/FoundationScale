"""Tests for ``engines.sglang.SGLangClient`` against an in-process fake SGLang
native HTTP server (a background thread, 127.0.0.1, ephemeral port).

WHAT IS CLAIMED: the request body SGLangClient.generate() sends matches the
module's documented ASSUMPTIONS (input_ids, sampling_params field names,
top_k==0 -> -1, return_logprob=true); a well-formed response parses into a
Generation with the exact token ids, logprobs and finish reason; every shape
violation (length mismatch, token id mismatch, unknown finish type, non-JSON
body, non-dict body, missing text) raises EngineInfraError(kind="bad_response")
naming the problem; a non-2xx status raises kind="http_<code>"; a connection
refusal raises kind="transport"; a slow handler past the client's timeout_s
raises kind="timeout"; health() never raises; update_weights_from_disk()
raises on a false "success" field; flush_cache/release_memory/resume_memory
reach their documented endpoints without raising on a 200.

WHAT IS NOT CLAIMED: no real SGLang server is involved, and no test imports
the ``sglang`` package (not installed in this environment, and this plane
must never import it).
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
from foundationscale.agentic_rl.engines.sglang import SGLangClient
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


def _default_generate_response() -> dict[str, Any]:
    return {
        "text": "hello world",
        "output_ids": [101, 102, 103],
        "meta_info": {
            "output_token_logprobs": [[-0.1, 101], [-0.2, 102], [-0.3, 103]],
            "finish_reason": {"type": "stop"},
        },
    }


class _Server(ThreadingHTTPServer):
    last_generate_body: dict[str, Any] | None = None
    last_update_body: dict[str, Any] | None = None
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

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's own naming
        if self.path == "/health":
            self._send_json(self.script.get("health_status", 200), {"status": "ok"})
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
        if self.path == "/generate":
            sleep_s = self.script.get("generate_sleep_s")
            if sleep_s:
                time.sleep(sleep_s)
            assert isinstance(self.server, _Server)
            self.server.last_generate_body = body
            if "generate_raw_body" in self.script:
                self._send_raw(
                    self.script.get("generate_status", 200), self.script["generate_raw_body"]
                )
                return
            response = self.script.get("generate_response", _default_generate_response())
            self._send_json(self.script.get("generate_status", 200), response)
            return
        if self.path == "/update_weights_from_disk":
            assert isinstance(self.server, _Server)
            self.server.last_update_body = body
            response = self.script.get("update_response", {"success": True})
            self._send_json(self.script.get("update_status", 200), response)
            return
        if self.path in ("/flush_cache", "/release_memory_occupation", "/resume_memory_occupation"):
            self._send_json(200, {"success": True})
            return
        self._send_json(404, {"error": "not found"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass


@pytest.fixture
def server() -> Generator[_Server, None, None]:
    srv = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
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


def _client(server: _Server, *, timeout_s: float = 5.0) -> SGLangClient:
    return SGLangClient(base_url=f"http://127.0.0.1:{server.server_port}", timeout_s=timeout_s)


def _closed_port_client(*, timeout_s: float = 2.0) -> SGLangClient:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return SGLangClient(base_url=f"http://127.0.0.1:{port}", timeout_s=timeout_s)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_client_strips_trailing_slash_from_base_url() -> None:
    client = SGLangClient(base_url="http://127.0.0.1:9999/", timeout_s=1.0)
    assert client.base_url == "http://127.0.0.1:9999"


@pytest.mark.parametrize("base_url", ["", 5])
def test_client_rejects_bad_base_url(base_url: Any) -> None:
    with pytest.raises(EngineRefusal):
        SGLangClient(base_url=base_url, timeout_s=1.0)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, True, float("inf"), float("nan"), "1"])
def test_client_rejects_bad_timeout(timeout_s: Any) -> None:
    with pytest.raises(EngineRefusal):
        SGLangClient(base_url="http://127.0.0.1:9999", timeout_s=timeout_s)


@pytest.mark.parametrize(
    "base_url",
    ["file:///etc/passwd", "ftp://example.com", "127.0.0.1:9999", "not a url", "http://"],
)
def test_client_rejects_non_http_base_url(base_url: str) -> None:
    # Security regression: only http(s) URLs with a host may be used -- anything
    # else (e.g. file://) would let _request() reach something that is not an
    # HTTP server, such as a local file.
    with pytest.raises(EngineRefusal):
        SGLangClient(base_url=base_url, timeout_s=1.0)


def test_client_accepts_https_base_url() -> None:
    client = SGLangClient(base_url="https://example.com:1234/", timeout_s=1.0)
    assert client.base_url == "https://example.com:1234"


# ---------------------------------------------------------------------------
# generate(): happy path and request shape
# ---------------------------------------------------------------------------


def test_generate_happy_path_parses_response(server: _Server) -> None:
    client = _client(server)
    generation = asyncio.run(client.generate((10, 20), _sampling()))
    assert isinstance(generation, Generation)
    assert generation.token_ids == (101, 102, 103)
    assert generation.logprobs == (-0.1, -0.2, -0.3)
    assert generation.finish_reason == "stop"
    assert generation.text == "hello world"


def test_generate_sends_documented_request_shape(server: _Server) -> None:
    client = _client(server)
    asyncio.run(client.generate((7, 8, 9), _sampling(top_p=0.9, max_new_tokens=16)))
    body = server.last_generate_body
    assert body is not None
    assert body["input_ids"] == [7, 8, 9]
    assert body["return_logprob"] is True
    params = body["sampling_params"]
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.9
    assert params["min_p"] == 0.0
    assert params["presence_penalty"] == 0.0
    assert params["repetition_penalty"] == 1.0
    assert params["max_new_tokens"] == 16


def test_generate_translates_top_k_zero_to_minus_one(server: _Server) -> None:
    client = _client(server)
    asyncio.run(client.generate((1,), _sampling(top_k=0)))
    assert server.last_generate_body is not None
    assert server.last_generate_body["sampling_params"]["top_k"] == -1


def test_generate_passes_nonzero_top_k_through_unchanged(server: _Server) -> None:
    client = _client(server)
    asyncio.run(client.generate((1,), _sampling(top_k=40)))
    assert server.last_generate_body is not None
    assert server.last_generate_body["sampling_params"]["top_k"] == 40


# ---------------------------------------------------------------------------
# generate(): bad_response shape violations
# ---------------------------------------------------------------------------


def test_generate_length_mismatch_names_both_counts(server: _Server) -> None:
    _Handler.script["generate_response"] = {
        "text": "x",
        "output_ids": [1, 2, 3],
        "meta_info": {
            "output_token_logprobs": [[-0.1, 1], [-0.2, 2]],
            "finish_reason": {"type": "stop"},
        },
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"
    assert "3" in str(excinfo.value)
    assert "2" in str(excinfo.value)


def test_generate_token_id_mismatch_names_both_ids(server: _Server) -> None:
    _Handler.script["generate_response"] = {
        "text": "x",
        "output_ids": [1, 2],
        "meta_info": {
            "output_token_logprobs": [[-0.1, 1], [-0.2, 999]],
            "finish_reason": {"type": "stop"},
        },
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"
    assert "999" in str(excinfo.value)
    assert "2" in str(excinfo.value)


def test_generate_bool_token_id_is_refused_even_if_numerically_equal(server: _Server) -> None:
    # Regression: `True != 1` is False in Python, so a loose `!=` comparison let a
    # bool token id through when it happened to equal the output id numerically.
    _Handler.script["generate_response"] = {
        "text": "x",
        "output_ids": [1],
        "meta_info": {"output_token_logprobs": [[-0.1, True]], "finish_reason": {"type": "stop"}},
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_float_token_id_is_refused_even_if_numerically_equal(server: _Server) -> None:
    # Regression: `1.0 != 1` is also False in Python -- a float token id must be
    # refused too; only type(x) is int is accepted.
    _Handler.script["generate_response"] = {
        "text": "x",
        "output_ids": [1],
        "meta_info": {"output_token_logprobs": [[-0.1, 1.0]], "finish_reason": {"type": "stop"}},
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_non_finite_logprob_is_refused(server: _Server) -> None:
    _Handler.script["generate_response"] = {
        "text": "x",
        "output_ids": [1],
        "meta_info": {
            "output_token_logprobs": [[float("nan"), 1]],
            "finish_reason": {"type": "stop"},
        },
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_bool_logprob_is_refused(server: _Server) -> None:
    _Handler.script["generate_response"] = {
        "text": "x",
        "output_ids": [1],
        "meta_info": {"output_token_logprobs": [[True, 1]], "finish_reason": {"type": "stop"}},
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_malformed_logprob_entry(server: _Server) -> None:
    _Handler.script["generate_response"] = {
        "text": "x",
        "output_ids": [1],
        "meta_info": {
            "output_token_logprobs": [["not-a-list-entry"]],
            "finish_reason": {"type": "stop"},
        },
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_unknown_finish_reason_type(server: _Server) -> None:
    _Handler.script["generate_response"] = {
        "text": "x",
        "output_ids": [1],
        "meta_info": {"output_token_logprobs": [[-0.1, 1]], "finish_reason": {"type": "bogus"}},
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_missing_text_field(server: _Server) -> None:
    _Handler.script["generate_response"] = {
        "output_ids": [1],
        "meta_info": {"output_token_logprobs": [[-0.1, 1]], "finish_reason": {"type": "stop"}},
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_missing_output_ids_field(server: _Server) -> None:
    _Handler.script["generate_response"] = {
        "text": "x",
        "meta_info": {"output_token_logprobs": [], "finish_reason": {"type": "stop"}},
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_missing_meta_info_field(server: _Server) -> None:
    _Handler.script["generate_response"] = {"text": "x", "output_ids": [1]}
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_missing_output_token_logprobs_field(server: _Server) -> None:
    _Handler.script["generate_response"] = {
        "text": "x",
        "output_ids": [1],
        "meta_info": {"finish_reason": {"type": "stop"}},
    }
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_non_200_success_status_is_http_kind(server: _Server) -> None:
    _Handler.script["generate_status"] = 201
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "http_201"


def test_generate_non_json_body_is_bad_response(server: _Server) -> None:
    _Handler.script["generate_raw_body"] = b"not json at all"
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


def test_generate_non_dict_body_is_bad_response(server: _Server) -> None:
    _Handler.script["generate_raw_body"] = b"[1, 2, 3]"
    client = _client(server)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "bad_response"


# ---------------------------------------------------------------------------
# generate(): transport-level failures
# ---------------------------------------------------------------------------


def test_generate_http_error_status_names_the_code(server: _Server) -> None:
    _Handler.script["generate_status"] = 500
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
    _Handler.script["generate_sleep_s"] = 0.5
    client = _client(server, timeout_s=0.05)
    with pytest.raises(EngineInfraError) as excinfo:
        asyncio.run(client.generate((1,), _sampling()))
    assert excinfo.value.kind == "timeout"


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


def test_update_weights_from_disk_success_posts_model_path(server: _Server) -> None:
    _client(server).update_weights_from_disk("/tmp/ckpt/step_000001")
    assert server.last_update_body == {"model_path": "/tmp/ckpt/step_000001"}


def test_update_weights_from_disk_false_success_raises(server: _Server) -> None:
    _Handler.script["update_response"] = {"success": False, "message": "boom"}
    with pytest.raises(EngineInfraError) as excinfo:
        _client(server).update_weights_from_disk("/tmp/ckpt")
    assert excinfo.value.kind == "update_failed"
    assert "boom" in str(excinfo.value)


def test_update_weights_from_disk_bad_success_type_is_bad_response(server: _Server) -> None:
    _Handler.script["update_response"] = {"success": "yes"}
    with pytest.raises(EngineInfraError) as excinfo:
        _client(server).update_weights_from_disk("/tmp/ckpt")
    assert excinfo.value.kind == "bad_response"


def test_update_weights_from_disk_rejects_empty_path(server: _Server) -> None:
    with pytest.raises(EngineRefusal):
        _client(server).update_weights_from_disk("")


def test_flush_cache_does_not_raise_on_200(server: _Server) -> None:
    _client(server).flush_cache()


def test_release_and_resume_memory_do_not_raise_on_200(server: _Server) -> None:
    client = _client(server)
    client.release_memory()
    client.resume_memory()


# ---------------------------------------------------------------------------
# EngineInfraError's own construction-time validation (engines/__init__.py)
# ---------------------------------------------------------------------------


def test_engine_infra_error_rejects_non_str_kind() -> None:
    with pytest.raises(EngineRefusal):
        EngineInfraError(5, "boom")  # type: ignore[arg-type]


def test_engine_infra_error_rejects_empty_kind() -> None:
    with pytest.raises(EngineRefusal):
        EngineInfraError("", "boom")


def test_engine_infra_error_rejects_non_str_message() -> None:
    with pytest.raises(EngineRefusal):
        EngineInfraError("transport", 5)  # type: ignore[arg-type]


def test_engine_infra_error_stores_kind_and_message() -> None:
    error = EngineInfraError("transport", "connection reset")
    assert error.kind == "transport"
    assert str(error) == "connection reset"
