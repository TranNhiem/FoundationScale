"""Shared fake-SGLang-server fixtures for the engine/fleet/weight-sync test plane.

NOT a ``test_*.py`` module (pytest never collects it directly): a standalone fake
HTTP server script, written to a caller's ``tmp_path`` and launched exactly the
way ``engines.fleet.SGLangServer.start`` launches a real SGLang server
(``python_bin -m <launch_module> --model-path ... --port ... --tp-size ...``),
never importing or starting the real ``sglang`` package. The same script also
backs ``EngineServer``/``EngineServerSpec`` tests (``make_engine_server_spec``
below): its ``--model-path`` value crashing the process when it contains "FAIL"
lets a restart-mode ``DiskWeightSync`` push be tested end to end, the same
"FAIL" convention ``/update_weights_from_disk`` already uses for reload-mode.
"""

from __future__ import annotations

import socket
import sys
import textwrap
from pathlib import Path
from typing import Literal

from foundationscale.agentic_rl.engines.fleet import EngineServerSpec, SGLangServerSpec

FAKE_SERVER_SOURCE = textwrap.dedent(
    """
    import argparse
    import json
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--mem-fraction-static", required=False)
    parser.add_argument("--crash-immediately", action="store_true")
    parser.add_argument("--never-healthy", action="store_true")
    parser.add_argument("--ignore-sigterm", action="store_true")
    parser.add_argument("--spawn-child-pidfile", required=False)
    args, _unknown = parser.parse_known_args()

    if args.crash_immediately or "FAIL" in args.model_path:
        sys.exit(7)

    if args.ignore_sigterm:
        import signal

        signal.signal(signal.SIGTERM, lambda signum, frame: None)

    if args.spawn_child_pidfile:
        import subprocess

        # No start_new_session here: this grandchild inherits the fake server's
        # own process group (exactly like a real engine's tensor-parallel worker
        # would), so it is only reachable by a killpg() targeting that group, not
        # by signalling the fake server's single pid.
        child = subprocess.Popen(["sleep", "60"])
        with open(args.spawn_child_pidfile, "w") as fh:
            fh.write(str(child.pid))

    # Tracks which checkpoint this fake server last successfully "loaded" via
    # /update_weights_from_disk, echoed into /generate's own response text so
    # a test can observe a reload/restart actually taking effect on the NEXT
    # rollout, not merely that the call didn't raise.
    state = {"loaded_model_path": args.model_path}

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                if args.never_healthy:
                    self._send_json(503, {"status": "not ready"})
                else:
                    # Real servers answer /health with 200 and an EMPTY body.
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8")) if raw else {}
            if self.path == "/generate":
                self._send_json(
                    200,
                    {
                        "text": "ok loaded:" + state["loaded_model_path"],
                        "output_ids": [1],
                        "meta_info": {
                            "output_token_logprobs": [[-0.1, 1]],
                            "finish_reason": {"type": "stop"},
                        },
                    },
                )
                return
            if self.path == "/update_weights_from_disk":
                model_path = body.get("model_path", "")
                if "FAIL" in model_path:
                    self._send_json(200, {"success": False, "message": "simulated failure"})
                else:
                    state["loaded_model_path"] = model_path
                    self._send_json(200, {"success": True})
                return
            self._send_json(200, {"success": True})

        def log_message(self, format, *args):
            pass


    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever(poll_interval=0.01)
    """
)

FAKE_MODULE_NAME = "fake_engine_server"


def write_fake_server(tmp_path: Path) -> str:
    """Write the fake server script into ``tmp_path`` and return its module name."""
    (tmp_path / f"{FAKE_MODULE_NAME}.py").write_text(FAKE_SERVER_SOURCE, encoding="utf-8")
    return FAKE_MODULE_NAME


def free_port() -> int:
    """An ephemeral 127.0.0.1 port, free at the instant of the call."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def make_server_spec(
    tmp_path: Path, fake_module: str, *, port: int, **overrides: object
) -> SGLangServerSpec:
    """A ``SGLangServerSpec`` pointed at the fake server script, PYTHONPATH-wired so
    ``-m <fake_module>`` resolves it.
    """
    fields: dict[str, object] = {
        "model_path": "/fake/model",
        "port": port,
        "tp_size": 1,
        "mem_fraction_static": None,
        "log_dir": str(tmp_path),
        "python_bin": sys.executable,
        "env": {"PYTHONPATH": str(tmp_path)},
        "launch_module": fake_module,
        "startup_timeout_s": 10.0,
        "health_poll_interval_s": 0.05,
        "request_timeout_s": 5.0,
    }
    fields.update(overrides)
    return SGLangServerSpec(**fields)  # type: ignore[arg-type]


def make_engine_server_spec(
    tmp_path: Path,
    fake_module: str,
    *,
    kind: Literal["sglang", "vllm"] = "sglang",
    port: int,
    model_path: str = "{model_path}",
    served_model_name: str | None = None,
    **overrides: object,
) -> EngineServerSpec:
    """A generic ``EngineServerSpec`` pointed at the fake server script, with a
    FULL argv equivalent to what ``SGLangServer.start`` assembles internally --
    ``--model-path`` carries the literal ``"{model_path}"`` placeholder by
    default so a restart-mode ``DiskWeightSync`` push has something to
    substitute.
    """
    command = (
        sys.executable,
        "-m",
        fake_module,
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--tp-size",
        "1",
    )
    fields: dict[str, object] = {
        "kind": kind,
        "command": command,
        "port": port,
        "log_dir": str(tmp_path),
        "env": {"PYTHONPATH": str(tmp_path)},
        "startup_timeout_s": 10.0,
        "health_poll_interval_s": 0.05,
        "request_timeout_s": 5.0,
        "served_model_name": served_model_name or ("served-model" if kind == "vllm" else None),
    }
    fields.update(overrides)
    return EngineServerSpec(**fields)  # type: ignore[arg-type]
