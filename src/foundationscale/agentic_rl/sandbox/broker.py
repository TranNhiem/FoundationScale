"""Host-side exec broker: lets a containerized trainer drive ``EnrootSandbox`` on its host.

A trainer that runs inside an enroot container (the GB200 recipe) cannot start sandboxes itself:
the container has no ``enroot`` binary, and the host's enroot tools need a newer glibc than the
container ships. enroot containers share the host's network and PID namespaces and see ``$HOME``
at the same path, though, so the only part of ``EnrootSandbox`` that has to leave the container
is its :class:`~foundationscale.agentic_rl.sandbox.enroot.Runner` (the ``enroot`` / ``unshare``
/ ``systemd-run`` process launches). This module moves exactly that:

- :func:`serve` runs on the host next to the container and executes runner requests with a
  :class:`~foundationscale.agentic_rl.sandbox.enroot.SubprocessRunner`;
- :class:`BrokerRunner` is the in-container ``Runner`` (``run``, ``spawn``, ``terminate``,
  ``kill_all``) that forwards each call to the broker.

The broker executes arbitrary argv as the user running it, so it binds loopback only and every
request must carry the token stored in a ``0600`` file that only that user can read (other users
on the node can reach the port, not the token). One request per connection, one JSON line each
way. Each ``BrokerRunner`` is its own session on the broker, so ``kill_all`` reaches only the
runs of the sandbox that issued it, never another episode's.

    python -m foundationscale.agentic_rl.sandbox.broker --port 18600 --token-file ~/.fsar-broker
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import socket
import socketserver
import stat
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from foundationscale.agentic_rl.sandbox.enroot import (
    RunResult,
    SandboxError,
    SandboxRefusal,
    SubprocessRunner,
)

_LOOPBACK = ("127.0.0.1", "localhost")
_MAX_REQUEST_BYTES = 4 * 1024 * 1024
# A broker answer arrives after the command finishes; allow this much on top of its own timeout.
_REPLY_GRACE_S = 60.0


def read_token(token_file: str | Path) -> str:
    """The token in ``token_file``; refuse a file other users could read or a blank token."""
    path = Path(token_file)
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise SandboxRefusal(f"token_file: expected a readable file, got {path} ({exc})") from exc
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SandboxRefusal(
            f"token_file: expected mode 0600 (owner only), got {oct(mode & 0o777)} on {path}: "
            f"a token other users can read lets them run commands as you"
        )
    token = path.read_text().strip()
    if not token:
        raise SandboxRefusal(f"token_file: expected a non-empty token, got a blank {path}")
    return token


def ensure_token_file(token_file: str | Path) -> str:
    """Create ``token_file`` (mode 0600, random token) unless it exists; return the token."""
    path = Path(token_file)
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(secrets.token_hex(32) + "\n")
    return read_token(path)


class _Sessions:
    """One ``SubprocessRunner`` per client session, so ``kill_all`` reaches only that client's
    runs: every ``EnrootSandbox`` calls ``kill_all`` when it stops, and on a shared runner one
    stopping sandbox SIGKILLed the commands of every other episode in flight. An idle session is
    an empty runner (about a kilobyte), so a run's thousands of sandboxes stay cheap."""

    def __init__(self) -> None:
        self._runners: dict[str, SubprocessRunner] = {}
        self._lock = threading.Lock()

    def get(self, session: str) -> SubprocessRunner:
        with self._lock:
            return self._runners.setdefault(session, SubprocessRunner())


def _handle(sessions: _Sessions, token: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one authenticated request; every failure is an ``{"error": ...}`` reply."""
    offered = request.get("token")
    if not isinstance(offered, str) or not hmac.compare_digest(offered, token):
        return {"error": "unauthorized"}
    session = request.get("session")
    if not isinstance(session, str) or not session:
        return {"error": "bad request: expected a non-empty session id"}
    runner = sessions.get(session)
    op = request.get("op")
    try:
        if op == "run":
            result = runner.run(
                _argv(request), env=request.get("env"), timeout_s=request.get("timeout_s")
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
            }
        if op == "spawn":
            ready = float(request.get("ready_timeout_s", 30.0))
            return {"pid": runner.spawn(_argv(request), ready_timeout_s=ready)}
        if op == "terminate":
            runner.terminate(int(request["pid"]))
            return {}
        if op == "kill_all":
            runner.kill_all()
            return {}
    except (SandboxError, SandboxRefusal, KeyError, TypeError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"error": f"unknown op {op!r}"}


def _argv(request: Mapping[str, Any]) -> list[str]:
    argv = request.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise SandboxRefusal(f"argv: expected a non-empty list of str, got {argv!r}")
    return argv


def make_server(host: str, port: int, token: str) -> socketserver.ThreadingTCPServer:
    """A threaded loopback server answering broker requests (not yet serving)."""
    if host not in _LOOPBACK:
        raise SandboxRefusal(
            f"host: expected a loopback address {_LOOPBACK}, got {host!r}: the broker runs "
            f"arbitrary commands and must not be reachable from other machines"
        )
    sessions = _Sessions()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline(_MAX_REQUEST_BYTES)
            try:
                request = json.loads(line)
            except ValueError:
                reply: dict[str, Any] = {"error": "bad request: not one JSON line"}
            else:
                if isinstance(request, dict):
                    reply = _handle(sessions, token, request)
                else:
                    reply = {"error": "bad request: not a JSON object"}
            self.wfile.write(json.dumps(reply).encode() + b"\n")

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer((host, port), Handler)
    server.daemon_threads = True
    return server


class BrokerRunner:
    """In-container ``Runner`` forwarding every call to a host broker (see the module doc)."""

    def __init__(self, host: str, port: int, token: str) -> None:
        self._address = (host, port)
        self._token = token
        self._session = secrets.token_hex(8)  # scopes kill_all to this runner's own runs

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> BrokerRunner | None:
        """``FS_SANDBOX_BROKER=host:port`` + ``FS_SANDBOX_BROKER_TOKEN_FILE``; None if unset."""
        address = environ.get("FS_SANDBOX_BROKER", "")
        if not address:
            return None
        host, _, port = address.rpartition(":")
        if not host or not port.isdigit():
            raise SandboxRefusal(f"FS_SANDBOX_BROKER: expected host:port, got {address!r}")
        token_file = environ.get("FS_SANDBOX_BROKER_TOKEN_FILE", "")
        if not token_file:
            raise SandboxRefusal(
                "FS_SANDBOX_BROKER_TOKEN_FILE: expected the broker's token file, got unset"
            )
        return cls(host, int(port), read_token(token_file))

    def _call(self, op: str, wait_s: float | None, **fields: Any) -> dict[str, Any]:
        payload = {"token": self._token, "session": self._session, "op": op, **fields}
        request = json.dumps(payload).encode() + b"\n"
        wait = None if wait_s is None else wait_s + _REPLY_GRACE_S
        try:
            with socket.create_connection(self._address, timeout=30.0) as conn:
                conn.settimeout(wait)
                conn.sendall(request)
                with conn.makefile("rb") as reader:
                    line = reader.readline()
        except OSError as exc:
            raise SandboxError(f"broker {self._address}: {op} failed: {exc}") from exc
        try:
            reply = json.loads(line)
        except ValueError as exc:
            raise SandboxError(f"broker {self._address}: {op}: unreadable reply {line!r}") from exc
        if not isinstance(reply, dict):
            raise SandboxError(f"broker {self._address}: {op}: unreadable reply {line!r}")
        if "error" in reply:
            raise SandboxError(f"broker {self._address}: {op}: {reply['error']}")
        return reply

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunResult:
        env_field = None if env is None else dict(env)
        reply = self._call("run", timeout_s, argv=list(argv), env=env_field, timeout_s=timeout_s)
        return RunResult(
            returncode=int(reply["returncode"]),
            stdout=str(reply["stdout"]),
            stderr=str(reply["stderr"]),
            timed_out=bool(reply["timed_out"]),
        )

    def spawn(self, argv: Sequence[str], *, ready_timeout_s: float = 30.0) -> int:
        reply = self._call(
            "spawn", ready_timeout_s, argv=list(argv), ready_timeout_s=ready_timeout_s
        )
        return int(reply["pid"])

    def terminate(self, pid: int) -> None:
        self._call("terminate", 30.0, pid=pid)

    def kill_all(self) -> None:
        self._call("kill_all", 30.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FoundationScale host-side sandbox exec broker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-file", required=True, help="created (0600) if missing")
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port, ensure_token_file(Path(args.token_file)))
    print(f"broker ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0  # pragma: no cover - serve_forever only returns on shutdown()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
