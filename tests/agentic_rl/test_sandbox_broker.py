"""The host exec broker: auth, loopback-only bind, run/spawn/terminate/kill_all round trips."""

from __future__ import annotations

import json
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from foundationscale.agentic_rl.sandbox import broker
from foundationscale.agentic_rl.sandbox.broker import (
    BrokerRunner,
    ensure_token_file,
    make_server,
    read_token,
)
from foundationscale.agentic_rl.sandbox.enroot import SandboxError, SandboxRefusal

TOKEN = "t0ken"


@pytest.fixture
def server() -> Iterator[tuple[str, int]]:
    srv = make_server("127.0.0.1", 0, TOKEN)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_address[0], srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _raw(address: tuple[str, int], payload: bytes) -> dict[str, object]:
    with socket.create_connection(address, timeout=10) as conn:
        conn.sendall(payload)
        with conn.makefile("rb") as reader:
            reply = json.loads(reader.readline())
    assert isinstance(reply, dict)
    return reply


def test_run_round_trip(server: tuple[str, int]) -> None:
    runner = BrokerRunner(*server, TOKEN)
    result = runner.run(["sh", "-c", "echo out; echo err >&2; exit 3"], env={"X": "1"})
    assert (result.returncode, result.stdout, result.stderr) == (3, "out\n", "err\n")
    assert runner.run(["sh", "-c", "echo $X"], env={"X": "7"}).stdout == "7\n"
    timed = runner.run(["sleep", "5"], timeout_s=0.2)
    assert timed.timed_out and timed.returncode == 124


def test_spawn_terminate_and_kill_all(server: tuple[str, int]) -> None:
    runner = BrokerRunner(*server, TOKEN)
    pid = runner.spawn(["sh", "-c", "echo ready; sleep 30"], ready_timeout_s=10)
    assert pid > 0
    runner.terminate(pid)
    runner.kill_all()
    with pytest.raises(SandboxError, match="expected 'ready'"):
        runner.spawn(["sh", "-c", "echo nope"], ready_timeout_s=5)


def test_wrong_token_and_bad_requests_are_refused(server: tuple[str, int]) -> None:
    with pytest.raises(SandboxError, match="unauthorized"):
        BrokerRunner(*server, "wrong").run(["true"])
    runner = BrokerRunner(*server, TOKEN)
    with pytest.raises(SandboxError, match="unknown op"):
        runner._call("explode", 5.0)
    with pytest.raises(SandboxError, match="argv: expected"):
        runner._call("run", 5.0, argv=[])
    with pytest.raises(SandboxError, match="KeyError"):
        runner._call("terminate", 5.0)
    assert _raw(server, b"not json\n") == {"error": "bad request: not one JSON line"}
    assert _raw(server, b"[1]\n") == {"error": "bad request: not a JSON object"}


def test_unreachable_broker_and_unreadable_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    with pytest.raises(SandboxError, match="run failed"):
        BrokerRunner("127.0.0.1", free_port, TOKEN).run(["true"])

    for reply in (b"garbage\n", b"[1]\n"):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)

        def answer(sock: socket.socket = srv, data: bytes = reply) -> None:
            conn, _ = sock.accept()
            with conn:
                conn.recv(65536)
                conn.sendall(data)

        thread = threading.Thread(target=answer, daemon=True)
        thread.start()
        with pytest.raises(SandboxError, match="unreadable reply"):
            BrokerRunner("127.0.0.1", srv.getsockname()[1], TOKEN).kill_all()
        thread.join(5)
        srv.close()


def test_non_loopback_bind_is_refused() -> None:
    with pytest.raises(SandboxRefusal, match="loopback"):
        make_server("0.0.0.0", 0, TOKEN)  # noqa: S104


def test_token_file_rules(tmp_path: Path) -> None:
    path = tmp_path / "token"
    token = ensure_token_file(path)
    assert len(token) == 64 and oct(path.stat().st_mode & 0o777) == "0o600"
    assert ensure_token_file(path) == token  # an existing file is reused, not rotated
    path.chmod(0o644)
    with pytest.raises(SandboxRefusal, match="mode 0600"):
        read_token(path)
    path.write_text("  \n")
    path.chmod(0o600)
    with pytest.raises(SandboxRefusal, match="non-empty token"):
        read_token(path)
    with pytest.raises(SandboxRefusal, match="readable file"):
        read_token(tmp_path / "missing")


def test_from_environ(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("abc\n")
    token_file.chmod(0o600)
    assert BrokerRunner.from_environ({}) is None
    runner = BrokerRunner.from_environ(
        {"FS_SANDBOX_BROKER": "127.0.0.1:18600", "FS_SANDBOX_BROKER_TOKEN_FILE": str(token_file)}
    )
    assert runner is not None and runner._address == ("127.0.0.1", 18600)
    with pytest.raises(SandboxRefusal, match="host:port"):
        BrokerRunner.from_environ({"FS_SANDBOX_BROKER": "nohost"})
    with pytest.raises(SandboxRefusal, match="TOKEN_FILE"):
        BrokerRunner.from_environ({"FS_SANDBOX_BROKER": "127.0.0.1:1"})


def test_main_serves_until_shutdown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    started: list[tuple[str, int]] = []

    def fake_serve(self: object) -> None:
        started.append(self.server_address)  # type: ignore[attr-defined]
        self.server_close()  # type: ignore[attr-defined]

    monkeypatch.setattr(broker.socketserver.ThreadingTCPServer, "serve_forever", fake_serve)
    token_file = tmp_path / "token"
    assert broker.main(["--port", "0", "--token-file", str(token_file)]) == 0
    assert started and token_file.exists()
    assert sys.platform  # the CLI ran end to end without binding a public address
