# Covers the egress proxy's error branches hermetically (127.0.0.1 and AF_UNIX only).

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import uuid
from pathlib import Path

import pytest

from foundationscale.agentic_rl.sandbox import egress_proxy as ep

# 60-61: _is_ip caught ValueError -> False


def test_is_ip_non_ip_host_returns_false() -> None:
    assert ep._is_ip("definitely-not-an-ip") is False
    assert ep._is_ip("") is False
    assert ep._is_ip("12345.1.1.1") is False


def test_is_ip_real_ips_return_true() -> None:
    assert ep._is_ip("127.0.0.1") is True
    assert ep._is_ip("[::1]") is True


# 74-75: _split_hostport IPv6-style [v6]:port branch


def test_split_hostport_bracket_v6_with_port() -> None:
    assert ep._split_hostport("[::1]:8443", 443) == ("::1", 8443)


def test_split_hostport_bracket_v6_no_port_default_used() -> None:
    assert ep._split_hostport("[2001:db8::1]", 443) == ("2001:db8::1", 443)


def test_split_hostport_bracket_v6_non_numeric_port_default_used() -> None:
    assert ep._split_hostport("[::1]:notaport", 443) == ("::1", 443)


def test_split_hostport_ipv6_literal_without_brackets_accurate_detect() -> None:
    # authority.count(":") != 1 so it routes to the ASCII path and uses the default port
    assert ep._split_hostport("2001:db8::1", 80) == ("2001:db8::1", 80)


# 88-89: _pipe swallows ConnectionError


async def _make_pipe_pair(reader_factory, writer_factory): ...


def test_pipe_swallows_connection_error_and_closes_writer() -> None:
    """Bytes raised from reader.read -> ConnectionError(88,89) is swallowed and writer is closed."""

    closed = {"flag": False}

    class _FakeReader:
        async def read(self, n: int) -> bytes:
            raise ConnectionResetError("simulated peer reset")

    class _FakeWriter:
        def write(self, data: bytes) -> None:  # pragma: no cover - should not be called
            raise AssertionError("write should not be reached on ConnectionError")

        async def drain(self) -> None:  # pragma: no cover
            raise AssertionError("drain should not be reached on ConnectionError")

        def close(self) -> None:
            closed["flag"] = True

    asyncio.run(ep._pipe(_FakeReader(), _FakeWriter()))  # must not raise
    assert closed["flag"] is True


def test_pipe_swallows_incomplete_read_error_and_closes_writer() -> None:
    closed = {"flag": False}

    class _FakeReader:
        async def read(self, n: int) -> bytes:
            raise asyncio.IncompleteReadError(partial=b"", expected=4)

    class _FakeWriter:
        def write(self, data: bytes) -> None:  # pragma: no cover
            raise AssertionError("write should not be reached")

        async def drain(self) -> None:  # pragma: no cover
            raise AssertionError("drain should not be reached")

        def close(self) -> None:
            closed["flag"] = True

    asyncio.run(ep._pipe(_FakeReader(), _FakeWriter()))  # must not raise
    assert closed["flag"] is True


# Over-the-wire helper: start proxy on a short unix socket


async def _open_proxy_and_send(tmp_path: Path, policy: dict, payload: bytes) -> bytes:
    """Start the proxy on a short AF_UNIX path, send `payload` in one write, read the reply."""
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    short = Path.home() / ".fs-sock"  # AF_UNIX paths are limited to ~108 bytes
    short.mkdir(exist_ok=True)
    sock = short / f"test-{uuid.uuid4().hex[:8]}.sock"
    proxy = asyncio.create_task(ep.serve(sock, policy_path))
    for _ in range(50):
        if sock.exists():
            break
        await asyncio.sleep(0.02)
    try:
        reader, writer = await asyncio.open_unix_connection(str(sock))
        writer.write(payload)
        await writer.drain()
        out = b""
        try:
            while chunk := await asyncio.wait_for(reader.read(4096), 2):
                out += chunk
        except (asyncio.TimeoutError, ConnectionError):
            pass
        with contextlib.suppress(Exception):
            writer.close()
        return out
    finally:
        proxy.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await proxy
        sock.unlink(missing_ok=True)


# 132-134: readuntil raises IncompleteReadError/ConnectionError -> close & return


def test_handle_incomplete_request_head_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request head that never ends in a blank line raises IncompleteReadError (132-134)."""

    got = {"closed": False}

    class _FakeReader:
        async def readuntil(self, sep: bytes) -> bytes:
            raise asyncio.IncompleteReadError(partial=b"GET / HTTP/1.1\r\n", expected=4)

    class _FakeWriter:
        def write(self, data: bytes) -> None:  # pragma: no cover
            raise AssertionError("should write nothing when head is truncated")

        def close(self) -> None:
            got["closed"] = True

        async def drain(self) -> None:  # pragma: no cover
            raise AssertionError("should drain nothing when head is truncated")

    asyncio.run(ep._handle(_FakeReader(), _FakeWriter(), tmp_path / "policy.json"))
    assert got["closed"] is True


def test_handle_limit_overrun_closes_connection(tmp_path: Path) -> None:
    got = {"closed": False}

    class _FakeReader:
        async def readuntil(self, sep: bytes) -> bytes:
            raise asyncio.LimitOverrunError("separators too far apart", 10)

    class _FakeWriter:
        def write(self, data: bytes) -> None:  # pragma: no cover
            raise AssertionError("should write nothing when head overruns the limit")

        def close(self) -> None:
            got["closed"] = True

        async def drain(self) -> None:  # pragma: no cover
            raise AssertionError("should drain nothing")

    asyncio.run(ep._handle(_FakeReader(), _FakeWriter(), tmp_path / "policy.json"))
    assert got["closed"] is True


def test_handle_connection_error_on_head_closes_connection(tmp_path: Path) -> None:
    got = {"closed": False}

    class _FakeReader:
        async def readuntil(self, sep: bytes) -> bytes:
            raise ConnectionResetError("peer reset mid-head")

    class _FakeWriter:
        def write(self, data: bytes) -> None:  # pragma: no cover
            raise AssertionError("should write nothing on ConnectionError")

        def close(self) -> None:
            got["closed"] = True

        async def drain(self) -> None:  # pragma: no cover
            raise AssertionError("should drain nothing")

    asyncio.run(ep._handle(_FakeReader(), _FakeWriter(), tmp_path / "policy.json"))
    assert got["closed"] is True


# 138-140: malformed (non-3-token) request line -> 400 Bad Request


def test_malformed_request_line_gets_400(tmp_path: Path) -> None:
    out = asyncio.run(
        _open_proxy_and_send(tmp_path, {"mode": "public"}, b"GARBAGE-NOT-A-REQUEST-LINE\r\n\r\n")
    )
    assert out.startswith(b"HTTP/1.1 400 Bad Request")
    assert b"Connection: close" in out


def test_too_many_tokens_in_request_line_gets_400(tmp_path: Path) -> None:
    out = asyncio.run(
        _open_proxy_and_send(
            tmp_path,
            {"mode": "public"},
            b"GET / HTTP/1.1 EXTRA-TOKEN\r\nHost: x\r\n\r\n",
        )
    )
    assert out.startswith(b"HTTP/1.1 400 Bad Request")


# 148-150: not CONNECT and not absolute-URI (http://) -> 400


def test_origin_form_request_gets_400(tmp_path: Path) -> None:
    """A plain origin-form request line (no target scheme) is rejected by 148-150."""
    out = asyncio.run(
        _open_proxy_and_send(
            tmp_path,
            {"mode": "public"},
            b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n",
        )
    )
    assert out.startswith(b"HTTP/1.1 400 Bad Request")
    assert b"Connection: close" in out


def test_https_scheme_without_connect_gets_400(tmp_path: Path) -> None:
    out = asyncio.run(
        _open_proxy_and_send(
            tmp_path,
            {"mode": "public"},
            b"GET https://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n",
        )
    )
    assert out.startswith(b"HTTP/1.1 400 Bad Request")


# 166-169: _connect_upstream ultimately raises OSError/TimeoutError -> 502


def test_upstream_connect_failure_returns_502(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def always_fail(host: str, port: int, method: str):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(ep, "_connect_upstream", always_fail)
    out = asyncio.run(
        _open_proxy_and_send(
            tmp_path,
            {"mode": "public"},
            b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n",
        )
    )
    assert out.startswith(b"HTTP/1.502".replace(b"HTTP/1.502", b"HTTP/1.1 502"))
    assert b"Bad Gateway" in out
    assert b"Connection: close" in out


def test_upstream_connect_timeout_returns_502(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def always_time_out(host: str, port: int, method: str):
        raise asyncio.TimeoutError("slow upstream")

    monkeypatch.setattr(ep, "_connect_upstream", always_time_out)
    out = asyncio.run(
        _open_proxy_and_send(
            tmp_path,
            {"mode": "public"},
            b"CONNECT example.com:443 HTTP/1.1\r\n\r\n",
        )
    )
    assert out.startswith(b"HTTP/1.1 502 Bad Gateway")
    assert b"Connection: close" in out


# 198-211: argparse main() -> --socket/--policy required, --log optional, __main__


def test_main_requires_socket_and_policy() -> None:
    with (
        pytest.raises(SystemExit) as excinfo,
        contextlib.redirect_stderr(__import__("io").StringIO()),
    ):
        ep.main()
    assert excinfo.value.code != 0


def test_main_rejects_unknown_argument() -> None:
    import io

    with pytest.raises(SystemExit) as excinfo:
        import sys as _sys

        old = _sys.argv
        _sys.argv = ["egress_proxy", "--socket", "s", "--policy", "p", "--bogus", "x"]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                ep.main()
        finally:
            _sys.argv = old
    assert excinfo.value.code != 0


def test_main_opens_log_file_and_runs_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """203-207: --log creates the parent dir and sets _log_file before serve runs."""
    import io as _io
    import sys as _sys

    called = {}

    async def fake_serve(socket_path: Path, policy_path: Path) -> None:
        called["socket"] = socket_path
        called["policy"] = policy_path

    monkeypatch.setattr(ep, "serve", fake_serve)
    sock = tmp_path / "deep" / "dir" / "proxy.sock"
    pol = tmp_path / "policy.json"
    pol.write_text("{}")
    log = tmp_path / "logs" / "nested" / "egress.log"

    old = _sys.argv
    _sys.argv = ["egress_proxy", "--socket", str(sock), "--policy", str(pol), "--log", str(log)]
    try:
        with contextlib.redirect_stderr(_io.StringIO()), contextlib.redirect_stdout(_io.StringIO()):
            ep.main()
    finally:
        _sys.argv = old

    assert called["socket"] == sock
    assert called["policy"] == pol
    assert log.parent.is_dir()  # created by args.log.parent.mkdir(parents=True)
    assert ep._log_file is not None
    # reset global to avoid polluting other tests
    ep._log_file.close()
    ep._log_file = None


def test_main_without_log_flag_runs_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io as _io
    import sys as _sys

    called = {}

    async def fake_serve(socket_path: Path, policy_path: Path) -> None:
        called["socket"] = socket_path

    monkeypatch.setattr(ep, "serve", fake_serve)
    monkeypatch.setattr(ep, "_log_file", None)
    sock = tmp_path / "proxy.sock"
    pol = tmp_path / "policy.json"
    pol.write_text("{}")

    old = _sys.argv
    _sys.argv = ["egress_proxy", "--socket", str(sock), "--policy", str(pol)]
    try:
        with contextlib.redirect_stderr(_io.StringIO()), contextlib.redirect_stdout(_io.StringIO()):
            ep.main()
    finally:
        _sys.argv = old

    assert called["socket"] == sock
    assert ep._log_file is None


def test_module_dunder_main_invokes_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing the module as __main__ reaches main(), which hands serve() to asyncio.run."""
    import runpy

    pol = tmp_path / "policy.json"
    pol.write_text('{"mode": "public"}')
    seen: list[str] = []

    def fake_run(coro):  # serve() never starts: close the coroutine and record what main() passed
        seen.append(coro.cr_code.co_name)
        coro.close()

    monkeypatch.setattr(asyncio, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["egress_proxy", "--socket", str(tmp_path / "p.sock"), "--policy", str(pol)]
    )
    runpy.run_module("foundationscale.agentic_rl.sandbox.egress_proxy", run_name="__main__")
    assert seen == ["serve"]
