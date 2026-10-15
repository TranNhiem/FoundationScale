"""Tests for foundationscale.agentic_rl.sandbox.egress_proxy (policy matching and a live CONNECT/HTTP round trip)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from foundationscale.agentic_rl.sandbox import egress_proxy as ep


@pytest.mark.parametrize(
    ("host", "policy", "allowed"),
    [
        ("registry.npmjs.org", {"mode": "public"}, True),
        ("registry.npmjs.org", {"mode": "no-network"}, False),
        ("registry.npmjs.org", {"mode": "bogus"}, False),
        ("registry.npmjs.org", {"mode": "allowlist", "hosts": ["registry.npmjs.org"]}, True),
        ("REGISTRY.npmjs.org.", {"mode": "allowlist", "hosts": ["registry.npmjs.org"]}, True),
        ("evil.org", {"mode": "allowlist", "hosts": ["registry.npmjs.org"]}, False),
        ("deb.debian.org", {"mode": "allowlist", "hosts": ["*.debian.org"]}, True),
        ("debian.org", {"mode": "allowlist", "hosts": ["*.debian.org"]}, False),
        ("notdebian.org", {"mode": "allowlist", "hosts": ["*.debian.org"]}, False),
    ],
)
def test_host_allowed(host: str, policy: dict, allowed: bool) -> None:
    assert ep.host_allowed(host, policy) is allowed


def test_unreadable_policy_denies(tmp_path: Path) -> None:
    assert ep.load_policy(tmp_path / "missing.json") == {"mode": "no-network"}
    (tmp_path / "bad.json").write_text("{not json")
    assert ep.load_policy(tmp_path / "bad.json") == {"mode": "no-network"}


async def _roundtrip(tmp_path: Path, policy: dict, request: bytes) -> bytes:
    """Start an upstream HTTP server and the proxy; send `request` through the proxy socket."""
    async def upstream(reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        body = b"upstream saw: " + head.split(b"\r\n")[0]
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    import uuid

    short = Path.home() / ".fs-sock"  # AF_UNIX paths are limited to ~108 bytes; pytest tmp paths are longer
    short.mkdir(exist_ok=True)
    sock = short / f"test-{uuid.uuid4().hex[:8]}.sock"
    proxy = asyncio.create_task(ep.serve(sock, policy_path))
    for _ in range(50):
        if sock.exists():
            break
        await asyncio.sleep(0.02)
    reader, writer = await asyncio.open_unix_connection(str(sock))
    writer.write(request.replace(b"PORT", str(port).encode()))
    await writer.drain()
    out = b""
    try:
        while chunk := await asyncio.wait_for(reader.read(4096), 2):
            out += chunk
            if b"upstream saw" in out or b"403" in out:
                break
    except asyncio.TimeoutError:
        pass
    writer.close()
    proxy.cancel()
    server.close()
    sock.unlink(missing_ok=True)
    return out


def test_plain_http_forwarded_in_origin_form(tmp_path: Path) -> None:
    out = asyncio.run(_roundtrip(tmp_path, {"mode": "allowlist", "hosts": ["127.0.0.1"]},
                                 b"GET http://127.0.0.1:PORT/pool/x.deb HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"))
    assert b"upstream saw: GET /pool/x.deb HTTP/1.1" in out


def test_denied_host_gets_403(tmp_path: Path) -> None:
    out = asyncio.run(_roundtrip(tmp_path, {"mode": "allowlist", "hosts": ["example.com"]},
                                 b"GET http://127.0.0.1:PORT/ HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"))
    assert out.startswith(b"HTTP/1.1 403")


def test_connect_tunnel_relays_bytes(tmp_path: Path) -> None:
    out = asyncio.run(_roundtrip(tmp_path, {"mode": "public"},
                                 b"CONNECT 127.0.0.1:PORT HTTP/1.1\r\n\r\nGET /tunnelled HTTP/1.1\r\n\r\n"))
    assert out.startswith(b"HTTP/1.1 200 Connection Established")
    assert b"upstream saw: GET /tunnelled HTTP/1.1" in out


def test_failed_upstream_connect_is_retried_and_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    log = tmp_path / "egress.log"
    monkeypatch.setattr(ep, "_log_file", log.open("a", encoding="utf-8"))
    with socket.socket() as s:  # a port nothing listens on: every attempt is refused
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(OSError):
        asyncio.run(ep._connect_upstream("127.0.0.1", port, "CONNECT"))
    lines = log.read_text().splitlines()
    assert [line.split()[1] for line in lines] == ["RETRY"] * (ep._CONNECT_ATTEMPTS - 1) + ["FAIL"]
    assert "ConnectionRefusedError" in lines[-1]
