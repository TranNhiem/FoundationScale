"""Per-sandbox egress proxy: the only route out of a sandbox's network namespace.

Listens on a host Unix socket (bridged into the sandbox netns as 127.0.0.1:<port> by socat) and
forwards HTTPS (CONNECT) and plain HTTP (absolute-URI requests, as apt sends) only to hosts the
current policy allows.
The policy file is re-read on every connection, so a Harbor phase switch is one atomic file replace:

    {"mode": "public" | "allowlist" | "no-network", "hosts": ["example.com", "*.example.org", ...]}

Fails closed: an unreadable or unknown policy denies; IP literals are allowed only
when listed exactly;
``*.example.org`` matches subdomains, not the apex. Every decision is logged as one line on stderr,
and also appended to ``--log`` when given (the sandbox runner discards helper stderr, so a failed
upstream connect left no record of its cause: B2.2 incident 5). A failed upstream connect is retried
before it becomes a 502.

Run: python -m foundationscale.agentic_rl.sandbox.egress_proxy --socket PATH --policy FILE
     [--log FILE]
Prints "ready" once listening.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import sys
from pathlib import Path
from typing import TextIO

_BUF = 65536
_CONNECT_ATTEMPTS = 3
_CONNECT_TIMEOUT_S = 15.0
_log_file: TextIO | None = None


def host_allowed(host: str, policy: dict) -> bool:
    mode = policy.get("mode")
    if mode == "public":
        return True
    if mode != "allowlist":
        return False
    host = host.strip().lower().rstrip(".")
    for entry in policy.get("hosts") or []:
        entry = str(entry).strip().lower().rstrip(".")
        if entry.startswith("*."):
            if host.endswith(entry[1:]) and host != entry[2:]:
                return True
        elif host == entry:
            return True
    return False


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def load_policy(path: Path) -> dict:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"mode": "no-network"}
    return policy if isinstance(policy, dict) else {"mode": "no-network"}


def _split_hostport(authority: str, default_port: int) -> tuple[str, int]:
    if authority.startswith("["):  # [v6]:port
        host, _, rest = authority[1:].partition("]")
        port = rest.lstrip(":")
    else:
        host, _, port = (
            authority.rpartition(":") if authority.count(":") == 1 else (authority, "", "")
        )
    return host, int(port) if port.isdigit() else default_port


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(_BUF):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(Exception):  # noqa: BLE001 - closing a broken transport
            writer.close()


def _log(verdict: str, method: str, host: str, port: int, detail: str = "") -> None:
    line = f"egress {verdict} {method} {host}:{port}" + (f" {detail}" if detail else "")
    print(line, file=sys.stderr, flush=True)
    if _log_file is not None:
        print(line, file=_log_file, flush=True)


async def _connect_upstream(
    host: str, port: int, method: str
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open the upstream connection, retrying transient failures (DNS, refused, timeout)."""
    for attempt in range(1, _CONNECT_ATTEMPTS):
        try:
            return await asyncio.wait_for(asyncio.open_connection(host, port), _CONNECT_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError) as exc:
            _log(
                "RETRY", method, host, port, f"attempt={attempt} error={type(exc).__name__}: {exc}"
            )
            await asyncio.sleep(0.5 * 2 ** (attempt - 1))
    try:
        return await asyncio.wait_for(asyncio.open_connection(host, port), _CONNECT_TIMEOUT_S)
    except (OSError, asyncio.TimeoutError) as exc:
        _log(
            "FAIL",
            method,
            host,
            port,
            f"attempt={_CONNECT_ATTEMPTS} error={type(exc).__name__}: {exc}",
        )
        raise


async def _handle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, policy_path: Path
) -> None:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        writer.close()
        return
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
        writer.close()
        return
    method, target, version = parts
    if method == "CONNECT":
        host, port = _split_hostport(target, 443)
    elif target.lower().startswith("http://"):
        authority, _, path = target[7:].partition("/")
        host, port = _split_hostport(authority, 80)
    else:
        writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
        writer.close()
        return
    policy = load_policy(policy_path)
    allowed = host_allowed(host, policy) and (
        not _is_ip(host) or policy.get("mode") == "public" or host in (policy.get("hosts") or [])
    )
    if not allowed:
        _log("DENY", method, host, port)
        writer.write(
            b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n"
            b"egress denied by sandbox network policy\n"
        )
        await writer.drain()
        writer.close()
        return
    try:
        up_reader, up_writer = await _connect_upstream(host.strip("[]"), port, method)
    except (OSError, asyncio.TimeoutError):
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
        writer.close()
        return
    _log("ALLOW", method, host, port)
    if method == "CONNECT":
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
    else:
        headers = [
            h for h in lines[1:] if h and not h.lower().startswith(("proxy-", "connection:"))
        ]
        request = "\r\n".join(
            [f"{method} /{path} {version}", *headers, "Connection: close", "", ""]
        )
        up_writer.write(request.encode("latin-1"))
        await up_writer.drain()
    await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))


async def serve(socket_path: Path, policy_path: Path) -> None:
    socket_path.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(
        lambda r, w: _handle(r, w, policy_path), path=str(socket_path), limit=_BUF
    )
    socket_path.chmod(0o600)
    print("ready", flush=True)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    if args.log is not None:
        global _log_file
        args.log.parent.mkdir(parents=True, exist_ok=True)
        _log_file = args.log.open("a", encoding="utf-8")  # noqa: SIM115 - lives as long as the proxy
    asyncio.run(serve(args.socket, args.policy))


if __name__ == "__main__":
    main()
