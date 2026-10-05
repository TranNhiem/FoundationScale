"""IMEX fabric probe (``/dev/tcp/master/8081``) + TTL cache.

States: ``ready`` | ``refused`` | ``unmeasured``. An unmeasured fabric blocks
launch exactly like a refused one (``launch_blocking``) -- never launch on an
unknown fabric. The probe is pre-launch only; the sbatch ``exit 96`` preamble
remains the in-job gate.
"""
from __future__ import annotations

import errno
import json
import os
import socket
import time
from pathlib import Path
from typing import Callable

FABRIC_CACHE_NAME = "fabric.json"
DEFAULT_TTL_S = 300.0

FABRIC_READY = "fabric_ready"
FABRIC_REFUSED = "fabric_refused"
FABRIC_UNMEASURED = "fabric_unmeasured"
FABRIC_CACHE_UNAVAILABLE = "fabric_cache_unavailable"

Connector = Callable[[str, int, float], object]
Probe = Callable[..., dict]


def _socket_connector(host: str, port: int, timeout_s: float) -> object:
    """Shell equivalent: ``(exec 3<>/dev/tcp/<host>/<port>)``."""
    con = socket.create_connection((host, port), timeout=timeout_s)
    con.close()
    return con


def _oserr_name(exc: OSError) -> str:
    if exc.errno is not None:
        return errno.errorcode.get(exc.errno, str(exc.errno))
    return type(exc).__name__


def probe_fabric(host: str = "master", port: int = 8081, timeout_s: float = 2.0,
                 connector: Connector | None = None) -> dict:
    """Probe the IMEX fabric -> ``{"state", "reason", "ts"}``, reason always named."""
    connect = connector if connector is not None else _socket_connector
    try:
        connect(host, port, timeout_s)
    except ConnectionRefusedError:
        state, reason = "refused", FABRIC_REFUSED
    except TimeoutError:  # socket.timeout is a TimeoutError alias (3.10+)
        state, reason = "unmeasured", FABRIC_UNMEASURED + ":timeout"
    except OSError as exc:
        state, reason = "unmeasured", FABRIC_UNMEASURED + ":" + _oserr_name(exc)
    else:
        state, reason = "ready", FABRIC_READY
    return {"state": state, "reason": reason, "ts": float(time.time())}


def _unmeasured_cache() -> dict:
    return {"state": "unmeasured", "cached": False, "age_s": 0.0, "reason": FABRIC_CACHE_UNAVAILABLE}


def _read_cache(cache_path: Path) -> dict:
    cache = json.loads(cache_path.read_text())
    return cache if isinstance(cache, dict) else {}


def _write_cache(cache_path: Path, data: dict) -> None:
    """Atomic write: tmp file in the same dir + ``os.replace``."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name("." + cache_path.name + "." + str(os.getpid()) + ".tmp")
    try:
        tmp.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")))
        os.replace(tmp, cache_path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def check_fabric(cache_path: Path, *, ttl_s: float = DEFAULT_TTL_S,
                 clock: Callable[[], float] = time.time,
                 probe: Probe = probe_fabric,
                 host: str = "master", port: int = 8081) -> dict:
    """Probe behind a TTL cache: one JSON object ``{"<host>:<port>": {"state","ts","reason"}}``.

    TTL hit replays the cached state with ``cached: True`` and no probe. Any
    storage error -> ``unmeasured`` / ``fabric_cache_unavailable`` (the safe
    direction: unknown texture never launches).
    """
    key = f"{host}:{port}"
    now = float(clock())
    try:
        cache = _read_cache(cache_path)
    except FileNotFoundError:
        cache = {}
    except ValueError:  # corrupt JSON: a miss, rewritten below
        cache = {}
    except OSError:
        return _unmeasured_cache()
    entry = cache.get(key)
    age_s: float | None = None
    if isinstance(entry, dict) and {"state", "ts", "reason"} <= entry.keys():
        try:
            age_s = now - float(entry["ts"])
        except (TypeError, ValueError):
            age_s = None
    # A future-dated entry (clock skew or a hand-edited cache) is trusted only when it blocks:
    # a stale "ready" must never outlive its TTL, but a refusal may stay sticky.
    fresh = age_s is not None and (0.0 <= age_s <= ttl_s or (age_s < 0.0 and entry.get("state") != "ready"))
    if fresh:
        return {"state": str(entry["state"]), "cached": True, "age_s": age_s, "reason": str(entry["reason"])}
    result = probe(host=host, port=port)
    try:
        cache[key] = {"state": result["state"], "ts": float(result.get("ts", now)), "reason": result["reason"]}
        _write_cache(cache_path, cache)
    except OSError:
        return _unmeasured_cache()
    return {"state": result["state"], "cached": False, "age_s": 0.0, "reason": result["reason"]}


def launch_blocking(state: str) -> bool:
    """Block for ``refused`` AND ``unmeasured``: never launch on an unknown fabric."""
    return state != "ready"
