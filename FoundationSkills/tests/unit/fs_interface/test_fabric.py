"""Fabric probe + TTL cache tests: state mapping, TTL semantics, launch blocking."""
from __future__ import annotations

import errno
import json

import pytest

import foundationskills.interfaces.fs.fabric as fabric_mod
from foundationskills.interfaces.fs.fabric import (
    FABRIC_CACHE_NAME,
    check_fabric,
    launch_blocking,
    probe_fabric,
)


def _probe(state: str, reason: str, ts: float = 12.5):
    calls: list = []

    def probe(host: str = "master", port: int = 8081, timeout_s: float = 2.0) -> dict:
        calls.append((host, port, timeout_s))
        return {"state": state, "reason": reason, "ts": ts}

    probe.calls = calls
    return probe


def _seed(cache_path, key: str = "master:8081", **entry) -> None:
    cache_path.write_text(json.dumps({key: entry}))


class TestProbeFabric:
    def test_ready(self):
        out = probe_fabric(connector=lambda host, port, timeout_s: None)
        assert out["state"] == "ready"
        assert out["reason"] == "fabric_ready"
        assert isinstance(out["ts"], float) and out["ts"] > 0.0

    def test_refused(self):
        def connector(host, port, timeout_s):
            raise ConnectionRefusedError(111, "connection refused")

        out = probe_fabric(connector=connector)
        assert out["state"] == "refused"
        assert out["reason"] == "fabric_refused"

    def test_timeout_maps_to_unmeasured(self):
        def connector(host, port, timeout_s):
            raise TimeoutError()

        out = probe_fabric(connector=connector)
        assert out["state"] == "unmeasured"
        assert out["reason"] == "fabric_unmeasured:timeout"

    def test_oserror_maps_errno_name(self):
        def connector(host, port, timeout_s):
            raise OSError(errno.ENETUNREACH, "network unreachable")

        out = probe_fabric(connector=connector)
        assert out["state"] == "unmeasured"
        assert out["reason"] == "fabric_unmeasured:ENETUNREACH"

    def test_oserror_without_errno_maps_class(self):
        def connector(host, port, timeout_s):
            raise OSError("boom")

        out = probe_fabric(connector=connector)
        assert out["state"] == "unmeasured"
        assert out["reason"] == "fabric_unmeasured:OSError"

    def test_default_connector_closes_socket(self, monkeypatch):
        seen: dict = {}

        class _Con:
            def close(self) -> None:
                seen["closed"] = True

        def fake_create(addr, timeout=None):
            seen["addr"] = addr
            seen["timeout"] = timeout
            return _Con()

        monkeypatch.setattr(fabric_mod.socket, "create_connection", fake_create)
        out = probe_fabric(host="h1", port=9, timeout_s=1.5)
        assert out["state"] == "ready"
        assert seen == {"addr": ("h1", 9), "timeout": 1.5, "closed": True}


class TestCheckFabric:
    def test_miss_probes_and_writes_cache(self, tmp_path):
        cache = tmp_path / FABRIC_CACHE_NAME
        probe = _probe("refused", "fabric_refused", ts=42.0)
        out = check_fabric(cache, ttl_s=300.0, clock=lambda: 100.0, probe=probe)
        assert out == {"state": "refused", "cached": False, "age_s": 0.0, "reason": "fabric_refused"}
        assert probe.calls == [("master", 8081, 2.0)]
        assert json.loads(cache.read_text()) == {
            "master:8081": {"state": "refused", "reason": "fabric_refused", "ts": 42.0}
        }

    def test_cache_key_is_host_port(self, tmp_path):
        cache = tmp_path / FABRIC_CACHE_NAME
        probe = _probe("ready", "fabric_ready", ts=1.0)
        check_fabric(cache, clock=lambda: 1.0, probe=probe, host="n1", port=8082)
        assert probe.calls == [("n1", 8082, 2.0)]
        assert json.loads(cache.read_text()) == {
            "n1:8082": {"state": "ready", "reason": "fabric_ready", "ts": 1.0}
        }

    def test_ttl_hit_skips_probe_and_connector(self, tmp_path):
        cache = tmp_path / FABRIC_CACHE_NAME
        _seed(cache, state="refused", reason="fabric_refused", ts=1000.0)
        probed: list = []
        connected: list = []

        def connector(host, port, timeout_s):
            connected.append((host, port, timeout_s))

        def probe(host="master", port=8081, timeout_s=2.0):
            probed.append((host, port))
            return probe_fabric(host=host, port=port, timeout_s=timeout_s, connector=connector)

        out = check_fabric(cache, ttl_s=300.0, clock=lambda: 1030.0, probe=probe)
        assert out == {"state": "refused", "cached": True, "age_s": 30.0, "reason": "fabric_refused"}
        assert probed == [] and connected == []

    def test_expiry_reprobes_and_rewrites_cache(self, tmp_path):
        cache = tmp_path / FABRIC_CACHE_NAME
        cache.write_text(json.dumps({
            "master:8081": {"state": "ready", "reason": "fabric_ready", "ts": 0.0},
            "other:1234": {"state": "refused", "reason": "fabric_refused", "ts": 1.0},
        }))
        probe = _probe("unmeasured", "fabric_unmeasured:timeout", ts=999.0)
        out = check_fabric(cache, ttl_s=300.0, clock=lambda: 1000.0, probe=probe)
        assert out == {"state": "unmeasured", "cached": False, "age_s": 0.0, "reason": "fabric_unmeasured:timeout"}
        assert len(probe.calls) == 1
        stored = json.loads(cache.read_text())
        assert stored["master:8081"] == {"state": "unmeasured", "reason": "fabric_unmeasured:timeout", "ts": 999.0}
        assert stored["other:1234"] == {"state": "refused", "reason": "fabric_refused", "ts": 1.0}

    def test_corrupt_cache_is_miss_and_gets_rewritten(self, tmp_path):
        cache = tmp_path / FABRIC_CACHE_NAME
        cache.write_text("not json")
        probe = _probe("ready", "fabric_ready", ts=7.0)
        out = check_fabric(cache, clock=lambda: 7.0, probe=probe)
        assert out == {"state": "ready", "cached": False, "age_s": 0.0, "reason": "fabric_ready"}
        assert json.loads(cache.read_text())["master:8081"]["ts"] == 7.0

    def test_write_failure_reports_unmeasured(self, tmp_path, monkeypatch):
        cache = tmp_path / FABRIC_CACHE_NAME

        def boom(src, dst):
            raise OSError(28, "no space left on device")

        monkeypatch.setattr(fabric_mod.os, "replace", boom)
        probe = _probe("ready", "fabric_ready", ts=5.0)
        out = check_fabric(cache, clock=lambda: 5.0, probe=probe)
        assert out == {"state": "unmeasured", "cached": False, "age_s": 0.0, "reason": "fabric_cache_unavailable"}
        assert len(probe.calls) == 1  # the probe ran; the storage failed
        assert list(tmp_path.iterdir()) == []  # no half-written tmp left behind

    def test_unreadable_cache_reports_unmeasured(self, tmp_path):
        cache = tmp_path / FABRIC_CACHE_NAME
        cache.mkdir()  # reading it raises IsADirectoryError: a storage failure
        probe = _probe("ready", "fabric_ready", ts=5.0)
        out = check_fabric(cache, clock=lambda: 5.0, probe=probe)
        assert out == {"state": "unmeasured", "cached": False, "age_s": 0.0, "reason": "fabric_cache_unavailable"}
        assert probe.calls == []


class TestLaunchBlocking:
    @pytest.mark.parametrize("state", ["ready"])
    def test_only_ready_allows_launch(self, state):
        assert launch_blocking(state) is False

    @pytest.mark.parametrize("state", ["refused", "unmeasured", "", "fabric_ready"])
    def test_blocks_refused_unmeasured_and_unknown(self, state):
        assert launch_blocking(state) is True


def test_future_dated_ready_entry_is_reprobed_but_refusal_stays_sticky(tmp_path):
    import json as _json
    from foundationskills.interfaces.fs.fabric import check_fabric
    cache = tmp_path / "fabric.json"
    calls = []

    def probe(**_kw):
        calls.append(1)
        return {"state": "refused", "reason": "fabric_refused", "ts": 1000.0}

    cache.write_text(_json.dumps({"master:8081": {"state": "ready", "ts": 9e9, "reason": "fabric_ready"}}))
    got = check_fabric(cache, clock=lambda: 1000.0, probe=probe)
    assert got["state"] == "refused" and calls == [1]
    cache.write_text(_json.dumps({"master:8081": {"state": "refused", "ts": 9e9, "reason": "fabric_refused"}}))
    got = check_fabric(cache, clock=lambda: 1000.0, probe=probe)
    assert got["state"] == "refused" and got["cached"] is True and calls == [1]
