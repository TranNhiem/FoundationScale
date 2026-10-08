"""``emit_trial`` train-caps probe: the JSON trial spec (hashed into the launch token) can
never carry an ``FSCapabilities`` object, so ``emit_trial`` must probe caps itself and REFUSE
-- never crash -- when the probe or the emitter cannot honour the request.

CPU-only, no network, no subprocess.
"""
from __future__ import annotations

import copy
from typing import Any

from foundationskills.interfaces.fs.emit_trial import emit_trial


class _Caps:
    """An ``FSCapabilities`` stand-in: only ``available`` and ``errors`` are consulted."""

    def __init__(self, available: bool = True, errors: list[str] | None = None) -> None:
        self.available = available
        self.errors = list(errors or [])


def _json_request() -> dict[str, Any]:
    """JSON-only data: exactly what ``trial_launch_token`` hashes, so no object may ride along."""
    return {
        "trial_spec": {
            "kind": "train",
            "trial": "t7",
            "nodes": 1,
            "gpus_per_node": 1,
            "train_request": {
                "stage": {"name": "pretrain"},
                "dataset": {"name": "slimpajama"},
                "model": "fs-1b",
                "output_dir": "/tmp/fs-out",
                "hardware": {"kind": "local"},
                "run_name": "trial-t7",
            },
        }
    }


def _ok_spec(**_: Any) -> dict[str, Any]:
    return {"executable": True, "notes": [], "drops": [], "missing": [], "argv": ["python", "train.py"]}


def _emit(request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return emit_trial(request, hardware_id="local", **kwargs)


def test_caps_probed_and_passed_to_emitter():
    """(a) JSON-only train_request + caps_fn -> the injected emitter receives caps; a fact returns."""
    seen: dict[str, Any] = {}

    def emitter(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return _ok_spec()

    fact = _emit(_json_request(), emit_train_fn=emitter, caps_fn=lambda: _Caps())
    assert isinstance(seen.get("caps"), _Caps), "the probed caps must reach emit_train"
    assert fact["executable"] is True and fact["missing"] == [] and fact["fs_launch_spec"]["executable"] is True
    assert seen["nodes"] == 1 and seen["run_name"] == "trial-t7"


def test_unavailable_caps_refuses_before_emit():
    """(b) available=False -> REFUSED fact with fs_capabilities_unavailable; emit_train is never called."""
    calls: list[int] = []

    def emitter(**kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        return _ok_spec()

    fact = _emit(_json_request(), emit_train_fn=emitter, caps_fn=lambda: _Caps(False, ["no gpu measured"]))
    assert calls == [], "emit_train must never run on unavailable capabilities"
    reason = fact["missing"][0]
    assert reason.startswith("fs_capabilities_unavailable") and "no gpu measured" in reason
    assert fact["fs_launch_spec"] == {"state": "REFUSED", "reason": reason}
    assert fact["executable"] is False and fact["drops"] == fact["missing"] == [reason]


def test_probe_crash_refuses_with_type_name():
    """(c) a probe that raises -> fs_capabilities_probe_failed:<Name>, no skill crash."""

    def boom() -> Any:
        raise OSError("no device")

    fact = _emit(_json_request(), emit_train_fn=_ok_spec, caps_fn=boom)
    assert fact["fs_launch_spec"] == {"state": "REFUSED", "reason": "fs_capabilities_probe_failed:OSError"}
    assert fact["executable"] is False and fact["drops"] == fact["missing"] == ["fs_capabilities_probe_failed:OSError"]


def test_emitter_type_error_refuses():
    """(d) a malformed opaque request -> emit_train_bad_request:<exc>, never an uncaught TypeError."""

    def bad_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise TypeError("unexpected keyword argument 'dataset'")

    fact = _emit(_json_request(), emit_train_fn=bad_request, caps_fn=lambda: _Caps())
    assert fact["executable"] is False and fact["fs_launch_spec"]["state"] == "REFUSED"
    assert fact["missing"][0].startswith("emit_train_bad_request:") and "unexpected keyword" in fact["missing"][0]


def test_trial_request_is_never_mutated():
    """(e) the caller's trial_request dict is identical after the call."""
    request = _json_request()
    snapshot = copy.deepcopy(request)
    fact = _emit(request, emit_train_fn=_ok_spec, caps_fn=lambda: _Caps())
    assert request == snapshot and "caps" not in request["trial_spec"]["train_request"]
    assert fact["executable"] is True


def test_caps_only_added_when_emitter_can_take_it():
    """Injected fakes without a ``caps`` parameter (per inspect.signature) keep working."""
    seen: dict[str, Any] = {}

    def narrow(*, nodes: int, gpus_per_node: int) -> dict[str, Any]:
        seen["nodes"], seen["gpus_per_node"] = nodes, gpus_per_node
        return _ok_spec()

    request = {"trial_spec": {"kind": "train", "trial": "t8", "nodes": 1, "gpus_per_node": 1, "train_request": {}}}
    fact = _emit(request, emit_train_fn=narrow, caps_fn=lambda: _Caps())
    assert seen == {"nodes": 1, "gpus_per_node": 1} and fact["executable"] is True


def test_injected_emitter_without_caps_fn_is_not_probed(monkeypatch):
    import foundationskills.interfaces.fs.emit_trial as mod

    def no_probe() -> Any:
        raise AssertionError("default caps probe must not run for an injected emitter")

    monkeypatch.setattr(mod, "_default_caps_probe", no_probe)
    seen: dict[str, Any] = {}

    def emitter(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return _ok_spec(**kwargs)

    _emit(_json_request(), emit_train_fn=emitter)
    assert "caps" not in seen


def test_emitter_attribute_error_refuses_with_type_name():
    """Found on the GPU run: a junk dataset shard (str, not mapping) raised AttributeError out of emit_train."""

    def junk(**_: Any) -> dict[str, Any]:
        raise AttributeError("'str' object has no attribute 'get'")

    fact = _emit(_json_request(), emit_train_fn=junk, caps_fn=lambda: _Caps())
    assert fact["executable"] is False
    assert fact["missing"][0].startswith("emit_train_bad_request:AttributeError:")


def test_bare_python_is_pinned_to_this_interpreter():
    """Found on the GPU run: a bare "python" in the job resolved to the node's system python (rc=96)."""
    import sys

    fact = _emit(_json_request(), emit_train_fn=_ok_spec, caps_fn=lambda: _Caps())
    assert fact["fs_launch_spec"]["argv"] == [sys.executable, "train.py"]
    assert f"python defaulted to {sys.executable}" in fact["notes"]


def test_explicit_python_is_honoured_and_not_passed_to_the_emitter():
    request = _json_request()
    request["trial_spec"]["train_request"]["python"] = "/opt/env/bin/python"
    seen: dict[str, Any] = {}

    def emitter(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return _ok_spec(**kwargs)

    fact = _emit(request, emit_train_fn=emitter, caps_fn=lambda: _Caps())
    assert "python" not in seen
    assert fact["fs_launch_spec"]["argv"][0] == "/opt/env/bin/python"
    assert not any(n.startswith("python defaulted") for n in fact["notes"])
