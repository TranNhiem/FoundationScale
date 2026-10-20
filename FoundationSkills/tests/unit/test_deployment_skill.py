"""Unit tests for skills/deployment/skill.py (A4.9): plan + verify with no GPU, no server and no network."""

from __future__ import annotations

import json
from pathlib import Path

from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills.deployment.skill import DeploymentSkill

GPU, PORT = "GPU-447bb30a", 18841
RULES = tuple(f"DP-{p}-{i:03d}" for p in ("IN", "HO") for i in range(1, 5))


def _model(tmp_path, name, model_type="gemma4_unified", config=True):
    d = tmp_path / name
    d.mkdir()
    if config:
        (d / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return str(d)


def _adapter(tmp_path, base):
    d = tmp_path / "adapter"
    d.mkdir(exist_ok=True)
    cfg = {"base_model_name_or_path": base, "r": 16}
    (d / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(d)


def _http(models=("model",), text="OK", dead=False):
    def call(url, body=None, timeout=120):
        if dead:
            raise ConnectionRefusedError("endpoint connection refused")
        if url.endswith("/v1/models"):
            return {"data": [{"id": n} for n in models]}
        return {"choices": [{"message": {"content": text}}]}
    return call


def _placement(gpu=GPU, measured=True):
    def call(port):
        if not measured:
            return {"measured": False, "reason": "x"}
        return {"measured": True, "gpus": [gpu], "server_pids": ["4321"]}
    return call


def _skill(monkeypatch, **kw):
    monkeypatch.setattr("foundationskills.interfaces.fs.emit_serve.port_free", lambda *_a: True)
    http = _http(kw.get("models", ("model",)), kw.get("text", "OK"), kw.get("dead", False))
    return DeploymentSkill(http=http, placement=_placement(kw.get("gpu", GPU), kw.get("measured", True)))


def _run(monkeypatch, tmp_path, request, **kw):
    return _skill(monkeypatch, **kw).execute(request, SkillContext(workdir=tmp_path))


def _rules(res):
    blob = " ".join(repr(x) for x in (res, getattr(res, "findings", ()), getattr(res, "payload", {}) or {}))
    return {r for r in RULES if r in blob}


def _artifact(res, kind):
    files = [Path(a.path) for a in res.artifacts if kind in Path(a.path).name]
    assert files, f"no {kind} artifact in {res.artifacts}"
    data = json.loads(files[0].read_text(encoding="utf-8"))
    return data, (data.get("payload", data) if isinstance(data, dict) else {})


def _argv(payload):
    argv = payload.get("argv", [])
    return argv if isinstance(argv, str) else " ".join(str(a) for a in argv)


def _verify_spec(tmp_path, family="gemma4_unified", mode="full", names=("model",)):
    serve = {"endpoint": "http://127.0.0.1:18841", "served_names": list(names), "family": family, "mode": mode,
             "modality": "text", "gpu": GPU, "port": PORT}
    spec = tmp_path / "fs_launch_spec.serve.json"
    spec.write_text(json.dumps({"payload": {"serve": serve}}), encoding="utf-8")
    return {"action": "verify", "spec": str(spec)}


def test_plan_full_gemma_pass_with_gpu_confined_argv(tmp_path, monkeypatch):
    model = _model(tmp_path, "gemma-4-12B-sft")
    res = _run(monkeypatch, tmp_path, {"action": "plan", "model": model, "gpu": GPU, "port": PORT})
    assert res.status is Status.PASS
    doc, payload = _artifact(res, "fs_launch_spec")
    assert payload.get("argv") and f"CUDA_VISIBLE_DEVICES={GPU}" in _argv(payload)
    assert "fskills-serve" in json.dumps(doc)          # the fs_launch_spec entry


def test_plan_unmeasured_family_is_unmeasured(tmp_path, monkeypatch):
    model = _model(tmp_path, "nemotron-nano", "nemotron_h")
    res = _run(monkeypatch, tmp_path, {"action": "plan", "model": model, "gpu": GPU, "port": PORT})
    assert res.status is Status.UNMEASURED and "DP-HO-001" in _rules(res)


def test_plan_lora_flags_in_argv(tmp_path, monkeypatch):
    model = _model(tmp_path, "gemma-4-12B-it")
    for extra, flag in (({}, "--enable-lora"), ({"modality": "vision"}, "--enable-tower-connector-lora")):
        request = {"action": "plan", "model": model, "adapter": _adapter(tmp_path, model), "gpu": GPU, "port": PORT,
                   **extra}
        res = _run(monkeypatch, tmp_path, request)
        _doc, payload = _artifact(res, "fs_launch_spec")
        assert flag in _argv(payload), flag


def test_plan_input_gaps_are_refused(tmp_path, monkeypatch):
    model = _model(tmp_path, "gemma-4-12B-sft")
    other = _model(tmp_path, "gemma-4-31B-it", "nemotron_h")
    base = {"action": "plan", "model": model, "gpu": GPU, "port": PORT}
    cases = (({**base, "gpu": ""}, "DP-IN-003"),                        # no leased GPU UUID
             ({**base, "adapter": _adapter(tmp_path, other)}, "DP-IN-002"),  # adapter on another base
             ({**base, "model": _model(tmp_path, "merged-missing", config=False)}, "DP-IN-001"))
    for request, rule in cases:
        res = _run(monkeypatch, tmp_path, request)
        assert res.status is Status.REFUSED and rule in _rules(res), rule


def test_verify_healthy_full_pass_and_serving_spec(tmp_path, monkeypatch):
    res = _run(monkeypatch, tmp_path, _verify_spec(tmp_path))
    assert res.status is Status.PASS
    _doc, record = _artifact(res, "serving_spec")
    assert record["verdict"] == "PASS" and record["checks"]["placement"]["measured"] is True


def test_verify_wrong_gpu_placement_is_red(tmp_path, monkeypatch):
    res = _run(monkeypatch, tmp_path, _verify_spec(tmp_path), gpu="GPU-29098e5e")
    assert res.status is Status.RED and "DP-HO-004" in _rules(res)


def test_verify_missing_served_name_is_red(tmp_path, monkeypatch):
    res = _run(monkeypatch, tmp_path, _verify_spec(tmp_path, names=("model", "model-extra")))
    assert res.status is Status.RED and "DP-HO-003" in _rules(res)


def test_verify_healthy_lora_is_unmeasured(tmp_path, monkeypatch):
    res = _run(monkeypatch, tmp_path, _verify_spec(tmp_path, mode="lora"))
    assert res.status is Status.UNMEASURED and "DP-HO-002" in _rules(res)


def test_verify_unmeasured_placement_never_passes(tmp_path, monkeypatch):
    res = _run(monkeypatch, tmp_path, _verify_spec(tmp_path), measured=False)
    assert res.status is Status.UNMEASURED and "DP-HO-001" in _rules(res)


def test_verify_dead_endpoint_is_refused(tmp_path, monkeypatch):
    res = _run(monkeypatch, tmp_path, _verify_spec(tmp_path), dead=True)
    assert res.status is Status.REFUSED and "DP-IN-004" in _rules(res)
