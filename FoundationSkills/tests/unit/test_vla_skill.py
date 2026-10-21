"""Unit tests for the VLA skill: plan renders an FS launch spec, verify judges FS's own report."""

import json
from pathlib import Path

import foundationskills.skills.vla.skill as vla_mod
from foundationskills.core.contract import SkillContext
from foundationskills.skills.vla.skill import VlaSkill

MID = "openpi-pi05-libero"


def _tag(res):
    words = " ".join({str(res.status), getattr(res.status, "name", ""),
                      str(getattr(res.status, "value", ""))}).upper().split()
    for tag in ("UNMEASURED", "REFUSED", "RED", "PASS"):
        if tag in words:
            return tag
    raise AssertionError(f"unknown status: {res.status}")


def _text(res):
    parts = [str(res.refusal), str(res.payload), str(res.status)]
    parts += [str(f) for f in res.findings]
    return " ".join(parts)


def _kinds(res):
    out = []
    for a in res.artifacts:
        out.append(str(getattr(a, "kind", None) or getattr(a, "type", None) or getattr(a, "path", None) or a))
    return out


def _report(tmp_path: Path, *, X=98.4, E=0, rate=True, backend="openpi", suite="libero_spatial"):
    body = {"backend": backend, "plan": {"suite": suite}, "episode_errors": E, "measured": E == 0,
            "trials": 500, "successes": int(round(X * 5)),
            "wilson95_pct": [max(0.0, X - 1.0), min(100.0, X + 1.0)]}
    if rate:
        body["success_rate_pct"] = X
    path = tmp_path / "r.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return str(path)


def _exec(tmp_path, request):
    return VlaSkill().execute(request, SkillContext(workdir=tmp_path))


def _plan(tmp_path, **kw):
    req = {"action": "plan", "model_id": MID, "mode": "eval", "out": str(tmp_path / "r.json")}
    req.update(kw)
    return _exec(tmp_path, req)


def test_plan_eval_pass_renders_fs_launch_spec(tmp_path):
    res = _plan(tmp_path)
    assert _tag(res) == "PASS", _text(res)
    spec = res.payload["spec"]
    argv = [str(a) for a in spec["argv"]]
    assert spec["entry"] == "fskills-vla-eval"
    assert "foundationscale.vla.eval" in argv, argv
    assert "--suite" in argv and "libero_spatial" in argv, argv
    assert any("fs_launch_spec" in k for k in _kinds(res)), _kinds(res)


def test_plan_finetune_unmeasured_names_adapter(tmp_path):
    res = _plan(tmp_path, mode="finetune", checkpoint=str(tmp_path / "cp.pt"))
    assert _tag(res) == "UNMEASURED", _text(res)
    assert "VL-HO-002" in _text(res)
    assert str(res.payload["fs_adapter"]).endswith("OpenPiFinetuneSpec")


def test_unregistered_policy_refused_lists_supported(tmp_path):
    res = _plan(tmp_path, model_id="octo-base-libero")
    assert _tag(res) == "REFUSED", _text(res)
    assert "VL-IN-001" in _text(res)
    assert "openpi-pi05-libero" in _text(res)


def test_plan_eval_out_is_existing_dir_refused(tmp_path):
    res = _plan(tmp_path, out=str(tmp_path))
    assert _tag(res) == "REFUSED", _text(res)
    assert "VL-IN-003" in _text(res)


def test_verify_pass_writes_vla_verdict(tmp_path):
    report = _report(tmp_path, X=98.4, E=0)
    res = _exec(tmp_path, {"action": "verify", "model_id": MID, "report": report})
    assert _tag(res) == "PASS", _text(res)
    assert res.payload["verdict"] == "PASS" and res.payload["measured"] == 98.4
    assert any("vla_verdict" in k for k in _kinds(res)), _kinds(res)


def test_verify_out_of_tolerance_red(tmp_path):
    report = _report(tmp_path, X=94.0, E=0)
    res = _exec(tmp_path, {"action": "verify", "model_id": MID, "report": report})
    assert _tag(res) == "RED", _text(res)
    assert "VL-HO-003" in _text(res) and res.payload["verdict"] == "RED"


def test_verify_episode_errors_unmeasured(tmp_path):
    report = _report(tmp_path, X=0.0, E=500)
    res = _exec(tmp_path, {"action": "verify", "model_id": MID, "report": report})
    assert _tag(res) == "UNMEASURED", _text(res)
    assert "VL-HO-001" in _text(res) and res.payload["verdict"] == "UNMEASURED"


def test_verify_missing_success_rate_refused(tmp_path):
    report = _report(tmp_path, rate=False)
    res = _exec(tmp_path, {"action": "verify", "model_id": MID, "report": report})
    assert _tag(res) == "REFUSED", _text(res)
    assert "VL-IN-004" in _text(res)


def test_verify_wrong_backend_refused(tmp_path):
    report = _report(tmp_path, backend="gr00t")
    res = _exec(tmp_path, {"action": "verify", "model_id": MID, "report": report})
    assert _tag(res) == "REFUSED", _text(res)
    assert "VL-IN-005" in _text(res)


def test_registry_not_importable_refused(tmp_path, monkeypatch):
    def _boom(*_a, **_k):
        raise ImportError("foundationscale missing here")

    monkeypatch.setattr(vla_mod, "_registry", _boom)
    res = _plan(tmp_path)
    assert _tag(res) == "REFUSED", _text(res)
    assert "VL-IN-002" in _text(res)
