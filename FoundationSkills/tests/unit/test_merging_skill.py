"""Unit tests for MergingSkill (A4.7): probe-first planning and structural + parity verification."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.skills.merging.skill import MergingSkill

PROBES = "foundationskills.skills.merging.skill.merge_probes"
EMIT_PROBES = "foundationskills.interfaces.fs.emit_merge.merge_probes"
FS_ROOT = "foundationskills.interfaces.fs.emit_merge.fs_repo_root"
HEX = re.compile(r"\b[0-9a-f]{12,}\b")

def _probes(base, adapter, out, store="float32", *_a, **_kw):
    gaps, base, adapter, out = [], Path(base), Path(adapter), Path(out)
    if not (base / "config.json").exists():
        gaps.append(f"base {base} is not a model directory (no config.json)")
    if not (adapter / "adapter_config.json").exists():
        gaps.append(f"adapter {adapter} has no adapter_config.json/weights")
    else:
        cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        declared = Path(str(cfg["base_model_name_or_path"])).expanduser()
        if declared != base.expanduser():
            gaps.append(f"adapter {adapter} was trained on {declared} (adapter_config.json)")
    if out.exists() and any(out.iterdir()):
        gaps.append(f"out {out} exists and is not empty (a merge never overwrites)")
    return gaps

def _doc(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("payload", "data", "artifact"):
        if isinstance(doc.get(key), dict):
            return doc, doc[key]
    return doc, doc

def _entry(pay, doc):
    for src in (pay, doc):
        for key in ("entry", "entries", "fs_entry"):
            val = src.get(key)
            val = val[0] if isinstance(val, (list, tuple)) else val
            if isinstance(val, str) and val:
                return val
    return ""

def _text(res):
    parts = [str(getattr(res, "refusal", "") or "")]
    for f in res.findings or ():
        parts += [str(getattr(f, "rule", "") or ""), str(getattr(f, "message", "") or ""), repr(f)]
    return " ".join(parts)

def _plan(**over):
    req = {"action": "plan", "base": str(C.base), "adapter": str(C.adapter), "out": str(C.out),
           "store_dtype": "float32", "device_map": "cpu"}
    req.update(over)
    return req

def _verify_spec(checks=None):
    results = C.root / "specs" / "merge_results.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    if checks is not None:
        results.write_text(json.dumps({"verification": {"checks": checks}}), encoding="utf-8")
    merge = {"base": str(C.base), "adapter": str(C.adapter), "out": str(C.out),
             "results": str(results), "store_dtype": "float32"}
    path = results.parent / "fs_launch_spec.merge.json"
    path.write_text(json.dumps({"merge": merge}), encoding="utf-8")
    return path

def _empty(ctx):
    return not any(p.is_file() for p in Path(ctx.artifacts_dir).rglob("*"))

TRUE, FALSE = {"param_count": True, "no_lora_weights_left_on_disk": True}, None

@pytest.fixture()
def case(tmp_path, monkeypatch):
    global C
    base = tmp_path / "models" / "gemma-4-12B-it"
    base.mkdir(parents=True)
    (base / "config.json").write_text('{"model_type": "gemma"}', encoding="utf-8")
    (base / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "runs" / "sft" / "final"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base), "peft_type": "LORA"}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
    script = tmp_path / "fscopy" / "examples" / "vlm-2xH200" / "scripts" / "merge_lora.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fake FS merge script\n", encoding="utf-8")
    monkeypatch.setattr(PROBES, _probes)
    monkeypatch.setattr(EMIT_PROBES, _probes)
    monkeypatch.setattr(FS_ROOT, lambda: tmp_path / "fscopy")
    ctx = SkillContext(workdir=tmp_path)
    Path(ctx.artifacts_dir).mkdir(parents=True, exist_ok=True)
    C = SimpleNamespace(ctx=ctx, root=tmp_path, base=base, adapter=adapter, out=tmp_path / "merged" / "m1")
    return C

def test_plan_pass_writes_launch_spec(case):
    res = MergingSkill().execute(_plan(), case.ctx)
    assert res.status.value == "PASS" and not _text(res)
    (ref,) = res.artifacts
    doc, pay = _doc(ref.path)
    assert _entry(pay, doc) == "fskills-merge" and "fs_launch_spec" in json.dumps(doc)
    assert not HEX.search(json.dumps(pay)), "the plan payload must never carry the confirm hash"
    assert "--confirm <hash>" in res.payload["next"]

def test_plan_bfloat16_warns_meho001(case):
    res = MergingSkill().execute(_plan(store_dtype="bfloat16"), case.ctx)
    assert res.status.value == "PASS" and "ME-HO-001" in _text(res)

@pytest.mark.parametrize("how,rule", [("no-base", "ME-IN-001"), ("no-adapter", "ME-IN-001"),
                                      ("other-base", "ME-IN-002"), ("nonempty-out", "ME-IN-003")])
def test_plan_refuses_input_gaps(case, how, rule):
    if how == "no-base":
        (case.base / "config.json").unlink()
    elif how == "no-adapter":
        (case.adapter / "adapter_config.json").unlink()
    elif how == "other-base":
        (case.adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(case.root / "other")}), encoding="utf-8")
    else:
        case.out.mkdir(parents=True)
        (case.out / "half.done").write_text("x", encoding="utf-8")
    res = MergingSkill().execute(_plan(), case.ctx)
    assert res.status.value == "REFUSED" and rule in _text(res)
    assert _empty(case.ctx), "96 REFUSED writes nothing"

def test_verify_without_results_refuses_m005(case):
    res = MergingSkill().execute({"action": "verify", "spec": str(_verify_spec())}, case.ctx)
    assert res.status.value == "REFUSED" and "ME-IN-005" in _text(res)
    assert _empty(case.ctx)

def test_verify_structural_check_red_meho002(case):
    spec = _verify_spec({"param_count": True, "no_lora_weights_left_on_disk": False})
    skill = MergingSkill(lambda *_: pytest.fail("parity must not run"))
    res = skill.execute({"action": "verify", "spec": str(spec)}, case.ctx)
    assert res.status.value == "RED" and "ME-HO-002" in _text(res)
    assert _empty(case.ctx), "RED writes no checkpoint artifact"

def test_verify_parity_red_meho003(case):
    skill = MergingSkill(lambda *_: {"verdict": "RED", "reason": "x"})
    res = skill.execute({"action": "verify", "spec": str(_verify_spec(TRUE))}, case.ctx)
    assert res.status.value == "RED" and "ME-HO-003" in _text(res)
    assert res.payload["parity"]["reason"] == "x" and _empty(case.ctx)

def test_verify_parity_pass_writes_checkpoint(case):
    skill = MergingSkill(lambda *_: {"verdict": "PASS", "argmax_agreement": 1.0, "ratio": 4e-5})
    res = skill.execute({"action": "verify", "spec": str(_verify_spec(TRUE))}, case.ctx)
    assert res.status.value == "PASS" and all(res.payload["structural_checks"].values())
    assert res.payload["path"] == str(C.out) and res.payload["store_dtype"] == "float32"
    (ref,) = res.artifacts
    doc, pay = _doc(ref.path)
    assert ref.type == "checkpoint" and pay["stage"] == "merge" and pay["run_manifest"] is None
    assert pay["parity"]["verdict"] == "PASS"

def test_verify_parity_disabled_unmeasured_meho004(case):
    skill = MergingSkill(lambda *_: pytest.fail("parity off must not measure"))
    res = skill.execute({"action": "verify", "spec": str(_verify_spec(TRUE)), "parity": False}, case.ctx)
    assert res.status.value == "UNMEASURED" and "ME-HO-004" in _text(res)
    assert res.payload["parity"] is None and _empty(case.ctx)
