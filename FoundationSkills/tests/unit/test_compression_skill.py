"""Unit tests for CompressionSkill (A4.8 v1): preflight mapping, probe-first planning, verify via judge.py.

No GPU and no subprocess: every test injects a fake preflight runner, and verify reads results files written here.
"""

import json
from pathlib import Path

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.core.orchestrator import plan_hash
from foundationskills.core.status import Status
from foundationskills.skills.compression.skill import CompressionSkill

PASS_CAP = (0, {"overall": "PASS", "checks": [{"name": "real_gpu_execution", "status": "PASS"}]})


class FakePreflight:
    def __init__(self, code=0, cap=None):
        self.code, self.cap, self.argv = code, (cap if cap is not None else PASS_CAP[1]), None

    def __call__(self, argv, timeout_s):
        self.argv = argv
        return self.code, self.cap


def _rules(res):
    return [f.rule_id for f in res.findings]


@pytest.fixture
def stage(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "w.safetensors").write_bytes(b"\0")
    text = tmp_path / "heldout.txt"
    text.write_text("held-out text\n", encoding="utf-8")
    return {"action": "plan", "model": str(model), "artifact_root": str(tmp_path / "pf"), "text_file": str(text),
            "results": str(tmp_path / "out" / "rtn.json"), "bits": 4, "budget_ppl_ratio": 1.15}


def _run(request, tmp_path, runner=None):
    skill = CompressionSkill(preflight_runner=runner or FakePreflight())
    return skill.execute(request, SkillContext(workdir=tmp_path / "wd"))


def test_unknown_action_is_refused(tmp_path):
    res = _run({"action": "quantize"}, tmp_path)
    assert res.status is Status.REFUSED and "CP-IN-007" in _rules(res)


@pytest.mark.parametrize("change,rule", [({"format": "gptq"}, "CP-IN-003"), ({"budget_ppl_ratio": None}, "CP-IN-004"),
                                         ({"budget_ppl_ratio": 2.5}, "CP-IN-004"), ({"budget_ppl_ratio": 1.0}, "CP-IN-004")])
def test_plan_refuses_format_and_budget(stage, tmp_path, change, rule):
    request = {k: v for k, v in {**stage, **change}.items() if v is not None}
    res = _run(request, tmp_path)
    assert res.status is Status.REFUSED and rule in _rules(res)


def test_plan_refuses_a_non_checkpoint(stage, tmp_path):
    (Path(stage["model"]) / "config.json").unlink()
    res = _run(stage, tmp_path)
    assert res.status is Status.REFUSED and "CP-IN-001" in _rules(res)


def test_plan_refuses_an_existing_results_file(stage, tmp_path):
    Path(stage["results"]).parent.mkdir(parents=True)
    Path(stage["results"]).write_text("{}", encoding="utf-8")
    res = _run(stage, tmp_path)
    assert res.status is Status.REFUSED and "CP-IN-001" in _rules(res)


@pytest.mark.parametrize("code,cap,status,rule", [
    (1, {"checks": [{"name": "real_gpu_execution", "status": "FAIL"}]}, Status.REFUSED, "CP-IN-002"),
    (2, None, Status.UNMEASURED, "CP-HO-010"),
    (0, {"overall": "PASS_WITH_WARNINGS", "checks": [{"name": "x", "status": "UNKN"}]}, Status.UNMEASURED, "CP-HO-010"),
    (0, PASS_CAP[1], Status.PASS, None),
])
def test_check_maps_preflight(stage, tmp_path, code, cap, status, rule):
    res = _run({**stage, "action": "check"}, tmp_path, FakePreflight(code, cap))
    assert res.status is status
    assert (rule in _rules(res)) if rule else not _rules(res)
    if code == 1:
        assert "real_gpu_execution" in res.refusal


def test_plan_happy_path_emits_a_hashless_spec(stage, tmp_path):
    fake = FakePreflight()
    res = _run(stage, tmp_path, fake)
    assert res.status is Status.PASS
    assert res.artifacts and Path(res.artifacts[0].path).name.endswith("fs_launch_spec.compress.json")
    spec = res.payload["spec"]
    assert spec["entry"] == "fskills-compress" and spec["executable"] and spec["env"]["PYTHONNOUSERSITE"] == "1"
    assert spec["compress"]["budget_ppl_ratio"] == 1.15 and "fskills hash" in res.payload["next"]
    written = json.loads(Path(res.artifacts[0].path).read_text(encoding="utf-8"))
    confirm = plan_hash(written["payload"] if "payload" in written and "type" in written else written)
    assert confirm not in json.dumps(res.to_dict())  # the hash is read by the user, never echoed
    assert "--planned-formats" in fake.argv and "rtnste_int4" in fake.argv


def test_plan_stops_when_preflight_cannot_tell(stage, tmp_path):
    res = _run(stage, tmp_path, FakePreflight(0, {"overall": "PASS_WITH_WARNINGS",
                                                  "checks": [{"name": "x", "status": "UNKN"}]}))
    assert res.status is Status.UNMEASURED and "CP-HO-010" in _rules(res) and not res.artifacts


def _verify_files(tmp_path, **over):
    results = tmp_path / "rtn.json"
    good = {"schema": "fs.compress.rtn_results.v1", "status": "MEASURED", "model": "m", "bits": 4,
            "group_size": 128, "text_sha256": "t", "instrument_certified": True, "instrument_rel_diff": 0.001,
            "base_ppl_a": 9.8, "compressed_ppl": 10.1, "ppl_ratio": 10.1 / 9.8, "quantized_linears": 5,
            "changed_fraction": 0.9, "effective_bits_total": 5.0}
    if over.pop("missing", False) is False:
        results.write_text(json.dumps({**good, **over}), encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"payload": {"compress": {"model": "m", "bits": 4, "group_size": 128,
                                                         "text_sha256": "t", "budget_ppl_ratio": 1.15,
                                                         "results": str(results)}}}), encoding="utf-8")
    return {"action": "verify", "spec": str(spec)}


@pytest.mark.parametrize("over,status,rule", [
    ({}, Status.PASS, None),
    ({"ppl_ratio": 1.31}, Status.RED, "CP-HO-011"),
    ({"instrument_certified": False}, Status.UNMEASURED, "CP-HO-001"),
    ({"status": "UNMEASURED", "reason": "no CUDA device"}, Status.UNMEASURED, "CP-HO-003"),
    ({"missing": True}, Status.REFUSED, "CP-IN-005"),
    ({"bits": 8}, Status.REFUSED, "CP-IN-006"),
])
def test_verify_verdicts(tmp_path, over, status, rule):
    res = _run(_verify_files(tmp_path, **over), tmp_path)
    assert res.status is status
    if rule:
        assert rule in _rules(res)
    if status is not Status.REFUSED:
        assert "CP-HO-005" in _rules(res)  # efficiency is never measured by v1, PASS included
        assert any(Path(a.path).name.endswith("compression_report.rtn.json") for a in res.artifacts)


def test_verify_refuses_a_missing_spec(tmp_path):
    res = _run({"action": "verify", "spec": str(tmp_path / "nope.json")}, tmp_path)
    assert res.status is Status.REFUSED and "CP-IN-005" in _rules(res)


def test_every_rule_has_a_must_fire_fixture():
    skill = CompressionSkill()
    assert {r.rule_id for r in skill.rules} == set(skill.must_fire_fixtures())
