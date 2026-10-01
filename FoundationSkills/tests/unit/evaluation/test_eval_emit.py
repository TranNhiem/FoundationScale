"""eval emit / dry-run / hash / launch: the GB200 Slurm path, no submission, no lm_eval import."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundationskills import cli
from foundationskills.core.orchestrator import ConfirmationRequired, plan_hash
from foundationskills.interfaces.fs.emit_eval import emit_eval
from foundationskills.interfaces.fs.launch import LaunchRefused, launch

POLICY = "policy_version: 1\ntasks:\n  mmlu: {metric: 'acc,none', num_fewshot: 5}\n"


@pytest.fixture
def req(tmp_path):
    for name in ("ck", "base", "cache"):
        (tmp_path / name).mkdir()
    (tmp_path / "ck" / "config.json").write_text("{}")
    (tmp_path / "base" / "config.json").write_text("{}")
    (tmp_path / "policy.yaml").write_text(POLICY)
    return {"checkpoint": str(tmp_path / "ck"), "base": str(tmp_path / "base"), "benchmarks": ["mmlu"],
            "policy": str(tmp_path / "policy.yaml"), "eval_cache": str(tmp_path / "cache"),
            "out": str(tmp_path / "eval" / "eval_report.json")}


def _ok(_python):
    return "0.4.12"


def test_single_gpu_spec_is_executable_with_gb200_sbatch(req):
    spec = emit_eval(req, version_probe=_ok)
    assert spec["executable"] is True and spec["missing"] is None
    assert spec["argv"][:5] == ["python", "-m", "foundationskills.cli", "eval", "run"]
    assert spec["argv"][spec["argv"].index("--device") + 1] == "cuda:0"
    assert spec["dry_run_argv"] == [*spec["argv"], "--dry-run"]
    sb = spec["sbatch"]
    for needle in ("--time=10-00:00:00", "--exclude=r01dgx02", "--gres=gpu:1", "dev/tcp/master/8081", "--nodes=1"):
        assert needle in sb
    assert spec["env"]["PYTHONPATH"] and spec["expected_outputs"] == [req["out"]]


def test_multi_gpu_parallelizes_and_drops_device(req):
    spec = emit_eval({**req, "device": "cuda:3"}, gpus_per_node=4, version_probe=_ok)
    assert "--parallelize" in spec["argv"] and "--device" not in spec["argv"]
    assert "--gres=gpu:4" in spec["sbatch"] and any("dropped" in n for n in spec["notes"])


def test_multi_node_is_refused_not_approximated(req):
    spec = emit_eval(req, nodes=2, version_probe=_ok)
    assert spec["executable"] is False and "one node" in spec["missing"]


def test_lm_eval_version_is_read_from_the_job_interpreter(req):
    seen = []
    spec = emit_eval(req, python="/opt/env/bin/python", version_probe=lambda p: seen.append(p) or "0.4.11")
    assert seen == ["/opt/env/bin/python"] and spec["argv"][0] == "/opt/env/bin/python"
    assert spec["executable"] is False and "EV-IN-004" in spec["missing"]


def test_missing_checkpoint_blocks_the_spec(req):
    spec = emit_eval({**req, "checkpoint": req["checkpoint"] + "-gone"}, version_probe=_ok)
    assert spec["executable"] is False and "EV-IN-001" in spec["missing"]


def test_local_hardware_emits_no_sbatch(req):
    assert emit_eval(req, hardware_id="local", version_probe=_ok)["sbatch"] is None


def test_launch_needs_the_confirm_hash_then_runs_dry_run_before_sbatch(req):
    spec = emit_eval(req, version_probe=_ok)
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        out = "Submitted batch job 4242\n" if argv[0] == "bash" else "[fskills:eval:dry-run] inputs PASS\n"
        return type("C", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    with pytest.raises(ConfirmationRequired):
        launch(spec, confirm=None, runner=runner)
    assert calls == []
    result = launch(spec, confirm=plan_hash(spec), runner=runner)
    assert calls[0][-1] == "--dry-run" and calls[1][0] == "bash" and result["job_id"] == "4242"


def test_failed_dry_run_blocks_submission(req):
    spec = emit_eval(req, version_probe=_ok)
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return type("C", (), {"returncode": 96, "stdout": "refuse", "stderr": ""})()

    with pytest.raises(LaunchRefused):
        launch(spec, confirm=plan_hash(spec), runner=runner)
    assert len(calls) == 1


def test_cli_dry_run_refuses_a_missing_cache_without_a_report(req, capsys):
    argv = ["eval", "run", "--checkpoint", req["checkpoint"], "--base", req["base"], "--benchmarks", "mmlu",
            "--policy", req["policy"], "--eval-cache", req["eval_cache"] + "-gone", "--out", req["out"], "--dry-run"]
    assert cli.main(argv) == 96
    assert "EV-IN-003" in capsys.readouterr().err and not Path(req["out"]).exists()


def test_cli_emit_writes_spec_and_eval_hash_matches(req, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("foundationskills.interfaces.fs.emit_eval.lm_eval_version_of", _ok)
    spec_path = tmp_path / "spec.json"
    rc = cli.main(["eval", "emit", "--checkpoint", req["checkpoint"], "--base", req["base"], "--benchmarks", "mmlu",
                   "--policy", req["policy"], "--eval-cache", req["eval_cache"], "--out", req["out"],
                   "--gpus", "2", "--spec-out", str(spec_path)])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert cli.main(["eval", "hash", "--spec", str(spec_path)]) == 0
    assert capsys.readouterr().out.strip() == printed["confirm"] == plan_hash(json.loads(spec_path.read_text()))
    assert cli.main(["eval", "launch", "--spec", str(spec_path)]) == 96  # no --confirm
