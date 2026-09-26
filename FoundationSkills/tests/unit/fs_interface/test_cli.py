
"""Tests for the fskills CLI: probe, hash, launch gating, argparse refusal."""
from __future__ import annotations

import json

from foundationskills.cli import main
from foundationskills.core.orchestrator import plan_hash


def write_spec(tmp_path):
    spec = {
        "stage_name": "sft1",
        "entry": "foundationscale-train",
        "argv": ["python", "-m", "foundationscale.train.cli"],
        "env": {},
        "sbatch": None,
        "dry_run_argv": None,
        "expected_outputs": [],
        "executable": False,
        "missing": "missing: deliberately non-executable test spec",
        "output_dir": str(tmp_path / "out"),
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def test_probe_json(capsys):
    rc = main(["probe", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "available" in data
    assert "train_flags" in data
    assert "rl_runnable" in data


def test_hash_prints_plan_hash(tmp_path, capsys):
    path, spec = write_spec(tmp_path)
    rc = main(["hash", "--spec", str(path)])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == plan_hash(spec)
    assert len(out) == 16


def test_hash_unwraps_artifact_envelope(tmp_path, capsys):
    _, spec = write_spec(tmp_path)
    envelope = tmp_path / "artifact.json"
    envelope.write_text(json.dumps({"type": "fs_launch_spec", "id": "x", "payload": spec}), encoding="utf-8")
    rc = main(["hash", "--spec", str(envelope)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == plan_hash(spec)


def test_launch_without_confirm_is_96(tmp_path, capsys):
    path, _ = write_spec(tmp_path)
    rc = main(["launch", "--spec", str(path)])
    assert rc == 96
    assert "confirmation" in capsys.readouterr().err.lower()


def test_launch_wrong_confirm_is_96(tmp_path):
    path, _ = write_spec(tmp_path)
    rc = main(["launch", "--spec", str(path), "--confirm", "0000000000000000", "--no-submit"])
    assert rc == 96


def test_launch_non_executable_with_correct_hash_is_96(tmp_path):
    path, spec = write_spec(tmp_path)
    rc = main(["launch", "--spec", str(path), "--confirm", plan_hash(spec), "--no-submit"])
    assert rc == 96


def test_argparse_error_is_96():
    assert main(["emit"]) == 96               # missing required optionals
    assert main(["frobnicate"]) == 96          # unknown subcommand
    assert main(["launch"]) == 96              # missing --spec


def test_help_exits_zero(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "probe" in out and "launch" in out
