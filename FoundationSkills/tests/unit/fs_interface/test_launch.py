
"""Tests for launch(): confirmation gate, refusals, fake-runner submissions."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from foundationskills.core.orchestrator import ConfirmationRequired, plan_hash
from foundationskills.interfaces.fs.launch import LaunchRefused, launch


class Recorder:
    """A subprocess.run-shaped fake capturing calls and replaying results."""

    def __init__(self, *results):
        self.calls = []
        self._results = list(results)

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self._results:
            return self._results.pop(0)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def make_spec(tmp_path, *, executable=True, sbatch=False):
    spec = {
        "stage_name": "sft1",
        "entry": "foundationscale-train",
        "argv": ["python", "-m", "foundationscale.train.cli", "--model", "m"],
        "env": {},
        "sbatch": "#!/bin/bash\n#SBATCH --time=10-00:00:00\ntrue\n" if sbatch else None,
        "dry_run_argv": ["python", "-m", "foundationscale.train.cli", "--model", "m", "--dry-run"],
        "expected_outputs": [f"{tmp_path}/out/run_manifest.json"],
        "executable": executable,
        "missing": None if executable else "missing: pp>1 is REFUSED by installed FS",
        "output_dir": str(tmp_path / "out"),
    }
    return spec


def test_wrong_or_missing_confirm_raises(tmp_path):
    spec = make_spec(tmp_path)
    runner = Recorder()
    with pytest.raises(ConfirmationRequired):
        launch(spec, confirm=None, runner=runner)
    with pytest.raises(ConfirmationRequired):
        launch(spec, confirm="deadbeefdeadbeef", runner=runner)
    assert runner.calls == []  # nothing ran before the gate


def test_non_executable_spec_refused(tmp_path):
    spec = make_spec(tmp_path, executable=False)
    runner = Recorder()
    with pytest.raises(LaunchRefused) as excinfo:
        launch(spec, confirm=plan_hash(spec), runner=runner)
    assert "pp>1" in str(excinfo.value)
    assert runner.calls == []


def test_sbatch_submission_via_bash_lc(tmp_path):
    spec = make_spec(tmp_path, sbatch=True)
    runner = Recorder(
        SimpleNamespace(returncode=0, stdout="dry ok", stderr=""),
        SimpleNamespace(returncode=0, stdout="Submitted batch job 777\n", stderr=""),
    )
    result = launch(spec, confirm=plan_hash(spec), runner=runner)
    assert result["job_id"] == "777"
    assert result["dry_run_rc"] == 0
    # dry-run ran first, then bash -lc "sbatch <file>"
    assert len(runner.calls) == 2
    submit_cmd, _ = runner.calls[1]
    assert submit_cmd[0:2] == ["bash", "-lc"]
    assert submit_cmd[2].startswith("sbatch ")
    sbatch_file = tmp_path / "out" / "launch-sft1.sbatch"
    assert sbatch_file.exists()
    assert "#SBATCH --time=10-00:00:00" in sbatch_file.read_text(encoding="utf-8")


def test_sbatch_without_job_id_refused(tmp_path):
    spec = make_spec(tmp_path, sbatch=True)
    runner = Recorder(
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=1, stdout="sbatch: error: ba-dum", stderr=""),
    )
    with pytest.raises(LaunchRefused):
        launch(spec, confirm=plan_hash(spec), runner=runner)


def test_direct_run_when_no_sbatch(tmp_path):
    spec = make_spec(tmp_path, sbatch=False)
    runner = Recorder(
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = launch(spec, confirm=plan_hash(spec), runner=runner)
    assert result["returncode"] == 0
    assert result["pid"] is None
    assert result["job_id"] is None
    assert runner.calls[1][0][0] == "python"


def test_dry_run_failure_blocks_submit(tmp_path):
    spec = make_spec(tmp_path, sbatch=True)
    runner = Recorder(
        SimpleNamespace(returncode=5, stdout="x" * 2200, stderr=""),
    )
    with pytest.raises(LaunchRefused) as excinfo:
        launch(spec, confirm=plan_hash(spec), runner=runner)
    assert "rc=5" in str(excinfo.value)
    assert len(runner.calls) == 1  # submit never attempted


def test_no_submit_runs_argv_directly(tmp_path):
    spec = make_spec(tmp_path, sbatch=True)
    runner = Recorder(
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = launch(spec, confirm=plan_hash(spec), submit=False, runner=runner)
    assert result["job_id"] is None
    assert result["returncode"] == 0
    assert runner.calls[1][0][0] == "python"
    # no sbatch file was written under no-submit
    assert not (tmp_path / "out" / "launch-sft1.sbatch").exists()
