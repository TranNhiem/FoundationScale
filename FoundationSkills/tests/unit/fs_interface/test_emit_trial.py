"""Trial emit tests: eval/train emission over the FS emitters and the AR-LN-006/AR-LN-003 gated submit.

Nothing here touches Slurm or a socket: emitters, runners and the launch call are
injected, and sbatch text is only ever produced by the real ``render_sbatch``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import foundationskills.interfaces.fs.emit_trial as emit_trial_mod
from foundationskills.core.orchestrator import plan_hash
from foundationskills.skills.evaluation.runner import PINNED_LM_EVAL
from foundationskills.interfaces.fs.emit_trial import emit_trial, submit_trial
from foundationskills.interfaces.fs.launch import LaunchRefused, launch
from foundationskills.interfaces.fs.sbatch import render_sbatch


def _sbatch(argv: list[str], *, hardware_id: str = "gb200", nodes: int = 1, gpus_per_node: int = 1,
            partition: str | None = "batch") -> str:
    """Render with the real renderer: --time and the IMEX preamble are estate rules, never test text."""
    return render_sbatch(
        argv,
        env={},
        hardware_id=hardware_id,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        job_name="fs-trial",
        log_dir="/tmp/fskills/logs",
        launcher=Path(str(argv[0])).name,
        partition=partition,
    )


class _Recorder:
    """``subprocess.run``-shaped stand-in: records argv, invents nothing, touches no scheduler."""

    def __init__(self, *, stdout: str = "", returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, argv, **_kwargs):
        self.calls.append([str(a) for a in argv])
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr="")


def _version_probe(*_args, **_kwargs) -> str:
    """``lm_eval_version_of`` stand-in: tests import no lm_eval and spawn no subprocess."""
    return PINNED_LM_EVAL


def _eval_request(tmp_path: Path) -> dict:
    for name in ("ck", "base", "cache"):
        (tmp_path / name).mkdir()
    (tmp_path / "ck" / "config.json").write_text("{}")
    (tmp_path / "base" / "config.json").write_text("{}")
    (tmp_path / "policy.yaml").write_text("policy_version: 1\ntasks:\n  mmlu: {metric: 'acc,none', num_fewshot: 5}\n")
    return {"checkpoint": str(tmp_path / "ck"), "base": str(tmp_path / "base"), "benchmarks": ["mmlu"],
            "policy": str(tmp_path / "policy.yaml"), "eval_cache": str(tmp_path / "cache"),
            "out": str(tmp_path / "eval" / "eval_report.json")}


def _train_request(tmp_path: Path) -> dict:
    return {
        "stage": "skills",
        "dataset": str(tmp_path / "data"),
        "model": str(tmp_path / "model"),
        "output_dir": str(tmp_path / "train-out"),
        "hardware": "gb200",
        "nodes": 1,
        "gpus_per_node": 8,
        "caps": {"max_steps": 4},
    }


def _request(tmp_path: Path, *, kind: str = "eval_only", **over) -> dict:
    """One submit-shaped request: ``{trial_spec, campaign_spec}``."""
    trial = {
        "trial": "c1-t2",
        "role": "baseline",
        "kind": kind,
        "delta": {},
        "seed": 7,
        "nodes": 1,
        "gpus_per_node": 1,
        "partition": "batch",
        "gpu_hours_est": 2.0,
    }
    if kind == "train":
        trial["train_request"] = _train_request(tmp_path)
    else:
        trial["eval_request"] = _eval_request(tmp_path)
    trial.update(over)
    return {
        "trial_spec": trial,
        "campaign_spec": {"campaign": "c1", "budget": {"max_runs": 4, "gpu_hours_total": 24.0}},
    }


def _fs_launch_spec(tmp_path: Path) -> dict:
    """One rendered, executable fs_launch_spec staged under tmp_path."""
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = ["python", "-m", "foundationskills.cli", "eval", "run", "--benchmarks", "mmlu"]
    return {
        "argv": argv,
        "env": {},
        "sbatch": _sbatch(argv),
        "executable": True,
        "missing": [],
        "notes": [],
        "output_dir": str(output_dir),
        "stage_name": "fs-trial",
    }


def _fabric(state: str = "ready", reason: str = "fabric_ready") -> dict:
    """Probe-contract fabric fact; injected -- no socket is opened here."""
    return {"state": state, "reason": reason, "cached": False, "age_s": 0.0}


class TestEmitTrial:
    def test_eval_only_emits_eval_run_argv_and_sbatch_gates(self, tmp_path):
        request = _request(tmp_path)
        fact = emit_trial(request, probe=_version_probe)
        spec = fact["fs_launch_spec"]
        argv = [str(a) for a in spec["argv"]]
        # an unnamed interpreter renders as this process's own, never a PATH-dependent bare "python"
        assert argv[:5] == [sys.executable, "-m", "foundationskills.cli", "eval", "run"]
        sbatch = str(spec["sbatch"])
        assert "--time=10-00:00:00" in sbatch  # estate wall time, asserted on the render that ships
        assert "/dev/tcp/master/8081" in sbatch  # in-job IMEX probe on gb200 ...
        assert "exit 96" in sbatch  # ... runs REFUSED (96) when the fabric refuses
        assert fact["trial_spec"] == request["trial_spec"]
        assert fact["confirm"] == plan_hash(spec)
        assert fact["executable"] is (spec.get("executable") is True)
        assert fact["missing"] == [str(m) for m in (spec.get("missing") or [])]
        assert fact["notes"] == [str(n) for n in (spec.get("notes") or [])]
        assert fact["drops"] == [str(d) for d in (spec.get("drops") or [])]

    def test_eval_only_forwards_nodes_gpus_hardware_and_probe(self, tmp_path):
        seen: list[dict] = []

        def fake_eval(
            eval_request, *, nodes=1, gpus_per_node=1, hardware_id="gb200", python="python", version_probe=None,
            partition=None,
        ):
            seen.append(
                {
                    "request": eval_request,
                    "nodes": nodes,
                    "gpus_per_node": gpus_per_node,
                    "hardware_id": hardware_id,
                    "version_probe": version_probe,
                }
            )
            argv = [python, "-m", "foundationskills.cli", "eval", "run"]
            return {
                "argv": argv,
                "env": {},
                "sbatch": _sbatch(argv, hardware_id=hardware_id, nodes=nodes, gpus_per_node=gpus_per_node),
                "executable": True,
                "missing": [],
                "notes": ["probed lm_eval 0.4.3"],
                "drops": ["skipped:limit"],
            }

        request = _request(tmp_path, nodes=2, gpus_per_node=4)
        fact = emit_trial(request, hardware_id="gb200", emit_eval_fn=fake_eval, probe=_version_probe)
        assert len(seen) == 1
        assert seen[0]["request"] == request["trial_spec"]["eval_request"]  # a copy: "python" is popped
        assert (seen[0]["nodes"], seen[0]["gpus_per_node"], seen[0]["hardware_id"]) == (2, 4, "gb200")
        assert seen[0]["version_probe"] is _version_probe  # probe facts are advisory: they gate nothing here
        assert fact["notes"] == [f"python defaulted to {sys.executable}", "probed lm_eval 0.4.3"]
        assert fact["drops"] == ["skipped:limit"]
        assert fact["executable"] is True
        assert fact["confirm"] == plan_hash(fact["fs_launch_spec"])

    def test_train_calls_emit_train_with_trial_topology(self, tmp_path):
        seen: dict = {}

        def fake_emit_train(*args, **train):
            seen["args"], seen["train"] = args, train
            return {
                "argv": ["torchrun", "--nproc_per_node=8", "train.py"],
                "env": {},
                "sbatch": _sbatch(["torchrun", "--nproc_per_node=8", "train.py"]),
                "executable": True,
                "missing": [],
                "notes": ["train"],
                "drops": [],
            }

        request = _request(tmp_path, kind="train")
        fact = emit_trial(request, emit_train_fn=fake_emit_train)
        assert seen["args"] == ()
        ts = request["trial_spec"]
        # opaque passthrough except topology: the gated trial_spec nodes/gpus_per_node win (G2)
        assert seen["train"] == {**ts["train_request"], "nodes": ts["nodes"], "gpus_per_node": ts["gpus_per_node"]}
        assert fact["fs_launch_spec"]["argv"] == ["torchrun", "--nproc_per_node=8", "train.py"]
        assert fact["notes"][-1] == "train" and fact["drops"] == []
        assert all(n.startswith("overrode train_request.") for n in fact["notes"][:-1])
        assert fact["executable"] is True and fact["missing"] == []
        assert fact["confirm"] == plan_hash(fact["fs_launch_spec"])

    def test_missing_emit_train_is_a_refused_fact(self, tmp_path, monkeypatch):
        monkeypatch.setattr(emit_trial_mod, "_load_emit_train", lambda: None)  # repo ships no emit_train
        fact = emit_trial(_request(tmp_path, kind="train"))
        assert fact["fs_launch_spec"] == {"state": "REFUSED", "reason": "emit_train_missing"}
        assert fact["executable"] is False
        assert fact["missing"] == ["emit_train_missing"]
        assert fact["drops"] == ["emit_train_missing"]  # a skip is a failure: named and counted
        assert fact["confirm"] == plan_hash({"state": "REFUSED", "reason": "emit_train_missing"})
        assert fact["trial_spec"]["trial"] == "c1-t2"

    def test_default_emit_train_import_missing_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "foundationskills.interfaces.fs.emit_train", None)
        fact = emit_trial(_request(tmp_path, kind="train"))
        assert fact["fs_launch_spec"] == {"state": "REFUSED", "reason": "emit_train_missing"}
        assert fact["drops"] == ["emit_train_missing"]

    def test_unknown_kind_is_a_refused_fact(self, tmp_path):
        request = _request(tmp_path, kind="train")
        request["trial_spec"]["kind"] = "eval"
        fact = emit_trial(request, emit_train_fn=lambda **_train: {})
        assert fact["fs_launch_spec"]["reason"] == "trial_kind_unknown:eval"
        assert fact["executable"] is False and fact["drops"] == ["trial_kind_unknown:eval"]

    def test_render_bug_missing_time_is_a_value_error(self, tmp_path):
        def fake_eval(eval_request, **_kwargs):
            return {
                "argv": ["python", "-m", "foundationskills.cli", "eval", "run"],
                "env": {},
                "sbatch": "#!/bin/bash\nsrun python\n",
                "executable": True,
                "missing": [],
                "notes": [],
            }

        with pytest.raises(ValueError, match="10-00:00:00"):
            emit_trial(_request(tmp_path), emit_eval_fn=fake_eval)

    def test_local_hardware_without_sbatch_is_not_a_render_bug(self, tmp_path):
        def fake_eval(eval_request, **_kwargs):
            return {
                "argv": ["python", "-m", "foundationskills.cli", "eval", "run"],
                "env": {},
                "sbatch": None,
                "executable": False,
                "missing": ["lm_eval not importable"],
                "notes": ["local run"],
            }

        fact = emit_trial(_request(tmp_path), hardware_id="local", emit_eval_fn=fake_eval, probe=_version_probe)
        assert fact["fs_launch_spec"]["sbatch"] is None
        assert fact["missing"] == ["lm_eval not importable"] and fact["executable"] is False
        assert fact["notes"] == [f"python defaulted to {sys.executable}", "local run"]


class TestSubmitTrial:
    def test_submit_happy_path_sbatches_through_launch(self, tmp_path):
        spec = _fs_launch_spec(tmp_path)
        runner = _Recorder(stdout="Submitted batch job 4242\n")
        result = submit_trial(
            spec,
            expected_token="tok",
            supplied_token="tok",
            confirm=plan_hash(spec),
            fabric=_fabric("ready", "fabric_ready"),
            launch_fn=launch,
            runner=runner,
        )
        assert result["job_id"] == "4242"
        assert runner.calls[0][:2] == ["bash", "-lc"]  # sbatch on this estate needs a login shell
        assert (tmp_path / "run" / "launch-fs-trial.sbatch").is_file()

    def test_submit_delegates_to_injected_launch_fn_with_confirm(self, tmp_path):
        spec = _fs_launch_spec(tmp_path)
        runner = _Recorder()
        seen: dict = {}

        def fake_launch(fs_launch_spec, *, confirm, runner):
            seen["spec"], seen["confirm"], seen["runner"] = fs_launch_spec, confirm, runner
            return {"job_id": "777", "command": "sbatch fake", "returncode": 0}

        result = submit_trial(
            spec,
            expected_token="tok",
            supplied_token="tok",
            confirm=plan_hash(spec),
            fabric=_fabric(),
            launch_fn=fake_launch,
            runner=runner,
        )
        assert result["job_id"] == "777"
        assert seen["spec"] is spec and seen["runner"] is runner
        assert seen["confirm"] == plan_hash(spec)  # forwarded untouched: launch re-checks it
        assert runner.calls == []

    def test_submit_refuses_on_token_mismatch_and_touches_nothing(self, tmp_path):
        spec = _fs_launch_spec(tmp_path)
        runner = _Recorder(stdout="Submitted batch job 1\n")
        with pytest.raises(LaunchRefused) as exc:
            submit_trial(
                spec,
                expected_token="derived-token",
                supplied_token="other",
                confirm=plan_hash(spec),
                fabric=_fabric("ready", "fabric_ready"),
                launch_fn=launch,
                runner=runner,
            )
        assert str(exc.value) == "AR-LN-006 token_mismatch"
        assert runner.calls == []  # no submit call may happen on a refusal
        assert not (tmp_path / "run" / "launch-fs-trial.sbatch").exists()

    @pytest.mark.parametrize(
        "state,reason", [("refused", "fabric_refused"), ("unmeasured", "fabric_unmeasured:timeout")]
    )
    def test_submit_refuses_when_the_fabric_is_unusable(self, tmp_path, state, reason):
        spec = _fs_launch_spec(tmp_path)
        runner = _Recorder(stdout="Submitted batch job 1\n")
        with pytest.raises(LaunchRefused) as exc:
            submit_trial(
                spec,
                expected_token="tok",
                supplied_token="tok",
                confirm=plan_hash(spec),
                fabric=_fabric(state, reason),
                launch_fn=launch,
                runner=runner,
            )
        assert str(exc.value) == f"AR-LN-003 {reason}"
        assert runner.calls == []  # refused and unmeasured alike: never launch on an unknown fabric
        assert not (tmp_path / "run" / "launch-fs-trial.sbatch").exists()

    def test_submit_checks_the_token_before_the_fabric(self, tmp_path):
        """Both gates would fire: AR-LN-006 first is the deliberate order in submit_trial."""
        runner = _Recorder()
        with pytest.raises(LaunchRefused) as exc:
            submit_trial(
                _fs_launch_spec(tmp_path),
                expected_token="a",
                supplied_token="b",
                confirm="c",
                fabric=_fabric("refused", "fabric_refused"),
                launch_fn=launch,
                runner=runner,
            )
        assert str(exc.value) == "AR-LN-006 token_mismatch"
        assert runner.calls == []


def test_submit_trial_refuses_empty_expected_token():
    calls = []
    with pytest.raises(LaunchRefused, match="AR-LN-006 token_mismatch"):
        submit_trial({}, expected_token="", supplied_token="", confirm="x",
                     fabric=_fabric(), launch_fn=lambda *a, **k: calls.append(a) or {})
    assert calls == []
