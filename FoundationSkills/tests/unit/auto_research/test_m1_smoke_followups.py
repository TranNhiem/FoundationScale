"""Follow-ups from the M1 live smoke (2026-10-05): node memory, partition, sbatch-less trials, repeated close."""
from __future__ import annotations

import pytest

from foundationskills.core.status import Status
from foundationskills.interfaces.fs.emit_trial import emit_trial
from foundationskills.interfaces.fs.sbatch import render_sbatch
from foundationskills.skills.auto_research.ledger import Ledger
from foundationskills.skills.auto_research.skill import _campaign, _spec

from .test_skill_m1 import _Fakes, _approved, _ctx, _request, _stage

ARGV = ["/env/bin/python", "-m", "foundationskills.cli", "eval", "run"]
SBATCH = "#!/bin/bash\n#SBATCH --time=10-00:00:00\n"


def _render(gpus, hardware_id="gb200", **kw):
    return render_sbatch(ARGV, env={}, hardware_id=hardware_id, nodes=1, gpus_per_node=gpus,
                         job_name="j", log_dir="logs", launcher="python", **kw).splitlines()


class TestNodeMemoryAndPartition:
    def test_partial_gb200_node_requests_memory_per_gpu(self):
        assert "#SBATCH --mem-per-gpu=200G" in _render(1)
        assert "#SBATCH --mem-per-gpu=200G" in _render(3)

    def test_full_gb200_node_and_other_hardware_keep_the_old_render(self):
        assert not any("--mem" in line for line in _render(4))
        assert not any("--mem" in line for line in _render(1, hardware_id="h100"))

    def test_explicit_memory_and_partition_are_rendered(self):
        lines = _render(2, mem_per_gpu="100G", partition="hhri-ai")
        assert "#SBATCH --mem-per-gpu=100G" in lines and "#SBATCH --partition=hhri-ai" in lines
        assert not any(line.startswith("#SBATCH --partition") for line in _render(2))

    @pytest.mark.parametrize("bad", [{"partition": "a;b"}, {"partition": "x y"}, {"mem_per_gpu": "lots"}])
    def test_unsafe_values_raise(self, bad):
        with pytest.raises(ValueError):
            _render(1, **bad)


def _trial(kind="eval_only", partition="hhri-ai"):
    trial = {"trial": "t", "role": "baseline", "kind": kind, "nodes": 1, "gpus_per_node": 1,
             "eval_request": {"benchmarks": ["arc_easy"], "python": "/env/bin/python"},
             "train_request": {"run_name": "r"}}
    if partition:
        trial["partition"] = partition
    return trial


class TestTrialRenderGates:
    def test_trial_without_sbatch_or_argv_is_refused_not_run_on_this_host(self):
        fact = emit_trial({"trial_spec": _trial(kind="train")},
                          emit_train_fn=lambda **kw: {"executable": True, "sbatch": None})
        assert fact["executable"] is False and fact["missing"] == ["sbatch_not_rendered"]
        assert fact["fs_launch_spec"]["state"] == "REFUSED"
        fact = emit_trial({"trial_spec": _trial()}, emit_eval_fn=lambda req, **kw: {"executable": True, "sbatch": None})
        assert fact["missing"] == ["sbatch_not_rendered"]

    def test_train_sbatch_is_layered_like_the_training_skill(self):
        trial = {**_trial(kind="train"), "gpus_per_node": 4,
                 "train_request": {"run_name": "ar-t1", "output_dir": "/runs/ar-t1"}}
        argv = ["torchrun", "--nproc-per-node", "4", "-m", "foundationscale.train.cli"]
        fact = emit_trial({"trial_spec": trial},
                          emit_train_fn=lambda **kw: {"executable": True, "sbatch": None, "argv": argv, "env": {}})
        lines = fact["fs_launch_spec"]["sbatch"].splitlines()
        assert fact["executable"] is True
        assert "#SBATCH --partition=hhri-ai" in lines and "#SBATCH --gres=gpu:4" in lines
        assert "#SBATCH --time=10-00:00:00" in lines and not any("--mem" in line for line in lines)
        assert any(line.startswith("srun torchrun") for line in lines)

    def test_train_render_failure_is_a_named_refusal(self):
        fact = emit_trial({"trial_spec": _trial(kind="train")},  # 1 GPU renders python, argv says torchrun
                          emit_train_fn=lambda **kw: {"executable": True, "sbatch": None, "argv": ["torchrun"]})
        assert fact["executable"] is False and fact["missing"][0].startswith("sbatch_render_failed:")

    def test_local_hardware_may_have_no_sbatch(self):
        fact = emit_trial({"trial_spec": _trial(partition="")}, hardware_id="local",
                          emit_eval_fn=lambda req, **kw: {"executable": True, "sbatch": None})
        assert fact["executable"] is True

    def test_eval_trial_forwards_partition_to_the_emitter(self):
        seen = {}

        def fake(req, **kw):
            seen.update(kw)
            return {"executable": True, "sbatch": SBATCH + "#SBATCH --partition=hhri-ai\n"}

        fact = emit_trial({"trial_spec": _trial()}, emit_eval_fn=fake)
        assert seen["partition"] == "hhri-ai" and fact["executable"] is True

    def test_render_that_drops_the_partition_is_refused(self):
        fact = emit_trial({"trial_spec": _trial(kind="train")},
                          emit_train_fn=lambda **kw: {"executable": True, "sbatch": SBATCH})
        assert fact["missing"] == ["partition_not_rendered:hhri-ai"]


def test_repeated_close_seals_the_ledger_once(tmp_path):
    spec = _spec()
    _stage(tmp_path, _approved(spec))
    fakes = _Fakes()
    first = fakes.skill.execute(_request(tmp_path, action="close", stop_reason="done"), _ctx(tmp_path))
    second = fakes.skill.execute(_request(tmp_path, action="close", stop_reason="done"), _ctx(tmp_path))
    ops = [e["op"] for e in Ledger(tmp_path / "ledger").entries()]
    assert ops.count("campaign_closed") == 1
    assert first.status is second.status and first.status is not Status.PASS
    assert Ledger(tmp_path / "ledger").verify() == []
    assert _campaign(spec) == first.payload["campaign"]["id"]
