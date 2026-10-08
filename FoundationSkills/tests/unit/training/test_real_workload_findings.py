"""Regressions from the 2026-10-07 real-workload run (E4B SFT on 4 GB200s, RL on ARC).

CPU-only: real knowledge pack, fake FS capabilities, no GPU, no network.
"""
from __future__ import annotations

from pathlib import Path

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.interfaces.fs.emit_rl import emit_rl
from foundationskills.skills.training.feasibility import check_feasibility
from foundationskills.skills.training.knowledge import find_variant, load_hardware
from foundationskills.skills.training.planner import _resolve_hardware, plan

E4B = "/home/hhri-ai/hh28144/pretraining_weights/Vision-Language-Models/Google/Gemma4/gemma-4-E4B-it"


def _caps() -> FSCapabilities:
    return FSCapabilities(
        available=True, fs_version="0.0.0+test", train_flags=frozenset({"--model", "--dataset", "--output-dir"}),
        train_objectives=("sft",), sharding_strategies=("ddp", "fsdp"), executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"), axes_measured=True, rl_algorithms=("dr_grpo", "gspo", "dapo", "grpo"),
        rl_runnable={"dr_grpo": None, "gspo": None, "dapo": None, "grpo": None}, rl_reward_kinds=("mcq_letter",),
        families={"gemma4": ("gemma4",)}, backends=("ddp", "fsdp"),
    )


def _goal(**over) -> dict:
    goal = {
        "objective": "Taiwan general chat", "target_capabilities": ["general_chat"], "domain": "taiwan_general",
        "preserve_general": True,
        "base_model": {"name_or_path": E4B, "family": "gemma4", "model_type": "vlm"},
        "data": {"sources": [{"uri": "/d/shards", "kind": "jsonl", "approx_tokens": 9079889,
                              "approx_examples": 19699, "use_for": ["sft"]}]},
        "hardware": {"gpu": "GB200", "gpus_per_node": 4, "nodes": 1},
    }
    goal.update(over)
    return goal


def _plan(goal: dict) -> dict:
    result = plan(goal, readiness=None, hardware_id="gb200", caps=_caps())
    return result if isinstance(result, dict) else result.payload


def test_e4b_plan_turns_gradient_checkpointing_off_and_prices_it():
    out = _plan(_goal())
    sft = next(s for s in out["stages"] if s["stage"] == "sft")
    assert sft["hparams"]["gradient_checkpointing"] is False
    assert any("gradient_checkpointing off" in d.get("choice", "") for d in out["decisions"])
    # feasibility prices the emitted configuration (no checkpointing), not the 77.7 GB default
    detail = out["feasibility"]["findings"][0]["detail"]
    assert detail.startswith(f"memory {sft['estimate']['memory_gb_per_gpu']:.1f} GB/GPU")


def test_feasibility_prices_the_planned_optimizer():
    hw = load_hardware()["gb200-189gb"]
    _family, variant = find_variant(E4B)
    kw = dict(gpus=1, method="full", seq_len=4096, micro_batch=2, tokens=1_000_000, sharding="ddp",
              grad_ckpt=False, stage="rl")
    on_gpu = check_feasibility(variant, hw, **kw)
    on_host = check_feasibility(variant, hw, **kw, mem_extra={"optimizer": "host_adamw"})
    assert on_gpu.verdict == "infeasible" and on_host.verdict != "infeasible"


def test_gb200_alias_resolves_to_the_knowledge_entry_with_its_nccl_pins():
    hw, hw_id = _resolve_hardware(load_hardware(), "gb200", None)
    assert hw_id == "gb200-189gb"
    assert hw.env["NCCL_SOCKET_IFNAME"] == "bond0" and hw.env["GLOO_SOCKET_IFNAME"] == "bond0"


def test_rl_max_steps_derived_from_the_dataset(tmp_path):
    stage = {"name": "rl", "stage": "rl", "algorithm": "dr_grpo", "method": "full", "hparams": {"prompts_per_step": 4}}
    dataset = {"format": "rl", "fs_columns": {"gold_key": "answer"},
               "shards": [{"path": f"{tmp_path}/s/a.jsonl", "records": 1000}, {"path": f"{tmp_path}/s/b.jsonl", "records": 111}]}
    spec = emit_rl(stage, dataset=dataset, model="m", output_dir=str(tmp_path / "o"), caps=_caps(), run_name="r")
    assert spec["rl_config"]["max_steps"] == 278  # ceil(1111 / 4): one pass, as the planner priced it
    assert any(n.startswith("max_steps derived") for n in spec["notes"])
    stage["hparams"]["max_steps"] = 7
    spec = emit_rl(stage, dataset=dataset, model="m", output_dir=str(tmp_path / "o"), caps=_caps(), run_name="r")
    assert spec["rl_config"]["max_steps"] == 7  # an explicit value always wins


def test_sft_estimate_prices_epochs_and_padding():
    from foundationskills.skills.training.planner import _sft_processed_tokens

    stats = {"num_records": 19699, "num_tokens": 9079889,
             "length_hist": {"128": 72, "256": 5219, "512": 7997, "1024": 5251, "2048": 1156, "4096": 4}}
    notes: list[str] = []
    processed = _sft_processed_tokens(9079889, {"epochs": 2}, stats, 4, notes)
    # measured on that run: 30.6M processed; the estimate must land near it, never at the 9.08M one-pass figure
    assert 30_000_000 < processed < 36_000_000
    assert any("2 epoch" in n for n in notes) and any("padding" in n for n in notes)
    bare: list[str] = []
    assert _sft_processed_tokens(1000, {"max_steps": 10, "epochs": 2}, {}, 4, bare) == 1000
    assert any("NOT priced" in n for n in bare)


def test_rl_spec_on_a_cluster_gets_an_sbatch_and_sbatchless_slurm_submit_refuses(tmp_path):
    import pytest

    from foundationskills.core.orchestrator import plan_hash
    from foundationskills.interfaces.fs.launch import LaunchRefused, launch

    spec = {"executable": True, "argv": ["python", "-c", "pass"], "sbatch": None, "launch_target": "slurm",
            "output_dir": str(tmp_path)}
    ran: list = []
    with pytest.raises(LaunchRefused, match="no sbatch"):
        launch(spec, confirm=plan_hash(spec), submit=True, runner=lambda *a, **k: ran.append(a))
    assert ran == []  # neither the argv nor sbatch ran


def test_sbatch_log_dir_is_absolute():
    """A relative --output resolved against wherever sbatch was invoked (found on the RL re-emit).

    Source-level guard: both render_sbatch calls in training.emit pass the resolved launch dir."""
    from foundationskills.skills.training import emit_skill

    src = Path(emit_skill.__file__).read_text(encoding="utf-8")
    assert src.count("log_dir=str(Path(launch_dir).resolve())") == 2 and "log_dir=str(launch_dir)" not in src


def test_emitted_env_carries_the_source_tree_when_not_installed(monkeypatch):
    """The rendered RL sbatch died with "No module named 'foundationskills'" outside the submitting shell."""
    import importlib.util
    import types as _types

    from foundationskills.interfaces.fs.emit_train import _code_path_env

    fake = {"foundationskills": "/src/tree/FoundationSkills/foundationskills/__init__.py",
            "foundationscale": "/venv/lib/python3.11/site-packages/foundationscale/__init__.py"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: _types.SimpleNamespace(origin=fake.get(n)))
    notes: list[str] = []
    env = _code_path_env(notes)
    assert env == {"PYTHONPATH": str(Path("/src/tree/FoundationSkills").resolve())}  # installed package adds nothing
    assert notes and "PYTHONPATH" in notes[0]
