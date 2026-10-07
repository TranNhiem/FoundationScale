"""Regressions from the 2026-10-07 real-workload run (E4B SFT on 4 GB200s, RL on ARC).

CPU-only: real knowledge pack, fake FS capabilities, no GPU, no network.
"""
from __future__ import annotations

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
