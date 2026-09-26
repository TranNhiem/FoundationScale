
from __future__ import annotations

import pytest

from foundationskills.skills.training.knowledge import Hardware, Variant
from foundationskills.skills.training.method import select_method


def variant(size_b: float, hidden: int = 4096, layers: int = 32) -> Variant:
    return Variant(
        id=f"v{size_b}", hf_id=None, local_path=None, size_b=size_b, active_b=None,
        arch="dense", hidden=hidden, layers=layers, heads=32, kv_heads=8, head_dim=128,
        vocab=32000, context_length=8192, num_experts=None, experts_per_token=None,
        tied_embeddings=False, instruct=False,
    )


def hw80() -> Hardware:
    return Hardware(
        id="hw80", gpu_name="HW", mem_gb=80.0, gpus_per_node=8,
        bf16_dense_tflops=1000.0, peak_provenance="datasheet", interconnect="nvlink",
        mfu={
            "dense": {"value": 0.318, "provenance": "literature", "evidence": "test"},
            "moe": {"value": 0.3, "provenance": "literature", "evidence": "test"},
        },
        scheduler="slurm", cluster_rules=(), notes=(),
    )


def test_cpt_is_always_full() -> None:
    choice = select_method(goal="domain_expert", stage="cpt", data_tokens=10**6, variant=variant(0.5), hardware=hw80(), gpus=1)
    assert choice.method == "full"
    assert any("LoRA" in b or "low-rank" in b for b in choice.because)
    assert choice.alternatives  # records why lora was rejected


def test_pretrain_is_always_full() -> None:
    choice = select_method(goal=None, stage="pretrain", data_tokens=None, variant=variant(0.5), hardware=hw80(), gpus=1)
    assert choice.method == "full"


def test_sft_small_data_picks_lora() -> None:
    choice = select_method(goal="general_chat", stage="sft", data_tokens=10**7, variant=variant(7.0), hardware=hw80(), gpus=8)
    assert choice.method == "lora"
    assert any("50,000,000" in b or "<" in b for b in choice.because)


def test_sft_large_data_that_fits_picks_full() -> None:
    choice = select_method(goal="general_chat", stage="sft", data_tokens=10**8, variant=variant(0.5), hardware=hw80(), gpus=1)
    assert choice.method == "full"
    assert any("fits" in b for b in choice.because)


def test_sft_memory_bound_picks_lora() -> None:
    # 7B full FT needs ~114 GB states on 1 GPU with FSDP world=1 -> memory-bound
    choice = select_method(goal="general_chat", stage="sft", data_tokens=10**8, variant=variant(7.0), hardware=hw80(), gpus=1)
    assert choice.method == "lora"
    assert any("memory-bound" in b for b in choice.because)


def test_rl_picks_full_only_when_it_fits_one_gpu() -> None:
    small = select_method(goal="reasoning", stage="rl", data_tokens=None, variant=variant(0.5), hardware=hw80(), gpus=4)
    assert small.method == "full"
    assert any("single-device" in b for b in small.because)


def test_rl_too_big_picks_lora_with_finding() -> None:
    choice = select_method(goal="reasoning", stage="rl", data_tokens=None, variant=variant(7.0), hardware=hw80(), gpus=4)
    assert choice.method == "lora"
    assert any("FINDING" in b and "single-device" in b for b in choice.because)


def test_prefer_is_honoured() -> None:
    choice = select_method(goal=None, stage="sft", data_tokens=10**7, variant=variant(7.0), hardware=hw80(), gpus=1, prefer="full")
    assert choice.method == "full"
    assert any("prefer" in b for b in choice.because)


def test_prefer_qlora_downgrades_to_lora() -> None:
    choice = select_method(goal=None, stage="sft", data_tokens=10**7, variant=variant(7.0), hardware=hw80(), gpus=1, prefer="qlora")
    assert choice.method == "lora"
    assert any("QLoRA" in b or "qlora" in b for b in choice.because)
    assert choice.alternatives and choice.alternatives[0]["executable"] is False


def test_unknown_stage_and_prefer_raise() -> None:
    with pytest.raises(ValueError):
        select_method(goal=None, stage="megatron", data_tokens=None, variant=variant(1.0), hardware=hw80(), gpus=1)
    with pytest.raises(ValueError):
        select_method(goal=None, stage="sft", data_tokens=None, variant=variant(1.0), hardware=hw80(), gpus=1, prefer="zz")
