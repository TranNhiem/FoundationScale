
from __future__ import annotations

import pytest

from foundationskills.skills.training.estimate import (
    estimate_memory,
    estimate_time,
    lora_trainable_params,
    tokens_per_step,
)
from foundationskills.skills.training.knowledge import Hardware, Variant


def dense7b() -> Variant:
    return Variant(
        id="dense7b", hf_id=None, local_path=None, size_b=7.0, active_b=None,
        arch="dense", hidden=4096, layers=32, heads=32, kv_heads=8, head_dim=128,
        vocab=32000, context_length=8192, num_experts=None, experts_per_token=None,
        tied_embeddings=False, instruct=False,
    )


def moe_variant() -> Variant:
    return Variant(
        id="moe", hf_id=None, local_path=None, size_b=60.0, active_b=3.0,
        arch="moe", hidden=2048, layers=24, heads=16, kv_heads=4, head_dim=128,
        vocab=32000, context_length=8192, num_experts=64, experts_per_token=8,
        tied_embeddings=False, instruct=False,
    )


def make_hw(mem: float = 80.0, tflops: float = 1000.0, mfu: float = 0.318, gpn: int = 8) -> Hardware:
    return Hardware(
        id="hw", gpu_name="HW", mem_gb=mem, gpus_per_node=gpn,
        bf16_dense_tflops=tflops, peak_provenance="datasheet", interconnect="nvlink",
        mfu={
            "dense": {"value": mfu, "provenance": "literature", "evidence": "test"},
            "moe": {"value": mfu, "provenance": "literature", "evidence": "test"},
        },
        scheduler="slurm", cluster_rules=(), notes=(),
    )


def test_full_ft_fsdp_states_7b_8gpus() -> None:
    est = estimate_memory(dense7b(), method="full", seq_len=2048, micro_batch=1, sharding="fsdp", world=8)
    # 7e9 params x 16 bytes = 112 GB of states, sharded over 8 -> 14 GB total states
    assert est.weights_gb == pytest.approx(1.75)   # 2 bytes/param / 8
    assert est.grads_gb == pytest.approx(1.75)     # 2 bytes/param / 8
    assert est.optimizer_gb == pytest.approx(10.5)  # 12 bytes/param / 8


def test_full_ft_ddp_replicates_states() -> None:
    est = estimate_memory(dense7b(), method="full", seq_len=2048, micro_batch=1, sharding="ddp", world=8)
    assert est.weights_gb == pytest.approx(14.0)
    assert est.grads_gb == pytest.approx(14.0)
    assert est.optimizer_gb == pytest.approx(84.0)  # 112 GB of replicated states total


def test_tp_divides_states() -> None:
    est = estimate_memory(dense7b(), method="full", seq_len=2048, micro_batch=1, sharding="ddp", world=1, tp=2)
    assert est.weights_gb == pytest.approx(7.0)
    assert est.optimizer_gb == pytest.approx(42.0)


def test_lora_trainable_params_formula() -> None:
    # layers * [4 * r * (h + h) + 3 * r * (h + 4h)] = layers * 23 * r * h
    expected = 32 * 23 * 16 * 4096
    assert expected == 48_234_496
    assert lora_trainable_params(dense7b(), rank=16) == expected


def test_lora_memory() -> None:
    est = estimate_memory(dense7b(), method="lora", seq_len=2048, micro_batch=1, sharding="fsdp", world=1, lora_rank=16)
    p = 48_234_496
    assert est.weights_gb == pytest.approx((2.0 * 7e9 + 2.0 * p) / 1e9)  # frozen base + lora weights
    assert est.grads_gb == pytest.approx(2.0 * p / 1e9)
    assert est.optimizer_gb == pytest.approx(12.0 * p / 1e9)


def test_qlora_base_is_half_byte_per_param() -> None:
    est = estimate_memory(dense7b(), method="qlora", seq_len=2048, micro_batch=1, sharding="fsdp", world=1)
    p = lora_trainable_params(dense7b(), 16)
    assert est.weights_gb == pytest.approx((0.5 * 7e9 + 2.0 * p) / 1e9)


def test_activations_no_ckpt_sdpa() -> None:
    est = estimate_memory(dense7b(), method="full", seq_len=4096, micro_batch=1, grad_ckpt=False, world=8)
    # per layer s*b*h*34 = 4096*4096*34 = 570,425,344 bytes; x32 layers
    expected_bytes = 32 * 4096 * 1 * 4096 * 34
    expected_bytes += 4096 * 1 * dense7b().vocab * 14  # F14 logits term, fitted to three GB200 measurements
    assert est.activations_gb == pytest.approx(expected_bytes / 1e9)


def test_activations_with_ckpt() -> None:
    est = estimate_memory(dense7b(), method="full", seq_len=4096, micro_batch=1, grad_ckpt=True, world=8)
    # 2*s*b*h per layer + one full layer: 32*2*4096*4096 + 4096*4096*34
    expected_bytes = 32 * 2 * 4096 * 4096 + 4096 * 4096 * 34
    expected_bytes += 4096 * 1 * dense7b().vocab * 14  # F14 logits term, fitted to three GB200 measurements
    assert est.activations_gb == pytest.approx(expected_bytes / 1e9)


def test_activations_without_sdpa_keep_quadratic_term() -> None:
    est = estimate_memory(
        dense7b(), method="full", seq_len=4096, micro_batch=1, grad_ckpt=False, world=8, attn="naive"
    )
    # factor 34 + 5*a*s/h = 34 + 5*32*4096/4096 = 194
    expected_bytes = 32 * 4096 * 4096 * 194
    expected_bytes += 4096 * 1 * dense7b().vocab * 14  # F14 logits term, fitted to three GB200 measurements
    assert est.activations_gb == pytest.approx(expected_bytes / 1e9)


def test_overhead_and_total() -> None:
    est = estimate_memory(dense7b(), method="full", seq_len=2048, micro_batch=1, sharding="fsdp", world=8)
    subtotal = est.weights_gb + est.grads_gb + est.optimizer_gb + est.activations_gb
    assert est.overhead_gb == pytest.approx(1.5 + 0.10 * subtotal)
    assert est.total_per_gpu_gb == pytest.approx(subtotal + est.overhead_gb)
    assert est.assumptions


def test_unknown_method_and_sharding_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_memory(dense7b(), method="megatron", seq_len=2048, micro_batch=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        estimate_memory(dense7b(), method="full", seq_len=2048, micro_batch=1, sharding="megatron")


def test_tokens_per_step() -> None:
    assert tokens_per_step(micro_batch=2, seq_len=4096, grad_accum=8, dp=4) == 262_144


def test_time_full_hand_case() -> None:
    hw = make_hw(tflops=1000.0, mfu=0.318)
    est = estimate_time(dense7b(), tokens=1_000_000_000, hardware=hw, gpus=4, method="full")
    assert est.total_flops == pytest.approx(6.0 * 7e9 * 1e9)
    assert est.hours == pytest.approx(4.2e19 / (4 * 1000e12 * 0.318 * 3600.0))
    assert est.gpu_hours == pytest.approx(est.hours * 4)
    assert est.tokens_per_s_per_gpu == pytest.approx(1e9 / (est.hours * 3600.0) / 4)
    assert est.mfu == pytest.approx(0.318)
    assert est.mfu_provenance == "literature"


def test_time_lora_is_two_thirds_of_full() -> None:
    hw = make_hw(tflops=1000.0, mfu=0.318)
    full = estimate_time(dense7b(), tokens=10**9, hardware=hw, gpus=4, method="full")
    lora = estimate_time(dense7b(), tokens=10**9, hardware=hw, gpus=4, method="lora")
    assert lora.total_flops == pytest.approx(full.total_flops * 2.0 / 3.0)
    assert lora.hours == pytest.approx(full.hours * 2.0 / 3.0)


def test_time_rl_generation_factor() -> None:
    hw = make_hw(tflops=1000.0, mfu=0.318)
    est = estimate_time(dense7b(), tokens=10**9, hardware=hw, gpus=4, method="rl")
    assert est.total_flops == pytest.approx(6.0 * 7e9 * 10**9 * 3.0)


def test_time_moe_uses_active_params() -> None:
    hw = make_hw(tflops=1000.0, mfu=0.3)
    est = estimate_time(moe_variant(), tokens=10**9, hardware=hw, gpus=4, method="full")
    assert est.total_flops == pytest.approx(6.0 * 3e9 * 10**9)  # active 3B, not total 60B
    assert any("active" in a for a in est.assumptions)
