"""CPU-only rungs 0-1 tests for the Megatron RL lane; no skips allowed."""

from __future__ import annotations

import importlib
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import foundationscale.rl.megatron as megatron_pkg
from foundationscale.rl.megatron.lane_config import MegatronLaneConfig
from foundationscale.rl.megatron.logprobs import (
    check_vocab_shard,
    softcap,
    vocab_parallel_token_logprobs,
)
from foundationscale.rl.megatron.normalization import (
    compute_denominators,
    normalized_loss,
)


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def test_softcap_identity_and_value() -> None:
    x = torch.tensor([-2.0, 0.0, 3.0])
    assert torch.equal(softcap(x, None), x)
    assert torch.equal(softcap(x, 0.0), x)
    capped = softcap(x, 30.0)
    assert torch.allclose(capped, 30.0 * torch.tanh(x / 30.0), atol=1e-7)
    with pytest.raises(ValueError, match="softcap_negative_cap"):
        softcap(x, -1.0)


def test_vocab_parallel_logprobs_tp1_matches_log_softmax() -> None:
    torch.manual_seed(7)
    logits = torch.randn(3, 5, 17, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 17, (3, 5), dtype=torch.int64)
    got = vocab_parallel_token_logprobs(logits, targets, None, 0, 1)
    ref = (
        torch.log_softmax(logits.detach().float(), dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    assert got.dtype == torch.float32
    assert torch.allclose(got.detach(), ref, atol=1e-6)
    got.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def _tp2_worker(rank: int, port: int) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=2,
        rank=rank,
    )
    try:
        torch.manual_seed(244)
        batch, seq_len, padded_vocab, tp_size = 2, 6, 10, 2
        shard_rows = padded_vocab // tp_size
        full = torch.randn(batch, seq_len, padded_vocab, dtype=torch.float32)
        targets = torch.randint(0, padded_vocab, (batch, seq_len), dtype=torch.int64)
        weights = torch.randn(batch, seq_len, dtype=torch.float32)

        shard = full[:, :, rank * shard_rows : (rank + 1) * shard_rows].clone().requires_grad_(True)
        got = vocab_parallel_token_logprobs(shard, targets, dist.group.WORLD, rank, tp_size)

        ref_logits = full.clone().requires_grad_(True)
        ref = (
            torch.log_softmax(ref_logits.float(), dim=-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
        )
        assert torch.allclose(got.detach(), ref.detach(), atol=1e-5)

        (got * weights).sum().backward()
        (ref * weights).sum().backward()
        ref_shard_grad = ref_logits.grad[:, :, rank * shard_rows : (rank + 1) * shard_rows]
        assert shard.grad is not None
        assert torch.allclose(shard.grad, ref_shard_grad, atol=1e-5)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_vocab_parallel_logprobs_tp2_forward_backward_gloo() -> None:
    mp.spawn(_tp2_worker, args=(_free_port(),), nprocs=2, join=True)


def test_chunk_size_path_equals_unchunked() -> None:
    torch.manual_seed(19)
    logits = torch.randn(2, 9, 23, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 23, (2, 9), dtype=torch.int64)
    ref = (
        torch.log_softmax(logits.detach().float(), dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    for chunk in (1, 3, 9, 32):
        got = vocab_parallel_token_logprobs(logits, targets, None, 0, 1, chunk)
        assert torch.allclose(got, ref, atol=1e-6)


def test_check_vocab_shard_refuses_v_over_tp_bug() -> None:
    check_vocab_shard(131072, 2, 262144, 262143)
    with pytest.raises(ValueError, match="vocab_shard_size_mismatch"):
        check_vocab_shard(131072, 1, 262144, 0)
    with pytest.raises(ValueError, match="vocab_shard_target_out_of_range"):
        check_vocab_shard(131072, 2, 262144, 262144)


@pytest.mark.parametrize("family", ("token", "sequence", "dr_grpo", "preference"))
def test_normalized_loss_sum_over_shards_equals_exact_mean(family: str) -> None:
    torch.manual_seed(41)
    batch, tokens = 6, 5
    loss_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    sample_mask = torch.ones(batch, dtype=torch.float32)
    per_token = torch.randn(batch, tokens, dtype=torch.float32).abs() + 0.1
    declared = (8.0, float(tokens)) if family == "dr_grpo" else None
    den = compute_denominators(family, loss_mask, sample_mask, None, declared=declared)

    chunks = [slice(0, 2), slice(2, 4), slice(4, 5), slice(5, 6)]
    shard_sum = sum(normalized_loss(family, per_token[s], loss_mask[s], den) for s in chunks)
    if family in ("token", "dr_grpo"):
        exact = (per_token * loss_mask).sum() / den.value
    elif family == "sequence":
        exact = ((per_token * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)).sum() / den.value
    else:
        exact = (per_token * loss_mask).sum(dim=1).sum() / den.value
    assert torch.allclose(shard_sum, exact, atol=1e-6)


def test_dr_grpo_declared_refusal() -> None:
    loss_mask = torch.ones(4, 3, dtype=torch.float32)
    sample_mask = torch.ones(4, dtype=torch.float32)
    with pytest.raises(ValueError, match="dr_grpo_declared_batch_too_small"):
        compute_denominators("dr_grpo", loss_mask, sample_mask, None, declared=(3.0, 3.0))


def test_lane_config_validate_refusal_matrix() -> None:
    valid = MegatronLaneConfig(
        tp=2,
        pp=2,
        cp=1,
        ep=1,
        etp=2,
        sp=True,
        mbs=1,
        gbs=1,
        seq_len=8192,
        softcap=30.0,
        head_dim=512,
        num_experts=8,
    )
    assert valid.validate(4) == 1 and valid.dp == 1

    cases: list[tuple[MegatronLaneConfig, int, str]] = [
        (MegatronLaneConfig(tp=3, gbs=1, mbs=1), 4, "lane_world_not_divisible"),
        (MegatronLaneConfig(tp=2, gbs=3, mbs=2), 4, "lane_gbs_not_divisible"),
        (MegatronLaneConfig(cp=2, head_dim=512, gbs=1, mbs=1), 2, "lane_cp_head_dim_unsupported"),
        (MegatronLaneConfig(ep=2, etp=0, num_experts=8, gbs=4, mbs=1), 4, "lane_ep_requires_etp"),
        (
            MegatronLaneConfig(ep=3, etp=1, num_experts=8, gbs=3, mbs=1),
            3,
            "lane_ep_experts_not_divisible",
        ),
        (MegatronLaneConfig(tp=1, sp=True, gbs=1, mbs=1), 1, "lane_sp_without_tp"),
        (MegatronLaneConfig(vpp=2, gbs=1, mbs=1), 1, "lane_vpp_not_supported"),
        (
            MegatronLaneConfig(refit_every=10, save_interval=11, gbs=1, mbs=1),
            1,
            "lane_refit_save_staleness",
        ),
    ]
    for cfg, world, reason in cases:
        with pytest.raises(ValueError, match=reason):
            cfg.validate(world)

    cp_ok = MegatronLaneConfig(tp=2, cp=2, head_dim=128, sp=True, gbs=4, mbs=1)
    assert cp_ok.validate(4) == 1
    cp_bad_world = MegatronLaneConfig(tp=2, cp=2, head_dim=128, sp=True, gbs=4, mbs=1)
    with pytest.raises(ValueError, match="lane_world_not_divisible"):
        cp_bad_world.validate(6)


def test_package_imports_without_megatron() -> None:
    assert megatron_pkg.softcap is softcap
    assert megatron_pkg.MegatronLaneConfig is MegatronLaneConfig
    # driver and pp_step keep every megatron import inside functions, so the
    # submodules themselves import on a host with no megatron installed.
    for name in ("driver", "pp_step"):
        importlib.import_module(f"foundationscale.rl.megatron.{name}")
