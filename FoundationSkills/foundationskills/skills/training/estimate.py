
"""Deterministic memory/time estimates for FoundationScale training runs.

All quantities are in GB = 1e9 bytes unless noted. Formulas (see also the
assumption lists attached to every estimate):

Parameter/state bytes (mixed precision AdamW):

- bf16 weights = 2 bytes/param; fp32 = 4.
- Full fine-tune with AdamW mixed precision = 16 bytes/param:
  2 (bf16 weights) + 2 (bf16 grads) + 12 (fp32 master + m + v).
- FSDP shards weights, grads AND optimizer over ``world`` (ZeRO-3-like);
  DDP replicates them. ``tp`` additionally divides weights/grads/optimizer.

LoRA (Hu et al. 2021):

- Frozen base bf16 weights = 2 bytes/param (sharded by FSDP like full FT,
  replicated under DDP); no base grads/optimizer.
- Trainable params = ``layers * 23 * rank * hidden``, i.e. per layer
  ``4 * rank * (hidden + hidden)`` for the 4 attention projections (d_in=d_out=hidden)
  plus ``3 * rank * (hidden + intermediate)`` for the MLP, with
  ``intermediate ~ 4 * hidden`` when unknown: 7 targeted linears per layer.
- Trainable states = 16 bytes/param (2 weights + 2 grads + 12 AdamW), sharded
  like full-FT states.

QLoRA: base weights = 0.5 bytes/param (4-bit) + the same LoRA terms as above.

Activations (Korthikanti et al. 2022, arXiv:2205.05198), per layer:

- without recompute: ``s * b * h * (34 + 5 * a * s / h)`` bytes for bf16,
  where s=seq_len, b=micro_batch (per GPU), h=hidden, a=attention heads.
- FS uses SDPA attention, which materialises no quadratic attention matrix,
  so the ``5 * a * s / h`` term is dropped (stated in assumptions); pass
  ``attn="naive"`` to keep it.
- with gradient checkpointing: ~``2 * s * b * h`` bytes per layer (layer inputs)
  plus one full layer of activations for the recomputed layer.
- MoE activations use hidden only (approximation).

Overhead: 1.5 GB CUDA context/cache + 10% fragmentation of the subtotal.

MoE: total params drive memory, active params drive FLOPs.

Time (full fine-tune): ``FLOPs = 6 * N_active * D`` (forward 2ND + backward 4ND).
LoRA skips the backward-to-weights pass, costing ~4 * N_active * D (~2/3 of
full; assumption stated). RL multiplies by ``rl_generation_factor`` (default 3.0)
to account for on-policy generation. Wall time =
``FLOPs / (gpus * bf16_peak * MFU)``; MFU and its provenance come from the
hardware card — estimates never invent an MFU.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any

from foundationskills.skills.training.knowledge import Hardware, KnowledgeError, Variant

__all__ = [
    "MemoryEstimate",
    "TimeEstimate",
    "estimate_memory",
    "estimate_time",
    "lora_trainable_params",
    "tokens_per_step",
]

GB = 1e9
_CUDA_CONTEXT_GB = 1.5
_FRAGMENTATION = 0.10


@dataclass(frozen=True)
class MemoryEstimate:
    weights_gb: float
    grads_gb: float
    optimizer_gb: float
    activations_gb: float
    overhead_gb: float
    total_per_gpu_gb: float
    assumptions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TimeEstimate:
    total_flops: float
    tokens_per_s_per_gpu: float
    hours: float
    gpu_hours: float
    mfu: float
    mfu_provenance: str
    assumptions: list[str] = field(default_factory=list)


def lora_trainable_params(variant: Variant, rank: int = 16) -> int:
    """Trainable LoRA parameter count: ``layers * 23 * rank * hidden``.

    7 targeted linears per layer (q/k/v/o with d_in=d_out=hidden; gate/up/down
    with intermediate ~ 4*hidden). Documented approximation.
    """
    hidden = variant.hidden
    intermediate = 4 * hidden
    per_layer = 4 * rank * (hidden + hidden) + 3 * rank * (hidden + intermediate)
    return variant.layers * per_layer


def tokens_per_step(micro_batch: int, seq_len: int, grad_accum: int, dp: int) -> int:
    """Tokens consumed per optimizer step across the data-parallel group."""
    return int(micro_batch) * int(seq_len) * int(grad_accum) * int(dp)


def _optimizer_bytes_per_param(optimizer: str, assumptions: list[str]) -> float:
    name = optimizer.lower()
    if "adam" in name:
        return 12.0  # fp32 master weights + m + v
    if name.startswith("sgd"):
        assumptions.append(f"optimizer {optimizer!r}: counted as fp32 momentum only (4 bytes/param)")
        return 4.0
    assumptions.append(f"optimizer {optimizer!r}: unknown; counted AdamW-like (12 bytes/param)")
    return 12.0


def estimate_memory(
    variant: Variant,
    *,
    method: str,
    seq_len: int,
    micro_batch: int,
    precision: str = "bf16",
    grad_ckpt: bool = True,
    sharding: str = "fsdp",
    world: int = 1,
    tp: int = 1,
    optimizer: str = "adamw",
    lora_rank: int = 16,
    attn: str = "sdpa",
) -> MemoryEstimate:
    """Per-GPU memory estimate for one training configuration."""
    if method not in {"full", "lora", "qlora"}:
        raise ValueError(f"unknown method {method!r}; expected full|lora|qlora")
    if sharding not in {"fsdp", "ddp"}:
        raise ValueError(f"unknown sharding {sharding!r}; expected fsdp|ddp (FS registers no other backend)")
    world = max(1, int(world))
    tp = max(1, int(tp))

    assumptions: list[str] = []
    n = variant.total_params
    if variant.arch == "moe":
        assumptions.append("MoE: total params drive memory (all experts materialise)")
    # FSDP shards over the whole world, TP groups included (dp*tp == world), so
    # the divisor is world -- not world*tp. DDP replicates; only TP splits.
    state_div = world if sharding == "fsdp" else tp
    assumptions.append(
        f"states divided by {state_div} ({sharding} over world={world}"
        f"{' = ZeRO-3-like sharding' if sharding == 'fsdp' else ' = replication, split by tp'}; tp={tp})"
    )

    wb = 2.0 if precision in {"bf16", "fp16"} else 4.0
    if precision not in {"bf16", "fp16", "fp32"}:
        assumptions.append(f"precision {precision!r} unknown; counted as fp32 (4 bytes/param)")
    opt_b = _optimizer_bytes_per_param(optimizer, assumptions)

    if method == "full":
        weights_b = wb * n
        grads_b = wb * n
        optim_b = opt_b * n
        assumptions.append(f"full FT {precision}: {wb} weights + {wb} grads + {opt_b} optimizer bytes/param")
    else:
        p = lora_trainable_params(variant, lora_rank)
        base_b = (0.5 if method == "qlora" else 2.0) * n
        weights_b = base_b + wb * p
        grads_b = wb * p
        optim_b = opt_b * p
        assumptions.append(
            f"{method}: frozen base {0.5 if method == 'qlora' else 2} bytes/param; "
            f"trainable {p:,} params (rank={lora_rank}, 7 targets/layer, intermediate≈4*hidden) at 16 bytes/param"
        )

    weights_gb = weights_b / state_div / GB
    grads_gb = grads_b / state_div / GB
    optimizer_gb = optim_b / state_div / GB

    s, b, h, a = int(seq_len), int(micro_batch), variant.hidden, variant.heads
    keep_quadratic = attn.lower() not in {"sdpa", "flash", "flash_attention", "flash_attention_2", "flash2"}
    quad = (5.0 * a * s / h) if keep_quadratic else 0.0
    full_layer_bytes = s * b * h * (34.0 + quad)
    if keep_quadratic:
        assumptions.append(f"attn={attn!r}: quadratic attention term 5*a*s/h retained")
    else:
        assumptions.append("attn=sdpa (as in FS): quadratic attention term 5*a*s/h dropped (approximation)")
    if grad_ckpt:
        act_bytes = variant.layers * 2.0 * s * b * h + full_layer_bytes
        assumptions.append("gradient checkpointing: 2*s*b*h per layer (layer inputs) + one full recomputed layer")
    else:
        act_bytes = variant.layers * full_layer_bytes
        assumptions.append("no gradient checkpointing: full per-layer activations")
    if variant.arch == "moe":
        assumptions.append("MoE activations use hidden only (approximation)")
    # Logits: s*b*V in bf16, the fp32 upcast for the loss, and its fp32 grad
    # (~10 bytes/element), unsharded. Measured on GB200 (Gemma-4 E4B, V=262k,
    # seq 4096, b1, FSDP x4): without this term the estimate was 38.1 GB against
    # 48.3 GB allocated -- the gap is exactly this ~10.7 GB.
    vocab = int(getattr(variant, "vocab", 0) or 0)
    if vocab:
        act_bytes += s * b * vocab * 10.0
        assumptions.append(f"logits: s*b*V*10 bytes (bf16 logits + fp32 upcast + fp32 grad), V={vocab:,}")
    else:
        assumptions.append("logits term omitted: variant vocab unknown (underestimates large-vocab models)")
    activations_gb = act_bytes / GB  # per-GPU micro-batch; not divided by world/tp (conservative)

    subtotal = weights_gb + grads_gb + optimizer_gb + activations_gb
    overhead_gb = _CUDA_CONTEXT_GB + _FRAGMENTATION * subtotal
    assumptions.append(f"overhead: {_CUDA_CONTEXT_GB} GB CUDA context + {_FRAGMENTATION:.0%} fragmentation")
    total = subtotal + overhead_gb
    return MemoryEstimate(
        weights_gb=weights_gb,
        grads_gb=grads_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb,
        overhead_gb=overhead_gb,
        total_per_gpu_gb=total,
        assumptions=assumptions,
    )


def estimate_time(
    variant: Variant,
    *,
    tokens: int,
    hardware: Hardware,
    gpus: int,
    method: str = "full",
    rl_generation_factor: float = 3.0,
    sharding: str | None = None,
    micro_batch: int | None = None,
    grad_ckpt: bool | None = None,
    stage: str | None = None,
) -> TimeEstimate:
    """Wall-clock estimate: ``FLOPs / (gpus * peak * MFU)``.

    FLOP factor per trained token is 6 * N_active for full FT, 4 * N_active
    (~2/3) for LoRA/QLoRA (backward-to-weights skipped; assumption stated), and
    6 * N_active * ``rl_generation_factor`` for RL (on-policy generation cost).
    """
    if method not in {"full", "lora", "qlora", "rl", "rl_lora"}:
        raise ValueError(f"unknown method {method!r}")
    if stage == "rl" and method in {"full", "lora", "qlora"}:
        method = "rl" if method == "full" else "rl_lora"  # generation cost applies to every RL stage
    gpus = max(1, int(gpus))
    tokens = int(tokens)
    assumptions: list[str] = []

    n_active = variant.active_params
    if method in {"lora", "qlora"}:
        flops_per_token = 4.0 * n_active
        assumptions.append("LoRA/QLoRA: backward-to-weights skipped, ~4*N*D (~2/3 of full 6*N*D)")
    elif method == "rl_lora":
        flops_per_token = 4.0 * n_active * rl_generation_factor
        assumptions.append(
            f"RL with LoRA: 4*N*D x generation factor {rl_generation_factor} (on-policy rollouts dominate)"
        )
    elif method == "rl":
        flops_per_token = 6.0 * n_active * rl_generation_factor
        assumptions.append(f"RL: 6*N*D x generation factor {rl_generation_factor} (on-policy rollouts dominate)")
    else:
        flops_per_token = 6.0 * n_active
        assumptions.append("full FT: 6*N_active*D (forward 2ND + backward 4ND)")
    if variant.arch == "moe":
        assumptions.append("MoE: FLOPs use active params, not total params")

    mfu_key = "moe" if variant.arch == "moe" else "dense"
    entry = hardware.mfu.get(mfu_key)
    if entry is None and mfu_key == "moe" and hardware.mfu.get("dense"):
        entry = hardware.mfu.get("dense")
        assumptions.append("no MoE MFU recorded: dense MFU used (MoE routing usually lowers it)")
        entry = {**entry, "provenance": "derived-from-dense"}
    if not isinstance(entry, dict) or "value" not in entry:
        raise KnowledgeError(f"hardware {hardware.id!r}: mfu.{mfu_key} missing; time is UNMEASURED without MFU")
    mfu = float(entry["value"])
    mfu_provenance = str(entry.get("provenance", "unknown"))
    evidence = entry.get("evidence", "no evidence recorded")
    # MFU depends on the configuration: measured on GB200, the same model ran at
    # 31.8% (DDP, b2, ga8) and 4.6% (FSDP, b1, grad ckpt). Use the nearest
    # measured point for this config when the profile records points.
    points = [pt for pt in (hardware.mfu.get("points") or []) if isinstance(pt, dict) and "value" in pt]
    if points and mfu_key == "dense" and any(v is not None for v in (sharding, micro_batch, grad_ckpt)):
        def distance(pt: dict) -> float:
            cfg = pt.get("config") or {}
            d = 0.0
            if sharding is not None and cfg.get("sharding") != sharding:
                d += 2.0
            if grad_ckpt is not None and bool(cfg.get("grad_ckpt")) != bool(grad_ckpt):
                d += 1.0
            point_method = cfg.get("method", "full")
            if (method in ("lora", "qlora", "rl_lora")) != (point_method in ("lora", "qlora")):
                d += 3.0  # LoRA vs full changes FLOPs per step: never borrow across
            if micro_batch is not None and cfg.get("micro_batch") is not None:
                d += abs(math.log2(max(1, int(micro_batch))) - math.log2(max(1, int(cfg["micro_batch"])))) * 0.5
            return d
        best = min(points, key=distance)
        mfu = float(best["value"])
        exact = distance(best) == 0.0
        mfu_provenance = str(best.get("provenance", "unknown")) if exact else "derived"
        evidence = best.get("evidence", evidence)
        if not exact:
            assumptions.append(f"mfu extrapolated from the nearest measured config point {best.get('config')}")
    assumptions.append(f"MFU {mfu} ({mfu_provenance}: {evidence})")

    total_flops = flops_per_token * tokens
    effective_flops_per_s = gpus * hardware.bf16_dense_tflops * 1e12 * mfu
    hours = total_flops / effective_flops_per_s / 3600.0
    tokens_per_s = tokens / (hours * 3600.0) / gpus if hours > 0 else 0.0
    return TimeEstimate(
        total_flops=total_flops,
        tokens_per_s_per_gpu=tokens_per_s,
        hours=hours,
        gpu_hours=hours * gpus,
        mfu=mfu,
        mfu_provenance=mfu_provenance,
        assumptions=assumptions,
    )
