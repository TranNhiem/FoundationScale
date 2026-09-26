
"""Method (full / LoRA) selection rules.

Rules (measured FS constraints in mind):

- CPT / pretrain -> ``full``: low-rank adapters cannot absorb large new corpora.
- SFT -> ``lora`` when the corpus is small (< 50M tokens — adapters suffice and
  forget less) or when full FT is memory-bound on the target hardware
  (estimated at > 0.80 x GPU memory). Otherwise ``full``.
- RL on FS -> the FS RLTrainer is single-device, so ``full`` only when the full
  model fits on ONE GPU; otherwise ``lora`` and the finding is recorded in
  ``because`` (an LoRA RL run on FS needs an adapter-capable RLTrainer config).
- ``prefer="qlora"`` is downgraded to ``lora``: QLoRA is not an FS feature today
  (``--adapter`` supports lora only). Other explicit ``prefer`` values are
  honoured and labelled as such.

Estimates use seq_len = min(context_length, 4096) and micro_batch 1 with
gradient checkpointing, the cheapest reasonable default for a method decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from foundationskills.skills.training.estimate import estimate_memory
from foundationskills.skills.training.knowledge import Hardware, Variant

__all__ = ["MethodChoice", "select_method"]

SFT_LORA_TOKEN_THRESHOLD = 50_000_000
_MEM_FIT_FRACTION = 0.80


@dataclass(frozen=True)
class MethodChoice:
    method: str  # "full" | "lora"
    because: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)


def _fits(variant: Variant, hardware: Hardware, *, world: int, method: str = "full") -> tuple[bool, float, float]:
    est = estimate_memory(
        variant,
        method=method,
        seq_len=min(variant.context_length, 4096),
        micro_batch=1,
        grad_ckpt=True,
        sharding="fsdp",
        world=max(1, world),
    )
    budget = _MEM_FIT_FRACTION * hardware.mem_gb
    return est.total_per_gpu_gb <= budget, est.total_per_gpu_gb, budget


def select_method(
    *,
    goal: str | None,
    stage: str,
    data_tokens: int | None,
    variant: Variant,
    hardware: Hardware,
    gpus: int,
    prefer: str | None = None,
) -> MethodChoice:
    """Pick a training method; every choice lists its reasons and trade-offs."""
    if prefer is not None:
        if prefer == "qlora":
            return MethodChoice(
                method="lora",
                because=[
                    "prefer=qlora requested; QLoRA is not supported by installed FS "
                    "(--adapter supports lora only) — downgraded to lora"
                ],
                alternatives=[
                    {
                        "method": "qlora",
                        "executable": False,
                        "missing": "missing: qlora (FS --adapter supports lora only; verify)",
                    }
                ],
            )
        if prefer in {"full", "lora"}:
            return MethodChoice(method=prefer, because=[f"method forced by prefer={prefer!r} (no rule applied)"], alternatives=[])
        raise ValueError(f"unknown method preference {prefer!r}")

    stage_norm = stage.lower()
    goal_note = f" for goal {goal}" if goal else ""

    if stage_norm in {"pretrain", "cpt"}:
        return MethodChoice(
            method="full",
            because=[
                f"{stage_norm}{goal_note} trains on large corpora; full fine-tuning is required — "
                "low-rank LoRA adapters lack the capacity to absorb a corpus-scale distribution shift",
            ],
            alternatives=[
                {
                    "method": "lora",
                    "why_not": "LoRA under-trains on corpus-scale CPT data (low-rank capacity bound)",
                }
            ],
        )

    if stage_norm == "sft":
        if data_tokens is not None and data_tokens < SFT_LORA_TOKEN_THRESHOLD:
            return MethodChoice(
                method="lora",
                because=[
                    f"SFT corpus is small ({data_tokens:,} tokens < {SFT_LORA_TOKEN_THRESHOLD:,}); "
                    "LoRA is cheaper and forgets less of the base model",
                ],
                alternatives=[{"method": "full", "tradeoff": "higher capacity, higher memory, more forgetting risk"}],
            )
        fits, total_gb, budget_gb = _fits(variant, hardware, world=gpus)
        if not fits:
            return MethodChoice(
                method="lora",
                because=[
                    f"full FT is memory-bound: ~{total_gb:.0f} GB/GPU needed vs {budget_gb:.0f} GB "
                    f"(0.80 x {hardware.mem_gb:.0f} GB) available on {gpus}x {hardware.gpu_name}; LoRA fits",
                ],
                alternatives=[{"method": "full", "tradeoff": "needs more GPUs or model parallelism to fit"}],
            )
        tokens_note = "token count unmeasured" if data_tokens is None else f"{data_tokens:,} tokens >= {SFT_LORA_TOKEN_THRESHOLD:,}"
        return MethodChoice(
            method="full",
            because=[
                f"SFT{goal_note}: {tokens_note} and full FT fits ({total_gb:.0f} GB/GPU <= {budget_gb:.0f} GB budget)",
            ],
            alternatives=[{"method": "lora", "tradeoff": "cheaper, but lower ceiling on a large SFT corpus"}],
        )

    if stage_norm in {"rl", "preference"}:
        fits, total_gb, budget_gb = _fits(variant, hardware, world=1)
        if fits:
            return MethodChoice(
                method="full",
                because=[
                    f"FS RLTrainer is single-device; full model fits on ONE {hardware.gpu_name} "
                    f"({total_gb:.0f} GB <= {budget_gb:.0f} GB budget)",
                ],
                alternatives=[{"method": "lora", "tradeoff": "lower memory but adapter bias in policy updates"}],
            )
        return MethodChoice(
            method="lora",
            because=[
                f"FINDING: FS RLTrainer is single-device and full RL needs ~{total_gb:.0f} GB "
                f"> {budget_gb:.0f} GB (0.80 x {hardware.mem_gb:.0f} GB) on one GPU; "
                "LoRA selected to fit single-device RL",
            ],
            alternatives=[
                {
                    "method": "full",
                    "why_not": "does not fit one GPU and FS RLTrainer has no multi-GPU support (measured)",
                }
            ],
        )

    raise ValueError(f"unknown stage {stage!r}; expected pretrain|cpt|sft|preference|rl")
