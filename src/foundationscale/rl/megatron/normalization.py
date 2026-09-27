"""Global pre-collapsed denominators for exact DP x CP x microbatch loss sums.

Design G section 3 rule: denominators are global scalars computed before the
pipeline loop; numerators are additive sums. ``normalized_loss`` therefore
returns a microbatch-local term whose SUM over every microbatch and rank is
the batch-exact mean, rather than an averaged average.

Supported families are the design's four named axes: ``token`` (dapo/bnpo),
``sequence`` (grpo/gspo/rloo; each supervised sequence contributes with its
own ``1/T_i``), ``dr_grpo`` (declared constant ``B_g * L_cap`` with the
``S_g <= B_g`` refusal), and ``preference`` ( ``1/P_g`` over co-resident
pairs ). A ``None`` group means no collective; a non-None group requires an
initialized torch.distributed world and is reduced with SUM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

__all__ = (
    "GlobalDenominators",
    "compute_denominators",
    "normalized_loss",
)

_FAMILIES = frozenset(("token", "sequence", "dr_grpo", "preference"))


@dataclass(frozen=True, slots=True)
class GlobalDenominators:
    """Pre-collapsed global scalars; numerators elsewhere stay additive."""

    family: str
    tokens: float = 0.0
    sequences: float = 0.0
    pairs: float = 0.0
    value: float = 0.0
    details: dict[str, float] = field(default_factory=dict)


def _require_masks(loss_mask: torch.Tensor, sample_mask: torch.Tensor) -> None:
    if loss_mask.dim() != 2:
        raise ValueError(
            f"denominator_loss_mask_shape: loss_mask has {tuple(loss_mask.shape)}; "
            f"a [B, T] matrix is required"
        )
    if sample_mask.dim() != 1 or sample_mask.shape[0] != loss_mask.shape[0]:
        raise ValueError(
            f"denominator_sample_mask_shape: sample_mask has {tuple(sample_mask.shape)}; "
            f"a [B] vector matching loss_mask rows {loss_mask.shape[0]} is required"
        )
    for name, mask in (("loss_mask", loss_mask), ("sample_mask", sample_mask)):
        bad = int(((mask != 0) & (mask != 1)).sum())
        if bad:
            raise ValueError(
                f"denominator_mask_not_binary: {bad} of {mask.numel()} entries in "
                f"{name} are neither 0 nor 1; fractional supervision cannot be "
                f"counted exactly once across ranks"
            )


def _global_sum(local_value: float, group: Any | None) -> float:
    if group is None:
        return float(local_value)
    if not (dist.is_available() and dist.is_initialized()):
        raise ValueError(
            "denominator_group_uninitialized: a process group was supplied but "
            "torch.distributed is not initialized; refusing to guess whether "
            "this rank sees the whole batch"
        )
    total = torch.tensor(float(local_value), dtype=torch.float64)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    return float(total.item())


def compute_denominators(
    family: str,
    loss_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    group: Any | None,
    declared: tuple[float, float] | float | None = None,
) -> GlobalDenominators:
    """Compute global denominators for one rollout batch before the loop."""
    if family not in _FAMILIES:
        raise ValueError(
            f"denominator_family_unknown: family={family!r}; known families are "
            f"{sorted(_FAMILIES)} and an undeclared normalization is not free"
        )
    _require_masks(loss_mask, sample_mask)
    mask = loss_mask.to(dtype=torch.float64)
    samples = sample_mask.to(dtype=torch.float64)
    t_local = float(mask.sum())
    s_local = float(samples.sum())
    t_g = _global_sum(t_local, group)
    s_g = _global_sum(s_local, group)
    p_g = s_g

    if samples.sum() > 0:
        row_counts = mask.sum(dim=1)
        empty_included = int(((samples.view(-1) > 0) & (row_counts == 0)).sum())
        if family == "sequence" and empty_included:
            raise ValueError(
                "sequence_empty_supervised_row: "
                f"{empty_included} sampled rows carry 0 supervised tokens; "
                f"sequence normalization needs 1/T_i for every counted row"
            )

    if family == "token":
        if t_g <= 0.0:
            raise ValueError(
                "token_denominator_empty: 0 supervised tokens globally; a token-"
                "level mean over an empty batch is unmeasured, never 0.0"
            )
        return GlobalDenominators(family=family, tokens=t_g, value=t_g)
    if family == "sequence":
        if s_g <= 0.0:
            raise ValueError(
                "sequence_denominator_empty: 0 sampled sequences globally; a "
                "sequence-level mean over an empty batch is unmeasured"
            )
        return GlobalDenominators(family=family, tokens=t_g, sequences=s_g, value=s_g)
    if family == "preference":
        if p_g <= 0.0:
            raise ValueError(
                "preference_denominator_empty: 0 preference pairs globally; "
                "pairs are co-resident and cannot be reconstructed from shards"
            )
        return GlobalDenominators(family=family, pairs=p_g, value=p_g)

    if not isinstance(declared, tuple) or len(declared) != 2:
        raise ValueError(
            "dr_grpo_declared_missing: declared must be (B_g, L_cap); dr_grpo "
            "uses declared constants rather than a measured allreduce denominator"
        )
    declared_sequences, length_cap = float(declared[0]), float(declared[1])
    if declared_sequences <= 0.0 or length_cap <= 0.0:
        raise ValueError(
            "dr_grpo_declared_nonpositive: "
            f"B_g={declared_sequences}, L_cap={length_cap}; both declared "
            f"constants must be positive"
        )
    if s_g > declared_sequences:
        raise ValueError(
            "dr_grpo_declared_batch_too_small: "
            f"measured S_g={s_g} exceeds declared B_g={declared_sequences}; "
            f"the constant denominator would under-count sequences"
        )
    constant = declared_sequences * length_cap
    return GlobalDenominators(
        family=family,
        tokens=t_g,
        sequences=s_g,
        value=constant,
        details={"declared_sequences": declared_sequences, "length_cap": length_cap},
    )


def normalized_loss(
    family: str,
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    den: GlobalDenominators,
) -> torch.Tensor:
    """Return the additive microbatch term whose global sum is the exact mean."""
    if family != den.family:
        raise ValueError(
            f"normalized_loss_family_mismatch: caller passed family={family!r} "
            f"but denominators were computed for {den.family!r}; mixing "
            f"normalizations makes summed microbatches a different objective"
        )
    if per_token_loss.shape != loss_mask.shape:
        raise ValueError(
            "normalized_loss_shape_mismatch: "
            f"per_token_loss {tuple(per_token_loss.shape)} vs loss_mask "
            f"{tuple(loss_mask.shape)}; each loss entry must be masked once"
        )
    mask = loss_mask.to(dtype=per_token_loss.dtype)
    bad = int(((mask != 0) & (mask != 1)).sum())
    if bad:
        raise ValueError(
            f"normalized_loss_mask_not_binary: {bad} entries are neither 0 nor "
            f"1; fractional masks break the additive global-denominator rule"
        )
    denom = float(den.value)
    if denom <= 0.0:
        raise ValueError(
            f"normalized_loss_denominator_nonpositive: denominator={denom}; "
            f"global denominators must be positive pre-collapsed scalars"
        )
    if family in ("token", "dr_grpo"):
        return (per_token_loss * mask).sum() / denom
    if family == "sequence":
        row_sum = (per_token_loss * mask).sum(dim=1)
        row_count = mask.sum(dim=1)
        nonempty = row_count > 0
        if not bool(nonempty.any()):
            return per_token_loss.sum() * 0.0
        row_mean = row_sum[nonempty] / row_count[nonempty].clamp(min=1.0)
        return row_mean.sum() / denom
    pair_sum = (per_token_loss * mask).sum(dim=1).sum()
    return pair_sum / denom
