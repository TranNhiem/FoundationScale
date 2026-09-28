"""Pipeline forward_step factories binding the FS objective into Megatron-Core.

This module is the seam between mcore's ``forward_backward_func`` schedule and
the FS objective plane. ``make_forward_step`` returns the
``forward_step(data_iterator, model) -> (output_tensor, loss_func)`` pair
mcore expects; ``loss_func`` runs on the LAST pipeline stage only, computes
vocab-parallel token logprobs (softcap first, then the
``vocab_parallel_token_logprobs`` reduction -- the sequence plane is gathered
at the LM head when SP is on, so the helper sees ``[B, S, V/TP]``), feeds them
to the FS ``TensorPolicyLoss`` kernel, rescales the scalar by the batch-exact
global denominators (design section 3), and returns
``(loss, metrics_dict)``.

With ``do_not_average_loss=True`` mcore SUMS the per-microbatch losses, so
each microbatch must contribute its additive share of the batch mean. The
kernel performs its own internal reduction (token_mean / sequence_mean /
constant); :func:`rescale_to_global_denominator` multiplies the scalar by
``local_denominator / global_denominator`` for the SAME unit the family
prices, which is exactly the rule 'denominators are pre-collapsed global
scalars; numerators are additive sums'. Summing the rescaled scalars over
microbatches and DP/CP ranks therefore reproduces the single-batch mean.

WHAT IS CLAIMED: the metric contract matches mcore 0.15 convention -- the
returned metrics dict values are 0-dim CUDA-detached tensors, the loss is
differentiable, and the keys (``loss``, ``ratio_mean``, ``clip_fraction``,
``tokens``) say what they measure. WHAT IS NOT CLAIMED: anything about CP
packing/zigzag (rung 1+ plumbing lives in the driver's collate path); VPP
(refused by ``MegatronLaneConfig.validate``).

All megatron imports are lazy (inside functions) so this module imports on a
plain CPU box.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch

from foundationscale.rl.megatron.lane_config import MegatronLaneConfig
from foundationscale.rl.megatron.logprobs import (
    check_vocab_shard,
    softcap,
    vocab_parallel_token_logprobs,
)
from foundationscale.rl.megatron.normalization import GlobalDenominators

__all__ = (
    "ForwardStepFn",
    "family_of",
    "local_denominator",
    "make_forward_step",
    "make_logprob_forward_step",
    "ratio_and_clip_metrics",
    "rescale_to_global_denominator",
)

ForwardStepFn = Callable[[Iterator[dict[str, Any]], Any], tuple[Any, Callable[..., Any]]]

_TOKEN_FAMILIES = frozenset({"token", "token_level", "dapo", "bnpo"})
_SEQUENCE_FAMILIES = frozenset({"sequence", "sequence_level", "grpo", "gspo", "rloo"})
_DR_GRPO_FAMILIES = frozenset({"dr_grpo"})
_PREFERENCE_FAMILIES = frozenset({"preference", "dpo", "cpo", "ipo", "kto", "orpo", "simpo"})


def family_of(family: str) -> str:
    """Collapse an algorithm/family name to one of the four denominator units.

    Returns one of ``"token"``, ``"sequence"``, ``"dr_grpo"``,
    ``"preference"``. Refuses with the offending name: an undeclared family
    has no declared denominator, and guessing one would silently price the
    batch on the wrong unit (design section 3).
    """
    name = family.strip().lower()
    if name in _TOKEN_FAMILIES:
        return "token"
    if name in _SEQUENCE_FAMILIES:
        return "sequence"
    if name in _DR_GRPO_FAMILIES:
        return "dr_grpo"
    if name in _PREFERENCE_FAMILIES:
        return "preference"
    raise ValueError(
        f"normalization family {family!r} is not one of the 4 declared units "
        f"(token, sequence, dr_grpo, preference); an undeclared family has no "
        f"denominator and cannot be priced"
    )


def local_denominator(
    family: str, loss_mask: torch.Tensor, sample_mask: torch.Tensor | None = None
) -> float:
    """The LOCAL additive unit count for one microbatch, matching the kernel's
    internal reduction for the same family:

    * token     -> supervised token count ``T_mb = loss_mask.sum()``
    * sequence  -> valid sequence count ``S_mb`` (rows with >=1 supervised
      token, or ``sample_mask.sum()`` when supplied)
    * dr_grpo   -> supervised token count (constant-reduction numerator unit;
      the kernel divided by ``rows * constant_length``, so the local scale
      uses the same numerator unit against the declared ``B_g``)
    * preference-> pair count = valid rows (pairs are co-resident per
      microbatch by the section 3 contract)

    Refuses a fully-masked microbatch only for the family whose unit is
    tokens; sequence/preference microbatches with zero valid rows contribute
    a zero scale, which is additive and correct.
    """
    unit = family_of(family)
    mask_f = loss_mask.detach().to(dtype=torch.float64)
    tokens = float(mask_f.sum())
    if sample_mask is not None:
        sequences = float(sample_mask.detach().to(dtype=torch.float64).sum())
    else:
        sequences = float((mask_f.sum(dim=-1) > 0).sum())
    if unit == "token":
        return tokens
    if unit == "sequence":
        return sequences
    if unit == "dr_grpo":
        return tokens
    return sequences


def _read_denominator(den: GlobalDenominators, unit: str) -> float:
    """Pull the global scalar for ``unit`` off a GlobalDenominators without
    presuming field names beyond the section 3 terms; refuses by naming every
    attribute it looked for when none is present."""
    candidates: dict[str, tuple[str, ...]] = {
        "token": ("tokens", "T_g", "global_tokens"),
        "sequence": ("sequences", "S_g", "global_sequences"),
        "dr_grpo": ("sequences", "S_g", "B_g", "global_sequences"),
        "preference": ("pairs", "P_g", "sequences", "global_pairs"),
    }
    names = candidates[unit]
    for name in names:
        value = getattr(den, name, None)
        if value is not None:
            scalar = float(value)
            if scalar <= 0.0:
                raise ValueError(
                    f"GlobalDenominators.{name} == {scalar}; the global {unit} "
                    f"denominator must be positive -- a zero denominator means "
                    f"the whole batch carries no supervised unit and dividing "
                    f"by it would report nan instead of 'nothing to train on'"
                )
            return scalar
    raise ValueError(
        f"GlobalDenominators carries none of {names} required for the "
        f"{unit!r} family; the denominators must be computed before the "
        f"pipeline loop via compute_denominators() (design section 3)"
    )


def rescale_to_global_denominator(
    loss: torch.Tensor,
    family: str,
    loss_mask: torch.Tensor,
    den: GlobalDenominators,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rescale the kernel's internally-reduced scalar so that SUMMING the
    returned values over microbatches and ranks (mcore ``do_not_average_loss``)
    yields the batch-exact mean.

    The kernel already divided by the LOCAL unit count; multiplying by
    ``local/global`` re-denominates onto the pre-collapsed global scalar. The
    local count uses :func:`local_denominator`, which is exactly the unit the
    kernel's reduction used, so the identity is algebraic, not approximate.
    """
    unit = family_of(family)
    local = local_denominator(unit, loss_mask, sample_mask)
    if local == 0.0:
        # An all-masked microbatch carries no supervised units; its additive
        # contribution is genuinely zero. Returning loss*0 keeps the graph
        # connected (mcore requires a differentiable scalar per microbatch).
        return loss * 0.0
    global_value = _read_denominator(den, unit)
    return loss * (local / global_value)


def ratio_and_clip_metrics(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_bounds: tuple[float, float],
) -> dict[str, torch.Tensor]:
    """Detached per-microbatch ratio diagnostics (token-scope readings).

    ``ratio_mean`` is the mask-weighted mean of ``exp(cur - old)``;
    ``clip_fraction`` is the fraction of supervised tokens whose ratio falls
    outside ``clip_bounds``; ``tokens`` is the local supervised count (the
    additive weight a reducer needs to pool microbatches correctly). All
    values are 0-dim detached fp32 tensors -- the metrics channel must never
    carry autograd state across the PP boundary.
    """
    with torch.no_grad():
        mask_f = loss_mask.detach().to(dtype=torch.float32)
        total = mask_f.sum()
        if not bool(total > 0):
            zero = torch.zeros((), dtype=torch.float32, device=current_logprobs.device)
            return {"ratio_mean": zero, "clip_fraction": zero, "tokens": zero}
        log_ratio = (current_logprobs.detach() - old_logprobs.detach()).to(torch.float32) * mask_f
        ratio = torch.exp(log_ratio)
        low, high = clip_bounds
        clipped = ((ratio < low) | (ratio > high)).to(torch.float32) * mask_f
        return {
            "ratio_mean": ((ratio * mask_f).sum() / total).detach(),
            "clip_fraction": (clipped.sum() / total).detach(),
            "tokens": total.to(torch.float32).detach(),
        }


def _require_batch_columns(batch: dict[str, Any], required: tuple[str, ...], owner: str) -> None:
    missing = tuple(name for name in required if name not in batch)
    if missing:
        raise ValueError(
            f"{owner}: microbatch is missing {len(missing)} of {len(required)} "
            f"required column(s): {missing}; present: {tuple(batch)}. Every "
            f"rank's data_iterator yields the same dict, so a missing column "
            f"here means the driver's collate dropped it"
        )


def _position_ids(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Explicit positions; mcore gets ``attention_mask=None`` (causal).

    Rows are RIGHT-padded, so the causal mask alone is exact for every real
    token: a pad position only ever attends backwards and is masked out of
    the loss. The batch's HF-style ``[B, S]`` float mask is NOT an mcore
    attention mask (mcore wants a boolean ``[B, 1, S, S]`` with True =
    masked), so it is never passed to the model.
    """
    input_ids = batch["input_ids"]
    position_ids = batch.get("position_ids")
    if position_ids is not None:
        return position_ids
    return (
        torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )


def _targets_from_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token targets: prediction at position t scores token t+1."""
    return input_ids[:, 1:].contiguous()


def make_forward_step(
    objective_loss_fn: Any,
    den: GlobalDenominators,
    cfg: MegatronLaneConfig,
) -> ForwardStepFn:
    """Build the training forward_step for mcore's forward_backward_func.

    ``objective_loss_fn`` is a ``foundationscale.rl.torch_backend.TensorPolicyLoss``
    (or any callable with that exact keyword signature); it is invoked on the
    last PP stage with ``current_logprobs`` (grad-carrying), ``old_logprobs``,
    ``advantages``, ``mask``, and optionally ``reference_logprobs`` from the
    microbatch dict. The returned loss_func returns
    ``(rescaled_loss, {"loss", "ratio_mean", "clip_fraction", "tokens"})``
    with detached 0-dim metric tensors, matching the
    ``do_not_average_loss=True`` convention.
    """
    family = (
        getattr(objective_loss_fn, "family", None)
        or getattr(getattr(objective_loss_fn, "objective", None), "normalization_family", None)
        or getattr(getattr(objective_loss_fn, "objective", None), "loss_family", None)
    )
    if family is None:
        raise ValueError(
            "make_forward_step could not read a normalization family off "
            f"{type(objective_loss_fn).__name__} (looked for .family and "
            f".objective.normalization_family / .loss_family); design section "
            f"3 requires the family to pick the global denominator unit"
        )
    unit = family_of(str(family))
    clip_bounds: tuple[float, float] = tuple(
        getattr(getattr(objective_loss_fn, "objective", None), "clip_bounds", (0.8, 1.2))
    )

    def forward_step(
        data_iterator: Iterator[dict[str, Any]], model: Any
    ) -> tuple[Any, Callable[..., Any]]:
        from megatron.core.parallel_state import (  # lazy: no megatron on CPU
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        batch = next(data_iterator)
        _require_batch_columns(
            batch,
            ("input_ids", "old_logprobs", "advantages", "loss_mask"),
            "forward_step(train)",
        )
        input_ids = batch["input_ids"]
        output_tensor = model(
            input_ids=input_ids,
            position_ids=_position_ids(batch),
            attention_mask=None,
            packed_seq_params=batch.get("packed_seq_params"),
        )

        def loss_func(out: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            # Logits arrive [b, s, V/TP] with parallel_output=True; the FS
            # helper takes [B, S, Vs], so transpose to batch-first here.
            logits = out.transpose(0, 1).contiguous() if out.shape[0] != input_ids.shape[0] else out
            logits = logits[:, :-1, :].to(torch.float32)
            logits = softcap(logits, cfg.softcap)
            targets = _targets_from_input_ids(input_ids)
            tp_size = int(get_tensor_model_parallel_world_size())
            check_vocab_shard(
                shard_rows=logits.shape[-1],
                tp_size=tp_size,
                padded_vocab=logits.shape[-1] * tp_size,
                max_target_id=int(targets.max()) if targets.numel() else 0,
            )
            current_logprobs = vocab_parallel_token_logprobs(
                logits,
                targets,
                tp_group=get_tensor_model_parallel_group(),
                tp_rank=int(get_tensor_model_parallel_rank()),
                tp_size=tp_size,
                chunk_size=None,
            )
            loss_mask = (
                batch["loss_mask"][:, 1:]
                if batch["loss_mask"].shape[1] == input_ids.shape[1]
                else batch["loss_mask"]
            )
            raw_loss = objective_loss_fn(
                current_logprobs=current_logprobs,
                old_logprobs=batch["old_logprobs"],
                advantages=batch["advantages"],
                mask=loss_mask,
                reference_logprobs=batch.get("reference_logprobs"),
            )
            scaled = rescale_to_global_denominator(
                raw_loss, unit, loss_mask, den, sample_mask=batch.get("sample_mask")
            )
            metrics = ratio_and_clip_metrics(
                current_logprobs, batch["old_logprobs"], loss_mask, clip_bounds
            )
            metrics = dict(metrics)
            metrics["loss"] = scaled.detach().to(torch.float32)
            return scaled, metrics

        return output_tensor, loss_func

    return forward_step


def make_logprob_forward_step(cfg: MegatronLaneConfig) -> ForwardStepFn:
    """Build the forward_only step that captures old/reference logprobs.

    Mirrors ``MegatronPolicyWorker.get_logprobs``: the collection function
    computes vocab-parallel logprobs under no-grad semantics imposed by the
    caller (``forward_only=True``), prepends a zero column so positions align
    with input tokens, and returns ``(zero_scalar, {"logprobs": ...})`` so the
    mcore schedule can collect per-microbatch outputs on the last stage.
    """

    def forward_step(
        data_iterator: Iterator[dict[str, Any]], model: Any
    ) -> tuple[Any, Callable[..., Any]]:
        from megatron.core.parallel_state import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        batch = next(data_iterator)
        _require_batch_columns(batch, ("input_ids",), "forward_step(logprobs)")
        input_ids = batch["input_ids"]
        output_tensor = model(
            input_ids=input_ids,
            position_ids=_position_ids(batch),
            attention_mask=None,
            packed_seq_params=batch.get("packed_seq_params"),
        )

        def collect_fn(out: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            logits = out.transpose(0, 1).contiguous() if out.shape[0] != input_ids.shape[0] else out
            logits = logits[:, :-1, :].to(torch.float32)
            logits = softcap(logits, cfg.softcap)
            targets = _targets_from_input_ids(input_ids)
            logprobs = vocab_parallel_token_logprobs(
                logits,
                targets,
                tp_group=get_tensor_model_parallel_group(),
                tp_rank=int(get_tensor_model_parallel_rank()),
                tp_size=int(get_tensor_model_parallel_world_size()),
                chunk_size=None,
            )
            logprobs = torch.cat([torch.zeros_like(logprobs[:, :1]), logprobs], dim=1).detach()
            zero = torch.zeros((), dtype=torch.float32, device=logprobs.device)
            return zero, {"logprobs": logprobs}

        return output_tensor, collect_fn

    return forward_step
