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

Each microbatch must contribute its additive share of the batch mean, but
mcore's schedule multiplies a ``(loss, metrics)`` return by the CP size and
divides it by ``num_microbatches`` (the ``do_not_average_loss`` opt-out does
not exist in every mcore release), and DDP then AVERAGES gradients over the
DP x CP group, so :func:`undo_mcore_loss_average` pre-applies the inverse of
both and the effective reduction is a SUM. The
kernel performs its own internal reduction (token_mean / sequence_mean /
constant); :func:`rescale_to_global_denominator` multiplies the scalar by
``local_denominator / global_denominator`` for the SAME unit the family
prices, which is exactly the rule 'denominators are pre-collapsed global
scalars; numerators are additive sums'. Summing the rescaled scalars over
microbatches and DP/CP ranks therefore reproduces the single-batch mean.

WHAT IS CLAIMED: the metric contract matches mcore 0.15 convention -- the
returned metrics dict values are 0-dim CUDA-detached tensors, the loss is
differentiable, and the keys (``loss``, ``ratio_mean``, ``clip_fraction``,
``tokens``) say what they measure. Under CP each rank feeds the model its
load-balanced pair of sequence chunks (:func:`cp_shard_index`) and the token
logprobs are all-gathered back to the full sequence before the objective, so
sequence-level objectives (GSPO ratio, per-sequence means) see every token.
WHAT IS NOT CLAIMED: packed sequences under CP; VPP (refused by
``MegatronLaneConfig.validate``).

All megatron imports are lazy (inside functions) so this module imports on a
plain CPU box.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from foundationscale.rl.megatron.lane_config import MegatronLaneConfig
from foundationscale.rl.megatron.logprobs import (
    check_vocab_shard,
    softcap,
    vocab_parallel_token_logprobs,
)
from foundationscale.rl.megatron.normalization import GlobalDenominators

if TYPE_CHECKING:
    import torch


__all__ = (
    "ForwardStepFn",
    "cp_shard_index",
    "family_of",
    "loss_unit",
    "local_denominator",
    "make_forward_step",
    "make_logprob_forward_step",
    "undo_mcore_loss_average",
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
    * dr_grpo   -> microbatch row count ``rows`` (the kernel's ``constant``
      reduction divides by ``rows * constant_length``, so the local scale is
      rows against the declared ``B_g``; scaling by supervised TOKENS inflated
      the loss by tokens/row -- measured: grad_norm tracked response length)
    * preference-> pair count = valid rows (pairs are co-resident per
      microbatch by the section 3 contract)

    Refuses a fully-masked microbatch only for the family whose unit is
    tokens; sequence/preference microbatches with zero valid rows contribute
    a zero scale, which is additive and correct.
    """
    import torch

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
        return float(loss_mask.shape[0])
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
    if unit == "dr_grpo" and den.details.get("declared_sequences"):
        # dr_grpo is denominated by the DECLARED batch B_g, not the measured S_g.
        return float(den.details["declared_sequences"])
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
    returned values over microbatches and ranks (see :func:`undo_mcore_loss_average`)
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


def objective_loss_or_zero(
    objective_loss_fn: Any,
    *,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    reference_logprobs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the objective kernel, or contribute an exact graph-connected zero.

    Under DP sharding one rank's microbatch can hold no supervised token while
    the global batch does (measured: a qwen3moe EP=2 rung-2 run died at step 14
    on rank 1 with "0 of 1023 mask entries are supervised"). The kernel rightly
    refuses to MEASURE a fully masked batch, but this microbatch's additive share
    of the globally denominated loss is exactly zero, which
    :func:`rescale_to_global_denominator` already returns for ``local == 0``. A
    batch with no supervised token anywhere is still refused, by the global
    denominator.
    """
    if not bool(mask.detach().sum() > 0):
        return current_logprobs.sum() * 0.0
    return objective_loss_fn(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        mask=mask,
        reference_logprobs=reference_logprobs,
    )


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
    import torch

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
    import torch

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


_REDUCTION_UNITS = {"token_mean": "token", "sequence_mean": "sequence", "constant": "dr_grpo"}


def loss_unit(objective_loss_fn: Any) -> str:
    """The global-denominator unit a loss callable declares (design section 3).

    An explicit ``.family`` (or ``.objective.normalization_family`` /
    ``.loss_family``) wins; otherwise the unit is read off
    ``.objective.reduction``, the same declaration ``TensorPolicyLoss`` uses
    to pick its own denominator, so the two cannot price one batch on
    different units. Refuses when neither is declared.
    """
    objective = getattr(objective_loss_fn, "objective", None)
    family = (
        getattr(objective_loss_fn, "family", None)
        or getattr(objective, "normalization_family", None)
        or getattr(objective, "loss_family", None)
    )
    if family is not None:
        return family_of(str(family))
    reduction = getattr(objective, "reduction", None)
    if reduction in _REDUCTION_UNITS:
        return _REDUCTION_UNITS[reduction]
    raise ValueError(
        "make_forward_step could not read a normalization unit off "
        f"{type(objective_loss_fn).__name__} (looked for .family, "
        ".objective.normalization_family / .loss_family, and "
        f".objective.reduction in {sorted(_REDUCTION_UNITS)}; got {reduction!r}); "
        "design section 3 requires the unit to pick the global denominator"
    )


def undo_mcore_loss_average(
    loss: torch.Tensor, num_microbatches: int, cp_size: int, dp_cp_size: int = 1
) -> torch.Tensor:
    """Pre-invert mcore's loss scaling (``* cp / num_microbatches``) and DDP's
    gradient average (``/ dp_cp_size``).

    ``forward_step_calc_loss`` multiplies a ``(loss, metrics)`` return by the
    CP group size and divides it by ``num_microbatches``; DDP then divides the
    summed gradient by the DP x CP group size. The FS loss is already an
    additive share of the batch-exact global mean, so both are inverted here
    and the net reduction over microbatches and ranks is a plain sum. Measured:
    without the ``dp_cp_size`` factor a DP=2 step-0 grad_norm read exactly half
    the 1-GPU value on identical rows.
    """
    if num_microbatches < 1 or cp_size < 1 or dp_cp_size < 1:
        raise ValueError(
            f"undo_mcore_loss_average needs num_microbatches, cp_size and "
            f"dp_cp_size >= 1, got {num_microbatches}, {cp_size}, {dp_cp_size}"
        )
    return loss * (num_microbatches * dp_cp_size / cp_size)


def cp_shard_index(seq_len: int, cp_size: int, cp_rank: int) -> list[int]:
    """Positions CP rank ``cp_rank`` holds under mcore's load-balanced split.

    ``seq_len`` (already padded to a multiple of ``2 * cp_size``) is cut into
    ``2 * cp_size`` equal chunks and rank ``r`` keeps chunks ``r`` and
    ``2 * cp_size - 1 - r``, the layout TE's causal CP attention expects.
    """
    if cp_size < 1 or not 0 <= cp_rank < cp_size or seq_len % (2 * cp_size):
        raise ValueError(
            f"cp_shard_index: seq_len={seq_len} must be a multiple of 2*cp_size "
            f"and 0 <= cp_rank={cp_rank} < cp_size={cp_size}"
        )
    chunk = seq_len // (2 * cp_size)
    first = range(cp_rank * chunk, (cp_rank + 1) * chunk)
    mirror = 2 * cp_size - 1 - cp_rank
    return [*first, *range(mirror * chunk, (mirror + 1) * chunk)]


@functools.cache
def _cp_gather_fn() -> Any:
    """Autograd all-gather of CP-local token logprobs back to full length.

    Every CP rank computes the same full-sequence loss, so backward keeps only
    this rank's slice of the gradient (no sum); DDP's CP average and the
    ``cp / num_microbatches`` inversion then yield the exact gradient.
    """
    import torch
    import torch.distributed as dist

    class _GatherCP(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, local: torch.Tensor, padded_len: int, group: Any) -> torch.Tensor:
            cp_size, cp_rank = dist.get_world_size(group), dist.get_rank(group)
            parts = [torch.empty_like(local) for _ in range(cp_size)]
            dist.all_gather(parts, local.contiguous(), group=group)
            full = local.new_empty((local.shape[0], padded_len))
            for rank, part in enumerate(parts):
                full[:, cp_shard_index(padded_len, cp_size, rank)] = part
            ctx.own = cp_shard_index(padded_len, cp_size, cp_rank)
            return full

        @staticmethod
        def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, None, None]:
            return grad[:, ctx.own].contiguous(), None, None

    return _GatherCP


def _cp_layout() -> tuple[int, int, Any]:
    from megatron.core.parallel_state import (
        get_context_parallel_group,
        get_context_parallel_rank,
        get_context_parallel_world_size,
    )

    cp_size = int(get_context_parallel_world_size())
    if cp_size == 1:
        return 1, 0, None
    return cp_size, int(get_context_parallel_rank()), get_context_parallel_group()


def sp_tp(cfg: MegatronLaneConfig) -> int:
    return cfg.tp if cfg.sp and cfg.tp > 1 else 1


def padded_seq_len(seq_len: int, cp_size: int, sp_tp: int) -> int:
    """Length every row is padded to: CP load balancing needs ``2 * cp`` chunks
    and SP reduce-scatters each chunk over TP. PP p2p buffers use this too."""
    multiple = (2 * cp_size if cp_size > 1 else 1) * sp_tp
    return -(-seq_len // multiple) * multiple


def _model_inputs(
    batch: dict[str, Any], cp_size: int, cp_rank: int, sp_tp: int = 1
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(input_ids, position_ids, padded_len)`` this CP rank feeds the model.

    Rows are padded to a multiple of ``2 * cp_size`` under CP times ``sp_tp``
    (the TP size when sequence parallelism is on: the embedding
    reduce-scatters dim 0, so each CP chunk must split evenly over TP). Pad
    positions only attend backwards and carry no loss. Under CP the rows are
    then sliced to this rank's load-balanced chunks.
    """
    import torch
    import torch.nn.functional as F

    input_ids = batch["input_ids"]
    position_ids = _position_ids(batch)
    seq_len = input_ids.shape[1]
    padded_len = padded_seq_len(seq_len, cp_size, sp_tp)
    extra = padded_len - seq_len
    if extra:
        input_ids = F.pad(input_ids, (0, extra))
        tail = position_ids[:, -1:] + torch.arange(1, extra + 1, device=position_ids.device)
        position_ids = torch.cat((position_ids, tail), dim=1)
    if cp_size == 1:
        return input_ids, position_ids, padded_len
    own = input_ids.new_tensor(cp_shard_index(padded_len, cp_size, cp_rank))
    return input_ids[:, own], position_ids[:, own], padded_len


def _full_token_logprobs(
    out: torch.Tensor,
    input_ids: torch.Tensor,
    cfg: MegatronLaneConfig,
    cp: tuple[int, int, Any],
    padded_len: int,
) -> torch.Tensor:
    """Next-token logprobs ``[B, S-1]`` over the FULL sequence on every CP rank."""
    import torch
    import torch.nn.functional as F
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    cp_size, cp_rank, cp_group = cp
    # Logits arrive [s, b, V/TP] with parallel_output=True; the FS helper takes
    # [B, S, Vs], so transpose to batch-first here.
    logits = out.transpose(0, 1).contiguous() if out.shape[0] != input_ids.shape[0] else out
    targets = _targets_from_input_ids(input_ids)
    if cp_size == 1:
        logits = logits[:, : input_ids.shape[1] - 1, :]
    else:
        # Position t scores token t+1; the final and pad positions score a dummy 0.
        targets = F.pad(targets, (0, padded_len - targets.shape[1]))
        targets = targets[:, targets.new_tensor(cp_shard_index(padded_len, cp_size, cp_rank))]
    logits = softcap(logits.to(torch.float32), cfg.softcap)
    tp_size = int(get_tensor_model_parallel_world_size())
    check_vocab_shard(
        shard_rows=logits.shape[-1],
        tp_size=tp_size,
        padded_vocab=logits.shape[-1] * tp_size,
        max_target_id=int(targets.max()) if targets.numel() else 0,
    )
    logprobs = vocab_parallel_token_logprobs(
        logits,
        targets,
        tp_group=get_tensor_model_parallel_group(),
        tp_rank=int(get_tensor_model_parallel_rank()),
        tp_size=tp_size,
        chunk_size=None,
    )
    if cp_size > 1:
        logprobs = _cp_gather_fn().apply(logprobs, padded_len, cp_group)
        logprobs = logprobs[:, : input_ids.shape[1] - 1]
    return logprobs


def make_forward_step(
    objective_loss_fn: Any,
    den: GlobalDenominators,
    cfg: MegatronLaneConfig,
    *,
    num_microbatches: int,
) -> ForwardStepFn:
    """Build the training forward_step for mcore's forward_backward_func.

    ``objective_loss_fn`` is a ``foundationscale.rl.torch_backend.TensorPolicyLoss``
    (or any callable with that exact keyword signature); it is invoked on the
    last PP stage with ``current_logprobs`` (grad-carrying), ``old_logprobs``,
    ``advantages``, ``mask``, and optionally ``reference_logprobs`` from the
    microbatch dict. The returned loss_func returns
    ``(rescaled_loss, {"loss", "ratio_mean", "clip_fraction", "tokens"})``
    with detached 0-dim metric tensors. ``rescaled_loss`` carries
    :func:`undo_mcore_loss_average` for ``num_microbatches``; the ``loss``
    metric is the un-inverted additive share.
    """
    import torch

    unit = loss_unit(objective_loss_fn)
    clip_bounds: tuple[float, float] = tuple(
        getattr(getattr(objective_loss_fn, "objective", None), "clip_bounds", (0.8, 1.2))
    )

    def forward_step(
        data_iterator: Iterator[dict[str, Any]], model: Any
    ) -> tuple[Any, Callable[..., Any]]:
        from megatron.core.parallel_state import (  # lazy: no megatron on CPU
            get_data_parallel_world_size,
        )

        batch = next(data_iterator)
        _require_batch_columns(
            batch,
            ("input_ids", "old_logprobs", "advantages", "loss_mask"),
            "forward_step(train)",
        )
        input_ids = batch["input_ids"]
        cp = _cp_layout()
        model_ids, model_pos, padded_len = _model_inputs(batch, cp[0], cp[1], sp_tp(cfg))
        output_tensor = model(
            input_ids=model_ids,
            position_ids=model_pos,
            attention_mask=None,
            packed_seq_params=batch.get("packed_seq_params"),
        )

        def loss_func(out: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            current_logprobs = _full_token_logprobs(out, input_ids, cfg, cp, padded_len)
            width = batch["loss_mask"].shape[1]
            if width == input_ids.shape[1]:
                loss_mask = batch["loss_mask"][:, 1:]
            elif width == input_ids.shape[1] - 1:
                loss_mask = batch["loss_mask"]
            else:
                raise ValueError(
                    f"loss_mask width {width} matches neither S={input_ids.shape[1]} nor S-1"
                )
            raw_loss = objective_loss_or_zero(
                objective_loss_fn,
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
            return (
                undo_mcore_loss_average(
                    scaled,
                    num_microbatches,
                    cp[0],
                    int(get_data_parallel_world_size(with_context_parallel=True)),
                ),
                metrics,
            )

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

    import torch

    def forward_step(
        data_iterator: Iterator[dict[str, Any]], model: Any
    ) -> tuple[Any, Callable[..., Any]]:
        batch = next(data_iterator)
        _require_batch_columns(batch, ("input_ids",), "forward_step(logprobs)")
        input_ids = batch["input_ids"]
        cp = _cp_layout()
        model_ids, model_pos, padded_len = _model_inputs(batch, cp[0], cp[1], sp_tp(cfg))
        output_tensor = model(
            input_ids=model_ids,
            position_ids=model_pos,
            attention_mask=None,
            packed_seq_params=batch.get("packed_seq_params"),
        )

        def collect_fn(out: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            logprobs = _full_token_logprobs(out, input_ids, cfg, cp, padded_len)
            logprobs = torch.cat([torch.zeros_like(logprobs[:, :1]), logprobs], dim=1).detach()
            zero = torch.zeros((), dtype=torch.float32, device=logprobs.device)
            return zero, {"logprobs": logprobs}

        return output_tensor, collect_fn

    return forward_step
