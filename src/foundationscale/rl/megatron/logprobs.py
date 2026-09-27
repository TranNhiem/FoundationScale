"""Vocab-parallel token logprobs for the Megatron RL lane.

Layout contract: logits are supplied as ``[B, S, Vs]`` where ``Vs`` is this
rank's contiguous vocabulary shard (``Vs == padded_vocab / TP``), and targets
are global token ids ``[B, S]`` aligned with the same positions. Callers that
need next-token semantics shift before calling; this kernel never silently
rolls sequences. The historical ``param.shape[0] == V/TP`` clamp class is a
refusal in :func:`check_vocab_shard`, never a repair.

WHAT IS CLAIMED: for ``tp_size == 1`` the result is exactly
``torch.log_softmax(logits.float(), -1).gather(targets)`` with no collective;
for ``tp_size > 1`` forward is the NeMo distributed log-softmax math (fp32
upcast, all_reduce MAX for the global max, all_reduce SUM for sum-exp, gather
only from the owning shard, all_reduce SUM of gathered values) and backward is
``(onehot_owned - softmax) * grad`` on the local shard. WHAT IS NOT CLAIMED:
any CP gathering, target shifting, or sequence packing; those live above this
kernels.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = (
    "check_vocab_shard",
    "softcap",
    "vocab_parallel_token_logprobs",
)


def softcap(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    """Apply ``cap * tanh(logits / cap)``; ``None`` and ``0`` are identity."""
    if cap is None:
        return logits
    cap_f = float(cap)
    if cap_f == 0.0:
        return logits
    if cap_f < 0.0:
        raise ValueError(
            f"softcap_negative_cap: cap={cap_f}; a negative softcap reflects "
            f"the logits and changes argmax order, so it is refused"
        )
    return cap_f * torch.tanh(logits / cap_f)


def check_vocab_shard(
    shard_rows: int,
    tp_size: int,
    padded_vocab: int,
    max_target_id: int,
) -> None:
    """Refuse the vocab-shard ambiguity instead of clamping target ids."""
    if int(tp_size) < 1:
        raise ValueError(
            f"vocab_shard_nonpositive_tp: tp_size={tp_size}; tensor parallel "
            f"size must be at least 1"
        )
    if int(shard_rows) < 1:
        raise ValueError(
            f"vocab_shard_empty: shard_rows={shard_rows}; a zero-row vocab "
            f"shard cannot own any target id"
        )
    implied = int(shard_rows) * int(tp_size)
    if implied != int(padded_vocab):
        raise ValueError(
            "vocab_shard_size_mismatch: "
            f"shard_rows*tp_size={implied} (shard_rows={shard_rows}, "
            f"tp_size={tp_size}) but padded_vocab={padded_vocab}; this is "
            f"the param.shape[0]=V/TP bug class, refused rather than clamped"
        )
    if int(max_target_id) >= int(padded_vocab) or int(max_target_id) < 0:
        raise ValueError(
            "vocab_shard_target_out_of_range: "
            f"max_target_id={max_target_id}, padded_vocab={padded_vocab}; "
            f"a target outside the padded vocab has no owning shard and is "
            f"refused, never clamped"
        )


@torch.no_grad()
def _distributed_log_softmax(logits_fp32: torch.Tensor, tp_group: Any) -> torch.Tensor:
    """Stable log-softmax across the TP vocab plane; input must be fp32."""
    logits_max = torch.amax(logits_fp32, dim=-1, keepdim=True)
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = logits_fp32 - logits_max
    sum_exp = shifted.exp().sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
    return shifted - sum_exp.log()


class _ShardVocabTokenLogprobs(torch.autograd.Function):
    """Differentiable gathered logprob with a vocab plane split across TP."""

    @staticmethod
    def forward(
        ctx: Any,
        logits_shard: torch.Tensor,
        targets: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
        tp_group: Any,
        chunk_size: int | None,
        inference_only: bool,
    ) -> torch.Tensor:
        batch, seq_len, shard_rows = logits_shard.shape
        owned = (targets >= int(vocab_start)) & (targets < int(vocab_end))
        local_targets = (targets - int(vocab_start)).clamp(min=0, max=shard_rows - 1)
        step = seq_len if chunk_size is None else int(chunk_size)
        out_chunks: list[torch.Tensor] = []
        for start in range(0, seq_len, step):
            stop = min(seq_len, start + step)
            lp = _distributed_log_softmax(
                logits_shard[:, start:stop, :].to(dtype=torch.float32), tp_group
            )
            vals = lp.gather(-1, local_targets[:, start:stop].unsqueeze(-1)).squeeze(-1)
            vals = vals * owned[:, start:stop].to(dtype=vals.dtype)
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=tp_group)
            out_chunks.append(vals)
        out = out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=1)
        if not inference_only:
            ctx.save_for_backward(logits_shard, owned, local_targets)
            ctx.tp_group = tp_group
            ctx.chunk_size = chunk_size if chunk_size is None else int(chunk_size)
        return out.contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[Any, ...]:
        (grad_output,) = grad_outputs
        logits_shard, owned, local_targets = ctx.saved_tensors
        tp_group = ctx.tp_group
        chunk_size = ctx.chunk_size
        batch, seq_len, shard_rows = logits_shard.shape
        step = seq_len if chunk_size is None else int(chunk_size)
        grad_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, seq_len, step):
                stop = min(seq_len, start + step)
                lp = _distributed_log_softmax(
                    logits_shard[:, start:stop, :].to(dtype=torch.float32), tp_group
                )
                probs = lp.exp()
                onehot = F.one_hot(local_targets[:, start:stop], num_classes=shard_rows).to(
                    dtype=torch.float32
                )
                onehot = onehot * owned[:, start:stop].unsqueeze(-1).to(dtype=torch.float32)
                g = grad_output[:, start:stop].to(dtype=torch.float32).unsqueeze(-1)
                grad_chunks.append((onehot - probs) * g)
        grad_input = (
            grad_chunks[0] if len(grad_chunks) == 1 else torch.cat(grad_chunks, dim=1)
        ).to(dtype=logits_shard.dtype)
        return grad_input, None, None, None, None, None, None


def vocab_parallel_token_logprobs(
    logits_shard: torch.Tensor,
    targets: torch.Tensor,
    tp_group: Any | None,
    tp_rank: int,
    tp_size: int,
    chunk_size: int | None = None,
    *,
    inference_only: bool = False,
) -> torch.Tensor:
    """Return fp32 global-id logprobs ``[B, S]`` from a ``[B, S, Vs]`` shard."""
    if logits_shard.dim() != 3:
        raise ValueError(
            f"logprob_layout: logits_shard has shape {tuple(logits_shard.shape)}; "
            f"this kernel declares [B, S, Vs] and does not accept [S, B, Vs]"
        )
    if tuple(targets.shape) != tuple(logits_shard.shape[:2]):
        raise ValueError(
            f"logprob_target_shape: targets {tuple(targets.shape)} but logits "
            f"are {tuple(logits_shard.shape[:2])}; every target must align "
            f"with exactly one (row, position)"
        )
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"logprob_target_dtype: targets dtype={targets.dtype}; global "
            f"token ids must be int32/int64"
        )
    if chunk_size is not None and int(chunk_size) < 1:
        raise ValueError(
            f"logprob_bad_chunk: chunk_size={chunk_size}; sequence chunking "
            f"requires a positive chunk"
        )
    targets_i64 = targets.to(dtype=torch.int64)
    if int(tp_size) == 1:
        return (
            torch.log_softmax(logits_shard.to(dtype=torch.float32), dim=-1)
            .gather(-1, targets_i64.unsqueeze(-1))
            .squeeze(-1)
        )
    if tp_group is None:
        raise ValueError(
            "logprob_missing_tp_group: tp_size>1 requires a torch.distributed "
            "process group; implying WORLD from a shard silently would hide a "
            "mis-sized model parallel group"
        )
    shard_rows = int(logits_shard.shape[-1])
    vocab_start = int(tp_rank) * shard_rows
    vocab_end = vocab_start + shard_rows
    return _ShardVocabTokenLogprobs.apply(
        logits_shard,
        targets_i64,
        vocab_start,
        vocab_end,
        tp_group,
        chunk_size,
        bool(inference_only),
    )
