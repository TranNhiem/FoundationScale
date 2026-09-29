"""FSDP2/DDP multi-GPU data-parallel plumbing for the RL trainers.

This module holds the distributed half of the two RL loops: process-group
setup from torchrun's environment, prompt-level sharding, the small
``agree_*`` collectives that keep every rank on the same number of
forward/backward passes, FSDP2 ``fully_shard`` wrapping, checkpoint saving
(full and sharded), and logging reductions.

torch is imported at module scope deliberately: nothing in this module is
meaningful without it, and both trainers already import it lazily before
they import this file inside ``run()``.

WHAT IS CLAIMED: every process group is created HERE and only here, so a
trainer can destroy exactly what it created; ``shard_indices`` splits
PROMPTS (never completions -- a group-relative objective needs all G
completions of one prompt on one rank), every rank receives the same count,
and the dropped remainder is stated; ``agree_max``/``agree_all`` turn a
rank-local condition into one identical decision everywhere, which is what
keeps FSDP collectives from deadlocking when ranks disagree about how many
rows survived; under fsdp the model is cast to FP32 BEFORE sharding, so the
sharded parameters ARE the fp32 masters and ``MixedPrecisionPolicy`` casts
to bf16 for compute -- this replaces ``MasterWeightOptimizer``, whose host
copies would break the DTensor plane; ``save_checkpoint`` gathers the full
state dict collectively under fsdp (every rank must call it) while only
rank 0 writes, and every rank returns the same result.

WHAT IS NOT CLAIMED: pipeline or tensor parallelism (the mesh is 1-D);
CPU/gloo process groups (``nccl`` only); that generation length is
synchronised beyond ``synced_gpus=True``; or that any particular sharding
fits a particular model -- that is measured per run.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = (
    "DistContext",
    "agree_all",
    "agree_max",
    "all_reduce_mean",
    "all_reduce_sum",
    "barrier",
    "destroy",
    "find_decoder_blocks",
    "generate_kwargs_for",
    "init_distributed",
    "is_main",
    "save_checkpoint",
    "save_sharded_dcp",
    "shard_indices",
    "wrap_ddp",
    "wrap_fsdp2",
)

# The only values any config may declare. "none" is single-process by
# definition and refuses to run under torchrun rather than silently training
# N independent replicas of the same model.
VALID_SHARDING: tuple[str, str, str] = ("none", "ddp", "fsdp")


@dataclass(frozen=True)
class DistContext:
    """Everything a trainer needs to know about the process group.

    ``is_distributed`` is False for BOTH spellings of single-process: no
    torchrun, or torchrun refused. Every collective below is an identity
    under ``is_distributed=False``, so single-process tests exercise the
    same call sites multi-GPU runs use.
    """

    rank: int
    world_size: int
    local_rank: int
    device: torch.device | str
    is_distributed: bool
    # True only when init_distributed created the group in this process;
    # destroy() then owns the teardown and nobody else may call it.
    owns_process_group: bool = False


def _default_single_device() -> torch.device:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_distributed(sharding: str) -> DistContext:
    """Bring up the process group declared by ``sharding``, or refuse.

    The environment is torchrun's: ``WORLD_SIZE``/``RANK``/``LOCAL_RANK``.
    sharding="none" with WORLD_SIZE > 1 is REFUSED -- letting it through
    would train one full replica per rank with no gradient sharing, while
    every rank printed as if it were the run. That silent replicate is a
    worse failure than any loud one this function could raise.
    """
    import torch
    import torch.distributed as dist

    if sharding not in VALID_SHARDING:
        raise ValueError(f"sharding={sharding!r}: one of {VALID_SHARDING!r} is required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if sharding == "none":
        if world_size > 1:
            raise ValueError(
                f"sharding='none' but WORLD_SIZE={world_size} is set in the "
                "environment (torchrun): every rank would train a full "
                "independent replica with no gradient sharing and no signal "
                "that it happened. Set sharding to 'ddp' or 'fsdp', or "
                "unset WORLD_SIZE for a genuine single process."
            )
        return DistContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=_default_single_device(),
            is_distributed=False,
            owns_process_group=False,
        )
    if world_size <= 1:
        # torchrun absent: ddp/fsdp declared but only one process exists.
        # One rank IS the world, so the sharding is degenerate, not wrong.
        return DistContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=_default_single_device(),
            is_distributed=False,
            owns_process_group=False,
        )
    if not torch.cuda.is_available():
        raise ValueError(
            "sharding requires CUDA: the process group backend is 'nccl' and "
            "no CUDA device is visible"
        )
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
        is_distributed=True,
        owns_process_group=True,
    )


def is_main(ctx: DistContext) -> bool:
    return ctx.rank == 0


def barrier(ctx: DistContext) -> None:
    import torch.distributed as dist

    if ctx.is_distributed:
        dist.barrier()


def destroy(ctx: DistContext) -> None:
    # Tears down only a group THIS module's init created; a group owned by
    # an outer harness is left standing.
    import torch.distributed as dist

    if ctx.owns_process_group and dist.is_initialized():
        dist.destroy_process_group()


def shard_indices(n_items: int, ctx: DistContext) -> list[int]:
    """This rank's contiguous slice of ``range(n_items)``, equal counts.

    Sharding is over PROMPTS: a group-relative objective forms its baseline
    over all G completions of one prompt, so splitting completions across
    ranks would break the group. The remainder -- n_items mod world_size
    trailing prompts -- is dropped identically on every rank, and the drop
    is stated. Ranks holding UNEQUAL counts would desynchronise FSDP
    collectives; ranks holding different counts is the bug this guards.
    """
    if n_items < 0:
        raise ValueError(f"n_items={n_items}: a negative count is not meaningful")
    per_rank = n_items // ctx.world_size
    dropped = n_items - per_rank * ctx.world_size
    if dropped and is_main(ctx):
        print(
            f"[distributed] shard_indices: dropping {dropped} trailing item(s) "
            f"so {ctx.world_size} rank(s) hold exactly {per_rank} each",
            file=sys.stderr,
        )
    start = ctx.rank * per_rank
    return list(range(start, start + per_rank))


def _reduce_int(value: int, ctx: DistContext, op: Any) -> int:
    import torch
    import torch.distributed as dist

    if not ctx.is_distributed:
        return value
    tensor = torch.tensor([value], dtype=torch.long, device=ctx.device)
    dist.all_reduce(tensor, op=op)
    return int(tensor.item())


def agree_max(value: int, ctx: DistContext) -> int:
    """The maximum of ``value`` across ranks; identity when world_size == 1.

    Used to fix one shared micro-slice COUNT: ranks with fewer real slices
    pad with zero-weight dummies so every rank issues the same number of
    forward/backward passes and FSDP's collectives line up.
    """
    import torch.distributed as dist

    return _reduce_int(value, ctx, dist.ReduceOp.MAX)


def agree_all(flag: bool, ctx: DistContext) -> bool:
    """True iff ``flag`` is True on every rank; identity when world_size == 1.

    A rank-local skip/raise must NEVER act alone: decide locally, agree,
    then act identically on every rank.
    """
    import torch.distributed as dist

    return bool(_reduce_int(int(flag), ctx, dist.ReduceOp.MIN))


def all_reduce_mean(x: float, ctx: DistContext) -> float:
    import torch
    import torch.distributed as dist

    if not ctx.is_distributed:
        return x
    tensor = torch.tensor([x], dtype=torch.float64, device=ctx.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item()) / ctx.world_size


def all_reduce_sum(x: float, ctx: DistContext) -> float:
    import torch
    import torch.distributed as dist

    if not ctx.is_distributed:
        return x
    tensor = torch.tensor([x], dtype=torch.float64, device=ctx.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_grads_mean(module: Any, ctx: DistContext) -> None:
    """Average a REPLICATED module's gradients across ranks in one collective.

    For modules outside FSDP/DDP (the PPO value head), matching the mean
    FSDP/DDP apply to the wrapped policy. Every parameter takes
    part, a missing grad as zeros, so each rank sends the same-shaped buffer
    whatever its local backward touched.
    """
    import torch
    import torch.distributed as dist

    if not ctx.is_distributed:
        return
    params = list(module.parameters())
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
    flat = torch.cat([param.grad.reshape(-1) for param in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= ctx.world_size
    offset = 0
    for param in params:
        count = param.grad.numel()
        param.grad.copy_(flat[offset : offset + count].view_as(param.grad))
        offset += count


def find_decoder_blocks(model: Any) -> list[Any]:
    """The transformer decoder blocks to FSDP-wrap individually.

    Structural rule, in order: (1) every module whose class name appears in
    HF's ``model._no_split_modules`` -- outermost matches only, so a block
    nested inside a matched block is counted once; (2) when that attribute
    is absent or names nothing present, the children of the largest
    ``nn.ModuleList`` in the tree. Embeddings and ``lm_head`` are never
    returned: tied weights (``lm_head.weight is embed_tokens.weight``) must
    stay in the ROOT fully_shard unit, and wrapping either separately would
    break the tie. Finding nothing refuses -- silently sharding-as-one-unit
    is worse than a named failure for a model this size.
    """
    import torch

    names = list(getattr(model, "_no_split_modules", None) or [])
    found: list[Any] = []
    if names:
        wanted = set(names)

        def walk(module: Any) -> None:
            for child in module.children():
                if type(child).__name__ in wanted:
                    found.append(child)
                else:
                    walk(child)

        walk(model)
        if type(model).__name__ in wanted and not found:
            found.append(model)
        if found:
            return found
    largest: torch.nn.ModuleList | None = None
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList) and (
            largest is None or len(module) > len(largest)
        ):
            largest = module
    if largest is None or len(largest) == 0:
        raise ValueError(
            "no decoder blocks found: the model exposes neither usable "
            "_no_split_modules classes nor a non-empty nn.ModuleList to "
            "fully_shard per block"
        )
    return list(largest.children())


def wrap_fsdp2(
    model: Any,
    ctx: DistContext,
    *,
    reshard_after_forward: bool = True,
    param_dtype: torch.dtype | None = None,
    reduce_dtype: torch.dtype | None = None,
) -> Any:
    """Shard ``model`` with FSDP2 ``fully_shard`` over a 1-D world mesh.

    The model is cast to FP32 BEFORE sharding: the sharded DTensors then
    hold the fp32 master weights, and MixedPrecisionPolicy casts them to
    ``param_dtype`` for compute and reduces grads in ``reduce_dtype``. This
    IS the master-weight scheme -- it replaces ``MasterWeightOptimizer``
    under fsdp, whose host-side fp32 copies would break the DTensor plane.

    Decoder blocks are sharded leaf-first and the root last, so parameters
    outside any block (embeddings, tied lm_head, final norm) live in the
    root unit and the tie survives. Cast before this call stays the
    caller's choice; this function performs it because every trainer wants
    exactly this order.
    """
    import torch

    param_dtype = torch.bfloat16 if param_dtype is None else param_dtype
    reduce_dtype = torch.float32 if reduce_dtype is None else reduce_dtype

    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError:  # torch < 2.6 kept it in the composable namespace
        from torch.distributed._composable.fsdp import (  # type: ignore[no-redef,unused-ignore]
            MixedPrecisionPolicy,
            fully_shard,
        )
    from torch.distributed.device_mesh import init_device_mesh

    model = model.to(dtype=torch.float32)
    mesh = init_device_mesh("cuda", (ctx.world_size,))
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    for block in find_decoder_blocks(model):
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    return model


def wrap_ddp(model: Any, ctx: DistContext) -> Any:
    """DistributedDataParallel for the replica-with-shared-gradients option."""

    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(model, device_ids=[ctx.local_rank])


def generate_kwargs_for(ctx: DistContext, sharding: str) -> dict[str, Any]:
    """Extra kwargs for ``model.generate`` under ``sharding``.

    ``synced_gpus=True`` makes HF generate keep stepping while ANY rank is
    still generating; without it a rank whose sequences all hit EOS would
    exit the loop early and the others would deadlock inside the next FSDP
    all-gather. Only fsdp needs it: DDP does no collectives in no_grad mode.
    """
    if sharding == "fsdp" and ctx.world_size > 1:
        return {"synced_gpus": True}
    return {}


def save_checkpoint(
    model: Any,
    tokenizer_or_processor: Any,
    out_dir: str,
    ctx: DistContext,
    *,
    sharding: str,
    step: int,
    save_dtype: torch.dtype | None = None,
) -> bool:
    """Write a loadable checkpoint plus an ``fs_rl_checkpoint.json`` marker.

    none/ddp: rank 0 saves the (unwrapped) model directly. fsdp:
    ``get_model_state_dict(full_state_dict=True, cpu_offload=True)`` is a
    COLLECTIVE -- every rank calls it -- then rank 0 casts to the model's
    original load dtype (bf16; fp32 sharded masters are the training plane,
    not the artifact) and saves a standard ``save_pretrained`` directory.
    A trailing barrier means any rank proceeding past this call may load
    the directory. Every rank returns the same bool.
    """
    import torch

    save_dtype = torch.bfloat16 if save_dtype is None else save_dtype

    from pathlib import Path

    out = Path(out_dir)
    target = model.module if hasattr(model, "module") else model
    if sharding == "fsdp":
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)
        if is_main(ctx):
            out.mkdir(parents=True, exist_ok=True)
            cast = {
                key: (
                    value.to(dtype=save_dtype)
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value
                )
                for key, value in state_dict.items()
            }
            target.save_pretrained(out, state_dict=cast, safe_serialization=True)
    elif is_main(ctx):
        out.mkdir(parents=True, exist_ok=True)
        target.save_pretrained(out, safe_serialization=True)
    if is_main(ctx):
        out.mkdir(parents=True, exist_ok=True)
        if tokenizer_or_processor is not None and hasattr(
            tokenizer_or_processor, "save_pretrained"
        ):
            tokenizer_or_processor.save_pretrained(out)
        (out / "fs_rl_checkpoint.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "sharding": sharding,
                    "world_size": ctx.world_size,
                    "dtype": str(save_dtype),
                },
                indent=2,
            )
            + "\n"
        )
    barrier(ctx)
    return True


def save_sharded_dcp(
    model: Any,
    optimizer: Any,
    out_dir: str,
    ctx: DistContext,
) -> None:
    """Sharded ``torch.distributed.checkpoint`` save for the too-big-to-gather case.

    Keeps DTensor shards in each rank's local files; collective, every rank
    calls it, and the result is resumable with dcp.load rather than with
    ``from_pretrained``.
    """

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    model_sd, opt_sd = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=False)
    )
    dcp.save({"model": model_sd, "optimizer": opt_sd}, checkpoint_id=out_dir)
    barrier(ctx)
