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
    from collections.abc import Callable

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
    "unshard_for_generation",
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


# Rank 0's host-side save work (device-to-host copies, the safetensors write)
# runs while its peers wait. That wait must not be bounded by the default
# NCCL watchdog: a 3-rank fsdp run on GB200 hit the 600 s watchdog while
# rank 0 was between two gathers, and the peers aborted a save rank 0 then
# finished (#453). Save waits use their own group with a long timeout.
# NCCL builds that group's communicator lazily, at its first collective, and
# the build waits on rank 0 under the STORE timeout (600 s), not the group's:
# a peer whose first save wait lands while rank 0 is busy dies there. So every
# save opens with an entry wait, reached in lockstep before any rank-0-only
# work, which builds the communicator while no rank is stalled.
SAVE_WAIT_TIMEOUT_S = 7200
# Rank 0 holds at most this many gathered bytes on device before copying
# them to host inside a save wait. Decided from GLOBAL shapes, so every rank
# flushes at the same keys.
SAVE_GATHER_CHUNK_BYTES = 2 << 30
_SAVE_WAIT_GROUPS: dict[int, Any] = {}


def _save_barrier(ctx: DistContext) -> None:
    import datetime

    import torch.distributed as dist

    if not ctx.is_distributed:
        return
    key = id(dist.group.WORLD)
    if key not in _SAVE_WAIT_GROUPS:
        _SAVE_WAIT_GROUPS[key] = dist.new_group(
            timeout=datetime.timedelta(seconds=SAVE_WAIT_TIMEOUT_S)
        )
    dist.barrier(group=_SAVE_WAIT_GROUPS[key])


def _gather_full_state_dict(
    model: Any, ctx: DistContext, save_dtype: torch.dtype
) -> dict[str, Any]:
    """Every rank all-gathers each shard in state-dict order; rank 0 alone
    keeps the result, cast to ``save_dtype`` on device.

    Collective: every rank must call it. Host copies happen only at chunk
    boundaries, followed by a long-timeout save wait, so no default-group
    collective is ever pending on a peer while rank 0 copies. Non-main
    ranks return an empty dict.
    """
    import torch
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    sharded = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))
    main = is_main(ctx)
    gathered: dict[str, Any] = {}
    pending: list[tuple[str, Any]] = []
    pending_bytes = 0

    def flush() -> None:
        if main:
            for name, tensor in pending:
                gathered[name] = tensor.cpu()
        pending.clear()
        _save_barrier(ctx)

    value: Any
    for key, value in sharded.items():
        # Tensor-like by duck type: a DTensor reports GLOBAL numel here.
        is_tensor = hasattr(value, "numel") and hasattr(value, "is_floating_point")
        floating = is_tensor and value.is_floating_point()
        if is_tensor:
            itemsize = (
                torch.empty((), dtype=save_dtype).element_size()
                if floating
                else value.element_size()
            )
            pending_bytes += value.numel() * itemsize
        full = value.full_tensor() if hasattr(value, "full_tensor") else value
        if main:
            pending.append((key, full.to(dtype=save_dtype) if floating else full))
        del full
        if pending_bytes >= SAVE_GATHER_CHUNK_BYTES:
            flush()
            pending_bytes = 0
    flush()
    return gathered


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


def _find_tower_modules(model: Any) -> list[Any]:
    """Non-language-model submodules (vision/audio towers, MTP heads) that
    need their OWN ``fully_shard`` unit, beyond the decoder blocks.

    MEASURED ROOT CAUSE (gemma-4-12B-it, FSDP2 + images, 2026-10-09):
    ``_no_split_modules`` on gemma4_unified names ONLY the text decoder
    layer class (``Gemma4UnifiedTextDecoderLayer``), so
    :func:`find_decoder_blocks` wraps the language model's blocks and
    nothing else -- the vision embedder (``model.embed_vision``) stayed
    inside the ROOT ``fully_shard`` unit, which only unshards its
    parameters when the ROOT's own ``forward()`` runs. ``generate()``'s
    multimodal preprocessing calls ``get_image_features()`` ->
    ``self.embed_vision(...)`` directly, BEFORE the decode loop's first
    ``forward()`` -- so the vision tower's parameters were still DTensor
    shards, not real tensors, the moment its own forward ran, and
    ``aten.native_layer_norm`` crashed on "mixed torch.Tensor and
    DTensor". CONFIRMED pre-existing and independent of LoRA: reproduced
    identically with ``adapter=None``.

    Resolved from the SAME family registry the SFT plane's LoRA target
    scoping already uses (:mod:`foundationscale.families`), not by name
    guessing here: every ``(dotted_path, modality)`` in the model's
    resolved ``FamilySpec.towers`` is tried via
    :func:`foundationscale.families.towers.resolve_module_path`,
    deduplicated by identity (a family may declare more than one dotted
    path for the SAME checkpoint attribute across its ``model_type``
    variants -- e.g. gemma4's ``model.vision_tower`` for E4B/26B/31B and
    ``model.embed_vision`` for the 12B/unified variant -- only one
    resolves on any given real checkpoint), and a resolved module with
    zero parameters is dropped: nothing to shard, nothing to unshard.

    Returns ``[]`` -- changing ``wrap_fsdp2``'s behaviour not at all --
    when the model's config names no registered family (an unregistered
    or text-only model has no declared extra towers) or the registered
    family declares none. This is a SAFE degradation, unlike the LoRA
    target selector's hard refusal for the same "family unknown" case: a
    tower this function fails to find stays in the root unit exactly as
    it does today, so the worst case is the SAME crash this function
    exists to fix, surfaced loudly on the first affected ``generate()``
    call -- never a silently wrong shard.

    MEASURED SECOND BUG, same date: calling this on a peft-wrapped model
    with ``model`` itself (rather than the ORIGINAL model peft wraps) made
    the first version of this fix a silent no-op. ``PeftModel`` has no
    ``.model`` attribute of its own; ``peft_model.model`` resolves through
    ``__getattr__`` delegation to ``peft_model.base_model.model`` -- which
    IS the original pre-peft model object, confirmed via
    ``peft_model.model is raw_model``. So a registered path like
    ``"model.embed_vision"``, meant to be read against the ORIGINAL model
    (``raw_model.model.embed_vision``), instead resolved as
    ``peft_model.model.embed_vision`` == ``raw_model.embed_vision`` -- one
    segment short, which does not exist on ``raw_model`` directly, so
    ``resolve_module_path`` correctly reported "absent" and this function
    returned ``[]`` even though the tower was real and reachable. Fixed by
    resolving against ``model.get_base_model()`` (peft's own accessor for
    the wrapped object) when present -- confirmed to return the IDENTICAL
    live module objects the wrapped tree actually holds
    (``base.model.embed_vision is raw_model.model.embed_vision``), so
    wrapping what this function finds really does wrap what
    ``generate()`` will later call.
    """
    from foundationscale.families import resolve_family
    from foundationscale.families.towers import resolve_module_path

    config_to_dict = getattr(getattr(model, "config", None), "to_dict", None)
    family_config: Any = config_to_dict() if callable(config_to_dict) else None
    if not isinstance(family_config, dict):
        family_config = {}
    family = resolve_family(family_config)
    if family is None:
        return []
    get_base_model = getattr(model, "get_base_model", None)
    resolve_root = get_base_model() if callable(get_base_model) else model
    found: list[Any] = []
    seen_ids: set[int] = set()
    for dotted_path, _modality in family.towers:
        resolved = resolve_module_path(resolve_root, dotted_path)
        if resolved is None or id(resolved) in seen_ids:
            continue
        if next(resolved.parameters(), None) is None:
            continue
        seen_ids.add(id(resolved))
        found.append(resolved)
    return found


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

    Decoder blocks AND any registered multimodal towers (see
    :func:`_find_tower_modules` -- the measured fix for FSDP2 + images
    crashing in ``generate()``) are sharded leaf-first and the root last,
    so parameters outside any of those units (embeddings, tied lm_head,
    final norm) live in the root unit and the tie survives. Cast before
    this call stays the caller's choice; this function performs it because
    every trainer wants exactly this order.
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
    for tower in _find_tower_modules(model):
        fully_shard(
            tower,
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


def unshard_for_generation(model: Any) -> Callable[[], None]:
    """Explicitly materialise the ROOT's (and every tower's) full
    parameters, for the duration of a ``generate()`` call that never
    triggers the root's own forward hook at all.

    MEASURED (gemma-4-12B-it AND Qwen3.6-27B, GRPO rollout, 2026-10-09),
    the ROOT CAUSE, found after three narrower-but-wrong theories (vision
    tower timing, lazy-init call order) each explained ONE symptom but not
    the next: ``peft.PeftModel.generate()`` is hard-coded to
    ``return self.get_base_model().generate(*args, **kwargs)`` -- it calls
    ``.generate()`` on the UNWRAPPED base model object, never on ``self``.
    ``fully_shard(model, ...)`` in :func:`wrap_fsdp2` was applied to the
    PEFT WRAPPER (``model`` there, at the point ``_apply_lora_adapter`` has
    already run) -- so the FSDP2 forward-pre-hook that unshards the root's
    own parameter group lives on the WRAPPER's ``__call__``, an object
    ``generate()`` never touches. CONFIRMED on a TEXT-ONLY model
    (Qwen3.6-27B, zero registered towers): the very same "mixed
    torch.Tensor and DTensor" crash occurs at ``embed_tokens`` on the
    FIRST ``generate()`` call, with no vision tower involved at all -- an
    earlier version of this function special-cased "no tower, no-op",
    which was precisely backwards; the root needs explicit unsharding for
    EVERY peft+FSDP2 model that calls ``generate()``, regardless of
    modality. (Decoder blocks and towers are each wrapped on the REAL
    nested module object, which is identical whether reached through the
    wrapper or through ``get_base_model()``, so their OWN per-unit hooks
    fire normally either way -- only the ROOT'S hook is the one
    ``generate()`` skips.)

    Explicitly ``unshard()``-ing the root (via its own ``FSDPModule.unshard``)
    before ``generate()`` materialises its group's parameters directly,
    without relying on a hook that will not fire. Towers are unsharded
    AFTER the root (not before): torch's ``_fsdp_state.py::_lazy_init``
    determines "the root" as whichever FSDP state's lazy-init runs FIRST,
    then walks that state's entire submodule tree marking every other FSDP
    state non-root; unsharding a tower first lets the tower self-elect as
    root instead, and the true root's later lazy-init then finds an
    inconsistent tree and raises "already been lazily initialized".
    Root-first avoids that: the true root's walk runs while every decoder
    block and tower is still untouched.

    A KNOWN REMAINING LIMITATION: on a real 30-step GRPO run with a vision
    tower (gemma-4-12B-it), steps 0 and 1 (both UNMEASURED -- every
    rollout row abstained, so neither reached a ``backward()``) completed
    cleanly with this fix, then step 2's ``generate()`` raised the
    "already lazily initialized" error again. torch's own docs
    (``FSDPModule.reset_iter_state``) name a forward running without a
    matching backward as leaving per-iteration trackers
    (``iter_forward_root`` among them) in an "undefined condition" --
    matching this plane's common case of consecutive forward-only
    rollouts. Calling ``reset_iter_state()`` after every ``generate()`` to
    clear that state was TRIED and made things WORSE: it raised
    ``AttributeError: 'FSDPCommContext' object has no attribute
    'all_gather_state'`` on the very first call, before any normal forward
    had ever primed that attribute -- its own documented precondition
    ("after an exception aborted a forward or backward mid-flight") is
    violated by this plane's common all-abstained-step case. NOT shipped.
    Until resolved, ``sharding='fsdp'`` for a VLM trained WITH images is
    validated only for short runs; the GPU-proven path for a longer
    VLM-with-images run today is ``sharding='ddp'``. TEXT-ONLY models are
    NOT affected by this remaining limitation -- their only FSDP2 unit this
    function touches is the root itself, which does not have the
    tower-self-election race to begin with.

    Returns a zero-argument callable that reshards every unit this
    unsharded, restoring the sharded (memory-saving) state from before --
    callers reshard in a ``finally`` so an exception inside ``generate()``
    does not leave parameters permanently unsharded. Finds zero units (a
    correct no-op) on a non-FSDP2 model: DDP and single-process runs never
    wrap anything in ``FSDPModule``.
    """
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:  # torch < 2.6 kept it in the composable namespace
        from torch.distributed._composable.fsdp import (  # type: ignore[no-redef,unused-ignore]
            FSDPModule,
        )

    # ROOT FIRST, towers after -- load-bearing order, see this function's
    # docstring: unsharding a tower first lets it self-elect as FSDP2's
    # root (root status is decided by whichever state's lazy-init runs
    # first), which then makes the TRUE root's own lazy-init raise when
    # its walk finds that tower already initialized.
    #
    # _find_tower_modules resolves towers from the FAMILY REGISTRY, which
    # is sharding-agnostic -- it finds the SAME vision embedder whether the
    # model was ever wrapped with fully_shard or not. MEASURED: under
    # sharding='ddp', that tower is a plain nn.Module (fully_shard was
    # never called on it), and unconditionally calling .unshard() on it
    # raised AttributeError. Filtered to isinstance(tower, FSDPModule) so
    # this function is a correct no-op whenever NEITHER the root NOR any
    # tower is actually FSDP2-wrapped, not just when no tower is declared.
    units: list[Any] = []
    if isinstance(model, FSDPModule):
        units.append(model)
    units.extend(tower for tower in _find_tower_modules(model) if isinstance(tower, FSDPModule))
    for unit in units:
        unit.unshard()

    def _reshard() -> None:
        for unit in units:
            unit.reshard()

    return _reshard


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


def _atomic_peft_save(target: Any, out_dir: str, state_dict: Any | None) -> None:
    """Adapter-only save, atomic: temp-dir-then-replace on the SAME filesystem.

    Reused, not reimplemented: ``foundationscale.train.fsdp_peft_save``'s
    SFT plane already built and tested this exact mechanism
    (``_atomically_overwrite_adapter`` -- write into a fresh ``mkdtemp``
    inside ``out_dir``, then ``Path.replace`` each file into place, which is
    an ``os.replace`` and therefore atomic because source and destination
    share a filesystem) for a DIFFERENT reason (FSDP1's per-layer wrap
    corrupts ``get_peft_model_state_dict``'s key derivation). That root
    cause does not apply here -- FSDP2's composable ``fully_shard`` never
    renames submodules, so ``target.save_pretrained`` derives correct
    adapter key prefixes directly from the live, wrapped model -- but the
    atomic-write MECHANISM has nothing FSDP1-specific about it, and a
    second, subtly different atomic-write routine written here would be
    exactly the kind of drift between two copies of one idea that this
    repository's doctrine refuses to let stand unmeasured.
    """
    from foundationscale.train.fsdp_peft_save import _atomically_overwrite_adapter

    _atomically_overwrite_adapter(target, out_dir, state_dict)


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

    none/ddp: rank 0 saves the (unwrapped) model directly. fsdp: a
    COLLECTIVE chunked gather (``_gather_full_state_dict``) -- every rank
    calls it -- leaves rank 0 holding the full state dict in the model's
    original load dtype (bf16; fp32 sharded masters are the training plane,
    not the artifact), which it saves as a standard ``save_pretrained``
    directory. An entry save wait builds the long-timeout group in lockstep;
    a trailing one means any rank proceeding past this call may load the
    directory. Every rank returns the same bool.
    """
    import torch

    save_dtype = torch.bfloat16 if save_dtype is None else save_dtype

    from pathlib import Path

    out = Path(out_dir)
    target = model.module if hasattr(model, "module") else model
    # A peft-wrapped target carries a non-empty peft_config: save_pretrained
    # on one writes ONLY adapter_model.safetensors + adapter_config.json
    # (peft's own get_peft_model_state_dict filters the state_dict it is
    # given down to the adapter tensors), which is the adapter-only
    # checkpoint contract -- and that write goes through the atomic
    # temp-dir-then-replace path, never a direct write into out_dir.
    is_peft = getattr(target, "peft_config", None) is not None
    _save_barrier(ctx)  # entry wait: builds the save group before rank 0 diverges
    if sharding == "fsdp":
        state_dict = _gather_full_state_dict(model, ctx, save_dtype)
        if is_main(ctx):
            out.mkdir(parents=True, exist_ok=True)
            if is_peft:
                _atomic_peft_save(target, str(out), state_dict)
            else:
                target.save_pretrained(out, state_dict=state_dict, safe_serialization=True)
    elif is_main(ctx):
        out.mkdir(parents=True, exist_ok=True)
        if is_peft:
            _atomic_peft_save(target, str(out), None)
        else:
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
    _save_barrier(ctx)
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
