"""Resumable training state for the Megatron RL online lane: weights + Adam moments + step.

The HF checkpoint written by :mod:`foundationscale.rl.megatron.save` is the trained
POLICY -- servable, but not resumable: it holds no optimizer state, so a run that
dies at step 180 of 300 restarts from step 0. This module writes the other kind of
checkpoint, the one a run continues from: the Megatron model shards and the
distributed optimizer's state through ``megatron.core.dist_checkpointing``, plus the
RL loop's own step counter.

The call sequence mirrors Megatron-Bridge's ``training/checkpointing.py``
(``generate_state_dict`` / ``_load_checkpoint_from_path``): unwrap the DDP chunks,
build each chunk's ``sharded_state_dict(metadata=...)`` under ``model`` (one chunk)
or ``model{i}`` (virtual pipeline chunks), then the optimizer's from that dict;
save with ``content_metadata`` so a load can rebuild the same layout; on load,
build the same sharded dict from the live objects, ``dist_checkpointing.load`` it,
and restore model then optimizer (the latter under ``torch.no_grad`` -- it copies
into the fp32 main params). The optimizer layout is ``fully_reshardable``, so a
checkpoint taken under one TP/PP/EP split can be loaded under another.

What is NOT restored, by design: RNG state. The lane draws prompts from
``sample_prompt_indices(seed, step)``, a pure function of the step, so the data order
continues exactly; sampled completions after a resume differ from the ones the dead
run would have drawn, which is a property of sampling, not a lost invariant.

The module imports without torch or megatron; both are imported inside the calls.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "LATEST_NAME",
    "latest_training_state",
    "load_training_state",
    "save_training_state",
    "state_dir_for_step",
]

# Rank 0 writes this tracker only after every rank has finished the save, so a
# reader never resumes from a half-written directory.
LATEST_NAME = "latest_step.txt"
_STATE_KEY = "fs_rl_state"


def state_dir_for_step(root: str | os.PathLike[str], step: int) -> Path:
    """The directory a training-state checkpoint for ``step`` lives in."""
    return Path(root) / f"step_{step:06d}"


def latest_training_state(root: str | os.PathLike[str]) -> Path | None:
    """The newest COMPLETE training-state checkpoint under ``root``, or ``None``.

    Completeness is the tracker, not the directory: a directory without a tracker
    entry pointing at it is a save that died part-way and must not be resumed.
    """
    tracker = Path(root) / LATEST_NAME
    if not tracker.is_file():
        return None
    text = tracker.read_text(encoding="utf-8").strip()
    if not text.isdigit():
        raise ValueError(f"{tracker}: expected a step number, found {text!r}")
    path = state_dir_for_step(root, int(text))
    if not path.is_dir():
        raise FileNotFoundError(f"{tracker} names step {text}, but {path} does not exist")
    return path


def _chunks(model: Any) -> list[Any]:
    """The bare mcore modules: DDP wrappers renamed every key under ``module.``."""
    chunks = model if isinstance(model, (list, tuple)) else [model]
    bare = []
    for chunk in chunks:
        while hasattr(chunk, "module"):
            chunk = chunk.module
        bare.append(chunk)
    return bare


def _metadata(pg: Any) -> dict[str, Any]:
    """Sharded-state metadata, identical on save and load (it changes key naming)."""
    return {
        "distrib_optim_sharding_type": "fully_reshardable",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
        "dp_cp_group": pg.dp_cp,
    }


def _sharded_state(
    model: Any, optimizer: Any, metadata: Mapping[str, Any], *, is_loading: bool
) -> dict[str, Any]:
    chunks = _chunks(model)
    state: dict[str, Any] = {}
    if len(chunks) == 1:
        state["model"] = chunks[0].sharded_state_dict(metadata=dict(metadata))
    else:
        for index, chunk in enumerate(chunks):
            state[f"model{index}"] = chunk.sharded_state_dict(metadata=dict(metadata))
    optim_kwargs: dict[str, Any] = {"metadata": dict(metadata)}
    if is_loading:
        optim_kwargs["is_loading"] = True
    state["optimizer"] = optimizer.sharded_state_dict(state, **optim_kwargs)
    return state


def save_training_state(
    root: str | os.PathLike[str],
    model: Any,
    optimizer: Any,
    pg: Any,
    *,
    step: int,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """COLLECTIVE: write model shards, optimizer state and the loop step for ``step``.

    Every rank must call this at the same step. ``extra`` must be JSON-serialisable;
    it rides in the checkpoint's common (rank-0) state beside the step.
    """
    import torch.distributed as dist
    from megatron.core import dist_checkpointing

    path = state_dir_for_step(root, step)
    if (Path(root) / LATEST_NAME).is_file() and latest_training_state(root) == path:
        raise FileExistsError(f"{path} is already the latest training state")
    metadata = _metadata(pg)
    state = _sharded_state(model, optimizer, metadata, is_loading=False)
    state[_STATE_KEY] = json.dumps({"step": int(step), **dict(extra or {})})
    if not dist.is_initialized() or dist.get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    content = {key: value for key, value in metadata.items() if key != "dp_cp_group"}
    dist_checkpointing.save(state, str(path), content_metadata=content)
    if dist.is_initialized():
        dist.barrier()
    if not dist.is_initialized() or dist.get_rank() == 0:
        tracker = Path(root) / LATEST_NAME
        tmp = tracker.with_name(f".{LATEST_NAME}.tmp")
        tmp.write_text(f"{int(step)}\n", encoding="utf-8")
        tmp.replace(tracker)
    if dist.is_initialized():
        dist.barrier()
    return path


def load_training_state(
    path: str | os.PathLike[str], model: Any, optimizer: Any, pg: Any
) -> dict[str, Any]:
    """COLLECTIVE: restore model shards and optimizer state; return the saved loop state.

    Strict on both sides: a key the checkpoint lacks, or one the live model does not
    expect, is a different model or a different parallel layout the reshard could not
    map, and resuming into it would silently train from partly-initial weights.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import dist_checkpointing

    metadata = _metadata(pg)
    state = _sharded_state(model, optimizer, metadata, is_loading=True)
    state[_STATE_KEY] = None
    loaded = dist_checkpointing.load(state, str(path))
    chunks = _chunks(model)
    if len(chunks) == 1:
        chunks[0].load_state_dict(loaded["model"], strict=True)
    else:
        for index, chunk in enumerate(chunks):
            chunk.load_state_dict(loaded[f"model{index}"], strict=True)
    with torch.no_grad():
        optimizer.load_state_dict(loaded["optimizer"])
    if dist.is_initialized():
        dist.barrier()
    raw = loaded.get(_STATE_KEY)
    if not isinstance(raw, str):
        raise ValueError(f"{path}: no {_STATE_KEY!r} entry; not a training-state checkpoint")
    restored: dict[str, Any] = json.loads(raw)
    return restored
