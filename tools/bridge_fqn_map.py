#!/usr/bin/env python3
"""Emit the declared tensor set of a Megatron-Bridge model -- the ``--fqn-map``
producer that ``live_save_gate`` has pointed at all along.

Why this exists. ``checkpoint.save_complete`` compares a checkpoint against an
independent list of the tensors it must contain. For an HF checkpoint the base
model's header supplies that list. A Megatron checkpoint names its tensors
differently, and the adjudicator rightly refuses to guess a mapping, so on every
Bridge checkpoint the gate had nothing to compare against and VACUOUS-blocked.

Where the list comes from. The model DEFINITION: the same recipe that trains the
run, built through Bridge's own model builder on the meta device, with no weights
and no checkpoint anywhere in the loop. Each chunk's ``sharded_state_dict()`` names
exactly what a training save writes. DCP keys are global, so a map built at
tp=pp=ep=etp=cp=1 describes a checkpoint saved at any parallel layout of the same
model -- measured: a TP2/EP2/ETP2 and a TP2/PP2/ETP2 checkpoint of
gemma-4-26B-A4B each hold exactly the 928 tensors this declares, and a map with one
planted extra tensor blocks the gate.

Known limit, stated not hidden: the list shares Megatron's module code with the
run it judges. It cannot catch a module the code never builds; it catches every
shard the code built and the save did not write, which is the failure that has
actually occurred.

Run inside the Bridge container, one GPU (TransformerEngine imports need CUDA):

    python3 tools/bridge_fqn_map.py --recipe <recipe_fn> --hf-path <hf_dir> --out map.json

Exit contract: 0 written, 95 UNMEASURED (no tensors collected -- an empty
denominator is the vacuity this tool exists to remove, so it is refused, never
written).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_UNMEASURED = 95

_SINGLE_RANK_LAYOUT = (
    ("tensor_model_parallel_size", 1),
    ("pipeline_model_parallel_size", 1),
    ("expert_model_parallel_size", 1),
    ("expert_tensor_parallel_size", 1),
    ("context_parallel_size", 1),
    ("sequence_parallel", False),
)


def collect_keys(
    obj: Any, tensor_types: tuple[type, ...], object_types: tuple[type, ...]
) -> tuple[set[str], set[str]]:
    """Walk a sharded state dict; return (tensor keys, object keys).

    ``tensor_types`` must include the sharded-tensor FACTORY type, not only the
    plain sharded tensor: Megatron saves a gated ``linear_fc1`` (gate and up
    fused) through a factory that merges back to one tensor under its own key.
    Leaving the factory out silently drops those tensors -- 60 of 928 on the
    26B, which would then read as undeclared rather than as present."""
    tensors: set[str] = set()
    objects: set[str] = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
        elif isinstance(cur, tensor_types):
            tensors.add(cur.key)
        elif isinstance(cur, object_types):
            objects.add(cur.key)
    return tensors, objects


def build_map(
    chunks: Iterable[Any],
    tensor_types: tuple[type, ...],
    object_types: tuple[type, ...],
    *,
    recipe: str,
    hf_path: str,
) -> dict[str, Any]:
    """The ``--fqn-map`` document for a built model (a list of chunks)."""
    chunks = list(chunks)
    tensors: set[str] = set()
    objects: set[str] = set()
    for chunk in chunks:
        t, o = collect_keys(chunk.sharded_state_dict(), tensor_types, object_types)
        tensors |= t
        objects |= o
    return {
        "declared_fqns": sorted(t for t in tensors if "_extra_state" not in t),
        "declared_objects": len(objects),
        "source": {
            "kind": "megatron-bridge model definition (meta device, no weights)",
            "recipe": recipe,
            "hf_config": str(Path(hf_path) / "config.json"),
            "parallel_layout_built": "tp=pp=ep=etp=cp=1",
            "model_parameters": sum(1 for c in chunks for _ in c.parameters()),
        },
    }


def _build_chunks(recipe: str, hf_path: str) -> list[Any]:  # pragma: no cover - Megatron + GPU
    import torch

    for k, v in (
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29577"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
    ):
        os.environ.setdefault(k, v)
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)

    import megatron.bridge.recipes as recipes

    provider = getattr(recipes, recipe)(hf_path=hf_path).model
    for field, value in _SINGLE_RANK_LAYOUT:
        if hasattr(provider, field):
            setattr(provider, field, value)
    if hasattr(provider, "finalize"):
        provider.finalize()
    # Bridge's own builder, as training uses it: it attaches the process-group
    # collection the providers read and applies the same mixed-precision wrapper,
    # so the keys carry exactly the prefixes a training save writes.
    return provider.provide_distributed_model(
        wrap_with_ddp=False, bf16=True, init_model_with_meta_device=True
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs Megatron + a GPU
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recipe", required=True, help="megatron.bridge.recipes function name")
    ap.add_argument("--hf-path", required=True, help="HF model dir the recipe is built from")
    ap.add_argument("--out", required=True, help="where to write the --fqn-map JSON")
    a = ap.parse_args(argv)

    from megatron.core.dist_checkpointing.mapping import (
        ShardedObject,
        ShardedTensor,
        ShardedTensorFactory,
    )

    doc = build_map(
        _build_chunks(a.recipe, a.hf_path),
        (ShardedTensor, ShardedTensorFactory),
        (ShardedObject,),
        recipe=a.recipe,
        hf_path=a.hf_path,
    )
    if not doc["declared_fqns"]:
        print("bridge_fqn_map: UNMEASURED -- the model declared no tensors", file=sys.stderr)
        return EXIT_UNMEASURED
    with Path(a.out).open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(
        f"bridge_fqn_map: wrote {a.out}: {len(doc['declared_fqns'])} tensors, "
        f"{doc['declared_objects']} objects, {doc['source']['model_parameters']} parameters"
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
