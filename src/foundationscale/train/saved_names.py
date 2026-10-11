"""The names a full save writes, asked of transformers itself.

transformers >= 5.x converts some architectures' module layout on load (Qwen2-Audio:
``audio_tower.*`` on disk becomes ``model.audio_tower.*`` in memory, measured with 5.19) and
reverts it on save (``save_pretrained(save_original_format=True)``, the default). So the tensor
names a run holds in memory are not the names its checkpoint carries, and every gate that pairs
the two -- save completeness, speech tower movement -- must translate. The oracle is
``revert_weight_conversion``, the function ``save_pretrained`` calls; FS copies none of its
mapping tables (ledger ``hf-revert-weight-conversion``).
"""

from __future__ import annotations

from typing import Any

__all__ = ["saved_name_map"]


def saved_name_map(model: Any) -> dict[str, str] | None:
    """``in-memory name -> on-disk name`` for every renamed tensor, or None when nothing renames.

    Tensors are replaced by meta tensors of the same shape and dtype, so no weight is copied.
    A name whose reversal is not one-to-one (a many-to-one weight converter) is left out:
    the gates then see it under its in-memory name and report it, rather than FS guessing.
    Older transformers (no ``core_model_loading``) revert nothing: None.
    """
    try:
        from transformers.core_model_loading import (
            revert_weight_conversion,
        )
    except ImportError:
        return None
    import torch

    state = getattr(model, "state_dict", None)
    if not callable(state):
        return None
    renamed: dict[str, str] = {}
    for name, tensor in state().items():
        shape = tuple(getattr(tensor, "shape", ()))
        meta = torch.empty(shape, dtype=getattr(tensor, "dtype", torch.float32), device="meta")
        reverted = revert_weight_conversion(model, {name: meta})
        if len(reverted) == 1:
            (on_disk,) = reverted
            if on_disk != name:
                renamed[name] = on_disk
    return renamed or None
