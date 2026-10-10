"""Which on-disk format a Hugging Face policy can be saved back to.

transformers 5.x records the conversions it applied at load (``model._weight_conversions``)
and, by default, ``save_pretrained`` reverses them to write the checkpoint's original
layout. A dequantizing conversion has no inverse: gpt-oss loads its MXFP4 experts
(``*_blocks`` + ``*_scales``) as bf16 ``gate_up_proj``/``down_proj``, and the reverse
step drops those tensors without an error -- a gpt-oss-20b save lost 48 of 411 tensors
(38 GB of experts; the save gates refused it). Such a model is saved in its loaded,
dequantized form instead, which reloads bit-identically.
"""

from __future__ import annotations

from typing import Any

__all__ = ["can_save_original_format", "save_format_kwargs"]


def can_save_original_format(model: Any) -> bool:
    """False when a load-time conversion dequantized weights; ``save_pretrained`` cannot undo it."""
    for conversion in getattr(model, "_weight_conversions", None) or ():
        for operation in getattr(conversion, "operations", None) or ():
            if "dequantize" in type(operation).__name__.lower():
                return False
    return True


def save_format_kwargs(model: Any) -> dict[str, Any]:
    """Extra ``save_pretrained`` kwargs: ``save_original_format=False`` only when required.

    Empty otherwise, so the call is unchanged on transformers 4.x (which neither records
    load conversions nor accepts the keyword) and for every reversible load.
    """
    return {} if can_save_original_format(model) else {"save_original_format": False}
