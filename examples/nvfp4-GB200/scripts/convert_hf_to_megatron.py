#!/usr/bin/env python3
"""Convert a Hugging Face checkpoint to a Megatron checkpoint with Megatron-Bridge.

usage: python scripts/convert_hf_to_megatron.py HF_MODEL_DIR MEGATRON_OUT_DIR

Run inside the NeMo Framework container. Upstream's ``scripts/conversion/convert.sh``
does the same through ``uv run``, which needs a writable uv cache the container does
not have; this calls the same API directly.
"""

from __future__ import annotations

import sys

from megatron.bridge import AutoBridge

if __name__ == "__main__":  # the checkpoint writer spawns workers that re-import this file
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    AutoBridge.import_ckpt(sys.argv[1], sys.argv[2])
    print(f"wrote Megatron checkpoint to {sys.argv[2]}")
