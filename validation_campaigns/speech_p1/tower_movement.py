"""Per-tower weight movement between a base and a trained Gemma-4 checkpoint.

Counts parameters whose bytes changed, bucketed by tower. Clipping buffers
(input_min/max, output_min/max) never train and are reported separately, so the
"moved" denominator is parameters, not checkpoint tensors.

Usage: tower_movement.py <base model.safetensors> <trained model.safetensors>
"""

from __future__ import annotations

import sys

import torch
from safetensors import safe_open

BUCKETS = ("audio_tower", "vision_tower", "embed_audio", "embed_vision")
BUFFER_SUFFIXES = ("input_min", "input_max", "output_min", "output_max")


def bucket(key: str) -> str:
    return next((b for b in BUCKETS if b in key), "language_model")


def main(base_path: str, new_path: str) -> None:
    base, new = safe_open(base_path, "pt"), safe_open(new_path, "pt")
    new_keys = set(new.keys())
    stats: dict[str, dict[str, int]] = {}
    for key in base.keys():  # noqa: SIM118 - safe_open handles are not mappings
        s = stats.setdefault(bucket(key), {"tensors": 0, "moved": 0, "buffers": 0, "missing": 0})
        if key not in new_keys:
            s["missing"] += 1
            continue
        s["tensors"] += 1
        if key.rsplit(".", 1)[-1] in BUFFER_SUFFIXES:
            s["buffers"] += 1
        elif not torch.equal(base.get_tensor(key), new.get_tensor(key)):
            s["moved"] += 1
    for name, s in sorted(stats.items()):
        print(
            f"{name:15s} tensors={s['tensors']} params={s['tensors'] - s['buffers']} "
            f"moved={s['moved']} buffers={s['buffers']} missing_in_ckpt={s['missing']}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
