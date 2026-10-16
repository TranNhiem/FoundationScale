#!/usr/bin/env python3
"""Print the fine-tune command FoundationScale renders for GR00T or openpi (the published LIBERO
recipes).

    python render_finetune.py gr00t  --gpus 4 --base $VLA_WEIGHTS/GR00T-N1.7-3B \
        --dataset $VLA_DATA/libero_spatial_no_noops_1.0.0_lerobot --out $VLA_RUNS/gr00t_spatial
    python render_finetune.py openpi --out $VLA_RUNS/openpi --assets $VLA_RUNS/openpi/assets

GR00T: stages the base checkpoint first (weights symlinked, only use_flash_attention patched --
aarch64 has no flash-attn) and renders torchrun over GR00T's own launcher, through FS's DDP hook
when more than one GPU trains. openpi: renders openpi's own scripts/train.py with the
pi05_libero config. Run the printed command from the backend's checkout, inside the backend's
own environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("backend", choices=["gr00t", "openpi"])
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--base", help="GR00T base checkpoint dir")
    p.add_argument("--stage", default="/tmp/gr00t_base_stage", help="GR00T staged copy of the base")
    p.add_argument("--dataset", help="GR00T LeRobot dataset dir (with meta/modality.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--assets", default="./assets", help="openpi norm-stats assets dir")
    a = p.parse_args()
    if a.backend == "gr00t":
        from foundationscale.vla.adapters.gr00t.finetune import (
            PUBLISHED_LIBERO_RECIPE,
            on_gpus,
            render_command,
            stage_base_checkpoint,
        )

        stage = stage_base_checkpoint(a.base, a.stage, use_flash_attention=False)
        spec = on_gpus(
            dataclasses.replace(
                PUBLISHED_LIBERO_RECIPE, base_model=str(stage), dataset=a.dataset, output_dir=a.out
            ),
            a.gpus,
        )
        print(shlex.join(render_command(spec, torchrun="torchrun")))
    else:
        from foundationscale.vla.adapters.openpi.finetune import (
            PUBLISHED_LIBERO_RECIPE,
            render_command,
            render_env,
        )

        spec = dataclasses.replace(
            PUBLISHED_LIBERO_RECIPE, checkpoint_base_dir=a.out, assets_base_dir=a.assets
        )
        env = " ".join(f"{k}={v}" for k, v in render_env(spec).items())
        print(env + " " + shlex.join(render_command(spec, python=".venv/bin/python")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
