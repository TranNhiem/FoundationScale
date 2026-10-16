#!/usr/bin/env python3
"""Print the verl-vla RL command FoundationScale renders for one published recipe.

    python render_rl.py --list
    python render_rl.py dsrl-pi05-libero-spatial-task2 --gpus 2 --out $VLA_RUNS/dsrl_task2 \
        --render osmesa

The recipe's own overrides stay verbatim; only placement changes (model GPUs, output dir,
renderer and the env vars Ray workers need). verl-vla's published curves are OSMesa-rendered:
EGL moved one checkpoint's start by ~26 points. Run the printed command from the verl-vla
checkout, inside verl-vla's environment.
"""

from __future__ import annotations

import argparse
import os
import shlex


def main() -> int:
    from foundationscale.vla.adapters.verlvla import (
        PUBLISHED_RECIPES,
        on_gpus,
        render_command,
        with_output_dir,
        with_runtime_env,
    )

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("recipe", nargs="?", choices=sorted(PUBLISHED_RECIPES))
    p.add_argument("--list", action="store_true")
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--out", default="./outputs/rl")
    p.add_argument("--render", choices=["osmesa", "egl"], default="osmesa")
    a = p.parse_args()
    if a.list or not a.recipe:
        for name, spec in PUBLISHED_RECIPES.items():
            c = spec.published
            curve = f"{c.start}/{c.trials} -> {c.end}/{c.trials} (best {c.best})"
            print(f"{name:40s} published {curve}")
        return 0
    env = {"MUJOCO_GL": a.render}
    if a.render == "osmesa":
        env["PYOPENGL_PLATFORM"] = "osmesa"
    for key in ("PYTHONPATH", "HF_HUB_CACHE", "LIBERO_CONFIG_PATH"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    placed = on_gpus(PUBLISHED_RECIPES[a.recipe], a.gpus)
    spec = with_runtime_env(with_output_dir(placed, a.out), env)
    print(shlex.join(render_command(spec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
