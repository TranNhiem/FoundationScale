"""``foundationscale-train`` -- the thinnest real training entry point.

Installed as a console script by ``[project.scripts]``; the heavy dependencies
it needs are the ``train`` extra (``pip install 'foundationscale[train]'``).
Both are declared in pyproject.toml.

They were not always. This docstring used to carry those two stanzas as a note
prefaced "shipped as a note because the manifest is not modifiable from here" --
packaging written in the register of documentation, which reads like a decision
that was made rather than a task that was skipped. Nobody ran it, so for one
release ``prog="foundationscale-train"`` below advertised a command that ``pip
install`` never created, and loop.py's refusal path recommended an extra that
did not exist. Finding #224. The lesson is not "remember to edit pyproject" --
it is that a TODO indistinguishable from a description will be read as one, so
checks/packaging_reachability.py now asks the installed distribution whether
every advertised name resolves.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from foundationscale.train.loop import TrainConfig, fs_version, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="foundationscale-train",
        description=(
            "Thin real training path: validate Topology against a "
            "ClusterProfile BEFORE touching a GPU, then launch "
            "transformers.Trainer DDP with the FoundationScale save gate "
            "wired in as a Trainer callback. Exit codes: 0 PASS, 5 RED, "
            "95 UNMEASURED, 96 REFUSE."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {fs_version()}")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--dataset",
        required=True,
        help="HF dataset id, or a .json/.jsonl file, or a directory of them",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    # Machine facts: no defaults, fail closed (doctrine 4).
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile-name", help="name of a built-in ClusterProfile")
    group.add_argument("--profile-path", type=Path, help="path to a ClusterProfile JSON")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="run the ENTIRE validation prologue and exit without touching a GPU",
    )
    p.add_argument(
        "--launch-corpus",
        type=Path,
        default=None,
        help=(
            "directory of LAUNCHER scripts (.sh/.sbatch/.slurm) to scan for "
            "partition spelling variants. Omit it and the scan reports "
            "UNMEASURED -- it is never assumed clean"
        ),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig(
        model=args.model,
        dataset=args.dataset,
        output_dir=args.output_dir,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        profile_name=args.profile_name,
        profile_path=args.profile_path,
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        seed=args.seed,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        cp=args.cp,
        dry_run=args.dry_run,
        launch_corpus=args.launch_corpus,
    )
    return train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
