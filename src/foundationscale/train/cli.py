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
    # Defaulted here and NOT on TrainConfig, and the asymmetry is deliberate.
    # This entry implements exactly one objective -- supervised fine-tuning, via
    # DataCollatorForLanguageModeling(mlm=False) -- so "sft" is a statement about
    # what the code does, not a plausible guess. A programmatic caller building
    # TrainConfig directly gets no such default: it may be driving a path this
    # entry does not, so it has to say which, and an unstated objective is
    # refused by the objective gate at the first observed step.
    p.add_argument(
        "--objective",
        default="sft",
        help=(
            "what the run optimises; recorded with provenance and checked by the "
            "objective gates at the first observed step (default: sft, the only "
            "objective this entry implements)"
        ),
    )
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
    # Precision and adapter flags default to None, NOT to a plausible value.
    # None means "not declared": the manifest records None, and the
    # observed-vs-declared precision check abstains rather than passing
    # (an unstated precision must never be laundered into "bf16" -- #342).
    p.add_argument(
        "--precision",
        default=None,
        help=(
            "declared training precision, one of: bf16, fp16, fp32, nvfp4. "
            "nvfp4 is declarable but train() refuses it (96) until the package "
            "plane grows an nvfp4 backend -- it never falls back silently. "
            "Omit to declare nothing"
        ),
    )
    p.add_argument(
        "--adapter",
        default=None,
        help=(
            "adapter mode, one of: lora. Omit for a full fine-tune -- that is "
            "the explicit default, not an unstated one. Requires --adapter-rank; "
            "a missing peft install is a REFUSE (96), never a silent full "
            "fine-tune"
        ),
    )
    p.add_argument("--adapter-rank", type=int, default=None)
    p.add_argument("--adapter-alpha", type=float, default=None)
    p.add_argument(
        "--adapter-target",
        action="append",
        default=None,
        help=(
            "LoRA target module pattern; repeatable, collected in order. "
            "Omit to use peft's per-model defaults. If the adapter resolves "
            "to zero modules the run is refused after wrapping"
        ),
    )
    p.add_argument("--adapter-dropout", type=float, default=None)
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
        objective=args.objective,
        seed=args.seed,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        cp=args.cp,
        dry_run=args.dry_run,
        launch_corpus=args.launch_corpus,
        precision=args.precision,
        adapter=args.adapter,
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        adapter_targets=(tuple(args.adapter_target) if args.adapter_target is not None else None),
        adapter_dropout=args.adapter_dropout,
    )
    return train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
