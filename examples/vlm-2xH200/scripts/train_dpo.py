#!/usr/bin/env python3
"""train_dpo.py — preference-alignment training (DPO / IPO / ORPO / SimPO / CPO) + LoRA.

Text-only preference training on 2x H200 with FSDP2, bf16 and gradient
checkpointing (all verified for this Foundationscale preference trainer).

Launch with:

    torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
        train_dpo.py --model ... --data ... --output-dir ...

Note: --master_addr must be 127.0.0.1 rather than "localhost" because
"localhost" resolves to an IPv6-only entry on the competition VM and torchrun's
rendezvous TCP store fails to bind/connect over it.

All algorithms exposed through --algorithm ("dpo", "ipo", "orpo", "simpo",
"cpo") consume prompt/chosen/rejected pairs. KTO is intentionally not exposed
here because KTO requires unpaired single-labelled (desired/undesired) data and
this script has no CLI for such an unpaired dataset format.

Run with torchrun --nproc_per_node 2. Rank 0 writes result.json into --output-dir.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from foundationscale.rl.preference_trainer import PreferenceTrainConfig, PreferenceTrainer
from foundationscale.rl.trainer import TrainerRefusal


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preference training (DPO, IPO, ORPO, SimPO, CPO) with LoRA."
    )
    p.add_argument("--model", required=True, help="HF model name or local model directory.")
    p.add_argument("--data", required=True, help="JSONL dataset of prompt/chosen/rejected triples.")
    p.add_argument(
        "--output-dir", required=True, help="Save/scratch directory (result.json lands here)."
    )
    p.add_argument(
        "--algorithm",
        choices=["dpo", "ipo", "orpo", "simpo", "cpo"],
        default="dpo",
        help="Preference objective; all use prompt/chosen/rejected pairs (KTO is not exposed).",
    )
    p.add_argument("--max-steps", type=int, default=30, help="Optimisation steps to run.")
    p.add_argument("--max-length", type=int, default=4096, help="Sequence truncation length.")
    p.add_argument("--learning-rate", type=float, default=1e-4, help="LoRA learning rate.")
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA adapter rank.")
    p.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA adapter alpha.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = PreferenceTrainConfig(
        model=args.model,
        dataset=args.data,
        algorithm=args.algorithm,
        learning_rate=args.learning_rate,
        pairs_per_step=4,
        max_steps=args.max_steps,
        max_length=args.max_length,
        logprob_micro_batch=1,
        sharding="fsdp",
        gradient_checkpointing=True,
        adapter="lora",
        adapter_rank=args.lora_rank,
        adapter_alpha=args.lora_alpha,
        save_dir=args.output_dir,
        save_every=0,
        seed=0,
    )

    if _is_rank0():
        print(
            f"[train_dpo] model={args.model} algorithm={args.algorithm} max_steps={args.max_steps}",
            file=sys.stderr,
        )

    trainer = PreferenceTrainer(cfg)
    t0 = time.time()
    try:
        reports = trainer.run()
    except TrainerRefusal as exc:
        # FoundationScale refuses rather than reporting a run that measured nothing
        # (for GRPO: every group abstained or had identical rewards). Exit 96 = refused.
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 96
    t1 = time.time()

    if _is_rank0():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        history = trainer.history
        result = {
            "ok": True,
            "model": args.model,
            "algorithm": args.algorithm,
            "max_length": args.max_length,
            "logprob_micro_batch": cfg.logprob_micro_batch,
            "n_steps_measured": len(reports),
            "wall_s": t1 - t0,
            "s_per_step": (t1 - t0) / max(1, len(reports)),
            "history_first": history[0] if history else None,
            "history_last": history[-1] if history else None,
            "ln2": math.log(2),
            "step0_minus_ln2": (history[0]["loss"] - math.log(2)) if history else None,
        }
        out_path = Path(args.output_dir) / "result.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print("RESULT_JSON " + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
