#!/usr/bin/env python3
"""Tutorial CLI driver: GRPO + LoRA + images (verified FoundationScale re-proof, item 4).

This is the cleaned-up tutorial version of the verified GPU re-proof (item 4):
GRPO + LoRA multimodal RL, after the generate()-eval-mode fix (item 1) and the
FSDP2-vision-tower fix (item 2). group_size=4, >=4 prompts/step, 30 steps.
It uses the MCQLetterReward answer pattern ``r"Answer:\\s*([A-Z])"`` against the
"Answer: X" instruction format of the bundled RL data.

Launch line (exactly as validated):

    torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
        train_grpo.py --model ...

Concretely, e.g.:

    torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
        train_grpo.py --model /workspace/fs-vlm/models/gemma-4-12B-it \
                      --data /workspace/fs-vlm/data/rl_scienceqa/scienceqa_train.jsonl \
                      --output-dir /workspace/fs-vlm/runs/grpo_lora_proof_v2

Why --master_addr 127.0.0.1: on the competition VM `localhost` is IPv6-only
(::1), so the torch rendezvous on `localhost` fails to bind/connect; 127.0.0.1
is the working IPv4 loopback and is what the validated run uses.

Why sharding='ddp': 'ddp' is the validated choice here — repeated generation
under FSDP is not yet reliable (sampled rollouts are produced by generate()
while the module is sharded), so this driver keeps the proven DDP setup.

Rank 0 writes a result.json summary (per-step loss/reward/KL, first5/last5)
including the step-0 sanity value. RNG seed is fixed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

from foundationscale.rl.trainer import RLTrainConfig, RLTrainer, TrainerRefusal

# MCQLetterReward pattern, unchanged from the verified re-proof.
ANSWER_PATTERN = r"Answer:\s*([A-Z])"

# Algotathm families exposed by foundationscale.rl.trainer.RLTrainConfig.
ALGORITHMS = ["grpo", "dr_grpo", "dapo", "gspo", "reinforce_pp", "rloo"]


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _supports(cls: type, name: str) -> bool:
    """True if `cls` is a dataclass with a field named `name` (never invent fields)."""
    try:
        return any(f.name == name for f in dataclasses.fields(cls))
    except TypeError:  # non-dataclass: pass the field through
        return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO + LoRA RL training tutorial (FoundationScale verified re-proof, item 4).",
    )
    # Paths / run length (originally env-var / hard-coded driven).
    p.add_argument("--model", required=True, help="HF-style model dir or id (was GRPO_MODEL).")
    p.add_argument(
        "--data", required=True, help="RL jsonl dataset (was the hard-coded scienceqa_train.jsonl)."
    )
    p.add_argument("--output-dir", required=True, help="Save/report directory (was GRPO_SAVE_DIR).")
    p.add_argument(
        "--max-steps", type=int, default=30, help="RL update steps (was GRPO_MAX_STEPS)."
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Generation length per rollout (RLTrainConfig.max_new_tokens).",
    )

    # Algorithm / sampling.
    p.add_argument(
        "--algorithm", choices=ALGORITHMS, default="grpo", help="RL algorithm (validated: grpo)."
    )
    p.add_argument(
        "--group-size", type=int, default=4, help="Rollouts per prompt (was group_size)."
    )
    p.add_argument(
        "--prompts-per-step",
        type=int,
        default=8,
        help="Prompts per step (was GRPO_PROMPTS_PER_STEP).",
    )

    # Optimizer / adapter.
    p.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    p.add_argument(
        "--lora-rank", type=int, default=16, help="LoRA rank (RLTrainConfig.adapter_rank)."
    )
    p.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="Optional LoRA alpha; only forwarded if RLTrainConfig defines the field.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg_kwargs = dict(
        model=args.model,
        dataset=args.data,
        gold_key="gold",
        algorithm=args.algorithm,
        group_size=args.group_size,
        answer_pattern=ANSWER_PATTERN,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        prompts_per_step=args.prompts_per_step,
        logprob_micro_batch=2,
        sharding="ddp",  # validated: repeated generation under fsdp is not yet reliable.
        gradient_checkpointing=True,
        adapter="lora",
        adapter_rank=args.lora_rank,
        save_dir=args.output_dir,
        save_every=0,
        seed=0,
    )
    if args.lora_alpha is not None:
        # Only forward an alpha if the config actually has that field (never invent fields).
        for alpha_field in ("adapter_alpha", "lora_alpha"):
            if _supports(RLTrainConfig, alpha_field):
                cfg_kwargs[alpha_field] = args.lora_alpha
                break

    cfg = RLTrainConfig(**cfg_kwargs)

    if _is_rank0():
        print(
            f"[train_grpo] model={args.model} algorithm={args.algorithm} "
            f"group_size={args.group_size} max_steps={args.max_steps} "
            f"prompts_per_step={args.prompts_per_step} sharding=ddp "
            f"gradient_checkpointing=True answer_pattern={ANSWER_PATTERN!r}",
            file=sys.stderr,
        )

    trainer = RLTrainer(cfg)
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
        per_step = []
        for r in reports:
            entry = {"step": r.step, "loss": r.loss.loss, "rows": r.rows}
            if r.reward_stats is not None:
                entry["reward_mean"] = r.reward_stats.mean
                entry["reward_std"] = r.reward_stats.std
                entry["reward_min"] = r.reward_stats.minimum
                entry["reward_max"] = r.reward_stats.maximum
            kl_component = next(
                (c.contribution for c in r.loss.components if "kl" in c.name.lower()), None
            )
            if kl_component is not None:
                entry["kl_contribution"] = kl_component
            per_step.append(entry)
        result = {
            "ok": True,
            "model": args.model,
            "algorithm": args.algorithm,
            "group_size": args.group_size,
            "prompts_per_step": args.prompts_per_step,
            "n_steps_attempted": args.max_steps,
            "n_steps_measured": len(reports),
            "n_steps_abstained": args.max_steps - len(reports),
            "wall_s": t1 - t0,
            "s_per_step": (t1 - t0) / max(1, len(reports)),
            "per_step": per_step,
            "step0_sanity": per_step[0] if per_step else None,
            "first5": per_step[:5],
            "last5": per_step[-5:],
        }
        out_path = Path(args.output_dir) / "result.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print("RESULT_JSON " + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
