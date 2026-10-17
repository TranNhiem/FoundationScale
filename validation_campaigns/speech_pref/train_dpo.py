"""SP-2: DPO-family preference training of Qwen3-ASR on audio pairs from build_pairs.py.

Thin runner over ``PreferenceTrainer`` in audio mode (``audio_column``/``audio_language``); the
trainer owns every gate and refusal. Writes ``<save-dir>/history.json`` (per-step loss, accuracy,
margin) next to the checkpoints the trainer saves (``step_<n>`` every --save-every, ``final``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SFT checkpoint (the reference policy)")
    ap.add_argument("--pairs", required=True, help="build_pairs.py output")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--algorithm", default="dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=0.0, help="NLL anchor on the chosen side (RPO)")
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--pairs-per-step", type=int, default=16)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--logprob-micro-batch", type=int, default=4)
    ap.add_argument("--language", default="en")
    ap.add_argument("--adapter", default=None, choices=[None, "lora"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from foundationscale.rl.preference_trainer import PreferenceTrainConfig, PreferenceTrainer

    cfg = PreferenceTrainConfig(
        model=args.model,
        dataset=args.pairs,
        algorithm=args.algorithm,
        beta=args.beta,
        sft_weight=args.sft_weight,
        learning_rate=args.learning_rate,
        pairs_per_step=args.pairs_per_step,
        max_steps=args.max_steps,
        max_length=args.max_length,
        logprob_micro_batch=args.logprob_micro_batch,
        device="cuda",
        gradient_checkpointing=True,
        adapter=args.adapter,
        save_dir=args.save_dir,
        save_every=args.save_every,
        seed=args.seed,
        audio_column="audio",
        audio_language=args.language,
    )
    trainer = PreferenceTrainer(cfg)
    trainer.run()
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(trainer.history, indent=1, default=str))
    last = trainer.history[-1] if trainer.history else {}
    print(json.dumps({k: last.get(k) for k in ("step", "loss", "accuracy", "margin")}))


if __name__ == "__main__":
    main()
