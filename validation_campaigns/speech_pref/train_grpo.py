"""SP-3: GRPO-family RL of Qwen3-ASR with a WER reward (RLTrainer audio mode).

The Earnings-22 manifests carry the instruction in ``text`` and the transcript in ``answer``;
``--transcript-key`` names the transcript field and this writes the ``{audio, text}`` manifest the
audio leg reads (scoring against the instruction would be a silent, total mismeasurement).
Writes ``<save-dir>/history.json``; checkpoints are the trainer's (``step_<n>``, ``final``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


def write_rl_manifest(src: Path, dst: Path, transcript_key: str) -> int:
    rows = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for n, line in enumerate(fin, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get(transcript_key)
            if not isinstance(text, str) or not text.strip() or not r.get("audio"):
                print(f"REFUSE: {src}:{n} lacks audio or {transcript_key!r}", file=sys.stderr)
                raise SystemExit(96)
            fout.write(json.dumps({"audio": r["audio"], "text": text}, ensure_ascii=False) + "\n")
            rows += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SFT checkpoint (policy init)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--transcript-key", default="answer")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--algorithm", default="dr_grpo")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--logprob-micro-batch", type=int, default=0, help="0 = score all rows at once")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--runaway-penalty", type=float, default=1.0)
    ap.add_argument("--loop-penalty", type=float, default=1.0)
    ap.add_argument("--language", default="en")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from foundationscale.rl.rewards import WERReward
    from foundationscale.rl.trainer import RLTrainConfig, RLTrainer

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "rl_manifest.jsonl"
    rows = write_rl_manifest(Path(args.manifest), manifest, args.transcript_key)
    cfg = RLTrainConfig(
        model=args.model,
        dataset=str(manifest),
        algorithm=args.algorithm,
        reward=WERReward(runaway_penalty=args.runaway_penalty, loop_penalty=args.loop_penalty),
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        logprob_micro_batch=args.logprob_micro_batch,
        temperature=args.temperature,
        device="cuda",
        gradient_checkpointing=True,
        save_dir=args.save_dir,
        save_every=args.save_every,
        seed=args.seed,
        audio_column="audio",
        audio_language=args.language,
    )
    trainer = RLTrainer(cfg)
    reports = trainer.run()
    history = [dataclasses.asdict(r) for r in reports]  # StepReport: frozen, slotted
    (out / "history.json").write_text(json.dumps(history, indent=1, default=str))
    print(json.dumps({"manifest_rows": rows, "steps": len(reports)}))


if __name__ == "__main__":
    main()
