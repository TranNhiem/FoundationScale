"""NeMo-lane fine-tune for Canary (FoundationScale feeds the data and adjudicates the result).

1. Convert a FoundationScale audio JSONL (audio, answer, duration) to a NeMo AED manifest under
   the same strict rules as the HF loader: 16 kHz mono only, 1-25 s, non-empty text; every refused
   row is COUNTED and recorded (never silently dropped).
2. Fine-tune through NeMo's own API (setup_training_data / setup_optimization / Lightning Trainer)
   and save a .nemo. NeMo owns the training loop here; FoundationScale does not wrap it.

Usage: nemo_finetune.py --model nvidia/canary-1b-flash --train JSONL --out-dir DIR
                        [--max-steps 500 --batch-size 8 --lr 1e-5 --warmup 50 --pnc no]
                        [--save-every N]  (also save DIR/step{N}.nemo, for checkpoint selection)
                        [--freeze PREFIX ...]  (e.g. transf_decoder; adjudicate with --frozen)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(
    train_jsonl: Path, out_manifest: Path, pnc: str, max_duration: float = 25.0
) -> dict[str, object]:
    import soundfile as sf

    refused: dict[str, int] = {}
    kept = 0
    rows = [json.loads(line) for line in train_jsonl.read_text().splitlines()]
    with out_manifest.open("w") as f:
        seen_paths: set[str] = set()
        for r in rows:
            info = sf.info(r["audio"])
            reason = None
            # A repeated path means two transcripts share one recording: every row still "loads",
            # so row coverage cannot see it. Measured: a builder bug left 6,000 rows on 820 files.
            if r["audio"] in seen_paths:
                reason = "duplicate_audio_path"
            elif info.samplerate != 16000:
                reason = "sample_rate_mismatch"
            elif info.channels != 1:
                reason = "not_mono"
            elif not (1.0 <= info.duration <= max_duration):
                reason = "duration_out_of_range"
            elif not str(r.get("answer", "")).strip():
                reason = "empty_text"
            if reason:
                refused[reason] = refused.get(reason, 0) + 1
                continue
            seen_paths.add(r["audio"])
            f.write(
                json.dumps(
                    {
                        "audio_filepath": r["audio"],
                        "duration": info.duration,
                        "text": r["answer"],
                        "source_lang": "en",
                        "target_lang": "en",
                        "taskname": "asr",
                        "pnc": pnc,
                    }
                )
                + "\n"
            )
            kept += 1
    return {
        "rows_expected": len(rows),
        "rows_checked": kept,
        "rows_refused": sum(refused.values()),
        "refused": refused,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--pnc", default="no")
    ap.add_argument("--save-every", type=int, default=0)
    # Longest clip accepted, in seconds. 25 s is the clip recipe; long-form training raises it
    # to 40 s (speech_longform), and a longer clip is refused and counted, never truncated.
    ap.add_argument("--max-duration", type=float, default=25.0)
    ap.add_argument("--freeze", action="append", default=[])
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coverage = convert(Path(args.train), out / "train_manifest.json", args.pnc, args.max_duration)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print("COVERAGE", json.dumps(coverage))

    import lightning.pytorch as pl
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf, open_dict

    model = ASRModel.from_pretrained(args.model, map_location="cpu")
    train_cfg = (
        OmegaConf.create(OmegaConf.to_container(model.cfg.train_ds, resolve=True))
        if model.cfg.get("train_ds") is not None
        else OmegaConf.create({})
    )
    with open_dict(train_cfg):
        train_cfg.manifest_filepath = str(out / "train_manifest.json")
        train_cfg.use_lhotse = True
        train_cfg.batch_size = args.batch_size
        train_cfg.shuffle = True
        train_cfg.num_workers = 4
        train_cfg.max_duration = args.max_duration
        train_cfg.prompt_format = model.cfg.get("prompt_format", train_cfg.get("prompt_format"))
        # The checkpoint's own train_ds names the transcript field "answer"; this manifest writes
        # it as "text". Left inherited, NeMo reads an absent field: targets would be empty and
        # nothing would say so. Declared explicitly instead.
        train_cfg.text_field = "text"
        for k in (
            "batch_duration",
            "use_bucketing",
            "bucket_duration_bins",
            "tarred_audio_filepaths",
            "is_tarred",
            "input_cfg",
            "bucket_batch_size",
            "max_tps",
            "num_buckets",
        ):
            train_cfg.pop(k, None)
    print("TRAIN_DS", OmegaConf.to_yaml(train_cfg).replace("\n", " | ")[:600])
    model.setup_training_data(train_cfg)
    for prefix in args.freeze:
        hit = [p for n, p in model.named_parameters() if n == prefix or n.startswith(prefix + ".")]
        if not hit:  # a typo would freeze nothing and train everything, silently
            raise SystemExit(f"REFUSE (96): --freeze {prefix!r} matches no parameter")
        for p in hit:
            p.requires_grad_(False)
        print(f"FROZEN {prefix}: {len(hit)} tensors, {sum(p.numel() for p in hit)} params", flush=True)
    optim = OmegaConf.create(
        {
            "name": "adamw",
            "lr": args.lr,
            "betas": [0.9, 0.98],
            "weight_decay": 1e-3,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": args.warmup,
                "min_lr": args.lr * 0.1,
                "max_steps": args.max_steps,
            },
        }
    )

    class _Snapshot(pl.Callback):
        """Save a .nemo every --save-every batches: the last step is not necessarily the best one
        (a 1500-step Earnings run collapsed where the 500-step one had not), so a held-out dev
        set must be able to choose among them."""

        n = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: ARG002
            self.n += 1
            if args.save_every and self.n % args.save_every == 0 and self.n < args.max_steps:
                pl_module.save_to(str(out / f"step{self.n}.nemo"))
                print(f"SNAPSHOT step{self.n}.nemo", flush=True)

    class _LossPrinter(pl.Callback):
        """Print the training loss: the evidence the targets are real (not empty)."""

        n = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: ARG002
            # Own batch counter: trainer.global_step did not advance once per batch here.
            self.n += 1
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if self.n <= 6:
                print(
                    f"BATCH {self.n} gs {trainer.global_step} out {type(outputs).__name__} "
                    f"keys {list(outputs) if isinstance(outputs, dict) else None}",
                    flush=True,
                )
            if loss is not None and (self.n <= 3 or self.n % 10 == 0):
                print(f"STEP {self.n} loss {float(loss):.4f} gs {trainer.global_step}", flush=True)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision="bf16-mixed",
        max_steps=args.max_steps,
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        use_distributed_sampler=False,
        callbacks=[_LossPrinter(), _Snapshot()],
    )
    model.set_trainer(trainer)
    model.setup_optimization(optim)
    trainer.fit(model)
    model.save_to(str(out / "finetuned.nemo"))
    print("SAVED", out / "finetuned.nemo")


if __name__ == "__main__":
    main()
