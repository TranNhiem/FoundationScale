"""NeMo speechlm2 SALM fine-tune (FoundationScale feeds the data and adjudicates the result).

Same contract as the Canary NeMo-lane scripts, one lane down (SALM = Canary encoder, trained
in full, + a frozen Qwen LLM with LoRA):
1. Convert a FoundationScale audio JSONL (audio, answer, duration) to a NeMo manifest via
   nemo_finetune.convert under the same strict rules: 16 kHz mono only, 1-25 s, non-empty
   text; every refused row is COUNTED and recorded (never silently dropped).
2. Fine-tune the released SALM through NeMo's own speechlm2 API (SALMDataset + DataModule +
   the model's own configure_optimizers) and save via HFHubMixin. NeMo owns the training loop;
   FoundationScale does not wrap it. The released freeze scope and LoRA stay exactly as shipped
   -- nothing about them is re-specified here.

Usage: salm_finetune.py --model nvidia/canary-qwen-2.5b --train JSONL --out-dir DIR
                        [--max-steps 500 --batch-size 8 --lr 1e-4 --warmup 50]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nemo_finetune import convert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # "pnc" is a Canary AED field; lhotse_as_conversation reads audio_filepath/duration/text only.
    coverage = convert(Path(args.train), out / "train_manifest.json", "no")
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print("COVERAGE", json.dumps(coverage))

    # No init_process_group here on purpose (single device), but SALM/DataModule still look for a
    # world with one process: give them the single-rank env defaults instead of an absent one.
    for k, v in (
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("WORLD_SIZE", "1"),
    ):
        os.environ.setdefault(k, v)

    import lightning.pytorch as pl
    from nemo.collections.speechlm2 import DataModule, SALM, SALMDataset
    from omegaconf import OmegaConf, open_dict

    import torch

    # cuDNN's fused attention fails in backward on this GB200 stack (cuDNN Frontend reshape
    # error); flash / memory-efficient SDPA remain enabled and compute the same attention.
    torch.backends.cuda.enable_cudnn_sdp(False)
    # from_pretrained keeps the released LoRA + freeze_params scope; do not re-specify either.
    model = SALM.from_pretrained(args.model)
    total = sum(p.numel() for p in model.parameters())
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for _, p in trainable)
    prefixes = {n.split(".")[0] for n, _ in trainable}
    print(
        "TRAINABLE",
        json.dumps(
            {
                "trainable_params": trainable_params,
                "total_params": total,
                "trainable_tensors": len(trainable),
                "total_tensors": sum(1 for _ in model.parameters()),
                "top_level_prefixes": sorted(prefixes),
            }
        ),
    )
    if not trainable:  # a frozen-everything checkpoint would train nothing and say nothing
        raise SystemExit("no trainable parameters after loading -- refusing to run an empty loop")

    data_cfg = OmegaConf.create(
        {
            "train_ds": {
                "sample_rate": 16000,
                "prompt_format": model.cfg.prompt_format,
                "input_cfg": [
                    {
                        "type": "lhotse_as_conversation",
                        "manifest_filepath": str(out / "train_manifest.json"),
                        "audio_locator_tag": model.audio_locator_tag,
                        # The same instruction the eval (eval_wer_salm.py) and SALM.generate use.
                        "tags": {"context": "Transcribe the following:"},
                    }
                ],
                "token_equivalent_duration": 0.08,
                "batch_size": args.batch_size,
                "shuffle": True,
                "use_bucketing": False,
                "shard_seed": "randomized",
                "num_workers": 4,
                "seed": 42,
                # NeMo's default (True) skips audio it cannot read; FoundationScale never drops a row
                # silently, and convert() has already refused and counted every bad one.
                "fault_tolerant_audio_loading": False,
            },
            # No validation_ds key at all: DataModule touches any present key, even a null one.
        }
    )
    print("DATA_CFG", OmegaConf.to_yaml(data_cfg).replace("\n", " | ")[:600])

    # Plain SALMDataset is what salm_train._create_salm_dataset returns when the optional
    # multispeaker/packing/batch_tokens knobs are all unset (they are, here).
    class _StrictSALMDataset(SALMDataset):
        """Strict policy (no FallbackDataset, drops raise) on the loader shape the collate expects.

        NeMo 3.1's strict view sets AudioSamples(fault_tolerant=False), which returns 2 values,
        but collate_conversation_audio_fault_tolerant always unpacks 3, so strict loading crashes.
        Keeping the 3-value loader while strict_audio_loading=True makes any dropped or reordered
        conversation/audio item a RuntimeError instead of a silent skip.
        """

        def with_fault_tolerant_audio_loading(self, enabled: bool) -> SALMDataset:
            view = super().with_fault_tolerant_audio_loading(enabled)
            view.load_audio.fault_tolerant = True
            return view

    dataset = _StrictSALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(data_cfg, tokenizer=model.tokenizer, dataset=dataset)

    # configure_optimizers() builds AdamW + CosineAnnealing from these two _target_ dicts.
    with open_dict(model.cfg):
        model.cfg.optimizer = {
            "_target_": "torch.optim.AdamW",
            "lr": args.lr,
            "betas": [0.9, 0.98],
            "weight_decay": 1e-3,
            "foreach": True,
        }
        model.cfg.lr_scheduler = {
            "_target_": "nemo.core.optim.lr_scheduler.CosineAnnealing",
            "warmup_steps": args.warmup,
            "min_lr": args.lr * 0.1,
            "max_steps": args.max_steps,
        }

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
        precision="bf16-true",
        max_steps=args.max_steps,
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        use_distributed_sampler=False,
        callbacks=[_LossPrinter()],
    )
    trainer.fit(model, datamodule)
    model.save_pretrained(str(out / "finetuned"))
    print("SAVED", out / "finetuned")


if __name__ == "__main__":
    main()
