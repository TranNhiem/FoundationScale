"""NeMo speechlm2 SALM (Canary-Qwen) fine-tune worker adapter: FoundationScale feeds the data,
NeMo trains.

PHASE 1.2b move target for ``validation_campaigns/speech_canary/salm_finetune.py``. Behaviour is
preserved verbatim -- same NeMo calls, same ``_StrictSALMDataset`` workaround, same cuDNN SDPA
switch, same data cfg, same printed COVERAGE/DATA_CFG/TRAINABLE/BATCH/STEP/SAVED lines, same save
(``DIR/finetuned`` via ``save_pretrained``) -- with ONE addition: ``--seed``.

1. Convert a FoundationScale audio JSONL (audio, answer, duration) to a NeMo manifest via
   :func:`convert` under the same strict rules as the HF loader: 16 kHz mono only, 1-25 s,
   non-empty text; every refused row is COUNTED and recorded (never silently dropped). See
   :func:`foundationscale.upstream.nemo.census.census`.
2. Fine-tune the released SALM through NeMo's own speechlm2 API (``SALMDataset`` + ``DataModule``
   + the model's own ``configure_optimizers``) and save via ``HFHubMixin``. NeMo owns the training
   loop; FoundationScale does not wrap it. The released freeze scope and LoRA stay exactly as
   shipped -- nothing about them is re-specified here.

The PURE parts -- :func:`data_config`, :func:`optimizer_config`, :func:`lr_scheduler_config`,
:func:`parse_args` -- are deterministic transforms callable from plain CI with no NeMo in sight:
they are exactly the dicts the campaign script built inline, so a run's loader/optimiser config can
be rebuilt and compared without a NeMo container. Three upstream workarounds live HERE and are
recorded in :mod:`foundationscale.upstream.ledger` (``nemo-salm-strict-loading``,
``nemo-salm-cudnn-sdpa-off``, ``nemo-datamodule-null-validation`` -- all anchored in this file).

Usage::

    python -m foundationscale.upstream.nemo.salm_finetune --model nvidia/canary-qwen-2.5b \\
        --train JSONL --out-dir DIR \\
        [--max-steps 500 --batch-size 8 --lr 1e-4 --warmup 50] [--seed N]

``--seed N`` seeds Lightning (``pl.seed_everything(N, workers=True)``) and pins the lhotse shard
shuffle to N as well (see :func:`data_config`); unset, the loader keeps the earlier runs' defaults.

Exit codes are the campaign script's, unchanged: 0 OK, and a loaded checkpoint with no trainable
parameter refuses with ``SystemExit("no trainable parameters ...")`` (exit 1) rather than running
an empty loop. ``main`` returns the run's status; the counted refusals here are the manifest ones,
recorded row-by-row in ``coverage.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from foundationscale.upstream.contracts import read_speech_manifest, to_nemo_aed_row
from foundationscale.upstream.nemo.census import census

__all__ = [
    "convert",
    "data_config",
    "lr_scheduler_config",
    "main",
    "optimizer_config",
    "parse_args",
]


def convert(
    train_jsonl: Path, out_manifest: Path, pnc: str, max_duration: float = 25.0
) -> dict[str, Any]:
    """Convert a FoundationScale audio JSONL to a NeMo manifest on disk. Same conversion as
    :func:`foundationscale.upstream.nemo.finetune.convert` (``read_speech_manifest`` ->
    :func:`foundationscale.upstream.nemo.census.census` -> ``to_nemo_aed_row``), kept in this
    worker so the SALM lane produces its own manifest under the identical rules.

    ONE REFUSAL, at the read stage: ``read_speech_manifest``'s report showing any per-row problem
    (``invalid_json``, ``missing:answer``, ...) or any ``duplicate_ids`` count is raised as
    ``ValueError`` (the campaign script let that propagate). ``duplicate_audio_paths`` is NOT a
    read-stage refusal -- it stays a counted ``census`` reason (``duplicate_audio_path``), so
    existing runs' refusal buckets line up one-for-one.

    Returns the coverage dict with the EXACT keys the campaign run wrote to ``coverage.json``:
    ``rows_expected``, ``rows_checked``, ``rows_refused``, ``refused`` (reason -> count). The NeMo
    manifest rows are the AED row shape at ``pnc="no"`` -- ``lhotse_as_conversation`` reads
    ``audio_filepath``/``duration``/``text`` only, so the Canary AED ``pnc`` field is out of place
    here.
    """
    rows, report = read_speech_manifest(train_jsonl)
    problems = dict(report.problems)
    if report.duplicate_ids:
        problems["duplicate_id"] = report.duplicate_ids
    if problems:
        raise ValueError(
            f"bad speech manifest {train_jsonl}: rows_read={report.rows_read} "
            f"rows_valid={report.rows_valid} problems "
            + ", ".join(f"{k}={v}" for k, v in sorted(problems.items()))
        )
    kept_rows, coverage = census(rows, max_duration=max_duration)
    with Path(out_manifest).open("w") as f:
        for r in kept_rows:
            f.write(json.dumps(to_nemo_aed_row(r, pnc=pnc)) + "\n")
    return coverage


def data_config(
    manifest_path: str,
    audio_locator_tag: str,
    prompt_format: Any,
    batch_size: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """The ``DataModule`` config the campaign script built inline. Pure.

    REPRODUCES TODAY'S DICT: with ``seed`` unset the values are the campaign's byte for byte --
    in particular ``"seed": 42`` and ``"shard_seed": "randomized"``, which is what the earlier runs
    used. Setting ``seed`` pins both: lhotse's shard shuffle is otherwise not reproducible,
    measured on the AED lane 2026-10-10 (two runs with one seed differed at step 1), so
    ``"seed"`` alone would leave the data order unreproducible.

    ``prompt_format`` and ``audio_locator_tag`` come from the loaded model (``model.cfg.
    prompt_format`` / ``model.audio_locator_tag``); the ``tags`` instruction is the same sentence
    the eval (:mod:`foundationscale.upstream.nemo.salm_decode`) and ``SALM.generate`` use.

    Two further upstream workarounds are visible in this dict and anchored here for the ledger:
    ``fault_tolerant_audio_loading`` is off (NeMo's default silently skips unreadable audio;
    :func:`convert` has already refused and COUNTED every bad row) and the missing
    ``validation_ds`` key.
    """
    train_ds: dict[str, Any] = {
        "sample_rate": 16000,
        "prompt_format": prompt_format,
        "input_cfg": [
            {
                "type": "lhotse_as_conversation",
                "manifest_filepath": str(manifest_path),
                "audio_locator_tag": audio_locator_tag,
                # The same instruction the eval (salm_decode.py) and SALM.generate use.
                "tags": {"context": "Transcribe the following:"},
            }
        ],
        "token_equivalent_duration": 0.08,
        "batch_size": batch_size,
        "shuffle": True,
        "use_bucketing": False,
        "shard_seed": "randomized",
        "num_workers": 4,
        "seed": 42,
        # NeMo's default (True) skips audio it cannot read; FoundationScale never drops a row
        # silently, and convert() has already refused and counted every bad one.
        "fault_tolerant_audio_loading": False,
    }
    if seed is not None:
        # lhotse's shard shuffle is otherwise not reproducible (measured on the AED lane,
        # 2026-10-10), so a seeded run pins both -- an unseeded run keeps the values above.
        train_ds["seed"] = seed
        train_ds["shard_seed"] = seed
    return {
        "train_ds": train_ds,
        # No validation_ds key at all: DataModule touches any present key, even a null one.
    }


def optimizer_config(lr: float) -> dict[str, Any]:
    """The ``configure_optimizers`` AdamW dict for a SALM fine-tune. Pure.

    Same shape the campaign script assigned into ``model.cfg.optimizer``: AdamW with
    ``betas=[0.9, 0.98]``, ``weight_decay=1e-3`` and ``foreach=True``. The caller assigns it under
    ``open_dict(model.cfg)`` -- that is the caller's business, not ours.
    """
    return {
        "_target_": "torch.optim.AdamW",
        "lr": lr,
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "foreach": True,
    }


def lr_scheduler_config(lr: float, warmup: int, max_steps: int) -> dict[str, Any]:
    """The ``configure_optimizers`` CosineAnnealing dict for a SALM fine-tune. Pure.

    Same shape the campaign script assigned into ``model.cfg.lr_scheduler``: NeMo's
    ``nemo.core.optim.lr_scheduler.CosineAnnealing`` with ``warmup_steps`` warmup and a
    ``min_lr = 0.1 * lr`` floor over ``max_steps``.
    """
    return {
        "_target_": "nemo.core.optim.lr_scheduler.CosineAnnealing",
        "warmup_steps": warmup,
        "min_lr": lr * 0.1,
        "max_steps": max_steps,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI. Pure (argparse-only; no NeMo, no IO).

    Arg names identical to ``salm_finetune.py``'s -- ``--model``, ``--train``, ``--out-dir``,
    ``--max-steps`` (500), ``--batch-size`` (8), ``--lr`` (1e-4), ``--warmup`` (50) -- plus the ONE
    PHASE 1.2b addition: ``--seed``. Unset keeps the earlier runs' behaviour (the ``seed=42`` /
    ``shard_seed="randomized"`` loader defaults above); set, it seeds Lightning and the lhotse
    shuffle so a result can be replicated across seeds (one seed per arm cannot separate an effect
    from noise).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    # Unset keeps the earlier runs' behaviour; set, it seeds Lightning and the lhotse shuffle so a
    # result can be replicated across seeds (one seed per arm cannot separate an effect from noise).
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the SALM fine-tune end to end. Return 0 on success; refusals are the campaign script's
    (see the module docstring) and are preserved exactly, ``SystemExit`` included.

    Lightning calls callback hooks positionally, so unused hook arguments carry a leading
    underscore throughout the callbacks below.
    """
    args = parse_args(argv)
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

    import lightning.pytorch as pl  # type: ignore[import-not-found]
    import torch
    from nemo.collections.speechlm2 import (  # type: ignore[import-not-found]
        SALM,
        DataModule,
        SALMDataset,
    )
    from omegaconf import OmegaConf, open_dict  # type: ignore[import-not-found]

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

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
        data_config(
            str(out / "train_manifest.json"),
            model.audio_locator_tag,
            model.cfg.prompt_format,
            args.batch_size,
            args.seed,
        )
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
        model.cfg.optimizer = optimizer_config(args.lr)
        model.cfg.lr_scheduler = lr_scheduler_config(args.lr, args.warmup, args.max_steps)

    class _LossPrinter(pl.Callback):
        """Print the training loss: the evidence the targets are real (not empty)."""

        n = 0

        # Lightning calls hooks positionally, so unused arguments carry a leading underscore.
        def on_train_batch_end(
            self, trainer: Any, _pl_module: Any, outputs: Any, _batch: Any, _batch_idx: int
        ) -> None:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
