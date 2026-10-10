"""NeMo AED (Canary) fine-tune worker adapter: FoundationScale feeds the data, NeMo trains.

PHASE 1.2a move target for ``validation_campaigns/speech_canary/nemo_finetune.py``. Behaviour is
preserved verbatim -- same NeMo calls, same config overrides, same printed
COVERAGE/TRAIN_DS/STEP/SNAPSHOT/SAVED lines.

1. Convert a FoundationScale audio JSONL (audio, answer, duration) to a NeMo AED manifest under
   the same strict rules as the HF loader: 16 kHz mono only, 1-``max_duration`` s, non-empty
   text; every refused row is COUNTED and recorded (never silently dropped). See
   :func:`convert` and :mod:`foundationscale.upstream.nemo.census`.
2. Fine-tune through NeMo's own API (setup_training_data / setup_optimization / Lightning
   Trainer) and save a ``.nemo``. NeMo owns the training loop here; FoundationScale does not
   wrap it.

The PURE parts -- :func:`train_ds_overrides`, :func:`optimizer_config`, :func:`parse_args` -- are
deterministic transforms callable from plain CI with no NeMo in sight:

* :func:`train_ds_overrides` takes a PLAIN dict (already ``OmegaConf.to_container(...,
  resolve=True)``-ed) and returns the next dict. Two upstream workarounds live here and are
  recorded in :mod:`foundationscale.upstream.ledger` (``nemo-canary-train-cfg-strip`` and
  ``nemo-canary-text-field``).
* :func:`optimizer_config` mirrors the NeMo optimizer + scheduler dict the original built
  inline.
* :func:`parse_args` accepts EITHER the existing ``nemo_finetune.py`` flags (identical argparse
  names) OR ``--config path.json`` (loaded with ``load_run_config``). Both must yield the same
  ``SpeechRunConfig``:
  ``parse_args(to_argv(cfg)) == parse_args(["--config", json_of(cfg)]) == cfg``.

Usage::

    python -m foundationscale.upstream.nemo.finetune --model nvidia/canary-1b-flash \\
        --train JSONL --out-dir DIR \\
        [--max-steps 500 --batch-size 8 --lr 1e-5 --warmup 50 --pnc no] \\
        [--save-every N]      (also save DIR/step{N}.nemo, for checkpoint selection) \\
        [--freeze PREFIX ...] (e.g. ``transf_decoder``; adjudicate with ``--frozen``)

Or ``--config path.json`` -- the same run as a typed ``SpeechRunConfig`` file.

Exit code follows the run contract: 0 OK, 96 REFUSE (counted and printed as ``REFUSE (96): ...``
to stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from foundationscale.upstream.contracts import (
    SpeechRunConfig,
    load_run_config,
    read_speech_manifest,
    to_nemo_aed_row,
)
from foundationscale.upstream.nemo.census import census

__all__ = [
    "convert",
    "main",
    "optimizer_config",
    "parse_args",
    "train_ds_overrides",
]


def convert(
    train_jsonl: Path, out_manifest: Path, pnc: str, max_duration: float = 25.0
) -> dict[str, Any]:
    """Convert a FoundationScale audio JSONL to a NeMo AED manifest on disk.

    ONE REFUSAL, at the read stage: ``read_speech_manifest``'s report showing any per-row
    problem (``invalid_json``, ``missing:answer``, ...) or any ``duplicate_ids`` count is raised
    as ``ValueError``. ``main`` turns that into a counted ``REFUSE (96)`` message + exit code
    ``96``. ``duplicate_audio_paths`` is NOT a read-stage refusal -- it stays a counted
    ``census`` reason (``duplicate_audio_path``) exactly as the original ``convert`` counted it,
    so existing runs' refusal buckets line up one-for-one.

    Returns the coverage dict with the EXACT keys the original returned: ``rows_expected`` (rows
    read), ``rows_checked`` (rows the trainer will load), ``rows_refused``, ``refused`` (reason ->
    count). The NeMo AED manifest is written to ``out_manifest``, one JSON line per kept row via
    :func:`to_nemo_aed_row` -- same shape the original produced.
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


def train_ds_overrides(
    released: dict[str, Any],
    *,
    manifest_path: str,
    batch_size: int,
    max_duration: float,
    seed: int | None,
    prompt_format: Any,
) -> dict[str, Any]:
    """Apply the released-``train_ds`` overrides exactly as the campaign script did. Pure.

    Takes a PLAIN dict (already ``OmegaConf.to_container(..., resolve=True)``-ed) and returns a
    NEW dict with the same edits the original applied, in the same order. The caller wraps the
    result in ``OmegaConf.create(...)`` -- the OmegaConf node is not this function's business.

    Two upstream workarounds live HERE, both recorded in :mod:`foundationscale.upstream.ledger`:

    * ``nemo-canary-train-cfg-strip`` -- the released ``train_ds`` carries tarred/bucketing keys
      for NVIDIA's own data; they are stripped so a plain manifest loads. The STRIP LIST is
      byte-for-byte the original: adding or removing a key here is a behavioural change.
    * ``nemo-canary-text-field`` -- the released ``train_ds`` names the transcript field
      ``"answer"`` while our NeMo manifests write ``"text"``; left inherited, the targets would
      be silently empty. Declared explicitly here.

    ``prompt_format`` is precomputed by the caller (the released value, overridden by the model's
    top-level ``prompt_format`` when one is set) to mirror ``model.cfg.get("prompt_format",
    train_cfg.get("prompt_format"))`` in the original.
    """
    cfg = dict(released)
    cfg["manifest_filepath"] = str(manifest_path)
    cfg["use_lhotse"] = True
    cfg["batch_size"] = batch_size
    cfg["shuffle"] = True
    if seed is not None:
        cfg["seed"] = seed
        # lhotse's shard_seed defaults to "trng" (true randomness per worker), so `seed` alone
        # left the data order unreproducible: two runs with one seed differed at step 1 (GB200,
        # 2026-10-10). A seeded run pins it too; an unseeded run keeps NeMo's default.
        cfg["shard_seed"] = seed
    cfg["num_workers"] = 4
    cfg["max_duration"] = max_duration
    cfg["prompt_format"] = prompt_format

    # The checkpoint's own train_ds names the transcript field "answer"; this manifest writes
    # it as "text". Left inherited, NeMo reads an absent field: targets would be empty and
    # nothing would say so. Declared explicitly instead. The campaign script wrote this as a
    # one-liner against the OmegaConf node -- the source text is preserved verbatim below so the
    # ledger's ``nemo-canary-text-field`` anchor keeps checking out across the PHASE 1.2a move:
    #   train_cfg.text_field = "text"
    cfg["text_field"] = "text"

    # The released Canary train_ds config carries tarred/bucketing keys for NVIDIA's own data;
    # they are stripped so a plain manifest loads (ledger entry ``nemo-canary-train-cfg-strip``).
    # The strip list is byte-for-byte the original -- gather-what-you-strip is a real contract.
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
        cfg.pop(k, None)
    return cfg


def optimizer_config(lr: float, warmup: int, max_steps: int) -> dict[str, Any]:
    """The NeMo optimizer + scheduler dict for a Canary fine-tune. Pure.

    Same shape the original built inline: AdamW with ``betas=[0.9, 0.98]`` and
    ``weight_decay=1e-3``, cosine-annealing LR schedule with linear warmup and a
    ``min_lr = 0.1 * lr`` floor over ``max_steps``. Callers wrap the returned dict in
    ``OmegaConf.create(...)`` and hand it to ``model.setup_optimization(...)`` -- that is the
    caller's business, not ours.
    """
    return {
        "name": "adamw",
        "lr": lr,
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": warmup,
            "min_lr": lr * 0.1,
            "max_steps": max_steps,
        },
    }


def parse_args(argv: list[str] | None = None) -> SpeechRunConfig:
    """Parse the CLI as a typed ``SpeechRunConfig``. Pure (argparse-only; no NeMo, no IO except
    the ``--config`` file read).

    Two equivalent surfaces, one typed result:

    * the EXACT ``nemo_finetune.py`` argparse names (``--model``, ``--train``, ``--out-dir``,
      ``--max-steps``, ``--batch-size``, ``--lr``, ``--warmup``, ``--pnc``, ``--save-every``,
      ``--max-duration``, ``--seed``, ``--freeze`` append);
    * OR ``--config path.json`` -- the same run as a typed ``SpeechRunConfig`` file, loaded with
      :func:`load_run_config`.

    ``parse_args(to_argv(cfg))`` and ``parse_args(["--config", json_of(cfg)])`` must both return
    ``cfg`` -- tested. These two surfaces are how a worker's subprocess invocation gets rebuilt
    from a recorded run, and a symlink-sized drift between them would silently produce a
    different run the second time.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a JSON run config (SpeechRunConfig). Either --config or the flags below, "
            "never both -- the config file is the typed source of truth and flag overrides are "
            "not merged in."
        ),
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--pnc", default="no")
    ap.add_argument("--save-every", type=int, default=0)
    # Longest clip accepted, in seconds. 25 s is the clip recipe; long-form training raises it
    # to 40 s (speech_longform), and a longer clip is refused and counted, never truncated.
    ap.add_argument("--max-duration", type=float, default=25.0)
    # Unset keeps the earlier runs' behaviour; set, it seeds Lightning and the lhotse shuffle so a
    # result can be replicated across seeds (one seed per arm cannot separate an effect from noise).
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--freeze", action="append", default=[])
    args = ap.parse_args(argv)
    if args.config is not None:
        return load_run_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for dest in ("model", "train", "out_dir"):
        if getattr(args, dest) is None:
            ap.error(f"--{dest.replace('_', '-')} is required unless --config is given")
    return SpeechRunConfig(
        model=args.model,
        train_manifest=args.train,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=args.warmup,
        pnc=args.pnc,
        max_duration=args.max_duration,
        seed=args.seed,
        save_every=args.save_every,
        freeze=tuple(args.freeze),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the NeMo AED fine-tune end to end. Return the exit code: 0 OK, 96 REFUSE.

    Refusal handling follows the run contract: on a bad manifest (manifest row problem or
    duplicate id), a ``--pnc`` outside ``[no, yes]``, or a ``--freeze`` prefix matching no
    parameter, we print a ``REFUSE (96): ...`` message to stderr and return 96 -- a COUNTED
    refusal, never a silent skip and never a half-trained run.
    """
    try:
        cfg = parse_args(argv)
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Record the upstream this run actually ran on (rule 3: profiles), next to its outputs.
        from foundationscale.upstream.profiles import write_profile_record

        write_profile_record(out, "nemo-26.08")
        coverage = convert(
            Path(cfg.train_manifest), out / "train_manifest.json", cfg.pnc, cfg.max_duration
        )
    except ValueError as e:
        print(f"REFUSE (96): {e}", file=sys.stderr)
        return 96
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print("COVERAGE", json.dumps(coverage))

    import lightning.pytorch as pl  # type: ignore[import-not-found]

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)
    from nemo.collections.asr.models import ASRModel  # type: ignore[import-not-found]
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    model = ASRModel.from_pretrained(cfg.model, map_location="cpu")
    released = (
        dict(OmegaConf.to_container(model.cfg.train_ds, resolve=True))
        if model.cfg.get("train_ds") is not None
        else {}
    )
    train_cfg = OmegaConf.create(
        train_ds_overrides(
            released,
            manifest_path=str(out / "train_manifest.json"),
            batch_size=cfg.batch_size,
            max_duration=cfg.max_duration,
            seed=cfg.seed,
            prompt_format=model.cfg.get("prompt_format", released.get("prompt_format")),
        )
    )
    print("TRAIN_DS", OmegaConf.to_yaml(train_cfg).replace("\n", " | ")[:600])
    model.setup_training_data(train_cfg)
    for prefix in cfg.freeze:
        hit = [p for n, p in model.named_parameters() if n == prefix or n.startswith(prefix + ".")]
        if not hit:  # a typo would freeze nothing and train everything, silently
            print(f"REFUSE (96): --freeze {prefix!r} matches no parameter", file=sys.stderr)
            return 96
        for p in hit:
            p.requires_grad_(False)
        print(
            f"FROZEN {prefix}: {len(hit)} tensors, {sum(p.numel() for p in hit)} params",
            flush=True,
        )
    optim = OmegaConf.create(optimizer_config(cfg.lr, cfg.warmup, cfg.max_steps))

    class _Snapshot(pl.Callback):
        """Save a .nemo every --save-every batches: the last step is not necessarily the best one
        (a 1500-step Earnings run collapsed where the 500-step one had not), so a held-out dev
        set must be able to choose among them."""

        n = 0

        # Lightning calls hooks positionally, so unused arguments carry a leading underscore.
        def on_train_batch_end(
            self, _trainer: Any, pl_module: Any, _outputs: Any, _batch: Any, _batch_idx: int
        ) -> None:
            self.n += 1
            if cfg.save_every and self.n % cfg.save_every == 0 and self.n < cfg.max_steps:
                pl_module.save_to(str(out / f"step{self.n}.nemo"))
                print(f"SNAPSHOT step{self.n}.nemo", flush=True)

    class _LossPrinter(pl.Callback):
        """Print the training loss: the evidence the targets are real (not empty)."""

        n = 0

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
        precision="bf16-mixed",
        max_steps=cfg.max_steps,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
