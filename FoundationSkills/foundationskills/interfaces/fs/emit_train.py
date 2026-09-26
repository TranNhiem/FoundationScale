
"""Emit an ``fs_launch_spec`` payload for one plan stage through ``foundationscale-train``.

The output is a plain dict that validates against
``schemas/artifacts/fs_launch_spec.json``. Nothing here *launches* anything;
the spec is reviewed, hashed, confirmed, and launched by
``foundationskills.interfaces.fs.launch``.

Contract notes (chosen where the interface spec is silent):

* ``stage`` is a training_plan stage dict. Hyperparameters are read from
  ``stage["hparams"]``; recipe CLI passthrough lives in ``stage["fs"]["args"]``
  (FS flag names without ``--``; ``_`` and ``-`` are normalised to ``-``).
* Parallelism axes (``tp``/``pp``/``ep``/``cp``) come from the hparams, default
  1. Executability is delegated to ``caps.check`` -- pp/ep>1 is a measured
  REFUSE on the installed FS and lands in ``missing`` here, never in argv.
* Every emitted flag is checked against ``caps.train_flags``. A flag the
  installed FS does not advertise is DROPPED from argv and named in
  ``missing`` ("FS flag --x not supported by installed FS"); the spec becomes
  non-executable. A flag that cannot be declared cannot be silently honoured.
* ``max_steps`` is derived from ``tokens / (micro_batch x seq_len x
  grad_accum x dp)`` when absent; tokens are looked up in hparams, then
  ``stage["data"]["tokens"]``, then ``stage["estimate"]["tokens"]``. If no
  token count is available the flag is omitted (FS applies its own default,
  which its manifest records -- we do not invent a number).
* Boolean values are emitted as lowercase ``true``/``false`` strings, matching
  FS's tri-state flags (``--gradient-checkpointing`` & friends); FS's only
  store_true flag is ``--dry-run``, which is never emitted in the main argv.
* The sbatch text is null here; TrainingEmitSkill layers it on via
  ``render_sbatch`` when a scheduler is in play.
* The env block mirrors the estate launcher's NCCL/allocation pins (measured
  GB200 settings, inert on a single-GPU/local run) and nothing else.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foundationskills.interfaces.fs.capabilities import FSCapabilities

if TYPE_CHECKING:  # pragma: no cover - typing only, owner D1 writes the module
    from foundationskills.skills.training.knowledge import Hardware

# hparams key -> FS CLI flag (without the leading "--").
_HPARAM_FLAGS: dict[str, str] = {
    "lr": "learning-rate",
    "lr_scheduler": "lr-scheduler-type",
    "seq_len": "max-sequence-length",
    "micro_batch": "per-device-batch-size",
    "grad_accum": "gradient-accumulation-steps",
    "precision": "precision",
    "sharding": "sharding-strategy",
    "save_interval": "save-interval",
    "seed": "seed",
    "optimizer": "optimizer",
    "logging_steps": "logging-steps",
    "max_grad_norm": "max-grad-norm",
    "dataloader_num_workers": "dataloader-num-workers",
    "dataloader_prefetch_factor": "dataloader-prefetch-factor",
}

# Essential env mirrored from the estate launcher
# (launchers/launch_g4e4b_lora_1tray.sh): NCCL/allocation pins measured on the
# GB200 trays. Kept deliberately minimal; every value is a string (schema).
_BASE_ENV: dict[str, str] = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NCCL_NVLS_ENABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "WANDB_MODE": "disabled",
}


def _hw(hardware: Any, key: str, default: Any = None) -> Any:
    """Read a hardware attribute from either a Hardware dataclass or a dict."""
    if isinstance(hardware, dict):
        return hardware.get(key, default)
    return getattr(hardware, key, default)


def _fmt(value: Any) -> str:
    """Render one flag value; booleans become FS's lowercase tri-state form."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _derive_max_steps(stage: dict[str, Any], hparams: dict[str, Any], dp: int) -> int | None:
    if hparams.get("max_steps") is not None:
        return int(hparams["max_steps"])
    tokens = hparams.get("tokens")
    if tokens is None:
        tokens = (stage.get("data") or {}).get("tokens")
    if tokens is None:
        tokens = (stage.get("estimate") or {}).get("tokens")
    if tokens is None:
        return None
    micro = max(1, int(hparams.get("micro_batch", 1)))
    seq = max(1, int(hparams.get("seq_len", 2048)))
    ga = max(1, int(hparams.get("grad_accum", 1)))
    return max(1, math.ceil(int(tokens) / (micro * seq * ga * max(1, dp))))


def _dataset_path(dataset: dict[str, Any], notes: list[str]) -> str | None:
    """The shard directory (FS accepts a directory of jsonl) or a single shard."""
    shards = dataset.get("shards") or []
    if not shards:
        return None
    parents = {str(Path(s["path"]).parent) for s in shards if s.get("path")}
    if len(parents) == 1:
        return parents.pop()
    if not parents:
        return None
    first = str(Path(shards[0]["path"]).parent)
    notes.append(f"dataset shards span {len(parents)} directories; passing the first shard instead of a directory")
    return str(Path(shards[0]["path"]))


def emit_train(
    stage: dict[str, Any],
    *,
    dataset: dict[str, Any],
    model: str,
    output_dir: str,
    hardware: "Hardware | dict",
    nodes: int,
    gpus_per_node: int,
    caps: FSCapabilities,
    run_name: str,
) -> dict[str, Any]:
    """Build one ``fs_launch_spec`` payload for ``foundationscale-train``.

    ``executable``/``missing`` come from two measured sources: FS capability
    gaps (``caps.check`` over stage/backend/axes) and per-flag support
    (``caps.train_flags``). Nothing is assumed about the installed FS.
    """
    notes: list[str] = []
    missing: list[str] = []
    hparams = dict(stage.get("hparams") or {})
    method = str(stage.get("method") or "full")
    stage_kind = str(stage.get("stage") or "sft")

    tp = int(hparams.get("tp", 1))
    pp = int(hparams.get("pp", 1))
    ep = int(hparams.get("ep", 1))
    cp = int(hparams.get("cp", 1))
    sharding = hparams.get("sharding")
    backend = str(sharding or "ddp")

    world = max(1, int(nodes) * int(gpus_per_node))
    dp = max(1, world // max(1, tp * cp))

    # Structural flags: required by the CLI contract itself.
    scheduler = _hw(hardware, "scheduler")
    profile = "slurm-generic" if (int(nodes) > 1 or (scheduler not in (None, "", "local"))) else "local-single-node"
    data_path = _dataset_path(dataset, notes)
    if data_path is None:
        missing.append("dataset payload has no shards to pass to --dataset")

    pairs: list[tuple[str, Any]] = [
        ("model", model),
        ("dataset", data_path or "<no shards>"),
        ("output-dir", output_dir),
        ("nodes", int(nodes)),
        ("gpus-per-node", int(gpus_per_node)),
        ("profile-name", profile),
        ("dp", dp),
    ]
    # Measured fact 1: --objective accepts only sft (causal LM over `text`);
    # CPT/pretraining run through this same entry. Declared explicitly so the
    # run manifest records it.
    if stage_kind in {"pretrain", "cpt", "sft"}:
        pairs.append(("objective", "sft"))

    max_steps = _derive_max_steps(stage, hparams, dp)
    if max_steps is not None:
        pairs.append(("max-steps", max_steps))
        if "max_steps" not in hparams:
            notes.append(f"max_steps derived from tokens/(micro_batch x seq_len x grad_accum x dp) = {max_steps}")

    for key, flag in _HPARAM_FLAGS.items():
        if key in hparams and hparams[key] is not None:
            pairs.append((flag, hparams[key]))

    if "warmup_steps" in hparams and hparams["warmup_steps"] is not None:
        pairs.append(("warmup-steps", hparams["warmup_steps"]))
    elif "warmup_ratio" in hparams and max_steps is not None:
        pairs.append(("warmup-steps", max(0, round(float(hparams["warmup_ratio"]) * max_steps))))
        notes.append("warmup-steps derived from warmup_ratio x max_steps")

    if "grad_ckpt" in hparams and hparams["grad_ckpt"] is not None:
        pairs.append(("gradient-checkpointing", bool(hparams["grad_ckpt"])))
    if tp > 1:
        pairs.append(("tp", tp))
    if cp > 1:
        pairs.append(("cp", cp))

    if method == "lora":
        rank = int(hparams.get("lora_rank", 16))
        pairs.append(("adapter", "lora"))
        pairs.append(("adapter-rank", rank))
        pairs.append(("adapter-alpha", float(hparams.get("lora_alpha", 2 * rank))))
        pairs.append(("adapter-dropout", float(hparams.get("lora_dropout", 0.0))))
        # Auto LoRA targeting only exists for FS-registered families
        # (measured: gemma4, qwen3.5). Any other family needs explicit targets.
        targets = list(stage.get("lora_targets") or hparams.get("lora_targets") or [])
        family = stage.get("family")
        registered = family is not None and str(family) in (caps.families or {})
        if targets and not registered:
            for target in targets:
                pairs.append(("adapter-target", str(target)))
        elif targets and registered:
            notes.append(f"family {family} is FS-registered; lora_targets left to FS auto-targeting")
        elif not registered and family:
            missing.append(
                f"family {family} is not FS-registered and no lora_targets were supplied for --adapter-target"
            )
    elif method == "qlora":
        missing.append("method qlora is not supported by installed FS (no qlora adapter mode)")

    # Recipe passthrough overrides hparam mappings when a key collides.
    fs_args = dict((stage.get("fs") or {}).get("args") or {})
    by_flag: dict[str, Any] = {}
    order: list[str] = []
    repeatables: list[tuple[str, Any]] = []
    for flag, value in pairs:
        by_flag[flag] = value
        order.append(flag)
    for raw_key, value in fs_args.items():
        flag = str(raw_key).lstrip("-").replace("_", "-")
        if flag == "adapter-target":  # repeatable
            repeatables.append((flag, value))
            continue
        if flag not in by_flag:
            order.append(flag)
        by_flag[flag] = value
    ordered_pairs = [(flag, by_flag[flag]) for flag in order] + repeatables

    # Every emitted flag must be advertised by the installed FS; otherwise the
    # declaration is dropped and named -- never silently honoured.
    argv_flags: list[str] = []
    for flag, value in ordered_pairs:
        if f"--{flag}" not in caps.train_flags:
            missing.append(f"FS flag --{flag} not supported by installed FS")
            continue
        argv_flags.extend([f"--{flag}", _fmt(value)])

    env = dict(_BASE_ENV)
    if str(dataset.get("format") or "") == "mm_sft":
        image_column = (dataset.get("fs_columns") or {}).get("image_column") or "image"
        env["FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN"] = str(image_column)
        notes.append(f"mm_sft: FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN={image_column}")

    check_reason = caps.check(
        stage_kind,
        algorithm=stage.get("algorithm"),
        backend=backend,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
    )
    if check_reason is not None:
        missing.append(check_reason)

    inner = ["-m", "foundationscale.train.cli", *argv_flags]
    if world > 1:
        # Estate launch shape: torchrun per node set, c10d rendezvous on the
        # head host (MASTER_ADDR is exported by the sbatch preamble).
        argv: list[str] = [
            "torchrun", "--nnodes", str(int(nodes)), "--nproc-per-node", str(int(gpus_per_node)),
            "--rdzv-backend", "c10d", "--rdzv-endpoint", "$MASTER_ADDR:29500", *inner,
        ]
    else:
        argv = ["python", *inner]

    dry_run_argv: list[str] | None
    if "--dry-run" in caps.train_flags:
        dry_run_argv = [*argv, "--dry-run"]
    else:
        dry_run_argv = None
        missing.append("FS flag --dry-run not supported by installed FS")

    executable = not missing
    return {
        "stage_name": str(stage.get("name") or run_name),
        "entry": "foundationscale-train",
        "argv": argv,
        "env": env,
        "sbatch": None,
        "dry_run_argv": dry_run_argv,
        "expected_outputs": [
            f"{output_dir}/run_manifest.json",
            f"{output_dir}/model.safetensors",
        ],
        "executable": executable,
        "missing": None if executable else "; ".join(missing),
        "output_dir": output_dir,
        "run_name": run_name,
        "notes": notes,
    }
