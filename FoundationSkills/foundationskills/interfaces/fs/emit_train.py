
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
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foundationskills.interfaces.fs.capabilities import FSCapabilities

if TYPE_CHECKING:  # pragma: no cover - typing only, owner D1 writes the module
    from foundationskills.skills.training.knowledge import Hardware

# hparams key (any alias) -> FS CLI flag (without the leading "--"). Plans and
# recipes spell keys differently (a real plan used learning_rate and
# max_sequence_length, which an lr/seq_len-only table silently dropped, so FS
# trained on its defaults). Every alias of one flag lives in one row.
_HPARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "learning-rate": ("learning_rate", "lr"),
    "lr-scheduler-type": ("lr_scheduler_type", "lr_scheduler", "scheduler"),
    "max-sequence-length": ("max_sequence_length", "seq_len", "max_seq_len", "sequence_length"),
    "per-device-batch-size": ("per_device_batch_size", "micro_batch", "micro_batch_size"),
    "gradient-accumulation-steps": ("gradient_accumulation_steps", "grad_accum"),
    "precision": ("precision",),
    "sharding-strategy": ("sharding_strategy", "sharding"),
    "save-interval": ("save_interval",),
    "seed": ("seed",),
    "optimizer": ("optimizer",),
    "logging-steps": ("logging_steps",),
    "max-grad-norm": ("max_grad_norm", "grad_clip"),
    "dataloader-num-workers": ("dataloader_num_workers",),
    "dataloader-prefetch-factor": ("dataloader_prefetch_factor",),
    "warmup-steps": ("warmup_steps",),
    "gradient-checkpointing": ("gradient_checkpointing", "grad_ckpt"),
    "attn-implementation": ("attn_implementation",),
    "sdp-backend": ("sdp_backend",),
}
# Keys consumed elsewhere (never flags) -- listed so an unmapped key is a note,
# not a silent drop: epochs/tokens -> max_steps, lora_* -> adapter flags,
# tp/pp/ep/cp -> axes, and data-side or RL-only knobs.
_CONSUMED_KEYS = frozenset({
    "max_steps", "epochs", "tokens", "warmup_ratio", "tp", "pp", "ep", "cp",
    "lora_rank", "lora_alpha", "lora_dropout", "lora_targets", "rank",
    "replay_ratio", "global_batch_tokens", "token_budget",
})
# Values that shape the run's meaning: if present they MUST reach FS.
_LOAD_BEARING = ("learning-rate", "max-sequence-length")
# Structural flags owned by the emitter; a recipe passthrough may not override
# them (a stray --dry-run would turn a launch into a no-op that exits 0).
_META_FLAGS = frozenset({"dry-run", "output-dir", "model", "dataset", "profile-name", "profile-path",
                         "nodes", "gpus-per-node"})
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _hp(hparams: dict[str, Any], flag: str) -> Any:
    for key in _HPARAM_ALIASES.get(flag, ()):
        if hparams.get(key) is not None:
            return hparams[key]
    return None


def _alias_conflicts(hparams: dict[str, Any]) -> list[str]:
    """Two spellings of one flag with different values: which one is meant is
    unknowable, so it is refused instead of letting alias order decide."""
    out = []
    for flag, aliases in _HPARAM_ALIASES.items():
        values = {k: hparams[k] for k in aliases if hparams.get(k) is not None}
        if len({str(v) for v in values.values()}) > 1:
            out.append(f"conflicting hparams for --{flag}: {values}")
    return out


def fs_repo_root() -> str | None:
    """The FS source tree root: FS records code provenance from the launch cwd
    (capture_code_provenance(Path.cwd())), so launching anywhere else yields an
    UNATTRIBUTABLE run. Walk up from the installed package to a .git entry."""
    try:
        import foundationscale  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    here = Path(foundationscale.__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return None


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
    micro = max(1, int(_hp(hparams, "per-device-batch-size") or 1))
    seq = max(1, int(_hp(hparams, "max-sequence-length") or 2048))
    ga = max(1, int(_hp(hparams, "gradient-accumulation-steps") or 1))
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
    if not _SAFE_NAME.match(str(run_name)):
        raise ValueError(f"run_name {run_name!r} must match [A-Za-z0-9._-]+ (it names files and jobs)")
    hparams = dict(stage.get("hparams") or {})
    method = str(stage.get("method") or "full")
    stage_kind = str(stage.get("stage") or "sft")

    tp = int(hparams.get("tp", 1))
    pp = int(hparams.get("pp", 1))
    ep = int(hparams.get("ep", 1))
    cp = int(hparams.get("cp", 1))
    sharding = _hp(hparams, "sharding-strategy")
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

    if hparams.get("max_steps") is None and hparams.get("tokens") is None and hparams.get("epochs") is not None \
            and dataset.get("num_tokens"):
        hparams["tokens"] = int(float(hparams["epochs"]) * int(dataset["num_tokens"]))
        notes.append(f"tokens = epochs {hparams['epochs']} x dataset num_tokens {dataset['num_tokens']:,}")
    max_steps = _derive_max_steps(stage, hparams, dp)
    if max_steps is not None:
        pairs.append(("max-steps", max_steps))
        if "max_steps" not in hparams:
            notes.append(f"max_steps derived from tokens/(micro_batch x seq_len x grad_accum x dp) = {max_steps}")

    for flag in _HPARAM_ALIASES:
        value = _hp(hparams, flag)
        if value is not None:
            pairs.append((flag, bool(value) if flag == "gradient-checkpointing" else value))
    if _hp(hparams, "warmup-steps") is None and hparams.get("warmup_ratio") is not None and max_steps is not None:
        pairs.append(("warmup-steps", max(0, round(float(hparams["warmup_ratio"]) * max_steps))))
        notes.append("warmup-steps derived from warmup_ratio x max_steps")
    known = {a for aliases in _HPARAM_ALIASES.values() for a in aliases} | _CONSUMED_KEYS
    for key in sorted(set(hparams) - known):
        notes.append(f"hparam {key!r} maps to no FS train flag; not passed (FS has no such knob)")
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
        if flag in _META_FLAGS:
            notes.append(f"recipe fs.args --{flag} ignored: structural flags are owned by the emitter")
            continue
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
    emitted: set[str] = set()
    choices = getattr(caps, "train_flag_choices", {}) or {}
    for flag, value in ordered_pairs:
        if f"--{flag}" not in caps.train_flags:
            missing.append(f"FS flag --{flag} not supported by installed FS")
            continue
        allowed = choices.get(f"--{flag}")
        if allowed and _fmt(value) not in allowed:
            missing.append(f"FS flag --{flag} value {_fmt(value)!r} not in allowed {list(allowed)}")
            continue
        argv_flags.extend([f"--{flag}", _fmt(value)])
        emitted.add(flag)
    missing.extend(_alias_conflicts(hparams))
    for flag in _LOAD_BEARING:
        if _hp(hparams, flag) is not None and flag not in emitted:
            missing.append(f"load-bearing hparam for --{flag} could not be passed to FS")
    if "max-sequence-length" not in emitted:
        # FS's own default is 128 tokens: an unplanned sequence length is not a default, it is a defect
        missing.append("max_sequence_length not planned (FS default is 128 tokens)")

    env = dict(_BASE_ENV)
    # Hardware-profile env (e.g. the GB200 NCCL pins, measured) and the device
    # peak FS needs to report MFU instead of UNMEASURED.
    for key, value in dict(_hw(hardware, "env", {}) or {}).items():
        env[str(key)] = str(value)
    peak = _hw(hardware, "bf16_dense_tflops")
    if peak:
        env["FS_DEVICE_PEAK_TFLOPS"] = str(peak)
        env["FS_DEVICE_PEAK_SOURCE"] = str(_hw(hardware, "peak_provenance", "declared"))
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
    if world > 1 and int(nodes) == 1:
        # One node: standalone rendezvous. No MASTER_ADDR, which a direct
        # (non-shell) launch would pass to torchrun as a literal string.
        argv: list[str] = ["torchrun", "--standalone", "--nproc-per-node", str(int(gpus_per_node)), *inner]
    elif world > 1:
        # Multi-node: c10d rendezvous on the head host; the sbatch exports
        # MASTER_ADDR and must leave ${MASTER_ADDR} expandable (see sbatch.py).
        argv = [
            "torchrun", "--nnodes", str(int(nodes)), "--nproc-per-node", str(int(gpus_per_node)),
            "--rdzv-backend", "c10d", "--rdzv-endpoint", "${MASTER_ADDR}:29500", *inner,
        ]
    else:
        argv = ["python", *inner]

    dry_run_argv: list[str] | None
    if "--dry-run" in caps.train_flags:
        dry_run_argv = [*argv, "--dry-run"]
    else:
        dry_run_argv = None
        missing.append("FS flag --dry-run not supported by installed FS")

    cwd = fs_repo_root()
    if cwd is None:
        notes.append("FS repo root not found: FS would record this run as UNATTRIBUTABLE (no commit)")
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
            f"{output_dir}/final",
        ],
        "cwd": cwd,
        "executable": executable,
        "missing": None if executable else "; ".join(missing),
        "output_dir": output_dir,
        "run_name": run_name,
        "notes": notes,
    }
