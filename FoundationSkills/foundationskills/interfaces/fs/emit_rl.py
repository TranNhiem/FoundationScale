
"""Emit an ``fs_launch_spec`` payload for an RL stage (entry ``fskills-rl``).

FS exposes RL only as a library (``foundationscale.rl.trainer.RLTrainer``)
with no CLI; the bridge is the ``fskills-rl`` console script backed by
``rl_driver.main``. The config it reads uses EXACTLY the ``RLTrainConfig``
field names, plus the fskills additions ``output_dir`` and ``save_final``.

Executability is measured, never assumed: ``caps.check("rl", algorithm=...)``
knows which of the 18 registered algorithms RLTrainer actually runs (on the
measured commit: dr_grpo, gspo, dapo only). The trainer is single-device, so
no sharding/parallel axes apply and every payload carries that warning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.interfaces.fs.emit_train import fs_repo_root

# hparams key -> RLTrainConfig field. Only keys present in the stage hparams
# are emitted, so RLTrainConfig's own dataclass defaults govern the rest.
_HPARAM_TO_RL: dict[str, str] = {
    "lr": "learning_rate",
    "learning_rate": "learning_rate",
    "group_size": "group_size",
    "max_steps": "max_steps",
    "max_new_tokens": "max_new_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "prompts_per_step": "prompts_per_step",
    "seed": "seed",
    "master_weights": "master_weights",
    "device": "device",
    "answer_pattern": "answer_pattern",
    "gold_key": "gold_key",
}

# consumed by the planner/estimator, not RLTrainConfig knobs
_RL_IGNORED = frozenset({"kl_weight", "epochs", "tokens", "lora_rank", "lora_alpha", "lora_dropout",
                         "lora_targets", "rl_generation_factor"})

_SINGLE_DEVICE_WARNING = (
    "FS RLTrainer is single-device (one GPU): multi-GPU RL is refused by the installed FS"
)


def emit_rl(
    stage: dict[str, Any],
    *,
    dataset: dict[str, Any],
    model: str,
    output_dir: str,
    caps: FSCapabilities,
    run_name: str,
) -> dict[str, Any]:
    """Build one ``fs_launch_spec`` payload for an RL stage."""
    missing: list[str] = []
    notes: list[str] = []
    hparams = dict(stage.get("hparams") or {})
    algorithm = stage.get("algorithm")
    if algorithm is not None:
        algorithm = str(algorithm)

    shards = dataset.get("shards") or []
    if shards:
        data_path = str(Path(shards[0]["path"]).parent)
    else:
        data_path = None
        missing.append("dataset payload has no shards to pass as the RL corpus")

    gold_key = hparams.get("gold_key")
    if gold_key is None:
        gold_key = (dataset.get("fs_columns") or {}).get("gold_key")

    rl_config: dict[str, Any] = {
        "model": model,
        "dataset": data_path or "<no shards>",
    }
    if algorithm is not None:
        rl_config["algorithm"] = algorithm
    if gold_key is not None:
        rl_config["gold_key"] = str(gold_key)
    # Any hparam that IS an installed RLTrainConfig field passes through (measured;
    # "learning_rate" used to be dropped because only the alias "lr" was mapped),
    # plus the aliases. Everything else is reported, never silently lost.
    from foundationskills.interfaces.fs.rl_driver import rl_config_fields

    kind = "preference" if str(stage.get("stage")) == "preference" else "rl"
    if kind == "preference":
        try:
            from foundationscale.rl.preference_trainer import PreferenceTrainConfig  # type: ignore

            fields = rl_config_fields(PreferenceTrainConfig)
        except Exception:  # noqa: BLE001 - older FS without the preference plane
            fields = frozenset()
            missing.append("missing: FS PreferenceTrainer (foundationscale.rl.preference_trainer)")
        rl_config["trainer"] = "preference"
        rl_config.pop("gold_key", None)  # preference records carry prompt/chosen/rejected, not a gold
    else:
        fields = rl_config_fields()
    for key, value in hparams.items():
        if value is None or key == "gold_key":
            continue
        field = _HPARAM_TO_RL.get(key, key)
        if field in fields and field not in ("model", "dataset", "algorithm"):
            rl_config[field] = value
        elif key not in _RL_IGNORED:
            notes.append(f"hparam {key!r} is not an RLTrainConfig field of the installed FS; not passed")
    if "answer_pattern" in rl_config and str(dataset.get("format")) == "rl":
        # Data Engine rl datasets carry single-letter MCQ gold (the only reward FS
        # verifies); a recipe's free-form extraction pattern would not match it.
        rl_config.pop("answer_pattern")
    rl_config["output_dir"] = output_dir
    rl_config["save_final"] = bool(stage.get("save_final", True))

    # Data Engine rl datasets are MCQ-letter by construction (format drops the rest).
    answer_kind = "mcq_letter" if str(dataset.get("format")) == "rl" else None
    check_stage = "preference" if rl_config.get("trainer") == "preference" else "rl"
    check_reason = caps.check(check_stage, algorithm=algorithm, answer_kind=answer_kind)
    if check_reason is not None:
        missing.append(check_reason)
    # FS RL persists no checkpoint today: the stage cannot hand off, but an
    # operator may still run it to MEASURE (rewards, advantages, throughput).
    measurement_only_ok = caps.check(check_stage, algorithm=algorithm, answer_kind=answer_kind,
                                     require_checkpoint=False) is None and bool(data_path)
    if measurement_only_ok and missing:
        # The only gap is the hand-off: a measurement run cannot save, so it must not
        # REQUIRE a save (every such run would read UNMEASURED). The driver still
        # records "checkpoint: UNMEASURED" in its manifest.
        rl_config["save_final"] = False
        notes.append("measurement-only: save_final=false (FS trainer persists no policy)")

    # `python -m` works whether or not the fskills-rl console script is installed
    # (a real launch failed with "No such file or directory: 'fskills-rl'").
    argv = ["python", "-m", "foundationskills.interfaces.fs.rl_driver", "--config", f"{output_dir}/rl_config.json"]
    expected_outputs = [
        f"{output_dir}/rl_reports.json",
        f"{output_dir}/fskills_rl_manifest.json",
    ]
    if rl_config["save_final"]:
        expected_outputs.append(f"{output_dir}/final")

    executable = not missing
    return {
        "stage_name": str(stage.get("name") or run_name),
        "entry": "fskills-rl",
        "argv": argv,
        "env": {},
        "sbatch": None,
        "dry_run_argv": [*argv, "--dry-run"],
        "expected_outputs": expected_outputs,
        "executable": executable,
        "missing": None if executable else "; ".join(missing),
        "output_dir": output_dir,
        "run_name": run_name,
        "rl_config": rl_config,
        "notes": notes,
        "measurement_only_ok": measurement_only_ok,
        "cwd": fs_repo_root(),
        "warnings": [_SINGLE_DEVICE_WARNING],
    }
