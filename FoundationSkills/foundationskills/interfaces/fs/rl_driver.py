
"""The ``fskills-rl`` console script: library-RL to CLI bridge for FS.

FS ships RL only as ``foundationscale.rl.trainer.RLTrainer(RLTrainConfig)``
with no entry point; this driver is the smallest honest wrapper around it.

Status lines are one line each, ``[fskills:rl:<status>] <reason>``, and the
exit codes are the contract's own:

* 0  PASS   -- the run measured at least one step;
* 5  RED    -- an exception escaped the training loop;
* 95 UNMEASURED -- zero measured steps, or a required final checkpoint that
  could not be produced (the shipped RLTrainer holds the model in a LOCAL
  variable of ``run()``; no instance attribute exposes it, so ``save_final``
  genuinely cannot save and says so instead of pretending);
* 96 REFUSED -- unknown config keys, a missing foundationscale module, an
  unparseable config file, or an FS ``TrainerRefusal``/exit-96 passthrough.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Fallback only: the live field set is read from the installed RLTrainConfig
# (a hard-coded list refused FS's new logprob_micro_batch, #546).
_RL_FIELDS: frozenset[str] = frozenset(
    {
        "model", "dataset", "algorithm", "answer_pattern", "gold_key",
        "group_size", "learning_rate", "master_weights", "max_steps",
        "max_new_tokens", "temperature", "top_p", "top_k",
        "prompts_per_step", "seed", "device",
    }
)
_FSKILLS_FIELDS: frozenset[str] = frozenset({"output_dir", "save_final", "trainer"})
# trainer kind -> (module, trainer class, config class). "preference" is FS's
# offline PreferenceTrainer (dpo/ipo/kto/orpo/simpo/cpo; main e17c1b2).
_TRAINERS: dict[str, tuple[str, str, str]] = {
    "rl": ("foundationscale.rl.trainer", "RLTrainer", "RLTrainConfig"),
    "preference": ("foundationscale.rl.preference_trainer", "PreferenceTrainer", "PreferenceTrainConfig"),
}


def _line(status: str, reason: str) -> None:
    print(f"[fskills:rl:{status}] {reason}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fskills-rl",
        description=(
            "Run one FoundationScale RLTrainConfig JSON through "
            "foundationscale.rl.trainer.RLTrainer. Exit codes: 0 PASS, "
            "5 RED, 95 UNMEASURED, 96 REFUSED."
        ),
    )
    parser.add_argument("--config", required=True, help="path to the RL config JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the algorithm objective via RLTrainer._resolve_objective (no model load) and exit",
    )
    return parser


def _fs_version() -> str | None:
    try:
        return str(importlib.metadata.version("foundationscale"))
    except Exception:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _report_to_dict(report: Any) -> dict[str, Any]:
    """Serialise a StepReport defensively: asdict, then vars, then repr."""
    try:
        if dataclasses.is_dataclass(report) and not isinstance(report, type):
            return dataclasses.asdict(report)
    except Exception:
        pass
    try:
        return {str(k): v for k, v in vars(report).items()}
    except TypeError:
        return {"repr": repr(report)}


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return repr(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def rl_config_fields(config_cls: Any = None) -> frozenset[str]:
    """The installed RLTrainConfig's field names (measured), else the fallback."""
    import dataclasses

    if config_cls is None:
        try:
            from foundationscale.rl.trainer import RLTrainConfig as config_cls  # type: ignore
        except Exception:  # noqa: BLE001
            return _RL_FIELDS
    try:
        return frozenset(f.name for f in dataclasses.fields(config_cls))
    except TypeError:
        return _RL_FIELDS


def _is_fs_refusal(exc: BaseException) -> bool:
    """FS signals 'a required input is missing' with *Refusal exception classes."""
    return type(exc).__name__.endswith("Refusal") and type(exc).__module__.startswith("foundationscale")


def _exit_from_systemexit(exc: SystemExit) -> int:
    """Only the four doctrine codes leave this process; anything else is RED."""
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 5)
    return code if code in (0, 5, 95, 96) else 5


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # Mirror the FS CLI idiom: a declaration that did not parse is REFUSED.
        if exc.code is None or exc.code == 0:
            return 0
        _line("refuse", f"declaration rejected: the command line did not parse (argparse exit {exc.code})")
        return 96

    config_path = Path(args.config)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("config root is not a JSON object")
    except Exception as exc:
        _line("refuse", f"config {config_path} unreadable: {type(exc).__name__}: {exc}")
        return 96
    config: dict[str, Any] = raw

    kind = str(config.get("trainer", "rl"))
    if kind not in _TRAINERS:
        _line("refuse", f"unknown trainer {kind!r} (known: {', '.join(sorted(_TRAINERS))})")
        return 96
    module_name, trainer_name, config_name = _TRAINERS[kind]
    try:
        import importlib

        module = importlib.import_module(module_name)
        RLTrainer = getattr(module, trainer_name)  # noqa: N806 - the bound FS trainer class
        RLTrainConfig = getattr(module, config_name)  # noqa: N806
        from foundationscale.rl.trainer import TrainerRefusal
    except Exception as exc:
        _line("refuse", f"missing FS trainer {module_name}.{trainer_name}: {type(exc).__name__}: {exc}")
        return 96
    if kind == "preference":
        # FS's PreferenceTrainer reads ONE JSONL file; the Data Engine writes a
        # shard directory. Concatenate in shard order (a pure copy, no rewrite).
        dataset = Path(str(config.get("dataset", "")))
        if dataset.is_dir():
            shards = sorted(dataset.glob("*.jsonl"))
            if not shards:
                _line("refuse", f"dataset directory {dataset} holds no .jsonl shards")
                return 96
            merged = Path(str(config.get("output_dir", "."))) / "pairs.jsonl"
            merged.parent.mkdir(parents=True, exist_ok=True)
            with merged.open("w", encoding="utf-8") as out:
                for shard in shards:
                    out.write(shard.read_text(encoding="utf-8"))
            config = {**config, "dataset": str(merged)}

    rl_fields = rl_config_fields(RLTrainConfig)
    unknown = sorted(set(config) - rl_fields - _FSKILLS_FIELDS)
    if unknown:
        _line("refuse", f"unknown config keys: {', '.join(unknown)} (allowed: {', '.join(sorted(rl_fields | _FSKILLS_FIELDS))})")
        return 96

    fs_fields = {key: value for key, value in config.items() if key in rl_fields}
    try:
        rl_config = RLTrainConfig(**fs_fields)
        trainer = RLTrainer(rl_config)
    except TrainerRefusal as exc:
        _line("refuse", str(exc))
        return 96
    except Exception as exc:
        _line("refuse", f"config could not be bound to RLTrainConfig: {type(exc).__name__}: {exc}")
        return 96

    if args.dry_run and kind == "preference":
        # construction already validated algorithm, knobs and family (no model load)
        _line("pass", f"dry-run: preference algorithm {rl_config.algorithm!r} accepted by FS PreferenceTrainer")
        return 0
    if args.dry_run:
        try:
            trainer._resolve_objective()
        except TrainerRefusal as exc:
            _line("refuse", str(exc))
            return 96
        except SystemExit as exc:
            if exc.code == 96:
                _line("refuse", "objective resolution refused (exit 96 passthrough)")
                return 96
            return _exit_from_systemexit(exc)
        except Exception as exc:  # noqa: BLE001
            if _is_fs_refusal(exc):
                _line("refuse", f"{type(exc).__name__}: {exc}")
                return 96
            _line("red", f"unhandled {type(exc).__name__} in dry-run: {exc}")
            return 5
        _line("pass", f"dry-run: algorithm {rl_config.algorithm!r} objective resolved (no model load)")
        return 0

    output_dir = Path(str(config.get("output_dir", ".")))
    try:
        reports = trainer.run()
    except TrainerRefusal as exc:
        _line("refuse", str(exc))
        return 96
    except SystemExit as exc:
        if exc.code == 96:
            # FS's _refuse_exit_96 path raises SystemExit(96) after printing
            # its own reason; pass the verdict through under our marker.
            _line("refuse", "FS RLTrainer refused (exit 96 passthrough; see its REFUSE line above)")
            return 96
        return _exit_from_systemexit(exc)
    except Exception as exc:  # noqa: BLE001 - the contract's RED arm
        if _is_fs_refusal(exc):
            # Measured on GB200: BatchRefusal (a corpus FS cannot read) escaped
            # RLTrainer.run and was reported RED. Every FS *Refusal is a 96.
            _line("refuse", f"{type(exc).__name__}: {exc}")
            return 96
        traceback.print_exc(file=sys.stderr)
        _line("red", f"unhandled {type(exc).__name__} escaped RLTrainer.run: {exc}")
        return 5

    measured = len(reports)
    max_steps = int(getattr(rl_config, "max_steps", measured))
    unmeasured = max(0, max_steps - measured)

    _write_json(output_dir / f"{kind}_reports.json", {"reports": [_report_to_dict(r) for r in reports]})

    manifest: dict[str, Any] = {
        "config": config,
        "fs_version": _fs_version(),
        "started_at": None,  # the driver wraps run() as one call; per-step timing lives in StepReports
        "finished_at": _utc_now(),
        "steps_measured": measured,
        "steps_unmeasured": unmeasured,
    }

    # Final checkpoint. The shipped RLTrainer keeps the model in a LOCAL
    # variable of run(); if a future trainer exposes one, use it -- but a
    # required save that cannot happen is UNMEASURED, never a quiet skip.
    checkpoint_status: str
    model_obj = None
    for attr in ("model", "policy", "actor", "model_", "_model"):
        candidate = getattr(trainer, attr, None)
        if candidate is not None and hasattr(candidate, "save_pretrained"):
            model_obj = candidate
            break
    if model_obj is not None:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model_obj.save_pretrained(str(final_dir))
        tokenizer = getattr(trainer, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(final_dir))
        checkpoint_status = f"saved: {final_dir}"
    else:
        checkpoint_status = f"checkpoint: UNMEASURED ({trainer_name} exposes no model)"
    manifest["checkpoint"] = checkpoint_status

    save_final = bool(config.get("save_final", False))
    if measured == 0:
        manifest["status"] = "UNMEASURED: 0 measured steps"
        _write_json(output_dir / f"fskills_{kind}_manifest.json", manifest)
        _line("unmeasured", f"0 of {max_steps} step(s) produced a measurable update; a run that trained nothing is not a pass")
        return 95
    if save_final and model_obj is None:
        manifest["status"] = "UNMEASURED: save_final requested but no model attribute is exposed"
        _write_json(output_dir / f"fskills_{kind}_manifest.json", manifest)
        _line("unmeasured", checkpoint_status + "; save_final was required")
        return 95

    manifest["status"] = "PASS"
    _write_json(output_dir / f"fskills_{kind}_manifest.json", manifest)
    _line("pass", f"{measured} measured step(s) ({unmeasured} unmeasured); {checkpoint_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
