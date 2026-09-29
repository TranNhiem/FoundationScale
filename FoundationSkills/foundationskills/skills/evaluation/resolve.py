"""Resolve what to evaluate: the checkpoint dir, whether it is a PEFT adapter, and its base.

Order for the base (the baseline model): ``base`` argument, then the FS run
manifest's ``config.model``, then ``adapter_config.json``'s
``base_model_name_or_path`` -- the last only when it is an existing local
directory. There is no model registry here, so an unresolvable base is a
refusal, never a fallback. Nothing is guessed: a run manifest names
``<output_dir>/final``; a missing ``final/`` is refused, not replaced by the
newest ``checkpoint-N``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ResolveError(ValueError):
    def __init__(self, rule_id: str, message: str) -> None:
        super().__init__(message)
        self.rule_id = rule_id


@dataclass(frozen=True)
class Resolved:
    checkpoint: Path
    base: Path
    adapter: bool
    base_source: str  # "argument" | "run_manifest" | "adapter_config"


def _manifest_value(manifest: dict[str, Any], key: str) -> str | None:
    entry = (manifest.get("config") or {}).get(key)
    value = entry.get("value") if isinstance(entry, dict) else entry
    if value in (None, "", "None"):
        return None
    return str(value)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResolveError("EV-IN-001", f"missing input: run manifest {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResolveError("EV-IN-001", f"precondition failed: run manifest {path} is not readable JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ResolveError("EV-IN-001", f"precondition failed: run manifest {path} is not a JSON object")
    return data


def _adapter_base(checkpoint: Path) -> str | None:
    try:
        cfg = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = cfg.get("base_model_name_or_path") if isinstance(cfg, dict) else None
    return str(value) if value else None


def resolve(checkpoint: str | None = None, base: str | None = None, run_manifest: str | None = None) -> Resolved:
    manifest_base = None
    if run_manifest:
        manifest = _load_manifest(Path(run_manifest))
        manifest_base = _manifest_value(manifest, "model")
        if checkpoint is None:
            output_dir = _manifest_value(manifest, "output_dir")
            if output_dir is None:
                raise ResolveError("EV-IN-001", f"missing input: config.output_dir in run manifest {run_manifest}")
            checkpoint = str(Path(output_dir) / "final")
    if checkpoint is None:
        raise ResolveError("EV-IN-001", "missing input: checkpoint (pass --checkpoint or --run-manifest)")
    ckpt = Path(checkpoint)
    if not ckpt.is_dir():
        raise ResolveError("EV-IN-001", f"missing input: checkpoint directory {ckpt}")
    adapter = (ckpt / "adapter_config.json").is_file()

    candidates = [(base, "argument"), (manifest_base, "run_manifest")]
    if adapter:
        candidates.append((_adapter_base(ckpt), "adapter_config"))
    for value, source in candidates:
        if not value:
            continue
        path = Path(value)
        if path.is_dir():
            return Resolved(ckpt, path, adapter, source)
        if source != "adapter_config":
            # A named base that does not exist is an error, not a cue to try the next source.
            raise ResolveError("EV-IN-002", f"missing input: base model directory {path} (from {source})")
    kind = "adapter" if adapter else "checkpoint"
    hint = f" (base_model_name_or_path={_adapter_base(ckpt)!r} is not a local path)" if adapter else ""
    raise ResolveError("EV-IN-002", f"missing input: base model for {kind} {ckpt}{hint}")


def ships_chat_template(model_dir: Path) -> bool:
    if (model_dir / "chat_template.jinja").is_file() or (model_dir / "chat_template.json").is_file():
        return True
    try:
        cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(cfg, dict) and bool(cfg.get("chat_template"))
