"""Launch-config schema and validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._artifacts import (
    _parse_iso,
)
from ._errors import (
    ConfigError,
    ToolError,
)

# ---------------------------------------------------------------------------
# Config schema + validation
# ---------------------------------------------------------------------------
# Validation is fail-closed in BOTH directions, matching the discipline of
# DeclaredCheckpoint.from_dict: missing keys are named, and unknown keys are
# named. A typo'd key a reader silently ignores is a denominator that never
# applied — the manifest's "num_expert (sic)" lesson, one tool down.


@dataclass(frozen=True)
class K:
    """Leaf spec: kind, whether the key is required, whether empty values are refused."""

    kind: str
    required: bool = True
    nonempty: bool = False
    default: Any = None


_SCHEMA: dict[str, Any] = {
    "run_name": K("str", nonempty=True),
    "world_size": K("int"),
    # -- item 1: frozen manifest ------------------------------------------------
    "frozen": {
        "model": {
            "files": K("list[str]", nonempty=True),
            "tensor_count": K("int"),
            "total_bytes": K("int"),
        },
        "corpus": {"files": K("list[dict]", nonempty=True)},
        "run_config": {"path": K("str", nonempty=True), "sha256": K("str", nonempty=True)},
    },
    # -- item 2: template audit ---------------------------------------------------
    "template": {
        "probe_command": K("list[str]", nonempty=True),
        "rows_per_file": K("int"),
        "files": K("list[str]", nonempty=True),
        "keep_cot_env": K("str", nonempty=True),
        "chat_template_path": K("str", nonempty=True),
        # md5 is RECORDED always (design item 2) but only COMPARED when pinned:
        # an unpinned md5 is a fact, not a claim, and doctrine 5 forbids
        # minting a check against a denominator nobody declared.
        "chat_template_md5": K("str", required=False, default=""),
    },
    # -- item 3: corpus wiring ----------------------------------------------------
    "corpus_wiring": {
        "env_var": K("str", nonempty=True),
        "recipe_files": K("list[str]", nonempty=True),
        "attestation_path": K("str", nonempty=True),
    },
    # -- item 5: conversion --------------------------------------------------------
    "conversion": {
        "hf_config_json": K("str", nonempty=True),
        "coverage_map_json": K("str", nonempty=True),
        "iter_metrics_jsonl": K("str", nonempty=True),
        "iter1_loss_band": K("list", nonempty=True),
        "expected_param_count": K("int"),
        "divisibility": K("list[dict]", nonempty=True),
        "tied_grounding": K("str", nonempty=True),
        "shared_kv_grounding": K("str|null"),
    },
    # -- item 6: LoRA probe ---------------------------------------------------------
    "lora": {
        "run_log": K("str", nonempty=True),
        "target_classes": K("list[str]", nonempty=True),
        "trainable_band": K("list", nonempty=True),
        "probe_metrics_jsonl": K("str", nonempty=True),
        "delta_audit_json": K("str", nonempty=True),
        "merged_dir": K("str", nonempty=True),
        "pinned_merged_total_bytes": K("int"),
        "expected_iters": K("int"),
    },
    # -- item 7: schedule ------------------------------------------------------------
    "schedule": {
        "train_iters": K("int"),
        "lr_decay_iters": K("int"),
        "save_interval": K("int"),
        "explicit_final_save": K("bool"),
        "smoke": K("bool"),
    },
    # -- item 8: evidence completeness ------------------------------------------------
    "evidence": {
        "log_glob": K("str", nonempty=True),
        "mem_regex": K("str", nonempty=True),
        "max_log_age_s": K("int"),
        # null job id is legitimate ONLY alongside a declared opt-out below —
        # the same declared-extension-point discipline as GateRegistry's
        # event_allow_empty: absence of Slurm must be stated, not silent.
        "slurm_job_id": K("str|null"),
        "allow_no_slurm": K("bool"),
        "slurm_absent_reason": K("str", required=False, default=""),
    },
    # -- item 9: dynamics -------------------------------------------------------------
    "dynamics": {
        "metrics_jsonl": K("str", nonempty=True),
        "bands": K("list[dict]", nonempty=True),
        "hard_floor": K("number", required=False, default=0.1),
    },
    # -- item 10: provenance ------------------------------------------------------------
    "provenance": {
        "checkpoint_dirs": K("list[str]", nonempty=True),
        "resume_guard_files": K("list[str]", nonempty=True),
        "min_walltime_s": K("number"),
        "job_window_utc": K("list", nonempty=True),
        "mtime_slack_s": K("int", required=False, default=5),
        "artifacts": K("list[str]", nonempty=True),
        "walltime_jsonl": K("str", nonempty=True),
    },
}


_KIND_CHECKS: dict[str, Callable[[Any], bool]] = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    # Underscored because it really is unexamined: "any" is the table's escape
    # hatch for a config slot whose kind is deliberately unconstrained, and the
    # parameter exists only to match the Callable[[Any], bool] shape every other
    # row obeys. Named `v`, it reads as a value someone forgot to check.
    "any": lambda _v: True,
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
    "str|null": lambda v: v is None or isinstance(v, str),
    "list[str]": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    "list[dict]": lambda v: isinstance(v, list) and all(isinstance(x, dict) for x in v),
}


def _walk_spec(
    spec: Mapping[str, Any], node: Any, path: str, problems: list[str], out: dict[str, Any]
) -> None:
    if not isinstance(node, dict):
        problems.append(f"{path or '<root>'}: must be an object, got {type(node).__name__}")
        return
    for key in sorted(set(node) - set(spec)):
        problems.append(
            f"{path + '.' if path else ''}{key}: unknown config key — a key this "
            f"tool cannot read is a pin it cannot enforce; fix the name or remove it"
        )
    for key, sub in spec.items():
        dotted = f"{path}.{key}" if path else key
        if key not in node:
            if isinstance(sub, K) and not sub.required:
                out[key] = sub.default
                continue
            problems.append(
                f"missing config key: {dotted!r} (fail closed: an absent pin is not a pass)"
            )
            continue
        value = node[key]
        if isinstance(sub, K):
            if not _KIND_CHECKS[sub.kind](value):
                problems.append(f"{dotted!r}: expected {sub.kind}, got {type(value).__name__}")
                continue
            if sub.nonempty and not value:
                problems.append(f"{dotted!r}: empty — a zero-length pin list is the vacuous case")
                continue
            if (
                sub.kind == "int"
                and dotted.endswith(
                    ("tensor_count", "expected_param_count", "expected_iters", "train_iters")
                )
                and value <= 0
            ):
                problems.append(f"{dotted!r}: must be a positive int, got {value!r}")
                continue
            out[key] = value
        else:
            child: dict[str, Any] = {}
            _walk_spec(sub, value, dotted, problems, child)
            out[key] = child


def _post_validate(cfg: Mapping[str, Any], problems: list[str]) -> None:
    """Composite shape rules the leaf walker cannot express. Same taxonomy, same exit."""

    fr = cfg.get("frozen", {})
    for i, entry in enumerate(fr.get("corpus", {}).get("files", []) or []):
        if not isinstance(entry, dict):
            continue  # leaf walker already named it
        base = f"frozen.corpus.files[{i}]"
        extra = sorted(set(entry) - {"path", "sha256", "lines"})
        if extra:
            problems.append(f"{base}: unknown keys {extra!r}")
        for name in ("path", "sha256", "lines"):
            if name not in entry:
                problems.append(f"missing config key: {base}.{name!r}")
        if "lines" in entry and (
            not isinstance(entry["lines"], int) or isinstance(entry["lines"], bool)
        ):
            problems.append(f"{base}.lines: expected int")

    conv = cfg.get("conversion", {})
    band = conv.get("iter1_loss_band")
    if isinstance(band, list) and band:
        if len(band) != 2 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in band
        ):
            problems.append("conversion.iter1_loss_band: expected [lo, hi] numbers")
        elif band[0] > band[1]:
            problems.append(
                f"conversion.iter1_loss_band: lo {band[0]} > hi {band[1]} — "
                f"an inverted band passes everything"
            )
    for i, assertion in enumerate(conv.get("divisibility", []) or []):
        if not isinstance(assertion, dict):
            continue
        base = f"conversion.divisibility[{i}]"
        if not isinstance(assertion.get("field"), str) or not assertion.get("field"):
            problems.append(f"{base}.field: required non-empty dotted key into the HF config")
        forms = [k for k in ("divisible_by", "equals") if k in assertion]
        if len(forms) != 1:
            problems.append(f"{base}: exactly one of 'divisible_by'/'equals' is required")
        elif forms[0] == "divisible_by" and (
            not isinstance(assertion["divisible_by"], int)
            or isinstance(assertion["divisible_by"], bool)
            or assertion["divisible_by"] <= 0
        ):
            problems.append(f"{base}.divisible_by: must be a positive int")

    lora = cfg.get("lora", {})
    tb = lora.get("trainable_band")
    if isinstance(tb, list) and tb:
        if len(tb) != 2 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in tb
        ):
            problems.append("lora.trainable_band: expected [lo, hi] numbers")
        elif tb[0] > tb[1]:
            problems.append("lora.trainable_band: lo > hi")

    dyn = cfg.get("dynamics", {})
    for i, b in enumerate(dyn.get("bands", []) or []):
        if not isinstance(b, dict):
            continue
        base = f"dynamics.bands[{i}]"
        for name in ("iteration", "lo", "hi"):
            if name not in b:
                problems.append(f"missing config key: {base}.{name!r}")
        if "iteration" in b and (
            not isinstance(b["iteration"], int)
            or isinstance(b["iteration"], bool)
            or b["iteration"] < 1
        ):
            problems.append(f"{base}.iteration: must be an int >= 1")
        if (
            "lo" in b
            and "hi" in b
            and isinstance(b["lo"], (int, float))
            and isinstance(b["hi"], (int, float))
            and b["lo"] > b["hi"]
        ):
            problems.append(f"{base}: lo {b['lo']} > hi {b['hi']}")

    ev = cfg.get("evidence", {})
    if (
        ev.get("slurm_job_id") is None
        and ev.get("allow_no_slurm") is True
        and not str(ev.get("slurm_absent_reason", "")).strip()
    ):
        problems.append(
            "evidence.slurm_absent_reason: required when allow_no_slurm is true — "
            "the declared opt-out must say WHY, or it is a silent hole"
        )

    prov = cfg.get("provenance", {})
    window = prov.get("job_window_utc")
    if isinstance(window, list) and window:
        parsed = [_parse_iso(w) if isinstance(w, str) else None for w in window]
        if len(window) != 2 or any(p is None for p in parsed):
            problems.append(
                "provenance.job_window_utc: expected [start_iso, end_iso] ISO-8601 strings"
            )
        elif parsed[0] >= parsed[1]:
            problems.append(
                "provenance.job_window_utc: start must precede end — "
                "an empty window contains every artifact or none"
            )


def _load_config(path: str) -> tuple[dict[str, Any], str]:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        # Unreadable config: the TOOL could not run. Exit 2, not 1 — a block
        # verdict over checks that never executed would be a claim without evidence.
        raise ToolError(f"config unreadable: {p}: {exc}") from exc
    cfg_sha = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolError(f"config at {p} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(["<root>: config must be a JSON object"])
    problems: list[str] = []
    out: dict[str, Any] = {}
    _walk_spec(_SCHEMA, data, "", problems, out)
    _post_validate(out, problems)
    if not out.get("frozen", {}).get("model") or not out.get("frozen", {}).get("corpus"):
        # Structural restatement of requirement E's "a config that pins nothing":
        # with neither a model pin nor a corpus pin, nothing downstream has a
        # denominator and every check would abstain against fiction.
        problems.append(
            "config pins nothing: frozen.model and frozen.corpus are both absent — "
            "BLOCK, by doctrine 1"
        )
    if problems:
        raise ConfigError(problems)
    return out, cfg_sha
