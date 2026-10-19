"""Experiment index (A4.6): one read-only view over every experiment record under a set of roots.

The reserved "Experiment management" row of docs/ARCHITECTURE.md (every artifact plus FS run manifests -> an
experiment index; reads ``journal.jsonl`` and the manifests), built into auto_research rather than a new skill.
Three record kinds are indexed, each copied verbatim from what the record itself measured:

- ``skill_step``: one row per orchestrator ``journal.jsonl`` line (core/orchestrator.py);
- ``fs_run``: one row per FS ``run_manifest.json``. The outcome is the ``exit_contract`` name of the manifest's
  ``config["extra.exit"]``; a missing exit or contract leaves it None with the reason, never a guess;
- ``campaign``: one row per campaign of an auto_research ledger (a directory holding ``chain.jsonl``), counted
  through the same ``Ledger`` methods the skill uses, and only over a chain that verifies.

A found record that cannot be read is listed under ``unreadable`` (never dropped); the caller decides the verdict.
Finding nothing at all is refused: a vacuous index is never a pass. No clock is read, so re-indexing an unchanged
tree returns an equal dict.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from .ledger import Ledger

INDEX_SCHEMA_VERSION = 1
JOURNAL = "journal.jsonl"
MANIFEST = "run_manifest.json"
CHAIN = "chain.jsonl"
# FS telemetry an index row carries when the manifest marks it measured (others are listed as unmeasured).
TELEMETRY_KEYS = ("train_loss", "train_runtime_s", "perf_tokens_total", "perf_mfu", "peak_memory_allocated_bytes",
                  "perf_steps_observed")
_STEP_KEYS = ("step_index", "skill", "status", "exit_code", "artifact_ids", "refusal", "timestamp")


class IndexRefused(ValueError):
    """The roots cannot carry an index (missing, not a directory, or nothing to index)."""


def _scan(roots: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    journals: dict[Path, None] = {}
    manifests: dict[Path, None] = {}
    ledgers: dict[Path, None] = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):  # a symlink loop is never entered
            dirnames.sort()
            here = Path(dirpath)
            for name in sorted(filenames):
                if name == JOURNAL:
                    journals[(here / name).resolve()] = None
                elif name == MANIFEST:
                    manifests[(here / name).resolve()] = None
                elif name == CHAIN:
                    ledgers[here.resolve()] = None
    return list(journals), list(manifests), list(ledgers)


def _value(config: dict[str, Any], key: str) -> Any:
    entry = config.get(key)
    return entry.get("value") if isinstance(entry, dict) else None


def _journal_rows(path: Path, rows: list[dict], unreadable: list[dict]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        unreadable.append({"source": str(path), "line": None, "reason": f"unreadable: {exc}"})
        return
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            record, error = None, f"not JSON ({exc.msg})"
        else:
            error = None if isinstance(record, dict) else "not a JSON object"
        if error:
            unreadable.append({"source": str(path), "line": number, "reason": error})
            continue
        rows.append({"kind": "skill_step", "source": str(path), "line": number,
                     **{key: record.get(key) for key in _STEP_KEYS}})


def _manifest_row(path: Path, rows: list[dict], unreadable: list[dict]) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        unreadable.append({"source": str(path), "line": None, "reason": f"not a readable JSON manifest ({exc})"})
        return
    if not isinstance(manifest, dict):
        unreadable.append({"source": str(path), "line": None, "reason": "manifest is not a JSON object"})
        return
    if manifest.get("schema_version") != 1:
        unreadable.append({"source": str(path), "line": None,
                           "reason": f"unsupported manifest schema_version {manifest.get('schema_version')!r}"})
        return
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    code = manifest.get("code") if isinstance(manifest.get("code"), dict) else {}
    raw_exit = _value(config, "extra.exit")
    try:
        exit_code = int(raw_exit) if raw_exit is not None else None
    except (TypeError, ValueError):
        exit_code = None
    try:
        contract = json.loads(_value(config, "exit_contract") or "null")
    except (TypeError, json.JSONDecodeError):
        contract = None
    outcome, reason = None, None
    if exit_code is None:
        reason = "the manifest records no exit code (config extra.exit)"
    elif not isinstance(contract, dict):
        reason = "the manifest records no exit_contract to name the exit code"
    else:
        names = [name for name, code_ in contract.items() if code_ == exit_code]
        outcome = names[0] if names else None
        reason = None if names else f"exit code {exit_code} is not in the manifest's exit_contract"
    telemetry = manifest.get("telemetry") if isinstance(manifest.get("telemetry"), dict) else {}
    measured = {k: telemetry[k].get("value") for k in TELEMETRY_KEYS
                if isinstance(telemetry.get(k), dict) and telemetry[k].get("source") == "measured"}
    environment = manifest.get("environment") if isinstance(manifest.get("environment"), dict) else {}
    env_values = environment.get("values") if isinstance(environment.get("values"), dict) else {}
    paths = manifest.get("artifact_paths") if isinstance(manifest.get("artifact_paths"), dict) else {}
    rows.append({
        "kind": "fs_run", "source": str(path),
        **{key: manifest.get(key) for key in ("run_id", "attempt", "job_id", "created_at", "fingerprint")},
        "code_status": code.get("status"), "code_commit": code.get("commit"),
        "output_dir": paths.get("output_dir"),
        "model": _value(config, "model"), "dataset": _value(config, "dataset"),
        "exit_code": exit_code, "outcome": outcome, "outcome_reason": reason,
        "video_frames": env_values.get("FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES"),
        "telemetry": measured,
        "unmeasured_telemetry": sorted(k for k in TELEMETRY_KEYS if k in telemetry and k not in measured),
    })


def _ledger_rows(root: Path, rows: list[dict], unreadable: list[dict]) -> None:
    ledger = Ledger(root)
    try:
        problems = ledger.verify()
        entries = ledger.entries()
    except Exception as exc:  # noqa: BLE001 - any failure to read the chain is an unreadable ledger, never a skip
        problems, entries = [f"{type(exc).__name__}: {exc}"], []
    if problems:
        unreadable.append({"source": str(root / CHAIN), "line": None,
                           "reason": "ledger chain does not verify: " + "; ".join(map(str, problems[:3]))})
        return
    for campaign in sorted({str(e.get("campaign")) for e in entries}):
        mine = [e for e in entries if str(e.get("campaign")) == campaign]
        rows.append({
            "kind": "campaign", "source": str(root), "campaign": campaign,
            "proposals": len(ledger.proposals(campaign)), "launches": len(ledger.launches(campaign)),
            "results": len(ledger.results(campaign)),
            "closed": any(e.get("op") == "campaign_closed" for e in mine),
            "last_ts": max((str(e.get("ts")) for e in mine), default=None),
        })


def build_index(scan_roots: Sequence[str | Path]) -> dict[str, Any]:
    """The experiment index over ``scan_roots`` (see the module docstring); raises IndexRefused."""
    if not scan_roots:
        raise IndexRefused("scan_roots is empty")
    roots: list[Path] = []
    for root in scan_roots:
        path = Path(root)
        if not path.is_dir():
            raise IndexRefused(f"scan root {root} is not a directory")
        roots.append(path.resolve())
    journals, manifests, ledgers = _scan(roots)
    if not (journals or manifests or ledgers):
        raise IndexRefused(f"nothing to index under {', '.join(map(str, roots))}: no {JOURNAL}, {MANIFEST} or "
                           f"ledger {CHAIN}")
    rows: list[dict] = []
    unreadable: list[dict] = []
    for path in journals:
        _journal_rows(path, rows, unreadable)
    for path in manifests:
        _manifest_row(path, rows, unreadable)
    for path in ledgers:
        _ledger_rows(path, rows, unreadable)
    rows.sort(key=lambda r: (r["kind"], r["source"], r.get("line") or 0, str(r.get("campaign") or "")))
    unreadable.sort(key=lambda u: (u["source"], u.get("line") or 0))
    counts = {kind: sum(r["kind"] == kind for r in rows) for kind in ("skill_step", "fs_run", "campaign")}
    return {"schema_version": INDEX_SCHEMA_VERSION, "roots": [str(r) for r in roots], "rows": rows,
            "unreadable": unreadable, "counts": {**counts, "unreadable": len(unreadable)}}
