"""Append-only cost/trace ledger (C4, atlas C10 observability-cost).

One JSON line per skill call, launch or eval call, written best effort next to the run: a failed write never
changes a verdict or blocks a launch. Counters that were not measured are ``null`` with a ``null_reason``; nothing is
estimated silently (no price table exists yet, so cost is always ``null``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import statistics
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from foundationskills.core.provenance import sha256_json

SCHEMA = "fskills.ledger/1"
FILENAME = "ledger.jsonl"


def _null(value: Any, reason: str) -> dict[str, Any]:
    return {"value": value, "null_reason": None if value is not None else reason}


def record(kind: str, skill_name: str, skill_version: str, *, verdict: str | None, exit_code: int | None,
           wall_clock_ms: float | None, request: Any = None, rules_fired: Iterable[str] = (),
           harness_hash: str | None = None, tokens_in: int | None = None, tokens_out: int | None = None,
           gpu_seconds: float | None = None, error_class: str | None = None, run_id: str | None = None,
           extra: dict[str, Any] | None = None, null_reasons: dict[str, str] | None = None) -> dict[str, Any]:
    """One ledger row (schema fskills.ledger/1). ``null_reasons`` overrides the default reason per counter."""
    why = {"tokens_in": "no_llm_call", "tokens_out": "no_llm_call", "gpu_seconds": "not_launched", **(null_reasons or {})}
    return {
        "schema": SCHEMA, "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": run_id or uuid.uuid4().hex, "kind": kind, "skill_name": skill_name, "skill_version": skill_version,
        "input_hash": sha256_json(request)[:16] if request is not None else None,
        "harness_hash": harness_hash, "rules_fired": sorted(set(rules_fired)), "verdict": verdict,
        "exit_code": exit_code, "wall_clock_ms": wall_clock_ms,
        "tokens_in": _null(tokens_in, why["tokens_in"]), "tokens_out": _null(tokens_out, why["tokens_out"]),
        "gpu_seconds": _null(gpu_seconds, why["gpu_seconds"]), "cost_estimate_usd": _null(None, "no_price_table"),
        "error_class": error_class, **(extra or {}),
    }


def append(directory: str | Path | None, row: dict[str, Any]) -> bool:
    """Append ``row`` to ``<directory>/ledger.jsonl``; False (never an exception) when it cannot be written."""
    if not directory:
        return False
    try:
        path = Path(directory) / FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(row, sort_keys=True, default=str) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)  # one write per row: concurrent appenders never interleave a line
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def shared_dir() -> Path | None:
    """FSKILLS_LEDGER_DIR: an optional ledger directory outside any skill workdir."""
    value = os.environ.get("FSKILLS_LEDGER_DIR")
    return Path(value) if value else None


def append_skill_row(workdir: str | Path | None, refused: bool, row: dict[str, Any]) -> None:
    """A REFUSED call writes nothing into the skill's workdir (doctrine); it is traced only to the shared dir."""
    if not refused:
        append(workdir, row)
    if shared_dir() is not None:
        append(shared_dir(), row)


def read(path: str | Path) -> list[dict[str, Any]]:
    """All parseable rows; a torn or foreign line is skipped, never fatal."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("schema") == SCHEMA:
            rows.append(row)
    return rows


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per skill: calls, verdict histogram, p50/max wall clock, measured token and GPU sums, null counts by reason."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("skill_name"))].append(row)
    out = {}
    for name, items in sorted(groups.items()):
        walls = [r["wall_clock_ms"] for r in items if isinstance(r.get("wall_clock_ms"), (int, float))]
        nulls: dict[str, int] = defaultdict(int)
        sums = {}
        for key in ("tokens_in", "tokens_out", "gpu_seconds"):
            values = [r[key]["value"] for r in items if isinstance(r.get(key), dict) and r[key]["value"] is not None]
            sums[key] = sum(values) if values else None
            for r in items:
                reason = (r.get(key) or {}).get("null_reason")
                if reason:
                    nulls[f"{key}:{reason}"] += 1
        verdicts: dict[str, int] = defaultdict(int)
        for r in items:
            verdicts[str(r.get("verdict"))] += 1
        out[name] = {"calls": len(items), "verdicts": dict(verdicts),
                     "wall_clock_ms_p50": statistics.median(walls) if walls else None,
                     "wall_clock_ms_max": max(walls) if walls else None, **sums, "nulls": dict(nulls)}
    return out
