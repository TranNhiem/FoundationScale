from __future__ import annotations

"""Experience bank: append-only, file-first memory of MEASURED skill outcomes.

Reads are ADVISORY ONLY: nothing returned here may ever feed a verdict — warn only.
A REFUSED outcome never writes into a skill workdir; REFUSED rows go only to the
shared experience dir, so skill workdirs stay untouched. All I/O is best-effort."""

import getpass
import json
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from foundationskills.core.provenance import sha256_json

SCHEMA = "fskills.experience/1"
FILENAME = "experience.jsonl"
UNMEASURED_TTL_DAYS = 30
MIN_SUPPORT = 3
VOLATILE_KEYS = frozenset(
    {
        "workdir", "out", "results", "spec", "artifact_root", "output_dir",
        "created", "timestamp", "run_id", "port", "gpu", "python",
    }
)
_MEASURED = frozenset({"PASS", "RED"})


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k == "action" or k not in VOLATILE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, str) and value[:1] in ("/", "~"):
        return "<path>"
    return value


def task_signature(skill: str, request: dict) -> str:
    return sha256_json({"skill": skill, "request": _stable(request)})[:16]


def _iso(ts: datetime | None = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return ts.isoformat().replace("+00:00", "Z")


def _parse_iso(text: Any) -> datetime | None:
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def shared_dir() -> Path | None:
    if os.environ.get("FSKILLS_EXPERIENCE_DIR"):
        return Path(os.environ["FSKILLS_EXPERIENCE_DIR"])
    return None


def bank_path(workdir=None) -> Path | None:
    shared = shared_dir()
    if shared is not None:
        return shared / FILENAME
    return Path(workdir) / ".fskills" / FILENAME if workdir else None


def outcome_row(skill, version, request, verdict: str, rule_ids, *, now=None) -> dict:
    ts = now if isinstance(now, datetime) else datetime.now(timezone.utc)
    rules = [rule_ids] if isinstance(rule_ids, str) else list(rule_ids or ())
    action = request.get("action") if isinstance(request, dict) else None
    return {
        "schema": SCHEMA,
        "ts_utc": _iso(ts),
        "run_id": uuid.uuid4().hex,
        "skill": skill,
        "version": version,
        "signature": task_signature(skill, request),
        "action": str(action) if action is not None else None,
        "verdict": verdict,
        "rule_ids": sorted({str(r) for r in rules}),
        "measured": verdict in _MEASURED,
        "expires_utc": _iso(ts + timedelta(days=UNMEASURED_TTL_DAYS)) if verdict == "UNMEASURED" else None,
        "host": socket.gethostname(),
        "user": getpass.getuser(),
    }


def record(workdir, refused: bool, row: dict) -> bool:
    try:
        shared = shared_dir()
        # REFUSED never writes into a skill workdir: shared dir only.
        if refused:
            target = shared / FILENAME if shared else None
        else:
            target = bank_path(workdir)
        if target is None:
            return False
        os.makedirs(str(target.parent), exist_ok=True)
        fd = os.open(str(target), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return True
    except Exception:
        return False


def read(path, *, now=None) -> list[dict]:
    rows: list[dict] = []
    if path is None or not Path(path).is_file():
        return rows
    cutoff = now or datetime.now(timezone.utc)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict) or row.get("schema") != SCHEMA:
            continue
        expires = _parse_iso(row.get("expires_utc"))
        if expires is not None and expires <= cutoff:  # expiry is inclusive
            continue
        rows.append(row)
    return rows


def lessons(rows, *, skill=None, signature=None, min_support=MIN_SUPPORT) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for row in rows or []:
        if not row.get("measured") or row.get("verdict") not in _MEASURED:
            continue
        if skill is not None and row.get("skill") != skill:
            continue
        if signature is not None and row.get("signature") != signature:
            continue
        key = (row.get("skill"), row.get("signature"), row.get("verdict"), tuple(row.get("rule_ids") or ()))
        group = groups.setdefault(key, {"runs": set(), "last_seen": ""})
        group["runs"].add(row.get("run_id") or row.get("ts_utc"))
        group["last_seen"] = max(group["last_seen"], str(row.get("ts_utc") or ""))
    out = [
        {"skill": k[0], "signature": k[1], "verdict": k[2], "rule_ids": sorted(k[3]),
         "support": len(g["runs"]), "last_seen": g["last_seen"]}
        for k, g in groups.items()
        if len(g["runs"]) >= min_support
    ]
    out.sort(key=lambda lesson: lesson["last_seen"], reverse=True)
    out.sort(key=lambda lesson: lesson["support"], reverse=True)
    return out


def advise(skill, request, rows, *, min_support=MIN_SUPPORT) -> list[str]:
    sig = task_signature(skill, request)
    messages: list[str] = []
    for lesson in lessons(rows, skill=skill, signature=sig, min_support=min_support):
        rules = ", ".join(lesson["rule_ids"]) if lesson["rule_ids"] else "no rules"
        if lesson["verdict"] == "RED":
            messages.append(
                f"advisory: {lesson['support']} earlier runs of this request shape ended RED ({rules})"
            )
        else:
            messages.append(
                f"advisory: {lesson['support']} earlier runs of this request shape ended PASS ({rules})"
                f" — history only, no guarantee for this run"
            )
    return messages
