"""Post-admission skill lifecycle: ledger history only PROPOSES stage changes.

Measured call history may suggest demoting a skill (active -> downweighted ->
quarantined -> retired), but nothing is applied: every stage change waits for an
operator approval recorded in a persistent, atomically-updated state file.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

SCHEMA = "fskills.skill_lifecycle/1"
STAGES = ("active", "downweighted", "quarantined", "retired")
THRESHOLDS = {"downweighted": 0.25, "quarantined": 0.50, "retired": 0.75}
MIN_CALLS = 20  # measured calls (PASS/RED skill rows); fewer -> "unmeasured", no proposal


def _field(row, key, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def failure_rate(rows, skill) -> tuple[int, float | None]:
    """Return (measured skill calls, share of failures) for one skill in the ledger."""
    n = 0
    failed = 0
    for row in rows:
        if _field(row, "kind") != "skill" or _field(row, "skill_name") != skill:
            continue
        verdict = _field(row, "verdict")
        if verdict not in ("PASS", "RED"):
            continue
        n += 1
        if verdict == "RED":  # a RED verdict covers FAIL signals such as exit code 5
            failed += 1
    return (n, (failed / n) if n else None)


def proposed_stage(n, rate) -> str | None:
    """Propose the highest escalation stage whose threshold the failure rate meets."""
    if rate is None or n < MIN_CALLS:
        return None
    proposed = "active"
    for stage in STAGES[1:]:
        if rate >= THRESHOLDS[stage]:
            proposed = stage
    return proposed


def load_state(path) -> dict:
    """Load the approval state file; a missing file yields empty skills."""
    state = {"schema": SCHEMA, "skills": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return state
    state["skills"] = {k: dict(v) for k, v in (raw.get("skills") or {}).items()}
    return state


def stage_of(state, skill) -> str:
    """Return the approved stage of a skill, defaulting to "active"."""
    entry = ((state or {}).get("skills") or {}).get(skill) or {}
    stage = entry.get("stage", "active")
    return stage if stage in STAGES else "active"


def proposals(rows, state, skills) -> list[dict]:
    """List stage changes the evidence suggests; this never applies anything."""
    out = []
    for skill in skills:
        n, rate = failure_rate(rows, skill)
        to_stage = proposed_stage(n, rate)
        from_stage = stage_of(state, skill)
        if to_stage is None or to_stage == from_stage:
            continue
        out.append({"skill": skill, "from": from_stage, "to": to_stage,
                    "n": n, "failure_rate": rate})
    out.sort(key=lambda p: p["skill"])
    return out


def approve(path, skill, to_stage, approver, evidence: dict) -> dict:
    """Record one operator-approved stage change (atomic replace) and return the entry."""
    if to_stage not in STAGES:
        raise ValueError(f"unknown skill lifecycle stage: {to_stage!r}")
    state = load_state(path)
    entry = {
        "stage": to_stage,
        "approved_by": approver,
        "approved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence": dict(evidence or {}),
    }
    state["skills"][skill] = entry
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".fskills-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dict(entry)


def routable(state, skill) -> bool:
    """Only routable skills may be selected; quarantined and retired skills are not."""
    return stage_of(state, skill) not in ("quarantined", "retired")
