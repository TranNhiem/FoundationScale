"""C10 governed triggers: events may only produce operator-gated launch proposals.

Single-process use only (no cross-process lock); run one Governor per ledger.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from foundationskills.core import ledger


@dataclass(frozen=True)
class TriggerRecord:
    trigger_id: str
    skill: str
    owner: str
    scope: str
    condition: str
    expires_at: str
    budget_cap: int
    daily_cap: int
    action_class: str
    confirm_policy: str
    provenance: str
    quiet_hours: tuple[int, int] | None


class GovernorRefusal(ValueError):
    """A trigger record or governor invocation is invalid."""


def _parse_dt(value: object, field: str) -> _dt.datetime:
    try:
        if not isinstance(value, str):
            raise ValueError
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = _dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise GovernorRefusal(f"{field}: valid timezone-aware ISO-8601 datetime required") from None
    return parsed


def _utc_now(now: _dt.datetime) -> _dt.datetime:
    if not isinstance(now, _dt.datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise GovernorRefusal("now: timezone-aware datetime required")
    return now.astimezone(_dt.timezone.utc)


def validate_record(rec: TriggerRecord) -> None:
    for field in ("trigger_id", "skill", "owner", "provenance"):
        value = getattr(rec, field)
        if not isinstance(value, str) or not value.strip():
            raise GovernorRefusal(f"{field}: must be a non-empty string")
    for field in ("scope", "condition"):
        if not isinstance(getattr(rec, field), str):
            raise GovernorRefusal(f"{field}: must be a string")
    _parse_dt(rec.expires_at, "expires_at")
    for field in ("budget_cap", "daily_cap"):
        value = getattr(rec, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GovernorRefusal(f"{field}: must be an integer >= 1")
    if rec.action_class != "proposing":
        raise GovernorRefusal("action_class: only 'proposing' triggers are allowed; acting triggers are refused")
    if rec.confirm_policy not in ("spec-only", "ask"):
        raise GovernorRefusal("confirm_policy: must be 'spec-only' or 'ask'")
    q = rec.quiet_hours
    if q is not None and (not isinstance(q, tuple) or len(q) != 2 or q[0] == q[1] or
                          any(isinstance(h, bool) or not isinstance(h, int) or not 0 <= h <= 23 for h in q)):
        raise GovernorRefusal("quiet_hours: expected two distinct integer hours in 0..23")


def payload_key(source: str, payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((source + canonical).encode("utf-8")).hexdigest()[:16]


class Governor:
    def __init__(self, records: list[TriggerRecord], ledger_dir: str | Path | None,
                 global_daily_cap: int = 20):
        if isinstance(global_daily_cap, bool) or not isinstance(global_daily_cap, int) or global_daily_cap < 1:
            raise GovernorRefusal("global_daily_cap: must be an integer >= 1")
        self.ledger_dir, self.global_daily_cap = ledger_dir, global_daily_cap
        self._records: dict[str, TriggerRecord] = {}
        for rec in records:
            validate_record(rec)
            if rec.trigger_id in self._records:
                raise GovernorRefusal(f"trigger_id: duplicate {rec.trigger_id!r}")
            self._records[rec.trigger_id] = rec
        self._seen: dict[str, set[str]] = {}
        self._lifetime: defaultdict[str, int] = defaultdict(int)
        self._daily: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._global: defaultdict[str, int] = defaultdict(int)
        self._proposed: defaultdict[str, list[_dt.datetime]] = defaultdict(list)
        self._reload()

    def _reload(self) -> None:
        if not self.ledger_dir:
            return
        try:
            rows = list(ledger.read(Path(self.ledger_dir) / ledger.FILENAME))
        except FileNotFoundError:
            return
        except Exception as exc:
            raise GovernorRefusal(
                "ledger unreadable: refusing to start with empty caps/dedup"
            ) from exc
        for row in rows:
            if row.get("kind") != "trigger" or row.get("verdict") != "propose":
                continue
            tid = row.get("trigger_id")
            if not isinstance(tid, str):
                continue
            key = row.get("payload_key")
            if isinstance(key, str):
                self._seen.setdefault(tid, set()).add(key)
            self._lifetime[tid] += 1
            day = row.get("day")
            if isinstance(day, str):
                self._daily[(tid, day)] += 1
                self._global[day] += 1
            try:
                ts = _parse_dt(row.get("ts_utc"), "ts_utc")
            except GovernorRefusal:
                continue
            self._proposed[tid].append(ts.astimezone(_dt.timezone.utc))

    def on_event(self, trigger_id: str, source: str, payload: dict, now: _dt.datetime,
                 condition_holds: bool) -> dict:
        now_utc = _utc_now(now)
        rec = self._records.get(trigger_id)
        key = payload_key(source, payload)
        day = now_utc.date().isoformat()

        def finish(decision: str, reason: str, proposal: dict | None = None) -> dict:
            request = {"trigger_id": trigger_id, "payload_key": key}
            extra = {**request, "decision": decision, "reason": reason, "day": day}
            row = ledger.record("trigger", rec.skill if rec and rec.skill else "unknown", "c10",
                                verdict=decision, exit_code=None, wall_clock_ms=None,
                                request=request, extra=extra)
            written = ledger.append(self.ledger_dir, row)
            if decision == "propose" and self.ledger_dir is not None and not written:
                # caps and dedup are rebuilt from the ledger on restart: an untraceable proposal would reset them
                self._seen[trigger_id].discard(key)
                self._lifetime[trigger_id] -= 1
                self._daily[(trigger_id, day)] -= 1
                self._global[day] -= 1
                self._proposed[trigger_id].pop()
                return {"trigger_id": trigger_id, "decision": "ledger_unwritable",
                        "reason": "the proposal could not be recorded in the ledger, so it is withheld", "proposal": None}
            return {"trigger_id": trigger_id, "decision": decision, "reason": reason, "proposal": proposal}

        if rec is None:
            return finish("unknown", "no such trigger_id")
        if key in self._seen.get(trigger_id, set()):
            return finish("duplicate", "payload key was already proposed for this trigger")
        if now_utc >= _parse_dt(rec.expires_at, "expires_at").astimezone(_dt.timezone.utc):
            return finish("expired", f"record expired at {rec.expires_at}")
        if condition_holds is not True:  # only a real True opens the gate; truthy junk does not
            return finish("condition_false", "condition does not hold against current state")
        q = rec.quiet_hours
        quiet = q is not None and ((q[0] <= now.hour < q[1]) if q[0] < q[1]
                                   else (now.hour >= q[0] or now.hour < q[1]))
        if quiet:
            return finish("quiet_hours", "local hour is inside the quiet-hours window")
        if self._lifetime[trigger_id] >= rec.budget_cap:
            return finish("budget_cap", "lifetime budget_cap reached")
        if self._daily[(trigger_id, day)] >= rec.daily_cap:
            return finish("daily_cap", "per-trigger daily_cap reached")
        if self._global[day] >= self.global_daily_cap:
            return finish("global_cap", "global_daily_cap reached")
        if self.ledger_dir is None:
            return finish(
                "ledger_unwritable",
                "no ledger_dir: proposals are withheld because they could not be traced",
            )
        self._seen.setdefault(trigger_id, set()).add(key)
        self._lifetime[trigger_id] += 1
        self._daily[(trigger_id, day)] += 1
        self._global[day] += 1
        self._proposed[trigger_id].append(now_utc)
        proposal = {"skill": rec.skill, "emit": "fs_launch_spec", "confirm_policy": rec.confirm_policy,
                    "requires": "fskills hash + fskills launch --confirm"}
        return finish("propose", "proposal only; operator approval is still required", proposal)

    def sweep_expected(self, now: _dt.datetime) -> list[dict]:
        now_utc = _utc_now(now)
        cutoff = now_utc - _dt.timedelta(hours=24)
        missing = []
        for tid, rec in sorted(self._records.items()):
            if now_utc >= _parse_dt(rec.expires_at, "expires_at").astimezone(_dt.timezone.utc):
                continue
            times = self._proposed.get(tid, [])
            if any(cutoff <= ts <= now_utc for ts in times):
                continue
            hours = 24.0 if not times else max(24.0, (now_utc - max(times)).total_seconds() / 3600)
            missing.append({"trigger_id": tid, "missing_since_hours": round(hours, 2)})
        return missing
