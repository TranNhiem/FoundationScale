"""Envelope consent, derived trial tokens and the budget-decremented trial gate (AR-LN-006).

Approval model: one human consent (the ``confirm`` hash) derives the envelope token over the approved
envelope payload. Per-trial launch tokens are then derived from it (``fs-ar-trial-v1``) and every
ledgered ``job_submitted`` burns runs and measured GPU hours out of the envelope budget, so an
exhausted envelope refuses the next trial with a named AR-LN-006 reason. ``budget_after`` is
advisory: the append-only chain, not the snapshot, is authoritative.
"""
from __future__ import annotations

import math
from typing import Any

from foundationskills.skills.auto_research.ledger import canonical, sha256_hex

RULE_ID = "AR-LN-006"
ENVELOPE_PREFIX = b"fs-ar-envelope-v1"
TRIAL_PREFIX = b"fs-ar-trial-v1"
MIRROR_KEYS = ("max_runs", "gpu_hours_total")

ENVELOPE_MISSING = "envelope_missing"
BUDGET_MISMATCH = "envelope_budget_mismatch"
TOKEN_MISMATCH = "token_mismatch"
EXHAUSTED_RUNS = "envelope_exhausted:runs"
EXHAUSTED_HOURS = "envelope_exhausted:gpu_hours"


def envelope_token(confirm: str, envelope: dict) -> str:
    """One human consent: sha256("fs-ar-envelope-v1|" + confirm + "|" + canonical(envelope)).

    ``envelope`` is the approved envelope payload *without* its own ``envelope_token`` field; the
    derived token is recorded alongside the payload and read back from there by ``envelope_check``.
    """
    return sha256_hex(ENVELOPE_PREFIX + b"|" + str(confirm).encode("utf-8") + b"|" + canonical(envelope))


def trial_launch_token(env_token: str, trial_spec: dict) -> str:
    """Derived per-trial token: sha256("fs-ar-trial-v1|" + env_token + "|" + canonical(trial_spec))."""
    return sha256_hex(TRIAL_PREFIX + b"|" + str(env_token).encode("utf-8") + b"|" + canonical(trial_spec))


def _number(value: Any) -> float | None:
    """``float(value)`` when it is a finite real number, else ``None`` (never raises)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _budget(source: Any) -> dict:
    """The ``budget`` block of ``source`` as a fresh dict ({} when absent or malformed)."""
    block = source.get("budget") if isinstance(source, dict) else None
    return dict(block) if isinstance(block, dict) else {}


def _runs(launches: Any) -> int:
    return len(launches) if isinstance(launches, list) else 0


def _mirrored(spec: Any, envelope: Any) -> bool:
    """True when ``envelope.budget`` mirrors ``spec.budget`` on the gate keys (numeric comparison)."""
    in_spec, in_env = _budget(spec), _budget(envelope)
    for key in MIRROR_KEYS:
        declared, consented = _number(in_spec.get(key)), _number(in_env.get(key))
        if declared is None or consented is None or declared != consented:
            return False
    return True


def budget_snapshot(envelope: dict, launches: list, usage: dict, trial_spec: dict) -> dict:
    """``budget_after`` admitting this trial: {"runs_left", "hours_left"}; advisory, the chain is authoritative."""
    budget = _budget(envelope)
    max_runs = _number(budget.get("max_runs"))
    total = _number(budget.get("gpu_hours_total")) or 0.0
    used = _number(usage.get("used_gpu_hours") if isinstance(usage, dict) else None) or 0.0
    est = _number(trial_spec.get("gpu_hours_est") if isinstance(trial_spec, dict) else None) or 0.0
    runs_left = (int(max_runs) if max_runs is not None else 0) - _runs(launches) - 1
    return {"budget_after": {"runs_left": runs_left, "hours_left": float(total - used - est)}}


def envelope_check(
    spec: dict,
    envelope: dict,
    trial_spec: dict,
    launches: list,
    usage: dict,
    supplied_token: str,
) -> list[tuple[str, str]]:
    """AR-LN-006 gate: [] admits the trial, else one (RULE_ID, reason) per block.

    Reasons: envelope_missing | envelope_budget_mismatch | token_mismatch |
    envelope_exhausted:runs | envelope_exhausted:gpu_hours. Burn is measured-or-declared
    ``usage["used_gpu_hours"]`` plus this trial's ``gpu_hours_est`` against the consented budget.
    """
    if not isinstance(envelope, dict) or not envelope.get("envelope_token") or not isinstance(envelope.get("budget"), dict):
        return [(RULE_ID, ENVELOPE_MISSING)]
    problems: list[tuple[str, str]] = []
    if not _mirrored(spec, envelope):
        problems.append((RULE_ID, BUDGET_MISMATCH))
    derived = trial_launch_token(envelope["envelope_token"], trial_spec if isinstance(trial_spec, dict) else {})
    if supplied_token != derived:
        problems.append((RULE_ID, TOKEN_MISMATCH))
    after = budget_snapshot(envelope, launches, usage, trial_spec)["budget_after"]
    if after["runs_left"] < 0:
        problems.append((RULE_ID, EXHAUSTED_RUNS))
    if after["hours_left"] < 0.0:
        problems.append((RULE_ID, EXHAUSTED_HOURS))
    return problems
