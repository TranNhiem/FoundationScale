"""Job-id ownership and safe cancel for auto_research campaigns.

Only job ids a campaign's own ``job_submitted`` entries recorded may reach ``scancel``:
malformed ids never enter a shell string and foreign ids fail the ownership check. Both are
dropped with a named reason, counted, and raised as AR-LN-007 findings without ever touching
the runner. A successful cancel is itself evidence (``job_cancelled``) appended through the
same append-only chain (no ledger API grows).
"""
from __future__ import annotations

import re
import subprocess
from typing import Any, Callable

from foundationskills.skills.auto_research.ledger import Ledger, canonical, sha256_hex

JOB_ID_RE = re.compile(r"^[0-9]+(_[0-9]+)?$")


def submitted_jobs(ledger: Ledger, campaign: str) -> list[dict[str, Any]]:
    """Payloads of every ``job_submitted`` entry for THIS campaign only."""
    return [
        ledger.payload(entry)
        for entry in ledger.entries()
        if entry.get("op") == "job_submitted" and entry.get("campaign") == campaign
    ]


def owned_job_ids(ledger: Ledger, campaign: str) -> set[str]:
    """Job ids this campaign's ledger submitted — the only ids cancel may ever touch."""
    return {
        str(payload["job_id"])
        for payload in submitted_jobs(ledger, campaign)
        if payload.get("job_id") is not None
    }


def cancel_token(job_ids: list[str], reason: str) -> str:
    """Deterministic id of one cancel intent: sha256 over the ``fs-ar-cancel-v1`` payload."""
    return sha256_hex(
        canonical({"prefix": "fs-ar-cancel-v1", "job_ids": list(job_ids), "reason": reason})
    )


def cancel_jobs(
    ledger: Ledger,
    campaign: str,
    job_ids: list[str],
    *,
    reason: str,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Cancel what THIS campaign submitted, in ONE ``bash -lc "scancel ..."`` batch call.

    Order per requested id: format validity, then ownership. Malformed ids get the named drop
    ``invalid_job_id:<id>`` and foreign ones ``refused_foreign_job:<id>`` — neither ever reaches
    the runner, and both raise an ``AR-LN-007`` finding. Owned ids are validated into the shell
    string and cancelled in one call (``capture_output=True, text=True, timeout=60``); a
    ``job_cancelled`` entry is appended only when that call succeeds. Runner failure or error
    names ``scancel_failed:<id>`` / ``scancel_unavailable:<id>`` and appends nothing.

    Returns ``{"cancelled", "drops", "returncode", "findings"}`` with
    ``findings == [(rule_id, message), ...]``.
    """
    owned = owned_job_ids(ledger, campaign)
    batch: list[str] = []
    drops: list[str] = []
    findings: list[tuple[str, str]] = []
    for job_id in dict.fromkeys(str(item) for item in job_ids):
        if not JOB_ID_RE.match(job_id):
            drops.append(f"invalid_job_id:{job_id}")
            findings.append(("AR-LN-007", f"job {job_id} is not a valid job id for ledger {campaign}"))
            continue
        if job_id not in owned:
            drops.append(f"refused_foreign_job:{job_id}")
            findings.append(("AR-LN-007", f"job {job_id} is not owned by ledger {campaign}"))
            continue
        batch.append(job_id)
    if not batch:
        return {"cancelled": [], "drops": drops, "returncode": 0, "findings": findings}
    try:
        result = runner(
            ["bash", "-lc", "scancel " + " ".join(batch)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:  # noqa: BLE001 — injected connector; failure is named, never silent
        drops.extend(f"scancel_unavailable:{job_id}" for job_id in batch)
        return {"cancelled": [], "drops": drops, "returncode": 1, "findings": findings}
    returncode = int(getattr(result, "returncode", 1))
    if returncode != 0:
        drops.extend(f"scancel_failed:{job_id}" for job_id in batch)
        return {"cancelled": [], "drops": drops, "returncode": returncode, "findings": findings}
    ledger.append(
        "job_cancelled",
        campaign,
        "-",
        {"job_ids": list(batch), "reason": reason, "token": cancel_token(batch, reason)},
    )
    return {"cancelled": list(batch), "drops": drops, "returncode": 0, "findings": findings}
