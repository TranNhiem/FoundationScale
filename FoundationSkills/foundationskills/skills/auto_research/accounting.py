"""Campaign GPU-hour accounting: measured (sacct) XOR declared (gpu_hours_est)."""
from __future__ import annotations

import math
from typing import Any, Callable

from foundationskills.interfaces.fs.sacct import query_job_gpu_hours


def _usable(value: Any) -> float | None:
    """A usable hour figure: finite non-negative real number; else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _measured_state(value: Any) -> str:
    """``"positive"`` for a finite measurement ``> 0``; ``"zero"`` when sacct measured ``<= 0`` (a
    measurement of nothing is NOT a measurement); ``"missing"`` for anything that is not a finite real
    number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "missing"
    number = float(value)
    if not math.isfinite(number):
        return "missing"
    return "zero" if number <= 0.0 else "positive"


def campaign_usage(
    job_entries: list[dict] | None,
    measure: Callable[[str], float | None] = query_job_gpu_hours,
) -> dict:
    """Account one campaign's GPU hours from its ``job_submitted`` (and synthetic) payloads.

    Per job the scheduler's measurement wins (``source`` ``measured_sacct``);
    else the declared ``gpu_hours_est`` (``declared_est``, drop
    ``sacct_unavailable:<id>``); else 0.0 (``uncounted``, drop
    ``no_accounting:<id>``). Measured and declared are never both counted for
    one job; every drop is named and counted. Accounting rules:

    - each real ``job_id`` counts ONCE: a later entry with the same id adds
      ``duplicate_job:<id>`` and no hours (and is never measured again);
    - an entry with a missing/empty ``job_id`` is NEVER measured (``measure`` is
      not called for it): it falls back to its declared ``gpu_hours_est`` with
      ``sacct_unavailable:entry:<index>`` or to ``uncounted`` with
      ``no_accounting:entry:<index>``;
    - a measurement of ``<= 0.0`` is not a measurement: the entry falls back to
      declared/uncounted and drops ``sacct_zero:<id>`` (instead of
      ``sacct_unavailable``).
    """
    per_job: dict[str, dict[str, Any]] = {}
    drops: list[str] = []
    used = 0.0
    counted: set[str] = set()
    for index, entry in enumerate(job_entries or []):
        payload = entry if isinstance(entry, dict) else {}
        raw_id = str(payload.get("job_id", "") or "").strip()
        job_id = raw_id or f"entry:{index}"
        if raw_id and job_id in counted:  # one real job id is never counted twice
            drops.append(f"duplicate_job:{job_id}")
            continue
        counted.add(job_id)
        value = measure(job_id) if (raw_id and measure is not None) else None  # never measure a missing id
        state = _measured_state(value)
        if state == "positive":
            hours, source = float(value), "measured_sacct"
        else:
            declared = _usable(payload.get("gpu_hours_est"))
            if declared is not None:
                hours, source = declared, "declared_est"
            else:
                hours, source = 0.0, "uncounted"
            if state == "zero":
                drops.append(f"sacct_zero:{job_id}")
            elif declared is not None:
                drops.append(f"sacct_unavailable:{job_id}")
            if declared is None:
                drops.append(f"no_accounting:{job_id}")
        per_job[job_id] = {"gpu_hours": hours, "source": source}
        used += hours
    return {"used_gpu_hours": used, "per_job": per_job, "drops": drops}
