"""Scheduler-measured GPU hours via ``sacct``.

AR production policy: the scheduler's measurement replaces the launch-time
``gpu_hours_est`` when available; the fallback is named
(``sacct_unavailable:<id>``) and counted by ``campaign_usage``.
Counters at 0 are uncountable: a zero elapsed or a zero GPU count yields
no measurement at all.
"""
from __future__ import annotations

import math
import re
import shlex
import subprocess
from typing import Any


def parse_gpu_count(tres: str) -> int | None:
    """Pull the GPU count out of an AllocTRES string; ``None`` when unusable.

    Reads ``gres/gpu=8,cpu=4``, ``gpu:8`` and type-qualified
    ``gres/gpu:a100:8``; garbage and ``gres/gpu=0`` (counters at 0 are
    uncountable) read ``None``.
    """
    if not isinstance(tres, str):
        return None
    for raw in re.split(r"[,;]", tres):
        field = raw.strip()
        low = field.lower()
        if not (
            low.startswith("gres/gpu")
            or low.startswith("gpu:")
            or low.startswith("gpu=")
            or low == "gpu"
        ):
            continue
        tail = re.split(r"[:=]", field)[-1].strip()
        if not tail.isdigit():
            return None
        count = int(tail)
        return count if count > 0 else None
    return None


def job_gpu_hours(elapsed_raw: str, tres: str) -> float | None:
    """``ElapsedRaw`` seconds x gpus / 3600; ``None`` when either is unusable."""
    gpus = parse_gpu_count(tres)
    if gpus is None:
        return None
    try:
        elapsed = float(str(elapsed_raw).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        return None
    return elapsed * gpus / 3600.0


_JOB_ID_RE = re.compile(r"^[0-9]+(_[0-9]+)?$")


def query_job_gpu_hours(job_id: str, runner: Any = subprocess.run) -> float | None:
    """Measure one job's GPU hours via ``sacct``; ``None`` when unusable.

    Nonzero rc, a parse miss and an unavailable sacct all read ``None``: the
    caller falls back to the declared estimate with a named drop.
    """
    if not _JOB_ID_RE.match(str(job_id)):  # the id reaches a shell: anything but a Slurm id is refused
        return None
    sacct = ["sacct", "-n", "-X", "-P", "-j", str(job_id), "--format", "ElapsedRaw,AllocTRES"]
    try:
        # Slurm on this estate needs a login shell (SLURM_CONF/PATH), like sbatch and scancel.
        result = runner(["bash", "-lc", shlex.join(sacct)], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    line = ""
    for raw in str(getattr(result, "stdout", "") or "").splitlines():
        if raw.strip():
            line = raw.strip()
            break
    if not line:
        return None
    fields = line.split("|")
    if len(fields) < 2:
        return None
    return job_gpu_hours(fields[0], fields[1])
