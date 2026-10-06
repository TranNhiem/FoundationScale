"""Optional live queue state for auto_research -- injected ``state_fn`` only.

``job_states`` answers per job id in the state vocabulary an injected
``state_fn`` returns: ``pending``|``running``|``terminal``|``unknown``|``None``.
A failed or missing ``squeue`` reads ``None`` per id: fail closed, the
ledger-derived default then holds every unresolved slot. Nothing imports this
module as a default (it is CPU-inert without an injected runner).
"""
from __future__ import annotations

import re
import shlex
import subprocess
from typing import Any, Callable

IN_FLIGHT_STATES = {"PD", "R", "RQ", "CG", "S", "RS"}

_JOB_ID_RE = re.compile(r"^[0-9]+(_[0-9]+)?$")


def parse_states(text: Any) -> dict[str, str]:
    """Parse ``squeue -h -o '%i %t'`` rows into ``{job_id: slurm_state}``.

    The first two whitespace fields are the id and the state (later fields such
    as runtimes are ignored); rows that are blank, one-field, or that do not
    start with a valid job id are garbage and skipped.
    """
    out: dict[str, str] = {}
    for line in str(text if text is not None else "").splitlines():
        fields = line.split()
        if len(fields) < 2 or not _JOB_ID_RE.match(fields[0]):
            continue
        out[fields[0]] = fields[1]
    return out


def _live_state(code: str) -> str:
    """Slurm state code into the ``state_fn`` vocabulary (fail closed)."""
    if code == "PD":
        return "pending"
    if code in IN_FLIGHT_STATES:
        return "running"
    return "unknown"


def job_states(job_ids: Any, runner: Callable[..., Any] = subprocess.run) -> dict[str, str | None]:
    """Live queue state per job id in ONE bash -lc squeue call (state_fn vocabulary).

    Deliberate: a valid id absent from a successful query reads ``terminal`` -- ``squeue``
    holds live jobs only, so absence is the measurement of finish.
    Per id (duplicates ask once): ``pending``/``running`` (squeue still knows the
    job), ``terminal`` (absent from a successful query -- squeue drops finished
    jobs), ``unknown`` (seen with an unclassifiable state) or ``None`` (invalid
    id; a missing, raising, or nonzero-rc squeue reads ``None`` per id). Invalid
    ids never reach the runner, and a request with no valid id does not run it
    at all. A ``bool`` is never a job id: it maps nothing and never reaches the
    runner.
    """
    if isinstance(job_ids, bool):
        return {}  # a bool is invalid: it maps nothing and never reaches the runner
    if isinstance(job_ids, (str, bytes, int)):
        job_ids = [job_ids]  # a scalar int job id is treated as ``[str(id)]``
    out: dict[str, str | None] = {}
    valid: list[str] = []
    for raw in list(job_ids or []):
        if isinstance(raw, bool):
            continue  # a bool is invalid: it maps nothing and never reaches the runner
        job_id = str(raw)
        out[job_id] = None
        if _JOB_ID_RE.match(job_id) and job_id not in valid:
            valid.append(job_id)
    if not valid:
        return out
    command = ["bash", "-lc", shlex.join(["squeue", "-h", "-o", "%i %t", "-j", ",".join(valid)])]
    try:
        result = runner(command, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return out
    if getattr(result, "returncode", 1) != 0:
        return out
    parsed = parse_states(str(getattr(result, "stdout", "") or ""))
    for job_id in valid:
        code = parsed.get(job_id)
        out[job_id] = "terminal" if code is None else _live_state(code)
    return out
