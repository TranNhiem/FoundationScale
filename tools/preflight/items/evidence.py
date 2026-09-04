"""Item 8 -- evidence completeness."""

from __future__ import annotations

import glob
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .._base import (
    Coverage,
    Verdict,
)
from .._core import (
    CheckResult,
    _finalize,
)

# ---------------------------------------------------------------------------
# Item 8 — evidence completeness
# ---------------------------------------------------------------------------


def _check_evidence_completeness(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 8: the evidence the other checks consumed is complete and LIVE.

    Contract: evidence.log_glob resolves to exactly world_size per-rank logs;
    every log parses evidence.mem_regex (one capture group, MiB, float) ≥1
    time; no log's mtime is older than evidence.max_log_age_s; if
    evidence.slurm_job_id is set, `bash -lc 'sacct …'` (item 8's wrapping
    discipline is satisfied BY CONSTRUCTION: the only Slurm query this file
    ever makes is that literal argv) must corroborate the job; a null job id
    requires the declared allow_no_slurm opt-out, recorded loudly.
    """
    world_size = int(cfg["world_size"])
    evidence: dict[str, Any] = {"world_size": world_size}
    # glob.glob, deliberately: log_glob is an operator-supplied PATTERN (an
    # absolute pattern in the real config), while Path.glob roots a relative
    # pattern at a base dir and refuses absolute ones — a rewrite would change
    # which rank logs this check enumerates. The sorted() order is itself part
    # of the denominator, so it stays exactly where it is.
    paths = sorted(glob.glob(s["log_glob"]))  # noqa: PTH207 — see comment above
    evidence["logs_found"] = len(paths)
    evidence["log_paths"] = paths
    if len(paths) != world_size:
        return _finalize(
            "evidence_completeness",
            "Evidence completeness",
            Verdict.FAIL,
            Coverage(len(paths), "per-rank logs", expected=world_size),
            f"found {len(paths)} logs for world size {world_size} via {s['log_glob']} — "
            f"per-rank evidence is incomplete",
            evidence,
        )

    try:
        mem_re = re.compile(s["mem_regex"])
    except re.error as exc:
        return _finalize(
            "evidence_completeness",
            "Evidence completeness",
            Verdict.ERROR,
            Coverage(0, "per-rank logs", expected=world_size),
            f"evidence.mem_regex does not compile: {exc}",
            evidence,
        )

    checked = 0
    max_mib = -1.0
    max_src = None
    unparsable = []
    stale = []
    now = time.time()
    for path in paths:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            unparsable.append(f"{path} (unreadable: {exc})")
            continue
        vals = [float(m.group(1)) for m in mem_re.finditer(text)]
        if not vals:
            unparsable.append(path)
            continue
        peak = max(vals)
        if peak > max_mib:
            max_mib, max_src = peak, path
        age = now - p.stat().st_mtime
        if age > int(s["max_log_age_s"]):
            stale.append(f"{path} (age {age:.0f}s > {s['max_log_age_s']}s)")
        checked += 1
    evidence["max_memory_mib"] = max_mib if max_src else None
    evidence["max_memory_source"] = max_src
    if unparsable:
        return _finalize(
            "evidence_completeness",
            "Evidence completeness",
            Verdict.FAIL,
            Coverage(checked, "per-rank logs", expected=world_size),
            f"{len(unparsable)} logs carry no parsed memory line ({s['mem_regex']}): "
            + ", ".join(unparsable[:4]),
            evidence,
        )
    if stale:
        return _finalize(
            "evidence_completeness",
            "Evidence completeness",
            Verdict.FAIL,
            Coverage(checked, "per-rank logs", expected=world_size),
            f"{len(stale)} logs violate .out mtime liveness: " + ", ".join(stale[:4]),
            evidence,
        )

    job = s.get("slurm_job_id")
    if job:
        # Item 8, satisfied structurally: every Slurm query this tool performs
        # goes through this exact bash -lc argv. Timeout-bounded; any failure
        # means the job identity could not be corroborated — BLOCK, not shrug.
        cmd = ["bash", "-lc", f"sacct -j {shlex.quote(str(job))} --format=JobID,Elapsed,State -Pn"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _finalize(
                "evidence_completeness",
                "Evidence completeness",
                Verdict.ERROR,
                Coverage(checked, "per-rank logs", expected=world_size),
                f"slurm corroboration could not run ({exc}); job identity unverified",
                evidence,
            )
        rows = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        evidence["slurm"] = {
            "job_id": job,
            "argv": " ".join(cmd)[:80] + "…",
            "rc": proc.returncode,
            "rows": len(rows),
        }
        if proc.returncode != 0 or not rows:
            return _finalize(
                "evidence_completeness",
                "Evidence completeness",
                Verdict.FAIL,
                Coverage(checked, "per-rank logs", expected=world_size),
                f"sacct for job {job} returned rc={proc.returncode}, {len(rows)} rows — "
                f"the evidence cannot be tied to a scheduler job",
                evidence,
            )
    else:
        evidence["slurm"] = {
            "job_id": None,
            "opt_out": {
                "allow_no_slurm": s["allow_no_slurm"],
                "reason": s.get("slurm_absent_reason", ""),
            },
        }
        # Config validation already refused an undeclared absence.

    return _finalize(
        "evidence_completeness",
        "Evidence completeness",
        Verdict.PASS,
        Coverage(checked, "per-rank logs", expected=world_size),
        f"{checked}/{world_size} rank logs live and parsed; peak memory {max_mib:.1f} MiB "
        f"({max_src}); slurm {'corroborated job ' + str(job) if job else 'absent BY DECLARATION'}",
        evidence,
    )
