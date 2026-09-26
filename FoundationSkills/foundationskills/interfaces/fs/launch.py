
"""Confirmation-gated launching of an ``fs_launch_spec`` payload.

Order of operations (deliberate, do not reorder):

1. ``require_confirmation(spec, token)`` -- the hash is ``plan_hash`` over the
   FULL fs_launch_spec payload; a mismatch raises ``ConfirmationRequired``
   before anything else happens.
2. A spec is executed only when positive evidence says it can be: an
   ``executable`` that is anything but ``True`` (including absent) refuses.
3. ``dry_run_argv`` runs first when present; FS's dry-run exercises the full
   validation prologue with 0 GPUs, so any non-zero exit blocks the launch.
4. Submit via ``sbatch`` (wrapped in ``bash -lc`` -- Slurm on this estate
   needs a login shell), else run the argv directly.

The runner is injectable (``subprocess.run``-shaped) so tests never submit.
For direct (non-sbatch) runs the return dict carries ``returncode`` and
``pid=None``: ``subprocess.run`` does not expose a pid, and doctrine prefers
an honest null to an invented one.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

from foundationskills.core.orchestrator import ConfirmationRequired, require_confirmation
from foundationskills.interfaces.fs.sbatch import write_sbatch

__all__ = ("LaunchRefused", "launch")


class LaunchRefused(RuntimeError):
    """The launch was refused; the message names the missing input or gap."""


def launch(
    spec: dict[str, Any],
    *,
    confirm: str | None,
    submit: bool = True,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Launch a confirmed fs_launch_spec. Returns {job_id|pid, command, dry_run_rc, returncode}."""
    require_confirmation(spec, confirm)

    if spec.get("executable") is not True:
        missing = spec.get("missing") or "spec.executable is not true (no positive evidence of executability)"
        raise LaunchRefused(f"launch refused: {missing}")

    dry_run_rc: int | None = None
    dry_run_argv = spec.get("dry_run_argv")
    if dry_run_argv:
        completed = runner(list(dry_run_argv), capture_output=True, text=True, check=False)
        dry_run_rc = int(completed.returncode)
        if dry_run_rc != 0:
            tail = (getattr(completed, "stdout", "") or "")[-2000:]
            raise LaunchRefused(
                f"dry-run failed with rc={dry_run_rc}; the FS validation prologue did not PASS. tail: {tail}"
            )

    job_id: str | None = None
    pid: int | None = None
    returncode: int | None = None
    command: str

    if submit and spec.get("sbatch"):
        output_dir = Path(str(spec.get("output_dir") or "."))
        sbatch_path = output_dir / f"launch-{spec.get('stage_name', 'stage')}.sbatch"
        write_sbatch(sbatch_path, str(spec["sbatch"]))
        command = f"sbatch {shlex.quote(str(sbatch_path))}"
        completed = runner(["bash", "-lc", command], capture_output=True, text=True, check=False)
        stdout = getattr(completed, "stdout", "") or ""
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if match is None:
            raise LaunchRefused(
                f"sbatch did not report a job id (rc={completed.returncode}): {stdout[-1000:]}"
            )
        job_id = match.group(1)
    else:
        argv = [str(a) for a in (spec.get("argv") or [])]
        if not argv:
            raise LaunchRefused("launch refused: spec.argv is empty and no sbatch path applies")
        command = shlex.join(argv)
        completed = runner(argv, capture_output=True, text=True, check=False)
        returncode = int(completed.returncode)

    return {
        "job_id": job_id,
        "pid": pid,
        "command": command,
        "dry_run_rc": dry_run_rc,
        "returncode": returncode,
    }
