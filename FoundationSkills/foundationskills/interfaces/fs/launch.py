
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
import time
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
    measurement_only: bool = False,
) -> dict[str, Any]:
    """Launch a confirmed fs_launch_spec. Returns {job_id|pid, command, dry_run_rc, returncode}."""
    started = time.monotonic()
    # FS records code provenance from the launch cwd, and the spec env carries
    # measured launch facts (e.g. GB200 NCCL pins); both apply to every process.
    import os as _os

    run_kwargs: dict[str, Any] = {"env": {**_os.environ, **{str(k): str(v) for k, v in (spec.get("env") or {}).items()}}}
    if spec.get("cwd"):
        run_kwargs["cwd"] = str(spec["cwd"])
    require_confirmation(spec, confirm)
    from foundationskills.core.harness import validate_manifest
    from foundationskills.core.harness_collect import attach
    from foundationskills.core.provenance import make_provenance

    if "harness_manifest" not in spec:  # an in-memory spec that never passed write_artifact: record the pair now,
        spec = attach(spec, make_provenance("fskills-launch", "1"))  # honestly marked as not covered by the hash
        spec["harness_manifest_hash_covered"] = False
    manifest_problems = validate_manifest(spec["harness_manifest"])
    if manifest_problems:  # C2: a malformed model-harness identity is refused, never repaired
        raise LaunchRefused("launch refused: harness_manifest " + "; ".join(manifest_problems[:3]))
    _write_manifest_sidecar(spec)

    if spec.get("executable") is not True:
        # A measurement-only run (e.g. FS RL, which trains but saves nothing) is
        # allowed ONLY when asked for explicitly and the spec says the sole gap is
        # the hand-off; it is still behind the same confirmation hash.
        if not (measurement_only and spec.get("measurement_only_ok") is True):
            missing = spec.get("missing") or "spec.executable is not true (no positive evidence of executability)"
            raise LaunchRefused(f"launch refused: {missing}")

    # The fskills-rl driver reads its config from disk; the spec carries it inline.
    # Write it where argv points BEFORE the dry-run (a launch never wrote it, so
    # every RL/preference launch would have failed on a missing file).
    if isinstance(spec.get("rl_config"), dict):
        argv_list = [str(a) for a in (spec.get("argv") or [])]
        if "--config" in argv_list and argv_list.index("--config") + 1 < len(argv_list):
            import json as _json

            config_path = Path(argv_list[argv_list.index("--config") + 1])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = config_path.with_name(f".{config_path.name}.tmp")
            tmp.write_text(_json.dumps(spec["rl_config"], indent=1, sort_keys=True), encoding="utf-8")
            tmp.replace(config_path)
    dry_run_rc: int | None = None
    dry_run_argv = spec.get("dry_run_argv")
    if dry_run_argv:
        completed = runner(list(dry_run_argv), capture_output=True, text=True, check=False, **run_kwargs)
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

    extra: dict[str, Any] = {}
    if submit and not spec.get("sbatch") and spec.get("launch_target") == "slurm":
        # A cluster spec without a script would run its argv HERE (e.g. a login node with no GPU).
        raise LaunchRefused(
            "launch refused: a slurm-target spec carries no sbatch, so submitting would run the argv on this "
            "host; re-emit it, or run it inside an allocation with --no-submit (srun --overlap ... fskills launch "
            "--no-submit)"
        )
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
        live_log = _launch_log_path(spec) if runner is subprocess.run else None
        fh = None
        if live_log is not None:
            try:
                fh = live_log.open("w", encoding="utf-8")
            except OSError:  # an unwritable log never blocks the run; output is captured instead
                live_log = None
        if fh is not None and live_log is not None:
            # Stream to disk as the run goes: a multi-hour run captured in memory
            # is unobservable until it ends (no step, loss or throughput to watch).
            with fh:
                completed = runner(argv, stdout=fh, stderr=subprocess.STDOUT, text=True, check=False, **run_kwargs)
            output = live_log.read_text(encoding="utf-8", errors="replace")
            log_path: str | None = str(live_log)
        else:
            completed = runner(argv, capture_output=True, text=True, check=False, **run_kwargs)
            output = f"{getattr(completed, 'stdout', '') or ''}{getattr(completed, 'stderr', '') or ''}"
            log_path = _write_launch_log(spec, output)
        returncode = int(completed.returncode)
        verdict = fs_verdict(output)
        extra = {"log": log_path, "tail": output.splitlines()[-40:], "fs_verdict": verdict}
        if verdict is not None and returncode not in (0, 5, 95, 96):
            # torchrun collapses ANY child failure to exit 1; FS's own verdict line
            # still says which it was (REFUSED 96 / UNMEASURED 95 / RED 5).
            extra["returncode_raw"] = returncode
            returncode = verdict

    out = {
        "job_id": job_id,
        "pid": pid,
        "command": command,
        "dry_run_rc": dry_run_rc,
        "returncode": returncode,
        **extra,
    }
    elapsed = time.monotonic() - started
    _enforce_launch_budget(spec, out, elapsed)
    _ledger_row(spec, out, elapsed)
    return out


def _enforce_launch_budget(spec: dict[str, Any], out: dict[str, Any], seconds: float) -> None:
    """C4.2: a declared ``budget.wall_seconds`` overrun turns a successful local run RED (5); unmeasured when submitted."""
    limit = (spec.get("budget") or {}).get("wall_seconds")
    if limit is None:
        return
    if out.get("returncode") is None:
        out["budget_breach"] = None
        out["budget_note"] = "submitted to the scheduler: wall time is not measured here (unmeasured, never a pass)"
        return
    out["budget_breach"] = seconds > float(limit)
    if out["budget_breach"] and out["returncode"] == 0:
        out["returncode"] = 5
        out["budget_note"] = f"ran {seconds:.1f} s against a declared budget of {float(limit):.1f} s"


def _ledger_row(spec: dict[str, Any], out: dict[str, Any], seconds: float) -> None:
    """C4: one launch row beside the launch log (best effort). GPU-seconds stay null: the GPU count is not measured."""
    from foundationskills.core import ledger
    from foundationskills.core.harness import manifest_hash

    log = _launch_log_path(spec)
    code = out.get("returncode")
    verdicts = {0: "PASS", 5: "RED", 95: "UNMEASURED", 96: "REFUSED"}
    try:
        ledger.append(log.parent if log else None, ledger.record(
            "launch", str(spec.get("entry") or "unknown"), str(spec.get("stage_name") or ""),
            verdict=verdicts.get(code) if code is not None else "SUBMITTED", exit_code=code,
            wall_clock_ms=round(seconds * 1000, 3), request={"argv": spec.get("argv")},
            harness_hash=manifest_hash(spec["harness_manifest"]) if spec.get("harness_manifest") else None,
            extra={"job_id": out.get("job_id")}, null_reasons={"gpu_seconds": "gpu_count_unmeasured"}))
    except Exception:  # noqa: BLE001 - never block or alter a confirmed launch
        pass


_FS_VERDICT = re.compile(r"^\[fs:train:(done|refuse)\]\s*(.*)$", re.M)


def fs_verdict(output: str) -> int | None:
    """Map FS's final [fs:train:*] line to its exit code; None if it printed none."""
    matches = _FS_VERDICT.findall(output or "")
    if not matches:
        return None
    kind, text = matches[-1]
    if kind == "refuse":
        return 96
    upper = text.upper()
    if "UNMEASURED" in upper:
        return 95
    if "PASS" in upper:
        return 0
    return 5


def _write_manifest_sidecar(spec: dict[str, Any]) -> None:
    """C2: ``harness_manifest.json`` + its 16-hex hash beside the launch log, before anything runs (best effort)."""
    import json as _json

    from foundationskills.core.harness import manifest_hash

    log = _launch_log_path(spec)
    if log is None:
        return
    try:
        (log.parent / "harness_manifest.json").write_text(
            _json.dumps(spec["harness_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (log.parent / "harness_manifest.sha256").write_text(manifest_hash(spec["harness_manifest"]) + "\n",
                                                           encoding="utf-8")
    except OSError:
        pass  # an unwritable run directory never blocks a confirmed launch (same rule as the launch log)


def _launch_log_path(spec: dict[str, Any]) -> Path | None:
    """``fskills_launch.log`` next to the run, directory created; None when unwritable."""
    target_dir = spec.get("output_dir") or spec.get("cwd")
    if not target_dir:
        return None
    try:
        path = Path(str(target_dir)) / "fskills_launch.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        return None


def _write_launch_log(spec: dict[str, Any], output: str) -> str | None:
    """Keep the full output next to the run (a failure with no log is undiagnosable)."""
    path = _launch_log_path(spec)
    if path is None:
        return None
    try:
        path.write_text(output, encoding="utf-8")
        return str(path)
    except OSError:
        return None
