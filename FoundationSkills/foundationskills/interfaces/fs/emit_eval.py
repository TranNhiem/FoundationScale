"""Emit an ``fs_launch_spec`` payload for an eval job (entry ``fskills-eval``).

The job runs ``python -m foundationskills.cli eval run`` -- the same skill, the
same refusal surface -- on ONE node: lm-eval's HF backend is a single process,
so more than one GPU means ``parallelize=True`` (the model is sharded across the
node's GPUs), never torchrun. Multi-node eval is refused, not approximated.

Executability is measured, never assumed: the eval request goes through the
skill's own ``check_inputs`` at emit time, with lm_eval's version read from the
interpreter the job will actually run (``--python``), not from this process.
``dry_run_argv`` repeats that check at launch time (``eval run --dry-run``), so
an input that vanished between emit and launch still blocks the submission.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from foundationskills.interfaces.fs.sbatch import render_sbatch

__all__ = ("emit_eval", "lm_eval_version_of")

_PROBE = "import importlib.metadata as m\ntry:\n print(m.version('lm_eval'))\nexcept m.PackageNotFoundError:\n print('')"
# request key -> eval run flag; booleans are bare flags
_FLAGS = {
    "checkpoint": "--checkpoint", "base": "--base", "run_manifest": "--run-manifest", "policy": "--policy",
    "eval_cache": "--eval-cache", "out": "--out", "baseline_cache": "--baseline-cache",
    "num_fewshot": "--num-fewshot", "seed": "--seed", "gen_kwargs": "--gen-kwargs", "limit": "--limit",
    "dtype": "--dtype", "batch_size": "--batch-size", "device": "--device", "include_path": "--include-path",
}


def lm_eval_version_of(python: str) -> str | None:
    """lm_eval version installed for ``python`` (None if absent or the interpreter cannot run)."""
    try:
        done = subprocess.run([python, "-c", _PROBE], capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (done.stdout.strip() or None) if done.returncode == 0 else None


def _skills_root() -> str:
    import foundationskills

    return str(Path(foundationskills.__file__).resolve().parent.parent)


def _argv(python: str, request: dict[str, Any]) -> list[str]:
    argv = [python, "-m", "foundationskills.cli", "eval", "run", "--benchmarks", ",".join(request["benchmarks"])]
    for key, flag in _FLAGS.items():
        if request.get(key) is not None:
            argv += [flag, str(request[key])]
    if request.get("parallelize"):
        argv.append("--parallelize")
    return argv


def emit_eval(
    request: dict[str, Any],
    *,
    nodes: int = 1,
    gpus_per_node: int = 1,
    hardware_id: str = "gb200",
    python: str = "python",
    run_name: str = "fskills-eval",
    version_probe=None,
) -> dict[str, Any]:
    """Build one fs_launch_spec for an eval job; ``executable`` only on positive evidence."""
    from foundationskills.core.contract import SkillContext
    from foundationskills.skills.evaluation.skill import EvalSkill

    request = dict(request)
    missing: list[str] = []
    notes: list[str] = []
    if int(nodes) != 1:
        missing.append(f"precondition failed: eval runs on one node (lm-eval HF backend is one process); got nodes={nodes}")
    if int(gpus_per_node) < 1:
        missing.append(f"precondition failed: gpus_per_node must be >= 1, got {gpus_per_node}")
    elif int(gpus_per_node) > 1:
        if request.get("device"):
            notes.append(f"device {request['device']!r} dropped: {gpus_per_node} GPUs shard the model via parallelize")
            request.pop("device")
        request["parallelize"] = True
    elif not request.get("parallelize"):
        request.setdefault("device", "cuda:0")

    out = Path(request["out"]).expanduser().resolve()
    request["out"] = str(out)
    output_dir = str(out.parent)
    probe = version_probe or lm_eval_version_of
    skill = EvalSkill(version_probe=lambda: probe(python))
    findings = skill.check_inputs(request, SkillContext(workdir=Path(output_dir)))
    missing += [f"{f.rule_id}: {f.message}" for f in findings if f.severity.name == "BLOCK"]

    argv = _argv(python, request)
    root = _skills_root()
    env = {"PYTHONPATH": root}
    sbatch = None
    if hardware_id.lower() != "local":
        sbatch = render_sbatch(argv, env=env, hardware_id=hardware_id, nodes=1, gpus_per_node=max(int(gpus_per_node), 1),
                               job_name=run_name, log_dir=output_dir, launcher="python", cwd=root)
    executable = not missing
    return {
        "stage_name": "eval",
        "entry": "fskills-eval",
        "argv": argv,
        "env": env,
        "sbatch": sbatch,
        "dry_run_argv": [*argv, "--dry-run"],
        "expected_outputs": [str(out)],
        "executable": executable,
        "missing": None if executable else "; ".join(missing),
        "output_dir": output_dir,
        "run_name": run_name,
        "cwd": root,
        "eval_request": request,
        "notes": notes,
        "warnings": [],
    }
