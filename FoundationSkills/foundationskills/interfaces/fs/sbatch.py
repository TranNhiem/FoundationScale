
"""Render sbatch text for GB200/Slurm clusters.

Cluster rules are MEASURED (estate facts, never optional decorations):

* every sbatch carries ``--time=10-00:00:00``;
* node ``r01dgx02`` is never used (``--exclude=r01dgx02``);
* on GB200 hardware (``hardware_id`` starting with ``gb200``) a preamble
  probes the IMEX fabric master ``(exec 3<>/dev/tcp/master/8081)`` and aborts
  with exit 96 when refused, because launching while it is refused drains the
  trays;
* the partition default is ``DefMemPerNode=UNLIMITED``, so a job without a
  memory request takes the whole node; a GB200 job using fewer than the
  node's 4 GPUs therefore requests ``--mem-per-gpu`` (default 200G: measured
  ``RealMemory=881343`` MB / 4 GPUs, with headroom) so it does not strand the
  node's other GPUs;
* ``partition`` is rendered when given (otherwise Slurm's default applies);
* Slurm command lines on this estate need a login shell -- the submit call in
  ``launch.py`` therefore wraps ``sbatch`` in ``bash -lc``.

Topology: ``--ntasks-per-node=1`` with ``srun`` running the launcher
(``torchrun`` when world>1, else ``python``); torchrun fans out the per-GPU
ranks, so one Slurm task per node is correct. ``render_sbatch`` asserts the
``launcher`` argument matches ``argv[0]`` so the script and the spec argv
cannot silently disagree.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

_LAUNCHERS = ("torchrun", "python")
_GB200_GPUS_PER_NODE = 4
_GB200_MEM_PER_GPU = "200G"
_MEM_RE = re.compile(r"^[0-9]+[KMGT]?$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
# argv tokens the shell must expand at run time (quoting them would hand
# torchrun the literal text "$MASTER_ADDR" as its rendezvous endpoint)
_RUNTIME_VARS = ("${MASTER_ADDR}",)


def _quote(token: str) -> str:
    if any(var in token for var in _RUNTIME_VARS):
        if '"' in token or "`" in token or "$(" in token:
            raise ValueError(f"refusing to render unsafe runtime token {token!r}")
        return f'"{token}"'  # double quotes: expands ${MASTER_ADDR}, nothing else is special here
    return shlex.quote(token)


def render_sbatch(
    argv: list[str],
    *,
    env: dict[str, str],
    hardware_id: str,
    nodes: int,
    gpus_per_node: int,
    job_name: str,
    log_dir: str,
    launcher: str,
    cwd: str | None = None,
    cpus_per_task: int | None = None,
    partition: str | None = None,
    mem_per_gpu: str | None = None,
) -> str:
    """Return the sbatch script text for one launch.

    ``cwd`` is the FS repo root: FS records code provenance from the launch
    directory, so the script cds there before srun."""
    if not _SAFE_NAME.match(str(job_name)):
        raise ValueError(f"job_name {job_name!r} must match [A-Za-z0-9._-]+")
    if launcher not in _LAUNCHERS:
        raise ValueError(f"launcher must be one of {_LAUNCHERS}, got {launcher!r}")
    if not argv:
        raise ValueError("argv must be non-empty")
    if partition is not None and not _SAFE_NAME.match(str(partition)):
        raise ValueError(f"partition {partition!r} must match [A-Za-z0-9._-]+")
    if mem_per_gpu is None and str(hardware_id).lower().startswith("gb200") and int(gpus_per_node) < _GB200_GPUS_PER_NODE:
        mem_per_gpu = _GB200_MEM_PER_GPU
    if mem_per_gpu is not None and not _MEM_RE.match(str(mem_per_gpu)):
        raise ValueError(f"mem_per_gpu {mem_per_gpu!r} must look like 200G")
    actual = Path(str(argv[0])).name
    if actual != launcher:
        raise ValueError(
            f"launcher {launcher!r} disagrees with argv[0] {argv[0]!r}; the script and the spec argv must name the same launcher"
        )

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={int(nodes)}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gres=gpu:{int(gpus_per_node)}",
        *( [f"#SBATCH --partition={partition}"] if partition else [] ),
        *( [f"#SBATCH --mem-per-gpu={mem_per_gpu}"] if mem_per_gpu else [] ),
        *( [f"#SBATCH --cpus-per-task={int(cpus_per_task)}"] if cpus_per_task else [] ),
        "#SBATCH --time=10-00:00:00",
        "#SBATCH --exclude=r01dgx02",
        f"#SBATCH --output={log_dir}/%x-%j.out",
        "",
        "set -euo pipefail",
        "",
    ]
    for key in sorted(env):
        lines.append(f"export {key}={shlex.quote(str(env[key]))}")
    if env:
        lines.append("")
    lines.append("# Head host for the c10d rendezvous; estate-measured derivation.")
    lines.append('export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)')
    lines.append("")

    if hardware_id.lower().startswith("gb200"):
        lines.extend(
            [
                "# GB200 cluster rule: never launch while the IMEX fabric master is",
                "# refused -- a launch in that state drains the trays.",
                "if ! (exec 3<>/dev/tcp/master/8081) 2>/dev/null; then",
                '  echo "[fskills:launch:refuse] IMEX fabric master refused; launching would drain trays"',
                "  exit 96",
                "fi",
                "",
            ]
        )

    if cwd:
        lines.append("# FS records code provenance from the launch directory (its git commit).")
        lines.append(f"cd {shlex.quote(str(cwd))}")
        lines.append("")
    lines.append("srun " + " ".join(_quote(str(a)) for a in argv))
    lines.append("")
    return "\n".join(lines)


def write_sbatch(path: str | Path, text: str) -> Path:
    """Write sbatch text to ``path`` (parents created) and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
