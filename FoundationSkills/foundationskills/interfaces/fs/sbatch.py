
"""Render sbatch text for GB200/Slurm clusters.

Cluster rules are MEASURED (estate facts, never optional decorations):

* every sbatch carries ``--time=10-00:00:00``;
* node ``r01dgx02`` is never used (``--exclude=r01dgx02``);
* on GB200 hardware (``hardware_id`` starting with ``gb200``) a preamble
  probes the IMEX fabric master ``(exec 3<>/dev/tcp/master/8081)`` and aborts
  with exit 96 when refused, because launching while it is refused drains the
  trays;
* Slurm command lines on this estate need a login shell -- the submit call in
  ``launch.py`` therefore wraps ``sbatch`` in ``bash -lc``.

Topology: ``--ntasks-per-node=1`` with ``srun`` running the launcher
(``torchrun`` when world>1, else ``python``); torchrun fans out the per-GPU
ranks, so one Slurm task per node is correct. ``render_sbatch`` asserts the
``launcher`` argument matches ``argv[0]`` so the script and the spec argv
cannot silently disagree.
"""
from __future__ import annotations

import shlex
from pathlib import Path

_LAUNCHERS = ("torchrun", "python")


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
) -> str:
    """Return the sbatch script text for one launch."""
    if launcher not in _LAUNCHERS:
        raise ValueError(f"launcher must be one of {_LAUNCHERS}, got {launcher!r}")
    if not argv:
        raise ValueError("argv must be non-empty")
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

    lines.append("srun " + shlex.join([str(a) for a in argv]))
    lines.append("")
    return "\n".join(lines)


def write_sbatch(path: str | Path, text: str) -> Path:
    """Write sbatch text to ``path`` (parents created) and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
