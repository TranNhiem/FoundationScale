
"""Tests for render_sbatch: cluster rules, IMEX preamble scoping, writing."""
from __future__ import annotations

import pytest

from foundationskills.interfaces.fs.sbatch import render_sbatch, write_sbatch

TARGV = [
    "torchrun", "--nnodes", "1", "--nproc-per-node", "4",
    "--rdzv-backend", "c10d", "--rdzv-endpoint", "$MASTER_ADDR:29500",
    "-m", "foundationscale.train.cli", "--model", "m",
]


def render(hardware_id="gb200-tray", launcher="torchrun", argv=None):
    return render_sbatch(
        argv if argv is not None else TARGV,
        env={"HF_HOME": "/data/.hf", "ODD": "a b'c"},
        hardware_id=hardware_id,
        nodes=1,
        gpus_per_node=4,
        job_name="fskills-sft1",
        log_dir="logs",
        launcher=launcher,
    )


def test_cluster_rules_and_structure():
    text = render()
    assert "#SBATCH --time=10-00:00:00" in text
    assert "#SBATCH --exclude=r01dgx02" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --gres=gpu:4" in text
    assert "#SBATCH --job-name=fskills-sft1" in text
    assert "#SBATCH --output=logs/%x-%j.out" in text
    assert "set -euo pipefail" in text
    assert 'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)' in text
    assert "srun torchrun" in text
    # env values are shlex-quoted (only values needing it)
    assert "export HF_HOME=/data/.hf" in text
    assert "export ODD='a b'\"'\"'c'" in text


def test_imex_preamble_gb200_only():
    gb200 = render(hardware_id="gb200-tray")
    assert "(exec 3<>/dev/tcp/master/8081)" in gb200
    assert '[fskills:launch:refuse] IMEX fabric master refused; launching would drain trays' in gb200
    assert "exit 96" in gb200
    h100 = render(hardware_id="dgx-h100")
    assert "IMEX" not in h100
    # the standing time/exclude rules apply regardless of hardware
    assert "#SBATCH --time=10-00:00:00" in h100
    assert "#SBATCH --exclude=r01dgx02" in h100


def test_launcher_must_match_argv():
    with pytest.raises(ValueError):
        render(launcher="python")  # TARGV starts with torchrun
    with pytest.raises(ValueError):
        render(launcher="torchrun", argv=["python", "-m", "foundationscale.train.cli"])
    ok = render_sbatch(
        ["python", "-m", "foundationscale.train.cli", "--model", "m"],
        env={}, hardware_id="gb200-tray", nodes=1, gpus_per_node=1,
        job_name="j", log_dir="logs", launcher="python",
    )
    assert "srun python -m foundationscale.train.cli" in ok


def test_write_sbatch(tmp_path):
    target = tmp_path / "sub" / "job.sbatch"
    returned = write_sbatch(target, "#!/bin/bash\ntrue\n")
    assert returned == target
    assert target.read_text(encoding="utf-8").startswith("#!/bin/bash")
