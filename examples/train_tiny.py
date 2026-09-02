"""The thinnest REAL FoundationScale training path, in one screen.

Runnable single-GPU:        python examples/train_tiny.py
Runnable multi-GPU (DDP):   torchrun --nproc_per_node=N python examples/train_tiny.py
                            (set dp=N below to match N; WORLD_SIZE is checked
                            against the declared topology BEFORE any GPU is used)
Validate only, zero GPUs:   set dry_run=True, or use
                            `foundationscale-train ... --dry-run`

Requires the optional extra: pip install 'foundationscale[train]'
The package itself stays torch-free; this example is where torch enters.
"""
from __future__ import annotations

from pathlib import Path

from foundationscale.topology import ClusterProfile
from foundationscale.train.loop import TrainConfig, train

# The cluster is DATA, not code: describe the machine to validate against.
# These values describe a placeholder single-GPU box -- edit for your estate.
PROFILE = ClusterProfile.from_dict(
    {
        "name": "example-single-node",
        "scheduler": "slurm",
        "partitions": ["batch"],
        "node_pattern": "compute-[01-08]",
        "gpus_per_node": 1,
        "nccl_socket_ifname": "eth0",
        "ib_hca_pattern": "mlx5_*",
        "mnnvl_available": False,
        "container_runtime": "none",
        "container_image": "",
        "filesystem_roots": ["/tmp"],
        "max_nodes": 1,
    }
)

if __name__ == "__main__":
    cfg = TrainConfig(
        # Any public HF causal-LM id works; NO model name lives in core code.
        model="sshleifer/tiny-gpt2",  # tiny PUBLIC model, downloads in seconds
        dataset="ag_news",  # tiny PUBLIC dataset exposing a 'text' column
        output_dir=Path("out/train_tiny"),
        # Machine facts -- deliberately NO defaults (fail closed):
        nodes=1,
        gpus_per_node=1,
        dp=1,  # under torchrun --nproc_per_node=N, set dp=N
        max_steps=20,
        save_interval=10,  # the save gate adjudicates checkpoint-10 and -20
        profile=PROFILE,
    )
    # train() validates topology against the profile and BLOCKS (exit 5)
    # before an allocation is burned; only then does it import torch and
    # launch transformers.Trainer DDP with the FoundationScaleSaveGate
    # callback attached. Exit codes: 0 PASS / 5 RED / 95 UNMEASURED / 96 REFUSE.
    raise SystemExit(train(cfg))
