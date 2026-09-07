"""The thinnest REAL FoundationScale training path, in one screen.

Runnable single-GPU:        python examples/train_tiny.py
Runnable multi-GPU (DDP):   torchrun --nproc_per_node=N examples/train_tiny.py
                            (set GPUS = N below to match; WORLD_SIZE is checked
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

# EDIT THIS ONE VALUE to match the box you are on: 1 for a laptop or a single
# card, 4 or 8 for a real node. It is the only place the GPU count is written --
# the profile, the declared topology and the DDP degree all read it, so a 4-GPU
# first run cannot half-succeed with three fields disagreeing.
# It is deliberately NOT derived from WORLD_SIZE: train() compares this DECLARED
# topology against the one torchrun actually built, and a declaration copied out
# of the runtime is a comparator that can never disagree with itself.
GPUS = 1

# The cluster is DATA, not code: describe the machine to validate against.
# These values describe a placeholder single-node box -- edit for your estate.
PROFILE = ClusterProfile.from_dict(
    {
        "name": "example-single-node",
        "scheduler": "slurm",
        "partitions": ["batch"],
        # A REGEX, not a Slurm hostlist. Spelling this same range the hostlist
        # way puts a 1-to-0 range inside the character class, which re.compile
        # refuses -- that is how this example shipped broken. (The literal is
        # deliberately not written out: a test greps for it here.)
        "node_pattern": r"compute-0[1-8]",
        "gpus_per_node": GPUS,
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
        # Any dataset exposing a 'text' column: an HF id, or a local .jsonl.
        # The id MUST be namespaced -- datasets>=5 rejects the bare legacy
        # "ag_news" spelling with HfUriError, which is how this example's
        # second wall was found. See docs/TRAINING.md for the offline variant.
        dataset="fancyzhx/ag_news",
        output_dir=Path("out/train_tiny"),
        # Machine facts -- deliberately NO defaults (fail closed):
        nodes=1,
        gpus_per_node=GPUS,
        dp=GPUS,  # pure DDP: one data-parallel rank per GPU
        max_steps=20,
        save_interval=10,  # the save gate adjudicates checkpoint-10 and -20
        # What this run optimises. Stated, not defaulted: the objective gates
        # refuse a run that declares nothing (exit 5 at the first observed step).
        objective="sft",
        profile=PROFILE,
    )
    # train() validates topology against the profile and BLOCKS (exit 5)
    # before an allocation is burned; only then does it import torch and
    # launch transformers.Trainer DDP with the FoundationScaleSaveGate
    # callback attached. Exit codes: 0 PASS / 5 RED / 95 UNMEASURED / 96 REFUSE.
    raise SystemExit(train(cfg))
