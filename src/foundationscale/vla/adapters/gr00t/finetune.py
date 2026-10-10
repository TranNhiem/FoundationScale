"""GR00T fine-tuning, driven through GR00T's own launcher from one FoundationScale spec.

Owned contract #3 of the VLA plan: FoundationScale holds the run declaration and renders it into
the backend's native command; GR00T trains. :data:`PUBLISHED_LIBERO_RECIPE` is NVIDIA's published
LIBERO recipe (Isaac-GR00T examples/LIBERO/README.md + examples/finetune.sh defaults), the run
FoundationScale reproduces. :func:`stage_base_checkpoint` makes the only declared runtime change --
``use_flash_attention`` -- on a symlinked copy, never on the shared weights; :func:`render_command`
produces the ``torchrun`` argv, through :mod:`.ddp_hook` when more than one GPU trains (DeepSpeed
has no aarch64 build). Both deviations are entries in the upstream LEDGER.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = [
    "PUBLISHED_LIBERO_RECIPE",
    "on_gpus",
    "Gr00tFinetuneError",
    "Gr00tFinetuneSpec",
    "render_command",
    "stage_base_checkpoint",
]

# Files of a training checkpoint that a fine-tune from it must not inherit.
_TRAINER_LEFTOVERS = frozenset(
    {"scheduler.pt", "training_args.bin", "zero_to_fp32.py", "trainer_state.json"}
)


class Gr00tFinetuneError(ValueError):
    """A fine-tune declaration GR00T's launcher cannot run as declared."""


@dataclass(frozen=True)
class Gr00tFinetuneSpec:
    """One GR00T fine-tune, in the units of GR00T's own ``launch_finetune.py``."""

    base_model: str
    dataset: str
    embodiment_tag: str
    output_dir: str
    num_gpus: int
    global_batch_size: int
    max_steps: int
    save_steps: int
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 1e-5
    dataloader_num_workers: int = 4
    shard_size: int = 1024
    num_shards_per_epoch: int = 100000
    episode_sampling_rate: float = 0.1
    color_jitter: tuple[tuple[str, float], ...] = (
        ("brightness", 0.3),
        ("contrast", 0.4),
        ("saturation", 0.5),
        ("hue", 0.08),
    )
    save_total_limit: int = 3

    def __post_init__(self) -> None:
        for name in ("base_model", "dataset", "embodiment_tag", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise Gr00tFinetuneError(f"{name} is {value!r}; expected a non-empty string")
        for name in (
            "num_gpus",
            "global_batch_size",
            "max_steps",
            "save_steps",
            "save_total_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise Gr00tFinetuneError(f"{name} is {value!r}; expected an int >= 1")
        if self.global_batch_size % self.num_gpus:
            raise Gr00tFinetuneError(
                f"global_batch_size {self.global_batch_size} is not divisible by num_gpus "
                f"{self.num_gpus}; GR00T splits the global batch evenly, so the per-GPU batch "
                "would not be the declared one"
            )

    @property
    def per_gpu_batch(self) -> int:
        return self.global_batch_size // self.num_gpus


# NVIDIA's published LIBERO recipe: 8 GPUs x 80 = 640 global, 20K steps, save every 1K, the
# finetune.sh data defaults; state dropout 0.2 is GR00T's finetune-config default.
PUBLISHED_LIBERO_RECIPE = Gr00tFinetuneSpec(
    base_model="nvidia/GR00T-N1.7-3B",
    dataset="IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
    embodiment_tag="LIBERO_PANDA",
    output_dir="libero_spatial",
    num_gpus=8,
    global_batch_size=640,
    max_steps=20000,
    save_steps=1000,
)


def stage_base_checkpoint(
    base_dir: str | os.PathLike[str],
    staging_dir: str | os.PathLike[str],
    *,
    use_flash_attention: bool,
) -> Path:
    """A fine-tunable copy of ``base_dir``: weights symlinked, ``config.json`` patched in a copy.

    The shared weights are never written. Trainer leftovers of the base run (scheduler, trainer
    state, ...) are not carried over, so the fine-tune starts its own schedule. The only config
    field changed is ``use_flash_attention`` (flash-attn has no aarch64 build: upstream LEDGER
    ``gr00t-flash-attn-off-aarch64``). Refuses a base without ``config.json``.
    """
    base = Path(base_dir)
    config = base / "config.json"
    if not config.is_file():
        raise Gr00tFinetuneError(f"{base} has no config.json; expected a GR00T checkpoint")
    stage = Path(staging_dir)
    stage.mkdir(parents=True, exist_ok=True)
    for item in sorted(base.iterdir()):
        if item.name in _TRAINER_LEFTOVERS or item.name == "config.json":
            continue
        target = stage / item.name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(item.resolve())
    data = json.loads(config.read_text(encoding="utf-8"))
    data["use_flash_attention"] = bool(use_flash_attention)
    (stage / "config.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return stage


def render_command(
    spec: Gr00tFinetuneSpec, *, torchrun: str, master_port: int = 29500
) -> list[str]:
    """GR00T's own launcher as a ``torchrun`` argv; multi-GPU runs go through :mod:`.ddp_hook`."""
    entry = (
        ["-m", "foundationscale.vla.adapters.gr00t.ddp_hook"]
        if spec.num_gpus > 1
        else ["gr00t/experiment/launch_finetune.py"]
    )
    jitter = [str(part) for pair in spec.color_jitter for part in pair]
    return [
        torchrun,
        f"--nproc_per_node={spec.num_gpus}",
        f"--master_port={master_port}",
        *entry,
        "--base_model_path",
        spec.base_model,
        "--dataset_path",
        spec.dataset,
        "--embodiment_tag",
        spec.embodiment_tag,
        "--num_gpus",
        str(spec.num_gpus),
        "--output_dir",
        spec.output_dir,
        "--save_steps",
        str(spec.save_steps),
        "--save_total_limit",
        str(spec.save_total_limit),
        "--max_steps",
        str(spec.max_steps),
        "--warmup_ratio",
        str(spec.warmup_ratio),
        "--weight_decay",
        str(spec.weight_decay),
        "--learning_rate",
        str(spec.learning_rate),
        "--global_batch_size",
        str(spec.global_batch_size),
        "--dataloader_num_workers",
        str(spec.dataloader_num_workers),
        "--shard_size",
        str(spec.shard_size),
        "--num_shards_per_epoch",
        str(spec.num_shards_per_epoch),
        "--episode_sampling_rate",
        str(spec.episode_sampling_rate),
        "--color_jitter_params",
        *jitter,
    ]


def on_gpus(spec: Gr00tFinetuneSpec, num_gpus: int) -> Gr00tFinetuneSpec:
    """``spec`` on ``num_gpus`` GPUs with the SAME global batch (so the same optimisation)."""
    return replace(spec, num_gpus=num_gpus)
