"""openpi fine-tuning through openpi's own ``scripts/train.py``, from one FoundationScale spec.

Owned contract #3 of the VLA plan, as for GR00T: FoundationScale holds the run declaration and
renders it into the backend's native command; openpi (JAX) trains. :data:`PUBLISHED_LIBERO_RECIPE`
is openpi's ``pi05_libero`` train config (``src/openpi/training/config.py``: batch 256, 30k steps,
from ``pi05_base``), the run whose checkpoint openpi publishes as ``pi05_libero``. The recipe's
optimizer, schedule, EMA and data transforms live in that named config and are not restated here:
the spec overrides only run-placement fields (where checkpoints and norm stats go, how often to
save) plus the two numbers a reproduction must pin, ``batch_size`` and ``num_train_steps``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PUBLISHED_LIBERO_RECIPE",
    "OpenPiFinetuneError",
    "OpenPiFinetuneSpec",
    "render_command",
    "render_env",
]


class OpenPiFinetuneError(ValueError):
    """A fine-tune declaration openpi's ``train.py`` cannot run as declared."""


@dataclass(frozen=True)
class OpenPiFinetuneSpec:
    """One openpi fine-tune, in the units of openpi's own ``TrainConfig``."""

    config_name: str
    exp_name: str
    checkpoint_base_dir: str
    assets_base_dir: str
    batch_size: int
    num_train_steps: int
    save_interval: int = 1000
    keep_period: int = 5000
    log_interval: int = 100
    fsdp_devices: int = 1
    resume: bool = False
    wandb_enabled: bool = False
    # JAX preallocates this share of each GPU; openpi's docs run training at 0.9
    xla_mem_fraction: float = 0.9

    def __post_init__(self) -> None:
        for name in ("config_name", "exp_name", "checkpoint_base_dir", "assets_base_dir"):
            if not str(getattr(self, name)).strip():
                raise OpenPiFinetuneError(f"{name} is empty; openpi needs it to place the run")
        for name in (
            "batch_size",
            "num_train_steps",
            "save_interval",
            "keep_period",
            "log_interval",
            "fsdp_devices",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise OpenPiFinetuneError(f"{name} is {value!r}; expected an int >= 1")
        if self.batch_size % self.fsdp_devices:
            raise OpenPiFinetuneError(
                f"batch_size {self.batch_size} does not split over fsdp_devices {self.fsdp_devices}"
            )
        if not 0.0 < self.xla_mem_fraction <= 1.0:
            raise OpenPiFinetuneError(
                f"xla_mem_fraction is {self.xla_mem_fraction!r}; expected a share in (0, 1]"
            )


# openpi src/openpi/training/config.py ``pi05_libero``: batch_size=256, num_train_steps=30_000.
PUBLISHED_LIBERO_RECIPE = OpenPiFinetuneSpec(
    config_name="pi05_libero",
    exp_name="fs_repro",
    checkpoint_base_dir="./checkpoints",
    assets_base_dir="./assets",
    batch_size=256,
    num_train_steps=30_000,
)


def render_command(spec: OpenPiFinetuneSpec, *, python: str) -> list[str]:
    """openpi's ``scripts/train.py`` argv (run from the openpi checkout).

    ``--resume`` continues the run directory; otherwise ``--overwrite`` starts it fresh, so a
    rerun never silently mixes two runs' checkpoints.
    """
    return [
        python,
        "scripts/train.py",
        spec.config_name,
        "--exp-name",
        spec.exp_name,
        "--checkpoint-base-dir",
        spec.checkpoint_base_dir,
        "--assets-base-dir",
        spec.assets_base_dir,
        "--batch-size",
        str(spec.batch_size),
        "--num-train-steps",
        str(spec.num_train_steps),
        "--save-interval",
        str(spec.save_interval),
        "--keep-period",
        str(spec.keep_period),
        "--log-interval",
        str(spec.log_interval),
        "--fsdp-devices",
        str(spec.fsdp_devices),
        "--resume" if spec.resume else "--overwrite",
        "--wandb-enabled" if spec.wandb_enabled else "--no-wandb-enabled",
    ]


def render_env(spec: OpenPiFinetuneSpec) -> dict[str, str]:
    """The environment ``train.py`` runs under (on top of the caller's)."""
    return {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(spec.xla_mem_fraction)}
