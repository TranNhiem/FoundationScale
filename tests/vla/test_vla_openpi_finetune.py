"""openpi fine-tune spec: the published recipe and the ``scripts/train.py`` argv it renders."""

from __future__ import annotations

import dataclasses

import pytest

from foundationscale.vla.adapters.openpi.finetune import (
    PUBLISHED_LIBERO_RECIPE,
    OpenPiFinetuneError,
    render_command,
    render_env,
)


def test_published_recipe_is_openpis_pi05_libero_config() -> None:
    spec = PUBLISHED_LIBERO_RECIPE
    assert (spec.config_name, spec.batch_size, spec.num_train_steps) == ("pi05_libero", 256, 30_000)


def test_render_command_is_openpis_train_script_with_the_declared_overrides() -> None:
    spec = dataclasses.replace(
        PUBLISHED_LIBERO_RECIPE, checkpoint_base_dir="/runs/ckpt", assets_base_dir="/runs/assets"
    )
    argv = render_command(spec, python="py")
    assert argv[:3] == ["py", "scripts/train.py", "pi05_libero"]
    flags = dict(zip(argv[3:-2:2], argv[4:-2:2], strict=True))
    assert flags == {
        "--exp-name": "fs_repro",
        "--checkpoint-base-dir": "/runs/ckpt",
        "--assets-base-dir": "/runs/assets",
        "--batch-size": "256",
        "--num-train-steps": "30000",
        "--save-interval": "1000",
        "--keep-period": "5000",
        "--log-interval": "100",
        "--fsdp-devices": "1",
    }
    assert argv[-2:] == ["--overwrite", "--no-wandb-enabled"]


def test_resume_continues_the_run_instead_of_overwriting_it() -> None:
    spec = dataclasses.replace(PUBLISHED_LIBERO_RECIPE, resume=True, wandb_enabled=True)
    assert render_command(spec, python="py")[-2:] == ["--resume", "--wandb-enabled"]


def test_render_env_pins_jax_preallocation() -> None:
    assert render_env(PUBLISHED_LIBERO_RECIPE) == {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"}


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": 0},
        {"num_train_steps": -1},
        {"save_interval": 1.5},
        {"fsdp_devices": True},
        {"exp_name": " "},
        {"batch_size": 255, "fsdp_devices": 2},
        {"xla_mem_fraction": 0.0},
        {"xla_mem_fraction": 1.5},
    ],
)
def test_a_spec_openpi_cannot_run_is_refused(change: dict[str, object]) -> None:
    with pytest.raises(OpenPiFinetuneError):
        dataclasses.replace(PUBLISHED_LIBERO_RECIPE, **change)  # type: ignore[arg-type]
