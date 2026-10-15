"""verl-vla RL spec: transcribed recipes, GPU placement, runtime env, rendered trainer argv."""

from __future__ import annotations

import dataclasses

import pytest

from foundationscale.vla.adapters.verlvla import (
    PUBLISHED_RECIPES,
    PublishedCurve,
    VerlVlaRlError,
    VerlVlaRlSpec,
    on_gpus,
    render_command,
    with_output_dir,
    with_runtime_env,
)

_ENV = "ray_kwargs.ray_init.runtime_env.env_vars"


def test_every_recipe_carries_its_published_curve_and_model() -> None:
    assert set(PUBLISHED_RECIPES) == {
        "fpo-pi05-libero-spatial-task2",
        "dsrl-pi05-libero-spatial-task2",
        "dsrl-pi05-libero-spatial-task9",
        "td3bc-pi05-libero-spatial-task2",
        "td3bc-gaussian-libero-spatial-task0",
    }
    for name, spec in PUBLISHED_RECIPES.items():
        assert spec.published is not None and spec.published.trials == 50, name
        assert spec.value("cluster.actor_rollout_ref.model.path").startswith("Miical/"), name
        assert spec.value("output_dir").startswith("./outputs/rl/"), name


@pytest.mark.parametrize(
    ("name", "start", "end", "best"),
    [
        ("fpo-pi05-libero-spatial-task2", 29, 50, 50),
        ("dsrl-pi05-libero-spatial-task2", 37, 44, 44),
        ("dsrl-pi05-libero-spatial-task9", 30, 37, 41),
        ("td3bc-pi05-libero-spatial-task2", 32, 40, 40),
        ("td3bc-gaussian-libero-spatial-task0", 2, 48, 48),
    ],
)
def test_published_counts_are_the_docs(name: str, start: int, end: int, best: int) -> None:
    curve = PUBLISHED_RECIPES[name].published
    assert curve is not None and (curve.start, curve.end, curve.best) == (start, end, best)


def test_dsrl_tasks_differ_only_in_task_noise_and_names() -> None:
    a = dict(PUBLISHED_RECIPES["dsrl-pi05-libero-spatial-task2"].overrides)
    b = dict(PUBLISHED_RECIPES["dsrl-pi05-libero-spatial-task9"].overrides)
    differ = {k for k in a if a[k] != b[k]}
    assert differ == {
        "output_dir",
        "cluster.env.env_worker.simulator.libero.task_ids",
        "cluster.actor_rollout_ref.model.adapter.dsrl.rollout_noise_scale",
        "trainer.experiment_name",
    }


def test_on_gpus_caps_model_gpu_counts_and_keeps_smaller_ones() -> None:
    fpo = on_gpus(PUBLISHED_RECIPES["fpo-pi05-libero-spatial-task2"], 2)
    assert fpo.value("cluster.resource.model.gpus_per_node") == "2"
    assert fpo.value("cluster.resource.model.workers_per_node") == "2"
    dsrl = on_gpus(PUBLISHED_RECIPES["dsrl-pi05-libero-spatial-task2"], 2)
    assert (
        dsrl.value("cluster.resource.model.workers_per_node") == "1"
    )  # one actor worker stays one
    gaussian = on_gpus(PUBLISHED_RECIPES["td3bc-gaussian-libero-spatial-task0"], 4)
    assert gaussian.value("cluster.resource.model.gpus_per_node") == "1"  # never inflated
    # env placement is the recipe's, untouched
    assert fpo.value("cluster.resource.env.workers_per_node") == "4"


def test_output_dir_moves_tensorboard_with_it() -> None:
    spec = with_output_dir(PUBLISHED_RECIPES["td3bc-pi05-libero-spatial-task2"], "/runs/td3")
    assert spec.value("output_dir") == "/runs/td3"
    assert spec.value(f"{_ENV}.TENSORBOARD_DIR") == "/runs/td3/tensorboard"


def test_runtime_env_overwrites_known_keys_and_adds_new_ones_with_plus() -> None:
    spec = with_runtime_env(
        PUBLISHED_RECIPES["fpo-pi05-libero-spatial-task2"],
        {"MUJOCO_GL": "egl", "HF_HUB_CACHE": "/hub"},
    )
    keys = [k for k, _ in spec.overrides]
    assert f"{_ENV}.MUJOCO_GL" in keys and spec.value(f"{_ENV}.MUJOCO_GL") == "egl"
    # ++ (add or override): the recipe YAML may already define a key its command line does not
    assert f"++{_ENV}.HF_HUB_CACHE" in keys and spec.value(f"{_ENV}.HF_HUB_CACHE") == "/hub"
    # applying it twice changes nothing (the ++ key is now known)
    assert with_runtime_env(spec, {"HF_HUB_CACHE": "/hub"}) == spec


def test_render_command_is_the_trainer_with_config_and_ordered_overrides() -> None:
    spec = PUBLISHED_RECIPES["dsrl-pi05-libero-spatial-task9"]
    argv = render_command(spec, extra=(("trainer.total_training_steps", "1000"),))
    assert argv[:5] == [
        "vvla-train-sac",
        "--config-dir",
        "./examples/rl/dsrl/pi05/libero_spatial_online_from_sft_step100",
        "--config-name",
        "dsrl",
    ]
    assert "cluster.env.env_worker.simulator.libero.task_ids=[9]" in argv
    assert argv[-1] == "trainer.total_training_steps=1000"
    assert (
        render_command(spec, extra=(("output_dir", "/x"),))[5] == "output_dir=/x"
    )  # replaced in place


@pytest.mark.parametrize(
    "change",
    [
        {"entrypoint": "vvla-train-ppo"},
        {"config_name": " "},
        {"overrides": (("a", "1"), ("+a", "2"))},
        {"gpu_keys": ("cluster.not.there",)},
    ],
)
def test_a_spec_verl_vla_cannot_run_is_refused(change: dict[str, object]) -> None:
    with pytest.raises(VerlVlaRlError):
        dataclasses.replace(PUBLISHED_RECIPES["fpo-pi05-libero-spatial-task2"], **change)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "args",
    [(0, 0, 0, 0, "s"), (50, 51, 0, 51, "s"), (50, 10, 20, 15, "s"), (50, 1.0, 2, 2, "s")],
)
def test_an_impossible_published_curve_is_refused(args: tuple[object, ...]) -> None:
    with pytest.raises(VerlVlaRlError):
        PublishedCurve(*args)  # type: ignore[arg-type]


def test_bad_placement_arguments_are_refused() -> None:
    spec = PUBLISHED_RECIPES["fpo-pi05-libero-spatial-task2"]
    with pytest.raises(VerlVlaRlError):
        on_gpus(spec, 0)
    with pytest.raises(VerlVlaRlError):
        with_output_dir(spec, " ")
    with pytest.raises(VerlVlaRlError):
        spec.value("no.such.key")
    assert isinstance(spec, VerlVlaRlSpec)
