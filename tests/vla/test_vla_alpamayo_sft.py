"""alpamayo2_sft spec: published recipes, chunk split, placement, rendered launcher argv."""

from __future__ import annotations

import dataclasses

import pytest

from foundationscale.vla.adapters.alpamayo.sft import (
    PUBLISHED_RECIPES,
    AlpamayoSftError,
    AlpamayoSftRecipe,
    AlpamayoSftSpec,
    PublishedLosses,
    PublishedMetric,
)

_TRAIN = AlpamayoSftSpec(
    config_name="sft_stage1_single_node_debug",
    model="nvidia/Alpamayo2-Super",
    pai_dir="/data/PhysicalAI-Autonomous-Vehicles",
    train_chunks="0-1",
    val_chunks="99-100",
    gpus=(0, 1, 2, 3, 4, 5, 6, 7),
)


def test_every_recipe_matches_the_readme_and_its_entry_module() -> None:
    assert set(PUBLISHED_RECIPES) == {
        "sft_stage1_single_node_debug",
        "sft_stage2_single_node_debug",
        "sft_eval_loss",
        "sft_eval_trajectory",
    }
    for name, recipe in PUBLISHED_RECIPES.items():
        assert recipe.config_name == name
        assert recipe.multi_process == (name != "sft_eval_trajectory"), name
        assert recipe.published, name


@pytest.mark.parametrize(
    ("name", "metric", "value"),
    [
        ("sft_eval_loss", "eval_loss", 1.699801),
        ("sft_eval_trajectory", "eval_min_ade", 0.8288),
    ],
)
def test_published_evaluation_metrics_are_the_readme(name: str, metric: str, value: float) -> None:
    published = PUBLISHED_RECIPES[name].published[metric]
    assert isinstance(published, PublishedMetric)
    assert (published.metric, published.value) == (metric, value)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("sft_stage1_single_node_debug", (1.8095, 1.7255)),
        ("sft_stage2_single_node_debug", (1.2710, 0.9367)),
    ],
)
def test_two_step_smoke_losses_are_the_readme(name: str, values: tuple[float, ...]) -> None:
    published = PUBLISHED_RECIPES[name].published["loss"]
    assert isinstance(published, PublishedLosses) and published.values == values


def test_render_command_is_the_train_hf_launcher_with_the_readme_overrides() -> None:
    assert (_TRAIN.with_output_dir("/runs/stage1")).render_command() == [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "alpamayo2_sft.train_hf",
        "--config-path",
        "pkg://alpamayo2_sft/configs",
        "--config-name",
        "sft_stage1_single_node_debug",
        "model.pretrained_model_name_or_path=nvidia/Alpamayo2-Super",
        "data.train_dataset.local_dir=/data/PhysicalAI-Autonomous-Vehicles",
        "data.train_dataset.chunk_ids=0-1",
        "data.val_dataset.local_dir=/data/PhysicalAI-Autonomous-Vehicles",
        "data.val_dataset.chunk_ids=99-100",
        "paths.output_dir=/runs/stage1",
        "hydra.sweep.subdir=0",
        "hydra.hydra_help.template=hydra_help",
        "hydra.job_logging.handlers.file.filename=/runs/stage1/train_hf.log",
    ]


def test_render_command_is_python_for_the_single_process_trajectory_eval() -> None:
    spec = AlpamayoSftSpec(
        config_name="sft_eval_trajectory",
        model="nvidia/Alpamayo2-Super",
        pai_dir="$PAI_DIR",
        val_chunks="99-100",
        gpus=(0,),
    )
    assert spec.render_command() == [
        "python",
        "-m",
        "alpamayo2_sft.evaluate_hf",
        "--config-path",
        "pkg://alpamayo2_sft/configs",
        "--config-name",
        "sft_eval_trajectory",
        "evaluate.eval_ckpt=nvidia/Alpamayo2-Super",
        "data.val_dataset.local_dir=$PAI_DIR",
        "data.val_dataset.chunk_ids=99-100",
        "evaluate.max_eval_steps=100",
    ]


def test_render_command_exposes_val_only_and_train_and_eval_keys() -> None:
    loss = AlpamayoSftSpec(
        config_name="sft_eval_loss",
        model="checkpoint-500",
        pai_dir="/data/pai",
        val_chunks="99-100",
        gpus=(0, 1),
    )
    argv = loss.render_command()
    assert argv[:2] == ["torchrun", "--standalone"]
    assert argv[2] == "--nproc_per_node=2"
    assert "evaluate.eval_ckpt=checkpoint-500" in argv
    assert "data.train_dataset.local_dir=/data/pai" not in argv
    assert "paths.output_dir" not in "".join(argv)
    assert loss.render_command() == loss.render_command()


def test_render_env_exposes_the_gpus_in_order() -> None:
    assert _TRAIN.render_env() == {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    bare = AlpamayoSftSpec(config_name="sft_eval_trajectory", model="m", pai_dir="/data/pai")
    assert bare.render_env() == {}
    assert bare.render_command()[:3] == ["python", "-m", "alpamayo2_sft.evaluate_hf"]


def test_on_gpus_and_with_output_dir_return_new_specs() -> None:
    few = _TRAIN.on_gpus((0, 1))
    assert few.gpus == (0, 1) and _TRAIN.gpus == (0, 1, 2, 3, 4, 5, 6, 7)
    out = _TRAIN.with_output_dir("/runs/x")
    assert out.output_dir == "/runs/x" and _TRAIN.output_dir is None
    assert isinstance(few, AlpamayoSftSpec) and isinstance(out, AlpamayoSftSpec)


def test_stage2_the_stage1_checkpoint_travels() -> None:
    spec = AlpamayoSftSpec(
        config_name="sft_stage2_single_node_debug",
        model="outputs/stage1/checkpoint-500",
        pai_dir="/data/pai",
        train_chunks="0-1",
        val_chunks="99-100",
        gpus=(0,),
        extra_overrides=("model.stage1_vlm_checkpoint_path=outputs/stage1/checkpoint-500",),
    )
    argv = (spec.with_output_dir("/runs/stage2")).render_command()
    assert argv[8] == "sft_stage2_single_node_debug"
    assert argv[-1] == "model.stage1_vlm_checkpoint_path=outputs/stage1/checkpoint-500"
    assert "paths.output_dir=/runs/stage2" in argv and argv[-2].endswith(
        "/runs/stage2/train_hf.log"
    )


@pytest.mark.parametrize("config_name", ["sft_stage1", "", "unknown"])
def test_an_unknown_recipe_is_refused(config_name: str) -> None:
    with pytest.raises(AlpamayoSftError):
        AlpamayoSftSpec(
            config_name=config_name,
            model="nvidia/Alpamayo2-Super",
            pai_dir="/data/pai",
            gpus=(0,),
        )


@pytest.mark.parametrize("model", ["", "  "])
def test_an_empty_model_is_refused(model: str) -> None:
    with pytest.raises(AlpamayoSftError):
        AlpamayoSftSpec(config_name="sft_eval_loss", model=model, pai_dir="/data/pai", gpus=(0,))


@pytest.mark.parametrize("pai_dir", ["", "  "])
def test_an_empty_pai_dir_is_refused(pai_dir: str) -> None:
    with pytest.raises(AlpamayoSftError):
        AlpamayoSftSpec(config_name="sft_eval_loss", model="m", pai_dir=pai_dir, gpus=(0,))


@pytest.mark.parametrize("train_chunks", ["0", "0-0", "9-0", "0-1x", " 0-1", "0-1-2", ""])
def test_a_chunk_range_that_is_not_half_open_is_refused(train_chunks: str) -> None:
    with pytest.raises(AlpamayoSftError):
        dataclasses.replace(_TRAIN, train_chunks=train_chunks)


@pytest.mark.parametrize("val_chunks", ["99", "100-99", "0-100"])
def test_a_bad_or_training_val_chunk_range_is_refused(val_chunks: str) -> None:
    with pytest.raises(AlpamayoSftError):
        dataclasses.replace(_TRAIN, val_chunks=val_chunks)


@pytest.mark.parametrize("train_chunks", ["0-100", "5-100", "98-100"])
def test_train_chunks_overlapping_val_chunks_are_refused(train_chunks: str) -> None:
    with pytest.raises(AlpamayoSftError):
        dataclasses.replace(_TRAIN, train_chunks=train_chunks)


def test_a_single_process_recipe_on_several_gpus_is_refused() -> None:
    with pytest.raises(AlpamayoSftError):
        AlpamayoSftSpec(
            config_name="sft_eval_trajectory",
            model="m",
            pai_dir="/data/pai",
            gpus=(0, 1),
        )


def test_a_multi_process_recipe_without_gpus_is_refused() -> None:
    with pytest.raises(AlpamayoSftError):
        dataclasses.replace(_TRAIN, gpus=())


def test_training_without_an_output_dir_is_refused_at_render_time() -> None:
    with pytest.raises(AlpamayoSftError):
        _TRAIN.render_command()


def test_an_extra_override_that_steals_a_owned_key_is_refused() -> None:
    for override in (
        "paths.output_dir=/x",
        "+data.val_dataset.chunk_ids=99-100",
        "++evaluate.eval_ckpt=/x",
    ):
        with pytest.raises(AlpamayoSftError):
            dataclasses.replace(_TRAIN, extra_overrides=(override,))


def test_an_extra_override_that_repeats_itself_is_refused() -> None:
    with pytest.raises(AlpamayoSftError):
        dataclasses.replace(_TRAIN, extra_overrides=("a.b=1", "a.b=2"))


def test_a_recipe_pipeline_cannot_publish_and_an_empty_loss_curve_cannot_exist() -> None:
    with pytest.raises(AlpamayoSftError):
        PublishedLosses((), "README.md")
    with pytest.raises(AlpamayoSftError):
        AlpamayoSftRecipe(
            "c", "alpamayo2_sft.launch", True, {"eval_loss": PublishedMetric("e", 1.0, "s")}
        )
    with pytest.raises(AlpamayoSftError):
        AlpamayoSftRecipe("c", "alpamayo2_sft.evaluate_hf", False, {})
    assert isinstance(_TRAIN, AlpamayoSftSpec)
