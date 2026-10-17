"""longwam finetune spec: the published recipe, placement math and the rendered train argv."""

from __future__ import annotations

import dataclasses

import pytest

from foundationscale.vla.adapters.longwam.finetune import (
    PUBLISHED_RECIPES,
    LongWamFinetuneError,
    LongWamFinetuneSpec,
    LongWamRecipe,
)

_SPEC = LongWamFinetuneSpec(
    task="libero",
    data_root="/data/longwam",
    checkpoint_root="/data/checkpoints",
    gpus=(0, 1, 2, 3),
    batch_size=4,
    gradient_accumulation_steps=8,
)


def test_the_libero_recipe_is_the_training_doc() -> None:
    recipe = PUBLISHED_RECIPES["libero"]
    assert (recipe.task, recipe.expected_global_batch_size) == ("libero", 128)
    assert (recipe.per_rank_batch_size, recipe.gradient_accumulation_steps) == (8, 2)
    assert (recipe.learning_rate, recipe.schedule) == (1e-4, "10 epochs, cosine")
    assert recipe.released_checkpoints == (
        "Efficient-Large-Model/Long-WAM-LIBERO-IDM",
        "Efficient-Large-Model/Long-WAM-LIBERO-COD",
    )
    assert _SPEC.with_output_dir("/runs/libero").render_command()[0] == "torchrun"


def test_render_command_is_train_py_with_the_spec_owned_overrides() -> None:
    spec = dataclasses.replace(_SPEC, extra_overrides=("denoising=codenoise", "resume=null"))
    assert (spec.with_output_dir("/runs/libero")).render_command() == [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        "scripts/train.py",
        "task=libero",
        "batch_size=4",
        "gradient_accumulation_steps=8",
        "output_dir=/runs/libero",
        "denoising=codenoise",
        "resume=null",
    ]


def test_render_env_exposes_roots_and_omits_the_outputs_root_until_an_output_dir() -> None:
    assert _SPEC.render_env() == {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "LONGWAM_DATA_ROOT": "/data/longwam",
        "LONGWAM_CHECKPOINT_ROOT": "/data/checkpoints",
    }
    assert (_SPEC.with_output_dir("/runs/libero")).render_env()["LONGWAM_OUTPUT_ROOT"] == (
        "/runs/libero"
    )
    out = _SPEC.with_output_dir("/runs/libero")
    assert out.on_gpus((0, 1, 2, 3)).output_dir == "/runs/libero"  # on_gpus leaves output_dir alone


def test_on_gpus_re_derives_accumulation_to_hold_the_published_global_batch() -> None:
    few = _SPEC
    assert few.gradient_accumulation_steps == 8
    four = LongWamFinetuneSpec(
        data_root="d",
        checkpoint_root="c",
        gpus=(0, 1, 2, 3),
        batch_size=4,
        gradient_accumulation_steps=8,
    ).on_gpus((0, 1, 2, 3))
    assert (len(four.gpus), four.batch_size, four.gradient_accumulation_steps) == (4, 4, 8)
    eight = four.on_gpus((0, 1, 2, 3, 4, 5, 6, 7))
    assert (len(eight.gpus), eight.batch_size, eight.gradient_accumulation_steps) == (8, 4, 4)
    assert few.batch_size == 4 and few.gradient_accumulation_steps == 8


def test_an_unknown_task_is_refused() -> None:
    with pytest.raises(LongWamFinetuneError):
        dataclasses.replace(_SPEC, task="robotwin2")


@pytest.mark.parametrize("field", ["data_root", "checkpoint_root"])
def test_an_empty_storage_root_is_refused(field: str) -> None:
    with pytest.raises(LongWamFinetuneError):
        dataclasses.replace(_SPEC, **{field: "  "})


def test_a_run_without_gpus_is_refused() -> None:
    with pytest.raises(LongWamFinetuneError):
        dataclasses.replace(_SPEC, gpus=())


@pytest.mark.parametrize("gpus", [(0, 0), (1, 2, 2, 3), (3, 2, 1, 3)])
def test_a_repeated_gpu_id_is_refused(gpus: tuple[int, ...]) -> None:
    with pytest.raises(LongWamFinetuneError):
        dataclasses.replace(_SPEC, gpus=gpus)


@pytest.mark.parametrize("gpus", [("a", "b"), (0, 1, -1), (0, True)])
def test_a_gpu_id_that_is_not_a_cuda_id_is_refused(gpus: tuple[int, ...]) -> None:
    with pytest.raises(LongWamFinetuneError):
        dataclasses.replace(_SPEC, gpus=gpus)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["batch_size", "gradient_accumulation_steps"])
@pytest.mark.parametrize("value", [0, -8, 2.5, "8"])
def test_a_non_integer_or_degenerate_per_rank_layout_is_refused(field: str, value: object) -> None:
    with pytest.raises(LongWamFinetuneError):
        dataclasses.replace(_SPEC, **{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("gpus", [(), (0,), (0, 1), (0, 1, 2, 3, 4, 5, 6)])
def test_a_layout_below_the_published_global_batch_is_refused(gpus: tuple[int, ...]) -> None:
    with pytest.raises(LongWamFinetuneError):
        dataclasses.replace(_SPEC, gpus=gpus)


@pytest.mark.parametrize(
    "override",
    ["task=libero", "+batch_size=2", "++gradient_accumulation_steps=4", "output_dir=/x"],
)
def test_an_extra_override_that_steals_an_owned_key_is_refused(override: str) -> None:
    with pytest.raises(LongWamFinetuneError):
        dataclasses.replace(_SPEC, extra_overrides=(override, "expected_global_batch_size=256"))


def test_a_repeated_extra_override_key_is_refused() -> None:
    with pytest.raises(LongWamFinetuneError):
        dataclasses.replace(_SPEC, extra_overrides=("model.b=1", "model.b=2"))


def test_rendering_without_an_output_dir_is_refused() -> None:
    with pytest.raises(LongWamFinetuneError):
        _SPEC.render_command()


def test_on_gpus_refuses_a_layout_that_does_not_divide_the_global_batch() -> None:
    with pytest.raises(LongWamFinetuneError):
        _SPEC.on_gpus((0, 1, 2))


def test_a_recipe_that_disagrees_with_itself_cannot_exist() -> None:
    with pytest.raises(LongWamFinetuneError):
        LongWamRecipe(
            task="libero",
            expected_global_batch_size=128,
            published_gpus=4,
            per_rank_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            schedule="10",
            released_checkpoints=("a", "b"),
            source="s",
        )
    with pytest.raises(LongWamFinetuneError):
        LongWamRecipe(
            task="libero",
            expected_global_batch_size=16,
            published_gpus=1,
            per_rank_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            schedule="10",
            released_checkpoints=(),
            source="s",
        )
