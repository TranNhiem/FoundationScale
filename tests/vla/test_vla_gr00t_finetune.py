"""GR00T fine-tuning through GR00T's own launcher: the published recipe, staging, the DDP hook."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from foundationscale.upstream.ledger import entries_for
from foundationscale.vla.adapters.gr00t import ddp_hook
from foundationscale.vla.adapters.gr00t.finetune import (
    PUBLISHED_LIBERO_RECIPE,
    Gr00tFinetuneError,
    Gr00tFinetuneSpec,
    on_gpus,
    render_command,
    stage_base_checkpoint,
)


def test_published_recipe_is_nvidias_libero_recipe() -> None:
    r = PUBLISHED_LIBERO_RECIPE
    assert (r.num_gpus, r.global_batch_size, r.max_steps, r.save_steps) == (8, 640, 20000, 1000)
    assert (r.learning_rate, r.warmup_ratio, r.weight_decay) == (1e-4, 0.05, 1e-5)
    assert (r.shard_size, r.num_shards_per_epoch, r.episode_sampling_rate) == (1024, 100000, 0.1)
    assert r.embodiment_tag == "LIBERO_PANDA" and r.per_gpu_batch == 80


def test_fewer_gpus_keep_the_global_batch() -> None:
    four = on_gpus(PUBLISHED_LIBERO_RECIPE, 4)
    assert four.global_batch_size == 640 and four.per_gpu_batch == 160


def test_a_global_batch_that_does_not_split_evenly_is_refused() -> None:
    with pytest.raises(Gr00tFinetuneError, match="not divisible"):
        on_gpus(PUBLISHED_LIBERO_RECIPE, 3)


@pytest.mark.parametrize(
    "field", [{"num_gpus": 0}, {"max_steps": True}, {"base_model": ""}, {"save_steps": -1}]
)
def test_invalid_fields_are_refused(field: dict[str, Any]) -> None:
    base = dict(
        base_model="b",
        dataset="d",
        embodiment_tag="E",
        output_dir="o",
        num_gpus=1,
        global_batch_size=8,
        max_steps=10,
        save_steps=5,
    )
    with pytest.raises(Gr00tFinetuneError):
        Gr00tFinetuneSpec(**(base | field))


def test_multi_gpu_command_goes_through_the_ddp_hook() -> None:
    argv = render_command(on_gpus(PUBLISHED_LIBERO_RECIPE, 4), torchrun="torchrun", master_port=1)
    assert argv[:5] == [
        "torchrun",
        "--nproc_per_node=4",
        "--master_port=1",
        "-m",
        "foundationscale.vla.adapters.gr00t.ddp_hook",
    ]
    flags = dict(zip(argv[5::2], argv[6::2], strict=False))
    assert flags["--global_batch_size"] == "640" and flags["--max_steps"] == "20000"
    assert flags["--num_gpus"] == "4" and flags["--embodiment_tag"] == "LIBERO_PANDA"
    i = argv.index("--color_jitter_params")
    assert argv[i + 1 : i + 9] == [
        "brightness",
        "0.3",
        "contrast",
        "0.4",
        "saturation",
        "0.5",
        "hue",
        "0.08",
    ]


def test_single_gpu_command_uses_gr00ts_launcher_directly() -> None:
    argv = render_command(on_gpus(PUBLISHED_LIBERO_RECIPE, 1), torchrun="torchrun")
    assert argv[3] == "gr00t/experiment/launch_finetune.py" and "-m" not in argv[:4]


def test_staging_symlinks_weights_patches_only_flash_attention(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"use_flash_attention": True, "model_name": "m"}))
    (base / "model.safetensors").write_text("w")
    (base / "trainer_state.json").write_text("{}")
    stage = stage_base_checkpoint(base, tmp_path / "stage", use_flash_attention=False)
    assert (stage / "model.safetensors").is_symlink()
    assert not (stage / "trainer_state.json").exists()  # a fine-tune starts its own schedule
    config = json.loads((stage / "config.json").read_text())
    assert config == {"use_flash_attention": False, "model_name": "m"}
    # the shared weights are never written
    assert json.loads((base / "config.json").read_text())["use_flash_attention"] is True
    # restaging is idempotent
    stage_base_checkpoint(base, tmp_path / "stage", use_flash_attention=False)


def test_staging_refuses_a_directory_without_config(tmp_path: Path) -> None:
    with pytest.raises(Gr00tFinetuneError, match="no config.json"):
        stage_base_checkpoint(tmp_path, tmp_path / "stage", use_flash_attention=False)


def test_ddp_hook_turns_on_ddp_and_runs_gr00ts_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    package = tmp_path / "gr00t"
    (package / "experiment").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    launcher = package / "experiment" / "launch_finetune.py"
    launcher.write_text(
        "import sys\n"
        "import gr00t.experiment.experiment as e\n"
        "class C:\n    class training:\n        use_ddp = False\n"
        "e.run(C)\n"
        "sys.modules['__hook_seen__'].argv = list(sys.argv)\n"
    )

    def run(config: Any) -> None:
        seen["use_ddp"] = config.training.use_ddp

    gr00t = types.ModuleType("gr00t")
    gr00t.__file__ = str(package / "__init__.py")
    experiment = types.ModuleType("gr00t.experiment.experiment")
    experiment.run = run  # type: ignore[attr-defined]
    recorder = types.ModuleType("__hook_seen__")
    monkeypatch.setitem(sys.modules, "gr00t", gr00t)
    monkeypatch.setitem(sys.modules, "gr00t.experiment", types.ModuleType("gr00t.experiment"))
    monkeypatch.setitem(sys.modules, "gr00t.experiment.experiment", experiment)
    monkeypatch.setitem(sys.modules, "__hook_seen__", recorder)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    ddp_hook.main(["--max_steps", "5"])
    assert seen["use_ddp"] is True
    assert recorder.argv == [str(launcher.resolve()), "--max_steps", "5"]  # type: ignore[attr-defined]


def test_both_deviations_are_in_the_upstream_ledger() -> None:
    assert {e.id for e in entries_for("gr00t")} == {
        "gr00t-launch-finetune-no-ddp-flag",
        "gr00t-flash-attn-off-aarch64",
    }
