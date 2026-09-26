
"""Tests for emit_rl: config shape, runnability gating, single-device warning."""
from __future__ import annotations

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.interfaces.fs.emit_rl import emit_rl


def make_caps(**over):
    base = dict(
        available=True,
        fs_version="0.1.0-test",
        train_flags=frozenset(),
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=("dr_grpo", "gspo", "dapo", "dpo"),
        rl_runnable={"dr_grpo": None, "gspo": None, "dapo": None,
                     "dpo": "0 of 1 required advantage estimators declared; preference family not wired"},
        families={"gemma4": ("gemma4",)},
        backends=("ddp", "fsdp"),
    )
    base.update(over)
    return FSCapabilities(**base)


def rl_stage(**hparams):
    return {
        "name": "rl1",
        "stage": "rl",
        "algorithm": "dr_grpo",
        "method": "full",
        "hparams": hparams,
        "data": {"format": "rl"},
    }


def rl_dataset(tmp_path):
    return {
        "format": "rl",
        "shards": [{"path": f"{tmp_path}/rl/shard-00000.jsonl", "sha256": "0" * 64, "records": 5}],
        "fs_columns": {"text_column": None, "image_column": None, "gold_key": "answer"},
    }


def test_runnable_algorithm_config_and_argv(tmp_path):
    caps = make_caps()
    stage = rl_stage(lr=2e-6, group_size=8, max_steps=12, temperature=0.9)
    spec = emit_rl(stage, dataset=rl_dataset(tmp_path), model="models/g4",
                   output_dir=str(tmp_path / "out"), caps=caps, run_name="fskills-rl1")
    assert spec["executable"] is True
    assert spec["missing"] is None
    assert spec["entry"] == "fskills-rl"
    assert spec["argv"] == ["fskills-rl", "--config", f"{tmp_path}/out/rl_config.json"]
    assert spec["dry_run_argv"] == spec["argv"] + ["--dry-run"]

    cfg = spec["rl_config"]
    assert cfg["model"] == "models/g4"
    assert cfg["dataset"] == f"{tmp_path}/rl"
    assert cfg["algorithm"] == "dr_grpo"
    assert cfg["learning_rate"] == 2e-6
    assert cfg["group_size"] == 8
    assert cfg["max_steps"] == 12
    assert cfg["temperature"] == 0.9
    assert cfg["gold_key"] == "answer"  # from dataset fs_columns
    assert cfg["output_dir"] == f"{tmp_path}/out"
    assert cfg["save_final"] is True
    # LR field names exactly; no FS-CLI-style keys leaked
    assert "learning-rate" not in cfg
    assert any("single-device" in w for w in spec["warnings"])
    assert f"{tmp_path}/out/rl_reports.json" in spec["expected_outputs"]
    assert f"{tmp_path}/out/final" in spec["expected_outputs"]


def test_non_runnable_algorithm_blocked(tmp_path):
    caps = make_caps()
    stage = rl_stage()
    stage["algorithm"] = "dpo"
    spec = emit_rl(stage, dataset=rl_dataset(tmp_path), model="m",
                   output_dir=str(tmp_path / "out"), caps=caps, run_name="fskills-rl1")
    assert spec["executable"] is False
    assert "dpo" in spec["missing"]
    assert "not wired" in spec["missing"] or "refuses" in spec["missing"]


def test_unregistered_algorithm_blocked(tmp_path):
    caps = make_caps()
    stage = rl_stage()
    stage["algorithm"] = "some_future_algo"
    spec = emit_rl(stage, dataset=rl_dataset(tmp_path), model="m",
                   output_dir=str(tmp_path / "out"), caps=caps, run_name="r")
    assert spec["executable"] is False
    assert "some_future_algo" in spec["missing"]
