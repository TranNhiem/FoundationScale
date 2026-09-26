
"""Tests for emit_train: flag mapping, capability gating, launch shape."""
from __future__ import annotations

import pytest

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.interfaces.fs.emit_train import emit_train

_ALL_FLAGS = frozenset(
    {
        "--model", "--dataset", "--output-dir", "--max-steps",
        "--per-device-batch-size", "--learning-rate", "--save-interval",
        "--max-sequence-length", "--objective", "--seed", "--dp",
        "--tp", "--pp", "--ep", "--cp", "--nodes", "--gpus-per-node",
        "--profile-name", "--profile-path", "--dry-run", "--precision",
        "--adapter", "--adapter-rank", "--adapter-alpha", "--adapter-target",
        "--adapter-dropout", "--optimizer", "--gradient-accumulation-steps",
        "--lr-scheduler-type", "--warmup-steps", "--sharding-strategy",
        "--gradient-checkpointing", "--logging-steps", "--max-grad-norm",
        "--dataloader-num-workers",
    }
)


def make_caps(**over):
    base = dict(
        available=True,
        fs_version="0.1.0-test",
        train_flags=_ALL_FLAGS,
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=("dr_grpo", "gspo", "dapo"),
        rl_saves_checkpoint=True,  # hypothetical: FS 77bfa65 measures False (see test_fix_*)
        rl_reward_kinds=("mcq_letter",),
        rl_runnable={"dr_grpo": None, "gspo": None, "dapo": None},
        families={"gemma4": ("gemma4",), "qwen3.5": ("qwen3_5",)},
        backends=("ddp", "fsdp"),
    )
    base.update(over)
    return FSCapabilities(**base)


def base_stage(**hparams):
    return {
        "name": "sft1",
        "stage": "sft",
        "algorithm": None,
        "method": "full",
        # a planned sequence length is required (FS's own default is 128 tokens)
        "hparams": hparams if ("seq_len" in hparams or "max_sequence_length" in hparams)
        else {"max_sequence_length": 2048, **hparams},
        "family": "gemma4",
        "data": {"format": "sft"},
    }


def dataset(fmt="sft", tmp_path=None, gold_key=None, image_column=None):
    root = tmp_path or "."
    return {
        "format": fmt,
        "shards": [{"path": f"{root}/ds/shard-00000.jsonl", "sha256": "0" * 64, "records": 10}],
        "fs_columns": {"text_column": "text", "image_column": image_column, "gold_key": gold_key},
    }


def flag_value(argv, flag):
    return argv[argv.index(flag) + 1]


def call(stage, caps, ds, tmp_path, nodes=1, gpus=1, hardware=None):
    return emit_train(
        stage,
        dataset=ds,
        model="models/gemma4-e4b",
        output_dir=str(tmp_path / "out"),
        hardware=hardware if hardware is not None else {"id": "local", "scheduler": None},
        nodes=nodes,
        gpus_per_node=gpus,
        caps=caps,
        run_name="fskills-sft1",
    )


def test_hparam_flag_mapping(tmp_path):
    caps = make_caps()
    stage = base_stage(
        lr=1e-4, lr_scheduler="cosine", warmup_steps=37, seq_len=4096,
        micro_batch=2, grad_accum=8, max_steps=120, precision="bf16",
        grad_ckpt=True, sharding="fsdp", save_interval=25, seed=7,
    )
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path)
    assert spec["executable"] is True
    argv = spec["argv"]
    assert float(flag_value(argv, "--learning-rate")) == pytest.approx(1e-4)
    assert flag_value(argv, "--lr-scheduler-type") == "cosine"
    assert flag_value(argv, "--warmup-steps") == "37"
    assert flag_value(argv, "--max-sequence-length") == "4096"
    assert flag_value(argv, "--per-device-batch-size") == "2"
    assert flag_value(argv, "--gradient-accumulation-steps") == "8"
    assert flag_value(argv, "--max-steps") == "120"
    assert flag_value(argv, "--precision") == "bf16"
    assert flag_value(argv, "--gradient-checkpointing") == "true"
    assert flag_value(argv, "--sharding-strategy") == "fsdp"
    assert flag_value(argv, "--save-interval") == "25"
    assert flag_value(argv, "--seed") == "7"
    assert flag_value(argv, "--objective") == "sft"
    assert spec["dry_run_argv"] == argv + ["--dry-run"]
    assert spec["entry"] == "foundationscale-train"
    assert f"{tmp_path}/out/run_manifest.json" in spec["expected_outputs"]


def test_max_steps_derived_from_tokens(tmp_path):
    caps = make_caps()
    # ceil(1_000_000 / (2 * 4096 * 4 * 4)) = ceil(7.629...) = 8
    stage = base_stage(micro_batch=2, seq_len=4096, grad_accum=4, tokens=1_000_000)
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path, gpus=4)
    assert flag_value(spec["argv"], "--max-steps") == "8"
    assert flag_value(spec["argv"], "--dp") == "4"


def test_warmup_ratio_derived(tmp_path):
    caps = make_caps()
    stage = base_stage(max_steps=200, warmup_ratio=0.03)
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path)
    assert flag_value(spec["argv"], "--warmup-steps") == "6"


def test_missing_flag_makes_spec_non_executable(tmp_path):
    caps = make_caps(train_flags=_ALL_FLAGS - {"--warmup-steps"})
    stage = base_stage(warmup_steps=10)
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path)
    assert spec["executable"] is False
    assert "FS flag --warmup-steps not supported by installed FS" in spec["missing"]
    assert "--warmup-steps" not in spec["argv"]


def test_pp2_refused_via_caps(tmp_path):
    caps = make_caps()
    stage = base_stage(pp=2)
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path)
    assert spec["executable"] is False
    assert "pp>1" in spec["missing"]


def test_tp2_executes_and_emits(tmp_path):
    caps = make_caps()
    stage = base_stage(tp=2)
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path, gpus=4)
    assert spec["executable"] is True
    assert flag_value(spec["argv"], "--tp") == "2"
    assert flag_value(spec["argv"], "--dp") == "2"


def test_profile_choice(tmp_path):
    caps = make_caps()
    stage = base_stage()
    multi = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path, nodes=2, gpus=4,
                 hardware={"id": "gb200-tray", "scheduler": "slurm"})
    assert flag_value(multi["argv"], "--profile-name") == "slurm-generic"
    single = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path, nodes=1, gpus=1)
    assert flag_value(single["argv"], "--profile-name") == "local-single-node"


def test_mm_sft_image_column_env(tmp_path):
    caps = make_caps()
    spec = call(base_stage(), caps, dataset(fmt="mm_sft", tmp_path=tmp_path, image_column="image"), tmp_path)
    assert spec["env"]["FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN"] == "image"


def test_torchrun_when_world_gt_1_else_python(tmp_path):
    caps = make_caps()
    stage = base_stage()
    multi = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path, gpus=4)
    assert multi["argv"][0] == "torchrun"
    assert flag_value(multi["argv"], "--nproc-per-node") == "4"
    # one node: standalone rendezvous (a direct launch never expands $MASTER_ADDR)
    assert "--standalone" in multi["argv"] and "--rdzv-endpoint" not in multi["argv"]
    assert "-m" in multi["argv"] and "foundationscale.train.cli" in multi["argv"]
    single = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path, gpus=1)
    assert single["argv"][:3] == ["python", "-m", "foundationscale.train.cli"]


def test_lora_registered_family_no_targets(tmp_path):
    caps = make_caps()
    stage = base_stage(lora_rank=32, lora_alpha=64, lora_dropout=0.05)
    stage["method"] = "lora"
    stage["lora_targets"] = ["linear_qkv"]  # ignored: gemma4 is FS-registered
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path)
    assert spec["executable"] is True
    assert flag_value(spec["argv"], "--adapter") == "lora"
    assert flag_value(spec["argv"], "--adapter-rank") == "32"
    assert float(flag_value(spec["argv"], "--adapter-alpha")) == pytest.approx(64.0)
    assert float(flag_value(spec["argv"], "--adapter-dropout")) == pytest.approx(0.05)
    assert "--adapter-target" not in spec["argv"]


def test_lora_unregistered_family_emits_targets(tmp_path):
    caps = make_caps()
    stage = base_stage(lora_rank=16)
    stage["method"] = "lora"
    stage["family"] = "llama3"
    stage["lora_targets"] = ["q_proj", "v_proj"]
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path)
    assert spec["executable"] is True
    assert spec["argv"].count("--adapter-target") == 2


def test_fs_args_passthrough_overrides(tmp_path):
    caps = make_caps()
    stage = base_stage(lr=1e-4)
    stage["fs"] = {"args": {"learning-rate": 5e-5, "max-grad-norm": 0.5}}
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path)
    assert float(flag_value(spec["argv"], "--learning-rate")) == pytest.approx(5e-5)
    assert float(flag_value(spec["argv"], "--max-grad-norm")) == pytest.approx(0.5)


def test_qlora_not_executable(tmp_path):
    caps = make_caps()
    stage = base_stage()
    stage["method"] = "qlora"
    spec = call(stage, caps, dataset(tmp_path=tmp_path), tmp_path)
    assert spec["executable"] is False
    assert "qlora" in spec["missing"]
