"""LoRA targets measured from the checkpoint for a family FS has not registered (measured: medgemma, gemma3)."""

from __future__ import annotations

import json
import struct

from foundationskills.skills.training.unknown_model import checkpoint_lora_leaves

GEMMA3 = [
    "language_model.model.layers.0.self_attn.q_proj.weight",
    "language_model.model.layers.0.self_attn.k_proj.weight",
    "language_model.model.layers.0.self_attn.v_proj.weight",
    "language_model.model.layers.0.self_attn.o_proj.weight",
    "language_model.model.layers.0.self_attn.q_norm.weight",
    "language_model.model.layers.0.mlp.gate_proj.weight",
    "language_model.model.layers.0.mlp.up_proj.weight",
    "language_model.model.layers.0.mlp.down_proj.weight",
    "language_model.model.embed_tokens.weight",
    "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
    "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.weight",
    "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
    "multi_modal_projector.mm_input_projection_weight",
]


def test_sharded_index_yields_language_leaves_and_names_the_tower_overlap(tmp_path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(GEMMA3, "s")}))
    got = checkpoint_lora_leaves(str(tmp_path))
    assert got == {
        "targets": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
        "shared_with_tower": ["q_proj"],
    }


def test_a_single_safetensors_header_is_read(tmp_path) -> None:
    header = json.dumps({"__metadata__": {}, "model.layers.0.self_attn.q_proj.weight": {}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    assert checkpoint_lora_leaves(str(tmp_path)) == {"targets": ["q_proj"], "shared_with_tower": []}


def test_unreadable_or_leafless_weights_measure_nothing(tmp_path) -> None:
    assert checkpoint_lora_leaves(str(tmp_path)) is None  # no weights at all
    (tmp_path / "model.safetensors.index.json").write_text("{broken")
    assert checkpoint_lora_leaves(str(tmp_path)) is None
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "s"}}))
    assert checkpoint_lora_leaves(str(tmp_path)) is None


def test_emit_passes_measured_targets_for_an_unregistered_family() -> None:
    from foundationskills.interfaces.fs.emit_train import emit_train
    from foundationskills.interfaces.fs.capabilities import FSCapabilities

    flags = {"--model", "--dataset", "--output-dir", "--nodes", "--gpus-per-node", "--profile-name", "--dp",
             "--objective", "--max-steps", "--learning-rate", "--max-sequence-length", "--per-device-batch-size",
             "--sharding-strategy", "--adapter", "--adapter-rank", "--adapter-alpha", "--adapter-dropout",
             "--adapter-target", "--warmup-steps", "--lr-scheduler-type", "--gradient-checkpointing", "--dry-run"}

    def caps() -> FSCapabilities:
        return FSCapabilities(
            available=True, fs_version="t", train_objectives=("sft",), sharding_strategies=("ddp", "fsdp"),
            executed_axes=(), refused_axes=("pp", "ep"), axes_measured=True, rl_algorithms=(), rl_runnable={},
            backends=("ddp", "fsdp"), families={"gemma4": ("gemma4",)}, rl_reward_kinds=(), rl_saves_checkpoint=False,
            train_flags=frozenset(flags), train_flag_choices={"--gradient-checkpointing": ("true", "false")})

    stage = {"stage": "sft", "method": "lora", "family": "gemma3", "executable": True,
             "hparams": {"lora_targets": ["q_proj", "v_proj"], "max_steps": 3}}
    ds = {"format": "sft", "shards": [{"path": "/d/shards/shard-00000.jsonl"}], "num_tokens": 10_000}
    hw = {"id": "gb200-189gb", "scheduler": "slurm", "env": {}, "bf16_dense_tflops": 1503.4, "peak_provenance": "measured"}
    spec = emit_train(stage, dataset=ds, model="/m", output_dir="/o", hardware=hw, nodes=1, gpus_per_node=1,
                      caps=caps(), run_name="r")
    argv = spec["argv"]
    targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--adapter-target"]
    assert targets == ["q_proj", "v_proj"]
