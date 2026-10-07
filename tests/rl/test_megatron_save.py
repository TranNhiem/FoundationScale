"""CPU legs for the Megatron RL lane's checkpoint save: HF weights + manifest + gates.

A real (tiny) ``LlamaForCausalLM`` writes a real safetensors directory, so the
completeness gate reads the same on-disk format a GB200 save produces. The
declared set is handed in explicitly, exactly as the driver hands in the names
its last refit wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foundationscale.rl.megatron.save import (
    MANIFEST_NAME,
    run_policy_save_gates,
    save_policy_checkpoint,
)

_TOPOLOGY = {
    "nodes": 1,
    "gpus_per_node": 1,
    "tensor_parallel": 1,
    "pipeline_parallel": 1,
    "data_parallel": 1,
    "expert_parallel": 1,
    "context_parallel": 1,
}


def _tiny_llama() -> Any:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _save(tmp_path: Path, model: Any, names: Any) -> Path:
    return save_policy_checkpoint(
        model,
        None,
        tmp_path / "final",
        declared_names=names,
        run_id="unit",
        topology=_TOPOLOGY,
        config={"steps": 1},
    )


def test_empty_declared_names_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="declared_names is empty"):
        _save(tmp_path, _tiny_llama(), [])
    assert not (tmp_path / "final").exists()


def test_save_writes_a_manifest_the_gate_reader_finds_with_the_declared_set(
    tmp_path: Path,
) -> None:
    from foundationscale.checkpoint import load_manifest

    model = _tiny_llama()
    names = sorted(model.state_dict())
    out = _save(tmp_path, model, names)
    assert (out / MANIFEST_NAME).is_file()
    manifest = load_manifest(out)
    assert manifest is not None and manifest.declared is not None
    assert list(manifest.declared.declared_fqns) == names
    assert manifest.declared.num_experts == 0  # a POSITIVE dense declaration


def test_save_gates_pass_when_every_declared_tensor_is_on_disk(tmp_path: Path) -> None:
    model = _tiny_llama()
    out = _save(tmp_path, model, model.state_dict())
    report = run_policy_save_gates(out, first=True)
    assert report.ok, str(report)


def test_save_gates_refuse_a_declared_tensor_the_checkpoint_lacks(tmp_path: Path) -> None:
    model = _tiny_llama()
    names = [*model.state_dict(), "model.layers.0.mlp.ghost_proj.weight"]
    out = _save(tmp_path, model, names)
    report = run_policy_save_gates(out, first=False)
    assert not report.ok
    assert "ghost_proj" in str(report)
