"""Gate contexts over FSDP SHARDED_STATE_DICT (DCP) checkpoints (#548).

#544 made ``_default_context_builder`` REFUSE the accelerate
``save_fsdp_model`` layout (a torch DCP store under ``pytorch_model_fsdp_0/``
whose every tensor key is wrapped as ``model.<fqn>``), and the callers scored
the refusal as UNMEASURED -- a Qwen2.5-7B sharded final saved 4 DCP shards
and got 0/0 gates run. #548 builds the context from the DCP metadata with
the wrapper stripped; these tests pin the strip, the loud refusal on an
unwrapped key, and the untouched safetensors path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foundationscale.gates.checkpoint_gates import CheckpointGateContext
from foundationscale.train import loop

torch = pytest.importorskip("torch")

_DCP_SUBDIR = "pytorch_model_fsdp_0"


def _write_dcp_store(ckpt_dir: Path, state_dict: dict) -> Path:
    """Persist ``state_dict`` as a real DCP store; single process, no init."""
    import torch.distributed.checkpoint as dcp

    dcp_dir = ckpt_dir / _DCP_SUBDIR
    dcp.save(state_dict, storage_writer=dcp.FileSystemWriter(str(dcp_dir)))
    return dcp_dir


def test_dcp_store_builds_a_context_with_unwrapped_fqns(tmp_path: Path) -> None:
    _write_dcp_store(
        tmp_path,
        {
            "model": {
                "a.weight": torch.zeros((2, 3), dtype=torch.float32),
                "b.bias": torch.ones((3,), dtype=torch.float16),
            }
        },
    )
    ctx = loop._default_context_builder(tmp_path)
    by_fqn = {t.fqn: t for t in ctx.tensors}
    assert set(by_fqn) == {"a.weight", "b.bias"}
    assert by_fqn["a.weight"].shape == (2, 3)
    assert by_fqn["a.weight"].dtype == "float32"
    assert by_fqn["a.weight"].kind == "tensor"
    assert by_fqn["b.bias"].shape == (3,)
    assert by_fqn["b.bias"].dtype == "float16"
    assert ctx.weights_path == str(tmp_path / "pytorch_model_fsdp_0")
    assert ctx.weights_key_prefix == "model."
    assert _DCP_SUBDIR in ctx.origin


def test_dcp_tensor_key_without_the_wrapper_raises(tmp_path: Path) -> None:
    _write_dcp_store(tmp_path, {"a.weight": torch.zeros((2,))})
    with pytest.raises(ValueError, match="a.weight"):
        loop._default_context_builder(tmp_path)


def test_safetensors_layout_still_goes_through_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"")
    sentinel = object()
    monkeypatch.setattr(
        CheckpointGateContext,
        "from_path",
        classmethod(lambda cls, path: sentinel),
    )
    assert loop._default_context_builder(tmp_path) is sentinel
