"""CPU tests: ``save_checkpoint`` writes an ADAPTER-ONLY checkpoint, atomically,
for a peft-wrapped model, and leaves the non-adapter save path untouched.

No GPU, no process group: every case runs at world_size == 1, the identity
path a single-process run also hits. FS_FORBID_SKIPS=1 clean.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import foundationscale.rl.distributed as dist_mod
from foundationscale.rl.distributed import DistContext, _atomic_peft_save, save_checkpoint


def _ctx(rank: int = 0, world: int = 1) -> DistContext:
    return DistContext(
        rank=rank,
        world_size=world,
        local_rank=rank,
        device=torch.device("cpu"),
        is_distributed=(world > 1),
        owns_process_group=False,
    )


class _FakePeftModel:
    """peft_config present -> save_checkpoint must treat this as adapter-only."""

    def __init__(self) -> None:
        self.peft_config = {"default": object()}
        self.save_pretrained_calls: list[dict] = []

    def save_pretrained(self, out_dir, **kwargs) -> None:
        # If save_checkpoint ever called this DIRECTLY for a peft model, the
        # adapter-only + atomic contract would be bypassed; tests below
        # assert _atomic_peft_save was used instead, not this method.
        self.save_pretrained_calls.append(dict(kwargs))


class _FakeNonPeftModel:
    def __init__(self) -> None:
        self.saved_with: dict | None = None

    def save_pretrained(self, out_dir, **kwargs) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "model.safetensors").write_text("weights")
        self.saved_with = dict(kwargs)


def test_save_checkpoint_peft_model_sharding_none_uses_atomic_adapter_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple] = []

    def fake_atomic(target, out_dir, state_dict) -> None:
        calls.append((target, out_dir, state_dict))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "adapter_model.safetensors").write_text("lora")
        (Path(out_dir) / "adapter_config.json").write_text("{}")

    monkeypatch.setattr(dist_mod, "_atomic_peft_save", fake_atomic)
    model = _FakePeftModel()
    out = tmp_path / "ckpt"
    ok = save_checkpoint(model, None, str(out), _ctx(), sharding="none", step=3)
    assert ok is True
    assert len(calls) == 1
    target, out_dir, state_dict = calls[0]
    assert target is model
    assert out_dir == str(out)
    assert state_dict is None
    assert model.save_pretrained_calls == []  # the direct path was NOT taken
    assert (out / "adapter_model.safetensors").exists()
    assert (out / "adapter_config.json").exists()
    assert not (out / "model.safetensors").exists()


def test_save_checkpoint_peft_model_sharding_fsdp_passes_gathered_state_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gathered = {"base.weight": torch.randn(2, 2), "lora_A.weight": torch.randn(2, 2)}
    monkeypatch.setattr(dist_mod, "_gather_full_state_dict", lambda model, ctx, dtype: gathered)
    monkeypatch.setattr(dist_mod, "_save_barrier", lambda ctx: None)

    calls: list[tuple] = []

    def fake_atomic(target, out_dir, state_dict) -> None:
        calls.append((target, out_dir, state_dict))
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dist_mod, "_atomic_peft_save", fake_atomic)
    model = _FakePeftModel()
    out = tmp_path / "ckpt"
    ok = save_checkpoint(model, None, str(out), _ctx(), sharding="fsdp", step=1)
    assert ok is True
    assert len(calls) == 1
    target, out_dir, state_dict = calls[0]
    assert target is model
    assert state_dict is gathered
    assert model.save_pretrained_calls == []


def test_save_checkpoint_non_peft_model_is_unaffected(tmp_path: Path) -> None:
    # No peft_config attribute at all: the historical direct-write path, byte
    # for byte -- the exact assertions test_save_checkpoint_world_one_none
    # (tests/rl/test_rl_distributed.py) already makes, repeated here as the
    # negative control for the is_peft branch added in this change.
    model = _FakeNonPeftModel()
    out = tmp_path / "ckpt"
    ok = save_checkpoint(model, None, str(out), _ctx(), sharding="none", step=5)
    assert ok is True
    assert model.saved_with == {"safe_serialization": True}
    assert (out / "model.safetensors").exists()


def test_save_checkpoint_model_without_peft_config_attribute_takes_direct_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Defensive: a model that raises on attribute access other than the ones
    # it defines must not be probed for peft_config in a way that crashes --
    # getattr(..., None) is the contract, confirmed by exercising a model
    # with no such attribute at all (the common case) rather than one that
    # raises, since raising on getattr would be the model's own defect.
    called = {"atomic": False}
    monkeypatch.setattr(
        dist_mod, "_atomic_peft_save", lambda *a, **k: called.__setitem__("atomic", True)
    )
    model = _FakeNonPeftModel()
    save_checkpoint(model, None, str(tmp_path / "ckpt"), _ctx(), sharding="none", step=1)
    assert called["atomic"] is False


def test_atomic_peft_save_reuses_the_real_sft_plane_helper_unmocked(tmp_path: Path) -> None:
    # Integration check, nothing mocked: _atomic_peft_save really imports and
    # calls foundationscale.train.fsdp_peft_save._atomically_overwrite_adapter
    # (the reuse this change's docstring claims), and that helper's
    # temp-dir-then-replace mechanism really lands files in out_dir.
    class _Skeleton:
        def save_pretrained(self, tmp_dir, state_dict=None) -> None:
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            (Path(tmp_dir) / "adapter_model.safetensors").write_text("lora-bytes")
            (Path(tmp_dir) / "adapter_config.json").write_text('{"r": 8}')

    out_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    # A stale file from an earlier (hypothetically broken) write already
    # sits in out_dir; the atomic replace must still leave the NEW bytes.
    (out_dir / "adapter_model.safetensors").write_text("stale")

    _atomic_peft_save(_Skeleton(), str(out_dir), None)

    assert (out_dir / "adapter_model.safetensors").read_text() == "lora-bytes"
    assert (out_dir / "adapter_config.json").read_text() == '{"r": 8}'
    # No leftover temp directory from the write.
    leftovers = [p for p in out_dir.iterdir() if p.name.startswith(".fs_fsdp1_peft_save_tmp_")]
    assert leftovers == []
