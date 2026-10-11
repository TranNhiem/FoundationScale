"""saved_name_map against transformers' real save path (Qwen2-Audio renames on save in 5.x)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("transformers.core_model_loading")


def _tiny_qwen2_audio():  # type: ignore[no-untyped-def]
    from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration

    config = Qwen2AudioConfig(
        audio_config={
            "d_model": 16,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "num_mel_bins": 8,
            "max_source_positions": 16,
        },
        text_config={
            "model_type": "qwen2",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 64,
        },
    )
    return Qwen2AudioForConditionalGeneration(config)


def test_saved_names_match_what_save_pretrained_writes(tmp_path: Path) -> None:
    import safetensors

    from foundationscale.train.saved_names import saved_name_map

    model = _tiny_qwen2_audio()
    renamed = saved_name_map(model)
    model.save_pretrained(tmp_path)
    with safetensors.safe_open(str(tmp_path / "model.safetensors"), framework="pt") as f:
        on_disk = set(f.keys())
    in_memory = set(model.state_dict())
    predicted = {(renamed or {}).get(n, n) for n in in_memory}
    # every saved tensor is predicted (tied weights may be dropped by the saver)
    assert on_disk <= predicted
    if renamed:  # the 5.x layout: the encoder sits under model. in memory only
        assert any(n.startswith("model.audio_tower.") for n in renamed)
        assert all(not d.startswith("model.audio_tower.") for d in on_disk)


def test_saved_tower_digests_key_like_the_base(tmp_path: Path) -> None:
    from foundationscale.train.saved_names import saved_name_map
    from foundationscale.train.speech_adjudication import (
        digests_from_named_tensors,
        digests_from_safetensors_dir,
    )

    model = _tiny_qwen2_audio()
    tower = (
        "model.audio_tower"
        if any(n.startswith("model.audio_tower.") for n in model.state_dict())
        else "audio_tower"
    )
    base = digests_from_named_tensors(model.state_dict().items(), [tower])
    model.save_pretrained(tmp_path)
    renamed = saved_name_map(model) or {}
    saved = digests_from_safetensors_dir(
        tmp_path, [tower], {disk: mem for mem, disk in renamed.items()}
    )
    assert base and saved.keys() == base.keys()
    assert saved == base  # nothing trained, so every digest is unchanged


def test_fp32_master_weights_are_compared_at_the_base_precision(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from foundationscale.train.speech_adjudication import (
        digests_from_named_tensors,
        digests_from_safetensors_dir,
    )

    still = torch.randn(4, 4).to(torch.bfloat16)
    moved = torch.randn(4, 4).to(torch.bfloat16)
    base = digests_from_named_tensors([("t.still", still), ("t.moved", moved)], ["t"])
    # FSDP mixed precision saves fp32 master weights: one untouched, one trained
    save_file(
        {"t.still": still.float(), "t.moved": moved.float() + 0.5},
        str(tmp_path / "model.safetensors"),
    )
    narrowed: list[str] = []
    saved = digests_from_safetensors_dir(
        tmp_path,
        ["t"],
        narrow_to={"t.still": "bfloat16", "t.moved": "bfloat16"},
        narrowed=narrowed,
    )
    assert saved["t.still"] == base["t.still"]
    assert saved["t.moved"] != base["t.moved"]
    assert saved["t.moved"].startswith("bfloat16:")
    assert sorted(narrowed) == ["t.moved", "t.still"]
    # without the base precision nothing is narrowed: a dtype change, not a comparison
    plain = digests_from_safetensors_dir(tmp_path, ["t"])
    assert plain["t.still"].startswith("float32:")


def test_sharded_dcp_save_is_read_under_its_container_prefix(tmp_path: Path) -> None:
    import torch
    import torch.distributed.checkpoint as dcp

    from foundationscale.train.speech_adjudication import (
        digests_from_dcp_dir,
        digests_from_named_tensors,
    )

    still = torch.randn(4, 4).to(torch.bfloat16)
    moved = torch.randn(4, 4).to(torch.bfloat16)
    base = digests_from_named_tensors(
        [("model.audio_tower.still", still), ("model.audio_tower.moved", moved)],
        ["model.audio_tower"],
    )
    # accelerate's FSDP sharded save: {"model": state_dict}, fp32 master weights
    state = {
        "model": {
            "model.audio_tower.still": still.float(),
            "model.audio_tower.moved": moved.float() + 0.5,
        }
    }
    dcp.save(state, checkpoint_id=str(tmp_path / "pytorch_model_fsdp_0"), no_dist=True)
    narrowed: list[str] = []
    saved = digests_from_dcp_dir(
        tmp_path / "pytorch_model_fsdp_0",
        ["model.audio_tower"],
        expected=list(base),
        narrow_to=dict.fromkeys(base, "bfloat16"),
        narrowed=narrowed,
    )
    assert saved is not None and saved.keys() == base.keys()
    assert saved["model.audio_tower.still"] == base["model.audio_tower.still"]
    assert saved["model.audio_tower.moved"] != base["model.audio_tower.moved"]
    assert len(narrowed) == 2
    # a base name the save does not carry under any container: unread, not guessed
    assert (
        digests_from_dcp_dir(
            tmp_path / "pytorch_model_fsdp_0", ["model.audio_tower"], expected=["absent.weight"]
        )
        is None
    )
