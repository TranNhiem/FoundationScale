"""Tests for ``agentic_rl.servable.complete_for_serving`` against tiny fake HF
checkpoints (never a real multimodal model).

WHAT IS CLAIMED: ``architectures`` equal to the base's is a no-op
(``"already_servable"``, nothing on disk touched); otherwise every published
tensor name must be a base tensor name with a matching shape (refused by
name, naming the count on both sides, if not); the extras file holds EXACTLY
the base tensors the published checkpoint lacked, with the base's OWN values
(never the published/trained values, which stay exactly where they were);
the published checkpoint's own shard file is NEVER linked or copied from the
base (a loader reading a trained language tensor through the merged index
must see the TRAINED value); the extras cache file is created once per
``(base dir, extras key set)`` and reused, never rewritten, on a later call
that needs the same extras; linking falls back to a copy only on a declared
cross-device/permission ``OSError``.

WHAT IS NOT CLAIMED: anything about a REAL vLLM/transformers load of the
completed directory, or about ``RolloutHost.publish``'s wiring (see
``test_rollout_host.py``).
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from foundationscale.agentic_rl.servable import (
    ServableRefusal,
    ServableReport,
    _link_or_copy,
    _tensor_bytes,
    _TensorInfo,
    complete_for_serving,
)

_LANG_EMBED = "model.language_model.embed_tokens.weight"
_LANG_HEAD = "lm_head.weight"
_VISION_PATCH = "model.visual.patch_embed.weight"
_VISION_BLOCK = "model.visual.blocks.0.weight"

_BASE_ARCHITECTURES = ["FakeForConditionalGeneration"]
_PUBLISHED_ARCHITECTURES = ["FakeForCausalLM"]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_base(
    base_dir: Path,
    *,
    embed_value: float = 1.0,
    head_value: float = 2.0,
    with_preprocessor: bool = True,
) -> None:
    """A tiny two-shard "multimodal" base model: one shard of language
    tensors, one of vision tensors -- exercising the sharded (index.json)
    read path ``_weight_map`` must support.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    lang_tensors = {
        _LANG_EMBED: torch.full((4, 3), embed_value),
        _LANG_HEAD: torch.full((4, 3), head_value),
    }
    vision_tensors = {
        _VISION_PATCH: torch.full((2, 2), 30.0),
        _VISION_BLOCK: torch.full((2, 2), 40.0),
    }
    save_file(lang_tensors, str(base_dir / "model-00001-of-00002.safetensors"))
    save_file(vision_tensors, str(base_dir / "model-00002-of-00002.safetensors"))
    weight_map = dict.fromkeys(lang_tensors, "model-00001-of-00002.safetensors")
    weight_map.update(dict.fromkeys(vision_tensors, "model-00002-of-00002.safetensors"))
    _write_json(
        base_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": 0}, "weight_map": weight_map},
    )
    _write_json(
        base_dir / "config.json",
        {
            "architectures": _BASE_ARCHITECTURES,
            "model_type": "fake_vl",
            "text_config": {},
            "vision_config": {},
        },
    )
    if with_preprocessor:
        _write_json(base_dir / "preprocessor_config.json", {"fake": True})
    # chat_template.jinja is deliberately absent: "if present" must skip it.


def _build_published(
    published_dir: Path,
    *,
    embed_value: float = 9.0,
    head_value: float = 8.0,
    embed_shape: tuple[int, int] = (4, 3),
    extra_keys: dict[str, torch.Tensor] | None = None,
) -> None:
    """A tiny text-only "published" checkpoint: a SINGLE ``model.safetensors``
    (no index.json) -- exercising the single-shard read path.
    """
    published_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        _LANG_EMBED: torch.full(embed_shape, embed_value),
        _LANG_HEAD: torch.full((4, 3), head_value),
    }
    if extra_keys:
        tensors.update(extra_keys)
    save_file(tensors, str(published_dir / "model.safetensors"))
    _write_json(
        published_dir / "config.json",
        {"architectures": _PUBLISHED_ARCHITECTURES, "model_type": "fake_text"},
    )


def _get_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt") as handle:
        return handle.get_tensor(key)


# ---------------------------------------------------------------------------
# already_servable: nothing changed
# ---------------------------------------------------------------------------


def test_already_servable_when_architectures_match(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)
    _write_json(published / "config.json", {"architectures": _BASE_ARCHITECTURES})
    before = (published / "config.json").read_text(encoding="utf-8")

    report = complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))

    assert report == ServableReport(
        status="already_servable", extras_keys=0, extras_bytes=0, copied_files=()
    )
    assert (published / "config.json").read_text(encoding="utf-8") == before
    assert not (published / "model-serving-extras.safetensors").exists()
    assert not (published / "fs_servable.json").exists()
    assert not (published / "config.trained_text_only.json").exists()


# ---------------------------------------------------------------------------
# completed: the full happy path
# ---------------------------------------------------------------------------


def test_completed_extracts_exactly_the_missing_keys(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)

    report = complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))

    assert report.status == "completed"
    assert report.extras_keys == 2  # the two vision tensors, and nothing else
    assert report.extras_bytes > 0
    assert report.copied_files == ("config.json", "preprocessor_config.json")


def test_completed_published_language_tensors_are_what_a_loader_sees(tmp_path: Path) -> None:
    # Pins the one invariant the whole module exists to keep: the base's own
    # shard files are NEVER linked/copied into publish_dir, so a loader
    # walking the merged index.json must land on the TRAINED values, never
    # the base's original ones.
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base, embed_value=1.0, head_value=2.0)
    _build_published(published, embed_value=9.0, head_value=8.0)

    complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))

    index = json.loads((published / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    assert weight_map[_LANG_EMBED] == "model.safetensors"
    assert weight_map[_LANG_HEAD] == "model.safetensors"
    assert weight_map[_VISION_PATCH] == "model-serving-extras.safetensors"
    assert weight_map[_VISION_BLOCK] == "model-serving-extras.safetensors"
    assert index["metadata"]["total_size"] > 0

    # Load exactly as a fresh engine would: follow the index, not a guess.
    embed_via_index = _get_tensor(published / weight_map[_LANG_EMBED], _LANG_EMBED)
    assert torch.equal(embed_via_index, torch.full((4, 3), 9.0))  # trained, not base's 1.0
    head_via_index = _get_tensor(published / weight_map[_LANG_HEAD], _LANG_HEAD)
    assert torch.equal(head_via_index, torch.full((4, 3), 8.0))  # trained, not base's 2.0

    # The extras file carries the base's OWN vision values (never trained,
    # since the published checkpoint never had them).
    patch_via_index = _get_tensor(published / weight_map[_VISION_PATCH], _VISION_PATCH)
    assert torch.equal(patch_via_index, torch.full((2, 2), 30.0))

    # The base's own shard files were never placed in publish_dir.
    assert not (published / "model-00001-of-00002.safetensors").exists()
    assert not (published / "model-00002-of-00002.safetensors").exists()


def test_completed_copies_config_and_keeps_a_backup(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)
    original_published_config = (published / "config.json").read_text(encoding="utf-8")

    complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))

    new_config = json.loads((published / "config.json").read_text(encoding="utf-8"))
    assert new_config["architectures"] == _BASE_ARCHITECTURES
    backup = (published / "config.trained_text_only.json").read_text(encoding="utf-8")
    assert backup == original_published_config


def test_completed_writes_provenance(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)

    complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))

    provenance = json.loads((published / "fs_servable.json").read_text(encoding="utf-8"))
    assert provenance["base_model_dir"] == str(base.resolve())
    assert provenance["extras_key_count"] == 2
    assert provenance["published_key_count"] == 2
    assert provenance["extras_link_method"] == "hardlink"
    assert provenance["extras_file"] == "model-serving-extras.safetensors"


# ---------------------------------------------------------------------------
# cache reuse
# ---------------------------------------------------------------------------


def test_cache_reuse_does_not_rewrite_the_extras_file(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published1 = tmp_path / "published1"
    published2 = tmp_path / "published2"
    cache_dir = tmp_path / "cache"
    _build_base(base)
    _build_published(published1)
    _build_published(published2, embed_value=99.0, head_value=77.0)  # different trained values

    complete_for_serving(str(published1), str(base), cache_dir=str(cache_dir))
    cache_files = list(cache_dir.glob("extras-*.safetensors"))
    assert len(cache_files) == 1
    cache_file = cache_files[0]
    stat_before = cache_file.stat()

    complete_for_serving(str(published2), str(base), cache_dir=str(cache_dir))
    stat_after = cache_file.stat()

    assert list(cache_dir.glob("extras-*.safetensors")) == [cache_file]
    assert stat_after.st_ino == stat_before.st_ino
    assert stat_after.st_mtime_ns == stat_before.st_mtime_ns

    # Both publish dirs end up hard-linked to the SAME cache file.
    link1 = published1 / "model-serving-extras.safetensors"
    link2 = published2 / "model-serving-extras.safetensors"
    assert link1.stat().st_ino == cache_file.stat().st_ino
    assert link2.stat().st_ino == cache_file.stat().st_ino

    # Each publish dir still shows its OWN trained values, never swapped.
    assert torch.equal(
        _get_tensor(published2 / "model.safetensors", _LANG_EMBED), torch.full((4, 3), 99.0)
    )


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_refuses_unknown_published_key(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published, extra_keys={"model.language_model.no_such_key": torch.zeros(1)})

    with pytest.raises(ServableRefusal, match="not among"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


def test_refuses_shape_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published, embed_shape=(5, 3))  # base's embed_tokens is (4, 3)

    with pytest.raises(ServableRefusal, match="shape differing"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# cross-device / permission fallback
# ---------------------------------------------------------------------------


def test_cross_device_fallback_copies_instead_of_linking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)

    def fake_link(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", fake_link)

    report = complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))

    assert report.status == "completed"
    extras_path = published / "model-serving-extras.safetensors"
    assert extras_path.is_file()
    provenance = json.loads((published / "fs_servable.json").read_text(encoding="utf-8"))
    assert provenance["extras_link_method"] == "copy"
    assert (
        "EXDEV" in provenance["extras_link_fallback_reason"]
        or "cross-device" in (provenance["extras_link_fallback_reason"])
    )
    # A real copy, not a link: the extras file's inode differs from the cache's.
    cache_file = next((tmp_path / "cache").glob("extras-*.safetensors"))
    assert extras_path.stat().st_ino != cache_file.stat().st_ino
    assert torch.equal(_get_tensor(extras_path, _VISION_PATCH), torch.full((2, 2), 30.0))


def test_unrelated_oserror_from_link_propagates_uncaught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)

    def fake_link(src: object, dst: object) -> None:
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(os, "link", fake_link)

    with pytest.raises(OSError, match="no space left"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# complete_for_serving's own top-level argument/config checks
# ---------------------------------------------------------------------------


def test_refuses_a_publish_dir_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ServableRefusal, match="publish_dir"):
        complete_for_serving(
            str(tmp_path / "no-such-dir"), str(tmp_path), cache_dir=str(tmp_path / "cache")
        )


def test_refuses_a_base_model_dir_that_is_not_a_directory(tmp_path: Path) -> None:
    published = tmp_path / "published"
    published.mkdir()
    with pytest.raises(ServableRefusal, match="base_model_dir"):
        complete_for_serving(
            str(published), str(tmp_path / "no-such-dir"), cache_dir=str(tmp_path / "cache")
        )


def test_refuses_a_published_config_that_is_missing(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    published.mkdir()  # no config.json at all
    with pytest.raises(ServableRefusal, match="does not exist or is not a file"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


def test_refuses_a_config_that_is_not_valid_json(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)
    (published / "config.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ServableRefusal, match="not valid JSON"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


def test_refuses_a_config_whose_top_level_is_not_an_object(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)
    (published / "config.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ServableRefusal, match="expected a JSON object"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


def test_refuses_a_base_model_with_no_declared_architectures(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)
    _write_json(base / "config.json", {"architectures": []})
    with pytest.raises(ServableRefusal, match="non-empty list"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


def test_refuses_a_base_model_with_an_empty_weight_map(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    index["weight_map"] = {}
    _write_json(base / "model.safetensors.index.json", index)
    with pytest.raises(ServableRefusal, match="weight_map"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


def test_refuses_a_directory_with_neither_an_index_nor_a_single_shard(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)
    (published / "model.safetensors").unlink()  # no shard left at all
    with pytest.raises(ServableRefusal, match="no safetensors checkpoint found"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


def test_refuses_an_index_naming_a_shard_file_that_does_not_exist(tmp_path: Path) -> None:
    base = tmp_path / "base"
    published = tmp_path / "published"
    _build_base(base)
    _build_published(published)
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    index["weight_map"][_LANG_EMBED] = "model-00099-of-00002.safetensors"  # never written
    _write_json(base / "model.safetensors.index.json", index)
    with pytest.raises(ServableRefusal, match="does not exist"):
        complete_for_serving(str(published), str(base), cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# private helpers, exercised directly for branches no end-to-end fixture
# cheaply reaches (same convention as weight_sync.py's own `sync._prune()`
# direct calls)
# ---------------------------------------------------------------------------


def test_tensor_bytes_refuses_an_unrecognised_dtype() -> None:
    info = _TensorInfo(file="x.safetensors", shape=(2, 2), dtype="NOT_A_REAL_DTYPE")
    with pytest.raises(ServableRefusal, match="dtype"):
        _tensor_bytes(info)


def test_tensor_bytes_multiplies_shape_by_the_declared_itemsize() -> None:
    info = _TensorInfo(file="x.safetensors", shape=(2, 3), dtype="F32")
    assert _tensor_bytes(info) == 2 * 3 * 4


def test_link_or_copy_replaces_an_existing_destination(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(b"new")
    dst.write_bytes(b"stale")
    method, reason = _link_or_copy(src, dst)
    assert method == "hardlink"
    assert reason is None
    assert dst.read_bytes() == b"new"
    assert dst.stat().st_ino == src.stat().st_ino
