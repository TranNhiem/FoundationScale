"""Layout tests built from empty files and symlinks; no network, no skips."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from fs_model_root import (
    LAYOUT_NESTED,
    LAYOUT_NESTED_OVERLAY,
    LAYOUT_OVERLAY,
    LAYOUT_SELF_CONTAINED,
    ModelRootError,
    describe,
    resolve_model_root,
)

SHA = "0e9e39f2" + "a" * 32  # a 40-hex commit directory spelling


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def _real(path: Path) -> str:
    return os.path.realpath(path)


def test_t1_self_contained(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    _touch(root / "config.json")
    mr = resolve_model_root(root)
    assert mr.layout == LAYOUT_SELF_CONTAINED
    assert mr.bind_closure == (_real(root),)
    assert mr.symlinks_total == 0
    assert mr.symlinks_escaping == 0
    assert mr.config_dir == _real(root)
    assert mr.config_candidates == 1


def test_t2_commit_nested(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _touch(root / SHA / "config.json")
    _touch(root / SHA / "model-00001-of-00002.safetensors")
    mr = resolve_model_root(root)
    assert mr.layout == LAYOUT_NESTED
    assert mr.config_dir == _real(root / SHA)
    assert mr.bind_closure == (_real(root),)
    assert mr.symlinks_total == 0
    assert mr.symlinks_escaping == 0


def test_t3_config_overlay(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    weights = tmp_path / "weights"
    _touch(root / "config.json")
    for i in range(3):
        shard = weights / f"shard{i}.safetensors"
        _touch(shard)
        os.symlink(_real(shard), root / f"shard{i}.safetensors")
    mr = resolve_model_root(root)
    assert mr.layout == LAYOUT_OVERLAY
    assert mr.bind_closure == (_real(root), _real(weights))
    assert len(mr.bind_closure) == 2
    assert mr.symlinks_escaping == 3
    assert mr.symlinks_total == 3


def test_t4_commit_nested_overlay(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _touch(root / SHA / "config.json")
    weights = tmp_path / "weights"
    payload = weights / "model.safetensors"
    _touch(payload)
    os.symlink(_real(payload), root / SHA / "model.safetensors")
    mr = resolve_model_root(root)
    assert mr.layout == LAYOUT_NESTED_OVERLAY
    assert mr.config_dir == _real(root / SHA)
    assert mr.bind_closure == (_real(root), _real(weights))
    assert mr.symlinks_escaping == 1
    assert mr.symlinks_total == 1


def test_t5a_shallowest_wins_over_nested_variant(tmp_path: Path) -> None:
    # MEASURED on the estate: stock upstream checkpoints ship a nested variant dir
    # beside an unambiguous root config (gpt-oss-20b `original/`, sentence-transformers
    # `1_Pooling/`). Refusing these is a false refusal on a legitimate checkpoint.
    root = tmp_path / "model"
    _touch(root / "config.json")
    _touch(root / "original" / "config.json")
    mr = resolve_model_root(root)
    assert mr.layout == LAYOUT_SELF_CONTAINED
    assert mr.config_dir == _real(root)
    assert mr.config_candidates == 1


def test_t5b_ambiguity_at_shallowest_depth_refuses_with_both_denominators(
        tmp_path: Path) -> None:
    # Genuine ambiguity: NO config at the root, several one level down. Both
    # denominators must appear -- the count at the chosen depth AND the subtree total --
    # so that narrowing to the shallowest depth stays visible rather than silently
    # discarding the rest of the subtree.
    root = tmp_path / "family"
    _touch(root / "a" / "config.json")
    _touch(root / "b" / "config.json")
    _touch(root / "a" / "deeper" / "config.json")
    with pytest.raises(ModelRootError) as excinfo:
        resolve_model_root(root)
    msg = str(excinfo.value)
    # Assert the full phrase, never a bare number: "2" also occurs in "depth <=2".
    assert "found 2 config.json candidates at depth 1" in msg
    assert "(3 in the subtree at depth <=2)" in msg
    assert "guess" in msg


def test_t6_emptiness_refuses_as_unmeasured(tmp_path: Path) -> None:
    root = tmp_path / "model"
    (root / "deep" / "deeper").mkdir(parents=True)
    _touch(root / "deep" / "deeper" / "config.json")  # beyond max_depth=2? keep a stray file
    (root / "deep" / "deeper" / "config.json").unlink()
    with pytest.raises(ModelRootError) as excinfo:
        resolve_model_root(root)
    msg = str(excinfo.value)
    assert "0" in msg
    assert "UNMEASURED" in msg


def test_t7_broken_symlink_inside_root_is_corruption(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    _touch(root / "config.json")
    os.symlink("missing-payload.st", root / "broken.st")
    with pytest.raises(ModelRootError) as excinfo:
        resolve_model_root(root)
    msg = str(excinfo.value)
    assert "corrupt" in msg
    assert "broken.st" in msg


def test_t8_broken_symlink_outside_root_is_escaping_bind(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    _touch(root / "config.json")
    target = tmp_path / "elsewhere" / "model.safetensors"  # deliberately absent
    os.symlink(str(target), root / "model.safetensors")
    mr = resolve_model_root(root)
    assert mr.layout == LAYOUT_OVERLAY
    assert mr.symlinks_escaping == 1
    assert mr.symlinks_total == 1
    assert mr.bind_closure == (_real(root), str(target.parent))


def test_t9_describe_leaks_no_absolute_path(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    _touch(root / "config.json")
    mr = resolve_model_root(root)
    line = describe(mr)
    assert str(tmp_path) not in line
    assert str(root) not in line
    assert line.startswith("model-root: layout=self-contained")
    assert "config=config.json" in line
    assert "binds=1" in line
    assert "symlinks=0/0 escaping" in line


def test_t10_determinism(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    weights = tmp_path / "weights"
    _touch(root / "config.json")
    for i in range(2):
        shard = weights / f"s{i}.st"
        _touch(shard)
        os.symlink(_real(shard), root / f"s{i}.st")
    first = resolve_model_root(root)
    second = resolve_model_root(root)
    assert first == second


def test_not_a_directory_refuses(tmp_path: Path) -> None:
    bogus = tmp_path / "no-such-dir"
    with pytest.raises(ModelRootError) as excinfo:
        resolve_model_root(bogus)
    assert str(bogus) in str(excinfo.value) or "realpath" in str(excinfo.value).lower() or "does not exist" in str(excinfo.value)
