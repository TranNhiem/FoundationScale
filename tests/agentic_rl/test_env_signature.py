from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl.envs import env_signature
from foundationscale.agentic_rl.envs.env_signature import (
    EnvSignatureRefusal,
    compute_signature,
    image_digest,
    tree_manifest,
)


@pytest.mark.parametrize("content", [False, True])
def test_tree_manifest_is_deterministic_and_ignores_timestamps(
    tmp_path: Path, content: bool
) -> None:
    root = tmp_path / "rootfs"
    root.mkdir()
    (root / "etc").mkdir()
    file_path = root / "etc" / "config"
    file_path.write_bytes(b"configuration")

    first = tree_manifest(root, content=content)
    second = tree_manifest(root, content=content)
    assert first == second

    os.utime(file_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(root, ns=(2_000_000_000, 2_000_000_000))
    assert tree_manifest(root, content=content) == first


def test_tree_manifest_file_content_and_size_semantics(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    root.mkdir()
    file_path = root / "payload"
    file_path.write_bytes(b"aa")

    manifest_before = tree_manifest(root, content=False)
    content_before = tree_manifest(root, content=True)

    file_path.write_bytes(b"bb")
    assert tree_manifest(root, content=False) == manifest_before
    assert tree_manifest(root, content=True) != content_before

    file_path.write_bytes(b"ccc")
    assert tree_manifest(root, content=False) != manifest_before


def test_tree_manifest_includes_permissions(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    root.mkdir()
    file_path = root / "executable"
    file_path.write_bytes(b"payload")
    file_path.chmod(0o640)
    before = tree_manifest(root, content=False)

    file_path.chmod(0o600)
    assert tree_manifest(root, content=False) != before


def test_tree_manifest_records_symlink_target_without_following_it(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    root.mkdir()
    outside_a = tmp_path / "outside-a"
    outside_b = tmp_path / "outside-b"
    outside_a.mkdir()
    outside_b.mkdir()
    outside_file = outside_a / "secret"
    outside_file.write_bytes(b"first secret")
    (outside_b / "secret").write_bytes(b"second secret")

    link = root / "external"
    link.symlink_to(outside_a, target_is_directory=True)
    before = tree_manifest(root, content=True)

    outside_file.write_bytes(b"changed outside the root")
    (outside_a / "new-file").write_bytes(b"also outside")
    assert tree_manifest(root, content=True) == before

    link.unlink()
    link.symlink_to(outside_b, target_is_directory=True)
    assert tree_manifest(root, content=True) != before


def test_tree_manifest_skips_top_level_and_nested_excludes(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    (root / "proc" / "pid").mkdir(parents=True)
    (root / "a" / "b").mkdir(parents=True)
    (root / "proc" / "pid" / "state").write_bytes(b"running")
    (root / "a" / "b" / "secret").write_bytes(b"hidden")
    kept = root / "a" / "kept"
    kept.write_bytes(b"visible")
    excludes = ("proc", "a/b")

    before = tree_manifest(root, content=True, exclude=excludes)
    unfiltered_before = tree_manifest(root, content=True)
    assert before != unfiltered_before

    (root / "proc" / "pid" / "state").write_bytes(b"stopped")
    (root / "proc" / "new").write_bytes(b"ignored")
    (root / "a" / "b" / "secret").write_bytes(b"changed hidden data")
    (root / "a" / "b" / "new").write_bytes(b"also ignored")
    assert tree_manifest(root, content=True, exclude=excludes) == before
    assert tree_manifest(root, content=True) != unfiltered_before

    kept.write_bytes(b"visible and changed")
    assert tree_manifest(root, content=True, exclude=excludes) != before


def test_tree_manifest_refuses_non_directory_roots(tmp_path: Path) -> None:
    file_path = tmp_path / "file"
    file_path.write_bytes(b"not a directory")

    with pytest.raises(EnvSignatureRefusal, match="root"):
        tree_manifest(file_path, content=False)
    with pytest.raises(EnvSignatureRefusal, match="root"):
        tree_manifest(tmp_path / "missing", content=True)


def test_image_digest_hashes_bytes_and_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "image.tar"
    payload = b"a" * (1024 * 1024) + b"tail"
    image.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert image_digest(str(image)) == expected

    def fail_digest(path: object) -> str:
        raise AssertionError("cached image was re-read")

    monkeypatch.setattr(env_signature, "_digest_file", fail_digest)
    assert image_digest(str(image)) == expected


def test_image_digest_cache_invalidates_on_size_and_mtime_changes(tmp_path: Path) -> None:
    image = tmp_path / "image.tar"

    image.write_bytes(b"first")
    assert image_digest(str(image)) == hashlib.sha256(b"first").hexdigest()

    image.write_bytes(b"second")
    assert image_digest(str(image)) == hashlib.sha256(b"second").hexdigest()

    previous_stat = image.stat()
    image.write_bytes(b"THIRD!")
    os.utime(
        image,
        ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000),
    )
    assert image_digest(str(image)) == hashlib.sha256(b"THIRD!").hexdigest()


def signature_kwargs() -> dict[str, Any]:
    return {
        "mode": "manifest",
        "rootfs_manifest": "rootfs-digest",
        "image": "image-digest",
        "workdir": "/workspace",
        "test_cmd": "pytest -q",
        "hidden_paths": ["tests/z_hidden", "tests/a_hidden"],
        "setup_commands": ["python setup.py", "pytest --collect-only"],
        "hidden_manifest": "hidden-digest",
    }


def test_compute_signature_is_stable_and_hidden_path_order_insensitive() -> None:
    kwargs = signature_kwargs()
    signature = compute_signature(**kwargs)

    assert signature == compute_signature(**kwargs)
    assert signature.startswith("envsig_")
    assert len(signature) == len("envsig_") + 32

    reordered = dict(kwargs)
    reordered["hidden_paths"] = list(reversed(kwargs["hidden_paths"]))
    assert compute_signature(**reordered) == signature


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("mode", "content"),
        ("rootfs_manifest", "other-rootfs-digest"),
        ("image", None),
        ("workdir", "/other-workdir"),
        ("test_cmd", "make test"),
        ("hidden_paths", ["tests/a_hidden"]),
        ("setup_commands", ["different setup"]),
        ("hidden_manifest", None),
    ],
)
def test_compute_signature_differs_for_each_changed_field(field: str, changed_value: Any) -> None:
    kwargs = signature_kwargs()
    baseline = compute_signature(**kwargs)
    changed = dict(kwargs)
    changed[field] = changed_value

    assert compute_signature(**changed) != baseline


def test_compute_signature_setup_command_order_is_significant() -> None:
    kwargs = signature_kwargs()
    baseline = compute_signature(**kwargs)
    reordered = dict(kwargs)
    reordered["setup_commands"] = list(reversed(kwargs["setup_commands"]))

    assert compute_signature(**reordered) != baseline


@pytest.mark.parametrize("mode", ["off", "invalid", ""])
def test_compute_signature_refuses_bad_modes(mode: str) -> None:
    kwargs = signature_kwargs()
    kwargs["mode"] = mode

    with pytest.raises(EnvSignatureRefusal, match="mode"):
        compute_signature(**kwargs)


def test_tree_manifest_excludes_a_file_by_name(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("k")
    before = tree_manifest(tmp_path, content=True, exclude=("lock",))
    (tmp_path / "lock").write_text("pid")
    assert tree_manifest(tmp_path, content=True, exclude=("lock",)) == before


def test_tree_manifest_records_a_fifo_as_other(tmp_path: Path) -> None:
    before = tree_manifest(tmp_path, content=False)
    os.mkfifo(tmp_path / "pipe")
    assert tree_manifest(tmp_path, content=False) != before


def test_tree_manifest_propagates_a_walk_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_walk(*args: object, onerror: Any = None, **kwargs: object) -> list[object]:
        onerror(PermissionError("denied"))
        return []

    monkeypatch.setattr(env_signature.os, "walk", failing_walk)
    with pytest.raises(PermissionError, match="denied"):
        tree_manifest(tmp_path, content=False)


def test_compute_signature_includes_the_execution_env() -> None:
    kwargs = signature_kwargs()
    plain = compute_signature(**kwargs)
    assert compute_signature(**kwargs, env={}) == plain
    assert compute_signature(**kwargs, env={"MODE": "a"}) != compute_signature(
        **kwargs, env={"MODE": "b"}
    )
    assert compute_signature(**kwargs, env={"A": "1", "B": "2"}) == compute_signature(
        **kwargs, env={"B": "2", "A": "1"}
    )
