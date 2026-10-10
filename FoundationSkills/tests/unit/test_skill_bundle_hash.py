"""skill_bundle_sha256 binds a hash-bound report to the whole skill bundle: SKILL.md plus every file
under references/ scripts/ and assets/, so an edited reference file makes the report stale instead of
leaving it looking fresh. A skill without such directories keeps the plain sha256(SKILL.md) binding."""
from __future__ import annotations

import hashlib
import importlib.resources
import re
from pathlib import Path

from foundationskills.agent import routing_eval

HEX64 = re.compile(r"^[0-9a-f]{64}$")

SKILL_MD = b"---\nname: demo\n---\nDemo skill.\n"


def _put(root: Path, relpath: str, data: bytes) -> Path:
    """Write one bundle file under root (parents created) and return its path."""
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _skill_root(tmp_path: Path) -> Path:
    """A minimal skill package whose bundle is SKILL.md only."""
    root = tmp_path / "demo_pkg"
    root.mkdir()
    _put(root, "SKILL.md", SKILL_MD)
    return root


def test_only_skill_md_hashes_the_skill_md_bytes(tmp_path: Path):
    root = _skill_root(tmp_path)
    assert routing_eval.skill_bundle_sha256(root) == hashlib.sha256(SKILL_MD).hexdigest()


def test_add_edit_rename_of_a_reference_file_all_change_the_hash(tmp_path: Path):
    root = _skill_root(tmp_path)
    bound = routing_eval.skill_bundle_sha256(root)
    _put(root, "references/a.md", b"one\n")
    added = routing_eval.skill_bundle_sha256(root)
    assert added != bound
    _put(root, "references/a.md", b"two\n")
    edited = routing_eval.skill_bundle_sha256(root)
    assert edited != added
    (root / "references" / "a.md").rename(root / "references" / "renamed.md")
    assert routing_eval.skill_bundle_sha256(root) != edited


def test_files_outside_the_bundle_dirs_do_not_change_the_hash(tmp_path: Path):
    root = _skill_root(tmp_path)
    bound = routing_eval.skill_bundle_sha256(root)
    _put(root, "evals/evals.json", b"{}\n")
    _put(root, "skill.py", b"VALUE = 1\n")
    assert routing_eval.skill_bundle_sha256(root) == bound


def test_pycache_directories_and_pyc_files_are_ignored(tmp_path: Path):
    root = _skill_root(tmp_path)
    _put(root, "references/a.md", b"one\n")
    bound = routing_eval.skill_bundle_sha256(root)
    _put(root, "references/__pycache__/a.cpython-310.pyc", b"\x00binary\n")
    _put(root, "scripts/__pycache__/run.cpython-310.pyc", b"\x00binary\n")
    _put(root, "references/a.pyc", b"\x00binary\n")
    assert routing_eval.skill_bundle_sha256(root) == bound


def test_hash_does_not_depend_on_file_creation_order(tmp_path: Path):
    first, second = tmp_path / "first_pkg", tmp_path / "second_pkg"
    for root in (first, second):
        root.mkdir()
        _put(root, "SKILL.md", SKILL_MD)
    _put(first, "references/a.md", b"a\n")
    _put(first, "references/sub/b.md", b"b\n")
    _put(first, "scripts/run.py", b"print()\n")
    _put(second, "scripts/run.py", b"print()\n")
    _put(second, "references/sub/b.md", b"b\n")
    _put(second, "references/a.md", b"a\n")
    assert routing_eval.skill_bundle_sha256(first) == routing_eval.skill_bundle_sha256(second)


def test_nested_reference_files_are_part_of_the_hash(tmp_path: Path):
    root = _skill_root(tmp_path)
    _put(root, "references/sub/b.md", b"nested\n")
    nested = routing_eval.skill_bundle_sha256(root)
    assert nested != hashlib.sha256(SKILL_MD).hexdigest()
    _put(root, "references/sub/b.md", b"nested edit\n")
    assert routing_eval.skill_bundle_sha256(root) != nested


def test_every_real_package_binds_skill_md_only_today():
    """No skill package ships references/ scripts/ assets/ yet: the bundle hash must still equal the
    SKILL.md hash, so every report published before A1.3 stays valid without regeneration."""
    for entry in routing_eval.load_index():
        assert HEX64.fullmatch(entry["skill_md_sha256"]), entry["package"]
        assert HEX64.fullmatch(entry["skill_bundle_sha256"]), entry["package"]
        root = importlib.resources.files(str(entry["package"]))
        if not any(root.joinpath(name).is_dir() for name in routing_eval.BUNDLE_DIRS):
            assert entry["skill_bundle_sha256"] == entry["skill_md_sha256"]
