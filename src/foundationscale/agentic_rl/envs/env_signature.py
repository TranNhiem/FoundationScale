"""Build deterministic environment signatures from filesystem and task inputs.

Tree manifests compare entry names, kinds, permission bits, regular-file sizes, and
symlink targets; content mode additionally hashes regular-file bytes. Timestamps are
intentionally omitted and symlinks are never followed, so clean starts do not differ
merely because of time. Image hashing is cached by real path, size, and mtime.

These signatures are comparison aids, not security boundaries or proofs of isolation
or reproducibility. Excluded paths and state not represented by the supplied inputs
remain unmeasured. Callers decide how to handle I/O errors and unmeasured environments.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import NoReturn

ENV_SIGNATURE_SCHEMA: str = "fs.env_signature/v1"
SIGNATURE_MODES: tuple[str, ...] = ("off", "manifest", "content")
DEFAULT_EXCLUDES: tuple[str, ...] = ("proc", "sys", "dev", "run", "tmp")

_HASH_BLOCK_SIZE: int = 1024 * 1024
_IMAGE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


class EnvSignatureRefusal(ValueError):
    """Refuse an environment-signature request with an invalid input."""


def _raise_walk_error(error: OSError) -> NoReturn:
    raise error


def _digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(_HASH_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _exclude_components(exclude: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    components: list[tuple[str, ...]] = []
    for item in exclude:
        parts = PurePosixPath(item).parts
        if parts:
            components.append(parts)
    return tuple(components)


def _is_excluded(
    relative: PurePosixPath,
    excludes: tuple[tuple[str, ...], ...],
) -> bool:
    parts = relative.parts
    return any(parts[: len(prefix)] == prefix for prefix in excludes)


def tree_manifest(root: Path, *, content: bool, exclude: Iterable[str] = ()) -> str:
    """Return a deterministic manifest digest for the entries beneath ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise EnvSignatureRefusal(
            f"field 'root' has value {str(root)!r}; legal set: existing directory paths"
        )

    excludes = _exclude_components(exclude)
    records: list[str] = []

    for dirpath, dirnames, filenames in os.walk(
        root,
        topdown=True,
        onerror=_raise_walk_error,
        followlinks=False,
    ):
        directory = Path(dirpath)
        dirnames.sort()
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_excluded(
                PurePosixPath((directory / name).relative_to(root).as_posix()),
                excludes,
            )
        ]

        for name in dirnames + filenames:
            entry = directory / name
            relative = PurePosixPath(entry.relative_to(root).as_posix())
            if _is_excluded(relative, excludes):
                continue

            metadata = entry.lstat()
            entry_mode = metadata.st_mode
            size_or_dash = "-"
            extra = ""

            if stat.S_ISLNK(entry_mode):
                kind = "symlink"
                extra = str(entry.readlink())
            elif stat.S_ISDIR(entry_mode):
                kind = "dir"
            elif stat.S_ISREG(entry_mode):
                kind = "file"
                size_or_dash = str(metadata.st_size)
                if content:
                    extra = _digest_file(entry)
            else:
                kind = "other"

            records.append(
                f"{relative.as_posix()}\0{kind}\0{entry_mode & 0o7777:o}\0{size_or_dash}\0{extra}"
            )

    serialized = "\n".join(sorted(records)).encode("utf-8", "surrogateescape")
    return hashlib.sha256(serialized).hexdigest()


def image_digest(path: str) -> str:
    """Return a streamed SHA-256 digest, caching unchanged images per process."""
    real_path = os.path.realpath(path)
    metadata = Path(real_path).stat()
    key = (real_path, metadata.st_size, metadata.st_mtime_ns)
    cached = _IMAGE_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached

    digest = _digest_file(real_path)
    _IMAGE_DIGEST_CACHE[key] = digest
    return digest


def compute_signature(
    *,
    mode: str,
    rootfs_manifest: str,
    image: str | None,
    workdir: str,
    test_cmd: str,
    hidden_paths: Sequence[str],
    setup_commands: Sequence[str],
    hidden_manifest: str | None,
) -> str:
    """Hash the canonical task and environment inputs into a compact signature."""
    legal_modes = ("manifest", "content")
    if mode not in legal_modes:
        raise EnvSignatureRefusal(f"field 'mode' has value {mode!r}; legal set: {legal_modes!r}")

    payload: dict[str, object] = {
        "schema": ENV_SIGNATURE_SCHEMA,
        "mode": mode,
        "rootfs": rootfs_manifest,
        "image": image,
        "workdir": workdir,
        "test_cmd": test_cmd,
        "hidden_paths": sorted(hidden_paths),
        "setup_commands": list(setup_commands),
        "hidden": hidden_manifest,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return "envsig_" + digest[:32]
