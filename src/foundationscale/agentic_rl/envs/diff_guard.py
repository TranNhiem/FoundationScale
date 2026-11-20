"""DiffGuard: detect edits to files that change how a hidden test suite runs (ARL-G14).

The hidden tests are restored before ``verify()`` runs them, so an agent cannot win by editing a
hidden test. It CAN win by editing a file pytest or the interpreter loads that is NOT hidden: a
``conftest.py`` hook that rewrites outcomes, ``pytest.ini``/``setup.cfg``/``pyproject.toml``/
``tox.ini`` options, or a ``sitecustomize.py``/``usercustomize.py``/``*.pth`` file in a
site-packages directory. This module fingerprints exactly those files and reports which ones the
episode added, removed or changed.

The fingerprint is taken from the HOST side of the sandbox (``host_path``), never by running a
command inside the container, so nothing the agent left running or rewrote in the container can
influence the measurement. Symlinks are recorded by their target string and never followed (a link
out of the rootfs is never read).

WHAT IS CLAIMED: the listed path classes are compared byte-for-byte (sha256) between the end of
``start()`` and the beginning of ``verify()``. WHAT IS NOT CLAIMED: that this catches every way to
influence a test run (an edit to the code under test that special-cases the test is the task
itself, and is out of reach by construction); the guarded set is a declared floor.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

# Base names pytest or the interpreter loads that can change test outcomes when edited.
GUARDED_NAMES: frozenset[str] = frozenset(
    {
        "conftest.py",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        "pyproject.toml",
        "sitecustomize.py",
        "usercustomize.py",
    }
)
# Suffixes guarded inside site-packages directories only (a *.pth runs at interpreter start).
GUARDED_SITE_SUFFIXES: tuple[str, ...] = (".pth",)
# Site-packages locations, relative to the container rootfs, globbed per run.
SITE_GLOBS: tuple[str, ...] = (
    "usr/lib/python3*/site-packages",
    "usr/lib/python3*/dist-packages",
    "usr/local/lib/python3*/site-packages",
    "usr/local/lib/python3*/dist-packages",
    "opt/conda/lib/python3*/site-packages",
    "root/.local/lib/python3*/site-packages",
)
# Never descended into (VCS metadata and caches are not loaded by a test run).
_SKIP_DIRS: frozenset[str] = frozenset({".git", "__pycache__", ".pytest_cache", "node_modules"})


def _digest(path: Path) -> str:
    if path.is_symlink():
        return "symlink:" + str(path.readlink())
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _walk(root: Path, base: Path, *, site: bool, out: dict[str, str]) -> None:
    if not base.is_dir() or base.is_symlink():
        return
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            if name in GUARDED_NAMES or (site and name.endswith(GUARDED_SITE_SUFFIXES)):
                path = Path(dirpath) / name
                out["/" + str(path.relative_to(root))] = _digest(path)


def fingerprint(rootfs: Path, workdir: str) -> dict[str, str]:
    """``{container_path: digest}`` for every guarded file under ``workdir`` and the site dirs.

    ``rootfs`` is the host path of the container's ``/``; ``workdir`` is a container path.
    """
    root = Path(rootfs)
    out: dict[str, str] = {}
    _walk(root, root / workdir.lstrip("/"), site=False, out=out)
    for pattern in SITE_GLOBS:
        for site_dir in sorted(root.glob(pattern)):
            _walk(root, site_dir, site=True, out=out)
    return out


def changed_paths(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """Sorted container paths added, removed or changed between two fingerprints."""
    keys: Iterable[str] = set(before) | set(after)
    return sorted(k for k in keys if before.get(k) != after.get(k))
