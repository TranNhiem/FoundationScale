"""Hidden-path vaulting for black-box agents (BINDING API section B, threats 1-3).

Stdlib only. Works against any object exposing the ``EnrootSandbox`` surface used
here (``host_path`` and ``exec``), so tests can substitute a fake over a tmp dir.
"""

from __future__ import annotations

import math
import os
import shlex
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from foundationscale.agentic_rl.integrity import IntegrityRefusal

_DEFAULT_PATHS: tuple[str, ...] = ("/tests", "/solution")
_MAX_ENTRIES = 4096
_MAX_DEPTH = 256


class ExecResult(Protocol):
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool


class SandboxLike(Protocol):
    def host_path(self, container_path: str) -> Path: ...

    def exec(
        self,
        command: str,
        *,
        timeout_s: float | None = None,
    ) -> ExecResult: ...


@dataclass(frozen=True)
class HiddenSpec:
    """Container paths hidden during the agent phase.

    ``paths`` are absolute container paths. ``git_worktrees`` are absolute
    container paths whose ``<wt>/.git`` is hidden too.
    """

    paths: tuple[str, ...] = _DEFAULT_PATHS
    git_worktrees: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, values in (("paths", self.paths), ("git_worktrees", self.git_worktrees)):
            if not isinstance(values, tuple):
                raise IntegrityRefusal(
                    f"{name}: expected tuple[str, ...], got {type(values).__name__}"
                )
            for value in values:
                if not isinstance(value, str) or not value.startswith("/") or value == "/":
                    raise IntegrityRefusal(
                        f"{name}: expected absolute container paths (not '/'), got {value!r}"
                    )
                if value.rstrip("/") != value:
                    raise IntegrityRefusal(
                        f"{name}: expected paths without trailing '/', got {value!r}"
                    )
        if len(set(self.paths)) != len(self.paths):
            raise IntegrityRefusal(f"paths: expected unique entries, got {self.paths!r}")
        if len(set(self.git_worktrees)) != len(self.git_worktrees):
            raise IntegrityRefusal(
                f"git_worktrees: expected unique entries, got {self.git_worktrees!r}"
            )
        hidden = self.hidden_container_paths()
        if len(set(hidden)) != len(hidden):
            raise IntegrityRefusal(
                f"hidden paths: expected unique entries, got {hidden!r} "
                f"({len(hidden)} entries, {len(set(hidden))} unique)"
            )

    def hidden_container_paths(self) -> tuple[str, ...]:
        """All container paths hidden by this spec, in a stable order."""
        entries: list[str] = list(self.paths)
        for worktree in self.git_worktrees:
            entries.append(worktree + "/.git")
        return tuple(entries)


def _is_within(child: Path, parent: Path) -> bool:
    """True when ``child`` is ``parent`` or lives under it (both resolved)."""
    if child == parent:
        return True
    return parent in child.parents


def _resolved(path: Path) -> Path:
    return Path(os.path.realpath(path))


class HiddenVault:
    """Move hidden container paths out of the sandbox rootfs and back again."""

    def __init__(
        self,
        sandbox: SandboxLike,
        spec: HiddenSpec,
        vault_dir: str | Path,
    ) -> None:
        if not isinstance(spec, HiddenSpec):
            raise IntegrityRefusal(f"spec: expected HiddenSpec, got {type(spec).__name__}")
        self._sandbox = sandbox
        self._spec = spec
        self._vault_dir = Path(vault_dir)
        self._root = _resolved(self._sandbox.host_path("/"))
        self._sealed = False
        self._moved: dict[str, Path] = {}
        self._check_vault_location()

    # -- validation --------------------------------------------------------

    def _check_vault_location(self) -> None:
        vault = _resolved(self._vault_dir)
        if _is_within(vault, self._root):
            raise IntegrityRefusal(
                f"vault_dir: expected a host path outside the sandbox rootfs {self._root}, "
                f"got {vault} inside it"
            )

    def _host(self, container_path: str) -> Path:
        if not isinstance(container_path, str) or not container_path.startswith("/"):
            raise IntegrityRefusal(
                f"container_path: expected absolute container path, got {container_path!r}"
            )
        if ".." in Path(container_path).parts:
            raise IntegrityRefusal(
                f"container_path: expected path without '..' parts, got {container_path!r}"
            )
        return self._root / container_path.lstrip("/")

    def _check_ancestors(self, loc: Path) -> None:
        """Refuse when any ancestor of ``loc`` (exclusive of root) is a symlink."""
        if not _is_within(loc, self._root):
            raise IntegrityRefusal(
                f"hidden path: expected host path inside {self._root}, got {loc}"
            )
        current = loc.parent
        while current != self._root:
            if Path(current).is_symlink():
                raise IntegrityRefusal(
                    f"hidden path ancestor is a symlink: {current} (for {loc}, under {self._root})"
                )
            if current.parent == current:
                break
            current = current.parent

    def _vault_target(self, container_path: str) -> Path:
        rel = container_path.lstrip("/")
        return self._vault_dir / rel

    # -- lifecycle ---------------------------------------------------------

    def seal(self) -> tuple[str, ...]:
        """Move each existing hidden path into the vault; return moved container paths."""
        if self._sealed:
            raise IntegrityRefusal(
                f"seal: expected unsealed vault, got already-sealed "
                f"({len(self._moved)} of {len(self._spec.hidden_container_paths())} paths moved)"
            )
        self._check_vault_location()
        self._vault_dir.mkdir(parents=True, exist_ok=True)
        moved: list[str] = []
        for container_path in self._spec.hidden_container_paths():
            source = self._host(container_path)
            self._check_ancestors(source)
            if not os.path.lexists(source):
                continue  # missing paths are skipped, not an error
            if Path(source).is_symlink():
                raise IntegrityRefusal(f"hidden path is a symlink: {container_path!r} -> {source}")
            target = self._vault_target(container_path)
            if os.path.lexists(target):
                raise IntegrityRefusal(
                    f"seal: vault target already exists for {container_path!r}: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            self._moved[container_path] = target
            moved.append(container_path)
        self._sealed = True
        return tuple(moved)

    def assert_sealed(self) -> None:
        """MUST_FIRE when any hidden path exists in the sandbox."""
        present = [
            path
            for path in self._spec.hidden_container_paths()
            if os.path.lexists(self._host(path))
        ]
        if present:
            raise IntegrityRefusal(
                f"hidden paths present in sandbox: {len(present)} of "
                f"{len(self._spec.hidden_container_paths())} hidden paths exist: {present!r}"
            )
        if not self._sealed:
            raise IntegrityRefusal(
                f"assert_sealed: expected sealed vault, got unsealed "
                f"(0 of {len(self._spec.hidden_container_paths())} paths moved)"
            )

    def unseal(self) -> tuple[str, ...]:
        """Move everything back for the verifier phase."""
        if not self._sealed:
            raise IntegrityRefusal(
                f"unseal: expected sealed vault, got unsealed "
                f"(0 of {len(self._spec.hidden_container_paths())} paths moved)"
            )
        restored: list[str] = []
        for container_path in self._spec.hidden_container_paths():
            target = self._host(container_path)
            self._check_ancestors(target)
            if os.path.lexists(target):
                raise IntegrityRefusal(
                    f"agent wrote to hidden path: {container_path!r} exists in the sandbox "
                    f"({len(restored)} of {len(self._moved)} vaulted paths restored)"
                )
            source = self._moved.get(container_path)
            if source is None:
                continue
            if not os.path.lexists(source):
                raise IntegrityRefusal(
                    f"unseal: vault entry missing for {container_path!r}: {source} "
                    f"({len(restored)} of {len(self._moved)} vaulted paths restored)"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.parent / f"{target.name}.fs-unseal-{uuid.uuid4().hex}"
            shutil.move(str(source), str(temp))
            try:
                Path(temp).rename(target)
            except OSError as exc:
                raise IntegrityRefusal(
                    f"agent wrote to hidden path: {container_path!r} appeared in the sandbox "
                    f"during unseal ({exc}); temp left in place for forensics: {temp} "
                    f"({len(restored)} of {len(self._moved)} vaulted paths restored)"
                ) from exc
            restored.append(container_path)
        self._sealed = False
        self._moved.clear()
        return tuple(restored)


def probe_egress(
    sandbox: SandboxLike,
    *,
    host: str = "example.com",
    timeout_s: float = 10.0,
) -> bool:
    """True iff the sandbox can resolve or connect to ``host``."""
    if not isinstance(host, str) or not host:
        raise IntegrityRefusal(f"host: expected non-empty str, got {host!r}")
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        raise IntegrityRefusal(f"timeout_s: expected finite number > 0, got {timeout_s!r}")
    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
        raise IntegrityRefusal(f"timeout_s: expected finite number > 0, got {timeout_s!r}")
    command = (
        f"getent hosts {shlex.quote(host)} >/dev/null 2>&1 || "
        f"python3 -c 'import socket;socket.create_connection(({host!r},443),5)' "
        f">/dev/null 2>&1"
    )
    result = sandbox.exec(command, timeout_s=float(timeout_s))
    return result.return_code == 0


def require_no_egress(
    sandbox: SandboxLike,
    *,
    host: str = "example.com",
    timeout_s: float = 10.0,
) -> None:
    """MUST_FIRE when the sandbox can reach the network."""
    if probe_egress(sandbox, host=host, timeout_s=timeout_s):
        raise IntegrityRefusal(
            f"egress probe succeeded for host {host!r}: 1 of 1 probes reached the network"
        )
