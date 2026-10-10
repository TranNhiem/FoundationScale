"""Enroot-backed sandboxes for black-box agents.

Stdlib only. Every external command goes through an injectable ``Runner`` so
tests can substitute a fake.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,63}$")
_NETWORKS = ("no-network", "public")
_STDERR_TAIL = 2000


class SandboxRefusal(ValueError):
    """Bad input or unsupported request; the message names expected vs got."""


class SandboxError(RuntimeError):
    """Infrastructure failure while talking to enroot."""


class SandboxBuildFailed(SandboxError):
    """Non-retryable build failure naming the failing Dockerfile line."""


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class Runner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunResult: ...


class SubprocessRunner:
    """``subprocess.run`` based runner; timeouts kill the process group."""

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunResult:
        run_env: dict[str, str] | None = None
        if env is not None:
            run_env = dict(os.environ)
            run_env.update(env)
        try:
            proc = subprocess.Popen(  # noqa: S603
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=run_env,
                start_new_session=True,
            )
        except OSError as exc:
            raise SandboxError(f"failed to spawn {list(argv)!r}: {exc}") from exc
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
            return RunResult(returncode=124, stdout=stdout, stderr=stderr, timed_out=True)
        return RunResult(returncode=proc.returncode, stdout=stdout, stderr=stderr)


@dataclass(frozen=True)
class SandboxSpec:
    name: str
    image: str
    data_root: str
    image_store: str
    scratch_root: str
    network: str = "no-network"
    cpu_set: tuple[int, ...] | None = None
    memory_mb: int | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    workdir: str = "/"

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise SandboxRefusal(f"name: expected [A-Za-z0-9_.-]{{1,63}}, got {self.name!r}")
        for field_name in ("data_root", "image_store", "scratch_root"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise SandboxRefusal(f"{field_name}: expected absolute path, got {value!r}")
        if self.network not in _NETWORKS:
            raise SandboxRefusal(f"network: expected one of {_NETWORKS}, got {self.network!r}")
        if self.cpu_set is not None:
            if not isinstance(self.cpu_set, tuple) or not self.cpu_set:
                raise SandboxRefusal(
                    f"cpu_set: expected non-empty tuple[int, ...] | None, got {self.cpu_set!r}"
                )
            for cpu in self.cpu_set:
                if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0:
                    raise SandboxRefusal(f"cpu_set: expected ints >= 0, got {self.cpu_set!r}")
        if self.memory_mb is not None:
            if isinstance(self.memory_mb, bool) or not isinstance(self.memory_mb, int):
                raise SandboxRefusal(f"memory_mb: expected int >= 1 | None, got {self.memory_mb!r}")
            if self.memory_mb < 1:
                raise SandboxRefusal(f"memory_mb: expected int >= 1, got {self.memory_mb!r}")
        if not isinstance(self.env, Mapping):
            raise SandboxRefusal(f"env: expected Mapping[str, str], got {type(self.env).__name__}")
        for key, value in self.env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SandboxRefusal(f"env: expected Mapping[str, str], got {key!r}: {value!r}")
        if not isinstance(self.workdir, str) or not self.workdir:
            raise SandboxRefusal(f"workdir: expected non-empty str, got {self.workdir!r}")
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False


class EnrootSandbox:
    def __init__(self, spec: SandboxSpec, *, runner: Runner | None = None) -> None:
        self.spec = spec
        self._runner: Runner = runner if runner is not None else SubprocessRunner()
        self._started = False
        self._image_path: str | None = None
        self._build_env: Mapping[str, str] = MappingProxyType({})
        self._build_workdir: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def image_path(self) -> str:
        if self._image_path is not None:
            return self._image_path
        image = self.spec.image
        if image.endswith(".sqsh"):
            if not Path(image).exists():
                raise SandboxRefusal(
                    f"image: expected existing *.sqsh at {image!r}, got missing file"
                )
            self._image_path = image
            return image
        if not image.startswith("docker://"):
            raise SandboxRefusal(
                f"image: expected 'docker://<ref>' or existing *.sqsh path, got {image!r}"
            )
        digest = hashlib.sha256(image.encode("utf-8")).hexdigest()[:16]
        path = str(Path(self.spec.image_store) / f"{digest}.sqsh")
        self._image_path = path
        if Path(path).exists():
            return path
        Path(self.spec.image_store).mkdir(parents=True, exist_ok=True)
        partial = path + ".partial"
        if Path(partial).exists():
            Path(partial).unlink()
        result = self._runner.run(
            ["enroot", "import", "-o", partial, image],
            env={"ENROOT_DATA_PATH": self.spec.data_root},
        )
        if result.returncode != 0:
            if Path(partial).exists():
                Path(partial).unlink()
            raise SandboxError(
                f"enroot import failed for {image!r} (exit {result.returncode}): "
                f"{result.stderr[-_STDERR_TAIL:]}"
            )
        if not Path(partial).exists():
            raise SandboxError(
                f"enroot import of {image!r} exited 0 but wrote no image at {partial!r}; "
                f"refusing to treat a missing file as an imported image"
            )
        Path(partial).replace(path)
        return path

    def start(self) -> None:
        if self._started:
            raise SandboxRefusal(
                f"start: expected not-started sandbox, got already-started {self.spec.name!r}"
            )
        image_path = self.image_path()
        result = self._runner.run(
            ["enroot", "create", "-n", self.spec.name, image_path],
            env={"ENROOT_DATA_PATH": self.spec.data_root},
        )
        if result.returncode != 0:
            raise SandboxError(
                f"enroot create failed for {self.spec.name!r} (exit {result.returncode}): "
                f"{result.stderr[-_STDERR_TAIL:]}"
            )
        Path(self._scratch_tmp()).mkdir(parents=True, exist_ok=True)
        metadata = Path(self.rootfs()) / "etc" / "fs-sandbox.json"
        if metadata.is_file():
            try:
                raw = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SandboxError(f"invalid build metadata {metadata}: {exc}") from exc
            if not isinstance(raw, dict):
                raise SandboxError(f"invalid build metadata {metadata}: expected object")
            env = raw.get("env", {})
            workdir = raw.get("workdir")
            if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env.items()
            ):
                raise SandboxError(f"invalid build metadata {metadata}: bad 'env'")
            if workdir is not None and not isinstance(workdir, str):
                raise SandboxError(f"invalid build metadata {metadata}: bad 'workdir'")
            self._build_env = MappingProxyType(dict(env))
            self._build_workdir = workdir
        self._started = True

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        if not self._started:
            raise SandboxRefusal(
                f"exec: expected started sandbox, got not-started {self.spec.name!r}"
            )
        if not isinstance(command, str) or not command:
            raise SandboxRefusal(f"command: expected non-empty str, got {command!r}")
        if env is not None:
            for key, value in env.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise SandboxRefusal(f"env: expected Mapping[str, str], got {key!r}: {value!r}")
        merged: dict[str, str] = {}
        merged.update(self._build_env)
        merged.update(self.spec.env)
        if env:
            merged.update(env)
        workdir = cwd if cwd is not None else (self._build_workdir or self.spec.workdir)
        argv: list[str] = []
        if self.spec.network == "no-network":
            argv += ["unshare", "-rn"]
        if self.spec.cpu_set:
            argv += ["taskset", "-c", ",".join(str(cpu) for cpu in self.spec.cpu_set)]
        if self.spec.memory_mb:
            argv += ["prlimit", f"--as={self.spec.memory_mb * 1048576}"]
        argv += [
            "enroot",
            "start",
            "--rw",
            "--mount",
            f"{self._scratch_tmp()}:/tmp",
        ]
        for key, value in merged.items():
            argv += ["--env", f"{key}={value}"]
        argv += [
            self.spec.name,
            "sh",
            "-c",
            f"cd {shlex.quote(workdir)} && {command}",
        ]
        result = self._runner.run(
            argv,
            env={"ENROOT_DATA_PATH": self.spec.data_root},
            timeout_s=timeout_s,
        )
        return_code = 124 if result.timed_out else result.returncode
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=return_code,
            timed_out=result.timed_out,
        )

    def stop(self, *, delete: bool = True) -> None:
        if delete and self._started:
            result = self._runner.run(
                ["enroot", "remove", "-f", self.spec.name],
                env={"ENROOT_DATA_PATH": self.spec.data_root},
            )
            if result.returncode != 0:
                raise SandboxError(
                    f"enroot remove failed for {self.spec.name!r} "
                    f"(exit {result.returncode}): {result.stderr[-_STDERR_TAIL:]}"
                )
            scratch = str(Path(self.spec.scratch_root) / self.spec.name)
            if Path(scratch).is_dir():
                shutil.rmtree(scratch, ignore_errors=True)
        self._started = False

    # -- paths -------------------------------------------------------------

    def rootfs(self) -> Path:
        return Path(self.spec.data_root) / self.spec.name

    def host_path(self, container_path: str) -> Path:
        if not isinstance(container_path, str) or not container_path:
            raise SandboxRefusal(
                f"container_path: expected non-empty absolute path, got {container_path!r}"
            )
        if not container_path.startswith("/"):
            raise SandboxRefusal(f"container_path: expected absolute path, got {container_path!r}")
        if container_path == "/tmp" or container_path.startswith("/tmp/"):
            root = Path(self._scratch_tmp())
            rel = container_path[len("/tmp") :].lstrip("/")
        else:
            root = self.rootfs()
            rel = container_path.lstrip("/")
        resolved_root = root.resolve()
        resolved = (resolved_root / rel).resolve() if rel else resolved_root
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise SandboxRefusal(
                f"container_path: expected path inside {resolved_root}, got "
                f"{container_path!r} resolving to {resolved}"
            )
        return resolved

    def _scratch_tmp(self) -> str:
        return str(Path(self.spec.scratch_root) / self.spec.name / "tmp")

    # -- transfer ----------------------------------------------------------

    def upload_file(self, source: str | Path, target: str) -> None:
        src = Path(source)
        if not src.is_file():
            raise SandboxRefusal(f"source: expected existing file, got {src}")
        dst = self.host_path(target)
        if dst.is_dir() and not dst.is_symlink():
            dst = self.host_path(target.rstrip("/") + "/" + src.name)
        self._write_file(src, dst)

    def upload_dir(self, source_dir: str | Path, target_dir: str) -> None:
        src = Path(source_dir)
        if not src.is_dir():
            raise SandboxRefusal(f"source_dir: expected existing directory, got {src}")
        base = target_dir.rstrip("/") or "/"
        self.host_path(base).mkdir(parents=True, exist_ok=True)
        for entry in sorted(src.rglob("*")):
            if entry.is_symlink():
                continue  # never carry host symlinks into a sandbox
            # Every destination is re-resolved through host_path: a directory the agent
            # replaced with a symlink inside the target can never redirect this write.
            out = self.host_path(base.rstrip("/") + "/" + entry.relative_to(src).as_posix())
            if entry.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif entry.is_file():
                self._write_file(entry, out)

    def download_file(self, source: str, target: str | Path) -> None:
        src = self.host_path(source)
        if not src.is_file():
            raise SandboxRefusal(f"source: expected existing file in sandbox, got {source!r}")
        dst = Path(target)
        if dst.is_dir():
            dst = dst / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def download_dir(self, source_dir: str, target_dir: str | Path) -> None:
        src = self.host_path(source_dir)
        if not src.is_dir():
            raise SandboxRefusal(
                f"source_dir: expected existing directory in sandbox, got {source_dir!r}"
            )
        dst = Path(target_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for entry in sorted(src.rglob("*")):
            # Symlinks are skipped, never followed: an agent could point one at a host
            # file (a key, a secrets file) and have the harness copy it into artifacts.
            if entry.is_symlink():
                continue
            resolved = entry.resolve()
            if resolved != src and src not in resolved.parents:
                continue
            out = dst / entry.relative_to(src)
            if entry.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif entry.is_file():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(entry, out)

    @staticmethod
    def _write_file(src: Path, dst: Path) -> None:
        """Write ``src`` to ``dst`` (contained), replacing -- never following -- a symlink."""
        if dst.is_symlink():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        shutil.copymode(src, dst)
