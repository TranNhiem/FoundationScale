from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import sys
import tempfile
import time

from foundationscale.agentic_rl.envs.base import (
    EnvCapabilities,
    EnvInfraError,
    Environment,
    EnvRefusal,
    EnvSpec,
    ExecResult,
    check_isolation,
)

__all__ = ("LocalEnvironment", "LocalSubprocessBackend")


def _described(value: object) -> str:
    """Short type-and-value rendering used in refusal messages."""
    return f"{type(value).__name__} {value!r}"


def _sanitise_prefix(instance_id: str) -> str:
    """Safe ``tempfile.mkdtemp`` prefix derived from ``instance_id``.

    Keeps only ``[A-Za-z0-9_.-]`` and replaces every other character with ``_``;
    a model-supplied-looking ``instance_id`` must never be interpolated raw into a
    path, so this guarantees the prefix is a single safe path component. Returns the
    fixed fallback ``"fsar-"`` when sanitising leaves nothing usable.
    """
    cleaned = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in "_.-") else "_" for ch in instance_id
    )
    if not cleaned or set(cleaned) == {"_"}:
        return "fsar-"
    return cleaned


class LocalEnvironment:
    """Bare-subprocess environment: commands run directly on this host via ``/bin/sh -c``.

    Gives NO process isolation -- the model's shell commands execute with the calling
    process's privileges and full host access. ``start()`` runs exactly once and creates
    the working directory; ``close()`` is idempotent and safe at any time.

    Every command's environment is ``spec.env`` (the caller's explicit allow-list)
    PLUS exactly two variables this backend adds itself: ``HOME`` and ``PWD``, both
    set to the instance's own working directory -- so a model-chosen command has a
    sane home directory and starts already in the workdir. These two are the ONLY
    additions; nothing else, and nothing from this process's own ``os.environ``, ever
    reaches the spawned command (see ``envs.base.EnvSpec``'s docstring for the
    allow-list contract this keeps truthful).
    """

    def __init__(self, spec: EnvSpec, *, instance_id: str) -> None:
        """Store configuration only; the filesystem is touched by ``start()``.

        Raises ``EnvRefusal`` on non-POSIX hosts because this backend uses
        ``os.killpg``/process groups, which are POSIX only.
        """
        if os.name != "posix":
            raise EnvRefusal(
                f"LocalEnvironment.__init__: host 'os.name' is {_described(os.name)}: this "
                f"backend uses os.killpg/process groups and is POSIX only -- running it on a "
                f"non-POSIX host cannot kill a timed-out command's whole process group, so "
                f"orphaned children would outlive the episode"
            )
        self._spec = spec
        self._instance_id = instance_id
        self._workdir: str | None = None
        self._closed = False
        self._deadline: float | None = None

    @property
    def episode_timeout_s(self) -> float:
        """The episode's wall-clock budget in seconds (read-only)."""
        return self._spec.episode_timeout_s

    @property
    def workdir(self) -> str:
        """The environment's working directory.

        Raises ``EnvRefusal`` before ``start()`` has run, since no workdir exists yet.
        """
        if self._workdir is None:
            raise EnvRefusal(
                "LocalEnvironment.workdir: property 'workdir' is None: start() has not run, "
                "so no working directory exists -- reading it before start() would hand back "
                "a path that was never created"
            )
        return self._workdir

    async def start(self) -> None:
        """Create the working directory and run ``spec.setup_commands`` in order.

        Runs exactly once; a second call raises ``EnvRefusal``. On a failing setup command
        the half-created temp dir is removed best-effort before raising ``EnvInfraError``
        (callers SHOULD still call ``close()`` defensively -- it tolerates a missing dir).
        """
        if self._workdir is not None or self._deadline is not None:
            raise EnvRefusal(
                "LocalEnvironment.start: method 'start' was called on an already-started "
                "environment: start() runs exactly once -- a second call would create a second "
                "working directory and silently orphan the first"
            )
        prefix = _sanitise_prefix(self._instance_id)
        workdir = tempfile.mkdtemp(prefix=prefix)
        self._workdir = workdir
        self._deadline = time.monotonic() + self._spec.episode_timeout_s
        for command in self._spec.setup_commands:
            result = await self._run(command, timeout_s=self._spec.exec_timeout_s)
            if result.timed_out or result.exit_code != 0:
                try:
                    shutil.rmtree(workdir, ignore_errors=False)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
                self._workdir = None
                self._deadline = None
                detail = "timed out" if result.timed_out else f"exit code {result.exit_code}"
                raise EnvInfraError(
                    "setup_failed",
                    f"LocalEnvironment.start: setup command {_described(command)} failed with "
                    f"{detail}: setup_commands must all succeed before the episode starts -- a "
                    f"failing setup leaves the environment in an unknown state",
                )

    async def execute(self, command: str, *, timeout_s: float | None = None) -> ExecResult:
        """Run ``command`` via ``/bin/sh -c`` in the working directory.

        Raises ``EnvRefusal`` if ``start()`` was never called or ``close()`` was already
        called. A timeout yields ``exit_code=None, timed_out=True`` rather than an error.
        """
        if self._workdir is None:
            raise EnvRefusal(
                f"LocalEnvironment.execute: method 'execute' was called before 'start' "
                f"(command {_described(command)}): execute() requires a started environment -- "
                f"there is no working directory to run in"
            )
        if self._closed:
            raise EnvRefusal(
                f"LocalEnvironment.execute: method 'execute' was called after 'close' "
                f"(command {_described(command)}): a closed environment has had its working "
                f"directory removed, so its commands would run somewhere unintended"
            )
        assert self._deadline is not None
        remaining = max(0.0, self._deadline - time.monotonic())
        requested = self._spec.exec_timeout_s if timeout_s is None else timeout_s
        effective = min(requested, remaining)
        return await self._run(command, timeout_s=effective)

    async def close(self) -> None:
        """Remove the working directory; idempotent, and safe before ``start()``.

        Tolerates the directory already being gone (``FileNotFoundError``) but lets any
        other ``OSError`` from ``shutil.rmtree`` propagate, so unrelated failures surface.
        """
        if self._closed:
            return
        self._closed = True
        if self._workdir is not None:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self._workdir, ignore_errors=False)

    async def _run(self, command: str, *, timeout_s: float) -> ExecResult:
        """Shared spawn/read/kill-group path for both setup commands and ``execute()``.

        Setup callers raise on failure themselves; this returns the honest ``ExecResult``.
        """
        assert self._workdir is not None
        started = time.monotonic()
        # The only two variables this backend adds beyond spec.env's declared
        # allow-list -- see the class docstring. Passing env= to
        # create_subprocess_exec REPLACES the child's whole environment, so nothing
        # from this process's own os.environ leaks in alongside them.
        env = dict(self._spec.env) | {"HOME": self._workdir, "PWD": self._workdir}
        try:
            process = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-c",
                command,
                cwd=self._workdir,
                env=env,
                start_new_session=True,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise EnvInfraError(
                "start_failed",
                f"LocalEnvironment._run: spawning command {_described(command)} raised "
                f"{type(exc).__name__} {exc}: the subprocess could not be created at all -- "
                f"this is a host/infra failure, not a command failure",
            ) from exc

        async def _read(stream: asyncio.StreamReader | None) -> bytes:
            if stream is None:
                return b""
            return await stream.read()

        stdout_task = asyncio.ensure_future(_read(process.stdout))
        stderr_task = asyncio.ensure_future(_read(process.stderr))

        async def _collect() -> tuple[bytes, bytes]:
            out_b, err_b = await asyncio.gather(stdout_task, stderr_task)
            await process.wait()
            return out_b, err_b

        cap = self._spec.max_output_bytes
        try:
            out_bytes, err_bytes = await asyncio.wait_for(_collect(), timeout=timeout_s)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            # Partial output after a hard kill is best-effort: if a reader task raced
            # the kill and cannot be cleanly recovered, "" is the accepted answer for
            # that stream here. Whatever WAS recovered is still subject to the same
            # head/tail cap as the non-timeout path below, with the real dropped-byte
            # count reported -- a timeout must not silently claim 0 bytes were cut.
            out_bytes = _finished_result(stdout_task)
            err_bytes = _finished_result(stderr_task)
            kept_out, dropped_out = _cap_bytes(out_bytes, cap)
            kept_err, dropped_err = _cap_bytes(err_bytes, cap)
            return ExecResult(
                stdout=kept_out.decode("utf-8", errors="replace"),
                stderr=kept_err.decode("utf-8", errors="replace"),
                exit_code=None,
                timed_out=True,
                truncated_bytes=dropped_out + dropped_err,
                seconds=time.monotonic() - started,
            )

        kept_out, dropped_out = _cap_bytes(out_bytes, cap)
        kept_err, dropped_err = _cap_bytes(err_bytes, cap)
        return ExecResult(
            stdout=kept_out.decode("utf-8", errors="replace"),
            stderr=kept_err.decode("utf-8", errors="replace"),
            exit_code=process.returncode,
            timed_out=False,
            truncated_bytes=dropped_out + dropped_err,
            seconds=time.monotonic() - started,
        )


def _finished_result(task: asyncio.Future[bytes]) -> bytes:
    """The task's result if it finished cleanly, else ``b""``.

    Used after a hard process-group kill: a reader task that raced the kill may
    still be pending or cancelled, and recovering its partial bytes is not worth
    the complexity -- ``b""`` is the accepted, documented answer in that case.
    """
    if task.done() and not task.cancelled():
        return task.result()
    return b""


def _cap_bytes(raw: bytes, cap: int) -> tuple[bytes, int]:
    """Keep the first and last ``cap // 2`` bytes of ``raw`` when it exceeds ``cap``.

    Returns ``(kept_bytes, dropped_count)`` where ``dropped_count`` is exactly
    ``len(raw) - len(kept_bytes)`` -- the bytes actually discarded.
    """
    n = len(raw)
    if n <= cap:
        return raw, 0
    half = cap // 2
    kept = raw[:half] + raw[n - half :] if half > 0 else b""
    return kept, n - len(kept)


class LocalSubprocessBackend:
    """Backend producing ``LocalEnvironment`` instances with NO isolation.

    ``isolation="none"`` because this backend gives no sandboxing whatsoever -- it is a
    bare subprocess on the calling host. ``network=True`` because nothing here blocks
    outbound network access (a known, accepted gap for this backend, not a bug).
    ``max_concurrency=None`` means no backend-imposed limit (bounded only by the host).
    """

    name: str = "local"

    def capabilities(self) -> EnvCapabilities:
        """Report host-level capabilities: no isolation, unrestricted network, no
        concurrency cap.
        """
        return EnvCapabilities(
            isolation="none",
            network=True,
            max_concurrency=None,
            platform=sys.platform,
        )

    def create(self, spec: EnvSpec, *, instance_id: str) -> Environment:
        """Construct (but do not start) a ``LocalEnvironment``.

        Raises ``EnvRefusal`` via ``check_isolation`` when ``spec.allow_unsandboxed`` is
        not True -- this is where the unsandboxed-use refusal fires for this backend.
        """
        check_isolation(spec, self.capabilities())
        return LocalEnvironment(spec, instance_id=instance_id)
