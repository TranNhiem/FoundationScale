"""Per-episode Enroot containers for software-engineering tasks (the ``enroot_task`` backend).

Every episode gets its own ``EnrootSandbox`` built from the task's metadata (image, workdir and
test command), so one task's repo, dependencies and daemons can never leak into another's. The
task's hidden tests are moved out of the container by a ``HiddenVault`` while the agent works;
``verify()`` moves them back, runs the test command and reports per-test outcomes in
``swe_status``/``swe_tests`` for ``rewards/swe.py`` to score.

The backend is NOT registered here -- registration happens where backends are assembled.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from foundationscale.agentic_rl.envs.base import (
    EnvCapabilities,
    EnvInfraError,
    Environment,
    EnvRefusal,
    EnvSpec,
    ExecResult,
)
from foundationscale.agentic_rl.integrity import IntegrityRefusal
from foundationscale.agentic_rl.sandbox.broker import BrokerRunner
from foundationscale.agentic_rl.sandbox.enroot import EnrootSandbox, SandboxError, SandboxSpec
from foundationscale.agentic_rl.sandbox.hidden import HiddenSpec, HiddenVault, SandboxLike

if TYPE_CHECKING:
    from foundationscale.agentic_rl.harness.base import EpisodeTask

__all__ = (
    "EnrootTaskBackend",
    "SandboxLikeFull",
    "SandboxRoots",
    "TaskSandboxConfig",
    "TaskSandboxEnvironment",
    "VaultLike",
    "parse_pytest_report",
)


def _described(value: object) -> str:
    """Short type-and-value rendering used in refusal messages."""
    return f"{type(value).__name__} {value!r}"


class SandboxExecResult(Protocol):
    """What ``EnrootSandbox.exec`` hands back: decoded output, status and a timeout flag."""

    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...
    @property
    def return_code(self) -> int: ...
    @property
    def timed_out(self) -> bool: ...


class SandboxLikeFull(Protocol):
    """The ``EnrootSandbox`` surface this module needs: lifecycle, exec and host mapping."""

    def start(self) -> None:
        """Build and start the container named by the sandbox spec."""

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> SandboxExecResult:
        """Run ``command`` in the container and return its output and status."""

    def stop(self, *, delete: bool = True) -> None:
        """Stop the container, deleting its data when ``delete`` is true."""

    def host_path(self, container_path: str) -> Path:
        """Map an absolute container path to the path it occupies on the host."""


class VaultLike(Protocol):
    """The ``HiddenVault`` surface this module needs: hide hidden paths, then restore them."""

    def seal(self) -> tuple[str, ...]:
        """Move each existing hidden path into the vault; return moved container paths."""

    def unseal(self) -> tuple[str, ...]:
        """Move everything back for the verifier phase."""


def _required_root(environ: Mapping[str, str], name: str) -> str:
    """Read one required absolute host directory from ``environ``, refusing anything else."""
    value = environ.get(name)
    if value is None:
        raise EnvRefusal(f"{name}: expected an absolute path, got a missing variable")
    if not Path(value).is_absolute():
        raise EnvRefusal(f"{name}: expected an absolute path, got {_described(value)}")
    return value


@dataclass(frozen=True)
class SandboxRoots:
    """Host directories one ``EnrootSandbox`` needs, normally read from the environment.

    ``data_root`` holds per-sandbox data, ``image_store`` caches imported images and
    ``scratch_root`` is the per-sandbox scratch area (each episode's hidden-test vault lives
    beside its sandbox there). ``enroot_sysconf`` is the optional enroot config dir
    (``ENROOT_SYSCONF_PATH``).
    """

    data_root: str
    image_store: str
    scratch_root: str
    enroot_sysconf: str | None = None

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> SandboxRoots:
        """Build the roots from the ``FS_SANDBOX_*`` variables of ``environ``.

        ``FS_SANDBOX_DATA_ROOT``, ``FS_SANDBOX_IMAGE_STORE`` and ``FS_SANDBOX_SCRATCH_ROOT`` must
        each be present and an absolute path -- a refusal names the offending variable, since a
        missing or relative root would scatter container data over the host. The optional
        ``FS_SANDBOX_ENROOT_SYSCONF`` is used as-is, with an empty or absent value becoming
        ``None`` (the enroot defaults).
        """
        return cls(
            data_root=_required_root(environ, "FS_SANDBOX_DATA_ROOT"),
            image_store=_required_root(environ, "FS_SANDBOX_IMAGE_STORE"),
            scratch_root=_required_root(environ, "FS_SANDBOX_SCRATCH_ROOT"),
            enroot_sysconf=environ.get("FS_SANDBOX_ENROOT_SYSCONF") or None,
        )


def _required_metadata_str(metadata: Mapping[str, str], key: str) -> str:
    """Read one required non-empty ``str`` from task metadata, refusing a missing/empty value."""
    if key not in metadata:
        raise EnvRefusal(
            f"{key}: expected a non-empty str in EpisodeTask.metadata, got a missing key"
        )
    value = metadata[key]
    if not value:
        raise EnvRefusal(
            f"{key}: expected a non-empty str in EpisodeTask.metadata, got {_described(value)}"
        )
    return value


def _workdir(raw: str) -> str:
    """Validate the absolute in-container working directory the agent's commands run in."""
    if not Path(raw).is_absolute():
        raise EnvRefusal(f"workdir: expected an absolute path, got {_described(raw)}")
    return raw


def _hidden_paths(raw: str) -> tuple[str, ...]:
    """Parse the JSON list of absolute container paths hidden during the agent phase."""
    try:
        loaded = json.loads(raw)
    except ValueError as exc:
        raise EnvRefusal(
            f"hidden_paths: expected a JSON list of absolute paths, got {_described(raw)}"
        ) from exc
    if not isinstance(loaded, list) or not loaded:
        raise EnvRefusal(
            f"hidden_paths: expected a non-empty JSON list of absolute paths, got {_described(raw)}"
        )
    paths: list[str] = []
    for entry in loaded:
        if not isinstance(entry, str) or not Path(entry).is_absolute():
            raise EnvRefusal(
                f"hidden_paths: expected a JSON list of absolute paths, "
                f"got entry {_described(entry)}"
            )
        paths.append(entry)
    return tuple(paths)


def _test_timeout_s(raw: str) -> float:
    """Parse the positive, finite timeout the verifier's test command gets."""
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise EnvRefusal(f"test_timeout_s: expected a float > 0, got {_described(raw)}") from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise EnvRefusal(f"test_timeout_s: expected a float > 0, got {_described(raw)}")
    return timeout_s


@dataclass(frozen=True)
class TaskSandboxConfig:
    """The sandbox's per-task settings, parsed from ``EpisodeTask.metadata``."""

    image: str
    workdir: str
    test_cmd: str
    hidden_paths: tuple[str, ...]
    test_timeout_s: float

    @classmethod
    def from_task(cls, task: EpisodeTask) -> TaskSandboxConfig:
        """Parse (and validate) one task's metadata into a sandbox configuration.

        ``image`` and ``test_cmd`` are required non-empty strings; ``workdir`` defaults to
        ``"/repo"`` and must be absolute; ``hidden_paths`` defaults to ``'["/tests"]'`` and must
        be a non-empty JSON list of absolute paths; ``test_timeout_s`` defaults to ``"600"`` and
        must parse as a finite float greater than zero. Any violation raises ``EnvRefusal``.
        """
        metadata = task.metadata
        return cls(
            image=_required_metadata_str(metadata, "image"),
            workdir=_workdir(metadata.get("workdir", "/repo")),
            test_cmd=_required_metadata_str(metadata, "test_cmd"),
            hidden_paths=_hidden_paths(metadata.get("hidden_paths", '["/tests"]')),
            test_timeout_s=_test_timeout_s(metadata.get("test_timeout_s", "600")),
        )


_SUMMARY_HEADER = "short test summary info"
_OUTCOME_BY_STATUS: tuple[tuple[str, str], ...] = (
    ("PASSED", "passed"),
    ("FAILED", "failed"),
    ("ERROR", "error"),
    ("SKIPPED", "skipped"),
    ("XFAIL", "skipped"),
    ("XPASS", "passed"),
)
_SEVERITY: Mapping[str, int] = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}


def _node_id(rest: str) -> str | None:
    """The node id at the head of ``rest``: the token up to the first ``" - "`` or whitespace."""
    cut = len(rest)
    separator = rest.find(" - ")
    if separator != -1:
        cut = separator
    for index, char in enumerate(rest):
        if char.isspace():
            cut = min(cut, index)
            break
    node_id = rest[:cut].strip()
    return node_id or None


def _summary_line(line: str) -> tuple[str, str] | None:
    """One ``(node_id, outcome)`` pair for a short-summary line, or ``None`` if it is not one."""
    for status, outcome in _OUTCOME_BY_STATUS:
        marker = status + " "
        if not line.startswith(marker):
            continue
        node_id = _node_id(line[len(marker) :])
        if node_id is None:
            return None
        if status == "SKIPPED" and "::" not in node_id:
            return None
        return node_id, outcome
    return None


def parse_pytest_report(stdout: str) -> dict[str, str]:
    """Parse pytest ``-rA`` short-summary lines into ``node id -> outcome``.

    Only lines starting with ``PASSED``/``FAILED``/``ERROR``/``SKIPPED``/``XFAIL``/``XPASS``
    followed by a space count; the node id is the token after that status, cut at the first
    ``" - "`` or whitespace. ``PASSED`` and ``XPASS`` map to ``"passed"``, ``FAILED`` to
    ``"failed"``, ``ERROR`` to ``"error"``, ``SKIPPED`` and ``XFAIL`` to ``"skipped"``.
    ``SKIPPED`` lines name a file:line rather than a node id
    (``"SKIPPED [1] tests/a.py:3: reason"``), so those whose node id cannot be determined --
    the token has no ``"::"`` -- are dropped. When a node id appears twice the worse outcome
    wins (error > failed > skipped > passed). Lines before the ``"short test summary info"``
    header are ignored when that header is present; without it every line is scanned.
    """
    lines = stdout.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if _SUMMARY_HEADER in line:
            start = index + 1
            break
    results: dict[str, str] = {}
    for line in lines[start:]:
        parsed = _summary_line(line)
        if parsed is None:
            continue
        node_id, outcome = parsed
        previous = results.get(node_id)
        if previous is None or _SEVERITY[outcome] > _SEVERITY[previous]:
            results[node_id] = outcome
    return results


def _cap_output(stdout: str, stderr: str, cap: int) -> tuple[str, str, int]:
    """Cap ``stdout``+``stderr`` at ``cap`` UTF-8 bytes total, giving ``stdout`` priority.

    Returns the two kept texts and the exact number of bytes dropped. Output is kept from the
    front of each stream: ``stdout`` is served first and ``stderr`` only gets what is left of the
    budget. Decoding uses ``errors="replace"`` so a cut in the middle of a character cannot
    raise -- the replacement char is the honest sign that output was cut there.
    """
    out_raw = stdout.encode("utf-8")
    err_raw = stderr.encode("utf-8")
    kept_out = out_raw[:cap]
    kept_err = err_raw[: max(0, cap - len(kept_out))]
    dropped = len(out_raw) - len(kept_out) + len(err_raw) - len(kept_err)
    return (
        kept_out.decode("utf-8", errors="replace"),
        kept_err.decode("utf-8", errors="replace"),
        dropped,
    )


class TaskSandboxEnvironment:
    """One episode's Enroot container: agent commands now, hidden tests at ``verify()``.

    ``start()`` starts the container, seals the hidden paths away and runs
    ``spec.setup_commands``; ``execute()`` runs agent commands while those paths are hidden;
    ``verify()`` -- once, after the agent phase -- restores them, runs ``config.test_cmd`` and
    returns the per-test mapping ``rewards/swe.py`` scores; ``close()`` is idempotent and
    deletes the container.

    Every command's environment is exactly ``spec.env`` (the caller's explicit allow-list), or
    the sandbox's own default when that allow-list is empty -- nothing from this process's
    ``os.environ`` is ever inherited (see ``envs.base.EnvSpec``'s allow-list contract).
    """

    def __init__(
        self,
        *,
        spec: EnvSpec,
        config: TaskSandboxConfig,
        sandbox: SandboxLikeFull,
        vault: VaultLike,
    ) -> None:
        """Store the spec, the parsed task config and the sandbox/vault pair; nothing starts yet."""
        self._spec = spec
        self._config = config
        self._sandbox = sandbox
        self._vault = vault
        self._started = False
        self._verified = False
        self._closed = False

    @property
    def workdir(self) -> str:
        """The task's working directory inside the container (``config.workdir``)."""
        return self._config.workdir

    @property
    def episode_timeout_s(self) -> float:
        """The episode's wall-clock budget in seconds (read-only)."""
        return self._spec.episode_timeout_s

    async def start(self) -> None:
        """Start the sandbox, hide the hidden paths and run ``spec.setup_commands`` in order.

        Runs exactly once: a second call (or a call after ``close()``) raises ``EnvRefusal``.
        A sandbox/vault/setup fault is an ``EnvInfraError`` chained from the original exception,
        never a scored failure -- callers SHOULD still call ``close()`` defensively.
        """
        if self._started:
            raise EnvRefusal(
                "start: expected an unstarted environment, got an already-started one: start() "
                "runs exactly once -- a second call would build a second lifecycle over the "
                "same sandbox"
            )
        if self._closed:
            raise EnvRefusal(
                "start: expected an open environment, got a closed one: a closed environment's "
                "container has already been deleted, so there is nothing left to start"
            )
        self._started = True
        await asyncio.to_thread(self._start_sync)

    async def execute(self, command: str, *, timeout_s: float | None = None) -> ExecResult:
        """Run one agent command in the container and return its capped output.

        Raises ``EnvRefusal`` before ``start()``, after ``verify()`` (the agent phase is over and
        the hidden tests are back) or after ``close()``. A timeout yields
        ``exit_code=None, timed_out=True`` rather than an error. ``stdout``+``stderr`` together
        are capped at ``spec.max_output_bytes`` UTF-8 bytes with ``stdout`` taking priority, and
        ``truncated_bytes`` reports exactly how many bytes were dropped.
        """
        if not self._started:
            raise EnvRefusal(
                f"execute: expected a started environment, got an unstarted one "
                f"(command {_described(command)}): execute() requires start() to have run -- "
                f"there is no started sandbox to run the command in"
            )
        if self._verified:
            raise EnvRefusal(
                f"execute: expected an environment still in its agent phase, got a verified one "
                f"(command {_described(command)}): verify() restores the hidden tests and ends "
                f"the agent phase, so no further agent command may run"
            )
        if self._closed:
            raise EnvRefusal(
                f"execute: expected an open environment, got a closed one "
                f"(command {_described(command)}): a closed environment's container has been "
                f"deleted, so its commands would run nowhere"
            )
        started = time.monotonic()
        result = await asyncio.to_thread(
            self._exec_sync, command, timeout_s=timeout_s or self._spec.exec_timeout_s
        )
        stdout, stderr, dropped = _cap_output(
            result.stdout, result.stderr, self._spec.max_output_bytes
        )
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=None if result.timed_out else result.return_code,
            timed_out=result.timed_out,
            truncated_bytes=dropped,
            seconds=time.monotonic() - started,
        )

    async def verify(self) -> dict[str, str]:
        """Restore the hidden tests, run ``config.test_cmd`` and report per-test outcomes.

        Allowed exactly once, after ``start()``; afterwards ``execute()`` refuses, because the
        agent phase is over. The mapping is what ``rewards/swe.py`` reads: ``swe_status`` is
        ``"ok"`` when pytest reported at least one test -- with ``swe_tests`` a JSON object of
        ``test_id -> "passed"|"failed"|"error"|"skipped"`` and ``swe_exit_code`` -- else
        ``"error"`` with ``swe_error``. A timeout or a report with no tests is an ERROR outcome,
        never a scored zero; only an infra fault raises ``EnvInfraError``.
        """
        if not self._started:
            raise EnvRefusal(
                "verify: expected a started environment, got an unstarted one: verify() restores "
                "hidden tests into a container that start() never built"
            )
        if self._verified:
            raise EnvRefusal(
                "verify: expected an unverified environment, got an already-verified one: "
                "verify() runs exactly once -- a second run would re-report the first result"
            )
        if self._closed:
            raise EnvRefusal(
                "verify: expected an open environment, got a closed one: a closed environment's "
                "container has been deleted, so its hidden tests cannot be restored"
            )
        self._verified = True
        return await asyncio.to_thread(self._verify_sync)

    async def close(self) -> None:
        """Delete the container; idempotent, and safe before ``start()``.

        Marks the environment closed BEFORE stopping, so a failing ``stop()`` can never leave a
        half-open environment that looks reusable. A ``stop()`` failure is never swallowed: it
        raises ``EnvInfraError`` chained from the original fault.
        """
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.to_thread(self._sandbox.stop, delete=True)
        except (SandboxError, IntegrityRefusal, OSError) as exc:
            raise EnvInfraError(
                "stop_failed",
                f"TaskSandboxEnvironment.close: deleting the episode sandbox raised "
                f"{type(exc).__name__} {exc}: the container may still exist on the host -- this "
                f"is an infra failure, never a silent success",
            ) from exc

    def _start_sync(self) -> None:
        """Blocking half of ``start()``: start the sandbox, seal the vault, run setup commands."""
        try:
            self._sandbox.start()
            self._vault.seal()
        except (SandboxError, IntegrityRefusal, OSError) as exc:
            raise EnvInfraError(
                "start_failed",
                f"TaskSandboxEnvironment.start: starting the episode sandbox and sealing its "
                f"hidden paths raised {type(exc).__name__} {exc}: the container could not be "
                f"prepared at all -- this is an infra failure, not a command failure",
            ) from exc
        for command in self._spec.setup_commands:
            try:
                result = self._sandbox.exec(
                    command,
                    cwd=self._config.workdir,
                    env=dict(self._spec.env) or None,
                    timeout_s=self._spec.exec_timeout_s,
                )
            except (SandboxError, IntegrityRefusal, OSError) as exc:
                raise EnvInfraError(
                    "setup_failed",
                    f"TaskSandboxEnvironment.start: setup command {_described(command)} raised "
                    f"{type(exc).__name__} {exc}: setup_commands must all run before the episode "
                    f"starts -- a failing setup leaves the environment in an unknown state",
                ) from exc
            if result.timed_out or result.return_code != 0:
                detail = "timed out" if result.timed_out else f"exit code {result.return_code}"
                raise EnvInfraError(
                    "setup_failed",
                    f"TaskSandboxEnvironment.start: setup command {_described(command)} failed "
                    f"with {detail}: setup_commands must all succeed before the episode starts "
                    f"-- a failing setup leaves the environment in an unknown state",
                )

    def _exec_sync(self, command: str, *, timeout_s: float) -> SandboxExecResult:
        """Blocking half of ``execute()``: run ``command`` in the container.

        ``SandboxError`` -- the command could not be run at all -- becomes ``EnvInfraError``; a
        command that merely fails or times out is an ordinary result.
        """
        try:
            return self._sandbox.exec(
                command,
                cwd=self._config.workdir,
                env=dict(self._spec.env) or None,
                timeout_s=timeout_s,
            )
        except SandboxError as exc:
            raise EnvInfraError(
                "transport",
                f"TaskSandboxEnvironment.execute: sandbox exec of {_described(command)} raised "
                f"{type(exc).__name__} {exc}: the command could not be run at all -- this is an "
                f"infra failure, not a command failure",
            ) from exc

    def _verify_sync(self) -> dict[str, str]:
        """Blocking half of ``verify()``: unseal the hidden tests and run ``config.test_cmd``."""
        try:
            self._vault.unseal()
            result = self._sandbox.exec(
                self._config.test_cmd,
                cwd=self._config.workdir,
                env=dict(self._spec.env) or None,
                timeout_s=self._config.test_timeout_s,
            )
        except (SandboxError, IntegrityRefusal, OSError) as exc:
            raise EnvInfraError(
                "verify_failed",
                f"TaskSandboxEnvironment.verify: restoring the hidden tests and running "
                f"{_described(self._config.test_cmd)} raised {type(exc).__name__} {exc}: the "
                f"verifier phase could not run at all -- this is an infra failure, never a "
                f"scored zero",
            ) from exc
        if result.timed_out:
            return {"swe_status": "error", "swe_error": "test_cmd timed out"}
        tests = parse_pytest_report(result.stdout)
        if not tests:
            return {
                "swe_status": "error",
                "swe_error": f"no test results (exit {result.return_code})",
            }
        return {
            "swe_status": "ok",
            "swe_tests": json.dumps(tests, sort_keys=True),
            "swe_exit_code": str(result.return_code),
        }


_INSTANCE_ID = re.compile(r"[A-Za-z0-9._#-]+")


def _sandbox_name(instance_id: str) -> str:
    """Safe sandbox name component derived from ``instance_id`` (``#`` becomes ``-``).

    ``instance_id`` is caller-chosen and is interpolated into container and vault paths, so it
    must already be a single safe component matching ``^[A-Za-z0-9._#-]+$``; ``#`` is the one
    character that is legal there but not in an enroot name, so it is sanitised away.
    """
    if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
        raise EnvRefusal(
            f"instance_id: expected a name matching ^[A-Za-z0-9._#-]+$, "
            f"got {_described(instance_id)}"
        )
    return instance_id.replace("#", "-")


def _hidden_vault(sandbox: SandboxLikeFull, spec: HiddenSpec, vault_dir: str) -> VaultLike:
    """The default vault factory: a real ``HiddenVault`` over the episode's sandbox."""
    # hidden.SandboxLike types exec's result with plain attributes; EnrootSandbox satisfies both.
    return HiddenVault(cast("SandboxLike", sandbox), spec, vault_dir)


class EnrootTaskBackend:
    """Backend building one Enroot container per episode task (``name`` is ``"enroot_task"``).

    ``create()`` always refuses: the image lives in the task's metadata, so an environment can
    only be built through ``create_for_task``. Host directories come from ``SandboxRoots``
    (resolved lazily from ``FS_SANDBOX_*`` on first use), and both the sandbox and the
    hidden-test vault are built through injectable factories so tests can substitute fakes.
    """

    name: str = "enroot_task"

    def __init__(
        self,
        *,
        roots: SandboxRoots | None = None,
        environ: Mapping[str, str] | None = None,
        sandbox_factory: Callable[[SandboxSpec], SandboxLikeFull] | None = None,
        vault_factory: Callable[[SandboxLikeFull, HiddenSpec, str], VaultLike] | None = None,
    ) -> None:
        """Store configuration only; ``FS_SANDBOX_*`` is read lazily on the first episode.

        ``roots`` (when given) is used as-is; otherwise ``SandboxRoots.from_environ`` reads
        ``environ`` -- defaulting to this process's ``os.environ`` -- on the first
        ``create_for_task``. ``sandbox_factory`` defaults to ``EnrootSandbox`` and
        ``vault_factory`` to ``HiddenVault``.
        """
        self._roots = roots
        self._environ: Mapping[str, str] = os.environ if environ is None else environ
        self._sandbox_factory: Callable[[SandboxSpec], SandboxLikeFull] = (
            sandbox_factory if sandbox_factory is not None else self._enroot_sandbox
        )
        self._vault_factory: Callable[[SandboxLikeFull, HiddenSpec, str], VaultLike] = (
            vault_factory if vault_factory is not None else _hidden_vault
        )

    def capabilities(self) -> EnvCapabilities:
        """Report a network-less Linux container with no backend-imposed concurrency cap."""
        return EnvCapabilities(
            isolation="container",
            network=False,
            max_concurrency=None,
            platform="linux",
        )

    def create(self, spec: EnvSpec, *, instance_id: str) -> Environment:
        """Always refuse: this backend has no task-less environment to build.

        The container image is per-task metadata, so ``create()`` -- which receives ``spec`` and
        ``instance_id`` but no task -- has nothing to start; callers must use
        ``create_for_task``.
        """
        del spec, instance_id
        raise EnvRefusal(
            "enroot_task builds its sandbox from the episode task; it is created through "
            "create_for_task, and a task-less create() has no image to start"
        )

    def create_for_task(
        self, spec: EnvSpec, task: EpisodeTask, *, instance_id: str
    ) -> TaskSandboxEnvironment:
        """Build one episode's sandbox, hidden-test vault and environment from ``task``.

        The sandbox is named ``"fsar-<sanitised instance_id>-<random hex>"`` and gets its own
        network namespace with no egress (``network="no-network"``, ``isolate_network=True``);
        the vault lives beside it in ``roots.scratch_root``. Nothing is started here --
        ``start()`` does that.
        """
        config = TaskSandboxConfig.from_task(task)
        roots = self._resolve_roots()
        # A random suffix: a crashed run leaves its container and vault dir behind, and a retry
        # reuses the same instance ids -- a fixed name then collided ("vault target already
        # exists") and turned every retried episode into an infra abstention.
        name = f"fsar-{_sandbox_name(instance_id)}-{secrets.token_hex(3)}"
        sandbox = self._sandbox_factory(
            SandboxSpec(
                name=name,
                image=config.image,
                data_root=roots.data_root,
                image_store=roots.image_store,
                scratch_root=roots.scratch_root,
                enroot_sysconf=roots.enroot_sysconf,
                network="no-network",
                env=dict(spec.env),
                workdir=config.workdir,
                isolate_network=True,
            )
        )
        vault = self._vault_factory(
            sandbox,
            HiddenSpec(paths=config.hidden_paths),
            str(Path(roots.scratch_root) / (name + ".vault")),
        )
        return TaskSandboxEnvironment(spec=spec, config=config, sandbox=sandbox, vault=vault)

    def _resolve_roots(self) -> SandboxRoots:
        """The configured roots, reading ``FS_SANDBOX_*`` once on first use when not given."""
        roots = self._roots
        if roots is None:
            roots = SandboxRoots.from_environ(self._environ)
            self._roots = roots
        return roots

    def _enroot_sandbox(self, spec: SandboxSpec) -> SandboxLikeFull:
        """The default sandbox factory: a real ``EnrootSandbox``.

        Its runner is a :class:`BrokerRunner` when ``FS_SANDBOX_BROKER`` is set (a trainer inside
        a container drives sandboxes through a host broker), else the local subprocess runner.
        """
        return EnrootSandbox(spec, runner=BrokerRunner.from_environ(self._environ))
