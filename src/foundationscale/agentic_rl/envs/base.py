"""The Environment / EnvBackend protocols and the EnvSpec/ExecResult/EnvCapabilities
contracts a backend must satisfy; a registry of named backends; the isolation
refusal that keeps a model's shell commands off a bare host by accident.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "EnvBackend",
    "EnvCapabilities",
    "EnvInfraError",
    "EnvRefusal",
    "EnvSpec",
    "Environment",
    "ExecResult",
    "available_env_backends",
    "check_isolation",
    "get_env_backend",
    "register_env_backend",
]


def _described(value: object) -> str:
    """Short ``type-name repr`` rendering of a rejected value for refusal messages."""
    return f"{type(value).__name__} {value!r}"


class EnvRefusal(ValueError):
    """A programming/config error in an env declaration or call -- not a runtime infra fault."""


class EnvInfraError(RuntimeError):
    """A RUNTIME fault (start failed, transport broke, env killed) whose ``kind`` the harness
    maps to ``abstention_reason=f"infra:{kind}"`` and ``Termination.INFRA``.
    """

    def __init__(self, kind: str, message: str) -> None:
        """Store the machine-readable ``kind`` tag ("start_failed", "transport", "killed",
        "setup_failed", ...) that ends up in ``infra:{kind}``; refuse a non-str or empty ``kind``
        because an untagged infra error cannot be routed.
        """
        if not isinstance(kind, str):
            raise EnvRefusal(
                f"EnvInfraError.__init__: parameter 'kind' is {_described(kind)}: an infra "
                f"error kind is a short machine-readable str tag -- a non-str kind cannot be "
                f"routed into abstention_reason"
            )
        if not kind:
            raise EnvRefusal(
                f"EnvInfraError.__init__: parameter 'kind' is {_described(kind)}: an infra "
                f"error kind is a non-empty str tag -- an empty kind cannot be routed into "
                f"abstention_reason"
            )
        if not isinstance(message, str):
            raise EnvRefusal(
                f"EnvInfraError.__init__: parameter 'message' is {_described(message)}: an "
                f"infra error message is a str -- a non-str message cannot be reported to a "
                f"harness"
            )
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class EnvSpec:
    """The declarative request for one environment instance: ``workdir_template`` is
    backend-interpreted, while the timeouts, output cap and ``env`` allow-list are backend-agnostic
    contracts every backend honours.

    ``env`` is the EXPLICIT allow-list passed to the spawned process -- nothing from
    this (the caller's own) process's ``os.environ`` is ever inherited. A backend MAY
    add a small, fixed set of its own operational variables on top of this allow-list
    for process-management needs it alone has (e.g. giving a model-chosen command a
    sane home directory); any such addition is named explicitly in that backend's own
    docstring, naming exactly which variables and why, so this allow-list contract
    stays truthful. See ``envs.local.LocalEnvironment`` for the one backend that does
    this today (it adds ``HOME`` and ``PWD``).
    """

    backend: str
    env: Mapping[str, str]
    workdir_template: str | None = None
    exec_timeout_s: float = 120.0
    episode_timeout_s: float = 3600.0
    max_output_bytes: int = 65536
    setup_commands: tuple[str, ...] = ()
    allow_unsandboxed: bool = False

    def __post_init__(self) -> None:
        """Freeze ``env``/``setup_commands`` into canonical read-only copies and refuse any
        field that violates its contract, so a bad declaration fails at construction rather
        than mid-episode.
        """
        where = "EnvSpec.__post_init__"
        if not isinstance(self.backend, str):
            raise EnvRefusal(
                f"{where}: field 'backend' is {_described(self.backend)}: a spec names its "
                f"backend with a non-empty str -- a non-str name cannot be looked up in the "
                f"registry"
            )
        if not self.backend:
            raise EnvRefusal(
                f"{where}: field 'backend' is {_described(self.backend)}: a spec names its "
                f"backend with a non-empty str -- an empty name matches no registered backend"
            )
        if self.workdir_template is not None:
            if not isinstance(self.workdir_template, str):
                raise EnvRefusal(
                    f"{where}: field 'workdir_template' is {_described(self.workdir_template)}: "
                    f"workdir_template is None or a non-empty str -- a non-str template cannot "
                    f"be interpreted by the backend"
                )
            if not self.workdir_template:
                raise EnvRefusal(
                    f"{where}: field 'workdir_template' is {_described(self.workdir_template)}: "
                    f"workdir_template is None or a non-empty str -- an empty template names no "
                    f"working directory"
                )
        for field_name, value in (
            ("exec_timeout_s", self.exec_timeout_s),
            ("episode_timeout_s", self.episode_timeout_s),
        ):
            if type(value) is bool:
                raise EnvRefusal(
                    f"{where}: field '{field_name}' is {_described(value)}: a timeout is a "
                    f"duration, not a flag -- bools are refused outright so True is never read "
                    f"as 1 second"
                )
            if not isinstance(value, (int, float)):
                raise EnvRefusal(
                    f"{where}: field '{field_name}' is {_described(value)}: a timeout is an "
                    f"int or float duration -- a non-numeric value cannot bound a command"
                )
            if not math.isfinite(value):
                raise EnvRefusal(
                    f"{where}: field '{field_name}' is {_described(value)}: a timeout is a "
                    f"finite duration greater than 0 -- nan/inf would never fire, leaving a "
                    f"hung command unbounded"
                )
            if not value > 0:
                raise EnvRefusal(
                    f"{where}: field '{field_name}' is {_described(value)}: a timeout is a "
                    f"finite duration greater than 0 -- a non-positive timeout would kill every "
                    f"command before it runs"
                )
            object.__setattr__(self, field_name, float(value))
        if type(self.max_output_bytes) is not int:
            raise EnvRefusal(
                f"{where}: field 'max_output_bytes' is {_described(self.max_output_bytes)}: the "
                f"output cap is a real int (type(x) is int) -- a bool or float cap cannot be "
                f"counted in bytes"
            )
        if not self.max_output_bytes > 0:
            raise EnvRefusal(
                f"{where}: field 'max_output_bytes' is {_described(self.max_output_bytes)}: the "
                f"output cap is greater than 0 -- a zero or negative cap would drop every byte "
                f"of output"
            )
        if not isinstance(self.env, Mapping):
            raise EnvRefusal(
                f"{where}: field 'env' is {_described(self.env)}: env is a Mapping[str, str] "
                f"allow-list -- nothing is ever inherited from os.environ, so the mapping must "
                f"be given explicitly"
            )
        frozen_env: dict[str, str] = {}
        for env_key, env_value in self.env.items():
            if not isinstance(env_key, str):
                raise EnvRefusal(
                    f"{where}: field 'env' key at position {len(frozen_env)} is "
                    f"{_described(env_key)}: every allow-listed variable name is a str -- a "
                    f"non-str name cannot be passed to a spawned process"
                )
            if not isinstance(env_value, str):
                raise EnvRefusal(
                    f"{where}: field 'env' value for key {env_key!r} at position "
                    f"{len(frozen_env)} is {_described(env_value)}: every allow-listed variable "
                    f"value is a str -- a non-str value cannot be passed to a spawned process"
                )
            frozen_env[env_key] = env_value
        object.__setattr__(self, "env", MappingProxyType(frozen_env))
        declared_setup: object = self.setup_commands
        if isinstance(declared_setup, (str, bytes, bytearray)):
            raise EnvRefusal(
                f"{where}: field 'setup_commands' is {_described(declared_setup)}: "
                f"setup_commands is a Sequence[str] of separate commands -- a bare str would be "
                f"iterated character by character"
            )
        if not isinstance(declared_setup, Sequence):
            raise EnvRefusal(
                f"{where}: field 'setup_commands' is {_described(declared_setup)}: "
                f"setup_commands is a Sequence[str] coerced to a tuple -- a non-sequence cannot "
                f"be ordered into setup steps"
            )
        frozen_setup: list[str] = []
        for index, command in enumerate(declared_setup):
            if not isinstance(command, str):
                raise EnvRefusal(
                    f"{where}: field 'setup_commands' element at index {index} is "
                    f"{_described(command)}: every setup command is a non-empty str -- a "
                    f"non-str element cannot be run"
                )
            if not command:
                raise EnvRefusal(
                    f"{where}: field 'setup_commands' element at index {index} is "
                    f"{_described(command)}: every setup command is a non-empty str -- an empty "
                    f"command names nothing to run"
                )
            frozen_setup.append(command)
        object.__setattr__(self, "setup_commands", tuple(frozen_setup))
        if type(self.allow_unsandboxed) is not bool:
            raise EnvRefusal(
                f"{where}: field 'allow_unsandboxed' is {_described(self.allow_unsandboxed)}: "
                f"allow_unsandboxed is a real bool (type(x) is bool) -- 1 is not True, and an "
                f"int opt-in would silently permit bare-host execution"
            )


@dataclass(frozen=True)
class ExecResult:
    """One finished (or timed-out) command: decoded output, exit status, wall time and the measured
    output bytes the cap dropped.
    """

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    truncated_bytes: int
    seconds: float

    def __post_init__(self) -> None:
        """Coerce ``seconds`` to float and refuse any result whose fields contradict each
        other, so a caller never guesses at an exit status.
        """
        where = "ExecResult.__post_init__"
        if not isinstance(self.stdout, str):
            raise EnvRefusal(
                f"{where}: field 'stdout' is {_described(self.stdout)}: stdout is a decoded "
                f"str -- a non-str cannot be scored or shown"
            )
        if not isinstance(self.stderr, str):
            raise EnvRefusal(
                f"{where}: field 'stderr' is {_described(self.stderr)}: stderr is a decoded "
                f"str -- a non-str cannot be scored or shown"
            )
        if type(self.timed_out) is not bool:
            raise EnvRefusal(
                f"{where}: field 'timed_out' is {_described(self.timed_out)}: timed_out is a "
                f"real bool (type(x) is bool) -- 1 is not True, and an int flag would make the "
                f"exit_code rule ambiguous"
            )
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise EnvRefusal(
                f"{where}: field 'exit_code' is {_described(self.exit_code)}: exit_code is None "
                f"or a real int (type(x) is int) -- a bool or float status cannot be reported "
                f"as a process exit code"
            )
        if self.timed_out and self.exit_code is not None:
            raise EnvRefusal(
                f"{where}: field 'exit_code' is {_described(self.exit_code)} but field "
                f"'timed_out' is {_described(self.timed_out)}: a timed-out command has no exit "
                f"code -- reporting one would invent a status the killed process never produced"
            )
        if not self.timed_out and self.exit_code is None:
            raise EnvRefusal(
                f"{where}: field 'exit_code' is {_described(self.exit_code)} but field "
                f"'timed_out' is {_described(self.timed_out)}: a non-timed-out command must "
                f"report an exit code, even 0 -- None here would hide a real status behind the "
                f"'unmeasured' convention"
            )
        if type(self.truncated_bytes) is not int:
            raise EnvRefusal(
                f"{where}: field 'truncated_bytes' is {_described(self.truncated_bytes)}: the "
                f"dropped-byte count is a real int (type(x) is int) -- a bool or float cannot "
                f"be counted in bytes"
            )
        if self.truncated_bytes < 0:
            raise EnvRefusal(
                f"{where}: field 'truncated_bytes' is {_described(self.truncated_bytes)}: the "
                f"dropped-byte count is >= 0 -- a negative count would claim output was added "
                f"by the cap"
            )
        if type(self.seconds) is bool:
            raise EnvRefusal(
                f"{where}: field 'seconds' is {_described(self.seconds)}: an elapsed time is a "
                f"duration, not a flag -- bools are refused outright so True is never read as "
                f"1.0 second"
            )
        if not isinstance(self.seconds, (int, float)):
            raise EnvRefusal(
                f"{where}: field 'seconds' is {_described(self.seconds)}: an elapsed time is an "
                f"int or float duration -- a non-numeric value cannot be compared to a timeout"
            )
        if not math.isfinite(self.seconds):
            raise EnvRefusal(
                f"{where}: field 'seconds' is {_described(self.seconds)}: an elapsed time is "
                f"finite -- nan/inf would poison every timing aggregate"
            )
        if self.seconds < 0:
            raise EnvRefusal(
                f"{where}: field 'seconds' is {_described(self.seconds)}: an elapsed time is "
                f">= 0 -- a negative duration never happened"
            )
        object.__setattr__(self, "seconds", float(self.seconds))


@dataclass(frozen=True)
class EnvCapabilities:
    """What a backend can actually provide: isolation level, network reach, concurrency ceiling and
    host platform.
    """

    isolation: Literal["none", "container", "vm", "remote"]
    network: bool
    max_concurrency: int | None
    platform: str

    def __post_init__(self) -> None:
        """Refuse an illegal isolation level or a malformed flag/ceiling, so a harness can
        trust these capabilities when it routes episodes.
        """
        where = "EnvCapabilities.__post_init__"
        allowed = ("none", "container", "vm", "remote")
        if self.isolation not in allowed:
            raise EnvRefusal(
                f"{where}: field 'isolation' is {_described(self.isolation)}: isolation is one "
                f"of {sorted(allowed)} -- an unknown level cannot be checked against "
                f"EnvSpec.allow_unsandboxed"
            )
        if type(self.network) is not bool:
            raise EnvRefusal(
                f"{where}: field 'network' is {_described(self.network)}: network is a real "
                f"bool (type(x) is bool) -- 1 is not True, and an int flag would make reach "
                f"ambiguous"
            )
        if self.max_concurrency is not None:
            if type(self.max_concurrency) is not int:
                raise EnvRefusal(
                    f"{where}: field 'max_concurrency' is {_described(self.max_concurrency)}: "
                    f"max_concurrency is None or a real int (type(x) is int) -- a bool or float "
                    f"ceiling cannot bound concurrent instances"
                )
            if self.max_concurrency < 1:
                raise EnvRefusal(
                    f"{where}: field 'max_concurrency' is {_described(self.max_concurrency)}: "
                    f"max_concurrency is None or >= 1 -- a lower ceiling would forbid the "
                    f"instance being created right now"
                )
        if not isinstance(self.platform, str):
            raise EnvRefusal(
                f"{where}: field 'platform' is {_described(self.platform)}: platform is a "
                f"non-empty str -- a non-str platform cannot be matched to a host"
            )
        if not self.platform:
            raise EnvRefusal(
                f"{where}: field 'platform' is {_described(self.platform)}: platform is a "
                f"non-empty str -- an empty platform names no host"
            )


@runtime_checkable
class Environment(Protocol):
    """One prepared environment instance a harness runs commands in."""

    async def start(self) -> None:
        """Prepare the instance (e.g. a temp dir); called exactly once before ``execute``."""

    async def execute(self, command: str, *, timeout_s: float | None = None) -> ExecResult:
        """Runs one command, with ``timeout_s`` overriding ``EnvSpec.exec_timeout_s`` for this call
        only; the implementation raises ``EnvRefusal`` if called before ``start()`` or after
        ``close()``.
        """

    async def close(self) -> None:
        """Idempotent; releases resources (e.g. removes a temp dir) -- calling it twice is not an
        error.
        """

    @property
    def workdir(self) -> str:
        """The instance's working directory path."""

    @property
    def episode_timeout_s(self) -> float:
        """The episode's wall-clock budget (``EnvSpec.episode_timeout_s``), read-only.

        A harness's ``run_episode`` receives ``env`` but not the ``EnvSpec`` that
        created it; this is the one value a harness's TIMEOUT rule needs from that
        spec, so it is exposed here instead of widening every harness signature.
        """


@runtime_checkable
class EnvBackend(Protocol):
    """A named factory of ``Environment`` instances with declared capabilities."""

    name: str

    def capabilities(self) -> EnvCapabilities:
        """What this backend provides: isolation, network, concurrency and platform."""

    def create(self, spec: EnvSpec, *, instance_id: str) -> Environment:
        """Constructs (but does not start) one ``Environment`` for ``spec``; ``instance_id`` is a
        caller-chosen unique string the backend may use to namespace resources (e.g. a temp dir
        name).
        """


_REGISTRY: dict[str, Callable[[], EnvBackend]] = {}


def register_env_backend(name: str, factory: Callable[[], EnvBackend]) -> None:
    """Register ``factory`` under ``name``; refuse a duplicate name so two backends never silently
    shadow one another.
    """
    where = "register_env_backend"
    if not isinstance(name, str):
        raise EnvRefusal(
            f"{where}: parameter 'name' is {_described(name)}: a backend name is a non-empty "
            f"str -- a non-str name cannot be looked up later"
        )
    if not name:
        raise EnvRefusal(
            f"{where}: parameter 'name' is {_described(name)}: a backend name is a non-empty "
            f"str -- an empty name matches no lookup"
        )
    if not callable(factory):
        raise EnvRefusal(
            f"{where}: parameter 'factory' for name {name!r} is {_described(factory)}: a "
            f"backend factory is callable -- a non-callable cannot produce a fresh backend"
        )
    if name in _REGISTRY:
        registered = tuple(sorted(_REGISTRY))
        raise EnvRefusal(
            f"{where}: parameter 'name' is {_described(name)} but name {name!r} is already "
            f"registered: a backend name is registered exactly once -- a duplicate would "
            f"silently shadow the earlier backend; currently registered: {list(registered)}"
        )
    _REGISTRY[name] = factory


def get_env_backend(name: str) -> EnvBackend:
    """Call the stored factory and return a fresh backend instance; refuse an unknown name so a typo
    fails before any episode starts.
    """
    where = "get_env_backend"
    if not isinstance(name, str):
        raise EnvRefusal(
            f"{where}: parameter 'name' is {_described(name)}: a backend name is a non-empty "
            f"str -- a non-str name cannot match a registration"
        )
    if not name:
        raise EnvRefusal(
            f"{where}: parameter 'name' is {_described(name)}: a backend name is a non-empty "
            f"str -- an empty name matches no registration"
        )
    if name not in _REGISTRY:
        registered = tuple(sorted(_REGISTRY))
        raise EnvRefusal(
            f"{where}: parameter 'name' is {_described(name)}: no backend is registered under "
            f"{name!r} -- an unknown name would leave the episode with nowhere to run; "
            f"currently registered: {list(registered)}"
        )
    return _REGISTRY[name]()


def available_env_backends() -> tuple[str, ...]:
    """Sorted tuple of registered backend names."""
    return tuple(sorted(_REGISTRY))


def check_isolation(spec: EnvSpec, caps: EnvCapabilities) -> None:
    """Refuse to run a backend with ``isolation == "none"`` unless
    ``spec.allow_unsandboxed`` is True -- the local backend runs model-chosen shell
    commands directly on the host, so that must be opted into explicitly via
    ``EnvSpec(allow_unsandboxed=True)``.
    """
    where = "check_isolation"
    if not isinstance(spec, EnvSpec):
        raise EnvRefusal(
            f"{where}: parameter 'spec' is {_described(spec)}: isolation is checked against an "
            f"EnvSpec -- a non-spec carries no allow_unsandboxed decision"
        )
    if not isinstance(caps, EnvCapabilities):
        raise EnvRefusal(
            f"{where}: parameter 'caps' is {_described(caps)}: isolation is checked against "
            f"EnvCapabilities -- non-capabilities declare no isolation level"
        )
    if caps.isolation == "none" and spec.allow_unsandboxed is not True:
        raise EnvRefusal(
            f"{where}: field 'allow_unsandboxed' is {_described(spec.allow_unsandboxed)} but "
            f"capabilities.isolation is {_described(caps.isolation)}: the local backend runs "
            f"model-chosen shell commands directly on the host, and bare-host execution must be "
            f"opted into explicitly via EnvSpec(allow_unsandboxed=True) -- running without that "
            f"opt-in would let a model alter the host by accident"
        )


def _make_local_backend() -> EnvBackend:
    """Build one local backend; imports ``envs.local`` lazily to avoid a circular import."""
    from foundationscale.agentic_rl.envs.local import LocalSubprocessBackend

    return LocalSubprocessBackend()


register_env_backend("local", _make_local_backend)
