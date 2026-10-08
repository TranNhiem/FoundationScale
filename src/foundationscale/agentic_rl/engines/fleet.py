"""A fleet of local SGLang server subprocesses: launch, health-poll, round-robin
client assignment, and concurrent disk weight pushes.

Nothing here imports ``sglang`` itself: ``SGLangServer.start`` launches a
subprocess by COMMAND LINE (``python_bin -m launch_module ...``), the same way
any other process launcher would, and never imports the package into this
process. Tests inject ``launch_module`` to point at a tiny fake HTTP server
script instead of ``sglang.launch_server``, so no test in this plane starts a
real SGLang engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from foundationscale.agentic_rl.engines import EngineInfraError, EngineRefusal
from foundationscale.agentic_rl.engines.sglang import SGLangClient

__all__ = (
    "EngineFleet",
    "SGLangServer",
    "SGLangServerSpec",
)


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


def _non_empty_str(value: object, *, where: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EngineRefusal(
            f"{where}: field {name!r} is {_described(value)}: it must be a non-empty str"
        )
    return value


def _positive_int(value: object, *, where: str, name: str) -> int:
    if type(value) is not int or value < 1:
        raise EngineRefusal(
            f"{where}: field {name!r} is {_described(value)}: it must be a real int "
            f">= 1 (bool excluded)"
        )
    return value


def _positive_float(value: object, *, where: str, name: str) -> float:
    if type(value) is bool or not isinstance(value, (int, float)) or value <= 0:
        raise EngineRefusal(
            f"{where}: field {name!r} is {_described(value)}: it must be a real number "
            f"> 0 (bool excluded)"
        )
    return float(value)


@dataclass(frozen=True)
class SGLangServerSpec:
    """One SGLang server subprocess's declared launch configuration.

    ``launch_module`` defaults to SGLang's own ``sglang.launch_server`` and is
    overridden by tests to point at a fake script -- see the package docstring.
    ``env`` is an explicit allow-list passed verbatim as the subprocess's COMPLETE
    environment (never merged with this process's ``os.environ``), matching
    ``envs.base.EnvSpec.env``'s "nothing is ever inherited" rule: a config value
    this plane needs (e.g. ``PYTHONPATH`` to make a fake launch module importable)
    must be named here, not assumed to leak in from the parent shell.
    """

    model_path: str
    port: int
    tp_size: int
    mem_fraction_static: float | None
    log_dir: str
    extra_args: tuple[str, ...] = ()
    python_bin: str = "python3"
    env: Mapping[str, str] = field(default_factory=dict)
    startup_timeout_s: float = 120.0
    health_poll_interval_s: float = 0.5
    request_timeout_s: float = 60.0
    launch_module: str = "sglang.launch_server"
    host: str = "127.0.0.1"

    def __post_init__(self) -> None:
        where = "SGLangServerSpec"
        _non_empty_str(self.model_path, where=where, name="model_path")
        _positive_int(self.port, where=where, name="port")
        _positive_int(self.tp_size, where=where, name="tp_size")
        if self.mem_fraction_static is not None and (
            type(self.mem_fraction_static) is bool
            or not isinstance(self.mem_fraction_static, (int, float))
            or not 0.0 < self.mem_fraction_static <= 1.0
        ):
            raise EngineRefusal(
                f"{where}: field 'mem_fraction_static' is "
                f"{_described(self.mem_fraction_static)}: it must be None or a real "
                f"number in (0, 1]"
            )
        _non_empty_str(self.log_dir, where=where, name="log_dir")
        if not isinstance(self.extra_args, tuple) or not all(
            isinstance(a, str) for a in self.extra_args
        ):
            raise EngineRefusal(
                f"{where}: field 'extra_args' is {_described(self.extra_args)}: it must be "
                f"a tuple[str, ...]"
            )
        _non_empty_str(self.python_bin, where=where, name="python_bin")
        if not isinstance(self.env, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.env.items()
        ):
            raise EngineRefusal(
                f"{where}: field 'env' is {_described(self.env)}: it must be a "
                f"Mapping[str, str] explicit allow-list -- nothing is ever inherited "
                f"from os.environ"
            )
        object.__setattr__(self, "env", dict(self.env))
        _positive_float(self.startup_timeout_s, where=where, name="startup_timeout_s")
        _positive_float(self.health_poll_interval_s, where=where, name="health_poll_interval_s")
        _positive_float(self.request_timeout_s, where=where, name="request_timeout_s")
        _non_empty_str(self.launch_module, where=where, name="launch_module")
        _non_empty_str(self.host, where=where, name="host")


class SGLangServer:
    """One SGLang server subprocess: launches it, polls ``/health``, and stops it.

    Not frozen/a dataclass: it owns a mutable ``subprocess.Popen`` handle and an
    open log file, neither of which is a value this plane compares or hashes.
    """

    def __init__(self, spec: SGLangServerSpec) -> None:
        self.spec = spec
        self._process: subprocess.Popen[bytes] | None = None
        self.log_path: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.spec.host}:{self.spec.port}"

    def client(self) -> SGLangClient:
        """A fresh ``SGLangClient`` bound to this server; cheap, stateless, callable
        as many times as needed.
        """
        return SGLangClient(base_url=self.base_url, timeout_s=self.spec.request_timeout_s)

    def start(self) -> None:
        """Launch the subprocess and block until ``/health`` answers 200 or
        ``spec.startup_timeout_s`` elapses.

        Raises ``EngineRefusal`` if already started, ``EngineInfraError`` with
        ``kind="startup_crashed"`` if the process exits before becoming healthy, or
        ``kind="startup_timeout"`` if it never does. The log file path is always
        recorded on ``self.log_path`` before either of those can fire.
        """
        if self._process is not None:
            raise EngineRefusal(
                f"SGLangServer.start: method 'start' was called on an already-started "
                f"server on port {self.spec.port}"
            )
        args = [
            self.spec.python_bin,
            "-m",
            self.spec.launch_module,
            "--model-path",
            self.spec.model_path,
            "--port",
            str(self.spec.port),
            "--tp-size",
            str(self.spec.tp_size),
        ]
        if self.spec.mem_fraction_static is not None:
            args += ["--mem-fraction-static", str(self.spec.mem_fraction_static)]
        args += list(self.spec.extra_args)
        log_path = f"{self.spec.log_dir}/sglang_{self.spec.port}.log"
        self.log_path = log_path
        log_file = Path(log_path).open("wb")  # noqa: SIM115 - kept open for the process's lifetime
        try:
            self._process = subprocess.Popen(
                args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=dict(self.spec.env),
                # Its own session (POSIX), so stop() can target the whole process
                # GROUP via os.killpg -- a server that forks tensor-parallel worker
                # processes must not leave them orphaned when this is stopped.
                start_new_session=True,
            )
        finally:
            log_file.close()
        client = self.client()
        deadline = time.monotonic() + self.spec.startup_timeout_s
        while time.monotonic() < deadline:
            returncode = self._process.poll()
            if returncode is not None:
                self.stop()
                raise EngineInfraError(
                    "startup_crashed",
                    f"SGLangServer.start: server on port {self.spec.port} exited with "
                    f"code {returncode} before becoming healthy; log at {log_path}",
                )
            if client.health():
                return
            time.sleep(self.spec.health_poll_interval_s)
        self.stop()
        raise EngineInfraError(
            "startup_timeout",
            f"SGLangServer.start: server on port {self.spec.port} did not become "
            f"healthy within {self.spec.startup_timeout_s}s; log at {log_path}",
        )

    def stop(self, *, grace_s: float = 5.0) -> None:
        """Terminate the subprocess GROUP, then kill the group if it ignores
        ``grace_s`` seconds.

        ``start()`` launches the server in its own session (``start_new_session=
        True``), so this targets the whole process group via ``os.killpg`` rather
        than just the one PID -- a server that forked tensor-parallel worker
        processes must not have them orphaned by a `stop()` that only signals its
        immediate child. Idempotent: a server that was never started, or already
        stopped, returns immediately.
        """
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@dataclass(frozen=True)
class EngineFleet:
    """A fixed set of ``SGLangServer`` instances: round-robin client assignment by
    session index, and a concurrent disk-weight push to every server.
    """

    servers: tuple[SGLangServer, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.servers, tuple) or not self.servers:
            raise EngineRefusal(
                f"EngineFleet: field 'servers' is {_described(self.servers)}: it must be a "
                f"non-empty tuple[SGLangServer, ...] -- a fleet with no server has nothing "
                f"to assign a client to or push weights to"
            )
        for position, server in enumerate(self.servers):
            if not isinstance(server, SGLangServer):
                raise EngineRefusal(
                    f"EngineFleet: field 'servers' entry {position} is "
                    f"{_described(server)}, not an SGLangServer"
                )

    def client_for(self, session_index: int) -> SGLangClient:
        """Round-robin an ``SGLangClient`` by ``session_index % len(servers)``."""
        if type(session_index) is bool or type(session_index) is not int or session_index < 0:
            raise EngineRefusal(
                f"EngineFleet.client_for: parameter 'session_index' is "
                f"{_described(session_index)}: it must be a real int >= 0"
            )
        server = self.servers[session_index % len(self.servers)]
        return server.client()

    def start_all(self) -> None:
        """Start every server in declaration order; a failure leaves earlier servers
        running (the caller decides whether to ``stop_all`` on a partial start).
        """
        for server in self.servers:
            server.start()

    def stop_all(self) -> None:
        """Stop every server; a failure to stop one does not prevent stopping the rest."""
        for server in self.servers:
            server.stop()

    async def update_weights_from_disk(self, model_path: str) -> tuple[bool, ...]:
        """Push ``model_path`` to every server concurrently.

        Returns one bool per server, in ``self.servers`` order: ``True`` if that
        server's ``update_weights_from_disk`` call succeeded, ``False`` if it raised
        ``EngineInfraError``. Any other exception propagates -- it is a programming
        error, not a per-server infra fault this method's contract reports.
        """

        async def _one(server: SGLangServer) -> bool:
            try:
                await asyncio.to_thread(server.client().update_weights_from_disk, model_path)
            except EngineInfraError:
                return False
            return True

        results = await asyncio.gather(*(_one(server) for server in self.servers))
        return tuple(results)
