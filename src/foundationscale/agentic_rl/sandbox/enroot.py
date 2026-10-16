"""Enroot-backed sandboxes for black-box agents.

Stdlib only. Every external command goes through an injectable ``Runner`` so
tests can substitute a fake.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import IO, Protocol

_NETNS_HOLDER_MAX_S = (
    172800  # 2 days: far beyond any trial; bounds a holder leaked by an external kill
)
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,63}$")
_NETWORKS = ("no-network", "public", "allowlist")
_PROXY_PORT = 3128  # the egress proxy, as seen from inside an isolated sandbox (127.0.0.1)
_ENDPOINT_RE = re.compile(r"^(\d{1,5})=([A-Za-z0-9_.-]+|\[[0-9A-Fa-f:]+\]):(\d{1,5})$")
_STDERR_TAIL = 2000
# After the exec process exits, how long its output readers may keep draining. A process the
# command backgrounded (a server, a VM) inherits the pipes and holds them open; waiting for EOF
# then blocked the exec until that process died, past its own timeout (B2.2 incident 6).
# Docker's exec returns when its process exits.
_DRAIN_GRACE_S = 2.0


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


def _memory_scope(memory_mb: int) -> list[str]:
    """argv prefix running a command in its own cgroup v2 scope capped at ``memory_mb`` (no
    swap)."""
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={memory_mb}M",
        "-p",
        "MemorySwapMax=0",
        # Docker kills only the offending process on OOM; systemd's default stops the whole scope
        "-p",
        "OOMPolicy=continue",
        "--",
    ]


class SubprocessRunner:
    """``subprocess.run`` based runner; timeouts kill the process group."""

    def __init__(self) -> None:
        self._live: set[int] = set()
        self._spawned: dict[int, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()

    def kill_all(self) -> None:
        """SIGKILL the process group of every in-flight run (each run is its own session)."""
        with self._lock:
            groups = list(self._live)
        for pgid in groups:
            with contextlib.suppress(OSError):
                os.killpg(pgid, signal.SIGKILL)

    def spawn(self, argv: Sequence[str], *, ready_timeout_s: float = 30.0) -> int:
        """Start a long-lived helper (its own session) and return its pid once it prints "ready".

        Waiting matters: a pid exists before the helper's own setup (e.g. ``unshare``) has run,
        and a caller that acts on it at once raced it (nsenter: "reassociate to namespace
        'ns/user' failed")."""
        try:
            proc = subprocess.Popen(  # noqa: S603
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise SandboxError(f"failed to spawn {list(argv)!r}: {exc}") from exc
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], ready_timeout_s)
        line = proc.stdout.readline() if ready else b""
        proc.stdout.close()
        if line.strip() != b"ready":
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise SandboxError(
                f"helper {list(argv)!r}: expected 'ready' within {ready_timeout_s}s, got {line!r}"
            )
        with self._lock:
            self._spawned[proc.pid] = proc
        return proc.pid

    def terminate(self, pid: int) -> None:
        """SIGKILL a helper started by spawn() and reap it."""
        with self._lock:
            proc = self._spawned.pop(pid, None)
        with contextlib.suppress(OSError):
            os.killpg(pid, signal.SIGKILL)
        if proc is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)

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
                env=run_env,
                start_new_session=True,
            )
        except OSError as exc:
            raise SandboxError(f"failed to spawn {list(argv)!r}: {exc}") from exc
        with self._lock:
            self._live.add(proc.pid)  # start_new_session: pgid == pid
        assert proc.stdout is not None and proc.stderr is not None
        out: list[bytes] = []
        err: list[bytes] = []
        readers = [
            threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True),
        ]
        for reader in readers:
            reader.start()
        timed_out = False
        try:
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        finally:
            with self._lock:
                self._live.discard(proc.pid)
        deadline = (
            time.monotonic() + _DRAIN_GRACE_S
        )  # one shared grace: a pipe still held by a backgrounded
        for reader in readers:  # process must not block the exec
            reader.join(max(0.0, deadline - time.monotonic()))
        stdout = b"".join(out).decode("utf-8", "replace")
        stderr = b"".join(err).decode("utf-8", "replace")
        if timed_out:
            return RunResult(returncode=124, stdout=stdout, stderr=stderr, timed_out=True)
        return RunResult(returncode=proc.returncode, stdout=stdout, stderr=stderr)


def _drain(stream: IO[bytes], sink: list[bytes]) -> None:
    """Read a pipe to EOF into sink; may outlive its exec while a background process holds it."""
    with contextlib.suppress(OSError, ValueError):
        while chunk := os.read(stream.fileno(), 65536):
            sink.append(chunk)


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
    # Optional enroot config dir (ENROOT_SYSCONF_PATH) for all enroot calls: a copy of
    # /etc/enroot that drops the
    # site's tmpfs over /tmp (mode 755, a separate fs: rename into /tmp failed EXDEV; Docker's
    # /tmp is the container fs)
    enroot_sysconf: str | None = None
    # Docker-like network isolation for task sandboxes: an own network namespace for every
    # network mode; egress
    # only through a per-sandbox proxy that enforces the policy ("public": any host;
    # "allowlist": allowed_hosts;
    # "no-network": none). Without it, "public" shared the host's network: TB 2.1 nginx could
    # not bind the
    # host-taken :8080 and mailman could not bind :25. Builds keep the host network
    # (isolate_network False).
    isolate_network: bool = False
    allowed_hosts: tuple[str, ...] = ()
    # "PORT=HOST:HOSTPORT": host service reachable inside the sandbox at 127.0.0.1:PORT (e.g.
    # the agent's model
    # endpoint; Node clients ignore proxy env vars, so it is bridged directly rather than proxied)
    endpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise SandboxRefusal(f"name: expected [A-Za-z0-9_.-]{{1,63}}, got {self.name!r}")
        if self.enroot_sysconf is not None and not (
            self.enroot_sysconf.startswith("/")
            and Path(self.enroot_sysconf, "enroot.conf").is_file()
        ):
            raise SandboxRefusal(
                f"enroot_sysconf: expected an absolute enroot config dir with "
                f"enroot.conf, got {self.enroot_sysconf!r}"
            )
        for field_name in ("data_root", "image_store", "scratch_root"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise SandboxRefusal(f"{field_name}: expected absolute path, got {value!r}")
        if self.network not in _NETWORKS:
            raise SandboxRefusal(f"network: expected one of {_NETWORKS}, got {self.network!r}")
        if self.network == "allowlist" and not self.isolate_network:
            raise SandboxRefusal(
                "network='allowlist': expected isolate_network=True (it is enforced by the proxy)"
            )
        for endpoint in self.endpoints:
            if not _ENDPOINT_RE.match(endpoint):
                raise SandboxRefusal(f"endpoints: expected 'PORT=HOST:HOSTPORT', got {endpoint!r}")
        if self.endpoints and not (self.isolate_network or self.network == "no-network"):
            raise SandboxRefusal(
                "endpoints: expected an isolated or no-network sandbox (host network needs none)"
            )
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
        self._netns_pid: int | None = None
        self._helpers: list[int] = []  # egress proxy and socat bridges (terminated by stop())
        self.entrypoint: tuple[str, ...] | None = (
            None  # the image's ENTRYPOINT, from build metadata
        )
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
        # enroot's per-user layer cache is not safe for concurrent imports that share
        # layers (two trials pulling
        # ubuntu:24.04 at once: "mv: not replacing .../enroot-cache/<layer>"), so imports
        # are serialized with a
        # file lock that covers threads and separate Harbor processes alike; an importer
        # that waited reuses the
        # image a concurrent one already produced.
        Path(self.spec.data_root).mkdir(parents=True, exist_ok=True)
        with (Path(self.spec.data_root) / ".enroot-import.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if Path(path).exists():
                return path
            return self._import_locked(image, path)

    def _import_locked(self, image: str, path: str) -> str:
        # A private temp name per import: concurrent sandboxes sharing a base image (R0
        # gate workers) each
        # wrote and deleted one shared "<digest>.sqsh.partial", so mksquashfs read the
        # other's half-written
        # file. Each importer now writes its own file; the atomic replace below makes the
        # last complete one win.
        partial = f"{path}.{uuid.uuid4().hex}.partial"
        result = self._runner.run(
            ["enroot", "import", "-o", partial, image],
            env=self._enroot_env(),
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
        if self.spec.memory_mb:
            # Fail closed: a requested memory limit that cannot be enforced must not run unbounded.
            probe = self._runner.run([*_memory_scope(self.spec.memory_mb), "true"])
            if probe.returncode != 0:
                raise SandboxError(
                    f"memory_mb={self.spec.memory_mb}: expected a cgroup v2 memory scope "
                    f"via systemd --user, "
                    f"got exit {probe.returncode}: {probe.stderr[-_STDERR_TAIL:]}"
                )
        result = self._runner.run(
            ["enroot", "create", "-n", self.spec.name, image_path],
            env=self._enroot_env(),
        )
        if result.returncode != 0:
            raise SandboxError(
                f"enroot create failed for {self.spec.name!r} (exit {result.returncode}): "
                f"{result.stderr[-_STDERR_TAIL:]}"
            )
        # /tmp lives in the container's own (writable) rootfs, as in Docker. A scratch
        # bind-mount over /tmp made
        # rename(2) into /tmp cross filesystems (TB 2.1 path-tracing-reverse: "Invalid
        # cross-device link").
        tmp = Path(self.rootfs()) / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        tmp.chmod(0o1777)
        (Path(self.spec.scratch_root) / self.spec.name).mkdir(
            parents=True, exist_ok=True
        )  # per-sandbox host scratch
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
            entrypoint = raw.get("entrypoint")
            if entrypoint is not None and not (
                isinstance(entrypoint, list) and all(isinstance(a, str) for a in entrypoint)
            ):
                raise SandboxError(f"invalid build metadata {metadata}: bad 'entrypoint'")
            self._build_env = MappingProxyType(dict(env))
            self._build_workdir = workdir
            self.entrypoint = tuple(entrypoint) if entrypoint else None
        if self.spec.network == "no-network" or self.spec.isolate_network:
            # One network namespace for the sandbox's lifetime, as a Docker container
            # has: every exec joins it.
            # A namespace per exec meant a service started by one exec (the oracle's
            # mongod) was unreachable
            # from the next (the verifier: "Connection refused"). Loopback up, no
            # route off the box; the
            # holder's finite sleep bounds a leak if this process is killed before stop().
            spawn = getattr(self._runner, "spawn", None)
            if not callable(spawn):
                raise SandboxRefusal(
                    "network='no-network': expected a runner with spawn(), got none"
                )
            self._netns_pid = spawn(
                [
                    "unshare",
                    "-rn",
                    "sh",
                    "-c",
                    # ports < 1024 (mailman's :25): container root runs in a nested user
                    # namespace without caps over this netns, so the netns owner lowers the
                    # unprivileged-port floor, as Docker root can bind them in its own netns
                    "ip link set lo up 2>/dev/null; echo 0 > /proc/sys/net/ipv4/"
                    f"ip_unprivileged_port_start 2>/dev/null; echo ready; "
                    f"exec sleep {_NETNS_HOLDER_MAX_S}",
                ]
            )
            if self.spec.isolate_network:
                self._write_proxy_config()
                self.set_policy(self.spec.network, self.spec.allowed_hosts)
                sock = self._socket_path("egress")
                self._helpers.append(
                    spawn(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("egress_proxy.py")),
                            "--socket",
                            str(sock),
                            "--policy",
                            str(self._policy_path()),
                            "--log",
                            str(self._egress_log_path()),
                        ]
                    )
                )
                self._bridge_into_netns(_PROXY_PORT, sock)
            for i, endpoint in enumerate(self.spec.endpoints):
                match = _ENDPOINT_RE.match(endpoint)
                assert match is not None  # validated by SandboxSpec.__post_init__
                port, host, host_port = match.groups()
                sock = self._socket_path(f"ep{i}")
                self._helpers.append(
                    spawn(
                        [
                            "sh",
                            "-c",
                            f"echo ready; exec socat UNIX-LISTEN:{sock},fork,mode=600,"
                            f"unlink-early TCP:{host.strip('[]')}:{host_port}",
                        ]
                    )
                )
                self._bridge_into_netns(int(port), sock)
        self._started = True

    # -- network helpers ---------------------------------------------------

    def _write_proxy_config(self) -> None:
        """System-level proxy settings for clients that drop the env: sudo resets the
        environment (TB 2.1 rstan-to-pystan: `sudo apt-get update` -> "Temporary failure
        resolving ports.ubuntu.com"), and apt reads
        its own config. Docker containers need neither because their egress is not proxied."""
        root = Path(self.rootfs())
        proxy = f"http://127.0.0.1:{_PROXY_PORT}"
        sudoers = root / "etc" / "sudoers.d"
        sudoers.mkdir(parents=True, exist_ok=True)
        keep = root / "etc" / "sudoers.d" / "fs-egress-proxy"
        keep.write_text(
            'Defaults env_keep += "http_proxy https_proxy HTTP_PROXY HTTPS_PROXY '
            'no_proxy NO_PROXY"\n'
        )
        keep.chmod(0o440)
        if (root / "etc" / "apt").is_dir():
            conf = root / "etc" / "apt" / "apt.conf.d"
            conf.mkdir(parents=True, exist_ok=True)
            (conf / "95fs-egress-proxy").write_text(
                f'Acquire::http::Proxy "{proxy}";\nAcquire::https::Proxy "{proxy}";\n'
            )

    def _socket_path(self, role: str) -> Path:
        # AF_UNIX paths are limited to ~108 bytes and sandbox names are long, so sockets
        # live in a short dir
        tag = hashlib.sha256(f"{self.spec.scratch_root}/{self.spec.name}".encode()).hexdigest()[:16]
        base = Path.home() / ".fs-sock"
        base.mkdir(mode=0o700, exist_ok=True)
        return base / f"{tag}-{role}.sock"

    def _egress_log_path(self) -> Path:
        # outside the per-sandbox scratch dir, which stop() removes: the record outlives the sandbox
        return Path(self.spec.scratch_root) / "egress-logs" / f"{self.spec.name}.log"

    def _policy_path(self) -> Path:
        return Path(self.spec.scratch_root) / self.spec.name / "egress-policy.json"

    def set_policy(self, mode: str, hosts: Sequence[str] = ()) -> None:
        """Atomically replace the egress policy of an isolated sandbox (Harbor phase switches)."""
        if mode not in _NETWORKS:
            raise SandboxRefusal(f"mode: expected one of {_NETWORKS}, got {mode!r}")
        path = self._policy_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps({"mode": mode, "hosts": list(hosts)}), encoding="utf-8")
        tmp.replace(path)

    def _bridge_into_netns(self, port: int, sock: Path) -> None:
        """socat inside the sandbox netns: 127.0.0.1:port -> host Unix socket; wait until it
        listens."""
        spawn = getattr(self._runner, "spawn", None)
        if not callable(spawn):
            raise SandboxRefusal("network bridge: expected a runner with spawn(), got none")
        join = [
            "nsenter",
            f"--target={self._netns_pid}",
            "--user",
            "--net",
            "--preserve-credentials",
            "--",
        ]
        self._helpers.append(
            spawn(
                [
                    *join,
                    "sh",
                    "-c",
                    f"echo ready; exec socat TCP-LISTEN:{port},bind=127.0.0.1,"
                    f"fork,reuseaddr UNIX-CONNECT:{sock}",
                ]
            )
        )
        probe = self._runner.run(
            [
                *join,
                "bash",
                "-c",
                f"for i in $(seq 1 200); do (exec 3<>/dev/tcp/127.0.0.1/{port}) "
                f"2>/dev/null && exit 0; sleep 0.05; done; exit 1",
            ],
            timeout_s=30,
        )
        if probe.returncode != 0:
            raise SandboxError(
                f"network bridge to {sock} did not start listening on 127.0.0.1:{port}"
            )

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
        if not self._image_env_defines("HOME"):
            # Docker sets HOME from /etc/passwd (/root: execs run as root); without a
            # login shell nothing
            # else does, and Go/pip/npm/git failed ("neither $XDG_CACHE_HOME nor $HOME
            # are defined").
            merged["HOME"] = "/root"
        if self.spec.isolate_network:
            proxy = f"http://127.0.0.1:{_PROXY_PORT}"
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                merged[key] = proxy
            merged["NO_PROXY"] = merged["no_proxy"] = "localhost,127.0.0.1,::1"
        merged.update(self._build_env)
        merged.update(self.spec.env)
        if env:
            merged.update(env)
        workdir = cwd if cwd is not None else (self._build_workdir or self.spec.workdir)
        argv: list[str] = []
        if self.spec.memory_mb:
            # A cgroup v2 memory limit, as Docker's --memory: it counts memory
            # actually resident. rlimits cannot
            # do this: RLIMIT_AS counted V8/WebAssembly/Chromium PROT_NONE
            # reservations and RLIMIT_DATA counts
            # ASan's 8 GiB MAP_NORESERVE shadow, so Node and sanitizer builds could
            # not start under a 4 GiB
            # task limit. Measured: an 8 GiB untouched reservation runs, touching 512
            # MiB under 256 MiB is killed.
            argv += _memory_scope(self.spec.memory_mb)
        if self._netns_pid is not None:
            # join the sandbox's network namespace (and its user namespace, which owns it)
            argv += [
                "nsenter",
                f"--target={self._netns_pid}",
                "--user",
                "--net",
                "--preserve-credentials",
                "--",
            ]
        if self.spec.cpu_set:
            argv += ["taskset", "-c", ",".join(str(cpu) for cpu in self.spec.cpu_set)]
        argv += [
            "enroot",
            "start",
            # Root inside the container, as Docker runs (the builder only accepts USER
            # root). Without it a
            # public-network exec kept the caller's uid and every apt/dpkg RUN failed
            # "requires superuser
            # privilege"; no-network execs were root only by accident of the outer `unshare -r`.
            "--root",
            "--rw",
        ]
        for key, value in merged.items():
            argv += ["--env", f"{key}={value}"]
        argv += [
            self.spec.name,
            self._shell(),
            "-c",
            f"cd {shlex.quote(workdir)} && {command}",
        ]
        result = self._runner.run(
            argv,
            # No login shell (enroot defaults to one): the image's /etc/profile then
            # reset PATH, dropping the
            # image ENV PATH (e.g. golang's /usr/local/go/bin -> "go: not found").
            # Docker never runs a login shell.
            # An empty value means "default" to enroot, so it must be "no".
            env={**self._enroot_env(), "ENROOT_LOGIN_SHELL": "no"},
            timeout_s=timeout_s,
        )
        return_code = 124 if result.timed_out else result.returncode
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=return_code,
            timed_out=result.timed_out,
        )

    def kill_running(self) -> None:
        """Kill every in-flight exec of this sandbox, as stopping a Docker container kills
        its processes.

        Callers that enforce their own deadline (Harbor's ``asyncio.wait_for`` around a
        verifier) cancel the coroutine but cannot stop the worker thread blocked in the exec;
        a verifier whose server never exits
        then held the thread, and the whole Harbor process, forever (B2.1 R0 run r3)."""
        kill_all = getattr(self._runner, "kill_all", None)
        if callable(kill_all):
            kill_all()

    def _container_pids(self) -> list[int]:
        """Host pids of this user's processes running inside this sandbox (daemons included).

        enroot pivots the root inside a private mount namespace, so /proc/<pid>/root reads
        "/" from the host; the reliable identity is that namespace's root mount: mount point
        "/" whose filesystem-relative root is a suffix of the rootfs host path (e.g.
        "/hhri-ai/hh28144/.enroot-sbx/<name>" on a /home export)."""
        rootfs = str(self.rootfs())
        uid = os.getuid()
        pids = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            with contextlib.suppress(OSError):
                if entry.stat().st_uid != uid:
                    continue
                with (entry / "mountinfo").open() as mountinfo:
                    for line in mountinfo:
                        fields = line.split()
                        if (
                            len(fields) > 4
                            and fields[4] == "/"
                            and fields[3] != "/"
                            and rootfs.endswith(fields[3])
                        ):
                            pids.append(int(entry.name))
                            break
        return pids

    def stop(self, *, delete: bool = True) -> None:
        self.kill_running()
        if self._started:
            # Docker's stop kills every container process; background daemons
            # (servers, mongod, browsers)
            # otherwise outlived their sandbox: 73 such processes had leaked on one
            # node across R0 runs.
            for pid in self._container_pids():
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
            terminate = getattr(self._runner, "terminate", None)
            if callable(terminate):
                for pid in [
                    *self._helpers,
                    *([self._netns_pid] if self._netns_pid is not None else []),
                ]:
                    terminate(pid)
            self._helpers.clear()
            self._netns_pid = None
            for role in ["egress", *(f"ep{i}" for i in range(len(self.spec.endpoints)))]:
                with contextlib.suppress(OSError):
                    self._socket_path(role).unlink()
        if delete and self._started:
            result = self._runner.run(
                ["enroot", "remove", "-f", self.spec.name],
                env=self._enroot_env(),
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

    def _enroot_env(self) -> dict[str, str]:
        env = {"ENROOT_DATA_PATH": self.spec.data_root}
        if self.spec.enroot_sysconf:
            env["ENROOT_SYSCONF_PATH"] = self.spec.enroot_sysconf
        return env

    def _shell(self) -> str:
        """bash when the image has it (Harbor's Docker backend runs every exec as `bash -c`;
        agent installers use
        `set -o pipefail`, which dash rejects), else sh for bash-less images such as Alpine."""
        root = Path(self.rootfs())
        return (
            "bash"
            if (root / "bin" / "bash").exists() or (root / "usr" / "bin" / "bash").exists()
            else "sh"
        )

    def _image_env_defines(self, name: str) -> bool:
        """True if the image's own environment (enroot writes it to the rootfs /etc/environment)
        sets ``name``."""
        try:
            lines = (
                (Path(self.rootfs()) / "etc" / "environment")
                .read_text(errors="replace")
                .splitlines()
            )
        except OSError:
            return False
        return any(line.split("=", 1)[0].strip() == name for line in lines if "=" in line)

    def rootfs(self) -> Path:
        return Path(self.spec.data_root) / self.spec.name

    def host_path(self, container_path: str) -> Path:
        if not isinstance(container_path, str) or not container_path:
            raise SandboxRefusal(
                f"container_path: expected non-empty absolute path, got {container_path!r}"
            )
        if not container_path.startswith("/"):
            raise SandboxRefusal(f"container_path: expected absolute path, got {container_path!r}")
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
