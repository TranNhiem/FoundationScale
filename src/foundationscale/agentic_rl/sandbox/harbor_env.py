"""Harbor environment backend backed by FoundationScale enroot sandboxes.

Harbor loads this module by import path
``foundationscale.agentic_rl.sandbox.harbor_env:EnrootEnvironment``.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from foundationscale.agentic_rl.sandbox.build import build_image
from foundationscale.agentic_rl.sandbox.enroot import EnrootSandbox, Runner, SandboxSpec, SubprocessRunner

_NAME_ALLOWED_RE = re.compile(r"[^A-Za-z0-9_.-]")
_MAX_NAME_LEN = 63


def _sanitize_name(raw: str) -> str:
    """Sanitize *raw* for use as an enroot sandbox name."""
    cleaned = _NAME_ALLOWED_RE.sub("-", raw)
    if not cleaned:
        cleaned = "sandbox"
    return cleaned[:_MAX_NAME_LEN]


# harbor.models.trial.paths.EnvironmentPaths: /logs/{agent,user-agent,verifier,artifacts}
_HARBOR_LOG_DIRS = ("/logs/agent", "/logs/user-agent", "/logs/verifier", "/logs/artifacts")


class EnrootEnvironment(BaseEnvironment):
    """Run Harbor trials inside FoundationScale enroot sandboxes."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        image_store: str | Path,
        scratch_root: str | Path,
        cpu_set: tuple[int, ...] | None = None,
        build_timeout_s: float = 3600.0,
        build_from_dockerfile: bool | str = False,
        enroot_sysconf: str | None = None,
        endpoints: str | Sequence[str] | None = None,
        runner: Runner | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._data_root = self._require_absolute_dir(data_root, "data_root")
        self._image_store = self._require_absolute_dir(image_store, "image_store")
        self._scratch_root = self._require_absolute_dir(scratch_root, "scratch_root")
        self._cpu_set = cpu_set
        self._build_timeout_s = build_timeout_s
        self._enroot_sysconf = enroot_sysconf or None
        # "PORT=HOST:HOSTPORT[,...]": host services bridged into every task sandbox (e.g. the agent's local model
        # endpoint at 127.0.0.1:18811); --ek passes one comma-separated string
        if isinstance(endpoints, str):
            endpoints = [e.strip() for e in endpoints.split(",") if e.strip()]
        self._endpoints = tuple(endpoints or ())
        # Build from environment/Dockerfile even when the task names a prebuilt docker_image: Terminal-Bench 2.1's
        # prebuilt images are single-arch amd64, so GB200 (arm64) must rebuild them. --ek passes strings.
        self._build_from_dockerfile = (
            build_from_dockerfile if isinstance(build_from_dockerfile, bool)
            else str(build_from_dockerfile).strip().lower() in ("1", "true", "yes")
        )
        # One runner for this environment's build and sandbox, so a cancelled start() can kill the build too
        self._runner: Runner = runner if runner is not None else SubprocessRunner()
        self._sandbox: EnrootSandbox | None = None

    @staticmethod
    def _require_absolute_dir(value: str | Path, name: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"{name}: expected absolute path, got {value!r}")
        return path

    # -- identity / capabilities ------------------------------------------

    @staticmethod
    def type() -> str:
        return "fs-enroot"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Truthful flags only: no-network (no route out of the sandbox netns), allowlists of hostnames and
        # *.wildcards (enforced by the per-sandbox egress proxy; proxy-unaware clients get no egress, never more),
        # and runtime phase switches (atomic policy-file replace). IP/CIDR entries are not enforced, so Harbor
        # rejects such tasks instead of leaking access.
        return EnvironmentCapabilities(
            disable_internet=True,
            network_allowlist=True,
            network_allowlist_hostnames=True,
            network_allowlist_wildcard_hostnames=True,
            dynamic_network_policy=True,
        )

    # -- definition validation --------------------------------------------

    def _validate_definition(self) -> None:
        docker_image = self._docker_image()
        dockerfile = self.environment_dir / "Dockerfile"
        if docker_image or dockerfile.is_file():
            return
        raise ValueError(
            "environment definition: expected task_env_config.docker_image to be set or "
            f"{dockerfile} to exist, got neither"
        )

    def _docker_image(self) -> str | None:
        value = getattr(self.task_env_config, "docker_image", None)
        return value if value else None

    # -- lifecycle ---------------------------------------------------------

    async def start(self, force_build: bool) -> None:
        try:
            await asyncio.to_thread(self._start_sync, force_build)
        except asyncio.CancelledError:
            # Harbor times environment start out by cancelling (EnvironmentStartTimeoutError); an image build
            # kept running in the thread, Harbor hung until the gate's own timeout killed it, and the killed
            # build left its container behind (B2.1 R0 run r7). Kill it so the build's cleanup runs.
            kill_all = getattr(self._runner, "kill_all", None)
            if callable(kill_all):
                kill_all()
            raise

    def _start_sync(self, force_build: bool) -> None:
        self._validate_definition()
        docker_image = self._docker_image()
        if self._build_from_dockerfile and (self.environment_dir / "Dockerfile").is_file():
            docker_image = None
        image = "docker://" + docker_image if docker_image else self._build_image_sync(force_build)
        name = _sanitize_name(self.session_id)
        network, hosts = _sandbox_policy(self.network_policy)
        memory_mb = getattr(self.task_env_config, "memory_mb", None)
        spec = SandboxSpec(
            name=name,
            image=image,
            data_root=str(self._data_root),
            image_store=str(self._image_store),
            scratch_root=str(self._scratch_root),
            network=network,
            cpu_set=self._cpu_set,
            memory_mb=memory_mb,
            enroot_sysconf=self._enroot_sysconf,
            isolate_network=True,
            allowed_hosts=hosts,
            endpoints=self._endpoints,
        )
        sandbox = EnrootSandbox(spec, runner=self._runner)
        sandbox.start()
        # Harbor's docker backend bind-mounts the trial's log dirs, so they always exist
        # there; an enroot rootfs has to create Harbor's convention layout itself.
        for container_dir in _HARBOR_LOG_DIRS:
            sandbox.host_path(container_dir).mkdir(parents=True, exist_ok=True)
        if sandbox.entrypoint:
            # Harbor's Docker backend starts the container as ENTRYPOINT + `sh -c "sleep infinity"`; images
            # whose entrypoint starts services (mongodb-schema-design's bootstrap.sh starts mongod) need it.
            # Detached in the sandbox's shared network namespace, so later execs reach its services.
            argv = shlex.join([*sandbox.entrypoint, "sh", "-c", "sleep infinity"])
            sandbox.exec(f"setsid nohup {argv} > /tmp/fs-entrypoint.log 2>&1 < /dev/null &")
        self._sandbox = sandbox

    def _build_image_sync(self, force_build: bool) -> str:
        out_sqsh = self._image_store / f"{self.environment_id}.sqsh"
        if out_sqsh.is_file() and not force_build:
            return str(out_sqsh)
        build_image(
            self.environment_dir / "Dockerfile",
            self.environment_dir,
            out_sqsh,
            data_root=str(self._data_root),
            image_store=str(self._image_store),
            scratch_root=str(self._scratch_root),
            runner=self._runner,
            timeout_s=self._build_timeout_s,
            enroot_sysconf=self._enroot_sysconf,
        )
        return str(out_sqsh)

    async def _apply_network_policy(self, network_policy: Any) -> None:
        """Harbor phase switch (agent / verifier policies): replace the running sandbox's egress policy."""
        mode, hosts = _sandbox_policy(network_policy)
        self._require_sandbox().set_policy(mode, hosts)

    async def stop(self, delete: bool) -> None:
        await asyncio.to_thread(self._stop_sync, delete)

    def _stop_sync(self, delete: bool) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        sandbox.stop(delete=delete)
        self._sandbox = None

    # -- transfers ---------------------------------------------------------

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await asyncio.to_thread(self._require_sandbox().upload_file, source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await asyncio.to_thread(self._require_sandbox().upload_dir, source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await asyncio.to_thread(self._require_sandbox().download_file, source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await asyncio.to_thread(self._require_sandbox().download_dir, source_dir, target_dir)

    # -- exec --------------------------------------------------------------

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self._validate_user(user)
        sandbox = self._require_sandbox()
        try:
            result = await asyncio.to_thread(
                sandbox.exec,
                command,
                cwd=cwd,
                env=env,
                timeout_s=float(timeout_sec) if timeout_sec is not None else None,
            )
        except asyncio.CancelledError:
            # Harbor times verifiers out by cancelling this coroutine; the thread keeps blocking on the
            # process unless it is killed, and Harbor then hangs at shutdown waiting for that thread.
            sandbox.kill_running()
            raise
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
        )

    @staticmethod
    def _validate_user(user: str | int | None) -> None:
        if user is None or user == "root" or user == 0:
            return
        raise ValueError(
            f"user: expected None, 'root', or 0 (enroot runs as root in its namespace), "
            f"got {user!r}"
        )

    def _require_sandbox(self) -> EnrootSandbox:
        if self._sandbox is None:
            raise RuntimeError("sandbox: expected started environment, got not-started")
        return self._sandbox


def _sandbox_policy(policy: Any) -> tuple[str, tuple[str, ...]]:
    """Harbor NetworkPolicy -> (sandbox network mode, allowed hosts)."""
    if policy is None:
        return "public", ()
    mode = getattr(policy, "network_mode", "public")
    mode = getattr(mode, "value", mode)
    hosts = tuple(getattr(policy, "allowed_hosts", ()) or ())
    if mode not in ("no-network", "public", "allowlist"):
        raise ValueError(f"network_mode: expected no-network, public or allowlist, got {mode!r}")
    return mode, hosts if mode == "allowlist" else ()
