"""Tests for the Harbor enroot environment backend."""

from __future__ import annotations

import asyncio
import importlib
import re
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_MODULE_UNDER_TEST = "foundationscale.agentic_rl.sandbox.harbor_env"
_ENVIRONMENTS_MODULE = "harbor.environments"
_BASE_MODULE = "harbor.environments.base"
_CAPABILITIES_MODULE = "harbor.environments.capabilities"
_ENVIRONMENT_ID = "0123456789abcdef0123456789abcdef"


@dataclass
class ExecResult:
    """Stand-in for harbor.environments.base.ExecResult."""

    stdout: str | None = None
    stderr: str | None = None
    return_code: int = 0


@dataclass
class EnvironmentCapabilities:
    """Stand-in for harbor.environments.capabilities.EnvironmentCapabilities."""

    disable_internet: bool = False
    network_allowlist: bool = False


class BaseEnvironment:
    """Stand-in for harbor.environments.base.BaseEnvironment."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: Any = None,
        task_env_config: Any = None,
        logger: Any = None,
        override_cpus: int | None = None,
        override_memory_mb: int | None = None,
        override_storage_mb: int | None = None,
        override_gpus: int | None = None,
        override_tpu: Any = None,
        network_policy: Any = None,
        **kwargs: Any,
    ) -> None:
        self.environment_dir = Path(environment_dir)
        self.environment_name = environment_name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.task_env_config = task_env_config
        self.logger = logger
        self.override_cpus = override_cpus
        self.override_memory_mb = override_memory_mb
        self.override_storage_mb = override_storage_mb
        self.override_gpus = override_gpus
        self.override_tpu = override_tpu
        self.network_policy = network_policy
        self.extra_kwargs = kwargs

    @property
    def environment_id(self) -> str:
        """Fixed content identity used by the tests."""
        return _ENVIRONMENT_ID

    @property
    def _network_disabled(self) -> bool:
        """True when the policy forbids all network access."""
        mode = getattr(self.network_policy, "network_mode", None)
        return mode == "no-network"

    @property
    def _network_is_public(self) -> bool:
        """True when the policy allows unrestricted network access."""
        mode = getattr(self.network_policy, "network_mode", None)
        return mode == "public"

    @property
    def _network_is_allowlist(self) -> bool:
        """True when the policy restricts access to an allowlist."""
        mode = getattr(self.network_policy, "network_mode", None)
        return mode == "allowlist"


@dataclass
class RunResult:
    """Result of one external command invocation."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class FakeRunner:
    """Runner that emulates enroot import/create/export/remove side effects."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunResult:
        self.calls.append(list(argv))
        argv_list = list(argv)
        if argv_list[:2] == ["enroot", "import"]:
            output = Path(argv_list[argv_list.index("-o") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("image", encoding="utf-8")
            return RunResult(returncode=0, stdout="", stderr="")
        if argv_list[:2] == ["enroot", "create"]:
            name = argv_list[argv_list.index("-n") + 1]
            data_root = Path(env["ENROOT_DATA_PATH"]) if env else Path()
            (data_root / name).mkdir(parents=True, exist_ok=True)
            return RunResult(returncode=0, stdout="", stderr="")
        if argv_list[:2] == ["enroot", "export"]:
            output = Path(argv_list[argv_list.index("-o") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("built", encoding="utf-8")
            return RunResult(returncode=0, stdout="", stderr="")
        if argv_list[:2] == ["enroot", "remove"]:
            name = argv_list[-1]
            data_root = Path(env["ENROOT_DATA_PATH"]) if env else Path()
            rootfs = data_root / name
            if rootfs.is_dir():
                for entry in sorted(rootfs.rglob("*"), reverse=True):
                    if entry.is_file():
                        entry.unlink()
                    else:
                        entry.rmdir()
                rootfs.rmdir()
            return RunResult(returncode=0, stdout="", stderr="")
        return RunResult(returncode=0, stdout="ok", stderr="")


@pytest.fixture()
def harbor_env_module(tmp_path: Path):
    """Install Harbor stand-ins, import the module under test, then remove the stubs."""
    saved = {
        name: sys.modules.get(name)
        for name in ("harbor", _ENVIRONMENTS_MODULE, _BASE_MODULE, _CAPABILITIES_MODULE)
    }
    harbor_pkg = types.ModuleType("harbor")
    harbor_pkg.__path__ = []
    environments_pkg = types.ModuleType(_ENVIRONMENTS_MODULE)
    environments_pkg.__path__ = []
    base_mod = types.ModuleType(_BASE_MODULE)
    base_mod.BaseEnvironment = BaseEnvironment
    base_mod.ExecResult = ExecResult
    capabilities_mod = types.ModuleType(_CAPABILITIES_MODULE)
    capabilities_mod.EnvironmentCapabilities = EnvironmentCapabilities
    sys.modules["harbor"] = harbor_pkg
    sys.modules[_ENVIRONMENTS_MODULE] = environments_pkg
    sys.modules[_BASE_MODULE] = base_mod
    sys.modules[_CAPABILITIES_MODULE] = capabilities_mod
    sys.modules.pop(_MODULE_UNDER_TEST, None)
    module = importlib.import_module(_MODULE_UNDER_TEST)
    try:
        yield module
    finally:
        sys.modules.pop(_MODULE_UNDER_TEST, None)
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@dataclass
class _NetworkPolicy:
    network_mode: str = "no-network"


@dataclass
class _TaskEnvConfig:
    docker_image: str | None = None
    memory_mb: int | None = None


def _make_env(
    module: Any,
    tmp_path: Path,
    *,
    session_id: str = "session-1",
    network_mode: str = "no-network",
    docker_image: str | None = "python:3.12",
    memory_mb: int | None = None,
    runner: FakeRunner | None = None,
) -> Any:
    """Build an EnrootEnvironment with FS directories rooted under tmp_path."""
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    data_root = tmp_path / "data"
    image_store = tmp_path / "images"
    scratch_root = tmp_path / "scratch"
    for path in (data_root, image_store, scratch_root):
        path.mkdir(parents=True, exist_ok=True)
    return module.EnrootEnvironment(
        environment_dir=environment_dir,
        environment_name="task",
        session_id=session_id,
        trial_paths=None,
        task_env_config=_TaskEnvConfig(docker_image=docker_image, memory_mb=memory_mb),
        network_policy=_NetworkPolicy(network_mode=network_mode),
        data_root=str(data_root),
        image_store=str(image_store),
        scratch_root=str(scratch_root),
        runner=runner,
    )


def _is_enroot_start(call: Sequence[str]) -> bool:
    """An exec argv may be prefixed (unshare -rn, taskset, prlimit) before `enroot start`."""
    args = list(call)
    return any(args[i : i + 2] == ["enroot", "start"] for i in range(len(args) - 1))


def test_type(harbor_env_module: Any) -> None:
    """type() reports the fs-enroot backend name."""
    assert harbor_env_module.EnrootEnvironment.type() == "fs-enroot"


def test_capabilities_disable_internet_without_allowlist(
    harbor_env_module: Any, tmp_path: Path
) -> None:
    """capabilities disables internet and declares no allowlist support."""
    env = _make_env(harbor_env_module, tmp_path)
    capabilities = env.capabilities
    assert capabilities.disable_internet is True
    assert capabilities.network_allowlist is False


def test_validate_definition_with_docker_image(harbor_env_module: Any, tmp_path: Path) -> None:
    """A docker_image alone satisfies the definition check."""
    env = _make_env(harbor_env_module, tmp_path, docker_image="python:3.12")
    env._validate_definition()


def test_validate_definition_with_dockerfile(harbor_env_module: Any, tmp_path: Path) -> None:
    """A Dockerfile alone satisfies the definition check."""
    env = _make_env(harbor_env_module, tmp_path, docker_image=None)
    (env.environment_dir / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    env._validate_definition()


def test_validate_definition_without_image_or_dockerfile(
    harbor_env_module: Any, tmp_path: Path
) -> None:
    """Missing docker_image and Dockerfile raises ValueError naming both."""
    env = _make_env(harbor_env_module, tmp_path, docker_image=None)
    with pytest.raises(ValueError, match="docker_image"):
        env._validate_definition()
    with pytest.raises(ValueError, match="Dockerfile"):
        env._validate_definition()


def test_start_with_docker_image_imports_and_uses_no_network(
    harbor_env_module: Any, tmp_path: Path
) -> None:
    """docker_image start imports docker://<image> and execs under unshare -rn."""
    runner = FakeRunner()
    env = _make_env(harbor_env_module, tmp_path, network_mode="no-network", runner=runner)
    asyncio.run(env.start(force_build=False))
    assert any(
        call[:2] == ["enroot", "import"] and "docker://python:3.12" in call for call in runner.calls
    )
    asyncio.run(env.exec("echo hi"))
    exec_calls = [call for call in runner.calls if _is_enroot_start(call)]
    assert exec_calls
    assert "unshare" in exec_calls[0] and "-rn" in exec_calls[0]


def test_start_with_docker_image_public_network_skips_unshare(
    harbor_env_module: Any, tmp_path: Path
) -> None:
    """A public network policy omits the unshare wrapper from exec argv."""
    runner = FakeRunner()
    env = _make_env(harbor_env_module, tmp_path, network_mode="public", runner=runner)
    asyncio.run(env.start(force_build=False))
    asyncio.run(env.exec("echo hi"))
    exec_calls = [call for call in runner.calls if _is_enroot_start(call)]
    assert exec_calls
    assert "unshare" not in exec_calls[0]


def test_start_with_dockerfile_builds_and_reuses_image(
    harbor_env_module: Any, tmp_path: Path
) -> None:
    """A Dockerfile builds to image_store/<environment_id>.sqsh and is reused unless forced."""
    runner = FakeRunner()
    env = _make_env(harbor_env_module, tmp_path, docker_image=None, runner=runner)
    (env.environment_dir / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    asyncio.run(env.start(force_build=False))
    built = Path(Path(tmp_path / "images")) / f"{env.environment_id}.sqsh"
    assert built.is_file()
    export_calls = [call for call in runner.calls if call[:2] == ["enroot", "export"]]
    assert len(export_calls) == 1
    asyncio.run(env.stop(delete=True))
    asyncio.run(env.start(force_build=False))
    export_calls = [call for call in runner.calls if call[:2] == ["enroot", "export"]]
    assert len(export_calls) == 1
    asyncio.run(env.stop(delete=True))
    asyncio.run(env.start(force_build=True))
    export_calls = [call for call in runner.calls if call[:2] == ["enroot", "export"]]
    assert len(export_calls) == 2


def test_session_id_is_sanitized(harbor_env_module: Any, tmp_path: Path) -> None:
    """Session ids are sanitized to [A-Za-z0-9_.-] and truncated to 63 chars."""
    runner = FakeRunner()
    env = _make_env(
        harbor_env_module,
        tmp_path,
        session_id="bad session/id!" + "x" * 80,
        runner=runner,
    )
    asyncio.run(env.start(force_build=False))
    create_calls = [call for call in runner.calls if call[:2] == ["enroot", "create"]]
    assert create_calls
    name = create_calls[0][create_calls[0].index("-n") + 1]
    assert re.fullmatch(r"[A-Za-z0-9_.-]{1,63}", name)
    assert len(name) <= 63


def test_exec_returns_result_and_refuses_non_root_user(
    harbor_env_module: Any, tmp_path: Path
) -> None:
    """exec maps EnrootSandbox.exec output to ExecResult and rejects non-root users."""
    runner = FakeRunner()
    env = _make_env(harbor_env_module, tmp_path, runner=runner)
    asyncio.run(env.start(force_build=False))
    result = asyncio.run(env.exec("echo hi", user=None))
    assert result.stdout == "ok"
    assert result.stderr == ""
    assert result.return_code == 0
    for user in ("root", 0):
        assert asyncio.run(env.exec("echo hi", user=user)).return_code == 0
    for user in ("nobody", 1000):
        with pytest.raises(ValueError):
            asyncio.run(env.exec("echo hi", user=user))


def test_upload_download_round_trip(harbor_env_module: Any, tmp_path: Path) -> None:
    """upload_file/upload_dir and download_file/download_dir round-trip through the sandbox."""
    runner = FakeRunner()
    env = _make_env(harbor_env_module, tmp_path, runner=runner)
    asyncio.run(env.start(force_build=False))
    source_file = tmp_path / "source.txt"
    source_file.write_text("payload", encoding="utf-8")
    asyncio.run(env.upload_file(source_file, "/work/source.txt"))
    downloaded_file = tmp_path / "downloaded.txt"
    asyncio.run(env.download_file("/work/source.txt", downloaded_file))
    assert downloaded_file.read_text(encoding="utf-8") == "payload"
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("a", encoding="utf-8")
    (source_dir / "nested").mkdir()
    (source_dir / "nested" / "b.txt").write_text("b", encoding="utf-8")
    asyncio.run(env.upload_dir(source_dir, "/work/dir"))
    downloaded_dir = tmp_path / "downloaded_dir"
    asyncio.run(env.download_dir("/work/dir", downloaded_dir))
    assert (downloaded_dir / "a.txt").read_text(encoding="utf-8") == "a"
    assert (downloaded_dir / "nested" / "b.txt").read_text(encoding="utf-8") == "b"


def test_stop_is_safe_before_start(harbor_env_module: Any, tmp_path: Path) -> None:
    """stop() before start() is a no-op and does not raise."""
    runner = FakeRunner()
    env = _make_env(harbor_env_module, tmp_path, runner=runner)
    asyncio.run(env.stop(delete=True))
    asyncio.run(env.stop(delete=False))
    assert not any(call[:2] == ["enroot", "remove"] for call in runner.calls)


def test_start_creates_harbors_log_layout(harbor_env_module: Any, tmp_path: Path) -> None:
    """/logs/{agent,user-agent,verifier,artifacts} exist after start (docker mounts them)."""
    runner = FakeRunner()
    env = _make_env(harbor_env_module, tmp_path, network_mode="public", runner=runner)
    asyncio.run(env.start(force_build=False))
    rootfs = Path(tmp_path / "data")
    for sub in ("agent", "user-agent", "verifier", "artifacts"):
        assert any(p.is_dir() for p in rootfs.glob(f"*/logs/{sub}"))
