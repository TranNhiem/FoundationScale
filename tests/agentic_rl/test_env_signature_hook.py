"""ARL-18 slice 1 integration: EnvSpec/config ``signature`` key, the TaskSandboxEnvironment
start-time signature, and ``HiddenVault.sealed_manifest``."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from foundationscale.agentic_rl.config import AgenticConfigRefusal, build_config
from foundationscale.agentic_rl.envs.base import EnvRefusal, EnvSpec
from foundationscale.agentic_rl.envs.sandbox_task import TaskSandboxConfig, TaskSandboxEnvironment
from foundationscale.agentic_rl.harness.base import EpisodeTask
from foundationscale.agentic_rl.integrity import IntegrityRefusal
from foundationscale.agentic_rl.sandbox.hidden import HiddenSpec, HiddenVault


def _sibling(name: str) -> ModuleType:
    """Load a sibling test module by path (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location(f"_sig_{name}", Path(__file__).with_name(name))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


_sandbox_tests = _sibling("test_envs_sandbox_task.py")
_raw_config = _sibling("test_config.py")._raw_config
FakeResult = _sandbox_tests.FakeResult
FakeSandbox = _sandbox_tests.FakeSandbox
FakeVault = _sandbox_tests.FakeVault

_META = {"image": "img:1", "test_cmd": "pytest -q", "workdir": "/repo"}


class RootfsSandbox(FakeSandbox):  # type: ignore[misc,valid-type]
    """FakeSandbox whose rootfs is a temp directory, never the real host ``/``."""

    def __init__(self, root: Path, results: list[Any]) -> None:
        super().__init__(results)
        self.root = root

    def host_path(self, container_path: str) -> Path:
        return self.root / container_path.lstrip("/")


class UnreadableSandbox(FakeSandbox):  # type: ignore[misc,valid-type]
    def host_path(self, container_path: str) -> Path:
        raise PermissionError(container_path)


def _rootfs(tmp_path: Path, name: str = "rootfs") -> Path:
    root = tmp_path / name
    (root / "repo").mkdir(parents=True)
    (root / "repo" / "app.py").write_text("print('hi')\n")
    (root / "proc").mkdir()
    return root


def _spec(signature: str = "manifest") -> EnvSpec:
    return EnvSpec(
        backend="enroot_task",
        env={},
        exec_timeout_s=5.0,
        episode_timeout_s=60.0,
        max_output_bytes=64,
        signature=signature,
    )


def _config() -> TaskSandboxConfig:
    task = EpisodeTask(
        uid="t", session_id="t#0", messages=({"role": "user", "content": "x"},), metadata=_META
    )
    return TaskSandboxConfig.from_task(task)


def _signature(sandbox: Any, signature: str = "manifest") -> dict[str, str]:
    env = TaskSandboxEnvironment(
        spec=_spec(signature), config=_config(), sandbox=sandbox, vault=FakeVault()
    )
    asyncio.run(env.start())
    return asyncio.run(env.verify())


def test_envspec_signature_defaults_off_and_refuses_unknown_mode() -> None:
    assert EnvSpec(backend="b", env={}).signature == "off"
    with pytest.raises(EnvRefusal, match="signature"):
        EnvSpec(backend="b", env={}, signature="full")


def test_two_clean_starts_give_the_same_signature(tmp_path: Path) -> None:
    """Reset fidelity: two fresh sandboxes from identical rootfs trees sign identically."""
    first = _signature(RootfsSandbox(_rootfs(tmp_path, "a"), [FakeResult()]))
    second = _signature(RootfsSandbox(_rootfs(tmp_path, "b"), [FakeResult()]))
    assert first["env_signature"].startswith("envsig_")
    assert first["env_signature"] == second["env_signature"]


def test_content_mode_sees_an_edit_that_keeps_size(tmp_path: Path) -> None:
    root_a = _rootfs(tmp_path, "a")
    root_b = _rootfs(tmp_path, "b")
    (root_b / "repo" / "app.py").write_text("print('ho')\n")
    a = _signature(RootfsSandbox(root_a, [FakeResult()]), "content")
    b = _signature(RootfsSandbox(root_b, [FakeResult()]), "content")
    assert a["env_signature"] != b["env_signature"]


def test_excluded_runtime_dirs_do_not_change_the_signature(tmp_path: Path) -> None:
    root_a = _rootfs(tmp_path, "a")
    root_b = _rootfs(tmp_path, "b")
    (root_b / "proc" / "1").write_text("pid")
    a = _signature(RootfsSandbox(root_a, [FakeResult()]))
    b = _signature(RootfsSandbox(root_b, [FakeResult()]))
    assert a["env_signature"] == b["env_signature"]


def test_mode_is_part_of_the_signature(tmp_path: Path) -> None:
    a = _signature(RootfsSandbox(_rootfs(tmp_path, "a"), [FakeResult()]), "manifest")
    b = _signature(RootfsSandbox(_rootfs(tmp_path, "b"), [FakeResult()]), "content")
    assert a["env_signature"] != b["env_signature"]


def test_unreadable_rootfs_is_unmeasured_not_partial() -> None:
    outcome = _signature(UnreadableSandbox([FakeResult()]))
    assert outcome["env_signature"] == "unmeasured"


def test_signature_off_reports_nothing(tmp_path: Path) -> None:
    sandbox = RootfsSandbox(_rootfs(tmp_path), [FakeResult()])
    assert "env_signature" not in _signature(sandbox, "off")


def _vault(tmp_path: Path, payload: str) -> HiddenVault:
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "t.py").write_text(payload)
    sandbox = RootfsSandbox(root, [])
    return HiddenVault(sandbox, HiddenSpec(paths=("/tests",)), tmp_path / "vault")


def test_sealed_manifest_is_refused_unless_sealed(tmp_path: Path) -> None:
    vault = _vault(tmp_path, "x")
    with pytest.raises(IntegrityRefusal, match="not sealed"):
        vault.sealed_manifest(content=True)


def test_sealed_manifest_tracks_hidden_content(tmp_path: Path) -> None:
    one = _vault(tmp_path / "1", "assert 1")
    two = _vault(tmp_path / "2", "assert 2")
    one.seal()
    two.seal()
    assert one.sealed_manifest(content=False) == two.sealed_manifest(content=False)
    assert one.sealed_manifest(content=True) != two.sealed_manifest(content=True)


def test_sealed_manifest_hashes_a_hidden_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "answer.txt").write_text("42")
    vault = HiddenVault(
        RootfsSandbox(root, []), HiddenSpec(paths=("/answer.txt",)), tmp_path / "vault"
    )
    vault.seal()
    assert vault.sealed_manifest(content=True) != vault.sealed_manifest(content=False)


def test_config_env_signature_key() -> None:
    def env(**extra: object) -> dict[str, Any]:
        raw = _raw_config()
        raw["env"] = {**raw["env"], **extra}
        return raw

    assert build_config(env(signature="content"), {}).env.signature == "content"
    assert build_config(env(), {}).env.signature == "off"
    assert build_config(env(), {"env.signature": "manifest"}).env.signature == "manifest"
    with pytest.raises(AgenticConfigRefusal, match="env.signature"):
        build_config(env(signature="bytes"), {})
