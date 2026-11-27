"""ARL-18 slice 3: environment snapshots, retained vaults, and enroot export/restore."""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import foundationscale.agentic_rl.envs.sandbox_task as sandbox_task
from foundationscale.agentic_rl.envs.base import EnvInfraError, EnvRefusal, EnvSpec
from foundationscale.agentic_rl.envs.sandbox_task import TaskSandboxConfig, TaskSandboxEnvironment
from foundationscale.agentic_rl.harness.base import EpisodeTask
from foundationscale.agentic_rl.integrity import IntegrityRefusal
from foundationscale.agentic_rl.sandbox.enroot import (
    EnrootSandbox,
    SandboxError,
    SandboxRefusal,
)
from foundationscale.agentic_rl.sandbox.hidden import HiddenSpec, HiddenVault
from foundationscale.agentic_rl.sandbox.snapshot_store import (
    SnapshotLimits,
    SnapshotRecord,
    SnapshotRefusal,
    SnapshotStore,
)


def _sibling(name: str) -> ModuleType:
    """Load a sibling test module by path (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        f"_snapshot_{name}", Path(__file__).with_name(name)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_sandbox_tests = _sibling("test_envs_sandbox_task.py")
FakeResult = _sandbox_tests.FakeResult
FakeSandbox = _sandbox_tests.FakeSandbox
FakeVault = _sandbox_tests.FakeVault

_META = {"image": "img:1", "test_cmd": "pytest -q", "workdir": "/repo"}
_HIDDEN_PATH = "/tests"
_HIDDEN_NAME = "test_hidden.py"
_HIDDEN_MARKER = "HIDDEN-MARKER-ARL18-7f3c9a: assert the_answer == 42\n"
_ARCHIVE = b"fake-squashfs-rootfs-archive-v1"


class SnapshotSandbox(FakeSandbox):  # type: ignore[misc,valid-type]
    """A FakeSandbox with a tmp_path rootfs and snapshot backend operations."""

    def __init__(
        self,
        root: Path,
        results: list[Any] | None = None,
        *,
        template: Path | None = None,
        archive: bytes = _ARCHIVE,
        export_exc: SandboxError | None = None,
    ) -> None:
        super().__init__([] if results is None else results)
        self.root = root
        self.template = template
        self.archive = archive
        self.export_exc = export_exc
        self.export_calls: list[tuple[Path, int, float]] = []
        self.restored_archives: list[str] = []

    def host_path(self, container_path: str) -> Path:
        self.host_paths.append(container_path)
        return self.root / container_path.lstrip("/")

    def export_rootfs(
        self,
        destination: str | Path,
        *,
        max_bytes: int,
        timeout_s: float,
    ) -> None:
        target = Path(destination)
        self.export_calls.append((target, max_bytes, timeout_s))
        if self.export_exc is not None:
            raise self.export_exc
        target.write_bytes(self.archive)

    def start_restored(self, archive: str) -> None:
        self.events.append("sandbox.start_restored")
        self.restored_archives.append(archive)
        self.started += 1
        assert self.template is not None
        if self.root.exists():
            shutil.rmtree(self.root)
        shutil.copytree(self.template, self.root)


def _spec(
    signature: str = "content",
    *,
    setup_commands: tuple[str, ...] = (),
) -> EnvSpec:
    return EnvSpec(
        backend="enroot_task",
        env={},
        exec_timeout_s=5.0,
        episode_timeout_s=60.0,
        max_output_bytes=4096,
        setup_commands=setup_commands,
        signature=signature,
    )


def _config() -> TaskSandboxConfig:
    task = EpisodeTask(
        uid="t",
        session_id="t#0",
        messages=({"role": "user", "content": "fix"},),
        metadata=_META,
    )
    return TaskSandboxConfig.from_task(task)


def _write_visible_tree(root: Path) -> None:
    (root / "repo").mkdir(parents=True)
    (root / "repo" / "app.py").write_text("print('hello')\n")
    (root / "proc").mkdir()


def _write_source_tree(root: Path) -> None:
    _write_visible_tree(root)
    (root / "tests").mkdir()
    (root / "tests" / _HIDDEN_NAME).write_text(_HIDDEN_MARKER)


def _new_store(tmp_path: Path, name: str = "store") -> SnapshotStore:
    limits = SnapshotLimits(
        max_bytes=1_000_000,
        total_bytes=4_000_000,
        timeout_s=5.0,
        ttl_s=3600.0,
    )
    return SnapshotStore(tmp_path / name, limits)


def _started_case(
    tmp_path: Path,
    name: str,
    *,
    signature: str = "content",
    setup_commands: tuple[str, ...] = (),
    export_exc: SandboxError | None = None,
) -> tuple[TaskSandboxEnvironment, SnapshotSandbox, HiddenVault]:
    root = tmp_path / f"{name}-rootfs"
    _write_source_tree(root)
    results = [FakeResult() for _ in range(len(setup_commands) + 1)]
    sandbox = SnapshotSandbox(root, results, export_exc=export_exc)
    vault = HiddenVault(
        sandbox,
        HiddenSpec(paths=(_HIDDEN_PATH,)),
        tmp_path / f"{name}-vault",
    )
    env = TaskSandboxEnvironment(
        spec=_spec(signature, setup_commands=setup_commands),
        config=_config(),
        sandbox=sandbox,
        vault=vault,
    )
    asyncio.run(env.start())
    return env, sandbox, vault


def _fresh_case(
    tmp_path: Path,
    name: str,
    *,
    signature: str = "content",
    setup_commands: tuple[str, ...] = (),
) -> tuple[TaskSandboxEnvironment, SnapshotSandbox, HiddenVault, Path]:
    template = tmp_path / f"{name}-template"
    _write_visible_tree(template)
    root = tmp_path / f"{name}-rootfs"
    root.mkdir()
    sandbox = SnapshotSandbox(root, [FakeResult()], template=template)
    vault_dir = tmp_path / f"{name}-vault"
    vault = HiddenVault(
        sandbox,
        HiddenSpec(paths=(_HIDDEN_PATH,)),
        vault_dir,
    )
    env = TaskSandboxEnvironment(
        spec=_spec(signature, setup_commands=setup_commands),
        config=_config(),
        sandbox=sandbox,
        vault=vault,
    )
    return env, sandbox, vault, vault_dir


@dataclass
class _Committed:
    store: SnapshotStore
    record: SnapshotRecord
    env: TaskSandboxEnvironment
    sandbox: SnapshotSandbox
    vault: HiddenVault


def _commit(
    tmp_path: Path,
    *,
    setup_commands: tuple[str, ...] = (),
) -> _Committed:
    env, sandbox, vault = _started_case(
        tmp_path,
        "source",
        setup_commands=setup_commands,
    )
    store = _new_store(tmp_path)
    record = asyncio.run(env.snapshot(store, vault_root=tmp_path / "retained-vaults"))
    return _Committed(store, record, env, sandbox, vault)


def _archive_path(store: SnapshotStore, record: SnapshotRecord) -> Path:
    return store.root / f"{record.snapshot_id}.sqsh"


def _descriptor_path(store: SnapshotStore, record: SnapshotRecord) -> Path:
    return store.root / f"{record.snapshot_id}.json"


def _assert_unleased(store: SnapshotStore, snapshot_id: str) -> None:
    assert store.get(snapshot_id).leased is False


def test_snapshot_refused_outside_agent_phase_and_without_signature(
    tmp_path: Path,
) -> None:
    fresh, _, _, _ = _fresh_case(tmp_path, "fresh")
    store = _new_store(tmp_path)
    with pytest.raises(EnvRefusal, match="started environment"):
        asyncio.run(fresh.snapshot(store, vault_root=tmp_path / "retained-a"))

    verified, _, _ = _started_case(tmp_path, "verified")
    asyncio.run(verified.verify())
    with pytest.raises(EnvRefusal, match="before verify"):
        asyncio.run(verified.snapshot(store, vault_root=tmp_path / "retained-b"))

    closed, _, _ = _started_case(tmp_path, "closed")
    asyncio.run(closed.close())
    with pytest.raises(EnvRefusal, match="before verify"):
        asyncio.run(closed.snapshot(store, vault_root=tmp_path / "retained-c"))

    unsigned, _, _ = _started_case(tmp_path, "unsigned", signature="off")
    with pytest.raises(EnvRefusal, match="signature is 'off'"):
        asyncio.run(unsigned.snapshot(store, vault_root=tmp_path / "retained-d"))


def test_snapshot_restore_round_trip_adopts_independent_vault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = ("echo setup-command",)
    committed = _commit(tmp_path, setup_commands=setup)
    record = committed.record
    clone_dir = Path(record.meta["vault_clone"])
    retained_file = clone_dir / "tests" / _HIDDEN_NAME
    assert retained_file.read_text() == _HIDDEN_MARKER

    restored, sandbox, vault, _ = _fresh_case(
        tmp_path,
        "restored",
        setup_commands=setup,
    )

    def unexpected_seal() -> tuple[str, ...]:
        raise AssertionError("restore must adopt the retained vault, not seal again")

    monkeypatch.setattr(vault, "seal", unexpected_seal)
    asyncio.run(restored.restore(committed.store, record.snapshot_id))

    assert restored._signature == record.env_signature
    assert sandbox.execs == []
    assert sandbox.restored_archives == [str(_archive_path(committed.store, record))]
    vault.assert_sealed()
    _assert_unleased(committed.store, record.snapshot_id)

    asyncio.run(restored.verify())
    assert (sandbox.root / "tests" / _HIDDEN_NAME).read_text() == _HIDDEN_MARKER
    assert retained_file.read_text() == _HIDDEN_MARKER

    second, second_sandbox, second_vault, second_vault_dir = _fresh_case(
        tmp_path,
        "second",
        setup_commands=setup,
    )
    asyncio.run(second.restore(committed.store, record.snapshot_id))
    assert second_sandbox.execs == []
    second_vault.assert_sealed()
    adopted = second_vault_dir / "tests" / _HIDDEN_NAME
    assert adopted.read_text() == _HIDDEN_MARKER
    assert retained_file.read_text() == _HIDDEN_MARKER
    _assert_unleased(committed.store, record.snapshot_id)


def test_archive_and_store_descriptor_never_contain_hidden_bytes(
    tmp_path: Path,
) -> None:
    committed = _commit(tmp_path)
    marker = _HIDDEN_MARKER.encode()
    assert marker not in _archive_path(committed.store, committed.record).read_bytes()
    descriptor = _descriptor_path(committed.store, committed.record).read_bytes()
    assert marker not in descriptor
    for path in committed.store.root.iterdir():
        if path.is_file():
            assert marker not in path.read_bytes(), path


def test_snapshot_refuses_setgid_rootfs_entries(tmp_path: Path) -> None:
    env, sandbox, _ = _started_case(tmp_path, "setgid")
    (sandbox.root / "repo").chmod(0o2755)
    store = _new_store(tmp_path)
    with pytest.raises(EnvInfraError, match="setuid/setgid") as caught:
        asyncio.run(env.snapshot(store, vault_root=tmp_path / "retained"))
    assert caught.value.kind == "snapshot_unsupported"


def test_snapshot_export_failure_removes_clone_and_commits_nothing(
    tmp_path: Path,
) -> None:
    env, _, _ = _started_case(
        tmp_path,
        "export-failure",
        export_exc=SandboxError("export exploded"),
    )
    store = _new_store(tmp_path)
    vault_root = tmp_path / "retained"
    with pytest.raises(EnvInfraError, match="export exploded") as caught:
        asyncio.run(env.snapshot(store, vault_root=vault_root))
    assert caught.value.kind == "snapshot_failed"
    assert list(vault_root.iterdir()) == []
    assert list(store.root.glob("snap_*.json")) == []
    assert list(store.root.glob("snap_*.sqsh")) == []


def test_restore_rejects_tampered_archive_stops_sandbox_and_closes_env(
    tmp_path: Path,
) -> None:
    committed = _commit(tmp_path)
    archive = _archive_path(committed.store, committed.record)
    original = archive.read_bytes()
    tampered = bytes([original[0] ^ 0x01]) + original[1:]
    assert len(tampered) == len(original)
    assert tampered != original
    archive.write_bytes(tampered)

    restored, sandbox, _, _ = _fresh_case(tmp_path, "tampered")
    with pytest.raises(EnvInfraError) as caught:
        asyncio.run(restored.restore(committed.store, committed.record.snapshot_id))
    assert caught.value.kind == "snapshot_integrity"
    assert sandbox.stopped == [True]
    with pytest.raises(EnvRefusal):
        asyncio.run(restored.execute("echo must-not-run"))
    _assert_unleased(committed.store, committed.record.snapshot_id)


def test_restore_rejects_rootfs_that_differs_from_snapshot(tmp_path: Path) -> None:
    committed = _commit(tmp_path)
    restored, sandbox, _, _ = _fresh_case(tmp_path, "changed-template")
    assert sandbox.template is not None
    (sandbox.template / "repo" / "app.py").write_text("print('changed')\n")

    with pytest.raises(EnvInfraError, match="restored signature") as caught:
        asyncio.run(restored.restore(committed.store, committed.record.snapshot_id))
    assert caught.value.kind == "snapshot_integrity"
    assert sandbox.stopped == [True]
    _assert_unleased(committed.store, committed.record.snapshot_id)


def test_restore_rejects_mismatched_signature_mode(tmp_path: Path) -> None:
    committed = _commit(tmp_path)
    restored, _, _, _ = _fresh_case(tmp_path, "manifest", signature="manifest")
    with pytest.raises(EnvRefusal, match="signature mode"):
        asyncio.run(restored.restore(committed.store, committed.record.snapshot_id))
    _assert_unleased(committed.store, committed.record.snapshot_id)


def test_restore_runs_exactly_once_and_replaces_start(tmp_path: Path) -> None:
    committed = _commit(tmp_path)
    restored, _, _, _ = _fresh_case(tmp_path, "twice")
    snapshot_id = committed.record.snapshot_id
    asyncio.run(restored.restore(committed.store, snapshot_id))
    with pytest.raises(EnvRefusal, match="fresh environment"):
        asyncio.run(restored.restore(committed.store, snapshot_id))
    _assert_unleased(committed.store, snapshot_id)

    started, _, _ = _started_case(tmp_path, "already-started")
    with pytest.raises(EnvRefusal, match="fresh environment"):
        asyncio.run(started.restore(committed.store, snapshot_id))


def test_restore_unknown_snapshot_is_integrity_failure(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    restored, _, _, _ = _fresh_case(tmp_path, "unknown")
    unknown = "snap_" + "0" * 32
    with pytest.raises(EnvInfraError) as caught:
        asyncio.run(restored.restore(store, unknown))
    assert caught.value.kind == "snapshot_integrity"
    with pytest.raises(EnvRefusal):
        asyncio.run(restored.execute("echo closed"))


def test_snapshot_and_restore_refuse_backends_without_support(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    plain = FakeSandbox()
    snapshot_env = TaskSandboxEnvironment(
        spec=_spec(),
        config=_config(),
        sandbox=plain,
        vault=FakeVault(),
    )
    snapshot_env._started = True
    with pytest.raises(EnvInfraError) as snapshot_caught:
        asyncio.run(snapshot_env.snapshot(store, vault_root=tmp_path / "retained"))
    assert snapshot_caught.value.kind == "snapshot_unsupported"

    restore_env = TaskSandboxEnvironment(
        spec=_spec(),
        config=_config(),
        sandbox=FakeSandbox(),
        vault=FakeVault(),
    )
    with pytest.raises(EnvInfraError) as restore_caught:
        asyncio.run(restore_env.restore(store, "snap_" + "1" * 32))
    assert restore_caught.value.kind == "snapshot_unsupported"


def _sealed_vault(tmp_path: Path, name: str) -> tuple[HiddenVault, Path, Path]:
    root = tmp_path / f"{name}-rootfs"
    _write_source_tree(root)
    sandbox = SnapshotSandbox(root)
    vault_dir = tmp_path / f"{name}-vault"
    vault = HiddenVault(sandbox, HiddenSpec(paths=(_HIDDEN_PATH,)), vault_dir)
    vault.seal()
    return vault, root, vault_dir


def _unsealed_vault(tmp_path: Path, name: str) -> tuple[HiddenVault, Path]:
    root = tmp_path / f"{name}-rootfs"
    _write_visible_tree(root)
    sandbox = SnapshotSandbox(root)
    vault_dir = tmp_path / f"{name}-vault"
    return HiddenVault(sandbox, HiddenSpec(paths=(_HIDDEN_PATH,)), vault_dir), vault_dir


def test_clone_sealed_refused_when_unsealed(tmp_path: Path) -> None:
    vault, _ = _unsealed_vault(tmp_path, "unsealed")
    with pytest.raises(IntegrityRefusal, match="not sealed"):
        vault.clone_sealed(tmp_path / "clone")


def test_clone_sealed_refused_when_destination_exists(tmp_path: Path) -> None:
    vault, _, _ = _sealed_vault(tmp_path, "existing")
    destination = tmp_path / "clone"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        vault.clone_sealed(destination)


def test_clone_sealed_refused_inside_rootfs(tmp_path: Path) -> None:
    vault, root, _ = _sealed_vault(tmp_path, "inside")
    with pytest.raises(IntegrityRefusal, match="inside the sandbox rootfs"):
        vault.clone_sealed(root / "retained-clone")


def test_adopt_sealed_refuses_undeclared_path(tmp_path: Path) -> None:
    source, _, _ = _sealed_vault(tmp_path, "declared-source")
    clone = tmp_path / "declared-clone"
    source.clone_sealed(clone)
    target, _ = _unsealed_vault(tmp_path, "declared-target")
    with pytest.raises(IntegrityRefusal, match="not a declared hidden path"):
        target.adopt_sealed(clone, ("/other",))


def test_adopt_sealed_refuses_missing_retained_copy(tmp_path: Path) -> None:
    target, _ = _unsealed_vault(tmp_path, "missing")
    empty_clone = tmp_path / "empty-clone"
    empty_clone.mkdir()
    with pytest.raises(IntegrityRefusal, match="retained copy.*missing"):
        target.adopt_sealed(empty_clone, (_HIDDEN_PATH,))


def test_adopt_sealed_refuses_existing_target(tmp_path: Path) -> None:
    source, _, _ = _sealed_vault(tmp_path, "target-source")
    clone = tmp_path / "target-clone"
    source.clone_sealed(clone)
    target, vault_dir = _unsealed_vault(tmp_path, "target")
    (vault_dir / "tests").mkdir(parents=True)
    with pytest.raises(IntegrityRefusal, match="target already exists"):
        target.adopt_sealed(clone, (_HIDDEN_PATH,))


def test_adopt_sealed_refuses_already_sealed_vault(tmp_path: Path) -> None:
    source, _, _ = _sealed_vault(tmp_path, "sealed-source")
    clone = tmp_path / "sealed-clone"
    source.clone_sealed(clone)
    with pytest.raises(IntegrityRefusal, match="expected an unsealed vault"):
        source.adopt_sealed(clone, (_HIDDEN_PATH,))


@dataclass(frozen=True)
class RunResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class FakeRunner:
    """Records enroot invocations and returns scripted RunResult objects."""

    def __init__(self, results: list[RunResult] | None = None) -> None:
        self.results = [] if results is None else list(results)
        self.calls: list[tuple[list[str], dict[str, str], float | None]] = []
        self.spawned: list[list[str]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> RunResult:
        del kwargs
        self.calls.append((list(argv), {} if env is None else dict(env), timeout_s))
        if self.results:
            return self.results.pop(0)
        return RunResult()

    def spawn(self, argv: Sequence[str], **kwargs: Any) -> int:
        del kwargs
        self.spawned.append(list(argv))
        return 1


def _bare_enroot(
    runner: FakeRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    started: bool = True,
) -> EnrootSandbox:
    sandbox = EnrootSandbox.__new__(EnrootSandbox)
    raw = cast(Any, sandbox)
    raw.spec = SimpleNamespace(name="box", image="img:1", memory_mb=0)
    raw._runner = runner
    sandbox._started = started
    enroot_data = tmp_path / "enroot-data"
    monkeypatch.setattr(
        sandbox,
        "_enroot_env",
        lambda: {"ENROOT_DATA_PATH": str(enroot_data)},
    )
    monkeypatch.setattr(sandbox, "_container_pids", lambda: [])
    return sandbox


def test_enroot_export_rootfs_argv_env_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner([RunResult()])
    sandbox = _bare_enroot(runner, monkeypatch, tmp_path)
    destination = tmp_path / "rootfs.sqsh"
    sandbox.export_rootfs(destination, max_bytes=4096, timeout_s=12.5)
    assert runner.calls == [
        (
            [
                "prlimit",
                "--fsize=4096",
                "--",
                "enroot",
                "export",
                "-o",
                str(destination),
                "box",
            ],
            {"ENROOT_DATA_PATH": str(tmp_path / "enroot-data")},
            12.5,
        )
    ]


def test_enroot_export_refused_when_not_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _bare_enroot(FakeRunner(), monkeypatch, tmp_path, started=False)
    with pytest.raises(SandboxRefusal, match="not-started"):
        sandbox.export_rootfs(tmp_path / "out.sqsh", max_bytes=1, timeout_s=1.0)


@pytest.mark.parametrize("max_bytes", [0, -1, True, 1.5])
def test_enroot_export_refuses_bad_max_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_bytes: object,
) -> None:
    sandbox = _bare_enroot(FakeRunner(), monkeypatch, tmp_path)
    with pytest.raises(SandboxRefusal, match="max_bytes"):
        sandbox.export_rootfs(
            tmp_path / "out.sqsh",
            max_bytes=cast(Any, max_bytes),
            timeout_s=1.0,
        )


def test_enroot_export_refuses_live_container_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    sandbox = _bare_enroot(runner, monkeypatch, tmp_path)
    monkeypatch.setattr(sandbox, "_container_pids", lambda: [17, 23])
    with pytest.raises(SandboxError, match="still running"):
        sandbox.export_rootfs(tmp_path / "out.sqsh", max_bytes=1, timeout_s=1.0)
    assert runner.calls == []


@pytest.mark.parametrize(
    "result",
    [
        RunResult(returncode=1, stderr="exporter failed"),
        RunResult(returncode=0, stderr="exporter stalled", timed_out=True),
    ],
)
def test_enroot_export_failure_and_timeout_are_sandbox_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: RunResult,
) -> None:
    sandbox = _bare_enroot(FakeRunner([result]), monkeypatch, tmp_path)
    with pytest.raises(SandboxError, match="enroot export failed"):
        sandbox.export_rootfs(tmp_path / "out.sqsh", max_bytes=1, timeout_s=1.0)


def test_enroot_start_restored_uses_archive_for_enroot_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner([RunResult(returncode=1, stderr="stop after create")])
    sandbox = _bare_enroot(runner, monkeypatch, tmp_path, started=False)
    archive = str(tmp_path / "committed.sqsh")
    with pytest.raises(SandboxError):
        sandbox.start_restored(archive)
    assert runner.calls
    create_argv = runner.calls[0][0]
    assert create_argv[:2] == ["enroot", "create"]
    assert archive in create_argv
    assert "img:1" not in create_argv


def test_snapshot_refuses_when_the_vault_is_not_sealed(tmp_path: Path) -> None:
    env, sandbox, _ = _started_case(tmp_path, "unsealed")
    sandbox.host_path("/tests").mkdir(parents=True, exist_ok=True)  # a hidden path reappears
    with pytest.raises(EnvInfraError) as caught:
        asyncio.run(env.snapshot(_new_store(tmp_path), vault_root=tmp_path / "retained"))
    assert caught.value.kind == "snapshot_integrity"


def test_snapshot_refuses_an_unwalkable_rootfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, _, _ = _started_case(tmp_path, "unwalkable")

    def failing(*args: object, **kwargs: object) -> tuple[str, ...]:
        raise PermissionError("denied")

    monkeypatch.setattr(sandbox_task, "rootless_lossy_entries", failing)
    with pytest.raises(EnvInfraError, match="could not be walked") as caught:
        asyncio.run(env.snapshot(_new_store(tmp_path), vault_root=tmp_path / "retained"))
    assert caught.value.kind == "snapshot_unsupported"


def test_snapshot_refuses_an_unmeasured_boundary_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, _, _ = _started_case(tmp_path, "unmeasured")
    monkeypatch.setattr(env, "_measure_signature", lambda: "unmeasured")
    with pytest.raises(EnvInfraError, match="unmeasured") as caught:
        asyncio.run(env.snapshot(_new_store(tmp_path), vault_root=tmp_path / "retained"))
    assert caught.value.kind == "snapshot_unsupported"


def test_deleting_or_reaping_a_snapshot_removes_its_retained_vault(tmp_path: Path) -> None:
    vault_root = tmp_path / "retained"
    clock = [1000.0]
    store = SnapshotStore(
        tmp_path / "store-cleanup",
        SnapshotLimits(
            max_bytes=1024 * 1024, total_bytes=8 * 1024 * 1024, ttl_s=10.0, timeout_s=1.0
        ),
        clock=lambda: clock[0],
        on_remove=sandbox_task.retained_vault_cleanup(vault_root),
    )
    env, _, _ = _started_case(tmp_path, "cleanup-a")
    deleted = asyncio.run(env.snapshot(store, vault_root=vault_root))
    env_b, _, _ = _started_case(tmp_path, "cleanup-b")
    reaped = asyncio.run(env_b.snapshot(store, vault_root=vault_root))
    assert len(list(vault_root.iterdir())) == 2

    store.delete(deleted.snapshot_id)
    assert not Path(deleted.meta["vault_clone"]).exists()
    assert Path(reaped.meta["vault_clone"]).exists()

    clock[0] += 100.0
    assert reaped.snapshot_id in store.reap()
    assert list(vault_root.iterdir()) == []


def test_retained_vault_cleanup_refuses_paths_outside_the_vault_root(tmp_path: Path) -> None:
    vault_root = tmp_path / "retained"
    vault_root.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    link = vault_root / ("vault-" + "a" * 32)
    link.symlink_to(victim)
    cleanup = sandbox_task.retained_vault_cleanup(vault_root)

    def record(path: str) -> SnapshotRecord:
        return SnapshotRecord(
            snapshot_id="snap_" + "0" * 32,
            archive_bytes=1,
            archive_sha256="0" * 64,
            env_signature="envsig_x",
            created_at=0.0,
            leases=0,
            meta={"vault_clone": path},
        )

    for bad in (str(victim), str(link), str(vault_root / "not-a-vault")):
        with pytest.raises(SnapshotRefusal, match="refusing to delete"):
            cleanup(record(bad))
    assert victim.exists()
    cleanup(record(str(vault_root / ("vault-" + "b" * 32))))  # already gone: nothing to do
    cleanup(
        SnapshotRecord(
            snapshot_id="snap_" + "1" * 32,
            archive_bytes=1,
            archive_sha256="0" * 64,
            env_signature="envsig_x",
            created_at=0.0,
            leases=0,
            meta={},
        )
    )


def test_snapshot_records_an_absolute_vault_clone_for_a_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, _, _ = _started_case(tmp_path, "relative")
    monkeypatch.chdir(tmp_path)
    record = asyncio.run(env.snapshot(_new_store(tmp_path), vault_root="retained-rel"))
    assert Path(record.meta["vault_clone"]).is_absolute()


def test_snapshot_removes_the_clone_when_the_store_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, _, _ = _started_case(tmp_path, "store-refuses")
    store = _new_store(tmp_path)

    def refuse(**kwargs: object) -> SnapshotRecord:
        raise SnapshotRefusal("refused by the store")

    monkeypatch.setattr(store, "commit", refuse)
    vault_root = tmp_path / "retained-refused"
    with pytest.raises(EnvInfraError) as caught:
        asyncio.run(env.snapshot(store, vault_root=vault_root))
    assert caught.value.kind == "snapshot_failed"
    assert list(vault_root.iterdir()) == []


def test_restore_closes_the_env_when_leasing_raises_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, _, _ = _started_case(tmp_path, "lease-src")
    store = _new_store(tmp_path)
    record = asyncio.run(env.snapshot(store, vault_root=tmp_path / "retained-lease"))
    fresh, _, _, _ = _fresh_case(tmp_path, "lease-dst")

    def broken(snapshot_id: str) -> SnapshotRecord:
        raise OSError("ESTALE")

    monkeypatch.setattr(store, "lease", broken)
    with pytest.raises(EnvInfraError) as caught:
        asyncio.run(fresh.restore(store, record.snapshot_id))
    assert caught.value.kind == "snapshot_integrity"
    with pytest.raises(EnvRefusal):
        asyncio.run(fresh.execute("ls"))
