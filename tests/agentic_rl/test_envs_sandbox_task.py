"""Tests for the ``enroot_task`` sandbox: config parsing, report parsing and lifecycle."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

import foundationscale.agentic_rl.envs.sandbox_task as sandbox_task
from foundationscale.agentic_rl.envs.base import (
    EnvCapabilities,
    EnvInfraError,
    EnvRefusal,
    EnvSpec,
    TaskAwareEnvBackend,
    VerifiableEnvironment,
    available_env_backends,
    get_env_backend,
    register_env_backend,
)
from foundationscale.agentic_rl.envs.sandbox_task import (
    EnrootTaskBackend,
    SandboxLikeFull,
    SandboxRoots,
    TaskSandboxConfig,
    TaskSandboxEnvironment,
    parse_pytest_report,
)
from foundationscale.agentic_rl.harness.base import EpisodeTask
from foundationscale.agentic_rl.integrity import IntegrityRefusal
from foundationscale.agentic_rl.sandbox.enroot import SandboxError, SandboxSpec
from foundationscale.agentic_rl.sandbox.hidden import HiddenSpec

_META: dict[str, str] = {"image": "img:1", "test_cmd": "pytest -q"}
_ROOTS = SandboxRoots(data_root="/data", image_store="/images", scratch_root="/scratch")


@dataclass(frozen=True)
class FakeResult:
    """One scripted sandbox ``exec`` result."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    timed_out: bool = False


class FakeSandbox:
    """Records calls; serves scripted results/exceptions from a queue."""

    def __init__(
        self,
        results: Sequence[FakeResult | BaseException] = (),
        *,
        events: list[str] | None = None,
        start_exc: BaseException | None = None,
        stop_exc: BaseException | None = None,
    ) -> None:
        self.events: list[str] = [] if events is None else events
        self.started: int = 0
        self.stopped: list[bool] = []
        self.execs: list[tuple[str, str | None, Mapping[str, str] | None, float | None]] = []
        self.host_paths: list[str] = []
        self._results: deque[FakeResult | BaseException] = deque(results)
        self._start_exc = start_exc
        self._stop_exc = stop_exc

    def start(self) -> None:
        self.events.append("sandbox.start")
        self.started += 1
        if self._start_exc is not None:
            raise self._start_exc

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> FakeResult:
        self.events.append("sandbox.exec")
        self.execs.append((command, cwd, env, timeout_s))
        item = self._results.popleft()
        if isinstance(item, BaseException):
            raise item
        return item

    def stop(self, *, delete: bool = True) -> None:
        self.events.append("sandbox.stop")
        self.stopped.append(delete)
        if self._stop_exc is not None:
            raise self._stop_exc

    def host_path(self, container_path: str) -> Path:
        self.host_paths.append(container_path)
        return Path(container_path)


class FakeVault:
    """Records ``seal``/``unseal`` order; optionally raises."""

    def __init__(
        self,
        *,
        events: list[str] | None = None,
        seal_exc: BaseException | None = None,
        unseal_exc: BaseException | None = None,
    ) -> None:
        self.events: list[str] = [] if events is None else events
        self.calls: list[str] = []
        self._seal_exc = seal_exc
        self._unseal_exc = unseal_exc

    def seal(self) -> tuple[str, ...]:
        self.events.append("vault.seal")
        self.calls.append("seal")
        if self._seal_exc is not None:
            raise self._seal_exc
        return ("/tests",)

    def unseal(self) -> tuple[str, ...]:
        self.events.append("vault.unseal")
        self.calls.append("unseal")
        if self._unseal_exc is not None:
            raise self._unseal_exc
        return ("/tests",)


def _task(metadata: Mapping[str, str]) -> EpisodeTask:
    return EpisodeTask(
        uid="t",
        session_id="t#0",
        messages=({"role": "user", "content": "fix"},),
        metadata=dict(metadata),
    )


def _spec(
    *,
    env: Mapping[str, str] | None = None,
    max_output_bytes: int = 64,
    setup_commands: tuple[str, ...] = (),
    exec_timeout_s: float = 5.0,
    episode_timeout_s: float = 60.0,
) -> EnvSpec:
    return EnvSpec(
        backend="enroot_task",
        env={} if env is None else dict(env),
        exec_timeout_s=exec_timeout_s,
        episode_timeout_s=episode_timeout_s,
        max_output_bytes=max_output_bytes,
        setup_commands=setup_commands,
    )


def _env(
    *,
    spec: EnvSpec | None = None,
    config: TaskSandboxConfig | None = None,
    sandbox: FakeSandbox | None = None,
    vault: FakeVault | None = None,
) -> tuple[TaskSandboxEnvironment, FakeSandbox, FakeVault]:
    real_sandbox = FakeSandbox() if sandbox is None else sandbox
    real_vault = FakeVault() if vault is None else vault
    environment = TaskSandboxEnvironment(
        spec=_spec() if spec is None else spec,
        config=TaskSandboxConfig.from_task(_task(_META)) if config is None else config,
        sandbox=real_sandbox,
        vault=real_vault,
    )
    return environment, real_sandbox, real_vault


def _fake_sandbox_factory(spec: SandboxSpec) -> FakeSandbox:
    del spec
    return FakeSandbox()


def _fake_vault_factory(sandbox: SandboxLikeFull, spec: HiddenSpec, vault_dir: str) -> FakeVault:
    del sandbox, spec, vault_dir
    return FakeVault()


def test_sandbox_roots_from_environ_reads_all_variables() -> None:
    environ = {
        "FS_SANDBOX_DATA_ROOT": "/var/data",
        "FS_SANDBOX_IMAGE_STORE": "/var/images",
        "FS_SANDBOX_SCRATCH_ROOT": "/var/scratch",
        "FS_SANDBOX_ENROOT_SYSCONF": "/etc/enroot",
    }
    roots = SandboxRoots.from_environ(environ)
    assert roots == SandboxRoots(
        data_root="/var/data",
        image_store="/var/images",
        scratch_root="/var/scratch",
        enroot_sysconf="/etc/enroot",
    )


def test_sandbox_roots_missing_variable_refused() -> None:
    environ = {"FS_SANDBOX_DATA_ROOT": "/d", "FS_SANDBOX_IMAGE_STORE": "/i"}
    with pytest.raises(EnvRefusal, match="FS_SANDBOX_SCRATCH_ROOT: expected an absolute path"):
        SandboxRoots.from_environ(environ)


def test_sandbox_roots_relative_path_refused() -> None:
    environ = {
        "FS_SANDBOX_DATA_ROOT": "relative",
        "FS_SANDBOX_IMAGE_STORE": "/i",
        "FS_SANDBOX_SCRATCH_ROOT": "/s",
    }
    with pytest.raises(EnvRefusal, match="expected an absolute path, got str 'relative'"):
        SandboxRoots.from_environ(environ)


def test_sandbox_roots_optional_sysconf_defaults_to_none() -> None:
    environ = {
        "FS_SANDBOX_DATA_ROOT": "/d",
        "FS_SANDBOX_IMAGE_STORE": "/i",
        "FS_SANDBOX_SCRATCH_ROOT": "/s",
        "FS_SANDBOX_ENROOT_SYSCONF": "",
    }
    assert SandboxRoots.from_environ(environ).enroot_sysconf is None
    environ.pop("FS_SANDBOX_ENROOT_SYSCONF")
    assert SandboxRoots.from_environ(environ).enroot_sysconf is None


def test_task_sandbox_config_defaults() -> None:
    config = TaskSandboxConfig.from_task(_task({"image": "img:1", "test_cmd": "pytest"}))
    assert config == TaskSandboxConfig(
        image="img:1",
        workdir="/repo",
        test_cmd="pytest",
        hidden_paths=("/tests",),
        test_timeout_s=600.0,
    )


def test_task_sandbox_config_missing_image_refused() -> None:
    with pytest.raises(EnvRefusal, match="image: expected a non-empty str"):
        TaskSandboxConfig.from_task(_task({"test_cmd": "pytest"}))


def test_task_sandbox_config_empty_test_cmd_refused() -> None:
    with pytest.raises(EnvRefusal, match="test_cmd: expected a non-empty str"):
        TaskSandboxConfig.from_task(_task({"image": "img:1", "test_cmd": ""}))


def test_task_sandbox_config_relative_workdir_refused() -> None:
    metadata = {"image": "img:1", "test_cmd": "pytest", "workdir": "repo"}
    with pytest.raises(EnvRefusal, match="workdir: expected an absolute path"):
        TaskSandboxConfig.from_task(_task(metadata))


def test_task_sandbox_config_hidden_paths_not_json_refused() -> None:
    metadata = {"image": "img:1", "test_cmd": "pytest", "hidden_paths": "not json"}
    with pytest.raises(EnvRefusal, match="hidden_paths: expected a JSON list"):
        TaskSandboxConfig.from_task(_task(metadata))


def test_task_sandbox_config_hidden_paths_empty_refused() -> None:
    metadata = {"image": "img:1", "test_cmd": "pytest", "hidden_paths": "[]"}
    with pytest.raises(EnvRefusal, match="hidden_paths: expected a non-empty JSON list"):
        TaskSandboxConfig.from_task(_task(metadata))


def test_task_sandbox_config_hidden_paths_non_list_refused() -> None:
    metadata = {"image": "img:1", "test_cmd": "pytest", "hidden_paths": '{"a": 1}'}
    with pytest.raises(EnvRefusal, match="hidden_paths: expected a non-empty JSON list"):
        TaskSandboxConfig.from_task(_task(metadata))


def test_task_sandbox_config_hidden_paths_bad_entry_refused() -> None:
    base = {"image": "img:1", "test_cmd": "pytest"}
    relative = dict(base, hidden_paths='["/tests", "tests"]')
    with pytest.raises(EnvRefusal, match="got entry str 'tests'"):
        TaskSandboxConfig.from_task(_task(relative))
    non_str = dict(base, hidden_paths="[1]")
    with pytest.raises(EnvRefusal, match="got entry int 1"):
        TaskSandboxConfig.from_task(_task(non_str))


def test_task_sandbox_config_test_timeout_not_a_number_refused() -> None:
    metadata = {"image": "img:1", "test_cmd": "pytest", "test_timeout_s": "soon"}
    with pytest.raises(EnvRefusal, match="test_timeout_s: expected a float > 0"):
        TaskSandboxConfig.from_task(_task(metadata))


def test_task_sandbox_config_test_timeout_non_positive_refused() -> None:
    base = {"image": "img:1", "test_cmd": "pytest"}
    for raw in ("0", "-1", "nan", "inf"):
        metadata = dict(base, test_timeout_s=raw)
        with pytest.raises(EnvRefusal, match="test_timeout_s: expected a float > 0"):
            TaskSandboxConfig.from_task(_task(metadata))


def test_task_sandbox_config_metadata_none_refused() -> None:
    try:
        task = EpisodeTask(
            uid="t",
            session_id="t#0",
            messages=({"role": "user", "content": "fix"},),
            metadata=None,
        )
    except (TypeError, ValueError):
        return
    with pytest.raises((EnvRefusal, TypeError)):
        TaskSandboxConfig.from_task(task)


_REPORT = "\n".join(
    (
        "============================= test session starts =============================",
        "collected 6 items",
        "",
        "PASSED tests/ignored.py::test_before_header",
        "========================== short test summary info ===========================",
        "PASSED tests/test_a.py::test_one",
        "FAILED tests/test_a.py::test_two - boom",
        "ERROR tests/test_a.py::test_three",
        "XPASS tests/test_a.py::test_four",
        "XFAIL tests/test_a.py::test_five",
        "SKIPPED [1] tests/test_a.py:3: no reason",
        "======================== 1 failed, 1 passed in 0.10s =========================",
    )
)


def test_parse_pytest_report_reads_short_summary_section() -> None:
    assert parse_pytest_report(_REPORT) == {
        "tests/test_a.py::test_one": "passed",
        "tests/test_a.py::test_two": "failed",
        "tests/test_a.py::test_three": "error",
        "tests/test_a.py::test_four": "passed",
        "tests/test_a.py::test_five": "skipped",
    }


def test_parse_pytest_report_worse_duplicate_wins() -> None:
    stdout = "\n".join(
        (
            "PASSED tests/test_a.py::test_dup",
            "SKIPPED tests/test_a.py::test_dup - flaky",
            "FAILED tests/test_a.py::test_dup - boom",
            "ERROR tests/test_a.py::test_dup",
            "PASSED tests/test_a.py::test_dup",
        )
    )
    assert parse_pytest_report(stdout) == {"tests/test_a.py::test_dup": "error"}


def test_parse_pytest_report_failed_duplicate_beats_passed() -> None:
    stdout = "\n".join(
        (
            "PASSED tests/test_a.py::test_dup",
            "FAILED tests/test_a.py::test_dup - boom",
            "PASSED tests/test_a.py::test_dup",
        )
    )
    assert parse_pytest_report(stdout) == {"tests/test_a.py::test_dup": "failed"}


def test_parse_pytest_report_skipped_duplicate_beats_passed() -> None:
    stdout = "\n".join(
        ("PASSED tests/test_a.py::test_dup", "SKIPPED tests/test_a.py::test_dup - flaky")
    )
    assert parse_pytest_report(stdout) == {"tests/test_a.py::test_dup": "skipped"}


def test_parse_pytest_report_without_header_scans_every_line() -> None:
    stdout = "\n".join(("PASSED tests/test_a.py::test_one", "FAILED tests/test_a.py::test_two - x"))
    assert parse_pytest_report(stdout) == {
        "tests/test_a.py::test_one": "passed",
        "tests/test_a.py::test_two": "failed",
    }


def test_parse_pytest_report_ignores_unparseable_lines() -> None:
    stdout = "\n".join(("PASSED", "garbage", "FAILED ", "PASSED tests/test_a.py::test_one"))
    assert parse_pytest_report(stdout) == {"tests/test_a.py::test_one": "passed"}


def test_parse_pytest_report_empty_output() -> None:
    assert parse_pytest_report("") == {}


def test_execute_caps_output_with_stdout_priority() -> None:
    sandbox = FakeSandbox([FakeResult(stdout="abcdefgh", stderr="XY", return_code=0)])
    env, _, _ = _env(spec=_spec(max_output_bytes=5), sandbox=sandbox)
    asyncio.run(env.start())
    result = asyncio.run(env.execute("cmd"))
    assert result.stdout == "abcde"
    assert result.stderr == ""
    assert result.truncated_bytes == 5
    assert result.exit_code == 0


def test_execute_cut_multibyte_character_decodes_with_replacement() -> None:
    sandbox = FakeSandbox([FakeResult(stdout="日本", stderr="")])
    env, _, _ = _env(spec=_spec(max_output_bytes=4), sandbox=sandbox)
    asyncio.run(env.start())
    result = asyncio.run(env.execute("cmd"))
    assert result.stdout == "\u65e5\ufffd"
    assert result.truncated_bytes == 2


def test_execute_gives_stderr_the_leftover_budget() -> None:
    sandbox = FakeSandbox([FakeResult(stdout="abc", stderr="wxyz")])
    env, _, _ = _env(spec=_spec(max_output_bytes=10), sandbox=sandbox)
    asyncio.run(env.start())
    result = asyncio.run(env.execute("cmd"))
    assert result.stdout == "abc"
    assert result.stderr == "wxyz"
    assert result.truncated_bytes == 0


def test_start_seals_after_sandbox_start_then_runs_setup() -> None:
    events: list[str] = []
    sandbox = FakeSandbox([FakeResult(), FakeResult()], events=events)
    vault = FakeVault(events=events)
    env, _, _ = _env(
        spec=_spec(env={"K": "V"}, setup_commands=("one", "two")),
        sandbox=sandbox,
        vault=vault,
    )
    asyncio.run(env.start())
    assert events == ["sandbox.start", "vault.seal", "sandbox.exec", "sandbox.exec"]
    assert sandbox.execs == [
        ("one", "/repo", {"K": "V"}, 5.0),
        ("two", "/repo", {"K": "V"}, 5.0),
    ]


def test_start_setup_non_zero_exit_is_infra_error() -> None:
    sandbox = FakeSandbox([FakeResult(return_code=3)])
    env, _, _ = _env(spec=_spec(setup_commands=("setup",)), sandbox=sandbox)
    with pytest.raises(EnvInfraError, match="failed with exit code 3"):
        asyncio.run(env.start())


def test_start_setup_timeout_is_infra_error() -> None:
    sandbox = FakeSandbox([FakeResult(timed_out=True)])
    env, _, _ = _env(spec=_spec(setup_commands=("setup",)), sandbox=sandbox)
    with pytest.raises(EnvInfraError, match="failed with timed out"):
        asyncio.run(env.start())


def test_start_setup_exception_is_infra_error() -> None:
    sandbox = FakeSandbox([SandboxError("boom")])
    env, _, _ = _env(spec=_spec(setup_commands=("setup",)), sandbox=sandbox)
    with pytest.raises(EnvInfraError, match="setup_commands must all run before the episode"):
        asyncio.run(env.start())


def test_start_sandbox_start_failure_is_infra_error() -> None:
    sandbox = FakeSandbox(start_exc=SandboxError("boom"))
    env, _, _ = _env(sandbox=sandbox)
    with pytest.raises(EnvInfraError, match="starting the episode sandbox and sealing"):
        asyncio.run(env.start())


def test_start_vault_seal_failure_is_infra_error() -> None:
    vault = FakeVault(seal_exc=IntegrityRefusal("no"))
    env, _, _ = _env(vault=vault)
    with pytest.raises(EnvInfraError, match="the container could not be prepared at all"):
        asyncio.run(env.start())


def test_start_twice_refused() -> None:
    env, _, _ = _env()
    asyncio.run(env.start())
    with pytest.raises(EnvRefusal, match="already-started one"):
        asyncio.run(env.start())


def test_start_after_close_refused() -> None:
    env, _, _ = _env()
    asyncio.run(env.close())
    with pytest.raises(EnvRefusal, match="there is nothing left to start"):
        asyncio.run(env.start())


def test_execute_before_start_refused() -> None:
    env, _, _ = _env()
    with pytest.raises(EnvRefusal, match="there is no started sandbox to run the command in"):
        asyncio.run(env.execute("ls"))


def test_execute_after_verify_refused() -> None:
    sandbox = FakeSandbox([FakeResult(stdout="short test summary info"), FakeResult()])
    env, _, _ = _env(sandbox=sandbox)
    asyncio.run(env.start())
    asyncio.run(env.verify())
    with pytest.raises(EnvRefusal, match="no further agent command may run"):
        asyncio.run(env.execute("ls"))


def test_execute_after_close_refused() -> None:
    env, _, _ = _env()
    asyncio.run(env.start())
    asyncio.run(env.close())
    with pytest.raises(EnvRefusal, match="its commands would run nowhere"):
        asyncio.run(env.execute("ls"))


def test_execute_passes_env_and_default_timeout() -> None:
    sandbox = FakeSandbox([FakeResult(stdout="ok")])
    env, _, _ = _env(spec=_spec(env={"K": "V"}), sandbox=sandbox)
    asyncio.run(env.start())
    asyncio.run(env.execute("ls"))
    assert sandbox.execs == [("ls", "/repo", {"K": "V"}, 5.0)]


def test_execute_empty_env_is_none() -> None:
    sandbox = FakeSandbox([FakeResult(stdout="ok")])
    env, _, _ = _env(spec=_spec(env={}), sandbox=sandbox)
    asyncio.run(env.start())
    asyncio.run(env.execute("ls"))
    assert sandbox.execs == [("ls", "/repo", None, 5.0)]


def test_execute_timeout_override_wins() -> None:
    sandbox = FakeSandbox([FakeResult(stdout="ok")])
    env, _, _ = _env(spec=_spec(env={"K": "V"}), sandbox=sandbox)
    asyncio.run(env.start())
    asyncio.run(env.execute("ls", timeout_s=1.5))
    assert sandbox.execs == [("ls", "/repo", {"K": "V"}, 1.5)]


def test_execute_timeout_yields_no_exit_code() -> None:
    sandbox = FakeSandbox([FakeResult(timed_out=True, return_code=0)])
    env, _, _ = _env(sandbox=sandbox)
    asyncio.run(env.start())
    result = asyncio.run(env.execute("ls"))
    assert result.exit_code is None
    assert result.timed_out is True


def test_execute_sandbox_error_is_infra_error() -> None:
    sandbox = FakeSandbox([SandboxError("boom")])
    env, _, _ = _env(sandbox=sandbox)
    asyncio.run(env.start())
    with pytest.raises(EnvInfraError, match="the command could not be run at all"):
        asyncio.run(env.execute("ls"))


def test_verify_unseals_then_reports_tests() -> None:
    report = "\n".join(
        (
            "short test summary info",
            "PASSED tests/test_a.py::test_one",
            "FAILED tests/test_a.py::test_two - boom",
        )
    )
    sandbox = FakeSandbox([FakeResult(stdout=report, return_code=1)])
    env, real_sandbox, vault = _env(sandbox=sandbox)
    asyncio.run(env.start())
    outcome = asyncio.run(env.verify())
    tests = {
        "tests/test_a.py::test_one": "passed",
        "tests/test_a.py::test_two": "failed",
    }
    assert outcome == {
        "swe_status": "ok",
        "swe_tests": json.dumps(tests, sort_keys=True),
        "swe_exit_code": "1",
    }
    assert vault.calls == ["seal", "unseal"]
    assert real_sandbox.execs == [("pytest -q", "/repo", None, 600.0)]


def test_verify_timeout_is_error_outcome() -> None:
    sandbox = FakeSandbox([FakeResult(timed_out=True)])
    env, _, _ = _env(sandbox=sandbox)
    asyncio.run(env.start())
    assert asyncio.run(env.verify()) == {"swe_status": "error", "swe_error": "test_cmd timed out"}


def test_verify_without_tests_is_error_outcome() -> None:
    sandbox = FakeSandbox([FakeResult(stdout="no tests ran", return_code=5)])
    env, _, _ = _env(sandbox=sandbox)
    asyncio.run(env.start())
    outcome = asyncio.run(env.verify())
    assert outcome == {"swe_status": "error", "swe_error": "no test results (exit 5)"}


def test_verify_vault_exception_is_infra_error() -> None:
    vault = FakeVault(unseal_exc=IntegrityRefusal("no"))
    env, _, _ = _env(vault=vault)
    asyncio.run(env.start())
    with pytest.raises(EnvInfraError, match="the verifier phase could not run at all"):
        asyncio.run(env.verify())


def test_verify_sandbox_exception_is_infra_error() -> None:
    sandbox = FakeSandbox([SandboxError("boom")])
    env, _, _ = _env(sandbox=sandbox)
    asyncio.run(env.start())
    with pytest.raises(EnvInfraError, match="the verifier phase could not run at all"):
        asyncio.run(env.verify())


def test_verify_twice_refused() -> None:
    sandbox = FakeSandbox([FakeResult(stdout="short test summary info")])
    env, _, _ = _env(sandbox=sandbox)
    asyncio.run(env.start())
    asyncio.run(env.verify())
    with pytest.raises(EnvRefusal, match="already-verified one"):
        asyncio.run(env.verify())


def test_verify_before_start_refused() -> None:
    env, _, _ = _env()
    with pytest.raises(EnvRefusal, match="verify: expected a started environment"):
        asyncio.run(env.verify())


def test_verify_after_close_refused() -> None:
    env, _, _ = _env()
    asyncio.run(env.start())
    asyncio.run(env.close())
    with pytest.raises(EnvRefusal, match="its hidden tests cannot be restored"):
        asyncio.run(env.verify())


def test_close_is_idempotent() -> None:
    env, sandbox, _ = _env()
    asyncio.run(env.start())
    asyncio.run(env.close())
    asyncio.run(env.close())
    assert sandbox.stopped == [True]


def test_close_before_start_is_fine() -> None:
    env, sandbox, _ = _env()
    asyncio.run(env.close())
    assert sandbox.stopped == [True]


def test_close_stop_failure_is_infra_error_and_second_close_no_op() -> None:
    sandbox = FakeSandbox(stop_exc=SandboxError("boom"))
    env, _, _ = _env(sandbox=sandbox)
    with pytest.raises(EnvInfraError, match="deleting the episode sandbox raised"):
        asyncio.run(env.close())
    asyncio.run(env.close())
    assert sandbox.stopped == [True]


def test_environment_properties() -> None:
    config = TaskSandboxConfig.from_task(_task(dict(_META, workdir="/work")))
    env, _, _ = _env(spec=_spec(episode_timeout_s=42.0), config=config)
    assert env.workdir == "/work"
    assert env.episode_timeout_s == 42.0


def test_enroot_task_backend_name_and_capabilities() -> None:
    backend = EnrootTaskBackend(roots=_ROOTS)
    assert backend.name == "enroot_task"
    assert backend.capabilities() == EnvCapabilities(
        isolation="container",
        network=False,
        max_concurrency=None,
        platform="linux",
    )


def test_enroot_task_backend_create_refuses() -> None:
    backend = EnrootTaskBackend(roots=_ROOTS)
    with pytest.raises(EnvRefusal, match="it is created through create_for_task"):
        backend.create(_spec(), instance_id="t#0")


def test_create_for_task_builds_named_sandbox_and_vault(tmp_path: Path) -> None:
    sysconf = tmp_path / "enroot"
    sysconf.mkdir()
    (sysconf / "enroot.conf").write_text("")
    sandbox_specs: list[SandboxSpec] = []
    sandboxes: list[FakeSandbox] = []
    vault_calls: list[tuple[object, HiddenSpec, str]] = []

    def fake_sandbox(spec: SandboxSpec) -> FakeSandbox:
        sandbox_specs.append(spec)
        sandbox = FakeSandbox()
        sandboxes.append(sandbox)
        return sandbox

    def fake_vault(sandbox: SandboxLikeFull, spec: HiddenSpec, vault_dir: str) -> FakeVault:
        vault_calls.append((sandbox, spec, vault_dir))
        return FakeVault()

    backend = EnrootTaskBackend(
        roots=SandboxRoots("/data", "/images", "/scratch", str(sysconf)),
        sandbox_factory=fake_sandbox,
        vault_factory=fake_vault,
    )
    metadata = {
        "image": "img:1",
        "test_cmd": "pytest",
        "hidden_paths": '["/tests", "/solution"]',
    }
    env = backend.create_for_task(_spec(env={"K": "V"}), _task(metadata), instance_id="t#0")
    assert isinstance(env, TaskSandboxEnvironment)
    assert len(sandbox_specs) == 1
    spec = sandbox_specs[0]
    assert spec.name == "fsar-t-0"
    assert spec.image == "img:1"
    assert spec.network == "no-network"
    assert spec.isolate_network is True
    assert spec.workdir == "/repo"
    assert dict(spec.env) == {"K": "V"}
    assert spec.data_root == "/data"
    assert spec.image_store == "/images"
    assert spec.scratch_root == "/scratch"
    assert spec.enroot_sysconf == str(sysconf)
    assert len(vault_calls) == 1
    assert vault_calls[0][0] is sandboxes[0]
    assert vault_calls[0][1] == HiddenSpec(paths=("/tests", "/solution"))
    assert vault_calls[0][2] == str(Path("/scratch") / "fsar-t-0.vault")
    assert vault_calls[0][2].endswith(".vault")


def test_create_for_task_bad_instance_id_refused() -> None:
    backend = EnrootTaskBackend(
        roots=_ROOTS,
        sandbox_factory=_fake_sandbox_factory,
        vault_factory=_fake_vault_factory,
    )
    for instance_id in ("a/b", "", "a b", "a;b"):
        with pytest.raises(EnvRefusal, match="instance_id: expected a name matching"):
            backend.create_for_task(_spec(), _task(_META), instance_id=instance_id)


def test_enroot_task_backend_resolves_roots_lazily_and_caches() -> None:
    environ: dict[str, str] = {}
    backend = EnrootTaskBackend(
        environ=environ,
        sandbox_factory=_fake_sandbox_factory,
        vault_factory=_fake_vault_factory,
    )
    environ.update(
        {
            "FS_SANDBOX_DATA_ROOT": "/data",
            "FS_SANDBOX_IMAGE_STORE": "/images",
            "FS_SANDBOX_SCRATCH_ROOT": "/scratch",
        }
    )
    backend.create_for_task(_spec(), _task(_META), instance_id="t#0")
    environ.clear()
    backend.create_for_task(_spec(), _task(_META), instance_id="t#0")


def test_enroot_task_backend_missing_environ_refused_only_on_create() -> None:
    backend = EnrootTaskBackend(
        environ={},
        sandbox_factory=_fake_sandbox_factory,
        vault_factory=_fake_vault_factory,
    )
    match = "FS_SANDBOX_DATA_ROOT: expected an absolute path"
    with pytest.raises(EnvRefusal, match=match):
        backend.create_for_task(_spec(), _task(_META), instance_id="t#0")


def test_enroot_task_backend_default_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox_specs: list[SandboxSpec] = []
    sandbox_instances: list[object] = []
    vault_calls: list[tuple[object, HiddenSpec, str]] = []

    runners: list[object] = []

    class RecordingSandbox:
        def __init__(self, spec: SandboxSpec, *, runner: object = None) -> None:
            sandbox_specs.append(spec)
            sandbox_instances.append(self)
            runners.append(runner)

    class RecordingVault:
        def __init__(self, sandbox: object, spec: HiddenSpec, vault_dir: str) -> None:
            vault_calls.append((sandbox, spec, vault_dir))

    monkeypatch.setattr(sandbox_task, "EnrootSandbox", RecordingSandbox)
    monkeypatch.setattr(sandbox_task, "HiddenVault", RecordingVault)
    backend = EnrootTaskBackend(roots=_ROOTS, environ={})
    env = backend.create_for_task(_spec(), _task(_META), instance_id="t#0")
    assert isinstance(env, TaskSandboxEnvironment)
    assert [spec.name for spec in sandbox_specs] == ["fsar-t-0"]
    assert runners == [None]  # no FS_SANDBOX_BROKER: EnrootSandbox's own subprocess runner
    assert len(vault_calls) == 1
    assert vault_calls[0][0] is sandbox_instances[0]
    assert vault_calls[0][1] == HiddenSpec(paths=("/tests",))
    assert vault_calls[0][2] == str(Path("/scratch") / "fsar-t-0.vault")


def test_get_env_backend_enroot_task_is_task_aware() -> None:
    if "enroot_task" not in available_env_backends():
        register_env_backend("enroot_task", EnrootTaskBackend)
    backend = get_env_backend("enroot_task")
    assert isinstance(backend, EnrootTaskBackend)
    assert isinstance(backend, TaskAwareEnvBackend)


def test_task_sandbox_environment_is_verifiable() -> None:
    env, _, _ = _env()
    assert isinstance(env, VerifiableEnvironment)


def test_default_factory_uses_the_broker_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runners: list[object] = []

    class RecordingSandbox:
        def __init__(self, spec: SandboxSpec, *, runner: object = None) -> None:
            del spec
            runners.append(runner)

    monkeypatch.setattr(sandbox_task, "EnrootSandbox", RecordingSandbox)
    token_file = tmp_path / "token"
    token_file.write_text("secret\n")
    token_file.chmod(0o600)
    environ = {
        "FS_SANDBOX_BROKER": "127.0.0.1:18600",
        "FS_SANDBOX_BROKER_TOKEN_FILE": str(token_file),
    }
    backend = EnrootTaskBackend(roots=_ROOTS, environ=environ, vault_factory=_fake_vault_factory)
    backend.create_for_task(_spec(), _task(_META), instance_id="t#0")
    assert [type(r).__name__ for r in runners] == ["BrokerRunner"]
