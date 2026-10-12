"""Tests for the mini-SWE-agent Enroot execution environment."""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from foundationscale.agentic_rl.sandbox.enroot import RunResult, SandboxRefusal
from foundationscale.agentic_rl.sandbox.mini_swe_env import EnrootMiniSweEnvironment

_SENTINEL = "FS_HOST_ENV_SENTINEL_9f3a"


class FakeRunner:
    """Records argv and emulates enroot import/create/remove side effects in temp dirs."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[float | None] = []
        self.envs: list[Mapping[str, str] | None] = []
        self.start_results: list[RunResult] = []
        self.default_result = RunResult(returncode=0, stdout="", stderr="")

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunResult:
        argv = list(argv)
        self.calls.append(argv)
        self.envs.append(env)
        self.timeouts.append(timeout_s)
        if argv[:2] == ["enroot", "import"]:
            out = Path(argv[argv.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("image", encoding="utf-8")
            return self.default_result
        if argv[:2] == ["enroot", "create"]:
            name = argv[argv.index("-n") + 1]
            data_root = Path((env or {}).get("ENROOT_DATA_PATH", "/tmp"))
            (data_root / name).mkdir(parents=True, exist_ok=True)
            return self.default_result
        if argv[:2] == ["enroot", "remove"]:
            name = argv[-1]
            data_root = Path((env or {}).get("ENROOT_DATA_PATH", "/tmp"))
            rootfs = data_root / name
            if rootfs.is_dir():
                for entry in sorted(rootfs.rglob("*"), reverse=True):
                    if entry.is_file() or entry.is_symlink():
                        entry.unlink()
                    else:
                        entry.rmdir()
                rootfs.rmdir()
            return self.default_result
        if "enroot" in argv and "start" in argv:
            return self.start_results.pop(0) if self.start_results else self.default_result
        return self.default_result


@pytest.fixture()
def minisweagent_modules():
    """Install a stand-in minisweagent package and remove it afterwards."""
    pkg = types.ModuleType("minisweagent")
    pkg.__path__ = []
    exc_mod = types.ModuleType("minisweagent.exceptions")

    class Submitted(Exception):
        """Carries the submission payload as args[0]."""

    exc_mod.Submitted = Submitted
    pkg.exceptions = exc_mod
    saved = {
        k: v for k, v in sys.modules.items() if k == "minisweagent" or k.startswith("minisweagent.")
    }
    sys.modules["minisweagent"] = pkg
    sys.modules["minisweagent.exceptions"] = exc_mod
    yield pkg
    for key in [k for k in sys.modules if k == "minisweagent" or k.startswith("minisweagent.")]:
        sys.modules.pop(key, None)
    sys.modules.update(saved)


@pytest.fixture()
def paths(tmp_path: Path) -> dict[str, str]:
    """Provide absolute data_root/image_store/scratch_root paths under tmp_path."""
    return {
        "image": "docker://example/img:latest",
        "data_root": str(tmp_path / "data"),
        "image_store": str(tmp_path / "store"),
        "scratch_root": str(tmp_path / "scratch"),
    }


def _make(paths: dict[str, str], runner: FakeRunner, **kwargs: object) -> EnrootMiniSweEnvironment:
    return EnrootMiniSweEnvironment(**paths, runner=runner, **kwargs)


def _start_calls(runner: FakeRunner) -> list[list[str]]:
    """Return argv lists that invoke `enroot start`, wherever it appears in argv."""
    return [c for c in runner.calls if "enroot" in c and "start" in c]


def test_unknown_config_key_refused(paths: dict[str, str]) -> None:
    """Unknown config keys raise SandboxRefusal naming them."""
    with pytest.raises(SandboxRefusal) as excinfo:
        EnrootMiniSweEnvironment(**paths, bogus_key=1)
    assert "bogus_key" in str(excinfo.value)


def test_derived_name_format(paths: dict[str, str]) -> None:
    """Derived names match mini-<8 hex> and are unique."""
    env1 = _make(paths, FakeRunner())
    env2 = _make(paths, FakeRunner())
    import re

    assert re.fullmatch(r"mini-[0-9a-f]{8}", env1.config.name)
    assert env1.config.name != env2.config.name


def test_sandbox_started_on_construction(paths: dict[str, str]) -> None:
    """Construction runs enroot import and enroot create immediately."""
    runner = FakeRunner()
    env = _make(paths, runner)
    assert any(c[:2] == ["enroot", "create"] for c in runner.calls)
    assert env.sandbox._started is True


def test_execute_returns_output_and_appends_stderr(paths: dict[str, str]) -> None:
    """execute returns stdout+stderr and returncode, appending stderr only when non-empty."""
    runner = FakeRunner()
    env = _make(paths, runner)
    # Construction already runs enroot import + create; queue exec results AFTER it.
    runner.start_results.append(RunResult(returncode=3, stdout="out", stderr="err"))
    result = env.execute({"command": "echo hi"})
    assert result["output"] == "outerr"
    assert result["returncode"] == 3
    assert result["exception_info"] == ""
    runner.start_results.append(RunResult(returncode=0, stdout="out", stderr=""))
    result = env.execute({"command": "echo hi"})
    assert result["output"] == "out"


def test_cwd_and_env_in_exec_argv(paths: dict[str, str]) -> None:
    """cwd and env reach the enroot start argv as the workdir and --env flags."""
    runner = FakeRunner()
    env = _make(paths, runner, cwd="/work", env={"A": "1"})
    env.execute({"command": "ls"}, cwd="/elsewhere")
    start = _start_calls(runner)[-1]  # the uname probe at start-up runs first
    assert "cd /elsewhere && ls" in start[-1]
    env_pairs = [start[i + 1] for i, x in enumerate(start) if x == "--env"]
    assert "A=1" in env_pairs


def test_timeout_returns_minus_one(paths: dict[str, str]) -> None:
    """A timed-out exec yields returncode -1 and exception_info naming the timeout."""
    runner = FakeRunner()
    env = _make(paths, runner)
    runner.start_results.append(RunResult(returncode=124, stdout="", stderr="", timed_out=True))
    result = env.execute({"command": "sleep 999"})
    assert result["returncode"] == -1
    assert "timed out" in result["exception_info"].lower()


def test_completion_sentinel_raises_submitted(paths: dict[str, str], minisweagent_modules) -> None:
    """The sentinel first line with rc 0 raises Submitted carrying the remaining lines."""
    runner = FakeRunner()
    env = _make(paths, runner)
    runner.start_results.append(
        RunResult(
            returncode=0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nline1\nline2\n", stderr=""
        )
    )
    with pytest.raises(minisweagent_modules.exceptions.Submitted) as excinfo:
        env.execute({"command": "done"})
    assert excinfo.value.args[0]["extra"]["submission"] == "line1\nline2\n"


def test_completion_sentinel_nonzero_does_not_raise(
    paths: dict[str, str], minisweagent_modules
) -> None:
    """The sentinel with a non-zero returncode does not raise Submitted."""
    runner = FakeRunner()
    env = _make(paths, runner)
    runner.start_results.append(
        RunResult(returncode=1, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nbody\n", stderr="")
    )
    result = env.execute({"command": "done"})
    assert result["returncode"] == 1


def test_template_vars_merge_and_no_host_env(paths: dict[str, str], monkeypatch) -> None:
    """get_template_vars merges config + kwargs and never leaks host environment variables."""
    monkeypatch.setenv(_SENTINEL, "secret")
    env = _make(paths, FakeRunner(), env={"A": "1"}, timeout=7)
    vars_ = env.get_template_vars(extra="x", timeout=9)
    assert vars_["env"]["A"] == "1"
    assert vars_["extra"] == "x"
    assert vars_["timeout"] == 9
    assert vars_["system"] == "Linux"
    assert vars_["machine"] == "aarch64"
    assert _SENTINEL not in vars_


def test_serialize_shape(paths: dict[str, str]) -> None:
    """serialize returns the LocalEnvironment-shaped dict with environment_type."""
    env = _make(paths, FakeRunner())
    data = env.serialize()
    config = data["info"]["config"]
    assert config["environment_type"] == (
        "foundationscale.agentic_rl.sandbox.mini_swe_env.EnrootMiniSweEnvironment"
    )
    assert config["environment"]["image"] == paths["image"]


def test_cleanup_idempotent_and_removes_container(paths: dict[str, str]) -> None:
    """cleanup stops and removes the container, and is safe to call repeatedly."""
    runner = FakeRunner()
    env = _make(paths, runner)
    name = env.config.name
    assert (Path(paths["data_root"]) / name).is_dir()
    env.cleanup()
    env.cleanup()
    assert not (Path(paths["data_root"]) / name).exists()
    assert sum(1 for c in runner.calls if c[:2] == ["enroot", "remove"]) == 1


def test_template_vars_report_the_sandbox_uname_not_the_host(tmp_path: Path) -> None:
    """uname fields come from inside the sandbox (probed once at start-up)."""
    runner = FakeRunner()
    runner.start_results = [
        RunResult(returncode=0, stdout="Linux\nsbx-node\n6.8.0-gb\n#1 SMP\naarch64\n", stderr="")
    ]
    env = _make(_paths_for(tmp_path), runner)
    tv = env.get_template_vars()
    assert (tv["system"], tv["node"], tv["release"], tv["machine"]) == (
        "Linux",
        "sbx-node",
        "6.8.0-gb",
        "aarch64",
    )


def test_template_vars_fall_back_when_the_uname_probe_fails(tmp_path: Path) -> None:
    """A failing probe yields fixed fallback values, never a crash or host values."""
    runner = FakeRunner()
    runner.start_results = [RunResult(returncode=1, stdout="", stderr="no uname")]
    env = _make(_paths_for(tmp_path), runner)
    tv = env.get_template_vars()
    assert tv["system"] == "Linux" and tv["machine"] == "aarch64" and tv["release"] == ""


def _paths_for(tmp_path: Path) -> dict[str, str]:
    """Same roots as the ``paths`` fixture, for tests that build them directly."""
    roots = {
        "data_root": str(tmp_path / "data"),
        "image_store": str(tmp_path / "store"),
        "scratch_root": str(tmp_path / "scratch"),
    }
    for root in roots.values():
        Path(root).mkdir(parents=True, exist_ok=True)
    return {"image": "docker://python:3.12-slim", **roots}
