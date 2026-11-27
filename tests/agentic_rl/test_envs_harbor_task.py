"""The ``harbor_task`` backend: Harbor reward files -> ``swe_tests`` that ``rewards/swe.py`` scores."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from foundationscale.agentic_rl.envs.base import EnvInfraError, EnvSpec, get_env_backend
from foundationscale.agentic_rl.envs.harbor_task import (
    ALL_TEST,
    HarborTaskBackend,
    HarborTaskEnvironment,
    reward_tests,
    short_instance_id,
)
from foundationscale.agentic_rl.envs.sandbox_task import SandboxRoots
from foundationscale.agentic_rl.harness.base import EpisodeTask
from foundationscale.agentic_rl.rewards.swe import SweReward
from foundationscale.agentic_rl.sandbox.enroot import SandboxError

META = {"image": "img.sqsh", "workdir": "/workspace", "test_cmd": "bash /tests/test.sh",
        "hidden_paths": '["/tests"]', "test_timeout_s": "60", "fail_to_pass": json.dumps([ALL_TEST]),
        "pass_to_pass": "[]", "verify_mode": "inplace"}


@dataclass(frozen=True)
class Result:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    timed_out: bool = False


class RootfsSandbox:
    """A fake sandbox whose container paths live under ``root``; ``on_exec`` plays test.sh."""

    def __init__(self, root: Path, on_exec: Callable[[Path], Result] | Exception) -> None:
        self.root, self.on_exec = root, on_exec

    def start(self) -> None:
        pass

    def exec(self, command: str, *, cwd=None, env=None, timeout_s=None) -> Result:
        if isinstance(self.on_exec, Exception):
            raise self.on_exec
        return self.on_exec(self.root)

    def stop(self, *, delete: bool = True) -> None:
        pass

    def host_path(self, container_path: str) -> Path:
        return self.root / container_path.lstrip("/")


class Vault:
    def seal(self) -> tuple[str, ...]:
        return ("/tests",)

    def unseal(self) -> tuple[str, ...]:
        return ("/tests",)


def _spec() -> EnvSpec:
    return EnvSpec(backend="harbor_task", env={}, exec_timeout_s=5.0, episode_timeout_s=60.0,
                   max_output_bytes=64, setup_commands=())


def _task(meta: Mapping[str, str] = META) -> EpisodeTask:
    return EpisodeTask(uid="t", session_id="t#0", messages=({"role": "user", "content": "do"},),
                       metadata=dict(meta))


def _env(tmp_path: Path, on_exec) -> HarborTaskEnvironment:
    backend = HarborTaskBackend(
        roots=SandboxRoots(data_root="/d", image_store="/i", scratch_root=str(tmp_path / "s")),
        sandbox_factory=lambda spec: RootfsSandbox(tmp_path / "rootfs", on_exec),
        vault_factory=lambda sandbox, spec, vault_dir: Vault(),
    )
    return backend.create_for_task(_spec(), _task(), instance_id="t#0")


def _writes(name: str, text: str) -> Callable[[Path], Result]:
    def run(root: Path) -> Result:
        out = root / "logs" / "verifier"
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(text, encoding="utf-8")
        return Result()
    return run


def _verify(env: HarborTaskEnvironment) -> dict[str, str]:
    return env._verify_sync()


def test_backend_is_registered() -> None:
    assert isinstance(get_env_backend("harbor_task"), HarborTaskBackend)


@pytest.mark.parametrize("name,text,expect_all", [
    ("reward.json", '{"a": 1, "b": {"c": 1.0}}', "passed"),
    ("reward.json", '{"a": 1, "b": 0.5}', "failed"),
    ("reward.txt", "1\n", "passed"),
    ("reward.txt", "0", "failed"),
])
def test_reward_tests(name, text, expect_all) -> None:
    tests = reward_tests(name, text)
    assert tests is not None and tests[ALL_TEST] == expect_all


@pytest.mark.parametrize("name,text", [("reward.json", "{}"), ("reward.json", '{"ok": true}'),
                                       ("reward.json", "not json"), ("reward.txt", "nan"), ("reward.txt", "")])
def test_reward_without_numeric_leaves_is_none(name, text) -> None:
    assert reward_tests(name, text) is None


def test_full_reward_scores_resolved(tmp_path) -> None:
    out = _verify(_env(tmp_path, _writes("reward.json", '{"x": 1, "y": 1}')))
    assert out["swe_status"] == "ok" and json.loads(out["swe_tests"]) == {
        "harbor::x": "passed", "harbor::y": "passed", ALL_TEST: "passed"}


def test_partial_reward_is_not_resolved_and_names_the_leaf(tmp_path) -> None:
    tests = json.loads(_verify(_env(tmp_path, _writes("reward.json", '{"x": 1, "y": 0}')))["swe_tests"])
    assert tests["harbor::y"] == "failed" and tests[ALL_TEST] == "failed"


def test_planted_reward_is_deleted_before_the_verifier_runs(tmp_path) -> None:
    planted = tmp_path / "rootfs" / "logs" / "verifier"
    planted.mkdir(parents=True)
    (planted / "reward.json").write_text('{"x": 1}', encoding="utf-8")
    (planted / "reward.txt").write_text("1", encoding="utf-8")
    out = _verify(_env(tmp_path, lambda root: Result(return_code=1)))  # test.sh writes nothing
    assert out["swe_status"] == "error" and "no reward file" in out["swe_error"]


def test_timeout_and_infra_failures(tmp_path) -> None:
    assert _verify(_env(tmp_path, lambda root: Result(timed_out=True)))["swe_status"] == "error"
    with pytest.raises(EnvInfraError):
        _verify(_env(tmp_path, SandboxError("boom")))


def test_leafless_reward_abstains(tmp_path) -> None:
    out = _verify(_env(tmp_path, _writes("reward.json", '{"note": "done"}')))
    assert out["swe_status"] == "error" and "no numeric reward" in out["swe_error"]


@dataclass
class _Outcome:
    metadata: dict[str, str]


@pytest.mark.parametrize("text,value", [('{"x": 1}', 1.0), ('{"x": 0.9}', 0.0)])
def test_swe_reward_scores_the_mapping(tmp_path, text, value) -> None:
    verdict = _verify(_env(tmp_path, _writes("reward.json", text)))
    got = asyncio.run(SweReward().score(_task(), _Outcome(verdict)))
    assert got.value == value


def test_long_instance_ids_fit_the_enroot_name_limit(tmp_path) -> None:
    long_id = "task_cuopt-numerical-optimization-api-cli_wdp6qs23#0"
    short = short_instance_id(long_id)
    assert len(short) <= 48 and short != short_instance_id(long_id[:-1] + "1")
    assert short_instance_id("t#0") == "t#0"
    names = []
    backend = HarborTaskBackend(
        roots=SandboxRoots(data_root="/d", image_store="/i", scratch_root=str(tmp_path / "s")),
        sandbox_factory=lambda spec: names.append(spec.name) or RootfsSandbox(tmp_path, lambda root: Result()),
        vault_factory=lambda sandbox, spec, vault_dir: Vault(),
    )
    backend.create_for_task(_spec(), _task(), instance_id=long_id)
    assert len(names[0]) <= 63


class FreshRootfs(RootfsSandbox):
    """Each instance owns its own rootfs (a pristine verifier container), and records being deleted."""

    stopped: list[str] = []

    def stop(self, *, delete: bool = True) -> None:
        FreshRootfs.stopped.append(str(self.root))


def _fresh_env(tmp_path, verifier):
    roots = iter([tmp_path / "agent", tmp_path / "verifier"])
    backend = HarborTaskBackend(
        roots=SandboxRoots(data_root="/d", image_store="/i", scratch_root=str(tmp_path / "s")),
        sandbox_factory=lambda spec: FreshRootfs(next(roots), verifier),
        vault_factory=lambda sandbox, spec, vault_dir: Vault(),
    )
    meta = {k: v for k, v in META.items() if k != "verify_mode"}
    return backend.create_for_task(_spec(), _task(meta), instance_id="t#0")


def test_fresh_mode_copies_the_workdir_and_ignores_tampering_outside_it(tmp_path):
    agent = tmp_path / "agent"
    (agent / "workspace").mkdir(parents=True)
    (agent / "workspace" / "answer.txt").write_text("42", encoding="utf-8")
    (agent / "usr" / "bin").mkdir(parents=True)
    (agent / "usr" / "bin" / "node").write_text("#!/bin/sh\necho hacked", encoding="utf-8")  # tampering, agent side
    (agent / "logs" / "verifier").mkdir(parents=True)
    (agent / "logs" / "verifier" / "reward.json").write_text('{"x": 1}', encoding="utf-8")  # planted, agent side

    def verifier(root):
        assert (root / "workspace" / "answer.txt").read_text(encoding="utf-8") == "42"
        assert not (root / "usr" / "bin" / "node").exists()  # the verifier never sees the agent's binaries
        out = root / "logs" / "verifier"
        out.mkdir(parents=True, exist_ok=True)
        (out / "reward.json").write_text('{"x": 0}', encoding="utf-8")
        return Result()

    FreshRootfs.stopped.clear()
    out = _verify(_fresh_env(tmp_path, verifier))
    assert out["harbor_verify_mode"] == "fresh" and json.loads(out["swe_tests"])[ALL_TEST] == "failed"
    assert FreshRootfs.stopped == [str(tmp_path / "verifier")]  # the verifier container is always deleted


def test_fresh_mode_infra_failure_still_deletes_the_verifier(tmp_path):
    FreshRootfs.stopped.clear()
    with pytest.raises(EnvInfraError):
        _verify(_fresh_env(tmp_path, SandboxError("boom")))
    assert FreshRootfs.stopped == [str(tmp_path / "verifier")]


@pytest.mark.parametrize("command", ["", "   ", "\n"])
def test_empty_model_command_is_a_failed_command_not_a_crash(tmp_path, command):
    env = _env(tmp_path, lambda root: Result())
    asyncio.run(env.start())
    out = asyncio.run(env.execute(command))
    assert out.exit_code == 2 and "empty command" in out.stderr and not out.timed_out


def test_build_context_scrubs_stray_hidden_test_copies(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mht", Path(__file__).resolve().parents[2] / "examples/agentic-rl-GB200/scripts/make_harbor_tasks.py")
    mht = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mht)
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("echo hi\n", encoding="utf-8")
    (task / "environment" / "Dockerfile").write_text("FROM x\nWORKDIR /workspace\nCOPY . /workspace\n", encoding="utf-8")
    df = (mht.build_context(task, tmp_path / "ctx") / "Dockerfile").read_text(encoding="utf-8").splitlines()
    copy_at = df.index(f"COPY {mht.HIDDEN_DIR}/ /tests/")
    assert df[copy_at + 1].startswith("RUN find / -xdev -path /tests -prune") and mht.HIDDEN_DIR in df[copy_at + 1]


def test_fresh_mode_refuses_root_workdir(tmp_path):
    env = _fresh_env(tmp_path, lambda root: Result())
    env._config = type(env._config)(**{**env._config.__dict__, "workdir": "/"}) if hasattr(env._config, "__dict__") else env._config
    if env._config.workdir == "/":
        out = env._verify_sync()
        assert out["swe_status"] == "error" and "non-root workdir" in out["swe_error"]
