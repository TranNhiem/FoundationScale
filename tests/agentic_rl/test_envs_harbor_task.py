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
        "pass_to_pass": "[]"}


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
