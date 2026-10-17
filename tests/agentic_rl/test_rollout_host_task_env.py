"""RolloutHost's optional task-aware backend and verifiable-environment hooks.

A backend with ``create_for_task`` gets each episode's task; an environment with ``verify`` is
checked after the episode, and its result reaches the reward through the outcome's metadata.
Helpers below are copied from test_engine_recovery.py (themselves from test_rollout_host.py).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from foundationscale.agentic_rl import markup
from foundationscale.agentic_rl.envs.base import (
    EnvInfraError,
    EnvSpec,
    TaskAwareEnvBackend,
    VerifiableEnvironment,
    get_env_backend,
)
from foundationscale.agentic_rl.harness.base import EpisodeBudget, Generation, SamplingParams
from foundationscale.agentic_rl.harness.native_tool_loop import NativeToolLoop
from foundationscale.agentic_rl.rewards.base import RewardVerdict
from foundationscale.agentic_rl.rollout_host import RolloutHost
from foundationscale.agentic_rl.tasks import TaskSource
from foundationscale.agentic_rl.tools import QwenXmlToolCallParser, default_tools


class ToyTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def render(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None,
        add_generation_prompt: bool,
    ) -> list[int]:
        parts: list[str] = []
        if tools:
            parts.append("<tools>")
        for message in messages:
            parts.append(f"<{message['role']}>")
            parts.extend(str(message["content"]).split())
        if add_generation_prompt:
            parts.append("<assistant>")
        return [self._id(part) for part in parts]

    def decode(self, ids: Sequence[int]) -> str:
        inverse = {value: key for key, value in self._vocab.items()}
        return " ".join(inverse[i] for i in ids)


def _bash_call_text(command: str) -> str:
    return (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "bash"
        + markup.TAG_CLOSE
        + markup.PARAMETER_OPEN_PREFIX
        + "command"
        + markup.TAG_CLOSE
        + command
        + markup.PARAMETER_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )


def _submit_call_text(answer: str) -> str:
    return (
        markup.TOOL_CALL_OPEN
        + markup.FUNCTION_OPEN_PREFIX
        + "submit"
        + markup.TAG_CLOSE
        + markup.PARAMETER_OPEN_PREFIX
        + "answer"
        + markup.TAG_CLOSE
        + answer
        + markup.PARAMETER_CLOSE
        + markup.FUNCTION_CLOSE
        + markup.TOOL_CALL_CLOSE
    )


_TOKEN_COUNTER = [50_000]


def _generation(text: str, *, finish_reason: str = "stop") -> Generation:
    count = max(1, len(text.split()))
    start = _TOKEN_COUNTER[0]
    _TOKEN_COUNTER[0] += count
    return Generation(
        token_ids=tuple(range(start, start + count)),
        logprobs=None,
        finish_reason=finish_reason,  # type: ignore[arg-type]
        text=text,
    )


class SubmitClient:
    """Always immediately submits a fixed answer."""

    def __init__(self, answer: str = "the answer") -> None:
        self._answer = answer

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        return _generation(_submit_call_text(self._answer))


# ---------------------------------------------------------------------------
# Shared construction helpers (verbatim from test_rollout_host.py; `fleet` is
# injected through `_host`'s existing `**overrides`).
# ---------------------------------------------------------------------------


class ToyReward:
    """Scores submitted_answer by its length; abstains (measured 0.0) when absent."""

    async def score(self, task: Any, outcome: Any) -> RewardVerdict:
        if outcome.submitted_answer is None:
            return RewardVerdict(0.0, "no_answer")
        return RewardVerdict(float(len(outcome.submitted_answer)), None)


def _write_tasks(tmp_path: Path, uids: list[str]) -> str:
    path = tmp_path / "tasks.jsonl"
    rows = [
        {"uid": uid, "messages": [{"role": "user", "content": f"please help {uid}"}]}
        for uid in uids
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _harness() -> NativeToolLoop:
    return NativeToolLoop(
        tools=default_tools(),
        parser=QwenXmlToolCallParser(),
        tokenizer=ToyTokenizer(),
        sampling=SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=64,
        ),
    )


def _env_spec(**overrides: Any) -> EnvSpec:
    fields: dict[str, Any] = {"backend": "local", "env": {}, "allow_unsandboxed": True}
    fields.update(overrides)
    return EnvSpec(**fields)


def _budget() -> EpisodeBudget:
    return EpisodeBudget(step_limit=5, max_response_tokens=4096, max_observation_chars=4096)


def _host(tmp_path: Path, *, client_factory: Any, uids: list[str], **overrides: Any) -> RolloutHost:
    fields: dict[str, Any] = {
        "tasks": TaskSource(path=_write_tasks(tmp_path, uids), seed=0),
        "tasks_per_step": 1,
        "group_size": 2,
        "harness": _harness(),
        "env_backend": get_env_backend("local"),
        "env_spec": _env_spec(),
        "reward": ToyReward(),
        "budget": _budget(),
        "max_concurrency": 4,
        "client_factory": client_factory,
    }
    fields.update(overrides)
    return RolloutHost(**fields)


# ---------------------------------------------------------------------------
# A fake fleet whose clients hang together: shared mutable "mode" state, first
# gather times out, `start_all` cures the mode (unless told not to).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fakes for the two hooks.
# ---------------------------------------------------------------------------


class VerifyingEnv:
    """Wraps a real local environment and adds ``verify`` (records that it ran)."""

    def __init__(self, inner: Any, uid: str, result: Mapping[str, str] | Exception) -> None:
        self._inner, self.uid, self._result, self.verified = inner, uid, result, 0

    async def start(self) -> None:
        await self._inner.start()

    async def execute(self, command: str, *, timeout_s: float | None = None) -> Any:
        return await self._inner.execute(command, timeout_s=timeout_s)

    async def close(self) -> None:
        await self._inner.close()

    @property
    def workdir(self) -> str:
        return self._inner.workdir

    @property
    def episode_timeout_s(self) -> float:
        return self._inner.episode_timeout_s

    async def verify(self) -> Mapping[str, str]:
        self.verified += 1
        if isinstance(self._result, Exception):
            raise self._result
        return {**self._result, "verified_uid": self.uid}


class TaskBackend:
    """A task-aware backend over the local backend; remembers the tasks it was given."""

    def __init__(self, result: Mapping[str, str] | Exception) -> None:
        self._local = get_env_backend("local")
        self._result = result
        self.tasks: list[str] = []
        self.envs: list[VerifyingEnv] = []

    def capabilities(self) -> Any:
        return self._local.capabilities()

    def create(self, spec: EnvSpec, *, instance_id: str) -> Any:  # pragma: no cover - not used
        raise AssertionError("RolloutHost must call create_for_task on a task-aware backend")

    def create_for_task(self, spec: EnvSpec, task: Any, *, instance_id: str) -> Any:
        self.tasks.append(task.uid)
        env = VerifyingEnv(
            self._local.create(spec, instance_id=instance_id), task.uid, self._result
        )
        self.envs.append(env)
        return env


class MetadataReward:
    """Scores 1.0 iff verify() reported pass for this very task."""

    async def score(self, task: Any, outcome: Any) -> RewardVerdict:
        ok = outcome.metadata.get("tests") == "pass"
        same_task = outcome.metadata.get("verified_uid") == task.uid
        return RewardVerdict(1.0 if ok and same_task else 0.0, None)


def test_the_fakes_satisfy_the_protocols() -> None:
    backend = TaskBackend({"tests": "pass"})
    assert isinstance(backend, TaskAwareEnvBackend)
    assert isinstance(VerifyingEnv(None, "u", {}), VerifiableEnvironment)
    assert not isinstance(get_env_backend("local"), TaskAwareEnvBackend)


def test_task_aware_backend_gets_each_task_and_verify_reaches_the_reward(tmp_path: Path) -> None:
    backend = TaskBackend({"tests": "pass"})
    host = _host(
        tmp_path, client_factory=lambda i: SubmitClient(), uids=["taskA"],
        env_backend=backend, reward=MetadataReward(),
    )  # fmt: skip
    batch = host.rollout(0)
    assert backend.tasks == ["taskA", "taskA"]  # group_size 2, one create_for_task per attempt
    assert [env.verified for env in backend.envs] == [1, 1]
    assert list(batch.columns["reward"]) == [1.0, 1.0]


def test_a_failing_verification_scores_zero(tmp_path: Path) -> None:
    backend = TaskBackend({"tests": "fail"})
    host = _host(
        tmp_path, client_factory=lambda i: SubmitClient(), uids=["taskA"],
        env_backend=backend, reward=MetadataReward(),
    )  # fmt: skip
    assert list(host.rollout(0).columns["reward"]) == [0.0, 0.0]


def test_a_verification_infra_error_is_an_abstention_not_a_zero(tmp_path: Path) -> None:
    backend = TaskBackend(EnvInfraError("test_runner_down", "the test runner did not start"))
    host = _host(
        tmp_path, client_factory=lambda i: SubmitClient(), uids=["taskA"],
        env_backend=backend, reward=MetadataReward(), gates=False,
    )  # fmt: skip
    batch = host.rollout(0)
    assert all(r is None for r in batch.columns["reward"])


def test_plain_environments_are_never_verified(tmp_path: Path) -> None:
    host = _host(tmp_path, client_factory=lambda i: SubmitClient(), uids=["taskA"])
    assert all(r is not None for r in host.rollout(0).columns["reward"])
