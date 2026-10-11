"""End-to-end tests for ``rollout_host.RolloutHost``: a scripted ``GenerationClient``
(injected via ``client_factory``), the REAL ``NativeToolLoop`` harness with a toy
tokenizer, the REAL ``local`` ``EnvBackend`` (``allow_unsandboxed=True``), and a
toy ``RewardFn`` -- no engine fleet, no real network.

WHAT IS CLAIMED: ``rollout(step)`` returns an ``ExperienceBatch`` carrying
exactly ``contracts.DECLARED_COLUMNS``, one row per ``tasks_per_step *
group_size`` attempt, ``session_id`` values ``f"{uid}#{k}"``, and
``policy_version_min == policy_version_max == step``; a MID-EPISODE
``EngineInfraError`` (caught internally by ``NativeToolLoop``, per its own
edit in this slice) yields a row with ``termination=INFRA``, ``reward=None``
and ``abstention_reason="infra:engine_<kind>"`` WITHOUT the toy reward ever
being called; an environment that fails to even START (``EnvInfraError`` from
``env.start()``, before ``NativeToolLoop.run_episode`` is ever called) is
converted by ``RolloutHost``'s own outer catch into a ZERO-TURN
``termination=INFRA`` row (``prompt_turns=()``, ``turns=()``) with
``abstention_reason="infra:EnvInfraError"``;
``max_concurrency`` actually bounds the number of in-flight episodes;
construction refuses a bad field; ``publish`` calls ``save_fn`` on every rank
but pushes to the fleet (observed via the prune side effect) only on rank 0.

WHAT IS NOT CLAIMED: no SGLang engine of any kind is started for the rollout
tests (``client_factory`` injects a pure-Python scripted client); the
``publish`` tests DO start fake-HTTP-server subprocesses (via
``_fake_sglang_fixtures``, shared with ``test_weight_sync.py``)
because ``DiskWeightSync`` requires a real ``EngineFleet``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fake_sglang_fixtures import free_port, make_server_spec, write_fake_server

from foundationscale.agentic_rl import markup
from foundationscale.agentic_rl.contracts import DECLARED_COLUMNS, Termination
from foundationscale.agentic_rl.engines import EngineInfraError
from foundationscale.agentic_rl.engines.fleet import EngineFleet, SGLangServer
from foundationscale.agentic_rl.envs.base import EnvSpec, get_env_backend
from foundationscale.agentic_rl.harness.base import EpisodeBudget, Generation, SamplingParams
from foundationscale.agentic_rl.harness.native_tool_loop import NativeToolLoop
from foundationscale.agentic_rl.rewards.base import RewardVerdict
from foundationscale.agentic_rl.rollout_host import RolloutHost, RolloutHostRefusal
from foundationscale.agentic_rl.tasks import TaskSource
from foundationscale.agentic_rl.tools import QwenXmlToolCallParser, default_tools
from foundationscale.agentic_rl.weight_sync import DiskWeightSync

# ---------------------------------------------------------------------------
# A toy tokenizer, toy markup-level call builders, a scripted client family,
# and a toy reward -- all deliberately tiny, mirroring test_native_tool_loop.py.
# ---------------------------------------------------------------------------


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


class ExplodingAfterOneClient:
    """First call succeeds with a harmless bash call; every call after that
    raises EngineInfraError -- so trajectory_turns is non-empty by the time the
    infra error fires (see test_native_tool_loop.py's identically-shaped fix).
    """

    def __init__(self, kind: str = "transport", message: str = "connection reset") -> None:
        self._calls = 0
        self._kind = kind
        self._message = message

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        self._calls += 1
        if self._calls == 1:
            return _generation(_bash_call_text("echo hi"))
        raise EngineInfraError(self._kind, self._message)


class TrackedSubmitClient:
    """Tracks the peak number of concurrently in-flight generate() calls."""

    def __init__(self, answer: str = "ok", sleep_s: float = 0.03) -> None:
        self.current = 0
        self.peak = 0
        self._answer = answer
        self._sleep_s = sleep_s

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        import asyncio

        self.current += 1
        self.peak = max(self.peak, self.current)
        await asyncio.sleep(self._sleep_s)
        self.current -= 1
        return _generation(_submit_call_text(self._answer))


class ToyReward:
    """Scores submitted_answer by its length; abstains (measured 0.0) when absent."""

    async def score(self, task: Any, outcome: Any) -> RewardVerdict:
        if outcome.submitted_answer is None:
            return RewardVerdict(0.0, "no_answer")
        return RewardVerdict(float(len(outcome.submitted_answer)), None)


# ---------------------------------------------------------------------------
# Shared construction helpers
# ---------------------------------------------------------------------------


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
# Happy path + batch shape
# ---------------------------------------------------------------------------


def test_rollout_returns_declared_columns_one_row_per_attempt(tmp_path: Path) -> None:
    host = _host(
        tmp_path,
        uids=["taskA"],
        group_size=3,
        client_factory=lambda index: SubmitClient("hello world"),
    )
    batch = host.rollout(0)
    assert tuple(batch.columns) == DECLARED_COLUMNS
    assert len(batch) == 3
    assert set(batch.columns["session_id"]) == {"taskA#0", "taskA#1", "taskA#2"}
    assert all(uid == "taskA" for uid in batch.columns["uid"])
    assert all(t is Termination.STOP for t in batch.columns["termination"])
    assert all(r == float(len("hello world")) for r in batch.columns["reward"])
    assert all(a is None for a in batch.columns["abstention_reason"])
    assert batch.columns["policy_version_min"] == (0, 0, 0)
    assert batch.columns["policy_version_max"] == (0, 0, 0)


def test_rollout_async_is_the_same_as_rollout(tmp_path: Path) -> None:
    import asyncio

    host = _host(tmp_path, uids=["taskA"], client_factory=lambda index: SubmitClient("x"))
    batch = asyncio.run(host.rollout_async(0))
    assert len(batch) == 2


def test_rollout_against_a_real_fleet_round_robins_clients(tmp_path: Path) -> None:
    # Exercises _client_for's `fleet` branch (every other test injects
    # client_factory instead): a REAL EngineFleet of fake HTTP server
    # subprocesses, never a real SGLang engine.
    fake_module = write_fake_server(tmp_path)
    specs = [make_server_spec(tmp_path, fake_module, port=free_port()) for _ in range(2)]
    servers = tuple(SGLangServer(spec) for spec in specs)
    fleet = EngineFleet(servers=servers)
    fleet.start_all()
    try:
        host = _host(tmp_path, uids=["taskA"], group_size=2, client_factory=None, fleet=fleet)
        batch = host.rollout(0)
        assert len(batch) == 2
        assert all(t is Termination.STOP for t in batch.columns["termination"])
        # The fake server's /generate echoes its currently-"loaded" model path
        # (see _fake_sglang_fixtures.FAKE_SERVER_SOURCE) -- "/fake/model" here,
        # since nothing has pushed a reload yet.
        expected_text = "ok loaded:/fake/model"
        assert all(r == float(len(expected_text)) for r in batch.columns["reward"])
    finally:
        fleet.stop_all()


# ---------------------------------------------------------------------------
# ROLLOUT agentic gate sweep: a blocking report raises RolloutHostRefusal
# ---------------------------------------------------------------------------


def test_rollout_raises_when_the_rollout_gate_sweep_blocks_on_infra_rate(
    tmp_path: Path,
) -> None:
    # declared_max_infra_rate=0.0 with one infra'd attempt out of two gives a
    # measured infra rate (1/2) that exceeds the declared bound, so
    # RolloutAbstentionGate FAILs and the ROLLOUT gate sweep (gates=True by
    # default) must turn that into a RolloutHostRefusal naming both the
    # infra count and the total row count, never a silently-returned batch.
    def client_factory(index: int) -> Any:
        return ExplodingAfterOneClient() if index == 1 else SubmitClient("ok")

    host = _host(
        tmp_path,
        uids=["taskA"],
        group_size=2,
        client_factory=client_factory,
        declared_max_infra_rate=0.0,
    )
    with pytest.raises(RolloutHostRefusal, match="ROLLOUT gates blocked") as excinfo:
        host.rollout(0)
    message = str(excinfo.value)
    assert "step 0" in message
    # RolloutAbstentionGate's own FAIL detail names both counts: 1 infra'd row
    # out of 2 total rows.
    assert "1/2" in message


# ---------------------------------------------------------------------------
# INFRA conversion: mid-episode EngineInfraError (NativeToolLoop's own catch)
# ---------------------------------------------------------------------------


def test_mid_episode_engine_infra_error_yields_infra_row_without_reward_call(
    tmp_path: Path,
) -> None:
    scored: list[str] = []

    class RecordingReward:
        async def score(self, task: Any, outcome: Any) -> RewardVerdict:
            scored.append(outcome.submitted_answer or "")
            return RewardVerdict(1.0, None)

    def client_factory(index: int) -> Any:
        return ExplodingAfterOneClient() if index == 1 else SubmitClient("ok")

    host = _host(
        tmp_path,
        uids=["taskA"],
        group_size=2,
        client_factory=client_factory,
        reward=RecordingReward(),
    )
    batch = host.rollout(0)
    assert len(batch) == 2
    rows = dict(zip(batch.columns["session_id"], range(len(batch)), strict=True))
    stop_row = rows["taskA#0"]
    infra_row = rows["taskA#1"]
    assert batch.columns["termination"][stop_row] is Termination.STOP
    assert batch.columns["reward"][stop_row] == 1.0
    assert batch.columns["termination"][infra_row] is Termination.INFRA
    assert batch.columns["reward"][infra_row] is None
    assert batch.columns["abstention_reason"][infra_row] == "infra:engine_transport"
    # The toy reward must never be asked to score the infra'd episode.
    assert scored == ["ok"]


def test_reward_scoring_infra_error_reuses_the_real_outcome_turns(tmp_path: Path) -> None:
    # Unlike the mid-episode client failure above, run_episode() succeeds here and
    # returns a real EpisodeOutcome; the infra error comes from reward.score()
    # itself, so RolloutHost must reuse that outcome's REAL prompt_turns/turns
    # rather than the no-outcome synthetic fallback.
    class ExplodingReward:
        async def score(self, task: Any, outcome: Any) -> RewardVerdict:
            raise EngineInfraError("transport", "scorer unreachable")

    host = _host(
        tmp_path,
        uids=["taskA"],
        group_size=2,
        client_factory=lambda index: SubmitClient("ok"),
        reward=ExplodingReward(),
    )
    batch = host.rollout(0)
    assert len(batch) == 2
    assert all(t is Termination.INFRA for t in batch.columns["termination"])
    assert all(r is None for r in batch.columns["reward"])
    assert all(a == "infra:EngineInfraError" for a in batch.columns["abstention_reason"])
    # The real (non-synthetic) response tokens from the successful submit call.
    assert all(len(ids) > 0 for ids in batch.columns["response_ids"])


# ---------------------------------------------------------------------------
# INFRA conversion: environment fails to even start (RolloutHost's own catch)
# ---------------------------------------------------------------------------


def test_env_start_failure_yields_a_zero_turn_infra_row(tmp_path: Path) -> None:
    host = _host(
        tmp_path,
        uids=["taskA"],
        group_size=2,
        client_factory=lambda index: SubmitClient("unused"),
        env_spec=_env_spec(setup_commands=("false",)),
    )
    batch = host.rollout(0)
    assert len(batch) == 2
    assert all(t is Termination.INFRA for t in batch.columns["termination"])
    assert all(r is None for r in batch.columns["reward"])
    assert all(a == "infra:EnvInfraError" for a in batch.columns["abstention_reason"])
    # Nothing was ever rendered (the environment failed before NativeToolLoop
    # ever ran), so there is nothing real to report on either side: both the
    # prompt and the response are genuinely empty, never a byte-encoded
    # placeholder standing in for text nothing tokenised.
    assert batch.columns["prompt_token_ids"][0] == ()
    assert batch.columns["response_ids"][0] == ()
    assert batch.columns["loss_mask"][0] == ()


# ---------------------------------------------------------------------------
# A MEASURED verdict carrying an informational reason (never an abstention)
# ---------------------------------------------------------------------------


class NoteReward:
    """Always returns a MEASURED zero with an informational (non-abstention) note."""

    async def score(self, task: Any, outcome: Any) -> RewardVerdict:
        return RewardVerdict(0.0, "no_abc")


def test_measured_zero_with_informational_note_does_not_abstain(tmp_path: Path) -> None:
    # Regression: RewardVerdict(0.0, "no_abc") is a MEASURED verdict (value is not
    # None) carrying an informational note, never an abstention. Before the fix,
    # RolloutHost copied verdict.reason straight into abstention_reason even
    # though reward_value was not None, so Trajectory.__post_init__ raised
    # TrajectoryRefusal ("a scored attempt has nothing to abstain about") and
    # rollout() crashed.
    host = _host(
        tmp_path,
        uids=["taskA"],
        group_size=2,
        client_factory=lambda index: SubmitClient("ok"),
        reward=NoteReward(),
    )
    batch = host.rollout(0)  # must not raise RolloutHostRefusal/TrajectoryRefusal
    assert len(batch) == 2
    assert all(t is Termination.STOP for t in batch.columns["termination"])
    assert all(r == 0.0 for r in batch.columns["reward"])
    assert all(a is None for a in batch.columns["abstention_reason"])


def test_measured_zero_with_informational_note_keeps_the_note_in_trajectory_metadata(
    tmp_path: Path,
) -> None:
    # ExperienceBatch never exports Trajectory.metadata (see contracts.Trajectory's
    # own docstring), so the note's survival is only observable by calling the
    # (private) per-episode builder directly and inspecting the Trajectory itself.
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=lambda index: SubmitClient("ok"),
        reward=NoteReward(),
    )
    task = host.tasks.batch(0, 1)[0]
    trajectory = asyncio.run(host._run_episode(task, "taskA#0", 0, 0, asyncio.Semaphore(1)))
    assert trajectory.reward == 0.0
    assert trajectory.abstention_reason is None
    assert trajectory.metadata.get("reward_note") == "no_abc"


# ---------------------------------------------------------------------------
# An UNMEASURED verdict (value=None) becomes the trajectory's abstention
# ---------------------------------------------------------------------------


class AbstainingReward:
    """Always returns an UNMEASURED verdict: value=None, reason names why."""

    async def score(self, task: Any, outcome: Any) -> RewardVerdict:
        return RewardVerdict(None, "policy_declined_to_answer")


def test_abstaining_verdict_value_none_becomes_the_abstention_reason(tmp_path: Path) -> None:
    # RewardVerdict(None, reason) is UNMEASURED: reward_value is None, so
    # RolloutHost must copy verdict.reason straight into abstention_reason
    # (never discard it), and reward itself must stay None on the resulting
    # trajectory/batch row.
    host = _host(
        tmp_path,
        uids=["taskA"],
        group_size=2,
        client_factory=lambda index: SubmitClient("ok"),
        reward=AbstainingReward(),
    )
    batch = host.rollout(0)
    assert len(batch) == 2
    assert all(t is Termination.STOP for t in batch.columns["termination"])
    assert all(r is None for r in batch.columns["reward"])
    assert all(a == "policy_declined_to_answer" for a in batch.columns["abstention_reason"])


# ---------------------------------------------------------------------------
# Concurrency bound
# ---------------------------------------------------------------------------


def test_max_concurrency_bounds_in_flight_episodes(tmp_path: Path) -> None:
    tracker = TrackedSubmitClient(sleep_s=0.03)
    host = _host(
        tmp_path,
        uids=["taskA"],
        group_size=4,
        client_factory=lambda index: tracker,
        max_concurrency=2,
    )
    batch = host.rollout(0)
    assert len(batch) == 4
    assert tracker.peak == 2


# ---------------------------------------------------------------------------
# Construction refusals
# ---------------------------------------------------------------------------


def test_rejects_neither_fleet_nor_client_factory(tmp_path: Path) -> None:
    with pytest.raises(RolloutHostRefusal):
        _host(tmp_path, uids=["taskA"], client_factory=None)


def test_rejects_both_fleet_and_client_factory(tmp_path: Path) -> None:
    # The refusal checks only `fleet is not None`, so any non-None sentinel proves
    # the point without constructing (and never using) a real EngineFleet.
    sentinel_fleet = object()
    with pytest.raises(RolloutHostRefusal):
        _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            fleet=sentinel_fleet,
        )


@pytest.mark.parametrize(
    "field,value", [("tasks_per_step", 0), ("group_size", 1), ("max_concurrency", 0)]
)
def test_rejects_bad_int_fields(tmp_path: Path, field: str, value: int) -> None:
    with pytest.raises(RolloutHostRefusal):
        _host(
            tmp_path, uids=["taskA"], client_factory=lambda index: SubmitClient(), **{field: value}
        )


def test_rejects_non_bool_group_by_harness(tmp_path: Path) -> None:
    with pytest.raises(RolloutHostRefusal):
        _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            group_by_harness=1,
        )


def test_rejects_bad_weight_sync_type(tmp_path: Path) -> None:
    with pytest.raises(RolloutHostRefusal):
        _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            weight_sync="not a sync",
        )


def test_rejects_bad_sharding(tmp_path: Path) -> None:
    with pytest.raises(RolloutHostRefusal, match="field 'sharding'") as excinfo:
        _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            sharding="tensor_parallel",
        )
    assert "'none'" in str(excinfo.value)
    assert "'ddp'" in str(excinfo.value)
    assert "'fsdp'" in str(excinfo.value)


def test_rejects_non_bool_gates(tmp_path: Path) -> None:
    with pytest.raises(RolloutHostRefusal, match="field 'gates'"):
        _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            gates=1,
        )


@pytest.mark.parametrize(
    "bad_rate",
    [
        1.5,  # out of [0.0, 1.0]
        -0.1,  # out of [0.0, 1.0]
        True,  # bool excluded even though isinstance(True, int)
        "0.5",  # not a real number
    ],
)
def test_rejects_bad_declared_max_infra_rate(tmp_path: Path, bad_rate: object) -> None:
    with pytest.raises(RolloutHostRefusal, match="field 'declared_max_infra_rate'"):
        _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            declared_max_infra_rate=bad_rate,
        )


def test_accepts_declared_max_infra_rate_boundaries(tmp_path: Path) -> None:
    # 0.0 and 1.0 are the inclusive boundary of the declared [0.0, 1.0] range
    # and must not raise -- only values strictly outside it are refused.
    for boundary in (0.0, 1.0):
        _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            declared_max_infra_rate=boundary,
        )


@pytest.mark.parametrize("bad_value", ["", 0, object()])
def test_rejects_bad_servable_base_model_dir(tmp_path: Path, bad_value: object) -> None:
    with pytest.raises(RolloutHostRefusal, match="field 'servable_base_model_dir'"):
        _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            servable_base_model_dir=bad_value,
        )


def test_accepts_none_and_a_non_empty_str_servable_base_model_dir(tmp_path: Path) -> None:
    _host(
        tmp_path,
        uids=["taskA"],
        client_factory=lambda index: SubmitClient(),
        servable_base_model_dir=None,
    )
    _host(
        tmp_path,
        uids=["taskA"],
        client_factory=lambda index: SubmitClient(),
        servable_base_model_dir="/some/base/model/dir",
    )


def test_rollout_rejects_bad_step(tmp_path: Path) -> None:
    host = _host(tmp_path, uids=["taskA"], client_factory=lambda index: SubmitClient())
    with pytest.raises(RolloutHostRefusal):
        host.rollout(-1)


def test_publish_rejects_bad_step(tmp_path: Path) -> None:
    host = _host(tmp_path, uids=["taskA"], client_factory=lambda index: SubmitClient())
    with pytest.raises(RolloutHostRefusal):
        host.publish(None, None, SimpleNamespace(rank=0), -1)


# ---------------------------------------------------------------------------
# publish(): no-op without weight_sync; save_fn on every rank; push only rank 0
# ---------------------------------------------------------------------------


def test_publish_is_a_noop_without_weight_sync(tmp_path: Path) -> None:
    host = _host(tmp_path, uids=["taskA"], client_factory=lambda index: SubmitClient())
    host.publish(None, None, SimpleNamespace(rank=0), 0)  # must not raise


def test_publish_calls_the_save_path_every_rank_but_pushes_only_rank_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # publish() builds its save around the REAL model/tokenizer/ctx it
    # receives (see rollout_host.py's own docstring on why save_fn cannot
    # serve this path), via rl.distributed.save_checkpoint -- faked here so
    # this test needs no real torch model.
    calls: list[tuple[str, object, object, int]] = []

    def fake_save_checkpoint(
        model: object, tokenizer: object, out_dir: str, ctx: object, *, sharding: str, step: int
    ) -> bool:
        calls.append((out_dir, model, tokenizer, step))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr("foundationscale.rl.distributed.save_checkpoint", fake_save_checkpoint)

    fake_module = write_fake_server(tmp_path)
    specs = [make_server_spec(tmp_path, fake_module, port=free_port()) for _ in range(2)]
    servers = tuple(SGLangServer(spec) for spec in specs)
    fleet = EngineFleet(servers=servers)
    fleet.start_all()
    try:
        publish_root = tmp_path / "publish"
        publish_root.mkdir()
        (publish_root / "step_000000").mkdir()
        (publish_root / "step_000001").mkdir()

        sync = DiskWeightSync(publish_root=str(publish_root), fleet=fleet, keep_last=1)
        host = _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            weight_sync=sync,
        )
        model, tokenizer = object(), object()

        # rank != 0: the save runs on EVERY rank (collective), but push (and
        # therefore prune) does not -- only rank 0 talks to the fleet.
        host.publish(model, tokenizer, SimpleNamespace(rank=1), step=5)
        assert calls == [(f"{publish_root}/step_000006", model, tokenizer, 6)]
        remaining = {p.name for p in publish_root.iterdir()}
        assert {"step_000000", "step_000001", "step_000006"} <= remaining

        # rank == 0: the save runs again, AND push runs -- observed via the
        # keep_last=1 prune leaving only the just-published directory.
        host.publish(model, tokenizer, SimpleNamespace(rank=0), step=6)
        assert calls == [
            (f"{publish_root}/step_000006", model, tokenizer, 6),
            (f"{publish_root}/step_000007", model, tokenizer, 7),
        ]
        remaining = {p.name for p in publish_root.iterdir()}
        assert remaining == {"step_000007"}
    finally:
        fleet.stop_all()


def test_publish_completes_for_serving_before_the_push_when_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # servable_base_model_dir=None (every other publish test here) means
    # complete_for_serving is never even imported -- this is the one test
    # exercising the opt-in path, and pinning its ORDER: the published
    # directory must be made servable BEFORE the fleet is asked to reload it.
    from foundationscale.rl.weightsync import SyncReport

    order: list[str] = []
    monkeypatch.setattr(
        "foundationscale.rl.distributed.save_checkpoint",
        lambda *a, **k: Path(a[2]).mkdir(parents=True, exist_ok=True) or True,
    )

    def fake_complete_for_serving(publish_dir: str, base_model_dir: str, *, cache_dir: str) -> Any:
        order.append("complete_for_serving")
        assert base_model_dir == "/fake/base/model"
        assert cache_dir == f"{tmp_path / 'publish'}/.serving_cache"
        from foundationscale.agentic_rl.servable import ServableReport

        return ServableReport(
            status="completed", extras_keys=3, extras_bytes=12, copied_files=("config.json",)
        )

    monkeypatch.setattr(
        "foundationscale.agentic_rl.servable.complete_for_serving", fake_complete_for_serving
    )

    def fake_push(self: DiskWeightSync, path: str) -> SyncReport:
        order.append("push")
        return SyncReport(
            transport="disk",
            offered=("server_0",),
            transferred=("server_0",),
            skipped=(),
            failed_ranks=(),
            bytes_moved=None,
            seconds=0.01,
            is_stale=False,
        )

    monkeypatch.setattr(DiskWeightSync, "push", fake_push)

    fake_module = write_fake_server(tmp_path)
    spec = make_server_spec(tmp_path, fake_module, port=free_port())
    fleet = EngineFleet(servers=(SGLangServer(spec),))
    publish_root = tmp_path / "publish"
    publish_root.mkdir()
    sync = DiskWeightSync(publish_root=str(publish_root), fleet=fleet)
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=lambda index: SubmitClient(),
        weight_sync=sync,
        servable_base_model_dir="/fake/base/model",
    )
    host.publish(object(), object(), SimpleNamespace(rank=0), step=0)
    assert order == ["complete_for_serving", "push"]


def test_publish_never_calls_complete_for_serving_when_not_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foundationscale.rl.weightsync import SyncReport

    monkeypatch.setattr(
        "foundationscale.rl.distributed.save_checkpoint",
        lambda *a, **k: Path(a[2]).mkdir(parents=True, exist_ok=True) or True,
    )

    def boom(*args: object, **kwargs: object) -> Any:
        raise AssertionError("complete_for_serving must not be called when not declared")

    monkeypatch.setattr("foundationscale.agentic_rl.servable.complete_for_serving", boom)

    def fake_push(self: DiskWeightSync, path: str) -> SyncReport:
        return SyncReport(
            transport="disk",
            offered=("server_0",),
            transferred=("server_0",),
            skipped=(),
            failed_ranks=(),
            bytes_moved=None,
            seconds=0.01,
            is_stale=False,
        )

    monkeypatch.setattr(DiskWeightSync, "push", fake_push)

    fake_module = write_fake_server(tmp_path)
    spec = make_server_spec(tmp_path, fake_module, port=free_port())
    fleet = EngineFleet(servers=(SGLangServer(spec),))
    publish_root = tmp_path / "publish"
    publish_root.mkdir()
    sync = DiskWeightSync(publish_root=str(publish_root), fleet=fleet)
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=lambda index: SubmitClient(),
        weight_sync=sync,
    )
    host.publish(object(), object(), SimpleNamespace(rank=0), step=0)  # must not raise


def test_publish_raises_when_the_push_is_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A push that does not fully succeed must be LOUD, never silently
    # swallowed: continuing would serve a stale policy on every subsequent
    # rollout without saying so (see rollout_host.py's own docstring).
    monkeypatch.setattr(
        "foundationscale.rl.distributed.save_checkpoint",
        lambda *a, **k: Path(a[2]).mkdir(parents=True, exist_ok=True) or True,
    )
    fake_module = write_fake_server(tmp_path)
    specs = [make_server_spec(tmp_path, fake_module, port=free_port()) for _ in range(2)]
    servers = tuple(SGLangServer(spec) for spec in specs)
    fleet = EngineFleet(servers=servers)
    fleet.start_all()
    try:
        publish_root = tmp_path / "publish"
        publish_root.mkdir()
        sync = DiskWeightSync(publish_root=str(publish_root), fleet=fleet)
        host = _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient(),
            weight_sync=sync,
        )
        # The fake server fails any model_path containing "FAIL" (see
        # _fake_sglang_fixtures.FAKE_SERVER_SOURCE); path_for's own step
        # naming never produces that, so the class method is patched instead
        # (DiskWeightSync is frozen -- an INSTANCE attribute cannot be set).
        monkeypatch.setattr(
            DiskWeightSync, "path_for", lambda self, step: f"{publish_root}/FAIL_step"
        )
        with pytest.raises(RolloutHostRefusal, match="NOT complete"):
            host.publish(object(), object(), SimpleNamespace(rank=0), step=0)
    finally:
        fleet.stop_all()


def test_publish_raises_when_the_weight_sync_gate_sweep_blocks_on_unmeasured_staleness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A push whose own SyncReport never measured is_stale (offered > 0 but
    # is_stale=None) must FAIL StalenessGate: "a weight sync was offered ...
    # but is_stale was never measured" cannot be reported healthy. gates=True
    # by default, so publish() must turn that into a RolloutHostRefusal naming
    # the WEIGHT_SYNC sweep, never silently accept the unmeasured push.
    from foundationscale.rl.weightsync import SyncReport

    monkeypatch.setattr(
        "foundationscale.rl.distributed.save_checkpoint",
        lambda *a, **k: Path(a[2]).mkdir(parents=True, exist_ok=True) or True,
    )
    fake_module = write_fake_server(tmp_path)
    spec = make_server_spec(tmp_path, fake_module, port=free_port())
    # Never started: push() is replaced below, so no real HTTP call is ever
    # made -- this fleet exists only to satisfy DiskWeightSync's own
    # isinstance(fleet, EngineFleet) construction check.
    fleet = EngineFleet(servers=(SGLangServer(spec),))

    def fake_push(self: DiskWeightSync, path: str) -> SyncReport:
        return SyncReport(
            transport="disk",
            offered=("server_0",),
            transferred=("server_0",),
            skipped=(),
            failed_ranks=(),
            bytes_moved=None,
            seconds=0.01,
            is_stale=None,  # UNMEASURED: offered > 0 but staleness never measured
        )

    monkeypatch.setattr(DiskWeightSync, "push", fake_push)
    publish_root = tmp_path / "publish"
    publish_root.mkdir()
    sync = DiskWeightSync(publish_root=str(publish_root), fleet=fleet)
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=lambda index: SubmitClient(),
        weight_sync=sync,
    )
    with pytest.raises(RolloutHostRefusal, match="WEIGHT_SYNC gates blocked") as excinfo:
        host.publish(object(), object(), SimpleNamespace(rank=0), step=0)
    message = str(excinfo.value)
    assert "step 0" in message
    # StalenessGate's own FAIL detail names the offered-server count.
    assert "1 server(s)" in message


def test_second_step_rollout_is_served_by_the_fleet_after_a_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end, a REAL EngineFleet of one fake HTTP server, through
    # RolloutHost.rollout() + .publish() exactly as RLTrainer.run() drives
    # them: step 2's rollout must be served by the RELOADED checkpoint, not
    # the one the fleet started with -- observed via the fake server's own
    # echo of its currently-"loaded" model path (see
    # _fake_sglang_fixtures.FAKE_SERVER_SOURCE), not merely "nothing raised".
    monkeypatch.setattr(
        "foundationscale.rl.distributed.save_checkpoint",
        lambda *a, **k: Path(a[2]).mkdir(parents=True, exist_ok=True) or True,
    )
    fake_module = write_fake_server(tmp_path)
    spec = make_server_spec(tmp_path, fake_module, port=free_port())
    fleet = EngineFleet(servers=(SGLangServer(spec),))
    fleet.start_all()
    try:
        publish_root = tmp_path / "publish"
        publish_root.mkdir()
        sync = DiskWeightSync(publish_root=str(publish_root), fleet=fleet)
        host = _host(
            tmp_path,
            uids=["taskA"],
            group_size=2,
            client_factory=None,
            fleet=fleet,
            weight_sync=sync,
        )

        # Step 0's rollout is served by the fleet's INITIAL checkpoint.
        batch0 = host.rollout(0)
        initial_len = len("ok loaded:/fake/model")
        assert all(r == float(initial_len) for r in batch0.columns["reward"])

        # RLTrainer.run() calls publish(model, tokenizer, ctx, step) after the
        # optimizer step; rank 0 saves (faked above), then pushes the reload.
        host.publish(object(), object(), SimpleNamespace(rank=0), step=0)

        # Step 1 ("step 2" in 1-indexed training-step terms) is served by the
        # RELOADED checkpoint.
        batch1 = host.rollout(1)
        reloaded_len = len(f"ok loaded:{publish_root}/step_000001")
        assert reloaded_len != initial_len
        assert all(r == float(reloaded_len) for r in batch1.columns["reward"])
    finally:
        fleet.stop_all()


def test_rollout_summary_counts_abstentions_and_never_averages_none() -> None:
    import json

    from foundationscale.agentic_rl.rollout_host import rollout_summary

    line = rollout_summary(3, [])
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["trajectories"] == 0
    assert payload["reward_mean"] is None
    assert payload["abstained"] == 0


def test_rollout_feeds_the_stability_guard_once_per_step(tmp_path: Path) -> None:
    """With a guard configured, each rollout updates it and leaves a verdict for publish."""
    from foundationscale.agentic_rl.stability import GuardConfig, StabilityGuard

    guard = StabilityGuard(GuardConfig())
    host = _host(
        tmp_path,
        uids=["taskA"],
        group_size=2,
        client_factory=lambda index: SubmitClient("hello world"),
        guard=guard,
    )
    host.rollout(0)
    assert guard.last_verdict is not None
    assert guard.last_verdict.step == 0


def test_publish_applies_the_stability_guard_after_a_complete_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a complete push, a stop-action alarm raises with the checkpoint published."""
    from foundationscale.agentic_rl.stability import (
        CollapseDetected,
        GuardConfig,
        RolloutStats,
        StabilityGuard,
    )
    from foundationscale.rl.weightsync import SyncReport

    monkeypatch.setattr(
        "foundationscale.rl.distributed.save_checkpoint",
        lambda *a, **k: Path(a[2]).mkdir(parents=True, exist_ok=True) or True,
    )

    def fake_push(self: DiskWeightSync, path: str) -> SyncReport:
        return SyncReport(
            transport="disk",
            offered=("server_0",),
            transferred=("server_0",),
            skipped=(),
            failed_ranks=(),
            bytes_moved=None,
            seconds=0.01,
            is_stale=False,
        )

    monkeypatch.setattr(DiskWeightSync, "push", fake_push)
    fake_module = write_fake_server(tmp_path)
    spec = make_server_spec(tmp_path, fake_module, port=free_port())
    fleet = EngineFleet(servers=(SGLangServer(spec),))
    publish_root = tmp_path / "publish"
    publish_root.mkdir()
    guard = StabilityGuard(GuardConfig(ema_alpha=1.0, min_steps=2, action="stop"))
    for step, mean in ((0, 0.6), (1, 0.6), (2, 0.05)):
        guard.update(RolloutStats(step, 4, 0, mean, 0.0))
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=lambda index: SubmitClient(),
        weight_sync=DiskWeightSync(publish_root=str(publish_root), fleet=fleet),
        servable_base_model_dir=None,
        guard=guard,
    )
    with pytest.raises(CollapseDetected):
        host.publish(object(), object(), SimpleNamespace(rank=0), step=2)
