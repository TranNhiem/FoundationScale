"""ENGINE hang recovery tests for ``rollout_host.RolloutHost``: a fake fleet
(``FakeFleet``) serving mode-driven clients, the REAL ``NativeToolLoop`` harness with a toy
tokenizer, the REAL local ``EnvBackend`` (``allow_unsandboxed=True``) and a toy reward -- no
real engine is EVER started (test (f)'s ``EngineFleet``, required by ``DiskWeightSync``'s
``isinstance`` check, is built from the shared fake-HTTP-server fixtures but never started:
its ``push`` is monkeypatched away).

WHAT IS CLAIMED: ``engine_recovery_retries`` (default ``0`` -- recovery OFF) makes
``rollout_async`` stop/start the fleet and re-run the SAME step's tasks when -- and ONLY when --
every trajectory in the step is an ENGINE infra failure; the failed attempt's summary is replaced
by exactly one ``[agentic-rl:engine-recovery]`` JSON line per attempt on stderr while the normal
``[agentic-rl:rollout]`` summary is printed once, for the FINAL gather; the final batch is
measured (rewards not ``None``) after a recovery that cured the fleet;
``_recover_engines`` re-pushes the checkpoint recorded in ``_state["last_pushed_path"]``
(``publish`` records it after a COMPLETE push) so the restarted engine serves the current policy;
the infra gate still refuses an all-infra batch once the retries are exhausted (and immediately,
with the default 0); a PARTIAL engine failure never restarts anything; construction refuses a
non-real-int/negative ``engine_recovery_retries`` and a positive one without a fleet.

WHAT IS NOT CLAIMED: no SGLang/vLLM engine, no real HTTP reload and no real checkpoint save is
performed anywhere here -- clients are pure-Python scripts and ``DiskWeightSync.push`` is
monkeypatched in the one test that names a checkpoint path.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl import markup
from foundationscale.agentic_rl.contracts import Termination
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
from foundationscale.rl.weightsync import SyncReport

# ---------------------------------------------------------------------------
# A toy tokenizer, toy markup-level call builders, a scripted client family,
# and a toy reward -- copied verbatim from test_rollout_host.py (nothing is
# imported across test modules).
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


class ToyReward:
    """Scores submitted_answer by its length; abstains (measured 0.0) when absent."""

    async def score(self, task: Any, outcome: Any) -> RewardVerdict:
        if outcome.submitted_answer is None:
            return RewardVerdict(0.0, "no_answer")
        return RewardVerdict(float(len(outcome.submitted_answer)), None)


# ---------------------------------------------------------------------------
# Shared construction helpers (verbatim from test_rollout_host.py; `fleet` is
# injected through `_host`'s existing `**overrides`).
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
# A fake fleet whose clients hang together: shared mutable "mode" state, first
# gather times out, `start_all` cures the mode (unless told not to).
# ---------------------------------------------------------------------------


class FleetMode:
    """Shared mutable engine state behind a ``FakeFleet`` and every client it serves.

    ``hung=True`` means the engines are in the recovered-from hang shape: every attempt
    times out on ``generate`` (after one harmless bash call, so the episode carries a real
    turn before the fault -- the shape ``test_rollout_host.ExplodingAfterOneClient`` uses).
    ``FakeFleet.start_all`` flips ``hung`` to ``False`` iff ``cured_by_restart``.
    """

    def __init__(self, *, cured_by_restart: bool = True) -> None:
        self.hung = True
        self.cured_by_restart = cured_by_restart


class ModeSwitchingClient:
    """A ``GenerationClient`` behaving like the engine behind its fleet's shared mode."""

    def __init__(self, fleet: FakeFleet, index: int) -> None:
        self._fleet = fleet
        self._index = index
        self._calls = 0

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        self._calls += 1
        if self._fleet.is_hung(self._index):
            if self._calls == 1:
                return _generation(_bash_call_text("echo hi"))
            raise EngineInfraError("timeout", "engine hung")
        return _generation(_submit_call_text("ok"))


class FakeFleet:
    """A fake ``EngineFleet``: serves mode-driven clients and counts lifecycle calls.

    ``hung_indices`` narrows the hang to some session indices only (test (c)'s PARTIAL
    engine failure); ``None`` hangs every one of them (a whole-fleet engine hang).
    """

    def __init__(self, mode: FleetMode, hung_indices: set[int] | None = None) -> None:
        self.mode = mode
        self.hung_indices = hung_indices
        self.stop_calls = 0
        self.start_calls = 0

    def is_hung(self, index: int) -> bool:
        return self.mode.hung and (self.hung_indices is None or index in self.hung_indices)

    def client_for(self, index: int) -> ModeSwitchingClient:
        return ModeSwitchingClient(self, index)

    def stop_all(self) -> None:
        self.stop_calls += 1

    def start_all(self) -> None:
        self.start_calls += 1
        if self.mode.cured_by_restart:
            self.mode.hung = False


_RECOVERY_PREFIX = "[agentic-rl:engine-recovery] "
_ROLLOUT_PREFIX = "[agentic-rl:rollout] "


# ---------------------------------------------------------------------------
# (a) all-engine-infra step + retries=1 -> fleet restarted, step retried and
#     finally measured; exactly one engine-recovery line on stderr.
# ---------------------------------------------------------------------------


def test_all_engine_infra_step_restarts_the_fleet_and_retries_the_same_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fleet = FakeFleet(FleetMode(cured_by_restart=True))
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=None,
        fleet=fleet,
        engine_recovery_retries=1,
        gates=True,
        declared_max_infra_rate=0.5,
    )
    batch = host.rollout(0)
    # The fleet was restarted exactly once -- recovery's own retry budget -- and the
    # retried gather finally produced a MEASURED batch (rewards not None).
    assert fleet.stop_calls == 1
    assert fleet.start_calls == 1
    assert len(batch) == 2
    assert all(reward is not None for reward in batch.columns["reward"])
    assert all(termination is Termination.STOP for termination in batch.columns["termination"])
    err = capsys.readouterr().err
    recovery_lines = [line for line in err.splitlines() if line.startswith(_RECOVERY_PREFIX)]
    assert len(recovery_lines) == 1
    payload = json.loads(recovery_lines[0][len(_RECOVERY_PREFIX) :])
    assert payload["rollout_step"] == 0
    assert payload["attempt"] == 1
    assert payload["of"] == 1
    # The failed attempt was every row an engine abstraction: 2 trajectories in, 2 counted.
    assert sum(payload["abstention_reasons"].values()) == 2
    # The normal summary is NOT printed per failed attempt: one line, for the final gather.
    assert sum(1 for line in err.splitlines() if line.startswith(_ROLLOUT_PREFIX)) == 1


# ---------------------------------------------------------------------------
# (b) default retries=0 -> no restart at all; the infra gate refuses as before.
# ---------------------------------------------------------------------------


def test_default_zero_retries_never_restarts_and_the_infra_gate_refuses(tmp_path: Path) -> None:
    fleet = FakeFleet(FleetMode(cured_by_restart=True))
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=None,
        fleet=fleet,
        gates=True,
        declared_max_infra_rate=0.5,
    )
    assert host.engine_recovery_retries == 0
    with pytest.raises(RolloutHostRefusal, match="ROLLOUT gates blocked"):
        host.rollout(0)
    assert fleet.stop_calls == 0
    assert fleet.start_calls == 0


# ---------------------------------------------------------------------------
# (c) PARTIAL engine failure (only some attempts fail) -> not an all-engine-infra
#     step -> no restart, no recovery line.
# ---------------------------------------------------------------------------


def test_partial_engine_failure_does_not_restart_the_fleet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fleet = FakeFleet(FleetMode(cured_by_restart=True), hung_indices={1})
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=None,
        fleet=fleet,
        engine_recovery_retries=1,
        gates=False,
    )
    batch = host.rollout(0)
    assert fleet.stop_calls == 0
    assert fleet.start_calls == 0
    terminations = tuple(batch.columns["termination"])
    assert terminations.count(Termination.INFRA) == 1
    assert terminations.count(Termination.STOP) == 1
    err = capsys.readouterr().err
    assert not any(line.startswith(_RECOVERY_PREFIX) for line in err.splitlines())


# ---------------------------------------------------------------------------
# (d) still all-engine-infra after every retry -> the fleet was restarted exactly
#     `retries` times and the infra gate refuses the exhausted step.
# ---------------------------------------------------------------------------


def test_recovery_retries_exhausted_and_the_infra_gate_still_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fleet = FakeFleet(FleetMode(cured_by_restart=False))
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=None,
        fleet=fleet,
        engine_recovery_retries=2,
        gates=True,
        declared_max_infra_rate=0.5,
    )
    with pytest.raises(RolloutHostRefusal, match="ROLLOUT gates blocked"):
        host.rollout(0)
    assert fleet.stop_calls == 2
    assert fleet.start_calls == 2
    err = capsys.readouterr().err
    recovery_lines = [line for line in err.splitlines() if line.startswith(_RECOVERY_PREFIX)]
    assert len(recovery_lines) == 2
    attempts = [json.loads(line[len(_RECOVERY_PREFIX) :])["attempt"] for line in recovery_lines]
    assert attempts == [1, 2]


# ---------------------------------------------------------------------------
# (e) validation: a bool/negative/non-int count is refused naming the field, and
#     a positive count with only a client_factory (no fleet) is refused too.
# ---------------------------------------------------------------------------


def test_engine_recovery_retries_must_be_a_real_non_negative_int(tmp_path: Path) -> None:
    for bad in (True, False, -1, 2.0, "1"):
        with pytest.raises(RolloutHostRefusal, match="engine_recovery_retries"):
            _host(
                tmp_path,
                uids=["taskA"],
                client_factory=lambda index: SubmitClient("x"),
                engine_recovery_retries=bad,
            )


def test_engine_recovery_retries_needs_a_fleet_to_restart(tmp_path: Path) -> None:
    with pytest.raises(RolloutHostRefusal, match="engine recovery needs a fleet to restart"):
        _host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: SubmitClient("x"),
            engine_recovery_retries=1,
        )


# ---------------------------------------------------------------------------
# (f) with a REAL DiskWeightSync (its __post_init__ demands a real EngineFleet),
#     recovery re-pushes the checkpoint recorded in _state["last_pushed_path"]
#     exactly once -- a restarted engine must serve the CURRENT policy, not the
#     base model. DiskWeightSync.push is monkeypatched (no real reload).
# ---------------------------------------------------------------------------


def test_recovery_repushes_last_published_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = pytest.importorskip("_fake_sglang_fixtures")
    fake_module = fixtures.write_fake_server(tmp_path)
    spec = fixtures.make_server_spec(tmp_path, fake_module, port=fixtures.free_port())
    real_fleet = EngineFleet(servers=(SGLangServer(spec),))
    weight_sync = DiskWeightSync(publish_root=str(tmp_path / "pub"), fleet=real_fleet)
    pushed: list[str] = []

    def fake_push(self: DiskWeightSync, path: str) -> SyncReport:
        pushed.append(path)
        return SyncReport(
            transport="disk",
            offered=("server_0",),
            transferred=("server_0",),
            skipped=(),
            failed_ranks=(),
            bytes_moved=None,
            seconds=0.0,
            is_stale=False,
        )

    monkeypatch.setattr(DiskWeightSync, "push", fake_push)
    fleet = FakeFleet(FleetMode(cured_by_restart=True))
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=None,
        fleet=fleet,
        weight_sync=weight_sync,
        engine_recovery_retries=1,
    )
    last = str(tmp_path / "pub" / "step_000002")
    # Exactly what publish() records after a COMPLETE push (there is no model here to
    # publish through save_checkpoint, so the recorded state is set directly).
    host._state["last_pushed_path"] = last
    batch = host.rollout(0)
    assert pushed == [last]
    assert fleet.stop_calls == 1
    assert fleet.start_calls == 1
    assert all(reward is not None for reward in batch.columns["reward"])


def test_restart_mode_recovery_refuses_before_any_checkpoint_was_published(tmp_path: Path) -> None:
    fixtures = pytest.importorskip("_fake_sglang_fixtures")
    fake_module = fixtures.write_fake_server(tmp_path)
    spec = fixtures.make_server_spec(tmp_path, fake_module, port=fixtures.free_port())
    real_fleet = EngineFleet(servers=(SGLangServer(spec),))
    weight_sync = DiskWeightSync(
        publish_root=str(tmp_path / "pub"), fleet=real_fleet, mode="restart"
    )
    fleet = FakeFleet(FleetMode(cured_by_restart=True))
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=None,
        fleet=fleet,
        weight_sync=weight_sync,
        engine_recovery_retries=1,
    )
    with pytest.raises(RolloutHostRefusal, match="before any checkpoint was published"):
        host.rollout(0)
    assert fleet.stop_calls == 0
    assert fleet.start_calls == 0


def _row(termination: Termination, reason: object) -> Any:
    # _all_engine_infra reads only these two attributes.
    from types import SimpleNamespace

    return SimpleNamespace(termination=termination, abstention_reason=reason)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([], False),
        ([_row(Termination.INFRA, "infra:engine_timeout"), _row(Termination.STOP, None)], False),
        ([_row(Termination.INFRA, None)], False),
        ([_row(Termination.INFRA, "infra:EnvInfraError")], False),
        (
            [
                _row(Termination.INFRA, "infra:engine_timeout"),
                _row(Termination.INFRA, "infra:EngineInfraError"),
            ],
            True,
        ),
    ],
)
def test_all_engine_infra_accepts_only_an_all_engine_failure(
    rows: list[Any], expected: bool
) -> None:
    from foundationscale.agentic_rl.rollout_host import _all_engine_infra

    assert _all_engine_infra(rows) is expected


def _report(*, complete: bool) -> SyncReport:
    return SyncReport(
        transport="disk",
        offered=("server_0", "server_1"),
        transferred=("server_0", "server_1") if complete else ("server_0",),
        skipped=() if complete else ("server_1",),
        failed_ranks=() if complete else (1,),
        bytes_moved=None,
        seconds=0.0,
        is_stale=False,
    )


def _recovering_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fleet: FakeFleet, *, mode: str, complete: bool
) -> tuple[RolloutHost, list[str]]:
    fixtures = pytest.importorskip("_fake_sglang_fixtures")
    fake_module = fixtures.write_fake_server(tmp_path)
    spec = fixtures.make_server_spec(tmp_path, fake_module, port=fixtures.free_port())
    weight_sync = DiskWeightSync(
        publish_root=str(tmp_path / "pub"),
        fleet=EngineFleet(servers=(SGLangServer(spec),)),
        mode=mode,
    )
    pushed: list[str] = []

    def fake_push(self: DiskWeightSync, path: str) -> SyncReport:
        pushed.append(path)
        fleet.mode.hung = False  # a restart-mode push relaunches the servers itself
        return _report(complete=complete)

    monkeypatch.setattr(DiskWeightSync, "push", fake_push)
    host = _host(
        tmp_path,
        uids=["taskA"],
        client_factory=None,
        fleet=fleet,
        weight_sync=weight_sync,
        engine_recovery_retries=1,
    )
    host._state["last_pushed_path"] = str(tmp_path / "pub" / "step_000003")
    return host, pushed


def test_restart_mode_recovery_relaunches_through_the_push_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet = FakeFleet(FleetMode(cured_by_restart=False))
    host, pushed = _recovering_host(tmp_path, monkeypatch, fleet, mode="restart", complete=True)
    batch = host.rollout(0)
    assert pushed == [str(tmp_path / "pub" / "step_000003")]
    assert (fleet.stop_calls, fleet.start_calls) == (0, 0)
    assert all(reward is not None for reward in batch.columns["reward"])


def test_recovery_refuses_when_the_fleet_cannot_restart(tmp_path: Path) -> None:
    class UnstartableFleet(FakeFleet):
        def start_all(self) -> None:
            self.start_calls += 1
            raise EngineInfraError("startup_crashed", "server exited before becoming healthy")

    fleet = UnstartableFleet(FleetMode(cured_by_restart=False))
    host = _host(
        tmp_path, uids=["taskA"], client_factory=None, fleet=fleet, engine_recovery_retries=1
    )
    with pytest.raises(RolloutHostRefusal, match="failed to restart the fleet"):
        host.rollout(0)
    assert fleet.start_calls == 1


def test_recovery_refuses_an_incomplete_re_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet = FakeFleet(FleetMode(cured_by_restart=True))
    host, pushed = _recovering_host(
        tmp_path, monkeypatch, fleet, mode="reload_endpoint", complete=False
    )
    with pytest.raises(RolloutHostRefusal, match="was NOT complete"):
        host.rollout(0)
    assert len(pushed) == 1


def test_recovery_repush_may_run_its_own_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DiskWeightSync.push drives the fleet with asyncio.run(); recovery is reached from inside
    # rollout_async's running loop, so the re-push must not run on that loop's thread.
    import asyncio

    fleet = FakeFleet(FleetMode(cured_by_restart=True))
    host, pushed = _recovering_host(
        tmp_path, monkeypatch, fleet, mode="reload_endpoint", complete=True
    )

    def push_with_its_own_loop(self: DiskWeightSync, path: str) -> SyncReport:
        asyncio.run(asyncio.sleep(0))  # what the real push does
        pushed.append(path)
        return _report(complete=True)

    monkeypatch.setattr(DiskWeightSync, "push", push_with_its_own_loop)
    batch = host.rollout(0)
    assert pushed == [str(tmp_path / "pub" / "step_000003")]
    assert all(reward is not None for reward in batch.columns["reward"])


def _publishing_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reports: list[bool], *, retries: int
) -> tuple[RolloutHost, FakeFleet, list[str]]:
    """A host whose pushes report ``reports`` in order (True = complete); no real save."""
    import foundationscale.rl.distributed as distributed

    fleet = FakeFleet(FleetMode(cured_by_restart=True))
    host, pushed = _recovering_host(
        tmp_path, monkeypatch, fleet, mode="reload_endpoint", complete=True
    )
    host = dataclasses.replace(host, engine_recovery_retries=retries, gates=False)
    outcomes = list(reports)

    def scripted_push(self: DiskWeightSync, path: str) -> SyncReport:
        pushed.append(path)
        return _report(complete=outcomes.pop(0))

    monkeypatch.setattr(DiskWeightSync, "push", scripted_push)
    monkeypatch.setattr(distributed, "save_checkpoint", lambda *args, **kwargs: None)
    return host, fleet, pushed


class _Rank0:
    rank = 0


def test_an_incomplete_publish_push_restarts_the_fleet_and_pushes_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A hang that timed out only part of a step trains anyway; the push then meets the hung
    # engine. Recovery restarts the fleet and pushes THIS checkpoint (GB200 math_grpo_s5).
    host, fleet, pushed = _publishing_host(tmp_path, monkeypatch, [False, True], retries=1)
    host.publish(object(), object(), _Rank0(), 4)
    new = host.weight_sync.path_for(5)  # type: ignore[union-attr]
    assert pushed == [new, new]
    assert (fleet.stop_calls, fleet.start_calls) == (1, 1)
    assert host._state["last_pushed_path"] == new
    assert '"publish_step": 4' in capsys.readouterr().err


def test_an_incomplete_publish_push_without_retries_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host, fleet, pushed = _publishing_host(tmp_path, monkeypatch, [False], retries=0)
    with pytest.raises(RolloutHostRefusal, match="weight push to the fleet was NOT complete"):
        host.publish(object(), object(), _Rank0(), 4)
    assert (fleet.stop_calls, len(pushed)) == (0, 1)


def test_a_failed_publish_recovery_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    host, fleet, pushed = _publishing_host(tmp_path, monkeypatch, [False, False], retries=1)
    with pytest.raises(RolloutHostRefusal, match="re-push of .* was NOT complete"):
        host.publish(object(), object(), _Rank0(), 4)
    assert (fleet.start_calls, len(pushed)) == (1, 2)


def test_partial_engine_failure_above_the_declared_rate_restarts_the_fleet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A hang that times out only PART of a step (GB200 swe_v1_s6: 21 of 32) is refused by the
    # infra gate anyway, so it is recovered like a full hang: restart, then retry the step.
    fleet = FakeFleet(FleetMode(cured_by_restart=True), hung_indices={1})
    host = _host(
        tmp_path, uids=["taskA"], client_factory=None, fleet=fleet,
        engine_recovery_retries=1, gates=False, declared_max_infra_rate=0.4,
    )  # fmt: skip
    batch = host.rollout(0)
    assert (fleet.stop_calls, fleet.start_calls) == (1, 1)
    assert Termination.INFRA not in tuple(batch.columns["termination"])
    err = capsys.readouterr().err
    assert any(line.startswith(_RECOVERY_PREFIX) for line in err.splitlines())


def test_partial_engine_failure_at_the_declared_rate_trains_as_is(tmp_path: Path) -> None:
    fleet = FakeFleet(FleetMode(cured_by_restart=True), hung_indices={1})
    host = _host(
        tmp_path, uids=["taskA"], client_factory=None, fleet=fleet,
        engine_recovery_retries=1, gates=False, declared_max_infra_rate=0.5,
    )  # fmt: skip
    host.rollout(0)
    assert fleet.stop_calls == 0  # 1 of 2 is not above 0.5: the gate would accept this step
