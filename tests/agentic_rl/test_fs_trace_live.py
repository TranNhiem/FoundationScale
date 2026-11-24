"""Live crash-evidence traces, loop observer hooks, and rollout-host integration."""

from __future__ import annotations

import asyncio
import json
import os
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.agentic_rl.fs_trace import (
    LiveTraceWriter,
    last_open_operation,
    read_live,
)

try:
    _helpers = import_module("tests.agentic_rl.test_native_tool_loop")
except ModuleNotFoundError:
    _helpers = import_module("test_native_tool_loop")

FakeEnvironment = _helpers.FakeEnvironment
ScriptedClient = _helpers.ScriptedClient
ToyTokenizer = _helpers.ToyTokenizer
_bash_call_text = _helpers._bash_call_text
_budget = _helpers._budget
_exec_result = _helpers._exec_result
_generation = _helpers._generation
_loop = _helpers._loop
_submit_call_text = _helpers._submit_call_text
_task = _helpers._task


def test_live_trace_writer_seq_fsync_and_format(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "live.jsonl"
    fsyncs: list[int] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        fsyncs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    writer = LiveTraceWriter(
        path,
        trace_id="tr_test",
        session_id="session/1",
    )
    writer.generation_started(2, 7)
    writer.generation_finished(2, "stop", 3)
    writer.tool_started(2, 0, "bash")
    writer.tool_finished(2, 0, 0, False)

    events = read_live(path)
    assert [event["seq"] for event in events] == [0, 1, 2, 3]
    assert len(fsyncs) == 4
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 4
    for line, event in zip(raw_lines, events, strict=True):
        assert json.loads(line) == event
        assert list(event) == sorted(event)
        assert event["schema_version"] == "fs.trace.live/v1"
        assert event["trace_id"] == "tr_test"
        assert event["session_id"] == "session/1"
        assert isinstance(event["at_ts"], float)
    assert events[0]["policy_version"] == 7
    assert events[1]["finish_reason"] == "stop"
    assert events[1]["n_tokens"] == 3
    assert events[2]["name"] == "bash"
    assert events[2]["call_index"] == 0
    assert events[3]["exit_code"] == 0
    assert events[3]["timed_out"] is False


def test_read_live_ignores_only_an_unterminated_final_line(tmp_path: Any) -> None:
    path = tmp_path / "live.jsonl"
    path.write_bytes(b'{"seq": 0}\n{"seq":')
    assert read_live(path) == [{"seq": 0}]
    path.write_bytes(b'{"seq": 0}')
    assert read_live(path) == []
    path.write_bytes(b'not json\n{"seq":')
    with pytest.raises(ValueError):
        read_live(path)


def _live_event(event: str, turn_index: int, **fields: Any) -> dict[str, Any]:
    return {"event": event, "turn_index": turn_index, **fields}


def test_last_open_operation_clean_hung_generation_and_hung_tool() -> None:
    generation_open = _live_event("generation_started", 2, policy_version=5)
    generation_closed = _live_event("generation_finished", 2)
    tool_open = _live_event("tool_started", 2, call_index=0, name="bash")
    tool_closed = _live_event("tool_finished", 2, call_index=0)
    clean = [generation_open, generation_closed, tool_open, tool_closed]
    assert last_open_operation(clean) is None
    assert last_open_operation([generation_open]) is generation_open
    assert generation_open["policy_version"] == 5
    assert last_open_operation([generation_open, generation_closed, tool_open]) is tool_open


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def generation_started(self, turn_index: int, policy_version: int | None) -> None:
        self.events.append(("generation_started", turn_index, policy_version))

    def generation_finished(
        self,
        turn_index: int,
        finish_reason: str | None,
        n_tokens: int,
    ) -> None:
        self.events.append(("generation_finished", turn_index, finish_reason, n_tokens))

    def tool_started(self, turn_index: int, call_index: int, name: str | None) -> None:
        self.events.append(("tool_started", turn_index, call_index, name))

    def tool_finished(
        self,
        turn_index: int,
        call_index: int,
        exit_code: int | None,
        timed_out: bool,
    ) -> None:
        self.events.append(
            (
                "tool_finished",
                turn_index,
                call_index,
                exit_code,
                timed_out,
            )
        )


async def _run_observed_episode(generations: list[Any], observer: Any) -> tuple[Any, Any]:
    loop = _loop(tokenizer=ToyTokenizer())
    client = ScriptedClient(generations)
    env = FakeEnvironment(execute_fn=lambda command: _exec_result("hi\n"))
    if observer is None:
        outcome = await loop.run_episode(
            _task(),
            client=client,
            env=env,
            budget=_budget(),
        )
    else:
        outcome = await loop.run_episode(
            _task(),
            client=client,
            env=env,
            budget=_budget(),
            observer=observer,
            policy_version=9,
        )
    return outcome, env


def test_native_tool_loop_observer_pairs_and_none_is_identical() -> None:
    generations = [
        _generation(_bash_call_text("echo hi")),
        _generation(_submit_call_text("done")),
    ]
    observer = RecordingObserver()
    observed, observed_env = asyncio.run(_run_observed_episode(generations, observer))
    baseline, baseline_env = asyncio.run(_run_observed_episode(generations, None))
    assert observed == baseline
    assert observed_env.commands == ["echo hi"]
    assert baseline_env.commands == ["echo hi"]
    assert observer.events == [
        ("generation_started", 2, 9),
        (
            "generation_finished",
            2,
            "stop",
            len(generations[0].token_ids),
        ),
        ("tool_started", 2, 0, "bash"),
        ("tool_finished", 2, 0, 0, False),
        ("generation_started", 4, 9),
        (
            "generation_finished",
            4,
            "stop",
            len(generations[1].token_ids),
        ),
    ]


class _Backend:
    def __init__(self, instance: Any) -> None:
        self.instance = instance

    def create(self, spec: Any, *, instance_id: str) -> Any:
        return self.instance


class _Reward:
    async def score(self, task: Any, outcome: Any) -> Any:
        return SimpleNamespace(value=1.0, reason=None)


def test_rollout_host_writes_closed_live_traces_that_join_the_episode_trace(
    tmp_path: Path,
) -> None:
    """A clean rollout leaves one closed live file per session, stamped with the policy
    version, whose trace id is the id of that session's episode trace (the two files join)."""
    from foundationscale.agentic_rl.fs_trace import read_trace

    try:
        import test_engine_recovery as fakes  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - package-style collection
        from tests.agentic_rl import test_engine_recovery as fakes  # type: ignore[no-redef]

    trace_dir = tmp_path / "run-live"
    host = fakes._host(
        tmp_path,
        uids=["taskA"],
        client_factory=lambda i: fakes.SubmitClient("xy"),
        trace_dir=str(trace_dir),
        gates=False,
    )
    host.rollout(7)
    live_files = sorted((trace_dir / "live" / "step_000007").glob("*.jsonl"))
    assert len(live_files) == 2  # tasks_per_step=1 x group_size=2
    episode_ids = set(read_trace(trace_dir / "step_000007.jsonl"))
    for path in live_files:
        events = read_live(path)
        assert last_open_operation(events) is None  # every started operation finished
        started = [e for e in events if e["event"] == "generation_started"]
        assert started and started[0]["policy_version"] == 7  # no fleet ledger: the step
        assert {e["trace_id"] for e in events} <= episode_ids
