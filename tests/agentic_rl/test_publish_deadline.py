"""RolloutHost.publish_deadline_s: a hung publish ends the run RED (5) instead of hanging."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl import rollout_host as rh
from foundationscale.agentic_rl.rollout_host import RolloutHost, RolloutHostRefusal

try:
    import test_engine_recovery as fakes  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - package-style collection
    from tests.agentic_rl import test_engine_recovery as fakes  # type: ignore[no-redef]


def _host(tmp_path: Path, **kw: Any) -> RolloutHost:
    return fakes._host(
        tmp_path, uids=["taskA"], client_factory=lambda i: fakes.SubmitClient("x"), **kw
    )


class _Exits(list[int]):
    """Exit codes the deadline handler requested, plus an event set when it fires."""

    def __init__(self) -> None:
        super().__init__()
        self.fired = threading.Event()


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> _Exits:
    codes = _Exits()

    def record(code: int) -> None:
        codes.append(code)
        codes.fired.set()

    monkeypatch.setattr(rh, "_deadline_exit", record)
    return codes


def test_without_a_deadline_publish_runs_the_body_and_arms_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exits: _Exits
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(RolloutHost, "_publish_body", lambda self, m, t, c, s: calls.append(s))
    started = threading.active_count()
    _host(tmp_path).publish(object(), object(), object(), 3)
    assert calls == [3] and exits == [] and threading.active_count() == started


def test_a_publish_past_its_deadline_dumps_stacks_and_exits_red(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exits: _Exits,
) -> None:
    def hung(self: RolloutHost, m: Any, t: Any, c: Any, s: int) -> None:
        exits.fired.wait(timeout=5.0)  # "hangs" until the deadline fires

    monkeypatch.setattr(RolloutHost, "_publish_body", hung)
    _host(tmp_path, publish_deadline_s=0.2).publish(object(), object(), object(), 81)
    assert exits == [5]
    err = capsys.readouterr().err
    assert "[fs:agentic-rl:red] RolloutHost.publish: step 81" in err
    assert "publish_deadline_s=0.2s" in err
    # The stack dump itself goes to the real stderr here (pytest's capture has no fd),
    # which is exactly the fallback path; the RED exit above is what must never be lost.


def test_a_publish_inside_its_deadline_cancels_the_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exits: _Exits
) -> None:
    monkeypatch.setattr(RolloutHost, "_publish_body", lambda self, m, t, c, s: None)
    _host(tmp_path, publish_deadline_s=0.2).publish(object(), object(), object(), 4)
    time.sleep(0.4)  # past the deadline: a cancelled timer must not fire
    assert exits == []


def test_a_failing_publish_still_cancels_the_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exits: _Exits
) -> None:
    def boom(self: RolloutHost, m: Any, t: Any, c: Any, s: int) -> None:
        raise RolloutHostRefusal("push incomplete")

    monkeypatch.setattr(RolloutHost, "_publish_body", boom)
    with pytest.raises(RolloutHostRefusal, match="push incomplete"):
        _host(tmp_path, publish_deadline_s=0.2).publish(object(), object(), object(), 4)
    time.sleep(0.4)
    assert exits == []  # the refusal stays the verdict; no second, RED exit on top


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan"), True, "60"])
def test_a_bad_deadline_is_refused(tmp_path: Path, bad: Any) -> None:
    with pytest.raises(RolloutHostRefusal, match="publish_deadline_s"):
        _host(tmp_path, publish_deadline_s=bad)


def test_config_key_defaults_to_none_and_refuses_non_positive() -> None:
    import importlib.util

    from foundationscale.agentic_rl.config import AgenticConfigRefusal, build_config

    spec = importlib.util.spec_from_file_location(
        "_test_config_raw_pd", Path(__file__).with_name("test_config.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def build(**rollout: Any) -> Any:
        raw = module._raw_config()
        raw["rollout"] = {**raw["rollout"], **rollout}
        return build_config(raw, {})

    assert build().rollout.publish_deadline_s is None
    assert build(publish_deadline_s=10800).rollout.publish_deadline_s == 10800
    with pytest.raises(AgenticConfigRefusal, match="publish_deadline_s"):
        build(publish_deadline_s=0)


def test_deadline_exit_ends_the_process_through_os_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real seam (not the monkeypatched one above): it must call os._exit with the code,
    which skips every finally/atexit a hung thread could block on."""
    codes: list[int] = []
    monkeypatch.setattr(rh.os, "_exit", codes.append)
    rh._deadline_exit(5)
    assert codes == [5]
