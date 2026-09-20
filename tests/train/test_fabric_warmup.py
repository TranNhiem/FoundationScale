"""The fabric warm-up bounds, checks and reports the first collective (#516).

The defect these tests pin is not "FoundationScale cannot do a collective". It
is that the plane's first collective was UNBOUNDED: accelerate builds the
process group with a 1800-second timeout, so a fabric fault measured on a GB200
estate -- 2 stalls in 21 two-rank runs of a 30-line probe that never imports
this framework -- presented as a 901-second hang at 100% GPU utilisation with no
diagnostic, where the same fault under an explicit ``timeout=`` named itself in
120 seconds. Every test below asserts one half of the difference between those
two outcomes.

``_warm_up_fabric`` imports ``torch.distributed`` inside the function, so these
tests substitute the module's attributes rather than the module. That is
deliberate: patching the real module keeps the function's own import path under
test, where injecting a fake module object would exercise a code path
production never takes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
import torch.distributed as dist

from foundationscale.train.loop import (
    _WARMUP_ATTEMPTS_VAR,
    _WARMUP_TIMEOUT_VAR,
    _warm_up_fabric,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _FakeGroup:
    """Stands in for a ProcessGroup; identity is all the function needs."""


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world: int = 2,
    backend: str = "gloo",
    all_reduce: Callable[..., Any] | None = None,
    new_group: Callable[..., Any] | None = None,
) -> dict[str, list[Any]]:
    """Substitute the distributed surface and record what the probe did to it."""
    log: dict[str, list[Any]] = {"new_group": [], "all_reduce": [], "destroyed": []}

    def _default_all_reduce(tensor: Any, op: Any = None, group: Any = None) -> None:
        log["all_reduce"].append((op, group))
        tensor.fill_(1)

    def _default_new_group(timeout: Any = None, **_: Any) -> Any:
        log["new_group"].append(timeout)
        return _FakeGroup()

    monkeypatch.setattr(dist, "ProcessGroup", _FakeGroup)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: world)
    monkeypatch.setattr(dist, "get_backend", lambda: backend)
    monkeypatch.setattr(dist, "new_group", new_group or _default_new_group)
    monkeypatch.setattr(dist, "all_reduce", all_reduce or _default_all_reduce)
    monkeypatch.setattr(
        dist, "destroy_process_group", lambda group=None: log["destroyed"].append(group)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return log


def test_uninitialised_distributed_is_warmed_by_doing_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-process run passes, and SAYS there was nothing to warm.

    Announcing the empty case is the point. A silent pass here and a silent
    pass over a healthy 8-rank fabric read identically in a log, and only one
    of them is evidence.
    """
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    ok, lines = _warm_up_fabric({})
    assert ok is True
    assert lines
    assert any("nothing to warm" in line for line in lines)


def test_world_size_one_is_warmed_by_doing_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """One rank crosses no device, so there is no fabric to probe -- and it says so."""
    _install(monkeypatch, world=1)
    ok, lines = _warm_up_fabric({})
    assert ok is True
    assert any("world size is 1" in line for line in lines)


def test_healthy_fabric_reports_the_rank_count_and_the_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed probe is reported with its rank count and its duration."""
    log = _install(monkeypatch, world=4)
    ok, lines = _warm_up_fabric({})
    assert ok is True
    joined = " | ".join(lines)
    assert "4 rank(s)" in joined
    assert "returned the agreed value" in joined
    assert len(log["all_reduce"]) == 1


def test_the_bound_is_carried_on_a_subgroup_not_the_main_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timeout reaches ``new_group``, so the training group's own timeout is untouched.

    A bound tight enough to catch a dead fabric is a bound certain to fire on a
    healthy long training step, so it must not be applied to the group the
    steps run on.
    """
    log = _install(monkeypatch)
    ok, _ = _warm_up_fabric({_WARMUP_TIMEOUT_VAR: "7"})
    assert ok is True
    assert len(log["new_group"]) == 1
    assert log["new_group"][0].total_seconds() == pytest.approx(7.0)


def test_defaults_and_declarations_are_distinguishable_in_the_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The announcement says whether each knob was declared or defaulted."""
    _install(monkeypatch)
    _, defaulted = _warm_up_fabric({})
    assert any("bound=120s (default)" in line for line in defaulted)
    assert any("attempts=1 (default)" in line for line in defaulted)

    _install(monkeypatch)
    _, declared = _warm_up_fabric({_WARMUP_TIMEOUT_VAR: "30", _WARMUP_ATTEMPTS_VAR: "2"})
    assert any("bound=30s (declared)" in line for line in declared)
    assert any("attempts=2 (declared)" in line for line in declared)


@pytest.mark.parametrize("value", ["", "soon", "12s", "nan-ish"])
def test_an_unparseable_timeout_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A declaration the framework cannot read is refused, naming the variable.

    Not demoted to the default: the operator declared something, and silently
    substituting 120 would hide a typo inside a healthy-looking run.
    """
    _install(monkeypatch)
    ok, lines = _warm_up_fabric({_WARMUP_TIMEOUT_VAR: value})
    assert ok is False
    assert any(_WARMUP_TIMEOUT_VAR in line for line in lines)


@pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
def test_a_non_positive_timeout_is_refused_because_it_is_not_a_bound(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Zero or negative seconds would make the probe fail on a healthy fabric."""
    _install(monkeypatch)
    ok, lines = _warm_up_fabric({_WARMUP_TIMEOUT_VAR: value})
    assert ok is False
    assert any("not a bound" in line for line in lines)


@pytest.mark.parametrize("value", ["", "two", "1.5"])
def test_an_unparseable_attempt_count_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """An attempt count that is not an integer is refused, naming the variable."""
    _install(monkeypatch)
    ok, lines = _warm_up_fabric({_WARMUP_ATTEMPTS_VAR: value})
    assert ok is False
    assert any(_WARMUP_ATTEMPTS_VAR in line for line in lines)


def test_zero_attempts_is_refused_rather_than_reported_as_a_healthy_fabric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero attempts would report a verdict about a fabric nobody probed."""
    log = _install(monkeypatch)
    ok, lines = _warm_up_fabric({_WARMUP_ATTEMPTS_VAR: "0"})
    assert ok is False
    assert any("at least 1" in line for line in lines)
    assert log["all_reduce"] == []


def test_a_stalled_collective_refuses_and_names_the_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising collective refuses (96) and reports the exception by type and text."""

    def _stall(tensor: Any, op: Any = None, group: Any = None) -> None:
        raise RuntimeError("Watchdog caught collective operation timeout")

    _install(monkeypatch, all_reduce=_stall)
    ok, lines = _warm_up_fabric({})
    assert ok is False
    joined = " | ".join(lines)
    assert "RuntimeError" in joined
    assert "Watchdog caught collective operation timeout" in joined
    assert "resubmitting the job" in joined


def test_every_attempt_is_reported_and_every_subgroup_is_destroyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures do not leak a communicator, and each attempt appears in the log.

    A probe that leaks one subgroup per attempt would make the diagnostic
    itself a resource fault, which is the shape of a safety feature that has
    to be turned off.
    """

    def _stall(tensor: Any, op: Any = None, group: Any = None) -> None:
        raise RuntimeError("boom")

    log = _install(monkeypatch, all_reduce=_stall)
    ok, lines = _warm_up_fabric({_WARMUP_ATTEMPTS_VAR: "3"})
    assert ok is False
    assert len(log["destroyed"]) == 3
    for attempt in (1, 2, 3):
        assert any(f"attempt {attempt}/3 FAILED" in line for line in lines)


def test_a_later_attempt_can_succeed_and_the_earlier_failure_is_still_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovered probe passes, and the log still records that the first try failed.

    Dropping the failed attempt would erase the one signal that says this
    estate's fabric is intermittent.
    """
    calls = {"n": 0}

    def _flaky(tensor: Any, op: Any = None, group: Any = None) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cold fabric")
        tensor.fill_(1)

    _install(monkeypatch, all_reduce=_flaky)
    ok, lines = _warm_up_fabric({_WARMUP_ATTEMPTS_VAR: "2"})
    assert ok is True
    joined = " | ".join(lines)
    assert "attempt 1/2 FAILED" in joined
    assert "on attempt 2/2" in joined


def test_a_collective_that_completes_with_the_wrong_value_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning is not passing: the reduced value must be the one every rank sent.

    This is the branch that separates "the call returned" from "the fabric
    works". A corrupted reduction is worse than a stall, because training would
    carry it silently.
    """

    def _wrong(tensor: Any, op: Any = None, group: Any = None) -> None:
        tensor.fill_(7)

    _install(monkeypatch, all_reduce=_wrong)
    ok, lines = _warm_up_fabric({})
    assert ok is False
    assert any("WRONG value" in line for line in lines)


def test_the_non_member_sentinel_is_a_failure_rather_than_a_silent_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subgroup this rank is not in cannot be probed, so it must not report success."""
    _install(monkeypatch, new_group=lambda timeout=None, **_: -100)
    ok, lines = _warm_up_fabric({})
    assert ok is False
    assert any("non-member sentinel" in line for line in lines)


def test_nothing_is_probed_before_the_knobs_are_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad declaration is refused without creating a group or touching the fabric."""
    log = _install(monkeypatch)
    ok, _ = _warm_up_fabric({_WARMUP_TIMEOUT_VAR: "later"})
    assert ok is False
    assert log["new_group"] == []
    assert log["all_reduce"] == []
