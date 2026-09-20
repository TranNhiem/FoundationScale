"""The host budget is the line that makes a starved rank legible (#517).

MEASURED 2026-09-20, two GB200 trays, 8 ranks, identical image and model: the
run took 619 s with each rank's affinity mask narrowed to 2 cores and 37 s with
the mask widened to the 128 the cgroup already allowed. 16.6x, entirely on the
host side -- and the narrow run read 82% mean GPU utilisation against the fast
run's 58%, because ``utilization.gpu`` counts a resident kernel and a blocked
collective is a resident kernel. Every assertion here defends one word of the
announcement that would have named it in one second.

These tests substitute ``os.sched_getaffinity`` rather than assume it: it does
not exist on macOS, where development happens, and does exist on the Linux CI
runner. Pinning it explicitly is what makes the same assertions mean the same
thing on both.
"""

from __future__ import annotations

import os

import pytest

from foundationscale.train.loop import _HOST_CORES_PER_RANK_FLOOR, _host_budget


def _pin(monkeypatch: pytest.MonkeyPatch, *, usable: int | None, machine: int | None) -> None:
    """Pin both CPU counts; ``usable=None`` removes sched_getaffinity entirely."""
    monkeypatch.setattr(os, "cpu_count", lambda: machine)
    if usable is None:
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    else:
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(usable)), raising=False)


def test_a_narrowed_affinity_mask_names_both_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gap between usable and installed cores IS the finding, so both are printed."""
    _pin(monkeypatch, usable=2, machine=128)
    lines = _host_budget({"LOCAL_WORLD_SIZE": "4"}, 0)
    joined = " | ".join(lines)
    assert "2 usable of 128" in joined
    assert "narrowed this rank's affinity mask" in joined


def test_an_unnarrowed_mask_says_so_rather_than_staying_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The healthy case still emits. A missing line and a healthy line read alike."""
    _pin(monkeypatch, usable=128, machine=128)
    lines = _host_budget({"LOCAL_WORLD_SIZE": "4"}, 0)
    assert any("128 usable, the whole machine" in line for line in lines)


def test_a_platform_without_sched_getaffinity_says_what_it_cannot_see(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On macOS the mask is unreadable, and reporting the machine count silently would lie."""
    _pin(monkeypatch, usable=None, machine=12)
    lines = _host_budget({}, 0)
    joined = " | ".join(lines)
    assert "no sched_getaffinity" in joined
    assert "would be invisible" in joined


def test_cores_are_divided_by_the_LOCAL_rank_count_not_the_global_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranks on the other node do not share these cores; dividing by 8 would understate it."""
    _pin(monkeypatch, usable=128, machine=128)
    lines = _host_budget({"LOCAL_WORLD_SIZE": "4", "WORLD_SIZE": "8"}, 0)
    assert any("cores per rank: 32 (128 usable / 4 rank(s)" in line for line in lines)


def test_slurm_supplies_the_rank_count_when_torchrun_has_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job launched by srun rather than torchrun still gets a real divisor."""
    _pin(monkeypatch, usable=64, machine=64)
    lines = _host_budget({"SLURM_NTASKS_PER_NODE": "8"}, 0)
    assert any("64 usable / 8 rank(s)" in line for line in lines)


def test_a_lone_rank_divides_by_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """No launcher variables at all is a single-process run, not an error."""
    _pin(monkeypatch, usable=16, machine=16)
    lines = _host_budget({}, 0)
    assert any("16 usable / 1 rank(s)" in line for line in lines)


def test_an_unparseable_rank_count_is_announced_rather_than_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back to 1 makes the figure OPTIMISTIC, so the fallback has to be visible."""
    _pin(monkeypatch, usable=8, machine=8)
    lines = _host_budget({"LOCAL_WORLD_SIZE": "four"}, 0)
    joined = " | ".join(lines)
    assert "is not an integer" in joined
    assert "the real figure is smaller" in joined
    assert "8 usable / 1 rank(s)" in joined


def test_a_starved_rank_is_warned_with_the_measured_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below the floor the line carries the measurement, not an adjective."""
    _pin(monkeypatch, usable=2, machine=128)
    lines = _host_budget({"LOCAL_WORLD_SIZE": "4"}, 0)
    joined = " | ".join(lines)
    assert "BELOW the" in joined
    assert "16.6x slower" in joined
    assert "widen the CPU allocation before suspecting anything else" in joined


def test_a_healthy_rank_is_not_warned(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warning on every run is a warning on no run."""
    _pin(monkeypatch, usable=128, machine=128)
    lines = _host_budget({"LOCAL_WORLD_SIZE": "4"}, 0)
    assert not any("BELOW the" in line for line in lines)


def test_the_floor_is_the_boundary_it_claims_to_be(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly at the floor passes; a hair under does not. Pinned so the constant cannot drift."""
    machine = int(_HOST_CORES_PER_RANK_FLOOR) * 4
    _pin(monkeypatch, usable=machine, machine=machine)
    assert not any("BELOW the" in line for line in _host_budget({"LOCAL_WORLD_SIZE": "4"}, 0))
    _pin(monkeypatch, usable=machine - 1, machine=machine)
    assert any("BELOW the" in line for line in _host_budget({"LOCAL_WORLD_SIZE": "4"}, 0))


def test_zero_dataloader_workers_is_reported_as_the_absence_of_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero is the DEFAULT, which is exactly why it must be stated rather than assumed."""
    _pin(monkeypatch, usable=128, machine=128)
    lines = _host_budget({}, 0)
    joined = " | ".join(lines)
    assert "dataloader workers: 0" in joined
    assert "no prefetch" in joined
    assert "framework default, stated rather than assumed" in joined


def test_configured_dataloader_workers_are_reported_as_prefetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-default case reads differently, so a log can be scanned for either."""
    _pin(monkeypatch, usable=128, machine=128)
    lines = _host_budget({}, 6)
    assert any("dataloader workers: 6 per rank, prefetching" in line for line in lines)


def test_the_budget_is_never_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every path emits a cores line and a workers line; an empty tuple is a bug, not a pass."""
    for usable, machine, environ, workers in (
        (2, 128, {"LOCAL_WORLD_SIZE": "4"}, 0),
        (128, 128, {}, 4),
        (None, 12, {"LOCAL_WORLD_SIZE": "nonsense"}, 2),
    ):
        _pin(monkeypatch, usable=usable, machine=machine)
        lines = _host_budget(environ, workers)
        assert any(line.startswith("cores:") for line in lines)
        assert any(line.startswith("dataloader workers:") for line in lines)
