"""Policy-version ledger and ``recovery_rerun='engine_rows'`` tests for ``RolloutHost``.

The fakes and fixtures of ``test_engine_recovery.py`` are REUSED (imported, never
re-invented): a fake fleet serving mode-driven clients, the real ``NativeToolLoop`` with a
toy tokenizer, the real local ``EnvBackend`` and a toy reward -- no engine is started.

WHAT IS CLAIMED: every session records ``policy_version_min/max`` equal to the version of
the SERVER it ran on -- the per-server ledger (``_state["server_versions"]``) seeded at
``fleet_launch_version`` (0) and advanced ONLY by push reports for the servers their
``.transferred`` names (a server the report does not mention keeps its old version;
``report is None`` re-launches every server at ``fleet_launch_version`` again);
``_state["last_step_versions"]`` publishes exactly those measured per-row versions and is
``()`` for a ``client_factory`` host whose routing is opaque; a
``recovery_rerun='engine_rows'`` retry re-runs only the ENGINE infra rows of the failed
attempt (one client == one episode, counted per session) and keeps every measured row as
the very same object, while ``_merge_recovered`` refuses when kept and re-run rows would
carry different policy versions; a bogus ``recovery_rerun`` and a bool/negative
``fleet_launch_version`` are refused at construction; ``rollout.recovery_rerun`` defaults
to ``'all'`` and any other config value raises ``AgenticConfigRefusal``.

WHAT IS NOT CLAIMED: no engine reload, no checkpoint save and no real HTTP push happens --
the ledger is fed from the same ``SyncReport`` shapes ``test_engine_recovery`` builds and
``DiskWeightSync.push`` is monkeypatched wherever a recovery push runs.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.agentic_rl import rollout_host as rh
from foundationscale.agentic_rl.config import AgenticConfigRefusal, RolloutConfig
from foundationscale.agentic_rl.rollout_host import RolloutHost, RolloutHostRefusal

try:  # pytest's default collection puts this file's directory on sys.path
    import test_engine_recovery as fakes
except ImportError:  # collected inside an importable tests package
    from tests.agentic_rl import test_engine_recovery as fakes


class LedgerFleet(fakes.FakeFleet):
    """``FakeFleet`` plus the two knobs the version ledger and (d) read.

    ``servers`` names the push targets (``DiskWeightSync`` addresses them ``"server_<i>"``,
    the shape ``_record_push`` parses the index off) -- ``FakeFleet`` predates the ledger
    and never exposed them; ``client_for`` counts one episode per call (``_run_episode``
    builds exactly one client per session), which is (d)'s "was this session re-run?"
    counter. ``healthy=True`` serves measured rows from the first attempt.
    """

    def __init__(
        self,
        mode: fakes.FleetMode,
        *,
        healthy: bool = False,
        hung_indices: set[int] | None = None,
        names: Sequence[str] = ("server_0", "server_1"),
    ) -> None:
        super().__init__(mode, hung_indices)
        self.servers = tuple(names)
        self.healthy = healthy
        self.episodes: dict[int, int] = {}

    def is_hung(self, index: int) -> bool:
        return not self.healthy and super().is_hung(index)

    def client_for(self, index: int) -> Any:
        self.episodes[index] = self.episodes.get(index, 0) + 1
        return super().client_for(index)


# ---------------------------------------------------------------------------
# (a) before any publish every session is served the fleet launch version (0).
# ---------------------------------------------------------------------------


def test_sessions_record_the_fleet_launch_version_before_any_publish(tmp_path: Path) -> None:
    fleet = LedgerFleet(fakes.FleetMode(cured_by_restart=True), healthy=True)
    host = fakes._host(tmp_path, uids=["taskA"], client_factory=None, fleet=fleet)
    assert host.fleet_launch_version == 0
    assert host._server_versions() == [0, 0]  # seeded from fleet_launch_version on first use
    batch = host.rollout(5)  # "step 5" is ASSERTED, not measured: nothing was published yet
    assert tuple(batch.columns["policy_version_min"]) == (0, 0)
    assert tuple(batch.columns["policy_version_max"]) == (0, 0)
    assert host._state["last_step_versions"] == (0, 0)
    assert [host._session_version(i, 5) for i in (0, 1, 2)] == [0, 0, 0]


# ---------------------------------------------------------------------------
# (b) `_record_push` advances exactly the servers the report transferred.
# ---------------------------------------------------------------------------


def test_a_push_report_advances_only_the_servers_it_transferred(tmp_path: Path) -> None:
    fleet = LedgerFleet(fakes.FleetMode(cured_by_restart=True), healthy=True)
    host = fakes._host(tmp_path, uids=["taskA"], client_factory=None, fleet=fleet)
    host._record_push(fakes._report(complete=True), 3)  # transferred = ("server_0", "server_1")
    assert host._server_versions() == [3, 3]
    assert [host._session_version(i, 9) for i in (0, 1, 2)] == [3, 3, 3]  # rounds over the ledger
    host._record_push(fakes._report(complete=False), 7)  # transferred = ("server_0",) only
    assert host._server_versions() == [7, 3]  # server_1 NOT in .transferred: keeps version 3
    assert [host._session_version(i, 9) for i in (0, 1)] == [7, 3]
    host._record_push(
        None, 99
    )  # a restart that re-pushed nothing re-launches the version-0 weights
    assert host._server_versions() == [0, 0]


# ---------------------------------------------------------------------------
# (c) `last_step_versions` is the measured per-row ledger reading -- and () for a
#     client_factory host, whose sessions cannot be mapped onto named servers.
# ---------------------------------------------------------------------------


def test_last_step_versions_are_the_measured_per_row_versions(tmp_path: Path) -> None:
    fleet = LedgerFleet(fakes.FleetMode(cured_by_restart=True), healthy=True)
    host = fakes._host(tmp_path, uids=["taskA"], client_factory=None, fleet=fleet)
    host._record_push(fakes._report(complete=False), 3)  # server_0 -> 3, server_1 stays 0
    batch = asyncio.run(host.rollout_async(0))
    per_row_min = tuple(batch.columns["policy_version_min"])
    per_row_max = tuple(batch.columns["policy_version_max"])
    assert per_row_min == per_row_max == (3, 0)  # session i ran on server i
    assert host._state["last_step_versions"] == per_row_min


def test_last_step_versions_are_empty_for_a_client_factory_host(tmp_path: Path) -> None:
    host = fakes._host(
        tmp_path,
        uids=["taskA"],
        fleet=None,
        client_factory=lambda index: fakes.SubmitClient("x"),
    )
    assert host._server_versions() is None  # no ledger: this host routes opaquely
    asyncio.run(host.rollout_async(0))
    assert host._state["last_step_versions"] == ()  # unmeasured -- never an asserted step
    assert host._session_version(0, 4) == 4  # the legacy assert-the-step fallback


# ---------------------------------------------------------------------------
# (d) recovery_rerun='engine_rows': only the ENGINE infra rows of the failed
#     attempt are re-run; the measured rows survive as the very same objects.
# ---------------------------------------------------------------------------


def test_engine_rows_recovery_re_runs_only_the_engine_infra_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet = LedgerFleet(fakes.FleetMode(cured_by_restart=True), hung_indices={1})
    driver, _pushed = fakes._recovering_host(  # its push is already monkeypatched away
        tmp_path, monkeypatch, fleet, mode="reload_endpoint", complete=True
    )
    host = fakes._host(
        tmp_path,
        uids=["taskA"],
        client_factory=None,
        fleet=fleet,
        weight_sync=driver.weight_sync,
        engine_recovery_retries=1,
        recovery_rerun="engine_rows",
        declared_max_infra_rate=0.4,  # 1 of 2 engine rows is above it -> the step is recovered
        gates=False,
    )
    host._state["last_pushed_path"] = driver._state["last_pushed_path"]
    gathers: list[tuple[frozenset[int] | None, list[Any]]] = []
    merges: list[tuple[list[Any], list[Any], list[int], list[Any]]] = []
    orig_gather, orig_merge = RolloutHost._gather_step, RolloutHost._merge_recovered

    async def gather_spy(self: RolloutHost, step: int, tasks: Sequence[Any], only: Any = None):
        out = await orig_gather(self, step, tasks, only)
        gathers.append((only, out))
        return out

    def merge_spy(
        self: RolloutHost, step: int, previous: list[Any], rerun: list[Any], indices: list[int]
    ):
        out = orig_merge(self, step, previous, rerun, indices)
        merges.append((previous, rerun, indices, out))
        return out

    monkeypatch.setattr(RolloutHost, "_gather_step", gather_spy)
    monkeypatch.setattr(RolloutHost, "_merge_recovered", merge_spy)
    batch = host.rollout(0)

    first, retry = gathers
    assert first[0] is None and len(first[1]) == 2  # every attempt of the step, once
    assert rh._is_engine_infra(first[1][1]) and first[1][1].abstention_reason.startswith("infra:")
    assert not rh._is_engine_infra(first[1][0])  # a MEASURED row must not be re-run
    assert retry[0] == frozenset({1}) and len(retry[1]) == 1  # only the engine row's index
    previous, rerun, indices, merged = merges[0]
    assert indices == [1]
    assert merged[0] is previous[0]  # the measured row is kept as the very same object
    assert merged[1] is rerun[0]
    assert fleet.episodes == {0: 1, 1: 2}  # session 0 once (kept), session 1 twice (re-run)
    assert all(reward is not None for reward in batch.columns["reward"])


# ---------------------------------------------------------------------------
# (e) `_merge_recovered` refuses a batch whose kept and re-run rows disagree on
#     the served policy version (one batch, one policy).
# ---------------------------------------------------------------------------


def test_merge_recovered_refuses_a_batch_mixing_policy_versions(tmp_path: Path) -> None:
    fleet = LedgerFleet(fakes.FleetMode(cured_by_restart=True), healthy=True)
    host = fakes._host(tmp_path, uids=["taskA"], client_factory=None, fleet=fleet)

    def row(version: int) -> Any:
        return SimpleNamespace(policy_version_min=version, policy_version_max=version)

    fresh = row(2)  # a recovery re-pushes the SAME checkpoint: one version for one batch
    merged = host._merge_recovered(0, [row(2), row(2)], [fresh], [1])
    assert merged[1] is fresh and [r.policy_version_min for r in merged] == [2, 2]
    with pytest.raises(RolloutHostRefusal, match="mix"):
        host._merge_recovered(0, [row(2), row(2)], [row(3)], [1])  # the re-push moved server_1


# ---------------------------------------------------------------------------
# (f) construction refuses a bogus `recovery_rerun` and a non-real-int/negative
#     `fleet_launch_version`, naming the field each time.
# ---------------------------------------------------------------------------


def test_bad_recovery_rerun_or_fleet_launch_version_is_refused_at_construction(
    tmp_path: Path,
) -> None:
    with pytest.raises(RolloutHostRefusal, match="recovery_rerun"):
        fakes._host(
            tmp_path,
            uids=["taskA"],
            client_factory=lambda index: fakes.SubmitClient("x"),
            recovery_rerun="bogus",
        )
    for bad in (-1, True, 2.0, "0"):
        with pytest.raises(RolloutHostRefusal, match="fleet_launch_version"):
            fakes._host(
                tmp_path,
                uids=["taskA"],
                client_factory=lambda index: fakes.SubmitClient("x"),
                fleet_launch_version=bad,
            )


# ---------------------------------------------------------------------------
# (g) config: `rollout.recovery_rerun` defaults to 'all', accepts 'engine_rows',
#     and anything else is refused by the config loader.
# ---------------------------------------------------------------------------


def _load(tmp_path: Path, **rollout: object) -> Any:
    """Build a config through the public ``build_config`` from test_config's raw
    document, with ``rollout`` keys merged into its rollout section."""
    import importlib.util

    from foundationscale.agentic_rl.config import build_config

    spec = importlib.util.spec_from_file_location(
        "_test_config_raw", Path(__file__).with_name("test_config.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = module._raw_config()
    raw["rollout"] = {**raw["rollout"], **rollout}
    return build_config(raw, {})


def test_rollout_recovery_rerun_defaults_to_all(tmp_path: Path) -> None:
    fields = {f.name: f for f in dataclasses.fields(RolloutConfig)}
    assert fields["recovery_rerun"].default == "all"  # what the dataclass falls back to
    assert _load(tmp_path).rollout.recovery_rerun == "all"  # and what a bare config loads


def test_rollout_recovery_rerun_accepts_engine_rows(tmp_path: Path) -> None:
    loaded = _load(tmp_path, recovery_rerun="engine_rows")
    assert loaded.rollout.recovery_rerun == "engine_rows"


def test_rollout_recovery_rerun_refuses_anything_else(tmp_path: Path) -> None:
    for bad in ("ALL", "engine_ro", "per_row", "engine_rows ", ""):
        with pytest.raises(AgenticConfigRefusal, match="recovery_rerun"):
            _load(tmp_path, recovery_rerun=bad)
