"""Tests for ``weight_sync.DiskWeightSync`` against a REAL ``EngineFleet`` of fake
SGLang server subprocesses (never a real SGLang engine).

WHAT IS CLAIMED: ``sync(step)`` saves via ``save_fn``, pushes to every fleet
server, and returns a ``SyncReport`` whose ``offered``/``transferred``/
``skipped``/``failed_ranks`` honestly reflect per-server success; a failed
server lands in both ``skipped`` and ``failed_ranks``; ``is_stale`` is always
``False`` and ``bytes_moved`` is always ``None`` for this synchronous
realisation; ``push(path)`` does NOT call ``save_fn`` (the rank-0-only half of
``RolloutHost.publish``'s split); only the ``keep_last`` highest-numbered
``step_<N>`` directories survive under ``publish_root`` after a push;
``capabilities()`` matches the realisation's actual report fields; every
construction-time refusal names the bad field.

``publish_root`` is kept in its OWN subdirectory, separate from ``tmp_path``'s
other use as the fake server fixture's script/log/PYTHONPATH directory: the
latter grows a ``__pycache__`` and a ``.log`` file that must never be mistaken
for a published checkpoint directory by the pruning tests.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from _fake_sglang_fixtures import (
    free_port,
    make_engine_server_spec,
    make_server_spec,
    write_fake_server,
)

from foundationscale.agentic_rl.engines.fleet import EngineFleet, EngineServer, SGLangServer
from foundationscale.agentic_rl.weight_sync import DiskWeightSync, DiskWeightSyncRefusal
from foundationscale.rl.weightsync import SyncCapabilities, SyncReport


@pytest.fixture
def fleet(tmp_path: Path) -> Generator[EngineFleet, None, None]:
    fake_module = write_fake_server(tmp_path)
    specs = [make_server_spec(tmp_path, fake_module, port=free_port()) for _ in range(2)]
    servers = tuple(SGLangServer(spec) for spec in specs)
    built = EngineFleet(servers=servers)
    built.start_all()
    yield built
    built.stop_all()


@pytest.fixture
def publish_root(tmp_path: Path) -> str:
    root = tmp_path / "publish"
    root.mkdir()
    return str(root)


def _save_fn(calls: list[str]) -> Callable[[str], None]:
    def _save(path: str) -> None:
        calls.append(path)
        Path(path).mkdir(parents=True, exist_ok=True)

    return _save


# ---------------------------------------------------------------------------
# Construction refusals
# ---------------------------------------------------------------------------


def test_rejects_non_callable_save_fn(fleet: EngineFleet, publish_root: str) -> None:
    with pytest.raises(DiskWeightSyncRefusal):
        DiskWeightSync(save_fn="not callable", publish_root=publish_root, fleet=fleet)  # type: ignore[arg-type]


def test_accepts_a_none_save_fn_for_a_publish_only_caller(
    fleet: EngineFleet, publish_root: str
) -> None:
    sync = DiskWeightSync(publish_root=publish_root, fleet=fleet)
    assert sync.save_fn is None


def test_sync_refuses_when_save_fn_is_none(fleet: EngineFleet, publish_root: str) -> None:
    sync = DiskWeightSync(publish_root=publish_root, fleet=fleet)
    with pytest.raises(DiskWeightSyncRefusal, match="save_fn"):
        sync.sync(0)


def test_rejects_empty_publish_root(fleet: EngineFleet) -> None:
    with pytest.raises(DiskWeightSyncRefusal):
        DiskWeightSync(save_fn=lambda path: None, publish_root="", fleet=fleet)


def test_rejects_non_fleet(publish_root: str) -> None:
    with pytest.raises(DiskWeightSyncRefusal):
        DiskWeightSync(save_fn=lambda path: None, publish_root=publish_root, fleet="not a fleet")  # type: ignore[arg-type]


def test_rejects_non_positive_keep_last(fleet: EngineFleet, publish_root: str) -> None:
    with pytest.raises(DiskWeightSyncRefusal):
        DiskWeightSync(
            save_fn=lambda path: None, publish_root=publish_root, fleet=fleet, keep_last=0
        )


def test_rejects_an_undeclared_mode(fleet: EngineFleet, publish_root: str) -> None:
    with pytest.raises(DiskWeightSyncRefusal, match="field 'mode'"):
        DiskWeightSync(
            save_fn=lambda path: None,
            publish_root=publish_root,
            fleet=fleet,
            mode="reload_from_memory",  # type: ignore[arg-type]
        )


def test_default_mode_is_reload_endpoint(fleet: EngineFleet, publish_root: str) -> None:
    sync = DiskWeightSync(save_fn=lambda path: None, publish_root=publish_root, fleet=fleet)
    assert sync.mode == "reload_endpoint"


# ---------------------------------------------------------------------------
# path_for
# ---------------------------------------------------------------------------


def test_path_for_pads_step_to_six_digits(fleet: EngineFleet, publish_root: str) -> None:
    sync = DiskWeightSync(save_fn=lambda path: None, publish_root=publish_root, fleet=fleet)
    assert sync.path_for(7) == f"{publish_root}/step_000007"


def test_path_for_rejects_negative_step(fleet: EngineFleet, publish_root: str) -> None:
    sync = DiskWeightSync(save_fn=lambda path: None, publish_root=publish_root, fleet=fleet)
    with pytest.raises(DiskWeightSyncRefusal):
        sync.path_for(-1)


# ---------------------------------------------------------------------------
# capabilities()
# ---------------------------------------------------------------------------


def test_capabilities_claims_disk_transport_and_both_reports(
    fleet: EngineFleet, publish_root: str
) -> None:
    sync = DiskWeightSync(save_fn=lambda path: None, publish_root=publish_root, fleet=fleet)
    caps = sync.capabilities()
    assert isinstance(caps, SyncCapabilities)
    assert caps.transports == ("disk",)
    assert caps.reports_per_rank_failure is True
    assert caps.reports_staleness is True


# ---------------------------------------------------------------------------
# sync() / push()
# ---------------------------------------------------------------------------


def test_sync_saves_then_pushes_and_reports_success(fleet: EngineFleet, publish_root: str) -> None:
    calls: list[str] = []
    sync = DiskWeightSync(
        save_fn=_save_fn(calls), publish_root=publish_root, fleet=fleet, keep_last=2
    )
    report = sync.sync(3)
    assert calls == [f"{publish_root}/step_000003"]
    assert isinstance(report, SyncReport)
    assert report.transport == "disk"
    assert report.offered == ("server_0", "server_1")
    assert report.transferred == ("server_0", "server_1")
    assert report.skipped == ()
    assert report.failed_ranks == ()
    assert report.bytes_moved is None
    assert report.seconds is not None and report.seconds >= 0.0
    assert report.is_stale is False
    assert report.complete is True


def test_push_does_not_call_save_fn(fleet: EngineFleet, publish_root: str) -> None:
    calls: list[str] = []
    sync = DiskWeightSync(save_fn=_save_fn(calls), publish_root=publish_root, fleet=fleet)
    path = sync.path_for(1)
    Path(path).mkdir(parents=True)
    report = sync.push(path)
    assert calls == []
    assert report.transferred == ("server_0", "server_1")


def test_push_rejects_empty_path(fleet: EngineFleet, publish_root: str) -> None:
    sync = DiskWeightSync(save_fn=lambda path: None, publish_root=publish_root, fleet=fleet)
    with pytest.raises(DiskWeightSyncRefusal):
        sync.push("")


def test_sync_records_a_failed_server_in_skipped_and_failed_ranks(
    fleet: EngineFleet, publish_root: str
) -> None:
    sync = DiskWeightSync(save_fn=lambda path: None, publish_root=publish_root, fleet=fleet)
    # The fake server fails any model_path containing "FAIL" -- see
    # _fake_sglang_fixtures.FAKE_SERVER_SOURCE's /update_weights_from_disk handler.
    report = sync.push(f"{publish_root}/step_FAIL_000001")
    assert report.failed_ranks == (0, 1)
    assert report.skipped == ("server_0", "server_1")
    assert report.transferred == ()
    assert report.complete is False
    assert report.partial is True


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------


def test_sync_keeps_only_the_last_keep_last_published_dirs(
    fleet: EngineFleet, publish_root: str
) -> None:
    calls: list[str] = []
    sync = DiskWeightSync(
        save_fn=_save_fn(calls), publish_root=publish_root, fleet=fleet, keep_last=2
    )
    for step in range(4):
        sync.sync(step)
    remaining = sorted(p.name for p in Path(publish_root).iterdir() if p.is_dir())
    assert remaining == ["step_000002", "step_000003"]


def test_prune_ignores_non_matching_entries(fleet: EngineFleet, publish_root: str) -> None:
    (Path(publish_root) / "not_a_step_dir").mkdir()
    (Path(publish_root) / "a_file.txt").write_text("hello")
    sync = DiskWeightSync(save_fn=_save_fn([]), publish_root=publish_root, fleet=fleet, keep_last=1)
    sync.sync(0)
    sync.sync(1)
    assert (Path(publish_root) / "not_a_step_dir").is_dir()
    assert (Path(publish_root) / "a_file.txt").is_file()
    assert (Path(publish_root) / "step_000000").exists() is False
    assert (Path(publish_root) / "step_000001").is_dir()


def test_prune_refuses_to_touch_a_symlinked_step_dir(
    fleet: EngineFleet, publish_root: str, tmp_path: Path
) -> None:
    # Security regression/hardening: a step_<N>-named entry that is actually a
    # SYMLINK must never be followed or deleted. Before the fix, child.is_dir()
    # silently followed the symlink (matching it into the prune candidates) and
    # shutil.rmtree(directory) was then called directly on that symlink path
    # (raising, rather than being refused up front and skipped); after the fix
    # every candidate is resolved and confirmed to be a real directory whose own
    # resolved path is a direct child of publish_root, confining every deletion
    # there and never touching this symlink or whatever it points at.
    outside = tmp_path / "outside_target"
    outside.mkdir()
    canary = outside / "canary.txt"
    canary.write_text("do not delete me")
    (Path(publish_root) / "step_000001").mkdir()
    (Path(publish_root) / "step_000002").mkdir()
    (Path(publish_root) / "step_000005").symlink_to(outside, target_is_directory=True)

    sync = DiskWeightSync(save_fn=_save_fn([]), publish_root=publish_root, fleet=fleet, keep_last=1)
    sync._prune()  # must not raise

    assert (Path(publish_root) / "step_000005").is_symlink()
    assert canary.exists()
    assert canary.read_text() == "do not delete me"
    # The two REAL step dirs are still pruned normally: only the highest-numbered
    # (keep_last=1) survives -- the symlink is simply ignored, not counted.
    assert not (Path(publish_root) / "step_000001").exists()
    assert (Path(publish_root) / "step_000002").is_dir()


def test_prune_is_a_noop_when_publish_root_does_not_exist(
    fleet: EngineFleet, tmp_path: Path
) -> None:
    missing_root = str(tmp_path / "does_not_exist_yet")
    sync = DiskWeightSync(save_fn=lambda path: None, publish_root=missing_root, fleet=fleet)
    sync._prune()  # must not raise


# ---------------------------------------------------------------------------
# mode="restart": stop + relaunch with {model_path} substituted
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_fleet(tmp_path: Path) -> Generator[EngineFleet, None, None]:
    fake_module = write_fake_server(tmp_path)
    specs = [make_engine_server_spec(tmp_path, fake_module, port=free_port()) for _ in range(2)]
    servers = tuple(EngineServer(spec) for spec in specs)
    built = EngineFleet(servers=servers)
    built.start_all()
    yield built
    built.stop_all()


def test_restart_mode_relaunches_every_server_with_the_checkpoint_substituted(
    engine_fleet: EngineFleet, publish_root: str
) -> None:
    sync = DiskWeightSync(
        save_fn=lambda path: None, publish_root=publish_root, fleet=engine_fleet, mode="restart"
    )
    report = sync.push("/ckpt/step_000001")
    assert report.transport == "disk"
    assert report.complete is True
    assert report.transferred == ("server_0", "server_1")
    for server in engine_fleet.servers:
        assert isinstance(server, EngineServer)
        # server.spec is NEVER mutated by a restart -- the declared template
        # (placeholder included) survives for the next restart too.
        assert "{model_path}" in server.spec.command
        assert server.client().health() is True

    # A second restart against the SAME (unmutated) template must also work --
    # the placeholder was not consumed by the first substitution.
    report2 = sync.push("/ckpt/step_000002")
    assert report2.complete is True
    for server in engine_fleet.servers:
        assert isinstance(server, EngineServer)
        assert server.client().health() is True


def test_restart_mode_records_a_crashed_restart_in_skipped_and_failed_ranks(
    engine_fleet: EngineFleet, publish_root: str
) -> None:
    sync = DiskWeightSync(
        save_fn=lambda path: None, publish_root=publish_root, fleet=engine_fleet, mode="restart"
    )
    # The fake server crashes immediately whenever --model-path contains "FAIL"
    # (see _fake_sglang_fixtures.FAKE_SERVER_SOURCE) -- the same convention
    # reload-mode's /update_weights_from_disk failure test already uses.
    report = sync.push("/ckpt/FAIL_step")
    assert report.failed_ranks == (0, 1)
    assert report.skipped == ("server_0", "server_1")
    assert report.transferred == ()
    assert report.complete is False
    assert report.partial is True


def test_restart_mode_refuses_a_fleet_of_legacy_sglang_servers(
    fleet: EngineFleet, publish_root: str
) -> None:
    # `fleet` (the module-level fixture) is built from plain SGLangServer
    # instances, which have no declared command / {model_path} placeholder.
    sync = DiskWeightSync(
        save_fn=lambda path: None, publish_root=publish_root, fleet=fleet, mode="restart"
    )
    with pytest.raises(DiskWeightSyncRefusal, match="server_0"):
        sync.push("/ckpt/step_000001")


def test_restart_mode_refuses_a_command_with_no_placeholder(
    tmp_path: Path, publish_root: str
) -> None:
    fake_module = write_fake_server(tmp_path)
    spec = make_engine_server_spec(
        tmp_path, fake_module, port=free_port(), model_path="/fixed/model"
    )
    fleet = EngineFleet(servers=(EngineServer(spec),))
    fleet.start_all()
    try:
        sync = DiskWeightSync(
            save_fn=lambda path: None, publish_root=publish_root, fleet=fleet, mode="restart"
        )
        with pytest.raises(DiskWeightSyncRefusal, match="server_0"):
            sync.push("/ckpt/step_000001")
    finally:
        fleet.stop_all()
