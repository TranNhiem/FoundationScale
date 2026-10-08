"""Tests for ``engines.fleet`` (``SGLangServerSpec``, ``SGLangServer``,
``EngineFleet``) against a REAL subprocess running a tiny fake HTTP server --
never a real SGLang engine (``sglang`` is not installed and is never imported).

The fake server is a standalone script written to ``tmp_path`` and launched
exactly the way ``SGLangServer.start`` launches a real one: ``python_bin -m
<launch_module> --model-path ... --port ... --tp-size ...``, with
``launch_module`` overridden to the fake script's module name and
``env={"PYTHONPATH": str(tmp_path)}`` (an explicit allow-list, per
``SGLangServerSpec.env``) making it importable via ``-m``.

WHAT IS CLAIMED: a server starts, answers ``/health``, and is reachable via
``SGLangServer.client()``; ``stop()`` terminates it; a script that exits
immediately yields ``EngineInfraError(kind="startup_crashed")``; a script
whose ``/health`` never returns 200 yields ``kind="startup_timeout"`` within
the declared budget; ``EngineFleet.client_for`` round-robins by session index;
``EngineFleet.update_weights_from_disk`` runs every server concurrently and
reports per-server success/failure; construction-time refusals name the bad
field.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest
from _fake_sglang_fixtures import (
    free_port,
    make_engine_server_spec,
    make_server_spec,
    write_fake_server,
)

from foundationscale.agentic_rl.engines import EngineInfraError, EngineRefusal
from foundationscale.agentic_rl.engines.fleet import (
    EngineFleet,
    EngineServer,
    EngineServerSpec,
    SGLangServer,
    SGLangServerSpec,
)
from foundationscale.agentic_rl.engines.sglang import SGLangClient
from foundationscale.agentic_rl.engines.vllm import VLLMClient


@pytest.fixture
def fake_module(tmp_path: Path) -> str:
    return write_fake_server(tmp_path)


def _spec(tmp_path: Path, fake_module: str, *, port: int, **overrides: object) -> SGLangServerSpec:
    return make_server_spec(tmp_path, fake_module, port=port, **overrides)


def _free_port() -> int:
    return free_port()


# ---------------------------------------------------------------------------
# SGLangServerSpec construction refusals
# ---------------------------------------------------------------------------


def test_spec_rejects_empty_model_path(tmp_path: Path) -> None:
    with pytest.raises(EngineRefusal):
        SGLangServerSpec(
            model_path="", port=1234, tp_size=1, mem_fraction_static=None, log_dir=str(tmp_path)
        )


def test_spec_rejects_non_positive_port(tmp_path: Path) -> None:
    with pytest.raises(EngineRefusal):
        SGLangServerSpec(
            model_path="/m", port=0, tp_size=1, mem_fraction_static=None, log_dir=str(tmp_path)
        )


def test_spec_rejects_bad_mem_fraction_static(tmp_path: Path) -> None:
    with pytest.raises(EngineRefusal):
        SGLangServerSpec(
            model_path="/m",
            port=1234,
            tp_size=1,
            mem_fraction_static=1.5,
            log_dir=str(tmp_path),
        )


def test_spec_rejects_non_str_env_values(tmp_path: Path) -> None:
    with pytest.raises(EngineRefusal):
        SGLangServerSpec(
            model_path="/m",
            port=1234,
            tp_size=1,
            mem_fraction_static=None,
            log_dir=str(tmp_path),
            env={"X": 5},  # type: ignore[dict-item]
        )


def test_spec_rejects_non_str_extra_args(tmp_path: Path) -> None:
    with pytest.raises(EngineRefusal):
        SGLangServerSpec(
            model_path="/m",
            port=1234,
            tp_size=1,
            mem_fraction_static=None,
            log_dir=str(tmp_path),
            extra_args=(5,),  # type: ignore[arg-type]
        )


def test_spec_rejects_non_positive_startup_timeout(tmp_path: Path) -> None:
    with pytest.raises(EngineRefusal):
        SGLangServerSpec(
            model_path="/m",
            port=1234,
            tp_size=1,
            mem_fraction_static=None,
            log_dir=str(tmp_path),
            startup_timeout_s=0.0,
        )


def test_spec_accepts_valid_fields_and_freezes_env(tmp_path: Path) -> None:
    spec = SGLangServerSpec(
        model_path="/m",
        port=1234,
        tp_size=2,
        mem_fraction_static=0.5,
        log_dir=str(tmp_path),
        env={"A": "B"},
    )
    assert spec.env == {"A": "B"}
    assert spec.launch_module == "sglang.launch_server"
    assert spec.host == "127.0.0.1"


# ---------------------------------------------------------------------------
# SGLangServer lifecycle
# ---------------------------------------------------------------------------


def test_server_starts_health_checks_and_stops(tmp_path: Path, fake_module: str) -> None:
    spec = _spec(tmp_path, fake_module, port=_free_port())
    server = SGLangServer(spec)
    try:
        server.start()
        assert server.log_path is not None
        assert Path(server.log_path).exists()
        client = server.client()
        assert isinstance(client, SGLangClient)
        assert client.health() is True
    finally:
        server.stop()
    assert server.client().health() is False


def test_server_start_twice_refuses(tmp_path: Path, fake_module: str) -> None:
    spec = _spec(tmp_path, fake_module, port=_free_port())
    server = SGLangServer(spec)
    try:
        server.start()
        with pytest.raises(EngineRefusal):
            server.start()
    finally:
        server.stop()


def test_server_crash_before_healthy_raises_startup_crashed(
    tmp_path: Path, fake_module: str
) -> None:
    spec = _spec(
        tmp_path,
        fake_module,
        port=_free_port(),
        extra_args=("--crash-immediately",),
        startup_timeout_s=5.0,
    )
    server = SGLangServer(spec)
    with pytest.raises(EngineInfraError) as excinfo:
        server.start()
    assert excinfo.value.kind == "startup_crashed"


def test_server_never_healthy_times_out(tmp_path: Path, fake_module: str) -> None:
    spec = _spec(
        tmp_path,
        fake_module,
        port=_free_port(),
        extra_args=("--never-healthy",),
        startup_timeout_s=0.3,
        health_poll_interval_s=0.05,
    )
    server = SGLangServer(spec)
    try:
        with pytest.raises(EngineInfraError) as excinfo:
            server.start()
        assert excinfo.value.kind == "startup_timeout"
    finally:
        server.stop()


def test_server_stop_is_idempotent(tmp_path: Path, fake_module: str) -> None:
    spec = _spec(tmp_path, fake_module, port=_free_port())
    server = SGLangServer(spec)
    server.stop()  # never started: must not raise
    server.start()
    server.stop()
    server.stop()  # already stopped: must not raise


def test_server_start_passes_mem_fraction_static(tmp_path: Path, fake_module: str) -> None:
    spec = _spec(tmp_path, fake_module, port=_free_port(), mem_fraction_static=0.5)
    server = SGLangServer(spec)
    try:
        server.start()
        assert server.client().health() is True
    finally:
        server.stop()


def test_server_stop_kills_a_process_that_ignores_sigterm(tmp_path: Path, fake_module: str) -> None:
    spec = _spec(
        tmp_path,
        fake_module,
        port=_free_port(),
        extra_args=("--ignore-sigterm",),
    )
    server = SGLangServer(spec)
    server.start()
    server.stop(grace_s=0.2)
    assert server.client().health() is False


def test_server_stop_kills_the_whole_process_group_including_children(
    tmp_path: Path, fake_module: str
) -> None:
    # Security/correctness regression: start() must launch the server in its own
    # session (start_new_session=True) and stop() must target the whole process
    # GROUP (os.killpg) -- a server that forks tensor-parallel worker processes
    # must not have them orphaned by a stop() that only signals its one pid.
    pidfile = tmp_path / "child.pid"
    spec = _spec(
        tmp_path,
        fake_module,
        port=_free_port(),
        extra_args=("--spawn-child-pidfile", str(pidfile)),
    )
    server = SGLangServer(spec)
    server.start()
    try:
        deadline = time.monotonic() + 5.0
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pidfile.exists(), "fake server never wrote the spawned child's pidfile"
        child_pid = int(pidfile.read_text().strip())
        os.kill(child_pid, 0)  # the child is alive before stop()
    finally:
        server.stop()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"child pid {child_pid} still alive after server.stop()")


# ---------------------------------------------------------------------------
# EngineFleet
# ---------------------------------------------------------------------------


def test_fleet_rejects_empty_servers() -> None:
    with pytest.raises(EngineRefusal):
        EngineFleet(servers=())


def test_fleet_rejects_non_server_entries() -> None:
    with pytest.raises(EngineRefusal):
        EngineFleet(servers=("not a server",))  # type: ignore[arg-type]


def test_fleet_client_for_round_robins(tmp_path: Path, fake_module: str) -> None:
    specs = [_spec(tmp_path, fake_module, port=_free_port()) for _ in range(3)]
    servers = tuple(SGLangServer(spec) for spec in specs)
    fleet = EngineFleet(servers=servers)
    for index in range(6):
        client = fleet.client_for(index)
        assert client.base_url == servers[index % 3].base_url


def test_fleet_client_for_rejects_bad_session_index(tmp_path: Path, fake_module: str) -> None:
    fleet = EngineFleet(servers=(SGLangServer(_spec(tmp_path, fake_module, port=_free_port())),))
    with pytest.raises(EngineRefusal):
        fleet.client_for(-1)


def test_fleet_start_all_and_stop_all(tmp_path: Path, fake_module: str) -> None:
    specs = [_spec(tmp_path, fake_module, port=_free_port()) for _ in range(2)]
    servers = tuple(SGLangServer(spec) for spec in specs)
    fleet = EngineFleet(servers=servers)
    fleet.start_all()
    try:
        for server in servers:
            assert server.client().health() is True
    finally:
        fleet.stop_all()
    for server in servers:
        assert server.client().health() is False


def test_fleet_update_weights_from_disk_reports_per_server_success(
    tmp_path: Path, fake_module: str
) -> None:
    specs = [_spec(tmp_path, fake_module, port=_free_port()) for _ in range(2)]
    servers = tuple(SGLangServer(spec) for spec in specs)
    fleet = EngineFleet(servers=servers)
    fleet.start_all()
    try:
        results = asyncio.run(fleet.update_weights_from_disk("/ckpt/step_000001"))
        assert results == (True, True)
        results = asyncio.run(fleet.update_weights_from_disk("/ckpt/FAIL_step"))
        assert results == (False, False)
    finally:
        fleet.stop_all()


# ---------------------------------------------------------------------------
# EngineServerSpec construction refusals
# ---------------------------------------------------------------------------


def _engine_spec(tmp_path: Path, fake_module: str, **overrides: object) -> EngineServerSpec:
    return make_engine_server_spec(tmp_path, fake_module, port=_free_port(), **overrides)  # type: ignore[arg-type]


def test_engine_spec_rejects_an_unknown_kind(tmp_path: Path, fake_module: str) -> None:
    with pytest.raises(EngineRefusal, match="field 'kind'"):
        _engine_spec(tmp_path, fake_module, kind="tgi")


def test_engine_spec_rejects_a_bare_str_command(tmp_path: Path, fake_module: str) -> None:
    with pytest.raises(EngineRefusal, match="field 'command'"):
        _engine_spec(tmp_path, fake_module, command="python -m vllm")


def test_engine_spec_rejects_an_empty_command(tmp_path: Path, fake_module: str) -> None:
    with pytest.raises(EngineRefusal, match="field 'command'"):
        _engine_spec(tmp_path, fake_module, command=())


def test_engine_spec_rejects_a_non_str_command_entry(tmp_path: Path, fake_module: str) -> None:
    with pytest.raises(EngineRefusal, match="field 'command' entry"):
        _engine_spec(tmp_path, fake_module, command=("python3", 5))


def test_engine_spec_rejects_non_str_env_values(tmp_path: Path, fake_module: str) -> None:
    with pytest.raises(EngineRefusal, match="field 'env'"):
        _engine_spec(tmp_path, fake_module, env={"X": 5})


def test_engine_spec_rejects_a_non_str_served_model_name(tmp_path: Path, fake_module: str) -> None:
    with pytest.raises(EngineRefusal, match="field 'served_model_name'"):
        _engine_spec(tmp_path, fake_module, kind="sglang", served_model_name=5)


def test_engine_spec_requires_served_model_name_for_vllm(tmp_path: Path) -> None:
    # Constructed directly (not via the fixture helper, whose own convenience
    # default would fill in a served_model_name for kind="vllm"): this checks
    # EngineServerSpec's OWN refusal when neither the caller nor a fixture
    # supplies one.
    with pytest.raises(EngineRefusal, match="served_model_name"):
        EngineServerSpec(
            kind="vllm",
            command=("python3", "-m", "vllm.entrypoints.openai.api_server"),
            port=_free_port(),
            log_dir=str(tmp_path),
        )


def test_engine_spec_accepts_a_valid_sglang_declaration(tmp_path: Path, fake_module: str) -> None:
    spec = _engine_spec(tmp_path, fake_module, kind="sglang")
    assert spec.served_model_name is None
    assert "{model_path}" in spec.command


# ---------------------------------------------------------------------------
# EngineServer lifecycle (kind="sglang", against the same fake server)
# ---------------------------------------------------------------------------


def test_engine_server_starts_health_checks_and_stops(tmp_path: Path, fake_module: str) -> None:
    spec = _engine_spec(tmp_path, fake_module, model_path="/fake/model")
    server = EngineServer(spec)
    try:
        server.start()
        assert server.log_path is not None
        assert Path(server.log_path).exists()
        client = server.client()
        assert isinstance(client, SGLangClient)
        assert client.health() is True
    finally:
        server.stop()
    assert server.client().health() is False


def test_engine_server_client_dispatches_to_vllm_client_for_vllm_kind(
    tmp_path: Path, fake_module: str
) -> None:
    # No subprocess needed: client() is cheap/stateless and never requires the
    # server to be live (same contract SGLangServer.client() documents).
    spec = _engine_spec(
        tmp_path, fake_module, kind="vllm", model_path="/fake/model", served_model_name="m"
    )
    server = EngineServer(spec)
    client = server.client()
    assert isinstance(client, VLLMClient)
    assert client.served_model_name == "m"
    assert client.reload_mode is None


def test_engine_server_client_forwards_reload_mode_to_vllm_client(
    tmp_path: Path, fake_module: str
) -> None:
    spec = _engine_spec(
        tmp_path,
        fake_module,
        kind="vllm",
        model_path="/fake/model",
        served_model_name="m",
        reload_mode="collective_rpc",
    )
    client = EngineServer(spec).client()
    assert isinstance(client, VLLMClient)
    assert client.reload_mode == "collective_rpc"


def test_engine_spec_rejects_an_undeclared_reload_mode(tmp_path: Path, fake_module: str) -> None:
    with pytest.raises(EngineRefusal, match="field 'reload_mode'"):
        _engine_spec(
            tmp_path,
            fake_module,
            kind="vllm",
            served_model_name="m",
            reload_mode="rpc_v2",
        )


def test_engine_server_start_twice_refuses(tmp_path: Path, fake_module: str) -> None:
    spec = _engine_spec(tmp_path, fake_module, model_path="/fake/model")
    server = EngineServer(spec)
    try:
        server.start()
        with pytest.raises(EngineRefusal):
            server.start()
    finally:
        server.stop()


def test_engine_server_crash_before_healthy_raises_startup_crashed(
    tmp_path: Path, fake_module: str
) -> None:
    spec = _engine_spec(tmp_path, fake_module, model_path="FAIL_immediately", startup_timeout_s=5.0)
    server = EngineServer(spec)
    with pytest.raises(EngineInfraError) as excinfo:
        server.start()
    assert excinfo.value.kind == "startup_crashed"


def test_engine_server_never_healthy_times_out(tmp_path: Path, fake_module: str) -> None:
    command = (
        sys.executable,
        "-m",
        fake_module,
        "--model-path",
        "/fake/model",
        "--port",
        str(_free_port()),
        "--tp-size",
        "1",
        "--never-healthy",
    )
    spec = EngineServerSpec(
        kind="sglang",
        command=command,
        port=int(command[command.index("--port") + 1]),
        log_dir=str(tmp_path),
        env={"PYTHONPATH": str(tmp_path)},
        startup_timeout_s=0.3,
        health_poll_interval_s=0.05,
    )
    server = EngineServer(spec)
    try:
        with pytest.raises(EngineInfraError) as excinfo:
            server.start()
        assert excinfo.value.kind == "startup_timeout"
    finally:
        server.stop()


def test_engine_server_stop_is_idempotent(tmp_path: Path, fake_module: str) -> None:
    spec = _engine_spec(tmp_path, fake_module, model_path="/fake/model")
    server = EngineServer(spec)
    server.stop()  # never started: must not raise
    server.start()
    server.stop()
    server.stop()  # already stopped: must not raise


def test_engine_server_stop_kills_a_process_that_ignores_sigterm(
    tmp_path: Path, fake_module: str
) -> None:
    # Same shape as SGLangServer's identically-named test above, but for the
    # generic EngineServer.stop(): a process that traps/ignores SIGTERM must
    # still be gone after stop() returns, via the killpg(..., SIGKILL) fallback
    # once process.wait(timeout=grace_s) raises TimeoutExpired.
    port = _free_port()
    command = (
        sys.executable,
        "-m",
        fake_module,
        "--model-path",
        "/fake/model",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--ignore-sigterm",
    )
    spec = EngineServerSpec(
        kind="sglang",
        command=command,
        port=port,
        log_dir=str(tmp_path),
        env={"PYTHONPATH": str(tmp_path)},
        startup_timeout_s=10.0,
        health_poll_interval_s=0.05,
    )
    server = EngineServer(spec)
    server.start()
    server.stop(grace_s=0.2)
    assert server.client().health() is False


def test_engine_server_start_command_override_does_not_mutate_spec(
    tmp_path: Path, fake_module: str
) -> None:
    spec = _engine_spec(tmp_path, fake_module, model_path="{model_path}")
    server = EngineServer(spec)
    override = tuple(part.replace("{model_path}", "/real/model") for part in spec.command)
    try:
        server.start(command=override)
        assert server.client().health() is True
    finally:
        server.stop()
    # The declared spec is untouched -- the placeholder survives.
    assert server.spec.command == spec.command
    assert "{model_path}" in server.spec.command


# ---------------------------------------------------------------------------
# EngineFleet with the generic server pair, and .clients()
# ---------------------------------------------------------------------------


def test_fleet_accepts_a_mix_of_sglang_and_generic_servers(
    tmp_path: Path, fake_module: str
) -> None:
    legacy = SGLangServer(_spec(tmp_path, fake_module, port=_free_port()))
    generic = EngineServer(_engine_spec(tmp_path, fake_module, model_path="/fake/model"))
    fleet = EngineFleet(servers=(legacy, generic))
    fleet.start_all()
    try:
        clients = fleet.clients()
        assert len(clients) == 2
        assert all(isinstance(c, SGLangClient) for c in clients)
    finally:
        fleet.stop_all()


def test_fleet_clients_returns_the_right_class_per_kind_without_starting(
    tmp_path: Path, fake_module: str
) -> None:
    sglang_server = EngineServer(_engine_spec(tmp_path, fake_module, kind="sglang"))
    vllm_server = EngineServer(
        _engine_spec(tmp_path, fake_module, kind="vllm", served_model_name="m")
    )
    fleet = EngineFleet(servers=(sglang_server, vllm_server))
    clients = fleet.clients()
    assert isinstance(clients[0], SGLangClient)
    assert isinstance(clients[1], VLLMClient)


def test_server_creates_a_missing_log_dir(tmp_path: Path, fake_module: str) -> None:
    # Regression from the first real GPU launch: a declared log_dir that did not
    # exist yet made start() raise FileNotFoundError before the engine ran.
    log_dir = tmp_path / "not" / "created" / "yet"
    assert not log_dir.exists()
    spec = _spec(tmp_path, fake_module, port=_free_port(), log_dir=str(log_dir))
    server = SGLangServer(spec)
    try:
        server.start()
        assert server.log_path is not None
        assert Path(server.log_path).parent == log_dir
        assert Path(server.log_path).exists()
    finally:
        server.stop()
