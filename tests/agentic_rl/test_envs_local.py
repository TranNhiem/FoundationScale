"""Real-subprocess tests for slice 3's bare-host local environment backend.

Claims covered here: the local backend runs real `/bin/sh` commands with an
explicitly declared environment (no inheritance from the parent process),
enforces execution timeouts by killing the whole process group (children
included), caps per-stream output while counting dropped bytes, sets HOME to
the per-instance workdir, runs setup commands and reports their failures as
infrastructure errors, removes its temp workdir on close (idempotently),
refuses use before start/after close and refuses unsandboxed specs by default,
and that the env backend registry exposes, refuses to duplicate, and refuses
unknown backend names.

Not claimed: no container, VM, or remote backend exists yet -- only the
bare-host subprocess backend is exercised.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

import foundationscale.agentic_rl.envs  # noqa: F401  (guarantees "local" registration)
from foundationscale.agentic_rl.envs.base import (
    EnvBackend,
    EnvCapabilities,
    EnvInfraError,
    Environment,
    EnvRefusal,
    EnvSpec,
    ExecResult,
    available_env_backends,
    check_isolation,
    get_env_backend,
    register_env_backend,
)
from foundationscale.agentic_rl.envs.local import (
    LocalEnvironment,
    LocalSubprocessBackend,
    _cap_bytes,
    _finished_result,
    _sanitise_prefix,
)


def _spec(**overrides: object) -> EnvSpec:
    """Build an EnvSpec with short timeouts and an explicit empty env allow-list."""
    kwargs: dict[str, object] = {
        "backend": "local",
        "workdir_template": None,
        "exec_timeout_s": 5.0,
        "episode_timeout_s": 30.0,
        "max_output_bytes": 65536,
        "env": {},
        "setup_commands": (),
        "allow_unsandboxed": True,
    }
    kwargs.update(overrides)
    return EnvSpec(**kwargs)  # type: ignore[arg-type]


def test_echo_and_exit_codes() -> None:
    async def body() -> None:
        env = LocalSubprocessBackend().create(_spec(), instance_id="echo")
        try:
            await env.start()
            result = await env.execute("echo hello")
            assert result.stdout.strip() == "hello"
            assert result.exit_code == 0
            assert result.timed_out is False
            result2 = await env.execute("exit 7")
            assert result2.exit_code == 7
            assert result2.timed_out is False
        finally:
            await env.close()

    asyncio.run(body())


def test_timeout_kills_the_whole_process_group_including_children() -> None:
    async def body() -> None:
        env = LocalSubprocessBackend().create(
            _spec(exec_timeout_s=1.0, episode_timeout_s=5.0), instance_id="pgkill"
        )
        try:
            await env.start()
            result = await env.execute("sh -c 'sleep 30 & echo $! > pidfile; wait'")
            assert result.timed_out is True
            assert result.exit_code is None
            pid_path = Path(env.workdir) / "pidfile"
            deadline = time.monotonic() + 1.0
            child_pid: int | None = None
            while time.monotonic() < deadline:
                if pid_path.exists():
                    text = pid_path.read_text(encoding="utf-8").strip()
                    if text:
                        child_pid = int(text)
                        break
                time.sleep(0.05)
            assert child_pid is not None, "child PID file was never written"
            while True:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() > deadline + 1.0:
                    raise AssertionError(f"child pid {child_pid} still alive after group kill")
                time.sleep(0.05)
        finally:
            await env.close()

    asyncio.run(body())


def test_output_cap_truncates_and_counts_dropped_bytes() -> None:
    async def body() -> None:
        env = LocalSubprocessBackend().create(
            _spec(max_output_bytes=100, exec_timeout_s=5.0), instance_id="cap"
        )
        try:
            await env.start()
            result = await env.execute("python3 -c \"print('x' * 5000, end='')\"")
            assert result.exit_code == 0
            assert len(result.stdout.encode("utf-8", errors="replace")) <= 100
            assert result.truncated_bytes > 0
            assert result.truncated_bytes == 5000 - 100
        finally:
            await env.close()

    asyncio.run(body())


def test_env_does_not_inherit_the_parent_process_environment() -> None:
    os.environ["FSAR_TEST_LEAK"] = "should_not_appear"
    try:

        async def body() -> None:
            env = LocalSubprocessBackend().create(
                _spec(env={"ONLY_THIS": "1"}), instance_id="isolate"
            )
            try:
                await env.start()
                result = await env.execute("echo [$FSAR_TEST_LEAK][$ONLY_THIS]")
                assert result.stdout.strip() == "[][1]"
            finally:
                await env.close()

        asyncio.run(body())
    finally:
        os.environ.pop("FSAR_TEST_LEAK", None)


def test_home_is_set_to_the_workdir() -> None:
    async def body() -> None:
        env = LocalSubprocessBackend().create(_spec(), instance_id="home")
        try:
            await env.start()
            result = await env.execute("echo $HOME")
            assert result.stdout.strip() == env.workdir
        finally:
            await env.close()

    asyncio.run(body())


def test_only_the_declared_allow_list_plus_home_and_pwd_reach_the_child() -> None:
    # Regression/contract test for the EnvSpec/backend docstrings (envs.base's
    # EnvSpec and this module's LocalEnvironment): the child must never see
    # anything from THIS process's os.environ beyond HOME/PWD (the two this
    # backend adds itself) and spec.env's declared allow-list. Exercises several
    # realistic parent-process variables at once (not just one sentinel),
    # including one whose NAME collides with a declared allow-list key but whose
    # VALUE must not win.
    sentinel_names = ("FSAR_TEST_LEAK_PATH_LIKE", "FSAR_TEST_SECRET", "ONLY_THIS")
    saved = {name: os.environ.get(name) for name in sentinel_names}
    os.environ["FSAR_TEST_LEAK_PATH_LIKE"] = "/should/not/leak"
    os.environ["FSAR_TEST_SECRET"] = "super-secret-value"
    os.environ["ONLY_THIS"] = "PARENT_VALUE_MUST_NOT_WIN"
    try:

        async def body() -> None:
            env = LocalSubprocessBackend().create(
                _spec(env={"ONLY_THIS": "1", "AND_THIS": "2"}), instance_id="allowlist"
            )
            try:
                await env.start()
                result = await env.execute("env")
                assert result.exit_code == 0
                pairs = dict(
                    line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
                )
                assert "FSAR_TEST_LEAK_PATH_LIKE" not in pairs
                assert "FSAR_TEST_SECRET" not in pairs
                assert pairs["ONLY_THIS"] == "1"  # the allow-list's value wins, not the parent's
                assert pairs["AND_THIS"] == "2"
                assert pairs["HOME"] == env.workdir
                assert pairs["PWD"] == env.workdir
                # Nothing else beyond the allow-list and HOME/PWD, except the
                # small set of bookkeeping variables /bin/sh itself synthesizes
                # for EVERY invocation regardless of input (never read from this
                # process's own os.environ): SHLVL (shell nesting depth) and _
                # (the invoked command's own path).
                extra = set(pairs) - {"ONLY_THIS", "AND_THIS", "HOME", "PWD", "SHLVL", "_"}
                assert extra == set(), f"unexpected variables reached the child: {extra}"
            finally:
                await env.close()

        asyncio.run(body())
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_timeout_path_applies_the_same_byte_cap_and_reports_real_truncated_bytes() -> None:
    # Regression: the timeout branch used to hard-code truncated_bytes=0 and never
    # capped recovered output. Here the child flushes well over max_output_bytes
    # to stdout and then closes its own fd 1 (so the reader task actually
    # completes with real data) before sleeping past exec_timeout_s, so the
    # process group kill still fires.
    async def body() -> None:
        env = LocalEnvironment(
            _spec(exec_timeout_s=1.0, episode_timeout_s=5.0, max_output_bytes=100),
            instance_id="tmocap",
        )
        try:
            await env.start()
            command = (
                'exec python3 -c "import sys, time, os; '
                "sys.stdout.write('x' * 5000); sys.stdout.flush(); os.close(1); "
                'time.sleep(5)"'
            )
            result = await env.execute(command)
            assert result.timed_out is True
            assert result.exit_code is None
            assert len(result.stdout.encode("utf-8")) <= 100
            assert result.truncated_bytes == 5000 - 100
        finally:
            await env.close()

    asyncio.run(body())


def test_close_removes_the_temp_directory_and_is_idempotent() -> None:
    async def body() -> None:
        env = LocalSubprocessBackend().create(_spec(), instance_id="cleanup")
        await env.start()
        workdir = env.workdir
        assert Path(workdir).is_dir()
        await env.close()
        assert not Path(workdir).exists()
        await env.close()

    asyncio.run(body())


def test_setup_command_failure_raises_env_infra_error() -> None:
    async def body() -> None:
        env = LocalSubprocessBackend().create(
            _spec(setup_commands=("exit 3",)), instance_id="setupfail"
        )
        try:
            with pytest.raises(EnvInfraError) as excinfo:
                await env.start()
            exc = excinfo.value
            assert exc.kind == "setup_failed"
            assert "3" in str(exc)
        finally:
            await env.close()

    asyncio.run(body())


def test_execute_before_start_and_after_close_both_refuse() -> None:
    async def body() -> None:
        fresh = LocalSubprocessBackend().create(_spec(), instance_id="fresh")
        try:
            with pytest.raises(EnvRefusal):
                await fresh.execute("echo nope")
        finally:
            await fresh.close()

        env = LocalSubprocessBackend().create(_spec(), instance_id="closed")
        try:
            await env.start()
        finally:
            await env.close()
        with pytest.raises(EnvRefusal):
            await env.execute("echo nope")

    asyncio.run(body())


def test_unsandboxed_use_is_refused_by_default() -> None:
    spec = EnvSpec(backend="local", env={})
    with pytest.raises(EnvRefusal) as excinfo:
        LocalSubprocessBackend().create(spec, instance_id="x")
    message = str(excinfo.value).lower()
    assert "allow_unsandboxed" in message or "sandbox" in message


def test_registry_has_local_backend_and_refuses_duplicates_and_unknowns() -> None:
    assert "local" in available_env_backends()
    backend = get_env_backend("local")
    assert backend.name == "local"

    with pytest.raises(EnvRefusal) as unknown_exc:
        get_env_backend("does-not-exist")
    assert "does-not-exist" in str(unknown_exc.value)

    with pytest.raises(EnvRefusal) as dup_exc:
        register_env_backend("local", lambda: LocalSubprocessBackend())
    assert "local" in str(dup_exc.value)


# --------------------------------------------------- coverage additions below


def test_env_infra_error_refuses_bad_kind_and_message() -> None:
    with pytest.raises(EnvRefusal):
        EnvInfraError(7, "boom")


def test_env_infra_error_refuses_empty_kind() -> None:
    with pytest.raises(EnvRefusal):
        EnvInfraError("", "boom")


def test_env_infra_error_refuses_non_str_message() -> None:
    with pytest.raises(EnvRefusal):
        EnvInfraError("transport", 7)


def test_env_spec_refuses_non_str_backend() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend=7, env={})  # type: ignore[arg-type]


def test_env_spec_refuses_empty_backend() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="", env={})


def test_env_spec_refuses_non_str_workdir_template() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, workdir_template=7)  # type: ignore[arg-type]


def test_env_spec_refuses_empty_workdir_template() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, workdir_template="")


@pytest.mark.parametrize("field_name", ["exec_timeout_s", "episode_timeout_s"])
@pytest.mark.parametrize("bad_value", [True, "5", float("nan"), 0.0])
def test_env_spec_refuses_bad_timeout(field_name: str, bad_value: object) -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, **{field_name: bad_value})  # type: ignore[arg-type]


def test_env_spec_refuses_non_int_max_output_bytes() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, max_output_bytes=True)  # type: ignore[arg-type]


def test_env_spec_refuses_non_positive_max_output_bytes() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, max_output_bytes=0)


def test_env_spec_refuses_non_mapping_env() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env=[("A", "1")])  # type: ignore[arg-type]


def test_env_spec_refuses_non_str_env_key() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={7: "1"})  # type: ignore[dict-item]


def test_env_spec_refuses_non_str_env_value() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={"A": 7})  # type: ignore[dict-item]


def test_env_spec_refuses_bare_str_setup_commands() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, setup_commands="echo hi")  # type: ignore[arg-type]


def test_env_spec_refuses_non_sequence_setup_commands() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, setup_commands=7)  # type: ignore[arg-type]


def test_env_spec_refuses_non_str_setup_command_element() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, setup_commands=(7,))  # type: ignore[arg-type]


def test_env_spec_refuses_empty_setup_command_element() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, setup_commands=("",))


def test_env_spec_refuses_non_bool_allow_unsandboxed() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, allow_unsandboxed=1)  # type: ignore[arg-type]


def test_exec_result_refuses_non_str_stdout() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout=b"x", stderr="", exit_code=0, timed_out=False, truncated_bytes=0, seconds=0.0
        )  # type: ignore[arg-type]


def test_exec_result_refuses_non_str_stderr() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="", stderr=b"x", exit_code=0, timed_out=False, truncated_bytes=0, seconds=0.0
        )  # type: ignore[arg-type]


def test_exec_result_refuses_non_bool_timed_out() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(stdout="", stderr="", exit_code=0, timed_out=1, truncated_bytes=0, seconds=0.0)  # type: ignore[arg-type]


def test_exec_result_refuses_non_int_exit_code() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="", stderr="", exit_code=1.5, timed_out=False, truncated_bytes=0, seconds=0.0
        )  # type: ignore[arg-type]


def test_exec_result_refuses_exit_code_with_timed_out() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=True, truncated_bytes=0, seconds=0.0
        )


def test_exec_result_refuses_missing_exit_code_without_timeout() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="", stderr="", exit_code=None, timed_out=False, truncated_bytes=0, seconds=0.0
        )


def test_exec_result_refuses_non_int_truncated_bytes() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=False, truncated_bytes=1.5, seconds=0.0
        )  # type: ignore[arg-type]


def test_exec_result_refuses_negative_truncated_bytes() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=False, truncated_bytes=-1, seconds=0.0
        )


def test_exec_result_refuses_bool_seconds() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=False, truncated_bytes=0, seconds=True
        )  # type: ignore[arg-type]


def test_exec_result_refuses_non_numeric_seconds() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=False, truncated_bytes=0, seconds="1"
        )  # type: ignore[arg-type]


def test_exec_result_refuses_non_finite_seconds() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="",
            stderr="",
            exit_code=0,
            timed_out=False,
            truncated_bytes=0,
            seconds=float("inf"),
        )


def test_exec_result_refuses_negative_seconds() -> None:
    with pytest.raises(EnvRefusal):
        ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=False, truncated_bytes=0, seconds=-1.0
        )


def test_env_capabilities_refuses_illegal_isolation() -> None:
    with pytest.raises(EnvRefusal):
        EnvCapabilities(isolation="sandbox", network=True, max_concurrency=None, platform="linux")  # type: ignore[arg-type]


def test_env_capabilities_refuses_non_bool_network() -> None:
    with pytest.raises(EnvRefusal):
        EnvCapabilities(isolation="none", network=1, max_concurrency=None, platform="linux")  # type: ignore[arg-type]


def test_env_capabilities_refuses_non_int_max_concurrency() -> None:
    with pytest.raises(EnvRefusal):
        EnvCapabilities(isolation="none", network=True, max_concurrency=1.5, platform="linux")  # type: ignore[arg-type]


def test_env_capabilities_refuses_zero_max_concurrency() -> None:
    with pytest.raises(EnvRefusal):
        EnvCapabilities(isolation="none", network=True, max_concurrency=0, platform="linux")


def test_env_capabilities_refuses_non_str_platform() -> None:
    with pytest.raises(EnvRefusal):
        EnvCapabilities(isolation="none", network=True, max_concurrency=None, platform=7)  # type: ignore[arg-type]


def test_env_capabilities_refuses_empty_platform() -> None:
    with pytest.raises(EnvRefusal):
        EnvCapabilities(isolation="none", network=True, max_concurrency=None, platform="")


def test_register_env_backend_refuses_non_str_name() -> None:
    with pytest.raises(EnvRefusal):
        register_env_backend(7, lambda: LocalSubprocessBackend())  # type: ignore[arg-type]


def test_register_env_backend_refuses_empty_name() -> None:
    with pytest.raises(EnvRefusal):
        register_env_backend("", lambda: LocalSubprocessBackend())


def test_get_env_backend_refuses_non_str_name() -> None:
    with pytest.raises(EnvRefusal):
        get_env_backend(7)  # type: ignore[arg-type]


def test_get_env_backend_refuses_empty_name() -> None:
    with pytest.raises(EnvRefusal):
        get_env_backend("")


def test_available_env_backends_returns_sorted_tuple() -> None:
    names = available_env_backends()
    assert isinstance(names, tuple)
    assert names == tuple(sorted(names))
    assert "local" in names


def test_check_isolation_refuses_non_env_spec() -> None:
    caps = EnvCapabilities(isolation="none", network=True, max_concurrency=None, platform="linux")
    with pytest.raises(EnvRefusal):
        check_isolation(object(), caps)  # type: ignore[arg-type]


def test_check_isolation_refuses_non_env_capabilities() -> None:
    spec = EnvSpec(backend="local", env={}, allow_unsandboxed=True)
    with pytest.raises(EnvRefusal):
        check_isolation(spec, object())  # type: ignore[arg-type]


def test_check_isolation_allows_sandboxed_backend_without_opt_in() -> None:
    spec = EnvSpec(backend="local", env={}, allow_unsandboxed=False)
    caps = EnvCapabilities(
        isolation="container", network=True, max_concurrency=None, platform="linux"
    )
    check_isolation(spec, caps)


def test_check_isolation_allows_unsandboxed_with_opt_in() -> None:
    spec = EnvSpec(backend="local", env={}, allow_unsandboxed=True)
    caps = EnvCapabilities(isolation="none", network=True, max_concurrency=None, platform="linux")
    check_isolation(spec, caps)


def test_make_local_backend_factory_returns_local_backend() -> None:
    backend = get_env_backend("local")
    assert isinstance(backend, LocalSubprocessBackend)
    assert backend.name == "local"


def test_local_environment_refuses_non_posix_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    with pytest.raises(EnvRefusal):
        LocalEnvironment(_spec(), instance_id="win")


def test_workdir_property_refuses_before_start() -> None:
    env = LocalEnvironment(_spec(), instance_id="noworkdir")
    with pytest.raises(EnvRefusal):
        _ = env.workdir


def test_start_twice_refuses() -> None:
    async def body() -> None:
        env = LocalEnvironment(_spec(), instance_id="twice")
        try:
            await env.start()
            with pytest.raises(EnvRefusal):
                await env.start()
        finally:
            await env.close()

    asyncio.run(body())


def test_execute_after_close_refuses() -> None:
    async def body() -> None:
        env = LocalEnvironment(_spec(), instance_id="afterclose")
        await env.start()
        await env.close()
        with pytest.raises(EnvRefusal):
            await env.execute("echo nope")

    asyncio.run(body())


def test_setup_command_timeout_raises_setup_failed() -> None:
    async def body() -> None:
        env = LocalEnvironment(
            _spec(setup_commands=("sleep 5",), exec_timeout_s=0.5, episode_timeout_s=5.0),
            instance_id="setuptmo",
        )
        try:
            with pytest.raises(EnvInfraError) as excinfo:
                await env.start()
            exc = excinfo.value
            assert exc.kind == "setup_failed"
            assert "timed out" in str(exc)
        finally:
            await env.close()

    asyncio.run(body())


def test_close_propagates_non_file_not_found_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    async def body() -> None:
        env = LocalEnvironment(_spec(), instance_id="rmtreefail")
        await env.start()

        def boom(path: str, **kwargs: object) -> None:
            raise PermissionError("nope")

        monkeypatch.setattr(shutil, "rmtree", boom)
        with pytest.raises(PermissionError):
            await env.close()

    asyncio.run(body())


def test_close_before_start_is_safe_and_idempotent() -> None:
    async def body() -> None:
        env = LocalEnvironment(_spec(), instance_id="noclose")
        await env.close()
        await env.close()

    asyncio.run(body())


def test_cap_bytes_keeps_head_and_tail_when_over_cap() -> None:
    kept, dropped = _cap_bytes(b"abcdefghij", 4)
    assert kept == b"ab" + b"ij"
    assert dropped == 6


def test_cap_bytes_zero_cap_keeps_nothing() -> None:
    kept, dropped = _cap_bytes(b"abcd", 0)
    assert kept == b""
    assert dropped == 4


def test_cap_bytes_under_cap_keeps_all() -> None:
    kept, dropped = _cap_bytes(b"ab", 10)
    assert kept == b"ab"
    assert dropped == 0


def test_finished_result_returns_empty_for_pending_task() -> None:
    async def body() -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bytes] = loop.create_future()
        assert _finished_result(fut) == b""

    asyncio.run(body())


def test_finished_result_returns_result_for_done_task() -> None:
    async def body() -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bytes] = loop.create_future()
        fut.set_result(b"done")
        assert _finished_result(fut) == b"done"

    asyncio.run(body())


def test_finished_result_returns_empty_for_cancelled_task() -> None:
    async def body() -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bytes] = loop.create_future()
        fut.cancel()
        assert _finished_result(fut) == b""

    asyncio.run(body())


def test_sanitise_prefix_fallback_for_unusable_instance_id() -> None:
    assert _sanitise_prefix("___") == "fsar-"
    assert _sanitise_prefix("") == "fsar-"
    assert _sanitise_prefix("a/b c") == "a_b_c"


def test_environment_protocol_instances_satisfy_runtime_checkable() -> None:
    env = LocalEnvironment(_spec(), instance_id="proto")
    assert isinstance(env, Environment)
    assert isinstance(LocalSubprocessBackend(), EnvBackend)


def test_episode_timeout_s_property_reads_spec() -> None:
    env = LocalEnvironment(_spec(episode_timeout_s=42.0), instance_id="budget")
    assert env.episode_timeout_s == 42.0


def test_exec_result_coerces_int_seconds_to_float() -> None:
    result = ExecResult(
        stdout="", stderr="", exit_code=0, timed_out=False, truncated_bytes=0, seconds=3
    )
    assert result.seconds == 3.0
    assert type(result.seconds) is float


def test_env_spec_coerces_int_timeouts_to_float() -> None:
    spec = EnvSpec(backend="local", env={}, exec_timeout_s=5, episode_timeout_s=10)
    assert spec.exec_timeout_s == 5.0
    assert spec.episode_timeout_s == 10.0


def test_env_spec_freezes_env_and_setup_commands() -> None:
    spec = EnvSpec(backend="local", env={"A": "1"}, setup_commands=["echo hi"])
    assert isinstance(spec.env, Mapping)
    assert spec.setup_commands == ("echo hi",)
    with pytest.raises(TypeError):
        spec.env["B"] = "2"  # type: ignore[index]


def test_env_capabilities_accepts_none_max_concurrency() -> None:
    caps = EnvCapabilities(
        isolation="remote", network=False, max_concurrency=None, platform="darwin"
    )
    assert caps.max_concurrency is None
    assert caps.isolation == "remote"


def test_register_env_backend_accepts_new_name_and_get_returns_it() -> None:
    name = "test-extra-backend"

    def factory() -> LocalSubprocessBackend:
        return LocalSubprocessBackend()

    register_env_backend(name, factory)
    assert name in available_env_backends()
    assert isinstance(get_env_backend(name), LocalSubprocessBackend)


def test_register_env_backend_refuses_non_callable_factory() -> None:
    with pytest.raises(EnvRefusal):
        register_env_backend("test-noncallable", 7)  # type: ignore[arg-type]


def test_execute_with_explicit_timeout_s_overrides_spec() -> None:
    async def body() -> None:
        env = LocalEnvironment(
            _spec(exec_timeout_s=30.0, episode_timeout_s=60.0), instance_id="ovr"
        )
        try:
            await env.start()
            result = await env.execute("sleep 5", timeout_s=0.5)
            assert result.timed_out is True
            assert result.exit_code is None
        finally:
            await env.close()

    asyncio.run(body())


def test_setup_command_success_runs_in_order() -> None:
    async def body() -> None:
        env = LocalEnvironment(
            _spec(setup_commands=("echo one > a.txt", "echo two > b.txt")),
            instance_id="setupok",
        )
        try:
            await env.start()
            assert (Path(env.workdir) / "a.txt").read_text(encoding="utf-8").strip() == "one"
            assert (Path(env.workdir) / "b.txt").read_text(encoding="utf-8").strip() == "two"
        finally:
            await env.close()

    asyncio.run(body())


def test_setup_failure_cleans_up_workdir() -> None:
    async def body() -> None:
        env = LocalEnvironment(_spec(setup_commands=("exit 1",)), instance_id="setupclean")
        try:
            with pytest.raises(EnvInfraError):
                await env.start()
        finally:
            await env.close()
        assert env._workdir is None

    asyncio.run(body())


def test_run_raises_env_infra_error_on_spawn_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    async def body() -> None:
        env = LocalEnvironment(_spec(), instance_id="spawnfail")
        await env.start()

        async def boom(*args: object, **kwargs: object) -> object:
            raise OSError("spawn broke")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
        with pytest.raises(EnvInfraError) as excinfo:
            await env.execute("echo hi")
        assert excinfo.value.kind == "start_failed"
        await env.close()

    asyncio.run(body())


def test_env_spec_env_is_read_only_mapping_proxy() -> None:
    spec = EnvSpec(backend="local", env={"A": "1"})
    assert dict(spec.env) == {"A": "1"}
    with pytest.raises(TypeError):
        spec.env["B"] = "2"  # type: ignore[index]


def test_exec_result_seconds_zero_is_allowed() -> None:
    result = ExecResult(
        stdout="", stderr="", exit_code=0, timed_out=False, truncated_bytes=0, seconds=0
    )
    assert result.seconds == 0.0


def test_exec_result_timed_out_with_none_exit_code_is_allowed() -> None:
    result = ExecResult(
        stdout="", stderr="", exit_code=None, timed_out=True, truncated_bytes=0, seconds=1.0
    )
    assert result.timed_out is True
    assert result.exit_code is None


def test_env_spec_max_concurrency_none_and_int_both_allowed() -> None:
    caps_none = EnvCapabilities(
        isolation="vm", network=True, max_concurrency=None, platform="linux"
    )
    caps_int = EnvCapabilities(isolation="vm", network=True, max_concurrency=4, platform="linux")
    assert caps_none.max_concurrency is None
    assert caps_int.max_concurrency == 4


def test_env_spec_setup_commands_coerced_from_list_to_tuple() -> None:
    commands = ["echo a", "echo b"]
    spec = EnvSpec(backend="local", env={}, setup_commands=commands)
    assert spec.setup_commands == ("echo a", "echo b")
    assert type(spec.setup_commands) is tuple


def test_env_spec_env_coerced_from_dict_to_mapping_proxy() -> None:
    source = {"A": "1"}
    spec = EnvSpec(backend="local", env=source)
    source["B"] = "2"
    assert dict(spec.env) == {"A": "1"}


def test_local_environment_instance_id_sanitised_into_temp_prefix() -> None:
    # "/" is replaced (it would otherwise split the prefix into path components);
    # "." is deliberately KEPT (it is in the allowed set and, with no "/" beside
    # it, a lone ".." substring inside one path component is not a traversal --
    # mkdtemp never interprets the prefix as a multi-component path). The real
    # safety property is: workdir is exactly one directory, freshly created,
    # directly under the system temp root.
    async def body() -> None:
        env = LocalEnvironment(_spec(), instance_id="weird/../id")
        try:
            await env.start()
            name = Path(env.workdir).name
            assert "/" not in name
            assert Path(env.workdir).is_dir()
        finally:
            await env.close()

    asyncio.run(body())


def test_episode_timeout_bounds_effective_execute_timeout() -> None:
    async def body() -> None:
        env = LocalEnvironment(
            _spec(exec_timeout_s=30.0, episode_timeout_s=1.0), instance_id="budget2"
        )
        try:
            await env.start()
            result = await env.execute("sleep 30")
            assert result.timed_out is True
        finally:
            await env.close()

    asyncio.run(body())


def test_exec_result_accepts_negative_exit_code() -> None:
    result = ExecResult(
        stdout="", stderr="", exit_code=-9, timed_out=False, truncated_bytes=0, seconds=0.5
    )
    assert result.exit_code == -9


def test_env_spec_accepts_bytes_in_setup_commands_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, setup_commands=b"echo hi")  # type: ignore[arg-type]


def test_env_spec_accepts_bytearray_in_setup_commands_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, setup_commands=bytearray(b"echo hi"))  # type: ignore[arg-type]


def test_env_spec_accepts_tuple_env_mapping() -> None:
    spec = EnvSpec(backend="local", env={"X": "y", "Z": "w"})
    assert spec.env["X"] == "y"
    assert spec.env["Z"] == "w"


def test_env_spec_accepts_float_timeouts() -> None:
    spec = EnvSpec(backend="local", env={}, exec_timeout_s=2.5, episode_timeout_s=7.5)
    assert spec.exec_timeout_s == 2.5
    assert spec.episode_timeout_s == 7.5


def test_env_capabilities_accepts_all_isolation_levels() -> None:
    for level in ("none", "container", "vm", "remote"):
        caps = EnvCapabilities(isolation=level, network=False, max_concurrency=1, platform="linux")  # type: ignore[arg-type]
        assert caps.isolation == level


def test_env_spec_accepts_nonempty_workdir_template() -> None:
    spec = EnvSpec(backend="local", env={}, workdir_template="/tmp/run-{id}")
    assert spec.workdir_template == "/tmp/run-{id}"


def test_env_spec_accepts_none_workdir_template() -> None:
    spec = EnvSpec(backend="local", env={}, workdir_template=None)
    assert spec.workdir_template is None


def test_env_spec_accepts_nonempty_setup_command() -> None:
    spec = EnvSpec(backend="local", env={}, setup_commands=("echo ok",))
    assert spec.setup_commands == ("echo ok",)


def test_env_spec_accepts_bool_false_allow_unsandboxed() -> None:
    spec = EnvSpec(backend="local", env={}, allow_unsandboxed=False)
    assert spec.allow_unsandboxed is False


def test_env_spec_accepts_bool_true_allow_unsandboxed() -> None:
    spec = EnvSpec(backend="local", env={}, allow_unsandboxed=True)
    assert spec.allow_unsandboxed is True


def test_env_spec_accepts_positive_max_output_bytes() -> None:
    spec = EnvSpec(backend="local", env={}, max_output_bytes=1)
    assert spec.max_output_bytes == 1


def test_env_spec_accepts_large_max_output_bytes() -> None:
    spec = EnvSpec(backend="local", env={}, max_output_bytes=10_000_000)
    assert spec.max_output_bytes == 10_000_000


def test_env_spec_accepts_empty_env_mapping() -> None:
    spec = EnvSpec(backend="local", env={})
    assert dict(spec.env) == {}


def test_env_spec_accepts_empty_setup_commands() -> None:
    spec = EnvSpec(backend="local", env={}, setup_commands=())
    assert spec.setup_commands == ()


def test_env_spec_accepts_list_setup_commands() -> None:
    spec = EnvSpec(backend="local", env={}, setup_commands=["a", "b"])
    assert spec.setup_commands == ("a", "b")


def test_env_spec_accepts_nonempty_backend_name() -> None:
    spec = EnvSpec(backend="custom", env={})
    assert spec.backend == "custom"


def test_env_spec_accepts_int_max_output_bytes() -> None:
    spec = EnvSpec(backend="local", env={}, max_output_bytes=128)
    assert spec.max_output_bytes == 128


def test_env_spec_accepts_int_timeout_values() -> None:
    spec = EnvSpec(backend="local", env={}, exec_timeout_s=1, episode_timeout_s=2)
    assert spec.exec_timeout_s == 1.0
    assert spec.episode_timeout_s == 2.0


def test_env_spec_accepts_float_max_output_bytes_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, max_output_bytes=1.5)  # type: ignore[arg-type]


def test_env_spec_accepts_negative_max_output_bytes_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, max_output_bytes=-1)


def test_env_spec_accepts_negative_timeout_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, exec_timeout_s=-1.0)


def test_env_spec_accepts_zero_episode_timeout_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, episode_timeout_s=0)


def test_env_spec_accepts_inf_timeout_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, exec_timeout_s=float("inf"))


def test_env_spec_accepts_nan_episode_timeout_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, episode_timeout_s=float("nan"))


def test_env_spec_accepts_str_timeout_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, exec_timeout_s="5")  # type: ignore[arg-type]


def test_env_spec_accepts_bool_episode_timeout_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, episode_timeout_s=True)  # type: ignore[arg-type]


def test_env_spec_accepts_bool_exec_timeout_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, exec_timeout_s=False)  # type: ignore[arg-type]


def test_env_spec_accepts_non_str_env_key_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={None: "1"})  # type: ignore[dict-item]


def test_env_spec_accepts_non_str_env_value_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={"A": None})  # type: ignore[dict-item]


def test_env_spec_accepts_non_str_setup_element_refused() -> None:
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, setup_commands=(None,))  # type: ignore[arg-type]


def test_env_spec_accepts_empty_setup_element_refused() -> None:
    # Only the empty string "" is refused ("every setup command is a non-empty
    # str"); a whitespace-only command is a non-empty str and is not this backend's
    # business to further judge.
    with pytest.raises(EnvRefusal):
        EnvSpec(backend="local", env={}, setup_commands=("",))
