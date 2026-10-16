"""Unit tests for the agentic RL probe/emit layer (no subprocess, no GPU).

Every boundary is injected: ``probe_agentic``/``emit_agentic`` take their
runner callable, and the skill's ``build_agentic_config``/``agentic_gates``
are patched in. What is asserted is the CONTRACT: the ``enroot start``
envelope of the measured launcher, measured-truthfulness of the probe
(never a raised error, never an assumed capability), BLOCK vs WARN rule
discipline, and FS's own ``--dry-run`` as the only schema authority.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import subprocess

import foundationskills.interfaces.fs.emit_agentic as emit_agentic_mod
from foundationskills.interfaces.fs.agentic_probe import (
    AGENTIC_PROBE_SNIPPET,
    Runtime,
    probe_agentic,
    runtime_argv,
    runtime_env,
    runtime_from_dict,
)

PROBE_PAYLOAD: dict[str, Any] = {
    "fs_version": "0.1.0",
    "algorithms": ["agentic_grpo", "agentic_grpo_token_mean"],
    "engines_importable": {"vllm": True, "sglang": False},
    "env_backends": ["local"],
    "reward_kinds": ["music"],
    "parsers": ["hermes_json", "qwen_xml"],
    "errors": [],
}


# --------------------------------------------------------------------------- helpers


def _recording_run(
    *,
    rc: int = 0,
    out: str = "",
    err: str = "",
    exc: BaseException | None = None,
) -> tuple[Callable[..., Any], list[tuple[list[str], dict[str, Any]]]]:
    """A fake ``subprocess.run`` that records ``(argv, kwargs)``."""

    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(argv: Any, **kwargs: Any) -> Any:
        calls.append((list(argv), dict(kwargs)))
        if exc is not None:
            raise exc
        return SimpleNamespace(returncode=rc, stdout=out, stderr=err)

    return run, calls


def _enroot_runtime() -> Runtime:
    return runtime_from_dict(
        {
            "kind": "enroot",
            "container": "vllmsrv3",
            "mounts": {"/data/shared": "/data/shared"},
            "env": {"PATH": "/usr/local/bin:/usr/bin:/bin"},
            "enroot_data_path": "/var/lib/enroot",
            "rw": True,
        }
    )


def _stage(runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    """A plan_styled agentic stage (content is read by the gates, not the emitter)."""

    return {
        "name": "agentic_rl",
        "stage": "agentic_rl",
        "algorithm": "agentic_grpo_token_mean",
        "method": "full",
        "gpus": 1,
        "executable": None,
        "runtime": runtime if runtime is not None else {"kind": "host"},
        "agentic": {
            "algorithm": "agentic_grpo_token_mean",
            "model": "/mnt/models/Qwen2.5-1.5B-Instruct",
            "tasks_path": "examples/agentic_rl/music_tasks_s0.jsonl",
            "reward": {"kind": "music", "abc2midi_bin": "abc2midi"},
            "engine": {
                "kind": "vllm",
                "served_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "port": 8000,
                "weight_sync_mode": "reload_endpoint",
                "allow_stale_policy": False,
                "request_timeout_s": None,
            },
            "env": {"backend": "local", "allow_unsandboxed": True},
            "harness": {"parser": "qwen_xml"},
            "rollout": {"group_size": 2, "tasks_per_step": 1, "max_concurrency": 2},
            "trainer": {"max_steps": 2, "lr": 1e-6, "sharding": "none"},
            "gpus": 1,
            "publish_root": "/tmp/fs_agentic_rl_s0/publish",
        },
    }


def _caps(measured: bool = True) -> dict[str, Any]:
    return {
        "measured": measured,
        "algorithms": ["agentic_grpo", "agentic_grpo_token_mean"],
        "engines_importable": {"vllm": True, "sglang": False},
        "env_backends": ["local"],
        "reward_kinds": ["music"],
        "parsers": ["hermes_json", "qwen_xml"],
    }


def _patch_skill(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gates: tuple[tuple[str, str], ...] = (),
    config: dict[str, Any] | None = None,
) -> None:
    """Inject the skill layer's two callables (they live in another task)."""

    if config is None:
        config = {
            "policy": {
                "model_path": "/mnt/models/Qwen2.5-1.5B-Instruct",
                "algorithm": "agentic_grpo_token_mean",
            }
        }
    monkeypatch.setattr(emit_agentic_mod, "build_agentic_config", lambda stage: config)
    monkeypatch.setattr(emit_agentic_mod, "agentic_gates", lambda stage, caps: list(gates))


def _emit(
    tmp_path: Path,
    *,
    stage: dict[str, Any] | None = None,
    caps: dict[str, Any] | None = None,
    dry: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if dry is None:
        dry, _ = _recording_run()
    return emit_agentic_mod.emit_agentic(
        stage if stage is not None else _stage(),
        output_dir=str(tmp_path),
        run_name="agentic-s0",
        fs_src="/fs/src",
        agentic_caps=_caps() if caps is None else caps,
        dry_run=dry,
    )


# ------------------------------------------------------------------ runtime_argv / _from_dict / _env


def test_runtime_argv_host_returns_cmd_verbatim() -> None:
    """Host runtime: no envelope -- the command is exactly what was asked for."""

    argv = runtime_argv(Runtime(kind="host"), ["python3", "-c", "print(1)"], extra_env={"PYTHONPATH": "/fs/src"})
    assert argv == ["python3", "-c", "print(1)"]


def test_runtime_argv_enroot_layers_flags_in_the_measured_order() -> None:
    """enroot start, --rw, mounts, envs (rt.env then extra_env), container, cmd."""

    argv = runtime_argv(
        _enroot_runtime(),
        ["python3", "-m", "foundationscale.agentic_rl"],
        extra_env={"PYTHONPATH": "/fs/src"},
    )
    assert argv == [
        "enroot",
        "start",
        "--rw",
        "--mount",
        "/data/shared:/data/shared",
        "--env",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "--env",
        "PYTHONPATH=/fs/src",
        "vllmsrv3",
        "python3",
        "-m",
        "foundationscale.agentic_rl",
    ]


def test_runtime_argv_refuses_whitespace_in_the_container_name() -> None:
    with pytest.raises(ValueError):
        runtime_argv(Runtime(kind="enroot", container="vl lm"), ["python3"])


def test_runtime_argv_refuses_whitespace_in_a_declared_env_value() -> None:
    with pytest.raises(ValueError):
        runtime_argv(
            Runtime(kind="enroot", container="vllmsrv3", env=(("CUDA_VISIBLE_DEVICES", "0 1"),)),
            ["python3"],
        )


def test_runtime_argv_refuses_newline_in_an_extra_env_value() -> None:
    with pytest.raises(ValueError):
        runtime_argv(
            Runtime(kind="enroot", container="vllmsrv3"),
            ["python3"],
            extra_env={"PYTHONPATH": "/fs/src\n/bin"},
        )


def test_runtime_argv_refuses_an_empty_container_for_enroot() -> None:
    with pytest.raises(ValueError):
        runtime_argv(Runtime(kind="enroot", container=""), ["python3"])
    with pytest.raises(ValueError):
        runtime_argv(Runtime(kind="enroot"), ["python3"])


def test_runtime_from_dict_none_and_empty_resolve_to_host() -> None:
    assert runtime_from_dict(None) == Runtime(kind="host")
    assert runtime_from_dict({}) == Runtime(kind="host")


def test_runtime_from_dict_enroot_round_trips_through_to_dict() -> None:
    rt = _enroot_runtime()
    again = runtime_from_dict(rt.to_dict())
    assert again == rt
    assert again.mounts == (("/data/shared", "/data/shared"),)
    assert again.env == (("PATH", "/usr/local/bin:/usr/bin:/bin"),)
    assert runtime_from_dict({"kind": "enroot", "mounts": [["/a", "/b"]]}).mounts == (("/a", "/b"),)


def test_runtime_env_carries_the_enroot_data_path_only() -> None:
    assert runtime_env(_enroot_runtime()) == {"ENROOT_DATA_PATH": "/var/lib/enroot"}
    assert runtime_env(Runtime(kind="enroot", container="vllmsrv3")) == {}
    assert runtime_env(Runtime()) == {}


# --------------------------------------------------------------------------- agentic probe


def test_probe_snippet_prints_one_json_line_and_never_raises(capsys: pytest.CaptureFixture[str]) -> None:
    """The snippet is self-contained: it prints one JSON line even without FS."""

    namespace: dict[str, Any] = {"__name__": "<agentic_probe>"}
    exec(compile(AGENTIC_PROBE_SNIPPET, "<agentic_probe>", "exec"), namespace)
    namespace["main"]()
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == 1
    payload = json.loads(out)
    assert set(payload) == {
        "fs_version",
        "algorithms",
        "engines_importable",
        "env_backends",
        "reward_kinds",
        "parsers",
        "errors",
    }
    assert isinstance(payload["errors"], list)


def test_probe_agentic_success_reports_the_measured_payload() -> None:
    run, calls = _recording_run(out=json.dumps(PROBE_PAYLOAD) + "\n")
    rt = Runtime(kind="host")
    report = probe_agentic(rt, run=run)
    assert report["measured"] is True
    assert report["algorithms"] == ["agentic_grpo", "agentic_grpo_token_mean"]
    assert report["engines_importable"] == {"vllm": True, "sglang": False}
    assert report["reward_kinds"] == ["music"]
    assert report["errors"] == []
    assert report["runtime"] == rt.to_dict()
    assert calls[0][0][1] == "-c"
    assert calls[0][0][2] == AGENTIC_PROBE_SNIPPET


def test_probe_agentic_injects_fs_src_into_the_declared_runtime() -> None:
    run, calls = _recording_run(out=json.dumps(PROBE_PAYLOAD) + "\n")
    rt = _enroot_runtime()
    report = probe_agentic(rt, fs_src="/fs/src", run=run, timeout_s=5.0)
    assert report["measured"] is True
    argv = calls[0][0]
    assert argv[:2] == ["enroot", "start"]
    assert "--env" in argv and "PYTHONPATH=/fs/src" in argv
    assert argv[-4:] == [rt.container, rt.python, "-c", AGENTIC_PROBE_SNIPPET]
    assert calls[0][1]["env"]["ENROOT_DATA_PATH"] == "/var/lib/enroot"
    assert calls[0][1]["timeout"] == 5.0


def test_probe_agentic_non_zero_exit_is_unmeasured() -> None:
    run, _ = _recording_run(rc=7, err="vllm: import refused")
    report = probe_agentic(Runtime(), run=run)
    assert report["measured"] is False
    assert report["runtime"] == Runtime().to_dict()
    assert "exited 7" in report["errors"][0]
    assert "vllm: import refused" in report["errors"][0]


def test_probe_agentic_timeout_is_unmeasured() -> None:
    run, _ = _recording_run(exc=subprocess.TimeoutExpired(cmd="probe", timeout=1))
    report = probe_agentic(Runtime(), run=run, timeout_s=0.5)
    assert report["measured"] is False
    assert "timed out" in report["errors"][0]


def test_probe_agentic_unparsable_output_is_unmeasured() -> None:
    run, _ = _recording_run(out="not json at all")
    report = probe_agentic(Runtime(), run=run)
    assert report["measured"] is False
    assert "no JSON line" in report["errors"][0]


def test_probe_agentic_never_raises_when_the_probe_cannot_start() -> None:
    run, _ = _recording_run(exc=OSError("enroot: No such file or directory"))
    report = probe_agentic(_enroot_runtime(), run=run)
    assert report["measured"] is False
    assert report["errors"] and "could not start" in report["errors"][0]


# --------------------------------------------------------------------------- emit_agentic


def test_emit_agentic_is_executable_only_when_caps_measured_gates_pass_and_dry_run_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trio of executability conditions, all satisfied."""

    _patch_skill(monkeypatch, config={"policy": {"algorithm": "agentic_grpo_token_mean"}})
    dry, calls = _recording_run(rc=0, out="config ok")
    spec = _emit(tmp_path, dry=dry)
    cfg_path = tmp_path / "agentic_rl_config.json"
    assert spec["kind"] == "agentic_rl"
    assert spec["run_name"] == "agentic-s0"
    assert spec["config_path"] == str(cfg_path)
    assert spec["argv"] == [
        "python3",
        "-m",
        "foundationscale.agentic_rl",
        "--config",
        str(cfg_path),
    ]
    assert spec["dry_run_argv"] == spec["argv"] + ["--dry-run"]
    assert spec["env"] == {}
    assert spec["executable"] is True
    assert spec["missing"] is None
    assert spec["dry_run"] == {"rc": 0, "tail": "config ok"}
    assert json.loads(cfg_path.read_text(encoding="utf-8")) == {
        "policy": {"algorithm": "agentic_grpo_token_mean"}
    }
    assert calls[0][0] == spec["dry_run_argv"]


def test_emit_agentic_wraps_the_fs_module_in_the_declared_enroot_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The argv is the measured launcher's envelope, not a re-assembled one."""

    _patch_skill(monkeypatch)
    dry, calls = _recording_run(rc=0)
    stage = _stage(runtime=_enroot_runtime().to_dict())
    spec = _emit(tmp_path, stage=stage, dry=dry)
    cfg_path = str(tmp_path / "agentic_rl_config.json")
    assert spec["argv"] == [
        "enroot",
        "start",
        "--rw",
        "--mount",
        "/data/shared:/data/shared",
        "--env",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "--env",
        "PYTHONPATH=/fs/src",
        "vllmsrv3",
        "python3",
        "-m",
        "foundationscale.agentic_rl",
        "--config",
        cfg_path,
    ]
    assert spec["dry_run_argv"] == spec["argv"] + ["--dry-run"]
    assert spec["env"] == {"ENROOT_DATA_PATH": "/var/lib/enroot"}
    assert calls[0][1]["env"]["ENROOT_DATA_PATH"] == "/var/lib/enroot"
    assert spec["executable"] is True


def test_emit_agentic_refuses_when_fs_dry_run_does_not_exit_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FS-HO-005: FS owns the schema, so FS's own --dry-run is the authority."""

    _patch_skill(monkeypatch)
    dry, _ = _recording_run(rc=2, err="AgenticConfigRefusal: engine.reload_mode is required")
    spec = _emit(tmp_path, dry=dry)
    assert spec["executable"] is False
    assert "FS-HO-005" in spec["missing"]
    assert "rc=2" in spec["missing"]
    assert "AgenticConfigRefusal" in spec["missing"]
    assert spec["dry_run"] == {"rc": 2, "tail": "AgenticConfigRefusal: engine.reload_mode is required"}
    assert (tmp_path / "agentic_rl_config.json").exists()


def test_emit_agentic_refuses_without_measured_caps_naming_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FS-IN-005: unprobed/unmeasured runtimes are REFUSED and name what to probe."""

    _patch_skill(monkeypatch)
    stage = _stage(runtime=_enroot_runtime().to_dict())
    spec = _emit(tmp_path, stage=stage, caps={"measured": False, "engines_importable": {"vllm": False}})
    assert spec["executable"] is False
    assert "FS-IN-005" in spec["missing"]
    assert "vllmsrv3" in spec["missing"]
    dry, _ = _recording_run()
    missing_caps = emit_agentic_mod.emit_agentic(stage, output_dir=str(tmp_path), run_name="agentic-s0",
                                                 fs_src="/fs/src", agentic_caps=None, dry_run=dry)
    assert missing_caps["executable"] is False
    assert "FS-IN-005" in missing_caps["missing"]


def test_emit_agentic_a_block_gate_refuses_and_a_warn_gate_only_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-IN- rules block; the -PL- handoff note (TR-PL-009) never blocks on its own."""

    _patch_skill(monkeypatch, gates=(("TR-IN-011", "multi-GPU agentic is unmeasured"),))
    blocked = _emit(tmp_path, caps=_caps())
    assert blocked["executable"] is False
    assert "TR-IN-011" in blocked["missing"]

    _patch_skill(monkeypatch, gates=(("TR-PL-009", "agentic_grpo_token_mean is the measured control arm"),))
    warned = _emit(tmp_path, caps=_caps())
    assert warned["executable"] is True
    assert warned["missing"] is None
    assert any("TR-PL-009" in note for note in warned["notes"])


def test_emit_agentic_refuses_when_the_skill_config_builder_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No builder -> no config -> no dry-run -> not executable (never a substitute)."""

    _patch_skill(monkeypatch)
    monkeypatch.setattr(emit_agentic_mod, "build_agentic_config", None)

    def _must_not_run(argv: Any, **kwargs: Any) -> Any:  # pragma: no cover - must not be called
        raise AssertionError(f"FS --dry-run ran without a config: {argv}")

    spec = emit_agentic_mod.emit_agentic(
        _stage(),
        output_dir=str(tmp_path),
        run_name="agentic-s0",
        fs_src="/fs/src",
        agentic_caps=_caps(),
        dry_run=_must_not_run,
    )
    assert spec["executable"] is False
    assert "build_agentic_config" in spec["missing"]
    assert spec["dry_run"] == {"rc": None, "tail": ""}


def test_enroot_rc_override_is_passed_before_the_container() -> None:
    """Real-run finding (A4.2): vllmsrv3's entrypoint is `vllm serve`; without `--rc` (the measured launcher's
    `exec "$@"` script) the probe/dry-run command became `vllm serve` arguments."""
    from foundationskills.interfaces.fs.agentic_probe import Runtime, runtime_argv, runtime_from_dict

    rt = runtime_from_dict({"kind": "enroot", "container": "vllmsrv3", "rc": "/h/fsar-data/rc_exec.sh"})
    argv = runtime_argv(rt, ["python3", "-c", "print(1)"])
    assert argv[argv.index("--rc") + 1] == "/h/fsar-data/rc_exec.sh"
    assert argv.index("--rc") < argv.index("vllmsrv3") < argv.index("python3")
    assert runtime_from_dict(rt.to_dict()) == rt
    assert "--rc" not in runtime_argv(Runtime(kind="enroot", container="c"), ["x"])
