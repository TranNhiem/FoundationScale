"""Unit tests for the agentic training skill: plan, TR-* gates, FS config rendering.

No subprocess and no GPU: capability probes are injected as plain mappings, and the two
handles the TR-IN-012 gate reads (abc2midi_bin, rollout.tasks_path) are tmp files whose
executability/readability is measured for real via os.access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foundationskills.skills.training.agentic import (
    BLOCK_RULES,
    GATE_SEVERITY,
    agentic_gates,
    blocking_findings,
    build_agentic_config,
    plan_agentic,
)

# The field names foundationscale/agentic_rl/config.py declares per section (dataclass
# fields at config.py:256-413): build_agentic_config may write nothing else.
_CONFIG_SECTION_KEYS: dict[str, set[str]] = {
    "policy": {"model_path", "algorithm", "servable_publish", "keep_last", "archive_every"},
    "engine": {
        "kind", "command", "port", "served_model_name", "startup_timeout_s",
        "request_timeout_s", "weight_sync_mode", "host", "log_dir", "allow_stale_policy",
        "max_policy_lag", "reload_mode", "env",
    },
    "env": {"backend", "exec_timeout_s", "episode_timeout_s", "max_output_bytes", "env",
            "allow_unsandboxed"},
    "harness": {"parser", "name", "tools"},
    "budget": {"step_limit", "max_response_tokens", "max_observation_chars"},
    "sampling": {"temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "repetition_penalty", "max_new_tokens"},
    "rollout": {"tasks_path", "tasks_per_step", "group_size", "max_concurrency",
                "group_by_harness", "seed", "max_infra_rate"},
    "reward": {"kind", "abc2midi_bin", "timeout_s"},
    "stability": {"ema_alpha", "drop_from_peak", "min_steps", "best_k", "action"},
}
_TRAINER_KEYS = {"max_steps", "learning_rate", "logprob_micro_batch", "sharding",
                 "save_dir", "seed"}
_TRAINER_FORBIDDEN = {"model", "algorithm", "rollout_source", "dataset"}
_PROBE_PHRASE = "run the agentic probe in the declared runtime"


def _handles(tmp_path: Path) -> tuple[str, str]:
    """(readable tasks.jsonl, executable abc2midi) on the host -- TR-IN-012's handles."""
    tasks = tmp_path / "music_tasks.jsonl"
    tasks.write_text('{"task": "compose"}\n', encoding="utf-8")
    abc = tmp_path / "abc2midi"
    abc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    abc.chmod(0o755)
    return str(tasks), str(abc)


def _goal(tmp_path: Path, **over: Any) -> dict[str, Any]:
    """One complete agentic goal block, as the A4.2 contract spells its keys."""
    tasks_path, abc2midi_bin = _handles(tmp_path)
    block: dict[str, Any] = {
        "algorithm": "agentic_grpo",
        "model": "/mnt/models/Qwen2.5-1.5B-Instruct",
        "tasks_path": tasks_path,
        "reward": {"kind": "music", "abc2midi_bin": abc2midi_bin},
        "engine": {
            "kind": "vllm",
            "served_model_name": "qwen-served",
            "port": 8000,
            "weight_sync_mode": "reload_endpoint",
            "allow_stale_policy": False,
        },
        "env": {"backend": "local", "allow_unsandboxed": True},
        "harness": {"parser": "qwen_xml"},
        "rollout": {"group_size": 2, "tasks_per_step": 1, "max_concurrency": 2},
        "trainer": {"max_steps": 3, "lr": 1e-5, "sharding": "none"},
        # declared values are deliberately NOT the music_s0.json ones, so a fill that
        # overwrote a declaration would show.
        "budget": {"step_limit": 4, "max_response_tokens": 256, "max_observation_chars": 1024},
        "sampling": {
            "temperature": 0.7, "top_p": 0.9, "top_k": 0, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0, "max_new_tokens": 128,
        },
        "gpus": 1,
        "publish_root": str(tmp_path / "publish"),
        "stability": None,
    }
    block.update(over)
    return {"agentic": block}


def _stage(tmp_path: Path, **over: Any) -> dict[str, Any]:
    return plan_agentic(_goal(tmp_path, **over))


def _caps(**over: Any) -> dict[str, Any]:
    caps: dict[str, Any] = {
        "measured": True,
        "algorithms": ["agentic_grpo", "agentic_grpo_token_mean"],
        "engines_importable": {"vllm": True, "sglang": False},
        "env_backends": ["local"],
        "reward_kinds": ["music"],
        "parsers": ["qwen_xml", "hermes_json"],
        "reward_handles": {},
        "runtime": {"kind": "host"},
        "multi_gpu": None,
    }
    caps.update(over)
    return caps


def _rules(findings: list[tuple[str, str]]) -> list[str]:
    return [rule for rule, _ in findings]


def _messages(findings: list[tuple[str, str]]) -> str:
    return "\n".join(message for _, message in findings)


# ------------------------------------------------------------------------ plan_agentic


def test_plan_agentic_stage_shape(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    assert set(stage) == {
        "name", "stage", "algorithm", "method", "agentic", "runtime", "gpus", "executable",
    }
    assert stage["name"] == "agentic_rl"
    assert stage["stage"] == "agentic_rl"
    assert stage["algorithm"] == "agentic_grpo"
    assert stage["method"] == "full"
    assert stage["executable"] is None  # executability is measured, never planned
    assert stage["gpus"] == 1
    assert stage["runtime"]["kind"] == "host"  # contract: absent runtime == host


def test_plan_agentic_declarations_pass_through(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    block = _stage(tmp_path)["agentic"]
    assert block["model"] == goal["agentic"]["model"]
    assert block["algorithm"] == "agentic_grpo"
    assert block["tasks_path"] == goal["agentic"]["tasks_path"]
    assert block["budget"] == goal["agentic"]["budget"]
    assert block["sampling"] == goal["agentic"]["sampling"]
    assert block["rollout"]["group_size"] == 2
    assert block["rollout"]["tasks_per_step"] == 1
    assert block["trainer"]["max_steps"] == 3
    assert block["trainer"]["lr"] == 1e-5


def test_plan_agentic_defaults_only_from_music_s0(tmp_path: Path) -> None:
    """Fills allowed only where examples/agentic_rl/music_s0.json declares a value."""
    block = _stage(tmp_path)["agentic"]
    assert block["harness"]["name"] == "native_tool_loop"      # music_s0.json:53
    assert block["harness"]["tools"] == "default"              # music_s0.json:55
    assert block["reward"]["timeout_s"] == 60.0                # music_s0.json:86
    assert block["env"]["exec_timeout_s"] == 60.0              # music_s0.json:45
    assert block["env"]["episode_timeout_s"] == 600.0          # music_s0.json:46
    assert block["env"]["max_output_bytes"] == 65536           # music_s0.json:47
    assert block["engine"]["startup_timeout_s"] == 300.0       # music_s0.json:29
    assert block["engine"]["host"] == "127.0.0.1"              # music_s0.json:32
    assert block["engine"]["reload_mode"] == "collective_rpc"  # music_s0.json:31
    assert block["rollout"]["group_by_harness"] is False       # music_s0.json:79
    assert block["rollout"]["seed"] == 0                       # music_s0.json:80
    assert block["trainer"]["logprob_micro_batch"] == 0        # music_s0.json:12
    assert block["trainer"]["seed"] == 0                       # music_s0.json:15


def test_plan_agentic_fills_nothing_the_example_does_not_declare(tmp_path: Path) -> None:
    """music_s0.json never declares these: absence stays absence, never a guess."""
    goal = _goal(tmp_path, budget=None, sampling=None)  # type: ignore[arg-type]
    goal["agentic"].pop("budget")
    goal["agentic"].pop("sampling")
    goal["agentic"]["engine"].pop("allow_stale_policy", None)  # the helper declares it; absence is what is tested
    block = plan_agentic(goal)["agentic"]
    for absent in ("request_timeout_s", "max_policy_lag", "allow_stale_policy", "command"):
        assert absent not in block["engine"], absent
    for absent in ("log_dir", "env"):
        assert absent not in block["engine"], absent
    assert "save_dir" not in block["trainer"]
    assert "max_infra_rate" not in block["rollout"]
    assert "tasks_path" not in block["rollout"]          # declared at block level instead
    for absent in ("servable_publish", "keep_last", "archive_every"):
        assert absent not in block["policy"], absent
    assert block["stability"] is None
    assert block["budget"] == {}                          # nothing invented
    assert block["sampling"] == {}


def test_plan_agentic_declared_values_win_over_fills(tmp_path: Path) -> None:
    block = _stage(
        tmp_path,
        harness={"parser": "qwen_xml", "name": "own_loop", "tools": "own"},
        engine={
            "kind": "vllm",
            "served_model_name": "qwen-served",
            "port": 8000,
            "weight_sync_mode": "reload_endpoint",
            "allow_stale_policy": True,
            "request_timeout_s": 1.0,
            "startup_timeout_s": 5.0,
        },
    )["agentic"]
    assert block["harness"]["name"] == "own_loop"
    assert block["harness"]["tools"] == "own"
    assert block["engine"]["startup_timeout_s"] == 5.0
    assert block["engine"]["request_timeout_s"] == 1.0
    assert block["engine"]["allow_stale_policy"] is True


def test_plan_agentic_runtime_enroot_declared(tmp_path: Path) -> None:
    stage = _stage(
        tmp_path,
        runtime={"kind": "enroot", "container": "vllmsrv3", "mounts": [("/home/me", "/home/me")]},
    )
    assert stage["runtime"]["kind"] == "enroot"
    assert stage["runtime"]["container"] == "vllmsrv3"
    assert stage["agentic"]["runtime"]["kind"] == "enroot"


def test_plan_agentic_runtime_defaults_to_host(tmp_path: Path) -> None:
    assert _stage(tmp_path, runtime=None)["runtime"]["kind"] == "host"


def test_plan_agentic_requires_an_agentic_block() -> None:
    with pytest.raises(ValueError):
        plan_agentic({})
    with pytest.raises(ValueError):
        plan_agentic({"agentic": None})
    with pytest.raises(ValueError):
        plan_agentic({"agentic": "music_s0"})


def test_plan_agentic_accepts_the_block_directly(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    assert plan_agentic(goal["agentic"])["algorithm"] == "agentic_grpo"


def test_plan_agentic_rejects_non_mapping_section(tmp_path: Path) -> None:
    goal = _goal(tmp_path, engine="vllm")
    with pytest.raises(ValueError):
        plan_agentic(goal)


def test_plan_agentic_rejects_malformed_gpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _stage(tmp_path, gpus=0)
    with pytest.raises(ValueError):
        _stage(tmp_path, gpus="one")


# ----------------------------------------------------------------------- agentic_gates


def test_agentic_gates_measured_control_is_clean(tmp_path: Path) -> None:
    assert agentic_gates(_stage(tmp_path), _caps()) == []


def test_agentic_gates_tr_in_010_unknown_algorithm(tmp_path: Path) -> None:
    caps = _caps(algorithms=["agentic_grpo_token_mean"])  # measured, just not this one
    findings = agentic_gates(_stage(tmp_path), caps)
    assert _rules(findings) == ["TR-IN-010"]
    assert "algorithm 'agentic_grpo'" in _messages(findings)
    assert _PROBE_PHRASE in _messages(findings)


def test_agentic_gates_tr_in_010_engine_not_importable(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    goal["agentic"]["engine"]["kind"] = "sglang"
    goal["agentic"]["engine"]["weight_sync_mode"] = "restart"
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert _rules(findings) == ["TR-IN-010"]
    assert "engine.kind 'sglang'" in _messages(findings)
    assert _PROBE_PHRASE in _messages(findings)


def test_agentic_gates_tr_in_010_unknown_env_backend(tmp_path: Path) -> None:
    goal = _goal(tmp_path, env={"backend": "harbor-gone", "allow_unsandboxed": False})
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert _rules(findings) == ["TR-IN-010"]
    assert "env.backend 'harbor-gone'" in _messages(findings)


def test_agentic_gates_tr_in_010_unknown_reward_kind(tmp_path: Path) -> None:
    goal = _goal(tmp_path, reward={"kind": "rl_env", "abc2midi_bin": "/bin/true"})
    findings = agentic_gates(plan_agentic(goal), _caps(reward_kinds=["music"]))
    assert _rules(findings) == ["TR-IN-010"]
    assert "reward.kind 'rl_env'" in _messages(findings)


def test_agentic_gates_tr_in_010_unknown_parser(tmp_path: Path) -> None:
    goal = _goal(tmp_path, harness={"parser": "xml_loose"})
    findings = agentic_gates(plan_agentic(goal), _caps(parsers=["qwen_xml", "hermes_json"]))
    assert _rules(findings) == ["TR-IN-010"]
    assert "harness.parser 'xml_loose'" in _messages(findings)


def test_agentic_gates_tr_in_010_unmeasured_caps(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    for caps in (None, _caps(measured=False), {k: v for k, v in _caps().items() if k != "algorithms"}):
        findings = agentic_gates(stage, caps)
        assert all(rule == "TR-IN-010" for rule in _rules(findings)), caps
        assert _rules(findings).count("TR-IN-010") >= 1
        assert _PROBE_PHRASE in _messages(findings), caps
    # and a probe result that simply recorded no names (e.g. parsers not introspectable)
    findings = agentic_gates(stage, _caps(parsers=[]))
    assert _rules(findings) == ["TR-IN-010"]
    assert "UNMEASURED" in _messages(findings) or "[]" in _messages(findings)


def test_agentic_gates_tr_in_011_multi_gpu_plan(tmp_path: Path) -> None:
    findings = agentic_gates(_stage(tmp_path, gpus=2), _caps())
    assert _rules(findings) == ["TR-IN-011"]
    assert "gpus=2" in _messages(findings)


def test_agentic_gates_tr_in_011_declared_sharding(tmp_path: Path) -> None:
    goal = _goal(tmp_path, trainer={"max_steps": 3, "lr": 1e-5, "sharding": "fsdp"})
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert _rules(findings) == ["TR-IN-011"]
    assert "trainer.sharding='fsdp'" in _messages(findings)


def test_agentic_gates_tr_in_011_undeclared_sharding_is_not_assumed(tmp_path: Path) -> None:
    # >1 GPU or a sharding that is not the measured "none" (including none at all) is
    # refused while agentic multi-GPU is UNMEASURED -- never assumed.
    goal = _goal(tmp_path)
    goal["agentic"]["trainer"] = {"max_steps": 3, "lr": 1e-5}
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert _rules(findings) == ["TR-IN-011"]
    assert "undeclared" in _messages(findings)


def test_agentic_gates_tr_in_011_measured_multi_gpu_ok(tmp_path: Path) -> None:
    goal = _goal(tmp_path, gpus=2, trainer={"max_steps": 3, "lr": 1e-5, "sharding": "fsdp"})
    findings = agentic_gates(plan_agentic(goal), _caps(multi_gpu=True))
    assert "TR-IN-011" not in _rules(findings)


def test_agentic_gates_tr_in_012_missing_handle_declaration(tmp_path: Path) -> None:
    goal = _goal(tmp_path, reward={"kind": "music"})
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert "TR-IN-012" in _rules(findings)
    assert "abc2midi_bin" in _messages(findings)
    assert "no score" in _messages(findings)


def test_agentic_gates_tr_in_012_non_executable_handle(tmp_path: Path) -> None:
    plain = tmp_path / "not_executable"
    plain.write_text("data\n", encoding="utf-8")
    plain.chmod(0o644)
    goal = _goal(tmp_path, reward={"kind": "music", "abc2midi_bin": str(plain)})
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert "TR-IN-012" in _rules(findings)
    assert str(plain) in _messages(findings)


def test_agentic_gates_tr_in_012_absent_handle_binary(tmp_path: Path) -> None:
    goal = _goal(tmp_path, reward={"kind": "music", "abc2midi_bin": "abc2midi-definitely-absent"})
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert "TR-IN-012" in _rules(findings)
    assert "abc2midi-definitely-absent" in _messages(findings)


def test_agentic_gates_tr_in_012_tasks_path_unreadable(tmp_path: Path) -> None:
    goal = _goal(tmp_path, tasks_path=str(tmp_path / "no_such_tasks.jsonl"))
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert "TR-IN-012" in _rules(findings)
    assert "tasks_path" in _messages(findings)


def test_agentic_gates_tr_in_012_env_backend_unset(tmp_path: Path) -> None:
    goal = _goal(tmp_path, env={"backend": "", "allow_unsandboxed": True})
    findings = agentic_gates(plan_agentic(goal), _caps(env_backends=["local", ""]))
    assert "TR-IN-012" in _rules(findings)
    assert "env.backend" in _messages(findings)


def test_agentic_gates_tr_in_012_allow_unsandboxed_opt_in(tmp_path: Path) -> None:
    goal = _goal(tmp_path, env={"backend": "local", "allow_unsandboxed": False})
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert "TR-IN-012" in _rules(findings)
    assert "allow_unsandboxed" in _messages(findings)
    assert "isolation 'none'" in _messages(findings)


def test_agentic_gates_tr_in_013_no_weight_sync_multi_step(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    goal["agentic"]["engine"]["weight_sync_mode"] = None
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert _rules(findings) == ["TR-IN-013"]
    assert "trainer.max_steps=3" in _messages(findings)


def test_agentic_gates_tr_in_013_near_misses(tmp_path: Path) -> None:
    # one step needs no weight sync at all ...
    one_step = _goal(tmp_path)
    one_step["agentic"]["trainer"] = {"max_steps": 1, "lr": 1e-5, "sharding": "none"}
    one_step["agentic"]["engine"]["weight_sync_mode"] = None
    assert _rules(agentic_gates(plan_agentic(one_step), _caps())) == []
    # ... a declared weight sync satisfies the rule ...
    synced = _goal(tmp_path)
    synced["agentic"]["engine"]["weight_sync_mode"] = "reload_endpoint"
    assert "TR-IN-013" not in _rules(agentic_gates(plan_agentic(synced), _caps()))
    # ... and so does the EXPLICIT staleness opt-in (a smoke run that says so).
    stale = _goal(tmp_path)
    stale["agentic"]["engine"]["weight_sync_mode"] = None
    stale["agentic"]["engine"]["allow_stale_policy"] = True
    assert "TR-IN-013" not in _rules(agentic_gates(plan_agentic(stale), _caps()))


def test_agentic_gates_tr_pl_009_default_arm(tmp_path: Path) -> None:
    goal = _goal(tmp_path, algorithm="agentic_grpo_token_mean", default_arm=True)
    findings = agentic_gates(plan_agentic(goal), _caps())
    assert _rules(findings) == ["TR-PL-009"]
    assert "default arm" in _messages(findings)
    # near miss: the measured control NOT declared as the default arm stays a silent plan
    goal["agentic"]["default_arm"] = False
    assert _rules(agentic_gates(plan_agentic(goal), _caps())) == []


def test_gate_severity_map() -> None:
    assert GATE_SEVERITY == {
        "TR-IN-010": "BLOCK",
        "TR-IN-011": "BLOCK",
        "TR-IN-012": "BLOCK",
        "TR-IN-013": "BLOCK",
        "TR-IN-014": "BLOCK",
        "TR-PL-009": "WARN",
    }
    assert BLOCK_RULES == frozenset({"TR-IN-010", "TR-IN-011", "TR-IN-012", "TR-IN-013", "TR-IN-014"})
    warn_only = [("TR-PL-009", "note")]
    assert blocking_findings(warn_only) == []


# ------------------------------------------------------------- build_agentic_config


def test_build_agentic_config_sections_mirror_music_s0(tmp_path: Path) -> None:
    config = build_agentic_config(_stage(tmp_path))
    assert list(config) == [
        "policy", "trainer", "engine", "env", "harness", "budget", "sampling",
        "rollout", "reward", "publish_root",
    ]
    assert config["policy"] == {
        "model_path": "/mnt/models/Qwen2.5-1.5B-Instruct",
        "algorithm": "agentic_grpo",
    }
    assert config["harness"] == {"name": "native_tool_loop", "parser": "qwen_xml",
                                 "tools": "default"}
    assert config["reward"] == {"kind": "music", "timeout_s": 60.0,
                                **{"abc2midi_bin": config["reward"]["abc2midi_bin"]}}
    assert config["publish_root"]
    assert "provenance" not in config          # the loader records provenance itself
    assert "stability" not in config           # None == "the section was absent"


def test_build_agentic_config_policy_and_rollout_mapping(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    config = build_agentic_config(plan_agentic(goal))
    assert config["policy"]["model_path"] == goal["agentic"]["model"]
    assert config["rollout"]["tasks_path"] == goal["agentic"]["tasks_path"]
    assert config["rollout"]["tasks_per_step"] == 1
    assert config["rollout"]["group_size"] == 2
    assert "tasks_path" not in config["policy"]  # only policy's own fields land there


def test_build_agentic_config_engine_command_from_the_measured_gb200_recipe(tmp_path: Path) -> None:
    # the measured GB200 single-GPU recipe (ablate_tm_s3) with --served-model-name / --port substituted;
    # 0.5 because the trainer shares the GPU (music_s0's 0.85 assumes a separate engine GPU)
    config = build_agentic_config(_stage(tmp_path))
    assert config["engine"]["command"] == [
        "vllm", "serve", "/mnt/models/Qwen2.5-1.5B-Instruct",
        "--served-model-name", "qwen-served",
        "--port", "8000",
        "--gpu-memory-utilization", "0.5", "--max-model-len", "16384",
    ]
    assert config["engine"]["kind"] == "vllm"
    assert config["engine"]["reload_mode"] == "collective_rpc"


def test_build_agentic_config_declared_command_wins(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    goal["agentic"]["engine"]["command"] = (
        "python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "own"
    )
    config = build_agentic_config(plan_agentic(goal))
    assert config["engine"]["command"] == [
        "python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "own"
    ]


def test_build_agentic_config_no_command_synthesis_outside_vllm(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    goal["agentic"]["engine"]["kind"] = "sglang"
    goal["agentic"]["engine"]["weight_sync_mode"] = "restart"
    config = build_agentic_config(plan_agentic(goal))
    assert "command" not in config["engine"]      # nothing to interpolate the example with
    assert "reload_mode" not in config["engine"]


def test_build_agentic_config_trainer_alias_and_forbidden_keys(tmp_path: Path) -> None:
    goal = _goal(
        tmp_path,
        trainer={
            "max_steps": 3,
            "lr": 1e-5,
            "sharding": "none",
            "save_dir": str(tmp_path / "ckpt"),
            "model": "m",
            "algorithm": "a",
            "rollout_source": "r",
            "dataset": "d",
        },
    )
    trainer = build_agentic_config(plan_agentic(goal))["trainer"]
    assert trainer["learning_rate"] == 1e-5     # "lr" is the goal spelling
    assert trainer["max_steps"] == 3
    assert trainer["sharding"] == "none"
    assert trainer["save_dir"] == str(tmp_path / "ckpt")
    assert not (set(trainer) & _TRAINER_FORBIDDEN)


def test_build_agentic_config_writes_only_declared_config_keys(tmp_path: Path) -> None:
    config = build_agentic_config(_stage(tmp_path))
    for section, keys in _CONFIG_SECTION_KEYS.items():
        if section in config:
            assert set(config[section]) <= keys, section
    assert set(config["trainer"]) <= _TRAINER_KEYS
    assert not (set(config["trainer"]) & _TRAINER_FORBIDDEN)


def test_build_agentic_config_never_forbidden_trainer_keys_from_any_block(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    goal["agentic"]["policy"] = {"model_path": "other", "servable_publish": False}
    config = build_agentic_config(plan_agentic(goal))
    assert config["policy"]["servable_publish"] is False   # policy field, allowed
    assert not (set(config["trainer"]) & _TRAINER_FORBIDDEN)
    assert "dataset" not in config and "rollout_source" not in config


def test_build_agentic_config_stability_optional(tmp_path: Path) -> None:
    assert "stability" not in build_agentic_config(_stage(tmp_path))
    goal = _goal(
        tmp_path,
        stability={"ema_alpha": 0.5, "drop_from_peak": 0.4, "min_steps": 5,
                   "best_k": 2, "action": "stop", "surprise": 1},
    )
    config = build_agentic_config(plan_agentic(goal))
    assert config["stability"] == {"ema_alpha": 0.5, "drop_from_peak": 0.4, "min_steps": 5,
                                  "best_k": 2, "action": "stop"}


def test_build_agentic_config_request_timeout_s_none_stays_undeclared(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    goal["agentic"]["engine"]["request_timeout_s"] = None   # declaration of absence
    config = build_agentic_config(plan_agentic(goal))
    assert "request_timeout_s" not in config["engine"]     # FS's own default + provenance
    goal["agentic"]["engine"]["request_timeout_s"] = 1.0   # the forced-infra ablation arm
    config = build_agentic_config(plan_agentic(goal))
    assert config["engine"]["request_timeout_s"] == 1.0


def test_build_agentic_config_declared_false_and_zero_survive(tmp_path: Path) -> None:
    goal = _goal(tmp_path)
    goal["agentic"]["env"]["allow_unsandboxed"] = False
    goal["agentic"]["sampling"]["top_k"] = 0
    goal["agentic"]["engine"]["request_timeout_s"] = 0.0
    config = build_agentic_config(plan_agentic(goal))
    assert config["env"]["allow_unsandboxed"] is False
    assert config["sampling"]["top_k"] == 0
    assert config["engine"]["request_timeout_s"] == 0.0


def test_engine_env_is_built_from_the_measured_recipe_and_gated_when_declared(tmp_path: Path) -> None:
    """Real-GPU finding (A4.2): FS gives the vLLM server a COMPLETE env (never merged with os.environ); with no
    engine.env the server died on libcudart.so.12. The builder fills the measured GB200 env; TR-IN-014 gates a
    declared env that lacks PATH, or a dev-RPC reload without VLLM_SERVER_DEV_MODE=1."""
    from foundationskills.skills.training.agentic import agentic_gates, build_agentic_config

    goal = _goal(tmp_path)
    goal["agentic"]["engine"].update(weight_sync_mode="reload_endpoint", reload_mode="collective_rpc")
    goal["agentic"]["engine"].pop("env", None)
    goal["agentic"]["engine"].pop("command", None)
    cfg = build_agentic_config(plan_agentic(goal))
    env = cfg["engine"]["env"]
    assert env["PATH"] and env["VLLM_SERVER_DEV_MODE"] == "1" and "CUDA_VISIBLE_DEVICES" in env
    assert cfg["engine"]["command"][:2] == ["vllm", "serve"] and "0.5" in cfg["engine"]["command"]
    assert "max_model_len" not in cfg["engine"]
    goal["agentic"]["engine"]["env"] = {"HOME": "/h"}
    ids = [rid for rid, _ in agentic_gates(plan_agentic(goal), None)]
    assert ids.count("TR-IN-014") == 2
