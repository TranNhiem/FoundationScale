"""Tests for ``agentic_rl.config``: the declared, provenance-tracked
``AgenticRLConfig`` loader (JSON file + ``--set dotted.key=value`` overrides).

WHAT IS CLAIMED: every leaf resolves from exactly one of cli/config/default,
recorded in ``provenance``; an unknown key, a wrong type, or a missing
required key is refused naming the dotted key; ``trainer{}``'s allowed keys
are read dynamically from ``RLTrainConfig`` (model/algorithm/rollout_source
refused by name); ``"_doc"`` is stripped everywhere and never read as a
field; a ``--set`` override wins over the file, which wins over the default;
``engine.kind="vllm"`` requires ``served_model_name``; a missing/unreadable
file and invalid JSON are refused, not raised as a bare exception.

WHAT IS NOT CLAIMED: anything about the REAL objects a resolved config can
build (see ``test_cli.py``) -- this module only tests the declarative
resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl.config import (
    AgenticConfigRefusal,
    AgenticRLConfig,
    build_config,
    load_config,
    parse_set_overrides,
)


def _raw_config(**section_overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "policy": {"model_path": "Qwen/Qwen2.5-1.5B-Instruct"},
        "trainer": {"max_steps": 5, "sharding": "none"},
        "engine": {
            "kind": "vllm",
            "command": ["python3", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000"],
            "port": 8000,
            "served_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "reload_mode": "collective_rpc",
        },
        "env": {"backend": "local", "allow_unsandboxed": True},
        "harness": {"parser": "qwen_xml"},
        "budget": {"step_limit": 4, "max_response_tokens": 256, "max_observation_chars": 512},
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 64,
        },
        "rollout": {
            "tasks_path": "tasks.jsonl",
            "tasks_per_step": 1,
            "group_size": 2,
            "max_concurrency": 2,
        },
        "reward": {"abc2midi_bin": "abc2midi"},
        "publish_root": "/tmp/publish",
    }
    for section, overrides in section_overrides.items():
        base[section] = overrides
    return base


def _write_config(tmp_path: Path, raw: dict[str, Any]) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# build_config: happy path, provenance
# ---------------------------------------------------------------------------


def test_build_config_resolves_every_section() -> None:
    config = build_config(_raw_config(), {})
    assert isinstance(config, AgenticRLConfig)
    assert config.policy.model_path == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.policy.algorithm == "agentic_grpo"
    assert config.engine.kind == "vllm"
    assert config.env.backend == "local"
    assert config.harness.parser == "qwen_xml"
    assert config.budget.step_limit == 4
    assert config.sampling.temperature == 1.0
    assert config.rollout.tasks_path == "tasks.jsonl"
    assert config.reward.abc2midi_bin == "abc2midi"
    assert config.publish_root == "/tmp/publish"


def test_provenance_distinguishes_config_cli_and_default() -> None:
    overrides = parse_set_overrides(["trainer.max_steps=100"])
    config = build_config(_raw_config(), overrides)
    assert config.provenance["trainer.max_steps"] == "cli"
    assert config.provenance["trainer.sharding"] == "config"
    assert config.provenance["trainer.learning_rate"] == "default"
    assert config.trainer["max_steps"] == 100
    assert config.trainer["learning_rate"] == 1e-6


def test_policy_servable_publish_defaults_to_true_with_default_provenance() -> None:
    config = build_config(_raw_config(), {})
    assert config.policy.servable_publish is True
    assert config.provenance["policy.servable_publish"] == "default"


def test_policy_servable_publish_from_config_file() -> None:
    raw = _raw_config(policy={"model_path": "m", "servable_publish": False})
    config = build_config(raw, {})
    assert config.policy.servable_publish is False
    assert config.provenance["policy.servable_publish"] == "config"


def test_policy_servable_publish_cli_override_wins_over_config_file() -> None:
    raw = _raw_config(policy={"model_path": "m", "servable_publish": False})
    overrides = parse_set_overrides(["policy.servable_publish=true"])
    config = build_config(raw, overrides)
    assert config.policy.servable_publish is True
    assert config.provenance["policy.servable_publish"] == "cli"


def test_as_json_round_trips_through_json_dumps() -> None:
    config = build_config(_raw_config(), {})
    dumped = json.dumps(config.as_json())
    reloaded = json.loads(dumped)
    assert reloaded["policy"]["model_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert reloaded["provenance"]["policy.algorithm"] == "default"


def test_doc_keys_are_stripped_at_every_level_and_never_read_as_fields() -> None:
    raw = _raw_config()
    raw["_doc"] = "top level"
    raw["policy"]["_doc"] = "policy level"
    raw["trainer"]["_doc"] = "trainer level"
    config = build_config(raw, {})
    assert config.policy.model_path == "Qwen/Qwen2.5-1.5B-Instruct"


# ---------------------------------------------------------------------------
# --set overrides
# ---------------------------------------------------------------------------


def test_parse_set_overrides_rejects_a_malformed_entry() -> None:
    with pytest.raises(AgenticConfigRefusal, match="expected 'dotted.key=value'"):
        parse_set_overrides(["no-equals-sign"])


def test_parse_set_overrides_rejects_an_empty_dotted_key() -> None:
    with pytest.raises(AgenticConfigRefusal):
        parse_set_overrides(["=value"])


def test_parse_set_overrides_rejects_a_repeated_key() -> None:
    with pytest.raises(AgenticConfigRefusal, match="given twice"):
        parse_set_overrides(["trainer.max_steps=1", "trainer.max_steps=2"])


def test_set_override_coerces_int_float_bool_and_str() -> None:
    overrides = parse_set_overrides(
        [
            "trainer.max_steps=7",
            "trainer.learning_rate=0.5",
            "trainer.gradient_checkpointing=true",
            "engine.host=0.0.0.0",
        ]
    )
    config = build_config(_raw_config(), overrides)
    assert config.trainer["max_steps"] == 7
    assert config.trainer["learning_rate"] == 0.5
    assert config.trainer["gradient_checkpointing"] is True
    assert config.engine.host == "0.0.0.0"


def test_set_override_null_clears_an_optional_field() -> None:
    raw = _raw_config()
    raw["trainer"]["save_dir"] = "/some/dir"
    overrides = parse_set_overrides(["trainer.save_dir=null"])
    config = build_config(raw, overrides)
    assert config.trainer["save_dir"] is None
    assert config.provenance["trainer.save_dir"] == "cli"


def test_set_override_rejects_a_bad_bool() -> None:
    overrides = parse_set_overrides(["trainer.gradient_checkpointing=yes"])
    with pytest.raises(AgenticConfigRefusal, match="expected 'true' or 'false'"):
        build_config(_raw_config(), overrides)


def test_set_override_rejects_a_bad_int() -> None:
    overrides = parse_set_overrides(["engine.port=notanint"])
    with pytest.raises(AgenticConfigRefusal, match="not a valid int"):
        build_config(_raw_config(), overrides)


def test_set_override_rejects_a_bad_float() -> None:
    overrides = parse_set_overrides(["budget.step_limit=1", "sampling.temperature=hot"])
    with pytest.raises(AgenticConfigRefusal, match="not a valid float"):
        build_config(_raw_config(), overrides)


def test_set_override_cannot_target_a_mapping_valued_field() -> None:
    overrides = parse_set_overrides(["env.env=whatever"])
    with pytest.raises(AgenticConfigRefusal, match="not settable via --set"):
        build_config(_raw_config(), overrides)


def test_set_override_refuses_engine_command_by_name() -> None:
    # engine.command is a JSON array (the full argv), never settable as a
    # single --set string; it used to be silently filtered into _section_
    # overrides and then never read by any field resolver, dropped with no
    # error at all.
    overrides = parse_set_overrides(['engine.command=["python3"]'])
    with pytest.raises(AgenticConfigRefusal, match="not settable via --set"):
        build_config(_raw_config(), overrides)


def test_set_override_refuses_an_unconsumed_key_typo_in_a_known_section() -> None:
    # Regression: "--set policy.model_paht=/b" (typo'd field name) used to
    # exit 0 with provenance["policy.model_path"] still "config" -- the
    # override was silently dropped because no field resolver was ever asked
    # about the dotted key "policy.model_paht".
    overrides = parse_set_overrides(["policy.model_paht=/b"])
    with pytest.raises(AgenticConfigRefusal, match=r"policy\.model_paht"):
        build_config(_raw_config(), overrides)
    assert build_config(_raw_config(), {}).policy.model_path == "Qwen/Qwen2.5-1.5B-Instruct"


def test_set_override_refuses_an_unconsumed_key_for_an_unknown_section() -> None:
    # Regression: "--set nosuch.section=1" used to exit 0 too -- no builder
    # ever reads a "nosuch.*" dotted key, so it was silently dropped.
    overrides = parse_set_overrides(["nosuch.section=1"])
    with pytest.raises(AgenticConfigRefusal, match=r"nosuch\.section"):
        build_config(_raw_config(), overrides)


def test_set_override_refuses_multiple_unconsumed_keys_naming_both() -> None:
    overrides = parse_set_overrides(["policy.model_paht=/b", "nosuch.section=1"])
    with pytest.raises(AgenticConfigRefusal) as excinfo:
        build_config(_raw_config(), overrides)
    assert "policy.model_paht" in str(excinfo.value)
    assert "nosuch.section" in str(excinfo.value)


def test_engine_reload_mode_set_override_to_null_clears_it() -> None:
    # Regression: "--set engine.reload_mode=null" used to take the raw CLI
    # string "null" (never coerced to None the way every other optional field
    # is), which then failed the "None or 'collective_rpc'" check below with
    # a confusing "is str 'null'" message.
    raw = _raw_config(
        engine={
            "kind": "sglang",
            "command": ["python3", "-m", "sglang.launch_server"],
            "port": 8001,
            "reload_mode": "collective_rpc",
        }
    )
    overrides = parse_set_overrides(["engine.reload_mode=null"])
    config = build_config(raw, overrides)
    assert config.engine.reload_mode is None
    assert config.provenance["engine.reload_mode"] == "cli"


# ---------------------------------------------------------------------------
# refusals: unknown keys, missing required, wrong types
# ---------------------------------------------------------------------------


def test_refuses_an_unknown_top_level_key() -> None:
    raw = _raw_config()
    raw["bogus_section"] = {}
    with pytest.raises(AgenticConfigRefusal, match="bogus_section"):
        build_config(raw, {})


def test_refuses_an_unknown_key_in_a_section() -> None:
    raw = _raw_config(policy={"model_path": "m", "bogus": 1})
    with pytest.raises(AgenticConfigRefusal, match="policy.bogus"):
        build_config(raw, {})


@pytest.mark.parametrize("forbidden", ["model", "algorithm", "rollout_source", "dataset"])
def test_refuses_forbidden_trainer_keys_naming_them(forbidden: str) -> None:
    raw = _raw_config(trainer={forbidden: "x"})
    with pytest.raises(AgenticConfigRefusal, match=f"trainer.{forbidden}"):
        build_config(raw, {})


def test_refuses_dataset_naming_the_reason() -> None:
    # The specific wording the coordinator asked for: the rollout source
    # replaces the corpus, so a declared dataset would never be read.
    raw = _raw_config(trainer={"dataset": "d.jsonl"})
    with pytest.raises(AgenticConfigRefusal, match="declared and never read"):
        build_config(raw, {})


def test_refuses_an_unknown_trainer_key() -> None:
    raw = _raw_config(trainer={"not_a_real_field": 1})
    with pytest.raises(AgenticConfigRefusal, match="trainer.not_a_real_field"):
        build_config(raw, {})


def test_refuses_a_missing_required_key() -> None:
    raw = _raw_config(policy={})
    with pytest.raises(AgenticConfigRefusal, match="policy.model_path"):
        build_config(raw, {})


def test_trainer_section_may_be_entirely_empty() -> None:
    # No trainer key is required any more: model/algorithm/rollout_source/
    # dataset are all forbidden (owned elsewhere or refused outright), and
    # every remaining RLTrainConfig field carries its own default.
    raw = _raw_config(trainer={})
    config = build_config(raw, {})
    assert "dataset" not in config.trainer
    assert config.trainer["max_steps"] == 10
    assert config.provenance["trainer.max_steps"] == "default"


def test_refuses_a_wrong_typed_value() -> None:
    raw = _raw_config(trainer={"max_steps": "five"})
    with pytest.raises(AgenticConfigRefusal, match="trainer.max_steps"):
        build_config(raw, {})


def test_refuses_null_for_a_non_optional_field() -> None:
    raw = _raw_config(policy={"model_path": None})
    with pytest.raises(AgenticConfigRefusal, match="does not accept null"):
        build_config(raw, {})


def test_refuses_a_non_object_section() -> None:
    raw = _raw_config()
    raw["policy"] = "not an object"
    with pytest.raises(AgenticConfigRefusal, match="expected a JSON object"):
        build_config(raw, {})


def test_optional_field_accepts_null_from_json() -> None:
    raw = _raw_config(
        engine={**_raw_config()["engine"], "served_model_name": None, "kind": "sglang"}
    )
    config = build_config(raw, {})
    assert config.engine.served_model_name is None


# ---------------------------------------------------------------------------
# engine section's cross-field rule
# ---------------------------------------------------------------------------


def test_engine_requires_served_model_name_for_vllm() -> None:
    raw = _raw_config()
    del raw["engine"]["served_model_name"]
    with pytest.raises(AgenticConfigRefusal, match="engine.served_model_name"):
        build_config(raw, {})


def test_engine_allows_sglang_without_served_model_name() -> None:
    raw = _raw_config(
        engine={
            "kind": "sglang",
            "command": ["python3", "-m", "sglang.launch_server"],
            "port": 8001,
        }
    )
    config = build_config(raw, {})
    assert config.engine.served_model_name is None


def test_engine_rejects_an_unknown_kind() -> None:
    raw = _raw_config(engine={**_raw_config()["engine"], "kind": "tgi"})
    with pytest.raises(AgenticConfigRefusal, match="engine.kind"):
        build_config(raw, {})


def test_engine_rejects_a_non_array_command() -> None:
    raw = _raw_config(engine={**_raw_config()["engine"], "command": "python3 -m vllm"})
    with pytest.raises(AgenticConfigRefusal, match="engine.command"):
        build_config(raw, {})


def test_engine_rejects_an_unknown_weight_sync_mode() -> None:
    raw = _raw_config(engine={**_raw_config()["engine"], "weight_sync_mode": "hot_swap"})
    with pytest.raises(AgenticConfigRefusal, match="weight_sync_mode"):
        build_config(raw, {})


def test_engine_weight_sync_mode_cli_override() -> None:
    raw = _raw_config()
    # weight_sync_mode="restart" requires the command to declare the
    # "{model_path}" placeholder (see test_engine_restart_mode_refuses_a_command_
    # with_no_placeholder below) -- the base command has none.
    raw["engine"]["command"] = [*raw["engine"]["command"], "--model", "{model_path}"]
    overrides = parse_set_overrides(["engine.weight_sync_mode=restart"])
    config = build_config(raw, overrides)
    assert config.engine.weight_sync_mode == "restart"
    assert config.provenance["engine.weight_sync_mode"] == "cli"


def test_engine_refuses_vllm_reload_endpoint_with_no_reload_mode() -> None:
    raw = _raw_config()
    del raw["engine"]["reload_mode"]
    with pytest.raises(AgenticConfigRefusal, match="engine.reload_mode"):
        build_config(raw, {})


def test_engine_restart_mode_needs_no_reload_mode() -> None:
    raw = _raw_config()
    del raw["engine"]["reload_mode"]
    raw["engine"]["weight_sync_mode"] = "restart"
    raw["engine"]["command"] = [*raw["engine"]["command"], "--model", "{model_path}"]
    config = build_config(raw, {})
    assert config.engine.reload_mode is None


def test_engine_restart_mode_refuses_a_command_with_no_placeholder() -> None:
    # Regression: weight_sync.DiskWeightSync._restart_push only discovers a
    # missing "{model_path}" placeholder at PUSH time, after a GPU is already
    # resident; config.py now refuses the same declaration at config time,
    # naming engine.command.
    raw = _raw_config()
    del raw["engine"]["reload_mode"]
    raw["engine"]["weight_sync_mode"] = "restart"
    with pytest.raises(AgenticConfigRefusal, match="engine.command"):
        build_config(raw, {})


def test_engine_sglang_needs_no_reload_mode() -> None:
    raw = _raw_config(
        engine={
            "kind": "sglang",
            "command": ["python3", "-m", "sglang.launch_server"],
            "port": 8001,
        }
    )
    config = build_config(raw, {})
    assert config.engine.reload_mode is None
    assert config.provenance["engine.reload_mode"] == "default"


def test_engine_rejects_an_unknown_reload_mode() -> None:
    raw = _raw_config()
    raw["engine"]["reload_mode"] = "rpc_v2"
    with pytest.raises(AgenticConfigRefusal, match="engine.reload_mode"):
        build_config(raw, {})


def test_engine_env_resolves_and_is_provenance_tracked() -> None:
    raw = _raw_config()
    raw["engine"]["env"] = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"}
    config = build_config(raw, {})
    assert config.engine.env == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"}
    assert config.provenance["engine.env"] == "config"


def test_engine_env_defaults_to_empty() -> None:
    config = build_config(_raw_config(), {})
    assert config.engine.env == {}
    assert config.provenance["engine.env"] == "default"


def test_engine_env_rejects_non_str_values() -> None:
    raw = _raw_config()
    raw["engine"]["env"] = {"PORT": 8000}
    with pytest.raises(AgenticConfigRefusal, match="engine.env"):
        build_config(raw, {})


def test_engine_env_cannot_be_set_via_cli() -> None:
    overrides = parse_set_overrides(["engine.env=whatever"])
    with pytest.raises(AgenticConfigRefusal, match="not settable via --set"):
        build_config(_raw_config(), overrides)


# ---------------------------------------------------------------------------
# harness/reward Literal-only fields
# ---------------------------------------------------------------------------


def test_harness_rejects_an_unknown_parser() -> None:
    raw = _raw_config(harness={"parser": "llama_special"})
    with pytest.raises(AgenticConfigRefusal, match="harness.parser"):
        build_config(raw, {})


def test_harness_requires_a_parser() -> None:
    raw = _raw_config(harness={})
    with pytest.raises(AgenticConfigRefusal, match="harness.parser"):
        build_config(raw, {})


def test_harness_rejects_a_non_native_tool_loop_name() -> None:
    raw = _raw_config(harness={"parser": "qwen_xml", "name": "custom_loop"})
    with pytest.raises(AgenticConfigRefusal, match="harness.name"):
        build_config(raw, {})


def test_harness_rejects_a_non_default_tools_value() -> None:
    raw = _raw_config(harness={"parser": "qwen_xml", "tools": "custom"})
    with pytest.raises(AgenticConfigRefusal, match="harness.tools"):
        build_config(raw, {})


def test_reward_rejects_a_non_music_kind() -> None:
    raw = _raw_config(reward={"kind": "code", "abc2midi_bin": "abc2midi"})
    with pytest.raises(AgenticConfigRefusal, match="reward.kind"):
        build_config(raw, {})


def test_reward_requires_abc2midi_bin() -> None:
    raw = _raw_config(reward={})
    with pytest.raises(AgenticConfigRefusal, match="reward.abc2midi_bin"):
        build_config(raw, {})


# ---------------------------------------------------------------------------
# load_config: file I/O and JSON parsing
# ---------------------------------------------------------------------------


def test_load_config_reads_a_real_file(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _raw_config())
    config = load_config(path, [])
    assert config.policy.model_path == "Qwen/Qwen2.5-1.5B-Instruct"


def test_load_config_applies_set_overrides(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _raw_config())
    config = load_config(path, ["publish_root=/elsewhere"])
    assert config.publish_root == "/elsewhere"
    assert config.provenance["publish_root"] == "cli"


def test_load_config_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AgenticConfigRefusal, match="could not be read"):
        load_config(str(tmp_path / "does_not_exist.json"), [])


def test_load_config_refuses_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(AgenticConfigRefusal, match="invalid JSON"):
        load_config(str(path), [])


def test_load_config_refuses_a_non_object_json_document(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(AgenticConfigRefusal, match="expected a JSON object"):
        load_config(str(path), [])


def test_engine_request_timeout_defaults_long_and_is_provenance_tracked() -> None:
    # Regression from the GPU smoke run: 12k-token generations outlasted a 60 s
    # client timeout and every episode abstained as infra.
    raw = _raw_config()
    cfg = build_config(raw, {})
    assert cfg.engine.request_timeout_s == 600.0
    assert cfg.provenance["engine.request_timeout_s"] == "default"
    raw["engine"]["request_timeout_s"] = 1800.0
    cfg2 = build_config(raw, {})
    assert cfg2.engine.request_timeout_s == 1800.0
    assert cfg2.provenance["engine.request_timeout_s"] == "config"
