
"""Unit tests for post-training stage/algorithm selection with a fake caps."""
from __future__ import annotations

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.training.posttrain import select_algorithm, select_posttrain_stages


def _fake_caps(**over) -> FSCapabilities:
    base = dict(
        available=True,
        fs_version="0.0.0+77bfa65",
        train_flags=frozenset({"--model", "--dataset", "--output-dir"}),
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=("dr_grpo", "gspo", "dapo", "grpo", "ppo", "dpo"),
        rl_runnable={
            "dr_grpo": None,
            "gspo": None,
            "dapo": None,
            "grpo": "refused: needs a reference model (kl_weight != 0)",
            "ppo": "refused: doesn't expose the declared axes",
            "dpo": "refused: preference/online family is not wired to this loop",
        },
        families={"gemma4": ("gemma4", "gemma4_text"), "qwen3.5": ("qwen3_5",)},
        backends=("ddp", "fsdp"),
    )
    base.update(over)
    return FSCapabilities(**base)


def test_rl_reroutes_grpo_to_reference_free() -> None:
    sel = select_algorithm("rl", {}, {"requested_rl_algorithm": "grpo", "has_verifiable_answers": True}, _fake_caps())
    assert sel["algorithm"] == "dr_grpo"
    assert sel["runnable"] is True
    assert "reference-free" in sel["because"]
    assert "GRPO" in sel["because"]
    assert sel["fallback"] in {"gspo", "dapo"}
    assert sel["missing"] is None


def test_rl_keeps_requested_when_runnable() -> None:
    sel = select_algorithm("rl", {"rl_algorithm": "dapo"}, {}, _fake_caps())
    assert sel["algorithm"] == "dapo"
    assert sel["runnable"] is True


def test_rl_unrunnable_when_registry_empty() -> None:
    caps = _fake_caps(rl_algorithms=(), rl_runnable={})
    sel = select_algorithm("rl", {}, {"requested_rl_algorithm": "grpo"}, caps)
    assert sel["runnable"] is False
    assert sel["algorithm"] == "grpo"
    assert sel["missing"]
    assert sel["fallback"] is None


def test_preference_proposes_rl_fallback_with_answers() -> None:
    sel = select_algorithm("preference", {}, {"has_verifiable_answers": True, "has_pairs": True}, _fake_caps())
    assert sel["runnable"] is False
    assert sel["fallback"] in {"dr_grpo", "gspo", "dapo"}
    assert "RL stage with verifiable rewards" in sel["because"]


def test_preference_without_answers_stays_non_executable() -> None:
    sel = select_algorithm("preference", {}, {"has_pairs": True}, _fake_caps())
    assert sel["runnable"] is False
    assert sel["fallback"] is None
    assert "cannot run it" in sel["because"]
    assert sel["missing"]


def test_supervised_stages_use_fs_objective_facts() -> None:
    for stage, expected in (("cpt", "causal_lm"), ("pretrain", "causal_lm"), ("sft", "sft")):
        sel = select_algorithm(stage, {}, {}, _fake_caps())
        assert sel["algorithm"] == expected
        assert sel["runnable"] is True
        assert "sft" in sel["because"]


def test_select_posttrain_stages_reasoning_data() -> None:
    stages = select_posttrain_stages({}, {"has_instructions": True, "has_verifiable_answers": True}, _fake_caps())
    assert [s["stage"] for s in stages] == ["sft", "rl"]


def test_select_posttrain_stages_pairs_only_propose_preference() -> None:
    stages = select_posttrain_stages({}, {"has_pairs": True}, _fake_caps())
    names = [s["stage"] for s in stages]
    assert "preference" in names and "rl" not in names


def test_select_posttrain_stages_default_is_sft() -> None:
    stages = select_posttrain_stages({}, {}, _fake_caps())
    assert [s["stage"] for s in stages] == ["sft"]
    assert all(s["because"] for s in stages)
