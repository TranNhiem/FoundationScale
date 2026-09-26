
"""End-to-end reference scenario with the REAL knowledge base and estimators.

Scenario: a manufacturing-domain reasoning model from an open-source 7B base
(llama3 Llama-3.1-8B, else qwen2.5 7B), ~2B tokens of domain text + 20k QA
pairs with verifiable answers, DGX H100 2 nodes x 8 GPUs, preserve general
capabilities; plus a second variant on gemma4 E4B / GB200.

No network, no GPU, no skips: if the knowledge base lacks the referenced
families/hardware these asserts fail (absence of evidence is a failure, not a
skip).
"""
from __future__ import annotations

from foundationskills.core import load_schema, validate
from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.training.knowledge import load_families, load_hardware
from foundationskills.skills.training.planner import PLANNING_STEPS, plan


def _fake_caps() -> FSCapabilities:
    """Hand-built caps mirroring the MEASURED FoundationScale facts (77bfa65)."""
    refused = "refused: not wired to this loop / does not expose the declared axes"
    rl_runnable = {"dr_grpo": None, "gspo": None, "dapo": None}
    for name in ("grpo", "ppo", "rloo", "reinforce_baseline", "reinforce_pp", "dpo", "ipo", "kto", "orpo", "simpo", "cpo", "online_dpo", "iterative_dpo", "raft", "best_of_n"):
        rl_runnable[name] = refused
    return FSCapabilities(
        available=True,
        fs_version="0.0.0+77bfa65",
        train_flags=frozenset(
            {
                "--model", "--dataset", "--output-dir", "--profile-name", "--objective",
                "--dry-run", "--sharding-strategy", "--tp", "--pp", "--ep", "--cp",
                "--adapter", "--adapter-target", "--max-sequence-length",
            }
        ),
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=tuple(rl_runnable),
        rl_runnable=rl_runnable,
        families={"gemma4": ("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"), "qwen3.5": ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")},
        backends=("ddp", "fsdp"),
    )


def _variants(family: dict) -> list[dict]:
    raw = getattr(family, "raw", None) or {}
    return [v for v in raw.get("variants", []) if isinstance(v, dict)]


def _pick_variant_ref() -> tuple[str, dict, dict]:
    """(family name, variant raw dict, family object) for the scenario base."""
    families = load_families()
    for family_name, needles in (("llama3", ("llama-3.1-8b", "3.1-8b", "8b")), ("qwen2.5", ("7b",))):
        family = families.get(family_name)
        if family is None:
            continue
        for variant in _variants(family_name and family):
            hay = f"{variant.get('id', '')} {variant.get('hf_id', '') or ''}".lower()
            if any(needle in hay for needle in needles):
                return family_name, variant, family
    raise AssertionError("knowledge base lacks the llama3 Llama-3.1-8B (or qwen2.5 7B) variant required by the reference scenario")


def _variant_ref(variant: dict) -> str:
    ref = variant.get("hf_id") or variant.get("id") or variant.get("local_path")
    assert ref, "scenario variant needs an id/hf_id that find_variant can match"
    return str(ref)


def _manufacturing_goal(variant: dict, hardware_gpu: str, nodes: int, gpus_per_node: int) -> dict:
    return {
        "objective": "manufacturing-domain reasoning model from an open-source base",
        "target_capabilities": ["reasoning", "manufacturing QA"],
        "goal": "domain_expert",
        "domain": "manufacturing",
        "preserve_general": True,
        "base_model": {
            "name_or_path": _variant_ref(variant),
            "size_b": float(variant.get("size_b") or 8.0),
            "arch": str(variant.get("arch") or "dense"),
            "model_type": "llm",
        },
        "data": {
            "sources": [
                {"uri": "file:///data/mfg_corpus", "kind": "local_dir", "approx_tokens": 2_000_000_000, "domain": "manufacturing"},
                {"uri": "file:///data/mfg_qa.jsonl", "kind": "jsonl", "approx_examples": 20_000, "domain": "manufacturing"},
            ]
        },
        "hardware": {"gpu": hardware_gpu, "gpus_per_node": gpus_per_node, "nodes": nodes},
        "data_facts": {"has_instructions": True, "has_pairs": False, "has_verifiable_answers": True},  # QA pairs with gold answers are NOT chosen/rejected preference pairs
        "rl_algorithm": "grpo",
    }


def test_reference_scenario_manufacturing_reasoning_on_h100():
    hardware = load_hardware()
    assert any("h100" in key.lower() or "h100" in str(getattr(hw, "gpu_name", "")).lower() for key, hw in hardware.items()), "hardware knowledge base must contain an H100 entry"

    _, variant, _ = _pick_variant_ref()
    plan_payload = plan(
        _manufacturing_goal(variant, "NVIDIA H100 (DGX)", nodes=2, gpus_per_node=8),
        caps=_fake_caps(),
        hardware_id="h100",
    )

    stages = plan_payload["stages"]
    order = {stage: idx for idx, stage in enumerate(s["stage"] for s in stages)}
    for expected in ("cpt", "sft", "rl"):
        assert expected in order, f"plan must include a {expected} stage; got {[s['stage'] for s in stages]}"
    assert order["cpt"] < order["sft"] < order["rl"]

    cpt = next(s for s in stages if s["stage"] == "cpt")
    assert cpt["method"] == "full"
    assert 0.05 <= cpt["hparams"]["replay_ratio"] <= 0.30

    rl = next(s for s in stages if s["stage"] == "rl")
    assert rl["algorithm"] in {"dr_grpo", "gspo", "dapo"}
    assert "grpo" in rl["because"].lower()
    assert "reference-free" in rl["because"].lower()
    assert rl["estimate"]["gpus"] == 1

    assert all(s["because"].strip() for s in stages)
    assert plan_payload["feasibility"]["verdict"] in {"ok", "warn", "infeasible"}
    assert all((s["estimate"].get("gpu_hours") or 0) > 0 for s in stages)

    decision_steps = {d["step"] for d in plan_payload["decisions"]}
    assert set(PLANNING_STEPS) <= decision_steps, f"missing decision steps: {set(PLANNING_STEPS) - decision_steps}"

    assert validate(plan_payload, load_schema("artifacts/training_plan")) == []

    assert cpt["executable"] is True
    for stage_name in ("sft", "rl"):
        stage = next(s for s in stages if s["stage"] == stage_name)
        assert stage["data"].get("handoff") == "data_engine"
        assert stage["data"].get("target_format") == stage_name


def test_reference_scenario_gemma4_e4b_on_gb200():
    hardware = load_hardware()
    assert any("gb200" in key.lower() for key in hardware), "hardware knowledge base must contain a GB200 entry"

    families = load_families()
    gemma4 = families.get("gemma4")
    assert gemma4 is not None, "knowledge base must contain the gemma4 family"
    e4b = None
    for variant in _variants(gemma4):
        hay = f"{variant.get('id', '')} {variant.get('hf_id', '') or ''}".lower()
        if "e4b" in hay:
            e4b = variant
            break
    assert e4b is not None, "gemma4 family must contain the E4B variant"

    goal = _manufacturing_goal(e4b, "NVIDIA GB200", nodes=1, gpus_per_node=4)
    goal["objective"] = "general chat assistant"
    goal["goal"] = "general_chat"
    goal["data_facts"] = {"has_instructions": True, "has_pairs": False, "has_verifiable_answers": False}

    plan_payload = plan(goal, caps=_fake_caps(), hardware_id="gb200")
    assert validate(plan_payload, load_schema("artifacts/training_plan")) == []
    assert plan_payload["stages"], "general-chat goal must produce at least one stage"
    assert any(s["executable"] for s in plan_payload["stages"])
    assert plan_payload["feasibility"]["verdict"] in {"ok", "warn"}
    assert all(s["because"].strip() for s in plan_payload["stages"])
    # gemma4 IS FS-registered: no adapter-target-model-analysis decision
    assert not any("adapter-target" in d["because"] for d in plan_payload["decisions"] if d["step"] == "model_analysis")
