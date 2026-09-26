
"""Planner unit tests with the D1-owned modules monkeypatched to simple fakes.

These isolate planner logic (rule evaluation, decision logging, executability,
schema validity). The end-to-end path with the real knowledge base lives in
tests/scenario/test_reference_scenario.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from foundationskills.core import load_schema, validate
from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.training import planner
from foundationskills.skills.training.planner import PLANNING_STEPS, PlanningRefusal


def _fake_caps(**over) -> FSCapabilities:
    base = dict(
        available=True,
        fs_version="0.0.0+77bfa65",
        train_flags=frozenset({"--model", "--dataset", "--output-dir", "--dry-run", "--adapter-target"}),
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=("dr_grpo", "gspo", "dapo", "grpo"),
        rl_runnable={"dr_grpo": None, "gspo": None, "dapo": None, "grpo": "refused: needs a reference model"},
        families={"gemma4": ("gemma4",), "qwen3.5": ("qwen3_5",)},
        backends=("ddp", "fsdp"),
    )
    base.update(over)
    return FSCapabilities(**base)


def _goal(**over) -> dict:
    goal = {
        "objective": "manufacturing-domain reasoning model",
        "target_capabilities": ["reasoning"],
        "goal": "domain_expert",
        "domain": "manufacturing",
        "preserve_general": True,
        "base_model": {"name_or_path": "Llama-3.1-8B", "family": "llama3", "size_b": 8.0, "arch": "dense"},
        "data": {
            "sources": [
                {"uri": "file:///data/corpus", "kind": "local_dir", "approx_tokens": 2_000_000_000},
                {"uri": "file:///data/qa.jsonl", "kind": "jsonl", "approx_examples": 20_000},
            ]
        },
        "hardware": {"gpu": "NVIDIA H100", "gpus_per_node": 8, "nodes": 2},
        "data_facts": {"has_instructions": True, "has_pairs": True, "has_verifiable_answers": True},
    }
    goal.update(over)
    return goal


def _rules() -> list[dict]:
    return [
        {"id": "SR-CPT", "when": {"domain_tokens_gte": 1}, "action": "add_stage", "stage": "cpt", "algorithm_hint": "causal_lm", "because": "domain corpus present"},
        {"id": "SR-SFT", "when": {"has_instructions": True}, "action": "add_stage", "stage": "sft", "algorithm_hint": "sft", "because": "instruction data present"},
        {"id": "SR-RL", "when": {"has_verifiable_answers": True}, "action": "add_stage", "stage": "rl", "algorithm_hint": "grpo", "because": "verifiable answers"},
        {"id": "SR-SKIP-PREF", "when": {"has_verifiable_answers": True}, "action": "skip_stage", "stage": "preference", "algorithm_hint": None, "because": "RL supersedes pairs"},
    ]


@pytest.fixture()
def patched(monkeypatch):
    family = SimpleNamespace(name="llama3", fs_family=None, raw={"name": "llama3"})
    variant = SimpleNamespace(
        id="Llama-3.1-8B", hf_id="meta-llama/Llama-3.1-8B", local_path=None,
        size_b=8.0, active_b=8.0, arch="dense", instruct=False,
        hidden=4096, layers=32, vocab=128256,
    )
    hw = SimpleNamespace(id="h100", gpu_name="NVIDIA H100", mem_gb=80.0, bf16_dense_tflops=989.0, gpus_per_node=8)
    recipe = SimpleNamespace(
        id="rcp-llama3-cpt",
        raw={
            "index": {"method": "full"},
            "stages": [{"name": "cpt", "stage": "cpt", "hparams": {"max_sequence_length": 4096}}],
            "provenance": {"status": "literature", "evidence": ["paper"]},
        },
    )
    match = SimpleNamespace(recipe=recipe, score=0.9, reasons=["stage+arch exact"], derived=True, adjustments=["size extrapolated"])

    monkeypatch.setattr(planner, "find_variant", lambda model: (family, variant))
    monkeypatch.setattr(planner, "load_hardware", lambda: {"h100": hw, "gb200": hw})
    monkeypatch.setattr(planner, "load_stage_rules", _rules)
    monkeypatch.setattr(planner, "select_recipe", lambda q: match)
    monkeypatch.setattr(planner, "rank_recipes", lambda q, k=3: [match])
    monkeypatch.setattr(
        planner,
        "select_method",
        lambda **kw: SimpleNamespace(
            method=("lora" if kw["stage"] in ("sft", "preference") else "full"),
            because=["fixture method rule"],
            alternatives=[],
        ),
    )
    monkeypatch.setattr(
        planner,
        "estimate_memory",
        lambda variant, **kw: SimpleNamespace(
            weights_gb=16.0, grads_gb=16.0, optimizer_gb=64.0, activations_gb=8.0,
            overhead_gb=4.0, total_per_gpu_gb=60.0, assumptions=["fixture"],
        ),
    )
    monkeypatch.setattr(
        planner,
        "estimate_time",
        lambda variant, **kw: SimpleNamespace(
            total_flops=6e19, tokens_per_s_per_gpu=50_000.0, hours=10.0,
            gpu_hours=float(kw["gpus"]) * 10.0, mfu=0.318, mfu_provenance="measured",
            assumptions=["fixture 6ND"],
        ),
    )
    monkeypatch.setattr(
        planner,
        "check_feasibility",
        lambda variant, hw_, **kw: SimpleNamespace(verdict="ok", findings=[], alternatives=[]),
    )
    return SimpleNamespace(family=family, variant=variant, hw=hw, recipe=recipe)


def test_plan_end_to_end_with_fakes(patched) -> None:
    payload = planner.plan(_goal(), caps=_fake_caps())

    stages = payload["stages"]
    assert [s["stage"] for s in stages] == ["cpt", "sft", "rl"]

    cpt = stages[0]
    assert cpt["method"] == "full"
    assert 0.05 <= cpt["hparams"]["replay_ratio"] <= 0.30
    assert cpt["hparams"]["learning_rate"] == 2e-5
    assert cpt["executable"] is True and cpt["missing"] is None

    rl = stages[2]
    assert rl["algorithm"] == "dr_grpo"
    assert "reference-free" in rl["because"]
    assert rl["estimate"]["gpus"] == 1  # FS RLTrainer is single-device

    assert all(s["because"] for s in stages)
    assert all(s["estimate"]["gpu_hours"] > 0 for s in stages)
    assert payload["feasibility"]["verdict"] == "ok"

    steps = {d["step"] for d in payload["decisions"]}
    assert set(PLANNING_STEPS) <= steps

    # llama3 is not FS-registered: the adapter-target decision must be present
    assert any("adapter-target" in d["because"] for d in payload["decisions"] if d["step"] == "model_analysis")

    errors = validate(payload, load_schema("artifacts/training_plan"))
    assert errors == []


def test_data_handoff_present_when_no_readiness(patched) -> None:
    payload = planner.plan(_goal(), caps=_fake_caps())
    for stage in payload["stages"]:
        assert stage["data"]["handoff"] == "data_engine"
        assert stage["data"]["target_format"] == stage["data"]["format"]


def test_readiness_pass_marks_matching_data_ready(patched) -> None:
    readiness = {"verdict": "PASS", "format": "cpt", "stats": {"num_tokens": 2_000_000_000}}
    payload = planner.plan(_goal(), caps=_fake_caps(), readiness=readiness)
    cpt = next(s for s in payload["stages"] if s["stage"] == "cpt")
    assert cpt["data"]["ready"] is True
    assert "handoff" not in cpt["data"]


def test_skip_rules_override_adds(patched, monkeypatch) -> None:
    monkeypatch.setattr(
        planner,
        "load_stage_rules",
        lambda: [
            {"id": "A", "when": {}, "action": "add_stage", "stage": "sft", "algorithm_hint": None, "because": "always"},
            {"id": "B", "when": {}, "action": "add_stage", "stage": "rl", "algorithm_hint": None, "because": "always"},
            {"id": "C", "when": {}, "action": "skip_stage", "stage": "rl", "algorithm_hint": None, "because": "not today"},
        ],
    )
    payload = planner.plan(_goal(), caps=_fake_caps())
    assert [s["stage"] for s in payload["stages"]] == ["sft"]


def test_no_stage_selected_refuses_with_details(patched, monkeypatch) -> None:
    monkeypatch.setattr(
        planner,
        "load_stage_rules",
        lambda: [{"id": "X", "when": {"has_pairs": True}, "action": "add_stage", "stage": "preference", "algorithm_hint": None, "because": "pairs"}],
    )
    goal = _goal(data_facts={"has_instructions": False, "has_pairs": False, "has_verifiable_answers": False})
    with pytest.raises(PlanningRefusal, match="no stage selected"):
        planner.plan(goal, caps=_fake_caps())


def test_unknown_model_without_size_refuses(patched, monkeypatch) -> None:
    monkeypatch.setattr(planner, "find_variant", lambda model: None)
    goal = _goal()
    goal["base_model"] = {"name_or_path": "mystery/model"}
    with pytest.raises(PlanningRefusal, match="base_model.size_b"):
        planner.plan(goal, caps=_fake_caps())


def test_unknown_model_with_size_synthesizes_variant(patched, monkeypatch) -> None:
    monkeypatch.setattr(planner, "find_variant", lambda model: None)
    monkeypatch.setattr(planner, "load_hardware", lambda: {"h100": patched.hw})
    payload = planner.plan(_goal(), caps=_fake_caps())
    assert payload["model"]["in_knowledge_base"] is False
    assert any("not in the knowledge base" in n for n in payload["provenance_notes"])


def test_unknown_hardware_refuses_naming_known_ids(patched, monkeypatch) -> None:
    monkeypatch.setattr(planner, "load_hardware", lambda: {"h100": patched.hw})
    goal = _goal(hardware={"gpu": "TPU v5p", "gpus_per_node": 8, "nodes": 1})
    with pytest.raises(PlanningRefusal, match="h100"):
        planner.plan(goal, caps=_fake_caps())


def test_missing_base_model_refuses(patched) -> None:
    with pytest.raises(PlanningRefusal, match="base_model.name_or_path"):
        planner.plan(_goal(base_model={}), caps=_fake_caps())


def test_fs_refusal_marks_stage_non_executable(patched) -> None:
    caps = _fake_caps(available=False, train_objectives=(), backends=())
    payload = planner.plan(_goal(), caps=caps)
    assert all(s["executable"] is False for s in payload["stages"])
    assert all(s["missing"] for s in payload["stages"])
