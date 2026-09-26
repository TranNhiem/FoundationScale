
"""Tests for the agent intake helpers."""
from __future__ import annotations

from foundationskills.agent import REQUIRED_FACTS, goal_from_answers, missing_facts
from foundationskills.core import load_schema, validate


def _answers() -> dict:
    return {
        "objective": "manufacturing-domain reasoning model",
        "capabilities": ["reasoning", "manufacturing QA"],
        "goal": "domain_expert",
        "domain": "manufacturing",
        "preserve_general": True,
        "base_model": {"name_or_path": "meta-llama/Llama-3.1-8B", "family": "llama3", "size_b": 8.0, "arch": "dense", "model_type": "llm"},
        "sources": [
            {"uri": "file:///data/mfg_corpus", "kind": "local_dir", "approx_tokens": 2_000_000_000, "domain": "manufacturing"},
            {"uri": "file:///data/mfg_qa.jsonl", "kind": "jsonl", "approx_examples": 20_000, "domain": "manufacturing"},
        ],
        "hardware": {"gpu": "NVIDIA H100", "gpus_per_node": 8, "nodes": 2},
        "data_facts": {"has_verifiable_answers": True, "has_pairs": True, "has_instructions": True},
        "rl_algorithm": "grpo",
    }


def test_goal_from_answers_validates_against_goal_spec_schema():
    payload = goal_from_answers(_answers())
    errors = validate(payload, load_schema("artifacts/goal_spec"))
    assert errors == []
    assert payload["goal"] == "domain_expert"
    assert payload["data_facts"]["has_verifiable_answers"] is True


def test_goal_from_answers_requires_load_bearing_answers():
    import pytest

    with pytest.raises(ValueError, match="objective"):
        goal_from_answers({})
    with pytest.raises(ValueError, match="base_model"):
        goal_from_answers({"objective": "x"})
    with pytest.raises(ValueError, match="source"):
        goal_from_answers({"objective": "x", "base_model": {"name_or_path": "m"}})
    with pytest.raises(ValueError, match="hardware"):
        goal_from_answers(
            {
                "objective": "x",
                "base_model": {"name_or_path": "m"},
                "sources": [{"uri": "u", "kind": "jsonl"}],
            }
        )


def test_missing_facts_on_empty_goal_are_load_bearing_weighted():
    missing = missing_facts({}, ["training.planner"])
    keys = {f["key"] for f in missing}
    assert {"objective", "base_model.name_or_path", "data.sources", "hardware.gpu", "hardware.nodes", "hardware.gpus_per_node"} <= keys
    assert all(f["load_bearing"] for f in missing if f["key"] in {"objective", "base_model.name_or_path"})


def test_missing_facts_none_for_complete_goal():
    payload = goal_from_answers(_answers())
    load_bearing_missing = [f for f in missing_facts(payload, ["training.planner"]) if f["load_bearing"]]
    assert load_bearing_missing == []


def test_missing_facts_detects_empty_values():
    goal = goal_from_answers(_answers())
    goal["objective"] = ""
    assert any(f["key"] == "objective" for f in missing_facts(goal, ["training.planner"]))


def test_required_facts_shape():
    for skill, facts in REQUIRED_FACTS.items():
        for fact in facts:
            assert set(fact) == {"key", "question", "why", "load_bearing"}
            assert isinstance(fact["load_bearing"], bool)
