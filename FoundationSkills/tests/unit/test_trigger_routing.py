"""Tests for MODULES A + B: trigger registry (A) -> stage-1 front gate (B)."""

import pytest

from foundationskills.agent.route_front_gate import front_gate, score
from foundationskills.agent.trigger_registry import (SCHEMA, compile_registry,
                                                     negatives, skill_entry,
                                                     tokens)

LOUD = ("Quantize and compress models: GPTQ, AWQ, INT8 and pruning. "
        "Do NOT use for serving a checkpoint (fskills-deployment) "
        "or LoRA merges (fskills-merging).")
INDEX = [
    {"name": "fskills-compression", "description": LOUD,
     "when_to_use": ["quantize this model", "quantize a 7b checkpoint for me"],
     "invocation": "model"},
    {"name": "fskills-deployment",
     "description": "Serve and deploy checkpoints with vLLM and Docker. "
                    "Do NOT use for LoRA merge jobs (fskills-merging).",
     "when_to_use": ["serve this quantized checkpoint", "deploy my model"],
     "invocation": "model"},
    {"name": "fs-data", "description": "Load and clean local datasets.",
     "when_to_use": ["load my csv"], "invocation": "user"},
]


def _reg():
    return compile_registry(INDEX)


def test_tokens_drop_stopwords_and_numbers():
    assert tokens("Use the 42 models for INT8") == ["model", "int8"]  # stemmed
    assert tokens("") == []


def test_negatives_extract_phrase_and_route_to():
    assert LOUD.endswith("Do NOT use for serving a checkpoint (fskills-deployment) "
                         "or LoRA merges (fskills-merging).")
    negs = {n["route_to"]: n["phrase"] for n in negatives(LOUD)}
    assert list(negs) == ["fskills-deployment", "fskills-merging"]
    assert negs["fskills-deployment"] == "serving a checkpoint"
    assert negs["fskills-merging"].lower() == "lora merges"
    assert negatives("Compress models before shipping.") == []


def test_skill_entry_keywords_and_aliases():
    entry = skill_entry(INDEX[0])
    assert entry["keywords"][0] == "quantiz"  # stemmed
    assert {"checkpoint", "model", "7b"} <= set(entry["keywords"])
    assert len(entry["keywords"]) <= 40
    assert entry["aliases"] == ["compression", "fskills-compression"]
    assert skill_entry(INDEX[2])["aliases"] == ["data", "fs-data"]


def test_compile_registry_deterministic_and_sorted():
    reg, twin = compile_registry(INDEX), compile_registry(INDEX)
    assert reg == twin and reg["schema"] == SCHEMA
    assert isinstance(reg["index_sha256"], str) and reg["index_sha256"]
    names = [s["name"] for s in reg["skills"]]
    assert names == sorted(names)
    assert names == ["fs-data", "fskills-compression", "fskills-deployment"]
    assert reg["skills"][1] == skill_entry(INDEX[0])


def test_front_gate_ranks_the_quantize_skill():
    out = front_gate("quantize this 7b model with gptq", _reg())
    names = [c["name"] for c in out["candidates"]]
    assert out["abstain"] is False and out["redirects"] == []
    assert names[:2] == ["fskills-compression", "fskills-deployment"]


def test_score_penalises_matched_negatives():
    comp = skill_entry(INDEX[0])
    assert score("please compress with fskills-compression", comp) >= 10.0
    assert score("quantize this model", comp) > 0.0
    penalised = score("serve this quantized checkpoint", comp)
    assert penalised <= 0.0 and penalised < score("quantize this model", comp)


def test_front_gate_demotes_compression_and_redirects():
    out = front_gate("serve this quantized checkpoint", _reg())
    names = [c["name"] for c in out["candidates"]]
    assert names[0] == "fskills-deployment"
    assert "fskills-compression" not in names  # it matched its own exclusion
    assert out["redirects"] == ["fskills-deployment"]


def test_front_gate_abstains_on_unrelated_request():
    out = front_gate("plan a hiking trip", _reg())
    assert out["abstain"] is True
    assert out["candidates"] == [] and out["redirects"] == []


def test_user_skills_hidden_unless_slash_or_flag():
    reg = _reg()
    hidden = front_gate("load my csv", reg)
    assert "fs-data" not in [c["name"] for c in hidden["candidates"]]
    for request in ("/data load my csv", "/fs-data load my csv"):
        assert front_gate(request, reg)["candidates"][0]["name"] == "fs-data"
    forced = front_gate("load my csv", reg, include_user_skills=True)
    assert "fs-data" in [c["name"] for c in forced["candidates"]]


def test_real_index_has_keywords_and_compression_negative():
    ev = pytest.importorskip("foundationskills.agent.routing_eval")
    registry = compile_registry(ev.load_index())
    model = [s for s in registry["skills"] if s["invocation"] == "model"]
    assert model, "real index has no model-invocable skills"
    for skill in model:
        assert skill["keywords"], skill["name"]
    routed = [n["route_to"] for s in registry["skills"] if "compression" in s["name"]
              for n in s["negatives"]]
    assert "fskills-deployment" in routed


def test_two_stage_arm_offers_only_front_gate_candidates():
    from foundationskills.agent import routing_eval as r

    idx = r.load_index()
    system = r.build_messages(idx, "two_stage", "Quantize gemma to int4 with RTN and check the perplexity ratio")[0]["content"]
    assert "fskills-compression" in system
    assert system.count("\n- ") <= r.TWO_STAGE_K
    off = r.build_messages(idx, "two_stage", "Plan a hiking itinerary in the Alps")[0]["content"]
    assert off.count("\n- ") <= r.TWO_STAGE_K
