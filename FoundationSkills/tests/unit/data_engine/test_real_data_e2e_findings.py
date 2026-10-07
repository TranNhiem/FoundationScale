"""Regressions from the 2026-10-07 real-workload run (SFT-Taiwan-AIEC formatted-gemma4-v3, E4B tokenizer).

CPU-only: no tokenizer download, no network.
"""
from __future__ import annotations

import sys
import types

import pytest

from foundationskills.skills.data_engine.ops import decontam
from foundationskills.skills.data_engine.ops.format import lift_inline_thought
from foundationskills.skills.data_engine.recommend import recommend_pipeline


def test_inline_thought_is_lifted_into_reasoning_content():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "<|channel>thought\nstep 1\nstep 2\n<channel|>answer"}]
    out, status = lift_inline_thought(msgs)
    assert status == "lifted"
    assert out[1] == {"role": "assistant", "content": "answer", "reasoning_content": "step 1\nstep 2"}
    assert msgs[1]["content"].startswith("<|channel>thought")  # input never mutated


def test_empty_inline_thought_is_stripped_not_lifted():
    out, status = lift_inline_thought([{"role": "user", "content": "q"},
                                       {"role": "assistant", "content": "<|channel>thought\n<channel|>answer"}])
    assert status == "empty_stripped"
    assert out[1] == {"role": "assistant", "content": "answer"}


def test_plain_and_user_turns_are_untouched():
    msgs = [{"role": "user", "content": "<|channel>thought\nx<channel|>q"}, {"role": "assistant", "content": "a"}]
    out, status = lift_inline_thought(msgs)
    assert status == "none" and out == msgs


def test_existing_reasoning_content_wins():
    msgs = [{"role": "assistant", "content": "<|channel>thought\nx<channel|>a", "reasoning_content": "kept"}]
    out, _ = lift_inline_thought(msgs)
    assert out == msgs


@pytest.mark.parametrize("name,configs", [("arc_easy", ("ARC-Easy",)), ("arc_challenge", ("ARC-Challenge",)),
                                          ("arc", ("ARC-Easy", "ARC-Challenge")), ("gsm8k", ("main",))])
def test_registry_names_carry_their_configs(name, configs):
    assert decontam._BENCHMARK_HF[name][1] == configs


def test_hf_loader_passes_config_and_needs_every_config(monkeypatch):
    calls: list[tuple] = []

    def load_dataset(hf_id, config=None, split=None):
        calls.append((hf_id, config, split))
        if config == "ARC-Challenge":
            raise ValueError("not cached")
        return [{"question": f"{config} q"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    assert decontam._load_hf_benchmark("arc_easy", "allenai/ai2_arc", ("ARC-Easy",)) == ["ARC-Easy q"]
    assert calls[0] == ("allenai/ai2_arc", "ARC-Easy", "test")
    with pytest.raises(decontam.IngestErrorLike):  # a partial benchmark would be a vacuous pass
        decontam._load_hf_benchmark("arc", "allenai/ai2_arc", ("ARC-Easy", "ARC-Challenge"))


def _decontam_cfg(**kwargs):
    spec = recommend_pipeline(target_format="sft", sources=[{"uri": "/d.jsonl", "kind": "jsonl"}], goal=None,
                              algorithm=None, tokenizer=None, chat_template_family="gemma4", domain=None,
                              benchmarks=["arc_easy"], **kwargs)
    return next(op for op in spec["ops"] if op["op"] == "decontam")["config"]


def test_recommended_decontam_can_reach_a_benchmark():
    assert _decontam_cfg() == {"benchmarks": ["arc_easy"]}  # default stays closed (UNMEASURED)
    assert _decontam_cfg(allow_benchmark_download=True)["allow_download"] is True
    assert _decontam_cfg(benchmark_sources={"arc_easy": "/b.jsonl"})["sources"] == {"arc_easy": "/b.jsonl"}


def test_hf_ingest_honours_config_and_names_load_failures(monkeypatch):
    from foundationskills.skills.data_engine.ops import ingest

    calls: list[tuple] = []

    def load_dataset(uri, name=None, split=None, streaming=False):
        calls.append((uri, name, split))
        if name is None:
            raise ValueError("There are multiple configurations")
        return [{"question": "q"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    assert list(ingest._read_hf_dataset("allenai/ai2_arc", {"config": "ARC-Challenge"})) == [{"question": "q"}]
    assert calls[-1] == ("allenai/ai2_arc", "ARC-Challenge", "train")
    with pytest.raises(ingest.IngestError, match="did not load"):
        list(ingest._read_hf_dataset("allenai/ai2_arc", {}))
