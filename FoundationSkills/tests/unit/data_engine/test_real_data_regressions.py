"""Regressions found by the first real-data validation run (2026-09-26).

Each test pins one defect that 357 synthetic unit tests did not catch:
annotated HF column names, question/response datasets, FS truncating long
corpus records, and double-counted op inputs."""
from __future__ import annotations

import json

from foundationskills.skills.data_engine.ops.base import OpStats
from foundationskills.skills.data_engine.ops.ingest import _map_record
from foundationskills.skills.data_engine.ops.tokenize import _chunk_text, _tokenize_records
from foundationskills.skills.data_engine.pipeline import run_pipeline


def test_annotated_column_names_map_to_messages() -> None:
    # FreedomIntelligence/medical-r1-distill-data headers, verbatim
    raw = {"question": "Q?", "reasoning (reasoning_content)": "think", "response (content)": "A."}
    rec = _map_record(raw, source_uri="hf", index=0, options={})
    assert rec["messages"] == [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]
    with_trace = _map_record(raw, source_uri="hf", index=0, options={"include_reasoning": True})
    # the trace travels as reasoning_content; format renders + verifies it per family
    assert with_trace["messages"][1] == {"role": "assistant", "content": "A.", "reasoning_content": "think"}


def test_problem_answer_columns_map_for_rl() -> None:
    # HuggingFaceH4/MATH-500 headers
    raw = {"problem": "1+1?", "solution": "It is 2.", "answer": "2", "subject": "x"}
    rec = _map_record(raw, source_uri="hf", index=0, options={})
    assert rec["prompt"] == "1+1?" and rec["answer"] == "2"


def test_chunking_keeps_every_paragraph_under_budget() -> None:
    stats = OpStats("tokenize")
    text = "\n\n".join(f"para {i} " + "word " * 50 for i in range(40))  # ~55 approx tokens per para
    chunks = _chunk_text(None, text, 200, stats)
    assert len(chunks) > 1
    assert all(len(c) // 4 <= 200 for c in chunks)
    assert "".join(chunks).replace("\n\n", "") == text.replace("\n\n", "")  # nothing lost


def test_tokenize_chunk_zeroes_truncation() -> None:
    stats = OpStats("tokenize")
    recs = [{"id": "d", "text": "\n\n".join("w " * 300 for _ in range(10)), "meta": {}}]
    out = list(_tokenize_records(recs, {"seq_len": 256, "chunk": True}, stats))
    assert len(out) > 1 and stats.extra["truncation_rate"] == 0.0
    assert stats.modified["chunked_records"] == 1 and out[0]["meta"]["chunk_of"] == "d"
    stats2 = OpStats("tokenize")  # MUST_FIRE control: without chunking the same record truncates
    list(_tokenize_records(recs, {"seq_len": 256, "chunk": False}, stats2))
    assert stats2.extra["truncation_rate"] == 1.0


def test_op_inputs_counted_once(tmp_path) -> None:
    src = tmp_path / "c.jsonl"
    src.write_text("".join(json.dumps({"text": f"document number {i} " * 20}) + "\n" for i in range(5)))
    spec = {"target_format": "cpt", "seed": 0, "tokenizer": None, "rationale": [],
            "ops": [{"op": "ingest", "config": {"sources": [{"uri": str(src), "kind": "jsonl"}]}},
                    {"op": "dedup", "config": {}},
                    {"op": "format", "config": {"target_format": "cpt"}}]}
    result = run_pipeline(spec, tmp_path / "out")
    by = {s["name"]: s for s in result.stats}
    assert by["dedup"]["records_in"] == 5 and by["format"]["records_in"] == by["dedup"]["records_out"]


def test_zero_records_finding_names_source_columns(tmp_path) -> None:
    from foundationskills.core.contract import SkillContext
    from foundationskills.skills.data_engine.skill import DataEngineSkill

    src = tmp_path / "odd.jsonl"
    src.write_text(json.dumps({"weird_col": "x", "other": "y"}) + "\n")
    res = DataEngineSkill().execute(
        {"sources": [{"uri": str(src), "kind": "jsonl"}], "target_format": "cpt"}, SkillContext(workdir=tmp_path))
    f = next(f for f in res.findings if f.rule_id == "DE-HO-003")
    assert "weird_col" in f.message and "field_map" in f.recovery
