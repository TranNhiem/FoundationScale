
"""Tests for the ``decontam`` op."""
from __future__ import annotations

import json
import sys

from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops import decontam as decontam_mod


def run_decontam(records, cfg):
    stats = OpStats(name="decontam")
    out = list(OPS["decontam"](iter(records), cfg, stats))
    return out, stats


BENCH_SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank today"
CJK_BENCH = "经营者集中的情形包括经营者合并以及通过合同取得控制权的方式"


def bench_file(tmp_path, name="bench.txt", content=None):
    p = tmp_path / name
    p.write_text(content if content is not None else BENCH_SENTENCE + "\n", encoding="utf-8")
    return p


def test_decontam_hit_drops_record(tmp_path):
    p = bench_file(tmp_path)
    rec = {"id": "1", "text": f"some lead in words and then {BENCH_SENTENCE} and a tail here"}
    out, stats = run_decontam(records=[rec], cfg={"benchmarks": ["gsm8k"], "sources": {"gsm8k": str(p)}})
    assert out == []
    assert stats.dropped["decontam"] == 1
    assert stats.extra["hits_by_benchmark"] == {"gsm8k": 1}
    assert stats.extra["decontam_hits"] == 1
    assert stats.extra["benchmarks_checked"] == ["gsm8k"]
    assert stats.extra["unmeasured_benchmarks"] == []


def test_decontam_miss_keeps_record(tmp_path):
    p = bench_file(tmp_path)
    rec = {"id": "1", "text": "a completely unrelated paragraph about gardening tools and soil types today"}
    out, stats = run_decontam([rec], {"benchmarks": ["gsm8k"], "sources": {"gsm8k": str(p)}})
    assert len(out) == 1
    assert stats.extra["decontam_hits"] == 0
    assert stats.extra["hits_by_benchmark"] == {"gsm8k": 0}


def test_decontam_jsonl_source(tmp_path):
    p = tmp_path / "bench.jsonl"
    p.write_text(json.dumps({"question": BENCH_SENTENCE}) + "\n", encoding="utf-8")
    rec = {"id": "1", "text": BENCH_SENTENCE + " with a tail"}
    out, stats = run_decontam([rec], {"benchmarks": ["ifeval"], "sources": {"ifeval": str(p)}})
    assert out == []
    assert stats.extra["hits_by_benchmark"]["ifeval"] == 1


def test_decontam_flag_action_keeps_with_meta(tmp_path):
    p = bench_file(tmp_path)
    rec = {"id": "1", "text": f"copy: {BENCH_SENTENCE}"}
    out, stats = run_decontam(
        [rec], {"benchmarks": ["gsm8k"], "sources": {"gsm8k": str(p)}, "action": "flag"}
    )
    assert len(out) == 1
    assert out[0]["meta"]["decontam"]["hits"] == ["gsm8k"]
    assert stats.modified["decontam_flagged"] == 1
    assert stats.extra["decontam_hits"] == 1


def test_decontam_cjk_char_ngrams(tmp_path):
    p = bench_file(tmp_path, name="bench_zh.txt", content=CJK_BENCH + "\n")
    rec = {"id": "1", "text": f"前面是引子。{CJK_BENCH}后面还有补充说明文字。"}
    out, stats = run_decontam([rec], {"benchmarks": ["ifeval"], "sources": {"ifeval": str(p)}})
    assert out == []
    assert stats.extra["hits_by_benchmark"]["ifeval"] == 1


def test_missing_benchmark_is_unmeasured_never_pass(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    rec = {"id": "1", "text": "perfectly fine original text about mathematics and shapes"}
    out, stats = run_decontam([rec], {"benchmarks": ["gsm8k"], "allow_download": False})
    assert len(out) == 1  # not dropped, but NOT credited either
    assert stats.extra["unmeasured_benchmarks"] == ["gsm8k"]
    assert stats.extra["benchmarks_checked"] == []
    assert stats.extra["hits_by_benchmark"] == {}
    assert "allow_download" in stats.extra["unmeasured_reasons"]["gsm8k"]


def test_unknown_benchmark_is_unmeasured(tmp_path):
    rec = {"id": "1", "text": "some original text that mentions many different common words and phrases here"}
    out, stats = run_decontam([rec], {"benchmarks": ["no_such_bench"]})
    assert len(out) == 1
    assert stats.extra["unmeasured_benchmarks"] == ["no_such_bench"]
    assert "unknown benchmark" in stats.extra["unmeasured_reasons"]["no_such_bench"]


def test_missing_source_file_is_unmeasured(tmp_path):
    rec = {"id": "1", "text": "original prose with enough distinct words to avoid any false matches"}
    out, stats = run_decontam(
        [rec], {"benchmarks": ["math"], "sources": {"math": str(tmp_path / "absent.txt")}}
    )
    assert len(out) == 1
    assert stats.extra["unmeasured_benchmarks"] == ["math"]
