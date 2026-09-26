
"""Tests for recommend_pipeline op sequencing."""
from __future__ import annotations

from foundationskills.core.schema import load_schema, validate
from foundationskills.skills.data_engine.recommend import recommend_pipeline

SOURCES_ONE = [{"uri": "/data/corpus.jsonl", "kind": "jsonl"}]
SOURCES_TWO = [
    {"uri": "/data/corpus.jsonl", "kind": "jsonl"},
    {"uri": "/data/docs", "kind": "documents"},
]


def _spec(target, sources=SOURCES_ONE, benchmarks=None, fmt_kwargs=None):
    kwargs = dict(fmt_kwargs or {})
    return recommend_pipeline(
        target_format=target,
        sources=sources,
        goal=kwargs.pop("goal", "domain_expert"),
        algorithm=None,
        tokenizer="tok-1",
        chat_template_family=kwargs.pop("chat_template_family", None),
        domain="manufacturing",
        benchmarks=list(benchmarks or []),
    )


def _names(spec):
    return [step["op"] for step in spec["ops"]]


class TestSequences:
    def test_schema_valid_for_every_format(self):
        for target in ("pretrain", "cpt", "sft", "preference", "rl", "mm_sft"):
            spec = _spec(target)
            errors = validate(spec, load_schema("artifacts/data_pipeline_spec"))
            assert errors == [], errors

    def test_pretrain_sequence(self):
        names = _names(_spec("pretrain"))
        assert names == ["ingest", "clean", "dedup", "quality", "format", "tokenize"]

    def test_cpt_with_benchmarks_and_multi_source_has_decontam_and_mix(self):
        spec = _spec("cpt", sources=SOURCES_TWO, benchmarks=["gsm8k"])
        names = _names(spec)
        assert names.index("decontam") < names.index("mix") < names.index("format")
        mix_cfg = next(s["config"] for s in spec["ops"] if s["op"] == "mix")
        assert mix_cfg["components"][0]["inline"] is True

    def test_cpt_without_benchmarks_skips_decontam(self):
        spec = _spec("cpt")
        assert "decontam" not in _names(spec)
        assert any("decontam" in line for line in spec["rationale"])

    def test_sft_sequence_pack_false(self):
        spec = _spec("sft", fmt_kwargs={"goal": "general_chat", "chat_template_family": "gemma4"})
        names = _names(spec)
        assert names == ["ingest", "clean", "dedup", "quality", "format", "tokenize"]
        tok_cfg = next(s["config"] for s in spec["ops"] if s["op"] == "tokenize")
        assert tok_cfg["pack"] is False
        fmt_cfg = next(s["config"] for s in spec["ops"] if s["op"] == "format")
        assert fmt_cfg["target_format"] == "sft"
        assert fmt_cfg["chat_template_family"] == "gemma4"
        assert spec["chat_template_family"] == "gemma4"

    def test_preference_and_rl(self):
        for target in ("preference", "rl"):
            names = _names(_spec(target))
            assert names[0] == "ingest" and names[-2:] == ["format", "tokenize"]
        rl_spec = _spec("rl")
        fmt_cfg = next(s["config"] for s in rl_spec["ops"] if s["op"] == "format")
        assert fmt_cfg.get("gold_key") == "answer"

    def test_every_op_choice_has_a_rationale_line(self):
        spec = _spec("cpt", sources=SOURCES_TWO, benchmarks=["gsm8k"])
        for step in spec["ops"]:
            assert any(line.startswith(f"{step['op']}:") for line in spec["rationale"])

    def test_unknown_format_raises(self):
        import pytest

        with pytest.raises(ValueError, match="unknown target_format"):
            recommend_pipeline(
                target_format="video_sft",
                sources=SOURCES_ONE,
                goal=None,
                algorithm=None,
                tokenizer=None,
                chat_template_family=None,
                domain=None,
                benchmarks=[],
            )
