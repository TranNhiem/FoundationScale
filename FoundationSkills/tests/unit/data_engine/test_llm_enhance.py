"""Unit tests for the ``llm_enhance`` op: scripted fake backend, no network."""
from __future__ import annotations

import json

import pytest
from foundationskills.skills.data_engine.llm_backend import (
    LLMResponse,
    register_backend,
)
from foundationskills.skills.data_engine.ops import llm_enhance as _llm_enhance  # noqa: F401
from foundationskills.skills.data_engine.ops.base import OPS, OpStats

TEXT = "Photosynthesis converts sunlight into chemical energy inside green plant leaves."
REWRITE = (
    "Photosynthesis converts sunlight into chemical energy inside "
    "green plant leaves, explained clearly."
)
UNRELATED = "Zephyr waffles juxtapose quirky vexed nymphs."


class FakeBackend:
    """Deterministic backend: first rule whose needle is in the last user msg wins."""

    kind = "fake"
    model = "fake-1"

    def __init__(self, rules=(), cache_hit=False):
        self.rules = list(rules)
        self.cache_hit = cache_hit
        self.calls = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, seed=0,
                 json_mode=False):
        self.calls.append(
            {"messages": messages, "temperature": temperature, "json_mode": json_mode}
        )
        last = messages[-1]["content"]
        for needle, payload in self.rules:
            if needle in last:
                if isinstance(payload, LLMResponse):
                    return payload
                return LLMResponse(
                    content=str(payload), finish_reason="stop",
                    prompt_tokens=7, completion_tokens=11,
                    cache_hit=self.cache_hit, error=None,
                )
        return LLMResponse(
            content=UNRELATED, finish_reason="stop",
            prompt_tokens=7, completion_tokens=11,
            cache_hit=self.cache_hit, error=None,
        )


def _error_response(msg="backend exploded"):
    return LLMResponse(
        content=None, finish_reason=None, prompt_tokens=2,
        completion_tokens=0, cache_hit=False, error=msg,
    )


def _setup(rules=(), cache_hit=False, **cfg_over):
    backend = FakeBackend(rules, cache_hit=cache_hit)
    register_backend("fake", lambda cfg, b=backend: b)
    cfg = {"backend": {"kind": "fake"}}
    cfg.update(cfg_over)
    return backend, cfg


def _run(cfg, records):
    stats = OpStats(name="llm_enhance")
    out = list(OPS["llm_enhance"](records, cfg, stats))
    return out, stats


def test_op_registered_with_strict_schema():
    op = OPS["llm_enhance"]
    assert op.config_schema["additionalProperties"] is False
    assert set(op.config_schema["required"]) == {"backend", "mode"}


def test_rephrase_append_yields_original_then_synthetic():
    _, cfg = _setup([("Photosynthesis", REWRITE)], mode="rephrase")
    recs = [{"id": "d1", "text": TEXT, "meta": {"domain": "science"}}]
    out, stats = _run(cfg, recs)
    assert [r["id"] for r in out] == ["d1", "d1#rephrase-wikipedia"]
    assert out[0]["text"] == TEXT
    prov0 = out[0]["meta"]["llm"]
    assert len(prov0) == 1 and prov0[0]["op"] == "llm_enhance"
    synth = out[1]
    assert synth["text"] == REWRITE
    assert synth["meta"]["synthetic"] is True
    assert synth["meta"]["source_id"] == "d1"
    assert 0.5 < synth["meta"]["overlap"] <= 1.0
    prov = synth["meta"]["llm"][0]
    assert prov["mode"] == "rephrase" and prov["model"] == "fake-1"
    assert prov["backend"] == "fake" and prov["cache_hit"] is False
    assert len(prov["prompt_hash"]) == 16
    assert stats.records_in == 1 and stats.records_out == 2
    assert stats.extra["calls"] == 1
    assert stats.extra["prompt_tokens"] == 7
    assert stats.extra["completion_tokens"] == 11
    assert stats.extra["overlap_mean"] > 0.5
    assert stats.modified["llm_provenance"] == 2


def test_rephrase_replace_yields_only_synthetic():
    _, cfg = _setup(
        [("Photosynthesis", REWRITE)], mode="rephrase", emit="replace", style="textbook"
    )
    out, _ = _run(cfg, [{"id": "d1", "text": TEXT}])
    assert len(out) == 1
    assert out[0]["id"] == "d1#rephrase-textbook"
    assert out[0]["meta"]["source_id"] == "d1"


def test_rephrase_ungrounded_dropped_original_kept():
    _, cfg = _setup([], mode="rephrase")  # default payload is unrelated
    out, stats = _run(cfg, [{"id": "d1", "text": TEXT}])
    assert [r["id"] for r in out] == ["d1"]
    assert stats.dropped["ungrounded"] == 1
    assert stats.extra["overlap_mean"] == 0.0


def test_rephrase_error_replace_falls_back_to_original():
    _, cfg = _setup(
        [("Photosynthesis", _error_response())], mode="rephrase", emit="replace"
    )
    out, stats = _run(cfg, [{"id": "d1", "text": TEXT}])
    assert [r["id"] for r in out] == ["d1"]
    assert stats.dropped["llm_error"] == 1
    assert stats.extra["errors"] == 1
    assert stats.extra["fallback_original"] == 1


def test_rephrase_missing_text_drops_no_text():
    backend, cfg = _setup([], mode="rephrase")
    out, stats = _run(cfg, [{"id": "d1"}])
    assert out == []
    assert stats.dropped["no_text"] == 1
    assert len(backend.calls) == 0


def test_rephrase_temperature_defaults_and_override():
    backend, cfg = _setup([("Photosynthesis", REWRITE)], mode="rephrase")
    _run(cfg, [{"id": "d1", "text": TEXT}])
    assert backend.calls[0]["temperature"] == 0.7
    backend, cfg = _setup(
        [("Photosynthesis", REWRITE)], mode="rephrase", temperature=0.3
    )
    _run(cfg, [{"id": "d1", "text": TEXT}])
    assert backend.calls[0]["temperature"] == 0.3
    backend, cfg = _setup([("hello", '{"score": 5}')], mode="judge")
    _run(cfg, [{"id": "d1", "text": "hello world"}])
    assert backend.calls[0]["temperature"] == 0.0


def test_qa_synth_emits_grounded_pairs_and_copies_meta():
    payload = json.dumps(
        {
            "pairs": [
                {
                    "question": "What do green plants convert?",
                    "answer": "They convert sunlight into chemical energy "
                    "inside green plant leaves.",
                },
                {"question": "Where does it happen?", "answer": "Inside green plant leaves."},
            ]
        }
    )
    backend, cfg = _setup([("Photosynthesis", payload)], mode="qa_synth", n_pairs=2)
    recs = [{"id": "doc1", "text": TEXT, "meta": {"domain": "sci", "license": "cc-by"}}]
    out, stats = _run(cfg, recs)
    assert [r["id"] for r in out] == ["doc1#qa1", "doc1#qa2"]
    msgs = out[0]["messages"]
    assert msgs[0]["role"] == "user" and "convert" in msgs[0]["content"]
    assert msgs[1]["role"] == "assistant"
    assert out[0]["meta"]["synthetic"] is True
    assert out[0]["meta"]["source_id"] == "doc1"
    assert out[0]["meta"]["domain"] == "sci"
    assert out[0]["meta"]["license"] == "cc-by"
    assert out[0]["meta"]["overlap"] >= 0.5
    assert stats.extra["pairs_generated"] == 2
    assert stats.extra["pairs_kept"] == 2
    assert backend.calls[0]["json_mode"] is True
    assert all(r["id"] != "doc1" for r in out)  # the source doc is not re-emitted


def test_qa_synth_invalid_json_drops_record():
    _, cfg = _setup([("Photosynthesis", "sure, here you go!")], mode="qa_synth")
    out, stats = _run(cfg, [{"id": "doc1", "text": TEXT}])
    assert out == []
    assert stats.dropped["llm_invalid_json"] == 1


def test_qa_synth_bad_pairs_counted_individually():
    payload = json.dumps(
        {
            "pairs": [
                {"question": "Grounded?", "answer": "Sunlight inside green plant leaves."},
                {"question": "Ungrounded?", "answer": UNRELATED},
                {"question": "No answer", "answer": ""},
            ]
        }
    )
    _, cfg = _setup([("Photosynthesis", payload)], mode="qa_synth")
    out, stats = _run(cfg, [{"id": "doc1", "text": TEXT}])
    assert [r["id"] for r in out] == ["doc1#qa1"]
    assert stats.dropped["ungrounded"] == 1
    assert stats.dropped["llm_invalid_pair"] == 1
    assert stats.extra["pairs_generated"] == 3
    assert stats.extra["pairs_kept"] == 1


def test_reasoning_trace_verified_exact_match():
    trace = "Recall the capital city of France.\nAnswer: Paris"
    _, cfg = _setup([("capital of France", trace)], mode="reasoning_trace")
    recs = [{"id": "t1", "prompt": "What is the capital of France?", "answer": "paris"}]
    out, stats = _run(cfg, recs)
    assert len(out) == 1
    rec = out[0]
    assert rec["messages"][0] == {"role": "user", "content": "What is the capital of France?"}
    assert rec["messages"][1]["role"] == "assistant"
    assert rec["messages"][1]["content"] == trace
    assert rec["answer"] == "paris"
    assert rec["meta"]["verified"] is True
    assert rec["meta"]["llm"][0]["mode"] == "reasoning_trace"
    assert stats.extra["verified_rate"] == 1.0


def test_reasoning_trace_prompt_from_last_user_message():
    trace = "Think it through.\nAnswer: Paris"
    _, cfg = _setup([("capital of France", trace)], mode="reasoning_trace")
    recs = [
        {
            "id": "t2",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "I will think."},
            ],
            "answer": "paris",
        }
    ]
    out, _ = _run(cfg, recs)
    assert len(out) == 1 and out[0]["meta"]["verified"] is True


def test_reasoning_trace_wrong_answer_dropped():
    trace = "Guesswork.\nAnswer: Paris"
    _, cfg = _setup([("capital", trace)], mode="reasoning_trace")
    recs = [{"id": "t1", "prompt": "name the capital", "answer": "london"}]
    out, stats = _run(cfg, recs)
    assert out == []
    assert stats.dropped["trace_wrong_answer"] == 1
    assert stats.extra["verified_rate"] == 0.0


def test_reasoning_trace_no_answer_line_dropped():
    _, cfg = _setup([("capital", "I think it might be Paris.")], mode="reasoning_trace")
    recs = [{"id": "t1", "prompt": "name the capital", "answer": "paris"}]
    out, stats = _run(cfg, recs)
    assert out == []
    assert stats.dropped["trace_no_answer"] == 1
    assert stats.extra["verified_rate"] is None  # nothing attempted


def test_reasoning_trace_no_gold_drops_unverifiable_by_default():
    trace = "Reasoning.\nAnswer: Paris"
    _, cfg = _setup([("capital", trace)], mode="reasoning_trace")
    out, stats = _run(cfg, [{"id": "t1", "prompt": "name the capital"}])
    assert out == []
    assert stats.dropped["unverifiable"] == 1


def test_reasoning_trace_keep_unverified_marks_none_not_guessed():
    trace = "Reasoning.\nAnswer: Paris"
    _, cfg = _setup(
        [("capital", trace)], mode="reasoning_trace", keep_unverified=True
    )
    out, _ = _run(cfg, [{"id": "t1", "prompt": "name the capital"}])
    assert len(out) == 1
    assert "verified" in out[0]["meta"]
    assert out[0]["meta"]["verified"] is None


def test_reasoning_trace_mcq_letter_case_insensitive():
    trace = "Eliminate options.\nAnswer: (B)"
    _, cfg = _setup(
        [("pick one", trace)], mode="reasoning_trace", answer_kind="mcq_letter"
    )
    recs = [{"id": "t1", "prompt": "pick one of A-D", "answer": "b"}]
    out, stats = _run(cfg, recs)
    assert len(out) == 1 and out[0]["meta"]["verified"] is True
    assert stats.extra["verified_rate"] == 1.0


def test_reasoning_trace_missing_prompt_drops_no_prompt():
    backend, cfg = _setup([], mode="reasoning_trace")
    out, stats = _run(cfg, [{"id": "t1", "answer": "paris"}])
    assert out == []
    assert stats.dropped["no_prompt"] == 1
    assert len(backend.calls) == 0


def test_judge_pointwise_high_score_kept():
    _, cfg = _setup([("Photosynthesis", '{"score": 5, "reason": "accurate"}')], mode="judge")
    out, stats = _run(cfg, [{"id": "s1", "text": TEXT}])
    assert len(out) == 1
    assert out[0]["meta"]["judge_score"] == 5
    assert "judge_low_score" not in stats.dropped
    assert out[0]["meta"]["llm"][0]["mode"] == "judge"


def test_judge_pointwise_low_score_dropped():
    _, cfg = _setup([("Photosynthesis", '{"score": 2}')], mode="judge")
    out, stats = _run(cfg, [{"id": "s1", "text": TEXT}])
    assert out == []
    assert stats.dropped["judge_low_score"] == 1


def test_judge_pointwise_respects_configured_min_score():
    _, cfg = _setup(
        [("Photosynthesis", '{"score": 4}')], mode="judge", min_score=5
    )
    out, stats = _run(cfg, [{"id": "s1", "text": TEXT}])
    assert out == [] and stats.dropped["judge_low_score"] == 1


def test_judge_pointwise_unparseable_kept_score_none_unmeasured():
    _, cfg = _setup([("Photosynthesis", "I cannot grade this.")], mode="judge")
    out, stats = _run(cfg, [{"id": "s1", "text": TEXT}])
    assert len(out) == 1
    assert "judge_score" in out[0]["meta"]
    assert out[0]["meta"]["judge_score"] is None  # never guessed
    assert stats.extra["unmeasured"] == 1
    assert stats.records_out == 1


def test_judge_pointwise_uses_messages_when_present():
    _, cfg = _setup([("user: hi", '{"score": 4}')], mode="judge")
    recs = [
        {
            "id": "s1",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
            ],
        }
    ]
    out, _ = _run(cfg, recs)
    assert len(out) == 1 and out[0]["meta"]["judge_score"] == 4


def test_judge_pairwise_consistent_chosen_kept_with_order_swap():
    rules = [
        ("Response A:\nALPHA", '{"winner": "A"}'),
        ("Response A:\nBETA", '{"winner": "B"}'),
    ]
    backend, cfg = _setup(rules, mode="judge")
    recs = [
        {
            "id": "p1",
            "prompt": "explain gravity",
            "chosen": "ALPHA concise correct answer",
            "rejected": "BETA wrong rambling answer",
        }
    ]
    out, stats = _run(cfg, recs)
    assert len(out) == 1
    assert out[0]["meta"]["judge_agrees"] is True
    assert len(out[0]["meta"]["llm"]) == 2  # one provenance entry per call
    assert len(backend.calls) == 2
    assert "Response A:\nALPHA" in backend.calls[0]["messages"][0]["content"]
    assert "Response A:\nBETA" in backend.calls[1]["messages"][0]["content"]
    assert stats.extra["calls"] == 2
    assert stats.extra["position_consistency"] == 1.0


def test_judge_pairwise_rejected_wins_dropped():
    rules = [
        ("Response A:\nALPHA", '{"winner": "B"}'),
        ("Response A:\nBETA", '{"winner": "A"}'),
    ]
    _, cfg = _setup(rules, mode="judge")
    recs = [
        {
            "id": "p1",
            "prompt": "explain gravity",
            "chosen": "ALPHA bad answer",
            "rejected": "BETA good answer",
        }
    ]
    out, stats = _run(cfg, recs)
    assert out == []
    assert stats.dropped["judge_prefers_rejected"] == 1
    assert stats.extra["position_consistency"] == 1.0


def test_judge_pairwise_inconsistent_kept_by_default():
    _, cfg = _setup([("Response A:", '{"winner": "A"}')], mode="judge")
    recs = [{"id": "p1", "prompt": "q", "chosen": "C answer", "rejected": "R answer"}]
    out, stats = _run(cfg, recs)
    assert len(out) == 1
    assert out[0]["meta"]["judge_agrees"] is None
    assert stats.extra["pairs_inconsistent"] == 1
    assert stats.extra["position_consistency"] == 0.0


def test_judge_pairwise_drop_inconsistent_drops_record():
    _, cfg = _setup(
        [("Response A:", '{"winner": "A"}')], mode="judge", drop_inconsistent=True
    )
    recs = [{"id": "p1", "prompt": "q", "chosen": "C answer", "rejected": "R answer"}]
    out, stats = _run(cfg, recs)
    assert out == []
    assert stats.dropped["judge_inconsistent"] == 1


def test_judge_pairwise_unusable_verdict_kept_never_dropped_as_inconsistent():
    _, cfg = _setup(
        [("Response A:", "cannot decide")], mode="judge", drop_inconsistent=True
    )
    recs = [{"id": "p1", "prompt": "q", "chosen": "C answer", "rejected": "R answer"}]
    out, stats = _run(cfg, recs)
    assert len(out) == 1
    assert out[0]["meta"]["judge_agrees"] is None
    assert stats.extra["unmeasured"] == 1
    assert stats.dropped.get("judge_inconsistent", 0) == 0
    assert stats.extra["position_consistency"] == 0.0


def test_budget_exhausted_records_pass_through_unenhanced():
    _, cfg = _setup(
        [("Photosynthesis", REWRITE)], mode="rephrase", max_calls=1, workers=1
    )
    recs = [{"id": "d1", "text": TEXT}, {"id": "d2", "text": TEXT}]
    out, stats = _run(cfg, recs)
    assert [r["id"] for r in out] == ["d1", "d1#rephrase-wikipedia", "d2"]
    assert "llm" not in (out[2].get("meta") or {})
    assert stats.extra["budget_exhausted"] is True
    assert stats.extra["calls"] == 1
    assert stats.records_out == 3


def test_output_order_preserved_with_multiple_workers():
    _, cfg = _setup(
        [("Photosynthesis", REWRITE)], mode="rephrase", emit="replace", workers=4
    )
    recs = [{"id": f"d{i}", "text": TEXT} for i in range(8)]
    out, stats = _run(cfg, recs)
    expected = [f"d{i}#rephrase-wikipedia" for i in range(8)]
    assert [r["id"] for r in out] == expected
    assert stats.records_in == 8 and stats.records_out == 8


def test_stats_backend_and_extra_reporting():
    _, cfg = _setup([("Photosynthesis", REWRITE)], mode="rephrase")
    _, stats = _run(cfg, [{"id": "d1", "text": TEXT}])
    assert stats.backend == "llm:fake:fake-1"
    assert stats.extra["model"] == "fake-1"
    assert stats.extra["mode"] == "rephrase"
    assert isinstance(stats.extra["prompt_hash"], str)
    assert stats.extra["cache_hits"] == 0
    assert stats.extra["errors"] == 0


def test_cache_hits_counted_and_in_provenance():
    _, cfg = _setup([("Photosynthesis", REWRITE)], mode="rephrase", cache_hit=True)
    out, stats = _run(cfg, [{"id": "d1", "text": TEXT}])
    assert stats.extra["cache_hits"] == 1
    assert stats.extra["cache_hits"] == stats.extra["calls"]
    assert out[1]["meta"]["llm"][0]["cache_hit"] is True


def test_max_chars_truncates_input_and_is_counted():
    backend, cfg = _setup(
        [("Photosynthesis", REWRITE)], mode="rephrase", max_chars=10
    )
    _, stats = _run(cfg, [{"id": "d1", "text": TEXT}])
    sent = backend.calls[0]["messages"][0]["content"]
    assert f"Document:\n{TEXT[:10]}\n" in sent
    assert TEXT[11:] not in sent
    assert stats.extra["input_truncated"] == 1


def test_provenance_appended_not_overwritten():
    _, cfg = _setup([("Photosynthesis", REWRITE)], mode="rephrase")
    recs = [{"id": "d1", "text": TEXT, "meta": {"llm": [{"op": "prior_op"}]}}]
    out, _ = _run(cfg, recs)
    llm_meta = out[1]["meta"]["llm"]
    assert llm_meta[0] == {"op": "prior_op"}
    assert llm_meta[1]["op"] == "llm_enhance"
    assert len(llm_meta) == 2


def test_unknown_mode_refuses_listing_modes():
    _, cfg = _setup([], mode="hax")
    with pytest.raises(ValueError, match="unknown mode") as excinfo:
        _run(cfg, [{"id": "d1", "text": TEXT}])
    message = str(excinfo.value)
    assert "rephrase" in message and "judge" in message


def test_missing_backend_refuses_naming_input():
    _setup()
    with pytest.raises(ValueError, match="config.backend") as excinfo:
        _run({"mode": "rephrase"}, [{"id": "d1", "text": TEXT}])
    assert "llm_enhance: missing input" in str(excinfo.value)
