"""Unit tests for the ``llm_classify`` op: scripted fake backend, no network."""
from __future__ import annotations

import hashlib
import re

import foundationskills.skills.data_engine.ops.llm_classify  # noqa: F401  (registers the op)
import pytest
from foundationskills.skills.data_engine.llm_backend import (
    LLMOpError,
    LLMResponse,
    register_backend,
)
from foundationskills.skills.data_engine.ops.base import OPS, OpStats

GOOD = '{"domain": "tech", "quality": 4}'


class FakeBackend:
    """Scripted backend: response keyed on a substring of the last user message."""

    kind = "fake"
    model = "fake-1"

    def __init__(self, handlers=(), default=GOOD):
        self._handlers = list(handlers)
        self._default = default
        self.calls: list[dict] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, seed=0,
                 json_mode=False) -> LLMResponse:
        self.calls.append({"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "seed": seed, "json_mode": json_mode})
        text = messages[-1]["content"]
        for needle, payload in self._handlers:
            if needle in text:
                return self._resp(payload)
        return self._resp(self._default)

    @staticmethod
    def _resp(payload) -> LLMResponse:
        if isinstance(payload, LLMResponse):
            return payload
        return LLMResponse(content=payload, finish_reason="stop", prompt_tokens=7,
                           completion_tokens=3, cache_hit=False, error=None)


@pytest.fixture()
def op():
    return OPS["llm_classify"]


def _use(backend: FakeBackend) -> FakeBackend:
    register_backend("fake", lambda cfg: backend)  # overwrites allowed across tests
    return backend


def _cfg(**overrides) -> dict:
    cfg = {
        "backend": {"kind": "fake"},
        "axes": [
            {"name": "domain", "description": "Topic area.",
             "labels": ["tech", "news", "fiction"]},
            {"name": "quality", "description": "1 low, 5 high.", "scale": [1, 5]},
        ],
        "workers": 1,
    }
    cfg.update(overrides)
    return cfg


def _run(op, records, cfg):
    stats = OpStats("llm_classify")
    out = list(op(iter(records), cfg, stats))
    return out, stats


def _in_sample(rec_id: str, seed: int, rate: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{rec_id}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64) < rate


def test_labels_and_stats_happy_path(op):
    backend = _use(FakeBackend(handlers=[
        ("alpha", '{"domain": "tech", "quality": 4}'),
        ("beta", '{"domain": "news", "quality": 2}'),
    ]))
    records = [{"id": "1", "text": "alpha doc"}, {"id": "2", "text": "beta doc"}]
    out, stats = _run(op, records, _cfg())
    assert [r["meta"]["llm_labels"] for r in out] == [
        {"domain": "tech", "quality": 4}, {"domain": "news", "quality": 2}]
    assert stats.records_in == 2 and stats.records_out == 2
    assert stats.backend == "llm:fake:fake-1"
    assert stats.modified["labels_added"] == 2
    assert stats.extra["calls"] == 2 and stats.extra["cache_hits"] == 0
    assert stats.extra["prompt_tokens"] == 14 and stats.extra["completion_tokens"] == 6
    assert stats.extra["errors"] == 0
    assert stats.extra["labeled"] == 2 and stats.extra["unmeasured"] == 0
    assert stats.extra["distribution"] == {
        "domain": {"tech": 1, "news": 1}, "quality": {"4": 1, "2": 1}}
    assert stats.extra["axes"] == ["domain", "quality"]
    assert stats.extra["model"] == "fake-1"
    assert re.fullmatch(r"[0-9a-f]{16}", stats.extra["prompt_hash"])
    assert not stats.dropped
    assert len(backend.calls) == 2


def test_system_prompt_lists_axes_and_defaults(op):
    backend = _use(FakeBackend())
    _run(op, [{"id": "1", "text": "anything"}], _cfg())
    call = backend.calls[0]
    system = call["messages"][0]
    assert system["role"] == "system" and call["messages"][1]["role"] == "user"
    assert '"domain"' in system["content"] and '"tech"' in system["content"]
    assert "[1, 5]" in system["content"]
    assert call["json_mode"] is True
    assert call["temperature"] == 0.0 and call["seed"] == 0 and call["max_tokens"] == 512


def test_case_insensitive_label_and_numeric_ordinals(op):
    _use(FakeBackend(handlers=[
        ("aaa", '{"domain": " TECH ", "quality": "4"}'),
        ("bbb", '{"domain": "News", "quality": 3.0}'),
    ]))
    out, stats = _run(op, [{"id": "1", "text": "aaa"}, {"id": "2", "text": "bbb"}], _cfg())
    assert out[0]["meta"]["llm_labels"] == {"domain": "tech", "quality": 4}
    assert out[1]["meta"]["llm_labels"] == {"domain": "news", "quality": 3}
    assert stats.extra["invalid"] == {"domain": 0, "quality": 0}


def test_invalid_label_and_out_of_range_become_none(op):
    _use(FakeBackend(default='{"domain": "space", "quality": 9}'))
    out, stats = _run(op, [{"id": "1", "text": "mystery"}], _cfg())
    assert out[0]["meta"]["llm_labels"] == {"domain": None, "quality": None}
    assert stats.extra["invalid"] == {"domain": 1, "quality": 1}
    assert stats.extra["unmeasured"] == 1 and stats.extra["labeled"] == 0
    assert stats.extra["distribution"] == {"domain": {}, "quality": {}}
    assert stats.records_out == 1  # unmeasured labels are kept, never faked


def test_unparseable_reply_counts_invalid_per_axis(op):
    _use(FakeBackend(default="I cannot classify this document, sorry."))
    out, stats = _run(op, [{"id": "1", "text": "plain prose"}], _cfg())
    assert out[0]["meta"]["llm_labels"] == {"domain": None, "quality": None}
    assert stats.extra["invalid"] == {"domain": 1, "quality": 1}
    assert stats.extra["errors"] == 0  # call succeeded; the content was unusable


def test_missing_backend_refuses(op):
    cfg = _cfg()
    del cfg["backend"]
    with pytest.raises(ValueError, match="config.backend"):
        list(op([], cfg, OpStats("llm_classify")))


def test_duplicate_axis_names_refuse(op):
    cfg = _cfg(axes=[{"name": "dup", "labels": ["a", "b"]},
                     {"name": "dup", "scale": [1, 3]}])
    with pytest.raises(LLMOpError, match="duplicate axis"):
        list(op([], cfg, OpStats("llm_classify")))


@pytest.mark.parametrize("axis", [
    {"name": "x"},  # neither kind
    {"name": "x", "labels": ["a", "b"], "scale": [1, 2]},  # both kinds
])
def test_axis_needs_exactly_one_kind(op, axis):
    with pytest.raises(LLMOpError, match="exactly one"):
        list(op([], _cfg(axes=[axis]), OpStats("llm_classify")))


@pytest.mark.parametrize("rate", [0.0, 1.5])
def test_sample_rate_out_of_bounds_refuses(op, rate):
    cfg = _cfg(sample={"rate": rate, "seed": 0})
    with pytest.raises(LLMOpError, match="sample.rate"):
        list(op([], cfg, OpStats("llm_classify")))


def test_unknown_filter_axis_refuses(op):
    cfg = _cfg(filters={"nope": {"in": ["a"]}})
    with pytest.raises(LLMOpError, match="unknown axis"):
        list(op([], cfg, OpStats("llm_classify")))


def test_filter_in_drops_with_reason(op):
    _use(FakeBackend(handlers=[
        ("keep", '{"domain": "tech", "quality": 4}'),
        ("drop", '{"domain": "news", "quality": 4}'),
    ]))
    records = [{"id": "1", "text": "keep me"}, {"id": "2", "text": "drop me"}]
    out, stats = _run(op, records, _cfg(filters={"domain": {"in": ["tech"]}}))
    assert [r["id"] for r in out] == ["1"]
    assert stats.dropped == {"llm_filter:domain": 1}
    # the dropped record WAS measured, so it still shows in the distribution
    assert stats.extra["distribution"]["domain"] == {"tech": 1, "news": 1}


def test_filter_gte_drops_with_reason(op):
    _use(FakeBackend(handlers=[
        ("low", '{"domain": "tech", "quality": 2}'),
        ("high", '{"domain": "tech", "quality": 5}'),
    ]))
    records = [{"id": "1", "text": "low"}, {"id": "2", "text": "high"}]
    out, stats = _run(op, records, _cfg(filters={"quality": {"gte": 3}}))
    assert [r["id"] for r in out] == ["2"]
    assert stats.dropped == {"llm_filter:quality": 1}


def test_none_label_kept_and_counted_unmeasured(op):
    _use(FakeBackend(default='{"domain": "news"}'))  # quality never answered
    out, stats = _run(op, [{"id": "1", "text": "partial"}], _cfg())
    assert out[0]["meta"]["llm_labels"] == {"domain": "news", "quality": None}
    assert stats.records_out == 1 and not stats.dropped
    assert stats.extra["unmeasured"] == 1 and stats.extra["labeled"] == 0


def test_drop_unmeasured_true_drops(op):
    _use(FakeBackend(default='{"domain": "news"}'))
    out, stats = _run(op, [{"id": "1", "text": "partial"}], _cfg(drop_unmeasured=True))
    assert out == [] and stats.records_out == 0
    assert stats.dropped == {"llm_unmeasured": 1}


def test_backend_error_all_axes_none(op):
    failure = LLMResponse(content=None, finish_reason=None, prompt_tokens=0,
                          completion_tokens=0, cache_hit=False, error="upstream 500")
    _use(FakeBackend(default=failure))
    out, stats = _run(op, [{"id": "1", "text": "anything"}], _cfg())
    assert out[0]["meta"]["llm_labels"] == {"domain": None, "quality": None}
    assert stats.extra["errors"] == 1 and stats.extra["calls"] == 1
    assert stats.extra["invalid"] == {"domain": 0, "quality": 0}
    entry = out[0]["meta"]["llm"][-1]  # the call WAS attempted: provenance stands
    assert entry["op"] == "llm_classify" and entry["cache_hit"] is False


def test_cache_hit_counted_and_in_provenance(op):
    cached = LLMResponse(content=GOOD, finish_reason="stop", prompt_tokens=7,
                         completion_tokens=3, cache_hit=True, error=None)
    backend = _use(FakeBackend(default=cached))
    out, stats = _run(op, [{"id": "1", "text": "anything"}], _cfg())
    assert stats.extra["cache_hits"] == 1 and stats.extra["calls"] == 1
    assert out[0]["meta"]["llm"][-1]["cache_hit"] is True
    assert backend.calls  # the cached backend still saw the request


def test_budget_exhaustion_marks_all_none(op):
    backend = _use(FakeBackend())
    records = [{"id": "1", "text": "first"}, {"id": "2", "text": "second"}]
    out, stats = _run(op, records, _cfg(max_calls=1))
    assert len(out) == 2  # unmeasured is not unusable: kept by default
    assert out[0]["meta"]["llm_labels"] == {"domain": "tech", "quality": 4}
    assert out[1]["meta"]["llm_labels"] == {"domain": None, "quality": None}
    assert "llm" not in out[1]["meta"]  # no call was made: no provenance claimed
    assert stats.extra["budget_exhausted"] is True
    assert stats.extra["calls"] == 1 and len(backend.calls) == 1
    assert stats.extra["unmeasured"] == 1 and stats.extra["labeled"] == 1


def test_sampling_deterministic_subset_untouched(op):
    backend = _use(FakeBackend())
    rate = 0.5
    seed = next(s for s in range(100)
                if 0 < sum(_in_sample(f"r{i}", s, rate) for i in range(10)) < 10)
    records = [{"id": f"r{i}", "text": f"doc {i}"} for i in range(10)]
    out, stats = _run(op, records, _cfg(sample={"rate": rate, "seed": seed}))
    expected = {r["id"] for r in records if _in_sample(r["id"], seed, rate)}
    assert len(backend.calls) == len(expected)
    assert stats.extra["sampled_out"] == 10 - len(expected)
    assert stats.extra["labeled"] == len(expected)
    assert stats.records_out == 10
    for original, result in zip(records, out):
        if original["id"] in expected:
            assert result is not original
            assert result["meta"]["llm_labels"] == {"domain": "tech", "quality": 4}
            assert "meta" not in original  # input record never mutated
        else:
            assert result is original  # untouched: no labels key, not even None
            assert "meta" not in result


def test_messages_fallback_to_role_lines(op):
    backend = _use(FakeBackend())
    rec = {"id": "1", "messages": [{"role": "system", "content": "be nice"},
                                   {"role": "user", "content": "hello agent"}]}
    out, stats = _run(op, [rec], _cfg())
    assert backend.calls[0]["messages"][1]["content"] == "system: be nice\nuser: hello agent"
    assert out[0]["meta"]["llm_labels"]["domain"] == "tech"


def test_preference_record_uses_prompt_and_chosen(op):
    backend = _use(FakeBackend())
    rec = {"id": "1", "prompt": "why is the sky blue", "chosen": "rayleigh scattering"}
    _run(op, [rec], _cfg())
    assert backend.calls[0]["messages"][1]["content"] == "why is the sky blue\nrayleigh scattering"


def test_custom_field_is_classified(op):
    backend = _use(FakeBackend())
    _run(op, [{"id": "1", "body": "custom body text"}], _cfg(field="body"))
    assert backend.calls[0]["messages"][1]["content"] == "custom body text"


def test_long_inputs_are_truncated_and_counted(op):
    backend = _use(FakeBackend())
    records = [{"id": "1", "text": "abcdefghij" * 10}, {"id": "2", "text": "short"}]
    out, stats = _run(op, records, _cfg(max_chars=15))
    assert backend.calls[0]["messages"][1]["content"] == "abcdefghijabcde"
    assert backend.calls[1]["messages"][1]["content"] == "short"
    assert stats.modified["input_truncated"] == 1
    assert out[0]["meta"]["llm_labels"]["quality"] == 4


def test_record_without_any_text_is_dropped(op):
    backend = _use(FakeBackend())
    records = [{"id": "1"}, {"id": "2", "text": "   \n  "}, {"id": "3", "text": "real"}]
    out, stats = _run(op, records, _cfg())
    assert [r["id"] for r in out] == ["3"]
    assert stats.dropped == {"empty_text": 2}
    assert stats.records_in == 3 and stats.records_out == 1
    assert len(backend.calls) == 1  # no wasted LLM call on empty documents


def test_order_preserved_with_workers(op):
    handlers = [(f"marker {i}", '{"domain": "news", "quality": 3}')
                for i in range(0, 30, 3)]
    _use(FakeBackend(handlers=handlers, default='{"domain": "tech", "quality": 5}'))
    records = [{"id": f"{i:02d}", "text": f"marker {i} body"} for i in range(30)]
    out, stats = _run(op, records, _cfg(workers=4))
    assert [r["id"] for r in out] == [f"{i:02d}" for i in range(30)]
    assert out[3]["meta"]["llm_labels"]["domain"] == "news"
    assert stats.extra["distribution"]["domain"] == {"news": 10, "tech": 20}


def test_provenance_appends_not_overwrites(op):
    _use(FakeBackend())
    rec = {"id": "1", "text": "seeded meta",
           "meta": {"llm": [{"op": "upstream_thing", "model": "m0"}]}}
    out, stats = _run(op, [rec], _cfg())
    history = out[0]["meta"]["llm"]
    assert history[0] == {"op": "upstream_thing", "model": "m0"}
    assert len(history) == 2
    entry = history[1]
    assert entry["op"] == "llm_classify"
    assert entry["backend"] == "fake" and entry["model"] == "fake-1"
    assert entry["prompt_hash"] == stats.extra["prompt_hash"]
    assert entry["cache_hit"] is False
