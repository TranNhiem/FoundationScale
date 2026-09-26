
"""Tests for the tokenize op: fallback approximation, counting rules, packing."""
from __future__ import annotations

from foundationskills.skills.data_engine.ops.base import OpStats
from foundationskills.skills.data_engine.ops import tokenize as tok_mod
from foundationskills.skills.data_engine.ops.tokenize import pack_lengths, tokenize_op


def run_tokenize(records, cfg):
    stats = OpStats("tokenize")
    out = list(tokenize_op(records, cfg, stats))
    return out, stats


def test_approx_backend_is_flagged_and_meta_num_tokens_added():
    out, stats = run_tokenize(
        [{"id": "a", "text": "abcd", "meta": {"domain": "general"}}],
        {"seq_len": 16},
    )
    assert stats.backend == "approx"
    assert stats.extra["approximate"] is True
    assert out[0]["meta"]["num_tokens"] == 2  # ceil(4/4) + 1 EOS
    assert stats.extra["num_tokens"] == 2
    assert stats.extra["domain_counts"] == {"general": 1}


def test_approx_cjk_ratio():
    out, _stats = run_tokenize(
        [{"id": "c", "text": "汉汉汉", "meta": {}}],
        {"seq_len": 16, "add_eos": False},
    )
    assert out[0]["meta"]["num_tokens"] == 2  # ceil(3 / 1.5)


def test_truncation_rate_and_histogram():
    out, stats = run_tokenize(
        [
            {"id": "long", "text": "a" * 16, "meta": {}},   # 4 tokens
            {"id": "short", "text": "ab", "meta": {}},       # 1 token
        ],
        {"seq_len": 2, "add_eos": False},
    )
    assert stats.extra["truncation_rate"] == 0.5
    assert stats.extra["length_hist"] == {"1": 1, "4": 1}  # power-of-two buckets
    assert stats.extra["max"] == 4
    assert stats.extra["p50"] == 1
    assert stats.extra["p90"] == 4
    assert {r["meta"]["num_tokens"] for r in out} == {4, 1}


def test_preference_counts_prompt_plus_chosen_not_rejected():
    rec = {"id": "p", "prompt": "abcd efgh", "chosen": "abcd efgh", "rejected": "x" * 400}
    out, stats = run_tokenize([rec], {"seq_len": 64, "add_eos": False})
    # each text: ceil(9/4) = 3 tokens -> prompt + chosen = 6
    assert out[0]["meta"]["num_tokens"] == 6
    assert stats.extra["num_tokens"] == 6


def test_rl_counts_human_prompt_plus_declared_answer():
    rec = {
        "id": "r",
        "conversations": [{"from": "human", "value": "abcd efgh"}, {"from": "gpt", "value": "z" * 400}],
        "answer": "abcd",
    }
    out, _stats = run_tokenize([rec], {"seq_len": 64, "add_eos": False, "field": "auto"})
    # human ceil(9/4)=3 plus answer ceil(4/4)=1 = 4; the long gpt turn is NOT counted
    assert out[0]["meta"]["num_tokens"] == 4


class _FakeTokenizer:
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def test_real_tokenizer_path_via_monkeypatched_loader(monkeypatch):
    monkeypatch.setattr(tok_mod, "_load_tokenizer", lambda path: (_FakeTokenizer(), None))
    out, stats = run_tokenize(
        [{"id": "t", "text": "one two three", "meta": {}}],
        {"seq_len": 16, "tokenizer": "/any/path"},
    )
    assert stats.backend == "transformers"
    assert "approximate" not in stats.extra
    assert out[0]["meta"]["num_tokens"] == 4  # 3 tokens + EOS


def test_broken_tokenizer_load_falls_back_with_error_recorded():
    out, stats = run_tokenize(
        [{"id": "e", "text": "abcd", "meta": {}}],
        {"seq_len": 16, "tokenizer": "/definitely/not/a/real/tokenizer-path-zz"},
    )
    assert stats.backend == "approx"
    assert stats.extra["approximate"] is True
    assert "tokenizer_error" in stats.extra
    assert out[0]["meta"]["num_tokens"] == 2


def test_broken_tokenizer_encode_flips_approximate_per_record(monkeypatch):
    class _FlakyTokenizer:
        def encode(self, text, add_special_tokens=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(tok_mod, "_load_tokenizer", lambda path: (_FlakyTokenizer(), None))
    _out, stats = run_tokenize(
        [{"id": "f", "text": "abcd", "meta": {}}],
        {"seq_len": 16, "tokenizer": "/any/path", "add_eos": False},
    )
    assert stats.extra["approximate"] is True
    assert "tokenizer_error" in stats.extra


def test_pack_lengths_first_fit_decreasing_hand_case():
    # FFD on [6,5,4,4,3] with seq_len 10: bins = [6+4], [5+4], [3] -> 3 bins.
    result = pack_lengths([6, 5, 4, 4, 3], 10)
    assert result["num_sequences"] == 3
    assert result["total_tokens"] == 22
    assert result["efficiency"] == 22 / 30
    assert result["seq_len"] == 10


def test_pack_lengths_overlong_record_takes_its_own_bin():
    result = pack_lengths([12, 2], 10)
    # 12 is truncated to 10 and fills a bin by itself; the 2 cannot join it.
    assert result["num_sequences"] == 2
    assert result["efficiency"] == 12 / 20


def test_pack_lengths_empty_is_zero():
    result = pack_lengths([], 8)
    assert result["num_sequences"] == 0
    assert result["efficiency"] == 0.0


def test_pack_cfg_adds_packing_extra():
    _out, stats = run_tokenize(
        [{"id": "a", "text": "a" * 40, "meta": {}}, {"id": "b", "text": "a" * 40, "meta": {}}],
        {"seq_len": 16, "add_eos": False, "pack": True},
    )
    packing = stats.extra["packing"]
    # two 10-token records (16 chars -> 4... no: 40 chars -> 10 tokens each), pack into one bin of 16
    assert packing["num_sequences"] == 2  # 10 + 10 does not fit in 16
    assert packing["total_tokens"] == 20
