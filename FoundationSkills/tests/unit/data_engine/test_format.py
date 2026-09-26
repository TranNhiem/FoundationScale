
"""Tests for the format op (every target format + chat rendering paths)."""
from __future__ import annotations

import pytest

from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops.format import format_op, render_chat


def run_format(records, cfg):
    stats = OpStats("format")
    out = list(format_op(records, cfg, stats))
    return out, stats


def test_registered_in_ops():
    assert OPS["format"] is format_op


def test_pretrain_and_cpt_pass_text_through():
    recs = [{"id": "a", "text": "hello world", "meta": {"source": "s"}}]
    for target in ("pretrain", "cpt"):
        out, stats = run_format(recs, {"target_format": target})
        assert out == [{"id": "a", "text": "hello world", "meta": {"source": "s"}}]
        assert stats.records_in == stats.records_out == 1
        assert stats.extra["fs_columns"]["text_column"] == "text"


def test_pretrain_unconvertible_without_text():
    out, stats = run_format([{"id": "x", "meta": {}}], {"target_format": "pretrain"})
    assert out == []
    assert stats.dropped["unconvertible:pretrain"] == 1


def test_sft_from_messages_has_exact_shape():
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    out, stats = run_format(
        [{"id": "s1", "messages": msgs, "meta": {}}],
        {"target_format": "sft", "chat_template_family": "chatml"},
    )
    assert set(out[0].keys()) == {"id", "messages", "text", "meta"}
    assert "<|im_start|>user\nQ<|im_end|>" in out[0]["text"]
    # FS has no assistant-only masking; the op must disclose full-sequence loss.
    assert stats.extra["sft_loss_scope"] == "full_sequence"
    assert stats.extra["fs_columns"]["text_column"] == "text"


def test_sft_from_alpaca_fields():
    alpaca = {"id": "al1", "instruction": "Say hi", "input": "loudly", "output": "HI", "meta": {"domain": "d"}}
    out, _stats = run_format([alpaca], {"target_format": "sft", "chat_template_family": "chatml"})
    assert out[0]["messages"] == [
        {"role": "user", "content": "Say hi\n\nloudly"},
        {"role": "assistant", "content": "HI"},
    ]


def test_sft_system_prompt_prepended():
    rec = {"id": "s2", "prompt": "Q", "answer": "A"}
    out, _stats = run_format(
        [rec],
        {"target_format": "sft", "chat_template_family": "chatml", "system_prompt": "SYS"},
    )
    assert out[0]["messages"][0] == {"role": "system", "content": "SYS"}


def test_mm_sft_uses_first_image_and_records_image_column():
    msgs = [{"role": "user", "content": "look"}, {"role": "assistant", "content": "ok"}]
    rec = {"id": "m1", "messages": msgs, "images": ["img/a.png", "img/b.png"], "meta": {}}
    out, stats = run_format([rec], {"target_format": "mm_sft", "chat_template_family": "chatml"})
    assert out[0]["image"] == "img/a.png"
    assert stats.extra["fs_columns"]["image_column"] == "image"
    assert stats.extra["sft_loss_scope"] == "full_sequence"


def test_mm_sft_without_images_is_unconvertible():
    msgs = [{"role": "user", "content": "look"}, {"role": "assistant", "content": "ok"}]
    out, stats = run_format([{"id": "m2", "messages": msgs}], {"target_format": "mm_sft", "chat_template_family": "chatml"})
    assert out == []
    assert stats.dropped["unconvertible:mm_sft"] == 1


def test_preference_shape():
    rec = {"id": "p1", "prompt": "Q", "chosen": "good", "rejected": "bad", "meta": {}}
    out, _stats = run_format([rec], {"target_format": "preference"})
    assert out == [{"id": "p1", "prompt": "Q", "chosen": "good", "rejected": "bad", "meta": {}}]


def test_preference_missing_field_dropped():
    out, stats = run_format([{"id": "p2", "prompt": "Q", "chosen": "good"}], {"target_format": "preference"})
    assert out == []
    assert stats.dropped["unconvertible:preference"] == 1


def test_rl_output_is_accepted_by_fs_corpus_parser():
    from foundationscale.rl.corpus import _parse_record

    rec = {"id": "r1", "prompt": "2+2?", "answer": "4", "meta": {}}
    out, stats = run_format([rec], {"target_format": "rl"})
    record = out[0]
    assert record["conversations"] == [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}]
    assert record["answer"] == "4"
    assert stats.extra["gold_key"] == "answer"
    assert stats.extra["fs_columns"]["gold_key"] == "answer"
    sample = _parse_record(record, 0, "answer")
    assert sample.response == "4"
    assert sample.prompt_turns == (("user", "2+2?"),)
    # "4" is not a single A-Z letter: FS abstains on the gold, never crashes.
    assert sample.gold is None


def test_rl_single_letter_gold_is_kept_by_fs():
    from foundationscale.rl.corpus import _parse_record

    rec = {"id": "r2", "question": "Pick A or B. Answer with a single letter.", "answer": "B"}
    out, _stats = run_format([rec], {"target_format": "rl"})
    sample = _parse_record(out[0], 0, "answer")
    assert sample.gold == "B"
    assert sample.response == "B"


def test_rl_from_messages_and_custom_gold_key():
    from foundationscale.rl.corpus import _parse_record

    rec = {
        "id": "r3",
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ],
    }
    out, stats = run_format([rec], {"target_format": "rl", "gold_key": "gold"})
    assert out[0]["system"] == "SYS"
    assert out[0]["gold"] == "A"
    assert stats.extra["gold_key"] == "gold"
    sample = _parse_record(out[0], 0, "gold")
    assert sample.response == "A"
    assert ("system", "SYS") in sample.prompt_turns


def test_unconvertible_strict_drop_and_non_strict_passthrough():
    rec = {"id": "u1", "meta": {}}
    out, stats = run_format([rec], {"target_format": "sft"})
    assert out == [] and stats.dropped["unconvertible:sft"] == 1

    out2, stats2 = run_format([dict(rec)], {"target_format": "sft", "strict": False})
    assert out2 == [rec]
    assert stats2.modified["unconvertible_passthrough"] == 1
    assert not stats2.dropped


def test_unknown_target_format_raises():
    with pytest.raises(ValueError, match="unknown target_format"):
        run_format([], {"target_format": "nope"})


def test_gemma4_builtin_template_exact():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    text = render_chat(msgs, family="gemma4")
    assert text == (
        "<start_of_turn>user\nSYS\n\nQ<end_of_turn>\n"
        "<start_of_turn>model\nA<end_of_turn>\n"
    )


def test_gemma4_generation_prompt_when_ending_on_user():
    msgs = [{"role": "user", "content": "Q"}]
    text = render_chat(msgs, family="gemma4")
    assert text.endswith("<start_of_turn>model\n")


def test_qwen35_chatml_template_exact():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    expected = (
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nQ<|im_end|>\n"
        "<|im_start|>assistant\nA<|im_end|>\n"
    )
    assert render_chat(msgs, family="qwen3.5") == expected
    assert render_chat(msgs, family="chatml") == expected


def test_generic_fallback_marks_template_fallback():
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    assert render_chat(msgs, family=None) == "USER: Q\nASSISTANT: A\n"
    out, stats = run_format([{"id": "g1", "messages": msgs, "meta": {}}], {"target_format": "sft"})
    assert stats.extra["template_fallback"] is True
    assert stats.extra["template_source"] == "generic"
    assert out[0]["text"] == "USER: Q\nASSISTANT: A\n"


class _FakeChatTokenizer:
    chat_template = "<fake>"

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return "RENDERED-BY-TOKENIZER"


def test_tokenizer_object_with_chat_template_wins():
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    out, stats = run_format(
        [{"id": "t1", "messages": msgs, "meta": {}}],
        {"target_format": "sft", "tokenizer": _FakeChatTokenizer(), "chat_template_family": "chatml"},
    )
    assert out[0]["text"] == "RENDERED-BY-TOKENIZER"
    assert stats.extra["template_source"] == "tokenizer"
    assert "template_fallback" not in stats.extra


def test_tokenizer_load_failure_falls_back_and_is_recorded():
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    out, stats = run_format(
        [{"id": "t2", "messages": msgs, "meta": {}}],
        {"target_format": "sft", "tokenizer": "/definitely/not/a/real/tokenizer-path-zz", "chat_template_family": "chatml"},
    )
    assert "tokenizer_error" in stats.extra  # never silently successful
    assert out[0]["text"].startswith("<|im_start|>user\n")  # builtin template used
