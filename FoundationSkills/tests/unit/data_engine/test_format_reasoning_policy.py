"""A reasoning trace the chat template drops: dropped by default, or the user's explicit policy."""

from __future__ import annotations

import pytest
from foundationskills.skills.data_engine.ops.base import OpStats
from foundationskills.skills.data_engine.ops.format import _format_records


class _NoThinkTok:
    """Measured on Gemma-3: the template renders content and ignores reasoning_content."""

    chat_template = "gemma3"

    def apply_chat_template(self, messages, tokenize=False, **_kw):
        return "".join(f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n" for m in messages)


MSGS = [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "A", "reasoning_content": "TRACE"},
]


def run(policy: str | None) -> tuple[list[dict], OpStats]:
    cfg = {"target_format": "sft", "tokenizer": _NoThinkTok()}
    if policy:
        cfg["reasoning_unsupported"] = policy
    st = OpStats("format")
    return list(_format_records([{"id": "1", "messages": MSGS}], cfg, st)), st


def test_default_drops_the_record() -> None:
    out, st = run(None)
    assert out == [] and st.dropped["reasoning_lost_in_template"] == 1
    assert "reasoning_unsupported" not in st.extra


def test_inline_think_keeps_the_trace_in_the_answer() -> None:
    out, st = run("inline_think")
    assert "<think>\nTRACE\n</think>\n\nA" in out[0]["text"]
    assert out[0]["meta"]["reasoning_render"] == "inline_think"
    assert not any("reasoning_content" in m for m in out[0]["messages"])
    assert st.extra["reasoning_unsupported"] == "inline_think"
    assert st.extra["reasoning_render"] == {"inline_think": 1}


def test_drop_trace_trains_on_the_answer_only_and_says_so() -> None:
    out, st = run("drop_trace")
    assert "TRACE" not in out[0]["text"] and "A" in out[0]["text"]
    assert st.extra["reasoning_render"] == {"trace_dropped": 1}
    assert st.extra["reasoning_unsupported"] == "drop_trace"


def test_an_unknown_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="reasoning_unsupported"):
        run("guess")
