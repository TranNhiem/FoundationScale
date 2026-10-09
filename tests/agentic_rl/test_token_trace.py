"""Tests for the ported token trace: incremental appends and rendering deltas.

WHAT IS CLAIMED: the prompt installs once and then observations and generated
turns extend one flat sequence with a mask (0 injected, 1 sampled) and a
logprob alignment whose absence is None; delta selection drops sampled
assistant turns and advances the cursor; every upstream pad/truncate/clip/guess
is refused with the field named and both counts stated; budget exhaustion is the
recorded control flow, not a refusal.

WHAT IS NOT CLAIMED: no tokenizer, chat template or engine is exercised -- every
token id is synthetic, and no rendering happens here at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl import token_trace
from foundationscale.agentic_rl.token_trace import (
    ResponseBudgetExhausted,
    TokenTrace,
    TokenTraceRefusal,
    select_delta_messages,
)


def _installed(response_length: int = 8) -> TokenTrace:
    trace = TokenTrace(response_length=response_length)
    trace.append_prompt([1, 2, 3])
    return trace


# ---------------------------------------------------------------------------
# select_delta_messages (rendering deltas)
# ---------------------------------------------------------------------------


def test_select_delta_messages_returns_the_environment_turns_and_the_new_cursor() -> None:
    messages: list[Any] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "calling a tool"},
        {"role": "tool", "content": "42"},
        {"role": "tool", "content": "43"},
        {"role": "user", "content": "and now?"},
    ]

    delta, cursor = select_delta_messages(messages, 0)

    assert [m["content"] for m in delta] == ["q", "42", "43", "and now?"]
    assert cursor == 5

    late, cursor = select_delta_messages(messages, cursor)
    assert late == []
    assert cursor == 5

    partial, _ = select_delta_messages(messages, 3)
    assert [m["content"] for m in partial] == ["43", "and now?"]


def test_select_delta_messages_refuses_a_cursor_outside_the_message_window() -> None:
    messages: list[Any] = [{"role": "tool", "content": "42"}]

    with pytest.raises(TokenTraceRefusal) as negative:
        select_delta_messages(messages, -1)
    assert "field 'cursor'" in str(negative.value)
    assert "1 message(s)" in str(negative.value)

    with pytest.raises(TokenTraceRefusal) as past_end:
        select_delta_messages(messages, 2)
    assert "field 'cursor'" in str(past_end.value)
    assert "is 2 with 1 message(s)" in str(past_end.value)


def test_select_delta_messages_refuses_a_cursor_that_is_not_a_real_int() -> None:
    with pytest.raises(TokenTraceRefusal, match="field 'cursor'"):
        select_delta_messages([], True)
    with pytest.raises(TokenTraceRefusal, match="field 'cursor'"):
        select_delta_messages([], "0")


def test_select_delta_messages_refuses_a_message_without_a_role() -> None:
    for entry in (
        {"content": "42"},
        {"role": "", "content": "42"},
        {"role": 7, "content": "42"},
    ):
        with pytest.raises(TokenTraceRefusal, match="role"):
            select_delta_messages([entry], 0)


def test_select_delta_messages_refuses_a_message_that_is_not_a_mapping() -> None:
    with pytest.raises(TokenTraceRefusal, match="message 0"):
        select_delta_messages(["42"], 0)


# ---------------------------------------------------------------------------
# construction and incremental appends
# ---------------------------------------------------------------------------


def test_token_trace_refuses_a_response_length_that_is_not_a_real_positive_int() -> None:
    with pytest.raises(TokenTraceRefusal, match="field 'response_length'"):
        TokenTrace(response_length=True)
    with pytest.raises(TokenTraceRefusal, match="field 'response_length'") as excinfo:
        TokenTrace(response_length=0)
    assert ">= 1" in str(excinfo.value)


def test_token_trace_refuses_prefilled_buffers() -> None:
    with pytest.raises(TokenTraceRefusal) as excinfo:
        TokenTrace(response_length=8, token_ids=[1], prompt_len=1)
    message = str(excinfo.value)
    assert "token_ids=1" in message
    assert "prompt_len=1" in message

    with pytest.raises(TokenTraceRefusal, match="pre-filled"):
        TokenTrace(response_length=8, response_mask=[1], response_logprobs=[None])


def test_append_prompt_installs_the_prompt_once() -> None:
    trace = TokenTrace(response_length=8)

    trace.append_prompt([1, 2, 3])

    assert trace.prompt_len == 3
    assert trace.prompt_ids == [1, 2, 3]
    assert trace.response_ids == []

    with pytest.raises(TokenTraceRefusal, match="already holds a prompt"):
        trace.append_prompt([4])


def test_append_refuses_empty_or_malformed_token_ids() -> None:
    for bad in ([], [1, True], [-1], "ab", [1, 1.5], 4):
        with pytest.raises(TokenTraceRefusal, match="token_ids"):
            _installed().append_observation(bad)

    for bad in ([], [1, True], [-1], "ab"):
        with pytest.raises(TokenTraceRefusal, match="token_ids"):
            _installed().append_generated(bad)


def test_appends_refuse_because_the_prompt_is_not_installed_yet() -> None:
    trace = TokenTrace(response_length=8)

    with pytest.raises(TokenTraceRefusal, match="append_observation()"):
        trace.append_observation([4])
    with pytest.raises(TokenTraceRefusal, match="append_generated()"):
        trace.append_generated([4], [None])


def test_append_observation_is_masked_out() -> None:
    trace = _installed()

    trace.append_observation([4, 5])

    assert trace.token_ids == [1, 2, 3, 4, 5]
    assert trace.response_ids == [4, 5]
    assert trace.response_mask == [0, 0]
    assert trace.response_logprobs == [None, None]


def test_append_generated_is_masked_in_and_records_logprobs() -> None:
    trace = _installed()

    trace.append_generated([6, 7], (-0.5, -0.25))

    assert trace.token_ids == [1, 2, 3, 6, 7]
    assert trace.response_mask == [1, 1]
    assert trace.response_logprobs == [-0.5, -0.25]


def test_logprob_length_mismatch_is_refused_never_padded_or_truncated() -> None:
    trace = _installed()

    with pytest.raises(TokenTraceRefusal) as short:
        trace.append_generated([6, 7], [-0.5])
    assert "carries 1 entries for 2 token(s)" in str(short.value)

    with pytest.raises(TokenTraceRefusal) as long:
        trace.append_generated([6, 7], [-0.5, -0.25, -0.1])
    assert "carries 3 entries for 2 token(s)" in str(long.value)

    assert trace.token_ids == [1, 2, 3]
    assert trace.response_mask == []


def test_logprob_values_must_be_finite_reals() -> None:
    trace = _installed()

    for bad in (["x", -0.5], [float("nan"), -0.5], "ab"):
        with pytest.raises(TokenTraceRefusal, match="logprobs"):
            trace.append_generated([6, 7], bad)


def test_absent_logprobs_are_recorded_as_none_not_zero() -> None:
    trace = _installed()

    trace.append_generated([6], [-0.5])
    trace.append_observation([7])
    trace.append_generated([8])

    _, _, mask, logprobs = trace.finalize()

    assert mask == [1, 0, 1]
    assert logprobs == [-0.5, None, None]
    assert logprobs is not None and logprobs[1] is None


# ---------------------------------------------------------------------------
# budget and finalize
# ---------------------------------------------------------------------------


def test_remaining_budget_and_fits_track_the_response_region() -> None:
    trace = _installed(response_length=5)

    assert trace.remaining_budget() == 5
    assert trace.fits(4) is True
    assert trace.fits(5) is False

    trace.append_observation([4, 5, 6])

    assert trace.remaining_budget() == 2
    assert trace.fits(2) is False
    assert trace.fits(1) is True

    trace.append_generated([7, 8, 9, 10])
    assert trace.remaining_budget() == 0


def test_fits_refuses_a_non_int_or_negative_incoming_count() -> None:
    trace = _installed()

    with pytest.raises(TokenTraceRefusal, match="field 'incoming'"):
        trace.fits(True)
    with pytest.raises(TokenTraceRefusal, match="field 'incoming'"):
        trace.fits(-1)


def test_check_budget_stops_the_rollout_and_records_truncated() -> None:
    trace = _installed(response_length=5)

    trace.check_budget(4)
    assert trace.truncated is False

    with pytest.raises(ResponseBudgetExhausted) as stop:
        trace.check_budget(5)
    message = str(stop.value)
    assert "0 used + 5 incoming" in message
    assert "response_length 5" in message
    assert trace.truncated is True

    with pytest.raises(TokenTraceRefusal, match="field 'incoming'"):
        trace.check_budget(True)


def test_finalize_returns_the_prompt_the_response_and_masks() -> None:
    trace = _installed(response_length=6)
    trace.append_observation([4])
    trace.append_generated([5, 6], [-0.5, -0.25])

    prompt, response, mask, logprobs = trace.finalize()

    assert prompt == [1, 2, 3]
    assert response == [4, 5, 6]
    assert mask == [0, 1, 1]
    assert logprobs == [None, -0.5, -0.25]


def test_finalize_logprobs_are_none_until_the_engine_reports_some() -> None:
    trace = _installed()
    trace.append_generated([4, 5])

    _, _, mask, logprobs = trace.finalize()

    assert mask == [1, 1]
    assert logprobs is None


def test_finalize_accepts_a_response_that_fills_the_budget_exactly() -> None:
    trace = _installed(response_length=3)
    trace.append_generated([4, 5, 6])

    _, response, mask, _ = trace.finalize()

    assert response == [4, 5, 6]
    assert mask == [1, 1, 1]


def test_finalize_refuses_a_trace_that_never_installed_a_prompt() -> None:
    trace = TokenTrace(response_length=8)

    with pytest.raises(TokenTraceRefusal, match="finalize()"):
        trace.finalize()


def test_finalize_refuses_a_response_that_would_be_clipped() -> None:
    trace = _installed(response_length=3)
    trace.append_generated([4, 5, 6, 7])

    with pytest.raises(TokenTraceRefusal) as excinfo:
        trace.finalize()

    message = str(excinfo.value)
    assert "holds 4 token(s)" in message
    assert "'response_length' is 3" in message
    assert "does not clip" in message


def test_finalize_refuses_hand_grown_response_mask_desync() -> None:
    trace = _installed()
    trace.append_generated([4])

    trace.response_mask.append(1)

    with pytest.raises(TokenTraceRefusal) as excinfo:
        trace.finalize()
    message = str(excinfo.value)
    assert "'response_mask' has 2 entries" in message
    assert "'response_ids' has 1" in message


def test_finalize_refuses_hand_grown_logprob_desync() -> None:
    trace = _installed()
    trace.append_generated([4], [-0.5])

    trace.response_logprobs.append(None)

    with pytest.raises(TokenTraceRefusal) as excinfo:
        trace.finalize()
    message = str(excinfo.value)
    assert "'response_logprobs' has 2 entries" in message
    assert "'response_mask' has 1" in message


# ---------------------------------------------------------------------------
# port hygiene
# ---------------------------------------------------------------------------


def test_only_the_required_upstream_header_carries_the_brand_token() -> None:
    needle = "mi" + "mo"
    assert token_trace.__file__ is not None
    source = Path(token_trace.__file__).read_text(encoding="utf-8")
    body_start = source.index('"""')

    assert needle in source[:body_start].lower()
    assert needle not in source[body_start:].lower()
