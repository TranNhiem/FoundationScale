"""Unit tests for the conversation-column declaration wiring in train/loop.py.

These cover the pure helper functions _train calls BEFORE the model or the
corpus is touched -- _conversations_declaration_conflict (mirrors
_audio_declaration_conflict's shape and test style exactly, see
test_audio_loop_wiring.py), and the two env-value resolvers for
FOUNDATIONSCALE_TRAIN_OVERLONG / FOUNDATIONSCALE_TRAIN_PAD_TO_MAX_LENGTH.
The collator itself (train/conversation.py) is covered by
tests/train/test_conversation_collator.py, which this file does not touch or
duplicate; the end-to-end wiring (declared column reaches the collator, the
image column survives remove_unused_columns, stats land in the manifest) is
proven on real hardware by the GPU proof runs (a)/(b).
"""

from __future__ import annotations

from foundationscale.train.loop import (
    _conversation_overlong_policy_or_refusal,
    _conversation_pad_to_max_length_or_refusal,
    _conversations_declaration_conflict,
)

# ---------------------------------------------------------------------------
# _conversations_declaration_conflict
# ---------------------------------------------------------------------------


def test_undeclared_conversations_never_conflicts() -> None:
    assert _conversations_declaration_conflict(conversations_column=None, audio_column=None) is None
    assert (
        _conversations_declaration_conflict(conversations_column=None, audio_column="audio") is None
    )


def test_conversations_alone_is_clean() -> None:
    assert (
        _conversations_declaration_conflict(conversations_column="conversations", audio_column=None)
        is None
    )


def test_conversations_and_audio_names_both_columns() -> None:
    refusal = _conversations_declaration_conflict(
        conversations_column="conversations", audio_column="waveforms"
    )
    assert refusal is not None
    assert "conversations" in refusal
    assert "waveforms" in refusal


# ---------------------------------------------------------------------------
# _conversation_overlong_policy_or_refusal -- REQUIRED, no default
# ---------------------------------------------------------------------------


def test_overlong_absent_is_a_refusal_not_a_default() -> None:
    policy, refusal = _conversation_overlong_policy_or_refusal(None)
    assert policy is None
    assert refusal is not None
    assert "FOUNDATIONSCALE_TRAIN_OVERLONG" in refusal


def test_overlong_drop_and_refuse_are_both_accepted() -> None:
    assert _conversation_overlong_policy_or_refusal("drop") == ("drop", None)
    assert _conversation_overlong_policy_or_refusal("refuse") == ("refuse", None)


def test_overlong_unknown_value_is_refused_naming_it() -> None:
    policy, refusal = _conversation_overlong_policy_or_refusal("truncate")
    assert policy is None
    assert refusal is not None
    assert "truncate" in refusal


# ---------------------------------------------------------------------------
# _conversation_pad_to_max_length_or_refusal -- optional, default False
# ---------------------------------------------------------------------------


def test_pad_to_max_length_absent_defaults_false_with_no_refusal() -> None:
    assert _conversation_pad_to_max_length_or_refusal(None) == (False, None)


def test_pad_to_max_length_true_and_false_are_both_accepted() -> None:
    assert _conversation_pad_to_max_length_or_refusal("true") == (True, None)
    assert _conversation_pad_to_max_length_or_refusal("false") == (False, None)


def test_pad_to_max_length_unknown_value_is_refused_naming_it() -> None:
    value, refusal = _conversation_pad_to_max_length_or_refusal("yes")
    assert value is False
    assert refusal is not None
    assert "yes" in refusal
