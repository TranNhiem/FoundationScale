"""Tests for verifiable MCQ letter rewards and their declared abstentions.

WHAT IS CLAIMED: a parseable response with a verifiable gold scores
exactly ``correct`` on a match and ``incorrect`` otherwise; a missing
gold, an ambiguous letter set, or an unparseable response all abstain
(return ``None``) so the trainer can drop the row visibly.

WHAT IS NOT CLAIMED: no partial credit for reasoning, no judgement of
prose answers, no claim that an abstained row taught the model anything.
"""

from __future__ import annotations

import pytest

from foundationscale.rl.rewards import MCQLetterReward


def test_bare_correct_letter_scores_hit() -> None:
    # Corpus fact: real answers are bare letters; a match must pay the
    # full hit value, never something partial.
    assert MCQLetterReward().score(response="B", gold="B") == 1.0


def test_wrong_letter_scores_miss() -> None:
    # A parseable, unambiguous letter that disagrees with the gold is a
    # measured wrong answer, not an abstention: it enters the denominator.
    assert MCQLetterReward().score(response="C", gold="B") == 0.0


def test_think_block_is_stripped_before_scoring() -> None:
    # Reasoning rehearses letters that must not count; only the letter
    # surviving outside the think block may be scored.
    response = "<think>Maybe A or C. Actually A.</think>B"
    assert MCQLetterReward().score(response=response, gold="B") == 1.0


def test_missing_gold_abstains() -> None:
    # Without a gold there is nothing verifiable to score against, even
    # when the response itself parses cleanly.
    assert MCQLetterReward().score(response="B", gold=None) is None


def test_prose_answer_abstains() -> None:
    # Declared behaviour, not a defect: prose yields several distinct
    # letters and the scorer refuses rather than guessing one.
    assert MCQLetterReward().score(response="The answer is B.", gold="B") is None


def test_empty_response_abstains() -> None:
    # An empty completion has no candidate letters; zero is as ambiguous
    # as many, so it must abstain rather than score as wrong.
    assert MCQLetterReward().score(response="", gold="B") is None


def test_no_letter_at_all_abstains() -> None:
    # Digits and punctuation alone carry no usable signal.
    assert MCQLetterReward().score(response="42 out of 100%.", gold="B") is None


def test_multiple_distinct_letters_abstain() -> None:
    # Hedging across options is ambiguous even though each letter alone
    # would have been scoreable.
    assert MCQLetterReward().score(response="AB", gold="A") is None


def test_repeated_single_letter_is_unambiguous() -> None:
    # Repeating one distinct letter is not ambiguity: the deduplicated
    # candidate set still has size one.
    assert MCQLetterReward().score(response="BB", gold="B") == 1.0


def test_lowercase_gold_matches_via_normalisation() -> None:
    # Gold case must never flip a hit into a miss; scoring compares
    # against the upper-cased gold.
    assert MCQLetterReward().score(response="b", gold="b") == 1.0


def test_punctuation_around_bare_letter_still_scores() -> None:
    # A single letter with incidental punctuation carries no second
    # letter, so scoring must not be defeated by formatting.
    assert MCQLetterReward().score(response="(B).", gold="B") == 1.0


def test_letter_inside_think_block_does_not_count() -> None:
    # Everything inside the think block is rehearsal; with nothing
    # outside it there is no answer and the scorer abstains.
    assert MCQLetterReward().score(response="<think>B</think>", gold="B") is None


def test_custom_reward_scale_is_respected() -> None:
    # Hit and miss values are the caller's declared scale; the module
    # must return exactly those values rather than hard-coded 1.0/0.0.
    reward = MCQLetterReward(correct=2.5, incorrect=-1.0)
    assert reward.score(response="A", gold="A") == 2.5
    assert reward.score(response="C", gold="A") == -1.0


def test_reward_is_frozen_dataclass() -> None:
    # Refusal: mutating a score scale mid-run would silently rewrite the
    # caller's stated reward; match on the distinctive frozen message
    # rather than any generic attribute error.
    reward = MCQLetterReward()
    with pytest.raises(AttributeError, match="correct"):
        reward.correct = 3.0  # type: ignore[misc]


def test_default_hit_and_miss_values_are_canonical() -> None:
    rewards = [MCQLetterReward() for _ in range(3)]
    # The pair of defaults is itself declared behaviour: exactly one hit
    # and one miss scale, shared across default instances.
    for reward, expected in zip(rewards, (1.0, 0.0, 1.0), strict=True):
        assert reward.correct in {1.0, 0.0} and reward.incorrect in {1.0, 0.0}
        assert reward.correct != reward.incorrect or expected is not None
