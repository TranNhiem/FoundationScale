"""The math-answer reward: extraction, normalization, equivalence and the scorer's notes."""

from __future__ import annotations

import pytest

from foundationscale.agentic_rl.rewards.math import (
    MAX_ANSWER_CHARS,
    MathAnswerReward,
    answers_equivalent,
    extract_final_answer,
    normalize_answer,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"so the answer is \boxed{42}.", "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"first \boxed{1} then corrected: \boxed{3}", "3"),
        (r"\fbox{x+1}", "x+1"),
        ("Working...\nFinal answer: 17", "17"),
        ("answer: 5 apples", "5 apples"),
        ("no answer here", None),
        (r"\boxed{unbalanced", None),
    ],
)
def test_extract_final_answer(text: str, expected: str | None) -> None:
    assert extract_final_answer(text) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$42$", "42"),
        (r"\left( 1, 2 \right)", "(1,2)"),
        (r"\dfrac{3}{4}", r"\frac{3}{4}"),
        (r"\text{5}", "5"),
        (r"90^\circ", "90"),
        (r"50\%", "50"),
        ("1,000", "1000"),
        ("12.", "12"),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    ("pred", "truth"),
    [
        ("0.5", r"\frac{1}{2}"),
        (r"\dfrac{3}{4}", "0.75"),
        (r"\sqrt{8}", r"2*\sqrt{2}"),
        (r"\frac{\pi}{2}", r"\frac{\pi}{2}"),
        ("1,000", "1000"),
        (r"50\%", "50"),
        ("(1, 2)", "(1,2)"),
        (r"\left(\frac{1}{2}, 3\right)", "(0.5, 3)"),
        ("x^2 + 2*x + 1", "(x+1)^2"),
        (r"\sqrt[3]{27}", "3"),
    ],
)
def test_equivalent(pred: str, truth: str) -> None:
    assert answers_equivalent(pred, truth)


@pytest.mark.parametrize(
    ("pred", "truth"),
    [
        ("0.51", r"\frac{1}{2}"),
        ("(1, 2)", "[1, 2]"),
        ("(1, 2, 3)", "(1, 2)"),
        ("x+1", "x+2"),
        ("", "1"),
        ("banana", "7"),
        ("1" * (MAX_ANSWER_CHARS + 1), "2"),
    ],
)
def test_not_equivalent(pred: str, truth: str) -> None:
    assert not answers_equivalent(pred, truth)


def test_garbage_never_raises() -> None:
    for junk in [r"\frac{", "}}}", "((((", r"\sqrt[", "1/0", "x**x**x**x**999"]:
        answers_equivalent(junk, "1")


@pytest.mark.parametrize(
    ("text", "reference", "value", "note"),
    [
        (r"\boxed{\frac{1}{2}}", "0.5", 1.0, "correct"),
        (r"\boxed{3}", "4", 0.0, "wrong"),
        ("I am not sure.", "4", 0.0, "no_answer"),
        (r"\boxed{3}", None, 0.0, "no_reference"),
        (r"\boxed{3}", "  ", 0.0, "no_reference"),
    ],
)
def test_scorer_notes(text: str, reference: str | None, value: float, note: str) -> None:
    score = MathAnswerReward().score(text, reference)
    assert (score.value, score.note) == (value, note)


def test_a_bare_submitted_answer_counts_only_when_submitted() -> None:
    reward = MathAnswerReward()
    assert reward.score("42", "42", submitted=True).note == "correct"
    assert reward.score("42", "42").note == "no_answer"  # free text needs \\boxed{} or "Answer:"
    assert reward.score("line one\nline two", "42", submitted=True).note == "no_answer"
    assert reward.score("9" * (MAX_ANSWER_CHARS + 1), "42", submitted=True).note == "no_answer"
