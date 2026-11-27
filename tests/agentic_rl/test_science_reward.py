import math

import pytest

from foundationscale.agentic_rl.rewards.science import (
    MAX_ANSWER_CHARS,
    ScienceAnswerReward,
)

REWARD = ScienceAnswerReward()


def test_label_exact_casefold_and_normalization() -> None:
    metadata = {"checker": "label", "answer": " Paris. "}
    assert REWARD.score(r"\boxed{paris}", metadata).value == 1.0
    assert REWARD.score(r"\boxed{London}", metadata).note == "wrong"


def test_mcq_letter_bracketed_letter_and_text() -> None:
    metadata = {"checker": "label", "answer": "blue", "choices": ["red", "blue"]}
    for prediction in (r"\boxed{B}", r"\boxed{(B)}", r"\boxed{ BLUE }"):
        result = REWARD.score(prediction, metadata)
        assert result.value == 1.0
        assert result.note == "correct"


def test_mcq_invalid_predictions_and_bad_answer() -> None:
    metadata = {"checker": "label", "answer": "blue", "choices": ["red", "blue"]}
    assert REWARD.score(r"\boxed{D}", metadata).note == "invalid_choice"
    assert REWARD.score(r"\boxed{green}", metadata).note == "invalid_choice"
    assert REWARD.score(r"\boxed{A}", metadata).note == "wrong"
    bad = {"checker": "label", "answer": "green", "choices": ["red", "blue"]}
    assert REWARD.score(r"\boxed{green}", bad).note.startswith("bad_task:answer:")


def test_span_f1_continuous_and_threshold() -> None:
    metadata = {"checker": "span_f1", "answers": ["quick cat"]}
    result = REWARD.score(r"\boxed{the quick brown cat}", metadata)
    assert result.value == 0.8
    assert result.note == "partial"
    assert REWARD.score(r"\boxed{the quick brown cat}", metadata | {"threshold": 0.8}).value == 1.0
    assert REWARD.score(r"\boxed{the quick brown cat}", metadata | {"threshold": 0.81}).value == 0.0


def test_span_f1_articles_punctuation_and_max_over_answers() -> None:
    metadata = {
        "checker": "span_f1",
        "answers": ["a dog", "quick brown cat"],
    }
    result = REWARD.score(r"\boxed{The quick, brown cat!}", metadata)
    assert result.value == 1.0
    assert result.note == "correct"


def test_numeric_tolerances_commas_and_unparseable() -> None:
    metadata = {
        "checker": "numeric",
        "answer": 1000,
        "rel_tol": 0,
        "abs_tol": 0.1,
    }
    assert REWARD.score(r"\boxed{$1,000.05}", metadata).value == 1.0
    assert REWARD.score(r"\boxed{1000.2}", metadata).value == 0.0
    assert REWARD.score(r"\boxed{nan}", metadata).note == "unparseable"
    assert REWARD.score(r"\boxed{10 meters}", metadata).note == "unparseable"


def test_numeric_bool_answer_is_bad_task() -> None:
    result = REWARD.score(r"\boxed{1}", {"checker": "numeric", "answer": True})
    assert result.value == 0.0
    assert result.note.startswith("bad_task:answer:")


@pytest.mark.parametrize("metadata", [{}, {"checker": "unknown"}, {"checker": 3}])
def test_missing_or_unknown_checker_names_legal_set(metadata: object) -> None:
    result = REWARD.score(r"\boxed{x}", metadata)  # type: ignore[arg-type]
    assert result.value == 0.0
    assert result.note.startswith("bad_task:checker:")
    for checker in ("label", "numeric", "span_f1"):
        assert checker in result.note


def test_no_answer_and_submitted_bare_line() -> None:
    metadata = {"checker": "label", "answer": "Paris"}
    assert REWARD.score("Paris", metadata).note == "no_answer"
    assert REWARD.score("  Paris  ", metadata, submitted=True).value == 1.0
    assert REWARD.score("one\ntwo", metadata, submitted=True).note == "no_answer"


def test_too_long() -> None:
    metadata = {"checker": "label", "answer": "x"}
    text = r"\boxed{" + "x" * (MAX_ANSWER_CHARS + 1) + "}"
    assert REWARD.score(text, metadata).note == "too_long"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {"checker": "span_f1", "answers": "str"},
        {"checker": "span_f1", "answers": []},
        {"checker": "span_f1", "answers": ["x"], "threshold": True},
        {"checker": "numeric", "answer": object()},
        {"checker": "numeric", "answer": 1, "rel_tol": False},
        {"checker": "label", "answer": "a", "choices": "ab"},
    ],
)
def test_score_never_raises_on_junk_metadata(metadata: object) -> None:
    result = REWARD.score(r"\boxed{1}", metadata)  # type: ignore[arg-type]
    assert isinstance(result.value, float)
    assert math.isfinite(result.value)
    assert 0.0 <= result.value <= 1.0
    assert result.note


def test_letter_plus_option_text_must_name_the_same_option() -> None:
    meta = {"checker": "label", "answer": "Paris", "choices": ["London", "Paris", "Rome"]}
    reward = ScienceAnswerReward()
    assert reward.score("Answer: (B) Paris", meta).value == 1.0
    assert reward.score("Answer: B. paris", meta).value == 1.0
    assert reward.score("Answer: (A) London", meta).note == "wrong"
    # A letter naming one option and text naming another is not an answer.
    assert reward.score("Answer: (A) Paris", meta).note == "invalid_choice"


def test_string_encoded_metadata_as_stored_by_episode_tasks() -> None:
    reward = ScienceAnswerReward()
    mcq = {"checker": "label", "answer": "no", "choices": '["yes", "no", "maybe"]'}
    assert reward.score("Answer: B", mcq).value == 1.0
    spans = {"checker": "span_f1", "answers": '["the BERT model"]', "threshold": "0.5"}
    assert reward.score("Answer: BERT", spans).value == 1.0
    num = {"checker": "numeric", "answer": "2.5", "rel_tol": "0.01"}
    assert reward.score("Answer: 2.52", num).value == 1.0
    broken = {"checker": "label", "answer": "no", "choices": "[not json"}
    assert reward.score("Answer: B", broken).note.startswith("bad_task:choices")
    assert reward.score("Answer: 1", {**num, "rel_tol": "x"}).note.startswith("bad_task:rel_tol")
