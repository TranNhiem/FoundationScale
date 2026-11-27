"""Deterministic, verifiable rewards for science question-answering tasks.

The task metadata selects one of the closed set of checkers in ``CHECKERS``.
Every call returns a :class:`ScienceScore`: malformed tasks, missing answers,
and invalid predictions receive zero rather than an exception or an abstention.
Successful and partially successful scores are always finite floats in [0, 1].

Task metadata values are strings (``EpisodeTask`` stores no other type), so list fields
(``choices``, ``answers``) arrive JSON-encoded (``'["yes", "no", "maybe"]'``) and number fields
(``threshold``, ``rel_tol``, ``abs_tol``) as decimal strings; :func:`decode_metadata` turns them
back into their types once, before any checker reads them. Native values are accepted as-is.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import string
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from foundationscale.agentic_rl.rewards.math import extract_final_answer

CHECKERS: tuple[str, ...] = ("label", "numeric", "span_f1")
MAX_ANSWER_CHARS = 2000

_OPTION_RE = re.compile(
    r"^\s*(?:option\s+)?\(?([A-Za-z])(?:\)|[.)])?\s*$",
    re.IGNORECASE,
)
# "(B) Paris" / "B. Paris": a letter plus option text -- accepted only when both name one option.
_LETTER_TEXT_RE = re.compile(r"^\s*\(?([A-Za-z])[.):]\s+(\S.*)$", re.DOTALL)
_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_SURROUNDING_PAIRS = {
    "'": "'",
    '"': '"',
    "(": ")",
    "[": "]",
    "{": "}",
}
_ARTICLES = frozenset({"a", "an", "the"})
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class ScienceScore:
    value: float
    note: str


def normalize_label(s: str) -> str:
    """Normalize a label without discarding meaningful internal punctuation."""
    value = s.casefold()
    while True:
        stripped = value.strip()
        if len(stripped) >= 2:
            closing = _SURROUNDING_PAIRS.get(stripped[0])
            if closing is not None and stripped[-1] == closing:
                stripped = stripped[1:-1].strip()
        stripped = stripped.rstrip(".").strip()
        if stripped == value:
            break
        value = stripped
    return " ".join(value.split())


def _normalize_span(s: str) -> list[str]:
    without_punctuation = s.lower().translate(_PUNCTUATION_TABLE)
    return [token for token in without_punctuation.split() if token not in _ARTICLES]


def token_f1(pred: str, truth: str) -> float:
    """Return SQuAD-style token F1 after normalization."""
    pred_tokens = _normalize_span(pred)
    truth_tokens = _normalize_span(truth)
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2.0 * precision * recall / (precision + recall)


_LIST_FIELDS = ("answers", "choices")
_NUMBER_FIELDS = ("abs_tol", "rel_tol", "threshold")


def decode_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """``metadata`` with JSON-encoded list fields and decimal-string number fields decoded.

    A field that does not decode keeps its raw string, so the checker refuses it by name.
    """
    decoded = dict(metadata)
    for key in _LIST_FIELDS:
        value = decoded.get(key)
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                decoded[key] = json.loads(value)
    for key in _NUMBER_FIELDS:
        value = decoded.get(key)
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                decoded[key] = float(value.strip())
    return decoded


def _bad_task(field: str, reason: str) -> ScienceScore:
    return ScienceScore(0.0, f"bad_task:{field}: {reason}")


def _as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_prediction_number(text: str) -> float | None:
    value = text.strip().strip("$").strip()
    value = _COMMA_RE.sub("", value)
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _score_label(pred: str, metadata: Mapping[str, Any]) -> ScienceScore:
    answer = metadata.get("answer")
    if not isinstance(answer, str):
        return _bad_task("answer", "expected a string")

    if "choices" not in metadata:
        correct = normalize_label(pred) == normalize_label(answer)
        return ScienceScore(float(correct), "correct" if correct else "wrong")

    raw_choices = metadata.get("choices")
    if not isinstance(raw_choices, list) or not all(
        isinstance(choice, str) for choice in raw_choices
    ):
        return _bad_task("choices", "expected a list of strings")
    choices: list[str] = list(raw_choices)
    if answer not in choices:
        return _bad_task("answer", "must exactly equal one of the choices")

    option_match = _OPTION_RE.fullmatch(pred)
    if option_match is not None:
        index = ord(option_match.group(1).upper()) - ord("A")
        if index >= len(choices):
            return ScienceScore(0.0, "invalid_choice")
        correct = choices[index] == answer
        return ScienceScore(float(correct), "correct" if correct else "wrong")

    letter_text = _LETTER_TEXT_RE.match(pred)
    if letter_text is not None:
        index = ord(letter_text.group(1).upper()) - ord("A")
        named = normalize_label(letter_text.group(2))
        if index < len(choices) and normalize_label(choices[index]) == named:
            correct = choices[index] == answer
            return ScienceScore(float(correct), "correct" if correct else "wrong")

    normalized_pred = normalize_label(pred)
    if not any(normalized_pred == normalize_label(choice) for choice in choices):
        return ScienceScore(0.0, "invalid_choice")
    correct = normalized_pred == normalize_label(answer)
    return ScienceScore(float(correct), "correct" if correct else "wrong")


def _score_span_f1(pred: str, metadata: Mapping[str, Any]) -> ScienceScore:
    raw_answers = metadata.get("answers")
    if (
        not isinstance(raw_answers, list)
        or not raw_answers
        or not all(isinstance(item, str) and item for item in raw_answers)
    ):
        return _bad_task("answers", "expected a non-empty list of non-empty strings")
    answers: list[str] = list(raw_answers)

    threshold: float | None = None
    if "threshold" in metadata:
        threshold = _as_finite_number(metadata.get("threshold"))
        if threshold is None or not 0.0 < threshold <= 1.0:
            return _bad_task("threshold", "expected a finite number in (0, 1]")

    best_f1 = max(token_f1(pred, answer) for answer in answers)
    value = float(best_f1 >= threshold) if threshold is not None else round(best_f1, 6)

    if value == 1.0:
        note = "correct"
    elif value == 0.0:
        note = "wrong"
    else:
        note = "partial"
    return ScienceScore(value, note)


def _score_numeric(pred: str, metadata: Mapping[str, Any]) -> ScienceScore:
    raw_answer = metadata.get("answer")
    if isinstance(raw_answer, str):
        try:
            reference: float | None = float(raw_answer.strip())
        except ValueError:
            reference = None
        if reference is not None and not math.isfinite(reference):
            reference = None
    else:
        reference = _as_finite_number(raw_answer)
    if reference is None:
        return _bad_task("answer", "expected a finite number or numeric string")

    rel_tol = _as_finite_number(metadata.get("rel_tol", 1e-6))
    if rel_tol is None or rel_tol < 0.0:
        return _bad_task("rel_tol", "expected a finite number >= 0")
    abs_tol = _as_finite_number(metadata.get("abs_tol", 0.0))
    if abs_tol is None or abs_tol < 0.0:
        return _bad_task("abs_tol", "expected a finite number >= 0")

    prediction = _parse_prediction_number(pred)
    if prediction is None:
        return ScienceScore(0.0, "unparseable")
    correct = math.isclose(
        prediction,
        reference,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    return ScienceScore(float(correct), "correct" if correct else "wrong")


@dataclass(frozen=True)
class ScienceAnswerReward:
    """Score extracted final answers against validated science task metadata."""

    def score(
        self,
        text: str,
        metadata: Mapping[str, Any] | None,
        *,
        submitted: bool = False,
    ) -> ScienceScore:
        try:
            return self._score(text, metadata, submitted=submitted)
        except Exception as exc:  # noqa: BLE001 -- score() never raises (module docstring)
            return _bad_task("metadata", f"could not score ({type(exc).__name__})")

    def _score(
        self,
        text: str,
        metadata: Mapping[str, Any] | None,
        *,
        submitted: bool,
    ) -> ScienceScore:
        if not isinstance(metadata, Mapping):
            return _bad_task("metadata", "expected a mapping")
        metadata = decode_metadata(metadata)
        checker = metadata.get("checker")
        if not isinstance(checker, str) or checker not in CHECKERS:
            legal = ", ".join(CHECKERS)
            return _bad_task("checker", f"expected one of: {legal}")

        answer = extract_final_answer(text)
        if answer is None and submitted:
            bare = text.strip()
            if bare and len(bare.splitlines()) == 1:
                answer = bare
        if answer is None:
            return ScienceScore(0.0, "no_answer")
        if len(answer) > MAX_ANSWER_CHARS:
            return ScienceScore(0.0, "too_long")

        if checker == "label":
            return _score_label(answer, metadata)
        if checker == "numeric":
            return _score_numeric(answer, metadata)
        return _score_span_f1(answer, metadata)
