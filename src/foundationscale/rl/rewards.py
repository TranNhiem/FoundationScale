"""Verifiable rewards: scoring functions that abstain rather than guess.

The RL trainer scores every generated completion against a verifiable
signal. Where no signal exists -- a non-MCQ record, an unparseable
response -- the reward function returns ``None`` and the trainer drops the
row. Abstention propagates the estimators' existing idiom: ``used <
offered`` is reported in the step report, so the drop stays visible in the
denominator rather than reading as a zero.

WHAT IS CLAIMED: a parseable response with a verifiable gold is scored
correctly (1.0 on a match, 0.0 otherwise); every other case abstains.

WHAT IS NOT CLAIMED: no partial credit for reasoning quality, no judgement
of open-ended answers, no claim that an abstaining row taught the model
anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ("MCQLetterReward",)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LETTER_RE = re.compile(r"[A-Z]")


def _parse_letter(response: str) -> str | None:
    """Parse the single answer letter from a completion, or abstain.

    The think block -- if any survived into the generated text -- is
    stripped before scanning, mirroring ``extract_mcq_gold``: reasoning
    rehearses candidate letters and must not count. Zero or multiple
    candidate letters is an AMBIGUITY and abstains.
    """
    stripped = _THINK_BLOCK_RE.sub(" ", response)
    candidates = tuple(_LETTER_RE.findall(stripped.upper()))
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) != 1:
        return None
    return unique[0]


@dataclass(frozen=True, slots=True)
class MCQLetterReward:
    """Verifiable reward: ``correct`` iff the generated letter equals the gold.

    Scoring is exact-letter agreement after stripping any think block from
    the response. The reward is binary by declaration -- ``correct`` for a
    match, ``incorrect`` otherwise -- both defaulted to the canonical
    ``1.0``/``0.0`` but overridable, since the values are the caller's
    stated reward scale, not this module's to hard-code.

    ``score`` returns ``None`` -- an ABSTENTION -- when:

    - ``gold is None``: the record carries no verifiable answer (open-ended
      or ambiguous-gold rows), so there is nothing to score against; or
    - the response yields no unambiguous letter.

    An unparseable response is UNMEASURED, not wrong: returning
    ``incorrect`` for a completion the parser could not read would punish
    the model for the parser's limitation while reporting a clean accuracy
    denominator. Abstention drops the row; the trainer reports
    ``used < offered`` and the absence stays visible.
    """

    correct: float = 1.0
    incorrect: float = 0.0

    def score(self, *, response: str, gold: str | None) -> float | None:
        """Score one completion; ``None`` means abstain, never ``0.0``."""
        if gold is None:
            return None
        letter = _parse_letter(response)
        if letter is None:
            return None
        if letter == gold.upper():
            return self.correct
        return self.incorrect
