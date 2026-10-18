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

import math
import re
from dataclasses import dataclass
from typing import ClassVar

from foundationscale.train.speech_metrics import (
    is_runaway,
    loop_words,
    normalize_transcript,
    word_errors,
)

__all__ = ("MCQLetterReward", "WERReward")


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LETTER_RE = re.compile(r"[A-Z]")


def strip_think(text: str) -> str:
    """Remove reasoning from a completion before an answer is read from it.

    Complete ``<think>...</think>`` blocks are removed. A closing tag with no
    opening tag before it means the chat template opened the block in the
    GENERATION PROMPT, so everything up to the last such tag is reasoning.
    Measured 2026-10-18 on Qwen3.5-4B (generation prompt ends ``<think>\n``):
    completions ending in a correct bare letter carried only ``</think>``,
    the block was never stripped, the reasoning's own letters made every
    completion ambiguous, and 32/32 GRPO steps abstained. An unclosed block
    (a completion truncated mid-reasoning) is left as is.
    """
    stripped = _THINK_BLOCK_RE.sub(" ", text)
    if "</think>" in stripped:
        stripped = stripped.rsplit("</think>", 1)[1]
    return stripped


def _parse_letter(response: str, pattern: re.Pattern[str] | None = None) -> str | None:
    """Parse the single answer letter from a completion, or abstain.

    The think block -- if any survived into the generated text -- is
    stripped before scanning, mirroring ``extract_mcq_gold``: reasoning
    rehearses candidate letters and must not count. Zero or multiple
    candidate letters is an AMBIGUITY and abstains.

    ``pattern`` is the answer surface the PROMPT asked for, when the prompt
    asked for one. Without it every A--Z in the completion is a candidate, and
    that only works while the completion is a bare letter: a reply that names
    the option it chose drags the letters of its own prose in with it, so a
    correct answer reading "(C) ...並不計入GDP" offers {C,G,D,P} and abstains.
    #512, measured on the estate exam corpus: that shape is 299 of 3,414 replies --
    and a model instructed to answer in prose produces it on nearly every
    rollout, which zeroes the reward signal rather than biasing it.

    Supplying the pattern narrows the candidate set to text the prompt
    designated as the answer; it does NOT weaken the ambiguity rule. All
    captures are still deduped and two distinct letters still abstain, so a
    completion that answers twice, differently, is unreadable exactly as
    before. The pattern removes false candidates, never a real disagreement.
    """
    stripped = strip_think(response)
    if pattern is None:
        candidates: tuple[str, ...] = tuple(_LETTER_RE.findall(stripped.upper()))
    else:
        candidates = tuple(match.upper() for match in pattern.findall(stripped))
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

    ``answer_pattern`` is the caller's declaration of the answer surface the
    prompt requested -- a regex whose single capture group is the letter. It is
    ``None`` by default because inventing a surface the prompt never asked for
    would read a letter out of text the model did not offer as an answer. See
    :func:`_parse_letter` for why an undeclared surface cannot score a corpus
    whose answers are prose.
    """

    correct: float = 1.0
    incorrect: float = 0.0
    answer_pattern: str | None = None

    def score(self, *, response: str, gold: str | None) -> float | None:
        """Score one completion; ``None`` means abstain, never ``0.0``."""
        if gold is None:
            return None
        compiled = re.compile(self.answer_pattern) if self.answer_pattern else None
        letter = _parse_letter(response, compiled)
        if letter is None:
            return None
        if letter == gold.upper():
            return self.correct
        return self.incorrect


@dataclass(frozen=True, slots=True)
class WERReward:
    """Verifiable reward: one minus the transcript's word error rate -- or abstain.

    ``score`` is ``1.0 - word_errors(gold, response).rate()`` with BOTH sides
    normalized by ``speech_metrics``' own ``normalize_transcript`` -- the one
    named in ``TRANSCRIPT_NORMALIZER_ID``. That normalizer is HALF of the WER
    claim (see speech_metrics' docstring: a rate without its normalizer is not
    reproducible), so this reward is comparable only against scores published
    under the same id, and it restates no normalisation of its own.

    ``score`` returns ``None`` -- an ABSTENTION -- when the gold normalizes to
    the empty token list. ``EditCounts.rate()`` raises on that zero reference
    length (an empty reference has no words to have measured) and 0.0 there
    would turn "nothing measured" into "perfect", so the row is dropped and the
    trainer's ``used < offered`` accounting keeps the gap visible -- the same
    reason :class:`MCQLetterReward` abstains on its own unreadable cases.

    The reward is NOT clamped to [0, 1]: the rate is not clamped either (100
    deletions against one reference word is a rate of 100.0, see
    ``EditCounts.rate``), so the score can run below 0.0 and the two declared
    penalties subtract further on top. Each penalty is taken when the response
    is flagged by the speech side's OWN detector (``is_runaway``, and
    ``loop_words`` over its normalized words returning nonzero) and
    contributes exactly nothing when the flag is absent. Both default to 0.0,
    so an undeclared penalty is a stated absence -- never a guessed strength --
    and the detectors' own thresholds (``RUNAWAY_RATIO``/``RUNAWAY_SLACK_WORDS``,
    ``LOOP_MIN_RUN``) live with the detectors and travel in their manifests.

    WHAT IS NOT CLAIMED: that this is any published implementation's WER
    reward; that the penalty magnitudes are tuned; or that a flagged response
    is punished by any particular amount beyond what the caller declared.
    """

    runaway_penalty: float = 0.0
    loop_penalty: float = 0.0

    # 1 - WER is unbounded below: a runaway transcript of 256 tokens against a
    # 3-word gold reads about -85, and under group-relative advantages that one
    # row swamps every other row of its group. The WER term is floored here;
    # the declared penalties then sit on top, so the worst score is
    # WER_FLOOR - runaway_penalty - loop_penalty.
    WER_FLOOR: ClassVar[float] = -1.0

    def __post_init__(self) -> None:
        for name in ("runaway_penalty", "loop_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"WERReward.{name}={value!r}: a number >= 0")
            if not math.isfinite(value) or value < 0:
                # A negative penalty would REWARD the degeneration it names.
                raise ValueError(f"WERReward.{name}={value!r}: finite and >= 0")

    def score(self, *, response: str, gold: str) -> float | None:
        """Score one generated transcript; ``None`` means abstain, never ``0.0``.

        ``gold`` is the record's VERIFIED transcript -- on the audio leg its
        manifest refuses a blank one with its line number (asr_rollout) -- and
        a gold that normalizes to nothing abstains rather than scoring against
        nothing. An empty RESPONSE is scored, not abstained: it deletes every
        gold word, WER 1.0, reward 0.0.
        """
        if not normalize_transcript(gold):
            return None
        score = max(1.0 - word_errors(gold, response).rate(), self.WER_FLOOR)
        if is_runaway(gold, response):
            score -= self.runaway_penalty
        if loop_words(normalize_transcript(response)) > 0:
            score -= self.loop_penalty
        return score
