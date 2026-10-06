"""Speech error rates: the claim a transcript score is allowed to make.

This is the transcript side of the measured-claim contract in ``train.audio``: the
metrics a speech run may print, the normalizer printed with them, and the coverage
document for the corpus that produced them. Four rules hold here:

  a rate without its        ``normalize_transcript`` is not plumbing AROUND the
  normalizer is not         metric, it IS half of it: ``"Hello, World!"`` scores zero
  reproducible              against ``"hello world"`` only because this module said
                            so. The manifest therefore names the normalizer
                            (``TRANSCRIPT_NORMALIZER_ID``) -- a second run can
                            reproduce the first run's numbers or refuse to compare,
                            and an "official WER" quoted without its normalizer is a
                            number nobody can re-derive.

  an empty denominator      ``EditCounts.rate`` raises on a zero reference length.
  is a refusal, never       One reference token and a hundred deletions IS a rate
  a 0.0                     (100.0); zero reference tokens has no rate at all, and
                            returning 0.0 there would turn "nothing measured" into
                            "perfect" -- the one lie a metric must never tell.

  the corpus rate is        Corpus scores are micro-averages, sums over sums:
  split-invariant           total errors / total reference length. The MEAN of the
                            per-utterance rates moves whenever a 200-word utterance
                            is segmented in two (that half gets a whole vote), and a
                            claim that moves with the segmentation of the corpus is
                            a claim about the segmenter, not about the model.

  a refusal counted is      ``CorpusErrorRate`` is the speech Coverage document:
  not an utterance lost     checked against what the caller expected, refusals
                            ACCOUNTED (empty normalized references), and zero scored
                            utterances is VACUOUS rather than a pass.

ties are canonical too: ``align`` decomposes the same pair into the same
(S, D, I) on every call by breaking equal-cost edits in a fixed order -- the
diagonal (match/substitution) first, then deletion, then insertion -- because a
report that changes its decomposition depending on which fold the DP walk tried
first cannot be diffed across runs.

stdlib only: the verification plane runs on boxes with no numpy and no torch, and
this module is part of that plane.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "TRANSCRIPT_NORMALIZER_ID",
    "CorpusErrorRate",
    "EditCounts",
    "align",
    "char_errors",
    "corpus_error_rate",
    "normalize_transcript",
    "word_errors",
]


# Named inside every manifest, and compared before any two runs' numbers are
# compared with each other: the normalizer is half of the measured claim, so an
# old manifest next to a new normalizer would silently rescore the same corpus.
TRANSCRIPT_NORMALIZER_ID = "foundationscale.speech_metrics.normalize_transcript/v1"


def normalize_transcript(text: str) -> list[str]:
    """``text`` as the tokens the metrics measure -- and half of the measured claim.

    ``casefold`` FIRST (so ``"Don't"`` and ``"DON'T"`` fold to one token, and so the
    order is fixed: casefolding can expand one character into several and the
    punctuation filter must see those), then every character that is neither
    alphanumeric nor an apostrophe becomes a space (``_`` is punctuation here too --
    ``"hello_world"`` is not one spoken word), then apostrophes are stripped at
    TOKEN edges so ``"'tis"`` is ``"tis"`` while ``"don't"`` stays one token, then
    the whitespace split.

    A token that normalizes to nothing (``"''"``) is dropped rather than kept as an
    empty word: it has no sound in it to have measured.

    Nothing here changes without ``TRANSCRIPT_NORMALIZER_ID`` changing with it -- the
    refusal is the point: a WER without its normalizer is not reproducible, and a
    restated transcript underneath a printed score is how a corpus gets rescored
    without anyone noticing.
    """
    folded = text.casefold()
    spaced = "".join(char if (char.isalnum() or char == "'") else " " for char in folded)
    tokens: list[str] = []
    for token in spaced.split():
        word = token.strip("'")
        if word:
            tokens.append(word)
    return tokens


@dataclass(frozen=True)
class EditCounts:
    """One alignment's decomposition, and the one rate it is allowed to publish.

    Frozen because these numbers ARE the measurement: a sum that mutates in place is
    a number whose provenance nobody can state.
    """

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def errors(self) -> int:
        """Every counted edit. Matches cost nothing and add nothing."""
        return self.substitutions + self.deletions + self.insertions

    def rate(self) -> float:
        """``errors`` over ``reference_length``.

        Raises on a zero reference length: an empty reference has NO defined rate,
        and 0.0 would be a fabricated claim of perfection over zero evidence. 100
        deletions against one reference token is a rate of 100.0 and is returned as
        such -- this is a rate, not a percentage, and is never clamped.
        """
        if self.reference_length == 0:
            raise ValueError(
                "error rate is undefined: reference length is zero and an empty "
                "reference has no words to have measured"
            )
        return self.errors / self.reference_length


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """``reference`` against ``hypothesis`` as substitution / deletion / insertion.

    The full matrix is kept because the decomposition is part of the claim and a
    rolling row cannot report it. Backtrace from the end, and on every equal-cost
    cell take -- in this order -- the diagonal (a match, or its mismatched form the
    substitution), then the deletion (a reference token the hypothesis never
    uttered), then the insertion. That order is a CONTRACT: ``["a", "b"]`` against
    ``["b", "a"]`` is 2 substitutions here and would be (deletion + insertion) for
    a walk that tried deletion first -- same distance, different report, and a
    report that shifts with the implementation order cannot be diffed between two
    runs of the same corpus.

    ``reference_length`` is the reference token count: the denominator this module
    uses at the word level, which ``char_errors`` overrides where a boundary is not
    content.
    """
    reference_tokens = list(reference)
    hypothesis_tokens = list(hypothesis)
    n = len(reference_tokens)
    m = len(hypothesis_tokens)

    rows: list[list[int]] = [list(range(m + 1))]
    for i in range(1, n + 1):
        row = [i] + [0] * m
        for j in range(1, m + 1):
            weight = 0 if reference_tokens[i - 1] == hypothesis_tokens[j - 1] else 1
            row[j] = min(
                rows[i - 1][j - 1] + weight,  # match, or substitution
                rows[i - 1][j] + 1,  # deletion
                row[j - 1] + 1,  # insertion
            )
        rows.append(row)

    i = n
    j = m
    substitutions = 0
    deletions = 0
    insertions = 0
    while i > 0 or j > 0:
        current = rows[i][j]
        if i > 0 and j > 0:
            weight = 0 if reference_tokens[i - 1] == hypothesis_tokens[j - 1] else 1
            if current == rows[i - 1][j - 1] + weight:
                substitutions += weight  # a match adds no edit
                i -= 1
                j -= 1
                continue
        if i > 0 and current == rows[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue
        if j > 0 and current == rows[i][j - 1] + 1:
            insertions += 1
            j -= 1
            continue
        raise RuntimeError(
            f"align backtrace reached a cell with no equal-cost fold: rows[{i}][{j}] = {current}"
        )

    return EditCounts(substitutions, deletions, insertions, n)


def word_errors(ref: str, hyp: str) -> EditCounts:
    """``ref`` against ``hyp`` per NORMALIZED WORD -- the WER claim.

    Both sides go through ``normalize_transcript`` before alignment, so the numbers
    here are only ever comparable with numbers whose normalizer is the one named in
    ``TRANSCRIPT_NORMALIZER_ID``.
    """
    return align(normalize_transcript(ref), normalize_transcript(hyp))


def char_errors(ref: str, hyp: str) -> EditCounts:
    """``ref`` against ``hyp`` per CHARACTER, boundaries out of the denominator.

    The normalized words are joined by single spaces -- one boundary per word pair
    -- and the alignment walks the characters INCLUDING those spaces. The
    reference_length, however, counts only NON-space characters, and that asymmetry
    is deliberate: the space is a boundary the normalizer invented (the speaker did
    not utter it), so it must never pad the denominator and flatter the score, while
    a boundary the model put in the wrong place DOES count in the numerator because
    moving a boundary changes what was said. Consequence, accepted rather than
    hidden: a rate here can exceed 1.0.
    """
    reference_words = normalize_transcript(ref)
    hypothesis_words = normalize_transcript(hyp)
    reference_chars = " ".join(reference_words)
    hypothesis_chars = " ".join(hypothesis_words)
    counts = align(reference_chars, hypothesis_chars)
    measured = sum(1 for char in reference_chars if char != " ")
    return EditCounts(counts.substitutions, counts.deletions, counts.insertions, measured)


@dataclass(frozen=True)
class CorpusErrorRate:
    """The speech Coverage document: one summed rate plus its verdict.

    ``utterances_refused`` counts ONLY empty normalized references -- the refusal
    this module can see. Everything else (missing column, undecodable audio) is
    refused upstream and belongs to that stage's coverage, not silently absorbed
    here.
    """

    metric: str
    counts: EditCounts
    utterances_expected: int
    utterances_scored: int
    utterances_refused: int

    @property
    def verdict(self) -> str:
        """VACUOUS first -- zero scored beats every accounting question.

        An empty corpus "passes" vacuously in every other arithmetic, and that is
        exactly the pass this refuses to hand out.
        """
        if self.utterances_scored == 0:
            return "VACUOUS"
        seen = self.utterances_scored + self.utterances_refused
        if seen < self.utterances_expected:
            return "UNDERCOVERED"
        if seen > self.utterances_expected:
            return "OVERCOVERED"
        return "COVERED"

    def rate(self) -> float:
        """Summed errors / summed reference length: the micro-average.

        NOT the mean of per-utterance rates: that mean gives a one-word utterance
        the same vote as a hundred-word one, so splitting one long utterance in two
        moves the published number even though the model said nothing different.
        The micro-average is invariant under re-segmentation, and that invariance is
        the whole reason to prefer it.

        Raises where there is nothing to divide by -- see ``EditCounts.rate``.
        """
        return self.counts.rate()

    def as_manifest(self) -> dict[str, int | float | str | None]:
        """The JSON-ready claim, normalizer named INSIDE it.

        ``error_rate`` is ``None`` where no rate is defined: a manifest is read by
        scripts that cannot see the difference between "0.0 because perfect" and
        "0.0 because divided by nothing", so the second case publishes nothing at
        all.
        """
        rate_value: float | None = None
        if self.counts.reference_length > 0:
            rate_value = self.rate()
        return {
            "metric": self.metric,
            "normalizer": TRANSCRIPT_NORMALIZER_ID,
            "substitutions": self.counts.substitutions,
            "deletions": self.counts.deletions,
            "insertions": self.counts.insertions,
            "errors": self.counts.errors,
            "reference_length": self.counts.reference_length,
            "error_rate": rate_value,
            "utterances_expected": self.utterances_expected,
            "utterances_scored": self.utterances_scored,
            "utterances_refused": self.utterances_refused,
            "verdict": self.verdict,
        }


def corpus_error_rate(
    pairs: Sequence[tuple[str, str]],
    *,
    metric: str,
    expected: int,
) -> CorpusErrorRate:
    """Every ``(reference, hypothesis)`` pair summed into one claim over the corpus.

    An utterance whose reference normalizes to NOTHING is refused and COUNTED as
    refused: it cannot carry a rate (see ``EditCounts.rate``), and dropping it
    silently would move both the numerator and the denominator of a claim about the
    model. ``expected`` is the caller's declaration of how many utterances this
    corpus should have carried -- it is what turns a silently short list into an
    UNDERCOVERED verdict instead of a clean score.

    Raises on a metric this module does not measure.
    """
    if metric == "wer":
        per_utterance = word_errors
    elif metric == "cer":
        per_utterance = char_errors
    else:
        raise ValueError(f'unknown speech error metric {metric!r}: expected "wer" or "cer"')

    substitutions = 0
    deletions = 0
    insertions = 0
    reference_length = 0
    scored = 0
    refused = 0
    for ref, hyp in pairs:
        if not normalize_transcript(ref):
            refused += 1
            continue
        counts = per_utterance(ref, hyp)
        substitutions += counts.substitutions
        deletions += counts.deletions
        insertions += counts.insertions
        reference_length += counts.reference_length
        scored += 1

    summed = EditCounts(substitutions, deletions, insertions, reference_length)
    return CorpusErrorRate(
        metric=metric,
        counts=summed,
        utterances_expected=expected,
        utterances_scored=scored,
        utterances_refused=refused,
    )
