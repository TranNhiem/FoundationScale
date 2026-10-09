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

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "LOOP_MIN_RUN",
    "RUNAWAY_RATIO",
    "RUNAWAY_SLACK_WORDS",
    "TRANSCRIPT_NORMALIZER_ID",
    "CorpusErrorRate",
    "EditCounts",
    "LoopCount",
    "PairedComparison",
    "RunawayCount",
    "align",
    "char_errors",
    "corpus_error_rate",
    "count_loop_words",
    "count_runaway",
    "is_runaway",
    "loop_words",
    "normalize_transcript",
    "paired_bootstrap",
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


@dataclass(frozen=True)
class PairedComparison:
    """Two corpus rates over the SAME utterances, and the gap between them.

    A 300-row eval can manufacture a "gain" that vanishes at 2,000-2,700 rows:
    two models scored on two different subsets of a corpus disagree about the
    subsets' difficulty, not about the models, and NO arithmetic on two
    independent corpus rates can tell the difference. Only a PAIRED comparison
    can -- one row per utterance under each model, the gap resampled over
    utterances -- and this is everything such a comparison is allowed to publish.

    ``diff`` is ``tuned - base``: a NEGATIVE diff is a gain the tuned model made,
    the sign convention named in the claim itself rather than left to the
    reader's memory of which side was subtracted from which. ``rows`` is the rows
    MEASURED (rows whose reference normalizes to nothing are refused exactly as
    ``corpus_error_rate`` refuses them and are not counted), and ``significant``
    is not an opinion about the gap: it is the interval's own statement about
    whether zero is still possible on this corpus.

    Frozen because these numbers ARE the measurement: a rate that mutates in
    place is a number whose provenance nobody can state.
    """

    rows: int
    base_rate: float
    tuned_rate: float
    diff: float
    ci_low: float
    ci_high: float
    resamples: int
    seed: int
    confidence: float
    p_tuned_better: float
    metric: str = "wer"

    @property
    def significant(self) -> bool:
        """True iff the bootstrap interval excludes zero.

        An interval CONTAINING zero is not evidence of any direction of gap: the
        resampled corpora saw the sign flip often enough that "the tuned model is
        better" would be a claim about the draw, not about the models. That is
        precisely the 300-row "win" this exists to refuse.
        """
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def as_manifest(self) -> dict[str, int | float | str | bool]:
        """The JSON-ready claim, metric AND normalizer named INSIDE it.

        Both names travel because both halves decide whether the number is
        reproducible: the rate is only a ``wer`` claim under
        ``TRANSCRIPT_NORMALIZER_ID`` and a ``cer`` claim refuses to be
        re-derived as one. ``diff`` keeps its ``tuned - base`` sign so a script
        reading this cannot invert the comparison by accident.
        """
        return {
            "metric": self.metric,
            "normalizer": TRANSCRIPT_NORMALIZER_ID,
            "rows": self.rows,
            "base_rate": self.base_rate,
            "tuned_rate": self.tuned_rate,
            "diff": self.diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "resamples": self.resamples,
            "seed": self.seed,
            "confidence": self.confidence,
            "p_tuned_better": self.p_tuned_better,
            "significant": self.significant,
        }


def paired_bootstrap(
    pairs_base: Sequence[tuple[str, str]],
    pairs_tuned: Sequence[tuple[str, str]],
    *,
    metric: str = "wer",
    resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> PairedComparison:
    """``pairs_base`` against ``pairs_tuned`` as one gap, resampled over utterances.

    The two sequences are ALIGNED BY POSITION: entry i of each is the SAME
    utterance (the references match) heard by a different model. That alignment
    is the whole point and protecting it is the main refusal here -- a comparison
    over two different utterance sets measures the sets, and the "gains" that
    vanished when a 300-row eval grew to 2,000-2,700 rows were exactly this
    defect wearing a percentage sign. The bootstrap then draws UTTERANCES WITH
    REPLACEMENT from the paired rows: ONE draw serves both models, so the draw
    cannot hand one model an easier subset, which is what two independent
    bootstraps would permit.

    Per-row error counts and reference lengths are computed ONCE
    (``word_errors`` or ``char_errors``, the reference length of
    ``char_errors`` included), and both corpus rates are micro-averages exactly
    as ``corpus_error_rate`` publishes them: sum of errors over sum of reference
    length, never the mean of per-utterance rates. Each resample's gap is
    ``tuned_rate - base_rate`` over the drawn rows; the interval is the
    percentile band of the ``resamples`` gaps at indices
    ``floor(alpha / 2 * R)`` .. ``ceil((1 - alpha / 2) * R) - 1``, clamped into
    the sorted list, and ``p_tuned_better`` is the fraction of gaps strictly
    below zero.

    Empty references are handled exactly the way ``corpus_error_rate`` handles
    them: a reference that normalizes to nothing cannot carry a rate, so that
    row is excluded from BOTH sides (it is one row -- the references match) and
    ``PairedComparison.rows`` reports what was measured rather than what was
    handed in. A pair list whose every reference is empty is therefore a refusal
    and not a 0.0, and no resample can divide by zero afterwards: every measured
    row carries reference length >= 1 and every draw is a full ``rows`` long.

    Raises on pair lists of different length, on ANY position whose two
    references differ (misalignment is the defect this guards), on empty pair
    lists, on ``resamples < 1``, on a ``confidence`` outside (0, 1), and on a
    metric this module does not measure. Deterministic: a fixed ``seed`` fixes
    the entire resample stream.
    """
    if metric == "wer":
        per_utterance = word_errors
    elif metric == "cer":
        per_utterance = char_errors
    else:
        raise ValueError(f'unknown speech error metric {metric!r}: expected "wer" or "cer"')
    if resamples < 1:
        raise ValueError(
            f"paired bootstrap refuses {resamples} resamples: an interval needs at least "
            "one draw, and one draw is a number rather than a confidence"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"paired bootstrap refuses confidence {confidence!r}: the level must be "
            "strictly between 0 and 1"
        )
    if len(pairs_base) != len(pairs_tuned):
        raise ValueError(
            f"paired comparison misaligned: {len(pairs_base)} base pairs against "
            f"{len(pairs_tuned)} tuned pairs -- a paired comparison scores the SAME "
            "utterances and refuses two lists of different length"
        )
    if not pairs_base:
        raise ValueError(
            "paired comparison refuses an empty pair list: zero rows is "
            "not a gap of 0.0 and cannot be resampled into one"
        )
    for position, (base_row, tuned_row) in enumerate(zip(pairs_base, pairs_tuned, strict=True)):
        if base_row[0] != tuned_row[0]:
            raise ValueError(
                f"paired comparison misaligned at row {position}: references differ -- "
                f"the base side recorded {base_row[0]!r} and the tuned side "
                f"{tuned_row[0]!r}, which is two different utterances wearing one index"
            )

    rows_base: list[tuple[int, int]] = []
    rows_tuned: list[tuple[int, int]] = []
    refused = 0
    for (ref, hyp_base), (_, hyp_tuned) in zip(pairs_base, pairs_tuned, strict=True):
        if not normalize_transcript(ref):
            # The corpus_error_rate refusal lane, mirrored: the row is dropped from
            # BOTH sides and counted, so the denominators cannot drift apart.
            refused += 1
            continue
        base_counts = per_utterance(ref, hyp_base)
        tuned_counts = per_utterance(ref, hyp_tuned)
        rows_base.append((base_counts.errors, base_counts.reference_length))
        rows_tuned.append((tuned_counts.errors, tuned_counts.reference_length))
    rows = len(rows_base)
    if rows == 0:
        raise ValueError(
            f"paired comparison refused all {refused} rows: every reference is empty "
            "once normalized, and an empty reference has no words to have measured"
        )

    def rate_over(per_row: Sequence[tuple[int, int]], draw: Sequence[int]) -> float:
        """Sum of errors over sum of reference length on ``draw``: the micro-average.

        Every measured row carries reference length >= 1 (the refusal lane above
        removed exactly the rows whose reference has nothing in it) and every
        draw is ``rows`` long, so a resample's denominator cannot reach zero.
        The refusal below states that as a contract instead of leaving it to be
        believed: an empty denominator is never a 0.0.
        """
        errors = sum(per_row[i][0] for i in draw)
        length = sum(per_row[i][1] for i in draw)
        if length == 0:
            raise ValueError(
                "error rate is undefined: the drawn rows have a zero reference length "
                "in sum and that draw has no words to have measured"
            )
        return errors / length

    full = list(range(rows))
    base_rate = rate_over(rows_base, full)
    tuned_rate = rate_over(rows_tuned, full)

    rng = random.Random(seed)
    sampled: list[float] = []
    for _ in range(resamples):
        draw = [rng.randrange(rows) for _ in range(rows)]
        sampled.append(rate_over(rows_tuned, draw) - rate_over(rows_base, draw))

    ascending = sorted(sampled)
    alpha = 1.0 - confidence
    low_index = min(max(int(math.floor(alpha / 2.0 * resamples)), 0), resamples - 1)
    high_index = min(max(int(math.ceil((1.0 - alpha / 2.0) * resamples)) - 1, 0), resamples - 1)

    return PairedComparison(
        rows=rows,
        base_rate=base_rate,
        tuned_rate=tuned_rate,
        diff=tuned_rate - base_rate,
        ci_low=ascending[low_index],
        ci_high=ascending[high_index],
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        p_tuned_better=sum(1 for gap in sampled if gap < 0.0) / resamples,
        metric=metric,
    )


RUNAWAY_RATIO = 2.0
"""How many times the reference's words a hypothesis may reach before it runs away.

The ratio arm of the runaway detector (``RUNAWAY_SLACK_WORDS`` is the slack arm
and ``is_runaway`` is the only place the formula is stated). Motivated by a
failure no artifact gate saw: a Canary-1B fine-tune trained on rows whose audio
did not match their transcripts passed every adjudication gate, yet ran 122 of
its 2504 Earnings-22 hypotheses into repetition loops and hallucinated domain
text where the base model ran 3. A decoder that stopped listening leaves nothing
in the artifact; only hypothesis length against the reference shows it.
"""

RUNAWAY_SLACK_WORDS = 5
"""Word slack past the ratio, so segmentation noise is not counted as a runaway.

A permissively segmented utterance must not fire the detector on length alone;
the same slack is all the headroom invention gets against an empty reference
(``is_runaway``). Both knobs travel in every manifest they underwrite
(``RunawayCount.as_manifest``): a threshold nobody can state is a threshold
nobody can reproduce.
"""


def is_runaway(reference: str, hypothesis: str) -> bool:
    """``hypothesis`` ran away past ``reference`` -- a decoder that stopped listening.

    True iff NORMALIZED hypothesis words exceed
    ``RUNAWAY_RATIO * reference words + RUNAWAY_SLACK_WORDS``. Both sides go
    through ``normalize_transcript`` first: case and punctuation cannot
    manufacture a runaway and cannot hide one. The comparison is STRICTLY
    greater -- a hypothesis landing exactly on the threshold is inside it. An
    empty reference contributes zero words on the right-hand side, so anything
    past the slack alone is runaway by the same formula and not a special case:
    with nothing to echo, invention is not a length the reference sanctioned.
    """
    return len(normalize_transcript(hypothesis)) > (
        RUNAWAY_RATIO * len(normalize_transcript(reference)) + RUNAWAY_SLACK_WORDS
    )


@dataclass(frozen=True)
class RunawayCount:
    """Rows measured against the runaway detector, and how many ran away.

    The count sibling of ``CorpusErrorRate``: no rate to divide and no refusal
    lane -- a runaway is a per-row fact and the claim is a summed count over the
    rows actually examined. Frozen because these two numbers ARE the
    measurement.
    """

    rows_checked: int
    rows_runaway: int

    def as_manifest(self) -> dict[str, int | float | str]:
        """The JSON-ready count claim with the detector's arms named INSIDE it.

        Ratio, slack and the normalizer travel with the numbers that were
        counted under them: ``is_runaway`` counts normalized words, so two runs'
        counts are comparable only under one normalizer
        (``TRANSCRIPT_NORMALIZER_ID``), and the counts are re-derivable only
        from the threshold the manifest itself publishes.
        """
        return {
            "rows_checked": self.rows_checked,
            "rows_runaway": self.rows_runaway,
            "runaway_ratio": RUNAWAY_RATIO,
            "runaway_slack_words": RUNAWAY_SLACK_WORDS,
            "normalizer": TRANSCRIPT_NORMALIZER_ID,
        }


def count_runaway(pairs: Iterable[tuple[str, str]]) -> RunawayCount:
    """Every ``(reference, hypothesis)`` pair folded into one runaway census.

    No refusals and no expected denominator here: ``rows_checked`` is the rows
    actually examined, and the gate that consumes this supplies its coverage
    denominator from outside the artifact exactly as the row-coverage gate takes
    ``rows_expected`` from the training manifest. Every pair is judged by the
    one formula, ``is_runaway``, including rows whose reference is empty --
    invention past the slack is not excused by having nothing to echo, and no
    row is dropped silently from the count of rows examined.
    """
    checked = 0
    runaway = 0
    for ref, hyp in pairs:
        checked += 1
        if is_runaway(ref, hyp):
            runaway += 1
    return RunawayCount(rows_checked=checked, rows_runaway=runaway)


LOOP_MIN_RUN = 4
"""How many identical tokens in a row make a repetition loop the metrics must count.

The run-length arm of the LOCAL loop detector (``loop_words`` is the only place
the formula is stated). Motivated by a failure no row-level detector saw: a
Canary-1B Earnings fine-tune, decoded with NeMo's chunked long-form inference,
fell into LOCAL repetition loops INSIDE chunks (``"the the the ..."``,
``"uh uh uh ..."``) across six whole Earnings-22 calls -- 1,428 loop words
beside 50,400 reference words where the base model looped none, turning a
17.30 -> 14.14 WER gain (the loops collapsed, a diagnostic measurement) into the
real 17.30 -> 16.77. ``is_runaway`` passed over that output and correctly so:
it judges whole rows (past twice the row's reference length) and a loop inside
one 40 s chunk of an hour-long call adds only 4-12% of that call's words. Loops
must be counted locally, in words, against the base model -- and four is the run
where ``"the the the ..."`` stops having a stuttering speaker in it.

The threshold travels in every manifest it underwrites
(``LoopCount.as_manifest``): a threshold nobody can state is a threshold nobody
can reproduce.
"""


def loop_words(tokens: Sequence[str]) -> int:
    """Tokens inside maximal runs of ONE identical token, of length >= ``LOOP_MIN_RUN``.

    Every token of a qualifying run counts -- a run of five ``"the"`` is 5 loop
    words -- because each is a word the decoder emitted where the audio carried
    another. A shorter run contributes NOTHING (a run of 3 is 0, and 0 for the
    whole run rather than partial credit for its tail): a stutter is not a
    decoder that lost the transcript, and charging runs only at ``LOOP_MIN_RUN``
    keeps the boundary honest in both directions -- a run of 4 is 4 and a run of
    5 is 5, the loop length IS the measurement and no per-run cap shrinks it.
    Maximal runs separated by ANY other token add (a call that loops twice has
    looped twice), while a short run between two loops bridges neither.

    ``tokens`` are NORMALIZED words (``normalize_transcript``): the caller runs
    the normalizer first, so ``"The the THE the"`` is one run of four and no
    loop can hide in case or punctuation while another shows through it. This is
    a census of words -- no denominator and no refusals: an empty token list is
    0 loop words and a fact about nothing said, not a 0.0 about everything.
    """
    total = 0
    run_length = 0
    previous: str | None = None
    for token in tokens:
        if token == previous:
            run_length += 1
        else:
            if run_length >= LOOP_MIN_RUN:
                total += run_length
            run_length = 1
            previous = token
    if run_length >= LOOP_MIN_RUN:
        total += run_length
    return total


@dataclass(frozen=True)
class LoopCount:
    """Rows measured against the local loop detector, and the loop words in each side.

    The counting sibling of ``RunawayCount`` with the words it needs beside the
    rows: ``reference_words`` is the summed NORMALIZED reference length (the
    yardstick a gate's corpus slack is spent against) and the two loop-word sums
    are what ``loop_words`` counted inside the hypotheses and inside the
    references of those rows. A reference can loop too (a transcript artifact, a
    genuinely repeated utterance) and its loops are COUNTED rather than assumed
    away: the base-vs-tuned comparison needs both sides' dirt.

    Frozen because these numbers ARE the measurement: a sum that mutates in
    place is a number whose provenance nobody can state.
    """

    rows_checked: int
    reference_words: int
    hypothesis_loop_words: int
    reference_loop_words: int

    def as_manifest(self) -> dict[str, int | str]:
        """The JSON-ready count claim with the detector's arm named INSIDE it.

        ``loop_min_run`` and the normalizer travel with the numbers counted
        under them: ``loop_words`` counts normalized words, so two runs' counts
        are comparable only under one normalizer
        (``TRANSCRIPT_NORMALIZER_ID``) and re-derivable only from the run length
        the manifest itself publishes -- a reader counting runs of three would
        re-derive a different census from these same four numbers.
        """
        return {
            "rows_checked": self.rows_checked,
            "reference_words": self.reference_words,
            "hypothesis_loop_words": self.hypothesis_loop_words,
            "reference_loop_words": self.reference_loop_words,
            "loop_min_run": LOOP_MIN_RUN,
            "normalizer": TRANSCRIPT_NORMALIZER_ID,
        }


def count_loop_words(pairs: Iterable[tuple[str, str]]) -> LoopCount:
    """Every ``(reference, hypothesis)`` TEXT pair folded into one loop-word census.

    Both sides run ``normalize_transcript`` before ``loop_words``, so the counts
    are only ever comparable with counts whose normalizer is the one named in
    ``TRANSCRIPT_NORMALIZER_ID``. Rows whose reference is empty are COUNTED --
    they contribute 0 reference words and 0 reference loop words -- and are
    never refused here: nothing on this side of the file divides by the
    reference, so an empty reference is a row examined with zero words in it,
    and dropping it in silence would lose the very hypothesis loops it may still
    carry (an empty recording is not a well-behaved one). No ``rows_expected``
    denominator either: ``rows_checked`` is the rows actually examined, and the
    gate that consumes these counts takes its coverage denominator from outside
    the eval artifact, exactly as ``count_runaway`` leaves it to
    ``RunawayHypothesisGate``.
    """
    checked = 0
    reference_words = 0
    hypothesis_loop_words = 0
    reference_loop_words = 0
    for ref, hyp in pairs:
        checked += 1
        ref_tokens = normalize_transcript(ref)
        hyp_tokens = normalize_transcript(hyp)
        reference_words += len(ref_tokens)
        hypothesis_loop_words += loop_words(hyp_tokens)
        reference_loop_words += loop_words(ref_tokens)
    return LoopCount(
        rows_checked=checked,
        reference_words=reference_words,
        hypothesis_loop_words=hypothesis_loop_words,
        reference_loop_words=reference_loop_words,
    )
