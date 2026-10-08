"""Known-case, contract and property tests for ``train.speech_metrics``.

Every test is a MUST_PASS / MUST_FIRE pair: the claim the accepted path must
publish, and the count or refusal that must fire beside it. Nothing here is
skipped, xfailed or parametrized away -- these assertions ARE the measured claim,
and a WER that quietly degrades under a test nobody runs is a number that was never
measured in the first place.
"""

from __future__ import annotations

import json
import random

import pytest

from foundationscale.train.speech_metrics import (
    RUNAWAY_RATIO,
    RUNAWAY_SLACK_WORDS,
    TRANSCRIPT_NORMALIZER_ID,
    EditCounts,
    RunawayCount,
    align,
    char_errors,
    corpus_error_rate,
    count_runaway,
    is_runaway,
    normalize_transcript,
    word_errors,
)


def _levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    """The plain edit distance, sharing no code with ``align``.

    A property check that compares an implementation with itself proves nothing;
    this one is the textbook two-row distance and knows nothing about
    substitutions, decompositions or tie-breaking.
    """
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, token in enumerate(reference, start=1):
        current = [i]
        for j, other in enumerate(hypothesis, start=1):
            weight = 0 if token == other else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + weight))
        previous = current
    return previous[-1]


def test_identity_is_zero_and_one_substitution_is_one_third() -> None:
    """MUST_PASS: the perfect utterance publishes 0 errors over 3 words.

    MUST_FIRE: the wrong word must be COUNTED -- exactly one substitution and 1/3,
    which is the only reading that survives comparison with another run.
    """
    perfect = word_errors("the cat sat", "the cat sat")
    assert (perfect.errors, perfect.reference_length) == (0, 3), (
        "MUST_PASS: identity is zero errors over a three-word reference"
    )
    assert perfect.rate() == 0.0, (
        "MUST_PASS: a perfect transcription is rate 0.0 -- the one case where 0.0 is measured"
    )

    one_sub = word_errors("the cat sat", "the bat sat")
    assert one_sub == EditCounts(1, 0, 0, 3), (
        "MUST_FIRE: one wrong word is exactly one substitution over three reference words"
    )
    assert one_sub.rate() == 1 / 3, (
        "MUST_FIRE: the accepted-as-flat rate would read 0.0 and erase the measurement"
    )


def test_deletion_and_empty_hypothesis() -> None:
    """MUST_PASS: a word the hypothesis never uttered is one deletion.

    MUST_FIRE: an empty hypothesis deletes the WHOLE reference -- nothing is
    "missing-ish", every reference word is gone and is counted gone.
    """
    one_deletion = word_errors("the cat sat", "the sat")
    assert one_deletion == EditCounts(0, 1, 0, 3), (
        "MUST_PASS: dropping 'cat' is one deletion over a three-word reference"
    )
    assert one_deletion.rate() == 1 / 3, "MUST_PASS: 1 deletion / 3 words"

    empty_hypothesis = word_errors("the cat sat", "")
    assert empty_hypothesis == EditCounts(0, 3, 0, 3), (
        "MUST_FIRE: an empty hypothesis deletes all three reference words"
    )
    assert empty_hypothesis.rate() == 1.0, (
        "MUST_FIRE: everything deleted is a rate of 1.0, not a shrug"
    )


def test_insertion_and_rate_above_one() -> None:
    """MUST_PASS: a word the reference never had is one insertion.

    MUST_FIRE: a hallucinated hypothesis rates ABOVE 1.0 and is not clamped -- a
    clamped 1.0 would make "one error per three words" and "four errors per three
    words" the same published number.
    """
    one_insertion = word_errors("the cat sat", "the fat cat sat")
    assert one_insertion == EditCounts(0, 0, 1, 3), (
        "MUST_PASS: the reference word 'fat' never appeared is one insertion"
    )
    assert one_insertion.rate() == 1 / 3, "MUST_PASS: 1 insertion / 3 words"

    hallucination = word_errors("the cat sat", "jump jump jump jump")
    assert hallucination == EditCounts(3, 0, 1, 3), (
        "MUST_FIRE: three substitutions plus the extra fourth word is one insertion"
    )
    assert hallucination.rate() == 4 / 3, (
        "MUST_FIRE: a rate above 1.0 is real measurement and must survive unclamped"
    )


def test_punctuation_and_case_are_not_measurements() -> None:
    """MUST_PASS: the normalizer, not the dictionary, defines the endpoints.

    MUST_FIRE: nothing beyond punctuation and case moves -- no stemming, no
    rhyming, no fuzzy matching. "cats at" is NOT "cat sat" and must score.
    """
    assert normalize_transcript("Hello, World!") == ["hello", "world"], (
        "MUST_PASS: casefold and punctuation-to-space, applied in that order"
    )
    assert word_errors("Hello, World!", "hello world") == EditCounts(0, 0, 0, 2), (
        "MUST_PASS: 'Hello, World!' equals 'hello world' under the normalizer -- zero errors"
    )
    assert word_errors("the  cat', sat!", "the cat sat").errors == 0, (
        "MUST_PASS: whitespace runs collapse and stray apostrophes at word edges go"
    )

    near_miss = word_errors("the cats at", "the cat sat")
    assert near_miss == EditCounts(2, 0, 0, 3), (
        "MUST_FIRE: normalization must not rhyme -- 'cats at' vs 'cat sat' is two substitutions"
    )


def test_apostrophes_hold_a_word_together() -> None:
    """MUST_PASS: an interior apostrophe is content -- "don't" is one token.

    MUST_FIRE: edge apostrophes are quote marks and go, while a real spelling
    difference ("dont") is still error and not quietly equal.
    """
    assert normalize_transcript("don't") == ["don't"], (
        "MUST_PASS: the interior apostrophe keeps 'don't' one token"
    )
    assert word_errors("don't stop", "don't stop").errors == 0, (
        "MUST_PASS: 'don't' transcribes to itself at zero cost"
    )

    assert normalize_transcript("'tis fine") == ["tis", "fine"], (
        "MUST_FIRE: an apostrophe at the token's edge is a quote mark and is stripped"
    )
    assert normalize_transcript("'' ") == [], (
        "MUST_FIRE: a token with no sound in it normalizes to nothing and is dropped"
    )
    assert word_errors("don't", "dont") == EditCounts(1, 0, 0, 1), (
        "MUST_FIRE: 'dont' is a different word and must cost one substitution"
    )


def test_char_errors_known_case_and_boundary_denominator() -> None:
    """MUST_PASS: 'abc' against 'abd' is one substitution over three characters.

    MUST_FIRE: the word boundary is aligned but OUT of the denominator -- counting
    it would flatter 1/4 into 1/5 on the same evidence.
    """
    typo = char_errors("abc", "abd")
    assert typo == EditCounts(1, 0, 0, 3), (
        "MUST_PASS: one character substituted over a three-character reference"
    )
    assert typo.rate() == 1 / 3, "MUST_PASS: 1 / 3"

    boundary = char_errors("ab cd", "ab ce")
    assert boundary == EditCounts(1, 0, 0, 4), (
        "MUST_PASS: the aligned space is not counted as measured reference content"
    )
    assert boundary.rate() == 1 / 4, "MUST_PASS: 1 error over 4 measured characters"
    assert boundary.rate() != 1 / 5, (
        "MUST_FIRE: a denominator that padded itself with the normalizer's own "
        "boundary character would publish 1/5 for the same utterance"
    )


def test_empty_reference_has_no_rate_at_all() -> None:
    """MUST_PASS: an empty reference still decomposes (pure insertions).

    MUST_FIRE: the rate of an empty reference RAISES -- never 0.0, because 0.0 is
    the number a reader would take for perfection.
    """
    empty_words = word_errors("", "hello world")
    assert empty_words == EditCounts(0, 0, 2, 0), (
        "MUST_PASS: everything the hypothesis said is an insertion over a zero-length reference"
    )

    empty_chars = char_errors("", "abc")
    assert empty_chars == EditCounts(0, 0, 3, 0), (
        "MUST_PASS: the same at character level, denominator still zero"
    )

    void = align([], [])
    assert void.errors == 0, "MUST_PASS: nothing against nothing decomposes to nothing"

    for counts in (empty_words, empty_chars, void):
        with pytest.raises(ValueError, match="reference length"):
            counts.rate()  # MUST_FIRE: the ValueError


def test_corpus_micro_average_is_not_the_mean_of_rates() -> None:
    """MUST_PASS: 2 errors over 5 reference words is 0.4 regardless of the split.

    MUST_FIRE: the mean of the per-utterance rates moves with the segmentation and
    must NOT be the published number (0.625 here on the same evidence).
    """
    pairs = [("a b c d", "x b c d"), ("e", "x")]
    corpus = corpus_error_rate(pairs, metric="wer", expected=2)

    assert corpus.counts == EditCounts(2, 0, 0, 5), (
        "MUST_PASS: 2 substitutions summed over a 5-word reference"
    )
    assert corpus.rate() == 2 / 5, (
        "MUST_PASS: the micro-average is 2/5 and does not care how the pair list splits the corpus"
    )

    macro = sum(word_errors(ref, hyp).rate() for ref, hyp in pairs) / len(pairs)
    assert macro == 5 / 8, "MUST_FIRE: the mean of per-utterance rates is 1/4 and 1.0"
    assert corpus.rate() != macro, (
        "MUST_FIRE: publishing 0.625 would make the claim depend on the segmentation"
    )


def test_every_verdict_state() -> None:
    """MUST_PASS: COVERED fires only when scored + refused == expected.

    MUST_FIRE: zero scored is VACUOUS -- a short, refusal-only run may not walk
    away with a pass -- and unaccounted rows are UNDERCOVERED/OVERCOVERED.
    """
    covered = corpus_error_rate(
        [("the cat sat", "the bat sat"), ("", "hello")], metric="wer", expected=2
    )
    assert (covered.utterances_scored, covered.utterances_refused) == (1, 1), (
        "MUST_PASS: the empty-reference refusal is ACCOUNTED, not lost"
    )
    assert covered.verdict == "COVERED", "MUST_PASS: scored + refused == expected"

    under = corpus_error_rate([("a", "a"), ("b", "b")], metric="wer", expected=3)
    assert under.verdict == "UNDERCOVERED", (
        "MUST_FIRE: two utterances where three were expected is a short run, not a clean 0.0"
    )
    over = corpus_error_rate([("a", "a"), ("b", "b")], metric="wer", expected=1)
    assert over.verdict == "OVERCOVERED", (
        "MUST_FIRE: more utterances than declared means the declaration is wrong"
    )
    vacuous = corpus_error_rate([("", "hello"), ("", "x")], metric="wer", expected=1)
    assert vacuous.verdict == "VACUOUS", (
        "MUST_FIRE: two refusals and zero scored is VACUOUS, never a pass"
    )
    with pytest.raises(ValueError, match="reference length"):
        vacuous.rate()  # MUST_FIRE: nothing scored means no rate to publish


def test_only_wer_and_cer_are_measured() -> None:
    """MUST_PASS: both supported metrics accept the same pair list.

    MUST_FIRE: an unmeasured metric name raises -- spelling a claim this module
    cannot back is worse than stopping.
    """
    wer = corpus_error_rate([("a", "ab")], metric="wer", expected=1)
    cer = corpus_error_rate([("a", "ab")], metric="cer", expected=1)
    assert (wer.metric, cer.metric) == ("wer", "cer"), (
        "MUST_PASS: 'wer' and 'cer' are the two metrics this module measures"
    )
    assert cer.counts.reference_length == 1, (
        "MUST_PASS: the one-character reference is one measured character"
    )

    with pytest.raises(ValueError, match="unknown speech error metric"):
        corpus_error_rate([("a", "a")], metric="per", expected=1)  # MUST_FIRE
    with pytest.raises(ValueError, match="unknown speech error metric"):
        corpus_error_rate([("a", "a")], metric="WER", expected=1)  # MUST_FIRE


def test_manifest_is_the_whole_claim() -> None:
    """MUST_PASS: the manifest carries exactly the keys below, normalizer named.

    MUST_FIRE: an empty corpus publishes ``error_rate: null`` -- never a 0.0 that a
    reading script cannot tell from perfection.
    """
    corpus = corpus_error_rate(
        [("the cat sat", "the bat sat"), ("", "x")], metric="wer", expected=2
    )
    manifest = corpus.as_manifest()

    assert set(manifest) == {
        "metric",
        "normalizer",
        "substitutions",
        "deletions",
        "insertions",
        "errors",
        "reference_length",
        "error_rate",
        "utterances_expected",
        "utterances_scored",
        "utterances_refused",
        "verdict",
    }, "MUST_PASS: the manifest is exactly these keys, nothing more and nothing less"
    assert (
        manifest["normalizer"]
        == TRANSCRIPT_NORMALIZER_ID
        == ("foundationscale.speech_metrics.normalize_transcript/v1")
    ), "MUST_PASS: the normalizer is named inside the claim it underwrote"
    assert manifest["metric"] == "wer", "MUST_PASS: metric"
    assert (manifest["substitutions"], manifest["deletions"], manifest["insertions"]) == (1, 0, 0)
    assert (manifest["errors"], manifest["reference_length"]) == (1, 3)
    assert manifest["error_rate"] == 1 / 3, "MUST_PASS: 1 error over 3 reference words"
    assert (manifest["utterances_expected"], manifest["utterances_scored"]) == (2, 1)
    assert manifest["utterances_refused"] == 1
    assert manifest["verdict"] == "COVERED"
    assert json.loads(json.dumps(manifest)) == manifest, (
        "MUST_PASS: the manifest round-trips through JSON unchanged"
    )

    empty_manifest = corpus_error_rate([], metric="wer", expected=1).as_manifest()
    assert empty_manifest["error_rate"] is None, (
        "MUST_FIRE: an unmeasured rate is null and cannot be misread as 0.0"
    )
    assert empty_manifest["verdict"] == "VACUOUS", (
        "MUST_FIRE: the vacuous verdict travels with the manifest"
    )


def test_tie_break_prefers_substitution_deterministically() -> None:
    """MUST_PASS: ['a', 'b'] against ['b', 'a'] decomposes as two substitutions.

    MUST_FIRE: the deletion + insertion decomposition may NOT appear -- the order
    is a contract, and both walk again the same way every time.
    """
    ref = ["a", "b"]
    hyp = ["b", "a"]
    first = align(ref, hyp)
    assert first == EditCounts(2, 0, 0, 2), (
        "MUST_PASS: the substitution-first tie-break reports (2, 0, 0) at distance 2"
    )
    assert align(list(ref), list(hyp)) == first, (
        "MUST_PASS: the decomposition is a function of the pair, not of the walk"
    )
    assert (first.deletions, first.insertions) == (0, 0), (
        "MUST_FIRE: a deletion-first walk would report (0, 1, 1) and make the same "
        "corpus score differently across implementations"
    )


def test_property_errors_equal_independent_levenshtein() -> None:
    """MUST_PASS: over 200 random pairs, S + D + I IS the plain edit distance.

    MUST_FIRE: the loop must never be vacuous -- enough differing pairs that an
    identity-only shortcut cannot finish it intact.
    """
    rng = random.Random(0)
    vocabulary = ["a", "b", "b", "c", "d", "e", "ee", "x"]
    differing = 0
    for _ in range(200):
        ref = [rng.choice(vocabulary) for _ in range(rng.randrange(0, 9))]
        hyp = [rng.choice(vocabulary) for _ in range(rng.randrange(0, 9))]
        counts = align(ref, hyp)
        assert counts.reference_length == len(ref), (
            "MUST_PASS: the denominator is the reference token count"
        )
        assert counts.errors == _levenshtein(ref, hyp), (
            f"MUST_PASS: S+D+I equals the independent edit distance for {ref!r} vs {hyp!r}"
        )
        if ref == hyp:
            assert counts.errors == 0, "MUST_PASS: equal sequences score zero"
        else:
            assert counts.errors > 0, (
                f"MUST_FIRE: differing sequences must register error: {ref!r} vs {hyp!r}"
            )
            differing += 1
    assert differing >= 50, (
        f"MUST_FIRE: only {differing} differing pairs were generated -- the "
        "property proved nothing and must fail rather than pass vacuously"
    )


def test_runaway_boundary_is_strictly_greater() -> None:
    """MUST_PASS: a hypothesis landing exactly on the threshold is inside it.

    MUST_FIRE: one word past the threshold is a runaway -- the comparison is
    strictly greater, and that boundary is where two detectors silently
    disagree about the same corpus.
    """
    # 4 reference words: threshold = 2.0 * 4 + 5 = 13 words.
    exactly_thirteen = "a b c d e f g h i j k l m"
    assert not is_runaway("one two three four", exactly_thirteen), (
        "MUST_PASS: 13 words against 4 reference words lands exactly on 2 * 4 + 5 "
        "and is inside the threshold"
    )
    assert is_runaway("one two three four", exactly_thirteen + " n"), (
        "MUST_FIRE: one more word is 14 > 2 * 4 + 5 and must fire"
    )


def test_runaway_against_an_empty_reference() -> None:
    """MUST_PASS: at or under the slack, an empty reference is not a runaway.

    MUST_FIRE: past the slack alone it fires -- with nothing to echo, invention
    is not excused by the reference having nothing in it, and the empty row is
    still a row the detector judged (count_runaway counts it checked).
    """
    assert not is_runaway("", ""), "MUST_PASS: empty against empty is below any threshold"
    assert not is_runaway("", "one two three four five"), (
        "MUST_PASS: exactly the slack (5 words) against an empty reference is inside it"
    )
    assert is_runaway("", "one two three four five six"), (
        "MUST_FIRE: one word past the slack against an empty reference must fire"
    )


def test_runaway_counts_normalized_words_only() -> None:
    """MUST_PASS: case and punctuation are not words -- a shout is its own length.

    MUST_FIRE: they cannot mask a real runaway either -- a dressed repetition
    loop fires exactly like a plain one, because both sides run the normalizer.
    """
    reference = "the cat sat in the yard"  # 6 words: threshold = 2 * 6 + 5 = 17
    shouted = "THE CAT SAT, IN THE YARD!!!"
    assert not is_runaway(reference, shouted), (
        "MUST_PASS: case and punctuation normalize away -- 6 words against 6 words "
        "is inside the threshold however loudly it is written"
    )
    assert not is_runaway(reference, "!! ... ??? (( - --"), (
        "MUST_PASS: pure punctuation normalizes to zero words and can never be a runaway"
    )
    loop = "the cat sat in the yard " * 4  # 24 words: the measured failure shape
    assert is_runaway(reference, loop.upper() + "!? ..."), (
        "MUST_FIRE: 24 normalized words against 6 reference words is a runaway "
        "whether or not it is dressed in case and punctuation"
    )


def test_count_runaway_counts_the_rows_it_checked() -> None:
    """MUST_PASS: healthy rows are counted checked and never counted runaway.

    MUST_FIRE: only the true runaways are counted runaway -- deletions are not
    runaways, the slack is not a runaway, and no examined row vanishes from
    either count.
    """
    pairs = [
        ("the cat sat", "the cat sat"),  # 3 vs 3: healthy
        ("the cat sat", "the bat sat"),  # 3 words, one substitution
        ("the cat sat", ""),  # a whole reference deleted is not a runaway
        ("", "one two three four five"),  # exactly the slack: healthy
        ("", "one two three four five six"),  # one word past the slack
        ("one two", "x x x x x x x x x x"),  # 10 words over a threshold of 9
    ]
    counts = count_runaway(pairs)
    assert counts == RunawayCount(rows_checked=6, rows_runaway=2), (
        "MUST_PASS: all six rows checked and exactly the two length runaways counted -- "
        "the empty-reference row past the slack and the ten-word echo of 'one two'"
    )
    assert count_runaway([]) == RunawayCount(rows_checked=0, rows_runaway=0), (
        "MUST_PASS: an empty pair list is zero rows checked and invents nothing"
    )


def test_runaway_manifest_names_the_detector_arms() -> None:
    """MUST_PASS: the manifest carries the counts WITH ratio, slack and normalizer.

    MUST_FIRE: the counts alone are not the claim -- the same rows counted under
    another slack publish another verdict, so a manifest missing its arms cannot
    be re-derived and must not be the shape this module hands out.
    """
    manifest = count_runaway([("", "one two three four five six")]).as_manifest()
    assert set(manifest) == {
        "rows_checked",
        "rows_runaway",
        "runaway_ratio",
        "runaway_slack_words",
        "normalizer",
    }, "MUST_PASS: the manifest is exactly these keys, nothing more and nothing less"
    assert manifest["runaway_ratio"] == RUNAWAY_RATIO == 2.0, (
        "MUST_PASS: the ratio arm is published and is the frozen constant"
    )
    assert manifest["runaway_slack_words"] == RUNAWAY_SLACK_WORDS == 5, (
        "MUST_PASS: the slack arm is published and is the frozen constant"
    )
    assert manifest["normalizer"] == TRANSCRIPT_NORMALIZER_ID, (
        "MUST_PASS: the normalizer that defines the counted words is named inside "
        "the claim it underwrote"
    )
    assert (manifest["rows_checked"], manifest["rows_runaway"]) == (1, 1), (
        "MUST_FIRE: 6 words over an empty reference is one checked row of one runaway "
        "under the published slack -- hide the slack and a reader at slack 6 would "
        "re-derive 'healthy' from these same two numbers"
    )
    assert json.loads(json.dumps(manifest)) == manifest, (
        "MUST_PASS: the manifest round-trips through JSON unchanged"
    )
