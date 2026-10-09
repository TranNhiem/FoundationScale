"""Tests for the speech gates: audio-row coverage, placeholder coverage, movement, runaways, loops.

Two layers of coverage per gate:

1. ``verify_controls`` — every gate's declared fixture matrix (MUST_FIRE /
   MUST_PASS / expect_skip) is routed through the framework's verifier, so the
   controls that prove the gate can block and can pass are themselves proven.
2. Direct :meth:`Gate.check` calls — every branch of every gate is walked and
   asserted against the expected ``Verdict`` and the exact ``Coverage`` numbers
   (``checked`` / ``expected`` / ``unit``) the branch must report.

The gate module + tests use only stdlib and ``foundationscale.gates.core`` — no
torch or numpy is imported anywhere here, so the test suite runs on a bare
interpreter.
"""

from __future__ import annotations

import pytest

from foundationscale.gates.core import (
    REGISTRY,
    AbstentionKind,
    GateRegistry,
    Verdict,
    verify_controls,
)
from foundationscale.gates.speech_gates import (
    AudioPlaceholderContext,
    AudioPlaceholderCoverageGate,
    AudioRowCoverageContext,
    AudioRowCoverageGate,
    RepetitionLoopContext,
    RepetitionLoopGate,
    RunawayHypothesisContext,
    RunawayHypothesisGate,
    TowerMovementContext,
    TowerMovementGate,
)


class TestSpeechGateControls:
    """Each gate's declared fixture matrix must hold under ``verify_controls``."""

    @pytest.mark.parametrize(
        ("gate_cls", "gate_id"),
        [
            (AudioRowCoverageGate, "speech.audio_row_coverage"),
            (AudioPlaceholderCoverageGate, "speech.audio_placeholder_coverage"),
            (TowerMovementGate, "speech.tower_movement"),
        ],
    )
    def test_verify_controls_per_gate(self, gate_cls, gate_id):
        registry = GateRegistry()
        registry.register(gate_cls())
        failures = verify_controls(registry, gate_ids=[gate_id])
        assert failures == [], f"controls for {gate_id} did not hold: {failures}"

    def test_all_three_gates_together_hold(self):
        registry = GateRegistry()
        registry.register(AudioRowCoverageGate())
        registry.register(AudioPlaceholderCoverageGate())
        registry.register(TowerMovementGate())
        failures = verify_controls(registry)
        assert failures == [], f"combined control suite failed: {failures}"


class TestAudioRowCoverageGate:
    def _gate(self) -> AudioRowCoverageGate:
        return AudioRowCoverageGate()

    def test_all_rows_loaded_passes(self):
        ctx = AudioRowCoverageContext(
            rows_expected=5,
            rows_checked=5,
            rows_refused=0,
            refused={},
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 5
        assert result.coverage.expected == 5
        assert result.coverage.unit == "audio rows"

    def test_no_rows_loaded_vacuous(self):
        ctx = AudioRowCoverageContext(
            rows_expected=5,
            rows_checked=0,
            rows_refused=0,
            refused={},
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert result.coverage.checked == 0
        assert result.coverage.unit == "audio rows"

    def test_no_rows_loaded_with_refusals_still_vacuous_with_evidence(self):
        # rows_checked==0 is first in the spec's rule ordering and produces
        # VACUOUS even when refusals exist; refusals surface as evidence so the
        # reader sees what happened alongside the blocking verdict.
        ctx = AudioRowCoverageContext(
            rows_expected=5,
            rows_checked=0,
            rows_refused=2,
            refused={"decoder_failure": 2},
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert result.coverage.checked == 0
        # refusals still visible in evidence for the audit trail.
        assert result.evidence["rows_refused"] == 2
        assert result.evidence["refused"] == {"decoder_failure": 2}

    def test_strict_refusal_leaked_fails(self):
        ctx = AudioRowCoverageContext(
            rows_expected=5,
            rows_checked=3,
            rows_refused=2,
            refused={"decoder_failure": 2},
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        # Coverage formula per spec: checked = loaded + refused = 3 + 2 = 5.
        assert result.coverage.checked == 5
        assert result.coverage.expected == 5
        assert "decoder_failure" in result.detail

    def test_undercover_row_count_blocks(self):
        ctx = AudioRowCoverageContext(
            rows_expected=5,
            rows_checked=3,
            rows_refused=0,
            refused={},
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.UNDERCOVERED
        assert result.blocking
        assert result.coverage.checked == 3
        assert result.coverage.expected == 5

    def test_overcover_row_count_blocks(self):
        ctx = AudioRowCoverageContext(
            rows_expected=3,
            rows_checked=5,
            rows_refused=0,
            refused={},
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.OVERCOVERED
        assert result.blocking
        assert result.coverage.checked == 5
        assert result.coverage.expected == 3


class TestAudioPlaceholderCoverageGate:
    def _gate(self) -> AudioPlaceholderCoverageGate:
        return AudioPlaceholderCoverageGate()

    def test_all_verified_passes(self):
        ctx = AudioPlaceholderContext(
            rows_checked=5,
            placeholder_rows_verified=5,
            placeholder_rows_unmeasured=0,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 5
        assert result.coverage.expected == 5
        assert result.coverage.unit == "audio rows"

    def test_denominator_mismatch_fails(self):
        ctx = AudioPlaceholderContext(
            rows_checked=10,
            placeholder_rows_verified=5,
            placeholder_rows_unmeasured=2,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        # Coverage reports the impossible ratio checked=5 against expected=10;
        # self.fail blocks regardless of coverage arithmetic.
        assert result.coverage.checked == 5
        assert result.coverage.expected == 10
        assert "denominator mismatch" in result.detail

    def test_zero_rows_vacuous(self):
        ctx = AudioPlaceholderContext(
            rows_checked=0,
            placeholder_rows_verified=0,
            placeholder_rows_unmeasured=0,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert result.coverage.checked == 0

    def test_all_unmeasured_takes_not_established_skip(self):
        ctx = AudioPlaceholderContext(
            rows_checked=5,
            placeholder_rows_verified=0,
            placeholder_rows_unmeasured=5,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.SKIP
        assert not result.blocking
        # The SKIP must carry the declared NOT_ESTABLISHED abstention kind so
        # composites can price it (stays in the denominator).
        assert result.abstention is (AbstentionKind.NOT_ESTABLISHED)

    def test_partial_unmeasured_undercovered(self):
        ctx = AudioPlaceholderContext(
            rows_checked=5,
            placeholder_rows_verified=3,
            placeholder_rows_unmeasured=2,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.UNDERCOVERED
        assert result.blocking
        assert result.coverage.checked == 3
        assert result.coverage.expected == 5


class TestTowerMovementGate:
    def _gate(self) -> TowerMovementGate:
        return TowerMovementGate()

    def test_no_tower_params_vacuous(self):
        ctx = TowerMovementContext(
            tower_prefix="speech.tower",
            base_digests={},
            saved_digests={},
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert result.coverage.checked == 0

    def test_buffer_suffixes_excluded_from_coverage_and_movement(self):
        ctx = TowerMovementContext(
            tower_prefix="speech.tower",
            base_digests={
                "speech.tower.encoder.w": "abc",
                "speech.tower.encoder.input_min": "b1",
                "speech.tower.encoder.input_max": "b2",
            },
            saved_digests={
                "speech.tower.encoder.w": "different",
                # Buffers differ but must NOT count toward movement.
                "speech.tower.encoder.input_min": "c1",
                "speech.tower.encoder.input_max": "c2",
            },
            exercised=True,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.PASS
        # Only encoder.w counts as a trainable parameter (buffer suffixes
        # filtered out); expected and checked are both 1.
        assert result.coverage.checked == 1
        assert result.coverage.expected == 1

    def test_missing_in_saved_undercovered(self):
        ctx = TowerMovementContext(
            tower_prefix="speech.tower",
            base_digests={
                "speech.tower.encoder.w": "abc",
                "speech.tower.encoder.b": "def",
                "speech.tower.decoder.w": "ghi",
            },
            saved_digests={
                "speech.tower.encoder.w": "abc",
            },
            exercised=True,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.UNDERCOVERED
        assert result.blocking
        assert result.coverage.checked == 1
        assert result.coverage.expected == 3

    def test_all_missing_vacuous(self):
        # All params in base but none in saved: checked drops to 0 and the
        # framework's self.ok() downgrade takes the VACUOUS path (checked == 0
        # outranks is_short / UNDERCOVERED in the downgrade chain).
        ctx = TowerMovementContext(
            tower_prefix="speech.tower",
            base_digests={
                "speech.tower.encoder.w": "abc",
                "speech.tower.encoder.b": "def",
            },
            saved_digests={},
            exercised=True,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert result.coverage.checked == 0
        assert result.coverage.expected == 2

    def test_exercised_but_unchanged_fails(self):
        ctx = TowerMovementContext(
            tower_prefix="speech.tower",
            base_digests={"speech.tower.encoder.w": "abc"},
            saved_digests={"speech.tower.encoder.w": "abc"},
            exercised=True,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert result.coverage.checked == 1
        assert result.coverage.expected == 1
        assert "carried unchanged" in result.detail

    def test_dormant_but_moved_fails(self):
        ctx = TowerMovementContext(
            tower_prefix="speech.tower",
            base_digests={"speech.tower.encoder.w": "abc"},
            saved_digests={"speech.tower.encoder.w": "different"},
            exercised=False,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.FAIL
        assert result.blocking

    def test_exercised_and_moved_passes(self):
        ctx = TowerMovementContext(
            tower_prefix="speech.tower",
            base_digests={"speech.tower.encoder.w": "abc"},
            saved_digests={"speech.tower.encoder.w": "different"},
            exercised=True,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 1
        assert result.coverage.expected == 1
        # Message reports moved/total per spec.
        assert "1 of 1 tower parameters moved" in result.detail

    def test_dormant_and_static_passes(self):
        ctx = TowerMovementContext(
            tower_prefix="speech.tower",
            base_digests={"speech.tower.encoder.w": "abc"},
            saved_digests={"speech.tower.encoder.w": "abc"},
            exercised=False,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 1
        assert result.coverage.expected == 1

    def test_prefix_mismatch_treated_as_no_params(self):
        # Params outside the tower prefix must not count toward movement even
        # if they differ; the gate is scoped to tower_prefix only.
        ctx = TowerMovementContext(
            tower_prefix="speech.tower",
            base_digests={"other.module.w": "abc"},
            saved_digests={"other.module.w": "different"},
            exercised=True,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking


def test_tower_prefix_matches_whole_segments_only() -> None:
    """A sibling module sharing the prefix text is not part of the tower."""
    from foundationscale.gates.speech_gates import TowerMovementContext, TowerMovementGate

    ctx = TowerMovementContext(
        tower_prefix="model.audio_tower",
        base_digests={"model.audio_tower.w": "a", "model.audio_tower_proj.w": "b"},
        saved_digests={"model.audio_tower.w": "a2", "model.audio_tower_proj.w": "b"},
    )
    result = TowerMovementGate().check(ctx)
    assert result.coverage.checked == 1
    assert result.coverage.expected == 1


class TestRunawayHypothesisGate:
    def _gate(self) -> RunawayHypothesisGate:
        return RunawayHypothesisGate()

    def test_measured_collapse_fails(self):
        # The measured failure that motivated the gate: a Canary-1B fine-tune
        # that passed every adjudication gate and still ran 122 of 2504
        # Earnings-22 hypotheses into repetition loops / hallucinated domain
        # text where the base model ran 3. limit = 2 * 3 + ceil(0.005 * 2504)
        # = 6 + 13 = 19 < 122.
        ctx = RunawayHypothesisContext(
            rows_expected=2504,
            rows_checked=2504,
            base_runaway=3,
            tuned_runaway=122,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert result.coverage.checked == 2504
        assert result.coverage.expected == 2504
        assert result.coverage.unit == "eval rows"
        # The detail must name both counts and the limit they were judged with.
        assert "122 of 2504 rows ran away" in result.detail
        assert "limit of 19" in result.detail
        assert "2 * base_runaway 3" in result.detail
        assert result.evidence["limit"] == 19

    def test_measured_healthy_run_passes(self):
        # The measured healthy run over the same corpus (a clean LibriSpeech+AMI
        # fine-tune): 7 of 2504 rows against 3 base runaways -- inside the limit of 19.
        ctx = RunawayHypothesisContext(
            rows_expected=2504,
            rows_checked=2504,
            base_runaway=3,
            tuned_runaway=7,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 2504
        assert result.coverage.expected == 2504
        assert result.coverage.unit == "eval rows"

    def test_limit_boundary_is_inclusive(self):
        # base 3 over 2504 rows: limit = 2 * 3 + ceil(0.005 * 2504) = 19.
        # Landing exactly on the limit is inside it (entirely declared drift);
        # one row past it is the first row outside the run's allowance and must
        # block.
        at_limit = RunawayHypothesisContext(
            rows_expected=2504,
            rows_checked=2504,
            base_runaway=3,
            tuned_runaway=19,
        )
        over_limit = RunawayHypothesisContext(
            rows_expected=2504,
            rows_checked=2504,
            base_runaway=3,
            tuned_runaway=20,
        )
        result = self._gate().check(at_limit)
        assert result.verdict is Verdict.PASS, "tuned == limit is inside the limit"
        result = self._gate().check(over_limit)
        assert result.verdict is Verdict.FAIL
        assert result.blocking, "tuned == limit + 1 must block"

    def test_no_rows_vacuous(self):
        ctx = RunawayHypothesisContext(
            rows_expected=5,
            rows_checked=0,
            base_runaway=0,
            tuned_runaway=0,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert result.coverage.checked == 0
        assert result.coverage.unit == "eval rows"

    def test_undercover_row_count_blocks(self):
        # 3 of 5 rows measured and the runaways inside the limit (2 * 1 +
        # ceil(0.005 * 3) = 3): the shortfall of 2 must block via the
        # framework's self.ok() downgrade.
        ctx = RunawayHypothesisContext(
            rows_expected=5,
            rows_checked=3,
            base_runaway=1,
            tuned_runaway=1,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.UNDERCOVERED
        assert result.blocking
        assert result.coverage.checked == 3
        assert result.coverage.expected == 5

    def test_overcover_row_count_blocks(self):
        # runaways inside the limit (2 * 0 + ceil(0.005 * 5) = 1), but the row
        # census contradicts its own denominator.
        ctx = RunawayHypothesisContext(
            rows_expected=3,
            rows_checked=5,
            base_runaway=0,
            tuned_runaway=1,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.OVERCOVERED
        assert result.blocking
        assert result.coverage.checked == 5
        assert result.coverage.expected == 3

    def test_impossible_census_is_refused(self):
        # A negative count, or a runaway count over more rows than were checked,
        # describes no run at all: refused as ValueError (the idiom Coverage's
        # own constructor uses for a negative count) and never priced as a
        # verdict about the model.
        with pytest.raises(ValueError, match="cannot be negative"):
            self._gate().check(
                RunawayHypothesisContext(
                    rows_expected=5,
                    rows_checked=5,
                    base_runaway=-1,
                    tuned_runaway=1,
                )
            )
        for base_runaway, tuned_runaway in ((6, 1), (0, 6)):
            with pytest.raises(ValueError, match="cannot outrun"):
                self._gate().check(
                    RunawayHypothesisContext(
                        rows_expected=5,
                        rows_checked=5,
                        base_runaway=base_runaway,
                        tuned_runaway=tuned_runaway,
                    )
                )

    def test_verify_controls_per_gate(self):
        registry = GateRegistry()
        registry.register(RunawayHypothesisGate())
        failures = verify_controls(registry, gate_ids=["speech.runaway_hypotheses"])
        assert failures == [], f"controls for speech.runaway_hypotheses did not hold: {failures}"

    def test_registered_under_declared_id(self):
        # @register must have the gate in the process-wide REGISTRY at import
        # time, under its declared id and wired to its own context type.
        gate = REGISTRY.get("speech.runaway_hypotheses")
        assert isinstance(gate, RunawayHypothesisGate), (
            f"speech.runaway_hypotheses must resolve to the runaway gate in "
            f"REGISTRY, got {type(gate).__name__}"
        )
        assert gate.context_type is RunawayHypothesisContext


class TestRepetitionLoopGate:
    def _gate(self) -> RepetitionLoopGate:
        return RepetitionLoopGate()

    def test_measured_long_form_loops_fail(self):
        # The measured failure that motivated the gate: NeMo chunked long-form
        # inference over 6 whole Earnings-22 calls (50,400 reference words)
        # where a Canary-1B Earnings fine-tune fell into LOCAL repetition loops
        # inside chunks ("the the the ...", "uh uh uh ...") -- 1,428 loop words
        # where the base model looped none -- turning a 17.30 -> 14.14 WER gain
        # into 17.30 -> 16.77. speech.runaway_hypotheses passed over the same
        # output and could not have fired: a loop inside one 40 s chunk of an
        # hour-long call adds 4-12% of the call's words.
        # limit = 2 * 0 + ceil(0.005 * 50400) = 252.
        ctx = RepetitionLoopContext(
            rows_expected=6,
            rows_checked=6,
            reference_words=50400,
            base_loop_words=0,
            tuned_loop_words=1428,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert result.coverage.checked == 6
        assert result.coverage.expected == 6
        assert result.coverage.unit == "eval rows"
        # The detail must name both counts, the reference words and the limit.
        assert "1428 loop words over 50400 reference words" in result.detail
        assert "the base model produced 0" in result.detail
        assert "limit of 252" in result.detail
        assert "2 * base_loop_words 0" in result.detail
        assert result.evidence["limit"] == 252

    def test_measured_corrupted_data_clips_fail(self):
        # The corrupted-data fine-tune's clip evaluation: 2,936 loop words where
        # the base model looped none. limit = 2 * 0 + ceil(0.005 * 47865) = 240.
        ctx = RepetitionLoopContext(
            rows_expected=2504,
            rows_checked=2504,
            reference_words=47865,
            base_loop_words=0,
            tuned_loop_words=2936,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.FAIL
        assert result.blocking
        assert "2936 loop words over 47865 reference words" in result.detail
        assert "limit of 240" in result.detail
        assert result.evidence["limit"] == 240

    def test_measured_clip_finetune_passes(self):
        # The clean clip fine-tune over the same corpus: 151 loop words over
        # 47,865 reference words with a base looping none -- inside limit 240.
        ctx = RepetitionLoopContext(
            rows_expected=2504,
            rows_checked=2504,
            reference_words=47865,
            base_loop_words=0,
            tuned_loop_words=151,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 2504
        assert result.coverage.expected == 2504
        assert result.coverage.unit == "eval rows"

    def test_ami_base_loops_too_passes(self):
        # A corpus the BASE model also loops on: 165 base loop words in 15,194
        # reference words and a fine-tune at 188 -- limit = 2 * 165 +
        # ceil(0.005 * 15194) = 330 + 76 = 406. The base model is the yardstick,
        # not a zero, or every repetition-prone corpus would block.
        ctx = RepetitionLoopContext(
            rows_expected=2000,
            rows_checked=2000,
            reference_words=15194,
            base_loop_words=165,
            tuned_loop_words=188,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.PASS
        assert result.coverage.checked == 2000
        assert result.coverage.expected == 2000
        assert "limit of 406" in result.detail

    def test_limit_boundary_is_inclusive(self):
        # base 25 loop words over 20,000 reference words: limit = 2 * 25 +
        # ceil(0.005 * 20000) = 50 + 100 = 150. Landing exactly on the limit is
        # inside it (entirely declared drift); one loop word past it is the
        # first word outside the run's allowance and must block.
        at_limit = RepetitionLoopContext(
            rows_expected=100,
            rows_checked=100,
            reference_words=20000,
            base_loop_words=25,
            tuned_loop_words=150,
        )
        over_limit = RepetitionLoopContext(
            rows_expected=100,
            rows_checked=100,
            reference_words=20000,
            base_loop_words=25,
            tuned_loop_words=151,
        )
        result = self._gate().check(at_limit)
        assert result.verdict is Verdict.PASS, "tuned == limit is inside the limit"
        result = self._gate().check(over_limit)
        assert result.verdict is Verdict.FAIL
        assert result.blocking, "tuned == limit + 1 must block"

    def test_no_rows_vacuous(self):
        ctx = RepetitionLoopContext(
            rows_expected=5,
            rows_checked=0,
            reference_words=0,
            base_loop_words=0,
            tuned_loop_words=0,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.VACUOUS
        assert result.blocking
        assert result.coverage.checked == 0
        assert result.coverage.unit == "eval rows"

    def test_undercover_row_count_blocks(self):
        # 3 of 5 rows measured with the loops inside the limit (2 * 1 +
        # ceil(0.005 * 100) = 3): the shortfall of 2 must block via the
        # framework's self.ok() downgrade.
        ctx = RepetitionLoopContext(
            rows_expected=5,
            rows_checked=3,
            reference_words=100,
            base_loop_words=1,
            tuned_loop_words=1,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.UNDERCOVERED
        assert result.blocking
        assert result.coverage.checked == 3
        assert result.coverage.expected == 5

    def test_overcover_row_count_blocks(self):
        # Loops inside the limit (2 * 0 + ceil(0.005 * 100) = 1), but the row
        # census contradicts its own denominator.
        ctx = RepetitionLoopContext(
            rows_expected=3,
            rows_checked=5,
            reference_words=100,
            base_loop_words=0,
            tuned_loop_words=1,
        )
        result = self._gate().check(ctx)
        assert result.verdict is Verdict.OVERCOVERED
        assert result.blocking
        assert result.coverage.checked == 5
        assert result.coverage.expected == 3

    def test_negative_count_is_refused_and_a_collapse_is_priced(self):
        # A negative count in any field describes no run at all: refused as
        # ValueError (the idiom Coverage's own constructor uses for a negative
        # count) and never priced as a verdict about the model. But loop words
        # ABOVE the reference's words are the measured collapse -- hypothesis
        # words are not bounded by the reference they failed to follow -- and
        # must be priced (RED) rather than refused as an impossible census.
        for impossible in (
            dict(
                rows_expected=-1,
                rows_checked=5,
                reference_words=50,
                base_loop_words=0,
                tuned_loop_words=1,
            ),
            dict(
                rows_expected=5,
                rows_checked=-1,
                reference_words=50,
                base_loop_words=0,
                tuned_loop_words=1,
            ),
            dict(
                rows_expected=5,
                rows_checked=5,
                reference_words=-1,
                base_loop_words=0,
                tuned_loop_words=1,
            ),
            dict(
                rows_expected=5,
                rows_checked=5,
                reference_words=50,
                base_loop_words=-1,
                tuned_loop_words=1,
            ),
            dict(
                rows_expected=5,
                rows_checked=5,
                reference_words=50,
                base_loop_words=0,
                tuned_loop_words=-1,
            ),
        ):
            with pytest.raises(ValueError, match="cannot be negative"):
                self._gate().check(RepetitionLoopContext(**impossible))

        collapse = self._gate().check(
            RepetitionLoopContext(
                rows_expected=1,
                rows_checked=1,
                reference_words=3,
                base_loop_words=0,
                tuned_loop_words=200,
            )
        )
        assert collapse.verdict is Verdict.FAIL
        assert collapse.blocking

    def test_verify_controls_per_gate(self):
        registry = GateRegistry()
        registry.register(RepetitionLoopGate())
        failures = verify_controls(registry, gate_ids=["speech.repetition_loops"])
        assert failures == [], f"controls for speech.repetition_loops did not hold: {failures}"

    def test_registered_under_declared_id(self):
        # @register must have the gate in the process-wide REGISTRY at import
        # time, under its declared id and wired to its own context type.
        gate = REGISTRY.get("speech.repetition_loops")
        assert isinstance(gate, RepetitionLoopGate), (
            f"speech.repetition_loops must resolve to the repetition-loop gate in "
            f"REGISTRY, got {type(gate).__name__}"
        )
        assert gate.context_type is RepetitionLoopContext
