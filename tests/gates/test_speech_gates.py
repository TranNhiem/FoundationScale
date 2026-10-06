"""Tests for the speech gates: audio-row coverage, placeholder coverage, tower movement.

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
