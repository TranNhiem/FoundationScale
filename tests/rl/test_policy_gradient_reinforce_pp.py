"""Hostile-branch tests for the Reinforce++ binding of the policy-gradient family.

Every test here targets a branch of ``ReinforcePlusPlusLoss`` or
``ReinforcePlusPlusAlgorithm`` that the sibling test module already in the
repository does not reach: refusal guards in ``compute_with_report`` and
``setup``, the pre-setup and dataloader-exhaustion refusals in ``step``, the
``semantics``/``requires`` one-line accessors, the ``__call__`` surface, and
the two PPO clip legs taken from both advantage signs at each bound.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from foundationscale.rl import policy_gradient
from foundationscale.rl.algorithm import (
    AlgorithmWiringRefusal,
    StepReportRefusal,
    verify_step,
)
from foundationscale.rl.interfaces import BatchRefusal, SupervisionRefusal
from foundationscale.rl.policy import PolicyPair
from foundationscale.rl.policy_gradient import (
    ReinforcePlusPlusAlgorithm,
    ReinforcePlusPlusLoss,
)

__all__ = ()


@dataclass(frozen=True, slots=True)
class _Batch:
    source: Mapping[str, Any]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.source)

    def column(self, name: str) -> Any:
        return self.source[name]

    def __len__(self) -> int:
        if not self.source:
            return 0
        return len(next(iter(self.source.values())))


def _two_row_batch() -> _Batch:
    # Row 0 has one supervised token and one masked-out token (so both
    # `continue` legs of the penalty and objective loops execute); row 1 has
    # one supervised token whose current/old gap of 0.1 keeps the ratio
    # exp(0.1) strictly inside the (0.8, 1.2) clip band. References equal the
    # currents, so the folded k1 penalty is exactly zero and the penalised
    # returns equal the rewards (2, 4): population mean 3, std 1.
    return _Batch(
        {
            "rewards": (2.0, 4.0),
            "loss_mask": ((1, 0), (1,)),
            "old_logprobs": ((-1.0, -1.0), (-1.0,)),
            "reference_logprobs": ((-1.0, -1.0), (-0.9,)),
            "current_logprobs": ((-1.0, -1.0), (-0.9,)),
        }
    )


def _forward_from_column(current_batch: _Batch) -> Any:
    return current_batch.column("current_logprobs")


def _valid_setup_kwargs(*, dataloader: list[_Batch]) -> dict[str, Any]:
    return {
        "policy_pair": PolicyPair(train_view=object(), references={"reference": object()}),
        "loss_fn": ReinforcePlusPlusLoss(),
        "dataloader": dataloader,
        "config": {"policy_logprob_column": "current_logprobs"},
    }


def _drifted_role_names(*, requires: Any, supplied: Any, origin: str) -> tuple[str, ...]:
    """Fabricate a consumed-role tuple that provably disagrees with the real maps."""
    return tuple(sorted(set(requires) ^ set(supplied))) + (f"drifted-at-{origin}",)


def test_loss_semantics_declares_token_scope_k1_clip_and_reference_need() -> None:
    """The loss's one-line semantics method returns its five declarations.

    WHAT IS CLAIMED: the returned record carries ``group_size=None`` (global
    normalisation constrains no group seam), token ratio scope, the k1
    estimator, the symmetric (0.8, 1.2) clip bounds derived from the default
    epsilon, and reference dependence.

    WHAT IS NOT CLAIMED: that these declarations agree with the algorithm's
    own; agreement is measured by ``setup`` and is tested separately.
    """
    semantics = ReinforcePlusPlusLoss().semantics()
    assert semantics.group_size is None
    assert semantics.ratio_scope == "token"
    assert semantics.kl_estimator == "k1"
    assert semantics.clip_bounds == (0.8, 1.2)
    assert semantics.reference_free is False


def test_algorithm_semantics_and_requires_accessors_return_declarations() -> None:
    """The one-line ``semantics`` and ``requires`` accessors answer honestly.

    WHAT IS CLAIMED: ``semantics()`` returns the constructor-derived record
    including the declared clip bounds, and ``requires()`` returns an
    immutable map whose reference-policy role is required while the
    advantage-function role is explicitly not; mutation of that map is
    refused by the mapping proxy itself.

    WHAT IS NOT CLAIMED: that the required roles are present; presence is a
    setup-time question and the static denominator never claims it.
    """
    algorithm = ReinforcePlusPlusAlgorithm()
    semantics = algorithm.semantics()
    assert semantics.group_size is None
    assert semantics.ratio_scope == "token"
    assert semantics.kl_estimator == "k1"
    assert semantics.clip_bounds == (0.8, 1.2)
    assert semantics.reference_free is False
    requires = algorithm.requires()
    assert requires["policy_pair"] is True
    assert requires["reference_policy"] is True
    assert requires["advantage_fn"] is False
    assert requires["weight_sync"] is False
    with pytest.raises(TypeError):
        requires["injected_role"] = True


def test_loss_call_surface_returns_only_the_declared_component() -> None:
    """``__call__`` prices a batch exactly as ``compute_with_report`` would.

    Hand computation: the penalised returns are (2, 4) (zero penalty), the
    global z-scores are (-1, +1), the two remaining supervised tokens carry
    ratios 1 and exp(0.1) inside the band, so the contribution is
    -( -1 + exp(0.1) ) / 2 = (1 - exp(0.1)) / 2.

    WHAT IS CLAIMED: the bare call returns a loss whose single component is
    the declared name and whose value matches the hand computation, with the
    masked-out token and the in-band ratio exercising the skip and no-clip
    legs.

    WHAT IS NOT CLAIMED: that the penalised-return statistics survived; the
    protocol surface drops them by design and nothing here checks otherwise.
    """
    output = ReinforcePlusPlusLoss()(_forward_from_column, _two_row_batch())
    expected = (1.0 - math.exp(0.1)) / 2.0
    assert output.loss == pytest.approx(expected)
    assert len(output.components) == 1
    assert output.components[0].name == "reinforce_pp_loss"
    assert output.components[0].contribution == pytest.approx(expected)
    assert output.components[0].observed is True


def test_compute_refuses_missing_required_column_counting_denominators() -> None:
    """An absent input column is refused by name and by count.

    WHAT IS CLAIMED: dropping ``reference_logprobs`` triggers the missing-
    columns refusal, whose message reports 1 missing input of 4 required
    and names reinforce_pp as the family.

    WHAT IS NOT CLAIMED: anything about column content; this guard fires
    before any value is read.
    """
    batch = _Batch(
        {
            "rewards": (1.0, 2.0),
            "loss_mask": ((1,), (1,)),
            "old_logprobs": ((-1.0,), (-1.0,)),
        }
    )
    with pytest.raises(BatchRefusal, match="absent for reinforce_pp") as exc_info:
        ReinforcePlusPlusLoss().compute_with_report(lambda _b: ((-1.0,), (-1.0,)), batch)
    assert "1 of 4 required inputs" in str(exc_info.value)


def test_compute_refuses_zero_offered_rows_as_unmeasured() -> None:
    """Zero rows refuse before any normalisation, never reading as 0.0.

    WHAT IS CLAIMED: a fully present but empty batch is refused when the row
    count is read as zero, with the refusal reporting the 0-of-at-least-2
    shortfall and the unmeasured-normalisation reason.

    WHAT IS NOT CLAIMED: that one row fails the same way; the singleton
    batch takes the distinct 1-of-2 branch already covered by the sibling
    test module.
    """
    batch = _Batch(
        {
            "rewards": (),
            "loss_mask": (),
            "old_logprobs": (),
            "reference_logprobs": (),
        }
    )
    with pytest.raises(
        BatchRefusal,
        match="0 of at least 2 required batch rows were supplied",
    ) as exc_info:
        ReinforcePlusPlusLoss().compute_with_report(lambda _b: (), batch)
    assert "global normalisation over zero samples" in str(exc_info.value)


def test_compute_refuses_column_length_mismatch_naming_both_counts() -> None:
    """One short or long column disagrees with the offered row denominator.

    WHAT IS CLAIMED: a three-row mask column against a two-row batch is
    refused with both counts named, because the per-column loop insists the
    four inputs share the one offered denominator.

    WHAT IS NOT CLAIMED: which column is authoritative; the check is
    symmetric and the batch length is only the first reader's answer.
    """
    batch = _Batch(
        {
            "rewards": (1.0, 2.0),
            "loss_mask": ((1,), (1,), (1,)),
            "old_logprobs": ((-1.0,), (-1.0,)),
            "reference_logprobs": ((-1.0,), (-1.0,)),
        }
    )
    with pytest.raises(
        BatchRefusal,
        match="3 entries were supplied for 2 batch rows",
    ):
        ReinforcePlusPlusLoss().compute_with_report(lambda _b: ((-1.0,), (-1.0,)), batch)


def test_compute_refuses_forward_row_count_mismatch() -> None:
    """A forward that returns fewer rows than the batch is refused.

    WHAT IS CLAIMED: a one-row forward output against a two-row batch is
    refused with both counts in the message, before any token of either row
    is priced.

    WHAT IS NOT CLAIMED: that the forward's values are otherwise sound; the
    count check precedes every per-token conversion.
    """
    batch = _Batch(
        {
            "rewards": (1.0, 3.0),
            "loss_mask": ((1,), (1,)),
            "old_logprobs": ((-1.0,), (-1.0,)),
            "reference_logprobs": ((-1.0,), (-1.0,)),
        }
    )
    with pytest.raises(
        BatchRefusal,
        match="forward_fn returned 1 rows for a batch of 2 rows",
    ) as exc_info:
        ReinforcePlusPlusLoss().compute_with_report(lambda _b: ((-1.0,),), batch)
    assert "one token row per batch row is required" in str(exc_info.value)


def test_compute_refuses_unequal_token_denominators_within_a_row() -> None:
    """Mismatched per-row token lengths refuse before pricing that row.

    WHAT IS CLAIMED: a row whose old-logprob row is shorter than its mask,
    current, and reference rows is refused with all four lengths printed,
    because a token ratio cannot be attributed across unequal denominators.

    WHAT IS NOT CLAIMED: that the other row was defective; row 1 is
    well-formed and never reached.
    """
    batch = _Batch(
        {
            "rewards": (1.0, 3.0),
            "loss_mask": ((1, 1), (1,)),
            "old_logprobs": ((-1.0,), (-1.0,)),
            "reference_logprobs": ((-1.0, -1.0), (-1.0,)),
        }
    )
    with pytest.raises(
        BatchRefusal,
        match="all 4 token denominators must be equal",
    ) as exc_info:
        ReinforcePlusPlusLoss().compute_with_report(
            lambda _b: ((-1.0, -1.0), (-1.0,)),
            batch,
        )
    assert "row 0" in str(exc_info.value)


def test_compute_refuses_non_binary_mask_entry() -> None:
    """A mask entry that is neither 0 nor 1 is refused at its position.

    WHAT IS CLAIMED: the entry ``2`` refuses with the row and position in
    the message, exercising the fall-through leg of the mask parser from
    this module's own call site.

    WHAT IS NOT CLAIMED: that 0.5 floats or string entries take a different
    branch; only the integer out-of-range shape is measured here.
    """
    batch = _Batch(
        {
            "rewards": (1.0, 3.0),
            "loss_mask": ((2,), (1,)),
            "old_logprobs": ((-1.0,), (-1.0,)),
            "reference_logprobs": ((-1.0,), (-1.0,)),
        }
    )
    with pytest.raises(
        BatchRefusal,
        match="supervision mask entries must be 0 or 1",
    ) as exc_info:
        ReinforcePlusPlusLoss().compute_with_report(
            lambda _b: ((-1.0,), (-1.0,)),
            batch,
        )
    assert "row 0, position 0" in str(exc_info.value)


def test_compute_refuses_zero_supervised_tokens() -> None:
    """A fully masked-out batch is unmeasured, not a perfect-looking 0.0.

    WHAT IS CLAIMED: with every token masked the penalty loop contributes
    nothing (the skip leg runs for all four entries), the penalised returns
    keep nonzero spread, and the refusal names the zero supervised count
    against the two offered rows.

    WHAT IS NOT CLAIMED: that partially masked batches fail; only total
    absence of supervision is measured here.
    """
    batch = _Batch(
        {
            "rewards": (1.0, 3.0),
            "loss_mask": ((0, 0), (0, 0)),
            "old_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
            "reference_logprobs": ((-1.0, -1.0), (-1.0, -1.0)),
        }
    )
    with pytest.raises(
        SupervisionRefusal,
        match="0 supervised tokens remain across 2 rows",
    ) as exc_info:
        ReinforcePlusPlusLoss().compute_with_report(
            lambda _b: ((-1.0, -1.0), (-1.0, -1.0)),
            batch,
        )
    assert "unmeasured" in str(exc_info.value)


def test_low_clip_bound_leg_wins_by_advantage_sign() -> None:
    """Below the low bound, the losing PPO leg flips with the advantage sign.

    Hand computation: penalised returns (2, 4) give z-scores (-1, +1);
    every ratio is exp(-1) = 0.3678 below the 0.8 bound. For row 0
    (A = -1) the legs are -0.3678 vs -0.8, so the clipped leg wins; for
    row 1 (A = +1) they are 0.3678 vs 0.8, so the raw leg wins. The total
    is -0.8 + exp(-1), the contribution (0.8 - exp(-1)) / 2.

    WHAT IS CLAIMED: the returned loss equals that value, proving the bound
    clamped row 0 and let row 1 through unclipped.

    WHAT IS NOT CLAIMED: equivalence with any reference PPO implementation;
    symmetric bounds are this binding's declared choice.
    """
    batch = _Batch(
        {
            "rewards": (2.0, 4.0),
            "loss_mask": ((1,), (1,)),
            "old_logprobs": ((0.0,), (0.0,)),
            "reference_logprobs": ((-1.0,), (-1.0,)),
        }
    )
    output, stats = ReinforcePlusPlusLoss().compute_with_report(
        lambda _b: ((-1.0,), (-1.0,)),
        batch,
    )
    assert stats.mean == pytest.approx(3.0)
    assert output.loss == pytest.approx((0.8 - math.exp(-1.0)) / 2.0)


def test_high_clip_bound_leg_wins_by_advantage_sign() -> None:
    """Above the high bound, the losing PPO leg flips the opposite way.

    Hand computation: z-scores (-1, +1); every ratio is exp(1) = 2.718
    above the 1.2 bound. For row 0 (A = -1) the legs are -2.718 vs -1.2,
    so the raw leg wins; for row 1 (A = +1) they are 2.718 vs 1.2, so the
    clipped leg wins. The total is 1.2 - exp(1), the contribution
    (exp(1) - 1.2) / 2.

    WHAT IS CLAIMED: the returned loss equals that value, proving the clamp
    bound only the row whose advantage sign makes clipping conservative.

    WHAT IS NOT CLAIMED: that ratios this large are expected in practice;
    the test measures the branch, not its frequency.
    """
    batch = _Batch(
        {
            "rewards": (2.0, 4.0),
            "loss_mask": ((1,), (1,)),
            "old_logprobs": ((0.0,), (0.0,)),
            "reference_logprobs": ((1.0,), (1.0,)),
        }
    )
    output, _stats = ReinforcePlusPlusLoss().compute_with_report(
        lambda _b: ((1.0,), (1.0,)),
        batch,
    )
    assert output.loss == pytest.approx((math.exp(1.0) - 1.2) / 2.0)


def test_setup_wires_and_first_step_matches_hand_computation() -> None:
    """A fully wired step prices, reports the offered rows, and verifies.

    Hand computation as in the call-surface test: contribution
    (1 - exp(0.1)) / 2 over 2 supervised tokens, with one masked-out token
    skipped by the inner ``forward_fn`` closure the step builds.

    WHAT IS CLAIMED: setup succeeds with a reference-carrying pair, the
    step's forward closure reads exactly the configured column, the report
    carries step 0, ``rows`` equal to the offered batch length (no
    compaction), no reward statistics, and the declared/observed surface
    passes ``verify_step``; the supplied mapping afterwards names the
    reference role.

    WHAT IS NOT CLAIMED: that the reference attestation means the reference
    produced the penalty's log-probabilities; the pair only attests
    presence.
    """
    algorithm = ReinforcePlusPlusAlgorithm()
    algorithm.setup(**_valid_setup_kwargs(dataloader=[_two_row_batch()]))
    report = algorithm.step()
    assert report.step == 0
    assert report.rows == 2
    assert report.reward_stats is None
    assert report.loss.loss == pytest.approx((1.0 - math.exp(0.1)) / 2.0)
    assert verify_step(report, algorithm.requirements()) == 1
    supplied = algorithm.supplied()
    assert supplied is not None
    assert "reference_policy" in supplied


def test_step_refuses_before_setup_with_zero_of_four_counts() -> None:
    """A pre-setup step refuses rather than reporting an unmeasured step.

    WHAT IS CLAIMED: with no wiring present the step refuses naming
    0-of-4 present inputs, and ``supplied()`` abstains as ``None`` before
    any of that is attempted.

    WHAT IS NOT CLAIMED: that the refusal says the wiring failed; it says
    the wiring was never measured.
    """
    algorithm = ReinforcePlusPlusAlgorithm()
    assert algorithm.supplied() is None
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="0 of 4 required inputs are present for reinforce_pp",
    ):
        algorithm.step()


def test_step_refuses_exhausted_dataloader_after_successful_step() -> None:
    """A drained dataloader is an unmeasured step, not a zero-row one.

    WHAT IS CLAIMED: after one valid step the second ``step`` call
    translates ``StopIteration`` into a StepReportRefusal naming the
    0-of-at-least-1 batch shortfall, and the first step's step index had
    already advanced the counter, so the refusal carries no half-built
    report.

    WHAT IS NOT CLAIMED: anything about epoch semantics or recycling; the
    binding consumes the iterator once.
    """
    algorithm = ReinforcePlusPlusAlgorithm()
    algorithm.setup(**_valid_setup_kwargs(dataloader=[_two_row_batch()]))
    first = algorithm.step()
    assert first.step == 0
    with pytest.raises(
        StepReportRefusal,
        match="0 of at least 1 required batches remain for reinforce_pp",
    ):
        algorithm.step()


def test_setup_refuses_a_second_wiring() -> None:
    """Calling setup on an already-wired algorithm refuses at the gate.

    WHAT IS CLAIMED: the second call refuses naming the already-wired
    object, before any of the body checks could replace the measured
    supplied mapping.

    WHAT IS NOT CLAIMED: that the first wiring is somehow invalidated by
    the attempt; the refusal leaves it untouched and nothing here re-steps.
    """
    kwargs = _valid_setup_kwargs(dataloader=[_two_row_batch()])
    algorithm = ReinforcePlusPlusAlgorithm()
    algorithm.setup(**kwargs)
    with pytest.raises(AlgorithmWiringRefusal, match="was already wired") as exc_info:
        algorithm.setup(**kwargs)
    assert "ReinforcePlusPlusAlgorithm" in str(exc_info.value)


def test_setup_refuses_a_loss_of_the_wrong_binding() -> None:
    """A non-ReinforcePlusPlusLoss object cannot occupy the loss slot.

    WHAT IS CLAIMED: an arbitrary object in the loss slot refuses with the
    required binding name and the 1-of-1 slot denominator.

    WHAT IS NOT CLAIMED: anything about the wrong object's own validity;
    only its type is read.
    """
    algorithm = ReinforcePlusPlusAlgorithm()
    kwargs = _valid_setup_kwargs(dataloader=[_two_row_batch()])
    kwargs["loss_fn"] = object()
    pattern = re.escape("Reinforce++ requires ReinforcePlusPlusLoss")
    with pytest.raises(AlgorithmWiringRefusal, match=pattern) as exc_info:
        algorithm.setup(**kwargs)
    assert "1 of 1 loss slots carries object" in str(exc_info.value)


def test_setup_refuses_semantics_disagreement_naming_field() -> None:
    """Two independently declared clip bounds that differ refuse at setup.

    WHAT IS CLAIMED: an algorithm at the default epsilon wired to a loss
    declared at epsilon 0.3 refuses inside the five-field comparison loop,
    naming the clip_bounds field and both declared tuples, exactly so the
    two declarations cannot silently drift.

    WHAT IS NOT CLAIMED: which side is right; only that they disagree, and
    that the refusal precedes any wiring check.
    """
    algorithm = ReinforcePlusPlusAlgorithm()
    kwargs = _valid_setup_kwargs(dataloader=[_two_row_batch()])
    kwargs["loss_fn"] = ReinforcePlusPlusLoss(clip_epsilon=0.3)
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="1 of 5 semantics fields disagrees",
    ) as exc_info:
        algorithm.setup(**kwargs)
    assert "field clip_bounds" in str(exc_info.value)


def test_setup_refuses_non_iterable_dataloader() -> None:
    """A dataloader that is not iterable refuses after the wiring check.

    WHAT IS CLAIMED: a bare object in the dataloader slot refuses with its
    type named, exercising the ``iter`` guard that stands between the
    role-map check and the record of the iterator.

    WHAT IS NOT CLAIMED: that an iterable-but-empty dataloader fails here;
    exhausted data is a step-time refusal, not a setup one.
    """
    algorithm = ReinforcePlusPlusAlgorithm()
    kwargs = _valid_setup_kwargs(dataloader=[_two_row_batch()])
    kwargs["dataloader"] = object()
    with pytest.raises(AlgorithmWiringRefusal, match="was not iterable") as exc_info:
        algorithm.setup(**kwargs)
    assert "1 of 1 mandatory step inputs" in str(exc_info.value)


def test_setup_refuses_non_mapping_config() -> None:
    """A setup configuration that is not a Mapping refuses by name.

    WHAT IS CLAIMED: a list in the config slot refuses with the Mapping
    requirement stated, exercising the guard before any key is read from
    the configuration.

    WHAT IS NOT CLAIMED: that an empty mapping fails; emptiness is caught
    by the column-name guard, which is the next refusal in sequence.
    """
    algorithm = ReinforcePlusPlusAlgorithm()
    kwargs = _valid_setup_kwargs(dataloader=[_two_row_batch()])
    kwargs["config"] = ["policy_logprob_column"]
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="setup configurations must implement Mapping",
    ):
        algorithm.setup(**kwargs)


def test_setup_refuses_absent_or_empty_policy_logprob_column() -> None:
    """No non-empty column name in config is refused with its 0-of-1 count.

    WHAT IS CLAIMED: both an empty config and one mapping the key to the
    empty string trigger the same refusal naming reinforce_pp, because
    current log-probabilities cannot be attributed without that column.

    WHAT IS NOT CLAIMED: that any particular name is the right one; only
    non-emptiness is measured at setup.
    """
    for config in ({}, {"policy_logprob_column": ""}):
        algorithm = ReinforcePlusPlusAlgorithm()
        kwargs = _valid_setup_kwargs(dataloader=[_two_row_batch()])
        kwargs["config"] = config
        with pytest.raises(
            AlgorithmWiringRefusal,
            match="0 of 1 required config values names a non-empty batch column",
        ):
            algorithm.setup(**kwargs)


def test_setup_refuses_drift_between_the_two_role_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the family's role-map helper ever drifted, setup must refuse.

    The disagreement branch cannot be reached with the real helper under
    any valid input -- the two checks measure the same union by
    construction -- so this test injects the drift directly: a stub helper
    whose consumed set provably differs from the existing handshake's, then
    asserts setup refuses naming both denominators rather than wiring on a
    disagreement it was built to catch.

    WHAT IS CLAIMED: the refusal message reports the drifted mapped roles,
    the handshake roles, and the "same consumed set" invariant.

    WHAT IS NOT CLAIMED: that the real helper can produce this drift; the
    branch is defensive against a future edit, and this test measures only
    that the tripwire fires.
    """
    monkeypatch.setattr(
        policy_gradient,
        "check_reinforce_pp_requirements",
        _drifted_role_names,
    )
    algorithm = ReinforcePlusPlusAlgorithm()
    with pytest.raises(
        AlgorithmWiringRefusal,
        match="must name the same consumed set",
    ) as exc_info:
        algorithm.setup(**_valid_setup_kwargs(dataloader=[_two_row_batch()]))
    assert "drifted-at-reinforce_pp" in str(exc_info.value)
