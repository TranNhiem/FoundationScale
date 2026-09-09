"""Tests for the GSPO, Dr.GRPO, and DAPO loss objects.

WHAT IS CLAIMED: these tests pin the declared axes matrix for the three
group-relative objectives (ratio scope, reduction, clip bounds, advantage
estimator binding), every construction and batch refusal the module raises,
and the forward arithmetic on hand-built batches small enough to check by
reading.

WHAT IS NOT CLAIMED: no training-quality, convergence, or
equivalence-to-the-paper property is tested here; dynamic sampling is
checked only as a declared (and unenforced) property of DAPO.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from foundationscale.rl.advantage import CentredAdvantage, GroupNormalisedAdvantage
from foundationscale.rl.group_policy_objectives import (
    DAPOLoss,
    DrGRPOLoss,
    GSPOLoss,
)
from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    LossConfigRefusal,
)


def _make_batch(columns: dict[str, list[Any]]) -> ExperienceBatch:
    return ExperienceBatch(columns=columns)


def _two_group_batch(**overrides: Any) -> ExperienceBatch:
    """Two prompts, two rows each, fully supervised, unit-length tokens."""
    columns: dict[str, list[Any]] = {
        "prompt_ids": ["p", "p", "q", "q"],
        "rewards": [1.0, 0.0, 1.0, 0.0],
        "loss_mask": [[1, 1], [1, 1], [1, 1], [1, 1]],
        "old_logprobs": [[-0.5, -0.5]] * 4,
        "reference_logprobs": [[-0.55, -0.55]] * 4,
    }
    columns.update(overrides)
    return _make_batch(columns)


def _forward(batch: ExperienceBatch) -> list[list[float]]:
    del batch
    return [[-0.5, -0.5]] * 4


# ---------------------------------------------------------------------------
# Declared semantics, columns, and declarations
# ---------------------------------------------------------------------------


def test_gspo_semantics_and_declaration() -> None:
    loss = GSPOLoss()
    semantics = loss.semantics()
    assert semantics.group_size == 2
    assert semantics.ratio_scope == "sequence"
    assert semantics.kl_estimator is None
    assert semantics.clip_bounds == (0.9997, 1.0003)
    assert semantics.reference_free is True
    assert loss.expects_dynamic_sampling is False
    assert loss.required_columns == (
        "prompt_ids",
        "rewards",
        "loss_mask",
        "old_logprobs",
    )
    assert loss.declaration().components == ("gspo_policy_loss",)


def test_gspo_semantics_with_active_kl() -> None:
    # An active k3 term must surface the reference column and estimator.
    loss = GSPOLoss(kl_weight=0.5)
    assert loss.semantics().kl_estimator == "k3"
    assert loss.semantics().reference_free is False
    assert "reference_logprobs" in loss.required_columns
    assert loss.declaration().components == ("gspo_policy_loss", "gspo_kl")


def test_drgrpo_semantics_and_declaration() -> None:
    loss = DrGRPOLoss(constant_length=8)
    semantics = loss.semantics()
    assert semantics.group_size == 2
    assert semantics.ratio_scope == "token"
    assert semantics.kl_estimator is None
    assert semantics.clip_bounds == (0.8, 1.2)
    assert semantics.reference_free is True
    assert loss.expects_dynamic_sampling is False
    assert loss.required_columns == (
        "prompt_ids",
        "rewards",
        "loss_mask",
        "old_logprobs",
    )
    assert loss.declaration().components == ("drgrpo_policy_loss",)


def test_drgrpo_semantics_with_active_kl() -> None:
    loss = DrGRPOLoss(constant_length=8, kl_weight=0.1)
    assert loss.semantics().kl_estimator == "k3"
    assert loss.semantics().reference_free is False
    assert loss.required_columns[-1] == "reference_logprobs"
    assert loss.declaration().components == ("drgrpo_policy_loss", "drgrpo_kl")


def test_dapo_semantics_and_declaration() -> None:
    loss = DAPOLoss()
    semantics = loss.semantics()
    assert semantics.group_size == 2
    assert semantics.ratio_scope == "token"
    assert semantics.kl_estimator is None
    assert semantics.clip_bounds == (0.8, 1.28)
    assert semantics.reference_free is True
    # The declared (unenforced) rollout expectation is the honest hook.
    assert loss.expects_dynamic_sampling is True
    assert loss.required_columns == (
        "prompt_ids",
        "rewards",
        "loss_mask",
        "old_logprobs",
    )
    assert loss.declaration().components == ("dapo_policy_loss",)


# ---------------------------------------------------------------------------
# Axis properties: the load-bearing declarations of the axes matrix
# ---------------------------------------------------------------------------


def test_gspo_axes() -> None:
    loss = GSPOLoss()
    assert loss.ratio_scope == "sequence"
    assert loss.reduction == "sequence_mean"
    assert loss.clip_bounds == (0.9997, 1.0003)
    assert isinstance(loss.advantage_fn, GroupNormalisedAdvantage)


def test_gspo_clip_interval_is_narrower_than_token_intervals() -> None:
    # The module pins the SCALE of the default, not the exact value.
    loss = GSPOLoss()
    low, high = loss.clip_bounds
    assert high - low < 1.2 - 0.8


def test_drgrpo_axes() -> None:
    loss = DrGRPOLoss()
    assert loss.ratio_scope == "token"
    assert loss.reduction == "constant"
    assert loss.clip_bounds == (0.8, 1.2)
    # Centred only: mean-subtracted without std normalisation is measured.
    assert isinstance(loss.advantage_fn, CentredAdvantage)


def test_dapo_axes() -> None:
    loss = DAPOLoss()
    assert loss.ratio_scope == "token"
    assert loss.reduction == "token_mean"
    # The only asymmetric interval in the family is load-bearing.
    low, high = loss.clip_bounds
    assert low == 0.8
    assert high == 1.28
    assert high - 1.0 != 1.0 - low
    assert isinstance(loss.advantage_fn, GroupNormalisedAdvantage)


# ---------------------------------------------------------------------------
# Construction refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss, DAPOLoss])
def test_group_size_below_two_refused(cls: type) -> None:
    with pytest.raises(LossConfigRefusal, match="group_size=1"):
        cls(group_size=1)


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss, DAPOLoss])
def test_bool_group_size_refused(cls: type) -> None:
    with pytest.raises(LossConfigRefusal, match="True is not"):
        cls(group_size=True)


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss, DAPOLoss])
def test_unordered_clip_bounds_refused(cls: type) -> None:
    with pytest.raises(LossConfigRefusal, match="strictly ordered"):
        cls(clip_low=1.1, clip_high=0.9)


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss, DAPOLoss])
def test_non_positive_lower_clip_refused(cls: type) -> None:
    with pytest.raises(LossConfigRefusal, match="clip_bounds"):
        cls(clip_low=0.0, clip_high=1.5)


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss, DAPOLoss])
def test_non_finite_clip_bound_refused(cls: type) -> None:
    with pytest.raises(LossConfigRefusal, match="clip_high"):
        cls(clip_high=math.inf)


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss, DAPOLoss])
def test_bool_clip_bound_refused(cls: type) -> None:
    with pytest.raises(LossConfigRefusal, match="clip_low"):
        cls(clip_low=True)


@pytest.mark.parametrize("bad_weight", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss, DAPOLoss])
def test_non_positive_or_non_finite_weight_refused(cls: type, bad_weight: float) -> None:
    with pytest.raises(LossConfigRefusal, match="positive finite weight"):
        cls(weight=bad_weight)


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss])
def test_negative_kl_weight_refused(cls: type) -> None:
    with pytest.raises(LossConfigRefusal, match="kl_weight"):
        cls(kl_weight=-0.1)


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss])
def test_active_kl_without_named_reference_refused(cls: type) -> None:
    # Two declared fields disagree about whether a reference exists.
    with pytest.raises(LossConfigRefusal, match="reference_logprob_column"):
        cls(kl_weight=0.1, reference_logprob_column="")


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss])
def test_shared_component_names_refused(cls: type) -> None:
    with pytest.raises(LossConfigRefusal, match="collapse the objective denominator"):
        cls(component_name="x", kl_component_name="x")


@pytest.mark.parametrize("cls", [GSPOLoss, DrGRPOLoss, DAPOLoss])
def test_empty_component_name_refused(cls: type) -> None:
    with pytest.raises(LossConfigRefusal, match="component_name"):
        cls(component_name="")


def test_wrong_estimator_refused() -> None:
    # A normalised substitute would silently become GRPO; it must be refused.
    with pytest.raises(LossConfigRefusal, match="CentredAdvantage"):
        DrGRPOLoss(advantage_fn=GroupNormalisedAdvantage())


def test_estimator_min_group_size_mismatch_refused() -> None:
    with pytest.raises(LossConfigRefusal, match="estimator minimum"):
        GSPOLoss(group_size=4, advantage_fn=GroupNormalisedAdvantage(min_group_size=2))


@pytest.mark.parametrize("bad_length", [0, -3, True, 4.5])
def test_drgrpo_constant_length_refused(bad_length: Any) -> None:
    with pytest.raises(LossConfigRefusal, match="constant_length"):
        DrGRPOLoss(constant_length=bad_length)


def test_dapo_has_no_kl_field() -> None:
    # DAPO omits kl_weight by design; setting it must be unrepresentable.
    with pytest.raises(TypeError, match="kl_weight"):
        DAPOLoss(kl_weight=0.1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Batch refusals through the shared pricing pipeline
# ---------------------------------------------------------------------------


def test_missing_required_column_refused() -> None:
    loss = GSPOLoss()
    batch = _two_group_batch()
    batch.columns.pop("old_logprobs")  # type: ignore[union-attr]
    batch = _make_batch(
        {
            "prompt_ids": ["p", "p", "q", "q"],
            "rewards": [1.0, 0.0, 1.0, 0.0],
            "loss_mask": [[1], [1], [1], [1]],
        }
    )
    with pytest.raises(BatchRefusal, match="old_logprobs"):
        loss(_forward, batch)


def test_empty_batch_refused() -> None:
    loss = GSPOLoss()
    batch = _make_batch({name: [] for name in loss.required_columns})
    with pytest.raises(BatchRefusal, match="0 of at least 1"):
        loss(lambda b: [], batch)


def test_group_smaller_than_declared_refused() -> None:
    # One prompt group of a single row cannot honour the declared K.
    loss = GSPOLoss(group_size=2)
    batch = _two_group_batch(prompt_ids=["p", "q", "q", "q"])
    with pytest.raises(BatchRefusal, match="group_size=2"):
        loss(_forward, batch)


def test_non_finite_old_logprob_refused() -> None:
    loss = GSPOLoss()
    batch = _two_group_batch(
        old_logprobs=[[math.nan, -0.5]] + [[-0.5, -0.5]] * 3,
    )
    with pytest.raises(BatchRefusal, match="not finite"):
        loss(_forward, batch)


def test_forward_row_count_mismatch_refused() -> None:
    loss = GSPOLoss()
    with pytest.raises(BatchRefusal, match="returned 3 rows"):
        loss(lambda b: [[-0.5, -0.5]] * 3, _two_group_batch())


def test_token_length_disagreement_refused() -> None:
    loss = GSPOLoss()
    batch = _two_group_batch(old_logprobs=[[-0.5], [-0.5], [-0.5], [-0.5]])
    with pytest.raises(BatchRefusal, match="token denominators"):
        loss(_forward, batch)


# ---------------------------------------------------------------------------
# Supervision refusals
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ratio and k3 guards
# ---------------------------------------------------------------------------


def test_sequence_ratio_underflow_refused() -> None:
    # exp of a sufficiently negative mean underflows to a silent zero.
    loss = GSPOLoss()
    batch = _two_group_batch()

    def tiny(b: ExperienceBatch) -> list[list[float]]:
        del b
        return [[-800.0, -800.0]] * 4

    with pytest.raises(BatchRefusal, match="the response"):
        loss(tiny, batch)


# ---------------------------------------------------------------------------
# Forward paths: declared components and finite losses
# ---------------------------------------------------------------------------


def test_gspo_forward_loss_finite_and_declared() -> None:
    loss = GSPOLoss()
    output = loss(_forward, _two_group_batch())
    assert isinstance(output.loss, float)
    assert math.isfinite(output.loss)
    assert [c.name for c in output.components] == ["gspo_policy_loss"]
    assert output.components[0].weight == 1.0
    assert output.components[0].observed is True


def test_gspo_forward_with_active_kl_declares_two_components() -> None:
    loss = GSPOLoss(kl_weight=0.5)
    batch = _two_group_batch()
    output, advantage = loss.compute_with_report(_forward, batch)
    assert math.isfinite(output.loss)
    assert [c.name for c in output.components] == ["gspo_policy_loss", "gspo_kl"]
    assert output.loss == pytest.approx(sum(c.contribution for c in output.components))
    assert advantage.used == 4


def test_gspo_report_is_identical_through_call() -> None:
    loss = GSPOLoss()
    batch = _two_group_batch()
    direct = loss(_forward, batch)
    reported, _ = loss.compute_with_report(_forward, batch)
    assert direct.loss == pytest.approx(reported.loss)


def test_drgrpo_forward_loss_finite_and_declared() -> None:
    loss = DrGRPOLoss(constant_length=2)
    output = loss(_forward, _two_group_batch())
    assert math.isfinite(output.loss)
    assert [c.name for c in output.components] == ["drgrpo_policy_loss"]


def test_drgrpo_keeps_zero_variance_groups() -> None:
    # Centred advantage keeps a zero-variance group; a normalised one
    # would drop it.  Equal rewards in group "q" exercise exactly that.
    loss = DrGRPOLoss(constant_length=2)
    batch = _two_group_batch(rewards=[1.0, 0.0, 0.5, 0.5])
    output, advantage = loss.compute_with_report(_forward, batch)
    assert math.isfinite(output.loss)
    assert advantage.used == advantage.offered == 4


def test_drgrpo_forward_with_active_kl_declares_two_components() -> None:
    loss = DrGRPOLoss(constant_length=2, kl_weight=0.2)
    output = loss(_forward, _two_group_batch())
    assert math.isfinite(output.loss)
    assert [c.name for c in output.components] == ["drgrpo_policy_loss", "drgrpo_kl"]


def test_dapo_forward_loss_finite_and_declared() -> None:
    loss = DAPOLoss()
    output = loss(_forward, _two_group_batch())
    assert math.isfinite(output.loss)
    assert [c.name for c in output.components] == ["dapo_policy_loss"]
    assert len(output.components) == 1


def test_dapo_drops_zero_variance_group_at_the_filter_half() -> None:
    # GroupNormalisedAdvantage drops "q" for zero variance; used < offered
    # is the filter half of dynamic sampling and must be visible.
    loss = DAPOLoss()
    batch = _two_group_batch(rewards=[1.0, 0.0, 0.5, 0.5])
    output, advantage = loss.compute_with_report(_forward, batch)
    assert math.isfinite(output.loss)
    assert advantage.used < advantage.offered


def test_ratio_one_yields_expected_drgrpo_value() -> None:
    # With current == old the ratio is exactly 1, so the surrogate is the
    # centred advantage per supervised token over rows*constant_length.
    loss = DrGRPOLoss(constant_length=2)
    output = loss(_forward, _two_group_batch())
    # Centred advantages: group p -> (+0.5, -0.5), group q -> (+0.5, -0.5);
    # summed over ratio-1 surrogates they cancel to zero.
    assert output.loss == pytest.approx(0.0)


def test_weight_scales_the_policy_contribution() -> None:
    loss_one = GSPOLoss(weight=2.0)
    loss_two = GSPOLoss(weight=1.0)
    batch = _two_group_batch(rewards=[1.0, 0.0, 1.0, 0.0], loss_mask=[[1, 1]] * 4)

    def shifted(b: ExperienceBatch) -> list[list[float]]:
        del b
        return [[-0.499, -0.5]] * 4

    out_one = loss_one(shifted, batch)
    out_two = loss_two(shifted, batch)
    assert out_one.loss == pytest.approx(2.0 * out_two.loss)
