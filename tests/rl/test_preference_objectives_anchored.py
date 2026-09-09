"""Adversarial tests for the reference-anchored preference objectives: IPO and KTO.

Every leg prices IPO's squared hinge or KTO's two sigmoid branches from a
batch built by hand, with the arithmetic written out where the number is
pinned, so a log-sigmoid IPO, a dropped ``not``, a constant-folded ``tau`` /
``beta`` / ``z0``, or a swapped chosen/rejected all fail loudly. ``z0``'
'

What is pinned here on purpose: the squared shape, the ``1 / (2 * tau)``
offset, accuracy's strict-greater-than tie handling, the documented ignore
of non-finite readings at masked positions, and the KNOWN LIMIT that KTO's
paired-batch refusal keys only on its configured marker names.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from foundationscale.rl import (
    BatchRefusal,
    ExperienceBatch,
    ForwardFn,
    IPOLoss,
    KTOLoss,
    LossConfigRefusal,
    SupervisionRefusal,
)

_IPO_NAME_FIELDS = (
    "chosen_mask_column",
    "rejected_mask_column",
    "reference_chosen_column",
    "reference_rejected_column",
    "component_name",
    "accuracy_metric_name",
    "margin_metric_name",
)

_KTO_NAME_FIELDS = (
    "mask_column",
    "reference_column",
    "kl_reference_point_column",
    "desirable_column",
    "component_name",
    "desirable_count_metric_name",
    "undesirable_count_metric_name",
)

# KTO working point used throughout: beta=0.1, policy sum -1.0, reference -2.0,
# so r = 0.1 * ((-1.0) - (-2.0)) = 0.1, and z0 = 0.3.
# desirable   -> sigmoid(0.3 - 0.1) = 1 / (1 + e^-0.2) = 0.5498339973124780
# undesirable -> sigmoid(0.1 - 0.3) = 1 / (1 + e^0.2)  = 0.4501660026875222
_KTO_DESIRABLE_LOSS = 1.0 / (1.0 + math.exp(-0.2))
_KTO_UNDESIRABLE_LOSS = 1.0 / (1.0 + math.exp(0.2))


def _forward(rows: Any) -> ForwardFn:
    materialised = list(rows)
    return lambda batch: list(materialised)


def _ipo_batch(
    chosen_masks: Any = ((1,),),
    rejected_masks: Any = ((1,),),
    ref_chosen: Any = (0.0,),
    ref_rejected: Any = (0.0,),
) -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "chosen_loss_mask": chosen_masks,
            "rejected_loss_mask": rejected_masks,
            "reference_chosen_logprob": ref_chosen,
            "reference_rejected_logprob": ref_rejected,
        }
    )


def _kto_batch(
    masks: Any = ((1,),),
    refs: Any = (-2.0,),
    kl_points: Any = (0.3,),
    flags: Any = (True,),
    extra_columns: Any = None,
    drop_columns: tuple[str, ...] = (),
) -> ExperienceBatch:
    columns: dict[str, Any] = {
        "loss_mask": masks,
        "reference_logprob": refs,
        "kl_reference_point": kl_points,
        "is_desirable": flags,
    }
    if extra_columns is not None:
        columns.update(extra_columns)
    for name in drop_columns:
        columns.pop(name)
    return ExperienceBatch(columns=columns)


def _pct_plus_one_batch() -> ExperienceBatch:
    # One positive-margin pair and one negative-margin pair. Row 0:
    # pi_c = -0.5, pi_r = -1.5, refs 0, so h0 = (-0.5) - (-1.5) = 1.0.
    # Row 1: pi_c = -1.5, pi_r = -1.0, so h1 = (-1.0) wait: h1 = (-1.5) - (-1.0)
    # = -0.5.
    return _ipo_batch(
        chosen_masks=((1,), (1,)),
        rejected_masks=((1,), (1,)),
        ref_chosen=(0.0, 0.0),
        ref_rejected=(0.0, 0.0),
    )


def _pct_plus_one_rows() -> Any:
    return [((-0.5,), (-1.5,)), ((-1.5,), (-1.0,))]


def _margin_one_batch() -> ExperienceBatch:
    # Single pair: pi_c = -0.5, pi_r = -1.5, refs 0 -> h = 1.0.
    return _ipo_batch()


def _margin_one_rows() -> Any:
    return [((-0.5,), (-1.5,))]


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_preference_objectives_are_reexported_from_the_public_path() -> None:
    import foundationscale.rl as rl_public
    import foundationscale.rl.preference_objectives as preference_module

    assert rl_public.IPOLoss is preference_module.IPOLoss
    assert rl_public.KTOLoss is preference_module.KTOLoss


# ---------------------------------------------------------------------------
# IPO: construction refusals, one leg each
# ---------------------------------------------------------------------------


def test_ipo_tau_zero_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        IPOLoss(tau=0.0)
    assert "tau=0.0" in str(exc_info.value)


def test_ipo_tau_non_finite_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        IPOLoss(tau=math.inf)
    assert "tau=inf" in str(exc_info.value)


def test_ipo_tau_negative_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        IPOLoss(tau=-0.25)
    assert "tau=-0.25" in str(exc_info.value)


def test_ipo_weight_zero_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        IPOLoss(weight=0.0)
    assert "weight=0.0" in str(exc_info.value)


def test_ipo_margin_metric_ceiling_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        IPOLoss(margin_metric_ceiling=0.0)
    assert "margin_metric_ceiling=0.0" in str(exc_info.value)


@pytest.mark.parametrize("field", _IPO_NAME_FIELDS)
def test_ipo_every_name_field_refuses_an_empty_string(field: str) -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        IPOLoss(**{field: ""})
    message = str(exc_info.value)
    assert field in message
    assert "not a name" in message


@pytest.mark.parametrize("field", _IPO_NAME_FIELDS)
def test_ipo_every_name_field_refuses_a_non_string(field: str) -> None:
    constructor_arguments: Any = {field: 7}
    with pytest.raises(LossConfigRefusal) as exc_info:
        IPOLoss(**constructor_arguments)
    assert field in str(exc_info.value)


# ---------------------------------------------------------------------------
# IPO: numerics -- the squared shape and the offset
# ---------------------------------------------------------------------------


def test_ipo_is_a_squared_loss_not_a_log_sigmoid() -> None:
    # h0 = 1.0, h1 = -0.5, tau = 1.0 so the offset is 1 / (2 * 1) = 0.5.
    # Squared:  ((1.0 - 0.5)^2 + (-0.5 - 0.5)^2) / 2 = (0.25 + 1.0) / 2 = 0.625.
    # A log-sigmoid mis-implementation would report
    # (-log s(1.0) - log s(-0.5)) / 2 = (0.31326169 + 0.97407699) / 2
    # = 0.64366934... -- distinguishable at this tolerance, so a sigmoid
    # substitution cannot pass.
    out = IPOLoss()(_forward(_pct_plus_one_rows()), _pct_plus_one_batch())
    assert out.loss == pytest.approx(0.625, rel=1e-12)
    assert out.loss != pytest.approx(0.64366934, rel=1e-6)


def test_ipo_tau_offset_moves_the_loss_by_the_formula_amount() -> None:
    # h = 1.0. At tau = 1.0 the offset is 0.5 and the loss is (1.0-0.5)^2 = 0.25.
    # At tau = 2.0 the offset is 0.25 and the loss is (1.0-0.25)^2 = 0.5625.
    # Difference: 0.3125. A constant-folded tau (or an offset of 1/(2*beta)
    # read from anywhere else) fails one of the two pins.
    rows = _forward(_margin_one_rows())
    batch = _margin_one_batch()
    loss_at_tau_one = IPOLoss(tau=1.0)(rows, batch).loss
    loss_at_tau_two = IPOLoss(tau=2.0)(rows, batch).loss
    assert loss_at_tau_one == pytest.approx(0.25, rel=1e-12)
    assert loss_at_tau_two == pytest.approx(0.5625, rel=1e-12)
    assert loss_at_tau_two - loss_at_tau_one == pytest.approx(0.3125, rel=1e-12)
    assert loss_at_tau_two > loss_at_tau_one


def test_ipo_weight_scales_the_mean_squared_loss() -> None:
    # h = 1.0 -> squared pair loss 0.25; weight 2.0 -> 0.5.
    out = IPOLoss(weight=2.0)(_forward(_margin_one_rows()), _margin_one_batch())
    assert out.loss == pytest.approx(0.5, rel=1e-12)
    assert out.components[0].weight == 2.0
    assert out.components[0].contribution == pytest.approx(0.5, rel=1e-12)


def test_ipo_sums_not_means_the_token_log_probabilities() -> None:
    # Chosen side duplicated: pi_c = -0.5 + -0.5 = -1.0, pi_r = -1.0, so h = 0.0
    # under a SUM. A length-normalised formulation would give h = 0.5 and a
    # different loss: this pins the sum reduction.
    rows = [((-0.5, -0.5), (-1.0,))]
    batch = _ipo_batch(
        chosen_masks=((1, 1),),
        rejected_masks=((1,),),
        ref_chosen=(0.0,),
        ref_rejected=(0.0,),
    )
    out = IPOLoss()(_forward(rows), batch)
    # (0.0 - 0.5)^2 = 0.25
    assert out.loss == pytest.approx(0.25, rel=1e-12)
    assert out.metrics[1].value == 0.0


def test_ipo_accuracy_counts_a_tie_as_incorrect() -> None:
    # Both sides score -0.5 with zero reference -> h = 0.0 exactly. A `>=`
    # mutation reads this as correct; the strict `>` contract reads 0.0.
    rows = [((-0.5,), (-0.5,))]
    out = IPOLoss()(_forward(rows), _ipo_batch())
    assert out.metrics[0].name == "accuracy"
    assert out.metrics[0].value == 0.0


# ---------------------------------------------------------------------------
# IPO: metrics content on a known mixed batch
# ---------------------------------------------------------------------------


def test_ipo_metrics_exact_on_a_mixed_batch() -> None:
    # h0 = 1.0 > 0 and h1 = -0.5, so accuracy = 1/2 = 0.5 exactly -- NOT 1.0,
    # the value a broken accuracy most often returns. Mean margin = 0.25.
    out = IPOLoss()(_forward(_pct_plus_one_rows()), _pct_plus_one_batch())
    assert isinstance(out.metrics, tuple)
    assert len(out.metrics) == 2
    assert out.metrics[0].name == "accuracy"
    assert out.metrics[0].value == 0.5
    assert out.metrics[1].name == "ipo_margin_mean"
    assert out.metrics[1].value == 0.25


def test_ipo_component_shape() -> None:
    out = IPOLoss()(_forward(_margin_one_rows()), _margin_one_batch())
    assert len(out.components) == 1
    component = out.components[0]
    assert component.name == "ipo_loss"
    assert component.observed is True
    assert component.contribution == out.loss


# ---------------------------------------------------------------------------
# IPO: batch refusals
# ---------------------------------------------------------------------------


def test_ipo_missing_columns_refused() -> None:
    batch = ExperienceBatch(
        columns={
            "chosen_loss_mask": ((1,),),
            "rejected_loss_mask": ((1,),),
            "reference_chosen_logprob": (0.0,),
        }
    )
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(_margin_one_rows()), batch)
    message = str(exc_info.value)
    assert "reference_rejected_logprob" in message
    assert "absent" in message


def test_ipo_zero_row_batch_refused_as_unmeasured() -> None:
    batch = _ipo_batch((), (), (), ())
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward([]), batch)
    assert "0 rows" in str(exc_info.value)


def test_ipo_wrong_forward_row_count_refused() -> None:
    # One pair row for a two-row batch.
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(_margin_one_rows()), _pct_plus_one_batch())
    message = str(exc_info.value)
    assert "1" in message
    assert "2" in message


def test_ipo_row_that_is_not_a_pair_refused() -> None:
    rows = [
        ((_pct_plus_one_rows())[0][0], (_pct_plus_one_rows())[0][1]),
        ((-0.1,), (-0.2,), (-0.3,)),
    ]
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(rows), _pct_plus_one_batch())
    message = str(exc_info.value)
    assert "row 1" in message
    assert "3" in message
    assert "2" in message


def test_ipo_row_that_is_not_a_sized_sequence_refused() -> None:
    rows = [7, (_pct_plus_one_rows())[1]]
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(rows), _pct_plus_one_batch())
    assert "row 0" in str(exc_info.value)


def test_ipo_mask_shorter_than_token_row_refused() -> None:
    rows = [((-0.5, -0.4), (-1.5,))]
    batch = _ipo_batch(chosen_masks=((1,),))
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "chosen" in message
    assert "2" in message
    assert "1" in message


def test_ipo_fractional_mask_entry_refused() -> None:
    batch = _ipo_batch(chosen_masks=((0.5,),))
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(_margin_one_rows()), batch)
    message = str(exc_info.value)
    assert "0.5" in message
    assert "0 or 1" in message


def test_ipo_unconvertible_mask_entry_refused() -> None:
    batch = _ipo_batch(chosen_masks=((object(),),))
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(_margin_one_rows()), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 0" in message


def test_ipo_zero_supervision_side_refused_by_name() -> None:
    batch = _ipo_batch(rejected_masks=((0,),))
    with pytest.raises(SupervisionRefusal) as exc_info:
        IPOLoss()(_forward(_margin_one_rows()), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "rejected" in message
    assert "0 supervised tokens" in message


def test_ipo_unconvertible_log_probability_refused() -> None:
    rows = [((object(),), (-1.5,))]
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(rows), _ipo_batch())
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 0" in message
    assert "does not convert" in message


def test_ipo_non_finite_supervised_log_probability_refused() -> None:
    rows = [((-0.5, float("nan")), (-1.5, -0.25))]
    batch = _ipo_batch(chosen_masks=((1, 1),), rejected_masks=((1, 1),))
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 1" in message
    assert "nan" in message
    assert "not finite" in message


def test_ipo_non_finite_reading_at_a_masked_position_is_not_converted() -> None:
    # The helper converts ONLY supervised entries: a NaN under mask 0 is never
    # read as a float. pi_c = -0.5, pi_r = -1.5 -> h = 1.0 -> loss 0.25.
    rows = [((-0.5, float("nan")), (-1.5,))]
    batch = _ipo_batch(chosen_masks=((1, 0),))
    out = IPOLoss()(_forward(rows), batch)
    assert out.loss == pytest.approx(0.25, rel=1e-12)


def test_ipo_non_finite_reference_score_refused() -> None:
    batch = _ipo_batch(ref_rejected=(float("nan"),))
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(_margin_one_rows()), batch)
    message = str(exc_info.value)
    assert "reference_rejected_logprob" in message
    assert "row 0" in message
    assert "nan" in message


def test_ipo_unconvertible_reference_score_refused() -> None:
    batch = _ipo_batch(ref_chosen=(object(),))
    with pytest.raises(BatchRefusal) as exc_info:
        IPOLoss()(_forward(_margin_one_rows()), batch)
    message = str(exc_info.value)
    assert "reference_chosen_logprob" in message
    assert "row 0" in message
    assert "does not convert" in message


def test_ipo_boolean_mask_entries_are_admitted() -> None:
    rows = [((-0.5, -0.4), (-1.5, -0.25))]
    batch = _ipo_batch(chosen_masks=((True, False),), rejected_masks=((True, False),))
    # Supervised tokens only: pi_c = -0.5, pi_r = -1.5 -> h = 1.0 -> 0.25.
    out = IPOLoss()(_forward(rows), batch)
    assert out.loss == pytest.approx(0.25, rel=1e-12)


# ---------------------------------------------------------------------------
# IPO: declarations
# ---------------------------------------------------------------------------


def test_ipo_semantics_constrains_reference_free_and_abstains_elsewhere() -> None:
    semantics = IPOLoss().semantics()
    assert semantics.reference_free is False
    assert semantics.ratio_scope is None
    assert semantics.clip_bounds is None
    assert semantics.kl_estimator is None
    assert semantics.group_size is None


def test_ipo_required_columns_is_an_ordered_tuple() -> None:
    assert IPOLoss().required_columns == (
        "chosen_loss_mask",
        "rejected_loss_mask",
        "reference_chosen_logprob",
        "reference_rejected_logprob",
    )
    assert isinstance(IPOLoss().required_columns, tuple)


def test_ipo_declaration_shape_and_bounds() -> None:
    declaration = IPOLoss().declaration()
    assert declaration.components == ("ipo_loss",)
    assert len(declaration.metrics) == 2
    accuracy, margin = declaration.metrics
    assert accuracy.name == "accuracy"
    assert accuracy.low == 0.0
    assert accuracy.high == 1.0
    assert accuracy.degenerate == (0.0,)
    assert margin.name == "ipo_margin_mean"
    assert margin.low == -100.0
    assert margin.high == 100.0
    assert margin.degenerate == ()


def test_ipo_declared_and_observed_components_cannot_diverge() -> None:
    loss_fn = IPOLoss()
    out = loss_fn(_forward(_margin_one_rows()), _margin_one_batch())
    declared = loss_fn.declaration().components
    observed = tuple(component.name for component in out.components)
    assert observed == declared
    observed_metric_names = tuple(metric.name for metric in out.metrics)
    declared_metric_names = tuple(metric.name for metric in loss_fn.declaration().metrics)
    assert observed_metric_names == declared_metric_names


# ---------------------------------------------------------------------------
# KTO: construction refusals, one leg each
# ---------------------------------------------------------------------------


def test_kto_beta_zero_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(beta=0.0)
    assert "beta=0.0" in str(exc_info.value)


def test_kto_beta_negative_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(beta=-0.1)
    assert "beta=-0.1" in str(exc_info.value)


def test_kto_weight_zero_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(weight=0.0)
    assert "weight=0.0" in str(exc_info.value)


def test_kto_lambda_desirable_zero_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(lambda_desirable=0.0)
    assert "lambda_desirable=0.0" in str(exc_info.value)


def test_kto_lambda_undesirable_negative_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(lambda_undesirable=-2.5)
    assert "lambda_undesirable=-2.5" in str(exc_info.value)


def test_kto_count_metric_ceiling_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(count_metric_ceiling=-1.0)
    assert "count_metric_ceiling=-1.0" in str(exc_info.value)


def test_kto_paired_batch_columns_refused_empty() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(paired_batch_columns=())
    assert "paired_batch_columns=()" in str(exc_info.value)


def test_kto_paired_batch_columns_refused_non_tuple() -> None:
    constructor_arguments: Any = {"paired_batch_columns": ["chosen_loss_mask"]}
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(**constructor_arguments)
    assert "paired_batch_columns" in str(exc_info.value)


def test_kto_paired_batch_columns_refuses_an_empty_marker_name() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(paired_batch_columns=("",))
    assert "paired_batch_columns[0]=" in str(exc_info.value)


@pytest.mark.parametrize("field", _KTO_NAME_FIELDS)
def test_kto_every_name_field_refuses_an_empty_string(field: str) -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(**{field: ""})
    message = str(exc_info.value)
    assert field in message
    assert "not a name" in message


@pytest.mark.parametrize("field", _KTO_NAME_FIELDS)
def test_kto_every_name_field_refuses_a_non_string(field: str) -> None:
    constructor_arguments: Any = {field: 7}
    with pytest.raises(LossConfigRefusal) as exc_info:
        KTOLoss(**constructor_arguments)
    assert field in str(exc_info.value)


# ---------------------------------------------------------------------------
# KTO: numerics -- two different branch numbers on the same r
# ---------------------------------------------------------------------------


def test_kto_desirable_branch_matches_hand_computation() -> None:
    # r = 0.1 * ((-1.0) - (-2.0)) = 0.1; desirable loss =
    # sigmoid(0.3 - 0.1) = sigmoid(0.2) = 1 / (1 + e^-0.2).
    out = KTOLoss()(_forward([(-1.0,)]), _kto_batch())
    assert out.loss == pytest.approx(_KTO_DESIRABLE_LOSS, rel=1e-12)
    assert out.components[0].name == "kto_loss"
    assert out.components[0].observed is True
    assert out.components[0].contribution == out.loss


def test_kto_undesirable_branch_matches_hand_computation() -> None:
    # Same r = 0.1; undesirable loss = sigmoid(0.1 - 0.3) = sigmoid(-0.2),
    # a DIFFERENT number than the desirable branch on the same row.
    out = KTOLoss()(_forward([(-1.0,)]), _kto_batch(flags=(False,)))
    assert out.loss == pytest.approx(_KTO_UNDESIRABLE_LOSS, rel=1e-12)
    assert out.loss != pytest.approx(_KTO_DESIRABLE_LOSS, rel=1e-6)


def test_kto_beta_reaches_the_formula() -> None:
    # beta = 0.5 -> r = 0.5; desirable loss = sigmoid(0.3 - 0.5) =
    # sigmoid(-0.2). A constant-folded beta keeps 0.54983399... and fails.
    out = KTOLoss(beta=0.5)(_forward([(-1.0,)]), _kto_batch())
    assert out.loss == pytest.approx(_KTO_UNDESIRABLE_LOSS, rel=1e-12)


def test_kto_reference_point_is_read_not_constant_folded() -> None:
    # Same row, z0 = 0.0: desirable loss = sigmoid(-0.1), distinct from the
    # z0 = 0.3 reading. Pins that the recorded column reaches the formula.
    out = KTOLoss()(_forward([(-1.0,)]), _kto_batch(kl_points=(0.0,)))
    assert out.loss == pytest.approx(1.0 / (1.0 + math.exp(0.1)), rel=1e-12)
    assert out.loss != pytest.approx(_KTO_DESIRABLE_LOSS, rel=1e-6)


def test_kto_lambda_scales_its_row_family() -> None:
    # lambda_undesirable = 2.0 -> 2 * sigmoid(-0.2).
    out = KTOLoss(lambda_undesirable=2.0)(_forward([(-1.0,)]), _kto_batch(flags=(False,)))
    assert out.loss == pytest.approx(2.0 * _KTO_UNDESIRABLE_LOSS, rel=1e-12)


def test_kto_non_finite_reading_at_a_masked_position_is_not_converted() -> None:
    # r = 0.1 as usual; the NaN sits under mask 0 and is never read.
    out = KTOLoss()(_forward([(-1.0, float("nan"))]), _kto_batch(masks=((1, 0),)))
    assert out.loss == pytest.approx(_KTO_DESIRABLE_LOSS, rel=1e-12)


def test_kto_boolean_mask_entries_are_admitted() -> None:
    out = KTOLoss()(_forward([(-1.0, -0.4)]), _kto_batch(masks=((True, False),)))
    assert out.loss == pytest.approx(_KTO_DESIRABLE_LOSS, rel=1e-12)


def test_kto_integer_flags_are_admitted_and_measured() -> None:
    # flags (1, 0) exercise the by-value admission path of the desirable flag;
    # desirable sigmoid(0.2) + undesirable sigmoid(-0.2) = 1.0, so the mean is
    # 0.5 up to rounding, and the family counts are one and one.
    out = KTOLoss()(
        _forward([(-1.0,), (-1.0,)]),
        _kto_batch(
            masks=((1,), (1,)),
            refs=(-2.0, -2.0),
            kl_points=(0.3, 0.3),
            flags=(1, 0),
        ),
    )
    assert out.loss == pytest.approx(0.5, rel=1e-12)
    assert out.metrics[0].name == "desirable_count"
    assert out.metrics[0].value == 1.0
    assert out.metrics[1].name == "undesirable_count"
    assert out.metrics[1].value == 1.0


# ---------------------------------------------------------------------------
# KTO: family-count metrics, including the edges
# ---------------------------------------------------------------------------


def test_kto_counts_exact_on_a_mixed_batch() -> None:
    out = KTOLoss()(
        _forward([(-1.0,), (-1.0,)]),
        _kto_batch(
            masks=((1,), (1,)),
            refs=(-2.0, -2.0),
            kl_points=(0.3, 0.3),
            flags=(True, False),
        ),
    )
    assert isinstance(out.metrics, tuple)
    assert out.metrics[0].value == 1.0
    assert out.metrics[1].value == 1.0


def test_kto_all_desirable_edge() -> None:
    out = KTOLoss()(
        _forward([(-1.0,), (-1.0,)]),
        _kto_batch(
            masks=((1,), (1,)),
            refs=(-2.0, -2.0),
            kl_points=(0.3, 0.3),
            flags=(True, True),
        ),
    )
    assert out.loss == pytest.approx(_KTO_DESIRABLE_LOSS, rel=1e-12)
    assert out.metrics[0].value == 2.0
    assert out.metrics[1].value == 0.0


def test_kto_all_undesirable_edge() -> None:
    out = KTOLoss()(
        _forward([(-1.0,), (-1.0,)]),
        _kto_batch(
            masks=((1,), (1,)),
            refs=(-2.0, -2.0),
            kl_points=(0.3, 0.3),
            flags=(False, False),
        ),
    )
    assert out.loss == pytest.approx(_KTO_UNDESIRABLE_LOSS, rel=1e-12)
    assert out.metrics[0].value == 0.0
    assert out.metrics[1].value == 2.0


# ---------------------------------------------------------------------------
# KTO: batch refusals, including the column-count discipline
# ---------------------------------------------------------------------------


def test_kto_absent_kl_column_refused_with_the_shuffle_reason() -> None:
    batch = ExperienceBatch(
        columns={
            "loss_mask": ((1,),),
            "reference_logprob": (-2.0,),
            "is_desirable": (True,),
        }
    )
    with pytest.raises(BatchRefusal, match="shuffling completions") as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "kl_reference_point" in message
    assert "microbatch" in message
    assert "composition-dependent" in message


def test_kto_missing_flag_column_refused() -> None:
    batch = _kto_batch(drop_columns=("is_desirable",))
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "is_desirable" in message
    assert "absent" in message


def test_kto_paired_marker_columns_refused_with_count_and_names() -> None:
    batch = _kto_batch(
        extra_columns={
            "chosen_loss_mask": ((1,),),
            "rejected_loss_mask": ((1,),),
        }
    )
    with pytest.raises(BatchRefusal, match="2 of 4") as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "chosen_loss_mask" in message
    assert "rejected_loss_mask" in message
    assert "paired" in message
    assert "halved" in message


def test_pin_known_limit_custom_paired_marker_names_are_not_refused() -> None:
    # KNOWN LIMIT, pinned, not blessed: the paired-batch refusal keys on the
    # loss's CONFIGURED paired_batch_columns, which default to IPO/DPO's names.
    # A pipeline whose paired columns carry different names can hand KTOLoss a
    # fully paired batch unchecked -- here "chosen_loss_mask" is present but
    # the loss was configured to watch only "pref_chosen_loss_mask", so KTO
    # prices the row exactly as if the paired columns were noise. If this test
    # ever starts refusing, a real hole was closed and the pin must move.
    loss_fn = KTOLoss(paired_batch_columns=("pref_chosen_loss_mask",))
    batch = _kto_batch(extra_columns={"chosen_loss_mask": ((1,),)})
    out = loss_fn(_forward([(-1.0,)]), batch)
    assert out.loss == pytest.approx(_KTO_DESIRABLE_LOSS, rel=1e-12)
    assert out.metrics[0].value == 1.0


def test_kto_zero_row_batch_refused_as_unmeasured() -> None:
    batch = _kto_batch(masks=(), refs=(), kl_points=(), flags=())
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([]), batch)
    assert "0 rows" in str(exc_info.value)


def test_kto_wrong_forward_row_count_refused() -> None:
    batch = _kto_batch(
        masks=((1,), (1,)), refs=(-2.0, -2.0), kl_points=(0.3, 0.3), flags=(True, True)
    )
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "1" in message
    assert "2" in message


def test_kto_mask_shorter_than_token_row_refused() -> None:
    batch = _kto_batch(masks=((1, 1),))
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "completion" in message
    assert "1" in message
    assert "2" in message


def test_kto_fractional_mask_entry_refused() -> None:
    batch = _kto_batch(masks=((0.5,),))
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "0.5" in message
    assert "0 or 1" in message


def test_kto_zero_supervision_row_refused() -> None:
    batch = _kto_batch(masks=((0,),))
    with pytest.raises(SupervisionRefusal) as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "0 supervised tokens" in message


def test_kto_non_finite_supervised_log_probability_refused() -> None:
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([(-1.0, float("inf"))]), _kto_batch(masks=((1, 1),)))
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 1" in message
    assert "inf" in message
    assert "not finite" in message


def test_kto_unconvertible_log_probability_refused() -> None:
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([(object(),)]), _kto_batch())
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 0" in message
    assert "does not convert" in message


def test_kto_non_finite_reference_score_refused() -> None:
    batch = _kto_batch(refs=(float("nan"),))
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "reference_logprob" in message
    assert "row 0" in message
    assert "nan" in message


def test_kto_negative_reference_point_refused() -> None:
    batch = _kto_batch(kl_points=(-1.0,))
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "kl_reference_point" in message
    assert "row 0" in message
    assert "non-negative" in message


def test_kto_fractional_desirable_flag_refused() -> None:
    batch = _kto_batch(flags=(0.5,))
    with pytest.raises(BatchRefusal) as exc_info:
        KTOLoss()(_forward([(-1.0,)]), batch)
    message = str(exc_info.value)
    assert "is_desirable" in message
    assert "row 0" in message
    assert "0.5" in message


# ---------------------------------------------------------------------------
# KTO: declarations
# ---------------------------------------------------------------------------


def test_kto_semantics_constrains_reference_free_and_abstains_elsewhere() -> None:
    semantics = KTOLoss().semantics()
    assert semantics.reference_free is False
    assert semantics.ratio_scope is None
    assert semantics.clip_bounds is None
    assert semantics.kl_estimator is None
    assert semantics.group_size is None


def test_kto_required_columns_is_an_ordered_tuple_without_paired_names() -> None:
    assert KTOLoss().required_columns == (
        "loss_mask",
        "reference_logprob",
        "kl_reference_point",
        "is_desirable",
    )
    assert isinstance(KTOLoss().required_columns, tuple)
    assert "chosen_loss_mask" not in KTOLoss().required_columns


def test_kto_declaration_shape_and_bounds() -> None:
    declaration = KTOLoss().declaration()
    assert declaration.components == ("kto_loss",)
    assert len(declaration.metrics) == 2
    desirable, undesirable = declaration.metrics
    assert desirable.name == "desirable_count"
    assert desirable.low == 0.0
    assert desirable.high == 1.0e6
    assert desirable.degenerate == ()
    assert undesirable.name == "undesirable_count"
    assert undesirable.low == 0.0
    assert undesirable.high == 1.0e6
    assert undesirable.degenerate == ()


def test_kto_declared_and_observed_channels_cannot_diverge() -> None:
    loss_fn = KTOLoss()
    out = loss_fn(_forward([(-1.0,)]), _kto_batch())
    observed = tuple(component.name for component in out.components)
    assert observed == loss_fn.declaration().components
    observed_metric_names = tuple(metric.name for metric in out.metrics)
    declared_metric_names = tuple(metric.name for metric in loss_fn.declaration().metrics)
    assert observed_metric_names == declared_metric_names
