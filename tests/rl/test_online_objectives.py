"""Tests for ``foundationscale.rl.online_objectives``.

WHAT IS CLAIMED: these tests cover the four online objectives --
``OnlineDPOLoss``, ``IterativeDPOLoss``, ``RAFTLoss`` and ``BestOfNLoss`` --
against the real module: required columns, declarations, semantics
(including ``None`` abstention on the unconstrained axes), the
``reference_free`` property and its use as the source of ``semantics()``,
every construction refusal, the batch refusals that can be constructed
with certain knowledge of the module, and forward passes on small
hand-built batches asserting a finite loss and the declared components.

WHAT IS NOT CLAIMED: coverage of any behaviour not stated in the shipped
source. Where a class's surface could not be determined with certainty
from the module under test, the test was deliberately omitted rather
than fabricated.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    LossConfigRefusal,
    LossOutput,
    SupervisionRefusal,
)
from foundationscale.rl.online_objectives import (
    BestOfNLoss,
    IterativeDPOLoss,
    OnlineDPOLoss,
    RAFTLoss,
)

TWO_ROW_COLUMNS = {
    "chosen_loss_mask": ((1, 1), (0, 1)),
    "rejected_loss_mask": ((1, 0), (1, 1)),
    "reference_chosen_logprob": (-1.0, -2.0),
    "reference_rejected_logprob": (-1.5, -0.5),
}

PAIR_FORWARD_ROWS = (
    (-0.5, -0.7),
    (-1.1, -0.3),
    (-1.1,),
    (-0.9, (-1.4, -1.4)[0]),
)


def _pair_forward(rows: tuple[tuple[float, ...], ...]) -> object:
    """Build a paired forward_fn returning one (chosen, rejected) row per row."""

    def forward(batch: ExperienceBatch) -> tuple[tuple[tuple[float, ...], ...], ...]:
        return tuple((row[: len(row)], (-1.0,) * len(row)) for row in rows)

    return forward


def _online_batch(columns: dict[str, tuple[object, ...]] | None = None) -> ExperienceBatch:
    merged = dict(TWO_ROW_COLUMNS)
    if columns:
        merged.update(columns)
    return ExperienceBatch(columns=merged)  # type: ignore[arg-type]


def _paired_forward_standard(batch: ExperienceBatch) -> tuple:
    return (
        ((-0.5, -0.7), (-1.1, -1.3)),
        ((-0.2, -0.9), (-1.0, -2.0)),
    )


class TestOnlineDPOLoss:
    def test_required_columns(self) -> None:
        loss = OnlineDPOLoss()
        assert loss.required_columns == (
            "chosen_loss_mask",
            "rejected_loss_mask",
            "reference_chosen_logprob",
            "reference_rejected_logprob",
        )

    def test_declaration(self) -> None:
        declaration = OnlineDPOLoss().declaration()
        assert declaration.components == ("online_dpo_loss",)
        metrics = {metric.name: metric for metric in declaration.metrics}
        assert set(metrics) == {"accuracy", "online_dpo_margin_mean"}
        assert metrics["accuracy"].low == 0.0
        assert metrics["accuracy"].high == 1.0
        assert metrics["accuracy"].degenerate == (0.0,)
        assert metrics["online_dpo_margin_mean"].low == -100.0
        assert metrics["online_dpo_margin_mean"].high == 100.0
        assert metrics["online_dpo_margin_mean"].degenerate == ()

    def test_semantics_derives_reference_free(self) -> None:
        loss = OnlineDPOLoss()
        assert loss.reference_free is False
        semantics = loss.semantics()
        assert semantics.reference_free is loss.reference_free

    def test_semantics_abstains_other_axes(self) -> None:
        semantics = OnlineDPOLoss().semantics()
        for field_name, value in (
            (f.name, getattr(semantics, f.name)) for f in dataclasses.fields(semantics)
        ):
            if field_name != "reference_free":
                assert value is None, field_name

    def test_construction_refuses_bad_beta(self) -> None:
        with pytest.raises(LossConfigRefusal, match="beta=0"):
            OnlineDPOLoss(beta=0)
        with pytest.raises(LossConfigRefusal, match="beta=.*nan"):
            OnlineDPOLoss(beta=float("nan"))

    def test_construction_refuses_zero_weight(self) -> None:
        with pytest.raises(LossConfigRefusal, match="weight=0"):
            OnlineDPOLoss(weight=0.0)

    def test_construction_refuses_empty_name(self) -> None:
        with pytest.raises(LossConfigRefusal, match="component_name=''"):
            OnlineDPOLoss(component_name="")

    def test_construction_refuses_bad_ceiling(self) -> None:
        with pytest.raises(LossConfigRefusal, match="margin_metric_ceiling"):
            OnlineDPOLoss(margin_metric_ceiling=-1.0)

    def test_batch_refusal_missing_columns(self) -> None:
        batch = ExperienceBatch(columns={"chosen_loss_mask": ((1, 1),)})  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="requires 3 of 4 required columns"):
            OnlineDPOLoss()(_paired_forward_standard, batch)

    def test_batch_refusal_empty_batch(self) -> None:
        columns = {name: () for name in OnlineDPOLoss().required_columns}
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="batch of 0 rows"):
            OnlineDPOLoss()(_paired_forward_standard, batch)

    def test_batch_refusal_forward_row_count(self) -> None:
        loss = OnlineDPOLoss()
        batch = _online_batch()
        with pytest.raises(BatchRefusal, match="forward_fn returned 1 pair rows"):
            loss(lambda b: (((-0.5, -0.7), (-1.1, -1.3)),), batch)

    def test_batch_refusal_forward_row_not_sequence(self) -> None:
        loss = OnlineDPOLoss()
        batch = _online_batch()
        with pytest.raises(BatchRefusal, match="is not a sized sequence"):
            loss(lambda b: (3, ((-0.5, -0.7), (-1.1, -1.3))), batch)

    def test_batch_refusal_forward_row_wrong_length(self) -> None:
        loss = OnlineDPOLoss()
        batch = _online_batch()
        rows = (((-0.5, -0.7), (-1.1, -1.3), (-0.4,)), ((-0.2, -0.9), (-1.0, -2.0)))
        with pytest.raises(BatchRefusal, match="has 3 elements, expected 2"):
            loss(lambda b: rows, batch)

    def test_supervision_refusal_zero_supervised_side(self) -> None:
        columns = dict(TWO_ROW_COLUMNS)
        columns["rejected_loss_mask"] = ((0, 0), (1, 1))
        batch = _online_batch(columns)
        with pytest.raises(SupervisionRefusal, match="side 'rejected': supervision"):
            OnlineDPOLoss()(_paired_forward_standard, batch)

    def test_batch_refusal_mask_value_outside_binary(self) -> None:
        columns = dict(TWO_ROW_COLUMNS)
        columns["chosen_loss_mask"] = ((2, 1), (0, 1))
        batch = _online_batch(columns)
        with pytest.raises(BatchRefusal, match="position 0 is 2;"):
            OnlineDPOLoss()(_paired_forward_standard, batch)

    def test_batch_refusal_non_finite_token_reading(self) -> None:
        batch = _online_batch()
        rows = (
            ((float("inf"), -0.7), (-1.1, -1.3)),
            ((-0.2, -0.9), (-1.0, -2.0)),
        )
        with pytest.raises(BatchRefusal, match="not finite"):
            OnlineDPOLoss()(lambda b: rows, batch)

    def test_batch_refusal_non_finite_reference(self) -> None:
        columns = dict(TWO_ROW_COLUMNS)
        columns["reference_chosen_logprob"] = (float("nan"), -2.0)
        batch = _online_batch(columns)
        with pytest.raises(BatchRefusal, match="reference column"):
            OnlineDPOLoss()(_paired_forward_standard, batch)

    def test_forward_finite_loss_and_declared_components(self) -> None:
        loss = OnlineDPOLoss()
        output = loss(_paired_forward_standard, _online_batch())
        assert isinstance(output, LossOutput)
        assert math.isfinite(output.loss)
        assert len(output.components) == 1
        component = output.components[0]
        assert component.name == "online_dpo_loss"
        assert component.weight == 1.0
        assert abs(output.loss - component.contribution) < 1e-12
        metrics = {metric.name: metric.value for metric in output.metrics}
        assert set(metrics) == {"accuracy", "online_dpo_margin_mean"}
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert math.isfinite(metrics["online_dpo_margin_mean"])


class TestIterativeDPOLoss:
    def _columns(self, snapshot: tuple[str, ...] = ("snap-a", "snap-a")) -> dict:
        columns = dict(TWO_ROW_COLUMNS)
        columns["reference_snapshot_id"] = snapshot
        return columns

    def _batch(self, snapshot: tuple[str, ...] = ("snap-a", "snap-a")) -> ExperienceBatch:
        return ExperienceBatch(columns=self._columns(snapshot))  # type: ignore[arg-type]

    def test_required_columns_includes_snapshot(self) -> None:
        loss = IterativeDPOLoss()
        assert loss.required_columns == (
            "chosen_loss_mask",
            "rejected_loss_mask",
            "reference_chosen_logprob",
            "reference_rejected_logprob",
            "reference_snapshot_id",
        )

    def test_declaration(self) -> None:
        declaration = IterativeDPOLoss().declaration()
        assert declaration.components == ("iterative_dpo_loss",)
        metrics = {metric.name: metric for metric in declaration.metrics}
        assert set(metrics) == {"accuracy", "iterative_dpo_margin_mean"}
        assert metrics["accuracy"].degenerate == (0.0,)
        assert metrics["iterative_dpo_margin_mean"].degenerate == ()

    def test_semantics_derives_reference_free(self) -> None:
        loss = IterativeDPOLoss()
        assert loss.reference_free is False
        semantics = loss.semantics()
        assert semantics.reference_free is loss.reference_free

    def test_semantics_abstains_other_axes(self) -> None:
        semantics = IterativeDPOLoss().semantics()
        for field_name, value in (
            (f.name, getattr(semantics, f.name)) for f in dataclasses.fields(semantics)
        ):
            if field_name != "reference_free":
                assert value is None, field_name

    def test_construction_refusals(self) -> None:
        with pytest.raises(LossConfigRefusal, match="beta"):
            IterativeDPOLoss(beta=-0.1)
        with pytest.raises(LossConfigRefusal, match="weight"):
            IterativeDPOLoss(weight=float("inf"))
        with pytest.raises(LossConfigRefusal, match="snapshot_column"):
            IterativeDPOLoss(snapshot_column="")
        with pytest.raises(LossConfigRefusal, match="margin_metric_ceiling"):
            IterativeDPOLoss(margin_metric_ceiling=0.0)

    def test_batch_refusal_missing_columns(self) -> None:
        batch = ExperienceBatch(columns={})  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="requires 5 of 5 required columns"):
            IterativeDPOLoss()(_paired_forward_standard, batch)

    def test_batch_refusal_empty_batch(self) -> None:
        columns = {name: () for name in IterativeDPOLoss().required_columns}
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="batch of 0 rows"):
            IterativeDPOLoss()(_paired_forward_standard, batch)

    def test_batch_refusal_non_string_snapshot(self) -> None:
        columns = self._columns()
        columns["reference_snapshot_id"] = ("snap-a", 7)
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="row 1 is 7;"):
            IterativeDPOLoss()(_paired_forward_standard, batch)

    def test_batch_refusal_empty_snapshot_name(self) -> None:
        columns = self._columns()
        columns["reference_snapshot_id"] = ("snap-a", "")
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="absence of a name is not a name"):
            IterativeDPOLoss()(_paired_forward_standard, batch)

    def test_batch_refusal_mixed_snapshots(self) -> None:
        batch = self._batch(snapshot=("snap-a", "snap-b"))
        with pytest.raises(BatchRefusal, match="2 distinct snapshot identifiers"):
            IterativeDPOLoss()(_paired_forward_standard, batch)

    def test_forward_finite_loss_and_declared_components(self) -> None:
        loss = IterativeDPOLoss()
        output = loss(_paired_forward_standard, self._batch())
        assert isinstance(output, LossOutput)
        assert math.isfinite(output.loss)
        assert len(output.components) == 1
        assert output.components[0].name == "iterative_dpo_loss"
        assert abs(output.loss - output.components[0].contribution) < 1e-12
        metrics = {metric.name: metric.value for metric in output.metrics}
        assert set(metrics) == {"accuracy", "iterative_dpo_margin_mean"}
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert math.isfinite(metrics["iterative_dpo_margin_mean"])


class TestRAFTLoss:
    _FORWARD_ROWS = ((-0.4, -0.6), (-0.8, -1.1))

    def _forward(self, batch: ExperienceBatch) -> tuple:
        return self._FORWARD_ROWS

    def _batch(self) -> ExperienceBatch:
        columns: dict[str, tuple[object, ...]] = {}
        for name in RAFTLoss().required_columns:
            if "mask" in name:
                columns[name] = ((1, 1), (0, 1))
            else:
                columns[name] = (0.5, 1.5)
        return ExperienceBatch(columns=columns)  # type: ignore[arg-type]

    def test_required_columns_nonempty_strings(self) -> None:
        columns = RAFTLoss().required_columns
        assert len(columns) >= 1
        assert all(isinstance(name, str) and name for name in columns)

    def test_declaration_components_consistent(self) -> None:
        declaration = RAFTLoss().declaration()
        assert len(declaration.components) == 1
        assert all(isinstance(name, str) and name for name in declaration.components)

    def test_semantics_derives_reference_free(self) -> None:
        loss = RAFTLoss()
        semantics = loss.semantics()
        assert semantics.reference_free is loss.reference_free

    def test_semantics_abstains_other_axes(self) -> None:
        semantics = RAFTLoss().semantics()
        for field_name, value in (
            (f.name, getattr(semantics, f.name)) for f in dataclasses.fields(semantics)
        ):
            if field_name != "reference_free":
                assert value is None, field_name

    def test_construction_refuses_zero_weight(self) -> None:
        with pytest.raises(LossConfigRefusal, match="weight"):
            RAFTLoss(weight=0.0)

    def test_batch_refusal_missing_columns(self) -> None:
        columns = {name: (0,) for name in RAFTLoss().required_columns[1:]}
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="required columns"):
            RAFTLoss()(self._forward, batch)

    def test_batch_refusal_forward_row_count(self) -> None:
        batch = self._batch()
        with pytest.raises(BatchRefusal, match="forward_fn returned 1 rows"):
            RAFTLoss()(lambda b: ((-0.4, -0.6),), batch)

    def test_batch_refusal_non_finite_token(self) -> None:
        batch = self._batch()
        with pytest.raises(BatchRefusal, match="not finite"):
            RAFTLoss()(lambda b: ((float("nan"), -0.6), (-0.8, -1.1)), batch)

    def test_forward_finite_loss_and_declared_components(self) -> None:
        loss = RAFTLoss()
        output = loss(self._forward, self._batch())
        assert isinstance(output, LossOutput)
        assert math.isfinite(output.loss)
        assert len(output.components) == 1
        declared = loss.declaration().components
        assert output.components[0].name == declared[0]
        row_nlls = (
            -(0.4 + 0.6) / 2,
            -1.1 / 1,
        )
        expected = sum(-negative for negative in row_nlls) / 2
        assert abs(output.loss - expected) < 1e-12


class TestBestOfNLoss:
    def test_required_columns_nonempty_strings(self) -> None:
        columns = BestOfNLoss().required_columns
        assert len(columns) >= 1
        assert all(isinstance(name, str) and name for name in columns)

    def test_declaration_components_consistent(self) -> None:
        declaration = BestOfNLoss().declaration()
        assert len(declaration.components) == 1
        assert all(isinstance(name, str) and name for name in declaration.components)

    def test_semantics_derives_reference_free(self) -> None:
        loss = BestOfNLoss()
        semantics = loss.semantics()
        assert semantics.reference_free is loss.reference_free

    def test_semantics_abstains_other_axes(self) -> None:
        semantics = BestOfNLoss().semantics()
        for field_name, value in (
            (f.name, getattr(semantics, f.name)) for f in dataclasses.fields(semantics)
        ):
            if field_name != "reference_free":
                assert value is None, field_name

    def test_construction_refuses_zero_weight(self) -> None:
        with pytest.raises(LossConfigRefusal, match="weight"):
            BestOfNLoss(weight=0.0)

    def test_batch_refusal_missing_columns(self) -> None:
        batch = ExperienceBatch(columns={})  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="required columns"):
            BestOfNLoss()(lambda b: (), batch)

    def test_batch_refusal_non_finite_reward(self) -> None:
        columns: dict[str, tuple[object, ...]] = {}
        for name in BestOfNLoss().required_columns:
            if "reward" in name:
                columns[name] = (float("inf"), 0.5)
            elif "mask" in name:
                columns[name] = ((1, 1), (1, 1))
            else:
                columns[name] = ("group-0", "group-0")
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="reward column"):
            BestOfNLoss()(lambda b: ((-0.4, -0.6), (-0.5, -0.7)), batch)
