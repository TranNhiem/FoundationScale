"""Coverage-driven legs: the scalar-coercion helpers of online_objectives.

Written because checks/coverage_floor.py measured the module at 89.7% and
named the coercion branches (bool->float, the non-numeric TypeError/ValueError
paths that raise BatchRefusal, and the token/mask length mismatch) as lines no
test reached. These drive those branches through the PUBLIC loss classes, not
through private underscore helpers.
"""

from __future__ import annotations

import math

import pytest

from foundationscale.rl.interfaces import BatchRefusal, ExperienceBatch
from foundationscale.rl.online_objectives import (
    BestOfNLoss,
    OnlineDPOLoss,
    RAFTLoss,
)

# Defined here rather than imported from test_online_objectives: a shared
# fixture would make a failure in one module attributable to the other.
TWO_ROW_COLUMNS = {
    "chosen_loss_mask": ((1, 1), (0, 1)),
    "rejected_loss_mask": ((1, 0), (1, 1)),
    "reference_chosen_logprob": (-1.0, -2.0),
    "reference_rejected_logprob": (-1.5, -0.5),
}


def _paired_forward_standard(batch: ExperienceBatch) -> tuple:
    return (
        ((-0.5, -0.7), (-1.1, -1.3)),
        ((-0.2, -0.9), (-1.0, -2.0)),
    )


def _online_batch(columns: dict[str, tuple[object, ...]] | None = None) -> ExperienceBatch:
    merged = dict(TWO_ROW_COLUMNS)
    if columns:
        merged.update(columns)
    return ExperienceBatch(columns=merged)  # type: ignore[arg-type]


class TestScalarCoercion:
    def test_bool_mask_entries_coerce_to_zero_and_one(self) -> None:
        columns: dict[str, tuple[object, ...]] = {
            "chosen_loss_mask": ((True, False),),
            "rejected_loss_mask": ((True, True),),
            "reference_chosen_logprob": (-1.0,),
            "reference_rejected_logprob": (-1.0,),
        }
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]

        def forward(b):  # noqa: ANN001,ANN202
            return (((-1.0, -1.0), (-1.0, -1.0)),)

        output = OnlineDPOLoss()(forward, batch)
        # pi_c = -1.0 (False suppresses position 1), pi_r = -2.0,
        # margin = (-1.0 - -1.0) - (-2.0 - -1.0) = 1.0.
        assert abs(output.loss - math.log1p(math.exp(-0.1))) < 1e-12
        metrics = {metric.name: metric.value for metric in output.metrics}
        assert metrics["accuracy"] == 1.0
        assert metrics["online_dpo_margin_mean"] == 1.0

    def test_mask_entries_admitted_by_value_through_float(self) -> None:
        columns: dict[str, tuple[object, ...]] = {
            "completion_loss_mask": (("1", 0.0),),
        }
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        output = RAFTLoss()(lambda b: ((-0.25, -9.0),), batch)
        assert abs(output.loss - 0.25) < 1e-12
        metrics = {metric.name: metric.value for metric in output.metrics}
        assert abs(metrics["raft_nll_mean"] - 0.25) < 1e-12

    def test_non_numeric_mask_entry_refused(self) -> None:
        columns: dict[str, tuple[object, ...]] = {
            "completion_loss_mask": ((None, 1),),
        }
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match=r"mask entry at row 0, position 0 is None"):
            RAFTLoss()(lambda b: ((-0.5, -0.5),), batch)

    def test_non_numeric_token_reading_refused(self) -> None:
        columns: dict[str, tuple[object, ...]] = {
            "completion_loss_mask": ((1, 1),),
        }
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]

        def forward(b):  # noqa: ANN001,ANN202
            return (("oops", -0.5),)

        with pytest.raises(
            BatchRefusal,
            match=r"row 0, position 0 does not convert to a scalar float \(str\)",
        ):
            RAFTLoss()(forward, batch)

    def test_non_numeric_reference_score_refused(self) -> None:
        columns = dict(TWO_ROW_COLUMNS)
        columns["reference_chosen_logprob"] = (None, -2.0)
        batch = _online_batch(columns)
        with pytest.raises(
            BatchRefusal,
            match=r"reference column 'reference_chosen_logprob' row 0 is None, which "
            r"does not convert",
        ):
            OnlineDPOLoss()(_paired_forward_standard, batch)

    def test_non_numeric_reward_refused(self) -> None:
        columns: dict[str, tuple[object, ...]] = {
            "completion_loss_mask": ((1, 1), (1, 1)),
            "reward": ("oops", 0.5),
            "group_id": ("g0", "g0"),
        }
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]

        def forward(b):  # noqa: ANN001,ANN202
            return ((-0.4, -0.6), (-0.5, -0.7))

        with pytest.raises(
            BatchRefusal,
            match=r"reward column 'reward' row 0 is 'oops', which does not convert",
        ):
            BestOfNLoss()(forward, batch)

    def test_reward_values_convert_and_select_argmax_winner(self) -> None:
        columns: dict[str, tuple[object, ...]] = {
            "completion_loss_mask": ((1, 1), (1, 1), (1, 1), (1, 1)),
            "reward": (1, 3, 2.0, 4.0),
            "group_id": ("g0", "g0", "g1", "g1"),
        }
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]

        def forward(b):  # noqa: ANN001,ANN202
            return (
                (-0.9, -0.9),
                (-0.2, -0.4),
                (-0.5, -0.5),
                (-0.6, -0.2),
            )

        output = BestOfNLoss()(forward, batch)
        # Winners: row 1 of g0 (reward 3) and row 3 of g1 (reward 4.0);
        # winner NLLs 0.3 and 0.4.
        assert abs(output.loss - 0.35) < 1e-12
        metrics = {metric.name: metric.value for metric in output.metrics}
        assert abs(metrics["best_of_n_nll_mean"] - 0.35) < 1e-12
        assert metrics["best_of_n_winner_reward_mean"] == 3.5

    def test_token_mask_length_mismatch_refused(self) -> None:
        columns: dict[str, tuple[object, ...]] = {
            "completion_loss_mask": ((1, 1),),
        }
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(
            BatchRefusal,
            match=r"forward_fn returned 3 per-token log-probabilities but the mask "
            r"has 2 entries",
        ):
            RAFTLoss()(lambda b: ((-0.1, -0.2, -0.3),), batch)
