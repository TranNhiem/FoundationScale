"""Additional tests for ``foundationscale.rl.online_objectives`` covering the
refusals and the Best-of-N selection path left uncovered by the existing tests.

WHAT IS CLAIMED: these tests execute the IterativeDPO zero-supervision refusal,
the RAFT empty-batch and zero-supervision refusals, every BestOfNLoss batch
refusal constructible from the shipped source (empty batch, missing group
identity, a group with fewer than two rows, and a tied maximum reward), and
the BestOfNLoss success path -- a group with a UNIQUE best reward whose winner
is selected and priced, with the returned loss asserted to equal the mean NLL
of exactly the argmax-reward rows.

WHAT IS NOT CLAIMED: coverage of any behaviour not stated in the shipped
source. Every assertion names the specific value or the specific refusal
message the module emits.
"""

from __future__ import annotations

import math

import pytest

from foundationscale.rl.interfaces import (
    BatchRefusal,
    ExperienceBatch,
    LossOutput,
    SupervisionRefusal,
)
from foundationscale.rl.online_objectives import (
    BestOfNLoss,
    IterativeDPOLoss,
    RAFTLoss,
)


def _paired_forward_standard(batch: ExperienceBatch) -> tuple:
    return (
        ((-0.5, -0.7), (-1.1, -1.3)),
        ((-0.2, -0.9), (-1.0, -2.0)),
    )


class TestBestOfNSelection:
    def _best_of_n_columns(
        self,
        rewards: tuple[object, ...] = (0.1, 0.9, 0.5, 0.2),
        groups: tuple[object, ...] = ("g0", "g0", "g1", "g1"),
    ) -> dict[str, tuple[object, ...]]:
        return {
            "completion_loss_mask": ((1, 1),) * len(rewards),
            "reward": rewards,
            "group_id": groups,
        }

    def _best_of_n_batch(
        self,
        rewards: tuple[object, ...] = (0.1, 0.9, 0.5, 0.2),
        groups: tuple[object, ...] = ("g0", "g0", "g1", "g1"),
    ) -> ExperienceBatch:
        return ExperienceBatch(  # type: ignore[arg-type]
            columns=self._best_of_n_columns(rewards, groups)
        )

    def test_iterative_dpo_supervision_refusal_zero_supervised_side(self) -> None:
        columns: dict[str, tuple[object, ...]] = {
            "chosen_loss_mask": ((0, 0), (1, 1)),
            "rejected_loss_mask": ((1, 0), (1, 1)),
            "reference_chosen_logprob": (-1.0, -2.0),
            "reference_rejected_logprob": (-1.5, -0.5),
            "reference_snapshot_id": ("snap-a", "snap-a"),
        }
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(SupervisionRefusal, match="side 'chosen': supervision mask selected 0"):
            IterativeDPOLoss()(_paired_forward_standard, batch)

    def test_raft_batch_refusal_empty_batch(self) -> None:
        columns = {name: () for name in RAFTLoss().required_columns}
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="batch of 0 rows"):
            RAFTLoss()(lambda b: (), batch)

    def test_raft_supervision_refusal_zero_supervised_row(self) -> None:
        columns: dict[str, tuple[object, ...]] = {
            "completion_loss_mask": ((0, 0), (1, 1)),
        }
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]

        def forward(b):  # noqa: ANN001,ANN202
            return ((-0.4, -0.6), (-0.8, -1.1))

        with pytest.raises(
            SupervisionRefusal, match="row 0: supervision mask selected 0 supervised tokens"
        ):
            RAFTLoss()(forward, batch)

    def test_best_of_n_batch_refusal_empty_batch(self) -> None:
        columns = {name: () for name in BestOfNLoss().required_columns}
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        with pytest.raises(BatchRefusal, match="batch of 0 rows"):
            BestOfNLoss()(lambda b: (), batch)

    def test_best_of_n_batch_refusal_none_group_id(self) -> None:
        batch = self._best_of_n_batch(
            rewards=(0.1, 0.9),
            groups=("g0", None),
        )

        def forward(b):  # noqa: ANN001,ANN202
            return ((-0.4, -0.6), (-0.5, -0.7))

        with pytest.raises(BatchRefusal, match="row 1 is None"):
            BestOfNLoss()(forward, batch)

    def test_best_of_n_batch_refusal_group_with_fewer_than_two_rows(self) -> None:
        batch = self._best_of_n_batch(
            rewards=(0.1, 0.9),
            groups=("g0", "g1"),
        )

        def forward(b):  # noqa: ANN001,ANN202
            return ((-0.4, -0.6), (-0.5, -0.7))

        with pytest.raises(BatchRefusal, match="has 1 of the minimum 2 rows"):
            BestOfNLoss()(forward, batch)

    def test_best_of_n_batch_refusal_tied_maximum_reward(self) -> None:
        batch = self._best_of_n_batch(
            rewards=(0.9, 0.9),
            groups=("g0", "g0"),
        )

        def forward(b):  # noqa: ANN001,ANN202
            return ((-0.4, -0.6), (-0.5, -0.7))

        with pytest.raises(BatchRefusal, match="tied at the maximum reward"):
            BestOfNLoss()(forward, batch)

    def test_best_of_n_supervision_refusal_winner_with_zero_supervised_tokens(self) -> None:
        columns = self._best_of_n_columns(
            rewards=(0.1, 0.9),
            groups=("g0", "g0"),
        )
        columns["completion_loss_mask"] = ((1, 1), (0, 0))
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]

        def forward(b):  # noqa: ANN001,ANN202
            return ((-0.4, -0.6), (-0.5, -0.7))

        with pytest.raises(
            SupervisionRefusal, match="row 1: the selected winner's supervision mask selected 0"
        ):
            BestOfNLoss()(forward, batch)

    def test_best_of_n_selects_argmax_winner_and_prices_its_nll(self) -> None:
        batch = self._best_of_n_batch()
        forward_rows = (
            (-5.0, -5.0),
            (-0.4, -0.6),
            (-0.2, -1.0),
            (-5.0, -5.0),
        )
        output = BestOfNLoss()(lambda b: forward_rows, batch)
        assert isinstance(output, LossOutput)

        winner_row_nll_g0 = -(-0.4 + -0.6) / 2
        winner_row_nll_g1 = -(-0.2 + -1.0) / 2
        expected_nll = (winner_row_nll_g0 + winner_row_nll_g1) / 2
        assert math.isfinite(output.loss)
        assert abs(output.loss - expected_nll) < 1e-12

        assert len(output.components) == 1
        component = output.components[0]
        assert component.name == "best_of_n_loss"
        assert component.weight == 1.0
        assert component.observed is True
        assert abs(component.contribution - expected_nll) < 1e-12

        metrics = {metric.name: metric.value for metric in output.metrics}
        assert set(metrics) == {"best_of_n_nll_mean", "best_of_n_winner_reward_mean"}
        assert abs(metrics["best_of_n_nll_mean"] - expected_nll) < 1e-12
        assert abs(metrics["best_of_n_winner_reward_mean"] - (0.9 + 0.5) / 2) < 1e-12

    def test_best_of_n_winner_is_max_reward_row_not_first_row(self) -> None:
        columns = self._best_of_n_columns(
            rewards=(0.7, 0.3),
            groups=("g0", "g0"),
        )
        batch = ExperienceBatch(columns=columns)  # type: ignore[arg-type]
        forward_rows = (
            (-0.1, -0.1),
            (-3.0, -3.0),
        )
        output = BestOfNLoss()(lambda b: forward_rows, batch)
        expected_nll = -(-0.1 + -0.1) / 2
        assert abs(output.loss - expected_nll) < 1e-12
        metrics = {metric.name: metric.value for metric in output.metrics}
        assert abs(metrics["best_of_n_nll_mean"] - expected_nll) < 1e-12
        assert abs(metrics["best_of_n_winner_reward_mean"] - 0.7) < 1e-12
