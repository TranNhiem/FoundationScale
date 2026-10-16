"""Driving trajectory metrics: ADE and minADE with Alpamayo's semantics."""

from __future__ import annotations

import numpy as np
import pytest

from foundationscale.vla.driving import DEFAULT_HORIZONS, TrajectoryMetricError, ade, min_ade


def _traj(xs: list[float], z: float = 0.0) -> np.ndarray:
    return np.array([[x, 0.0, z] for x in xs])


def test_ade_is_the_mean_xy_distance_and_ignores_height_by_default() -> None:
    gt = _traj([0.0, 1.0, 2.0])
    pred = _traj([1.0, 1.0, 5.0], z=9.0)  # errors 1, 0, 3 in x; z off by 9 everywhere
    assert ade(pred, gt) == pytest.approx(4.0 / 3.0)
    assert ade(pred, gt, horizon=2) == pytest.approx(0.5)
    assert ade(pred, gt, only_xy=False) == pytest.approx(
        np.mean([np.hypot(1, 9), 9.0, np.hypot(3, 9)])
    )


def test_ade_scores_every_trajectory_of_a_batch() -> None:
    gt = _traj([0.0, 0.0])
    pred = np.stack([_traj([1.0, 1.0]), _traj([2.0, 4.0])])
    assert ade(pred, gt).tolist() == pytest.approx([1.0, 3.0])


def test_min_ade_keeps_the_full_horizon_winner_for_every_prefix() -> None:
    t = 50
    gt = np.zeros((t, 3))
    # sample A: perfect for 5 steps then 2 m off; B: 1 m off throughout (lower full-horizon ADE)
    a = np.zeros((t, 3))
    a[5:, 0] = 2.0
    b = np.zeros((t, 3))
    b[:, 0] = 1.0
    out = min_ade(np.stack([a, b]), gt)
    assert out["min_ade"] == pytest.approx(1.0)  # B wins: ADE 1.0 vs 1.8
    # by_t of the SAME sample B, not a per-horizon minimum (which would be 0.0 for A at 0.5 s)
    assert out["min_ade/by_t=0.5"] == pytest.approx(1.0)
    assert set(out) == {
        "min_ade",
        "min_ade/by_t=0.5",
        "min_ade/by_t=1.0",
        "min_ade/by_t=3.0",
        "min_ade/by_t=5.0",
    }


def test_min_ade_averages_groups_and_skips_horizons_longer_than_the_trajectory() -> None:
    gt = _traj([0.0] * 10)
    g1 = np.stack([_traj([1.0] * 10), _traj([3.0] * 10)])  # best 1.0
    g2 = np.stack([_traj([2.0] * 10), _traj([5.0] * 10)])  # best 2.0
    out = min_ade(np.stack([g1, g2]), gt)
    assert out["min_ade"] == pytest.approx(1.5)
    assert set(out) == {"min_ade", "min_ade/by_t=0.5", "min_ade/by_t=1.0"}  # 30 and 50 > T=10
    assert DEFAULT_HORIZONS == (5, 10, 30, 50)


@pytest.mark.parametrize(
    ("pred", "gt"),
    [
        (np.zeros((2, 4, 3)), np.zeros((5, 3))),  # length mismatch
        (np.zeros((2, 4, 2)), np.zeros((4, 2))),  # not 3-D points
        (np.zeros((4, 3)), np.zeros((4, 3))),  # min_ade needs a sample axis
        (np.full((2, 4, 3), np.nan), np.zeros((4, 3))),  # non-finite
        (np.zeros((0, 4, 3)), np.zeros((4, 3))),  # no samples
        (np.zeros((1, 1, 1, 4, 3)), np.zeros((4, 3))),  # too many axes
    ],
)
def test_min_ade_refuses_what_it_cannot_score(pred: np.ndarray, gt: np.ndarray) -> None:
    with pytest.raises(TrajectoryMetricError):
        min_ade(pred, gt)


@pytest.mark.parametrize("bad", [0, 4, 1.5])
def test_ade_refuses_a_horizon_outside_the_trajectory(bad: object) -> None:
    with pytest.raises(TrajectoryMetricError):
        ade(_traj([0.0, 1.0, 2.0]), _traj([0.0, 1.0, 2.0]), horizon=bad)  # type: ignore[arg-type]


def test_min_ade_refuses_a_non_positive_time_step() -> None:
    with pytest.raises(TrajectoryMetricError):
        min_ade(np.zeros((1, 3, 3)), np.zeros((3, 3)), time_step=0.0)
