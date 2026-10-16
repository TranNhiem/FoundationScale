"""Trajectory metrics for driving VLAs, owned by FoundationScale, matched to Alpamayo's definitions.

The reference is alpamayo-recipes ``alpamayo/metrics/distance_metrics.py`` (``compute_ade``,
``compute_minade``), reproduced here as a contract so an FS evaluation scores a driving policy the
way its published numbers were scored:

* **ADE** is the L2 distance per timestep, in the bird's-eye (XY) plane by default, averaged over
  the horizon.
* **minADE** picks, within each group of K sampled trajectories, the sample whose full-horizon ADE
  is lowest; ``min_ade`` is that sample's ADE and ``min_ade/by_t=<s>`` is the ADE of **the same
  sample** over the first ``t`` steps -- not a fresh minimum per horizon, which would read lower.
  Groups are averaged. Horizons longer than the trajectory are skipped, as in the reference.

Shapes follow the reference: predictions ``[N, K, T, 3]`` (N groups of K samples; ``[K, T, 3]`` is
one group) and ground truth ``[T, 3]``, in metres. Non-finite values, mismatched lengths and empty
inputs are refused with :class:`TrajectoryMetricError`: a NaN would make minADE silently pick
another sample.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

__all__ = ["DEFAULT_HORIZONS", "TrajectoryMetricError", "ade", "min_ade"]

# alpamayo-recipes defaults: 5 / 10 / 30 / 50 steps at 0.1 s = 0.5 / 1 / 3 / 5 s
DEFAULT_HORIZONS = (5, 10, 30, 50)


class TrajectoryMetricError(ValueError):
    """Inputs a trajectory metric cannot score as declared."""


def _checked(pred: Any, gt: Any, *, min_pred_dims: int) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np  # noqa: PLC0415

    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if g.ndim != 2 or g.shape[-1] != 3 or g.shape[0] < 1:
        raise TrajectoryMetricError(f"ground truth has shape {g.shape}; expected [T >= 1, 3]")
    if p.ndim < min_pred_dims or p.shape[-1] != 3 or p.shape[-2] != g.shape[0]:
        raise TrajectoryMetricError(
            f"prediction has shape {p.shape}; expected [..., T={g.shape[0]}, 3] "
            f"with at least {min_pred_dims} dims"
        )
    if p.size == 0:
        raise TrajectoryMetricError("prediction holds no trajectory")
    if not (np.isfinite(p).all() and np.isfinite(g).all()):
        raise TrajectoryMetricError("a trajectory holds a non-finite value; nothing is scored")
    return p, g


def _l2(p: np.ndarray, g: np.ndarray, only_xy: bool) -> np.ndarray:
    import numpy as np  # noqa: PLC0415

    diff = p - g
    if only_xy:
        diff = diff[..., :2]
    return np.linalg.norm(diff, axis=-1)  # [..., T]


def ade(pred: Any, gt: Any, *, horizon: int | None = None, only_xy: bool = True) -> np.ndarray:
    """ADE of each trajectory in ``pred`` ``[..., T, 3]`` against ``gt`` ``[T, 3]`` -> ``[...]``."""
    p, g = _checked(pred, gt, min_pred_dims=2)
    l2 = _l2(p, g, only_xy)
    if horizon is not None:
        if type(horizon) is not int or not 1 <= horizon <= l2.shape[-1]:
            raise TrajectoryMetricError(
                f"horizon {horizon!r} must be an int in [1, {l2.shape[-1]}]"
            )
        l2 = l2[..., :horizon]
    return l2.mean(axis=-1)


def min_ade(
    pred: Any,
    gt: Any,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    time_step: float = 0.1,
    only_xy: bool = True,
) -> dict[str, float]:
    """minADE over K samples (``[K, T, 3]`` or ``[N, K, T, 3]``), averaged over the N groups."""
    p, g = _checked(pred, gt, min_pred_dims=3)
    if p.ndim == 3:
        p = p[None]
    if p.ndim != 4:
        raise TrajectoryMetricError(
            f"prediction has shape {p.shape}; expected [K, T, 3] or [N, K, T, 3]"
        )
    if not time_step > 0:
        raise TrajectoryMetricError(f"time_step {time_step!r} must be > 0")
    l2 = _l2(p, g, only_xy)  # [N, K, T]
    best = l2.mean(axis=-1).argmin(axis=1)  # [N]: chosen on the FULL horizon
    chosen = l2[range(l2.shape[0]), best]  # [N, T]
    out = {"min_ade": float(chosen.mean(axis=-1).mean())}
    for t in horizons:
        if t <= chosen.shape[-1]:
            out[f"min_ade/by_t={t * time_step:.1f}"] = float(chosen[:, :t].mean(axis=-1).mean())
    return out
