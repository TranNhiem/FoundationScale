"""FoundationScale-owned VLA evaluation harness (owned contract #4 of the VLA plan).

FS owns the episode plan, the rollout loop, success scoring and the report; a backend's policy is
reached only through its own serving protocol by an adapter in ``foundationscale.vla.adapters``.
LIBERO first: :func:`run_libero` reproduces both the openpi protocol (fixed suite init states,
settle steps) and the GR00T protocol (seeded resets) as declared in :class:`EpisodePlan`.
"""

from __future__ import annotations

from foundationscale.vla.eval.libero_runner import (
    EpisodeRecord,
    EvalReport,
    LiberoRunnerError,
    run_libero,
    within_published,
)
from foundationscale.vla.eval.plan import LIBERO_MAX_STEPS, EpisodePlan, EvalPlanError
from foundationscale.vla.eval.protocol import PolicyAdapter

__all__ = [
    "LIBERO_MAX_STEPS",
    "EpisodePlan",
    "EpisodeRecord",
    "EvalPlanError",
    "EvalReport",
    "LiberoRunnerError",
    "PolicyAdapter",
    "run_libero",
    "within_published",
]
