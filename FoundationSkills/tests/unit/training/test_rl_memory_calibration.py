"""RL memory estimate against the GB200 measurements behind RL_ACTIVATION_FACTOR (gemma-4-E4B-it, FS RLTrainer)."""

from __future__ import annotations

import pytest
from foundationskills.skills.training.estimate import RL_ACTIVATION_FACTOR, estimate_memory
from foundationskills.skills.training.feasibility import INFEASIBLE_FRACTION
from foundationskills.skills.training.knowledge import find_variant

# logprob_micro_batch -> measured torch peak allocated (GB), seq 1280, no grad ckpt, one backward step each
MEASURED = {1: 64.1, 2: 81.5, 4: 115.9}
GB200_GB = 189.0


def _est(rows: int) -> float:
    found = find_variant("gemma-4-E4B-it")
    if found is None:
        pytest.skip("gemma-4-E4B-it not in the knowledge base")
    return estimate_memory(found[1], method="full", seq_len=1280, micro_batch=rows, sharding="ddp", grad_ckpt=False,
                           world=1, optimizer="host_adamw", activation_factor=RL_ACTIVATION_FACTOR).total_per_gpu_gb


@pytest.mark.parametrize("rows", sorted(MEASURED))
def test_the_estimate_tracks_each_measured_peak_within_15_percent(rows: int) -> None:
    assert abs(_est(rows) - MEASURED[rows]) / MEASURED[rows] < 0.15


def test_the_shape_that_ran_out_of_memory_is_priced_infeasible() -> None:
    assert _est(8) > INFEASIBLE_FRACTION * GB200_GB  # measured: 8 rows resident OOMed on a 184 GiB GB200


def test_without_the_factor_the_oom_shape_looked_feasible() -> None:
    found = find_variant("gemma-4-E4B-it")
    if found is None:
        pytest.skip("gemma-4-E4B-it not in the knowledge base")
    plain = estimate_memory(found[1], method="full", seq_len=1280, micro_batch=8, sharding="ddp", grad_ckpt=False,
                            world=1, optimizer="host_adamw").total_per_gpu_gb
    assert plain < INFEASIBLE_FRACTION * GB200_GB
