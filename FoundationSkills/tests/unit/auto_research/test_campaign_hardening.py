"""Campaign hardening: objective.direction validation and overflow-safe float() handling."""
from __future__ import annotations

import pytest

from foundationskills.skills.auto_research.campaign import (
    CLUSTER_TIME,
    _finite,
    check_launch,
    check_spec,
)

FINGER = "sha256:" + "a1" * 32
HUGE = 10**400  # too large for float(); must never raise


def spec(**overrides):
    """Baseline campaign spec: overrides replace a whole section or set one budget knob."""
    out = {
        "id": "c1",
        "objective": {"metric": "val_accuracy", "direction": "max", "benchmarks": []},
        "eval_policy": {"fingerprint": FINGER, "metrics": ["val_accuracy"]},
        "base": {"model": "gemma4-4b", "fingerprint": FINGER},
        "confirm": {"k": 2.0, "noise_floor_rel": 0.005, "guardrails": [], "guard_abs_epsilon": 0.01},
        "seeds": {"baseline_repeats": 3, "confirm_repeats": 3, "seed_list": [1]},
        "budget": {"gpu_hours_total": 24.0, "max_runs": 6, "per_run_timeout_h": 8.0, "reserve_frac": 0.3},
        "stopping": {"no_gain_streak": 3, "max_crash_streak": 2},
        "axes": [
            {"key": "optim.lr", "type": "log_float", "min": 1e-6, "max": 1e-3},
            {"key": "lora.rank", "type": "int", "min": 4, "max": 64},
            {"key": "train.method", "type": "categorical", "values": ["full", "lora"]},
        ],
        "cluster": {"partition": "rally", "max_nodes": 2, "gpus_per_node": 8,
                    "time": "10-00:00:00", "exclude": ["r01dgx02"]},
    }
    out.update({k: v for k, v in overrides.items() if k in out})
    out["budget"] = {**out["budget"], **{k: v for k, v in overrides.items() if k not in out}}
    return out


def launch(**overrides):
    entry = {
        "trial": "t1", "role": "candidate", "seed": 1, "delta": {"optim.lr": 5e-4},
        "nodes": 1, "gpus_per_node": 8, "partition": "rally", "time": CLUSTER_TIME,
        "gpu_hours_est": 4.0, "commands": ["uv run train.py"],
    }
    entry.update(overrides)
    return entry


def rule_ids(problems) -> set:
    return {rule_id for rule_id, _ in problems}


class TestObjectiveDirection:
    @pytest.mark.parametrize("direction", ["minimise", "maximize", "Min", "up", "", None, 3, True])
    def test_ar_in_001_objective_direction_must_be_max_or_min(self, direction):
        broken = spec()
        broken["objective"] = {**broken["objective"], "direction": direction}
        problems = check_spec(broken)
        assert "AR-IN-001" in rule_ids(problems)
        assert any("objective.direction" in message for _, message in problems)

    def test_max_min_and_missing_direction_are_clean(self):
        assert check_spec(spec()) == []
        minimum = spec()
        minimum["objective"] = {**minimum["objective"], "direction": "min"}
        assert check_spec(minimum) == []
        missing = spec()
        missing["objective"] = {"metric": "val_accuracy", "benchmarks": []}
        assert check_spec(missing) == []


class TestOverflowSafeNumbers:
    def test_finite_never_raises_on_huge_ints(self):
        assert _finite(HUGE) == 0.0
        assert _finite(HUGE, -1.0) == -1.0

    def test_budget_total_overflow_is_a_normal_ar_in_003_finding(self):
        assert rule_ids(check_spec(spec(gpu_hours_total=HUGE))) == {"AR-IN-003"}

    def test_per_run_timeout_overflow_is_a_normal_ar_ln_005_finding(self):
        assert rule_ids(check_spec(spec(per_run_timeout_h=HUGE))) == {"AR-LN-005"}

    def test_axis_bounds_overflow_is_a_normal_ar_in_005_finding(self):
        broken = spec(axes=[{"key": "optim.lr", "type": "float", "min": HUGE, "max": HUGE + 1}])
        assert "AR-IN-005" in rule_ids(check_spec(broken))

    def test_gpu_hours_est_overflow_is_a_normal_ar_ln_001_finding(self):
        problems = check_launch(spec(), launch(gpu_hours_est=HUGE), [{"gpu_hours_est": HUGE}])
        assert rule_ids(problems) == {"AR-LN-001"}
        assert any("gpu_hours_est must be a positive number" in message for _, message in problems)

    def test_delta_value_overflow_stays_inside_the_ar_ln_001_gates(self):
        problems = check_launch(spec(), launch(delta={"optim.lr": HUGE}), [])
        assert rule_ids(problems) == {"AR-LN-001"}
        assert any("delta optim.lr" in message for _, message in problems)

    def test_huge_axis_bounds_never_blow_up_the_launch_gates(self):
        broken = spec(axes=[{"key": "optim.lr", "type": "float", "min": HUGE, "max": HUGE * 2}])
        assert "AR-LN-001" in rule_ids(check_launch(broken, launch(delta={"optim.lr": 5e-4}), []))

    def test_huge_prior_hours_count_as_zero(self):
        assert check_launch(spec(), launch(gpu_hours_est=4.0), [{"gpu_hours_est": HUGE}]) == []
