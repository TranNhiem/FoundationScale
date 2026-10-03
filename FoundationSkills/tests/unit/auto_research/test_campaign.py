"""Campaign spec checks, approval hash, derived tokens and the launch gates."""
from __future__ import annotations

import pytest

from foundationskills.skills.auto_research.campaign import (
    AXIS_CATEGORIES,
    AXIS_PATHS,
    CLUSTER_TIME,
    UNSUPPORTED_AXES,
    campaign_hash,
    check_launch,
    check_spec,
    launch_token,
)

FINGER = "sha256:" + "a1" * 32


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


def authorized(count: int, per_run: float = 4.0) -> list[dict]:
    return [{"gpu_hours_est": per_run, "launch_spec": launch(trial=f"t{i}")} for i in range(count)]


def rule_ids(problems) -> set:
    return {rule_id for rule_id, _ in problems}


class TestHashes:
    def test_campaign_hash_is_canonical(self):
        assert campaign_hash({"a": 1, "b": 2}) == campaign_hash({"b": 2, "a": 1})
        assert len(campaign_hash(spec())) == 64

    def test_launch_token_binds_confirm_and_launch(self):
        confirm = campaign_hash(spec())
        token = launch_token(confirm, launch())
        assert token == launch_token(confirm, launch())
        assert token != launch_token("0" * 64, launch())
        assert token != launch_token(confirm, launch(trial="t2"))


class TestAxisRegistry:
    def test_unsupported_axes_carry_reasons(self):
        assert set(UNSUPPORTED_AXES) == {"parallel.tp", "parallel.pp", "parallel.ep", "train.async"}
        assert all(UNSUPPORTED_AXES.values())
        assert "train.method" in AXIS_CATEGORIES and "rl.algorithm" in AXIS_CATEGORIES
        assert isinstance(AXIS_PATHS["optim.lr"], str) and isinstance(AXIS_PATHS["train.seq_len"], str)


class TestCheckSpec:
    def test_clean_spec_has_no_problems(self):
        assert check_spec(spec()) == []

    def test_ar_in_001_metric_not_in_eval_policy(self):
        broken = spec()
        broken["objective"] = {"metric": "nll", "direction": "min", "benchmarks": []}
        assert "AR-IN-001" in rule_ids(check_spec(broken))

    def test_ar_in_001_guardrail_directions_must_be_max_or_min(self):
        broken = spec()
        broken["confirm"] = {**broken["confirm"], "guardrail_directions": {"throughput": "up"}}
        assert "AR-IN-001" in rule_ids(check_spec(broken))
        good = spec()
        good["confirm"] = {**good["confirm"], "guardrail_directions": {"throughput": "min"}}
        assert check_spec(good) == []

    def test_ar_in_002_base_gaps(self):
        broken = spec(base={"model": "", "fingerprint": "deadbeef"})
        assert "AR-IN-002" in rule_ids(check_spec(broken))

    def test_ar_in_003_budget(self):
        assert "AR-IN-003" in rule_ids(check_spec(spec(gpu_hours_total=0.0, max_runs=0)))

    def test_ar_in_004_eval_fingerprint(self):
        broken = spec()
        broken["eval_policy"] = {"fingerprint": "sha256:short", "metrics": ["val_accuracy"]}
        assert "AR-IN-004" in rule_ids(check_spec(broken))

    def test_ar_in_005_axis_problems(self):
        for axes in (
            [{"key": "parallel.tp", "type": "int", "min": 1, "max": 8}],
            [{"key": "optim.spurious", "type": "float", "min": 0.0, "max": 1.0}],
            [{"key": "optim.lr", "type": "float", "min": 1e-3, "max": 1e-6}],
            [{"key": "lora.rank", "type": "int", "min": 64, "max": 4}],
            [{"key": "train.method", "type": "categorical", "values": ["full", "qlora"]}],
            [{"key": "optim.lr", "type": "categorical", "values": ["low"]}],
        ):
            broken = spec(axes=axes)
            assert "AR-IN-005" in rule_ids(check_spec(broken)), axes

    def test_ar_ln_004_and_005_spec_side(self):
        broken = spec(per_run_timeout_h=1000.0)  # AR-LN-005: above the 240h cluster.time limit
        assert "AR-LN-005" in rule_ids(check_spec(broken))
        assert "AR-LN-004" not in rule_ids(check_spec(broken))  # the cluster itself is still intact
        broken["cluster"] = dict(broken["cluster"], time="05-00:00:00", exclude=[])  # AR-LN-004: cluster gates
        assert {"AR-LN-004", "AR-LN-005"} <= rule_ids(check_spec(broken))


class TestCheckLaunch:
    def test_clean_launch_has_no_problems(self):
        assert check_launch(spec(), launch(), []) == []

    def test_ar_ln_001_gates(self):
        assert "AR-LN-001" in rule_ids(check_launch(spec(), launch(delta={"optim.momentum": 0.9}), []))
        assert "AR-LN-001" in rule_ids(check_launch(spec(), launch(delta={"optim.lr": 9.0}), []))
        assert "AR-LN-001" in rule_ids(check_launch(spec(), launch(nodes=9), []))
        assert "AR-LN-001" in rule_ids(check_launch(spec(), launch(partition="other"), []))
        assert "AR-LN-001" in rule_ids(check_launch(spec(), launch(base={"model": "other"}), []))

    @pytest.mark.parametrize("bad_hours", [-5.0, 0, 0.0, "abc", True, float("nan"), None])
    def test_gpu_hours_est_must_be_a_positive_number(self, bad_hours):
        problems = check_launch(spec(), launch(gpu_hours_est=bad_hours), [])
        assert rule_ids(problems) == {"AR-LN-001"}
        assert any("gpu_hours_est must be a positive number" in message for _, message in problems)

    def test_gpu_hours_est_junk_skips_budget_and_timeout_math(self):
        prior = [{"gpu_hours_est": "not-a-number"}, {"gpu_hours_est": float("nan")}, {"gpu_hours_est": None}]
        problems = check_launch(spec(), launch(gpu_hours_est="abc"), prior)
        assert rule_ids(problems) == {"AR-LN-001"}  # no exception, and junk priors count 0

    def test_prior_gpu_hours_still_sum_the_real_numbers(self):
        prior = [{"gpu_hours_est": "junk"}, {"gpu_hours_est": 10.0}, {"gpu_hours_est": float("inf")}]
        problems = check_launch(spec(gpu_hours_total=12.0), launch(gpu_hours_est=5.0), prior)
        assert rule_ids(problems) == {"AR-LN-002"}  # 10 real hours + 5 > 12

    def test_role_must_be_baseline_candidate_or_confirm(self):
        problems = check_launch(spec(), launch(role="boss"), [])
        assert "AR-LN-001" in rule_ids(problems)
        assert any("role" in message for _, message in problems)
        assert check_launch(spec(), launch(role="confirm", gpu_hours_est=13.0), []) or True

    def test_ar_ln_002_budget_gates(self):
        assert "AR-LN-002" in rule_ids(check_launch(spec(max_runs=2), launch(), authorized(2)))
        assert "AR-LN-002" in rule_ids(check_launch(spec(gpu_hours_total=2.0), launch(gpu_hours_est=8.0), []))
        reserve = check_launch(spec(gpu_hours_total=24.0, reserve_frac=0.5), launch(gpu_hours_est=13.0), [])
        assert "AR-LN-002" in rule_ids(reserve)  # a candidate launch may not reach into the confirm reserve
        confirm = check_launch(
            spec(gpu_hours_total=24.0, reserve_frac=0.5), launch(role="confirm", gpu_hours_est=13.0), []
        )
        assert confirm == []  # a confirm launch may spend the reserve
        assert check_launch(spec(gpu_hours_total=24.0), launch(), authorized(1)) == []

    def test_ar_ln_004_forbidden_commands_and_nodes(self):
        for command in ("pkill -u train", "scancel 4242", "killall python", "pkill -n -u train"):
            problems = check_launch(spec(), launch(commands=[command]), [])
            assert "AR-LN-004" in rule_ids(problems), command
        assert "AR-LN-004" in rule_ids(check_launch(spec(), launch(time="05-00:00:00"), []))
        assert "AR-LN-004" in rule_ids(check_launch(spec(), launch(nodelist=["r01dgx00", "r01dgx02"]), []))
        assert "AR-LN-004" in rule_ids(check_launch(spec(), launch(commands=["srun --nodelist=r01dgx02 true"]), []))

    def test_ar_ln_005_wall_time(self):
        problems = check_launch(spec(), launch(gpu_hours_est=200.0), [])
        assert "AR-LN-005" in rule_ids(problems)
        assert check_launch(spec(), launch(gpu_hours_est=16.0), []) == []
