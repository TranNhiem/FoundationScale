"""Acceptance math tests: noise floor, tau band, verdicts and guardrails."""
from __future__ import annotations

import statistics

import pytest

from foundationskills.skills.auto_research.accept import decide, noise_floor, values

FINGER = "sha256:" + "a1" * 32

SPEC = {
    "id": "c1",
    "objective": {"metric": "val_accuracy", "direction": "max", "benchmarks": []},
    "eval_policy": {"fingerprint": FINGER, "metrics": ["val_accuracy", "throughput"]},
    "confirm": {"k": 2.0, "noise_floor_rel": 0.005, "guardrails": [], "guard_abs_epsilon": 0.01},
    "seeds": {"baseline_repeats": 3, "confirm_repeats": 3, "seed_list": [1, 2, 3]},
}


def result(trial, role, seed, value, se=0.001, **overrides):
    row = {
        "trial": trial, "role": role, "seed": seed, "status": "ok", "limited": False, "steps": 100,
        "eval_policy_fingerprint": FINGER,
        "metrics": {"val_accuracy": {"value": value, "se": se}},
    }
    row.update(overrides)
    return row


def row_metrics(trial, role, seed, metrics, **overrides):
    return result(trial, role, seed, None, metrics=metrics, **overrides)


BASELINE = [result("baseline", "baseline", 1, 0.5), result("baseline", "baseline", 2, 0.502),
            result("baseline", "baseline", 3, 0.501)]


def guarded(trial, role, pairs, metric, se=1.0):
    """(seed, objective value, guard value) rows carrying both the objective metric and one guard metric."""
    return [
        row_metrics(trial, role, seed, {
            "val_accuracy": {"value": value, "se": 0.001},
            metric: {"value": guard, "se": se},
        })
        for seed, value, guard in pairs
    ]


class TestNoiseFloor:
    def test_none_when_repeats_missing(self):
        assert noise_floor(BASELINE[:2], "val_accuracy") is None
        assert noise_floor([], "val_accuracy") is None

    def test_robust_mad_for_three_repeats(self):
        floor = noise_floor([result("b", "baseline", 1, 1.0), result("b", "baseline", 2, 2.0),
                            result("b", "baseline", 3, 3.0)], "val_accuracy")
        assert floor == pytest.approx(1.4826 * 1.0)

    def test_falls_back_to_std_when_mad_is_zero(self):
        equal = [result("b", "baseline", i, 2.0) for i in (1, 2, 3)]
        assert noise_floor(equal, "val_accuracy") == pytest.approx(0.0)

    def test_sample_std_for_more_repeats(self):
        rows = [result("b", "baseline", i, v) for i, v in enumerate((0.0, 1.0, 2.0, 3.0))]
        assert noise_floor(rows, "val_accuracy") == pytest.approx(statistics.stdev([0.0, 1.0, 2.0, 3.0]))

    def test_crashes_and_limited_runs_are_not_values(self):
        rows = [*BASELINE, result("b", "baseline", 4, 0.9, status="crash", limited=True)]
        assert values(rows, "val_accuracy") == [0.5, 0.502, 0.501]


class TestDecide:
    def test_accepted_gain(self):
        cand = [result("t1", "candidate", 1, 0.9), result("t1", "candidate", 2, 0.902),
                result("t1", "candidate", 3, 0.901)]
        decision = decide(SPEC, BASELINE, cand, "val_accuracy")
        assert decision["verdict"] == "accepted_gain"
        assert decision["n_pairs"] == 3 and decision["mean_delta"] > decision["tau"]

    def test_accepted_flat_for_simpler_candidate(self):
        cand = [result("t1", "candidate", 1, 0.501, simpler=True), result("t1", "candidate", 2, 0.5, simpler=True),
                result("t1", "candidate", 3, 0.502, simpler=True)]
        decision = decide(SPEC, BASELINE, cand, "val_accuracy")
        assert decision["verdict"] == "accepted_flat"
        assert abs(decision["mean_delta"]) <= decision["tau"]

    def test_no_gain_without_simpler(self):
        cand = [result("t1", "candidate", 1, 0.5), result("t1", "candidate", 2, 0.501),
                result("t1", "candidate", 3, 0.499)]
        assert decide(SPEC, BASELINE, cand, "val_accuracy")["verdict"] == "no_gain"

    def test_rejected_regress(self):
        cand = [result("t1", "candidate", 1, 0.1), result("t1", "candidate", 2, 0.1),
                result("t1", "candidate", 3, 0.1)]
        decision = decide(SPEC, BASELINE, cand, "val_accuracy")
        assert decision["verdict"] == "rejected_regress" and decision["mean_delta"] < -decision["tau"]

    def test_min_direction_sign(self):
        spec = {**SPEC, "objective": {"metric": "val_accuracy", "direction": "min", "benchmarks": []}}
        cand = [result("t1", "candidate", 1, 0.1), result("t1", "candidate", 2, 0.1),
                result("t1", "candidate", 3, 0.1)]
        assert decide(spec, BASELINE, cand, "val_accuracy")["verdict"] == "accepted_gain"

    def test_unmeasured_triggers(self):
        limited = [result("t1", "candidate", 1, 0.9, limited=True)]
        decision = decide(SPEC, BASELINE, limited, "val_accuracy")
        assert decision["verdict"] == "unmeasured"
        assert any(r.startswith("limited_run:") for r in decision["reasons"])

        wrong_fp = [result("t1", "candidate", 1, 0.9, eval_policy_fingerprint="sha256:" + "0" * 64)]
        decision = decide(SPEC, BASELINE, wrong_fp, "val_accuracy")
        assert decision["verdict"] == "unmeasured"
        assert any(r.startswith("fingerprint_mismatch:") for r in decision["reasons"])

        decision = decide(SPEC, BASELINE[:1], [result("t1", "candidate", 1, 0.9)], "val_accuracy")
        assert decision["verdict"] == "unmeasured"
        assert any(r.startswith("uncalibrated_noise_floor:") for r in decision["reasons"])
        assert any(r.startswith("n_pairs=") for r in decision["reasons"])

        decision = decide(SPEC, BASELINE, [result("t1", "candidate", 1, 0.9)], "missing")
        assert decision["verdict"] == "unmeasured"
        assert any(r.startswith("missing_metric:") for r in decision["reasons"])

    def test_crashes_are_counted_never_evidence(self):
        cand = [result("t1", "candidate", 1, 0.9), result("t1", "candidate", 2, 0.9),
                result("t1", "candidate", 3, 0.9)]
        crashed = [*cand, result("t1", "candidate", 4, 0.0, status="crash", metrics={})]
        decision = decide(SPEC, BASELINE, crashed, "val_accuracy")
        assert decision["verdict"] == "accepted_gain" and decision["n_pairs"] == 3
        assert any(r.startswith("crash_excluded:") for r in decision["reasons"])

    def test_guardrail_breach_forces_rejected_regress(self):
        spec = {**SPEC, "confirm": {"k": 2.0, "noise_floor_rel": 0.005,
                                    "guardrails": ["throughput"], "guard_abs_epsilon": 0.01}}
        baseline = guarded("b", "baseline", [(1, 0.5, 1000.0), (2, 0.502, 1000.0), (3, 0.501, 1000.0)], "throughput")
        cand = guarded("t1", "candidate", [(1, 0.6, 900.0), (2, 0.602, 899.0), (3, 0.601, 901.0)], "throughput")
        decision = decide(spec, baseline, cand, "val_accuracy")
        assert decision["verdict"] == "rejected_regress"
        assert "guardrail:throughput" in decision["reasons"]

    def test_guard_direction_is_not_the_objectives_sign(self):
        # objective direction "min": a lower val_accuracy is a gain, but throughput 1000 -> 500 still breaches
        spec = {**SPEC,
                "objective": {"metric": "val_accuracy", "direction": "min", "benchmarks": []},
                "confirm": {"k": 2.0, "noise_floor_rel": 0.005,
                            "guardrails": ["throughput"], "guard_abs_epsilon": 0.01}}
        baseline = guarded("b", "baseline", [(1, 0.5, 1000.0), (2, 0.502, 1000.0), (3, 0.501, 1000.0)], "throughput")
        cand = guarded("t1", "candidate", [(1, 0.1, 500.0), (2, 0.1, 500.0), (3, 0.1, 500.0)], "throughput")
        decision = decide(spec, baseline, cand, "val_accuracy")
        assert decision["verdict"] == "rejected_regress"  # the objective would have been accepted_gain
        assert "guardrail:throughput" in decision["reasons"]

    def test_guardrail_direction_min_breaches_on_a_rise(self):
        spec = {**SPEC, "confirm": {"k": 2.0, "noise_floor_rel": 0.005, "guardrails": ["latency"],
                                    "guard_abs_epsilon": 0.01, "guardrail_directions": {"latency": "min"}}}
        baseline = guarded("b", "baseline", [(1, 0.5, 1.0), (2, 0.502, 1.0), (3, 0.501, 1.0)], "latency", se=0.01)
        cand = guarded("t1", "candidate", [(1, 0.6, 2.0), (2, 0.602, 2.0), (3, 0.601, 2.0)], "latency", se=0.01)
        decision = decide(spec, baseline, cand, "val_accuracy")
        assert decision["verdict"] == "rejected_regress"
        assert "guardrail:latency" in decision["reasons"]

    def test_guardrail_direction_min_accepts_an_improvement(self):
        spec = {**SPEC, "confirm": {"k": 2.0, "noise_floor_rel": 0.005, "guardrails": ["latency"],
                                    "guard_abs_epsilon": 0.01, "guardrail_directions": {"latency": "min"}}}
        baseline = guarded("b", "baseline", [(1, 0.5, 2.0), (2, 0.502, 2.0), (3, 0.501, 2.0)], "latency")
        cand = guarded("t1", "candidate", [(1, 0.6, 0.5), (2, 0.602, 0.5), (3, 0.601, 0.5)], "latency")
        decision = decide(spec, baseline, cand, "val_accuracy")
        assert decision["verdict"] == "accepted_gain"
        assert not any(r.startswith("guardrail:") for r in decision["reasons"])
