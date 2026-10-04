"""Guardrail measurement tests: an unmeasured guardrail is a failure, never a silent skip."""
from __future__ import annotations

from foundationskills.skills.auto_research.accept import decide

FINGER = "sha256:" + "a1" * 32

SPEC = {
    "id": "c1",
    "objective": {"metric": "val_accuracy", "direction": "max", "benchmarks": []},
    "eval_policy": {"fingerprint": FINGER, "metrics": ["val_accuracy", "throughput"]},
    "confirm": {
        "k": 2.0, "noise_floor_rel": 0.005, "guardrails": ["throughput"], "guard_abs_epsilon": 0.01,
    },
    "seeds": {"baseline_repeats": 3, "confirm_repeats": 3, "seed_list": [1, 2, 3]},
}


def result(trial, role, seed, value, guard=None, guard_metric="throughput", se=0.01, **overrides):
    """One run carrying the objective metric and, when measured, one guard metric."""
    metrics = {"val_accuracy": {"value": value, "se": 0.001}}
    if guard is not None:
        metrics[guard_metric] = {"value": guard, "se": se}
    row = {
        "trial": trial, "role": role, "seed": seed, "status": "ok", "limited": False, "steps": 100,
        "eval_policy_fingerprint": FINGER, "metrics": metrics,
    }
    row.update(overrides)
    return row


BASELINE = [result("b", "baseline", 1, 0.5, guard=1000.0), result("b", "baseline", 2, 0.502, guard=1000.0),
            result("b", "baseline", 3, 0.501, guard=1000.0)]


class TestUnmeasuredGuardrail:
    def test_single_guard_pair_is_a_failure(self):
        # three objective pairs, but only one measured guard pair: a skip must not be accepted
        cand = [result("t1", "candidate", 1, 0.9, guard=1000.0),
                result("t1", "candidate", 2, 0.902),
                result("t1", "candidate", 3, 0.901)]
        decision = decide(SPEC, BASELINE, cand, "val_accuracy")
        assert decision["verdict"] == "unmeasured"
        assert decision["n_pairs"] == 3  # only the guardrail is unmeasured
        assert "guardrail_unmeasured:throughput:1 pairs" in decision["reasons"]
        assert not any(r.startswith("guardrail_skipped:") for r in decision["reasons"])

    def test_zero_guard_pairs_is_a_failure(self):
        cand = [result("t1", "candidate", 1, 0.9, guard=1000.0, guard_metric="latency"),
                result("t1", "candidate", 2, 0.902, guard=1000.0, guard_metric="latency"),
                result("t1", "candidate", 3, 0.901, guard=1000.0, guard_metric="latency")]
        decision = decide(SPEC, BASELINE, cand, "val_accuracy")
        assert decision["verdict"] == "unmeasured"
        assert "guardrail_unmeasured:throughput:0 pairs" in decision["reasons"]

    def test_measured_guardrail_is_not_unmeasured(self):
        cand = [result("t1", "candidate", 1, 0.9, guard=1000.0),
                result("t1", "candidate", 2, 0.902, guard=1000.0),
                result("t1", "candidate", 3, 0.901, guard=1000.0)]
        decision = decide(SPEC, BASELINE, cand, "val_accuracy")
        assert decision["verdict"] == "accepted_gain"
        assert not any(r.startswith("guardrail_unmeasured:") or r.startswith("guardrail_skipped:")
                       for r in decision["reasons"])
