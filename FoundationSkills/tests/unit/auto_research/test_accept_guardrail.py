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


class TestRlFloorGuardrail:
    """M6 (D6 revised): rl_measured_fraction is checked against the declared floor, never paired with a base
    model that has no RL fraction (the GPU campaign closed UNMEASURED on every rl spec before this)."""

    RL_SPEC = {**SPEC, "rl_min_measured_fraction": 0.5,
               "confirm": {**SPEC["confirm"], "guardrails": ["rl_measured_fraction"],
                           "guardrail_directions": {"rl_measured_fraction": "max"}}}
    RL_BASE = [result("b", "baseline", s, v) for s, v in ((1, 0.5), (2, 0.502), (3, 0.501))]

    def _cand(self, fractions, value=0.9):
        return [result("t1", "candidate", s, value + s * 0.001, guard=f, guard_metric="rl_measured_fraction", se=0.0)
                for s, f in zip((1, 2, 3), fractions)]

    def test_base_model_without_fraction_is_measured(self):
        decision = decide(self.RL_SPEC, self.RL_BASE, self._cand([0.66, 0.76, 0.74]), "val_accuracy")
        assert decision["verdict"] == "accepted_gain"
        assert not any(r.startswith("guardrail_unmeasured") for r in decision["reasons"])

    def test_candidate_row_below_floor_breaches(self):
        decision = decide(self.RL_SPEC, self.RL_BASE, self._cand([0.66, 0.4, 0.74]), "val_accuracy")
        assert decision["verdict"] == "rejected_regress"
        assert any(r.startswith("guardrail:rl_measured_fraction:below_floor:0.4<0.5") for r in decision["reasons"])

    def test_too_few_candidate_fractions_stay_unmeasured(self):
        cand = self._cand([0.66, 0.76, 0.74])
        for row in cand[1:]:
            del row["metrics"]["rl_measured_fraction"]
        decision = decide(self.RL_SPEC, self.RL_BASE, cand, "val_accuracy")
        assert decision["verdict"] == "unmeasured"
        assert "guardrail_unmeasured:rl_measured_fraction:1 candidate rows" in decision["reasons"]

    def test_without_declared_floor_the_guard_stays_paired(self):
        spec = {k: v for k, v in self.RL_SPEC.items() if k != "rl_min_measured_fraction"}
        decision = decide(spec, self.RL_BASE, self._cand([0.66, 0.76, 0.74]), "val_accuracy")
        assert "guardrail_unmeasured:rl_measured_fraction:0 pairs" in decision["reasons"]
