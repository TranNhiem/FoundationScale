"""Envelope consent tokens, derived trial tokens and the budget-decremented AR-LN-006 gate."""
from __future__ import annotations

import re

from foundationskills.skills.auto_research.envelope import (
    budget_snapshot,
    envelope_check,
    envelope_token,
    trial_launch_token,
)
from foundationskills.skills.auto_research.ledger import sha256_hex

HEX64 = re.compile(r"^[0-9a-f]{64}$")
CONFIRM = "sha256:" + "c" * 64
SPEC_BUDGET = {"max_runs": 3, "gpu_hours_total": 30.0, "per_run_timeout_h": 8.0}


def _envelope(budget: dict | None = None) -> dict:
    """The consent record: the derived token hangs off the approved payload."""
    payload = {
        "campaign": "c1",
        "spec_hash": "sha256:" + "a" * 64,
        "budget": dict(budget or SPEC_BUDGET),
        "approver": "operator",
    }
    return {"envelope_token": envelope_token(CONFIRM, payload), **payload}


def _spec(budget: dict | None = None) -> dict:
    return {"budget": dict(budget or SPEC_BUDGET)}


def _trial(est: float = 5.0, name: str = "t1") -> dict:
    return {"trial": name, "role": "candidate", "kind": "train", "seed": 1, "gpu_hours_est": est}


def _usage(used: float = 0.0) -> dict:
    return {"used_gpu_hours": used, "per_job": {}, "drops": []}


class TestTokens:
    def test_formulae_are_pinned(self):
        assert envelope_token("c", {"a": 1}) == sha256_hex(b"fs-ar-envelope-v1|c|" + b'{"a":1}')
        assert trial_launch_token("tok", {"a": 1}) == sha256_hex(b"fs-ar-trial-v1|tok|" + b'{"a":1}')

    def test_envelope_token_is_deterministic_hex_over_the_payload(self):
        record = _envelope()
        payload = {k: v for k, v in record.items() if k != "envelope_token"}
        token = envelope_token(CONFIRM, payload)
        assert HEX64.match(token) and token == envelope_token(CONFIRM, payload)
        assert record["envelope_token"] == token

    def test_envelope_token_changes_on_any_payload_byte_and_confirm(self):
        payload = {"campaign": "c1", "budget": {"max_runs": 3, "gpu_hours_total": 30.0}}
        token = envelope_token(CONFIRM, payload)
        assert envelope_token(CONFIRM, {"campaign": "c2", "budget": {"max_runs": 3, "gpu_hours_total": 30.0}}) != token
        assert envelope_token(CONFIRM, {"campaign": "c1", "budget": {"max_runs": 4, "gpu_hours_total": 30.0}}) != token
        assert envelope_token(CONFIRM + "x", payload) != token

    def test_trial_token_is_deterministic_and_key_order_independent(self):
        env_token = envelope_token(CONFIRM, {"campaign": "c1"})
        spec = {"trial": "t1", "delta": {"optim.lr": 0.05}, "gpu_hours_est": 5.0}
        flipped = {"gpu_hours_est": 5.0, "delta": {"optim.lr": 0.05}, "trial": "t1"}
        token = trial_launch_token(env_token, spec)
        assert HEX64.match(token)
        assert token == trial_launch_token(env_token, spec) == trial_launch_token(env_token, flipped)

    def test_trial_token_changes_on_any_trial_spec_byte(self):
        env_token = "e" * 8
        base_spec = {"trial": "t1", "role": "candidate", "seed": 1, "delta": {"optim.lr": 0.05}, "gpu_hours_est": 5.0}
        base = trial_launch_token(env_token, base_spec)
        mutations = [
            {"trial": "t2", "role": "candidate", "seed": 1, "delta": {"optim.lr": 0.05}, "gpu_hours_est": 5.0},
            {"trial": "t1", "role": "baseline", "seed": 1, "delta": {"optim.lr": 0.05}, "gpu_hours_est": 5.0},
            {"trial": "t1", "role": "candidate", "seed": 2, "delta": {"optim.lr": 0.05}, "gpu_hours_est": 5.0},
            {"trial": "t1", "role": "candidate", "seed": 1, "delta": {"optim.lr": 0.5}, "gpu_hours_est": 5.0},
            {"trial": "t1", "role": "candidate", "seed": 1, "delta": {"optim.lr": 0.05}, "gpu_hours_est": 5.1},
            {"trial": "t1", "role": "candidate", "seed": 1, "delta": {"optim.lr": 0.05}},
        ]
        tokens = [trial_launch_token(env_token, mutation) for mutation in mutations]
        assert all(token != base for token in tokens)
        assert len(set(tokens) | {base}) == len(mutations) + 1

    def test_trial_token_changes_on_the_envelope_token(self):
        spec = _trial()
        assert trial_launch_token("a" * 8, spec) != trial_launch_token("b" * 8, spec)


class TestEnvelopeCheck:
    def test_admits_a_funded_trial_with_its_derived_token(self):
        record = _envelope()
        trial = _trial()
        token = trial_launch_token(record["envelope_token"], trial)
        prior = [{"trial": "t0", "job_id": "1", "launch_token": "l0", "kind": "train", "gpu_hours_est": 5.0}]
        assert envelope_check(_spec(), record, trial, prior, _usage(5.0), token) == []

    def test_envelope_missing(self):
        trial, token = _trial(), "x"
        for absent in (None, {}, "envelope"):
            assert envelope_check(_spec(), absent, trial, [], _usage(), token) == [("AR-LN-006", "envelope_missing")]
        bare = {"envelope_token": "tok", "approver": "operator"}  # no budget block
        assert envelope_check(_spec(), bare, trial, [], _usage(), token) == [("AR-LN-006", "envelope_missing")]
        no_token = _envelope()
        del no_token["envelope_token"]
        assert envelope_check(_spec(), no_token, trial, [], _usage(), token) == [("AR-LN-006", "envelope_missing")]

    def test_token_mismatch(self):
        record = _envelope()
        trial = _trial()
        forgery = trial_launch_token(record["envelope_token"], _trial(name="t9"))
        for supplied in ("forged", forgery, None):
            assert envelope_check(_spec(), record, trial, [], _usage(), supplied) == [("AR-LN-006", "token_mismatch")]

    def test_envelope_budget_must_mirror_the_spec_budget(self):
        record = _envelope()
        trial = _trial()
        token = trial_launch_token(record["envelope_token"], trial)
        wrong_runs = _spec({"max_runs": 4, "gpu_hours_total": 30.0})
        assert envelope_check(wrong_runs, record, trial, [], _usage(), token) == [("AR-LN-006", "envelope_budget_mismatch")]
        wrong_hours = _spec({"max_runs": 3, "gpu_hours_total": 3.0})
        assert envelope_check(wrong_hours, record, trial, [], _usage(), token) == [("AR-LN-006", "envelope_budget_mismatch")]

    def test_budget_mirror_compares_numerically(self):
        record = _envelope()
        trial = _trial()
        token = trial_launch_token(record["envelope_token"], trial)
        numeric = _spec({"max_runs": 3.0, "gpu_hours_total": 30})
        assert envelope_check(numeric, record, trial, [], _usage(), token) == []

    def test_envelope_exhausted_on_runs(self):
        budget = {"max_runs": 1, "gpu_hours_total": 100.0}
        record, spec = _envelope(budget), _spec(budget)
        trial = _trial()
        token = trial_launch_token(record["envelope_token"], trial)
        prior = [{"trial": "t0", "job_id": "123456", "launch_token": "x", "kind": "train", "gpu_hours_est": 1.0}]
        assert envelope_check(spec, record, trial, prior, _usage(1.0), token) == [("AR-LN-006", "envelope_exhausted:runs")]

    def test_measured_hours_burn_the_envelope_not_the_declared_estimate(self):
        budget = {"max_runs": 5, "gpu_hours_total": 10.0}
        record, spec = _envelope(budget), _spec(budget)
        prior = [{"trial": "t0", "job_id": "1", "launch_token": "l0", "kind": "train", "gpu_hours_est": 1.0}]
        trial = _trial(est=6.0)
        token = trial_launch_token(record["envelope_token"], trial)
        measured = {"used_gpu_hours": 4.5, "per_job": {"1": {"gpu_hours": 4.5, "source": "measured_sacct"}}, "drops": []}
        declared = {"used_gpu_hours": 1.0, "per_job": {"1": {"gpu_hours": 1.0, "source": "declared_est"}}, "drops": ["sacct_unavailable:1"]}
        # measured 4.5 + est 6.0 > 10.0 ⇒ exhausted, where declared 1.0 + est 6.0 admits
        assert envelope_check(spec, record, trial, prior, measured, token) == [("AR-LN-006", "envelope_exhausted:gpu_hours")]
        assert envelope_check(spec, record, trial, prior, declared, token) == []

    def test_exhaustion_boundary_is_strict(self):
        budget = {"max_runs": 5, "gpu_hours_total": 10.0}
        record, spec = _envelope(budget), _spec(budget)
        trial = _trial(est=6.0)
        token = trial_launch_token(record["envelope_token"], trial)
        exact = {"used_gpu_hours": 4.0, "per_job": {}, "drops": []}
        assert envelope_check(spec, record, trial, [], exact, token) == []
        over = {"used_gpu_hours": 4.5, "per_job": {}, "drops": []}
        assert envelope_check(spec, record, trial, [], over, token) == [("AR-LN-006", "envelope_exhausted:gpu_hours")]

    def test_problems_accumulate_in_named_order(self):
        record = _envelope({"max_runs": 1, "gpu_hours_total": 1.0})
        trial = _trial(est=5.0)
        prior = [{"trial": "t0", "job_id": "1", "gpu_hours_est": 1.0}]
        problems = envelope_check(_spec(), record, trial, prior, _usage(0.5), "forged")
        assert problems == [
            ("AR-LN-006", "envelope_budget_mismatch"),
            ("AR-LN-006", "token_mismatch"),
            ("AR-LN-006", "envelope_exhausted:runs"),
            ("AR-LN-006", "envelope_exhausted:gpu_hours"),
        ]


class TestBudgetSnapshot:
    def test_budget_after_decrements_per_launch(self):
        record = _envelope({"max_runs": 3, "gpu_hours_total": 30.0})
        trial = _trial(est=5.0)
        snaps = [
            budget_snapshot(record, [], _usage(0.0), trial),
            budget_snapshot(record, [{"trial": "p1", "gpu_hours_est": 5.0}], _usage(5.0), trial),
            budget_snapshot(record, [{"trial": "p1"}, {"trial": "p2"}], _usage(10.0), trial),
        ]
        assert [s["budget_after"]["runs_left"] for s in snaps] == [2, 1, 0]
        assert [s["budget_after"]["hours_left"] for s in snaps] == [25.0, 20.0, 15.0]

    def test_budget_after_counts_measured_hours(self):
        record = _envelope({"max_runs": 3, "gpu_hours_total": 10.0})
        trial = _trial(est=2.0)
        snap = budget_snapshot(record, [{"trial": "p1", "gpu_hours_est": 1.0}], _usage(4.5), trial)
        assert snap == {"budget_after": {"runs_left": 1, "hours_left": 3.5}}

    def test_snapshot_is_numeric_and_tolerant_of_junk(self):
        record = _envelope({"max_runs": 3, "gpu_hours_total": 30.0})
        snap = budget_snapshot(record, [], {}, _trial(est=5.0))
        assert snap["budget_after"] == {"runs_left": 2, "hours_left": 25.0}
        assert isinstance(snap["budget_after"]["runs_left"], int)
        assert isinstance(snap["budget_after"]["hours_left"], float)
        empty = budget_snapshot({}, [], {}, _trial())
        assert empty["budget_after"] == {"runs_left": -1, "hours_left": -5.0}
