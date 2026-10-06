"""CPU-only unit tests: in-flight cap (AR-LN-008) and confirm run-slot reserve (AR-LN-009)."""
from __future__ import annotations

from foundationskills.skills.auto_research.concurrency import (
    concurrency_check,
    reserve_check,
    reserve_state,
    resolve_max_in_flight,
    state_index,
)


def job_entry(job_id="111", trial="t1", seed=101, phase="confirm", **extra):
    entry = {"trial": trial, "seed": seed, "job_id": job_id, "phase": phase}
    entry.update(extra)
    return entry


def result_entry(trial="t1", seed=101, **extra):
    entry = {"trial": trial, "seed": seed, "ok": True}
    entry.update(extra)
    return entry


def spec(max_in_flight=None, max_nodes=7, max_runs=6, reserve_frac=0.3):
    cluster = {"max_nodes": max_nodes}
    if max_in_flight is not None:
        cluster["max_in_flight"] = max_in_flight
    return {"cluster": cluster, "budget": {"max_runs": max_runs, "reserve_frac": reserve_frac}}


def test_resolve_max_in_flight_explicit_wins():
    assert resolve_max_in_flight(spec(max_in_flight=3)) == 3


def test_resolve_max_in_flight_default_is_max_nodes():
    assert resolve_max_in_flight({"cluster": {"max_nodes": 7}}) == 7


def test_resolve_max_in_flight_invalid_falls_back_to_max_nodes():
    for bad in (0, -1, False, True, 2.0, "2", None, float("nan"), [], 1.5):
        cluster = {"cluster": {"max_nodes": 7, "max_in_flight": bad}}
        assert resolve_max_in_flight(cluster) == 7, bad


def test_pending_and_running_both_hold_a_slot():
    # ledger default: a submitted job with no settlement is in flight (fail closed)
    index = state_index([job_entry()], [], set())
    assert index["in_flight"] == ["111"]
    assert index["terminal"] == []
    assert index["count"] == 1
    assert index["drops"] == []

    def state_fn(job_id):  # live states: PENDING and RUNNING both hold a slot
        return {"111": "pending", "222": "running"}[job_id]

    index = state_index([job_entry(), job_entry("222", trial="t2")], [], set(), state_fn)
    assert index["in_flight"] == ["111", "222"]
    assert index["terminal"] == []
    assert index["count"] == 2
    assert index["drops"] == []


def test_trial_result_retires_the_slot():
    index = state_index([job_entry()], [result_entry()], set())
    assert index["in_flight"] == []
    assert index["terminal"] == ["111"]
    assert index["count"] == 0
    assert index["drops"] == []
    # a crash result and a (trial, seed) results map retire it the same way
    crash = result_entry(ok=False, returncode=1)
    assert state_index([job_entry()], [crash], set())["terminal"] == ["111"]
    assert state_index([job_entry()], {("t1", 101): {"ok": True}}, set())["terminal"] == ["111"]


def test_cancelled_job_ids_retire_the_slot():
    assert state_index([job_entry()], [], {"111"})["terminal"] == ["111"]
    payloads = [{"job_ids": ["111"], "reason": "prune", "token": "x"}]
    assert state_index([job_entry()], [], payloads)["terminal"] == ["111"]


def test_state_fn_unknown_holds_a_slot_and_unmeasures_the_count():
    for answer in ("unknown", None):
        index = state_index([job_entry()], [], set(), lambda job_id, a=answer: a)
        assert index["in_flight"] == ["111"]
        assert index["terminal"] == []
        assert index["drops"] == ["state_unmeasured:111"]
        assert index["count"] is None  # never 0


def test_state_fn_terminal_settles_and_ledger_settlement_beats_unknown():
    index = state_index([job_entry()], [], set(), lambda job_id: "terminal")
    assert index["terminal"] == ["111"]
    assert index["count"] == 0
    assert index["drops"] == []
    # a ledgered trial_result settles the run even when the probe says "unknown"
    index = state_index([job_entry()], [result_entry()], set(), lambda job_id: "unknown")
    assert index["terminal"] == ["111"]
    assert index["drops"] == []
    assert index["count"] == 0


def test_entry_without_any_settle_path_is_unmeasured():
    index = state_index([{"phase": "screening"}], [], set())
    assert index["in_flight"] == ["entry:0"]
    assert index["drops"] == ["state_unmeasured:entry:0"]
    assert index["count"] is None


def test_duplicate_job_id_is_a_named_drop_and_counts_once():
    index = state_index([job_entry(), job_entry(job_id="111")], [], set())
    assert index["in_flight"] == ["111"]
    assert index["drops"] == ["duplicate_job:111"]
    assert index["count"] == 1


def test_ledger_shaped_entries_are_unwrapped():
    wrapped = {"op": "job_submitted", "campaign": "c1", "payload": job_entry()}
    assert state_index([wrapped], [], set())["in_flight"] == ["111"]


def test_concurrency_check_below_cap_admits_at_cap_refuses():
    cap2 = spec(max_in_flight=2, max_nodes=2)
    two = [job_entry(), job_entry("222", trial="t2")]
    assert concurrency_check(cap2, two[:1], [], set()) == []  # count == cap-1 admits
    findings = concurrency_check(cap2, two, [], set())
    assert findings == [("AR-LN-008", "in_flight_cap:2/2")]  # count == cap refuses
    cap1 = spec(max_in_flight=1, max_nodes=4)  # the AR-LN-008 fixture shape
    assert concurrency_check(cap1, two[:1], [], set()) == [("AR-LN-008", "in_flight_cap:1/1")]
    assert concurrency_check(cap1, two[:1], [result_entry()], set()) == []


def test_concurrency_check_unmeasured_fails_closed():
    cap2 = spec(max_in_flight=2, max_nodes=2)
    two = [job_entry(), job_entry("222", trial="t2")]
    findings = concurrency_check(cap2, two, [], set(), lambda job_id: "unknown")
    assert findings == [("AR-LN-008", "in_flight_unmeasured:2")]
    mixed = concurrency_check(
        cap2, two, [], set(), lambda job_id: "unknown" if job_id == "222" else "running"
    )
    assert mixed == [("AR-LN-008", "in_flight_unmeasured:1")]
    settled = [result_entry(trial="t1", seed=101)]
    closed = concurrency_check(cap2, two, settled, set(), lambda job_id: "unknown")
    assert closed == [("AR-LN-008", "in_flight_unmeasured:1")]


def test_reserve_state_six_runs_three_tenths_reserves_two():
    launches = [{"trial": "t1", "seed": 101}, {"trial": "t1", "seed": 102}, {}, {}]
    state = reserve_state({"budget": {"max_runs": 6, "reserve_frac": 0.3}}, launches)
    assert state == {"reserved_runs": 2, "max_runs": 6, "used_runs": 4, "free_runs": 0}


def test_reserve_state_defaults_and_unusable_fracs_read_point_three():
    for budget in (
        {"max_runs": 10},
        {"max_runs": 10, "reserve_frac": float("nan")},
        {"max_runs": 10, "reserve_frac": float("inf")},
        {"max_runs": 10, "reserve_frac": "0.5"},
        {"max_runs": 10, "reserve_frac": True},
        {"max_runs": 10, "reserve_frac": -0.2},
        {"max_runs": 10, "reserve_frac": 1.5},
    ):
        state = reserve_state({"budget": budget}, [])
        assert state == {"reserved_runs": 3, "max_runs": 10, "used_runs": 0, "free_runs": 7}


def test_reserve_state_ceil_and_one_row_per_run():
    state = reserve_state({"budget": {"max_runs": 7, "reserve_frac": 0.5}}, [{"trial": "t1", "seed": 101}])
    assert state["reserved_runs"] == 4  # ceil(3.5)
    assert state["used_runs"] == 1  # one launch row == one (trial, seed) run
    assert state["free_runs"] == 2


def test_reserve_check_non_confirm_locked_at_the_reserve_boundary():
    budget = {"budget": {"max_runs": 6, "reserve_frac": 0.3}}  # 2 run slots kept for confirm
    three = [{}, {}, {}]  # used + 1 == max_runs - reserved_runs: admitted
    four = [{}, {}, {}, {}]  # used + 1 >  max_runs - reserved_runs: 5 + 2 > 6 runs
    assert reserve_check(budget, "screening", three) == []
    assert reserve_check(budget, "baseline", three) == []
    assert reserve_check(budget, "screening", four) == [("AR-LN-009", "reserve_locked_for_confirm:runs")]
    assert reserve_check(budget, "baseline", four + [{}]) == [("AR-LN-009", "reserve_locked_for_confirm:runs")]


def test_reserve_check_confirm_may_spend_the_reserve():
    budget = {"budget": {"max_runs": 6, "reserve_frac": 0.3}}
    rows = [{}, {}, {}, {}, {}]  # 5 used runs already reach into the reserve
    assert reserve_check(budget, "confirm", rows) == []
    assert reserve_check(budget, "confirm", rows + [{}]) == []
    assert reserve_check(budget, "screening", rows) == [("AR-LN-009", "reserve_locked_for_confirm:runs")]


def test_reserve_check_omits_the_gpu_hours_variant_per_amendment_a3():
    # amendment A3: run slots only; the hours reserve stays AR-LN-002's role gate
    findings = reserve_check({"budget": {"max_runs": 6, "reserve_frac": 0.3}}, "screening", [{}, {}, {}, {}])
    assert findings == [("AR-LN-009", "reserve_locked_for_confirm:runs")]
    assert all(message.endswith(":runs") for _, message in findings)
