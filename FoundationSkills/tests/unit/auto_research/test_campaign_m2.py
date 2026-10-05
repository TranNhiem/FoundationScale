"""AR-IN-007 M2 spec shims: seeds.screening_repeats, cluster.max_in_flight and the seed-list shape."""
from __future__ import annotations

from foundationskills.skills.auto_research.campaign import check_spec

_FINGERPRINT = "sha256:" + "a" * 64


def _spec(**overrides):
    """Minimal check_spec-clean campaign spec (these tests stand alone; no shared helper in this tree)."""
    spec = {
        "objective": {"metric": "val/loss", "direction": "min"},
        "eval_policy": {"metrics": ["val/loss"], "fingerprint": _FINGERPRINT},
        "base": {"model": "m-core", "fingerprint": "sha256:" + "b" * 64},
        "budget": {"gpu_hours_total": 120.0, "max_runs": 6, "per_run_timeout_h": 24.0, "reserve_frac": 0.3},
        "cluster": {"time": "10-00:00:00", "exclude": ["r01dgx02"], "max_nodes": 2, "gpus_per_node": 8},
        "seeds": {"baseline_repeats": 3, "confirm_repeats": 3, "seed_list": [101, 102, 103], "screening_repeats": 1},
        "axes": [],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(spec.get(key), dict):
            spec[key] = {**spec[key], **value}
        else:
            spec[key] = value
    return spec


def _hits(problems, needle):
    return [msg for rule, msg in problems if rule == "AR-IN-007" and needle in msg]


def test_valid_spec_stays_clean():
    assert check_spec(_spec()) == []
    assert check_spec(_spec(cluster={"max_in_flight": 2}, seeds={"screening_repeats": 2})) == []


def test_max_in_flight_is_optional_and_binds_only_when_set():
    assert check_spec(_spec()) == []  # absent by default: the cap resolves to cluster.max_nodes
    assert check_spec(_spec(cluster={"max_in_flight": None})) == []
    assert check_spec(_spec(cluster={"max_in_flight": 6})) == []  # == budget.max_runs is fine
    assert check_spec(_spec(cluster={"max_in_flight": 5})) == []


def test_max_in_flight_must_be_an_int_ge_one_not_bool():
    for bad in (True, False, 0, -1, 2.5, "2", [2]):
        problems = check_spec(_spec(cluster={"max_in_flight": bad}))
        assert _hits(problems, "max_in_flight_invalid"), bad


def test_max_in_flight_over_budget_runs():
    msgs = _hits(check_spec(_spec(cluster={"max_in_flight": 7})), "max_in_flight_over_max_runs")
    assert len(msgs) == 1 and "7" in msgs[0] and "6" in msgs[0]
    assert not _hits(check_spec(_spec(cluster={"max_in_flight": 6})), "max_in_flight_over_max_runs")


def test_max_in_flight_over_check_silent_without_an_int_max_runs():
    problems = check_spec(_spec(budget={"max_runs": "many"}, cluster={"max_in_flight": 50}))
    assert [rule for rule, _ in problems] == ["AR-IN-003"]  # the budget complaint owns this shape error
    assert not _hits(problems, "max_in_flight")


def test_screening_repeats_is_optional_int_ge_one():
    spec = _spec()
    del spec["seeds"]["screening_repeats"]
    assert check_spec(spec) == []
    assert check_spec(_spec(seeds={"screening_repeats": None})) == []
    for bad in (True, 0, -2, 2.5, "1"):
        problems = check_spec(_spec(seeds={"screening_repeats": bad}))
        assert _hits(problems, "screening_repeats_invalid"), bad


def test_seed_list_must_be_unique_ints():
    assert check_spec(_spec(seeds={"seed_list": [101, 102], "confirm_repeats": 2})) == []
    assert _hits(check_spec(_spec(seeds={"seed_list": [101, 101, 103]})), "seed_list_duplicate")
    assert _hits(check_spec(_spec(seeds={"seed_list": [101, "102", True], "confirm_repeats": 2})), "seed_list_nonint")
    problems = check_spec(_spec(seeds={"seed_list": []}))
    assert _hits(problems, "seed_list_empty")
    assert not _hits(problems, "confirm_repeats_over_seed_list")  # no list left to compare against
    problems = check_spec(_spec(seeds={"seed_list": "101"}))
    assert _hits(problems, "seed_list_invalid")
    assert not _hits(problems, "seed_list_empty")


def test_confirm_repeats_bounded_by_seed_list():
    assert check_spec(_spec(seeds={"confirm_repeats": 3, "seed_list": [101, 102, 103]})) == []
    msgs = _hits(check_spec(_spec(seeds={"confirm_repeats": 5, "seed_list": [101, 102, 103]})),
                 "confirm_repeats_over_seed_list")
    assert len(msgs) == 1 and "5" in msgs[0] and "3" in msgs[0]
    for bad in (True, 0, -1, "3", 3.0):
        problems = check_spec(_spec(seeds={"confirm_repeats": bad}))
        assert _hits(problems, "confirm_repeats_invalid"), bad
    spec = _spec()
    del spec["seeds"]["confirm_repeats"]  # requiredness belongs to the AR-IN required-field rule (M1)
    assert check_spec(spec) == []


def test_must_fire_fixture_seed_plan_invalid():
    problems = check_spec(_spec(seeds={"baseline_repeats": 3, "confirm_repeats": 5, "seed_list": [101, 102, 103]}))
    assert [rule for rule, _ in problems] == ["AR-IN-007"]


def test_findings_keep_the_rule_message_tuple_shape():
    problems = check_spec(_spec(cluster={"max_in_flight": 0}, seeds={"screening_repeats": 0}))
    assert len(problems) == 2
    assert all(isinstance(rule, str) and isinstance(msg, str) and rule == "AR-IN-007" for rule, msg in problems)


def test_seeds_absence_is_left_to_the_required_field_path():
    spec = _spec()
    spec.pop("seeds")  # no crash and no invented seed complaint on M0-style specs
    assert check_spec(spec) == []


def test_other_rules_stay_untouched():
    problems = check_spec(_spec(cluster={"time": "01-00:00:00", "max_in_flight": 2}))
    assert [rule for rule, _ in problems] == ["AR-LN-004"]
