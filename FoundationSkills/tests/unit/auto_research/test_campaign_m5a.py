from __future__ import annotations

from typing import Any

import pytest

from foundationskills.skills.auto_research.campaign import (
    check_objectives_spec,
    check_spec,
)

METRICS = ["val_accuracy", "toks_per_sec", "mem_gb", "val_loss"]


def _o(metric: str, direction: str) -> dict[str, str]:
    """One valid objectives[] entry."""
    return {"metric": metric, "direction": direction}


def _spec(**overrides: Any) -> dict[str, Any]:
    """Minimal launchable spec carrying a valid 2-entry objectives block."""
    spec: dict[str, Any] = {
        "objective": {"metric": "val_accuracy", "direction": "max"},
        "objectives": [_o("val_accuracy", "max"), _o("toks_per_sec", "max")],
        "eval_policy": {"metrics": list(METRICS), "fingerprint": "sha256:" + "a" * 64},
        "base": {"model": "m-core", "fingerprint": "sha256:" + "b" * 64},
        "budget": {"gpu_hours_total": 120.0, "max_runs": 6, "per_run_timeout_h": 24.0, "reserve_frac": 0.3},
        "cluster": {"time": "10-00:00:00", "exclude": ["r01dgx02"], "max_nodes": 2, "gpus_per_node": 8},
        "seeds": {
            "baseline_repeats": 3,
            "confirm_repeats": 3,
            "seed_list": [101, 102, 103],
            "screening_repeats": 1,
        },
        "axes": [],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(spec.get(key), dict):
            spec[key] = {**spec[key], **value}
        else:
            spec[key] = value
    return spec


def test_absent_and_none_objectives_are_clean():
    """AR-IN-009 stays silent without the block (M4 path untouched)."""
    free = _spec()
    del free["objectives"]
    assert check_objectives_spec(free) == []
    assert check_objectives_spec(_spec(objectives=None)) == []


def test_valid_two_and_four_objective_specs_are_clean():
    assert check_objectives_spec(_spec()) == []
    assert check_spec(_spec()) == []
    four = _spec(objectives=[
        _o("val_accuracy", "max"),
        _o("toks_per_sec", "max"),
        _o("mem_gb", "min"),
        _o("val_loss", "min"),
    ])
    assert check_objectives_spec(four) == []
    assert check_spec(four) == []


@pytest.mark.parametrize("raw, expected", [
    ({"metric": "toks_per_sec"}, "objectives_count:dict"),
    ("val_accuracy,toks_per_sec", "objectives_count:str"),
    ([_o("val_accuracy", "max")], "objectives_count:1"),
    ([_o("val_accuracy", "max")] * 5, "objectives_count:5"),
])
def test_objectives_count_gate(raw: Any, expected: str):
    assert check_objectives_spec(_spec(objectives=raw)) == [("AR-IN-009", expected)]


@pytest.mark.parametrize("bad", [
    "oops",
    {"metric": "toks_per_sec"},
    {"metric": "", "direction": "max"},
    {"metric": 7, "direction": "max"},
    {"metric": "toks_per_sec", "direction": "up"},
    {"metric": "toks_per_sec", "direction": "max", "weight": 1},
])
def test_objective_shape_messages(bad: Any):
    spec = _spec(objectives=[_o("val_accuracy", "max"), bad])
    assert check_objectives_spec(spec) == [("AR-IN-009", "objective_shape:1")]


def test_objective_shape_at_index_zero_skips_cross_checks():
    spec = _spec(objectives=[{"metric": "", "direction": "max"}, _o("toks_per_sec", "max")])
    assert check_objectives_spec(spec) == [("AR-IN-009", "objective_shape:0")]


@pytest.mark.parametrize("first, tag", [
    (_o("toks_per_sec", "max"), "toks_per_sec"),
    (_o("val_accuracy", "min"), "val_accuracy"),
])
def test_objectives0_mismatch(first: dict[str, str], tag: str):
    spec = _spec(objectives=[first, _o("mem_gb", "min")])
    assert check_objectives_spec(spec) == [("AR-IN-009", f"objectives0_mismatch:{tag}")]


def test_objective_direction_absent_defaults_to_max():
    spec = _spec()
    spec["objective"] = {"metric": "val_accuracy"}
    assert check_objectives_spec(spec) == []
    spec["objective"] = {"metric": "val_accuracy", "direction": "min"}
    assert check_objectives_spec(spec) == [("AR-IN-009", "objectives0_mismatch:val_accuracy")]


@pytest.mark.parametrize("extra", [
    [_o("toks_per_sec", "min")],
    [_o("toks_per_sec", "min"), _o("toks_per_sec", "max")],
])
def test_objective_metric_duplicate_once_per_metric(extra: list[dict[str, str]]):
    spec = _spec(objectives=[_o("val_accuracy", "max"), _o("toks_per_sec", "max"), *extra])
    assert check_objectives_spec(spec) == [("AR-IN-009", "objective_metric_duplicate:toks_per_sec")]


def test_objective_not_in_eval_policy():
    spec = _spec(objectives=[_o("val_accuracy", "max"), _o("ghost", "max")])
    assert check_objectives_spec(spec) == [("AR-IN-009", "objective_not_in_eval_policy:ghost")]


def test_guardrail_is_objective():
    for guardrails in (["toks_per_sec"], {"toks_per_sec": {"direction": "max"}}):
        spec = _spec(confirm={"guardrails": guardrails})
        assert check_objectives_spec(spec) == [("AR-IN-009", "guardrail_is_objective:toks_per_sec")]


def test_must_fire_objectives_invalid_mismatch_and_guardrail_overlap():
    spec = _spec(
        objectives=[_o("val_accuracy", "min"), _o("toks_per_sec", "max")],
        confirm={"guardrails": ["toks_per_sec"]},
    )
    assert check_objectives_spec(spec) == [
        ("AR-IN-009", "objectives0_mismatch:val_accuracy"),
        ("AR-IN-009", "guardrail_is_objective:toks_per_sec"),
    ]


def test_finding_order_count_shapes_mismatch_duplicates_eval_policy_guardrails():
    spec = _spec(
        objectives=[_o("val_accuracy", "max"), {"metric": "toks_per_sec"}, _o("ghost", "min"), _o("ghost", "max")],
        confirm={"guardrails": ["ghost"]},
    )
    assert check_objectives_spec(spec) == [
        ("AR-IN-009", "objective_shape:1"),
        ("AR-IN-009", "objective_metric_duplicate:ghost"),
        ("AR-IN-009", "objective_not_in_eval_policy:ghost"),
        ("AR-IN-009", "guardrail_is_objective:ghost"),
    ]


def test_check_spec_surfaces_ar_in_009_only_for_multi_objective_specs():
    broken = _spec(objectives=[_o("val_accuracy", "min"), _o("toks_per_sec", "max")])
    assert check_spec(broken) == [("AR-IN-009", "objectives0_mismatch:val_accuracy")]


def test_check_spec_unchanged_without_objectives_block():
    clean = _spec()
    del clean["objectives"]
    assert check_spec(clean) == []
    assert check_spec(clean) == check_spec(_spec(objectives=None))
    free = _spec(budget={"max_runs": 0})
    del free["objectives"]
    findings = check_spec(free)
    assert findings == check_spec(_spec(objectives=None, budget={"max_runs": 0}))
    assert not any(rule == "AR-IN-009" for rule, _ in findings)


@pytest.mark.parametrize("block", ["objective", "eval_policy", "confirm", "budget", "base"])
def test_junk_spec_blocks_refuse_instead_of_crashing(block: str):
    """A non-mapping spec block reads as empty: check_spec returns findings, it never raises."""
    spec = _spec(objectives=[_o("val_accuracy", "max"), _o("throughput", "max")])
    spec[block] = "junk"
    assert check_spec(spec)
