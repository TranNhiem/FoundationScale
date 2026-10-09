from __future__ import annotations

from typing import Any

import pytest

from foundationskills.skills.auto_research.campaign import (
    PROPOSER_NAMES,
    check_llm_spec,
    check_proposer_spec,
    check_spec,
)

METRICS = ["val_accuracy", "toks_per_sec", "mem_gb", "val_loss"]

VALID_LLM: dict[str, Any] = {
    "pool_key": "nightly-llm",
    "model": "m-core",
    "max_cards": 3,
    "max_calls": 6,
    "max_evidence_chars": 8000,
    "parse_spec_version": 1,
    "sampling": {"temperature": 0.2, "max_tokens": 1200},
}


def _spec(**overrides: Any) -> dict[str, Any]:
    """Minimal launchable spec; dict overrides merge into the base block."""
    spec: dict[str, Any] = {
        "objective": {"metric": "val_accuracy", "direction": "max"},
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


def _llm_spec(llm: Any) -> dict[str, Any]:
    """Spec carrying an llm proposer with the given llm block."""
    return _spec(proposer={"name": "llm", "llm": llm})


def test_non_llm_or_absent_proposer_never_checked():
    assert check_llm_spec({}) == []
    assert check_llm_spec(_spec()) == []
    assert check_llm_spec(_spec(proposer=None)) == []
    assert check_llm_spec(_spec(proposer={"name": "catalog", "llm": {"bogus": 1}})) == []


def test_valid_llm_block_is_clean():
    assert check_llm_spec(_llm_spec(dict(VALID_LLM))) == []
    assert check_llm_spec(_llm_spec({"pool_key": "p", "model": "m"})) == []
    assert check_spec(_llm_spec(dict(VALID_LLM))) == []


def test_block_missing_or_not_dict_stops():
    assert check_llm_spec(_spec(proposer={"name": "llm"})) == [("AR-IN-010", "llm_config:block")]
    for value in (None, "x", 4, ["pool_key"], ("p",)):
        assert check_llm_spec(_spec(proposer={"name": "llm", "llm": value})) == [("AR-IN-010", "llm_config:block")]
    # the block finding stops the walk: no other llm_config message follows
    stopped = _spec(proposer={"name": "llm", "llm": "x", "min_rows": "nope"})
    assert check_llm_spec(stopped) == [("AR-IN-010", "llm_config:block")]
    assert check_proposer_spec(stopped) == [("AR-IN-008", "proposer.min_rows must be an int >= 0 (got 'nope')")]


BAD_LLM_CASES = [
    ({"model": "m"}, "llm_config:pool_key"),
    ({"pool_key": "", "model": "m"}, "llm_config:pool_key"),
    ({"pool_key": 7, "model": "m"}, "llm_config:pool_key"),
    ({"pool_key": "p"}, "llm_config:model"),
    ({"pool_key": "p", "model": ""}, "llm_config:model"),
    ({"pool_key": "p", "model": "m", "max_cards": 0}, "llm_config:max_cards"),
    ({"pool_key": "p", "model": "m", "max_cards": -1}, "llm_config:max_cards"),
    ({"pool_key": "p", "model": "m", "max_cards": "2"}, "llm_config:max_cards"),
    ({"pool_key": "p", "model": "m", "max_cards": True}, "llm_config:max_cards"),
    ({"pool_key": "p", "model": "m", "max_cards": 2.5}, "llm_config:max_cards"),
    ({"pool_key": "p", "model": "m", "max_cards": 7}, "llm_config:max_cards"),  # > budget.max_runs
    ({"pool_key": "p", "model": "m", "max_calls": 0}, "llm_config:max_calls"),
    ({"pool_key": "p", "model": "m", "max_calls": -3}, "llm_config:max_calls"),
    ({"pool_key": "p", "model": "m", "max_calls": 1.0}, "llm_config:max_calls"),
    ({"pool_key": "p", "model": "m", "max_calls": False}, "llm_config:max_calls"),
    ({"pool_key": "p", "model": "m", "max_evidence_chars": 0}, "llm_config:max_evidence_chars"),
    ({"pool_key": "p", "model": "m", "max_evidence_chars": "8k"}, "llm_config:max_evidence_chars"),
    ({"pool_key": "p", "model": "m", "parse_spec_version": 0}, "llm_config:parse_spec_version"),
    ({"pool_key": "p", "model": "m", "parse_spec_version": True}, "llm_config:parse_spec_version"),
    ({"pool_key": "p", "model": "m", "sampling": "temperature=0"}, "llm_config:sampling"),
    ({"pool_key": "p", "model": "m", "sampling": []}, "llm_config:sampling"),
    ({"pool_key": "p", "model": "m", "aaa": 1, "zzz": 2}, "llm_config:unknown:['aaa', 'zzz']"),
    ({"pool_key": "http://host:8000", "model": "m"}, "llm_config:url_shaped:pool_key"),
    ({"pool_key": "p", "model": "https://host/m"}, "llm_config:url_shaped:model"),
]


@pytest.mark.parametrize("llm,message", BAD_LLM_CASES)
def test_llm_field_messages(llm: dict[str, Any], message: str):
    assert check_llm_spec(_llm_spec(llm)) == [("AR-IN-010", message)]
    assert ("AR-IN-010", message) in check_spec(_llm_spec(llm))


def test_url_shaped_pool_key_and_model_both_flagged():
    problems = check_llm_spec(_llm_spec({"pool_key": "http://k", "model": "grpc://m"}))
    assert problems == [
        ("AR-IN-010", "llm_config:url_shaped:pool_key"),
        ("AR-IN-010", "llm_config:url_shaped:model"),
    ]


def test_llm_field_check_order():
    llm = {
        "max_cards": 0,
        "max_calls": 0,
        "max_evidence_chars": 0,
        "parse_spec_version": 0,
        "sampling": 0,
        "zzz": 1,
        "model": 7,
        "pool_key": 7,
    }
    assert check_llm_spec(_llm_spec(llm)) == [
        ("AR-IN-010", "llm_config:pool_key"),
        ("AR-IN-010", "llm_config:model"),
        ("AR-IN-010", "llm_config:max_cards"),
        ("AR-IN-010", "llm_config:max_calls"),
        ("AR-IN-010", "llm_config:max_evidence_chars"),
        ("AR-IN-010", "llm_config:parse_spec_version"),
        ("AR-IN-010", "llm_config:sampling"),
        ("AR-IN-010", "llm_config:unknown:['zzz']"),
    ]


def test_max_cards_upper_bound_only_against_int_max_runs():
    spec = _llm_spec({"pool_key": "p", "model": "m", "max_cards": 7})
    assert check_llm_spec(spec) == [("AR-IN-010", "llm_config:max_cards")]
    spec["budget"] = {**spec["budget"], "max_runs": "6"}
    assert check_llm_spec(spec) == []
    # boundary and default caps stay clean
    assert check_llm_spec(_llm_spec({"pool_key": "p", "model": "m", "max_cards": 6})) == []
    assert check_llm_spec(_llm_spec({"pool_key": "p", "model": "m", "max_cards": 3})) == []


def test_min_rows_zero_accepted_for_llm():
    assert check_proposer_spec({"proposer": {"name": "llm", "min_rows": 0}}) == []
    assert check_proposer_spec({"proposer": {"name": "llm", "min_rows": 0, "llm": VALID_LLM}}) == []


@pytest.mark.parametrize("min_rows,message", [(-1, "(got -1)"), (0.5, "(got 0.5)"), (True, "(got True)")])
def test_min_rows_still_typed_for_llm(min_rows: Any, message: str):
    problems = check_proposer_spec({"proposer": {"name": "llm", "min_rows": min_rows}})
    assert problems == [("AR-IN-008", f"proposer.min_rows must be an int >= 0 {message}")]


@pytest.mark.parametrize("name", ["catalog", "optuna", "optuna-cma"])
def test_min_rows_zero_still_refused_for_every_other_name(name: str):
    problems = check_proposer_spec({"proposer": {"name": name, "min_rows": 0}})
    assert problems == [("AR-IN-008", "proposer.min_rows must be an int >= 1 (got 0)")]


def test_ar_in_008_vizier_fixture_still_fires():
    problems = check_proposer_spec({"axes": [], "proposer": {"name": "vizier", "min_rows": 0}})
    assert problems == [
        ("AR-IN-008", f"proposer.name 'vizier' not in {list(PROPOSER_NAMES)}"),
        ("AR-IN-008", "proposer.min_rows must be an int >= 1 (got 0)"),
    ]


def test_llm_key_only_valid_with_name_llm():
    message = "proposer.llm is only valid with name 'llm'"
    assert check_proposer_spec({"proposer": {"name": "catalog", "llm": VALID_LLM}}) == [("AR-IN-008", message)]
    assert check_proposer_spec({"proposer": {"llm": {"pool_key": "p"}}}) == [("AR-IN-008", message)]
    assert check_proposer_spec({"proposer": {"name": "optuna", "llm": {}}}) == [("AR-IN-008", message)]
    # with name 'llm' the key is known and never an unknown-key complaint
    assert check_proposer_spec({"proposer": {"name": "llm", "llm": VALID_LLM}}) == []


def test_check_spec_of_non_llm_spec_unchanged():
    assert check_spec(_spec()) == []
    assert check_spec(_spec(proposer={"name": "catalog", "min_rows": 2, "require_model": False, "seed": 0})) == []
    problems = check_spec(_spec(proposer={"name": "vizier", "min_rows": 0}))
    assert problems == [
        ("AR-IN-008", f"proposer.name 'vizier' not in {list(PROPOSER_NAMES)}"),
        ("AR-IN-008", "proposer.min_rows must be an int >= 1 (got 0)"),
    ]
    assert all(rule != "AR-IN-010" for rule, _ in problems)


def test_check_spec_streams_llm_findings_last():
    spec = _llm_spec({"pool_key": "p", "model": "m", "sampling": 3})
    spec["base"] = {**spec["base"], "model": ""}
    problems = check_spec(spec)
    assert problems == [
        ("AR-IN-002", "base.model missing or empty: ''"),
        ("AR-IN-010", "llm_config:sampling"),
    ]
