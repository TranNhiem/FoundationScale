"""Regression tests for admission hashing, safety gates, pairing, and governor refusals."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

import foundationskills.agent.admit as admit_module
from foundationskills.agent.self_improve_evidence import paired_cases, pairing_coverage
import foundationskills.core.activation_governor as ag


NOW = datetime(2030, 1, 15, 12, 0, tzinfo=timezone.utc)
FUTURE = "2030-01-16T00:00:00+00:00"


def make_record(**changes):
    values = {
        "trigger_id": "trigger-1",
        "skill": "example-skill",
        "owner": "test-owner",
        "scope": "test",
        "condition": "an operator-approved condition",
        "expires_at": FUTURE,
        "budget_cap": 3,
        "daily_cap": 3,
        "action_class": "proposing",
        "confirm_policy": "spec-only",
        "provenance": "unit-test",
        "quiet_hours": None,
    }
    values.update(changes)
    return ag.TriggerRecord(**values)


def evidence():
    return {
        "beacon": {"condition": "activate", "fired": 1},
        "paired": {"target": "behaviour_case_pass", "pairs": [[0.0, 1.0]]},
        "eval_case_ids": ["case-a", "case-b"],
    }


def payload_hash(value):
    return admit_module.admission_hash(
        admit_module.admission_payload("skill", "old text", "new text", value)
    )


@pytest.mark.parametrize(
    "key,replacement",
    [
        ("beacon", {"condition": "activate", "fired": 2}),
        ("paired", {"target": "behaviour_case_pass", "pairs": [[0.0, 0.0]]}),
        ("eval_case_ids", ["case-a", "case-c"]),
    ],
)
def test_admission_hash_binds_each_gate_input(key, replacement):
    original = evidence()
    changed = deepcopy(original)
    changed[key] = replacement

    assert payload_hash(changed) != payload_hash(original)



@pytest.mark.parametrize(
    "screen,expected_verdict,expected_reason",
    [
        (
            ([], ["X"]),
            "UNMEASURED",
            "safety screen incomplete: X unmeasured",
        ),
        (
            (["unsafe span added (Y): ..."], ["X"]),
            "RED",
            "unsafe span added (Y): ...",
        ),
    ],
)
def test_admit_distinguishes_unmeasured_and_red_safety_screens(
    monkeypatch, screen, expected_verdict, expected_reason
):
    old_text, new_text, supplied_evidence = "old text", "new text", evidence()
    confirm = payload_hash(supplied_evidence)
    monkeypatch.setattr(
        admit_module, "_safety_screen", lambda old, new: screen
    )

    result = admit_module.admit(
        "skill", old_text, new_text, supplied_evidence, confirm
    )

    assert result["verdict"] == expected_verdict
    assert expected_reason in result["reasons"]


def call(case_id, *, rep=0, graded=1, met=None):
    if met is None:
        met = graded
    return {
        "case_id": case_id,
        "rep": rep,
        "arm": "with_skill",
        "package": "pkg",
        "measured": True,
        "items_graded": graded,
        "items_met": met,
    }


def test_pairing_skips_ungraded_calls_and_reports_coverage():
    incumbent = {
        "calls": [
            call("both", met=0),
            call("ungraded", graded=0),
            call("incumbent-only"),
        ]
    }
    candidate = {
        "calls": [
            call("both", met=1),
            call("ungraded"),
            call("candidate-only"),
        ]
    }

    assert paired_cases(incumbent, candidate, "pkg") == [[0.0, 1.0]]
    assert pairing_coverage(incumbent, candidate, "pkg") == {
        "incumbent_keys": 3,
        "candidate_keys": 3,
        "paired": 1,
        "incumbent_only": 1,
        "candidate_only": 1,
        "ungraded": 1,
    }


def test_duplicate_case_rep_key_raises_value_error():
    incumbent = {"calls": [call("duplicate-case"), call("duplicate-case")]}

    with pytest.raises(ValueError, match="duplicate"):
        paired_cases(incumbent, {"calls": []}, "pkg")


def test_none_ledger_never_proposes_and_does_not_consume_caps():
    governor = ag.Governor(
        [make_record(budget_cap=1, daily_cap=1)], None, global_daily_cap=1
    )

    for value in (1, 2, 3):
        result = governor.on_event(
            "trigger-1", "source", {"value": value}, NOW, True
        )
        assert result["decision"] == "ledger_unwritable"
        assert result["proposal"] is None


def test_permission_error_reading_ledger_refuses_governor_start(
    tmp_path, monkeypatch
):
    def deny_read(path):
        raise PermissionError(f"cannot read {path}")

    monkeypatch.setattr(ag.ledger, "read", deny_read)

    with pytest.raises(ag.GovernorRefusal, match="ledger unreadable"):
        ag.Governor([make_record()], tmp_path)


def test_unidentifiable_calls_make_pairing_unmeasured() -> None:
    from foundationskills.agent.self_improve_evidence import paired_cases

    good = {"arm": "with_skill", "package": "p", "measured": True, "case_id": "c1", "rep": 0,
            "items_graded": 1, "items_met": 1}
    bad = {**good, "case_id": None}
    with pytest.raises(ValueError, match="usable"):
        paired_cases({"calls": [good]}, {"calls": [bad]}, "p")
