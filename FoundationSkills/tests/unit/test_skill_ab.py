"""A2.2 keep-only-if: paired with-skill A/B of two behaviour reports of one package."""

from __future__ import annotations

import copy

import pytest
from foundationskills.agent.skill_ab import ab_compare
from foundationskills.agent.skill_impact import SkillImpactRefused


def report(rates: dict[str, list[int]], *, bundle: str, unmeasured: int = 0) -> dict:
    """rates: case -> items met per rep (3 graded per call); one blind call per case."""
    calls = []
    for case, mets in rates.items():
        for rep, met in enumerate(mets):
            calls.append(
                {
                    "package": "fskills-x",
                    "arm": "with_skill",
                    "case_id": case,
                    "rep": rep,
                    "measured": True,
                    "items_met": met,
                    "items_graded": 3,
                }
            )
        calls.append(
            {
                "package": "fskills-x",
                "arm": "without_skill",
                "case_id": case,
                "rep": 0,
                "measured": True,
                "items_met": 0,
                "items_graded": 3,
            }
        )
    for i in range(unmeasured):
        calls.append(
            {
                "package": "fskills-x",
                "arm": "with_skill",
                "case_id": f"c{i}",
                "rep": 9,
                "measured": False,
            }
        )
    return {
        "schema": "fskills.behaviour_eval/1",
        "cases": "oracle",
        "agent_model": "a",
        "judge_model": "j",
        "reps": 3,
        "agent_temperature": 0.0,
        "calls": calls,
        "index": [
            {
                "name": "fskills-x",
                "package": "pkg.x",
                "cases_sha256": "c",
                "evals_sha256": "e",
                "skill_bundle_sha256": bundle,
            }
        ],
    }


BEFORE = {f"c{i}": [2, 2, 2] for i in range(8)}


def test_a_real_rise_with_an_upward_interval_is_kept() -> None:
    after = {f"c{i}": [3, 3, 3] for i in range(8)}
    out = ab_compare(report(BEFORE, bundle="b0"), report(after, bundle="b1"))
    assert (
        out["verdict"] == "keep" and out["rise"] > 0 and out["ci_low"] > 0 and out["n_paired"] == 8
    )


def test_a_rise_whose_interval_includes_zero_is_rejected() -> None:
    after = dict(BEFORE, c0=[3, 3, 3], c1=[1, 1, 1], c2=[3, 3, 2])
    out = ab_compare(report(BEFORE, bundle="b0"), report(after, bundle="b1"))
    assert out["rise"] > 0 and out["ci_low"] <= 0 and out["verdict"] == "reject"


def test_a_drop_is_rejected() -> None:
    after = {f"c{i}": [1, 1, 1] for i in range(8)}
    assert (
        ab_compare(report(BEFORE, bundle="b0"), report(after, bundle="b1"))["verdict"] == "reject"
    )


def test_unmeasured_calls_make_it_unmeasured_never_keep() -> None:
    after = {f"c{i}": [3, 3, 3] for i in range(8)}
    out = ab_compare(report(BEFORE, bundle="b0"), report(after, bundle="b1", unmeasured=1))
    assert out["verdict"] == "unmeasured" and "unmeasured" in out["reasons"][0]


@pytest.mark.parametrize(("field", "value"), [("cases_sha256", "other"), ("evals_sha256", "other")])
def test_different_case_sets_refuse(field: str, value: str) -> None:
    after = report(BEFORE, bundle="b1")
    after["index"][0][field] = value
    with pytest.raises(SkillImpactRefused, match="different cases"):
        ab_compare(report(BEFORE, bundle="b0"), after)


@pytest.mark.parametrize("field", ["agent_model", "judge_model", "reps", "cases"])
def test_different_measurement_settings_refuse(field: str) -> None:
    after = report(BEFORE, bundle="b1")
    after[field] = "changed"
    with pytest.raises(SkillImpactRefused, match=field):
        ab_compare(report(BEFORE, bundle="b0"), after)


def test_the_same_bundle_refuses_nothing_changed() -> None:
    with pytest.raises(SkillImpactRefused, match="nothing changed"):
        ab_compare(report(BEFORE, bundle="b0"), copy.deepcopy(report(BEFORE, bundle="b0")))
