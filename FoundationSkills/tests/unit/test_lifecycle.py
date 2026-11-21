"""Unit tests for foundationskills.core.lifecycle (C10.2, card 2608.12851)."""

import json

import pytest

from foundationskills.core.ledger import record
from foundationskills.core.lifecycle import (
    MIN_CALLS,
    SCHEMA,
    STAGES,
    THRESHOLDS,
    approve,
    failure_rate,
    load_state,
    proposed_stage,
    proposals,
    routable,
    stage_of,
)


def line(skill, verdict="PASS", exit_code=0, kind="skill"):
    return record(kind, skill, "1", verdict=verdict, exit_code=exit_code, wall_clock_ms=1)


def measured(skill, n, rate):
    bad = round(n * rate)
    return [line(skill, "RED", 5) for _ in range(bad)] + [line(skill) for _ in range(n - bad)]


def test_failure_rate_counts_only_measured_skill_rows():
    rows = [
        line("s"),
        line("s", "RED", 5),
        line("s", "AMBER", 2),
        line("s", "ERROR", 7),
        record("tool", "s", "1", verdict="RED", exit_code=5, wall_clock_ms=1),
        line("other"),
    ]
    n, rate = failure_rate(rows, "s")
    assert n == 2  # only PASS/RED verdicts are measured outcomes
    assert rate == pytest.approx(1 / 2)


def test_failure_rate_for_unmeasured_skill_is_none():
    assert failure_rate([], "ghost") == (0, None)
    stray = [
        line("ghost", "AMBER", 2),
        record("tool", "ghost", "1", verdict="PASS", exit_code=0, wall_clock_ms=1),
    ]
    assert failure_rate(stray, "ghost") == (0, None)


@pytest.mark.parametrize(
    "n, rate, want",
    [
        (MIN_CALLS - 1, 1.0, None),
        (0, None, None),
        (MIN_CALLS, None, None),
        (MIN_CALLS, 0.0, "active"),
        (MIN_CALLS, 0.249, "active"),
        (MIN_CALLS, THRESHOLDS["downweighted"], "downweighted"),
        (MIN_CALLS, 0.40, "downweighted"),
        (MIN_CALLS, THRESHOLDS["quarantined"], "quarantined"),
        (MIN_CALLS, 0.74, "quarantined"),
        (MIN_CALLS, THRESHOLDS["retired"], "retired"),
        (MIN_CALLS * 5, 1.0, "retired"),
    ],
)
def test_proposed_stage_threshold_bands(n, rate, want):
    assert proposed_stage(n, rate) == want


def test_proposals_lists_only_changes_and_applies_nothing(tmp_path):
    path = tmp_path / "state.json"
    approve(path, "zebra", "downweighted", "bob", {"seed": True})
    before = path.read_bytes()
    state = load_state(path)
    rows = (
        measured("changed", MIN_CALLS, 0.30)
        + measured("steady", MIN_CALLS, 0.05)
        + measured("zebra", MIN_CALLS, 0.55)
        + measured("fresh", 3, 0.0)
    )
    got = proposals(rows, state, ["steady", "fresh", "changed", "zebra"])
    assert [d["skill"] for d in got] == ["changed", "zebra"]
    assert got[0]["from"] == "active" and got[0]["to"] == "downweighted"
    assert got[0]["n"] == MIN_CALLS and got[0]["failure_rate"] == pytest.approx(0.30)
    assert got[1]["from"] == "downweighted" and got[1]["to"] == "quarantined"
    assert path.read_bytes() == before
    assert stage_of(load_state(path), "zebra") == "downweighted"
    assert stage_of(load_state(path), "changed") == "active"


def test_approve_writes_stage_approver_and_evidence(tmp_path):
    path = tmp_path / "state.json"
    evidence = {"n": 21, "failure_rate": 0.55}
    entry = approve(path, "s", "quarantined", "alice", evidence)
    state = load_state(path)
    assert state["schema"] == SCHEMA
    stored = state["skills"]["s"]
    assert stored == entry
    assert stored["stage"] == "quarantined"
    assert stored["approved_by"] == "alice"
    assert stored["evidence"] == evidence
    assert stored["approved_utc"].replace("Z", "+00:00").endswith("+00:00") or ":" in stored["approved_utc"]
    assert stage_of(state, "s") == "quarantined"
    assert {p.name for p in tmp_path.iterdir()} == {"state.json"}
    assert isinstance(json.loads(path.read_text()), dict)
    for i, name in enumerate(STAGES):
        assert approve(path, "s", name, "op%d" % i, {"i": i})["stage"] == name
        assert stage_of(load_state(path), "s") == name


def test_approve_rejects_unknown_stage(tmp_path):
    path = tmp_path / "state.json"
    approve(path, "s", "active", "alice", {"ok": True})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        approve(path, "s", "sideways", "bob", {})
    assert load_state(path)["skills"]["s"] == json.loads(before.decode())["skills"]["s"]


def test_routable_false_for_quarantined_or_retired():
    state = load_state("/nonexistent/fskills/state.json")
    assert routable(state, "fresh") is True
    for name in ("active", "downweighted", "quarantined", "retired"):
        state["skills"][name] = {
            "stage": name,
            "approved_by": "a",
            "approved_utc": "2024-01-01T00:00:00+00:00",
            "evidence": {},
        }
    assert routable(state, "active") is True
    assert routable(state, "downweighted") is True
    assert routable(state, "quarantined") is False
    assert routable(state, "retired") is False


def test_load_state_missing_file_gives_empty_skills(tmp_path):
    missing = tmp_path / "no" / "such" / "state.json"
    state = load_state(missing)
    assert state == {"schema": SCHEMA, "skills": {}}
    assert not missing.exists()


def test_front_gate_skips_an_approved_quarantined_skill(tmp_path, monkeypatch):
    from foundationskills.agent.route_front_gate import front_gate
    from foundationskills.agent.trigger_registry import compile_registry

    index = [{"name": "fskills-compression", "description": "Quantize and compress model weights with RTN.",
              "when_to_use": ["quantize gemma to int4"], "invocation": "model"}]
    reg = compile_registry(index)
    assert front_gate("quantize gemma int4", reg)["candidates"]
    state = tmp_path / "lifecycle.json"
    approve(state, "fskills-compression", "quarantined", "op", {"note": "test"})
    monkeypatch.setenv("FSKILLS_LIFECYCLE_FILE", str(state))
    assert front_gate("quantize gemma int4", reg)["candidates"] == []
