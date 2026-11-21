"""Unit tests for foundationskills.agent.skill_triage (C8 phase 1, atlas 2608.11888)."""

import pytest

from foundationskills.agent import skill_triage as st

def _call(arm, pkg, case_id, met, graded, tokens, prompt=None, excerpt="", measured=True):
    call = {"arm": arm, "package": pkg, "case_id": case_id, "rep": 0, "measured": measured,
            "items_met": met, "items_graded": graded, "agent_tokens": tokens,
            "judge_tokens": 7, "answer_excerpt": excerpt}
    if prompt is not None:
        call["agent_prompt_tokens"] = prompt
    return call

def _report():
    """Hurt, helped, cost and neutral pairs in two packages, plus one unpaired case."""
    return {"schema": st.SCHEMA, "calls": [
        _call("with_skill", "alpha", "hurt", 1, 2, 100, 20, "A" * 500),  # hurts the skill
        _call("without_skill", "alpha", "hurt", 2, 2, 40, 10, "B" * 450),
        _call("with_skill", "beta", "helped", 2, 2, 50, 10),  # helps the skill
        _call("without_skill", "beta", "helped", 1, 2, 50, 10),
        _call("with_skill", "alpha", "cost", 2, 2, 150),  # 3x cost, prompt tokens absent
        _call("without_skill", "alpha", "cost", 2, 2, 40, 10),
        _call("with_skill", "beta", "neutral", 2, 2, 60, 0),  # flat cost
        _call("without_skill", "beta", "neutral", 2, 2, 40, 10),
        _call("with_skill", "beta", "unpaired", 2, 2, 10),  # only one arm measured
        _call("without_skill", "beta", "unpaired", 2, 2, 10, measured=False)]}

def _pair(case_id):
    return next(p for p in st.pair_calls(_report()) if p["case_id"] == case_id)

def _findings():
    return [f for f in (st.classify(p) for p in st.pair_calls(_report())) if f]

def _proposals(tmp_path):
    report, path = _report(), tmp_path / "proposals.jsonl"
    assert st.propose(_findings(), path, report) == 2
    return path

def test_case_passed_requires_measured_arms_and_fully_graded_items():
    assert not st.case_passed(_call("w", "a", "c", 1, 2, 1))
    assert not st.case_passed(_call("w", "a", "c", 0, 0, 1))
    assert not st.case_passed(_call("w", "a", "c", 3, 0, 1))
    assert not st.case_passed(_call("w", "a", "c", 2, 2, 1, measured=False))
    assert st.case_passed(_call("w", "a", "c", 2, 2, 1))

def test_agent_cost_sums_tokens_and_defaults_missing_prompt_tokens_to_zero():
    assert st.agent_cost(_call("w", "a", "c", 1, 1, 30)) == 30
    assert st.agent_cost(_call("w", "a", "c", 1, 1, 30, 5)) == 35

def test_pair_calls_pairs_both_measured_arms_sorted_and_skips_unpaired_case():
    pairs = st.pair_calls(_report())
    assert [(p["package"], p["case_id"], p["rep"]) for p in pairs] == [
        ("alpha", "cost", 0), ("alpha", "hurt", 0), ("beta", "helped", 0),
        ("beta", "neutral", 0)]
    assert [p["with"]["arm"] for p in pairs] == ["with_skill"] * 4
    assert [p["without"]["arm"] for p in pairs] == ["without_skill"] * 4

def test_classify_hurt_case_without_passes_with_fails_is_functional_skill_hurt():
    finding = st.classify(_pair("hurt"))
    assert (finding["kind"], finding["direction"]) == ("functional", "skill_hurt")
    assert (finding["package"], finding["case_id"], finding["rep"]) == ("alpha", "hurt", 0)

def test_classify_helped_case_with_passes_without_fails_is_functional_skill_helped():
    finding = st.classify(_pair("helped"))
    assert (finding["kind"], finding["direction"]) == ("functional", "skill_helped")
    assert (finding["package"], finding["case_id"]) == ("beta", "helped")

def test_classify_cost_case_both_pass_at_triple_tokens_is_cost_skill_cost():
    finding = st.classify(_pair("cost"))
    assert (finding["kind"], finding["direction"]) == ("cost", "skill_cost")
    assert finding["ratio"] == 3.0 and finding["case_id"] == "cost"

def test_classify_neutral_case_both_pass_at_flat_cost_returns_none():
    assert st.classify(_pair("neutral")) is None

def test_triage_reports_pairs_sorted_findings_and_per_package_counts():
    result = st.triage(_report())
    assert result["schema"] == st.SCHEMA and result["pairs"] == 4
    assert sorted(f["direction"] for f in result["findings"]) == [
        "skill_cost", "skill_helped", "skill_hurt"]
    assert result["per_package"]["alpha"] == {
        "skill_hurt": 1, "skill_helped": 0, "skill_cost": 1, "pairs": 2}
    assert result["per_package"]["beta"] == {
        "skill_hurt": 0, "skill_helped": 1, "skill_cost": 0, "pairs": 2}

def test_proposal_id_is_stable_across_reps_and_distinct_per_finding():
    hurt = st.classify(_pair("hurt"))
    assert st.proposal_id(hurt) == st.proposal_id(dict(hurt, rep=9))
    assert st.proposal_id(hurt) != st.proposal_id(st.classify(_pair("cost")))
    assert len(st.proposal_id(hurt)) == 12

def test_propose_appends_only_hurt_and_cost_rows_with_truncated_evidence(tmp_path):
    path = _proposals(tmp_path)
    rows = {row["case_id"]: row for row in st.load_proposals(path)}
    assert sorted(rows) == ["cost", "hurt"] and len(rows) == 2
    hurt = rows["hurt"]
    assert (hurt["schema"], hurt["status"], hurt["kind"]) == (st.SCHEMA, "proposed", "functional")
    assert (hurt["package"], hurt["direction"]) == ("alpha", "skill_hurt")
    assert hurt["evidence"]["with_excerpt"] == "A" * 400
    assert hurt["evidence"]["without_excerpt"] == "B" * 400
    assert (hurt["evidence"]["with_cost"], hurt["evidence"]["without_cost"]) == (120, 50)
    assert rows["cost"]["ratio"] == 3.0 and rows["cost"]["direction"] == "skill_cost"

def test_propose_is_idempotent_and_the_second_call_appends_zero_rows(tmp_path):
    report, path = _report(), tmp_path / "proposals.jsonl"
    assert st.propose(_findings(), path, report) == 2  # creates the missing file
    assert st.propose(_findings(), path, report) == 0
    assert [r["status"] for r in st.load_proposals(path)] == ["proposed"] * 2

def test_load_proposals_on_missing_file_returns_empty_list(tmp_path):
    assert st.load_proposals(tmp_path / "absent.jsonl") == []

def test_approve_flips_status_and_records_approver_and_timestamp(tmp_path):
    path = _proposals(tmp_path)
    rows = st.load_proposals(path)
    row = st.approve(path, rows[0]["id"], "reviewer@ai")
    assert row["status"] == "approved" and row["approved_by"] == "reviewer@ai"
    assert "approved_utc" in row
    stored = {r["id"]: r for r in st.load_proposals(path)}
    assert stored[row["id"]]["status"] == "approved" and len(stored) == 2

def test_approve_unknown_id_raises_key_error(tmp_path):
    _proposals(tmp_path)
    with pytest.raises(KeyError):
        st.approve(tmp_path / "proposals.jsonl", "missing-id", "reviewer@ai")

def test_approve_twice_raises_value_error(tmp_path):
    path = _proposals(tmp_path)
    pid = st.load_proposals(path)[0]["id"]
    st.approve(path, pid, "first@ai")
    with pytest.raises(ValueError):
        st.approve(path, pid, "second@ai")
