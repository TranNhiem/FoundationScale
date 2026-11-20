"""Unit tests for foundationskills.core.plan_dag (C7 increment 1: typed plan DAG)."""

from __future__ import annotations

import pytest

from foundationskills.core.plan_dag import (SCHEMA, Plan, PlanNode, closure, node_done,
                                            plan_from_dict, plan_payload, topo_order, validate_plan)

KNOWN = ["skill_a", "skill_b", "skill_c"]


def nd(nid, skill="skill_a", deps=(), done=("status:PASS",), budget=None, request=None):
    return PlanNode(id=nid, skill=skill, goal="g", request={} if request is None else request,
                    deps=tuple(deps), done_when=tuple(done), budget_wall_seconds=budget)


def pl(*nodes, pid="p1"):
    return Plan(id=pid, goal="demo", nodes=tuple(nodes))


def oc(status, rules=()):          # one closure outcome record
    return {"status": status, "rule_ids": list(rules)}


def txt(items):
    return " | ".join(str(i) for i in items).lower()


SRC = {
    "schema": SCHEMA, "id": "p1", "goal": "demo",
    "nodes": [
        {"id": "A", "skill": "skill_a", "goal": "do a", "request": {"q": 1}, "deps": [],
         "done_when": ["status:PASS"], "budget_wall_seconds": 3.5},
        {"id": "B", "skill": "skill_b", "goal": "do b", "request": {}, "deps": ["A"],
         "done_when": ["rule:R1", "no_rule:R2"], "budget_wall_seconds": None},
    ],
}


def test_plan_from_dict_round_trip_with_plan_payload():
    plan = plan_from_dict(SRC)
    assert isinstance(plan.nodes[1].deps, tuple) and plan.nodes[1].deps == ("A",)
    assert plan.nodes[1].done_when == ("rule:R1", "no_rule:R2") and plan.nodes[0].deps == ()
    pay = plan_payload(plan)
    assert pay == SRC                        # deps/done_when back to lists, plan order kept
    assert plan_from_dict(pay) == plan and plan_payload(plan_from_dict(pay)) == pay


def test_plan_from_dict_raises_value_error_on_missing_keys_or_bad_types():
    for bad in ({"id": "p1"}, {"id": "p1", "goal": "demo", "nodes": "nope"}):
        with pytest.raises(ValueError):
            plan_from_dict(bad)


def test_validate_diamond_A_to_BC_to_D_reports_no_problems():
    dia = pl(nd("A"), nd("B", deps=("A",)), nd("C", deps=("A",)), nd("D", deps=("B", "C")))
    assert validate_plan(dia, KNOWN) == []   # A -> B, C -> D diamond is clean


def test_validate_reports_each_defect():
    cases = [
        ("duplicate id", pl(nd("A"), nd("A", skill="skill_b")), "duplicate twice repeated", None),
        ("unknown skill", pl(nd("A", skill="nope")), "skill unknown invalid", "nope"),
        ("missing dep", pl(nd("A", deps=("GHOST",))), "dep missing unknown", "ghost"),
        ("self dep", pl(nd("N1", deps=("N1",))), "self itself cycle", "n1"),
        ("empty done_when", pl(nd("A", done=())), "done_when empty condition", None),
        ("malformed rule", pl(nd("A", done=("rule:", "junk"))), "done_when rule malformed", None),
        ("negative budget", pl(nd("A", budget=-1.5)), "budget negative", None),
    ]
    for label, bad, words, must in cases:
        problems = validate_plan(bad, KNOWN)
        t = txt(problems)
        assert problems, f"{label}: defect not reported"
        assert any(w in t for w in words.split()), f"{label}: {t}"
        if must:
            assert must in t, f"{label}: {t}"


def test_validate_reports_3_node_cycle_naming_its_nodes():
    cyc = pl(nd("CYC1", deps=("CYC3",)), nd("CYC2", deps=("CYC1",)), nd("CYC3", deps=("CYC2",)))
    t = txt(validate_plan(cyc, KNOWN))
    assert t and "cycle" in t and all(n.lower() in t for n in ("CYC1", "CYC2", "CYC3"))


def test_topo_order_respects_deps_then_plan_order():
    p = pl(nd("D", deps=("B", "C")), nd("B", deps=("A",)), nd("C", deps=("A",)), nd("A"))
    assert topo_order(p) == ["A", "B", "C", "D"]     # plan order breaks the B/C tie
    q = pl(nd("B"), nd("C", deps=("B",)), nd("A", deps=("B",)), nd("D", deps=("A", "C")))
    assert topo_order(q) == ["B", "C", "A", "D"]     # deps first, remaining ties by plan order


def test_topo_order_raises_on_cycle():
    cyc = pl(nd("CYC1", deps=("CYC3",)), nd("CYC2", deps=("CYC1",)), nd("CYC3", deps=("CYC2",)))
    with pytest.raises(ValueError):
        topo_order(cyc)


def test_node_done_status_pass_rule_and_no_rule_forms():
    plain, must, forb = nd("A"), nd("A", done=("rule:R1",)), nd("A", done=("no_rule:R2",))
    assert node_done(plain, "PASS", []) and not node_done(plain, "RED", [])
    assert not node_done(plain, "UNMEASURED", [])
    assert node_done(must, "PASS", ["R1", "R2"]) and not node_done(must, "PASS", ["R2"])
    assert node_done(forb, "PASS", ["R1"]) and not node_done(forb, "PASS", ["R1", "R2"])


def test_closure_all_done_is_pass():
    p = pl(nd("A"), nd("B", deps=("A",)))
    res = closure(p, {"A": oc("PASS"), "B": oc("PASS", ["R1"])})
    assert (res["schema"], res["plan_id"], res["verdict"]) == ("fs.plan_closure.v1", "p1", "PASS")
    assert res["nodes"] == {"A": "done", "B": "done"} and res["reasons"] == []


def test_closure_skipped_node_is_red_and_is_named():
    p = pl(nd("A"), nd("BA", deps=("A",)), nd("CC", deps=("BA",)))
    res = closure(p, {"A": oc("PASS"), "BA": oc("PASS")})     # "CC" did not run
    assert res["nodes"]["CC"] == "skipped"
    assert res["verdict"] == "RED" and "cc" in txt(res["reasons"])


def test_closure_unmet_done_when_marks_failed_then_red():
    p = pl(nd("A", done=("status:PASS", "no_rule:R2")))
    res = closure(p, {"A": oc("PASS", ["R2"])})               # no_rule:R2 violated -> failed
    assert res["nodes"]["A"] == "failed" and res["verdict"] == "RED"


def test_closure_unmeasured_node_gives_unmeasured_verdict():
    p = pl(nd("A"), nd("B", deps=("A",)), nd("C", deps=("B",)))
    res = closure(p, {"A": oc("PASS"), "B": oc("PASS"), "C": oc("UNMEASURED")})
    assert res["nodes"] == {"A": "done", "B": "done", "C": "unmeasured"}
    assert res["verdict"] == "UNMEASURED"


def test_closure_red_beats_unmeasured():
    p = pl(nd("A"), nd("B", deps=("A",)), nd("C", deps=("B",)))
    res = closure(p, {"A": oc("RED"), "B": oc("UNMEASURED"), "C": oc("PASS")})
    assert res["nodes"]["A"] == "failed" and res["nodes"]["B"] == "unmeasured"
    assert res["verdict"] == "RED" and res["nodes"]["C"] == "done"
