"""C7.2: run_plan executes a typed plan DAG through run_step and closes it (skips are RED)."""

from types import SimpleNamespace

from foundationskills.core.contract import Finding, Severity, SkillContext, SkillResult
from foundationskills.core.status import Status
from foundationskills.skills._workflow import run_plan

CALLER = SimpleNamespace(name="fs-plan", version="1", invokes=("a", "b", "c"))


class Stub:
    def __init__(self, status, rules=()):
        self.status, self.rules = status, rules

    def execute(self, request, ctx):
        findings = tuple(Finding(r, Severity.WARN, "m", {}, "") for r in self.rules)
        refusal = "no" if self.status is Status.REFUSED else None
        return SkillResult(self.status, {}, findings=findings, refusal=refusal)


class Reg:
    def __init__(self, **skills):
        self.skills = skills

    def get(self, name):
        return self.skills[name]


def _plan(**over):
    nodes = [{"id": "n1", "skill": "a", "goal": "g", "request": {}},
             {"id": "n2", "skill": "b", "goal": "g", "request": {}, "deps": ["n1"]},
             {"id": "n3", "skill": "c", "goal": "g", "request": {}, "deps": ["n2"]}]
    return {"id": "p1", "goal": "demo", "nodes": nodes, **over}


def test_all_pass_closes_pass_and_writes_the_plan_artifact(tmp_path):
    out = run_plan(CALLER, _plan(), SkillContext(workdir=tmp_path),
                   Reg(a=Stub(Status.PASS), b=Stub(Status.PASS), c=Stub(Status.PASS)))
    assert out["verdict"] == "PASS" and len(out["steps"]) == 3 and out["plan_path"].endswith("plan.p1.json")


def test_a_failed_node_skips_its_dependents_and_the_plan_is_red(tmp_path):
    out = run_plan(CALLER, _plan(), SkillContext(workdir=tmp_path),
                   Reg(a=Stub(Status.PASS), b=Stub(Status.RED), c=Stub(Status.PASS)))
    assert out["verdict"] == "RED" and [s["node"] for s in out["steps"]] == ["n1", "n2"]
    assert any("n3 skipped" in r for r in out["reasons"])


def test_unmeasured_node_never_passes(tmp_path):
    plan = _plan(nodes=[{"id": "n1", "skill": "a", "goal": "g", "request": {}}])
    out = run_plan(CALLER, plan, SkillContext(workdir=tmp_path), Reg(a=Stub(Status.UNMEASURED)))
    assert out["verdict"] == "UNMEASURED"


def test_an_invalid_plan_is_refused_before_anything_runs(tmp_path):
    plan = _plan(nodes=[{"id": "n1", "skill": "zzz", "goal": "g", "request": {}}])
    out = run_plan(CALLER, plan, SkillContext(workdir=tmp_path), Reg())
    assert out["verdict"] == "REFUSED" and out["steps"] == [] and not (tmp_path / "artifacts").exists()


def test_done_when_no_rule_turns_a_pass_into_a_failure(tmp_path):
    plan = _plan(nodes=[{"id": "n1", "skill": "a", "goal": "g", "request": {}, "done_when": ["no_rule:CP-HO-005"]}])
    out = run_plan(CALLER, plan, SkillContext(workdir=tmp_path), Reg(a=Stub(Status.PASS, ("CP-HO-005",))))
    assert out["verdict"] == "RED"
