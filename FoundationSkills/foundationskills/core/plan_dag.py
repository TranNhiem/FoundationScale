from __future__ import annotations
"""Typed plan DAG for skill composition (fs.plan.v1).

Plans live outside the transcript as typed artifacts: nodes declare a skill,
a request and done_when closure conditions.  Provide validate_plan/topo_order
plus closure() so "done" is claimed only when every closure check holds.
"""
from dataclasses import dataclass
from typing import Iterable, Mapping

SCHEMA = "fs.plan.v1"
CLOSURE_SCHEMA = "fs.plan_closure.v1"
VERDICTS = ("PASS", "RED", "UNMEASURED", "REFUSED")
_RULE_FORMS = ("rule:", "no_rule:")


@dataclass(frozen=True)
class PlanNode:
    id: str
    skill: str
    goal: str
    request: dict
    deps: tuple[str, ...] = ()
    done_when: tuple[str, ...] = ("status:PASS",)
    budget_wall_seconds: float | None = None


@dataclass(frozen=True)
class Plan:
    id: str
    goal: str
    nodes: tuple[PlanNode, ...]


def _bad(msg: str) -> ValueError:
    return ValueError(msg)


def plan_from_dict(d: dict) -> Plan:
    if not isinstance(d, dict):
        raise _bad("plan must be a dict")
    for key in ("id", "goal", "nodes"):
        if key not in d:
            raise _bad(f"plan missing key: {key}")
    if not isinstance(d["id"], str) or not isinstance(d["goal"], str):
        raise _bad("plan id/goal must be strings")
    if not isinstance(d["nodes"], (list, tuple)):
        raise _bad("plan nodes must be a list")
    nodes: list[PlanNode] = []
    for raw in d["nodes"]:
        if not isinstance(raw, dict):
            raise _bad("each node must be a dict")
        for key in ("id", "skill", "goal", "request"):
            if key not in raw:
                raise _bad(f"node missing key: {key}")
        if not all(isinstance(raw[k], str) for k in ("id", "skill", "goal")):
            raise _bad("node id/skill/goal must be strings")
        if not isinstance(raw["request"], dict):
            raise _bad("node request must be a dict")
        pairs = {}
        for key in ("deps", "done_when"):
            val = raw.get(key, ())
            val = tuple(val) if isinstance(val, (list, tuple)) else ()
            if any(not isinstance(v, str) for v in val):
                raise _bad(f"node {key} must be strings")
            pairs[key] = val
        budget = raw.get("budget_wall_seconds", None)
        if budget is not None and not isinstance(budget, (int, float)):
            raise _bad("budget_wall_seconds must be a number or None")
        nodes.append(PlanNode(
            id=raw["id"], skill=raw["skill"], goal=raw["goal"],
            request=dict(raw["request"]), deps=pairs["deps"],
            done_when=pairs["done_when"] or ("status:PASS",),
            budget_wall_seconds=None if budget is None else float(budget)))
    return Plan(id=d["id"], goal=d["goal"], nodes=tuple(nodes))


def plan_payload(plan: Plan) -> dict:
    return {
        "schema": SCHEMA,
        "id": plan.id,
        "goal": plan.goal,
        "nodes": [{
            "id": n.id, "skill": n.skill, "goal": n.goal, "request": dict(n.request),
            "deps": list(n.deps), "done_when": list(n.done_when),
            "budget_wall_seconds": n.budget_wall_seconds,
        } for n in plan.nodes],
    }


def _done_when_ok(entry: str) -> bool:
    if entry == "status:PASS":
        return True
    return any(entry.startswith(p) and len(entry) > len(p) for p in _RULE_FORMS)


def _find_cycle(nodes: tuple[PlanNode, ...]) -> list[str] | None:
    order = {n.id: i for i, n in enumerate(nodes)}
    edges = {n.id: [d for d in n.deps if d in order and d != n.id] for n in nodes}
    color = {nid: 0 for nid in order}
    stack: list[str] = []

    def visit(nid: str) -> list[str] | None:
        color[nid] = 1
        stack.append(nid)
        for dep in sorted(edges[nid], key=order.__getitem__):
            if color[dep] == 1:
                return stack[stack.index(dep):] + [dep]
            if color[dep] == 0:
                found = visit(dep)
                if found:
                    return found
        stack.pop()
        color[nid] = 2
        return None

    for node in nodes:
        if color[node.id] == 0:
            found = visit(node.id)
            if found:
                return found
    return None


def validate_plan(plan: Plan, known_skills: Iterable[str]) -> list[str]:
    problems: list[str] = []
    skills = set(known_skills)
    if not plan.nodes:
        return ["plan has no nodes"]
    seen: set[str] = set()
    for node in plan.nodes:
        if node.id in seen:
            problems.append(f"duplicate node id: {node.id}")
        seen.add(node.id)
        if node.skill not in skills:
            problems.append(f"node {node.id}: unknown skill: {node.skill}")
        problems += [f"node {node.id}: dep {d} references no node"
                     for d in node.deps if d not in {n.id for n in plan.nodes}]
        if node.id in node.deps:
            problems.append(f"node {node.id}: depends on itself")
        if not node.done_when:
            problems.append(f"node {node.id}: done_when is empty")
        problems += [f"node {node.id}: malformed done_when entry: {e}"
                     for e in node.done_when if not _done_when_ok(e)]
        if node.budget_wall_seconds is not None and node.budget_wall_seconds < 0:
            problems.append(f"node {node.id}: negative budget_wall_seconds")
    cyc = _find_cycle(plan.nodes)
    if cyc:
        problems.append("cycle: " + " -> ".join(cyc))
    return problems


def topo_order(plan: Plan) -> list[str]:
    order = {n.id: i for i, n in enumerate(plan.nodes)}
    edges = {n.id: [d for d in n.deps if d in order] for n in plan.nodes}
    indeg = {nid: 0 for nid in order}
    for nid, deps in edges.items():
        for d in deps:
            indeg[nid] += (nid != d)  # self-edge: permanently blocked
            if d == nid:
                indeg[nid] = 1
    ready = sorted((n for n, k in indeg.items() if k == 0), key=order.__getitem__)
    out: list[str] = []
    while ready:
        nid = ready.pop(0)
        out.append(nid)
        for node in plan.nodes:
            if nid in edges[node.id] and node.id != nid:
                indeg[node.id] -= 1
                if indeg[node.id] == 0:
                    ready.append(node.id)
        ready.sort(key=order.__getitem__)
    if len(out) != len(plan.nodes):
        cyc = _find_cycle(plan.nodes) or [n.id for n in plan.nodes if n.id not in out]
        raise _bad("cycle: " + " -> ".join(cyc))
    return out


def node_done(node: PlanNode, status: str, rule_ids: Iterable[str]) -> bool:
    rules = set(rule_ids)
    if not node.done_when:
        return False
    for entry in node.done_when:
        if entry.startswith("status:"):
            if status != entry[len("status:"):]:
                return False
        elif entry.startswith("no_rule:"):
            if entry[len("no_rule:"):] in rules:
                return False
        elif entry.startswith("rule:"):
            if entry[len("rule:"):] not in rules:
                return False
        else:
            return False
    return True


def closure(plan: Plan, outcomes: Mapping[str, Mapping]) -> dict:
    states: dict[str, str] = {}
    reasons: list[str] = []
    have_red = have_unmeasured = False
    for node in plan.nodes:
        out = outcomes.get(node.id)
        if out is None:
            states[node.id] = "skipped"
            reasons.append(f"node {node.id} skipped: no outcome")
            have_red = True
            continue
        status = str(out.get("status", ""))
        rules = out.get("rule_ids", ()) or ()
        if status == "UNMEASURED":
            states[node.id] = "unmeasured"
            reasons.append(f"node {node.id} unmeasured: status=UNMEASURED")
            have_unmeasured = True
        elif status in ("RED", "REFUSED") or not node_done(node, status, rules):
            states[node.id] = "failed"
            reasons.append(f"node {node.id} failed: status={status or '?'}")
            have_red = True
        else:
            states[node.id] = "done"
    verdict = "RED" if have_red else ("UNMEASURED" if have_unmeasured else "PASS")
    return {"schema": CLOSURE_SCHEMA, "plan_id": plan.id, "verdict": verdict,
            "nodes": states, "reasons": reasons}
