"""C8 phase 1 skill triage: label functional failures and cost regressions to a skill via no-skill pairing.

Phase 1: no-skill pairing only; proposals enter eval sets only after operator approval.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

SCHEMA = "fskills.triage/1"
COST_RATIO = 1.5


def case_passed(call) -> bool:
    graded = int(call.get("items_graded", 0) or 0)
    return bool(call.get("measured")) and graded > 0 and int(call.get("items_met", 0) or 0) == graded


def agent_cost(call) -> int:
    return int(call.get("agent_tokens", 0) or 0) + int(call.get("agent_prompt_tokens", 0) or 0)


def pair_calls(report) -> list[dict]:
    buckets: dict[tuple, dict[str, dict]] = {}
    for call in report.get("calls", []):
        if call.get("measured") is not True:
            continue
        key = (str(call.get("package")), str(call.get("case_id")), call.get("rep"))
        buckets.setdefault(key, {})[call.get("arm")] = call
    pairs = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1], str(k[2]))):
        arms = buckets[key]
        if "with_skill" in arms and "without_skill" in arms:
            pairs.append({"package": key[0], "case_id": key[1], "rep": key[2],
                          "with": arms["with_skill"], "without": arms["without_skill"]})
    return pairs


def classify(pair) -> dict | None:
    with_ok, without_ok = case_passed(pair["with"]), case_passed(pair["without"])
    base = {"package": pair["package"], "case_id": pair["case_id"], "rep": pair["rep"]}
    if without_ok and not with_ok:
        return {"kind": "functional", "direction": "skill_hurt", **base}
    if with_ok and not without_ok:
        return {"kind": "functional", "direction": "skill_helped", **base}
    wc, nc = agent_cost(pair["with"]), agent_cost(pair["without"])
    if with_ok and without_ok and wc > COST_RATIO * max(1, nc):
        return {"kind": "cost", "direction": "skill_cost", "ratio": round(wc / max(1, nc), 2), **base}
    return None


def triage(report) -> dict:
    pairs = pair_calls(report)
    findings = [f for f in (classify(p) for p in pairs) if f]
    findings.sort(key=lambda f: (f["package"], f["case_id"], str(f["rep"]), f["kind"], f["direction"]))
    per_package: dict[str, dict] = {}
    for p in pairs:
        slot = per_package.setdefault(p["package"],
                                     {"skill_hurt": 0, "skill_helped": 0, "skill_cost": 0, "pairs": 0})
        slot["pairs"] += 1
    for f in findings:
        per_package[f["package"]][f["direction"]] += 1
    return {"schema": SCHEMA, "pairs": len(pairs), "findings": findings, "per_package": per_package}


def proposal_id(finding) -> str:
    payload = f"{finding['package']}|{finding['case_id']}|{finding['kind']}|{finding['direction']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_proposals(path) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def propose(findings, path, report) -> int:
    known = {row.get("id") for row in load_proposals(path)}
    by_key = {(p["package"], p["case_id"], str(p["rep"])): p for p in pair_calls(report)}
    appended = 0
    for finding in findings:
        if finding.get("direction") not in ("skill_hurt", "skill_cost"):
            continue
        pid = proposal_id(finding)
        if pid in known:
            continue
        pair = by_key.get((finding["package"], finding["case_id"], str(finding.get("rep"))), {})
        with_call, without_call = pair.get("with", {}), pair.get("without", {})
        row = {"schema": SCHEMA, "id": pid, "status": "proposed",
               "package": finding["package"], "case_id": finding["case_id"],
               "kind": finding["kind"], "direction": finding["direction"]}
        if "ratio" in finding:
            row["ratio"] = finding["ratio"]
        row["evidence"] = {
            "with_excerpt": (with_call.get("answer_excerpt") or "")[:400],
            "without_excerpt": (without_call.get("answer_excerpt") or "")[:400],
            "with_cost": agent_cost(with_call),
            "without_cost": agent_cost(without_call),
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        known.add(pid)
        appended += 1
    return appended


def approve(path, proposal_id, approver: str) -> dict:
    rows = load_proposals(path)
    target = None
    for row in rows:
        if row.get("id") == proposal_id:
            target = row
            break
    if target is None:
        raise KeyError(proposal_id)
    if target.get("status") != "proposed":
        raise ValueError(f"proposal {proposal_id} is not 'proposed'")
    target["status"] = "approved"
    target["approved_by"] = approver
    target["approved_utc"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return target
