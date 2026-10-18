"""A3.2 increment 5: /fs-goal grilling, /fs-research over measured_or_unmeasured, /fs-handoff by-path + redaction.
Real skill code, no mocks."""

from __future__ import annotations

import json
from pathlib import Path

from foundationskills.core.artifacts import Artifact, write_artifact
from foundationskills.core.contract import SkillContext
from foundationskills.core.provenance import make_provenance
from foundationskills.core.status import Status
from foundationskills.skills.fs_goal.skill import FsGoalSkill
from foundationskills.skills.fs_handoff.skill import FsHandoffSkill
from foundationskills.skills.fs_research.skill import FsResearchSkill

COMPLETE = {
    "objective": "a Traditional-Chinese customer-support assistant", "capabilities": ["chat"], "goal": "general_chat",
    "domain": "support", "preserve_general": True,
    "base_model": {"name_or_path": "gemma-4-E4B-it", "family": "gemma4", "size_b": 8.0, "arch": "dense"},
    "data_facts": {"has_verifiable_answers": False, "has_pairs": False, "has_instructions": True},
    "rl_algorithm": "none",
    "sources": [{"uri": "/data/raw/support_chats", "kind": "local_dir", "approx_examples": 2000}],
    "hardware": {"gpu": "NVIDIA GB200", "gpus_per_node": 4, "nodes": 1},
}


def test_fs_goal_asks_until_the_frontier_is_empty_then_needs_confirmation(tmp_path) -> None:
    ctx = SkillContext(workdir=tmp_path)
    partial = FsGoalSkill().execute({"answers": {"objective": "a support assistant"}}, ctx)
    assert partial.status is Status.UNMEASURED and "FG-HO-001" in [f.rule_id for f in partial.findings]
    assert partial.payload["questions"] and all(q["question"] for q in partial.payload["questions"])
    unconfirmed = FsGoalSkill().execute({"answers": COMPLETE}, ctx)
    assert unconfirmed.status is Status.UNMEASURED and "FG-HO-002" in [f.rule_id for f in unconfirmed.findings]
    assert not any(ref.type == "goal_spec" for ref in unconfirmed.artifacts)  # nothing written before the user confirms


def test_fs_goal_writes_a_goal_spec_only_when_confirmed(tmp_path) -> None:
    result = FsGoalSkill().execute({"answers": COMPLETE, "confirmed": True}, SkillContext(workdir=tmp_path))
    assert result.status is Status.PASS, [f.message for f in result.findings]
    spec = [ref for ref in result.artifacts if ref.type == "goal_spec"]
    assert spec and json.loads(Path(spec[0].path).read_text())["payload"]["base_model"]["name_or_path"] == "gemma-4-E4B-it"


def test_fs_goal_without_answers_is_refused(tmp_path) -> None:
    result = FsGoalSkill().execute({}, SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and "FG-IN-001" in [f.rule_id for f in result.findings]


def test_fs_research_never_reports_an_unmeasured_claim_as_a_pass(tmp_path) -> None:
    claims = [{"id": "c1", "claim": "loss went down", "asserted_status": "PASS", "measured": False}]
    result = FsResearchSkill().execute({"claims": claims}, SkillContext(workdir=tmp_path))
    # exactly UNMEASURED: a crash is RED too, which is how a looser assertion hid a real defect (measured on GPU)
    assert result.status is Status.UNMEASURED, (result.status, [f.message for f in result.findings])
    assert "FR-HO-001" in [f.rule_id for f in result.findings]
    assert all(not f.rule_id.startswith("MU-") for f in result.findings)  # inner rules travel as evidence only


def test_fs_research_without_claims_is_refused(tmp_path) -> None:
    result = FsResearchSkill().execute({"claims": []}, SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and "FR-IN-001" in [f.rule_id for f in result.findings]


def _artifact(tmp_path: Path) -> str:
    ref = write_artifact(Artifact("probe_report", "p1", {"plan_draft": "x", "checks": [{"check": "a", "missing": None}],
                                                         "capabilities": {"available": True}, "verdict": "PASS"},
                                  make_provenance("test", "0", {}), 1), tmp_path / "artifacts")
    return ref.path


def test_fs_handoff_references_by_path_and_redacts(tmp_path) -> None:
    path = _artifact(tmp_path)
    notes = "contact rick@example.com; OPENAI_API_KEY=sk-ABCDEFGHIJKLMNOP; token ghp_ABCDEFGHIJKL"
    result = FsHandoffSkill().execute({"artifacts": [path], "notes": notes, "next_steps": ["run /fs-plan"]},
                                      SkillContext(workdir=tmp_path))
    assert result.status is Status.PASS, [f.message for f in result.findings]
    payload = result.payload
    assert payload["artifacts"][0]["path"] == path and payload["artifacts"][0]["type"] == "probe_report"
    for secret in ("rick@example.com", "sk-ABCDEFGHIJKLMNOP", "ghp_ABCDEFGHIJKL"):
        assert secret not in json.dumps(payload)
    assert sum(payload["redactions"].values()) >= 3


def test_fs_handoff_refuses_a_path_that_is_not_an_artifact(tmp_path) -> None:
    bad = tmp_path / "plain.json"; bad.write_text("{}")
    result = FsHandoffSkill().execute({"artifacts": [str(bad)]}, SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and "FH-IN-002" in [f.rule_id for f in result.findings]


def test_fs_research_with_an_evidenced_claim_and_an_unmeasured_one(tmp_path) -> None:
    # the GPU journey's shape: one claim backed by a real eval report path, one with no measurement
    report = tmp_path / "arc_easy.json"; report.write_text("{}")
    claims = [{"id": "c1", "claim": "SFT raised arc_easy", "asserted_status": "PASS", "evidence": str(report), "measured": True},
              {"id": "c2", "claim": "throughput is 10k tok/s", "asserted_status": "PASS", "measured": False}]
    result = FsResearchSkill().execute({"claims": claims}, SkillContext(workdir=tmp_path))
    assert result.status is Status.UNMEASURED, [f.message for f in result.findings]
    assert result.refusal is None and any(ref.type == "research_findings" for ref in result.artifacts)
