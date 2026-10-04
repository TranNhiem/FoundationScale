"""Duplicate-delta suppression over launch payloads and the broken-chain close recommendation."""
from __future__ import annotations

from foundationskills.core.artifacts import read_artifact
from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills.auto_research.campaign import campaign_hash
from foundationskills.skills.auto_research.propose import propose
from foundationskills.skills.auto_research.skill import AutoResearchSkill, _result, _spec, _val

RESTORE_LEDGER = "restore an intact ledger: chain verification failed; no verdict is trustworthy until it verifies"


def _ctx(path):
    return SkillContext(workdir=path)


def _materialize(fixture: dict, tmp_path):
    for rel, content in fixture.get("files", {}).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def substitute(value, confirm):
        if isinstance(value, str):
            return value.replace("{tmp}", str(tmp_path)).replace("{confirm}", confirm)
        if isinstance(value, list):
            return [substitute(v, confirm) for v in value]
        if isinstance(value, dict):
            return {k: substitute(v, confirm) for k, v in value.items()}
        return value

    request = fixture["request"]
    spec = substitute(request.get("campaign_spec"), "")
    confirm = campaign_hash(spec) if isinstance(spec, dict) else ""
    return substitute(request, confirm)


def _baseline_rows():
    return [_result("baseline", "baseline", seed, _val(v)) for seed, v in ((101, 0.5), (102, 0.502), (103, 0.501))]


class TestProposeSeesLaunchDeltas:
    def test_launched_delta_is_dropped_as_duplicate_delta(self):
        spec, results = _spec(), _baseline_rows()
        current, symptoms = {"optim.lr": 1e-3}, ["loss diverging"]
        before = propose(spec, results, current, symptoms)
        target = next(card["delta"] for card in before if card["delta"])
        after = propose(spec, results, current, symptoms, launches=[{"trial": "t1", "delta": dict(target)}])
        assert target in [card["delta"] for card in before]
        assert target not in [card["delta"] for card in after]  # the launch already covers this delta
        assert after  # the remaining catalog ideas are untouched

    def test_unrelated_launch_delta_keeps_the_idea_live(self):
        spec, results = _spec(), _baseline_rows()
        current, symptoms = {"optim.lr": 1e-3}, ["loss diverging"]
        before = propose(spec, results, current, symptoms)
        target = next(card["delta"] for card in before if card["delta"])
        after = propose(spec, results, current, symptoms, launches=[{"trial": "tX", "delta": {"optim.lr": 42.0}}])
        assert target in [card["delta"] for card in after]


class TestBrokenChainRecommendation:
    def test_broken_chain_recommends_restoring_the_ledger(self, tmp_path):
        skill = AutoResearchSkill()
        request = _materialize(skill.must_fire_fixtures()["AR-HO-003"], tmp_path)
        result = skill.execute(request, _ctx(tmp_path))
        assert result.status is Status.RED
        assert result.payload["outcome"] == "unmeasured"
        assert result.payload["recommendation"] == RESTORE_LEDGER
        assert read_artifact(result.artifacts[0].path).payload["recommendation"] == RESTORE_LEDGER

    def test_missing_evidence_still_recommends_collecting_more(self, tmp_path):
        skill = AutoResearchSkill()
        request = _materialize(skill.must_fire_fixtures()["AR-HO-004"], tmp_path)
        result = skill.execute(request, _ctx(tmp_path))
        assert result.status is Status.UNMEASURED
        assert result.payload["recommendation"] == (
            "collect more evidence: the noise floor or the paired repeats are missing"
        )
