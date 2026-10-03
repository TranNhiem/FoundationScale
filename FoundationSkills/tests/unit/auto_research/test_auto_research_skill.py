"""End-to-end AutoResearchSkill tests, must-fire fixtures, and SKILL.md coverage."""
from __future__ import annotations

from importlib import resources

import pytest

from foundationskills.core.artifacts import read_artifact
from foundationskills.core.contract import SkillContext, SkillResult
from foundationskills.core.status import Status
from foundationskills.skills import register_builtin_skills
from foundationskills.skills.auto_research.campaign import campaign_hash, launch_token
from foundationskills.skills.auto_research.ledger import Ledger, ledger_files
from foundationskills.skills.auto_research.skill import (
    AutoResearchSkill,
    _launch,
    _result,
    _spec,
    _val,
    result_problems,
)


def _ctx(path):
    return SkillContext(workdir=path)


def _write_files(tmp_path, events, tamper=False):
    for rel, text in ledger_files(events, tamper=tamper).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path / "ledger")


def _stage(tmp_path, spec, rows, tamper=False):
    """Materialise a fresh ledger for ``spec``: the approval envelope plus the pre-recorded results."""
    campaign = str(spec.get("id") or "unnamed-campaign")
    events = [("campaign_approved", campaign, "-",
               {"spec_hash": campaign_hash(spec), "approver": "reviewer"})]
    events += [("trial_result", campaign, str(r.get("trial") or "-"), dict(r)) for r in rows]
    return _write_files(tmp_path, events, tamper=tamper)


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


def _request(tmp_path, **overrides):
    request = {
        "campaign_spec": _spec(),
        "campaign_confirm": campaign_hash(_spec()),
        "approver": "reviewer",
        "ledger_dir": str(tmp_path / "ledger"),
    }
    request.update(overrides)
    return request


def _baseline_rows():
    return [_result("baseline", "baseline", seed, _val(v)) for seed, v in ((101, 0.5), (102, 0.502), (103, 0.501))]


class TestActionsEndToEnd:
    def test_check_writes_nothing(self, tmp_path):
        skill = AutoResearchSkill()
        result = skill.execute({"action": "check", "campaign_spec": _spec()}, _ctx(tmp_path))
        assert result.status is Status.PASS
        assert result.payload["spec_hash"] == campaign_hash(_spec())
        assert "max_runs" in result.payload["budget"]
        assert not (tmp_path / "ledger").exists()

    def test_launch_authorises_and_derives_token(self, tmp_path):
        skill = AutoResearchSkill()
        confirm = campaign_hash(_spec())
        result = skill.execute(_request(tmp_path, action="launch", launch_spec=_launch()), _ctx(tmp_path))
        assert result.status is Status.PASS
        assert result.payload["launch_token"] == launch_token(confirm, _launch())
        assert result.payload["budget_left"] == pytest.approx(20.0)
        ledger = Ledger(tmp_path / "ledger")
        assert ledger.verify() == [] and len(ledger.launches("ar-fixture")) == 1

    def test_record_stores_an_immutable_result(self, tmp_path):
        skill = AutoResearchSkill()
        result = skill.execute(
            _request(tmp_path, action="record", result=_result("t1", "candidate", 101, _val(0.5))),
            _ctx(tmp_path),
        )
        assert result.status is Status.PASS
        approved = [e for e in Ledger(tmp_path / "ledger").entries() if e["op"] == "campaign_approved"]
        assert len(approved) == 1 and approved[0]["seq"] == 0

    def test_propose_returns_cards_without_writes(self, tmp_path):
        skill = AutoResearchSkill()
        _stage(tmp_path, _spec(), _baseline_rows())
        request = _request(tmp_path, action="propose", current={"optim.lr": 1e-3}, symptoms=["loss diverging"])
        result = skill.execute(request, _ctx(tmp_path))
        assert result.status is Status.PASS
        assert result.payload["cards"] and result.payload["cards"][0]["idea"] == "lr_down"
        ops = [e["op"] for e in Ledger(tmp_path / "ledger").entries()]
        assert ops == ["campaign_approved"] + ["trial_result"] * 3  # propose never writes

    def test_close_reports_improved_with_a_report_artifact(self, tmp_path):
        skill = AutoResearchSkill()
        rows = _baseline_rows()
        rows += [_result("t1", "candidate", seed, _val(v)) for seed, v in ((101, 0.9), (102, 0.902), (103, 0.901))]
        _stage(tmp_path, _spec(), rows)
        result = skill.execute(
            _request(tmp_path, action="close", stop_reason="budget exhausted"), _ctx(tmp_path)
        )
        assert result.status is Status.PASS
        assert result.payload["outcome"] == "improved"
        assert result.payload["decisions"]["t1"]["verdict"] == "accepted_gain"
        assert result.payload["ledger"]["verified"] is True and result.payload["tsv"].startswith("seq\t")
        sealed = Ledger(tmp_path / "ledger")  # the report describes the sealed chain (campaign_closed counted)
        assert result.payload["ledger"]["count"] == 8
        assert result.payload["ledger"]["head_hash"] == sealed.head()["head_hash"]
        artifact = read_artifact(result.artifacts[0].path)
        assert artifact.payload["outcome"] == "improved" and "recommendation" in artifact.payload

    def test_launch_budget_counts_only_this_campaign(self, tmp_path):
        skill = AutoResearchSkill()
        spec = _spec()
        events = [
            ("campaign_approved", "ar-fixture", "-", {"spec_hash": campaign_hash(spec), "approver": "reviewer"}),
            ("launch_authorised", "other-campaign", "tX",
             {"launch_spec": _launch(trial="tX"), "spec_hash": "0" * 64,
              "launch_token": "1" * 64, "gpu_hours_est": 22.0}),
        ]
        _write_files(tmp_path, events)
        result = skill.execute(_request(tmp_path, action="launch", launch_spec=_launch()), _ctx(tmp_path))
        assert result.status is Status.PASS
        assert result.payload["budget_left"] == pytest.approx(20.0)  # campaign B's 22h are not ours


class TestLaunchValidationThroughExecute:
    @pytest.mark.parametrize("bad_hours", [-5.0, 0, "abc", True, float("nan")])
    def test_gpu_hours_est_shape_refuses_without_exception(self, tmp_path, bad_hours):
        skill = AutoResearchSkill()
        request = _request(tmp_path, action="launch", launch_spec=_launch(gpu_hours_est=bad_hours))
        result = skill.execute(request, _ctx(tmp_path))
        fired = [f.rule_id for f in result.findings]
        assert result.status is Status.REFUSED
        assert "AR-LN-001" in fired and "CORE-EXC" not in fired

    def test_role_shape_refuses_without_exception(self, tmp_path):
        skill = AutoResearchSkill()
        result = skill.execute(
            _request(tmp_path, action="launch", launch_spec=_launch(role="boss")), _ctx(tmp_path)
        )
        fired = [f.rule_id for f in result.findings]
        assert result.status is Status.REFUSED
        assert "AR-LN-001" in fired and "CORE-EXC" not in fired


class TestRecordShape:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True, "0.5"])
    def test_metric_values_must_be_finite(self, tmp_path, bad):
        skill = AutoResearchSkill()
        metrics = {"val_accuracy": {"value": bad, "se": 0.0}}
        request = _request(tmp_path, action="record", result=_result("t1", "candidate", 101, metrics))
        result = skill.execute(request, _ctx(tmp_path))
        fired = [f.rule_id for f in result.findings]
        assert result.status is Status.REFUSED
        assert "AR-IN-006" in fired and "CORE-EXC" not in fired

    def test_metric_se_must_be_finite_or_null(self, tmp_path):
        skill = AutoResearchSkill()
        metrics = {"val_accuracy": {"value": 0.5, "se": float("nan")}}
        request = _request(tmp_path, action="record", result=_result("t1", "candidate", 101, metrics))
        result = skill.execute(request, _ctx(tmp_path))
        assert result.status is Status.REFUSED and "AR-IN-006" in [f.rule_id for f in result.findings]

    def test_se_may_be_null(self):
        row = _result("t1", "candidate", 1, {"val_accuracy": {"value": 0.5, "se": None}})
        assert result_problems(row) == []


class TestApprovalEnvelope:
    def test_ledger_with_a_different_spec_hash_refuses(self, tmp_path):
        skill = AutoResearchSkill()
        _write_files(tmp_path, [("campaign_approved", "ar-fixture", "-",
                                {"spec_hash": "0" * 64, "approver": "other"})])
        request = _request(tmp_path, action="propose", current={"optim.lr": 1e-3}, symptoms=[])
        result = skill.execute(request, _ctx(tmp_path))
        assert result.status is Status.REFUSED
        assert "AR-AP-001" in [f.rule_id for f in result.findings]
        assert Ledger(tmp_path / "ledger").head()["count"] == 1  # refused before any write

    def test_empty_approver_opens_no_envelope(self, tmp_path):
        skill = AutoResearchSkill()
        request = _request(tmp_path, action="propose", approver="", current={}, symptoms=[])
        result = skill.execute(request, _ctx(tmp_path))
        assert result.status is Status.REFUSED
        assert "AR-AP-001" in [f.rule_id for f in result.findings]
        assert not (tmp_path / "ledger").exists()


class TestInertRequestKnobs:
    def test_replay_and_tamper_keys_record_nothing_extra(self, tmp_path):
        skill = AutoResearchSkill()
        row = _result("t1", "candidate", 101, _val(0.5))
        skill.execute(
            _request(tmp_path / "plain", action="record", result=dict(row)), _ctx(tmp_path / "plain")
        )
        skill.execute(
            _request(tmp_path / "extra", action="record", result=dict(row),
                     replay=[_result("zz", "candidate", 999, _val(0.9))], tamper=True),
            _ctx(tmp_path / "extra"),
        )
        plain, extra = Ledger(tmp_path / "plain" / "ledger"), Ledger(tmp_path / "extra" / "ledger")
        assert plain.head()["count"] == extra.head()["count"] == 2  # envelope + one result, nothing more
        assert [e["op"] for e in plain.entries()] == [e["op"] for e in extra.entries()]
        assert {p["trial"] for p in plain.results("ar-fixture")} == {"t1"}
        assert {p["trial"] for p in extra.results("ar-fixture")} == {"t1"}
        assert extra.verify() == []  # the ignored keys tamper nothing

    def test_fixtures_carry_pure_data_only(self):
        for rule_id, fixture in AutoResearchSkill().must_fire_fixtures().items():
            assert "replay" not in fixture["request"], rule_id
            assert "tamper" not in fixture["request"], rule_id


class TestDiagnose:
    def _diagnose(self, failure):
        return AutoResearchSkill().diagnose(failure)

    def test_missing_ledger_object_maps_to_the_objects_dir(self):
        diagnosis = self._diagnose(FileNotFoundError("objects/ab.json"))
        assert diagnosis.symptom == "ledger object missing"
        assert all(diagnosis.likely_causes) and all(diagnosis.checks) and all(diagnosis.recovery)

    def test_non_canonical_payload_maps_to_floats(self):
        diagnosis = self._diagnose(ValueError("Out of range float values are not JSON compliant"))
        assert diagnosis.symptom == "non-canonical payload"
        assert all(diagnosis.likely_causes) and all(diagnosis.recovery)

    def test_key_error_names_the_missing_field(self):
        diagnosis = self._diagnose(KeyError("campaign_spec"))
        assert diagnosis.symptom and "campaign_spec" in diagnosis.symptom
        assert all(diagnosis.checks) and all(diagnosis.recovery)

    def test_runtime_error_falls_back_to_the_generic_branch(self):
        diagnosis = self._diagnose(RuntimeError("boom"))
        assert "boom" in diagnosis.symptom and "unexpected" in diagnosis.likely_causes[0]

    def test_branches_are_branch_specific(self):
        symptoms = [self._diagnose(f).symptom for f in (
            FileNotFoundError("objects/ab.json"),
            ValueError("Out of range float values are not JSON compliant"),
            KeyError("campaign_spec"),
            RuntimeError("boom"),
        )]
        assert all(s.strip() for s in symptoms)
        assert len(set(symptoms)) == len(symptoms)

    def test_skill_result_with_ar_ho_003_maps_to_the_ledger_branch(self):
        skill = AutoResearchSkill()
        finding = skill.finding("AR-HO-003", "ledger chain verification failed: seq 2: payload hash mismatch")
        diagnosis = skill.diagnose(SkillResult(Status.RED, {}, findings=(finding,)))
        assert diagnosis.symptom == "ledger chain verification failed"
        assert all(diagnosis.likely_causes) and all(diagnosis.checks) and all(diagnosis.recovery)


class TestMustFireFixtures:
    def test_fixtures_cover_every_declared_rule(self):
        skill = AutoResearchSkill()
        assert set(skill.must_fire_fixtures()) == {r.rule_id for r in skill.rules}

    @pytest.mark.parametrize("rule_id", [r.rule_id for r in AutoResearchSkill.rules])
    def test_each_fixture_fires_its_rule(self, tmp_path, rule_id):
        skill = AutoResearchSkill()
        fixture = skill.must_fire_fixtures()[rule_id]
        request = _materialize(fixture, tmp_path)
        result = skill.execute(request, _ctx(tmp_path))
        fired = [f.rule_id for f in result.findings]
        assert rule_id in fired, f"{rule_id} did not fire (fired: {fired}, status: {result.status})"

    def test_input_rule_fixtures_refuse(self, tmp_path):
        skill = AutoResearchSkill()
        for rule in skill.rules:
            if rule.phase != "input":
                continue
            request = _materialize(skill.must_fire_fixtures()[rule.rule_id], tmp_path)
            result = skill.execute(request, _ctx(tmp_path))
            assert result.status is Status.REFUSED, rule.rule_id

    def test_handoff_outcomes_match_the_semantics(self, tmp_path):
        skill = AutoResearchSkill()
        expected = {"AR-HO-001": Status.RED, "AR-HO-002": Status.RED, "AR-HO-003": Status.RED,
                    "AR-HO-004": Status.UNMEASURED, "AR-HO-005": Status.PASS}
        fixtures = skill.must_fire_fixtures()
        for rule_id, status in expected.items():
            request = _materialize(fixtures[rule_id], tmp_path / rule_id.replace("-", "_"))
            tmp_path.joinpath("x").mkdir(exist_ok=True)
            result = skill.execute(request, SkillContext(workdir=tmp_path / "x"))
            assert result.status is status, f"{rule_id} -> {result.status}"


class TestSkillRegistration:
    def test_register_builtin_skills_idempotent(self):
        from foundationskills.core.registry import SkillRegistry

        registry = SkillRegistry()
        registered = register_builtin_skills(registry)
        assert "data_engine" in registered and "auto_research" in registered
        again = register_builtin_skills(registry)
        assert "auto_research" not in again and "auto_research" in registry.names()


class TestSkillDoc:
    def test_skill_md_headings(self):
        doc = (resources.files("foundationskills.skills.auto_research")
               .joinpath("SKILL.md").read_text(encoding="utf-8"))
        for heading in ("## Purpose", "## When to use", "## Inputs", "## Outputs", "## Scope",
                        "## The loop", "## The approval model", "## Acceptance statistics",
                        "## Validation rules", "## Failure handling", "## FS interface",
                        "## Increments", "## Worked examples"):
            assert heading in doc, heading

    def test_skill_md_lists_every_rule_id(self):
        doc = (resources.files("foundationskills.skills.auto_research")
               .joinpath("SKILL.md").read_text(encoding="utf-8"))
        for rule in AutoResearchSkill.rules:
            assert rule.rule_id in doc, rule.rule_id
