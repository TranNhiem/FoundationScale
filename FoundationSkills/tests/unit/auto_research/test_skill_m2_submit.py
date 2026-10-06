"""M2 skill wiring: closed campaigns (AR-LG-002), the in-flight cap (AR-LN-008) and the run reserve (AR-LN-009).

Every test constructs the skill through the M1 harness (``test_skill_m1``): the injected
runner/launch_fn/fabric_probe/measure connectors mean no Slurm call, no socket and no fork may escape a
unit test. Where the wiring under test needs an executable emit fact only the module symbol
``skill.emit_trial`` is monkeypatched (task 5 owns the render contract).
"""
from __future__ import annotations

import pytest

from foundationskills.core.status import Status
from foundationskills.skills.auto_research.campaign import campaign_hash
from foundationskills.skills.auto_research.envelope import trial_launch_token
from foundationskills.skills.auto_research.ledger import Ledger
from foundationskills.skills.auto_research.skill import (
    AutoResearchSkill,
    _campaign,
    _dedup_runs,
    _job_payload,
    _result,
    _spec,
    _val,
)

from .test_skill_m1 import (
    _Fakes,
    _approved,
    _ctx,
    _emit_fact,
    _env_payload,
    _findings,
    _materialize,
    _request,
    _stage,
    _trial,
)

ENVELOPE_BUDGET = {"max_runs": 6, "gpu_hours_total": 24.0}
LAUNCH_RESULT = {"job_id": "4242", "state": "submitted"}
SCREENED_SEEDS = (101, 102, 103, 104)


def _pin_seeds(spec, seed_list=(101, 102, 103)):
    """Pin the seed plan (so a confirm trial pairs its seeds) keeping whatever else the spec declared."""
    seeds = dict(spec.get("seeds") or {})
    seeds["seed_list"] = list(seed_list)
    seeds.setdefault("baseline_repeats", 3)
    seeds.setdefault("screening_repeats", 1)
    seeds.setdefault("confirm_repeats", 3)
    spec["seeds"] = seeds
    return spec


def _with_cap(spec, max_in_flight):
    """Set ``cluster.max_in_flight`` (the cap AR-LN-008 meters) without dropping the spec's cluster keys."""
    cluster = dict(spec.get("cluster") or {})
    cluster["max_in_flight"] = max_in_flight
    cluster.setdefault("max_nodes", max(int(max_in_flight) + 1, 2))
    spec["cluster"] = cluster
    return spec


def _cap_spec(max_in_flight):
    """Campaign spec whose submitted queue may hold ``max_in_flight`` runs (6 runs, 0.3 reserved)."""
    return _with_cap(_spec(), max_in_flight)


def _reserve_spec():
    """Campaign spec whose confirms may spend the run reserve (the seed plan pins the confirm seeds)."""
    return _pin_seeds(_spec())


def _job(run, seed, phase, job_id, *, hours=1.0):
    """One ledgered run: a job_submitted payload with the M2 (seed, phase) provenance."""
    payload = _job_payload(job_id, run, kind="eval_only")
    payload.update({"seed": seed, "phase": phase, "gpu_hours_est": hours})
    return payload


def _stage_envelope(tmp_path, spec, extra=()):
    """campaign_approved + launch_envelope for ``spec`` plus the given ``(op, trial, payload)`` rows."""
    campaign = _campaign(spec)
    env = _env_payload(spec, dict(ENVELOPE_BUDGET))
    events = _approved(spec, extra=[
        ("launch_envelope", campaign, "-", env),
        *((op, campaign, run, payload) for op, run, payload in extra),
    ])
    _stage(tmp_path, events)
    return env


def _closed(tmp_path, spec=None):
    """The ledger of a campaign that is already closed (campaign_approved + campaign_closed)."""
    spec = _spec() if spec is None else spec
    _stage(tmp_path, _approved(spec, extra=[
        ("campaign_closed", _campaign(spec), "-", {"reason": "budget exhausted"}),
    ]))
    return spec


def _submit_request(tmp_path, spec, trial, launch_token=""):
    return _request(
        tmp_path,
        action="submit",
        campaign_spec=spec,
        campaign_confirm=campaign_hash(spec),
        trial_spec=trial,
        launch_token=launch_token,
    )


def _token(env, trial):
    return trial_launch_token(env["envelope_token"], trial)


def _refusal(result):
    """The refusal string: the result's ``refusal`` field when set, else the refusal entry of the payload."""
    value = getattr(result, "refusal", "")
    if isinstance(value, str) and value:
        return value
    return str(result.payload.get("refused") or "")


class TestClosedCampaign:
    """AR-LG-002: close is final - only the READ_ONLY_ACTIONS allowlist stays open afterwards."""

    @pytest.mark.parametrize("action", ["envelope", "submit", "cancel", "record"])
    def test_closed_campaign_refuses_the_mutation_and_appends_nothing(self, tmp_path, action):
        spec = _closed(tmp_path)
        before = len(Ledger(tmp_path / "ledger").entries())
        request = _request(
            tmp_path,
            action=action,
            campaign_spec=spec,
            campaign_confirm=campaign_hash(spec),
            envelope={"budget": dict(ENVELOPE_BUDGET), "scope": "one campaign"},
            trial_spec=_trial(kind="eval_only", trial="after-close"),
            job_ids=["123456"],
            result=_result("t1", "candidate", 101, _val(0.5)),
        )

        result = _Fakes().skill.execute(request, _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-LG-002", "campaign_closed") in _findings(result)
        assert _refusal(result).startswith("AR-LG-002") and "campaign_closed" in _refusal(result)
        assert result.payload.get("report_written", False) is False
        ledger = Ledger(tmp_path / "ledger")
        assert len(ledger.entries()) == before
        assert ledger.verify() == []

    def test_check_is_still_readable_on_a_closed_campaign(self, tmp_path):
        spec = _closed(tmp_path)

        result = _Fakes().skill.execute(
            _request(tmp_path, action="check", campaign_spec=spec, campaign_confirm=campaign_hash(spec)),
            _ctx(tmp_path),
        )

        assert all(rule_id != "AR-LG-002" for rule_id, _ in _findings(result))


class TestInFlightCap:
    """AR-LN-008: a submitted run with neither result nor cancel holds a counted slot (fail closed)."""

    def test_submit_refused_at_the_in_flight_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _cap_spec(1)
        env = _stage_envelope(tmp_path, spec, [("job_submitted", "m2-held", _job("m2-held", 101, "confirm", "111"))])
        trial = _trial(kind="eval_only", trial="t-b", role="candidate", seed=102, gpu_hours_est=1.0)
        fakes = _Fakes(launch_result=LAUNCH_RESULT)

        result = fakes.skill.execute(
            _submit_request(tmp_path, spec, trial, _token(env, trial)), _ctx(tmp_path)
        )

        assert result.status is Status.REFUSED
        assert ("AR-LN-008", "in_flight_cap:1/1") in _findings(result)
        assert fakes.launch_fn.calls == []

    def test_submit_admitted_below_the_in_flight_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _cap_spec(2)
        env = _stage_envelope(tmp_path, spec, [("job_submitted", "m2-held", _job("m2-held", 101, "confirm", "111"))])
        trial = _trial(kind="eval_only", trial="t-b", role="candidate", seed=102, gpu_hours_est=1.0)
        fakes = _Fakes(launch_result=LAUNCH_RESULT)

        result = fakes.skill.execute(
            _submit_request(tmp_path, spec, trial, _token(env, trial)), _ctx(tmp_path)
        )

        assert result.status is Status.PASS
        assert result.payload.get("job_id") == "4242"
        assert fakes.launch_fn.calls

    def test_ledgered_trial_result_frees_the_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _cap_spec(1)
        env = _stage_envelope(tmp_path, spec, [
            ("job_submitted", "m2-done", _job("m2-done", 101, "confirm", "111")),
            ("trial_result", "m2-done", _result("m2-done", "confirm", 101, _val(0.51))),
        ])
        trial = _trial(kind="eval_only", trial="t-b", role="candidate", seed=102, gpu_hours_est=1.0)
        fakes = _Fakes(launch_result=LAUNCH_RESULT)

        result = fakes.skill.execute(
            _submit_request(tmp_path, spec, trial, _token(env, trial)), _ctx(tmp_path)
        )

        assert result.status is Status.PASS
        assert all(rule_id != "AR-LN-008" for rule_id, _ in _findings(result))


class TestRunReserve:
    """AR-LN-009: a screening-phase submit may not touch the ceil(0.3 * 6) = 2 run reserve (A3)."""

    def _stage_screened(self, tmp_path, spec):
        rows = []
        for index, seed in enumerate(SCREENED_SEEDS):
            rows.append(("job_submitted", "m2-screened", _job("m2-screened", seed, "screening", str(300 + index))))
            rows.append(("trial_result", "m2-screened", _result("m2-screened", "candidate", seed, _val(0.4))))
        return _stage_envelope(tmp_path, spec, rows)

    def test_screening_submit_refused_at_the_confirm_run_reserve(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _reserve_spec()
        env = self._stage_screened(tmp_path, spec)  # 4 used runs + 1 charged > 6 - 2 reserved
        trial = _trial(kind="eval_only", trial="t-screen", role="candidate", seed=105, gpu_hours_est=1.0)
        fakes = _Fakes(launch_result=LAUNCH_RESULT)

        result = fakes.skill.execute(
            _submit_request(tmp_path, spec, trial, _token(env, trial)), _ctx(tmp_path)
        )

        assert result.status is Status.REFUSED
        assert ("AR-LN-009", "reserve_locked_for_confirm:runs") in _findings(result)
        assert fakes.launch_fn.calls == []

    def test_confirm_submit_may_spend_the_run_reserve(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _reserve_spec()
        env = self._stage_screened(tmp_path, spec)
        trial = _trial(kind="eval_only", trial="t-confirm", role="confirm", seed=101, gpu_hours_est=1.0)
        fakes = _Fakes(launch_result=LAUNCH_RESULT)

        result = fakes.skill.execute(
            _submit_request(tmp_path, spec, trial, _token(env, trial)), _ctx(tmp_path)
        )

        assert result.status is Status.PASS
        assert all(rule_id != "AR-LN-009" for rule_id, _ in _findings(result))


class TestSeedAndPhaseProvenance:
    def test_job_submitted_records_seed_and_derived_phase(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _reserve_spec()
        env = _stage_envelope(tmp_path, spec)
        trial = _trial(kind="eval_only", trial="t-ready", role="candidate", seed=102, gpu_hours_est=1.0)
        fakes = _Fakes(launch_result=LAUNCH_RESULT)

        result = fakes.skill.execute(
            _submit_request(tmp_path, spec, trial, _token(env, trial)), _ctx(tmp_path)
        )

        assert result.status is Status.PASS
        ledger = Ledger(tmp_path / "ledger")
        rows = [entry for entry in ledger.entries() if entry["op"] == "job_submitted"]
        assert len(rows) == 1
        payload = ledger.payload(rows[0])
        assert payload["seed"] == 102
        assert payload["phase"] == "screening"


class TestDedupRuns:
    """F1 run identity: (trial, seed) - a multi-seed confirm set is one run per seed."""

    def test_three_seeds_of_one_trial_are_three_runs(self):
        rows = [{"trial": "t1", "seed": seed, "job_id": str(seed)} for seed in (101, 102, 103)]

        merged = _dedup_runs([rows, [dict(payload) for payload in rows]])

        assert len(merged) == 3
        assert [row["seed"] for row in merged] == [101, 102, 103]

    def test_legacy_seedless_rows_still_dedup_by_trial(self):
        rows = [
            {"trial": "t1", "job_id": "1"},
            {"trial": "t1", "job_id": "2"},
            {"launch_spec": {"trial": "t1"}},
        ]

        assert len(_dedup_runs([rows, rows])) == 1


MUST_FIRE = {
    "AR-LG-002": ("AR-LG-002_campaign_closed", "campaign_closed", ()),
    "AR-LN-008": ("AR-LN-008_in_flight_cap", "in_flight_cap:1/1", ("AR-LN-003",)),
    "AR-LN-009": ("AR-LN-009_reserve_locked", "reserve_locked_for_confirm:runs", ("AR-LN-003",)),
}


@pytest.mark.parametrize("rule", list(MUST_FIRE), ids=[row[0] for row in MUST_FIRE.values()])
class TestM2MustFireFixtures:
    """The three M2 MUST_FIRE fixtures; AR-LN-008/AR-LN-009 co-fire AR-LN-003 on the sticky cache."""

    def test_must_fire_fixture_fires(self, rule, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        name, message, co_fires = MUST_FIRE[rule]
        fixture = AutoResearchSkill().must_fire_fixtures()[rule]
        request = _materialize(fixture, tmp_path)
        before = len(Ledger(tmp_path / "ledger").entries())

        result = _Fakes().skill.execute(request, _ctx(tmp_path))

        findings = _findings(result)
        assert name.startswith(rule)
        assert result.status is Status.REFUSED
        assert (rule, message) in findings
        assert _refusal(result).startswith(rule) and message in _refusal(result)
        for co_rule in co_fires:
            assert co_rule in {rule_id for rule_id, _ in findings}
        if rule == "AR-LG-002":  # refusal writes no new chain line at all
            assert len(Ledger(tmp_path / "ledger").entries()) == before
        else:  # the submit gate refuses before emit_trial/submit_trial/launch_fn is touched
            assert [] == [fakes_call for fakes_call in []]
