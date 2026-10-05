"""Regression tests for the auto_research M1 review fixes F1..F6.

F1 budget burn without a job id (the envelope gate meters every run attempt, including an authorised
run that never reported a job id), F2 the fabric gate bypass (fixed cache path, clamped TTL and a fresh
probe immediately before submission), F3 duplicate job-id accounting, F4 emit-drop carrying, F5 zero
sacct measurements (a measurement of nothing is not a measurement) and F6 the AR-HO-006 baseline
provenance rule.

The construction mirrors ``test_skill_m1.py`` (``_Fakes``-style injection against a real
``AutoResearchSkill``): nothing in these tests touches the network or a subprocess. Where the wiring
needs an emit fact (or must not use the real submit path) only the module symbols ``skill.emit_trial``
and ``skill.submit_trial`` are monkeypatched.
"""
from __future__ import annotations

import json

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.interfaces.fs.fabric import DEFAULT_TTL_S
from foundationskills.interfaces.fs.launch import LaunchRefused
from foundationskills.skills.auto_research.accounting import campaign_usage
from foundationskills.skills.auto_research.campaign import campaign_hash
from foundationskills.skills.auto_research.envelope import envelope_token, trial_launch_token
from foundationskills.skills.auto_research.ledger import Ledger, ledger_files
from foundationskills.skills.auto_research.skill import (
    AutoResearchSkill,
    _campaign,
    _clamp_ttl_s,
    _result,
    _spec,
    _val,
)

CLOCK_S = 1750000000.0
READY_PROBE = {"state": "ready", "reason": "fabric_ready", "ts": CLOCK_S}


def _ctx(path):
    return SkillContext(workdir=path)


def _approved(spec, extra=()):
    campaign = _campaign(spec)
    events = [("campaign_approved", campaign, "-", {"spec_hash": campaign_hash(spec), "approver": "reviewer"})]
    events.extend(list(extra))
    return events


def _stage(tmp_path, events):
    """Materialise ``ledger_files`` archive bytes under ``tmp_path`` (same harness as test_skill_m1)."""
    for rel, text in ledger_files(events).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path / "ledger")


def _env_payload(spec, budget=None, approver="reviewer"):
    env = {"budget": dict(budget or {"max_runs": 6, "gpu_hours_total": 24.0}), "scope": "one campaign"}
    return {
        "envelope_token": envelope_token(campaign_hash(spec), env),
        "spec_hash": campaign_hash(spec),
        "envelope": env,
        "budget": dict(env["budget"]),
        "approver": approver,
    }


def _trial(kind="eval_only", **overrides):
    trial = {
        "trial": "t-sub", "role": "candidate", "kind": kind, "delta": {"optim.lr": 5e-4},
        "seed": 101, "nodes": 1, "gpus_per_node": 8, "partition": "rally", "time": "10-00:00:00",
        "commands": ["uv run train.py --config campaign.yaml"], "gpu_hours_est": 4.0,
        "eval_request": {"command": ["eval", "run", "gsm8k"]},
    }
    trial.update(overrides)
    return trial


def _auth(trial, gpu_hours_est=4.0):
    """A ledgered ``launch_authorised`` payload exactly as ``AutoResearchSkill._launch`` appends it."""
    return {
        "launch_spec": {"trial": trial, "gpu_hours_est": gpu_hours_est, "delta": {"optim.lr": 5e-4}},
        "spec_hash": "sha256:00",
        "launch_token": "fs-ar-trial-v1-" + str(trial),
        "gpu_hours_est": gpu_hours_est,
    }


def _job(job_id, trial, kind="eval_only", **overrides):
    payload = {
        "trial": trial, "job_id": job_id, "launch_token": "fs-ar-trial-v1-fixture", "kind": kind,
        "gpu_hours_est": 8.0, "budget_after": {"runs_left": 2, "hours_left": 16.0},
        "submitted_at": CLOCK_S,
    }
    payload.update(overrides)
    return payload


def _fabric(state, ts, reason):
    return json.dumps({"master:8081": {"state": state, "ts": ts, "reason": reason}}, sort_keys=True)


def _request(tmp_path, **overrides):
    request = {
        "campaign_spec": _spec(),
        "campaign_confirm": campaign_hash(_spec()),
        "approver": "reviewer",
        "ledger_dir": str(tmp_path / "ledger"),
    }
    request.update(overrides)
    return request


def _submit_request(tmp_path, payload, trial, **overrides):
    return _request(
        tmp_path, action="submit", trial_spec=trial,
        launch_token=trial_launch_token(payload["envelope_token"], trial), **overrides,
    )


def _baseline_rows():
    return [_result("baseline", "baseline", seed, _val(v)) for seed, v in ((101, 0.5), (102, 0.502), (103, 0.501))]


def _gain_rows():
    return [_result("t1", "candidate", seed, _val(v)) for seed, v in ((101, 0.9), (102, 0.902), (103, 0.901))]


class _Rec:
    """Injected connector: records ``(args, kwargs)`` and returns ``result`` (never forks or connects)."""

    def __init__(self, result=None):
        self.calls: list[tuple[tuple, dict]] = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _Measure(_Rec):
    """Injected ``measure``: measured GPU hours per job id (missing key = sacct-unavailable fallback)."""

    def __init__(self, measured=None):
        super().__init__(None)
        self.measured = dict(measured or {})

    def __call__(self, job_id):
        self.calls.append(((job_id,), {}))
        return self.measured.get(str(job_id))


class _Fakes:
    def __init__(self, *, runner_result=None, launch_result=None, probe_result=None, measured=None):
        self.runner = _Rec(runner_result)
        self.launch_fn = _Rec(launch_result)
        self.fabric_probe = _Rec(READY_PROBE if probe_result is None else probe_result)
        self.measure = _Measure(measured)
        self.skill = AutoResearchSkill(
            runner=self.runner, launch_fn=self.launch_fn, fabric_probe=self.fabric_probe,
            measure=self.measure, clock=lambda: CLOCK_S,
        )


def _emit_fact(confirm="sha256:c0ffee", executable=True, missing=(), drops=()):
    """An ``emit_trial`` result for the wiring under test (the render contract lives in test_emit_trial)."""

    def fake(request, **kwargs):
        return {
            "fs_launch_spec": {"sbatch": "#!/bin/bash\n#SBATCH --time=10-00:00:00\n", "executable": True},
            "trial_spec": dict(request["trial_spec"]),
            "confirm": confirm,
            "notes": [], "drops": list(drops), "executable": executable, "missing": list(missing),
        }

    return fake


def _measure_fn(table):
    """Pure ``measure`` table for the accounting unit tests (records every job id it is asked about)."""
    seen: list[str] = []

    def _fake(job_id):
        seen.append(str(job_id))
        return table.get(str(job_id))

    _fake.seen = seen
    return _fake


def _findings(result):
    return [(f.rule_id, f.message) for f in result.findings]


class TestF1BudgetBurnWithoutAJobId:
    """The envelope gate meters every run of the campaign - a job id is not what makes it burn (F1)."""

    def test_an_authorised_run_without_a_job_entry_still_burns_a_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _spec(budget={"gpu_hours_total": 24.0, "max_runs": 1, "per_run_timeout_h": 8.0, "reserve_frac": 0.3})
        payload = _env_payload(spec, {"max_runs": 1, "gpu_hours_total": 24.0})
        _stage(tmp_path, _approved(spec, extra=[
            ("launch_envelope", _campaign(spec), "-", payload),
            ("launch_authorised", _campaign(spec), "t-ghost", _auth("t-ghost", 4.0)),  # never reported a job id
        ]))
        fakes = _Fakes(launch_result={"job_id": "1"})
        result = fakes.skill.execute(
            _submit_request(tmp_path, payload, _trial(trial="t-sub"),
                            campaign_spec=spec, campaign_confirm=campaign_hash(spec)),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert ("AR-LN-006", "envelope_exhausted:runs") in _findings(result)
        # the synthetic {job_id: None} entry is declared, never measured
        assert fakes.measure.calls == [] and fakes.launch_fn.calls == []

    def test_an_authorised_run_without_a_job_entry_still_burns_its_declared_hours(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _spec(budget={"gpu_hours_total": 10.0, "max_runs": 6, "per_run_timeout_h": 8.0, "reserve_frac": 0.3})
        payload = _env_payload(spec, {"max_runs": 6, "gpu_hours_total": 10.0})
        _stage(tmp_path, _approved(spec, extra=[
            ("launch_envelope", _campaign(spec), "-", payload),
            ("launch_authorised", _campaign(spec), "t-ghost", _auth("t-ghost", 8.0)),
        ]))
        fakes = _Fakes(launch_result={"job_id": "1"})
        result = fakes.skill.execute(
            _submit_request(tmp_path, payload, _trial(trial="t-sub"),
                            campaign_spec=spec, campaign_confirm=campaign_hash(spec)),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert ("AR-LN-006", "envelope_exhausted:gpu_hours") in _findings(result)  # 10 - 8 declared - 4 est < 0
        assert fakes.measure.calls == [] and fakes.launch_fn.calls == []

    def test_one_submitted_run_burns_exactly_one_run_and_one_job_row(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _spec()
        payload = _env_payload(spec)
        _stage(tmp_path, _approved(spec, extra=[("launch_envelope", _campaign(spec), "-", payload)]))
        fakes = _Fakes(launch_result={"job_id": "1"}, measured={"1": 3.0})
        after = []
        for name in ("t-a", "t-b"):
            result = fakes.skill.execute(
                _submit_request(tmp_path, payload, _trial(trial=name)), _ctx(tmp_path)
            )
            assert result.status is Status.PASS
            after.append(result.payload["budget_after"])
        # the launch_authorised + job_submitted pair of one submission is ONE run (never two)
        assert after[0]["runs_left"] == 5 and after[0]["hours_left"] == pytest.approx(20.0)
        assert after[1]["runs_left"] == 4 and after[1]["hours_left"] == pytest.approx(17.0)
        # t-a has a job entry: measured once and no synthetic row is invented for it
        assert fakes.measure.calls and all(c == (("1",), {}) for c in fakes.measure.calls)  # real ids only, never synthetic


class TestF2FabricGate:
    """Fixed cache path, clamped TTL and a fresh probe right before submit_trial (F2)."""

    def test_the_request_cannot_point_the_fabric_cache_elsewhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _spec()
        payload = _env_payload(spec)
        _stage(tmp_path, _approved(spec, extra=[("launch_envelope", _campaign(spec), "-", payload)]))
        (tmp_path / "ledger" / "fabric.json").write_text(_fabric("ready", CLOCK_S, "fabric_ready"), encoding="utf-8")
        override = tmp_path / "elsewhere.json"
        override.write_text(_fabric("refused", 4102444800.0, "fabric_refused"), encoding="utf-8")
        fakes = _Fakes(launch_result={"job_id": "4242"})
        result = fakes.skill.execute(
            _submit_request(tmp_path, payload, _trial(), fabric_cache=str(override)), _ctx(tmp_path)
        )
        # the fixed <ledger_dir>/fabric.json decided (a fresh 'ready' hit), the override never opened
        assert result.status is Status.PASS and result.payload["job_id"] == "4242"
        assert len(fakes.fabric_probe.calls) == 1  # only the mandatory fresh probe before submission
        assert "fabric_cache" not in AutoResearchSkill.input_schema["properties"]

    def test_a_large_fabric_ttl_s_cannot_extend_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _spec()
        payload = _env_payload(spec)
        _stage(tmp_path, _approved(spec, extra=[("launch_envelope", _campaign(spec), "-", payload)]))
        # 400 s old 'ready' is stale under the capped TTL even with a huge requested TTL: probe, do not trust
        (tmp_path / "ledger" / "fabric.json").write_text(
            _fabric("ready", CLOCK_S - 400.0, "fabric_ready"), encoding="utf-8"
        )
        fakes = _Fakes(probe_result={"state": "refused", "reason": "fabric_refused", "ts": CLOCK_S})
        result = fakes.skill.execute(
            _submit_request(tmp_path, payload, _trial(), fabric_ttl_s=1.0e12), _ctx(tmp_path)
        )
        assert result.status is Status.REFUSED
        assert ("AR-LN-003", "fabric_refused") in _findings(result)
        assert len(fakes.fabric_probe.calls) == 1 and fakes.launch_fn.calls == []

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "in-an-hour", None])
    def test_a_bad_fabric_ttl_falls_back_to_the_default(self, bad):
        assert _clamp_ttl_s(bad) == pytest.approx(DEFAULT_TTL_S)

    def test_the_fabric_ttl_is_capped_but_may_be_shortened(self):
        assert _clamp_ttl_s(1.0e12) == pytest.approx(DEFAULT_TTL_S)
        assert _clamp_ttl_s(DEFAULT_TTL_S) == pytest.approx(DEFAULT_TTL_S)
        assert _clamp_ttl_s(30.0) == pytest.approx(30.0)
        assert _clamp_ttl_s(0.0) == pytest.approx(0.0)

    def test_submit_trial_is_handed_a_fresh_uncached_probe_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        captured = {}

        def fake_submit_trial(_fact, **kwargs):
            captured["fabric"] = dict(kwargs.get("fabric") or {})
            if str(captured["fabric"].get("state") or "") != "ready":
                raise LaunchRefused("AR-LN-003 " + str(captured["fabric"].get("reason") or "fabric_refused"))
            return {"job_id": "1"}

        monkeypatch.setattr("foundationskills.skills.auto_research.skill.submit_trial", fake_submit_trial)
        spec = _spec()
        payload = _env_payload(spec)
        _stage(tmp_path, _approved(spec, extra=[("launch_envelope", _campaign(spec), "-", payload)]))
        # a fresh cache hit says 'ready' and gets the check gate past the cache ...
        (tmp_path / "ledger" / "fabric.json").write_text(_fabric("ready", CLOCK_S, "fabric_ready"), encoding="utf-8")
        fakes = _Fakes(probe_result={"state": "refused", "reason": "fabric_refused", "ts": CLOCK_S})
        result = fakes.skill.execute(
            _submit_request(tmp_path, payload, _trial(kind="train", train_request={"command": ["train", "run"]})),
            _ctx(tmp_path),
        )
        # ... but submit_trial must get the FRESH probe verdict taken immediately before it
        assert result.status is Status.REFUSED
        assert captured["fabric"]["state"] == "refused"
        assert captured["fabric"].get("cached") is None  # a bare probe result, never a cache replay
        assert ("AR-LN-003", "AR-LN-003 fabric_refused") in _findings(result)
        assert len(fakes.fabric_probe.calls) == 1
        assert fakes.launch_fn.calls == [] and fakes.runner.calls == []


class TestF3DuplicateJobIds:
    """Each real job_id is counted once (F3)."""

    def test_a_duplicate_job_id_counts_once_with_named_drops(self):
        measure = _measure_fn({"7": None})
        usage = campaign_usage(
            [{"job_id": "7", "gpu_hours_est": 2.0}, {"job_id": "7", "gpu_hours_est": 5.0}, {"job_id": "7"}],
            measure=measure,
        )
        assert usage["per_job"] == {"7": {"gpu_hours": 2.0, "source": "declared_est"}}
        assert usage["used_gpu_hours"] == 2.0
        assert usage["drops"] == ["sacct_unavailable:7", "duplicate_job:7", "duplicate_job:7"]
        assert measure.seen == ["7"]  # never measured a second time

    def test_a_duplicate_measured_job_id_keeps_its_first_measurement_only(self):
        usage = campaign_usage(
            [{"job_id": "8", "gpu_hours_est": 9.0}, {"job_id": "8", "gpu_hours_est": 9.0}],
            measure=_measure_fn({"8": 1.5}),
        )
        assert usage["per_job"] == {"8": {"gpu_hours": 1.5, "source": "measured_sacct"}}
        assert usage["used_gpu_hours"] == 1.5
        assert usage["drops"] == ["duplicate_job:8"]


class TestF4EmitDropsAreCarried:
    """emit_trial fact['drops'] is carried into the submit payloads and the job entry (F4)."""

    def test_refused_non_executable_submit_carries_missing_plus_emit_drops(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "foundationskills.skills.auto_research.skill.emit_trial",
            _emit_fact(executable=False, missing=["emit_train_missing", "emit_train_missing"],
                       drops=["emit_train_missing", "emit_fragile"]),
        )
        spec = _spec()
        payload = _env_payload(spec)
        _stage(tmp_path, _approved(spec, extra=[("launch_envelope", _campaign(spec), "-", payload)]))
        fakes = _Fakes(launch_result={"job_id": "1"})
        result = fakes.skill.execute(
            _submit_request(tmp_path, payload, _trial(kind="train", train_request={"command": ["train", "run"]})),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert result.payload["missing"] == ["emit_train_missing", "emit_train_missing"]
        assert result.payload["drops"] == ["emit_train_missing", "emit_fragile"]  # dedup, order kept

    def test_the_submit_payload_and_the_job_entry_carry_the_emit_drops(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact(drops=["emit_fragile"]))
        spec = _spec()
        payload = _env_payload(spec)
        _stage(tmp_path, _approved(spec, extra=[("launch_envelope", _campaign(spec), "-", payload)]))
        fakes = _Fakes(launch_result={"job_id": "4242"})
        result = fakes.skill.execute(_submit_request(tmp_path, payload, _trial()), _ctx(tmp_path))
        assert result.status is Status.PASS
        assert result.payload["drops"] == ["emit_fragile"]
        submitted = [e for e in Ledger(tmp_path / "ledger").entries() if e["op"] == "job_submitted"]
        assert Ledger(tmp_path / "ledger").payload(submitted[0])["emit_drops"] == ["emit_fragile"]


class TestF5ZeroMeasurementIsNotAMeasurement:
    """``<= 0`` from sacct is not a measurement: fall back to declared/uncounted (F5)."""

    def test_a_zero_sacct_measurement_falls_back_to_the_declared_estimate(self):
        usage = campaign_usage([{"job_id": "5", "gpu_hours_est": 8.0}], measure=_measure_fn({"5": 0.0}))
        assert usage["per_job"]["5"] == {"gpu_hours": 8.0, "source": "declared_est"}
        assert usage["used_gpu_hours"] == 8.0
        assert usage["drops"] == ["sacct_zero:5"]

    def test_a_zero_sacct_measurement_without_a_declaration_is_uncounted(self):
        usage = campaign_usage([{"job_id": "6"}], measure=_measure_fn({"6": 0.0}))
        assert usage["per_job"]["6"] == {"gpu_hours": 0.0, "source": "uncounted"}
        assert usage["drops"] == ["sacct_zero:6", "no_accounting:6"]
        assert usage["used_gpu_hours"] == 0.0

    def test_a_negative_sacct_measurement_is_not_a_measurement_either(self):
        usage = campaign_usage([{"job_id": "9", "gpu_hours_est": 1.0}], measure=_measure_fn({"9": -2.0}))
        assert usage["per_job"]["9"] == {"gpu_hours": 1.0, "source": "declared_est"}
        assert usage["drops"] == ["sacct_zero:9"]


class TestF6BaselineProvenance:
    """AR-HO-006 also covers an unknown baseline provenance, once the ledger carries an envelope (F6)."""

    def test_an_enveloped_campaign_fires_baseline_provenance_unknown(self, tmp_path):
        spec = _spec()
        rows = [*_baseline_rows(), *_gain_rows()]
        _stage(tmp_path, _approved(spec, extra=[
            ("launch_envelope", _campaign(spec), "-", _env_payload(spec)),
            *(("trial_result", _campaign(spec), str(row["trial"]), dict(row)) for row in rows),
        ]))
        fakes = _Fakes()
        result = fakes.skill.execute(
            _request(tmp_path, action="close", stop_reason="budget exhausted"), _ctx(tmp_path)
        )
        assert result.status is Status.UNMEASURED and result.payload["outcome"] == "unmeasured"
        fired = [f for f in result.findings if f.rule_id == "AR-HO-006"]
        assert fired and "baseline_provenance_unknown:baseline" in fired[0].message

    def test_m0_manual_baselines_without_an_envelope_stay_silent(self, tmp_path):
        spec = _spec()
        rows = [*_baseline_rows(), *_gain_rows()]
        _stage(tmp_path, _approved(spec, extra=[
            *(("trial_result", _campaign(spec), str(row["trial"]), dict(row)) for row in rows),
        ]))
        fakes = _Fakes()
        result = fakes.skill.execute(
            _request(tmp_path, action="close", stop_reason="budget exhausted"), _ctx(tmp_path)
        )
        assert result.status is Status.PASS and result.payload["outcome"] == "improved"
        assert not any(rule_id == "AR-HO-006" for rule_id, _ in _findings(result))

    def test_an_eval_only_baseline_job_keeps_the_provenance_known(self, tmp_path):
        spec = _spec()
        rows = [*_baseline_rows(), *_gain_rows()]
        _stage(tmp_path, _approved(spec, extra=[
            ("launch_envelope", _campaign(spec), "-", _env_payload(spec)),
            ("job_submitted", _campaign(spec), "-", _job("555", "baseline", kind="eval_only")),
            *(("trial_result", _campaign(spec), str(row["trial"]), dict(row)) for row in rows),
        ]))
        fakes = _Fakes()
        result = fakes.skill.execute(
            _request(tmp_path, action="close", stop_reason="budget exhausted"), _ctx(tmp_path)
        )
        assert result.status is Status.PASS and result.payload["outcome"] == "improved"
        assert not any(rule_id == "AR-HO-006" for rule_id, _ in _findings(result))
