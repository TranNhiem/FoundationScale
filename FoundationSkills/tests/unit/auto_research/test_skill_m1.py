"""M1 skill wiring: envelope, gated submit, safe cancel, measured close + the four new must-fire fixtures.

Every test constructs ``AutoResearchSkill`` with the four injected connectors (``runner``/``launch_fn``/
``fabric_probe``/``measure``): no Slurm call, no socket and no fork may escape a unit test. The ``emit_trial``
render contract is task 5's own test surface - where the wiring under test needs an executable emit fact,
only the module symbol ``skill.emit_trial`` is monkeypatched and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills.auto_research.campaign import campaign_hash
from foundationskills.skills.auto_research.envelope import envelope_token, trial_launch_token
from foundationskills.skills.auto_research.jobs import cancel_jobs, submitted_jobs
from foundationskills.skills.auto_research.ledger import Ledger, ledger_files
from foundationskills.skills.auto_research.skill import (
    AutoResearchSkill,
    _campaign,
    _result,
    _spec,
    _val,
)

CLOCK_S = 1750000000.0
READY_PROBE = {"state": "ready", "reason": "fabric_ready", "ts": CLOCK_S}
NEW_RULES = ("AR-LN-003", "AR-LN-006", "AR-LN-007", "AR-HO-006")


def _ctx(path):
    return SkillContext(workdir=path)


def _approved(spec, extra=()):
    campaign = _campaign(spec)
    events = [("campaign_approved", campaign, "-", {"spec_hash": campaign_hash(spec), "approver": "reviewer"})]
    events.extend(list(extra))
    return events


def _stage(tmp_path, events):
    """Materialise ``ledger_files`` archive bytes under ``tmp_path`` (same harness as the M0 test)."""
    for rel, text in ledger_files(events).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path / "ledger")


def _env(budget=None):
    return {"budget": dict(budget or {"max_runs": 6, "gpu_hours_total": 24.0}), "scope": "one campaign"}


def _env_payload(spec, budget=None, approver="reviewer"):
    env = _env(budget)
    return {
        "envelope_token": envelope_token(campaign_hash(spec), env),
        "spec_hash": campaign_hash(spec),
        "envelope": env,
        "budget": dict(env["budget"]),
        "approver": approver,
    }


def _trial(kind="eval_only", **overrides):
    spec = {
        "trial": "t-sub", "role": "candidate", "kind": kind, "delta": {"optim.lr": 5e-4},
        "seed": 101, "nodes": 1, "gpus_per_node": 8, "partition": "rally", "time": "10-00:00:00",
        "commands": ["uv run train.py --config campaign.yaml"], "gpu_hours_est": 4.0,
        "eval_request": {"command": ["eval", "run", "gsm8k"]},
    }
    spec.update(overrides)
    return spec


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
    """The four injectable connectors, wired into a real ``AutoResearchSkill``."""

    def __init__(self, *, runner_result=None, launch_result=None, probe_result=None, measured=None):
        self.runner = _Rec(runner_result)
        self.launch_fn = _Rec(launch_result)
        self.fabric_probe = _Rec(READY_PROBE if probe_result is None else probe_result)
        self.measure = _Measure(measured)
        self.skill = AutoResearchSkill(
            runner=self.runner, launch_fn=self.launch_fn, fabric_probe=self.fabric_probe,
            measure=self.measure, clock=lambda: CLOCK_S,
        )


def _emit_fact(confirm="sha256:c0ffee", executable=True, missing=()):
    """An ``emit_trial`` result for the happy path (the brittle render assertion lives in test_emit_trial)."""
    def fake(request, **kwargs):
        return {
            "fs_launch_spec": {"sbatch": "#!/bin/bash\n#SBATCH --time=10-00:00:00\n", "executable": True},
            "trial_spec": dict(request["trial_spec"]),
            "confirm": confirm,
            "notes": [], "drops": [], "executable": executable, "missing": list(missing),
        }

    return fake


def _findings(result):
    return [(f.rule_id, f.message) for f in result.findings]


def _materialize(fixture, tmp_path):
    """Pure-data fixture harness (same as ``test_auto_research_skill.py``): {tmp}/{confirm} substitution."""
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


class TestEnvelopeAction:
    def test_envelope_appends_launch_envelope_and_returns_token(self, tmp_path):
        fakes = _Fakes()
        env = _env()
        result = fakes.skill.execute(_request(tmp_path, action="envelope", envelope=env), _ctx(tmp_path))
        assert result.status is Status.PASS
        assert result.payload == {"envelope_token": envelope_token(campaign_hash(_spec()), env)}
        ledger = Ledger(tmp_path / "ledger")
        rows = [e for e in ledger.entries() if e["op"] == "launch_envelope"]
        assert len(rows) == 1
        payload = ledger.payload(rows[0])
        assert payload["envelope_token"] == result.payload["envelope_token"]
        assert payload["spec_hash"] == campaign_hash(_spec()) and payload["approver"] == "reviewer"
        assert payload["envelope"] == {"budget": {"max_runs": 6, "gpu_hours_total": 24.0}, "scope": "one campaign"}
        assert payload["budget"] == payload["envelope"]["budget"]
        assert ledger.verify() == []
        assert fakes.runner.calls == [] and fakes.launch_fn.calls == []

    def test_envelope_budget_mismatch_refuses(self, tmp_path):
        fakes = _Fakes()
        env = {"budget": {"max_runs": 2, "gpu_hours_total": 24.0}}  # the spec says max_runs: 6
        result = fakes.skill.execute(_request(tmp_path, action="envelope", envelope=env), _ctx(tmp_path))
        assert result.status is Status.REFUSED
        assert result.findings and ("AR-LN-006", "envelope_budget_mismatch") in _findings(result)
        assert not [e for e in Ledger(tmp_path / "ledger").entries() if e["op"] == "launch_envelope"]


class TestSubmitAction:
    def _seed_envelope(self, tmp_path, spec=None, env_budget=None):
        spec = _spec() if spec is None else spec
        payload = _env_payload(spec, env_budget)
        _stage(tmp_path, _approved(spec, extra=[("launch_envelope", _campaign(spec), "-", payload)]))
        return payload

    def test_submit_happy_path_ledgers_job_and_budget_after(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        fakes = _Fakes(launch_result={"job_id": "4242", "state": "submitted"})
        env_payload = self._seed_envelope(tmp_path)
        spec, trial = _spec(), _trial()
        request = _request(
            tmp_path, action="submit", trial_spec=trial,
            launch_token=trial_launch_token(env_payload["envelope_token"], trial),
        )
        result = fakes.skill.execute(request, _ctx(tmp_path))
        assert result.status is Status.PASS
        assert result.payload["job_id"] == "4242"
        after = result.payload["budget_after"]
        assert after["runs_left"] == 5 and after["hours_left"] == pytest.approx(20.0)
        ledger = Ledger(tmp_path / "ledger")
        assert [e["op"] for e in ledger.entries()] == [
            "campaign_approved", "launch_envelope", "launch_authorised", "job_submitted",
        ]
        authorised = ledger.launches(_campaign(spec))
        assert authorised[0]["launch_token"] == request["launch_token"]
        assert authorised[0]["gpu_hours_est"] == 4.0
        submitted = submitted_jobs(ledger, _campaign(spec))
        assert submitted[0]["job_id"] == "4242" and submitted[0]["kind"] == "eval_only"
        assert submitted[0]["trial"] == "t-sub" and submitted[0]["launch_token"] == request["launch_token"]
        assert submitted[0]["gpu_hours_est"] == 4.0 and submitted[0]["submitted_at"] == CLOCK_S
        assert submitted[0]["budget_after"]["runs_left"] == 5
        assert ledger.verify() == []
        # confirm passthrough into the injected launch, and no real sbatch/scancel fork ever
        assert fakes.launch_fn.calls[0][0][0]["sbatch"].startswith("#!/bin/bash")
        assert fakes.launch_fn.calls[0][1]["confirm"] == "sha256:c0ffee"
        assert fakes.runner.calls == []

    def test_submit_without_envelope_is_refused_before_any_launch(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        fakes = _Fakes(launch_result={"job_id": "4242"})
        _stage(tmp_path, _approved(_spec()))  # approval only: no launch_envelope in the chain
        result = fakes.skill.execute(
            _request(tmp_path, action="submit", trial_spec=_trial(), launch_token="fs-ar-trial-v1-anything"),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert result.findings and ("AR-LN-006", "envelope_missing") in _findings(result)
        assert fakes.launch_fn.calls == [] and fakes.runner.calls == []

    def test_submit_forged_envelope_token_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        fakes = _Fakes(launch_result={"job_id": "4242"})
        spec, trial = _spec(), _trial()
        forged = dict(_env_payload(spec), envelope_token="forged-token")  # valid chain entry, underived token
        _stage(tmp_path, _approved(spec, extra=[("launch_envelope", _campaign(spec), "-", forged)]))
        result = fakes.skill.execute(
            _request(tmp_path, action="submit", trial_spec=trial,
                     launch_token=trial_launch_token("forged-token", trial)),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert result.findings and ("AR-LN-006", "envelope_token_forged") in _findings(result)
        assert fakes.launch_fn.calls == [] and fakes.runner.calls == []

    def test_submit_wrong_launch_token_is_refused_with_token_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        fakes = _Fakes(launch_result={"job_id": "4242"})
        self._seed_envelope(tmp_path)
        result = fakes.skill.execute(
            _request(tmp_path, action="submit", trial_spec=_trial(), launch_token="not-what-you-derived"),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert result.findings and ("AR-LN-006", "token_mismatch") in _findings(result)
        assert fakes.launch_fn.calls == [] and fakes.runner.calls == []

    def test_submit_refused_on_refused_fabric_and_sticky_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        fakes = _Fakes(launch_result={"job_id": "4242"})
        env_payload = self._seed_envelope(tmp_path)
        trial = _trial()
        (tmp_path / "ledger" / "fabric.json").write_text(
            _fabric("refused", 4102444800.0, "fabric_refused"), encoding="utf-8"
        )
        result = fakes.skill.execute(
            _request(
                tmp_path, action="submit", trial_spec=trial,
                launch_token=trial_launch_token(env_payload["envelope_token"], trial),
                fabric_cache=str(tmp_path / "ledger" / "fabric.json"),
            ),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert result.findings and ("AR-LN-003", "fabric_refused") in _findings(result)
        # a future-dated refusal is a sticky cache hit: the probe never opens a socket
        assert fakes.fabric_probe.calls == [] and fakes.launch_fn.calls == [] and fakes.runner.calls == []

    def test_submit_refused_on_unmeasured_fabric_names_the_reason(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        fakes = _Fakes(
            launch_result={"job_id": "4242"},
            probe_result={"state": "unmeasured", "reason": "fabric_unmeasured:timeout", "ts": 0.0},
        )
        env_payload = self._seed_envelope(tmp_path)
        trial = _trial()
        result = fakes.skill.execute(
            _request(tmp_path, action="submit", trial_spec=trial,
                     launch_token=trial_launch_token(env_payload["envelope_token"], trial)),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert result.findings and ("AR-LN-003", "fabric_unmeasured:timeout") in _findings(result)
        assert len(fakes.fabric_probe.calls) == 1  # probe runs, its verdict blocks
        assert fakes.launch_fn.calls == [] and fakes.runner.calls == []

    def test_submit_refused_when_envelope_runs_exhausted_with_measured_usage(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _spec(budget={"gpu_hours_total": 24.0, "max_runs": 1, "per_run_timeout_h": 8.0, "reserve_frac": 0.3})
        trial = _trial(trial="second-run")
        env_budget = {"max_runs": 1, "gpu_hours_total": 24.0}  # mirrors spec.budget on the gate keys
        payload = _env_payload(spec, env_budget)
        _stage(tmp_path, _approved(spec, extra=[
            ("launch_envelope", _campaign(spec), "-", payload),
            ("job_submitted", _campaign(spec), "-", _job("123456", "t-first", gpu_hours_est=4.0)),
        ]))
        fakes = _Fakes(launch_result={"job_id": "1"}, measured={"123456": 3.5})
        result = fakes.skill.execute(
            _request(tmp_path, campaign_spec=spec, campaign_confirm=campaign_hash(spec),
                     action="submit", trial_spec=trial,
                     launch_token=trial_launch_token(payload["envelope_token"], trial)),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert result.findings and ("AR-LN-006", "envelope_exhausted:runs") in _findings(result)
        # the measured hour figure (not the declared 4.0h est) is what fed the envelope gate
        assert fakes.measure.calls == [(('123456',), {})]
        assert not any(message == "envelope_exhausted:gpu_hours" for _, message in _findings(result))
        assert fakes.launch_fn.calls == [] and fakes.runner.calls == []

    def test_submit_uses_measured_hours_before_the_declared_estimate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        spec = _spec(budget={"gpu_hours_total": 24.0, "max_runs": 6, "per_run_timeout_h": 8.0, "reserve_frac": 0.3})
        trial = _trial(trial="last-hours")
        payload = _env_payload(spec, {"max_runs": 6, "gpu_hours_total": 24.0})
        _stage(tmp_path, _approved(spec, extra=[
            ("launch_envelope", _campaign(spec), "-", payload),
            ("job_submitted", _campaign(spec), "-", _job("123456", "t-first", gpu_hours_est=1.0)),
        ]))
        fakes = _Fakes(launch_result={"job_id": "1"}, measured={"123456": 20.5})  # measured, not 1.0 declared
        result = fakes.skill.execute(
            _request(tmp_path, campaign_spec=spec, campaign_confirm=campaign_hash(spec),
                     action="submit", trial_spec=trial,
                     launch_token=trial_launch_token(payload["envelope_token"], trial)),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert ("AR-LN-006", "envelope_exhausted:gpu_hours") in _findings(result)  # 24 - 20.5 - 4 < 0

    def test_submit_refuses_when_the_emit_fact_is_not_executable(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial",
                            _emit_fact(executable=False, missing=["emit_train_missing"]))
        fakes = _Fakes(launch_result={"job_id": "4242"})
        env_payload = self._seed_envelope(tmp_path)
        trial = _trial(kind="train", train_request={"command": ["train", "run"]})
        result = fakes.skill.execute(
            _request(tmp_path, action="submit", trial_spec=trial,
                     launch_token=trial_launch_token(env_payload["envelope_token"], trial)),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert result.payload["missing"] == ["emit_train_missing"] and result.payload["drops"] == ["emit_train_missing"]
        assert fakes.launch_fn.calls == [] and fakes.runner.calls == []

    def test_submit_launch_refused_at_run_time_is_refused_with_reason(self, tmp_path, monkeypatch):
        from foundationskills.interfaces.fs.launch import LaunchRefused

        def refuse(*_a, **_k):
            raise LaunchRefused("AR-LN-003 fabric_refused")

        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.submit_trial", refuse)
        fakes = _Fakes(launch_result={"job_id": "4242"})
        env_payload = self._seed_envelope(tmp_path)
        trial = _trial(kind="train", train_request={"command": ["train", "run"]})
        result = fakes.skill.execute(
            _request(tmp_path, action="submit", trial_spec=trial,
                     launch_token=trial_launch_token(env_payload["envelope_token"], trial)),
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED and result.refusal == "AR-LN-003 fabric_refused"
        assert ("AR-LN-003", "AR-LN-003 fabric_refused") in _findings(result)
        assert fakes.launch_fn.calls == []

    def test_submit_without_job_id_is_unmeasured_and_logs_no_job_submitted(self, tmp_path, monkeypatch):
        monkeypatch.setattr("foundationskills.skills.auto_research.skill.emit_trial", _emit_fact())
        fakes = _Fakes(launch_result={"job_id": ""})  # the launch reported no job id at all
        env_payload = self._seed_envelope(tmp_path)
        trial = _trial(kind="train", train_request={"command": ["train", "run"]})
        result = fakes.skill.execute(
            _request(tmp_path, action="submit", trial_spec=trial,
                     launch_token=trial_launch_token(env_payload["envelope_token"], trial)),
            _ctx(tmp_path),
        )
        assert result.status is Status.UNMEASURED
        assert "no_job_id" in result.payload["drops"]
        ops = [e["op"] for e in Ledger(tmp_path / "ledger").entries()]
        assert "launch_authorised" in ops and "job_submitted" not in ops


class TestCancelAction:
    def test_cancel_owned_ids_batches_scancel_and_appends_job_cancelled(self, tmp_path):
        spec = _spec()
        ledger_dir = _stage(tmp_path, _approved(spec, extra=[
            ("job_submitted", _campaign(spec), "-", _job("123456", "t1")),
        ]))
        fakes = _Fakes(runner_result=SimpleNamespace(returncode=0))
        result = fakes.skill.execute(
            _request(tmp_path, action="cancel", job_ids=["123456"], reason="operator stop"), _ctx(tmp_path)
        )
        assert result.status is Status.PASS
        assert result.payload == {"cancelled": ["123456"], "drops": []}
        assert fakes.runner.calls[0][0][0] == ["bash", "-lc", "scancel 123456"]
        assert fakes.runner.calls[0][1] == {"capture_output": True, "text": True, "timeout": 60}
        ledger = Ledger(Path(ledger_dir))
        cancelled = [ledger.payload(e) for e in ledger.entries() if e["op"] == "job_cancelled"]
        assert cancelled[0]["job_ids"] == ["123456"] and cancelled[0]["reason"] == "operator stop"
        assert ledger.verify() == [] and fakes.launch_fn.calls == []

    def test_cancel_foreign_id_refuses_without_touching_the_runner(self, tmp_path):
        spec = _spec()
        ledger_dir = _stage(tmp_path, _approved(spec, extra=[
            ("job_submitted", _campaign(spec), "-", _job("123456", "t1")),
        ]))
        fakes = _Fakes(runner_result=SimpleNamespace(returncode=0))
        result = fakes.skill.execute(
            _request(tmp_path, action="cancel", job_ids=["999999"], reason="not mine"), _ctx(tmp_path)
        )
        assert result.status is Status.REFUSED
        assert result.findings and ("AR-LN-007", "job 999999 is not owned by ledger ar-fixture") in _findings(result)
        assert fakes.runner.calls == []
        assert [e["op"] for e in Ledger(Path(ledger_dir)).entries()] == ["campaign_approved", "job_submitted"]

    def test_cancel_without_job_ids_refuses(self, tmp_path):
        spec = _spec()
        _stage(tmp_path, _approved(spec, extra=[("job_submitted", _campaign(spec), "-", _job("123456", "t1"))]))
        fakes = _Fakes(runner_result=SimpleNamespace(returncode=0))
        result = fakes.skill.execute(_request(tmp_path, action="cancel", job_ids=[]), _ctx(tmp_path))
        assert result.status is Status.REFUSED
        assert result.findings and ("AR-LN-007", "no job ids") in _findings(result)
        assert fakes.runner.calls == []

    def test_cancel_scancel_failure_is_red_with_named_drops(self, tmp_path):
        spec = _spec()
        ledger_dir = _stage(tmp_path, _approved(spec, extra=[
            ("job_submitted", _campaign(spec), "-", _job("123456", "t1")),
        ]))
        fakes = _Fakes(runner_result=SimpleNamespace(returncode=3))
        result = fakes.skill.execute(
            _request(tmp_path, action="cancel", job_ids=["123456"], reason="operator"), _ctx(tmp_path)
        )
        assert result.status is Status.RED
        assert result.payload == {"cancelled": [], "drops": ["scancel_failed:123456"]}
        assert not [e for e in Ledger(Path(ledger_dir)).entries() if e["op"] == "job_cancelled"]

    def test_cancel_jobs_never_sends_a_foreign_id_to_the_runner(self, tmp_path):
        """jobs.cancel_jobs itself: foreign ids yield AR-LN-007 + a named drop and never reach argv."""
        spec = _spec()
        ledger_dir = _stage(tmp_path, _approved(spec, extra=[
            ("job_submitted", _campaign(spec), "-", _job("123456", "t1")),
        ]))
        ledger = Ledger(Path(ledger_dir))
        runner = _Rec(SimpleNamespace(returncode=0))
        out = cancel_jobs(ledger, _campaign(spec), ["999999", "123456"], reason="mix", runner=runner)
        assert out["cancelled"] == ["123456"] and out["drops"] == ["refused_foreign_job:999999"]
        assert ("AR-LN-007", "job 999999 is not owned by ledger ar-fixture") in out["findings"]
        assert runner.calls[0][0][0] == ["bash", "-lc", "scancel 123456"]
        runner2 = _Rec(SimpleNamespace(returncode=0))
        out2 = cancel_jobs(ledger, _campaign(spec), ["999999"], reason="all foreign", runner=runner2)
        assert runner2.calls == [] and out2["cancelled"] == []
        assert out2["drops"] == ["refused_foreign_job:999999"]


class TestCloseMeasuresUsage:
    def test_close_ar_ho006_fires_for_train_baseline_jobs(self, tmp_path):
        spec = _spec()
        rows = [*_baseline_rows(), *_gain_rows()]
        events = _approved(spec, extra=[
            ("job_submitted", _campaign(spec), "-", _job("555", "baseline", kind="train", gpu_hours_est=8.0)),
            *(("trial_result", _campaign(spec), "baseline", dict(row)) for row in rows),
        ])
        _stage(tmp_path, events)
        fakes = _Fakes(measured={"555": 2.5})  # measured hours replace the 8.0h declaration
        result = fakes.skill.execute(
            _request(tmp_path, action="close", stop_reason="budget exhausted"), _ctx(tmp_path)
        )
        assert result.status is Status.UNMEASURED
        assert result.payload["outcome"] == "unmeasured"
        fired = [f for f in result.findings if f.rule_id == "AR-HO-006"]
        assert fired and "baseline" in fired[0].message and "eval_only" in fired[0].message
        budgets = result.payload["budgets"]
        assert budgets["used_gpu_hours"] == pytest.approx(2.5)
        assert budgets["measured_gpu_hours"] == pytest.approx(2.5) and budgets["declared_gpu_hours"] == 0.0
        assert budgets["per_job"]["555"] == {"gpu_hours": pytest.approx(2.5), "source": "measured_sacct"}
        assert budgets["drops"] == []

    def test_close_records_the_named_fallback_drop(self, tmp_path):
        spec = _spec()
        rows = [*_baseline_rows(), *_gain_rows()]
        events = _approved(spec, extra=[
            ("job_submitted", _campaign(spec), "-", _job("555", "baseline", kind="eval_only", gpu_hours_est=8.0)),
            *(("trial_result", _campaign(spec), "baseline", dict(row)) for row in rows),
        ])
        _stage(tmp_path, events)
        fakes = _Fakes()  # measure returns None -> declared fallback, named and counted
        result = fakes.skill.execute(
            _request(tmp_path, action="close", stop_reason="budget exhausted"), _ctx(tmp_path)
        )
        assert result.status is Status.PASS and result.payload["outcome"] == "improved"
        budgets = result.payload["budgets"]
        assert budgets["drops"] == ["sacct_unavailable:555"]
        assert budgets["used_gpu_hours"] == pytest.approx(8.0)
        assert budgets["per_job"]["555"]["source"] == "declared_est"
        assert budgets["declared_gpu_hours"] == pytest.approx(8.0) and budgets["measured_gpu_hours"] == 0.0

    def test_close_ar_ho006_stays_silent_for_eval_only_baseline_jobs(self, tmp_path):
        spec = _spec()
        rows = [*_baseline_rows(), *_gain_rows()]
        events = _approved(spec, extra=[
            ("job_submitted", _campaign(spec), "-", _job("555", "baseline", kind="eval_only")),
            *(("trial_result", _campaign(spec), "baseline", dict(row)) for row in rows),
        ])
        _stage(tmp_path, events)
        fakes = _Fakes()
        result = fakes.skill.execute(
            _request(tmp_path, action="close", stop_reason="budget exhausted"), _ctx(tmp_path)
        )
        assert result.status is Status.PASS and result.payload["outcome"] == "improved"
        assert not any(f.rule_id == "AR-HO-006" for f in result.findings)

    def test_close_ar_ho006_stays_silent_for_legacy_manual_baselines(self, tmp_path):
        """M0 backward compat: baseline results without a job_submitted entry never fire without an envelope.

        The same rows inside a campaign whose ledger carries any launch_envelope entry DO fire AR-HO-006
        with 'baseline_provenance_unknown:<trial>' (F6): an M1 campaign's baseline must be provenance-known.
        """
        spec = _spec()
        rows = [*_baseline_rows(), *_gain_rows()]
        events = _approved(spec, extra=[
            *(("trial_result", _campaign(spec), str(row["trial"]), dict(row)) for row in rows),
        ])
        _stage(tmp_path, events)
        fakes = _Fakes()
        result = fakes.skill.execute(
            _request(tmp_path, action="close", stop_reason="budget exhausted"), _ctx(tmp_path)
        )
        assert result.status is Status.PASS and result.payload["outcome"] == "improved"
        assert not any(f.rule_id == "AR-HO-006" for f in result.findings)
        assert fakes.measure.calls == []  # no job_submitted entry means no measurement either

        # the same manual rows in an M1 campaign (a launch_envelope is ledgered) lose that compat:
        root = tmp_path / "enveloped"
        _stage(root, _approved(spec, extra=[
            ("launch_envelope", _campaign(spec), "-", _env_payload(spec)),
            *(("trial_result", _campaign(spec), str(row["trial"]), dict(row)) for row in rows),
        ]))
        fakes2 = _Fakes()
        result2 = fakes2.skill.execute(
            _request(root, action="close", stop_reason="budget exhausted"), _ctx(root)
        )
        assert result2.status is Status.UNMEASURED and result2.payload["outcome"] == "unmeasured"
        fired = [f for f in result2.findings if f.rule_id == "AR-HO-006"]
        assert fired and "baseline_provenance_unknown:baseline" in fired[0].message


class TestNewMustFireFixtures:
    @pytest.mark.parametrize(
        "rule,message",
        [
            ("AR-LN-003", "fabric_refused"),
            ("AR-LN-006", "envelope_missing"),
            ("AR-LN-007", "is not owned by ledger"),
            ("AR-HO-006", "non-eval_only"),
        ],
    )
    def test_fixture_fires_with_the_named_reason(self, rule, message, tmp_path):
        fixtures = AutoResearchSkill().must_fire_fixtures()
        assert rule in fixtures
        request = _materialize(fixtures[rule], tmp_path)
        fakes = _Fakes(launch_result={"job_id": "1"})
        result = fakes.skill.execute(request, SkillContext(workdir=tmp_path))
        fired = [f for f in result.findings if f.rule_id == rule]
        assert fired and message in " ".join(f.message for f in fired)
        # pure-data fixtures: nothing was probed, forked or submitted
        assert fakes.runner.calls == [] and fakes.fabric_probe.calls == [] and fakes.launch_fn.calls == []

    def test_new_fixtures_are_input_blocks_except_the_handoff_warn(self, tmp_path):
        fixtures = AutoResearchSkill().must_fire_fixtures()
        for rule in ("AR-LN-003", "AR-LN-006", "AR-LN-007"):
            fakes = _Fakes(launch_result={"job_id": "1"})
            result = fakes.skill.execute(_materialize(fixtures[rule], tmp_path / rule), SkillContext(workdir=tmp_path / rule))
            assert result.status is Status.REFUSED
        fakes = _Fakes(launch_result={"job_id": "1"})
        result = fakes.skill.execute(_materialize(fixtures["AR-HO-006"], tmp_path / "ho"), SkillContext(workdir=tmp_path / "ho"))
        assert result.status is Status.UNMEASURED  # handoff WARN changes the close status
