""""Job ownership and safe cancel tests: only owned ids reach scancel, every refusal is named."""
from __future__ import annotations

import foundationskills.skills.auto_research.jobs as jobs_mod
from foundationskills.skills.auto_research.jobs import (
    cancel_jobs,
    cancel_token,
    owned_job_ids,
    submitted_jobs,
)
from foundationskills.skills.auto_research.ledger import Ledger


def _submit(
    ledger: Ledger, campaign: str, job_id: str, trial: str = "t1", kind: str = "train"
) -> dict:
    return ledger.append(
        "job_submitted",
        campaign,
        trial,
        {
            "trial": trial,
            "job_id": job_id,
            "kind": kind,
            "launch_token": "lt",
            "gpu_hours_est": 1.0,
            "budget_after": {"runs_left": 1, "hours_left": 8.0},
            "submitted_at": "2026-01-01T00:00:00Z",
        },
    )


def _recorder(returncode: int = 0):
    calls: list[tuple[list[str], dict]] = []

    def runner(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return type("Result", (), {"returncode": returncode})()

    return calls, runner


class TestOwnership:
    def test_submitted_jobs_scoped_to_one_campaign(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        _submit(ledger, "c1", "111111")
        _submit(ledger, "c1", "222222", trial="t2")
        _submit(ledger, "c2", "333333")
        # a job_id inside some other op's payload is NOT an ownership record
        ledger.append(
            "trial_result",
            "c1",
            "t1",
            {"trial": "t1", "job_id": "444444", "status": "ok", "metrics": {}},
        )
        found = submitted_jobs(ledger, "c1")
        assert [payload["job_id"] for payload in found] == ["111111", "222222"]
        assert owned_job_ids(ledger, "c1") == {"111111", "222222"}
        assert owned_job_ids(ledger, "c2") == {"333333"}
        assert owned_job_ids(ledger, "c3") == set()

    def test_job_id_forms_whitelisted_for_the_shell(self):
        assert jobs_mod.JOB_ID_RE.match("123456")
        assert jobs_mod.JOB_ID_RE.match("123456_7")
        for bad in ("", "abc", "123; rm -rf /", "-1", "12 34"):
            assert not jobs_mod.JOB_ID_RE.match(bad)


class TestCancelJobs:
    def test_owned_ids_cancelled_in_one_call_and_appended(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        _submit(ledger, "c1", "123456")
        _submit(ledger, "c1", "123456_7", trial="t2")
        calls, runner = _recorder()
        out = cancel_jobs(
            ledger, "c1", ["123456", "123456_7", "123456"], reason="trial stop", runner=runner
        )
        assert out["cancelled"] == ["123456", "123456_7"]
        assert out["drops"] == []
        assert out["returncode"] == 0
        assert out["findings"] == []
        assert calls == [
            (
                ["bash", "-lc", "scancel 123456 123456_7"],
                {"capture_output": True, "text": True, "timeout": 60},
            )
        ]
        appended = [entry for entry in ledger.entries() if entry.get("op") == "job_cancelled"]
        assert len(appended) == 1
        assert appended[0]["campaign"] == "c1" and appended[0]["trial"] == "-"
        payload = ledger.payload(appended[0])
        assert payload["job_ids"] == ["123456", "123456_7"]
        assert payload["reason"] == "trial stop"
        assert payload["token"] == cancel_token(["123456", "123456_7"], "trial stop")
        assert ledger.verify() == []
        assert [entry["op"] for entry in ledger.entries()][-2:] == ["job_submitted", "job_cancelled"]

    def test_foreign_id_never_reaches_runner(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        _submit(ledger, "c1", "123456")
        calls, runner = _recorder()
        out = cancel_jobs(ledger, "c1", ["999999"], reason="stop", runner=runner)
        assert calls == []
        assert out["cancelled"] == []
        assert out["returncode"] == 0
        assert out["drops"] == ["refused_foreign_job:999999"]
        assert ("AR-LN-007", "job 999999 is not owned by ledger c1") in out["findings"]
        assert [entry["op"] for entry in ledger.entries()] == ["job_submitted"]

    def test_invalid_job_id_never_reaches_runner(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        _submit(ledger, "c1", "123456")
        calls, runner = _recorder()
        hostile = "123456; rm -rf /"
        out = cancel_jobs(ledger, "c1", [hostile], reason="stop", runner=runner)
        assert calls == []
        assert out["cancelled"] == []
        assert out["returncode"] == 0
        assert out["drops"] == [f"invalid_job_id:{hostile}"]
        assert any(rule == "AR-LN-007" for rule, _ in out["findings"])
        assert [entry["op"] for entry in ledger.entries()] == ["job_submitted"]

    def test_all_foreign_runner_untouched(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        calls, runner = _recorder()
        out = cancel_jobs(ledger, "c1", ["777", "888"], reason="stop all", runner=runner)
        assert calls == []
        assert out["cancelled"] == []
        assert out["drops"] == ["refused_foreign_job:777", "refused_foreign_job:888"]
        assert ledger.entries() == []

    def test_campaign_scope_only_own_submissions(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        _submit(ledger, "c2", "555555")
        calls, runner = _recorder()
        foreign = cancel_jobs(ledger, "c1", ["555555"], reason="stop", runner=runner)
        assert calls == []
        assert foreign["drops"] == ["refused_foreign_job:555555"]
        own = cancel_jobs(ledger, "c2", ["555555"], reason="stop", runner=runner)
        assert own["cancelled"] == ["555555"]
        assert calls == [
            (["bash", "-lc", "scancel 555555"], {"capture_output": True, "text": True, "timeout": 60})
        ]
        assert [entry["op"] for entry in ledger.entries() if entry["campaign"] == "c2"] == [
            "job_submitted",
            "job_cancelled",
        ]

    def test_mixed_batch_only_owned_reach_runner(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        _submit(ledger, "c1", "123456")
        calls, runner = _recorder()
        out = cancel_jobs(
            ledger, "c1", ["999999", "123456", "abc"], reason="mixed stop", runner=runner
        )
        assert out["cancelled"] == ["123456"]
        assert out["drops"] == ["refused_foreign_job:999999", "invalid_job_id:abc"]
        assert [rule for rule, _ in out["findings"]] == ["AR-LN-007", "AR-LN-007"]
        assert calls == [
            (["bash", "-lc", "scancel 123456"], {"capture_output": True, "text": True, "timeout": 60})
        ]

    def test_runner_failure_names_drops_and_appends_nothing(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        _submit(ledger, "c1", "123456")
        calls, runner = _recorder(returncode=2)
        out = cancel_jobs(ledger, "c1", ["123456"], reason="stop", runner=runner)
        assert len(calls) == 1
        assert out["cancelled"] == []
        assert out["returncode"] == 2
        assert out["drops"] == ["scancel_failed:123456"]
        assert [entry["op"] for entry in ledger.entries()] == ["job_submitted"]

    def test_runner_error_reads_as_unavailable(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        _submit(ledger, "c1", "123456")
        calls: list[list[str]] = []

        def boom(argv, **kwargs):
            calls.append(list(argv))
            raise OSError("no slurm")

        out = cancel_jobs(ledger, "c1", ["123456"], reason="stop", runner=boom)
        assert out["cancelled"] == []
        assert out["returncode"] == 1
        assert out["drops"] == ["scancel_unavailable:123456"]
        assert [entry["op"] for entry in ledger.entries()] == ["job_submitted"]


class TestCancelToken:
    def test_deterministic_and_payload_sensitive(self):
        base = cancel_token(["123456"], "stop")
        assert base == cancel_token(["123456"], "stop")
        assert base != cancel_token(["654321"], "stop")
        assert base != cancel_token(["123456"], "other")
        assert base != cancel_token(["123456", "654321"], "stop")
