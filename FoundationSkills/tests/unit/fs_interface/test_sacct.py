"""sacct accounting tests: GPU-count parsing, measured-vs-declared GPU hours."""
from __future__ import annotations

import inspect
import subprocess
from types import SimpleNamespace

import pytest

from foundationskills.interfaces.fs.sacct import (
    job_gpu_hours,
    parse_gpu_count,
    query_job_gpu_hours,
)
from foundationskills.skills.auto_research.accounting import campaign_usage


def _runner(stdout: str = "", returncode: int = 0):
    calls: list = []

    def _fake(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=stdout)

    _fake.calls = calls
    return _fake


def _measure(table: dict):
    seen: list[str] = []

    def _fake(job_id: str):
        seen.append(job_id)
        return table.get(job_id)

    _fake.seen = seen
    return _fake


class TestParseGpuCount:
    def test_alloc_tres_with_cpu(self):
        assert parse_gpu_count("gres/gpu=8,cpu=4") == 8

    def test_short_form(self):
        assert parse_gpu_count("gpu:8") == 8

    def test_type_qualified(self):
        assert parse_gpu_count("gres/gpu:a100:4,mem=64G") == 4

    def test_zero_gpu_is_uncountable(self):
        assert parse_gpu_count("gres/gpu=0") is None

    def test_garbage_is_none(self):
        assert parse_gpu_count("garbage") is None
        assert parse_gpu_count("cpu=32,mem=128G") is None
        assert parse_gpu_count("gres/gpu") is None
        assert parse_gpu_count("") is None
        assert parse_gpu_count(None) is None


class TestJobGpuHours:
    def test_seconds_x_gpus_over_3600(self):
        assert job_gpu_hours("7200", "gres/gpu=2,cpu=4") == 4.0
        assert job_gpu_hours("3600", "gpu:8") == 8.0
        assert job_gpu_hours(900, "gpu:4") == 1.0

    def test_junk_is_none(self):
        assert job_gpu_hours("junk", "gres/gpu=2") is None
        assert job_gpu_hours("3600", "junk") is None
        assert job_gpu_hours("", "gres/gpu=2") is None
        assert job_gpu_hours("nan", "gres/gpu=2") is None

    def test_zero_counters_are_uncountable(self):
        assert job_gpu_hours("0", "gres/gpu=8") is None
        assert job_gpu_hours("3600", "gres/gpu=0") is None


class TestQueryJobGpuHours:
    def test_measures_with_the_pinned_sacct_command(self):
        runner = _runner("3600|gres/gpu=2,cpu=4\n")
        assert query_job_gpu_hours("4242", runner=runner) == 2.0
        assert len(runner.calls) == 1
        argv, kwargs = runner.calls[0]
        # login shell: a bare sacct is not on PATH outside one (found by the M1 live smoke)
        assert argv == ["bash", "-lc", "sacct -n -X -P -j 4242 --format ElapsedRaw,AllocTRES"]
        assert kwargs == {"capture_output": True, "text": True, "timeout": 30}

    def test_non_slurm_job_id_never_reaches_the_shell(self):
        runner = _runner("3600|gpu:1\n")
        for bad in ("4242; rm -rf ~", "$(id)", "", "12 34", "abc"):
            assert query_job_gpu_hours(bad, runner=runner) is None
        assert runner.calls == []
        assert query_job_gpu_hours("4242_3", runner=runner) == 1.0

    def test_first_non_empty_line_wins(self):
        runner = _runner("\n\n3600|gpu:1\n7200|gpu:8\n")
        assert query_job_gpu_hours("1", runner=runner) == 1.0

    def test_nonzero_rc_reads_none(self):
        assert query_job_gpu_hours("7", runner=_runner("3600|gpu:2", returncode=1)) is None

    def test_parse_miss_reads_none(self):
        assert query_job_gpu_hours("7", runner=_runner("")) is None
        assert query_job_gpu_hours("7", runner=_runner("\n   \n")) is None
        assert query_job_gpu_hours("7", runner=_runner("no-pipe-separator")) is None
        assert query_job_gpu_hours("7", runner=_runner("3600|junk-tres")) is None

    def test_unavailable_sacct_reads_none(self):
        def _missing(argv, **kwargs):
            raise FileNotFoundError("no sacct")

        def _timeout(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 30)

        def _oserror(argv, **kwargs):
            raise OSError("sacct unavailable")

        assert query_job_gpu_hours("7", runner=_missing) is None
        assert query_job_gpu_hours("7", runner=_timeout) is None
        assert query_job_gpu_hours("7", runner=_oserror) is None


class TestCampaignUsage:
    def test_measured_or_declared_never_both(self):
        entries = [
            {"job_id": "3", "gpu_hours_est": 64.0},
            {"job_id": "1", "gpu_hours_est": 10.0},
            {"job_id": "2"},
        ]
        measure = _measure({"3": 3.5, "1": None, "2": None})
        usage = campaign_usage(entries, measure=measure)
        assert measure.seen == ["3", "1", "2"]
        assert usage["per_job"]["3"] == {"gpu_hours": 3.5, "source": "measured_sacct"}
        assert usage["per_job"]["1"] == {"gpu_hours": 10.0, "source": "declared_est"}
        assert usage["per_job"]["2"] == {"gpu_hours": 0.0, "source": "uncounted"}
        # 3.5 + 10.0 + 0.0: the 64.0h declaration is never added to its measurement.
        assert usage["used_gpu_hours"] == 13.5
        assert sum(v["gpu_hours"] for v in usage["per_job"].values()) == usage["used_gpu_hours"]
        assert usage["drops"] == ["sacct_unavailable:1", "no_accounting:2"]

    def test_drops_are_named_and_counted(self):
        entries = [{"job_id": "a", "gpu_hours_est": 1.0}, {"job_id": "b"}, {"job_id": "c"}]
        usage = campaign_usage(entries, measure=_measure({}))
        assert usage["drops"] == ["sacct_unavailable:a", "no_accounting:b", "no_accounting:c"]

    def test_measurement_closest_to_the_burn_is_authoritative(self):
        entries = [{"job_id": "9", "gpu_hours_est": 128.0}]
        usage = campaign_usage(entries, measure=_measure({"9": 0.25}))
        assert usage["used_gpu_hours"] == 0.25
        assert usage["per_job"]["9"] == {"gpu_hours": 0.25, "source": "measured_sacct"}
        assert usage["drops"] == []

    def test_measured_zero_is_not_a_measurement(self):
        # a sacct measurement of <= 0 is not a measurement: fall back to declared/uncounted and name it
        usage = campaign_usage(
            [{"job_id": "5", "gpu_hours_est": 8.0}], measure=_measure({"5": 0.0})
        )
        assert usage["per_job"]["5"] == {"gpu_hours": 8.0, "source": "declared_est"}
        assert usage["used_gpu_hours"] == 8.0
        assert usage["drops"] == ["sacct_zero:5"]
        usage2 = campaign_usage([{"job_id": "6"}], measure=_measure({"6": 0.0}))
        assert usage2["per_job"]["6"] == {"gpu_hours": 0.0, "source": "uncounted"}
        assert usage2["drops"] == ["sacct_zero:6", "no_accounting:6"]
        assert usage2["used_gpu_hours"] == 0.0

    @pytest.mark.parametrize("bad", [None, "n/a", float("nan"), -1.0, True])
    def test_unusable_declared_estimate_is_no_accounting(self, bad):
        usage = campaign_usage([{"job_id": "x", "gpu_hours_est": bad}], measure=_measure({}))
        assert usage["per_job"]["x"] == {"gpu_hours": 0.0, "source": "uncounted"}
        assert usage["drops"] == ["no_accounting:x"]
        assert usage["used_gpu_hours"] == 0.0

    def test_empty_campaign_accounts_nothing(self):
        measure = _measure({})
        empty = {"used_gpu_hours": 0.0, "per_job": {}, "drops": []}
        assert campaign_usage([], measure=measure) == empty
        assert campaign_usage(None, measure=measure) == empty
        assert measure.seen == []

    def test_default_measure_is_the_sacct_query(self):
        assert inspect.signature(campaign_usage).parameters["measure"].default is query_job_gpu_hours
