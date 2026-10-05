"""CPU-only unit tests for the optional squeue state connector (injected runner only)."""
from __future__ import annotations

import shlex
import subprocess
from types import SimpleNamespace

from foundationskills.interfaces.fs.squeue import IN_FLIGHT_STATES, job_states, parse_states


def fake_runner(stdout="", returncode=0, error=None):
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        if error is not None:
            raise error
        return SimpleNamespace(returncode=returncode, stdout=stdout)

    return runner, calls


def test_parse_states_rows():
    assert parse_states("42 PD") == {"42": "PD"}
    assert parse_states("42 R 12:00") == {"42": "R"}  # trailing fields ignored
    assert parse_states("\n\n 111 PD \n42_7 CG\n") == {"111": "PD", "42_7": "CG"}


def test_parse_states_blank_and_garbage():
    assert parse_states("") == {}
    assert parse_states(None) == {}
    assert parse_states("\n   \n") == {}
    assert parse_states("hello") == {}  # one field: no state
    assert parse_states("123") == {}
    assert parse_states("abc PD") == {}  # not a job id
    assert parse_states("squeue: error: invalid job id specified") == {}


def test_job_states_present_absent_and_unknown_codes():
    runner, calls = fake_runner(stdout="111 PD\n222 R\n333 CG\n444 XX\n")
    out = job_states(["111", "222", "333", "444", "555"], runner=runner)
    assert out == {
        "111": "pending",  # PD
        "222": "running",  # R
        "333": "running",  # CG: the in-flight family is past the pending gate
        "444": "unknown",  # seen but unclassifiable: fail closed
        "555": "terminal",  # absent from a successful query (squeue drops finished jobs)
    }
    argv, kwargs = calls[0]
    expected = shlex.join(["squeue", "-h", "-o", "%i %t", "-j", "111,222,333,444,555"])
    assert argv == ["bash", "-lc", expected]
    assert kwargs == {"capture_output": True, "text": True, "timeout": 30}
    assert set(out.values()) <= {"pending", "running", "terminal", "unknown", None}


def test_job_states_argv_shape():
    runner, calls = fake_runner(stdout="111 PD")
    assert job_states(["111"], runner=runner)["111"] == "pending"
    assert calls[0][0] == ["bash", "-lc", "squeue -h -o '%i %t' -j 111"]


def test_job_states_invalid_ids_never_reach_the_runner():
    runner, calls = fake_runner(stdout="111 PD", error=AssertionError("runner must not be called"))
    out = job_states(["abc", "12x", "11.5"], runner=runner)
    assert out == {"abc": None, "12x": None, "11.5": None}
    assert calls == []

    runner, calls = fake_runner(stdout="111 PD")
    out = job_states(["abc", "111", "42_7", "-1"], runner=runner)
    assert out == {"abc": None, "111": "pending", "42_7": "terminal", "-1": None}
    assert calls[0][0][2].endswith("-j 111,42_7")  # valid ids only, deduped in order


def test_job_states_failure_reads_none_per_id():
    for error in (
        FileNotFoundError("squeue"),
        OSError("squeue"),
        subprocess.TimeoutExpired(cmd="squeue", timeout=30),
    ):
        runner, calls = fake_runner(error=error)
        assert job_states(["111", "42_7"], runner=runner) == {"111": None, "42_7": None}
        assert len(calls) == 1
    # nonzero rc: nothing reads 'terminal' (absent -> 'terminal' only when rc == 0)
    runner, calls = fake_runner(stdout="111 PD", returncode=1)
    assert job_states(["111", "222"], runner=runner) == {"111": None, "222": None}
    runner, calls = fake_runner(stdout="111 PD", returncode=2)
    assert job_states(["111"], runner=runner) == {"111": None}


def test_job_states_empty_request_and_state_fn_use():
    runner, calls = fake_runner(stdout="")
    assert job_states([], runner=runner) == {}
    assert calls == []

    runner, calls = fake_runner(stdout="111 PD")
    state = job_states(["111"], runner=runner)
    state_fn = state.get  # injected state_fn shape: one id -> state | None
    assert state_fn("111") == "pending"
    assert state_fn("999") is None
    assert calls == calls  # one query for the batch
    assert IN_FLIGHT_STATES == {"PD", "R", "RQ", "CG", "S", "RS"}
