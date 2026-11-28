"""The test-based SWE reward: resolved / regression / unresolved / partial / abstentions."""

from __future__ import annotations

import asyncio
import json

import pytest

from foundationscale.agentic_rl.contracts import Termination
from foundationscale.agentic_rl.harness.base import EpisodeOutcome, EpisodeTask
from foundationscale.agentic_rl.rewards.swe import SweReward

F2P = ["tests/test_a.py::test_bug", "tests/test_a.py::test_bug2"]
P2P = ["tests/test_a.py::test_ok"]


def _task(f2p: object = F2P, p2p: object = P2P) -> EpisodeTask:
    meta = {"fail_to_pass": json.dumps(f2p), "pass_to_pass": json.dumps(p2p)}
    return EpisodeTask(
        uid="t", session_id="t#0", messages=({"role": "user", "content": "fix"},), metadata=meta
    )


def _outcome(meta: dict[str, str]) -> EpisodeOutcome:
    return EpisodeOutcome(
        trajectory_turns=(), prompt_turns=(), termination=Termination.STOP,
        submitted_answer="done", abstention_reason=None, metadata=meta,
    )  # fmt: skip


def _report(results: dict[str, str], status: str = "ok") -> dict[str, str]:
    return {"swe_status": status, "swe_tests": json.dumps(results)}


def _score(task: EpisodeTask, outcome: EpisodeOutcome, partial: bool = False) -> tuple:
    verdict = asyncio.run(SweReward(partial_credit=partial).score(task, outcome))
    return verdict.value, verdict.reason


ALL_PASS = {F2P[0]: "passed", F2P[1]: "passed", P2P[0]: "passed"}


def test_resolved() -> None:
    assert _score(_task(), _outcome(_report(ALL_PASS))) == (1.0, "resolved")


def test_regression_beats_a_fix() -> None:
    results = {**ALL_PASS, P2P[0]: "failed"}
    assert _score(_task(), _outcome(_report(results))) == (0.0, "regression")


def test_unresolved_and_partial() -> None:
    results = {**ALL_PASS, F2P[1]: "failed"}
    assert _score(_task(), _outcome(_report(results))) == (0.0, "unresolved")
    assert _score(_task(), _outcome(_report(results)), partial=True) == (0.5, "partial")


def test_a_test_missing_from_the_report_does_not_pass() -> None:
    results = {F2P[0]: "passed", P2P[0]: "passed"}
    assert _score(_task(), _outcome(_report(results))) == (0.0, "unresolved")


@pytest.mark.parametrize(
    "meta",
    [{}, _report(ALL_PASS, status="error"), {"swe_status": "ok", "swe_tests": "not json"},
     {"swe_status": "ok", "swe_tests": "[1, 2]"}],
)  # fmt: skip
def test_a_suite_that_did_not_run_abstains(meta: dict[str, str]) -> None:
    assert _score(_task(), _outcome(meta)) == (None, "swe_tests_did_not_run")


def test_broken_tasks_abstain() -> None:
    assert _score(_task(f2p=[]), _outcome(_report(ALL_PASS))) == (None, "no_fail_to_pass")
    assert _score(_task(f2p={"a": 1}), _outcome(_report(ALL_PASS))) == (None, "bad_task_test_lists")
    bad = EpisodeTask(
        uid="t", session_id="t#0", messages=({"role": "user", "content": "fix"},),
        metadata={"fail_to_pass": "not json", "pass_to_pass": "[]"},
    )  # fmt: skip
    assert _score(bad, _outcome(_report(ALL_PASS))) == (None, "bad_task_test_lists")


@pytest.mark.parametrize("status", ["error", "timeout"])
def test_tampering_that_breaks_the_test_run_still_scores_zero(status: str) -> None:
    # An edited conftest.py that makes the run error out must score 0.0 (tampering), never
    # abstain as "tests did not run" -- that would let the agent escape the penalty.
    meta = {
        "swe_status": status,
        "swe_tests": "",
        "swe_guard": "measured",
        "swe_tampered": json.dumps(["/repo/conftest.py"]),
    }
    verdict = asyncio.run(SweReward(tamper="zero").score(_task(), _outcome(meta)))
    assert (verdict.value, verdict.reason) == (0.0, "test_harness_tampering")


# --- C2 format gate -------------------------------------------------------------------------
def _with_calls(meta: dict[str, str], parse_errors: list[str | None]) -> EpisodeOutcome:
    from foundationscale.agentic_rl.contracts import SegmentKind, ToolCall, Turn

    calls = tuple(
        ToolCall(name="bash", arguments='{"command": "ls"}', call_id=None, parse_error=e)
        for e in parse_errors
    )
    turn = Turn(index=0, kind=SegmentKind.ASSISTANT, token_ids=(1, 2), generated=True,
                logprobs=None, tool_calls=calls)  # fmt: skip
    return EpisodeOutcome(
        trajectory_turns=(turn,), prompt_turns=(), termination=Termination.STOP,
        submitted_answer="done", abstention_reason=None, metadata=meta,
    )  # fmt: skip


HALF = {F2P[0]: "passed", F2P[1]: "failed", P2P[0]: "passed"}


def _gated(outcome: EpisodeOutcome) -> tuple:
    verdict = asyncio.run(SweReward(partial_credit=True, format_gate=True).score(_task(), outcome))
    return verdict.value, verdict.reason


def test_format_gate_withholds_partial_credit_from_a_malformed_trace() -> None:
    assert _gated(_with_calls(_report(HALF), [None, "malformed tool_call JSON"])) == (0.0, "format_gate")


def test_format_gate_keeps_partial_credit_for_a_clean_trace() -> None:
    assert _gated(_with_calls(_report(HALF), [None, None])) == (0.5, "partial")


def test_format_gate_never_takes_a_verified_resolve() -> None:
    assert _gated(_with_calls(_report(ALL_PASS), ["bad"])) == (1.0, "resolved")


def test_format_gate_off_by_default() -> None:
    verdict = asyncio.run(SweReward(partial_credit=True).score(_task(), _with_calls(_report(HALF), ["bad"])))
    assert (verdict.value, verdict.reason) == (0.5, "partial")
