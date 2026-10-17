"""Test-based reward for software-engineering agent RL.

A task names the tests the fix must make pass (``task.metadata["fail_to_pass"]``) and the tests that
already pass and must keep passing (``task.metadata["pass_to_pass"]``), both JSON lists of test ids.
After the episode, a verifiable environment runs the hidden tests against the agent's edits and
reports per-test results in ``outcome.metadata``: ``swe_status`` (``"ok"`` when the suite ran) and
``swe_tests`` (a JSON object ``{test_id: "passed" | "failed" | "error" | "skipped"}``).

:class:`SweReward` scores that report:

- suite did not run (no report, bad JSON, status not ``ok``): abstain (``swe_tests_did_not_run``);
- a ``pass_to_pass`` test no longer passes: 0.0 (``regression``);
- every ``fail_to_pass`` test passes: 1.0 (``resolved``);
- otherwise 0.0 (``unresolved``), or with ``partial_credit=True`` the fraction of ``fail_to_pass``
  tests passing (``partial``).

A test missing from the report counts as not passing. A task with an empty ``fail_to_pass`` list is
a broken task and abstains (``no_fail_to_pass``) rather than scoring every attempt as resolved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from foundationscale.agentic_rl.harness.base import EpisodeOutcome, EpisodeTask
from foundationscale.agentic_rl.rewards.base import RewardVerdict


def _test_ids(raw: Any) -> list[str] | None:
    """A JSON list of test ids (or an already-decoded list) as strings; None if malformed."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
        return None
    return list(raw)


@dataclass(frozen=True)
class SweReward:
    """Scores an episode by its hidden-test report (see the module docstring)."""

    partial_credit: bool = False

    async def score(self, task: EpisodeTask, outcome: EpisodeOutcome) -> RewardVerdict:
        meta = task.metadata or {}
        fail_to_pass = _test_ids(meta.get("fail_to_pass", "[]"))
        pass_to_pass = _test_ids(meta.get("pass_to_pass", "[]"))
        if fail_to_pass is None or pass_to_pass is None:
            return RewardVerdict(None, "bad_task_test_lists")
        if not fail_to_pass:
            return RewardVerdict(None, "no_fail_to_pass")
        if outcome.metadata.get("swe_status") != "ok":
            return RewardVerdict(None, "swe_tests_did_not_run")
        try:
            results = json.loads(outcome.metadata.get("swe_tests", ""))
        except ValueError:
            return RewardVerdict(None, "swe_tests_did_not_run")
        if not isinstance(results, dict):
            return RewardVerdict(None, "swe_tests_did_not_run")

        def passed(test_id: str) -> bool:
            return results.get(test_id) == "passed"

        if not all(passed(t) for t in pass_to_pass):
            return RewardVerdict(0.0, "regression")
        fixed = sum(passed(t) for t in fail_to_pass)
        if fixed == len(fail_to_pass):
            return RewardVerdict(1.0, "resolved")
        if self.partial_credit:
            return RewardVerdict(fixed / len(fail_to_pass), "partial")
        return RewardVerdict(0.0, "unresolved")
