"""A judge that spends its whole budget answers nothing; the eval re-asks once with more budget."""

from __future__ import annotations

import json

from foundationskills.agent import behaviour_eval
from foundationskills.agent.behaviour_eval import run_behaviour_eval
from foundationskills.agent.routing_eval import load_index
from foundationskills.skills.data_engine.llm_backend import LLMResponse


class Agent:
    kind, model = "fake-agent", "fake-agent-1"

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, seed=0, json_mode=False):  # noqa: ARG002
        return LLMResponse("a concrete plan", "stop", 0, 7, False, None)


class Judge:
    """Empty (truncated or not) until the budget reaches `answers_at`; records each budget."""

    kind, model = "fake-judge", "fake-judge-1"

    def __init__(self, answers_at: int, truncated: bool = True) -> None:
        self.answers_at, self.truncated, self.budgets = answers_at, truncated, []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, seed=0, json_mode=False):  # noqa: ARG002
        self.budgets.append(max_tokens)
        if max_tokens < self.answers_at:
            used = max_tokens if self.truncated else 10
            return LLMResponse(None, "length" if self.truncated else "stop", 0, used, False, None)
        return LLMResponse(
            json.dumps({"items": [{"i": 0, "met": True, "why": "ok"}]}), "stop", 0, 9, False, None
        )


def entry() -> dict:
    base = dict(next(e for e in load_index() if e["skill_bundle_sha256"] == e["skill_md_sha256"]))
    base["evals"] = [
        {
            "id": "j-1",
            "question": "q",
            "expected_skill": None,
            "should_trigger": True,
            "ground_truth": "gt",
            "expected_behavior": ["names the rule"],
        }
    ]
    return base


def run(judge: Judge) -> dict:
    report = run_behaviour_eval(
        [entry()],
        Agent(),
        judge,
        arms=("with_skill", "without_skill"),
        reps=1,
        workers=1,
        threshold=0.7,
        created="2026-10-17T00:00:00Z",
        judge_max_tokens=4096,
    )
    return next(c for c in report["calls"] if c["arm"] == "with_skill")


def test_a_truncated_empty_judgement_is_reasked_with_a_doubled_budget() -> None:
    judge = Judge(answers_at=8192)
    call = run(judge)
    assert call["measured"] is True and call["judge_budget_escalated"] == 8192
    assert 8192 in judge.budgets and max(judge.budgets) <= behaviour_eval._JUDGE_BUDGET_CAP


def test_an_empty_judgement_that_was_not_truncated_is_not_escalated() -> None:
    call = run(Judge(answers_at=8192, truncated=False))
    assert call["measured"] is False and "judge_budget_escalated" not in call


def test_a_normal_judgement_never_escalates() -> None:
    judge = Judge(answers_at=0)
    call = run(judge)
    assert (
        call["measured"] is True
        and "judge_budget_escalated" not in call
        and set(judge.budgets) == {4096}
    )


def test_an_agent_reply_truncated_empty_is_reasked_once_with_a_larger_budget() -> None:
    from foundationskills.agent.behaviour_eval import _AGENT_BUDGET_CAP, _escalated_budget
    from foundationskills.skills.data_engine.llm_backend import LLMResponse

    empty = LLMResponse(None, "length", 0, 16384, False, None)
    assert _escalated_budget(empty, 16384, _AGENT_BUDGET_CAP) == 32768
    assert _escalated_budget(empty, 32768, _AGENT_BUDGET_CAP) is None  # already at the cap: stays unmeasured
    assert _escalated_budget(LLMResponse("ok", "stop", 10, 0, False, None), 16384, _AGENT_BUDGET_CAP) is None
