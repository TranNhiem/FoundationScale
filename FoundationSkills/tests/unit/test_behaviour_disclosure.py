"""A1.4: the behaviour eval models progressive disclosure for skills that ship references/*.md."""

from __future__ import annotations

import json
import re

import pytest
from foundationskills.agent import behaviour_eval
from foundationskills.agent.behaviour_eval import AGENT_SYSTEM, references_index, run_behaviour_eval
from foundationskills.agent.routing_eval import load_index
from foundationskills.skills.data_engine.llm_backend import LLMResponse

REFS = {
    "references/rules.md": "# Rule catalogue\nAR-XX-001 fires when the ledger is broken.",
    "references/examples.md": "# Worked examples\nexample body",
}
CREATED = "2026-10-16T00:00:00Z"


class ScriptedAgent:
    """Replies from a script keyed by how many turns this conversation has had."""

    kind, model = "fake-agent", "fake-agent-1"

    def __init__(self, script):
        self.script = script
        self.conversations: list[list[dict]] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, seed=0, json_mode=False):  # noqa: ARG002
        self.conversations.append(list(messages))
        return LLMResponse(self.script(messages), "stop", 0, 7, False, None)


class AllMetJudge:
    kind, model = "fake-judge", "fake-judge-1"

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, seed=0, json_mode=False):  # noqa: ARG002
        user = messages[-1]["content"]
        section = user.split("EXPECTED BEHAVIOURS", 1)[1].split("AGENT ANSWER:", 1)[0]
        n = len([ln for ln in section.splitlines() if re.match(r"^\d+\.", ln)])
        items = [{"i": i, "met": True, "why": "ok"} for i in range(n)]
        return LLMResponse(json.dumps({"items": items}), "stop", 0, 5, False, None)


@pytest.fixture()
def entry(monkeypatch: pytest.MonkeyPatch) -> dict:
    base = dict(load_index()[0])
    base["evals"] = [
        {
            "id": "d-1",
            "question": "q",
            "expected_skill": None,
            "should_trigger": True,
            "ground_truth": "gt",
            "expected_behavior": ["names the rule"],
        }
    ]
    monkeypatch.setattr(behaviour_eval, "_load_references", lambda _e: dict(REFS))
    return base


def run(entry: dict, agent: ScriptedAgent) -> dict:
    return run_behaviour_eval(
        [entry],
        agent,
        AllMetJudge(),
        arms=("with_skill", "without_skill"),
        reps=1,
        workers=1,
        threshold=0.7,
        created=CREATED,
    )


def test_index_lists_each_file_with_its_heading_and_the_read_protocol() -> None:
    text = references_index(REFS)
    assert (
        "- references/examples.md: Worked examples" in text
        and "- references/rules.md: Rule catalogue" in text
    )
    assert "READ references/<file>.md" in text and references_index({}) == ""


def test_gated_agent_reads_a_reference_then_answers(entry: dict) -> None:
    def script(messages):
        return "READ references/rules.md" if len(messages) == 2 else "final plan citing AR-XX-001"

    agent = ScriptedAgent(script)
    report = run(entry, agent)
    gated = [c for c in report["calls"] if c["arm"] == "with_skill"][0]
    blind = [c for c in report["calls"] if c["arm"] == "without_skill"][0]
    assert gated["references_read"] == ["references/rules.md"]
    assert gated["answer_excerpt"].startswith(
        "final plan"
    )  # the graded answer is the post-read one
    assert "references_read" not in blind  # the blind arm never sees the index
    gated_convo = next(
        c for c in agent.conversations if c[0]["content"] != AGENT_SYSTEM and len(c) == 4
    )
    assert "Reference files for this skill" in gated_convo[0]["content"]
    assert "AR-XX-001 fires" in gated_convo[-1]["content"]  # the file body was supplied
    assert all(
        c[0]["content"] == AGENT_SYSTEM
        for c in agent.conversations
        if "Reference files" not in c[0]["content"]
    )


def test_read_budget_is_bounded_and_unknown_files_are_answered(entry: dict) -> None:
    agent = ScriptedAgent(lambda _messages: "READ references/nope.md")
    report = run(entry, agent)
    gated = [c for c in report["calls"] if c["arm"] == "with_skill"][0]
    assert gated["references_read"] == ["references/nope.md"] * behaviour_eval._MAX_READS
    longest = max(agent.conversations, key=len)
    assert "No such reference file" in longest[-1]["content"]


def test_skills_without_references_keep_the_exact_old_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = dict(load_index()[0])
    base["evals"] = [
        {
            "id": "d-2",
            "question": "q",
            "expected_skill": None,
            "should_trigger": True,
            "ground_truth": "gt",
            "expected_behavior": ["x"],
        }
    ]
    monkeypatch.setattr(behaviour_eval, "_load_references", lambda _e: {})
    agent = ScriptedAgent(lambda _messages: "READ references/rules.md")
    report = run(base, agent)
    assert all(len(c) == 2 for c in agent.conversations)  # one call per arm: no READ loop
    assert all("references_read" not in c for c in report["calls"])
    assert all("Reference files" not in c[0]["content"] for c in agent.conversations)


def test_read_request_accepts_batches_and_rejects_prose() -> None:
    # the measured failure: agents batch their requests, 2-3 READ lines in one reply
    batch = "READ references/rules.md\nREAD references/examples.md"
    assert behaviour_eval.read_request(batch) == ["references/rules.md", "references/examples.md"]
    assert behaviour_eval.read_request("`READ references/rules.md`") == ["references/rules.md"]
    assert behaviour_eval.read_request("READ references/rules.md\nThen I would launch.") == []
    assert behaviour_eval.read_request("") == [] and behaviour_eval.read_request(None) == []


def test_a_batched_request_is_served_in_one_turn_within_the_budget(entry: dict) -> None:
    many = "\n".join(f"READ references/{n}.md" for n in ("rules", "examples", "rules", "examples"))

    def script(messages):
        return many if len(messages) == 2 else "final plan citing AR-XX-001"

    agent = ScriptedAgent(script)
    report = run(entry, agent)
    gated = [c for c in report["calls"] if c["arm"] == "with_skill"][0]
    assert len(gated["references_read"]) == behaviour_eval._MAX_READS  # 4 asked, budget 3
    assert gated["answer_excerpt"].startswith("final plan")
    served = next(c for c in agent.conversations if len(c) == 4)[-1]["content"]
    assert "=== references/rules.md ===" in served and "AR-XX-001 fires" in served
    assert "read budget" in served
