"""Unit tests for foundationskills.agent.behaviour_eval (no network, no real LLM).

Plan-level behaviour is exercised against fake agent/judge backends: the agent echoes one
canned plan and the judge replies from a scripted JSON payload. The index (and every
SKILL.md hash) comes from the real skill packages through ``load_index`` so the stale-sha
guard and the prompt assembly run against the shipped files, while the case items are
synthesised to keep the arithmetic exact. Doctrine under test: exit 0 PASS, 5 RED,
95 UNMEASURED (a call that did not run is never PASS), 96 REFUSED (refusal raises
``BehaviourEvalRefused`` and writes no report).
"""
from __future__ import annotations

import importlib.resources
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from foundationskills.agent import behaviour_eval, oracle_cases
from foundationskills.agent.behaviour_eval import (
    AGENT_SYSTEM,
    ARMS,
    BehaviourEvalRefused,
    render_behaviour_md,
    run_behaviour_eval,
    write_behaviour,
    write_report,
)
from foundationskills.agent.routing_eval import load_index
from foundationskills.skills.data_engine.llm_backend import LLMResponse

CREATED = "2026-10-02T00:00:00Z"
ANSWER = "Plan: build the loader, write the artefact, validate the schema, refuse a missing column."
HEX64 = re.compile(r"[0-9a-f]{64}")


class FakeAgent:
    """Deterministic agent with kind="fake-agent", model="fake-agent-1"; records every call."""

    def __init__(
        self,
        answer: str = ANSWER,
        *,
        stamp: Callable[[str], str] | None = None,
        fail_when: Callable[[dict], str | None] | None = None,
        finish_reason: str = "stop",
        model: str = "fake-agent-1",
    ) -> None:
        self.kind = "fake-agent"
        self.model = model
        self.answer = answer
        self.stamp = stamp
        self.fail_when = fail_when if fail_when is not None else (lambda call: None)
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 0,
        json_mode: bool = False,
    ) -> LLMResponse:
        call = {
            "index": len(self.calls),
            "system": messages[0]["content"],
            "user": messages[-1]["content"],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "json_mode": json_mode,
        }
        self.calls.append(call)
        error = self.fail_when(call)
        if error is not None:
            return LLMResponse(None, "stop", 0, 0, False, error)
        content = self.answer
        if self.stamp is not None:
            content += "\n" + self.stamp(call["system"])
        return LLMResponse(content, self.finish_reason, 0, 7, False, None)


class FakeJudge:
    """Deterministic judge with kind="fake-judge", model="fake-judge-1"; scripted payloads."""

    def __init__(
        self,
        payload: Callable[[str, str], object] | None = None,
        *,
        fail_when: Callable[[dict], str | None] | None = None,
        model: str = "fake-judge-1",
    ) -> None:
        self.kind = "fake-judge"
        self.model = model
        self.payload = payload if payload is not None else all_met
        self.fail_when = fail_when if fail_when is not None else (lambda call: None)
        self.calls: list[dict] = []

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 0,
        json_mode: bool = False,
    ) -> LLMResponse:
        call = {
            "index": len(self.calls),
            "system": messages[0]["content"],
            "user": messages[-1]["content"],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "json_mode": json_mode,
        }
        self.calls.append(call)
        error = self.fail_when(call)
        if error is not None:
            return LLMResponse(None, "stop", 0, 0, False, error)
        payload = self.payload(call["user"], call["system"])
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(content, "stop", 0, 5, False, None)


def numbered_behaviours(user: str) -> list[str]:
    """The numbered expected behaviours embedded in a judge user message."""

    section = user.split("EXPECTED BEHAVIOURS", 1)[1].split("AGENT ANSWER:", 1)[0]
    return [re.sub(r"^\d+\.\s*", "", line) for line in section.splitlines() if re.match(r"^\d+\.", line)]


def all_met(user: str, _system: str) -> dict:
    """Judge script: every numbered behaviour of the prompt is met."""

    return {"items": [{"i": i, "met": True, "why": "stated"} for i in range(len(numbered_behaviours(user)))]}


def scripted(rule: Callable[[int, str, str], bool]) -> Callable[[str, str], dict]:
    """Judge script builder: rule(index, item text, user message) decides `met`."""

    def script(user: str, _system: str) -> dict:
        items = numbered_behaviours(user)
        return {
            "items": [{"i": i, "met": bool(rule(i, text, user)), "why": "graded"} for i, text in enumerate(items)],
        }

    return script


def arm_stamp(system: str) -> str:
    """Stamp recording which arm the fake agent observed in its system prompt."""

    return "[skill-loaded]" if system != AGENT_SYSTEM else "[skill-blind]"


def make_case(case_id: str, behaviours: list[str]) -> dict:
    return {
        "id": case_id,
        "question": f"question for {case_id}",
        "expected_skill": None,
        "should_trigger": True,
        "ground_truth": f"ground truth for {case_id}",
        "expected_behavior": list(behaviours),
    }


def make_entry(index: list[dict], cases: list[dict]) -> dict:
    entry = dict(index[0])
    entry["evals"] = [dict(case) for case in cases]
    return entry


@pytest.fixture(scope="session")
def index() -> list[dict]:
    return load_index()


@pytest.fixture(scope="session")
def skill_md(index: list[dict]) -> str:
    return importlib.resources.files(str(index[0]["package"])).joinpath("SKILL.md").read_text(encoding="utf-8")


def test_load_index_reads_real_packages(index: list[dict]) -> None:
    assert len(index) == 4  # training, evaluation, data_engine, auto_research
    for entry in index:
        assert entry["package"] and entry["name"]
        assert HEX64.fullmatch(str(entry["skill_md_sha256"])) is not None
        assert HEX64.fullmatch(str(entry["evals_sha256"])) is not None
        assert isinstance(entry["evals"], list) and entry["evals"]


def test_process_only_items_are_ungraded_and_leave_the_denominator(index: list[dict]) -> None:
    entry = make_entry(index, [
        make_case("be-1", [
            "read SKILL.md before acting",
            "loaded the skill index first",
            "before acting, plan the run",
            "writes output/ladder.parquet with the data engine skill",
        ]),
    ])
    judge = FakeJudge()
    report = run_behaviour_eval(
        [entry], FakeAgent(), judge, arms=("with_skill",), reps=1, workers=1, created=CREATED,
    )
    call = report["calls"][0]
    assert call["ungraded_items"] == 3
    assert call["items_graded"] == 1 and call["items_met"] == 1
    assert call["met"] == [True]
    assert call["item_texts"] == ["writes output/ladder.parquet with the data engine skill"]
    user = judge.calls[0]["user"]
    assert "0. writes output/ladder.parquet with the data engine skill" in user
    assert "read SKILL.md before acting" not in user
    assert "loaded the skill index first" not in user
    assert "before acting, plan the run" not in user
    score = report["results"]["with_skill"]
    assert score["item_pass_rate"] == 1.0 and score["case_pass_rate"] == 1.0
    assert report["index"][0]["positives"] == 1
    assert report["plan_level"] is True


def test_with_skill_and_without_skill_build_the_right_system_prompts(
    index: list[dict], skill_md: str,
) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes output/ladder.parquet"])])
    agent = FakeAgent()
    judge = FakeJudge()
    report = run_behaviour_eval(
        [entry], agent, judge, arms=ARMS, reps=1, workers=1, created=CREATED,
    )
    with_skill = [call["system"] for call in agent.calls if skill_md in call["system"]]
    without_skill = [call["system"] for call in agent.calls if skill_md not in call["system"]]
    assert without_skill == [AGENT_SYSTEM]
    assert len(with_skill) == 1 and with_skill[0].startswith(AGENT_SYSTEM)
    assert all(call["user"] == "question for be-1" for call in agent.calls)
    assert {call["arm"] for call in report["calls"]} == set(ARMS)
    assert all(call["json_mode"] is False for call in agent.calls)
    assert all(call["temperature"] == 0.6 and call["max_tokens"] == 4096 for call in agent.calls)
    user = judge.calls[0]["user"]
    assert len(judge.calls) == 2  # one judge call per agent answer: 2 arms x 1 case x 1 rep
    assert judge.calls[0]["json_mode"] is True
    assert judge.calls[0]["temperature"] == 0.0 and judge.calls[0]["seed"] == 0
    assert judge.calls[0]["max_tokens"] == 2048
    assert "question for be-1" in user and "ground truth for be-1" in user
    assert "0. writes output/ladder.parquet" in user and ANSWER in user


def test_scoring_and_uplift_arithmetic(index: list[dict]) -> None:
    entry = make_entry(index, [
        make_case("be-1", ["always: writes the artefact", "hard: refuses the missing column"]),
        make_case("be-2", ["always: names the CLI call", "hard: applies the validation rule"]),
    ])
    agent = FakeAgent(stamp=arm_stamp)
    judge = FakeJudge(scripted(lambda _i, text, user: text.startswith("always") or "[skill-loaded]" in user))
    report = run_behaviour_eval(
        [entry], agent, judge, arms=ARMS, reps=2, workers=1, threshold=0.7, created=CREATED,
    )
    assert report["schema"] == "fskills.behaviour_eval/1"
    assert report["created"] == CREATED and report["gated_arm"] == "with_skill"
    assert report["agent_model"] == "fake-agent-1" and report["judge_model"] == "fake-judge-1"
    assert report["backend_kind"] == {"agent": "fake-agent", "judge": "fake-judge"}
    loaded, blind = report["results"]["with_skill"], report["results"]["without_skill"]
    assert (loaded["calls"], loaded["measured"], loaded["unmeasured"]) == (4, 4, 0)
    assert loaded["item_pass_rate"] == 1.0 and loaded["case_pass_rate"] == 1.0
    assert (blind["calls"], blind["measured"], blind["unmeasured"]) == (4, 4, 0)
    assert blind["item_pass_rate"] == 0.5 and blind["case_pass_rate"] == 0.0
    for score in (loaded, blind):
        block = score["packages"][entry["name"]]
        assert block["item_pass_rate"] == score["item_pass_rate"]
        assert block["case_pass_rate"] == score["case_pass_rate"]
    assert report["uplift"]["item_pass_rate"] == 0.5
    assert report["uplift"]["packages"][entry["name"]] == 0.5
    calls = report["calls"]
    assert len(calls) == 8
    assert [call["met"] for call in calls if call["arm"] == "with_skill"] == [[True, True]] * 4
    assert [call["met"] for call in calls if call["arm"] == "without_skill"] == [[True, False]] * 4
    assert all(call["agent_tokens"] == 7 and call["judge_tokens"] == 5 for call in calls)
    assert all(len(call["answer_excerpt"]) <= 300 for call in calls)
    assert calls[0]["answer_excerpt"].startswith(ANSWER)
    assert report["verdict"] == "pass" and report["exit_code"] == 0 and report["reasons"] == []


def test_unmeasured_on_agent_error_judge_garbage_and_missing_index(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact", "refuses the missing column"])])
    report = run_behaviour_eval(
        [entry], FakeAgent(fail_when=lambda _call: "HTTP 500"), FakeJudge(),
        arms=("with_skill",), reps=1, workers=1, created=CREATED,
    )
    call = report["calls"][0]
    assert call["measured"] is False and "500" in (call["error"] or "")
    assert call["answer_excerpt"] == "" and call["judge_tokens"] == 0
    assert report["verdict"] == "unmeasured" and report["exit_code"] == 95
    with pytest.raises(BehaviourEvalRefused):
        write_behaviour(report, [entry])
    report = run_behaviour_eval(
        [entry], FakeAgent(), FakeJudge(lambda user, system: "not json at all"),
        arms=("with_skill",), reps=1, workers=1, created=CREATED,
    )
    call = report["calls"][0]
    assert call["measured"] is False and "unparseable" in (call["error"] or "").lower()
    assert call["answer_excerpt"].startswith(ANSWER)
    assert report["verdict"] == "unmeasured"
    missing = FakeJudge(lambda user, system: {"items": [{"i": 1, "met": True, "why": "ok"}]})
    report = run_behaviour_eval(
        [entry], FakeAgent(), missing, arms=("with_skill",), reps=1, workers=1, created=CREATED,
    )
    assert report["calls"][0]["measured"] is False and "missing" in (report["calls"][0]["error"] or "")
    assert report["verdict"] == "unmeasured" and report["exit_code"] == 95


def test_non_bool_met_is_unmeasured_but_truncated_answer_is_graded(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact", "refuses the missing column"])])
    judge = FakeJudge(lambda user, system: {"items": [{"i": 0, "met": True}, {"i": 1, "met": "yes"}]})
    report = run_behaviour_eval(
        [entry], FakeAgent(), judge, arms=("with_skill",), reps=1, workers=1, created=CREATED,
    )
    assert report["calls"][0]["measured"] is False and "bool" in (report["calls"][0]["error"] or "")
    report = run_behaviour_eval(
        [entry], FakeAgent(finish_reason="length"), FakeJudge(),
        arms=("with_skill",), reps=1, workers=1, created=CREATED,
    )
    call = report["calls"][0]
    assert call["measured"] is True and call["met"] == [True, True]


def test_red_when_the_gated_arm_is_below_the_threshold(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["always: writes the artefact", "hard: refuses the column"])])
    judge = FakeJudge(scripted(lambda _i, text, _user: text.startswith("always")))
    report = run_behaviour_eval(
        [entry], FakeAgent(), judge, arms=ARMS, reps=1, workers=1, threshold=0.7, created=CREATED,
    )
    assert report["results"]["with_skill"]["item_pass_rate"] == 0.5
    assert report["verdict"] == "red" and report["exit_code"] == 5
    assert any("with_skill" in reason and "item_pass_rate" in reason for reason in report["reasons"])
    assert any(entry["name"] in reason and "item_pass_rate" in reason for reason in report["reasons"])


def test_self_grading_is_refused(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact"])])
    shared = FakeAgent()
    with pytest.raises(BehaviourEvalRefused) as excinfo:
        run_behaviour_eval([entry], shared, shared, reps=1, workers=1, created=CREATED)
    assert "self-grading" in str(excinfo.value)
    twin = FakeJudge(model=shared.model)
    with pytest.raises(BehaviourEvalRefused) as same_model:
        run_behaviour_eval([entry], shared, twin, reps=1, workers=1, created=CREATED)
    assert "self-grading" in str(same_model.value)


def test_stale_skill_md_sha_is_refused(index: list[dict]) -> None:
    entry = dict(make_entry(index, [make_case("be-1", ["writes the artefact"])]), skill_md_sha256="0" * 64)
    with pytest.raises(BehaviourEvalRefused) as excinfo:
        run_behaviour_eval([entry], FakeAgent(), FakeJudge(), reps=1, workers=1, created=CREATED)
    assert "stale" in str(excinfo.value) and str(entry["package"]) in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reps": 0},
        {"workers": 0},
        {"threshold": 0.0},
        {"threshold": 1.0001},
        {"arms": ("without_skill",)},
        {"arms": ("with_skill", "not-an-arm")},
    ],
    ids=["no-reps", "no-workers", "zero-threshold", "threshold-above-one", "gate-missing", "unknown-arm"],
)
def test_bad_inputs_are_refused(index: list[dict], kwargs: dict) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact"])])
    with pytest.raises(BehaviourEvalRefused):
        run_behaviour_eval([entry], FakeAgent(), FakeJudge(), **kwargs)


def test_render_behaviour_md_header_line(index: list[dict], tmp_path: Path) -> None:
    entry = make_entry(index, [make_case("be-1", ["always: writes the artefact", "hard: refuses the column"])])
    report = run_behaviour_eval(
        [entry], FakeAgent(stamp=arm_stamp), FakeJudge(), arms=ARMS, reps=1, workers=1,
        threshold=0.7, created=CREATED,
    )
    md = render_behaviour_md(report, str(entry["name"]))
    generated = ("Generated by `fskills behaviour-eval` against SKILL.md "
                 f"{entry['skill_md_sha256'][:12]} and eval cases {entry['evals_sha256'][:12]}; do not edit by hand.")
    assert md.startswith(f"# Behaviour evaluation: {entry['name']}\n")
    assert generated in md
    assert "- Agent model: fake-agent-1" in md and "- Judge model: fake-judge-1" in md
    assert "| arm | item pass | case pass | measured |" in md
    assert "process-only items ungraded" in md
    assert "be-1" in md and f"{entry['name']}" in md
    path = tmp_path / "behaviour.json"
    write_report(report, path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["schema"] == "fskills.behaviour_eval/1"
    assert stored["verdict"] == report["verdict"]
    assert {"package", "name", "skill_md_sha256", "evals_sha256", "positives"} <= set(stored["index"][0])


def test_write_behaviour_refuses_an_unmeasured_report(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact"])])
    report = run_behaviour_eval(
        [entry], FakeAgent(fail_when=lambda _call: "HTTP 500"), FakeJudge(),
        arms=("with_skill",), reps=1, workers=1, created=CREATED,
    )
    assert report["verdict"] == "unmeasured"
    with pytest.raises(BehaviourEvalRefused):
        write_behaviour(report, [entry])


def test_write_behaviour_writes_behaviour_md_next_to_each_skill_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: list[dict],
) -> None:
    entry = make_entry(index, [make_case("be-1", ["always: writes the artefact"])])
    report = run_behaviour_eval(
        [entry], FakeAgent(), FakeJudge(), arms=ARMS, reps=1, workers=1, threshold=0.7, created=CREATED,
    )
    assert report["verdict"] == "pass"
    seen: list[str] = []

    def fake_package_dir(package: str) -> Path:
        seen.append(str(package))
        (tmp_path / str(package)).mkdir()
        return tmp_path / str(package)

    monkeypatch.setattr(behaviour_eval, "_package_dir", fake_package_dir)
    paths = write_behaviour(report, [entry])
    assert [path.name for path in paths] == ["BEHAVIOUR.md"]
    assert seen == [str(entry["package"])]
    assert paths[0].exists()
    text = paths[0].read_text(encoding="utf-8")
    assert text.startswith(f"# Behaviour evaluation: {entry['name']}")
    assert "do not edit by hand" in text


def test_cli_run_refuses_a_missing_key_env_without_writing_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from foundationskills import cli

    monkeypatch.delenv("FSKILLS_NO_SUCH_KEY", raising=False)
    out = tmp_path / "behaviour.json"
    rc = cli.main([
        "behaviour-eval", "run",
        "--agent-base-url", "http://127.0.0.1:9", "--agent-model", "m",
        "--agent-api-key-env", "FSKILLS_NO_SUCH_KEY",
        "--judge-base-url", "http://127.0.0.1:9", "--judge-model", "j",
        "--out", str(out),
    ])
    assert rc == 96 and not out.exists()
    assert "FSKILLS_NO_SUCH_KEY" in capsys.readouterr().err


def test_cli_render_refuses_an_unmeasured_report(tmp_path: Path, index: list[dict]) -> None:
    from foundationskills import cli

    entry = make_entry(index, [make_case("be-1", ["writes the artefact"])])
    report = run_behaviour_eval(
        [entry], FakeAgent(fail_when=lambda _call: "HTTP 500"), FakeJudge(),
        arms=("with_skill",), reps=1, workers=1, created=CREATED,
    )
    path = tmp_path / "behaviour.json"
    write_report(report, path)
    assert cli.main(["behaviour-eval", "render", "--report", str(path)]) == 96


def test_a_failed_agent_call_is_retried_once_under_another_seed(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact"])])
    agent = FakeAgent(fail_when=lambda call: "empty response content" if call["seed"] < 1000 else None)
    report = run_behaviour_eval([entry], agent, FakeJudge(), arms=("with_skill",), reps=1, workers=1,
                                created=CREATED)
    (call,) = report["calls"]
    assert call["measured"] is True and call["agent_retried"] is True
    assert [c["seed"] for c in agent.calls] == [0, 1000]
    assert call["answer"].startswith(ANSWER[:20])


def test_a_second_failure_stays_unmeasured(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact"])])
    agent = FakeAgent(fail_when=lambda call: "empty response content")
    report = run_behaviour_eval([entry], agent, FakeJudge(), arms=("with_skill",), reps=1, workers=1,
                                created=CREATED)
    assert report["calls"][0]["measured"] is False and len(agent.calls) == 2
    assert report["verdict"] == "unmeasured" and report["exit_code"] == 95


"""Oracle case-source tests, appended to tests/unit/test_behaviour_eval.py.

The fake payload stands in for ``oracle_cases.load_cases``: two code-grounded cases (an input BLOCK and a
handoff WARN) with their must-fire fixture, ``expected_behavior``, ``ground_truth`` and LLM-built question.
Doctrine under test: the oracle source grades every stored case, the report records ``cases: oracle`` and the
behaviour.json sha per index view, ``write_behaviour`` writes ``BEHAVIOUR_ORACLE.md`` whose header says
"oracle cases" and whose gated arm grows the per-severity table; an unknown ``cases`` value is refused.
"""

_FAKE_CASES_SHA = "c" * 64

QUESTION_MISSING_COLUMN = (
    "Please turn stages/ladder.parquet into out/ladder.parquet keeping only the columns step and wall_time, "
    "as a parquet table with a monotone int64 step key. The run log at logs/run.json is the source of truth "
    "for which steps finished, so keep every staged row that has a step value. Write out/report.json next to "
    "the table when you are done."
)

QUESTION_SCHEMA_DRIFT = (
    "Here is what I have so far: out/ladder.parquet was produced by the export step with columns step and "
    "wall_time, while out/manifest.json lists step, wall_time and peak_mem. Please hand the result on to the "
    "eval stage: finalise the manifest, record the artifacts and tell me what you would put in the summary. "
    "The eval stage only consumes parquet plus a manifest."
)


def fake_oracle_cases() -> list[dict]:
    """A fresh behaviour.json payload: one input BLOCK rule, one handoff WARN rule."""
    return [
        {
            "id": "demo-oracle-DAT-COL",
            "skill": "data_engine",
            "rule_id": "DAT-COL",
            "severity": "BLOCK",
            "phase": "input",
            "description": "the requested column is not present in the staged table",
            "fixture": {"table": "stages/ladder.parquet", "keep": ["step"], "read": {}},
            "fixture_sha256": "f" * 64,
            "expected_behavior": [
                "Identifies the problem: the requested column is not present in the staged table",
                "Refuses before doing any work (status REFUSED, nothing written) instead of running",
                "Cites rule DAT-COL",
            ],
            "ground_truth": "data_engine: rule DAT-COL (BLOCK, input phase) fires: "
                            "the requested column is not present in the staged table.",
            "question": QUESTION_MISSING_COLUMN,
            "question_model": "fake-question-1",
        },
        {
            "id": "demo-oracle-DAT-DRIFT",
            "skill": "data_engine",
            "rule_id": "DAT-DRIFT",
            "severity": "WARN",
            "phase": "handoff",
            "description": "the produced schema drifted from the declared output",
            "fixture": {"produced": "out/ladder.parquet", "declared": ["step"], "read": {}},
            "fixture_sha256": "e" * 64,
            "expected_behavior": [
                "Identifies the problem: the produced schema drifted from the declared output",
                "Reports it as a warning that travels with the result, without refusing on this alone",
                "Cites rule DAT-DRIFT",
            ],
            "ground_truth": "data_engine: rule DAT-DRIFT (WARN, handoff phase) fires: "
                            "the produced schema drifted from the declared output.",
            "question": QUESTION_SCHEMA_DRIFT,
            "question_model": "fake-question-1",
        },
    ]


def load_fake_oracle_cases(_package: str) -> tuple[list[dict], str]:
    """Stands in for ``oracle_cases.load_cases``: the fake payload and a fixed behaviour.json sha."""
    return fake_oracle_cases(), _FAKE_CASES_SHA


def test_oracle_case_source_grades_every_behaviour_json_case(
    monkeypatch: pytest.MonkeyPatch, index: list[dict],
) -> None:
    monkeypatch.setattr(behaviour_eval, "load_cases", load_fake_oracle_cases)
    entry = dict(index[0])
    agent, judge = FakeAgent(), FakeJudge()
    report = run_behaviour_eval(
        [entry], agent, judge, cases="oracle", arms=("with_skill",), reps=1, workers=1, created=CREATED,
    )
    cases = fake_oracle_cases()
    assert report["cases"] == "oracle"
    view = report["index"][0]
    assert view["cases_sha256"] == _FAKE_CASES_SHA and view["positives"] == 2
    assert view["evals_sha256"] == entry["evals_sha256"]
    assert [call["case_id"] for call in report["calls"]] == [case["id"] for case in cases]
    assert [call["severity"] for call in report["calls"]] == ["BLOCK", "WARN"]
    assert [call["phase"] for call in report["calls"]] == ["input", "handoff"]
    assert [call["user"] for call in agent.calls] == [case["question"] for case in cases]
    assert report["results"]["with_skill"]["item_pass_rate"] == 1.0


def test_oracle_report_renders_behaviour_oracle_md_with_a_per_severity_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: list[dict],
) -> None:
    monkeypatch.setattr(behaviour_eval, "load_cases", load_fake_oracle_cases)
    entry = dict(index[0])
    report = run_behaviour_eval(
        [entry], FakeAgent(), FakeJudge(), cases="oracle", arms=("with_skill",), reps=1, workers=1,
        threshold=0.7, created=CREATED,
    )
    md = render_behaviour_md(report, str(entry["name"]))
    generated = ("Generated by `fskills behaviour-eval` against SKILL.md "
                 f"{entry['skill_md_sha256'][:12]} and oracle cases {_FAKE_CASES_SHA[:12]}; do not edit by hand.")
    assert generated in md
    assert "| severity/phase | cases | item pass |" in md
    assert "| BLOCK/input | 1 | 100.0% |" in md
    assert "| WARN/handoff | 1 | 100.0% |" in md

    def fake_package_dir(package: str) -> Path:
        target = tmp_path / str(package)
        target.mkdir()
        return target

    monkeypatch.setattr(behaviour_eval, "_package_dir", fake_package_dir)
    (path,) = write_behaviour(report, [entry])
    assert path.name == "BEHAVIOUR_ORACLE.md"
    assert md == path.read_text(encoding="utf-8")


def test_routing_is_the_default_case_source_and_the_index_keeps_both_shas(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact"])])
    report = run_behaviour_eval([entry], FakeAgent(), FakeJudge(), arms=("with_skill",), reps=1, workers=1,
                                created=CREATED)
    assert report["cases"] == "routing"
    assert report["index"][0]["cases_sha256"] == entry["evals_sha256"]
    assert report["index"][0]["positives"] == 1


def test_unknown_case_source_is_refused(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact"])])
    with pytest.raises(BehaviourEvalRefused) as excinfo:
        run_behaviour_eval([entry], FakeAgent(), FakeJudge(), cases="widgets", reps=1, workers=1)
    assert "widgets" in str(excinfo.value)


def test_oracle_case_source_refuses_a_missing_behaviour_json(
    monkeypatch: pytest.MonkeyPatch, index: list[dict],
) -> None:
    def missing(_package: str) -> tuple[list[dict], str]:
        raise oracle_cases.OracleCasesRefused("load_cases: demo: evals/behaviour.json is missing")

    monkeypatch.setattr(behaviour_eval, "load_cases", missing)
    entry = dict(index[0])
    with pytest.raises(BehaviourEvalRefused) as excinfo:
        run_behaviour_eval([entry], FakeAgent(), FakeJudge(), cases="oracle", reps=1, workers=1, created=CREATED)
    assert str(entry["package"]) in str(excinfo.value) and "behaviour.json" in str(excinfo.value)


def test_resume_reuses_measured_calls_and_reasks_only_the_rest(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact"]), make_case("be-2", ["writes it"])])
    flaky = FakeAgent(fail_when=lambda call: "empty response content" if "be-2" in call["user"] else None)
    first = run_behaviour_eval([entry], flaky, FakeJudge(), arms=("with_skill",), reps=1, workers=1,
                               created=CREATED)
    assert first["verdict"] == "unmeasured"
    agent = FakeAgent()
    second = run_behaviour_eval([entry], agent, FakeJudge(), arms=("with_skill",), reps=1, workers=1,
                                created=CREATED, resume=first)
    assert second["reused_calls"] == 1 and len(agent.calls) == 1 and "be-2" in agent.calls[0]["user"]
    assert all(call["measured"] for call in second["calls"])


def test_resume_refuses_a_report_from_another_run(index: list[dict]) -> None:
    entry = make_entry(index, [make_case("be-1", ["writes the artefact"])])
    first = run_behaviour_eval([entry], FakeAgent(), FakeJudge(), arms=("with_skill",), reps=1, workers=1,
                               created=CREATED)
    with pytest.raises(BehaviourEvalRefused):
        run_behaviour_eval([entry], FakeAgent(model="other-agent"), FakeJudge(), arms=("with_skill",), reps=1,
                           workers=1, created=CREATED, resume=first)
    with pytest.raises(BehaviourEvalRefused):
        run_behaviour_eval([entry], FakeAgent(), FakeJudge(), arms=("with_skill",), reps=2, workers=1,
                           created=CREATED, resume=first)
