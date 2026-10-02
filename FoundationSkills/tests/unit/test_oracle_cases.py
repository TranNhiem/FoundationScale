"""Unit tests for the code-grounded ("oracle") behaviour cases."""
from __future__ import annotations

import hashlib
import json

import pytest

from foundationskills.agent import oracle_cases, routing_eval
from foundationskills.agent.oracle_cases import _question_flaw as check_cases_question
from foundationskills.agent.oracle_cases import (
    OracleCasesRefused,
    build_cases,
    check_cases,
    expected_items,
    ground_truth,
    load_cases,
    oracle_skeleton,
    question_messages,
    write_cases,
)
from foundationskills.core import SkillRegistry
from foundationskills.core.contract import RuleSpec, Severity
from foundationskills.skills import register_builtin_skills
from foundationskills.skills.data_engine.llm_backend import LLMResponse

GOOD_QUESTION = (
    "Please run the requested data pipeline over the staged files and write the summary report into the "
    "artifacts folder so that I can review the numbers afterwards."
)
LEAKY_WORD_QUESTION = "Please fix the reported rule problem in the staged payload and send me the report back."

_OUTCOMES = {
    ("input", Severity.BLOCK): (
        "Refuses before doing any work (status REFUSED, nothing written) instead of running"
    ),
    ("handoff", Severity.BLOCK): (
        "Does not report success: the result is RED (failed), neither PASS nor a refusal"
    ),
    ("input", Severity.WARN): (
        "Reports it as a warning that travels with the result, without refusing on this alone"
    ),
    ("handoff", Severity.WARN): (
        "Reports it as a warning that travels with the result, without refusing on this alone"
    ),
    ("input", Severity.INFO): ("Records it as information only; it neither blocks nor fails the run"),
    ("handoff", Severity.INFO): ("Records it as information only; it neither blocks nor fails the run"),
}


def _rule(
    severity: Severity,
    phase: str,
    rule_id: str = "TS-BAD",
    description: str = "a staged file is wrong for the task",
) -> RuleSpec:
    return RuleSpec(rule_id=rule_id, description=description, severity=severity, phase=phase)


def _case_jobs() -> list[tuple[str, dict]]:
    skeleton = oracle_skeleton()
    return [(package, dict(case)) for package in sorted(skeleton) for case in skeleton[package]]


def _patch_package_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        oracle_cases, "_package_dir", lambda package: tmp_path / str(package).replace(".", "_")
    )


class FakeBackend:
    """Deterministic test backend: scripted replies keyed by call index (str or callable)."""

    kind = "fake"
    model = "fake/question-model"

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, seed=0, json_mode=False):
        index = len(self.calls)
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "json_mode": json_mode,
            }
        )
        reply = self.replies[min(index, len(self.replies) - 1)]
        text = reply(index, messages, seed) if callable(reply) else reply
        return LLMResponse(
            content=text,
            finish_reason="stop",
            prompt_tokens=7,
            completion_tokens=13,
            cache_hit=False,
            error=None,
        )


# --- skeleton ---------------------------------------------------------------------------


def test_oracle_skeleton_covers_every_declared_rule_exactly_once() -> None:
    skeleton = oracle_skeleton()
    registry = SkillRegistry()
    register_builtin_skills(registry)
    declared: dict[tuple[str, str], tuple] = {}
    for key in sorted(registry.names()):
        skill = registry.get(key)
        for spec in skill.rules:
            declared[(str(skill.name), spec.rule_id)] = (skill, spec)
    frontmatter = {str(entry["package"]): str(entry["name"]) for entry in routing_eval.load_index()}

    assert set(skeleton) == set(frontmatter)
    seen: list[tuple[str, str]] = []
    for package, cases in skeleton.items():
        assert [str(c["rule_id"]) for c in cases] == sorted(str(c["rule_id"]) for c in cases)
        for case in cases:
            seen.append((str(case["skill"]), str(case["rule_id"])))
            assert case["id"] == f"{frontmatter[package]}-oracle-{case['rule_id']}"
    assert len(seen) == len(set(seen))
    assert set(seen) == set(declared)

    for (skill_name, rule_id), (skill, spec) in sorted(declared.items()):
        matches = [
            case
            for cases in skeleton.values()
            for case in cases
            if case["skill"] == skill_name and case["rule_id"] == rule_id
        ]
        assert len(matches) == 1
        case = matches[0]
        assert case["severity"] == spec.severity.value
        assert case["phase"] == spec.phase
        assert case["description"] == spec.description
        assert case["expected_behavior"] == expected_items(spec)
        assert case["ground_truth"] == ground_truth(spec, skill.name)
        want_fixture = json.loads(json.dumps(skill.must_fire_fixtures()[rule_id], sort_keys=True, default=str))
        assert case["fixture"] == want_fixture
        assert case["fixture_sha256"] == hashlib.sha256(
            json.dumps(want_fixture, sort_keys=True).encode("utf-8")
        ).hexdigest()


def test_oracle_skeleton_is_stable_across_builds() -> None:
    one, two = oracle_skeleton(), oracle_skeleton()
    digest = {
        package: [(case["id"], case["fixture_sha256"], case["ground_truth"]) for case in cases]
        for package, cases in one.items()
    }
    again = {
        package: [(case["id"], case["fixture_sha256"], case["ground_truth"]) for case in cases]
        for package, cases in two.items()
    }
    assert digest == again


# --- expectations and ground truth ------------------------------------------------------


@pytest.mark.parametrize(("phase", "severity"), sorted(_OUTCOMES))
def test_expected_items_is_deterministic_per_phase_and_severity(phase: str, severity: Severity) -> None:
    spec = _rule(severity, phase)
    assert expected_items(spec) == [
        f"Identifies the problem: {spec.description}",
        _OUTCOMES[(phase, severity)],
        f"Cites rule {spec.rule_id}",
    ]


@pytest.mark.parametrize(("phase", "severity"), sorted(_OUTCOMES))
def test_ground_truth_names_skill_rule_severity_and_phase(phase: str, severity: Severity) -> None:
    spec = _rule(severity, phase, rule_id="TS-BAD", description="the staged file breaks one rule")
    assert ground_truth(spec, "training.planner") == (
        f"training.planner: rule TS-BAD ({severity.value}, {phase} phase) fires: the staged file breaks one rule."
    )


def test_ground_truth_accepts_a_case_shaped_record() -> None:
    record = {
        "skill": "training.emit",
        "rule_id": "TS-BAD",
        "severity": "BLOCK",
        "phase": "input",
        "description": "the staged file breaks one rule",
    }
    assert ground_truth(record) == (
        "training.emit: rule TS-BAD (BLOCK, input phase) fires: the staged file breaks one rule."
    )


def test_fixture_sha256_is_stable_across_key_order() -> None:
    first = json.loads(json.dumps({"b": [{"y": 2, "x": 1}, "text"], "a": {"q": 3, "p": 4}}, sort_keys=True))
    second = json.loads(json.dumps({"a": {"p": 4, "q": 3}, "b": [{"x": 1, "y": 2}, "text"]}, sort_keys=True))
    want = hashlib.sha256(json.dumps(first, sort_keys=True).encode("utf-8")).hexdigest()
    assert oracle_cases._fixture_sha256(first) == want
    assert oracle_cases._fixture_sha256(second) == want


# --- questioning -------------------------------------------------------------------------


def test_question_messages_never_names_the_rule() -> None:
    case = {
        "id": "training-oracle-TS-BAD",
        "skill": "training.planner",
        "rule_id": "TS-BAD",
        "fixture": {"staged": [{"file": "rows.csv", "rows": 12}]},
    }
    messages = question_messages(case, "plans the data work end to end")
    assert [str(m["role"]) for m in messages] == ["system", "user"]
    system = str(messages[0]["content"])
    assert "NEVER name a rule id" in system
    assert "the words rule/refuse/invalid/warning" in system
    user = str(messages[1]["content"])
    assert user.startswith("Skill: training.planner - plans the data work end to end\nPayload JSON:\n")
    assert "rows.csv" in user
    assert "TS-BAD" not in user


def _good(_index: int, messages: list[dict], _seed: int) -> str:
    """A writer that obeys the prompt: the good question plus every OBSERVED FACT value it was given."""
    user = messages[-1]["content"]
    lines = user.split("OBSERVED FACTS", 1)[1].splitlines()[1:] if "OBSERVED FACTS" in user else []
    values = [line.split(" = ", 1)[1].strip('"') for line in lines if line.startswith("- ")]
    return f"{GOOD_QUESTION} I saw: {'; '.join(values)}." if values else GOOD_QUESTION


def test_build_cases_writes_one_question_per_case() -> None:
    jobs = _case_jobs()
    backend = FakeBackend([_good] * len(jobs))
    built = build_cases(backend, workers=1)
    flat = [case for package in sorted(built) for case in built[package]]
    assert [case["id"] for case in flat] == [case["id"] for _, case in jobs]
    assert all(case["question"].startswith(GOOD_QUESTION) for case in flat)
    assert all(case["question_model"] == "fake/question-model" for case in flat)
    assert len(backend.calls) == len(jobs)
    assert all(call["seed"] == 0 for call in backend.calls)
    assert all(call["temperature"] == 0.7 and call["max_tokens"] == 4096 for call in backend.calls)


def test_build_cases_retries_a_leaking_question_once_with_seed_1() -> None:
    jobs = _case_jobs()
    first = jobs[0][1]
    leak = f"Please take care of {first['rule_id']} and the staged files as soon as you can, thanks."
    backend = FakeBackend([leak if index == 0 else _good for index in range(len(jobs) + 2)])
    built = build_cases(backend, workers=1)
    flat = [case for package in sorted(built) for case in built[package]]
    assert flat[0]["id"] == first["id"]
    assert flat[0]["question"].startswith(GOOD_QUESTION)
    assert len(backend.calls) == len(jobs) + 1
    assert [call["seed"] for call in backend.calls[:2]] == [0, 1]
    assert backend.calls[0]["messages"] == backend.calls[1]["messages"]


def test_build_cases_refuses_when_the_retry_still_leaks() -> None:
    jobs = _case_jobs()
    first = jobs[0][1]
    bad = "Please fix the reported rule problem in the staged payload and send me the report back."
    backend = FakeBackend([bad])
    with pytest.raises(OracleCasesRefused) as excinfo:
        build_cases(backend, workers=1)
    message = str(excinfo.value)
    assert str(first["id"]) in message
    assert "uses the forbidden word 'rule'" in message
    assert len(backend.calls) == 2
    assert [call["seed"] for call in backend.calls] == [0, 1]


# --- the file and its staleness checks --------------------------------------------------

# a question that passes the question rules (a bare skeleton has none, so check_cases rejects it)
QUESTION = "Please take the payload exactly as given and carry the job through for me, thanks."


def _asked(case: dict) -> dict:
    """The case with a question that passes the rules: the neutral text plus every observed fact."""
    facts = "; ".join(str(f["value"]) for f in case.get("trigger_facts") or [])  # values only, like a user
    return dict(case, question=f"{QUESTION} I observed: {facts}." if facts else QUESTION)


def test_write_and_load_cases_round_trip_under_a_patched_package_dir(tmp_path, monkeypatch) -> None:
    _patch_package_dir(tmp_path, monkeypatch)
    skeleton = {pkg: [_asked(case) for case in cases] for pkg, cases in oracle_skeleton().items()}
    paths = write_cases(skeleton)
    wanted = sorted(oracle_cases._package_dir(package) / "evals" / "behaviour.json" for package in skeleton)
    assert sorted(paths) == wanted

    for package in sorted(skeleton):
        cases, sha = load_cases(package)
        assert cases == skeleton[package]
        raw = (oracle_cases._package_dir(package) / "evals" / "behaviour.json").read_bytes()
        assert sha == hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        assert text.endswith("}\n") and not text.endswith("}\n\n")
        document = json.loads(text)
        assert document["schema"] == oracle_cases.SCHEMA
        assert document["cases"] == skeleton[package]
        assert check_cases(package, cases) == []


def test_load_cases_refuses_when_behaviour_json_is_missing(tmp_path, monkeypatch) -> None:
    _patch_package_dir(tmp_path, monkeypatch)
    with pytest.raises(OracleCasesRefused) as excinfo:
        load_cases("training.has_no_cases")
    assert "evals/behaviour.json is missing" in str(excinfo.value)


def _current_cases() -> tuple[str, list[dict]]:
    skeleton = oracle_skeleton()
    package = sorted(skeleton)[0]
    return package, [_asked(case) for case in skeleton[package]]


def test_check_cases_accepts_a_current_file_only() -> None:
    package, cases = _current_cases()
    assert check_cases(package, cases) == []


def test_check_cases_detects_a_changed_fixture_sha() -> None:
    package, cases = _current_cases()
    cases[0]["fixture_sha256"] = "0" * 64
    problems = check_cases(package, cases)
    assert any("field fixture_sha256 differs" in problem for problem in problems)


def test_check_cases_detects_an_edited_expected_item() -> None:
    package, cases = _current_cases()
    edited = list(cases[0]["expected_behavior"])
    edited[1] = "Reports success after running anyway"
    cases[0]["expected_behavior"] = edited
    problems = check_cases(package, cases)
    assert any("field expected_behavior differs" in problem for problem in problems)


def test_check_cases_detects_an_edited_ground_truth() -> None:
    package, cases = _current_cases()
    cases[0]["ground_truth"] = f"{cases[0]['skill']}: rule {cases[0]['rule_id']} never fires."
    problems = check_cases(package, cases)
    assert any("field ground_truth differs" in problem for problem in problems)


def test_check_cases_detects_an_id_set_difference() -> None:
    package, cases = _current_cases()
    dropped = cases.pop(0)
    problems = check_cases(package, cases)
    assert any(f"case {dropped['id']} is missing from behaviour.json" in problem for problem in problems)


def test_check_cases_detects_a_question_naming_the_rule_id() -> None:
    package, cases = _current_cases()
    cases[0]["question"] = f"Please handle {cases[0]['rule_id']} and the staged files as soon as you can."
    problems = check_cases(package, cases)
    assert any(f"names the rule id {cases[0]['rule_id']}" in problem for problem in problems)


def test_check_cases_detects_a_leaky_or_short_question() -> None:
    package, cases = _current_cases()
    cases[0]["question"] = LEAKY_WORD_QUESTION
    problems = check_cases(package, cases)
    assert any("uses the forbidden word 'rule'" in problem for problem in problems)
    cases[0]["question"] = "too short"
    problems = check_cases(package, cases)
    assert any("is shorter than 40 characters" in problem for problem in problems)
    cases[0]["question"] = ""
    problems = check_cases(package, cases)
    assert any("question is empty" in problem for problem in problems)


def test_trigger_facts_are_the_fields_a_fixture_changed() -> None:
    cases = {c["rule_id"]: c for cs in oracle_skeleton().values() for c in cs}
    paths = {f["path"] for f in cases["TR-PL-002"]["trigger_facts"]}
    assert "fake_plan.stages[0].feasibility.verdict" in paths
    assert not any(p.startswith("fake_plan.goal") for p in paths)   # shared by every sibling plan
    assert [f["value"] for f in cases["EV-HO-001"]["trigger_facts"]] == [0.5, 0.4, 0.004]
    assert all("expected" not in f["path"] for c in cases.values() for f in c["trigger_facts"])


def test_the_question_writer_never_sees_the_expected_outcome() -> None:
    case = next(c for cs in oracle_skeleton().values() for c in cs if c["rule_id"] == "EV-HO-001")
    user = question_messages(case, "eval skill")[1]["content"]
    assert "RED with EV-HO-001" not in user and "OBSERVED FACTS" in user and "0.004" in user


def test_a_question_that_omits_an_observed_fact_is_flawed() -> None:
    case = next(c for cs in oracle_skeleton().values() for c in cs if c["rule_id"] == "EV-HO-001")
    assert check_cases_question(case, QUESTION) is not None
    assert check_cases_question(case, _asked(case)["question"]) is None


def test_a_forbidden_word_inside_the_case_facts_is_the_users_data() -> None:
    case = next(c for cs in oracle_skeleton().values() for c in cs if c["rule_id"] == "FS-HO-002")
    said = f"{QUESTION} My cluster capabilities list refused axes pp and ep."
    assert check_cases_question(case, said) is None
    other = next(c for cs in oracle_skeleton().values() for c in cs if c["rule_id"] == "EV-HO-001")
    assert "refus" in str(check_cases_question(other, said + " 0.5 0.4 0.004"))
