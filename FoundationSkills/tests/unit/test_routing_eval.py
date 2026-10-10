"""Unit tests for foundationskills.agent.routing_eval (no network, no real LLM).

The routing policy is exercised against a fake backend that answers from the
expected_skill field of the shipped eval cases. The index is read from the real
skill packages (``load_index``) so the fake router can be perfect, always null or
ablated on frontmatter content. Doctrine under test: exit 0 PASS, 5 RED,
95 UNMEASURED (a call that did not run is never PASS), 96 REFUSED (refusal raises
``RoutingEvalRefused`` and writes no report).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from foundationskills.agent import routing_eval
from foundationskills.agent.routing_eval import (
    ARMS,
    GATED_ARM,
    RoutingEvalRefused,
    load_index,
    render_benchmark_md,
    render_index,
    run_routing_eval,
    write_benchmarks,
    write_report,
)
from foundationskills.skills.data_engine.llm_backend import LLMResponse

CREATED = "2026-10-02T00:00:00Z"
PACKAGE_NAMES = ("fskills-auto-research", "fskills-confirm-before-launch", "fskills-data-engine", "fskills-evaluation",
                 "fskills-measured-or-unmeasured", "fskills-probe-first", "fskills-refusal-surface", "fskills-training")
HEX64 = re.compile(r"[0-9a-f]{64}")


class FakeBackend:
    """In-memory backend with kind="fake", model="fake-1"; records every call."""

    kind = "fake"
    model = "fake-1"

    def __init__(
        self,
        router: Callable[[str, str], object] | None = None,
        fail_when: Callable[[str, dict], str | None] | None = None,
    ) -> None:
        self.router = router if router is not None else (lambda question, system: {"skill": None})
        self.fail_when = fail_when if fail_when is not None else (lambda question, call: None)
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
        question = messages[-1]["content"]
        system = messages[0]["content"]
        call = {
            "index": len(self.calls),
            "question": question,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "json_mode": json_mode,
        }
        self.calls.append(call)
        error = self.fail_when(question, call)
        if error is not None:
            return LLMResponse(None, "stop", 0, 0, False, error)
        payload = self.router(question, system)
        content = json.dumps(payload) if isinstance(payload, dict) else payload
        return LLMResponse(content, "stop", 0, 0, False, None)


def ground_truth_router(answers: dict[str, object]) -> Callable[[str, str], object]:
    """Router that names the expected skill of the shipped eval case."""

    return lambda question, system: {"skill": answers[question]}


def _calls_for(report: dict, case_id: str) -> list[dict]:
    return [call for call in report["calls"] if call["case_id"] == case_id]


def _package_metrics(score: dict, name: str) -> dict:
    """Per-package metrics for one package wherever an arm score nests them."""

    for value in score.values():
        block = value.get(name) if isinstance(value, dict) else None
        if isinstance(block, dict) and "trigger_recall" in block:
            return block
    raise AssertionError(f"no per-package metrics for {name!r} in arm score keys {sorted(score)}")


@pytest.fixture(scope="session")
def index() -> list[dict]:
    return load_index()


@pytest.fixture(scope="session")
def answers(index: list[dict]) -> dict[str, object]:
    return {case["question"]: case["expected_skill"] for pkg in index for case in pkg["evals"]}


@pytest.fixture(scope="session")
def cases(index: list[dict]) -> list[tuple[str, dict]]:
    return [(pkg["name"], case) for pkg in index for case in pkg["evals"]]


def test_load_index_finds_every_evaluated_package(index: list[dict]) -> None:
    names = [pkg["name"] for pkg in index]
    assert names == sorted(PACKAGE_NAMES)
    for pkg in index:
        assert pkg["package"]
        assert pkg["description"]
        assert pkg["when_to_use"] and all(isinstance(phrase, str) and phrase for phrase in pkg["when_to_use"])
        assert len(pkg["evals"]) >= 10
        assert HEX64.fullmatch(pkg["skill_md_sha256"]) is not None
        assert HEX64.fullmatch(pkg["evals_sha256"]) is not None
        for case in pkg["evals"]:
            assert case["id"] and case["question"] and "expected_skill" in case


def test_render_index_separates_the_three_arms(index: list[dict]) -> None:
    full = render_index(index, "full")
    names_only = render_index(index, "names_only")
    description = render_index(index, "description")
    for pkg in index:
        assert pkg["name"] in full
        assert any(phrase in full for phrase in pkg["when_to_use"])
        assert pkg["name"] in names_only
        assert pkg["description"] not in names_only
        for phrase in pkg["when_to_use"]:
            assert phrase not in names_only
        assert pkg["name"] in description
        assert pkg["description"] in description
        for phrase in pkg["when_to_use"]:
            assert phrase not in description


def test_perfect_router_passes_and_report_is_written(index: list[dict], answers: dict, tmp_path: Path) -> None:
    backend = FakeBackend(ground_truth_router(answers))
    report = run_routing_eval(
        index, backend, arms=ARMS, reps=2, workers=2, temperature=0.4, max_tokens=512,
        threshold=0.9, created=CREATED,
    )
    assert report["schema"] == "fskills.routing_eval/1"
    assert report["created"] == CREATED
    assert report["model"] == "fake-1"
    assert report["backend_kind"] == "fake"
    assert report["arms"] == list(ARMS)
    assert GATED_ARM in ARMS and report["gated_arm"] == GATED_ARM
    assert report["verdict"] == "pass"
    assert report["exit_code"] == 0
    assert report["reasons"] == []
    total_cases = sum(len(pkg["evals"]) for pkg in index)
    assert len(report["calls"]) == len(ARMS) * total_cases * 2
    assert all(call["measured"] for call in report["calls"])
    assert all(call["correct"] for call in report["calls"])
    for arm in ARMS:
        score = report["results"][arm]
        assert score["accuracy"] == 1.0
        assert score["measured"] == score["calls"]
        assert score["unmeasured"] == 0
    assert report["uplift"]["full_minus_names_only"] == 0.0
    assert report["uplift"]["full_minus_description"] == 0.0
    report_path = tmp_path / "report.json"
    write_report(report, report_path)
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    assert stored["schema"] == "fskills.routing_eval/1"
    assert stored["verdict"] == "pass"
    assert stored["exit_code"] == 0


def test_null_router_is_red(index: list[dict]) -> None:
    backend = FakeBackend(lambda question, system: {"skill": None})
    report = run_routing_eval(index, backend, arms=("full",), reps=1, workers=1, created=CREATED)
    assert report["verdict"] == "red"
    assert report["exit_code"] == 5
    assert report["reasons"]
    score = report["results"]["full"]
    for pkg in index:
        metrics = _package_metrics(score, pkg["name"])
        assert metrics["trigger_recall"] == 0.0
        assert metrics["false_triggers"] == 0


def test_when_to_use_text_is_the_ablation_signal(index: list[dict], answers: dict) -> None:
    phrases = [pkg["when_to_use"][0] for pkg in index]

    def router(question: str, system: str) -> object:
        # Only the "full" arm carries when_to_use phrases, so only that arm routes well.
        if any(phrase in system for phrase in phrases):
            return {"skill": answers[question]}
        return {"skill": None}

    backend = FakeBackend(router)
    report = run_routing_eval(
        index, backend, arms=("full", "names_only"), reps=1, workers=1, threshold=0.9, created=CREATED,
    )
    assert report["results"]["full"]["accuracy"] == 1.0
    assert report["results"]["names_only"]["accuracy"] < 1.0
    assert report["uplift"]["full_minus_names_only"] > 0.0
    assert report["verdict"] == "pass"


def test_backend_error_is_unmeasured_and_blocks_benchmarks(index: list[dict], answers: dict, cases: list) -> None:
    target = next(case for name, case in cases if name == "fskills-evaluation")
    question = target["question"]

    def fail_on_first_rep(question_asked: str, call: dict) -> str | None:
        return "HTTP 500" if question_asked == question and call["seed"] == 0 else None

    backend = FakeBackend(ground_truth_router(answers), fail_when=fail_on_first_rep)
    report = run_routing_eval(index, backend, arms=("full",), reps=2, workers=1, created=CREATED)
    assert report["verdict"] == "unmeasured"
    assert report["exit_code"] == 95
    records = _calls_for(report, target["id"])
    assert len(records) == 2
    first = next(record for record in records if record["rep"] == 0)
    assert first["measured"] is False
    assert "500" in (first["error"] or "")
    assert all(record["measured"] is True for record in records if record["rep"] != 0)
    assert all(call["measured"] for call in report["calls"] if call["case_id"] != target["id"])
    with pytest.raises(RoutingEvalRefused):
        write_benchmarks(report, index)


def test_unparseable_content_is_unmeasured(index: list[dict], answers: dict, cases: list) -> None:
    target = cases[0][1]
    question = target["question"]

    def router(question_asked: str, system: str) -> object:
        return "I think data engine" if question_asked == question else {"skill": answers[question_asked]}

    backend = FakeBackend(router)
    report = run_routing_eval(index, backend, arms=("full",), reps=1, workers=1, created=CREATED)
    record = _calls_for(report, target["id"])[0]
    assert record["measured"] is False
    assert record.get("predicted") is None
    error = record["error"] or ""
    assert "unparseable" in error.lower()
    assert "I think data engine" in error
    assert all(call["measured"] for call in report["calls"] if call["case_id"] != target["id"])


def test_unknown_skill_is_wrong_but_measured_and_none_string_normalises(
    index: list[dict], answers: dict, cases: list,
) -> None:
    unknown_case = cases[0][1]
    normalised = next((case for _, case in cases if case["expected_skill"] is None), cases[1][1])

    def router(question: str, system: str) -> object:
        if question == unknown_case["question"]:
            return {"skill": "fskills-foo"}
        if question == normalised["question"]:
            return {"skill": "None"}
        return {"skill": answers[question]}

    backend = FakeBackend(router)
    report = run_routing_eval(index, backend, arms=("full",), reps=1, workers=1, created=CREATED)
    bad = _calls_for(report, unknown_case["id"])[0]
    assert bad["measured"] is True
    assert bad["predicted"] == "fskills-foo"
    assert bad["correct"] is False
    assert "fskills-foo" in (bad["error"] or "")
    assert "unknown" in (bad["error"] or "").lower()
    good = _calls_for(report, normalised["id"])[0]
    assert good["measured"] is True
    assert good["predicted"] is None
    assert good["correct"] is (normalised["expected_skill"] is None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reps": 0},
        {"arms": ("names_only", "description")},
        {"arms": ("full", "definitely_not_an_arm")},
        {"arms": ("full",), "threshold": 0.0},
        {"arms": ("full",), "threshold": 1.0001},
        {"arms": ("full",), "workers": 0},
    ],
    ids=["no-reps", "gate-missing", "unknown-arm", "zero-threshold", "threshold-above-one", "no-workers"],
)
def test_bad_inputs_are_refused(index: list[dict], answers: dict, kwargs: dict) -> None:
    backend = FakeBackend(ground_truth_router(answers))
    with pytest.raises(RoutingEvalRefused):
        run_routing_eval(index, backend, **kwargs)


def test_render_benchmark_md_covers_one_package_only(index: list[dict], answers: dict) -> None:
    report = run_routing_eval(
        index, FakeBackend(ground_truth_router(answers)), arms=ARMS, reps=1, workers=1, created=CREATED,
    )
    assert report["verdict"] == "pass"
    md = render_benchmark_md(report, "fskills-evaluation")
    assert md.startswith("# Routing evaluation: fskills-evaluation")
    assert "do not edit by hand" in md
    assert report["index_sha256"][:12] in md
    assert "## Uplift" in md
    by_name = {pkg["name"]: pkg for pkg in index}
    own_ids = [case["id"] for case in by_name["fskills-evaluation"]["evals"]]
    other_ids = [
        case["id"]
        for pkg in index
        if pkg["name"] != "fskills-evaluation"
        for case in pkg["evals"]
    ]
    assert any(case_id in md for case_id in own_ids)
    assert all(case_id not in md for case_id in other_ids)
    assert "## Caveats" in md
    assert "100.0%" in md


def test_write_benchmarks_writes_next_to_each_skill_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: list[dict], answers: dict,
) -> None:
    report = run_routing_eval(
        index, FakeBackend(ground_truth_router(answers)), arms=ARMS, reps=1, workers=1, created=CREATED,
    )
    assert report["verdict"] == "pass"
    seen: list[str] = []

    def fake_package_dir(package: str) -> Path:
        seen.append(str(package))
        (tmp_path / str(package)).mkdir()
        return tmp_path / str(package)

    monkeypatch.setattr(routing_eval, "_package_dir", fake_package_dir)
    paths = write_benchmarks(report, index)
    assert len(paths) == len(index)
    assert len(seen) == len(index)
    assert {path.parent.name for path in paths} == set(seen)
    for package in seen:
        assert (tmp_path / package / "BENCHMARK.md").exists()
    for path in paths:
        assert path.name == "BENCHMARK.md"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# Routing evaluation:")
        assert "do not edit by hand" in text


def test_reps_cycle_through_seeds_and_always_use_json_mode(index: list[dict], answers: dict) -> None:
    backend = FakeBackend(ground_truth_router(answers))
    report = run_routing_eval(
        index, backend, arms=("full",), reps=3, workers=1, temperature=0.4, max_tokens=512, created=CREATED,
    )
    assert len(backend.calls) == 3 * len(answers)
    assert report["reps"] == 3
    by_question: dict[str, list[dict]] = {}
    for call in backend.calls:
        by_question.setdefault(call["question"], []).append(call)
    assert set(by_question) == set(answers)
    for calls in by_question.values():
        assert sorted(call["seed"] for call in calls) == [0, 1, 2]
    for call in backend.calls:
        assert call["json_mode"] is True
        assert call["temperature"] == 0.4
        assert call["max_tokens"] == 512
    for case_id in {case["id"] for case in index[0]["evals"]}:
        assert sorted(record["rep"] for record in _calls_for(report, case_id)) == [0, 1, 2]


def test_cli_run_refuses_a_missing_key_env_without_writing_a_report(tmp_path: Path, monkeypatch, capsys) -> None:
    from foundationskills import cli

    monkeypatch.delenv("FSKILLS_NO_SUCH_KEY", raising=False)
    out = tmp_path / "routing.json"
    rc = cli.main(["routing-eval", "run", "--base-url", "http://127.0.0.1:9", "--model", "m",
                   "--api-key-env", "FSKILLS_NO_SUCH_KEY", "--out", str(out)])
    assert rc == 96 and not out.exists()
    assert "FSKILLS_NO_SUCH_KEY" in capsys.readouterr().err


def test_cli_render_refuses_an_unmeasured_report(tmp_path: Path, index: list[dict], answers: dict) -> None:
    from foundationskills import cli

    backend = FakeBackend(ground_truth_router(answers), fail_when=lambda question, call: "HTTP 500")
    report = run_routing_eval(index, backend, arms=("full",), reps=1, workers=1, created=CREATED)
    path = tmp_path / "routing.json"
    write_report(report, path)
    assert cli.main(["routing-eval", "render", "--report", str(path)]) == 96
