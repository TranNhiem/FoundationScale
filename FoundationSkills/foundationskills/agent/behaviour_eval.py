"""Behaviour evaluation for FoundationSkills skill packages.

This module measures behaviour: once a request is routed to a skill, does the agent do what the skill
prescribes? It is the FoundationSkills version of NVSkills-Eval's plan-level tier (Tier 3). For every
POSITIVE routing case (``should_trigger: true``) of each package an agent model answers the case's
``question`` in two arms and a separate judge model checks every ``expected_behavior`` item against
the case's ``GROUND TRUTH``.

Arms (:data:`ARMS`) show what the skill text adds:

* ``with_skill`` -- the agent sees :data:`AGENT_SYSTEM` plus the package's full ``SKILL.md`` text
  (frontmatter included); this is the gated arm (:data:`GATED_ARM`).
* ``without_skill`` -- the agent sees :data:`AGENT_SYSTEM` only: the agent knows FoundationSkills
  exists and nothing more.

Uplift is ``item_pass_rate(with_skill) - item_pass_rate(with_skill's ablation)``, overall and per
package.

Plan-level limit
----------------
The agent never runs tools; it states the plan it would execute. Items that describe process, not
content, are NOT graded: an item is process-only when it matches (case-insensitive)
``\\bread\\b.*SKILL\\.md|before acting|loaded the``. Those items are counted under ``ungraded_items``
and excluded from every denominator.

Measurement policy
------------------
An answer is *measured* only when the agent produced content and every graded item was judged. An
agent error or empty content is UNMEASURED. A judge error, unparseable JSON, missing, duplicate or
out-of-range item indices or a non-bool ``met`` make the call UNMEASURED: a check that did not run is
never PASS. An agent ``finish_reason == "length"`` with content is graded (truncation is the agent's
loss). A judge call is skipped only when a case has no graded item at all: such an answer is measured
with an empty ``met`` list.

Only the gated arm decides the verdict: its overall ``item_pass_rate`` and every package's
``item_pass_rate`` must reach ``threshold``; the other arm is informational (uplift).

Exit codes follow the repo doctrine: ``0`` PASS, ``5`` RED, ``95`` UNMEASURED. Refusals raise
:class:`BehaviourEvalRefused` whose message names the missing input or precondition, and
:func:`write_behaviour` refuses an unmeasured run because an unmeasured run must not become a
published behaviour report. Reports are written as atomic JSON.

Secrets: this module never prints or stores an API key. The backend reads the key from the
environment variable named by ``api_key_env`` and only hands it to the endpoint.
"""
from __future__ import annotations

import hashlib
import importlib.resources
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from foundationskills.agent.routing_eval import (
    EXIT_PASS,
    EXIT_RED,
    EXIT_UNMEASURED,
    RoutingEvalRefused,
    _atomic_write_text,
    _package_dir,
    _pct,
    _pct_delta,
    _stamp,
)
from foundationskills.core.artifacts import _atomic_write_json
from foundationskills.skills.data_engine.llm_backend import (
    LLMBackend,
    LLMResponse,
    map_ordered,
    parse_json_object,
)

ARMS: tuple[str, ...] = ("with_skill", "without_skill")
GATED_ARM = "with_skill"
SCHEMA = "fskills.behaviour_eval/1"

AGENT_SYSTEM = (
    "You are an engineering agent working with the FoundationScale training framework and its "
    "FoundationSkills skill library. You cannot run tools in this setting: answer with the concrete plan you "
    "would execute - which skill/CLI calls, which inputs, which artifacts you would produce, which validation "
    "rules or refusals apply and why - and the decisions you would make. Be specific; do not ask questions."
)

JUDGE_SYSTEM = (
    "You grade whether an agent's plan satisfies each expected behaviour. Judge only what the answer states; "
    "a behaviour is met only if the answer clearly commits to it (naming the right skill/rule/artifact/decision); "
    "vague or hedged mentions are not met. Reply with JSON "
    "{\"items\": [{\"i\": <index>, \"met\": true|false, \"why\": \"<= 20 words\"}]} covering every index."
)

PROCESS_ONLY = re.compile(r"\bread\b.*SKILL\.md|before acting|loaded the", re.IGNORECASE)

_RETRY_SEED_OFFSET = 1000  # retry seed for an empty/failed completion
_MAX_EXCERPT = 300
_WEAKEST_WIDTH = 80


class BehaviourEvalRefused(RoutingEvalRefused):
    """The behaviour eval refused to run; the message names the missing input."""


# --- case selection and plan-level limits -------------------------------------------------------


def _is_positive(case: dict) -> bool:
    """Return True for the routing cases that expect a skill to trigger."""
    if "should_trigger" in case:
        return case["should_trigger"] is True
    return bool(_expected_skill(case))


def _expected_skill(case: dict) -> str:
    value = case.get("expected_skill")
    return value.strip() if isinstance(value, str) else ""


def _behaviours(case: dict) -> list[str]:
    value = case.get("expected_behavior")
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _graded_items(case: dict) -> list[str]:
    """Content items only: process-only items describe process, not content, and are not graded."""
    return [text for text in _behaviours(case) if not PROCESS_ONLY.search(text)]


def _ungraded_count(case: dict) -> int:
    return sum(1 for text in _behaviours(case) if PROCESS_ONLY.search(text))


# --- prompting -------------------------------------------------------------------------------


def build_agent_messages(arm: str, skill_md: str, question: str) -> list[dict]:
    """Build the system/user message pair that asks the agent for one plan-level answer."""
    _require_arm(arm, "build_agent_messages")
    system = AGENT_SYSTEM if arm != GATED_ARM else f"{AGENT_SYSTEM}\n\n{skill_md}"
    return [{"role": "system", "content": system}, {"role": "user", "content": str(question)}]


def build_judge_messages(question: str, ground_truth: Any, texts: list[str], answer: str) -> list[dict]:
    """Build the system/user message pair that grades one answer against the numbered behaviours."""
    truth = "n/a" if ground_truth is None else str(ground_truth)
    lines = ["QUESTION:", str(question), "", "GROUND TRUTH:", truth, "", "EXPECTED BEHAVIOURS:"]
    lines += [f"{index}. {text}" for index, text in enumerate(texts)]
    lines += ["", "AGENT ANSWER:", str(answer)]
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


# --- the run ---------------------------------------------------------------------------------


def run_behaviour_eval(
    index: list[dict],
    agent: LLMBackend,
    judge: LLMBackend,
    *,
    arms: Iterable[str] = ARMS,
    reps: int = 2,
    workers: int = 4,
    agent_temperature: float = 0.6,
    agent_max_tokens: int = 4096,
    judge_max_tokens: int = 2048,
    threshold: float = 0.7,
    packages: Iterable[str] | None = None,
    created: datetime | str | None = None,
) -> dict:
    """Run the behaviour evaluation over the positive cases and return the report dictionary."""
    arm_list: tuple[str, ...] = (arms,) if isinstance(arms, str) else tuple(arms)
    _check_run_args(arm_list, reps, workers, threshold)
    _check_self_grading(agent, judge)
    selected = _select_packages(index, packages)
    skill_mds = {str(entry["package"]): _load_skill_md(entry) for entry in selected}
    names = sorted({str(entry["name"]) for entry in selected})
    jobs: list[tuple[str, dict, dict, int]] = []
    for arm in arm_list:
        for entry in selected:
            for case in entry["evals"]:
                if _is_positive(case):
                    for rep in range(reps):
                        jobs.append((arm, entry, case, rep))

    def run_one(job: tuple[str, dict, dict, int]) -> dict:
        arm, entry, case, rep = job
        response, retried = _complete_retrying(
            agent,
            build_agent_messages(arm, skill_mds[str(entry["package"])], str(case["question"])),
            temperature=float(agent_temperature),
            max_tokens=int(agent_max_tokens),
            seed=rep,
            json_mode=False,
        )
        row = _record_call(response, arm, entry, case, rep, judge, int(judge_max_tokens))
        row["agent_retried"] = retried
        return row

    calls = list(map_ordered(run_one, jobs, int(workers)))
    results = {arm: _score_arm(calls, arm, names) for arm in arm_list}
    verdict, exit_code, reasons = _verdict(results, names, float(threshold))
    return {
        "schema": SCHEMA,
        "created": _stamp(created),
        "agent_model": str(getattr(agent, "model", "")),
        "judge_model": str(getattr(judge, "model", "")),
        "backend_kind": {"agent": str(getattr(agent, "kind", "")), "judge": str(getattr(judge, "kind", ""))},
        "arms": list(arm_list),
        "reps": reps,
        "agent_temperature": agent_temperature,
        "agent_max_tokens": agent_max_tokens,
        "judge_max_tokens": judge_max_tokens,
        "threshold": threshold,
        "gated_arm": GATED_ARM,
        "plan_level": True,
        "index": [_index_view(entry) for entry in selected],
        "results": results,
        "uplift": _uplift(results, names),
        "calls": calls,
        "verdict": verdict,
        "exit_code": exit_code,
        "reasons": reasons,
    }


def _check_run_args(arms: tuple[str, ...], reps: int, workers: int, threshold: float) -> None:
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise BehaviourEvalRefused(f"run_behaviour_eval: reps must be an integer >= 1, got {reps!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise BehaviourEvalRefused(f"run_behaviour_eval: workers must be an integer >= 1, got {workers!r}")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 < threshold <= 1.0:
        raise BehaviourEvalRefused(f"run_behaviour_eval: threshold must be in (0, 1], got {threshold!r}")
    for arm in arms:
        _require_arm(arm, "run_behaviour_eval")
    if GATED_ARM not in arms:
        raise BehaviourEvalRefused(f"run_behaviour_eval: arms must include the gated arm {GATED_ARM!r}, got {arms!r}")


def _require_arm(arm: str, where: str = "build_agent_messages") -> None:
    if arm not in ARMS:
        raise BehaviourEvalRefused(f"{where}: unknown arm {arm!r}; expected one of {', '.join(ARMS)}")


def _check_self_grading(agent: LLMBackend, judge: LLMBackend) -> None:
    if agent is judge:
        raise BehaviourEvalRefused("run_behaviour_eval: self-grading refused: agent and judge are one backend object")
    if str(agent.model) == str(judge.model):
        raise BehaviourEvalRefused(
            f"run_behaviour_eval: self-grading refused: agent and judge both run model {agent.model}")


def _select_packages(index: list[dict], packages: Iterable[str] | None) -> list[dict]:
    """Filter the index to the requested skill names (or package paths); unknown names select nothing."""
    if packages is None:
        return list(index)
    wanted = {str(name).strip() for name in packages if str(name).strip()}
    return [entry for entry in index if str(entry["name"]) in wanted or str(entry["package"]) in wanted]


def _load_skill_md(entry: dict) -> str:
    """Read the package's SKILL.md text and refuse when the index no longer matches it."""
    package = str(entry["package"])
    skill_md = importlib.resources.files(package).joinpath("SKILL.md")
    if not skill_md.is_file():
        raise BehaviourEvalRefused(f"load_skill_md: {package}: SKILL.md is missing")
    data = skill_md.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = str(entry.get("skill_md_sha256") or "")
    if digest != expected:
        raise BehaviourEvalRefused(
            f"load_skill_md: {package}: SKILL.md sha256 {digest[:12]} is not the index sha256 "
            f"{expected[:12]}; the index is stale")
    return data.decode("utf-8")


def _complete_retrying(backend: LLMBackend, messages: list[dict], *, seed: int, **kwargs: Any
                       ) -> tuple[LLMResponse, bool]:
    """One completion, retried ONCE under another seed when it failed or came back empty (a thinking model
    that spends its whole budget answers nothing). A second failure stays a failure: the call is unmeasured."""
    response = backend.complete(messages, seed=seed, **kwargs)
    if response.content and not response.error:
        return response, False
    return backend.complete(messages, seed=seed + _RETRY_SEED_OFFSET, **kwargs), True


def _record_call(response: LLMResponse, arm: str, entry: dict, case: dict, rep: int,
                 judge: LLMBackend, judge_max_tokens: int) -> dict:
    texts = _graded_items(case)
    row: dict[str, Any] = {
        "arm": arm,
        "package": str(entry["name"]),
        "case_id": case["id"],
        "rep": rep,
        "measured": False,
        "items_met": 0,
        "items_graded": len(texts),
        "ungraded_items": _ungraded_count(case),
        "agent_tokens": int(response.completion_tokens),
        "judge_tokens": 0,
        "error": None,
        "met": [],
        "item_texts": list(texts),
        "answer_excerpt": "",
        "answer": "",
        "judge_retried": False,
    }
    if response.error:
        row["error"] = str(response.error)
        return row
    if not response.content:
        row["error"] = "empty response content"
        return row
    answer = str(response.content)
    row["answer_excerpt"] = answer[:_MAX_EXCERPT]
    row["answer"] = answer  # kept whole so the judge can be audited (a second judge regrades these)
    if not texts:
        row["measured"] = True
        return row
    verdict_response, row["judge_retried"] = _complete_retrying(
        judge,
        build_judge_messages(str(case["question"]), case.get("ground_truth"), texts, answer),
        temperature=0.0,
        max_tokens=judge_max_tokens,
        seed=0,
        json_mode=True,
    )
    row["judge_tokens"] = int(verdict_response.completion_tokens)
    met, error = _parse_judgement(verdict_response, len(texts))
    if met is None:
        row["error"] = error
        return row
    row["measured"] = True
    row["met"] = met
    row["items_met"] = sum(1 for value in met if value)
    return row


def _parse_judgement(response: LLMResponse, graded: int) -> tuple[list[bool] | None, str | None]:
    """Return (per-item met flags, None) or (None, error); the judge must cover every index exactly once."""
    if response.error:
        return None, f"judge error: {response.error}"
    if not response.content:
        return None, "judge: empty response content"
    parsed = parse_json_object(str(response.content))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        return None, "judge: unparseable: " + str(response.content)[:80]
    met: list[bool | None] = [None] * graded
    for record in parsed["items"]:
        if not isinstance(record, dict):
            return None, "judge: unparseable: " + str(response.content)[:80]
        position = record.get("i")
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < graded:
            return None, f"judge: unexpected index {position!r}"
        if met[position] is not None:
            return None, f"judge: duplicate index {position}"
        value = record.get("met")
        if not isinstance(value, bool):
            return None, f"judge: non-bool met at index {position}"
        met[position] = value
    missing = [index for index, value in enumerate(met) if value is None]
    if missing:
        return None, f"judge: missing index {missing[0]}"
    return [bool(value) for value in met], None


def _score_arm(calls: list[dict], arm: str, names: list[str]) -> dict:
    rows = [c for c in calls if c["arm"] == arm]
    score = _score_rows(rows)
    score["packages"] = {
        name: _score_rows([c for c in rows if c.get("package") == name]) for name in names
    }
    return score


def _score_rows(rows: list[dict]) -> dict:
    """Item pass rate is sum(met)/sum(graded) over measured calls; case pass rate counts full sweeps."""
    measured = [c for c in rows if c["measured"]]
    items_met = sum(int(c["items_met"]) for c in measured)
    items_graded = sum(int(c["items_graded"]) for c in measured)
    case_pass = sum(1 for c in measured if int(c["items_met"]) == int(c["items_graded"]))
    return {
        "item_pass_rate": items_met / items_graded if items_graded else None,
        "case_pass_rate": case_pass / len(measured) if measured else None,
        "calls": len(rows),
        "measured": len(measured),
        "unmeasured": len(rows) - len(measured),
        "items_met": items_met,
        "items_graded": items_graded,
        "ungraded_items": sum(int(c["ungraded_items"]) for c in rows),
    }


def _uplift(results: dict[str, dict], names: list[str]) -> dict:
    first = results.get(str(ARMS[0])) or {}
    second = results.get(str(ARMS[1])) or {}
    return {
        "item_pass_rate": _delta(first.get("item_pass_rate"), second.get("item_pass_rate")),
        "packages": {
            name: _delta(((first.get("packages") or {}).get(name) or {}).get("item_pass_rate"),
                         ((second.get("packages") or {}).get(name) or {}).get("item_pass_rate"))
            for name in names
        },
    }


def _delta(arm_rate: Any, base_rate: Any) -> float | None:
    if isinstance(arm_rate, bool) or not isinstance(arm_rate, (int, float)):
        return None
    if isinstance(base_rate, bool) or not isinstance(base_rate, (int, float)):
        return None
    return float(arm_rate) - float(base_rate)


def _verdict(results: dict[str, dict], names: list[str], threshold: float) -> tuple[str, int, list[str]]:
    score = results.get(GATED_ARM) or {}
    gated_calls = int(score.get("calls") or 0)
    if not gated_calls:
        return "unmeasured", EXIT_UNMEASURED, [f"verdict: the gated arm {GATED_ARM!r} made no calls to measure"]
    unmeasured = int(score.get("unmeasured") or 0)
    if unmeasured:
        return "unmeasured", EXIT_UNMEASURED, [
            f"verdict: {unmeasured} of {gated_calls} {GATED_ARM!r} calls were unmeasured; "
            "an unmeasured run is never PASS"
        ]
    overall = score.get("item_pass_rate")
    if isinstance(overall, bool) or not isinstance(overall, (int, float)):
        return "unmeasured", EXIT_UNMEASURED, [
            f"verdict: overall item_pass_rate is undefined; the {GATED_ARM!r} arm graded no behaviour"]
    reasons: list[str] = []
    red = False
    if float(overall) < threshold:
        red = True
        reasons.append(f"verdict: {GATED_ARM} overall item_pass_rate {_pct(overall)} "
                       f"is under the gate {_pct(threshold)}")
    for name in names:
        stats = (score.get("packages") or {}).get(name) or {}
        rate = stats.get("item_pass_rate")
        if int(stats.get("measured") or 0) == 0:
            reasons.append(f"verdict: package {name} has no measured {GATED_ARM!r} call")
            return "unmeasured", EXIT_UNMEASURED, reasons
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            reasons.append(f"verdict: package {name} item_pass_rate is undefined; no behaviour was graded")
            return "unmeasured", EXIT_UNMEASURED, reasons
        if float(rate) < threshold:
            red = True
            reasons.append(f"verdict: {GATED_ARM} package {name} item_pass_rate {_pct(rate)} "
                           f"is under the gate {_pct(threshold)}")
    return ("red", EXIT_RED, reasons) if red else ("pass", EXIT_PASS, [])


# --- report output ----------------------------------------------------------------------------


def write_report(report: dict, path: Path) -> None:
    """Write the behaviour eval report as atomic JSON."""
    _atomic_write_json(Path(path), report)


def render_behaviour_md(report: dict, package_name: str) -> str:
    """Render the BEHAVIOUR.md body for one skill package from a run report."""
    entry, name = _report_entry(report, package_name)
    results: dict[str, dict] = report.get("results") or {}
    rows = [c for c in report.get("calls") or [] if c.get("package") == name]
    gated = [c for c in rows if c.get("arm") == GATED_ARM]
    lines: list[str] = [
        f"# Behaviour evaluation: {name}",
        "",
        f"Generated by `fskills behaviour-eval` against SKILL.md {str(entry.get('skill_md_sha256') or '')[:12]}"
        f" and eval cases {str(entry.get('evals_sha256') or '')[:12]}; do not edit by hand.",
        "",
        "## Summary",
        "",
        f"- Verdict: {report.get('verdict')}",
        f"- Agent model: {report.get('agent_model')}",
        f"- Judge model: {report.get('judge_model')}",
        f"- Created: {report.get('created')}",
        f"- Reps: {report.get('reps')}",
        f"- Threshold: {_pct(report.get('threshold'))}",
        f"- Plan level: no tool execution; {_package_ungraded(rows)} process-only items ungraded.",
        "",
        "## Results",
        "",
        "| arm | item pass | case pass | measured |",
        "| --- | --- | --- | --- |",
    ]
    for arm in report.get("arms") or []:
        stats = ((results.get(arm) or {}).get("packages") or {}).get(name) or {}
        lines.append(f"| {arm} | {_pct(stats.get('item_pass_rate'))} | {_pct(stats.get('case_pass_rate'))} "
                     f"| {stats.get('measured', 0)} |")
    uplift_all = report.get("uplift") or {}
    lines += [
        "",
        f"Uplift (with_skill - without_skill, item pass): overall {_pct_delta(uplift_all.get('item_pass_rate'))},"
        f" this package {_pct_delta((uplift_all.get('packages') or {}).get(name))}.",
        "",
        f"## Cases (gated arm: {GATED_ARM})",
        "",
        "| case id | items met (mean over reps) | weakest item |",
        "| --- | --- | --- |",
    ]
    for case_id in _case_ids(gated):
        lines.append(_case_row(gated, case_id))
    return "\n".join(lines) + "\n"


def _report_entry(report: dict, package_name: str) -> tuple[dict, str]:
    wanted = str(package_name)
    for entry in report.get("index") or []:
        if wanted in (str(entry.get("name")), str(entry.get("package"))):
            return entry, str(entry.get("name"))
    raise BehaviourEvalRefused(f"render_behaviour_md: {wanted} is not in the report index")


def _case_ids(rows: list[dict]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id"))
        if case_id not in ordered:
            ordered.append(case_id)
    return ordered


def _case_row(gated: list[dict], case_id: str) -> str:
    rows = [c for c in gated if str(c.get("case_id")) == case_id and c.get("measured")]
    if not rows:
        return f"| {case_id} | n/a | n/a |"
    graded = int(rows[0]["items_graded"])
    mean = sum(int(c["items_met"]) for c in rows) / len(rows)
    return f"| {case_id} | {mean:.2f} of {graded} | {_weakest_item(rows, graded)} |"


def _weakest_item(rows: list[dict], graded: int) -> str:
    judged = [c for c in rows if len(c.get("met") or []) == graded]
    if graded < 1 or not judged:
        return "n/a"
    rates = [sum(bool(c["met"][index]) for c in judged) / len(judged) for index in range(graded)]
    worst = min(range(graded), key=lambda index: (rates[index], index))
    texts = [str(text) for text in (rows[0].get("item_texts") or [])]
    return f"item {worst}: {_clip(texts[worst] if worst < len(texts) else '', _WEAKEST_WIDTH)}"


def _package_ungraded(rows: list[dict]) -> int:
    seen: dict[str, int] = {}
    for row in rows:
        seen.setdefault(str(row.get("case_id")), int(row.get("ungraded_items") or 0))
    return sum(seen.values())


def write_behaviour(report: dict, index: list[dict]) -> list[Path]:
    """Write one BEHAVIOUR.md per package next to its SKILL.md; refuse unmeasured runs."""
    if report.get("verdict") == "unmeasured":
        raise BehaviourEvalRefused("write_behaviour: an unmeasured run must not become a published behaviour report")
    packages = {str(entry["name"]): str(entry["package"]) for entry in index}
    paths: list[Path] = []
    for view in report.get("index") or []:
        name = str(view.get("name"))
        target = _package_dir(packages.get(name) or str(view.get("package"))) / "BEHAVIOUR.md"
        _atomic_write_text(target, render_behaviour_md(report, name))
        paths.append(target)
    return paths


# --- small formatting and view helpers ------------------------------------------------------------


def _index_view(entry: dict) -> dict:
    return {
        "package": str(entry["package"]),
        "name": str(entry["name"]),
        "skill_md_sha256": str(entry["skill_md_sha256"]),
        "evals_sha256": str(entry["evals_sha256"]),
        "positives": sum(1 for case in entry["evals"] if _is_positive(case)),
    }


def _clip(text: Any, width: int) -> str:
    """Collapse whitespace and truncate to width characters for one table cell."""
    return " ".join(str(text).split())[:width]
