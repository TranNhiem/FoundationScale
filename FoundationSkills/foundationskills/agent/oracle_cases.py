"""Code-grounded ("oracle") behaviour cases for FoundationSkills skill packages.

Routing and behaviour evaluations are written from skill prose, so a with-skill score is high partly
by construction. Oracle cases take their ground truth from the IMPLEMENTATION: every declared rule
(``RuleSpec(rule_id, description, severity, phase)``) of every registered skill becomes exactly one
case, its fixture is the rule's must-fire negative fixture, and its expected behaviour is spelled out
from the outcome semantics of :mod:`foundationskills.core.contract` and :meth:`BaseSkill.execute`:

* input-phase ``BLOCK`` -- the run is refused before any work: status ``REFUSED``, nothing written.
* handoff-phase ``BLOCK`` -- the run does not report success: the result is ``RED`` (failed).
* ``WARN`` / ``INFO`` -- non-blocking findings that travel with the result.

Every case carries its ground truth and its expectations next to the fixture (and, once built, the
natural question an evaluator asks about it). The file is re-derivable from code: :func:`oracle_skeleton`
rebuilds it and :func:`check_cases` reports anything that drifted in a stored ``evals/behaviour.json``:
a changed rule, a changed fixture hash, an edited expectation or a question that leaked the oracle.

Questions come from an LLM that sees the must-fire fixture only. A question leaks the case when it is
empty, shorter than :data:`_MIN_QUESTION_CHARS`, names the rule id or uses the words rule,
:wdata:`refus` or :data:`invalid` (case-insensitively): such a question is regenerated once
(``seed=1``) and then refused (:class:`OracleCasesRefused`) instead of stored -- a leaked question is
never measured. The unit suite drives every fixture through ``execute()`` on every run; this module
only materialises the cases.

Schema: :data:`SCHEMA`. Secrets: this module never prints or stores an API key.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from foundationskills.agent.routing_eval import (
    RoutingEvalRefused,
    _atomic_write_text,
    _package_dir,
    _skill_package,
    load_index,
)
from foundationskills.core import SkillRegistry
from foundationskills.core.contract import BaseSkill, RuleSpec, Severity
from foundationskills.skills import register_builtin_skills
from foundationskills.skills.data_engine.llm_backend import LLMBackend, map_ordered

SCHEMA = "fskills.oracle_cases/1"
BEHAVIOUR_FILE = "behaviour.json"

_CLEARED_KEYS: tuple[str, ...] = (
    "skill",
    "rule_id",
    "severity",
    "phase",
    "description",
    "fixture_sha256",
    "expected_behavior",
    "ground_truth",
    "trigger_facts",
)
# fixture keys the question writer must never see: the authors' expected outcome is the answer
_HIDDEN_KEYS: tuple[str, ...] = ("expected",)
_LEAKY_WORDS: tuple[str, ...] = ("rule ", "refus", "invalid")
_MIN_QUESTION_CHARS = 40

_QUESTION_SYSTEM = (
    "You write the message a real ML engineer would send to an AI agent. Turn the given request "
    "payload (and any staged files or simulated results in it) into a natural first-person request with ALL "
    "the concrete values. If OBSERVED FACTS are listed, the user has already seen them (a run's results, their "
    "cluster's capabilities, what a planner produced, what happened to a file): state every one explicitly, "
    "with its exact value, as something they observed or know, then ask what the outcome is. NEVER name a rule "
    "id, never say or hint what is wrong, missing or risky, never use the words rule/refuse/invalid/warning. "
    "2-7 sentences. Reply with the message only."
)


class OracleCasesRefused(ValueError):
    """Oracle case work refused; the message names the missing input or the leaking case."""


# --- the skeleton ------------------------------------------------------------------------


def oracle_skeleton(registry: SkillRegistry | None = None) -> dict[str, list[dict]]:
    """Return one case per declared rule, keyed by package path and sorted by rule_id.

    A case is grounded in code: the rule's must-fire fixture (JSON-normalised) plus the behaviour the
    outcome semantics demand. Refuses when a package has no SKILL.md name or when a declared rule has
    no must-fire fixture -- a rule without a negative fixture cannot be trusted to fire.
    """
    reg = _built_registry(registry)
    try:
        entries = load_index(reg)
    except RoutingEvalRefused as exc:
        raise OracleCasesRefused(f"oracle_skeleton: {exc}") from exc
    names = {str(entry["package"]): str(entry["name"]) for entry in entries}
    cases: dict[str, list[dict]] = {package: [] for package in sorted(names)}
    for key in sorted(reg.names()):
        skill = reg.get(key)
        package = _skill_package(skill)
        display = names.get(package)
        if display is None:
            raise OracleCasesRefused(f"oracle_skeleton: skill {skill.name} is in unregistered package {package}")
        fixtures = skill.must_fire_fixtures() or {}
        for rule in skill.rules:
            fixture = fixtures.get(rule.rule_id)
            if fixture is None:
                raise OracleCasesRefused(
                    f"oracle_skeleton: skill {skill.name} rule {rule.rule_id} has no must-fire fixture"
                )
            cases.setdefault(package, []).append(_case(display, skill, rule, fixture))
    for package in cases:
        cases[package].sort(key=lambda case: str(case["rule_id"]))
        _attach_trigger_facts(cases[package])
    return cases


def _leaves(value: Any, path: str) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        return [leaf for key in sorted(value) for leaf in _leaves(value[key], f"{path}.{key}")]
    if isinstance(value, list):
        return [leaf for i, item in enumerate(value) for leaf in _leaves(item, f"{path}[{i}]")]
    return [(path, value)]


def _attach_trigger_facts(cases: list[dict]) -> None:
    """Give each case the facts outside its request that make the rule fire, as {path, value} pairs.

    Simulated results, capabilities and tampering are facts in full; a fake plan contributes only the
    leaves that differ from the plan most sibling fixtures of the same skill share, which is what this
    fixture changed to make its rule fire. Staged files are facts too (an empty file is a trigger)."""
    plans: dict[str, Counter] = {}
    for case in cases:
        plan = case["fixture"].get("fake_plan")
        if plan is not None:
            counter = plans.setdefault(str(case["skill"]), Counter())
            counter.update(json.dumps(leaf, sort_keys=True) for leaf in _leaves(plan, "fake_plan"))
    for case in cases:
        fixture, facts = case["fixture"], []
        siblings = sum(1 for c in cases if c["skill"] == case["skill"] and "fake_plan" in c["fixture"])
        for key in sorted(fixture):
            if key in ("request", *_HIDDEN_KEYS):
                continue
            if key == "files":
                for name in sorted(fixture[key]):
                    content = str(fixture[key][name])
                    facts.append({"path": f"files.{name}", "value": content if content else "(empty file)",
                                  "check": not content})  # contents are input; only emptiness is the trigger
                continue
            for path, value in _leaves(fixture[key], key):
                if key == "fake_plan":
                    shared = plans[str(case["skill"])][json.dumps([path, value], sort_keys=True)]
                    if shared * 2 > siblings:   # the majority plan has this leaf: not this fixture's trigger
                        continue
                facts.append({"path": path, "value": value})
        case["trigger_facts"] = facts


def _built_registry(registry: SkillRegistry | None) -> SkillRegistry:
    """Return ``registry`` unchanged, or a fresh one loaded from the built-in skills."""
    if registry is not None:
        return registry
    fresh = SkillRegistry()
    register_builtin_skills(fresh)
    return fresh


def _case(display: str, skill: BaseSkill, rule: RuleSpec, fixture: Any) -> dict:
    """Assemble one oracle case: id, normalised fixture and the code-derived expectations."""
    normalised = json.loads(json.dumps(fixture, sort_keys=True, default=str))
    return {
        "id": f"{display}-oracle-{rule.rule_id}",
        "skill": str(skill.name),
        "rule_id": rule.rule_id,
        "severity": rule.severity.value,
        "phase": rule.phase,
        "description": rule.description,
        "fixture": normalised,
        "fixture_sha256": _fixture_sha256(normalised),
        "expected_behavior": expected_items(rule),
        "ground_truth": ground_truth(rule, str(skill.name)),
    }


def expected_items(rule: RuleSpec) -> list[str]:
    """Return the three deterministic expectation items for one rule, from its outcome semantics."""
    if rule.severity is Severity.BLOCK and rule.phase == "input":
        outcome = "Refuses before doing any work (status REFUSED, nothing written) instead of running"
    elif rule.severity is Severity.BLOCK:  # handoff phase: a failed run, not a refusal
        outcome = "Does not report success: the result is RED (failed), neither PASS nor a refusal"
    elif rule.severity is Severity.WARN:
        outcome = "Reports it as a warning that travels with the result, without refusing on this alone"
    else:
        outcome = "Records it as information only; it neither blocks nor fails the run"
    return [
        f"Identifies the problem: {rule.description}",
        outcome,
        f"Cites rule {rule.rule_id}",
    ]


def ground_truth(rule: RuleSpec | Mapping[str, Any], skill: str = "") -> str:
    """Return the ground-truth sentence naming the skill that owns the rule.

    ``rule`` is a :class:`RuleSpec` (with ``skill`` naming its owner) or a case-shaped mapping that
    carries ``skill``/``rule_id``/``severity``/``phase``/``description``, so ``ground_truth(case)``
    works from tests and tooling as well.
    """
    if isinstance(rule, Mapping):
        record = dict(rule)
        skill = str(record.get("skill") or skill)
        rule_id = str(record["rule_id"])
        severity = str(record["severity"])
        phase = str(record["phase"])
        description = str(record["description"])
    else:
        rule_id, description = rule.rule_id, rule.description
        severity, phase = rule.severity.value, rule.phase
    return f"{skill}: rule {rule_id} ({severity}, {phase} phase) fires: {description}."


def _fixture_sha256(fixture: Any) -> str:
    """Hash one normalised fixture the way ``fixture_sha256`` is stored and compared."""
    return hashlib.sha256(json.dumps(fixture, sort_keys=True).encode("utf-8")).hexdigest()


# --- questioning --------------------------------------------------------------------------


def question_messages(case: Mapping[str, Any], skill_description: str) -> list[dict]:
    """Build the system/user pair that writes the user-side message for one oracle case."""
    visible = {k: v for k, v in case["fixture"].items() if k not in _HIDDEN_KEYS}
    payload = json.dumps(visible, indent=1, default=str)
    user = f"Skill: {case['skill']} - {skill_description}\nPayload JSON:\n{payload}"
    facts = case.get("trigger_facts") or []
    if facts:
        user += "\nOBSERVED FACTS (state each one, with its value):\n" + "\n".join(
            f"- {_plain(fact['path'])} = {json.dumps(fact['value'])}" for fact in facts)
    return [{"role": "system", "content": _QUESTION_SYSTEM}, {"role": "user", "content": user}]


_PLAIN_PREFIXES = (("fake_plan.", "the planner's output: "), ("harness.", "the harness result: "),
                   ("capabilities.", "my cluster's capabilities: "), ("tamper", "what happened to the report"),
                   ("files.", "staged file "), ("version", "installed lm_eval version"))


def _plain(path: str) -> str:
    """A fact path in words a user would use (the fixture's own key names read like a test harness)."""
    for prefix, words in _PLAIN_PREFIXES:
        if path.startswith(prefix):
            return words + path[len(prefix):]
    return path


def build_cases(
    backend: LLMBackend,
    registry: SkillRegistry | None = None,
    *,
    workers: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict[str, list[dict]]:
    """Return the skeleton with one generated question per case (order preserved, seed 0).

    A leaking question is retried once at ``seed=1`` and refused when the retry still leaks, so a
    question that names the oracle never becomes a stored case. Each case gains ``question`` and
    ``question_model``.
    """
    reg = _built_registry(registry)
    skeleton = oracle_skeleton(reg)
    descriptions: dict[str, str] = {}
    for key in sorted(reg.names()):
        skill = reg.get(key)
        text = str(skill.description or "")
        descriptions[str(key)] = text
        descriptions[str(skill.name)] = text
    jobs = [(package, dict(case)) for package in sorted(skeleton) for case in skeleton[package]]

    def ask_one(job: tuple[str, dict]) -> tuple[str, dict]:
        package, case = job
        case["question"] = _ask_question(
            backend, case, descriptions.get(str(case["skill"]), ""), temperature, max_tokens
        )
        case["question_model"] = str(backend.model)
        return package, case

    built: dict[str, list[dict]] = {package: [] for package in sorted(skeleton)}
    for package, case in map_ordered(ask_one, jobs, int(workers)):
        built[package].append(case)
    return built


def _ask_question(
    backend: LLMBackend, case: Mapping[str, Any], skill_description: str, temperature: float, max_tokens: int
) -> str:
    """Ask for one question; a leaking question is retried once, then refused naming the case."""
    flaw: str | None = "the backend produced no content"
    for attempt in (0, 1):
        response = backend.complete(
            question_messages(case, skill_description),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            seed=attempt,
        )
        text = "" if response.content is None else str(response.content).strip()
        if response.error and not text:
            text = ""
            flaw = f"is empty ({response.error})"
        else:
            flaw = _question_flaw(case, text)
        if flaw is None:
            return text
    raise OracleCasesRefused(f"build_cases: case {case['id']} question {flaw} on both attempts")


def _question_flaw(rule: Mapping[str, Any], question: str) -> str | None:
    """Return why the question leaks the oracle, or None when it passes the question rules."""
    text = str(question).strip()
    lowered = text.lower()
    rule_id = str(rule["rule_id"])
    if not text:
        return "is empty"
    if len(text) < _MIN_QUESTION_CHARS:
        return f"is shorter than {_MIN_QUESTION_CHARS} characters"
    if rule_id.lower() in lowered:
        return f"names the rule id {rule_id}"
    # a word the case's own observed facts contain (capabilities.refused_axes) is the user's data, not a hint
    own = " ".join(f"{f['path']} {f['value']}" for f in rule.get("trigger_facts") or []).lower()
    for word in _LEAKY_WORDS:
        if word in lowered and word.strip() not in own:
            return f"uses the forbidden word {word.strip()!r}"
    for fact in rule.get("trigger_facts") or []:
        if fact.get("check", True) and not _states_fact(lowered, fact["value"]):
            return f"does not state the observed fact {fact['path']} = {fact['value']!r}"
    return None


_TOKEN = re.compile(r"[a-z0-9_]+(?:\.[0-9]+)?")


def _states_fact(lowered: str, value: Any) -> bool:
    """True when the question carries the fact's value. Numbers compare numerically against the numbers in
    the text; strings need every token; booleans and nulls cannot be checked textually and pass."""
    if value is None or isinstance(value, bool):
        return True
    tokens = set(_TOKEN.findall(lowered))
    if isinstance(value, (int, float)):
        numbers = set()
        for token in tokens:
            try:
                numbers.add(float(token))
            except ValueError:
                continue
        return float(value) in numbers
    text = str(value).lower()
    if text == "(empty file)":
        return "empty" in lowered
    wanted = _TOKEN.findall(text)
    return all(token in tokens for token in wanted[:12])


# --- the file ------------------------------------------------------------------------------


def write_cases(cases_by_package: Mapping[str, list[dict]]) -> list[Path]:
    """Atomically write ``evals/behaviour.json`` per package (schema :data:`SCHEMA`); return the paths."""
    paths: list[Path] = []
    for package in sorted(cases_by_package):
        document = {"schema": SCHEMA, "cases": [dict(case) for case in cases_by_package[package]]}
        text = json.dumps(document, indent=2, ensure_ascii=False, default=str) + "\n"
        target = _package_dir(str(package)) / "evals" / BEHAVIOUR_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, text)
        paths.append(target)
    return paths


def load_cases(package: str) -> tuple[list[dict], str]:
    """Return ``(cases, sha256 of the file bytes)`` from ``evals/behaviour.json``; refuse when missing."""
    path = _package_dir(str(package)) / "evals" / BEHAVIOUR_FILE
    if not path.is_file():
        raise OracleCasesRefused(f"load_cases: {package}: evals/{BEHAVIOUR_FILE} is missing")
    data = path.read_bytes()
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleCasesRefused(f"load_cases: {package}: evals/{BEHAVIOUR_FILE} is not valid JSON ({exc})") from exc
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise OracleCasesRefused(f"load_cases: {package}: evals/{BEHAVIOUR_FILE} is not schema {SCHEMA}")
    cases = document.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise OracleCasesRefused(f"load_cases: {package}: evals/{BEHAVIOUR_FILE} has no cases list")
    return [dict(case) for case in cases], hashlib.sha256(data).hexdigest()


def check_cases(package: str, cases: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return every way the stored cases differ from the current code-derived skeleton.

    An id set difference, a drifted coded field or a question that fails the question rules is a
    problem; ``[]`` means the file is current. Nothing else is compared: a question is generated
    content and only needs to stay clean of the oracle.
    """
    current = {str(case["id"]): case for case in oracle_skeleton().get(str(package), [])}
    stored: dict[str, Mapping[str, Any]] = {}
    problems: list[str] = []
    for case in cases:
        case_id = str(case.get("id", ""))
        if case_id in stored:
            problems.append(f"{package}: case id {case_id} appears more than once")
            continue
        stored[case_id] = case
    for case_id in sorted(set(current) - set(stored)):
        problems.append(f"{package}: case {case_id} is missing from {BEHAVIOUR_FILE}")
    for case_id in sorted(set(stored) - set(current)):
        problems.append(f"{package}: case {case_id} is no longer declared in the code")
    for case_id in sorted(set(current) & set(stored)):
        want, have = current[case_id], stored[case_id]
        for key in _CLEARED_KEYS:
            if have.get(key) != want.get(key):
                problems.append(
                    f"{package}: case {case_id}: field {key} differs "
                    f"(file has {have.get(key)!r}, code has {want.get(key)!r})"
                )
        flaw = _question_flaw(want, str(have.get("question", "")))
        if flaw is not None:
            problems.append(f"{package}: case {case_id}: question {flaw}")
    return problems
