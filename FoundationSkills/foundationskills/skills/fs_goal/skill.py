"""FsGoalSkill: the goal-spec grilling composer (A3.2 user-invoked workflow skill, /fs-goal).

Exit contract: 96 REFUSED writes nothing and names the missing input (FG-IN-001: no "answers" object in
the request -- fs_goal grills what the user actually said and never invents an objective, a base model or
data sources). 95 UNMEASURED while the grilling frontier is not empty (FG-HO-001: load-bearing facts are
missing and the payload carries one question per missing fact) or while the assembled goal is unconfirmed
(FG-HO-002: the payload carries the goal_spec_preview and the confirm question). 5 RED only from
check_handoff on a PASS result whose payload flag goal_spec_valid is False (FG-HO-003: goal_from_answers
raised or the assembled goal_spec is invalid). 0 PASS only when the frontier is empty, the user confirmed,
and goal_from_answers builds a valid goal_spec -- that goal_spec is the one artifact written. run() never
returns Status.RED (the handoff BLOCK travels through the payload flag), and ``invokes`` is empty, so no
composed skill's rule id can leak into the findings and no other run_write can land in our directory.
"""
from __future__ import annotations

from typing import Any

from foundationskills.agent.intake import goal_from_answers, missing_facts
from foundationskills.core.artifacts import Artifact, write_artifact
from foundationskills.core.contract import (
    BaseSkill,
    Diagnosis,
    Finding,
    FSInterface,
    RuleSpec,
    Scope,
    Severity,
    SkillContext,
    SkillResult,
)
from foundationskills.core.provenance import make_provenance
from foundationskills.core.status import Status

DEFAULT_SKILLS: tuple[str, ...] = ("training.planner", "data_engine")

_CONFIRM_QUESTION: dict[str, Any] = {
    "key": "confirm",
    "question": "Is this goal correct? Answer yes to confirm.",
    "recommended": "yes",
}

# The full intake frontier answered once (fixtures and tests rest on this exact set).
_COMPLETE_ANSWERS: dict[str, Any] = {
    "objective": "answer Turkish medical board questions with short explanations",
    "capabilities": ["medical QA in Turkish"],
    "goal": "domain_expert",
    "domain": "medicine",
    "preserve_general": True,
    "base_model": {"name_or_path": "hf/Qwen2.5-7B", "size_b": 7, "arch": "dense"},
    "sources": [{"uri": "s3://data/med-qa-tr.jsonl", "kind": "qa", "approx_tokens": 120000000}],
    "hardware": {"gpu": "gb200", "gpus_per_node": 8, "nodes": 2},
    "data_facts": {"has_verifiable_answers": True, "has_pairs": False, "has_instructions": True},
    "rl_algorithm": "dr_grpo",
}

# Same answers, but the one source carries no `kind`: missing_facts is empty and goal_from_answers raises.
_BROKEN_ANSWERS: dict[str, Any] = {
    **_COMPLETE_ANSWERS,
    "sources": [{"uri": "s3://data/med-qa-tr.jsonl", "approx_tokens": 120000000}],
}


class FsGoalSkill(BaseSkill):
    name = "fs_goal"
    version = "0.1.0"
    description = (
        "User-invoked slash command: run ONLY when the user types /fs-goal; never choose it on your own. "
        "Grilling composer (/fs-goal) for one FoundationSkills goal_spec: takes the intake answers so far "
        "({objective, goal?, domain?, preserve_general?, base_model {name_or_path, size_b?, arch?}, "
        "sources [{uri, kind, ...}], hardware {gpu, gpus_per_node, nodes}, data_facts {has_verifiable_answers, "
        "has_pairs, has_instructions}, rl_algorithm?}), asks ONLY for the facts intake.missing_facts still "
        "names and then one confirm question, and writes one goal_spec artifact only when the frontier is "
        "empty AND confirmed is true. Do NOT use to plan, train, launch, fix or explain a run: this skill "
        "only collects and confirms intake facts (fskills-training plans). Side-effect free: composes and "
        "invokes nothing (invokes is empty) and writes only its own goal_spec."
    )
    scope = Scope(  # any LLM/VLM goal can be grilled, so it spans every axis the catalogue does
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ()
    produces = ("goal_spec",)
    invocation = "user"
    invokes = ()
    requires_fs = False

    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answers": {"type": "object"},
            "confirmed": {"type": ["boolean", "string"]},
            "skills": {"type": "array", "items": {"type": "string"}},
            "record_dir": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["phase"],
        "additionalProperties": True,
        "properties": {
            "phase": {"type": "string", "enum": ["questions", "reject", "confirm", "complete"]},
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["key", "question", "recommended"],
                    "properties": {
                        "key": {"type": "string"},
                        "question": {"type": "string"},
                        "recommended": {"type": ["string", "null"]},
                    },
                },
            },
            "missing_facts": {"type": "array"},
            "goal_spec_preview": {"type": "object"},
            "goal_spec": {"type": "object"},
            "goal_spec_valid": {"type": "boolean"},
            "goal_spec_error": {"type": ["string", "null"]},
        },
    }
    rules = (
        RuleSpec("FG-IN-001", "no \"answers\" object was given, so fs_goal has nothing to grill and invents nothing",
                 Severity.BLOCK, "input"),
        RuleSpec("FG-HO-001", "load-bearing facts are missing from the answers so far (the grilling frontier is not empty)",
                 Severity.WARN, "handoff"),
        RuleSpec("FG-HO-002", "the assembled goal is complete but the user has not confirmed it",
                 Severity.WARN, "handoff"),
        RuleSpec("FG-HO-003", "goal_from_answers raised or the assembled goal_spec is invalid",
                 Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("fskills goal",),
        emits=(),
        apis=(),
        notes=(
            "side-effect free griller: reads the intake answers and foundationskills.agent.intake "
            "(missing_facts, goal_from_answers) and writes one goal_spec artifact to request.record_dir "
            "or ctx.artifacts_dir"
        ),
    )

    # -- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _no_answers_message(request: dict[str, Any]) -> str:
        why = ("no `answers` key in the request" if "answers" not in request
               else f"`answers` is {type(request.get('answers')).__name__}, not an object")
        return (f"missing input: no \"answers\" object was given ({why}) — fs_goal grills what the user "
                f"actually said and never invents an objective, a base model or data sources")

    def _no_answers_finding(self, request: dict[str, Any]) -> Finding:
        return self.finding(
            "FG-IN-001",
            self._no_answers_message(request),
            {"answers_type": type(request.get("answers")).__name__, "keys": sorted(str(k) for k in request)},
            "pass `answers` (an object: the intake answer keys of goal_from_answers, holding whatever the "
            "user said so far) and re-run /fs-goal — `confirmed` and `skills` are optional",
        )

    @staticmethod
    def _skills(request: dict[str, Any]) -> list[str]:
        picked = [str(s) for s in (request.get("skills") or list(DEFAULT_SKILLS))]
        return picked or list(DEFAULT_SKILLS)

    @staticmethod
    def _confirmed(request: dict[str, Any]) -> bool:
        """True only on an explicit confirmation (bool true or a 'yes' the user answered to the confirm question)."""
        raw = request.get("confirmed")
        if isinstance(raw, bool):
            return raw
        return str(raw or "").strip().lower() in {"yes", "y", "true", "confirmed"}

    @staticmethod
    def _recommended(fact: dict[str, Any]) -> str | None:
        """A fact's default/example when REQUIRED_FACTS gives one, else None (the question then asks for the value)."""
        for key in ("default", "example", "recommended", "value"):
            if fact.get(key) not in (None, "", [], {}):
                return str(fact[key])
        return None

    @classmethod
    def _question(cls, fact: dict[str, Any]) -> dict[str, Any]:
        """One grilling question: answering 'yes' accepts `recommended`; without one the question asks for the value."""
        recommended = cls._recommended(fact)
        ask = str(fact.get("question") or f"what is {fact.get('key')}?").strip()
        if recommended is None:
            wording = ask if ask.endswith(("?", ".", "!")) else f"{ask}?"
        else:
            wording = (f"{ask.rstrip('?')} == {recommended!r}? Answer 'yes' to accept {recommended!r} — "
                       f"or reply with a different value.")
        return {"key": str(fact.get("key") or ""), "question": wording, "recommended": recommended}

    @staticmethod
    def _goal_so_far(answers: dict[str, Any]) -> dict[str, Any]:
        """The partial goal_spec the answers would build (never raises): the frontier is measured on THIS.

        Spec-silent choice: intake answer keys are mapped onto goal_spec paths here (sources ->
        data.sources, capabilities -> target_capabilities), and a payload already carrying goal_spec
        paths is read back the same way, so `missing_facts` sees one dotted-key space either way.
        """
        out: dict[str, Any] = {}
        for key in ("objective", "goal", "domain", "preserve_general", "rl_algorithm"):
            if answers.get(key) not in (None, "", [], {}):
                out[key] = answers[key]
        base = answers.get("base_model") if isinstance(answers.get("base_model"), dict) else {}
        if base:
            out["base_model"] = {k: v for k, v in base.items() if v not in (None, "", [], {})}
        hw = answers.get("hardware") if isinstance(answers.get("hardware"), dict) else {}
        if hw:
            out["hardware"] = {k: v for k, v in hw.items() if v not in (None, "", [], {})}
        facts = answers.get("data_facts") if isinstance(answers.get("data_facts"), dict) else {}
        if facts:
            out["data_facts"] = {k: v for k, v in facts.items() if v is not None}
        data = answers.get("data") if isinstance(answers.get("data"), dict) else {}
        sources = answers.get("sources") or data.get("sources")
        if sources:
            out["data"] = {"sources": list(sources)}
        caps = answers.get("capabilities") or answers.get("target_capabilities")
        if caps:
            out["target_capabilities"] = list(caps)
        return out

    @staticmethod
    def _answers_for_goal(answers: dict[str, Any]) -> dict[str, Any]:
        """Raw intake answers as goal_from_answers wants them: a goal_spec-shaped input is read back to it."""
        out = dict(answers)
        data = answers.get("data") if isinstance(answers.get("data"), dict) else {}
        if not out.get("sources") and data.get("sources"):
            out["sources"] = data["sources"]
        if not out.get("capabilities") and answers.get("target_capabilities"):
            out["capabilities"] = answers["target_capabilities"]
        return out

    @staticmethod
    def _invalid_goal(goal: Any) -> str | None:
        """None when `goal` is a usable goal_spec payload, else the structural fault (FG-HO-003)."""
        if not isinstance(goal, dict):
            return f"the assembled goal_spec is {type(goal).__name__}, not an object"
        if not str(goal.get("objective") or "").strip():
            return "the assembled goal_spec has no objective"
        base = goal.get("base_model")
        if not isinstance(base, dict) or not str(base.get("name_or_path") or "").strip():
            return "the assembled goal_spec has no base_model.name_or_path"
        data = goal.get("data")
        sources = data.get("sources") if isinstance(data, dict) else None
        if not isinstance(sources, list) or not sources:
            return "the assembled goal_spec has no data.sources"
        for index, src in enumerate(sources):
            if not isinstance(src, dict) or not str(src.get("uri") or "") or not str(src.get("kind") or ""):
                return f"the assembled goal_spec data.sources[{index}] has no uri and kind"
        hw = goal.get("hardware")
        if not isinstance(hw, dict):
            return "the assembled goal_spec has no hardware"
        for key in ("gpu", "nodes", "gpus_per_node"):
            if hw.get(key) is None:
                return f"the assembled goal_spec hardware.{key} is missing"
        return None

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        if not isinstance(request.get("answers"), dict):
            findings.append(self._no_answers_finding(request))
        return findings

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        answers = request.get("answers")
        provenance = make_provenance(
            self.name, self.version,
            {
                "skills": ",".join(self._skills(request)),
                "confirmed": str(self._confirmed(request)),
                "answers_keys": ",".join(sorted(str(k) for k in answers)) if isinstance(answers, dict) else "",
            },
        )
        if not isinstance(answers, dict):  # FG-IN-001 the moment the base lets run() see it
            finding = self._no_answers_finding(request)
            return SkillResult(
                status=Status.REFUSED, payload={}, artifacts=(), findings=(finding,),
                refusal=finding.message, next_actions=(), provenance=provenance,
            )

        skills = self._skills(request)
        confirmed = self._confirmed(request)
        goal_so_far = self._goal_so_far(answers)
        missing = missing_facts(goal_so_far or dict(answers), skills)

        if missing:  # FG-HO-001: the frontier is not empty, so the grilling is not done
            load_bearing = [m["key"] for m in missing if m.get("load_bearing")]
            head = (f"load-bearing facts are missing ({', '.join(load_bearing)})"
                    if load_bearing else "facts the intake must record are still missing")
            message = (f"unmeasured: {head} — {len(missing)} question(s) left before the frontier is "
                       f"empty ({', '.join(m['key'] for m in missing)}); no goal_spec is written")
            payload = {
                "phase": "questions",
                "missing_facts": [
                    {"key": m["key"], "skill": m.get("skill", ""), "load_bearing": bool(m.get("load_bearing"))}
                    for m in missing
                ],
                "questions": [self._question(m) for m in missing],
            }
            finding = self.finding(
                "FG-HO-001", message,
                {"missing": [m["key"] for m in missing], "load_bearing": load_bearing, "skills": skills},
                "answer the questions in the payload (reply 'yes' to accept a recommended value), merge the "
                "answers and re-run /fs-goal",
            )
            return SkillResult(
                status=Status.UNMEASURED, payload=payload, artifacts=(), findings=(finding,),
                next_actions=(), provenance=provenance,
            )

        # frontier empty: what does goal_from_answers make of exactly these answers?
        goal: dict[str, Any] | None = None
        error: str | None = None
        try:
            goal = goal_from_answers(self._answers_for_goal(dict(answers)))
            error = self._invalid_goal(goal)
        except Exception as exc:  # noqa: BLE001 - goal_from_answers raising is exactly FG-HO-003
            error = f"goal_from_answers raised {type(exc).__name__}: {exc}"

        if error is not None or goal is None:  # the flag check_handoff turns into RED 5 (FG-HO-003)
            payload = {
                "phase": "reject",
                "goal_spec_valid": False,
                "goal_spec_error": error or "goal_from_answers produced no goal_spec",
                "missing_facts": [],
                "questions": [{
                    "key": "answers",
                    "question": f"{error or 'the assembled goal_spec is not valid'} — which answer is wrong? "
                                f"Reply with the corrected `answers`.",
                    "recommended": None,
                }],
            }
            return SkillResult(
                status=Status.PASS, payload=payload, artifacts=(), findings=(),
                next_actions=(), provenance=provenance,
            )

        if not confirmed:  # FG-HO-002: complete, whole, unconfirmed -- never a pass
            payload = {
                "phase": "confirm",
                "goal_spec_preview": goal,
                "goal_spec_valid": True,
                "goal_spec_error": None,
                "missing_facts": [],
                "questions": [dict(_CONFIRM_QUESTION)],
            }
            finding = self.finding(
                "FG-HO-002",
                "unmeasured: the goal is complete but the user has not confirmed it — the goal_spec_preview "
                "is in the payload and no goal_spec is written until they answer 'yes'",
                {"confirmed": False, "objective": str(goal.get("objective") or "")},
                "read goal_spec_preview and re-run /fs-goal with confirmed: true (answer 'yes' to "
                "«Is this goal correct? Answer yes to confirm.»)",
            )
            return SkillResult(
                status=Status.UNMEASURED, payload=payload, artifacts=(), findings=(finding,),
                next_actions=(), provenance=provenance,
            )

        artifact = Artifact("goal_spec", "goal_spec", goal, provenance, 1)
        ref = write_artifact(artifact, request.get("record_dir") or ctx.artifacts_dir)
        payload = {
            "phase": "complete",
            "goal_spec": goal,
            "goal_spec_valid": True,
            "goal_spec_error": None,
            "missing_facts": [],
            "questions": [],
        }
        return SkillResult(
            status=Status.PASS, payload=payload, artifacts=(ref,), findings=(),
            next_actions=(), provenance=provenance,
        )

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """The only place FG-HO-003 fires: run() passes and flags the invalid goal, check_handoff turns it RED."""
        if result.status is not Status.PASS:
            return []
        payload = result.payload or {}
        if payload.get("goal_spec_valid") is not False:
            return []
        error = str(payload.get("goal_spec_error") or "the assembled goal_spec is not valid")
        return [self.finding(
            "FG-HO-003",
            f"{error} — the frontier is empty and the answers were confirmed, so one answer contradicts the "
            f"intake contract and no goal_spec is written",
            {"goal_spec_error": error},
            "correct the answer the message names (every source needs uri and kind, hardware needs "
            "gpu/gpus_per_node/nodes, objective and base_model.name_or_path must be set) and re-run /fs-goal",
        )]

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"fs_goal {type(failure).__name__}: {failure}"
                   if isinstance(failure, BaseException)
                   else f"fs_goal {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                "no \"answers\" object was given (FG-IN-001): fs_goal grills what the user said and invents nothing",
                "the frontier is not empty (FG-HO-001) or the goal is unconfirmed (FG-HO-002): the grilling is not done",
                "goal_from_answers raised or the assembled goal_spec is invalid (FG-HO-003): one answer contradicts the intake contract",
            ),
            checks=(
                "read the FG-* finding message and the payload `questions` (each names one fact and asks exactly for it)",
                "check `answers` against the intake keys of goal_from_answers (objective, base_model.name_or_path, sources[].uri+kind, hardware.gpu/gpus_per_node/nodes)",
            ),
            recovery=(
                "answer the questions in the payload (reply 'yes' to accept a recommended value) and re-run /fs-goal with the merged answers",
                "confirm the preview with confirmed: true; correct the answer FG-HO-003 names and re-run",
            ),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "FG-IN-001": {
                "request": {"skills": ["training.planner"], "confirmed": True},
                "expected": "REFUSED 96 naming FG-IN-001 (no \"answers\" object was given: nothing written, refusal names the missing input)",
            },
            "FG-HO-001": {
                "request": {"answers": {"objective": "answer Turkish medical board questions"}},
                "expected": "UNMEASURED 95 naming FG-HO-001 with one {key, question, recommended} entry per missing fact and no goal_spec written",
            },
            "FG-HO-002": {
                "request": {"answers": _COMPLETE_ANSWERS},
                "expected": "UNMEASURED 95 naming FG-HO-002 with goal_spec_preview and the confirm question (recommended 'yes') and no goal_spec written",
            },
            "FG-HO-003": {
                "request": {"answers": _BROKEN_ANSWERS, "confirmed": True},
                "expected": "RED 5 naming FG-HO-003 (goal_from_answers raised: the one source has no `kind`), fired by check_handoff on the PASS result and no goal_spec written",
            },
        }
