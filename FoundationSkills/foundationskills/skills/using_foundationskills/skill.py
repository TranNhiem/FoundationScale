"""UsingFoundationSkillsSkill: the FoundationSkills routing entry point (A3.2 meta skill).

Exit contract: 96 REFUSED writes nothing and names the missing input ("want" names
an artifact type no registered skill produces, or `plan_chain` finds no route);
95 UNMEASURED when the routing answer is unmeasured ("intent" matches no registered
skill); 0 PASS when the route is measured against the registered skill list. The
skill writes one skill_route artifact and declares no handoff rule, so RED 5 never fires here.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

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
from foundationskills.core.registry import SkillRegistry
from foundationskills.core.status import Status, worst
from foundationskills.skills import register_builtin_skills

_WORDS = re.compile(r"[A-Za-z0-9]+")


def words(text: str) -> set[str]:
    """Lower-cased word atoms, ignoring words of 3 letters or fewer (the documented ranking)."""
    return {w for w in _WORDS.findall(str(text).lower()) if len(w) > 3}


def _worst(*statuses: Status) -> Status:
    """Aggregate with core `worst` under either call shape (var-positional statuses or one iterable)."""
    seq = tuple(statuses)
    if len(seq) == 1:
        return seq[0]
    try:
        return worst(*seq)
    except TypeError:  # worst(statuses) accepts one iterable of statuses
        return worst(seq)  # type: ignore[arg-type]


class UsingFoundationSkillsSkill(BaseSkill):
    name = "using_foundationskills"
    version = "0.1.0"
    description = (
        "Entry point for FoundationSkills: routes a user's intent (or a have→want chain, e.g. "
        "{raw_data_ref, goal_spec} to fs_launch_spec) to the right FoundationSkills skill, lists every "
        "registered skill, and states the workflow and the non-negotiable rules the workflow must keep. "
        "Do NOT use for building datasets (use fskills-data-engine), vetting a plan draft against a "
        "probe (use fskills-probe-first), launching training (fskills-training) or checkpoint scoring "
        "(fskills-evaluation): it routes and documents, runs no measured work, writes only a skill_route artifact and "
        "never invents a capability."
    )
    scope = Scope(  # routes for every skill, so it spans every axis the catalogue does
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ()
    produces = ("skill_route",)
    invocation = "user"
    invokes = ()

    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {"type": "string", "minLength": 1},
            "have": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "want": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["skills", "candidates", "chain"],
        "additionalProperties": True,
        "properties": {
            "skills": {"type": "array"},
            "candidates": {"type": "array"},
            "chain": {"type": "array"},
        },
    }
    rules = (
        RuleSpec("UF-IN-001", "\"want\" names an artifact type no registered skill produces", Severity.BLOCK, "input"),
        RuleSpec("UF-IN-002", "\"intent\" matches no registered skill", Severity.WARN, "input"),
    )
    fs_interface = FSInterface(
        entries=("fskills skills",),
        emits=(),
        apis=("SkillRegistry.plan_chain",),
        notes="routing entry point: lists registered skills and chains them; never runs measured work",
    )

    # -- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _registry() -> tuple[SkillRegistry, list[Any]]:
        """A fresh SkillRegistry with every built-in skill registered (the registry is the measured source here)."""
        registry = SkillRegistry()
        register_builtin_skills(registry)
        getter = getattr(registry, "get", None) or getattr(registry, "__getitem__", None)
        skills = [getter(name) for name in sorted(set(registry.names()))] if callable(getter) else []
        return registry, [skill for skill in skills if skill is not None]

    @staticmethod
    def _entry(skill: Any) -> dict[str, Any]:
        """One routing payload row per registered skill."""
        return {
            "name": getattr(skill, "name", ""),
            "invocation": getattr(skill, "invocation", "model"),
            "invokes": [str(v) for v in (getattr(skill, "invokes", ()) or ())],
            "consumes": [str(v) for v in (getattr(skill, "consumes", ()) or ())],
            "produces": [str(v) for v in (getattr(skill, "produces", ()) or ())],
            "description": str(getattr(skill, "description", "")),
        }

    @staticmethod
    def _candidates(intent: str, skills: list[Any]) -> list[dict[str, Any]]:
        """A lexical shortlist, not an authoritative router: skill packages ranked by the IDF-weighted overlap of
        >3-letter words between `intent` and each package's SKILL.md description plus when_to_use phrasings.

        Measured 2026-10-18 on the 98 should-trigger routing questions of the 8 model-invoked packages (top-1):
        plain overlap 71/98, IDF-weighted 76/98; adding the routing questions themselves lowered it (69/98).
        Top 3, score > 0 only, so the absence of a match is never PASS. Names are SKILL.md names.
        """
        import math  # noqa: PLC0415

        from foundationskills.agent.routing_eval import load_index  # noqa: PLC0415

        registry = SkillRegistry()
        for skill in skills:
            if skill.name not in registry.names():
                registry.register(skill)
        docs: dict[str, set[str]] = {}
        for entry in load_index(registry):
            if entry["name"] == "using-foundationskills":
                continue  # the router never routes to itself
            phrasings = entry["when_to_use"]
            text = " ".join(phrasings) if isinstance(phrasings, list) else str(phrasings)
            docs[str(entry["name"])] = words(f"{entry['description']} {text}")
        frequency: dict[str, int] = {}
        for package_words in docs.values():
            for word in package_words:
                frequency[word] = frequency.get(word, 0) + 1
        intent_words = words(intent)
        scored = []
        for name, package_words in docs.items():
            score = sum(math.log(len(docs) / frequency[w]) for w in intent_words & package_words)
            if score > 0:
                scored.append({"name": name, "score": round(score, 3)})
        scored.sort(key=lambda row: (-row["score"], row["name"]))
        return scored[:3]

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        _, skills = self._registry()
        want = request.get("want")
        if want:
            producible = {p for skill in skills for p in getattr(skill, "produces", ()) or ()}
            if want not in producible:
                findings.append(self.finding(
                    "UF-IN-001",
                    f"missing input: want {want!r} is not an artifact type any registered skill produces, "
                    "so no chain leads to it",
                    {"want": str(want), "producible": sorted(producible)},
                    "name a `want` the registered skills produce (see `fskills skills` or SKILLS.md), e.g. fs_launch_spec",
                ))
        intent = request.get("intent")
        if intent and not self._candidates(str(intent), skills):
            findings.append(self.finding(
                "UF-IN-002",
                f"no registered skill matches the intent {intent!r}, so the routing answer is unmeasured",
                {"intent": str(intent)},
                "restate the intent against the skill list (this payload's `skills`), or ask the user which skill they mean",
            ))
        return findings

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        registry, skills = self._registry()
        rows = [self._entry(skill) for skill in skills]
        intent = request.get("intent")
        candidates = self._candidates(str(intent), skills) if intent else []

        caveats: list[str] = []
        findings: list[Finding] = []
        if intent and not candidates:  # UF-IN-002 itself comes from check_inputs; here it only lowers the status
            caveats.append(f"no registered skill matches the intent {intent!r}")

        chain: list[str] = []
        have = request.get("have") or ()
        want = request.get("want")
        if have and want:
            try:
                chain = [str(getattr(item, "name", item)) for item in registry.plan_chain(set(map(str, have)), str(want))]
            except Exception as exc:  # noqa: BLE001 - plan_chain names its own failure; surface it as UF-IN-001
                finding = self.finding(
                    "UF-IN-001",
                    f"missing input: no chain from {sorted(set(map(str, have)))} to {want} ({exc})",
                    {"have": sorted(set(map(str, have))), "want": str(want)},
                    "drop the unreachable `want`, or add the artifacts the chain needs to `have`",
                )
                return self._refused(f"UF-IN-001: {finding.message}", [finding])

        payload = {"skills": rows, "candidates": list(candidates), "chain": chain}
        provenance = make_provenance(
            self.name, self.version,
            {
                "intent": str(request.get("intent") or ""),
                "have": ",".join(str(v) for v in have),
                "want": str(want or ""),
            },
        )
        status = _worst(Status.PASS, *([Status.UNMEASURED] * len(caveats)))
        next_actions = () if status is Status.PASS else (
            "restate the intent against the registered skill list, then re-run using-foundationskills",)
        artifact = Artifact("skill_route", "route", payload, provenance, 1)
        ref = write_artifact(artifact, request.get("out") or ctx.artifacts_dir)
        return SkillResult(
            status=status, payload=payload, artifacts=(ref,), findings=tuple(findings),
            next_actions=next_actions, provenance=provenance,
        )

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        return []  # the skill_route payload is schema-validated on write; no handoff rule is declared

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"using-foundationskills {type(failure).__name__}: {failure}"
                   if isinstance(failure, BaseException)
                   else f"using-foundationskills {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                "the requested `want` is not an artifact type any registered skill produces",
                "the `intent` names work outside every skill's scope",
            ),
            checks=(
                "read the routing payload's `skills` list (or run `fskills skills`)",
                "check `SkillRegistry.plan_chain(have, want)` for the chain you expected",
            ),
            recovery=(
                "name a `want` that a registered skill produces (see SKILLS.md)",
                "restate the intent against the skill list and re-run",
            ),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "UF-IN-001": {
                "request": {"want": "no_such_type"},
                "expected": "REFUSED 96 naming UF-IN-001 (want names an artifact type no registered skill produces)",
            },
            "UF-IN-002": {
                "request": {"intent": "zzqx qqzz wwyy"},
                "expected": "UNMEASURED 95 naming UF-IN-002 (the intent matches no registered skill)",
            },
        }

