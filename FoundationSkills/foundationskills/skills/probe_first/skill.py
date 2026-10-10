"""ProbeFirstSkill: a FoundationScale plan draft vetted against the measured probe.

Exit contract: 96 REFUSED writes nothing and names the missing input (the probe
was skipped, the plan claims an unmeasured capability as available/PASS, or no
checks are named); 95 UNMEASURED when the measured probe is incomplete (FS
unavailable, probe errors, or a needed parallel axis is unmeasured) so the
vetted report cannot PASS; 5 RED when the emitted probe_report is missing or
altered on disk at handoff; 0 PASS only when every requested check runs against
a clean measured probe.
"""
from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from foundationskills.core.artifacts import Artifact, ArtifactRef, write_artifact
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
from foundationskills.core.provenance import make_provenance, sha256_file
from foundationskills.core.status import Status, worst
from foundationskills.interfaces.fs.capabilities import FSCapabilities

CHECK_KEYS = frozenset(
    {"stage", "algorithm", "backend", "tp", "pp", "ep", "cp",
     "multi_gpu_rl", "require_checkpoint", "answer_kind"}
)
PARALLEL_AXES = ("tp", "pp", "ep", "cp")
_TUPLE_FIELDS = frozenset(
    {"train_objectives", "sharding_strategies", "executed_axes", "refused_axes",
     "rl_algorithms", "rl_reward_kinds", "backends", "notes", "errors"}
)
_MAP_TUPLE_FIELDS = frozenset({"families", "train_flag_choices"})
_MAP_FIELDS = frozenset({"rl_runnable", "pref_runnable"})
_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_CHECK_ITEM: dict[str, Any] = {
    "type": "object",
    "required": ["stage"],
    "additionalProperties": False,
    "properties": {
        "stage": {"type": "string", "minLength": 1},
        "algorithm": {"type": "string", "minLength": 1},
        "backend": {"type": "string", "minLength": 1},
        "tp": {"type": "integer"},
        "pp": {"type": "integer"},
        "ep": {"type": "integer"},
        "cp": {"type": "integer"},
        "multi_gpu_rl": {"type": "boolean"},
        "require_checkpoint": {"type": "boolean"},
        "answer_kind": {"type": "string", "minLength": 1},
    },
}


def unwrap_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return an artifact envelope's payload, or the dict itself when it is a bare payload JSON."""
    payload = data.get("payload")
    if "type" in data and isinstance(payload, dict):
        return dict(payload)
    return dict(data)


def load_json_payload(path: str | Path) -> dict[str, Any]:
    """Read a JSON object (artifact envelope `training_plan.<id>.json` or bare payload) and unwrap it."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a JSON object")
    return unwrap_payload(raw)


def probe_payload(**overrides: Any) -> dict[str, Any]:
    """A measured probe payload exactly as `fskills probe [--deep] --json` prints it (FSCapabilities.to_dict())."""
    kwargs: dict[str, Any] = {"available": True, "fs_version": "x", "backends": ("fsdp",), "axes_measured": False}
    kwargs.update(overrides)
    return FSCapabilities(**kwargs).to_dict()


def caps_from_probe(data: Mapping[str, Any]) -> FSCapabilities:
    """Rebuild FSCapabilities from a probe JSON dict (lists back to tuples; unknown keys ignored)."""
    known = {f.name for f in fields(FSCapabilities)}
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        if name == "train_flags":
            kwargs[name] = frozenset(str(v) for v in (value or ()))
        elif name in _TUPLE_FIELDS:
            kwargs[name] = tuple(value or ())
        elif name in _MAP_TUPLE_FIELDS:
            kwargs[name] = {str(k): tuple(v or ()) for k, v in dict(value or {}).items()}
        elif name in _MAP_FIELDS:
            kwargs[name] = {str(k): (None if v is None else str(v)) for k, v in dict(value or {}).items()}
        else:
            kwargs[name] = value
    kwargs.setdefault("available", False)
    kwargs.setdefault("fs_version", None)
    return FSCapabilities(**kwargs)


def run_check(caps: FSCapabilities, check: Mapping[str, Any]) -> str | None:
    """Measured verdict for one check: None when FS runs it (so it can PASS), else its 'missing: …' reason.

    `FSCapabilities.check` returns the FIRST gap it finds, so a caller that wants a
    specific gap (e.g. the unmeasured axis) must build a probe that leaves only that
    gap possible.
    """
    try:
        return caps.check(**dict(check))
    except TypeError as exc:  # defensive: check keys are schema- and normalization-clamped on the way in
        return f"missing: unusable check {dict(check)!r} ({exc})"


def plan_claims_available(plan_payload_: Mapping[str, Any], check: Mapping[str, Any]) -> bool:
    """True when the plan draft lists this same check as available/PASS ("treat UNMEASURED as PASS")."""
    for entry in plan_payload_.get("checks") or ():
        if isinstance(entry, Mapping) and all(entry.get(k) == v for k, v in check.items()):
            return entry.get("status") == "PASS" or entry.get("available") is True
    return False


def _fmt_check(check: Mapping[str, Any]) -> str:
    return json.dumps(dict(check), sort_keys=True, separators=(",", ":"))


def _worst(*statuses: Status) -> Status:
    """Aggregate with core `worst` under either call shape (var-positional statuses or one iterable)."""
    seq = tuple(statuses)
    if len(seq) == 1:
        return seq[0]
    try:
        return worst(*seq)
    except TypeError:  # worst(statuses) accepts one iterable of statuses
        return worst(seq)


class ProbeFirstSkill(BaseSkill):
    name = "probe_first"
    version = "0.1.0"
    description = (
        "Refuse a FoundationScale plan draft written before the measured probe (fskills probe --deep --json) "
        "or claiming a capability the probe cannot run, and emit a vetted probe_report bound to the plan draft."
    )
    scope = Scope(
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("training_plan",)
    produces = ("probe_report",)

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["plan_draft"],
        "additionalProperties": False,
        "properties": {
            "plan_draft": {"type": "string", "minLength": 1},
            "probe_report": {"type": "string", "minLength": 1},
            "checks": {"type": "array", "minItems": 1, "items": _CHECK_ITEM},
            "out": {"type": "string", "minLength": 1},
            "artifact_id": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["plan_draft", "checks", "capabilities", "verdict"],
        "additionalProperties": True,
        "properties": {
            "plan_draft": {"type": "string"},
            "checks": {"type": "array"},
            "capabilities": {"type": "object", "required": ["available"], "additionalProperties": True},
            "verdict": {"enum": ["PASS", "UNMEASURED"]},
        },
    }
    rules = (
        RuleSpec("PF-IN-001", "probe_report absent, unreadable, or not a measured probe JSON (the probe was skipped)", Severity.BLOCK, "input"),
        RuleSpec("PF-IN-002", "a requested check is missing/unmeasured on the measured probe, or the plan claims it available/PASS", Severity.BLOCK, "input"),
        RuleSpec("PF-IN-003", "no requested checks named (the plan names none and request checks is absent)", Severity.BLOCK, "input"),
        RuleSpec("PF-HO-001", "the vetted report cannot PASS (probe unavailable, probe errors, or a needed unmeasured axis)", Severity.WARN, "handoff"),
        RuleSpec("PF-HO-002", "the emitted probe_report artifact is missing or altered on disk at handoff", Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(
        entries=(),
        emits=("probe_report",),
        apis=("fskills probe --deep --json",),
        notes="measured capabilities per docs/ARCHITECTURE.md:57-67; never probes itself",
    )

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        probe: dict[str, Any] | None
        try:
            probe = self._probe_payload(request)
        except (OSError, ValueError, TypeError) as exc:
            probe = None
            findings.append(self.finding(
                "PF-IN-001",
                f"missing input: no usable probe JSON ({exc}) — the probe was skipped",
                {"probe_report": request.get("probe_report")},
                "probe first: run `fskills probe --deep --json` and pass its output as probe_report",
            ))
        plan_payload, plan_error = self._plan_payload(request)
        checks = self._requested_checks(request, plan_payload)
        if not checks:
            detail = f" (plan draft unreadable: {plan_error})" if plan_error else ""
            findings.append(self.finding(
                "PF-IN-003",
                f"missing input: no requested checks named{detail}, so the plan cannot be vetted",
                {"plan_draft": str(request["plan_draft"]), "plan_error": plan_error},
                "name the checks the plan needs (stage, algorithm, backend, tp, pp, ep, cp) in the plan draft or request `checks`",
            ))
        if probe is not None:
            caps = caps_from_probe(probe)
            for check in checks:
                missing = run_check(caps, check)
                if missing is None:
                    continue  # measured and runnable: only a measured check can PASS
                claimed = plan_claims_available(plan_payload, check)
                message = f"{missing} for check {_fmt_check(check)} — absence of evidence is never PASS"
                if claimed:
                    message += "; the plan draft claims it available/PASS (treat UNMEASURED as PASS)"
                findings.append(self.finding(
                    "PF-IN-002",
                    message,
                    {"check": dict(check), "missing": missing, "plan_claims_pass": claimed},
                    "probe first: run `fskills probe --deep --json`, then re-draft the plan against what it measured",
                ))
        return findings

    @staticmethod
    def _probe_payload(request: dict[str, Any]) -> dict[str, Any]:
        """Load the measured probe JSON (envelope or bare payload); anything else is PF-IN-001 material."""
        path = request.get("probe_report")
        if not path:
            raise ValueError("probe_report is absent — the probe was skipped")
        data = load_json_payload(str(path))
        if "available" not in data:
            raise ValueError(f"{path} is not a measured probe JSON (no 'available' field)")
        return data

    @staticmethod
    def _plan_payload(request: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        try:
            return load_json_payload(str(request["plan_draft"])), None
        except (OSError, ValueError, TypeError) as exc:
            return {}, str(exc)

    @staticmethod
    def _requested_checks(request: Mapping[str, Any], plan_payload_: Mapping[str, Any]) -> list[dict[str, Any]]:
        """request `checks`, else the plan's, else one per plan stage (the planner's training_plan carries
        `stages`, not `checks`); each stripped to the keys `FSCapabilities.check` accepts."""
        raw = request.get("checks") or plan_payload_.get("checks") or [
            {key: stage[key] for key in ("stage", "algorithm") if stage.get(key)}
            for stage in plan_payload_.get("stages") or () if isinstance(stage, Mapping)
        ]
        checks: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            check = {k: v for k, v in entry.items() if k in CHECK_KEYS}
            if isinstance(check.get("stage"), str) and check["stage"]:
                checks.append(check)
        return checks

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        plan_payload, _ = self._plan_payload(request)
        probe = self._probe_payload(request)
        checks = self._requested_checks(request, plan_payload)
        if not checks:
            raise ValueError("no requested checks resolved (check_inputs refuses this case)")
        caps = caps_from_probe(probe)
        rows = [{"check": dict(check), "missing": run_check(caps, check)} for check in checks]

        # A vetted report can PASS only on measured evidence (PF-HO-001 caveats force UNMEASURED).
        caveats: list[str] = [str(row["missing"]) for row in rows if row["missing"]]
        if probe.get("available") is not True:
            caveats.append("the measured probe reports FoundationScale unavailable")
        errors = [str(e) for e in (probe.get("errors") or ())]
        if errors:
            caveats.append("the probe recorded errors: " + "; ".join(errors))
        needs_axes = any(int(check.get(axis, 1)) > 1 for check in checks for axis in PARALLEL_AXES)
        if needs_axes and not probe.get("axes_measured"):
            caveats.append("a needed parallel axis is unmeasured (probe was not --deep)")
        findings: list[Finding] = []
        if caveats:
            findings.append(self.finding(
                "PF-HO-001",
                "the vetted probe report cannot PASS: " + "; ".join(caveats),
                {"plan_draft": str(request["plan_draft"]), "caveats": list(caveats)},
                "probe first: run `fskills probe --deep --json`, then re-run this plan draft",
            ))
        verdict = "PASS" if not caveats else "UNMEASURED"
        payload = {
            "plan_draft": str(request["plan_draft"]),
            "checks": rows,
            "capabilities": dict(probe),
            "verdict": verdict,
        }
        provenance = make_provenance(
            self.name, self.version,
            {"plan_draft": str(request["plan_draft"]), "probe_report": str(request["probe_report"])},
        )
        artifact = Artifact("probe_report", self._artifact_id(request), payload, provenance, 1)
        ref = write_artifact(artifact, request.get("out") or ctx.artifacts_dir)
        status = _worst(Status.PASS, *([Status.UNMEASURED] * len(caveats)))
        next_actions = () if status is Status.PASS else (
            "rerun `fskills probe --deep --json`, then re-run probe-first on the plan draft",)
        return SkillResult(
            status=status, payload=payload, artifacts=(ref,), findings=tuple(findings),
            next_actions=next_actions, provenance=provenance,
        )

    @staticmethod
    def _artifact_id(request: Mapping[str, Any]) -> str:
        raw = str(request.get("artifact_id") or "") or Path(str(request["plan_draft"])).stem or "plan"
        return _UNSAFE_ID.sub("-", raw).strip("._-") or "plan"

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        problems: list[str] = []
        if not result.artifacts:
            problems.append("no probe_report artifact was written")
        for ref in result.artifacts:  # type: ArtifactRef
            try:
                if sha256_file(ref.path) != ref.sha256:
                    problems.append(f"{ref.path} changed after it was written")
            except OSError:
                problems.append(f"{ref.path} is missing on disk")
        return [self.finding("PF-HO-002", problem, {},
                             "re-run probe-first so the vetted probe_report is re-emitted before handoff")
                for problem in problems]

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"probe-first {type(failure).__name__}: {failure}"
                   if isinstance(failure, BaseException)
                   else f"probe-first {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                "`fskills probe --deep --json` was not run before the plan draft",
                "the plan claims a capability FS does not run",
            ),
            checks=(
                "read the probe JSON's `backends`/`executed_axes`/`errors` fields",
                "rerun `fskills probe --deep --json` and diff its output",
            ),
            recovery=(
                "probe first (`fskills probe --deep --json`), then re-run the plan draft",
                "drop the unmeasured capability from the plan draft before trusting it",
            ),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        one = {"stage": "sft", "backend": "fsdp"}
        plan = {"checks": [dict(one)]}
        base = {"plan_draft": "plan.json", "checks": [dict(one)], "out": "artifacts"}
        return {
            "PF-IN-001": {
                "request": dict(base),
                "probe": None,
                "plan": plan,
                "expected": "REFUSED 96 naming PF-IN-001 (the probe was skipped; nothing written)",
            },
            "PF-IN-002": {
                "request": {"plan_draft": "plan.json", "checks": [{"stage": "sft", "backend": "fsdp", "pp": 2}], "out": "artifacts"},
                "probe": probe_payload(train_objectives=("sft",)),
                "plan": {"checks": [{"stage": "sft", "backend": "fsdp", "pp": 2, "status": "PASS"}]},
                "expected": "REFUSED 96 naming PF-IN-002 (an unmeasured axis claimed available/PASS)",
            },
            "PF-IN-003": {
                "request": {"plan_draft": "plan.json", "out": "artifacts"},
                "probe": probe_payload(),
                "plan": {},
                "expected": "REFUSED 96 naming PF-IN-003 (no requested checks named)",
            },
            "PF-HO-001": {
                "request": dict(base),
                "probe": probe_payload(train_objectives=("sft",), errors=("probe: deep axis measurement failed",)),
                "plan": plan,
                "expected": "UNMEASURED 95 with PF-HO-001 (the vetted report cannot PASS)",
            },
            "PF-HO-002": {
                "request": dict(base),
                "probe": probe_payload(train_objectives=("sft",)),
                "plan": plan,
                "tamper": "delete report before handoff",
                "expected": "RED 5 with PF-HO-002 (probe_report missing on disk)",
            },
        }
