"""FsLaunchSkill: the FoundationSkills launch gate (A3.2 user-invoked workflow skill, /fs-launch).

Exit contract: 96 REFUSED writes nothing and names the missing input (FL-IN-001: user_plan_hash is absent or
empty -- the refusal names `fskills hash --spec` for the USER to read the hash; it never echoes or fills it in; FL-IN-002: the
composed confirm_before_launch skill did not PASS (wrong hash, unreadable or non-fs_launch_spec spec);
FL-IN-003: interfaces.fs.launch refused the launch -- ConfirmationRequired or LaunchRefused) and 0 PASS when
the launch went out. FL-HO-001 turns a non-zero LOCAL wait (submit false) into RED 5: the run itself failed.
One handoff rule only, so the RED below is FL-HO-001 and never a composition leak: this is THE ONLY launch
path -- it composes confirm_before_launch, launches only on the exact token the USER typed back, and never
confirms without launching.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from foundationskills.core.orchestrator import ConfirmationRequired, plan_hash
from foundationskills.core.provenance import make_provenance
from foundationskills.core.registry import SkillRegistry
from foundationskills.skills import register_builtin_skills
from foundationskills.core.status import Status
from foundationskills.interfaces.fs.launch import LaunchRefused, launch
from foundationskills.skills.confirm_before_launch.skill import (
    ConfirmBeforeLaunchSkill,
    SpecError,
    load_launch_spec,
)

# -- the launcher indirection ---------------------------------------------------------------------
# run() calls _LAUNCH (never `launch` directly). The template's must_fire_fixtures (fs_status) has NO
# request-level stub mechanism -- every rule there fires from a real input -- so FL-IN-003 and FL-HO-001
# cannot be reached through the request alone. That is why the two fixtures below carry a
# "simulated_launch" fact (as other skills use "tamper"/"fake_plan"): the skill's tests patch the
# module-level `_LAUNCH` to make that fact true before executing the fixture's request. (There is deliberately no
# request["_launcher_stub"]: the template has none.) FL-IN-003 also fires with NO monkeypatch from a
# spec whose `executable` is not True -- launch() refuses such a spec deterministically (LaunchRefused).
_LAUNCH = launch

_BACK = (
    "run `fskills hash --spec <launch_spec>`, read that plan hash to the USER and ask them to type it back as "
    "`user_plan_hash`, then re-run `fskills launch` (this skill never fills it in)"
)


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so the launcher's returned dict is safe to seal in a launch_record."""
    return json.loads(json.dumps(value, default=str, sort_keys=True))


class FsLaunchSkill(BaseSkill):
    name = "fs_launch"
    version = "0.1.0"
    description = (
        "Slash-command launch gate (/fs-launch) for FoundationSkills and THE ONLY launch path: takes an "
        "fs_launch_spec artifact path plus the plan hash the USER typed back (user_plan_hash), names `fskills hash --spec` for the user instead of echoing the "
        "plan hash recomputed from the spec when it is missing and never fills it in, composes "
        "confirm_before_launch (its confirmation_record is the only confirmation accepted) and then launches "
        "through interfaces.fs.launch.launch -- submit true submits to the scheduler, submit false runs it here "
        "and waits -- writing one launch_record {launch_spec, plan_hash, confirmation, submitted, returncode, "
        "result}. Do NOT use for planning, emitting specs, or confirming without launching."
    )
    scope = Scope(  # identical scope to fs_status: it launches any confirmed spec the catalogue emits
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("fs_launch_spec",)
    produces = ("launch_record",)
    invocation = "user"
    invokes = ("confirm_before_launch",)
    requires_fs = True

    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["launch_spec"],
        "properties": {
            "launch_spec": {"type": "string", "minLength": 1},  # path of an fs_launch_spec artifact
            "user_plan_hash": {  # NO minLength: an empty string MUST reach check_inputs (FL-IN-001)
                "type": "string",
                "description": "the plan hash the USER typed back; absent/empty is REFUSED by FL-IN-001",
            },
            "submit": {  # default True: submit to the scheduler; False runs it here and waits (local)
                "type": "boolean",
                "default": True,
            },
            "out": {"type": "string"},  # optional directory for the launch_record
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["launch_spec", "plan_hash", "submitted", "returncode", "result"],
        "additionalProperties": True,
        "properties": {
            "launch_spec": {"type": "string"},
            "plan_hash": {"type": "string"},
            "confirmation": {"type": ["string", "null"]},
            "submitted": {"type": "boolean"},
            "returncode": {"type": ["integer", "null"]},
            "result": {"type": "object"},
        },
    }
    rules = (
        RuleSpec("FL-IN-001", "user_plan_hash absent or empty (the user reads the hash from `fskills hash`; never echoed or filled in)", Severity.BLOCK, "input"),
        RuleSpec("FL-IN-002", "the composed confirm_before_launch skill did not PASS (wrong hash, unreadable or non-fs_launch_spec spec)", Severity.BLOCK, "input"),
        RuleSpec("FL-IN-003", "the launcher refused (interfaces.fs.launch raised ConfirmationRequired or LaunchRefused)", Severity.BLOCK, "input"),
        RuleSpec("FL-HO-001", "a LOCAL launch (submit false) returned a non-zero returncode (the run itself failed)", Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("fskills launch",),
        emits=(),
        apis=("interfaces.fs.launch.launch",),
        notes=(
            "THE ONLY launch path: composes confirm_before_launch, then interfaces.fs.launch.launch "
            "(submit true -> sbatch; submit false -> run here and wait); shows the user `fskills hash --spec` and never echoes or fills in the hash"
        ),
    )

    # -- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _spec(raw: str, ctx: SkillContext) -> Path:
        """Exactly the file confirm_before_launch reads (its resolver): the hash comes from THE file."""
        return ConfirmBeforeLaunchSkill._spec_path(raw, ctx)

    def _load(self, raw: str, ctx: SkillContext) -> tuple[dict[str, Any] | None, SpecError | None]:
        """The fs_launch_spec payload exactly as confirm_before_launch hashes it (CB-IN-003 on failure)."""
        try:
            return load_launch_spec(self._spec(raw, ctx)), None
        except SpecError as exc:
            return None, exc

    @staticmethod
    def _confirm(request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Compose confirm_before_launch (invokes ("confirm_before_launch",)): its confirmation_record is
        the ONLY confirmation this skill accepts, and its artifact ref is sealed in the launch_record."""
        registry = SkillRegistry()
        register_builtin_skills(registry)
        inner_request: dict[str, Any] = {
            "launch_spec": str(request["launch_spec"]),
            "user_plan_hash": str(request.get("user_plan_hash") or ""),
        }
        if request.get("out"):
            inner_request["out"] = str(request["out"])
        return registry.get("confirm_before_launch").execute(inner_request, ctx)

    @staticmethod
    def _evidence_findings(inner: SkillResult) -> list[dict[str, str]]:
        """[{rule_id, severity, message}] of the INNER skill's findings.

        They are placed in evidence and NEVER returned as this skill's findings: their rule ids
        (CB-IN-001/002/003, CB-HO-001) are not declared in self.rules and an undeclared id is RED.
        """
        rows: list[dict[str, str]] = []
        for f in inner.findings or ():
            rows.append({
                "rule_id": str(f.rule_id),
                "severity": str(getattr(f.severity, "name", f.severity)),
                "message": str(f.message),
            })
        return rows

    @staticmethod
    def _confirmation_ref(inner: SkillResult) -> str:
        """The composed confirmation_record's artifact id or path (a ref dict carries both)."""
        refs = inner.artifacts or ()
        ref = refs[0] if refs else None
        if isinstance(ref, dict):
            return str(ref.get("id") or ref.get("path") or json.dumps(ref, sort_keys=True))
        return str(ref or "")

    def _fl_in_001(self, raw: str, expected: str) -> Finding:  # noqa: ARG002 - the hash is deliberately not echoed
        # "Re-show the plan hash" is done by the USER running `fskills hash --spec`, never by echoing it here:
        # confirm_before_launch never echoes the expected hash (CB-IN-001; README.md:45), because a hash an agent can
        # read out of a refusal is a hash an agent can type back for the user, which is the auto-fill this rule forbids.
        return self.finding(
            "FL-IN-001",
            (
                "missing input: user_plan_hash is absent or empty; launching needs the plan hash the USER reads "
                f"from `fskills hash --spec {raw}` and types back. Neither this skill nor the agent fills it in; {_BACK}"
            ),
            {"launch_spec": raw, "show_with": f"fskills hash --spec {raw}", "auto_filled": False},
            f"show the user the spec and `fskills hash --spec {raw}`, then pass the hash THEY type back as "
            f"`user_plan_hash`; {_BACK}",
        )

    def _fl_in_002(self, raw: str, inner: SkillResult) -> Finding:
        return self.finding(
            "FL-IN-002",
            f"the composed confirm_before_launch skill did not PASS: {inner.refusal or 'see its findings'}; {_BACK}",
            {"launch_spec": raw, "confirm_before_launch": self._evidence_findings(inner)},
            "fix what the confirm_before_launch findings in evidence[\"confirm_before_launch\"] name (wrong hash, "
            "unreadable or non-fs_launch_spec spec), then re-run `fskills launch`",
        )

    def _fl_in_003(self, raw: str, exc: BaseException) -> Finding:
        return self.finding(
            "FL-IN-003",
            f"the launcher refused: {exc}",  # the launcher's message, verbatim
            {"launch_spec": raw, "exception_type": type(exc).__name__},
            "fix what the launcher's message names and re-run `fskills launch` (it is THE ONLY launch path)",
        )

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        raw = str(request["launch_spec"])
        user_hash = str(request.get("user_plan_hash") or "")
        if not user_hash.strip():
            payload, _ = self._load(raw, ctx)
            if payload is not None:
                # FL-IN-001: the refusal names `fskills hash --spec` for the USER; the hash is never echoed.
                findings.append(self._fl_in_001(raw, plan_hash(payload)))
                return findings
            # "If the spec cannot be read, FL-IN-002 applies instead": there is no hash to show, so the
            # composed refusal (CB-IN-003) is what travels in the evidence — FL-IN-001 does not fire.
        inner = self._confirm(request, ctx)
        if inner.status is not Status.PASS:
            # The composed findings are evidence only; returning them as findings would be CORE-UNDECLARED-RULE.
            findings.append(self._fl_in_002(raw, inner))
        return findings

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        raw = str(request["launch_spec"])
        user_hash = str(request.get("user_plan_hash") or "")
        # (1) recompute the plan hash exactly as confirm_before_launch computes it, then confirm by
        #     composition (it is re-run here to seal its artifact ref into the launch_record).
        try:
            payload = load_launch_spec(self._spec(raw, ctx))
        except SpecError as exc:
            inner = self._confirm(request, ctx)
            return self._refused(f"FL-IN-002: {exc}", [self._fl_in_002(raw, inner)])
        if not user_hash.strip():  # unreachable through execute(): FL-IN-001 blocks first (BLOCK input)
            return self._refused("FL-IN-001", [self._fl_in_001(raw, plan_hash(payload))])
        inner = self._confirm(request, ctx)
        if inner.status is not Status.PASS:
            return self._refused(
                f"FL-IN-002: {inner.refusal or 'see findings'}", [self._fl_in_002(raw, inner)]
            )
        expected = plan_hash(payload)  # == what the USER typed back (confirm_before_launch proved it)
        # (2) THE launch: the user-typed hash and the composed confirmation both gate it.
        submitted = bool(request.get("submit", True))
        try:
            launched = _LAUNCH(payload, confirm=user_hash, submit=submitted)
        except (ConfirmationRequired, LaunchRefused) as exc:
            return self._refused(f"FL-IN-003: {exc}", [self._fl_in_003(raw, exc)])
        # (3) one launch_record artifact, written like fs_status writes its status_explanation.
        record = {
            "launch_spec": raw,
            "plan_hash": expected,
            "confirmation": self._confirmation_ref(inner),
            "submitted": submitted,
            "returncode": launched.get("returncode"),
            "result": _jsonable(launched),
        }
        provenance = make_provenance(
            self.name, self.version,
            {
                "launch_spec": raw,
                "plan_hash": expected,
                "confirmation": record["confirmation"],
                "submitted": "true" if submitted else "false",
                "returncode": str(record["returncode"]),
            },
        )
        artifact = Artifact("launch_record", "launch", record, provenance, 1)
        ref = write_artifact(artifact, request.get("out") or ctx.artifacts_dir)
        return SkillResult(
            status=Status.PASS, payload=record, artifacts=(ref,), findings=(),
            next_actions=(), provenance=provenance,
        )

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """FL-HO-001: a LOCAL launch (submit false) whose returncode is non-zero means the run itself
        failed — the launch_record is already written, but the verdict must not read PASS."""
        record = result.payload
        returncode = record.get("returncode")
        if record.get("submitted") is False and returncode is not None and int(returncode) != 0:
            detail = record.get("result") or {}
            return [self.finding(
                "FL-HO-001",
                (
                    f"the run itself failed: the local launch (submit False) returned {returncode} "
                    f"(fs_verdict {detail.get('fs_verdict')}, log {detail.get('log')}); the artifacts of this run "
                    "must not be used"
                ),
                {
                    "launch_spec": str(record.get("launch_spec") or ""),
                    "returncode": int(returncode),
                    "fs_verdict": detail.get("fs_verdict"),
                    "log": detail.get("log"),
                },
                "fix what the launch_record's result.tail / result.fs_verdict name and re-run `fskills launch`",
            )]
        return []

    # -- diagnosis ------------------------------------------------------------------------------
    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"fs_launch {type(failure).__name__}: {failure}"
                   if isinstance(failure, BaseException)
                   else f"fs_launch {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                "user_plan_hash is absent or empty (the USER reads the hash from `fskills hash --spec` and types it back; never echoed or filled in)",
                "the composed confirm_before_launch did not PASS (wrong hash, unreadable or non-fs_launch_spec spec)",
                "interfaces.fs.launch refused the launch (no exact token, no positive executability evidence, dry-run failed)",
                "a local submit:false run returned a non-zero returncode (FL-HO-001: the run itself failed)",
            ),
            checks=(
                "read the FL-IN-001/FL-IN-002/FL-IN-003 finding message (FL-IN-001 names `fskills hash --spec` for the user; FL-IN-002 embeds the confirm_before_launch findings in evidence)",
                "run `fskills hash --spec <launch_spec>` and compare with what the USER typed back",
                "read the launch_record's result.tail / result.fs_verdict / result.log",
            ),
            recovery=(
                "ask the USER to type the plan hash back as `user_plan_hash` (never auto-fill it)",
                "fix what the confirm_before_launch findings in the evidence name, then re-run fs_launch",
                "fix what the launcher's message names, then re-run fs_launch — it is THE ONLY launch path",
            ),
        )

    # -- must-fire fixtures ---------------------------------------------------------------------
    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        # FL-IN-003 and FL-HO-001 cannot be reached without a launcher run, and the template fixture
        # shape carries no request-level stub. "simulated_launch" states the fact the tests make true by patching
        # on the module-level `_LAUNCH` before executing that fixture's request. (FL-IN-003 also fires
        # with NO monkeypatch from a spec whose `executable` is not True: launch() refuses it with
        # LaunchRefused deterministically.)
        return {
            "FL-IN-001": {
                "request": {"launch_spec": "launch/fs_launch_spec.q27.json"},  # no hash at all
                "expected": "REFUSED 96 naming FL-IN-001 and the `fskills hash --spec` command for the user, the hash itself never echoed "
                            "(the user must type it back; it is never filled in); nothing launched and no launch_record written",
            },
            "FL-IN-002": {
                "request": {"launch_spec": "/nonexistent/spec.json", "user_plan_hash": "deadbeef"},
                "expected": "REFUSED 96 naming FL-IN-002 with confirm_before_launch's findings as "
                            "{rule_id, severity, message} rows in evidence[\"confirm_before_launch\"] (never as findings)",
            },
            "FL-IN-003": {
                "request": {"launch_spec": "launch/fs_launch_spec.q27.json", "user_plan_hash": "3b543c4f2ad0e91f", "submit": False},
                "hash_check": "typed hash matches spec",  # the user typed 3b543c4f2ad0e91f back; it equals `fskills hash --spec`
                "simulated_launch": "the launcher raises LaunchRefused (e.g. the spec is not executable)",
                "expected": "REFUSED 96 naming FL-IN-003 with the launcher's message verbatim; no launch_record written",
            },
            "FL-HO-001": {
                "request": {"launch_spec": "launch/fs_launch_spec.q27.json", "user_plan_hash": "3b543c4f2ad0e91f", "submit": False},
                "hash_check": "typed hash matches spec",  # the user typed 3b543c4f2ad0e91f back; it equals `fskills hash --spec`
                "simulated_launch": "the local launch ran and returned returncode 5",
                "expected": "RED 5 naming FL-HO-001 (the run itself failed: the local launch returned 5); the launch_record is written but must not be used",
            },
        }
