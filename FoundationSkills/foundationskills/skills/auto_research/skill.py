"""AutoResearchSkill: validate, authorise, record, propose and decide a research campaign (M0, CPU-only).

NeMo-RL ``nemo-rl-auto-research`` loop shape: baseline -> propose -> run -> record -> decide -> close.
M0 launches nothing: ``launch`` validates a launch spec and authorises it with a derived token in the
append-only hash-chained ledger, ``record`` stores immutable trial results, ``propose`` ranks catalog
ideas, and ``close`` decides statistically against a calibrated noise floor and writes the
``auto_research_report`` artifact. Submitting jobs is M1.

Note on rule phases: handoff rules AR-HO-* are emitted inside ``run()`` (not in ``check_handoff``)
because ``BaseSkill.execute`` only calls check_handoff for PASS results, while a RED/UNMEASURED close
must still carry its findings. ``check_handoff`` therefore re-verifies nothing on PASS and returns no
findings (same pattern as the data_engine skill).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

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
from foundationskills.skills.auto_research.accept import decide, noise_floor
from foundationskills.skills.auto_research.campaign import campaign_hash, check_launch, check_spec, launch_token
from foundationskills.skills.auto_research.ledger import Ledger, canonical, ledger_files, sha256_hex
from foundationskills.skills.auto_research.propose import propose

_ACTIONS = ("check", "launch", "record", "propose", "close")
_RESULT_ROLES = ("baseline", "candidate", "confirm")
_RESULT_FIELDS = ("trial", "role", "seed", "status", "limited", "steps", "eval_policy_fingerprint", "metrics")
_RANK = {"accepted_gain": 4, "accepted_flat": 3, "no_gain": 2, "rejected_regress": 1, "unmeasured": 0}

FINGERPRINT = "sha256:" + "a1" * 32
BASE_FINGERPRINT = "sha256:" + "b2" * 32

_RECOVERY = {
    "AR-IN-001": "set objective.metric to one of eval_policy.metrics",
    "AR-IN-002": "set base.model and base.fingerprint = 'sha256:<64 hex>' of the sealed base model",
    "AR-IN-003": "set budget.gpu_hours_total > 0 and budget.max_runs > 0",
    "AR-IN-004": "set eval_policy.fingerprint = 'sha256:<64 hex>' of the frozen eval policy",
    "AR-IN-005": "pick axes from campaign.AXIS_PATHS with a matching type/range (UNSUPPORTED_AXES refuse)",
    "AR-IN-006": "record {trial, role, seed, status, limited, steps, eval_policy_fingerprint, metrics}; no crash metrics",
    "AR-AP-001": "pass campaign_confirm = the human-approved campaign_hash(spec) and a named approver",
    "AR-LN-001": "keep delta/shape/partition inside spec (base and model are locked campaign fields)",
    "AR-LN-002": "stay inside max_runs/gpu hours and keep the confirm reserve for confirm runs",
    "AR-LN-004": "only run time '10-00:00:00'; never pkill -u / scancel / killall or name a quarantined node",
    "AR-LN-005": "lower gpu_hours_est (or raise per_run_timeout_h within the 240h cluster limit)",
    "AR-LG-001": "record a new trial/seed pair; recorded results are immutable",
}


class AutoResearchSkill(BaseSkill):
    """Run a bounded experiment campaign: authorise, record, propose, decide, close (M0: no launches)."""

    name = "auto_research"
    version = "0.1.0"
    description = (
        "Drive a FoundationSkills research campaign: validate the campaign spec, open one human approval "
        "envelope (campaign_confirm hash), authorise launches with derived tokens, record immutable trial "
        "results, propose catalog ideas and close with statistical acceptance against a calibrated noise floor."
    )
    scope = Scope(
        model_types=("llm", "vlm"),
        families="any",
        stages=("sft", "preference", "rl"),
        algorithms=("dr_grpo", "gspo", "dapo"),
        methods=("full", "lora"),
        hardware=(),
    )
    consumes = ()
    produces = ("auto_research_report",)
    input_schema = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS)},
            "campaign_spec": {"type": "object"},
            "campaign_confirm": {"type": "string"},
            "approver": {"type": "string"},
            "ledger_dir": {"type": "string"},
            "launch_spec": {"type": "object"},
            "result": {"type": "object"},
            "current": {"type": "object"},
            "symptoms": {"type": "array"},
            "stop_reason": {"type": "string"},
        },
    }
    output_schema = {"type": "object", "additionalProperties": True}
    rules = (
        RuleSpec("AR-IN-001", "objective.metric missing or not in eval_policy.metrics",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-IN-002", "base.model or base.fingerprint missing or not a sha256 reference",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-IN-003", "budget.gpu_hours_total or budget.max_runs absent or <= 0",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-IN-004", "eval_policy.fingerprint missing or malformed",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-IN-005", "axis key outside AXIS_PATHS (or unsupported) or range/values invalid for its kind",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-IN-006", "result payload malformed or a crash result carries metric values",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-AP-001",
                 "campaign_confirm missing or not the hash of the spec (or ledger approved a different hash)",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LN-001", "delta outside spec.axes, shape/partition mismatch or a locked base/model override",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LN-002", "launch would exceed the run or gpu-hour budget (including the confirm reserve)",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LN-004",
                 "launch carries a forbidden command (pkill -u / scancel / killall) or names a quarantined node",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LN-005", "gpu_hours_est implies a wall time above per_run_timeout_h",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LG-001", "a recorded trial result would be overwritten (results are immutable; record a new trial)",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-HO-001", "campaign closed with no accepted gain", Severity.BLOCK, "handoff"),
        RuleSpec("AR-HO-002", "best candidate breaches a guardrail band", Severity.BLOCK, "handoff"),
        RuleSpec("AR-HO-003", "ledger chain verification failed", Severity.BLOCK, "handoff"),
        RuleSpec("AR-HO-004", "load-bearing evidence missing (uncalibrated noise floor, screening-only seeds, limited or "
                              "crashed runs) - status UNMEASURED", Severity.WARN, "handoff"),
        RuleSpec("AR-HO-005", "accepted a flat-but-simpler change", Severity.INFO, "handoff"),
    )
    fs_interface = FSInterface(
        entries=(),
        emits=("auto_research_report",),
        apis=("campaign_hash", "launch_token", "check_spec", "check_launch", "noise_floor", "decide", "propose", "Ledger"),
        notes=(
            "M0 validates/authorises/records/decides over the hash-chained ledger and writes "
            "auto_research_report; job submission to training.emit (M1) is deliberately not wired yet."
        ),
    )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _ledger(request: dict[str, Any], ctx: SkillContext) -> Ledger:
        return Ledger(Path(str(request.get("ledger_dir") or (ctx.workdir / "ledger"))))

    @staticmethod
    def _payload(ledger: Ledger, entry: dict[str, Any]) -> dict[str, Any]:
        try:
            return dict(ledger.payload(entry))
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def _approvals(self, ledger: Ledger, campaign: str) -> list[dict[str, Any]]:
        return [
            self._payload(ledger, e)
            for e in ledger.entries()
            if e.get("op") == "campaign_approved" and e.get("campaign") == campaign
        ]

    # ---- input checks ------------------------------------------------------

    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        action = str(request.get("action") or "check")
        spec = dict(request.get("campaign_spec") or {})
        for rule_id, message in check_spec(spec):
            findings.append(self.finding(rule_id, message, {"spec": spec.get("id")}, _RECOVERY[rule_id]))
        ledger = self._ledger(request, ctx)
        if action != "check":
            findings.extend(self._check_approval(request, spec, ledger))
        if action == "launch":
            findings.extend(self._check_launch_request(request, spec, ledger))
        if action == "record":
            findings.extend(self._check_record_request(request, spec, ledger))
        return findings

    def _check_approval(self, request: dict[str, Any], spec: dict[str, Any], ledger: Ledger) -> list[Finding]:
        findings: list[Finding] = []
        confirm = str(request.get("campaign_confirm") or "")
        expected = campaign_hash(spec)
        approved = [str(p.get("spec_hash") or "") for p in self._approvals(ledger, _campaign(spec))]
        if not confirm or confirm != expected:
            findings.append(
                self.finding(
                    "AR-AP-001",
                    f"campaign_confirm {confirm!r} is not the campaign hash {expected}",
                    {"expected": expected},
                    _RECOVERY["AR-AP-001"],
                )
            )
        elif any(h != confirm for h in approved):
            findings.append(
                self.finding(
                    "AR-AP-001",
                    f"the ledger approved spec_hash {approved}, not {confirm}",
                    {"approved": approved},
                    "open a new campaign: the approval envelope bounds exactly one spec hash",
                )
            )
        if not approved and not str(request.get("approver") or "").strip():
            findings.append(
                self.finding(
                    "AR-AP-001",
                    "approver is required to open the approval envelope on the first action",
                    {},
                    "pass approver: the human who approved campaign_confirm",
                )
            )
        return findings

    def _check_launch_request(
        self, request: dict[str, Any], spec: dict[str, Any], ledger: Ledger
    ) -> list[Finding]:
        launch_spec = dict(request.get("launch_spec") or {})
        campaign = _campaign(spec)  # only OUR campaign's authorisations consume the budget
        launches = [
            self._payload(ledger, e)
            for e in ledger.entries()
            if e.get("op") == "launch_authorised" and e.get("campaign") == campaign
        ]
        return [
            self.finding(rule_id, message, {"trial": launch_spec.get("trial")}, _RECOVERY[rule_id])
            for rule_id, message in check_launch(spec, launch_spec, launches)
        ]

    def _check_record_request(self, request: dict[str, Any], spec: dict[str, Any], ledger: Ledger) -> list[Finding]:
        findings: list[Finding] = []
        result = dict(request.get("result") or {})
        for message in result_problems(result):
            findings.append(self.finding("AR-IN-006", message, {"result": result}, _RECOVERY["AR-IN-006"]))
        prior = _safe_list(lambda: ledger.results(_campaign(spec)))
        key = (result.get("trial"), result.get("seed"))
        recorded = {(row.get("trial"), row.get("seed")) for row in prior}
        if key in recorded:
            findings.append(
                self.finding(
                    "AR-LG-001",
                    f"trial result {key} is already recorded; results are immutable",
                    {"recorded": sorted(str(k) for k in recorded)},
                    _RECOVERY["AR-LG-001"],
                )
            )
        return findings

    # ---- run ---------------------------------------------------------------

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        action = str(request.get("action") or "check")
        spec = dict(request.get("campaign_spec") or {})
        spec_hash = campaign_hash(spec)
        campaign = _campaign(spec)
        if action == "check":
            return SkillResult(
                Status.PASS,
                {
                    "spec_hash": spec_hash,
                    "axes": [dict(a) for a in (spec.get("axes") or [])],
                    "budget": dict(spec.get("budget") or {}),
                },
            )
        ledger = self._ledger(request, ctx)
        if not self._approvals(ledger, campaign):  # the first action opens the approval envelope
            ledger.append(
                "campaign_approved", campaign, "-",
                {"spec_hash": spec_hash, "approver": str(request.get("approver") or "")},
            )
        if action == "launch":
            return self._launch(request, ledger, spec, campaign, spec_hash)
        if action == "record":
            return self._record(request, ledger, campaign)
        if action == "propose":
            cards = propose(
                spec, _safe_list(lambda: ledger.results(campaign)),
                dict(request.get("current") or {}), list(request.get("symptoms") or []),
            )
            return SkillResult(Status.PASS, {"cards": cards})
        return self._close(request, ctx, ledger, spec, campaign, spec_hash)

    def _launch(
        self, request: dict[str, Any], ledger: Ledger, spec: dict[str, Any], campaign: str, spec_hash: str
    ) -> SkillResult:
        launch_spec = dict(request.get("launch_spec") or {})
        confirm = str(request.get("campaign_confirm") or spec_hash)
        token = launch_token(confirm, launch_spec)
        ledger.append(
            "launch_authorised", campaign, str(launch_spec.get("trial") or "-"),
            {
                "launch_spec": launch_spec,
                "spec_hash": sha256_hex(canonical(launch_spec)),
                "launch_token": token,
                "gpu_hours_est": float(launch_spec.get("gpu_hours_est") or 0.0),
            },
        )
        launches = _safe_list(lambda: ledger.launches(campaign))
        used = sum(float(p.get("gpu_hours_est") or 0.0) for p in launches)
        total = float(dict(spec.get("budget") or {}).get("gpu_hours_total") or 0.0)
        return SkillResult(
            Status.PASS,
            {"launch_token": token, "budget_left": max(0.0, total - used), "trial": str(launch_spec.get("trial") or "-")},
        )

    def _record(self, request: dict[str, Any], ledger: Ledger, campaign: str) -> SkillResult:
        result = dict(request.get("result") or {})
        ledger.append("trial_result", campaign, str(result.get("trial") or "-"), result)
        return SkillResult(
            Status.PASS,
            {"recorded": {"trial": result.get("trial"), "seed": result.get("seed")}, "ledger": ledger.head()},
        )

    def _close(
        self, request: dict[str, Any], ctx: SkillContext, ledger: Ledger, spec: dict[str, Any],
        campaign: str, spec_hash: str,
    ) -> SkillResult:
        stop_reason = str(request.get("stop_reason") or "")
        problems = ledger.verify()
        results = _safe_list(lambda: ledger.results(campaign))
        launches = _safe_list(lambda: ledger.launches(campaign))
        metric = str(dict(spec.get("objective") or {}).get("metric") or "")
        baseline = [row for row in results if row.get("role") == "baseline"]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in results:
            if row.get("role") != "baseline":
                grouped.setdefault(str(row.get("trial") or "?"), []).append(row)
        decisions = {trial: decide(spec, baseline, rows, metric) for trial, rows in sorted(grouped.items())}
        floor = noise_floor(baseline, metric, int(dict(spec.get("seeds") or {}).get("baseline_repeats", 3)))
        gains = [d for d in decisions.values() if d["verdict"] == "accepted_gain"]
        flats = [d for d in decisions.values() if d["verdict"] == "accepted_flat"]
        regressions = [d for d in decisions.values() if d["verdict"] == "rejected_regress"]
        top = _top_candidate(decisions)
        best_trial, best = top if top else (None, None)

        used_hours = sum(float(p.get("gpu_hours_est") or 0.0) for p in launches)
        budgets = {"planned": dict(spec.get("budget") or {}), "used_gpu_hours": used_hours, "runs": len(launches)}
        verdicts = {trial: d["verdict"] for trial, d in decisions.items()}
        findings: list[Finding] = []
        if problems:
            outcome, status = "unmeasured", Status.RED
            findings.append(
                self.finding(
                    "AR-HO-003", f"ledger chain verification failed: {'; '.join(problems)}",
                    {"problems": problems}, "restore an intact ledger; entries and objects are immutable",
                )
            )
        elif (not gains and any(v == "unmeasured" for v in verdicts.values())) or floor is None:
            outcome, status = "unmeasured", Status.UNMEASURED
            findings.append(
                self.finding(
                    "AR-HO-004",
                    f"load-bearing evidence missing (noise floor {floor!r}, "
                    f"{sum(1 for v in verdicts.values() if v == 'unmeasured')} unmeasured candidate(s))",
                    {"noise_floor": floor, "verdicts": verdicts},
                    "collect calibrated baseline repeats or un-limit the screening runs before adopting anything",
                )
            )
        elif best is not None and any(str(r).startswith("guardrail:") for r in best["reasons"]):
            outcome, status = "regressed", Status.RED
            breaches = [r for r in best["reasons"] if str(r).startswith("guardrail:")]
            findings.append(
                self.finding(
                    "AR-HO-002", f"best candidate {best_trial} breaches a guardrail band: {', '.join(breaches)}",
                    {"trial": best_trial, "breaches": breaches},
                    "keep the baseline, or relax the guardrail explicitly in a new campaign spec",
                )
            )
        elif not gains and not flats:
            outcome = "regressed" if regressions else "no_gain"
            status = Status.RED
            findings.append(
                self.finding(
                    "AR-HO-001", f"campaign closed with no accepted gain (stop reason: {stop_reason or 'n/a'})",
                    {"verdicts": verdicts}, "propose new ideas and record more trials, or accept the baseline",
                )
            )
        elif flats and not gains:
            outcome, status = "flat_simplified", Status.PASS
            findings.append(
                self.finding(
                    "AR-HO-005", f"accepted a flat-but-simpler change: {best_trial}", {"verdicts": verdicts}, ""
                )
            )
        else:
            outcome, status = "improved", Status.PASS

        if outcome == "improved":
            recommendation = f"adopt {best_trial}: mean_delta {best['mean_delta']} > tau {best['tau']}"
        elif outcome == "flat_simplified":
            recommendation = f"adopt {best_trial} for simplicity: |mean_delta| {best['mean_delta']} <= tau {best['tau']}"
        elif outcome == "regressed":
            recommendation = f"keep the baseline; {best_trial} regressed or breached a guardrail"
        elif outcome == "unmeasured":
            recommendation = "collect more evidence: the noise floor or the paired repeats are missing"
        else:
            recommendation = "keep the baseline; no candidate beat tau"

        preseal = ledger.head()
        ledger.append(  # seal the ledger first: the report must describe the sealed chain
            "campaign_closed", campaign, "-",
            {"stop_reason": stop_reason, "outcome": outcome, "head_hash": preseal["head_hash"]},
        )
        head = ledger.head()
        report = {
            "campaign": {"id": campaign, "spec_hash": spec_hash, "approver": str(request.get("approver") or "")},
            "outcome": outcome,
            "recommendation": recommendation,
            "decisions": decisions,
            "budgets": budgets,
            "ledger": {"count": head["count"], "head_hash": head["head_hash"], "verified": not problems},
            "tsv": _safe_text(lambda: ledger.tsv_view(campaign)),
        }
        tags = {"campaign": campaign, "spec_hash": spec_hash}
        stamp = make_provenance(self.name, self.version, tags)
        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact = Artifact("auto_research_report", campaign, report, stamp)
        artifacts = (write_artifact(artifact, ctx.artifacts_dir),)
        payload = {
            "spec_hash": spec_hash,
            "campaign": report["campaign"],
            "outcome": outcome,
            "recommendation": recommendation,
            "best": best_trial,
            "decisions": decisions,
            "budgets": budgets,
            "ledger": report["ledger"],
            "tsv": report["tsv"],
        }
        return SkillResult(status, payload, artifacts, tuple(findings), provenance=stamp)

    # ---- handoff / diagnosis / fixtures ------------------------------------

    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """No-op: AR-HO-* findings are raised in run() (see the module docstring)."""
        return []

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        if isinstance(failure, SkillResult):
            ids = {f.rule_id for f in failure.findings}
            if "AR-HO-003" in ids:
                return Diagnosis(
                    "ledger chain verification failed",
                    ("an object or chain line was modified after append", "a copy step truncated the chain"),
                    ("read the verify() problems named by seq in AR-HO-003",),
                    ("restore chain.jsonl and objects/ from an intact copy", "re-run on a fresh ledger"),
                )
            return Diagnosis(
                failure.refusal or "campaign closed RED/UNMEASURED",
                ("invalid campaign spec or launch gate", "unmeasured acceptance evidence"),
                ("read the findings and the auto_research_report artifact decisions",),
                ("fix the refused field and re-run the action", "record calibrated baseline repeats"),
            )
        text = f"{type(failure).__name__}: {failure}"
        if isinstance(failure, FileNotFoundError):
            return Diagnosis(
                "ledger object missing",
                ("objects/ was not copied with chain.jsonl",),
                ("check that the payload hashes named in chain.jsonl exist under objects/",),
                ("restore the missing objects", "start a fresh ledger directory"),
            )
        if isinstance(failure, (ValueError, TypeError)) and ("NaN" in text or "Out of range" in text):
            return Diagnosis(
                "non-canonical payload",
                ("a payload carried NaN/inf floats which canonical() refuses",),
                ("inspect the failing request 'result'/'launch_spec' floats",),
                ("record finite numbers only",),
            )
        if isinstance(failure, KeyError):
            return Diagnosis(
                text,
                ("a request field or ledger entry key is missing",),
                ("check campaign_spec/result payloads against the SKILL.md request shapes",),
                ("supply the missing field and re-run",),
            )
        return Diagnosis(
            text,
            ("unexpected campaign failure",),
            ("inspect the ledger and the request payload",),
            ("fix per the exception message and re-run the action",),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        """One negative fixture per rule (pure data, no request-side test knobs).

        String values in the request are materialised by the test: ``{tmp}`` becomes the test
        tmp_path (optional ``files`` are written there first, each key a relative path) and
        ``{confirm}`` becomes ``campaign_hash(campaign_spec)``. Fixtures that need prior ledger
        evidence ship the exact archive bytes under ``files`` (built by the pure ``ledger_files``):
        the approval envelope event first, then one event per pre-recorded trial result.
        """
        fix = "arbiter"
        guard = _spec(confirm={"k": 2.0, "noise_floor_rel": 0.005,
                               "guardrails": ["throughput"], "guard_abs_epsilon": 0.01})
        flat_rows = [*_BASELINE_ROWS,
                     _result("t1", "candidate", 101, _val(0.5)),
                     _result("t1", "candidate", 102, _val(0.501)),
                     _result("t1", "candidate", 103, _val(0.499))]
        simple_rows = [*_BASELINE_ROWS,
                       _result("t1", "candidate", 101, _val(0.501), simpler=True),
                       _result("t1", "candidate", 102, _val(0.5), simpler=True),
                       _result("t1", "candidate", 103, _val(0.502), simpler=True)]
        screening_rows = [_result("baseline", "baseline", 101, _val(0.5)),
                          _result("t1", "candidate", 101, _val(0.6), limited=True),
                          _result("t1", "candidate", 102, _val(0.6), limited=True),
                          _result("t1", "candidate", 103, _val(0.6), limited=True)]
        guard_rows = [
            _result("baseline", "baseline", 101, _val_throughput(0.5, 1000.0)),
            _result("baseline", "baseline", 102, _val_throughput(0.502, 1000.0)),
            _result("baseline", "baseline", 103, _val_throughput(0.501, 1000.0)),
            _result("t1", "candidate", 101, _val_throughput(0.6, 900.0)),
            _result("t1", "candidate", 102, _val_throughput(0.602, 899.0)),
            _result("t1", "candidate", 103, _val_throughput(0.601, 901.0)),
        ]
        return {
            "AR-IN-001": {"request": {"action": "check", "campaign_spec": _spec(objective={
                "metric": "nll", "direction": "min", "benchmarks": []})}},
            "AR-IN-002": {"request": {"action": "check", "campaign_spec": _spec(base={"model": "", "fingerprint": "deadbeef"})}},
            "AR-IN-003": {"request": {"action": "check", "campaign_spec": _spec(budget={
                "gpu_hours_total": 0, "max_runs": 0, "per_run_timeout_h": 8.0, "reserve_frac": 0.3})}},
            "AR-IN-004": {"request": {"action": "check", "campaign_spec": _spec(eval_policy={
                "fingerprint": "md5:1234", "metrics": ["val_accuracy", "throughput"]})}},
            "AR-IN-005": {"request": {"action": "check", "campaign_spec": _spec(axes=[
                {"key": "parallel.tp", "type": "int", "min": 1, "max": 8},
                {"key": "optim.spurious", "type": "float", "min": 0.0, "max": 1.0}])}},
            "AR-IN-006": {
                "request": {
                    "action": "record", "campaign_spec": _spec(), "campaign_confirm": "{confirm}", "approver": fix,
                    "ledger_dir": "{tmp}/ledger",
                    "result": {"trial": "t1", "role": "candidate", "seed": 101, "status": "crash", "limited": False,
                               "steps": 0, "eval_policy_fingerprint": FINGERPRINT,
                               "metrics": {"val_accuracy": {"value": 0.1, "se": 0.0}}},
                }
            },
            "AR-AP-001": {
                "request": {"action": "propose", "campaign_spec": _spec(), "approver": fix,
                            "ledger_dir": "{tmp}/ledger", "current": {"optim.lr": 1e-3}, "symptoms": []}
            },
            "AR-LN-001": {
                "request": {"action": "launch", "campaign_spec": _spec(), "campaign_confirm": "{confirm}",
                            "approver": fix, "ledger_dir": "{tmp}/ledger",
                            "launch_spec": _launch(delta={"optim.momentum": 0.9})}
            },
            "AR-LN-002": {
                "request": {
                    "action": "launch", "campaign_confirm": "{confirm}", "approver": fix,
                    "ledger_dir": "{tmp}/ledger",
                    "campaign_spec": _spec(budget={
                        "gpu_hours_total": 2.0, "max_runs": 6, "per_run_timeout_h": 8.0, "reserve_frac": 0.3}),
                    "launch_spec": _launch(gpu_hours_est=8.0),
                }
            },
            "AR-LN-004": {
                "request": {"action": "launch", "campaign_spec": _spec(), "campaign_confirm": "{confirm}",
                            "approver": fix, "ledger_dir": "{tmp}/ledger",
                            "launch_spec": _launch(commands=["scancel 4242"])}
            },
            "AR-LN-005": {
                "request": {
                    "action": "launch", "campaign_confirm": "{confirm}", "approver": fix,
                    "ledger_dir": "{tmp}/ledger",
                    "campaign_spec": _spec(budget={
                        "gpu_hours_total": 400.0, "max_runs": 6, "per_run_timeout_h": 8.0, "reserve_frac": 0.3}),
                    "launch_spec": _launch(gpu_hours_est=200.0),
                }
            },
            "AR-LG-001": {
                **_stage(_spec(), [_result("t1", "candidate", 101, _val(0.5))]),
                "request": {
                    "action": "record", "campaign_spec": _spec(), "campaign_confirm": "{confirm}", "approver": fix,
                    "ledger_dir": "{tmp}/ledger",
                    "result": _result("t1", "candidate", 101, _val(0.51)),
                },
            },
            "AR-HO-001": {
                **_stage(_spec(), flat_rows),
                "request": {
                    "action": "close", "campaign_spec": _spec(), "campaign_confirm": "{confirm}", "approver": fix,
                    "ledger_dir": "{tmp}/ledger", "stop_reason": "budget exhausted",
                },
            },
            "AR-HO-002": {
                **_stage(guard, guard_rows),
                "request": {
                    "action": "close", "campaign_spec": guard, "campaign_confirm": "{confirm}", "approver": fix,
                    "ledger_dir": "{tmp}/ledger", "stop_reason": "budget exhausted",
                },
            },
            "AR-HO-003": {
                **_stage(_spec(), flat_rows, tamper=True),
                "request": {
                    "action": "close", "campaign_spec": _spec(), "campaign_confirm": "{confirm}", "approver": fix,
                    "ledger_dir": "{tmp}/ledger", "stop_reason": "budget exhausted",
                },
            },
            "AR-HO-004": {
                **_stage(_spec(), screening_rows),
                "request": {
                    "action": "close", "campaign_spec": _spec(), "campaign_confirm": "{confirm}", "approver": fix,
                    "ledger_dir": "{tmp}/ledger", "stop_reason": "screening only",
                },
            },
            "AR-HO-005": {
                **_stage(_spec(), simple_rows),
                "request": {
                    "action": "close", "campaign_spec": _spec(), "campaign_confirm": "{confirm}", "approver": fix,
                    "ledger_dir": "{tmp}/ledger", "stop_reason": "budget exhausted",
                },
            },
        }


# ---- fixture builders (pure data) -----------------------------------------

def _spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": "ar-fixture",
        "objective": {"metric": "val_accuracy", "direction": "max", "benchmarks": ["gsm8k"]},
        "eval_policy": {"fingerprint": FINGERPRINT, "metrics": ["val_accuracy", "throughput"]},
        "base": {"model": "gemma4-4b", "fingerprint": BASE_FINGERPRINT},
        "confirm": {"k": 2.0, "noise_floor_rel": 0.005, "guardrails": [], "guard_abs_epsilon": 0.01},
        "seeds": {"baseline_repeats": 3, "confirm_repeats": 3, "seed_list": [101, 102, 103]},
        "budget": {"gpu_hours_total": 24.0, "max_runs": 6, "per_run_timeout_h": 8.0, "reserve_frac": 0.3},
        "stopping": {"no_gain_streak": 3, "max_crash_streak": 2},
        "axes": [
            {"key": "optim.lr", "type": "log_float", "min": 1e-6, "max": 1e-3},
            {"key": "train.method", "type": "categorical", "values": ["full", "lora"]},
            {"key": "lora.rank", "type": "int", "min": 4, "max": 64},
        ],
        "cluster": {"partition": "rally", "max_nodes": 2, "gpus_per_node": 8, "time": "10-00:00:00",
                    "exclude": ["r01dgx02"]},
    }
    spec.update(overrides)
    return spec


def _launch(**overrides: Any) -> dict[str, Any]:
    launch: dict[str, Any] = {
        "trial": "t1", "role": "candidate", "seed": 101, "delta": {"optim.lr": 5e-4},
        "nodes": 1, "gpus_per_node": 8, "partition": "rally", "time": "10-00:00:00",
        "gpu_hours_est": 4.0, "commands": ["uv run train.py --config campaign.yaml"],
    }
    launch.update(overrides)
    return launch


def _val(value: float) -> dict[str, Any]:
    return {"val_accuracy": {"value": value, "se": 0.001}}


def _val_throughput(value: float, throughput: float) -> dict[str, Any]:
    return {"val_accuracy": {"value": value, "se": 0.001}, "throughput": {"value": throughput, "se": 1.0}}


def _result(trial: str, role: str, seed: int, metrics: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial": trial, "role": role, "seed": seed, "status": "ok", "limited": False,
        "steps": 200, "eval_policy_fingerprint": FINGERPRINT, "metrics": dict(metrics),
    }
    row.update(overrides)
    return row


_BASELINE_ROWS = [
    _result("baseline", "baseline", 101, _val(0.5)),
    _result("baseline", "baseline", 102, _val(0.502)),
    _result("baseline", "baseline", 103, _val(0.501)),
]


def _stage(spec: dict[str, Any], rows: list[dict[str, Any]], tamper: bool = False) -> dict[str, Any]:
    """Fixture ledger archive: the approval envelope plus pre-recorded trial results (pure data)."""
    campaign = _campaign(spec)
    events: list[tuple[str, str, str, dict[str, Any]]] = [
        ("campaign_approved", campaign, "-", {"spec_hash": campaign_hash(spec), "approver": "arbiter"}),
        *[("trial_result", campaign, str(row.get("trial") or "-"), dict(row)) for row in rows],
    ]
    return {"files": ledger_files(events, tamper=tamper)}


# ---- module helpers -------------------------------------------------------

def _campaign(spec: dict[str, Any]) -> str:
    return str(spec.get("id") or "unnamed-campaign")


def _safe_list(fn: Callable[[], list[Any]]) -> list[Any]:
    try:
        return list(fn())
    except (OSError, ValueError, KeyError, TypeError):
        return []


def _safe_text(fn: Callable[[], str]) -> str:
    try:
        return str(fn())
    except (OSError, ValueError, KeyError, TypeError):
        return ""


def _top_candidate(decisions: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    def key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, str]:
        trial, decision = item
        delta = decision.get("mean_delta")
        return (_RANK.get(str(decision.get("verdict")), 0), float(delta) if delta is not None else float("-inf"), trial)

    return sorted(decisions.items(), key=key)[-1] if decisions else None


def _finite_number(item: Any) -> bool:
    """True for a finite int/float measurement (bool is never a measurement)."""
    return not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(float(item))


def result_problems(result: dict[str, Any]) -> list[str]:
    """AR-IN-006 shape problems for one trial-result payload (crashes must carry no metric values)."""
    problems: list[str] = []
    for field in _RESULT_FIELDS:
        if field not in result:
            problems.append(f"result is missing field {field!r}")
    trial, role = result.get("trial"), result.get("role")
    if "trial" in result and (not isinstance(trial, str) or not trial.strip()):
        problems.append(f"trial must be a non-empty string: {trial!r}")
    if "role" in result and role not in _RESULT_ROLES:
        problems.append(f"role must be one of {list(_RESULT_ROLES)}: {role!r}")
    if "seed" in result and (isinstance(result.get("seed"), bool) or not isinstance(result.get("seed"), int)):
        problems.append(f"seed must be an int: {result.get('seed')!r}")
    if result.get("status") not in {"ok", "crash"}:
        problems.append(f"status must be 'ok' or 'crash': {result.get('status')!r}")
    if "limited" in result and not isinstance(result.get("limited"), bool):
        problems.append(f"limited must be a bool: {result.get('limited')!r}")
    steps = result.get("steps")
    if "steps" in result and (isinstance(steps, bool) or not isinstance(steps, int) or steps < 0):
        problems.append(f"steps must be an int >= 0: {steps!r}")
    fingerprint = result.get("eval_policy_fingerprint")
    if "eval_policy_fingerprint" in result and (not isinstance(fingerprint, str) or not fingerprint):
        problems.append("eval_policy_fingerprint must be a non-empty string")
    metrics = result.get("metrics")
    if "metrics" in result and not isinstance(metrics, dict):
        problems.append("metrics must be a mapping name -> {value, se}")
    elif isinstance(metrics, dict):
        for name, point in metrics.items():
            if not isinstance(point, dict) or "value" not in point:
                problems.append(f"metric {name!r} must carry {{value, se}}")
                continue
            for field in ("value", "se"):
                item = point.get(field)
                if item is None and field == "se":
                    continue  # an unreported standard error is allowed
                if not _finite_number(item):
                    problems.append(f"metric {name!r} {field} must be a finite number: {item!r}")
        if result.get("status") == "crash" and any(
            isinstance(point, dict) and point.get("value") is not None for point in metrics.values()
        ):
            problems.append("a crash result carries metric values; crashes are never evidence")
    return problems
