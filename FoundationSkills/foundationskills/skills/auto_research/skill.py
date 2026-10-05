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

import json
import math
import subprocess
import time
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
from foundationskills.interfaces.fs.emit_trial import emit_trial, submit_trial
from foundationskills.interfaces.fs.fabric import (
    DEFAULT_TTL_S,
    FABRIC_CACHE_NAME,
    check_fabric,
    launch_blocking,
    probe_fabric,
)
from foundationskills.interfaces.fs.launch import LaunchRefused, launch as fs_launch
from foundationskills.interfaces.fs.sacct import query_job_gpu_hours
from foundationskills.skills.auto_research.accept import decide, noise_floor
from foundationskills.skills.auto_research.accounting import campaign_usage
from foundationskills.skills.auto_research.campaign import (
    campaign_hash,
    check_launch,
    check_spec,
    launch_token,
    scan_command_text,
    scan_rendered,
)
from foundationskills.skills.auto_research.envelope import (
    BUDGET_MISMATCH,
    ENVELOPE_MISSING,
    MIRROR_KEYS,
    budget_snapshot,
    envelope_check,
    envelope_token,
    trial_launch_token,
)
from foundationskills.skills.auto_research.jobs import JOB_ID_RE, cancel_jobs, owned_job_ids, submitted_jobs
from foundationskills.skills.auto_research.ledger import Ledger, canonical, ledger_files, sha256_hex
from foundationskills.skills.auto_research.propose import propose
from .concurrency import concurrency_check, reserve_check
from .locks import closing_check
from .claims import ROOT as CLAIM_ROOT
from .claims import build_claim, champion, claim_entries, derive_chain, reference_rows
from .seeds import phase_of, phase_problems, set_status

_ACTIONS = ("check", "launch", "record", "propose", "close", "envelope", "submit", "cancel", "claim")
_RESULT_ROLES = ("baseline", "candidate", "confirm")
_RESULT_FIELDS = ("trial", "role", "seed", "status", "limited", "steps", "eval_policy_fingerprint", "metrics")
_RANK = {"accepted_gain": 4, "accepted_flat": 3, "no_gain": 2, "rejected_regress": 1, "unmeasured": 0}

def _claim_live(entry: dict[str, Any]) -> bool:
    """An accepted chain entry (derive_chain marks a claim that lost a seed 'downgraded')."""
    return entry.get("status") == "accepted"


def _claim_reference(entry: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows a claim was earned against: baseline-role rows at the root, else its reference trial's rows."""
    ref = entry.get("reference")
    if ref == CLAIM_ROOT:
        return [row for row in results if row.get("role") == "baseline"]
    return [row for row in results if row.get("trial") == ref]


def _claim_well_formed(body: dict[str, Any]) -> bool:
    """A stored claim body as build_claim wrote it: scalar ids, a seed list and a stats mapping."""
    scalars = all(isinstance(body.get(key), str) and body.get(key) for key in ("claim_id", "prev", "trial", "reference"))
    seeds = body.get("seeds")
    return scalars and isinstance(seeds, list) and bool(seeds) and isinstance(body.get("stats", {}), dict)


def _decide(spec: dict[str, Any], reference: list[dict[str, Any]], rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """decide() over reference rows: an unmeasurable comparison reports unmeasured, it never raises."""
    try:
        return dict(decide(spec, reference, rows, metric))
    except (ArithmeticError, ValueError, TypeError, KeyError):
        return {"verdict": "unmeasured", "mean_delta": None, "tau": None, "n_pairs": 0, "reasons": ["decide_failed"]}


def _op_payloads(ledger: Ledger, campaign: str, op: str) -> list[dict[str, Any]]:
    """Ledger payloads of one op for a campaign in append order (a run carries its trial name)."""
    payloads: list[dict[str, Any]] = []
    for entry in _safe_list(lambda: ledger.entries()):
        if entry.get("op") != op or entry.get("campaign") != campaign:
            continue
        try:
            body = dict(ledger.payload(entry))
        except (OSError, ValueError, KeyError, TypeError):
            body = {}
        if not body.get("trial") and entry.get("trial") and entry.get("trial") != "-":
            body["trial"] = entry.get("trial")
        payloads.append(body)
    return payloads


def _claim_must_fire_fixtures() -> dict[str, dict[str, Any]]:
    """M2 fixtures (pure data): an incomplete confirm set (AR-RS-007) and a broken seed plan (AR-IN-007)."""
    fix = "arbiter"
    ar_spec = _spec()
    ar_campaign = _campaign(ar_spec)
    ar_ledger: list[tuple[str, str, str, dict[str, Any]]] = [
        ("campaign_approved", ar_campaign, "-", {"spec_hash": campaign_hash(ar_spec), "approver": fix}),
        ("launch_envelope", ar_campaign, "-", _env_payload(ar_spec, {"max_runs": 6, "gpu_hours_total": 24.0})),
    ]
    for run_seed in (101, 102, 103):
        ar_ledger.append(("trial_result", ar_campaign, "baseline", _result("baseline", "baseline", run_seed, _val(0.5))))
    for run_seed in (101, 102):  # only two of the three confirm seeds ran; 103 stays a named gap
        job = _job_payload(str(700 + run_seed), "t1", kind="eval_only")
        job.update({"seed": run_seed, "phase": "confirm", "gpu_hours_est": 1.0})
        ar_ledger.append(("job_submitted", ar_campaign, "t1", job))
        ar_ledger.append(("trial_result", ar_campaign, "t1", _result("t1", "confirm", run_seed, _val(0.9))))
    bad_spec = _spec(seeds={"baseline_repeats": 3, "confirm_repeats": 5, "seed_list": [101, 102, 103]})
    return {
        "AR-RS-007": {
            "files": ledger_files(ar_ledger),
            "request": {
                "action": "claim", "campaign_spec": ar_spec, "campaign_confirm": "{confirm}",
                "approver": fix, "ledger_dir": "{tmp}/ledger", "trial": "t1",
            },
        },
        "AR-IN-007": {
            "request": {
                "action": "check", "campaign_spec": bad_spec, "campaign_confirm": "{confirm}",
                "ledger_dir": "{tmp}/ledger",
            },
        },
    }


FINGERPRINT = "sha256:" + "a1" * 32
BASE_FINGERPRINT = "sha256:" + "b2" * 32

_RECOVERY = {
    "AR-IN-001": "set objective.metric to one of eval_policy.metrics",
    "AR-IN-002": "set base.model and base.fingerprint = 'sha256:<64 hex>' of the sealed base model",
    "AR-IN-003": "set budget.gpu_hours_total > 0 and budget.max_runs > 0",
    "AR-IN-004": "set eval_policy.fingerprint = 'sha256:<64 hex>' of the frozen eval policy",
    "AR-IN-005": "pick axes from campaign.AXIS_PATHS with a matching type/range (UNSUPPORTED_AXES refuse)",
    "AR-IN-006": "record {trial, role, seed, status, limited, steps, eval_policy_fingerprint, metrics}; no crash metrics",
    "AR-IN-007": "declare seeds.seed_list as distinct positive ints with 1 <= repeats per phase (confirm_repeats <= len(seed_list)), and 0 < cluster.max_in_flight <= budget.max_runs",
    "AR-AP-001": "pass campaign_confirm = the human-approved campaign_hash(spec) and a named approver",
    "AR-LN-001": "keep delta/shape/partition inside spec (base and model are locked campaign fields)",
    "AR-LN-002": "stay inside max_runs/gpu hours and keep the confirm reserve for confirm runs",
    "AR-LN-004": "only run time '10-00:00:00'; never pkill -u / scancel / killall or name a quarantined node",
    "AR-LN-005": "lower gpu_hours_est (or raise per_run_timeout_h within the 240h cluster limit)",
    "AR-LG-001": "record a new trial/seed pair; recorded results are immutable",
    "AR-LN-003": "wait for the IMEX fabric: a refused or unmeasured fabric never launches (the sbatch preamble is the in-job gate)",
    "AR-LN-006": "open a launch_envelope mirroring spec.budget, use the derived per-trial launch_token (fs-ar-trial-v1), and stop when runs or measured GPU hours are gone",
    "AR-LN-007": "cancel only job ids this campaign's ledger submitted (job_submitted); foreign ids never reach scancel",
    "AR-HO-006": "submit the noise-floor baseline repeats as kind 'eval_only' (decision 3): candidates keep confirm_repeats",
    "AR-LG-002": "campaigns never re-open: close is final (check/close stay readable, nothing else runs)",
    "AR-LN-008": "let a submitted run settle (record its result or cancel it) before submitting again",
    "AR-LN-009": "keep the run reserve for confirm-phase runs: screen with fewer repeats or confirm first",
    "AR-RS-007": "claim only a measured, complete confirm set whose decide() verdict is accepted_gain (record the missing seeds first)",
}


class AutoResearchSkill(BaseSkill):
    """Run a bounded experiment campaign: envelope consent, gated submit, safe cancel, record, decide, close."""

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
            "envelope": {"type": "object"},
            "trial_spec": {"type": "object"},
            "launch_token": {"type": "string"},
            "job_ids": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
            "fabric_ttl_s": {"type": "number"},
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
        RuleSpec("AR-IN-007",
                 "seed plan invalid: seed_list empty/duplicate/non-int, repeats < 1, confirm_repeats beyond the "
                 "seed list, or cluster.max_in_flight invalid/above max_runs",
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
        RuleSpec("AR-LN-003",
                 "IMEX fabric at submit is not 'ready' (refused or unmeasured); the named reason is carried in the message",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LN-006",
                 "envelope missing or forged, envelope budget != spec budget, the trial token is not derived, or the envelope budget is exhausted (runs or GPU hours)",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LN-007", "cancel names a job id this campaign's ledger never submitted (or requests no job ids)",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LN-008", "in-flight (running or pending) jobs at cluster.max_in_flight; unmeasured stations fail closed",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LN-009", "a baseline/screening submit would spend the run reserve kept for confirm-phase runs",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LG-001", "a recorded trial result would be overwritten (results are immutable; record a new trial)",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-LG-002", "a mutating action after campaign_closed (close is final: only check/close stay open)",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-RS-007",
                 "claim over an incomplete/unmeasured confirm set (missing, pending, crashed, limited or unpaired "
                 "seed) or of a decision that is not accepted_gain",
                 Severity.BLOCK, "input"),
        RuleSpec("AR-HO-001", "campaign closed with no accepted gain", Severity.BLOCK, "handoff"),
        RuleSpec("AR-HO-002", "best candidate breaches a guardrail band", Severity.BLOCK, "handoff"),
        RuleSpec("AR-HO-003", "ledger chain verification failed", Severity.BLOCK, "handoff"),
        RuleSpec("AR-HO-004", "load-bearing evidence missing (uncalibrated noise floor, screening-only seeds, limited or "
                              "crashed runs) - status UNMEASURED", Severity.WARN, "handoff"),
        RuleSpec("AR-HO-005", "accepted a flat-but-simpler change", Severity.INFO, "handoff"),
        RuleSpec("AR-HO-006",
                 "a noise-floor baseline repeat was submitted as a non-eval_only job (decision 3: the floor is eval-only repeats) - status UNMEASURED",
                 Severity.WARN, "handoff"),
    )
    fs_interface = FSInterface(
        entries=(),
        emits=("auto_research_report",),
        apis=("campaign_hash", "launch_token", "check_spec", "check_launch", "noise_floor", "decide", "propose", "Ledger"),
        notes=(
            "M1 submits emitted trials through training.emit behind the envelope budget gate (AR-LN-006), the "
            "IMEX fabric probe (AR-LN-003) and job ownership for cancel (AR-LN-007), all append-only over the "
            "same hash-chained ledger that writes auto_research_report."
        ),
    )

    # ---- constructor / helpers ---------------------------------------------

    def __init__(
        self,
        *,
        runner: Callable[..., Any] | None = None,
        launch_fn: Callable[..., dict[str, Any]] | None = None,
        fabric_probe: Callable[..., dict[str, Any]] | None = None,
        measure: Callable[[str], float | None] | None = None,
        clock: Callable[[], float] | None = None,
        state_fn: Callable[[str], str | None] | None = None,
    ) -> None:
        """Injectable connectors (None -> the real implementation; tests never see Slurm or sockets)."""
        super().__init__()
        self._runner: Callable[..., Any] = runner if runner is not None else subprocess.run
        self._launch_fn: Callable[..., dict[str, Any]] = launch_fn if launch_fn is not None else fs_launch
        self._fabric_probe: Callable[..., dict[str, Any]] = fabric_probe if fabric_probe is not None else probe_fabric
        self._measure: Callable[[str], float | None] = measure if measure is not None else query_job_gpu_hours
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._state_fn: Callable[[str], str | None] | None = state_fn

    @staticmethod
    def _ledger_dir(request: dict[str, Any], ctx: SkillContext) -> Path:
        return Path(str(request.get("ledger_dir") or (ctx.workdir / "ledger")))

    @classmethod
    def _ledger(cls, request: dict[str, Any], ctx: SkillContext) -> Ledger:
        return Ledger(cls._ledger_dir(request, ctx))

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
        # AR-LG-002 (M2): close is final - locks.closing_check owns the read-only allowlist (check/close),
        # and the refusal lands BEFORE the action-specific checks so nothing else is ever appended.
        closed = closing_check(_safe_list(lambda: ledger.entries()), _campaign(spec), action)
        if closed:
            return findings + [
                self.finding(rule_id, message, {"campaign": _campaign(spec), "action": action}, _RECOVERY[rule_id])
                for rule_id, message in closed
            ]
        if action != "check":
            findings.extend(self._check_approval(request, spec, ledger))
        if action == "launch":
            findings.extend(self._check_launch_request(request, spec, ledger))
        if action == "record":
            findings.extend(self._check_record_request(request, spec, ledger))
        if action == "envelope":
            findings.extend(self._check_envelope_request(request, spec))
        if action == "submit":
            findings.extend(self._check_submit_request(request, spec, ledger, ctx))
        if action == "cancel":
            findings.extend(self._check_cancel_request(request, spec, ledger))
        if action == "claim":
            findings.extend(self._check_claim_request(request, spec, ledger))
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

    def _check_envelope_request(self, request: dict[str, Any], spec: dict[str, Any]) -> list[Finding]:
        """AR-LN-006: the consented envelope must mirror spec.budget on the gate keys."""
        envelope = request.get("envelope")
        if not isinstance(envelope, dict):
            return [self.finding("AR-LN-006", ENVELOPE_MISSING, {}, _RECOVERY["AR-LN-006"])]
        if not _budget_mirror_ok(spec, envelope):
            return [
                self.finding(
                    "AR-LN-006", BUDGET_MISMATCH,
                    {
                        "spec_budget": dict(spec.get("budget") or {}),
                        "envelope_budget": dict(envelope.get("budget") or {}),
                    },
                    _RECOVERY["AR-LN-006"],
                )
            ]
        return []

    def _check_submit_request(
        self, request: dict[str, Any], spec: dict[str, Any], ledger: Ledger, ctx: SkillContext
    ) -> list[Finding]:
        """AR-LN-006 (envelope + derived token) -> AR-LN-001/002/004/005 (campaign) -> AR-LN-003 (fabric)."""
        findings: list[Finding] = []
        campaign = _campaign(spec)
        confirm = str(request.get("campaign_confirm") or "")
        trial_spec = dict(request.get("trial_spec") or {})
        target = {"trial": trial_spec.get("trial")}
        envelope = self._latest_envelope(ledger, campaign)
        # F1: the envelope gate meters EVERY run attempt of this campaign - the ledger launch payloads
        # (Ledger.launches, reconciled with the job entries so one run counts once) and usage that still
        # burns for an authorised run that never reported a job id (a synthetic {job_id: None} entry).
        launches, job_entries = self._budget_rows(ledger, campaign)
        if not envelope.get("envelope_token"):
            findings.append(self.finding("AR-LN-006", ENVELOPE_MISSING, dict(target), _RECOVERY["AR-LN-006"]))
        else:
            derived = envelope_token(confirm, dict(envelope.get("envelope") or {}))
            if derived != str(envelope.get("envelope_token") or ""):
                findings.append(
                    self.finding("AR-LN-006", "envelope_token_forged", dict(target), _RECOVERY["AR-LN-006"])
                )
            usage = campaign_usage(job_entries, measure=self._measure)
            for rule_id, reason in envelope_check(
                spec, envelope, trial_spec, launches, usage, str(request.get("launch_token") or "")
            ):
                findings.append(self.finding(rule_id, reason, dict(target), _RECOVERY.get(rule_id, "")))
        findings.extend(self._check_launch_request({"launch_spec": trial_spec}, spec, ledger))
        # G2 input-time gate: the argv that will run is rendered from the opaque train/eval request, so its
        # JSON serialisation is scanned with the SAME forbidden-command patterns that gate check_launch
        # (a command list and its json.dumps string never disagree about what is refused).
        for request_key in ("train_request", "eval_request"):
            block = trial_spec.get(request_key)
            if not isinstance(block, dict):
                continue
            for rule_id, message in scan_command_text(json.dumps(block, sort_keys=True)):
                findings.append(self.finding(rule_id, message, {**target, "request": request_key}, _RECOVERY[rule_id]))
        # M2 run-slot gates (A2/A3): the phase derives from the role alone, the in-flight queue comes from
        # THIS campaign's job_submitted payloads (ledger-derived and fail closed; ``state_fn`` only refines
        # what is live) and the run reserve is AR-LN-009's alone (the GPU-hours reserve stays AR-LN-002).
        for rule_id, message in phase_problems(trial_spec, spec):
            findings.append(self.finding(rule_id, message, dict(target), _RECOVERY[rule_id]))
        submitted = submitted_jobs(ledger, campaign)
        results = _safe_list(lambda: ledger.results(campaign))
        cancelled_ids = _cancelled_job_ids(
            [
                self._payload(ledger, entry)
                for entry in ledger.entries()
                if entry.get("op") == "job_cancelled" and entry.get("campaign") == campaign
            ]
        )
        for rule_id, message in concurrency_check(spec, submitted, results, cancelled_ids, self._state_fn):
            findings.append(self.finding(rule_id, message, dict(target), _RECOVERY[rule_id]))
        for rule_id, message in reserve_check(spec, phase_of(trial_spec.get("role")), launches):
            findings.append(self.finding(rule_id, message, dict(target), _RECOVERY[rule_id]))
        state = self._fabric_state(request, ctx)
        if launch_blocking(str(state.get("state") or "unmeasured")):
            findings.append(
                self.finding(
                    "AR-LN-003", str(state.get("reason") or "fabric_unmeasured"),
                    {**target, "state": state.get("state")}, _RECOVERY["AR-LN-003"],
                )
            )
        return findings

    def _check_cancel_request(
        self, request: dict[str, Any], spec: dict[str, Any], ledger: Ledger
    ) -> list[Finding]:
        """AR-LN-007: only the job ids THIS campaign submitted may ever reach scancel."""
        campaign = _campaign(spec)
        job_ids = [str(item) for item in request.get("job_ids") or []]
        if not job_ids:
            return [self.finding("AR-LN-007", "no job ids", {}, _RECOVERY["AR-LN-007"])]
        owned = owned_job_ids(ledger, campaign)
        findings: list[Finding] = []
        for job_id in job_ids:
            if not JOB_ID_RE.match(job_id):
                findings.append(
                    self.finding(
                        "AR-LN-007", f"job {job_id} is not a valid job id for ledger {campaign}",
                        {"job_id": job_id}, _RECOVERY["AR-LN-007"],
                    )
                )
            elif job_id not in owned:
                findings.append(
                    self.finding(
                        "AR-LN-007", f"job {job_id} is not owned by ledger {campaign}",
                        {"job_id": job_id}, _RECOVERY["AR-LN-007"],
                    )
                )
        return findings

    def _latest_envelope(self, ledger: Ledger, campaign: str) -> dict[str, Any]:
        rows = [
            self._payload(ledger, entry)
            for entry in ledger.entries()
            if entry.get("op") == "launch_envelope" and entry.get("campaign") == campaign
        ]
        return rows[-1] if rows else {}

    def _fabric_state(self, request: dict[str, Any], ctx: SkillContext) -> dict[str, Any]:
        """Cached IMEX state at the FIXED path ``<ledger_dir>/fabric.json`` (F2: no request override).

        ``fabric_ttl_s`` is clamped to ``fabric.DEFAULT_TTL_S`` (min(requested, 300)); a non-finite,
        negative or non-numeric TTL falls back to the default - a caller may shorten the cache but can
        never extend it (a future-dated refusal stays sticky until the cache file is removed).
        """
        cache = self._ledger_dir(request, ctx) / FABRIC_CACHE_NAME
        return check_fabric(
            cache, ttl_s=_clamp_ttl_s(request.get("fabric_ttl_s")), clock=self._clock, probe=self._fabric_probe
        )

    def _budget_rows(self, ledger: Ledger, campaign: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """(launches, job_entries) for the envelope gate: every run attempt burns, whatever recorded it.

        F1 - ``launches`` are the ledger launch payloads (``Ledger.launches``) reconciled with this
        campaign's ``launch_authorised``/``job_submitted`` payloads one row per run (a run is never
        counted twice), and ``job_entries`` is what ``campaign_usage`` meters: the ``job_submitted``
        payloads plus one synthetic ``{"job_id": None, "trial", "gpu_hours_est"}`` per ``launch_authorised``
        payload whose trial has no job entry - an authorised run that never reported a job id still burns
        its declared budget, and its synthetic entry is never measured (see accounting.campaign_usage).
        """
        jobs = submitted_jobs(ledger, campaign)
        authorisations = [
            self._payload(ledger, e)
            for e in ledger.entries()
            if e.get("op") == "launch_authorised" and e.get("campaign") == campaign
        ]
        launches = _dedup_runs([_safe_list(lambda: ledger.launches(campaign)), authorisations, jobs])
        job_entries: list[dict[str, Any]] = [dict(payload) for payload in jobs]
        claimed = {str(payload.get("trial") or "") for payload in jobs}
        for payload in authorisations:
            trial = _run_trial(payload)
            if trial not in claimed:
                job_entries.append({"job_id": None, "trial": trial, "gpu_hours_est": payload.get("gpu_hours_est")})
        return launches, job_entries

    # ---- run ---------------------------------------------------------------

    # ---- M2 claims (A6: claims are explicit; close never writes them) ------

    def _claim_context(self, spec: dict[str, Any], ledger: Ledger) -> dict[str, Any]:
        """Ledger-derived claim chain: results, run provenance, the chain, its drops and the reference rows."""
        campaign = _campaign(spec)
        results = _safe_list(lambda: ledger.results(campaign))
        job_entries = _op_payloads(ledger, campaign, "job_submitted")
        results_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
        for row in results:
            results_by_key[(row.get("trial"), row.get("seed"))] = row
            if row.get("role") == "baseline":
                results_by_key[("baseline", row.get("seed"))] = row
        claims, malformed = [], []
        for index, body in enumerate(_op_payloads(ledger, campaign, "claim")):
            if _claim_well_formed(body):
                claims.append(body)
            else:  # never re-link or re-hash a body the ledger did not store intact
                malformed.append(f"claim_malformed:{index}")
        chain, drops = derive_chain(claim_entries(claims), results_by_key)
        drops = [*malformed, *drops]
        return {
            "campaign": campaign,
            "results": results,
            "jobs": job_entries,
            "results_by_key": results_by_key,
            "claims": claims,
            "chain": chain,
            "drops": drops,
            "ref": reference_rows(chain, results),
        }

    def _claim_inputs(self, request: dict[str, Any], spec: dict[str, Any], ledger: Ledger) -> dict[str, Any]:
        """Shared claim inputs (check and run recompute them): the confirm set, its decision, the chain link."""
        trial = str(request.get("trial") or "")
        state = self._claim_context(spec, ledger)
        confirm_rows = [row for row in state["results"] if row.get("trial") == trial and row.get("role") == "confirm"]
        status = set_status(spec, trial, state["results"], state["jobs"], state["ref"], phase="confirm")
        metric = str(dict(spec.get("objective") or {}).get("metric") or "")
        decision = _decide(spec, list(state["ref"]), confirm_rows, metric)
        state.update({"trial": trial, "confirm_rows": confirm_rows, "status": status, "decision": decision})
        return state

    def _check_claim_request(self, request: dict[str, Any], spec: dict[str, Any], ledger: Ledger) -> list[Finding]:
        """AR-RS-007: claim one explicit earned gain over a complete, measured confirm set."""
        trial = str(request.get("trial") or "")
        if not trial:
            return [self.finding("AR-RS-007", "claim_trial_missing", {"trial": trial}, _RECOVERY["AR-RS-007"])]
        data = self._claim_inputs(request, spec, ledger)
        chain = list(data["chain"])
        if any(_claim_live(entry) and str(entry.get("trial") or "") == trial for entry in chain):
            return [
                self.finding(
                    "AR-RS-007", "claim_exists:" + trial,
                    {"trial": trial, "claims": [str(entry.get("claim_id") or "") for entry in chain]},
                    _RECOVERY["AR-RS-007"],
                )
            ]
        status = dict(data["status"])
        if not status.get("complete"):
            return [
                self.finding(
                    "AR-RS-007", "claim_set_incomplete:" + trial,
                    {"trial": trial, "drops": list(status.get("drops") or []), "status": status},
                    _RECOVERY["AR-RS-007"],
                )
            ]
        if str(data["decision"].get("verdict") or "") != "accepted_gain":
            return [
                self.finding(
                    "AR-RS-007", "claim_no_gain:" + trial,
                    {"trial": trial, "decision": data["decision"]},
                    _RECOVERY["AR-RS-007"],
                )
            ]
        return []

    def _claim(self, request: dict[str, Any], ledger: Ledger, campaign: str) -> SkillResult:
        """One explicit ``claim`` op per earned gain, chained to the current champion claim."""
        spec = dict(request.get("campaign_spec") or {})
        data = self._claim_inputs(request, spec, ledger)
        trial = str(data["trial"])
        top = champion(list(data["chain"]))
        prev = str(top.get("claim_id") or CLAIM_ROOT)
        claim = build_claim(spec, trial, data["decision"], data["status"], prev)
        if not claim:
            return SkillResult(Status.REFUSED, {"refused": "claim_no_gain:" + trial, "trial": trial})
        ledger.append("claim", campaign, trial, dict(claim))
        chain, _ = derive_chain(claim_entries([*data["claims"], dict(claim)]), data["results_by_key"])
        return SkillResult(Status.PASS, {"claim_id": claim["claim_id"], "champion": champion(chain), "prev": prev})

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
        if action == "envelope":
            return self._envelope(request, ledger, spec, campaign, spec_hash)
        if action == "submit":
            return self._submit(request, ctx, ledger, spec, campaign, spec_hash)
        if action == "cancel":
            return self._cancel(request, ledger, campaign)
        if action == "record":
            return self._record(request, ledger, campaign)
        if action == "claim":
            return self._claim(request, ledger, campaign)
        if action == "propose":
            cards = propose(
                spec, _safe_list(lambda: ledger.results(campaign)),
                dict(request.get("current") or {}), list(request.get("symptoms") or []),
                launches=_safe_list(lambda: ledger.launches(campaign)),
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

    def _envelope(
        self, request: dict[str, Any], ledger: Ledger, spec: dict[str, Any], campaign: str, spec_hash: str
    ) -> SkillResult:
        """AR-LN-006 consent: one human approval derives the campaign envelope token (budget-decremented)."""
        confirm = str(request.get("campaign_confirm") or spec_hash)
        envelope = dict(request.get("envelope") or {})
        envelope.pop("envelope_token", None)  # the token is derived over the payload, never supplied
        token = envelope_token(confirm, envelope)
        ledger.append(
            "launch_envelope", campaign, "-",
            {
                "envelope_token": token,
                "spec_hash": spec_hash,
                "envelope": envelope,
                "budget": dict(envelope.get("budget") or {}),
                "approver": str(request.get("approver") or ""),
            },
        )
        return SkillResult(Status.PASS, {"envelope_token": token})

    def _submit(
        self, request: dict[str, Any], ctx: SkillContext, ledger: Ledger, spec: dict[str, Any],
        campaign: str, spec_hash: str,
    ) -> SkillResult:
        """Emit ONE trial and gate its submission: the full submit gate first, then the rendered argv +
        sbatch gate, then the derived token, then the IMEX fabric, then launch (G1 + G2).

        G1 - defence in depth: the IDENTICAL ``_check_submit_request`` gate ``check_inputs`` runs is re-run
        right here (a direct ``run()`` call can never skip it). Any finding refuses BEFORE ``emit_trial``,
        ``submit_trial``, ``launch_fn`` or the runner is touched (refusal = the first finding message, the
        findings attached), and an absent/empty envelope token never derives an expected trial token.
        G2 - what actually runs is the render: the emitted argv + ``sbatch`` are scanned with the same
        AR-LN-004 forbidden-command patterns and AR-LN-005 quarantined-node names before ``submit_trial``.
        """
        trial_spec = dict(request.get("trial_spec") or {})
        trial = str(trial_spec.get("trial") or "-")
        gate_findings = self._check_submit_request(request, spec, ledger, ctx)
        if gate_findings:
            first_message = _finding_message(gate_findings[0])
            return SkillResult(
                Status.REFUSED,
                {"refused": first_message, "drops": [_finding_message(f) for f in gate_findings]},
                (),
                tuple(gate_findings),
                refusal=first_message,
            )
        envelope = self._latest_envelope(ledger, campaign)
        fact = emit_trial({"trial_spec": trial_spec, "campaign_spec": spec})
        emit_drops = _ordered_unique([str(item) for item in fact.get("drops") or []])
        if not fact.get("executable"):
            missing = [str(item) for item in fact.get("missing") or []]
            return SkillResult(Status.REFUSED,
                               {"missing": missing, "drops": _ordered_unique([*missing, *emit_drops])}, (), (),
                               refusal="trial emit not executable: " + ("; ".join(missing) or "unspecified"))
        fs_launch_spec = fact.get("fs_launch_spec") or {}
        sbatch = fs_launch_spec.get("sbatch")
        rendered_hits = scan_rendered(fs_launch_spec, sbatch if isinstance(sbatch, str) else "")
        if rendered_hits:
            rule_id, message = rendered_hits[0]
            return SkillResult(
                Status.REFUSED,
                {"refused": message, "drops": [hit for _, hit in rendered_hits], "emit_drops": emit_drops},
                (),
                tuple(self.finding(rid, hit, {"trial": trial}, _RECOVERY.get(rid, "")) for rid, hit in rendered_hits),
                refusal=message,
            )
        envelope_token_value = str(envelope.get("envelope_token") or "")
        expected_token = (trial_launch_token(envelope_token_value, trial_spec) if envelope_token_value else "")
        # F2: a FRESH, uncached IMEX probe immediately before submission - the cached gate can only admit;
        # what submit_trial launches against is probed right now (submit_trial refuses unless 'ready').
        fabric = self._fabric_probe(host="master", port=8081)
        try:
            out = submit_trial(
                fact["fs_launch_spec"],
                expected_token=expected_token,
                supplied_token=str(request.get("launch_token") or ""),
                confirm=str(fact.get("confirm") or spec_hash),
                fabric=fabric,
                launch_fn=self._launch_fn,
                runner=self._runner,
            )
        except LaunchRefused as exc:
            message = str(exc).strip() or "launch refused"
            head = message.split(" ", 1)[0]
            rule_id = head if head in _RECOVERY else "AR-LN-006"
            return SkillResult(
                Status.REFUSED,
                {"refused": message, "drops": [message]},
                (),
                (self.finding(rule_id, message, {"trial": trial}, _RECOVERY.get(rule_id, "")),),
                refusal=message,
            )
        job_id = str(out.get("job_id") or "").strip()
        launches, job_entries = self._budget_rows(ledger, campaign)
        usage = campaign_usage(job_entries, measure=self._measure)
        budget_after = budget_snapshot(envelope, launches, usage, trial_spec)["budget_after"]
        ledger.append(
            "launch_authorised", campaign, trial,
            {
                "launch_spec": trial_spec,
                "spec_hash": sha256_hex(canonical(trial_spec)),
                "launch_token": expected_token,
                "gpu_hours_est": float(trial_spec.get("gpu_hours_est") or 0.0),
            },
        )
        if not job_id:  # no job_submitted without a job id: the account stays UNMEASURED ('no_job_id')
            return SkillResult(Status.UNMEASURED, {"job_id": "", "budget_after": budget_after, "drops": ["no_job_id"]}, (), ())
        ledger.append(
            "job_submitted", campaign, trial,
            {
                "trial": trial,
                "job_id": job_id,
                "seed": trial_spec.get("seed"),
                "phase": phase_of(trial_spec.get("role")),
                "launch_token": expected_token,
                "kind": str(trial_spec.get("kind") or ""),
                "gpu_hours_est": float(trial_spec.get("gpu_hours_est") or 0.0),
                "budget_after": budget_after,
                "emit_drops": emit_drops,
                "submitted_at": float(self._clock()),
            },
        )
        return SkillResult(Status.PASS, {"job_id": job_id, "budget_after": budget_after, "drops": emit_drops})

    def _cancel(self, request: dict[str, Any], ledger: Ledger, campaign: str) -> SkillResult:
        """Cancel only what THIS campaign submitted (AR-LN-007): foreign ids never reach the runner."""
        out = cancel_jobs(
            ledger, campaign, [str(item) for item in request.get("job_ids") or []],
            reason=str(request.get("reason") or "operator"), runner=self._runner,
        )
        payload = {"cancelled": list(out.get("cancelled") or []), "drops": list(out.get("drops") or [])}
        findings = tuple(
            self.finding(rule_id, message, {}, _RECOVERY.get(rule_id, ""))
            for rule_id, message in out.get("findings") or ()
        )
        if int(out.get("returncode") or 0) != 0:
            return SkillResult(Status.RED, payload, (), findings)
        return SkillResult(Status.PASS, payload, (), findings)

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
        claims_state = self._claim_context(spec, ledger)
        chain = list(claims_state["chain"])
        chain_drops = list(claims_state["drops"])
        reference = list(claims_state["ref"])
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in results:
            if row.get("role") != "baseline":
                grouped.setdefault(str(row.get("trial") or "?"), []).append(row)
        earned = {str(entry.get("trial")): _claim_reference(entry, results) for entry in chain if _claim_live(entry)}
        decisions = {
            trial: _decide(spec, earned.get(trial, reference), rows, metric) for trial, rows in sorted(grouped.items())
        }
        claimed_gains = {str(entry.get("trial") or "") for entry in chain}  # A6: a downgraded claim is still a claim
        unclaimed_gains = sorted(
            trial for trial, decision in decisions.items()
            if decision.get("verdict") == "accepted_gain" and trial not in claimed_gains
        )
        chain_champion = champion(chain)
        floor = noise_floor(baseline, metric, int(dict(spec.get("seeds") or {}).get("baseline_repeats", 3)))
        gains = [d for d in decisions.values() if d["verdict"] == "accepted_gain"]
        flats = [d for d in decisions.values() if d["verdict"] == "accepted_flat"]
        regressions = [d for d in decisions.values() if d["verdict"] == "rejected_regress"]
        top = _top_candidate(decisions)
        best_trial, best = top if top else (None, None)

        job_entries = submitted_jobs(ledger, campaign)
        usage = campaign_usage(job_entries, measure=self._measure)
        per_job = dict(usage.get("per_job") or {})
        measured = sum(float(v.get("gpu_hours") or 0.0) for v in per_job.values() if v.get("source") == "measured_sacct")
        declared = sum(float(v.get("gpu_hours") or 0.0) for v in per_job.values() if v.get("source") == "declared_est")
        budgets = {
            "planned": dict(spec.get("budget") or {}),
            "used_gpu_hours": float(usage.get("used_gpu_hours") or 0.0),
            "measured_gpu_hours": measured,
            "declared_gpu_hours": declared,
            "per_job": per_job,
            "drops": [*list(usage.get("drops") or []), *chain_drops],
            "runs": len(launches),
        }
        ho006 = _baseline_non_eval_trials(
            results, job_entries, has_envelope=bool(self._latest_envelope(ledger, campaign))
        )
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
        elif chain_drops or ho006:
            outcome, status = "unmeasured", Status.UNMEASURED
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
            if problems:
                recommendation = (
                    "restore an intact ledger: chain verification failed; no verdict is trustworthy until it verifies"
                )
            elif chain_drops:
                recommendation = (
                    "re-run the claimed gains before adopting anything (downgraded "
                    + ", ".join(chain_drops)
                    + ")"
                )
            elif ho006:
                recommendation = (
                    "resubmit the noise-floor baseline repeats as kind 'eval_only' jobs (decision 3): "
                    + _ho006_summary(ho006)
                )
            else:
                recommendation = "collect more evidence: the noise floor or the paired repeats are missing"
        else:
            recommendation = "keep the baseline; no candidate beat tau"

        for drop in chain_drops:
            dropped = str(drop).split(":", 1)[1] if ":" in str(drop) else str(drop)
            findings.append(
                self.finding(
                    "AR-HO-004",
                    "claim " + dropped + " downgraded at close: " + str(drop) + " (a claimed gain lost its evidence)",
                    {"drop": drop, "claims": chain},
                    "record (or re-run) the lost seed result and claim the gain again",
                )
            )
        if ho006:
            findings.append(
                self.finding(
                    "AR-HO-006",
                    f"noise-floor baseline repeat(s) {_ho006_summary(ho006)}"
                    " (decision 3: the floor is eval-only repeats)",
                    {"trials": ho006},
                    _RECOVERY["AR-HO-006"],
                )
            )
        preseal = ledger.head()
        entries = _safe_list(lambda: ledger.entries())
        already_sealed = bool(entries) and entries[-1].get("op") == "campaign_closed" and entries[-1].get("campaign") == campaign
        if not already_sealed:  # a repeated close re-reports the sealed chain; it never seals twice
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
            "champion": chain_champion,
            "claims": chain,
            "unclaimed_gains": unclaimed_gains,
            "drops": list(budgets.get("drops") or []),
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
            "champion": chain_champion,
            "claims": chain,
            "unclaimed_gains": unclaimed_gains,
            "drops": list(budgets.get("drops") or []),
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
        # M1 fixtures (pure data): one consented envelope + derived per-trial token and a sticky future-dated
        # REFUSAL fabric cache seeded at the fixed <ledger_dir>/fabric.json path (a cache hit, so no socket
        # opens - no fixture passes fabric_ttl_s or relies on a fresh 'ready' cache). A submit fixture whose
        # rule under test is not the fabric gate therefore also fires AR-LN-003 alongside. No fixture touches
        # the network or a subprocess.
        ar_spec = _spec()
        ar_campaign = _campaign(ar_spec)
        ar_gate = _env_payload(ar_spec, {"max_runs": 6, "gpu_hours_total": 24.0})
        ar_gate_trial = _trial_payload(kind="eval_only", trial="fabric-gate")
        ar_noenv_trial = _trial_payload(kind="eval_only", trial="need-an-envelope")
        ar_approval = [
            ("campaign_approved", ar_campaign, "-", {"spec_hash": campaign_hash(ar_spec), "approver": fix}),
        ]
        ar_envelope_ledger = [*ar_approval, ("launch_envelope", ar_campaign, "-", ar_gate)]
        ar_jobs_ledger = [
            *ar_approval,
            ("job_submitted", ar_campaign, "-", _job_payload("123456", "baseline", kind="eval_only")),
        ]
        # M2 fixtures (pure data): one closed campaign (close is final) and two guarded submits over ONE
        # consented envelope. The sticky future-dated REFUSAL fabric cache is reused so the submit fixtures
        # co-fire AR-LN-003 exactly like the M1 ones (a cache hit: no socket opens). The budget is the
        # spec default (max_runs 6, reserve_frac 0.3 -> ceil(1.8) = 2 reserved runs), and its hours are
        # kept small enough that AR-LN-002's hour check stays out of the way.
        ar_m2_spec = _spec()
        ar_m2 = _campaign(ar_m2_spec)
        ar_m2_env = _env_payload(ar_m2_spec, {"max_runs": 6, "gpu_hours_total": 24.0})
        ar_m2_trial = _trial_payload(kind="eval_only", trial="m2-screen")
        ar_m2_trial.update({"role": "candidate", "seed": 105, "gpu_hours_est": 1.0})
        ar_m2_jobs: list[tuple[str, str, str, dict[str, Any]]] = []
        ar_m2_done: list[tuple[str, str, str, dict[str, Any]]] = []
        for run_seed in (101, 102, 103, 104):
            job = _job_payload(str(200 + run_seed), "m2-screened", kind="eval_only")
            job.update({"seed": run_seed, "phase": "screening", "gpu_hours_est": 1.0})
            ar_m2_jobs.append(("job_submitted", ar_m2, "m2-screened", job))
            # every screened run has its trial_result: resolved terminal (and still ONE used run slot)
            ar_m2_done.append(
                ("trial_result", ar_m2, "m2-screened", _result("m2-screened", "candidate", run_seed, _val(0.4)))
            )
        ar_cap_spec = _spec()
        ar_cap_spec["cluster"] = {**(ar_cap_spec.get("cluster") or {}), "max_in_flight": 1, "max_nodes": 2}
        ar_cap = _campaign(ar_cap_spec)
        ar_cap_env = _env_payload(ar_cap_spec, {"max_runs": 6, "gpu_hours_total": 24.0})
        ar_cap_trial = _trial_payload(kind="eval_only", trial="trial-b")
        ar_cap_trial.update({"role": "candidate", "seed": 105, "gpu_hours_est": 1.0})
        ar_cap_job = _job_payload("111", "m2-held", kind="eval_only")
        ar_cap_job.update({"seed": 101, "phase": "confirm", "gpu_hours_est": 1.0})
        ar_m2_fabric = {"ledger/fabric.json": _fabric_cache("refused", 4102444800.0, "fabric_refused")}
        ar_ho006_rows = [*_BASELINE_ROWS, _result("t1", "candidate", 101, _val(0.9)),
                         _result("t1", "candidate", 102, _val(0.902)), _result("t1", "candidate", 103, _val(0.901))]
        ar_ho006_ledger = [
            *ar_approval,
            ("job_submitted", ar_campaign, "-", _job_payload("123456", "baseline", kind="train")),
            *(("trial_result", ar_campaign, str(row.get("trial") or "-"), dict(row)) for row in ar_ho006_rows),
        ]
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
            "AR-LN-003": {
                "files": {
                    **ledger_files(ar_envelope_ledger),
                    "ledger/fabric.json": _fabric_cache("refused", 4102444800.0, "fabric_refused"),
                },
                "request": {
                    "action": "submit", "campaign_spec": ar_spec, "campaign_confirm": "{confirm}",
                    "approver": fix, "ledger_dir": "{tmp}/ledger", "trial_spec": ar_gate_trial,
                    "launch_token": trial_launch_token(ar_gate["envelope_token"], ar_gate_trial),
                },
            },
            "AR-LN-006": {
                "files": {
                    **ledger_files(ar_approval),
                    "ledger/fabric.json": _fabric_cache("refused", 4102444800.0, "fabric_refused"),
                },
                "request": {
                    "action": "submit", "campaign_spec": ar_spec, "campaign_confirm": "{confirm}",
                    "approver": fix, "ledger_dir": "{tmp}/ledger", "trial_spec": ar_noenv_trial,
                    "launch_token": "fs-ar-trial-v1-not-derived",
                },
            },
            "AR-LN-007": {
                "files": ledger_files(ar_jobs_ledger),
                "request": {
                    "action": "cancel", "campaign_spec": ar_spec, "campaign_confirm": "{confirm}",
                    "approver": fix, "ledger_dir": "{tmp}/ledger", "job_ids": ["999999"], "reason": "operator",
                },
            },
            "AR-HO-006": {
                "files": ledger_files(ar_ho006_ledger),
                "request": {
                    "action": "close", "campaign_spec": ar_spec, "campaign_confirm": "{confirm}",
                    "approver": fix, "ledger_dir": "{tmp}/ledger", "stop_reason": "budget exhausted",
                },
            },
            "AR-LG-002": {
                "files": ledger_files([
                    ("campaign_approved", ar_m2, "-", {"spec_hash": campaign_hash(ar_m2_spec), "approver": fix}),
                    ("campaign_closed", ar_m2, "-", {"reason": "budget exhausted", "decided": "unmeasured"}),
                ]),
                "request": {
                    "action": "envelope", "campaign_spec": ar_m2_spec, "campaign_confirm": "{confirm}",
                    "approver": fix, "ledger_dir": "{tmp}/ledger",
                    "envelope": {"budget": {"max_runs": 6, "gpu_hours_total": 24.0}, "scope": "one campaign"},
                },
            },
            "AR-LN-008": {
                "files": {
                    **ledger_files([
                        ("campaign_approved", ar_cap, "-", {"spec_hash": campaign_hash(ar_cap_spec), "approver": fix}),
                        ("launch_envelope", ar_cap, "-", ar_cap_env),
                        ("job_submitted", ar_cap, "m2-held", ar_cap_job),
                    ]),
                    **ar_m2_fabric,
                },
                "request": {
                    "action": "submit", "campaign_spec": ar_cap_spec, "campaign_confirm": "{confirm}",
                    "approver": fix, "ledger_dir": "{tmp}/ledger", "trial_spec": ar_cap_trial,
                    "launch_token": trial_launch_token(ar_cap_env["envelope_token"], ar_cap_trial),
                },
            },
            "AR-LN-009": {
                "files": {
                    **ledger_files([
                        ("campaign_approved", ar_m2, "-", {"spec_hash": campaign_hash(ar_m2_spec), "approver": fix}),
                        ("launch_envelope", ar_m2, "-", ar_m2_env),
                        *ar_m2_jobs,
                        *ar_m2_done,
                    ]),
                    **ar_m2_fabric,
                },
                "request": {
                    "action": "submit", "campaign_spec": ar_m2_spec, "campaign_confirm": "{confirm}",
                    "approver": fix, "ledger_dir": "{tmp}/ledger", "trial_spec": ar_m2_trial,
                    "launch_token": trial_launch_token(ar_m2_env["envelope_token"], ar_m2_trial),
                },
            },
            **_claim_must_fire_fixtures(),
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


def _env_payload(spec: dict[str, Any], budget: dict[str, Any], approver: str = "arbiter") -> dict[str, Any]:
    env = {"budget": dict(budget), "scope": "one campaign"}
    return {
        "envelope_token": envelope_token(campaign_hash(spec), env),
        "spec_hash": campaign_hash(spec),
        "envelope": env,
        "budget": dict(budget),
        "approver": approver,
    }


def _job_payload(job_id: str, trial: str, kind: str = "train", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trial": trial,
        "job_id": job_id,
        "launch_token": "fs-ar-trial-v1-fixture",
        "kind": kind,
        "gpu_hours_est": 8.0,
        "budget_after": {"runs_left": 2, "hours_left": 16.0},
        "submitted_at": 1750000000.0,
    }
    payload.update(overrides)
    return payload


def _trial_payload(kind: str = "eval_only", **overrides: Any) -> dict[str, Any]:
    trial: dict[str, Any] = {
        "trial": "t-sub", "role": "candidate", "kind": kind, "delta": {"optim.lr": 5e-4},
        "seed": 101, "nodes": 1, "gpus_per_node": 8, "partition": "rally", "time": "10-00:00:00",
        "commands": ["uv run train.py --config campaign.yaml"], "gpu_hours_est": 4.0,
        "eval_request": {"command": ["eval", "run", "gsm8k"]},
    }
    trial.update(overrides)
    return trial


def _fabric_cache(state: str, ts: float, reason: str) -> str:
    return json.dumps({"master:8081": {"state": state, "ts": ts, "reason": reason}}, sort_keys=True)


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


def _budget_mirror_ok(spec: dict[str, Any], envelope: Any) -> bool:
    """True when envelope.budget mirrors spec.budget on the gate keys (numeric, NaN-safe)."""
    if not isinstance(envelope, dict):
        return False
    block = envelope.get("budget")
    granted = dict(block) if isinstance(block, dict) else {}
    plan_block = spec.get("budget")
    plan = dict(plan_block) if isinstance(plan_block, dict) else {}
    for key in MIRROR_KEYS:
        try:
            if float(granted.get(key)) != float(plan.get(key)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _baseline_non_eval_trials(
    results: list[dict[str, Any]], job_entries: list[dict[str, Any]], *, has_envelope: bool = False
) -> list[str]:
    """AR-HO-006 evidence labels for the noise-floor baseline repeats (decision 3).

    ``<trial>`` when the repeat shipped as a job that is NOT kind 'eval_only'. When the campaign ledger
    carries any ``launch_envelope`` entry, also ``baseline_provenance_unknown:<trial>`` for a baseline whose
    trial has NO ``job_submitted`` entry - its provenance is then unprovable. Ledgers with no launch_envelope
    keep the M0 backward compat: those manual records never fire.
    """
    jobs = [dict(job) for job in job_entries if isinstance(job, dict)]
    submitted = {str(job.get("trial") or "") for job in jobs}
    submitted_as = {str(job.get("trial") or "") for job in jobs if str(job.get("kind") or "") != "eval_only"}
    baselines = {str(row.get("trial") or "") for row in results if row.get("role") == "baseline"}
    labels = sorted(name for name in (baselines & submitted_as) if name)
    if has_envelope:
        labels += [f"baseline_provenance_unknown:{name}" for name in sorted(baselines - submitted) if name]
    return labels


def _ho006_summary(labels: list[str]) -> str:
    """Tail naming every AR-HO-006 label: '<trial> shipped as non-eval_only jobs' and each
    ``baseline_provenance_unknown:<trial>`` entry verbatim."""
    kinds = [label for label in labels if not label.startswith("baseline_provenance_unknown:")]
    unknown = [label for label in labels if label.startswith("baseline_provenance_unknown:")]
    parts: list[str] = []
    if kinds:
        parts.append(", ".join(kinds) + " shipped as non-eval_only jobs")
    parts.extend(unknown)
    return ", ".join(parts)


def _run_trial(payload: dict[str, Any]) -> str | None:
    """The run name of a launch/job payload (``launch_spec.trial`` / ``trial``); ``None`` when unknown."""
    trial = payload.get("trial")
    if isinstance(trial, str) and trial.strip():
        return trial
    spec = payload.get("launch_spec")
    if isinstance(spec, dict):
        trial = spec.get("trial")
        if isinstance(trial, str) and trial.strip():
            return trial
    return None


def _cancelled_job_ids(payloads: list[dict[str, Any]]) -> list[str]:
    """Job ids named by this campaign's ``job_cancelled`` payloads (a single ``job_id`` or a ``job_ids`` list)."""
    ids: list[str] = []
    rows: list[Any]
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        rows = payload.get("job_ids")
        if isinstance(rows, (list, tuple)):
            for item in rows:
                if isinstance(item, dict):
                    item = item.get("job_id")
                if isinstance(item, (str, int)):
                    ids.append(str(item))
        single = payload.get("job_id")
        if isinstance(single, (str, int)):
            ids.append(str(single))
    return _ordered_unique(ids)


def _run_key(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    """The run identity of a payload: ``(trial, seed)`` when the row names a seed, else the trial alone.

    A row without a seed keeps the trial-only key, so M0/M1 ledger rows reconcile exactly as before,
    while a multi-seed confirm set of ONE trial stays one run per seed (task 6, F1).
    """
    trial = _run_trial(payload)
    if trial is None:
        return None
    seed = payload.get("seed")
    spec = payload.get("launch_spec")
    if seed is None and isinstance(spec, dict):  # launch_authorised rows carry the seed in launch_spec
        seed = spec.get("seed")
    return (trial, seed) if seed is not None else (trial,)


def _dedup_runs(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """One row per run across the given payload groups (reconciled by (trial, seed): first row wins), F1."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for payload in group:
            if not isinstance(payload, dict):
                continue
            key = _run_key(payload)
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            merged.append(dict(payload))
    return merged


def _ordered_unique(items: list[str]) -> list[str]:
    """List with duplicates removed and the original order kept (F4 drop names)."""
    return list(dict.fromkeys(items))


def _finding_message(finding: Finding) -> str:
    """The finding's human message (``Finding`` exposes ``message``); never raises (G1 refusal text)."""
    for attr in ("message", "msg", "text", "reason"):
        value = getattr(finding, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(finding)


def _clamp_ttl_s(requested: Any) -> float:
    """Clamped IMEX probe TTL (F2): at most ``fabric.DEFAULT_TTL_S``.

    ``min(requested, DEFAULT_TTL_S)`` for a finite non-negative number; anything else (non-numeric,
    non-finite, negative) falls back to ``DEFAULT_TTL_S``.
    """
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return float(DEFAULT_TTL_S)
    if not math.isfinite(value) or value < 0.0:
        return float(DEFAULT_TTL_S)
    return min(value, float(DEFAULT_TTL_S))


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
