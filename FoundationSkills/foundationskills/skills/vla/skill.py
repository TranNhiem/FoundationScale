"""VlaSkill (A4.10, re-scoped): plan and judge VLA work strictly against FoundationScale's own VLA registry.

FoundationScale ships a VLA lane (``foundationscale.vla``) and a registry of VLA policies
(``foundationscale.upstream.vla_models.VLA_MODELS``). A row is SUPPORTED only when FS reproduced the upstream's
published LIBERO success rate on GB200 within a two-sample binomial tolerance. This skill advertises exactly that and
nothing more:

- ``plan`` (default):
  - an **eval** of a SUPPORTED row renders FS's own Level 3 step (``levels.level3_plan``: entry point
    ``foundationscale.vla.eval`` with the backend's published protocol) as an ``fs_launch_spec``. It is launched only
    through ``fskills launch --confirm <hash>``;
  - a **fine-tune / RL / driving** request is UNMEASURED: FS has adapters for it but no SUPPORTED row and no acceptance
    tolerance. The result names the FS adapter and the measurement gap, and renders no spec;
  - a policy with no registry row is REFUSED, naming the SUPPORTED ids.
- ``verify``: read the report ``foundationscale.vla.eval`` wrote and judge it with FS's own ``levels.judge`` (|measured
  - card| <= tolerance). Any episode that raised makes it UNMEASURED: FS keeps such an episode in the denominator as a
  failure, so the rate is a lower bound, not a measurement. Out of tolerance is RED. It writes a ``vla_verdict``.

Exit contract: 96 REFUSED (VL-IN-001..005), 95 UNMEASURED (VL-HO-001/002), 5 RED (VL-HO-003), 0 PASS.
"""

from __future__ import annotations

import json
import sys
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
from foundationskills.core.provenance import make_provenance
from foundationskills.core.status import Status

MODES = ("eval", "finetune", "rl", "driving")
# FS adapters that render fine-tune / RL / driving launches (pure renderers). None has a SUPPORTED registry row.
ADAPTERS = {
    ("finetune", "gr00t"): "foundationscale.vla.adapters.gr00t.finetune.Gr00tFinetuneSpec",
    ("finetune", "openpi"): "foundationscale.vla.adapters.openpi.finetune.OpenPiFinetuneSpec",
    ("finetune", "longwam"): "foundationscale.vla.adapters.longwam.finetune.LongWamFinetuneSpec",
    ("rl", "*"): "foundationscale.vla.adapters.verlvla.rl.VerlVlaRlSpec",
    ("driving", "*"): "foundationscale.vla.adapters.alpamayo.sft.AlpamayoSftSpec",
}
_GAPS = {
    "finetune": "no fine-tune acceptance tolerance exists (the binomial margin is a rollout-rate margin; one measured "
                "fine-tune sits 3.8 pt under its card) - it needs seeded fine-tune replicas on GB200",
    "rl": "verl-vla RL has no registry row and no declared margin (published curves are success counts) - it needs one "
          "GB200 reproduction through FS-owned eval",
    "driving": "Alpamayo driving has no registry row (its evidence is eval_loss, not minADE) - it needs one GB200 "
               "reproduction through FS's driving metrics",
}


def _registry() -> tuple[Any, Any, Any, Any]:
    """FS's VLA registry and Level 3 helpers (imported lazily: FS may be absent here)."""
    from foundationscale.upstream.levels import judge, level3_plan  # noqa: PLC0415
    from foundationscale.upstream.models import registry_problems  # noqa: PLC0415
    from foundationscale.upstream.vla_models import VLA_MODELS  # noqa: PLC0415

    return VLA_MODELS, level3_plan, judge, registry_problems


class VlaSkill(BaseSkill):
    name = "vla"
    version = "0.1.0"
    description = (
        "Plans and judges vision-language-action (VLA) policy work strictly against FoundationScale's VLA registry: an "
        "eval of a SUPPORTED policy (LIBERO reproductions of gr00t, openpi, Long-WAM on GB200) becomes a launch spec "
        "behind the plan hash, and its report is judged against the published success rate and tolerance. Fine-tune, "
        "RL and driving requests are UNMEASURED (adapters exist, nothing reproduced) and unregistered policies are "
        "REFUSED. Do NOT use for LLM/VLM language training (fskills-training) or text benchmarks (fskills-evaluation)."
    )
    scope = Scope(model_types=("vlm",), families="any", stages=("sft", "rl"), algorithms=(),
                  methods=("full", "lora"), hardware=("gb200",))
    consumes = ("checkpoint",)  # a VLA policy checkpoint (registry row); never reachable from raw data alone
    produces = ("fs_launch_spec", "vla_verdict")
    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "action": {"type": "string", "enum": ["plan", "verify"]},
            "model_id": {"type": "string", "minLength": 1},
            "mode": {"type": "string", "enum": list(MODES)},
            "checkpoint": {"type": "string", "minLength": 1},
            "out": {"type": "string", "minLength": 1},
            "serve_cmd": {"type": "string"},
            "report": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {"type": "object", "additionalProperties": True}
    rules = (
        RuleSpec("VL-IN-001", "the policy has no row in FoundationScale's VLA registry (the refusal names the SUPPORTED "
                 "ids)", Severity.BLOCK, "input"),
        RuleSpec("VL-IN-002", "FoundationScale's VLA registry is not importable here, or registry_problems(VLA_MODELS) "
                 "is non-empty", Severity.BLOCK, "input"),
        RuleSpec("VL-IN-003", "plan eval: no report path (out), or out is an existing directory", Severity.BLOCK, "input"),
        RuleSpec("VL-IN-004", "verify: the report is missing or unreadable, or lacks success_rate_pct/episode_errors",
                 Severity.BLOCK, "input"),
        RuleSpec("VL-IN-005", "verify: the report's backend or suite does not match the registry row being judged",
                 Severity.BLOCK, "input"),
        RuleSpec("VL-HO-001", "verify: an episode raised instead of reaching a verdict (kept in the denominator as a "
                 "failure, so the rate is a lower bound) - UNMEASURED, never PASS", Severity.WARN, "handoff"),
        RuleSpec("VL-HO-002", "fine-tune / RL / driving request, or a registry row that is not SUPPORTED: no "
                 "reproduction or acceptance tolerance exists - UNMEASURED, never advertised", Severity.WARN, "handoff"),
        RuleSpec("VL-HO-003", "verify: |measured - published| exceeds the row's tolerance - RED", Severity.BLOCK,
                 "handoff"),
    )
    fs_interface = FSInterface(entries=("fskills vla",), emits=("fs_launch_spec", "vla_verdict"),
                               apis=("foundationscale.upstream.levels.level3_plan", "levels.judge", "VLA_MODELS"),
                               notes="Renders FS's own Level 3 eval step; never re-implements a verdict.")

    def _row(self, model_id: str) -> tuple[Any, Any]:
        models, plan, _, _ = _registry()
        row = next((m for m in models if m.id == model_id), None)
        step = next((s for s in plan(models) if s.model_id == model_id), None)
        return row, step

    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        try:
            models, _, _, problems = _registry()
            issues = problems(models)
        except Exception as exc:  # noqa: BLE001 - FS missing or broken: nothing can be judged
            return [self.finding("VL-IN-002", f"FoundationScale's VLA registry is not importable: {exc}", {},
                                 "install FoundationScale (pip install -e <repo>) and re-run")]
        if issues:
            return [self.finding("VL-IN-002", f"registry_problems(VLA_MODELS): {issues}", {"problems": issues},
                                 "fix the registry in FoundationScale first")]
        supported = sorted(m.id for m in models if str(getattr(m.status, "value", m.status)) == "supported")
        model_id = str(request.get("model_id") or "")
        if model_id not in {m.id for m in models}:
            return [self.finding("VL-IN-001", f"policy {model_id!r} is not in FoundationScale's VLA registry; SUPPORTED: "
                                 f"{', '.join(supported)}", {"supported": supported},
                                 "pick a SUPPORTED id, or reproduce the policy in FS and register it first")]
        action = str(request.get("action") or "plan")
        if action == "verify":
            report = self._report(request)
            if report is None or "success_rate_pct" not in report or "episode_errors" not in report:
                return [self.finding("VL-IN-004", f"missing input: report {request.get('report')!r} is missing or lacks "
                                     "success_rate_pct/episode_errors", {}, "run the eval spec, then verify its report")]
            row, step = self._row(model_id)
            if report.get("backend") != step.backend or (report.get("plan") or {}).get("suite", step.manifest) != step.manifest:
                return [self.finding("VL-IN-005", f"report backend/suite {report.get('backend')}/"
                                     f"{(report.get('plan') or {}).get('suite')} does not match {model_id} "
                                     f"({step.backend}/{step.manifest})", {}, "verify the report against its own row")]
            return []
        if str(request.get("mode") or "eval") == "eval":
            out = request.get("out")
            if not out or Path(str(out)).is_dir():
                return [self.finding("VL-IN-003", f"missing input: eval report path out={out!r} (a file, not a dir)", {},
                                     "pass out=<report.json>")]
        return []

    @staticmethod
    def _report(request: dict[str, Any]) -> dict[str, Any] | None:
        try:
            data = json.loads(Path(str(request.get("report") or "")).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        model_id = str(request["model_id"])
        row, step = self._row(model_id)
        stamp = make_provenance(self.name, self.version, {"model_id": model_id})
        if str(request.get("action") or "plan") == "verify":
            return self._verify(request, ctx, step, stamp)
        mode = str(request.get("mode") or "eval")
        status = str(getattr(row.status, "value", row.status))
        if mode != "eval" or status != "supported":
            adapter = ADAPTERS.get((mode, step.backend)) or ADAPTERS.get((mode, "*"))
            gap = _GAPS.get(mode, f"registry row {model_id} is {status}, not SUPPORTED")
            payload = {"model_id": model_id, "mode": mode, "fs_adapter": adapter, "gap": gap, "registry_status": status}
            return SkillResult(Status.UNMEASURED, payload, provenance=stamp, findings=(self.finding(
                "VL-HO-002", f"{mode} for {model_id}: {gap}" + (f"; FS renders it with {adapter}" if adapter else ""),
                payload, "measure it on GB200 and register the result in FS before claiming it"),))
        model = str(request.get("checkpoint") or step.upstream_ref)
        args = [a.format(model=model, manifest=step.manifest, out=str(request["out"])) for a in step.args]
        if request.get("serve_cmd"):
            args += ["--serve-cmd", str(request["serve_cmd"])]
        spec = {"stage_name": f"vla-eval-{model_id}", "entry": "fskills-vla-eval",
                "argv": [sys.executable, "-m", step.entry_point, *args], "env": {}, "sbatch": None,
                "dry_run_argv": None, "expected_outputs": [str(request["out"])], "executable": True, "missing": None,
                "cwd": None, "vla": {"model_id": model_id, "backend": step.backend, "suite": step.manifest,
                                     "card_value": step.card_value, "tolerance": step.tolerance,
                                     "profile": step.profile, "report": str(request["out"])}}
        ref = write_artifact(Artifact("fs_launch_spec", f"vla-{model_id}", spec, stamp), ctx.artifacts_dir)
        return SkillResult(Status.PASS, {"spec": spec, "spec_path": ref.path,
                                         "next": f"fskills hash --spec {ref.path} (read it yourself), then fskills "
                                                 f"launch --spec {ref.path} --confirm <hash> --no-submit, then fskills "
                                                 f"vla verify --model-id {model_id} --report {request['out']}"},
                           artifacts=(ref,), provenance=stamp)

    def _verify(self, request: dict[str, Any], ctx: SkillContext, step: Any, stamp: Any) -> SkillResult:
        _, _, judge, _ = _registry()
        report = self._report(request) or {}
        errors = int(report.get("episode_errors") or 0)
        measured = float(report["success_rate_pct"])
        verdict = judge(step, measured)
        record = {"model_id": step.model_id, "suite": step.manifest, "card_value": step.card_value,
                  "tolerance": step.tolerance, "measured": measured, "gap": round(verdict.gap, 4),
                  "episode_errors": errors, "trials": report.get("trials"), "successes": report.get("successes"),
                  "wilson95_pct": report.get("wilson95_pct"), "report": str(request.get("report"))}
        findings: list[Finding] = []
        if errors or report.get("measured") is False:
            record["verdict"] = "UNMEASURED"
            findings.append(self.finding("VL-HO-001", f"{errors} episode(s) raised; {measured:.2f}% is a lower bound, "
                                         "not a measurement", {"episode_errors": errors}, "fix the server/adapter and "
                                         "re-run every episode"))
        elif not verdict.passed:
            record["verdict"] = "RED"
            findings.append(self.finding("VL-HO-003", f"{step.model_id}: measured {measured:.2f} vs published "
                                         f"{step.card_value} (gap {verdict.gap:+.2f}, tolerance {step.tolerance})",
                                         {"gap": verdict.gap}, "do not claim the reproduction; inspect the protocol"))
        else:
            record["verdict"] = "PASS"
        ref = write_artifact(Artifact("vla_verdict", f"vla-{step.model_id}", record, stamp), ctx.artifacts_dir)
        status = {"PASS": Status.PASS, "RED": Status.RED, "UNMEASURED": Status.UNMEASURED}[record["verdict"]]
        return SkillResult(status, record, artifacts=(ref,), findings=tuple(findings), provenance=stamp)

    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        return []  # every VL-HO verdict is decided in run(), next to its measured evidence

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"vla {type(failure).__name__}: {failure}" if isinstance(failure, BaseException)
                   else f"vla {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(symptom=symptom,
                         likely_causes=("the policy is not registered (VL-IN-001)", "FS missing (VL-IN-002)",
                                        "fine-tune/RL/driving has no reproduction yet (VL-HO-002)",
                                        "an episode raised during the eval (VL-HO-001)"),
                         checks=("read the finding: it names the rule and the SUPPORTED ids",
                                 "open the eval report: episode_errors, success_rate_pct"),
                         recovery=("pick a SUPPORTED id", "re-run every episode", "reproduce and register in FS first"))

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        ev = {"action": "plan", "model_id": "openpi-pi05-libero", "mode": "eval", "out": "vla/openpi_spatial.json"}
        vf = {"action": "verify", "model_id": "openpi-pi05-libero", "report": "vla/openpi_spatial.json"}
        return {
            "VL-IN-001": {"request": {**ev, "model_id": "octo-base-libero"}, "expected": "REFUSED naming VL-IN-001"},
            "VL-IN-002": {"request": ev, "observed": "foundationscale not importable here",
                          "expected": "REFUSED naming VL-IN-002"},
            "VL-IN-003": {"request": {**ev, "out": ""}, "expected": "REFUSED naming VL-IN-003 (no report path)"},
            "VL-IN-004": {"request": vf, "observed": "report lacks success_rate_pct",
                          "expected": "REFUSED naming VL-IN-004"},
            "VL-IN-005": {"request": vf, "observed": "report backend is gr00t", "expected": "REFUSED naming VL-IN-005"},
            "VL-HO-001": {"request": vf, "observed": "12 episodes raised errors",
                          "expected": "UNMEASURED naming VL-HO-001 (the rate is a lower bound)"},
            "VL-HO-002": {"request": {**ev, "mode": "finetune"},
                          "expected": "UNMEASURED naming VL-HO-002 (no fine-tune reproduction)"},
            "VL-HO-003": {"request": vf, "observed": "measured 94.0, published 98.8",
                          "expected": "RED naming VL-HO-003 (outside the 1.4 tolerance)"},
        }
