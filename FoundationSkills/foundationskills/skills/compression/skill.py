"""CompressionSkill (A4.8 v1): RTN weight-only quantize-dequantize, preflighted, planned probe-first, ppl-verified.

Actions:
- ``check``: run ``preflight.py``, vendored verbatim from the estate skill ``model-compression`` (FORGE). Exit 1
  (FAIL) is REFUSED (CP-IN-002). Exit 2 (PROBE_ERROR), or any check that could not be evaluated (UNKN), is
  UNMEASURED (CP-HO-010): preflight itself exits 0 on UNKN, this skill does not.
- ``plan``: refuse formats outside v1 (CP-IN-003) and a missing or out-of-range pre-registered budget (CP-IN-004), run
  ``check``, then write an ``fs_launch_spec`` (interfaces/fs/emit_compress.py) running ``rtn_worker.py`` on one GPU.
  The user launches it with ``fskills hash --spec S`` then ``fskills launch --spec S --confirm <hash> --no-submit``.
  This skill never launches and never echoes a hash.
- ``verify``: judge the worker's results against the plan (judge.py) and write a ``compression_report``.

Exit contract: 96 REFUSED names the missing input and writes nothing (CP-IN-001..007). 5 RED for an implausible
baseline (CP-HO-002), a no-op quantization (CP-HO-006) or a ratio above budget (CP-HO-011). 95 UNMEASURED when the
worker did not measure (CP-HO-003), the instrument is not certified (CP-HO-001) or preflight could not tell (CP-HO-010).
0 PASS otherwise, always with the INFO finding CP-HO-005: v1 never measures efficiency (W3).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
from foundationskills.interfaces.fs.emit_compress import WORKER_ENV, compress_probes, emit_compress, preflight_argv
from foundationskills.skills.compression.judge import EFFICIENCY_NOTE, judge_results

ACTIONS = ("check", "plan", "verify")
DEFERRED = "gptq, awq, nvfp4, mxfp4, fp8, modelopt, llm-compressor, kv-cache quantization, QAT/distillation recovery"
_STATUS = {"PASS": Status.PASS, "RED": Status.RED, "UNMEASURED": Status.UNMEASURED}


def _subprocess_preflight(argv: list[str], timeout_s: int) -> tuple[int, dict[str, Any] | None]:
    """Run preflight.py under the worker's env (PYTHONNOUSERSITE=1); return (exit code, cap.json report or None).

    The artifact root is created first (preflight measures its free disk and crashes on a missing directory), and the
    full unfiltered output lands in ``preflight.log`` beside ``cap.json``: a grepped tail is never evidence.
    """
    cap = Path(argv[argv.index("--out") + 1])
    cap.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **WORKER_ENV}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        (cap.parent / "preflight.log").write_text(f"preflight did not run: {exc!r}\n", encoding="utf-8")
        return 2, None
    (cap.parent / "preflight.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    try:
        return proc.returncode, json.loads(cap.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return proc.returncode, None


class CompressionSkill(BaseSkill):
    name = "compression"
    version = "0.1.0"
    description = (
        "Model compression and quantization of a HF checkpoint (v1: RTN weight-only quantize-dequantize, int4/int8, "
        "group-wise): preflights the host (vendored preflight.py; FAIL/UNKN/PROBE_ERROR), plans probe-first and emits "
        "an fs_launch_spec running rtn_worker.py on one GPU (launched only with the plan hash the user types back), "
        "then "
        "verifies the measured perplexity ratio against a pre-registered budget into a compression_report. GPTQ/AWQ/"
        "NVFP4/FP8/modelopt/llm-compressor or speedup requests also come here and are refused as deferred (CP-IN-003). "
        "Do NOT use for LoRA merges (fskills-merging), serving (fskills-deployment) or benchmarks "
        "(fskills-evaluation)."
    )
    scope = Scope(model_types=("llm", "vlm"), families="any", stages=("sft", "preference", "rl"), algorithms=(),
                  methods=("full", "lora", "qlora"), hardware=("gb200", "dgx-h100", "local"))
    consumes = ("checkpoint",)
    produces = ("fs_launch_spec", "compression_report")
    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "action": {"type": "string"},
            "model": {"type": "string", "minLength": 1},
            "artifact_root": {"type": "string", "minLength": 1},
            "text_file": {"type": "string", "minLength": 1},
            "bits": {"type": "integer"},
            "group_size": {"type": "integer"},
            "budget_ppl_ratio": {"type": "number"},
            "results": {"type": "string", "minLength": 1},
            "format": {"type": "string"},
            "python": {"type": "string"},
            "spec": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {"type": "object", "additionalProperties": True}
    rules = (
        RuleSpec("CP-IN-001", "the model directory is missing or not a HF checkpoint, or a plan probe failed",
                 Severity.BLOCK, "input"),
        RuleSpec("CP-IN-002", "check: preflight.py exited 1 (FAIL)", Severity.BLOCK, "input"),
        RuleSpec("CP-IN-003", f"plan: a format outside v1 RTN weight-only int4/int8 (deferred: {DEFERRED})",
                 Severity.BLOCK, "input"),
        RuleSpec("CP-IN-004", "plan/verify: no pre-registered ppl-ratio budget, or the budget is outside (1.0, 2.0]",
                 Severity.BLOCK, "input"),
        RuleSpec("CP-IN-005", "verify: the spec or the worker results are missing, unreadable or of the wrong schema",
                 Severity.BLOCK, "input"),
        RuleSpec("CP-IN-006", "verify: the results are not from this plan (model, bits, group_size or text hash)",
                 Severity.BLOCK, "input"),
        RuleSpec("CP-IN-007", "the action is not check, plan or verify", Severity.BLOCK, "input"),
        RuleSpec("CP-HO-001", "verify: ppl harness A vs B disagree by more than 2% on the base model - UNMEASURED",
                 Severity.WARN, "handoff"),
        RuleSpec("CP-HO-002", "verify: the base ppl is not finite or <= 1 - RED", Severity.BLOCK, "handoff"),
        RuleSpec("CP-HO-003", "verify: the worker did not measure (CPU fallback, crash) - UNMEASURED",
                 Severity.WARN, "handoff"),
        RuleSpec("CP-HO-005", "every verify: efficiency is never measured by v1 (quantize-dequantize keeps bf16; W3)",
                 Severity.INFO, "handoff"),
        RuleSpec("CP-HO-006", "verify: the quantization was a no-op (nothing quantized or changed) - RED",
                 Severity.BLOCK, "handoff"),
        RuleSpec("CP-HO-010", "check: preflight PROBE_ERROR (exit 2) or a check it could not evaluate (UNKN) - "
                 "UNMEASURED, never PASS", Severity.WARN, "handoff"),
        RuleSpec("CP-HO-011", "verify: the ppl ratio is above the plan's pre-registered budget - RED",
                 Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("fskills compress",), emits=("fs_launch_spec", "compression_report"),
        apis=("emit_compress", "skills/compression/preflight.py", "skills/compression/rtn_worker.py"),
        notes="Plans and verifies; launching goes through `fskills launch --confirm` like every other spec.",
    )

    def __init__(self, preflight_runner: Callable[[list[str], int], tuple[int, dict[str, Any] | None]]
                 = _subprocess_preflight) -> None:
        super().__init__()
        self.preflight_runner = preflight_runner  # injectable: unit tests never probe a GPU

    # -- inputs ---------------------------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        action = str(request.get("action") or "")
        if action not in ACTIONS:
            return [self.finding("CP-IN-007", f"unknown action {action!r}; use one of {', '.join(ACTIONS)}",
                                 {"action": action}, "pass action check, plan or verify")]
        if action == "verify":
            if "compress" not in self._spec(request):
                return [self.finding("CP-IN-005", f"missing input: spec {request.get('spec')!r} is missing or is not "
                                     "a compress fs_launch_spec", {"spec": request.get("spec")},
                                     "pass the spec written by `fskills compress plan`")]
            return []
        if action == "plan":
            fmt = str(request.get("format") or "rtn").lower()
            if fmt != "rtn":
                return [self.finding("CP-IN-003", f"format {fmt!r} is outside v1 (RTN weight-only int4/int8); "
                                     f"deferred: {DEFERRED}", {"format": fmt},
                                     "use format rtn, or wait for the isolated-env backends")]
            budget = request.get("budget_ppl_ratio")
            if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not 1.0 < budget <= 2.0:
                return [self.finding("CP-IN-004", f"missing input: a pre-registered budget_ppl_ratio in (1.0, 2.0] "
                                     f"(got {budget!r})", {"budget_ppl_ratio": budget},
                                     "decide the acceptable compressed/base ppl ratio before measuring")]
        need = ["model", "artifact_root"] + (["text_file", "results"] if action == "plan" else [])
        missing = [k for k in need if not request.get(k)]
        if missing:
            return [self.finding("CP-IN-001", f"missing input: {', '.join(missing)}", {"missing": missing},
                                 f"pass {', '.join(need)}")]
        model = Path(request["model"]).expanduser()
        if not (model / "config.json").is_file():
            return [self.finding("CP-IN-001", f"model {model} is not a HF checkpoint directory (no config.json)",
                                 {"model": str(model)}, "pass a local HF model directory")]
        if action == "plan":
            gaps = compress_probes(model, Path(request["text_file"]).expanduser(), int(request.get("bits") or 4),
                                   int(request.get("group_size") or 128), Path(request["results"]).expanduser(),
                                   str(request.get("python") or sys.executable))
            return [self.finding("CP-IN-001", gap, {"gap": gap}, "fix the named input and re-plan") for gap in gaps]
        return []

    @staticmethod
    def _spec(request: dict[str, Any]) -> dict[str, Any]:
        try:
            data = json.loads(Path(str(request.get("spec") or "")).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data.get("payload", data) if isinstance(data, dict) else {}

    # -- run ------------------------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        action = str(request["action"])
        if action == "verify":
            return self._verify(request, ctx)
        checked = self._check(request)
        if action == "check" or checked.status is not Status.PASS:
            return checked
        bits, group = int(request.get("bits") or 4), int(request.get("group_size") or 128)
        spec = emit_compress(request["model"], request["text_file"], bits, group, float(request["budget_ppl_ratio"]),
                             request["results"], python=str(request.get("python") or sys.executable))
        stamp = make_provenance(self.name, self.version, {"model": spec["compress"]["model"], "bits": bits})
        ref = write_artifact(Artifact("fs_launch_spec", "compress", spec, stamp), ctx.artifacts_dir)
        payload = {"spec": spec, "spec_path": ref.path, "preflight": checked.payload,
                   "next": f"fskills hash --spec {ref.path}  (read the hash yourself), then fskills launch --spec "
                           f"{ref.path} --confirm <hash> --no-submit, then fskills compress verify --spec {ref.path}"}
        return SkillResult(Status.PASS, payload, artifacts=(ref,), provenance=stamp)

    def _check(self, request: dict[str, Any]) -> SkillResult:
        argv = preflight_argv(request["model"], request["artifact_root"], int(request.get("bits") or 4),
                              str(request.get("python") or sys.executable))
        code, cap = self.preflight_runner(argv, 900)
        checks = (cap or {}).get("checks") or []
        payload = {"preflight_exit": code, "overall": (cap or {}).get("overall"), "cap": argv[argv.index("--out") + 1]}
        stamp = make_provenance(self.name, self.version, {"model": request["model"], "preflight_exit": code})
        if code == 1:
            failed = [c.get("name") for c in checks if c.get("status") == "FAIL"]
            return self._refused(f"preflight FAIL: {failed}", (self.finding(
                "CP-IN-002", f"preflight.py exited 1; failing checks: {failed}", payload,
                "fix the host/toolchain the failing checks name, then re-run check"),))
        unknown = [c.get("name") for c in checks if c.get("status") == "UNKN"]
        if code != 0 or cap is None or unknown:
            why = f"unevaluated checks {unknown}" if unknown and code == 0 else f"PROBE_ERROR (exit {code})"
            return SkillResult(Status.UNMEASURED, payload, findings=(self.finding(
                "CP-HO-010", f"preflight could not tell: {why}", payload,
                "read the FULL preflight log, fix the probe, re-run check"),), provenance=stamp)
        return SkillResult(Status.PASS, payload, provenance=stamp)

    def _verify(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        compress = self._spec(request)["compress"]
        judged = judge_results(compress, compress.get("results"))
        rule, note, report = judged["rule"], judged["note"] or "", judged["report"]
        if judged["verdict"] == "REFUSED":
            return self._refused(f"{rule}: {note}", (self.finding(rule, note, {"results": compress.get("results")},
                                 "launch the plan's spec first, or re-plan"),))
        efficiency = self.finding("CP-HO-005", EFFICIENCY_NOTE, {}, "measure a served, real low-bit format later")
        findings = ((self.finding(rule, note, {"report": report}, "see the compression_report"),) if rule else ())
        stamp = make_provenance(self.name, self.version, {"results": compress.get("results")})
        ref = write_artifact(Artifact("compression_report", "rtn", report, stamp), ctx.artifacts_dir)
        return SkillResult(_STATUS[judged["verdict"]], report, artifacts=(ref,), findings=findings + (efficiency,),
                           provenance=stamp)

    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        return []  # every CP-HO verdict is decided in run(), where the measured evidence is

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"compression {type(failure).__name__}: {failure}" if isinstance(failure, BaseException)
                   else f"compression {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=("a path, budget or format is wrong (CP-IN-001/003/004)", "the host failed preflight "
                           "(CP-IN-002/CP-HO-010)", "the worker fell back to CPU or crashed (CP-HO-003)",
                           "the ppl harnesses disagree (CP-HO-001)", "the quantization cost too much (CP-HO-011)"),
            checks=("read the finding message: it names the rule", "open <artifact_root>/cap.json",
                    "open the worker results JSON: base_ppl_a/b, ppl_ratio, changed_fraction"),
            recovery=("fix the input and re-plan to a fresh results path", "fix the probe and re-run check",
                      "use int8 or a smaller group size, or re-register a budget before measuring again"),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        plan = {"action": "plan", "model": "models/gemma-4-E4B-it", "artifact_root": "runs/a48", "bits": 4,
                "text_file": "data/wikitext2_test.txt", "results": "runs/a48/rtn.json", "budget_ppl_ratio": 1.15}
        verify = {"action": "verify", "spec": "artifacts/fs_launch_spec.compress.json"}
        ok = "measured, certified, base_ppl 9.8"
        return {
            "CP-IN-001": {"request": {**plan, "model": "models/missing"}, "observed": "no config.json",
                          "expected": "REFUSED naming CP-IN-001"},
            "CP-IN-002": {"request": {**plan, "action": "check"},
                          "observed": "preflight exit 1: real_gpu_execution FAIL",
                          "expected": "REFUSED naming CP-IN-002"},
            "CP-IN-003": {"request": {**plan, "format": "gptq"}, "expected": "REFUSED naming CP-IN-003 (deferred)"},
            "CP-IN-004": {"request": {k: v for k, v in plan.items() if k != "budget_ppl_ratio"},
                          "expected": "REFUSED naming CP-IN-004 (no pre-registered budget)"},
            "CP-IN-005": {"request": verify, "observed": "results file missing",
                          "expected": "REFUSED naming CP-IN-005"},
            "CP-IN-006": {"request": verify, "observed": "results bits 8, plan bits 4",
                          "expected": "REFUSED naming CP-IN-006"},
            "CP-IN-007": {"request": {"action": "quantize", "model": "models/gemma-4-E4B-it"},
                          "observed": "the request's action field is 'quantize' (not check, plan or verify)",
                          "expected": "REFUSED naming CP-IN-007"},
            "CP-HO-001": {"request": verify, "observed": "measured, instrument_rel_diff 0.031",
                          "expected": "UNMEASURED naming CP-HO-001"},
            "CP-HO-002": {"request": verify, "observed": "measured, certified, base_ppl NaN",
                          "expected": "RED naming CP-HO-002"},
            "CP-HO-003": {"request": verify, "observed": "worker status UNMEASURED: no CUDA device",
                          "expected": "UNMEASURED naming CP-HO-003"},
            "CP-HO-005": {"request": verify, "observed": f"{ok}, changed_fraction 0.93, ppl_ratio 1.04, budget 1.15",
                          "expected": "PASS with INFO CP-HO-005 (efficiency not measured)"},
            "CP-HO-006": {"request": verify, "observed": f"{ok}, changed_fraction 0.0",
                          "expected": "RED naming CP-HO-006"},
            "CP-HO-010": {"request": {**plan, "action": "check"}, "observed": "preflight exit 0 with an UNKN check",
                          "expected": "UNMEASURED naming CP-HO-010"},
            "CP-HO-011": {"request": verify, "observed": f"{ok}, changed_fraction 0.93, ppl_ratio 1.31, budget 1.15",
                          "expected": "RED naming CP-HO-011"},
        }
