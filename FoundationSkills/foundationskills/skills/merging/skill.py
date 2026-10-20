"""MergingSkill (A4.7): fold a LoRA adapter into its base, planned probe-first, launched on a re-shown hash, verified twice.

Actions:
- ``plan`` (default): probe the merge (interfaces/fs/emit_merge.py) and write an ``fs_launch_spec`` that runs
  FoundationScale's own ``merge_lora.py``. The user launches it with ``fskills hash --spec S`` and then
  ``fskills launch --spec S --confirm <hash> --no-submit``. This skill never launches and never echoes a hash.
- ``verify``: after the launch, read the merge script's results (seven structural checks) and measure **numeric
  parity** (parity.py). The merged checkpoint must reproduce base + adapter on the logits, because structure alone
  passes a merge that lost the adapter's effect. A PASS writes the merged ``checkpoint``.

Exit contract: 96 REFUSED names the missing input and writes nothing (ME-IN-001..005). 5 RED when a structural check
failed (ME-HO-002) or parity is outside tolerance (ME-HO-003). 95 UNMEASURED when parity was not run or could not tell
merged from base (ME-HO-004). 0 PASS only when the structure and parity both pass.

Measured (A4.7, gemma-4-12B-it + the A4.4 adapter): every structural check passes for a bf16-stored merge, yet parity
gives ratio 0.237 and argmax agreement 0.915. bf16 storage rounds away LoRA updates smaller than the weight's precision,
and the loss is bit-identical whether the merge arithmetic ran in bf16 or fp32. A float32-stored merge gives ratio 4e-5
and agreement 1.0. In bf16 inference any merged copy scores held-out NLL 0.9391 against 0.9253 for base + adapter,
i.e. 2.5% of the adapter's gain is lost by merging. Hence float32 storage is the default and bf16 storage is a WARN.
"""

from __future__ import annotations

import json
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
from foundationskills.interfaces.fs.emit_merge import STORE_DTYPES, emit_merge, merge_probes

PARITY_SCRIPT = Path(__file__).with_name("parity.py")
_GAP_RULES = (  # emit_merge gap text -> the input rule that refuses it
    ("is not a model directory", "ME-IN-001"), ("has no adapter", "ME-IN-001"),
    ("was trained on", "ME-IN-002"), ("exists and is not empty", "ME-IN-003"),
)
_BF16_NOTE = ("bfloat16 storage rounds away LoRA updates smaller than the weight's precision (measured on gemma-4-12B-it: "
              "parity ratio 0.237, 2.5% of the adapter's held-out gain lost); parity will be RED at the default tolerance")


def _subprocess_parity(argv: list[str], timeout_s: int) -> dict[str, Any]:
    """Run parity.py; return its JSON report (the script writes it to --out)."""
    out = Path(argv[argv.index("--out") + 1])
    subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    return json.loads(out.read_text(encoding="utf-8"))


class MergingSkill(BaseSkill):
    name = "merging"
    version = "0.1.0"
    description = (
        "Folds a LoRA adapter into its base model as merged full weights: plans the merge probe-first and emits an "
        "fs_launch_spec running FoundationScale's merge_lora.py (launched only with the plan hash the user types back), "
        "then verifies the merged checkpoint structurally and by numeric logit parity against base + adapter before "
        "writing a checkpoint. Do NOT use to train adapters (fskills-training), score checkpoints on benchmarks "
        "(fskills-evaluation) or combine several full models (TIES/SLERP-style model soups are out of scope)."
    )
    scope = Scope(model_types=("llm", "vlm"), families="any", stages=("sft", "preference", "rl"), algorithms=(),
                  methods=("lora", "qlora"), hardware=("gb200", "dgx-h100", "local"))
    consumes = ("checkpoint",)
    produces = ("fs_launch_spec", "checkpoint")
    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "action": {"type": "string", "enum": ["plan", "verify"]},
            "base": {"type": "string", "minLength": 1},
            "adapter": {"type": "string", "minLength": 1},
            "out": {"type": "string", "minLength": 1},
            "store_dtype": {"type": "string", "enum": list(STORE_DTYPES)},
            "device_map": {"type": "string"},
            "spec": {"type": "string", "minLength": 1},
            "parity": {"type": "boolean"},
            "device": {"type": "string"},
            "max_ratio": {"type": "number", "exclusiveMinimum": 0},
            "min_agreement": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    output_schema: dict[str, Any] = {"type": "object", "additionalProperties": True}
    rules = (
        RuleSpec("ME-IN-001", "plan: base is not a model directory, or the adapter has no adapter_config.json/weights",
                 Severity.BLOCK, "input"),
        RuleSpec("ME-IN-002", "plan: the adapter was trained on a different base than the one declared",
                 Severity.BLOCK, "input"),
        RuleSpec("ME-IN-003", "plan: the output directory exists and is not empty (a merge never overwrites)",
                 Severity.BLOCK, "input"),
        RuleSpec("ME-IN-004", "plan: the merge cannot run here (FS merge script not found, peft/transformers/torch not "
                 "importable, or not enough free disk)", Severity.BLOCK, "input"),
        RuleSpec("ME-IN-005", "verify: the merge results JSON is missing or unreadable (the merge was not launched)",
                 Severity.BLOCK, "input"),
        RuleSpec("ME-HO-001", "plan: bfloat16 storage requested; sub-precision LoRA updates are rounded away (measured)",
                 Severity.WARN, "handoff"),
        RuleSpec("ME-HO-002", "verify: a structural check of the merge script failed (parameter counts, LoRA weights "
                 "left, processor) - RED", Severity.BLOCK, "handoff"),
        RuleSpec("ME-HO-003", "verify: numeric parity outside tolerance (merged does not reproduce base + adapter) - RED",
                 Severity.BLOCK, "handoff"),
        RuleSpec("ME-HO-004", "verify: parity not run, or the adapter's effect too small to tell merged from base - "
                 "UNMEASURED, never PASS", Severity.WARN, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("fskills merge",), emits=("fs_launch_spec", "checkpoint"),
        apis=("emit_merge", "examples/vlm-2xH200/scripts/merge_lora.py"),
        notes="Plans and verifies; launching goes through `fskills launch --confirm` like every other spec.",
    )

    def __init__(self, parity_runner: Callable[[list[str], int], dict[str, Any]] = _subprocess_parity) -> None:
        super().__init__()
        self.parity_runner = parity_runner  # injectable: unit tests never load a model

    # -- inputs ---------------------------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        if str(request.get("action") or "plan") == "verify":
            return self._check_verify(request)
        missing = [k for k in ("base", "adapter", "out") if not request.get(k)]
        if missing:
            return [self.finding("ME-IN-001", f"missing input: {', '.join(missing)}", {"missing": missing},
                                 "pass base, adapter and out")]
        gaps = merge_probes(Path(request["base"]).expanduser(), Path(request["adapter"]).expanduser(),
                            Path(request["out"]).expanduser(), str(request.get("store_dtype") or "float32"))
        findings = []
        for gap in gaps:
            rule = next((r for text, r in _GAP_RULES if text in gap), "ME-IN-004")
            findings.append(self.finding(rule, gap, {"gap": gap}, "fix the named input and re-plan"))
        return findings

    def _check_verify(self, request: dict[str, Any]) -> list[Finding]:
        spec = self._spec(request)
        results = Path(str((spec.get("merge") or {}).get("results") or ""))
        try:
            json.loads(results.read_text(encoding="utf-8"))["verification"]
        except (OSError, ValueError, KeyError, TypeError):
            return [self.finding("ME-IN-005", f"missing input: merge results {results} are missing or unreadable",
                                 {"results": str(results)},
                                 "launch the merge spec first: fskills hash --spec S, then fskills launch --spec S "
                                 "--confirm <hash> --no-submit")]
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
        if str(request.get("action") or "plan") == "verify":
            return self._verify(request, ctx)
        store = str(request.get("store_dtype") or "float32")
        spec = emit_merge(request["base"], request["adapter"], request["out"], store,
                          str(request.get("device_map") or "cpu"))
        stamp = make_provenance(self.name, self.version, {"base": spec["merge"]["base"], "adapter": spec["merge"]["adapter"]})
        ref = write_artifact(Artifact("fs_launch_spec", "merge", spec, stamp), ctx.artifacts_dir)
        payload = {"spec": spec, "spec_path": ref.path,
                   "next": f"fskills hash --spec {ref.path}  (read the hash yourself), then fskills launch --spec "
                           f"{ref.path} --confirm <hash> --no-submit, then fskills merge verify --spec {ref.path}"}
        findings = ([self.finding("ME-HO-001", _BF16_NOTE, {"store_dtype": store},
                                  "store float32, or serve base + adapter unmerged to keep the adapter's full effect")]
                    if store == "bfloat16" else [])
        return SkillResult(Status.PASS, payload, artifacts=(ref,), findings=tuple(findings), provenance=stamp)

    def _verify(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        spec = self._spec(request)
        merge = spec["merge"]
        results = json.loads(Path(merge["results"]).read_text(encoding="utf-8"))
        checks = results["verification"]["checks"]
        failed = sorted(k for k, ok in checks.items() if not ok)
        parity: dict[str, Any] | None = None
        if request.get("parity", True) and not failed:
            out = Path(merge["out"]).parent / f"{Path(merge['out']).name}.parity.json"
            argv = [sys.executable, str(PARITY_SCRIPT), "--base", merge["base"], "--adapter", merge["adapter"],
                    "--merged", merge["out"], "--out", str(out), "--device", str(request.get("device") or "cuda:0"),
                    "--max-ratio", str(request.get("max_ratio", 0.1)),
                    "--min-agreement", str(request.get("min_agreement", 0.95))]
            try:
                parity = self.parity_runner(argv, 7200)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                parity = {"verdict": "UNMEASURED", "reason": f"parity did not run: {type(exc).__name__}: {exc}"}
        payload = {"path": merge["out"], "run_manifest": None, "stage": "merge", "base_model": merge["base"],
                   "adapter": merge["adapter"], "store_dtype": merge["store_dtype"], "merge_results": merge["results"],
                   "structural_checks": checks, "parity": parity}
        stamp = make_provenance(self.name, self.version, {"results": merge["results"]})
        if failed:
            return SkillResult(Status.RED, payload, findings=(self.finding(
                "ME-HO-002", f"structural checks failed: {failed}", {"failed": failed},
                "inspect the merge log; re-plan into a fresh output directory"),), provenance=stamp)
        verdict = (parity or {}).get("verdict")
        if verdict == "RED":
            return SkillResult(Status.RED, payload, findings=(self.finding(
                "ME-HO-003", f"parity outside tolerance: {parity.get('reason')}", {"parity": parity},
                "store float32 (bf16 storage drops sub-precision LoRA updates), or serve base + adapter unmerged"),),
                provenance=stamp)
        if verdict != "PASS":
            return SkillResult(Status.UNMEASURED, payload, findings=(self.finding(
                "ME-HO-004", f"parity not measured: {(parity or {}).get('reason', 'parity disabled')}",
                {"parity": parity}, "run verify with parity on a GPU host"),), provenance=stamp)
        ref = write_artifact(Artifact("checkpoint", "merged", payload, stamp), ctx.artifacts_dir)
        return SkillResult(Status.PASS, payload, artifacts=(ref,), provenance=stamp)

    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        return []  # every ME-HO verdict is decided in run(), where the measured evidence is

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"merging {type(failure).__name__}: {failure}" if isinstance(failure, BaseException)
                   else f"merging {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=("a path or adapter_config is wrong (ME-IN-001/002/003)", "the toolchain or disk (ME-IN-004)",
                           "the merge was not launched before verify (ME-IN-005)",
                           "bf16 storage dropped the adapter's small updates (ME-HO-003)"),
            checks=("read the finding message: it names the rule and the gap", "open <out>.merge_results.json",
                    "open <out>.parity.json: ratio and argmax_agreement"),
            recovery=("fix the input and re-plan into a fresh directory", "store float32, or serve base + adapter",
                      "run verify on a GPU host with parity on"),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        merge = {"action": "plan", "base": "models/gemma-4-12B-it", "adapter": "runs/sft/final", "out": "merged/m1"}
        return {
            "ME-IN-001": {"request": {**merge, "adapter": "runs/missing"}, "observed": "adapter_config.json missing",
                          "expected": "REFUSED naming ME-IN-001"},
            "ME-IN-002": {"request": merge, "observed": "adapter trained on gemma-4-26B-A4B-it",
                          "expected": "REFUSED naming ME-IN-002 (a different base)"},
            "ME-IN-003": {"request": merge, "observed": "merged/m1 already holds files",
                          "expected": "REFUSED naming ME-IN-003 (never overwrite)"},
            "ME-IN-004": {"request": merge, "observed": "peft not importable here",
                          "expected": "REFUSED naming ME-IN-004"},
            "ME-IN-005": {"request": {"action": "verify", "spec": "artifacts/fs_launch_spec.merge.json"},
                          "observed": "merge results file missing", "expected": "REFUSED naming ME-IN-005"},
            "ME-HO-001": {"request": {**merge, "store_dtype": "bfloat16"},
                          "expected": "PASS with WARN ME-HO-001 (bf16 storage drops sub-precision updates)"},
            "ME-HO-002": {"request": {"action": "verify", "spec": "artifacts/fs_launch_spec.merge.json"},
                          "observed": "no_lora_weights_left_on_disk is false", "expected": "RED naming ME-HO-002"},
            "ME-HO-003": {"request": {"action": "verify", "spec": "artifacts/fs_launch_spec.merge.json"},
                          "observed": "parity ratio 0.237, agreement 0.915",
                          "expected": "RED naming ME-HO-003 (merged does not reproduce base + adapter)"},
            "ME-HO-004": {"request": {"action": "verify", "spec": "artifacts/fs_launch_spec.merge.json", "parity": False},
                          "expected": "UNMEASURED naming ME-HO-004 (parity not run)"},
        }
