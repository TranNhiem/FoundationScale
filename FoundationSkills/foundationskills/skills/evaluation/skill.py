"""EvalSkill: checkpoint -> lm-evaluation-harness -> one-sided policy verdict -> eval_report.json.

Exit contract: 96 REFUSED writes no report (every refusal names the missing
input); 5 RED when a measured benchmark breached its band or a configured
benchmark was skipped; 95 UNMEASURED when a benchmark could not run (dataset
not staged, harness error) or the run was ``limit``-ed; 0 PASS only when every
benchmark was measured against a fingerprint-matched baseline and none breached.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from foundationskills.core.artifacts import ArtifactRef
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
from foundationskills.core.status import Status
from foundationskills.skills.evaluation.baseline import BaselineCache, fingerprint, model_identity
from foundationskills.skills.evaluation.policy import Policy, PolicyError, TaskPolicy, compare, load_policy
from foundationskills.skills.evaluation.report import verdict, write_report
from foundationskills.skills.evaluation.resolve import Resolved, ResolveError, resolve, ships_chat_template
from foundationskills.skills.evaluation.runner import (
    PINNED_LM_EVAL,
    HarnessRequest,
    HarnessRunner,
    SubprocessLmEvalRunner,
    extract_metric,
    harness_version,
)


def dataset_staged(eval_cache: Path, dataset: str) -> bool:
    """True when an HF hub snapshot or a prepared ``datasets`` cache for ``dataset`` exists under the eval cache."""
    hub = eval_cache / "hub" / f"datasets--{dataset.replace('/', '--')}"
    prepared = eval_cache / "datasets" / dataset.replace("/", "___")
    return hub.is_dir() or prepared.is_dir()


class EvalSkill(BaseSkill):
    name = "evaluation"
    version = "0.1.0"
    description = (
        "Evaluate a checkpoint (full or PEFT adapter) with the pinned lm-evaluation-harness, offline, "
        "against its base model; one-sided stderr-band verdict into eval_report.json."
    )
    scope = Scope(
        model_types=("llm",),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("checkpoint",)
    produces = ("eval_report",)

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["benchmarks", "policy", "eval_cache", "out"],
        "additionalProperties": False,
        "properties": {
            "checkpoint": {"type": "string", "minLength": 1},
            "base": {"type": "string", "minLength": 1},
            "run_manifest": {"type": "string", "minLength": 1},
            "benchmarks": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            "policy": {"type": "string", "minLength": 1},
            "eval_cache": {"type": "string", "minLength": 1},
            "out": {"type": "string", "minLength": 1},
            "baseline_cache": {"type": "string", "minLength": 1},
            "num_fewshot": {"type": "integer", "minimum": 0},
            "seed": {"type": "integer"},
            "gen_kwargs": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1},
            "dtype": {"enum": ["bfloat16", "float16", "float32"]},
            "batch_size": {"type": "string"},
            "device": {"type": "string", "minLength": 1},
            "parallelize": {"type": "boolean"},
            "include_path": {"type": "string"},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["checkpoint", "benchmarks", "verdict"],
        "additionalProperties": True,
        "properties": {"verdict": {"enum": ["PASS", "RED", "UNMEASURED"]}, "benchmarks": {"type": "array"}},
    }
    rules = (
        RuleSpec("EV-IN-001", "checkpoint (or the run manifest naming it) missing or not a directory", Severity.BLOCK, "input"),
        RuleSpec("EV-IN-002", "no resolvable local base model to serve as the baseline", Severity.BLOCK, "input"),
        RuleSpec("EV-IN-003", "eval cache (--eval-cache) missing", Severity.BLOCK, "input"),
        RuleSpec("EV-IN-004", "installed lm_eval is absent or not the pinned version", Severity.BLOCK, "input"),
        RuleSpec("EV-IN-005", "eval policy invalid, a benchmark has no policy entry, or needs an LLM judge", Severity.BLOCK, "input"),
        RuleSpec("EV-HO-001", "a benchmark breached its regression tolerance", Severity.BLOCK, "handoff"),
        RuleSpec("EV-HO-002", "a configured benchmark was skipped (harness reported no metric)", Severity.BLOCK, "handoff"),
        RuleSpec("EV-HO-003", "a benchmark is unmeasured, or the run was limited (smoke) and cannot PASS", Severity.WARN, "handoff"),
        RuleSpec("EV-HO-004", "a PASS report is missing on disk, altered, limited, or has a null score", Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("lm-eval",),
        emits=("eval_report",),
        apis=("run_manifest.json",),
        notes="consumes FS run manifests (config.model, config.output_dir/final); runs lm-eval 0.4.12 offline",
    )

    def __init__(self, runner: HarnessRunner | None = None,
                 version_probe: Callable[[], str | None] = harness_version) -> None:
        self.runner = runner or SubprocessLmEvalRunner()
        self._version_probe = version_probe

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        installed = self._version_probe()
        if installed != PINNED_LM_EVAL:
            msg = ("missing input: lm_eval is not installed" if installed is None
                   else f"precondition failed: lm_eval {installed} installed, pinned {PINNED_LM_EVAL}")
            findings.append(self.finding("EV-IN-004", msg, {"installed": installed, "pinned": PINNED_LM_EVAL},
                                         f"use the env with lm_eval=={PINNED_LM_EVAL}"))
        cache = Path(request["eval_cache"])
        if not cache.is_dir():
            findings.append(self.finding("EV-IN-003", f"missing input: eval cache directory {cache}", {"eval_cache": str(cache)},
                                         "stage datasets offline into a home-scoped HF_HOME and pass it as --eval-cache"))
        try:
            policy = load_policy(request["policy"])
            for name in request["benchmarks"]:
                if policy.task(name).judge:
                    raise PolicyError(f"precondition failed: benchmark {name!r} needs an LLM judge (judge-free only)")
        except PolicyError as exc:
            findings.append(self.finding("EV-IN-005", str(exc), {"policy": request["policy"]},
                                         "add a judge-free policy entry (ask first) or drop the benchmark"))
        try:
            self._resolve(request)
        except ResolveError as exc:
            findings.append(self.finding(exc.rule_id, str(exc), {}, "pass --checkpoint/--base or --run-manifest"))
        return findings

    @staticmethod
    def _resolve(request: dict[str, Any]) -> Resolved:
        return resolve(request.get("checkpoint"), request.get("base"), request.get("run_manifest"))

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        policy = load_policy(request["policy"])
        res = self._resolve(request)
        cache = Path(request["eval_cache"])
        out = Path(request["out"])
        work = out.parent / f"harness-{uuid.uuid4().hex[:8]}"
        baselines = BaselineCache(request.get("baseline_cache") or ctx.artifacts_dir / "eval" / "baselines")
        installed = self._version_probe()
        limit = request.get("limit")
        # The chat template is applied only when both sides ship one (SPEC-eval assumption 11).
        chat = ships_chat_template(res.base) and (res.adapter or ships_chat_template(res.checkpoint))
        tokenizer = str(res.checkpoint) if res.adapter and (res.checkpoint / "tokenizer_config.json").is_file() else None
        base_ident = model_identity(res.base)

        rows: list[dict[str, Any]] = []
        findings: list[Finding] = []
        policy_errors: list[str] = []
        for name in request["benchmarks"]:
            task = policy.task(name)
            row = self._benchmark(task, request, res, cache, work, baselines, base_ident, installed, chat, tokenizer)
            if row.pop("_policy_error", None):
                policy_errors.append(row["error"])
            rows.append(row)
            findings.extend(self._row_findings(row))
        if policy_errors:
            return self._refused("; ".join(policy_errors),
                                 [self.finding("EV-IN-005", e, {"policy": request["policy"]}) for e in policy_errors])

        final = verdict((r["status"] for r in rows), limited=limit is not None)
        if limit is not None:
            findings.append(self.finding("EV-HO-003", f"limited smoke run (--limit {limit}) can never PASS", {"limit": limit}))
        report = {
            "checkpoint": str(res.checkpoint),
            "base": str(res.base),
            "base_source": res.base_source,
            "adapter": res.adapter,
            "verdict": final,
            "limited": limit is not None,
            "benchmarks": rows,
            "harness": {"name": getattr(self.runner, "name", type(self.runner).__name__),
                        "version": installed, "pinned": PINNED_LM_EVAL, "backend": "hf"},
            "policy": {"path": str(request["policy"]), "version": policy.version, "fingerprint": policy.fingerprint},
            "seed": request.get("seed", 1234),
            "dtype": request.get("dtype", "bfloat16"),
            "gen_kwargs": request.get("gen_kwargs"),
            "chat_template": chat,
            "eval_cache": str(cache),
            "base_identity": base_ident,
            "harness_dir": str(work),
        }
        sha = write_report(out, report)
        ref = ArtifactRef(type="eval_report", id=res.checkpoint.name or "checkpoint", path=str(out), sha256=sha)
        status = {"PASS": Status.PASS, "RED": Status.RED, "UNMEASURED": Status.UNMEASURED}[final]
        next_actions = () if status is Status.PASS else (f"inspect {out} and the harness logs under {work}",)
        return SkillResult(
            status=status, payload=report, artifacts=(ref,), findings=tuple(findings), next_actions=next_actions,
            provenance=make_provenance(self.name, self.version, {"eval_report": sha, "policy": policy.fingerprint}),
        )

    def _benchmark(self, task: TaskPolicy, request: dict[str, Any], res: Resolved, cache: Path, work: Path,
                   baselines: BaselineCache, base_ident: dict[str, Any], installed: str | None, chat: bool,
                   tokenizer: str | None) -> dict[str, Any]:
        fewshot = request.get("num_fewshot", task.num_fewshot)
        common = dict(
            task=task.name, eval_cache=str(cache), num_fewshot=fewshot, seed=request.get("seed", 1234),
            dtype=request.get("dtype", "bfloat16"), chat_template=chat, fewshot_as_multiturn=task.fewshot_as_multiturn,
            system_instruction=task.system_instruction, gen_kwargs=request.get("gen_kwargs"), limit=request.get("limit"),
            batch_size=request.get("batch_size"), device=request.get("device"), parallelize=bool(request.get("parallelize", False)),
            trust_remote_code=task.trust_remote_code, include_path=request.get("include_path"),
        )
        fp_fields = {k: v for k, v in common.items() if k not in ("eval_cache", "batch_size", "device", "parallelize")}
        fp = fingerprint({**fp_fields, "metric": task.metric, "harness_version": installed, "backend": "hf",
                          "base": base_ident})
        row: dict[str, Any] = {"name": task.name, "metric": task.metric, "score": None, "baseline": None,
                               "num_fewshot": fewshot, "chat_template": chat, "fingerprint": fp, "dataset": task.dataset}
        if task.dataset and not dataset_staged(cache, task.dataset):
            return {**row, "status": "unmeasured",
                    "error": f"missing input: dataset {task.dataset} not staged in eval cache {cache}"}

        ck = self.runner.run(HarnessRequest(role="checkpoint", model_dir=str(res.base if res.adapter else res.checkpoint),
                                            adapter_dir=str(res.checkpoint) if res.adapter else None,
                                            tokenizer_dir=tokenizer, output_path=str(work / "checkpoint" / task.name),
                                            **common))
        row["argv"] = list(ck.argv)
        if ck.results is None:
            return {**row, "status": "unmeasured", "error": f"checkpoint harness failed: {ck.error}"}
        measured = extract_metric(ck.results, task.name, task.metric)
        if measured is None:
            return {**row, "status": "skipped",
                    "error": f"harness completed but reported no {task.metric!r} for {task.name!r} (checkpoint)"}
        row.update(score=measured["score"], stderr=measured["stderr"], n=measured["n"],
                   task_hash=(ck.results.get("task_hashes") or {}).get(task.name))

        cached = baselines.get(base_ident, fp)
        if cached is not None:
            base_m = {"score": cached.get("score"), "stderr": cached.get("stderr"), "n": cached.get("n")}
            row["baseline_provenance"] = "cached"
        else:
            bl = self.runner.run(HarnessRequest(role="baseline", model_dir=str(res.base),
                                                output_path=str(work / "baseline" / task.name), **common))
            row["baseline_argv"] = list(bl.argv)
            if bl.results is None:
                return {**row, "status": "unmeasured", "error": f"baseline harness failed: {bl.error}"}
            base_m = extract_metric(bl.results, task.name, task.metric)
            if base_m is None:
                return {**row, "status": "skipped",
                        "error": f"harness completed but reported no {task.metric!r} for {task.name!r} (baseline)"}
            row["baseline_provenance"] = "measured"
            if request.get("limit") is None:  # a smoke slice never seeds the cache
                baselines.put(base_ident, fp, {**base_m, "task": task.name, "metric": task.metric,
                                               "fields": {**fp_fields, "harness_version": installed}})
        if not isinstance(base_m.get("score"), (int, float)):
            return {**row, "status": "unmeasured", "error": "cached baseline has no numeric score"}
        row.update(baseline=float(base_m["score"]), baseline_stderr=base_m.get("stderr"), baseline_n=base_m.get("n"))
        try:
            cmp = compare(task, row["score"], row["baseline"], row["stderr"], row["baseline_stderr"])
        except PolicyError as exc:
            return {**row, "status": "unmeasured", "error": str(exc), "_policy_error": True}
        row.update(drop=cmp.drop, threshold=cmp.threshold, band=cmp.band, status="red" if cmp.breach else "pass")
        return row

    def _row_findings(self, row: dict[str, Any]) -> list[Finding]:
        ev = {"benchmark": row["name"], "metric": row["metric"]}
        if row["status"] == "red":
            return [self.finding("EV-HO-001",
                                 f"{row['name']} regressed: {row['score']:.4f} vs baseline {row['baseline']:.4f} "
                                 f"(drop {row['drop']:.4f} > {row['threshold']:.4f})",
                                 {**ev, "drop": row["drop"], "threshold": row["threshold"]},
                                 "inspect the checkpoint's training data/LR; compare per-sample logs")]
        if row["status"] == "skipped":
            return [self.finding("EV-HO-002", row["error"], ev, "check the task name/metric against lm-eval ls")]
        if row["status"] == "unmeasured":
            return [self.finding("EV-HO-003", row["error"], ev, "stage the dataset / read the harness log, then rerun")]
        return []

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        problems = []
        report = result.payload
        if report.get("limited"):
            problems.append("report is from a limited run")
        if any(b.get("score") is None or b.get("baseline") is None for b in report.get("benchmarks", [])):
            problems.append("a benchmark has a null score or baseline")
        for ref in result.artifacts:
            try:
                if sha256_file(ref.path) != ref.sha256:
                    problems.append(f"{ref.path} changed after it was written")
            except OSError:
                problems.append(f"{ref.path} is missing on disk")
        if not result.artifacts:
            problems.append("no eval_report artifact was written")
        return [self.finding("EV-HO-004", p, {}) for p in problems]

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"{type(failure).__name__}: {failure}" if isinstance(failure, BaseException)
                   else f"evaluation returned {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                "dataset not staged in the eval cache (runs are offline)",
                "harness failed to load the model (dtype/memory/trust_remote_code)",
                "policy metric name does not match what lm-eval reports for the task",
            ),
            checks=("read harness.log under the report's harness_dir", "lm-eval ls tasks | grep <task>",
                    "ls <eval_cache>/hub/datasets--*"),
            recovery=("stage the dataset and rerun", "fix the policy metric (ask first)", "rerun with --limit to smoke-test"),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        req = {"checkpoint": "ck", "base": "base", "benchmarks": ["mmlu"], "policy": "eval_policy.yaml",
               "eval_cache": "cache", "out": "eval/eval_report.json"}
        return {
            "EV-IN-001": {"request": {**req, "checkpoint": "missing"}, "expected": "REFUSED naming EV-IN-001"},
            "EV-IN-002": {"request": {k: v for k, v in req.items() if k != "base"},
                          "expected": "REFUSED naming EV-IN-002 (full checkpoint, no base)"},
            "EV-IN-003": {"request": {**req, "eval_cache": "missing"}, "expected": "REFUSED naming EV-IN-003"},
            "EV-IN-004": {"request": req, "version": "0.4.11", "expected": "REFUSED naming EV-IN-004"},
            "EV-IN-005": {"request": {**req, "benchmarks": ["not_in_policy"]}, "expected": "REFUSED naming EV-IN-005"},
            "EV-HO-001": {"request": req, "harness": {"checkpoint": 0.40, "baseline": 0.50, "stderr": 0.004},
                          "expected": "RED with EV-HO-001"},
            "EV-HO-002": {"request": req, "harness": {"checkpoint": None, "baseline": 0.50},
                          "expected": "RED with EV-HO-002 (metric absent from results)"},
            "EV-HO-003": {"request": {**req, "limit": 8}, "harness": {"checkpoint": 0.50, "baseline": 0.50},
                          "expected": "UNMEASURED with EV-HO-003"},
            "EV-HO-004": {"request": req, "tamper": "delete report before handoff", "expected": "RED with EV-HO-004"},
        }
