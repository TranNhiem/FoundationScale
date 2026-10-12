"""EvalSkill: checkpoint -> lm-evaluation-harness -> one-sided policy verdict -> eval_report.json.

Exit contract: 96 REFUSED writes no report (every refusal names the missing
input); 5 RED when a measured benchmark breached its band or a configured
benchmark was skipped; 95 UNMEASURED when a benchmark could not run (dataset
not staged, harness error) or the run was ``limit``-ed; 0 PASS only when every
benchmark was measured against a fingerprint-matched baseline and none breached.
"""
from __future__ import annotations

import json
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
from foundationskills.skills.evaluation.resolve import Resolved, ResolveError, manifest_video_frames, resolve, ships_chat_template
from foundationskills.skills.evaluation.runner import (
    PINNED_LM_EVAL,
    HarnessRequest,
    HarnessRunner,
    HeldoutRequest,
    SubprocessLmEvalRunner,
    VlmHeldoutRunner,
    build_heldout_argv,
    default_heldout_script,
    extract_heldout,
    extract_metric,
    harness_version,
)


def dataset_staged(eval_cache: Path, dataset: str) -> bool:
    """True when an HF hub snapshot or a prepared ``datasets`` cache for ``dataset`` exists under the eval cache."""
    hub = eval_cache / "hub" / f"datasets--{dataset.replace('/', '--')}"
    prepared = eval_cache / "datasets" / dataset.replace("/", "___")
    return hub.is_dir() or prepared.is_dir()


HELDOUT_ROW_NAME = "heldout_nll"
_HELDOUT_MIRRORS = ("data", "max_length", "video_frames", "image_column", "video_column", "limit",
                    "heldout_script", "abs_epsilon", "python")

_HELDOUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "data": {"type": "string", "minLength": 1},
        "max_length": {"type": "integer", "minimum": 1},
        "video_frames": {"type": ["integer", "null"], "minimum": 1},
        "image_column": {"type": "string", "minLength": 1},
        "video_column": {"type": "string", "minLength": 1},
        "limit": {"type": ["integer", "null"], "minimum": 1},
        "heldout_script": {"type": ["string", "null"], "minLength": 1},
        "abs_epsilon": {"type": "number", "minimum": 0},
        "python": {"type": "string", "minLength": 1},
    },
}


def heldout_config(request: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise the optional ``heldout`` block into one config dict (None for a pure lm-eval request).

    The block wins where both it and the draft-form top-level mirror keys are present.
    """
    raw = request.get("heldout")
    if not isinstance(raw, dict) and not any(
            request.get(k) is not None for k in ("data", "max_length", "video_frames", "heldout_script")):
        return None
    cfg: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    for key in _HELDOUT_MIRRORS:
        if cfg.get(key) is None and request.get(key) is not None:
            cfg[key] = request[key]
    cfg.setdefault("image_column", "image")
    cfg.setdefault("video_column", "video")
    return cfg


def _heldout_harness(hold: dict[str, Any]) -> dict[str, Any]:
    """Held-out mode never runs lm-eval: its provenance is the scoring script itself, bound by sha256."""
    script = _heldout_script(hold)
    digest = sha256_file(script) if script is not None and script.is_file() else None
    return {"name": "fs-heldout", "backend": "hf", "pinned": None,
            "version": f"sha256:{digest[:16]}" if digest else None, "script": str(script) if script else None}


def _heldout_script(hold: dict[str, Any]) -> Path | None:
    """The eval_heldout_loss.py to run: the request's ``heldout_script``, else the copy shipped with FoundationScale."""
    named = hold.get("heldout_script")
    return Path(named) if named else default_heldout_script()


def _video_rows_present(data: Path, video_column: str) -> bool:
    """True when a held-out JSONL row carries a non-empty video column (those rows need --video-frames)."""
    with Path(data).open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict) and bool(row.get(video_column)):
                return True
    return False


def _heldout_of(outcome: Any) -> tuple[int, dict[str, Any] | None, str]:
    """Normalise a held-out outcome (``HeldoutOutcome``, or a fake returning a HarnessOutcome-shaped one)."""
    code = getattr(outcome, "rc", getattr(outcome, "returncode", 1))
    doc = getattr(outcome, "result", None)
    if doc is None:
        doc = getattr(outcome, "results", None)
    tail = getattr(outcome, "stderr_tail", None) or getattr(outcome, "error", None) or ""
    return int(code), doc if isinstance(doc, dict) else None, str(tail)


def _score_heldout(target: Any, req: HeldoutRequest) -> tuple[int, dict[str, Any] | None, str, list[str]]:
    outcome = target.run(req)
    code, doc, tail = _heldout_of(outcome)
    argv = list(getattr(outcome, "argv", ()) or ())
    if not argv:
        try:
            argv = build_heldout_argv(req)
        except ValueError:
            argv = []
    return code, doc, tail, argv


class EvalSkill(BaseSkill):
    name = "evaluation"
    version = "0.1.0"
    description = (
        "Evaluate a checkpoint (full or PEFT adapter) offline against its base model: pinned lm-evaluation-harness "
        "benchmarks under a one-sided regression band, plus VLM held-out loss (image/video/text rows, video included)"
        " via FoundationScale's eval_heldout_loss.py; one verdict into eval_report.json."
    )
    scope = Scope(
        model_types=("llm", "vlm"),
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
        "required": ["eval_cache", "out"],
        "additionalProperties": False,
        "properties": {
            "checkpoint": {"type": "string", "minLength": 1},
            "base": {"type": "string", "minLength": 1},
            "run_manifest": {"type": "string", "minLength": 1},
            "benchmarks": {"type": "array", "items": {"type": "string", "minLength": 1}},
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
            "heldout": _HELDOUT_SCHEMA,
            # draft-form mirrors of the heldout block (a single rAFU request may declare them top-level)
            "data": {"type": "string", "minLength": 1},
            "max_length": {"type": "integer", "minimum": 1},
            "video_frames": {"type": ["integer", "null"], "minimum": 1},
            "image_column": {"type": "string", "minLength": 1},
            "video_column": {"type": "string", "minLength": 1},
            "heldout_script": {"type": ["string", "null"], "minLength": 1},
            "abs_epsilon": {"type": "number", "minimum": 0},
            "python": {"type": "string", "minLength": 1},
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
        RuleSpec("EV-IN-004", "installed lm_eval is absent or not the pinned version (lm-eval mode)", Severity.BLOCK, "input"),
        RuleSpec("EV-IN-005", "eval policy invalid, a benchmark has no policy entry, needs an LLM judge, or the request holds neither a benchmark nor a held-out hit", Severity.BLOCK, "input"),
        RuleSpec("EV-IN-006", "held-out data missing/unreadable, or the eval_heldout_loss.py script is missing (never a silent fallback)", Severity.BLOCK, "input"),
        RuleSpec("EV-IN-007", "held-out rows carry video but video_frames is unset", Severity.BLOCK, "input"),
        RuleSpec("EV-IN-008", "video_frames disagrees with the run manifest's FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES", Severity.BLOCK, "input"),
        RuleSpec("EV-RU-001", "vacuous run (held-out rows_used == 0 for either role): the rule template has no run phase, so it is declared a handoff BLOCK and yielded as REFUSED (no report) through the existing refusal path", Severity.BLOCK, "handoff"),
        RuleSpec("EV-HO-001", "a benchmark breached its regression tolerance", Severity.BLOCK, "handoff"),
        RuleSpec("EV-HO-002", "a configured benchmark was skipped (harness reported no metric)", Severity.BLOCK, "handoff"),
        RuleSpec("EV-HO-003", "a benchmark is unmeasured, or the run was limited (smoke) and cannot PASS", Severity.WARN, "handoff"),
        RuleSpec("EV-HO-004", "a PASS report is missing on disk, altered, limited, or has a null score", Severity.BLOCK, "handoff"),
        RuleSpec("EV-HO-005", "base and checkpoint held-out rows_used/rows_dropped disagree: the delta is unmeasured. BLOCK, but deliberately mapped to UNMEASURED 95 (never RED): an incomparable delta is no failure", Severity.BLOCK, "handoff"),
        RuleSpec("EV-HO-006", "a held-out row lacks rows_used/rows_dropped/score/baseline", Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("lm-eval", "foundationscale/eval_heldout_loss.py"),
        emits=("eval_report",),
        apis=("run_manifest.json",),
        notes="consumes FS run manifests (config.model, config.output_dir/final, environment.values.FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES); "
              "runs lm-eval 0.4.12 offline, or eval_heldout_loss.py for VLM held-out loss (rows incl. video)",
    )

    def __init__(self, runner: HarnessRunner | None = None,
                 version_probe: Callable[[], str | None] = harness_version,
                 heldout_runner: Any = None) -> None:
        self.runner = runner or SubprocessLmEvalRunner()
        self.heldout_runner = heldout_runner
        self._version_probe = version_probe

    def _heldout_target(self) -> Any:
        """Scores held-out rows.

        An injected ``runner`` is used as-is (test fakes may serve both request
        shapes); the default ``SubprocessLmEvalRunner`` only understands
        ``HarnessRequest``, so real held-out runs get ``VlmHeldoutRunner``.
        """
        if isinstance(self.runner, SubprocessLmEvalRunner):
            return self.heldout_runner or VlmHeldoutRunner()
        return self.heldout_runner or self.runner

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        benchmarks = list(request.get("benchmarks") or ())
        hold = heldout_config(request)
        if not benchmarks and hold is None:
            findings.append(self.finding(
                "EV-IN-005", "missing input: the request holds neither a benchmark nor a held-out hit", {},
                "pass --benchmarks and --policy (lm-eval mode) or a heldout block (held-out loss mode)"))
        if benchmarks:  # lm-eval mode: the pinned harness, offline, judge-free tasks
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
        if benchmarks:
            try:
                policy = load_policy(request.get("policy") or "")
                for name in benchmarks:
                    if policy.task(name).judge:
                        raise PolicyError(f"precondition failed: benchmark {name!r} needs an LLM judge (judge-free only)")
            except PolicyError as exc:
                findings.append(self.finding("EV-IN-005", str(exc), {"policy": request.get("policy")},
                                             "add a judge-free policy entry (ask first) or drop the benchmark"))
        try:
            self._resolve(request)
        except ResolveError as exc:
            findings.append(self.finding(exc.rule_id, str(exc), {}, "pass --checkpoint/--base or --run-manifest"))
        if hold is not None:
            findings.extend(self._heldout_inputs(request, hold))
        return findings

    @staticmethod
    def _resolve(request: dict[str, Any]) -> Resolved:
        return resolve(request.get("checkpoint"), request.get("base"), request.get("run_manifest"))

    # -- held-out loss -----------------------------------------------------------------------
    def _heldout_epsilon(self, request: dict[str, Any], hold: dict[str, Any]) -> float | None:
        """Explicit abs_epsilon for held-out NLL (no stderr channel): the block's, else the policy's heldout_nll."""
        eps = hold.get("abs_epsilon")
        if isinstance(eps, (int, float)) and not isinstance(eps, bool) and eps >= 0:
            return float(eps)
        try:
            policy = load_policy(request.get("policy") or "")
        except PolicyError:
            return None
        entry = policy.tasks.get(HELDOUT_ROW_NAME)
        return float(entry.abs_epsilon) if entry is not None and entry.explicit_epsilon else None

    def _heldout_inputs(self, request: dict[str, Any], hold: dict[str, Any]) -> list[Finding]:
        """EV-IN-005/006/007/008: the held-out block must be runnable before any GPU is spent."""
        findings: list[Finding] = []
        video_column = hold.get("video_column") or "video"
        details = {"data": hold.get("data"), "max_length": hold.get("max_length"), "video_frames": hold.get("video_frames")}
        if self._heldout_epsilon(request, hold) is None:
            findings.append(self.finding(
                "EV-IN-005",
                "missing input: heldout.abs_epsilon (held-out mean NLL carries no stderr, so the band must name abs_epsilon)",
                details, "pass heldout.abs_epsilon (or an explicit abs_epsilon on the policy's heldout_nll entry); ask first"))
        max_length = hold.get("max_length")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            findings.append(self.finding("EV-IN-005", f"missing input: heldout.max_length (got {max_length!r})",
                                         details, "pass max_length (e.g. 16384)"))
        data = hold.get("data")
        if not isinstance(data, str) or not data:
            findings.append(self.finding("EV-IN-006", "missing input: heldout.data (the canonical held-out JSONL)", details,
                                         "pass the held-out split's JSONL as heldout.data"))
            return findings
        path = Path(data)
        if not path.is_file():
            findings.append(self.finding("EV-IN-006", f"missing input: held-out data file {path}", details,
                                         "stage the held-out split and pass its JSONL as heldout.data"))
            return findings
        try:
            has_video = _video_rows_present(path, video_column)
        except (OSError, ValueError) as exc:
            findings.append(self.finding("EV-IN-006", f"precondition failed: held-out data {path} is not readable JSONL ({exc})",
                                         details, "write one JSON object per line (jsonl)"))
            return findings
        frames = hold.get("video_frames")
        if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or frames < 1):
            findings.append(self.finding("EV-IN-007", f"precondition failed: video_frames {frames!r} is not a positive integer",
                                         details, "pass video_frames (e.g. 16)"))
            frames = None
        if has_video and frames is None:
            findings.append(self.finding(
                "EV-IN-007",
                "missing input: held-out rows carry a non-empty " + repr(video_column) + " field but video_frames is unset",
                {**details, "video_column": video_column},
                "pass video_frames equal to the training run's FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES (e.g. 16)"))
        declared = manifest_video_frames(request.get("run_manifest"))
        if declared is not None and frames is not None and frames != declared:
            findings.append(self.finding(
                "EV-IN-008",
                "precondition failed: video_frames {} != the run manifest's FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES {}".format(
                    frames, declared),
                {**details, "FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES": declared},
                "rerun with video_frames {}".format(declared)))
        script = _heldout_script(hold)
        if script is None or not Path(script).is_file():
            findings.append(self.finding(
                "EV-IN-006",
                "missing input: held-out script {}".format(script if script else "(not shipped next to FoundationScale)"),
                {"heldout_script": hold.get("heldout_script")},
                "pass heldout_script pointing at examples/vlm-2xH200/scripts/eval_heldout_loss.py"))
        return findings

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        benchmarks = list(request.get("benchmarks") or ())
        hold = heldout_config(request)
        policy = load_policy(request["policy"]) if benchmarks else None
        res = self._resolve(request)
        cache = Path(request["eval_cache"])
        out = Path(request["out"])
        work = out.parent / f"harness-{uuid.uuid4().hex[:8]}"
        baselines = BaselineCache(request.get("baseline_cache") or ctx.artifacts_dir / "eval" / "baselines")
        installed = self._version_probe()
        limit = request.get("limit")
        hold_limit = hold.get("limit") if hold is not None else None
        limited = limit is not None or hold_limit is not None
        # The chat template is applied only when both sides ship one (SPEC-eval assumption 11).
        chat = ships_chat_template(res.base) and (res.adapter or ships_chat_template(res.checkpoint))
        tokenizer = str(res.checkpoint) if res.adapter and (res.checkpoint / "tokenizer_config.json").is_file() else None
        base_ident = model_identity(res.base)

        rows: list[dict[str, Any]] = []
        findings: list[Finding] = []
        policy_errors: list[str] = []
        for name in benchmarks:
            task = policy.task(name)
            row = self._benchmark(task, request, res, cache, work, baselines, base_ident, installed, chat, tokenizer)
            if row.pop("_policy_error", None):
                policy_errors.append(row["error"])
            rows.append(row)
            findings.extend(self._row_findings(row))
        if policy_errors:
            return self._refused("; ".join(policy_errors),
                                 [self.finding("EV-IN-005", e, {"policy": request["policy"]}) for e in policy_errors])

        if hold is not None:
            held_rows, held_findings, refusal = self._heldout(request, hold, res, work, baselines, base_ident, limited)
            if refusal is not None:
                return self._refused(*refusal)
            rows.extend(held_rows)
            findings.extend(held_findings)

        final = verdict((r["status"] for r in rows), limited=limited)
        if limited:
            shown_limit = limit if limit is not None else hold_limit
            findings.append(self.finding("EV-HO-003", f"limited smoke run (--limit {shown_limit}) can never PASS",
                                         {"limit": shown_limit}))
        report = {
            "checkpoint": str(res.checkpoint),
            "base": str(res.base),
            "base_source": res.base_source,
            "adapter": res.adapter,
            "verdict": final,
            "limited": limited,
            "benchmarks": rows,
            "harness": (_heldout_harness(hold) if hold is not None and not request.get("benchmarks") else
                        {"name": getattr(self._heldout_target() if hold is not None else self.runner,
                                         "name", type(self.runner).__name__),
                         "version": installed, "pinned": PINNED_LM_EVAL, "backend": "hf"}),
            "policy": ({"path": str(request["policy"]), "version": policy.version, "fingerprint": policy.fingerprint}
                       if policy is not None
                       else {"path": request.get("policy"), "version": None, "fingerprint": None}),
            "seed": request.get("seed", 1234),
            "dtype": request.get("dtype", "bfloat16"),
            "gen_kwargs": request.get("gen_kwargs"),
            "chat_template": chat,
            "eval_cache": str(cache),
            "base_identity": base_ident,
            "harness_dir": str(work),
            **({"heldout": {"script": str(_heldout_script(hold) or ""), "data": str(hold.get("data")),
                            "max_length": hold.get("max_length"), "video_frames": hold.get("video_frames"),
                            "image_column": hold.get("image_column"), "video_column": hold.get("video_column"),
                            "limit": hold_limit, "abs_epsilon": self._heldout_epsilon(request, hold)}}
               if hold is not None else {}),
        }
        sha = write_report(out, report)
        ref = ArtifactRef(type="eval_report", id=res.checkpoint.name or "checkpoint", path=str(out), sha256=sha)
        status = {"PASS": Status.PASS, "RED": Status.RED, "UNMEASURED": Status.UNMEASURED}[final]
        next_actions = () if status is Status.PASS else (f"inspect {out} and the harness logs under {work}",)
        return SkillResult(
            status=status, payload=report, artifacts=(ref,), findings=tuple(findings), next_actions=next_actions,
            provenance=make_provenance(self.name, self.version,
                                      {"eval_report": sha, "policy": policy.fingerprint} if policy is not None
                                      else {"eval_report": sha}),
        )

    def _heldout(self, request: dict[str, Any], hold: dict[str, Any], res: Resolved, work: Path,
                 baselines: BaselineCache, base_ident: dict[str, Any], limited: bool,
                 ) -> tuple[list[dict[str, Any]], list[Finding], tuple[str, list[Finding]] | None]:
        """Held-out loss for base and checkpoint on the same rows -> one report row.

        The third element is a ``(message, findings)`` REFUSED refusal (exit 96, no
        report): the script itself refused, or the run was vacuous (EV-RU-001).
        """
        eps = self._heldout_epsilon(request, hold)
        script = _heldout_script(hold)
        if eps is None or script is None or hold.get("max_length") is None:
            detail = {"abs_epsilon": eps, "script": str(script) if script else None, "max_length": hold.get("max_length")}
            return [], [], self._refuse_heldout(
                "missing input: heldout.abs_epsilon / max_length / the eval_heldout_loss.py script", "EV-IN-005", detail,
                "complete the heldout block (abs_epsilon, max_length) and pass heldout_script")
        common = dict(data=str(hold.get("data")), max_length=int(hold.get("max_length")), video_frames=hold.get("video_frames"),
                      image_column=hold.get("image_column") or "image", video_column=hold.get("video_column") or "video",
                      limit=hold.get("limit"), script=str(script), python=hold.get("python"))
        target = self._heldout_target()
        out_dir = work / "heldout"
        ck_req = HeldoutRequest(model=str(res.base) if res.adapter else str(res.checkpoint),
                                adapter=str(res.checkpoint) if res.adapter else None,
                                out=str(out_dir / "checkpoint.json"), **common)
        base_req = HeldoutRequest(model=str(res.base), out=str(out_dir / "baseline.json"), **common)
        fp = fingerprint({"kind": "heldout", "data_sha256": sha256_file(Path(hold["data"])),
                          "max_length": common["max_length"], "video_frames": common["video_frames"],
                          "image_column": common["image_column"], "video_column": common["video_column"],
                          "limit": common["limit"], "base": base_ident})
        row: dict[str, Any] = {
            "name": HELDOUT_ROW_NAME, "metric": "mean_nll_nats", "score": None, "baseline": None,
            "higher_is_better": False, "rows_used": None, "rows_dropped": None, "baseline_rows_used": None,
            "baseline_rows_dropped": None, "perplexity": None, "baseline_perplexity": None,
            "tokens_scored": None, "dataset": str(hold.get("data")), "fingerprint": fp,
        }

        code, doc, tail, argv = _score_heldout(target, ck_req)
        row["argv"] = argv
        if code == 96:
            return [], [], self._refuse_heldout(f"held-out script refused (exit 96): {tail}", "EV-IN-006",
                                                {"role": "checkpoint", "argv": argv},
                                                "apply the fix the script names (frames, rows, flags), then rerun")
        if code != 0:
            row.update(status="unmeasured", error=f"checkpoint held-out run failed (exit {code}): {tail}")
            return [row], [self.finding("EV-HO-003", row["error"], {"name": HELDOUT_ROW_NAME, "argv": argv},
                                        "read the script's stderr, fix the rows/flags, rerun")], None
        measured = extract_heldout(doc)
        if measured is None:
            row.update(status="red",
                       error="the checkpoint result lacks its measurement block (rows_used/rows_dropped/mean_nll_nats)")
            return [row], [self.finding("EV-HO-006", row["error"], {"name": HELDOUT_ROW_NAME, "argv": argv},
                                        "the --out JSON must carry rows_used/rows_dropped/mean_nll_nats/perplexity")], None
        row.update(score=measured["nll"], rows_used=measured["rows_used"], rows_dropped=measured["rows_dropped"],
                   perplexity=measured["perplexity"], tokens_scored=measured["tokens_scored"])

        cached = baselines.get(base_ident, fp)
        if cached is not None:
            base_m = cached
            row["baseline_provenance"] = "cached"
        else:
            code, doc, tail, base_argv = _score_heldout(target, base_req)
            row["baseline_argv"] = base_argv
            if code == 96:
                return [], [], self._refuse_heldout(f"held-out script refused (exit 96): {tail}", "EV-IN-006",
                                                    {"role": "baseline", "argv": base_argv},
                                                    "apply the fix the script names (frames, rows, flags), then rerun")
            if code != 0:
                row.update(status="unmeasured", error=f"baseline held-out run failed (exit {code}): {tail}")
                return [row], [self.finding("EV-HO-003", row["error"], {"name": HELDOUT_ROW_NAME, "argv": base_argv},
                                            "read the script's stderr, fix the rows/flags, rerun")], None
            base_m = extract_heldout(doc)
            if base_m is None:
                row.update(status="red",
                           error="the baseline result lacks its measurement block (rows_used/rows_dropped/mean_nll_nats)")
                return [row], [self.finding("EV-HO-006", row["error"], {"name": HELDOUT_ROW_NAME, "argv": base_argv},
                                            "the --out JSON must carry rows_used/rows_dropped/mean_nll_nats/perplexity")], None
            row["baseline_provenance"] = "measured"
            if not limited:  # a smoke slice never seeds the cache
                baselines.put(base_ident, fp, {"kind": "heldout", "metric": "mean_nll_nats", **base_m})
        row.update(baseline=base_m["nll"], baseline_rows_used=base_m["rows_used"],
                   baseline_rows_dropped=base_m["rows_dropped"], baseline_perplexity=base_m["perplexity"])
        return self._heldout_verdict(row, float(eps))

    def _heldout_verdict(self, row: dict[str, Any], abs_epsilon: float
                         ) -> tuple[list[dict[str, Any]], list[Finding], tuple[str, list[Finding]] | None]:
        if row["rows_used"] == 0 or row["baseline_rows_used"] == 0:  # EV-RU-001: a role that scored nothing measured nothing
            role = "checkpoint" if row["rows_used"] == 0 else "base"
            details = {"rows_used": row["rows_used"], "baseline_rows_used": row["baseline_rows_used"]}
            return [], [], self._refuse_heldout(
                f"vacuous run: held-out rows_used == 0 for the {role} role", "EV-RU-001", details,
                "widen max_length (overlong rows are dropped) or fix the held-out rows, then rerun")
        if (row["rows_used"], row["rows_dropped"]) != (row["baseline_rows_used"], row["baseline_rows_dropped"]):
            # EV-HO-005: base and checkpoint without the same rows leave the delta unmeasured, never PASS.
            error = (f"held-out row counts disagree: checkpoint {row['rows_used']} used / {row['rows_dropped']} dropped vs "
                     f"base {row['baseline_rows_used']} / {row['baseline_rows_dropped']}; the delta is unmeasured")
            details = {"rows_used": row["rows_used"], "rows_dropped": row["rows_dropped"],
                       "baseline_rows_used": row["baseline_rows_used"], "baseline_rows_dropped": row["baseline_rows_dropped"]}
            return [{**row, "status": "unmeasured", "error": error}], [self.finding(
                "EV-HO-005", error, details,
                "score both roles over one identical held-out slice (same max_length / limit / row cleaning)")], None
        task = TaskPolicy(name=HELDOUT_ROW_NAME, metric="mean_nll_nats", higher_is_better=False,
                          abs_epsilon=abs_epsilon, explicit_epsilon=True)
        cmp = compare(task, row["score"], row["baseline"])
        row.update(drop=cmp.drop, threshold=cmp.threshold, band=cmp.band, status="red" if cmp.breach else "pass")
        findings: list[Finding] = []
        if cmp.breach:
            findings.append(self.finding("EV-HO-001",
                                         f"{HELDOUT_ROW_NAME} regressed: {row['score']:.4f} vs baseline {row['baseline']:.4f} "
                                         f"(drop {row['drop']:.4f} > {row['threshold']:.4f})",
                                         {"drop": row["drop"], "threshold": row["threshold"], "band": row["band"]},
                                         "inspect the LoRA's training data/LR; compare base and checkpoint rows"))
        return [row], findings, None

    def _refuse_heldout(self, message: str, rule_id: str, details: dict[str, Any], fix: str) -> tuple[str, list[Finding]]:
        return message, [self.finding(rule_id, message, details, fix)]

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
        problems, counted, blocks = [], [], []
        report = result.payload
        if report.get("limited"):
            problems.append("report is from a limited run")
        for benchmark in report.get("benchmarks", []):
            if benchmark.get("name") == HELDOUT_ROW_NAME:
                if any(benchmark.get(k) is None for k in ("rows_used", "rows_dropped", "baseline_rows_used",
                                                          "baseline_rows_dropped", "score", "baseline")):
                    blocks.append("the held-out row lacks its measurement block (rows_used/rows_dropped/score/baseline)")
                elif ((benchmark.get("rows_used"), benchmark.get("rows_dropped"))
                      != (benchmark.get("baseline_rows_used"), benchmark.get("baseline_rows_dropped"))
                      and benchmark.get("status") != "unmeasured"):
                    # a run already reported a mismatch as UNMEASURED 95 (EV-HO-005) and it is forgiven here;
                    # this branch is the row that was altered after it was measured.
                    counted.append(f"held-out row counts disagree after measurement: checkpoint "
                                   f"{benchmark.get('rows_used')}/{benchmark.get('rows_dropped')} vs base "
                                   f"{benchmark.get('baseline_rows_used')}/{benchmark.get('baseline_rows_dropped')}")
                continue
            if benchmark.get("score") is None or benchmark.get("baseline") is None:
                problems.append("a benchmark has a null score or baseline")
        for ref in result.artifacts:
            try:
                if sha256_file(ref.path) != ref.sha256:
                    problems.append(f"{ref.path} changed after it was written")
            except OSError:
                problems.append(f"{ref.path} is missing on disk")
        if not result.artifacts:
            problems.append("no eval_report artifact was written")
        return ([self.finding("EV-HO-004", p, {}) for p in problems]
                + [self.finding("EV-HO-005", p, {}) for p in counted]
                + [self.finding("EV-HO-006", p, {}) for p in blocks])

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
        held_req = {**req, "benchmarks": [],
                    "heldout": {"data": "heldout.jsonl", "max_length": 16384, "video_frames": 16, "abs_epsilon": 0.05}}
        return {
            "EV-IN-001": {"request": {**req, "checkpoint": "missing"}, "expected": "REFUSED naming EV-IN-001"},
            "EV-IN-002": {"request": {k: v for k, v in req.items() if k != "base"},
                          "expected": "REFUSED naming EV-IN-002 (full checkpoint, no base)"},
            "EV-IN-003": {"request": {**req, "eval_cache": "missing"}, "expected": "REFUSED naming EV-IN-003"},
            "EV-IN-004": {"request": req, "version": "0.4.11", "expected": "REFUSED naming EV-IN-004"},
            "EV-IN-005": {"request": {**req, "benchmarks": ["not_in_policy"]}, "expected": "REFUSED naming EV-IN-005"},
            "EV-IN-006": {"request": {**held_req, "heldout": {"data": "missing.jsonl", "max_length": 16384,
                                                              "abs_epsilon": 0.05}},
                          "expected": "REFUSED naming EV-IN-006 (held-out data missing; a missing script too)"},
            "EV-IN-007": {"request": {**held_req, "heldout": {"data": "video_rows.jsonl", "max_length": 16384,
                                                              "abs_epsilon": 0.05}},
                          "expected": "REFUSED naming EV-IN-007 (video rows and no video_frames)"},
            "EV-IN-008": {"request": {**held_req, "run_manifest": "run_manifest.json",
                                      "heldout": {"data": "heldout.jsonl", "max_length": 16384, "video_frames": 24,
                                                  "abs_epsilon": 0.05}},
                          "expected": "REFUSED naming EV-IN-008 (24 != FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES 16)"},
            "EV-RU-001": {"request": held_req,
                          "harness": {"heldout": {"checkpoint": {"rows_used": 0, "rows_dropped": 5, "mean_nll_nats": 1.2},
                                                  "baseline": {"rows_used": 5, "rows_dropped": 0, "mean_nll_nats": 1.4}}},
                          "expected": "REFUSED naming EV-RU-001 (vacuous run writes no report)"},
            "EV-HO-001": {"request": req, "harness": {"checkpoint": 0.40, "baseline": 0.50, "stderr": 0.004},
                          "expected": "RED with EV-HO-001"},
            "EV-HO-002": {"request": req, "harness": {"checkpoint": None, "baseline": 0.50},
                          "expected": "RED with EV-HO-002 (metric absent from results)"},
            "EV-HO-003": {"request": {**req, "limit": 8}, "harness": {"checkpoint": 0.50, "baseline": 0.50},
                          "expected": "UNMEASURED with EV-HO-003"},
            "EV-HO-004": {"request": req, "tamper": "delete report before handoff", "expected": "RED with EV-HO-004"},
            "EV-HO-005": {"request": held_req,
                          "harness": {"heldout": {"checkpoint": {"rows_used": 171, "rows_dropped": 0, "mean_nll_nats": 0.5,
                                                                 "perplexity": 1.66},
                                                  "baseline": {"rows_used": 170, "rows_dropped": 1, "mean_nll_nats": 1.4,
                                                               "perplexity": 4.19}}},
                          "expected": "UNMEASURED 95 with EV-HO-005 (row counts differ: never PASS)"},
            "EV-HO-006": {"request": held_req,
                          "harness": {"heldout": {"checkpoint": {"rows_used": 171, "rows_dropped": 0, "mean_nll_nats": 0.5},
                                                  "baseline": {"rows_used": 171}}},
                          "expected": "RED 5 with EV-HO-006 (row lacks the measurement block)"},
        }
