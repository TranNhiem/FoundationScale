
"""DataEngineSkill: turn raw sources into a measured, FS-ready dataset.

Pipeline: recommend (or accept an explicit) spec -> ``run_pipeline`` ->
``build_readiness`` -> three artifacts (data_pipeline_spec, dataset,
readiness_report) written to ``ctx.artifacts_dir``. The result status mirrors
the readiness verdict: PASS/UNMEASURED/RED. Phase-2 ops refuse, naming the
missing capability.

Note on rule phases: handoff rules DE-HO-* are emitted inside ``run()`` (not
in ``check_handoff``) because ``BaseSkill.execute`` only calls check_handoff
for PASS results, while a RED/UNMEASURED readiness must still carry its
finding. ``check_handoff`` therefore re-verifies nothing on PASS and returns
no findings.
"""
from __future__ import annotations

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
from foundationskills.core.provenance import make_provenance, sha256_json
from foundationskills.core.status import Status
from foundationskills.skills.data_engine.catalog import discover
from foundationskills.skills.data_engine.mix import design_mixture
from foundationskills.skills.data_engine.phase2 import PHASE2, Phase2NotImplemented
from foundationskills.skills.data_engine.pipeline import PipelineError, run_pipeline
from foundationskills.skills.data_engine.recommend import FORMATS, recommend_pipeline
from foundationskills.skills.data_engine.report import build_readiness

LOCAL_KINDS = {"local_dir", "jsonl", "json", "parquet", "csv", "documents", "image_text"}
_PREFERENCE_ALGOS = ("dpo", "ipo", "kto", "orpo", "simpo", "cpo", "online_dpo", "iterative_dpo")


class DataEngineSkill(BaseSkill):
    """Recommend, run, and measure a data pipeline; hand a readiness report downstream."""

    name = "data_engine"
    version = "0.1.0"
    description = (
        "Turn raw data sources (local files/dirs or HF datasets) into an FS-ready "
        "dataset with per-op accounting and a measured readiness report."
    )
    scope = Scope(
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=(),
        hardware=(),
    )
    consumes = ("raw_data_ref",)
    produces = ("data_pipeline_spec", "dataset", "readiness_report")
    input_schema = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "sources": {"type": "array"},
            "target_format": {},
            "pipeline": {"type": "object"},
            "requirements": {"type": "object"},
        },
    }
    output_schema = {"type": "object", "additionalProperties": True}
    rules = (
        RuleSpec("DE-IN-001", "no sources provided (and no explicit pipeline)", Severity.BLOCK, "input"),
        RuleSpec("DE-IN-002", "unknown target_format", Severity.BLOCK, "input"),
        RuleSpec("DE-IN-003", "local source uri does not exist", Severity.BLOCK, "input"),
        RuleSpec("DE-IN-004", "a phase-2 op was requested; capability not implemented", Severity.BLOCK, "input"),
        RuleSpec("DE-IN-005", "sft/mm_sft without a chat_template_family or tokenizer", Severity.WARN, "input"),
        RuleSpec("DE-IN-006", "preference dataset requested but FS cannot train the preference family", Severity.WARN, "input"),
        RuleSpec("DE-HO-001", "readiness verdict is RED", Severity.BLOCK, "handoff"),
        RuleSpec("DE-HO-002", "readiness verdict is UNMEASURED; some checks were not run", Severity.WARN, "handoff"),
        RuleSpec("DE-HO-003", "zero records were written", Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(
        entries=(),
        emits=("data_pipeline_spec", "dataset", "readiness_report"),
        apis=("run_pipeline", "recommend_pipeline", "build_readiness", "design_mixture", "discover"),
        notes=(
            "Emits datasets in the record shapes consumed by foundationscale-train "
            "(text column / mm image column / rl gold key); consumes nothing from FS."
        ),
    )

    # ---- input checks -----------------------------------------------------

    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        sources = list(request.get("sources") or [])
        if not sources and not request.get("pipeline"):
            findings.append(
                self.finding(
                    "DE-IN-001",
                    "no data sources provided",
                    {},
                    "pass `sources: [{uri, kind}]` (a raw_data_ref payload) or an explicit `pipeline` spec with self-sufficient op configs",
                )
            )

        target = request.get("target_format")
        pipeline = request.get("pipeline")
        effective_target = target if target is not None else (pipeline or {}).get("target_format")
        if not isinstance(effective_target, str) or effective_target not in FORMATS:
            findings.append(
                self.finding(
                    "DE-IN-002",
                    f"unknown target_format {effective_target!r}; known: {list(FORMATS)}",
                    {"target_format": effective_target, "known": list(FORMATS)},
                    "choose one of: pretrain, cpt, sft, mm_sft, preference, rl",
                )
            )

        if isinstance(pipeline, dict):
            for step in list(pipeline.get("ops") or []):
                op_name = str((step or {}).get("op", ""))
                if op_name in PHASE2:
                    findings.append(
                        self.finding(
                            "DE-IN-004",
                            f"phase-2 op requested: phase-2: {op_name} ({PHASE2[op_name]})",
                            {"op": op_name},
                            "remove the phase-2 op or implement the phase-2 capability first",
                        )
                    )

        for source in sources:
            kind = str((source or {}).get("kind", ""))
            uri = str((source or {}).get("uri", ""))
            if kind in LOCAL_KINDS and uri and not Path(uri).exists():
                findings.append(
                    self.finding(
                        "DE-IN-003",
                        f"source uri does not exist: {uri!r} (kind={kind!r})",
                        {"uri": uri, "kind": kind},
                        "fix the path, mount the data, or switch to an hf_dataset source",
                    )
                )

        if effective_target in {"sft", "mm_sft"} and (
            not request.get("chat_template_family") or not request.get("tokenizer")
        ):
            findings.append(
                self.finding(
                    "DE-IN-005",
                    "sft/mm_sft target without a chat_template_family or tokenizer",
                    {
                        "chat_template_family": request.get("chat_template_family"),
                        "tokenizer": request.get("tokenizer"),
                    },
                    "pass the base model's tokenizer and chat template family so rendered text matches training templates",
                )
            )

        if effective_target == "preference":
            caps = getattr(ctx, "capabilities", None)
            if caps is None or not getattr(caps, "rl_runnable", None):
                detail = "FS capabilities were not probed for this run"
            else:
                runnable = [
                    a for a in _PREFERENCE_ALGOS if caps.rl_runnable.get(a) is None and a in caps.rl_runnable
                ]
                detail = (
                    "no preference algorithm is runnable on the probed FS"
                    if not runnable
                    else f"runnable preference algorithms on probed FS: {runnable}"
                )
            findings.append(
                self.finding(
                    "DE-IN-006",
                    "preference dataset requested; the measured FS preference family is not trainable "
                    f"({detail}). The dataset is still useful later, but not trainable on the installed FS.",
                    {"capabilities": detail},
                    "train SFT now and convert to an RL stage with verifiable rewards, or run preference training outside FS",
                )
            )
        return findings

    # ---- run ---------------------------------------------------------------

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        sources = list(request.get("sources") or [])
        # rv11: the same effective target check_inputs validated (explicit pipeline wins when set)
        target = str(request.get("target_format") or (request.get("pipeline") or {}).get("target_format"))
        explicit = request.get("pipeline")
        if isinstance(explicit, dict) and explicit:
            spec = dict(explicit)
            spec.setdefault("tokenizer", request.get("tokenizer"))
            spec.setdefault("chat_template_family", request.get("chat_template_family"))
        else:
            spec = recommend_pipeline(
                target_format=target,
                sources=sources,
                goal=request.get("goal"),
                algorithm=request.get("algorithm"),
                tokenizer=request.get("tokenizer"),
                chat_template_family=request.get("chat_template_family"),
                domain=request.get("domain"),
                benchmarks=list(request.get("benchmarks") or []),
                seq_len=int(request.get("seq_len") or 4096),
                drop_overlong=bool(request.get("drop_overlong", False)),
            )

        dataset_id = f"{target}-{sha256_json({'sources': sources, 'spec': spec})[:12]}"
        out_dir = Path(request.get("out_dir") or (ctx.workdir / "data" / dataset_id))

        try:
            result = run_pipeline(spec, out_dir)
        except Phase2NotImplemented as exc:
            op_name = str(exc).split("phase-2: ", 1)[-1]
            finding = self.finding(
                "DE-IN-004",
                f"pipeline selected a phase-2 op: phase-2: {op_name}",
                {"op": op_name},
                "remove the phase-2 op from the spec",
            )
            return SkillResult(status=Status.REFUSED, payload={}, findings=(finding,), refusal=f"phase-2: {op_name}")
        except PipelineError:
            raise  # a measured spec failure: RED via CORE-EXC with the diagnosis below

        dataset = result.dataset
        requirements = {
            "target_format": target,
            "min_tokens": None,
            "min_records": 1,
            "max_truncation_rate": 0.02,
            "require_dedup": True,
            "require_decontam": bool(request.get("benchmarks")),
            "max_pii_remaining": 0,
            "chat_template_family": request.get("chat_template_family"),
        }
        requirements.update(dict(request.get("requirements") or {}))
        readiness = build_readiness(dict(dataset), list(result.stats), requirements)

        provenance = make_provenance(
            self.name, self.version, {"sources": sha256_json(sources)[:16], "spec": sha256_json(spec)[:16]}
        )
        artifacts = []
        artifacts.append(write_artifact(Artifact("data_pipeline_spec", f"{dataset_id}-spec", spec, provenance), ctx.artifacts_dir))
        dataset_ref = None
        if int(dataset.get("num_records", 0)) > 0:
            dataset_ref = write_artifact(Artifact("dataset", dataset_id, dataset, provenance), ctx.artifacts_dir)
            artifacts.append(dataset_ref)
        artifacts.append(
            write_artifact(Artifact("readiness_report", f"{dataset_id}-readiness", readiness, provenance), ctx.artifacts_dir)
        )

        payload = {
            "dataset_id": dataset_id,
            "out_dir": str(out_dir),
            "data_pipeline_spec": spec,
            "dataset": dataset,
            "readiness_report": readiness,
        }

        findings: list[Finding] = []
        if int(dataset.get("num_records", 0)) == 0:
            ingest_stats = next((st for st in result.stats if st.get("name") == "ingest"), {})
            unmapped = int((ingest_stats.get("modified") or {}).get("no_payload_mapped", 0))
            columns = (ingest_stats.get("extra") or {}).get("source_columns", {})
            if unmapped:
                message = (f"pipeline wrote zero records; {unmapped} ingested record(s) had no text/messages/prompt "
                           f"mapped from source columns {columns}")
                recovery = ('map the columns explicitly via source options.field_map, e.g. '
                            '{"prompt": "<question col>", "answer": "<answer col>"} or {"text": "<body col>"}')
            else:
                message = "pipeline wrote zero records; the dataset artifact was not written"
                recovery = "inspect stats.json dropped-reason counters; the clean/quality thresholds may be too strict"
            findings.append(
                self.finding(
                    "DE-HO-003",
                    message,
                    {"stats_path": str(out_dir / "stats.json"), "dropped": _dropped_summary(result.stats),
                     "source_columns": columns},
                    recovery,
                )
            )
            return SkillResult(Status.RED, payload, tuple(artifacts), tuple(findings))

        verdict = str(readiness.get("verdict", "UNMEASURED"))
        if verdict == "RED":
            failed = [c.get("rule_id") for c in readiness.get("checks", []) if c.get("passed") is False]
            findings.append(
                self.finding(
                    "DE-HO-001",
                    f"readiness verdict is RED; failed checks: {', '.join(map(str, failed)) or '<none listed>'}",
                    {"failed_checks": failed},
                    "fix the failing readiness checks (see the readiness_report artifact) before training",
                )
            )
            return SkillResult(Status.RED, payload, tuple(artifacts), tuple(findings))
        if verdict != "PASS":
            unmeasured = [c.get("rule_id") for c in readiness.get("checks", []) if c.get("passed") is None]
            findings.append(
                self.finding(
                    "DE-HO-002",
                    f"readiness verdict is UNMEASURED; checks not run: {', '.join(map(str, unmeasured)) or '<none listed>'}",
                    {"null_checks": unmeasured},
                    "provide the missing inputs (e.g. benchmark files for decontam) so every check runs",
                )
            )
            return SkillResult(Status.UNMEASURED, payload, tuple(artifacts), tuple(findings))
        return SkillResult(Status.PASS, payload, tuple(artifacts), tuple(findings))

    # ---- handoff / diagnosis / fixtures ------------------------------------

    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """No-op: DE-HO-* findings are raised in run() (see module docstring)."""
        return []

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        if isinstance(failure, SkillResult):
            ids = {f.rule_id for f in failure.findings}
            if "DE-HO-003" in ids:
                return Diagnosis(
                    "empty dataset",
                    ("all records were dropped by clean/quality thresholds", "source was empty or unreadable"),
                    ("inspect out_dir/stats.json per-op `dropped` counters",),
                    ("relax clean/quality thresholds", "check the ingest source mapping",),
                )
            return Diagnosis(
                failure.refusal or "readiness RED/UNMEASURED",
                ("failing readiness checks", "unmeasured checks (missing inputs)"),
                ("read the readiness_report artifact checks and their details",),
                ("fix the failing check inputs", "provide benchmark files for decontam"),
            )
        text = f"{type(failure).__name__}: {failure}"
        if "IngestError" in text or "not installed" in text or ("ImportError" in text and "data" in text):
            return Diagnosis(
                "ingest backend missing",
                ("optional ingest dependency is not installed",),
                ("look for the op that raised and its required package",),
                ("pip install foundationskills[data]",),
            )
        if isinstance(failure, Phase2NotImplemented):
            return Diagnosis(
                str(failure),
                ("a phase-2 capability was requested",),
                ("check data_engine.phase2.PHASE2",),
                ("remove the op or implement the phase-2 capability",),
            )
        if isinstance(failure, PipelineError):
            return Diagnosis(
                str(failure),
                ("unknown op name or first op is not ingest",),
                ("compare the spec ops against data_engine.ops.OPS",),
                ("fix the op sequence; the first op must be 'ingest'",),
            )
        if "tokenizer" in text.lower():
            return Diagnosis(
                text,
                ("unknown tokenizer id or unreachable tokenizer backend",),
                ("check the tokenizer name; the builtin fallback counts words when transformers is absent",),
                ("pass a loadable tokenizer id or install transformers",),
            )
        if isinstance(failure, MemoryError):
            return Diagnosis(
                "out of memory during dedup",
                ("MinHash LSH tables too large for RAM",),
                ("lower num_perm/bands in dedup config",),
                ("use the datatrove dedup backend", "shard the input before dedup"),
            )
        if "decontam" in text.lower():
            return Diagnosis(
                text,
                ("benchmark files missing: decontamination is UNMEASURED without them",),
                ("pass `benchmarks` and benchmark data files",),
                ("provide the evaluation benchmark files",),
            )
        return Diagnosis(
            text,
            ("unexpected pipeline failure",),
            ("rerun with verbose logging; inspect out_dir/stats.json",),
            ("fix per the exception message and rerun",),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        """One negative fixture per rule.

        String values containing ``{tmp}`` are replaced by the test with an
        actual tmp_path; the optional ``files`` map (relpath -> content) is
        materialized there first.
        """
        src = lambda uri, kind: [{"uri": uri, "kind": kind}]  # noqa: E731 - fixture shorthand
        one_record = json_line('{"id": "r1", "text": "hello world from fixture"}')
        return {
            "DE-IN-001": {"request": {"target_format": "cpt"}, "files": {}},
            "DE-IN-002": {
                "request": {"target_format": "bogomips", "sources": src("{tmp}/corpus.jsonl", "jsonl")},
                "files": {"corpus.jsonl": one_record},
            },
            "DE-IN-003": {
                "request": {"target_format": "cpt", "sources": src("{tmp}/missing.jsonl", "jsonl")},
                "files": {},
            },
            "DE-IN-004": {
                "request": {
                    "target_format": "pretrain",
                    "sources": src("{tmp}/corpus.jsonl", "jsonl"),
                    "pipeline": {
                        "target_format": "pretrain",
                        "seed": 0,
                        "tokenizer": None,
                        "rationale": [],
                        "ops": [
                            {"op": "ingest", "config": {"sources": src("{tmp}/corpus.jsonl", "jsonl")}},
                            {"op": "semantic_dedup", "config": {}},
                        ],
                    },
                },
                "files": {"corpus.jsonl": one_record},
            },
            "DE-IN-005": {
                "request": {
                    "target_format": "sft",
                    "sources": src("{tmp}/corpus.jsonl", "jsonl"),
                    "pipeline": {
                        "target_format": "sft",
                        "seed": 0,
                        "tokenizer": None,
                        "rationale": [],
                        "ops": [
                            {"op": "ingest", "config": {"sources": src("{tmp}/corpus.jsonl", "jsonl")}},
                            {"op": "format", "config": {"target_format": "sft"}},
                        ],
                    },
                },
                "files": {"corpus.jsonl": one_record},
            },
            "DE-IN-006": {
                "request": {
                    "target_format": "preference",
                    "sources": src("{tmp}/corpus.jsonl", "jsonl"),
                    "pipeline": {
                        "target_format": "preference",
                        "seed": 0,
                        "tokenizer": None,
                        "rationale": [],
                        "ops": [
                            {"op": "ingest", "config": {"sources": src("{tmp}/corpus.jsonl", "jsonl")}},
                            {"op": "format", "config": {"target_format": "preference"}},
                        ],
                    },
                },
                "files": {"corpus.jsonl": one_record},
            },
            "DE-HO-001": {
                "request": {
                    "target_format": "cpt",
                    "sources": src("{tmp}/corpus.jsonl", "jsonl"),
                    "requirements": {"min_records": 1000000, "require_decontam": False, "require_dedup": False},
                },
                "files": {"corpus.jsonl": one_record},
            },
            "DE-HO-002": {
                "request": {
                    "target_format": "cpt",
                    "sources": src("{tmp}/corpus.jsonl", "jsonl"),
                    "pipeline": {
                        "target_format": "cpt",
                        "seed": 0,
                        "tokenizer": None,
                        "rationale": [],
                        "ops": [
                            {"op": "ingest", "config": {"sources": src("{tmp}/corpus.jsonl", "jsonl")}},
                            {"op": "dedup", "config": {}},
                            {"op": "format", "config": {"target_format": "cpt"}},
                        ],
                    },
                    # no clean op ran -> remaining PII is unmeasured (null) -> UNMEASURED.
                    # (A missing REQUIRED dedup is a measured failure, i.e. RED, not this.)
                    "requirements": {"require_dedup": True, "require_decontam": False, "min_records": 1},
                },
                "files": {"corpus.jsonl": one_record},
            },
            "DE-HO-003": {
                "request": {"target_format": "cpt", "sources": src("{tmp}/corpus.jsonl", "jsonl")},
                "files": {"corpus.jsonl": ""},
            },
        }


def json_line(payload: str) -> str:
    """Keep fixture files as explicit JSONL strings (one record per line)."""
    return payload + "\n"


def _dropped_summary(stats: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in stats:
        for reason, count in (entry.get("dropped") or {}).items():
            out[f"{entry.get('name')}:{reason}"] = int(count)
    return out
