---
name: fs-data
license: MIT
description: "User-invoked slash command: run ONLY when the user types /fs-data; never choose it on your own. Slash-command data preparation workflow (/fs-data) that composes the model skill data_engine to turn raw data refs into a dataset, data_pipeline_spec and readiness_report. Inputs: the data_engine request fields (sources, target_format, tokenizer, chat_template_family, out_dir, requirements, pipeline, reasoning_unsupported, benchmarks, ...) passed through unchanged, plus optional goal_spec (a path recorded in outputs, not interpreted). The workflow record says which step ran, what it produced and the verdict (PASS/RED/UNMEASURED). Do NOT use for explaining fskills results (use /fs-status), fixing data problems yourself, or training a model (use /fs-training): this composes data_engine and produces data artifacts only."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI). It requires no FoundationScale: it composes the model skill data_engine through the composition helper (_workflow) and writes one workflow_record artifact. (requires_fs is False)."
metadata:
  version: "0.1.0"
  invocation: "user"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["use fs-data to run the data preparation workflow","call fs-data to prepare a training dataset from raw data sources","run /fs-data to validate my data sources and generate a dataset","create a data pipeline and check readiness via fs-data","data_engine composed through fs-data to build a dataset with a readiness report"]'
---

# fs-data — the data preparation workflow composer (/fs-data)

## Purpose

The data-preparation workflow: `/fs-data` composes the model skill `data_engine` through the
composition helper (`_workflow`) to turn raw data refs (`sources`) into a training dataset,
`data_pipeline_spec` and `readiness_report`. The caller provides the `data_engine` request fields
(passed through unchanged) plus an optional `goal_spec` path. The skill writes exactly one
`workflow_record` artifact whose payload says which step ran, what it produced (by path), and the
verdict (PASS, RED or UNMEASURED). A user skill composes model skills (here `data_engine`), never
another user skill. Do NOT use for explaining fskills results (use `/fs-status`), fixing data problems
yourself, or launching a training run (use `/fs-training`): this composes `data_engine` to produce
data artifacts only.

## When to use

- "Use fs-data to run the data preparation workflow for my raw data on S3" / "Run /fs-data to validate
  my data sources and generate a dataset".
- To compose `data_engine` to build a dataset from raw data refs, with a data_pipeline_spec and
  readiness_report as outputs.
- To run the data preparation step with a `goal_spec` path (recorded in outputs, not interpreted) and
  know the verdict (PASS ready for training; RED fix and re-run; UNMEASURED readiness checks not run).
- Not for explaining fskills results (use `/fs-status`), fixing data problems yourself, or training a
  model (use `/fs-training`): compose `data_engine` here, act there.

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `sources` | one of `sources` / `pipeline` | list of data source URIs (e.g. `s3://...`, `local:/...`) |
| `target_format` | no | the target data format (e.g. `sft`, `preference`, `chat`) |
| `tokenizer` | no | tokenizer name or path |
| `chat_template_family` | no | chat template to apply |
| `out_dir` | no | output directory for the dataset |
| `requirements` | no | extra data requirements |
| `pipeline` | one of `sources` / `pipeline` | explicit pipeline definition (object) |
| `reasoning_unsupported` | no | flag for reasoning data handling |
| `benchmarks` | no | benchmark references |
| `goal_spec` | no | path to a goal spec (recorded in outputs, not interpreted) |

## Outputs

One `workflow_record` artifact (typed, schema-validated, with provenance) whose payload is:
`workflow` (the slash name `"fs_data"`), `steps` (a list of `Step` rows: `{skill, status, exit_code,
refusal, artifacts, findings}`), `outputs` (`{"dataset": path|None, "readiness_report": path|None,
"data_pipeline_spec": path|None, "goal_spec": path|None}` — artifacts referenced BY PATH, never
copied), and `verdict` (`"PASS"`, `"RED"` or `"UNMEASURED"`). Exit codes: 0 PASS, 5 RED (FD-HO-001),
95 UNMEASURED (FD-HO-002 as WARN), 96 REFUSED (FD-IN-001 or FD-IN-002); nothing is written for
REFUSED.

## Scope

The full FoundationSkills data catalogue: any model type, family and stage. Composes exactly one
model skill (`data_engine`); writes one `workflow_record` artifact; requires no FoundationScale.
A user skill composes model skills (`data_engine`), never another user skill.

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK
refuses before any work — status REFUSED, exit 96, nothing written. A **handoff**-phase BLOCK turns
the output to RED, exit 5. WARN findings never block; they travel with the result. Phase by prefix:
FD-IN-* rules are input phase, FD-HO-* rules are handoff phase.

| Rule | Severity | Fires when |
|---|---|---|
| FD-IN-001 | BLOCK | no `sources` and no explicit `pipeline` is given → REFUSED 96 |
| FD-IN-002 | BLOCK | the composed `data_engine` refused (e.g. unknown target_format, missing local source) → REFUSED 96 |
| FD-HO-001 | BLOCK | `data_engine` came back RED (e.g. zero records, readiness RED) → RED 5 |
| FD-HO-002 | WARN | `data_engine` came back UNMEASURED (readiness checks not run) → UNMEASURED 95 (WARN finding; never PASS) |

The composed model skill `data_engine` has its own rules (DE-IN-001..DE-IN-008, DE-HO-001..DE-HO-003)
that govern its own run; those rule ids are not returned here — inner findings travel as step evidence
in the workflow record and the FD-HO-001 / FD-HO-002 findings.

## Failure handling

- No `sources` and no explicit `pipeline`: REFUSED (96) naming FD-IN-001; nothing is written. Pass
  `sources` or `pipeline` and re-run.
- `data_engine` refused: REFUSED (96) naming FD-IN-002; fix the data_engine findings or refusal it
  names (e.g. unknown target_format DE-IN-002, missing local source DE-IN-003) and re-run.
- `data_engine` came back RED: RED (5) naming FD-HO-001 (e.g. zero records, readiness RED). Fix the
  data_engine findings and re-run; never treat a RED dataset as ready for training.
- `data_engine` came back UNMEASURED: UNMEASURED (95) with FD-HO-002 WARN finding (readiness checks
  not run). Resolve the readiness issues and re-run; never treat UNMEASURED as PASS.
- This skill composes and explains only. On the owning data_engine's failure, call that skill's
  `diagnose()`; do not improvise a fix here.

## FS interface

Composes `data_engine` through `_workflow` (run_step, outcome_class, registry, Step) to turn raw data
refs into a dataset with a readiness report. (`fskills data` entry point.) Never imports or launches
FoundationScale; writes one `workflow_record` artifact and touches no cluster resource. `requires_fs`
is False: on a host with only the `foundationskills` package installed it still composes `data_engine`
and records the workflow. On a host where even the `foundationskills` package is missing, report
REFUSED with the install step (`pip install -e <FoundationScale repo>/FoundationSkills`) and never
improvise a data preparation run.

## Worked examples

- `fs-data {"sources": ["s3://bucket/train.jsonl"], "target_format": "sft"}` — runs `data_engine` on
  the S3 source, produces `dataset` (JSONL), `data_pipeline_spec` and `readiness_report`, and returns a
  `workflow_record` with verdict PASS (data is ready).
- `fs-data {"pipeline": {"raw_data_ref": "local:/data/*.parquet"}}` — runs `data_engine` with an
  explicit pipeline definition, no `sources` required.
- `fs-data {}` — REFUSED 96 (FD-IN-001): no `sources` and no `pipeline` given.
