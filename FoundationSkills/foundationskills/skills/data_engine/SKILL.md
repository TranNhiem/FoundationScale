
# Data Engine

## Purpose
Turn raw data sources into a measured, FS-ready dataset. The skill recommends
(or accepts) a `data_pipeline_spec`, streams it through the registered ops
(`ingest -> clean -> dedup -> quality -> [decontam] -> [mix] -> format ->
tokenize`), writes sharded JSONL plus per-op accounting, and emits a readiness
report whose verdict becomes the skill status. Nothing trains: output rows are
exactly the record shapes `foundationscale-train` / `fskills-rl` consume.

## When to use
- You have raw sources (JSONL/JSON/CSV/parquet files, document dirs of
  txt/html/pdf, HF datasets, image-text folders) and need CPT/pretrain corpus,
  SFT rows, preference pairs, or an RL verifiable prompt set.
- A training plan hands you `{"handoff": "data_engine", "target_format": ...}`.
- You want per-stage drop accounting (`stats.json`) before spending GPU hours.

Do NOT use it for: training itself (training.planner / training.emit), semantic
dedup, synthetic generation, tool-call normalization or video ingest (phase 2).

## Inputs
Request object (validated against the skill input schema):
- `sources` (required unless `pipeline` is given): `[{uri, kind}]` where kind
  is one of `local_dir | jsonl | json | parquet | csv | hf_dataset | documents
  | image_text` (a `raw_data_ref` payload).
- `target_format` (required): `pretrain | cpt | sft | mm_sft | preference | rl`.
- Optional: `goal`, `algorithm`, `tokenizer`, `chat_template_family`, `domain`,
  `benchmarks` (triggers decontam), `pipeline` (explicit spec; overrides the
  recommendation), `requirements` (readiness overrides), `out_dir`.

## Outputs
Three artifacts written to `<workdir>/artifacts/`:
- `data_pipeline_spec` — the executed op sequence with rationale.
- `dataset` — shards (path+sha256+records), schema columns, `fs_columns`
  (`text_column` / `image_column` / `gold_key`), `num_tokens` (null when only
  an approximate count exists; see `num_tokens_approx`).
- `readiness_report` — verdict PASS/RED/UNMEASURED plus per-check details.

Status mirrors the readiness verdict. REFUSED names the missing input or the
phase-2 capability (`phase-2: <op>`). Zero records written is RED with
DE-HO-003 (the dataset artifact is skipped, since an empty shard list is
schema-invalid).

Payload extras: `stats.json` in `out_dir` with every op's in/out/dropped
counters and the realized mixture.

## Scope
- Stages: pretrain, cpt, sft, preference, rl (mm_sft via image_column).
- Model types: llm and vlm. Families: any. Methods/hardware: not applicable.
- FS interface: emits artifacts only; consumes nothing from FS directly, but
  consults `ctx.capabilities` for the preference-family warning (DE-IN-006).

## Validation rules
| Rule id | Phase | Severity | Fires when |
|---|---|---|---|
| DE-IN-001 | input | BLOCK | no sources given (and no explicit pipeline) |
| DE-IN-002 | input | BLOCK | `target_format` not in the known set |
| DE-IN-003 | input | BLOCK | a local source uri does not exist |
| DE-IN-004 | input | BLOCK | a phase-2 op name appears in the spec → REFUSED `phase-2: <op>` |
| DE-IN-005 | sft/mm_sft | WARN | chat_template_family or tokenizer missing |
| DE-IN-006 | input | WARN | preference target while the installed FS cannot run the preference family |
| DE-HO-001 | handoff | BLOCK | readiness verdict is RED |
| DE-HO-002 | handoff | WARN | readiness verdict is UNMEASURED (null checks listed) |
| DE-HO-003 | handoff | BLOCK | zero records written |
| DE-RDY-001 | readiness | check | dataset format matches `requirements.target_format` |
| DE-RDY-002 | readiness | check | `num_tokens` >= `min_tokens` (null → unmeasured) |
| DE-RDY-003 | readiness | check | `num_records` >= `min_records` |
| DE-RDY-004 | readiness | check | tokenize truncation rate <= `max_truncation_rate` |
| DE-RDY-005 | readiness | check | dedup ran when `require_dedup` (missing → unmeasured) |
| DE-RDY-006 | readiness | check | dedup duplicate-rate sanity from dedup stats |
| DE-RDY-007 | readiness | check | decontam ran when `require_decontam`, zero hits required |
| DE-RDY-008 | readiness | check | PII remaining <= `max_pii_remaining` |
| DE-RDY-009 | readiness | check | chat template family matches `chat_template_family` when given |
| DE-RDY-010 | readiness | check | fs_columns present and non-empty shards |

(Readiness rule ids DE-RDY-001..010 are emitted by
`foundationskills.skills.data_engine.report.build_readiness`; consult the
readiness_report artifact `checks` for per-rule details.)

## Failure handling
- REFUSED: unknown format (DE-IN-002), missing local files (DE-IN-003), no
  sources (DE-IN-001), phase-2 op (DE-IN-004) — fix the request, not the data.
- RED: readiness failure (DE-HO-001) or empty output (DE-HO-003). Read
  `out_dir/stats.json`: every op reports `dropped` by reason, so find which
  stage ate the records (usually strict `clean`/`quality` thresholds).
- UNMEASURED: some readiness checks never ran (DE-HO-002) — e.g. decontam
  without benchmark files, or dedup missing from the pipeline while
  `require_dedup` is true.
- Ingest backend missing → `pip install foundationskills[data]`; the builtin
  fallback is an approximation and marks its stats (`extra["approximate"]`),
  which surfaces as UNMEASURED where it matters. OOM in dedup → lower
  `num_perm`/`bands` or use the datatrove backend.

## FS interface
- Emits: `data_pipeline_spec`, `dataset`, `readiness_report` artifacts.
- Record shapes (exactly what FS consumes): pretrain/cpt `{id, text, meta}`;
  sft `{id, messages, text, meta}` (text = chat-template-rendered); mm_sft adds
  `image` (training sets `FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN=image`);
  preference `{id, prompt, chosen, rejected, meta}`; rl ShareGPT-style
  `{conversations: [{from, value}...], answer}` with `fs_columns.gold_key`.
- APIs for callers: `recommend_pipeline`, `run_pipeline`, `build_readiness`,
  `design_mixture`, `catalog.discover`. No direct processes are launched.

## Worked examples
1. CPT from internal PDFs + HTML docs:
   ```
   request = {
     "sources": [{"uri": "/data/manuals", "kind": "documents", "domain": "manufacturing"}],
     "target_format": "cpt", "domain": "manufacturing",
     "goal": "domain_expert", "tokenizer": "gemma4-4b",
     "benchmarks": ["gsm8k"]          # triggers decontam
   }
   ```
   Recommended flow: ingest → clean (strict PII, doc cleaning) → dedup
   (document+near) → quality (gopher_fineweb) → decontam → format(cpt) →
   tokenize (pack). Design the general-replay share with
   `design_mixture(stage="cpt", preserve_general=True, ...)`.

2. SFT from Alpaca-style JSONL for gemma4:
   ```
   request = {
     "sources": [{"uri": "/data/alpaca.jsonl", "kind": "jsonl"}],
     "target_format": "sft", "tokenizer": "gemma4-4b",
     "chat_template_family": "gemma4",
     "goal": "general_chat"
   }
   ```
   Rows are rendered with the gemma4 chat template into `text`; note FS trains
   loss over the full rendered text (no assistant-only masking).

3. RL verifiable math from a HF dataset:
   ```
   request = {
     "sources": [{"uri": "openai/gsm8k", "kind": "hf_dataset",
                  "options": {"config": "main", "split": "train"}}],
     "target_format": "rl", "goal": "math", "algorithm": "dr_grpo",
     "benchmarks": ["gsm8k"]
   }
   ```
   Output records carry the gold answer under `fs_columns.gold_key="answer"`;
   the emit skill turns this into an `fskills-rl` config for the runnable
   dr_grpo/gspo/dapo family.

## Phase 2
Declared (via `data_engine.phase2.PHASE2`) but NOT implemented; selecting any
of these op names refuses with `phase-2: <name>`:
- `semantic_dedup` — embedding-based near-dedup.
- `synthesize` — teacher-model synthetic data generation.
- `toolcall_format` — tool-calling trace normalization (tool_calling goals are
  flagged in mixtures for this reason).
- `video_ingest` — video → text records (frames + ASR alignment).
