# FoundationSkills architecture

## 1. Layers and responsibilities

| Layer | Owns | Never does |
|---|---|---|
| Foundation Agent (Claude Code or any agent) | the conversation, choosing skills, asking the user, the launch confirmation | invent facts or bypass a refusal |
| FoundationSkills (`foundationskills`) | deciding **what**: data pipelines, stages, algorithms, method, estimates, commands | train |
| FoundationScale (`foundationscale`) | executing **how**: SFT/CPT through `foundationscale-train`, RL through `RLTrainer`, save gates, run manifests | plan |

## 2. The standard skill contract (`foundationskills/core/contract.py`)

Every skill, current and future, is a `BaseSkill`:

| Contract element | Where |
|---|---|
| Inputs and outputs with schemas | `input_schema` / `output_schema` (JSON-Schema subset, `core/schema.py`); `consumes` / `produces` artifact types |
| Supported scope | `scope: Scope(model_types, families, stages, algorithms, methods, hardware)` |
| Validation rules | `rules: tuple[RuleSpec]` (id, severity BLOCK/WARN/INFO, phase input/handoff); `check_inputs`, `check_handoff` |
| Failure handling | `diagnose(failure) -> Diagnosis(symptom, likely_causes, checks, recovery)` |
| FoundationScale interface | `fs_interface: FSInterface(entries, emits, apis)` |
| Documentation and examples | the skill's `SKILL.md` with the standard sections |
| Proof the rules work | `must_fire_fixtures()`: one bad input per rule that must trigger it |

`BaseSkill.execute()` is the one template every skill runs through:
1. The request is checked against its schema; a failure is REFUSED, naming the error.
2. `check_inputs` runs; a BLOCK finding is REFUSED, naming the rule.
3. `run()` runs; an exception is RED with a diagnosis.
4. The output is checked against its schema.
5. `check_handoff` runs; a BLOCK finding is RED.
6. A finding from an undeclared rule makes the result RED.

Statuses map to FoundationScale's exit codes: PASS 0, RED 5, UNMEASURED 95, REFUSED 96.

`tests/contract/test_conformance.py` checks every registered skill for:
- attributes;
- known artifact types;
- unique rule ids;
- a MUST_FIRE fixture for every rule;
- a `SKILL.md` with all standard sections that lists every rule id.

## 3. Artifacts: how skills compose

Skills exchange **typed artifacts**. Each is a JSON file (`<type>.<id>.json`, written atomically) holding a schema-validated payload and provenance: the skill and its version, the FoundationScale version, input hashes and a timestamp.

```
raw_data_ref ─ data_engine ─▶ data_pipeline_spec, dataset, readiness_report
goal_spec (+readiness_report) ─ training.planner ─▶ training_plan
training_plan + dataset ─ training.emit ─▶ fs_launch_spec (+ sbatch)
fs_launch_spec ─ fskills launch (user-confirmed hash) ─▶ FoundationScale run ─▶ run_manifest.json
```

`SkillRegistry.plan_chain(have, want)` finds the shortest chain of skills by artifact type. `Orchestrator` runs a chain, injects prior artifacts, stops at the first non-PASS result and writes `journal.jsonl`.

## 4. Measuring the installed FoundationScale (`interfaces/fs/capabilities.py`)

`probe(deep=True)` measures the following; nothing is hard-coded:

| Capability | How it is measured |
|---|---|
| Training flags and their legal values | the FS argparse parser, plus transformers' `OptimizerNames` / `SchedulerType` for flags FS forwards unvalidated |
| Objectives, sharding | parser and `loop.py` AST |
| Parallel axes | `--dry-run` per axis, with a control run; a refusal must name the axis |
| RL algorithms that *run* | FS's own `RLTrainer._resolve_objective` for each registered algorithm |
| RL reward kinds | FS's own gold reader given a letter and free-form golds |
| RL checkpoint persistence | the `RLTrainer` source |
| Model families | `foundationscale.families.registry` |

`FSCapabilities.check(stage, ...)` returns `None` or `missing: <what>`. The planner and the emitters mark stages and specs `executable: false`, with that string as the reason.

## 5. Knowledge as data (`skills/training/knowledge/`, `skills/data_engine/*.yaml`)

| Knowledge | File(s) | Schema |
|---|---|---|
| Model families (architecture, tokenizer, chat template, special tokens, context length, LoRA targets, quirks) | `families/<family>.yaml` | `schemas/knowledge/family.json` |
| GPU profiles (peak, measured MFU points by configuration, launch env, cluster rules) | `hardware/<id>.yaml` | `hardware.json` |
| Algorithm cards (stage, data format, requirements, key hyperparameters, failure modes) | `algorithms/<name>.yaml` | `algorithm_card.json` |
| Recipes | `recipes/<id>.yaml` | `recipe.json` (see `docs/RECIPES.md`) |
| Stage-selection rules | `stage_rules.yaml` | `stage_rules.json` |
| Mixing rules, public-dataset catalogue | `data_engine/mixing_rules.yaml`, `catalog/public_datasets.yaml` | |

Adding a model family, GPU, algorithm or recipe means adding a YAML file. Tests validate every file against its schema and cross-check it against the installed FoundationScale:
- families against FS's family registry;
- algorithms against FS's RL registry;
- recipe flags and values against the measured parser vocabulary.

## 6. Data Engine

A pipeline is an ordered list of ops. Each op has a strict config schema that the pipeline validates before running, and each records `OpStats` (in / out / dropped-by-reason / modified).

```
ingest → clean → dedup → quality → decontam → [mix] → format → tokenize
```

- The **recommender** chooses ops and configs from the target format, source kinds, goal and domain, and writes a rationale line for each choice.
- The **format** op writes exactly the records FoundationScale consumes:
  - `{text}` for pretrain/cpt;
  - `{messages, text}` for sft;
  - an MCQ ShareGPT record plus a gold letter for rl;
  - `{prompt, chosen, rejected}` for preference;
  - plus an image column for multimodal SFT.
- **Reasoning traces** are rendered through each family's template and then *verified* to have survived; Gemma-4's template drops them, so they are injected into its thinking channel.
- The **readiness report** turns every check into passed / failed / not-run. Its verdict is PASS, RED or UNMEASURED.

Phase 2 (`phase2.py`) defines `SemanticDedup`, `SyntheticGenerator`, `ToolCallFormatter` and `VideoTextIngest` as interfaces. Requesting one returns REFUSED `phase-2: <name>`, never a no-op.

## 7. Training skill

`plan()` works through these steps:

```
goal → model analysis (family registry) → data analysis (sources tagged use_for, readiness) → hardware
     → stage selection (stage_rules; canonical order cpt < sft < preference < rl unless goal.stage_order)
     → per stage: method (rules) → recipe (method is a hard key) → hparams (recipe hparams + fs.args)
                  → memory/time estimate (measured MFU point for this configuration) → feasibility
                  → executability (caps.check) → data hand-off
```

Every decision is recorded with a `because`. The emitted command **is** the configuration that was estimated: every estimate-relevant value is written into the stage. `training.emit` chains stages: stage k starts from stage k−1's `final`, and names the gap when that output is a LoRA adapter (a merge is needed) or an RL stage (FS saves no checkpoint).

## 8. Adding a skill (merging, benchmarking, compression, deployment, ...)

1. Choose the artifact types. Reserved and future types:

| Future skill | Consumes | Produces | Plugs in at |
|---|---|---|---|
| Model merging | `checkpoint` (LoRA adapter + base) | `checkpoint` (merged full weights) | closes the gap `training.emit` reports for LoRA → next stage |
| Benchmarking / evaluation | `checkpoint` | `eval_report` (already defined) | the planner reads `eval_report` to recommend the next stage |
| Compression / quantization | `checkpoint` | `checkpoint` (+ a quality report) | after training; use the model-compression preflight |
| Deployment | `checkpoint` | a serving spec (new type via `register_artifact_type`) | after evaluation |
| Experiment management | every artifact plus FS run manifests | an experiment index | reads `journal.jsonl` and the manifests |

2. Subclass `BaseSkill`, declare `rules` with MUST_FIRE fixtures, write `SKILL.md` with the standard sections, and register it through `register_builtin_skills` or the `foundationskills.skills` entry-point group.
3. `plan_chain` and the orchestrator route to it by artifact type. No change to existing skills is needed.
