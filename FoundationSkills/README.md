# FoundationSkills

FoundationSkills is the agent layer on top of FoundationScale. An engineer states a goal ("a manufacturing-domain reasoning model from a 7B base, on our H100 cluster, keep general ability"). An agent then uses the skills in this package to do the following:

1. prepare the data;
2. choose the stages, algorithms and method;
3. estimate resources and check feasibility;
4. emit ready-to-run FoundationScale commands.

The agent then launches them, but only after an explicit confirmation from the user.

```
Foundation Agent        (Claude Code / any agent)  -> reads SKILLS.md, calls skills
FoundationSkills        decides WHAT: data_engine, training.planner, training.emit, ...
FoundationScale         executes HOW: foundationscale-train (SFT/CPT), RLTrainer, gates, manifests
GPU infrastructure      GB200 (measured), H100/A100 (literature)
```

The skills never train. They produce data, plans and commands. FoundationScale trains.

## Principles

- **Measured, not assumed.** A probe asks the *installed* FoundationScale what it can run:
  - parser flags and their legal values;
  - the parallel axes, via dry-runs;
  - the RL algorithms that actually run;
  - the reward kinds;
  - whether RL saves checkpoints;
  - the registered model families.

  Every plan and command is checked against it. When the core gains a feature, the skills pick it up without edits.
- **FoundationScale's status doctrine.** Every result is PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96). A refusal names what is missing. A check that did not run is UNMEASURED, never PASS.
- **Knowledge is data.** Model families, GPU profiles, algorithm cards, recipes, stage rules, mixing rules and the public-dataset catalogue are schema-validated YAML with provenance. You add a family, recipe or GPU by adding a file, with no code change.
- **Skills talk through typed artifacts.** These are JSON on disk, written atomically, with provenance. That makes skills composable and lets future skills plug in (see `docs/ARCHITECTURE.md`).
- **No launch without confirmation.** `fskills launch` refuses (96) unless it gets the plan hash, which the agent must obtain from the user.

## Install

```bash
pip install -e FoundationSkills            # core: stdlib + PyYAML
pip install -e 'FoundationSkills[data,tokenize,test]'   # datasets/pyarrow, transformers, pytest
```

FoundationScale itself must be importable, because the probe measures it. Optional extras are `dedup`, `langid`, `pii`, `docs`, `html` and `pipeline`. They swap builtin implementations for stronger backends. An explicitly requested backend that is missing is a refusal, not a silent downgrade.

## Quick start

```bash
fskills probe --deep                         # what the installed FoundationScale actually runs
fskills data run  --request data.json        # raw data -> FS-ready dataset + readiness report
fskills plan      --goal goal.json           # goal -> training plan (stages, method, estimate, feasibility)
fskills emit      --plan plan.json --dataset ds.json --model M --output-root DIR --hardware gb200-189gb
fskills hash      --spec launch/fs_launch_spec.<run>.json     # the user confirms THIS hash
fskills launch    --spec launch/fs_launch_spec.<run>.json --confirm <hash>
fskills recipes list --stage sft --goal reasoning
fskills datasets discover --goal math --stage rl
```

Every command prints a JSON result and exits 0/5/95/96.

## Layout

```
foundationskills/
  core/            contract (Skill, Finding, SkillResult), artifacts, registry, orchestrator, schema, provenance
  interfaces/fs/   capability probe, foundationscale-train emitter, sbatch, fskills-rl driver, launcher
  skills/
    data_engine/   Phase 1 ops (ingest, clean, dedup, quality, decontam, format, tokenize, mix), pipeline,
                   recommender, mixture designer, readiness report, catalog, Phase-2 interfaces, SKILL.md
    training/      planner, CPT policy, post-training selection, method, estimator, feasibility,
                   recipes, knowledge base (families/hardware/algorithms/recipes/rules), emit skill, SKILL.md
  agent/intake.py  what to ask the user when a load-bearing fact is missing
  schemas/         JSON-Schema for every artifact and knowledge file
tests/             unit, contract conformance (every skill), reference scenario, real-data regressions
docs/              ARCHITECTURE, RECIPES, WALKTHROUGH (reference scenario), VALIDATION (GB200 evidence)
```

## Status (2026-09-26)

- **Tests:** 419, with 0 skips. Skips count as failures.
- **Validated on GB200** (details and numbers in `docs/VALIDATION.md`):
  - real Data Engine runs over three real corpora;
  - FoundationScale CPT and LoRA-SFT runs launched from emitted commands (the SFT run by `fskills launch` itself), with FoundationScale's save gates passing and the commits recorded;
  - memory estimates within 15% of measured values;
  - `fskills-rl` running Dr.GRPO.
- **Honest limits of the installed FoundationScale** (measured; the skills report them rather than hide them):
  - RL rewards only single-letter multiple-choice gold and saves no checkpoint;
  - the preference family is not wired to RL;
  - SFT uses full-sequence loss;
  - PP/EP are refused;
  - there is no Megatron backend.
