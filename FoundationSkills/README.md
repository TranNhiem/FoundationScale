# FoundationSkills

FoundationSkills is the agent layer on top of FoundationScale. An engineer states a goal ("a manufacturing-domain reasoning model from a 7B base, on our H100 cluster, keep general ability"). An agent then uses the skills in this package to do the following:

1. prepare the data;
2. choose the stages, algorithms and method;
3. estimate resources and check feasibility;
4. emit ready-to-run FoundationScale commands;
5. evaluate the trained checkpoint against its base;
6. run a budgeted, statistically gated research campaign over several trials.

The agent then launches them, but only after an explicit confirmation from the user.

| Skill | What it does | Guide |
|---|---|---|
| `data_engine` | raw sources → FS-ready dataset + readiness report (clean, dedup, quality, decontam, format, tokenize, LLM labelling) | `foundationskills/skills/data_engine/SKILL.md` |
| `training.planner` | goal → stages, algorithm, method, memory/time estimate, feasibility | `foundationskills/skills/training/SKILL.md` |
| `training.emit` | plan + dataset → confirmed `fs_launch_spec` + sbatch (SFT/CPT via `foundationscale-train`, RL via `fskills-rl`) | `foundationskills/skills/training/SKILL.md` |
| `evaluation` | checkpoint (adapter or full) vs base on the versioned eval policy, regression-banded | `foundationskills/skills/evaluation/SKILL.md` |
| `auto_research` | approved multi-trial campaign: launch envelope, seed-paired acceptance, multi-objective trade-offs, claim chain, append-only ledger | `foundationskills/skills/auto_research/SKILL.md` |

```
Foundation Agent        (Claude Code / any agent)  -> reads SKILLS.md, calls skills
FoundationSkills        decides WHAT: data_engine, training.planner/emit, evaluation, auto_research
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
                                             # add --no-submit to run inside an existing allocation
                                             # (e.g. under `srun --overlap --jobid=<hold>`)
fskills eval run  --run-manifest DIR/run_manifest.json --benchmarks arc_easy,arc_challenge \
                  --policy foundationskills/skills/evaluation/eval_policy.yaml --eval-cache ~/.cache/huggingface \
                  --out eval_report.json [--baseline-cache DIR]
fskills eval emit ... --spec-out spec.json   # the same eval as a Slurm job; then `eval hash` / `eval launch`
fskills recipes list --stage sft --goal reasoning
fskills datasets discover --goal math --stage rl
```

Every command prints a JSON result and exits 0/5/95/96. `auto_research` has no CLI subcommand yet; an
agent drives it through `AutoResearchSkill().execute(request, ctx)` (actions: envelope, submit, record, claim,
close; see its SKILL.md).

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
    evaluation/    lm-eval runner, eval policy, baseline cache, regression bands, report, SKILL.md
    auto_research/ campaign spec checks, launch envelope, acceptance statistics (single- and
                   multi-objective), claim chain, hash-chained ledger, SKILL.md
  agent/intake.py  what to ask the user when a load-bearing fact is missing
  schemas/         JSON-Schema for every artifact and knowledge file
tests/             unit, contract conformance (every skill), reference scenario, real-data regressions
docs/              ARCHITECTURE, RECIPES, WALKTHROUGH (reference scenario), VALIDATION (GB200 evidence)
```

## Status (2026-10-07)

- **Tests:** 1,528 passed with 0 skips in the full GB200 env (`~/envs/bench`, with `PYTHONPATH=../src:.`). In a
  bare interpreter, 3 tests skip because lm_eval, sentence-transformers and transformers are absent. Skips count as failures.
- **Verified on real workloads on GB200**, with gemma-4-E4B-it and every step driven through the skills' own commands.
  Evidence is in `artifacts/gpu-verification/` on the cluster; earlier validation numbers are in `docs/VALIDATION.md`.
  - `data_engine` ran on SFT-Taiwan-AIEC v3 (19,731 rows): PASS with 9.08M tokens, chain-of-thought traces kept,
    and 0 arc_easy decontamination hits. The ARC RL datasets also passed, and `llm_classify` ran through an
    OpenAI-compatible endpoint.
  - `training.planner` → `training.emit` → `fskills launch`:
    - a 4-GPU LoRA SFT, 308 steps, loss 3.64 → 1.16, with FS save gates clear;
    - a 556-step RL run on 1 GPU, with its checkpoint saved;
    - the time estimate is within about 10% of measured once epochs and padding are priced.
  - `evaluation` on the SFT adapter (from its run manifest), the RL full checkpoint, and the `eval emit`/`launch` path:
    all PASS against the base on arc_easy / arc_challenge.
  - `auto_research` ran a 9-trial, two-objective campaign (arc_easy accuracy and held-out NLL) on real
    evals. One candidate dominated and was claimed; the other was a measured trade-off, refused
    (AR-RS-008) and disclosed. The ledger hash chain verified. A training trial submitted through the skill trained on GPU.
- **Honest limits of the installed FoundationScale** (measured; the skills report them rather than hide them):
  - RL rewards only single-letter multiple-choice gold, so free-form corpora such as gsm8k are dropped by `data_engine`;
  - Gemma-4 E4B is reward-saturated on ARC: about 11% of RL steps carry signal at temperature 1.0;
  - Gemma-4 E4B cannot use gradient checkpointing (KV-shared layers), and the planner turns it off;
  - RL has no LoRA support, and the preference trainer saves no checkpoint;
  - SFT uses full-sequence loss;
  - PP/EP are refused;
  - there is no Megatron backend.
