# Recipe library

A recipe is a starting configuration for one training stage (or a short sequence of stages). It is indexed by model family × size × architecture × stage × goal/domain × method × hardware. Recipes live in `foundationskills/skills/training/knowledge/recipes/<id>.yaml` and are validated against `schemas/knowledge/recipe.json`.

## Schema

```yaml
id: gemma4-e4b-general-chat-sft-lora     # must equal the file stem
version: 1.0.0
title: Gemma-4 E4B general-chat SFT (LoRA)
index:                                     # selection keys
  family: gemma4                           # must exist in knowledge/families/
  size_b: [3.0, 10.0]                      # parameter range, billions
  arch: dense                              # dense | moe
  stage: sft                               # pretrain | cpt | sft | preference | rl
  goal: general_chat                       # general_chat | reasoning | math | code | domain_expert | tool_calling
  domain: any                              # or manufacturing, semiconductor, ...
  method: lora                             # full | lora | qlora
  hardware: [gb200-189gb]
data:
  format: sft
  min_examples: 5000
  recommended_examples: 80000
  mixture: {general_chat: 0.8, instruction_mix: 0.2}
stages:
  - name: sft-chat-lora
    stage: sft
    algorithm: sft                         # an FS RL registry name for rl/preference; causal_lm for cpt
    hparams: {learning_rate: 1.0e-4, epochs: 2, lora_rank: 16, lora_alpha: 32}
    fs:
      entry: foundationscale-train         # or fskills-rl
      args:                                # FS CLI flags without "--"; values must be legal (see below)
        sharding-strategy: fsdp
        per-device-batch-size: 4
        gradient-accumulation-steps: 8
        optimizer: adamw_torch_fused
        max-sequence-length: 4096
hardware: {min_gpus: 1, gpu_mem_gb: 48, est_gpu_hours: 5.0}
evaluation: {benchmarks: [mt_bench], success: ">= 6.5 MT-Bench"}
risks: [...]
provenance:
  status: literature                       # validated | literature | community
  evidence: ["Tulu 3, arXiv:2411.15124"]
  validated_on: null                       # REQUIRED object when status == validated
```

## How a recipe is chosen

1. The planner first decides the **method** from goal, data and hardware rules. For example, CPT on a corpus uses full fine-tuning, a small SFT set uses LoRA, and single-GPU RL that doesn't fit uses LoRA.
2. `select_recipe` filters on the hard keys: stage, architecture and method. It then scores the soft keys: family +3, size range +2, goal +2, domain +1.5 (or +0.5 for `any`), hardware +1.
3. Unless family, size and goal all match, the match is `derived: true`, and the plan's `provenance_notes` say what to re-check.
4. The recipe's `hparams` **and** its `fs.args` are folded into the stage's hparams. The planner writes every value it estimated with (sequence length, batch, gradient checkpointing, sharding) into the stage. The emitted command is therefore exactly the configuration that was estimated.

## Rules enforced by tests

- The id equals the file stem and is unique. The family exists. Every stage algorithm has an algorithm card, and every rl/preference card names an algorithm FoundationScale registers.
- Every `fs.args` key is a real `foundationscale-train` flag. Every value is legal for the installed stack: the parser's `choices`, plus transformers' optimizer and scheduler names. A recipe that said `optimizer: adamw` was refused by transformers 5.13 on a real GB200 run, after the GPUs were allocated.
- `status: validated` requires `validated_on: {fs_commit, hardware, run_id, date}`, pointing at a real FoundationScale run.

## Provenance levels

| Status | Meaning |
|---|---|
| `validated` | Emitted by FoundationSkills and run on FoundationScale on the named hardware, with FoundationScale's save gates passing. **Scope:** it runs correctly and reproducibly. It does **not** mean benchmark quality was measured; that is the future benchmarking skill's job. |
| `literature` | Hyperparameters from papers or standard practice (cited). Not yet run on FoundationScale. |
| `community` | Widely used community settings. Not yet run on FoundationScale. |

## Adding a recipe

1. Copy the closest recipe and change `id`, `index`, `stages` and `provenance.evidence`.
2. Run `pytest tests/unit/training/test_knowledge_files.py tests/unit/test_validation_round2.py`. They check the schema, cross-references and FoundationScale flags and values.
3. To promote a recipe to `validated`:
   - plan and emit with it;
   - launch it with `fskills launch` on the target hardware;
   - check that `run_manifest.json` records the commit and that the save gates pass;
   - fill in `validated_on`.
