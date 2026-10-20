---
name: fskills-merging
license: MIT
description: "Folds a LoRA adapter into its base model as merged full weights: plans the merge probe-first and emits an fs_launch_spec running FoundationScale's merge_lora.py (launched only with the plan hash the user types back), then verifies the merged checkpoint structurally and by numeric logit parity against base + adapter before writing a checkpoint. Do NOT use to train adapters (fskills-training), score checkpoints on benchmarks (fskills-evaluation) or combine several full models (TIES/SLERP-style model soups are out of scope)."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI). Planning needs peft/transformers/torch importable, an FS checkout holding merge_lora.py, and enough free disk. Verifying (default) runs three forward passes over base, base + adapter and merged and needs a GPU; parity off verifies structure only. The skill never imports FoundationScale and never launches anything."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Fold the A4.4 LoRA adapter into gemma-4-12B-it as float32 full weights for a single-file deploy","Plan the merge first — I will read the spec hash myself and launch it on the cluster","Verify the merged checkpoint: does it really reproduce base + adapter on the logits?","Structural checks passed but answers got worse after merging — did bf16 storage drop the adapter updates?"]'
---

# merging — FoundationSkills LoRA adapter merge (plan, launch by hash, verify)

## Purpose

Fold a LoRA adapter into its base as merged full weights, planned probe-first and
verified twice. `plan` probes the merge (`interfaces/fs/emit_merge.py`: model dir,
adapter_config/weights, matching base, empty output, toolchain, disk) and emits an
`fs_launch_spec` that runs FoundationScale's own `merge_lora.py`; the skill never
launches and never echoes a hash — the user reads it from `fskills hash` and types it
back to `fskills launch --confirm`. `verify` then checks the merge structurally
(seven checks from the merge script) **and** by numeric logit parity (parity.py):
the merged checkpoint must reproduce base + adapter, because structure alone passes a
merge that lost the adapter's effect. Only structure + parity PASS writes a `checkpoint`.

## When to use

- To ship one model file instead of base + adapter ("fold the A4.4 adapter into gemma-4-12B-it").
- When a merge must be provably lossless: parity, not eyeballing ("does the merged copy reproduce base + adapter?").
- To plan a merge for a handoff/launch spec you will confirm yourself with the hash.
- Not for training adapters (fskills-training), scoring checkpoints (fskills-evaluation), or TIES/SLERP-style model soups.

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `action` | no | `plan` (default) or `verify` |
| `base` | yes (`plan`) | path to the base model directory |
| `adapter` | yes (`plan`) | path to the LoRA adapter (`adapter_config.json` + weights) |
| `out` | yes (`plan`) | output directory for merged full weights (must be absent or empty — never overwrites) |
| `store_dtype` | no | `float32` (default) or `bfloat16` storage of the merged weights (ME-HO-001) |
| `device_map` | no | device map the merge script loads the base with (default `cpu`) |
| `spec` | yes (`verify`) | path to the `fs_launch_spec` written by `plan` |
| `parity` | no | run numeric parity (default true); `false` → UNMEASURED, never PASS (ME-HO-004) |
| `device` | no | device parity.py runs on (default `cuda:0`) |
| `max_ratio` | no | parity ratio tolerance (default `0.1`) |
| `min_agreement` | no | argmax-agreement floor (default `0.95`) |

## Outputs

`plan` → `fs_launch_spec.merge.json` (atomically, with provenance): a `merge` block
(`base`, `adapter`, `out`, `store_dtype`, `device_map`, `results`, the merge-lora argv)
plus `next` steps. `verify` → `<out>.parity.json`, and only on PASS a `checkpoint`
artifact (`path`, `base_model`, `adapter`, `store_dtype`, `structural_checks`, `parity`).

Exit codes: 0 PASS (plan; verify only when all seven structural checks and parity pass),
5 RED (ME-HO-002/ME-HO-003), 95 UNMEASURED (ME-HO-004 — parity absent, not run, or too weak
to tell merged from base), 96 REFUSED (ME-IN-*; nothing written). A bf16 `plan` is still
PASS but carries the WARN ME-HO-001.

## Running it on GB200

Merging is a local plan + verify; the merge itself runs on the GPU host through the
universal launch gate:


1. `fskills merge plan --base <base> --adapter <run>/final --out <merged-dir> [--store-dtype float32] [--device-map cuda:0]`:
   probes the merge and writes `fs_launch_spec.merge.json`. It never launches.
2. `fskills hash --spec <spec>`: **you** read the hash. Then
   `fskills launch --spec <spec> --confirm <the hash you read> --no-submit` runs FoundationScale's `merge_lora.py`
   inside your allocation. Without the hash, the launch is refused (96).
3. `fskills merge verify --spec <spec> [--device cuda:0]`: reads the seven structural checks and runs numeric parity on
   the GPU. On PASS it writes the merged `checkpoint`.

Run step 2 and step 3 on the leased GPU: `CUDA_VISIBLE_DEVICES=<the leased GPU UUID>` for a direct run, or inside
`srun --overlap --jobid=<hold> --gres=gpu:N`.

## Scope

One LoRA or QLoRA adapter folded into the base it was trained on (LLM or VLM, any family FS's merge script loads:
gemma4 and qwen3_5 are measured), on GB200, H100 or local. Combining several full models (TIES, DARE, SLERP soups) is
out of scope. Training adapters belongs to fskills-training, and scoring the merged model to fskills-evaluation.

## Validation rules

| Rule | Phase | Severity | When |
|---|---|---|---|
| ME-IN-001 | input | BLOCK | plan: base is not a model directory, or the adapter has no `adapter_config.json`/weights |
| ME-IN-002 | input | BLOCK | plan: the adapter was trained on a different base than the one declared (its `base_model_name_or_path`) |
| ME-IN-003 | input | BLOCK | plan: the output directory exists and is not empty (a merge never overwrites) |
| ME-IN-004 | input | BLOCK | plan: the merge cannot run here (FS merge script not found, peft/transformers/torch not importable, or not enough free disk) |
| ME-IN-005 | input | BLOCK | verify: the merge results JSON is missing or unreadable (the merge was not launched) |
| ME-HO-001 | handoff | WARN | plan: bfloat16 storage requested; sub-precision LoRA updates are rounded away (measured) |
| ME-HO-002 | handoff | BLOCK | verify: a structural check of the merge script failed (parameter counts, LoRA weights left, processor): RED |
| ME-HO-003 | handoff | BLOCK | verify: numeric parity outside tolerance (merged does not reproduce base + adapter): RED |
| ME-HO-004 | handoff | WARN | verify: parity not run, or the adapter's effect too small to tell merged from base: UNMEASURED, never PASS |

## Precision (measured)

Measured on gemma-4-12B-it with the A4.4 16-frame LoRA adapter, on one GB200 GPU:
- **bfloat16 storage:** all seven structural checks PASS, yet parity gives ratio **0.237** and argmax agreement
  **0.915** (RED). Merging in fp32 and then storing bf16 is bit-identical, so the loss is bf16 storage: LoRA updates
  smaller than a weight's bf16 precision are rounded away.
- **float32 storage:** ratio **4e-5**, agreement **1.0** (PASS).
- **Inference in bf16:** any merged copy scores held-out NLL **0.9391**, against **0.9253** for base + adapter unmerged
  (base 1.4677). Merging loses 2.5% of the adapter's held-out gain, and serving base + adapter unmerged keeps all of
  it.
- **Negative control:** the plain base passed as "merged" gives ratio exactly 1.0 (RED).

Parity: `ratio = mean|merged − adapter| / mean|base − adapter|` over fixed prompts, in fp32 forwards. PASS needs
ratio ≤ 0.1 and argmax agreement ≥ 0.95. An adapter effect below 1e-3 is UNMEASURED.

## Failure handling

- Missing or wrong inputs: REFUSED (96) with ME-IN-001..004, and nothing is written. Fix the named input and re-plan into
  a fresh output directory.
- `verify` before the merge was launched: REFUSED (96) with ME-IN-005, which names the hash, then launch, then verify
  steps.
- A structural check failed: RED (5) with ME-HO-002. Read `<out>.merge_results.json` and the merge log.
- Parity outside tolerance: RED (5) with ME-HO-003. Store float32, or serve base + adapter unmerged.
- Parity not run, or the adapter's effect too small: UNMEASURED (95) with ME-HO-004, never PASS.

## FS interface

- Merge step: FoundationScale's `examples/vlm-2xH200/scripts/merge_lora.py` (PEFT `merge_and_unload`, safetensors, seven
  structural checks), run through `fskills launch` behind the confirmation hash.
- Spec builder: `foundationskills.interfaces.fs.emit_merge` (entry `fskills-merge`).
- Parity: `foundationskills/skills/merging/parity.py`.
- `training.emit` names this skill when a LoRA stage feeds a later stage.

## Worked examples

- **PASS:** `fskills merge plan --base gemma-4-12B-it --adapter runs/fskills-sft/final --out merged_fp32 --device-map
  cuda:0`, then hash and launch, then `fskills merge verify`. Seven of seven structural checks pass, parity ratio is
  4e-5 with agreement 1.0, and `checkpoint.merged.json` is written.
- **RED:** the same adapter stored as bfloat16. `verify` gives RED with ME-HO-003 (ratio 0.237, agreement 0.915),
  although every structural check passed.
