---
name: fskills-compression
license: MIT
description: "Model compression and quantization of a HF checkpoint (v1 method: RTN weight-only quantize-dequantize, int4 or int8, group-wise). Use it to preflight a host for compression (vendored preflight.py; classifies FAIL, UNKN and PROBE_ERROR results), to plan a probe-first run that emits an fs_launch_spec running rtn_worker.py on one GPU (launched only with the plan hash the user types back), and to verify the measured perplexity ratio (harness A vs B instrument certification, no-op check) against a pre-registered budget into a compression_report. Also route here any GPTQ, AWQ, NVFP4, MXFP4, FP8, modelopt, llm-compressor or quantized-speedup request: this skill answers it with a deferral refusal (CP-IN-003). Do NOT use for serving a checkpoint, already-quantized ones included, or asking whether it fits the serving envelope (fskills-deployment), LoRA merges (fskills-merging), benchmark accuracy suites (fskills-evaluation) or fine-tuning (fskills-training)."
compatibility: "Needs the foundationskills package (fskills CLI). `check` runs preflight.py, vendored verbatim from the estate skill model-compression. `plan` probes a HF model dir with safetensors, a non-empty held-out text, an absent results path, and torch + transformers. The worker rtn_worker.py runs on ONE GPU, only via fskills hash + fskills launch behind the hash the operator types back; the skill never launches."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Plan an int4 RTN weight-only run of gemma-4-12B-it (group size 128) against the 1.3x ppl-ratio budget I pre-registered","Check first — will a real GPU execution run in this environment at all before I plan","Verify the run: is the instrument certified, did the quantization change anything, is the ratio within my budget","Verify says PASS but where is my GPU speedup? (v1 never measures tok/s, latency, memory or energy — CP-HO-005)"]'
---

# compression — FoundationSkills RTN weight-only quantize-dequantize (check, plan, launch by hash, verify)

## Purpose

Compress a HF checkpoint with v1's one quantization method — RTN (round-to-nearest)
weight-only quantize-dequantize (int4 or int8), asymmetric min-max per group (group_size
default 128, any power of two in [32,1024]) — planned probe-first and judged against a
ppl-ratio budget the operator pre-registers. Embeddings, lm_head and multimodal towers
(module names containing embed, lm_head, vision, audio) stay bf16. `check` runs
`preflight.py`, vendored VERBATIM from the operator's estate skill `model-compression`
(FORGE campaign; schema `forge.preflight.v1`, 17 named checks: real GPU execution, CUDA/driver,
toolchain split, hardware-format gates, memory feasibility). `plan` requires a pre-registered
ppl-ratio budget (compressed_ppl/base_ppl) in (1.0, 2.0], runs `check`, probes the run (model
dir + safetensors, held-out text, bits, group size, a results path that is still absent —
never overwritten, torch/transformers importable, worker present) and emits an
`fs_launch_spec` (entry `fskills-compress`) running `rtn_worker.py` on ONE GPU. The skill
never launches and never echoes a hash — the operator reads it from `fskills hash` and types
it back to `fskills launch --confirm`. `verify` reads the plan's spec and its results and
writes a `compression_report`; only a PASS is a result within the budget.

## When to use

- To see what RTN weight-only quantization costs in held-out perplexity before shipping it ("int4, group 128 — is the ratio under 1.3x?").
- When quality must be judged against a budget you pre-registered (compressed_ppl/base_ppl in (1.0, 2.0]), not eyeballed.
- To plan the run as a launch spec you confirm yourself with the hash you read from `fskills hash`.
- Not for gptq/awq/nvfp4/mxfp4/fp8, modelopt/llm-compressor exports, kv-cache quantization, QAT/distillation recovery or serving speedups (CP-IN-003, deferred), folding a LoRA adapter (fskills-merging), serving a model (fskills-deployment) or scoring benchmarks (fskills-evaluation).

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `action` | yes | `check`, `plan` or `verify`; an unknown action is REFUSED (CP-IN-007) |
| `model` | yes (`check`, `plan`) | path to the HF model directory (config + safetensors) |
| `artifact_root` | yes (`check`, `plan`) | directory where `preflight.py` writes `cap.json` |
| `text_file` | yes (`plan`) | held-out `.txt`/`.jsonl` for perplexity (must be non-empty) |
| `bits` | no | `4` (default) or `8`, RTN weight-only; any other `format` than `rtn` is REFUSED (CP-IN-003) |
| `group_size` | no | quantization group size (default 128; must be a power of two in [32,1024]) |
| `budget_ppl_ratio` | yes (`plan`) | pre-registered ppl ratio (compressed_ppl/base_ppl) in (1.0, 2.0] (CP-IN-004) |
| `results` | yes (`plan`) | path `rtn_worker.py` writes its results to (must be absent — never overwritten) |
| `python` | no | interpreter that will run the worker (default: the current one) |
| `spec` | yes (`verify`) | path to the `fs_launch_spec` written by `plan` |

## Outputs

`check` → the preflight verdict (17 named checks). `plan` → `fs_launch_spec.compress.json`
(atomically, with provenance): a `compress` block (`model`, `text_file`, `bits`, `group_size`,
`results`, `budget_ppl_ratio`, the rtn-worker argv) plus `next` steps. `verify` → a `compression_report`
artifact (`budget_ppl_ratio`, `base_ppl`, `compressed_ppl`, `ppl_ratio`, `changed_fraction`, effective
bits incl. scales and zero points, model-config and text hashes, GPU name/uuid, host,
versions), always carrying the non-blocking CP-HO-005 finding.

Exit codes: 0 PASS (plan; verify only when the instrument is certified, the quantization
changed something and ppl_ratio is within budget), 5 RED (CP-HO-002/CP-HO-006/CP-HO-011),
95 UNMEASURED (CP-HO-001/CP-HO-003/CP-HO-010), 96 REFUSED (CP-IN-*; nothing written).
Skips are failures, and unmeasured is never PASS.

## Running it on one GPU

Compression is a local check/plan + verify; the measurement itself runs the worker on one
GPU through the universal launch gate:


1. `fskills compress plan --model <dir> --artifact-root <dir> --text-file <held-out.txt> --bits 4 --group-size 128
   --results <results.json> --budget-ppl-ratio <pre-registered, in (1.0, 2.0]>`: without a valid budget it is REFUSED
   (CP-IN-004) before anything runs, preflight included. Otherwise it runs `check` (preflight), probes the run and
   writes `fs_launch_spec.compress.json`. It never launches.
2. `fskills hash --spec <spec>`: **you** read the hash. Then
   `fskills launch --spec <spec> --confirm <the hash you read> --no-submit` runs `rtn_worker.py` on one GPU
   inside your allocation. Without the hash the launch is refused (96), and a refusal never echoes the hash.
3. `fskills compress verify --spec <spec>`: reads the worker's results and judges them against the plan's budget.
   Every measured verdict (PASS, RED, UNMEASURED) writes a `compression_report`; a REFUSED verify writes nothing.

Run the whole chain with `CUDA_VISIBLE_DEVICES=<the leased GPU UUID>`, directly or inside
`srun --overlap --jobid=<hold> --gres=gpu:<N>`; the worker records the GPU uuid it ran on.

## Scope

One HF checkpoint (Gemma-4-class included), one method — RTN weight-only quantize-dequantize (int4 or int8,
asymmetric min-max per group), measured on one GPU. Everything else is deferred and REFUSED CP-IN-003: gptq, awq,
nvfp4, mxfp4, fp8, modelopt, llm-compressor, kv-cache quantization, QAT/distillation recovery and serving speedups
need an isolated env install the operator has not approved (GPTQ/AWQ tooling pins transformers 4.x, which cannot
load Gemma-4-class models). Embeddings, lm_head and multimodal towers stay bf16. A LoRA merge belongs to
fskills-merging, serving to fskills-deployment, benchmarks/accuracy suites to fskills-evaluation, fine-tuning to
fskills-training.

## Validation rules

| Rule | Phase | Severity | When |
|---|---|---|---|
| CP-IN-001 | input | BLOCK | plan: the model directory is missing, is not a HF checkpoint, or the probes failed |
| CP-IN-002 | input | BLOCK | check: `preflight.py` exited 1 (FAIL): REFUSED |
| CP-IN-003 | input | BLOCK | a format outside v1 (GPTQ, AWQ, NVFP4, MXFP4, FP8, modelopt, llm-compressor, KV-cache quantization, QAT/distillation recovery, serving speedups) was requested: REFUSED, named as deferred |
| CP-IN-004 | input | BLOCK | plan or verify: no pre-registered ppl-ratio budget, or the budget is outside (1.0, 2.0]: REFUSED |
| CP-IN-005 | input | BLOCK | verify: the results are missing, or their schema is wrong: REFUSED |
| CP-IN-006 | input | BLOCK | verify: the results are not from this plan (model, bits, group_size or text_sha256 differ): REFUSED |
| CP-IN-007 | input | BLOCK | the requested action is not `check`, `plan` or `verify`: REFUSED |
| CP-HO-001 | handoff | WARN | verify: the instrument is not certified (base ppl A vs B disagree by more than 2% rel diff): UNMEASURED, never PASS |
| CP-HO-002 | handoff | BLOCK | verify: base ppl is not finite, or is <= 1: RED |
| CP-HO-003 | handoff | WARN | verify: the worker reported UNMEASURED (CPU fallback, crash): UNMEASURED, never PASS |
| CP-HO-005 | handoff | INFO | every verify result: efficiency (tok/s, latency, memory, energy) is never measured in v1 — a speed claim would be fake. Non-blocking, PASS included |
| CP-HO-006 | handoff | BLOCK | verify: the quantization was a no-op (nothing changed): RED |
| CP-HO-010 | handoff | WARN | check: `preflight.py` exited 2 (PROBE_ERROR), or any check is UNKN: UNMEASURED (the preflight itself exits 0 on UNKN; the skill does not) |
| CP-HO-011 | handoff | BLOCK | verify: ppl_ratio is above the budget: RED |

## Measurement (v1)

The worker measures perplexity on the held-out text with the base model measured fresh in the same run, twice:
harness A (manual next-token cross-entropy over non-overlapping ctx windows) and harness B (the model's own
`labels=` loss). The instrument is certified only if base ppl A vs B agree within 2% (rel diff), else CP-HO-001
UNMEASURED. It then quantizes and re-measures with harness A, and records hashes (model config, text), GPU
name/uuid, host, versions, `changed_fraction` and effective bits incl. scales and zero points.

- `ppl_ratio = compressed_ppl / base_ppl`; PASS needs the ratio within the pre-registered budget of (1.0, 2.0].
- Non-finite (or <= 1) base ppl is CP-HO-002 RED; a no-op quantization is CP-HO-006 RED; a ratio above budget is CP-HO-011 RED.
- **Nothing about speed is measured.** Quantize-dequantize keeps bf16 weights, so tok/s, latency, memory and energy
  stay unmeasured (CP-HO-005, estate wall W3) — a speed claim would be fake. Disk-size savings are not realized
  either: effective bits are a computed figure, not a shipped file.

## Failure handling

- Missing or wrong inputs: REFUSED (96) with CP-IN-001/CP-IN-004/CP-IN-007, nothing written and nothing run (a missing
  budget refuses before preflight). Fix the named input and re-plan; never pick a budget on the user's behalf.
- A format outside v1: REFUSED (96) with CP-IN-003, which names the format as deferred to an isolated env install the
  operator has not approved.
- `preflight.py` FAIL: REFUSED (96) with CP-IN-002; PROBE_ERROR or any check UNKN: UNMEASURED (95) with CP-HO-010.
- `verify` with missing or wrong-schema results (CP-IN-005), or results from another plan (CP-IN-006): REFUSED (96).
- Worker fell back to CPU or crashed: UNMEASURED (95) with CP-HO-003. Instrument not certified: UNMEASURED (95) with
  CP-HO-001. Neither is ever PASS.
- No-op quantization, non-finite base ppl, or a ratio above the budget: RED (5) with CP-HO-006/CP-HO-002/CP-HO-011.
  Re-plan at int8 or a smaller group size, and measure against a budget registered before that run.

## FS interface

- Preflight: `foundationskills/skills/compression/preflight.py`, vendored VERBATIM from the operator's estate skill
  `model-compression` (FORGE campaign, schema `forge.preflight.v1`, 17 named checks), run by `fskills compress check`.
- Worker: `rtn_worker.py` (RTN quantize-dequantize plus dual-harness perplexity), run through `fskills launch` behind
  the confirmation hash.
- Spec: `fs_launch_spec.compress.json` (entry `fskills-compress`), built by `fskills compress plan`.
- Neighbouring skills, route there instead: `fskills-merging` (LoRA merge), `fskills-deployment` (serve a model with
  vLLM), `fskills-evaluation` (benchmarks/accuracy suites), `fskills-training` (fine-tuning).

## Worked examples

Measured on GB200 (A4.8, gemma-4-E4B-it, wikitext-2-raw test, 65,504 tokens, budgets registered before the run):

- **RED:** `fskills compress plan --model <gemma-4-E4B-it> --artifact-root runs/a48/pf --text-file wikitext2.txt
  --bits 4 --group-size 128 --results runs/a48/int4.json --budget-ppl-ratio 1.10`, then hash, launch and verify. The
  instrument is certified (harness A vs B rel diff 2.6e-8), 343 Linear layers are quantized (98.8% of weights
  changed), and ppl goes from 64.71 to 126.50: ratio 1.955 > 1.10, RED with CP-HO-011. Plain RTN int4 roughly
  doubles this model's perplexity.
- **PASS:** the same chain at `--bits 8 --budget-ppl-ratio 1.01`: ratio 0.993, PASS, with CP-HO-005 attached (no
  efficiency claim). The int8 control shows the harness is sound, so the int4 RED is the model's real cost.
