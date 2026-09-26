# Validation record (GB200, 2026-09-25 → 2026-09-26)

Scope: FoundationSkills against FoundationScale `origin/main` 77bfa65 (plus this branch). The runs used GB200 hold 2495 on r04dgx06, attached with `srun --overlap`, with the holds left intact. Model: Gemma-4 E4B (base and `-it`). The environment was `envs/bench` (torch 2.12, transformers 5.13).

**Validated means:** the skill produced it, FoundationScale ran it, and FoundationScale's own gates passed. It is **not** a claim about benchmark quality.

## 1. Test suite

420 tests, **0 skips** (skips count as failures). They cover:
- unit tests per module;
- contract conformance for every registered skill: artifact types, unique rule ids, MUST_FIRE coverage of every rule, and SKILL.md sections;
- the reference scenario;
- regressions pinned to each real-run defect below;
- deep-probe tests that run FoundationScale dry-runs.

## 2. Data Engine on real corpora (all readiness verdicts PASS)

| Dataset | Source | In → out | Measured highlights |
|---|---|---|---|
| CPT | FoundationScale `docs/` (41 documents) as "internal documents" | 41 → 96 chunks, 295k tokens (Gemma-4 tokenizer) | 26 long documents chunked; truncation 0% (was 63%); remaining PII 0 |
| SFT (reasoning) | FreedomIntelligence/medical-r1-distill-data (22,000) | 22,000 → 20,289 | traces rendered into Gemma-4's thinking channel, 22,000/22,000 **verified**; 1,711 over-length examples dropped and counted; PII false positives 0 (were 73) |
| RL (MCQ) | allenai/ai2_arc ARC-Challenge (1,119) | 1,119 → 1,118 (1 exact duplicate) | each record parsed by FoundationScale's own `_parse_record` with a gold letter |

## 3. FoundationScale runs from FoundationSkills output

| Run | Launched by | Result | Measured |
|---|---|---|---|
| Gemma-4 E4B CPT, full, FSDP×4, seq 4096, 20 steps | emitted command (by hand, before the F12 env was automated) | **PASS**; save gates 4/4; final checkpoint; manifest `done` | loss 3.56 → ~3.1; 48.3 GB/GPU peak allocated; 69.7 model-TF/s/GPU (4.6% MFU) |
| Gemma-4 E4B-it LoRA SFT, DDP×4, 20 steps | emitted command (by hand) | **PASS**; adapter-only checkpoint (516/516 `lora_` tensors) | loss 1.78 → 1.19 |
| Gemma-4 E4B-it LoRA SFT, FSDP×4, b2, grad ckpt, 30 steps | **`fskills launch` (plan → emit → confirm hash → dry-run → run)** | **PASS**; commit `bf4f8fe` recorded by FoundationScale | **37.7 GB/GPU vs 32.4 GB planned (−14%)**; MFU 1.5% *measured by FoundationScale* (device peak passed by the emitter) |
| Recipe `gemma4-e4b-general-chat-sft-lora`, first attempt | `fskills launch` | **REFUSED 96** by transformers: `optimizer adamw` is not a valid name. Recovered through torchrun's exit 1 | defect F20 below; fixed |
| Recipe `gemma4-e4b-general-chat-sft-lora` (exact recipe config: FSDP×4, b4, ga8, adamw_torch_fused, 30 steps) | **`fskills launch`** | **PASS**; save gates 4/4; run `recipe-sft-3df648c2`, commit `f4643a7` | 75.9 GB/GPU; 55.0 model-TF/s/GPU (MFU 3.7% measured by FS). **Recipe promoted to `validated`** (scope: execution) |
| `fskills-rl` Dr.GRPO, E4B base | driver | **REFUSED 96**: base checkpoint has no chat template | expected; this is why stage chaining exists (F4) |
| `fskills-rl` Dr.GRPO, E4B-it, ARC-Easy MCQ, 6 steps | driver | **UNMEASURED 95**: rewards saturated on 5 of 6 steps; no checkpoint (FoundationScale saves none) | the RL loop runs on GB200: generation, scoring, advantages |

## 4. Estimator calibration (Gemma-4 E4B on GB200)

| Configuration | Estimate | Measured | Note |
|---|---|---|---|
| full FT, FSDP×4, b1, grad ckpt | 54.2 GB/GPU | 48.3 GB | +12% (38.1 GB before the logits term, F14) |
| LoRA, FSDP×4, b2, grad ckpt | 41 GB/GPU | 37.7 GB | +9% |
| LoRA, FSDP×4, b4, grad ckpt | 75.8 GB/GPU | 75.9 GB | 0% |
| full FT, FSDP b1 grad ckpt, throughput | 2,882 tokens/s/GPU | ~2,900 (from FoundationScale's 69.7 TF/s) | measured MFU point 4.6% |

The logits term (14 bytes per s·b·V element) was fitted to these three points and is pinned by a test. The hardware profile stores **measured MFU points by configuration**: DDP b2 31.8%, FSDP b1 grad ckpt 4.6%, LoRA FSDP b2 1.5%. The estimator uses the nearest point and flags any extrapolation as `derived`.

## 5. Defects found only by real runs (all fixed; each has a regression test)

| ID | Defect | Found by |
|---|---|---|
| F1 | `learning_rate` and `max_sequence_length` silently dropped at emit, so FoundationScale would train on its defaults | real plan |
| F2 | FoundationScale records provenance from the working directory; launches now `cd` to the FoundationScale repo root | FS dry-run |
| F3 | HTML stripper deleted `<think>`; credit-card PII destroyed binary strings and decimals; remaining PII unmeasured; Gemma-4's template silently dropped reasoning traces | real data |
| F4 | every stage got the base model; stages now chain, and LoRA or RL outputs name the missing merge or checkpoint | RL on GB200 |
| F5 | FoundationScale loaders parse every `.json` in a dataset directory; `stats.json` broke RLTrainer; shards moved to `shards/` | RL on GB200 |
| F6 | FoundationScale `*Refusal` exceptions reported as RED | RL on GB200 |
| F7 | FoundationScale RL rewards only single-letter multiple-choice gold; the Data Engine now builds MCQ records and drops free-form gold with a named reason | RL on GB200 |
| F8/F9 | per-stage data sizing (`use_for`); RL generation cost factor | real plan |
| F10 | SFT over-length examples: `drop_overlong` | real data |
| F11 | boolean flags take values (`--gradient-checkpointing true`); values are now validated against the parser's choices | CPT on GB200 |
| F12 | GB200 NCCL environment and CPUs per task (the fabric probe refused 96 without them) | CPT on GB200 |
| F13 | FoundationScale reports MFU as UNMEASURED without `FS_DEVICE_PEAK_TFLOPS`; the emitter now passes it | CPT on GB200 |
| F14/F15 | estimator missing the logits term; MFU depends on configuration | CPT on GB200 |
| F16 | RLTrainer never saves the policy | RL on GB200 |
| F17 | ARC-Easy saturates E4B-it rewards | RL on GB200 |
| F18 | recipe `fs.args` never reached FoundationScale | recipe emit |
| F19 | torchrun collapses exit codes and the launcher discarded output; the log is now kept and the FoundationScale verdict recovered | E2E launch |
| F20 | recipe `optimizer: adamw` refused by transformers; optimizer and scheduler vocabularies are now measured | E2E launch |
| — | recommender sent op config keys the ops ignored, so quality measured nothing on 23k records; op configs are now validated | real data |
| — | the plan estimated grad ckpt and batch it never emitted (63 GB vs 32 GB); the emitted config is now the estimated one; missing sequence length refused (FoundationScale's default is 128) | E2E launch |
| — | `--adapter-target` emitted per character, and later only the last module | reference scenario |

## 6. Measured FoundationScale core gaps (for the core workstream)

1. RL checkpoint persistence: `RLTrainer.run()` keeps the policy in a local variable.
2. RL reward beyond single-letter multiple-choice: free-form, numeric and `\boxed{}` answers.
3. The preference family (DPO/IPO/KTO/ORPO/SimPO/CPO) and the online family are not wired to RLTrainer.
4. Multi-GPU RL.
5. A reference-policy path (grpo/ppo with `kl_weight ≠ 0`).
6. Assistant-only loss masking for SFT.
7. PP/EP execution; no Megatron backend.
8. A dedicated pretrain/CPT objective, and a minimum-LR floor for cosine schedules.

The skills report each of these as `missing: ...` in plans and specs. They are never silently skipped.
