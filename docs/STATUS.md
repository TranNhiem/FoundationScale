# FoundationScale — status at the close of the build phase (2026-09-24)

This is the handoff between the build phase that ends here and the feature work that follows.
It states what the framework executes, what it refuses and why, what has been measured on
hardware and what has only been unit-tested, and where the next work should go. Every claim
below points at the evidence that backs it; nothing is estimated.

Machine-readable evidence for the hardware results below: `validation_campaigns/fsdp_6d_validation/results.json` and `validation_campaigns/gb200_scaling/results.json`.

## 1. The architecture, as it now stands

FoundationScale is two things with a deliberate boundary between them:

- **An adjudicator.** Gates, manifests, provenance and verification decide whether a training
  run and its checkpoints are what they claim to be. This is the part that is unique to this
  framework and it is the part that has been measured most.
- **A thin HF training plane** (`src/foundationscale/train/`) built on `transformers.Trainer`.
  It trains, records every declaration, and refuses what it cannot execute rather than running
  something else under the declared label.

Six-dimensional parallelism at production scale does **not** live in the thin plane. It lives in
the Megatron-Bridge lane that already runs production training on this estate, and
FoundationScale's job there is to adjudicate the checkpoints Bridge produces. That division was
recorded in the Bridge leverage plan (2026-09-20) and confirmed by the official 26B run below.
New axis machinery should not be built into the thin trainer to compete with it.

## 2. What the thin training plane executes

| Axis | State | Evidence |
|---|---|---|
| data parallelism (replication) | executes | the GB200 scaling campaign, 1→8 GPUs |
| `sharding_strategy=fsdp` | executes | **measured end to end on one tray**: gemma-4-26B-A4B, 60 steps, save gate PASS 4/4 at step 50, final adjudication clear, exit 0, train loss 1.308 (#535, #536, #537). Not yet run multi-node |
| `cpu_optimizer_offload` | executes under fsdp only, **since #538** | before #538 it was a silent no-op (a key transformers ignores; on and off arms reached a byte-identical peak). Measured after the fix on one tray: GPU ~181 → ~106 GiB per GPU, 2–2.6× step time; the first full-state-dict save exhausts host memory, so no offloaded 26B run has completed |
| `--tp` | trains; **refused (96) wherever it cannot also save** | measured on the 26B (#539, #540): tp=2 trained 10 steps (loss 1.674, 6.35 s/step), then the first save deadlocked, because transformers' Trainer saves on one rank and a tensor-parallel save gathers with a collective. tp without fsdp now refuses; tp with fsdp needs FSDP version 2, which a tied model cannot use, so it refuses there too. A tp degree that does not divide every head count refuses (tp=4 against 2 global KV heads died in the first forward). On an untied model (Qwen2.5-7B, tp=2 dp=2 fsdp, #541) the load now lands on accelerate's own mesh and ten steps trained (5.3 s/step), then the first save failed the same way: the writing rank gathers tp shards alone. That save path is transformers' own; the installed release cannot checkpoint a tensor-parallel model under any sharding, so tp refuses wherever that release is installed, detected by the reworked save API rather than a version. Net: no tp layout runs end to end in this plane on this toolchain |
| `--cp` | executes via accelerate `ParallelismConfig`; batches padded to a multiple of 2×cp | **measured end to end** on Qwen2.5-7B, cp=2 dp=2 fsdp (#542): mesh matched, 12 steps, loss 1.757, 13.9 s/step, save gate PASS at both checkpoints, exit 0. Before #542 the first step died on a bare assertion: context parallelism splits each sequence into 2×cp chunks. cp with an image column refuses (its collator cannot pad to that multiple). Gemma-4 cannot use it here: the tie needs FSDP version 1 |
| gradient checkpointing, torch.compile (3 axes), dataloader axes | execute | #532, #534, scaling campaign |
| `--pp`, `--ep` | **refused (96)** | `ParallelismConfig` has no pipeline or expert field — a backend fact, not an omission |
| `sharding_strategy=zero3` | **refused (96)** | ZeRO/DeepSpeed adjudicated and unbuilt |
| tp/cp beside a *replicated* data dimension | **refused (96)** | accelerate composes tp/cp with sharded data parallelism only |

Two FSDP behaviours were found by running a real 26B checkpoint and are now encoded, not
assumed: the wrap class is derived from the model actually loaded (an omni checkpoint declares a
`Gemma4AudioLayer` its causal-LM build never instantiates), and a model with tied embeddings
selects FSDP version 1 while keeping per-layer wrapping (version 2 rejects the shared tensor;
a single group OOMs on one flat buffer the size of the model).

## 3. What the adjudicator can and cannot establish on a Megatron checkpoint

Measured on `iter_0000750` of the official run (a real 26B-A4B Bridge checkpoint):

- **Expert layout** — recognised. All 60 stacked expert tensors carry the declared 128 experts
  in distinct storage. A Megatron-Bridge router nested at `mlp.experts.router.*` is no longer
  mistaken for an expert tensor (this is what made every gate fail closed before).
- **Expert distinctness inside a stacked tensor** — now **settled by the gate itself** (#537).
  Metadata cannot see aliasing inside one tensor, so when the gate can reach the data it hashes
  every expert slice, one slice at a time, on DCP and safetensors alike; without the data it
  abstains as before. Measured: all 7,680 slices distinct on the official run's checkpoint, on
  the FSDP run's checkpoints, and on the tensor/pipeline/expert-tensor-sharded Megatron
  checkpoints of the 6D runs; a planted duplicate is caught and named. Cost: ~3.5 minutes to
  hash the 26B's 91 GB of fp32 expert weights. Known limit: identical experts are legitimate
  right after sparse upcycling, and this check would fail that first save.
- **Expert byte volume** — skips: no run manifest declares the expected volume.
- **Checkpoint completeness (`save_complete`)** — **real since #543.** `tools/bridge_fqn_map.py`
  builds the model definition on the meta device (no checkpoint in the loop) and emits the
  declared tensor set for `--fqn-map`. Measured: 928 declared tensors; CLEAR 928/928 on the
  TP2/EP2/ETP2 and TP2/PP2/ETP2 checkpoints and on the official run's iteration 1750; a map with
  one planted extra tensor BLOCKS (1 of 929 absent). Known limit: the map shares Megatron's
  module code with the run, so it cannot catch a module the code never builds.

## 4. Megatron 6D: what trains, measured

Short training runs (20 steps each, 26B, one tray) through the Megatron-Bridge launcher, i.e.
real optimizer steps and a saved checkpoint, not just process-group setup:

| Layout | Result |
|---|---|
| TP=2, EP=2, ETP=2, DP=2 + sequence parallel | trains: loss 1.45 → 1.06, 123 s/step, checkpoint saved across tensor ranks |
| TP=2, PP=2, ETP=2 | trains: loss 1.43 → 1.16, 132 s/step, checkpoint saved across tensor AND pipeline ranks |
| EP=4 (the official run) | trains: all 2,467 steps completed; final checkpoint exported and evaluated (§5) |
| CP=2 | **does not run**: no TransformerEngine attention kernel covers Gemma-4's head_dim-512 full-attention layers with context parallelism (FlashAttention caps at 256; the unfused path cannot split the sequence; cuDNN has no kernel for the combination). A kernel limit, not configuration |

Pipeline parallelism has no native launcher knob and is passed as a recipe override; the
resolved config confirms it took effect. None of this runs through FoundationScale's own
trainer, whose `--pp`/`--ep` still refuse — six-dimensional execution is the Bridge lane's job.

## 5. The official run

gemma-4-26B-A4B full fine-tune on the v3 corpus through the Bridge lane (EP=4, one tray,
2,467 iterations). At iteration 750 it scores 279/356 on sfteval against 290/356 for the prior
run of the same recipe at the same iteration — paired exact McNemar p = 0.27, not a difference.
81 of 356 questions flip between two runs of one recipe at one iteration; that is the noise
floor, and a gap of that size is not a regression.

**Final, iteration 2467.** The run completed. Its HF export holds the expected 1,013 tensors
(60 stacked expert tensors, 51.6 GB), and all 7,680 expert slices are byte-distinct (a planted
duplicate is caught). On sfteval it scores 316/356 against 321/356 for the prior run's final
checkpoint: 13 questions only the official run gets right, 18 only the prior one, paired exact
McNemar p = 0.47. The Bridge lane reproduces the prior recipe's quality; it does not exceed it.

## 6. Known limits worth knowing before building on this

- **sfteval scores no think-mode rows.** 140 think rows, none with a parseable reference. Every
  number it produces describes non-thinking behaviour only.
- **No attention kernel with an sm_100 build** is installed in the benchmark environment; SDPA
  falls back to an Ampere-generation CUTLASS kernel and runs 1.68× slower than eager where both
  fit.
- **A step inside a held tray gets 2 CPUs**, which alone cost 3× throughput at world 4. Direct
  node login is the only measured escape.
- **Tensor parallelism in the thin plane has no end-to-end layout on this toolchain** (§2).
  It trains, on Gemma-4 and on an untied 7B, but the installed transformers cannot save a
  tensor-parallel model, so every tp layout is refused. Tensor parallelism belongs in the
  Bridge lane, which saves across tensor ranks (§4).
- **Offloaded 26B cannot save on one tray.** With offload engaged, host memory sits near
  912 GiB through training and the first full-state-dict save exhausts the node. Setting
  `cpu_ram_efficient_loading` did not move it; the source of the host footprint is open.
- **FSDP saves experts in fp32** under mixed precision (982 fp32 tensors to 32 bf16), so one
  26B checkpoint is ~414 GB. The byte-volume check only fails on a shortfall, so a 2x excess
  passes -- and its message calls that "matches", which overstates.
- **Context parallelism is unavailable for Gemma-4** on the installed TransformerEngine (§4).

## 7. Candidate next features, ranked

1. **Emit the fqn map from the Bridge launcher** beside every checkpoint, so `save_complete`
   runs without an operator step. The producer exists and is measured (#543); wiring it in
   makes FoundationScale the adjudicator of record for every production run.
2. **An in-distribution held-out set that scores thinking.** Replaces an instrument that cannot
   see half the behaviour it is used to judge.
3. **An offloaded save that fits in host memory** — offload, tp and cp were measured
   (#538–#542) and each surfaced a real defect; cp now runs end to end.
4. **An sm_100 attention kernel** (FlashAttention-3 or TransformerEngine) in the benchmark image —
   the single largest measured throughput gap, and the reason long context OOMs early.
5. **Multi-node FSDP** — the one FSDP claim not yet measured; needs two free trays.
