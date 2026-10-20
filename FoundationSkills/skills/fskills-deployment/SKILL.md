---
name: fskills-deployment
license: MIT
description: "Serves a checkpoint (full or merged weights, or base + LoRA adapter unmerged) with the measured vLLM container: plans a GPU-confined serving fs_launch_spec probe-first (launched only with the plan hash the user types back), then verifies the running endpoint (served names, GPU placement, smoke completion) and records a serving_spec. Targets outside the measured serving envelope are UNMEASURED, never PASS. Do NOT use to merge adapters (fskills-merging), train (fskills-training) or benchmark checkpoints (fskills-evaluation)."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI), a leased GB200 GPU, and the vllm031 enroot container (vLLM 0.31.0, arm64). Planning probes a model dir, an optional adapter, the GPU UUID and the port and writes a spec; verifying needs the endpoint answering on this host (HTTP) and pgrep/nvidia-smi for GPU placement. The skill never starts the server and never echoes a hash — launching goes through `fskills launch --confirm`."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Serve the merged fp32 gemma-4-12B-it checkpoint on the leased GPU — plan the spec first, I read the hash myself","Serve gemma-4-12B-it plus the A4.4 LoRA adapter unmerged (--enable-lora) with two served names","Verify the running endpoint: are the served names up, on the leased GPU UUID, and answering?","Serve Nemotron-3-Nano-Omni — has that family ever been served here, or is that unmeasured?"]'
---

# deployment — FoundationSkills GPU-confined serving (plan, launch by hash, verify)

## Purpose

Serve a trained checkpoint with the measured vLLM container, confined to the leased GPU by
construction. `plan` probes the deployment (`interfaces/fs/emit_serve.py`: model dir, adapter
config + matching base, GPU UUID, port) and emits an `fs_launch_spec` that starts the `vllm031`
server on exactly one leased GPU UUID; nothing launches without the plan hash **you** read and
type back to `fskills launch --confirm`. `verify` then checks the running endpoint: `/v1/models`
is queried against HTTP, every server process is tied to its GPU UUID (`nvidia-smi`), and each
served name gets a smoke completion. A target outside the measured serving envelope — or a LoRA
deployment whose served fidelity is not measured — is UNMEASURED, never PASS.

## When to use

- To ship a trained checkpoint as an endpoint ("serve the A4.7 merged fp32 gemma-4-12B-it"), single or two served names.
- To serve base + LoRA unmerged (`--enable-lora`, optional vision-tower connector) without a merge.
- To prove where the server runs: "is the server actually confined to my leased GPU?" (placement is measured).
- Not for merging adapters (fskills-merging), training (fskills-training) or benchmarking (fskills-evaluation).

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `action` | no | `plan` (default) or `verify` |
| `model` | yes (`plan`) | path to the model directory (base, or merged full weights) |
| `adapter` | no | path to a LoRA adapter served unmerged (`--enable-lora`; needs `adapter_config.json`) |
| `gpu` | yes (`plan`) | the leased GPU UUID the server is confined to (never a device index) |
| `port` | no | server port (default 18841; must be free) |
| `served_name` | no | served model id (default `model`) |
| `modality` | no | `text` (default) or `vision` (adds `--enable-tower-connector-lora` for a vision stage) |
| `max_model_len` | no | context the server loads (default 16384) |
| `spec` | yes (`verify`) | path to the `fs_launch_spec` written by `plan` |

## Outputs

`plan` → `fs_launch_spec.serve.json` (atomically, with provenance): the `serve` block (model,
adapter, family, mode, modality, gpu, port, served_names, endpoint) plus `next` steps and the
measured envelope cell. `verify` → `serving_spec` (endpoint, served_names, family/mode/modality,
engine, envelope, checks, smoke texts, gpu, verdict).

Exit codes: 0 PASS (plan in a measured cell; verify only when names listed, smoke text and
placement all check out), 5 RED (DP-HO-003/DP-HO-004), 95 UNMEASURED (DP-HO-001/002), 96 REFUSED
(DP-IN-*; nothing written).

## Running it on GB200

Serving is a plan + verify; the server runs in the foreground of your launch, through the
universal launch gate (it must stay inside your Slurm step: on 2026-10-20 an enroot vLLM started
outside one ran ~9 min on another user's GPU, so the spec confines it to one leased UUID):

1. `fskills serve plan --model merged/gemma-4-12B-it --gpu <leased GPU UUID> --port 18841 [--adapter runs/sft/final] [--served-name <id>] [--modality text|vision] [--max-model-len 16384]`
   probes and writes `fs_launch_spec.serve.json`. It never launches.
2. `fskills hash --spec <spec>`, then `nohup fskills launch --spec <spec> --confirm <the hash you read> --no-submit &`
   (or the same inside `srun --overlap --jobid=<hold> --gres=gpu:1`). Without the hash the launch is refused (96).
3. `fskills serve verify --spec <spec>` on the server's host: names listed, GPU placement measured, one smoke
   completion per served name. It writes the `serving_spec` record.

## Scope

One checkpoint (full or merged weights, or one base + one LoRA adapter unmerged), served on one leased GB200 GPU by the
measured `vllm031` container. Families ever served here (gemma4_unified, qwen3_5; see the envelope) — every other family is
UNMEASURED. Merging belongs to fskills-merging, training to fskills-training, scoring to fskills-evaluation.

## Validation rules

| Rule | Phase | Severity | When |
|---|---|---|---|
| DP-IN-001 | input | BLOCK | plan: the model is not a model directory, or the adapter has no adapter_config.json |
| DP-IN-002 | input | BLOCK | plan: the adapter was trained on a different base than the served model |
| DP-IN-003 | input | BLOCK | plan: no leased GPU UUID given, or the port is already in use (the server must be confined to one leased GPU) |
| DP-IN-004 | input | BLOCK | verify: the spec is unreadable or the endpoint does not answer /v1/models |
| DP-HO-001 | handoff | WARN | the deployment target (engine x family x mode x modality) is outside the measured serving envelope — UNMEASURED, never PASS |
| DP-HO-002 | handoff | WARN | LoRA deployment: served-adapter fidelity is not measured yet — UNMEASURED even when healthy |
| DP-HO-003 | handoff | BLOCK | verify: a served name is missing from /v1/models or a smoke completion returned no text — RED |
| DP-HO-004 | handoff | BLOCK | verify: a server GPU process is on a GPU other than the leased UUID — RED |

## Measured serving envelope

`vllm-0.31.0-arm64` cells actually served on this estate (`interfaces/fs/serving_envelope.py`); outside it, UNMEASURED:

| Family | Mode | Modality | Fidelity | Evidence |
|---|---|---|---|---|
| qwen3_5 | full | text | unmeasured | B2.2 + B6.1: Qwen3.5-4B and Qwen3.8-27B served as pi agent endpoints on GB200 |
| gemma4_unified | full | text | unmeasured | A4.9 a49/FINDINGS.md: gemma-4-12B-it served on GB200, prompt logprobs returned |
| gemma4_unified | lora | text | unmeasured | A4.9 a49/FINDINGS.md: gemma-4-12B-it + A4.4 LoRA unmerged (`--enable-lora`, rank 16); LoRA applied (0.91 nats/token vs base), token-level fidelity inside bf16 noise |

Served-LoRA fidelity is not measured: token-level logprobs sit inside bf16 noise (floor 0.40, served 0.63,
a49/FINDINGS.md); a task-level held-out check is the follow-up.

## Failure handling

- Missing or wrong inputs: REFUSED (96) with DP-IN-001..003 naming the gap (no config.json, wrong base, no GPU UUID, port in
  use). Fix the input and re-plan; nothing is written.
- `verify` against a spec whose endpoint will not answer: REFUSED (96) with DP-IN-004 — launch the serving spec first.
- A served name missing or an empty smoke completion: RED (5) with DP-HO-003. Read the server log; re-launch.
- A server process on a GPU other than the leased UUID: RED (5) with DP-HO-004. Stop the server now and re-plan with the
  leased UUID — this is what happened on 2026-10-20 outside a step.
- Outside the measured envelope (or placement unmeasurable, e.g. verify off the server host): UNMEASURED (95) with
  DP-HO-001, never PASS. Unmerged LoRA: UNMEASURED with DP-HO-002 — serve an A4.7 float32-merged copy for a measured
  full-weights cell.

## FS interface

- Entry: `fskills serve` (`plan` / `verify`); emits `fs_launch_spec` and `serving_spec`.
- Spec builder: `foundationskills.interfaces.fs.emit_serve` (`emit_serve`, `model_family`, inside `serve_probes`, both
  proxied and injectable for tests).
- Envelope: `foundationskills.interfaces.fs.serving_envelope` (`lookup` over `MEASURED_SERVING`; widen only with a new
  measured row and its evidence).
- Launch: `fskills launch --confirm <the hash you read from fskills hash> --no-submit` — the same universal gate as every
  other spec.

## Worked examples

- **PASS:** `fskills serve plan --model merged/gemma-4-12B-it --gpu <leased GPU UUID> --port 18841`, hash, launch, then
  `fskills serve verify --spec <spec>`: both served names listed, smoke "OK", every server process on the leased UUID →
  `serving_spec` written with verdict PASS.
- **UNMEASURED (LoRA):** `--model models/gemma-4-12B-it --adapter runs/sft/final`: both ids answer and placement is right,
  → verify UNMEASURED with DP-HO-002 (served-LoRA fidelity not measured: 0.63 vs 0.40 bf16-noise floor).
- **UNMEASURED (family):** `--model models/nemotron-nano`: plan UNMEASURED with DP-HO-001 (the family was never served
  here); measure it first, then add a row to the envelope.
