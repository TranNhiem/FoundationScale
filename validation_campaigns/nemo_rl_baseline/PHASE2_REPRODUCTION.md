# PHASE2_REPRODUCTION.md — NeMo-RL Phase 2 reproduction record

> Four NeMo-RL recipes — SFT, DPO, Reward Model, and GRPO — were run for 10 steps
> each on a single GB200 GPU (one reserved tray), using the DTensor policy worker
> with `_v2 = false`. All four exited `rc = 0` and wrote populated
> `training_info.json` artifacts. This is a correctness smoke result: the claim
> is "the training loops run and the loss moves on this hardware", nothing
> larger. A fifth arm using the `_v2` DTensor worker failed to build and is
> unmeasured.

## 1. What was reproduced

Four recipes from an operator-supplied NeMo-RL checkout at `<estate home>/NeMo-RL`
(the estate path itself is not published — see `PUBLISH_SET.txt` in the H100
campaign for why estate identifiers are kept out of this repository):

| Recipe | Model | Steps | Exit code |
|---|---|---|---|
| SFT | `Qwen2.5-0.5B` | 10 | `rc = 0` |
| DPO | `Qwen2.5-0.5B-Instruct` | 10 | `rc = 0` |
| Reward Model (Bradley-Terry) | `Qwen2.5-0.5B-Instruct` | 10 | `rc = 0` |
| GRPO | `Qwen2.5-1.5B` | 10 | `rc = 0` |

These four were chosen because they cover the four training-loop shapes the
campaign needs before any FoundationScale comparison: supervised fine-tuning,
preference optimization, reward-model training, and RL with generation. Each
ran 10 steps. Ten steps is enough to show the loop executes end to end — data
loads, loss computes, optimizer steps, checkpoint writes — and nothing more.

## 2. Environment and configuration

| Item | Value |
|---|---|
| Cluster / tray | GB200, one reserved tray (identifier not published) |
| Nodes x GPUs | 1 node x 1 GB200 GPU |
| `accelerator_type` reported by the run | `GB200` |
| NVML total memory reported | 189471 MB |
| NeMo-RL checkout | `<estate home>/NeMo-RL` (operator-supplied) |
| Policy worker | DTensor (`_v2 = false`) |
| Megatron backend | `megatron_cfg.enabled = false` (not exercised) |
| Steps per recipe | 10 |
| Exit code, all four recipes | `rc = 0` |

## 3. Results

### 3.1 SFT, `Qwen2.5-0.5B`

| Metric | Step 1 | Step 10 | Source |
|---|---|---|---|
| Loss | 1.5612 | 0.4643 | step log lines |
| Training FLOPS | 3.06 TFLOPS | 3.60 TFLOPS | step log lines |
| Validation loss | 2.0275 (first) | 0.3444 (last) | validation block |
| MFU across run | 0.12%–0.17% | — | step log lines |

Global batch size 32, micro-batch 1. The artifact
`results/sft/step_10/training_info.json` (134 bytes) reads
`{"epoch": 0, "step": 10, "total_steps": 10, "consumed_samples": 320, "total_valid_tokens": 1875.0, "val:val_loss": 0.3444017469882965}`.
The loss fell from 1.5612 to 0.4643 over 10 steps, and the artifact's
`val:val_loss` agrees with the last printed validation loss. The MFU figures
are smoke-run numbers and are not a performance claim (see Section 6).

### 3.2 DPO, `Qwen2.5-0.5B-Instruct`

| Metric | Step 1 | Step 10 | Source |
|---|---|---|---|
| Loss | 0.6931 | 0.6823 | step log lines |
| `sft_loss` | 0.0000 | 0.0000 | step log lines |
| `accuracy` | 0.0000 | 0.0000 | step log lines |

Global batch size 128, micro-batch 2; bf16; learning rate `5e-06`. Wall clock
14:13:24 to 14:17:27 (+08:00), about 4 minutes. The artifact
`results/dpo/step_10/training_info.json` (101 bytes) reads
`{"epoch": 0, "step": 10, "total_steps": 10, "consumed_samples": 1280, "total_valid_tokens": 487254.0}`.
The DPO loss moved down slightly. `sft_loss` and `accuracy` read exactly
`0.0000` for the entire run; this is unexplained and is listed in Section 6.

### 3.3 Reward Model (Bradley-Terry), `Qwen2.5-0.5B-Instruct`

| Metric | Step 1 | Step 10 | Source |
|---|---|---|---|
| `loss` | 0.7198 | 0.6515 | step log lines |
| `accuracy` | 0.5781 | 0.5859 | step log lines |
| `rewards_chosen_mean` | 4.4013 | 5.6790 | step log lines |

Wall clock 14:23:50 to 14:27:45 (+08:00), about 4 minutes. The artifact
`results/rm/step_10/training_info.json` (103 bytes) reads
`{"epoch": 0, "step": 10, "total_steps": 10, "consumed_samples": 1280, "total_valid_tokens": 12188672.0}`.
`score.weight` was missing from the checkpoint and was newly initialized for
`Qwen2ForSequenceClassification`; the classification head is randomly
initialized, which is why step-1 accuracy sits near chance. Loss fell and
accuracy rose slightly over the 10 steps.

### 3.4 GRPO, `Qwen2.5-1.5B`

| Metric | Step 1 | Step 10 | Source |
|---|---|---|---|
| Loss | 0.0481 | 0.0519 | step log lines |
| Generation KL Error | 0.0007 | — | step-1 log line |
| Avg Reward | 0.0742 | 0.0859 | step log lines |
| Mean Generation Length | 325.3184 | — | step-1 log line |
| Validation accuracy (256 samples) | — | 0.1133 | step-10 validation block |
| Validation avg response length | — | 324.4 | step-10 validation block |

vLLM generation reported `KV cache usage: 0.6%` and `Prefix cache hit rate:
81.3%`. The artifact `results/grpo/step_10/training_info.json` (250 bytes)
reads
`{"consumed_samples": 320, "current_step": 10, "current_epoch": 0, "total_steps": 10, "total_valid_tokens": 1652292.0, "val_reward": 0.11328125, "trainer_version": null, "sampler_name": null, "sampler_dispatch_index": null, "val:accuracy": 0.11328125}`.
The artifact and the log agree independently: `val:accuracy 0.11328125` rounds
to the `0.1133` printed by the validation block. Average reward rose from
0.0742 to 0.0859 over the 10 steps.

## 4. The `_v2` arm did not build

A parallel GRPO arm using the `_v2` DTensor worker (`DTensorPolicyWorkerV2`)
failed with `rc = 1`. It died in `create_local_venv`: `uv run --locked --extra
automodel ...` exited 1.

This independently reproduces existing finding #285: the `automodel` extra
drags in DeepEP, which fails the CCCL guard and pins `sm_90`, so it cannot
build on GB200's `sm_100`. This is a finding about the system — the `_v2`
worker is unavailable on this hardware — not about the instrument.

Consequence: the four green recipes in Section 3 all ran on the non-`_v2`
DTensor worker. The `_v2` worker's behaviour on GB200 is unmeasured, not
"fine".

## 5. Two instrument findings

Both findings in this section are about the measuring, not about NeMo-RL. The
training logs and artifacts were correct in both cases; the readings of them
were wrong.

### 5.1 CJK corruption in the summariser's output

The first machine-generated digest of the GRPO run reported a
`steps_completed` field reading `nemo_rl_grpo_v2.log manual<6 CJK chars>  0
training steps`. The six CJK characters are elided rather than reproduced: this
document ships in the repository, and a record that carries the exact corrupted
byte sequence is a hit for any scanner whose denominator includes the record —
the scanner-self-hit class. The shape is what the finding needs, not the bytes.

Measured: the four source prompts contain 0 CJK characters, counted with a
`[\u4e00-\u9fff]` scan over each of the four raw prompt texts (sft 0, dpo 0,
rm 0, grpo 0) — written by codepoint, not as a literal range, for the same
reason the corrupted string above is elided. The corruption was therefore
introduced by the summariser, not present in the training logs.

The true value is 10 steps, established from three independent places: step
markers `1..10` all present in the log, the `tmp_step_10` checkpoint
directory, and `total_steps: 10` in `training_info.json`.

The word `manual` does occur in the source, in the unrelated sentence "you may
need to set this variable manually". The corrupted string is a splice of real
text and invented text, which is the failure mode that is hardest to catch by
eye.

### 5.2 The "0 lines" reading was a `wc -l` artifact

An earlier note recorded that all four `training_info.json` files were "0
lines", which reads as "the runs wrote nothing". Measured: all four files
exist under `results/<recipe>/step_10/`, are 101–250 bytes, and are fully
populated (contents quoted in Section 3).

`wc -l` counts newlines. These are single-line JSON documents with no trailing
newline, so `wc -l` correctly reports 0 and the reading of that 0 was wrong.
This is the same class as the campaign's recurring lesson: an absent count is
not an absent thing.

## 6. What this does NOT measure

* Scaling: 1 GPU, 1 node. No multi-GPU, no multi-node, no scaling curve.
* Throughput: SFT MFU is 0.12%–0.17%. These are correctness smoke runs at 10
  steps; the TFLOPS figures are not a performance claim and must not be quoted
  as one.
* Convergence: 10 steps proves the loop runs and the loss moves. It proves
  nothing about final quality for any recipe.
* The Megatron backend path (`megatron_cfg.enabled = false`) was never
  entered.
* The `_v2` DTensor worker could not be built, so its behaviour is unmeasured
  on GB200, not "fine".
* DPO `accuracy` and `sft_loss` are both exactly `0.0000` for the whole run.
  This is still unexplained as to CAUSE — the run's config is on the cluster and
  has not been read back — so no conclusion should be drawn from the DPO
  accuracy number. What HAS been measured is what FoundationScale's objective
  gates do when handed the reading (PHASE3_DESIGN section 7 item 4,
  `tests/rl/test_dpo_anomaly_gate_response.py`): the `sft_loss` half is caught,
  but only if the objective declares `sft_loss` as a loss component, and the
  `accuracy` half is caught by nothing because the gate context has no channel
  for a diagnostic metric.
* No FoundationScale-side comparison exists yet. That is Phase 4, and nothing
  here licenses a claim about relative performance.

## 7. What Phase 3 needs from this

* Four recipe configurations are known to run end to end on GB200 with the
  non-`_v2` DTensor worker; Phase 3 can build longer runs on these four.
* The `_v2` worker is blocked by finding #285 (`automodel` extra -> DeepEP ->
  `sm_90` pin). Phase 3 must either avoid `_v2` or resolve #285 first.
* The DPO `accuracy`/`sft_loss` zero readings need investigation before any
  DPO quality claim is made.
* Digest tooling must be checked against source logs before its numbers are
  trusted; `wc -l` must not be used as an emptiness test on JSON artifacts.
