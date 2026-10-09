# Speech P7: input-path profile and Canary-Qwen (SALM)

Hardware: one GB200 GPU. Date: 2026-10-07.

## 1. Where a Gemma-4 E4B audio step spends its time (`profile_step.py`)

| stage | measured |
|---|---|
| collate, batch 8 (decode + chat template + placeholder check) | 0.146 s, hidden by dataloader workers |
| GPU step, 16 different real batches (fwd + bwd + clip + AdamW, bf16) | 0.656 s avg; sequence length 425-491 |
| steady-state Trainer throughput | 1.55-1.60 it/s, about 1 / GPU step |

Training is compute-bound, and FoundationScale/Trainer adds about 0 per step. The earlier
0.99 steps/s came from 60-step runs, where the one-time warmup and the final save dominate.
Sequence lengths are nearly uniform on this data, so duration grouping has nothing to fix,
which matches the P6 no-gain result.

## 2. Canary-Qwen-2.5B (NeMo speechlm2 `SALM`) zero-shot, FoundationScale-scored

NeMo loads the model and generates through SALM's own chat API (`audio_locator_tag`,
"Transcribe the following:"). FoundationScale applies the same normaliser and corpus WER it
uses for every other model (`speech_canary/eval_wer_salm.py`). Audio that is not 16 kHz is
refused, never resampled.

| set (rows) | Canary-1B-flash base | Canary-1B-flash LS+AMI FT | **Canary-Qwen-2.5B base** |
|---|---|---|---|
| dev-clean (300) | 1.59 | - | **1.54** |
| VoxPopuli (300) | 5.17 | - | **5.51** |
| AMI (300) | 16.24 | 12.56 | **10.52** |
| Earnings-22 (148) | 23.39 | 26.86 | **19.52** |

WER in %. Hypotheses were spot-checked: no prompt echo and no tags. Environment: the
`make_nemo_env.sh` venv plus `peft`/`accelerate` installed with `--no-deps` (SALM imports peft);
the container torch is unchanged.

## 3. Canary-Qwen-2.5B fine-tune through NeMo, adjudicated by FoundationScale

`speech_canary/salm_finetune.py`: NeMo owns the loop (`SALM.from_pretrained`, its own
`SALMDataset`/`DataModule`/`configure_optimizers`, Lightning bf16-true). The released trainable
scope is kept as shipped: encoder + projector in full, LoRA r128 on the LLM's q/v_proj, LLM and
embeddings frozen. Data: the same LibriSpeech + AMI mix as the Canary-1B recipe (12,000 rows),
500 steps, batch 8, lr 5e-5, warmup 50, 3.1 it/s on one GPU.

`speech_canary/salm_adjudicate.py` (exit 0 = PASS):

| verdict | result |
|---|---|
| speech.audio_row_coverage | 12000/12000 rows, no refusals |
| tower_movement perception.encoder | 1041/1196 tensors moved |
| tower_movement perception.proj | 2/2 moved |
| tower_movement lora | 112/112 moved |
| salm.frozen_llm_unchanged | 310/310 frozen LLM/embedding tensors bit-identical |
| salm.param_names_unchanged | 1620/1620 identical keys |

| set (rows) | base | LS+AMI fine-tune |
|---|---|---|
| dev-clean (300) | 1.54 | 1.64 |
| VoxPopuli (300) | 5.51 | 5.27 |
| AMI (300) | 10.52 | 10.08 |
| Earnings-22 (148) | 19.52 | 20.32 |
| **dev-clean full (2683)** | **1.61** | **1.72** |
| **AMI test (2000)** | **11.66** | **11.55** |

The AMI drop on 300 rows did not survive a larger set. A paired bootstrap over the same
utterances (`paired_bootstrap.py`, 2000 resamples) gives:

- dev-clean full: +0.12 points, 95% CI [+0.04, +0.20]. A significant **regression**.
- AMI-2000: -0.11 points, 95% CI [-0.38, +0.17]. **Not significant.**

So this recipe does **not** improve Canary-Qwen. It already trained on this kind of data, and the
small fine-tune only costs a little in-domain accuracy. The engineering result stands: the lane
trains SALM through NeMo with its released scope, and FoundationScale adjudicates it. The
methodological lesson is that 300-row sets are too small to call shifts under about 1 point.
`build_devclean_full.py` and `speech_p5/build_ood.py` (`OOD_LIMIT=2000`, the extra AMI test
shards, disjoint from the AMI train split) build the larger sets.

Three NeMo 3.1 traps the lane now handles explicitly:
- **Silent audio drops.** `fault_tolerant_audio_loading` defaults to True, and `FallbackDataset`
  then swaps a failed batch for the previous one. The strict view crashes, because
  `AudioSamples(fault_tolerant=False)` returns 2 values and the collate unpacks 3. The script
  uses the strict policy (drops raise) on the 3-value loader.
- **cuDNN fused attention fails in backward** on this GB200 stack (cuDNN Frontend reshape
  error). It is disabled; flash/efficient SDPA compute the same attention.
- **A `validation_ds: null` key** crashes `DataModule`, so the key is omitted.

The adjudicator's first draft named a projector prefix that does not exist
(`perception.modality_adapter`). The movement gate returned VACUOUS (0 tensors) rather than
PASS, which is what surfaced the real name, `perception.proj`.

## 4. Earnings-domain training data: gated, not used

- `kensho/spgispeech` is gated: it requires accepting its terms with the user's credentials.
  That is the user's decision, so it was not done on autopilot.
- `revdotcom/earnings22` and `distil-whisper/earnings22` are the Earnings-22 evaluation data
  itself. Training on them would leak the test set.

## 5. Re-verifying the earlier HF-native results at scale

Section 3 showed that 300-row sets can manufacture a gain, so every earlier fine-tune claim was
re-scored on full dev-clean (2683 rows) and AMI-2000. Each comparison uses a paired bootstrap
over the same utterances. `speech_p4/eval_wer.py` now writes every row (`all`), which the
bootstrap needs, and `run_hf_bigeval.sh` spreads the jobs over 8 GPUs with an idle guard per
launch. The checkpoints are the ones the committed results came from, read from each result's
recorded `args`.

| model (claim) | dev-clean full: base -> FT, diff [95% CI] | AMI-2000: base -> FT, diff [95% CI] |
|---|---|---|
| Gemma-4 E4B full FT (P3) | 4.07 -> **2.71**, -1.36 [-1.54, -1.18] | 28.24 -> **24.50**, -3.74 [-4.72, -2.67] |
| Gemma-4 E4B LoRA (P3) | 4.07 -> **2.96**, -1.11 [-1.30, -0.93] | 28.24 -> **24.85**, -3.40 [-4.92, -1.32] |
| Gemma-4 LoRA vs full FT | +0.25 [+0.13, +0.37], full FT better | +0.35 [-1.32, +2.52], not significant |
| Whisper-large-v3 FT (P4/P5) | 2.25 -> **1.70**, -0.55 [-0.67, -0.43] | 20.15 -> 20.75, +0.60 [-0.49, +2.51], **not significant** |
| Parakeet-CTC-1.1B full FT (P4/P5) | 1.86 -> 1.80, -0.06 [-0.11, -0.01] | 17.93 -> **17.41**, -0.52 [-0.81, -0.26] |
| Parakeet-CTC-1.1B encoder LoRA (P6) | 1.86 -> 1.81, -0.05 [-0.11, -0.00] | 17.93 -> **17.53**, -0.40 [-0.67, -0.14] |
| Qwen2-Audio-7B FT (P4/P5) | 35.63 -> **1.79** | 99.91 -> **18.18** (base mostly answers instead of transcribing) |

What changes:
- **Holds, with smaller effects:** the Gemma-4 full and LoRA gains, Whisper on dev-clean,
  Qwen2-Audio, and the Parakeet AMI gains (full and LoRA). The 300-row sets overstated most of
  these by roughly 1.5-2x.
- **Corrected:** P5's "AMI improves for all four" does not hold for Whisper (not significant).
  P4's "Parakeet: no gain" is a small but significant -0.06 points on full dev-clean.
- **Newly resolved:** full fine-tuning beats LoRA on dev-clean for Gemma-4. On AMI the two are
  indistinguishable.

Process note: a first pass picked Gemma-4 run directories by name and timestamp. Those were
30-step runs from later experiments, and they produced a spurious "LoRA is worse than base".
The rerun reads the checkpoint from each committed result's recorded `args`. Those first-pass
results were discarded (kept apart under `artifacts/speech/big`, not cited).
