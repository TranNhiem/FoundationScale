# Qwen3-ASR-1.7B: fine-tuned in FoundationScale on Earnings-22 (2026-10-10/11)

The first model adopted through the upstream process (validation_campaigns/speech_repro) that FS
also trains. Profile `hf-cand-519` (transformers 5.19.0); GB200, r04dgx03 GPUs 2-3.

## 1. The training format, measured before any code

Qwen3-ASR's prompt comes from upstream's own `apply_transcription_request(audio, language=...)`,
which ends in the forced prefill `language English<asr_text>`. The training target is the
transcript plus the end-of-turn token (`<|im_end|>`, the eos), and everything before it is masked.
Per-token loss on 8 dev-clean clips:

| target | loss |
|---|---|
| the model's own greedy transcript | **0.014** |
| the LibriSpeech reference (upper case, no punctuation) | 1.61 |
| an unrelated sentence | 6.45 |

The format is right: the model is near-certain of its own output. The 1.61 is a style mismatch,
not a wrong format. Fine-tuning Qwen3-ASR on LibriSpeech-style targets would retrain its casing
and punctuation, so the training data here is Earnings-22, whose transcripts are cased,
punctuated and verbatim (fillers, false starts).

FS changes: FamilySpec `qwen3_asr`; speech kind `audio_llm`; `train_transcription_collator_or_refuse`,
which wraps `apply_transcription_request`, requires a declared language
(`FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE`, refusing 96 without one), and keeps every audio-LLM check
(strict loading, coverage, placeholder count, width refusal, feature-drop refusal). The per-clip
placeholder count copies the processor's private formula (ledger `hf-qwen3-asr-token-formula`),
and it matches upstream on six lengths. On the real processor, all 4 probe rows' labels decode to
exactly the answer plus `<|im_end|>`, and placeholders are 4/4 verified.

## 2. The run

`train.sh 2,3 OUT 750`: full fine-tune, bf16, lr 1e-5, 2 GPUs x per-device batch 8 (global 16),
750 steps (2 epochs of `earnings_train6000`: 6,000 rows from the 119 non-test calls), seed 0.
Every gate passed: audio rows 12,000/12,000 (all-reduced over both ranks); placeholders
12,000/12,000 verified, 0 unmeasured; tower movement 349/393 encoder and 4/4 projector parameters.

Checkpoint selection on `earnings_validation1000` (rule fixed in advance: lowest FS-normalizer WER):

| checkpoint | FS normalizer | Whisper normalizer | hypothesis / reference words |
|---|---|---|---|
| base | 18.20 | 13.44 | 17,754 / 18,585 |
| 250 | 16.71 | 15.82 | 19,323 / 18,585 |
| **500** | **12.73** | **11.44** | 18,631 / 18,585 |
| 750 | 12.89 | 11.61 | |

Step 250 over-generates, which is why the Whisper normalizer reads it as worse than base.

## 3. Held-out result

`ood_earnings223000_conv`: 2,504 rows from the 6 Earnings-22 test calls, with zero call overlap
with training.

| model | FS normalizer | Whisper normalizer |
|---|---|---|
| Qwen3-ASR zero-shot | 14.54 | 9.80 |
| **Qwen3-ASR + Earnings, FS fine-tune (step 500)** | **9.66** | **8.10** |
| Canary-1B-flash + Earnings (speech_p8, same test rows) | 11.33 | |

Paired bootstrap on identical rows (FS normalizer): **-4.88 points, 95% CI [-5.26, -4.48]**
(dev: -5.47 [-6.22, -4.78]). Decode health on the test set: `speech.runaway_hypotheses` PASS
(1 of 2,504 rows, limit 13); `speech.repetition_loops` PASS (48 loop words, limit 256; the base
model produced 8, so loops rose but stayed within bounds).

Forgetting, LibriSpeech test-clean (2,620): Whisper normalizer 1.643 -> 1.709 (the README value
is 1.63, so it is still inside the 0.3 tolerance); FS normalizer 1.883 -> 2.193. The FS normalizer
moves more because it counts the verbatim style the model has just learned.

## 4. Found along the way (speech plane, fixed)

- **Unresolvable towers now refuse before training.** Under transformers 5.18/5.19,
  Qwen2-Audio's modules sit under `model.`; the 5.5-measured FamilySpec names (`audio_tower`)
  resolve to nothing. That used to surface after the whole run, as a VACUOUS tower-movement gate
  (RED, exit 5; torchrun then reports 1). `speech_tower_resolution_refusal` refuses (96) at
  model load and names the actual top-level modules. The qwen2_audio FamilySpec itself is
  correct for the pinned hf-26.04 lane; re-measuring it is a blocker for promoting hf-cand-519.
- **Layerdrop under DDP.** Parakeet-CTC ships `encoder_config.layerdrop=0.1`, and a 2-GPU run
  aborted at step 2 ("Expected to have finished reduction in the prior iteration", exit 5,
  control run on the committed code). FS now tells DDP to expect unused parameters when the
  config declares layerdrop, leaving the recipe as upstream ships it. The same 2-GPU run passes
  (336/336 audio rows).
