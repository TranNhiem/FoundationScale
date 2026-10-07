# Speech P6: a Canary recipe that does not regress, duration grouping, LoRA for CTC models

Measured 2026-10-07 on one GB200 tray (direct runs, staggered launches).

## 1. Canary: mixing short conversational data removes the out-of-domain regression

The LibriSpeech-only Canary fine-tune (`../speech_canary`) doubled AMI WER, ~85% of it runaway
hallucination on one-word backchannels. Two remedies were examined:

- **Decode-side length bound: rejected from source.** NeMo's AED decoders cap a hypothesis at
  `encoder length + max_generation_delta`, with the encoder length of the padded BATCH and an
  additive delta; at 12.5 encoder frames per second even delta 0 allows ~30 tokens for a 2.4 s
  "Yeah.", about the length of the runaway outputs. It would trim the garbage, not prevent it.
- **Data-side mix: measured.** 6,000 AMI ihm TRAIN clips (1 to 25 s, 7.0 h, 1,511 under 2 s;
  `build_ami_train.py`; AMI train and test meetings are disjoint) shuffled 50/50 with 6,000
  LibriSpeech clips, same recipe otherwise (500 steps, batch 8, lr 1e-5, pnc=no).

| Set (WER, pnc=no) | Base | LibriSpeech-only FT | LibriSpeech + AMI mix |
|---|---|---|---|
| LibriSpeech dev-clean | 1.43% | 1.43% | 1.49% |
| VoxPopuli | 5.32% | 5.67% | 5.44% |
| AMI | 16.24% | 30.83% | **12.56%** |
| Earnings-22 | 23.39% | 37.91% | 26.86% |

AMI runaways fall from 13 to 4 of 300 (base: 3). Earnings-22 keeps a +3.5 point gap with only 2
runaways: errors spread over financial-call speech that is in neither training source; closing it
would need in-domain data.

## 2. Duration-grouped batching: declared, tested, no measurable gain here

`FOUNDATIONSCALE_TRAIN_AUDIO_GROUP_BY_DURATION=1` groups batches by the manifest's measured
`duration` (transformers 5.5 `train_sampling_strategy="group_by_length"`); it refuses (96) when the
column is absent. Gemma-4-E4B, 60 steps, batch 8, 4 dataloader workers:

| Mode | Off | On |
|---|---|---|
| Full fine-tune | 0.993 steps/s | 0.942 steps/s |
| LoRA r16 | 1.153 steps/s | 1.102 steps/s |

Padding is not the bottleneck at this scale; the remaining per-step cost is unprofiled (the
collator's two chat-template passes per batch and CPU feature extraction are the candidates). The
option stays as a declared, tested no-op-when-off; profiling is follow-up work.

## 3. LoRA for CTC models: an explicit adapter scope inside the encoder

Parakeet calls its encoder by keyword (no peft wrap point at the root) and its only non-encoder
module is a 1x1 `Conv1d` head, so "LoRA on the language side" attaches to nothing useful. New
`FamilySpec.adapter_prefixes` declares where adapter targets are selected (default: the language
prefixes, so existing families are unchanged); declaring a prefix inside a tower is the explicit
opt-in that lets adapters reach it. Parakeet declares `encoder.layers` with leaves
q/k/v/o_proj + linear1/linear2 (measured), and trains `encoder.subsampling` in full (called
positionally). Under an adapter, the tower-movement gate now judges the declared full-train
modules (saved whole by `modules_to_save`) instead of the whole tower.

| Run | Adapter attach | Movement | Result |
|---|---|---|---|
| Parakeet LoRA (smoke) | 336 encoder modules, 22M / 1.08B trainable | subsampling 12/12 | PASS |
| Gemma-4 LoRA (regression) | 294 language modules, towers excluded | audio tower 217/271, projector 1/1 | PASS |

Parakeet LoRA, 500 steps (lr 1e-4, warmup 50) vs base and full fine-tune:

| Set | Base | Full FT (BN frozen) | LoRA (encoder scope) |
|---|---|---|---|
| dev-clean | 1.66% | 1.67% | 1.76% |
| AMI | 16.38% | 15.25% | 15.54% |

LoRA keeps most of full fine-tuning's AMI gain with ~2% of the parameters trainable.
