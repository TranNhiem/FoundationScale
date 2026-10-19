# Fine-tune ASR models on 2x H200 with FoundationScale

This is the **ASR** (speech-to-text) example. For text-to-speech and omni speech-LLMs, and a map of all speech work, see [../SPEECH.md](../SPEECH.md).

This tutorial fine-tunes **four speech/ASR models of three different kinds** on LibriSpeech with
FoundationScale — a CTC encoder (full fine-tune), a seq2seq encoder-decoder (full fine-tune), and two
audio-LLMs (LoRA fine-tune) — and evaluates every one base vs. fine-tuned with paired-bootstrap
significance testing on the WER difference. The raw validation evidence behind this tutorial lives at
[`validation_campaigns/speech_p0`](../../validation_campaigns/speech_p0),
[`speech_p1`](../../validation_campaigns/speech_p1),
[`speech_p2`](../../validation_campaigns/speech_p2),
[`speech_p3`](../../validation_campaigns/speech_p3),
[`speech_p4`](../../validation_campaigns/speech_p4),
[`speech_p5`](../../validation_campaigns/speech_p5),
[`speech_p6`](../../validation_campaigns/speech_p6),
[`speech_p7`](../../validation_campaigns/speech_p7),
[`speech_p8`](../../validation_campaigns/speech_p8),
[`speech_ddp`](../../validation_campaigns/speech_ddp),
[`speech_canary`](../../validation_campaigns/speech_canary),
[`speech_repro`](../../validation_campaigns/speech_repro) and
[`speech_q3asr`](../../validation_campaigns/speech_q3asr). The 500-step 2x H200 runs below were measured
on this hardware; the evidence for those runs is in
[`validation_campaigns/speech_2xh200/EVIDENCE.md`](../../validation_campaigns/speech_2xh200/EVIDENCE.md).

## What you get

- A conda/uv environment for speech workloads, shared with the VLM tutorial (chapter 01).
- A LibriSpeech → FLAC + JSONL data pipeline: `scripts/prepare_librispeech.py` turns parquet shards into
  16 kHz mono audio plus manifests (chapter 02).
- One training launcher, `scripts/train_speech.sh`, covering three model kinds with the same interface:
  CTC full fine-tune, seq2seq full fine-tune, and audio-LLM LoRA fine-tune (chapter 03).
- A base-vs-fine-tuned WER evaluation harness — `eval_wer.py` plus `paired_bootstrap.py` — reporting the
  WER delta with a 95% confidence interval on identical held-out rows (chapter 04).
- 5 ready-to-run configs (`configs/*.env`): one per trained model, plus a second Qwen2-Audio config for its
  full fine-tune recipe.

## Chapters

1. [01_installation.md](01_installation.md) — conda/uv environment, weights download, the one speech-specific
   dependency (librosa), and the VM quirks you need before training.
2. [02_data_preparation.md](02_data_preparation.md) — LibriSpeech parquet → FLAC + JSONL manifests, and
   why the eval split must be the full dev-clean set.
3. [03_training.md](03_training.md) — the `train_speech.sh` launcher, per-model knobs, Qwen2-Audio's LoRA
   and full-fine-tune recipes, and the Whisper target-case recipe lesson.
4. [04_evaluation.md](04_evaluation.md) — `eval_wer.py` and `paired_bootstrap.py`, base vs. tuned across
   the 4 trained models, and how to read the confidence intervals.

## Quick start (after chapters 1 and 2)

Everything below runs from the repository root, once chapter 01 (env + weights) and chapter 02
(LibriSpeech → manifests) are done:

```bash
bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/parakeet-ctc-1.1b.env
```

Measured on 2x H200: data prep for the FULL train-clean-100 (14 parquet shards, 28,539 rows) takes
~20–25 min (parquet download ~2.5 min + conversion); the `parakeet-ctc-1.1b` 500-step full fine-tune takes
~245 s wall (220 s pure train time) on 2 GPUs; evaluation (base + tuned, full 2,683-row dev-clean,
parallel on 2 GPUs) takes a few minutes each.

## Training techniques in this tutorial

**VERIFIED** (measured on 2x NVIDIA H200 (141 GB), 500 steps, full train-clean-100, full dev-clean):

| model | task formulation | fine-tuning method | config file | measured WER base -> tuned |
|---|---|---|---|---|
| `nvidia/parakeet-ctc-1.1b` | CTC | full fine-tune | `examples/speech-2xH200/configs/parakeet-ctc-1.1b.env` | 1.87% -> 1.77% |
| `openai/whisper-large-v3` | encoder-decoder seq2seq | full fine-tune | `examples/speech-2xH200/configs/whisper-large-v3.env` | 2.25% -> 1.72% |
| `google/gemma-4-E4B-it` | audio-LLM SFT | LoRA r16/a32 (audio tower + `embed_audio` projector train in FULL automatically) | `examples/speech-2xH200/configs/gemma-4-E4B-it.env` | 4.04% -> 2.99% |
| `Qwen/Qwen2-Audio-7B-Instruct` | audio-LLM SFT | LoRA r16/a32, automatic adapter scoping | `examples/speech-2xH200/configs/qwen2-audio-7b-lora.env` | 35.58% -> 3.41% |
| `Qwen/Qwen2-Audio-7B-Instruct` | audio-LLM SFT | full fine-tune, `SHARDING=fsdp`, `FSDP_STATE_DICT=full` | `examples/speech-2xH200/configs/qwen2-audio-7b.env` | 35.58% -> 2.57% |

**CTC full fine-tune (Parakeet).** A CTC head maps the encoder frame-wise to characters with no decoder and
no generation, so training computes a fixed loss over alignments. Pick it when the task is pure
transcription and the base model is already close: `parakeet-ctc-1.1b` needed only -0.09 WER points (95% CI
[-0.14, -0.05]) and is the cheapest run here — 14.3 GB/GPU peak, 0.44 s/step, 245 s wall. FoundationScale
freezes BatchNorm running statistics automatically.

**Encoder-decoder seq2seq full fine-tune (Whisper).** Whisper's encoder-decoder generates text tokens
autoregressively and is trained by teacher forcing on the transcript. Choose it for a strong pretrained ASR
model where you can afford full fine-tuning and want the largest clean ASR gain here (-0.53 points, best
tuned ASR WER at 1.72%, reference 1.69%). Cost: 27.6 GB/GPU, 0.41 s/step, 230 s wall at BATCH `4`
per-device. `FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en` is required for `seq2seq` and must be unset for
`ctc` / `audio_llm`, and Whisper targets must be **sentence-cased** — raw upper-case LibriSpeech targets
regress WER (see chapter 03 for the recipe lesson).

**Audio-LLM SFT with LoRA (Gemma-4, Qwen2-Audio).** A pretrained audio-LLM is fine-tuned on the
transcription instruction (`text` is the fixed literal `"Transcribe this audio exactly."`) with LoRA
adapters on the language model while the audio tower and `embed_audio` projector train in FULL
automatically. Pick it when you want a cheap adapter plus instruction-following behaviour, or cannot fit a
full fine-tune of a 7B model comfortably. Cost: gemma-4-E4B-it 86.2 GB/GPU, 0.94 s/step, 470 s wall;
qwen2-audio-7b 82.0 GB/GPU, 1.07 s/step, 541 s wall. Qwen2-Audio needs no `--adapter-target` flag
(automatic scoping: 224 LoRA target modules); 500 steps = ~0.14 epoch over 28,539 rows.

**Audio-LLM SFT with full fine-tune (Qwen2-Audio).** Every weight of the 7B audio-LLM is updated;
`SHARDING=fsdp` with `FSDP_STATE_DICT=full` is required to fit it. It gave the lowest Qwen2-Audio tuned WER
(2.57%) at the highest cost here: 92.9 GB/GPU peak, 0.83 s/step, 597 s wall, and ~16 GB of weights on disk.
Prefer LoRA unless you need that last bit of quality or plan to publish a fully merged model.

**How we evaluate.** All numbers above are WER on the full 2,683-row dev-clean split, scored by
`validation_campaigns/speech_p4/eval_wer.py` and checked with the 2000-resample paired bootstrap
`validation_campaigns/speech_p7/paired_bootstrap.py` (fixed seed). Both use
`foundationscale.train.speech_metrics.normalize_transcript`, which casefolds and strips punctuation. We use
the full split because 300-row held-out sets cannot resolve WER shifts under ~1 point, and we use
FoundationScale's normaliser everywhere because it "reads about 0.2 WER points higher than Whisper's
normalizer on the same hypotheses" — never mix the two in one table. Median eval cost: 5-10 minutes per
model on the 2,683-row split (full-FT eval 258 s).

**IN PROGRESS / PLANNED** (not measured, no commands here):

- `Qwen3-ASR-1.7B` trains on this stack (transformers >= 5.13) but needs cased, punctuated targets, so it is
  outside this tutorial's LibriSpeech recipe; evidence lives in `validation_campaigns/speech_q3asr/EVIDENCE.md`.

## Qwen3-ASR on real-world speech (Earnings-22)

LibriSpeech is read audiobook speech. To show the audio-LLM recipe on harder data, Qwen3-ASR was fully fine-tuned on Earnings-22 earnings calls and tested on 6 calls never seen in training (`ood_earnings223000_conv`, 2,504 rows). Recipe for both sizes: bf16, lr 1e-5, global batch 16, 750 steps on `earnings_train6000`, checkpoint chosen on `earnings_validation1000` before looking at the test set. Scripts and full evidence: [validation_campaigns/speech_q3asr/](../../validation_campaigns/speech_q3asr/) (`train.sh`, `decode.sh`, `sp4_run.sh`). Measured on GB200 (1.7B on 2 GPUs x batch 8; 0.6B on 1 GPU x batch 16, same global batch).

| model | zero-shot WER | fine-tuned WER | paired difference, 95% CI | checkpoint |
|---|---|---|---|---|
| Qwen3-ASR-1.7B | 14.54 | **9.66** | -4.88 [-5.26, -4.48] | step 500 |
| Qwen3-ASR-0.6B | 16.09 | **10.93** | -5.16 [-5.60, -4.75] | step 250 |

WER uses the FS normalizer. Neither model produced runaway or repetition-loop hypotheses. The 0.6B peaks earlier (step 250) and then drifts up slightly on validation, which is why the checkpoint is chosen on validation rather than taken from the last step.

### Preference training on audio (RPO)

On top of the fine-tuned Qwen3-ASR-0.6B, preference training with RPO (DPO plus an NLL term on the chosen transcript, `sft_weight` 1.0, lr 1e-6, 2 epochs, checkpoint picked on validation) lowers held-out WER from 10.93 to **10.65** (paired [-0.49, -0.06]), with no runaway or loop rows. Plain DPO without the NLL anchor made WER worse (14.87 on validation), so keep the anchor. Two candidates were decoded on the test set, so the reported result carries one extra look. Evidence and commands: [validation_campaigns/speech_pref/](../../validation_campaigns/speech_pref/).

RL with a WER reward also helps, with care: GRPO with a KL term, 16 clips x 8 rollouts per step, lr 1e-6, checkpoint picked on validation (step 50) reaches **10.57** held-out WER (paired [-0.58, -0.15]). Two conditions matter: a large batch (4 clips per step never beat the SFT model) and early stopping, because after about 100 steps the model learns to truncate transcripts (deletions 666 -> 7,424 at step 200). The recipe is `build_pairs.py`, `train_dpo.py`, `train_grpo.py` and `select_and_test.py` in the same folder.

The two stages stack. GRPO started from the RPO model (pass `INIT=<checkpoint+processor dir>` to the runner), with checkpoints picked on validation, reaches **10.39** held-out WER on average over 3 seeds (10.30 / 10.41 / 10.47). Every seed beats both the SFT model and RPO alone, with no runaway or loop rows and no collapse. The full recipe on Earnings-22: zero-shot 16.09, SFT 10.93, RPO then GRPO 10.39.

## Measured results: 4 ASR models, LibriSpeech, 2x H200

Workload: 500-step runs, global batch 8 (parakeet/whisper/qwen2-audio) or 16 (gemma-4), trained on the FULL
train-clean-100 (28,539 utterances, 100.59 h, 14 parquet shards) and evaluated base-vs-tuned on the FULL
2,683-row dev-clean split (never a 300-row slice — see chapter 02 §4 / chapter 04 §2 for why).
Measurement semantics: peak GB/GPU is the single highest 5-second `nvidia-smi` sample during each run;
WER/CI come from `eval_wer.py` + `paired_bootstrap.py` on identical held-out rows.

| model | fine-tune | exit | peak GB/GPU | s/step | WER base -> tuned | 95% CI (pts) |
|---|---|---|---|---|---|---|
| parakeet-ctc-1.1b | full FT | 0 | 14.3 | 0.44 | 1.87% -> 1.77% | [-0.14, -0.05] (significantly better) |
| whisper-large-v3 | full FT, sentence-cased targets | 0 | n/a (1 GPU, see ch.03) | 0.73 | 2.25% -> **1.72%** | [-0.65, -0.41] (significantly better) |
| gemma-4-E4B-it | LoRA r16/a32 + towers in full, sentence-cased targets | 0 | 86.2 | 0.94 | 4.04% -> **2.99%** | [-1.24, -0.85] (significantly better) |
| qwen2-audio-7b | LoRA r16 + towers in full, automatic adapter scoping | 0 | 82.0 | 1.07 | 35.58% -> **3.41%** | [-0.3303, -0.3134] (significantly better) |
| qwen2-audio-7b | full FT, FSDP (`full` state dict) | 0 | 92.9 | 0.83 | 35.58% -> **2.57%** | [-0.3385, -0.3216] (significantly better) |

- **parakeet-ctc-1.1b**: 1.87% -> 1.77%, CI [-0.14, -0.05], significantly better — essentially matching this
  repo's own longer-campaign ceiling
  ([`validation_campaigns/speech_p7/EVIDENCE.md`](../../validation_campaigns/speech_p7/EVIDENCE.md):
  1.86% -> 1.80%). No config workaround is needed: FoundationScale (current `origin/main`) auto-handles the
  model's LayerDrop+DDP interaction.
- **whisper-large-v3**: **2.25% -> 1.72%**, CI [-0.65, -0.41], significantly better (reference 1.69%). The
  one recipe trap: train on **sentence-cased** targets — training on LibriSpeech's raw UPPER CASE targets
  regressed WER to 3.32–3.58% because it fragments words Whisper already transcribes correctly zero-shot.
  Chapter 03 has the full recipe lesson and before/after numbers;
  [`validation_campaigns/speech_2xh200/EVIDENCE.md`](../../validation_campaigns/speech_2xh200/EVIDENCE.md)
  has the diagnostic trail.
- **gemma-4-E4B-it**: **4.04% -> 2.99%**, CI [-1.24, -0.85], significantly better — within 0.03 points of
  the repo's own longer-campaign number
  ([`validation_campaigns/speech_p7/EVIDENCE.md`](../../validation_campaigns/speech_p7/EVIDENCE.md):
  4.07% -> 2.96%). An upstream fix (`speech_tower_resolution_refusal` now resolves FamilySpec paths via
  `get_base_model()`) resolved a refusal that blocked re-verifying on sentence-cased targets; gates
  confirmed PASS (`speech.audio_row_coverage` 8016/8016, `speech.tower_movement` 271/271 + 1/1).
- **qwen2-audio-7b** has two measured recipes, both clean PASS. **LoRA r16 + towers in full**:
  **35.58% -> 3.41%**, CI [-0.3303, -0.3134], significantly better. The `qwen2_audio` `FamilySpec` is on the
  `transformers>=5.18` module layout with **automatic** LoRA adapter-target scoping — no `--adapter-target`
  flag needed, 224 modules resolve on their own. Gates confirmed PASS (`speech.audio_row_coverage`
  8016/8016, `speech.tower_movement` 487/487 + 2/2). **Full fine-tune** (lr 1e-5, global batch 8, FSDP across
  2 GPUs): **35.58% -> 2.57%**, CI [-0.3385, -0.3216], significantly better, within 0.8 points of the
  1-GPU reference (35.6% -> 1.79%). This recipe had a checkpoint-save bug on `transformers>=5.18`
  (`checkpoint.save_complete` found 0/876 tensors because the save-completeness check compared against the
  wrong tensor names) — **fixed upstream**; if you see it, `git pull` first. Gates now confirmed PASS
  (`speech.audio_row_coverage` 4008/4008, `speech.tower_movement` 487/487 + 2/2). The base number (35.58%)
  matches the quoted reference 35.6% in both recipes.
- **Qwen3-ASR-1.7B** also trains in FoundationScale (transformers >= 5.13) but is not used with this
  tutorial's LibriSpeech manifests (it needs cased, punctuated targets); see
  [`validation_campaigns/speech_q3asr/EVIDENCE.md`](../../validation_campaigns/speech_q3asr/EVIDENCE.md)
  if you want to try it on different data.

**Every model and recipe in the table improved** — but a 500-step run on 2 GPUs is a plumbing check and a
teaching tool, not a quality claim.

## Ready-to-run configs for each model

`configs/` holds one file per trained model. Every config sets `MODEL`, `DATASET`, `OUTPUT_DIR`,
`MODEL_KIND` and `ADAPTER`; everything else lives in `scripts/train_speech.sh`.

| config | model | method |
|---|---|---|
| `configs/parakeet-ctc-1.1b.env` | `nvidia/parakeet-ctc-1.1b` | full fine-tune |
| `configs/whisper-large-v3.env` | `openai/whisper-large-v3` | full fine-tune |
| `configs/gemma-4-E4B-it.env` | `google/gemma-4-E4B-it` | LoRA r16/alpha32 |
| `configs/qwen2-audio-7b-lora.env` | `Qwen/Qwen2-Audio-7B-Instruct` | LoRA r16/alpha32, automatic adapter scoping |
| `configs/qwen2-audio-7b.env` | `Qwen/Qwen2-Audio-7B-Instruct` | full fine-tune, FSDP, `--fsdp-state-dict full` — the owner's primary recipe |

```bash
bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/parakeet-ctc-1.1b.env
bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/whisper-large-v3.env
bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/gemma-4-E4B-it.env
bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/qwen2-audio-7b-lora.env
bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/qwen2-audio-7b.env
```

You can override any setting from the shell — `STEPS`, `LR`, `BATCH`, `GRAD_ACCUM`, `SAVE_EVERY`, `RANK`,
`ALPHA`, `GPUS`, `WORKERS`, `WARMUP`, `GROUP_BY_DURATION` — e.g.
`STEPS=500 LR=2e-5 bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/whisper-large-v3.env`.
Shell overrides win over the config file. `GROUP_BY_DURATION` is off by default; it is an optional
throughput toggle, not a correctness knob.

## Things that will bite you in this environment

- `torchrun` hangs forever at rendezvous → `/etc/hosts` lists `::1` before the IPv4 entry for `localhost`, so
  the rendezvous picks the IPv6 address and hangs → always pass `--master_addr 127.0.0.1` explicitly (this
  tutorial's launcher already does; it uses port 29880 vs. the VLM tutorial's 29500, so the two never
  collide).
- `nvidia/parakeet-ctc-1.1b` refuses to load with exit 96 naming
  `ParakeetFeatureExtractor requires the librosa library` → the feature extractor needs `librosa`, not just
  `soundfile` → `pip install librosa` (chapter 01).
- `Qwen/Qwen2-Audio-7B-Instruct` full fine-tune (`SHARDING=fsdp`, `--fsdp-state-dict full`) reporting RED,
  `checkpoint.save_complete` finding 0/876 tensors → a transformers >= 5.18 save-name mismatch, **fixed
  upstream**; `git pull` first — see chapter 03 §5.
- `torchrun` reports a bare `exit 1` and you can't tell which FoundationScale verdict fired → read the
  `exitcode : N` line inside torchrun's own per-rank failure report (under "Root Cause (first observed
  failure)") — that is the real 5/95/96 code, not torchrun's own wrapper exit.
- Unexpected OOM or slow runs → if the machine is shared with other training jobs, check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` for other processes before assuming your
  own config is wrong.
- Clips get skipped and counted instead of loading → audio must already be 16 kHz mono;
  `scripts/prepare_librispeech.py` and `eval_wer.py` both REFUSE to resample, and a non-16kHz or non-mono
  clip is skipped and counted, never silently converted (chapter 02).
- Small eval slices report flat or noisy WER deltas → 300-row eval sets cannot resolve WER shifts under
  ~1 point → always evaluate on the full 2,683-row dev-clean split, never a convenient slice
  (chapter 02 §4, chapter 04 §2).

## Sample data

This tutorial uses LibriSpeech (`openslr/librispeech_asr`, CC-BY-4.0): train-clean-100 (14 parquet shards,
28,539 utterances, 100.59 h) for training — the default — and the full dev-clean split (2,683 utterances,
5.22 h) for evaluation. A smaller 2-shard / 4,078-row train-clean-100 manifest is available as an optional
quick-smoke variant (chapter 02). `validation_campaigns/speech_p5` through `speech_p8` in this repository
already explore AMI (CC-BY-4.0, a hard out-of-domain benchmark) and Earnings-22 at much larger scale if
you want a harder test; this tutorial does not re-run those.
