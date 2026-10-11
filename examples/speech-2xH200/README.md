# Fine-tune ASR models on 2x H200 with FoundationScale

This tutorial takes you end to end through fine-tuning **four speech/ASR models of three different kinds**
on LibriSpeech with FoundationScale — a CTC encoder (full fine-tune), a seq2seq encoder-decoder (full
fine-tune), and two audio-LLMs (LoRA fine-tune) — and then evaluating every one of them honestly,
base vs. fine-tuned, with paired-bootstrap significance testing on the WER difference. The raw validation
evidence this tutorial draws on already lives in this repository at
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
[`speech_q3asr`](../../validation_campaigns/speech_q3asr). Every number specific to *this* tutorial — the
500-step 2x H200 runs below, plus traps not covered by the campaigns above — was freshly **MEASURED** on
2026-10-10/11; the raw evidence for this tutorial's own runs is in
[`validation_campaigns/speech_2xh200/EVIDENCE.md`](../../validation_campaigns/speech_2xh200/EVIDENCE.md),
which now covers **seven measurement rounds**, **three owner-reported fixes applied from origin/main
`ced0c5c`**, a resolved Whisper finding (an apparent regression traced to raw-upper-case training
targets, fixed by sentence-casing them: 2.25% -> 1.72%, matching the owner's own 1.69% — not a
transformers/torch version issue), and Qwen2-Audio's restoration through two further upstream fixes (a
`FamilySpec` update, then a full-fine-tune checkpoint-save fix).

## What you get

- A working conda/uv environment for speech workloads, shared with the VLM tutorial so you only set it up
  once (chapter 01).
- A LibriSpeech → FLAC + JSONL data pipeline: `scripts/prepare_librispeech.py` turns parquet shards into
  16 kHz mono audio plus manifests (chapter 02).
- One training launcher, `scripts/train_speech.sh`, that covers three model kinds with the same interface:
  CTC full fine-tune, seq2seq full fine-tune, and audio-LLM LoRA fine-tune (chapter 03).
- An honest base-vs-fine-tuned WER evaluation harness — `eval_wer.py` plus `paired_bootstrap.py` — that
  reports the WER delta with a 95% confidence interval on identical held-out rows (chapter 04).
- 5 ready-to-run configs (`configs/*.env`): one per trained model, plus a second Qwen2-Audio config for its
  full fine-tune recipe, copy-pasteable as-is.

## Chapters

1. [01_installation.md](01_installation.md) — conda/uv environment, weights download, the one speech-specific
   dependency (librosa), and the VM quirks you need to know before training.
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

**MEASURED** by a fresh user on 2x H200, 2026-10-10/11: data prep for the FULL train-clean-100 (14 parquet
shards, 28,539 rows) takes ~20–25 min (parquet download ~2.5 min **MEASURED** + conversion); the
`parakeet-ctc-1.1b` 500-step full fine-tune takes ~245 s wall (220 s pure train time) on 2 GPUs;
evaluation (base + tuned, full 2,683-row dev-clean, parallel on 2 GPUs) takes a few minutes each.

## Measured results: 4 ASR models, LibriSpeech, 2x H200, 2026-10-10/11

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

- **parakeet-ctc-1.1b**: a clean win — 1.87% -> 1.77% with a CI of [-0.14, -0.05], significantly better,
  and it essentially matches this repo's own longer-campaign ceiling
  ([`validation_campaigns/speech_p7/EVIDENCE.md`](../../validation_campaigns/speech_p7/EVIDENCE.md):
  1.86% -> 1.80%). No config workaround is needed any more — FoundationScale (current `origin/main`)
  auto-handles the model's LayerDrop+DDP interaction.
- **whisper-large-v3**: **2.25% -> 1.72%**, CI [-0.65, -0.41], significantly better — within 0.03 points of
  the speech-plane owner's own reference run (1.69%). An earlier round measured a REGRESSION (2.25% ->
  3.32-3.58% across several configurations) and ruled out GPU count, attention implementation, training
  divergence and generation-config drift as causes — none of it was a software-stack bug. The real cause:
  this tutorial's manifest builder wrote training targets in LibriSpeech's raw UPPER CASE, which fragments
  words Whisper already transcribes correctly zero-shot into upper-case BPE pieces
  (`"Sweetwater"` -> `"SWEET WATER"`). Training on **sentence-cased** targets instead (the same transform
  this repo's own `speech_p3`/`speech_p6` campaigns always used) fixed it. See chapter 03 for the full
  recipe lesson and before/after numbers, and
  [`validation_campaigns/speech_2xh200/EVIDENCE.md`](../../validation_campaigns/speech_2xh200/EVIDENCE.md)
  for the diagnostic trail that found it.
- **gemma-4-E4B-it**: **4.04% -> 2.99%**, CI [-1.24, -0.85], significantly better — within 0.03 points of
  the repo's own longer-campaign number
  ([`validation_campaigns/speech_p7/EVIDENCE.md`](../../validation_campaigns/speech_p7/EVIDENCE.md):
  4.07% -> 2.96%). A FoundationScale fix (`speech_tower_resolution_refusal` now resolves FamilySpec paths
  via `get_base_model()`, landing after this tutorial's first Gemma-4 round) resolved a separate refusal
  that had blocked re-verifying on sentence-cased targets; gates confirmed PASS
  (`speech.audio_row_coverage` 8016/8016, `speech.tower_movement` 271/271 + 1/1).
- **qwen2-audio-7b** has two measured recipes, both clean PASS. **LoRA r16 + towers in full**:
  **35.58% -> 3.41%**, CI [-0.3303, -0.3134], significantly better. The owner updated the `qwen2_audio`
  `FamilySpec` to the `transformers>=5.18` module layout (`origin/main` `0fa19be`) with **automatic** LoRA
  adapter-target scoping — no `--adapter-target` flag needed any more, 224 modules resolve on their own.
  Gates confirmed PASS (`speech.audio_row_coverage` 8016/8016, `speech.tower_movement` 487/487 + 2/2).
  **Full fine-tune** (lr 1e-5, global batch 8, FSDP across 2 GPUs — the owner's primary recipe):
  **35.58% -> 2.57%**, CI [-0.3385, -0.3216], significantly better, within 0.8 points of the owner's own
  1-GPU reference (35.6% -> 1.79%). This recipe hit a real checkpoint-save bug in an earlier round
  (`checkpoint.save_complete` found 0/876 tensors on `transformers>=5.18`, because transformers reverts the
  on-disk tensor names on save and the old save-completeness check compared against the wrong names) —
  **fixed upstream** (`origin/main` `152f43c`); gates now confirmed PASS (`speech.audio_row_coverage`
  4008/4008, `speech.tower_movement` 487/487 + 2/2). The base number (35.58%) independently matches the
  owner's own quoted 35.6% in both recipes. See chapter 03 §5 for the full story.
- **Qwen3-ASR-1.7B** also trains in FoundationScale (transformers >= 5.13) but is not used with this
  tutorial's LibriSpeech manifests (it needs cased, punctuated targets); see
  [`validation_campaigns/speech_q3asr/EVIDENCE.md`](../../validation_campaigns/speech_q3asr/EVIDENCE.md)
  if you want to try it on different data.
- Stated once, clearly: **every model and recipe in the table above improved** — parakeet, whisper and
  gemma-4 each within ~0.03 points of the speech-plane owner's own reference numbers; qwen2-audio-7b's LoRA
  path has no owner LoRA reference to compare against (the owner's reference is for the full fine-tune),
  but its full fine-tune lands within 0.8 points of that reference (2.57% vs. 1.79%), and both recipes'
  base numbers independently match the owner's quoted base within 0.02 points — **but a 500-step run on 2
  GPUs is still a plumbing check and a teaching tool, not a quality claim** — the same spirit as the VLM
  tutorial's own "a 6-step adapter is a plumbing check, not a quality claim". Getting there took real
  debugging (see chapter 03): the first Whisper attempt trained on the wrong target TEXT FORMAT, the first
  Gemma-4 rerun hit a FoundationScale bug now fixed upstream, and Qwen2-Audio needed two separate upstream
  fixes (a `FamilySpec` update before it could train at all, then a checkpoint-save fix before its full
  fine-tune could be evaluated). All of it is kept in the record, not hidden.

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
Shell overrides win over the config file **(VERIFIED on this VM)**. `GROUP_BY_DURATION` is off by default;
it is an optional throughput toggle, not a correctness knob.

## Things that will bite you on this VM (all measured)

All **MEASURED**, 2026-10-10/11.

- `torchrun` hangs forever at rendezvous → `/etc/hosts` lists `::1` before the IPv4 entry for `localhost`,
  so the rendezvous picks the IPv6 address and hangs → always pass `--master_addr 127.0.0.1` explicitly
  (this tutorial's launcher already does this; it uses port 29880 vs. the VLM
  tutorial's 29500, so the two never collide — **TODO(verify)** the port is still free if you run other
  jobs concurrently).
- `nvidia/parakeet-ctc-1.1b` refuses to load with exit 96 naming
  `ParakeetFeatureExtractor requires the librosa library` → the feature extractor needs `librosa`, not
  just `soundfile` → `pip install librosa` (chapter 01).
- An earlier round found `Qwen/Qwen2-Audio-7B-Instruct` full fine-tune (`SHARDING=fsdp`,
  `--fsdp-state-dict full`) reporting RED, `checkpoint.save_complete` finding 0/876 tensors → transformers
  reverts the on-disk tensor names on save (`transformers>=5.18`), and FoundationScale's old
  save-completeness check compared against the wrong names → **fixed upstream** (`origin/main` `152f43c`);
  if you see this, `git pull` first — see chapter 03 §5.
- `torchrun` reports a bare `exit 1` and you can't tell which FoundationScale verdict fired → read the
  `exitcode : N` line inside torchrun's own per-rank failure report (under "Root Cause (first observed
  failure)") — that is the real 5/95/96 code, not torchrun's own wrapper exit.
- This VM may be shared with other unrelated training jobs outside this tutorial's GPU-lock protocol → if
  a run OOMs unexpectedly or runs slower than expected, check
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
