# 01 — Installation: shared base + one speech dependency

This chapter is short on purpose. The base stack for FoundationScale speech/ASR fine-tuning — the repo
clone, the conda/uv/Docker environment, FoundationScale itself, torch, transformers, peft — is installed
exactly the way the VLM tutorial installs it. Read and run
[`../vlm-2xH200/01_installation.md`](../vlm-2xH200/01_installation.md) first, then come back here. This
chapter covers only what speech adds: one extra Python dependency, the three target checkpoints, and the
VM quirks that bite ASR fine-tuning specifically.

**Time needed:** ~10 min of hands-on work if the shared base is already in place (librosa install: <1 min —
a fresh user; weight downloads: ~24 min for all three models, 42 GB total — **MEASURED** on 2x H200
(141 GB), 2026-10-10). From a bare VM, add the vlm chapter's own timings.

**The VM you're on:** 2x NVIDIA H200 (141 GB) — the same 2x H200 VM class the vlm tutorial was verified on.

## Target models

| model | HF id | kind | fine-tune method used in this tutorial |
|---|---|---|---|
| Parakeet CTC 1.1B | `nvidia/parakeet-ctc-1.1b` | CTC | full fine-tune |
| Whisper Large v3 | `openai/whisper-large-v3` | seq2seq | full fine-tune |
| Gemma 4 E4B-it | `google/gemma-4-E4B-it` | audio-LLM | LoRA (r16, alpha32) |

Two CTC/seq2seq ASR models, one audio-LLM. The audio-LLM is fine-tuned with LoRA only; the two ASR
models are fine-tuned end to end.

**Qwen/Qwen2-Audio-7B-Instruct is not in the primary lineup on this VM.** Its FoundationScale FamilySpec
declares module paths measured against an older transformers release; on this VM's
transformers==5.18.0 the real module nesting differs, and FoundationScale now correctly REFUSES
(exit 96) rather than training against an unresolved audio tower — see chapter 03 for the exact refusal
message and a verified reproduction. This is pending a FamilySpec update by the speech-plane owner, out
of scope for this examples-only tutorial. (If you want to run that refusal demo, download the checkpoint
once with `hf download Qwen/Qwen2-Audio-7B-Instruct --local-dir models/Qwen2-Audio-7B-Instruct` — it is
not part of the primary download list below.)

**Qwen3-ASR-1.7B** also trains in FoundationScale (transformers >= 5.13) but needs cased, punctuated
training targets; LibriSpeech's upper-case, unpunctuated references would just retrain its output style
rather than demonstrate ASR fine-tuning, so it is not used in this tutorial's LibriSpeech runs. See
`validation_campaigns/speech_q3asr/EVIDENCE.md` if you want to try it on a different dataset.

## Shared environment

This tutorial reuses the exact conda/uv/Docker setup from
[`../vlm-2xH200/01_installation.md`](../vlm-2xH200/01_installation.md) — same FoundationScale editable
install, same torch/transformers/peft — and nothing else in that base needs to change, with one exception
covered below. Run every command in this chapter from the repository root.

## Speech-specific dependency: librosa (MEASURED, required)

`soundfile` alone is not enough. **MEASURED** on 2x H200 (141 GB), 2026-10-10: attempting to fine-tune
`nvidia/parakeet-ctc-1.1b` without `librosa` installed produces this exact refusal, exit code 96:

```text
REFUSAL (exit 96): FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN is declared but AutoProcessor failed to load for
'models/parakeet-ctc-1.1b': ImportError('\nParakeetFeatureExtractor requires the librosa library. But that
was not found in your environment. You can install them with pip:\n`pip install librosa`\nPlease note that
you may need to restart your runtime after installation.\n'). Refusing rather than downgrading to a
tokenizer, which would train on the text and drop the audio
```

FoundationScale refuses instead of quietly training on text. Install `librosa` and the refusal disappears.

```bash
# conda env
pip install librosa

# uv env
uv pip install --python venv/bin/python librosa
```

This is the only extra dependency beyond the vlm tutorial's base stack, and it is needed for all three
models in this tutorial's stack only because of Parakeet. `soundfile`, which the base stack already
includes, is sufficient for whisper-large-v3 and gemma-4-E4B-it; only Parakeet's feature extractor
imports `librosa`.

**Check it worked.** You get a version and no traceback:

```bash
python3 -c "import librosa; print(librosa.__version__)"
```

```text
1.0.0
```

## Download the weights

Pull each checkpoint into `models/<name>`. Sizes are **MEASURED** on download, 2x H200 (141 GB), 2026-10-10.

| model | local path | on-disk size (MEASURED) |
|---|---|---|
| `nvidia/parakeet-ctc-1.1b` | `models/parakeet-ctc-1.1b` | ~8 GB |
| `openai/whisper-large-v3` | `models/whisper-large-v3` | ~19 GB (full repo, all weight formats) |
| `google/gemma-4-E4B-it` | `models/gemma-4-E4B-it` | ~15 GB |

```bash
hf download nvidia/parakeet-ctc-1.1b --local-dir models/parakeet-ctc-1.1b
```

```bash
hf download openai/whisper-large-v3 --local-dir models/whisper-large-v3
```

The plain command above pulls the **full repo** (~19 GB): `openai/whisper-large-v3` ships both
`pytorch_model*.bin` and `model*.safetensors` weight files, plus fp32 copies of each, side by side.
`foundationscale.train` only ever reads the safetensors ones. To fetch **safetensors-only** (~8.7 GB,
**MEASURED**), exclude every `.bin` weight file:

```bash
hf download openai/whisper-large-v3 --local-dir models/whisper-large-v3 \
    --exclude "*.bin" --exclude "*.msgpack" --exclude "*.h5" --exclude "*.gguf" --exclude "*.ot"
```

```bash
hf download google/gemma-4-E4B-it --local-dir models/gemma-4-E4B-it
```

Once the weights are local, training and evaluation run with `HF_HUB_OFFLINE=1`. No network calls are
needed after this step.

**Check it worked.** All three directories exist and the sizes match the table:

```bash
du -sh models/parakeet-ctc-1.1b models/whisper-large-v3 models/gemma-4-E4B-it
```

```text
8.0G    models/parakeet-ctc-1.1b
19G     models/whisper-large-v3
15G     models/gemma-4-E4B-it
```

## VM quirks you must know before the next chapter

Both are **MEASURED** on a 2x H200 VM, 2026-10-10.

- **`torchrun` dies on rendezvous / hangs at startup.** Cause: `localhost` resolves to IPv6 only on this VM
  class. Fix: always pass `--master_addr 127.0.0.1` — never `localhost`, never `::1`. This tutorial's
  launcher always uses port 29880. The vlm tutorial's own launcher uses 29500 on the same VM class, so the
  two never collide if you run them back to back.

- **Some clips never reach training.** Cause: audio must already be 16 kHz mono. This tutorial's data step
  (chapter 02) refuses to resample — any non-16 kHz or non-mono clip is skipped and counted, never silently
  converted. Fix: convert your corpus to 16 kHz mono before chapter 02, and read its skip counts.

## Troubleshooting

**Training refuses with exit 96 and `ParakeetFeatureExtractor requires the librosa library`.** Cause:
`librosa` is missing — `soundfile` alone does not satisfy Parakeet's feature extractor. Fix: `pip install
librosa` (conda env) or `uv pip install --python venv/bin/python librosa` (uv env), then rerun. See
[Speech-specific dependency](#speech-specific-dependency-librosa-measured-required).

**Training refuses with exit 96 naming a missing `FOUNDATIONSCALE_TRAIN_AUDIO_*` env var.** Cause: the model
kind needs a specific audio declaration before FoundationScale will load its processor. Fix: declare the
right `FOUNDATIONSCALE_TRAIN_AUDIO_*` variable for your model kind — covered in chapter 03.

**Next:** [02 — Data preparation](02_data_preparation.md)
