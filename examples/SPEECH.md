# Speech in FoundationScale: ASR, TTS and omni speech-LLM

This page is the entry point for all speech work: what each example does, which models and training/inference techniques it covers, and what is measured versus planned.

## Overview

| task | example folder | models | techniques | status |
|---|---|---|---|---|
| ASR (speech to text) | [./speech-2xH200/](./speech-2xH200/), [./speech-2xA100/](./speech-2xA100/) | `nvidia/parakeet-ctc-1.1b`, `openai/whisper-large-v3`, `google/gemma-4-E4B-it`, `Qwen/Qwen2-Audio-7B-Instruct` | training: full fine-tune (CTC, seq2seq, FSDP) and LoRA r16/alpha32 (audio tower + `embed_audio` projector in full) | VERIFIED — all five configs measured |
| TTS (text to speech) | [./tts-GB200/](./tts-GB200/) | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, `openbmb/VoxCPM2` | zero-shot voice cloning scored by the Seed-TTS-eval harness (WER + WavLM SIM); LoRA fine-tuning for a new language (VoxCPM2, Vietnamese) | VERIFIED — evaluation and Vietnamese LoRA fine-tuning measured |
| omni speech-LLM | [./omni-GB200/](./omni-GB200/) | `Qwen3-ASR-1.7B` + `Qwen3-4B-Instruct-2507` (cascade), `Qwen3-Omni-30B-A3B-Instruct` (end-to-end) | inference only: VoiceBench benchmarking, upstream `evaluate.py` unmodified | VERIFIED — cascade (O0, MMSU not finished) and Qwen3-Omni (O1, all five subsets); fine-tuning PLANNED |

All ASR numbers on this page were measured on 2 GPUs of type H200. The A100 entry point for the same task lives at [./speech-2xA100/](./speech-2xA100/); no numbers on this page come from it.

## Which one do I need?

- Transcribe speech and score with WER → ASR, [./speech-2xH200/](./speech-2xH200/).
- Train a recognizer on your own audio (full fine-tune or a small adapter) → ASR.
- Generate speech or clone a voice and measure WER/SIM → TTS, [./tts-GB200/](./tts-GB200/).
- Answer questions from speech on a spoken benchmark → omni, [./omni-GB200/](./omni-GB200/): cascade if the parts must be swappable, Qwen3-Omni if one checkpoint has to read audio and answer.
- Teach a TTS model a new language or voice style → TTS Step 6, [./tts-GB200/](./tts-GB200/) (VoxCPM2 Vietnamese LoRA).
- Nothing here fine-tunes an omni model yet.

## ASR — VERIFIED on 2x H200 ([./speech-2xH200/](./speech-2xH200/))

Data: `openslr/librispeech_asr` clean/train.100 (28,539 rows / 100.59 h), evaluated on full dev-clean (2,683 rows), 500-step runs. One launcher, `examples/speech-2xH200/scripts/train_speech.sh`, one config per model under `examples/speech-2xH200/configs/`; scoring is `validation_campaigns/speech_p4/eval_wer.py` with paired bootstrap in `validation_campaigns/speech_p7/paired_bootstrap.py`.

- `nvidia/parakeet-ctc-1.1b` — CTC model, full fine-tune. Pick it for a compact, accurate recognizer with little tuning; FoundationScale freezes its BatchNorm running statistics automatically.
- `openai/whisper-large-v3` — seq2seq (encoder-decoder) full fine-tune on sentence-cased targets. Pick it for the lowest tuned WER here when full fine-tuning is affordable.
- `google/gemma-4-E4B-it` — audio-LLM with LoRA r16/alpha32 on the language model (audio tower and `embed_audio` projector train in full). Pick it to keep the trainable footprint small on a 4B audio-LLM.
- `Qwen/Qwen2-Audio-7B-Instruct` LoRA r16/alpha32 — same idea on 7B, with automatic adapter scoping (224 target modules, no `--adapter-target` flag). Pick it when you want the 7B audio-LLM without training 7B of weights.
- `Qwen/Qwen2-Audio-7B-Instruct` full fine-tune (`SHARDING=fsdp`, `FSDP_STATE_DICT=full`) — pick it when only the best WER matters.

Headline result (2026-10-10/11): whisper-large-v3 goes from 2.25% to 1.72% WER (-0.53, 95% CI [-0.65, -0.41]); the best overall tuned WER is 1.72% (Whisper) and 2.57% (Qwen2-Audio full fine-tune). Full table: [./speech-2xH200/04_evaluation.md](./speech-2xH200/04_evaluation.md).

## TTS — VERIFIED: Seed-TTS-eval harness and Vietnamese LoRA fine-tuning ([./tts-GB200/](./tts-GB200/))

Evaluation (Steps 1-5) plus one fine-tuning recipe (Step 6). Two scripts synthesize (`synth_qwen3tts.py`, `synth_voxcpm2.py`), `seedtts_score.py` scores, and the upstream Seed-TTS-eval scripts run unmodified as subprocesses (`run_wer.py`, UniSpeech `verification_pair_list_v2.py`). Scoring rule: the judges must reproduce the published scores of the real recordings ("Human" row) before they score any synthesizer. Weights and the `seedtts_testset.zip` / `wavlm_large_finetune.pth` artefacts are fetched by `cluster/fetch_t0.sh` with `sha256` manifests.

- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` / `Qwen/Qwen3-TTS-12Hz-0.6B-Base` — inference-only voice cloning in Qwen's published protocol (`Qwen3TTSModel`, `generate_voice_clone`). Pick these to reproduce Qwen's Seed-TTS numbers with or without the smaller variant.
- `openbmb/VoxCPM2` — inference-only "ultimate cloning" (prompt wav plus its text, `--cfg-value 2.0`, `--inference-timesteps 10`). Pick it when you have both a prompt clip and its transcript.
- Judges: `run_wer.py` (Whisper-large-v3 for en, `funasr/paraformer-zh` for zh) and the WavLM-large SIM judge (`--model_name wavlm_large`) — they are the measurement instrument, not candidates.

Headline result: the harness scores the ground-truth rows of the en (1,088 as read) and zh (2,020 as read) splits first, validating the judges on real recordings, and then scores each synthesizer the same way. Per-model WER/SIM tables are in [./tts-GB200/](./tts-GB200/); the zh-hard split (400 rows) has no result row yet.

Fine-tuning (Step 6): VoxCPM2 with LoRA r32/alpha 32 on 64.9 h of commercial-safe Vietnamese (43,907 rows), 2,000 steps on one GB200 GPU. WER on FLEURS-vi 8.59 -> 6.79 (95% CI [-3.39, -0.58], n=857) and on MiniMax-vi 5.75 -> 1.44, with a small speaker-similarity cost (SIM -1.07, CI [-1.47, -0.68]). LoRA rather than full fine-tuning keeps the base model's punctuation and prosody, because two of the three Vietnamese corpora are lowercase without punctuation.

## Omni speech-LLM — VERIFIED (O0) on one GB200 tray ([./omni-GB200/](./omni-GB200/))

Inference and benchmarking only: `voicebench_run.py` wraps upstream VoiceBench (commit `b561541`) and runs each assistant over the objectively scored subsets (`openbookqa`, `mmsu`, `ifeval`, `advbench`, `bbh`); the remaining subsets need a GPT judge and are not run.

- `Qwen3-ASR-1.7B` → `Qwen3-4B-Instruct-2507` (assistant `cascade`) — inference-only cascade: audio is transcribed (its own language identification left on) and the 4B chat model answers with the upstream cascade system prompt. Pick it when the transcriber and the answer model should be swappable.
- `Qwen3-Omni-30B-A3B-Instruct` (assistant `qwen3_omni`) — inference-only end-to-end speech-LLM, speech in and text out, talker disabled (`disable_talker()`, `return_audio=False`), greedy, `max_new_tokens 2048`. Pick it when speech understanding and answering must live in one checkpoint.

Headline result (2026-10-12/13, cascade): IFEval 79.7, AdvBench refusal rate 99.8%, BBH acc 88.3, OpenBookQA acc 75.4, zero failed rows. O1 was measured the same way on 2026-10-13/14 and compared against the published model card in [./omni-GB200/](./omni-GB200/).

## What is not done yet (IN PROGRESS / PLANNED)

These are not on main and have no measured numbers here.

- TTS: full fine-tuning versus LoRA on the same Vietnamese data (running); Qwen3-TTS single-speaker fine-tuning (so far no better than zero-shot cloning, so not a recipe yet).
- Omni: fine-tuning recipes (thinker-only first) are planned; this page covers inference only.
- Unfinished runs: TTS zh-hard has no result row, and the cascade's `mmsu` run was stopped when its GPUs were released.
