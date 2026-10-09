# Speech P5: out-of-domain evaluation, LoRA for Whisper and Qwen2-Audio, Qwen2-Audio placeholder check

Measured 2026-10-06 on one GB200 tray (direct runs, staggered launches).

## 1. Out-of-domain held-out WER

The four P3/P4 checkpoints (500 steps on LibriSpeech train-clean-100) against their base models, on
three out-of-domain test sets from `hf-audio/open-asr-leaderboard` (ungated parquet), built by
`build_ood.py`: 1 to 25 s clips at 16 kHz, transcripts as published. VoxPopuli (European
Parliament, 300 utterances), AMI (meetings, 300), Earnings-22 (earnings calls, 148: only 148 clips
in two shards fit the 1 to 25 s window). Same eval script and normaliser as P3/P4.

| Model | VoxPopuli | AMI | Earnings-22 |
|---|---|---|---|
| Gemma-4-E4B (full FT) | 10.38 -> 9.43 | 26.72 -> 25.23 | 26.75 -> 27.55 |
| Whisper-large-v3 (full FT) | 9.39 -> 9.88 | 20.57 -> 18.89 | 19.35 -> 19.18 |
| Parakeet-CTC-1.1B (full FT, BN statistics frozen) | 6.28 -> 6.30 | 16.38 -> 15.25 | 24.42 -> 24.53 |
| Qwen2-Audio-7B (full FT) | 38.09 -> 7.36 | 82.45 -> 17.07 | 73.48 -> 24.36 |

Reading:
- Pipeline check: the base Whisper-large-v3 (9.4%) and Parakeet-CTC-1.1B (6.3%) VoxPopuli numbers sit
  next to the Open ASR Leaderboard's published figures (about 9.5% and 6.6%), though our normaliser is
  simpler than the leaderboard's.
- LibriSpeech-only fine-tuning moves out-of-domain WER little: AMI improves for all four models
  (including Parakeet, which had no in-domain headroom), VoxPopuli and Earnings-22 are mixed within
  about 0.8 points, and no model regresses badly.
  **Correction (speech_p7 section 5):** on AMI-2000 with a paired bootstrap, the AMI gain is
  significant for Gemma-4, Qwen2-Audio and Parakeet but not for Whisper (+0.60 points, 95% CI
  [-0.49, +2.51]).
- Qwen2-Audio's large drops are mostly the base instruct model's wrapper text ("The original content
  of this audio is: ...") disappearing once trained on plain transcripts.

## 2. LoRA wrap points for Whisper and Qwen2-Audio

Both models call their audio modules with a positional input (transformers 5.5:
`self.encoder(input_features, ...)`; `self.audio_tower(input_features, ...)`,
`self.multi_modal_projector(feature)`), so the roots are the peft `modules_to_save` wrap points.
Registered in `FamilySpec.adapter_full_train` and confirmed by 30-step LoRA runs (rank 16, lr 1e-4):

| Model | Full-train modules | LoRA modules | Audio tower moved | Run |
|---|---|---|---|---|
| Whisper-large-v3 | model.encoder | 320 decoder modules | 476 / 487 | PASS |
| Qwen2-Audio-7B | audio_tower, multi_modal_projector | 224 language-model modules | 473 / 487, projector 2 / 2 | PASS |

Parakeet-CTC stays unmeasured, so a LoRA run that declares audio on it still refuses (96).

## 3. Qwen2-Audio placeholder coverage

Qwen2AudioProcessor has no per-clip count helper, so the placeholder gate used to SKIP. It expands
`<|AUDIO|>` by a fixed formula over each clip's feature-mask length (stride-2 conv, stride-2 pool),
now applied to the batch's own `feature_attention_mask` from a declared processor table. In the Qwen2-Audio
LoRA run the gate PASSed: 248 / 248 rows verified, 0 unmeasured.

## 4. Canary

Not built: see `docs/research/speech_canary.md` (Canary is NeMo-only and neither container has
NeMo's ASR collection; recommended route is a separate NeMo lane that FoundationScale adjudicates).
