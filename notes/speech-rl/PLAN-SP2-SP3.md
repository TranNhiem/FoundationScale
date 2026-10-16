# SP-2 / SP-3 plan: speech preference training and speech RL (for approval, 2026-10-16)

Both items change the shared RL plane (`src/foundationscale/rl/`), so they wait for approval.
Scoped by reading the code (MiMo-Pro read the files; two load-bearing claims re-checked by hand).

## Verified starting point
- `rl/preference_trainer.py`: DPO/IPO/KTO/ORPO/SimPO/CPO are modality-agnostic; images and video ride
  through `image_column`/`video_column`. Audio is absent: `_MODALITY_KEYS = ("image", "images",
  "video", "videos")` (line 172), so an `audio` field today is silently ignored, not refused.
- `rl/trainer.py` hard-codes `MCQLetterReward` (line 1758); `rl/rewards.py` exports only that class.
- Ready to reuse: Qwen3-ASR collation in `train/audio.py` (`train_transcription_collator_or_refuse`),
  WER, normalizer, `is_runaway`, `loop_words` and the paired bootstrap in `train/speech_metrics.py`.

## SP-2 first (smaller, offline data, no rollout loop) - about 125 lines + a pair builder
1. `PreferenceTrainConfig.audio_column` / `audio_language`, refused when partial or mixed with image/video.
2. Add `"audio"` to `_MODALITY_KEYS` so an undeclared audio carrier refuses (closes the silent drop).
3. `_encode_batch_audio`: chosen/rejected rows through the existing Qwen3-ASR transcription collator.
4. Pair builder (campaign script): sample N hypotheses per utterance from the SP-4/1.7B SFT model on
   `earnings_train6000`; chosen = lowest WER, rejected = highest; drop ties, runaways, empty refs.
5. Done-when: held-out Earnings WER below the SFT model's, paired bootstrap CI below 0, runaway gate PASS.

## SP-3 after SP-2 - about 265 lines
1. `WERReward` in `rewards.py` (1 - WER, abstain on empty reference, optional runaway/loop penalty).
2. `RLTrainConfig.reward` instead of the hard-coded MCQ reward (refused together with `answer_pattern`).
3. Audio rollouts: new `rl/asr_rollout.py` (an ExperienceBatch source with audio features), plus
   passing modality columns through the rows lane (`_one_step_rows` passes `{}` today).
4. Raise `max_new_tokens` for transcripts (64 truncates them).
5. Done-when: held-out WER below SFT with zero runaway/loop regressions.

## Questions for the user
- Approve touching `rl/preference_trainer.py` (SP-2), then `rl/trainer.py` and `rewards.py` (SP-3)?
- Coordinate with the agentic-RL thread, which also edits `rl/trainer.py`.
