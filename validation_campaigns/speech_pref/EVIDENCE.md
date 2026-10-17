# Speech preference training (SP-2) and speech RL (SP-3) on Qwen3-ASR-0.6B (2026-10-17)

ROADMAP SP-2 / SP-3, approved plan. Starting point for both: the SP-4 SFT model (Qwen3-ASR-0.6B +
Earnings-22, step 250), held-out `ood_earnings223000_conv` WER **10.93** (FS normalizer, 2,503 rows),
validation `earnings_validation1000` **14.09**. Every candidate is picked on validation by a rule fixed
in advance (lowest FS-normalizer WER, the SFT model itself a candidate), and only the winner is decoded
on the held-out test (`select_and_test.py`, which reproduces SP-4's paired CI exactly).

## What was built (src/foundationscale/rl)

- `preference_trainer.py`: audio mode (`audio_column`/`audio_language`). Pair records carry the clip and
  the sentinel prompt `<|asr_audio|>`; prompts go through the same Qwen3-ASR transcription request as
  SFT, so DPO prices the tokens SFT trained. `"audio"` is now a modality key, so an undeclared audio
  field refuses instead of being silently ignored (it was ignored before). Audio features take the
  policy dtype (bf16 conv weights reject float32 log-mels).
- `rewards.py`: `WERReward` = 1 - WER (FS normalizer), floored at -1 so one runaway row cannot swamp
  its group's advantages, optional runaway/loop penalties (refused if negative or non-finite);
  abstains on an empty gold. `trainer.py`: `RLTrainConfig.reward` replaces the hard-coded MCQ reward.
- `asr_rollout.py` + `trainer.py`: audio rollouts (transcription request per clip, group sampling,
  row-by-row decode), sharing the generate-score-price leg (`_price_generated`) with text rollouts.
- Bugs real hardware exposed and fixed, each with a regression test: per-clip features padded with
  size 1 on the mel axis; a pre-existing per-token width heuristic padded the 128 mel bins whenever a
  prompt was exactly 128 tokens long (GRPO crashed at step ~228); keyword-only call passed positionally.

GPU checks (`test_preference_trainer_audio_dpo.py`, real weights): audio-declared DPO step-0 loss
0.693147 = ln 2; after one update the same pairs give 0.371 with real speech vs 0.479 with silence.

## SP-2: preference training (DPO family) on audio pairs

Pairs (`build_pairs.py`): 8 sampled + 1 greedy transcript per `earnings_train6000` clip from the SFT
model; chosen = lowest WER, rejected = highest; ties, runaways and empty references dropped and counted.
5,604 pairs from 6,000 clips (mean per-utterance WER: chosen 0.109, rejected 0.428).

| run | best validation WER (checkpoint) | verdict on validation |
|---|---|---|
| DPO, beta 0.1, lr 1e-6, 1 epoch | 14.87 (final; step 87 = 22.22) | worse: deletions doubled at step 87 |
| RPO (DPO + NLL on chosen, sft_weight 1.0), lr 5e-7, 1 epoch | 14.08 (final) | tie with SFT |
| **RPO, sft_weight 1.0, lr 1e-6, 2 epochs** | **13.89 (step 525)** | winner |

Plain DPO fits the preference perfectly (loss 0.69 -> 0.06, accuracy 1.0) while transcription degrades:
the margin grows by pushing every transcript down. The NLL anchor (RPO) prevents it.

Held-out test (2,503 rows), paired bootstrap vs SFT:
- RPO 1 epoch: 10.67, [-0.50, +0.01] (not significant).
- **RPO 2 epochs, step 525: 10.65, [-0.49, -0.06]**; runaway rows 0, repetition-loop rows 0.

**SP-2 met**: preference training on audio improves held-out WER over the SFT model. Disclosure: two SP-2
candidates were decoded on the test set (the 1-epoch winner, then the 2-epoch winner), so the second
result carries one extra look.

## SP-3: GRPO with a WER reward (in progress)

All runs: 4 clips x 8 samples per step, `earnings_train6000`, runaway and loop penalties 1.0.

| run | validation WER per checkpoint | verdict |
|---|---|---|
| dr_grpo, lr 1e-6, 400 steps | 14.59 / 14.10 / 14.16 / 14.16 | no change: policy ratio stayed 1.0 |
| dr_grpo, lr 5e-6, 800 steps | 18.71 / 41.16 / 31.51 / 23.78 | collapse; training reward fell |
| dr_grpo, lr 2e-6, 800 steps | 14.66 / 15.31 / 16.33 / 15.61 | worse than SFT |

No GRPO run has beaten the SFT model yet. `dr_grpo` is reference-free, so nothing anchors the policy
once updates are large enough to matter; a `grpo` run with its KL-to-reference term (lr 3e-6) is next.
