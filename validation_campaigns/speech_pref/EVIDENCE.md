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

## SP-3: GRPO with a WER reward

All runs: 8 samples per clip, `earnings_train6000`, runaway and loop penalties 1.0; 4 clips per step unless noted.

| run | validation WER per checkpoint | verdict |
|---|---|---|
| dr_grpo, lr 1e-6, 400 steps | 14.59 / 14.10 / 14.16 / 14.16 | no change: policy ratio stayed 1.0 |
| dr_grpo, lr 5e-6, 800 steps | 18.71 / 41.16 / 31.51 / 23.78 | collapse; training reward fell |
| dr_grpo, lr 2e-6, 800 steps | 14.66 / 15.31 / 16.33 / 15.61 | worse than SFT |
| grpo (KL to reference), lr 3e-6, 800 steps | 17.25 / 16.19 / 18.18 / 17.85 | worse; training reward fell 0.79 -> 0.76 |

No GRPO run has beaten the SFT model. Falling training reward made a bug the first suspect, so the
implementation was checked by an overfit test (60 steps on 4 fixed clips, lr 3e-6): reward rose
0.76 -> 0.98 and its spread shrank 0.13 -> 0.03. The audio rollout, scoring and update are therefore
correct -- gradients do raise reward on what is trained. On the full data the problem is
optimization: 4 clips (32 rollouts) per step gives updates that help some clips and hurt others,
which matches a training reward that falls as the learning rate rises.

**Large batch** (16 clips = 128 rollouts per step, `grpo` with its KL term, lr 1e-6, 200 steps, log-probs
scored in micro-batches of 16): training reward now rises steadily (0.79 -> 0.83 over 200 steps).

| checkpoint | 50 | 100 | 150 | 200 |
|---|---|---|---|---|
| validation WER (SFT 14.09) | **13.85** | 13.92 | 16.47 | 48.82 |

Step 50 wins the fixed rule. Later checkpoints collapse while the training reward keeps rising: at step
200, deletions jump from 666 to 7,424. The policy learned to stop early under greedy decoding, which the
per-rollout sampled reward never sees. So the reported checkpoint is the early one, picked on validation.

Held-out test (2,503 rows), paired bootstrap vs SFT: **step 50 = 10.57, [-0.58, -0.15]**. Repetition-loop
rows: 0. One row is flagged by the runaway detector, where SFT has none: reference "Yeah." (1 word),
RL "Yeah, the only thing I had to say." (8 words), SFT "Yeah, the only thing I heard." (6 words). Both
models transcribe the same background speech; 8 words crosses the detector's 2x+5 bound and 6 does
not. It is not a degenerate loop.

**SP-3 met**: RL with a WER reward improves held-out WER over the SFT model, with no repetition loops
and one borderline runaway flag (above). The method needs a large batch and early stopping on
validation. Beyond ~100 steps it learns to truncate.

## Seed replicates (2026-10-18)

The two reported recipes were rerun with seeds 1 and 2 (`SEED=` in `run_dpo.sh` / `run_grpo.sh`). Each
replicate used the same fixed validation rule and the same held-out test, with a paired bootstrap vs
SFT (10.93).

| recipe | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| SP-2 RPO, 2 epochs (validation winner: step 525 every seed) | 10.65 [-0.49, -0.06] | 10.56 [-0.59, -0.15] | 10.53 [-0.63, -0.18] | **10.58** |
| SP-3 GRPO + KL, 16 clips x 8 (winner: step 50 / 150 / 50) | 10.57 [-0.58, -0.15] | 10.37 [-0.76, -0.36] | 10.48 [-0.65, -0.25] | **10.47** |

All six runs beat SFT on the held-out test with a 95% CI that excludes zero. The four replicates
show 0 runaway and 0 repetition-loop rows. This also settles the two caveats above: SP-2's extra look
at the test set, and SP-3's single borderline runaway flag. SP-3 seed 2 collapsed after step 50
(validation 28.87 at step 100), the same truncation as seed 0, while seed 1 stayed near SFT on
validation (14.00-14.15) and still scored best on test. Early stopping on validation is part of the
recipe, not an option.
