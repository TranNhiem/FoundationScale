# Speech P8: Earnings-domain training data for Canary-1B-flash, and the runaway gate

Hardware: GB200, one GPU per job. Date: 2026-10-08.

## 1. Data: open, and leak-free by construction

SPGISpeech stayed unavailable: the account's token is valid, but the account is not on the
dataset's authorized list. The open alternative is `sanchit-gandhi/earnings22_robust_split`.
That split divides each call's segments across train/validation/test, so all 125 calls are in
every split, including all 6 calls of our Earnings-22 eval (Open ASR Leaderboard test).
`build_earnings_train.py`:

- **excludes every segment of the 6 test calls by `source_id`** (333 training rows and 57
  dev rows refused as `excluded_test_call`) and asserts that no test call reaches a manifest;
- converts non-16 kHz / multi-channel clips at build time with `--convert` (polyphase
  resampling, channel mean). The step is declared, and every row is counted by its original
  format and marked in its manifest row. The training loader still never resamples;
- writes each clip once, to `{split}_{source_id}_{segment_id}.flac`, and refuses a repeated
  path (see section 2).

Train set: 6,000 rows, 12.4 h, all 119 non-test calls. Checkpoint-selection dev set: 1,000
validation rows, 2.1 h, from 117 non-test calls. These are different segments from training but
the same calls, so dev WER is optimistic. Test sets: full dev-clean (2,683), AMI-2000, and
`ood_earnings223000_conv` (2,504 rows, 5.2 h, the 6 test calls, zero call overlap with
training; built by `speech_p5/build_ood.py`, `OOD_LIMIT=3000 OOD_CONVERT=1`). This replaces the
148-row native-16 kHz Earnings set.

## 2. Retraction: the first P8 runs trained on mismatched audio

The first builder named clips `{segment_id}.flac`. `segment_id` is a per-call index, so it
repeats across calls: **6,000 training rows pointed at 820 files**, and most transcripts
trained against another segment's audio. Every first-round result is withdrawn: the "+Earnings
500 steps" row, the "1500-step collapse" (Earnings 43.9%, 122 runaway rows), and the claim that
the decoder learned a domain prior. All of them measured corrupted data. The collapse was a
decoder taught that audio and text are unrelated.

Every FoundationScale gate passed those runs. Row coverage counted 18,000/18,000 rows loaded,
and they did load, just someone else's audio. Two guards now exist:

- `speech_canary/nemo_finetune.py` refuses and counts a repeated audio path
  (`duplicate_audio_path`). The corrected run refused 0 of 18,000.
- `speech.runaway_hypotheses` (section 4) blocks the symptom: on the corrupted run it returns
  FAIL (122 of 2504 against a limit of 19, exit 5).

The bug surfaced because the base model's dev WER was missing as a sanity anchor and the
checkpoint dev numbers exceeded 80%. The base model is now scored on every dev set.

## 3. Results on corrected data (WER %, paired bootstrap, 2,000 resamples)

The three-way mix is LibriSpeech 6k + AMI 6k + Earnings 6k. Training: 1,500 steps, batch 8,
lr 1e-5, warmup 150, bf16-mixed, a snapshot every 250 steps. Arm B froze `transf_decoder` (and
the tied head's bias). Adjudication passed in both arms: 18,000/18,000 rows; encoder 1196/1196
moved; arm A decoder 109/109 moved; arm B decoder bit-identical 109/109.

Dev WER (Earnings validation, base 23.83) falls almost monotonically, and the last step is best
in both arms: arm A 24.9 / 24.2 / 20.5 / 16.5 / 17.9 / **15.1**, arm B 25.0 / 24.7 / 22.5 / 20.0
/ 18.2 / **16.0** (steps 250 to 1500).

| model | dev-clean full | AMI-2000 | Earnings-22 (held-out calls) | runaway rows (Earnings) |
|---|---|---|---|---|
| base | **1.37** | 16.86 | 19.38 | 3 |
| LS+AMI mix, 500 steps | 1.50 | 15.01 | 18.63 | 7 |
| **+ Earnings, 1500 steps (A)** | 1.48 | **13.58** | **11.33** | 3 |
| + Earnings, decoder frozen (B) | 1.51 | 13.42 | 12.29 | 6 |

- **Earnings-22, A vs base: -8.05 points, 95% CI [-8.66, -7.42]**, on calls never seen in
  training. A vs the LS+AMI mix: -7.30, CI [-8.59, -6.29].
- **AMI, A vs base: -3.28, CI [-5.03, -1.26]**. A vs the LS+AMI mix: -1.43, CI [-2.76, -0.07].
- **dev-clean: +0.11, CI [+0.04, +0.18]**, the same small cost as every Canary fine-tune;
  Canary-1B is near its LibriSpeech ceiling. A vs the mix: no difference.
- **Freezing the decoder is not needed.** B matches A on AMI and dev-clean, but is 0.97 points
  worse on Earnings (CI [-1.89, -0.26] for A minus B). On clean data the full fine-tune does
  not collapse.
- `speech.runaway_hypotheses` PASSES for both arms on all three sets (Earnings 3 and 6 against
  a limit of 19; AMI 5 and 4 against 30; dev-clean 0 against 14).

Also corrected: the Canary lane's earlier "LS+AMI mix regresses Earnings (23.4 -> 26.9)" came
from the 148-row subset. On 2,504 rows the mix is -0.75 (not significant). The LS+AMI AMI gain
holds, at -1.86 (300 rows had suggested -3.6).

## 4. Framework change: `speech.runaway_hypotheses`

`train/speech_metrics.py` gains `is_runaway` (normalized hypothesis words > 2 x reference words
+ 5) and `count_runaway`. `gates/speech_gates.py` gains `RunawayHypothesisGate`: it blocks when
`tuned > 2 x base + ceil(0.5% x rows)`. Coverage is rows checked over rows expected, where rows
expected come from the manifest, never from the eval artifact. Its MUST_FIRE control is the
measured 122-vs-3 collapse; its MUST_PASS control is the measured clean mix (7 vs 3).
`runaway_verdict.py` applies it to any base/fine-tune pair of eval files.
