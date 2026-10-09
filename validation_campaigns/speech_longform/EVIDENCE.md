# Long-form ASR: whole Earnings-22 calls (Canary-1B-flash)

Hardware: GB200, one GPU per model. Date: 2026-10-09.

## Set

`build_longform.py`: the 6 Open ASR Leaderboard Earnings-22 test calls, as whole calls with their
full transcripts, from `distil-whisper/earnings22` (config "full"): 5.6 h, 50,400 normalized
reference words. They do not overlap the Earnings training data (`speech_p8` excluded exactly these
calls). The source metadata says 16 kHz, but 4 calls are 24 kHz, 1 is 22.05 kHz stereo and 1 is
16 kHz stereo. All were converted at build time and recorded per call. libmpg123 reported one
corrupt MP3 frame in the source audio, and decoding continued past it.

Inference: NeMo's own `speech_to_text_aed_chunked_infer.py` (non-overlapping chunks).
Scoring: `score_longform.py`, using FoundationScale's normalizer and a rolling-row edit distance
that is cross-checked against `speech_metrics.align` on a slice of every call, plus the
runaway gate.

## Results (WER %, whole calls)

| call | hours | base, 40 s | Earnings FT (speech_p8 arm A), 40 s | base, 20 s | FT, 20 s |
|---|---|---|---|---|---|
| 4432298 | 0.73 | 21.1 | 25.4 (len 1.06) | 21.0 | 26.6 |
| 4450488 | 1.21 | 16.1 | **13.9** | 16.6 | 20.6 |
| 4470290 | 0.45 | 19.6 | 19.8 | 19.3 | 16.6 |
| 4479741 | 1.02 | 18.1 | 23.2 (len 1.09) | 18.3 | 23.1 |
| 4483338 | 1.16 | 17.3 | **14.5** | 17.3 | 16.6 |
| 4485244 | 1.02 | 14.9 | **10.7** | 15.1 | **9.4** |
| **all calls** | 5.6 | **17.30** | **16.77** | 17.43 | 18.13 |

The runaway gate PASSES in every run (no call reaches 2x the reference length).

## What it means

- The clip-level gain (Earnings-22 clips 19.4 -> 11.3, `speech_p8`) **does not transfer to whole
  calls**. The 40 s total is -0.5 points, made of 3 calls clearly better (-2.2 to -4.2) and 2
  clearly worse (+4.3, +5.1). With 20 s chunks the fine-tune is worse overall.
- Refuted: **chunk length.** The fine-tune trained on clips of at most 25 s, so 40 s chunks were
  a suspect, but 20 s chunks are no better.
- Refuted: **transcript conventions.** The full-call and clip references both contain about 4.3%
  fillers ("uh", "um"). The base model emits none (0.0%) and is charged a deletion for each; the
  fine-tune emits 5.1%.
- **Found: local repetition loops.** Words inside runs of one token repeated 4 or more times:

  | | 4432298 | 4450488 | 4470290 | 4479741 | 4483338 | 4485244 | total |
  |---|---|---|---|---|---|---|---|
  | base, 40 s and 20 s | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
  | fine-tune, 40 s | 393 | 4 | 0 | 1019 | 8 | 4 | **1428** |
  | fine-tune, 20 s | 420 | 470 | 4 | 818 | 459 | 8 | **2179** |

  The fine-tune gets stuck on one token inside a chunk ("the the the ...", "uh uh uh ...",
  "in in in ..."). More chunks give more chances to loop, which is why 20 s chunks are worse.
- **The loops explain the whole regression.** Diagnostic only (post-processing output is not a
  fix): with every run of 4 or more collapsed to one token, the fine-tune is better than or equal
  to base on all 6 calls, **17.30 -> 14.14 WER** (call 4479741: 23.2 -> 12.1, against base 18.1).
  The Earnings fine-tune does help whole calls; a decoding failure on long chunks hides it.
- **Framework gap:** `speech.runaway_hypotheses` passed every run. It judges whole rows (more than
  2x the reference), and a loop inside one chunk of an hour-long call adds only 4-12% words. Long
  form needs a local loop measure (loop words per call, against base) as a gate.
- **Now a gate: `speech.repetition_loops`** (`speech_metrics.loop_words`, runs of 4 or more):
  it blocks when `tuned > 2 x base + ceil(0.5% x reference words)`. On real outputs: the 40 s
  long-form fine-tune FAILS (1,428 against a limit of 252) where the runaway gate passed it; the
  clip-level fine-tune PASSES (151 against 240). It would also flag the old LS+AMI mix on Earnings
  clips (507 against 240), a loop defect the runaway gate missed. `score_longform.py` and
  `speech_eval/compare.py` report both gates.
- **Next:** (1) done above; (2) a training-side
  fix: segments built from consecutive clips of a call, so the model sees 30-40 s targets.
  Decoding-side repetition penalties stay out of the measured comparison unless the base gets
  them too.
