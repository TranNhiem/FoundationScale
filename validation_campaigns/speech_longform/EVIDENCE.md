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

## Training-side fix: long segments (30-40 s)

`build_earnings_long.py` joins consecutive train segments of one call (ordered by `start_ts`)
into 1,456 examples: 14.5 h, 36 s on average, 5.1 segments each, from all 119 non-test calls (459
test-call segments excluded). Model L is arm A's recipe plus these segments: 1,500 steps, clips up
to 40 s (`nemo_finetune.py --max-duration 40`). Census 19,456/19,456, adjudication PASS.

| whole calls, 40 s chunks | base | arm A (clips up to 25 s) | **L (+ long segments)** |
|---|---|---|---|
| 4432298 | 21.1 | 25.4 | 23.1 |
| 4450488 | 16.1 | 13.9 | **12.9** |
| 4470290 | 19.6 | 19.8 | **16.0** |
| 4479741 | 18.1 | 23.2 | 21.1 |
| 4483338 | 17.3 | 14.5 | **11.8** |
| 4485244 | 14.9 | 10.7 | **8.1** |
| **all 6 calls** | 17.30 | 16.77 | **14.47** |
| loop words | 0 | 1,428 | 1,164 |
| `speech.repetition_loops` | -- | FAIL | **FAIL** |

On clips L matches A on Earnings (11.35 vs 11.33) and dev-clean (1.58 vs 1.48), and is a little
worse on AMI (14.33 vs 13.58).

Long targets recover most of the long-form gain: 4 of 6 calls are clearly better than both base and
A. They do **not** remove the loops. The same two calls still loop (1,164 words), and the loop
gate correctly still blocks the model. Next: find what those two calls share. Both have converted
audio (24 kHz mono and 16 kHz stereo), and the fine-tune also emits fillers that base does not,
so look at where in those calls the loops start.

## Why those two calls loop, and decoding

`loop_chunks.py` transcribes the two looping calls chunk by chunk with model L: 14 of 159 chunks
loop. Loudness does not explain it: some looping chunks are loud speech (-22 dBFS, 10% silence).
What the looping chunks share is heavily disfluent speakers ("um so so ...", "in uh in a scenario
where where ..."). Immediate word repeats per 1,000 words: LibriSpeech train 4.6, AMI train 4.2,
**Earnings train 15.4**, Earnings test references 13.4. The fine-tune learned to transcribe
repetitions, which these references contain, and base Canary never does. Under greedy decoding a
repeat makes the next repeat more likely, and on a disfluent speaker it does not stop. A separate
failure: an all-silent chunk (-114 dBFS) came out as "1 1 1 ...".

Beam search (`decoding.strategy=beam decoding.beam.beam_size=4`, applied to every model):

| whole calls, beam 4 | WER | loop words (greedy -> beam 4) | `speech.repetition_loops` |
|---|---|---|---|
| base | 17.36 | 0 -> 0 | -- |
| A (clips up to 25 s) | 15.48 | 1,428 -> 172 | PASS (limit 252) |
| **L (+ long segments)** | **12.81** | 1,164 -> 285 | **FAIL** (limit 252) |

Beam 4 removes 80-88% of the loops and the silence artifact. L reaches 12.81 on whole calls
(-4.55 points against base, the best long-form result). It still exceeds the declared loop limit,
with "the" runs on call 4432298 and "in" runs on 4479741, so the gate correctly blocks it. The
threshold is a declared policy and is not loosened to admit a model.

## Thinning repeated words in the training text

`thin_repeats.py` caps runs of one word in the Earnings training transcripts (train + long
segments); the audio is untouched. keep 2: 195 + 224 tokens dropped, about 300 rows changed.
keep 1: every immediate repeat removed. Same recipe as L otherwise.

| whole calls | greedy WER / loop words | beam 4 WER / loop words | `speech.repetition_loops` (beam 4) | Earnings clips |
|---|---|---|---|---|
| base | 17.30 / 0 | 17.36 / 0 | -- | 19.38 |
| L | 14.47 / 1,164 | 12.81 / 285 | FAIL | 11.35 |
| **T2** (keep 2) | 14.73 / 1,242 | **12.36 / 153** | **PASS** | 11.67 |
| T1 (keep 1) | **13.30 / 524** | 12.37 / 174 | **PASS** | 11.82 |

- **T2 with beam 4 is the first long-form model that passes both gates:** 12.36 on whole calls,
  -5.0 points against base. It costs about 0.3 points on clips.
- **T1 halves the loops even under greedy decoding** (524 vs 1,164) and improves greedy WER to
  13.30. It is the clearest evidence that repetition-heavy transcripts drive the loops.
- Caution: T2 barely changes the training text, so L -> T2 under beam (285 -> 153 loop words)
  may partly be run-to-run variation. Each arm is one seed; seeds are needed before the keep-2
  effect is claimed. The T1 greedy effect is large and goes the expected way.

## Seeds: L vs T2 (beam 4, whole calls)

`nemo_finetune.py --seed` (seeds Lightning and the lhotse shuffle). Each cell: WER / loop words.

| | seed 0 (above) | seed 1 | seed 2 | mean WER | mean loop words | passes `repetition_loops` |
|---|---|---|---|---|---|---|
| L | 12.81 / 285 | 12.35 / 195 | 12.75 / 518 | 12.64 | 333 | 1 of 3 |
| **T2** | 12.36 / 153 | **11.99 / 45** | 12.96 / 270 | **12.44** | **156** | **2 of 3** |

Earnings clips: L 11.35 / 11.48 / 11.58, T2 11.67 / 11.79 / 11.73. Capping repeats at 2
costs about 0.2-0.3 points on clips, consistently.

- **Capping repeats at 2 halves the loops**: T2 has fewer loop words than L at every seed. The
  WER gain (-0.2 on average) is inside the seed spread.
- **Loops are seed-sensitive** (45 to 518 loop words for one recipe), and neither recipe passes
  every seed. The gate is what makes that safe: a looping seed is blocked whatever its WER. The
  best run, T2 seed 1, scores 11.99 on whole calls (base 17.36) with 45 loop words and passes
  both gates. A long-form recipe should select among seeds or checkpoints under the gate, not
  trust one run.

## Selecting under the gates on held-out calls (dev6), then test once

`dev6`: 6 more whole calls (5.6 h, `build_longform.py --dev 6`), none of them test calls. Their 408
training rows were removed from every training manifest (`filter_calls.py`). Ten runs (T2 and L,
seeds 1-5, 1,500 steps, snapshots at 500/1000) give 30 candidates. `select_under_gates.py` scores each
on dev6 with beam 4: whole-call WER plus both speech gates against base, and picks the lowest WER
among the candidates that pass. Test is never seen during selection.

- Base dev WER 22.51. **10 of 30 candidates pass.** Every 500-step snapshot fails (591-4,443 loop
  words) and most 1,000-step ones fail, so **loops are worst early and fall with training**.
  Final checkpoints pass for L in 2 of 5 seeds and for T2 in 3 of 5 (mean loop words 397 vs 317).
  With 5 seeds the keep-2 advantage is weaker than with 3; mostly the seed decides.
- Chosen: **Ls1 final, dev WER 17.24 with 8 loop words.**

Test, once (beam 4, whole calls):

| | WER | loop words | `speech.repetition_loops` |
|---|---|---|---|
| base | 17.36 | 0 | -- |
| **Ls1 (chosen on dev6)** | **13.00** | **460** | **FAIL** |

Ls1 is better than base on 5 of 6 calls (14.9 -> 7.3, 18.1 -> 11.1, 17.3 -> 12.1), but it **loops on
call 4432298** (25.9 WER, length ratio 1.09), the disfluent speaker that trapped earlier models.
**Choosing loop-free models on held-out calls does not carry over to new calls**: loops are
triggered by particular speakers, and a dev set without such a speaker cannot screen for them.
Selection lowers the risk; the gate still catches what it misses. A robust long-form recipe
needs a decoding-side guard that every model gets (a cap on identical-token runs at decode time),
not only training and selection.

## Decoding-side guard: cap identical-token runs at 3

A declared output filter, applied identically to every model: a run of one token is kept up to 3
(the references contain genuine triples, about 1.3 per 1,000 words). The gates keep judging the
**raw** decoder output; otherwise the guard would hide the loops they exist to catch. The guarded
WER is reported separately, as the product-level number.

| test calls, beam 4 | raw WER | cap-3 guarded WER | `speech.repetition_loops` (raw) |
|---|---|---|---|
| base | 17.36 | 17.36 | -- |
| Ls1 (selected on dev6) | 13.00 | **12.18** | FAIL (460) |

**Long-form recipe:** 30-40 s training segments + beam 4 + selection under the gates on held-out
calls + the cap-3 guard. On whole held-out Earnings-22 calls it scores **12.18 WER against base
17.36 (-5.2 points)**. The raw-output gate stays the disclosure that this model still loops on
one disfluent speaker. Fixing that in the decoder (a repetition-aware beam search) remains open.
