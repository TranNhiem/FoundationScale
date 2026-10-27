# Pseudo-label sources: refusing audio that overlaps the evaluation sets

`pseudo_label_guard.py` turns teacher decodes of untranscribed audio into ASR pseudo-labels and
refuses a source that overlaps the evaluation sets. Unlabeled public corpora are often cut from the
same broadcasts, calls or podcasts an evaluation set was cut from, and they are segmented
differently, so matching whole sentences misses the overlap. The guard therefore compares word
8-grams of the teacher transcripts with the evaluation references:

- a clip sharing any 8-gram with a reference is dropped;
- a clip sharing >= 3 RARE 8-grams (shared by <= 2 pool clips) is "strong", and its whole recording
  is dropped, because the neighbouring segments are the same recording;
- with a student decode, only clips the teacher and the student transcribe within 3% CER are kept.

Verdict: RED_FLAG (exit 5) when recordings dropped >= 30%, strong clips >= 1%, or kept hours < 50.
Bare 8-gram matches are not a verdict arm: template phrases (dates, greetings, operator scripts) make
them common on broadcast and call audio.

## Controls on Earnings-22 (2026-10-27)

The gold transcripts stand in for a perfect teacher, so only the overlap logic is measured.
References: `ood_earnings223000_conv` (the 6 test calls) and, where noted, `earnings_validation1000`.

| pool | references | strong clips | recordings dropped | kept | verdict |
|---|---|---|---|---|---|
| the test clips themselves | test + validation | 1,948 / 2,504 (77.8%) | 6 / 6 | 0 h | **RED_FLAG** (exit 5) |
| `earnings_train6000` (119 calls, none of them test calls) | test | 5 / 6,000 (0.08%) | 5 / 119 (4.2%) | 11.8 of 12.4 h | OK |
| `earnings_train6000` | test + validation | 8 / 6,000 (0.13%) | 8 / 119 (6.7%) | 11.5 of 12.4 h | OK |

- The contaminated pool is refused on every arm: each test call holds clips with several rare
  8-grams from the references.
- Against the disjoint test calls, 5 training calls are still dropped. Their shared 8-grams are the
  conference operator's script ("if you wish to ask a question please press star one ..."), which
  repeats across calls but is rare inside one pool. That is the guard's false-positive rate on
  this kind of audio: 4.2% of recordings, well under the 30% arm.
- `earnings_validation1000` is drawn from 117 of the same 119 calls as the training clips (the
  SP-4 split holds out calls only for test). Against it, 3 more training calls are dropped, where
  training clips repeat validation sentences at different clip boundaries.

## In use (a Vietnamese medical ASR campaign, 2026-10-26/27)

Teacher and student were two independent ASR models; references were an 82k-clip medical test,
a pilot set, an evaluation-platform set and a development set (internal data, numbers only).

| source | recordings dropped | strong clips | any-8-gram clips | verdict |
|---|---|---|---|---|
| a 966 h public unlabeled medical corpus (672 recordings) | 647 (96%) | 2.0% | 7.8% | refused |
| a 4,331 h radio and podcast pool (18,727 recordings) | 1,872 (10.0%) | 0.47% | 12.9% | accepted |

The refused corpus shared broadcast sources with the test's medical-broadcast tier: training on it
would have trained on test material. On the accepted pool, the any-8-gram matches were spoken dates
("năm hai nghìn không trăm hai mươi lăm") and broadcast greetings; those clips were dropped anyway.
Of the clean audio, 1,040 h passed the teacher/student agreement filter (69%).

WHAT IS NOT CLAIMED: that a pool the guard accepts shares no audio with an evaluation set (overlap
below 8 consecutive words, or with a different transcription, is invisible to it), or that
pseudo-labels the guard accepts improve a model (in the campaign above they did not, under beam-4
decoding).
