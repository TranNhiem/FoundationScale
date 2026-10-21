# Speech continued pre-training, SP-1 (Qwen3-ASR-0.6B, 2026-10-19/21)

ROADMAP SP-1 done-when: CPT lowers held-out loss and, after the usual SFT, matches or beats the
SFT-only model on WER. No trainer change was needed: FS runs CPT through the same causal-LM objective
as SFT (the stage emitter documents this), so SP-1 is campaign scripts only (`run_sp1.sh`).

## Recipe: noisy-student CPT on weakly labelled in-domain audio

- Pool (`build_cpt_pool.py`): every Earnings-22 robust-train segment on disk that the SFT run never
  used and that is not from a test call. 14,880 seen, 6,000 excluded (earnings_train6000), 761
  excluded (test calls), 1 skipped (near-empty segment, which wrote a 0-byte FLAC before the fix):
  **8,118 clips, 15.96 h, from 119 training calls**, 0 unaccounted.
- Weak labels (`make_cpt_manifest.py`): a teacher model transcribes the pool; the real transcripts are
  never used for training. Empty, looping and speaking-rate-implausible outputs are dropped and
  counted, and an incomplete teacher decode is refused.
- CPT: 1 epoch on the 0.6B base (`train.sh`, batch 16). Then the SP-4 SFT recipe (earnings_train6000,
  750 steps, the fixed validation rule), and the held-out test with a paired bootstrap vs SFT-only.
- Held-out loss (`heldout_loss.py`): teacher-forced cross-entropy of the transcript + eos on
  earnings_validation1000, with the audio features in the forward (the first draft dropped them).

## Results

| round | teacher (WER of its labels vs real) | CPT rows | held-out loss base -> CPT | best validation WER after SFT (SFT-only 14.09) | held-out test (SFT-only 10.93) |
|---|---|---|---|---|---|
| 1 | Qwen3-ASR-1.7B zero-shot (27.2%) | 8,092 | 1.052 -> 0.463 | 14.37 (ckpt 500) | 11.09, paired [-0.17, +0.55] |
| 2 | RPO -> GRPO 0.6B, SP-2/3 stack (21.0%) | 8,102 | 1.052 -> 0.463 | 14.13 (ckpt 500) | 10.98, paired [-0.19, +0.32] |

**Verdict.** Criterion 1 is met: CPT lowers held-out loss in both rounds. Criterion 2 is not shown as a
gain: after SFT, CPT neither beats nor detectably differs from SFT-only (round 2: 95% CI within about
+-0.3 WER of zero). By the fixed validation rule SFT-only stays the winner in both rounds. Cleaner weak
labels narrowed the gap (validation 14.37 -> 14.13, test 11.09 -> 10.98), so label quality matters,
but 16 h of weakly labelled in-domain audio does not add accuracy on top of SFT here. Larger
unlabelled pools are the obvious next lever: only 13 of the 45 Earnings-22 train shards are on disk.

Operational notes: a colleague's job took the GPU between stages once (the pool lock binds pool users
only), and one launch failed silently and went unnoticed for two days. The stages are restartable,
and both rounds were rerun to completion.
