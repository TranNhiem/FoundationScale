# Speech P2: the speech-plane gates adjudicate a real audio run

P2 of `docs/research/speech.md`. Same hardware run shape as `../speech_p1` (FoundationScale
`train()`, Gemma-4-E4B full fine-tune, one GB200 GPU, 5 steps, 64 LibriSpeech dev-clean rows,
`../speech_p1/run_smoke.sh`), now on commit 1009381, which adds the speech gates and the post-save
speech adjudication. Measured 2026-10-06. Verdict: **PASS**.

| Gate | Verdict | Checked / expected | Detail |
|---|---|---|---|
| speech.audio_row_coverage | PASS | 12 / 12 audio rows | every collated row loaded, no refusals |
| speech.audio_placeholder_coverage | PASS | 12 / 12 audio rows | placeholder count equal to the processor's own count on every row; 0 unmeasured |
| speech.tower_movement `model.audio_tower` (exercised) | PASS | 271 / 271 parameters | 187 moved |
| speech.tower_movement `model.embed_audio` (exercised) | PASS | 1 / 1 | 1 moved |
| speech.tower_movement `model.vision_tower` (dormant control) | PASS | 210 / 210 | 0 moved |
| speech.tower_movement `model.embed_vision` (dormant control) | PASS | 1 / 1 | 0 moved |

The six results are also recorded in the run manifest under `extra.speech_gates`, and the
transformers `processor_kwargs` warning seen in P1 no longer appears (0 occurrences).

## Caveats

- Row coverage counts rows COLLATED, not rows trained: 5 steps of batch 2 is 10 rows, and the
  dataloader prefetched one more batch (12). The two construction-time survival-probe rows are
  excluded.
- The speech gates are registered on the SAVE event so the controls walk certifies their
  MUST_FIRE/MUST_PASS fixtures; the generic save sweep therefore lists them as SKIP (no context),
  and the run prints a line saying they are adjudicated afterwards with their contexts. A
  dedicated lifecycle event would remove that, but `gates/core.py` is the frozen contract.
- Adapter (LoRA) runs and sharded (DCP) saves report tower movement as UNMEASURED: base-vs-saved
  digests are not comparable there. LoRA scope for the audio tower is the next step.
