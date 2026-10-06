# Speech P1: FoundationScale trains Gemma-4-E4B's audio tower from a declared audio column

Hardware check for P1 of `docs/research/speech.md`. Unlike P0, this runs FoundationScale's own
`train()` (commit 48ff0fa, `torchrun -m foundationscale.train`, launched by `run_smoke.sh`), with
`FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN=audio`, on one GPU of a GB200 tray in the thin-plane container
(transformers 5.5.0). Measured 2026-10-06.

Verdict: **PASS** (run exit 0, save gates PASS), and the audio tower moved.

## Run

| Item | Value |
|---|---|
| Model | Gemma-4-E4B-it, full fine-tune, bf16 |
| Data | 64 LibriSpeech dev-clean utterances (2 to 15 s, 16 kHz), JSONL rows `{text: "Transcribe this audio exactly.", answer: <transcript>, audio: <flac path>}` |
| Steps | 5, per-device batch 2, lr 1e-5 (linear decay), max sequence length 1024, dp=1 |
| Loss by step | 14.93, 9.58, 10.13, 8.21, 10.96 |
| Dormancy announcement | `model.vision_tower, model.embed_vision present on the checkpoint but not exercised by this run's declaration (image_column=None, audio_column='audio')` -- the audio tower is no longer listed as dormant |
| Save gates | PASS 4/4 at checkpoint-5; `checkpoint.save_complete` 2130/2130 tensors; precision bf16 verified over 2130 tensors |

## Per-tower movement, final checkpoint vs base (`tower_movement.py`)

| Bucket | Parameters moved / total | Buffers (never train) |
|---|---|---|
| audio_tower | **187 / 271** | 480 |
| embed_audio | **1 / 1** | 0 |
| language_model | 512 / 719 | 0 |
| vision_tower | **0 / 210** | 448 |
| embed_vision | **0 / 1** | 0 |

Before this change the same audio tower was carried dormant: `docs/VERIFICATION_MATRIX.md`
T1-20/T1-21 recorded "audio 0 of 751" (271 parameters + 480 buffers; see
`../speech_p0/EVIDENCE.md`). The undeclared vision tower staying at exactly 0 of 210 is the
negative control: declaring audio exercises the audio tower and nothing else.

## Caveats

- 84 of 271 audio-tower parameters did not change after 5 steps at lr 1e-5 in bf16. That is
  consistent with the bf16 rounding measured in P0 (updates below one bf16 ulp round away); it is
  not evidence of a frozen subset, and a longer run or fp32 master weights would settle it.
- The 2-row survival probe at collator construction also feeds the collator's coverage counters;
  coverage is not yet written to the run manifest (a P2 item), so no reported number is affected.
- transformers prints `Kwargs passed to processor.__call__ have to be in processor_kwargs dict`
  once per batch. Padding still took effect (batches of 2 collated and trained); routing the
  kwargs the way transformers 5.5 asks is a P2 cleanup.
- This is a mechanism check (5 steps, n=64), not an accuracy claim. WER before and after training
  is P3.
