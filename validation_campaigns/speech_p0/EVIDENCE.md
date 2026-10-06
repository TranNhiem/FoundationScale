# Speech P0: can Gemma-4-E4B's audio tower train through the HF API?

Phase P0 of `docs/research/speech.md`. No FoundationScale code is involved: a plain HF probe
(`probe.py`, launched by `run_p0.sh`) runs on one GPU of a GB200 tray inside the thin-plane
container. Raw record: `results.json` (cluster paths pseudonymised as `<CLUSTER_HOME>`).

Verdict: **PASS** (exit 0), measured 2026-10-06.

| Step | Measured |
|---|---|
| Environment | transformers 5.5.0, torch 2.11.0a0 (nv26.02), CUDA 13.1, one NVIDIA GB200, bf16 |
| Data | 4 LibriSpeech dev-clean utterances (2 to 15 s), 16 kHz native, equal to the feature extractor rate; no resampling |
| Class resolution | `AutoModelForCausalLM` on this checkpoint builds `Gemma4ForConditionalGeneration`; the audio tower is present (271 tensors, 304,824,608 of 7,996,156,448 parameters) |
| Prompt route | `processor.apply_chat_template` accepts `{"type": "audio"}` content directly; batch carries `input_features` [4, 1248, 128] and `input_features_mask` [4, 1248] |
| Placeholder coverage | audio placeholder tokens per sample [147, 121, 312, 248] equal the audio tower's valid output length per sample [147, 121, 312, 248] (forward hook) |
| Labels | only the assistant transcript is supervised: [28, 19, 47, 39] tokens; prompt and audio positions are -100 |
| Zero-shot ASR (before any update) | WER [0.059, 0.400, 0.031, 0.042], mean 0.133: audio reaches the language model |
| One training step | loss 17.48, finite; 271 of 271 audio-tower tensors receive a non-zero gradient |
| Movement (SGD, lr 1e-2) | audio tower 220 of 271 tensors changed; language model 614 of 677 |

## What this settles for P1/P2

- The thin plane already loads the audio tower. The audio gap is the declaration, data and
  freezing surface in `train/loop.py`, not model construction.
- Gemma-4 needs no FoundationScale-side feature extraction or placeholder splicing: the HF
  processor owns both, and its placeholder count agrees with the tower's output length. The
  placeholder-coverage gate should still measure that equality per sample, because it is a
  property of the processor/model pair, not a constant.

## Caveats

- **bf16 rounding hides small updates.** With bf16 weights and no fp32 master copy, an SGD step
  of lr 1e-5 rounds to no change for most tensors. The movement check therefore used lr 1e-2 as a
  mechanism probe, not as a training setting. The 51 audio-tower tensors with a non-zero gradient
  that did not change are consistent with that rounding; the gradient census, not the movement
  count, is the primary signal here.
- Denominator reconciliation with `docs/VERIFICATION_MATRIX.md` T1-20/T1-21 ("audio 0 of 751"):
  the checkpoint's safetensors header holds 751 `audio_tower` tensors = 271 parameters
  (258 `weight`, 12 `per_dim_scale`, 1 `bias`) + 480 clipping buffers (`input_min`,
  `input_max`, `output_min`, `output_max`, 120 each). Buffers never train, so a movement gate
  over checkpoint tensors must use 271 as its expected count for "moved", or report buffers as
  a separate bucket; otherwise a fully trained tower reads as 271 of 751.
- n=4 utterances. This is a mechanism probe, not an accuracy claim.
