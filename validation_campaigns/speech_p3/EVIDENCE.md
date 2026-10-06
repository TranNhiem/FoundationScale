# Speech P3: Gemma-4-E4B ASR fine-tuning beats the base model on held-out audio

P3 of `docs/research/speech.md`. FoundationScale `train()` on a GB200 tray, Gemma-4-E4B, LibriSpeech
train-clean-100 (28,539 utterances of 1 to 25 s, 100.6 h), 500 steps at global batch 16.
Held-out set: 300 dev-clean utterances (0.57 h; dev-clean speakers are disjoint from
train-clean-100), greedy decoding, scored by `foundationscale.train.speech_metrics` (micro-averaged
WER/CER after the versioned normaliser) with `eval_wer.py`. Measured 2026-10-06.

Pre-declared criterion: held-out WER after training <= 0.8 x base WER, speech gates PASS, run PASS.
Verdict: **PASS** for both arms.

| Model | WER | x base | CER | Truncated | Speech gates |
|---|---|---|---|---|---|
| Base Gemma-4-E4B-it (no training) | 3.83% | 1.00 | 2.00% | 0 | n/a |
| Full fine-tune, bf16, lr 1e-5, dp=2 x batch 8 | **2.42%** | **0.63** | 0.91% | 0 | PASS |
| LoRA r16 (language model) + audio towers in full, bf16, lr 1e-4, 1 GPU x batch 8 x accum 2 | **2.73%** | **0.71** | 1.22% | 0 | PASS |

## The defect P3 found first

The first three arms (before commit 2c14cf6a) all FAILED the criterion: bf16 full fine-tune 3.96%,
bf16 LoRA 4.18%, fp32 LoRA 5.02%. Two hypotheses were refuted on hardware (language-model prior
erosion: LoRA did not help either; bf16 rounding: fp32 was worse). The tell was the first-step
loss: 15 to 27 through `train()`, against 0.912 for a direct forward on the same batch. Bisected
with `diag_loss.py`: direct forward 0.912, stock HF Trainer 13.92, the model changed by
`Trainer()` construction alone, and the only change was `config.use_cache: None -> False`.

| `config.use_cache` | loss, same batch, base model |
|---|---|
| None (checkpoint default) | 0.912 |
| True | 0.912 |
| False (what `Trainer.__init__` sets) | 13.941 |
| False, `use_cache=True` passed to forward | 0.912 |

Gemma-4-E4B shares key/value states across 18 of its 42 decoder layers
(`num_kv_shared_layers=18`), and its forward is only correct with the cache on. Generation
re-enables the cache, so zero-shot evaluation looked healthy while every training step optimised a
corrupted forward. Of the Gemma-4 checkpoints on the estate only E4B (base and -it) declares KV
sharing; 12B, 26B-A4B and 31B declare 0. The fix (`train/loop.py::_kv_shared_layer_count`) pins
`use_cache=True` after `Trainer()` and refuses gradient checkpointing on such models. On the same
4 rows the first-step loss went 13.92 -> 0.901 through `train()`, and the training losses of the
rerun arms start at 1.18 / 0.82 instead of 25.8 / 27.6.

## Caveats

- n=300 utterances (about 5,500 reference words); the two arms differ by ~0.3 points, within
  sampling noise of each other, and both are well below the criterion.
- Part of the gain is LibriSpeech transcript STYLE (no punctuation, spelled-out forms) learned from
  the training references; the normaliser removes punctuation and case, but not every style
  difference (for example `boy's` vs `boys`).
- Speech gate row coverage under data parallelism is rank 0's view (about half the global rows).
- Text-capability retention after speech fine-tuning was not measured.
- The audio input pipeline is CPU-bound (GPU utilisation often below 30%: audio is decoded in the
  collator, in the main process); throughput is a follow-up.
