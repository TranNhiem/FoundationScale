# Rule 4: reproducing the model cards (2026-10-10)

A model or technique is `supported` only after FoundationScale reproduces the upstream reference
result on the same test set the upstream reports, scored by our harness, with the normalizer
difference reported (docs/research/upstream_integration.md).

Reference numbers come from each card's structured `model-index`. Only three speech models publish
a LibriSpeech result there; Whisper-large-v3, Qwen2-Audio-7B-Instruct and Gemma-4-E4B-it publish
none and stay `experimental`.

Runs: the full splits (`build_librispeech.py`, no duration cap: test-clean 2,620 utterances, test-other
2,939, up to 35 s), through the NeMo worker modules (`foundationscale.upstream.nemo.decode`,
`salm_decode` with 256 new tokens) and the HF lane (`speech_p4/eval_wer.py`). Scored by
`score_repro.py` with both normalizers. Tolerance fixed in advance at 0.3 WER points.

| model | split | model card | ours, Whisper normalizer | gap | ours, FS normalizer | status |
|---|---|---|---|---|---|---|
| Canary-Qwen-2.5B | test-clean | 1.61 | 1.624 | +0.014 | 1.801 | **supported** |
| Parakeet-CTC-1.1B | test-clean | 1.83 | 1.846 | +0.016 | 2.054 | **supported** |
| Canary-1B-flash | test-other | 2.87 | 2.874 | +0.004 | 3.084 | **supported** |

All three reproduce within 0.02 points. FoundationScale's normalizer is stricter than Whisper's and
reads about 0.2 points higher on every model; FS numbers elsewhere in the evidence use the FS
normalizer and are comparable with each other, not with model cards. These runs are the
regression tests for future upstream upgrades (Level 3 of the upgrade process).

# The first model adopted through the upgrade process: Qwen3-ASR-1.7B (2026-10-10)

The release tracker (Phase 5) surfaced Qwen/Qwen3-ASR-1.7B (Apache-2.0). Its class needs
transformers >= 5.13 and the HF lane runs 5.5, so it went through the full upgrade process:

1. **Candidate profile** `hf-cand-519`: the hf-26.04 container plus a venv with transformers 5.19.0,
   torch unchanged. peft and accelerate are not visible in it, so it is decode-only.
2. **Regress** under the candidate: Parakeet-CTC-1.1B, the supported HF model, reads 1.839 on
   test-clean against its card's 1.83 (gap +0.009). FS's HF eval is unaffected by transformers 5.19.
3. **Reproduce** with `eval_qwen3_asr.py` (bf16, greedy, max_new_tokens 1024, no forced language,
   Qwen's published setting). The reference is the README results table; the card has no `model-index`.

| split | Qwen README | ours, Whisper normalizer | gap | ours, FS normalizer |
|---|---|---|---|---|
| test-clean (2,620) | 1.63 | 1.643 | +0.013 | 1.883 |
| test-other (2,939) | 3.38 | 3.368 | -0.012 | 3.660 |

Reproduced on both splits, so the registry entry `qwen3-asr-1.7b` is `supported`. It is
decode-only, with no FamilySpec. Training support waits for the HF lane to move to a transformers
version that carries the class.

The first attempt read 11.4 / 14.2 because every hypothesis kept the model's
`language English<asr_text>` header. In transformers 5.19 only `Qwen3ASRProcessor.decode` honours
`return_format="transcription_only"`; `batch_decode` silently ignores it. The script now decodes
row by row. This is the "integrated means reproduced" rule doing its job: the model loaded and
produced fluent text, and only the comparison against the published number exposed the bug.
