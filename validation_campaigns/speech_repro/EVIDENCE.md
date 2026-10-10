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
