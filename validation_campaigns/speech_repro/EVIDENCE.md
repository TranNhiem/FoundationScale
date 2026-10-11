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
   torch unchanged. A `.pth` exposes the container's `/opt/venv` after the venv's own packages, so
   peft 0.18.1, accelerate 1.11.0 and everything else are hf-26.04's.
2. **Regress** under the candidate: Parakeet-CTC-1.1B, the supported HF model, reads 1.839 on
   test-clean against its card's 1.83 (gap +0.009). FS's HF eval is unaffected by transformers 5.19.
3. **Reproduce** with `eval_qwen3_asr.py` (bf16, greedy, max_new_tokens 1024, no forced language,
   Qwen's published setting). The reference is the README results table; the card has no `model-index`.

| split | Qwen README | ours, Whisper normalizer | gap | ours, FS normalizer |
|---|---|---|---|---|
| test-clean (2,620) | 1.63 | 1.643 | +0.013 | 1.883 |
| test-other (2,939) | 3.38 | 3.368 | -0.012 | 3.660 |

**Training-lane regression** (Parakeet-CTC-1.1B, 30 steps, seed 0, batch 8, train-clean-100, one
GPU, `artifacts/speech/cand519_train`): both profiles PASS every gate (audio rows 248/248; tower
movement 1352 vs 1350 of 1566). A same-seed rerun is bit-identical within each profile. Across
profiles the per-step loss differs by at most 0.007, already at step 1 (0.2597 vs 0.2666, before
any update), so this is a deterministic change in transformers' forward numerics, not
nondeterminism. It is small and decode still reproduces the card, so it is accepted and recorded.

Whisper-large-v3 under both profiles (2026-10-11, `artifacts/speech/whisper_ab`): the P4 recipe
(500 steps, 1 GPU x batch 8, lr 1e-5, seed 0, `FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en`), each
checkpoint scored by `speech_p4/eval_wer.py` on full dev-clean (2,683 rows). Both read 2.25 -> 1.69,
with a paired difference of +0.00 [-0.02, +0.02]; losses agree to three decimals (0.343 at step
10, 0.042 at step 250). A Whisper regression reported from a transformers 5.18 / torch 2.13 /
peft 0.21.2 VM (2.25 -> 3.34) is therefore not caused by transformers alone.

Reproduced on both splits, so the registry entry `qwen3-asr-1.7b` is `supported`. It is
decode-only, with no FamilySpec. Training support waits for the HF lane to move to a transformers
version that carries the class.

The first attempt read 11.4 / 14.2 because every hypothesis kept the model's
`language English<asr_text>` header. In transformers 5.19 only `Qwen3ASRProcessor.decode` honours
`return_format="transcription_only"`; `batch_decode` silently ignores it. The script now decodes
row by row. This is the "integrated means reproduced" rule doing its job: the model loaded and
produced fluent text, and only the comparison against the published number exposed the bug.

# Promotion: hf-26.10 becomes the HF lane (2026-10-11)

The candidate profile was renamed `hf-26.10` and made the HF default (docs/research/upstream_integration.md).
Before promotion:
- **Gemma-4-E4B:** 30 seeded steps, two runs per profile (`artifacts/speech/gemma_ab`). Step 1 is
  identical (0.3488). The largest cross-profile difference (0.022) equals the within-profile
  rerun difference (0.021), and tower movement is 187 vs 186 of 271.
- **Qwen2-Audio-7B:** the FamilySpec was re-measured for the 5.19 layout (`artifacts/speech/q2_lora_check`).
  LoRA r16 with the towers in full, 10 steps: PASS under 5.19 (automatic scope, 224 modules; 88/88
  rows and placeholders; tower movement 473/487 and 2/2); refused at load under 5.5.

Level 3 with `run_levels.sh` under the promoted default (`artifacts/speech/levels_promoted_20261011`,
runner exit 0; HF steps in hf-26.10, NeMo steps in nemo-26.08):

| model | measured | card | gap |
|---|---|---|---|
| Parakeet-CTC-1.1B | 1.839 | 1.83 | +0.009 |
| Canary-1B-flash | 2.874 | 2.87 | +0.004 |
| Canary-Qwen-2.5B | 1.624 | 1.61 | +0.014 |
| Qwen3-ASR-1.7B | 1.643 | 1.63 | +0.013 |

## Follow-up: full saves under transformers 5.19 (2026-10-11)

Found from a 2x H200 report and reproduced on GB200 (`artifacts/speech/q2_fsdp`). transformers 5.19
reverts its load-time layout conversion on save, so a full Qwen2-Audio checkpoint carries
`audio_tower.*` while the model in memory holds `model.audio_tower.*`. FS declared the in-memory
names, so `checkpoint.save_complete` found 0 of 876 and went RED; this hit every full fine-tune of
Qwen2-Audio on 5.19, on any GPU count. `train/saved_names.py` now asks transformers'
`revert_weight_conversion` (the function `save_pretrained` calls) for the on-disk names. The
declaration and the speech tower digests use them; FS copies no mapping table (ledger
`hf-revert-weight-conversion`).

FSDP mixed precision then saves fp32 master weights of the bf16 model, and the dtype-qualified
digests refused the cross-dtype pairs (UNMEASURED). Saved fp32 tower tensors are now narrowed to
their base precision before hashing, and the log line says so. An untouched weight round-trips
exactly, so narrowing can only under-report movement.

Qwen2-Audio-7B full fine-tune, 10 steps:

| run | before | after |
|---|---|---|
| 1 GPU | (save_complete RED, by the same naming) | PASS: tower movement 466/487, 2/2 |
| FSDP 2 GPUs, `--fsdp-state-dict full` | RED (save_complete, first_save) | PASS: 469/487, 2/2; 489 tensors narrowed |
| FSDP 2 GPUs, `--fsdp-state-dict sharded` | tower movement UNMEASURED | unchanged: a DCP save is not read by the speech gates |
