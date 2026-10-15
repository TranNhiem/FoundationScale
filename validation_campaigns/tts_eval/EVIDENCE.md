# TTS evaluation harness (plan phase T0, 2026-10-12)

Plan: docs/research/tts_omni_plan.md. Rule 4 of the upstream design ("integrated" means "reproduced")
applies to the judges before it applies to any model: the harness must reproduce the published
scores of the real recordings ("Human" row) before it scores a synthesizer.

## Harness

- `seedtts_score.py` runs the upstream Seed-TTS-eval scoring scripts **unmodified, as subprocesses**
  (seed-tts-eval commit 752f429): `run_wer.py` (Whisper-large-v3 for en, Paraformer-zh for zh) and
  UniSpeech `verification_pair_list_v2.py` (WavLM-large speaker verification). FS owns only list
  building, coverage (missing audio is counted and exits 95, never dropped), refusals (96) and the
  per-row output that `speech_p7/paired_bootstrap.py` reads.
- Test set and SV checkpoint are Google Drive downloads, pinned by sha256 (`cluster/fetch_t0.sh`).
  Splits as read: en 1,088, zh 2,020, zh-hard 400 (the README says ~1,000 / ~2,000).
- Judges run in their own venv (`cluster/build_tts_judge.sh`: official torch 2.11+cu130). Three
  compatibility fixes were needed to run 2023-era judge code, none touching how a score is
  computed: jiwer pinned to 3.0.5 (`run_wer.py` imports `compute_measures`, removed in jiwer 4);
  s3prl pinned (commit ec8064b) in the torch hub cache and marked trusted (an untrusted hub repo
  waits on an interactive prompt, which hung batch jobs); and that pinned s3prl's top-level
  `hubconf.py` narrowed to its own `wavlm_large` function (the full hub imports ~60 upstreams, some
  calling torchaudio APIs removed in 2.9; the SIM path uses only `wavlm_large`).

## T0 result: the judges reproduce the published "Human" row

Ground truth of each split scored with `--gt` (reference: CosyVoice README, Seed-TTS-eval "Human").

| split | rows scored | WER / CER (ours) | published | SIM (ours) | published |
|---|---|---|---|---|---|
| test-en | 1,088 / 1,088 | **2.143** | 2.14 | **73.38** | 73.4 |
| test-zh | 2,020 / 2,020 | **1.254** | 1.26 | **75.50** | 75.5 |

Every row scored, none missing; all four numbers match the published values to rounding. The harness
is accepted for T1 (Qwen3-TTS and VoxCPM2 reproductions).

## T1: reproducing the published Seed-TTS numbers (2026-10-12)

Decode-only, each model in its own worker venv (`cluster/build_tts_qwen.sh`, `build_tts_voxcpm.sh`),
scored by the T0 harness. Tolerance fixed in advance: +-0.3 WER/CER, +-1.0 SIM.

| model | split | ours WER/CER | published | SIM (ours / published) | verdict |
|---|---|---|---|---|---|
| VoxCPM2 | test-en | 1.706 | 1.84 | 75.0 / 75.3 | reproduced |
| VoxCPM2 | test-zh | 0.946 | 0.97 | 79.22 / 79.5 | reproduced |
| Qwen3-TTS-12Hz-1.7B-Base | test-zh | 0.959 | 0.77 | 76.91 / - | reproduced |
| Qwen3-TTS-12Hz-0.6B-Base | test-zh | 1.132 | 0.92 | 76.33 / - | reproduced |
| Qwen3-TTS-12Hz-1.7B-Base | test-en | 1.561 | 1.24 | 71.65 / - | **not reproduced (+0.32)** |
| Qwen3-TTS-12Hz-0.6B-Base | test-en | 4.224 | 1.32 | 70.3 / - | **not reproduced** |

Qwen3-TTS uses its README protocol (bf16, max_new_tokens 2048, checkpoint sampling defaults,
language auto); VoxCPM2's README states no evaluation settings, so its documented cloning call and
defaults were used (recorded in each run's `synth_record.json`).

The two English misses, examined:
- **Apostrophe style.** Upstream `run_wer.py` strips curly quotes (zhon's punctuation set) but
  keeps the ASCII apostrophe, so a reference "aren’t" against a transcript "aren't" is an error.
  Folding both to the ASCII form moves Qwen 1.7B en from 1.561 to 1.441 (and the ground truth from
  2.143 to 2.069); the official comparison keeps the unmodified scorer.
- **One runaway utterance.** Qwen 0.6B en has a single row in a repetition loop ("a pile of p- a
  pile of p- ..." for a sentence about epilepsy, row WER 27.18); because Seed-TTS averages
  per-utterance WER, that one row adds ~2.5 points (without it: ~1.7). Generation is sampled, so
  seed and batching variance runs are the next measurement; the loop is exactly what FS's
  runaway / repetition-loop gates detect on the ASR side.

Variance of Qwen3-TTS-1.7B on test-en (same harness, 2026-10-14): seed 0 batch 8 = 1.561, seed 1
batch 8 = 1.446, seed 0 unbatched = 1.691. The spread (+-0.12) is sampling noise; all three sit
0.2-0.45 above the card's 1.24, about 0.12 of which is the apostrophe effect. So the English gap is
real and quantified, not a bad draw: our harness reads Qwen3-TTS English about 0.3 above the card.

## T0b: the Vietnamese judge (2026-10-13)

`run_wer.py` covers en and zh only, and the MiniMax multilingual set (CC BY-SA 4.0; 99 Vietnamese
sentences, two reference voices) has no ground-truth target audio. So the Vietnamese judge is chosen
on FLEURS-vi test recordings (CC BY 4.0, 857 utterances) with `judge_multilingual.py`: audio decoded
to float32 at 16 kHz, Whisper's multilingual BasicTextNormalizer on both sides, pooled WER.

| judge | rows scored | WER on ground truth |
|---|---|---|
| **Qwen3-ASR-1.7B** (language forced to vi) | 857 / 857 | **7.49** |
| Whisper-large-v3 (vi, long-form timestamps) | 857 / 857 | 10.25 |

Qwen3-ASR is pinned as the Vietnamese judge; 7.49 is the floor synthesized Vietnamese is measured
against. The first version of this scorer had three silent audio bugs (it passed the sample rate in
place of the waveform, left int16 samples unscaled when no resampling was needed, and gave Whisper
no language or long-form setting); all were fixed before this run.
