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

## Not covered yet

- Vietnamese: `run_wer.py` supports en and zh only. The MiniMax multilingual set (CC BY-SA 4.0) has
  99 Vietnamese sentences and two reference voices but no ground-truth target audio, so the
  Vietnamese judge is validated separately on FLEURS-vi test recordings (CC BY 4.0) - plan T0b.
