# Omni speech-LLM on GB200: speech in, answers out (VoiceBench)

This example builds and scores two speech-in, text-out assistant stacks on a GB200 tray and reports their VoiceBench scores side by side: a two-model cascade (ASR + chat LLM) and one end-to-end omni model. Everything below is marked **VERIFIED** (measured on 2026-10-12 – 2026-10-14 on one GB200 tray, 2 GPUs) or **PLANNED** (not run). Related example: [../speech-2xH200/](../speech-2xH200/).

**What you get:**
- Two working assistant arms behind one runner: `validation_campaigns/omni_eval/voicebench_run.py` ("VoiceBench O0/O1 runner (local parquet in, upstream evaluate.py scores)").
- Measured VoiceBench numbers for both arms on GB200, plus the published model card numbers for comparison (Step 4).
- A cluster launcher (`cluster/run_vb.sh`), a sequential lane driver (`cluster/run_o0_lane.sh`) and a CPU evaluator venv build script (`cluster/build_vb_eval.sh`).
- A resumable generation loop: an interrupted subset continues instead of regenerating finished rows.

No fine-tuning has been run anywhere in this example. All three models are used at inference only.

---

## Two system types: cascade vs end-to-end omni

| | O0 `cascade` | O1 `qwen3_omni` |
|---|---|---|
| Chain | `Qwen3-ASR-1.7B` → `Qwen3-4B-Instruct-2507` → `text` | `Qwen3-Omni-30B-A3B-Instruct` → `text` |
| Model count | 2 (ASR + LLM) | 1 |
| Processing | one ASR transcript, then one chat turn | audio handed straight to the model |
| Decoding | greedy, `max_new_tokens 2048` | greedy, `max_new_tokens 2048` |
| Output | text | text (talker disabled: `disable_talker()`, `return_audio=False`) |
| Bigger subset wins here | BBH, AdvBench | OpenBookQA, IFEval (and MMSU, which the cascade never finished) |

**Cascade (`cascade`)** — INFERENCED PAIR, training status: no fine-tuning. `Qwen3-ASR-1.7B` is a dedicated ASR front end run in transcription-only mode with its own language identification left on, "as in its own evaluation"; `Qwen3-4B-Instruct-2507` is a text-only chat LLM run in bf16 with greedy decoding. Pick this when you want a small stack where you can read the intermediate transcript and swap either half independently.

**End-to-end (`qwen3_omni`)** — SINGLE END-TO-END OMNI MODEL (MoE), training status: no fine-tuning. `Qwen3-Omni-30B-A3B-Instruct` takes the audio directly (audio-only user turn, talker disabled), loaded in `dtype=torch.bfloat16`. Pick this when you want one model for speech in → answer out and no ASR transcript at all; here it beats the 4B cascade on OpenBookQA and IFEval but loses BBH and AdvBench.

**When to use each:** Use the cascade when a text-only LLM is already good enough and you want an auditable transcript step. Use the end-to-end omni model when you want the audio to reach the language model untouched and can carry a 30B-A3B footprint. Measured head-to-head, "the end-to-end model beats the 4B cascade on OpenBookQA and IFEval; the cascade still wins BBH and AdvBench".

Speech out is not covered: `O1` uses `disable_talker()` and `return_audio=False` ("text out (talker disabled)"), and no speech-out evaluation results are present.

---

## Step 1 — Environments

Two separate environments, both under `<path>` on the cluster.

**Runtime (generation):** enroot container `fs-g4e4b-nemo-automodel-26-04_compute`, started with `PYTHONNOUSERSITE=1`, `CUDA_VISIBLE_DEVICES`, `HF_HUB_OFFLINE=1`, `ENROOT_DATA_PATH=<path>/.local/share/enroot`; the python interpreter that runs `voicebench_run.py` is `<path>/envs/hf-cand-519/bin/python`.

**Evaluator venv (CPU)** — `cluster/build_vb_eval.sh`, verbatim (cluster paths written `<path>`):

Usage: `bash cluster/build_vb_eval.sh`

This script builds the CPU evaluator venv at `<path>/envs/vb-eval`, on CPU at the cluster `<path>`, installing only what `src/evaluator` imports (mcq, ifeval, harm, bbh) and locking the result in `requirements.lock` under that venv directory.

**Weights and data locations** (as used by `cluster/run_vb.sh`):

| what | path |
|---|---|
| ASR weights | `<path>/pretraining_weights/Speech-Models/Qwen3-ASR-1.7B-hf` |
| LLM weights | `<path>/pretraining_weights/Alibaba-Qwen/qwen3/Qwen3-4B-Instruct-2507` |
| VoiceBench data / code / runs | `<path>/datasets/speech/voicebench` (`data`, `code`, `runs`) |
| Repository | your `FoundationScale/` checkout (run commands from its root) |

All generation runs go through `HF_HUB_OFFLINE=1`; nothing is downloaded at run time. Keep `PYTHONNOUSERSITE=1` in every shell (it is set in the enroot command line): shared `~/.local` packages otherwise leak into the venv and the container.

### Models and techniques

- `Qwen3-ASR-1.7B` — INFERENCED DEDICATED ASR MODEL (front end of the cascade). It runs the qwen3_asr API (`AutoProcessor.apply_transcription_request`, row-wise, `language=None`, `return_format="transcription_only"`), so you pick it when you want a small transcription stage with an auditable transcript.
- `Qwen3-4B-Instruct-2507` — INFERENCED TEXT CHAT LLM, bf16, greedy, `max_new_tokens 2048`, upstream's cascade system prompt. You pick it as the second half of the baseline "every end-to-end omni model must beat".
- `Qwen3-Omni-30B-A3B-Instruct` — INFERENCED END-TO-END OMNI MODEL (`Qwen3OmniMoeForConditionalGeneration` + `Qwen3OmniMoeProcessor`, bf16, talker disabled, greedy). You pick it to get speech in → answer out from a single model.

### Check it worked

The evaluator env prints its own sanity line at the end of the build:

```
evaluators ['bbh', 'harm', 'ifeval', 'mcq']
```

(The build script prints `evaluators` plus the sorted evaluator names it can import.)

---

## Step 2 — VoiceBench data and subsets

Upstream VoiceBench repo at commit `b561541`; data `hlt-lab/voicebench`, Apache-2.0. The snapshot lives at `<path>/datasets/speech/voicebench` (`data`, `code`, `runs`). Audio is read from LOCAL parquet: column `"audio"` is `struct{bytes, path}`, decoded with soundfile, mixed to mono, resampled to 16 kHz (`TARGET_SR = 16_000`) and handed to the assistant as `{"array": float32 ndarray, "sampling_rate": 16000}`. "Most fixture audio is 16 kHz hex-native, so this is normally a no-op."

Only the objectively scored subsets are run — "the others need a GPT judge". The subset → evaluator mapping is `EVALUATOR_BY_SUBSET` in `voicebench_run.py`:

| subset (CLI choice) | upstream evaluator | metric — VERIFIED (as written in the measured tables) | rows |
|---|---|---|---|
| `openbookqa` | `mcq` | acc (unparsable answers %) | 455 |
| `mmsu` | `mcq` | score (unparsable %) | 3,074 (rows, published) |
| `ifeval` | `ifeval` | final (strict-prompt, loose-instruction) | 345 |
| `advbench` | `harm` | refusal rate % | 520 |
| `bbh` | `bbh` | acc | 1,000 |

Metric provenance: the metric names above are those written in the measured results; `voicebench_run.py` itself records only `eval_stdout` (upstream evaluator output) and latency means. Scoring is upstream `evaluate.py`, unmodified, so "the score is over exactly the expected row set; any failed row still scores".

Latency is split per row: means go into the summary JSON, raw per-row values into a sidecar file. `fs_*` metrics go to the sidecar "NOT into the scored file".

---

## Step 3 — Run

### Top-level launcher — `cluster/run_vb.sh`, verbatim (cluster paths written `<path>`)

Usage: `bash cluster/run_vb.sh <GPU> <ASSISTANT> <SUBSET> [LIMIT]` (recorded as `usage: run_vb.sh GPU ASSISTANT SUBSET [LIMIT]`, one GPU per call)

It runs one subset of one assistant arm through `voicebench_run.py` in the enroot container `fs-g4e4b-nemo-automodel-26-04_compute`, with `PYTHONNOUSERSITE=1`, `CUDA_VISIBLE_DEVICES` and `HF_HUB_OFFLINE=1` set.

The GPU guard refuses (`REFUSE gpu $G busy $U`, `exit 96`) unless the chosen GPU has under 1000 MB in use. Console/log output goes to `$V/runs/$A-$S.log` in `<path>/datasets/speech/voicebench/runs`.

### Exact CLI of `voicebench_run.py` (argument parser, verbatim)

`validation_campaigns/omni_eval/voicebench_run.py` runs one VoiceBench subset through the chosen assistant arm (either the ASR + chat-LLM cascade or the end-to-end Qwen3-Omni model) on local parquet audio and then scores the produced JSONL with upstream `evaluate.py`.

| flag | meaning | default/required |
|---|---|---|
| `--assistant` | which arm to run; `ASSISTANT_ARMS` = `("cascade", "qwen3_omni")` | required, those two choices |
| `--subset` | VoiceBench subset to score (`EVALUATOR_BY_SUBSET` CLI choices) | required: `openbookqa`, `mmsu`, `ifeval`, `advbench`, `bbh` |
| `--data-dir` | VoiceBench data directory (parquet) | required |
| `--vb-code` | upstream VoiceBench code directory that holds `evaluate.py` | required |
| `--eval-python` | "python that runs evaluate.py" | required |
| `--asr-model` | "Qwen3-ASR path" | default `None`; required for `cascade` |
| `--llm` | "chat LLM path (cascade)" | default `None`; required for `cascade`; for `qwen3_omni` it is the Qwen3-Omni checkpoint |
| `--out-dir` | directory for the run outputs | required |
| `--limit` | cap on rows processed | default `None` |
| `--device` | device to run on | default `cuda:0` |

Constraint enforced by the runner: "`cascade needs --asr-model and --llm`"; it exits otherwise. For the `qwen3_omni` arm, "For this arm `--llm` is the Qwen3-Omni checkpoint; `--asr-model` is unused", and `--llm` must be `Qwen3-Omni-30B-A3B-Instruct`.

Example (the `cascade` invocation, paths as recorded):

```
<path>/envs/hf-cand-519/bin/python validation_campaigns/omni_eval/voicebench_run.py --assistant cascade --subset <subset> --data-dir <path>/datasets/speech/voicebench/data --vb-code <path>/datasets/speech/voicebench/code --eval-python <path>/envs/vb-eval/bin/python --asr-model <path>/pretraining_weights/Speech-Models/Qwen3-ASR-1.7B-hf --llm <path>/pretraining_weights/Alibaba-Qwen/qwen3/Qwen3-4B-Instruct-2507 --out-dir <path>/datasets/speech/voicebench/runs
```

### Cascade run (O0) — VERIFIED usage

Usage: `bash cluster/run_vb.sh <GPU> cascade <SUBSET> [LIMIT]`

The flags `run_vb.sh` passes to `voicebench_run.py` are exactly the ones in the example command above (`--assistant cascade`, `--subset`, `--data-dir`, `--vb-code`, `--eval-python`, `--asr-model`, `--llm`, `--out-dir`, and optionally `--limit`).

For several subsets on one GPU sequentially:

Usage: `bash cluster/run_o0_lane.sh <GPU> <SUBSET>...`

It runs the O0 cascade baseline over several VoiceBench subsets on one GPU, sequentially, each subset through `run_vb.sh` in the enroot container `fs-g4e4b-nemo-automodel-26-04_compute`, continuing past a failing subset and echoing its exit code.

### Qwen3-Omni run (O1)

Same CLI with `--assistant qwen3_omni`. `voicebench_run.py` states: "For this arm `--llm` is the Qwen3-Omni checkpoint; `--asr-model` is unused", and `--llm` must be `Qwen3-Omni-30B-A3B-Instruct`. The rest of the flag set is identical to the cascade list above (data-dir, vb-code, eval-python, out-dir); `run_vb.sh`'s recorded body hardcodes the cascade model paths into `--asr-model`/`--llm`, and the recorded files contain no separate O1 launcher script.

The assistant mirrors upstream VoiceBench's Qwen2.5-Omni assistant: "same system prompt, audio-only user turn, `disable_talker()`, `return_audio=False`"; greedy decoding, `max_new_tokens 2048`. The system prompt (verbatim, upstream `src/models/naive.py` audio-path):

> "You are a helpful assistant who tries to help answer the user's question. Please note that the user's query is transcribed from speech, and the transcription may contain errors."

### Constants and the generation loop

| constant | value | note |
|---|---|---|
| `TARGET_SR` | `16_000` | resample target, 16 kHz |
| `MAX_TRANSCRIPTION_TOKENS` | `1024` | Qwen3-ASR published eval setting |
| `MAX_RESPONSE_TOKENS` | `2048` | upstream base.generate_audio default |

The generation loop mirrors upstream `main.py`'s record shape (one row of non-audio fields plus `"response"`). FS assistant classes subclass upstream's `VoiceAssistant`, "loaded from `src/models/base.py` by path, because importing the package pulls every upstream model's dependencies". Scoring is upstream `evaluate.py`, unmodified, run as a subprocess in its own venv through `--eval-python`, and handed `--src_file` (the scored JSONL path) and `--evaluator` (the evaluator name).

Upstream writes its score through loguru on stderr, so both streams are kept together (`output = result.stdout + result.stderr`).

Outputs per run in `--out-dir`:

| file | content |
|---|---|
| `<assistant>-<subset>-test-audio.jsonl` | scored file (fed to `evaluate.py`) |
| `<assistant>-<subset>-test-audio.fs.jsonl` | sidecar: `fs_*` metrics and raw per-row latency |
| `<assistant>-<subset>.eval.txt` | upstream evaluator output (`eval_stdout`) |
| `<assistant>-<subset>.summary.json` | row counts, failed rows, latency means |
| `<jsonl>.partial` | transient per-row progress file, removed at the end |

Exit codes: clean `0`; run with failed rows `95`; refused scoring `96` (run_vb.sh also exits `96` on a busy GPU).

### Resume

The generation loop is resumable via a per-row progress file, added after the interrupted MMSU run "lost 8 hours of generation". Every finished row is appended and flushed to `<jsonl>.partial`; on restart, rows without `fs_error` are reused (`done`) and the others are re-run; the final JSONL is written in row order and the progress file removed (`partial.unlink()`). "The loop's outputs are unchanged and the resume path is unit-tested (failed and missing rows are re-run, finished rows reused, order kept)." Re-running the same `run_vb.sh …`/`voicebench_run.py …` command to continue an unfinished subset.

### Honesty rules baked into the runner

- `rows_expected = parquet rows after --limit`; "every number is measured, nothing assumed".
- "A row whose generation raises is recorded with response `""` and counted in `failed` and is never dropped from the scored file … any failed row still scores and the run exits 95".
- "If upstream `evaluate.py` itself exits non-zero, the run is refused (exit 96) with the stderr tail — it is better to fall back to `score_phase.py` with `thousands_fs_calibration.json` than to publish a number a broken scorer produced." (`REFUSED scoring:` on stderr.)

---

## Step 4 — Measured results vs card (VERIFIED)

### O0 cascade — VoiceBench, 2026-10-12/13, "one GB200 tray, 2 GPUs"

| subset | score (upstream evaluator) | rows | failed | mean ASR s | mean LLM s |
|---|---|---|---|---|---|
| OpenBookQA | acc **75.4** (unparsable answers 0.9%) | 455 | 0 | 2.83 | 10.82 |
| IFEval | final **79.7** (strict-prompt 75.1, loose-instruction 84.1) | 345 | 0 | 2.20 | 21.58 |
| AdvBench | refusal rate **99.8%** | 520 | 0 | 0.84 | 4.88 |
| BBH | acc **88.3** | 1,000 | 0 | 2.91 | 24.39 |
| MMSU | **NOT FINISHED** — "the run was stopped when its GPUs were released" | 3,074 (rows, published) | – | – | – |

### O1 end-to-end `Qwen3-Omni-30B-A3B-Instruct` — 2026-10-13/14, vs the model card (`30B-A3B-Instruct` column)

| subset | ours | card | gap | O0 cascade |
|---|---|---|---|---|
| OpenBookQA | 92.5 | 89.7 | +2.8 | 75.4 |
| IFEval (final) | 81.5 | 77.8 | +3.7 | 79.7 |
| AdvBench (refusal %) | 99.4 | 99.3 | +0.1 | 99.8 |
| BBH | 85.0 | 80.4 | +4.6 | 88.3 |
| MMSU | 76.8 (unparsable 3.3%) | 68.1 | +8.7 | (stopped) |

**Decoding caveat.** Both arms decode greedily (`do_sample=False`, `max_new_tokens` capped at `MAX_RESPONSE_TOKENS = 2048`, "upstream base.generate_audio default"). Relative to the card this is a documented gap in decoding settings, not a reproduction: "The card does not publish its decoding settings (we decode greedily), so this is recorded as a documented upward gap, not as 'reproduced'." And: "Our scores sit 2.8-8.7 points above the card on four of five subsets; at n=455-1,000 that is systematic, not sampling noise."

Hardware note: the O0 numbers are measured on "one GB200 tray, 2 GPUs"; the O1 tray/GPU count is not stated in the recorded files.

Stated technical deviation: the resampler uses `resample_poly` ("not identical to the upstream library's filter, which is a known and stated deviation"). `fs_*` keys are kept out of the scored file because "the only evidence of what an evaluator does with an unknown key is in evaluator source that is not among the wrapped files"; "keeping the fed file byte-shaped as upstream produce makes the score comparable to any upstream run". For `O1` the sidecar rows carry `fs_transcript: ""`, `fs_asr_seconds: 0.0` (no separate ASR stage).

---

## Fine-tuning: starting point measured, training not started

- `O0` (cascade) and `O1` (Qwen3-Omni) are done as evaluations; see `docs/research/tts_omni_plan.md`.
- **O2 starting point (VERIFIED, 2026-10-15):** before teaching Qwen3-Omni Vietnamese speech input, its existing ability was measured. Prompted with "Transcribe the Vietnamese audio into text." (talker disabled, greedy), it reaches **5.44 WER** on all 857 FLEURS-vi test rows, better than the dedicated Vietnamese judge Qwen3-ASR-1.7B (7.49). There is little headroom on read speech, so the O2 training target is being re-scoped before any GPU time is spent on it. This is the point of measuring first.
- The cascade's MMSU run is being re-run with the resumable runner; its row stays "not finished" until it lands.
- Speech-out evaluation is not recorded: `O1` runs text out with the talker disabled (`disable_talker()`, `return_audio=False`).

---

## Troubleshooting

**Bugs already found in this example — "each one a silent wrong result":**
1. "the local Qwen3-4B weights folder was empty (downloaded)" — check the weights path under `pretraining_weights` before a long subset run.
2. "ASR features were not cast to the model's bf16" — wrong results, no crash.
3. "transformers 5.x returns a `BatchEncoding` from `apply_chat_template`" — both transformers 4.x `torch_dtype` and 5.x `dtype` names are probed rather than assumed (`_load_bf16`).
4. "upstream's score is logged to stderr, which was being discarded" — the runner now keeps both streams.
5. "`max_new_tokens` was ignored (upstream's time-to-first-token measurement passes 1)" — fixed; code comment: "max_new_tokens is honoured: upstream generate_ttft times a call with max_new_tokens=1".

**Launch refused — `REFUSE gpu $G busy $U`, `exit 96`:** the GPU has 1000 MB or more in use. Pick another GPU number as the first argument of `run_vb.sh`.

**Scoring refused — `REFUSED scoring:` on stderr, `exit 96`:** upstream `evaluate.py` exited non-zero. It is better to fall back to `score_phase.py` with `thousands_fs_calibration.json` than to publish a number a broken scorer produced.

**Run finished with `exit=95` in the log:** some rows raised during generation. They are kept in the scored file with response `""` and counted in `failed`; re-running the same command re-runs those rows only.

**A subset was interrupted (e.g. MMSU stopped "when its GPUs were released"):** re-run the same command. Rows without `fs_error` are reused from `<jsonl>.partial`; failed and missing rows are re-run, finished rows reused, order kept.

**Score looks too good to be a reproduction:** for O1 that is expected and documented — greedy decoding here vs unpublished card decoding settings (see the decoding caveat in Step 4). Also check the unparsable rate recorded per subset (OpenBookQA O0 0.9%, MMSU O1 3.3%; no unparsable figure is given for other rows).

**Evaluator import failures:** rebuild the CPU venv with `cluster/build_vb_eval.sh`; it installs only what `src/evaluator` imports (mcq, ifeval, harm, bbh) and locks with `$V/requirements.lock`.