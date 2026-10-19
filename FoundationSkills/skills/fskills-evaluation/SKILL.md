---
name: fskills-evaluation
license: MIT
description: 'Scores a trained checkpoint (full HF model dir or PEFT adapter) against its base model on
  judge-free benchmarks with pinned lm-evaluation-harness 0.4.12, fully offline (no downloads), and emits
  a confirm-hashed fs_launch_spec (entry fskills-eval) for GB200. Output: eval_report.json written atomically
  with verdict PASS/RED/UNMEASURED and provenance (harness version, task hash, few-shot, seed, baseline
  fingerprint). Do NOT use for building FS-ready datasets (use fskills-data-engine) or Slurm training
  runs (use fskills-training). For a VLM LoRA checkpoint it also measures held-out loss (NLL, image/video/text
  rows) against the base on the same rows. Not for VLM benchmarks, LLM-judged or RL/agentic evals.'
compatibility: "Requires Python>=3.10 and the foundationskills package (fskills CLI); launches additionally need FoundationScale on a Slurm GPU cluster. Without them the skill reports REFUSED or UNMEASURED and gives the install step instead of improvising."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Score this checkpoint against its base on mmlu and gsm8k","Give me a PASS/RED verdict before promoting this trained model","Evaluate this LoRA adapter offline with lm-eval 0.4.12 and write eval_report.json","Smoke-test the eval setup with --limit 8 on arc_easy","The run exited 5 RED / 95 UNMEASURED / 96 REFUSED, what does that mean","Emit the eval job for GB200 and give me the confirm hash, then launch it","Why is my benchmark unmeasured with ''dataset not staged in eval cache''","Which rule fired EV-IN-005 / EV-HO-001 in my eval policy"]'
---

# evaluation — FoundationSkills checkpoint evaluation

## Purpose

Score a trained checkpoint (a full HF model dir or a PEFT adapter) on judge-free
benchmarks with the pinned **lm-evaluation-harness 0.4.12**, run fully offline,
and judge it against its base model with a one-sided stderr band. The result is
an `eval_report` whose every number is attributable to a harness version,
task hash, few-shot count, seed, chat-template mode and baseline fingerprint.
The skill never implements a scorer and never downloads data or models.

## When to use

- After a CPT/SFT/LoRA run, to get PASS/RED against the base before promoting a checkpoint.
- To smoke-test an eval setup (`--limit N`): the verdict is at best UNMEASURED.
- For a VLM LoRA checkpoint: `heldout` (or `fskills eval run --heldout BLOCK.json`) scores held-out mean NLL for
  base and checkpoint on the same rows with FoundationScale's `examples/vlm-2xH200/scripts/eval_heldout_loss.py`.
  Video rows need `video_frames` equal to training's `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` (above the measured 16-frame
  envelope a passing score is UNMEASURED, EV-HO-009); NLL has no stderr, so
  `abs_epsilon` is required. Like benchmark mode it is a non-regression gate: PASS = not worse than base beyond the band.
- Not for VLM benchmarks (lmms-eval is a deferred follow-up), LLM-judged tasks, or RL/agentic evals.

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `benchmarks` | yes | policy task names, e.g. `["mmlu", "gsm8k"]` |
| `policy` | yes | eval policy YAML (the shipped default is `eval_policy.yaml`) |
| `eval_cache` | yes | pre-staged offline `HF_HOME` (home-scoped on this estate); exported as `HF_HOME` with `HF_*_OFFLINE=1` |
| `out` | yes | path of `eval_report.json` |
| `checkpoint` / `base` / `run_manifest` | one of checkpoint or run_manifest | base order: `base`, then the manifest's `config.model`, then an adapter's local `base_model_name_or_path` |
| `baseline_cache` | no | default `<workdir>/artifacts/eval/baselines` |
| `num_fewshot`, `seed` (1234), `gen_kwargs`, `limit`, `dtype` (bfloat16), `batch_size`, `device`, `parallelize`, `include_path` | no | harness knobs, identical for checkpoint and baseline |

## Outputs

`eval_report.json` (artifact type `eval_report`, written atomically). Top level:
`checkpoint`, `base`, `adapter`, `verdict`, `limited`, `harness`, `policy`
(path/version/fingerprint), `seed`, `dtype`, `chat_template`, `base_identity`,
`harness_dir`. Per benchmark: `name`, `metric`, `score`, `baseline`, `stderr`,
`baseline_stderr`, `n`, `drop`, `threshold`, `band`, `status`
(`pass|red|skipped|unmeasured`), `baseline_provenance` (`measured|cached`),
`fingerprint`, `task_hash`, `argv`, and `error` when not measured.

Exit codes: 0 PASS, 5 RED (a breach or a skip), 95 UNMEASURED (a benchmark
could not run, or a limited run), 96 REFUSED (no report is written).

## Running it on GB200

`fskills eval emit ... --gpus N --python <job interpreter> --spec-out spec.json`
writes an `fs_launch_spec` (entry `fskills-eval`) and prints its confirm hash;
nothing is submitted. One node only: `--nodes` other than 1 makes the spec
non-executable, and N > 1 GPUs shard the model with `parallelize` (any
`device` is dropped). `fskills eval launch --spec spec.json --confirm <hash>`
is the only submission path; it first runs `eval run --dry-run` (inputs check,
no report, exit 0 or 96) and refuses to submit if that fails.

## Speech long-form suite

`fskills eval run --suite speech-longform --manifest CALLS.jsonl --base-predictions BASE.jsonl --tuned-predictions
TUNED.jsonl --out REPORT.json` scores whole-call ASR for a base and a tuned model from their NeMo output manifests
(rows plus `pred_text`). It composes FoundationScale's own `speech_metrics` (normalisation, WER cross-checked against
FS's aligner, loop words, runaway rows) and `speech_gates` (`RunawayHypothesisGate`, `RepetitionLoopGate`), exactly as
the measured campaign scorer does. A gate FAIL is RED whatever the WER (EV-HO-007): measured, the best-WER decode
(12.81%) carried 285 loop words and was blocked. A call without a prediction is UNMEASURED (EV-HO-008). No harness, no
eval cache. Producing the predictions (NeMo chunked inference) is outside this suite.

## Scope

LLMs of any family, any training stage, full or adapter checkpoints, on
GB200/H100/local. HF backend only (no vLLM, no API backends).

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses before any work - status REFUSED, exit 96, nothing written. A **handoff**-phase BLOCK does not refuse: the work ran, and the result is RED (exit 5), never PASS. WARN and INFO findings never block; they travel with the result (EV-HO-003 makes the run UNMEASURED, exit 95). Phase by prefix: EV-IN-* rules are input phase, EV-HO-* rules are handoff phase.

| Rule | Severity | Fires when |
|---|---|---|
| EV-IN-001 | BLOCK | checkpoint (or the run manifest naming `<output_dir>/final`) is missing |
| EV-IN-002 | BLOCK | no local base model resolves (a named base that does not exist is not skipped over) |
| EV-IN-003 | BLOCK | lm-eval (benchmark) mode: `eval_cache` not given or not a directory (held-out and suite modes read only local files) |
| EV-IN-004 | BLOCK | `lm_eval` is absent or not 0.4.12 |
| EV-IN-005 | BLOCK | policy invalid, a benchmark has no policy entry, needs a judge, or a stderr-less metric has no explicit `abs_epsilon` |
| EV-HO-001 | BLOCK | `drop > max(abs_epsilon, k*sqrt(se_b^2+se_c^2))` (improvements never fail) |
| EV-HO-002 | BLOCK | the harness completed but reported no metric for a configured benchmark (a skip) |
| EV-HO-003 | WARN | a benchmark is unmeasured (dataset not staged, harness error) or the run was limited |
| EV-HO-004 | BLOCK | a PASS report is missing/altered on disk, limited, or has a null score |
| EV-IN-006 | BLOCK | held-out `data` missing/unreadable, or `eval_heldout_loss.py` missing (never a silent fallback) |
| EV-IN-007 | BLOCK | held-out rows carry video but `video_frames` is unset |
| EV-IN-008 | BLOCK | `video_frames` disagrees with the run manifest's `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` |
| EV-IN-009 | BLOCK | `--suite speech-longform`: the manifest or a prediction file is missing, unreadable or cannot carry a WER |
| EV-RU-001 | BLOCK | vacuous held-out run (`rows_used == 0` for either role): REFUSED, no report |
| EV-HO-005 | BLOCK | base and checkpoint `rows_used`/`rows_dropped` disagree: the delta is UNMEASURED (95), never RED |
| EV-HO-006 | BLOCK | a held-out row lacks `rows_used`/`rows_dropped`/`score`/`baseline`: RED |
| EV-HO-007 | BLOCK | speech-longform: a gate FAILED (runaway or repetition loops), whatever the WER: RED |
| EV-HO-008 | WARN | speech-longform: calls without a prediction (undercovered): UNMEASURED, never PASS |
| EV-HO-009 | WARN | held-out video rows scored at `video_frames` above the measured frame envelope (16 per clip): a passing row is UNMEASURED (95), never PASS; a regression stays RED |

## Failure handling

- Dataset not staged: the benchmark is `unmeasured` with `missing input: dataset <id> not staged in eval cache <dir>`; the harness is not started.
- Harness error: the benchmark is `unmeasured`; the report carries the log tail and `harness_dir` holds `harness.log`.
- Baseline cache: reused only when the stored fingerprint matches; a limited run never seeds it.
- Policy cannot judge a result (no stderr, no explicit epsilon): REFUSED, no report.

## FS interface

Consumes FS `run_manifest.json` (`config.model.value`, `config.output_dir.value`,
checkpoint at `<output_dir>/final`). Runs `lm-eval run --model hf` as a
subprocess, one task per call, with an argv list and an offline environment.

**Host portability (A1.5).** On a host where the `fskills` CLI or the `foundationskills` package is missing, do not improvise a plan or simulate a result: report REFUSED with the install step (`pip install -e <FoundationScale repo>/FoundationSkills`, or run where it is installed), and report every check that could not run as UNMEASURED, never PASS.

## Worked examples

```bash
# LoRA SFT run -> verdict against its base, from the run manifest.
PYTHONPATH=../src python -m foundationskills.cli eval run \
  --run-manifest runs/fskills_r3/train_q27/fskills-sft/run_manifest.json \
  --benchmarks mmlu,gsm8k --policy foundationskills/skills/evaluation/eval_policy.yaml \
  --eval-cache ~/.cache/fskills_eval_hf --out artifacts/eval/q27-sft/eval_report.json

# Smoke slice (never PASS): 8 examples per task.
PYTHONPATH=../src python -m foundationskills.cli eval run --checkpoint <ckpt> --base <base> \
  --benchmarks arc_easy --policy <policy> --eval-cache <cache> --limit 8 --out <dir>/eval_report.json
```
