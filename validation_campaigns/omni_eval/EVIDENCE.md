# Omni speech-LLM evaluation: O0 cascade baseline on VoiceBench (2026-10-12/13)

Plan phase O0 (docs/research/tts_omni_plan.md): the cascade every end-to-end omni model must
beat, measured with the same harness that will score Qwen3-Omni in O1.

## Harness

`voicebench_run.py` wraps the upstream VoiceBench repo (commit b561541; data `hlt-lab/voicebench`,
Apache-2.0): FS assistants subclass upstream's `VoiceAssistant` (loaded from `src/models/base.py`
by path, because importing the package pulls every upstream model's dependencies); the
generation loop mirrors upstream `main.py`'s record shape; scoring is upstream `evaluate.py`,
unmodified, run as a subprocess in its own venv (`cluster/build_vb_eval.sh`). Only the
objectively scored subsets are used (the others need a GPT judge): OpenBookQA and MMSU
(`mcq`), IFEval (`ifeval`), AdvBench (`harm`), BBH (`bbh`).

Cascade (assistant `cascade`): Qwen3-ASR-1.7B (language identification left on, as in its own
evaluation) -> Qwen3-4B-Instruct-2507 (bf16, greedy, max_new_tokens 2048, upstream's cascade
system prompt from `src/models/naive.py`) -> text. Per-row ASR and LLM latency go to a sidecar file.

## O0 results (one GB200 tray, 2 GPUs)

| subset | score (upstream evaluator) | rows | failed | mean ASR s | mean LLM s |
|---|---|---|---|---|---|
| OpenBookQA | acc **75.4** (unparsable answers 0.9%) | 455 | 0 | 2.83 | 10.82 |
| IFEval | final **79.7** (strict-prompt 75.1, loose-instruction 84.1) | 345 | 0 | 2.20 | 21.58 |
| AdvBench | refusal rate **99.8%** | 520 | 0 | 0.84 | 4.88 |
| BBH | acc **88.3** | 1,000 | 0 | 2.91 | 24.39 |
| MMSU | not finished: the run was stopped when its GPUs were released | 3,074 | - | - | - |

Bugs found while building it, each one a silent wrong result: the local Qwen3-4B weights folder
was empty (downloaded); ASR features were not cast to the model's bf16; transformers 5.x returns a
BatchEncoding from `apply_chat_template`; upstream's score is logged to stderr, which was being
discarded; `max_new_tokens` was ignored (upstream's time-to-first-token measurement passes 1).
After these runs the generation loop was made resumable (per-row progress file), because the
interrupted MMSU run lost 8 hours of generation; the loop's outputs are unchanged and the resume
path is unit-tested (failed and missing rows are re-run, finished rows reused, order kept).
