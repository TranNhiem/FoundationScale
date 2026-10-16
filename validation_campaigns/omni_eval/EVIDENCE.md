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
| MMSU | acc **70.5** (unparsable answers 1.4%) | 3,074 | 0 | 3.46 | 15.40 |

Bugs found while building it, each one a silent wrong result: the local Qwen3-4B weights folder
was empty (downloaded); ASR features were not cast to the model's bf16; transformers 5.x returns a
BatchEncoding from `apply_chat_template`; upstream's score is logged to stderr, which was being
discarded; `max_new_tokens` was ignored (upstream's time-to-first-token measurement passes 1).
After these runs the generation loop was made resumable (per-row progress file), because the
interrupted MMSU run lost 8 hours of generation; the loop's outputs are unchanged and the resume
path is unit-tested (failed and missing rows are re-run, finished rows reused, order kept).
The MMSU rerun (2026-10-15/16) resumed from its progress file and was finished on two GPUs with
`--rows`: a second GPU took rows 2420-3074 into its own progress file, and a final pass merged both,
generated nothing missing, and scored all 3,074 rows. Qwen3-Omni (O1) leads the cascade on MMSU by
6.3 points (76.8 vs 70.5).

## O1: Qwen3-Omni-30B-A3B-Instruct (2026-10-13/14)

Assistant `qwen3_omni`: mirrors upstream VoiceBench's Qwen2.5-Omni assistant (same system prompt,
audio-only user turn, `disable_talker()`, `return_audio=False`); greedy decoding, max_new_tokens 2048.
Card values from the Qwen3-Omni README (30B-A3B-Instruct column).

| subset | ours | card | gap | O0 cascade |
|---|---|---|---|---|
| OpenBookQA | 92.5 | 89.7 | +2.8 | 75.4 |
| IFEval (final) | 81.5 | 77.8 | +3.7 | 79.7 |
| AdvBench (refusal %) | 99.4 | 99.3 | +0.1 | 99.8 |
| BBH | 85.0 | 80.4 | +4.6 | 88.3 |
| MMSU | 76.8 (unparsable 3.3%) | 68.1 | +8.7 | (stopped) |

All rows generated, none failed, scored by upstream's evaluator. Our scores sit 2.8-8.7 points above
the card on four of five subsets; at n=455-1,000 that is systematic, not sampling noise. The card does not
publish its decoding settings (we decode greedily), so this is recorded as a documented upward gap,
not as "reproduced". The end-to-end model beats the 4B cascade on OpenBookQA and IFEval; the
cascade still wins BBH and AdvBench.

## O2 starting point: Qwen3-Omni already hears Vietnamese (2026-10-15)

Qwen3-Omni as an ASR judge on FLEURS-vi test (`judge_multilingual.py --asr qwen3omni`: prompt
"Transcribe the Vietnamese audio into text.", talker disabled, greedy): **5.44 WER** on 857/857
rows, better than the pinned Vietnamese judge Qwen3-ASR-1.7B (7.49). So O2's planned target -
teaching the omni model Vietnamese speech input - has little headroom on read speech; the O2 target
is being re-scoped before any training is spent on it.
