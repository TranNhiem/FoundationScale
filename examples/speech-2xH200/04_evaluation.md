# 04 — Evaluation: WER, base vs fine-tuned and paired bootstrap

Compare the base model and the fine-tuned checkpoint on the **same held-out rows**: same manifest, same scoring, then compared utterance by utterance. Every model below was evaluated with two `eval_wer.py` calls (base and tuned) over the full 2,683-row dev-clean set and one `paired_bootstrap.py` call per pair, on a 2x H200 setup.

Headline result: **all four models improved after fine-tuning**, each confirmed by the paired bootstrap with high confidence (every CI below excludes zero). Getting there required two upstream fixes and one recipe fix; both debugging rounds are recorded in chapter 03.

## 1. Confirm your held-out split is clean

dev-clean speakers are disjoint from train-clean-100 speakers by LibriSpeech's own construction (standard partition), and chapter 02 built both manifests independently from the same filter policy — this is inherited from the dataset, not re-verified here.

## 2. Why >= 2000-row eval sets

**300-row held-out sets cannot resolve WER shifts under about 1 point** — the paired-bootstrap confidence interval doesn't narrow enough with that few utterances (full case in [chapter 02 §4](02_data_preparation.md)). Every number in this chapter is measured on the FULL 2,683-row dev-clean set for exactly that reason: several of the differences below are under 1.5 points, and a 300-row eval would have reported some of them as noise.

## 3. Score WER with eval_wer.py

`eval_wer.py` flags (these are the actual argparse flags):

| flag | default | meaning |
| --- | --- | --- |
| `--model` | *(required)* | model directory to load |
| `--processor` | same as `--model` | tokenizer/preprocessor source |
| `--adapter` | none | LoRA adapter directory |
| `--manifest` | *(required)* | eval manifest (`.jsonl`) |
| `--out` | *(required)* | output JSON |
| `--batch-size` | `16` | batch size |
| `--max-new-tokens` | `128` | generation cap |
| `--limit` | none | score only the first N rows |
| `--prompt` | `"Transcribe this audio exactly."` | instruction prompt |
| `--language` | none | language hint (required for Whisper) |

Two real invocations, base and fine-tuned, for a CTC/full-FT model:

```bash
python validation_campaigns/speech_p4/eval_wer.py \
    --model models/parakeet-ctc-1.1b --manifest data/librispeech/manifests/eval.jsonl \
    --out eval_logs/parakeet_base.json

python validation_campaigns/speech_p4/eval_wer.py \
    --model runs/parakeet-ctc-1.1b/final --processor models/parakeet-ctc-1.1b \
    --manifest data/librispeech/manifests/eval.jsonl --out eval_logs/parakeet_tuned.json
```

Note the `--processor` on the second call. A full-fine-tune `final/` checkpoint directory only contains `config.json` + `model.safetensors` — no tokenizer or preprocessor files — so `--processor` must point at the original base model directory. Omitting it fails trying to load a processor from a directory that doesn't have one.

`eval_wer.py` only ever uses `cuda:0` inside its own process (no multi-GPU awareness), so on a 2-GPU box run base and tuned in parallel, one process per GPU pinned with `CUDA_VISIBLE_DEVICES` (roughly halves wall-clock time for a pair). `--adapter`/`--processor` flags carry over unchanged per call; just vary `CUDA_VISIBLE_DEVICES` and the output path.

```bash
CUDA_VISIBLE_DEVICES=0 python validation_campaigns/speech_p4/eval_wer.py \
    --model models/parakeet-ctc-1.1b --manifest data/librispeech/manifests/eval.jsonl \
    --out eval_logs/parakeet_base.json &
CUDA_VISIBLE_DEVICES=1 python validation_campaigns/speech_p4/eval_wer.py \
    --model runs/parakeet-ctc-1.1b/final --processor models/parakeet-ctc-1.1b \
    --manifest data/librispeech/manifests/eval.jsonl --out eval_logs/parakeet_tuned.json &
wait
```

For a LoRA run, point `--model` at the **original base model** and `--adapter` at the `final/` adapter directory instead:

```bash
python validation_campaigns/speech_p4/eval_wer.py \
    --model models/gemma-4-E4B-it --adapter runs/gemma-4-E4B-it/final \
    --manifest data/librispeech/manifests/eval.jsonl --out eval_logs/gemma4_tuned.json
```

`qwen2-audio-7b-lora` (chapter 03 §5) evaluates the same way:

```bash
python validation_campaigns/speech_p4/eval_wer.py \
    --model models/Qwen2-Audio-7B-Instruct --adapter runs/qwen2-audio-7b-lora/final \
    --manifest data/librispeech/manifests/eval.jsonl --out eval_logs/qwen2audio_lora_tuned.json
```

`qwen2-audio-7b`'s full fine-tune (chapter 03 §5) saves a normal full-fine-tune checkpoint with its own `generation_config.json`, so it evaluates exactly like parakeet/whisper above — no `--adapter`, just `--processor` pointing at the original base model:

```bash
python validation_campaigns/speech_p4/eval_wer.py \
    --model runs/qwen2-audio-7b/final --processor models/Qwen2-Audio-7B-Instruct \
    --manifest data/librispeech/manifests/eval.jsonl --out eval_logs/qwen2audio_fullft_tuned.json
```

For Whisper (seq2seq), `--language en` is required on **both** the base and the tuned eval call — same reason as training, see chapter 03.

**Check it worked.** The output JSON must hold one entry per manifest row, in `"all"`:

```bash
jq '.all | length' eval_logs/parakeet_base.json
```

```
2683
```

> [!WARNING]
> Both calls in a pair must use the **same** `--manifest` and the same `--limit` (ideally no `--limit` at
> all). Scoring base on 2,683 rows and tuned on 300 rows produces a WER difference that means nothing, and
> `paired_bootstrap.py` will refuse the pair outright (§5).

## 4 — Measured results: base vs fine-tuned, full dev-clean (2,683 rows), 500-step runs on full train-clean-100 (28,539 rows), 2x H200

| model | fine-tune | base WER | tuned WER | diff (pts) | 95% CI (paired bootstrap, pts) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| parakeet-ctc-1.1b | full FT | 1.87% | 1.77% | -0.09 | [-0.14, -0.05] | **significantly BETTER**; P(diff<0)=1.000 |
| whisper-large-v3 | full FT, sentence-cased targets | 2.25% | **1.72%** | **-0.53** | [-0.65, -0.41] | **significantly BETTER**; P(diff<0)=1.000 |
| gemma-4-E4B-it | LoRA r16/a32 + towers in full, sentence-cased targets | 4.04% | **2.99%** | **-1.05** | [-1.24, -0.85] | **significantly BETTER**; P(diff<0)=1.000 |
| qwen2-audio-7b | LoRA r16/a32 + towers in full, automatic adapter scoping | 35.58% | **3.41%** | **-32.17** | [-33.03, -31.34] | **significantly BETTER**; P(diff<0)=1.000 |
| qwen2-audio-7b | full FT, FSDP (`full` state dict) | 35.58% | **2.57%** | **-33.01** | [-33.85, -32.16] | **significantly BETTER**; P(diff<0)=1.000 |

**parakeet-ctc-1.1b** improves and essentially matches the longer-campaign ceiling (`speech_p7`: 1.86% -> 1.80%). **whisper-large-v3** improves from 2.25% to 1.72% — within 0.03 points of the repo's own reference run (1.69%) — once the recipe fix in the note below is applied. **gemma-4-E4B-it** improves from 4.04% to 2.99% — within 0.03 points of the longer-campaign number (`speech_p7`: 4.07% -> 2.96% LoRA). This last result needed an upstream fix first: `speech_tower_resolution_refusal` was resolving tower paths against the `peft`-wrapped model (top-level `base_model`) instead of the unwrapped one and so misfired on this otherwise-working LoRA family; it now resolves via `model.get_base_model()` — see chapter 03. Speech gates confirmed PASS: `speech.audio_row_coverage` 8016/8016, `speech.tower_movement` 271/271 (220 moved) + 1/1 (1 moved).

**qwen2-audio-7b** improves from 35.58% to 3.41% (LoRA) and to 2.57% (full fine-tune) — the largest deltas in this tutorial; the base number (35.58%) independently matches the quoted base (35.6%) in both recipes. LoRA needed an upstream `FamilySpec` update first (automatic LoRA adapter-target scoping, 224 modules). The primary recipe is the **full** fine-tune (reference, 1 GPU: 35.6% -> 1.79%); this tutorial's 2-GPU FSDP run lands at 2.57%, within 0.8 points of that reference. The full-FT recipe hit a checkpoint-save bug of its own — `checkpoint.save_complete` found 0/876 tensors because transformers reverts on-disk tensor names on save under `transformers>=5.18` while the save-completeness check compared against the wrong names — fixed upstream (`origin/main` `152f43c`), see chapter 03 §5. Speech gates confirmed PASS for both: LoRA `speech.audio_row_coverage` 8016/8016, `speech.tower_movement` 487/487 (473 moved) + 2/2 (2 moved); full FT `speech.audio_row_coverage` 4008/4008, `speech.tower_movement` 487/487 (474 moved) + 2/2 (2 moved).

**whisper-large-v3: the target-text-format lesson.** Earlier rounds measured whisper-large-v3 REGRESSING (2.25% -> 3.32-3.58% depending on configuration) because the manifest builder wrote training targets in LibriSpeech's RAW UPPER CASE (`"SWEETWATER SOMEONE DREW..."`), which pushes Whisper onto upper-case BPE pieces it rarely uses and fragments words it already transcribes correctly zero-shot (`"Sweetwater"` -> `"SWEET WATER"`). Training targets in SENTENCE CASE (`"Sweetwater someone drew..."` — the same transform the `speech_p3`/`speech_p6` campaigns have always used) reproduce the reference loss trajectory (step 10/250/500: 0.3314/0.04538/0.04485 vs reference 0.3434/0.0424/0.0450) and land WER at 1.72%. **Never describe this as "a known issue on transformers >= 5.18"** — that hypothesis was specifically tested and ruled out; the cause is target case, not the stack. The rule: when a fine-tune underperforms, check the exact training-target TEXT FORMAT the reference recipe uses before assuming a stack or version difference.

**A 500-step run on 2 GPUs is still a short campaign (~0.14 epoch) — a plumbing check and a teaching tool, not a quality claim.**

## 5. Compare models with paired_bootstrap.py

`paired_bootstrap.py` takes two positional arguments and no flags:

```bash
python validation_campaigns/speech_p7/paired_bootstrap.py eval_logs/parakeet_base.json eval_logs/parakeet_tuned.json
```

It reads the `"all"` array from both eval JSON files (same manifest, same utterance ids required), recomputes per-utterance error/reference-length pairs, and reports a 2000-resample paired bootstrap (fixed seed) of the corpus WER difference with a 95% CI and `P(diff<0)`. One real output line, from the whisper-large-v3 pair (100-step run):

```
n=2683 fields=(errors,reference_length) base=0.0225 tuned=0.0358 diff=+0.0133 95%CI=[+0.0114,+0.0152] P(diff<0)=0.000
```

**Check it worked.** `n=` must equal your manifest row count (2,683 for full dev-clean), and the sign of `diff` must match the tuned-minus-base direction in §4's table.

The script asserts the two files cover the exact same utterance ids (`assert keys == sorted(tuned)`) — comparing across different manifests or `--limit` values fails loudly rather than silently misaligning rows.

> [!WARNING]
> `P(diff<0)` is the fraction of bootstrap resamples where the tuned model wins. Read the CI first: if it
> straddles 0, a favourable `P(diff<0)` like 0.966 is a lean, **not** a win (none of this tutorial's four
> models land there — all four CIs above exclude zero cleanly — but a short run or a small eval set
> elsewhere easily could). Do not report "fine-tuning improved WER" off a CI that includes zero.

## 6. Model-card comparison (optional, not run in this tutorial)

```bash
python validation_campaigns/speech_repro/score_repro.py EVAL.json --whisper-normalizer \
    models/whisper-large-v3/normalizer.json --card-wer <published number> [--tolerance 0.3]
```

This applies Whisper's own `EnglishTextNormalizer` before scoring, for comparing against a model's published WER (test-clean/test-other, not dev-clean). FoundationScale's own normalizer — the one used by `eval_wer.py` and `paired_bootstrap.py` above — is stricter and reads about 0.2 WER points higher than Whisper's normalizer on the same hypotheses. **Never mix the two**: use `eval_wer.py` numbers to compare models/checkpoints against EACH OTHER (as in §4), and `score_repro.py` only when reproducing a published model-card number specifically. This tutorial does not run a model-card reproduction (out of scope for a quick multi-model demo); see [`validation_campaigns/speech_repro/EVIDENCE.md`](../../validation_campaigns/speech_repro/EVIDENCE.md) for a worked example on the full LibriSpeech test splits.

> [!WARNING]
> Scoring the same hypotheses with two different normalizers and putting both numbers in one table is a
> scoring bug, not a result. Pick one normalizer per comparison and say which one you used.

## Troubleshooting

- **`AssertionError` in `paired_bootstrap.py`** -> the two eval JSON files don't cover identical utterance ids; re-run both `eval_wer.py` calls against the SAME `--manifest` with no `--limit` mismatch.
- **Eval raises a sample-rate mismatch error** -> `eval_wer.py` refuses to resample, same policy as training and data prep; re-check chapter 02's manifest was built with `--max-duration` matching what you expect.
- **A LoRA eval looks like it loaded the base model unchanged** -> confirm `--adapter` points at the `final/` directory that actually contains `adapter_model.safetensors`, not the training `OUTPUT_DIR` root.

### Reading exit codes through torchrun

When `torchrun` reports a bare `exit 1` (or some other wrapper code), the real FoundationScale verdict (5 RED / 95 UNMEASURED / 96 REFUSE) is in the `exitcode : N` line of its own per-rank failure report under "Root Cause (first observed failure)" — not the wrapper's own exit code:

```
Root Cause (first observed failure):
[0]:
  rank      : 1 (local_rank: 1)
  exitcode  : 5 (pid: 1566864)
```

`torchrun`'s own process exit code (what `$?` sees in a launcher script) is **not** the FS verdict; always read the per-rank `exitcode` line instead.

**Next:** back to the [README](README.md) for the full results table and ready-to-run configs.
