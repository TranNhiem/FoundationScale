# 04 — Evaluation: WER, base vs fine-tuned and paired bootstrap

Always compare the base model and the fine-tuned checkpoint **on the same held-out rows**. Not on the
training loss, not on a handful of cherry-picked utterances, and not on two different eval sets — on the
identical manifest, scored the same way, then compared utterance by utterance. Everything below was measured
on a 2x H200 VM on 2026-10-10/11: two `eval_wer.py` calls per model (base and tuned) over the full 2,683-row
dev-clean set, then one `paired_bootstrap.py` call per pair. The headline result of this chapter is
unflattering and important: **one of the three models got measurably WORSE after fine-tuning**, and the
paired bootstrap says so with high confidence.

## 1. Confirm your held-out split is clean

dev-clean speakers are disjoint from train-clean-100 speakers by LibriSpeech's own construction (standard
partition), and chapter 02 built both manifests independently from the same filter policy — so this is
inherited from the dataset, not re-verified here.

## 2. Why >= 2000-row eval sets

The repo's own measured finding, stated plainly: **300-row held-out sets cannot resolve WER shifts under
about 1 point** — the paired-bootstrap confidence interval doesn't narrow enough with that few utterances
(full case in [chapter 02 §4](02_data_preparation.md)). Every number in this chapter is measured on the FULL 2,683-row
dev-clean set for exactly that reason. Several of the differences below are under 1.5 points, and a 300-row
eval would have reported some of them as noise.

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

Note the `--processor` on the second call. A full-fine-tune `final/` checkpoint directory only contains
`config.json` + `model.safetensors` — no tokenizer or preprocessor files — so `--processor` must point at the
original base model directory. Omitting it fails trying to load a processor from a directory that doesn't
have one.

For a LoRA run, point `--model` at the **original base model** and `--adapter` at the `final/` adapter
directory instead:

```bash
python validation_campaigns/speech_p4/eval_wer.py \
    --model models/gemma-4-E4B-it --adapter runs/gemma-4-E4B-it/final \
    --manifest data/librispeech/manifests/eval.jsonl --out eval_logs/gemma4_tuned.json
```

For Whisper (seq2seq), `--language en` is required on **both** the base and the tuned eval call — same reason
as training, see chapter 03.

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

## 4. Measured results: base vs fine-tuned, full dev-clean (2,683 rows), 500-step runs on full train-clean-100 (28,539 rows), 2x H200, 2026-10-10/11

| model | fine-tune | base WER | tuned WER | diff (pts) | 95% CI (paired bootstrap, pts) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| parakeet-ctc-1.1b | full FT | 1.87% | 1.77% | -0.09 | [-0.14, -0.05] | **significantly BETTER**; P(diff<0)=1.000 |
| whisper-large-v3 | full FT | 2.25% | 3.34% | +1.09 | [+0.89, +1.28] | **significantly WORSE** (under investigation — see below); P(diff<0)=0.000 |
| gemma-4-E4B-it | LoRA r16/a32 | 4.04% | 3.78% | -0.26 | [-0.51, +0.02] | favourable, borderline (CI nearly excludes 0); P(diff<0)=0.966 |

Read that table honestly. **parakeet-ctc-1.1b improves and essentially matches this repo's own longer-campaign
ceiling** (`speech_p7`: 1.86% -> 1.80%) — a real, if small, win that the paired bootstrap resolves cleanly at
2,683 rows. **gemma-4-E4B-it moves in the right direction** — the repo's longer campaign
(`speech_p7`: 4.07% -> 2.96% LoRA) shows where this model can go — **but stops short of it at 500 steps
here** (4.04% -> 3.78%). The shortfall is plausibly a scheduler / data-order / transformers-version gap; it
was not chased further; GPU count has since been ruled out as its explanation too (see below), so this
gap remains open on its own. **whisper-large-v3 is the one that needs the strongest, most honest framing:
it regresses on the VM stack (transformers 5.18 / torch 2.13 / peft 0.21.2) in every configuration tried**
— 100 steps: +1.33 pts; 500 steps: +1.09 pts; 500 steps + 50-step warmup: +1.11 pts; 500 steps on 1 GPU
with per-device batch 8 (the speech-plane owner's exact recipe): +1.07 pts. **This is NOT reproduced on
transformers 5.5 or 5.19 with torch 2.11 / peft 0.18.1** — the owner measured 2.25% -> 1.69% on both
(paired diff +0.00 pt, CI [-0.02, +0.02]). It is also not a GPU-count effect: re-running the owner's exact
1-GPU / per-device-batch-8 recipe reproduced the same regression. Training loss tracks the owner's
reference closely at steps 10/250/500 (this VM: 0.2993/0.0617/0.0594; owner's reference:
0.3434/0.0424/0.0450) — training is not diverging. Row-by-row inspection shows the fine-tuned checkpoint
correctly learns the target ALL-CAPS/no-punctuation format, but loses previously-correct zero-shot
transcriptions of a recurring subset of proper nouns and uncommon words (`Sweetwater`->`SWEET WATER`,
`mountebank`->`MOUNTAIN BANK`, `Gwynplaine`->`GUINPLAIN`), net-negative across 2,683 rows (592 worse,
230 better) — a real degradation in the trained model on this stack, not an eval-harness or
generation-config artifact (`generation_config.json` is byte-identical to the base model's). State it
plainly: this **regresses on the VM stack; is not reproduced on transformers 5.5 or 5.19 with torch 2.11
(1.69% both); and is under investigation** — never call it "a known issue on transformers >= 5.18", that
hypothesis has been specifically ruled out. Do **not** present Whisper's fine-tune as a quality win here.

> [!WARNING]
> An earlier revision of this chapter listed a fourth model, qwen2-audio-7b. Those results are **RETRACTED**:
> they were obtained by bypassing a gap that FoundationScale now formally refuses before training. See
> chapter 03 §5. Do not quote them.

**A 500-step run on 2 GPUs is still a short campaign (~0.14 epoch) — a plumbing check and a teaching tool,
not a quality claim.**

## 5. Compare models with paired_bootstrap.py

`paired_bootstrap.py` takes two positional arguments and no flags:

```bash
python validation_campaigns/speech_p7/paired_bootstrap.py eval_logs/parakeet_base.json eval_logs/parakeet_tuned.json
```

It reads the `"all"` array from both eval JSON files (same manifest, same utterance ids required), recomputes
per-utterance error/reference-length pairs, and reports a 2000-resample paired bootstrap (fixed seed) of the
corpus WER difference with a 95% CI and `P(diff<0)`. One real output line, from the whisper-large-v3 pair
(100-step run) measured on 2x H200, 2026-10-10:

```
n=2683 fields=(errors,reference_length) base=0.0225 tuned=0.0358 diff=+0.0133 95%CI=[+0.0114,+0.0152] P(diff<0)=0.000
```

**Check it worked.** `n=` must equal your manifest row count (2,683 for full dev-clean), and the sign of
`diff` must match the tuned-minus-base direction in §4's table.

The script asserts the two files cover the exact same utterance ids (`assert keys == sorted(tuned)`) —
comparing across different manifests or `--limit` values fails loudly rather than silently misaligning rows.

> [!WARNING]
> `P(diff<0)` is the fraction of bootstrap resamples where the tuned model wins. Read the CI first: if it
> straddles 0 (as gemma-4-E4B-it's nearly does), a favourable `P(diff<0)` like 0.966 is a lean, **not** a win.
> Do not report "fine-tuning improved WER" off a CI that includes zero.

## 6. Model-card comparison (optional, not run in this tutorial)

```bash
python validation_campaigns/speech_repro/score_repro.py EVAL.json --whisper-normalizer \
    models/whisper-large-v3/normalizer.json --card-wer <published number> [--tolerance 0.3]
```

This applies Whisper's own `EnglishTextNormalizer` before scoring, for comparing against a model's published
WER (test-clean/test-other, not dev-clean). FoundationScale's own normalizer — the one used by `eval_wer.py`
and `paired_bootstrap.py` above — is stricter and reads about 0.2 WER points higher than Whisper's normalizer
on the same hypotheses. **Never mix the two**: use `eval_wer.py` numbers to compare models/checkpoints against
EACH OTHER (as in §4), and `score_repro.py` only when reproducing a published model-card number specifically.
This tutorial does not run a model-card reproduction (out of scope for a quick multi-model demo); see
[`validation_campaigns/speech_repro/EVIDENCE.md`](../../validation_campaigns/speech_repro/EVIDENCE.md) for a
worked example on the full LibriSpeech test splits.

> [!WARNING]
> Scoring the same hypotheses with two different normalizers and putting both numbers in one table is a
> scoring bug, not a result. Pick one normalizer per comparison and say which one you used.

## Troubleshooting

- **`AssertionError` in `paired_bootstrap.py`** -> the two eval JSON files don't cover identical utterance
  ids; re-run both `eval_wer.py` calls against the SAME `--manifest` with no `--limit` mismatch.
- **Eval raises a sample-rate mismatch error** -> `eval_wer.py` refuses to resample, same policy as training
  and data prep; re-check chapter 02's manifest was built with `--max-duration` matching what you expect.
- **A LoRA eval looks like it loaded the base model unchanged** -> confirm `--adapter` points at the `final/`
  directory that actually contains `adapter_model.safetensors`, not the training `OUTPUT_DIR` root.

### Reading exit codes through torchrun

When `torchrun` reports a bare `exit 1` (or some other wrapper code), the real FoundationScale verdict
(5 RED / 95 UNMEASURED / 96 REFUSE) is in the `exitcode : N` line of its own per-rank failure report under
"Root Cause (first observed failure)" — not the wrapper's own exit code:

```
Root Cause (first observed failure):
[0]:
  rank      : 1 (local_rank: 1)
  exitcode  : 5 (pid: 1566864)
```

`torchrun`'s own process exit code (what `$?` sees in a launcher script) is **not** the FS verdict; always
read the per-rank `exitcode` line instead.

**Next:** back to the [README](README.md) for the full results table and ready-to-run configs.
