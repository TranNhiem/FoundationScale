# 03 — Training configuration: models, LoRA and memory on 2x H200

This chapter is the whole training interface for the speech/ASR tutorial: one launch script
(`scripts/train_speech.sh`) and one dotenv file per model (`configs/*.env`). Everything below is
**MEASURED** on a real 2x H200 VM verification run on **2026-10-10/11** (this VM's `transformers` is
**5.18.0**, FoundationScale is origin/main **`ced0c5c`**), with the exact numbers from that run — no
rounding, no extrapolation. Anything unproven is marked TODO(verify).

The measured stack is: 2x NVIDIA H200 (140 GiB per GPU), `torchrun --nnodes 1 --nproc_per_node 2`,
`foundationscale.train`, three models (`nvidia/parakeet-ctc-1.1b`, `openai/whisper-large-v3`,
`google/gemma-4-E4B-it`), 500 steps, full train-clean-100 (28,539 rows,
`data/librispeech/manifests/train_full.jsonl`), dp=2.

## 1. The launch command

One script, one config file — the same shape as `../vlm-2xH200/scripts/train_sft.sh`:

```bash
bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/parakeet-ctc-1.1b.env
```

That is the entire interface. The config file sets exactly five things:

| variable | meaning |
|---|---|
| `MODEL` | local weights directory (predownloaded; see chapter 01) |
| `DATASET` | manifest path (the JSONL built in chapter 02) — the measured recipe uses `data/librispeech/manifests/train_full.jsonl` (28,539 rows) |
| `OUTPUT_DIR` | where `checkpoint-<N>/` and `final/` land |
| `MODEL_KIND` | one of `ctc` \| `seq2seq` \| `audio_llm` — **pure shell branching, never passed to the training CLI** |
| `ADAPTER` | empty = full fine-tune, or `lora` |

Everything else is a shell-overridable tunable with a default, so a one-off experiment never needs a new
config file. `train_speech.sh` itself defaults to `STEPS=500`, `LR=1e-5`, `BATCH=4`, `WARMUP=0`,
`GPUS=2` when a config doesn't specify otherwise:

| variable | default | meaning |
|---|---|---|
| `STEPS` | `500` | `--max_steps` |
| `LR` | `1e-5` | learning rate |
| `BATCH` | `4` | per-device batch size |
| `GRAD_ACCUM` | `1` | gradient accumulation steps |
| `SAVE_EVERY` | `500` | checkpoint every N steps |
| `RANK` | `16` | LoRA rank |
| `ALPHA` | `32` | LoRA alpha |
| `WARMUP` | `0` | warmup steps |
| `GPUS` | `2` | `--nproc_per_node` |
| `WORKERS` | `0` | dataloader workers (`--dataloader-num-workers`, `0` throughout) |
| `GROUP_BY_DURATION` | `0` | set `1` to enable `FOUNDATIONSCALE_TRAIN_AUDIO_GROUP_BY_DURATION`; the measured recipe runs **without** it (see §2) |

Each model's own `configs/<name>.env` supplies its launch tunables as defaults using the
`: "${VAR:=default}"` shell idiom, so a shell-level override still wins over the config file. The
measured per-model tunables, exactly:

| model | STEPS | LR | BATCH (per-device) | WARMUP | GPUS |
|---|---|---|---|---|---|
| `nvidia/parakeet-ctc-1.1b` | 500 | 5e-5 | 4 | 50 | 2 |
| `openai/whisper-large-v3` | 500 | 1e-5 | 4 | 0 | 2 |
| `google/gemma-4-E4B-it` | 500 | 1e-4 | 8 | 0 | 2 |

(global batch 8, 8 and 16 respectively). The LR scheduler is FoundationScale's default (linear) — this
tutorial passes **no** `--lr-scheduler-type` override.

Override on the command line, before the script:

```bash
STEPS=500 LR=2e-5 bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/whisper-large-v3.env
```

Inside, the script runs exactly this shape:

```bash
torchrun --nnodes 1 --nproc_per_node "$GPUS" --master_addr 127.0.0.1 --master_port 29880 \
  -m foundationscale.train ...
```

Two details there are load-bearing:

- `--master_addr 127.0.0.1`, **never `localhost`**. This VM class resolves `localhost` to IPv6 only (see
  chapter 01), and the rendezvous then hangs forever instead of failing loudly.
- `--master_port 29880` is reserved for this tutorial so it never collides with the vlm tutorial's own
  launcher (port `29500`) when both run on the same box.

## 2. Declaring the data: the `FOUNDATIONSCALE_TRAIN_AUDIO_*` environment variables

The training CLI itself has no audio-specific flags — audio behaviour is entirely env-var declared. These
are the **only** ones this tutorial uses:

| variable | values | note |
|---|---|---|
| `FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN` | `audio` | always set; names the JSONL column holding the audio path. Required or training is refused. |
| `FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE` | `en` | **required** for multilingual seq2seq tokenizers whose prompt carries a language token — Whisper (`seq2seq`) in this tutorial, and also Qwen3-ASR (not used here, see §5) — omitting it is a REFUSAL, exit 96. Must be **unset** for `ctc` and `audio_llm` (`parakeet-ctc-1.1b`, `gemma-4-E4B-it`), whose tokenizers don't carry one. |
| `FOUNDATIONSCALE_TRAIN_AUDIO_GROUP_BY_DURATION` | `1` | **off by default** in the measured recipe. Optional: set `GROUP_BY_DURATION=1` to enable it and bucket batches by the manifest's `duration` field for less padding. A throughput nicety, not a correctness requirement — the manifest from chapter 02 always has `duration`, so it's safe to turn on anywhere, but the measured numbers below were taken with it off. |
| `HF_HUB_OFFLINE` | `1` | always set; weights are predownloaded, no network calls at train/eval time. |

The row schema itself uses hardcoded `text` (prompt) / `answer` (target) column names inside FoundationScale
— there is no env var to rename those, and chapter 02's manifest already matches.

### the pre-pass: validate and measure before touching a GPU

`scripts/train_speech.sh` itself does no manifest validation of its own — it is a thin `torchrun` wrapper,
same as `../vlm-2xH200/scripts/train_sft.sh`. The validation happens one layer down, inside
`foundationscale.train`'s own startup prologue, every run, before the first batch: topology is checked
against the declared profile, `memory.weights_fit` confirms the model fits before loading it, and the
family registry resolves the model's kind and adapter scope. A malformed manifest or an undeclared audio
column fails there (a REFUSAL, exit 96), before any GPU memory is touched — see the `[fs:train:validated]`
and `[fs:train:refuse]` lines in every log below for what that looks like in practice.

## 3. The three models and what each needs

| model | kind | fine-tune method | special env/flags |
|---|---|---|---|
| `nvidia/parakeet-ctc-1.1b` | `ctc` | full fine-tune | none beyond the table above |
| `openai/whisper-large-v3` | `seq2seq` | full fine-tune | `FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en` (required) |
| `google/gemma-4-E4B-it` | `audio_llm` | LoRA r16/alpha32 on the language model | see "why no gradient checkpointing" below |

For `google/gemma-4-E4B-it`, the LoRA adapter covers the language model only — but the `gemma4`
FamilySpec trains the audio tower and the `embed_audio` projector in **full** automatically alongside it.
That is not an extra flag and not an extra config knob; it happens by family declaration.

For `nvidia/parakeet-ctc-1.1b`, FoundationScale **freezes the model's BatchNorm running statistics**
during speech training automatically (affine weights still train) — small, zero-padded audio batches
corrupt BatchNorm's running mean/variance otherwise, and this is measured elsewhere in this repo
(`validation_campaigns/speech_p4`) to cost real WER if it isn't done. That freeze is why the recipe above
"just works" with no BatchNorm-specific flag in `configs/parakeet-ctc-1.1b.env`: it is not something you
need to set up, FoundationScale does it for every `ctc`-kind run.

`MODEL_KIND` is consumed only by shell branching in `scripts/train_speech.sh` (which env vars to set/unset,
which CLI flags to emit). It is never handed to `foundationscale.train`, which infers the family from the
model itself.

## 4. LoRA settings and why no `--gradient-checkpointing`, ever

`--gradient-checkpointing` is **deliberately never passed**, for any of the 3 models, by design (one uniform
code path): `google/gemma-4-E4B-it` has 18 KV-shared decoder layers whose forward is only correct with the
KV cache ON; `transformers.Trainer` sets `use_cache=False` by default, and FoundationScale pins it back to
`True` for this family — but gradient checkpointing would re-break that invariant, so it is refused rather
than silently miscomputed. None of the 3 models in this tutorial need checkpointing at these step
counts/batch sizes (peak memory stayed under 90 GiB of 140 GiB per GPU in every run below), so leaving it
off uniformly costs nothing and removes a whole class of model-specific exceptions from the script.

LoRA itself is `RANK`/`ALPHA` (16/32 in every measured run) with `ADAPTER=lora`; an empty `ADAPTER` is a
full fine-tune. The defaults just work for all three models — there is no per-model adapter plumbing left
in `scripts/train_speech.sh`.

## 5. Qwen2-Audio-7B-Instruct: a clean REFUSE, not a result

**MEASURED, 2026-10-11, this VM's transformers 5.18.0, FoundationScale origin/main `ced0c5c`.**
`Qwen/Qwen2-Audio-7B-Instruct` is **not** part of this tutorial's measured or recommended path. On this VM
it refuses before training, verbatim:

```
[fs:train:refuse]  family 'qwen2_audio' declares audio tower(s) ['audio_tower', 'multi_modal_projector']
that do not resolve on the loaded model (its top-level modules: ['lm_head', 'model']). The module layout
differs from the one the FamilySpec was measured on -- usually a transformers version that nests the
model differently. Re-measure the module paths for this transformers version and update
FamilySpec.towers. Refusing (96) before training
```

Root cause in plain terms: `Qwen2AudioForConditionalGeneration` nests its towers one level deeper
(`model.audio_tower`, `model.multi_modal_projector`, `model.language_model`) on this VM's
`transformers==5.18.0`, while the `qwen2_audio` FamilySpec declares the bare names `audio_tower` /
`multi_modal_projector` / `language_model`. FoundationScale (as of origin/main `ced0c5c`) resolves the
declared tower names against the loaded model's top-level modules and, when they do not resolve, **refuses
rather than training against an unresolved tower**. This replaces an earlier, now-retracted workaround that
bypassed this exact gap via `--adapter-target`; that workaround and everything it produced are retracted
and must not be re-attempted.

Cost of the refusal: it fires in **~20 seconds**, before any GPU training time is spent, and it produces
**NO checkpoint**.

This is pending a FamilySpec re-measurement for `transformers >= 5.18` — the speech-plane owner's call,
out of scope for an `examples/`-only tutorial. Do not attempt the old workaround.

One alternative worth knowing about: `Qwen3-ASR-1.7B` is an audio-LLM that **does** train on this stack
(`transformers >= 5.13`), verified in `validation_campaigns/speech_q3asr/EVIDENCE.md`. It is not used with
this tutorial's LibriSpeech manifests, though, because its training format needs cased, punctuated targets
and LibriSpeech's upper-case, unpunctuated references would only retrain its casing/punctuation style
rather than demonstrate ASR fine-tuning.

## 6. Memory and throughput, measured (500 steps, 2x H200)

| model | fine-tune | exit | peak GB/GPU | s/step | wall (500 steps) |
|---|---|---|---|---|---|
| parakeet-ctc-1.1b | full FT | 0 | 14.3 | 0.44 | 245 s |
| whisper-large-v3 | full FT | 0 | 27.6 | 0.41 | 230 s |
| gemma-4-E4B-it | LoRA r16/a32 | 0 | 89.8 | 0.97 | 511 s |

**MEASURED, 2026-10-10/11, 2x H200.** `peak GB/GPU` is the single highest sample from a 3–5 second
`nvidia-smi` poll during each run; `s/step` is `transformers.Trainer`'s own reported
`train_runtime / max_steps`. All three runs stayed comfortably under 140 GiB per H200, so none of the
memory levers `../vlm-2xH200` documents (fused-loss, FSDP sharding) were needed at this scale — a longer
run or a bigger model would need to revisit that chapter.

**Shared-VM caveat:** this VM runs other, unrelated jobs outside this campaign's GPU-lock protocol, so
timing numbers can vary run to run under contention. The peak-memory numbers above were filtered to this
campaign's own process PIDs (`nvidia-smi --query-compute-apps`) where contention was detected.

### Whisper: regresses on the VM stack; not reproduced on transformers 5.5 or 5.19; under investigation

`openai/whisper-large-v3` fine-tuning **regresses WER on this VM's stack** (transformers 5.18.0 / torch
2.13.0+cu132 / peft 0.21.2) **in every configuration tried** — this is a real, measured finding:

| run | GPUs | steps | train rows | warmup | tuned WER | diff vs base (2.2543%) |
|---|---|---|---|---|---|---|
| Round 1 | 2 | 100 | 4,078 | 0 | 3.58% | +1.33 pts |
| Round 2 | 2 | 500 | 28,539 | 0 | 3.34% | +1.09 pts |
| Round 2 (warmup probe) | 2 | 500 | 28,539 | 50 | 3.36% | +1.11 pts |
| Round 2 (1-GPU, owner's exact recipe) | 1 | 500 | 28,539 | 0 | 3.32% | +1.07 pts |

**This is NOT a transformers-version issue** — the speech-plane owner reproduced 2.25% -> 1.69% on both
transformers 5.5 and transformers 5.19 with torch 2.11 / peft 0.18.1 (paired diff +0.00 pt,
CI [-0.02, +0.02]). The regression is specific to the software stack on this VM, not to a transformers
release window. Four diagnostics were run, in order, to localise it:

1. **GPU count.** The owner's recipe is 1 GPU x per-device batch 8, not 2 GPUs x per-device batch 4.
   Re-ran exactly that (500 steps, lr 1e-5, seed 0, `AUDIO_LANGUAGE=en`, same train set, single GPU).
   **Result: WER is still 3.32%, statistically indistinguishable from the 2-GPU run (3.34%). GPU count
   does not explain the regression.**
2. **Attention implementation.** `model.config._attn_implementation` is `sdpa` — the expected default, not
   an anomaly (no `--attn-implementation` override is passed anywhere in this tutorial).
3. **Training-loss comparison**, this VM's 1-GPU run vs the owner's reference:

   | step | this VM | owner's reference |
   |---|---|---|
   | 10 | 0.2993 | 0.3434 |
   | 250 | 0.0617 | 0.0424 |
   | 500 | 0.0594 | 0.0450 |

   Same order of magnitude at every checkpoint, converging normally — **training is not diverging** on
   this stack (torch 2.13).
4. Per the owner's decision rule — *if the loss matches and WER is still worse, it's the saved checkpoint
   or its eval* — the loss matches, so the checkpoint/eval was inspected directly. `generation_config.json`
   is byte-identical to the base model's (only `transformers_version` differs), ruling out a
   generation-config regression. Reading the actual hypotheses row by row (sorted by per-row WER
   regression vs base) shows a consistent pattern: of 2,683 rows, 592 score worse after fine-tuning and
   230 score better. The fine-tuned model **correctly learns the LibriSpeech ALL-CAPS, no-punctuation
   target format** (base: `"Sweetwater, someone drew that weapon out."`; tuned:
   `"SWEET WATER SOME ONE DREW THAT WEAPON OUT"`) — but on a recurring subset of proper nouns and uncommon
   words it had PREVIOUSLY transcribed correctly zero-shot, the fine-tuned checkpoint is now wrong:
   `Sweetwater` (correct, base) -> `SWEET WATER` (wrong, tuned, two instances); `mountebank` (correct,
   base) -> `MOUNTAIN BANK` (wrong, tuned); `Gwynplaine` (correct, base) -> `GUINPLAIN`/`GWINPLAIN` (wrong,
   tuned, two instances). At least one row also shows apostrophe characters replaced with a Unicode
   replacement character (`�`) in the tuned hypothesis only. None of this is an eval-harness defect — the
   WER scorer (`foundationscale.train.speech_metrics.normalize_transcript`) casefolds and strips
   punctuation but does not merge/split compound words or convert digits to words, so it is scoring both
   models identically and correctly; the fine-tuned MODEL itself is less accurate on this specific subset
   of vocabulary at 500 steps (~0.14 epoch) on this stack, even though its aggregate training loss looks
   healthy.

**Wording for this finding, verified and intentional:** Whisper-large-v3 **regresses on the VM stack**
(transformers 5.18 / torch 2.13 / peft 0.21.2); **not reproduced** on transformers 5.5 or 5.19 with torch
2.11 (1.69% both); **under investigation**. Do not call this "a known issue on transformers >= 5.18" —
that has been specifically ruled out. Do not present Whisper's fine-tune as a working quality demo on
this VM until the stack-specific cause (torch 2.13 and/or peft 0.21.2 are the remaining suspects) is
identified.

`google/gemma-4-E4B-it`'s own shortfall (4.04% -> 3.78% here vs the owner's 4.07% -> 2.96%) is the same
KIND of gap — a worse outcome on this VM's stack than the owner's reference run — but GPU count has now
been ruled out as a general explanation (it did not fix Whisper), so it is not assumed to explain Gemma's
gap either. Gemma's shortfall remains open and uninvestigated beyond what chapter 04 already states.

## 7. Reading the output

Artifacts land in two places per run:

- `$OUTPUT_DIR/checkpoint-<N>/` — intermediate checkpoints, every `SAVE_EVERY` steps.
- `$OUTPUT_DIR/final/` — the end-of-run artifact.

Contents depend on `ADAPTER`:

- full fine-tune (`ADAPTER` empty): `config.json` + `model.safetensors`.
- LoRA (`ADAPTER=lora`): `adapter_config.json` + `adapter_model.safetensors`, **no base weights**.

Exit-code semantics: `0` means every gate passed; `1` from torchrun wrapping FoundationScale's own exit `5`
means **some gate failed** — RED is not automatically "harmless", and it is not automatically "the
checkpoint is fine" either. Find out WHICH gate failed before trusting anything: it can be a save gate
such as `checkpoint.save_complete` (the write itself was incomplete — do not use that checkpoint) or a
VACUOUS measurement gate on an unrelated axis (e.g. `speech.tower_movement` reporting it couldn't prove
movement) while the checkpoint itself is fully intact. Read the gate name and its own reported reason in
the log before deciding whether to evaluate the checkpoint or discard it. A refusal is **REFUSED (exit
96)** and writes no checkpoint at all. Remember that torchrun's own exit code is not the FoundationScale
verdict (see the first Troubleshooting bullet below).

When evaluating a LoRA run, point `eval_wer.py --model` at the ORIGINAL base model dir and `--adapter` at
`$OUTPUT_DIR/final` (chapter 04). When evaluating a full-fine-tune run whose `final/` directory lacks
tokenizer/processor files, pass `--processor` pointing at the original base model dir.

## 8. Troubleshooting

- **`torchrun` reports a bare `exit 1` (or some other wrapper code) and you can't tell which
  FoundationScale verdict it was** — the wrapper's own exit code is NOT the FS verdict. Read the
  `exitcode : N` line inside torchrun's own failure report instead (e.g. `exitcode : 5 (pid: ...)` under
  "Root Cause (first observed failure)") — that N is the real 5/95/96 FoundationScale code.
- **librosa / layerdrop traps** — already covered in chapter 01; see that chapter rather than repeating them
  here.
- **Qwen2-Audio-7B-Instruct refuses with exit 96 naming unresolved audio towers** — that is by design on
  this VM's `transformers==5.18.0`; see §5. Do not attempt the old `--adapter-target` workaround (it has
  been removed from `train_speech.sh` along with all Qwen2-Audio-specific plumbing).
- **Whisper refuses with exit 96 naming `AUDIO_LANGUAGE`** — set `FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en`
  (and make sure it is unset for the other kinds).
- **Whisper fine-tunes but WER gets worse** — measured on this VM's stack (transformers 5.18 / torch 2.13
  / peft 0.21.2); not reproduced on transformers 5.5 or 5.19 with torch 2.11 (1.69% both); under
  investigation. See §6. It is not fixed by GPU count, warmup, or more steps/data (all tried) — don't
  spend time tweaking the launch command.
- **Training hangs at rendezvous** — `--master_addr` must be `127.0.0.1`, not `localhost`.
- **Port already in use** — this tutorial reserves `29880`; the vlm tutorial uses `29500`. Don't run two
  speech launches at once on the same box.

**Next:** [04 — Evaluation](04_evaluation.md)
