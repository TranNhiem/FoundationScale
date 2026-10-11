# 03 — Training configuration: models, LoRA and memory on 2x H200

This chapter is the whole training interface for the speech/ASR tutorial: one launch script
(`scripts/train_speech.sh`) and one dotenv file per model (`configs/*.env`). Everything below is
**MEASURED** on a real 2x H200 VM verification run on **2026-10-10/11** (this VM's `transformers` is
**5.18.0**; FoundationScale is origin/main **`ced0c5c`** for the first three models, **`0fa19be`** for
Qwen2-Audio-7B-Instruct's restoration below), with the exact numbers from that run — no rounding, no
extrapolation. Anything unproven is marked TODO(verify).

The measured stack is: 2x NVIDIA H200 (140 GiB per GPU), `torchrun --nnodes 1 --nproc_per_node 2`,
`foundationscale.train`, four models (`nvidia/parakeet-ctc-1.1b`, `openai/whisper-large-v3`,
`google/gemma-4-E4B-it`, `Qwen/Qwen2-Audio-7B-Instruct`), 500 steps, full train-clean-100 (28,539 rows,
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
| `Qwen/Qwen2-Audio-7B-Instruct` (LoRA, measured) | 500 | 1e-4 | 8 | 0 | 2 |
| `Qwen/Qwen2-Audio-7B-Instruct` (full FT, trains but see §5) | 500 | 1e-5 | 4 | 0 | 2 |

(global batch 8, 8, 16, 16 and 8 respectively). The LR scheduler is FoundationScale's default (linear) —
this tutorial passes **no** `--lr-scheduler-type` override.

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

- `--master_addr 127.0.0.1`, **never `localhost`**. `/etc/hosts` on this VM class lists `::1` before the
  IPv4 entry for `localhost` (see chapter 01), so `torchrun`'s rendezvous picks the IPv6 address and hangs
  forever instead of failing loudly.
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

## 3. The four models and what each needs

| model | kind | fine-tune method | special env/flags |
|---|---|---|---|
| `nvidia/parakeet-ctc-1.1b` | `ctc` | full fine-tune | none beyond the table above |
| `openai/whisper-large-v3` | `seq2seq` | full fine-tune | `FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en` (required) |
| `google/gemma-4-E4B-it` | `audio_llm` | LoRA r16/alpha32 on the language model | see "why no gradient checkpointing" below |
| `Qwen/Qwen2-Audio-7B-Instruct` | `audio_llm` | LoRA r16/alpha32, automatic adapter scoping (measured) — or full fine-tune, FSDP (trains, see §5) | none beyond `SHARDING=fsdp`/`FSDP_STATE_DICT=sharded` for the full-FT config |

For `google/gemma-4-E4B-it` and `Qwen/Qwen2-Audio-7B-Instruct`, the LoRA adapter covers the language model
only — but their FamilySpecs (`gemma4`, `qwen2_audio`) train the audio tower and projector in **full**
automatically alongside it, with **automatic** adapter-target scoping (no `--adapter-target` flag for
either model). That is not an extra flag and not an extra config knob; it happens by family declaration.

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

`--gradient-checkpointing` is **deliberately never passed**, for any of the 4 models, by design (one uniform
code path): `google/gemma-4-E4B-it` has 18 KV-shared decoder layers whose forward is only correct with the
KV cache ON; `transformers.Trainer` sets `use_cache=False` by default, and FoundationScale pins it back to
`True` for this family — but gradient checkpointing would re-break that invariant, so it is refused rather
than silently miscomputed. None of the 4 models in this tutorial need checkpointing at these step
counts/batch sizes to train (Qwen2-Audio-7B-Instruct's full fine-tune needs FSDP *sharding* to fit, a
different lever — see §5 — not gradient checkpointing), so leaving it off uniformly costs nothing and
removes a whole class of model-specific exceptions from the script.

LoRA itself is `RANK`/`ALPHA` (16/32 in every measured run) with `ADAPTER=lora`; an empty `ADAPTER` is a
full fine-tune. The defaults just work for all four models — there is no per-model adapter plumbing left
in `scripts/train_speech.sh` (Qwen2-Audio-7B-Instruct's 224 LoRA target modules resolve automatically, same
as gemma-4-E4B-it's).

## 5. Qwen2-Audio-7B-Instruct: LoRA is the measured recipe; full fine-tune trains but hits a checkpoint gap

**MEASURED, 2026-10-11, this VM's transformers 5.18.0, FoundationScale origin/main `0fa19be`.** An earlier
round of this tutorial documented this model as a clean REFUSE: its `qwen2_audio` FamilySpec declared tower
paths (`audio_tower`, `multi_modal_projector`) that did not resolve against this VM's `transformers==5.18.0`
module nesting (`model.audio_tower`, one level deeper than declared), and FoundationScale correctly refused
(exit 96) rather than train against an unresolved tower. **The speech-plane owner has since fixed this**:
`origin/main` `0fa19be` updates the `qwen2_audio` FamilySpec to the `transformers>=5.18` layout, with
**automatic** LoRA adapter-target scoping (224 modules resolve with no `--adapter-target` flag). Their own
lane moved to `transformers==5.19` and retired `5.5` for this model (on `5.5`, `qwen2_audio` now refuses at
load — not relevant to this VM, which stays on `5.18.0`). The old `--adapter-target` enumeration workaround
from the retracted earlier round is obsolete and must not be re-attempted; it is no longer needed.

### LoRA r16/alpha32 + towers in full — the measured recipe, clean PASS

```bash
bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/qwen2-audio-7b-lora.env
```

lr 1e-4, global batch 16 (per-device 8 x dp 2), 500 steps — same shape as `gemma-4-E4B-it`'s recipe.
`ADAPTER=lora` is the only adapter-related setting in the config; FoundationScale resolves the 224 LoRA
target modules and the audio-tower/projector full-train set automatically, exactly like `gemma4`. Training:
exit 0, PASS. Speech gates: `speech.audio_row_coverage` 8016/8016, `speech.audio_placeholder_coverage`
8016/8016, `speech.tower_movement` 487/487 + 2/2 (moved). WER: **35.58% -> 3.41%**, CI
[-0.3303, -0.3134], significantly better — see the README's results table and
`validation_campaigns/speech_2xh200/EVIDENCE.md` Round 6 for the full numbers. **This is the measured
result for Qwen2-Audio in this tutorial.**

### Full fine-tune — the owner's primary recipe, trains correctly, but a real FSDP checkpoint gap on this VM

```bash
bash examples/speech-2xH200/scripts/train_speech.sh examples/speech-2xH200/configs/qwen2-audio-7b.env
```

lr 1e-5, global batch 8 (per-device 4 x dp 2), `SHARDING=fsdp` (9.08B params does not fit under plain DDP
replication on 2x H200 — same OOM-at-~140 GiB/GPU as every other full-FT-of-a-large-model lever in this
repo). Training itself is clean: loss went 0.32 (step 10) -> 0.15 (step 500), the expected shape for "stop
answering the prompt in English prose, start transcribing".

The checkpoint save is where it gets real:

- **`--fsdp-state-dict full` (the FSDP default): RED, exit 5.** `checkpoint.save_complete` and
  `checkpoint.first_save` both fail — the save writes a duplicate `model.safetensors` AND
  `pytorch_model_fsdp.bin` of near-identical size, with every tensor reported F32 instead of the declared
  bf16. No usable checkpoint comes out of this path for this model's architecture, on this stack.
- **`--fsdp-state-dict sharded` (what `configs/qwen2-audio-7b.env` now sets via `FSDP_STATE_DICT=sharded`):
  the save gates PASS** (`checkpoint.save_complete` 876/876 tensors, `speech.audio_row_coverage` 4008/4008,
  `speech.audio_placeholder_coverage` 4008/4008) — **but `speech.tower_movement` comes back `UNMEASURED`**:
  "saved_digests unavailable (e.g. sharded save layout) — base-vs-saved tower digests cannot be compared".
  The run's own final verdict is therefore `UNMEASURED` (exit 95), not `PASS` — nothing regressed, the gate
  simply has no code path for a sharded save layout yet.

Two more gaps surface if you try to actually evaluate that `sharded` checkpoint:

1. It's raw `torch.distributed.checkpoint` (DCP) shards (`final/pytorch_model_fsdp_0/*.distcp` +
   `.metadata`), not HF-loadable safetensors — `eval_wer.py`'s `AutoModel.from_pretrained` can't read it.
   A manual, one-time conversion works and is mechanical (the converted state dict's 876 keys matched
   `Qwen2AudioForConditionalGeneration`'s native naming exactly — `model.audio_tower.*`,
   `model.language_model.*`, `lm_head.weight`):
   ```python
   from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

   dcp_to_torch_save("runs/qwen2-audio-7b/final/pytorch_model_fsdp_0", "consolidated.pth")
   # then: load, cast the inner state dict to bf16, safetensors.torch.save_file() it next to a
   # copy of final/config.json
   ```
   This script is **not** part of this tutorial — it is a diagnostic step, not a shipped recipe.
2. Evaluating the converted checkpoint behaved abnormally: ~102 GiB on a single GPU (vs. ~35 GiB for the
   same base model + a LoRA adapter) and still running past 55 minutes wall clock when it was killed (base
   and LoRA both finish the same 2,683-row eval in 5-10 minutes). The likely cause: the hand-built
   checkpoint directory has no `generation_config.json` (only `config.json` was recovered), so batched
   generation may never hit an early EOS stop and always runs the full `max_new_tokens=128` for every row —
   plausible, not confirmed, and not worth blocking two other testers' GPU time to confirm.

**No full fine-tune WER number is reported for Qwen2-Audio-7B-Instruct.** The recipe trains correctly and
the `sharded` workaround produces a checkpoint FoundationScale's own gates call complete, but getting a
trustworthy, directly-measured WER out of it needs tooling this tutorial does not ship. Use the LoRA number
above. See `validation_campaigns/speech_2xh200/EVIDENCE.md` Round 6 for the full diagnostic trail, including
what's worth fixing upstream (why `--fsdp-state-dict full` double-writes this architecture, and wiring
`speech.tower_movement` / a DCP-to-HF path for sharded saves).

One alternative worth knowing about: `Qwen3-ASR-1.7B` is an audio-LLM that **does** train on this stack
(`transformers >= 5.13`), verified in `validation_campaigns/speech_q3asr/EVIDENCE.md`. It is not used with
this tutorial's LibriSpeech manifests, though, because its training format needs cased, punctuated targets
and LibriSpeech's upper-case, unpunctuated references would only retrain its casing/punctuation style
rather than demonstrate ASR fine-tuning.

## 6. Memory and throughput, measured (500 steps, 2x H200)

| model | fine-tune | exit | peak GB/GPU | s/step | wall (500 steps) |
|---|---|---|---|---|---|
| parakeet-ctc-1.1b | full FT, 2 GPUs | 0 | 14.3 | 0.44 | 245 s |
| whisper-large-v3 | full FT, 2 GPUs (batch 4/GPU) | 0 | 27.6 | 0.41 | 230 s |
| whisper-large-v3 | full FT, 1 GPU (batch 8, owner's recipe, sentence-cased targets) | 0 | not cleanly isolated (shared-VM contention) | 0.73 | 364 s |
| gemma-4-E4B-it | LoRA r16/a32 + towers in full, sentence-cased targets, 2 GPUs | 0 | 86.2 | 0.94 | 470 s |
| qwen2-audio-7b | LoRA r16/a32 + towers in full, automatic adapter scoping, 2 GPUs | 0 | 82.0 | 1.07 | 541 s |
| qwen2-audio-7b | full FT, FSDP (`sharded` state dict), 2 GPUs | 95 (see §5) | 93.3 | 0.85 | 586 s |

**MEASURED, 2026-10-10/11, 2x H200.** `peak GB/GPU` is the single highest sample from a 3–5 second
`nvidia-smi` poll during each run; `s/step` is `transformers.Trainer`'s own reported
`train_runtime / max_steps`. The qwen2-audio-7b full-FT row's `wall` includes its (slower) sharded-DCP
checkpoint write; its `exit` is 95 (UNMEASURED), not 0 — see §5 for why. Every other run here stayed
comfortably under 140 GiB per H200 and exited 0, so none of the memory levers `../vlm-2xH200` documents
beyond FSDP (fused-loss) were needed at this scale.

**Shared-VM caveat:** this VM runs other, unrelated jobs outside this campaign's GPU-lock protocol, so
timing numbers can vary run to run under contention. The peak-memory numbers above were filtered to this
campaign's own process PIDs (`nvidia-smi --query-compute-apps`) where contention was detected.

### The recipe lesson: train on sentence-cased targets, not raw LibriSpeech upper case

`openai/whisper-large-v3` fine-tuning initially **regressed** dev-clean WER on this VM (2.25% -> 3.32-3.58%
across several configurations) — root-caused, with the speech-plane owner, to the TRAINING TARGET FORMAT,
not the software stack:

| cause tried | tuned WER | diff vs base (2.2543%) |
|---|---|---|
| raw LibriSpeech upper case (this tutorial's early rounds, several configs) | 3.32% – 3.58% | +1.07 to +1.33 pts |
| **sentence-cased targets** (`"Marguerite to be unable..."`, not `"MARGUERITE..."`) | **1.72%** | **-0.53 pts** |

Diagnostics that ruled out the wrong explanations first (kept here as the record of how this was found):
GPU count (owner's 1-GPU/per-device-batch-8 recipe reproduced the same regression, 3.32%, as 2-GPU/batch-4
— not a GPU-count effect); `model.config._attn_implementation` was `sdpa`, the expected default; training
loss tracked the owner's reference closely at every configuration (not diverging). Reading hypotheses row
by row then showed WHY: training on LibriSpeech's raw upper-case transcripts pushes Whisper onto upper-case
BPE pieces it rarely uses, which **fragments words it already transcribes correctly zero-shot** —
`"Sweetwater"` -> `"SWEET WATER"`, `"mountebank"` -> `"MOUNTAIN BANK"`, `"Gwynplaine"` -> `"GUINPLAIN"` — a
real degradation concentrated on proper nouns and uncommon words, net-negative across the full 2,683-row
eval set despite healthy aggregate training loss. Loss alone cannot see this: it scores teacher-forced
next-token prediction, not autoregressive generation quality on rare vocabulary.

**The fix**: `scripts/prepare_librispeech.py` writes training targets SENTENCE-CASED —
`" ".join(text.split()).lower().capitalize()` — whitespace collapsed, lower-cased, then only the first
character upper-cased, no punctuation added. This is the same transform this repo's own
`validation_campaigns/speech_p3`/`speech_p6` LibriSpeech/AMI campaigns have always used; this tutorial's
early rounds simply didn't match it. Re-running the exact owner's recipe (1 GPU, per-device batch 8, 500
steps, lr 1e-5, seed 0) on sentence-cased targets reproduced the owner's own loss trajectory almost exactly
(step 10/250/500: this VM 0.3314/0.04538/0.04485 vs owner's reference 0.3434/0.0424/0.0450) and landed WER
at **1.72%**, within 0.03 points of the owner's own 1.69% — **confirmed fix, not a coincidence.**

**The eval manifest is intentionally NOT rebuilt** with this change — it keeps LibriSpeech's raw upper-case
reference text. `eval_wer.py`'s scorer (`foundationscale.train.speech_metrics.normalize_transcript`)
casefolds both the reference and the hypothesis before computing WER, so an upper-case reference scores
identically to a sentence-cased one; only the TRAINING target's case mattered, never the eval reference's.

`google/gemma-4-E4B-it` was re-verified on the sentence-cased manifest once the speech-plane owner fixed a
separate issue upstream: `speech_tower_resolution_refusal` had been firing on the LoRA-wrapped model (whose
top-level module becomes `base_model` after `peft.get_peft_model()`), misfiring on an already-working LoRA
family instead of only catching Qwen2-Audio's genuine nesting mismatch. The fix resolves FamilySpec tower
paths via `model.get_base_model()` instead. Re-running the owner's exact recipe (LoRA r16/alpha32, lr 1e-4,
global batch 16, 500 steps, no gradient checkpointing) on the sentence-cased manifest: exit 0, all speech
gates PASS (`speech.audio_row_coverage` 8016/8016, `speech.audio_placeholder_coverage` 8016/8016,
`speech.tower_movement` 271/271 moved 220 + 1/1 moved 1), dev-clean WER **4.04% -> 2.99%**
(CI [-1.24, -0.85] pts, significantly better) — within 0.03 points of the owner's own 4.07% -> 2.96%.

## 7. Reading the output

Artifacts land in two places per run:

- `$OUTPUT_DIR/checkpoint-<N>/` — intermediate checkpoints, every `SAVE_EVERY` steps.
- `$OUTPUT_DIR/final/` — the end-of-run artifact.

Contents depend on `ADAPTER` and `SHARDING`:

- full fine-tune, no FSDP (`ADAPTER` empty, `SHARDING` unset — parakeet, whisper): `config.json` +
  `model.safetensors`, directly HF-loadable.
- LoRA (`ADAPTER=lora` — gemma-4-E4B-it, qwen2-audio-7b-lora): `adapter_config.json` +
  `adapter_model.safetensors`, **no base weights**.
- full fine-tune, FSDP sharded (`ADAPTER` empty, `SHARDING=fsdp`, `FSDP_STATE_DICT=sharded` —
  qwen2-audio-7b): `config.json` + raw `torch.distributed.checkpoint` shards
  (`pytorch_model_fsdp_0/*.distcp` + `.metadata`), **not** directly HF-loadable — see §5 for the manual
  conversion this needs before `eval_wer.py` can read it.

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
- **Qwen2-Audio-7B-Instruct full fine-tune reports RED, `checkpoint.save_complete`/`checkpoint.first_save`
  failing** — that's the `--fsdp-state-dict full` default; see §5. Use `FSDP_STATE_DICT=sharded`
  (already the config default) to get a gate-passing checkpoint; be aware of what that trades away.
- Do not attempt the old `--adapter-target` enumeration workaround — it predates the owner's `0fa19be`
  FamilySpec fix and is obsolete; LoRA's 224 target modules now resolve automatically.
- **Whisper refuses with exit 96 naming `AUDIO_LANGUAGE`** — set `FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en`
  (and make sure it is unset for the other kinds).
- **Whisper fine-tunes but WER gets worse** — check your manifest's `answer` field is SENTENCE-CASED
  (`"Sweetwater..."`), not LibriSpeech's raw upper case (`"SWEETWATER..."`). Raw upper-case targets
  measurably regress Whisper's WER on this stack by fragmenting words into upper-case BPE pieces it rarely
  uses — see §6. `scripts/prepare_librispeech.py` writes sentence-cased targets by default; this symptom
  means an older manifest or a hand-rolled builder is in use. Not fixed by GPU count, warmup, or more
  steps/data (all ruled out before the real cause was found).
- **`gemma-4-E4B-it` (or any LoRA + audio-tower family) refuses with exit 96 naming unresolved audio
  towers** — this was a real FoundationScale bug (`speech_tower_resolution_refusal` resolving tower paths
  against the `peft`-wrapped model, whose top-level module is `base_model`, instead of the unwrapped one),
  **fixed upstream**: the check now resolves via `model.get_base_model()`. If you see this on an older
  FoundationScale checkout, `git pull` first rather than working around it.
- **Training hangs at rendezvous** — `--master_addr` must be `127.0.0.1`, not `localhost`.
- **Port already in use** — this tutorial reserves `29880`; the vlm tutorial uses `29500`. Don't run two
  speech launches at once on the same box.

**Next:** [04 — Evaluation](04_evaluation.md)
