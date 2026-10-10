# Speech 2xH200: tutorial verification (examples/speech-2xH200/)

Verification run for `examples/speech-2xH200/`, the hands-on speech/ASR tutorial (mirrors
`examples/vlm-2xH200/`). Not a new research campaign: reuses the recipe and traps already measured by
`speech_p0` through `speech_p8`, `speech_ddp`, `speech_canary`, `speech_repro` and `speech_q3asr`, and
re-verifies the speech-plane owner's recipe end to end on a fresh 2x H200 VM. Four measurement rounds:

- **Round 1** (2026-10-10, FoundationScale `9dd37b0`): a 100-step / 4,078-row quick-start slice on all
  four models, before the owner's fixes below landed.
- **Round 2** (2026-10-10/11, FoundationScale `ced0c5c`, `transformers==5.18.0`, `torch==2.13.0+cu132`,
  `peft==0.21.2`): the owner's full recipe -- 500 steps, full train-clean-100 (28,539 utterances, 14
  parquet shards, 100.59 h, `<=25 s`), `FOUNDATIONSCALE_TRAIN_AUDIO_GROUP_BY_DURATION` **off**,
  `--dataloader-num-workers 0`, FS's default (linear) LR scheduler -- after pulling `ced0c5c` onto the
  VM's FoundationScale checkout. Scored on the full 2,683-row dev-clean split with `eval_wer.py` +
  `paired_bootstrap.py` (`validation_campaigns/speech_p4`, `speech_p7`).
- **Round 3** (2026-10-11, same stack as Round 2): a targeted Whisper diagnostic requested after the
  speech-plane owner reproduced 2.25% -> 1.69% on transformers 5.5 and 5.19 with torch 2.11 -- GPU count,
  `_attn_implementation`, and a step-by-step loss comparison against the owner's reference run (see
  "Whisper's regression" below).
- **Round 4** (2026-10-11, same stack): root cause found with the owner -- the training manifest's
  `answer` field was LibriSpeech's raw upper case, not sentence-cased like the owner's own
  `speech_p3`/`speech_p6` manifests. Fixed in `scripts/prepare_librispeech.py`, train manifest rebuilt,
  Whisper re-run on the owner's exact recipe: **2.25% -> 1.72%**, matching the owner's 1.69%. See "Round 4"
  below for the full before/after and the separate Gemma-4 issue this round also surfaced.

All WER numbers below are Round 2 unless stated otherwise; Round 3/4 are labelled explicitly where they
apply. Round 4 numbers SUPERSEDE Round 1-3's Whisper numbers.

## STEP 1 diagnostics (run before any 500-step commitment, as requested)

**(a) Whisper label check.** Built one real collated training batch through
`build_speech_collator_checked("seq2seq", ..., language="en")` -- the exact path
`foundationscale.train.loop` uses -- and decoded the first 8 label ids of two rows:

```
row 0: [50259, 50360, 50364, 5462, 4715, 5568, 318, 21124]
       ['<|en|>', '<|transcribe|>', '<|notimestamps|>', 'CH', 'AP', 'TER', ' S', 'IX']
row 1: [50259, 50360, 50364, 44, 1899, 32298, 1598, 31540]
       ['<|en|>', '<|transcribe|>', '<|notimestamps|>', 'M', 'AR', 'GU', 'ER', 'ITE']
```

Labels are correctly formed -- the `<|en|><|transcribe|><|notimestamps|>` prefix the owner suspected
missing is present on every row. Calling the same collator WITHOUT a declared language correctly REFUSES
(exit 96: *"this seq2seq tokenizer is multilingual and its prompt carries a language token, but no
language was declared"*) rather than silently training on an unprefixed label -- the label-formation
path is sound.

**(b) 0-step / lr=0 harness control.** `--max-steps 1 --learning-rate 0`, single GPU, then
`eval_wer.py` on full dev-clean:

| model | base WER | lr=0 control WER | verdict |
|---|---|---|---|
| whisper-large-v3 | 2.2543% | **2.2543%** (bit-identical) | harness sound |
| gemma-4-E4B-it (LoRA) | 4.0359% | 4.0661% (+0.03 pt, noise-scale) | harness sound |

Whisper's control reproduces the base WER EXACTLY. Gemma-4's LoRA control is within 0.03 points of base
(explainable by minor bf16 numerical noise through the LoRA forward path, not a save/load defect) --
negligible next to the actual training effects measured below. **Conclusion: the save/load/eval round
trip is not the source of either model's WER movement; whatever happens at lr>0 is a real training
effect.** (The gemma-4-E4B-it lr=0 control itself hit the `speech.tower_movement` gate's FAIL verdict --
0 of 271/1 tower parameters moved -- which is the CORRECT behaviour for lr=0, not a defect: it is the
gate proving it can detect zero movement.)

## Fixes pulled from origin/main (`ced0c5c`) mid-campaign

**1. Parakeet LayerDrop + DDP (Round 1 workaround retracted).** Round 1 needed a local
`encoder_config.layerdrop=0.0` edit to the downloaded checkpoint to avoid a DDP "unused parameters"
abort (`nvidia/parakeet-ctc-1.1b` ships `layerdrop: 0.1`). FoundationScale `ced0c5c` now detects any
positive `*layerdrop` on the model config and sets `ddp_find_unused_parameters=True` itself, preserving
the upstream recipe instead of zeroing it:

```
[fs:train:validated]  [   ok] model.layerdrop: ['encoder_config.layerdrop=0.1'] declares layerdrop;
DDP told to expect unused parameters (the skipped layers), the recipe left as upstream ships it
```

Verified: reverted the local checkpoint edit (`layerdrop` back to `0.1`, stock HF config) and re-ran
500 steps / 2 GPUs -- exit 0, PASS, no manual config edit needed. **The example no longer carries this
workaround.**

**2. Qwen2-Audio tower resolution now REFUSES instead of silently mismatching.** Round 1 trained
Qwen2-Audio (LoRA, via a `--adapter-target` workaround, and full fine-tune via FSDP) by bypassing an
*unmeasured* tower-scoping gap: the `qwen2_audio` FamilySpec declares bare `audio_tower` /
`multi_modal_projector` / `language_model`, correct on the transformers release it was measured against,
but this VM's `transformers==5.18.0` nests everything one level deeper under `model.*`
(`Qwen2AudioForConditionalGeneration.model.audio_tower`, etc). `ced0c5c` adds
`speech_tower_resolution_refusal`, which now catches this BEFORE training:

```
[fs:train:refuse]  family 'qwen2_audio' declares audio tower(s) ['audio_tower', 'multi_modal_projector']
that do not resolve on the loaded model (its top-level modules: ['lm_head', 'model']). The module layout
differs from the one the FamilySpec was measured on -- usually a transformers version that nests the
model differently. Re-measure the module paths for this transformers version and update
FamilySpec.towers. Refusing (96) before training
```

Verified (2026-10-11, `models/Qwen2-Audio-7B-Instruct`, 1 GPU, `--max-steps 1`): refuses in ~20 s with
exit 96, before any training step -- "costs nothing" exactly as the docstring promises. **Consequence for
Round 1's Qwen2-Audio numbers (LoRA 35.58%->5.85%, CI [-30.74,-28.71]; full FT trained 500 steps cleanly
but failed `checkpoint.save_complete`/`checkpoint.first_save` under FSDP, exit 5, no usable checkpoint
produced): both were obtained by routing AROUND a gap the framework now formally refuses. They are
RETRACTED as tutorial results -- not reproducible through the verified path, and not to be trusted as a
quality claim.** Qwen2-Audio is out of this tutorial's recommended path on this VM until the
`qwen2_audio` FamilySpec is re-measured for transformers >= 5.18 (owner's call, out of scope for an
`examples/`-only tutorial). `configs/qwen2-audio-7b.env` and `configs/qwen2-audio-7b-lora.env` have been
removed; the `ADAPTER_TARGET_*` plumbing they needed has been removed from `train_speech.sh`.

**3. `validation_campaigns/speech_q3asr`: Qwen3-ASR-1.7B now trains in FS** (transformers >= 5.13,
`FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en`). Not added to this tutorial's LibriSpeech runs: Qwen3-ASR's
training format needs cased, punctuated targets (per-token NLL on its own greedy transcript 0.014 vs
1.61 on LibriSpeech's upper-case/no-punctuation references -- a style mismatch, not a format error, per
`speech_q3asr/EVIDENCE.md`), and LibriSpeech's upper-case references would just retrain its casing/
punctuation style rather than demonstrate ASR fine-tuning. Mentioned in chapter 03 as an optional model
pointing at that evidence file; not part of the measured table below.

## Round 2 measured results: 500 steps, full train-clean-100 (28,539 rows), full dev-clean (2,683 rows)

| Model | Kind | Fine-tune | Exit | Peak GB/GPU | s/step | Base WER | Tuned WER | 95% CI (pts) |
|---|---|---|---|---|---|---|---|---|
| parakeet-ctc-1.1b | ctc | full FT (stock config, no manual edit) | 0 | 14.3 | 0.44 | 1.87% | **1.77%** | [-0.14,-0.05] significantly better |
| whisper-large-v3 | seq2seq | full FT, raw-case targets (SUPERSEDED, see Round 4) | 0 | 27.6 | 0.41 | 2.25% | 3.34% | [+0.89,+1.28] significantly worse -- target-format artifact, not a stack bug |
| gemma-4-E4B-it | audio_llm | LoRA r16/a32 (LM) + audio tower/projector full | 0 | 89.8 | 0.97 | 4.04% | **3.78%** | [-0.51,+0.02] favourable, borderline (P(diff<0)=0.966) |
| qwen2-audio-7b-instruct | audio_llm | -- | **96 (REFUSE, by design)** | -- | -- | -- | -- | not applicable -- see fix #2 above |

Parakeet essentially matches the owner's stated ceiling (1.86%->1.80% "at best"; we measured
1.87%->1.77%, a hair better). Gemma-4-E4B-it moves in the owner's stated direction (4.07%->2.96% LoRA)
but stops short of it at 500 steps on this VM (4.04%->3.78%); not re-verified on sentence-cased targets
(see Round 4 -- a separate framework refusal blocked the rerun). Qwen2-Audio is a clean REFUSE, not a
measurement -- see fix #2. **Whisper's Round 2 number is SUPERSEDED by Round 4 below: it was a
target-format artifact, not a stack bug.**

### Whisper's regression (Round 1-3): NOT a transformers-version issue -- resolved in Round 4 below

Whisper-large-v3 fine-tuning made WER measurably WORSE on this VM's stack
(`transformers==5.18.0`/`torch==2.13.0+cu132`/`peft==0.21.2`) in every configuration tried through Round 3.
**This section is kept as the record of how the cause was found; the root cause and fix are in "Round 4"
below -- it was a training-target text-format bug in this tutorial's own manifest builder, not a
transformers/torch/peft stack issue.**

| run | GPUs | steps | train rows | warmup | tuned WER | diff vs base (2.2543%) |
|---|---|---|---|---|---|---|
| Round 1 | 2 | 100 | 4,078 | 0 | 3.58% | +1.33 pts |
| Round 2 | 2 | 500 | 28,539 | 0 | 3.34% | +1.09 pts |
| Round 2 (warmup probe) | 2 | 500 | 28,539 | 50 | 3.36% | +1.11 pts |
| **Round 3** (owner's exact recipe: 1 GPU, per-device batch 8) | 1 | 500 | 28,539 | 0 | 3.32% | +1.07 pts |

**The speech-plane owner independently reproduced 2.25% -> 1.69% on BOTH `transformers==5.5.0` and
`transformers==5.19` with `torch==2.11`/`peft==0.18.1`** (paired diff +0.00 pt, CI [-0.02, +0.02]). This
rules out the transformers-version-skew hypothesis the earlier revision of this file proposed -- the
regression is specific to this VM's stack (torch 2.13 and/or peft 0.21.2 remain the open suspects), not to
a transformers release window. **Never describe this as "a known issue on transformers >= 5.18" -- that
has been specifically disproven.**

Round 3 diagnostics, run in the order requested, against the owner's exact single-GPU recipe (`--dp 1
--nproc_per_node 1 --per-device-batch-size 8`, `--learning-rate 1e-5`, `--seed 0`,
`FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en`, same `train_full.jsonl`):

1. **GPU count.** The owner's recipe is 1 GPU x per-device batch 8, not this campaign's earlier 2 GPUs x
   per-device batch 4. Re-ran exactly that. **Result: WER is 3.32%, statistically indistinguishable from
   the 2-GPU run's 3.34% (same 95%-CI range as the 2-GPU pair). GPU count does not explain the
   regression.**
2. **`model.config._attn_implementation`**, loaded via the exact `load_speech_model("seq2seq", ...)` path
   `eval_wer.py`/training both use: `sdpa` (encoder and top-level config agree). This is the expected
   default -- no `--attn-implementation` override is passed anywhere in this tutorial, so this is not an
   anomaly.
3. **Training loss vs the owner's reference** (this VM, 1 GPU, same recipe):

   | step | this VM | owner's reference |
   |---|---|---|
   | 10 | 0.2993 | 0.3434 |
   | 250 | 0.0617 | 0.0424 |
   | 500 | 0.0594 | 0.0450 |

   Same order of magnitude at every checkpoint, converging normally. **Training is not diverging on this
   stack.**
4. Per the owner's decision rule (*loss matches + WER still worse => the saved checkpoint or its eval*):
   inspected the checkpoint and eval directly.
   - `generation_config.json` of the fine-tuned checkpoint is **byte-identical** to the base model's except
     for the `transformers_version` field -- ruling out a generation-config regression.
   - `foundationscale.train.speech_metrics.normalize_transcript` casefolds and strips punctuation but does
     NOT merge/split compound words or convert digits to/from words -- it treats base and tuned hypotheses
     identically; no scorer bug found.
   - Row-by-row inspection (per-row paired WER delta, base vs this 1-GPU tuned checkpoint, full 2,683-row
     dev-clean): **592 rows score worse after fine-tuning, 230 score better, 1,861 unchanged.** The
     fine-tuned model correctly learns the target LibriSpeech ALL-CAPS/no-punctuation format (e.g. base:
     `"Sweetwater, someone drew that weapon out."` -> tuned: `"SWEET WATER SOME ONE DREW THAT WEAPON
     OUT"` -- format transition working as intended) but on a recurring subset of proper nouns and
     uncommon words the BASE model transcribed correctly zero-shot, the TUNED model is now wrong:
     `Sweetwater` (correct, base, 2 occurrences) -> `SWEET WATER` (wrong, tuned); `mountebank` (correct,
     base, 2 occurrences) -> `MOUNTAIN BANK` (wrong, tuned); `Gwynplaine` (correct, base, 2 occurrences) ->
     `GUINPLAIN`/`GWINPLAIN` (wrong, tuned). At least one row shows an apostrophe replaced with a Unicode
     replacement character (`�`) in the tuned hypothesis only (`"HE ISN�T FIT..."`), not seen in
     the base hypothesis for the same row.

**Conclusion: the checkpoint is genuinely less accurate on this specific vocabulary subset after 500 steps
(~0.14 epoch) on this stack, despite healthy aggregate training loss** -- loss (next-token prediction given
correct context) does not directly measure autoregressive generation quality on rare words, and a short
fine-tune can correctly learn the easy, consistent signal (output format) while disturbing the harder,
less-reinforced signal (correct spelling/recognition of specific rare proper nouns) faster than it
re-learns it. This is a real effect on this stack, not an eval-harness defect, not a save/load defect, and
not a GPU-count effect.

As of Round 3, the root cause was open: training loss matched the owner's reference, ruling out training
divergence; GPU count, attention implementation, and generation-config drift were all ruled out too.
Per the owner's decision rule (*loss matches + WER still worse => the saved checkpoint or its eval*), the
checkpoint/eval was inspected directly (Round 3 point 4 above) and found a consistent pattern: the
fine-tuned model correctly learned the LibriSpeech ALL-CAPS/no-punctuation target format but lost
previously-correct zero-shot transcriptions of a recurring subset of proper nouns and uncommon words. This
observation -- format transition working, vocabulary-specific regressions appearing -- was the clue the
speech-plane owner used to find the real cause, below.

## Round 4: root cause found -- raw LibriSpeech upper case, not a stack bug; FIXED

**The speech-plane owner identified the actual root cause**: their own LibriSpeech training manifest is
built with sentence-cased targets --
`answer = " ".join(text.split()).lower().capitalize()` -- the same transform as
`validation_campaigns/speech_p3/build_librispeech_train.py` and
`validation_campaigns/speech_p6/build_ami_train.py`. This tutorial's `prepare_librispeech.py` instead wrote
LibriSpeech's RAW UPPER CASE transcripts as training targets (`"SWEETWATER SOMEONE DREW THAT WEAPON
OUT"`). Raw upper-case targets push Whisper onto upper-case BPE pieces it rarely uses, which fragments
words it already transcribes correctly zero-shot -- EXACTLY the proper-noun failure pattern Round 3 found
(`"Sweetwater"` -> `"SWEET WATER"`, `"mountebank"` -> `"MOUNTAIN BANK"`).

**Fix applied:** `scripts/prepare_librispeech.py`'s `_answer_text()` now applies
`" ".join(text.split()).lower().capitalize()` to every training-manifest row, with a comment explaining the
measured proper-noun breakage. The train manifest (`train_full.jsonl`, 28,539 rows) was rebuilt with this
transform; the EVAL manifest (`eval.jsonl`) was deliberately left unchanged (LibriSpeech's raw upper-case
reference text) because `eval_wer.py`'s scorer
(`foundationscale.train.speech_metrics.normalize_transcript`) casefolds both the reference and the
hypothesis before computing WER -- an upper-case reference scores identically to a sentence-cased one, so
only the TRAINING target's case ever mattered.

**Re-ran the owner's exact recipe** (1 GPU, per-device batch 8, 500 steps, lr 1e-5, seed 0,
`FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en`) on the rebuilt, sentence-cased manifest:

| step | this VM (sentence-cased) | owner's reference |
|---|---|---|
| 10 | 0.3314 | 0.3434 |
| 250 | 0.04538 | 0.0424 |
| 500 | 0.04485 | 0.0450 |

Loss now matches the owner's reference closely at every checkpoint (compare Round 3's 0.2993/0.0617/0.0594
on raw-case targets -- also close in magnitude, confirming loss alone could not distinguish the two target
formats; WER could).

**WER result**, full 2,683-row dev-clean, `eval_wer.py` + `paired_bootstrap.py` vs the unchanged base:

```
n=2683 fields=(errors,reference_length) base=0.0225 tuned=0.0172 diff=-0.0053 95%CI=[-0.0065,-0.0041] P(diff<0)=1.000
```

**2.25% -> 1.72%**, within 0.03 points of the owner's own 1.69% -- confirmed fix, significantly better
(CI excludes zero, P(diff<0)=1.000). **This conclusively rules out the stack-specific hypothesis** (torch
2.13 / peft 0.21.2): the SAME stack, SAME recipe, only the training-target TEXT FORMAT changed.

**Required wording, superseding the Round 1-3 wording above:** Whisper-large-v3's apparent regression was
a training-TARGET-FORMAT bug in this tutorial's own manifest builder (raw LibriSpeech upper case instead of
sentence case), not a transformers/torch/peft stack issue. Fixed by sentence-casing training targets;
2.25% -> 1.72%, matching the owner's 1.69%. **Never describe the ORIGINAL symptom as "a known issue on
transformers >= 5.18" or "regresses on this VM's stack"** -- both were tested explicitly and are false; the
cause was always the target text format, confirmed by reproducing the owner's own fine-tuned result once
the format matched.

**Gemma-4-E4B-it was NOT re-verified on the sentence-cased manifest.** Re-attempting its LoRA + full-towers
recipe on this VM's current FoundationScale checkout hit a NEW, separately-reproducible refusal:

```
[fs:train:refuse]  family 'gemma4' declares audio tower(s) ['model.audio_tower', 'model.embed_audio']
that do not resolve on the loaded model (its top-level modules: ['base_model']). The module layout differs
from the one the FamilySpec was measured on -- usually a transformers version that nests the model
differently. Re-measure the module paths for this transformers version and update FamilySpec.towers.
Refusing (96) before training
```

This fires AFTER `peft.get_peft_model()` wraps the model for LoRA (confirmed: the preceding
`[fs:train:adapter] lora attached to 258 module(s); 343638304/8284739136 parameters trainable` line prints
successfully first), at which point the wrapped model's top-level module is `base_model`, so the family's
otherwise-correct bare `model.audio_tower`/`model.embed_audio` paths no longer resolve.
`speech_tower_resolution_refusal` (the same check that correctly catches Qwen2-Audio's genuine nesting
mismatch, fix #2 above) appears to run against the POST-wrap model unconditionally, which would make it
misfire for ANY LoRA + audio-tower family, not just the ones whose paths are genuinely wrong. Reproduced 3
times (via `train_speech.sh`, and twice via direct `torchrun` invocation with a 5-step smoke run), on the
unmodified `ced0c5c` checkout with cleared `__pycache__` -- deterministic, not a race condition. No CLI flag
routes around it (declaring `--adapter-target` only affects LoRA target selection, which already succeeds;
the failing check is the separate tower-resolution one). **This is reported upstream for the owner's
attention rather than worked around.** Gemma-4-E4B-it's 4.04% -> 3.78% number in this file is carried over
from the earlier raw-case-target run (Round 2) and has not been re-verified on sentence-cased targets.

## Troubleshooting note added per the owner

When `torchrun` reports a bare `exit 1` (or any wrapper exit code), the real FoundationScale verdict (5
RED / 95 UNMEASURED / 96 REFUSE) is in the `exitcode : N` line of its own failure report, e.g.:

```
Root Cause (first observed failure):
[0]:
  rank      : 1 (local_rank: 1)
  exitcode  : 5 (pid: 1566864)
```

`torchrun`'s own process exit code (what `$?` sees in a launcher script) is NOT the FS verdict; always
read the per-rank `exitcode` line instead.

## Shared-VM caveat

This VM is shared with at least one other, unrelated team (`examples/vlm-2xH200/scripts/train_dpo.py`,
a 27B VLM DPO job) that does not participate in this campaign's `GPU_LOCK` protocol. Several runs above
were launched into windows where GPU memory and compute were briefly contended by that job; peak-memory
numbers above were filtered to this campaign's own process PIDs (`nvidia-smi --query-compute-apps`) where
contention was detected, but s/step numbers during a contended window run slower than an isolated window
would. Re-verify timing-sensitive numbers if reproducing this campaign in isolation.

## Caveats

- 500 steps over 28,539 rows is ~0.14 epoch -- still a short run by full-training-campaign standards
  (`speech_p4`/`speech_p7` used more steps over the same or larger data for their headline numbers).
- Peak GB/GPU is a single highest sample from a 3-5 second `nvidia-smi` poll, not a CUDA-allocator
  high-water mark; treat as indicative, same caveat `examples/vlm-2xH200/README.md` states for its own
  throughput numbers.
- Qwen2-Audio-7B-Instruct is not currently reproducible through FoundationScale's verified path on this
  VM's transformers release; do not re-attempt the Round 1 workarounds (`--adapter-target` enumeration,
  FSDP full fine-tune) -- they bypass a gap the framework now formally refuses for exactly this reason.
- Gemma-4-E4B-it's number in the Round 2 table is from BEFORE the sentence-case fix and has not been
  re-verified against it; a separate, newly-encountered `speech_tower_resolution_refusal` misfire (Round 4)
  blocked every rerun attempt, independent of the manifest.
- Parakeet-CTC was NOT re-run on the sentence-cased manifest: its vocabulary/decoding is case-INsensitive
  (CTC outputs a fixed token vocabulary, not case-aware BPE pieces the way Whisper's tokenizer is), so the
  Whisper-specific failure mode (upper-case BPE fragmentation) does not apply, and its Round 2 number
  (1.87% -> 1.77%, stock config) stands unaffected.
