# NVIDIA Canary in FoundationScale: design note

Status: proposal, not built. Written 2026-10-06 after P4 (`validation_campaigns/speech_p4`).

## The constraint

Canary (canary-1b, canary-1b-flash, canary-qwen-2.5b) exists only as NeMo checkpoints (`.nemo`):
it is not a transformers 5.5 architecture, so the HF-native path that trains Gemma-4, Whisper,
Qwen2-Audio and Parakeet cannot load it. Neither FoundationScale container carries NeMo's ASR or
speechlm2 collections (measured: `nemo.collections.asr` and `lhotse` are absent from the training
image; the Megatron image has `nemo_fw`/`nemo_run`/`nemo_evaluator` but no ASR collection).

## Options

| Option | What it means | Verdict |
|---|---|---|
| A. Port Canary into the HF path | re-implement the FastConformer encoder + transformer decoder (or the SALM wrapper for canary-qwen) as transformers modules and convert weights | REJECT: a weight-conversion project with no upstream owner; every NeMo release would drift |
| B. Embed NeMo's trainer in `train()` | import NeMo ModelPT/Lightning inside the thin plane | REJECT: the research record already rejected NeMo's Lightning/Hydra runtime as a competing control plane (`speech.md` decision table) |
| C. NeMo lane, FoundationScale adjudicates | a separate NeMo ASR environment runs NeMo's own Canary fine-tune; FoundationScale supplies the data contract, launch, and the verdicts | RECOMMENDED: the same shape as the Megatron-Bridge lane, where FS judges checkpoints it did not train |

## Option C in detail

1. **Environment.** An NGC NeMo image (or `nemo_toolkit[asr]` in a dedicated venv) built as its own
   sqsh. Kept out of the thin-plane image, so text/vision/HF-speech runs are unaffected.
2. **Data contract.** Convert FoundationScale audio JSONL (`audio`, `answer`, `duration`) to a NeMo
   manifest (`audio_filepath`, `text`, `duration`, plus Canary's `source_lang`/`target_lang`/`pnc`
   fields). The same strict loader rules apply before conversion: no resampling, no clips over the
   model window, refused rows counted.
3. **Training.** NeMo's `speech_to_text_finetune.py` (or `speechlm2/salm_train.py` for canary-qwen),
   launched by a FoundationScale launcher with the same holding-job / direct-ssh conventions.
4. **Adjudication, all by FoundationScale:**
   - speech row coverage from the converted manifest (checked vs expected, refusals counted);
   - tower movement between base and fine-tuned `.nemo` (a tar holding `model_weights.ckpt`):
     encoder parameters digested by name, same TowerMovementGate, parameters only;
   - held-out WER on the same LibriSpeech and out-of-domain sets, decoded by NeMo `transcribe()` and
     scored by `foundationscale.train.speech_metrics` (one metric across every model kind).
5. **What it does not do.** No FSDP/TP topology validation from FS (NeMo owns it in that lane), and
   no LoRA scope from `FamilySpec` (NeMo's adapter mechanism is separate).

## Cost and first milestone

Build the NeMo ASR sqsh and run `transcribe()` for canary-1b-flash on the existing held-out sets
(zero-shot WER, scored by FS). If that reproduces published numbers, the evaluation bridge is
proven; fine-tuning plus checkpoint adjudication follows. Roughly 1 to 2 days, most of it the image.
