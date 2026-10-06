# Speech P4: FoundationScale trains Whisper, Parakeet-CTC and Qwen2-Audio

P4 of `docs/research/speech.md`: wide model support. Three model KINDS now train through the same
`train()` (`train/speech_kinds.py`): audio-LLM (Gemma-4, Qwen2-Audio), encoder-decoder (Whisper)
and CTC (NVIDIA Parakeet). Same protocol as P3 (`../speech_p3`): LibriSpeech train-clean-100, 500
steps at batch 8 on one GB200 GPU, full fine-tune in bf16; held-out WER on the same 300 dev-clean
utterances with `eval_wer.py` (now kind-aware). Measured 2026-10-06.

| Model | Kind | Base WER | Fine-tuned WER | x base | Speech gates | Verdict (<= 0.8x) |
|---|---|---|---|---|---|---|
| Gemma-4-E4B-it (P3) | audio_llm | 3.83% | 2.42% | 0.63 | PASS | PASS |
| Whisper-large-v3 (lr 1e-5, language en) | seq2seq | 2.13% | **1.47%** | 0.69 | PASS | PASS |
| Qwen2-Audio-7B-Instruct (lr 1e-5) | audio_llm | 35.5% | **1.62%** | 0.05 | PASS (placeholder: SKIP, see caveats) | PASS |
| Parakeet-CTC-1.1B (lr 5e-5, warmup 50, BatchNorm statistics frozen) | ctc | 1.66% | **1.67%** | 1.01 | PASS | no gain (no headroom) |

Every run: audio rows loaded and counted, the audio tower moved (Whisper encoder 473/487
parameters, Parakeet encoder 1,394/1,566, Qwen2-Audio audio tower 466/487 + projector 2/2), save
gates PASS. Qwen2-Audio's base WER is high because the instruct model wraps a correct transcript
("The original content of this audio is: '...'"); fine-tuning on the transcript format removes it.

## Defects found and fixed on the way (each measured on GB200 first)

| Defect | Evidence | Fix |
|---|---|---|
| `AutoModelForCausalLM` builds a decoder-only `WhisperForCausalLM` on a Whisper config, and raises for `parakeet_ctc` and `qwen2_audio` | meta-device class probe | load per kind; audio-LLMs load the class named in `config.architectures` |
| Whisper labels missing `<|en|><|transcribe|>` | per-token NLL 20 and 24 on the two prompt positions; loss 1.94 | declared language (`FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE`), `set_prefix_tokens`, every batch checked against the prefix; loss 0.61 |
| Parakeet CTC loss masks labels with `!= pad_token_id`, not -100 | transformers 5.5 `ParakeetForCTC.forward` | CTC labels padded with the blank (`pad_token_id`) |
| Precision gate read BatchNorm `num_batches_tracked` (I64) as a bf16 disagreement | 42 I64 tensors turned a healthy save RED | exempt by name and dtype together (an I64 weight still counts) |
| Tower movement counted BatchNorm running statistics as moved parameters | Parakeet: 1,692 counted vs 1,566 parameters | base digests from `named_parameters()` |
| Fine-tuning corrupted Parakeet's BatchNorm running statistics (small, zero-padded batches) | 1.66% -> 2.39%; restoring only the base BN buffers (`bn_restore.py`) gave exactly 1.66% | BatchNorm statistics frozen during speech training (affine weights still train) |
| Audio coverage lost with `--dataloader-num-workers` (counters per worker process) | 4 workers: gates VACUOUS, run RED | shared-memory `SharedAudioCoverage`; 4 workers: 312/312 rows, PASS |

## Caveats

- n=300 utterances, in-domain LibriSpeech only. Parakeet (and likely Whisper) saw LibriSpeech in
  pre-training, so an out-of-domain set is the real test of headroom.
- Qwen2-Audio's processor exposes no per-row placeholder count, so the placeholder gate SKIPs
  (NOT_ESTABLISHED) rather than passing.
- LoRA wrap points (`FamilySpec.adapter_full_train`) are measured for Gemma-4 only; LoRA with an
  audio column refuses (96) on Whisper, Parakeet and Qwen2-Audio until measured.
- NVIDIA Canary is NeMo-only (not in transformers 5.5) and is not covered.
- On the GB200 tray used here, CUDA contexts started at the same moment can SIGSEGV inside the
  forward-compatibility driver before training starts; staggered launches are clean.
