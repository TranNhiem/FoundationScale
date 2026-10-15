# Vietnamese TTS: VoxCPM2 LoRA on commercial-safe data (plan phase T2b, 2026-10-14/15)

Plan: docs/research/tts_omni_plan.md (Vietnamese first-class, commercial-safe data only). Qwen3-TTS
and Qwen3-Omni cannot speak Vietnamese; VoxCPM2 (Apache-2.0) can, and its official fine-tuning
recipe keeps zero-shot cloning.

## Data (all CC BY 4.0, measured)

`cluster/build_voxcpm_vi.py`: FLEURS-vi train+dev (10.0 h, cased and punctuated), InfoRe-25h (24.8 h,
one paid reader), FPT-FOSD (30.1 h; its dataset card says 100 h). 64.9 h, 43,907 train rows plus a
300-row dev slice, decoded to 16 kHz (VoxCPM2's AudioVAE input rate); no empty transcripts, no
duplicate audio. InfoRe and FOSD transcripts are lowercase without punctuation, a prosody risk on
punctuated input; LoRA (not full fine-tuning) and a punctuated test set guard against it. FLEURS-vi
test is never trained on. The epoch fraction in the trainer log implies all 43,907 rows were used
(no silent drops by `max_batch_tokens`).

## Training

Upstream trainer `scripts/train_voxcpm_finetune.py` at VoxCPM tag 2.0.3, upstream
`conf/voxcpm_v2/voxcpm_finetune_lora.yaml` with only paths and steps filled
(`cluster/make_t2b_config.sh`): LoRA r32/alpha 32 on LM and DiT, lr 1e-4, effective batch 16, 2,000
steps (0.73 epoch), one GB200 GPU, ~10 s/step.

## Evaluation

Two sets, both scored by `tts_eval/judge_multilingual.py` (Qwen3-ASR-1.7B judge, the T0b choice;
WavLM SIM against the prompt clip):
- MiniMax-vi (CC BY-SA 4.0): 100 sentences, two Common Voice voices (`cluster/build_minimax_vi.py`).
- FLEURS-vi cloning set (`cluster/build_fleurs_vi_tts.py`): the 857 test sentences, each cloned from
  a different test utterance of the same gender (537 distinct prompts).

| checkpoint | MiniMax-vi WER / SIM | FLEURS-vi WER / SIM |
|---|---|---|
| base VoxCPM2 | 5.75 / 79.50 | 8.59 / 82.56 |
| step 0 (zero-initialised LoRA = base) | 4.76 / 78.87 | 8.37 / 82.55 |
| step 500 | 1.26 / 76.61 | 7.26 / 80.51 |
| step 1000 | 1.35 / 77.53 | 6.63 / 81.41 |
| step 1500 | 1.26 / 76.88 | 6.64 / 81.55 |
| **step 2000 (reported)** | **1.44 / 77.01** | **6.79 / 81.49** |

The reported checkpoint is the final one, fixed in advance; picking the best-looking step would be
selection on the test sets. Step 0 against base shows the sampling noise of a 100-row set (~1 point).

Paired bootstrap, base vs step 2000:
- FLEURS-vi WER -1.80 points, 95% CI [-3.39, -0.58], n=857; MiniMax-vi WER -4.31 [-8.59, -1.10].
- FLEURS-vi SIM -1.07 points, 95% CI [-1.47, -0.68].

So LoRA on 65 h of commercial-safe Vietnamese makes VoxCPM2 significantly more intelligible, at a
small but significant cost in speaker similarity - consistent with the training mix (one dominant
reader, unpunctuated text). For reference, the same judge reads real FLEURS-vi recordings at 7.49
WER: the fine-tuned output is at the judge's floor on read speech. Not measured yet: naturalness
(UTMOS licence unrecorded) and full fine-tuning (running).
