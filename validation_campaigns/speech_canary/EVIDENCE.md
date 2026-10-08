# Canary through the NeMo lane: FoundationScale feeds, NeMo trains, FoundationScale adjudicates

First milestone of `docs/research/speech_canary.md` (option C), measured 2026-10-07 on one GB200
tray. Canary is NeMo-only, so NeMo owns loading, decoding and the training loop; FoundationScale
owns the data contract, the verdicts and the metric (the same `speech_metrics` WER as every
HF-native model).

## Environment

The estate's NeMo Framework 26.08 image ships the NeMo 3.1 source (`/opt/NeMo`, with `asr` and
`speechlm2`) but not the installed package. `make_nemo_env.sh` builds a venv layered on the
image's Python with `--system-site-packages`, installs `/opt/NeMo[asr]`, and pins the image's
torch 2.13 (nv26.06) / transformers 5.12.1 so pip cannot replace them.

## Zero-shot (nvidia/canary-1b-flash, `eval_wer_nemo.py`)

| Set | Canary-1B-flash | Parakeet-CTC-1.1B (P5, same metric) |
|---|---|---|
| LibriSpeech dev-clean (300) | 1.59% (pnc=yes) / 1.43% (pnc=no) | 1.66% |
| VoxPopuli (300) | 5.17% | 6.28% |
| AMI (300) | 16.49% | 16.38% |
| Earnings-22 (148) | 24.70% | 24.42% |

## Fine-tune (`nemo_finetune.py`) and adjudication (`nemo_adjudicate.py`)

500 steps on LibriSpeech train-clean-100, batch 8, bf16-mixed (fp32 master weights), cosine with
50 warmup steps, pnc=no. FoundationScale converted the JSONL under its strict rules (28,539 of
28,539 rows kept, none refused) and adjudicated the saved `.nemo` against the base:

| Gate | Verdict |
|---|---|
| speech.audio_row_coverage | PASS 28,539 / 28,539 |
| speech.tower_movement `encoder` | PASS 1,196 / 1,196 parameters moved |
| speech.tower_movement `transf_decoder` | PASS 109 / 109 moved |

Two traps the conversion now encodes: the checkpoint's own `train_ds` names the transcript field
`answer` (left inherited, NeMo would have trained on an absent field: declared `text_field: text`
explicitly), and NeMo's lhotse loader refuses multi-channel audio (the OOD builder now writes mono).

## Held-out WER after fine-tuning (pnc=no)

| Set | Base | lr 1e-5 | lr 2e-6 |
|---|---|---|---|
| dev-clean | 1.43% | 1.43% | 1.52% |
| VoxPopuli | 5.32% | 5.67% | 5.76% |
| AMI | 16.24% | 30.83% | 39.61% |
| Earnings-22 | 23.39% | 37.91% | 35.29% |

The out-of-domain regression is located, not a framework defect. On AMI the base model already
produces runaway hallucinations on 3 of 300 clips (e.g. "Mm." -> "mm hm hm hm ..."); after
LibriSpeech-only fine-tuning it does so on 13, almost all one-word backchannels ("Yeah." ->
"Madam and i will be here to put it in the future of the future ..."). Those clips carry about 85%
of the regression:

| AMI | all 300 | without the 14 runaway-prone clips (n=286) |
|---|---|---|
| base | 16.2% | 13.4% |
| fine-tuned (lr 1e-5) | 30.8% | 15.7% |

Lowering the learning rate did not help (step size is not the cause), and the canary2 prompt is
the same in training and inference (NeMo's `encode_turn` maps manifest values to special tokens on
every turn). Read speech with no backchannels makes the attention decoder less robust on very
short conversational clips. The HF-native models did not show this, plausibly because their pure
bf16 updates rounded many small steps away while NeMo's fp32 master weights apply every step.
A production recipe would mix short conversational data in, or bound decoding length by audio
duration; that is follow-up work, not part of this milestone.
