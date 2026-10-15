# Qwen3-TTS single-speaker SFT (plan phase T2a, 2026-10-15)

Upstream `finetuning/sft_12hz.py` (qwen-tts 0.1.1) run unmodified through `qwen3tts_sft.py`, which
only swaps the hard-coded `flash_attention_2` for `sdpa` (flash-attn is not installed in this env) and
puts the upstream script dir on `sys.path`. Upstream supports one fixed new voice per run
(`custom_voice`); multi-speaker and cloning fine-tuning are not supported upstream.

Data (`cluster/build_t2a.py`): LibriTTS-R train-clean-100 (CC BY 4.0), the speaker with the most
utterances (6209): 398 train utterances (28 min), 60 held out, one fixed `ref_audio` on every row
as the README recommends. Held-out eval (`synth_qwen3tts_heldout.py`): the fine-tuned custom voice
against the base 1.7B model cloning from the same reference clip; judged by `judge_multilingual.py`
(Whisper-large-v3, SIM against the real held-out recording).

| run (README recipe: batch 32, 10 epochs) | held-out WER | SIM |
|---|---|---|
| base 1.7B, zero-shot clone | 2.309 | 44.75 |
| SFT lr 2e-6 (README value), final epoch | 2.165 | 44.47 |
| SFT lr 2e-5 (upstream script default) | 217.17 | 4.49 |

Paired bootstrap, SFT lr 2e-6 minus base clone (n=60): WER [-0.50, 0.00] points, SIM [-2.5, +1.8].
**Null result**: at the README learning rate, 28 minutes of one speaker gives a voice no more
intelligible or similar than zero-shot cloning. At 10x (the script's own default with the README's
batch and epochs) the model diverges into runaway generation. Single-speaker SFT is not worth
shipping for this model; zero-shot cloning plus the VoxCPM2 LoRA path (`tts_vi/`) is the recipe.
