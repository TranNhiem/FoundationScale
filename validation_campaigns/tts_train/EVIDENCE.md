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

## T3: VoxCPM2 LoRA speaker adaptation on the same speaker (2026-10-15)

Same data and held-out rows as T2a, upstream `voxcpm_finetune_lora.yaml` (LoRA r32 on LM and DiT, lr 1e-4,
effective batch 16) for 10 epochs = 250 steps, final step fixed in advance (`cluster/run_t3.sh`, audio
resampled to the AudioVAE's 16 kHz). Both arms clone from the T2a reference clip with
`synth_voxcpm2.py` (cfg 2.0, 10 timesteps); judged exactly like T2a. One GB200 GPU, 26 min end to end.

| run | held-out WER | SIM |
|---|---|---|
| VoxCPM2 zero-shot clone | 2.597 | 44.48 |
| VoxCPM2 + speaker LoRA, step 250 | 2.886 | 47.43 |

Paired bootstrap, LoRA minus base (n=60): SIM **+2.95, 95% CI [+1.44, +4.49]**; WER +0.29, CI [-0.50, +1.47]
(4 rows changed, one by +3 errors). Against the plan's criteria: SIM up at 95% is **met**; WER
non-regression within +0.3 holds on the point estimate but 60 rows cannot resolve it. So, unlike Qwen3-TTS
SFT, a 28-minute LoRA does move VoxCPM2 toward the target voice; a larger held-out set is needed before
claiming intelligibility is unchanged.

Does the speaker LoRA still clone other voices? Seed-TTS test-en (1,088 rows, other speakers' prompts)
with the LoRA loaded, scored by the T0 harness (`cluster/run_t3_zeroshot.sh`), against the T1 base run:
WER 1.624 vs 1.706 (paired diff [-0.27, +0.10]: no regression), SIM 74.22 vs 75.0 (paired diff -0.78,
[-1.03, -0.53]). Zero-shot cloning survives with a small, significant similarity cost, the same direction
and size as T2b's -1.07 on Vietnamese. Ship the speaker LoRA as an optional adapter, not merged weights.
