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

## SP-5: multi-speaker VoxCPM2 LoRA (2026-10-16/17)

ROADMAP SP-5 asks for speaker similarity up on the fine-tuned speakers without losing zero-shot
similarity or intelligibility. Upstream Qwen3-TTS fine-tuning is single-speaker with no cloning
(T2a), so the criterion is tested on VoxCPM2, the family where T3 showed LoRA adapts a voice.
`cluster/build_sp5.py` takes the 8 LibriTTS-R train-clean-100 speakers with the most utterances
(21-33 min each, 2,740 train rows), 20 held out per speaker (160 rows), one fixed 6-12 s prompt clip
per speaker; `cluster/run_sp5.sh` trains the upstream LoRA config for 3 epochs (514 steps, final step
fixed in advance) and runs both evaluations.

| check (paired bootstrap, LoRA minus base) | base | LoRA | difference, 95% CI | criterion |
|---|---|---|---|---|
| held-out SIM, 8 fine-tuned speakers (n=160) | 49.91 | 52.95 | +3.05 [+1.59, +4.58] | met |
| held-out WER (n=160) | 2.939 | 2.493 | -0.45 [-1.77, +0.54] | no regression |
| zero-shot WER, Seed-TTS test-en (n=1,088) | 1.706 | 1.728 | [-0.23, +0.34] | no regression |
| zero-shot SIM, Seed-TTS test-en | 75.0 | 74.05 | -0.95 [-1.19, -0.70] | **not met** |

Training on 8 speakers instead of 1 does not remove the zero-shot similarity cost (-0.95 here vs
-0.78 for T3's single speaker). So SP-5 is not met by a single merged model. What is met: serving the
LoRA as an adapter loaded only for the fine-tuned voices (zero-shot requests use the base weights, so
their scores are T1's by construction). Next lever to test: whether a lower LoRA rank or fewer steps
keeps the held-out gain while shrinking the zero-shot cost.

### SP-5 sweep: LoRA rank vs the zero-shot cost (2026-10-17)

Same data, evaluation and paired bootstraps (`run_sp5.sh` with `RUN`/`RANK`/`EPOCHS`; alpha = rank).

| variant | held-out SIM gain (n=160) | held-out WER diff | zero-shot SIM diff (n=1,088) | zero-shot WER diff |
|---|---|---|---|---|
| r32, 3 epochs (above) | +3.05 [+1.59, +4.58] | [-1.77, +0.54] | -0.95 [-1.19, -0.70] | [-0.23, +0.34] |
| **r8, 3 epochs** | **+2.49 [+0.95, +4.14]** | [-1.78, +0.46] | **-0.28 [-0.53, -0.03]** | [-0.20, +0.16] |
| r32, 1 epoch | +1.18 [-0.32, +2.79] | [-1.32, +1.15] | -0.47 [-0.71, -0.22] | [-0.26, +0.08] |

Rank, not training length, sets the trade-off: r8 keeps most of the speaker gain and cuts the
zero-shot cost by ~70%, still just significant. Because this sweep reads Seed-TTS test-en, picking a
variant from it is selection on that set. Rule fixed before the r4 result: the chosen variant is the
one with the smallest zero-shot SIM cost whose held-out SIM gain CI excludes 0; it is then confirmed
on a set the sweep never read (Seed-TTS test-zh, zero-shot SIM and CER) before SP-5 is called met.

| r4, 3 epochs (run after the rule was fixed) | +1.90 [+0.51, +3.44] | [-1.63, +0.75] | -0.33 [-0.57, -0.08] | **+0.36 [+0.14, +0.58]** |

r4 is no cheaper on zero-shot SIM than r8 and regresses zero-shot WER, so the fixed rule picks **r8**.

**Confirmation on Seed-TTS test-zh** (2,020 rows, never read by the sweep), r8 LoRA vs base VoxCPM2:
SIM 79.24 vs 79.22, paired diff [-0.11, +0.15]; CER 0.954 vs 0.946, paired diff [-0.08, +0.10].
No zero-shot loss on the independent set. **SP-5 verdict:** with LoRA rank 8, speaker similarity on
the 8 fine-tuned speakers rises +2.49 [+0.95, +4.14] with no WER loss, and zero-shot similarity and
intelligibility are unchanged on the confirmation set. The one residual cost is on English zero-shot,
the language of the fine-tuned speakers and the set used for selection: SIM -0.28 [-0.53, -0.03].
