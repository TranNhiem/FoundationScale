# Speech T4: LoRA runs train and verify the audio tower

A LoRA run that declares an audio column now trains the audio towers in full (peft
`modules_to_save`) with LoRA on the language model -- the SALM shape -- and the speech gates
verify the audio tower moved in the ADAPTER checkpoint. Same run shape as `../speech_p1`
(FoundationScale `train()`, Gemma-4-E4B, one GB200 GPU, 5 steps, 64 LibriSpeech dev-clean rows)
plus `--adapter lora --adapter-rank 16 --learning-rate 1e-4`. Measured 2026-10-06.
Verdict: **PASS**.

## Why it needed a measured, per-family declaration

Without `modules_to_save`, peft freezes every non-target parameter, so before this change a
LoRA run with an audio column would have trained with its audio tower frozen. Two designs failed
on hardware before the one that shipped:

| Attempt | modules_to_save | Result on GB200 (peft 0.18.1) |
|---|---|---|
| 1 | tower roots `model.audio_tower`, `model.embed_audio` | `AuxiliaryTrainingWrapper.forward() missing 1 required positional argument: 'x'` -- Gemma-4 calls `embed_audio(inputs_embeds=...)` by keyword |
| 2 | every parameter-owning module under the audio towers (271) | same error -- the audio attention owns `per_dim_scale` and is called `self_attn(hidden_states=...)` |
| 3 | `FamilySpec.adapter_full_train = ("model.audio_tower", "model.embed_audio.embedding_projection")` | trains (below) |

peft's wrapper forwards one positional argument, so which modules can be wrapped depends on how
the family CALLS them, which the module tree does not show. It is therefore a measured field on
the family; a family without it refuses (96) an adapter run that declares audio.

## Result (commit 3837651)

| Item | Value |
|---|---|
| Announcement | `model.audio_tower, model.embed_audio.embedding_projection train in full (peft modules_to_save); LoRA covers the language model` |
| Trainable | 345,456,928 of 8,341,613,376 parameters (LoRA on 294 language-model modules + the audio tower and projector) |
| Loss by step | 14.93, 12.72, 15.62, 13.38, 15.29 |
| speech.audio_row_coverage | PASS 12/12 |
| speech.audio_placeholder_coverage | PASS 12/12 |
| speech.tower_movement `model.audio_tower` | PASS, **219 of 271** parameters moved (read from the adapter checkpoint) |
| speech.tower_movement `model.embed_audio` | PASS, **1 of 1** moved |
| Dormant-tower control | `[n/a]`: NOT_APPLICABLE under an adapter -- peft freezes non-target parameters by construction, and the adapter checkpoint does not carry frozen towers |
| Run | PASS |

The movement comparison works across peft's renaming because names are canonicalised (measured
peft 0.18.1 spellings: frozen original `base_model.model.<name with .original_module>`, trainable
copy `...modules_to_save.default...`, saved as `base_model.model.<name>`).

## Caveat

5 steps is a mechanism check, not an accuracy claim; loss on 5 noisy steps does not trend. WER
before and after training is P3.
