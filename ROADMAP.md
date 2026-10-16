# Roadmap

The [training-stages figure](assets/training_stages.png) shows the goal: every training stage working
for every model family. This page tracks the combinations that are not built yet, so each one can be
picked up as its own piece of work.

## Coverage

| Stage | Speech | LLM | Multimodal LLM | VLA |
|---|---|---|---|---|
| Pre-training & CPT | Planned — [R1](#r1-speech-pre-training--cpt) | Available (CPT) | Planned — [R2](#r2-multimodal-pre-training--cpt) | Planned — [R3](#r3-vla-pre-training) |
| SFT | Available | Available | Available | Available |
| Preference | Planned — [R4](#r4-speech-preference-optimization) | Available | Available | Planned — [R5](#r5-vla-preference-optimization) |
| RL | Planned — [R6](#r6-speech-rl) | Available | Available | Available |
| Agentic RL | Planned — [R9](#r9-speech-agentic-rl) | Available | Planned — [R7](#r7-multimodal-agentic-rl) | Planned — [R8](#r8-vla-agentic-rl) |

**Available** means the stage runs in the framework today for that family. **Planned** items are
listed below with where to start and what "done" means.

## Definition of done

An item moves to **Available** when all of these hold:

1. It runs through the existing entry points (`foundationscale-train`, the RL trainers, the agentic RL CLI or a VLA adapter), not a one-off script.
2. Unit tests cover the new path, and any new gate has controls that make it fire.
3. A real-GPU run beats its base model on a held-out set, with the numbers recorded under `validation_campaigns/`.
4. A step-by-step guide exists under `examples/`, and the README model-family table and the figure are updated.

## Planned items

### R1. Speech pre-training & CPT

- **Goal:** continued pre-training of speech and audio-LLM models on large unlabeled or weakly labeled audio, before ASR/TTS fine-tuning.
- **Start from:** the audio column in [src/foundationscale/train/audio.py](src/foundationscale/train/audio.py) and the CPT stage FoundationSkills already emits ([emit_train.py](FoundationSkills/foundationskills/interfaces/fs/emit_train.py)).
- **Done when:** a CPT run on an audio-LLM lowers held-out loss and, after the usual SFT, matches or beats the SFT-only baseline on WER ([src/foundationscale/train/speech_metrics.py](src/foundationscale/train/speech_metrics.py)).

### R2. Multimodal pre-training & CPT

- **Goal:** continued pre-training of vision-language models on image-text and interleaved corpora, including video frames.
- **Start from:** the image arm of the training loop and frame sampling in [src/foundationscale/video.py](src/foundationscale/video.py); the CPT stage above.
- **Done when:** a VLM CPT run improves held-out loss on the target domain without regressing a general VLM benchmark.

### R3. VLA pre-training

- **Goal:** pre-train or broadly continue-train a VLA policy on many robot datasets before task fine-tuning.
- **Start from:** LeRobot data loading in [src/foundationscale/vla/lerobot.py](src/foundationscale/vla/lerobot.py) and the GR00T / openpi adapters in `src/foundationscale/vla/adapters/`.
- **Done when:** a policy pre-trained across several datasets fine-tunes to a higher LIBERO success rate than the same recipe from the base checkpoint, measured with `src/foundationscale/vla/eval/`.

### R4. Speech preference optimization

- **Goal:** DPO-family training for speech models — for example TTS pairs ranked by intelligibility and speaker similarity, or ASR hypotheses ranked by WER.
- **Start from:** the preference trainer [src/foundationscale/rl/preference_trainer.py](src/foundationscale/rl/preference_trainer.py), which already takes images ([multimodal DPO test](validation_campaigns/vlm_2xh200/test_preference_trainer_multimodal_dpo.py)), plus the TTS and ASR evaluation in `validation_campaigns/tts_eval/` and `validation_campaigns/speech_eval/`.
- **Done when:** preference training on audio inputs improves WER (ASR) or Seed-TTS-eval scores (TTS) over the SFT model.

### R5. VLA preference optimization

- **Goal:** prefer successful over failed action trajectories.
- **Start from:** LIBERO rollouts from `src/foundationscale/vla/eval/` labeled by success, fed to the preference trainer.
- **Done when:** a preference-trained policy beats its SFT starting point on LIBERO success rate.

### R6. Speech RL

- **Goal:** GRPO-family RL with speech rewards — WER/CER for ASR, intelligibility and speaker similarity for TTS.
- **Start from:** the RL trainers ([src/foundationscale/rl/trainer.py](src/foundationscale/rl/trainer.py), [grpo.py](src/foundationscale/rl/grpo.py)) and reward plumbing in [rewards.py](src/foundationscale/rl/rewards.py), with WER from [speech_metrics.py](src/foundationscale/train/speech_metrics.py).
- **Done when:** an RL run improves held-out WER (or TTS scores) over the SFT model without runaway or repetition failures.

### R7. Multimodal agentic RL

- **Goal:** multi-turn tool use where observations and prompts include images or video frames.
- **Start from:** the agentic RL harness and trajectory format ([trajectory_v2.py](src/foundationscale/agentic_rl/trajectory_v2.py)); today tool turns are text-only.
- **Done when:** an agentic GRPO run on a visual tool-use task improves task success over the starting VLM.

### R8. VLA agentic RL

- **Goal:** multi-step embodied tasks where the policy acts in a simulator as an agentic environment.
- **Start from:** the environment interface in [src/foundationscale/agentic_rl/envs/base.py](src/foundationscale/agentic_rl/envs/base.py), which is shell-shaped today (`execute(command)`) and needs a `reset` / `step(action)` form, with LIBERO from `src/foundationscale/vla/eval/` as the first environment.
- **Done when:** agentic RL on a LIBERO suite raises success rate over the fine-tuned policy.

### R9. Speech agentic RL

- **Goal:** voice agents — speech in, tool calls and spoken or text answers out.
- **Start from:** the omni speech-LLM recipe in [examples/omni-GB200](examples/omni-GB200/README.md) and the agentic RL harness.
- **Done when:** agentic RL improves task success on a spoken tool-use benchmark over the base omni model.

## Across all families

- **Curated from-scratch pre-training recipes** — today pre-training runs as CPT; full recipes are still to come.
- **Unified data contract** — one dataset layer for every stage and family (see [docs/deliverables/D_roadmap.md](docs/deliverables/D_roadmap.md)).
- **Measured LLM preference & RL recipes** — the algorithms are tested; add an `examples/` guide with hardware results.

Current state in detail: [docs/STATUS.md](docs/STATUS.md).
