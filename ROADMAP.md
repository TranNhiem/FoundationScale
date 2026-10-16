# Roadmap

The goal is every training stage working for every model family, on a platform that scales.
This page is the shared to-do list: work is grouped into **workstreams**, and every item has a
stable ID, a status, where to start, and what "done" means.

## How to use this roadmap

1. **Find your workstream** in the index below; each one names the code it owns.
2. **Pick an item** by ID (for example `ARL-1`). Read its *Goal*, *Start from* and *Done when*.
3. **Update the status** when you start (`Planned` → `In progress`) and when it lands (`Available`).
   In the same change, update the [coverage table](#coverage) and, for new stage/family support,
   the README model-family table and the [training-stages figure](assets/training_stages.png).
4. Items marked **With:** need a second workstream; coordinate before changing its code.

### Workstream index

| ID prefix | Workstream | Owns | Items |
|---|---|---|---|
| `SP` | [Speech — ASR, TTS, Omni](#sp--speech-asr-tts-omni) | `src/foundationscale/train/audio.py`, `src/foundationscale/upstream/nemo/`, speech/TTS/omni examples | SP-1 … SP-7 |
| `MM` | [Multimodal LLM](#mm--multimodal-llm) | image/video paths in `src/foundationscale/train/`, `src/foundationscale/video.py` | MM-1 … MM-4 |
| `VLA` | [Vision-language-action](#vla--vision-language-action) | `src/foundationscale/vla/` | VLA-1 … VLA-3 |
| `RL` | [Preference & RL core](#rl--preference--rl-core) | `src/foundationscale/rl/` | RL-1 … RL-4 |
| `ARL` | [Agentic RL](#arl--agentic-rl) | `src/foundationscale/agentic_rl/` | ARL-1 … ARL-10 |
| `PT` | [Pre-training & CPT](#pt--pre-training--cpt) | CPT path in `src/foundationscale/train/`, FoundationSkills CPT stage | PT-1 … PT-2 |
| `FSK` | [FoundationSkills](#fsk--foundationskills) | `FoundationSkills/` | FSK-1 … FSK-4 |
| `PLAT` | [Platform — distributed, checkpoints, releases](#plat--platform) | `src/foundationscale/topology.py`, `src/foundationscale/checkpoint/`, `launchers/` | PLAT-1 … PLAT-4 |

## Coverage

Which training stage works for which model family today. Planned cells link to their item.

| Stage | Speech | LLM | Multimodal LLM | VLA |
|---|---|---|---|---|
| Pre-training & CPT | Planned — [SP-1](#sp-1--speech-continued-pre-training) | Available (CPT) | Planned — [MM-1](#mm-1--multimodal-continued-pre-training) | Planned — [VLA-1](#vla-1--vla-pre-training) |
| SFT | Available | Available | Available | Available |
| Preference | Planned — [SP-2](#sp-2--speech-preference-optimization) | Available | Available | Planned — [VLA-2](#vla-2--vla-preference-optimization) |
| RL | Planned — [SP-3](#sp-3--speech-rl) | Available | Available | Available |
| Agentic RL | Planned — [ARL-4](#arl-4--speech-agentic-rl) | Available | Planned — [ARL-2](#arl-2--multimodal-agentic-rl) | Planned — [ARL-3](#arl-3--vla-agentic-rl) |

## Definition of done

An item becomes **Available** when:

1. It runs through the existing entry points (`foundationscale-train`, the RL trainers, the agentic RL CLI or a VLA adapter), not a one-off script.
2. Unit tests cover the new path, and any new gate has controls that make it fire.
3. A real-GPU run beats its base model on a held-out set, with the numbers recorded under `validation_campaigns/`.
4. A step-by-step guide exists under `examples/` (for user-facing features), and the README and figures are updated.

---

## SP — Speech (ASR, TTS, Omni)

#### SP-1 · Speech continued pre-training
**Status:** Planned · **With:** PT
- **Goal:** continued pre-training of audio-LLMs on large unlabeled or weakly labeled audio, before ASR/TTS fine-tuning.
- **Start from:** the audio column in [src/foundationscale/train/audio.py](src/foundationscale/train/audio.py) and the CPT stage FoundationSkills emits ([emit_train.py](FoundationSkills/foundationskills/interfaces/fs/emit_train.py)).
- **Done when:** CPT lowers held-out loss and, after the usual SFT, matches or beats the SFT-only baseline on WER.

#### SP-2 · Speech preference optimization
**Status:** Planned · **With:** RL
- **Goal:** DPO-family training for speech — TTS pairs ranked by intelligibility and speaker similarity, ASR hypotheses ranked by WER.
- **Start from:** the preference trainer [src/foundationscale/rl/preference_trainer.py](src/foundationscale/rl/preference_trainer.py), which already takes images ([multimodal DPO test](validation_campaigns/vlm_2xh200/test_preference_trainer_multimodal_dpo.py)), and the evaluation in `validation_campaigns/tts_eval/` and `validation_campaigns/speech_eval/`.
- **Done when:** preference training on audio improves WER (ASR) or Seed-TTS-eval scores (TTS) over the SFT model.

#### SP-3 · Speech RL
**Status:** Planned · **With:** RL
- **Goal:** GRPO-family RL with speech rewards — WER/CER for ASR, intelligibility and speaker similarity for TTS.
- **Start from:** [src/foundationscale/rl/trainer.py](src/foundationscale/rl/trainer.py), [grpo.py](src/foundationscale/rl/grpo.py), [rewards.py](src/foundationscale/rl/rewards.py) and WER from [speech_metrics.py](src/foundationscale/train/speech_metrics.py).
- **Done when:** RL improves held-out WER (or TTS scores) over the SFT model with no runaway or repetition failures.

#### SP-4 · Qwen3-ASR-0.6B recipe
**Status:** Planned
- **Goal:** train the smaller Qwen3-ASR through the same process as the 1.7B model.
- **Start from:** the speech guide [examples/speech-2xH200](examples/speech-2xH200/README.md).
- **Done when:** a measured WER improvement over the base model, added to the guide.

#### SP-5 · Multi-speaker Qwen3-TTS fine-tuning
**Status:** Planned
- **Goal:** fine-tune Qwen3-TTS on many speakers while keeping zero-shot voice cloning (the upstream recipe is single-speaker).
- **Start from:** [examples/tts-GB200](examples/tts-GB200/README.md) and the plan in [docs/research/tts_omni_plan.md](docs/research/tts_omni_plan.md).
- **Done when:** speaker similarity improves on the fine-tuned speakers without losing zero-shot similarity or intelligibility.

#### SP-6 · Vietnamese speech output for omni models
**Status:** Planned
- **Goal:** an omni model that answers in natural Vietnamese speech; today its talker is much weaker in Vietnamese than a fine-tuned TTS.
- **Start from:** [examples/omni-GB200](examples/omni-GB200/README.md); compare against the ASR → LLM → TTS cascade baseline.
- **Done when:** the omni model's Vietnamese speech WER and naturalness reach the cascade baseline.

#### SP-7 · Omni fine-tuning
**Status:** Planned
- **Goal:** fine-tune the omni thinker with the talker frozen, then train speech output.
- **Start from:** [docs/research/tts_omni_plan.md](docs/research/tts_omni_plan.md).
- **Done when:** VoiceBench improves over the base omni model.

Related speech items elsewhere: [ARL-4](#arl-4--speech-agentic-rl) speech agentic RL.

---

## MM — Multimodal LLM

#### MM-1 · Multimodal continued pre-training
**Status:** Planned · **With:** PT
- **Goal:** continued pre-training of VLMs on image-text and interleaved corpora, including video frames.
- **Start from:** the image arm of the training loop and frame sampling in [src/foundationscale/video.py](src/foundationscale/video.py).
- **Done when:** held-out loss improves on the target domain without regressing a general VLM benchmark.

#### MM-2 · Native video training
**Status:** Planned
- **Goal:** train video as video (temporal encoders, video processors), beyond sampling frames onto the image path.
- **Start from:** [src/foundationscale/video.py](src/foundationscale/video.py) and the video refusal in `src/foundationscale/train/loop.py`.
- **Done when:** a video model trains with its video tower measured as updated, and beats frame sampling on a video benchmark.

#### MM-3 · Standard VLM benchmark harness
**Status:** Planned
- **Goal:** one command to score a checkpoint on standard VLM benchmarks.
- **Start from:** the evaluation chapter [examples/vlm-2xH200/05_evaluation.md](examples/vlm-2xH200/05_evaluation.md).
- **Done when:** the harness reproduces a published model-card score within tolerance.

#### MM-4 · Faster mixed-modality preference training
**Status:** Planned · **With:** RL
- **Goal:** remove the slow rows in mixed image/text DPO on Qwen VLMs.
- **Start from:** [validation_campaigns/vlm_2xh200/EVIDENCE.md](validation_campaigns/vlm_2xh200/EVIDENCE.md).
- **Done when:** mixed-modality DPO runs at the same step time as image-only DPO.

Related multimodal items elsewhere: [ARL-2](#arl-2--multimodal-agentic-rl) multimodal agentic RL, [RL-4](#rl-4--multimodal-rl-on-the-megatron-lane) multimodal RL on Megatron.

---

## VLA — Vision-language-action

#### VLA-1 · VLA pre-training
**Status:** Planned · **With:** PT
- **Goal:** pre-train or broadly continue-train a VLA policy on many robot datasets before task fine-tuning.
- **Start from:** [src/foundationscale/vla/lerobot.py](src/foundationscale/vla/lerobot.py) and the GR00T / openpi adapters in `src/foundationscale/vla/adapters/`.
- **Done when:** the pre-trained policy fine-tunes to a higher LIBERO success rate than the same recipe from the base checkpoint.

#### VLA-2 · VLA preference optimization
**Status:** Planned · **With:** RL
- **Goal:** prefer successful over failed action trajectories.
- **Start from:** LIBERO rollouts from `src/foundationscale/vla/eval/` labeled by success, fed to the preference trainer.
- **Done when:** a preference-trained policy beats its SFT starting point on LIBERO success rate.

#### VLA-3 · Reproduce DSRL, TD3+BC and RECAP
**Status:** Planned
- **Goal:** reproduce the published curves of the remaining verl-vla recipes, as was done for FPO.
- **Start from:** the verl-vla adapter in `src/foundationscale/vla/adapters/verlvla/` and the reproduction table in [docs/research/vla.md](docs/research/vla.md).
- **Done when:** each recipe lands within tolerance of its published curve on several seeds.

Related VLA items elsewhere: [ARL-3](#arl-3--vla-agentic-rl) VLA agentic RL.

---

## RL — Preference & RL core

#### RL-1 · Measured LLM preference & RL recipes
**Status:** Planned
- **Goal:** a text-LLM guide that runs DPO-family and GRPO-family training with hardware results (multimodal already has one).
- **Start from:** [examples/llm-2xA100](examples/llm-2xA100/README.md) and [docs/RL_ALGORITHMS.md](docs/RL_ALGORITHMS.md).
- **Done when:** the guide reports measured gains over the SFT model for at least one preference and one RL algorithm.

#### RL-2 · DAPO dynamic sampling
**Status:** Planned
- **Goal:** implement DAPO's group filtering (`filter_groups`) and rollout refill, which are not implemented yet.
- **Start from:** [src/foundationscale/rl/group_policy.py](src/foundationscale/rl/group_policy.py) and `group_policy_objectives.py`.
- **Done when:** groups with identical rewards are filtered and refilled, with tests and a run that shows the effect.

#### RL-3 · Rewards beyond multiple choice
**Status:** Planned · **With:** FSK
- **Goal:** verifiable rewards for free-form answers (math, code, short answers); RL data is single-letter multiple choice today.
- **Start from:** [src/foundationscale/rl/rewards.py](src/foundationscale/rl/rewards.py), [corpus.py](src/foundationscale/rl/corpus.py) and the RL formatting in FoundationSkills' data engine.
- **Done when:** GRPO trains on a free-form math set and improves held-out accuracy.

#### RL-4 · Multimodal RL on the Megatron lane
**Status:** Planned · **With:** MM
- **Goal:** image inputs in the Megatron-Bridge RL lane, for large multimodal models.
- **Start from:** `src/foundationscale/rl/megatron/`.
- **Done when:** a multimodal GRPO run completes on the Megatron lane and matches the FSDP2 path on a small model.

---

## ARL — Agentic RL

#### ARL-1 · Environment `reset` / `step(action)` interface
**Status:** Planned · **With:** VLA
- **Goal:** an environment interface for non-shell tasks; today environments execute shell commands only.
- **Start from:** [src/foundationscale/agentic_rl/envs/base.py](src/foundationscale/agentic_rl/envs/base.py) (`execute(command)`).
- **Done when:** a non-shell environment (a simulator) runs through the same trainer, with tests. Unblocks ARL-3.

#### ARL-2 · Multimodal agentic RL
**Status:** Planned · **With:** MM
- **Goal:** multi-turn tool use where prompts and observations include images or video frames; tool turns are text-only today.
- **Start from:** the trajectory format [trajectory_v2.py](src/foundationscale/agentic_rl/trajectory_v2.py) and the harness in `src/foundationscale/agentic_rl/harness/`.
- **Done when:** agentic GRPO on a visual tool-use task improves task success over the starting VLM.

#### ARL-3 · VLA agentic RL
**Status:** Planned · **With:** VLA · **Needs:** ARL-1
- **Goal:** multi-step embodied tasks where the policy acts in a simulator as an agentic environment.
- **Start from:** ARL-1 plus the LIBERO harness in `src/foundationscale/vla/eval/`.
- **Done when:** agentic RL on a LIBERO suite raises success rate over the fine-tuned policy.

#### ARL-4 · Speech agentic RL
**Status:** Planned · **With:** SP
- **Goal:** voice agents — speech in, tool calls and spoken or text answers out.
- **Start from:** the omni recipe [examples/omni-GB200](examples/omni-GB200/README.md) and the agentic RL harness.
- **Done when:** agentic RL improves task success on a spoken tool-use benchmark over the base omni model.

#### ARL-5 · Rollout staleness tracking
**Status:** Planned
- **Goal:** record the policy version of every in-flight rollout so the staleness gate can fire; the rollout host does not track it yet.
- **Start from:** [src/foundationscale/agentic_rl/rollout_host.py](src/foundationscale/agentic_rl/rollout_host.py) and the gate in [src/foundationscale/gates/agentic_gates.py](src/foundationscale/gates/agentic_gates.py).
- **Done when:** the gate receives real policy versions and blocks an over-stale batch in a test and a live run.

#### ARL-6 · In-place weight reload for vLLM
**Status:** Planned
- **Goal:** update rollout weights without restarting the engine, reliably on hybrid-attention models.
- **Start from:** [src/foundationscale/agentic_rl/engines/vllm.py](src/foundationscale/agentic_rl/engines/vllm.py) and [weight_sync.py](src/foundationscale/agentic_rl/weight_sync.py).
- **Done when:** a long run reloads weights every step with no engine hangs, faster than restart mode.

#### ARL-7 · SGLang rollout engine on hardware
**Status:** Planned
- **Goal:** validate the SGLang client end to end, so the fleet is not vLLM-only.
- **Start from:** [src/foundationscale/agentic_rl/engines/sglang.py](src/foundationscale/agentic_rl/engines/sglang.py).
- **Done when:** an agentic run completes on SGLang with results matching vLLM.

#### ARL-8 · Directly servable checkpoints
**Status:** Planned
- **Goal:** published agentic-RL checkpoints load in vLLM without config edits.
- **Start from:** the checkpoint publishing in the agentic trainer and [src/foundationscale/agentic_rl/servable.py](src/foundationscale/agentic_rl/servable.py).
- **Done when:** a published checkpoint serves in vLLM as-is.

#### ARL-9 · Multi-seed reference comparison
**Status:** Planned
- **Goal:** compare FoundationScale's agentic GRPO with an external reference implementation over at least three seeds.
- **Start from:** [validation_campaigns/agentic_rl_verl/EVIDENCE.md](validation_campaigns/agentic_rl_verl/EVIDENCE.md).
- **Done when:** both reach the same held-out score within seed variance.

#### ARL-10 · Asynchronous rollouts and NCCL weight sync
**Status:** Planned
- **Goal:** overlap rollout and training (one-step-off) and sync weights over NCCL instead of disk.
- **Start from:** [weight_sync.py](src/foundationscale/agentic_rl/weight_sync.py) and the rollout host.
- **Done when:** step time drops on the same task with no loss in reward.

---

## PT — Pre-training & CPT

#### PT-1 · Curated from-scratch pre-training recipes
**Status:** Planned
- **Goal:** recipes that pre-train an LLM from scratch (today pre-training runs as continued pre-training).
- **Start from:** the CPT path through `foundationscale-train` and [docs/TRAINING.md](docs/TRAINING.md).
- **Done when:** a small model pre-trains from scratch and reaches a reference loss or benchmark score.

#### PT-2 · Data mixtures and replay for CPT
**Status:** Planned · **With:** FSK
- **Goal:** declared data mixtures and replay ratios for CPT to limit forgetting.
- **Start from:** the CPT stage in [emit_train.py](FoundationSkills/foundationskills/interfaces/fs/emit_train.py).
- **Done when:** CPT with replay keeps general benchmarks within tolerance while improving the target domain.

Related pre-training items elsewhere: [SP-1](#sp-1--speech-continued-pre-training), [MM-1](#mm-1--multimodal-continued-pre-training), [VLA-1](#vla-1--vla-pre-training).

---

## FSK — FoundationSkills

#### FSK-1 · `auto_research` CLI
**Status:** Planned
- **Goal:** an `fskills` subcommand for auto_research; today an agent drives it through the Python API.
- **Start from:** [FoundationSkills/README.md](FoundationSkills/README.md) and the auto_research skill.
- **Done when:** a campaign runs from the CLI with the same confirmation flow as `fskills launch`.

#### FSK-2 · Ledger anchoring and approver binding
**Status:** Planned
- **Goal:** make a full rewrite of the research ledger detectable and bind each approval to its approver.
- **Start from:** the limitations section of the auto_research `SKILL.md`.
- **Done when:** a rewritten ledger is detected in a test.

#### FSK-3 · Plan the new stage/family combinations
**Status:** Planned
- **Goal:** as each roadmap item lands, register it so `fskills probe` and `fskills plan` can use it.
- **Start from:** [FoundationSkills/foundationskills/interfaces/fs/capabilities.py](FoundationSkills/foundationskills/interfaces/fs/capabilities.py).
- **Done when:** the planner proposes the new combination only when the probe measures it.

#### FSK-4 · Free-form RL data
**Status:** Planned · **With:** RL
- **Goal:** the data engine emits free-form RL datasets once RL-3 lands.
- **Start from:** the RL formatting in the data engine.
- **Done when:** `data_engine` produces an RL dataset that RL-3's rewards score.

---

## PLAT — Platform

#### PLAT-1 · Multi-node FSDP
**Status:** Planned
- **Goal:** verified multi-node FSDP2 training.
- **Start from:** [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md) and [src/foundationscale/topology.py](src/foundationscale/topology.py).
- **Done when:** a two-node run matches single-node loss curves and saves a checkpoint that reloads.

#### PLAT-2 · Multi-rank checkpoints in CI
**Status:** Planned
- **Goal:** test distributed checkpoint save and reload with several ranks in CI, not only single-process.
- **Start from:** [docs/CHECKPOINTING.md](docs/CHECKPOINTING.md).
- **Done when:** CI saves and reloads a multi-rank checkpoint on every change.

#### PLAT-3 · Unified data contract
**Status:** Planned
- **Goal:** one dataset layer shared by every stage and model family.
- **Start from:** [docs/DATASETS.md](docs/DATASETS.md) and [docs/deliverables/D_roadmap.md](docs/deliverables/D_roadmap.md).
- **Done when:** at least SFT, preference and RL read the same dataset format.

#### PLAT-4 · Container image and model releases
**Status:** Planned
- **Goal:** an official container image, and reference checkpoints published on Hugging Face.
- **Start from:** [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) and the measured example recipes.
- **Done when:** the quick start runs from the container, and released checkpoints reproduce their guide's numbers.

---

Current state in detail: [docs/STATUS.md](docs/STATUS.md).
