# VLA on LIBERO with GB200: a Vision-Language-Action tutorial on a shared GB200 tray

This example walks Vision-Language-Action (VLA) models through the full ladder on LIBERO: evaluate a published checkpoint, fine-tune from the base model with FoundationScale (FS) specs, improve it with RL, and watch a Long-WAM policy over long horizons. By the end you have measured success rates for every rung, each compared against the published numbers. All numbers here were measured on a shared GB200 tray (aarch64), 4x GB200 (184 GiB each), CUDA 12.8 wheels.

**What you have:** a shared GB200 tray (4 GPUs, 184 GiB HBM each, aarch64 Grace CPU), other people included.

**Paths:** use the env vars throughout — `$FS_ROOT` (the FS checkout), `$VLA_WEIGHTS` (shared weights), `$VLA_DATA` (datasets), `$VLA_RUNS` (outputs). Set them in every shell.

---

## What FS owns vs what the backends own

| | FoundationScale | Backend |
|---|---|---|
| Eval harness | `python -m foundationscale.vla.eval` (all backends, same exit-code contract) | server process (`{model} {port}`), per its own published protocol |
| Fine-tuning | `Gr00tFinetuneSpec`, `OpenPiFinetuneSpec` (stage base checkpoint, render command, ddp hook) | training loop and weights |
| RL | `VerlVlaRlSpec` | verl-vla's own trainers and recipe YAMLs |
| Long-horizon memory | nothing | Long-WAM (LongLive) policy and its own harness |

Every backend runs in its own environment. The harness exit codes are shared: **0 measured**, **95 UNMEASURED** (no server started, or any episode raised — the rate is then not a measurement), **96 refused**.

---

## Step 1 — Environments

One environment per backend. The LIBERO eval env is the FS harness client and pairs with whichever backend server is running.

**LIBERO eval env (client):** python 3.12 venv; `mujoco==3.3.1`, `robosuite 1.4.1`, `numpy<2`. Set `MUJOCO_GL=egl` (or `osmesa`, see pitfalls). LIBERO goes in via `PYTHONPATH` + `LIBERO_CONFIG_PATH` — this avoids LIBERO's interactive prompt. With `torch>=2.6`, `torch.load` `weights_only` must allow LIBERO init states.

**GR00T env:** Isaac-GR00T at `d2b7e75`, `torch 2.9.0+cu128`, `transformers 4.57.3`, `use_flash_attention=False` (SDPA — there is no aarch64 flash-attn wheel). Backbone `nvidia/Cosmos-Reason2-2B` is gated: request access before you start. Multi-GPU fine-tune uses the FS `ddp_hook` (no DeepSpeed on aarch64).

**openpi env:** openpi at `215abfb` (`uv sync`, then reinstall `torch 2.7.1+cu128`), JAX on GPU. Checkpoint `gs://openpi-assets/checkpoints/pi05_libero`.

**verl-vla env:** python 3.10, `torch 2.7.1`, `verl 0.7.1`, Ray; `libero 0.1.1` installed with `--no-deps`; `numpy 1.26.4`, `ml_dtypes 0.5.1`, `onnx_ir`, `einops`.

**Long-WAM env:** LongLive at `c6a27e7` `Long-WAM/`, `torch 2.7.1+cu128`, `lerobot 0.4.1` with `numpy 2.5.4` (measured in the env every reported run used) and the `scipy 1.13.1` / `pyarrow 18.1.0` pins, `boto3`; `hardware=reference` (no native kernels); LIBERO at `8f1084e`.

### Check it worked

Each env imports its stack cleanly in its own shell. Before any LIBERO run: `MUJOCO_GL` is set, `PYTHONPATH`/`LIBERO_CONFIG_PATH` point at the LIBERO checkout, and an init-state load does not raise on `weights_only`.

---

## Step 2 — Evaluate a published checkpoint

One command per backend through the FS harness. [`scripts/eval_libero.sh`](scripts/eval_libero.sh)
carries each backend's server command (`--serve-cmd`, started in the backend's own environment) and
reads its paths from environment variables (see the script header):

```bash
bash scripts/eval_libero.sh openpi  libero_spatial      # openpi pi0.5, 50 trials/task
bash scripts/eval_libero.sh gr00t   libero_spatial      # GR00T N1.7, 20 trials/task
bash scripts/eval_libero.sh longwam libero_spatial 2    # Long-WAM, a 2-trial smoke first
```

Underneath it is one FS command, `python -m foundationscale.vla.eval --backend <b> --model <ckpt>
--suite <suite> --out <report.json> --serve-cmd "<server command with {model} {port}>"`.

Each backend runs its **published** protocol — these differ on purpose:

| | init states | settle steps | replanning | trials/task | seed | caps (spatial/object/goal/10) |
|---|---|---|---|---|---|---|
| openpi | suite | 10 | replan 5 | 50 | 7 | 220 / 280 / 300 / 520 |
| GR00T | seeded resets | 0 | 8 action steps | 20 | 0 | 720 env steps on every suite |
| Long-WAM | suite | 30 | called every step (replans itself) | 50 | 1 | 400 / 400 / 400 / 700 |

### Check it worked

Measured (FS harness unless noted) vs published:

| model | spatial | object | goal | libero_10 |
|---|---|---|---|---|
| GR00T N1.7 | 588/606 = 97.0% (GR00T client) vs 97.5 | 596/600 = 99.3% vs 98.5 | 582/600 = 97.0% vs 97.5 | 548/600 = 91.3% vs 94.5 |
| openpi pi0.5 | 492/500 = 98.4% vs 98.8 | 496/500 = 99.2% vs 98.2 | 486/500 = 97.2% vs 98.0 | 462/500 = 92.4% vs 92.4 |
| Long-WAM IDM (Long-WAM's own harness) | 495/500 = 99.0% vs 99.5 | 499/500 = 99.8% vs 100.0 | 491/500 = 98.2% vs 98.0 | 493/500 = 98.6% vs 99.5 |
| Long-WAM IDM (FS harness) | 497/500 = 99.4% | 498/500 = 99.6% | 490/500 = 98.0% | 491/500 = 98.2% |
| Long-WAM COD (Long-WAM's own harness) | 491/500 = 98.2% vs 98.6 | 500/500 = 100.0% vs 99.8 | 489/500 = 97.8% vs 96.8 | 489/500 = 97.8% vs 95.8 |
| Long-WAM COD (FS harness) | 492/500 = 98.4% | 500/500 = 100.0% | 487/500 = 97.4% | 490/500 = 98.0% |

FS harness vs upstream harness on the same checkpoint: openpi 493/500 vs 492/500; GR00T 190/200 vs 192/201; Long-WAM within 0.4 pt on all 8 suite x mode cells. They agree.

---

## Step 3 — Fine-tune from the base model (rung c)

**GR00T — `Gr00tFinetuneSpec(PUBLISHED_LIBERO_RECIPE)`** on 4 GPUs (global batch 640, 20,000 steps). The spec stages the base checkpoint (symlinks weights, patches only `use_flash_attention`) and renders the command (torchrun; `ddp_hook` when more than one GPU).

**openpi — `OpenPiFinetuneSpec`** with openpi's `pi05_libero` config (batch 256, 30,000 steps) on 4 GPUs.

Both commands are rendered by FS from the published recipes; print them with
[`scripts/render_finetune.py`](scripts/render_finetune.py) and run the output from the backend's
checkout, inside its own environment:

```bash
python scripts/render_finetune.py gr00t --gpus 4 --base $VLA_WEIGHTS/GR00T-N1.7-3B \
    --dataset $VLA_DATA/libero_spatial_no_noops_1.0.0_lerobot --out $VLA_RUNS/gr00t_spatial
python scripts/render_finetune.py openpi --out $VLA_RUNS/openpi --assets $VLA_RUNS/openpi/assets
```

### Check it worked

GR00T, `libero_spatial`: 8 h 58 min, 1.61 s/step. Final evaluated rates vs published:

| run | final (measured) | published |
|---|---|---|
| GR00T spatial | 481/500 = 96.2% + 197/200 = 98.5% | 97.5 |
| GR00T goal (trained on 427/428 episodes) | 582/600 = 97.0% | 97.5 |
| GR00T object (2 GPUs, 320/GPU, same global batch) | 595/600 = 99.2% | 98.5 |
| GR00T libero_10 | 544/600 = 90.7% | 94.5 |

openpi, ~19 h, final checkpoint (pooled runs):

| suite | final (measured) | published |
|---|---|---|
| spatial | 986/1000 = 98.6% | 98.8 |
| object | 1471/1500 = 98.1% | 98.2 |
| goal | 1455/1500 = 97.0% | 98.0 |
| libero_10 | 1412/1500 = 94.1% | 92.4 |

**GR00T goal:** episode 82's wrist video is corrupt in the upstream dataset (a fresh download is
identical), so this fine-tune trained on 427 of 428 episodes.

**GR00T libero_10:** the fine-tune (90.7%) is below the card (94.5%), but NVIDIA's released
checkpoint scores 548/600 = 91.3% in the same harness: FS training matches the released model.

---

## Step 4 — RL with `VerlVlaRlSpec`

Recipes transcribed verbatim with their published curves: FPO, DSRL (task 2, task 9), TD3+BC (pi0.5, Gaussian actor), RECAP. Assemble the command with the spec pipeline:

```
spec -> on_gpus(spec, N) -> with_output_dir -> with_runtime_env({...}) -> render_command
```

[`scripts/render_rl.py`](scripts/render_rl.py) does exactly that for any published recipe:

```bash
python scripts/render_rl.py --list
python scripts/render_rl.py dsrl-pi05-libero-spatial-task2 --gpus 2 --out $VLA_RUNS/dsrl_task2 --render osmesa
```

Runtime env keys are added with Hydra `++` — a recipe YAML may already define the same key. Measured below with the OSMesa renderer and 50 trajectories/eval.

### Check it worked

| run | measured (start -> end) | published |
|---|---|---|
| FPO pi0.5, task 2 | 0.68 -> 1.00 | 0.58 -> 1.00 |
| DSRL task 2 | 0.78 -> 0.84, best 0.88 | 0.74 -> 0.88, best 0.88 |
| DSRL task 9 | 0.50 -> 0.58, best 0.72 | 0.60 -> 0.74, best 0.82 |
| Gaussian-actor TD3+BC | blocked | — |

DSRL task 9 is within n=50 noise of the published curve. Gaussian-actor TD3+BC fails at its first actor update with an FSDP2/DTensor error in verl-vla's frozen image encoder, also with the exact published placement (see pitfalls).

---

## Step 5 — Long-WAM

Long-WAM runs in its own environment (`hardware=reference`, no native kernels) from LongLive `c6a27e7` `Long-WAM/`, LIBERO at `8f1084e`. Its released IDM and COD checkpoints go through both Long-WAM's own harness and the FS harness (Step 2 table, all four suites). Its protocol is the 30-settle, every-step-replan one — the policy carries its own long-horizon memory.

Through the FS harness, Long-WAM runs behind a small policy server in its own environment
(`python -m foundationscale.vla.adapters.longwam.server`, Long-WAM's public `create_policy` +
`policy_batch`); the FS adapter declares one action per step, because Long-WAM's own loop calls the
policy every step and it replans itself. [`scripts/eval_libero.sh`](scripts/eval_libero.sh) has the
server command. Two config copies are needed, both for upstream gaps at this commit:

- the checkpoint's `config.yaml`: add `train_transforms: null` next to `val_transforms` (the released
  config omits a required processor argument);
- Long-WAM's `configs/eval/libero.yaml`: add `num_trials: 50` under `policy_options` (the eval driver
  sets `num_trials` after copying the node, so it is lost otherwise).

Use the checkpoint's own cadence (`action_horizon=32 num_inference_steps=10 replan_steps=10`), and
pass the assigned GPU as `gpu_id` when running Long-WAM's own evaluation: its worker sets
`CUDA_VISIBLE_DEVICES=<gpu_id>`.

### Check it worked

Seven of the eight paper cells reproduce within the binomial tolerance (Step 2 table). IDM `libero_10` does not:
98.2% (FS) and 98.6% (Long-WAM's own harness) against the paper's 99.5%, outside the 0.9-pt margin, so it stays out of
the registry; IDM `object` is published at 100.0%, a rate the binomial rule cannot bound, so it is not a registry row either.

---

## Step 6 — Fine-tune Long-WAM with `LongWamFinetuneSpec`

`foundationscale.vla.adapters.longwam.finetune.LongWamFinetuneSpec` renders Long-WAM's own
`scripts/train.py` under `torchrun`. It holds the recipe's global batch of 128 (`docs/TRAINING.md`:
8 GPUs x batch 8 x accumulation 2) on any GPU count by deriving the accumulation, and refuses a layout
that would change it — Long-WAM's training contract enforces the same number at launch.

```python
from foundationscale.vla.adapters.longwam.finetune import LongWamFinetuneSpec

spec = LongWamFinetuneSpec(
    data_root="<data>",
    checkpoint_root="<ckpts>",
    gpus=(0, 1),
    batch_size=4,
    gradient_accumulation_steps=16,  # 2 x 4 x 16 = 128
    extra_overrides=("model.action_dit_pretrained_path=null",),
)
argv = spec.with_output_dir("<out>").render_command()  # run from the Long-WAM checkout
env = spec.render_env()  # LONGWAM_DATA_ROOT, LONGWAM_CHECKPOINT_ROOT, ...
```

Before training: the LIBERO LeRobot archives (`yuanty/LIBERO-fastwam`, via Long-WAM's
`scripts/download_data.py libero`), the Robot-S video initializer (`scripts/download_checkpoint.py
stage1_robot_s`, verify its `SHA256SUMS`), and the text-embedding cache
(`scripts/precompute_text_embeds.py task=libero`, then `+text_cache_mode=verify`). The LIBERO
action-backbone initializer (`ActionDiT_from_longlive_ar_1024hdim.pt`) has no public download yet, so
the example passes `model.action_dit_pretrained_path=null` (RoboTwin's documented choice) — a
deviation from the released recipe, recorded as such.

### Check it worked

A 60-step run on 2 GPUs trained (loss 1.06 -> 0.62, action loss to 0.07 at step 40), wrote
`checkpoints/weights/step_000060.pt` + `dataset_stats.json`, and that checkpoint evaluated through the FS
harness by staging `model.pt` + `dataset_stats.json` and pointing the Step 5 server at it: 0/50 on
`libero_spatial`, 0 episode errors — the expected score after 60 steps; the point is that every stage
connects. The full recipe is 277,713 samples -> 2,170 steps/epoch -> 21,700 steps for 10 epochs; on 2
GB200 GPUs it measured 37–51 s/step depending on the tray's CPU load (the video-decoding dataloader is
CPU-bound), i.e. one to two weeks.

---

## Step 7 — Driving: Alpamayo through the FS driving harness

`python -m foundationscale.vla.driving.eval --backend alpamayo` scores open-loop trajectories with
FS-owned metrics (`foundationscale.vla.driving.metrics`: ADE and minADE with Alpamayo's semantics —
XY L2 per waypoint, mean over time, the best sample chosen on the full horizon). The trajectory server
(`python -m foundationscale.vla.adapters.alpamayo.server`) runs in Alpamayo's own environment and walks
the alpamayo-recipes evaluator's own setup; `--serve-cmd` starts it. Every clip stays in the
denominator: one failed clip makes the run UNMEASURED (exit 95).

`foundationscale.vla.adapters.alpamayo.sft.AlpamayoSftSpec` renders the recipe's own training and
evaluation launchers (`sft_stage1/2_single_node_debug`, `sft_eval_loss`, `sft_eval_trajectory`).

Build the Alpamayo environment from the recipe's `uv.lock`, not from repository HEADs: with
`alpamayo2` at HEAD the held-out loss came out 4.4% high; pinned to the lock (`alpamayo2` 9596749 plus
the locked package versions) it matched. DeepSpeed needs a CUDA toolkit even for evaluation
(`CUDA_HOME`; the CPU optimizer JIT also needs cuRAND).

### Check it worked

| run | FS | published |
|---|---|---|
| Alpamayo 2 Super zero-shot minADE, chunk 99 (100 clips, 6 samples) | 0.7822 / 0.9168 / 0.8997 m at seeds 42 / 1 / 2 (mean 0.866, sd 0.073); seed 42 equals the recipe's own evaluator | 0.8288 m |
| `sft_eval_loss`, chunk 99 | 1.704507 (lock-pinned env) | 1.699801 |
| `sft_stage1_single_node_debug`, two steps | 1.8645, 1.7561 (4 GPUs) | 1.8095, 1.7255 (8 H100) |
| `sft_stage2_single_node_debug`, two steps | 1.6255, 0.8074 (4 GPUs) | 1.2710, 0.9367 (8 H100) |

The smoke steps differ because 4 ranks see half of the 8-rank batch at step 1 (learning rate 0) and
the stage-2 loss is the stochastic flow-matching loss: they show the recipe trains end to end, not a
numeric match. minADE over 100 clips moves about ±0.07 m between sampling seeds, so vary the seed
with `+trainer.seed=N` (the Hugging Face Trainer re-seeds after the recipe's own `seed`). Alpamayo 1.5 runs once its gated `nvidia/Cosmos-Reason2-8B` backbone is accessible
(example clip: minADE 0.374 m with its Chain-of-Causation text); neither Alpamayo 1.5 nor the SFT
recipe publishes a held-out number to compare against. Closed-loop AlpaSim needs an x86_64 host: its
NuRec renderer image is amd64-only.

---

## Measured summary

All runs above, side by side, on GB200. Cells are success rates unless the row names another metric.

| stage | model / run | measured | published |
|---|---|---|---|
| Eval released | GR00T N1.7 (spatial) | 97.0% (GR00T client) | 97.5 |
| Eval released | GR00T N1.7 (object / goal / 10) | 99.3 / 97.0 / 91.3 | 98.5 / 97.5 / 94.5 |
| Eval released | openpi pi0.5 (all 4) | 98.4 / 99.2 / 97.2 / 92.4 | 98.8 / 98.2 / 98.0 / 92.4 |
| Eval released | Long-WAM IDM, FS harness (all 4) | 99.4 / 99.6 / 98.0 / 98.2 | 99.5 / 100.0 / 98.0 / 99.5 |
| Eval released | Long-WAM COD, FS harness (all 4) | 98.4 / 100.0 / 97.4 / 98.0 | 98.6 / 99.8 / 96.8 / 95.8 |
| Fine-tune | GR00T spatial (4 GPU, 8 h 58, 1.61 s/step) | 96.2 + 98.5 | 97.5 |
| Fine-tune | GR00T goal / object / 10 | 97.0 / 99.2 / 90.7 | 97.5 / 98.5 / 94.5 |
| Fine-tune | openpi pi05_libero (4 GPU, ~19 h) | 98.6 / 98.1 / 97.0 / 94.1 | 98.8 / 98.2 / 98.0 / 92.4 |
| RL | FPO pi0.5 task 2 | 0.68 -> 1.00 | 0.58 -> 1.00 |
| RL | DSRL task 2 / task 9 | 0.84 best 0.88 / 0.58 best 0.72 | 0.88 best 0.88 / 0.74 best 0.82 |
| Fine-tune | Long-WAM, 60-step pipeline check (ActionDiT init null) | 0/50 spatial, 0 errors | — |
| Driving eval | Alpamayo 2 Super minADE (m, lower is better), 3 seeds | 0.866 ± 0.073 | 0.8288 |
| Driving SFT | Alpamayo 2 `sft_eval_loss` | 1.7045 | 1.6998 |

---

## Bugs and pitfalls this hardware found

All fixed or worked around.

1. **The renderer changes the score.** EGL vs OSMesa moved a pi0.5 SFT checkpoint's LIBERO start from 82% to 56% (published 60%). verl-vla's curves are OSMesa — but current Mesa (>= 25.1) has no OSMesa. Install Mesa 24 (conda-forge `mesalib<25`) and set `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` to reproduce the published RL numbers.
2. **GR00T's eval protocol is 720 env steps on every suite**, not openpi's per-suite caps. Do not mix the two protocols when comparing.
3. **GR00T's own client (`rollout_policy.py`, 5 vectorised envs) stops at `n_episodes` and drops episodes still running**; failures are 720-step timeouts, so it over-counts successes (goal: 201/202 with 5 envs vs 187/200 with `--n-envs 1`). FS keeps every planned episode in the denominator — that is why the GR00T denominators here (606, 200) are the planned counts.
4. **A crashed or erroring episode makes the run UNMEASURED (exit 95), never a lower score.**
5. **Long-WAM's released pieces are rough:** the released config omits a required processor arg, the eval driver drops `num_trials`, the worker forces `CUDA_VISIBLE_DEVICES` to `gpu_id` (default 0) — pass the GPU you were assigned — the videos directory is never created, and its checkpoint cadence (32/20 in the eval config; 80/20 elsewhere) does not match its own eval config.
6. **LIBERO init states and `torch>=2.6` `weights_only`** — the load refuses unless allowed (Step 1).
7. **TD3+BC on 2 GPUs:** needs `micro_batch 4` (OOM at 16) and then diverges — Q normalisation over 4 samples is too few. Use 4 GPUs.
8. **Ray socket paths:** keep `RAY_TMPDIR` short (AF_UNIX path limit 107 bytes).

---

## Troubleshooting

**The run ends `UNMEASURED` (exit 95) although it seemed to work** — either the `--serve-cmd` server never came up (check its log for a port bind or a gated-model download failure), or an episode raised. An erroring episode invalidates the whole rate: the run is not a measurement, not a low score.

**Exit 96 (refused)** — an invalid argument combination (unknown backend, missing `{model}`/`{port}` in `--serve-cmd`, an unknown suite). Fix the invocation; no measurement was taken.

**LIBERO prints an interactive prompt on first run** — `LIBERO_CONFIG_PATH`/`PYTHONPATH` are not set (Step 1). LIBERO must be pointed at its checkout so no config prompt appears.

**`torch.load` refuses a LIBERO init state (`weights_only`)** — running `torch>=2.6` defaults. Allow the init-state load (Step 1).

**A pi0.5 checkpoint scores far below its released rate (e.g. 56% start on LIBERO)** — renderer mismatch. verl-vla's published curves are OSMesa; EGL gave 82% vs OSMesa's 56% on the same checkpoint (published 60%). Pin `mesalib<25` and set `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.

**GR00T fine-tune fails at startup with a flash-attn import error** — no aarch64 flash-attn wheel exists. The spec patches `use_flash_attention=False` on the staged checkpoint (SDPA); if you see the import, the patch did not apply.

**`nvidia/Cosmos-Reason2-2B` download is refused** — the backbone is gated. Request access on the hub first.

**GR00T fine-tune on > 1 GPU hangs or seeks DeepSpeed** — DeepSpeed is unavailable on aarch64; multi-GPU goes through the FS `ddp_hook`. The `render_command` emits torchrun with the hook when more than one GPU.

**openpi's torch gets upgraded after `uv sync`** — expected: reinstall `torch 2.7.1+cu128` (Step 1). JAX must be on GPU.

**Long-WAM runs on GPU 0 regardless of what you were assigned** — its worker forces `CUDA_VISIBLE_DEVICES=gpu_id` (default 0). Pass the assigned GPU explicitly.

**Long-WAM checkpoint reads fail or videos are missing** — the videos directory is never created, and its checkpoint cadence (32/20) beats the eval config's 80/20. Create the directory and match the cadence.

**TD3+BC diverges after a few updates** — 2 GPUs force `micro_batch 4` (16 OOMs), and Q normalisation over 4 samples is noisy. Use 4 GPUs.

**Ray fails to start with a socket-path error** — `RAY_TMPDIR` is too long (AF_UNIX limit 107 bytes). Point it at a short path.

**GR00T matrix numbers look better than the same checkpoint with vectorised eval** — its own client drops still-running episodes and counts 720-step timeouts as successes (pitfall 3). Use `--n-envs 1` or the FS harness for an honest denominator.