# VLA plane — supported models, design, and how to reproduce them

Status: 2026-10-10. Two VLA backends are `SUPPORTED` and one RL recipe is reproduced within tolerance;
all numbers below were measured on our GB200 trays (aarch64, Blackwell) between 2026-10-09 and
2026-10-10. "Reproduced" has its strict meaning here: an FS-side run matches the upstream's published
number under a tolerance fixed before the run, from the published rate and the trial counts.

## 1. What is supported

| entry | published | FS-measured | trials (pub / meas) | tolerance | verdict |
|---|---|---|---|---|---|
| GR00T N1.7 `libero_spatial` (`nvidia/GR00T-N1.7-LIBERO`) | 195/200 = 97.5% (Isaac-GR00T `examples/LIBERO/README.md`; the README labels it "97.65%", which is 97.5%) | 588/606 = 97.0% (95% CI ~95.7–98.4); passes 192/201, 198/202, 198/203 | 200 / 606 | 2.5 pt = `binomial_tolerance(0.975, 200, 606)` | REPRODUCED within tolerance — published rate inside our CI; the first pass's 2.1-point gap was sampling noise |
| openpi pi0.5 `pi05_libero` @30k | 494/500 = 98.8% (`examples/libero/README.md`) | 492/500 = 98.4% (per task 100/98/100/98/98/92/100/100/100/98) | 500 / 500 | 1.4 pt = `binomial_tolerance(0.988, 500, 500)` | REPRODUCED within tolerance (seed 7, 50 trials × 10 tasks) |
| GR00T N1.7 `libero_goal` | 195/200 = 97.5% | 582/600 = 97.0% (FS harness, seeds 0/1/2: 195, 195, 192) | 200 / 600 | 2.5 pt | REPRODUCED within tolerance |
| GR00T N1.7 `libero_object` | 197/200 = 98.5% (README "98.45%") | 596/600 = 99.3% (FS harness, seeds 0/1/2: 199, 199, 198) | 200 / 600 | 2.0 pt | REPRODUCED within tolerance |
| GR00T N1.7 `libero_10` | 189/200 = 94.5% (README "94.35%") | 548/600 = 91.3% (FS harness, seeds 0/1/2: 181, 186, 181) | 200 / 600 | 3.7 pt | REPRODUCED within tolerance (gap 3.2 pt; closest to its margin) |
| openpi pi0.5 `libero_object` | 98.2% | 496/500 = 99.2% (FS harness) | 500 / 500 | 1.7 pt | REPRODUCED within tolerance |
| openpi pi0.5 `libero_goal` | 98.0% | 486/500 = 97.2% (FS harness) | 500 / 500 | 1.8 pt | REPRODUCED within tolerance |
| openpi pi0.5 `libero_10` | 92.4% | 462/500 = 92.4% (FS harness) | 500 / 500 | 3.3 pt | REPRODUCED (exact) |
| Long-WAM IDM (`Efficient-Large-Model/Long-WAM-LIBERO-IDM`), spatial / goal | 99.5 / 98.0 (paper Table 1) | 497/500 = 99.4%, 490/500 = 98.0% (FS harness; Long-WAM's own harness 495, 491) | 500 / 500 | 0.9 / 1.8 pt | REPRODUCED within tolerance |
| Long-WAM COD (`Efficient-Large-Model/Long-WAM-LIBERO-COD`), spatial / object / goal / libero_10 | 98.6 / 99.8 / 96.8 / 95.8 | 492, 500, 487, 490 of 500 (FS harness; own harness 491, 500, 489, 489) | 500 / 500 | 1.5 / 0.6 / 2.2 / 2.5 pt | REPRODUCED within tolerance on all 4 |
| Long-WAM IDM libero_10 / object | 99.5 / 100.0 | 491/500 = 98.2% (own harness 98.6%) / 498/500 = 99.6% | 500 / 500 | 0.9 pt / undefined at p = 1 | libero_10 NOT reproduced (both harnesses below the margin); object has no binomial margin -- neither is a registry row |
| Alpamayo 2 Super (driving), PhysicalAI-AV held-out chunk 99, open-loop minADE | 0.8288 m (alpamayo-recipes `sft_eval_trajectory`, 100 clips) | 0.7822 m, 100/100 clips (FS driving harness + FS metric); the recipe's own evaluator on the same hardware: 0.7822 m | 100 / 100 | none set (a distance, not a rate) | FS == upstream evaluator to 4 dp; both 0.047 m better than the card (lower is better); not a registry row yet |
| Alpamayo 2 Super held-out `eval_loss` (chunk 99), through `AlpamayoSftSpec` | 1.699801 (alpamayo2_sft README) | 1.704507 in an env built from the recipe's `uv.lock` (alpamayo2 9596749); 1.774952 from upstream HEADs | full chunk 99 | none set (a loss) | REPRODUCED within 0.3% once pinned: build backend envs from the recipe lockfile, not from HEAD |
| verl-vla FPO on pi0.5, `libero_spatial` task 2 (from SFT step-100) | 29/50 = 58% at start → ~98% by update 25 | three seeds, val at updates 0 → 25: seed 42 0.70 → 0.94, seed 1 0.64 → 0.96, seed 2 0.72 → 0.98 (mean 0.69 → 0.96) | 50 / 50 per seed | two-sample binomial margin 5.5 pt at n=50 vs 50 | REPRODUCED within tolerance on three seeds: every endpoint ≤ 4 pt from ~98%; the start (58% published) sits inside n=50 noise (±13 pt) for all three |

Tolerance rule (binding): `foundationscale.upstream.vla_models.binomial_tolerance(rate, n_published, n_measured)` = `1.96 * sqrt(p(1-p)/n_pub + p(1-p)/n_meas)` percentage points, fixed from the published rate and both denominators — never from our measured gap. A row passes when the measured rate sits inside that margin of the published rate. This is the registry's two-sample rule; the harness's earlier one-sample `within_published` check disagreed with it on the GR00T row and was replaced.

Harness calibration (owned contract #4): the FS harness scores what each upstream's own harness scores —
openpi 493/500 = 98.6% (per task 50,50,50,49,46,49,50,50,50,49) vs 492/500 = 98.4% (0.2 pt);
GR00T 190/200 = 95.0% vs 192/201 = 95.5% first pass / 588/606 = 97.0% (0.5 pt); 0 episode errors in either.
The registry rows themselves were scored by the upstream harnesses; the FS harness reproduces them.

T1 evidence underneath: FS data path vs GR00T's loader over 8 replanning steps on trajectories 0 and 1 —
max |input diff| = 0.0 on every state key, both cameras, identical task text; same-seed actions identical. FS norm
stats vs GR00T `calculate_dataset_statistics`: max diff 0.0 on all 4 LIBERO suites. FS sample builder end-to-end on
real AV1: 2 cameras 256×256, state [1,8], action [16,7], edge pad 1/16 valid, `q01_q99` unapply error 7.45e-9 unclipped
(22% clipped, mostly the binary gripper), 238 samples / 20 episodes in 1.9 s = 126 samples/s single process.

Runs whose GR00T backbone processor was substituted (public Qwen3-VL stand-in) are labelled SMOKE and are not
reproduction evidence; the reproduced GR00T row used the real `nvidia/Cosmos-Reason2-2B` backbone.

## 2. Design

- **Own the contracts, wrap the rest.** FS owns four layers: the robot data schema + converters (built: LeRobot v2 +
  GR00T `modality.json`; planned: other upstream layouts and our own robot / egocentric data as first-class sources),
  the checkpoint sidecar (planned: upstream-native weights untouched plus an FS `RunManifest` recording upstream pin,
  modality/chunk contract, norm-stats digest, embodiment, licences), the run spec (built for GR00T fine-tuning:
  `Gr00tFinetuneSpec`), and the evaluation harness (built: LIBERO, both upstream protocols).
- **Upstreams run unmodified in their own environments** (`fs-vla-gr00t`, `fs-vla-openpi`, `fs-vla-verl`); adapters call
  public entry points only and live in `foundationscale/vla/adapters/<backend>/`. Every unavoidable workaround is an
  entry in the **upstream LEDGER** (what, why, upstream version, expiry) — see §5.
- **Registry.** `foundationscale/upstream/vla_models.py` holds `VLA_MODELS` (GR00T N1.7 spatial, openpi pi0.5) with the
  speech registry's shared `ModelEntry`/`ReferenceResult` types, `registry_problems` checks and `level3_plan`/`judge`.
  Every `SUPPORTED` VLA row joins the same Level 3 regression as the speech backends; the table stays separate so the
  speech transcription is pinned exactly as its owners wrote it.
- **Integrated == reproduced.** Rungs: (a) bit-exact input parity + seeded-action parity — done for GR00T; (b) the
  published checkpoint evaluated through a harness — done for both (table §1); (c) an FS-run fine-tune or RL run that
  reproduces the published number — done for GR00T (spatial) and openpi (all four suites) fine-tuning (§4). The passing run becomes the upgrade regression.

## 3. Modules

`foundationscale.vla` data contract (refuses rather than guesses):

- `lerobot` — LeRobot v2 reader (`open_lerobot`, `EpisodeMeta`, `read_episode_columns`); rejects non-contiguous episodes and unknown codebase versions.
- `modality` — GR00T `modality.json` as `ModalitySpec`/`SliceSpec`; `check_against_dataset` refuses any dimension the config does not cover.
- `norm` — per-dimension stats with GR00T's semantics: mean, population std, min, max, q01, q99 (`compute_norm_stats`, `stats_digest`, exact JSON round-trip), degenerate dims reported.
- `chunking` — `ChunkSpec` (GR00T `delta_indices`) + `valid_anchor_indices`; a declared pad mode (`refuse` or `edge` with an `action_valid` mask).
- `frames` — `decode_frames`: RGB frames at exact indices via PyAV (optional; torchcodec has no aarch64 wheel), refusing a video whose frame count is not its episode length.
- `sample` — `build_sample` → `VlaSample` (camera frames, state, action chunk in modality-slice order, `action_valid`); `Normalizer` (`mean_std`, `q01_q99`, `min_max`) with exact inverses.

Evaluation harness + CLI:

- `foundationscale.vla.eval` — FS LIBERO harness + `python -m foundationscale.vla.eval` CLI: one named `Protocol` row per backend (init mode, settle steps, replan cadence, trials/task, seed), policy-server lifecycle via `--serve-cmd`, report JSON with `success_rate_pct` and Wilson CI for `level3_plan`; exit 0 report written, 95 server never came up, 96 refused declaration.

Driving (`foundationscale.vla.driving`):

- `metrics` — `ade`, `min_ade` with Alpamayo's semantics: XY L2 per waypoint, mean over time, the best sample chosen on the full horizon and reused for every `by_t` horizon; refuses non-finite or mis-shaped input.
- `eval` — `python -m foundationscale.vla.driving.eval --backend alpamayo`: every clip stays in the denominator; exit 0 measured, 95 unmeasured (no server or any clip failed), 96 refused.
- `adapters/alpamayo` — trajectory server run in Alpamayo's own environment (alpamayo-recipes' evaluator setup, public functions only) and its client.

Adapters (`foundationscale/vla/adapters/`):

- `gr00t/contract` — reads a GR00T checkpoint's own `processor_config.json` into its embodiment contract, maps it onto `ChunkSpec`, and builds GR00T's observation dict from the FS data path (bit-exact against GR00T's loader).
- `gr00t/libero` — `Gr00tLiberoPolicy`: client to GR00T's own `run_gr00t_server.py` (ZMQ), `n_action_steps`.
- `gr00t/finetune` — `Gr00tFinetuneSpec` → GR00T's own `launch_finetune.py` argv; `stage_base_checkpoint` symlinks weights and patches only `use_flash_attention`; `PUBLISHED_LIBERO_RECIPE` = NVIDIA's LIBERO recipe.
- `gr00t/ddp_hook` — routes `num_gpus > 1` to DDP (DeepSpeed has no aarch64 build), with no upstream edit.
- `openpi/libero` — `OpenPiLiberoPolicy`: openpi websocket client (the FS policy protocol) with `replan_steps`.

## 4. How to reproduce

Published evaluation protocol rows (what a Level 3 rerun rolls out exactly):

| backend | init | settle steps | replan / action steps | trials per task | seed |
|---|---|---|---|---|---|
| openpi | suite init states | 10 | 5 | 50 | 7 |
| gr00t | seeded reset | 0 | 8 | 20 | 0 |

Step budget per episode: openpi caps each suite at its longest demo plus margin (220 / 280 / 300 / 520 for
spatial / object / goal / 10); GR00T runs `--max-episode-steps 720` env steps on every suite.

Rung (b) through the FS harness — the policy server is the upstream's own; `--serve-cmd` is a
deployment-specific command line with `{model}` and `{port}` placeholders and is kept out of the repo
(omit it if the server is already listening at `--host/--port`):

```bash
python -m foundationscale.vla.eval --backend gr00t \
  --model nvidia/GR00T-N1.7-LIBERO/libero_spatial \
  --suite libero_spatial --out reports/gr00t_n17_spatial.json \
  --serve-cmd '<deployment: start GR00T run_gr00t_server.py for {model} on port {port}>'

python -m foundationscale.vla.eval --backend openpi \
  --model gs://openpi-assets/checkpoints/pi05_libero \
  --suite libero_spatial --out reports/openpi_pi05_spatial.json \
  --serve-cmd '<deployment: start openpi serve_policy.py for {model} on port {port}>'
```

`--trials-per-task` / `--seed` override the protocol row (published baseline = defaults).
The three GR00T tightening passes used the same protocol at the published denominator (20 episodes per task,
5 envs, 8 action steps, 220 max episode steps), which is where 588/606 comes from.

Level 3 regression on any pin bump:
`from foundationscale.upstream.levels import level3_plan; level3_plan(VLA_MODELS)` renders the plan whose `judge`
compares each report's `success_rate_pct` against `ReferenceResult.upstream_value ± tolerance`; the openpi run above and
the GR00T 606-trial run are the current baselines (baseline pins: openpi `215abfb` + `pi05_libero`;
GR00T `d2b7e75` + `GR00T-N1.7-LIBERO/libero_spatial` + `Cosmos-Reason2-2B` snapshot `9ce19a19`, SDPA).

Rung (c), GR00T fine-tune through the FS run spec — REPRODUCED 2026-10-11:

```python
from foundationscale.vla.adapters.gr00t.finetune import (
    PUBLISHED_LIBERO_RECIPE, on_gpus, render_command, stage_base_checkpoint)
spec = on_gpus(PUBLISHED_LIBERO_RECIPE, 4)   # same global batch 640 -> 160/GPU, 20K steps
argv  = render_command(spec, torchrun="torchrun", master_port=29500)  # DDP hook for num_gpus>1
```

Measured: 20,000 steps × global batch 640 from the N1.7 base on one 4-GPU GB200 tray in 8 h 58 min (1.61 s/step,
SDPA, real Cosmos backbone), then `python -m foundationscale.vla.eval --backend gr00t --suite libero_spatial` with the
GR00T protocol: step 12,000 191/200 = 95.5%; final step 20,000 **481/500 = 96.2%** (Wilson 94.1–97.6) vs published
195/200 = 97.5% — gap 1.3 ≤ two-sample margin 2.6 pt. Same harness on NVIDIA's released checkpoint: 190/200.
A second evaluation of the final checkpoint under the corrected GR00T protocol (720 env steps, seed 1): 197/200 = 98.5%.

GR00T fine-tune on `libero_10` through the same spec (2026-10-13, one 4-GPU tray, 20,000 steps in 7 h 47 min): final
checkpoint 544/600 = 90.7% (seeds 0/1/2: 182, 183, 179; checkpoints 16k and 18k: 181/200 each). Against the card
(189/200 = 94.5%, margin 3.65 pt at 200/600) the gap is 3.8 pt: NOT reproduced against the card on this one training run.
Against NVIDIA's released libero_10 checkpoint in the same FS harness (548/600 = 91.3%) the gap is 0.7 pt, so FS training
matches NVIDIA's model and both sit below the card on long-horizon tasks under this harness. Open check: GR00T's own rollout
client on libero_10, to separate a harness effect from a protocol one.

GR00T fine-tunes on `libero_goal` and `libero_object` through the same spec (2026-10-14, 20,000 steps each; goal trained on
427 of 428 episodes because episode 82's wrist video is corrupt in the upstream dataset, a fresh download is byte-identical):

| suite | FS-trained checkpoint (FS harness, seeds 0/1/2) | card | margin (200/600) | verdict |
|---|---|---|---|---|
| goal | 197 + 192 + 193 = 582/600 = 97.0% | 195/200 = 97.5% | 2.5 pt | REPRODUCED |
| object | 197 + 199 + 199 = 595/600 = 99.2% | 197/200 = 98.5% | 2.0 pt | REPRODUCED |

GR00T rung (c) therefore holds on spatial, goal and object; libero_10 matches NVIDIA's released checkpoint but not the card.

Why the cards can sit above a strict harness: GR00T's own client (`gr00t/eval/rollout_policy.py`, README command with
`--n-envs 5`) counts episodes as they finish across the vectorised envs and stops at `n_episodes`, dropping the episodes
still running. Failures here are 720-step timeouts and successes take ~150 steps, so the dropped episodes are mostly
failures. On NVIDIA's released `libero_goal` checkpoint the same client gives 201/202 = 99.5% with 5 envs (one task
reports 22/20) and 187/200 = 93.5% with `--n-envs 1`, where nothing is dropped. On `libero_10` (n = 200 per run, ±4.5 pt)
the two settings give 178/200 and 186/200 for the released checkpoint and 186/200 and 180/200 for the FS-trained one:
the effect is inside the noise there. The FS harness keeps every planned episode in the denominator by construction.

Rung (c), openpi fine-tune through the FS run spec — REPRODUCED 2026-10-13: `OpenPiFinetuneSpec(PUBLISHED_LIBERO_RECIPE)`
(`pi05_libero`: batch 256, 30,000 steps from `pi05_base`) rendered into openpi's own `scripts/train.py` and run on one
4-GPU GB200 tray (~2.0-2.7 s/step, ~19 h; norm stats from openpi's `compute_norm_stats.py`). The final checkpoint,
evaluated with the FS harness under the openpi protocol (several runs per suite; openpi's fixed init states make
seeds less than independent, so the pooled count is the evidence, not a CI over seeds):

| suite | FS-trained checkpoint | published | tolerance (pooled n) | verdict |
|---|---|---|---|---|
| spatial | 495 + 491 = 986/1000 = 98.6% | 98.8% | 1.2 pt | REPRODUCED |
| object | 489 + 488 + 494 = 1471/1500 = 98.1% | 98.2% | 1.4 pt | REPRODUCED |
| goal | 481 + 486 + 488 = 1455/1500 = 97.0% | 98.0% | 1.5 pt | REPRODUCED (gap 1.0) |
| libero_10 | 463 + 476 + 473 = 1412/1500 = 94.1% | 92.4% | 2.7 pt | REPRODUCED (above the card) |

At 60% of the schedule (step 18,000) the same run already scored 494/500 = 98.8% on spatial.

## 5. Environment notes (aarch64 / Blackwell data-centre)

- LIBERO stack: `mujoco==3.3.1` (the GR00T/verl-vla pin; 3.15 breaks robosuite 1.4 with an `mjJNT_HINGE/SLIDE`
  assertion) + `robosuite==1.4.1` + `numpy<2` + `bddl==1.0.1` + `gym==0.25.2`; LIBERO itself importable via `PYTHONPATH`.
- Renderer: headless EGL (`MUJOCO_GL=egl`, NVIDIA `libEGL_nvidia`) — 1.70 ms step+render 256×256, 66.4 ms per env
  step with 2 cameras; no OSMesa. Harmless: `EGLError` from `mujoco.egl.GLContext.__del__` at interpreter exit.
- torch ≥ 2.6 `weights_only`: register safe globals (`numpy.core.multiarray._reconstruct`, `numpy.ndarray`,
  `numpy.dtype`, `Float64DType`) for LIBERO init states — never `weights_only=False`.
- `LIBERO_CONFIG_PATH` pointed at a private config: avoids the interactive first-run prompt and any shared home directory.
- flash-attn has no aarch64 build: `use_flash_attention=False` (SDPA) is the only runtime change in the GR00T path.
- torchcodec has no aarch64 wheel: GR00T's one decode call (`VideoDecoder.get_frames_at`, NHWC) goes through a small
  PyAV-backed shim (deterministic dav1d decode, the decoder FFmpeg uses); FS `frames` decodes AV1 via PyAV directly.
- DeepSpeed has no aarch64 build (and no prebuilt wheel): GR00T `num_gpus > 1` is routed to DDP through the
  `ddp_hook` wrapper before `experiment.run()` (upstream LEDGER entry; `launch_finetune.py` exposes no
  `training.use_ddp`).
- verl-vla rollouts are HF-based — **no vLLM needed**; Ray lives only inside the `fs-vla-verl` backend environment.
- `libero 0.1.1` (pip) fails at the `egl_probe` build on aarch64 → install it with `--no-deps` and supply its runtime
  set explicitly (`robosuite`, `mujoco`, `bddl`, `easydict`, `gym`, `termcolor`). The verl-vla environment also pins
  `numpy==1.26.4` and `ml_dtypes==0.5.1` and needs `onnx_ir`.

## 6. Simulators vs hardware

- LIBERO / robosuite (MuJoCo, EGL) runs on GB200- and H200-class accelerators — measured on our GB200 trays:
  66.4 ms per env step, headless. RoboCasa is the same stack; SimplerEnv (SAPIEN/Vulkan) is unprobed.
- Isaac Sim / Isaac Lab / Lab Arena need RTX ray tracing; GB200, H100 and H200 all lack RT cores. So Phases 1–2
  reproduce on LIBERO on our GB200 trays; Isaac-dependent published results (e.g. verl-vla DSRL GR00T on Lab Arena)
  wait for an RTX machine.

## 7. Known gaps / next

- **Long-WAM fine-tune (rung 3), pipeline only**: `LongWamFinetuneSpec` renders Long-WAM's own `scripts/train.py` at the
  published global batch of 128 (8 GPUs x 8 x 2 in `docs/TRAINING.md`; 2 GPUs x 4 x 16 here). A 60-step run trained
  (loss 1.06 -> 0.62), wrote a checkpoint, and that checkpoint evaluated through the FS harness (0/50 on spatial, as
  expected after 60 steps, 0 episode errors). Not a reproduction: the LIBERO ActionDiT init asset is not public, so the
  run sets `model.action_dit_pretrained_path=null`; a full 10-epoch run is the next step.
- **RL at the published placement**: `across_nodes` spreads a recipe's 8 model GPUs over 2 nodes x 4; TD3+BC and RECAP
  reruns at that placement are pending GPU availability (both collapse at 4 GPUs).

- **GR00T rung (c) done, one seed**: the training run is not yet a registry row (its checkpoint is ours, not an upstream
  artifact), so the Level 3 regression still replays the published checkpoint. Scaling measured on the way: 1 GPU
  109 samples/s; 3 GPUs DDP 283 samples/s (87% efficiency); 4 GPUs ~400 samples/s (1.61 s per 640-sample step).
- **GR00T rung (c)**: spatial, goal, object reproduced; libero_10 equals the released checkpoint, 3.8 pt under the card.
  A second libero_10 training run would separate training variance from the card gap.
- **More suites**: only `libero_spatial` is reproduced; `libero_goal` (97.5), `libero_object` (98.45), `libero_long`
  (94.35), and openpi's goal/object/long (98.2/98.0/92.4) are the remaining published targets at the same row rules.
- **RL entry in the registry**: verl-vla FPO evidence is reproduced within tolerance on one seed (2 GPU policy workers
  instead of 8, env sampling kept at 4 workers × 8 envs) but `VLA_MODELS` has no RL row yet — it needs the multi-seed
  denominator decision and the verl-vla metrics reader. DSRL and TD3+BC follow in the same wrapper.
- GR00T open-loop numbers (MSE 3.41e-4 on training-set trajectory 0 vs zero-action 0.160) are fit, not generalisation;
  they stay T1 diagnostics, never evidence rows.